# Owner(s): ["oncall: distributed"]

import functools
import os
import sys
import unittest
import warnings
from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FlatParameter,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp._common_utils import clean_tensor_name
from torch.distributed.fsdp._flat_param import _FSDP_USE_UNSAFE_SETATTR
from torch.distributed.fsdp._runtime_utils import HOMOGENEOUS_ATTR_NAMES
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    ModuleWrapPolicy,
    transformer_auto_wrap_policy,
)
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _assert_module_states,
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    FSDPTestMultiThread,
    MLP,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class MyModel(nn.Module):
    # [CLASS DOC] HELPER: Two 2x2 linear layers a and b; forward adds the two inputs and
    # applies a then b.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear layers a and b (2 to 2).
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Returns b(a(x + y)).
        return self.b(self.a(x + y))


class TestFSDPMiscMultiProcess(FSDPTestContinuous):
    # [CLASS DOC] Two-rank multi-process test case with assorted FSDP behavior checks that
    # need real collectives; see the individual tests for the scenarios. Most need 2 GPUs.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property giving the number of ranks; always 2.
        return 2

    @property
    def process_group(self):
        # [FUNCTION DOC] HELPER: Property returning the default (world) process group.
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    @parametrize("use_index", [True, False])
    def test_fsdp_device_id(self, use_index):
        # [FUNCTION DOC] Test (2 GPUs): wraps NestedWrappedModule with device_id given as an
        # integer index or a device object. Checks the flat parameters land on that device
        # for a CPU module, that a module already on it is accepted, and that a device
        # without an explicit index on a GPU module emits a UserWarning.
        """
        Tests the FSDP ``device_id`` argument:
          - Wrapping a CPU module should move the module to the GPU matching
          ``device_id``
          - Wrapping a GPU module already on the GPU matching ``device_id``
          should not raise an error
          - Wrapping a GPU module already on GPU and passing a GPU device
          without specifying a device ID (i.e. ``torch.device("cuda")``) warns
        """
        dev_id = (
            torch.accelerator.current_device_index()
            if use_index
            else torch.device(device_type, torch.accelerator.current_device_index())
        )

        def _check_device_matches(module, device_id):
            # [FUNCTION DOC] HELPER: Collects the devices of all flat parameters in module,
            # asserts there is at least one and exactly one distinct device, and asserts it
            # equals device_id (an index is converted to a device of the current type).
            """Checks that the ``FlatParameter``s in ``module`` have device
            matching ``device_id``."""
            devices = {
                p.device for p in module.parameters() if isinstance(p, FlatParameter)
            }
            if not (len(devices) > 0):
                raise AssertionError(
                    f"Expected at least one device, but got {len(devices)}"
                )
            self.assertEqual(1, len(devices))
            found_device = devices.pop()
            if use_index and not isinstance(device_id, torch.device):
                device = torch.device(device_type, device_id)
            else:
                device = device_id
            self.assertEqual(found_device, device)

        # Check that FSDP parameters are moved to `device_id` for a CPU module
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_NEVER,
            fsdp_kwargs={"device_id": dev_id},
        )
        _check_device_matches(nested_wrapped_module, dev_id)
        # Check that specifying `device_id` for a GPU module already on that
        # device does not raise an error
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs={"device_id": dev_id},
        )
        _check_device_matches(nested_wrapped_module, dev_id)
        # Check that passing in `torch.device("cuda")` for a GPU module warns
        regex = "does not have an explicit index"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                fsdp_kwargs={"device_id": torch.device(device_type)},
            )
        _check_device_matches(
            nested_wrapped_module,
            torch.device(device_type, torch.accelerator.current_device_index()),
        )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_zero2_eval_with_prefetch(self):
        # [FUNCTION DOC] Test (2 GPUs): trains an FSDP model (SHARD_GRAD_OP, forward prefetch,
        # original params, linear and conv layers wrapped) and an identical DDP model for 5
        # steps on the same random inputs and seeds, requiring close losses and input grads;
        # then evaluates both under no_grad for 5 batches requiring close losses; then
        # trains 5 more steps again requiring close losses and grads.
        # KNOWN ISSUE: the input tensor is shared by both models, so its grad accumulates in
        # place and both stored grads are the same tensor; the grads comparison is trivially true.
        # Test FSDP validation with SHARD_GRAD_OP and forward_prefetch

        class Mnist(nn.Module):
            # [CLASS DOC] HELPER: Small MNIST-style CNN (two convs, dropouts, layer norm, two
            # linear layers) whose forward takes images and labels and returns the loss.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the conv, dropout, linear and layer norm layers.
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)
                self.ln = nn.LayerNorm(9216)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs convs with relu, pooling, dropout, flatten, layer
                # norm and the linear layers on images x, then returns the cross entropy of the
                # log-softmax output against labels y.
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.ln(x)
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = torch.nn.functional.log_softmax(x, dim=1)
                loss = torch.nn.functional.cross_entropy(output, y)
                return loss

        model = Mnist().to(device=device_type)
        model1 = Mnist().to(device=device_type)
        model1.load_state_dict(model.state_dict())
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            forward_prefetch=True,
            use_orig_params=True,
            auto_wrap_policy=ModuleWrapPolicy([nn.Linear, nn.Conv2d]),
        )
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model1,
        )

        fsdp_opt = torch.optim.SGD(fsdp_model.parameters(), lr=1e-4)
        ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=1e-4)

        seed = self.rank + 20231010
        torch.manual_seed(seed)
        torch.get_device_module(device_type).manual_seed(seed)

        losses = []
        grads = []
        for i in range(5):
            x = torch.randn(8, 1, 28, 28, device=device_type).requires_grad_()
            y = torch.randint(low=0, high=9, size=(8,), device=device_type)
            for model, opt in ((fsdp_model, fsdp_opt), (ddp_model, ddp_opt)):
                seed = self.rank + i
                torch.manual_seed(seed)
                torch.get_device_module(device_type).manual_seed(seed)
                loss = model(x, y).sum()
                losses.append(loss)
                loss.backward()
                opt.step()
                grads.append(x.grad)
                opt.zero_grad()
            if not torch.allclose(losses[0], losses[1]):
                raise AssertionError(
                    f"Expected losses to be close: {losses[0]} vs {losses[1]}"
                )
            if not torch.allclose(grads[0], grads[1]):
                raise AssertionError(
                    f"Expected grads to be close: {grads[0]} vs {grads[1]}"
                )
            losses.clear()
            grads.clear()

        with torch.no_grad():
            fsdp_model.eval()
            ddp_model.eval()
            for _ in range(5):
                x = torch.randn(8, 1, 28, 28, device=device_type).requires_grad_()
                y = torch.randint(low=0, high=9, size=(8,), device=device_type)
                fsdp_loss = fsdp_model(x, y)
                ddp_loss = ddp_model(x, y)
                if not torch.allclose(fsdp_loss, ddp_loss):
                    raise AssertionError(
                        f"Expected fsdp_loss and ddp_loss to be close: {fsdp_loss} vs {ddp_loss}"
                    )

        fsdp_model.train()
        ddp_model.train()
        for i in range(5):
            x = torch.randn(8, 1, 28, 28, device=device_type).requires_grad_()
            y = torch.randint(low=0, high=9, size=(8,), device=device_type)
            for model, opt in ((fsdp_model, fsdp_opt), (ddp_model, ddp_opt)):
                seed = self.rank + i
                torch.manual_seed(seed)
                torch.get_device_module(device_type).manual_seed(seed)
                loss = model(x, y).sum()
                losses.append(loss)
                loss.backward()
                opt.step()
                grads.append(x.grad)
                opt.zero_grad()
            if not torch.allclose(losses[0], losses[1]):
                raise AssertionError(
                    f"Expected losses to be close: {losses[0]} vs {losses[1]}"
                )
            if not torch.allclose(grads[0], grads[1]):
                raise AssertionError(
                    f"Expected grads to be close: {grads[0]} vs {grads[1]}"
                )
            losses.clear()
            grads.clear()

    @skip_if_lt_x_gpu(2)
    @parametrize("use_second_layer", [True, False])
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD, None])
    def test_fsdp_module_no_compute_grad(self, use_second_layer, sharding_strategy):
        # [FUNCTION DOC] Test (2 GPUs): wraps a two-layer model with every layer in its own FSDP
        # unit. Runs 4 forward/backward passes where the loss only uses the first layer's
        # output, and asserts the first unit's flat parameter gets a gradient while the second
        # (used in forward or not) gets none. Parametrized by use_second_layer and by
        # NO_SHARD vs. the default strategy.
        # When use_second_layer=True, b is involved in forward computation but does
        # not receive grad in backward. Otherwise, b is not involved in forward
        # computation.

        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Two 10x10 linear layers a and b; b only runs when
            # use_second_layer (from the enclosing test) is true.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layers a and b.
                super().__init__()
                self.a = nn.Linear(10, 10)
                self.b = nn.Linear(10, 10)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a(x); if use_second_layer, returns the pair
                # (a(x), b(y)) instead.
                out1 = self.a(x)
                if use_second_layer:
                    out2 = self.b(y)
                    return out1, out2
                else:
                    return out1

        fsdp = FSDP(
            MyModel().to(device=device_type),
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=always_wrap_policy,
        )
        x = torch.randn(10, 10, device=device_type)
        y = torch.randn(10, 10, device=device_type)
        for _ in range(4):
            if use_second_layer:
                a, _ = fsdp(x, y)
            else:
                a = fsdp(x, y)
            loss = a.sum()
            loss.backward()

            # self.a receives grad, self.b does not
            a_grad = fsdp.module.a._handle.flat_param.grad
            b_grad = fsdp.module.b._handle.flat_param.grad
            self.assertIsNotNone(a_grad)
            self.assertIsNone(b_grad)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_not_all_outputs_used_in_loss(self):
        # [FUNCTION DOC] Test driver (2 GPUs): runs the unused-output scenario for FULL_SHARD,
        # SHARD_GRAD_OP and NO_SHARD.
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_fsdp_not_all_outputs_used_in_loss,
        )

    def _test_fsdp_not_all_outputs_used_in_loss(
        self, sharding_strategy: ShardingStrategy
    ):
        # [FUNCTION DOC] HELPER: Trains an FSDP copy (lin1 and root wrapped) and a local copy
        # of a two-output module for 6 SGD steps; the first 2 losses use both outputs, later
        # ones only the first output. After each step checks the root unit is resharded,
        # parameters equal the local copy's, and parameters actually changed. Ends with a barrier.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Two 4x4 linear layers applied to the same input, returning
            # both outputs as a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layers lin1 and lin2.
                super().__init__()
                self.lin1 = nn.Linear(4, 4)
                self.lin2 = nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the pair (lin1(x), lin2(x)).
                a = self.lin1(x)
                b = self.lin2(x)
                return (a, b)

        def _check_resharded(fsdp_module):
            # [FUNCTION DOC] HELPER: If the module has a flat parameter handle, asserts that
            # for sharded strategies the unsharded padded storage is freed (size zero), and
            # that the flat parameter again points at the local shard memory.
            handle = fsdp_module._handle
            if not handle:
                return
            param = handle.flat_param
            if handle.uses_sharded_strategy:
                full_param = param._full_param_padded
                self.assertEqual(full_param.storage().size(), 0)

            self.assertEqual(param.data_ptr(), param._local_shard.data_ptr())

        def _check_equal(local, fsdp):
            # [FUNCTION DOC] HELPER: Summons full parameters of the FSDP module and asserts
            # each is close to the matching parameter of the local module.
            with FSDP.summon_full_params(fsdp):
                for p1, p2 in zip(fsdp.parameters(), local.parameters()):
                    torch.testing.assert_close(p1, p2)

        fsdp_ctor = functools.partial(FSDP, sharding_strategy=sharding_strategy)
        m = MyModule().to(device=device_type)
        m_local = deepcopy(m)
        local_m = m_local
        prev_params = [p.clone() for p in m_local.parameters()]

        m.lin1 = fsdp_ctor(m.lin1)
        m = fsdp_ctor(m)
        _check_equal(m_local, m)

        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        opt_local = torch.optim.SGD(local_m.parameters(), lr=1e-3)

        for i in range(6):
            t = torch.ones(4, device=device_type)
            a, b = m(t)
            local_a, local_b = local_m(t)
            if i < 2:
                # use both params in loss computation. Later,
                # b will go unused and we check grads are the
                # same as local training.
                loss = (a @ b).sum()
                loss_local = (local_a @ local_b).sum()
            else:
                loss = a.sum()
                loss_local = local_a.sum()

            loss.backward()
            loss_local.backward()
            _check_resharded(m)
            opt.step()
            opt_local.step()
            _check_equal(m_local, m)
            # Ensure at least some change from previous params, otherwise
            # above check would be vacuously true.
            self.assertTrue(
                any(
                    not torch.equal(p1, p2)
                    for p1, p2 in zip(prev_params, m_local.parameters())
                )
            )
            prev_params = [p.clone() for p in local_m.parameters()]
            opt.zero_grad()
            opt_local.zero_grad()

        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_optim_overlap_no_use_orig_params_error(self):
        # [FUNCTION DOC] Test (2 GPUs): registers an in-backward SGD optimizer on the
        # parameters of an FSDP model built with use_orig_params false; running forward must
        # raise RuntimeError saying overlap is only supported with use_orig_params true.
        fsdp_overlap = FSDP(
            MyModel().to(device=device_type),
            auto_wrap_policy=always_wrap_policy,
            use_orig_params=False,
        )
        optim_cls = torch.optim.SGD
        optim_kwargs = {"lr": 0.03}
        _apply_optimizer_in_backward(
            optimizer_class=optim_cls,
            params=fsdp_overlap.parameters(),
            optimizer_kwargs=optim_kwargs,
            register_hook=False,
        )

        inp = torch.randn(10, 10, device=device_type)
        with self.assertRaisesRegex(
            RuntimeError, "only supported with use_orig_params=True"
        ):
            fsdp_overlap(inp, inp)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_optimizer_overlap(self):
        # [FUNCTION DOC] Test (2 GPUs): for CPU offload on and off, compares a regular FSDP
        # model stepped by an ordinary SGD optimizer with an identical copy whose parameters
        # use an in-backward SGD optimizer. Over 6 steps asserts the overlapped model's
        # handles have no sharded grad, its parameters change each step, parameters equal the
        # regular model's, and gradients are None on both after the step.
        torch.manual_seed(0)
        for cpu_offload in [True, False]:
            offload = CPUOffload(offload_params=cpu_offload)
            model = MyModel().to(device=device_type)
            model_overlap = deepcopy(model)
            fsdp = FSDP(
                model.to(device=device_type),
                auto_wrap_policy=always_wrap_policy,
                use_orig_params=True,
                cpu_offload=offload,
            )
            fsdp_overlap = FSDP(
                model_overlap.to(device=device_type),
                auto_wrap_policy=always_wrap_policy,
                use_orig_params=True,
                cpu_offload=offload,
            )
            optim_cls = torch.optim.SGD
            optim_kwargs = {"lr": 0.03}
            _apply_optimizer_in_backward(
                optimizer_class=optim_cls,
                params=fsdp_overlap.parameters(),
                optimizer_kwargs=optim_kwargs,
                register_hook=False,
            )
            for p in fsdp_overlap.parameters():
                if not hasattr(p, "_in_backward_optimizers"):
                    raise AssertionError(
                        "Expected parameter to have '_in_backward_optimizers' attribute"
                    )
            optim = optim_cls(fsdp.parameters(), **optim_kwargs)

            # Verify params initially equal
            for p1, p2 in zip(fsdp.parameters(), fsdp_overlap.parameters()):
                self.assertEqual(p1, p2)

            with FSDP.summon_full_params(fsdp_overlap):
                fsdp_overlap_prev_params = [
                    (n, p.clone()) for n, p in fsdp_overlap.named_parameters()
                ]

            for i in range(6):
                inp = torch.randn(2, 2, device=device_type)
                with torch.no_grad():
                    inp_clone = inp.clone()
                fsdp(inp, inp).sum().backward()
                fsdp_overlap(inp_clone, inp_clone).sum().backward()

                optim.step()
                optim.zero_grad()

                # Overlapped optimizer FSDP module should have sharded_grad as None.
                for fsdp_unit in FSDP.fsdp_modules(fsdp_overlap):
                    handle = fsdp_unit._handle
                    if handle:
                        handle_grad = handle.sharded_grad
                        self.assertEqual(
                            None,
                            handle_grad,
                            "Overlapped FSDP sharded_grad is not None!",
                        )

                # Note: FSDP without optimizer overlap won't set sharded_grad to None until the next
                # pre-forward since it needs to run FSDP specific logic that picks up that set_to_none=True
                # has been called (or that the gradients have been otherwise set to None)

                # Verify parameters are different than prev iteration
                with FSDP.summon_full_params(fsdp_overlap, with_grads=True):
                    for (n, p), (n_prev, p_prev) in zip(
                        fsdp_overlap.named_parameters(), fsdp_overlap_prev_params
                    ):
                        self.assertEqual(n, n_prev)
                        self.assertNotEqual(
                            p,
                            p_prev,
                            lambda msg: f"{msg}\n{n_prev} Params at iter {i} same as previous iter!",
                        )

                # Verify overlap and non overlapped are the same
                with FSDP.summon_full_params(fsdp_overlap):
                    with FSDP.summon_full_params(fsdp):
                        for (n_overlap, p_overlap), (n, p) in zip(
                            fsdp_overlap.named_parameters(), fsdp.named_parameters()
                        ):
                            self.assertEqual(n_overlap, n)
                            self.assertEqual(
                                p,
                                p_overlap,
                                lambda msg: f"{msg}\nRank {self.rank}: Params not equal at iteration {i}: {n_overlap} - {p} vs {p_overlap}",
                            )
                            self.assertEqual(
                                None,
                                p.grad,
                                lambda msg: f"{msg}\nExpected param {n} grad to be None",
                            )
                            self.assertEqual(
                                None,
                                p_overlap.grad,
                                lambda msg: f"{msg}\nExpected param {n_overlap} grad to be None",
                            )

                    fsdp_overlap_prev_params = [
                        (n, p.clone()) for n, p in fsdp_overlap.named_parameters()
                    ]

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_training(self):
        # [FUNCTION DOC] Test (2 GPUs): over a gloo process group, trains an FSDP CPU model
        # (NO_SHARD, FULL_SHARD, SHARD_GRAD_OP in turn) and a DDP reference with Adam for 10
        # steps on the same input, asserting equal losses at every step.
        """Tests FSDP training on CPU."""
        gloo_pg = dist.new_group(backend="gloo")
        for ss in [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
        ]:
            torch.manual_seed(42)
            model = MyModel()
            ref_model = DDP(deepcopy(model), process_group=gloo_pg)
            model = FSDP(
                model,
                sharding_strategy=ss,
                auto_wrap_policy=always_wrap_policy,
                process_group=gloo_pg,
                device_id=torch.device("cpu"),
            )
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
            optim = torch.optim.Adam(model.parameters(), lr=1e-2)
            torch.manual_seed(42 + self.rank)
            inp = torch.randn(2, 2)
            for _ in range(10):
                losses = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    loss = _model(inp, inp).sum()
                    losses.append(loss)
                    loss.backward()
                    _optim.step()
                    _optim.zero_grad()
                self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_summon_full_params_offload_to_cpu_on_cpu_model(self):
        # [FUNCTION DOC] Test driver (2 GPUs): creates a gloo group and runs the CPU offload
        # summon scenario over FULL_SHARD and SHARD_GRAD_OP, use_orig_params on/off and
        # writeback on/off.
        """
        Tests ``summon_full_params(offload_to_cpu=True)`` when the FSDP compute
        device is already CPU. There, the unsharded ``FlatParameter`` is already
        on CPU, so ``to_cpu()`` must skip freeing the unsharded storage that
        ``flat_param`` and its views alias; otherwise the summoned parameters
        read freed memory.
        """
        gloo_pg = dist.new_group(backend="gloo")
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                ],
                "use_orig_params": [False, True],
                "writeback": [False, True],
            },
            self._test_summon_full_params_offload_to_cpu_on_cpu_model,
            gloo_pg,
        )

    def _test_summon_full_params_offload_to_cpu_on_cpu_model(
        self,
        gloo_pg,
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
        writeback: bool,
    ):
        # [FUNCTION DOC] HELPER: Wraps a CPU copy of MyModel in FSDP over the gloo group and
        # summons full params with offload_to_cpu. Asserts the summoned parameters are on CPU
        # and equal the reference; with writeback adds 1 to every parameter inside. After exit
        # asserts parameters are on CPU and a second summon sees the reference values plus 1
        # only if writeback was enabled.
        torch.manual_seed(42)
        ref_model = MyModel()
        ref_params = dict(ref_model.named_parameters())
        model = FSDP(
            deepcopy(ref_model),
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=always_wrap_policy,
            process_group=gloo_pg,
            device_id=torch.device("cpu"),
            use_orig_params=use_orig_params,
        )
        cpu = torch.device("cpu")
        with FSDP.summon_full_params(model, offload_to_cpu=True, writeback=writeback):
            summoned = dict(model.named_parameters())
            self.assertEqual(len(summoned), len(ref_params))
            for name, param in summoned.items():
                self.assertEqual(param.device, cpu)
                self.assertEqual(param, ref_params[clean_tensor_name(name)])
            if writeback:
                with torch.no_grad():
                    for param in model.parameters():
                        param.add_(1.0)
        for param in model.parameters():
            self.assertEqual(param.device, cpu)
        # Re-summon to confirm the exit path resharded correctly and that the
        # in-place edit was (not) persisted according to `writeback`.
        delta = 1.0 if writeback else 0.0
        with FSDP.summon_full_params(model, offload_to_cpu=True):
            for name, param in model.named_parameters():
                self.assertEqual(param, ref_params[clean_tensor_name(name)] + delta)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_init_stays_on_cpu(self):
        # [FUNCTION DOC] Test (2 GPUs): wraps a CPU NestedWrappedModule in FSDP, expecting a
        # UserWarning that the module is on CPU, and asserts all parameters stay on CPU. Then
        # moves the model to the accelerator and runs forward and backward with a CPU input,
        # which FSDP must move to the device. Sets the current device index to the rank.
        # Move me to MT test once warning logging and backward collective issue
        # is resolved.
        """Tests that passing a CPU module to FSDP preserves that the wrapped
        module is on CPU after FSDP initialization, albeit after logging a
        warning, and that FSDP moves CPU input to GPU before the forward."""
        torch.accelerator.set_device_index(self.rank)
        regex = "passed-in `module` is on CPU"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_NEVER,
            )
            fsdp_model = FSDP(nested_wrapped_module, self.process_group)
        devices = {p.device for p in fsdp_model.parameters()}
        self.assertEqual(1, len(devices))
        self.assertEqual(torch.device("cpu"), devices.pop())
        fsdp_model = fsdp_model.to(device=device_type)
        # Ensure fwd + backward can be performed after moving to CUDA.
        # CPU input also tests that input is correctly moved to appropriate
        # CUDA device.
        inp = fsdp_model.module.get_input(device=torch.device("cpu"))
        fsdp_model(*inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_cpu_init_with_sync_module_states(self):
        # [FUNCTION DOC] Test (2 GPUs): wrapping a CPU module with sync_module_states must
        # raise ValueError about CPU parameters or buffers. With device_id given it must
        # succeed even with extra CPU buffers filled with the rank value, after which every
        # rank's buffers are zeros (rank 0's values) and live on the accelerator device.
        """
        Tests that passing ``sync_module_states=True`` raises an error for
        a CPU module since the synchronization requires GPU communication,
        while additionally passing ``device_id`` does not raise an error, even
        when the model has CPU buffers.
        """

        def init_nested_wrapped_module():
            # [FUNCTION DOC] HELPER: Returns a fresh unwrapped NestedWrappedModule on CPU.
            return NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_NEVER,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The module has CPU parameters or buffers when `sync_module_states=True`",
        ):
            FSDP(
                init_nested_wrapped_module(),
                self.process_group,
                sync_module_states=True,
            )

        # Check that `device_id` with `sync_module_states=True` works
        nested_wrapped_module = init_nested_wrapped_module()
        nested_wrapped_module.buf = nn.Buffer(
            torch.ones((2, 2), device="cpu") * self.rank
        )
        nested_wrapped_module.module[0].buf = nn.Buffer(
            torch.ones((3, 2), device="cpu") * self.rank
        )
        nested_wrapped_module = FSDP(
            nested_wrapped_module,
            self.process_group,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=torch.accelerator.current_device_index(),
            sync_module_states=True,
        )
        # Each rank's buffers should be 0s since rank 0 is the source, and they
        # should be on GPU since we specified `device_id`
        self.assertEqual(
            nested_wrapped_module.buf.device,
            torch.device(device_type, torch.accelerator.current_device_index()),
        )
        self.assertEqual(nested_wrapped_module.buf, torch.zeros((2, 2)))
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf.device,
            torch.device(device_type, torch.accelerator.current_device_index()),
        )
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf, torch.zeros((3, 2))
        )


class TestFSDPMiscMultiThread(FSDPTestMultiThread):
    # [CLASS DOC] Two-rank multi-threaded test case with assorted FSDP construction and
    # device placement checks (device_id, ignored modules, sync_module_states, homogeneous
    # attributes, unsupported containers). Threads share the process, so tests that set
    # environment variables or the current device affect each other.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property giving the number of ranks; always 2.
        return 2

    @property
    def process_group(self):
        # [FUNCTION DOC] HELPER: Property returning the default (world) process group.
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_namedtuple(self):
        # [FUNCTION DOC] Test (2 GPUs): passes a namedtuple of tensors that require grad
        # through an FSDP-wrapped identity module and asserts every returned tensor has
        # backward hooks registered, i.e. FSDP handled the namedtuple output.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an unused 100x100 linear whose forward returns its
            # input unchanged.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer lin.
                super().__init__()
                self.lin = nn.Linear(100, 100)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        m = MyModule().to(device=device_type)
        m = FSDP(m)
        t = torch.ones(1, device=device_type, requires_grad=True)

        MyOutputType = namedtuple(
            "MyOutputType", ["a", "b", "c", "d"], defaults=(t, t, t, t)
        )

        inp = MyOutputType()
        out = m(inp)
        # Ensure hooks are registered
        for x in out:
            self.assertNotEqual([], list(x._backward_hooks.values()))

        # TODO: we should check backward() and param is resharded
        # as well, but this is blocked by
        # https://github.com/pytorch/pytorch/issues/83107 and
        # https://github.com/pytorch/pytorch/issues/83129

    @skip_if_lt_x_gpu(2)
    def test_device_id_auto_wrap(self):
        # [FUNCTION DOC] Test driver (2 GPUs): runs the device_id propagation scenario with
        # the wrap policy given as a callable and as a ModuleWrapPolicy.
        """Tests that ``auto_wrap_policy`` propagates ``device_id`` to all
        nested FSDP instances."""
        self.run_subtests(
            {"use_callable": [False, True]},
            self._test_device_id_auto_wrap,
        )

    def _test_device_id_auto_wrap(self, use_callable: bool):
        # [FUNCTION DOC] HELPER: Wraps a transformer test model by encoder/decoder layer
        # (policy as callable or ModuleWrapPolicy) with device_id set, and asserts every
        # resulting FSDP instance has the compute device equal to that accelerator index.
        module_classes = {TransformerEncoderLayer, TransformerDecoderLayer}
        if use_callable:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=module_classes,
            )
        else:
            auto_wrap_policy = ModuleWrapPolicy(module_classes)
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.accelerator.current_device_index(),
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs,
        )
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            self.assertEqual(
                fsdp_module.compute_device,
                torch.device(device_type, torch.accelerator.current_device_index()),
            )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_cpu_offload(self):
        # [FUNCTION DOC] Test driver (2 GPUs): runs the device_id plus parameter CPU offload
        # scenario with use_orig_params on and off.
        """
        Tests FSDP when specifying both ``device_id`` and parameter CPU
        offloading.
        """
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_fsdp_device_id_cpu_offload,
        )

    def _test_fsdp_device_id_cpu_offload(self, use_orig_params: bool):
        # [FUNCTION DOC] HELPER: Wraps a model with nested FSDP units (Sequential wrapped, root
        # also owning a linear) using CPU parameter offload and a device_id, and asserts every
        # flat parameter handle's flat parameter is on CPU.
        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: A sequential of two 10x10 linears followed by a third
            # 10x10 linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the sequential (seq) and the linear (lin).
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                )
                self.lin = nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies seq then lin to x.
                return self.lin(self.seq(x))

        model = MyModel()
        # Choose a wrapping policy such that there are (1) nested FSDP
        # instances and (2) the parent FSDP instance has managed parameters
        auto_wrap_policy = ModuleWrapPolicy({nn.Sequential})
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.accelerator.current_device_index(),
            use_orig_params=use_orig_params,
        )
        cpu_device = torch.device("cpu")
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.device, cpu_device)

    @skip_if_lt_x_gpu(2)
    def test_module_device_mismatches_device_id(self):
        # [FUNCTION DOC] Test (2 GPUs): builds and wraps a module already on the rank's device
        # with device_id fixed to 0. Rank 0 must succeed; other ranks must raise ValueError
        # naming the mismatching devices. Sets the current device index to the rank.
        """Tests that specifying a ``device_id`` argument to FSDP for a GPU
        module that does not match the GPU device ID raises an error."""
        # TODO: override FSDP MT Thread _run to set this instead of here for
        # every test.
        torch.accelerator.set_device_index(self.rank)

        context = (
            self.assertRaisesRegex(
                ValueError, f"{device_type}:{self.rank} vs {device_type}:0"
            )
            if self.rank != 0
            else nullcontext()
        )

        with context:
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                # Move wrapped modules to CUDA before wrapping with FSDP
                device_init_mode=DEVICEInitMode.DEVICE_BEFORE,
                # Should raise error since rank 1 is given `device_id=0` when
                # the model is on cuda:1
                fsdp_kwargs={"device_id": 0},
            )

    @skip_if_lt_x_gpu(2)
    def test_cpu_gpu_module(self):
        # [FUNCTION DOC] Test (2 GPUs): a module with one layer on the accelerator and one on
        # CPU can be wrapped when device_id is given (all parameters then end up on the
        # accelerator), and raises RuntimeError asking for device_id when it is not.
        """Tests a CPU + GPU module supported if device_id is passed
        in, errors if device_id is not.
        """
        torch.accelerator.set_device_index(self.rank)

        class CPUGPUModule(nn.Module):
            # [CLASS DOC] HELPER: Module with layer a on the accelerator and layer b on CPU.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates 1x1 linears a (moved to device) and b (CPU).
                super().__init__()
                self.a = nn.Linear(1, 1).to(device=device_type)
                self.b = nn.Linear(1, 1)

        cpu_gpu = CPUGPUModule()
        fsdp = FSDP(cpu_gpu, device_id=torch.accelerator.current_device_index())
        for param in fsdp.parameters():
            self.assertEqual(
                param.device, torch.device(torch.accelerator.current_device_index())
            )

        # without device_id, we hit an error
        with self.assertRaisesRegex(RuntimeError, "please pass in device_id"):
            FSDP(CPUGPUModule())

    @skip_if_lt_x_gpu(2)
    def test_fsdp_ignored_module_meta(self):
        # [FUNCTION DOC] Test (2 GPUs): wraps a meta-device module with one layer ignored, with
        # and without a param_init_fn that materializes modules, and asserts the ignored
        # layer's parameters remain on the meta device. Sets the current device index to rank.
        torch.accelerator.set_device_index(self.rank)

        class CPUGPUModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two 1x1 linear layers a and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear layers a and b.
                super().__init__()
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)

        with torch.device("meta"):
            m = CPUGPUModule()
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        meta_device = torch.device("meta")
        self.assertEqual(meta_device, next(m.a.parameters()).device)

        # Test with param_init_fn
        with torch.device("meta"):
            m = CPUGPUModule()
        m = FSDP(
            m,
            device_id=torch.accelerator.current_device_index(),
            ignored_modules=[m.a],
            use_orig_params=True,
            param_init_fn=lambda m: m.to_empty(
                device=torch.accelerator.current_device_index(), recurse=False
            ),
        )
        self.assertEqual(meta_device, next(m.a.parameters()).device)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_no_move_ignored_params_and_bufs(self):
        # [FUNCTION DOC] Test (2 GPUs): wraps a CPU module with device_id while ignoring one
        # layer (which also has a buffer), and asserts the ignored layer's parameters and
        # buffers stay on CPU.
        class CPUGPUModule(nn.Module):
            # [CLASS DOC] HELPER: Module with two 1x1 linears a and b; a also carries a buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linears a and b and a buffer of ones on a.
                super().__init__()
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)
                self.a.buf = torch.nn.Buffer(torch.ones(1))

        m = CPUGPUModule()
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        ignored_params = m.a.parameters()
        ignored_bufs = m.a.buffers()
        for t in chain(ignored_params, ignored_bufs):
            self.assertEqual(torch.device("cpu"), t.device)

    @skip_if_lt_x_gpu(2)
    def test_multigpu_module(self):
        # [FUNCTION DOC] Test (2 GPUs): wrapping a module whose layers sit on two different
        # CUDA devices must raise RuntimeError saying FSDP supports only single device modules.
        # KNOWN ISSUE: uses cuda directly, so it is not device generic.
        """
        Module on multiple GPUs wrapped in FSDP should raise an error.
        """

        class MultiGPUModule(nn.Module):
            # [CLASS DOC] HELPER: Module with layer a on this rank's CUDA device and layer b
            # on the next rank's device.
            def __init__(self, rank):
                # [FUNCTION DOC] HELPER: Stores the rank and creates 1x1 linears a and b on
                # CUDA devices rank and rank+1 (modulo world size).
                super().__init__()
                self.rank = rank
                self.a = nn.Linear(1, 1).cuda(self.rank)
                self.b = nn.Linear(1, 1).cuda((self.rank + 1) % dist.get_world_size())

        with self.assertRaisesRegex(
            RuntimeError, "FSDP only supports single device modules"
        ):
            FSDP(MultiGPUModule(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_no_params(self):
        # [FUNCTION DOC] Test (2 GPUs): wraps a parameter-free module on CPU, on the device,
        # and on CPU with device_id; all must succeed. Wrapping a device module with device_id
        # 0 must succeed on rank 0 and raise ValueError about inconsistent devices on others.
        # Sets the current device index to the rank.
        """
        Test that device_id and cpu init work if module has no params
        (they are effective noops, but ensure FSDP does not assume module
        has parameters during init)
        """
        # TODO: override FSDP MT Thread _run to set this instead of here for
        # every test.
        torch.accelerator.set_device_index(self.rank)
        # Test CPU
        no_params = nn.ReLU()
        FSDP(no_params)
        # Test CUDA
        no_params = nn.ReLU().to(device=device_type)
        FSDP(no_params)
        # Test CPU + device_id
        no_params = nn.ReLU()
        FSDP(no_params, device_id=torch.accelerator.current_device_index())
        # For modules with no params, wrong device_id will raise error about
        # inconsistency between compute_device and device_id, since compute_device
        # is computed as torch.cuda.current_device when there are no params.
        no_params = nn.ReLU().to(device=device_type)
        context = (
            (
                self.assertRaisesRegex(
                    ValueError, f"Inconsistent.*cuda:{self.rank} vs cuda:0"
                )
            )
            if self.rank != 0
            else nullcontext()
        )
        with context:
            FSDP(no_params, device_id=0)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_same_model_across_ranks(self):
        # [FUNCTION DOC] Test (2 GPUs): builds per-rank different models (rank-seeded weights
        # and buffers), asserts they differ across ranks, wraps with sync_module_states
        # (once for a device model, once for a CPU model plus device_id) and asserts parameters
        # and buffers are identical across ranks once full params are summoned.
        """
        FSDP broadcasts model from rank 0 to ensure it starts off with the same
        values.
        """

        class MyModel(nn.Module):
            # [CLASS DOC] HELPER: Module whose weights and buffer depend on the rank given at
            # construction, so models differ between ranks.
            def __init__(self, rank):
                # [FUNCTION DOC] HELPER: Seeds the CPU and device RNGs with the rank, then
                # creates a bias-free 10x10 linear and a one-element buffer holding the rank.
                super().__init__()
                # Seed via rank to make model different across ranks
                torch.manual_seed(rank)
                torch.get_device_module(device_type).manual_seed(rank)
                self.lin = nn.Linear(10, 10, bias=False)
                self.buffer = nn.Buffer(torch.ones(1) * rank)

        m = MyModel(self.rank).to(device=device_type)
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        # Passing sync_module_states into FSDP makes model the same during init.
        fsdp = FSDP(m, sync_module_states=True)
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )

        # sync_module_states also works with CPU module with device_id passed in
        m = MyModel(self.rank)
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        # Passing sync_module_states into FSDP makes model the same during init.
        fsdp = FSDP(
            m,
            device_id=torch.accelerator.current_device_index(),
            sync_module_states=True,
        )
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/105024")
    @skip_if_lt_x_gpu(2)
    def test_homogeneous_attributes(self):
        # [FUNCTION DOC] Test driver (2 GPUs, skipped on Linux for a tracked issue): checks that
        # the hard-coded list of homogeneous attributes (use_orig_params, limit_all_gathers,
        # full precision in eval) matches the library's HOMOGENEOUS_ATTR_NAMES, then runs the
        # heterogeneity check for each attribute.
        """
        Tests that passing heterogeneous values for attributes designated as
        homogeneous raises an error.
        """
        # Manually construct this list but verify against the global list of
        # homogeneous attribute names
        all_attr_name_and_values = [
            ("_use_orig_params", False, True),
            ("limit_all_gathers", False, True),
            ("_use_full_prec_in_eval", False, True),
        ]
        self.assertEqual(
            [
                attr_name_and_values[0]
                for attr_name_and_values in all_attr_name_and_values
            ],
            HOMOGENEOUS_ATTR_NAMES,
        )

        self.run_subtests(
            {"attr_name_and_values": all_attr_name_and_values},
            self._test_homogeneous_attributes,
        )

    def _test_homogeneous_attributes(self, attr_name_and_values: tuple[str, Any, Any]):
        # [FUNCTION DOC] HELPER: Wraps an inner submodule and the outer model with different
        # values of the attribute (for the full-precision attribute, by setting an environment
        # variable between the two wraps) and asserts the first forward raises ValueError
        # saying it expects one homogeneous value. Side effect: may leave the
        # FSDP_USE_FULL_PREC_IN_EVAL environment variable set.
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            {},
        )
        attr_name = attr_name_and_values[0]

        if "_use_full_prec_in_eval" == attr_name:
            model.module[1] = FSDP(model.module[1])
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = "1"
            fsdp_model = FSDP(model)
        else:
            fsdp_kwargs_inner = {attr_name.lstrip("_"): attr_name_and_values[1]}
            fsdp_kwargs_outer = {attr_name.lstrip("_"): attr_name_and_values[2]}
            model.module[1] = FSDP(model.module[1], **fsdp_kwargs_inner)
            fsdp_model = FSDP(model, **fsdp_kwargs_outer)

        # Run a forward to trigger lazy initialization and the error
        with self.assertRaisesRegex(
            ValueError, f"Expects one homogeneous value for {attr_name}"
        ):
            inp = fsdp_model.module.get_input(torch.device(device_type))
            fsdp_model(*inp)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/137948")
    @skip_if_lt_x_gpu(2)
    def test_fsdp_unsupported_module_cls(self):
        # [FUNCTION DOC] Test (2 GPUs, skipped on Linux for a tracked issue): wrapping a
        # ModuleList and a ModuleDict (containers without forward) must warn that FSDP will
        # not all-gather parameters for such containers.
        regex = r"FSDP will not all-gather parameters for containers that do not implement forward"
        model = nn.ModuleList([MLP(8, torch.device("cpu")) for _ in range(3)])
        with self.assertWarnsRegex(UserWarning, regex):
            FSDP(model, device_id=device_type)
        model = nn.ModuleDict(
            {"1": MLP(8, torch.device("cpu")), "2": MLP(8, torch.device("cpu"))}
        )
        with self.assertWarnsRegex(UserWarning, regex):
            FSDP(model)


class TestFSDPMiscWorldSize1(FSDPTestMultiThread):
    # [CLASS DOC] Single-rank multi-threaded test case for FSDP behavior specific to world
    # size 1: strategy fallback warnings, device mismatch errors, and the unsafe setattr switch.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property giving the number of ranks; always 1.
        return 1

    @skip_if_lt_x_gpu(1)
    def test_world_size_1_sharding_strategy_warning(self):
        # [FUNCTION DOC] Test (1 GPU): with world size 1, explicitly passing NO_SHARD must not
        # emit the "switching to NO_SHARD" UserWarning, while FULL_SHARD (explicit or default)
        # and SHARD_GRAD_OP must emit it with the strategy name.
        """
        Tests that FSDP issues a warning when it switches to using ``NO_SHARD``
        when the world size is 1.
        """
        warning_prefix = "FSDP is switching to use `NO_SHARD` instead of"
        # If the user already passes `NO_SHARD`, then there should not be a
        # warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # trigger all warnings
            FSDP(
                nn.Linear(3, 3).to(device=device_type),
                sharding_strategy=ShardingStrategy.NO_SHARD,
            )
            for warning in w:
                self.assertTrue(
                    warning.category is not UserWarning
                    or not str(warning.message).startswith(warning_prefix)
                )

        # Check that a warning is issued
        warning_suffix = " since the world size is 1."
        # - Pass `FULL_SHARD` or `None`
        expected_regex_full_shard = (
            warning_prefix + " " + str(ShardingStrategy.FULL_SHARD) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(
                nn.Linear(3, 3).to(device=device_type),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            )
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(nn.Linear(3, 3).to(device=device_type))
        # - Pass `SHARD_GRAD_OP`
        expected_regex_shard_grad_op = (
            warning_prefix + " " + str(ShardingStrategy.SHARD_GRAD_OP) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_shard_grad_op):
            FSDP(
                nn.Linear(3, 3).to(device=device_type),
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            )

    @skip_if_lt_x_gpu(1)
    def test_training_device_mismatch_errors(self):
        # [FUNCTION DOC] Test (1 GPU): a CPU module wrapped without a device move must raise
        # RuntimeError at forward telling the user to move it to the GPU; a module with
        # parameter CPU offload that was moved to the GPU must raise RuntimeError telling the
        # user not to move it from CPU.
        # KNOWN ISSUE: the expected messages hardcode cuda:0, so it fails on other accelerators.
        """
        Tests that, when training starts, if FSDP parameters are not on the
        expected device, then an informative error is raised. This applies for
        both no parameter CPU offloading and parameter CPU offloading.
        """
        # Incorrectly not moving from CPU -> GPU
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model)
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module unexpectedly has parameters on cpu. Make "
            "sure to move the module to cuda:0 before training.",
        ):
            fsdp_model(inp)

        # Incorrectly moving from CPU -> GPU
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
        fsdp_model.to(torch.device(device_type))
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module with parameter CPU offloading enabled has "
            "parameters on cuda:0. Make sure to not move the module from CPU "
            "when offloading parameters.",
        ):
            fsdp_model(inp)

    @skip_if_lt_x_gpu(2)
    def test_unsafe_setattr(self):
        # [FUNCTION DOC] Test driver (2 GPUs): runs the unsafe setattr scenario with
        # use_orig_params on and off.
        """
        Tests that the environment variable for using unsafe setattr gates as
        expected.
        """
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_unsafe_setattr,
        )

    def _test_unsafe_setattr(self, use_orig_params: bool):
        # [FUNCTION DOC] HELPER: Wraps a module that records whether its attribute-setting
        # override ran, and runs forward under three settings of the unsafe setattr
        # environment variable: unset (override runs), "1" (override skipped) and "0"
        # (override runs). Side effect: leaves the variable set to "0" afterwards.
        called_setattr_override = False

        class SetattrLinear(nn.Module):
            # [CLASS DOC] HELPER: Matrix-product layer with a single weight whose attribute
            # setter records (in the enclosing test's flag) that it was called.
            def __init__(self, in_dim: int, out_dim: int, device: torch.device) -> None:
                # [FUNCTION DOC] HELPER: Creates the random in_dim by out_dim weight on device.
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn((in_dim, out_dim), device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and the weight.
                return x @ self.weight

            def __setattr__(self, name: str, value: Any) -> None:
                # [FUNCTION DOC] HELPER: Sets the enclosing flag to true, then performs the
                # normal attribute assignment and returns its result.
                nonlocal called_setattr_override
                called_setattr_override = True
                return super().__setattr__(name, value)

        # Construct FSDP module without changing any environment variables and
        # run forward, which triggers both unsharded and sharded view setting
        module = SetattrLinear(5, 5, torch.device(device_type))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        inp = torch.randn((8, 5), device=torch.device(device_type))
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)

        # Repeat with unsafe setattr explicitly enabled
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "1"
        module = SetattrLinear(5, 5, torch.device(device_type))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertFalse(called_setattr_override)

        # Repeat with unsafe setattr explicitly disabled
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "0"
        module = SetattrLinear(5, 5, torch.device(device_type))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)


instantiate_parametrized_tests(TestFSDPMiscMultiThread)
instantiate_parametrized_tests(TestFSDPMiscMultiProcess)

if __name__ == "__main__":
    run_tests()
