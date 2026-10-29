# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Training tests for the composable replicate API built on the FSDP2 machinery
# (replicate_with_fsdp): forward input device moves, parameter DTensor versus tensor
# registration, dtype casts, train parity against a manually gradient-averaged reference
# (single and multiple groups, delays, CPU offload, non-root runs, prefetching, post-optimizer
# events), composition with activation checkpointing, shared parameters, gradient
# accumulation, microbatching, custom forward methods and tensor parallelism. Import-time:
# defines c10d_ops and funcol aliases and device_type from the detected device.

import contextlib
import copy
import functools
import itertools
import unittest
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    apply_activation_checkpointing,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    OffloadPolicy,
    register_fsdp_forward_method,
)
from torch.distributed.tensor import DTensor, init_device_mesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    compiled_fsdp_test,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    MLPStack,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_HPU,
    wrapSwapTensorsTest,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional

from torch.testing._internal.common_fsdp import get_devtype


device_type = torch.device(get_devtype())


class TestReplicateForwardInputs(FSDPTestMultiThread):
    # [CLASS DOC] Multi-threaded (2 ranks) test case for how the replicate wrapper treats
    # forward inputs.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the thread world size 2.
        return 2

    @skip_if_lt_x_gpu(1)
    def test_root_move_forward_input_to_device(self):
        # [FUNCTION DOC] Wraps a parameter-less module on the accelerator with replicate,
        # calls it with CPU tensors (a tensor and a tuple of two tensors), and relies on the
        # module's forward raising an AssertionError if replicate did not move all inputs,
        # including those nested in the tuple, to the accelerator.
        device = torch.device(device_type.type, 0)

        class ParamlessModule(nn.Module):
            # [CLASS DOC] HELPER: Module without parameters whose forward verifies input
            # devices.
            def forward(self, x: torch.Tensor, ys: tuple[torch.Tensor, ...]):
                # [FUNCTION DOC] HELPER: Raises AssertionError if x or either tensor in ys
                # is not on the target device; otherwise returns x plus the sum of ys plus 1.
                # Check that Replicate moved the inputs to GPU, including recursing
                # into the tuple data structure
                if not (x.device == device):
                    raise AssertionError(f"Expects {device} but got {x.device}")
                if not (ys[0].device == device):
                    raise AssertionError(f"Expects {device} but got {ys[0].device}")
                if not (ys[1].device == device):
                    raise AssertionError(f"Expects {device} but got {ys[1].device}")
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule().to(device)
        replicate(model).to(device)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


class TestReplicateRegisteredParams(FSDPTestMultiThread):
    # [CLASS DOC] Multi-threaded (4 ranks) test case checking whether module parameters are
    # DTensors (sharded/replicated form) or plain tensors at different points of a replicate
    # module's lifecycle, plus helper assertions for that.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the thread world size 4.
        return 4

    @skip_if_lt_x_gpu(1)
    def test_param_registration_after_forward(self):
        # [FUNCTION DOC] For a single root replicate group and for nested groups, asserts
        # parameters are DTensors after wrapping, plain tensors after a forward, and DTensors
        # again after a manual reshard, always equal in value to a reference copy. Seeds are
        # synced across threads by broadcasting parameters from rank 0.
        """Tests the parameter registration after forward."""
        device = torch.device(device_type.type, 0)
        # Single Replicate group
        torch.manual_seed(42)
        model = MLP(3, device)
        # Since seed is per process, not per thread, we broadcast to ensure
        # the same parameters across ranks
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        replicate(model)  # root only
        inp = torch.randn((2, 3), device=device_type.type)
        self._assert_dtensor_params(model.parameters())
        self._assert_same_params(model.parameters(), ref_model.parameters())
        model(inp)
        self._assert_tensor_params(model.parameters())
        self._assert_same_params(model.parameters(), ref_model.parameters())
        model.reshard()  # however, we can manually reshard
        self._assert_dtensor_params(model.parameters())
        self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple Replicate groups
        torch.manual_seed(42)
        model = nn.Sequential(MLP(3, device), MLP(3, device))
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model)
        replicate(model[0].in_proj)
        replicate(model[0].out_proj)
        replicate(model)

        self._assert_dtensor_params(model.parameters())
        self._assert_same_params(model.parameters(), ref_model.parameters())
        model(inp)
        non_root_params = list(model[0].in_proj.parameters()) + list(
            model[0].out_proj.parameters()
        )
        root_params = list(set(model.parameters()) - set(non_root_params))
        self._assert_tensor_params(non_root_params)
        self._assert_tensor_params(root_params)
        self._assert_same_params(model.parameters(), ref_model.parameters())
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()  # however, we can manually reshard
        self._assert_dtensor_params(model.parameters())
        self._assert_same_params(model.parameters(), ref_model.parameters())

    @skip_if_lt_x_gpu(1)
    def test_param_registration_after_backward(self):
        # [FUNCTION DOC] For a single root group and for nested groups, asserts parameters
        # are DTensors both before forward and after a forward plus backward.
        """Tests the parameter registration after backward."""
        device = torch.device(device_type.type, 0)
        # Single Replicate group
        model = MLP(8, device)
        replicate(model)  # root only
        inp = torch.randn((2, 8), device=device_type.type)
        self._assert_dtensor_params(model.parameters())
        model(inp).sum().backward()
        self._assert_dtensor_params(model.parameters())

        # Multiple Replicate groups
        model = MLP(8, device)
        replicate(model.in_proj)
        replicate(model.out_proj)
        replicate(model)
        self._assert_dtensor_params(model.parameters())
        model(inp).sum().backward()
        self._assert_dtensor_params(model.parameters())

    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        # [FUNCTION DOC] HELPER: Materializes params and asserts the list is non-empty and
        # that each element is a plain tensor and not a DTensor.
        # need to iterate over the list multiple times
        params = list(params)
        self.assertGreater(len(params), 0)
        for param in params:
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        # [FUNCTION DOC] HELPER: Materializes params and asserts the list is non-empty and
        # every element is a DTensor.
        params = list(params)
        self.assertGreater(len(params), 0)
        for param in params:
            self.assertIsInstance(param, DTensor)

    def _assert_same_params(
        self, params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter]
    ):
        # [FUNCTION DOC] HELPER: Asserts both parameter lists have the same length and that
        # each pair has equal shape and values, converting DTensor parameters to full tensors
        # first (a collective operation).
        params, ref_params = list(params), list(ref_params)
        self.assertEqual(len(params), len(ref_params))
        for param, ref_param in zip(params, ref_params):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            self.assertEqual(param.shape, ref_param.shape)
            self.assertEqual(param, ref_param)


class TestReplicateCastAfterInit(FSDPTestMultiThread):
    # [CLASS DOC] Multi-threaded (2 ranks) test case for changing a replicated module's dtype
    # after wrapping.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the thread world size 2.
        return 2

    @skip_if_lt_x_gpu(1)
    @wrapSwapTensorsTest(True)
    def test_to_float64_after_init(self):
        # [FUNCTION DOC] Wraps an MLP with replicate at three levels, casts it to float64,
        # and asserts parameter dtypes (including the local shard and DTensor metadata). Then
        # trains 10 iterations against a float64 reference with manually averaged gradients,
        # asserting equal losses, sharded parity, float64 gradients and DTensor metadata,
        # with Adam steps and alternating set-to-none zeroing. Uses swap-tensors mode.
        """Tests that the user can cast the module to float64 after init."""
        # NOTE: Test fp64 instead of a lower precision dtype like bf16 for
        # better numerics. The important part is changing the dtype.

        torch.manual_seed(42)
        mlp_dim, device, dtype = 4, device_type, torch.float64
        model = MLP(mlp_dim, device=device)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model).to(dtype)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in (model.in_proj, model.out_proj, model):
            replicate(module)
        model.to(dtype)
        for param in model.parameters():
            self.assertEqual(param.dtype, dtype)
            self.assertEqual(param.to_local().dtype, dtype)
            self.assertEqual(param._spec.tensor_meta.dtype, dtype)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device=device_type.type, dtype=dtype)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for param in model.parameters():
                self.assertEqual(param.dtype, dtype)
                self.assertEqual(param.to_local().dtype, dtype)
                self.assertEqual(param._spec.tensor_meta.dtype, dtype)
                self.assertEqual(param.grad.dtype, dtype)
                self.assertEqual(param.grad.to_local().dtype, dtype)
                self.assertEqual(param.grad._spec.tensor_meta.dtype, dtype)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


class TestReplicate1DTrainingCore(FSDPTest):
    # [CLASS DOC] Multi-process test case on a 1D replicate mesh: training parity of
    # replicate against a manually gradient-averaged reference, with overlap delays, CPU
    # offload, non-root runs, multi-forward modules, explicit prefetching and optimizer
    # events.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of 8 and the device count of the
        # current device type.
        return min(8, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_single_group(self):
        # [FUNCTION DOC] Runs the single-group parity check as subtests over three pairs of
        # linear layer shapes (including dimensions not divisible by the world size).
        """
        Tests train parity with DDP for a single FSDP group when sharding
        parameters on dim-0.
        """
        self.run_subtests(
            {
                "lin_shapes": [
                    [(16, 15), (15, 8)],
                    [(7, 15), (15, 3)],
                    [(16, 17), (17, 8)],
                ],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(self, lin_shapes: list[tuple[int, int]]):
        # [FUNCTION DOC] HELPER: Builds a two-linear-layer ReLU network from lin_shapes, a
        # reference copy on the device, and replicates the original. Trains both 10 steps
        # with Adam on rank-dependent input; reference gradients are summed with all-reduce
        # and divided by world size. Alternates set-to-none zeroing; asserts equal losses.
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
        )
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        inp = (torch.randn((4, lin_shapes[0][0]), device=device_type.type),)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(*inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep kernel not supported for HPU")
    @compiled_fsdp_test(compile_compute_on_module=Transformer)
    def test_train_parity_multi_groups(self):
        # [FUNCTION DOC] Runs the multi-group parity check as subtests with the default
        # offload policy and all combinations of artificial delays (after forward, before
        # all-gather, before reduce-scatter, before optimizer). Skipped on HPU; the
        # transformer compute is compiled when the compiled test mode is enabled.
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "test_device_type": [device_type.type],
                "offload_policy": [OffloadPolicy()],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
                "unshard_async_op": [False],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "sleep kernel not supported on HPU")
    def test_train_parity_multi_group_cpu_offload_eager(self):
        # [FUNCTION DOC] Runs the multi-group parity check as subtests with CPU offload,
        # pinned and unpinned memory, over the same delay combinations. Skipped on HPU.
        """
        Tests train parity when using multiple parameter groups for
        communication and CPU offloading.
        """
        self.run_subtests(
            {
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "test_device_type": [device_type.type],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
                "unshard_async_op": [False],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        offload_policy: OffloadPolicy,
        test_device_type: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
        unshard_async_op: bool,
    ):
        # [FUNCTION DOC] HELPER: Returns early (skips) when exactly two or three of the four
        # delay flags are set. Raises AssertionError for an unexpected device type. Builds a
        # 3-layer Transformer (vocab 1024, no dropout), a reference copy, and replicates each
        # transformer block and the root on a 1D mesh with the given offload policy,
        # optionally enabling async unshard. When requested, it patches all-gather and
        # reduce-scatter with versions that sleep 100 ms first, and sleeps after forward or
        # before the optimizer step. Trains 10 Adam steps against the reference (gradients
        # all-reduced and averaged) and asserts equal losses.
        # Only test individual delays or all four delays to save test time
        if (
            delay_after_forward
            + delay_before_all_gather
            + delay_before_reduce_scatter
            + delay_before_optim
            in (2, 3)
        ):
            return
        if test_device_type not in ("cuda", "hpu", "xpu", "cpu"):
            raise AssertionError(f"Unexpected device type: {test_device_type}")
        torch.manual_seed(42)
        vocab_size = 1024
        model_args = ModelArgs(
            n_layers=3,
            n_heads=4,
            vocab_size=vocab_size,
            max_seq_len=64,
            dropout_p=0,
        )
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mesh = init_device_mesh(
            test_device_type,
            (self.world_size,),
            mesh_dim_names=("replicate",),
        )
        fully_shard_fn = functools.partial(
            replicate,
            mesh=mesh,
            offload_policy=offload_policy,
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
        fully_shard_fn(model)
        if unshard_async_op:
            model._set_unshard_async_op(unshard_async_op)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_single
        orig_reduce_scatter = dist.reduce_scatter_single

        def delayed_all_gather(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Sleeps the device for 100 ms of cycles, then calls the
            # original all-gather with the same arguments and returns its result.
            torch.get_device_module(device_type)._sleep(
                int(delay_in_ms * get_cycles_per_ms())
            )
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Sleeps the device for 100 ms of cycles, then calls the
            # original reduce-scatter with the same arguments and returns its result.
            torch.get_device_module(device_type)._sleep(
                int(delay_in_ms * get_cycles_per_ms())
            )
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
        patch_all_gather_ctx = (
            patch_all_gather(delayed_all_gather)
            if delay_before_all_gather
            else contextlib.nullcontext()
        )
        patch_reduce_scatter_ctx = (
            patch_reduce_scatter(delayed_reduce_scatter)
            if delay_before_reduce_scatter
            else contextlib.nullcontext()
        )
        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                inp = torch.randint(0, vocab_size, (3, 64), device=device_type)
                losses: list[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    losses.append(_model(inp).sum())
                    if _model is model and delay_after_forward:
                        torch.get_device_module(device_type)._sleep(
                            int(delay_in_ms * get_cycles_per_ms())
                        )
                    losses[-1].backward()
                    if _model is model and delay_before_optim:
                        torch.get_device_module(device_type)._sleep(
                            int(delay_in_ms * get_cycles_per_ms())
                        )

                for param in ref_model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)
                        param.grad.div_(self.world_size)

                for _optim in (ref_optim, optim):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    _optim.step()
                self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_non_root_forward_backward(self):
        # [FUNCTION DOC] Replicates three stacked MLPs and the root, runs a root
        # forward/backward and step, then a forward/backward through only the first child
        # (non-root) and another step, mirroring this on a reference with manual gradient
        # averaging. Asserts equal root and non-root losses and equal final outputs.
        """
        Tests running forward/backward through the root and then through a
        non-root. The non-root needs to synchronize streams/queue the callback.
        """
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            replicate(mlp)
        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=device_type)

        ref_root_loss = ref_model(inp).sum()
        ref_root_loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad)
            param.grad.detach().div_(self.world_size)
        ref_optim.step()
        ref_optim.zero_grad()
        ref_nonroot_loss = ref_model[0](inp).sum()
        ref_nonroot_loss.backward()
        for param in ref_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.detach().div_(self.world_size)
        ref_optim.step()

        root_loss = model(inp).sum()
        root_loss.backward()
        torch.get_device_module(device_type)._sleep(int(100 * get_cycles_per_ms()))
        optim.step()
        optim.zero_grad()
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        optim.step()

        self.assertEqual(ref_root_loss, root_loss)
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_module(self):
        # [FUNCTION DOC] Delegates to the helper that checks a module used twice per forward.
        """
        Tests parity when running a module that participates multiple
        times in forward.
        """

        self._test_multi_forward_module()

    def _test_multi_forward_module(self):
        # [FUNCTION DOC] HELPER: Builds a module where the inner linear layer runs twice per
        # forward, replicates the inner layer and the root, and trains 10 Adam steps against
        # a reference with manually averaged gradients, asserting equal losses.
        class MultiForwardModule(nn.Module):
            # [CLASS DOC] HELPER: Module with inner (4 to 4) and outer (4 to 5) linear
            # layers; forward applies inner twice to the same input and feeds the sum to outer.
            def __init__(self, device: torch.device):
                # [FUNCTION DOC] HELPER: Creates both linear layers on the given device.
                super().__init__()
                self.inner = nn.Linear(4, 4, device=device)
                self.outer = nn.Linear(4, 5, device=device)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns outer applied to the sum of two identical
                # inner applications on x.
                i = self.inner(x)
                j = self.inner(x)
                return self.outer(i + j)

        torch.manual_seed(42)
        model = MultiForwardModule(device=device_type.type)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        replicate(model.inner)
        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((32, 4), device=device_type.type)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()

            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_explicit_prefetching(self):
        # [FUNCTION DOC] Replicates each layer and the root of an 8-layer Transformer and
        # configures explicit forward prefetch (next two layers) and backward prefetch
        # (previous two layers). Trains 10 AdamW steps against a reference with manually
        # averaged gradients and asserts equal losses.
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=8, dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)

        for layer in itertools.chain(model.layers, [model]):
            replicate(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        num_to_forward_prefetch = num_to_backward_prefetch = 2
        for i, layer in enumerate(model.layers):
            if i >= len(model.layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [
                model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
            ]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)
        for i, layer in enumerate(model.layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [
                model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device=device_type.type)
        for _ in range(10):
            losses: list[torch.Tensor] = []

            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad()
                _optim.step()

            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_post_optim_event(self):
        # [FUNCTION DOC] Skipped on HPU. Replicates each layer and the root of a Transformer
        # and registers an optimizer post-step hook that records a stream event and sets it as
        # the post-optimizer event on the model. Trains a reference 10 AdamW steps with manual
        # gradient averaging, then the replicated model 10 steps sleeping 25 ms after each
        # optimizer step so the CPU runs ahead; finally asserts the losses are equal.
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type.type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for layer in itertools.chain(model.layers, [model]):
            replicate(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def step_post_hook(
            fsdp_module: FSDPModule, opt: torch.optim.Optimizer, args, kwargs
        ) -> None:
            # [FUNCTION DOC] HELPER: Optimizer post-step hook that records an event on the
            # current device stream and passes it to the module's set_post_optim_event.
            post_optim_event = (
                torch.get_device_module(device_type).current_stream().record_event()
            )
            fsdp_module.set_post_optim_event(post_optim_event)

        optim.register_step_post_hook(functools.partial(step_post_hook, model))

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device=device_type.type)
        # Track all losses and check for equality at the end to avoid a CPU
        # sync point after each iteration
        ref_losses: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for _ in range(10):
            ref_optim.zero_grad()
            ref_losses.append(ref_model(inp).sum())
            ref_losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            ref_optim.step()
        for _ in range(10):
            optim.zero_grad()
            losses.append(model(inp).sum())
            losses[-1].backward()
            optim.step()
            # Sleep after the optimizer step to allow CPU to run ahead into the
            # next iteration's forward, exercising the post-optim stream sync
            torch.get_device_module(device_type)._sleep(int(25 * get_cycles_per_ms()))
        for ref_loss, loss in zip(ref_losses, losses):
            self.assertEqual(ref_loss, loss)


class TestReplicateTrainingCompose(FSDPTest):
    # [CLASS DOC] Multi-process test case (up to 2 ranks) checking replicate composed with
    # activation checkpointing in several implementations and module groupings.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of the device count and 2.
        # Since these tests run with a larger transformer model, they may see
        # some numeric drift with >2 GPUs
        return min(torch.get_device_module(device_type).device_count(), 2)

    @skip_if_lt_x_gpu(2)
    @compiled_fsdp_test(compile_compute_on_module=Transformer)
    def test_train_parity_with_activation_checkpointing(self):
        # [FUNCTION DOC] Runs the checkpointing parity check as subtests over checkpoint
        # implementation (composable, utils, wrapper) and module grouping (block, mem_eff,
        # mem_eff_weight_tied). The docstring says parity is against DDP, but the reference
        # is a plain model with manually averaged gradients.
        """
        Tests train parity against DDP when composing with activation
        checkpointing.
        """
        self.run_subtests(
            {
                "checkpoint_impl": ["composable", "utils", "wrapper"],
                "module_grouping": ["block", "mem_eff", "mem_eff_weight_tied"],
                "test_device_type": [device_type.type],
            },
            self._test_train_parity_with_activation_checkpointing,
        )

    def _test_train_parity_with_activation_checkpointing(
        self,
        checkpoint_impl: str,
        module_grouping: str,
        test_device_type: str,
    ):
        # [FUNCTION DOC] HELPER: Raises AssertionError for an unknown checkpoint_impl, and
        # NotImplementedError for an unknown module_grouping. Returns early when testing
        # with a compiled replace of replicate and impl is composable. Builds a 3-layer
        # Transformer (weight tying except for mem_eff), applies the chosen activation
        # checkpointing (config flag, wrapper, or composable checkpoint per block), then
        # replicates in the chosen grouping (per block; or mem_eff lists of modules with
        # embedding unshard-in-backward disabled; or tied embedding/output together) and
        # the root, on a 1D replicate mesh. Trains 10 Adam steps against a reference with
        # manual gradient averaging, checking sharded parity before and after each step
        # (except when compiled) and equal losses.
        if checkpoint_impl not in ("composable", "utils", "wrapper"):
            raise AssertionError(f"Unexpected checkpoint_impl: {checkpoint_impl}")
        testing_compile = (
            replicate is not torch.distributed._composable.replicate_with_fsdp
        )
        if testing_compile and checkpoint_impl == "composable":
            return
        torch.manual_seed(42)
        vocab_size = 1024
        with torch.device(device_type):
            model_args = ModelArgs(
                n_layers=3,
                n_heads=4,
                vocab_size=vocab_size,
                max_seq_len=64,
                dropout_p=0,
                checkpoint_activations=(checkpoint_impl == "utils"),
                # For the mem-efficient module grouping, we separate the
                # embeddings from the output projection, which does not support
                # weight tying
                weight_tying=module_grouping != "mem_eff",
            )
            model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        # Apply activation checkpointing
        prefixes_to_ignore = ()
        if checkpoint_impl == "wrapper":
            prefixes_to_ignore = (_CHECKPOINT_PREFIX,)
            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, TransformerBlock)
            )
        elif checkpoint_impl == "composable":
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    checkpoint(module)

        # Apply Replicate
        device_mesh = init_device_mesh(
            test_device_type,
            (self.world_size,),
            mesh_dim_names=("replicate",),
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
        }
        if module_grouping == "mem_eff":
            if not (model_args.n_layers == 3):
                raise AssertionError(
                    f"Expected n_layers == 3, got {model_args.n_layers}"
                )
            replicate(model.layers[0], **fsdp_kwargs)
            replicate([model.layers[1], model.layers[2]], **fsdp_kwargs)
            replicate([model.tok_embeddings, model.pos_embeddings], **fsdp_kwargs)
            # Embedding weights are not needed for embedding backward
            model.tok_embeddings.set_unshard_in_backward(False)
            replicate([model.norm, model.output], **fsdp_kwargs)
        elif module_grouping == "mem_eff_weight_tied":
            replicate([model.tok_embeddings, model.output], **fsdp_kwargs)
            for layer in model.layers:
                replicate(layer, **fsdp_kwargs)
        elif module_grouping == "block":
            for layer in model.layers:
                replicate(layer, **fsdp_kwargs)
        else:
            raise NotImplementedError(f"Unknown module grouping: {module_grouping}")
        replicate(model, **fsdp_kwargs)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        # Reuse the same input across iterations to avoid loss explosion from
        # trying to learn from random inputs
        inp = torch.randint(0, vocab_size, (3, 64), device=device_type.type)
        check_sharded_parity(
            self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
        )
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                torch.manual_seed(iter_idx + 1)  # for dropout determinism
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            if not testing_compile:
                check_sharded_parity(
                    self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
                )
            self.assertEqual(losses[0], losses[1])
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            if not testing_compile:
                check_sharded_parity(
                    self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
                )


class TestReplicateSharedParams(FSDPTest):
    # [CLASS DOC] Multi-process test case (up to 4 ranks) for replicate with weight-tied
    # (shared) parameters.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of 4 and the device count.
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_shared_params(self):
        # [FUNCTION DOC] Runs the shared-parameter parity check as subtests with activation
        # checkpointing off and on.
        self.run_subtests(
            {
                "use_activation_checkpointing": [False, True],
            },
            self._test_train_shared_params,
        )

    def _test_train_shared_params(
        self,
        use_activation_checkpointing: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds a 3-layer weight-tied Transformer, replicates each
        # block (optionally with composable checkpointing) and the root, and trains 10 Adam
        # steps with fresh random token inputs against a reference with manually averaged
        # gradients, alternating set-to-none zeroing; asserts equal losses.
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if use_activation_checkpointing:
                    checkpoint(module)
                replicate(module)
        replicate(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(10):
            inp = torch.randint(
                0, model_args.vocab_size, (2, 16), device=device_type.type
            )
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(self.world_size)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()

            self.assertEqual(losses[0], losses[1])


class TestReplicateGradientAccumulation(FSDPTest):
    # [CLASS DOC] Multi-process test case (up to 4 ranks) for gradient accumulation across
    # microbatches with replicate, and 1F1B-style microbatching.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of 4 and the device count.
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        # [FUNCTION DOC] Builds a 1D replicate mesh over the world and runs the accumulation
        # check as subtests over reshard flags, which modules skip gradient sync (all, root
        # only, some MLPs), offload policy, and reduce-scatter-only mode.
        # KNOWN ISSUE: the mesh passed is 1D, but the helper returns early for any mesh whose
        # dimension count is not 2, so every subtest is skipped and nothing is verified.
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """

        replicate_size = self.world_size
        meshes = init_device_mesh(
            device_type.type,
            (replicate_size,),
            mesh_dim_names=("replicate",),
        )
        self.run_subtests(
            {
                "mesh": [meshes],
                "reshard_after_forward": [True, False],
                # "all": disable reduce-scatter for all modules
                # "root_only": disable reduce-scatter for root's linear only
                # "some_mlps": disable reduce-scatter for some MLPs
                "mode": ["all", "root_only", "some_mlps"],
                "reshard_after_backward": [False, True],
                "offload_policy": [OffloadPolicy(), CPUOffloadPolicy()],
                # For HSDP only:
                # `True`: reduce-scatter only (no all-reduce) each microbatch
                # until the last microbatch
                # `False`: neither reduce-scatter nor all-reduce each
                # microbatch until the last microbatch
                "reduce_scatter_only": [False, True],
            },
            self._test_gradient_accumulation,
        )

    def _test_gradient_accumulation(
        self,
        mesh: DeviceMesh,
        reshard_after_forward: bool | int,
        mode: str,
        reshard_after_backward: bool,
        offload_policy: OffloadPolicy,
        reduce_scatter_only: bool,  # for HSDP
    ):
        # [FUNCTION DOC] HELPER: Returns early for uncommon flag combinations and for any
        # mesh that is not 2D. Otherwise builds a linear layer plus 3 MLPs, replicates the
        # MLPs and root, and per iteration runs 3 microbatches, setting gradient sync or
        # all-reduce and reshard-after-backward flags per mode on non-last microbatches.
        # Counts collectives with a comm debug mode, asserts the all-reduce count matches the
        # expectation for the mode and that no all-gathers occur, then averages reference
        # gradients, checks sharded parity, steps both optimizers and zeroes grads.
        # KNOWN ISSUE: comments mention reduce-scatter and all-gather expectations that do
        # not match the asserted counts (all-gather is expected to be 0); reduce-scatter check
        # is commented out.
        if (
            (
                not reshard_after_backward
                and (reshard_after_forward is not False or mode == "some_mlps")
            )
            or (
                isinstance(offload_policy, CPUOffloadPolicy)
                and reshard_after_forward is not True
            )
            or (
                mesh.ndim != 2
            )  # may eventually need to change once decision on device mesh is made
        ):
            return  # skip since not common or applicable

        torch.manual_seed(42)
        batch_size, lin_dim, num_mlps, num_microbatches = (2, 32, 3, 3)
        if mode == "some_mlps":
            num_mlps_to_disable_reduce_scatter = 2
        modules = [nn.Linear(lin_dim, lin_dim)]
        modules.extend(MLP(lin_dim) for _ in range(num_mlps))
        model = nn.Sequential(*modules)
        ref_model = copy.deepcopy(model).to(device_type)
        replicate_fn = functools.partial(
            replicate,
            mesh=mesh,
            offload_policy=offload_policy,
        )
        for mlp in model[1:]:
            replicate_fn(mlp)
        replicate_fn(model)  # root gets the 1st linear
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        def set_grad_sync_flag(
            module: nn.Module, is_last_microbatch: bool, recurse: bool = True
        ):
            # [FUNCTION DOC] HELPER: Sets on the module either the all-reduce requirement (in
            # reduce-scatter-only mode) or the gradient-sync requirement to whether this is
            # the last microbatch, optionally recursing into children.
            if reduce_scatter_only:
                module.set_requires_all_reduce(is_last_microbatch, recurse=recurse)
            else:
                module.set_requires_gradient_sync(is_last_microbatch, recurse=recurse)

        def set_backward_flags(_model: nn.Module, is_last_microbatch: bool):
            # [FUNCTION DOC] HELPER: Per mode, sets gradient sync flags (and, if resharding
            # after backward is off, the reshard-after-backward flag) on the whole model, on
            # the first MLPs that skip reduction, or on the root only without recursion.
            if mode == "all":
                set_grad_sync_flag(_model, is_last_microbatch)
                if not reshard_after_backward:
                    _model.set_reshard_after_backward(is_last_microbatch)
            elif mode == "some_mlps":
                for mlp in model[1 : 1 + num_mlps_to_disable_reduce_scatter]:
                    set_grad_sync_flag(mlp, is_last_microbatch)
                    if not reshard_after_backward:
                        mlp.set_reshard_after_backward(is_last_microbatch)
            elif mode == "root_only":
                set_grad_sync_flag(model, is_last_microbatch, recurse=False)
                if not reshard_after_backward:
                    model.set_reshard_after_backward(is_last_microbatch, recurse=False)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            comm_count_list = []

            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                set_backward_flags(model, is_last_microbatch)
                inp = torch.randn(batch_size, lin_dim, device=device_type.type)
                losses: list[torch.Tensor] = []
                for _model in (ref_model, model):
                    with CommDebugMode() as comm_mode:
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                    comm_count_list.append(comm_mode.get_comm_counts())
                self.assertEqual(losses[0], losses[1])

            comm_counts = defaultdict(int)
            for comm_count_dict in comm_count_list:
                for collective, count in comm_count_dict.items():
                    comm_counts[collective] += count

            all_gather_count = comm_counts[c10d_ops._allgather_base_]
            # reduce_scatter_count = comm_counts[c10d_ops._reduce_scatter_base_]
            all_reduce_count = comm_counts[c10d_ops.allreduce_]

            # Expect one reduce-scatter per MLP plus one for the root's linear
            # on the last microbatch
            # expected_reduce_scatter_count = 0
            expected_all_reduce_count = num_mlps + 1

            if mode == "some_mlps":
                # Expect additional reduce-scatters for non-disabled MLPs and
                # the root's linear
                expected_all_reduce_count += (
                    num_mlps - num_mlps_to_disable_reduce_scatter + 1
                ) * (num_microbatches - 1)
            elif mode == "root_only":
                # Expect additional reduce-scatters for all MLPs
                expected_all_reduce_count += (num_mlps) * (num_microbatches - 1)

            # self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            self.assertEqual(all_reduce_count, expected_all_reduce_count)

            # Expect one all-gather per MLP plus one for the root's linear in
            # the first microbatch's forward
            expected_all_gather_count = 0

            self.assertEqual(all_gather_count, expected_all_gather_count)

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                # When `set_to_none=False`, we are exercising mixing
                # gradient accumulation with and without communication
                _optim.zero_grad(set_to_none=(iter_idx % 2))

    @skip_if_lt_x_gpu(2)
    def test_1f1b_microbatching(self):
        # [FUNCTION DOC] Runs the 1F1B microbatching check as subtests over explicit async
        # unshard on/off and reshard-after-backward on/off.
        self.run_subtests(
            {
                "use_explicit_unshard": [False, True],
                "reshard_after_backward": [False, True],
            },
            self._test_1f1b_microbatching,
        )

    def _test_1f1b_microbatching(
        self, use_explicit_unshard: bool, reshard_after_backward: bool
    ):
        # [FUNCTION DOC] HELPER: Replicates a Transformer's blocks and root, optionally
        # issues async unshards up front, then runs 3 microbatches (forward and backward
        # each) enabling gradient sync and last-backward only on the final one, and
        # resharding after backward only then if that flag is off. A reference runs the same
        # microbatches with averaged gradients; asserts equal losses, steps both AdamW
        # optimizers once and checks sharded parity.
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                replicate(module)
        replicate(model)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        num_microbatches = 3
        local_batch_size = 2
        torch.manual_seed(42 + self.rank + 1)
        inps = [
            torch.randint(
                0,
                model_args.vocab_size,
                (local_batch_size, 16),
                device=device_type.type,
            )
            for _ in range(num_microbatches)
        ]

        # Before pipelining, we may prefer to issue all all-gathers ahead of
        # time to increase overlap opportunity at no difference in parameter
        # memory usage since we do not reshard after forward
        if use_explicit_unshard:
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.unshard(async_op=True)

        # Emulate the 1f1b pipeline schedule and only reduce gradients on the
        # last microbatch
        losses: list[torch.Tensor] = []
        ref_losses: list[torch.Tensor] = []
        for inp_idx, inp in enumerate(inps):
            is_last_microbatch = inp_idx == num_microbatches - 1
            model.set_requires_gradient_sync(is_last_microbatch)
            model.set_is_last_backward(is_last_microbatch)
            if not reshard_after_backward:
                model.set_reshard_after_backward(is_last_microbatch)
            losses.append(model(inp).sum())
            losses[-1].backward()
            ref_losses.append(ref_model(inp).sum())
            ref_losses[-1].backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        for loss, ref_loss in zip(losses, ref_losses):
            self.assertEqual(loss, ref_loss)
        optim.step()
        ref_optim.step()
        check_sharded_parity(self, ref_model, model)


class TestReplicateCustomForwardMethod(FSDPTest):
    # [CLASS DOC] Multi-process test case (up to 2 ranks) for registering a non-forward
    # method as an entry point of a replicated module.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of the device count and 2.
        return min(torch.get_device_module(device_type).device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_register_fsdp_forward_method(self):
        # [FUNCTION DOC] Defines a vision-transformer-like module whose features method is
        # called directly by a parent model instead of forward, replicates the submodules and
        # root, and registers the features method as a forward method of the vit. Asserts the
        # loss equals a reference model's, runs both backward passes, averages reference
        # gradients and checks sharded parity.
        class VisionTransformer(nn.Module):
            # [CLASS DOC] HELPER: Patch-embedding convolution (3 to 1024 channels, 14x14
            # patches); forward_features returns the flattened patch tokens and forward sums
            # them over the token dimension.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the patch projection convolution.
                super().__init__()
                self.patch_proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

            def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the patch convolution to images, flattens
                # the spatial dims and returns a batch by tokens by channels tensor.
                return self.patch_proj(imgs).flatten(2).transpose(1, 2)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns forward_features summed over tokens.
                return self.forward_features(imgs).sum(dim=1)

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Wraps a VisionTransformer and a 1024 to 256 linear
            # projector; its forward bypasses the vit forward and calls forward_features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the vit and the projector.
                super().__init__()
                self.vit, self.projector = VisionTransformer(), nn.Linear(1024, 256)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Runs the vit features method on images and
                # projects the result.
                # Run `vit.forward_features`, which is not `forward`!
                patch_embeddings = self.vit.forward_features(imgs)
                return self.projector(patch_embeddings)

        torch.manual_seed(42)
        model = Model()
        ref_model = copy.deepcopy(model).to(device_type)
        replicate(model.vit)
        replicate(model.projector)
        replicate(model)
        register_fsdp_forward_method(model.vit, "forward_features")

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(4, 3, 224, 224, device=device_type.type)
        ref_loss = ref_model(inp).sum()
        loss = model(inp).sum()
        self.assertEqual(ref_loss, loss)
        ref_loss.backward()
        loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        check_sharded_parity(self, ref_model, model)


class TestReplicateTPTraining(FSDPTest):
    # [CLASS DOC] Multi-process test case for replicate combined with tensor parallelism on
    # a 2D (data-parallel replicate by tensor-parallel) device mesh.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the smaller of 4 and the device count.
        return min(4, torch.get_device_module(device_type).device_count())

    def init_global_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC] HELPER: Creates a 2x2 device mesh named dp_replicate and tp.
        return init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "tp"),
        )

    @skip_if_lt_x_gpu(8)
    def test_replicate_tp(self):
        # [FUNCTION DOC] Builds the 2x2 mesh and runs the TP parity check as subtests over
        # activation checkpointing on/off and MLP dimensions 3, 5, 16, 17, with foreach off.
        # KNOWN ISSUE: the test requires 8 GPUs but the class world size is capped at 4 and
        # the mesh needs only 4.
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 5, 16, 17],
                "foreach": [False],
            },
            functools.partial(self._test_replicate_tp, global_mesh),
        )

    def _test_replicate_tp(
        self,
        global_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        foreach: bool,
    ):
        # [FUNCTION DOC] HELPER: Builds an MLP stack and a reference copy, applies column and
        # row tensor-parallel plans on the tp mesh (keeping DTensor outputs for most layers),
        # then replicates each non-LayerNorm child (optionally checkpointed) and the root on
        # the dp mesh. Asserts parameters are DTensors whose full tensors equal the
        # reference. Trains 10 Adam steps with rank-dependent inputs, averaging reference
        # gradients, asserting equal losses and sharded parity, and finally asserts all
        # parameters are 2D-mesh DTensors with two placements and mesh dims named dp_replicate
        # and tp.
        dp_mesh, tp_mesh = global_mesh["dp_replicate"], global_mesh["tp"]
        dp_pg = dp_mesh._flatten().get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).to(device_type)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=foreach)

        parallelize_plan = {
            # Pass `use_local_output=False` to keep as DTensor to preserve
            # uneven activation dims
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": (RowwiseParallel()),
        }

        model = parallelize_module(model, tp_mesh, parallelize_plan)

        for module in model:
            if isinstance(module, nn.LayerNorm):
                continue
            if use_activation_checkpointing:
                checkpoint(module)
            replicate(module, mesh=dp_mesh)
        replicate(model, mesh=dp_mesh)

        # Checking parameters match orig model is critical to validate .full_tensor correctly replicates the
        # strided-sharded layers.
        for ref_p, p in zip(ref_model.parameters(), model.parameters()):
            self.assertIsInstance(p, DTensor)
            self.assertEqual(ref_p, p.full_tensor())

        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = device_type
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            for _optim in (ref_optim, optim):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                _optim.step()
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)

        for _, p in model.named_parameters():
            self.assertIsInstance(p, DTensor)
            self.assertEqual(p.device_mesh.ndim, 2)
            self.assertEqual(len(p.placements), 2)
            self.assertEqual(p.device_mesh.mesh_dim_names, ("dp_replicate", "tp"))


if __name__ == "__main__":
    run_tests()
