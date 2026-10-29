# Owner(s): ["oncall: distributed"]

# [MODULE DOC]
# Tests FSDP2 (fully_shard) with MixedPrecisionPolicy against reference models that emulate
# the policy by hand (a low precision compute copy plus a float32 master copy, with explicit
# gradient reduction). Covers param_dtype and reduce_dtype choices, reduce-scatter output
# dtype assertions via patched collectives, gradient accumulation without sync, structured
# (dataclass / NamedTuple) activations, jvp without warmup, input/output casting rules
# between nested FSDP modules, checkpoint recompute casting, norm layers, and reduce dtype
# clamping. Most tests need 1-4 GPUs; some require NCCL 2.10+ for bf16 collectives. Import
# time defines device_type from the available accelerator.

import copy
import dataclasses
import functools
import unittest
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_gradient_divide_factors,
)
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    DISTRIBUTED_BACKEND,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_assert,
)
from torch.testing._internal.common_utils import (
    MI300_ARCH,
    run_tests,
    skipIfRocmArch,
    skipIfRocmVersionLessThan,
    TEST_CUDA,
    TEST_XPU,
)
from torch.utils.checkpoint import checkpoint


device_type = torch.device(get_devtype())


class TestFullyShardMixedPrecisionTraining(FSDPTest):
    # [CLASS DOC] Multi-process test case (up to 4 ranks) for numerical parity of FSDP training
    # under mixed precision: shared setup builds an fp32 reference model and an FSDP model with
    # the policy.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property: rank count is the accelerator count capped at 4.
        return min(4, torch.get_device_module(device_type).device_count())

    def _init_models_and_optims(
        self,
        reshard_after_forward: bool | int,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
        use_shard_placement_fn,
    ):
        # [FUNCTION DOC] HELPER: Seeds, builds a three-MLP model, a deep-copied reference model on
        # the device with Adam, and shards the MLPs and root with the given reshard_after_forward
        # and a policy with the given param/reduce dtypes (optionally with a shard placement
        # function sharding on the largest dim). Returns (ref_model, ref_optim, model, fsdp_optim
        # with Adam foreach).
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        def _shard_placement_fn(param: nn.Parameter) -> Shard | None:
            # [FUNCTION DOC] HELPER: Returns a Shard on the largest parameter dimension (first on
            # ties); raises AssertionError if the parameter has no dimensions.
            largest_dim = -1
            largest_dim_size = -1
            for dim, dim_size in enumerate(param.shape):
                if dim_size > largest_dim_size:
                    largest_dim = dim
                    largest_dim_size = dim_size
            if not (largest_dim >= 0):
                raise AssertionError(
                    f"Expected largest_dim >= 0, got {largest_dim} for {param.shape}"
                )
            return Shard(largest_dim)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        shard_placement_fn = _shard_placement_fn if use_shard_placement_fn else None
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            shard_placement_fn=shard_placement_fn,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        return ref_model, ref_optim, model, optim

    def _get_use_shard_placement_fn_vals_for_bf16_reduce(self):
        # [FUNCTION DOC] HELPER: Returns the list of shard placement function settings to test with
        # bf16 reduction: only False, plus True when world size is 2 (larger groups reduce in a
        # different order and give numeric differences).
        use_shard_placement_fn_vals = [False]
        if self.world_size == 2:
            # For world size >2, gradient elements get reduced in different
            # orders for the baseline vs. dim-1 sharding, leading to numeric
            # differences for bf16 reduction, so only test world size 2.
            use_shard_placement_fn_vals.append(True)
        return use_shard_placement_fn_vals

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(DISTRIBUTED_BACKEND != "nccl", "Requires NCCL backend")
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        # [FUNCTION DOC] Sweeps bf16/fp16 param dtype, reshard_after_forward (false, true, 2) and
        # shard placement use, running the compute dtype check. Requires NCCL backend, NCCL 2.10+,
        # ROCm 7+ if ROCm, and at least 2 GPUs.
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "param_dtype": [torch.bfloat16, torch.float16],
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": use_shard_placement_fn_vals,
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(
        self,
        param_dtype: torch.dtype,
        reshard_after_forward: bool | int,
        use_shard_placement_fn: bool,
    ):
        # [FUNCTION DOC] HELPER: Trains 10 iterations of FSDP with only param_dtype set (reduce
        # dtype follows it) and asserts the reduce-scatter output has that dtype via a patched
        # collective. A reference low precision model computes gradients, reduces them via reduce-
        # scatter then all-gather with matching pre/post divide factors, and updates an fp32 master
        # model. Each step asserts equal losses and sharded parameter/gradient parity.
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=None,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has the compute
            # param dtype.
            self.assertEqual(output.dtype, param_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        predivide_factor, postdivide_factor, _, _ = _get_gradient_divide_factors(
            self.process_group, all_reduce_group=None, reduce_dtype=param_dtype
        )

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_bf16.parameters():
                # Use reduce-scatter -> all-gather as all-reduce because for
                # world size >=4, NCCL all-reduce shows numeric differences
                # compared with NCCL reduce-scatter
                if predivide_factor is not None and predivide_factor > 1:
                    param.grad.div_(predivide_factor)
                elif predivide_factor is None:
                    param.grad.div_(self.world_size)
                output = torch.zeros_like(torch.chunk(param.grad, self.world_size)[0])
                dist.reduce_scatter_single(output, param.grad)
                dist.all_gather_single(param.grad, output)
                if postdivide_factor is not None and postdivide_factor > 1:
                    param.grad.div_(postdivide_factor)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad.to(param_fp32.dtype)
                param_bf16.grad = None
            ref_optim.step()  # fp32 optimizer step
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            self.assertEqual(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(DISTRIBUTED_BACKEND != "nccl", "Requires NCCL backend")
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        # [FUNCTION DOC] Runs the fp32-reduce check for each reshard setting and shard placement
        # option, then the bf16-reduce check with the placement options valid for bf16. Needs NCCL
        # 2.10+, 2+ GPUs.
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": [False, True],
            },
            self._test_reduce_dtype_fp32_reduce,
        )
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": use_shard_placement_fn_vals,
            },
            self._test_reduce_dtype_bf16_reduce,
        )

    def _test_reduce_dtype_fp32_reduce(
        self, reshard_after_forward: bool | int, use_shard_placement_fn: bool
    ):
        # [FUNCTION DOC] HELPER: Trains 10 iterations with bf16 params and fp32 reduction; asserts
        # the reduce-scatter output is fp32. The reference does a bf16 backward, upcasts grads,
        # all-reduces in fp32, divides by world size and steps an fp32 master copy; asserts loss
        # and sharded parity. Returns early for world size above 2 with an integer reshard factor
        # and placement fn.
        if (
            self.world_size > 2
            and isinstance(reshard_after_forward, int)
            and use_shard_placement_fn
        ):
            return
        param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has the reduce
            # dtype (fp32).
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_bf16.parameters():
                param.grad.data = param.grad.to(torch.float32)
                dist.all_reduce(param.grad)  # fp32 reduction
                param.grad.div_(self.world_size)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad
                param_bf16.grad = None
            ref_optim.step()  # fp32 optimizer step
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            self.assertEqual(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    def test_reduce_dtype_after_frozen_first_forward(self):
        # [FUNCTION DOC] Shards a model with an int64 non-grad parameter under bf16 param / fp32
        # reduce policy, freezes everything and runs a no-grad forward, checks the param group's
        # original and reduce dtypes are float32, then unfreezes the layers and checks reduce-
        # scatter output and resulting gradients are float32.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Small model: two linear layers and a frozen int64 parameter;
            # forward applies the layers.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the layers and the non-float, non-grad parameter.
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
                self.non_float = nn.Parameter(
                    torch.ones(16, dtype=torch.int64), requires_grad=False
                )

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sequential layers applied to the input.
                return self.layers(inp)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        model = Model().to(device_type)
        fully_shard(model, mp_policy=mp_policy)
        model.requires_grad_(False)

        inp = torch.randn((4, 16), device=device_type.type)
        with torch.no_grad():
            model(inp)

        fsdp_param_group = fully_shard.state(model)._fsdp_param_group
        if fsdp_param_group is None:
            raise AssertionError("Expected root FSDP parameter group")
        self.assertEqual(fsdp_param_group._orig_dtype, torch.float32)
        self.assertEqual(fsdp_param_group._reduce_dtype, torch.float32)

        model.layers.requires_grad_(True)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor is float32.
            self.assertEqual(output.dtype, torch.float32)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        with patch_reduce_scatter(reduce_scatter):
            model(inp).sum().backward()
        for param in model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float32)

    def _test_reduce_dtype_bf16_reduce(
        self, reshard_after_forward: bool | int, use_shard_placement_fn: bool
    ):
        # [FUNCTION DOC] HELPER: Trains 10 iterations with fp32 params and bf16 reduction;
        # asserts the reduce-scatter output is bf16. The reference reduces gradients in bf16
        # using reduce-scatter then all-gather (average), upcasts to fp32 and steps; asserts loss
        # and sharded parity.
        param_dtype, reduce_dtype = torch.float32, torch.bfloat16
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        group = dist.distributed_c10d._get_default_group()
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has the reduce
            # dtype (bf16).
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                param_grad = param.grad.to(reduce_dtype)
                # Use reduce-scatter -> all-gather to implement all-reduce
                # since for world size >2, bf16 all-reduce and reduce-scatter
                # have numeric differences
                sharded_grad = funcol.reduce_scatter_single(
                    param_grad, scatter_dim=0, reduceOp="avg", group=group
                )  # bf16 reduction
                param.grad = funcol.all_gather_single(
                    sharded_grad, gather_dim=0, group=group
                ).to(param.dtype)  # upcast to fp32
            ref_optim.step()  # fp32 optimizer step

            self.assertEqual(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    def test_grad_acc_with_reduce_dtype(self):
        # [FUNCTION DOC] Checks that gradient accumulation without reduce-scatter, under bf16
        # compute with fp32 reduce, accumulates unsharded grads in fp32, for reshard_after_forward
        # true and false.
        """
        Tests that gradient accumulation without reduce-scatter when using
        bf16 compute and fp32 reduction accumulates the unsharded gradients in
        fp32.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False]},
            self._test_grad_acc_with_reduce_dtype,
        )

    def _test_grad_acc_with_reduce_dtype(self, reshard_after_forward: bool):
        # [FUNCTION DOC] HELPER: Trains 10 iterations of 4 microbatches, syncing gradients and
        # resharding only on the last (or always resharding if reshard_after_forward). Compares
        # losses against a bf16 compute reference whose grads are manually accumulated in fp32 and
        # all-reduced on the last microbatch; asserts loss dtype bf16, reduce-scatter dtype fp32,
        # and parity.
        torch.manual_seed(42)
        param_dtype, reduce_dtype = (torch.bfloat16, torch.float32)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        # To emulate the mixed precision implementation where forward/backward
        # compute use bf16 and optimizer uses fp32, we maintain both an fp32
        # and a bf16 copy of the reference model
        ref_model = copy.deepcopy(model).to(device_type)
        ref_model_compute = copy.deepcopy(ref_model).to(param_dtype)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(
                mlp, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy
            )
        fully_shard(
            model, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has the reduce
            # dtype (fp32).
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        device = device_type
        # Train on the same input to avoid loss explosion
        num_microbatches = 4
        inp = torch.randn((2 * num_microbatches, 16), device=device, dtype=param_dtype)
        for iter_idx in range(10):
            microbatch_inps = torch.chunk(inp, 4)
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                model.set_requires_gradient_sync(is_last_microbatch)
                model.set_reshard_after_backward(
                    is_last_microbatch or reshard_after_forward
                )
                losses: list[torch.Tensor] = []
                for _model in (ref_model_compute, model):
                    losses.append(
                        _model(microbatch_inps[microbatch_idx].detach()).sum()
                    )
                    self.assertEqual(losses[-1].dtype, param_dtype)
                    with patch_reduce_scatter(reduce_scatter):
                        losses[-1].backward()
                self.assertEqual(losses[0], losses[1])
                # Manually accumulate gradients into the base reference model
                # from the compute reference model in fp32
                for ref_param, ref_param_compute in zip(
                    ref_model.parameters(), ref_model_compute.parameters()
                ):
                    self.assertTrue(ref_param_compute.grad is not None)
                    self.assertEqual(ref_param.dtype, torch.float32)
                    if ref_param.grad is not None:
                        ref_param.grad += ref_param_compute.grad
                    else:
                        ref_param.grad = ref_param_compute.grad.to(ref_param.dtype)
                    ref_param_compute.grad = None
                # Manually reduce gradients for the reference model on the last
                # microbatch to implement data parallelism
                if is_last_microbatch:
                    for ref_param in ref_model.parameters():
                        self.assertTrue(ref_param.grad is not None)
                        dist.all_reduce(ref_param.grad)
                        ref_param.grad /= self.world_size
            check_sharded_parity(self, ref_model, model)
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # Manually copy parameters from the base reference model to the
            # compute reference model to run the optimizer step for the latter
            for ref_param, ref_param_compute in zip(
                ref_model.parameters(), ref_model_compute.parameters()
            ):
                ref_param_compute.detach().copy_(ref_param)

    def _reduce_1d_partial_grads(
        self, module: nn.Module, group: dist.ProcessGroup | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Divides each present gradient of the module by the group size
        # (default group if none), emulating the averaging the FSDP all-reduce does when the
        # reference saw the global batch.
        # The reference model is replicated and sees the full global batch, so
        # its gradients are already the global sum. Dividing by world_size
        # simulates the average that FSDP's all-reduce produces.
        group = group or dist.distributed_c10d._get_default_group()
        for param in module.parameters():
            if param.grad is not None:
                param.grad.div_(group.size())

    @skip_if_lt_x_gpu(2)
    def test_structured_input_output(self):
        # [FUNCTION DOC] Sweeps container types (dataclass, nested dataclass, NamedTuple) and three
        # mixed precision configs (none, bf16/bf16, bf16 compute with fp32 reduce) for structured
        # activations passed between FSDP modules. Needs 2+ GPUs.
        """
        Tests numeric parity between FSDP and a reference model when using
        structured (dataclass, nested dataclass, NamedTuple) activations
        between FSDP-wrapped modules, with and without mixed precision.
        """
        self.run_subtests(
            {
                "container_type": ["dataclass", "nested_dataclass", "namedtuple"],
                "mp_config": [
                    (None, None),  # fp32
                    (torch.bfloat16, torch.bfloat16),  # bf16 compute + reduce
                    (torch.bfloat16, torch.float32),  # bf16 compute, fp32 reduce
                ],
            },
            self._test_structured_input_output,
        )

    def _test_structured_input_output(
        self,
        container_type: str,
        mp_config: tuple[torch.dtype | None, torch.dtype | None],
    ):
        # [FUNCTION DOC] HELPER: Builds a model that converts a tensor to the container type,
        # applies two linear layers per field, and sums the fields back; shards each module and the
        # root with the policy, and trains 10 iterations on per-rank slices of a global batch
        # against a reference using the global batch (bf16 compute copy and fp32 master when param
        # dtype is set, with matching gradient reduction). Asserts loss equality, and full sharded
        # parity only in fp32.
        param_dtype, reduce_dtype = mp_config

        @dataclasses.dataclass
        class DC:
            # [CLASS DOC] HELPER: Two-tensor dataclass container.
            x: torch.Tensor
            y: torch.Tensor

        @dataclasses.dataclass
        class Inner:
            # [CLASS DOC] HELPER: Two-tensor dataclass used as the inner part of the
            # nested container.
            x: torch.Tensor
            y: torch.Tensor

        @dataclasses.dataclass
        class Outer:
            # [CLASS DOC] HELPER: Dataclass holding an Inner container and an extra tensor.
            inner: Inner
            z: torch.Tensor

        class TensorPair(NamedTuple):
            # [CLASS DOC] HELPER: Two-tensor named tuple container.
            x: torch.Tensor
            y: torch.Tensor

        class ContainerModule(nn.Module):
            # [CLASS DOC] HELPER: Applies one shared linear layer to every tensor in the container
            # type chosen by name.
            """Applies a linear layer to each tensor in the container."""

            def __init__(self, ctype: str, dim: int):
                # [FUNCTION DOC] HELPER: Stores the container type name and creates the
                # linear layer.
                super().__init__()
                self.ctype = ctype
                self._layer = nn.Linear(dim, dim)

            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns a container of the same type with the layer
                # applied to each tensor field.
                if self.ctype == "dataclass":
                    return DC(x=self._layer(inp.x), y=self._layer(inp.y))
                elif self.ctype == "nested_dataclass":
                    return Outer(
                        inner=Inner(
                            x=self._layer(inp.inner.x),
                            y=self._layer(inp.inner.y),
                        ),
                        z=self._layer(inp.z),
                    )
                else:  # namedtuple
                    return TensorPair(x=self._layer(inp.x), y=self._layer(inp.y))

        class ToContainer(nn.Module):
            # [CLASS DOC] HELPER: Module turning a plain tensor into the chosen container type.
            """Converts a plain tensor into a structured container."""

            def __init__(self, ctype: str):
                # [FUNCTION DOC] HELPER: Stores the container type name.
                super().__init__()
                self.ctype = ctype

            def forward(self, x: torch.Tensor) -> Any:
                # [FUNCTION DOC] HELPER: Wraps the input in the container; extra fields are clones
                # so each has an independent autograd path.
                # clone() so each field has an independent autograd path
                if self.ctype == "dataclass":
                    return DC(x=x, y=x.clone())
                elif self.ctype == "nested_dataclass":
                    return Outer(inner=Inner(x=x, y=x.clone()), z=x.clone())
                else:  # namedtuple
                    return TensorPair(x=x, y=x.clone())

        class FromContainer(nn.Module):
            # [CLASS DOC] HELPER: Module reducing a container back to a tensor by summing
            # its fields.
            """Extracts and sums tensors from a structured container."""

            def __init__(self, ctype: str):
                # [FUNCTION DOC] HELPER: Stores the container type name.
                super().__init__()
                self.ctype = ctype

            def forward(self, inp: Any) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of the container fields (three fields for
                # the nested type, else two).
                if self.ctype == "nested_dataclass":
                    return inp.inner.x + inp.inner.y + inp.z
                else:
                    return inp.x + inp.y

        torch.manual_seed(42)
        dim = 16
        local_batch_size = 2
        global_batch_size = self.world_size * local_batch_size

        model = nn.Sequential(
            ToContainer(container_type),
            ContainerModule(container_type, dim),
            ContainerModule(container_type, dim),
            FromContainer(container_type),
        )
        ref_model = copy.deepcopy(model).to(device_type)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        for module in model:
            fully_shard(module, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        if param_dtype is not None:
            # Maintain a bf16 copy for compute, fp32 copy for optimizer
            ref_model_compute = copy.deepcopy(ref_model).to(param_dtype)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(10):
            global_inp = torch.rand((global_batch_size, dim), device=device_type.type)
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()

            # FSDP forward/backward
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(local_inp).sum()
            fsdp_loss.backward()
            optim.step()

            # Ref forward/backward
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            if param_dtype is not None:
                ref_loss = ref_model_compute(global_inp.to(param_dtype)).sum()
                ref_loss.backward()
                # Simulate gradient reduction matching FSDP's behavior
                if reduce_dtype is not None and reduce_dtype != param_dtype:
                    # Cast grads to reduce_dtype, all-reduce, divide
                    for p in ref_model_compute.parameters():
                        p.grad.data = p.grad.to(reduce_dtype)
                        dist.all_reduce(p.grad)
                        p.grad.div_(self.world_size)
                else:
                    self._reduce_1d_partial_grads(ref_model_compute)
                for p_fp32, p_compute in zip(
                    ref_model.parameters(), ref_model_compute.parameters()
                ):
                    p_fp32.grad = p_compute.grad.to(p_fp32.dtype)
                    p_compute.grad = None
                ref_optim.step()
                for p_fp32, p_compute in zip(
                    ref_model.parameters(), ref_model_compute.parameters()
                ):
                    p_compute.detach().copy_(p_fp32)
            else:
                ref_loss = ref_model(global_inp).sum()
                ref_loss.backward()
                self._reduce_1d_partial_grads(ref_model)
                ref_optim.step()

            dist.all_reduce(fsdp_loss)  # partial -> replicated
            self.assertEqual(fsdp_loss, ref_loss)
            # bf16 gradient accumulation introduces drift beyond default
            # fp32 tolerances, so only check full param/grad parity for fp32.
            if param_dtype is None:
                check_sharded_parity(self, ref_model, model)


class TestFullyShardMixedPrecisionJVP(FSDPTest):
    # [CLASS DOC] Two-rank test case checking forward-mode autodiff (jvp) through FSDP modules with
    # mixed precision and no warmup forward.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property: always two ranks.
        return 2

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_no_warmup_jvp_reshard_after_forward_false(self):
        # [FUNCTION DOC] Shards two meta-device linears over a mesh with a bf16 policy and
        # reshard_after_forward false, materializes them, runs the first on its own, then runs jvp
        # of a function calling the second on the saved activation. Asserts the output shape and
        # dtype and that the tangent output is zero. Needs 2+ GPUs and NCCL 2.10+.
        dtype = torch.bfloat16
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )

        with torch.device("meta"):
            enc = nn.Linear(128, 256, bias=False)
            dec = nn.Linear(256, 320, bias=False)

        for module in (enc, dec):
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=False,
            )

        gen = torch.Generator(device_type.type)
        for idx, module in enumerate((enc, dec)):
            module.to_empty(device=device_type.type)
            with torch.no_grad():
                module.weight.normal_(
                    std=module.in_features**-0.5,
                    generator=gen.manual_seed(42 + idx),
                )

        x = torch.randn(
            (128,),
            device=device_type.type,
            dtype=dtype,
            generator=gen.manual_seed(0),
        )
        primal = torch.randn(
            (320,),
            device=device_type.type,
            dtype=dtype,
            generator=gen.manual_seed(1),
            requires_grad=True,
        )
        tangent = torch.randn(
            (320,),
            device=device_type.type,
            dtype=dtype,
            generator=gen.manual_seed(2),
        )
        e = enc(x)

        def fn(unused: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the second linear applied to
            # the saved activation.
            return dec(e)

        output, tangent_output = torch.func.jvp(fn, (primal,), (tangent,))

        self.assertEqual(output.shape, (320,))
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(tangent_output, torch.zeros_like(output))


class TestFullyShardMixedPrecisionCasts(FSDPTestMultiThread):
    # [CLASS DOC] Two-thread test case for how MixedPrecisionPolicy casts forward inputs, outputs
    # and buffers between nested FSDP modules, including norm layers and reduce dtype handling.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property: always two ranks.
        return 2

    @skip_if_lt_x_gpu(1)
    def test_float16_on_one_submodule(self):
        # [FUNCTION DOC] Checks recorded forward input dtypes in three setups: fp16 on one child
        # with automatic input casting, fp16 on a child where the user module owns the cast (no
        # policy casting), and fp16 on the first child with output_dtype float32 so the second
        # needs no casting. Needs 1+ GPU.
        x = torch.zeros(2, 100, device=device_type)

        # Subtest 1: use fp16 on the second child submodule -- does not require
        # any additional casting logic
        forward_inputs: dict[str, nn.Module] = {}
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).to(device_type)
        fully_shard(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

        # Subtest 2: use fp16 on the second child module, where the user module
        # owns the cast
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).to(device_type)
        fully_shard(
            model.c2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=False
            ),
        )
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

        # Subtest 3: use fp16 on the first child module and specify its output
        # dtype so that the second child module does not need to cast
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).to(device_type)
        fully_shard(
            model.c1,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, output_dtype=torch.float32
            ),
        )
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float16)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(1)
    def test_submodules_with_external_inputs(self):
        # [FUNCTION DOC] Runs the external inputs check with submodule input casting disabled
        # and enabled.
        self.run_subtests(
            {"enable_submodule_cast": [False, True]},
            self._test_submodules_with_external_inputs,
        )

    def _test_submodules_with_external_inputs(self, enable_submodule_cast: bool):
        # [FUNCTION DOC] HELPER: Builds a model whose submodule takes the activation plus an
        # external fp32 tensor. With an fp16 policy on both, asserts the model and submodule inputs
        # are fp16 and the external input is fp16 only when the submodule policy casts forward
        # inputs, else fp32.
        class ToyModule(nn.Module):
            # [CLASS DOC] HELPER: Linear layer wrapper that records its two inputs in a
            # shared dict.
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer and keeps the shared record dict.
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records both inputs under fixed keys and returns the
                # linear applied to the first.
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Model with a linear layer then the submodule, recording its
            # own input.
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates a linear layer and the submodule, keeping the
                # shared record dict.
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records the input, creates an external fp32 ones tensor,
                # and returns the submodule applied to the first layer output and that tensor.
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(
                    2, 100, device=device_type.type, dtype=torch.float32
                )  # external input
                return self.l2(self.l1(x), y)

        forward_inputs: dict[str, torch.Tensor] = {}
        model = ToyModel(forward_inputs).to(device_type)
        x = torch.zeros(2, 100, device=device_type.type, dtype=torch.float32)
        fully_shard(
            model.l2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=enable_submodule_cast
            ),
        )
        fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        model(x).sum().backward()

        # If we enable `model.l2` to cast (as default), then `l2_input_y` gets
        # cast to fp16, and if we disable, then it says as fp32.
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        self.assertEqual(
            forward_inputs["l2_input_y"].dtype,
            torch.float16 if enable_submodule_cast else torch.float32,
        )

    @skip_if_lt_x_gpu(1)
    @skipIfRocmArch(MI300_ARCH)  # https://github.com/pytorch/pytorch/issues/182988
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_checkpoint_recompute_casts_forward_inputs(self):
        # [FUNCTION DOC] Runs the checkpoint recompute check where the checkpoint input is fp32 and
        # casting happens on inputs. Skipped on MI300; needs NCCL 2.10+.
        self._test_checkpoint_recompute_casts(
            recompute_cast="input",
        )

    @skip_if_lt_x_gpu(1)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_checkpoint_recompute_casts_outputs(self):
        # [FUNCTION DOC] Runs the checkpoint recompute check where the policy casts outputs back to
        # fp32. Needs NCCL 2.10+.
        self._test_checkpoint_recompute_casts(
            recompute_cast="output",
        )

    def _test_checkpoint_recompute_casts(self, recompute_cast: str):
        # [FUNCTION DOC] HELPER: Shards a model whose checkpointed region calls a recording linear
        # under a bf16 policy. In the input mode, the region input is fp32 and the linear must see
        # bf16 in both the forward and the recompute. In the output mode, the policy has
        # output_dtype fp32 and input casting off, and recorded outputs must be fp32 in both.
        # Raises AssertionError on an unknown mode.
        class RecordingLinear(nn.Linear):
            # [CLASS DOC] HELPER: Linear layer (1 to 1) that records the dtype of every input
            # it receives.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 1 to 1 linear and the empty dtype list.
                super().__init__(1, 1)
                self.input_dtypes: list[torch.dtype] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records the input dtype and returns the parent
                # linear output.
                self.input_dtypes.append(x.dtype)
                return super().forward(x)

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Model casting its input to a configured dtype and running a
            # checkpointed linear.
            def __init__(self, checkpoint_input_dtype: torch.dtype) -> None:
                # [FUNCTION DOC] HELPER: Creates the recording linear and stores the checkpoint
                # input dtype and output dtype list.
                super().__init__()
                self.linear1 = RecordingLinear()
                self.checkpoint_input_dtype = checkpoint_input_dtype
                self.linear1_output_dtypes: list[torch.dtype] = []

            def checkpointed_linear1(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the linear, records the output dtype, and returns
                # the output.
                x = self.linear1(x)
                self.linear1_output_dtypes.append(x.dtype)
                return x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Casts the input, runs the linear under non-reentrant
                # checkpoint without early stop (so recompute reruns it fully), and returns the
                # float sum.
                x = x.to(dtype=self.checkpoint_input_dtype)
                x = checkpoint(
                    self.checkpointed_linear1,
                    x,
                    use_reentrant=False,
                    early_stop=False,
                )
                return x.float().sum()

        if recompute_cast == "input":
            checkpoint_input_dtype = torch.float32
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
        elif recompute_cast == "output":
            checkpoint_input_dtype = torch.bfloat16
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.float32,
                cast_forward_inputs=False,
            )
        else:
            raise AssertionError(f"Unknown recompute_cast: {recompute_cast}")

        torch.manual_seed(42)
        model = Model(checkpoint_input_dtype).to(device_type)
        fully_shard(model.linear1, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        inp = torch.randn(8, 1, device=device_type.type)
        model(inp).backward()
        if recompute_cast == "input":
            self.assertEqual(
                model.linear1.input_dtypes,
                [torch.bfloat16, torch.bfloat16],
            )
        else:
            self.assertEqual(
                model.linear1_output_dtypes,
                [torch.float32, torch.float32],
            )

    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(DISTRIBUTED_BACKEND != "nccl", "Requires NCCL backend")
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_norm_modules_bf16(self):
        # [FUNCTION DOC] Runs the norm module check with a bf16 policy. Needs NCCL backend and
        # 1+ GPU.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._test_norm_modules(mp_policy)

    @skip_if_lt_x_gpu(1)
    @skipIfRocmArch(MI300_ARCH)  # https://github.com/pytorch/pytorch/issues/182988
    def test_norm_modules_fp16(self):
        # [FUNCTION DOC] Runs the norm module check with an fp16 policy. Skipped on MI300.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self._test_norm_modules(mp_policy)

    def _test_norm_modules(self, mp_policy: MixedPrecisionPolicy):
        # [FUNCTION DOC] HELPER: Shards models with layer norm, batch norm 1D and batch norm 2D
        # under the policy and runs forward/backward without type mismatch. For 2D batch norm on
        # CUDA/XPU the default raises a RuntimeError about running_mean type; other cases (buffers
        # cast down, or a batch norm module with only an output dtype policy) must succeed.
        def inner(model: nn.Module, x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs forward, asserts output dtype equals the policy param
            # dtype, and runs backward.
            # Run forward and backward to check for no type mismatch errors
            z = model(x)
            self.assertEqual(z.dtype, mp_policy.param_dtype)
            z.sum().backward()

        # Layer norm
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 1D
        model = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 2D: CUDA and XPU raise a dtype mismatch error in backward
        # from buffer dtype mismatch; other hardware types do not. Preserve the
        # original semantics for CUDA/XPU and default all other devices to the
        # no-error path.
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        if TEST_CUDA or TEST_XPU:
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected running_mean to have type",
            ):
                inner(model, torch.randn((3, 1, 9, 9)))
        else:
            inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: cast buffers down to lower precision
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # Casting batch norm buffers to the lower precision allows backward
        model[1].running_mean = model[1].running_mean.to(mp_policy.param_dtype)
        model[1].running_var = model[1].running_var.to(mp_policy.param_dtype)
        inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: use special mixed precision policy
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        bn_mp_policy = MixedPrecisionPolicy(output_dtype=mp_policy.param_dtype)
        fully_shard(model[1], mp_policy=bn_mp_policy)
        for module in (model[0], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((3, 1, 9, 9)))

    @skip_if_lt_x_gpu(1)
    def test_clamp_reduce_dtype(self):
        # [FUNCTION DOC] Initializes a bf16 model, shards it with bf16 param and reduce dtypes,
        # then converts the model to float32 and checks the policy reduce dtype remains bf16 and
        # reduce-scatter output is bf16 in a forward/backward. Needs 1+ GPU.
        # Initialize the model directly in bf16
        init_dtype = torch.bfloat16
        model = nn.Sequential(
            nn.Linear(32, 32, dtype=init_dtype),
            nn.Linear(32, 32, dtype=init_dtype),
        ).to(device_type.type)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        # Check that we did not clamp the reduce dtype
        self.assertEqual(mp_policy.reduce_dtype, torch.bfloat16)
        for module in model:
            fully_shard((module), mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        # Check that the reduce-scatter runs in bf16 even after we change the
        # model from bf16 to fp32
        model.to(torch.float32)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor is bf16.
            self.assertEqual(output.dtype, torch.bfloat16)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        with patch_reduce_scatter(reduce_scatter):
            inp = torch.randn((4, 32), device=device_type.type)
            loss = model(inp).sum()
            loss.backward()


if __name__ == "__main__":
    run_tests()
