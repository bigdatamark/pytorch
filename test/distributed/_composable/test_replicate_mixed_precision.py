# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Tests for mixed precision (MixedPrecisionPolicy) with the FSDP-style replicate()
# from torch.distributed._composable.replicate_with_fsdp. Training-parity tests compare
# replicated models against reference models that emulate bf16/fp16 compute with fp32 optimizer
# steps and manually reduced gradients, checking param/reduce dtypes, jvp/vmap inputs and
# gradient accumulation. A multi-thread class checks forward-input casting, norm-module
# behavior, reduce dtype clamping and dataclass inputs. Import-time side effect: sets
# module-level device_type from get_devtype.

import copy
import dataclasses
import functools

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_gradient_divide_factors,
)
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTestContinuous,
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
    TEST_HPU,
)


device_type = torch.device(get_devtype())


class TestReplicateMixedPrecisionTraining(FSDPTestContinuous):
    # [CLASS DOC] Two-rank continuous multi-process tests of replicate() under mixed precision
    # training. Each test builds a reference model (plain Adam, manual casting and all-reduce)
    # and a replicated model, trains both and asserts losses are equal and parameters match
    # (check_sharded_parity). Reduce-scatter dtypes are verified through a patched
    # reduce_scatter_single.
    world_size = 2

    def _init_models_and_optims(
        self,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ):
        # [FUNCTION DOC] HELPER: Seeds 42 and builds a three-layer MLP stack. Returns
        # (ref_model, ref_optim, model, optim): ref_model is a device copy trained with Adam (lr
        # 0.01); model has replicate applied to each MLP and then to the whole stack with a
        # MixedPrecisionPolicy of the given param_dtype and reduce_dtype, with its own Adam
        # (foreach).
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        replicate_fn = functools.partial(
            replicate,
            mp_policy=mp_policy,
        )
        for mlp in model:
            replicate_fn(mlp)
        replicate_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        return ref_model, ref_optim, model, optim

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        # [FUNCTION DOC] Runs the compute-dtype check for bfloat16 and float16 param dtypes
        # through run_subtests. Needs 2 GPUs and NCCL 2.10+, and ROCm 7.0+ on ROCm.
        self.run_subtests(
            {
                "param_dtype": [torch.bfloat16, torch.float16],
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(
        self,
        param_dtype: torch.dtype,
    ):
        # [FUNCTION DOC] HELPER: For the given param_dtype (no reduce dtype), trains the
        # replicated model and a reference copy cast to that dtype for 10 iterations. Asserts
        # the reduce-scatter output dtype equals param_dtype. The reference grads are divided by
        # the gradient divide factors and reduced via reduce-scatter plus all-gather, then moved
        # to an fp32 model for the optimizer step and copied back. Asserts equal loss and
        # sharded parity each iteration.
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            param_dtype=param_dtype,
            reduce_dtype=None,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output tensor has dtype
            # param_dtype.
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

    @skip_if_lt_x_gpu(2)
    def test_input_jvp(self):
        # [FUNCTION DOC] Trains replicated models with cast_forward_inputs and bf16 params
        # through vmap and torch.func.jvp. Compares vmap outputs with a bf16 reference for a
        # three-input MLP and for one returning a (tensor, scalar) pair, checks loss and one
        # step, then runs 5 iterations of a jvp-based loss (including the tangent) with input
        # grads required, comparing outputs, tangents, losses and parameters against the
        # reference. Needs 2 GPUs.
        class ThreeInputMLP(nn.Module):
            # [CLASS DOC] HELPER: MLP taking three inputs (z, t, r); concatenates them, then
            # applies in_proj, relu and out_proj.
            def __init__(self, dim: int) -> None:
                # [FUNCTION DOC] HELPER: Creates in_proj (dim+2 to dim) and out_proj (dim to
                # dim).
                super().__init__()
                self.in_proj = nn.Linear(dim + 2, dim)
                self.out_proj = nn.Linear(dim, dim)

            def forward(
                self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Concatenates z, t, r on the last dim and returns
                # out_proj of relu of in_proj.
                x = torch.cat((z, t, r), dim=-1)
                return self.out_proj(torch.relu(self.in_proj(x)))

        class MixedOutputMLP(ThreeInputMLP):
            # [CLASS DOC] HELPER: ThreeInputMLP variant whose forward also returns the sum of
            # the out_proj weight as a second output.
            def forward(
                self, z: torch.Tensor, t: torch.Tensor, r: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns the parent forward output and the out_proj
                # weight sum.
                return super().forward(z, t, r), self.out_proj.weight.sum()

        torch.manual_seed(42)
        dim = 16
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=True,
        )
        mesh = init_device_mesh(
            device_type.type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("replicate",),
        )
        replicate_fn = functools.partial(replicate, mesh=mesh, mp_policy=mp_policy)

        def init_model(model_cls):
            # [FUNCTION DOC] HELPER: Builds model_cls(dim), an fp32 device reference with Adam
            # (lr 0.001) and a bf16 copy of it, then applies replicate to in_proj, out_proj and
            # the whole model with the test's policy (via mesh). Returns model, optim,
            # ref_model, ref_optim, ref_model_bf16.
            model = model_cls(dim)
            ref_model = copy.deepcopy(model).to(device_type.type)
            ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-3)
            ref_model_bf16 = copy.deepcopy(ref_model).to(torch.bfloat16)
            replicate_fn(model.in_proj)
            replicate_fn(model.out_proj)
            replicate_fn(model)
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            return model, optim, ref_model, ref_optim, ref_model_bf16

        def reduce_ref_grads_and_step(ref_model, ref_model_bf16, ref_optim) -> None:
            # [FUNCTION DOC] HELPER: Upcasts the bf16 reference grads to fp32, all-reduces them,
            # averages by world size, assigns them to the fp32 reference, steps its optimizer,
            # and copies the updated values back into the bf16 model.
            for param in ref_model_bf16.parameters():
                param.grad.data = param.grad.to(torch.float32)
                dist.all_reduce(param.grad)
                param.grad.div_(self.world_size)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad
                param_bf16.grad = None
            ref_optim.step()
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

        model, optim, ref_model, ref_optim, ref_model_bf16 = init_model(ThreeInputMLP)

        z_batch = torch.randn((6, dim), device=device_type.type)
        t_batch = torch.rand((6, 1), device=device_type.type)
        r_batch = t_batch * torch.rand((6, 1), device=device_type.type)
        fsdp_vmap_out = torch.vmap(model)(z_batch, t_batch, r_batch)
        ref_vmap_out = torch.vmap(
            lambda z_, t_, r_: ref_model_bf16(
                z_.to(torch.bfloat16),
                t_.to(torch.bfloat16),
                r_.to(torch.bfloat16),
            )
        )(z_batch, t_batch, r_batch)
        self.assertEqual(fsdp_vmap_out, ref_vmap_out)

        mixed_model, mixed_optim, mixed_ref_model, mixed_ref_optim, mixed_ref_bf16 = (
            init_model(MixedOutputMLP)
        )
        mixed_optim.zero_grad(set_to_none=True)
        mixed_ref_optim.zero_grad(set_to_none=True)
        fsdp_mixed_out, fsdp_aux = torch.vmap(mixed_model)(z_batch, t_batch, r_batch)
        ref_mixed_out, ref_aux = torch.vmap(
            lambda z_, t_, r_: mixed_ref_bf16(
                z_.to(torch.bfloat16),
                t_.to(torch.bfloat16),
                r_.to(torch.bfloat16),
            )
        )(z_batch, t_batch, r_batch)
        self.assertEqual(fsdp_mixed_out, ref_mixed_out)
        self.assertEqual(fsdp_aux, ref_aux)
        fsdp_mixed_loss = fsdp_mixed_out.square().mean()
        ref_mixed_loss = ref_mixed_out.square().mean()
        self.assertEqual(fsdp_mixed_loss, ref_mixed_loss)
        fsdp_mixed_loss.backward()
        mixed_optim.step()
        ref_mixed_loss.backward()
        reduce_ref_grads_and_step(mixed_ref_model, mixed_ref_bf16, mixed_ref_optim)
        check_sharded_parity(self, mixed_ref_model, mixed_model)

        num_iters = 5
        for iter_idx in range(num_iters):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            torch.manual_seed(42 + self.rank * 10 + iter_idx)
            x = torch.randn((4, dim), device=device_type.type)
            e = torch.randn((4, dim), device=device_type.type)
            t = torch.rand((4, 1), device=device_type.type)
            r = t * torch.rand((4, 1), device=device_type.type)
            # Require input grads to exercise backward through the JVP primal.
            z = ((1 - t) * x + t * e).detach().requires_grad_()
            t = t.detach().requires_grad_()
            r = r.detach().requires_grad_()
            v = e - x

            fsdp_out, fsdp_tangent = torch.func.jvp(
                lambda z_, t_, r_: model(z_, t_, r_),
                (z, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
            )
            ref_out, ref_tangent = torch.func.jvp(
                lambda z_, t_, r_: ref_model_bf16(
                    z_.to(torch.bfloat16),
                    t_.to(torch.bfloat16),
                    r_.to(torch.bfloat16),
                ),
                (z, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
            )
            self.assertEqual(
                fsdp_out, ref_out, msg=lambda msg: f"{msg}\niter {iter_idx}"
            )
            self.assertEqual(
                fsdp_tangent, ref_tangent, msg=lambda msg: f"{msg}\niter {iter_idx}"
            )

            fsdp_target = v - (t - r) * fsdp_tangent
            ref_target = v - (t - r) * ref_tangent
            fsdp_loss = ((fsdp_out - fsdp_target.detach()) ** 2).mean()
            ref_loss = ((ref_out - ref_target.detach()) ** 2).mean()
            # Include the tangent in the loss so backward enters FSDP through
            # the JVP tangent output, not only through the primal output.
            fsdp_loss = fsdp_loss + fsdp_tangent.square().mean()
            ref_loss = ref_loss + ref_tangent.square().mean()
            self.assertEqual(
                fsdp_loss, ref_loss, msg=lambda msg: f"{msg}\niter {iter_idx}"
            )

            fsdp_loss.backward()
            optim.step()

            ref_loss.backward()
            reduce_ref_grads_and_step(ref_model, ref_model_bf16, ref_optim)

            check_sharded_parity(self, ref_model, model)

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        # [FUNCTION DOC] Runs the fp32-reduce and bf16-reduce checks in order. Needs 2 GPUs and
        # NCCL 2.10+.
        self._test_reduce_dtype_fp32_reduce()
        self._test_reduce_dtype_bf16_reduce()

    def _test_reduce_dtype_fp32_reduce(self):
        # [FUNCTION DOC] HELPER: With bf16 params and fp32 reduce dtype, trains for 10
        # iterations, asserting reduce-scatter outputs are fp32; the bf16 reference all-reduces
        # fp32 grads and averages. Asserts equal loss and parity each iteration.
        param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output dtype equals
            # reduce_dtype.
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

    def _test_reduce_dtype_bf16_reduce(
        self,
    ):
        # [FUNCTION DOC] HELPER: With fp32 params and bf16 reduce dtype, trains for 10
        # iterations, asserting reduce-scatter outputs are bf16; the fp32 reference casts grads
        # to bf16, averages them via functional reduce-scatter then all-gather on the default
        # group, and upcasts. Asserts equal loss and parity each iteration.
        param_dtype, reduce_dtype = torch.float32, torch.bfloat16
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        )
        group = dist.distributed_c10d._get_default_group()
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output dtype equals
            # reduce_dtype.
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
        # [FUNCTION DOC] Runs gradient accumulation with a reduce dtype with and without
        # reshard_after_forward, via run_subtests. Needs 2 GPUs.
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
        # [FUNCTION DOC] HELPER: With bf16 compute and fp32 reduce, runs 4 microbatches per
        # iteration for 10 iterations, disabling gradient sync and (conditionally) resharding
        # except on the last microbatch. A bf16 compute reference accumulates grads into an fp32
        # reference, which is all-reduced on the last microbatch. Asserts losses are bf16 and
        # equal, reduce-scatter outputs are fp32, and parity holds.
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
            replicate(mlp, mp_policy=mp_policy)
        replicate(model, mp_policy=mp_policy)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            # [FUNCTION DOC] HELPER: Asserts the reduce-scatter output dtype equals
            # reduce_dtype.
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


class TestReplicateMixedPrecisionCasts(FSDPTestMultiThread):
    # [CLASS DOC] Multi-thread (world size 2) tests of how replicate() with MixedPrecisionPolicy
    # casts forward inputs, handles norm module buffers, clamps the reduce dtype, and handles
    # dataclass inputs.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the fixed world size, 2.
        return 2

    @skip_if_lt_x_gpu(1)
    def test_float16_on_one_submodule(self):
        # [FUNCTION DOC] Uses SaveForwardInputsModel to record each module's input dtype.
        # Subtests: fp16 on child c2 (root and c1 see fp32, c2 sees fp16); fp16 on c2 with its
        # input casting disabled (all see fp32); fp16 on c1 with output dtype fp32 (c1 sees
        # fp16, others fp32). Needs 1 GPU.
        x = torch.zeros(2, 100, device=device_type)

        # Subtest 1: use fp16 on the second child submodule -- does not require
        # any additional casting logic
        forward_inputs: dict[str, nn.Module] = {}
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).to(device_type)
        replicate(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        replicate(model)
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
        replicate(
            model.c2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=False
            ),
        )
        replicate(model)
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
        replicate(
            model.c1,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, output_dtype=torch.float32
            ),
        )
        replicate(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float16)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(1)
    def test_submodules_with_external_inputs(self):
        # [FUNCTION DOC] Runs the external-input check with submodule input casting disabled and
        # enabled.
        self.run_subtests(
            {"enable_submodule_cast": [False, True]},
            self._test_submodules_with_external_inputs,
        )

    def _test_submodules_with_external_inputs(self, enable_submodule_cast: bool):
        # [FUNCTION DOC] HELPER: Builds a model with a child that also receives an external fp32
        # tensor y created in the parent forward, and applies fp16 policies. Asserts the model
        # and child x inputs are fp16, and y is fp16 only if enable_submodule_cast is true, else
        # fp32.
        class ToyModule(nn.Module):
            # [CLASS DOC] HELPER: Child module with a Linear layer that records its x and y
            # inputs into a shared dictionary.
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates Linear layer l and stores the shared
                # forward_inputs dictionary.
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records x and y in the dictionary and returns l applied
                # to x only.
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Parent module with Linear l1 and a ToyModule child l2; forward
            # creates an external fp32 ones tensor and passes it to l2 along with l1(x).
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Creates l1, the l2 child and stores the shared
                # forward_inputs dictionary.
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Records x, builds an external fp32 y of ones, and
                # returns l2 on l1(x), y.
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(
                    2, 100, device=device_type.type, dtype=torch.float32
                )  # external input
                return self.l2(self.l1(x), y)

        forward_inputs: dict[str, torch.Tensor] = {}
        model = ToyModel(forward_inputs).to(device_type)
        x = torch.zeros(2, 100, device=device_type.type, dtype=torch.float32)
        replicate(
            model.l2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=enable_submodule_cast
            ),
        )
        replicate(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
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
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_norm_modules_bf16(self):
        # [FUNCTION DOC] Runs the norm module checks with a bf16 param dtype. Needs 1 GPU and
        # NCCL 2.10+.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._test_norm_modules(mp_policy)

    @skip_if_lt_x_gpu(1)
    @skipIfRocmArch(MI300_ARCH)  # https://github.com/pytorch/pytorch/issues/182988
    def test_norm_modules_fp16(self):
        # [FUNCTION DOC] Runs the norm module checks with an fp16 param dtype; skipped on ROCm
        # MI300.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self._test_norm_modules(mp_policy)

    def _test_norm_modules(self, mp_policy: MixedPrecisionPolicy):
        # [FUNCTION DOC] HELPER: Applies replicate with mp_policy to Linear/LayerNorm,
        # Linear/BatchNorm1d and Conv/BatchNorm2d stacks, running forward and backward.
        # BatchNorm2d with fp32 buffers is expected to fail with RuntimeError about running_mean
        # type (except on HPU); it succeeds after casting its buffers to the param dtype, or
        # when the norm gets a policy with only output_dtype.
        def inner(model: nn.Module, x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Runs forward, asserts the output dtype equals the policy
            # param_dtype, and runs backward on the sum.
            # Run forward and backward to check for no type mismatch errors
            z = model(x)
            self.assertEqual(z.dtype, mp_policy.param_dtype)
            z.sum().backward()

        # Layer norm
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            replicate(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 1D
        model = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            replicate(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 2D: error in backward from buffer dtype mismatch
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            replicate(module, mp_policy=mp_policy)
        if TEST_HPU:
            inner(model, torch.randn((3, 1, 9, 9)))
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected running_mean to have type",  # Error not seen on HPUs and hence it can be skipped
            ):
                # Errors in batch norm 2D backward
                inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: cast buffers down to lower precision
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            replicate(module, mp_policy=mp_policy)
        # Casting batch norm buffers to the lower precision allows backward
        model[1].running_mean = model[1].running_mean.to(mp_policy.param_dtype)
        model[1].running_var = model[1].running_var.to(mp_policy.param_dtype)
        inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: use special mixed precision policy
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        bn_mp_policy = MixedPrecisionPolicy(output_dtype=mp_policy.param_dtype)
        replicate(model[1], mp_policy=bn_mp_policy)
        for module in (model[0], model[2], model):
            replicate(module, mp_policy=mp_policy)
        inner(model, torch.randn((3, 1, 9, 9)))

    @skip_if_lt_x_gpu(1)
    def test_clamp_reduce_dtype(self):
        # [FUNCTION DOC] Builds a bf16-initialized model with a bf16 reduce dtype policy,
        # converts the model to fp32, and asserts reduce-scatter outputs are still bf16 (the
        # reduce dtype is not clamped). Needs 1 GPU.
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
            replicate((module), mp_policy=mp_policy)
        replicate(model, mp_policy=mp_policy)

        # Check that the reduce-scatter runs in bf16 even after we change the
        # model from bf16 to fp32
        model.to(torch.float32)
        orig_reduce_scatter = dist.reduce_scatter_single

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, torch.bfloat16)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        with patch_reduce_scatter(reduce_scatter):
            inp = torch.randn((4, 32), device=device_type.type)
            loss = model(inp).sum()
            loss.backward()

    @skip_if_lt_x_gpu(1)
    def test_dataclass_input(self):
        # [FUNCTION DOC] Checks replicate with a casting policy works when the forward argument
        # is a dataclass holding a tensor, by running forward and backward.
        # KNOWN ISSUE: the input is moved with a hard-coded cuda call instead of the device
        # type.
        @dataclasses.dataclass
        class Input:
            # [CLASS DOC] HELPER: Dataclass wrapping a single tensor field x.
            x: torch.Tensor

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with one 10x10 Linear layer applied to the x field of
            # an Input.
            def __init__(self, *args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Forwards arguments to the base class and creates Linear
                # layer _layer.
                super().__init__(*args, **kwargs)
                self._layer = nn.Linear(10, 10)

            def forward(self, input: Input):
                # [FUNCTION DOC] HELPER: Returns the layer applied to the input's x tensor.
                return self._layer(input.x)

        mp_policy = MixedPrecisionPolicy(
            torch.bfloat16, torch.bfloat16, torch.bfloat16, True
        )
        model = Model()
        inp = Input(torch.randn(2, 10).cuda())

        replicate(model, mp_policy=mp_policy)
        loss = model(inp).sum()
        loss.backward()


if __name__ == "__main__":
    run_tests()
