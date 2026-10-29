# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DTensor behavior on assorted aten ops:
# contiguous/clone/copy_, in-place and out= variants, *_like and new_* factories,
# stack/cat/split/unbind/slice, scatter/gather, index/index_put family (including
# Partial inputs), bucketize, dtype casts (_to_copy), argmax and argmin, and
# sharding-propagation cache-key behavior. Covers sharding/placement results and
# numerics against plain tensors. Import time: registers two custom ops (testlib::
# optional_clamp_op, testlib::modified_cat_op) and creates LocalTensor-based test
# class variants.

import itertools
import unittest

import torch
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._sharding_prop import ShardingPropagator
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    serialTest,
    TEST_WITH_ROCM,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorContinuousTestBase,
    DTensorConverter,
    DTensorTestBase,
    LocalDTensorContinuousTestBase,
    LocalDTensorTestBase,
    op_strategy_context,
    with_comms,
)


class DistTensorOpsTest(DTensorContinuousTestBase):
    # [CLASS DOC] Multi-process (world_size 4) DTensor op tests on a 1-D mesh unless noted. Each
    # test builds a mesh, runs an op on distributed tensors, and checks placements,
    # communication counts (via CommDebugMode) and values versus the plain-tensor result. Also
    # re-instantiated for LocalTensor mode via create_local_tensor_test_class below.
    world_size = 4

    def test_aten_contiguous(self):
        # [FUNCTION DOC] Checks that the aten contiguous op, which dtensor_ops
        # does not cover, gives the same full tensor for every sharding that
        # _test_op generates for a random 16x32 input.
        # this op not covered by dtensor_ops
        mesh = self.build_device_mesh()
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @with_comms
    def test_auto_registered_batch_norm_functional(self):
        # [FUNCTION DOC] Verifies that aten._native_batch_norm_legit_functional
        # gets a single-dim strategy auto-registered, then runs it with the input
        # sharded on the channel dim and the weight, bias and running stats
        # sharded on dim 0. Expects five outputs with the placements Shard(1)
        # followed by four Shard(0), and each full tensor equal to the plain-
        # tensor result.
        self.assertIn(
            torch.ops.aten._native_batch_norm_legit_functional.default,
            DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs,
        )
        device_mesh = self.build_device_mesh()
        input = torch.arange(
            8 * 4 * 3 * 3, device=self.device_type, dtype=torch.float32
        ).reshape(8, 4, 3, 3)
        input = input / input.numel()
        weight = torch.linspace(0.5, 1.1, 4, device=self.device_type)
        bias = torch.linspace(-0.2, 0.2, 4, device=self.device_type)
        running_mean = torch.zeros(4, device=self.device_type)
        running_var = torch.ones(4, device=self.device_type)

        expected = torch.ops.aten._native_batch_norm_legit_functional(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            True,
            0.1,
            1e-5,
        )

        dinput = distribute_tensor(input, device_mesh, [Shard(1)])
        dweight = distribute_tensor(weight, device_mesh, [Shard(0)])
        dbias = distribute_tensor(bias, device_mesh, [Shard(0)])
        drunning_mean = distribute_tensor(running_mean, device_mesh, [Shard(0)])
        drunning_var = distribute_tensor(running_var, device_mesh, [Shard(0)])

        result = torch.ops.aten._native_batch_norm_legit_functional(
            dinput,
            dweight,
            dbias,
            drunning_mean,
            drunning_var,
            True,
            0.1,
            1e-5,
        )

        self.assertEqual(len(result), 5)
        expected_placements = (Shard(1), Shard(0), Shard(0), Shard(0), Shard(0))
        for output, placements in zip(result, expected_placements):
            self.assertEqual(output.placements, (placements,))
        for actual, expected_tensor in zip(result, expected):
            self.assertEqual(actual.full_tensor(), expected_tensor)

    @with_comms
    def test_auto_registered_explicit_out_variant(self):
        # [FUNCTION DOC] Checks that an op whose out overload has an explicit out argument
        # (max.dim_max, whose overload name does not contain 'out') still gets a single-dim
        # strategy registered. Calls max with out=(values, indices) on Shard(0) DTensors;
        # expects the very same output objects returned, keeping Shard(0) placements, with full
        # tensors equal to the plain result.
        self.assertNotIn("out", torch.ops.aten.max.dim_max._schema.overload_name)
        self.assertTrue(
            any(
                argument.is_out
                for argument in torch.ops.aten.max.dim_max._schema.arguments
            )
        )
        self.assertIn(
            torch.ops.aten.max.dim_max,
            DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs,
        )
        device_mesh = self.build_device_mesh()
        input = torch.arange(
            8 * 4, device=self.device_type, dtype=torch.float32
        ).reshape(8, 4)
        expected_values, expected_indices = torch.max(input, dim=1)

        dinput = distribute_tensor(input, device_mesh, [Shard(0)])
        dvalues = distribute_tensor(
            torch.empty_like(expected_values), device_mesh, [Shard(0)]
        )
        dindices = distribute_tensor(
            torch.empty_like(expected_indices), device_mesh, [Shard(0)]
        )

        result = torch.max(dinput, dim=1, out=(dvalues, dindices))

        self.assertIs(result[0], dvalues)
        self.assertIs(result[1], dindices)
        self.assertEqual(dvalues.placements, (Shard(0),))
        self.assertEqual(dindices.placements, (Shard(0),))
        self.assertEqual(dvalues.full_tensor(), expected_values)
        self.assertEqual(dindices.full_tensor(), expected_indices)

    def test_detach(self):
        # [FUNCTION DOC] Checks that detach on a sharded DTensor returns a new
        # object (not the input).
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    def test_detach_inplace_inference_mode(self):
        # [FUNCTION DOC] Under inference_mode, calls aten.detach_ on DTensors sharded or
        # replicated and checks the input object itself is returned with unchanged placements,
        # i.e. the op has a registered sharding strategy that propagates the input spec.
        # Under inference_mode, autograd does not intercept detach_, so the
        # call falls through to DTensor's dispatcher. Ensure aten.detach_ has a
        # registered sharding strategy and propagates the input spec.
        device_mesh = self.build_device_mesh()
        for spec in ([Shard(0)], [Replicate()]):
            mat = distribute_tensor(torch.randn(12, 8), device_mesh, spec)
            with torch.inference_mode():
                out = torch.ops.aten.detach_(mat)
            self.assertTrue(out is mat)
            self.assertEqual(out.placements, tuple(spec))

    def test_clone(self):
        # [FUNCTION DOC] Clones a DTensor under Replicate and Shard(0); expects a distinct
        # object whose local tensor equals the original local tensor.
        device_mesh = self.build_device_mesh()
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    def test_copy_(self):
        # [FUNCTION DOC] Exercises DTensor.copy_ across several cases on a 1-D mesh: same-shape
        # replicate/shard pairs, broadcasting a smaller source, sources with incompatible
        # shardings (redistribute needed), and Partial source/destination (sum default, then max
        # and min), where the destination must keep its Partial placement and local data.
        # Compares to plain copy_.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # basic test
        src_tensor = torch.randn((12, 12))
        dst_tensor = torch.zeros(12, 12)
        src_specs = [[Replicate()], [Shard(0)]]
        dst_specs = [[Replicate()], [Shard(0)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, dst_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, src_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # simple broadcasting
        src_tensor = torch.randn((128,))
        dst_tensor = torch.zeros(128, 128)
        src_specs = [[Replicate()], [Shard(0)]]
        dst_specs = [[Replicate()], [Shard(1)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, src_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # The src specs in this case are designed to not be compatible with the dst_specs, redistribute should happen
        src_tensor = torch.randn((64, 1))
        dst_tensor = torch.zeros(16, 32, 64, 128)
        src_specs = [[Shard(1)], [Shard(1)], [Shard(1)], [Shard(1)]]
        dst_specs = [[Replicate()], [Shard(0)], [Shard(1)], [Shard(2)]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = distribute_tensor(src_tensor, device_mesh, src_spec)
            dst_dtensor = distribute_tensor(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.full_tensor(), dst_tensor)

        # as a pointwise op, need to keep Partial placements without redistribute
        src_tensor = torch.randn((64, 1))
        dst_tensor = torch.zeros(16, 32, 64, 128)
        src_specs = [[Partial()]]
        dst_specs = [[Partial()]]
        for dst_spec, src_spec in zip(dst_specs, src_specs):
            src_dtensor = DTensor.from_local(src_tensor, device_mesh, src_spec)
            dst_dtensor = DTensor.from_local(dst_tensor, device_mesh, dst_spec)
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.placements, (Partial(),))
            self.assertEqual(dst_dtensor._local_tensor, dst_tensor)

        # test that copy_ preserves any Partial type, not just sum/avg
        for reduce_op in ["max", "min"]:
            src_tensor = torch.randn((64, 1))
            dst_tensor = torch.zeros(16, 32, 64, 128)
            partial_placement = Partial(reduce_op)
            src_dtensor = DTensor.from_local(
                src_tensor, device_mesh, [partial_placement]
            )
            dst_dtensor = DTensor.from_local(
                dst_tensor, device_mesh, [partial_placement]
            )
            dst_dtensor.copy_(src_dtensor)
            dst_tensor.copy_(src_tensor)
            self.assertEqual(dst_dtensor.placements, (partial_placement,))
            self.assertEqual(dst_dtensor._local_tensor, dst_tensor)

    def test_contiguous(self):
        # [FUNCTION DOC] Checks contiguity and strides of a Shard(0) DTensor:
        # transpose makes it (and its local tensor) non-contiguous with the
        # expected strides, contiguous() restores both, and a backward through the
        # local tensor yields an all-ones gradient on the source tensor.
        device_mesh = self.build_device_mesh()
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        # shard on dim 0 should not change stride (30, 6, 1)
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # check backward
        new_dt.to_local().sum().backward()
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    def test_inplace_op(self):
        # [FUNCTION DOC] Checks that in-place add_ and mul_ return the same DTensor object with
        # correct local values, and that add_ with a Partial operand leaves the Shard(0)
        # placement of the target unchanged.
        mesh = self.build_device_mesh()
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

        # test inplace op self and other dtensor with other specs
        # and make sure out spec not change
        shard_spec = [Shard(0)]
        partial_spec = [Partial()]
        dt_to_inplace_add = distribute_tensor(input_tensor, mesh, shard_spec)
        partial_grad = DTensor.from_local(torch.randn(12, 3), mesh, partial_spec)
        res = dt_to_inplace_add.add_(partial_grad)
        self.assertTrue(res is dt_to_inplace_add)
        self.assertTrue(res.placements == tuple(shard_spec))

    def test_op_out_variant(self):
        # [FUNCTION DOC] Checks the out= variant of torch.add: the returned object is the out
        # DTensor, values match, and an out tensor with Replicate placement keeps that placement
        # even when the input is Shard(0).
        mesh = self.build_device_mesh()
        input_tensor = torch.randn((12, 3), device=self.device_type)
        sharded_dt_input = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = sharded_dt_input.clone() + 3
        sharded_dt_out = sharded_dt_input.clone()
        res = torch.add(sharded_dt_input, 3, out=sharded_dt_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is sharded_dt_out)
        self.assertEqual(sharded_dt_out.to_local(), expected_dt.to_local())

        # test op out variant with other spec and make sure out spec not change
        replica_spec = [Replicate()]
        replicate_out = distribute_tensor(input_tensor, mesh, replica_spec)
        expected_dt = replicate_out.clone() + 3
        res = torch.add(sharded_dt_input, 3, out=replicate_out)
        self.assertTrue(res is replicate_out)
        self.assertTrue(res.placements == tuple(replica_spec))
        self.assertEqual(replicate_out.to_local(), expected_dt.to_local())

    def test_eye_out_sharded_rejected(self):
        # [FUNCTION DOC] Builds Shard(0) and Shard(1) DTensors of global size 8x8
        # and checks that torch.eye with out= pointing at them raises RuntimeError
        # mentioning 'no valid sharding strategy'.
        mesh = self.build_device_mesh()
        input_size = (8, 8)

        for placement in (Shard(0), Shard(1)):
            local_size = list(input_size)
            local_size[placement.dim] //= self.world_size
            local_tensor = torch.empty(*local_size, device=self.device_type)
            dtensor = DTensor.from_local(
                local_tensor,
                mesh,
                [placement],
                shape=torch.Size(input_size),
                stride=(input_size[1], 1),
            )

            with self.assertRaisesRegex(RuntimeError, "no valid sharding strategy"):
                torch.eye(input_size[0], input_size[1], out=dtensor)

    def test_empty_like(self):
        # [FUNCTION DOC] Checks empty_like on a Shard(0) DTensor gives a local tensor of the
        # expected shape (values are not checked, only that sharding propagated).
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        empty_like_dt = torch.empty_like(dist_tensor)
        # empty is not deterministic, so we only check that the shard propagation worked
        self.assertEqual((4, 8), empty_like_dt.to_local().shape)

    def test_meta_init_partial(self):
        # [FUNCTION DOC] Builds a module on the meta device with a scalar buffer replaced by a
        # Partial DTensor, calls to_empty, and checks the buffer keeps its Partial placement.
        device_mesh = self.build_device_mesh()
        partial_spec = [Partial()]

        class ToyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module holding one scalar float buffer
            # named scalar_buffer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the scalar buffer (value 0.0, float32).
                super().__init__()
                self.register_buffer(
                    "scalar_buffer", torch.tensor(0.0, dtype=torch.float32)
                )

        with torch.device("meta"):
            module = ToyModule()
            module._buffers["scalar_buffer"] = DTensor.from_local(
                module.scalar_buffer,
                device_mesh=device_mesh,
                placements=partial_spec,
            )
        module.to_empty(device=None)

        # check that to_empty preserves partial
        self.assertEqual(module.scalar_buffer.placements, (Partial(),))

    def test_fill_inplace(self):
        # [FUNCTION DOC] Checks torch.fill_ on a Shard(0) DTensor fills the local data in place
        # and returns an object with the same local values.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    def test_full_like(self):
        # [FUNCTION DOC] Checks torch.full_like on a Shard(0) DTensor yields local data of the
        # same shape filled with 42.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.full_like(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())

    def test_ones_like(self):
        # [FUNCTION DOC] Checks torch.ones_like on a Shard(0) DTensor yields all-
        # ones local data.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        ones_like_dt = torch.ones_like(dist_tensor)
        ones_expected = torch.ones(4, 8)
        self.assertEqual(ones_expected, ones_like_dt.to_local())

    def test_ones_like_partial_sum(self):
        # [FUNCTION DOC] Checks ones_like on a Partial DTensor: no communication occurs, the
        # result is Replicate, and its full tensor is all ones (not scaled by world size).
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        comm_mode = CommDebugMode()
        with comm_mode:
            ones_like_dt = torch.ones_like(dist_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(ones_like_dt.placements, (Replicate(),))
        ones_expected = torch.ones(dist_tensor.shape)
        self.assertEqual(ones_expected, ones_like_dt.full_tensor())

    def test_fill_inplace_partial_sum(self):
        # [FUNCTION DOC] Checks in-place fill_ on a Partial(sum) DTensor keeps it Partial, so
        # the reduced full tensor equals the fill value times world_size.
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        # inplace partial sum should keep partial
        torch.fill_(dist_tensor, 8)
        fill_expected = torch.full(
            dist_tensor.shape, 8 * self.world_size, dtype=input_tensor.dtype
        )
        self.assertEqual(fill_expected, dist_tensor.full_tensor())

    def test_zeros_like_partial_sum(self):
        # [FUNCTION DOC] Checks zeros_like on a Partial DTensor: no communication, Replicate
        # result, full tensor all zeros.
        device_mesh = self.build_device_mesh()
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        if not (dist_tensor.shape == (4, 8)):
            raise AssertionError(f"Expected shape (4, 8), got {dist_tensor.shape}")

        comm_mode = CommDebugMode()
        with comm_mode:
            zeros_like_dt = torch.zeros_like(dist_tensor)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(zeros_like_dt.placements, (Replicate(),))
        zeros_expected = torch.zeros(dist_tensor.shape)
        self.assertEqual(zeros_expected, zeros_like_dt.full_tensor())

    def test_zero_inplace(self):
        # [FUNCTION DOC] Checks torch.zero_ on a Shard(0) DTensor zeroes the local
        # data in place.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    def test_zeros_like(self):
        # [FUNCTION DOC] Checks zeros_like with dtype bfloat16 on a float32 Shard(0) DTensor:
        # result is bfloat16 zeros and the input dtype is unchanged.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zeros_like(dist_tensor, dtype=torch.bfloat16)
        zeros_expected = torch.zeros(4, 8, dtype=torch.bfloat16)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        # make sure there is no side effect on the input tensor dtype
        self.assertEqual(dist_tensor.dtype, torch.float32)
        self.assertEqual(zeros_like_dt.dtype, torch.bfloat16)

    @skip_if_lt_x_gpu(4)
    def test_stack(self):
        # [FUNCTION DOC] Requires at least 4 GPUs. Stacks DTensors on a 2x2 mesh with
        # Partial/Replicate inputs and expects Partial/Partial output with shape (2, 4, 8). On a
        # 1-D mesh checks stacking Shard(1) inputs: dim 0 and dim 1 shift the shard to Shard(2),
        # dim=-1 keeps Shard(1); full tensors are compared with plain stack.
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        partial_replicate_placement = [Partial(), Replicate()]
        partial_placement = [Partial(), Partial()]

        partial_replicate_dt = DTensor.from_local(
            torch.randn(4, 8), mesh_2d, partial_replicate_placement
        )
        partial_dt = DTensor.from_local(torch.randn(4, 8), mesh_2d, partial_placement)

        stack_dt = torch.stack([partial_replicate_dt, partial_dt])
        self.assertEqual(stack_dt.placements, tuple(partial_placement))
        self.assertEqual(stack_dt.shape, (2, 4, 8))

        mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # stack before/after shard dim
        global_input = torch.randn(8, 8)
        shard1_input = distribute_tensor(global_input, mesh_1d, [Shard(1)])
        cloned_shard1_input = shard1_input.clone()
        stack_shard1_dt = torch.stack([shard1_input, cloned_shard1_input])
        self.assertEqual(stack_shard1_dt.placements, (Shard(2),))
        self.assertEqual(stack_shard1_dt.shape, (2, 8, 8))
        self.assertEqual(
            stack_shard1_dt.full_tensor(), torch.stack([global_input, global_input])
        )

        stack_dim1_shard1_dt = torch.stack([shard1_input, cloned_shard1_input], dim=1)
        self.assertEqual(stack_dim1_shard1_dt.placements, (Shard(2),))
        self.assertEqual(stack_dim1_shard1_dt.shape, (8, 2, 8))
        self.assertEqual(
            stack_dim1_shard1_dt.full_tensor(),
            torch.stack([global_input, global_input], dim=1),
        )

        # stack with negative dim: dim=-1 inserts at the last position of the
        # output (ndim+1), so Shard(1) should stay Shard(1)
        stack_neg_dim_dt = torch.stack([shard1_input, cloned_shard1_input], dim=-1)
        self.assertEqual(stack_neg_dim_dt.placements, (Shard(1),))
        self.assertEqual(stack_neg_dim_dt.shape, (8, 8, 2))
        self.assertEqual(
            stack_neg_dim_dt.full_tensor(),
            torch.stack([global_input, global_input], dim=-1),
        )

    def test_stack_cache(self):
        # [FUNCTION DOC] Stacks a list of three Replicate DTensors, then stacks a second list of
        # the same specs while patching ShardingPropagator._propagate_tensor_meta_non_cached to
        # raise; passes only if the second call hits the sharding propagation cache.
        device_mesh = self.build_device_mesh()

        shape = (4, 8)
        placements = [Replicate()]
        dtensor_list = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements)
            dtensor_list.append(dt)

        _ = torch.stack(dtensor_list)

        dtensor_list2 = []
        for _ in range(3):
            local_tensor = torch.randn(shape)
            dt = DTensor.from_local(local_tensor, device_mesh, placements)
            dtensor_list2.append(dt)

        def error(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement that accepts any arguments and always raises
            # AssertionError, used to prove the cached path is taken.
            raise AssertionError

        with unittest.mock.patch.object(
            ShardingPropagator, "_propagate_tensor_meta_non_cached", error
        ):
            _ = torch.stack(dtensor_list2)

    def test_equal(self):
        # [FUNCTION DOC] Checks DTensor.equal and is_same_size: equal sharded tensors are equal,
        # a difference on only some ranks makes equal return False (result is all-reduced), and
        # comparisons against a Replicate DTensor of different local shape behave as asserted.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        input_tensor_1 = torch.ones(4, 4)
        dist_tensor_1 = DTensor.from_local(input_tensor_1, device_mesh, shard_spec)

        # tensors are equal
        input_tensor_2 = torch.ones(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        self.assertTrue(eq_result)

        # tensors are different on some shards
        if self.rank == 0:
            input_tensor_2 = torch.ones(4, 4)
        else:
            input_tensor_2 = torch.randn(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # equal op all reduces each shard's local result
        self.assertFalse(eq_result)
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_2))

        # test if sharding are different
        replica_spec = [Replicate()]
        global_input = torch.ones(4 * self.world_size, 4)
        dist_tensor_3 = DTensor.from_local(
            global_input, device_mesh, replica_spec, run_check=False
        )

        self.assertTrue(dist_tensor_1.equal(dist_tensor_3))
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))

        # test sharding difference with only some shards content difference
        self.assertFalse(dist_tensor_2.equal(dist_tensor_3))
        self.assertTrue(dist_tensor_1.is_same_size(dist_tensor_3))
        self.assertFalse(input_tensor_2.is_same_size(dist_tensor_3))

    def test_is_pinned(self):
        # [FUNCTION DOC] Checks is_pinned is False for a DTensor built from a
        # CPU/regular local tensor.
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        dt = DTensor.from_local(torch.ones(4, 4), device_mesh, shard_spec)
        self.assertFalse(dt.is_pinned())

    def _test_op(self, mesh, op_call, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs op_call on the plain arguments, then for
        # every DTensor sharding combination produced by DTensorConverter over the
        # same arguments asserts the conversion succeeded and the full tensor of
        # the DTensor result equals the plain result.
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            self.assertTrue(dtc.successful())
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def test_new_full(self):
        # [FUNCTION DOC] Checks new_full on Shard(0) and Replicate inputs: a different shape
        # gives a Replicate result and the same shape keeps the input placement; neither
        # communicates, and values are 42.
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        global_tensor = torch.randn(12, 8)
        placements = [[Shard(0)], [Replicate()]]
        for placement in placements:
            input_dt = distribute_tensor(global_tensor, device_mesh, placement)
            with comm_mode:
                new_full_diff_dt = input_dt.new_full((4, 8), 42.0)
                # new_full_diff_dt creates a replicated tensor, regardless of input_dt placement,
                # which should not trigger any communication.
                self.assertEqual(comm_mode.get_total_counts(), 0)
            new_full_diff_expected = torch.full((4, 8), 42.0)
            self.assertTrue(new_full_diff_dt.placements[0].is_replicate())
            self.assertEqual(new_full_diff_expected, new_full_diff_dt.to_local())

            with comm_mode:
                new_full_same_dt = input_dt.new_full((12, 8), 42.0)
                # new_full_same_dt creates a tensor with the same placement as input_dt,
                # which should not trigger any communication.
                self.assertEqual(comm_mode.get_total_counts(), 0)
            new_full_same_expected = torch.full((12, 8), 42.0)
            self.assertEqual(new_full_same_dt.placements, placement)
            self.assertEqual(new_full_same_expected, new_full_same_dt.full_tensor())

    def test_new_empty_strided(self):
        # [FUNCTION DOC] Checks new_empty_strided on a Shard(1) DTensor with no
        # communication: same output shape keeps the sharding and local
        # size/stride (even or uneven split), while a different output shape gives
        # a Replicate result with the requested size and stride.
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        shard_dim = 1
        placement = (Shard(shard_dim),)

        # output shape same as input shape, evenly sharded input -> output same sharding as input
        global_tensor = torch.randn(12, 8)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size == 0)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 8), (8, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, placement)
        self.assertEqual(
            new_empty_strided_dt._local_tensor.size(), (12, 8 // self.world_size)
        )
        self.assertEqual(
            new_empty_strided_dt._local_tensor.stride(), (8 // self.world_size, 1)
        )
        self.assertTrue(new_empty_strided_dt.contiguous() is new_empty_strided_dt)

        # output shape same as input shape, unevenly sharded input -> output follows input
        global_tensor = torch.randn(12, 7)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        self.assertTrue(input_dt.shape[shard_dim] % self.world_size != 0)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 7), (7, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, placement)
        self.assertEqual(
            new_empty_strided_dt._local_tensor.size(),
            input_dt._local_tensor.size(),
        )
        self.assertEqual(
            new_empty_strided_dt._local_tensor.stride(),
            input_dt._local_tensor.stride(),
        )

        # output shape different from input shape -> output replicated
        global_tensor = torch.randn(12, 8)
        input_dt = distribute_tensor(global_tensor, device_mesh, placement)
        with comm_mode:
            new_empty_strided_dt = input_dt.new_empty_strided((12, 4), (4, 1))
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(new_empty_strided_dt.placements, (Replicate(),))
        self.assertEqual(new_empty_strided_dt._local_tensor.size(), (12, 4))
        self.assertEqual(new_empty_strided_dt._local_tensor.stride(), (4, 1))

    def test_scatter(self):
        # [FUNCTION DOC] Checks torch.scatter on DTensors against plain scatter. Case 1:
        # everything Replicate, no communication, Replicate output, for tensor and scalar src.
        # Case 2: input, index, src sharded on a non-scatter dim, no communication and output
        # stays sharded. Case 3: src longer than index on the shard dim with replicated src,
        # expects a Replicate output (sharded strategy not offered) with correct values.
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        # case 1 all replicate: input replicated, index/src replicated, output replicated
        global_indexs = [
            torch.tensor([[0, 1, 2, 0]]),
            torch.tensor([[0, 1, 2], [0, 1, 4]]),
        ]
        for scatter_dim in [0, 1]:
            srcs = [torch.arange(1, 11).reshape((2, 5)), 4]
            for global_src in srcs:
                global_input = torch.zeros(3, 5, dtype=torch.int64)
                global_index = global_indexs[scatter_dim]

                input_dt = distribute_tensor(
                    global_input.clone(), device_mesh, [Replicate()]
                )
                index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
                if isinstance(global_src, torch.Tensor):
                    src_dt = distribute_tensor(global_src, device_mesh, [Replicate()])
                else:
                    src_dt = global_src
                global_output = torch.scatter(
                    global_input, scatter_dim, global_index, global_src
                )
                with comm_mode:
                    output_dt = torch.scatter(input_dt, scatter_dim, index_dt, src_dt)

                self.assertEqual(comm_mode.get_total_counts(), 0)
                self.assertEqual(output_dt.placements, [Replicate()])
                self.assertEqual(output_dt.to_local(), global_output)

        # case 2 shard on a non-scatter dim: input/index/src and output all
        # sharded on dim 0, scatter performed along dim 1. This must run as a
        # purely local op with no communication.
        shard_dim, scatter_dim = 0, 1
        rows = self.world_size * 2
        global_input = torch.zeros(rows, 5, dtype=torch.int64)
        # deterministic index so every spawned rank builds the same tensor
        global_index = torch.arange(rows * 3).reshape(rows, 3) % 5
        srcs = [torch.arange(1, rows * 3 + 1).reshape((rows, 3)), 4]
        for global_src in srcs:
            global_output = torch.scatter(
                global_input, scatter_dim, global_index, global_src
            )
            input_dt = distribute_tensor(
                global_input.clone(), device_mesh, [Shard(shard_dim)]
            )
            index_dt = distribute_tensor(global_index, device_mesh, [Shard(shard_dim)])
            if isinstance(global_src, torch.Tensor):
                src_dt = distribute_tensor(global_src, device_mesh, [Shard(shard_dim)])
            else:
                src_dt = global_src
            with comm_mode:
                output_dt = torch.scatter(input_dt, scatter_dim, index_dt, src_dt)

            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(output_dt.placements, [Shard(shard_dim)])
            self.assertEqual(output_dt.full_tensor(), global_output)

        # case 3 src larger than index on the shard dim: sharding src to match
        # index would misalign boundaries, so the sharded strategy must not be
        # offered and the op falls back to replicate. Result must stay correct.
        global_input = torch.zeros(rows, 5, dtype=torch.int64)
        global_index = torch.arange(rows * 3).reshape(rows, 3) % 5
        global_src = torch.arange(1, (rows + 2) * 3 + 1).reshape((rows + 2, 3))
        global_output = torch.scatter(
            global_input, scatter_dim, global_index, global_src
        )
        input_dt = distribute_tensor(
            global_input.clone(), device_mesh, [Shard(shard_dim)]
        )
        index_dt = distribute_tensor(global_index, device_mesh, [Shard(shard_dim)])
        src_dt = distribute_tensor(global_src, device_mesh, [Replicate()])
        output_dt = torch.scatter(input_dt, scatter_dim, index_dt, src_dt)
        # the sharded strategy is not offered, so the op falls back to replicate
        self.assertEqual(output_dt.placements, [Replicate()])
        self.assertEqual(output_dt.full_tensor(), global_output)

    def test_gather(self):
        # [FUNCTION DOC] Checks torch.gather on DTensors with no communication:
        # replicated input and index give Replicate output for each dim; input
        # sharded on the gather dim with an index of size 1 there gives a
        # _MaskPartial output; replicated input with index sharded on the gather
        # dim gives output sharded on that dim. Values are compared with plain
        # gather.
        device_mesh = self.build_device_mesh()
        comm_mode = CommDebugMode()

        # case 1 all replicate: input replicated, index replicated, output replicated
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 4, 8))
        input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        for gather_dim in [0, 1, 2]:
            global_output = torch.gather(global_input, gather_dim, global_index)
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(output_dt.placements, [Replicate()])
            self.assertEqual(output_dt.to_local(), global_output)

        # case 2 input sharding: input sharded, index replicated, output mask partial
        # only works when index has size 1 on the gather dimension and
        # input is sharded on the gather dimension
        from torch.distributed.tensor.placement_types import _MaskPartial

        gather_dim = 1
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 1, 8))
        global_output = torch.gather(global_input, gather_dim, global_index)
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(gather_dim)])
        index_dt = distribute_tensor(global_index, device_mesh, [Replicate()])
        with comm_mode:
            output_dt = torch.gather(input_dt, gather_dim, index_dt)
            self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertIsInstance(output_dt.placements[0], _MaskPartial)
        self.assertEqual(output_dt.full_tensor(), global_output)

        # case 3 index sharding: input replicated, index sharded, output sharded
        # only works when the sharding dimension is the gather dimension
        global_input = torch.randn(12, 8, 16)
        global_index = torch.randint(8, (4, 4, 8))
        for gather_dim in range(len(global_index.shape)):
            input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
            index_dt = distribute_tensor(global_index, device_mesh, [Shard(gather_dim)])
            global_output = torch.gather(global_input, gather_dim, global_index)
            with comm_mode:
                output_dt = torch.gather(input_dt, gather_dim, index_dt)
                self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(output_dt.placements, [Shard(gather_dim)])
            self.assertEqual(output_dt.full_tensor(), global_output)

    @unittest.skipIf(TEST_WITH_ROCM, "https://github.com/pytorch/pytorch/issues/175064")
    @serialTest()  # heavy combinatorial _test_op calls, serialize to avoid OOM
    def test_index(self):
        # [FUNCTION DOC] Runs _test_op over many indexing forms (advanced indexing with one or
        # two index tensors, index_select on dims 0 and 1, ellipsis, broadcast of index shapes)
        # on a 1-D mesh. Skipped on ROCm; run serially because of the number of sharding
        # combinations. Several 3-tensor variants are left commented out to avoid CI timeouts.
        meshes = [
            self.build_device_mesh(),  # 1D mesh
            # TODO(@azzolini): un-comment when DTensorConverter supports N-D mesh
            # DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, -1)), # 2D mesh
        ]
        for mesh in meshes:
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(1, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(0, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[:, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8, 16)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            # Commented out to fix distributed CI timeout: each 3-tensor call
            # generates 40-80 sharding combinations via itertools.product,
            # causing combinatorial explosion.
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, y],
            #     torch.randn(16, 32, 16),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[:, z, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (12, 8, 12)),
            # )
            # broadcast in inner dimensions
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 1, 12)),
            )
            # Commented out to fix distributed CI timeout: each 3-tensor call
            # generates 40-80 sharding combinations via itertools.product,
            # causing combinatorial explosion.
            # # implicit (left-padded) broadcast
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[:, z, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(5, (12, 8, 12)),
            #     torch.randint(2, (8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, y, :, :],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 12)),
            #     torch.randint(5, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, y, :],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 12)),
            #     torch.randint(5, (12, 8, 12)),
            # )
            # self._test_op(
            #     mesh,
            #     lambda x, y, z: x[z, :, :, y],
            #     torch.randn(16, 32, 16, 12),
            #     torch.randint(2, (8, 1)),
            #     torch.randint(5, (12, 8, 12)),
            # )

    def test_index_put_scalar(self):
        # [FUNCTION DOC] On a 2-D mesh with scalar indices on all dims and a scalar replicated
        # value, tries all Shard(0)/Shard(1)/Replicate combinations for the input and expects a
        # DTensor with (Replicate, Replicate) placements equal to plain index_put.
        device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        global_input = torch.randn(2, 4, 8, device=self.device_type)
        global_index = [
            torch.randint(global_input.shape[i], size=(), device=self.device_type)
            for i in range(3)
        ]
        global_value = torch.randn(size=(), device=self.device_type)
        value_dt = distribute_tensor(
            global_value, device_mesh, [Replicate(), Replicate()]
        )
        placement_choice_pool = [Shard(0), Shard(1), Replicate()]
        for i in placement_choice_pool:
            for j in placement_choice_pool:
                input_dt = distribute_tensor(global_input, device_mesh, [i, j])
                ref = torch.index_put(global_input, global_index, global_value)
                output_dt = torch.index_put(input_dt, global_index, value_dt)
                if not isinstance(output_dt, DTensor):
                    raise AssertionError(f"Expected DTensor, got {type(output_dt)}")
                # for value is a scalar case, output placement must be replicate
                self.assertEqual(output_dt.placements, (Replicate(), Replicate()))
                self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_tensor(self):
        # [FUNCTION DOC] On a 2-D mesh, index_put with a scalar index on dim 0 and a tensor
        # value, for accumulate True and False. Expects any Shard placements of the output to be
        # on dims 1 or 2 and values equal to plain index_put.
        device_mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        global_input = torch.randn(2, 4, 8, device=self.device_type)
        global_index = [
            torch.randint(global_input.shape[0], size=(), device=self.device_type)
        ]
        global_value = torch.zeros([4, 8], device=self.device_type)
        value_dt = distribute_tensor(global_value, device_mesh, [Shard(1), Replicate()])
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0), Replicate()])
        for accumulate in [True, False]:
            ref = torch.index_put(global_input, global_index, global_value, accumulate)
            output_dt = torch.index_put(input_dt, global_index, value_dt, accumulate)
            if not isinstance(output_dt, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(output_dt)}")
            # Output should be sharded on non-indexed dims (dim 1 or 2).
            # The exact placement depends on cost-based strategy selection.
            for p in output_dt.placements:
                if isinstance(p, Shard):
                    self.assertIn(p.dim, [1, 2])
            self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_requires_replicated_index(self):
        # [FUNCTION DOC] Checks index_put accepts a Shard(0) index DTensor (it must be
        # redistributed to Replicate internally) and matches the plain result.
        """Test that index_put correctly replicates sharded indices."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_value = torch.zeros([8], device=self.device_type)

        # Create sharded index - should be redistributed to replicated
        idx_tensor = torch.tensor([0, 1], device=self.device_type)
        idx_dt = distribute_tensor(idx_tensor, device_mesh, [Shard(0)])

        input_dt = distribute_tensor(global_input, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        # The op should work - index gets redistributed to replicated internally
        ref = torch.index_put(global_input, [idx_tensor], global_value)
        output_dt = torch.index_put(input_dt, [idx_dt], value_dt)
        self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_no_shard_on_indexed_dim(self):
        # [FUNCTION DOC] Shards the input on the indexed dim 0 and checks the index_put output
        # is not sharded on dim 0 and matches the plain result.
        """Test that index_put output is not sharded on indexed dims."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_value = torch.zeros([8], device=self.device_type)
        global_index = [torch.tensor(0, device=self.device_type)]

        # Shard input on indexed dim 0
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0)])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        ref = torch.index_put(global_input, global_index, global_value)
        output_dt = torch.index_put(input_dt, global_index, value_dt)

        # Verify output is not sharded on the indexed dim (dim 0)
        for p in output_dt.placements:
            if isinstance(p, Shard):
                self.assertNotEqual(
                    p.dim, 0, "Output should not be sharded on indexed dim"
                )
        self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_partial_numerics(self):
        # [FUNCTION DOC] index_put with Partial(sum) input and value built from local shares
        # divided by world size, for accumulate True and False. Expects a Partial output whose
        # reduced value equals the plain result.
        """Test index_put with Partial placements produces correct numerics."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        global_index = [torch.tensor(1, device=self.device_type)]
        global_value = torch.randn(8, device=self.device_type)

        # Create Partial tensors - each rank has partial sum
        local_input = global_input / self.world_size
        local_value = global_value / self.world_size
        input_dt = DTensor.from_local(
            local_input, device_mesh, [Partial()], run_check=False
        )
        value_dt = DTensor.from_local(
            local_value, device_mesh, [Partial()], run_check=False
        )

        for accumulate in [True, False]:
            ref = torch.index_put(global_input, global_index, global_value, accumulate)
            output_dt = torch.index_put(input_dt, global_index, value_dt, accumulate)
            self.assertIsInstance(output_dt, DTensor)
            self.assertEqual(output_dt.placements, (Partial(),))
            self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_duplicated_indices(self):
        # [FUNCTION DOC] index_put with a repeated index entry, input and value sharded on dim 1
        # and a replicated index, for accumulate False and True, compared with the plain result.
        """Test index_put with duplicated indices for both accumulate modes."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.zeros(4, 8, device=self.device_type)
        # Duplicated indices - index 1 appears twice
        idx_tensor = torch.tensor([1, 1, 2], device=self.device_type)
        global_index = [idx_tensor]
        global_value = torch.ones(3, 8, device=self.device_type)

        # Shard on non-indexed dim, index must be replicated
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(1)])
        idx_dt = distribute_tensor(idx_tensor, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Shard(1)])

        # accumulate=False: last write wins (non-deterministic with duplicates, but
        # since values are the same, result is deterministic)
        ref_no_accum = torch.index_put(
            global_input, global_index, global_value, accumulate=False
        )
        output_no_accum = torch.index_put(
            input_dt, [idx_dt], value_dt, accumulate=False
        )
        self.assertEqual(output_no_accum.full_tensor(), ref_no_accum)

        # accumulate=True: values are added (index 1 gets 2x the value)
        ref_accum = torch.index_put(
            global_input, global_index, global_value, accumulate=True
        )
        output_accum = torch.index_put(input_dt, [idx_dt], value_dt, accumulate=True)
        self.assertEqual(output_accum.full_tensor(), ref_accum)

    def test_index_put_broadcast_values(self):
        # [FUNCTION DOC] index_put where the values tensor has a size-1 dim to
        # broadcast, input sharded on the non-indexed dim 1 and values replicated;
        # compares with the plain result.
        """Test index_put where values has size-1 broadcast dims."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # self shape (4, 8), index on dim 0 -> non-indexed dim is 1
        global_input = torch.randn(4, 8, device=self.device_type)
        idx = torch.tensor([0, 1], device=self.device_type)
        global_index = [idx]
        # values shape (2, 1) — broadcast on the non-indexed dim
        global_value = torch.randn(2, 1, device=self.device_type)

        # Shard self on non-indexed dim 1, values should be replicated (size 1)
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(1)])
        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        ref = torch.index_put(global_input, global_index, global_value)
        output_dt = torch.index_put(input_dt, [idx_dt], value_dt)
        self.assertEqual(output_dt.full_tensor(), ref)

    def _test_index_op_partial_propagation(self, op, reduce_op):
        # [FUNCTION DOC] HELPER: For an index op taking (input, dim 1, index, source) such as
        # index_copy or index_add, builds Partial(reduce_op) input and source (sum divides
        # shares by world size; other ops copy the full value on each rank) with a replicated
        # index, and checks the DTensor result full tensor equals the plain result.
        """Helper: verify Partial placement propagates correctly through an
        index op that decomposes to index_put."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        idx = torch.tensor([1, 3], device=self.device_type)
        global_source = torch.randn(4, 2, device=self.device_type)

        # Create Partial tensors — each rank holds partial contribution
        if reduce_op == "sum":
            local_input = global_input / self.world_size
            local_source = global_source / self.world_size
        else:
            # avg/max/min: each rank holds the full value
            local_input = global_input.clone()
            local_source = global_source.clone()

        input_dt = DTensor.from_local(
            local_input, device_mesh, [Partial(reduce_op)], run_check=False
        )
        source_dt = DTensor.from_local(
            local_source, device_mesh, [Partial(reduce_op)], run_check=False
        )

        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])

        ref = op(global_input, 1, idx, global_source)
        output_dt = op(input_dt, 1, idx_dt, source_dt)
        self.assertIsInstance(output_dt, DTensor)
        self.assertEqual(output_dt.full_tensor(), ref)

    @parametrize("reduce_op", ["sum", "avg", "max", "min"])
    def test_index_copy_partial_propagation(self, reduce_op):
        # [FUNCTION DOC] Parametrized by reduce_op (sum, avg, max, min); runs the partial
        # propagation check for index_copy.
        self._test_index_op_partial_propagation(torch.index_copy, reduce_op)

    @parametrize("reduce_op", ["sum", "avg"])
    def test_index_add_partial_propagation(self, reduce_op):
        # [FUNCTION DOC] Parametrized by reduce_op (sum, avg); runs the partial propagation
        # check for index_add.
        self._test_index_op_partial_propagation(torch.index_add, reduce_op)

    @parametrize(
        "op",
        [torch.ops.aten.index_put, torch.index_copy, torch.index_add],
    )
    def test_index_ops_leading_none_dim_mapping(self, op):
        # [FUNCTION DOC] Regression test, parametrized over aten index_put, index_copy,
        # index_add: input sharded on dim 0 and source sharded on dim 1 with the index on dim 1
        # (leading None index). The result must still match plain numerics, i.e. the strategy
        # must not accept sharding the broadcast dim of the values.
        """Regression: index_put dim-mapping bug when indexed dim is not dim 0.

        When indices = (None, index), indexed_dims=[1] is contiguous so
        broadcast replaces dim 1 in-place:
            values shape = (self.size(0), len(index))  i.e. (non_indexed, broadcast)

        The buggy strategy assumed broadcast at front, so it wrongly allowed
        S(0) on self + S(1) on values (sharding the broadcast dim).
        We force that exact placement — if the strategy accepts it without
        redistributing, the numerics will be wrong.

        index_copy and index_add decompose to index_put with leading None,
        so they hit the same bug.
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        n = self.world_size
        global_input = torch.randn(n, 2 * n, device=self.device_type)
        idx = torch.arange(n, device=self.device_type)
        global_source = torch.randn(n, n, device=self.device_type)

        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0)])
        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        source_dt = distribute_tensor(global_source, device_mesh, [Shard(1)])

        if op is torch.ops.aten.index_put:
            ref = op(global_input, [None, idx], global_source)
            output_dt = op(input_dt, [None, idx_dt], source_dt)
        else:
            ref = op(global_input, 1, idx, global_source)
            output_dt = op(input_dt, 1, idx_dt, source_dt)

        self.assertIsInstance(output_dt, DTensor)
        self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_fill(self):
        # [FUNCTION DOC] Checks index_fill on a Shard(0) input with a replicated index along dim
        # 1 returns a DTensor equal to the plain result.
        """Test index_fill with sharded input."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        idx = torch.tensor([1, 3], device=self.device_type)
        fill_value = 5.0

        ref = global_input.index_fill(1, idx, fill_value)

        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0)])
        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])

        output_dt = input_dt.index_fill(1, idx_dt, fill_value)

        self.assertIsInstance(output_dt, DTensor)
        self.assertEqual(output_dt.full_tensor(), ref)

    @parametrize("reduce_op", ["mean", "amax", "amin", "prod"])
    def test_index_reduce(self, reduce_op):
        # [FUNCTION DOC] Parametrized by reduce_op (mean, amax, amin, prod);
        # checks index_reduce on a Shard(0) input and Shard(0) source with
        # replicated index matches the plain result.
        """Test index_reduce with sharded input."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, device=self.device_type)
        idx = torch.tensor([1, 3], device=self.device_type)
        global_source = torch.randn(4, 2, device=self.device_type)

        ref = global_input.index_reduce(1, idx, global_source, reduce_op)

        input_dt = distribute_tensor(global_input, device_mesh, [Shard(0)])
        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        source_dt = distribute_tensor(global_source, device_mesh, [Shard(0)])

        output_dt = input_dt.index_reduce(1, idx_dt, source_dt, reduce_op)

        self.assertIsInstance(output_dt, DTensor)
        self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_none_indices(self):
        # [FUNCTION DOC] Calls aten index_put with indices [idx, None, None]
        # directly, sharding the input on non-indexed dims 1 and 2 with replicated
        # index and values; compares with plain.
        """Test index_put with None entries in the indices list.

        None indices (meaning "select all along this dim") are only
        valid at the ATen level, not through Python's index_put.
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 3D tensor indexed on dim 0, with trailing None
        global_input = torch.randn(4, 8, 16, device=self.device_type)
        idx = torch.tensor([1, 2], device=self.device_type)
        # indices = [idx, None, None] -> indexed on dim 0, dims 1,2 not indexed
        # values shape = (2, 8, 16) matching (*broadcast_shape, *non_indexed_dims)
        global_value = torch.randn(2, 8, 16, device=self.device_type)

        ref = torch.ops.aten.index_put.default(
            global_input, [idx, None, None], global_value
        )

        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        # Shard self on non-indexed dims (1 or 2)
        for shard_dim in [1, 2]:
            input_dt = distribute_tensor(global_input, device_mesh, [Shard(shard_dim)])
            output_dt = torch.ops.aten.index_put.default(
                input_dt, [idx_dt, None, None], value_dt
            )
            self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_inplace_none_indices(self):
        # [FUNCTION DOC] Same as the previous test but for the in-place aten
        # index_put_ with None index entries, input sharded on dim 1; checks the
        # input DTensor is updated to the plain result.
        """Test in-place aten.index_put_ with None entries in the indices list.

        The in-place variant with None indices is generated by autograd
        (e.g. backward of aten.index.Tensor), not by Python-level
        index_put_. We call the ATen op directly.
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        global_input = torch.randn(4, 8, 16, device=self.device_type)
        idx = torch.tensor([1, 2], device=self.device_type)
        global_value = torch.randn(2, 8, 16, device=self.device_type)

        ref = torch.ops.aten.index_put_.default(
            global_input.clone(), [idx, None, None], global_value
        )

        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])
        input_dt = distribute_tensor(global_input.clone(), device_mesh, [Shard(1)])
        torch.ops.aten.index_put_.default(input_dt, [idx_dt, None, None], value_dt)
        self.assertEqual(input_dt.full_tensor(), ref)

    def test_index_put_noncontiguous_indexed_dims(self):
        # [FUNCTION DOC] aten index_put with non-contiguous indexed dims (indices idx0, None,
        # idx2) on a 4-D input sharded on non-indexed dim 1 with replicated indices and values;
        # compares with plain.
        """Test index_put with non-contiguous indexed dims (None gaps).

        For non-contiguous indexed dims like [idx, None, idx2], advanced
        indexing moves the broadcast shape to the front of the output:
        output = (*broadcast_shape, *non_indexed_dim_sizes).
        """
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # self[4, 8, 3, 16], index dims 0 and 2 (non-contiguous)
        global_input = torch.randn(4, 8, 3, 16, device=self.device_type)
        idx0 = torch.tensor([0, 1], device=self.device_type)
        idx2 = torch.tensor([2, 0], device=self.device_type)
        # output shape = (2, 8, 16), values must be broadcastable to that
        global_value = torch.randn(2, 8, 16, device=self.device_type)

        ref = torch.ops.aten.index_put.default(
            global_input, [idx0, None, idx2], global_value
        )

        idx0_dt = distribute_tensor(idx0, device_mesh, [Replicate()])
        idx2_dt = distribute_tensor(idx2, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        # Shard self on non-indexed dim 1 (size 8)
        input_dt = distribute_tensor(global_input, device_mesh, [Shard(1)])
        output_dt = torch.ops.aten.index_put.default(
            input_dt, [idx0_dt, None, idx2_dt], value_dt
        )
        self.assertEqual(output_dt.full_tensor(), ref)

    def test_index_put_mid_dim_indexed(self):
        # [FUNCTION DOC] aten index_put with the indexed dim (dim 2) in the
        # middle; shards the input on each non-indexed dim (0, 1, 3) with
        # replicated index and values and compares with plain.
        """Test index_put when indexed dims are in the middle of the tensor."""
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # self[16, 8, 4, 16]: index dim 2, non-indexed dims = [0, 1, 3]
        global_input = torch.randn(16, 8, 4, 16, device=self.device_type)
        idx = torch.tensor([1], device=self.device_type)
        # values shape = [16, 8, 1, 16] (broadcast replaces dim 2 in-place)
        global_value = torch.randn(16, 8, 1, 16, device=self.device_type)

        ref = torch.ops.aten.index_put.default(
            global_input, [None, None, idx], global_value
        )

        idx_dt = distribute_tensor(idx, device_mesh, [Replicate()])
        value_dt = distribute_tensor(global_value, device_mesh, [Replicate()])

        # Shard self on each non-indexed dim
        for shard_dim in [0, 1, 3]:
            input_dt = distribute_tensor(global_input, device_mesh, [Shard(shard_dim)])
            output_dt = torch.ops.aten.index_put.default(
                input_dt, [None, None, idx_dt], value_dt
            )
            self.assertEqual(output_dt.full_tensor(), ref)

    def test_where_type_promotion(self):
        # [FUNCTION DOC] Checks torch.where with scalar branches (type promotion to int) on a
        # mask from a Shard(0) or Replicate DTensor equals the plain result.
        mesh = self.build_device_mesh()  # 1D mesh

        specs = [[Shard(0)], [Replicate()]]
        for spec in specs:
            global_tensor = torch.randn(12, 8)
            mat = distribute_tensor(global_tensor, mesh, spec)
            res = torch.where(mat > 0, 1, 0)
            ref = torch.where(global_tensor > 0, 1, 0)
            self.assertEqual(res.full_tensor(), ref)

    def test_dtensor_dtype_conversion(self):
        # [FUNCTION DOC] Converts a bfloat16 Shard(0) DTensor to float and back
        # with type_as, checking dtypes of the DTensor and local tensor. Clears
        # the fast-path sharding prop cache first and expects 0 hits and 2 misses
        # after the conversions, then 1 hit after repeating float().
        from torch.distributed.tensor.debug import (
            _clear_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )

        _clear_sharding_prop_cache()
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        # by default we start from bf16 dtype
        local_tensor = torch.randn(2, 8, dtype=torch.bfloat16)
        bf16_sharded_dtensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(bf16_sharded_dtensor.dtype, torch.bfloat16)
        self.assertEqual(bf16_sharded_dtensor.to_local().dtype, torch.bfloat16)

        # convert to float dtype
        fp32_sharded_dtensor = bf16_sharded_dtensor.float()
        self.assertEqual(fp32_sharded_dtensor.dtype, torch.float32)
        self.assertEqual(fp32_sharded_dtensor.to_local().dtype, torch.float32)

        # convert back to bf16 dtype
        bf16_sharded_dtensor1 = fp32_sharded_dtensor.type_as(bf16_sharded_dtensor)
        self.assertEqual(bf16_sharded_dtensor1.dtype, torch.bfloat16)
        self.assertEqual(bf16_sharded_dtensor1.to_local().dtype, torch.bfloat16)

        # by this point we only have cache misses
        hits, misses = _get_fast_path_sharding_prop_cache_stats()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 2)

        # convert to fp32 again and see if there's cache hit
        bf16_sharded_dtensor1.float()
        hits, misses = _get_fast_path_sharding_prop_cache_stats()
        # by now we should have cache hit
        self.assertEqual(hits, 1)
        self.assertEqual(misses, 2)

    def test_single_dim_strategy_dtype_cache_key(self):
        # [FUNCTION DOC] Registers a temporary single-dim strategy for aten._to_copy with schema
        # info marking dtype as a static kwarg key (patching the propagator's dicts), then casts
        # a DTensor to int32 and float64. Expects the strategy function to be called twice,
        # showing dtype is part of the cache key.
        """Test that schema_info from single-dim strategy affects cache key.

        When @register_single_dim_strategy specifies static_kwargkey=["dtype"],
        the C++ dispatch path should include dtype in the cache key. This ensures
        calls with different dtypes don't return the same cached result.
        """
        from unittest.mock import patch

        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
            _SingleDimStrategyInfo,
        )
        from torch.distributed.tensor.debug import _clear_sharding_prop_cache

        call_count = [0]

        def to_copy_single_dim_strategy(op, args_schema, kwargs_schema):
            # [FUNCTION DOC] HELPER: Fake single-dim strategy for _to_copy: counts calls in
            # call_count and returns, for each tensor dim, a strategy list pairing the same
            # sharding placeholder for output and input. Raises AssertionError if the first arg
            # is not TensorMeta.
            call_count[0] += 1
            self_meta = args_schema[0]
            if not isinstance(self_meta, TensorMeta):
                raise AssertionError(f"Expected TensorMeta, got {type(self_meta)}")
            single_dim_strategies = []
            for dim in range(len(self_meta.shape)):
                single_dim_strategies.append(
                    [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
                )
            return single_dim_strategies

        _clear_sharding_prop_cache()
        mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(2, 8, dtype=torch.float32)
        sharded_dtensor = DTensor.from_local(local_tensor, mesh, shard_spec)

        propagator = DTensor._op_dispatcher.sharding_propagator
        op = torch.ops.aten._to_copy.default
        schema_info = RuntimeSchemaInfo(static_kwargkey=["dtype"])

        with (
            patch.dict(
                propagator.op_single_dim_strategy_funcs,
                {op: _SingleDimStrategyInfo(func=to_copy_single_dim_strategy)},
            ),
            patch.dict(
                propagator.op_to_schema_info_for_single_dim_strategy, {op: schema_info}
            ),
            patch.dict(propagator.op_strategy_funcs, clear=True),
            patch.dict(propagator.op_to_schema_info, clear=True),
        ):
            call_count[0] = 0
            sharded_dtensor.to(torch.int32)
            sharded_dtensor.to(torch.float64)

            # With dtype in cache key, strategy should be called twice (different dtypes)
            self.assertEqual(call_count[0], 2)

    def test_slice(self):
        # [FUNCTION DOC] Slices a Shard(1) DTensor with [:, 8:], expecting one collective in the
        # forward and one in the backward (checked with CommDebugMode), and that the full output
        # and the gradient equal the plain results.
        mesh = self.build_device_mesh()  # 1D mesh
        comm_mode = CommDebugMode()

        shard_spec = [Shard(1)]
        global_tensor = torch.randn(8, 16, requires_grad=True)
        sharded_dtensor = distribute_tensor(global_tensor, mesh, shard_spec)

        global_out = global_tensor[:, 8:]
        with comm_mode:
            sharded_out = sharded_dtensor[:, 8:]

        self.assertEqual(comm_mode.get_total_counts(), 1)

        global_out.backward(gradient=torch.ones_like(global_out))
        with comm_mode:
            sharded_out_grad = torch.distributed.tensor.ones(
                sharded_out.shape, device_mesh=mesh, placements=shard_spec
            )
            sharded_out.backward(gradient=sharded_out_grad)

        self.assertEqual(comm_mode.get_total_counts(), 1)

        self.assertEqual(sharded_out.full_tensor(), global_out)
        self.assertEqual(sharded_dtensor.grad.full_tensor(), global_tensor.grad)

    def test_slice_full_size_on_sharded_dim(self):
        # [FUNCTION DOC] Regression test for issue 170427: slices of a Shard(1) DTensor,
        # including a full-size and beyond-size slice, which must work (they may use aten.alias)
        # and match plain.
        """
        Test for the issue #170427 where slicing with a size that equals or
        exceeds the full dimension size should work correctly on sharded
        dimensions.

        So when slicing [:, :N] where N >= dim_size on a tensor sharded on that
        dimension, the operation may be optimized to use aten.alias.default,
        which must have a proper sharding strategy registered.
        """
        mesh = self.build_device_mesh()

        global_tensor = torch.randn(2, 4)
        sharded_dtensor = distribute_tensor(global_tensor, mesh, [Shard(1)])

        result1 = sharded_dtensor[:, :2]  # partial slice
        self.assertEqual(result1.full_tensor(), global_tensor[:, :2])

        result2 = sharded_dtensor[:, 2:]  # partial slice from middle
        self.assertEqual(result2.full_tensor(), global_tensor[:, 2:])

        # This used to fail with: NotImplementedError: Operator aten.alias.default
        # does not have a sharding strategy registered
        result3 = sharded_dtensor[:, :4]  # full dimension slice
        self.assertEqual(result3.full_tensor(), global_tensor[:, :4])

        result4 = sharded_dtensor[:, :8]  # beyond dimension size
        self.assertEqual(result4.full_tensor(), global_tensor[:, :8])

        result5 = sharded_dtensor[:2, :]  # full slice on dim 0
        self.assertEqual(result5.full_tensor(), global_tensor[:2, :])

    def test_split_on_partial(self):
        # [FUNCTION DOC] Runs _test_split_on_partial over the product of reduce_op (sum, avg,
        # min, max), split_size (2, 3, 4) and split_dim (0, 1) via run_subtests.
        self.run_subtests(
            {
                "reduce_op": ["sum", "avg", "min", "max"],
                "split_size": [2, 3, 4],
                "split_dim": [0, 1],
            },
            self._test_split_on_partial,
        )

    def _test_split_on_partial(self, reduce_op: str, split_size: int, split_dim: int):
        # [FUNCTION DOC] HELPER: Seeds per-rank RNG, builds a Partial(reduce_op)
        # DTensor from an 8x8 local tensor and runs _test_op_on_dtensor with
        # torch.split with the given size and dim.
        self.init_manual_seed_for_rank()
        mesh = self.build_device_mesh()

        partial_tensor = torch.randn(8, 8, device=self.device_type)
        partial_dt = DTensor.from_local(
            local_tensor=partial_tensor,
            device_mesh=mesh,
            placements=[Partial(reduce_op=reduce_op)],
        )
        self._test_op_on_dtensor(
            torch.split,
            partial_dt,
            split_size,
            dim=split_dim,
        )

    def test_unbind(self):
        # [FUNCTION DOC] Unbinds a Shard(shard_dim) DTensor over each unbind dim.
        # When the dims are equal it expects a RuntimeError containing 'Sharding
        # propagation failed'; otherwise outputs are sharded on the shifted dim
        # and each equals the plain unbind result.
        device_mesh = self.build_device_mesh()
        shard_dims = [0, 1]
        unbind_dims = [0, 1]
        local_tensor = torch.randn(4, 8, requires_grad=True)
        for shard_dim, unbind_dim in itertools.product(shard_dims, unbind_dims):
            dist_tensor = distribute_tensor(
                local_tensor, device_mesh, (Shard(shard_dim),)
            )

            if shard_dim == unbind_dim:
                with self.assertRaisesRegex(
                    RuntimeError, "Sharding propagation failed"
                ):
                    dist_tensor.unbind(dim=unbind_dim)
            else:
                unbinded_dist_tensors = dist_tensor.unbind(dim=unbind_dim)
                new_shard_dim = shard_dim if shard_dim < unbind_dim else shard_dim - 1
                self.assertTrue(
                    all(
                        elem.placements[0].is_shard(dim=new_shard_dim)
                        for elem in unbinded_dist_tensors
                    )
                )
                for x, y in zip(
                    unbinded_dist_tensors, local_tensor.unbind(dim=unbind_dim)
                ):
                    self.assertEqual(x.full_tensor(), y)

    @with_comms
    def test_select_scatter(self):
        # [FUNCTION DOC] select_scatter on DTensors for dims 0,1,2 with the input and source
        # sharded on every dim other than the selected one; result matches plain and keeps the
        # input shard dim.
        device_mesh = self.build_device_mesh()
        inp = torch.randn(8, 4, 6, device=self.device_type)
        # select_scatter(self, src, dim, index): insert src at self[:, :, index]
        for dim, idx in [(0, 2), (1, 1), (2, 3)]:
            src_shape = list(inp.shape)
            src_shape.pop(dim)
            src = torch.randn(src_shape, device=self.device_type)
            expected = inp.select_scatter(src, dim, idx)

            for shard_dim in range(inp.ndim):
                if shard_dim == dim:
                    continue
                dt_inp = distribute_tensor(inp, device_mesh, [Shard(shard_dim)])
                src_shard_dim = shard_dim if shard_dim < dim else shard_dim - 1
                dt_src = distribute_tensor(src, device_mesh, [Shard(src_shard_dim)])
                result = dt_inp.select_scatter(dt_src, dim, idx)
                self.assertEqual(result.full_tensor(), expected)
                self.assertTrue(result.placements[0].is_shard(shard_dim))

    @with_comms
    def test_diagonal_scatter(self):
        # [FUNCTION DOC] diagonal_scatter over dims 1 and 2 with input and source sharded on
        # batch dim 0; result matches plain and stays Shard(0).
        device_mesh = self.build_device_mesh()
        inp = torch.randn(8, 4, 6, device=self.device_type)

        # diagonal_scatter(self, src, offset, dim1, dim2)
        src = torch.randn(8, 4, device=self.device_type)
        expected = inp.diagonal_scatter(src, 0, 1, 2)

        # shard on dim 0 (batch), which is not one of the diagonal dims
        dt_inp = distribute_tensor(inp, device_mesh, [Shard(0)])
        dt_src = distribute_tensor(src, device_mesh, [Shard(0)])
        result = dt_inp.diagonal_scatter(dt_src, 0, 1, 2)
        self.assertEqual(result.full_tensor(), expected)
        self.assertTrue(result.placements[0].is_shard(0))


class DistBucketizeTest(LocalDTensorTestBase):
    # [CLASS DOC] Tests of torch.bucketize on DTensors using LocalTensor mode (single process).
    @with_comms
    def test_bucketize_partial_input(self):
        # [FUNCTION DOC] Checks Partial inputs to bucketize with replicated boundaries: sum and
        # avg are redistributed to Replicate, while max and min stay Partial with the same
        # reduce_op; values are compared with plain bucketize.
        # Bucketize is non-linear, so Partial("sum")/Partial("avg") inputs
        # must be converted to Replicate. But bucketize is monotone, so
        # Partial("max") and Partial("min") can propagate directly.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.tensor(
                [[2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 5.0, 9.0]],
                device=self.device_type,
            )

            # Non-linear reductions: must redistribute to Replicate
            for reduce_op in ("sum", "avg"):
                partial_input = DTensor.from_local(
                    input_tensor, mesh, [Partial(reduce_op)]
                )
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(partial_input, dist_boundaries)

                self.assertTrue(
                    result.placements[0].is_replicate(),
                    lambda msg: f"{msg}\nExpected Replicate output but got {result.placements[0]} "
                    f"for Partial({reduce_op}) input",
                )
                global_input = partial_input.full_tensor()
                expected = torch.bucketize(global_input, boundaries)
                self.assertEqual(result.to_local(), expected)

            # Monotone reductions: output inherits the same partial type
            for reduce_op in ("max", "min"):
                partial_input = DTensor.from_local(
                    input_tensor, mesh, [Partial(reduce_op)]
                )
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(partial_input, dist_boundaries)

                self.assertTrue(
                    result.placements[0].is_partial(),
                    lambda msg: f"{msg}\nExpected Partial output but got {result.placements[0]} "
                    f"for Partial({reduce_op}) input",
                )
                self.assertEqual(
                    result.placements[0].reduce_op,
                    reduce_op,
                    lambda msg: f"{msg}\nExpected Partial({reduce_op}) output but got {result.placements[0]}",
                )
                expected = torch.bucketize(input_tensor, boundaries)
                self.assertEqual(result.full_tensor(), expected)

    @with_comms
    def test_bucketize_sharded_input(self):
        # [FUNCTION DOC] Checks bucketize of a sharded input (dim 0 and 1) with replicated
        # boundaries keeps the sharding on the output and matches plain.
        # Sharded inputs should propagate sharding to output normally.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.randn(8, 4, device=self.device_type)
            expected = torch.bucketize(input_tensor, boundaries)

            for shard_dim in range(2):
                dist_input = distribute_tensor(input_tensor, mesh, [Shard(shard_dim)])
                dist_boundaries = distribute_tensor(boundaries, mesh, [Replicate()])
                result = torch.bucketize(dist_input, dist_boundaries)

                self.assertTrue(result.placements[0].is_shard(shard_dim))
                self.assertEqual(result.full_tensor(), expected)

    @with_comms
    def test_bucketize_sharded_boundaries(self):
        # [FUNCTION DOC] Checks bucketize of a replicated input with Shard(0) boundaries gives
        # the plain result.
        # When boundaries are sharded on dim 0, each rank counts how many of
        # its local boundary values each input exceeds. The sum across ranks
        # (Partial("sum")) gives the correct global bucket index.
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0], device=self.device_type)
            input_tensor = torch.tensor(
                [[2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 5.0, 9.0]],
                device=self.device_type,
            )
            expected = torch.bucketize(input_tensor, boundaries)

            dist_input = distribute_tensor(input_tensor, mesh, [Replicate()])
            dist_boundaries = distribute_tensor(boundaries, mesh, [Shard(0)])
            result = torch.bucketize(dist_input, dist_boundaries)
            self.assertEqual(result.full_tensor(), expected)


class DistToCopyTest(LocalDTensorTestBase):
    # [CLASS DOC] Tests of the _to_copy cast on Partial DTensors using LocalTensor mode.
    @with_comms
    def test_to_copy_partial_reduces_for_nonlinear_cast(self):
        # [FUNCTION DOC] For a table of (reduce_op, dtype, expected) casts of a Partial DTensor,
        # checks the result stays Partial with the same reduce_op when the cast is safe (float
        # to float for sum, monotonic int cast for max) and is Replicate otherwise (sum to int
        # or bool, max to bool).
        # (reduce_op, target_dtype, expect_partial)
        cases = [
            ("sum", torch.int32, False),  # truncation breaks additivity
            ("sum", torch.bool, False),  # thresholding
            ("sum", torch.float64, True),  # float→float is safe
            ("max", torch.int32, True),  # monotonic
            ("max", torch.bool, False),  # thresholding
        ]
        with LocalTensorMode(ranks=self.world_size):
            mesh = self.build_device_mesh()
            input_tensor = torch.randn(4, 4, device=self.device_type)
            for reduce_op, target_dtype, expect_partial in cases:
                dt = DTensor.from_local(input_tensor, mesh, [Partial(reduce_op)])
                result = dt.to(target_dtype)
                p = result.placements[0]
                if expect_partial:
                    self.assertTrue(
                        p.is_partial(),
                        lambda msg: f"{msg}\n{reduce_op}→{target_dtype}: {p}",
                    )
                    self.assertEqual(p.reduce_op, reduce_op)
                else:
                    self.assertTrue(
                        p.is_replicate(),
                        lambda msg: f"{msg}\n{reduce_op}→{target_dtype}: {p}",
                    )


class DistArgMaxArgMinTest(DTensorContinuousTestBase):
    # [CLASS DOC] Tests argmax and argmin on a 2x2 mesh with world size 4. Holds the sample 8x8
    # int data, the ops under test and the list of placements pairs used.
    world_size = 4
    _ops = [torch.argmax, torch.argmin]
    sample = [
        [0, 2, 1, 11, 5, 9, -2, -23],
        [3, 5, 7, 9, 0, -1, 4, 2],
        [8, 4, 6, -5, -10, 12, 7, 1],
        [13, 6, 9, -5, 0, 4, 2, 8],
        [4, 9, 2, 1, -6, -3, 5, 7],
        [0, -4, -2, 8, 6, 3, 12, -7],
        [20, 6, -3, 1, -8, 4, 2, 0],
        [5, 9, 11, -1, -4, 2, 3, 8],
    ]
    placements_tuples = (
        [Partial(), Shard(1)],
        [Partial(), Shard(0)],
        [Shard(0), Shard(1)],
        [Replicate(), Shard(0)],
        [Replicate(), Shard(1)],
    )

    @skip_if_lt_x_gpu(4)
    def test_argmax_argmin_with_placements(self):
        # [FUNCTION DOC] Requires 4 GPUs. For each placements pair (including Partial) and for
        # argmax and argmin over dim 1, checks the DTensor full result equals the plain result.
        device_mesh = self.build_device_mesh()
        local_tensor = torch.tensor(self.sample, device=self.device_type)
        for placements in self.placements_tuples:
            dtensor_input = distribute_tensor(local_tensor, device_mesh, placements)
            for op in self._ops:
                d_result = op(dtensor_input, dim=1)
                full_dresult = d_result.full_tensor()
                local_result = op(local_tensor, dim=1)
                self.assertEqual(full_dresult, local_result)

    def test_argmax_argmin_sharded_reduction_dim(self):
        # [FUNCTION DOC] On a 1-D mesh with a Shard(0) tensor, checks argmax/argmin over the
        # sharded dim 0 and over all elements match plain results (inputs are redistributed to
        # Replicate, no Partial output).
        """Unlike max/min which use reduction_linear=True and produce
        Partial("max")/Partial("min") outputs, argmax/argmin return indices
        that can't be combined across shards with an element-wise max/min.
        The strategy sets reduction_linear=False, which forces the input to
        be redistributed to Replicate on the sharded reduction dim before
        the op runs. No Partial placement appears in the output."""
        mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.tensor(self.sample, device=self.device_type, dtype=torch.float)
        dtensor = distribute_tensor(tensor, mesh, [Shard(0)])

        for op in self._ops:
            self.assertEqual(op(dtensor, dim=0).full_tensor(), op(tensor, dim=0))
            self.assertEqual(op(dtensor).full_tensor(), op(tensor))

    def build_device_mesh(self):
        # [FUNCTION DOC] Returns a 2x2 device mesh, overriding the 1-D mesh default.
        return init_device_mesh(self.device_type, (2, 2))


DistArgMaxArgMinTestWithLocalTensor = create_local_tensor_test_class(
    DistArgMaxArgMinTest,
    base_class=LocalDTensorContinuousTestBase,
)

instantiate_parametrized_tests(DistTensorOpsTest)
DistTensorOpsTestWithLocalTensor = create_local_tensor_test_class(
    DistTensorOpsTest,
    base_class=LocalDTensorContinuousTestBase,
)


@torch.library.custom_op("testlib::optional_clamp_op", mutates_args=())
def optional_clamp_op(
    x: torch.Tensor, lo: torch.Tensor | None, hi: torch.Tensor | None
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Custom op testlib::optional_clamp_op; returns clamp of x with
    # optional lo and hi tensor bounds.
    return torch.clamp(x, lo, hi)


@optional_clamp_op.register_fake
def _optional_clamp_op_fake(x, lo, hi):
    # [FUNCTION DOC] HELPER: Fake (meta) kernel for optional_clamp_op; returns an
    # empty tensor like x.
    return torch.empty_like(x)


@torch.library.custom_op("testlib::modified_cat_op", mutates_args=())
def modified_cat_op(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Custom op testlib::modified_cat_op; concatenates list a along dim 0
    # and ignores list b (b only exists to test caching of two list args).
    return torch.cat(a, dim=0)


@modified_cat_op.register_fake
def _modified_cat_op_fake(a, b):
    # [FUNCTION DOC] HELPER: Fake kernel for modified_cat_op; returns an empty tensor whose rows
    # are the sum of the first dims of list a and whose other dims, dtype and device come from
    # the first tensor of a.
    total_rows = sum(t.shape[0] for t in a)
    return torch.empty(
        total_rows, *a[0].shape[1:], dtype=a[0].dtype, device=a[0].device
    )


class DistTensorCppPyTree(DTensorContinuousTestBase):
    # [CLASS DOC] Tests the C++ fast-path sharding propagation cache with pytree-style (list and
    # optional tensor) arguments, using hit/miss statistics from the debug module.
    def test_cpp_cat(self):
        # [FUNCTION DOC] For three cat cases (Shard(1), Replicate with different sizes, Shard(0)
        # on the cat dim giving Replicate), clears the cache, checks result placement and
        # values, and that a second identical cat is a cache hit (hits equals 1).
        from torch.distributed.tensor.debug import (
            _clear_fast_path_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )

        mesh = self.build_device_mesh()
        cases = [
            ((8, 8), (8, 8), 0, [Shard(1)], (Shard(1),)),
            ((4, 8), (6, 8), 0, [Replicate()], (Replicate(),)),
            ((8, 8), (8, 8), 0, [Shard(0)], (Replicate(),)),
        ]
        for shape_a, shape_b, cat_dim, placements, expected_placements in cases:
            _clear_fast_path_sharding_prop_cache()
            global_a = torch.randn(shape_a)
            global_b = torch.randn(shape_b)
            dt_a = distribute_tensor(global_a, mesh, placements)
            dt_b = distribute_tensor(global_b, mesh, placements)
            # first call: cache miss
            result = torch.cat([dt_a, dt_b], dim=cat_dim)
            self.assertEqual(result.placements, expected_placements)
            self.assertEqual(
                result.full_tensor(), torch.cat([global_a, global_b], dim=cat_dim)
            )
            # second call: should be a cache hit
            result = torch.cat([dt_a, dt_b], dim=cat_dim)
            hits, _ = _get_fast_path_sharding_prop_cache_stats()
            self.assertEqual(hits, 1)
            self.assertEqual(result.placements, expected_placements)
            self.assertEqual(
                result.full_tensor(), torch.cat([global_a, global_b], dim=cat_dim)
            )

    def test_foreach_mixed_dtensor_and_tensor(self):
        # [FUNCTION DOC] Under implicit_replication, runs _foreach_add_ on a list of a DTensor
        # and a regular tensor twice; checks both are updated, the first call is a cache miss
        # and the second a hit.
        from torch.distributed.tensor.debug import (
            _clear_fast_path_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )
        from torch.distributed.tensor.experimental import implicit_replication

        mesh = self.build_device_mesh()
        with implicit_replication():
            _clear_fast_path_sharding_prop_cache()
            dt = distribute_tensor(
                torch.zeros(4, 4, device=self.device_type), mesh, [Shard(0)]
            )
            regular = torch.zeros(4, 4, device=self.device_type)

            torch._foreach_add_([dt, regular], 1)
            hits, misses = _get_fast_path_sharding_prop_cache_stats()
            self.assertEqual(misses, 1)
            self.assertEqual(
                dt.full_tensor(), torch.ones(4, 4, device=self.device_type)
            )
            self.assertEqual(regular, torch.ones(4, 4, device=self.device_type))

            torch._foreach_add_([dt, regular], 1)
            hits, misses = _get_fast_path_sharding_prop_cache_stats()
            self.assertEqual(hits, 1)

    def test_two_list_op_cache_collision(self):
        # [FUNCTION DOC] Registers replicate_op_strategy for modified_cat_op with pytree-aware
        # schema info, calls it with two different list-length combinations, and expects two
        # misses and no hits (no cache collision) and an output shape of 8x8.
        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops.utils import replicate_op_strategy
        from torch.distributed.tensor.debug import (
            _clear_fast_path_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )

        mesh = self.build_device_mesh()
        op = torch.ops.testlib.modified_cat_op

        with op_strategy_context(
            op.default,
            replicate_op_strategy,
            schema_info=RuntimeSchemaInfo(needs_pytree=True),
        ):
            _clear_fast_path_sharding_prop_cache()
            a = distribute_tensor(torch.randn(4, 8), mesh, [Replicate()])
            b = distribute_tensor(torch.randn(4, 8), mesh, [Replicate()])

            op([a], [b])  # cache miss, populates cache
            result = op([a, b], [])  # different list sizes, should also miss
            hits, misses = _get_fast_path_sharding_prop_cache_stats()
            self.assertEqual(hits, 0)
            self.assertEqual(misses, 2)
            self.assertEqual(result.shape, torch.Size([8, 8]))

    def test_optional_tensor_cache_key(self):
        # [FUNCTION DOC] Regression test: the cache key must distinguish None in different
        # positions of optional tensor args. Calls optional_clamp_op with None in the last two
        # positions and expects 0 hits and 2 misses.
        """Cache must distinguish op(t1, None, t2) from op(t1, t2, None).

        Uses a custom op with high static_argnum so that None args for
        optional Tensor? parameters would be dropped from the cache key
        without the is_none_or_undefined fix in handle_non_dtensor_arg.
        With same-spec DTensors the collision is deterministic.
        """
        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops.utils import replicate_op_strategy
        from torch.distributed.tensor.debug import (
            _clear_fast_path_sharding_prop_cache,
            _get_fast_path_sharding_prop_cache_stats,
        )

        mesh = self.build_device_mesh()
        op = torch.ops.testlib.optional_clamp_op

        with op_strategy_context(
            op.default,
            replicate_op_strategy,
            schema_info=RuntimeSchemaInfo(static_argnum=3),
        ):
            _clear_fast_path_sharding_prop_cache()
            a = distribute_tensor(torch.randn(4, 8), mesh, [Shard(0)])

            op(a, a, None)  # miss 1: key should include None at position 2
            op(a, None, a)  # miss 2: key should include None at position 1

            hits, misses = _get_fast_path_sharding_prop_cache_stats()
            self.assertEqual(hits, 0)
            self.assertEqual(misses, 2)

    def test_optional_tensor_cache_key_python_slow_path(self):
        # [FUNCTION DOC] Builds two OpSchema objects differing only in the position of a None
        # arg, recomputes their comparison keys, and checks their hashes and equality differ
        # (Python slow path).
        """Same as above but exercises the Python slow path via OpSchema directly.

        DTensor_OpSchema_recompute_comparison_key_impl must include None
        args for optional Tensor? parameters in the comparison key.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
        from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo

        mesh = self.build_device_mesh()
        spec = DTensorSpec(mesh, (Shard(0),))
        op = torch.ops.testlib.optional_clamp_op.default

        schema1 = OpSchema(op, (spec, None, spec), {})
        schema1.schema_info = RuntimeSchemaInfo(static_argnum=3)
        schema1._recompute_comparison_key()

        schema2 = OpSchema(op, (spec, spec, None), {})
        schema2.schema_info = RuntimeSchemaInfo(static_argnum=3)
        schema2._recompute_comparison_key()

        self.assertNotEqual(hash(schema1), hash(schema2))
        self.assertNotEqual(schema1, schema2)


class TestNewEmptyStridedUneven(DTensorTestBase):
    # [CLASS DOC] Tests new_empty/new_empty_strided behavior and autograd on unevenly sharded
    # DTensors, using the standard (not local-tensor) multi-process test base.
    @with_comms
    def test_backward_no_allgather(self):
        # [FUNCTION DOC] Regression test for issue 107661: backward through to_local of an
        # unevenly sharded Shard(1) DTensor causes no communication and gives a contiguous grad
        # with the same placement and value 2.
        """Backward on unevenly-sharded DTensor should not allgather (issue #107661)."""
        mesh = self.build_device_mesh()
        placement = (Shard(1),)
        x = torch.randn(12, self.world_size * 2 + 1, requires_grad=True)
        dt = distribute_tensor(x, mesh, placement)
        comm_mode = CommDebugMode()
        with comm_mode:
            (dt.to_local() * 2).sum().backward()
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(dt.grad.placements, placement)
        self.assertTrue(dt.grad._local_tensor.is_contiguous())
        self.assertEqual(
            dt.grad._local_tensor,
            torch.full_like(dt.grad._local_tensor, 2.0),
        )

    @with_comms
    def test_new_empty_propagates_partial(self):
        # [FUNCTION DOC] Checks new_empty and new_empty_strided on a Partial DTensor keep the
        # Partial placement, while new_ones gives Replicate (issue 180486).
        """new_empty/new_empty_strided on a Partial DTensor should inherit Partial.

        Uninitialized memory will be overwritten immediately, so the placement
        only needs to match the source of the subsequent write. Without this,
        copy_ from a Partial source to a Replicate destination triggers an
        unwanted all-reduce (issue #180486).
        """
        mesh = self.build_device_mesh()
        partial_dt = DTensor.from_local(
            torch.randn(4, 8, device=self.device_type),
            device_mesh=mesh,
            placements=[Partial()],
        )

        empty_dt = partial_dt.new_empty(partial_dt.shape)
        self.assertEqual(empty_dt.placements, (Partial(),))

        empty_strided_dt = partial_dt.new_empty_strided(
            partial_dt.shape, partial_dt.stride()
        )
        self.assertEqual(empty_strided_dt.placements, (Partial(),))

        # Initialized factories must keep Replicate, otherwise their values
        # would be incorrect after a Partial reduction (e.g. ones * world_size).
        ones_dt = partial_dt.new_ones(partial_dt.shape)
        self.assertEqual(ones_dt.placements, (Replicate(),))

    @with_comms
    def test_backward_partial_grad_with_transpose(self):
        # [FUNCTION DOC] Regression test (issue 180486): a Replicate DTensor parameter used with
        # to_local(grad_placements=Partial) and a transpose, so the gradient is non-contiguous;
        # a post-accumulate-grad hook records the grad placements, which must all be Partial.
        """Backward preserves Partial placement when grad is non-contiguous (issue #180486).

        When a Replicate DTensor parameter is used with to_local(grad_placements=[Partial()]),
        and the backward produces a non-contiguous gradient (e.g. via transpose), autograd's
        layout invariant calls new_empty_strided + copy_ to fix strides. This must preserve
        the Partial placement rather than defaulting to Replicate.
        """
        mesh = self.build_device_mesh()

        class _Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 4x8x8 weight parameter that records the
            # placements of its gradient in _grad_placement via a post-accumulate hook.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the random weight parameter and
                # _grad_placement (None).
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 8, 8))
                self._grad_placement = None

            def forward(self, x):
                # [FUNCTION DOC] HELPER: If the weight is a DTensor, converts it to local with
                # Partial grad placements and registers a hook that stores the grad placements.
                # Then transposes and makes contiguous, and returns the matrix product of x with
                # the first slice.
                w = self.weight
                if isinstance(w, DTensor):
                    w = w.to_local(grad_placements=[Partial()])

                    def _capture(param, model=self):
                        # [FUNCTION DOC] HELPER: Hook that stores the DTensor grad
                        # placements on the model.
                        if param.grad is not None and isinstance(param.grad, DTensor):
                            model._grad_placement = param.grad.placements

                    self.weight.register_post_accumulate_grad_hook(_capture)

                # transpose backward produces non-contiguous grad
                w = w.transpose(1, 2).contiguous()
                return torch.mm(x, w[0])

        model = _Model().to(self.device_type)
        with torch.no_grad():
            model.weight = torch.nn.Parameter(
                DTensor.from_local(
                    model.weight.data,
                    device_mesh=mesh,
                    placements=[Replicate()],
                )
            )

        x = torch.randn(2, 8, device=self.device_type, requires_grad=True)
        out = model(x)
        out.sum().backward()

        self.assertIsNotNone(model._grad_placement)
        self.assertTrue(
            all(isinstance(p, Partial) for p in model._grad_placement),
            lambda msg: f"{msg}\nExpected Partial grad placement, got {model._grad_placement}",
        )

    @with_comms
    def test_backward_channels_last(self):
        # [FUNCTION DOC] Backward through to_local of an unevenly sharded
        # channels-last Shard(2) DTensor: no communication, the grad keeps the
        # placement, channels-last strides and value 2.
        """Backward preserves channels-last stride order for unevenly-sharded DTensor."""
        mesh = self.build_device_mesh()
        placement = (Shard(2),)
        # H=2*world_size+1 ensures uneven sharding on dim 2
        x = torch.randn(2, 3, self.world_size * 2 + 1, 4).to(
            memory_format=torch.channels_last
        )
        x.requires_grad_(True)
        dt = distribute_tensor(x, mesh, placement)
        comm_mode = CommDebugMode()
        with comm_mode:
            (dt.to_local() * 2).sum().backward()
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(dt.grad.placements, placement)
        self.assertTrue(
            dt.grad._local_tensor.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertEqual(
            dt.grad._local_tensor,
            torch.full_like(dt.grad._local_tensor, 2.0),
        )


if __name__ == "__main__":
    run_tests()
