# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the experimental register_sharding decorator, which installs a custom
# DTensor sharding strategy for an operator. Covers softmax, argmax, ops with tensor keyword
# outputs, and a custom op whose first argument is not a tensor. No import-time side effects
# beyond defining the aten alias.
import itertools

import torch
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.experimental import register_sharding
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


aten = torch.ops.aten


class TestRegisterSharding(DTensorTestBase):
    # [CLASS DOC] Multi-process DTensor test case for register_sharding. Tests register
    # strategies globally, so tearDown removes the custom ops they create.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown, then destroys every entry in the custom op
        # global registry whose key starts with test_dtensor:: and removes the test_dtensor
        # namespace from torch.ops if present. Mutates global registry state.
        super().tearDown()
        # Clean up any custom ops registered during tests to avoid test pollution
        # This addresses the concern about proper deregistration of custom ops
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if key.startswith("test_dtensor::"):
                torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, "test_dtensor"):
            delattr(torch.ops, "test_dtensor")

    @with_comms
    def test_softmax_fwd(self):
        # [FUNCTION DOC] Removes the built-in softmax strategy, registers a custom one, checks
        # the derived schema info has one static arg, then runs softmax on a 3-D tensor sharded
        # on each dim for several softmax dims. Expects a replicated result when sharded dim
        # equals the softmax dim, otherwise sharding preserved; both match the local result.
        # After registering the custom softmax sharding strategy,
        # the original entry would have been replaced.
        # The following line is for showcasing purpose only.
        DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs.pop(
            aten._softmax.default, None
        )

        @register_sharding(aten._softmax.default)
        def custom_softmax_sharding(
            x: DTensorSpec,
            dim: int,
            half_to_float: torch.dtype,
        ):
            # [FUNCTION DOC] HELPER: Strategy for softmax. Normalizes a negative dim, then
            # returns a list of (output placements, input placements) options: all replicate,
            # plus sharding on every dim other than the softmax dim. Non-tensor args get None.
            softmax_dim = dim if dim >= 0 else dim + x.ndim

            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            for sharding_dim in range(x.ndim):
                if sharding_dim != softmax_dim:
                    all_sharded = (
                        [Shard(sharding_dim)],
                        [Shard(sharding_dim), None, None],
                    )
                    acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten._softmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1]
        shard_dims = [0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = torch.nn.functional.softmax(
                x, dim=softmax_dim, dtype=torch.float32
            )
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            dist_y = torch.nn.functional.softmax(
                dist_x, dim=softmax_dim, dtype=torch.float32
            )
            if dims[shard_dim] == dims[softmax_dim]:
                self.assertTrue(dist_y.placements[0].is_replicate())
                self.assertEqual(dist_y.to_local(), local_y)
            else:
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                self.assertEqual(dist_y.full_tensor(), local_y)

    @with_comms
    def test_argmax(self):
        # [FUNCTION DOC] Registers a custom argmax strategy, checks the schema info marks one
        # static arg even though the dim argument is optional, then runs argmax along dim 1
        # with keepdim on an input sharded on dim 0. Expects the output to stay sharded on
        # dim 0 and to match the local result.
        @register_sharding(aten.argmax.default)
        def custom_argmax_sharding(x, dim, keepdim):
            # [FUNCTION DOC] HELPER: Strategy for argmax. Always allows full replication; when
            # keepdim is set also allows sharding on any dim other than the reduced dim.
            acceptable_shardings = []

            all_replicate = ([Replicate()], [Replicate(), None, None])
            acceptable_shardings.append(all_replicate)

            if keepdim:
                for sharding_dim in range(x.ndim):
                    if sharding_dim != dim:
                        all_sharded = (
                            [Shard(sharding_dim)],
                            [Shard(sharding_dim), None, None],
                        )
                        acceptable_shardings.append(all_sharded)

            return acceptable_shardings

        # check if the RuntimeSchemaInfo is derived correctly
        # when the first int arg is optional
        schema_info = DTensor._op_dispatcher.sharding_propagator.op_to_schema_info[
            aten.argmax.default
        ]
        self.assertEqual(schema_info.static_argnum, 1)

        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, device=self.device_type)
        dist_x = distribute_tensor(x, device_mesh, [Shard(0)])

        local_y = torch.argmax(x, dim=1, keepdim=True)
        dist_y = torch.argmax(dist_x, dim=1, keepdim=True)

        self.assertTrue(dist_y.placements[0].is_shard(dim=0))
        self.assertEqual(dist_y.full_tensor(), local_y)

    @with_comms
    def test_register_sharding_for_tensor_kwargs(self):
        # [FUNCTION DOC] Registers a replicate-only strategy for min.dim_min (tensor out
        # arguments passed as keywords) and calls min with replicated DTensor out tensors.
        # Expects DTensor results whose full tensors equal the local min values and indices.
        mesh = self.build_device_mesh()
        x = torch.randn(4, 4, device=self.device_type)
        x_dt = distribute_tensor(x, mesh, [Replicate()])

        @register_sharding(aten.min.dim_min)
        def min_dim_strategy(x, dim, keepdim, min, min_indices):
            # [FUNCTION DOC] HELPER: Strategy offering one option only: both outputs and the
            # tensor inputs (x, min, min_indices) replicated, scalar args given None.
            all_replicate = (
                [Replicate(), Replicate()],
                [Replicate(), None, None, Replicate(), Replicate()],
            )
            return [all_replicate]

        value = torch.randn(4, 1, device=self.device_type)
        indices = torch.randn(4, 1, device=self.device_type).long()
        value_dt = distribute_tensor(value, mesh, [Replicate()])
        indices_dt = distribute_tensor(indices, mesh, [Replicate()])

        result = torch.min(x_dt, dim=1, keepdim=True, out=(value_dt, indices_dt))

        self.assertIsInstance(result[0], DTensor)
        self.assertIsInstance(result[1], DTensor)

        expected_values, expected_indices = torch.min(x, dim=1, keepdim=True)
        self.assertEqual(result[0].full_tensor(), expected_values)
        self.assertEqual(result[1].full_tensor(), expected_indices)

    @with_comms
    def test_register_sharding_non_tensor_first_arg(self):
        # [FUNCTION DOC] Regression test for issue 167915: defines a custom op taking a string
        # then a tensor, registers a replicate-only strategy, and calls it with a replicated
        # DTensor. Expects mesh lookup to succeed despite the non-tensor first arg and a
        # DTensor result equal to the input. Registers a custom op (cleaned in tearDown).
        # Test that get_mesh_from_args works when the first arg is not a DTensor
        # Regression test for https://github.com/pytorch/pytorch/issues/167915
        @torch.library.custom_op("test_dtensor::scalar_first_arg", mutates_args=())
        def scalar_first_arg(name: str, x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op body; ignores the name and returns a clone of x.
            return x.clone()

        @scalar_first_arg.register_fake
        def _(name, x):
            # [FUNCTION DOC] HELPER: Fake (meta) implementation; returns a clone of x.
            return x.clone()

        @register_sharding(torch.ops.test_dtensor.scalar_first_arg.default)
        def scalar_first_arg_sharding(name, x):
            # [FUNCTION DOC] HELPER: Strategy returning a single option: replicated output and
            # replicated tensor input, with None for the string argument.
            return [([Replicate()], [None, Replicate()])]

        mesh = self.build_device_mesh()
        x = torch.randn(4, 4, device=self.device_type)
        x_dt = distribute_tensor(x, mesh, [Replicate()])

        # This should not raise "Cannot find device mesh from args"
        result = torch.ops.test_dtensor.scalar_first_arg("test", x_dt)

        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.full_tensor(), x)


if __name__ == "__main__":
    run_tests()
