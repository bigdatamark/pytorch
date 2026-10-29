# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Unit tests for the DTensor sharding propagation helpers einop_rule and
# pointwise_rule in torch.distributed.tensor._ops._common_rules. Tests build DTensorSpec
# objects from dimension maps on a device mesh, wrap them in an OpSchema for an aten op,
# and check the resulting output spec or the suggested redistribute schema. Runs under a
# 4-rank DTensorContinuousTestBase; no import-time side effects beyond defining aten alias.
import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._ops._common_rules import einop_rule, pointwise_rule
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
)


aten = torch.ops.aten


class CommonRulesTest(DTensorContinuousTestBase):
    # [CLASS DOC] Test case for einop_rule and pointwise_rule sharding propagation. Fixed
    # at world_size 4 so that both 1-D meshes (4 ranks) and 2x2 meshes can be built.
    # Each test constructs specs and asserts on output_spec / redistribute_schema only;
    # no tensors are communicated.
    # hard code world size to 4 as we need to test
    # at least with 2d mesh
    world_size = 4

    def _gen_tensor_meta(self, shape):
        # [FUNCTION DOC] HELPER: Allocates an empty CPU tensor of the given shape and returns
        # a TensorMeta holding its shape, stride and dtype (default float32).
        empty_tensor = torch.empty(shape)
        return TensorMeta(
            empty_tensor.shape,
            empty_tensor.stride(),
            empty_tensor.dtype,
        )

    def test_einop_basic_propagation(self):
        # [FUNCTION DOC] Checks einop_rule on a matmul equation over a 1-D mesh: sharding on
        # the n dim of mat2 yields output dim map [-1, 0], sharding on the m dim of mat1
        # yields [0, -1], and sharding the contracted k dim on both inputs yields a
        # partial placement on the output.
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        mm_call = aten.mm.default
        # propagate col-wise sharding
        mat1, mat2 = [-1, -1], [-1, 0]

        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # propagate row-wise sharding
        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # generate partial
        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertTrue(output_spec.placements[0].is_partial())

    def test_einop_pointwise_propagation(self):
        # [FUNCTION DOC] Checks einop_rule with pointwise-style equations: identical specs
        # keep their dim map, a lower-rank operand broadcasts against a higher-rank one
        # (dim map of the larger operand wins), and a size-1 dim labeled "1" in the
        # equation broadcasts to the common shape.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        add_call = aten.add.Tensor
        # addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema(add_call, (mat1_spec, mat1_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # broadcast addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [-1, 0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )

        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([2]))
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

        # broadcast to a common shape
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, [0, -1, -1], [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1, -1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,1k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1, -1])

    def test_einop_merge_sharding(self):
        # [FUNCTION DOC] On a 2x2 mesh, mat1 sharded on m (mesh dim 0) and mat2 sharded on
        # n (mesh dim 1) merge into an output sharded as dim map [0, 1].
        # 2d mesh einop merge sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, 1])

    def test_einop_linearity(self):
        # [FUNCTION DOC] Tests the linearity flag of einop_rule with a partial input on mesh
        # dim 1. Without linearity there is no output spec and the suggested schema
        # redistributes input 0 to a non-partial placement. With linearity=True (mm and a
        # pointwise add) there is also no output spec, and the suggestion converts the
        # second input's mesh dim 1 placement to partial.
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # if not turn on linearity, partial sum is not eligible to propagate, we return
        # suggestion to reshard inputs with no partial sum (i.e. all_reduce one input)
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        suggested_spec = suggestions.args_schema[0]
        self.assertFalse(suggested_spec.placements[1].is_partial())

        # einop prop with linearity on mm, should give back suggestion
        # on converting placements to partial
        output_sharding = einop_rule(
            "mk,kn->mn",
            OpSchema(mm_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

        # einop prop with linearity on point-wise, should give back suggestion
        # on converting placements to partial
        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        output_sharding = einop_rule(
            "ij,ij->ij",
            OpSchema(add_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

    def test_einop_multi_sharding_on_mesh_dim(self):
        # [FUNCTION DOC] On a 1-D mesh both mm operands are sharded on their first dim over
        # the same mesh dim, which cannot be propagated: expects no output spec and a
        # redistribute suggestion keeping mat1 as [0, -1] and replicating mat2.
        # einop prop with multi sharding on same mesh dim
        mesh_shape = torch.arange(self.world_size)
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 12]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the second
        # arg by all_gather its tensor dim sharding
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [0, -1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, [-1, -1])

    def test_einop_errors(self):
        # [FUNCTION DOC] Verifies einop_rule raises RuntimeError mentioning "sharded two
        # different ways" when the same equation dim is sharded on different mesh dims in
        # the two add operands.
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        with self.assertRaisesRegex(RuntimeError, "sharded two different ways:"):
            einop_rule("ij,ij->ij", OpSchema(add_call, (mat1_spec, mat2_spec), {}))

    def test_pointwise_rules_broadcasting(self):
        # [FUNCTION DOC] pointwise_rule on where with a 1-D sharded condition, a scalar-shaped
        # self and a 1x1 other: broadcasting to rank 2 gives output dim map [-1, 0].
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        where_call = aten.where.self
        inp1, inp2, inp3 = [0], [], [-1, -1]
        inp1_tensor_meta = self._gen_tensor_meta(torch.Size([8]))
        inp2_tensor_meta = self._gen_tensor_meta(torch.Size([]))
        inp3_tensor_meta = self._gen_tensor_meta(torch.Size([1, 1]))
        condition = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=inp1_tensor_meta
        )
        self_tensor = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=inp2_tensor_meta
        )
        other_tensor = DTensorSpec.from_dim_map(
            mesh, inp3, [], tensor_meta=inp3_tensor_meta
        )
        # propagate point-wise sharding with broadcasting
        output_sharding = pointwise_rule(
            OpSchema(where_call, (condition, self_tensor, other_tensor), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

    def test_pointwise_rules_suggestion(self):
        # [FUNCTION DOC] pointwise_rule on lerp with mismatched shardings and a trailing
        # non-spec scalar argument: no output spec, and the redistribute schema keeps all
        # three positional args, with the scalar -1 unchanged in position 2.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        lerp_call = aten.lerp.Scalar
        # propagate point-wise sharding
        inp1, inp2 = [-1, -1], [-1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=mat2_tensor_meta
        )
        # adding a positional argument -1 to arg schema
        output_sharding = pointwise_rule(
            OpSchema(lerp_call, (mat1_spec, mat2_spec, -1), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion from pointwise rules still have
        # the positional args that are not DTensorSpec
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(len(schema_suggestion.args_schema), 3)
        self.assertEqual(schema_suggestion.args_schema[2], -1)

    def test_pointwise_multi_sharding_on_mesh_dim(self):
        # [FUNCTION DOC] pointwise_rule on a 2x2 mesh. A rank-2 and rank-1 add align by
        # broadcasting and yield dim map [-1, 0]. A second case where two tensor dims map
        # to conflicting mesh dims yields no output spec; the suggestion un-shards the
        # first input's dim 0 (giving [-1, -1, -1, 1]) and leaves the second input as is.
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add.Tensor

        # basic case to test implicit broadcasting shape alignment
        mat1, mat2 = [-1, 0], [0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([20, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, -1, 1], [0, -1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 1, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the first
        # arg by all_gather first tensor dim sharding
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [-1, -1, -1, 1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat2)

    def test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim(self):
        # [FUNCTION DOC] pointwise_rule on the in-place add_ overload over a 2x2 mesh with
        # conflicting shardings: no output spec; the suggestion keeps the first (mutated)
        # argument's sharding and redistributes the second input to match it.
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add_.Tensor

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, 1], [-1, -1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the second
        # arg as we should enforce the sharding of the first arg
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, mat1)
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat1)


if __name__ == "__main__":
    run_tests()
