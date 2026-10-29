# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for DTensor operator sharding strategies: einsum dimension parsing and
# strategy generation, the redistribute cost model, reduction/norm/loss single-dim
# strategy helpers, OpSpec/OpSchema meta properties, custom strategy registration through
# op_strategy_context, and the error for ops without a registered strategy. Import time
# side effects: registers custom ops in the mylib and testlib namespaces (with autograd
# or fake implementations) and creates local-tensor variants of two test classes.

import itertools
import random
from itertools import chain
from unittest.mock import patch

import numpy as np

import torch
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops._einsum_strategy import (
    EinsumDims,
    gen_einsum_strategies,
)
from torch.distributed.tensor._ops._math_ops import common_reduction_strategy
from torch.distributed.tensor._ops.utils import replicate_op_strategy
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorOpTestBase,
    DTensorTestBase,
    op_strategy_context,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


try:
    from torch.utils._cxx_pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_leaves  # type: ignore[no-redef]


def extract_tensor_meta(t) -> TensorMeta:
    # HELPER: Builds a TensorMeta from a real tensor, using its shape, stride and dtype.
    return TensorMeta(t.shape, t.stride(), t.dtype)


class TestEinsumDims(TestCase):
    # [CLASS DOC] Pure-Python tests of the einsum equation parser: checks how EinsumDims
    # classifies labels into batch, contracting, lhs-only and rhs-only output dimensions.
    def test_batch_dims(self):
        # [FUNCTION DOC] For an equation where all labels appear in both inputs and the output,
        # all labels are batch dims and the other three groups are empty.
        equation = "abc,abc->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b", "c"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, [])
        self.assertEqual(edims.rhs_out_only_dims, [])

    def test_mm_dims(self):
        # [FUNCTION DOC] Matrix-multiply equation: k is contracting, m is lhs-only, n is rhs-only,
        # and there are no batch dims.
        equation = "mk,kn->mn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, [])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_bmm_dims(self):
        # [FUNCTION DOC] Batched matmul equations with one and two leading batch labels: the
        # batch dims are the shared leading labels; k, m and n classify as in a plain matmul.
        equation = "bmk,bkn->bmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

        equation = "bcmk,bckn->bcmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b", "c"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_free_dims(self):
        # [FUNCTION DOC] Equations where the second input has fewer dims: labels in both inputs
        # and the output are batch dims, and labels present only in one input and the output
        # become that side's output-only dims, including an unusual output label ordering.
        equation = "abc,ab->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["c"])
        self.assertEqual(edims.rhs_out_only_dims, [])

        equation = "abd,bf->abfd"  # codespell:ignore
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["a", "d"])
        self.assertEqual(edims.rhs_out_only_dims, ["f"])


class TestEinsumStrategies(DTensorOpTestBase):
    # [CLASS DOC] Tests that gen_einsum_strategies enumerates the expected number of
    # sharding strategies on a 4-rank device mesh (1D and 2D) for several equations.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property returning the world size of 4 ranks for this test class.
        return 4

    def test_mm_1d_mesh(self):
        # [FUNCTION DOC] Matmul on a 1D mesh yields 4 strategies.
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        self.assertEqual(len(all_strats.strategies), 4)

    def test_mm_2d_mesh(self):
        # [FUNCTION DOC] Matmul on a 2x2 mesh yields 16 strategies.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        all_strats = gen_einsum_strategies("mk,kn->mn", mesh)
        self.assertEqual(len(all_strats.strategies), 16)

    def test_bmm_1d_mesh(self):
        # [FUNCTION DOC] Batched matmul on a 1D mesh yields 5 strategies.
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 5)

    def test_bmm_diffinndim_2d_mesh(self):
        # [FUNCTION DOC] Equation whose lhs has an extra batch dim relative to the rhs, on a 2x2
        # mesh, yields 25 strategies.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        all_strats = gen_einsum_strategies("bmk,kn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 25)

    def test_bmm_diffoutndim_2d_mesh(self):
        # [FUNCTION DOC] Equation with a lower-rank output than the lhs (matrix-vector style),
        # on a 2x2 mesh, yields 16 strategies.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))
        all_strats = gen_einsum_strategies("bmk,k->bm", mesh)
        self.assertEqual(len(all_strats.strategies), 16)

    def test_bmm_2d_mesh(self):
        # [FUNCTION DOC] Batched matmul on a 2x2 mesh yields 25 strategies.
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        all_strats = gen_einsum_strategies("bmk,bkn->bmn", mesh)
        self.assertEqual(len(all_strats.strategies), 25)

    def test_pointwise_1d_mesh(self):
        # [FUNCTION DOC] Pointwise equations on a 1D mesh, with and without a broadcast operand,
        # each yield 5 strategies.
        mesh = self.build_device_mesh()

        simple_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh)
        self.assertEqual(len(simple_strats.strategies), 5)

        broadcast_strats = gen_einsum_strategies("bcd,abcd->abcd", mesh)
        self.assertEqual(len(broadcast_strats.strategies), 5)

    def test_linearity_1d_mesh(self):
        # [FUNCTION DOC] With the linearity option enabled, the pointwise equation on a 1D mesh
        # yields 6 strategies.
        mesh = self.build_device_mesh()

        all_strats = gen_einsum_strategies("abcd,abcd->abcd", mesh, linearity=True)
        self.assertEqual(len(all_strats.strategies), 6)


class TestCostModel(DTensorOpTestBase):
    # [CLASS DOC] Tests of redistribute_cost and strategy selection on a 4-rank mesh:
    # relative costs of collectives, infinite cost for unsupported transitions, and that
    # sharding propagation finds no-redistribution strategies for mm, bmm and t.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] HELPER: Property returning the world size of 4 ranks for this test class.
        return 4

    def test_redistribute_cost_mesh_1d(self):
        # [FUNCTION DOC] On a 1D mesh checks that redistributing a spec to itself costs 0, that
        # shard-to-replicate equals partial-to-shard, that partial-to-replicate plus one is below
        # the sum of those two costs, and that shard-to-partial has infinite cost.
        mesh_1d = self.build_device_mesh()
        shard_placement = (Shard(0),)
        replica_placement = (Replicate(),)
        partial_placement = (Partial(),)

        global_tensor = torch.randn(10, 10)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        # shard spec
        shard_spec = DTensorSpec(mesh_1d, shard_placement, global_tensor_meta)
        # replica spec
        replica_spec = DTensorSpec(mesh_1d, replica_placement, global_tensor_meta)
        # partial spec
        partial_spec = DTensorSpec(mesh_1d, partial_placement, global_tensor_meta)

        # make sure reshard cost is 0 for the same spec redistribute
        for spec in [shard_spec, replica_spec, partial_spec]:
            cost = redistribute_cost(spec, spec)
            self.assertEqual(cost, 0)

        # shard -> replicate
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # partial -> shard
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        # partial -> replicate
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        self.assertEqual(allgather_cost, reduce_scatter_cost)
        self.assertTrue(allreduce_cost + 1 < allgather_cost + reduce_scatter_cost)
        # shard to partial
        cost = redistribute_cost(shard_spec, partial_spec)
        self.assertEqual(cost, float("inf"))

    def test_redistribute_cost_strided_shard(self):
        # [FUNCTION DOC] A _StridedShard source spec to replicate has infinite cost, while a
        # replicated source to a _StridedShard target costs 0.0 (replicated short-circuit).
        """_StridedShard specs get inf cost (shard_order is None bail-out)."""
        mesh_1d = self.build_device_mesh()

        global_tensor_meta = extract_tensor_meta(torch.randn(10, 10))

        strided_shard_spec = DTensorSpec(
            mesh_1d, (_StridedShard(0, split_factor=2),), global_tensor_meta
        )
        replica_spec = DTensorSpec(mesh_1d, (Replicate(),), global_tensor_meta)

        # _StridedShard as source: shard_order is None → inf
        self.assertEqual(
            redistribute_cost(strided_shard_spec, replica_spec), float("inf")
        )
        # Replicate as source: is_replicated() short-circuits to 0 before
        # reaching the shard_order check
        self.assertEqual(redistribute_cost(replica_spec, strided_shard_spec), 0.0)

    def test_addmm_partial_redistribute(self):
        # [FUNCTION DOC] Runs addmm on a sharded bias, Partial mat1 and column-sharded mat2
        # across 4 CPU ranks. Inputs come from a seeded local Generator so all ranks hold identical
        # global values. Asserts the output placement is Shard(1) and that its full tensor equals
        # the plain addmm of the global tensors.
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        # bias/mat1/mat2 are the GLOBAL (unsharded) tensors, distributed below;
        # the final assert checks full_tensor() against the global addmm. Every
        # rank must therefore draw identical values: distribute_tensor broadcasts
        # from rank 0, and the Partial() mat1 reconstruction sums mat1/world_size
        # across ranks, which only recovers the global mat1 if all ranks match.
        # This class is a MultiThreadedTestCase, so ranks are threads sharing the
        # process-global default generator; a per-thread local Generator gives
        # each rank its own state seeded identically, avoiding interleaved draws.
        gen = torch.Generator().manual_seed(0)
        bias = torch.randn(8, generator=gen)
        mat1 = torch.randn(50, 6, generator=gen)
        mat2 = torch.randn(6, 8, generator=gen)

        dist_bias = distribute_tensor(bias, mesh, [Shard(0)])
        dist_mat1 = DTensor.from_local(
            mat1 / self.world_size,
            mesh,
            [Partial()],
            run_check=False,
        )
        dist_mat2 = distribute_tensor(mat2, mesh, [Shard(1)])

        dist_out = torch.addmm(dist_bias, dist_mat1, dist_mat2)

        self.assertEqual(dist_out.placements, (Shard(1),))
        self.assertEqual(dist_out.full_tensor(), torch.addmm(bias, mat1, mat2))

    def test_redistribute_cost_mesh_2d(self):
        # [FUNCTION DOC] On a 2x2 mesh checks same-spec cost is 0 and that partial-to-replicate
        # costs more than both shard-to-replicate and partial-to-shard.
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        shard_placement = (Shard(0), Shard(0))
        replica_placement = (Replicate(), Replicate())
        partial_placement = (Partial(), Partial())

        global_tensor = torch.randn(8, 8)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        # shard spec
        shard_spec = DTensorSpec(mesh_2d, shard_placement, global_tensor_meta)
        # replica spec
        replica_spec = DTensorSpec(mesh_2d, replica_placement, global_tensor_meta)
        # partial spec
        partial_spec = DTensorSpec(mesh_2d, partial_placement, global_tensor_meta)

        # make sure reshard cost is 0 for the same spec redistribute
        for spec in [shard_spec, replica_spec, partial_spec]:
            cost = redistribute_cost(spec, spec)
            self.assertEqual(cost, 0)

        # shard -> replicate
        allgather_cost = redistribute_cost(shard_spec, replica_spec)
        # partial -> replicate
        allreduce_cost = redistribute_cost(partial_spec, replica_spec)
        # partial -> shard
        reduce_scatter_cost = redistribute_cost(partial_spec, shard_spec)
        self.assertTrue(allreduce_cost > allgather_cost)
        self.assertTrue(allreduce_cost > reduce_scatter_cost)

    def test_mm_strategies(self):
        # [FUNCTION DOC] For four lhs/rhs placement combinations of mm on a 1D mesh, builds an
        # OpSchema and runs non-cached sharding propagation; asserts no redistribution is needed.
        mesh = self.build_device_mesh()
        lhs_tensor = torch.randn(6, 8)
        rhs_tensor = torch.randn(8, 12)
        lhs_tensor_meta = extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = extract_tensor_meta(rhs_tensor)

        mm_combs = (
            (Shard(0), Replicate()),
            (Replicate(), Shard(1)),
            (Shard(1), Shard(0)),
            (Replicate(), Replicate()),
        )
        for lhs, rhs in mm_combs:
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            op_schema = OpSchema(
                torch.ops.aten.mm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            # test sharding prop
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            self.assertFalse(output_sharding.needs_redistribute)

    def test_t_prunes_unproven_unbacked_strided_shard_candidates(self):
        # [FUNCTION DOC] Uses a fake tensor mode with an unbacked symbolic split factor to check
        # that t's strategy (allow_unbacked_sharding false) does not consider the unproven
        # _StridedShard on dim 1 shardable, while propagation from _StridedShard on dim 0 succeeds
        # without redistribution and produces a _StridedShard on dim 1 with the same split factor.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.distributed.tensor._ops.utils import is_tensor_shardable
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        mesh = DeviceMesh("cpu", torch.arange(2), _init_backend=False, _rank=0)
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
        strategy_info = (
            DTensor._op_dispatcher.sharding_propagator.op_single_dim_strategy_funcs[
                torch.ops.aten.t.default
            ]
        )
        self.assertFalse(strategy_info.allow_unbacked_sharding)

        with fake_mode:
            split_factor = fake_mode.shape_env.create_unbacked_symint()
            torch._dynamo.override_optimization_hint(split_factor, 8)
            input_meta = TensorMeta(
                torch.Size([2048 * split_factor, 8]),
                (8, 1),
                torch.float32,
            )

            unproven_input_spec = DTensorSpec(
                mesh,
                (_StridedShard(1, split_factor=split_factor),),
                input_meta,
            )
            self.assertFalse(
                is_tensor_shardable(
                    input_meta.shape,
                    unproven_input_spec,
                    allow_unbacked_sharding=strategy_info.allow_unbacked_sharding,
                )
            )

            input_spec = DTensorSpec(
                mesh,
                (_StridedShard(0, split_factor=split_factor),),
                input_meta,
            )
            op_schema = OpSchema(torch.ops.aten.t.default, (input_spec,), {})
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )

        self.assertFalse(output_sharding.needs_redistribute)
        self.assertEqual(
            output_sharding.output_spec.placements,
            (_StridedShard(1, split_factor=split_factor),),
        )

    def test_bmm_strategies(self):
        # [FUNCTION DOC] Same as the mm test but for bmm with five placement combinations:
        # non-cached sharding propagation must need no redistribution for each.
        mesh = self.build_device_mesh()
        lhs_tensor = torch.randn(8, 6, 8)
        rhs_tensor = torch.randn(8, 8, 12)
        lhs_tensor_meta = extract_tensor_meta(lhs_tensor)
        rhs_tensor_meta = extract_tensor_meta(rhs_tensor)

        bmm_combs = (
            (Shard(0), Shard(0)),
            (Shard(1), Replicate()),
            (Replicate(), Shard(2)),
            (Shard(2), Shard(1)),
            (Replicate(), Replicate()),
        )
        for lhs, rhs in bmm_combs:
            lhs_spec = DTensorSpec(mesh, (lhs,), lhs_tensor_meta)
            rhs_spec = DTensorSpec(mesh, (rhs,), rhs_tensor_meta)

            op_schema = OpSchema(
                torch.ops.aten.bmm.default,
                (lhs_spec, rhs_spec),
                {},
            )
            # test sharding prop
            output_sharding = DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding_non_cached(
                op_schema
            )
            self.assertFalse(output_sharding.needs_redistribute)

    def test_redistribute_cost_partial_to_different_partial_is_infinite(self):
        # [FUNCTION DOC] Asserts redistributing from a sum Partial to an avg Partial on a 1D mesh
        # has infinite cost.
        """Test that redistributing from Partial("sum") to Partial("avg") has infinite cost.

        Converting between different Partial types (e.g., sum -> avg) is not supported,
        so the redistribute cost should be infinite to prevent this strategy from being chosen.
        """
        mesh_1d = self.build_device_mesh()
        global_tensor = torch.randn(10, 10)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        partial_sum_spec = DTensorSpec(mesh_1d, (Partial("sum"),), global_tensor_meta)
        partial_avg_spec = DTensorSpec(mesh_1d, (Partial("avg"),), global_tensor_meta)

        # Cost should be infinite since converting between different Partial types is unsupported
        cost = redistribute_cost(partial_sum_spec, partial_avg_spec)
        self.assertEqual(cost, float("inf"))

    def test_redistribute_cost_with_order(self):
        # [FUNCTION DOC] On a 2x2 mesh, compares the cost of gathering over mesh dim 0 versus
        # mesh dim 1 from a doubly-sharded source; asserts the dim 0 cost is strictly greater.
        mesh_2d = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )

        # Source: Shard on dim 0 across all three mesh dimensions
        source_placement = (Shard(0), Shard(0))

        # Target: Replicate on first mesh dimension, shard on others
        # This requires 2 allgathers, one on dim=0 and one on dim=1
        replicate_mesh_dim0 = (Replicate(), Shard(0))

        # Target: Replicate on second mesh dimension, shard on others
        # This requires 1 allgather on dim=1
        replicate_mesh_dim1 = (Shard(0), Replicate())

        global_tensor = torch.randn(4, 4)
        global_tensor_meta = extract_tensor_meta(global_tensor)

        source_spec = DTensorSpec(mesh_2d, source_placement, global_tensor_meta)
        target_spec_dim0 = DTensorSpec(mesh_2d, replicate_mesh_dim0, global_tensor_meta)
        target_spec_dim1 = DTensorSpec(mesh_2d, replicate_mesh_dim1, global_tensor_meta)

        # Calculate costs for allgather on each mesh dimension
        cost_mesh_dim0 = redistribute_cost(source_spec, target_spec_dim0)
        cost_mesh_dim1 = redistribute_cost(source_spec, target_spec_dim1)

        # Cost increases with earlier mesh dimensions due to the way
        # mesh dimensions are ordered (outer to inner in device hierarchy)
        self.assertGreater(cost_mesh_dim0, cost_mesh_dim1)


class TestReductionStrategy(TestCase):
    # [CLASS DOC] Tests common_reduction_strategy. Each test initializes a fake process
    # group with 4 ranks in setUp and destroys it in tearDown.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets world_size to 4 and initializes the fake process group
        # backend (rank 0) with a FakeStore.
        super().setUp()
        self.world_size = 4
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown and destroys the default process group.
        super().tearDown()
        torch.distributed.destroy_process_group()

    def _make_spec(self, mesh: DeviceMesh, placements):
        # [FUNCTION DOC] HELPER: Builds a DTensorSpec for an 8x8 float32 tensor on the given
        # mesh with the given placements.
        tensor_meta = TensorMeta(
            shape=torch.Size([8, 8]), stride=(8, 1), dtype=torch.float32
        )
        return DTensorSpec(mesh, tuple(placements), tensor_meta)

    def test_common_reduction_strategy_uses_per_strategy_linearity(self):
        # [FUNCTION DOC] Feeds a strategy with a Partial avg spec and a Shard/Shard spec into
        # the sum reduction over dim 0, and checks the second result keeps its input placements and
        # outputs Partial sum on mesh dim 0 and Shard(0) on mesh dim 1; i.e. linearity is judged
        # per input strategy rather than from the first one.
        mesh = DeviceMesh("cpu", torch.arange(self.world_size).reshape(2, 2))
        partial_avg_spec = self._make_spec(mesh, (Partial("avg"), Replicate()))
        sharded_spec = self._make_spec(mesh, (Shard(0), Shard(1)))

        reduction_strategy = common_reduction_strategy(
            OpStrategy([OpSpec(partial_avg_spec), OpSpec(sharded_spec)]),
            [0],
            reduction_op="sum",
        )

        sharded_reduction_spec = reduction_strategy.strategies[1]
        self.assertEqual(
            sharded_reduction_spec.input_specs[0].placements, (Shard(0), Shard(1))
        )
        self.assertEqual(
            sharded_reduction_spec.output_spec.placements,
            (Partial("sum"), Shard(0)),
        )


# -------------Test op strategy registration-------------
# custom op without List[Tensor] as input
# reference: https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd
@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # HELPER: Custom op body: computes elementwise sin of x plus sin of y through numpy on
    # CPU and returns the result on x's device.
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    out_np = np.sin(x_np) + np.sin(y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_context(ctx, inputs, output):
    # HELPER: Autograd setup_context for the custom op: saves both inputs for backward.
    (x, y) = inputs
    ctx.save_for_backward(x, y)


def backward(ctx, grad):
    # HELPER: Autograd backward for the custom op: returns grad times cos of each input.
    (x, y) = ctx.saved_tensors
    return grad * x.cos(), grad * y.cos()


@numpy_sin.register_fake
def _fw(x, y):
    # HELPER: Fake implementation: returns an empty tensor shaped like x.
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_sin", backward, setup_context=setup_context
)


# custom op with List[Tensor] as input
@torch.library.custom_op("mylib::numpy_tuple_sin", mutates_args=())
def numpy_tuple_sin(
    x: torch.Tensor, y: list[torch.Tensor], z: torch.Tensor
) -> torch.Tensor:
    # HELPER: Custom op body taking a tensor, a list of tensors and a tensor: sums numpy sin
    # of all of them and returns the result on x's device.
    x_np = x.cpu().numpy()
    y_np = [i.cpu().numpy() for i in y]
    z_np = z.cpu().numpy()

    out_np = np.sin(x_np) + np.sin(z_np) + sum(np.sin(i) for i in y_np)
    return torch.from_numpy(out_np).to(device=x.device)


def setup_tuple_context(ctx, inputs, output):
    # HELPER: Autograd setup_context for the list-input custom op: saves x, the list and z.
    (x, y, z) = inputs
    ctx.save_for_backward(x, y, z)


def tuple_backward(ctx, grad):
    # HELPER: Autograd backward for the list-input op: returns grad times cos for x, a list of
    # the same for each y entry, and for z.
    (x, y, z) = ctx.saved_tensors
    return grad * x.cos(), [grad * i.cos() for i in y], grad * z.cos()


@numpy_tuple_sin.register_fake
def _fw_tuple(x, y, z):
    # HELPER: Fake implementation: returns an empty tensor shaped like x.
    return torch.empty_like(x)


torch.library.register_autograd(
    "mylib::numpy_tuple_sin", tuple_backward, setup_context=setup_tuple_context
)


def detect_exists_identical_opspec(*args, op, mesh, strategy_function) -> bool:
    # HELPER: Builds, for each tensor argument, every combination of Replicate/Shard(dim)
    # placements over the mesh dims (shuffled), wraps them as OpStrategy inputs, runs the
    # given strategy function under op_strategy_context and returns True when the string forms
    # of all produced OpSpecs are distinct (no duplicates).
    """
    Given sample input args, detect if identical OpSpecs exists under the same
    OpStrategy.

    """
    tree_args = tree_leaves(args)
    # metadata for each argument
    arg_tensor_metadata = [extract_tensor_meta(i) for i in args]
    # possible combination of placements for each arg
    arg_placement_comb = []
    for i in tree_args:
        if isinstance(i, torch.Tensor):
            # possible placement choice for argument i
            placement_choices = (Replicate(), *[Shard(i) for i in range(i.ndim)])
            # expand placement choice into full Placements for argument i
            arg_placement_comb.append(
                list(itertools.product(placement_choices, repeat=mesh.ndim))
            )
            random.shuffle(arg_placement_comb[-1])

    arg_opspec_list = []
    for idx, arg_placement in enumerate(arg_placement_comb):
        arg_opspec_list.append([])
        for placement in arg_placement:
            arg_opspec_list[idx].append(
                OpSpec(
                    output_specs=DTensorSpec(
                        mesh, placement, tensor_meta=arg_tensor_metadata[idx]
                    )
                )
            )

    op_schema = OpSchema(
        op,
        args_schema=(tuple(OpStrategy(i) for i in arg_opspec_list)),
        kwargs_schema={},
    )
    with op_strategy_context(op, strategy_function):
        output_strategy = strategy_function(op_schema)
        # OpSpec doesn't have hashing, convert to str to compare
        output_strategy_str_list = [
            str(j) for i in tree_leaves(output_strategy) for j in i.strategies
        ]
        return len(output_strategy_str_list) == len(set(output_strategy_str_list))


class DistTensorReplicateStrategyRegistrationTest(DTensorTestBase):
    # [CLASS DOC] Tests registering a replicate-everything strategy for custom ops via
    # op_strategy_context, for a plain tensor op and an op taking a list of tensors.
    @with_comms
    @patch("torch.distributed.tensor._sharding_prop._select_min_cost_strategy")
    def test_replicate_strategy_placement(self, mock_select_strategy):
        # [FUNCTION DOC] Patches the strategy selector to record per-input redistribute costs,
        # runs the custom op on 2D-mesh-sharded inputs under the replicate strategy and checks: the
        # number of all-gather calls equals world size, recorded costs equal the cost of going to
        # fully replicated, the result matches the non-distributed output, output placements are
        # replicated, and the strategy yields no duplicate OpSpecs.
        costs_from__select_strategy = []

        def mock_select_func(strategy, op_schema=None):
            # HELPER: Stand-in for the strategy selector: with one strategy it records and returns
            # its redistribute costs; otherwise appends each OpSpec cost (raising AssertionError if
            # unset) and returns the strategy with minimal total cost.
            """function copied from _select_strategy but with cost capturing"""
            nonlocal costs_from__select_strategy
            if len(strategy.strategies) == 1:
                costs_from__select_strategy = strategy.strategies[0].redistribute_cost
                return strategy.strategies[0]

            op_spec_costs: list[float] = []
            for op_spec in strategy.strategies:
                if op_spec.redistribute_cost is None:
                    raise AssertionError("must set redistribute cost each OpSpec!")
                costs_from__select_strategy.append(op_spec.redistribute_cost)
                redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
                op_spec_costs.append(redistribute_cost)
            return strategy.strategies[op_spec_costs.index(min(op_spec_costs))]

        mock_select_strategy.side_effect = mock_select_func
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        comm_mode = CommDebugMode()
        test_op = torch.ops.mylib.numpy_sin
        input_x = torch.randn([8, 16, 32], device=self.device_type)
        input_y = torch.randn([8, 16, 32], device=self.device_type)
        output = test_op(input_x, input_y)
        input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
        input_y_dt = distribute_tensor(input_y, mesh, [Shard(0), Shard(1)])
        x_spec = DTensorSpec(mesh, input_x_dt.placements, extract_tensor_meta(input_x))
        new_x_spec = DTensorSpec(
            mesh, (Replicate(), Replicate()), extract_tensor_meta(input_x)
        )
        y_spec = DTensorSpec(mesh, input_y_dt.placements, extract_tensor_meta(input_y))
        new_y_spec = DTensorSpec(
            mesh, (Replicate(), Replicate()), extract_tensor_meta(input_y)
        )
        with comm_mode:
            with op_strategy_context(test_op.default, replicate_op_strategy):
                output_dt = test_op(input_x_dt, input_y_dt)
                self.assertEqual(
                    comm_mode.get_comm_counts(),
                    {
                        torch.ops.c10d_functional.all_gather_into_tensor: self.world_size,
                    },
                )
                expected_cost = [
                    [redistribute_cost(x_spec, new_x_spec)],
                    [redistribute_cost(y_spec, new_y_spec)],
                ]
                self.assertEqual(expected_cost, costs_from__select_strategy)
                self.assertEqual(output_dt.full_tensor(), output)
                self.assertEqual(output_dt.placements, [Replicate(), Replicate()])
                self.assertTrue(
                    detect_exists_identical_opspec(
                        input_x,
                        input_y,
                        op=test_op.default,
                        mesh=mesh,
                        strategy_function=replicate_op_strategy,
                    )
                )

    @with_comms
    def test_tuple_replicate_strategy_placement(self):
        # [FUNCTION DOC] Registers the replicate strategy with needs_pytree schema info for the
        # list-input custom op, runs it on differently sharded DTensors and checks the full tensor
        # equals the local result with Replicate placements on both mesh dims.
        mesh = init_device_mesh(self.device_type, (2, self.world_size // 2))
        test_op = torch.ops.mylib.numpy_tuple_sin
        with op_strategy_context(
            test_op.default,
            replicate_op_strategy,
            schema_info=RuntimeSchemaInfo(needs_pytree=True),
        ):
            input_x = torch.randn([8, 16, 8], device=self.device_type)
            input_y = [
                torch.randn([8, 16, 8], device=self.device_type) for _ in range(3)
            ]
            input_z = torch.randn([8, 16, 8], device=self.device_type)
            output = test_op(input_x, input_y, input_z)
            input_x_dt = distribute_tensor(input_x, mesh, [Shard(0), Shard(1)])
            input_y_dt = [
                distribute_tensor(i, mesh, [Shard(1), Shard(1)]) for i in input_y
            ]
            input_z_dt = distribute_tensor(input_z, mesh, [Shard(1), Shard(0)])
            output_dt = test_op(input_x_dt, input_y_dt, input_z_dt)
            self.assertEqual(output_dt.full_tensor(), output)
            self.assertEqual(output_dt.placements, [Replicate(), Replicate()])


class TestStrategyHashing(DTensorTestBase):
    # [CLASS DOC] Tests that strategy caching distinguishes calls differing in non-tensor args.
    @with_comms
    def test_call_with_different_nontensor_args(self):
        # [FUNCTION DOC] Under the replicate strategy for sort, sorts a sharded DTensor along dim 0
        # and then dim 1 without supplying schema info, and again along dim 1 in a fresh context;
        # asserts the dim 1 results from both contexts agree.
        mesh = self.build_device_mesh()
        global_tensor = torch.tensor(
            [
                [29.0, 45.0, 3.0, 61.0],
                [25.0, 6.0, 21.0, 0.0],
                [1.0, 63.0, 49.0, 38.0],
                [48.0, 9.0, 55.0, 18.0],
            ]
        )
        shard_spec = [Shard(1)]
        sharded_dtensor = distribute_tensor(global_tensor, mesh, shard_spec)
        with op_strategy_context(torch.ops.aten.sort.default, replicate_op_strategy):
            # intentionally do not supply `schema_info=RuntimeSchemaInfo(1)`
            torch.sort(sharded_dtensor, dim=0)  # sort each column
            out1, _ = torch.sort(sharded_dtensor, dim=1)  # sort each row
        with op_strategy_context(torch.ops.aten.sort.default, replicate_op_strategy):
            out2, _ = torch.sort(sharded_dtensor, dim=1)
        self.assertEqual(out1.full_tensor(), out2.full_tensor())


class TestStrategyOperation(DTensorTestBase):
    # [CLASS DOC] Tests lifecycle of operator strategy registration, with a 2-rank world.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning the world size of 2 ranks for this test class.
        return 2

    @with_comms
    def test_cache_clean(self):
        # [FUNCTION DOC] Runs the custom op on DTensors inside op_strategy_context, then outside
        # it and expects NotImplementedError saying no sharding strategy is registered, showing the
        # registration and propagation cache were cleaned up.
        mesh = self.build_device_mesh()
        test_op = torch.ops.mylib.numpy_sin
        x = torch.randn(2, device=self.device_type)
        y = torch.randn(2, device=self.device_type)
        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Shard(0)])
        with op_strategy_context(test_op.default, replicate_op_strategy):
            self._test_op_on_dtensor(test_op, x_dt, y_dt)
        with self.assertRaisesRegex(
            NotImplementedError,
            f"Operator {test_op.default} does not have a sharding strategy registered",
        ):
            self._test_op_on_dtensor(test_op, x_dt, y_dt)


DistTensorReplicateStrategyRegistrationTestWithLocalTensor = (
    create_local_tensor_test_class(
        DistTensorReplicateStrategyRegistrationTest,
    )
)

TestStrategyHashingWithLocalTensor = create_local_tensor_test_class(
    TestStrategyHashing,
)


class TestOpSchemaMetaProperties(TestCase):
    # [CLASS DOC] Tests OpSchema meta properties and several single-dim strategy helper
    # functions, using a fake process group with 8 ranks set up per test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets world_size to 8 and initializes the fake process group
        # backend (rank 0) with a FakeStore.
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown and destroys the default process group.
        super().tearDown()
        torch.distributed.destroy_process_group()

    def test_args_meta_mixed_opstrategy_and_tuplestrategy(self):
        # [FUNCTION DOC] Builds an OpSchema whose positional args are an OpStrategy, a
        # TupleStrategy and an OpStrategy; checks args_meta has three entries: a TensorMeta,
        # a tuple of two TensorMetas, and a TensorMeta, with the expected shapes.
        """Test args_meta with both OpStrategy and TupleStrategy"""
        # Create a simple mesh
        mesh = DeviceMesh("cpu", torch.arange(4))

        # Create tensor metadata
        tensor_meta1 = TensorMeta(
            shape=torch.Size([10, 20]),
            stride=(20, 1),
            dtype=torch.float32,
        )
        tensor_meta2 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta3 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta4 = TensorMeta(
            shape=torch.Size([20, 30]),
            stride=(30, 1),
            dtype=torch.float32,
        )

        # Create OpStrategies
        op_strategy1 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(0),), tensor_meta1))]
        )
        op_strategy2 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta2))]
        )
        op_strategy3 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Replicate(),), tensor_meta3))]
        )
        op_strategy4 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta4))]
        )

        # Create TupleStrategy
        tuple_strategy = TupleStrategy([op_strategy2, op_strategy3])

        # Create OpSchema: (OpStrategy, TupleStrategy, OpStrategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema=(op_strategy1, tuple_strategy, op_strategy4),
            kwargs_schema={},
        )

        # Test args_meta
        args_meta = op_schema.args_meta
        self.assertEqual(len(args_meta), 3)
        # First arg should be TensorMeta
        self.assertIsInstance(args_meta[0], TensorMeta)
        self.assertEqual(args_meta[0].shape, torch.Size([10, 20]))
        # Second arg should be tuple of TensorMeta
        self.assertIsInstance(args_meta[1], tuple)
        self.assertEqual(len(args_meta[1]), 2)
        self.assertIsInstance(args_meta[1][0], TensorMeta)
        self.assertIsInstance(args_meta[1][1], TensorMeta)
        self.assertEqual(args_meta[1][0].shape, torch.Size([5, 10]))
        self.assertEqual(args_meta[1][1].shape, torch.Size([5, 10]))
        # Third arg should be TensorMeta
        self.assertIsInstance(args_meta[2], TensorMeta)
        self.assertEqual(args_meta[2].shape, torch.Size([20, 30]))

    def test_kwargs_meta_mixed(self):
        # [FUNCTION DOC] Builds an OpSchema with keyword args mixing strategies and plain values;
        # checks kwargs_meta has four entries, mapping strategies to TensorMeta (or tuple of them)
        # and passing the plain values through unchanged.
        """Test kwargs_meta with mixed types"""
        # Create a simple mesh
        mesh = DeviceMesh("cpu", torch.arange(4))

        # Create tensor metadata
        tensor_meta1 = TensorMeta(
            shape=torch.Size([10, 20]),
            stride=(20, 1),
            dtype=torch.float32,
        )
        tensor_meta2 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )
        tensor_meta3 = TensorMeta(
            shape=torch.Size([5, 10]),
            stride=(10, 1),
            dtype=torch.float32,
        )

        # Create OpStrategies
        op_strategy1 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(0),), tensor_meta1))]
        )
        op_strategy2 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Shard(1),), tensor_meta2))]
        )
        op_strategy3 = OpStrategy(
            [OpSpec(DTensorSpec(mesh, (Replicate(),), tensor_meta3))]
        )

        # Create TupleStrategy
        tuple_strategy = TupleStrategy([op_strategy2, op_strategy3])

        # Create OpSchema with mixed kwargs
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema=(),
            kwargs_schema={
                "input": op_strategy1,
                "tensors": tuple_strategy,
                "dim": 0,
                "alpha": 1.0,
            },
        )

        # Test kwargs_meta
        kwargs_meta = op_schema.kwargs_meta
        self.assertEqual(len(kwargs_meta), 4)
        self.assertIsInstance(kwargs_meta["input"], TensorMeta)
        self.assertIsInstance(kwargs_meta["tensors"], tuple)
        self.assertEqual(len(kwargs_meta["tensors"]), 2)
        self.assertEqual(kwargs_meta["dim"], 0)
        self.assertEqual(kwargs_meta["alpha"], 1.0)

    def test_max_min_dim_single_dim_strategy(self):
        # [FUNCTION DOC] Checks max_min_dim_single_dim_strategy for max.dim and min.dim: only the
        # non-reduced dim is shardable, yielding one strategy; the output dim index is 0 in both
        # cases and the input dim is 0 or 1 depending on the reduced dim.
        """max.dim/min.dim only allow sharding on non-reduction dims."""
        from torch.distributed.tensor._ops._math_ops import (
            max_min_dim_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]), stride=(4, 1), dtype=torch.float32
        )
        # reduce along dim=1: only dim 0 should be shardable
        strategies = max_min_dim_single_dim_strategy(
            torch.ops.aten.max.dim, (input_meta, 1), {}
        )
        # Should have exactly 1 strategy: shard on dim 0
        self.assertEqual(len(strategies), 1)
        values_plc, indices_plc, input_plc = strategies[0]
        self.assertEqual(values_plc.dim, 0)
        self.assertEqual(indices_plc.dim, 0)
        self.assertEqual(input_plc.dim, 0)

        # reduce along dim=0: only dim 1 should be shardable,
        # output dim maps to 0 (since reduction collapses dim 0)
        strategies = max_min_dim_single_dim_strategy(
            torch.ops.aten.min.dim, (input_meta, 0), {}
        )
        self.assertEqual(len(strategies), 1)
        values_plc, indices_plc, input_plc = strategies[0]
        self.assertEqual(values_plc.dim, 0)
        self.assertEqual(indices_plc.dim, 0)
        self.assertEqual(input_plc.dim, 1)

    def test_foreach_max_single_dim_strategy(self):
        # [FUNCTION DOC] Checks linear_reduction_single_dim_strategy for foreach max: three
        # strategies, Partial max with input sharded on dim 0 and dim 1, and a pure Partial max
        # pair.
        from torch.distributed.tensor._ops._math_ops import (
            linear_reduction_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]), stride=(4, 1), dtype=torch.float32
        )

        strategies = linear_reduction_single_dim_strategy(
            torch.ops.aten._foreach_max.default, (input_meta,), {}
        )

        self.assertEqual(len(strategies), 3)
        self.assertEqual(strategies[0][0], Partial("max"))
        self.assertEqual(strategies[0][1].dim, 0)
        self.assertEqual(strategies[1][0], Partial("max"))
        self.assertEqual(strategies[1][1].dim, 1)
        self.assertEqual(strategies[2], [Partial("max"), Partial("max")])

    def test_var_mean_single_dim_strategy(self):
        # [FUNCTION DOC] Checks std_var_single_dim_strategy for var_mean reducing dim 1: one
        # strategy with outputs and input sharded on dim 0, three placements for the functional
        # overload and five when the out variant supplies two output metas.
        from torch.distributed.tensor._ops._math_ops import std_var_single_dim_strategy

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]), stride=(4, 1), dtype=torch.float32
        )
        output_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )

        strategies = std_var_single_dim_strategy(
            torch.ops.aten.var_mean.correction,
            (input_meta, [1]),
            {},
        )
        self.assertEqual(len(strategies), 1)
        self.assertEqual(len(strategies[0]), 3)
        var_out, mean_out, inp = strategies[0]
        self.assertEqual(var_out.dim, 0)
        self.assertEqual(mean_out.dim, 0)
        self.assertEqual(inp.dim, 0)

        strategies = std_var_single_dim_strategy(
            torch.ops.aten.var_mean.correction_out,
            (input_meta, [1]),
            {"out0": output_meta, "out1": output_meta},
        )
        self.assertEqual(len(strategies), 1)
        self.assertEqual(len(strategies[0]), 5)
        var_out, mean_out, inp, out0, out1 = strategies[0]
        self.assertEqual(var_out.dim, 0)
        self.assertEqual(mean_out.dim, 0)
        self.assertEqual(inp.dim, 0)
        self.assertEqual(out0.dim, 0)
        self.assertEqual(out1.dim, 0)

    def test_nll_loss_forward_single_dim_strategy(self):
        # [FUNCTION DOC] Checks nll_loss_forward_single_dim_strategy for 2d forward: with no
        # reduction and a weight, two strategies of five placements (replicated total weight and
        # weight); with sum reduction and no weight, outputs are Partial sum; with mean reduction
        # no strategy is returned.
        from torch.distributed.tensor._ops._math_ops import (
            nll_loss_forward_single_dim_strategy,
            Reduction,
        )
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
        )

        input_meta = TensorMeta(
            shape=torch.Size([8, 16, 12]), stride=(192, 12, 1), dtype=torch.float32
        )
        target_meta = TensorMeta(
            shape=torch.Size([8, 12]), stride=(12, 1), dtype=torch.int64
        )
        weight_meta = TensorMeta(
            shape=torch.Size([16]), stride=(1,), dtype=torch.float32
        )

        strategies = nll_loss_forward_single_dim_strategy(
            torch.ops.aten.nll_loss2d_forward.default,
            (input_meta, target_meta, weight_meta, Reduction.NONE.value, -100),
            {},
        )
        self.assertEqual(len(strategies), 2)
        self.assertEqual(len(strategies[0]), 5)
        out, total_weight, inp, target, weight = strategies[0]
        self.assertIsInstance(out, _ShardingPlaceholder)
        self.assertEqual(out.dim, 0)
        self.assertIsInstance(total_weight, Replicate)
        self.assertEqual(inp.dim, 0)
        self.assertEqual(target.dim, 0)
        self.assertIsInstance(weight, Replicate)

        out, total_weight, inp, target, weight = strategies[1]
        self.assertEqual(out.dim, 1)
        self.assertIsInstance(total_weight, Replicate)
        self.assertEqual(inp.dim, 2)
        self.assertEqual(target.dim, 1)
        self.assertIsInstance(weight, Replicate)

        strategies = nll_loss_forward_single_dim_strategy(
            torch.ops.aten.nll_loss2d_forward.default,
            (input_meta, target_meta, None, Reduction.SUM.value, -100),
            {},
        )
        self.assertEqual(len(strategies), 2)
        self.assertEqual(len(strategies[0]), 4)
        out, total_weight, inp, target = strategies[0]
        self.assertEqual(out, Partial("sum"))
        self.assertEqual(total_weight, Partial("sum"))
        self.assertEqual(inp.dim, 0)
        self.assertEqual(target.dim, 0)

        strategies = nll_loss_forward_single_dim_strategy(
            torch.ops.aten.nll_loss2d_forward.default,
            (input_meta, target_meta, None, Reduction.MEAN.value, -100),
            {},
        )
        self.assertEqual(strategies, [])

    def test_nll_loss_backward_single_dim_strategy(self):
        # [FUNCTION DOC] Checks nll_loss_backward_single_dim_strategy: with no reduction,
        # three strategies with replicated weight and total weight; with sum reduction and no
        # weight, five strategies where total weight is replicated or Partial sum.
        from torch.distributed.tensor._ops._math_ops import (
            nll_loss_backward_single_dim_strategy,
            Reduction,
        )
        from torch.distributed.tensor._ops.single_dim_strategy import (
            _ShardingPlaceholder,
        )

        grad_out_none_meta = TensorMeta(
            shape=torch.Size([8, 12]), stride=(12, 1), dtype=torch.float32
        )
        grad_out_scalar_meta = TensorMeta(
            shape=torch.Size([]), stride=(), dtype=torch.float32
        )
        input_meta = TensorMeta(
            shape=torch.Size([8, 16, 12]), stride=(192, 12, 1), dtype=torch.float32
        )
        target_meta = TensorMeta(
            shape=torch.Size([8, 12]), stride=(12, 1), dtype=torch.int64
        )
        weight_meta = TensorMeta(
            shape=torch.Size([16]), stride=(1,), dtype=torch.float32
        )
        total_weight_meta = TensorMeta(
            shape=torch.Size([]), stride=(), dtype=torch.float32
        )

        strategies = nll_loss_backward_single_dim_strategy(
            torch.ops.aten.nll_loss2d_backward.default,
            (
                grad_out_none_meta,
                input_meta,
                target_meta,
                weight_meta,
                Reduction.NONE.value,
                -100,
                total_weight_meta,
            ),
            {},
        )
        self.assertEqual(len(strategies), 3)
        self.assertEqual(len(strategies[0]), 6)
        out, grad_out, inp, target, weight, total_weight = strategies[0]
        self.assertIsInstance(out, _ShardingPlaceholder)
        self.assertEqual(out.dim, 0)
        self.assertEqual(grad_out.dim, 0)
        self.assertEqual(inp.dim, 0)
        self.assertEqual(target.dim, 0)
        self.assertIsInstance(weight, Replicate)
        self.assertIsInstance(total_weight, Replicate)

        out, grad_out, inp, target, weight, total_weight = strategies[1]
        self.assertEqual(out.dim, 2)
        self.assertEqual(grad_out.dim, 1)
        self.assertEqual(inp.dim, 2)
        self.assertEqual(target.dim, 1)
        self.assertIsInstance(weight, Replicate)
        self.assertIsInstance(total_weight, Replicate)

        strategies = nll_loss_backward_single_dim_strategy(
            torch.ops.aten.nll_loss2d_backward.default,
            (
                grad_out_scalar_meta,
                input_meta,
                target_meta,
                None,
                Reduction.SUM.value,
                -100,
                total_weight_meta,
            ),
            {},
        )
        self.assertEqual(len(strategies), 5)
        self.assertEqual(len(strategies[0]), 5)
        out, grad_out, inp, target, total_weight = strategies[0]
        self.assertEqual(out.dim, 0)
        self.assertIsInstance(grad_out, Replicate)
        self.assertEqual(inp.dim, 0)
        self.assertEqual(target.dim, 0)
        self.assertIsInstance(total_weight, Replicate)

        out, grad_out, inp, target, total_weight = strategies[1]
        self.assertEqual(out.dim, 0)
        self.assertIsInstance(grad_out, Replicate)
        self.assertEqual(inp.dim, 0)
        self.assertEqual(target.dim, 0)
        self.assertEqual(total_weight, Partial("sum"))

    def test_layer_norm_fwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks layer_norm_single_dim_strategy: normalizing the last dim leaves
        # one shardable dim (dim 0) with replicated weight and bias; omitting bias drops one
        # placement; normalizing over all dims gives no strategies.
        """layer_norm produces sharding rules for each outer dim."""
        from torch.distributed.tensor._ops._math_ops import (
            layer_norm_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([4, 8]), stride=(8, 1), dtype=torch.float32
        )
        weight_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )
        bias_meta = TensorMeta(shape=torch.Size([8]), stride=(1,), dtype=torch.float32)

        # normalized_shape=[8] => axis=1, only dim 0 is shardable
        strategies = layer_norm_single_dim_strategy(
            torch.ops.aten.native_layer_norm.default,
            (input_meta, [8], weight_meta, bias_meta, 1e-5),
            {},
        )
        self.assertEqual(len(strategies), 1)
        # [out, mean, rstd, input, weight, bias]
        out, mean, rstd, inp, w, b = strategies[0]
        self.assertEqual(out.dim, 0)
        self.assertEqual(inp.dim, 0)
        self.assertIsInstance(w, Replicate)
        self.assertIsInstance(b, Replicate)

        # Without bias: 5 elements per rule
        strategies = layer_norm_single_dim_strategy(
            torch.ops.aten.native_layer_norm.default,
            (input_meta, [8], weight_meta, None, 1e-5),
            {},
        )
        self.assertEqual(len(strategies), 1)
        self.assertEqual(len(strategies[0]), 5)

        # normalized_shape=[4, 8] => axis=0, no shardable dims
        strategies = layer_norm_single_dim_strategy(
            torch.ops.aten.native_layer_norm.default,
            (input_meta, [4, 8], weight_meta, bias_meta, 1e-5),
            {},
        )
        self.assertEqual(len(strategies), 0)

    def test_rms_norm_fwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks rms_norm_single_dim_strategy: one strategy sharding output, rrms
        # and input on dim 0 with replicated weight; without a weight each rule has three entries.
        """rms_norm produces sharding rules for each outer dim."""
        from torch.distributed.tensor._ops._math_ops import rms_norm_single_dim_strategy

        input_meta = TensorMeta(
            shape=torch.Size([4, 8]), stride=(8, 1), dtype=torch.float32
        )
        weight_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )

        strategies = rms_norm_single_dim_strategy(
            torch.ops.aten._fused_rms_norm.default,
            (input_meta, [8], weight_meta, 1e-5),
            {},
        )
        self.assertEqual(len(strategies), 1)
        # [out, rrms, input, weight]
        out, rrms, inp, w = strategies[0]
        self.assertEqual(out.dim, 0)
        self.assertEqual(rrms.dim, 0)
        self.assertEqual(inp.dim, 0)
        self.assertIsInstance(w, Replicate)

        # Without weight: 3 elements per rule
        strategies = rms_norm_single_dim_strategy(
            torch.ops.aten._fused_rms_norm.default,
            (input_meta, [8], None, 1e-5),
            {},
        )
        self.assertEqual(len(strategies), 1)
        self.assertEqual(len(strategies[0]), 3)

    def test_layer_norm_bwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks layer_norm_bwd_single_dim_strategy: with all mask outputs on,
        # input grad is sharded and weight/bias grads are Partial (nine entries); absent weight and
        # bias gives None grads (seven entries); output mask can independently turn off any grad
        # while weight and bias inputs stay Replicate.
        """layer_norm backward produces correct output/input placements."""
        from torch.distributed.tensor._ops._math_ops import (
            layer_norm_bwd_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([4, 8]), stride=(8, 1), dtype=torch.float32
        )
        stat_meta = TensorMeta(shape=torch.Size([4]), stride=(1,), dtype=torch.float32)
        weight_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )

        # With weight and bias
        strategies = layer_norm_bwd_single_dim_strategy(
            torch.ops.aten.native_layer_norm_backward.default,
            (
                input_meta,
                input_meta,
                [8],
                stat_meta,
                stat_meta,
                weight_meta,
                weight_meta,
                [True, True, True],
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        # outputs: [d_input, d_weight, d_bias] + inputs: [grad_out, input, mean, rstd, weight, bias]
        self.assertEqual(len(rule), 9)
        d_input, d_weight, d_bias = rule[0], rule[1], rule[2]
        self.assertEqual(d_input.dim, 0)  # sharded
        self.assertIsInstance(d_weight, Partial)  # reduced
        self.assertIsInstance(d_bias, Partial)  # reduced

        # Without weight/bias: d_weight=None, d_bias=None
        strategies = layer_norm_bwd_single_dim_strategy(
            torch.ops.aten.native_layer_norm_backward.default,
            (
                input_meta,
                input_meta,
                [8],
                stat_meta,
                stat_meta,
                None,
                None,
                [True, False, False],
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        # outputs: [d_input, None, None] + inputs: [grad_out, input, mean, rstd]
        self.assertEqual(len(rule), 7)
        self.assertIsNone(rule[1])  # d_weight
        self.assertIsNone(rule[2])  # d_bias

        # With weight/bias inputs but masked-off weight/bias outputs
        strategies = layer_norm_bwd_single_dim_strategy(
            torch.ops.aten.native_layer_norm_backward.default,
            (
                input_meta,
                input_meta,
                [8],
                stat_meta,
                stat_meta,
                weight_meta,
                weight_meta,
                [True, False, False],
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        self.assertEqual(len(rule), 9)
        self.assertEqual(rule[0].dim, 0)  # d_input sharded
        self.assertIsNone(rule[1])  # d_weight masked off
        self.assertIsNone(rule[2])  # d_bias masked off
        self.assertIsInstance(rule[7], Replicate)  # weight input still present
        self.assertIsInstance(rule[8], Replicate)  # bias input still present

        # d_input can also be masked independently of d_weight/d_bias
        strategies = layer_norm_bwd_single_dim_strategy(
            torch.ops.aten.native_layer_norm_backward.default,
            (
                input_meta,
                input_meta,
                [8],
                stat_meta,
                stat_meta,
                weight_meta,
                weight_meta,
                [False, True, False],
            ),
            {},
        )
        rule = strategies[0]
        self.assertIsNone(rule[0])  # d_input masked off
        self.assertIsInstance(rule[1], Partial)  # d_weight reduced
        self.assertIsNone(rule[2])  # d_bias masked off

    def test_rms_norm_bwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks rms_norm_bwd_single_dim_strategy: input grad sharded and weight
        # grad Partial with weight present; None weight grad without weight or when masked off;
        # input grad can be masked independently.
        """rms_norm backward produces correct output/input placements."""
        from torch.distributed.tensor._ops._math_ops import (
            rms_norm_bwd_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([4, 8]), stride=(8, 1), dtype=torch.float32
        )
        stat_meta = TensorMeta(shape=torch.Size([4]), stride=(1,), dtype=torch.float32)
        weight_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )

        # With weight
        strategies = rms_norm_bwd_single_dim_strategy(
            torch.ops.aten._fused_rms_norm_backward.default,
            (
                input_meta,  # grad_out
                input_meta,  # input
                [8],  # normalized_shape
                stat_meta,  # rstd
                weight_meta,  # weight
                [True, True],  # output_mask
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        # outputs: [d_input, d_weight] + inputs: [grad_out, input, rstd, weight]
        self.assertEqual(len(rule), 6)
        self.assertEqual(rule[0].dim, 0)  # d_input sharded
        self.assertIsInstance(rule[1], Partial)  # d_weight reduced
        self.assertIsInstance(rule[5], Replicate)  # weight replicated

        # Without weight: d_weight=None
        strategies = rms_norm_bwd_single_dim_strategy(
            torch.ops.aten._fused_rms_norm_backward.default,
            (input_meta, input_meta, [8], stat_meta, None, [True, False]),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        # outputs: [d_input, None] + inputs: [grad_out, input, rstd]
        self.assertEqual(len(rule), 5)
        self.assertIsNone(rule[1])  # d_weight

        # With weight input but masked-off d_weight output
        strategies = rms_norm_bwd_single_dim_strategy(
            torch.ops.aten._fused_rms_norm_backward.default,
            (
                input_meta,  # grad_out
                input_meta,  # input
                [8],  # normalized_shape
                stat_meta,  # rstd
                weight_meta,  # weight
                [True, False],  # output_mask
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        self.assertEqual(len(rule), 6)
        self.assertEqual(rule[0].dim, 0)  # d_input sharded
        self.assertIsNone(rule[1])  # d_weight masked off
        self.assertIsInstance(rule[5], Replicate)  # weight input still present

        # d_input can be masked independently of d_weight
        strategies = rms_norm_bwd_single_dim_strategy(
            torch.ops.aten._fused_rms_norm_backward.default,
            (
                input_meta,
                input_meta,
                [8],
                stat_meta,
                weight_meta,
                [False, True],
            ),
            {},
        )
        rule = strategies[0]
        self.assertIsNone(rule[0])  # d_input masked off
        self.assertIsInstance(rule[1], Partial)  # d_weight reduced

    def test_grid_sampler_bwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks grid_sampler_backward_strategy honors the output mask for
        # grad_input (None when masked off) while the grid gradient is always sharded on dim 0
        # regardless of the second mask entry.
        """grid_sampler backward respects the native output_mask contract."""
        from torch.distributed.tensor._ops._math_ops import (
            grid_sampler_backward_strategy,
        )

        grad_output_meta = TensorMeta(
            shape=torch.Size([2, 3, 4, 4]), stride=(48, 16, 4, 1), dtype=torch.float32
        )
        input_meta = TensorMeta(
            shape=torch.Size([2, 3, 8, 8]), stride=(192, 64, 8, 1), dtype=torch.float32
        )
        grid_meta = TensorMeta(
            shape=torch.Size([2, 4, 4, 2]), stride=(32, 8, 2, 1), dtype=torch.float32
        )

        strategies = grid_sampler_backward_strategy(
            torch.ops.aten.grid_sampler_2d_backward.default,
            (grad_output_meta, input_meta, grid_meta, 0, 0, False, [True, False]),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        self.assertEqual(len(rule), 5)
        self.assertEqual(rule[0].dim, 0)  # grad_input sharded
        self.assertEqual(rule[1].dim, 0)  # grad_grid is always computed
        self.assertEqual(rule[2].dim, 0)  # grad_output sharded
        self.assertEqual(rule[3].dim, 0)  # input sharded
        self.assertEqual(rule[4].dim, 0)  # grid sharded

        strategies = grid_sampler_backward_strategy(
            torch.ops.aten.grid_sampler_2d_backward.default,
            (grad_output_meta, input_meta, grid_meta, 0, 0, False, [False, True]),
            {},
        )
        rule = strategies[0]
        self.assertIsNone(rule[0])  # grad_input masked off
        self.assertEqual(rule[1].dim, 0)  # grad_grid sharded

        strategies = grid_sampler_backward_strategy(
            torch.ops.aten.grid_sampler_2d_backward.default,
            (grad_output_meta, input_meta, grid_meta, 0, 0, False, [False, False]),
            {},
        )
        rule = strategies[0]
        self.assertIsNone(rule[0])  # grad_input masked off
        self.assertEqual(rule[1].dim, 0)  # grad_grid ignores output_mask[1]

    def test_batch_norm_bwd_single_dim_strategy(self):
        # [FUNCTION DOC] Checks batch_norm_backward_strategy output mask handling: masked-off
        # grads are None, grad_input shards on the channel dim and weight/bias grads on dim 0.
        """batch_norm backward respects output_mask."""
        from torch.distributed.tensor._ops._math_ops import batch_norm_backward_strategy

        input_meta = TensorMeta(
            shape=torch.Size([4, 8, 16, 16]),
            stride=(2048, 256, 16, 1),
            dtype=torch.float32,
        )
        channel_meta = TensorMeta(
            shape=torch.Size([8]), stride=(1,), dtype=torch.float32
        )

        strategies = batch_norm_backward_strategy(
            torch.ops.aten.native_batch_norm_backward.default,
            (
                input_meta,  # grad_out
                input_meta,  # input
                channel_meta,  # weight
                channel_meta,  # running_mean
                channel_meta,  # running_var
                channel_meta,  # save_mean
                channel_meta,  # save_invstd
                True,  # train
                1e-5,  # eps
                [True, False, True],  # output_mask
            ),
            {},
        )
        self.assertEqual(len(strategies), 1)
        rule = strategies[0]
        self.assertEqual(len(rule), 10)
        self.assertEqual(rule[0].dim, 1)  # grad_input sharded on channel
        self.assertIsNone(rule[1])  # grad_weight masked off
        self.assertEqual(rule[2].dim, 0)  # grad_bias sharded
        self.assertEqual(rule[3].dim, 1)  # grad_out sharded on channel
        self.assertEqual(rule[4].dim, 1)  # input sharded on channel

        strategies = batch_norm_backward_strategy(
            torch.ops.aten.native_batch_norm_backward.default,
            (
                input_meta,
                input_meta,
                channel_meta,
                channel_meta,
                channel_meta,
                channel_meta,
                channel_meta,
                True,
                1e-5,
                [False, True, False],
            ),
            {},
        )
        rule = strategies[0]
        self.assertIsNone(rule[0])  # grad_input masked off
        self.assertEqual(rule[1].dim, 0)  # grad_weight sharded
        self.assertIsNone(rule[2])  # grad_bias masked off

    def test_constant_pad_nd_allows_shard_on_non_padded_dim(self):
        # [FUNCTION DOC] Asserts constant_pad_nd's single-dim strategy offers sharding on the
        # non-padded dim 0 of an 8x4 input when only dim 1 is padded.
        """constant_pad_nd should allow sharding on non-padded dims."""
        from torch.distributed.tensor._ops._matrix_ops import (
            constant_pad_nd_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]), stride=(4, 1), dtype=torch.float32
        )
        # Pad only dim 1: [1, 1] means pad last dim by 1 on each side
        strategies = constant_pad_nd_single_dim_strategy(
            torch.ops.aten.constant_pad_nd.default,
            (input_meta, [1, 1], 0.0),
            {},
        )
        # Should include a strategy that shards on dim 0
        shard_dims = {s[0].dim for s in strategies if hasattr(s[0], "dim")}
        self.assertIn(0, shard_dims)

    def test_constant_pad_nd_bans_shard_on_padded_dim(self):
        # [FUNCTION DOC] Asserts constant_pad_nd's single-dim strategy never shards the padded
        # dim 1 of an 8x4 input.
        """constant_pad_nd should ban sharding on padded dims."""
        from torch.distributed.tensor._ops._matrix_ops import (
            constant_pad_nd_single_dim_strategy,
        )

        input_meta = TensorMeta(
            shape=torch.Size([8, 4]), stride=(4, 1), dtype=torch.float32
        )
        # Pad only dim 1
        strategies = constant_pad_nd_single_dim_strategy(
            torch.ops.aten.constant_pad_nd.default,
            (input_meta, [1, 1], 0.0),
            {},
        )
        shard_dims = {s[0].dim for s in strategies if hasattr(s[0], "dim")}
        self.assertNotIn(1, shard_dims, "Should not shard on padded dim")


class TestOpSpecMesh(TestCase):
    # [CLASS DOC] Tests how OpSpec.mesh finds the device mesh when output or input specs
    # contain None entries, and string formatting of an all-None OpStrategy. Uses a fake
    # 4-rank process group per test.
    """Tests for OpSpec.mesh property handling of None specs."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Initializes a fake process group with 4 ranks and a FakeStore.
        super().setUp()
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=4, store=store
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown and destroys the default process group.
        super().tearDown()
        torch.distributed.destroy_process_group()

    def _make_spec(self, placements):
        # [FUNCTION DOC] HELPER: Builds a DTensorSpec on a 4-rank CPU mesh for an 8x16 float32
        # tensor with the given placements.
        mesh = DeviceMesh("cpu", torch.arange(4))
        tm = TensorMeta(
            shape=torch.Size([8, 16]),
            stride=(16, 1),
            dtype=torch.float32,
        )
        return DTensorSpec(mesh, placements, tm)

    def test_mesh_from_single_output_spec(self):
        # [FUNCTION DOC] Mesh is derived from a single output spec (shape of 4).
        spec = self._make_spec((Replicate(),))
        op_spec = OpSpec(output_specs=spec, input_specs=(spec,))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_from_tuple_output_specs(self):
        # [FUNCTION DOC] Mesh is derived from a tuple of output specs.
        spec = self._make_spec((Shard(0),))
        op_spec = OpSpec(output_specs=(spec, spec), input_specs=(spec,))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_from_tuple_output_specs_with_leading_none(self):
        # [FUNCTION DOC] Mesh is taken from the first non-None entry when the first output is None.
        """When the first output spec is None, mesh should come from the
        next non-None entry."""
        spec = self._make_spec((Replicate(),))
        op_spec = OpSpec(output_specs=(None, spec), input_specs=(spec,))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_from_input_specs_when_output_is_none(self):
        # [FUNCTION DOC] With None output specs, mesh falls back to the input specs.
        """When output_specs is None, mesh should come from input_specs."""
        spec = self._make_spec((Shard(0),))
        op_spec = OpSpec(output_specs=None, input_specs=(spec,))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_from_input_specs_with_leading_none(self):
        # [FUNCTION DOC] With None outputs, mesh comes from the first non-None input spec.
        """When output_specs is None and input_specs has leading None entries,
        mesh should come from the first non-None input spec."""
        spec = self._make_spec((Replicate(),))
        op_spec = OpSpec(output_specs=None, input_specs=(None, spec))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_from_input_specs_when_tuple_output_all_none(self):
        # [FUNCTION DOC] A tuple of all-None outputs falls back to the input specs like None.
        """When output_specs is a tuple of all None, mesh should fall through
        to input_specs — same behavior as output_specs=None."""
        spec = self._make_spec((Shard(0),))
        op_spec = OpSpec(output_specs=(None, None), input_specs=(spec,))
        self.assertEqual(op_spec.mesh.shape, (4,))

    def test_mesh_raises_when_all_specs_none(self):
        # [FUNCTION DOC] When all output and input specs are None, accessing mesh raises
        # AssertionError mentioning that the mesh cannot be determined.
        """When both output_specs and all input_specs are None, mesh should
        raise AssertionError."""
        op_spec = OpSpec(output_specs=None, input_specs=(None, None))
        with self.assertRaisesRegex(AssertionError, "Cannot determine mesh"):
            _ = op_spec.mesh

    def test_op_strategy_str_handles_all_specs_none(self):
        # [FUNCTION DOC] Regression test: str of an OpStrategy whose only OpSpec has all-None
        # specs gives the string OpStrategy[(None) -> None] without raising.
        """Regression test for https://github.com/pytorch/pytorch/issues/182370."""
        op_strategy = OpStrategy([OpSpec(output_specs=None, input_specs=(None,))])

        self.assertEqual(str(op_strategy), "OpStrategy[(None) -> None]")


class TestExpandToFullMeshOpStrategy(TestCase):
    # [CLASS DOC] Tests expand_to_full_mesh_op_strategy assigns the right tensor_meta to
    # input specs, particularly when the placement lists contain None output entries. Uses a fake
    # 4-rank process group per test.
    """Tests for expand_to_full_mesh_op_strategy function.

    These tests verify that tensor_meta is correctly assigned to input/output specs,
    especially when the placement list contains None entries (e.g., optional outputs
    like grad_bias in SDPA backward when attn_bias is not used).
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets world_size to 4 and initializes a fake process group with a
        # FakeStore (kept on the instance).
        from torch.testing._internal.distributed.fake_pg import FakeStore

        super().setUp()
        self.world_size = 4
        self.fake_store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=self.fake_store
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown and destroys the default process group.
        super().tearDown()
        torch.distributed.destroy_process_group()

    def _create_op_strategy_with_tensor_meta(
        self, mesh: DeviceMesh, shape: tuple, dtype: torch.dtype
    ) -> OpStrategy:
        # [FUNCTION DOC] HELPER: Builds an OpStrategy with one OpSpec whose input and output spec
        # is fully replicated over the mesh, with contiguous strides computed from the shape and
        # the given dtype, and a redistribute cost of 0.0.
        """Create an OpStrategy with tensor_meta for testing."""
        placements = tuple(Replicate() for _ in range(mesh.ndim))
        strides = []
        stride = 1
        for s in reversed(shape):
            strides.append(stride)
            stride *= s
        strides = tuple(reversed(strides))

        tensor_meta = TensorMeta(shape=torch.Size(shape), stride=strides, dtype=dtype)
        spec = DTensorSpec(mesh=mesh, placements=placements, tensor_meta=tensor_meta)
        op_spec = OpSpec(
            output_specs=spec, input_specs=(spec,), redistribute_cost=[[0.0]]
        )
        return OpStrategy([op_spec])

    def test_expand_strategy_with_none_in_outputs(self):
        # [FUNCTION DOC] With a None output entry in the placement list and distinct input shapes
        # (different value head dim), checks the expanded strategy's input specs get their own
        # tensor_meta rather than shifted ones.
        """Test that None entries in outputs don't shift input tensor_meta assignment.

        This test simulates SDPA backward where grad_bias output is None when
        attn_bias is not used. The bug was that the None entry caused subsequent
        input specs to get wrong tensor_meta (shifted by one position).
        """
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # Simulate MLA-style attention with different dimensions for q/k vs v
        batch, heads, seq_len = 2, 8, 64
        d_qk = 192  # query/key head dimension
        d_v = 128  # value head dimension (different!)

        # Create input OpStrategy objects with distinct shapes
        grad_out_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_v), torch.float32
        )
        query_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_qk), torch.float32
        )
        key_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_qk), torch.float32
        )
        value_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (batch, heads, seq_len, d_v), torch.float32
        )

        # Build OpSchema (attn_bias is None at position 4)
        args_schema = (
            grad_out_strategy,
            query_strategy,
            key_strategy,
            value_strategy,
        )
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,  # dummy op for testing
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list with None at position 3 (simulating grad_bias = None)
        # Structure: [out0, out1, out2, None, in0, in1, in2, in3]
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output 0
                Replicate(),  # output 1
                Replicate(),  # output 2
                None,  # output 3 (None, like grad_bias when attn_bias is None)
                Replicate(),  # input 0: grad_out
                Replicate(),  # input 1: query
                Replicate(),  # input 2: key
                Replicate(),  # input 3: value
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,  # 4 outputs (including the None)
        )

        # Verify the input specs have correct tensor_meta
        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 4)

        # Check that each input got the correct tensor_meta shape
        # Before the fix, these would be shifted by one position
        self.assertEqual(input_specs[0].tensor_meta.shape[-1], d_v)  # grad_out
        self.assertEqual(input_specs[1].tensor_meta.shape[-1], d_qk)  # query
        self.assertEqual(input_specs[2].tensor_meta.shape[-1], d_qk)  # key
        self.assertEqual(input_specs[3].tensor_meta.shape[-1], d_v)  # value

    def test_expand_strategy_without_none_in_outputs(self):
        # [FUNCTION DOC] With no None entries, the expanded strategy has two input specs with
        # the original 10x20 and 10x30 shapes.
        """Test that expand_to_full_mesh_op_strategy works correctly without None entries."""
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        # Create input OpStrategy objects
        input1_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (10, 20), torch.float32
        )
        input2_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (10, 30), torch.float32
        )

        args_schema = (input1_strategy, input2_strategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list without any None entries
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output
                Replicate(),  # input 0
                Replicate(),  # input 1
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=1,
        )

        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 2)
        self.assertEqual(input_specs[0].tensor_meta.shape, torch.Size([10, 20]))
        self.assertEqual(input_specs[1].tensor_meta.shape, torch.Size([10, 30]))

    def test_expand_strategy_with_multiple_nones_in_outputs(self):
        # [FUNCTION DOC] With several None output entries, input specs still get the matching
        # tensor_meta shapes (5x10 and 5x20).
        """Test handling of multiple None entries in outputs."""
        from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

        mesh = DeviceMesh("cpu", torch.arange(self.world_size))

        input1_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (5, 10), torch.float32
        )
        input2_strategy = self._create_op_strategy_with_tensor_meta(
            mesh, (5, 20), torch.float32
        )

        args_schema = (input1_strategy, input2_strategy)
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            args_schema,
            {},
            RuntimeSchemaInfo(needs_pytree=False),
        )

        # Placement list with multiple None entries in outputs
        single_mesh_dim_strategies = [
            [
                Replicate(),  # output 0 (non-None)
                None,  # output 1 (None)
                Replicate(),  # output 2 (non-None)
                None,  # output 3 (None)
                Replicate(),  # input 0
                Replicate(),  # input 1
            ]
        ]

        result = expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,  # 4 outputs (2 None, 2 non-None)
        )

        strategy = result.strategies[0]
        input_specs = strategy.input_specs

        self.assertEqual(len(input_specs), 2)
        # Verify correct tensor_meta assignment despite multiple Nones
        self.assertEqual(input_specs[0].tensor_meta.shape, torch.Size([5, 10]))
        self.assertEqual(input_specs[1].tensor_meta.shape, torch.Size([5, 20]))


# Define a dummy op for testing unsupported DTensor ops
@torch.library.custom_op("testlib::unsupported_add", mutates_args=())
def unsupported_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # HELPER: Dummy custom op body returning x plus y; no DTensor strategy is registered.
    """A dummy op that has no DTensor sharding strategy registered."""
    return x + y


@unsupported_add.register_fake
def _unsupported_add_fake(x, y):
    # HELPER: Fake implementation: returns an empty tensor shaped like x.
    return torch.empty_like(x)


# Define a dummy op similar to torch.cat that takes a list of tensors
@torch.library.custom_op("testlib::unsupported_cat", mutates_args=())
def unsupported_cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    # HELPER: Dummy cat-like custom op body returning the concatenation of the tensors on dim.
    """A dummy cat-like op that has no DTensor sharding strategy registered."""
    return torch.cat(tensors, dim=dim)


@unsupported_cat.register_fake
def _unsupported_cat_fake(tensors, dim=0):
    # HELPER: Fake implementation: raises ValueError for an empty list, otherwise returns an
    # empty tensor whose size along dim is the sum over the inputs, other dims from the first.
    # Return a tensor with concatenated shape
    if len(tensors) == 0:
        raise ValueError("Expected at least one tensor")
    first = tensors[0]
    shape = list(first.shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    return torch.empty(shape, dtype=first.dtype, device=first.device)


class TestUnsupportedDTensorOp(TestCase):
    # [CLASS DOC] Tests the error raised for ops with no registered DTensor strategy; uses a
    # fake 4-rank process group per test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Sets world_size to 4 and initializes the fake process group
        # backend (rank 0) with a FakeStore.
        super().setUp()
        self.world_size = 4
        store = FakeStore()
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs base teardown and destroys the default process group.
        super().tearDown()
        torch.distributed.destroy_process_group()

    def test_unsupported_op_error(self):
        # [FUNCTION DOC] Expects NotImplementedError naming the operator and saying no sharding
        # strategy is registered, for both the two-tensor add op and the list-input cat op.
        """Test that running an unsupported op on DTensor raises NotImplementedError with appropriate message."""
        mesh = DeviceMesh("cpu", torch.arange(self.world_size))
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)
        x_dt = distribute_tensor(x, mesh, [Shard(0)])
        y_dt = distribute_tensor(y, mesh, [Shard(0)])

        with self.assertRaisesRegex(
            NotImplementedError,
            r"Operator.*testlib\.unsupported_add.*does not have a sharding strategy registered",
        ):
            _ = torch.ops.testlib.unsupported_add(x_dt, y_dt)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"Operator.*testlib\.unsupported_cat.*does not have a sharding strategy registered",
        ):
            _ = torch.ops.testlib.unsupported_cat([x_dt, y_dt], dim=0)


if __name__ == "__main__":
    run_tests()
