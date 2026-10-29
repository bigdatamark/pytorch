# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor loop ordering after fusion (LOAF), loop
# reindexing, memory-coalescing based tiling analysis, split of iteration ranges
# and index-expression inversion. Contains mock-scheduler unit tests built from
# hand-made IR buffers, end-to-end compile tests that count generated kernels and
# bytes accessed, and sympy-based checks of tiling and inversion utilities.
# Import-time side effects: enables inductor metrics logging, reads the
# DO_PERF_TEST environment flag, and when a GPU is available sets the default
# torch device to the GPU type. Running as a script invokes the test runner only
# when a GPU is available.

import contextlib
import os
import unittest
from unittest import mock, skipUnless

import numpy as np
import sympy

import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config, ir, metrics
from torch._inductor.codegen.simd import MemoryCoalescing, SIMDScheduling
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.graph import GraphLowering
from torch._inductor.invert_expr_analysis import generate_inverse_formula
from torch._inductor.scheduler import (
    _LoopMutationTracker,
    ForeachKernelSchedulerNode,
    FusedSchedulerNode,
    refresh_group_node_dependencies,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.test_operators import realize
from torch._inductor.utils import is_big_gpu, run_and_get_code, sympy_index_symbol
from torch._inductor.virtualized import ops, V
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)
DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)


class MockScheduler:
    # [CLASS DOC] HELPER: Minimal stand-in for the Inductor Scheduler, installed as
    # the scheduler of a graph handler in the tests. It exposes an empty tuple of
    # available buffer names, a backend lookup that yields a Triton scheduling object,
    # and a buffer-removal query that always answers false. Stateless.
    available_buffer_names = ()

    @staticmethod
    def get_backend(cls, *args):
        # [FUNCTION DOC] HELPER: Static stand-in for the backend lookup. The first
        # positional argument (named cls, it is the scheduler object the caller passes)
        # is used to construct and return a new Triton scheduling backend; any extra
        # arguments are ignored.
        return TritonScheduling(cls)

    def can_buffer_be_removed_through_fusion(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always returns False, so tests never treat a buffer as
        # removable through fusion. All arguments are ignored.
        return False


class MockSchedulerTest(TestCase):
    # [CLASS DOC] HELPER: Base test case that makes a mock scheduler available through
    # the global Inductor virtualized graph handler. The class-level exit stack, set
    # in the class setup, holds the active graph-handler context and is closed in the
    # class teardown. Subclasses build scheduler nodes against the graph scheduler.
    _exit_stack = None

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: runs the parent setup, symbolically traces a trivial
        # function into an FX module, wraps it in a graph lowering object whose scheduler
        # attribute is the mock scheduler class, creates the exit stack and enters the
        # context that installs the graph as the current virtualized graph. Side effect:
        # global graph handler stays set until teardown.
        super().setUpClass()

        gm = torch.fx.symbolic_trace(lambda: 0)
        graph = GraphLowering(gm)
        graph.scheduler = MockScheduler
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(V.set_graph_handler(graph))

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: runs the parent teardown, then closes the exit
        # stack, which uninstalls the graph handler set in the class setup.
        super().tearDownClass()
        cls._exit_stack.close()


@inductor_config.patch(loop_ordering_after_fusion=True)
@instantiate_parametrized_tests
class ImplDetailTest(MockSchedulerTest):
    # [CLASS DOC] Unit tests for scheduler-node loop manipulation internals (loop
    # reordering, loop merging, tiling caches, loop-mutation rollback trackers and the
    # reindexing memory guard). Runs with loop ordering after fusion enabled. Uses the
    # mock scheduler from the base class and the GPU device for IR layouts.
    @staticmethod
    def _get_snode_body_sym_prefix(snode):
        # [FUNCTION DOC] HELPER: Returns the first character of the name of the first
        # iteration variable in the scheduler node body, which identifies the symbol
        # prefix used for loop variables. Raises AssertionError if the body has no
        # iteration variables or the name is empty.
        body = snode._body
        prefix = ""

        for var in body.var_ranges:
            prefix = str(var)[0]
            break

        if not prefix:
            raise AssertionError
        return prefix

    @staticmethod
    def _create_computed_buffer_ax2(sizes=(32, 64), strides=None):
        # [FUNCTION DOC] HELPER: Builds a realized computed buffer that multiplies a fixed
        # input buffer named a by two. The input has the given sizes (default 32 by 64)
        # and strides (default contiguous strides), float32 on the GPU device. Creates the
        # pointwise IR, realizes it, decides its layout and returns the resulting computed
        # buffer object.
        """
        Create a ComputedBuffer for 'a x 2'
        """
        if strides is None:
            strides = ir.FlexibleLayout.contiguous_strides(sizes)

        box_a = ir.TensorBox.create(
            ir.Buffer(
                name="a",
                layout=ir.FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=sizes,
                    stride=strides,
                ),
            )
        )
        box_a_loader = box_a.make_loader()

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body used for the computed buffer; loads the
            # input at the given index and returns the loaded value times two.
            return box_a_loader(index) * 2

        buf = ir.Pointwise.create(
            device=box_a.get_device(),
            dtype=box_a.get_dtype(),
            inner_fn=inner_fn,
            ranges=box_a.get_size(),
        )
        buf.realize()
        computed_buf = buf.data.data
        computed_buf.decide_layout()
        return computed_buf

    def test_reorder_twice(self):
        # [FUNCTION DOC] Checks that applying a loop reorder twice in a row on a scheduler
        # node still works and keeps the loop variable prefix as p after each reorder, as
        # happens when a second order is chosen after fusing a further node.
        """
        This may happen in practice if we pick a order when fusing A and B.
        Then we pick another order for AB when we fusion C into it.

        E.g. happens for BertForMaskedLM.
        """

        buf = self._create_computed_buffer_ax2()
        snode = SchedulerNode(V.graph.scheduler, buf)
        snode.apply_new_loop_order([1, 0])
        prefix1 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix1 == "p")
        snode.apply_new_loop_order([1, 0])
        prefix2 = self._get_snode_body_sym_prefix(snode)
        self.assertTrue(prefix2 == "p")

    def test_reorder_and_merge_loops(self):
        # [FUNCTION DOC] Builds a buffer with sizes 1024 by 2048 and strides 1, 1024, so
        # the loops are first reordered by simplification (checked via the reversed
        # sizes), then merges the loops and checks that a single loop with the product of
        # the sizes results.
        sizes = (1024, 2048)
        strides = (1, 1024)
        buf = self._create_computed_buffer_ax2(sizes, strides)
        old_sizes, old_body = buf.simplify_and_reorder()

        # Make sure loop reordering happens here
        self.assertTrue(
            tuple(old_sizes[0]) == tuple(reversed(sizes)),
            lambda msg: f"{msg}\n{old_sizes=}",
        )
        new_body = old_body.merge_loops()
        new_sizes = new_body.sizes
        self.assertTrue(
            tuple(new_sizes[0]) == (np.prod(sizes),), lambda msg: f"{msg}\n{new_sizes=}"
        )

    def test_merge_loops_invalidate_pw_dep_cache(self):
        # [FUNCTION DOC] Verifies that merging loops of a scheduler node invalidates the
        # cached pointwise read/writes: before merging two iteration variables exist,
        # after merging only one, so the cache must be refreshed.
        sizes = (1024, 2048)
        strides = (2048, 1)
        buf = self._create_computed_buffer_ax2(sizes, strides)

        snode = SchedulerNode(V.graph.scheduler, buf)
        old_var_ranges = snode.pointwise_read_writes().var_ranges
        self.assertTrue(len(old_var_ranges) == 2)  # 2 dimension not merged
        snode.merge_loops()
        new_var_ranges = snode.pointwise_read_writes().var_ranges

        # we cache pointwise_read_writes result on a scheduler node
        # make sure new_var_ranges is refreshed by invalidating the cache.
        self.assertTrue(len(new_var_ranges) == 1)  # 2 dimensions get merged

    def test_reorder_invalidates_tiling_cache(self):
        # [FUNCTION DOC] Verifies that applying a new loop order invalidates the node
        # tiling cache. Wraps the SIMD tiling selection with a counting mock: repeated
        # tiling queries hit the cache (one selection), after a reorder the selection is
        # run again and the new tiling differs from the old one.
        buf = self._create_computed_buffer_ax2()
        snode = SchedulerNode(V.graph.scheduler, buf)

        with mock.patch.object(
            SIMDScheduling,
            "select_tiling",
            wraps=SIMDScheduling.select_tiling,
        ) as select_tiling:
            tiling_before = snode.get_tiling(*snode.group[1])
            self.assertIs(tiling_before, snode.get_tiling(*snode.group[1]))
            self.assertEqual(select_tiling.call_count, 1)

            snode.apply_new_loop_order([1, 0])
            tiling_after = snode.get_tiling(*snode.group[1])
            self.assertEqual(select_tiling.call_count, 2)
            self.assertNotEqual(tiling_before, tiling_after)

    def test_read_writes_invalidate_tiling_cache(self):
        # [FUNCTION DOC] Verifies that replacing the read/writes of a scheduler node
        # invalidates its tiling cache: two queries cause one tiling selection, and a
        # query after setting the read/writes causes a second one.
        buf = self._create_computed_buffer_ax2()
        snode = SchedulerNode(V.graph.scheduler, buf)

        with mock.patch.object(
            SIMDScheduling,
            "select_tiling",
            wraps=SIMDScheduling.select_tiling,
        ) as select_tiling:
            snode.get_tiling(*snode.group[1])
            snode.get_tiling(*snode.group[1])
            self.assertEqual(select_tiling.call_count, 1)

            snode.set_read_writes(snode.read_writes)
            snode.get_tiling(*snode.group[1])
            self.assertEqual(select_tiling.call_count, 2)

    def test_tiling_cache_is_per_node(self):
        # [FUNCTION DOC] Verifies that tiling caches are per node: two equal-shaped nodes
        # have the same group, and querying each twice results in exactly two tiling
        # selections (one per node).
        snode1 = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        snode2 = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        self.assertEqual(snode1.group, snode2.group)

        with mock.patch.object(
            SIMDScheduling,
            "select_tiling",
            wraps=SIMDScheduling.select_tiling,
        ) as select_tiling:
            snode1.get_tiling(*snode1.group[1])
            snode2.get_tiling(*snode2.group[1])
            snode1.get_tiling(*snode1.group[1])
            snode2.get_tiling(*snode2.group[1])
            self.assertEqual(select_tiling.call_count, 2)

    def test_tiling_cache_does_not_require_direct_simd_backend(self):
        # [FUNCTION DOC] Verifies that tiling lookup on a node does not go through the
        # scheduler backend lookup: the lookup is mocked to raise AssertionError and the
        # tiling query must still succeed.
        snode = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())

        with mock.patch.object(
            MockScheduler,
            "get_backend",
            side_effect=AssertionError("unexpected backend lookup"),
        ):
            snode.get_tiling(*snode.group[1])

    def test_tiling_cache_keys_iteration_extents(self):
        # [FUNCTION DOC] Verifies that the tiling cache is keyed by the iteration extents:
        # with tiling selection mocked to return an empty mapping, repeating the same
        # extents hits the cache while a different first extent misses, giving two
        # selections for three queries.
        snode = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())

        with mock.patch.object(
            SIMDScheduling, "select_tiling", return_value={}
        ) as select_tiling:
            snode.get_tiling(sympy.Integer(1), sympy.Integer(1))
            snode.get_tiling(sympy.Integer(1), sympy.Integer(1))
            snode.get_tiling(sympy.Integer(2), sympy.Integer(1))
            self.assertEqual(select_tiling.call_count, 2)

    def test_fusion_reuses_node_tiling_cache(self):
        # [FUNCTION DOC] Verifies that checking whether two nodes can fuse reuses node
        # tiling caches: the first fusion check triggers three tiling selections and an
        # identical second check triggers exactly one more.
        snode1 = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        snode2 = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        backend = TritonScheduling(V.graph.scheduler)

        with mock.patch.object(
            SIMDScheduling,
            "select_tiling",
            wraps=SIMDScheduling.select_tiling,
        ) as select_tiling:
            self.assertTrue(backend.can_fuse(snode1, snode2))
            self.assertEqual(select_tiling.call_count, 3)

            self.assertTrue(backend.can_fuse(snode1, snode2))
            self.assertEqual(select_tiling.call_count, 4)

    def test_template_producer_loop_state_rollback(self):
        # [FUNCTION DOC] Verifies rollback of loop state through the loop mutation tracker
        # when a template buffer node is paired with a computed node. Applying indexing
        # expressions to the computed node changes its body; rolling back restores the
        # original body and leaves the template node body as None.
        layout = ir.FixedLayout(
            torch.device(GPU_TYPE), torch.float32, [sympy.Integer(8)], [1]
        )
        template = ir.TemplateBuffer(layout, [], None)
        template_node = SchedulerNode(V.graph.scheduler, template)
        computed_node = SchedulerNode(
            V.graph.scheduler, self._create_computed_buffer_ax2()
        )
        original_body = computed_node._body
        tracker = _LoopMutationTracker.create((template_node, computed_node))

        try:
            computed_node.apply_indexing_exprs({})
            self.assertIsNot(computed_node._body, original_body)
        finally:
            tracker.finish(rollback=True)

        self.assertIsNone(template_node._body)
        self.assertIs(computed_node._body, original_body)

    def test_nested_loop_mutation_trackers(self):
        # [FUNCTION DOC] Verifies nested loop mutation trackers: an inner tracker finishes
        # without rollback so the body stays changed, while the outer tracker, which saw
        # the mutation through the chained listener, rolls it back. Afterwards the body is
        # the original object and the node has no mutation listener left.
        """A nested scope committing must not hide the mutation from the outer one."""
        snode = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        original_body = snode._body
        outer = _LoopMutationTracker.create((snode,))
        inner = _LoopMutationTracker.create((snode,))

        try:
            snode.apply_new_loop_order([1, 0])
            self.assertIsNot(snode._body, original_body)
            # The inner scope keeps the mutation, but the outer scope still saw
            # it via the chained listener and can roll it back.
            inner.finish(rollback=False)
            self.assertIsNot(snode._body, original_body)
        finally:
            outer.finish(rollback=True)

        self.assertIs(snode._body, original_body)
        self.assertIsNone(snode._loop_mutation_listener)

    def test_expand_dimension_loop_state_rollback(self):
        # [FUNCTION DOC] Verifies that expanding a dimension of a pointwise node changes
        # its snapshot of loop state and that finishing the tracker with rollback
        # restores a snapshot equal to the original.
        snode = SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
        original_state = snode.snapshot_loop_state()
        tracker = _LoopMutationTracker.create((snode,))

        snode.expand_dimension_for_pointwise_node(0, 64)
        self.assertNotEqual(snode.snapshot_loop_state(), original_state)
        tracker.finish(rollback=True)

        self.assertEqual(snode.snapshot_loop_state(), original_state)

    def test_foreach_nested_fused_loop_state_rollback(self):
        # [FUNCTION DOC] Verifies rollback for a foreach node containing a fused node of
        # two scheduler nodes. It sets orders and input distances, expands a dimension of
        # one node, changes the fused group and refreshes dependencies, then rolls back
        # and checks that the body object, the fused group and the foreach read/writes
        # are restored.
        snodes = [
            SchedulerNode(V.graph.scheduler, self._create_computed_buffer_ax2())
            for _ in range(2)
        ]
        for order, snode in enumerate(snodes):
            snode.min_order = snode.max_order = order
            snode.min_input_distance = snode.max_input_distance = 0
        subkernel = FusedSchedulerNode(V.graph.scheduler, snodes)
        foreach = ForeachKernelSchedulerNode(
            V.graph.scheduler,
            [subkernel],
            use_custom_partition_algo=False,
        )
        original_body = snodes[0]._body
        original_group = subkernel.group
        original_read_writes = foreach.read_writes
        tracker = _LoopMutationTracker.create((foreach,))

        snodes[0].expand_dimension_for_pointwise_node(0, 64)
        subkernel.group = (original_group[0], (sympy.S.One, sympy.S.One))
        refresh_group_node_dependencies(subkernel)
        refresh_group_node_dependencies(foreach)
        tracker.finish(rollback=True)

        self.assertIs(snodes[0]._body, original_body)
        self.assertEqual(subkernel.group, original_group)
        self.assertEqual(foreach.read_writes, original_read_writes)

    def test_reorder_modular_indexing(self):
        # [FUNCTION DOC] Regression test for reordering loops of a buffer whose index uses
        # a modular indexing term: the iteration vars of the body and of the reordered
        # body are checked against expected ranges and the modular indexing is kept on
        # the right variable instead of being optimized away.
        """
        There was a bug that we wrongly map i0 to the dimension with size 49
        when reordering the loop and cause ModularIndexing get optimized away
        as an no-op.
        """

        def _create_computed_buffer():
            # [FUNCTION DOC] HELPER: Creates and realizes a pointwise IR with ranges 128, 4,
            # 49, 49 whose load index combines the last two dims with a modular indexing of
            # the first one, decides its layout and returns the computed buffer.
            def inner_fn(index):
                # [FUNCTION DOC] HELPER: Index function that unpacks four indices and loads from
                # a buffer named primal using the modular-indexing based flat index.
                i0, _, i2, i3 = index
                return ops.load(
                    "primal", i3 + 49 * i2 + 2401 * ModularIndexing(i0, 1, 64)
                )

            buf = ir.Pointwise.create(
                device=torch.device(GPU_TYPE),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=[128, 4, 49, 49],
            )
            buf.realize()
            cbuf = buf.data.data
            cbuf.decide_layout()
            return cbuf

        buf = _create_computed_buffer()
        _, body = buf.simplify_and_reorder()
        new_body = body.reorder_iter_loops([1, 2, 3, 0])

        z0, z1, z2, z3 = (sympy_index_symbol(f"p{i}") for i in range(4))
        self.assertEqual(body.var_ranges, {z0: 128, z1: 4, z2: 49, z3: 49})
        self.assertEqual(
            body.indexing_exprs["index0"],
            z3 + 49 * z2 + 2401 * ModularIndexing(z0, 1, 64),
        )
        self.assertEqual(new_body.var_ranges, {z0: 4, z1: 49, z2: 49, z3: 128})
        self.assertEqual(
            new_body.indexing_exprs["index0"],
            z2 + 49 * z1 + 2401 * ModularIndexing(z3, 1, 64),
        )

    def test_weighted_cost_penalizes_uncoalesced(self):
        # [FUNCTION DOC] Checks the weighted cost of memory coalescing records: coalesced
        # bytes count once, uncoalesced bytes are weighted sixteen times, and a small
        # uncoalesced amount outweighs a larger coalesced one.
        self.assertEqual(MemoryCoalescing(10, 0).weighted_cost(), 10)
        self.assertEqual(MemoryCoalescing(0, 10).weighted_cost(), 160)
        # Uncoalesced traffic is what the fusion guard weighs against.
        self.assertGreater(
            MemoryCoalescing(0, 1).weighted_cost(),
            MemoryCoalescing(10, 0).weighted_cost(),
        )

    @parametrize(
        "fused_cost,reject",
        [(100_999_999, False), (101_000_000, False), (101_000_001, True)],
    )
    def test_reindexing_memory_guard_credits_launch(self, fused_cost, reject):
        # [FUNCTION DOC] Parametrized by the fused cost and the expected rejection flag.
        # Builds a scheduler without running its constructor, with unfused memory of two
        # records of 50,000,000 each, a mocked DRAM bandwidth of 1000 and a mocked fused
        # memory. The guard converts costs to time and credits a fixed launch overhead to
        # the unfused pair, so a fused cost up to 101,000,000 is accepted and 101,000,001
        # is rejected. Asserts the guard result equals the expected flag.
        scheduler = Scheduler.__new__(Scheduler)
        unfused_memory = (
            MemoryCoalescing(50_000_000, 0),
            MemoryCoalescing(50_000_000, 0),
        )
        fused_memory = MemoryCoalescing(fused_cost, 0)

        with (
            mock.patch(
                "torch._inductor.scheduler.get_gpu_dram_gbps", return_value=1_000
            ),
            mock.patch.object(
                scheduler, "_selected_tiling_memory", return_value=fused_memory
            ),
        ):
            self.assertEqual(
                scheduler._reindexing_regresses_memory_coalescing(
                    mock.Mock(), mock.Mock(), unfused_memory
                ),
                reject,
            )

    @parametrize("dram_gbps", [None, 0, -1, float("nan"), float("inf")])
    def test_reindexing_memory_guard_with_invalid_bandwidth(self, dram_gbps):
        # [FUNCTION DOC] Parametrized over invalid DRAM bandwidth values (missing, zero,
        # negative, NaN, infinite). The guard then falls back to comparing raw weighted
        # costs; with a fused cost of 101 against an unfused cost of 100 it must reject.
        scheduler = Scheduler.__new__(Scheduler)
        with (
            mock.patch(
                "torch._inductor.scheduler.get_gpu_dram_gbps",
                return_value=dram_gbps,
            ),
            mock.patch.object(
                scheduler,
                "_selected_tiling_memory",
                return_value=MemoryCoalescing(101, 0),
            ),
        ):
            self.assertTrue(
                scheduler._reindexing_regresses_memory_coalescing(
                    mock.Mock(), mock.Mock(), (MemoryCoalescing(100, 0),)
                )
            )

    def test_reindexing_memory_guard_when_bandwidth_lookup_fails(self):
        # [FUNCTION DOC] Checks that when the DRAM bandwidth lookup raises an error, the
        # reindexing guard handles it and rejects the reindexing when the fused cost is
        # slightly larger than the unfused one.
        scheduler = Scheduler.__new__(Scheduler)
        with (
            mock.patch(
                "torch._inductor.scheduler.get_gpu_dram_gbps",
                side_effect=RuntimeError,
            ),
            mock.patch.object(
                scheduler,
                "_selected_tiling_memory",
                return_value=MemoryCoalescing(101, 0),
            ),
        ):
            self.assertTrue(
                scheduler._reindexing_regresses_memory_coalescing(
                    mock.Mock(), mock.Mock(), (MemoryCoalescing(100, 0),)
                )
            )


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "loop_ordering_after_fusion": True,
        "triton.unique_kernel_names": True,
    }
)
@instantiate_parametrized_tests
class LoopOrderingTest(TestCase):
    # [CLASS DOC] End-to-end tests of loop ordering after fusion and loop reindexing.
    # Config patched for kernel benchmarking, loop ordering after fusion and unique
    # kernel names. Tests compile small functions on the GPU, then check numerical
    # equivalence, generated kernel counts (global metrics reset in setUp), bytes
    # accessed or loop reordering counts. The class attribute device names the GPU.
    device = GPU_TYPE

    def do_acc_test(self, f, *args, cast_fp8=True):
        # [FUNCTION DOC] HELPER: Runs the function eagerly and compiled on the same
        # arguments and asserts the results match within tolerance 1e-3. When cast_fp8 is
        # true, any fp8 tensors in the results (nested structures supported) are first
        # cast to float32 because comparing fp8 tensors raises an error.
        expect = f(*args)
        actual = torch.compile(f)(*args)

        if cast_fp8:

            def _cast(x):
                # [FUNCTION DOC] HELPER: Converts a float8 (e5m2 or e4m3fn) tensor to float32 and
                # returns every other value unchanged.
                if isinstance(x, torch.Tensor) and x.dtype in (
                    torch.float8_e5m2,
                    torch.float8_e4m3fn,
                ):
                    return x.to(torch.float32)
                return x

            # Workaround the issue that call allclose on fp8 tensor triggers error
            #   RuntimeError: "mul_cuda" not implemented for 'Float8_e4m3fn'
            expect = tree_map(_cast, expect)
            actual = tree_map(_cast, actual)
        self.assertTrue(same(expect, actual, tol=1e-3))

    def setUp(self):
        # [FUNCTION DOC] Runs the parent setup and resets the global Inductor metrics so
        # kernel counts and byte counts start from zero for each test.
        super().setUp()
        metrics.reset()

    @staticmethod
    def _get_node_origins(node):
        # [FUNCTION DOC] HELPER: Returns the set of origin targets (FX call targets) of
        # the IR nodes behind all plain scheduler nodes contained in the given scheduler
        # node; scheduler nodes without an IR node are skipped.
        return {
            origin.target
            for snode in node.get_nodes()
            if isinstance(snode, SchedulerNode) and snode.node is not None
            for origin in snode.node.get_origins()
        }

    @contextlib.contextmanager
    def _record_reindexing_memory_decisions(self):
        # [FUNCTION DOC] HELPER: Context manager that records each reindexing memory guard
        # decision made during compilation. It temporarily wraps the scheduler guard and
        # the SIMD tiling selection with memory so that, for every guard call, a tuple of
        # the origin targets of both nodes, the guard result and the last tiling selection
        # seen during that call (the fused one when reached, otherwise none) is appended
        # to the list that the context manager yields. Patches are undone on exit.
        """Record (origins, reject, fused_selection) per reindexing guard call."""
        decisions = []
        original = Scheduler._reindexing_regresses_memory_coalescing
        original_selection = SIMDScheduling.select_tiling_with_memory
        last_selection = {}

        def record_selection(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the original tiling selection with memory, stores
            # its result as the latest selection and returns it unchanged.
            selection = original_selection(*args, **kwargs)
            last_selection["value"] = selection
            return selection

        def record(scheduler, node1, node2, unfused_memory):
            # [FUNCTION DOC] HELPER: Replacement for the guard. Clears the stored selection,
            # calls the original guard, appends the origins, the rejection flag and the last
            # selection to the decisions list, and returns the original rejection flag.
            last_selection.clear()
            reject = original(scheduler, node1, node2, unfused_memory)
            origins = self._get_node_origins(node1) | self._get_node_origins(node2)
            # The guard's last tiling selection is the fused one, when reached.
            decisions.append((origins, reject, last_selection.get("value")))
            return reject

        with (
            mock.patch.object(
                Scheduler, "_reindexing_regresses_memory_coalescing", record
            ),
            mock.patch.object(
                SIMDScheduling,
                "select_tiling_with_memory",
                staticmethod(record_selection),
            ),
        ):
            yield decisions

    def test_for_reordering_reindex(self):
        # [FUNCTION DOC] Checks that a reduction over a non-contiguous input followed by a
        # matmul compiles to a single kernel and that the total bytes accessed match the
        # fused reduction plus matmul traffic, i.e. the iter reordering reindex no longer
        # blocks fusion.
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Test function summing the last dimension, adding one and
            # multiplying by a matrix (the matmul prevents the output layout being free).
            """
            Add a matmul since inductor may force layout for output.
            """
            return (x.sum(dim=-1) + 1) @ y

        A, B = 20, 30
        # Make the first 2 dimension not able to merge on purpose so that
        # ComputedBuffer.iter_reoredering_reindex will be updated.
        x = rand_strided([A, A, B], [B, B * A + 300, 1], device=GPU_TYPE)
        y = torch.randn(A, A)

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)
        expected_num_bytes = 0
        expected_num_bytes += A * A * B + A * A  # for the fused reduction
        expected_num_bytes += A * A * 3  # for matmul
        expected_num_bytes *= x.itemsize
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

    def test_apbt_realize(self):
        # [FUNCTION DOC] Checks that two realized pointwise ops on a contiguous and a
        # transposed input end up fused into one kernel with loop ordering after fusion.
        M = 1024
        N = 2048

        def f(x, y):
            # [FUNCTION DOC] HELPER: Realizes the doubled first input and the tripled second
            # input as separate buffers and returns their sum.
            """
            There will be 2 kernels being generated without loop ordering after fusion:
              https://gist.github.com/shunting314/44df83f71de2c110232c50ac6638ed69
            """
            x = realize(x * 2)
            y = realize(y * 3)
            return x + y

        x = torch.randn(M, N)
        y = torch.randn(N, M).t()

        self.do_acc_test(f, x, y)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_sum_and_t(self):
        # [FUNCTION DOC] Checks that a row sum and a transposed contiguous copy of the
        # same input are fused into a single kernel.
        N = 1024

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum over the last dimension and the
            # contiguous transpose of the input.
            return x.sum(dim=-1), x.t().contiguous()

        x = torch.randn(N, N * 2)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red(self):
        # [FUNCTION DOC] Checks that a realized pointwise op followed by a sum over the
        # first two dimensions of a small tensor (no reduction split) gives one kernel.
        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one and sums over dims 0 and 1.
            x = realize(x + 1)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_pw_outer_red_2(self):
        # [FUNCTION DOC] Same as the preceding test but with three chained realized
        # pointwise ops, so the pointwise producer is itself a fused kernel; expects one
        # kernel.
        """
        The pointwise kernel is a fused kernel
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Realizes three chained pointwise ops and sums dims 0, 1.
            x = realize(x + 1)
            x = realize(x - 2)
            x = realize(x * 3)
            return x.sum(dim=[0, 1])

        # make the first 2 dimension small so we don't split the reduction
        x = torch.randn(2, 4, 512)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    @inductor_config.patch(split_reductions=False)
    def test_different_reduction_order(self):
        # [FUNCTION DOC] With reduction splitting disabled, checks that reducing the same
        # input along two different dims gives two kernels and no loop reordering is done,
        # since reordering would not help.
        """
        We should not reorder loops in this case. Since reordering loops does
        not help!
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns sums over dim 0 and over dim 1 of the input.
            return x.sum(dim=0), x.sum(dim=1)

        x = torch.randn(1024, 2048)
        self.do_acc_test(f, x)
        self.assertEqual(2, metrics.generated_kernel_count)
        self.assertEqual(0, metrics.num_loop_reordering)

    def test_keep_fake_dep(self):
        # [FUNCTION DOC] Regression test that loop reordering keeps fake (star) dependencies
        # between a scatter and a following mutation kernel when backpropagating through
        # word and position embeddings with layer norm. The test passes if compiling and
        # running the function raises no exception.
        """
        In this model, there are fake dependencies (StarDep) between Scatter
        and a following mutation kernel that computes the gradients of
        the embedding tables.

        When we do loop reordering for the mutation kernel, we re-analyze
        the node's dependencies. But the analysis result does not contains
        those fake dependencies. Have to add them back manually.
        """
        V = 2048
        hidden_size = 64
        max_seqlen = 512
        batch_size = 8

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with word and position embeddings and a layer
            # norm over the hidden size.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two embedding tables and the layer norm.
                super().__init__()
                self.word_embeddings = nn.Embedding(V, hidden_size)
                self.position_embeddings = nn.Embedding(max_seqlen, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)

            def forward(self, input_ids, labels, position_ids):
                # [FUNCTION DOC] HELPER: Adds word and position embeddings and applies layer norm;
                # the labels argument is unused.
                emb = self.word_embeddings(input_ids) + self.position_embeddings(
                    position_ids
                )
                return self.layer_norm(emb)

        m = Model()

        @torch.compile
        def f(*args):
            # [FUNCTION DOC] HELPER: Compiled function that runs the module and calls backward
            # on the sum of its output; returns nothing.
            m(*args).sum().backward()

        input_ids = torch.randint(0, V, (batch_size, max_seqlen))
        labels = torch.randint(0, V, (batch_size, max_seqlen))
        position_ids = torch.arange(max_seqlen)[None, :]
        # Make sure this line does not raise exceptions. If we miss
        # fake dependencies after loop reordering, we may get exception that
        # some buffer is used before being defined.
        f(input_ids, labels, position_ids)

    def test_different_broadcast_shapes(self):
        # [FUNCTION DOC] Checks that two pointwise ops sharing a broadcast input of
        # different shapes are not fused, giving two kernels.
        def f(x, y, c):
            # [FUNCTION DOC] HELPER: Adds the third argument to each of the first two.
            return x + c, y + c

        x = torch.randn(4, 256, 1024)
        y = torch.randn(2, 512, 1024)
        c = torch.randn(1024)
        self.do_acc_test(f, x, y, c)

        # The two kernels are not fused due to c is broadcasted
        self.assertEqual(2, metrics.generated_kernel_count)

    def test_view(self):
        # [FUNCTION DOC] Checks that a pointwise result and a view of the input, realized,
        # fuse into one kernel; this relies on comparing normalized memory dependencies
        # (contiguous loops merged) when matching dependencies.
        """
        Passing this test relies that we compare normalized MemoryDep.
        Normlaization here means merging contiguous loops.

        To make loop reordering work, we don't merge loops when creating
        SchedulerNode. Thus we need explicitly normalize MemoryDep when
        we check if two MemeoryDep matches.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sine of the input and a realized view of the
            # input as 10 by 10.
            y = x.sin()
            x = realize(x.view(10, 10))
            return x, y

        x = torch.randn(100)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_reshape_reindexing_for_reduction(self):
        # [FUNCTION DOC] Checks the RMS-norm pattern where a reshape changes the loop
        # decomposition: reindexing lets the pointwise and the reduction fuse into one
        # kernel. Uses a sliced non-contiguous bfloat16 input, compares compiled against
        # eager with tolerance 1e-2, and optionally benchmarks when the perf flag is set.
        """
        RMS norm pattern where reshape(-1, head_dim) changes the loop
        decomposition from [M, N] to [M*num_heads, head_dim]. Without
        reindexing, the pointwise and reduction have different MemoryDep
        indexing and can't fuse. With reindexing, the pointwise's
        iteration is re-factored to match the reduction's, enabling fusion
        into a single kernel.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: RMS norm over groups of 128 elements: reshape, float
            # conversion, mean of squares, scale by reciprocal square root and reshape back.
            head_dim = 128
            M, N = x.shape
            x_reshaped = x.reshape(-1, head_dim)
            x_f32 = x_reshaped.float()
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
            return x_normed.reshape(M, N).to(x.dtype)

        if DO_PERF_TEST:
            M = 1024
        else:
            M = 16
        # Non-contiguous input (simulating a slice from qkv projection)
        qkv = torch.randn(M, 10240, dtype=torch.bfloat16)
        x = qkv[:, :8192]

        ref = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(1, metrics.generated_kernel_count)

        if DO_PERF_TEST:
            from triton.testing import do_bench

            optf = torch.compile(f)
            print(f"ms={do_bench(lambda: optf(x))}")

    @inductor_config.patch("triton.coalesce_tiling_analysis", True)
    def test_reshape_reindexing_transposed_input(self):
        # [FUNCTION DOC] Same RMS-norm pattern as the previous test with a transposed
        # input and coalescing tiling analysis enabled. Besides numerical and one-kernel
        # checks, asserts through the recorded guard decisions that a non-rejected
        # decision involving the mean exists whose tiling is split into y, x and r0_
        # with extents 16, 64 and 128.
        """
        Same RMS norm pattern but with a transposed input. The reshape
        sees transposed strides, so the reduction's memory access
        pattern differs from the contiguous case. Reindexing should
        still enable fusion.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: RMS norm over groups of 128 elements as in the previous
            # test, with reshape back to the input shape.
            head_dim = 128
            M, N = x.shape
            x_reshaped = x.reshape(-1, head_dim)
            x_f32 = x_reshaped.float()
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
            return x_normed.reshape(M, N).to(x.dtype)

        M, N = 16, 8192
        # Transposed input: shape [M, 8192] but stride (1, M)
        x = torch.randn(N, M, dtype=torch.bfloat16).T

        ref = f(x)
        with self._record_reindexing_memory_decisions() as decisions:
            actual = torch.compile(f)(x)
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(1, metrics.generated_kernel_count)
        self.assertTrue(
            any(
                not reject
                and selection is not None
                and selection.tiling == {"y": M, "x": 64, "r0_": 128}
                for origins, reject, selection in decisions
                if torch.ops.aten.mean.dim in origins
            )
        )

    @inductor_config.patch("loop_ordering_after_fusion", False)
    def test_reshape_reindexing_without_loop_ordering(self):
        # [FUNCTION DOC] Checks that reindexing alone enables fusion of the RMS-norm
        # pattern even when loop ordering after fusion is turned off; expects numerical
        # agreement and a single kernel.
        """
        Reindexing should enable fusion even when loop ordering is
        disabled. Same RMS norm pattern as test_reshape_reindexing_for_reduction.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: RMS norm over groups of 128 elements as above.
            head_dim = 128
            M, N = x.shape
            x_reshaped = x.reshape(-1, head_dim)
            x_f32 = x_reshaped.float()
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
            return x_normed.reshape(M, N).to(x.dtype)

        M = 16
        qkv = torch.randn(M, 10240, dtype=torch.bfloat16)
        x = qkv[:, :8192]

        ref = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_reindex_unfusable_write_read_dep(self):
        # [FUNCTION DOC] Checks grouped quantization where a registered custom op consumes
        # both the quantized tensor and the scale, so the reduction and scale epilogue are
        # fused but the quantization pointwise stays separate. Asserts that reindexing
        # still fires (write-read dependency prioritization) and results in one kernel.
        """
        Grouped quantization where a custom op consumes both the
        quantized tensor and the scale, forcing the reduction (amax)
        and scale epilogue to fuse into a FusedSchedulerNode while
        the quantize pointwise remains separate.

        The FusedSchedulerNode's dep on the input has 3 vars (from the
        reduction + scale bodies), while the pointwise has 2 vars.
        This num_vars mismatch causes _try_reorder_loops_for_candidates
        to return a score based on the shared read (input), short-
        circuiting reindexing. The write-read dep prioritization
        detects the unfusable dep and returns -1, letting reindexing
        fire.
        """
        HIDDEN, GROUP_SIZE = 7168, 128

        @torch.library.custom_op("test::opaque_gemm", mutates_args=())
        def opaque_gemm(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom operator (no mutation) standing in for a gemm; it
            # returns a zero bfloat16 tensor with one row per input row and 1024 columns.
            return torch.zeros(
                x_q.shape[0], 1024, device=x_q.device, dtype=torch.bfloat16
            )

        @opaque_gemm.register_fake
        def _(x_q, scale):
            # [FUNCTION DOC] HELPER: Fake (meta) implementation of the custom op; returns a
            # zero bfloat16 tensor of the same shape as the real op.
            return torch.zeros(
                x_q.shape[0], 1024, device=x_q.device, dtype=torch.bfloat16
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Groups the last dimension into blocks of 128, computes
            # the per-block absolute max, derives a scale clamped to a minimum, divides and
            # clamps the values, converts to float16, restores the shape and passes the
            # quantized values and scale to the custom op.
            grouped = x.reshape(-1, HIDDEN // GROUP_SIZE, GROUP_SIZE).float()
            absmax = grouped.abs().amax(dim=-1, keepdim=True)
            scale = (absmax / FP8_MAX).clamp(min=1e-6)
            x_q = (
                (grouped / scale)
                .clamp(-FP8_MAX, FP8_MAX)
                .to(torch.float16)
                .reshape(x.shape)
            )
            scale = scale.squeeze(-1)
            return torch.ops.test.opaque_gemm(x_q, scale)

        FP8_MAX = 448.0  # arbitrary clamp range
        x = torch.randn(8, HIDDEN, dtype=torch.bfloat16)
        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_reindex_rollback_on_no_improvement(self):
        # [FUNCTION DOC] Checks that when reindexing does not improve the fusion score
        # (a reduction and a pointwise read different slices of the same buffer) the node
        # state is rolled back; the test only checks the compiled result matches eager.
        """
        When reindexing is attempted but doesn't improve the fusion
        score, the node state should be rolled back. Here a reduction
        and pointwise both read from x but at different slices (offsets).
        They share the buffer and have the same iteration numel, so
        reindexing is attempted, but the offset means deps still don't
        match after reindexing. The rollback restores the original
        node state so the pointwise isn't left with a wrong iteration
        domain.
        """
        M, N = 16, 128

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the first N columns and the double of
            # the remaining columns.
            r = x[:, :N].sum(dim=-1)
            p = x[:, N:] * 2
            return r, p

        x = torch.randn(M, N * 2, device=GPU_TYPE)
        self.do_acc_test(f, x)

    def test_floordiv_broadcast_vertical_fusion(self):
        # [FUNCTION DOC] Regression test (issue 183542) for a blockwise absmax reduction
        # followed by a pointwise that broadcasts the per-block value back to elements
        # through a floor division index. Expects match with eager and a single kernel.
        """
        Block-wise quantization pattern: a reduction produces per-block
        values (e.g., amax) and a following pointwise broadcasts them
        back to element granularity via unsqueeze, creating a FloorDiv
        index pattern.

        The reduction writes buf[G*p0 + p1] (p1 in [0, G)) and the
        pointwise reads buf[G*p0 + (p1 // block_size)] (p1 in [0, N)).
        Without reindexing in can_fuse_vertical, this FloorDiv mismatch
        causes "memory deps did not match" rejection, producing an
        extra kernel.

        Regression test for https://github.com/pytorch/pytorch/issues/183542
        """
        BLOCK_SIZE = 32

        def f(x):
            # [FUNCTION DOC] HELPER: Splits the last dim into blocks of 32, takes block max
            # of absolute values, clamps it and divides the blocks by it; flattens back.
            M, N = x.shape
            x_blocked = x.unflatten(-1, (N // BLOCK_SIZE, BLOCK_SIZE))
            # Per-block reduction (like amax in MXFP8)
            block_max = x_blocked.abs().amax(dim=-1)
            scale = block_max.clamp(min=1e-6)
            # Broadcast scale back to elements via unsqueeze (FloorDiv pattern)
            x_scaled = x_blocked / scale.unsqueeze(-1)
            return x_scaled.flatten(-2)

        M, N = 8, 8192
        x = torch.randn(M, N, dtype=torch.float32)
        self.do_acc_test(f, x)
        # Block reduction + broadcast pointwise should fuse into 1 kernel
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_floordiv_broadcast_with_preceding_reduction(self):
        # [FUNCTION DOC] Regression test (issue 183542) for RMS norm followed by blockwise
        # quantization. Resets dynamo and metrics, compares eager and compiled results
        # with tolerance 1e-2 and expects two kernels (variance reduction and block
        # reduction), i.e. the floor division broadcast must not add a third.
        """
        RMSNorm followed by block-wise quantization: two reductions
        with different rnumel (variance over K, then amax over
        block_size=32) separated by pointwise ops.

        Expected: 2 kernels (variance reduction fused with norm,
        block reduction fused with broadcast pointwise).
        The FloorDiv broadcast between block reduction and the final
        pointwise must not cause a third kernel.

        Regression test for https://github.com/pytorch/pytorch/issues/183542
        """
        BLOCK_SIZE = 32

        def f(x, weight):
            # [FUNCTION DOC] HELPER: RMS norm with weight in bfloat16 followed by blockwise
            # absmax scaling in float32 over blocks of 32; returns the flattened result.
            # RMSNorm
            x_fp32 = x.to(torch.float32)
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            normed = x_fp32 * torch.rsqrt(variance + 1e-5)
            normed = normed.to(torch.bfloat16) * weight

            # Block-wise quantization (simplified, no FP8 cast)
            M, N = normed.shape
            normed_fp32 = normed.to(torch.float32)
            blocked = normed_fp32.unflatten(-1, (N // BLOCK_SIZE, BLOCK_SIZE))
            block_max = blocked.abs().amax(dim=-1)
            scale = block_max.clamp(min=1e-6)
            x_scaled = blocked / scale.unsqueeze(-1)
            return x_scaled.flatten(-2)

        M, K = 8, 8192
        x = torch.randn(M, K, dtype=torch.bfloat16)
        weight = torch.randn(K, dtype=torch.bfloat16)

        torch._dynamo.reset()
        metrics.reset()
        expect = f(x, weight)
        actual = torch.compile(f)(x, weight)
        self.assertTrue(same(expect, actual, tol=1e-2))
        # variance reduction + block reduction = 2 kernels
        self.assertEqual(2, metrics.generated_kernel_count)

    @inductor_config.patch(
        layout_optimization=True,
        force_layout_optimization=True,
    )
    @inductor_config.patch("triton.coalesce_tiling_analysis", True)
    def test_upsample_into_reduction_keeps_coalesced_loops(self):
        # [FUNCTION DOC] Regression test (issue 189488) with layout optimization forced
        # and coalescing tiling analysis on. Runs a module with upsampling, convolution
        # and group norm under no-grad, a mocked DRAM bandwidth of 3350 and the recorded
        # guard decisions. Asserts at least one decision involves the var_mean origin, was
        # rejected and had a selection with uncoalesced memory, so reindexing the
        # upsample onto the split group-norm reduction is refused.
        """Reindexing upsample onto GroupNorm's split loses coalescing (#189488)."""

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, a group norm and a 1 by 1
            # skip convolution.
            def __init__(self, channels, groups):
                # [FUNCTION DOC] HELPER: Creates the convolution, group norm (eps 1e-6) and skip
                # convolution for the given channel and group counts.
                super().__init__()
                self.conv = nn.Conv2d(channels, channels, 3, padding=1)
                self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
                self.skip = nn.Conv2d(channels, channels, 1)

            def forward(self, x, residual):
                # [FUNCTION DOC] HELPER: Adds the residual, scales by an inverse square root of
                # two, upsamples by two with nearest mode, applies the convolution, then returns
                # the silu of the group norm plus a small multiple of the skip convolution.
                x = (x + residual) / 1.41421356237
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = self.conv(x)
                y = F.silu(self.norm(x))
                return y + (self.skip(x) * 0.01)

        batch, channels, groups, spatial = 8, 128, 32, 64
        mod = Mod(channels, groups).eval()
        x = torch.randn(batch, channels, spatial // 2, spatial // 2)
        residual = torch.randn_like(x)
        target_origins = {torch.ops.aten.var_mean.correction}

        with (
            torch.no_grad(),
            mock.patch(
                "torch._inductor.scheduler.get_gpu_dram_gbps",
                return_value=3_350,
            ),
            self._record_reindexing_memory_decisions() as decisions,
        ):
            self.do_acc_test(mod, x, residual)
        self.assertTrue(
            any(
                target_origins <= origins
                and reject
                and selection is not None
                and selection.memory.uncoalesced > 0
                for origins, reject, selection in decisions
            ),
            decisions,
        )

    @inductor_config.patch("triton.coalesce_tiling_analysis", True)
    def test_horizontal_reindexing_memory_guard(self):
        # [FUNCTION DOC] Checks that the memory guard is also applied to horizontal
        # (non-ancestor) reindexing. The guard is patched to force rejection for the add
        # and sum pair when the nodes are not vertically related, while recording the real
        # decision; expects two kernels and that some recorded real decision was accept.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one and the row sums after
            # reshaping to rows of 128.
            return x + 1, x.reshape(-1, 128).sum(dim=-1)

        target_origins = {torch.ops.aten.add.Tensor, torch.ops.aten.sum.dim_IntList}
        target_decisions = []
        original = Scheduler._reindexing_regresses_memory_coalescing

        def reject_target(scheduler, node1, node2, unfused_memory):
            # [FUNCTION DOC] HELPER: Guard replacement: computes the original decision; if the
            # nodes contain the add and sum origins and neither is an ancestor of the other
            # (vertical check via operation names and ancestors), records the original
            # decision and returns True; otherwise returns the original decision.
            reject = original(scheduler, node1, node2, unfused_memory)
            origins = self._get_node_origins(node1) | self._get_node_origins(node2)
            is_vertical = bool(node1.get_operation_names() & node2.ancestors)
            if target_origins <= origins and not is_vertical:
                target_decisions.append(reject)
                return True
            return reject

        x = torch.randn(16, 8192)
        with mock.patch.object(
            Scheduler, "_reindexing_regresses_memory_coalescing", reject_target
        ):
            self.assertEqual(torch.compile(f)(x), f(x))
        self.assertEqual(2, metrics.generated_kernel_count)
        self.assertIn(False, target_decisions)

    @inductor_config.patch(
        {
            "assert_indirect_indexing": False,
            "benchmark_kernel": False,
            "fx_graph_cache": False,
            "loop_index_inversion_in_fusion": True,
            "loop_ordering_after_fusion": False,
            "loop_reindexing_after_fusion": False,
        }
    )
    def test_rejected_index_inversion_rollback(self):
        # [FUNCTION DOC] Checks rollback after a fusion that needed index inversion is
        # rejected. Config disables loop ordering, reindexing, caching and indirect index
        # asserts and enables index inversion in fusion. A custom choices handler refuses
        # fusion when the second node has a non-contiguous write. Expects the handler to
        # have rejected something, two generated kernels and a result equal to eager.
        from torch._inductor.choices import InductorChoices

        class RejectInvertedFusion(InductorChoices):
            # [CLASS DOC] HELPER: Choices subclass that rejects fusion when the second node
            # writes a non-contiguous index, remembering this in the rejected_inversion flag.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the flag recording that a fusion was rejected
                # to False; does not call the parent initializer.
                self.rejected_inversion = False

            def can_fuse(self, scheduler, node1, node2, shared_data_score):
                # [FUNCTION DOC] HELPER: Fusion decision hook. Returns False and sets the flag if
                # any write of the second node is not contiguous; otherwise defers to the default
                # fusion decision of the base class.
                if any(not write.is_contiguous() for write in node2.read_writes.writes):
                    self.rejected_inversion = True
                    return False
                return InductorChoices.can_fuse(
                    scheduler, node1, node2, shared_data_score
                )

        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one, then gathers it with a
            # permuted index built from an arange, realizes it, and multiplies by three.
            y = realize(x + 1)
            p = torch.arange(8, device=x.device)
            z = realize(y[4 * (p % 2) + p // 2])
            return z * 3

        x = torch.arange(8, device=self.device, dtype=torch.float32)
        choices = RejectInvertedFusion()

        with V.set_choices_handler(choices):
            actual = torch.compile(f, fullgraph=True)(x)

        self.assertTrue(choices.rejected_inversion)
        self.assertEqual(metrics.generated_kernel_count, 2)
        self.assertEqual(actual, f(x))

    def test_reindex_attention_layout_clone_for_index_inversion(self):
        # [FUNCTION DOC] Regression test (issue 188635): an attention-style layout clone
        # (view then transpose then contiguous) after a realized add should invert the
        # index exactly once (counted via a wrapped inverse formula generator) and yield
        # one kernel, with results matching eager.
        """Regression test for https://github.com/pytorch/pytorch/issues/188635."""

        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one, views it as four dims,
            # swaps dims 1 and 2 and makes it contiguous.
            y = realize(x + 1)
            return y.view(2, 16, 4, 8).transpose(1, 2).contiguous()

        x = torch.randn(2, 16, 32, device=self.device)
        with mock.patch(
            "torch._inductor.invert_expr_analysis.generate_inverse_formula",
            wraps=generate_inverse_formula,
        ) as inverse:
            self.do_acc_test(f, x)

        self.assertEqual(inverse.call_count, 1)
        self.assertEqual(metrics.generated_kernel_count, 1)

    def test_reindex_block_scale_swizzle_with_epilogue(self):
        # [FUNCTION DOC] Checks reindexing of a block scale swizzle layout pattern with an
        # extra pointwise epilogue: results match eager and one kernel is generated.
        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one, applies a blocked
            # permutation and reshape swizzle of the 2D tensor and adds one.
            y = realize(x + 1)
            rows, cols = y.shape
            blocks = y.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
            blocks = blocks.reshape(-1, 4, 32, 4).transpose(1, 2)
            return blocks.reshape(rows, cols) + 1

        x = torch.randn(128, 128, device=self.device)
        self.do_acc_test(f, x)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @inductor_config.patch(fx_graph_cache=False)
    def test_noninvertible_reindex_skips_loop_mutation(self):
        # [FUNCTION DOC] Checks that when the access pattern is not invertible, loop
        # reindexing is never applied to a scheduler node. The reindex method is wrapped to
        # set a flag; asserts the flag is still False and the result matches eager.
        reindexed = False
        original_apply = SchedulerNode.apply_loop_reindexing

        def record_apply(node, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that marks the enclosing flag as set and forwards
            # to the original reindexing method.
            nonlocal reindexed
            reindexed = True
            return original_apply(node, *args, **kwargs)

        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one and returns a cloned strided
            # view with overlapping reads.
            y = realize(x + 1)
            return torch.as_strided(y, (2, 3, 2), (6, 1, 1)).clone()

        x = torch.randn(2, 6, device=self.device)
        with mock.patch.object(
            SchedulerNode,
            "apply_loop_reindexing",
            record_apply,
        ):
            actual = torch.compile(f, fullgraph=True)(x)

        self.assertFalse(reindexed)
        self.assertEqual(actual, f(x))

    @inductor_config.patch(fx_graph_cache=False)
    def test_rejected_reindex_for_index_inversion_rollback(self):
        # [FUNCTION DOC] Checks rollback when a fusion on a reindexed node is rejected.
        # A choices handler rejects fusions where the second node has loop sizes of 1024
        # and records that size; the test expects at least one record, that all records
        # equal that size, and that the result matches eager.
        from torch._inductor.choices import InductorChoices

        observed_sizes = []

        class RejectReindexedFusion(InductorChoices):
            # [CLASS DOC] HELPER: Choices subclass whose fusion hook rejects exactly the
            # second nodes with a single loop of size 1024.
            def can_fuse(self, scheduler, node1, node2, shared_data_score):
                # [FUNCTION DOC] HELPER: Fusion decision hook. When the second node is a plain
                # scheduler node with sizes equal to 1024 it records the size in the outer list
                # and returns False; otherwise defers to the base class decision.
                node2_sizes = (
                    tuple(node2._sizes[0]) if isinstance(node2, SchedulerNode) else ()
                )
                if node2_sizes == (1024,):
                    observed_sizes.append(node2_sizes)
                    return False
                return InductorChoices.can_fuse(
                    scheduler, node1, node2, shared_data_score
                )

        def f(x):
            # [FUNCTION DOC] HELPER: Realizes the input plus one then views, swaps dims 1, 2
            # and makes it contiguous.
            y = realize(x + 1)
            return y.view(2, 16, 4, 8).transpose(1, 2).contiguous()

        x = torch.randn(2, 16, 32, device=self.device)
        choices = RejectReindexedFusion()
        with V.set_choices_handler(choices):
            actual = torch.compile(f, fullgraph=True)(x)

        self.assertTrue(observed_sizes)
        self.assertEqual(set(observed_sizes), {(1024,)})
        self.assertEqual(actual, f(x))

    def test_reshape_reindexing_fused_pointwise(self):
        # [FUNCTION DOC] Checks reindexing where the pointwise side is a fused scheduler
        # node: add and clamp are realized separately, fuse together, then fuse with the
        # RMS-norm reduction after reindexing. Expects match with eager and one kernel,
        # and resets dynamo at the end.
        """
        Redecomposition where the pointwise side is a FusedSchedulerNode.
        realize() forces ops to materialize as separate nodes, so
        add and clamp become two SchedulerNodes that fuse into a
        FusedSchedulerNode before the reindexing fuses them with
        the reduction.
        """

        def f(x, bias):
            # [FUNCTION DOC] HELPER: Realizes an add of a bias and a clamp to -1 and 1, then
            # applies the RMS norm over groups of 128.
            head_dim = 128
            M, N = x.shape
            y = realize(x + bias)
            z = realize(y.clamp(-1, 1))
            x_reshaped = z.reshape(-1, head_dim)
            x_f32 = x_reshaped.float()
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
            return x_normed.reshape(M, N).to(x.dtype)

        M = 16
        qkv = torch.randn(M, 10240, dtype=torch.bfloat16)
        x = qkv[:, :8192]
        bias = torch.randn(8192, dtype=torch.bfloat16)

        ref = f(x, bias)
        actual = torch.compile(f)(x, bias)
        torch.testing.assert_close(actual, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(1, metrics.generated_kernel_count)
        torch._dynamo.reset()

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 requires H100+ and MI300+")
    @unittest.skipIf(not SM90OrLater, "sm89 errors out on this test")
    def test_fp8_cast_and_t(self):
        # [FUNCTION DOC] Reproduces the non-fusion of fp8 cast and transpose (issue
        # 130015). Skipped without fp8 platform support or before SM90. Expects match with
        # eager (fp8 results compared as float32) and one kernel.
        """
        This test repros the not able to fuses issue in
        https://github.com/pytorch/pytorch/issues/130015
        for fp8 cast and transpose
        """

        def f(x, scale):
            # [FUNCTION DOC] HELPER: Scales and clamps the input to the fp8 e4m3 range, casts
            # to fp8 and returns both it and a transposed contiguous copy turned back to the
            # original orientation. Uses the max value defined later in the enclosing test.
            x = x * scale
            x = x.clamp(-1 * E4M3_MAX_POS, E4M3_MAX_POS)
            x = x.to(torch.float8_e4m3fn)
            x_t = x.t().contiguous().t()
            return x, x_t

        x = torch.randn(4096, 4096, dtype=torch.bfloat16)
        scale = torch.Tensor([10.0]).to(GPU_TYPE)
        E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max

        self.do_acc_test(f, x, scale)
        self.assertEqual(1, metrics.generated_kernel_count)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 requires H100+ and MI300+")
    def test_fp8_pattern_2(self):
        # [FUNCTION DOC] Reproduces the fp8 fusion issue 133242. Skipped without fp8
        # support. Compiles the pattern, expects one kernel and that bytes accessed equal
        # scale plus input plus both fp8 outputs.
        """
        This test repros the fp8 fusion relation issue here:
            https://github.com/pytorch/pytorch/issues/133242
        """
        ref_dtype = torch.bfloat16
        M, K = 4096, 4096

        input_tensor = torch.randn(
            M, K, device=GPU_TYPE, dtype=ref_dtype, requires_grad=False
        )
        scale = torch.Tensor([10.0]).to(GPU_TYPE)

        E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max

        def test_pattern2(tensor_x_inp, scale_x):
            # [FUNCTION DOC] HELPER: Computes an fp8 cast of the scaled clamped input and an
            # fp8 cast of the transposed scaled clamped input made contiguous and transposed
            # back; returns both.
            tensor_x = tensor_x_inp * scale_x
            tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8 = tensor_x.to(torch.float8_e4m3fn)

            tensor_x_t = (tensor_x_inp * scale_x).t()
            tensor_x_t = tensor_x_t.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8_t = tensor_x_t.to(torch.float8_e4m3fn)

            tensor_fp8_t = tensor_fp8_t.contiguous().t()

            return (tensor_fp8, tensor_fp8_t)

        test_pattern = torch.compile(test_pattern2)
        tensor_fp8, tensor_fp8_t = test_pattern(input_tensor, scale)

        self.assertEqual(1, metrics.generated_kernel_count)

        expected_numbytes = scale.nbytes  # scalar
        expected_numbytes += input_tensor.nbytes  # input
        expected_numbytes += tensor_fp8.nbytes + tensor_fp8_t.nbytes  # output
        self.assertEqual(expected_numbytes, metrics.num_bytes_accessed)

    def test_outer_dimension_softmax(self):
        # [FUNCTION DOC] Reproduces the non-fusion of softmax over the outer dimension
        # (issue 93718): compares with eager and expects one kernel.
        """
        This test repros the not able to fuse problem for outer dimension
        softmax reported here: https://github.com/pytorch/pytorch/issues/93718

        Perf data on h100:
        - without loop ordering after fusion 0.564 ms
        - with loop ordering after fusion 0.302 ms
        This is 1.87x speedup.

        """
        x = torch.randn(32, 2**21, device=GPU_TYPE)

        def f(x):
            # [FUNCTION DOC] HELPER: Softmax over dim 0.
            return F.softmax(x, dim=0)

        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_outer_dimension_sum_fuse_with_pw(self):
        # [FUNCTION DOC] Checks fusion of an outer-dimension sum with a following
        # pointwise that uses it: one kernel and optional benchmark if perf flag is set.
        """
        Test the fusion of an outer dimension sum with a followed pointwise.
        Perf data on h100:
        - without loop ordering after fusion 0.436 ms
        - with loop ordering after fusion 0.260 ms
        This is 1.68x speedup.
        """
        x = torch.randn(32, 2**21, device=GPU_TYPE)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the sum over dim 0 (kept) added to the input.
            return x.sum(dim=0, keepdim=True) + x

        self.do_acc_test(f, x)
        self.assertEqual(1, metrics.generated_kernel_count)

        if DO_PERF_TEST:
            from triton.testing import do_bench

            optf = torch.compile(f)
            print(f"ms={do_bench(lambda: optf(x))}")

    # Disable split reduction to make it easier to calculate the expected
    # number of bytes accessed. In this case, split reduction does not
    # help perf much.
    @inductor_config.patch(split_reductions=False)
    def test_fuse_reduction_with_tiled_pw(self):
        # [FUNCTION DOC] With reduction splitting disabled, checks that a double sum and
        # two pointwise results (scaled copy and transposed contiguous copy) are fused so
        # the input is read once. Asserts the compiled output matches eager and that
        # bytes accessed equal three times the input plus a small term for the
        # intermediate reduction traffic. Prints the expected bytes; optionally benchmarks.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns a double sum, the input divided by ten and its
            # transposed contiguous copy turned back to original orientation.
            y = torch.sum(torch.sum(x, dim=-1))

            z = x / 10.0
            z_t = z.t().contiguous().t()
            return y, z, z_t

        # use this input sizes to test for perf
        if DO_PERF_TEST:
            M, N = 1024 * 32, 1024 * 8
        else:
            M, N = 200, 100
        x = torch.randn(M, N, device=GPU_TYPE)
        actual = f(x)
        opt_f = torch.compile(f)
        expected = opt_f(x)
        self.assertTrue(same(actual, expected, tol=1e-3))

        # We should fuse the first sum with the two pointwise.
        # Overall we read x once for all these three kernels and write
        # out 2 buffers with the same size as x.
        # This should be sort of 'optimal' for this workload.
        expected_numbytes = x.nbytes * 3

        # A small amount of extra memory access for:
        # - store output for the first reduction
        # - load input for the second reduction
        # - store output for the second reduction
        expected_numbytes += (M * 2 + 1) * x.itemsize

        print(expected_numbytes)
        self.assertEqual(expected_numbytes, metrics.num_bytes_accessed)

        if DO_PERF_TEST:
            from triton.testing import do_bench

            ms = do_bench(lambda: opt_f(x))
            print(f"{ms=:.3f}")

    def test_factored_vs_expanded_pw_numel_in_fused_group(self):
        # [FUNCTION DOC] Regression test (issue 181563) for a fused group whose pointwise
        # element count appears in a factored form in the node body and an expanded form
        # in the group (from a strided convolution with dynamic spatial sizes), which
        # previously fell through an assertion in splitting pointwise and reduction
        # ranges. Passes if compiling with dynamic dims under no-grad matches eager.
        """
        Regression test for https://github.com/pytorch/pytorch/issues/181563

        Conv2d(kernel=1, stride=4) decomposes its output spatial size as
        ``1 + (T - 1)//4``, so when those ranges are multiplied together for
        a node body via ``sympy_product`` the result stays factored
        (``s*(a+1)*(b+1)``). The fused-group's ``pointwise_numel`` for the
        same value goes through ``SizeVarAllocator.simplify`` which calls
        ``sympy.expand`` and yields the expanded form
        (``s + s*a + s*b + s*a*b``). Sympy's structural ``==`` says these are
        not equal, so without the fix the early-return guard in
        ``get_pw_red_splits`` is missed and we fall through to an assert
        that can never hold for a non-reduction body in a group whose
        ``red_numel > 1``.
        """

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module containing a 1 by 1 convolution of stride 4 with
            # eight channels and no bias.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the stride-4 convolution.
                super().__init__()
                self.proj = nn.Conv2d(8, 8, kernel_size=1, stride=4, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution, builds a bias from the first
                # channel, adds it to a channel-last copy, and returns the softmax over channels
                # together with the bias; the bias is a separate node sharing the symbolic shape.
                a = self.proj(x)
                # Outer-only pointwise sharing `a`'s sym shape; returning it
                # forces materialization as its own SchedulerNode rather than
                # being inlined into the softmax kernel.
                bias = a[:, 0]
                bias = bias * 2 + 1
                a = a.permute(0, 2, 3, 1).contiguous()
                a = a + bias.unsqueeze(-1)
                b = a.softmax(dim=-1)  # red_numel = 8 in the fused group
                return b, bias

        mod = Mod().to(GPU_TYPE)
        x = torch.randn(2, 8, 17, 19, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.mark_dynamic(x, 3)

        with torch.no_grad():
            self.do_acc_test(mod, x)

    @inductor_config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.max_mm_configs": 4,
        }
    )
    @skipUnless(HAS_GPU and is_big_gpu(), "Need big gpu for max-autotune")
    def test_interaction_with_triton_template(self):
        # [FUNCTION DOC] Checks, for max-autotune with Triton gemm templates on big GPUs,
        # that the dependency prefix of a Triton template and its prologue match. Compiles
        # an expand plus add followed by a matmul, then checks via the generated code that
        # the call function exists and the kernel run count is within one of the fused
        # expectation (one kernel, plus one if kernel benchmarking is enabled).
        """
        Make sure the dependency prefix for TritonTempalate and its
        prologue match.
        """

        @torch.compile
        def f(x, y):
            # [FUNCTION DOC] HELPER: Expands the first input, adds one and matrix-multiplies
            # with the second input.
            return (x.expand([1, y.shape[0]]) + 1) @ y

        x = torch.randn([1, 1], device=GPU_TYPE)
        y = torch.randn([64, 128], device=GPU_TYPE)

        out, code = run_and_get_code(f, x, y)

        FileCheck().check("def call(").run(code[0])
        # Prologue fused with mm: 1 kernel. Unfused: 2 kernels (expand+add + mm).
        # With benchmark_kernel, add 1 for the benchmarking code path.
        base_expected = 1 + int(inductor_config.benchmark_kernel)
        run_count = code[0].count(".run(")
        self.assertGreaterEqual(
            run_count, base_expected, "Expected at least one kernel launch"
        )
        self.assertLessEqual(
            run_count,
            base_expected + 1,
            "Prologue fusion produces 1 kernel; unfused produces 2",
        )

    @inductor_config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.max_mm_configs": 4,
        }
    )
    @skipUnless(HAS_GPU and is_big_gpu(), "Need big gpu for max-autotune")
    def test_interaction_with_multi_template(self):
        # [FUNCTION DOC] Checks that multi-template buffers are skipped during loop
        # reordering: a matmul and a separate add of small tensors give exactly two Triton
        # kernels in the generated code, since fusion was not worth it.
        """
        Skip MultiTemplateBuffer during loop reordering
        """

        @torch.compile
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the matmul of the inputs and the first input plus
            # one.
            return (x @ y), x + 1

        N = 2
        x = torch.randn([N, N], device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn([N, N], device=GPU_TYPE, dtype=torch.bfloat16)

        out, code = run_and_get_code(f, x, y)
        # didn't fuse due to small savings
        FileCheck().check_count("@triton.jit", 2, exactly=True).run(code[0])

    def test_fuse_with_scalar_shared_memory(self):
        # [FUNCTION DOC] Checks that two nodes sharing a scalar can still be fused when
        # loop ordering after fusion is applied: mean of a small tensor yields exactly one
        # Triton kernel in the generated code.
        """
        Make sure if we can fuse two nodes sharing a scalar before,
        we can still do it with LOAF applied.

        This is not really a big deal. But some tests rely on this and
        less number of kernels has some small benefits.
        """

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the mean of the input.
            return torch.mean(x)

        x = torch.randn([5, 5], device=GPU_TYPE)
        out, code = run_and_get_code(f, x)
        FileCheck().check_count("@triton.jit", 1, exactly=True).run(code[0])

    def test_3dred_pw_2d_outer_red(self):
        # [FUNCTION DOC] Checks that a three-level pattern (inner reduction, pointwise add
        # of one, outer reduction) is not loop-reordered: results match eager and the
        # loop reordering counter remains zero, keeping the natural fusion of the first
        # two steps.
        """
        Test a pattern as follows. We have a 3d contiguous tensor [m, n, k] as input.
        1. do reduction on the k dimension and get a [m, n] tensor
        2. do a pointwise operation on this [m, n] tensor (and realize the computation)
        3. do a outer reduction on the output of step 2 on the m dimension.

        Each of these step generate a kernel before fusion.
        Without any loop reorder, kernel 1 and kernel 2 will get fused. And kernel 3 will be separeate.

        But if we reorder the loop for kernel 2, then kernel 2 will get fused with kernel 3.
        And the fused kernel-2-3 can not be fused with kernel 1.

        The older version of LOAF algorithm will do reorder in this case. But there is no real
        benefits. There are even some slight downsides
        1. the original fusion without loop reordering is more natural
        2. fusion kernel 1 with kernel 2 may help precision when the output of kernel 1 is in low precision.
           By fusion kernel 1 and kernel 2, the pointwise operation will operate on fp32 precision thanks
           to fusion.
        """
        M, N, K = 64, 64, 64

        def f(x):
            # [FUNCTION DOC] HELPER: Sums the last dim, adds one and returns it with its sum
            # over dim 0.
            x = x.sum(dim=-1)
            x = x + 1  # can be more complex like sigmoid or other ops
            return x, x.sum(dim=0)

        x = torch.randn(M, N, K, device=GPU_TYPE)
        self.do_acc_test(f, x)
        self.assertEqual(0, metrics.num_loop_reordering)

    def test_qknorm_rope_fusion(self):
        # [FUNCTION DOC] Checks that RMS norm followed by rotary embedding that uses
        # concatenation fuses into one kernel (pointwise cat is used). Results are compared
        # with eager.
        """
        When qknorm (RMS norm) is followed by RoPE which uses cat, the cat
        inputs read from the same buffers. Pointwise cat should be used so
        everything fuses into a single kernel.
        """
        B, H, S, D = 4, 8, 128, 64

        def rms_norm(x, weight):
            # [FUNCTION DOC] HELPER: RMS norm with epsilon 1e-6 and elementwise weight.
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6)
            return x * weight

        def apply_rope(x, freqs_cos, freqs_sin):
            # [FUNCTION DOC] HELPER: Rotary embedding that splits the last dim in half,
            # rotates the halves with cosine and sine tables and concatenates them.
            half = x.shape[-1] // 2
            x1 = x[..., :half]
            x2 = x[..., half:]
            out1 = x1 * freqs_cos - x2 * freqs_sin
            out2 = x2 * freqs_cos + x1 * freqs_sin
            return torch.cat([out1, out2], dim=-1)

        def f(q, norm_weight, freqs_cos, freqs_sin):
            # [FUNCTION DOC] HELPER: Applies RMS norm with the weight then the rotary
            # embedding.
            q = rms_norm(q, norm_weight)
            return apply_rope(q, freqs_cos, freqs_sin)

        q = torch.randn(B, H, S, D)
        norm_weight = torch.randn(D)
        freqs_cos = torch.randn(1, 1, S, D // 2)
        freqs_sin = torch.randn(1, 1, S, D // 2)

        self.do_acc_test(f, q, norm_weight, freqs_cos, freqs_sin)
        self.assertEqual(1, metrics.generated_kernel_count)

    def test_qknorm_interleaved_rope_fusion(self):
        # [FUNCTION DOC] Checks that RMS norm followed by interleaved rotary embedding
        # (stack then flatten) fuses into one kernel and matches eager.
        """
        Interleaved RoPE (stack + flatten) should also fuse with qknorm.
        """
        B, H, S, D = 4, 8, 128, 64

        def rms_norm(x, weight):
            # [FUNCTION DOC] HELPER: RMS norm with epsilon 1e-6 and elementwise weight.
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6)
            return x * weight

        def apply_interleaved_rope(x, cos, sin):
            # [FUNCTION DOC] HELPER: Interleaved rotary embedding: pairs adjacent elements,
            # rotates each pair using cosine and sine tables, then stacks and flattens.
            pairs = x.reshape(*x.shape[:-1], -1, 2)
            a, b = pairs[..., 0], pairs[..., 1]
            out_real = a * cos - b * sin
            out_imag = a * sin + b * cos
            return torch.stack([out_real, out_imag], dim=-1).flatten(-2)

        def f(q, norm_weight, cos, sin):
            # [FUNCTION DOC] HELPER: Applies RMS norm with the weight, then interleaved
            # rotary embedding.
            q = rms_norm(q, norm_weight)
            return apply_interleaved_rope(q, cos, sin)

        q = torch.randn(B, H, S, D)
        norm_weight = torch.randn(D)
        cos = torch.randn(1, 1, S, D // 2)
        sin = torch.randn(1, 1, S, D // 2)

        self.do_acc_test(f, q, norm_weight, cos, sin)
        self.assertEqual(1, metrics.generated_kernel_count)


@inductor_config.patch(
    {
        "triton.unique_kernel_names": True,
        "loop_ordering_after_fusion": True,
        "triton.max_tiles": 3,
        "triton.coalesce_tiling_analysis": True,
    }
)
@instantiate_parametrized_tests
class MemoryCoalescingTest(MockSchedulerTest):
    # [CLASS DOC] Tests of memory coalescing analysis and tiling selection, built on
    # the mock scheduler base class. Config enables unique kernel names, loop
    # ordering after fusion, up to three tiles and coalescing tiling analysis. Has a
    # device attribute naming the GPU; the setUp resets metrics, and helper methods
    # create buffers, scheduler nodes and check memory metrics.
    """Tests for memory coalescing analysis with specific tensor sizes."""

    device = GPU_TYPE
    _exit_stack = None

    def setUp(self):
        # [FUNCTION DOC] Runs the parent setup and resets global Inductor metrics.
        super().setUp()
        metrics.reset()

    def _check_memory_metrics(self, node, expected_tiling):
        # [FUNCTION DOC] HELPER: For a scheduler node, obtains its coalesce analysis,
        # selects a tiling with memory information from the node group extents and asserts
        # that the selection has scores and memory, that the tiling and its scores list
        # exactly the expected tiling dimension names, that coalesced memory is at least
        # the sum of scores, and that coalesced plus uncoalesced memory equals the total
        # coalesced-per-variable plus uncoalesced bytes. Returns the analysis and the
        # selection.
        analysis = node.get_coalesce_analysis()
        self.assertIsNotNone(analysis)
        _, (numel, rnumel) = node.group
        selection = SIMDScheduling.select_tiling_with_memory(
            node.get_nodes(), numel, rnumel, analysis
        )
        self.assertIsNotNone(selection.tiling_scores)
        self.assertIsNotNone(selection.memory)
        self.assertEqual(list(selection.tiling), expected_tiling)
        self.assertEqual(list(selection.tiling_scores), expected_tiling)

        # Coalesced memory is never penalized, so it dominates the tiling score.
        self.assertGreaterEqual(
            selection.memory.coalesced, sum(selection.tiling_scores.values())
        )
        # Every access either coalesces under the selected tiling or does not.
        total_cost = sum(analysis.coalesced_by_var.values()) + sum(
            analysis.uncoalesced_addrs.values()
        )
        self.assertEqual(
            selection.memory.coalesced + selection.memory.uncoalesced, total_cost
        )
        return analysis, selection

    def _create_buffer(self, name, sizes):
        # [FUNCTION DOC] HELPER: Creates a realized computed buffer that doubles a
        # float32 input buffer of the given name and sizes with contiguous strides on the
        # test device, with its layout decided. Returns the computed buffer.
        """Create a buffer with specified sizes"""

        strides = ir.FlexibleLayout.contiguous_strides(sizes)

        box = ir.TensorBox.create(
            ir.Buffer(
                name=name,
                layout=ir.FixedLayout(
                    torch.device(self.device),
                    dtype=torch.float32,
                    size=sizes,
                    stride=strides,
                ),
            )
        )
        box_loader = box.make_loader()

        def inner_fn(index):
            # [FUNCTION DOC] HELPER: Pointwise body: loads the input at the index and
            # doubles it.
            return box_loader(index) * 2

        buf = ir.Pointwise.create(
            device=box.get_device(),
            dtype=box.get_dtype(),
            inner_fn=inner_fn,
            ranges=box.get_size(),
        )
        buf.realize()
        computed_buf = buf.data.data
        computed_buf.decide_layout()

        return computed_buf

    def _create_scheduler_node(self, buf):
        # [FUNCTION DOC] HELPER: Wraps a buffer in a scheduler node and sets its order
        # range to 0 through 100 so it can be fused; returns the node.
        s = SchedulerNode(V.graph.scheduler, buf)
        s.min_order = 0
        s.max_order = 100
        return s

    @parametrize(
        "inps",
        (
            ((128, 384, 196), (768, 64, 196), (128, 6, 64, 196)),
            ((64,), (16, 4), (16, 4)),
            ((5, 6), (3, 10), (30,)),
            ((5, 6, 20), (3, 10, 20), (30, 20)),
        ),
    )
    def test_inferred_splits(self, inps):
        # [FUNCTION DOC] Parametrized with two input sizes and the expected loop sizes.
        # Builds buffers and scheduler nodes for the two input sizes, fuses them and
        # extracts the normalized read/writes of the fused node; asserts the iteration
        # variable ranges equal the expected merged/split sizes, i.e. the inferred splits.
        """
        Test memory coalescing analysis with the specified tensor sizes.
        Using direct SchedulerNode creation with sizes (128, 384, 196) and (768, 64, 196).
        """

        s1, s2, expected_size = inps

        # Create buffers with the specified sizes
        buf1 = self._create_buffer("buffer1", s1)
        buf2 = self._create_buffer("buffer2", s2)

        # Create scheduler nodes
        snode1 = self._create_scheduler_node(buf1)
        snode2 = self._create_scheduler_node(buf2)

        # Create a fused node
        fused_node = torch._inductor.scheduler.FusedSchedulerNode.fuse(snode1, snode2)

        from torch._inductor import tiling_utils

        fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(fused_node)

        var_ranges = fused_norm_read_writes.var_ranges
        self.assertEqual(list(var_ranges.values()), list(expected_size))

    def test_remapped_reads(self):
        # [FUNCTION DOC] Compiles an addition of a contiguous and a transposed 4 by 4
        # tensor with a post-fusion hook that asserts one fused node whose normalized
        # read/writes have two iteration variables and whose read expressions contain the
        # two expected index forms (one for each memory layout).
        from torch._inductor import tiling_utils

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass. Raises AssertionError unless there is
            # exactly one node, checks the normalized reads have two variables and the two
            # read index patterns, then returns the nodes unchanged.
            if len(nodes) != 1:
                raise AssertionError(f"Expected 1 node, got {len(nodes)}")
            fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(
                nodes[0]
            )

            self.assertTrue(len(fused_norm_read_writes.var_ranges) == 2)

            # both reads remapped correctly
            FileCheck().check("4*n0 + n1").run(
                repr(fused_norm_read_writes.reads.keys())
            )
            FileCheck().check("n0 + 4*n1").run(
                repr(fused_norm_read_writes.reads.keys())
            )

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn):

            @torch.compile()
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Compiled function adding its two inputs.
                return x + y

            foo(
                torch.rand([4, 4], device=GPU_TYPE),
                torch.rand([4, 4], device=GPU_TYPE).T,
            )

    def test_remapped_reads_split(self):
        # [FUNCTION DOC] Compiles a function with a contiguous add of a transposed pair
        # plus a realized flattened transposed add on 6 by 6 inputs, with a hook that
        # checks the second fused sub-node has a single loop of 36 with a read expression
        # that splits into a floor division and a modular term, and that the normalized
        # read/writes have two variables of size 6 with the expected remapped read.
        from torch._inductor import tiling_utils

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting a single node, checking the
            # expected loop extent, the read expression in the second sub-node and the
            # remapped normalized read; returns the nodes unchanged.
            self.assertTrue(len(nodes) == 1)
            fused_norm_read_writes = tiling_utils.extract_normalized_read_writes(
                nodes[0]
            )

            inp_node_reads = nodes[0].get_nodes()[1]._body.get_read_exprs()
            node_ranges = nodes[0].get_nodes()[1]._body.var_ranges
            self.assertTrue(len(node_ranges) == 1)
            self.assertTrue(next(iter(node_ranges.values())) == 36)
            var = next(iter(node_ranges.keys()))

            r = FloorDiv(var, 6) + 6 * ModularIndexing(var, 1, 6)
            self.assertTrue(r in inp_node_reads)

            # mapped reads
            self.assertTrue(list(fused_norm_read_writes.var_ranges.values()) == [6, 6])
            n0, n1 = list(fused_norm_read_writes.var_ranges.keys())

            # translation of above is n0 + 6 * n1
            self.assertTrue((n0 + 6 * n1) in fused_norm_read_writes.reads)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn):

            @torch.compile()
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Compiled function adding the inputs, flattening, and
                # adding a realized flattened transposed incremented second input.
                return (
                    x + y
                ).contiguous().flatten() + torch.ops._inductor_test.realize(
                    (y.T + 1).flatten()
                )

            foo(
                torch.rand([6, 6], device=GPU_TYPE),
                torch.rand([6, 6], device=GPU_TYPE).T,
            )

    @parametrize("dynamic", (False, True))
    def test_reduction_pointwise(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes. Compiles a realized add followed
        # by a row sum of 256 by 256 tensors under no-grad with a post-fusion hook, then
        # checks compiled and eager results agree.
        # test one pw var, one red var
        from torch._inductor import tiling_utils

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting one node with one index and
            # one reduction variable, a single write on the index variable, a single read
            # (applied to two buffers) in terms of both variables with stride 256, then
            # returns the nodes.
            self.assertTrue(len(nodes) == 1)
            fused_rw = tiling_utils.extract_normalized_read_writes(nodes[0])

            i_vars, r_vars = fused_rw.index_vars, fused_rw.reduce_vars
            self.assertTrue(len(i_vars) == 1)
            self.assertTrue(len(r_vars) == 1)

            # single write to index var
            self.assertTrue(
                fused_rw.index_vars[0] == next(iter(fused_rw.writes.keys()))
            )

            # the write to the fused intermediary node should be removed
            self.assertTrue(len(fused_rw.writes) == 1)

            # single read
            self.assertTrue(len(fused_rw.reads) == 1)
            # that is applied to two bufs
            self.assertTrue(len(next(iter(fused_rw.reads.values()))) == 2)

            # and the read should be in terms of the index + reduce var,
            # even though node is pointwise
            self.assertTrue(256 * i_vars[0] + r_vars[0] in fused_rw.reads)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            def foo(x, y):
                # [FUNCTION DOC] HELPER: Realizes the sum of the inputs and sums over dim 1.
                out = torch.ops._inductor_test.realize(x + y)
                return out.sum(dim=1)

            x = torch.rand(256, 256, device=GPU_TYPE)
            y = torch.rand(256, 256, device=GPU_TYPE)
            if dynamic:
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(y, 0)

            self.assertEqual(foo(x, y), torch.compile(foo)(x, y))

    @parametrize("dynamic", (False, True))
    def test_reduction_no_pointwise(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes. Compiles the sum of a 1024
        # vector with a post-fusion hook and checks it matches eager.
        # test one pw var, one red var
        from torch._inductor import tiling_utils

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting one node with no index
            # variables and one reduction variable; returns the nodes unchanged.
            self.assertTrue(len(nodes) == 1)
            fused_rw = tiling_utils.extract_normalized_read_writes(nodes[0])

            i_vars, r_vars = fused_rw.index_vars, fused_rw.reduce_vars
            self.assertTrue(len(i_vars) == 0)
            self.assertTrue(len(r_vars) == 1)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the total sum of the input.
                return x.sum()

            x = torch.rand(1024, device=GPU_TYPE)
            if dynamic:
                torch._dynamo.mark_dynamic(x, 0)

            self.assertEqual(foo(x), torch.compile(foo)(x))

    def test_coalescing(self):
        # [FUNCTION DOC] Table-driven check of the helper that finds the coalesced
        # variable of an index expression given variable ranges, covering direct,
        # floor division, modular, none, division by two and nested modular add cases;
        # asserts each result equals the expected variable or None.
        from torch._inductor import tiling_utils

        # Define symbolic variables
        i, j, n, m = sympy.symbols("i j n m", integer=True)

        # Test cases: (expression, var_ranges, expected_result)
        test_cases = [
            # Simple direct case
            (i + j * 5, {i: 10, j: 8}, i),
            # Floor division case
            (i + FloorDiv(j, 2), {i: 4, j: 8}, i),
            # Modular indexing
            (i * 10 + ModularIndexing(j, 1, 3), {i: 5, j: 10}, j),
            # Case with no coalescing variable
            (i * 2 + j * 3, {i: 8, j: 5}, None),
            # Division case
            (i / 2, {i: 10}, None),
            # More complex floor division
            (j + FloorDiv(i, 3), {i: 6, j: 12}, j),
            # Addition inside modular indexing
            (ModularIndexing(i + 3, 1, 6), {i: 8, j: 12}, i),
        ]

        for expr, var_ranges, expected in test_cases:
            # Test the function
            result = tiling_utils.find_coalesced_var(expr, var_ranges)
            self.assertEqual(result, expected)

    @parametrize("downcast_transposed_v", (False, True))
    @parametrize("dynamic", (False, True))
    def test_tiled_coalesce_analysis(self, downcast_transposed_v, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes and by whether the transposed
        # operand is downcast (here float64 converted to the other dtype). Compiles an add
        # under no-grad with a hook that compares the coalesced bytes of the two index
        # variables: the contiguous variable is three times the transposed one (two
        # reads and a write weighted two) without the dtype change and 1.5 times with it.
        # test one pw var, one red var
        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass that asserts one node and checks the
            # ratio of coalesced reads of the contiguous variable to the transposed one
            # according to the downcast flag; returns the nodes.
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = nodes[0].get_coalesce_analysis()

            i_vars = coalesce_analysis.norm_read_writes.index_vars

            # because output is contiguous, second dimension should
            # coalesce twice as many bytes as first dimension
            # if not downcasted
            # if downcasted, should be equal, bc larger dtype size
            # we also weight writes x 2
            cont_reads = coalesce_analysis.coalesced_by_var[i_vars[1]]
            t_reads = coalesce_analysis.coalesced_by_var[i_vars[0]]

            if not downcast_transposed_v:
                self.assertEqual(cont_reads, t_reads * 3)
            else:
                self.assertEqual(cont_reads, t_reads * 3 / 2)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            def foo(x, y):
                # [FUNCTION DOC] HELPER: Adds the first input to the second converted to the
                # first dtype.
                return x + y.to(x.dtype)

            x = torch.rand(256, 256, device=GPU_TYPE)
            y_dtype = torch.float if not downcast_transposed_v else torch.float64
            y = torch.rand(256, 256, device=GPU_TYPE, dtype=y_dtype).T
            if dynamic:
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(y, 0)

            self.assertEqual(foo(x, y), torch.compile(foo)(x, y))

    def test_multiple_reduction_dims(self):
        # [FUNCTION DOC] Checks the coalescing analysis of a variance and mean over two
        # reduction dims of a 4D tensor: two reduce variables, the last reduce variable
        # coalesces all input bytes, and the index variable coalesces output bytes
        # weighted twice for writes. Compiled result must equal eager.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns variance and mean over dims 1 and 3 as a tuple.
            return (*torch.var_mean(x, [1, 3]),)

        inp = torch.randn(1, 2, 4, 8, device=GPU_TYPE)
        out_eager = foo(inp)

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass that checks reduction variable count and
            # the coalesced byte counts for the last reduction variable and the index
            # variable; returns the nodes.
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = nodes[0].get_coalesce_analysis()
            red_vars = coalesce_analysis.norm_read_writes.reduce_vars

            self.assertTrue(len(red_vars) == 2)

            n1, n2 = red_vars
            n0 = coalesce_analysis.norm_read_writes.index_vars[0]

            self.assertEqual(
                coalesce_analysis.coalesced_by_var[n2], inp.numel() * inp.itemsize
            )
            WRITE_WEIGHTING = 2

            out_bytes = sum(a.numel() * a.itemsize * WRITE_WEIGHTING for a in out_eager)
            self.assertEqual(coalesce_analysis.coalesced_by_var[n0], out_bytes)

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():
            out = torch.compile(foo)(inp)
            self.assertEqual(out, out_eager)

    def test_reduction_memory_costs_follow_selected_tiling(self):
        # [FUNCTION DOC] Checks that memory costs follow the selected tiling for a
        # reduction over two dims of a sum of a regular and a permuted input. A hook checks
        # the tiling is x and r0_, scores and coalesced/uncoalesced memory relate to the
        # per-variable coalesced bytes (x score penalized, first reduction variable counted
        # uncoalesced because it was not chosen). Result must equal eager.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Sums the added inputs over dims 1 and 3.
            return (x + y).sum((1, 3))

        def check_metrics(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting one node, running the memory
            # metric check with tiling x and r0_, and comparing scores and memory against the
            # analysis values; returns the nodes.
            self.assertEqual(len(nodes), 1)
            node = nodes[0]
            analysis, selection = self._check_memory_metrics(node, ["x", "r0_"])
            index_vars = analysis.norm_read_writes.index_vars
            reduction_vars = analysis.norm_read_writes.reduce_vars
            self.assertEqual(len(index_vars), 1)
            self.assertEqual(len(reduction_vars), 2)
            self.assertGreater(analysis.coalesced_by_var[reduction_vars[0]], 0)
            self.assertGreater(analysis.coalesced_by_var[reduction_vars[1]], 0)

            self.assertEqual(
                selection.tiling_scores["r0_"],
                analysis.coalesced_by_var[reduction_vars[-1]],
            )
            # The x score is penalized for leaving a small block; the memory is not.
            self.assertLess(
                selection.tiling_scores["x"],
                analysis.coalesced_by_var[index_vars[-1]],
            )
            expected_coalesced = (
                analysis.coalesced_by_var[index_vars[-1]]
                + analysis.coalesced_by_var[reduction_vars[-1]]
            )
            self.assertEqual(selection.memory.coalesced, expected_coalesced)
            # reduction_vars[0] could coalesce, but the selected tiling did not
            # pick it, so its memory counts as uncoalesced.
            self.assertEqual(
                selection.memory.uncoalesced,
                analysis.coalesced_by_var[reduction_vars[0]],
            )
            return nodes

        x = torch.randn(1, 2, 4, 8, device=GPU_TYPE)
        y = torch.randn(1, 8, 4, 2, device=GPU_TYPE).permute(0, 3, 2, 1)
        with (
            torch._inductor.config.patch(_post_fusion_custom_pass=check_metrics),
            torch.no_grad(),
        ):
            self.assertEqual(torch.compile(foo)(x, y), foo(x, y))

    def test_solve_for_zero(self):
        # [FUNCTION DOC] Table-driven check of the helper that solves an index expression
        # for the value that makes it zero: linear forms give the root, constants and
        # floor division and nonlinear forms give None, modular forms give their modulus.
        from torch._inductor import tiling_utils

        x, y = sympy.symbols("x y", integer=True)
        # Test cases: (expression, expected_result)
        test_cases = [
            # Simple linear expressions
            (x + 5, (-5)),
            (2 * x - 10, (5)),
            # Constant expressions (should return None)
            (sympy.Integer(7), None),
            (sympy.Integer(0), None),
            # FloorDiv cases (should return None per function)
            (FloorDiv(x, 2), None),
            (FloorDiv(x, 2) + 5, None),
            # ModularIndexing cases
            (ModularIndexing(x, 1, 5), (5)),
            (ModularIndexing(x, 1, 3), (3)),
            # Expressions with no constant solution
            (x**2 + 1, None),  # No real solution
        ]
        for expr, expected in test_cases:
            result = tiling_utils.solve_for_zero(expr)
            self.assertEqual(result, expected)

    def test_solve_for_tiling(self):
        # [FUNCTION DOC] Table-driven check of the helper that infers a tiling factor
        # from an index expression, including division, floor division, nested modular
        # forms and linear, nonlinear and mixed cases where None is expected.
        from torch._inductor import tiling_utils

        x = sympy.Symbol("x", integer=True)

        test_cases = [
            # Simple linear cases that coalesce
            (3 * x, None),
            # # # # Expression with no free symbols
            # (sympy.Integer(5), None),
            (x / 3, 3),
            (FloorDiv(x * 2, 6), 3),
            # # ModularIndexing expressions
            (ModularIndexing(FloorDiv(x, 4), 1, 64), 4),
            (x + ModularIndexing(x, 1, 5), None),
            (x**2, None),  # Non-linear, diff is not constant
            (4096 * (ModularIndexing(32 * x, 1, 2048)) + FloorDiv(x, 64), 64),
            (4096 * (ModularIndexing(x, 1, 2048)) + FloorDiv(x, 2048), 2048),
        ]

        for expr, expected in test_cases:
            result = tiling_utils.solve_for_tiling(expr)
            self.assertEqual(result, expected)

    def test_solve_for_tiling_nested_modularindexing(self):
        # [FUNCTION DOC] Checks that the tiling solver handles nested modular indexing
        # without raising: three nesting depths all give the same solution (252253), and a
        # mixed nested floor division and modular expression gives None.
        from torch._inductor import tiling_utils

        n0 = sympy.Symbol("n0", integer=True, nonnegative=True)
        inner = ModularIndexing(n0 - 252252, 1, 50)
        for expr in (
            inner,
            ModularIndexing(inner, 1, 50),
            ModularIndexing(ModularIndexing(inner, 1, 50), 1, 50),
        ):
            # must not raise, and nested forms solve identically to the flat form
            self.assertEqual(tiling_utils.solve_for_tiling(expr), sympy.Integer(252253))

        # mixed nesting has no tiling solution, but must not raise either
        mixed = FloorDiv(
            ModularIndexing(5 * FloorDiv(ModularIndexing(n0 + 8, 1, 6), 5), 1, 36), 8
        )
        self.assertEqual(tiling_utils.solve_for_tiling(mixed), None)

    @parametrize("dynamic", (False, True))
    def test_induced_fused_tiling(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes. Compiles a clone of a permuted
        # bfloat16 tensor, view as rows of 32 and amax with a hook checking the suggested
        # split tiling factor is 64 and, for static shapes, that the tiling is y, x and
        # r0_ with the y score equal to the suggested split score. Also checks the output
        # equals eager and the generated code mentions both YBLOCK and XBLOCK.
        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting one node and checking the
            # suggested split and, when not dynamic, the memory metrics; returns the nodes.
            self.assertTrue(len(nodes) == 1)

            node = nodes[0]
            coalesce_analysis = node.get_coalesce_analysis()
            self.assertEqual(coalesce_analysis.suggested_split.tiling_factor, 64)
            if not dynamic:
                expected_tiling = ["y", "x", "r0_"]
                coalesce_analysis, memory_metrics = self._check_memory_metrics(
                    node, expected_tiling
                )
                suggested_score = coalesce_analysis.suggested_split.score
                self.assertEqual(memory_metrics.tiling_scores["y"], suggested_score)
            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():

            def forward(permute):
                # [FUNCTION DOC] HELPER: Clones with contiguous format, views as rows of 32 and
                # returns the amax over dim 1.
                clone = torch.ops.aten.clone.default(
                    permute, memory_format=torch.contiguous_format
                )
                view_2 = torch.ops.aten.view.default(clone, [-1, 32])
                amax_1 = torch.ops.aten.amax.default(view_2, [1])
                return amax_1

            XDIM = 2048
            YDIM = 4096

            arg0_1 = torch.randn([XDIM, YDIM], device=GPU_TYPE, dtype=torch.bfloat16)
            if dynamic:
                torch._dynamo.mark_dynamic(arg0_1, 0)

            permute = torch.ops.aten.permute.default(arg0_1, [1, 0])

            out, code = run_and_get_code(torch.compile(forward), (permute))

            self.assertEqual(out, forward(permute))
            # Assert that we captured code, before checking it.
            self.assertTrue(code)
            FileCheck().check("YBLOCK").check("XBLOCK").run(code[0])


layouts = ("cont", "NHWC", "T")


@inductor_config.patch(
    {
        "triton.unique_kernel_names": True,
        "loop_ordering_after_fusion": True,
        "triton.coalesce_tiling_analysis": True,
    }
)
@instantiate_parametrized_tests
class TestTiling(TestCase):
    # [CLASS DOC] Tests that generated Triton kernels choose 1D, 2D or 3D tiling from
    # memory coalescing analysis for various memory layouts (contiguous, channels
    # last, transposed). Config enables unique kernel names, loop ordering after
    # fusion and coalescing tiling analysis. Checks the code for dimension numel
    # names. Has a helper creating the test tensors.
    def T(self, layout: str):
        # [FUNCTION DOC] HELPER: Returns a GPU float tensor with a leading dim of 1 and
        # logical size 128 by 256 by 512 (or the channels-last 4D version) laid out in
        # the memory layout named by the argument: contiguous, transposed (made
        # contiguous in the transposed order) or channels last. Raises AssertionError for
        # other names.
        SIZE_A = 128
        SIZE_B = 256
        SIZE_C = 512

        if layout == "cont":
            return torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE).unsqueeze(0)
        elif layout == "T":
            return (
                torch.rand(SIZE_A, SIZE_B, SIZE_C, device=GPU_TYPE)
                .transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .unsqueeze(0)
            )
        else:
            if layout != "NHWC":
                raise AssertionError(f"Unexpected layout: {layout}")
            return torch.rand([1, SIZE_A, SIZE_B, SIZE_C], device=GPU_TYPE).to(
                memory_format=torch.channels_last
            )

    @parametrize("a", layouts)
    @parametrize("b", layouts)
    @parametrize("dynamic", (False, True))
    def test_pointwise(self, a, b, dynamic):
        # [FUNCTION DOC] Parametrized by two layouts and dynamic shapes. Compiles an add
        # of two tensors in those layouts. If layouts differ the generated code must have a
        # y dimension numel (2D tiling) and otherwise not; the output matches eager.
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
            return x + y

        x, y = self.T(a), self.T(b)
        res, code = run_and_get_code(torch.compile(foo, dynamic=dynamic), x, y)

        if a != b:
            FileCheck().check("ynumel").run(code[0])
        else:
            FileCheck().check_not("ynumel").run(code[0])

        self.assertEqual(res, foo(x, y))

    def test_tiled_reduction(self):
        # [FUNCTION DOC] Compiles a product reduction over the last dim of two permuted
        # 512 cubed inputs and checks the generated code has x numel 512, y numel 512 and
        # a reduction numel in any order; output must match eager within tolerances.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Sums the elementwise product over the last dim.
            return (a * b).sum(dim=-1)

        N = 512
        inps = (
            torch.randn(N, N, N, device=GPU_TYPE).permute(2, 1, 0),
            torch.randn(N, N, N, device=GPU_TYPE).permute(1, 2, 0),
        )
        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)

        FileCheck().check_dag("xnumel = 512").check_dag("ynumel = 512").check_dag(
            "rnumel"
        ).run(code[0])
        self.assertEqual(out, f(*inps), atol=0.001, rtol=0.04)

    @parametrize("dynamic", (False, True))
    def test_3d_pointwise(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes. Compiles the sum of three tensors
        # with different layouts and checks the code uses z, y and x numel, and that
        # results match eager.
        inps = (self.T("cont"), self.T("T"), self.T("NHWC"))

        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of the three inputs.
            return x + y + z

        f_c = torch.compile(f, dynamic=dynamic)
        out, code = run_and_get_code(f_c, *inps)

        FileCheck().check_dag("znumel").check_dag("ynumel").check_dag("xnumel").run(
            code[0]
        )
        self.assertEqual(out, f(*inps))

    def test_cat(self):
        # [FUNCTION DOC] Compiles a concatenation of a contiguous and a transposed tensor
        # plus one, and checks that tiling uses y and x numel and matches eager; relates to
        # unwrapping an identity indexing.
        # test unwrapping Identity

        def f(x, y):
            # [FUNCTION DOC] HELPER: Concatenates the inputs along dim 0 and adds one.
            return torch.cat((x, y)) + 1

        x = self.T("cont")
        y = self.T("T")

        inps = (x, y)

        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)
        FileCheck().check_dag("ynumel").check_dag("xnumel").run(code[0])
        self.assertEqual(out, f(*inps))

    def test_penalized_small_dim(self):
        # [FUNCTION DOC] Adds a 2000 by 1 tensor to a transposed 1 by 4 style tensor and
        # checks that the generated code does not tile in y (only x), since small
        # dimension tiling would not improve coalescing; output matches eager.
        x = torch.rand([2000, 1], device=GPU_TYPE)
        y = torch.rand([4, 1], device=GPU_TYPE).T

        # don't tile when it doesn't affect total coalesced mem accesses much
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
            return x + y

        inps = (x, y)

        f_c = torch.compile(f)
        out, code = run_and_get_code(f_c, *inps)
        FileCheck().check_not("ynumel").check_dag("xnumel").run(code[0])
        self.assertEqual(out, f(*inps))

    def test_mutation_deps(self):
        # [FUNCTION DOC] Compiles an in-place add under no-grad with a hook checking one
        # node whose normalized reads and writes each have a single entry, with the read
        # being the argument buffer and the write a new buffer name.
        def f(x):
            # [FUNCTION DOC] HELPER: In-place increment of the input by one.
            return x.add_(1)

        x = self.T("cont")

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass asserting one node, a coalesce analysis
            # exists (raises AssertionError otherwise), and that normalized reads and writes
            # have the expected buffer names; returns the nodes.
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = nodes[0].get_coalesce_analysis()
            if coalesce_analysis is None:
                raise AssertionError

            reads = coalesce_analysis.norm_read_writes.reads
            writes = coalesce_analysis.norm_read_writes.writes

            self.assertTrue(len(reads) == 1 and len(writes) == 1)
            self.assertEqual(
                list(coalesce_analysis.norm_read_writes.reads.values()),
                [OrderedSet(("arg0_1",))],
            )
            self.assertEqual(
                list(coalesce_analysis.norm_read_writes.writes.values()),
                [OrderedSet(("buf1",))],
            )

            return nodes

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn), torch.no_grad():
            torch.compile(f)(x)

    def test_find_broadcast_var(self):
        # [FUNCTION DOC] Checks the helper that detects a broadcast variable in an index
        # expression: a floor division expression returns its variable and a purely
        # linear expression returns None.
        """Test broadcast variable detection for tiling improvements."""
        from torch._inductor import tiling_utils

        i, j, k = sympy.symbols("i j k", integer=True)

        # Test broadcast pattern detection: FloorDiv creates broadcast
        result = tiling_utils.find_broadcast_var(
            FloorDiv(i, 10), {i: 100, j: 50, k: 20}
        )
        self.assertEqual(result, i)

        # Test non-broadcast: linear access pattern
        result = tiling_utils.find_broadcast_var(i + j * 10, {i: 10, j: 8, k: 20})
        self.assertEqual(result, None)

    def test_indirect_access_not_coalesced(self):
        # [FUNCTION DOC] Checks that indirect indexing (embedding lookup) is treated as
        # uncoalesced. A hook asserts one node, exactly one uncoalesced access whose
        # expression contains an indirect symbol, at least one coalesced variable and no
        # indirect symbol among coalesced variables. The compiled output must match eager.
        """Test that memory accesses with indirect indexing are treated as uncoalesced.

        When we have an access pattern like weights[indices[i]], the weight load
        uses an indirect index and should NOT be counted as coalesced since the
        access pattern is effectively random.

        For w[idx] where idx is (4096,) and w is (1024, 256):
        - Read of idx: coalesced (simple linear access)
        - Read of w: uncoalesced (uses indirect index)
        - Write to output: coalesced (contiguous)
        """
        from torch.utils._sympy.symbol import symbol_is_type, SymT

        def fn(nodes):
            # [FUNCTION DOC] HELPER: Post-fusion pass performing the checks on the coalesce
            # analysis described in the test; returns the nodes.
            self.assertTrue(len(nodes) == 1)

            coalesce_analysis = nodes[0].get_coalesce_analysis()
            self.assertIsNotNone(coalesce_analysis)

            # Should have exactly 1 uncoalesced access (the indirect weight read)
            self.assertEqual(
                len(coalesce_analysis.uncoalesced_addrs),
                1,
                lambda msg: f"{msg}\nExpected 1 uncoalesced access, got {len(coalesce_analysis.uncoalesced_addrs)}",
            )

            # The uncoalesced access should have an INDIRECT symbol
            for expr in coalesce_analysis.uncoalesced_addrs:
                has_indirect = any(
                    symbol_is_type(s, SymT.INDIRECT) for s in expr.free_symbols
                )
                self.assertTrue(
                    has_indirect,
                    lambda msg: f"{msg}\nExpected uncoalesced expr {expr} to have INDIRECT symbol",
                )

            # Should have coalesced accesses (idx read + output write)
            self.assertGreater(
                len(coalesce_analysis.coalesced_by_var),
                0,
                "Expected at least one coalesced variable",
            )

            # Verify no INDIRECT symbols ended up as coalesced vars
            for var in coalesce_analysis.coalesced_by_var:
                self.assertFalse(
                    symbol_is_type(var, SymT.INDIRECT),
                    lambda msg: f"{msg}\nINDIRECT symbol {var} should not be in coalesced_by_var",
                )

            return nodes

        V, D = 1024, 256
        indices = torch.randint(0, V, (4096,), device=GPU_TYPE)
        weights = torch.randn(V, D, device=GPU_TYPE)

        def embedding_1d(idx, w):
            # [FUNCTION DOC] HELPER: Looks up rows of the weight table using the index vector.
            return w[idx]

        with torch._inductor.config.patch(_post_fusion_custom_pass=fn):
            out = torch.compile(embedding_1d)(indices, weights)

        # Verify correctness
        expected = embedding_1d(indices, weights)
        self.assertEqual(out, expected)

    @parametrize("dynamic", (False, True))
    def test_scatter_broadcast_no_tiling(self, dynamic):
        # [FUNCTION DOC] Parametrized by dynamic shapes. Compiles a gather followed by a
        # scatter add with broadcast index and checks the generated code has no y numel (no
        # 2D tiling) and that the result equals eager.
        """Scatter with broadcast loads should not trigger 2D tiling."""
        num_nodes = 4096
        num_edges = 16384
        feat_dim = 64

        src = torch.randint(0, num_nodes, (num_edges,), device=GPU_TYPE)
        features = torch.randn(num_nodes, feat_dim, device=GPU_TYPE)

        if dynamic:
            torch._dynamo.mark_dynamic(src, 0)
            torch._dynamo.mark_dynamic(features, 0)

        def f(src, features):
            # [FUNCTION DOC] HELPER: Gathers feature rows by index and scatter-adds them into
            # a zero table using an expanded index.
            gathered = features[src]
            out = torch.zeros(num_nodes, feat_dim, device=GPU_TYPE)
            return out.scatter_add_(0, src.unsqueeze(1).expand_as(gathered), gathered)

        out, code = run_and_get_code(torch.compile(f), src, features)
        FileCheck().check_not("ynumel").run(code[0])
        self.assertEqual(out, f(src, features))

    def test_cont_plus_transposed_picks_2d(self):
        # [FUNCTION DOC] Checks that adding a contiguous and a transposed 256 by 256 tensor
        # still selects 2D tiling (y numel present) and matches eager.
        """Contiguous + transposed addition should still pick 2D tiling."""
        x = torch.randn(256, 256, device=GPU_TYPE)
        y = torch.randn(256, 256, device=GPU_TYPE).T

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
            return x + y

        out, code = run_and_get_code(torch.compile(f), x, y)
        FileCheck().check("ynumel").run(code[0])
        self.assertEqual(out, f(x, y))

    def test_mixed_broadcast_transpose_picks_2d(self):
        # [FUNCTION DOC] Checks that adding a contiguous tensor, a transposed one and a
        # broadcast vector still selects 2D tiling due to the real transposed access, and
        # the output matches eager.
        """Mixed broadcast and transposed access: 2D chosen for real coalescing only."""
        x = torch.randn(256, 256, device=GPU_TYPE)
        y = torch.randn(256, 256, device=GPU_TYPE).T
        z = torch.randn(256, device=GPU_TYPE)

        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Adds the three inputs, unsqueezing the last on dim 1.
            return x + y + z.unsqueeze(1)

        out, code = run_and_get_code(torch.compile(f), x, y, z)
        # 2D tiling should be selected because of real transposed coalescing
        # from y, even though z's broadcast score is filtered as already-
        # coalesced-in-1D. Both y and z contribute to the same variable (n0),
        # so this tests per-expression filtering within a single variable.
        FileCheck().check("ynumel").run(code[0])
        self.assertEqual(out, f(x, y, z))


class TestSplitIterationRanges(MockSchedulerTest):
    # [CLASS DOC] Unit tests for the SIMD kernel static method that splits iteration
    # ranges into groups, using the mock scheduler base class. Covers exact matches,
    # splits, factorable symbolic cases, size-one dims and the CantSplit error cases.
    """Unit tests for SIMDKernel._split_iteration_ranges."""

    def test_exact_match(self):
        # [FUNCTION DOC] Checks that lengths exactly equal to the group sizes yield two
        # groups with one range each.
        """Groups exactly match lengths — no splitting needed."""
        from torch._inductor.codegen.simd import SIMDKernel

        new_ranges, getters = SIMDKernel._split_iteration_ranges(
            [sympy.Integer(4), sympy.Integer(8)],
            [[sympy.Integer(4)], [sympy.Integer(8)]],
        )
        self.assertEqual(len(new_ranges), 2)
        self.assertEqual(len(new_ranges[0]), 1)
        self.assertEqual(len(new_ranges[1]), 1)

    def test_two_way_split(self):
        # [FUNCTION DOC] Checks that a single range of 32 is split across two groups of 4
        # and 8, returning two groups.
        """A single large dimension splits across two groups."""
        from torch._inductor.codegen.simd import SIMDKernel

        new_ranges, getters = SIMDKernel._split_iteration_ranges(
            [sympy.Integer(4), sympy.Integer(8)],
            [[sympy.Integer(32)], []],
        )
        # 32 should split into 4 * 8 across the two groups
        self.assertEqual(len(new_ranges), 2)

    def test_two_way_split_with_factorable_add_floordiv(self):
        # [FUNCTION DOC] Checks splitting with symbolic sizes where the numerator is a
        # multiple of the denominator via a floor division factor: the new ranges equal the
        # denominator and 128, and the first getter maps the two new indices to 128 times
        # the first plus the second.
        from torch._inductor.codegen.simd import SIMDKernel

        s52, s97 = sympy.symbols("s52 s97", integer=True, positive=True)
        k = FloorDiv(s97, s52)
        den = s52 * k + k
        num = 128 * s52 * k + 128 * k

        new_ranges, getters = SIMDKernel._split_iteration_ranges(
            [den, sympy.Integer(128)],
            [[num], []],
        )

        self.assertEqual(new_ranges, [[den], [sympy.Integer(128)]])
        i0, i1 = sympy.symbols("i0 i1", integer=True)
        self.assertEqual(getters[0][0]([i0, i1]), 128 * i0 + i1)

    def test_groups_exhausted_raises_cant_split(self):
        # [FUNCTION DOC] Checks that CantSplit is raised when all groups are consumed but
        # sizes remain (sizes 1, 2, 2 against three ranges of 2).
        """When all groups are consumed but sizes remain, CantSplit is raised."""
        from torch._inductor.codegen.simd import CantSplit, SIMDKernel

        # groups=[1, 2, 2] can only absorb 2 sizes of 2 (consuming groups 1 and 2),
        # leaving the third size=2 with no group to map to.
        with self.assertRaises(CantSplit):
            SIMDKernel._split_iteration_ranges(
                [sympy.Integer(1), sympy.Integer(2), sympy.Integer(2)],
                [[], [sympy.Integer(2), sympy.Integer(2), sympy.Integer(2)]],
            )

    def test_single_group_multiple_sizes(self):
        # [FUNCTION DOC] Checks that three sizes of 2 fit in one group of 8: one group with
        # three ranges.
        """Multiple sizes fitting within a single group."""
        from torch._inductor.codegen.simd import SIMDKernel

        # groups=[8], lengths=[[2, 2, 2], []] — all 3 sizes fit in group 0
        new_ranges, getters = SIMDKernel._split_iteration_ranges(
            [sympy.Integer(8)],
            [[sympy.Integer(2), sympy.Integer(2), sympy.Integer(2)], []],
        )
        self.assertEqual(len(new_ranges), 1)
        self.assertEqual(len(new_ranges[0]), 3)

    def test_size_one_skipped(self):
        # [FUNCTION DOC] Checks that a dimension of size 1 consumes no range: the first
        # getter returns zero for any input and the group has two getters.
        """Dimensions of size 1 produce a zero-constant getter."""
        from torch._inductor.codegen.simd import SIMDKernel

        new_ranges, getters = SIMDKernel._split_iteration_ranges(
            [sympy.Integer(4)],
            [[sympy.Integer(1), sympy.Integer(4)], []],
        )
        # Size-1 dim should not consume any range
        self.assertEqual(len(getters[0]), 2)
        # The first getter should return 0 for any input
        self.assertEqual(getters[0][0]([sympy.Integer(99)]), sympy.Integer(0))

    def test_leftover_extent_raises_cant_split(self):
        # [FUNCTION DOC] Checks that CantSplit (not an AssertionError) is raised when the
        # lengths consume cleanly but leave a non-unit leftover group extent, which
        # happens for an epilogue whose iteration domain is a strict sub-multiple of a
        # template tiling.
        """Lengths consume cleanly but leave a non-unit group extent.

        Each size divides cleanly as it is consumed, so none of the add_range
        divisibility checks trip, but the groups are larger than the lengths and
        a non-unit extent remains at the end. This is the case a fused epilogue
        whose iteration domain is a strict sub-multiple of a template's tiling
        hits (e.g. [s, N] into [K*s, N]); it must surface as CantSplit so callers
        skip the fusion rather than crashing the compile with an AssertionError.
        """
        from torch._inductor.codegen.simd import CantSplit, SIMDKernel

        # groups=[2, 2], lengths=[[2], []]: size 2 maps onto group 0, leaving
        # group 1 (extent 2) unconsumed -> remaining=[1, 2], not all ones.
        with self.assertRaises(CantSplit):
            SIMDKernel._split_iteration_ranges(
                [sympy.Integer(2), sympy.Integer(2)],
                [[sympy.Integer(2)], []],
            )


class TestIndexInversion(TestCase):
    # [CLASS DOC] Tests for inverting index expressions made of floor division and
    # modular indexing terms with the inverse formula generator, and for the scheduler
    # cache of flattened read inverses. The class setup installs a mock scheduler graph
    # handler like the mock scheduler base class, closed in the class teardown.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: runs the parent setup, builds a graph lowering from a
        # trivial traced function with the mock scheduler class and enters the virtualized
        # graph handler context on a class-level exit stack.
        super().setUpClass()

        gm = torch.fx.symbolic_trace(lambda: 0)
        graph = GraphLowering(gm)
        graph.scheduler = MockScheduler
        cls._exit_stack = contextlib.ExitStack()
        cls._exit_stack.enter_context(V.set_graph_handler(graph))

    def _check_expr(self, expr, reconstruction, val_range):
        # [FUNCTION DOC] HELPER: Verifies that a reconstruction expression inverts the
        # given one over the integers 0 up to the given range. Requires exactly one free
        # symbol (raises AssertionError otherwise). Replaces floor division and modular
        # indexing with integer operations, evaluates both with numpy, applies the inverse
        # to the forward values and asserts the original values are recovered.
        import numpy as np
        from sympy import lambdify

        if len(expr.free_symbols) != 1:
            raise AssertionError(
                f"Expected 1 free symbol, got {len(expr.free_symbols)}"
            )
        p0 = next(iter(expr.free_symbols))

        def floordiv_replacement(a, b):
            # [FUNCTION DOC] HELPER: Integer floor division of a by b.
            """Replace FloorDiv(a, b) with a // b"""
            return a // b

        def modularindexing_replacement(x, base, divisor):
            # [FUNCTION DOC] HELPER: Integer floor division of x by base, modulo divisor.
            """Replace ModularIndexing(x, base, divisor) with (x // base) % divisor"""
            return (x // base) % divisor

        # Replace custom functions with sympy equivalents
        expr_numpy_ready = expr.replace(FloorDiv, floordiv_replacement).replace(
            ModularIndexing, modularindexing_replacement
        )
        reconstruction_numpy_ready = reconstruction.replace(
            FloorDiv, floordiv_replacement
        ).replace(ModularIndexing, modularindexing_replacement)

        # Now lambdify with standard numpy
        forward_func = lambdify(p0, expr_numpy_ready, modules="numpy")
        inverse_func = lambdify(p0, reconstruction_numpy_ready, modules="numpy")

        test_values = np.arange(0, val_range, dtype=np.int64)
        forward_values = forward_func(test_values).astype(np.int64)
        recovered_values = inverse_func(forward_values).astype(np.int64)
        torch.testing.assert_close(test_values, recovered_values)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: runs the parent teardown then closes the exit
        # stack, removing the graph handler.
        super().tearDownClass()
        cls._exit_stack.close()

    def test_original_complex_expression(self):
        # [FUNCTION DOC] Checks inversion of the original motivating expression made of
        # several floor division and modular terms: the inverse exists and reconstructs
        # the values over a range of 2097152.
        """Test the original motivating complex expression."""
        p0 = sympy.Symbol("p0")
        expr = (
            32768 * FloorDiv(p0, 32768)
            + 8192 * FloorDiv(ModularIndexing(p0, 1, 16), 4)
            + ModularIndexing(p0, 1, 4)
            + 256 * ModularIndexing(p0, 16, 32)
            + 4 * ModularIndexing(p0, 512, 64)
        )

        reconstruction = generate_inverse_formula(expr, p0)
        self.assertIsNotNone(reconstruction)
        self._check_expr(expr, reconstruction, 2097152)

    def test_inversion_cases(self):
        # [FUNCTION DOC] Table-driven check of inversion: for each expression, an inverse
        # must be found (and verified numerically over the given range) or not found as
        # flagged. Cases include base-10 and base-2 decompositions, sorted terms, missing
        # chunks, wrong coefficient ratios, overlapping ranges, nonlinear and
        # trigonometric forms. Afterwards checks bounded inversion with an extra range
        # argument (accepted only for a suitable bound, with a specific expected formula),
        # that nested reconstruction gives the same result, that overlapping floors and
        # blockwise expressions are rejected or accepted depending on the bound.
        """Test various expressions for correct inversion behavior."""
        p = sympy.Symbol("p")

        cases = [
            # (expression, should_be_invertible, test_range)
            # Simple 2-term base-10 style: 10 = 1 × 10 ✓
            (10 * ModularIndexing(p, 10, 10) + ModularIndexing(p, 1, 10), True, 100),
            # Simple 2-term base-2 style: 2 = 1 × 2 ✓
            (2 * ModularIndexing(p, 2, 2) + ModularIndexing(p, 1, 2), True, 4),
            # 3-term decimal: 100 = 10×10, 10 = 1×10 ✓
            (
                100 * FloorDiv(p, 100)
                + 10 * FloorDiv(ModularIndexing(p, 1, 100), 10)
                + ModularIndexing(p, 1, 10),
                True,
                1000,
            ),
            (4 * p, False, 64),  # expr and inverse not bijections
            # when sorted, invertible
            (ModularIndexing(p, 1, 10) + 10 * ModularIndexing(p, 10, 10), True, None),
            (
                4 * FloorDiv(p, 512)
                + ModularIndexing(p, 1, 4)
                + 4096 * ModularIndexing(p, 4, 4)
                + 128 * ModularIndexing(p, 16, 32),
                False,
                None,
            ),
            # Missing the source chunk ModularIndexing(p, 4, 4).
            (
                4 * FloorDiv(p, 512)
                + ModularIndexing(p, 1, 4)
                + 128 * ModularIndexing(p, 16, 32),
                False,
                None,
            ),
            (
                4 * FloorDiv(p, 4)
                + 2 * FloorDiv(ModularIndexing(p, 1, 5), 2)
                + ModularIndexing(p, 1, 2),
                False,
                None,
            ),
            (
                4 * FloorDiv(p, 4) + ModularIndexing(p, 1, 4) + 100,
                False,
                None,
            ),
            # Wrong coefficient ratios: 4 ≠ 1×2
            (4 * ModularIndexing(p, 1, 8) + ModularIndexing(p, 8, 2), False, None),
            (
                100 * FloorDiv(p, 100) + 7 * ModularIndexing(p, 1, 100),
                False,
                None,
            ),  # Wrong ratios
            (FloorDiv(p, 100) + FloorDiv(p, 10) + p, False, None),  # Overlapping ranges
            (p**2 + 10 * p + 1, False, None),  # Quadratic
            (sympy.sin(p) + sympy.cos(p), False, None),  # Trigonometric
        ]

        for expr, should_invert, test_range in cases:
            reconstruction = generate_inverse_formula(expr, p)

            if should_invert:
                self.assertIsNotNone(
                    reconstruction, lambda msg: f"{msg}\nExpected invertible: {expr}"
                )
                # Test correctness on sample values
                self._check_expr(expr, reconstruction, test_range)
            else:
                self.assertIsNone(
                    reconstruction,
                    lambda msg: f"{msg}\nExpected non-invertible: {expr}",
                )

        bounded_expr = (
            4 * FloorDiv(p, 512)
            + ModularIndexing(p, 1, 4)
            + 4096 * ModularIndexing(p, 4, 4)
            + 128 * ModularIndexing(p, 16, 32)
        )
        self.assertIsNone(generate_inverse_formula(bounded_expr, p))
        self.assertEqual(
            generate_inverse_formula(bounded_expr, p, 16384),
            4 * FloorDiv(p, 4096)
            + ModularIndexing(p, 1, 4)
            + 512 * ModularIndexing(p, 4, 32)
            + 16 * ModularIndexing(p, 128, 32),
        )
        self.assertIsNone(generate_inverse_formula(bounded_expr, p, 4096))
        self.assertIsNone(generate_inverse_formula(bounded_expr, p, 32768))

        nested_reconstruction = 128 * FloorDiv(p, 128) + ModularIndexing(p, 1, 128)
        nested_bounded_expr = bounded_expr.xreplace(
            {ModularIndexing(p, 16, 32): ModularIndexing(nested_reconstruction, 16, 32)}
        )
        self.assertEqual(
            generate_inverse_formula(nested_bounded_expr, p, 16384),
            generate_inverse_formula(bounded_expr, p, 16384),
        )

        overlapping_floors = (
            ModularIndexing(p, 1, 2) + 2 * FloorDiv(p, 2) + 4 * FloorDiv(p, 4)
        )
        self.assertIsNone(generate_inverse_formula(overlapping_floors, p, 8))

        blockwise_expr = (
            100 * FloorDiv(p, 100)
            + 10 * ModularIndexing(p, 1, 10)
            + FloorDiv(ModularIndexing(p, 1, 100), 10)
        )
        self.assertIsNone(generate_inverse_formula(blockwise_expr, p, 15))
        self.assertIsNotNone(generate_inverse_formula(blockwise_expr, p, 100))

    def test_flattened_read_inverse_is_cached(self):
        # [FUNCTION DOC] Checks that the scheduler caches flattened read inverses: with
        # the inverse generator wrapped by a counting mock, two identical calls both return
        # a result but the generator runs once.
        scheduler = object.__new__(Scheduler)
        i0, i1 = sympy.symbols("i0 i1", integer=True, nonnegative=True)
        args = (
            2 * i1 + i0,
            (i0, i1),
            (sympy.Integer(2), sympy.Integer(4)),
            sympy.Integer(8),
        )

        with mock.patch(
            "torch._inductor.invert_expr_analysis.generate_inverse_formula",
            wraps=generate_inverse_formula,
        ) as inverse:
            self.assertIsNotNone(scheduler._get_flattened_read_inverse(*args))
            self.assertIsNotNone(scheduler._get_flattened_read_inverse(*args))

        self.assertEqual(inverse.call_count, 1)

    def test_failed_flattened_read_inverse_is_cached(self):
        # [FUNCTION DOC] Checks that a failed inverse (generator returns none) is also
        # cached: both calls return none and the generator runs once.
        scheduler = object.__new__(Scheduler)
        i0, i1 = sympy.symbols("i0 i1", integer=True, nonnegative=True)
        args = (
            2 * i1 + i0,
            (i0, i1),
            (sympy.Integer(2), sympy.Integer(4)),
            sympy.Integer(8),
        )

        with mock.patch(
            "torch._inductor.invert_expr_analysis.generate_inverse_formula",
            return_value=None,
        ) as inverse:
            self.assertIsNone(scheduler._get_flattened_read_inverse(*args))
            self.assertIsNone(scheduler._get_flattened_read_inverse(*args))

        self.assertEqual(inverse.call_count, 1)

    def test_zero_numel_flattened_read_is_rejected(self):
        # [FUNCTION DOC] Checks that a flattened read with a zero-sized dimension and zero
        # total element count is rejected, returning none.
        scheduler = object.__new__(Scheduler)
        i0, i1 = sympy.symbols("i0 i1", integer=True, nonnegative=True)

        self.assertIsNone(
            scheduler._get_flattened_read_inverse(
                i0 + i1,
                (i0, i1),
                (sympy.Integer(2), sympy.S.Zero),
                sympy.S.Zero,
            )
        )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
