# Owner(s): ["module: inductor"]
# [MODULE DOC] Unit tests for Inductor overlap scheduling and collective bucketing. Graphs are
# traced with make_fx under FakeTensorMode on a fake process group, then run through
# OverlapPreservingBucketer, ManualOverlapPreservingBucketer, OverlapScheduler or
# schedule_overlap_bucketing and the resulting FX graph is checked (FileCheck, node counts,
# hiding nodes, exposed time). Also covers the profile-guided runtime estimator (synthetic
# Chrome trace files), foreach pre-bucketing, FSDP pre-bucketing, comm saturation model and
# BitsetAncestors. Import-time side effects: only reads the device type via get_devtype; test
# classes are gated on an accelerator dist backend and GPU availability.
# KNOWN ISSUE: the module repeats imports of torch, torch._dynamo, torch._dynamo.logging and
# torch._dynamo.test_case and carries a second Owner comment; harmless duplicates.
import json
import os
import tempfile
import unittest

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed as dist
import torch.fx as fx

# for some reason importing functional collectives after dynamo breaks collectives handling!
from torch._C import FileCheck
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.profile_guided_estimation import (
    ProfileData,
    ProfileGuidedEstimator,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._ordered_set import OrderedSet


# Owner(s): ["module: inductor"]


aten = torch.ops.aten

from torch.testing._internal.common_fsdp import get_devtype


device_type = get_devtype().type


import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case


# for some reason importing functional collectives after dynamo breaks collectives handling!


@requires_accelerator_dist_backend(["nccl", "xccl"])
def build_collective_info(graph, hiding_annotations):
    # [FUNCTION DOC] HELPER: Builds the mapping from collective start node to a CollectiveInfo
    # for a traced graph. Each wait_tensor call node whose first argument is an FX node
    # defines a start/wait pair. hiding_annotations maps a start node to one hiding node or a
    # list/OrderedSet of hiding nodes (missing means none). Uses fixed fake numbers: size 64
    # bytes, estimated time 1.0 ms, exposed time 0.0 ms if any hiding node else 1.0 ms.
    # Returns the dict; the graph is not modified.
    """
    Build CollectiveInfo dict from manual hiding annotations.

    hiding_annotations: dict mapping collective_start -> hiding_compute_node(s)
                        Can be a single node or a list/OrderedSet of nodes
    """
    from torch._inductor.fx_passes.overlap_scheduling import CollectiveInfo

    collective_info = {}

    # Find all collective starts and their corresponding waits
    start_to_wait = {}
    for node in graph.nodes:
        if node.op == "call_function" and "wait_tensor" in str(node.target):
            wait_input = node.args[0]
            if isinstance(wait_input, fx.Node):
                start_to_wait[wait_input] = node

    # Build CollectiveInfo for each collective
    for start_node, wait_node in start_to_wait.items():
        hiding_annotation = hiding_annotations.get(start_node)

        # Convert to OrderedSet
        hiding_nodes = OrderedSet()
        if hiding_annotation is not None:
            if isinstance(hiding_annotation, list | OrderedSet):
                hiding_nodes = OrderedSet(hiding_annotation)
            else:
                hiding_nodes = OrderedSet([hiding_annotation])

        # Estimate size and time
        size_bytes = 16 * 4  # 4x4 tensor of floats
        estimated_time_ms = 1.0  # Dummy time
        exposed_time_ms = 0.0 if hiding_nodes else 1.0  # Hidden if has hiding_nodes

        collective_info[start_node] = CollectiveInfo(
            start_node=start_node,
            wait_node=wait_node,
            size_bytes=size_bytes,
            estimated_time_ms=estimated_time_ms,
            exposed_time_ms=exposed_time_ms,
            hiding_nodes=hiding_nodes,
        )

    return collective_info


@requires_accelerator_dist_backend()
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
@instantiate_parametrized_tests
class TestOverlapPreservingBucketing(InductorTestCase):
    # [CLASS DOC] Test case for the overlap-preserving bucketer: checks which independent,
    # nested or interleaved all_gather, all_reduce and reduce_scatter collectives get merged,
    # in several bucket modes. Needs a fake process group (world size 2) created in setUpClass.
    """
    Unit tests for overlap-preserving bucketing pass.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 2, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_can_bucket_independent_collectives(self):
        # [FUNCTION DOC] Two all_gathers, each with its own hiding mm, are independent, so the
        # bucketer merges them: the final graph has at least one all_gather_into_tensor_out node.
        """
        Test that independent collectives with separate hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 (hides ag1) -> mm2 (hides ag2) -> ag1_wait -> ag2_wait
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two all_gather_into_tensor starts, two
            # independent mm calls, then two waits; returns the sum of all results.
            group_name = "0"
            group_size = 1

            # Start both collectives
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )

    def test_cant_bucket_nested_hiding_intervals(self):
        # [FUNCTION DOC] Nested hiding intervals (ag2 starts, is hidden and waited inside
        # ag1's interval) must prevent bucketing: the graph still has at least two
        # all_gather_into_tensor entries after the pass.
        """
        Test that nested hiding intervals prevent bucketing.

        Graph structure:
        ag1_start -> ag2_start -> mm2 (hides ag2) -> ag2_wait -> mm1 (hides ag1) -> ag1_wait

        ag2's hiding interval is nested inside ag1's hiding interval.
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: ag1 start, ag2 start, mm2, wait on ag2, mm1
            # consuming the ag2 result, wait on ag1; returns a sum.
            group_name = "0"
            group_size = 1

            # ag1 starts first
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )

            # ag2 starts (inside ag1's interval)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # mm2 hides ag2
            mm2 = torch.mm(b[:2, :2], b[:2, :2])

            # ag2 waits (still inside ag1's interval)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            # mm1 uses ag2's result and hides ag1
            mm1 = torch.mm(a + ag2_out[:4, :4], a)

            # ag1 waits last
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        # mm2 is the first mm, mm1 is the second (based on graph order)
        mm2 = mm_nodes[0]
        mm1 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: nested hiding intervals should prevent bucketing
        # Should have 2 separate all_gathers, not 1 bucketed one
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor", 2, exactly=False).run(
            graph_str
        )

    @parametrize("final_mm_hidden", (True, False))
    def test_cant_bucket_ag_with_rs_hiding_interval_between(self, final_mm_hidden):
        # [FUNCTION DOC] Parametrized by final_mm_hidden. Graph is all_gather, reduce_scatter,
        # all_gather each hidden by an mm. The reduce_scatter is annotated as hidden only when
        # final_mm_hidden is true; in that case its hiding interval sits between the
        # all_gathers and they must stay separate (two all_gather_into_tensor nodes),
        # otherwise they merge into one all_gather_into_tensor_out. In both cases the three mm
        # nodes must keep their order.
        """
        Test that all_gathers can't bucket when a reduce_scatter's hiding interval is between them.

        Graph structure:
        ag1_start -> mm1 (hides ag1) -> ag1_wait ->
        rs_start -> mm2 (hides rs) -> rs_wait ->

        if final_mm_hidden:
            ag2_start -> mm3 (hides ag2) -> ag2_wait

        if final_mm_hidden:
            Bucketing ag1 and ag2 would require moving one of them, which would break hiding relationships:
            - Moving ag2 earlier would break ag2's hiding by mm3
            - Moving ag1 later would break ag1's hiding by mm1
            - The rs hiding interval creates an obstacle between them

        otherwise, we can bucket
        """

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: Traced body: all_gather on the default group name with
            # mm1, reduce_scatter with mm2, then a second all_gather with mm3; returns the sum
            # and the three mm results.
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 1

            # First all_gather
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            mm1 = torch.mm(a, a)  # hides ag1
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            # Reduce scatter in between
            rs = torch.ops._c10d_functional.reduce_scatter_tensor(
                b, "sum", group_size, group_name
            )
            mm2 = torch.mm(b[:4, :4], b[:4, :4])  # hides rs
            rs_out = torch.ops._c10d_functional.wait_tensor(rs)

            # Second all_gather
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )
            mm3 = torch.mm(c, c)  # hides ag2
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + rs_out.sum() + ag2_out.sum(), mm1, mm2, mm3

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(8, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)

            # Trace with make_fx
            traced = make_fx(func)(a, b, c)

        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        (rs,) = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            # rs: mm2,   # mm2 hides rs
            ag2: mm3,
        }
        if final_mm_hidden:
            hiding_annotations[rs] = mm2

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing logic to find buckets (without applying them, which would require process groups)
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        graph_str = str(traced.graph)

        # check order of mms preserved
        FileCheck().check("%mm").check("%mm_1").check("%mm_2").run(graph_str)

        if final_mm_hidden:
            # Should NOT bucket - 2 separate all_gathers
            # Count all_gather node names (works even when wrapped in control_deps)
            FileCheck().check_count("%all_gather_into_tensor", 2, exactly=False).run(
                graph_str
            )
        else:
            # Should bucket - 1 bucketed all_gather (all_gather_into_tensor_out)
            FileCheck().check_count(
                "%all_gather_into_tensor_out", 1, exactly=False
            ).run(graph_str)

    def test_can_bucket_all_reduce(self):
        # [FUNCTION DOC] Two all_reduce collectives hidden by separate mms are bucketed
        # (internode-only restriction turned off): the graph shows one cat, one all_reduce and
        # a split_with_sizes, and both mm nodes remain.
        """
        Test that all_reduce operations CAN bucket together.

        Graph structure:
        ar1_start -> ar2_start -> mm1 (hides ar1) -> mm2 (hides ar2) -> ar1_wait -> ar2_wait
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two all_reduce starts, two mm calls, two
            # waits; returns a sum.
            group_name = "0"

            # Start both all_reduce operations
            ar1 = torch.ops._c10d_functional.all_reduce(a, "sum", group_name)
            ar2 = torch.ops._c10d_functional.all_reduce(b, "sum", group_name)

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)

            # Wait for both
            ar1_out = torch.ops._c10d_functional.wait_tensor(ar1)
            ar2_out = torch.ops._c10d_functional.wait_tensor(ar2)

            return ar1_out.sum() + ar2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes
        ar1, ar2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # For all_reduce, start_node == wait_node (no separate wait)
        hiding_annotations = {
            ar1: mm1,
            ar2: mm2,
        }

        # Build collective info
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_only_internode_comms=False,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed all_reduce
        # After bucketing, there should be only one all_reduce node (the bucketed one)
        # Check for cat (bucketing input) and split_with_sizes (bucketing output)
        graph_str = str(traced.graph)
        FileCheck().check("cat.default").check("all_reduce.default").check(
            "split_with_sizes"
        ).check_count("%mm", 2).run(graph_str)

    def test_manual_bucket_splits_dependent_all_reduce(self):
        # [FUNCTION DOC] Regression test for the manual bucketer: four sum all_reduces share
        # one bucket key but two depend on the others (sumexp feeds result). Checks that
        # _split_independent_collectives returns exactly two buckets (the independent sumexps,
        # and the dependent results), then that manual_bucket_collectives runs without a
        # cycle, graph lint passes, and the graph ends with two all_reduce and two cat nodes.
        """Bucketing must split dependent same-key all_reduces, not fuse them.

        Reproduces the loss-parallel cross-entropy pattern with two independent
        chunks: a "sumexp" all_reduce whose result feeds a "result" all_reduce.
        All four are sum/same-group/same-dtype (one bucket key), but fusing a
        sumexp with its dependent result would make the merged collective's
        input depend on its own output -- a cycle that failed region
        topological sort ("stable topological sort of region failed").

        Correct partitioning fuses the two independent sumexps into one bucket
        and the two independent results into another, so four all_reduces
        collapse to exactly two bucketed all_reduces (via two cats), no cycle.
        """

        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Traced body: two independent waited all_reduces (sumexp),
            # each result plus another input feeds a second waited all_reduce; returns the sum
            # of both results.
            group_name = "0"
            ar = torch.ops._c10d_functional.all_reduce
            wait = torch.ops._c10d_functional.wait_tensor
            sumexp0 = wait(ar(a, "sum", group_name))
            sumexp1 = wait(ar(b, "sum", group_name))
            result0 = wait(ar(sumexp0 + c, "sum", group_name))
            result1 = wait(ar(sumexp1 + d, "sum", group_name))
            return result0.sum() + result1.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)
            d = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b, c, d)

        collective_info = build_collective_info(traced.graph, {})
        scheduled = OrderedSet(traced.graph.nodes)

        from torch._inductor.fx_passes.overlap_manual_scheduling import (
            ManualOverlapPreservingBucketer,
        )

        bucketer = ManualOverlapPreservingBucketer(
            traced.graph, collective_info, scheduled
        )

        # find_nodes returns nodes in graph (topological) order.
        sumexp0, sumexp1, result0, result1 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )

        # The partition itself: independent sumexps in one bucket, dependent
        # results in another -- a sumexp is never grouped with its own result.
        buckets = bucketer._split_independent_collectives(
            OrderedSet([sumexp0, sumexp1, result0, result1]),
            list(traced.graph.nodes),
        )
        bucket_sets = [set(b) for b in buckets]
        self.assertEqual(len(buckets), 2)
        self.assertIn({sumexp0, sumexp1}, bucket_sets)
        self.assertIn({result0, result1}, bucket_sets)

        # End to end: bucketing must not raise, and the partition above means the
        # two independent pairs each fuse (two cats) while the dependent pairs
        # stay separate, so four all_reduces collapse to exactly two.
        bucketer.manual_bucket_collectives(list(traced.graph.nodes))
        traced.graph.lint()

        all_reduces = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )
        cats = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.cat.default
        )
        self.assertEqual(len(all_reduces), 2)
        self.assertEqual(len(cats), 2)

    def test_no_cross_type_bucketing_ar_and_rs(self):
        # [FUNCTION DOC] all_reduce and reduce_scatter with the same group, reduce op and
        # dtype must not be merged across types. With no hiding at all, the bucketer
        # (internode restriction off) ends with exactly one all_reduce and one
        # reduce_scatter_tensor node.
        """
        Test that all_reduce and reduce_scatter on the same PG with
        matching reduce_op and dtype are NOT bucketed together.

        bucket_key() returns (group_name, reduce_op, dtype) for both
        all_reduce and reduce_scatter. Without the collective type in
        the key, they would be incorrectly grouped together.
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two all_reduce and two reduce_scatter_tensor
            # starts (group size 2), all waited; returns a sum.
            group_name = "0"
            group_size = 2

            ar1 = torch.ops._c10d_functional.all_reduce(a, "sum", group_name)
            ar2 = torch.ops._c10d_functional.all_reduce(b, "sum", group_name)

            rs1 = torch.ops._c10d_functional.reduce_scatter_tensor(
                a, "sum", group_size, group_name
            )
            rs2 = torch.ops._c10d_functional.reduce_scatter_tensor(
                b, "sum", group_size, group_name
            )

            ar1_out = torch.ops._c10d_functional.wait_tensor(ar1)
            ar2_out = torch.ops._c10d_functional.wait_tensor(ar2)
            rs1_out = torch.ops._c10d_functional.wait_tensor(rs1)
            rs2_out = torch.ops._c10d_functional.wait_tensor(rs2)

            return ar1_out.sum() + ar2_out.sum() + rs1_out.sum() + rs2_out.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2
            traced = make_fx(func)(a, b)

        ar1, ar2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )
        rs1, rs2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )

        # No hiding — all exposed
        hiding_annotations = {}

        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_only_internode_comms=False,
        )
        bucketer.bucket_collectives()

        # all_reduce ops should be bucketed together (1 bucketed all_reduce)
        ar_nodes = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )
        self.assertEqual(len(ar_nodes), 1)

        # reduce_scatter ops should be bucketed together (1 bucketed reduce_scatter)
        rs_nodes = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        self.assertEqual(len(rs_nodes), 1)

    def test_reduce_scatter_coalesced_mode(self):
        # [FUNCTION DOC] In coalesced bucket mode two reduce_scatters become a
        # reduce_scatter_tensor_coalesced call with no cat node in the graph.
        """
        Test that 'coalesced' bucket mode uses reduce_scatter_tensor_coalesced
        instead of cat + single reduce_scatter.
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two reduce_scatter_tensor calls with waits;
            # returns a sum.
            group_name = "0"
            group_size = 2

            rs1 = torch.ops._c10d_functional.reduce_scatter_tensor(
                a, "sum", group_size, group_name
            )
            rs2 = torch.ops._c10d_functional.reduce_scatter_tensor(
                b, "sum", group_size, group_name
            )

            rs1_out = torch.ops._c10d_functional.wait_tensor(rs1)
            rs2_out = torch.ops._c10d_functional.wait_tensor(rs2)

            return rs1_out.sum() + rs2_out.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2
            traced = make_fx(func)(a, b)

        rs1, rs2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )

        hiding_annotations = {}
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="coalesced",
        )
        bucketer.bucket_collectives()

        graph_str = str(traced.graph)
        self.assertIn("reduce_scatter_tensor_coalesced", graph_str)
        self.assertNotIn("cat.default", graph_str)

    def test_all_gather_coalesced_mode_falls_back(self):
        # [FUNCTION DOC] Coalesced bucket mode only applies to reduce_scatter, so two
        # all_gathers fall back to default bucketing: the graph has a cat node and no
        # all_gather_into_tensor_coalesced.
        """
        Test that 'coalesced' bucket mode falls back to default bucketing
        for all_gather (coalesced is only supported for reduce_scatter).
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two all_gather_into_tensor calls with waits;
            # returns a sum.
            group_name = "0"
            group_size = 2

            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2
            traced = make_fx(func)(a, b)

        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )

        hiding_annotations = {}
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="coalesced",
        )
        bucketer.bucket_collectives()

        # Should use default bucketing (cat + single all_gather), not coalesced
        graph_str = str(traced.graph)
        self.assertIn("cat.default", graph_str)
        self.assertNotIn("all_gather_into_tensor_coalesced", graph_str)

    def test_reduce_scatter_coalesced_mode_single_rs(self):
        # [FUNCTION DOC] In coalesced mode a single reduce_scatter (one wait) must not crash;
        # the group has one element so nothing is merged and the reduce_scatter_tensor remains
        # in the graph.
        """
        Test that 'coalesced' bucket mode correctly identifies the collective
        start node even when there is only a single reduce_scatter (so only
        one wait node is produced).  This exercises the unified
        _get_collective_node_from_wait path instead of the .args[0] shortcut.
        """

        def func(a):
            # [FUNCTION DOC] HELPER: Traced body: one reduce_scatter_tensor with its wait;
            # returns the sum.
            group_name = "0"
            group_size = 2

            rs1 = torch.ops._c10d_functional.reduce_scatter_tensor(
                a, "sum", group_size, group_name
            )

            rs1_out = torch.ops._c10d_functional.wait_tensor(rs1)

            return rs1_out.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a)

        rs_nodes = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        self.assertEqual(len(rs_nodes), 1)

        hiding_annotations = {}
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="coalesced",
        )
        # Single RS should not crash — bucket_collectives groups by key,
        # and a single-element group is a no-op (no merge needed).
        bucketer.bucket_collectives()

        # Graph should still contain the original reduce_scatter (no merge happened)
        graph_str = str(traced.graph)
        self.assertIn("reduce_scatter_tensor", graph_str)

    def test_can_bucket_multidtype_collectives(self):
        # [FUNCTION DOC] Float32 and bfloat16 all_gathers, each hidden by its own mm, are
        # merged in the custom_ops_multidtype bucket mode into a single all_gather_into_tensor_out.
        """
        Test that all_gathers with different dtypes CAN bucket together.

        Graph structure:
        ag1_float32 -> mm1 (hides ag1) -> ag1_wait
        ag2_bfloat16 -> mm2 (hides ag2) -> ag2_wait
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: all_gather of a float32 and of a bfloat16
            # input, two mm calls (second on float-converted input), two waits; returns a sum.
            group_name = "0"
            group_size = 1

            # Start both collectives with different dtypes
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a,
                group_size,
                group_name,  # float32
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b,
                group_size,
                group_name,  # bfloat16
            )

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b.float(), b.float())

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            b = torch.ones(4, 4, device=self.device, dtype=torch.bfloat16)

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        mm1 = mm_nodes[0]
        mm2 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing with multidtype mode
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="custom_ops_multidtype",
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        # even though dtypes are different
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )

    def test_can_bucket_with_multiple_hiding_nodes(self):
        # [FUNCTION DOC] Collectives with overlapping multiple hiding nodes (ag1 by mm1,mm2;
        # ag2 by mm2,mm3) can bucket. Also asserts the hiding_nodes sets recorded in
        # collective_info, then expects one all_gather_into_tensor_out and exactly three mm calls.
        """
        Test that collectives with multiple hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 -> mm2 -> mm3 -> ag1_wait -> ag2_wait

        Where:
        - ag1 is hidden by mm1 and mm2
        - ag2 is hidden by mm2 and mm3
        - Both collectives share mm2 as a hiding node
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two all_gather starts, three mm calls (third
            # on a sum of inputs), two waits; returns a sum.
            group_name = "0"
            group_size = 1

            # Start both collectives
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # Three compute operations that hide the collectives
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)
            mm3 = torch.mm(a + b, a + b)

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum() + mm3.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships with multiple hiding nodes
        hiding_annotations = {
            ag1: [mm1, mm2],  # ag1 is hidden by mm1 and mm2
            ag2: [mm2, mm3],  # ag2 is hidden by mm2 and mm3
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Verify hiding_nodes are correctly set
        self.assertEqual(len(collective_info[ag1].hiding_nodes), 2)
        self.assertIn(mm1, collective_info[ag1].hiding_nodes)
        self.assertIn(mm2, collective_info[ag1].hiding_nodes)
        self.assertEqual(len(collective_info[ag2].hiding_nodes), 2)
        self.assertIn(mm2, collective_info[ag2].hiding_nodes)
        self.assertIn(mm3, collective_info[ag2].hiding_nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
        )
        bucketer.bucket_collectives()

        FileCheck().check_count(
            "all_gather_into_tensor_out", 1, exactly=False
        ).check_count("torch.ops.aten.mm.default", 3, exactly=True).run(
            str(traced.graph)
        )

    def test_can_bucket_with_convert_dtype_as_hiding_nodes(self):
        # [FUNCTION DOC] Uses dtype-conversion nodes as hiding nodes in custom_ops_multidtype
        # mode: ag1 (hidden by a convert) stays separate and ag2/ag3 (hidden by an mm) are
        # merged. FileCheck expects an unbucketed all_gather, a wait_tensor, then
        # pre_bucket_all_gather and all_gather_into_tensor_out.
        """
        Test that all_gathers can bucket when convert_element_type ops ARE the hiding nodes.

        Graph structure:
        ag1_start -> convert1 (hides ag1) -> ag1_wait -> ag2_start -> convert2 (hides ag2) -> ag2_wait

        The convert_element_type ops ARE hiding nodes - no matmuls.
        This tests that dependencies are transferred correctly when convert nodes are erased.
        """

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: Traced body: all_gather of a, convert b to float16, wait,
            # two more all_gathers (b converted and c), an mm on the first result, waits;
            # returns the four results.
            group_name = "0"
            group_size = 1

            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = torch.ops.prims.convert_element_type.default(b, torch.float16)
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag3 = torch.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )

            mm = ag1_out @ ag1_out

            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)
            ag3_out = torch.ops._c10d_functional.wait_tensor(ag3)

            return ag1_out, ag2_out, ag3_out, mm

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            b = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            c = torch.ones(4, 4, device=self.device, dtype=torch.float32)

            traced = make_fx(func)(a, b, c)

        # Find nodes
        ag1, ag2, ag3 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        convert1 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops.prims.convert_element_type.default,
        )[0]
        mm = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.mm.default,
        )[0]

        hiding_annotations = {
            ag1: convert1,
            ag2: mm,
            ag3: mm,
        }

        # Build collective info and scheduled
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            scheduled,
            bucket_mode="custom_ops_multidtype",
        )
        bucketer.bucket_collectives()

        graph_str = str(traced.graph)

        # Expect: ag1 (separate, hidden by convert1) -> wait -> ag2+ag3 bucketed (hidden by mm)
        # Check for pre_bucket (for ag2+ag3) and all_gather_into_tensor_out (bucketed)
        f = FileCheck()
        f.check("all_gather_into_tensor.default").check("wait_tensor")
        f.check("pre_bucket_all_gather").check("all_gather_into_tensor_out")
        f.run(graph_str)

    @torch._inductor.config.patch(deterministic=True)
    def test_deterministic_mode_no_benchmark_error(self):
        # [FUNCTION DOC] With inductor deterministic config patched on, running
        # schedule_overlap_bucketing on a traced graph (all_gather, gemm and pointwise
        # compute) must not raise; compute estimation falls back to analytical instead of
        # benchmarking. Only checks absence of an error.
        """
        Test that deterministic mode doesn't error when running overlap scheduling.

        Before the fix, deterministic mode would error when trying to benchmark
        compute nodes. Now it uses analytical estimation instead.
        """
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: all_gather of a, mm of a and b plus one,
            # then wait; returns the sum of the pointwise result and gathered tensor.
            group_name = "0"
            group_size = 1

            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )

            # Compute with gemm
            mm_result = torch.mm(a, b)
            pointwise = mm_result + 1.0

            ag_out = torch.ops._c10d_functional.wait_tensor(ag)

            return (pointwise + ag_out).sum()

        with FakeTensorMode():
            a = torch.randn(16, 16, device=self.device)
            b = torch.randn(16, 16, device=self.device)
            gm = make_fx(func)(a, b)

        # Should not error in deterministic mode (would have errored before fix)
        schedule_overlap_bucketing(gm)

    def test_assume_bucketed_latency_exceeds_exposed_time(self):
        # [FUNCTION DOC] Runs schedule_overlap_bucketing with a custom runtime estimator whose
        # latency-only estimate (override_size zero) for all_gather is larger than the full
        # estimate, which could make the latency subtraction go negative. Passes if scheduling
        # completes without error; no assertion.
        """
        When assume_bucketing_reduces_latency is True and two same-type
        collectives are in-flight, the latency subtraction in
        _handle_collective_start must not go negative.

        Trigger: custom_runtime_estimation returns a higher value for
        override_size=0 (latency) than for override_size=None (full estimate).
        This can happen after analytical model recalibration for small collectives.
        """
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: all_gather ag1, an mm, a second all_gather
            # ag2, both waited; returns a sum.
            group_name = "0"
            group_size = 1

            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            mm1 = torch.mm(a, b)

            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return mm1.sum() + ag1_out.sum() + ag2_out.sum()

        with FakeTensorMode():
            a = torch.randn(4, 4, device=self.device)
            b = torch.randn(4, 4, device=self.device)
            gm = make_fx(func)(a, b)

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            # [FUNCTION DOC] HELPER: Fake runtime estimator: all_gather returns 10.0 when
            # override_size is zero else 3.0; mm nodes return 5.0; anything else 0.0.
            if "all_gather" in str(node.target):
                if override_size == 0:
                    return 10.0  # latency only
                return 3.0  # full estimate < latency
            if "mm" in str(node.target):
                return 5.0
            return 0.0

        schedule_overlap_bucketing(
            gm,
            custom_runtime_estimation=custom_runtime,
            pre_bucketing_fsdp_collectives=False,
        )


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestCrossPGOverlap(InductorTestCase):
    # [CLASS DOC] Test case for scheduling across two different process groups. setUpClass
    # creates the default fake group plus two extra groups (pg1, pg2) over ranks 0 and 1 and
    # stores their names.
    """
    Tests for cross-PG overlap scheduling.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 2, in-memory FakeStore) and
        # sets the class attribute device to cuda. Also creates two new groups over ranks 0
        # and 1 and stores them and their group names as pg1, pg2, pg1_name, pg2_name.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

        # Create two separate process groups for cross-PG testing
        cls.pg1 = dist.new_group(ranks=[0, 1])
        cls.pg2 = dist.new_group(ranks=[0, 1])
        cls.pg1_name = cls.pg1.group_name
        cls.pg2_name = cls.pg2.group_name

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the two
        # extra groups and then the default process group.
        super().tearDownClass()
        dist.destroy_process_group(cls.pg1)
        dist.destroy_process_group(cls.pg2)
        dist.destroy_process_group()

    def test_cross_pg_prefetch_during_exposed_wait(self):
        # [FUNCTION DOC] With OverlapScheduler (no bucketing, analytical collective estimator,
        # long 10.0 collective estimate), the second all_gather on PG2 should be started
        # during the exposed wait of the first on PG1. Expects both starts before the first
        # wait_tensor and the overlap_scheduling_exposed counter equal to 1.
        """
        Test that ag2 on PG2 gets prefetched during exposed wait of ag1 on PG1.
        """
        pg1_name = self.pg1_name
        pg2_name = self.pg2_name

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: all_gather on PG1, wait, mm of the result,
            # all_gather on PG2, wait, mm; returns the sum of the mms.
            group_size = 1

            # First collective on PG1
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, pg1_name
            )
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            mm1 = torch.mm(ag1_out[:4, :4], ag1_out[:4, :4])

            # Second collective on PG2
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, pg2_name
            )
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)
            mm2 = torch.mm(ag2_out[:4, :4], ag2_out[:4, :4])

            return mm1 + mm2

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            traced = make_fx(func)(a, b)

        # Find nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        wait1, wait2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.wait_tensor.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 10.0 for all_gather nodes, else 0.0.
            if "all_gather" in str(node.target):
                return 10.0  # Long collective to ensure exposed wait
            return 0.0

        # Run overlap scheduler
        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        )
        out = scheduler.run()
        FileCheck().check("%all_gather_into_tensor").check(
            "%all_gather_into_tensor"
        ).check("%wait_tensor").run(str(out.graph))

        self.assertEqual(counters["inductor"]["overlap_scheduling_exposed"], 1)

    def test_two_queue_scheduling_off_path_nodes(self):
        # [FUNCTION DOC] Two-queue scheduling: off-path reduce_scatters (results only waited
        # at the end) must stay interspersed with compute instead of drifting past the last
        # mm. Runs schedule_overlap_bucketing with a fake estimator and asserts some
        # reduce_scatter start precedes the last mm in the resulting order.
        """
        Test that off-path nodes (reduce_scatters whose results don't block compute)
        are scheduled near their original position rather than drifting to the end.

        Without two-queue scheduling, off-path nodes get domination=inf and drift
        to end. With two-queue, they stay near original position.
        """

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: an on-path all_gather feeding mm1, then
            # reduce_scatter of mm1, mm2, reduce_scatter of mm2, mm3, with both reduce_scatter
            # waits at the end.
            group_name = "0"
            group_size = 2

            # On-path: all_gather whose result is used by compute
            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)

            # mm1 uses all_gather result (makes ag on-path)
            mm1 = torch.mm(a, ag_out[:4, :4])

            # Off-path: reduce_scatter result not used by further compute
            rs1 = torch.ops._c10d_functional.reduce_scatter_tensor(
                mm1, "sum", group_size, group_name
            )

            mm2 = torch.mm(a, a)
            rs2 = torch.ops._c10d_functional.reduce_scatter_tensor(
                mm2, "sum", group_size, group_name
            )

            mm3 = torch.mm(a, a)

            # Waits at end (like gradient outputs)
            rs1_out = torch.ops._c10d_functional.wait_tensor(rs1)
            rs2_out = torch.ops._c10d_functional.wait_tensor(rs2)

            return mm3.sum() + rs1_out.sum() + rs2_out.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 1.0 for all_gather or
            # reduce_scatter nodes, else 0.0.
            if "all_gather" in str(node.target) or "reduce_scatter" in str(node.target):
                return 1.0
            return 0.0

        out = schedule_overlap_bucketing(
            traced, custom_runtime_estimation=custom_runtime, max_off_bucket_gb=None
        )

        # Get scheduled order
        node_names = [n.name for n in out.graph.nodes if n.op == "call_function"]
        rs_starts = [
            i
            for i, name in enumerate(node_names)
            if "reduce_scatter" in name and "wait" not in name
        ]
        mm_positions = [i for i, name in enumerate(node_names) if name.startswith("mm")]

        # Off-path reduce_scatters should be interspersed with compute, not all at end
        last_mm = max(mm_positions)
        self.assertTrue(
            any(p < last_mm for p in rs_starts),
            lambda msg: f"{msg}\nOff-path reduce_scatters drifted to end: rs={rs_starts}, mm={mm_positions}, names={node_names}",
        )

    @torch._inductor.config.patch(
        {"test_configs.assume_bucketing_reduces_latency": False}
    )
    def test_prefetch_prioritizes_larger_hidden_time(self):
        # [FUNCTION DOC] With assume_bucketing_reduces_latency off and at most one collective
        # in flight (tiny memory limit), the scheduler must prefetch the collective that can
        # use more of the mm's overlap window: the large all_gather is hidden by the mm
        # (exposed time 0.0), the small one is not (exposed 1.0), total exposed time 1.0.
        """
        When multiple future collectives have the same semantic priority, choose
        the one that can consume more of the current overlap window first.
        """
        group_name = self.pg1_name

        def func(a, b, c):
            # [FUNCTION DOC] HELPER: Traced body: an mm, two all_gathers of different inputs
            # (small and large by estimator), both waited; returns a sum.
            group_size = 1
            mm = torch.mm(a, a)

            ag_small = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag_large = torch.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )

            wait_small = torch.ops._c10d_functional.wait_tensor(ag_small)
            wait_large = torch.ops._c10d_functional.wait_tensor(ag_large)
            return mm.sum() + wait_small.sum() + wait_large.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b, c)

        ag_small, ag_large = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        (mm,) = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        def custom_runtime(node: fx.Node, override_size: int | None) -> float | None:
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 1.0 for ag_small, 8.0 for
            # ag_large, 8.0 for the mm node, else 0.0.
            if node is ag_small:
                return 1.0
            if node is ag_large:
                return 8.0
            if node.target == torch.ops.aten.mm.default:
                return 8.0
            return 0.0

        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            # Each all_gather has a 128B footprint in this graph.  Limit in-flight
            # collectives to one so picking the larger hidden-time candidate
            # reduces total exposed time, rather than only changing order.
            max_in_flight_gb=0.0000002,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        )
        scheduler.run()

        self.assertIn(mm, scheduler.collective_info[ag_large].hiding_nodes)
        self.assertNotIn(mm, scheduler.collective_info[ag_small].hiding_nodes)
        self.assertEqual(scheduler.collective_info[ag_large].exposed_time_ms, 0.0)
        self.assertEqual(scheduler.collective_info[ag_small].exposed_time_ms, 1.0)
        self.assertEqual(
            sum(info.exposed_time_ms for info in scheduler.collective_info.values()),
            1.0,
        )


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
@instantiate_parametrized_tests
class TestFusibleNodeOverlap(InductorTestCase):
    # [CLASS DOC] Test case checking that non-matmul fusible compute (pointwise chains, fusion
    # regions) is counted as hiding nodes for collectives. Uses a fake process group of world
    # size 2.
    """Test that fusible nodes are used for overlapping with collectives."""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 2, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_fusible_nodes_hide_collective(self):
        # [FUNCTION DOC] OverlapScheduler without a custom estimator should treat a pointwise
        # add/mul/sub chain as hiding the all_gather: the collective's hiding_nodes is non-
        # empty and the graph order is all_gather, add, mul, sub, wait_tensor.
        """Test that fusible (non-mm) nodes can hide collectives."""

        def func(a):
            # [FUNCTION DOC] HELPER: Traced body: all_gather start, three chained pointwise
            # ops, then wait; returns the sum of the gathered result and the pointwise result.
            group_name = "0"
            group_size = 1

            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            # Chain of pointwise ops - should be estimated and used for overlap
            b = a + 1
            b = b * 2
            b = b - 3
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return ag_out.sum() + b.sum()

        with FakeTensorMode():
            a = torch.ones(1024, 1024, device=self.device)
            traced = make_fx(func)(a)

        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
        )
        scheduler.run()

        # The collective should have hiding nodes (the pointwise chain)
        (ag_start,) = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        info = scheduler.collective_info[ag_start]
        self.assertGreater(
            len(info.hiding_nodes), 0, "Fusible nodes should hide the collective"
        )

        # Verify graph structure: ag -> pointwise ops -> wait
        graph_str = str(traced.graph)
        FileCheck().check("all_gather_into_tensor").check("add").check("mul").check(
            "sub"
        ).check("wait_tensor").run(graph_str)

    def test_fusion_regions_hide_collective(self):
        # [FUNCTION DOC] Same as the fusible-node test but with enable_fusion_regions=True:
        # the scheduler constructed up front already knows the collective; after run the
        # hiding_nodes is non-empty and the graph order is all_gather, add, mul, sub, wait_tensor.
        """Test that fusion regions can hide collectives when enabled."""

        def func(a):
            # [FUNCTION DOC] HELPER: Traced body: all_gather start, three chained pointwise
            # ops, then wait; returns a sum.
            group_name = "0"
            group_size = 1

            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = a + 1
            b = b * 2
            b = b - 3
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return ag_out.sum() + b.sum()

        with FakeTensorMode():
            a = torch.ones(1024, 1024, device=self.device)
            traced = make_fx(func)(a)

        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=True,
        )
        ag_start = next(iter(scheduler.collective_info.keys()))
        info = scheduler.collective_info[ag_start]
        scheduler.run()

        self.assertGreater(
            len(info.hiding_nodes), 0, "Fusion region should hide the collective"
        )

        # Verify graph structure: ag -> pointwise ops -> wait
        graph_str = str(traced.graph)
        FileCheck().check("all_gather_into_tensor").check("add").check("mul").check(
            "sub"
        ).check("wait_tensor").run(graph_str)

    @parametrize("enable_fusion_regions", [False, True])
    def test_precomputed_estimations_via_custom_runtime(self, enable_fusion_regions):
        # [FUNCTION DOC] Parametrized by enable_fusion_regions. Precomputes estimations with
        # gather_node_runtime_estimations and feeds them to OverlapScheduler through a custom
        # estimator that looks nodes up in the mapping. Expects the final graph to still have
        # all_gather then wait_tensor and exactly one collective in collective_info.
        """Pre-computed estimations from gather_node_runtime_estimations can be
        fed into OverlapScheduler via custom_runtime_estimation wrapping a dict."""

        def func(a):
            # [FUNCTION DOC] HELPER: Traced body: all_gather start, two pointwise ops, wait;
            # returns a sum.
            group_name = "0"
            group_size = 1
            ag = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = a + 1
            b = b * 2
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return ag_out.sum() + b.sum()

        with FakeTensorMode():
            a = torch.ones(1024, 1024, device=self.device)
            traced = make_fx(func)(a)

        from torch._inductor.fx_passes.overlap_scheduling import (
            gather_node_runtime_estimations,
            OverlapScheduler,
        )

        estimations = gather_node_runtime_estimations(
            traced,
            collective_estimator="analytical",
            enable_fusion_regions=enable_fusion_regions,
        )

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=lambda node, _: estimations.get(node),
            collective_estimator="analytical",
        )
        out = scheduler.run()
        FileCheck().check("all_gather_into_tensor").check("wait_tensor").run(
            str(out.graph)
        )
        self.assertEqual(len(scheduler.collective_info), 1)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestOverlapSchedulingFixes(InductorTestCase):
    # [CLASS DOC] Regression tests for specific overlap scheduling bugs (self-dependency
    # cycles with dtype conversion, fusion regions with bucketing, graphsafe rng state). Fake
    # process group world size 16.
    """
    Test cases for specific bug fixes in overlap scheduling.
    These tests would fail without their corresponding fixes.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 16, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=16, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_no_self_dependency_cycle_with_dtype_conversion(self):
        # [FUNCTION DOC] Regression test: bucketing four all_gathers whose inputs are dtype
        # conversions with timeline dependencies must not create self-dependencies. Runs
        # OverlapScheduler with bucketing and overlap deps enabled and fusion regions off, and
        # checks run completes and graph lint passes.
        """
        Test that bucketing collectives with dtype conversion doesn't create
        self-dependency cycles.

        This tests the fix in augmented_graph_helper.py that adds != new_node
        checks to prevent self-dependencies when merging nodes.

        The bug: When two convert_element_type nodes (inputs to all_gathers)
        have timeline dependencies between them and both get merged into
        _pre_bucket_all_gather, the dependency becomes a self-dependency
        which causes _stable_topological_sort to fail.
        """

        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Traced body: four convert-to-bfloat16 plus all_gather
            # pairs (group size 16) with an mm between them, then four waits; returns a sum.
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 16

            # Multiple all_gathers with dtype conversion
            # The convert nodes will have timeline dependencies between them
            conv_a = torch.ops.prims.convert_element_type.default(a, torch.bfloat16)
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_a, group_size, group_name
            )

            conv_b = torch.ops.prims.convert_element_type.default(b, torch.bfloat16)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_b, group_size, group_name
            )

            # Compute between all_gathers
            mm = torch.mm(c, d)

            conv_c = torch.ops.prims.convert_element_type.default(c, torch.bfloat16)
            ag3 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_c, group_size, group_name
            )

            conv_d = torch.ops.prims.convert_element_type.default(d, torch.bfloat16)
            ag4 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_d, group_size, group_name
            )

            # Wait for all
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)
            w3 = torch.ops._c10d_functional.wait_tensor(ag3)
            w4 = torch.ops._c10d_functional.wait_tensor(ag4)

            return w1.sum() + w2.sum() + w3.sum() + w4.sum() + mm.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)
            d = torch.ones(4, 4, device=self.device)

            traced = make_fx(func)(a, b, c, d)

        # Run full overlap scheduling with bucketing enabled
        # This would fail with AssertionError in _stable_topological_sort
        # before the self-dependency fix
        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=True,
            insert_overlap_deps=True,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=False,
        )
        # This should complete without cycle error
        result = scheduler.run()
        result.graph.lint()

    def test_no_cycle_with_fusion_regions_and_bucketing(self):
        # [FUNCTION DOC] Regression test: OverlapScheduler with bucketing, overlap deps and
        # fusion regions enabled over a graph with all_gathers and fusible pointwise chains
        # must finish without a cycle and pass graph lint.
        """Test that fusion regions + bucketing doesn't create cycles."""

        def func(a, b, c, d):
            # [FUNCTION DOC] HELPER: Traced body: converted all_gathers ag1 and ag2, a
            # pointwise chain, their waits, two more converted all_gathers, another pointwise
            # chain, and waits; returns a sum.
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 16

            # Start collectives
            conv_a = torch.ops.prims.convert_element_type.default(a, torch.bfloat16)
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_a, group_size, group_name
            )

            conv_b = torch.ops.prims.convert_element_type.default(b, torch.bfloat16)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_b, group_size, group_name
            )

            # Fusible compute chain (will become a fusion region)
            x = c + 1
            x = x * 2
            x = x - 3
            x = x / 4

            # Wait and use results
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)

            # More collectives
            conv_c = torch.ops.prims.convert_element_type.default(c, torch.bfloat16)
            ag3 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_c, group_size, group_name
            )

            conv_d = torch.ops.prims.convert_element_type.default(d, torch.bfloat16)
            ag4 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_d, group_size, group_name
            )

            # Another fusible chain
            y = d + 1
            y = y * 2

            w3 = torch.ops._c10d_functional.wait_tensor(ag3)
            w4 = torch.ops._c10d_functional.wait_tensor(ag4)

            return w1.sum() + w2.sum() + w3.sum() + w4.sum() + x.sum() + y.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)
            d = torch.ones(4, 4, device=self.device)

            traced = make_fx(func)(a, b, c, d)

        # Run with fusion regions enabled - this exercises all the fixes
        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=True,
            insert_overlap_deps=True,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=None,
            collective_estimator="analytical",
            enable_fusion_regions=True,
        )
        # This should complete without errors
        result = scheduler.run()
        result.graph.lint()

    def test_no_cycle_from_dtype_convert_timeline_deps(self):
        # [FUNCTION DOC] Regression test for transfer_erased_node_deps in
        # AugmentedGraphHelper: builds a traced graph of two converted all_gathers, adds an
        # extra dependency of one convert on the first all_gather start, creates replacement
        # pre_bucket, new_start and new_wait nodes where new_start takes pre_bucket as its out
        # kwarg, transfers dependencies from the erased nodes, and asserts the helper has no cycle.
        """
        Test that transfer_erased_node_deps doesn't create cycles when
        convert_element_type nodes with timeline deps get fused into a
        _pre_bucket node that is a data dependency of the bucketed start.

        Uses make_fx to trace a realistic graph with dtype conversions and
        collectives, then simulates the bucketing transfer that would create
        a cycle: _pre_bucket extra-depends-on new_start, while new_start
        already data-depends on _pre_bucket.
        """
        from torch._inductor.augmented_graph_helper import AugmentedGraphHelper

        def func(a, b):
            # [FUNCTION DOC] HELPER: Traced body: two converted all_gathers with waits;
            # returns a sum.
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 16

            conv_a = torch.ops.prims.convert_element_type.default(a, torch.bfloat16)
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_a, group_size, group_name
            )
            conv_b = torch.ops.prims.convert_element_type.default(b, torch.bfloat16)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                conv_b, group_size, group_name
            )
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)
            return w1.sum() + w2.sum()

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        graph = traced.graph
        nodes = {n.name: n for n in graph.nodes}
        conv_a = nodes["convert_element_type"]
        conv_b = nodes["convert_element_type_1"]
        ag_start_1 = nodes["all_gather_into_tensor"]
        ag_start_2 = nodes["all_gather_into_tensor_1"]
        ag_wait_1 = nodes["wait_tensor"]
        ag_wait_2 = nodes["wait_tensor_1"]

        helper = AugmentedGraphHelper(graph)

        # Simulate PG timeline dep: conv_b scheduled after ag_start_1
        helper.add_extra_dep(n=conv_b, dep=ag_start_1)

        # Simulate bucketed replacement nodes as custom_ops_multidtype would:
        # _pre_bucket is a data dep of new_start (new_start.kwargs["out"])
        x = nodes["a_1"]
        pre_bucket = graph.call_function(torch.sigmoid, args=(x,), name="pre_bucket")
        new_start = graph.call_function(
            torch.tanh, kwargs={"out": pre_bucket}, name="new_start"
        )
        new_wait = graph.call_function(torch.exp, args=(new_start,), name="new_wait")

        # This is what _apply_bucket does: converts → pre_bucket,
        # starts → new_start, waits → new_wait
        helper.transfer_erased_node_deps(
            {
                conv_a: pre_bucket,
                conv_b: pre_bucket,
                ag_start_1: new_start,
                ag_start_2: new_start,
                ag_wait_1: new_wait,
                ag_wait_2: new_wait,
            }
        )

        # Without the fix, pre_bucket gets extra dep on new_start,
        # but new_start already data-depends on pre_bucket → cycle
        self.assertFalse(
            helper.has_cycle(),
            "Cycle: pre_bucket <-> new_start via data + extra deps",
        )

    def test_graphsafe_rng_state_with_insert_overlap_deps(self):
        # [FUNCTION DOC] Checks schedule_overlap_bucketing with insert_overlap_deps on a graph
        # containing a graphsafe rng-state mm and an all_gather: after scheduling, the
        # generator placeholder's user must be control_deps, and the graph must still lower in
        # a GraphLowering run under the fake mode. Initializes CUDA.
        from torch._inductor.fx_passes.control_dependencies import control_deps
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V
        from torch._prims.rng_prims import graphsafe_run_with_rng_state

        torch.cuda.init()

        def func(a, b, rng_state):
            # [FUNCTION DOC] HELPER: Traced body: all_gather of a, an mm run with rng state
            # through graphsafe_run_with_rng_state, wait; returns the sum of the rng result
            # and gathered tensor.
            ag = torch.ops._c10d_functional.all_gather_into_tensor(a, 1, "0")
            rng = graphsafe_run_with_rng_state(
                torch.ops.aten.mm.default, a, b, rng_state=rng_state
            )
            wait = torch.ops._c10d_functional.wait_tensor(ag)
            return rng + wait

        fake_mode = FakeTensorMode()
        with fake_mode:
            a = torch.empty(4, 4, device=self.device)
            b = torch.empty(4, 4, device=self.device)
            gen = torch.cuda.default_generators[0].clone_state()
            traced = make_fx(func)(a, b, gen)

        def custom_runtime_estimation(
            node: fx.Node, override_size: int | None
        ) -> float | None:
            # [FUNCTION DOC] HELPER: Fake runtime estimator: returns 10.0 for the graphsafe
            # rng run and all_gather call nodes, otherwise None (no estimate).
            if node.op != "call_function":
                return None
            if node.target is graphsafe_run_with_rng_state:
                return 10.0
            if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
                return 10.0
            return None

        schedule_overlap_bucketing(
            traced,
            insert_overlap_deps=True,
            collective_bucketing=False,
            custom_runtime_estimation=custom_runtime_estimation,
            collective_estimator="analytical",
            compute_estimator="analytical",
            pre_bucketing_fsdp_collectives=False,
        )

        rng_placeholder = next(
            n
            for n in traced.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Generator)
        )
        user = next(iter(rng_placeholder.users))
        self.assertIs(user.target, control_deps)

        graph = GraphLowering(traced, [a, b, gen])
        with V.set_fake_mode(fake_mode), V.set_graph_handler(graph):
            graph.run(a, b, gen)


class TestForeachGroupsUnit(InductorTestCase):
    # [CLASS DOC] Test case for the foreach grouping helper used in all_gather pre-bucketing.
    """Unit tests for _compute_foreach_groups and _pre_bucket_all_gather foreach optimization."""

    @unittest.skipIf(not HAS_GPU, "Requires GPU")
    def test_foreach_groups_correctness(self):
        # [FUNCTION DOC] On GPU, builds float32/float16/float32 tensors, checks
        # _compute_foreach_groups returns a non-None grouping containing a -1 delimiter for
        # mixed dtypes, and that _pre_bucket_all_gather gives the same values with and without
        # the groups.
        """Test that foreach grouping computes correct groups and copies data correctly."""
        from torch._inductor.fx_passes.bucketing import (
            _ALL_DTYPES,
            _compute_foreach_groups,
            _pre_bucket_all_gather,
        )

        t1 = torch.randn(10, device="cuda")
        t2 = torch.randn(20, device="cuda", dtype=torch.float16)
        t3 = torch.randn(10, device="cuda")
        ag_ins = [t1, t2, t3]
        out_dtypes = [torch.float32, torch.float16, torch.float32]
        out_dtype_ints = [_ALL_DTYPES.index(d) for d in out_dtypes]

        # Mixed dtypes should produce groups with -1 delimiter
        groups = _compute_foreach_groups(ag_ins, out_dtypes)
        self.assertIsNotNone(groups)
        self.assertIn(-1, groups)

        # With and without groups should produce identical results
        result_with = _pre_bucket_all_gather(
            ag_ins, 2, torch.float32, out_dtype_ints, 0, groups
        )
        result_without = _pre_bucket_all_gather(
            ag_ins, 2, torch.float32, out_dtype_ints, 0, None
        )
        self.assertTrue(torch.allclose(result_with, result_without))


class TestNodeRuntimeEstimationUnit(InductorTestCase):
    # [CLASS DOC] Test case for node runtime estimation logging helpers.
    def test_compute_estimation_logging_handles_symbolic_scalar_meta(self):
        # [FUNCTION DOC] Builds a small graph whose placeholder has an unbacked symbolic int
        # as its value and calls _log_compute_estimations on the add node; passes if it does
        # not raise.
        from torch._inductor.fx_passes.node_runtime_estimation import (
            _log_compute_estimations,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        x.meta["val"] = torch.empty(2, 3)
        y.meta["val"] = ShapeEnv().create_unbacked_symint()
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
        add.meta["val"] = torch.empty(2, 3)
        graph.output(add)

        _log_compute_estimations([add], [1.0], [1.0])


def _make_pge_trace(
    collectives=None,
    matmuls=None,
    sdpa_ops=None,
    pg_config=None,
):
    # [FUNCTION DOC] HELPER: Builds a minimal Chrome-trace-style dict for the profile-guided
    # estimator. collectives (dicts with name, dur, optional pg_name, ranks, group_size,
    # nelems, out_nelems, dtype), matmuls (shapes, dur, optional strides, dtypes) and sdpa_ops
    # (input_dims, dur, optional op_name, strides, dtypes) become kernel events; ops get a
    # cpu_op event linked to the kernel by a running External id. Returns a dict with
    # traceEvents and distributedInfo whose pg_config is pg_config or a default group 0 over
    # ranks 0 and 1.
    """Build a minimal Chrome Trace JSON dict for PGE testing."""
    events = []
    eid = 1000

    for coll in collectives or []:
        events.append(
            {
                "cat": "kernel",
                "dur": coll["dur"],
                "name": "nccl_kernel",
                "args": {
                    "Collective name": coll["name"],
                    "Process Group Name": coll.get("pg_name", "0"),
                    "Process Group Ranks": coll.get("ranks", "[0, 1]"),
                    "Group size": coll.get("group_size", 2),
                    "In msg nelems": coll.get("nelems", 1024),
                    "Out msg nelems": coll.get("out_nelems", coll.get("nelems", 1024)),
                    "dtype": coll.get("dtype", "Float"),
                },
            }
        )

    for mm in matmuls or []:
        eid += 1
        events.append(
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "dur": 0,
                "args": {
                    "External id": eid,
                    "Input Dims": mm["shapes"],
                    "Input Strides": mm.get(
                        "strides", [[s[-1], 1] for s in mm["shapes"]]
                    ),
                    "Input type": mm.get("dtypes", ["float", "float"]),
                },
            }
        )
        events.append(
            {
                "cat": "kernel",
                "dur": mm["dur"],
                "name": "sm80_xmma_gemm",
                "args": {"External id": eid},
            }
        )

    for sdpa in sdpa_ops or []:
        eid += 1
        op_name = sdpa.get("op_name", "aten::_scaled_dot_product_flash_attention")
        events.append(
            {
                "cat": "cpu_op",
                "name": op_name,
                "dur": 0,
                "args": {
                    "External id": eid,
                    "Input Dims": sdpa["input_dims"],
                    "Input Strides": sdpa.get(
                        "input_strides", [[1] * len(d) for d in sdpa["input_dims"]]
                    ),
                    "Input type": sdpa.get("dtypes", ["c10::BFloat16"]),
                },
            }
        )
        events.append(
            {
                "cat": "kernel",
                "dur": sdpa["dur"],
                "name": "flash_fwd_kernel",
                "args": {"External id": eid},
            }
        )

    dist_info = {}
    if pg_config is not None:
        dist_info["pg_config"] = pg_config
    else:
        dist_info["pg_config"] = {"0": {"ranks": [0, 1]}}

    return {"traceEvents": events, "distributedInfo": dist_info}


def _load_pge_profile(data):
    # [FUNCTION DOC] HELPER: Writes the trace dict as JSON to a temporary file, loads it into
    # a new ProfileData and returns it; the file is always deleted afterwards.
    """Write trace dict to a temp file and load via ProfileData."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        path = f.name
    try:
        profile = ProfileData()
        profile.load(path)
        return profile
    finally:
        os.unlink(path)


class TestProfileGuidedEstimation(TestCase):
    # [CLASS DOC] Test case for ProfileData loading and lookup.
    def test_profile_loading_and_lookup(self):
        # [FUNCTION DOC] Registers a custom library op, builds a synthetic trace with two all-
        # reduce timings, one mm and one custom op, loads it, and checks lookup_collective
        # (known sizes, interpolation, other ranks, unknown collective) and lookup_op results
        # (aten mm, unknown shape gives None, custom op). Then builds a ProfileGuidedEstimator
        # from the saved trace and checks it estimates at least one node of the traced custom
        # op graph.
        """Load a trace with collectives, aten ops, and a custom op; verify lookups."""
        with torch.library._scoped_library("test_pge", "DEF") as lib:
            lib.define("my_op(Tensor x, Tensor w) -> Tensor")
            lib.impl("my_op", lambda x, w: x @ w, "CPU")
            lib.impl("my_op", lambda x, w: x @ w, "Meta")

            trace = _make_pge_trace(
                collectives=[
                    {
                        "name": "allreduce",
                        "dur": 100.0,
                        "nelems": 1000,
                        "dtype": "Float",
                        "ranks": "[0,2,4,6]",
                        "group_size": 4,
                    },
                    {
                        "name": "allreduce",
                        "dur": 800.0,
                        "nelems": 8000,
                        "dtype": "Float",
                        "ranks": "[0,2,4,6]",
                        "group_size": 4,
                    },
                ],
                matmuls=[
                    {
                        "shapes": [[128, 256], [256, 512]],
                        "dur": 50.0,
                        "dtypes": ["float", "float"],
                    }
                ],
                pg_config={"0": {"ranks": [0, 2, 4, 6]}},
            )
            # Inject a custom op event
            eid = 9000
            trace["traceEvents"].extend(
                [
                    {
                        "cat": "cpu_op",
                        "name": "test_pge::my_op",
                        "dur": 0,
                        "args": {
                            "External id": eid,
                            "Input Dims": [[32, 64], [64, 128]],
                            "Input Strides": [[64, 1], [128, 1]],
                            "Input type": ["c10::BFloat16", "c10::BFloat16"],
                        },
                    },
                    {
                        "cat": "kernel",
                        "dur": 42.0,
                        "name": "custom_kernel",
                        "args": {"External id": eid},
                    },
                ]
            )
            profile = _load_pge_profile(trace)

            # Collectives
            self.assertAlmostEqual(
                profile.lookup_collective("all_reduce", (0, 2, 4, 6), 1000, "Float")[0],
                0.1,
                places=4,
            )
            self.assertGreater(
                profile.lookup_collective("all_reduce", (0, 2, 4, 6), 4000, "Float")[0],
                0.1,
            )
            self.assertIsNotNone(
                profile.lookup_collective("all_reduce", (1, 3, 5, 7), 1000, "Float")
            )
            self.assertIsNone(
                profile.lookup_collective("all_to_all", (0, 2, 4, 6), 1000, "Float")
            )

            # Aten op
            self.assertAlmostEqual(
                profile.lookup_op(
                    "aten::mm",
                    ((128, 256), (256, 512)),
                    ((256, 1), (512, 1)),
                    torch.float32,
                ),
                0.05,
                places=4,
            )
            self.assertIsNone(
                profile.lookup_op(
                    "aten::mm", ((999,), (999,)), ((1,), (1,)), torch.float32
                )
            )

            # Custom op
            self.assertAlmostEqual(
                profile.lookup_op(
                    "test_pge::my_op",
                    ((32, 64), (64, 128)),
                    ((64, 1), (128, 1)),
                    torch.bfloat16,
                ),
                0.042,
                places=4,
            )

            # End-to-end: FX graph with custom op -> estimator match
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(trace, f)
                f.flush()
                trace_path = f.name
            try:
                estimator = ProfileGuidedEstimator(trace_path)
                with FakeTensorMode():
                    x = torch.randn(32, 64, dtype=torch.bfloat16, device="cpu")
                    w = torch.randn(64, 128, dtype=torch.bfloat16, device="cpu")
                    gm = make_fx(torch.ops.test_pge.my_op)(x, w)
                self.assertTrue(
                    any(estimator(n) is not None for n in gm.graph.nodes),
                    "Estimator should match custom op",
                )
            finally:
                os.unlink(trace_path)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestCoalescedCollectiveOverlap(InductorTestCase):
    # [CLASS DOC] Test case for overlap scheduling with coalesced collectives, whose single
    # start node returns a list and has several getitem-based waits. Fake process group world
    # size 8.
    """
    Tests for coalesced collective support in overlap scheduling.

    Coalesced collectives (e.g., reduce_scatter_tensor_coalesced) return
    Tensor[] instead of Tensor. Wait nodes follow the pattern:
      wait_tensor(getitem(coalesced_collective, idx))
    Multiple waits share the same collective start node.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 8, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=8, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def _make_coalesced_rs_graph(self, num_tensors=3):
        # [FUNCTION DOC] HELPER: Traces a graph with an mm, a reduce_scatter_tensor_coalesced
        # over flattened num_tensors inputs (8x8 ones), a wait for each output and a final mm;
        # returns the traced module.
        """Create an FX graph with a coalesced reduce_scatter and surrounding compute."""
        group_name = dist.distributed_c10d._get_default_group().group_name

        def func(*inputs):
            # [FUNCTION DOC] HELPER: Traced body: mm of first input, coalesced reduce_scatter
            # of the flattened inputs, a wait per output, another mm; returns a sum.
            # Compute before collective
            mm1 = torch.mm(inputs[0], inputs[0])

            # Coalesced reduce_scatter
            flat_inputs = [x.view(-1) for x in inputs]
            rs_outs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                flat_inputs, "sum", 8, group_name
            )

            # Wait on each output
            waited = [torch.ops._c10d_functional.wait_tensor(o) for o in rs_outs]

            # Compute after collective
            mm2 = torch.mm(inputs[0], inputs[0])

            return mm1.sum() + mm2.sum() + sum(w.sum() for w in waited)

        with FakeTensorMode():
            inputs = [torch.ones(8, 8, device=self.device) for _ in range(num_tensors)]
            traced = make_fx(func)(*inputs)

        return traced

    def test_identify_collectives_coalesced(self):
        # [FUNCTION DOC] OverlapScheduler construction on a coalesced graph (three tensors)
        # must register exactly one collective_info entry, while wait_to_start has three waits
        # all mapping to the same single start node.
        """
        _identify_collectives should register one CollectiveInfo per coalesced
        collective, not overwrite for each wait.
        """
        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        traced = self._make_coalesced_rs_graph(num_tensors=3)

        def custom_runtime(node, override_size):
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 5.0 for reduce_scatter nodes,
            # else 0.1.
            if "reduce_scatter" in str(node.target):
                return 5.0
            return 0.1

        scheduler = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        )

        # Should have exactly 1 collective (the coalesced RS), not 3
        self.assertEqual(len(scheduler.collective_info), 1)
        # All 3 waits should map to the same start
        self.assertEqual(len(scheduler.wait_to_start), 3)
        starts = set(scheduler.wait_to_start.values())
        self.assertEqual(len(starts), 1)

    def test_overlap_scheduler_coalesced_runs(self):
        # [FUNCTION DOC] Full OverlapScheduler.run on a coalesced graph (four tensors) must
        # not crash and the result still contains reduce_scatter_tensor_coalesced and wait_tensor.
        """
        Full overlap scheduler run with coalesced collectives should not crash.
        Previously crashed with assertion errors in _handle_wait and
        normalize_function failures in comm_analysis.
        """
        from torch._inductor.fx_passes.overlap_scheduling import OverlapScheduler

        traced = self._make_coalesced_rs_graph(num_tensors=4)

        def custom_runtime(node, override_size):
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 5.0 for reduce_scatter nodes,
            # else 0.1.
            if "reduce_scatter" in str(node.target):
                return 5.0
            return 0.1

        result = OverlapScheduler(
            traced,
            max_in_flight_gb=5.0,
            max_compute_pre_fetch=200,
            collective_bucketing=False,
            insert_overlap_deps=False,
            compute_overlap_multipler=1.0,
            max_coll_distance=200,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        ).run()

        # Graph should still contain the coalesced collective and waits
        graph_str = str(result.graph)
        self.assertIn("reduce_scatter_tensor_coalesced", graph_str)
        self.assertIn("wait_tensor", graph_str)

    def test_gather_node_runtime_estimations_coalesced(self):
        # [FUNCTION DOC] gather_node_runtime_estimations on a coalesced graph must yield
        # exactly one reduce_scatter estimation key rather than one per wait.
        """
        gather_node_runtime_estimations should estimate coalesced collectives
        exactly once (not once per wait).
        """
        from torch._inductor.fx_passes.overlap_scheduling import (
            gather_node_runtime_estimations,
        )

        traced = self._make_coalesced_rs_graph(num_tensors=3)

        def custom_runtime(node, override_size):
            # [FUNCTION DOC] HELPER: Fake runtime estimator: 5.0 for reduce_scatter nodes,
            # else None.
            if "reduce_scatter" in str(node.target):
                return 5.0
            return None

        estimations = gather_node_runtime_estimations(
            traced,
            custom_runtime_estimation=custom_runtime,
            collective_estimator="analytical",
        )

        # The coalesced RS node should appear exactly once in estimations
        rs_estimations = {
            k: v for k, v in estimations.items() if "reduce_scatter" in str(k.target)
        }
        self.assertEqual(len(rs_estimations), 1)

    def test_estimate_fx_collective_size_coalesced(self):
        # [FUNCTION DOC] estimate_fx_collective_size must return a positive size for a
        # coalesced reduce_scatter node whose meta value is a list of tensors.
        """
        estimate_fx_collective_size should handle coalesced collectives
        whose output meta is a list of tensors, not a single tensor.
        """
        from torch._inductor.comm_analysis import estimate_fx_collective_size

        traced = self._make_coalesced_rs_graph(num_tensors=2)

        rs_nodes = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default,
        )
        self.assertEqual(len(rs_nodes), 1)
        size = estimate_fx_collective_size(rs_nodes[0])
        # Should return non-zero for coalesced collectives
        self.assertGreater(size, 0)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestProfileGuidedEstimatorIntegration(InductorTestCase):
    # [CLASS DOC] Integration test case for ProfileGuidedEstimator called on nodes of traced
    # graphs. Fake process group world size 2.
    """Integration tests: ProfileGuidedEstimator.__call__ on traced FX graphs."""

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 2, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_estimator_call_on_fx_graph(self):
        # [FUNCTION DOC] Writes a synthetic trace matching the default group ranks (an all-
        # reduce of 200 and an mm of 50 in trace units), builds a ProfileGuidedEstimator,
        # traces an all_reduce plus mm graph and checks the estimator returns a positive
        # estimate for the collective and about 0.05 for the mm. Deletes the temp trace file.
        """ProfileGuidedEstimator returns estimates for collective and mm nodes in a traced graph."""
        group_name = dist.distributed_c10d._get_default_group().group_name
        pg_ranks = tuple(sorted(dist.get_process_group_ranks(dist.group.WORLD)))

        # Build a trace with matching collective and matmul data
        trace = _make_pge_trace(
            collectives=[
                {
                    "name": "allreduce",
                    "dur": 200.0,
                    "nelems": 256,
                    "out_nelems": 256,
                    "dtype": "Float",
                    "ranks": json.dumps(list(pg_ranks)),
                    "group_size": len(pg_ranks),
                },
            ],
            matmuls=[
                {
                    "shapes": [[16, 16], [16, 16]],
                    "dur": 50.0,
                    "dtypes": ["float", "float"],
                },
            ],
            pg_config={"0": {"ranks": list(pg_ranks)}},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(trace, f)
            f.flush()
            trace_path = f.name

        try:
            estimator = ProfileGuidedEstimator(trace_path)

            def func(a, b):
                # [FUNCTION DOC] HELPER: Traced body: all_reduce of a, wait, mm of b with
                # itself; returns their sum.
                ar = torch.ops._c10d_functional.all_reduce(a, "sum", group_name)
                ar_out = torch.ops._c10d_functional.wait_tensor(ar)
                mm = torch.mm(b, b)
                return ar_out + mm

            with FakeTensorMode():
                a = torch.randn(16, 16, device=self.device)
                b = torch.randn(16, 16, device=self.device)
                gm = make_fx(func)(a, b)

            # Call estimator on each node
            collective_hit = False
            mm_hit = False
            for node in gm.graph.nodes:
                est = estimator(node)
                if "all_reduce" in str(node.target) and "wait" not in str(node.target):
                    if est is not None:
                        collective_hit = True
                        self.assertGreater(est, 0)
                if node.target == torch.ops.aten.mm.default:
                    if est is not None:
                        mm_hit = True
                        self.assertAlmostEqual(est, 0.05, places=4)

            self.assertTrue(
                collective_hit, "Estimator should match the collective node"
            )
            self.assertTrue(mm_hit, "Estimator should match the mm node")
        finally:
            os.unlink(trace_path)


@requires_accelerator_dist_backend()
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestPreBucketingFsdpCollectives(InductorTestCase):
    # [CLASS DOC] Test case for pre-bucketing of FSDP collectives and the comm saturation
    # model. Fake process group world size 64.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setUpClass, then initializes a default
        # process group on the fake backend (rank 0, world size 64, in-memory FakeStore) and
        # sets the class attribute device to cuda.
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=64, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: calls the base tearDownClass, then destroys the
        # default process group.
        super().tearDownClass()
        dist.destroy_process_group()

    def test_saturation_model(self):
        # [FUNCTION DOC] Checks compute_min_saturation_bytes: zero for a group of one, at
        # least the interconnect profile floor for 16 ranks, strictly increasing from 16 to 64
        # to 128 ranks, and between 50 MB and 200 MB for 8 ranks (NVLink).
        """IB floor activates for small groups; NVLink uses formula; monotonic."""
        from torch._inductor.comm_analysis import (
            compute_min_saturation_bytes,
            detect_interconnect,
            INTERCONNECT_PROFILES,
            NCCL_COLL,
        )

        _MB = 1024 * 1024
        sat = compute_min_saturation_bytes

        self.assertEqual(sat(1, NCCL_COLL.ALL_GATHER), 0)

        ib_profile = INTERCONNECT_PROFILES[detect_interconnect(16)]
        self.assertGreaterEqual(
            sat(16, NCCL_COLL.ALL_GATHER), ib_profile.min_saturation_bytes
        )

        sat_64 = sat(64, NCCL_COLL.ALL_GATHER)
        sat_128 = sat(128, NCCL_COLL.ALL_GATHER)
        self.assertGreater(sat_64, sat(16, NCCL_COLL.ALL_GATHER))
        self.assertGreater(sat_128, sat_64)

        sat_nv = sat(8, NCCL_COLL.ALL_GATHER)
        self.assertGreater(sat_nv, 50 * _MB)
        self.assertLess(sat_nv, 200 * _MB)

    def test_pre_bucketing_only_merges_fsdp_collectives(self):
        # [FUNCTION DOC] pre_bucket_fsdp_collectives with a large cap merges the two
        # all_gathers on the default (FSDP) group into one while the all_gather on a different
        # TP group name is left untouched; counts by group before and after.
        """Pre-bucketing merges FSDP all-gathers but leaves TP all-gathers alone."""
        from torch._inductor.fx_passes.bucketing import is_all_gather_into_tensor
        from torch._inductor.fx_passes.fsdp import (
            _get_group_name,
            pre_bucket_fsdp_collectives,
        )

        fsdp_group = dist.distributed_c10d._get_default_group().group_name
        tp_group = "tp_test_group"

        def func(fsdp_p1, fsdp_p2, tp_a1, tp_a2):
            # [FUNCTION DOC] HELPER: Traced body: two all_gathers on the default group (size
            # 64) with waits and a third all_gather of the sum of two inputs on a TP group
            # (size 8) with wait; returns the three waited results.
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                fsdp_p1, 64, fsdp_group
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                fsdp_p2, 64, fsdp_group
            )
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)
            tp_ag = torch.ops._c10d_functional.all_gather_into_tensor(
                tp_a1 + tp_a2, 8, tp_group
            )
            tp_w = torch.ops._c10d_functional.wait_tensor(tp_ag)
            return w1, w2, tp_w

        with FakeTensorMode():
            traced = make_fx(func)(
                *(torch.ones(4, 4, device=self.device) for _ in range(4))
            )

        def count_ag(group):
            # [FUNCTION DOC] HELPER: Counts all_gather_into_tensor nodes in the traced graph
            # with the given group name.
            return sum(
                1
                for n in traced.graph.nodes
                if is_all_gather_into_tensor(n) and _get_group_name(n) == group
            )

        self.assertEqual(count_ag(fsdp_group), 2)
        self.assertEqual(count_ag(tp_group), 1)

        pre_bucket_fsdp_collectives(traced, bucket_cap_mb=2000.0)

        self.assertEqual(count_ag(fsdp_group), 1)  # 2 merged into 1
        self.assertEqual(count_ag(tp_group), 1)  # TP untouched

    def test_pre_bucketing_handles_dtype_cast(self):
        # [FUNCTION DOC] Parameters cast to bfloat16 before all_gather are still treated as
        # FSDP: pre_bucket_fsdp_collectives merges the two all_gathers into one.
        """Parameter -> dtype cast -> all_gather is identified as FSDP."""
        from torch._inductor.fx_passes.bucketing import is_all_gather_into_tensor
        from torch._inductor.fx_passes.fsdp import (
            _get_group_name,
            pre_bucket_fsdp_collectives,
        )

        fsdp_group = dist.distributed_c10d._get_default_group().group_name

        def func(p1, p2):
            # [FUNCTION DOC] HELPER: Traced body: all_gathers of two inputs cast to bfloat16
            # on the default group, waits; returns both results.
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                p1.to(torch.bfloat16), 64, fsdp_group
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                p2.to(torch.bfloat16), 64, fsdp_group
            )
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)
            return w1, w2

        with FakeTensorMode():
            traced = make_fx(func)(
                torch.ones(4, 4, device=self.device),
                torch.ones(4, 4, device=self.device),
            )

        def count_ag(group):
            # [FUNCTION DOC] HELPER: Counts all_gather_into_tensor nodes in the traced graph
            # with the given group name.
            return sum(
                1
                for n in traced.graph.nodes
                if is_all_gather_into_tensor(n) and _get_group_name(n) == group
            )

        self.assertEqual(count_ag(fsdp_group), 2)

        pre_bucket_fsdp_collectives(traced, bucket_cap_mb=2000.0)

        self.assertEqual(count_ag(fsdp_group), 1)


class TestBitsetAncestors(TestCase):
    # [CLASS DOC] Test case for BitsetAncestors (transitive ancestor index over FX nodes) on a
    # small diamond graph and a long chain.
    """Tests for BitsetAncestors -- int-bitset transitive ancestor sets."""

    def _make_graph(self):
        # [FUNCTION DOC] HELPER: Builds an FX graph with placeholder a, b=relu(a), c=neg(a),
        # d=add(b,c), e=abs(d); returns the graph and the list of nodes a to e.
        """Build a small diamond graph for testing.

        Graph topology (edges go downward):
              a
             / \\
            b   c
             \\ /
              d
              |
              e
        """
        g = fx.Graph()
        a = g.placeholder("a")
        b = g.call_function(torch.relu, (a,))
        c = g.call_function(torch.neg, (a,))
        d = g.call_function(torch.add, (b, c))
        e = g.call_function(torch.abs, (d,))
        g.output(e)
        return g, [a, b, c, d, e]

    def test_membership(self):
        # [FUNCTION DOC] is_ancestor is true along dependency paths (a to b,c,d,e; b,c to d; d
        # to e) and false for siblings b/c and for reversed direction.
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        self.assertTrue(ancestors.is_ancestor(a, b))
        self.assertTrue(ancestors.is_ancestor(a, c))
        self.assertTrue(ancestors.is_ancestor(a, d))
        self.assertTrue(ancestors.is_ancestor(b, d))
        self.assertTrue(ancestors.is_ancestor(c, d))
        self.assertTrue(ancestors.is_ancestor(a, e))
        self.assertTrue(ancestors.is_ancestor(d, e))

        self.assertFalse(ancestors.is_ancestor(b, c))
        self.assertFalse(ancestors.is_ancestor(c, b))
        self.assertFalse(ancestors.is_ancestor(d, a))
        self.assertFalse(ancestors.is_ancestor(e, a))

    def test_empty_ancestors(self):
        # [FUNCTION DOC] The root node has no ancestors: iter_ancestors on it yields nothing.
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        self.assertEqual(list(ancestors.iter_ancestors(a)), [])

    def test_iteration(self):
        # [FUNCTION DOC] iter_ancestors on d yields exactly a, b and c.
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        d_ancestors = list(ancestors.iter_ancestors(d))
        self.assertEqual(len(d_ancestors), 3)
        self.assertIn(a, d_ancestors)
        self.assertIn(b, d_ancestors)
        self.assertIn(c, d_ancestors)

    def test_len(self):
        # [FUNCTION DOC] Counts of iter_ancestors results: a has 0, b has 1, d has 3, e has 4.
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        self.assertEqual(len(list(ancestors.iter_ancestors(a))), 0)
        self.assertEqual(len(list(ancestors.iter_ancestors(b))), 1)
        self.assertEqual(len(list(ancestors.iter_ancestors(d))), 3)
        self.assertEqual(len(list(ancestors.iter_ancestors(e))), 4)

    def test_has_dep(self):
        # [FUNCTION DOC] has_dep is symmetric: true for a and d in both orders, false for the
        # unrelated b and c.
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        self.assertTrue(ancestors.has_dep(a, d))
        self.assertTrue(ancestors.has_dep(d, a))
        self.assertFalse(ancestors.has_dep(b, c))
        self.assertFalse(ancestors.has_dep(c, b))

    def test_intersection(self):
        # [FUNCTION DOC] The bitwise AND of the ancestor bit sets of d and e has three bits
        # set (a, b, c).
        """Verifies bitwise AND of ancestor sets matches expected common ancestors."""
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g, [a, b, c, d, e] = self._make_graph()
        ancestors = BitsetAncestors([a, b, c, d, e])

        common_bits = ancestors.get_ancestor_bits(d) & ancestors.get_ancestor_bits(e)
        self.assertEqual(common_bits.bit_count(), 3)

    def test_extra_inputs(self):
        # [FUNCTION DOC] Extra input edges (e depending on c) are included: c and a are both
        # ancestors of e.
        """Extra edges beyond the FX graph (e.g. hiding-interval deps)."""
        from torch._inductor.fx_passes.utils import BitsetAncestors
        from torch.utils._ordered_set import OrderedSet

        g, [a, b, c, d, e] = self._make_graph()
        extra = {e: OrderedSet([c])}
        ancestors = BitsetAncestors([a, b, c, d, e], extra_inputs=extra)

        self.assertTrue(ancestors.is_ancestor(c, e))
        self.assertTrue(ancestors.is_ancestor(a, e))

    def test_linear_chain(self):
        # [FUNCTION DOC] Builds a 100-node linear chain and checks ancestor counts are 0, 50
        # and 99 for nodes 0, 50 and 99, and is_ancestor holds forward only.
        """N-node linear chain: node i's ancestors are 0..i-1."""
        from torch._inductor.fx_passes.utils import BitsetAncestors

        g = fx.Graph()
        nodes = [g.placeholder("x")]
        for i in range(99):
            nodes.append(g.call_function(torch.relu, (nodes[-1],)))
        g.output(nodes[-1])

        ancestors = BitsetAncestors(nodes)
        self.assertEqual(len(list(ancestors.iter_ancestors(nodes[0]))), 0)
        self.assertEqual(len(list(ancestors.iter_ancestors(nodes[50]))), 50)
        self.assertEqual(len(list(ancestors.iter_ancestors(nodes[99]))), 99)
        self.assertTrue(ancestors.is_ancestor(nodes[0], nodes[99]))
        self.assertFalse(ancestors.is_ancestor(nodes[99], nodes[0]))


if __name__ == "__main__":
    run_tests()
