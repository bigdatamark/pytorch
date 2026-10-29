# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor fusion region detection in
# torch._inductor.fx_passes.fusion_regions (build_fusion_regions, estimate_fused_node_costs,
# is_fusible_node). Graphs are traced with make_fx under fake tensor mode on a CUDA device, so the
# whole test class is skipped when CUDA is unavailable. HAS_GPU is computed at import time.

"""Tests for fusion region detection."""

import unittest

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.ops import aten
from torch.testing._internal.common_utils import xfailIfNoAcceleratorTriton


HAS_GPU = torch.cuda.is_available()


@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestFusionRegionDetection(InductorTestCase):
    # [CLASS DOC] GPU-only tests that fusible pointwise ops are grouped into regions, that matmul
    # and fallback ops are excluded, that regions never span a non-fusible op, and that fused cost
    # estimation is consistent and leaves the graph untouched. Attribute device is set in setUp.
    """Tests for fusion region detection and grouping."""

    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, then sets the device attribute to cuda.
        super().setUp()
        self.device = "cuda"

    def test_fusion_region_grouping(self):
        # [FUNCTION DOC] Traces a chain of add, mul and sub on a fake tensor and checks the three
        # nodes map to the identical region object in the mapping from build_fusion_regions.
        """Test that connected fusible ops are grouped into regions."""
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: (a + 1) * 2 - 3)(a)

        region_of = build_fusion_regions(traced)
        (add_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.add.Tensor
        )
        (mul_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.mul.Tensor
        )
        (sub_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.sub.Tensor
        )

        # All pointwise ops should be in the same region
        self.assertIs(region_of[add_node], region_of[mul_node])
        self.assertIs(region_of[mul_node], region_of[sub_node])

    def test_mm_not_in_fusion_region(self):
        # [FUNCTION DOC] Traces pointwise ops around a matrix multiply and checks the matmul node
        # is absent from the region mapping.
        """Test that mm ops are not included in fusion regions."""
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            traced = make_fx(lambda a: torch.mm(a + 1, a) * 2)(a)

        region_of = build_fusion_regions(traced)
        (mm_node,) = traced.graph.find_nodes(op="call_function", target=aten.mm.default)
        self.assertNotIn(mm_node, region_of)

    def test_strict_local_fusion_no_cross_mm(self):
        # [FUNCTION DOC] Traces a graph where an add feeds a later div, with a matmul between them
        # in topological order. Checks both nodes are in regions but in different region objects,
        # i.e. a region is not allowed to cross a non-fusible node even if its members are data-
        # connected.
        """Test that fusion regions don't cross non-fusible (mm) boundaries.

        Pattern:
            b = a + 1
            b2 = b - 1
            c = y @ y   # mm boundary
            d = b * 2
            d2 = d / 2

        Even though b/b2 and d/d2 are connected (d uses b), they should NOT be
        in the same fusion region because mm appears between them in topological order.
        """
        from torch._inductor.fx_passes.fusion_regions import build_fusion_regions

        def model(a, y):
            # [FUNCTION DOC] HELPER: Model computing two pointwise chains separated by a matrix
            # multiply of the second input; returns the end of the first chain, the matmul and the
            # end of the second.
            b = a + 1
            b2 = b - 1
            c = y @ y  # mm boundary
            d = b * 2
            d2 = d / 2
            return b2, c, d2

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            y = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, y)

        region_of = build_fusion_regions(traced)

        (add_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.add.Tensor
        )
        (div_node,) = traced.graph.find_nodes(
            op="call_function", target=aten.div.Tensor
        )

        # add and div are connected but mm is between them - must be separate regions
        self.assertIn(add_node, region_of)
        self.assertIn(div_node, region_of)
        self.assertIsNot(
            region_of[add_node],
            region_of[div_node],
            "Ops before and after mm should not be in the same fusion region",
        )

    @xfailIfNoAcceleratorTriton
    def test_estimate_fused_node_costs(self):
        # [FUNCTION DOC] Expected-failure without an accelerator Triton. Builds regions and fused
        # costs, then checks: every region node has a cost entry, the graph gains no call_module
        # nodes, some nodes have zero cost (only internal reads and writes) and some positive, and
        # the sum of fused costs is at most the sum of per-node roofline runtime estimates.
        """Test that fused costs correctly exclude internal I/O."""
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            estimate_fused_node_costs,
        )

        def model(a, b):
            # [FUNCTION DOC] HELPER: Model with a three-node pointwise chain, a matmul, a second
            # pointwise chain on its result, and an independent third chain on the other input.
            # region 1: 3-node chain, middle node has only internal I/O
            x = a + 1  # reads a (ext), writes to neg (int)
            x = -x  # reads add (int), writes to mul (int) -> cost=0
            x = x * 2  # reads neg (int), writes to mm (ext)
            x = torch.mm(x, a)  # mm boundary
            x = (x - 1) / 2  # region 2
            y = (b + 3) * 4  # region 3 (independent)
            return x, y

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        region_of = build_fusion_regions(traced)
        fused_costs = estimate_fused_node_costs(region_of)

        # All region nodes should have a fused cost entry
        for node in region_of:
            self.assertIn(node, fused_costs)

        # Graph should NOT be mutated (no call_module nodes)
        self.assertEqual(len(list(traced.graph.find_nodes(op="call_module"))), 0)

        # Nodes with only internal I/O should have cost 0
        zero_cost_nodes = [n for n, c in fused_costs.items() if c == 0.0]
        nonzero_cost_nodes = [n for n, c in fused_costs.items() if c > 0.0]
        self.assertGreater(len(zero_cost_nodes), 0, "Should have internal-only nodes")
        self.assertGreater(len(nonzero_cost_nodes), 0, "Should have external I/O nodes")

        # Total fused cost should be less than sum of individual roofline estimates
        from torch._inductor.fx_passes.overlap_scheduling import (
            estimate_roofline_runtime_ms,
        )

        individual_total = sum(estimate_roofline_runtime_ms(n) for n in region_of)
        fused_total = sum(fused_costs.values())
        self.assertLessEqual(fused_total, individual_total)

    def test_is_fusible_node(self):
        # [FUNCTION DOC] Traces add, matmul and QR decomposition. Checks is_fusible_node is true
        # for the pointwise add, and false for the matmul (it has flop counting) and the QR op
        # (fallback kernel).
        """Test is_fusible_node correctly classifies ops."""
        from torch._inductor.fx_passes.fusion_regions import is_fusible_node

        with FakeTensorMode():
            a = torch.randn(64, 64, device=self.device)
            traced = make_fx(lambda a: torch.linalg.qr(torch.mm(a + 1, a)))(a)

        def find_node(target):
            # [FUNCTION DOC] HELPER: Returns the first call_function node in the traced graph with
            # the given target, or none if there is no such node.
            for n in traced.graph.nodes:
                if n.op == "call_function" and n.target == target:
                    return n
            return None

        # aten.add - pointwise, should be fusible
        add_node = find_node(aten.add.Tensor)
        self.assertIsNotNone(add_node)
        self.assertTrue(is_fusible_node(add_node))

        # aten.mm - has flop counter, should NOT be fusible
        mm_node = find_node(aten.mm.default)
        self.assertIsNotNone(mm_node)
        self.assertFalse(is_fusible_node(mm_node))

        # aten.linalg_qr - fallback op, should NOT be fusible
        qr_node = find_node(aten.linalg_qr.default)
        self.assertIsNotNone(qr_node)
        self.assertFalse(is_fusible_node(qr_node))

    @xfailIfNoAcceleratorTriton
    def test_fused_costs_does_not_mutate_graph(self):
        # [FUNCTION DOC] Expected-failure without an accelerator Triton. Captures the readable text
        # of the traced graph, builds regions and estimates fused costs, then checks the text is
        # unchanged.
        """estimate_fused_node_costs must not mutate the graph."""
        from torch._inductor.fx_passes.fusion_regions import (
            build_fusion_regions,
            estimate_fused_node_costs,
        )

        def model(a, b):
            # [FUNCTION DOC] HELPER: Model with pointwise ops, a matmul, more pointwise ops, and a
            # separate chain on the second input; returns the final value of the first chain and
            # both of the second.
            x = a + 1
            x = x * 2
            x = torch.mm(x, a)
            x = x - 1
            x = x / 2
            y = b + 3
            z = y * 4
            return x, y, z

        with FakeTensorMode():
            a = torch.ones(64, 64, device=self.device)
            b = torch.ones(64, 64, device=self.device)
            traced = make_fx(model)(a, b)

        graph_str_before = traced.print_readable(print_output=False)

        region_of = build_fusion_regions(traced)
        estimate_fused_node_costs(region_of)

        graph_str_after = traced.print_readable(print_output=False)
        self.assertEqual(
            graph_str_before,
            graph_str_after,
            "Graph must not be mutated by estimate_fused_node_costs",
        )

    @xfailIfNoAcceleratorTriton
    def test_fused_costs_handles_forced_bad_region(self):
        # [FUNCTION DOC] Expected-failure without an accelerator Triton. Forces one region
        # containing all the neg, relu and abs nodes of a graph even though a matmul separates them
        # (an invalid grouping), and checks estimate_fused_node_costs still returns one non-
        # negative cost per node.
        """estimate_fused_node_costs works for any region_of mapping."""
        from torch._inductor.fx_passes.fusion_regions import estimate_fused_node_costs

        with FakeTensorMode():
            t = torch.randn(64, 64, device=self.device)

            def fn(x):
                # [FUNCTION DOC] HELPER: Function applying neg, relu, a matmul, a view, then neg
                # and abs, and adding the relu result to the final value.
                a = x.neg()
                b = a.relu()
                mm_out = torch.mm(b, x)
                v = mm_out.view(64, 64)
                c = v.neg()
                d = c.abs()
                return b + d

            traced = make_fx(fn)(t)

        fusible_nodes = [
            n
            for n in traced.graph.nodes
            if n.op == "call_function"
            and n.target
            in (
                aten.neg.default,
                aten.relu.default,
                aten.abs.default,
            )
        ]

        self.assertGreaterEqual(len(fusible_nodes), 4)
        from torch.utils._ordered_set import OrderedSet

        bad_group = OrderedSet(fusible_nodes)
        region_of = dict.fromkeys(fusible_nodes, bad_group)

        costs = estimate_fused_node_costs(region_of)
        self.assertEqual(len(costs), len(fusible_nodes))
        self.assertTrue(all(c >= 0.0 for c in costs.values()))


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
