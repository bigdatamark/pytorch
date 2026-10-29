# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the flex attention packed mask interval selector in
# torch._inductor.kernel.flex.interval_mask_packing. Each test builds a mask_mod graph by hand
# (placeholders for batch, head, query index and key/value index, plus optional auxiliary
# inputs), runs the selector under a mock graph handler, and checks that it returns either no
# result (unsupported mask) or a list of lane intervals. Interval results are validated by
# evaluating their rendered bounds over a 32-lane group and comparing with a reference
# predicate.

from types import SimpleNamespace

import sympy

import torch
from torch._inductor.kernel.flex.interval_mask_packing import (
    select_packed_mask_intervals,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import MockGraphHandler


@instantiate_parametrized_tests
class TestIntervalMaskPacking(InductorTestCase):
    # [CLASS DOC] Parametrized tests of the packed mask interval selector. It has two reference
    # helpers (a Python predicate per named mask and an evaluator that turns rendered interval
    # bounds into a 32-bit lane mask), a helper that attaches fake tensor metadata to graph
    # nodes, one parametrized test over many mask shapes, and tests for rejection and acceptance
    # of auxiliary tensor and scalar patterns.
    def _expected_packed_mask_interval(self, case_name, q_idx, kv_idx, lane):
        # [FUNCTION DOC] HELPER: Reference predicate for the named mask cases (causal, strict
        # causal, sliding window of 256, disjoint or of two diagonals, block equality with block
        # size 16, causal bounded to rows below 100, negated causal, off-diagonal, and a
        # lane-uniform check that the query index equals 16). Given the query index, the base
        # key index and a lane offset, returns whether that lane is unmasked. Raises
        # AssertionError for an unknown case name.
        match case_name:
            case "causal":
                return q_idx >= kv_idx + lane
            case "strict_causal":
                return q_idx > kv_idx + lane
            case "sliding_window":
                return q_idx >= kv_idx + lane and q_idx - (kv_idx + lane) <= 256
            case "disjoint_or":
                return kv_idx + lane in (q_idx, q_idx + 2)
            case "block_equality":
                return q_idx // 16 == (kv_idx + lane) // 16
            case "row_bounded_causal":
                return q_idx >= kv_idx + lane and q_idx < 100
            case "not_causal":
                return not q_idx >= kv_idx + lane
            case "ne_diagonal":
                return kv_idx + lane != q_idx
            case "lane_uniform_eq":
                return q_idx == 16
            case _:
                raise AssertionError(case_name)

    def _eval_packed_mask_intervals(self, intervals, q_idx, kv_idx):
        # [FUNCTION DOC] HELPER: Convert interval objects to a 32-bit lane mask for given query
        # and key indices. Evaluates each interval's rendered lower and upper bound text with a
        # restricted evaluator that only knows max, min and the two indices, after stripping the
        # Int32 cast wrapper and the zero-subscript index notation. Sets the bit of every lane
        # in the half-open range from lower to upper and returns the combined integer.
        def eval_index(expr):
            # [FUNCTION DOC] HELPER: Evaluate one rendered bound expression string to an
            # integer. Removes the cutlass Int32 wrapper and rewrites the indexed query and key
            # names to plain variables, then evaluates with empty builtins and only max, min,
            # q_idx and kv_idx available.
            source = (
                expr.replace("cutlass.Int32", "")
                .replace("q_idx[0]", "q_idx")
                .replace("kv_idx[0]", "kv_idx")
            )
            return int(
                eval(
                    source,
                    {"__builtins__": {}},
                    {"max": max, "min": min, "q_idx": q_idx, "kv_idx": kv_idx},
                )
            )

        mask = 0
        for interval in intervals:
            lower = eval_index(interval.render_lower())
            upper = eval_index(interval.render_upper())
            for lane in range(32):
                if lower <= lane < upper:
                    mask |= 1 << lane
        return mask

    @parametrize(
        "case_name",
        [
            "causal",
            "strict_causal",
            "sliding_window",
            "disjoint_or",
            "block_equality",
            "row_bounded_causal",
            "not_causal",
            "ne_diagonal",
            "lane_uniform_eq",
            "batch_dependent",
        ],
    )
    def test_packed_mask_interval_selector(self, case_name):
        # [FUNCTION DOC] Builds an FX mask graph for the named case with the corresponding aten
        # ops (comparison, bitwise and, or, not, floor division, scalar adds), including one
        # batch dependent case. Runs the interval selector under a mock graph handler. For the
        # batch dependent case the result must be None. For all other cases the result must be
        # non-empty and, over a grid of query and key base indices, the 32-lane mask from the
        # intervals must equal the mask from the reference predicate.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        full = graph.call_function(
            torch.ops.aten.full.default,
            ([], True),
            {"dtype": torch.bool, "layout": torch.strided, "device": "cuda"},
        )
        causal = graph.call_function(torch.ops.aten.ge.Tensor, (q_idx, kv_idx))
        distance = graph.call_function(torch.ops.aten.sub.Tensor, (q_idx, kv_idx))
        match case_name:
            case "causal":
                output = causal
            case "strict_causal":
                output = graph.call_function(torch.ops.aten.gt.Tensor, (q_idx, kv_idx))
            case "sliding_window":
                in_window = graph.call_function(
                    torch.ops.aten.le.Scalar, (distance, 256)
                )
                output = graph.call_function(
                    torch.ops.aten.bitwise_and.Tensor,
                    (
                        graph.call_function(
                            torch.ops.aten.bitwise_and.Tensor, (full, in_window)
                        ),
                        causal,
                    ),
                )
            case "disjoint_or":
                q_plus_2 = graph.call_function(torch.ops.aten.add.Scalar, (q_idx, 2))
                eq_q = graph.call_function(torch.ops.aten.eq.Tensor, (kv_idx, q_idx))
                eq_q_plus_2 = graph.call_function(
                    torch.ops.aten.eq.Tensor, (kv_idx, q_plus_2)
                )
                output = graph.call_function(
                    torch.ops.aten.bitwise_or.Tensor, (eq_q, eq_q_plus_2)
                )
            case "block_equality":
                q_block = graph.call_function(
                    torch.ops.aten.div.Tensor_mode,
                    (q_idx, 16),
                    {"rounding_mode": "floor"},
                )
                kv_block = graph.call_function(
                    torch.ops.aten.div.Tensor_mode,
                    (kv_idx, 16),
                    {"rounding_mode": "floor"},
                )
                output = graph.call_function(
                    torch.ops.aten.eq.Tensor, (q_block, kv_block)
                )
            case "batch_dependent":
                output = graph.call_function(torch.ops.aten.ge.Tensor, (b, kv_idx))
            case "row_bounded_causal":
                row_bound = graph.call_function(torch.ops.aten.lt.Scalar, (q_idx, 100))
                output = graph.call_function(
                    torch.ops.aten.bitwise_and.Tensor, (causal, row_bound)
                )
            case "not_causal":
                output = graph.call_function(
                    torch.ops.aten.logical_not.default, (causal,)
                )
            case "ne_diagonal":
                output = graph.call_function(torch.ops.aten.ne.Tensor, (kv_idx, q_idx))
            case "lane_uniform_eq":
                output = graph.call_function(torch.ops.aten.eq.Scalar, (q_idx, 16))
            case _:
                raise AssertionError(case_name)
        graph.output(output)
        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))
        if case_name == "batch_dependent":
            self.assertIsNone(intervals)
            return
        self.assertIsNotNone(intervals)
        for q in (0, 15, 16, 31, 32, 127, 256):
            for kv in (0, 16, 31, 32, 64, 127, 255, 288):
                expected_mask = sum(
                    int(self._expected_packed_mask_interval(case_name, q, kv, lane))
                    << lane
                    for lane in range(32)
                )
                self.assertEqual(
                    self._eval_packed_mask_intervals(intervals, q, kv), expected_mask
                )

    def _set_node_dtype(self, node, dtype, shape=()):
        # [FUNCTION DOC] HELPER: Attach fake tensor metadata (a namespace carrying the given
        # dtype and shape, shape defaults to scalar) to the graph node, as the selector reads
        # dtype and shape from node metadata.
        node.meta["tensor_meta"] = SimpleNamespace(dtype=dtype, shape=shape)

    def test_packed_mask_interval_selector_rejects_non_affine_floor_div_lane(self):
        # [FUNCTION DOC] Mask compares a shifted key index with half of another shifted key
        # index using floor division, so the lane dependence is not affine. The selector must
        # return None.
        graph = torch.fx.Graph()
        graph.placeholder("b")
        graph.placeholder("h")
        graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        lhs = graph.call_function(torch.ops.aten.sub.Scalar, (kv_idx, 17))
        shifted_kv = graph.call_function(torch.ops.aten.add.Scalar, (kv_idx, 74))
        rhs = graph.call_function(
            torch.ops.aten.div.Tensor_mode,
            (shifted_kv, 2),
            {"rounding_mode": "floor"},
        )
        graph.output(graph.call_function(torch.ops.aten.le.Tensor, (lhs, rhs)))

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_rejects_non_integral_aux_bound(self):
        # [FUNCTION DOC] Mask compares the key index to a looked up threshold whose dtype is
        # float32 rather than integer, combined with causality. The selector must reject it and
        # return None.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        thresholds = graph.placeholder("thresholds")
        start = graph.call_function(
            torch.ops.aten.index.Tensor, (thresholds, [b, q_idx])
        )
        self._set_node_dtype(start, torch.float32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_supports_int64_aux_index_load(self):
        # [FUNCTION DOC] Mask looks up a document id from a 2D int64 tensor by batch and query
        # index, uses it to index an int32 offsets tensor, and combines with causality. The
        # selector must accept it and return exactly one interval whose lower bound text
        # references both auxiliary tensors and a conditional, and whose upper bound text
        # references the query index and a constant 1 cast.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int64, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int64)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("aux_tensors[0][b_idx[0], q_idx[0]]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())
        self.assertIn(" if ", intervals[0].render_lower())
        self.assertIn("q_idx[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(1)", intervals[0].render_upper())

    def test_packed_mask_interval_selector_rejects_interval_explosion(self):
        # [FUNCTION DOC] Mask is an or of nine equality terms against the query index shifted by
        # 0 through 8, which would need too many disjoint intervals. The selector must give up
        # and return None.
        graph = torch.fx.Graph()
        graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        terms = []
        for offset in range(9):
            shifted_q = graph.call_function(torch.ops.aten.add.Scalar, (q_idx, offset))
            terms.append(
                graph.call_function(torch.ops.aten.eq.Tensor, (kv_idx, shifted_q))
            )
        expr = terms[0]
        for term in terms[1:]:
            expr = graph.call_function(torch.ops.aten.bitwise_or.Tensor, (expr, term))
        graph.output(expr)

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_rejects_partial_aux_index_bound(self):
        # [FUNCTION DOC] Mask indexes a table by batch only, producing a row, then indexes that
        # row with a document id, so the table is only partially indexed by known bounds.
        # Combined with causality, the selector must return None.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        table = graph.placeholder("table")
        doc_ids = graph.placeholder("doc_ids")
        self._set_node_dtype(table, torch.int32, (1, 5))
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        row = graph.call_function(torch.ops.aten.index.Tensor, (table, [b]))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (row, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNone(intervals)

    def test_packed_mask_interval_selector_mixes_aux_scalars_and_tensors(self):
        # [FUNCTION DOC] Mask mixes an auxiliary scalar limit symbol (given a code string for
        # the scalar array) with the document offset lookup from auxiliary tensors. The other
        # buffers list contains the symbol and two opaque objects. Expects one interval whose
        # upper bound mentions the scalar array and whose lower bound mentions both auxiliary
        # tensors.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        limit = graph.placeholder("limit")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        below_limit = graph.call_function(torch.ops.aten.lt.Tensor, (kv_idx, limit))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, below_limit)
            )
        )

        limit_symbol = sympy.Symbol("s_limit", integer=True, nonnegative=True)
        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(
                torch.fx.GraphModule({}, graph),
                other_buffers=[limit_symbol, object(), object()],
                aux_scalar_symbol_codes={limit_symbol: "aux_scalars[0]"},
            )

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("aux_scalars[0]", intervals[0].render_upper())
        self.assertIn("aux_tensors[0]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())

    def test_packed_mask_interval_selector_aux_loaded_lower_bound(self):
        # [FUNCTION DOC] Mask whose lower bound is loaded from an auxiliary offsets tensor via a
        # document id lookup, combined with causality. Expects one interval whose lower bound
        # text contains a max of both auxiliary tensors and whose upper bound contains a min
        # with the query index and a constant 1 cast.
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        doc_ids = graph.placeholder("doc_ids")
        offsets = graph.placeholder("offsets")
        self._set_node_dtype(doc_ids, torch.int32, (1, 128))
        self._set_node_dtype(offsets, torch.int32, (5,))
        doc = graph.call_function(torch.ops.aten.index.Tensor, (doc_ids, [b, q_idx]))
        self._set_node_dtype(doc, torch.int32)
        start = graph.call_function(torch.ops.aten.index.Tensor, (offsets, [doc]))
        self._set_node_dtype(start, torch.int32)
        above_start = graph.call_function(torch.ops.aten.ge.Tensor, (kv_idx, start))
        causal = graph.call_function(torch.ops.aten.le.Tensor, (kv_idx, q_idx))
        graph.output(
            graph.call_function(
                torch.ops.aten.bitwise_and.Tensor, (above_start, causal)
            )
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("max(", intervals[0].render_lower())
        self.assertIn("aux_tensors[0]", intervals[0].render_lower())
        self.assertIn("aux_tensors[1]", intervals[0].render_lower())
        self.assertIn("min(", intervals[0].render_upper())
        self.assertIn("q_idx[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(1)", intervals[0].render_upper())

    def test_packed_mask_interval_selector_padding_mask(self):
        """Regression for attention-gym #208: BERT-style padding masks must pack.

        (q_idx < seq_lens[b]) & (kv_idx < seq_lens[b]) mixes a lane-affine kv bound
        with a lane-uniform q bound; the lane-uniform conjunct used to reject the
        whole mask into the per-lane fallback path.
        """
        graph = torch.fx.Graph()
        b = graph.placeholder("b")
        graph.placeholder("h")
        q_idx = graph.placeholder("q_idx")
        kv_idx = graph.placeholder("kv_idx")
        seq_lens = graph.placeholder("seq_lens")
        self._set_node_dtype(seq_lens, torch.int64, (8,))
        length = graph.call_function(torch.ops.aten.index.Tensor, (seq_lens, [b]))
        self._set_node_dtype(length, torch.int64)
        q_ok = graph.call_function(torch.ops.aten.lt.Tensor, (q_idx, length))
        kv_ok = graph.call_function(torch.ops.aten.lt.Tensor, (kv_idx, length))
        graph.output(
            graph.call_function(torch.ops.aten.bitwise_and.Tensor, (q_ok, kv_ok))
        )

        with V.set_graph_handler(MockGraphHandler()):
            intervals = select_packed_mask_intervals(torch.fx.GraphModule({}, graph))

        self.assertIsNotNone(intervals)
        self.assertEqual(len(intervals), 1)
        self.assertIn("min(", intervals[0].render_upper())
        self.assertIn("aux_tensors[0]", intervals[0].render_upper())
        self.assertIn("cutlass.Int32(32)", intervals[0].render_upper())


if __name__ == "__main__":
    run_tests()
