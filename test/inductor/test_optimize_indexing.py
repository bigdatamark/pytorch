# Owner(s): ["module: inductor"]
# [MODULE DOC] Unit tests for the Inductor pass that rewrites index_expr nodes in a loop body
# into value_expr nodes when their results are used as values instead of memory indices. Each
# test builds a small FX graph of ops-handler calls (index_expr, load, store, add, sort, scan,
# masked subblocks, indirect indexing), wraps it in a fake loop body object carrying value
# bounds, indexing expressions and symbol ranges, runs the conversion pass, and checks which
# nodes became value_expr and that the requested dtype is preserved. Also checks dtype deduction
# for the value_expr op. No import-time side effects besides imports.

import operator

import sympy

import torch
from torch._inductor.codegen.common import deduce_output_dtype_by_name
from torch._inductor.optimize_indexing import convert_index_expr_to_value_expr
from torch.fx import Graph
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._sympy.value_ranges import ValueRanges


class TestOptimizeIndexing(TestCase):
    # [CLASS DOC] Test case for the index_expr to value_expr conversion. Provides a static
    # factory that creates a lightweight fake loop body, and tests covering mixed index/value
    # use, unknown consumers, indirect indexing, async device asserts, sort and scan sinks,
    # masked subblocks, dtype preservation, already converted nodes and tuple getitem
    # propagation.
    @staticmethod
    def _make_loop_body(
        graph,
        bounds,
        indexing_exprs,
        replacement_vals,
        subgraphs=(),
        indirect_vars=(),
    ):
        # [FUNCTION DOC] HELPER: Static factory returning a fake loop body that exposes just
        # what the conversion pass reads: the root block holding the given graph, a mapping of
        # masked subblock names to blocks built from the optional subgraphs, the list of
        # indirect variables, the indexing expressions mapping, and a bounds accessor returning
        # a cached bounds object whose bound getter yields the given per-node value ranges and
        # whose replacement values come from the given mapping.
        class FakeBounds:
            # [CLASS DOC] HELPER: Fake bounds object. Stores the replacement values mapping
            # given by the enclosing factory and returns the enclosing per-node bounds mapping
            # from its getter method.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the closed-over replacement values mapping on
                # the instance.
                self.replacement_vals = replacement_vals

            def get_bounds(self):
                # [FUNCTION DOC] HELPER: Returns the per-node value ranges mapping captured from
                # the factory args.
                return bounds

        class FakeBlock:
            # [CLASS DOC] HELPER: Fake block that only holds the FX graph the pass walks.
            def __init__(self, graph):
                # [FUNCTION DOC] HELPER: Stores the given graph on the instance.
                self.graph = graph

        class FakeLoopBody:
            # [CLASS DOC] HELPER: Fake loop body with the attributes used by the pass: root
            # block, subblocks named by position (masked subblock followed by the index),
            # indirect variable list, indexing expressions, and a bounds object.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the root block from the factory graph, one block
                # per supplied subgraph, a copy of the indirect variables as a list, and the
                # fake bounds object.
                self.root_block = FakeBlock(graph)
                self.subblocks = {
                    f"masked_subblock{i}": FakeBlock(subgraph)
                    for i, subgraph in enumerate(subgraphs)
                }
                self.indirect_vars = list(indirect_vars)
                self.indexing_exprs = indexing_exprs
                self._bounds = FakeBounds()

            def bounds(self):
                # [FUNCTION DOC] HELPER: Returns the fake bounds object.
                return self._bounds

        return FakeLoopBody()

    def test_index_expr_mixed_use_converts_in_place(self):
        # [FUNCTION DOC] Builds a graph where one index_expr feeds both a load index and an add
        # value (whose result is stored). Checks the node is converted in place to value_expr,
        # and that both consumers still reference that same node.
        # When the same index_expr is used both as an index (load) and as a
        # value (add), we convert it in-place to value_expr. The load still
        # works because value_expr is a superset of index_expr semantics.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, index_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(load.args[2], index_expr)
        self.assertEqual(add.args[2], index_expr)

    def test_value_expr_dtype_deduction_uses_requested_dtype(self):
        # [FUNCTION DOC] Checks dtype deduction by op name returns the requested dtype (float64)
        # for the value_expr op.
        self.assertEqual(
            deduce_output_dtype_by_name("value_expr", "expr", torch.float64),
            torch.float64,
        )

    def test_index_expr_unknown_value_use_converts_conservatively(self):
        # [FUNCTION DOC] An index_expr consumed by an op the pass does not know, whose result is
        # stored, is treated as a value use: it must be converted to value_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        unknown = graph.call_method("unknown_value_op", (ops, index_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, unknown, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                unknown: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")

    def test_index_expr_load_index_stays_indexing(self):
        # [FUNCTION DOC] An index_expr used only as the index of a load keeps its op (no value
        # demand), the load still references it, and no value_expr node appears in the graph,
        # even though the load result feeds values.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, load))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                load: ValueRanges(0, 2**40),
                add: ValueRanges(0, 2**41),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")
        self.assertEqual(load.args[2], index_expr)
        self.assertEqual([], [n for n in graph.nodes if n.target == "value_expr"])

    def test_index_expr_set_indirect_stays_indexing(self):
        # [FUNCTION DOC] An index_expr whose result only feeds the indirect index setter (for a
        # load index expression referencing the indirect symbol) stays an index_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        graph.call_module("set_indirect0", (index_expr,))
        load_index = graph.call_module("get_index", ("i1",))
        load = graph.call_method("load", (ops, "arg0", load_index))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, load, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        indirect0 = sympy.Symbol("indirect0", integer=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
            },
            {"i0": i0, "i1": indirect0},
            {i0: ValueRanges(0, 1)},
            indirect_vars=(indirect0,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_set_indirect_value_use_converts(self):
        # [FUNCTION DOC] Indirect indexing where the stored value comes from a second index_expr
        # whose indexing expression references the indirect symbol: following that value use
        # back to the indirect setter input converts both index_expr nodes (the setter input and
        # the value one) to value_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        set_input = graph.call_method("index_expr", (ops, get_index, torch.int64))
        graph.call_module("set_indirect0", (set_input,))
        value_get_index = graph.call_module("get_index", ("i1",))
        value_index_expr = graph.call_method(
            "index_expr", (ops, value_get_index, torch.int64)
        )
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method(
            "store", (ops, "buf0", store_index, value_index_expr, None)
        )
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        indirect0 = sympy.Symbol("indirect0", integer=True)
        loop_body = self._make_loop_body(
            graph,
            {
                set_input: ValueRanges(0, 2**40),
                value_index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0, "i1": indirect0 + 1},
            {i0: ValueRanges(0, 1)},
            indirect_vars=(indirect0,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(set_input.target, "value_expr")
        self.assertEqual(value_index_expr.target, "value_expr")

    def test_index_expr_device_assert_async_stays_indexing(self):
        # [FUNCTION DOC] An index_expr used only in a comparison that feeds an async device
        # assert (with a store of a constant alongside) is not a value use: it stays an
        # index_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        cond = graph.call_method("lt", (ops, index_expr, 16))
        assert_result = graph.call_method("device_assert_async", (ops, cond, "msg"))
        store_index = graph.call_module("get_index", ("i0",))
        store_value = graph.call_method("constant", (ops, 0, torch.int64))
        store = graph.call_method(
            "store", (ops, "buf0", store_index, store_value, None)
        )
        graph.output((assert_result, store))

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**40),
                cond: ValueRanges.unknown(),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_sort_and_scan_value_sinks(self):
        # [FUNCTION DOC] index_expr nodes passed as inputs of a sort op and of a scan submodule
        # are value sinks: both get converted to value_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        sort_index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        scan_index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        sort = graph.call_method(
            "sort",
            (ops, (torch.int64,), (sort_index_expr,), False, False),
        )
        scan = graph.call_module(
            "scan0",
            ((torch.int64,), (scan_index_expr,)),
        )
        graph.output((sort, scan))

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                sort_index_expr: ValueRanges(0, 2**40),
                scan_index_expr: ValueRanges(0, 2**40),
                sort: ValueRanges(0, 2**40),
                scan: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(sort.args[2][0].target, "value_expr")
        self.assertEqual(scan.args[1][0].target, "value_expr")

    def test_index_expr_masked_subblock_value_use(self):
        # [FUNCTION DOC] An outer index_expr passed to a masked subblock whose output is a graph
        # output, with the subblock itself containing an index_expr that is returned, are both
        # value uses: the outer argument and the inner node are converted to value_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        graph.output(masked)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(masked.args[1].target, "value_expr")
        self.assertEqual(index_expr.target, "value_expr")

    def test_index_expr_float_value_use_preserves_requested_dtype(self):
        # [FUNCTION DOC] An index_expr requesting float32 that is stored (with a large strided
        # indexing expression) is converted to value_expr and its dtype argument remains
        # float32.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.float32))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, index_expr, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 2**31),
            },
            {"i0": 2147483648 * i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.float32)

    def test_index_expr_value_use_preserves_requested_dtype(self):
        # [FUNCTION DOC] An int64 index_expr that is stored is converted to value_expr and its
        # dtype argument remains int64.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, index_expr, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.int64)

    def test_existing_value_expr_dtype_is_not_rewritten(self):
        # [FUNCTION DOC] A pre-existing value_expr node is left unchanged in target and dtype,
        # and a separate index_expr that is used only as a load index remains an index_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        value_expr = graph.call_method("value_expr", (ops, get_index, torch.int64))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.int64))
        load = graph.call_method("load", (ops, "arg0", index_expr))
        add = graph.call_method("add", (ops, load, value_expr))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                value_expr: ValueRanges(0, 1),
                index_expr: ValueRanges(0, 1),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(value_expr.target, "value_expr")
        self.assertEqual(value_expr.args[2], torch.int64)
        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_getitem_value_use_propagates_to_tuple_source(self):
        # [FUNCTION DOC] A float32 index_expr consumed by a frexp op whose first tuple element
        # is stored through a getitem: the value demand propagates through the getitem to the
        # tuple producer's input, so the index_expr is converted with dtype float32 kept.
        graph = Graph()
        ops = graph.placeholder("ops")
        get_index = graph.call_module("get_index", ("i0",))
        index_expr = graph.call_method("index_expr", (ops, get_index, torch.float32))
        frexp = graph.call_method("frexp", (ops, index_expr))
        getitem = graph.call_function(operator.getitem, (frexp, 0))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, getitem, None))
        graph.output(store)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                index_expr: ValueRanges(0, 1),
                frexp: ValueRanges(0, 1),
                getitem: ValueRanges(0, 1),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(index_expr.target, "value_expr")
        self.assertEqual(index_expr.args[2], torch.float32)

    def test_index_expr_masked_subblock_index_use_stays_indexing(self):
        # [FUNCTION DOC] An outer index_expr goes into a masked subblock whose result is only
        # used as a load index, and the subblock contains its own index_expr: neither is a value
        # use, both stay index_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        load = graph.call_method("load", (ops, "arg0", masked))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, load, None))
        graph.output(store)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                masked: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        self.assertEqual(other_index_expr.target, "index_expr")
        self.assertEqual(index_expr.target, "index_expr")

    def test_index_expr_masked_subblock_mixed_use_converts_in_place(self):
        # [FUNCTION DOC] Like the previous masked subblock test, but the subblock result is also
        # added to the load result (a value use). The pass does not clone, so the single value
        # use converts both the outer node feeding the subblock and the inner index_expr to
        # value_expr.
        graph = Graph()
        ops = graph.placeholder("ops")
        mask = graph.placeholder("mask")
        other_get_index = graph.call_module("get_index", ("i0",))
        other_index_expr = graph.call_method(
            "index_expr", (ops, other_get_index, torch.int64)
        )
        masked = graph.call_module("masked_subblock0", (mask, other_index_expr))
        load = graph.call_method("load", (ops, "arg0", masked))
        add = graph.call_method("add", (ops, load, masked))
        store_index = graph.call_module("get_index", ("i0",))
        store = graph.call_method("store", (ops, "buf0", store_index, add, None))
        graph.output(store)

        subgraph = Graph()
        sub_ops = subgraph.placeholder("ops")
        get_index = subgraph.call_module("get_index", ("i0",))
        index_expr = subgraph.call_method(
            "index_expr", (sub_ops, get_index, torch.int64)
        )
        subgraph.output(index_expr)

        i0 = sympy.Symbol("i0", integer=True, nonnegative=True)
        loop_body = self._make_loop_body(
            graph,
            {
                other_index_expr: ValueRanges(0, 2**40),
                masked: ValueRanges(0, 2**40),
                load: ValueRanges(0, 1),
                add: ValueRanges(0, 2**40),
                index_expr: ValueRanges(0, 2**40),
            },
            {"i0": i0},
            {i0: ValueRanges(0, 1)},
            subgraphs=(subgraph,),
        )

        convert_index_expr_to_value_expr(loop_body)

        # The subblock result has both indexing and value uses. We do not clone,
        # so the value use makes both paths use value_expr.
        self.assertEqual(masked.args[1].target, "value_expr")
        self.assertEqual(index_expr.target, "value_expr")


if __name__ == "__main__":
    run_tests()
