# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for graph breaks inside list/dict comprehensions on
# Python 3.12+, where comprehensions are inlined into the parent bytecode. Defines
# the count_op helper plus ComprehensionTests (breaks in the compiled frame) and
# NestedGraphBreakTests (breaks in an inlined callee). Importing has no side effects;
# some tests set the module global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST. A __main__
# guard runs the suite through torch._dynamo.test_case.run_tests.

"""
Tests for comprehension graph break handling

In Python 3.12+, comprehensions are inlined into the parent function's bytecode
rather than being compiled as separate code objects. These tests verify
that handling is the same as in previous Python versions, i.e.,
when a graph break occurs inside a comprehension, only the comprehension is
skipped (not the entire frame), resulting in multiple graphs being created.
"""

import operator

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import skipIfNotPy312


def count_op(graph, op):
    # [FUNCTION DOC] HELPER: Count the nodes in a recorded graph whose target equals op.
    # Takes a GraphModule (walks its inner graph's node list) and an operation such
    # as operator.add; returns the integer count. Pure, no side effects.
    """Count occurrences of a specific operation in the graph."""
    return sum(1 for node in graph.graph.nodes if node.target == op)


@skipIfNotPy312
class ComprehensionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo tests (Python 3.12+ only, via skipIfNotPy312) for graph breaks
    # raised inside comprehensions. Each test compiles a function with the
    # EagerAndRecordGraphs backend and checks the result matches eager, how many
    # graphs were recorded, and how many add nodes each graph holds. Covers list and
    # dict comprehensions, walrus targets, closures, globals, outer-state mutation,
    # nested inlined calls, and cell/locals() handling.
    def test_list_comprehension_graph_break(self):
        # [FUNCTION DOC] A list comprehension whose element expression calls graph_break
        # splits the frame into two graphs: results equal eager, two graphs recorded,
        # one add node in each (the adds before and after the comprehension).
        """Test that list comprehension with graph break creates 2 graphs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: adds 1 to x, builds a list comprehension over
            # three ints with a graph break per element, adds 2 to x; returns all three.
            y = x + 1
            result = [torch._dynamo.graph_break() or i for i in range(3)]
            z = x + 2
            return y, result, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_dict_comprehension_graph_break(self):
        # [FUNCTION DOC] Same as the list case but for a dict comprehension: output matches
        # eager, two graphs, one add node in each.
        """Test that dict comprehension with graph break creates 2 graphs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, dict comprehension (graph break per
            # item, value is the square), add 2; returns the two tensors and the dict.
            y = x + 1
            result = {i: torch._dynamo.graph_break() or i**2 for i in range(3)}
            z = x + 2
            return y, result, z

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_list_comprehension_with_graph_break_function(self):
        # [FUNCTION DOC] The graph break comes from a called function, not the comprehension
        # body itself. Expects eager-equal results, two graphs, one multiply node in
        # the first graph and one add node in the second.
        """Test list comprehension calling a function that causes graph break."""

        def inner(i):
            # [FUNCTION DOC] HELPER: Returns its argument unchanged, but calls graph_break first
            # when the argument equals 3.
            if i == 3:
                torch._dynamo.graph_break()
            return i

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: multiplies x by 2, builds a list by calling
            # inner over five ints (break at 3), adds x and the list sum; returns (z, y).
            y = x * 2
            lst = [inner(i) for i in range(5)]
            z = x + sum(lst)
            return z, y

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.mul), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_simple_list_comprehension_no_break(self):
        # [FUNCTION DOC] Control case: a comprehension with no graph break stays in one
        # graph. Checks eager-equal result, one graph, and three add nodes.
        """Test that simple list comprehension without graph break creates 1 graph."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: adds a constant tensor to x, builds a list of
            # doubled ints, returns x plus y plus the list sum (three adds).
            y = x + torch.tensor([1, 2, 3, 4])
            result = [i * 2 for i in range(3)]
            return x + y + sum(result)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)

    def test_multiple_comprehensions_one_break(self):
        # [FUNCTION DOC] Only one of three comprehensions contains a graph break, so only
        # that one splits the frame: two graphs with one add node each.
        """Test function with multiple comprehensions where only one has graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, a plain comprehension, a breaking
            # comprehension, add 2, another plain comprehension; returns all five values.
            a = x + 1
            list1 = [i for i in range(2)]  # noqa: C416
            list2 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            list3 = [i * 2 for i in range(2)]
            return a, list1, list2, b, list3

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_inner_break(self):
        # [FUNCTION DOC] Graph break in the inner comprehension of a nested list
        # comprehension: eager-equal result, two graphs, one add node in each.
        """Test nested comprehension where inner comprehension causes graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, nested list comprehension whose inner
            # element expression breaks the graph, add 2.
            a = x + 1
            result = [
                [torch._dynamo.graph_break() or i * j for j in range(2)]
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multi_iterator_comprehension_break(self):
        # [FUNCTION DOC] Comprehension with two for clauses (two iterators) and a break in
        # the element expression: two graphs, one add node each, eager-equal output.
        """Test comprehension with multiple iterators where graph break occurs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension of pairs over two
            # ranges with a graph break in the first element, add 2.
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or i, j)
                for i in range(2)
                for j in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_discarded_comprehension_graph_break(self):
        # [FUNCTION DOC] A comprehension used as a bare statement (result discarded) that
        # breaks the graph still gives two graphs with one add node each.
        """Test that discarded comprehension (result not assigned) with graph break works."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, discarded breaking comprehension,
            # add 2; returns the two tensors.
            a = x + 1
            [torch._dynamo.graph_break() or i for i in range(3)]
            b = x + 2
            return a, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_in_expression_graph_break(self):
        # [FUNCTION DOC] Comprehension nested inside a sum call that breaks the graph:
        # eager-equal result, two graphs, one add node each.
        """Test comprehension used in expression (e.g., sum([...])) with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, sum over a breaking comprehension, add
            # 2; returns the tensors and the integer total.
            a = x + 1
            total = sum([torch._dynamo.graph_break() or i for i in range(3)])
            b = x + 2
            return a, total, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_return_directly(self):
        # [FUNCTION DOC] Comprehension returned directly from the frame. Expects a single
        # recorded graph holding the one add that precedes the comprehension.
        """Test comprehension returned directly with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: computes an unused add of 1, then returns a
            # breaking list comprehension.
            a = x + 1  # noqa: F841
            return [torch._dynamo.graph_break() or i for i in range(3)]

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)

    def test_walrus_operator_in_comprehension(self):
        # [FUNCTION DOC] Walrus assignment in the element expression of a breaking
        # comprehension; the bound name y must remain visible after the comprehension.
        # Two graphs, one add node each, eager-equal output.
        """Test walrus operator (:=) inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension binding y to twice the
            # loop value after a graph break, add 2; returns tensors, list and y.
            a = x + 1
            result = [(y := (torch._dynamo.graph_break() or i * 2)) for i in range(3)]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_walrus_operator_in_if_in_comprehension(self):
        # [FUNCTION DOC] Walrus assignment inside the if clause of a breaking comprehension
        # (only values above 2 pass the filter); y must leak to the enclosing frame.
        # Two graphs, one add node each.
        """Test walrus operator (:=) in if clause of comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, filtered comprehension over five ints
            # that binds y in the filter and breaks in the element, add 2.
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or y) for i in range(5) if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_walrus_operator_in_comprehension_with_tensor(self):
        # [FUNCTION DOC] Like the walrus-in-filter test, but the element expression also
        # reads x.numel(). Two graphs, one add node each, eager-equal output.
        """Test walrus operator with tensor operation in comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, filtered comprehension that breaks and
            # adds numel of x to y, add 2; returns tensors, list and y.
            a = x + 1
            result = [
                (torch._dynamo.graph_break() or y + x.numel())
                for i in range(5)
                if (y := i * 2) > 2
            ]
            b = x + 2
            return a, result, y, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multiple_walrus_operators_in_comprehension(self):
        # [FUNCTION DOC] Two walrus targets (y and z) bound inside one breaking
        # comprehension; both must be visible afterwards. Two graphs, one add each.
        """Test multiple walrus operators (:=) inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension of pairs binding y after
            # a break and z directly, add 2; returns tensors, list, y and z.
            a = x + 1
            result = [
                ((y := (torch._dynamo.graph_break() or i * 2)), (z := i * 3))
                for i in range(3)
            ]
            b = x + 2
            return a, result, y, z, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_with_walrus_operators(self):
        # [FUNCTION DOC] Walrus targets in both the outer comprehension (outer) and the
        # inner breaking comprehension (inner); both names must survive. Two graphs,
        # one add node each.
        """Test nested comprehension with walrus operators in both inner and outer."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, outer comprehension building tuples of
            # a walrus value and an inner breaking comprehension, add 2.
            a = x + 1
            result = [
                (
                    outer := i * 10,
                    [
                        (inner := (torch._dynamo.graph_break() or j * 2))
                        for j in range(2)
                    ],
                )
                for i in range(3)
            ]
            b = x + 2
            return a, result, outer, inner, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_with_captured_outer_variable(self):
        # [FUNCTION DOC] Inner comprehension reads a variable (outer_val) defined outside
        # the outer comprehension, with a break in its filter. Two graphs, one add each.
        """Test nested comprehension with captured outer variable in inner loop."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, nested comprehension adding outer_val
            # to each j with a break in the filter, add 2.
            outer_val = 100
            a = x + 1
            result = [
                [outer_val + j for j in range(2) if torch._dynamo.graph_break() or True]
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_triple_nested_comprehension_with_walrus(self):
        # [FUNCTION DOC] Three levels of comprehension, the innermost with a walrus target
        # (w) and a graph break in its filter. Two graphs, one add node each.
        """Test triple-nested comprehension with walrus at middle level."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, triple nested comprehension binding w
            # to i plus j plus k, add 2; returns tensors, list and w.
            a = x + 1
            result = [
                [
                    [
                        (w := i + j + k)
                        for k in range(2)
                        if torch._dynamo.graph_break() or True
                    ]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            b = x + 2
            return a, result, w, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_dict_comprehension_with_walrus_and_captured(self):
        # [FUNCTION DOC] Nested dict comprehensions where the inner one binds inner_val by
        # walrus, reads the captured multiplier, and breaks in its filter. Two graphs,
        # one add node each.
        """Test nested dict comprehension with walrus in inner and captured variable."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, dict of dicts using multiplier and a
            # walrus, add 2; returns tensors, the dict and inner_val.
            multiplier = 10
            a = x + 1
            result = {
                i: {
                    j: (inner_val := j * multiplier)
                    for j in range(2)
                    if torch._dynamo.graph_break() or True
                }
                for i in range(3)
            }
            b = x + 2
            return a, result, inner_val, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_variable_read(self):
        # [FUNCTION DOC] Breaking comprehension reads a local (outer_val) of the enclosing
        # frame. Two graphs, one add node each, eager-equal output.
        """Test accessing outer variable inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, filtered comprehension adding outer_val
            # to each item, add 2.
            outer_val = 10
            a = x + 1
            result = [
                i + outer_val for i in range(3) if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_mutation(self):
        # [FUNCTION DOC] Breaking comprehension appends to a list of the enclosing frame;
        # the mutation must be visible in the returned list. Two graphs, one add each.
        """Test mutating outer list inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension appending each item to
            # outer_list, add 2; returns tensors, result and outer_list.
            outer_list = []
            a = x + 1
            result = [
                outer_list.append(i) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_dict_mutation(self):
        # [FUNCTION DOC] Breaking comprehension updates a dict of the enclosing frame; the
        # mutation must be visible afterwards. Two graphs, one add node each.
        """Test mutating outer dict inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension updating outer_dict with
            # item to ten times item, add 2; returns outer_dict too.
            outer_dict = {}
            a = x + 1
            result = [
                outer_dict.update({i: i * 10}) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_dict, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_extend(self):
        # [FUNCTION DOC] Breaking comprehension extends an outer list (initially holding
        # 100) with two values per item. Two graphs, one add node each.
        """Test extending outer list inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension extending outer_list,
            # add 2; returns tensors, result and outer_list.
            outer_list = [100]
            a = x + 1
            result = [
                outer_list.extend([i, i * 10]) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_outer_list_pop(self):
        # [FUNCTION DOC] Breaking comprehension pops from an outer list and records the
        # popped values in a second list; both must match eager. Two graphs, one add each.
        """Test popping from outer list inside comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension popping outer_list into
            # popped_values, add 2; returns both lists.
            outer_list = [10, 20, 30, 40, 50]
            popped_values = []
            a = x + 1
            result = [
                popped_values.append(outer_list.pop()) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, outer_list, popped_values, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_global_variable(self):
        # [FUNCTION DOC] Sets the module global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST to 100
        # (side effect), then checks a breaking comprehension that reads it. Two graphs,
        # one add node each.
        """Test global variable access inside comprehension with graph break."""

        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, filtered comprehension adding the
            # global to each item, add 2.
            a = x + 1
            result = [
                i + _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_closure_variable(self):
        # [FUNCTION DOC] Breaking comprehension reads a closure variable of the compiled
        # function. Two graphs, one add node each, eager-equal output.
        """Test closure variable access inside comprehension with graph break."""

        def make_fn():
            # [FUNCTION DOC] HELPER: Factory: defines closure_val (50) and returns the inner
            # function that closes over it.
            closure_val = 50

            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled body: add 1, filtered comprehension adding the
                # closure value, add 2.
                a = x + 1
                result = [
                    i + closure_val
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 2
                return a, result, b

            return fn

        fn = make_fn()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_with_closure_list_mutation(self):
        # [FUNCTION DOC] Breaking comprehension appends to a closure list. Because the
        # closure list is mutated by each run, the reference result comes from a freshly
        # built second function. Two graphs, one add node each.
        """Test closure list mutation inside comprehension with graph break."""

        def make_fn():
            # [FUNCTION DOC] HELPER: Factory: creates an empty closure_list and returns a
            # function closing over it.
            closure_list = []

            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled body: add 1, comprehension appending to
                # closure_list, add 2; returns a copy of the list.
                a = x + 1
                result = [
                    closure_list.append(i) or i * 2
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 2
                return a, result, closure_list.copy(), b

            return fn

        fn = make_fn()
        fn_ref = make_fn()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_multi_for_comprehension_graph_break(self):
        # [FUNCTION DOC] Inner comprehension breaks while the outer comprehension has two
        # for clauses. Two graphs, one add node each, eager-equal output.
        """Test nested comprehension with multiple for loops and graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, list of lists over two outer ranges and
            # an inner range with a break per element, add 2.
            a = x + 1
            result = [
                [(torch._dynamo.graph_break() or i + j + k) for k in range(2)]
                for i in range(2)
                for j in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_multiple_comprehension_graph_breaks(self):
        # [FUNCTION DOC] Two separate breaking comprehensions split the frame into three
        # graphs, one add node in each.
        """Test multiple comprehensions with graph breaks producing 3 graphs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, breaking list comprehension, add 2,
            # second breaking comprehension, add 3.
            a = x + 1
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            list2 = [torch._dynamo.graph_break() or i * 2 for i in range(2)]
            c = x + 3
            return a, list1, b, list2, c

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    def test_comprehension_modifying_global_variable(self):
        # [FUNCTION DOC] The function increments a global before a breaking comprehension
        # that reads it. The global is reset before each of the compiled and eager runs
        # and at the end (side effect on module state). Two graphs, one add each.
        """Test modifying a global variable inside comprehension with graph break."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100
        original_value = _GLOBAL_VALUE_FOR_COMPREHENSION_TEST

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, increment the global by one, breaking
            # comprehension adding the global to each item, add 2.
            global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
            a = x + 1
            _GLOBAL_VALUE_FOR_COMPREHENSION_TEST += 1
            result = [
                torch._dynamo.graph_break() or _GLOBAL_VALUE_FOR_COMPREHENSION_TEST + i
                for i in range(3)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        compiled_result = compiled(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        expected_result = fn(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value

        self.assertEqual(compiled_result, expected_result)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_comprehension_modifying_closure_variable(self):
        # [FUNCTION DOC] The function mutates a closure list cell before a breaking
        # comprehension that reads it; the reference run uses a freshly built function.
        # Two graphs, one add node each.
        """Test modifying a closure variable inside comprehension with graph break."""

        def make_fn():
            # [FUNCTION DOC] HELPER: Factory: creates closure_val as a one-element list and
            # returns a function closing over it.
            closure_val = [0]

            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled body: add 1, increment closure_val's element,
                # breaking comprehension using it, add 2.
                a = x + 1
                closure_val[0] += 1
                result = [
                    torch._dynamo.graph_break() or closure_val[0] + i for i in range(3)
                ]
                b = x + 2
                return a, result, closure_val[0], b

            return fn

        fn = make_fn()
        fn_ref = make_fn()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_list_and_dict_comprehension_graph_breaks(self):
        # [FUNCTION DOC] A breaking list comprehension followed by a breaking dict
        # comprehension gives three graphs with one add node each.
        """Test list and dict comprehensions with graph breaks together."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, list comprehension, add 2, dict
            # comprehension, add 3 (each comprehension breaks the graph).
            a = x + 1
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 2
            dict1 = {i: torch._dynamo.graph_break() or i * 10 for i in range(2)}
            c = x + 3
            return a, list1, b, dict1, c

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    def test_nested_dict_in_list_comprehension_graph_break(self):
        # [FUNCTION DOC] A dict comprehension nested in a list comprehension with a break in
        # the dict value. Two graphs, one add node each.
        """Test dict comprehension nested in list comprehension with graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: add 1, list of dicts keyed by j with a graph
            # break in each value, add 2.
            a = x + 1
            result = [
                {j: torch._dynamo.graph_break() or i * j for j in range(2)}
                for i in range(2)
            ]
            b = x + 2
            return a, result, b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_function_calls_with_comprehension_graph_break(self):
        # [FUNCTION DOC] A breaking comprehension three calls deep (f calls g calls h). Two
        # graphs are expected, each with three add nodes (adds before and after the break
        # across the inlined frames).
        """Test nested function calls where inner function has comprehension with graph break."""

        def h(x):
            # [FUNCTION DOC] HELPER: Innermost function: add 3, discarded breaking comprehension,
            # add 4; returns x.
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 4
            return x

        def g(x):
            # [FUNCTION DOC] HELPER: Middle function: add 2, call h, add 5.
            x = x + 2
            x = h(x)
            x = x + 5
            return x

        def f(x):
            # [FUNCTION DOC] HELPER: Compiled entry point: add 1, call g, add 6.
            x = x + 1
            x = g(x)
            x = x + 6
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 2)
        for i in range(2):
            self.assertEqual(count_op(backend.graphs[i], operator.add), 3)

    def test_reuse_name(self):
        # [FUNCTION DOC] Comprehension target name i is also the assigned variable. With a
        # graph break in the element the whole frame is expected to produce one graph
        # and match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: assigns i from a comprehension of pairs (loop
            # value plus x, graph_break result), returns i and the sine of x.
            i = [(i + x, torch._dynamo.graph_break()) for i in range(5)]
            return i, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_store_multiple(self):
        # [FUNCTION DOC] Unpacks a breaking comprehension of three items into i, j, k.
        # Expects eager-equal output and one recorded graph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: unpack three pairs from a comprehension into
            # i, j, k; return them with the sine of x.
            i, j, k = [(i + x, torch._dynamo.graph_break()) for i in range(3)]
            return i, j, k, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_store_nonlocal(self):
        # [FUNCTION DOC] Assigns a breaking comprehension result to a nonlocal variable i.
        # Checks eager-equal result, that the closure cell i holds the same value as the
        # returned list, and one recorded graph. Mutates the enclosing local i.
        i = 0

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: declares i nonlocal, assigns it a
            # comprehension of pairs (also named i as loop target), returns i and sine of x.
            nonlocal i
            i = [(i + x, torch._dynamo.graph_break()) for i in range(2)]
            return i, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        res = fn(x)
        self.assertEqual(res, opt_fn(x))
        self.assertEqual(i, res[0])
        self.assertEqual(len(backend.graphs), 1)

    def test_reconstruct_cell(self):
        # [FUNCTION DOC] i is a cell variable of the compiled function (captured by g)
        # and also the comprehension loop target; the frame must be reconstructed
        # correctly after the break. Compares to eager with the eager backend only.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: sets i to 100, defines g, builds a list via
            # a breaking comprehension over i, returns (list, g(), i).
            i = 100

            def g():
                # [FUNCTION DOC] HELPER: Closure returning i; exists only to make i a cell var.
                return i  # makes `i` a cellvar of fn

            r = [(i + x, torch._dynamo.graph_break())[0] for i in range(2)]
            return (r, g(), i)

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        res = fn(x)
        self.assertEqual(res, opt_fn(x))

    def test_store_nonlocal_lambda_capture(self):
        # [FUNCTION DOC] i is simultaneously a fast local, a comprehension cell and a free
        # variable of the compiled function. Compiling normally must match eager;
        # fullgraph compile must raise Unsupported mentioning "MAKE_CELL on a free
        # variable name". Resets dynamo and the nonlocal i between runs.
        # `i` is a fast local, a comprehension cell, and a free var at once;
        # Dynamo cannot model two same-named cells and must fall back
        # (gb_type "MAKE_CELL on a free variable name") rather than silently
        # capture the wrong cell.
        i = 0

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: nonlocal i is assigned a list of lambdas
            # capturing the loop value; returns whether each lambda result is the list and x+1.
            nonlocal i
            i = [lambda: i for i in range(2)]
            return [f() is i for f in i], x + 1

        x = torch.randn(3, 3)
        res = fn(x)
        i = 0
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(res, opt_fn(x))

        i = 0
        torch._dynamo.reset()
        full_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "MAKE_CELL on a free variable name"
        ):
            full_fn(x)

    def test_locals_snapshot_colliding_cell(self):
        # [FUNCTION DOC] For both locals() and vars() (subTest each), checks the frame
        # locals snapshot when i is both a fast local and free variable and an unused
        # argument is pruned from tracked locals. Compares to eager and checks the
        # nonlocal i ends as the list of 1 and 2.
        # `i` is both a fast local and a free var, so it is registered in
        # symbolic_cellvars; `unused` is never read, so it is pruned from
        # symbolic_locals and must be rebuilt from f_locals on demand. Both
        # paths of _call_frame_locals_snapshot run in one frame here.
        for snapshot in (locals, vars):
            with self.subTest(snapshot=snapshot.__name__):
                i = 0

                def fn(x, unused):
                    # [FUNCTION DOC] HELPER: Compiled body: nonlocal i gets a comprehension, then
                    # returns the sorted items of the snapshot() mapping and x+1.
                    nonlocal i
                    i = [i + 1 for i in range(2)]
                    return sorted(snapshot().items(), key=str), x + 1

                x = torch.randn(3, 3)
                res = fn(x, 7)
                i = 0
                torch._dynamo.reset()
                opt_fn = torch.compile(fn, backend="eager")
                self.assertEqual(res, opt_fn(x, 7))
                self.assertEqual(i, [1, 2])

    def test_locals_snapshot_cell_shadowed_by_itervar(self):
        # [FUNCTION DOC] The comprehension iteration variable shadows the cell i; locals()
        # inside must report the iteration value, not the cell's 100. Eager-equal.
        # Inside the comprehension the fast local `i` holds the iteration
        # value while the cell still holds 100. They share a name but not a
        # localsplus slot, and CPython reports the fast local, so the
        # snapshot must let it shadow the cell.
        i = 100

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: nonlocal i; returns a comprehension of
            # locals-lookups of i over a range of two, and x+1.
            nonlocal i
            return [locals()["i"] for i in range(2)], x + 1

        x = torch.randn(3, 3)
        res = fn(x)
        i = 100
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(res, opt_fn(x))

    def test_store_global(self):
        # [FUNCTION DOC] Stores a breaking comprehension result into the module global
        # _comprehension_global (side effect). Checks eager-equal output, the global
        # equals the returned list, and one graph.
        global _comprehension_global

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: assigns the global from a comprehension of
            # pairs, returns it and the sine of x.
            global _comprehension_global
            _comprehension_global = [
                (i + x, torch._dynamo.graph_break()) for i in range(2)
            ]
            return _comprehension_global, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        res = fn(x)
        self.assertEqual(res, opt_fn(x))
        self.assertEqual(_comprehension_global, res[0])
        self.assertEqual(len(backend.graphs), 1)

    def test_store_attribute(self):
        # [FUNCTION DOC] Stores a breaking comprehension result into an attribute of a local
        # object. Eager-equal output, one recorded graph.
        class C:
            # [CLASS DOC] HELPER: Empty plain class used as an attribute holder.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: makes a C, sets vals from a comprehension of
            # pairs, returns obj.vals and the sine of x.
            obj = C()
            obj.vals = [(i + x, torch._dynamo.graph_break()) for i in range(2)]
            return obj.vals, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_augmented_assign(self):
        # [FUNCTION DOC] In-place add of a breaking comprehension result onto an existing
        # list. Eager-equal output, one recorded graph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: acc starts as a list holding x, extended
            # with a comprehension of pairs via augmented assignment; returns acc, sine of x.
            acc = [x]
            acc += [(i + x, torch._dynamo.graph_break()) for i in range(2)]
            return acc, x.sin()

        x = torch.randn(3, 3)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_comprehension_lambda_captures_outer_cellvar(self):
        # [FUNCTION DOC] Lambdas created in a comprehension capture the loop value i while
        # i is also an outer cell variable. Compiled with fullgraph (no break) and must
        # equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: i = 100, creates lambdas per loop value, then
            # returns their results, g(), the outer i, and x+1.
            i = 100

            def g():
                # [FUNCTION DOC] HELPER: Closure returning i; makes i a cell variable.
                return i  # makes `i` a cellvar of fn, not a freevar

            fns = [lambda: i for i in range(3)]
            return [h() for h in fns], g(), i, x + 1

        x = torch.randn(3, 3)
        self.assertEqual(fn(x), torch.compile(fn, backend="eager", fullgraph=True)(x))

    def test_locals_snapshot_with_cell(self):
        # [FUNCTION DOC] locals() is taken after a comprehension of lambdas while i is a
        # cell variable; the snapshot value of i must be the outer 100. Fullgraph compile,
        # eager-equal.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled body: builds lambdas, snapshots locals into a dict,
            # returns the snapshot's i, g(), lambda results, and x+1.
            i = 100

            def g():
                # [FUNCTION DOC] HELPER: Closure returning i; makes i a cell variable.
                return i

            fns = [lambda: i for i in range(3)]
            snap = dict(locals())
            return snap["i"], g(), [h() for h in fns], x + 1

        x = torch.randn(3, 3)
        self.assertEqual(fn(x), torch.compile(fn, backend="eager", fullgraph=True)(x))


@skipIfNotPy312
class NestedGraphBreakTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Python 3.12+ only. Same breaking-comprehension scenarios as
    # ComprehensionTests, but the comprehension lives in a function inlined into a
    # root function, so the break occurs in a nested frame. Each test checks
    # eager-equal output, graph count and per-graph add counts.
    def test_nested_function_calls_with_comprehension_graph_break_nested_graph_breaks_true(
        self,
    ):
        # [FUNCTION DOC] Three-deep calls (f, g, h) with a breaking comprehension in h:
        # two graphs, three add nodes in each.
        """Test nested function calls where inner function has comprehension with graph break."""

        def h(x):
            # [FUNCTION DOC] HELPER: Innermost: add 3, discarded breaking comprehension, add 4.
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 4
            return x

        def g(x):
            # [FUNCTION DOC] HELPER: Middle: add 2, call h, add 5.
            x = x + 2
            x = h(x)
            x = x + 5
            return x

        def f(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call g, add 6.
            x = x + 1
            x = g(x)
            x = x + 6
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    def test_nested_comprehension_result_var_name_collision(self):
        # [FUNCTION DOC] The temporary result name for the comprehension in the leaf frame
        # equals a local (x) in the root frame; the root's x must be unchanged. Two
        # graphs with one and three add nodes.
        """Comprehension result_var in leaf frame shares a name with a root frame local."""

        def inner(t):
            # [FUNCTION DOC] HELPER: Leaf: x is a breaking comprehension list; returns its sum
            # plus t.
            x = [torch._dynamo.graph_break() or i for i in range(3)]
            return sum(x) + t

        def root(t):
            # [FUNCTION DOC] HELPER: Root: x = t+1, calls inner, asserts x still equals t+1,
            # returns the combined value.
            x = t + 1
            result = inner(t) + x
            self.assertEqual(x, t + 1)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(root, backend=backend)
        t = torch.randn(4)

        self.assertEqual(compiled(t), root(t))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    def test_nested_function_comprehension_with_walrus(self):
        # [FUNCTION DOC] Inlined function with a walrus-binding comprehension whose iterable
        # contains the graph break. Two graphs with two and four add nodes.
        """Nested function with comprehension containing walrus operator."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Inlined callee: add 2, comprehension binding y (iterable
            # list triggers a break), add 3, returns x plus sum plus y.
            x = x + 2
            result = [(y := i * 2) for i in [torch._dynamo.graph_break() or 1, 2, 3]]
            x = x + 3
            return x + sum(result) + y

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            x = inner(x)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    def test_nested_function_comprehension_with_captured_var(self):
        # [FUNCTION DOC] Inlined function's comprehension multiplies items by a parameter
        # passed from the root. Two graphs with two and three add nodes.
        """Nested function with comprehension capturing outer variable passed as parameter."""

        def inner(x, multiplier):
            # [FUNCTION DOC] HELPER: Inlined callee: add 2, list scaled by multiplier (break in
            # the iterable), add 3 plus the list sum.
            x = x + 2
            result = [multiplier * i for i in [torch._dynamo.graph_break() or 1, 2, 3]]
            x = x + 3 + sum(result)
            return x

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: sets multiplier 5, add 1, call inner, add 4.
            multiplier = 5
            x = x + 1
            x = inner(x, multiplier)
            x = x + 4
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    def test_triple_nested_function_comprehension(self):
        # [FUNCTION DOC] Four-deep call chain (f, g, h, deepest) with the breaking
        # comprehension in deepest. Two graphs, four add nodes each.
        """Triple nested function calls (f->g->h->deepest) with comprehension in deepest."""

        def deepest(x):
            # [FUNCTION DOC] HELPER: Deepest: add 4, discarded breaking comprehension, add 5.
            x = x + 4
            [torch._dynamo.graph_break() or i for i in range(3)]
            x = x + 5
            return x

        def h(x):
            # [FUNCTION DOC] HELPER: add 3, call deepest, add 6.
            x = x + 3
            x = deepest(x)
            x = x + 6
            return x

        def g(x):
            # [FUNCTION DOC] HELPER: add 2, call h, add 7.
            x = x + 2
            x = h(x)
            x = x + 7
            return x

        def f(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call g, add 8.
            x = x + 1
            x = g(x)
            x = x + 8
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 4)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    def test_multiple_nested_comprehensions_in_different_functions(self):
        # [FUNCTION DOC] Two sequential inlined calls each containing a breaking
        # comprehension: three graphs with one, two and one add nodes.
        """Two sequential nested function calls each with comprehensions."""

        def inner1(x):
            # [FUNCTION DOC] HELPER: Add 1, discarded breaking comprehension, add 2.
            x = x + 1
            [torch._dynamo.graph_break() or i for i in range(2)]
            x = x + 2
            return x

        def inner2(x):
            # [FUNCTION DOC] HELPER: Add 3, discarded breaking comprehension, add 4.
            x = x + 3
            [torch._dynamo.graph_break() or i for i in range(2)]
            x = x + 4
            return x

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: calls inner1 then inner2.
            x = inner1(x)
            x = inner2(x)
            return x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        # Graph 1: x+1, Graph 2: x+2, x+3, Graph 3: x+4
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)

    def test_nested_function_comprehension_with_outer_list_mutation(self):
        # [FUNCTION DOC] Inlined callee appends to a list passed in by the root inside a
        # breaking comprehension. Two graphs with two add nodes each.
        """Nested function with comprehension mutating outer list."""

        def inner(x, results):
            # [FUNCTION DOC] HELPER: Add 2, comprehension appending to results with a break, add 3.
            x = x + 2
            [results.append(torch._dynamo.graph_break() or i) for i in range(3)]
            x = x + 3
            return x

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: creates results, add 1, call inner, add 4; returns
            # x and results.
            results = []
            x = x + 1
            x = inner(x, results)
            x = x + 4
            return x, results

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_list_comprehension_graph_break(self):
        # [FUNCTION DOC] Inlined version of the basic list comprehension break. Two graphs,
        # two add nodes each.
        """Nested version of test_list_comprehension_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, breaking list comprehension, add 3; returns all.
            y = x + 2
            result = [torch._dynamo.graph_break() or i for i in range(3)]
            z = x + 3
            return y, result, z

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4, returns inner's result.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_dict_comprehension_graph_break(self):
        # [FUNCTION DOC] Inlined version of the dict comprehension break. Two graphs, two
        # add nodes each.
        """Nested version of test_dict_comprehension_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, breaking dict comprehension of squares, add 3.
            y = x + 2
            result = {i: torch._dynamo.graph_break() or i**2 for i in range(3)}
            z = x + 3
            return y, result, z

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_list_comprehension_with_graph_break_function(self):
        # [FUNCTION DOC] Inlined version where the break comes from a called function. Two
        # graphs: first with one add and one multiply, second with three adds.
        """Nested version of test_list_comprehension_with_graph_break_function."""

        def graph_break_fn(i):
            # [FUNCTION DOC] HELPER: Returns i, breaking the graph first when i is 3.
            if i == 3:
                torch._dynamo.graph_break()
            return i

        def inner(x):
            # [FUNCTION DOC] HELPER: Multiply x by 2, list via graph_break_fn, add 3 plus sum.
            y = x * 2
            lst = [graph_break_fn(i) for i in range(5)]
            z = x + 3 + sum(lst)
            return z, y

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 5.
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[0], operator.mul), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 3)

    def test_nested_multiple_comprehensions_one_break(self):
        # [FUNCTION DOC] Inlined callee with three comprehensions, one breaking. Two graphs,
        # two add nodes each.
        """Nested version of test_multiple_comprehensions_one_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, plain list, breaking list, add 3, plain doubled list.
            a = x + 2
            list1 = [i for i in range(2)]  # noqa: C416
            list2 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            list3 = [i * 2 for i in range(2)]
            return a, list1, list2, b, list3

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_nested_comprehension_inner_break(self):
        # [FUNCTION DOC] Inlined callee with a nested comprehension whose inner part breaks.
        # Two graphs, two add nodes each.
        """Nested version of test_nested_comprehension_inner_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, nested breaking comprehension, add 3.
            a = x + 2
            result = [
                [torch._dynamo.graph_break() or i * j for j in range(2)]
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_multi_iterator_comprehension_break(self):
        # [FUNCTION DOC] Inlined callee with a two-iterator comprehension that breaks. Two
        # graphs, two add nodes each.
        """Nested version of test_multi_iterator_comprehension_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension of pairs over two ranges, add 3.
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or i, j)
                for i in range(2)
                for j in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_discarded_comprehension_graph_break(self):
        # [FUNCTION DOC] Inlined callee with a discarded breaking comprehension. Two graphs,
        # two add nodes each.
        """Nested version of test_discarded_comprehension_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, discarded breaking comprehension, add 3.
            a = x + 2
            [torch._dynamo.graph_break() or i for i in range(3)]
            b = x + 3
            return a, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_in_expression_graph_break(self):
        # [FUNCTION DOC] Inlined callee with a breaking comprehension inside a sum call. Two
        # graphs, two add nodes each.
        """Nested version of test_comprehension_in_expression_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, sum of breaking comprehension, add 3.
            a = x + 2
            total = sum([torch._dynamo.graph_break() or i for i in range(3)])
            b = x + 3
            return a, total, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_return_directly(self):
        # [FUNCTION DOC] Inlined callee returns a breaking comprehension directly. Two
        # graphs: two adds in the first, one in the second.
        """Nested version of test_comprehension_return_directly."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Unused add 2, then returns a breaking comprehension.
            a = x + 2  # noqa: F841
            return [torch._dynamo.graph_break() or i for i in range(3)]

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 3 (after the break).
            x = x + 1
            result = inner(x)
            x = x + 3
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_walrus_operator_in_comprehension(self):
        # [FUNCTION DOC] Inlined callee with a walrus in a breaking comprehension. Two
        # graphs, two add nodes each.
        """Nested version of test_walrus_operator_in_comprehension."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension binding y, add 3; returns y too.
            a = x + 2
            result = [(y := (torch._dynamo.graph_break() or i * 2)) for i in range(3)]
            b = x + 3
            return a, result, y, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_walrus_operator_in_if_in_comprehension(self):
        # [FUNCTION DOC] Inlined callee with a walrus in the filter of a breaking
        # comprehension. Two graphs, two add nodes each.
        """Nested version of test_walrus_operator_in_if_in_comprehension."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, filtered comprehension binding y in its filter, add 3.
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or y) for i in range(5) if (y := i * 2) > 2
            ]
            b = x + 3
            return a, result, y, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_walrus_operator_in_comprehension_with_tensor(self):
        # [FUNCTION DOC] Inlined callee with filter walrus and a numel read from x. Two
        # graphs, two add nodes each.
        """Nested version of test_walrus_operator_in_comprehension_with_tensor."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, filtered comprehension adding x.numel() to y, add 3.
            a = x + 2
            result = [
                (torch._dynamo.graph_break() or y + x.numel())
                for i in range(5)
                if (y := i * 2) > 2
            ]
            b = x + 3
            return a, result, y, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_multiple_walrus_operators_in_comprehension(self):
        # [FUNCTION DOC] Inlined callee binding y and z by walrus in one comprehension. Two
        # graphs, two add nodes each.
        """Nested version of test_multiple_walrus_operators_in_comprehension."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension of pairs binding y and z, add 3.
            a = x + 2
            result = [
                ((y := (torch._dynamo.graph_break() or i * 2)), (z := i * 3))
                for i in range(3)
            ]
            b = x + 3
            return a, result, y, z, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_nested_comprehension_with_walrus_operators(self):
        # [FUNCTION DOC] Inlined callee with walrus targets at both comprehension levels.
        # Two graphs, two add nodes each.
        """Nested version of test_nested_comprehension_with_walrus_operators."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, nested comprehension binding outer_val and inner_val,
            # add 3.
            a = x + 2
            result = [
                (
                    outer_val := i * 10,
                    [
                        (inner_val := (torch._dynamo.graph_break() or j * 2))
                        for j in range(2)
                    ],
                )
                for i in range(3)
            ]
            b = x + 3
            return a, result, outer_val, inner_val, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_nested_comprehension_with_captured_outer_variable(self):
        # [FUNCTION DOC] Inlined callee whose inner comprehension reads outer_val. Two
        # graphs, two add nodes each.
        """Nested version of test_nested_comprehension_with_captured_outer_variable."""

        def inner(x):
            # [FUNCTION DOC] HELPER: outer_val = 100, add 2, nested comprehension, add 3.
            outer_val = 100
            a = x + 2
            result = [
                [outer_val + j for j in range(2) if torch._dynamo.graph_break() or True]
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_triple_nested_comprehension_with_walrus(self):
        # [FUNCTION DOC] Inlined callee with triple nested comprehension and a walrus w. Two
        # graphs, two add nodes each.
        """Nested version of test_triple_nested_comprehension_with_walrus."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, triple nested comprehension binding w, add 3.
            a = x + 2
            result = [
                [
                    [
                        (w := i + j + k)
                        for k in range(2)
                        if torch._dynamo.graph_break() or True
                    ]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            b = x + 3
            return a, result, w, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_nested_dict_comprehension_with_walrus_and_captured(self):
        # [FUNCTION DOC] Inlined callee with nested dict comprehensions, a walrus and a
        # captured multiplier. Two graphs, two add nodes each.
        """Nested version of test_nested_dict_comprehension_with_walrus_and_captured."""

        def inner(x):
            # [FUNCTION DOC] HELPER: multiplier 10, add 2, nested dicts binding inner_val, add 3.
            multiplier = 10
            a = x + 2
            result = {
                i: {
                    j: (inner_val := j * multiplier)
                    for j in range(2)
                    if torch._dynamo.graph_break() or True
                }
                for i in range(3)
            }
            b = x + 3
            return a, result, inner_val, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_outer_variable_read(self):
        # [FUNCTION DOC] Inlined callee comprehension reads a local outer_val. Two graphs,
        # two add nodes each.
        """Nested version of test_comprehension_with_outer_variable_read."""

        def inner(x):
            # [FUNCTION DOC] HELPER: outer_val = 10, add 2, filtered breaking comprehension, add 3.
            outer_val = 10
            a = x + 2
            result = [
                i + outer_val for i in range(3) if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_outer_list_mutation(self):
        # [FUNCTION DOC] Inlined callee's comprehension appends to a list local. Two graphs,
        # two add nodes each.
        """Nested version of test_comprehension_with_outer_list_mutation."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension appending to outer_list, add 3.
            outer_list = []
            a = x + 2
            result = [
                outer_list.append(i) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_outer_dict_mutation(self):
        # [FUNCTION DOC] Inlined callee's comprehension updates a dict local. Two graphs,
        # two add nodes each.
        """Nested version of test_comprehension_with_outer_dict_mutation."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension updating outer_dict, add 3.
            outer_dict = {}
            a = x + 2
            result = [
                outer_dict.update({i: i * 10}) or i * 2
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_dict, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_outer_list_extend(self):
        # [FUNCTION DOC] Inlined callee's comprehension extends a list that starts with 100.
        # Two graphs, two add nodes each.
        """Nested version of test_comprehension_with_outer_list_extend."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension extending outer_list, add 3.
            outer_list = [100]
            a = x + 2
            result = [
                outer_list.extend([i, i * 10]) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_outer_list_pop(self):
        # [FUNCTION DOC] Inlined callee's comprehension pops from a list and records the
        # values. Two graphs, two add nodes each.
        """Nested version of test_comprehension_with_outer_list_pop."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension popping outer_list into popped_values,
            # add 3.
            outer_list = [10, 20, 30, 40, 50]
            popped_values = []
            a = x + 2
            result = [
                popped_values.append(outer_list.pop()) or i
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, outer_list, popped_values, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_global_variable(self):
        # [FUNCTION DOC] Sets the test global to 100 (side effect) and checks an inlined
        # comprehension that reads it. Two graphs, two add nodes each.
        """Nested version of test_comprehension_with_global_variable."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, comprehension adding the global, add 3.
            a = x + 2
            result = [
                i + _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
                for i in range(3)
                if torch._dynamo.graph_break() or True
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_closure_variable(self):
        # [FUNCTION DOC] Inlined callee reads a closure value (50). Two graphs, two add nodes
        # each.
        """Nested version of test_comprehension_with_closure_variable."""

        def make_outer():
            # [FUNCTION DOC] HELPER: Factory: defines closure_val, inner and outer; returns outer.
            closure_val = 50

            def inner(x):
                # [FUNCTION DOC] HELPER: Add 2, comprehension adding closure_val, add 3.
                a = x + 2
                result = [
                    i + closure_val
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 3
                return a, result, b

            def outer(x):
                # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_with_closure_list_mutation(self):
        # [FUNCTION DOC] Inlined callee appends to a closure list; reference uses a freshly
        # built function. Two graphs, two add nodes each.
        """Nested version of test_comprehension_with_closure_list_mutation."""

        def make_outer():
            # [FUNCTION DOC] HELPER: Factory: defines closure_list, inner and outer; returns outer.
            closure_list = []

            def inner(x):
                # [FUNCTION DOC] HELPER: Add 2, comprehension appending to closure_list, add 3;
                # returns a copy of the list.
                a = x + 2
                result = [
                    closure_list.append(i) or i * 2
                    for i in range(3)
                    if torch._dynamo.graph_break() or True
                ]
                b = x + 3
                return a, result, closure_list.copy(), b

            def outer(x):
                # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        outer_ref = make_outer()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_nested_multi_for_comprehension_graph_break(self):
        # [FUNCTION DOC] Inlined callee with two outer for clauses and a breaking inner
        # comprehension. Two graphs, two add nodes each.
        """Nested version of test_nested_multi_for_comprehension_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, list of lists over two ranges, add 3.
            a = x + 2
            result = [
                [(torch._dynamo.graph_break() or i + j + k) for k in range(2)]
                for i in range(2)
                for j in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_multiple_comprehension_graph_breaks(self):
        # [FUNCTION DOC] Inlined callee with two breaking comprehensions. Three graphs with
        # two, one and two add nodes.
        """Nested version of test_multiple_comprehension_graph_breaks."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, list break, add 3, second break, add 4.
            a = x + 2
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            list2 = [torch._dynamo.graph_break() or i * 2 for i in range(2)]
            c = x + 4
            return a, list1, b, list2, c

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 5.
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 2)

    def test_nested_comprehension_modifying_global_variable(self):
        # [FUNCTION DOC] Inlined callee increments a global before a breaking comprehension;
        # the global is reset around each run (side effect). Two graphs, two adds each.
        """Nested version of test_comprehension_modifying_global_variable."""
        global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = 100
        original_value = _GLOBAL_VALUE_FOR_COMPREHENSION_TEST

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, increment global, breaking comprehension, add 3.
            global _GLOBAL_VALUE_FOR_COMPREHENSION_TEST
            a = x + 2
            _GLOBAL_VALUE_FOR_COMPREHENSION_TEST += 1
            result = [
                torch._dynamo.graph_break() or _GLOBAL_VALUE_FOR_COMPREHENSION_TEST + i
                for i in range(3)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        compiled_result = compiled(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value
        expected_result = outer(x)
        _GLOBAL_VALUE_FOR_COMPREHENSION_TEST = original_value

        self.assertEqual(compiled_result, expected_result)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_modifying_closure_variable(self):
        # [FUNCTION DOC] Inlined callee mutates a closure list cell before a breaking
        # comprehension; reference uses a fresh function. Two graphs, two adds each.
        """Nested version of test_comprehension_modifying_closure_variable."""

        def make_outer():
            # [FUNCTION DOC] HELPER: Factory: creates closure_val, inner and outer; returns outer.
            closure_val = [0]

            def inner(x):
                # [FUNCTION DOC] HELPER: Add 2, increment closure_val's element, breaking
                # comprehension, add 3.
                a = x + 2
                closure_val[0] += 1
                result = [
                    torch._dynamo.graph_break() or closure_val[0] + i for i in range(3)
                ]
                b = x + 3
                return a, result, closure_val[0], b

            def outer(x):
                # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
                x = x + 1
                result = inner(x)
                x = x + 4
                return result

            return outer

        outer = make_outer()
        outer_ref = make_outer()

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer_ref(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_list_and_dict_comprehension_graph_breaks(self):
        # [FUNCTION DOC] Inlined callee with a breaking list then dict comprehension. Three
        # graphs with two, one and two add nodes.
        """Nested version of test_list_and_dict_comprehension_graph_breaks."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, list break, add 3, dict break, add 4.
            a = x + 2
            list1 = [torch._dynamo.graph_break() or i for i in range(2)]
            b = x + 3
            dict1 = {i: torch._dynamo.graph_break() or i * 10 for i in range(2)}
            c = x + 4
            return a, list1, b, dict1, c

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 5.
            x = x + 1
            result = inner(x)
            x = x + 5
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 2)

    def test_nested_nested_dict_in_list_comprehension_graph_break(self):
        # [FUNCTION DOC] Inlined callee with a dict comprehension in a list comprehension that
        # breaks. Two graphs, two add nodes each.
        """Nested version of test_nested_dict_in_list_comprehension_graph_break."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Add 2, list of dicts with breaking values, add 3.
            a = x + 2
            result = [
                {j: torch._dynamo.graph_break() or i * j for j in range(2)}
                for i in range(2)
            ]
            b = x + 3
            return a, result, b

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, call inner, add 4.
            x = x + 1
            result = inner(x)
            x = x + 4
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_nonlocal_int_mutation(self):
        # [FUNCTION DOC] Nonlocal int cell incremented before and after a discarded breaking
        # comprehension in an inner function; asserts the cell ends at 3. Two graphs
        # with one add node each.
        """Nonlocal int mutation around a comprehension graph break."""

        def f1(x):
            # [FUNCTION DOC] HELPER: Defines cell = 1 and f2, calls f2, asserts cell is 3.
            cell = 1

            def f2(x):
                # [FUNCTION DOC] HELPER: Unused add 1, nonlocal cell plus 1, break, cell plus 1,
                # returns x plus cell.
                b = x + 1  # noqa: F841
                nonlocal cell
                cell = cell + 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell += 1
                return x + cell

            result = f2(x)
            self.assertEqual(cell, 3)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_nonlocal_tensor_mutation(self):
        # [FUNCTION DOC] Same as the int case with a tensor cell (in-place add before the
        # break, rebinding add after). Cell must equal 3s. Two graphs: one and two adds.
        """Nonlocal tensor mutation around a comprehension graph break."""

        def f1(x):
            # [FUNCTION DOC] HELPER: Defines cell as ones, calls f2, asserts cell is all 3.
            cell = torch.ones(4)

            def f2(x):
                # [FUNCTION DOC] HELPER: Unused add 1, in-place cell add, break, cell plus 1,
                # returns x plus cell.
                b = x + 1  # noqa: F841
                nonlocal cell
                cell += 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell = cell + 1
                return x + cell

            result = f2(x)
            self.assertEqual(cell, torch.full((4,), 3, dtype=torch.float32))
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_nonlocal_different_varnames(self):
        # [FUNCTION DOC] Inlined function's locals use different names from the root. Cell
        # sums reductions of tensors around a break; asserts cell equals 14. Two graphs,
        # three and four add nodes.
        """Inner function uses different variable names than root function."""

        def f1(x):
            # [FUNCTION DOC] HELPER: Defines cell = 1 and f2, calls f2 on x, asserts cell is 14.
            cell = 1

            def f2(y):
                # [FUNCTION DOC] HELPER: Computes a and b, adds a reduction to nonlocal cell, break
                # in comprehension, adds 1 plus y.sum to cell; returns x plus a plus y.
                a = x + 1
                b = y + 2
                nonlocal cell
                cell = cell + (b - y).sum()
                [torch._dynamo.graph_break() or y.sum() * i for i in range(2)]
                cell = cell + 1 + y.sum()
                return x + a + y

            result = f2(x)
            self.assertEqual(cell, 14)
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.ones(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 3)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 4)

    def test_nested_comprehension_nonlocal_tensor_different_varnames(self):
        # [FUNCTION DOC] Tensor-cell variant with different parameter and local names. Two
        # graphs, two add nodes each. Cell must end as all 3.
        """Inner function uses different variable names than root function (tensor variant)."""

        def f1(x):
            # [FUNCTION DOC] HELPER: Defines cell as ones, calls f2, asserts cell is all 3.
            cell = torch.ones(4)

            def f2(y):
                # [FUNCTION DOC] HELPER: Unused add, b, in-place cell add, break, cell plus 1;
                # returns b plus cell.
                a = y + 1  # noqa: F841
                b = x + 2
                nonlocal cell
                cell += 1
                [torch._dynamo.graph_break() or i for i in range(2)]
                cell = cell + 1
                return b + cell

            result = f2(x)
            self.assertEqual(cell, torch.full((4,), 3, dtype=torch.float32))
            return result

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(f1, backend=backend)
        x = torch.randn(4)

        self.assertEqual(compiled(x), f1(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 2)

    def test_nested_comprehension_captured_tensor(self):
        # [FUNCTION DOC] Comprehension in an inlined function reads a tensor via item()
        # from the callee. Two graphs with two and one add nodes.
        """Comprehension in inlined function captures a tensor from the inlined function."""

        def inner(x):
            # [FUNCTION DOC] HELPER: t = x + 2, breaking comprehension of t.item(), returns x+3.
            t = x + 2
            result = [torch._dynamo.graph_break() or t.item() for i in range(2)]
            return x + 3, result

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: add 1, return inner(x).
            x = x + 1
            return inner(x)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.tensor(1.0)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 1)

    def test_nested_comprehension_name_shadowing(self):
        # [FUNCTION DOC] Root and leaf frames both have a local named a. Values must stay
        # separate after the break. Four graphs recorded (two adds, none, one, one).
        """Root and leaf frames have variables with the same name."""

        def inner(x):
            # [FUNCTION DOC] HELPER: a = x+2, breaking item() comprehension, asserts a is 4, b.
            a = x + 2
            result = [torch._dynamo.graph_break() or a.item() for i in range(2)]
            self.assertEqual(a, torch.full((), 4, dtype=torch.float32))
            b = x + 3
            return b, result

        def outer(x):
            # [FUNCTION DOC] HELPER: Root: a = x+1, call inner, asserts a is 2, returns a+4.
            a = x + 1
            r = inner(a)
            self.assertEqual(a, torch.full((), 2, dtype=torch.float32))
            return a + 4, r

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        compiled = torch.compile(outer, backend=backend)
        x = torch.tensor(1.0)

        self.assertEqual(compiled(x), outer(x))
        self.assertEqual(len(backend.graphs), 4)
        self.assertEqual(count_op(backend.graphs[0], operator.add), 2)
        self.assertEqual(count_op(backend.graphs[1], operator.add), 0)
        self.assertEqual(count_op(backend.graphs[2], operator.add), 1)
        self.assertEqual(count_op(backend.graphs[3], operator.add), 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
