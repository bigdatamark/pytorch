# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo's graph deduplication, which replaces repeated regions of the
# traced graph with a shared subgraph invoked through the invoke_subgraph higher-order
# operator. Covers single and multiple repeated regions, dependent regions, input mutation and
# aliasing, cycle detection in a graph, stable topological sorting with extra dependencies
# (autocast ordering, output node placement, mutation ordering), tuple returns and inputs, and
# static address parameters. Most tests compare printed graphs with expected strings. Import
# time has no side effects.
import contextlib

import torch
import torch.fx
from torch._dynamo.graph_deduplication import apply_graph_deduplication
from torch._dynamo.graph_utils import _detect_cycles
from torch._dynamo.output_graph import FakeRootModule
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import extract_graph, extract_graph_and_tracker, normalize_gm
from torch.compiler import allow_in_graph
from torch.utils._ordered_set import OrderedSet


def graph_str(gm):
    # [FUNCTION DOC] HELPER: Return the printed readable text of a graph module, normalized
    # (comments, trailing spaces and repeated blank lines removed), for snapshot comparison.
    return normalize_gm(gm.print_readable(print_output=False))


class GraphDededuplicationTests(TestCase):
    # [CLASS DOC] Tests for graph deduplication. Each test runs with the deduplication config
    # option turned on through an ExitStack set up in setUp.
    def setUp(self):
        # [FUNCTION DOC] Run base setup, then create an ExitStack and enter a config patch
        # enabling graph deduplication.
        super().setUp()
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            torch._dynamo.config.patch("use_graph_deduplication", True)
        )

    def tearDown(self):
        # [FUNCTION DOC] Close the ExitStack (restoring the config) and then run
        # base teardown.
        self.exit_stack.close()
        super().tearDown()

    def run_and_return_graphs(self, fn, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Compile and run the function with the AOT recording backend
        # and return the first three items of the result: the output, the dynamo graphs and
        # the forward graphs.
        return extract_graph(fn, *args, **kwargs)[0:3]

    def run_and_get_simple_graph(self):
        # [FUNCTION DOC] HELPER: Compile a tiny function (two additions, two sums, a final
        # addition) on 10x10 and 10x20 inputs without grad and return its first forward graph.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, 2 to y and returns the
            # sum of the two reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        return fw_graphs[0]

    def test_single_subgraph(self):
        # [FUNCTION DOC] A repeated inner function called three times is deduplicated into one
        # subgraph; checks compiled output graphs and forward graphs against expected strings
        # and runs backward.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Computes x plus 1 and y plus 2, sums
            # both and adds the sums.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function three times
            # (once with sin of y as second argument), combines results by multiplication and
            # returns the product.
            _o0 = inner_fn(x, y)
            o1 = torch.sin(y)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            o4 = o3 * o3
            return o2 * o4

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()

        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]", L_y_: "f32[10, 20]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  invoke_subgraph = None
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)

        getitem: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        mul: "f32[]" = getitem * getitem;  getitem = None

        sin: "f32[10, 20]" = torch.sin(l_y_);  l_y_ = None

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, sin);  subgraph_0 = l_x_ = sin = None

        getitem_1: "f32[]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        mul_1: "f32[]" = getitem_1 * mul;  getitem_1 = mul = None
        return (mul_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_l_x_, subgraph_input_l_y_):
            x0: "f32[10, 10]" = subgraph_input_l_x_ + 1;  subgraph_input_l_x_ = None

            y0: "f32[10, 20]" = subgraph_input_l_y_ + 2;  subgraph_input_l_y_ = None

            sum_1: "f32[]" = x0.sum();  x0 = None
            sum_2: "f32[]" = y0.sum();  y0 = None
            z: "f32[]" = sum_1 + sum_2;  sum_1 = sum_2 = None
            return (z,)
""",
        )

        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = None

        getitem_1: "f32[]" = invoke_subgraph_5[0];  invoke_subgraph_5 = None

        mul: "f32[]" = torch.ops.aten.mul.Tensor(getitem_1, getitem_1)

        sin: "f32[10, 20]" = torch.ops.aten.sin.default(primals_2)

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, sin);  partitioned_fw_subgraph_0_1 = primals_1 = sin = None

        getitem_2: "f32[]" = invoke_subgraph_7[0];  invoke_subgraph_7 = None

        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(getitem_2, mul);  mul = None
        return (mul_1, primals_2, getitem_1, getitem_2)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None
            add_2: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add_2,)
""",
        )

    def test_single_subgraph2(self):
        # [FUNCTION DOC] The same inner function used twice in series (with a cosine in
        # between) is deduplicated into one subgraph; checks printed dynamo and forward graphs
        # and runs backward.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds 2, applies the inner function,
            # takes the cosine, applies the inner function again and returns the sine.
            x0 = x + 2
            o = inner_fn(x0)
            o = torch.cos(o)
            o = inner_fn(o)
            return torch.sin(o)

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Multiplies by 7 and adds 1 and 2
            # in place.
            o = x * 7
            o += 1
            o += 2
            return o

        x = torch.rand(10, 10, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)

        ref_result = fn(x)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]"):
        l_x_ = L_x_

        add: "f32[10, 10]" = l_x_ + 2;  l_x_ = None

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', add);  add = None

        getitem: "f32[10, 10]" = invoke_subgraph[0];  invoke_subgraph = None

        cos: "f32[10, 10]" = torch.cos(getitem);  getitem = None

        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', cos);  subgraph_0 = cos = None

        getitem_1: "f32[10, 10]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sin: "f32[10, 10]" = torch.sin(getitem_1);  getitem_1 = None
        return (sin,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_x0):
            o: "f32[10, 10]" = subgraph_input_x0 * 7;  subgraph_input_x0 = None

            o += 1;  o_1: "f32[10, 10]" = o;  o = None

            o_1 += 2;  o_2: "f32[10, 10]" = o_1;  o_1 = None
            return (o_2,)
""",
        )
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]"):
        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', add);  partitioned_fw_subgraph_0_0 = add = None

        getitem: "f32[10, 10]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        cos: "f32[10, 10]" = torch.ops.aten.cos.default(getitem)

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', cos);  partitioned_fw_subgraph_0_1 = cos = None

        getitem_1: "f32[10, 10]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None

        sin: "f32[10, 10]" = torch.ops.aten.sin.default(getitem_1)

        cos_1: "f32[10, 10]" = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None

        sin_1: "f32[10, 10]" = torch.ops.aten.sin.default(getitem);  getitem = None
        neg: "f32[10, 10]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
        return (sin, cos_1, neg)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(primals_0, 7);  primals_0 = None

            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(mul, 1);  mul = None

            add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, 2);  add = None
            return (add_1,)
""",
        )

    def test_multiple_subgraphs(self):
        # [FUNCTION DOC] Two different repeated inner functions each deduplicate into their
        # own subgraph; checks printed graphs and runs backward.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            # [FUNCTION DOC] HELPER: Function under test. Adds 2 to a, 3 to b and multiplies a
            # by the sum of cosine of b.
            a0 = a + 2
            b0 = b + 3
            c = a0 * b0.cos().sum()
            return c

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls both inner functions several
            # times, mixing results through sine, product and addition, and returns the
            # combined result.
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o3 = inner_fn2(x0, y0)
            o4 = inner_fn(x, y)
            return o1 * o2 * o3 + o4

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)

        self.assertExpectedInline(
            graph_str(graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10, 10]", L_y_: "f32[10, 20]"):
        l_x_ = L_x_
        l_y_ = L_y_

        cos: "f32[10, 10]" = torch.cos(l_x_)

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)

        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None

        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)

        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sin: "f32[10, 20]" = torch.sin(l_y_);  l_y_ = None

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, sin);  subgraph_0 = l_x_ = None

        getitem_2: "f32[]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', cos, sin);  invoke_subgraph_3 = None
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', cos, sin);  subgraph_1 = cos = sin = None

        getitem_3: "f32[10, 10]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        sin_1: "f32[]" = torch.sin(getitem);  getitem = None

        mul: "f32[]" = sin_1 * getitem_2;  sin_1 = getitem_2 = None
        mul_1: "f32[10, 10]" = mul * getitem_3;  mul = getitem_3 = None
        add: "f32[10, 10]" = mul_1 + getitem_1;  mul_1 = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, subgraph_input_l_x_, subgraph_input_l_y_):
            x1: "f32[10, 10]" = subgraph_input_l_x_ + 1;  subgraph_input_l_x_ = None

            y1: "f32[10, 20]" = subgraph_input_l_y_ + 2;  subgraph_input_l_y_ = None

            sum_2: "f32[]" = x1.sum();  x1 = None
            sum_3: "f32[]" = y1.sum();  y1 = None
            z: "f32[]" = sum_2 + sum_3;  sum_2 = sum_3 = None
            return (z,)

    class subgraph_1(torch.nn.Module):
        def forward(self, subgraph_input_x0, subgraph_input_y0):
            a0: "f32[10, 10]" = subgraph_input_x0 + 2;  subgraph_input_x0 = None

            b0: "f32[10, 20]" = subgraph_input_y0 + 3;  subgraph_input_y0 = None

            cos_1: "f32[10, 20]" = b0.cos();  b0 = None
            sum_1: "f32[]" = cos_1.sum();  cos_1 = None
            c: "f32[10, 10]" = a0 * sum_1;  a0 = sum_1 = None
            return (c,)
""",
        )
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        cos: "f32[10, 10]" = torch.ops.aten.cos.default(primals_1)

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_9 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = None

        getitem: "f32[]" = invoke_subgraph_9[0];  invoke_subgraph_9 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_11 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_1 = None

        getitem_1: "f32[]" = invoke_subgraph_11[0];  invoke_subgraph_11 = None

        sin: "f32[10, 20]" = torch.ops.aten.sin.default(primals_2)

        partitioned_fw_subgraph_0_2 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_13 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_2, 'partitioned_fw_subgraph_0_0', primals_1, sin);  partitioned_fw_subgraph_0_2 = None

        getitem_2: "f32[]" = invoke_subgraph_13[0];  invoke_subgraph_13 = None
        partitioned_fw_subgraph_1_0 = self.partitioned_fw_subgraph_1_0

        invoke_subgraph_15 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_1_0, 'partitioned_fw_subgraph_1_0', cos, sin);  partitioned_fw_subgraph_1_0 = cos = sin = None
        getitem_19: "f32[]" = invoke_subgraph_15[3]
        getitem_18: "f32[10, 20]" = invoke_subgraph_15[2]
        getitem_17: "f32[10, 10]" = invoke_subgraph_15[1]

        getitem_4: "f32[10, 10]" = invoke_subgraph_15[0];  invoke_subgraph_15 = None

        sin_1: "f32[]" = torch.ops.aten.sin.default(getitem)

        mul: "f32[]" = torch.ops.aten.mul.Tensor(sin_1, getitem_2);  sin_1 = None
        mul_1: "f32[10, 10]" = torch.ops.aten.mul.Tensor(mul, getitem_4);  mul = None
        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(mul_1, getitem_1);  mul_1 = getitem_1 = None
        return (add, primals_1, primals_2, getitem, getitem_2, getitem_19, getitem_18, getitem_17, getitem_4)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None
            add_2: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add_2,)

    class partitioned_fw_subgraph_1_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 20]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 2)

            add_1: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_1, 3)

            cos: "f32[10, 20]" = torch.ops.aten.cos.default(add_1);  add_1 = None
            sum_1: "f32[]" = torch.ops.aten.sum.default(cos);  cos = None
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(add, sum_1);  add = None
            return (mul, primals_0, primals_1, sum_1)
""",
        )

    def test_dependent_subgraphs(self):
        # [FUNCTION DOC] Two calls where the second consumes the first's output are
        # deduplicated; checks the forward graph string and runs backward.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x and 2 to y and returns
            # the sum of their reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the inner function with x and
            # y, then with x and the first result, returning the second result.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, o0)
            return o1

        x = torch.rand(10, 10, requires_grad=True)
        y = torch.rand(10, 20, requires_grad=True)
        x_clone = x.clone().requires_grad_(True)
        y_clone = y.clone().requires_grad_(True)

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        ref_result.sum().backward()
        result.sum().backward()
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10, 10]", primals_2: "f32[10, 20]"):
        add: "f32[10, 20]" = torch.ops.aten.add.Tensor(primals_2, 2);  primals_2 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, sum_1);  partitioned_fw_subgraph_0_0 = sum_1 = None

        getitem: "f32[]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        add_1: "f32[]" = torch.ops.aten.add.Tensor(getitem, 2);  getitem = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(add_1);  add_1 = None

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_1, sum_2);  partitioned_fw_subgraph_0_1 = primals_1 = sum_2 = None

        getitem_1: "f32[]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        return (getitem_1,)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[]"):
            add: "f32[10, 10]" = torch.ops.aten.add.Tensor(primals_0, 1);  primals_0 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
            add_1: "f32[]" = torch.ops.aten.add.Tensor(sum_1, primals_1);  sum_1 = primals_1 = None
            return (add_1,)
""",
        )

    def test_input_mutation(self):
        # [FUNCTION DOC] Two calls of a function that mutates its inputs in place: only the
        # reductions form the shared subgraph, the in-place adds stay in the outer graph and
        # the mutation is written back with a copy; checks the forward graph string.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds in place one plus x to x and
            # one plus y to y, and returns the sum of the reductions.
            x0 = x + 1
            y0 = y + 1
            x.add_(x0)
            y.add_(y0)
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls the mutating function on the
            # sine of x with y, and on clones of the same, returning the sum.
            x0 = torch.sin(x)
            o2 = inner_fn2(x0, y)
            o3 = inner_fn2(x0.clone(), y.clone())
            return o2 + o3

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        add: "f32[10, 20]" = torch.ops.aten.add.Tensor(arg1_1, 1)

        sin: "f32[10, 10]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(sin, 1)

        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(sin, add_1);  sin = add_1 = None

        add_3: "f32[10, 20]" = torch.ops.aten.add.Tensor(arg1_1, add);  add = None

        clone: "f32[10, 20]" = torch.ops.aten.clone.default(add_3)

        add_4: "f32[10, 20]" = torch.ops.aten.add.Tensor(clone, 1)

        clone_1: "f32[10, 10]" = torch.ops.aten.clone.default(add_2)

        add_5: "f32[10, 10]" = torch.ops.aten.add.Tensor(clone_1, 1)

        add_6: "f32[10, 10]" = torch.ops.aten.add.Tensor(clone_1, add_5);  clone_1 = add_5 = None

        add_7: "f32[10, 20]" = torch.ops.aten.add.Tensor(clone, add_4);  clone = add_4 = None

        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', add_2, add_3);  repeated_subgraph0 = add_2 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', add_6, add_7);  repeated_subgraph0_1 = add_6 = add_7 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add_8: "f32[]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None

        copy_: "f32[10, 20]" = torch.ops.aten.copy_.default(arg1_1, add_3);  arg1_1 = add_3 = copy_ = None
        return (add_8,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_input_aliasing(self):
        # [FUNCTION DOC] Repeated functions with view-only (aliasing) outputs are not
        # deduplicated, while the multiplying function is; checks the forward graph string.
        # KNOWN ISSUE: the boolean result of the allclose check is discarded, so the
        # comparison against the eager result is never actually asserted.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Creates a view of x with its own
            # shape and a view of that, returning the second view.
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Doubles both inputs and returns the
            # sum of their reductions.
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls each inner function twice and
            # combines results by addition of views and sums.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        ref_result = fn(x, y)
        result, graphs, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)

        torch.allclose(ref_result, result)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(fw_graphs), 1)
        self.assertExpectedInline(
            graph_str(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])
        view_1: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10]);  arg0_1 = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None
        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_1, [10, 10]);  view_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_cycle_detection_no_cycle(self):
        # [FUNCTION DOC] Cycle detection on an unmodified simple graph reports that no cycle
        # was detected.
        mod = self.run_and_get_simple_graph()
        self.assertExpectedInline(
            _detect_cycles(mod.graph, {}), """no cycle detected"""
        )

    def test_cycle_detection_single_node(self):
        # [FUNCTION DOC] Rewires the first addition to depend on a later node and gives an
        # extra dependency map entry for that later node on itself; cycle detection reports
        # the expected path.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, 2 to y and returns the
            # sum of reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        mod = fw_graphs[0]
        add_node = next(n for n in mod.graph.nodes if n.name == "add")
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = add_node.args
        add_node.args = (args[0], add_2)
        self.assertExpectedInline(
            _detect_cycles(mod.graph, {add_2: OrderedSet([add_2])}),
            """cycle detected in path: [output, add_2, sum_1, add, add_2]""",
        )

    def test_cycle_detection_two_node(self):
        # [FUNCTION DOC] As the single node test but the extra dependency map has two entries
        # pointing at each other; cycle detection reports the expected path.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, 2 to y and returns the
            # sum of reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        mod = fw_graphs[0]
        add_node = next(n for n in mod.graph.nodes if n.name == "add")
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = add_node.args
        add_node.args = (args[0], add_2)
        self.assertExpectedInline(
            _detect_cycles(
                mod.graph,
                {add_2: OrderedSet([add_node]), add_node: OrderedSet([add_2])},
            ),
            """cycle detected in path: [output, add_2, sum_1, add, add_2]""",
        )

    def test_cycle_detection_arg_and_additional_deps(self):
        # [FUNCTION DOC] Cycle created by an argument edge, plus an extra dependency entry;
        # cycle detection reports the expected path.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, 2 to y and returns the
            # sum of reductions.
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)

        _, _, fw_graphs = self.run_and_return_graphs(fn, x, y)
        mod = fw_graphs[0]
        add_node = next(n for n in mod.graph.nodes if n.name == "add")
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = add_node.args
        add_node.args = (args[0], add_2)
        self.assertExpectedInline(
            _detect_cycles(mod.graph, {add_2: OrderedSet([add_node])}),
            """cycle detected in path: [output, add_2, sum_1, add, add_2]""",
        )

    def test_cycle_detection_simple(self):
        # [FUNCTION DOC] Argument edge rewired to create a cycle in the simple graph, no extra
        # dependencies; cycle detection reports the expected path.
        mod = self.run_and_get_simple_graph()
        add_node = next(n for n in mod.graph.nodes if n.name == "add")
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = add_node.args
        add_node.args = (args[0], add_2)
        self.assertExpectedInline(
            _detect_cycles(mod.graph, {}),
            """cycle detected in path: [output, add_2, sum_1, add, add_2]""",
        )

    def test_cycle_detection_complex(self):
        # [FUNCTION DOC] Rewires the first invoke_subgraph node to depend on a later add
        # node in a graph with deduplicated regions; cycle detection reports the expected
        # longer path.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Creates a view of x and a view of
            # that, returning the second.
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Doubles x and y and returns the sum
            # of their reductions.
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls each inner function twice and
            # combines the results.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]
        invoke_subgraph_node = next(
            n for n in mod.graph.nodes if n.name == "invoke_subgraph"
        )
        add_2 = next(n for n in mod.graph.nodes if n.name == "add_2")
        args = invoke_subgraph_node.args
        invoke_subgraph_node.args = (add_2, args[1])
        self.assertExpectedInline(
            _detect_cycles(mod.graph, {}),
            """cycle detected in path: [output, add_2, add_1, sum_1, getitem, invoke_subgraph, add_2]""",
        )

    def test_autocast_ordering(self):
        # [FUNCTION DOC] Inserts enter and exit autocast nodes into a deduplicated graph,
        # computes extra dependencies, moves the autocast nodes, then checks the printed graph
        # before and after a stable topological sort with the dependencies (the nodes end up
        # around the second sum).
        from torch._dynamo.graph_deduplication import (
            _populate_additional_deps,
            _stable_topological_sort,
        )

        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Creates a view of x and a view of
            # that, returning the second.
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Doubles x and y and returns the sum
            # of their reductions.
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls each inner function twice and
            # combines the results.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]

        def get_node(name):
            # [FUNCTION DOC] HELPER: Return the first graph node whose name equals the given
            # name; raises StopIteration if none exists.
            return next(n for n in mod.graph.nodes if n.name == name)

        sum_1 = get_node("sum_1")
        enter_autocast = mod.graph.call_function(torch.amp._enter_autocast)
        sum_1.append(enter_autocast)
        sum_2 = get_node("sum_2")
        exit_autocast = mod.graph.call_function(torch.amp._exit_autocast)
        sum_2.append(exit_autocast)
        additional_deps = _populate_additional_deps(mod.graph, {})
        invoke_subgraph = get_node("invoke_subgraph")
        invoke_subgraph.append(enter_autocast)
        getitem_1 = get_node("getitem_1")
        getitem_1.append(exit_autocast)
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        _enter_autocast = torch.amp.autocast_mode._enter_autocast();  _enter_autocast = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        _exit_autocast = torch.amp.autocast_mode._exit_autocast();  _exit_autocast = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])
        view_1: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10]);  arg0_1 = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None
        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_1, [10, 10]);  view_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )
        _stable_topological_sort(mod.graph, additional_deps)
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None

        _enter_autocast = torch.amp.autocast_mode._enter_autocast();  _enter_autocast = None

        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

        _exit_autocast = torch.amp.autocast_mode._exit_autocast();  _exit_autocast = None

        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])
        view_1: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10]);  arg0_1 = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None
        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_1, [10, 10]);  view_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_output_nodes_last(self):
        # [FUNCTION DOC] Moves the output node to a mid-graph position (after the node named
        # sum_2) and checks that a stable topological sort without extra dependencies puts it
        # back last.
        # KNOWN ISSUE: the variable is named add_2 although it holds the node named sum_2.
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Creates a view of x and a view of
            # that, returning the second.
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Doubles x and y and returns the sum
            # of their reductions.
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Calls each inner function twice and
            # combines the results.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            o2 = inner_fn2(x, y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10, requires_grad=False)
        y = torch.rand(10, 20, requires_grad=False)
        x_clone = x.clone()
        y_clone = y.clone()

        _, _, fw_graphs = self.run_and_return_graphs(fn, x_clone, y_clone)
        mod = fw_graphs[0]
        output = next(n for n in mod.graph.nodes if n.op == "output")
        add_2 = next(n for n in mod.graph.nodes if n.name == "sum_2")
        add_2.append(output)

        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None
        return (add_2,)

        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])
        view_1: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10]);  arg0_1 = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None
        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_1, [10, 10]);  view_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )
        _stable_topological_sort(mod.graph, {})
        self.assertExpectedInline(
            graph_str(mod),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0 = None
        getitem: "f32[]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', arg0_1, arg1_1);  repeated_subgraph0_1 = arg1_1 = None
        getitem_1: "f32[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem);  getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_1);  getitem_1 = None

        view: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10])
        view_1: "f32[10, 10]" = torch.ops.aten.view.default(arg0_1, [10, 10]);  arg0_1 = None

        view_2: "f32[10, 10]" = torch.ops.aten.view.default(view, [10, 10]);  view = None
        view_3: "f32[10, 10]" = torch.ops.aten.view.default(view_1, [10, 10]);  view_1 = None

        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(view_2, view_3);  view_2 = view_3 = None
        add_1: "f32[10, 10]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
        add_2: "f32[10, 10]" = torch.ops.aten.add.Tensor(add_1, sum_2);  add_1 = sum_2 = None
        return (add_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[10, 10]", arg1_1: "f32[10, 20]"):
            mul: "f32[10, 10]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None

            mul_1: "f32[10, 20]" = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
            sum_2: "f32[]" = torch.ops.aten.sum.default(mul_1);  mul_1 = None
            add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            return (add,)
""",
        )

    def test_mutation_ordering(self):
        # [FUNCTION DOC] Traces a function with in-place mutations between repeated regions,
        # checks the generated code, moves the mutation nodes around, and checks that a stable
        # sort using the module's recorded extra dependencies restores the original order of
        # the mutations relative to subgraph calls.
        from torch._dynamo.graph_deduplication import _stable_topological_sort

        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Creates a view of x and a view of
            # that, returning the second.
            x0 = x.view(x.size())
            return x0.view(x.size())

        def inner_fn2(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Doubles x and y and returns the sum
            # of their reductions.
            x = x * 2
            y = y * 2
            return x.sum() + y.sum()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Computes views, mutates x by adding
            # itself in place, uses the doubling function, mutates y by multiplying in place,
            # uses the doubling function again and combines the results.
            o0 = inner_fn(x, y)
            o1 = inner_fn(x, y)
            x.add_(x)
            o2 = inner_fn2(x, y)
            y.mul_(y)
            o3 = inner_fn2(x, y)
            return o0 + o1 + o2.sum() + o3.sum()

        x = torch.rand(10, 10)
        y = torch.rand(10, 20)
        x_clone = x.clone()
        y_clone = y.clone()

        graph, _ = extract_graph_and_tracker(fn, x_clone, y_clone)

        def graph_code(graph):
            # [FUNCTION DOC] HELPER: Return the Python source text generated for the graph.
            return graph.python_code("self").src

        def get_node(name):
            # [FUNCTION DOC] HELPER: Return the first graph node whose name equals the given
            # name; raises StopIteration if none exists.
            return next(n for n in graph.nodes if n.name == name)

        self.assertExpectedInline(
            graph_code(graph),
            """\



def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l_y_ = L_y_
    view = l_x_.view((10, 10))
    view_1 = l_x_.view((10, 10))
    view_2 = view.view((10, 10));  view = None
    view_3 = view_1.view((10, 10));  view_1 = None
    add_ = l_x_.add_(l_x_);  add_ = None
    add = view_2 + view_3;  view_2 = view_3 = None
    subgraph_0 = self.subgraph_0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)
    mul_ = l_y_.mul_(l_y_);  mul_ = None
    getitem = invoke_subgraph[0];  invoke_subgraph = None
    invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = l_y_ = None
    getitem_1 = invoke_subgraph_1[0];  invoke_subgraph_1 = None
    sum_1 = getitem.sum();  getitem = None
    add_1 = add + sum_1;  add = sum_1 = None
    sum_2 = getitem_1.sum();  getitem_1 = None
    add_2 = add_1 + sum_2;  add_1 = sum_2 = None
    return (add_2,)
    """,
        )

        # Shuffle nodes in the graph
        add_ = get_node("add_")
        mul_ = get_node("mul_")
        view_3 = get_node("view_3")
        view_3.append(mul_)
        add_node = get_node("add")
        add_node.append(add_)

        self.assertExpectedInline(
            graph_code(graph),
            """\



def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l_y_ = L_y_
    view = l_x_.view((10, 10))
    view_1 = l_x_.view((10, 10))
    view_2 = view.view((10, 10));  view = None
    view_3 = view_1.view((10, 10));  view_1 = None
    mul_ = l_y_.mul_(l_y_);  mul_ = None
    add = view_2 + view_3;  view_2 = view_3 = None
    add_ = l_x_.add_(l_x_);  add_ = None
    subgraph_0 = self.subgraph_0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)
    getitem = invoke_subgraph[0];  invoke_subgraph = None
    invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = l_y_ = None
    getitem_1 = invoke_subgraph_1[0];  invoke_subgraph_1 = None
    sum_1 = getitem.sum();  getitem = None
    add_1 = add + sum_1;  add = sum_1 = None
    sum_2 = getitem_1.sum();  getitem_1 = None
    add_2 = add_1 + sum_2;  add_1 = sum_2 = None
    return (add_2,)
    """,
        )
        _stable_topological_sort(
            graph, torch._dynamo.graph_deduplication.last_node_to_additional_deps
        )
        self.assertExpectedInline(
            graph_code(graph),
            """\



def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l_y_ = L_y_
    view = l_x_.view((10, 10))
    view_1 = l_x_.view((10, 10))
    view_2 = view.view((10, 10));  view = None
    view_3 = view_1.view((10, 10));  view_1 = None
    add = view_2 + view_3;  view_2 = view_3 = None
    add_ = l_x_.add_(l_x_);  add_ = None
    subgraph_0 = self.subgraph_0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_)
    mul_ = l_y_.mul_(l_y_);  mul_ = None
    getitem = invoke_subgraph[0];  invoke_subgraph = None
    invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = l_y_ = None
    getitem_1 = invoke_subgraph_1[0];  invoke_subgraph_1 = None
    sum_1 = getitem.sum();  getitem = None
    add_1 = add + sum_1;  add = sum_1 = None
    sum_2 = getitem_1.sum();  getitem_1 = None
    add_2 = add_1 + sum_2;  add_1 = sum_2 = None
    return (add_2,)
    """,
        )

    def test_tuple_return(self):
        # [FUNCTION DOC] A deduplicated region whose function returns a tuple from an
        # allow-in-graph function compiles under fullgraph with the eager backend; results
        # equal eager.
        @allow_in_graph
        def tuple_return(x, y):
            # [FUNCTION DOC] HELPER: Allow-in-graph function returning its two arguments as
            # a tuple.
            return x, y

        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Computes x doubled plus one and y
            # doubled plus one, then returns the tuple function applied to them.
            x0 = x + x + 1
            y0 = y + y + 1
            return tuple_return(x0, y0)

        def fn(x0, x1, x2, y0, y1, y2):
            # [FUNCTION DOC] HELPER: Function under test. Applies the inner function to three
            # pairs and returns the three results.
            x0 = inner_fn(x0, y0)
            x1 = inner_fn(x1, y1)
            x2 = inner_fn(x2, y2)
            return x0, x1, x2

        fn_opt = torch.compile(fn, fullgraph=True, backend="eager")
        inps = [torch.rand(10, 10) for _ in range(6)]
        result_compiled = fn_opt(*inps)
        result_eager = fn(*inps)
        self.assertEqual(result_compiled, result_eager)

    def test_tuple_inputs(self):
        # [FUNCTION DOC] With deduplication off but node tracking on, tuple-valued split
        # inputs are removed from the duplicates map and graph deduplication is applied
        # manually with a mock output graph; checks the resulting graph string.
        with (
            torch._dynamo.config.patch("use_graph_deduplication", False),
            torch._dynamo.config.patch("track_nodes_for_deduplication", True),
        ):

            def inner(x, y):
                # [FUNCTION DOC] HELPER: Function under test. Splits x in two and returns the
                # sum of the halves and y.
                x0, x1 = torch.split(x, 5)
                return x0 + x1 + y

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Function under test. Calls the inner function four
                # times and returns the sum of all outputs' reductions.
                o1 = inner(x, y)
                o2 = inner(x, y)
                o3 = inner(x, y)
                o4 = inner(x, y)
                return o1.sum() + o2.sum() + o3.sum() + o4.sum()

            graph, tracker = extract_graph_and_tracker(
                fn, torch.rand(10, 10), torch.rand(5, 10)
            )

            class MockOutputGraph:
                # [CLASS DOC] HELPER: Minimal stand-in for an output graph exposing the traced
                # graph, the region tracker and an empty fake root module.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Store the captured graph and tracker and create
                    # an empty fake root module for nn modules.
                    self.graph = graph
                    self.region_tracker = tracker
                    self.nn_modules = FakeRootModule({})

                def install_subgraph(self, name, subgraph):
                    # [FUNCTION DOC] HELPER: Accept a subgraph name and module and return an
                    # empty string without installing anything.
                    return ""

            splits = [
                n
                for n in graph.nodes
                if n.op == "call_function" and n.target is torch.split
            ]
            for split in splits:
                tracker.node_to_duplicates.pop(split)

            apply_graph_deduplication(MockOutputGraph())
            self.assertExpectedInline(
                graph,
                """\
graph():
    %_unnamed : [num_users=4] = get_attr[target=]
    %l_x_ : torch.Tensor [num_users=4] = placeholder[target=L_x_]
    %l_y_ : torch.Tensor [num_users=4] = placeholder[target=L_y_]
    %split : [num_users=2] = call_function[target=torch.functional.split](args = (%l_x_, 5), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    %split_1 : [num_users=2] = call_function[target=torch.functional.split](args = (%l_x_, 5), kwargs = {})
    %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%split_1, 0), kwargs = {})
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%split_1, 1), kwargs = {})
    %split_2 : [num_users=2] = call_function[target=torch.functional.split](args = (%l_x_, 5), kwargs = {})
    %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%split_2, 0), kwargs = {})
    %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%split_2, 1), kwargs = {})
    %split_3 : [num_users=2] = call_function[target=torch.functional.split](args = (%l_x_, 5), kwargs = {})
    %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%split_3, 0), kwargs = {})
    %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%split_3, 1), kwargs = {})
    %invoke_subgraph : [num_users=1] = call_function[target=torch.ops.higher_order.invoke_subgraph](args = (%_unnamed, , %getitem, %getitem_1, %l_y_), kwargs = {})
    %getitem_8 : [num_users=1] = call_function[target=operator.getitem](args = (%invoke_subgraph, 0), kwargs = {})
    %sum_1 : [num_users=1] = call_method[target=sum](args = (%getitem_8,), kwargs = {})
    %invoke_subgraph_1 : [num_users=1] = call_function[target=torch.ops.higher_order.invoke_subgraph](args = (%_unnamed, , %getitem_2, %getitem_3, %l_y_), kwargs = {})
    %getitem_9 : [num_users=1] = call_function[target=operator.getitem](args = (%invoke_subgraph_1, 0), kwargs = {})
    %sum_2 : [num_users=1] = call_method[target=sum](args = (%getitem_9,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=operator.add](args = (%sum_1, %sum_2), kwargs = {})
    %invoke_subgraph_2 : [num_users=1] = call_function[target=torch.ops.higher_order.invoke_subgraph](args = (%_unnamed, , %getitem_4, %getitem_5, %l_y_), kwargs = {})
    %getitem_10 : [num_users=1] = call_function[target=operator.getitem](args = (%invoke_subgraph_2, 0), kwargs = {})
    %sum_3 : [num_users=1] = call_method[target=sum](args = (%getitem_10,), kwargs = {})
    %add_9 : [num_users=1] = call_function[target=operator.add](args = (%add_8, %sum_3), kwargs = {})
    %invoke_subgraph_3 : [num_users=1] = call_function[target=torch.ops.higher_order.invoke_subgraph](args = (%_unnamed, , %getitem_6, %getitem_7, %l_y_), kwargs = {})
    %getitem_11 : [num_users=1] = call_function[target=operator.getitem](args = (%invoke_subgraph_3, 0), kwargs = {})
    %sum_4 : [num_users=1] = call_method[target=sum](args = (%getitem_11,), kwargs = {})
    %add_10 : [num_users=1] = call_function[target=operator.add](args = (%add_9, %sum_4), kwargs = {})
    return (add_10,)""",
            )

    def test_param_transfer_to_submodule(self):
        # [FUNCTION DOC] Parameters marked as static addresses can be passed into deduplicated
        # regions without error; only runs the compiled function.
        def inner_fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Returns x plus y plus y plus x.
            return x + y + y + x

        def fn(x0, x1, x2, y0, y1, y2):
            # [FUNCTION DOC] HELPER: Function under test. Applies the inner function to three
            # pairs and returns the sum of the reductions.
            x0 = inner_fn(x0, y0)
            x1 = inner_fn(x1, y1)
            x2 = inner_fn(x2, y2)
            return x0.sum() + x1.sum() + x2.sum()

        fn_opt = torch.compile(fn, fullgraph=True, backend="eager")
        args = [torch.rand(10, 10) for _ in range(6)]
        for arg in args:
            torch._dynamo.mark_static_address(arg)

        fn_opt(*args)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
