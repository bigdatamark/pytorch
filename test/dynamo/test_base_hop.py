# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for BaseHOP, the base class for simple higher-order ops that take a
# subgraph plus operands. Defines a test subclass (invoke_quant_test, one module-level
# instance) and checks Dynamo graph capture, schema generation (pytree in/out, mutation),
# None/int inputs, auto-functionalization, AOT forward/backward graphs and eager-call errors.
# Import-time: creates the invoke_quant_test instance.
import unittest.mock as mock

import torch
import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.schema import find_hop_schema
from torch.testing._internal.common_utils import instantiate_parametrized_tests


def normalize_graph(gm):
    # [FUNCTION DOC] HELPER: Returns the readable code of gm (without printing it) after
    # normalize_gm, giving a string suitable for comparing to expected-inline text.
    return normalize_gm(gm.print_readable(print_output=False))


class InvokeQuantTest(torch._higher_order_ops.BaseHOP):
    # [CLASS DOC] HELPER: BaseHOP subclass registered under the name invoke_quant_test whose
    # call signature takes a subgraph, positional operands and a required keyword scheme.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes the BaseHOP with the op name invoke_quant_test.
        super().__init__("invoke_quant_test")

    def __call__(self, subgraph, *operands, scheme):
        # [FUNCTION DOC] HELPER: Forwards subgraph, operands and the keyword scheme to the
        # BaseHOP call and returns its result.
        return super().__call__(subgraph, *operands, scheme=scheme)


invoke_quant_test = InvokeQuantTest()


class BaseHOPTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo TestCase for invoke_quant_test: graph capture, schema generation,
    # mutation handling, functionalization, AOT graphs and eager-call errors. Stateless.
    # TODO: flip to False later, we're landing a refactor PR and don't want to merge conflict
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_dynamo(self):
        # [FUNCTION DOC] With static shapes by default, compiles a function calling the op
        # on a matmul/sin/cos subgraph with scheme nf4. Asserts the result equals eager, that
        # exactly one graph was recorded, and that it matches the expected text (the op call
        # plus a subgraph_0 submodule).
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns cos(sin(x @ y)).
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        out = f(x, y)
        self.assertEqual(out, inner(x, y))

        if len(backend.graphs) != 1:
            raise AssertionError(f"Expected 1 graph, got {len(backend.graphs)}")
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            matmul: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            cos: "f32[3, 3]" = sin.cos();  sin = None
            return (cos,)
""",
        )

    def test_schema_gen_single_return(self):
        # [FUNCTION DOC] Compiles a call of the op whose subgraph returns one tensor, checks
        # equality with eager, then looks up the op schema in the recorded graph via
        # find_hop_schema. Asserts one schema is found with two tensor args, keyword scheme
        # defaulting to nf4 and a single tensor return.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns cos(sin(x @ y)).
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=False)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        out = f(x.clone(), y)
        self.assertEqual(out, inner(x.clone(), y))
        schemas = find_hop_schema(backend.graphs[0], invoke_quant_test)
        self.assertEqual(len(schemas), 1)
        self.assertExpectedInline(
            str(schemas[0]),
            """invoke_quant_test(Any subgraph, Tensor arg0, Tensor arg1, *, str scheme="nf4") -> ((Tensor))""",
        )

    def test_schema_gen_pytree_in_out(self):
        # [FUNCTION DOC] Calls the op with a list input [x, y] and a subgraph returning a nested
        # list/tuple/dict structure. Checks equality with eager and that the generated schema
        # flattens inputs to two tensors (arg0, arg1) and outputs to four tensors.
        def inner(x_y):
            # [FUNCTION DOC] HELPER: Unpacks the pair and returns a list of cos(sin(x@y)),
            # a tuple (x+y, x-y) and a dict with key "out" holding a one-item tuple of x@y.
            x, y = x_y
            return [
                (x @ y).sin().cos(),
                (x + y, x - y),
                {"out": (x @ y,)},
            ]

        # make x not require grad because we want to inplace mutate it
        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with the list [x, y].
            return invoke_quant_test(inner, [x, y], scheme="nf4")

        out = f(x.clone(), y)
        self.assertEqual(out, inner([x.clone(), y]))
        schemas = find_hop_schema(backend.graphs[0], invoke_quant_test)
        self.assertEqual(len(schemas), 1)
        self.assertExpectedInline(
            str(schemas[0]),
            """invoke_quant_test(Any subgraph, Tensor arg0, Tensor arg1, *, str scheme="nf4") -> (Tensor, Tensor, Tensor, Tensor)""",
        )

    def test_schema_gen_single_return_with_mutation(self):
        # [FUNCTION DOC] With BaseHOPVariable.supports_input_mutation patched to True, compiles
        # (fullgraph) a call whose subgraph mutates both inputs in place. Asserts one graph
        # with the expected text and a schema that marks both tensor args as mutated (alias
        # annotations a1! and a2!) with a single tensor return.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Adds 1 to x in place, multiplies y by -1 in place, returns
            # cos(sin(x @ y)).
            x.add_(1)
            y.mul_(-1)
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=False)

        backend = EagerAndRecordGraphs()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        with mock.patch(
            "torch._dynamo.variables.higher_order_ops.BaseHOPVariable.supports_input_mutation",
            True,
        ):
            torch.compile(f, backend=backend, fullgraph=True)(x.clone(), y)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            add_: "f32[3, 3]" = l_x_.add_(1);  add_ = None

            mul_: "f32[3, 3]" = l_y_.mul_(-1);  mul_ = None

            matmul: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            cos: "f32[3, 3]" = sin.cos();  sin = None
            return (cos,)
""",
        )
        self.assertExpectedInline(
            str(find_hop_schema(backend.graphs[0], invoke_quant_test)[0]),
            """invoke_quant_test(Any subgraph, Tensor(a1!) arg0, Tensor(a2!) arg1, *, str scheme="nf4") -> ((Tensor))""",
        )

    def test_schema_gen_pytree_in_out_with_mutation(self):
        # [FUNCTION DOC] Combines list input, nested output and in-place mutation of the first
        # item: compiles under no_grad with input mutation support patched on, and asserts one
        # recorded graph with the expected text. The schema must mark only arg0 as mutated and
        # return four tensors.
        def inner(x_y):
            # [FUNCTION DOC] HELPER: Unpacks the pair, adds 1 to x in place, returns a list of
            # cos(sin(x@y)), a tuple (x+y, x-y) and a dict with "out" holding a tuple of x@y.
            x, y = x_y
            x.add_(1)
            return [
                (x @ y).sin().cos(),
                (x + y, x - y),
                {"out": (x @ y,)},
            ]

        # make x not require grad because we want to inplace mutate it
        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=True)

        bk = EagerAndRecordGraphs()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with the list [x, y].
            return invoke_quant_test(inner, [x, y], scheme="nf4")

        with (
            mock.patch(
                "torch._dynamo.variables.higher_order_ops.BaseHOPVariable.supports_input_mutation",
                True,
            ),
            torch.no_grad(),
        ):
            torch.compile(f, backend=bk, fullgraph=True)(x.clone(), y)

        self.assertEqual(len(bk.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(bk.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0]
        getitem_1: "f32[3, 3]" = invoke_quant_test[1]
        getitem_2: "f32[3, 3]" = invoke_quant_test[2]
        getitem_3: "f32[3, 3]" = invoke_quant_test[3];  invoke_quant_test = None
        return (getitem, getitem_1, getitem_2, getitem_3)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            add_: "f32[3, 3]" = l_x_.add_(1);  add_ = None

            matmul: "f32[3, 3]" = l_x_ @ l_y_
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            cos: "f32[3, 3]" = sin.cos();  sin = None

            add: "f32[3, 3]" = l_x_ + l_y_
            sub: "f32[3, 3]" = l_x_ - l_y_

            matmul_1: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            return (cos, add, sub, matmul_1)
""",
        )
        self.assertExpectedInline(
            str(find_hop_schema(bk.graphs[0], invoke_quant_test)[0]),
            """invoke_quant_test(Any subgraph, Tensor(a1!) arg0, Tensor arg1, *, str scheme="nf4") -> (Tensor, Tensor, Tensor, Tensor)""",
        )

    def test_none_input(self):
        # [FUNCTION DOC] Passes None as the first operand under fullgraph compile. Asserts
        # the result equals eager and the graph takes only y: the None is specialized away and
        # the subgraph reduces to cos of y.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns sin(y) if x is not None, otherwise cos(y).
            if x is not None:
                return y.sin()
            return y.cos()

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = None
        y = torch.randn(3, 4)
        out = f(x, y)
        self.assertEqual(out, inner(x, y))
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[3, 4]"):
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_y_, scheme = 'nf4');  subgraph_0 = l_y_ = None
        getitem: "f32[3, 4]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_y_: "f32[3, 4]"):
            cos: "f32[3, 4]" = l_y_.cos();  l_y_ = None
            return (cos,)
""",
        )

    def test_int_input(self):
        # [FUNCTION DOC] Passes the Python int 1 as the first operand under fullgraph compile.
        # Asserts the result equals eager and the graph input list contains only y, with the
        # constant 1 inlined into the subgraph's add.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns x + y.
            return x + y

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = 1
        y = torch.randn(3, 4)
        out = f(x, y)
        self.assertEqual(out, inner(x, y))
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[3, 4]"):
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_y_, scheme = 'nf4');  subgraph_0 = l_y_ = None
        getitem: "f32[3, 4]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_y_: "f32[3, 4]"):
            add: "f32[3, 4]" = 1 + l_y_;  l_y_ = None
            return (add,)
""",
        )

    def test_auto_functionalize(self):
        # [FUNCTION DOC] Compiles with the aot_eager recording backend, no_grad and input
        # mutation support patched on, a call whose subgraph mutates x in place. Asserts x was
        # incremented by 1 in the caller, the output equals x_clone + y + 1, one forward graph
        # was recorded, and it uses auto_functionalized_v2 followed by a copy_ back into x.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Adds 1 to x in place, returns x + y.
            x.add_(1)
            return x + y

        backend = AotEagerAndRecordGraphs()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = torch.randn(3, 3, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(3, 3, requires_grad=True)
        with (
            mock.patch(
                "torch._dynamo.variables.higher_order_ops.BaseHOPVariable.supports_input_mutation",
                True,
            ),
            torch.no_grad(),
        ):
            compiled_out = torch.compile(f, backend=backend, fullgraph=True)(x, y)
        self.assertEqual(x, x_clone + 1)
        self.assertEqual(compiled_out, x_clone + y + 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.invoke_quant_test, subgraph = auto_functionalized_subgraph_0, arg1 = arg1_1, scheme = 'nf4', _arg0_base_index = 0, _all_bases = [arg0_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = arg1_1 = _tree_spec_constant0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0]
        getitem_1: "f32[3, 3]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[3, 3]" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return (getitem,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, arg1_1);  arg1_1 = None
            copy_: "f32[3, 3]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (add_1,)
""",
        )

    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_aot_eager(self):
        # [FUNCTION DOC] With static shapes by default, compiles with the aot_eager recording
        # backend a call with a subgraph using an in-place sin, then computes gradients of the
        # output with respect to x using y as the grad output and compares to eager gradients.
        # Asserts exactly one forward and one backward graph, each matching expected text
        # (forward saves primals; backward re-runs the op on a subgraph with tangents).
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns cos of the in-place sin of x @ y.
            return (x @ y).sin_().cos()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        backend = AotEagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on inner with x, y, scheme nf4.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        out = f(x, y)
        result = torch.autograd.grad(out, x, y)
        out = inner(x, y)
        expected = torch.autograd.grad(out, x, y)
        self.assertEqual(result, expected)

        if len(backend.fw_graphs) != 1:
            raise AssertionError(f"Expected 1 fw_graph, got {len(backend.fw_graphs)}")
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]"):
        subgraph0 = self.subgraph0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph0, primals_1, primals_2, scheme = 'nf4');  subgraph0 = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem, primals_1, primals_2)

    class subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(mm);  mm = None
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(sin);  sin = None
            return (cos,)
""",
        )

        if len(backend.bw_graphs) != 1:
            raise AssertionError(f"Expected 1 bw_graph, got {len(backend.bw_graphs)}")
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        subgraph1 = self.subgraph1
        invoke_quant_test_1 = torch.ops.higher_order.invoke_quant_test(subgraph1, primals_1, primals_2, tangents_1, scheme = 'nf4');  subgraph1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 3]" = invoke_quant_test_1[0]
        getitem_2: "f32[3, 3]" = invoke_quant_test_1[1];  invoke_quant_test_1 = None
        return (getitem_1, getitem_2)

    class subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
            clone: "f32[3, 3]" = torch.ops.aten.clone.default(mm)
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(mm);  mm = None
            sin_1: "f32[3, 3]" = torch.ops.aten.sin.default(sin);  sin = None
            neg: "f32[3, 3]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg2_1, neg);  arg2_1 = neg = None
            cos_1: "f32[3, 3]" = torch.ops.aten.cos.default(clone);  clone = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(mul, cos_1);  mul = cos_1 = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t, mul_1);  t = None
            t_1: "f32[3, 3]" = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
            mm_2: "f32[3, 3]" = torch.ops.aten.mm.default(mul_1, t_1);  mul_1 = t_1 = None
            return (mm_2, mm_1)
""",
        )

    def test_aliasing_mutation_error(self):
        # [FUNCTION DOC] Without enabling mutation support, asserts fullgraph compile raises
        # RuntimeError "Encountered aliasing during higher order op tracing" when the subgraph
        # returns its input, and "Encountered input mutation ..." when it mutates an input.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns x itself (output aliases an input).
            return x

        def inner2(x, y):
            # [FUNCTION DOC] HELPER: Applies sin to x in place (input mutation), returns x + y.
            x.sin_()
            return x + y

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(inner, x, y):
            # [FUNCTION DOC] HELPER: Calls invoke_quant_test on the given subgraph function.
            return invoke_quant_test(inner, x, y, scheme="nf4")

        with self.assertRaisesRegex(
            RuntimeError, "Encountered aliasing during higher order op tracing"
        ):
            f(inner, x, y)

        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered input mutation during higher order op tracing",
        ):
            f(inner2, x, y)

    def test_eager_call(self):
        # [FUNCTION DOC] Calls the op eagerly (no compile). Asserts passing a plain Python
        # function raises RuntimeError mentioning torch.fx.GraphModule, then smoke-tests that
        # passing a make_fx traced GraphModule of the same function is accepted.
        def inner(x, y):
            # [FUNCTION DOC] HELPER: Returns x + y.
            return x + y

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with self.assertRaisesRegex(RuntimeError, "torch.fx.GraphModule"):
            invoke_quant_test(inner, x, y, scheme="nf4")

        from functorch import make_fx

        result = make_fx(inner)(x, y)
        # smoke test
        invoke_quant_test(result, x, y, scheme="nf4")


instantiate_parametrized_tests(BaseHOPTest)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
