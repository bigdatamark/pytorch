# Owner(s): ["oncall: pt2"]
# [MODULE DOC] Tests for the leaf_function decorator, which marks a Python function as opaque to
# tracing so that tracers emit a single invoke_leaf_function higher-order-op node instead of
# tracing its body; a separately registered fake implementation supplies output metadata. The
# module covers make_fx, aot_function (forward, backward, gradients, mutation, closures, escaped
# gradients) and torch.compile (many backends, pytrees, modules, nesting, errors). Many tests
# compare printed graphs against expected strings. Import-time side effects: none beyond imports.

"""Tests for @leaf_function with make_fx, aot_function, and torch.compile."""

import contextlib
import copy
import io
import re
from functools import partial
from unittest.mock import patch

import torch
import torch._dynamo.config as config
import torch._dynamo.testing
from functorch.compile import aot_function, nop
from torch._dynamo.decorators import leaf_function
from torch._dynamo.testing import normalize_gm
from torch._higher_order_ops.invoke_leaf_function import invoke_leaf_function
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    skipIfXpu,
    TestCase,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def extract_graph(fx_g, _, graph_cell):
    # [FUNCTION DOC] HELPER: Compiler callback for aot_function. Stores the given graph module in
    # the first slot of the supplied one-element list graph_cell and returns it unchanged, so tests
    # can inspect the forward or backward graph afterwards.
    graph_cell[0] = fx_g
    return fx_g


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFx(TestCase):
    # [CLASS DOC] Tests for leaf_function traced with make_fx: graph structure (a single
    # invoke_leaf_function node), closures, data-dependent outputs, pytree inputs, modules, missing
    # fake implementations and input mutation. Stateless apart from a graph-check helper.
    def _has_invoke_leaf_function_node(self, gm):
        # [FUNCTION DOC] HELPER: Returns true if any call_function node of the given graph module
        # targets the invoke_leaf_function higher-order op, otherwise false.
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is invoke_leaf_function:
                return True
        return False

    def test_make_fx_simple(self):
        # [FUNCTION DOC] Traces a leaf function (with a branch on the sign of the input sum) and a
        # fake returning the sum of inputs with make_fx. Asserts the graph holds an
        # invoke_leaf_function node (compared with an expected printout) and that the traced module
        # agrees with eager on fresh inputs.
        @leaf_function
        def my_fn(x, y):
            # [FUNCTION DOC] HELPER: Real leaf body; returns a one-tuple with the sum when the
            # input sum is positive, else the difference.
            if x.sum() > 0:
                return (x + y,)
            return (x - y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake implementation; returns a one-tuple with the
            # sum of inputs.
            return (x + y,)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Traced wrapper that calls the leaf function.
            return my_fn(x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_with_closure(self):
        # [FUNCTION DOC] Checks a leaf function reading a value from a closure dictionary: the
        # graph shape matches the expected printout, and after the closure value is changed
        # the traced module still matches eager, showing the real body runs at runtime and is
        # not baked in.
        config = {"scale": 2.0}

        @leaf_function
        def scaled_fn(x):
            # [FUNCTION DOC] HELPER: Real leaf body; returns the input scaled by the closure value.
            return (x * config["scale"],)

        @scaled_fn.register_fake
        def scaled_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an empty tensor shaped like the input
            # in a tuple.
            return (torch.empty_like(x),)

        def f(x):
            # [FUNCTION DOC] HELPER: Traced wrapper calling the leaf function.
            return scaled_fn(x)

        x = torch.randn(3, 3)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        # Closure change reflected at runtime
        config["scale"] = 3.0
        x2 = torch.randn(3, 3)
        self.assertEqual(gm(x2), f(x2))

    def test_make_fx_data_dependent(self):
        # [FUNCTION DOC] Leaf function with a data-dependent output shape (nonzero indices).
        # Asserts make_fx produces the invoke node and that both outputs of the traced module equal
        # eager outputs on fresh input.
        @leaf_function
        def data_dep_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns the relu of the input and the indices of
            # its positive entries.
            nz = (x > 0).nonzero()
            return (x.relu(), nz)

        @data_dep_fn.register_fake
        def data_dep_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; computes the same two outputs as the real body.
            return (x.relu(), (x > 0).nonzero())

        def f(x):
            # [FUNCTION DOC] HELPER: Traced wrapper calling the leaf function.
            return data_dep_fn(x)

        x = torch.randn(4, 4)
        gm = make_fx(f)(x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        x2 = torch.randn(4, 4)
        eager_out = f(x2)
        gm_out = gm(x2)
        self.assertEqual(gm_out[0], eager_out[0])
        self.assertEqual(gm_out[1], eager_out[1])

    def test_make_fx_pytree_inputs(self):
        # [FUNCTION DOC] Leaf function taking a dictionary of tensors as its single argument.
        # Checks the invoke node and expected printout, and equality with eager on new inputs.
        @leaf_function
        def pytree_fn(inputs):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the sum of
            # two dict entries.
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            # [FUNCTION DOC] HELPER: Fake; same computation as the real body.
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Builds a dictionary from the two inputs and calls
            # the leaf function.
            return pytree_fn({"x": x, "y": y})

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_no_fake_impl(self):
        # [FUNCTION DOC] A leaf function without a registered fake must make tracing fail; asserts
        # that make_fx raises an error whose message mentions a required fake implementation.
        @leaf_function
        def no_fake_fn(x):
            # [FUNCTION DOC] HELPER: Leaf body doubling its input; deliberately has
            # no fake registered.
            return (x * 2,)

        def f(x):
            # [FUNCTION DOC] HELPER: Traced wrapper calling the leaf function.
            return no_fake_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            make_fx(f)(x)

    def test_make_fx_with_module(self):
        # [FUNCTION DOC] Leaf function receiving a Linear module as an argument. The traced wrapper
        # reparametrizes the module with the weight and bias inputs. Asserts the invoke node exists
        # and that with new weights, bias and input the traced module matches eager.
        @leaf_function
        def mod_fn(mod, x):
            # [FUNCTION DOC] HELPER: Real body; applies the given module to the input, returned
            # in a tuple.
            return (mod(x),)

        @mod_fn.register_fake
        def mod_fn_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; same as the real body.
            return (mod(x),)

        linear = torch.nn.Linear(3, 3)

        def f(w, b, x):
            # [FUNCTION DOC] HELPER: Runs the leaf function on the module with its weight and bias
            # temporarily replaced by the supplied tensors (stateless reparametrization).
            with torch.nn.utils.stateless._reparametrize_module(
                linear, {"weight": w, "bias": b}
            ):
                return mod_fn(linear, x)

        w = linear.weight.detach().clone()
        b = linear.bias.detach().clone()
        x = torch.randn(2, 3)
        gm = make_fx(f)(w, b, x)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        w2 = torch.randn(3, 3)
        b2 = torch.randn(3)
        x2 = torch.randn(2, 3)
        self.assertEqual(gm(w2, b2, x2), f(w2, b2, x2))

    def test_make_fx_over_invoke_leaf_function_hop(self):
        # [FUNCTION DOC] Calls the invoke_leaf_function higher-order op directly, with callables
        # wrapped for real and fake bodies and an input spec from flattening the arguments, under
        # make_fx. Asserts a single opaque invoke node appears (expected printout) and the traced
        # module matches eager.
        """Tracing invoke_leaf_function HOP directly with make_fx should produce
        a single opaque invoke_leaf_function node, not trace into its internals."""
        import torch.utils._pytree as pytree
        from torch._higher_order_ops.invoke_leaf_function import _LeafCallable

        def real_fn(x, y):
            # [FUNCTION DOC] HELPER: Real callable; returns a one-tuple with product
            # plus first input.
            return (x * y + x,)

        def fake_fn(x, y):
            # [FUNCTION DOC] HELPER: Fake callable; same computation as the real one.
            return (x * y + x,)

        real_fn_callable = _LeafCallable(real_fn)
        fake_fn_callable = _LeafCallable(fake_fn)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Flattens the arguments to get an input spec and
            # calls the higher-order op with the wrapped real and fake callables and an
            # empty string argument.
            _, input_spec = pytree.tree_flatten(((x, y), {}))
            return invoke_leaf_function(
                real_fn_callable, fake_fn_callable, input_spec, "", x, y
            )

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        gm = make_fx(f)(x, y)

        self.assertTrue(self._has_invoke_leaf_function_node(gm))
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', x_1, y_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = x_1 = y_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        self.assertEqual(gm(x2, y2), f(x2, y2))

    def test_make_fx_input_mutation(self):
        # [FUNCTION DOC] Leaf function declared to mutate the argument buf. Compares eager output
        # and mutated buffer against the traced module run on a fresh clone; asserts the invoke
        # node exists, outputs agree, and the buffer ends up equal in both runs.
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            # [FUNCTION DOC] HELPER: Real body; adds one to buf in place and returns
            # input plus buf.
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            # [FUNCTION DOC] HELPER: Fake; performs the same in-place mutation and sum.
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            # [FUNCTION DOC] HELPER: Traced wrapper calling the leaf function with both arguments.
            return mutate_fn(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        buf_fx = buf.clone()
        gm = make_fx(f)(x, buf_fx)
        self.assertTrue(self._has_invoke_leaf_function_node(gm))

        buf_gm = buf.clone()
        gm_out = gm(x, buf_gm)
        self.assertEqual(gm_out, eager_out)
        self.assertEqual(buf_gm, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionAotFunction(TestCase):
    # [CLASS DOC] Tests for leaf_function under aot_function: expected forward and backward
    # graphs (with effect tokens), gradient parity with eager, closures, pytree inputs and
    # input mutation. Stateless.
    def test_aot_function_simple(self):
        # [FUNCTION DOC] Compiles a function using a matrix-multiply leaf function with
        # graph-capturing compilers, then checks output and both input gradients equal eager
        # results, and compares printed forward and backward graphs against expected strings
        # (forward threads an effect token through with_effects around the invoke node).
        @leaf_function
        def my_fn(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the matrix product.
            return (x @ y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; same matrix product.
            return (x @ y,)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return my_fn(x, y)[0]

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        )

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        self.assertExpectedInline(
            normalize_gm(fw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_2, primals_3, requires_grad_indices = '0,1');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_2 = primals_3 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            normalize_gm(bw_graph_cell[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_3: "f32[3, 3]" = with_effects_1[1]
        getitem_4: "f32[3, 3]" = with_effects_1[2];  with_effects_1 = None
        return (getitem_3, getitem_4, getitem_2)
""",
        )

    def test_aot_function_gradients(self):
        # [FUNCTION DOC] Compiles a function summing a leaf function output with no-op compilers;
        # checks the output and the gradients of both inputs match eager after backward.
        @leaf_function
        def my_fn(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with product plus first input.
            return (x * y + x,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; same computation.
            return (x * y + x,)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the first output of the leaf function.
            return my_fn(x, y)[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_closure(self):
        # [FUNCTION DOC] Leaf function whose behavior depends on a closure flag (doubles or triples
        # the matrix product). Runs eager and compiled with the flag true then false, checking
        # outputs and gradients each time, and that results differ across flag values, showing the
        # real body is re-run each call with current closure state.
        config = {"use_double": True}

        @leaf_function
        def closure_fn(x, y):
            # [FUNCTION DOC] HELPER: Real body; multiplies the matrix product by two or three
            # according to the closure flag.
            if config["use_double"]:
                return (x @ y * 2,)
            return (x @ y * 3,)

        @closure_fn.register_fake
        def closure_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns a plain matrix product (shape only matters).
            return (x @ y,)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return closure_fn(x, y)[0]

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        x_val = torch.randn(3, 3)
        y_val = torch.randn(3, 3)

        # Test with use_double=True
        config["use_double"] = True
        x = x_val.clone().requires_grad_(True)
        y = y_val.clone().requires_grad_(True)
        x_clone = x_val.clone().requires_grad_(True)
        y_clone = y_val.clone().requires_grad_(True)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.sum().backward()
        out_compiled.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Change closure and verify result changes
        config["use_double"] = False
        x2 = x_val.clone().requires_grad_(True)
        y2 = y_val.clone().requires_grad_(True)
        x2_clone = x_val.clone().requires_grad_(True)
        y2_clone = y_val.clone().requires_grad_(True)

        out_eager2 = f(x2, y2)
        out_compiled2 = compiled_f(x2_clone, y2_clone)
        self.assertEqual(out_eager2, out_compiled2)

        out_eager2.sum().backward()
        out_compiled2.sum().backward()
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        # Different closures => different results
        self.assertNotEqual(out_eager, out_eager2)

    def test_aot_function_pytree_inputs(self):
        # [FUNCTION DOC] Leaf function with a dictionary argument under aot_function with no-op
        # compilers; checks output and gradients of both tensors equal eager.
        @leaf_function
        def pytree_fn(inputs):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the sum of
            # two dict entries.
            return (inputs["x"] + inputs["y"],)

        @pytree_fn.register_fake
        def pytree_fn_fake(inputs):
            # [FUNCTION DOC] HELPER: Fake; same computation.
            return (inputs["x"] + inputs["y"],)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Wraps inputs in a dictionary, calls the leaf function and
            # sums the result.
            return pytree_fn({"x": x, "y": y})[0].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)

        out_eager = f(x, y)
        out_compiled = compiled_f(x_clone, y_clone)
        self.assertEqual(out_eager, out_compiled)

        out_eager.backward()
        out_compiled.backward()
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function_input_mutation(self):
        # [FUNCTION DOC] Leaf function mutating buf under aot_function with a no-op forward
        # compiler; asserts the compiled output and the mutated buffer both equal eager.
        @leaf_function(mutates_args={"buf"})
        def mutate_fn(x, buf):
            # [FUNCTION DOC] HELPER: Real body; adds one to buf in place and returns
            # input plus buf.
            buf.add_(1)
            return (x + buf,)

        @mutate_fn.register_fake
        def mutate_fn_fake(x, buf):
            # [FUNCTION DOC] HELPER: Fake; same in-place update and sum.
            buf.add_(1)
            return (x + buf,)

        def f(x, buf):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return mutate_fn(x, buf)[0]

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        eager_out = f(x, buf_eager)

        compiled_f = aot_function(f, nop)
        buf_compiled = buf.clone()
        compiled_out = compiled_f(x, buf_compiled)
        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(buf_compiled, buf_eager)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionEscapedGradients(TestCase):
    # [CLASS DOC] Tests for the escaped-gradient check: leaf functions that use tensors captured
    # from closures that require grad would silently lose gradients. The config flag
    # leaf_function_check_escaped_gradients controls whether this raises. Stateless.
    def test_aot_function_escaped_gradient_multiple_closures(self):
        # [FUNCTION DOC] With the check enabled, a leaf function closing over two
        # grad-requiring weights compiled by aot_function raises a runtime error whose message
        # mentions two tensors.
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies input by each closed-over
            # weight and adds.
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor with
            # matching output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return uses_multiple_closures(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                compiled_f(x)

    def test_aot_function_escaped_gradient_disabled(self):
        # [FUNCTION DOC] With the check disabled via config, a leaf function over a grad-requiring
        # closure weight compiles and runs under aot_function; asserts output shape is 2 by 3.
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=False):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_input_no_grad(self):
        # [FUNCTION DOC] With the check enabled but an input that does not require grad, no error
        # should be raised; asserts the compiled result has shape 2 by 3.
        """No false positive when input doesn't require grad."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=False)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        with config.patch(leaf_function_check_escaped_gradients=True):
            result = compiled_f(x)
            self.assertEqual(result.shape, (2, 3))

    def test_aot_function_escaped_gradient_actually_lost(self):
        # [FUNCTION DOC] Runs forward and backward with default config and shows that the input
        # receives a gradient while the closed-over weight does not (its grad stays unset).
        """Verify gradients don't flow to closure tensors."""
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return uses_closure(x)[0]

        x = torch.randn(2, 3, requires_grad=True)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        result = compiled_f(x)
        result.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_make_fx_escaped_gradient_multiple_closures(self):
        # [FUNCTION DOC] With the check enabled, tracing with make_fx a leaf function closing over
        # two grad-requiring weights raises a runtime error mentioning two tensors.
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies input by each closed-over
            # weight and adds.
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the full leaf function result tuple.
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                make_fx(f)(x)

    def test_make_fx_escaped_gradient_disabled(self):
        # [FUNCTION DOC] With the check disabled, make_fx tracing succeeds and the traced module,
        # run on a new input, returns an output of shape 2 by 3.
        weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the full leaf function result tuple.
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        with config.patch(leaf_function_check_escaped_gradients=False):
            gm = make_fx(f)(x)
            x2 = torch.randn(2, 3)
            result = gm(x2)
            self.assertEqual(result[0].shape, (2, 3))


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionMakeFxAndCompile(TestCase):
    # [CLASS DOC] Tests mixing leaf_function with torch.compile and make_fx: call
    # counts across compilation, nesting of compiled regions inside traced ones, and
    # related behaviors. Stateless.
    """Tests for @leaf_function when mixing torch.compile and make_fx."""

    def test_not_called_during_compilation(self):
        # [FUNCTION DOC] After resetting dynamo, compiles with the eager backend (full graph) a
        # function calling a counting leaf function. Asserts the real body was run once on the
        # first call (at runtime, not during compilation) and exactly once more on the second call,
        # each with correct results.
        """The real leaf_fn body runs only at runtime, not during compilation."""
        torch._dynamo.reset()
        call_count = 0

        @leaf_function
        def my_leaf(x, y):
            # [FUNCTION DOC] HELPER: Real body; increments the enclosing call counter and returns a
            # one-tuple with the sum of inputs.
            nonlocal call_count
            call_count += 1
            return (x + y,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the first element of the leaf function result.
            return my_leaf(x, y)[0]

        compiled_f = torch.compile(f, backend="eager", fullgraph=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Compilation + first call
        result = compiled_f(x, y)
        self.assertEqual(call_count, 1)
        self.assertEqual(result, x + y)

        # Second call reuses compiled code, leaf_fn called again exactly once
        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        result2 = compiled_f(x2, y2)
        self.assertEqual(call_count, 2)
        self.assertEqual(result2, x2 + y2)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_compile(self):
        # [FUNCTION DOC] With force_compile_during_fx_trace enabled via config, a leaf function
        # used only inside a function compiled by torch.compile with the invoke_subgraph
        # backend, which is called from an outer function traced by make_fx in fake mode.
        # Asserts the expected printout (an invoke_subgraph node whose subgraph wraps the leaf
        # invoke node with effects) and that the traced module computes the expected arithmetic
        # result on new inputs.
        """Leaf function only inside the torch.compile'd region."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with product plus first input.
            return (x * y + x,)

        @my_leaf.register_fake
        def my_leaf_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; same computation.
            return (x * y + x,)

        def inner(x, y):
            # [FUNCTION DOC] HELPER: Region compiled with torch.compile; returns the
            # first leaf output.
            return my_leaf(x, y)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x, y):
            # [FUNCTION DOC] HELPER: Adds one to the input, runs the compiled
            # region, subtracts one.
            z = x + 1
            return compiled(z, y) - 1

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(getitem_1, 1);  getitem_1 = None
        return sub

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, arg2_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = arg2_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",
        )

        x2 = torch.randn(3, 3)
        y2 = torch.randn(3, 3)
        expected = (x2 + 1) * y2 + (x2 + 1) - 1
        self.assertEqual(traced(x2, y2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_only_in_make_fx(self):
        # [FUNCTION DOC] Leaf function used only in the make_fx-traced outer code, while a plain
        # function is compiled with the invoke_subgraph backend. Asserts the expected printout
        # (subgraph node, then the leaf invoke node) and that the traced result equals the expected
        # tuple holding twice the input plus one.
        """Leaf function only in the make_fx-traced code, outside torch.compile."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        def inner(x):
            # [FUNCTION DOC] HELPER: Compiled region body; adds one to the input.
            return x + 1

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            # [FUNCTION DOC] HELPER: Runs the compiled region, then the leaf function
            # on its result.
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem = None
        getitem_1: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_1,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            return (add,)
""",
        )

        x2 = torch.randn(3, 3)
        expected = ((x2 + 1) * 2,)
        self.assertEqual(traced(x2), expected)

    @config.patch(force_compile_during_fx_trace=True)
    def test_leaf_fn_in_both_compile_and_make_fx(self):
        # [FUNCTION DOC] The same leaf function is called both inside a torch.compile region
        # (invoke_subgraph backend) and in the make_fx-traced outer code. Asserts the expected
        # printout: the subgraph holds the effect-wrapped leaf invoke and the outer graph holds a
        # plain leaf invoke node.
        """Same leaf function used inside torch.compile and in make_fx-traced code."""
        torch._dynamo.reset()

        @leaf_function
        def my_leaf(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_leaf.register_fake
        def my_leaf_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        def inner(x):
            # [FUNCTION DOC] HELPER: Compiled region body; returns the first output of
            # the leaf function.
            return my_leaf(x)[0]

        compiled = torch.compile(inner, backend="invoke_subgraph")

        def outer(x):
            # [FUNCTION DOC] HELPER: Runs the compiled region, then the leaf function
            # on its result.
            y = compiled(x)
            return my_leaf(y)

        x = torch.randn(3, 3)
        traced = make_fx(
            outer, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', None, x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[0]" = invoke_subgraph[0];  getitem = None
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', getitem_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = getitem_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem_2,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1, arg1_1: "f32[3, 3]"):
            _opaque_obj0 = self._opaque_obj0
            _opaque_obj1 = self._opaque_obj1
            _tree_spec_constant0 = self._tree_spec_constant0
            with_effects = torch.ops.higher_order.with_effects(None, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', arg1_1, requires_grad_indices = '');  _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = arg1_1 = None
            getitem: "f32[0]" = with_effects[0]
            getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
            return (getitem, getitem_1)
""",
        )


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionDynamo(PytreeRegisteringTestCase):
    # [CLASS DOC] Tests for leaf_function under torch.compile with dynamo, comparing eager,
    # eager-backend compile and aot_eager compile, including expected dynamo, forward and backward
    # graphs. The class-level setup turns on canonicalized output graph node order in the dynamo
    # config and restores it afterwards. Provides helper methods for model comparison.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup; runs the parent setup, then sets the dynamo config option
        # canonicalize_output_graph_node_order to true (global state, undone in tearDownClass).
        super().setUpClass()
        config.canonicalize_output_graph_node_order = True

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown; sets the dynamo config option
        # canonicalize_output_graph_node_order back to false, then runs the parent teardown.
        config.canonicalize_output_graph_node_order = False
        super().tearDownClass()

    def _assert_models_equal(
        self,
        model_expected,
        model_test,
        x_expected,
        x_test,
    ):
        # [FUNCTION DOC] HELPER: Runs both models on their respective inputs and asserts outputs
        # equal; backpropagates the sum of each output, asserts input gradients equal, and asserts
        # both models have the same parameter names and, for each parameter whose expected gradient
        # exists, equal gradients (with a message naming the parameter).
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

        expected_grads = {
            name: param.grad for name, param in model_expected.named_parameters()
        }
        test_grads = {name: param.grad for name, param in model_test.named_parameters()}

        self.assertEqual(set(expected_grads.keys()), set(test_grads.keys()))
        for name in expected_grads:
            if expected_grads[name] is not None:
                self.assertEqual(
                    expected_grads[name],
                    test_grads[name],
                    msg=lambda msg: f"{msg}\nGradient mismatch for parameter {name}",
                )

    def _test_leaf_function_helper(self, mod_class, args_fn, loss_fn):
        # [FUNCTION DOC] HELPER: Core comparison driver. Creates three instances of the module
        # class (plain eager, compiled with the eager recording backend, compiled with the
        # aot-eager recording backend), the last two initialized from the first one's state. Both
        # compiled with full graph. For two iterations it zeroes gradients, builds arguments
        # through args_fn and two cloned copies, runs forward and backward with the loss function
        # on each variant, and asserts outputs, parameter gradients and tensor argument gradients
        # agree. Returns a tuple of three normalized printed graphs: the dynamo graph, the aot
        # forward graph and the aot backward graph (first recorded graph of each).
        import torch.utils._pytree as pytree
        from torch._dynamo.testing import AotEagerAndRecordGraphs, EagerAndRecordGraphs

        mod_eager = mod_class()
        mod_compile_eager = mod_class()
        mod_compile_eager.load_state_dict(dict(mod_eager.state_dict()))
        mod_compile_aot = mod_class()
        mod_compile_aot.load_state_dict(dict(mod_eager.state_dict()))

        eager_backend = EagerAndRecordGraphs()
        compiled_eager = torch.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )

        backend = AotEagerAndRecordGraphs()
        compiled_aot = torch.compile(mod_compile_aot, backend=backend, fullgraph=True)

        for _ in range(2):
            mod_eager.zero_grad()
            mod_compile_eager.zero_grad()
            mod_compile_aot.zero_grad()

            args = args_fn()
            args_clone = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )
            args_clone2 = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )

            out_eager = mod_eager(*args)
            loss_fn(out_eager).backward()

            out_compile_eager = compiled_eager(*args_clone)
            loss_fn(out_compile_eager).backward()

            out_compile_aot = compiled_aot(*args_clone2)
            loss_fn(out_compile_aot).backward()

            self.assertEqual(out_eager, out_compile_eager)
            self.assertEqual(out_eager, out_compile_aot)

            for (name_eager, param_eager), (_, param_compile_eager), (
                _,
                param_compile_aot,
            ) in zip(
                mod_eager.named_parameters(),
                mod_compile_eager.named_parameters(),
                mod_compile_aot.named_parameters(),
            ):
                self.assertEqual(
                    param_eager.grad,
                    param_compile_eager.grad,
                    msg=lambda msg: f"{msg}\nGradient mismatch for {name_eager} between eager and compile_eager",
                )
                self.assertEqual(
                    param_eager.grad,
                    param_compile_aot.grad,
                    msg=lambda msg: f"{msg}\nGradient mismatch for {name_eager} between eager and compile_aot",
                )

            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone,
            )
            pytree.tree_map(
                lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
                if isinstance(x, torch.Tensor) and x.requires_grad
                else None,
                args,
                args_clone2,
            )

        def _normalize(gm):
            # [FUNCTION DOC] HELPER: Prints a graph module readably, normalizes it, and rewrites
            # the nn module index number following an empty string argument to zero, since that
            # index depends on whether an accelerator stream is present.
            s = normalize_gm(gm.print_readable(print_output=False))
            # Normalize nn_module_index which varies depending on whether an
            # accelerator is available (stream reserves index 0).
            return re.sub(r"'', \d+, ", "'', 0, ", s)

        return (
            _normalize(eager_backend.graphs[0]),
            _normalize(backend.fw_graphs[0]),
            _normalize(backend.bw_graphs[0]),
        )

    def test_leaf_function_simple(self):
        # [FUNCTION DOC] Module whose forward calls a leaf function that branches on the sign of
        # the input sum and applies a submodule linear layer. Runs the comparison driver and checks
        # three expected printouts: the dynamo graph (invoke_leaf_function with parameters as
        # inputs), aot forward graph (with effects and an effect token) and backward graph.
        @leaf_function
        def non_tracable_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; applies the module's linear layer to the input,
            # adding the input back when the input sum is not positive; returns a one-tuple.
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @non_tracable_forward.register_fake
        def non_tracable_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the linear layer output.
            return (mod.linear(x),)

        class NonTracable(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a 3 by 3 linear layer; forward delegates to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return non_tracable_forward(self, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            NonTracable, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_x_ = L_x_

        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_2, primals_4, requires_grad_indices = '1,2,3');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_2 = primals_4 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4];  with_effects_1 = None
        return (getitem_5, getitem_4, getitem_6, getitem_2)
""",
        )

    def test_leaf_function_with_logging(self):
        # [FUNCTION DOC] A leaf function that prints a message each time its real body runs. With
        # the builtin print patched by a mock, runs the comparison driver and asserts the message
        # was printed, and that print was called exactly six times in total across the variants.
        @leaf_function
        def logging_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; prints a fixed message then returns the linear
            # output in a tuple.
            print("Processing input")
            return (mod.linear(x),)

        @logging_forward.register_fake
        def logging_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the linear layer output.
            return (mod.linear(x),)

        class LoggingModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a 3 by 3 linear layer; forward delegates to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return logging_forward(self, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            self.assertEqual(mock_print.call_count, 6)

    def test_leaf_function_dynamic_autograd_module_config(self):
        # [FUNCTION DOC] A leaf function reads a boolean attribute of its module argument to pick a
        # scale of two or three. Compiles the module once with a counting aot_eager backend, runs
        # forward and backward with the attribute true and then false, checking outputs and input
        # gradients match eager each time, that the two gradients differ, and that only one frame
        # was compiled (the attribute change does not trigger recompilation).
        from torch._dynamo.testing import CompileCounterWithBackend

        @leaf_function
        def configurable_scale(mod, x):
            # [FUNCTION DOC] HELPER: Real body; scales the linear output by two or three according
            # to the module's flag; returns a one-tuple.
            if mod.use_double_scale:
                return (mod.linear(x) * 2,)
            else:
                return (mod.linear(x) * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the unscaled linear output.
            return (mod.linear(x),)

        class ConfigurableModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer and a scale
            # flag initially true.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer and sets the flag to true.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.use_double_scale = True

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return configurable_scale(self, x)

        mod_eager = ConfigurableModule()
        mod_compiled = ConfigurableModule()
        mod_compiled.load_state_dict(dict(mod_eager.state_dict()))

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(mod_compiled, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)

        mod_eager.use_double_scale = True
        mod_compiled.use_double_scale = True

        x1 = x_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)

        out_eager_1 = mod_eager(x1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)

        mod_eager.zero_grad()
        mod_compiled.zero_grad()

        mod_eager.use_double_scale = False
        mod_compiled.use_double_scale = False

        x2 = x_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)

        out_eager_2 = mod_eager(x2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)

        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_dynamic_autograd_closure(self):
        # [FUNCTION DOC] Same idea as the module-config test but the switch is a dictionary
        # captured by closure and the function takes two tensors. Checks outputs and gradients of
        # both inputs against eager for flag true and false, that gradients differ between the two
        # settings, and that only one frame was compiled.
        from torch._dynamo.testing import CompileCounterWithBackend

        closure_config = {"use_double_scale": True}

        @leaf_function
        def configurable_scale(x, y):
            # [FUNCTION DOC] HELPER: Real body; multiplies the matrix product by two or three
            # according to the closure flag; returns a one-tuple.
            if closure_config["use_double_scale"]:
                return (x @ y * 2,)
            else:
                return (x @ y * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the plain matrix product.
            return (x @ y,)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under compilation; returns the leaf function result.
            return configurable_scale(x, y)

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)
        y_value = torch.randn(3, 3)

        closure_config["use_double_scale"] = True

        x1 = x_value.clone().requires_grad_(True)
        y1 = y_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)
        y1_clone = y_value.clone().requires_grad_(True)

        out_eager_1 = fn(x1, y1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone, y1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)
        self.assertEqual(y1.grad, y1_clone.grad)

        closure_config["use_double_scale"] = False

        x2 = x_value.clone().requires_grad_(True)
        y2 = y_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)
        y2_clone = y_value.clone().requires_grad_(True)

        out_eager_2 = fn(x2, y2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone, y2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        self.assertNotEqual(x1.grad, x2.grad)
        self.assertNotEqual(y1.grad, y2.grad)

        self.assertEqual(counter.frame_count, 1)

    @skipIfXpu(msg="https://github.com/pytorch/pytorch/issues/180662")
    def test_leaf_function_closure_constants_without_grad(self):
        # [FUNCTION DOC] Leaf function whose body uses a closed-over float and tensor plus module
        # attributes (a float scale and a learnable offset parameter). Runs the comparison driver
        # and checks expected dynamo, forward and backward printouts, in which the offset parameter
        # appears among the inputs and gradients. Skipped on XPU (linked issue).
        closure_scale = 2.0
        closure_tensor = torch.tensor([1.0, 2.0, 3.0])

        @leaf_function
        def closure_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; computes the linear output times a closure constant
            # and the module scale, adds a closure tensor and the offset, returns it in a tuple.
            out = mod.linear(x) * closure_scale * mod.scale
            out = out + closure_tensor + mod.offset
            return (out,)

        @closure_forward.register_fake
        def closure_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the linear output
            # plus the offset.
            return (mod.linear(x) + mod.offset,)

        class ClosureModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer, a float scale and a learnable
            # offset parameter of ones; forward delegates to the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer, the scale value and
                # the offset parameter.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return closure_forward(self, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            ClosureModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_parameters_offset_: "f32[3]", L_x_: "f32[3, 3]"):
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_parameters_offset_ = L_self_parameters_offset_
        l_x_ = L_x_

        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_parameters_offset_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_parameters_offset_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3]", primals_5: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_4, primals_3, primals_2, primals_5, requires_grad_indices = '1,2,3,4');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_4 = primals_3 = primals_2 = primals_5 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3]" = with_effects_1[2]
        getitem_5: "f32[3, 3]" = with_effects_1[3]
        getitem_6: "f32[3]" = with_effects_1[4]
        getitem_7: "f32[3, 3]" = with_effects_1[5];  with_effects_1 = None
        return (getitem_6, getitem_5, getitem_4, getitem_7, getitem_2)
""",
        )

    def test_leaf_function_pytree_inputs(self):
        # [FUNCTION DOC] Leaf function taking a dictionary with two tensors and returning two
        # outputs (a linear output and a shifted second tensor). Runs the comparison driver, which
        # checks eager, eager-compiled and aot-compiled agreement; no printouts are asserted.
        @leaf_function
        def pytree_forward(mod, inputs):
            # [FUNCTION DOC] HELPER: Real body; returns a pair whose second element is the second
            # entry shifted by one, and whose first depends on the first entry's sum sign.
            if inputs["x"].sum() > 0:
                return (mod.linear(inputs["x"]), inputs["y"] + 1)
            return (mod.linear(inputs["x"]) + inputs["y"], inputs["y"] - 1)

        @pytree_forward.register_fake
        def pytree_forward_fake(mod, inputs):
            # [FUNCTION DOC] HELPER: Fake; returns the linear output of the first entry and
            # the second entry.
            return (mod.linear(inputs["x"]), inputs["y"])

        class PytreeModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer; forward delegates to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the inputs.
                return pytree_forward(self, inputs)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple holding a dictionary of two random
            # tensors requiring grad.
            return (
                {
                    "x": torch.randn(3, 3, requires_grad=True),
                    "y": torch.randn(3, 3, requires_grad=True),
                },
            )

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; sum of both output elements.
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PytreeModule, args_fn, loss_fn)

    def test_leaf_function_nested_annotations(self):
        # [FUNCTION DOC] A leaf function whose module has a submodule that is itself a
        # leaf-function module (inner leaf nested in the outer leaf's body). Runs the comparison
        # driver and checks expected dynamo, forward and backward printouts, where only the outer
        # leaf call appears with the parameters of both linear layers as inputs.
        @leaf_function
        def inner_leaf_forward(mod, x):
            # [FUNCTION DOC] HELPER: Inner real body; linear output plus the input, in a tuple.
            y = mod.linear(x)
            return (y + x,)

        @inner_leaf_forward.register_fake
        def inner_leaf_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Inner fake; returns a one-tuple with the linear output.
            return (mod.linear(x),)

        class InnerLeaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer whose forward calls the
            # inner leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the inner leaf function result for this module
                # and the input.
                return inner_leaf_forward(self, x)

        @leaf_function
        def outer_leaf_forward(mod, x):
            # [FUNCTION DOC] HELPER: Outer real body; applies the linear layer, adds the input, and
            # returns the result of calling the inner module on it.
            z = mod.linear(x)
            return mod.inner(z + x)

        @outer_leaf_forward.register_fake
        def outer_leaf_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Outer fake; returns the inner module applied to
            # the linear output.
            return mod.inner(mod.linear(x))

        class OuterLeaf(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an inner leaf module and a 3 by 3 linear layer;
            # forward delegates to the outer leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the inner leaf module and the linear layer.
                super().__init__()
                self.inner = InnerLeaf()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the outer leaf function result for this module
                # and the input.
                return outer_leaf_forward(self, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            OuterLeaf, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_inner_modules_linear_parameters_bias_: "f32[3]", L_self_modules_inner_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_self_modules_inner_modules_linear_parameters_bias_ = L_self_modules_inner_modules_linear_parameters_bias_
        l_self_modules_inner_modules_linear_parameters_weight_ = L_self_modules_inner_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_x_ = L_x_

        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_inner_modules_linear_parameters_weight_, l_self_modules_inner_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_inner_modules_linear_parameters_weight_ = l_self_modules_inner_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3]", primals_5: "f32[3, 3]", primals_6: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_2, primals_5, primals_4, primals_6, requires_grad_indices = '1,2,3,4,5');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_2 = primals_5 = primals_4 = primals_6 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = '');  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4]
        getitem_7: "f32[3]" = with_effects_1[5]
        getitem_8: "f32[3, 3]" = with_effects_1[6];  with_effects_1 = None
        return (getitem_5, getitem_4, getitem_7, getitem_6, getitem_8, getitem_2)
""",
        )

    def test_leaf_function_data_dependent_nonzero(self):
        # [FUNCTION DOC] Leaf function returning a tensor and the indices of its positive entries
        # (data-dependent shape), used inside a larger module that applies a linear layer and relu
        # before it and uses the count of indices after it. Runs the comparison driver without
        # asserting printouts; checks eager, eager-compiled and aot-compiled agreement.
        @leaf_function
        def nonzero_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; returns the linear output and indices of
            # its positive entries.
            out = mod.linear(x)
            nonzero_indices = (out > 0).nonzero()
            return (out, nonzero_indices)

        @nonzero_forward.register_fake
        def nonzero_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; computes the same two outputs as the real body.
            out = mod.linear(x)
            return out, (out > 0).nonzero()

        class NonzeroModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer; forward delegates to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return nonzero_forward(self, x)

        class OuterModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module with a pre-linear layer, the nonzero module and a
            # learnable scalar scale; forward applies relu, calls the leaf module, then scales the
            # output and adds the number of positive entries.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the pre-linear layer, the nonzero module and
                # the scale parameter.
                super().__init__()
                self.pre_linear = torch.nn.Linear(3, 3)
                self.nonzero_module = NonzeroModule()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs linear, relu, the leaf module; returns the output
                # times the scale plus the index count, together with the indices.
                x = self.pre_linear(x)
                x = torch.relu(x)
                out, nonzero_indices = self.nonzero_module(x)
                num_nonzero = nonzero_indices.shape[0]
                scaled_out = out * self.scale + num_nonzero
                return scaled_out, nonzero_indices

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        self._test_leaf_function_helper(OuterModule, args_fn, loss_fn)

    @skipIfCrossRef
    def test_leaf_function_data_dependent_item(self):
        # [FUNCTION DOC] Leaf function returning a tensor and a Python scalar obtained from a
        # tensor sum (data-dependent scalar). Runs the comparison driver without printouts. Skipped
        # under cross-ref mode.
        @leaf_function
        def item_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; returns the linear output and the scalar sum of it.
            out = mod.linear(x)
            scalar_value = out.sum().item()
            return (out, scalar_value)

        @item_forward.register_fake
        def item_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; computes the same two outputs as the real body.
            out = mod.linear(x)
            return (out, out.sum().item())

        class ItemModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer; forward delegates to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return item_forward(self, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        self._test_leaf_function_helper(ItemModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_multiple_compiled_submodules(self, backend):
        # [FUNCTION DOC] Parametrized over the eager and aot_eager backends. Builds a top-level
        # module with three submodules each containing leaf-function modules, and a compiled copy
        # that compiles each submodule separately with the backend. Asserts both models agree on
        # outputs and gradients through the model comparison helper.
        @leaf_function
        def leaf_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; linear output, adding the input when the input sum
            # is not positive.
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the linear output.
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer of given sizes; forward calls
            # the leaf function.
            def __init__(self, in_features, out_features):
                # [FUNCTION DOC] HELPER: Creates the linear layer with the given feature sizes.
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return leaf_forward(self, x)

        class CompiledSubmodule1(torch.nn.Module):
            # [CLASS DOC] HELPER: First submodule; linear layer, relu, then a leaf module.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 4 by 4 linear layer and a leaf module.
                super().__init__()
                self.pre_linear = torch.nn.Linear(4, 4)
                self.leaf = LeafModule(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear and relu, then returns the
                # first leaf output.
                x = self.pre_linear(x)
                x = torch.relu(x)
                out = self.leaf(x)[0]
                return out

        class CompiledSubmodule2(torch.nn.Module):
            # [CLASS DOC] HELPER: Second submodule; a leaf module followed by linear and sigmoid.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a leaf module and a 4 by 4 linear layer.
                super().__init__()
                self.leaf = LeafModule(4, 4)
                self.post_linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the leaf module, then linear and sigmoid.
                out = self.leaf(x)[0]
                out = self.post_linear(out)
                return torch.sigmoid(out)

        class CompiledSubmodule3(torch.nn.Module):
            # [CLASS DOC] HELPER: Third submodule; two leaf modules applied to the same
            # input and summed.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two leaf modules.
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the first outputs of both leaf modules.
                out1 = self.leaf1(x)[0]
                out2 = self.leaf2(x)[0]
                return out1 + out2

        class TopLevelModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Chain of the three submodules followed by a final linear layer; a
            # constructor flag selects whether each submodule is wrapped by torch.compile with the
            # test's backend on every forward call.
            def __init__(self, compile_submodules=False):
                # [FUNCTION DOC] HELPER: Creates the submodules, final linear layer and stores
                # the compile flag.
                super().__init__()
                self.submodule1 = CompiledSubmodule1()
                self.submodule2 = CompiledSubmodule2()
                self.submodule3 = CompiledSubmodule3()
                self.final_linear = torch.nn.Linear(4, 4)
                self.compile_submodules = compile_submodules

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the three submodules in sequence (compiled or not
                # depending on the flag) and applies the final linear layer.
                if self.compile_submodules:
                    out1 = torch.compile(self.submodule1, backend=backend)(x)
                    out2 = torch.compile(self.submodule2, backend=backend)(out1)
                    out3 = torch.compile(self.submodule3, backend=backend)(out2)
                else:
                    out1 = self.submodule1(x)
                    out2 = self.submodule2(out1)
                    out3 = self.submodule3(out2)
                final = self.final_linear(out3)
                return final

        model_eager = TopLevelModule(compile_submodules=False)
        model_compiled = TopLevelModule(compile_submodules=True)
        model_compiled.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_compiled = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(
            model_eager,
            model_compiled,
            x,
            x_compiled,
        )

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("do_compile", [False, True])
    def test_leaf_function_with_graph_breaks(self, backend, do_compile):
        # [FUNCTION DOC] Parametrized over backend and a compile flag. Builds a module chaining
        # three leaf-function modules with explicit dynamo graph breaks between them; when the flag
        # is set, the chain is compiled without full-graph. Asserts the test model matches the
        # eager model in outputs and gradients.
        @leaf_function
        def leaf_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; linear output, plus one when the input sum
            # is not positive.
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + 1,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the linear output.
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer; forward calls the leaf function.
            def __init__(self, in_features, out_features):
                # [FUNCTION DOC] HELPER: Creates the linear layer with the given feature sizes.
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return leaf_forward(self, x)

        class TopLevelModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Three leaf modules and a final linear layer; stores a compile
            # flag and a backend name used to optionally compile the inner chain.
            def __init__(self, do_compile=False, backend="eager"):
                # [FUNCTION DOC] HELPER: Creates the leaf modules and linear layer, and stores the
                # flag and backend.
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)
                self.leaf3 = LeafModule(4, 4)
                self.final_linear = torch.nn.Linear(4, 4)
                self.do_compile = do_compile
                self.backend = backend

            def _forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the three leaf modules in order with a graph break
                # after the first two, then the final linear layer.
                out1 = self.leaf1(x)[0]
                torch._dynamo.graph_break()
                out2 = self.leaf2(out1)[0]
                torch._dynamo.graph_break()
                out3 = self.leaf3(out2)[0]
                result = self.final_linear(out3)
                return result

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the chain compiled (without requiring a full graph)
                # with the stored backend when the flag is set, otherwise runs it directly.
                if self.do_compile:
                    return torch.compile(
                        self._forward, backend=self.backend, fullgraph=False
                    )(x)
                else:
                    return self._forward(x)

        model_eager = TopLevelModule(do_compile=False)
        model_test = TopLevelModule(do_compile=do_compile, backend=backend)
        model_test.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_test = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(model_eager, model_test, x, x_test)

    def test_leaf_function_with_module_in_pytree(self):
        # [FUNCTION DOC] Leaf function whose first argument is a dictionary of modules. Its body
        # combines two helper modules by sum or difference depending on the input sum. Runs the
        # comparison driver, so modules inside a pytree argument must work in all three variants.
        @leaf_function
        def main_forward(modules_dict, x):
            # [FUNCTION DOC] HELPER: Real body; sum or difference of the two module outputs by
            # input sum sign.
            if x.sum() > 0:
                return (modules_dict["first"](x) + modules_dict["second"](x),)
            else:
                return (modules_dict["first"](x) - modules_dict["second"](x),)

        @main_forward.register_fake
        def main_forward_fake(modules_dict, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the sum of both module outputs.
            return (modules_dict["first"](x) + modules_dict["second"](x),)

        class HelperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer and a float scale; forward scales
            # the linear output.
            def __init__(self, scale=1.0):
                # [FUNCTION DOC] HELPER: Creates the linear layer and stores the scale.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = scale

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear output multiplied by the scale.
                return self.linear(x) * self.scale

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds two helper modules (scales one and one half); forward
            # passes a dictionary of them to the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two helper modules.
                super().__init__()
                self.helper1 = HelperModule(scale=1.0)
                self.helper2 = HelperModule(scale=0.5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a dictionary of the helpers and calls
                # the leaf function.
                modules_dict = {"first": self.helper1, "second": self.helper2}
                return main_forward(modules_dict, x)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_as_kwarg(self):
        # [FUNCTION DOC] Leaf function receiving a module as a keyword argument (default none).
        # Runs the comparison driver to check eager, eager-compiled and aot-compiled agreement.
        @leaf_function
        def main_forward(x, helper_mod=None):
            # [FUNCTION DOC] HELPER: Real body; returns the helper module output, plus the input
            # when the input sum is not positive, in a tuple.
            if x.sum() > 0:
                return (helper_mod(x),)
            else:
                return (helper_mod(x) + x,)

        @main_forward.register_fake
        def main_forward_fake(x, helper_mod=None):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with the helper module output.
            return (helper_mod(x),)

        class HelperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a 3 by 3 linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a helper module and passes it as a keyword argument to
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the helper module.
                super().__init__()
                self.helper = HelperModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the leaf function with the input and the helper
                # as a keyword.
                return main_forward(x, helper_mod=self.helper)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_missing_fake_impl_error(self):
        # [FUNCTION DOC] A leaf function without a fake implementation must raise an error whose
        # message says a fake implementation is required, both when the module is called eagerly
        # and when it is compiled with the eager backend and full graph.
        @leaf_function
        def no_fake_impl_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the linear output. No
            # fake is registered.
            return (mod.linear(x),)

        class SimpleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward calls the leaf function
            # lacking a fake.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return no_fake_impl_forward(self, x)

        mod = SimpleModule()
        x = torch.randn(3, 3)

        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            mod(x)

        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_constant_tensor_closure_error(self, backend):
        # [FUNCTION DOC] Parametrized over backend. A leaf function closes over a plain
        # constant tensor (no grad). Eager execution works and matches the matrix product.
        # Compiling with full graph then calling must raise an error asking to convert all
        # tensors to fake tensors.
        constant_weight = torch.randn(3, 3)

        @leaf_function
        def constant_closure_forward(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the
            # closed-over constant tensor.
            return (x @ constant_weight,)

        @constant_closure_forward.register_fake
        def constant_closure_forward_fake(x):
            # [FUNCTION DOC] HELPER: Fake; same computation using the same constant tensor.
            return (x @ constant_weight,)

        class ConstantClosureModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Parameterless module whose forward calls the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the module with no state.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for the input.
                return constant_closure_forward(x)

        mod = ConstantClosureModule()
        x = torch.randn(3, 3, requires_grad=True)

        result = mod(x)
        expected = x @ constant_weight
        self.assertEqual(result[0], expected)

        compiled_mod = torch.compile(mod, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_error(self, backend):
        # [FUNCTION DOC] Parametrized over backend. A leaf function mutates its input without
        # declaring mutation. Eagerly this raises a runtime error about undeclared in-place
        # mutation. When compiled with full graph and called with a leaf tensor requiring grad it
        # raises a runtime error about an in-place operation on a leaf variable requiring grad.
        @leaf_function
        def mutate_input(x):
            # [FUNCTION DOC] HELPER: Real body; adds one to the input in place and returns it
            # in a tuple.
            x.add_(1)
            return (x,)

        @mutate_input.register_fake
        def mutate_input_fake(x):
            # [FUNCTION DOC] HELPER: Fake; same in-place update, returning the input in a tuple.
            x.add_(1)
            return (x,)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the leaf function result.
            return mutate_input(x)

        x = torch.randn(3, 3)

        x_eager = x.clone()
        with self.assertRaisesRegex(RuntimeError, "Undeclared in-place mutation"):
            fn(x_eager)

        x = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
            compiled_fn(x.clone().requires_grad_(True))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_validation_dtype_mismatch(self, backend):
        # [FUNCTION DOC] Parametrized over backend. The fake implementation returns a double
        # precision tensor while the real body returns the default dtype. With output validation
        # enabled via config, calling the compiled module must raise a runtime error mentioning
        # a dtype mismatch.
        @leaf_function
        def dtype_mismatch_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the linear output.
            return (mod.linear(x),)

        @dtype_mismatch_forward.register_fake
        def dtype_mismatch_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns the linear output converted to double
            # in a tuple.
            return (mod.linear(x).double(),)

        class DtypeMismatchModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward calls the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return dtype_mismatch_forward(self, x)

        mod = DtypeMismatchModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=True):
            compiled_mod = torch.compile(mod, backend=backend)
            with self.assertRaisesRegex(RuntimeError, "Dtype mismatch"):
                compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("validate_outputs", [True, False])
    def test_leaf_function_validation_shape_mismatch(self, backend, validate_outputs):
        # [FUNCTION DOC] Parametrized over backend and validation flag. The fake returns a tensor
        # with 6 columns while the real body returns 3. With validation on (config), calling the
        # compiled module must raise a runtime or assertion error; with validation off it must
        # succeed and return an output of shape 3 by 3.
        @leaf_function
        def mismatched_forward(mod, x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the linear output.
            return (mod.linear(x),)

        @mismatched_forward.register_fake
        def mismatched_forward_fake(mod, x):
            # [FUNCTION DOC] HELPER: Fake; returns a one-tuple with zeros of shape batch by
            # 6 (wrong width).
            return (torch.zeros(x.shape[0], 6),)

        class MismatchedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward calls the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for this module
                # and the input.
                return mismatched_forward(self, x)

        mod = MismatchedModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=validate_outputs):
            compiled_mod = torch.compile(mod, backend=backend)
            if validate_outputs:
                with self.assertRaises((RuntimeError, AssertionError)):
                    compiled_mod(x)
            else:
                result = compiled_mod(x)
                self.assertEqual(result[0].shape, (3, 3))

    def test_leaf_function_no_module_inputs(self):
        # [FUNCTION DOC] Leaf function with no module arguments: a dictionary of two tensors, a
        # float scale and an integer offset, returning two outputs. Runs the comparison driver to
        # check eager, eager-compiled and aot-compiled agreement.
        @leaf_function
        def my_custom_fn(inputs: dict[str, torch.Tensor], scale: float, offset: int):
            # [FUNCTION DOC] HELPER: Real body; scales one tensor, adds or subtracts the
            # other by input sum sign, plus offset, and returns that with the sum or
            # difference of tensor sums.
            x = inputs["x"]
            y = inputs["y"]
            if x.sum() > 0:
                return (x * scale + y + offset, x.sum() + y.sum())
            return (x * scale - y + offset, x.sum() - y.sum())

        @my_custom_fn.register_fake
        def my_custom_fn_fake(
            inputs: dict[str, torch.Tensor], scale: float, offset: int
        ):
            # [FUNCTION DOC] HELPER: Fake; returns the positive-case results of the real body.
            x = inputs["x"]
            y = inputs["y"]
            return (x * scale + y + offset, x.sum() + y.sum())

        class NoModuleInputsModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a float scale of two and an integer offset of one.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the module with the scale and offset.
                super().__init__()
                self.scale = 2.0
                self.offset = 1

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Packs both inputs in a dictionary and calls the leaf
                # function with the scale and offset.
                inputs = {"x": x, "y": y}
                return my_custom_fn(inputs, self.scale, self.offset)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns two random 3 by 3 tensors requiring grad.
            return (
                torch.randn(3, 3, requires_grad=True),
                torch.randn(3, 3, requires_grad=True),
            )

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; sum of both output elements' sums (first
            # output plus second).
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(NoModuleInputsModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_multiple_tensors(
        self, backend, check_escaped_gradients
    ):
        # [FUNCTION DOC] Parametrized over backend and a check flag. A leaf function closes over
        # two weights requiring grad. Compiled with full graph under a config setting of the
        # escaped gradient check. With the check on, calling must raise a runtime error mentioning
        # two tensors; with it off, the call succeeds and the output has shape 2 by 3.
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies input by each closed-over
            # weight and adds.
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x)
            else:
                result = compiled_fn(x)
                self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_input_no_grad(
        self, backend, check_escaped_gradients
    ):
        # [FUNCTION DOC] Parametrized over backend and the escaped-gradient check flag. The leaf
        # function closes over a weight requiring grad, but the input does not require grad, so no
        # error is expected whichever flag value is set; asserts the result has shape 2 by 3.
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            result = compiled_fn(x)
            self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_mixed_inputs(
        self, backend, check_escaped_gradients
    ):
        # [FUNCTION DOC] Parametrized over backend and the check flag. The closed-over weights are
        # derived (by scaling) from grad-requiring base tensors; one input requires grad and one
        # does not. With the check on, the compiled call must raise a runtime error mentioning two
        # tensors; with it off, outputs are produced with shapes 2 by 3 and 2 by 4.
        base1 = torch.randn(3, 3, requires_grad=True)
        base2 = torch.randn(3, 4, requires_grad=True)
        closure_weight1 = base1 * 2
        closure_weight2 = base2 * 3

        @leaf_function
        def mixed_inputs(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns two outputs, each built from the input and
            # a closed-over weight (the first also adds the second input).
            out1 = x @ closure_weight1 + y
            out2 = x @ closure_weight2
            return (out1, out2)

        @mixed_inputs.register_fake
        def mixed_inputs_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns two uninitialized tensors with shapes batch by 3
            # and batch by 4.
            return (torch.empty(x.shape[0], 3), torch.empty(x.shape[0], 4))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return mixed_inputs(x, y)

        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x, y)
            else:
                result = compiled_fn(x, y)
                self.assertEqual(result[0].shape, (2, 3))
                self.assertEqual(result[1].shape, (2, 4))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_error_message_contains_tensor_info(
        self, backend
    ):
        # [FUNCTION DOC] Parametrized over backend. With the escaped-gradient check enabled, the
        # raised runtime error message must describe the offending closed-over tensor, matching a
        # pattern with its shape (4 by 5) followed later by a dtype entry.
        closure_weight = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over 4
            # by 5 weight.
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of shape batch by 5.
            return (torch.empty(x.shape[0], 5),)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return uses_closure(x)

        x = torch.randn(2, 4, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, r"shape=\[4, 5\].*dtype="):
                compiled_fn(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_actually_lost(self, backend):
        # [FUNCTION DOC] Parametrized over backend. With default config, runs a compiled forward
        # and backward through a leaf function using a grad-requiring closure weight; asserts the
        # input receives a gradient while the closure weight gets none (the gradient is lost).
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            # [FUNCTION DOC] HELPER: Real body; multiplies the input by the closed-over weight.
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor of output shape.
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        loss = result[0].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_leaf_function_and_nonstrict_trace_mutually_exclusive(self):
        # [FUNCTION DOC] Applying leaf_function and nonstrict_trace to the same function, in either
        # order, must raise a ValueError saying it cannot be both marked as leaf_function and
        # nonstrict_trace. Checked for both decoration orders.
        from torch._dynamo.decorators import leaf_function, nonstrict_trace

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @leaf_function
            @nonstrict_trace
            def bad_fn1(x):
                # [FUNCTION DOC] HELPER: Function decorated with both decorators (never
                # successfully created); returns its input in a tuple.
                return (x,)

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @nonstrict_trace
            @leaf_function
            def bad_fn2(x):
                # [FUNCTION DOC] HELPER: Same as the first, with the decorator order reversed.
                return (x,)

    @skipIfCrossRef
    def test_leaf_function_no_return_value(self):
        # [FUNCTION DOC] Leaf function with no return value (returns nothing, prints a message in
        # its body) used by a module that also computes a linear layer. With print replaced by a
        # recorder, runs the comparison driver, asserts the message was printed, and compares
        # expected printouts: the dynamo graph keeps the invoke call but discards its result, the
        # forward graph has an effect-wrapped invoke with only the token returned, and the backward
        # graph contains only the linear layer gradients. Skipped under cross-ref mode.
        printed = []

        @leaf_function
        def fn_no_return(x):
            # [FUNCTION DOC] HELPER: Real body; prints a message and returns nothing.
            print("processing")

        @fn_no_return.register_fake
        def fn_no_return_fake(x):
            # [FUNCTION DOC] HELPER: Fake; does nothing and returns nothing.
            pass

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer; forward calls the no-return leaf
            # function, ignoring its result, then returns a one-tuple with the linear output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the leaf function for its side effect, then applies
                # the linear layer.
                fn_no_return(x)
                return (self.linear(x),)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the first output element.
            return out[0].sum()

        with patch("builtins.print", lambda *args, **kwargs: printed.append(args)):
            eager_graph, fw_graph, bw_graph = self._test_leaf_function_helper(
                Mod, args_fn, loss_fn
            )
        self.assertTrue(any("processing" in p for p in printed))
        self.assertExpectedInline(
            eager_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_x_ = L_x_

        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', l_x_);  real_fn = fake_fn = input_spec = invoke_leaf_function = None

        linear: "f32[3, 3]" = torch._C._nn.linear(l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  l_x_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        return (linear,)
""",
        )
        self.assertExpectedInline(
            fw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3, 3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_4, requires_grad_indices = '0');  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None

        t: "f32[3, 3]" = torch.ops.aten.t.default(primals_3);  primals_3 = None
        addmm: "f32[3, 3]" = torch.ops.aten.addmm.default(primals_2, primals_4, t);  primals_2 = None

        t_1: "f32[3, 3]" = torch.ops.aten.t.default(t);  t = None
        return (getitem, addmm, primals_4, t_1)
""",
        )
        self.assertExpectedInline(
            bw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[3, 3]", t_1: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        mm: "f32[3, 3]" = torch.ops.aten.mm.default(tangents_1, t_1);  t_1 = None
        t_2: "f32[3, 3]" = torch.ops.aten.t.default(tangents_1)
        mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t_2, primals_4);  t_2 = primals_4 = None
        t_3: "f32[3, 3]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
        sum_1: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[3]" = torch.ops.aten.view.default(sum_1, [3]);  sum_1 = None
        t_4: "f32[3, 3]" = torch.ops.aten.t.default(t_3);  t_3 = None
        return (view, t_4, mm)
""",
        )

    def test_leaf_function_output_structure_mismatch(self):
        # [FUNCTION DOC] The real body returns a dictionary while the fake returns a tuple.
        # Compiling with the eager backend and calling must raise an assertion error mentioning an
        # output structure mismatch.
        @leaf_function
        def mismatched_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a dictionary with the input and its double.
            return {"a": x, "b": x * 2}

        @mismatched_fn.register_fake
        def mismatched_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns a tuple with the input and its
            # double (wrong structure).
            return (x, x * 2)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return mismatched_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(AssertionError, "output structure mismatch"):
            torch.compile(fn, backend="eager")(x)

    def test_leaf_function_nested_output(self):
        # [FUNCTION DOC] Leaf function taking three linear modules and an input and returning a
        # nested structure: a dictionary with a pair of tensors, a tensor and an integer. The
        # module combines the pieces into one tensor. Runs the comparison driver.
        @leaf_function
        def nested_output_fn(linear1, linear2, linear3, x):
            # [FUNCTION DOC] HELPER: Real body; returns the nested dictionary of module
            # outputs, each shifted by one when the input sum is not positive, with a
            # constant integer count.
            if x.sum() > 0:
                return {
                    "out": (linear1(x), linear2(x)),
                    "extra": linear3(x),
                    "count": 42,
                }
            else:
                return {
                    "out": (linear1(x) + 1, linear2(x) + 1),
                    "extra": linear3(x) + 1,
                    "count": 42,
                }

        @nested_output_fn.register_fake
        def nested_output_fn_fake(linear1, linear2, linear3, x):
            # [FUNCTION DOC] HELPER: Fake; returns the same nested structure without the shift.
            return {
                "out": (linear1(x), linear2(x)),
                "extra": linear3(x),
                "count": 42,
            }

        class NestedOutputModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three linear layers; forward combines the nested
            # output pieces (first times the count, plus second, plus extra).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the three linear layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the leaf function with the three layers and the
                # input and combines the result.
                result = nested_output_fn(self.linear1, self.linear2, self.linear3, x)
                return (
                    result["out"][0] * result["count"]
                    + result["out"][1]
                    + result["extra"]
                )

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; the sum of the single output tensor.
            return out.sum()

        self._test_leaf_function_helper(NestedOutputModule, args_fn, loss_fn)

    def test_leaf_function_custom_pytree_output(self):
        # [FUNCTION DOC] Leaf function returning an instance of a custom class (holding two
        # tensors) along with a float. The class is registered as a pytree node through the test
        # base class helper, which records it so the registration is removed in teardown. Runs
        # the comparison driver.
        class Point:
            # [CLASS DOC] HELPER: Plain container class with two tensor attributes x and y.
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the two tensors as attributes.
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @leaf_function
        def point_fn(linear1, linear2, x):
            # [FUNCTION DOC] HELPER: Real body; returns a point of the two linear outputs and 0.5.
            return (Point(linear1(x), linear2(x)), 0.5)

        @point_fn.register_fake
        def point_fn_fake(linear1, linear2, x):
            # [FUNCTION DOC] HELPER: Fake; same as the real body.
            return (Point(linear1(x), linear2(x)), 0.5)

        class PointModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers; forward unpacks the leaf result
            # and returns both point tensors multiplied by the scale.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the leaf function, then scales the two tensors
                # of the point.
                p, scale = point_fn(self.linear1, self.linear2, x)
                return (p.x * scale, p.y * scale)

        def args_fn():
            # [FUNCTION DOC] HELPER: Returns a one-tuple with a random 3 by 3
            # tensor requiring grad.
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            # [FUNCTION DOC] HELPER: Loss; sum of both output elements.
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PointModule, args_fn, loss_fn)

    def test_leaf_function_fake_requires_grad_ignored(self):
        # [FUNCTION DOC] The fake implementation returns a tensor that does not require grad, yet
        # the compiled output must still require grad and backward must give the input a gradient.
        # Also inspects the first recorded dynamo graph: for each invoke_leaf_function call node,
        # asserts the recorded example value is present and its first element requires grad.
        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # with requires_grad off.
            return (torch.empty_like(x).requires_grad_(False),)

        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function compiled with full graph and the
            # recording eager backend.
            return my_fn(x)

        x = torch.randn(3, 3, requires_grad=True)
        out = fn(x)

        self.assertTrue(out[0].requires_grad)
        out[0].sum().backward()
        self.assertIsNotNone(x.grad)

        graph = backend.graphs[0]
        for node in graph.graph.nodes:
            if node.op == "call_function" and "invoke_leaf_function" in str(
                node.target
            ):
                example_value = node.meta.get("example_value")
                self.assertIsNotNone(example_value)
                self.assertTrue(example_value[0].requires_grad)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_non_grad(self, backend):
        # [FUNCTION DOC] Parametrized over backend. A leaf function declared to mutate buf, with
        # inputs not requiring grad. Compares eager results and mutated buffer with the expected
        # values, then the compiled run (full graph) gives the same result and buffer mutation.
        @leaf_function(mutates_args={"buf"})
        def mutate_buffer(x, buf):
            # [FUNCTION DOC] HELPER: Real body; adds one to buf in place and returns input plus buf
            # in a tuple.
            buf.add_(1)
            return (x + buf,)

        @mutate_buffer.register_fake
        def mutate_buffer_fake(x, buf):
            # [FUNCTION DOC] HELPER: Fake; same in-place update and sum.
            buf.add_(1)
            return (x + buf,)

        def fn(x, buf):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return mutate_buffer(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x + buf + 1
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_mixed(self, backend):
        # [FUNCTION DOC] Parametrized over backend. Leaf function declared to mutate buf (doubling
        # it in place) while the other input requires grad. Eager and compiled runs (full graph)
        # must produce the expected product result and leave buf doubled.
        @leaf_function(mutates_args={"buf"})
        def mixed_fn(x, buf):
            # [FUNCTION DOC] HELPER: Real body; doubles buf in place and returns input times buf
            # in a tuple.
            buf.mul_(2)
            return (x * buf,)

        @mixed_fn.register_fake
        def mixed_fn_fake(x, buf):
            # [FUNCTION DOC] HELPER: Fake; same in-place doubling and product.
            buf.mul_(2)
            return (x * buf,)

        def fn(x, buf):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return mixed_fn(x, buf)

        x = torch.randn(3, 3, requires_grad=True)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x * (buf * 2)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf * 2)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf * 2)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_module_buffer(self, backend):
        # [FUNCTION DOC] Parametrized over backend. Leaf function declares mutation of a buffer
        # reached through an attribute path of the module argument. Deep-copies a fresh module for
        # the eager and compiled runs and checks both outputs equal and the running mean buffer
        # equals the mean of the input over the batch in each copy.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a zero-initialized running mean buffer and a linear
            # layer; forward calls the mutating leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the buffer and the linear layer.
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the mutating leaf function result for this
                # module and input.
                return update_stats(self, x)

        @leaf_function(mutates_args={"model.running_mean"})
        def update_stats(model, x):
            # [FUNCTION DOC] HELPER: Real body; adds the batch mean of the input to the module's
            # running mean buffer in place and returns the linear output in a tuple.
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        @update_stats.register_fake
        def update_stats_fake(model, x):
            # [FUNCTION DOC] HELPER: Fake; same in-place update and linear output.
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        mod = MyModule()
        x = torch.randn(4, 3)

        mod_eager = copy.deepcopy(mod)
        result_eager = mod_eager(x)
        expected_mean = torch.zeros(3) + x.mean(dim=0)
        self.assertEqual(mod_eager.running_mean, expected_mean)

        mod_compiled = copy.deepcopy(mod)
        compiled_mod = torch.compile(mod_compiled, backend=backend, fullgraph=True)
        result_compiled = compiled_mod(x)
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(mod_compiled.running_mean, expected_mean)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree(self, backend):
        # [FUNCTION DOC] Parametrized over backend. Leaf function declared to mutate a list-valued
        # argument; all list elements are incremented in place. Eager and compiled runs (full
        # graph) must give the expected sum result and increment both buffers.
        @leaf_function(mutates_args={"buffers"})
        def update_buffers(x, buffers):
            # [FUNCTION DOC] HELPER: Real body; adds one in place to each tensor in the list and
            # returns the input plus their sum, in a tuple.
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        @update_buffers.register_fake
        def update_buffers_fake(x, buffers):
            # [FUNCTION DOC] HELPER: Fake; same in-place updates and sum.
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        def fn(x, buffers):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return update_buffers(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + (bufs[1] + 1)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1] + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1] + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree_fine_grained(self, backend):
        # [FUNCTION DOC] Parametrized over backend. Leaf function declares mutation of only the
        # first element of a list argument. Eager and compiled runs (full graph) must increment the
        # first buffer and leave the second unchanged, with the expected result.
        @leaf_function(mutates_args={"buffers[0]"})
        def update_first(x, buffers):
            # [FUNCTION DOC] HELPER: Real body; adds one in place to the first list element and
            # returns the input plus both elements in a tuple.
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        @update_first.register_fake
        def update_first_fake(x, buffers):
            # [FUNCTION DOC] HELPER: Fake; same in-place update and sum.
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        def fn(x, buffers):
            # [FUNCTION DOC] HELPER: Function compiled by the test; returns the
            # leaf function result.
            return update_first(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + bufs[1]
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1])

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1])

    def test_leaf_function_mutates_args_invalid_parameter(self):
        # [FUNCTION DOC] Declaring mutated arguments that name a parameter the function lacks must
        # raise a ValueError at decoration time, mentioning the referenced parameter name; checked
        # for a plain name and for an attribute path whose root is not a parameter.
        with self.assertRaisesRegex(ValueError, "refers to parameter 'buf'"):

            @leaf_function(mutates_args={"buf"})
            def bad_fn(x, buffers):
                # [FUNCTION DOC] HELPER: Decorated function whose declared mutated name does not
                # match a parameter; creation is expected to raise, so it is never called.
                buffers.add_(1)
                return (x + buffers,)

        with self.assertRaisesRegex(ValueError, "refers to parameter 'mdl'"):

            @leaf_function(mutates_args={"mdl.running_mean"})
            def bad_fn2(x, model):
                # [FUNCTION DOC] HELPER: Decorated function whose declared attribute path root does
                # not match a parameter; creation is expected to raise, so it is never called.
                model.running_mean.add_(1)
                return (x,)

    def test_leaf_function_mutates_args_non_leaf_expression(self):
        # [FUNCTION DOC] Leaf function declares mutation of a whole module argument (not a leaf
        # tensor path) while its body mutates a buffer inside that module. Compiling the module
        # with full graph and calling it must raise a dynamo user error saying the expression
        # resolved to a non-leaf value.
        @leaf_function(mutates_args={"model"})
        def bad_fn(x, model):
            # [FUNCTION DOC] HELPER: Real body; adds one to the module's running mean buffer in
            # place and returns the input.
            model.running_mean.add_(1)
            return (x,)

        @bad_fn.register_fake
        def bad_fn_fake(x, model):
            # [FUNCTION DOC] HELPER: Fake; same in-place update and return.
            model.running_mean.add_(1)
            return (x,)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a zero-initialized running mean buffer; forward calls
            # the leaf function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the buffer.
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the leaf function result for the input
                # and this module.
                return bad_fn(x, self)

        mod = MyModule()
        x = torch.randn(3)
        compiled_fn = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, "resolved to a non-leaf value"
        ):
            compiled_fn(x)


instantiate_parametrized_tests(TestLeafFunctionDynamo)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionRegisterHook(TestCase):
    # [CLASS DOC] Tests for the register_multi_grad_hook API of leaf functions: a hook runs during
    # backward and receives the gradients of exactly the tensor inputs that require grad.
    # Stateless; each test collects hook observations in local lists.
    """Tests for @leaf_function's register_multi_grad_hook API."""

    def test_hook_fires_on_backward(self):
        # [FUNCTION DOC] Registers a hook that records a copy of its gradient argument; after
        # forward and backward of a doubling leaf function, asserts the hook fired once with a
        # gradient of all twos.
        hook_grads = []

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; appends a copy of the received gradient to the list.
            hook_grads.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()

        self.assertEqual(len(hook_grads), 1)
        self.assertEqual(hook_grads[0], torch.full((3,), 2.0))

    def test_hook_with_non_tensor_args(self):
        # [FUNCTION DOC] The leaf function takes a tensor plus a string and a float; the hook still
        # takes one gradient only, since non-tensor arguments are not passed. Asserts one hook call
        # with a gradient equal to the float scale.
        # Hook receives only grads of requires_grad tensors. Non-tensor args
        # in the leaf signature do not flow into the hook.
        hook_grads = []

        @leaf_function
        def my_fn(x, tag, scale):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input times the scale.
            return (x * scale,)

        @my_fn.register_fake
        def my_fn_fake(x, tag, scale):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; appends a copy of the received gradient to the list.
            hook_grads.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x, "hello", 5.0)[0]
        out.sum().backward()

        self.assertEqual(len(hook_grads), 1)
        self.assertEqual(hook_grads[0], torch.full((3,), 5.0))

    def test_hook_closure_captures_external_state(self):
        # [FUNCTION DOC] The hook is a closure over an external string tag and prints a line with
        # the tag and the gradient norm. With standard output redirected to a buffer during
        # backward, asserts the output contains the tag text and a norm entry.
        # Recommended pattern for threading non-tensor context into the hook:
        # capture it via closure at hook-registration time. Verify the captured
        # tag actually reaches printed output (mirrors the docstring example).
        tag = "intermediate"

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; prints the captured tag and the norm of the gradient
            # with four decimals.
            print(f"[{tag}][bwd] norm={x_grad.norm().item():.4f}")

        buf = io.StringIO()
        x = torch.randn(4, requires_grad=True)
        with contextlib.redirect_stdout(buf):
            my_fn(x)[0].sum().backward()

        output = buf.getvalue()
        self.assertIn("[intermediate][bwd]", output)
        self.assertIn("norm=", output)

    def test_hook_extra_signature_arg_raises_at_backward(self):
        # [FUNCTION DOC] A hook declaring an extra positional parameter beyond the gradient cannot
        # be called correctly. Forward succeeds, but backward must raise a TypeError about a
        # missing required positional argument named tag.
        # Capability limit: hooks must accept exactly N grad tensors, where N
        # is the number of requires_grad tensor inputs to the leaf function.
        # Declaring extra positional args crashes when the hook fires.
        @leaf_function
        def my_fn(x, tag):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled
            # (second argument unused).
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x, tag):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad, tag):  # extra `tag` arg — never supplied
            # [FUNCTION DOC] HELPER: Hook with an extra parameter that is never
            # supplied; does nothing.
            pass

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x, "label")[0]
        with self.assertRaisesRegex(
            TypeError,
            r"missing 1 required positional argument.*'tag'",
        ):
            out.sum().backward()

    def test_hook_multiple_tensor_inputs(self):
        # [FUNCTION DOC] Two tensor inputs requiring grad; the hook takes two gradients in input
        # order. Asserts it fires once with gradients of all twos for the first input and all
        # threes for the second.
        hook_calls = []

        @leaf_function
        def my_fn(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with two times the first input
            # plus three times the second.
            return (x * 2 + y * 3,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the first input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad, y_grad):
            # [FUNCTION DOC] HELPER: Hook; appends the pair of gradient copies to the list.
            hook_calls.append((x_grad.clone(), y_grad.clone()))

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        out = my_fn(x, y)[0]
        out.sum().backward()

        self.assertEqual(len(hook_calls), 1)
        self.assertEqual(hook_calls[0][0], torch.full((3,), 2.0))
        self.assertEqual(hook_calls[0][1], torch.full((3,), 3.0))

    def test_hook_only_fires_for_requires_grad_inputs(self):
        # [FUNCTION DOC] With one input requiring grad and one not, the hook receives only the
        # gradient of the requiring-grad input. Asserts one call with a gradient of all fives.
        # No-grad tensors are filtered out: hook receives grads only for the
        # requires_grad inputs, in order.
        hook_calls = []

        @leaf_function
        def my_fn(x, y):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with five times the first input
            # plus the second.
            return (x * 5 + y,)

        @my_fn.register_fake
        def my_fn_fake(x, y):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the first input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook taking one gradient; appends a copy to the list.
            hook_calls.append(x_grad.clone())

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=False)
        out = my_fn(x, y)[0]
        out.sum().backward()

        self.assertEqual(len(hook_calls), 1)
        self.assertEqual(hook_calls[0], torch.full((3,), 5.0))

    def test_hook_no_requires_grad_no_fire(self):
        # [FUNCTION DOC] When no input requires grad the hook never fires; asserts its
        # counter stays zero.
        hook_count = [0]

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; increments the single-element counter list.
            hook_count[0] += 1

        x = torch.randn(3, requires_grad=False)
        my_fn(x)[0]
        self.assertEqual(hook_count[0], 0)

    def test_hook_side_effect_only_fn(self):
        # [FUNCTION DOC] A leaf function that returns nothing but has a hook. It is called on a
        # tensor derived from a grad-requiring tensor, then backward is run on that tensor's sum.
        # Asserts the real body ran (flag set) and the hook fired once.
        fwd_called = [False]
        hook_grads = []

        @leaf_function
        def log_fn(x, tag):
            # [FUNCTION DOC] HELPER: Real body; sets the flag in the enclosing list
            # and returns nothing.
            fwd_called[0] = True
            return None

        @log_fn.register_fake
        def log_fn_fake(x, tag):
            # [FUNCTION DOC] HELPER: Fake; returns nothing.
            return None

        @log_fn.register_multi_grad_hook
        def log_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; appends a copy of the received gradient to the list.
            hook_grads.append(x_grad.clone())

        x = torch.randn(4, requires_grad=True)
        y = x * 2
        log_fn(y, "test")
        y.sum().backward()

        self.assertTrue(fwd_called[0])
        self.assertEqual(len(hook_grads), 1)

    def test_hook_gradient_values_correct(self):
        # [FUNCTION DOC] Leaf function squaring its input with a hook. For a known input, asserts
        # the hook gradient and the input's own gradient both equal twice the input.
        hook_grads = []

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input squared.
            return (x**2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; appends a copy of the received gradient to the list.
            hook_grads.append(x_grad.clone())

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()

        self.assertEqual(hook_grads[0], torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(x.grad, torch.tensor([2.0, 4.0, 6.0]))

    def test_hook_with_downstream_computation(self):
        # [FUNCTION DOC] The leaf output is multiplied by three downstream; the hook gradient for
        # the doubling function must therefore be six for every element.
        hook_grads = []

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; appends a copy of the received gradient to the list.
            hook_grads.append(x_grad.clone())

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = my_fn(x)[0]
        z = y * 3
        z.sum().backward()

        self.assertEqual(hook_grads[0], torch.tensor([6.0, 6.0]))

    def test_hook_with_retain_graph(self):
        # [FUNCTION DOC] Runs a single backward and asserts the hook counter is exactly one.
        # Despite its name, the test does not pass retain_graph or run a second backward.
        # KNOWN ISSUE: test name suggests retain_graph coverage but the code never
        # retains the graph.
        hook_count = [0]

        @leaf_function
        def my_fn(x):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with the input doubled.
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized tensor like the input
            # in a tuple.
            return (torch.empty_like(x),)

        @my_fn.register_multi_grad_hook
        def my_fn_hook(x_grad):
            # [FUNCTION DOC] HELPER: Hook; increments the single-element counter list.
            hook_count[0] += 1

        x = torch.randn(3, requires_grad=True)
        out = my_fn(x)[0]
        out.sum().backward()
        self.assertEqual(hook_count[0], 1)


@skipIfTorchDynamo("leaf_function tests manage their own compilation")
class TestLeafFunctionGradDtype(TestCase):
    # [CLASS DOC] Tests that leaf functions preserve the gradient dtype attribute of inputs: when a
    # tensor's gradient dtype differs from its own dtype, the backward through a leaf function must
    # still deliver gradients in the requested dtype, bitwise equal to eager. Stateless.
    def _make_pair(self, size, dtype, grad_dtype):
        # [FUNCTION DOC] HELPER: Seeds the random generator (42), creates a random tensor of the
        # given size and dtype requiring grad, and a detached clone also requiring grad. If a
        # gradient dtype is given, sets it on both. Returns the reference and test tensors.
        torch.manual_seed(42)
        x_ref = torch.randn(size, dtype=dtype, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_(True)
        if grad_dtype is not None:
            x_ref.grad_dtype = grad_dtype
            x_test.grad_dtype = grad_dtype
        return x_ref, x_test

    @staticmethod
    def _precision_sensitive_fn(x, w):
        # [FUNCTION DOC] HELPER: Static; converts both arguments to float32 and returns a scalar:
        # the sum of their elementwise product plus a seventh of the sum of cubes of the first.
        return (x.float() * w.float()).sum() + (x.float() ** 3 / 7.0).sum()

    def test_grad_bitwise_equal_with_grad_dtype(self):
        # [FUNCTION DOC] With bfloat16 inputs and a float32 gradient dtype, compares the gradient
        # from calling the computation directly with the one from calling it through a leaf
        # function; asserts they are bitwise identical.
        @leaf_function
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with
            # the precision-sensitive scalar.
            return (self._precision_sensitive_fn(x, w),)

        @fn.register_fake
        def fn_fake(x, w):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized float32 scalar on the input's
            # device in a tuple.
            return (torch.empty((), dtype=torch.float32, device=x.device),)

        x_ref, x_test = self._make_pair(64, torch.bfloat16, torch.float32)
        torch.manual_seed(7)
        w = torch.randn(64, dtype=torch.bfloat16)

        self._precision_sensitive_fn(x_ref, w).backward()
        fn(x_test, w)[0].backward()

        self.assertTrue(torch.equal(x_test.grad, x_ref.grad))

    @parametrize(
        "dtype,grad_dtype",
        [
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float16),
        ],
    )
    def test_backward_engine_with_dtype_grad_dtype_mismatch(self, dtype, grad_dtype):
        # [FUNCTION DOC] Parametrized over dtype and gradient dtype pairs that differ. Creates a
        # grad-requiring tensor with the gradient dtype set, calls a leaf function that records the
        # dtype and gradient dtype it sees, runs backward, and asserts the inner function saw the
        # original dtypes and that the resulting gradient has the requested gradient dtype.
        observed_inner = {}

        @leaf_function
        def fn(x, w):
            # [FUNCTION DOC] HELPER: Real body; records the input's dtype and grad dtype into the
            # enclosing dictionary and returns a one-tuple with the float32 dot-product sum.
            observed_inner["dtype"] = x.dtype
            observed_inner["grad_dtype"] = x.grad_dtype
            return ((x.float() * w.float()).sum(),)

        @fn.register_fake
        def fn_fake(x, w):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized float32 scalar on the input's
            # device in a tuple.
            return (torch.empty((), dtype=torch.float32, device=x.device),)

        torch.manual_seed(0)
        x = torch.randn(8, dtype=dtype, requires_grad=True)
        x.grad_dtype = grad_dtype
        w = torch.randn(8, dtype=dtype)
        self.assertNotEqual(x.dtype, x.grad_dtype)

        fn(x, w)[0].backward()

        self.assertEqual(observed_inner["dtype"], dtype)
        self.assertEqual(observed_inner["grad_dtype"], grad_dtype)
        self.assertEqual(x.grad.dtype, grad_dtype)

    def test_nonleaf_input_does_not_raise(self):
        # [FUNCTION DOC] Like the bitwise-equality test but the second input is a non-leaf tensor
        # (computed from another by multiplication); asserts no error occurs and the gradient of
        # the first input is bitwise equal to the direct computation.
        @leaf_function
        def fn(x, b):
            # [FUNCTION DOC] HELPER: Real body; returns a one-tuple with
            # the precision-sensitive scalar.
            return (self._precision_sensitive_fn(x, b),)

        @fn.register_fake
        def fn_fake(x, b):
            # [FUNCTION DOC] HELPER: Fake; returns an uninitialized float32 scalar on the input's
            # device in a tuple.
            return (torch.empty((), dtype=torch.float32, device=x.device),)

        x_ref, x_test = self._make_pair(64, torch.bfloat16, torch.float32)
        torch.manual_seed(7)
        b_leaf = torch.randn(64, dtype=torch.bfloat16)
        b_ref = b_leaf * 2
        b_test = b_leaf * 2

        self._precision_sensitive_fn(x_ref, b_ref).backward()
        fn(x_test, b_test)[0].backward()

        self.assertTrue(torch.equal(x_test.grad, x_ref.grad))


instantiate_parametrized_tests(TestLeafFunctionGradDtype)


if __name__ == "__main__":
    run_tests()
