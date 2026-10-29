# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the torch._dynamo decorator and stance APIs: disable, allow_in_graph,
# nonstrict_trace, graph_break, skip_frame, step_unsupported, substitute_in_graph, set_stance,
# dont_skip_tracing, patch_dynamo_config, error_on_graph_break, mark_static_address,
# assume_constant_result and override_cudagraphs. Contains one module-level helper function and
# one test class, instantiated for parametrization at import. Run as a script it calls the
# dynamo test runner.
import functools
import operator
import os
import re
import unittest
import unittest.mock as mock
from unittest.mock import patch

import torch
import torch._dynamo.testing
from torch._dynamo.backends.debugging import invoke_subgraph_inner_compiler
from torch._dynamo.exc import Unsupported
from torch._dynamo.trace_rules import is_callable_allowed
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    IS_MACOS,
    parametrize,
    skipIfWindows,
    TEST_WITH_ASAN,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def my_custom_function(x):
    # [FUNCTION DOC] HELPER: Trivial module-level function returning its input plus one; used as
    # the target of allow_in_graph registration in tests.
    return x + 1


class DecoratorTests(PytreeRegisteringTestCase):
    # [CLASS DOC] Test class for the dynamo decorator and stance APIs: disable, allow_in_graph,
    # nonstrict_trace, graph_break and skip helpers, set_stance, dont_skip_tracing,
    # error_on_graph_break, mark_static, and override_cudagraphs. Pytree registration helpers
    # come from the base class. Tests are stateless apart from global dynamo registries that
    # individual tests restore.
    def test_disallow_in_graph(self):
        # [FUNCTION DOC] Compiles a function with five tensor ops while disallow_in_graph is
        # active for torch.sub. Expects a graph break at the sub call: two frames and four ops
        # in total. The allow_in_graph call afterwards restores global registry state.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.disallow_in_graph(torch.sub)
        fn(torch.randn(10))
        torch._dynamo.allow_in_graph(torch.sub)

        # check for graph break on sub
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_invoke_subgraph_wrapper_is_allow_in_graph(self):
        # [FUNCTION DOC] Builds a boxed wrapper with invoke_subgraph_inner_compiler on a traced
        # add-one module, checks its only free variable is the unboxed wrapper, and asserts that
        # inner closure is registered as allowed in graph so dynamo does not fall back to a
        # disable graph break.
        # `invoke_subgraph_inner_compiler` returns a boxed wrapper that, when
        # called, invokes an inner closure decorated with `@disable` and
        # `@torch._dynamo.allow_in_graph`. The id of that inner closure must
        # be in the allow_in_graph registry — otherwise dynamo's
        # `lookup_callable` will fall through to the disable check and graph
        # break, defeating the point of `allow_in_graph`.
        def f(x):
            return x + 1

        gm = torch.fx.symbolic_trace(f)
        boxed = invoke_subgraph_inner_compiler(gm, [torch.randn(4)])

        # The boxed wrapper closes over `invoke_subgraph_wrapper_unboxed`.
        self.assertEqual(
            boxed.__code__.co_freevars, ("invoke_subgraph_wrapper_unboxed",)
        )
        unboxed = boxed.__closure__[0].cell_contents

        self.assertTrue(is_callable_allowed(unboxed))

    def test_disable_for_custom_op(self):
        # [FUNCTION DOC] Defines a scratch custom op library with a CPU kernel using nonzero,
        # wraps the op in dynamo disable, compiles a relu/op/cos function and expects two frames
        # and results equal to eager. The original op attribute is restored in a finally block.
        import torch.library

        with torch.library._scoped_library("foo", "DEF") as foo:
            foo.define("custom(Tensor self) -> Tensor")

            # Dynamic shape data dependent operator. For static shape compilation, Dynamo
            # should graph break on it. But, the meta kernel is not implemented properly.
            @torch.library.impl(foo, "custom", "CPU")
            def foo_cpu(x):
                return x.nonzero()

            # Disallow does not work because of extra python frames with torch.library python API
            orig_custom = torch.ops.foo.custom
            try:
                torch.ops.foo.custom = torch._dynamo.disable(torch.ops.foo.custom)

                def fn(x):
                    a = torch.nn.functional.relu(x)
                    b = torch.ops.foo.custom(a)
                    c = torch.cos(b)
                    return c

                x = torch.randint(2, (100,))
                ref = fn(x)

                cnts = torch._dynamo.testing.CompileCounter()
                opt_fn = torch.compile(fn, backend=cnts)
                res = opt_fn(x)
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(ref, res)
            finally:
                torch.ops.foo.custom = orig_custom

    def test_disable_not_traced_and_correct(self):
        # [FUNCTION DOC] A disabled inner function must not be traced: compiled output equals
        # eager and the graph break around the call yields two compiled frames.
        # disable must keep Dynamo from tracing into the function while still
        # returning correct results (exercises the non-export hot path).
        @torch._dynamo.disable
        def inner(x):
            return x + 1

        def fn(x):
            return torch.cos(inner(torch.sin(x)))

        x = torch.randn(4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        res = torch.compile(fn, backend=cnts)(x)
        self.assertEqual(ref, res)
        # inner is disabled -> graph break around it -> two compiled frames.
        self.assertEqual(cnts.frame_count, 2)

    def test_disable_under_eager_on_recompile_stance(self):
        # [FUNCTION DOC] Calls a disabled function twice under the eager_on_recompile stance and
        # checks the results are still correct (no compilation expected).
        # A disabled function honors the stance for its body: under
        # eager_on_recompile the stance callback is False (run-only), not fully
        # off. Pin that it still returns correct results.
        @torch._dynamo.disable
        def inner(x):
            return x + 1

        x = torch.randn(4)
        with torch.compiler.set_stance("eager_on_recompile"):
            self.assertEqual(inner(x), x + 1)
            self.assertEqual(inner(x), x + 1)

    def test_disable_ignores_outer_wraps(self):
        # [FUNCTION DOC] Builds a wrapper via functools wraps around a function carrying an
        # original-callable attribute and passes it to disable with recursive set. Only checks
        # the call does not run the wrapper or raise; the known limitation is documented in the
        # inline comment.
        def orig_inner():
            pass

        def inner():
            pass

        inner._torchdynamo_orig_callable = orig_inner

        @functools.wraps(inner)
        def wrapper():
            raise AssertionError("wrapper called")

        # This behavior is not ideal, but supporting it would add overhead
        # to callsites of eval_frame.innermost_fn. A warning would also be very noisy.
        torch._dynamo.disable(fn=wrapper, recursive=True)

    def test_disable_nn_modules_forward_hook(self):
        # [FUNCTION DOC] Disables a submodule that has a forward pre-hook and compiles the
        # parent with the eager counting backend. Expects two frames; the first graph has the
        # sin node and no sigmoid, and the second graph has no sigmoid either, showing neither
        # hook nor disabled module is traced.
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        # Disable my monkeypatching
        model.layer0 = torch._dynamo.disable(model.layer0)

        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_model = torch.compile(model, backend=cnts)
        opt_model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_disable_nn_module_with_class_decorator(self):
        # [FUNCTION DOC] Same scenario as the forward-hook test but disable is applied as a
        # class decorator on the submodule class and torch.compile as a class decorator on the
        # model. Expects two frames and no sigmoid node in either captured graph.
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch._dynamo.disable
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        @torch.compile(backend=cnts)
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_allow_in_graph(self):
        # [FUNCTION DOC] With allow_in_graph registered for the module-level custom function, a
        # compiled function calling it compiles to a single frame with five ops. Unregisters
        # afterwards.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.allow_in_graph(my_custom_function)
        fn(torch.randn(10))
        torch._dynamo.disallow_in_graph(my_custom_function)

        # check for no graph break
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_allow_in_graph_no_id_reuse(self):
        # [FUNCTION DOC] Registers a local function via allow_in_graph, deletes it, then
        # compiles a function calling a different function that graph breaks. Checks two frames,
        # i.e. registry lookups must not confuse a new function with a deleted one that had the
        # same identity.
        cnts = torch._dynamo.testing.CompileCounter()

        def do_allow_in_graph(x):
            return x + 1

        torch._dynamo.allow_in_graph(do_allow_in_graph)
        del do_allow_in_graph

        # `id(dont_allow_in_graph)` would likely match `id(do_allow_in_graph)`
        # We want to make sure Dynamo always trace through
        # `dont_allow_in_graph`, by checking for the explicit graph break.
        def dont_allow_in_graph(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = dont_allow_in_graph(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        fn(torch.randn(10))

        # Check for graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_incorrect_usage_disallow_in_graph(self):
        # [FUNCTION DOC] Applying disallow_in_graph as a bare decorator must raise a
        # RuntimeError whose message says disallow_in_graph is expected (to be given a list of
        # functions).
        with self.assertRaisesRegex(RuntimeError, "disallow_in_graph is expected"):

            @torch._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    def test_nonstrict_trace_tensor_args(self):
        # [FUNCTION DOC] nonstrict_trace on a function with tensor args that contains a graph
        # break: with fullgraph and aot_eager the compiled result must equal eager.
        @torch._dynamo.nonstrict_trace
        def trace_me(x, y, z):
            torch._dynamo.graph_break()
            return x * y + z

        def fn(x, y):
            t0 = x + 1
            t1 = trace_me(x, y, t0)
            t2 = t1 + y
            return t0 * t2

        x, y = torch.randn(10), torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_from_torch_compiler(self):
        # [FUNCTION DOC] Same as the tensor-args test but using the nonstrict_trace exposed
        # under torch.compiler.
        @torch.compiler.nonstrict_trace
        def trace_me(x, y, z):
            torch._dynamo.graph_break()
            return x * y + z

        def fn(x, y):
            t0 = x + 1
            t1 = trace_me(x, y, t0)
            t2 = t1 + y
            return t0 * t2

        x, y = torch.randn(10), torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict(self):
        # [FUNCTION DOC] A pre-existing dict argument is passed to a nonstrict_trace-ed function
        # that reads one key; compiled output must match eager under fullgraph.
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            t0 = trace_me(x, d)
            return t0 + 1

        x = torch.randn(10)
        d = {"a": 2}
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d)
        res = opt_fn(x, d)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_newly_constructed_dict_with_side_effects(self):
        # [FUNCTION DOC] A dict is created and mutated inside the compiled function, then passed
        # to a nonstrict_trace-ed function. Compiled output must match eager.
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x):
            d = {}
            d["a"] = 2
            t0 = trace_me(x, d)
            return t0 + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict_with_side_effects(self):
        # [FUNCTION DOC] The compiled function mutates an input dict before calling a
        # nonstrict_trace-ed function. Checks outputs match eager and that the mutated dict
        # contents are identical to the eager run, i.e. the side effect is replayed.
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            d["a"] = x + 1
            t0 = trace_me(x, d)
            return t0 + 2

        x = torch.randn(10)
        d0 = {"a": 0}
        d1 = dict(d0)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d0)
        res = opt_fn(x, d1)
        self.assertEqual(ref, res)
        self.assertEqual(d0, d1)

    def test_nonstrict_trace_pre_existing_custom_class(self):
        # [FUNCTION DOC] Registers a Point class as a pytree node and passes an instance into a
        # nonstrict_trace-ed function; results (including the point fields) must match eager.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            res = trace_me(p)
            return res, p.x, p.y

        p = Point(torch.ones(10), torch.ones(1))
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p)
        res = opt_fn(p)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_custom_class_with_side_effects(self):
        # [FUNCTION DOC] Like the previous test but the compiled function mutates the point
        # fields first. Checks outputs match and that the mutated fields on the compiled run
        # equal those of the eager run.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        p1 = Point(torch.ones(10), torch.ones(1))
        p2 = Point(torch.ones(10), torch.ones(1))
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p1)
        res = opt_fn(p2)
        self.assertEqual(ref, res)
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.y, p2.y)

    def test_nonstrict_trace_newly_constructed_custom_class_with_side_effects(self):
        # [FUNCTION DOC] The point is constructed, mutated, and passed to a nonstrict_trace-ed
        # function entirely inside the compiled function; output must match eager.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(x, y):
            p = Point(x, y)
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        x, y = torch.ones(10), torch.ones(1)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_nested_custom_class(self):
        # [FUNCTION DOC] Registers two pytree node classes (a point and a point-with-tensor
        # holding it) and traces a nonstrict function that takes the outer one and calls an
        # untraced helper on the inner. Compiled result must match eager.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        def trace_point(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        x, y = torch.ones(10), torch.ones(1)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_register_constant_type_guard(self):
        # [FUNCTION DOC] Registers an opaque custom class as a constant type and calls a
        # nonstrict_trace-ed function with instances. Same-valued instances reuse the compiled
        # code (frame count stays one) while a differently valued instance triggers a recompile
        # (frame count two).
        class State(torch._custom_class_base.CustomClassBase):
            def __init__(self, n):
                self.n = n

            def get_num(self):
                torch._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

            def __repr__(self):
                return f"State({self.n})"

            def __fx_repr__(self):
                return f"State({self.n})", {"State": State}

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `torch.compile`.
        torch._library.opaque_object.register_custom_class(State, typ="constant")

        @torch._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(fullgraph=True, backend=cnts)
        def fn(x, s):
            res = trace_me(x, s)
            return res

        x = torch.ones(10)
        # Make sure recompilation didn't happen.
        self.assertEqual(cnts.frame_count, 0)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)

        # Make sure recompilation did happen.
        fn(x, State(41))
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_int_and_float_output(self):
        # [FUNCTION DOC] The nonstrict function returns a Python int and a float; they are
        # usable as constants in the compiled graph and results match eager.
        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return len(x.shape), 0.42

        def fn(x):
            n1, n2 = trace_me(x)
            return x * n1 + n2

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_tuple_and_sym_int_output(self):
        # [FUNCTION DOC] The nonstrict function returns a tensor and a size value; with dynamic
        # shapes the size is a symbolic int and results must still match eager.
        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 1, x.size(0)

        def fn(x):
            t0, n = trace_me(x)
            return t0 * n

        x = torch.randn(10)
        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function(self):
        # [FUNCTION DOC] Calls nonstrict_trace on a module-external function inside the compiled
        # function and invokes it immediately; with fullgraph the result must match eager.
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = torch._dynamo.nonstrict_trace(trace_me)(x)
            return res + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function_kwarg(self):
        # [FUNCTION DOC] Same as above but passes the function to nonstrict_trace by its keyword
        # name, traceable_fn. Results must match eager.
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = torch._dynamo.nonstrict_trace(traceable_fn=trace_me)(x)
            return res + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_on_method(self):
        # [FUNCTION DOC] Decorates a method of a pytree-registered class with nonstrict_trace,
        # calls it on an instance built inside the compiled function, and compares with eager.
        class Num:
            def __init__(self, n):
                self.n = n

            @torch._dynamo.nonstrict_trace
            def trace_me(self, t):
                torch._dynamo.graph_break()
                return t + self.n

        self.register_pytree_node(
            Num,
            lambda num: ((num.n,), ()),
            lambda n, _: Num(n[0]),
            serialized_type_name=f"{Num.__module__}.{Num.__qualname__}",
        )

        def fn(x, n):
            num = Num(n)
            return num.trace_me(x)

        x, n = torch.randn(10), 42
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, n)
        res = opt_fn(x, n)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_captured_external_tensor(self):
        # [FUNCTION DOC] The nonstrict function closes over a tensor created outside the
        # compiled region; compiled output must match eager.
        cst = torch.ones(1)

        @torch._dynamo.nonstrict_trace
        def trace_me(x, y):
            torch._dynamo.graph_break()
            return x * y + cst

        def fn(x, y):
            return trace_me(x, y)

        x, y = torch.randn(10), torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_no_action_at_a_distance(self):
        # [FUNCTION DOC] Calling nonstrict_trace on a function without using its return value
        # must not change the original function: compiling a caller still graph breaks inside it
        # (two frames).
        def trace_me(x):
            x = x + 4
            torch._dynamo.graph_break()
            return x + 8

        # No effect on traceability of `trace_me`
        torch._dynamo.nonstrict_trace(trace_me)

        def fn(x):
            res = trace_me(x + 1)
            return res + 2

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnts)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        # There should be 1 graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_inside_compiled_function_error(self):
        # [FUNCTION DOC] Applying nonstrict_trace to a function defined inside the compiled
        # region must raise Unsupported with a message saying the function must be defined
        # outside the compile region.
        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            def trace_me(x, y):
                torch._dynamo.graph_break()
                return x * y

            res = torch._dynamo.nonstrict_trace(trace_me)(x, y)
            return res + 1

        try:
            fn(torch.ones(10), torch.ones(1))
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            msg = "Applying `nonstrict_trace` to function <trace_me>; however, `nonstrict_trace` currently requires the function to be defined outside `torch.compile` region."
            self.assertIn(msg, str(e))

    def test_nonstrict_trace_custom_class_error(self):
        # [FUNCTION DOC] Passing an instance of a class not registered with pytree to a
        # nonstrict_trace-ed function must raise Unsupported mentioning an invalid input type.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(p):
            res = trace_me(p)
            return res + 1

        try:
            p = Point(torch.ones(10), torch.ones(1))
            fn(p)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_nested_custom_class_error(self):
        # [FUNCTION DOC] A registered outer class contains an unregistered inner class instance;
        # calling a nonstrict function with it must raise Unsupported for an invalid input type.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        def trace_point(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        try:
            fn(torch.ones(10), torch.ones(1))
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_custom_class_output_error(self):
        # [FUNCTION DOC] A nonstrict_trace-ed function returning an unregistered custom class
        # must raise Unsupported mentioning an unsupported output type.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return Point(x, x + 1)

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            p = trace_me(x)
            return p.x * p.y

        try:
            x = torch.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Unsupported output type for nonstrict_trace-ed function", str(e)
            )

    def test_nonstrict_newly_constructed_trace_register_constant_type_error(self):
        # [FUNCTION DOC] Registers an opaque class as symbolic, constructs it inside the
        # compiled function and passes it to a nonstrict function; expects Unsupported saying an
        # opaque object was created in the middle of the program.
        class State(torch._custom_class_base.CustomClassBase):
            def __init__(self, n):
                self.n = n

            def get_num(self):
                torch._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `torch.compile`.
        torch._library.opaque_object.register_custom_class(State, typ="symbolic")

        @torch._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            s = State(10)
            res = trace_me(x, s)
            return res

        try:
            x = torch.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "An opaque object was created in the middle of the program.",
                str(e),
            )

    def test_nonstrict_trace_object_in_context_error(self):
        # [FUNCTION DOC] A pytree node stores a non-tensor object in its context; flattening it
        # for a nonstrict function must raise Unsupported about invalid use of pytree flatten.
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.t,), pt.p),
            lambda ts, p: PointTensor(p, ts[0]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(pt):
            torch._dynamo.graph_break()
            return pt.t + pt.p.x * pt.p.y

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_me(pt)
            return res

        try:
            x, y = torch.ones(10), torch.ones(1)
            fn(x, y)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Invalid use of pytree_flatten with nonstrict_trace-ed function", str(e)
            )

    def test_nonstrict_trace_nn_module_dict_input(self):
        # [FUNCTION DOC] Passes a dict of linear modules into a nonstrict function that applies
        # them; compiled output must match eager under fullgraph.
        @torch._dynamo.nonstrict_trace
        def trace_me(x, modules):
            torch._dynamo.graph_break()
            return modules["a"](x) + modules["b"](x)

        def fn(x, modules):
            return trace_me(x, modules) + 1

        linear_a = torch.nn.Linear(4, 4)
        linear_b = torch.nn.Linear(4, 4)
        modules = {"a": linear_a, "b": linear_b}
        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, modules)
        res = opt_fn(x, modules)
        self.assertEqual(ref, res)

    def test_graph_break(self):
        # [FUNCTION DOC] Three runs of two cosines separated by explicit graph breaks give three
        # frames and six ops.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            return x

        fn(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_skip_frame(self):
        # [FUNCTION DOC] skip_frame at the end of a function leaves it uncompiled (zero frames,
        # correct result). After an earlier graph break, skip_frame in the resumed part still
        # leaves one compiled frame for the first part and a correct result.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            torch._dynamo.skip_frame()
            return x + 1

        inp = torch.ones(3, 3)
        self.assertEqual(fn(inp), inp + 2)
        self.assertEqual(cnts.frame_count, 0)

        @torch.compile(backend=cnts)
        def gn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 1
            torch._dynamo.skip_frame()
            return x + 1

        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 1)

    def test_step_unsupported(self):
        # [FUNCTION DOC] step_unsupported after some ops: the ops before it compile (one frame,
        # two ops) and the rest runs eagerly, giving the correct sum.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1 + 2
            torch._dynamo.step_unsupported()
            return x + 4

        inp = torch.ones(3)
        self.assertEqual(fn(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_step_unsupported_empty_checkpoint(self):
        # [FUNCTION DOC] step_unsupported as the first statement must not fail and the function
        # result is correct.
        @torch.compile(backend="eager")
        def fn(x):
            torch._dynamo.step_unsupported()
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp), inp + 1)

    @skipIfWindows(
        msg="TODO: (xuhancn), confirm if torch.compiler.disable work on Windows."
    )
    def test_disable_recursive_false(self):
        # [FUNCTION DOC] Checks that disable with recursive false skips only the decorated
        # function body (two frames; compiling is reported as false inside it), and that
        # applying it to a function does not change the original: the original is compiled when
        # called, the disabled copy is run eagerly, in either order and across resets. Compiling
        # the disabled copy directly compiles it.
        def fn2(x):
            return x + 1

        @torch._dynamo.disable(recursive=False)
        def fn1(x):
            if torch.compiler.is_compiling():
                raise RuntimeError("bad")
            x = x.sigmoid()
            return fn2(x.cos())

        def fn(x):
            return fn1(x.tan())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 2)

        # test that applying disable nonrecursive doesn't modify the original function
        def fn3(x):
            if torch.compiler.is_compiling():
                return x - 1
            return fn2(x) + 2

        @torch.compile(backend=cnts)
        def outer(f, x):
            return f(x)

        inp = torch.ones(3)
        fn3_disabled = torch._dynamo.disable(fn3, recursive=False)

        torch._dynamo.reset()

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        torch._dynamo.reset()

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        # directly compiling a disabled function should result in a compile
        torch._dynamo.reset()
        cnts.clear()
        res = torch.compile(fn3_disabled, backend=cnts)(inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

    def test_disable_recursive_false_weird(self):
        # [FUNCTION DOC] Sets the code execution strategy of a function to skip its next run,
        # then calls a non-recursive disabled version through a compiled caller (eager result
        # expected). After restoring the default strategy, directly compiling the function must
        # give the compiled result.
        from torch._dynamo.types import FrameAction, FrameExecStrategy

        # test the case where the next invocation of the function is
        # manually skipped
        def fn(x):
            if torch.compiler.is_compiling():
                return x - 1
            return x + 1

        fn_disabled = torch._dynamo.disable(fn, recursive=False)

        torch._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
        )

        @torch.compile(backend="eager")
        def outer(fn, x):
            return fn(x)

        inp = torch.ones(3)
        self.assertEqual(outer(fn_disabled, inp), inp + 1)

        torch._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT)
        )

        self.assertEqual(torch.compile(fn, backend="eager")(inp), inp - 1)

    def test_fullgraph_eval_frame_override(self):
        # [FUNCTION DOC] Patches the instruction translator return code generation to append a
        # call to a global uncompiled function. With non-fullgraph compile that function is
        # itself compiled (two frames); with fullgraph it is skipped (zero frames) unless the
        # config flag making nested compile an error is set, in which case a RuntimeError is
        # expected. Afterwards non-fullgraph compiles it again after reset of its code. Uses
        # mock patch and config patch.
        # NOTE it is NOT enough to just call a torch.compile'd function in a compiled
        # function returned by the backend - this is because we apply disable(recursive=True)
        # to compiled functions and if we call a directly torch.compile'd function, that
        # "overrides" the disable(recursive=True) - i.e. this behavior is intentional.

        # Instead, we will patch symbolic_convert.InstructionTranslator.codegen_return_with_pops to
        # append a bunch of additional bytecode that will run a function that is not disabled.
        global inner

        y = torch.ones(3)

        def inner():
            nonlocal y
            y += 1

        from torch._dynamo.bytecode_transformation import (
            create_call_function,
            create_instruction,
            Instruction,
        )
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase

        old_codegen_return = InstructionTranslatorBase.codegen_return_with_pops

        def codegen_return_with_pops(self, *args) -> list[Instruction]:
            insts = old_codegen_return(*args)
            if not insts[-1].opname.startswith("RETURN"):
                raise AssertionError(
                    f"Expected RETURN instruction, got {insts[-1].opname}"
                )
            # to prevent infinite recursion
            if self.f_code.co_name != "inner":
                insts[-1:-1] = [
                    create_instruction("LOAD_GLOBAL", argval="inner"),
                    *create_call_function(0, True),
                    create_instruction("POP_TOP"),
                ]
            return insts

        def fn(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()

        with mock.patch(
            "torch._dynamo.symbolic_convert.InstructionTranslatorBase.codegen_return_with_pops",
            codegen_return_with_pops,
        ):
            # fullgraph=False will result in inner being traced!
            opt_fn_1 = torch.compile(fn, backend=cnts, fullgraph=False)

            # inner compiled
            opt_fn_1(torch.ones(3))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(y, torch.ones(3) + 1)

            torch._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # NOTE do not fully reset dynamo - to ensure eval frame override is applied for cache hits
            opt_fn_2 = torch.compile(fn, backend=cnts, fullgraph=True)

            with torch._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=False
            ):
                # fullgraph=True will result in inner being skipped!
                opt_fn_2(torch.ones(3))
                self.assertEqual(cnts.frame_count, 0)
                self.assertEqual(y, torch.ones(3) + 2)

            with torch._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=True
            ):
                # fullgraph=True will result in error when attempting to compile inner
                with self.assertRaisesRegex(
                    RuntimeError, "Dynamo: expected not to compile nested code"
                ):
                    opt_fn_2(torch.ones(3))

            torch._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # if we run fullgraph=False again, inner is compiled again (because we reset_code)
            opt_fn_1(torch.ones(3))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(y, torch.ones(3) + 3)

    def test_substitute_in_graph(self):
        # [FUNCTION DOC] Without a polyfill, a C function call causes one graph break.
        # Registering a substitute with a mismatching signature raises TypeError. With a
        # correctly matching polyfill both the original C function and the polyfill compile
        # under fullgraph with zero graph breaks and matching results.
        counters.clear()

        # NB: Choose another C function for test when we support operator.indexOf
        #     out of the box
        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 1)

        torch._dynamo.reset()
        counters.clear()

        with self.assertRaisesRegex(TypeError, "Signature mismatch"):

            @torch._dynamo.substitute_in_graph(operator.indexOf)
            def _(sequence, x):
                for i, item in enumerate(sequence):
                    if item is x or item == x:
                        return i
                raise ValueError("sequence.index(x): x not in sequence")

        @torch._dynamo.substitute_in_graph(operator.indexOf)
        def polyfill(a, b):
            for i, item in enumerate(a):
                if item is b or item == b:
                    return i
            raise ValueError("sequence.index(x): x not in sequence")

        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

        torch._dynamo.reset()
        counters.clear()

        cnts = torch._dynamo.testing.CompileCounter()
        fn = polyfill
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

    @patch.object(torch._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        # [FUNCTION DOC] With suppress_errors patched on, a compiled function calling a disabled
        # function gives two frames and four ops; compiling it from a fullgraph outer function
        # must raise Unsupported about skipping a disabled function.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        @torch.compile(backend=cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @torch.compile(backend=cnts, fullgraph=True)
        def fn3(x):
            return fn2(x)

        fn2(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with self.assertRaisesRegex(
            Unsupported, r"Skip calling `torch.compiler.disable\(\)`d function"
        ):
            fn3(torch.randn(4, 5))

    def test_disable_optimize(self):
        # [FUNCTION DOC] The compile option disable and the TORCHDYNAMO_DISABLE environment
        # variable (patched via a dict patch) both prevent any compilation: frame count stays
        # zero.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, disable=True)
        def f1(x):
            return x + 1

        f1(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        @torch.compile(backend=cnt, disable=True)
        def f2(x):
            return x + 1

        f2(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        with patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):

            @torch.compile(backend=cnt)
            def f3(x):
                return x + 1

            f3(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_torch_guards_stack_frame_register_inlining_disable(self):
        # [FUNCTION DOC] Compiles a module whose only helper is disabled while the tracing
        # context current-frame hook is mocked; no frame summaries may be recorded.
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.a = y

            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.a)

        e = encoder(2.0)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(e, backend="eager")(x)

        self.assertEqual(len(seen_frames), 0)

    def test_torch_guards_stack_frame_register_inlining_partially_disable(self):
        # [FUNCTION DOC] A module with one disabled and one normal helper compiles into two
        # frames and three ops (before and after the disabled call).
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @torch._dynamo.disable
            def helper_disabled(self, x, y):
                return x.sin() * y.cos()

            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

        e = encoder(y)

        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(e, backend=cnt)(x)

        # first frame is before disable, second frame is after disable
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)

    def _test_mark_static_address(self, guarded):
        # [FUNCTION DOC] HELPER: Shared body for the static address tests. Marks an input tensor
        # as static address with the given guard flag, compiles with a counting backend
        # recording whether the graph owns buffers, then calls with a fresh tensor. If guarded,
        # the first compile has a buffer and a second compile happens without one; if unguarded,
        # only one compile happens.
        # This test verifies that dynamo properly marks inputs as static
        # when using the mark_static_address API.
        # We expect the tensor to be present in the buffers attribute of the graph.

        compiles_with_buffers = 0
        compiles = 0

        def debug_compiler(gm, _):
            nonlocal compiles_with_buffers
            nonlocal compiles
            compiles_with_buffers += len(gm._buffers) > 0
            compiles += 1
            return gm

        @torch.compile(backend=debug_compiler)
        def fn(x):
            return x + 1

        inp = torch.ones(2)

        torch._dynamo.mark_static_address(inp, guard=guarded)

        fn(inp)
        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        inp2 = torch.ones(2)

        # if guarded, should trigger another recompile
        # since it was not marked static, compiles with buffers
        # should not be incremented
        fn(inp2)

        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        self.assertEqual(compiles, 2 if guarded else 1)

    def test_mark_static_address_guarded(self):
        # [FUNCTION DOC] Runs the shared helper with guarding on.
        self._test_mark_static_address(guarded=True)

    def test_mark_static_address_unguarded(self):
        # [FUNCTION DOC] Runs the shared helper with guarding off.
        self._test_mark_static_address(guarded=False)

    @parametrize("compile_outer", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_compile_staticmethod(self, compile_outer, fullgraph):
        # [FUNCTION DOC] Parametrized over decorator order and fullgraph. Applies torch.compile
        # to a static method either outside or inside the staticmethod wrapper; calling through
        # class and instance gives sin of the input with one frame and one op.
        cnt = torch._dynamo.testing.CompileCounter()
        compile_decorator = torch.compile(backend=cnt, fullgraph=fullgraph)

        if compile_outer:

            class Foo:
                @compile_decorator
                @staticmethod
                def bar(x):
                    return x.sin()

        else:

            class Foo:
                @staticmethod
                @compile_decorator
                def bar(x):
                    return x.sin()

        x = torch.randn(4)
        expected = x.sin()
        self.assertEqual(Foo.bar(x), expected)
        self.assertEqual(Foo().bar(x), expected)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_compile_staticmethod_caching_precompile(self):
        # [FUNCTION DOC] With the caching precompile config on and the dynamo cache cleared,
        # compiles a static method via decorator and checks class and instance calls work with
        # one frame.
        from torch._dynamo.package import DynamoCache

        DynamoCache.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class Foo:
            @torch.compile(backend=cnt)
            @staticmethod
            def bar(x):
                return x.sin()

        x = torch.randn(4)
        expected = x.sin()
        self.assertEqual(Foo.bar(x), expected)
        self.assertEqual(Foo().bar(x), expected)
        self.assertEqual(cnt.frame_count, 1)

    def test_class_methods(self):
        # [FUNCTION DOC] Compiles a function calling class methods, static methods and regular
        # methods on instances and classes of a base class and two subclasses, with a random op
        # to force custom bytecode. Checks returned values and one frame. The check for a
        # subclass class method called from a regular method is commented out in the test and
        # flagged by a TODO as unsupported.
        class A:
            @classmethod
            def my_class_method(cls, arg1):
                return cls, arg1

            @staticmethod
            def my_static_method(arg1):
                return None, arg1

            def my_regular_method(self, arg1):
                return self, arg1

        class B(A):
            def my_class_method(self, arg1):
                return super().my_class_method(arg1)

            def my_static_method(self, arg1):
                return super().my_static_method(arg1)

        class C(A):
            @classmethod
            def my_class_method(cls, arg1):
                return super().my_class_method(arg1)

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(a, b, c):
            # We want a function that does not graph break but
            # does generate custom bytecode
            v1 = a.my_class_method(1)
            v2 = A.my_class_method(2)
            v3 = a.my_static_method(3)
            v4 = A.my_static_method(4)
            v5 = a.my_regular_method(5)
            v6 = b.my_class_method(6)
            v7 = b.my_static_method(7)
            v8 = c.my_class_method(8)
            v9 = C.my_class_method(9)
            torch.rand(2)
            return v1, v2, v3, v4, v5, v6, v7, v8, v9

        a, b, c = A(), B(), C()
        v1, v2, v3, v4, v5, _, v7, v8, v9 = fn(a, b, c)

        self.assertEqual(v1, (A, 1))
        self.assertEqual(v2, (A, 2))
        self.assertEqual(v3, (None, 3))
        self.assertEqual(v4, (None, 4))
        self.assertEqual(v5, (a, 5))
        # TODO fix me: we do not resolve classmethods properly
        # from a regular method
        # self.assertEqual(v6, (B, 6))
        self.assertEqual(v7, (None, 7))
        self.assertEqual(v8, (C, 8))
        self.assertEqual(v9, (C, 9))

        self.assertEqual(cnt.frame_count, 1)

    def test_assume_constant_result_on_user_defined_fn(self):
        # [FUNCTION DOC] Marks a function building a full tensor as assumed constant, uses it in
        # a function that returns a Python list, with the batch dimension marked static and
        # scalar capture and dynamic-output-shape configs on. Compiled fullgraph dynamic result
        # must equal eager.
        @torch._dynamo.assume_constant_result
        def const_fn(n, s):
            return torch.full([n], s)

        def fn(B):
            B = const_fn(B.size(0), 13)
            X = B * 2
            return X.tolist()

        B_list = [8] * 32

        B = torch.tensor(B_list, dtype=torch.int32)
        torch._dynamo.decorators.mark_static(B, 0)

        with torch._dynamo.config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            self.assertEqual(
                fn(B),
                torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B),
            )

    def test_assume_constant_result_on_computation_with_graph_input(self):
        # [FUNCTION DOC] An assumed-constant predicate over a graph input chooses a branch; the
        # compiled eager backend result must equal eager.
        @torch._dynamo.assume_constant_result
        def check(y):
            return y[0].item() == 1

        def fn(x, y):
            if check(y):
                return x + 2
            else:
                return x + 1

        y = torch.tensor([1])
        x = torch.tensor(1)

        self.assertEqual(fn(x, y), torch.compile(fn, backend="eager")(x, y))

    def test_justknobs_check(self):
        # [FUNCTION DOC] A justknobs check inside a function compiles under fullgraph with
        # aot_eager; the result equals eager.
        def fn(x, y):
            if torch._utils_internal.justknobs_check("test", True):
                return x + y
            else:
                return x - y

        x = torch.randn(2, 2, device="cpu")
        y = torch.randn(2, 2, device="cpu")
        eager_out = fn(x, y)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        compiled_out = compiled_fn(x, y)
        self.assertEqual(eager_out, compiled_out)

    def test_assume_constant_result_on_cached_property(self):
        # [FUNCTION DOC] Marks the getter of a cached property as assumed constant and compiles
        # a module that multiplies by it under fullgraph; result must be close to eager.
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._cached_id = None

            @property
            def cached_id(self) -> int:
                if self._cached_id is None:
                    self._cached_id = 42
                return self._cached_id

            def forward(self, x):
                return x * self.cached_id

        torch._dynamo.assume_constant_result(MyModule.cached_id.fget)

        model = MyModule()
        x = torch.tensor([1.0, 2.0, 3.0])
        compiled_model = torch.compile(model, backend="eager", fullgraph=True)
        result_eager = model(x)
        result_compiled = compiled_model(x)

        self.assertTrue(torch.allclose(result_eager, result_compiled))

    def test_set_stance_aot_eager_then_compile(self):
        # [FUNCTION DOC] Under the aot_eager_then_compile stance three calls with different
        # sizes are made; the op count is two where it would have been four without the stance.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with torch.compiler.set_stance("aot_eager_then_compile"):
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})
            fn(4, torch.randn(4), {0: torch.randn(4)})

        # Would have been 4 without stance
        self.assertEqual(cnts.op_count, 2)

    def test_mark_static_nn_module(self):
        # [FUNCTION DOC] Marks a module class as static, compiles three instances with different
        # constants and expects three frames (without static marking the constants would become
        # symbolic ints and fewer compiles would occur).
        @torch._dynamo.mark_static
        class Mock(torch.nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c

            def forward(self, x):
                return x * self.c

        cnts = torch._dynamo.testing.CompileCounter()
        mod1 = Mock(10)
        mod2 = Mock(20)
        mod3 = Mock(30)
        opt_mod1 = torch.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = torch.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = torch.compile(mod3, backend=cnts, fullgraph=True)

        x = torch.randn(4, 4)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because self.c would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    @unittest.skipIf(
        IS_LINUX or IS_MACOS, "https://github.com/pytorch/pytorch/issues/148515"
    )
    def test_set_stance_eager_then_compile(self):
        # [FUNCTION DOC] Under the eager_then_compile stance three differently sized calls end
        # up with one compiled frame. Skipped on Linux and macOS (issue 148515).
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with torch.compiler.set_stance("eager_then_compile"):
            fn(1, torch.randn(1), {0: torch.randn(1)})
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})

        self.assertEqual(cnts.frame_count, 1)

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or IS_MACOS,
        "https://github.com/pytorch/pytorch/issues/148463",
    )
    def test_set_stance_eager_then_compile_with_graph_break(self):
        # [FUNCTION DOC] Like the previous test but the function has a graph break, so two
        # frames result. Skipped under ASAN, Linux and macOS (issue 148463).
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            y = torch.sin(y)
            torch._dynamo.graph_break()
            y = torch.cos(y)
            return x * y * z[0]

        with torch.compiler.set_stance("eager_then_compile"):
            fn(1, torch.randn(1), {0: torch.randn(1)})
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})

        # frame count 2 since we added a graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_set_stance_eager_then_compile_rank_change(self):
        # [FUNCTION DOC] Dynamic compile under eager_then_compile: two same-rank calls then a
        # rank change gives two frames and correct outputs.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True)
        def fn(x):
            return x + 1

        with torch.compiler.set_stance("eager_then_compile"):
            x1 = torch.randn(10)
            self.assertEqual(fn(x1), x1 + 1)
            x2 = torch.randn(10)
            self.assertEqual(fn(x2), x2 + 1)
            x3 = torch.randn(10, 10)
            self.assertEqual(fn(x3), x3 + 1)

        self.assertEqual(cnts.frame_count, 2)

    def test_set_stance_eager_then_compile_delayed_rank_change(self):
        # [FUNCTION DOC] Repeats same-rank calls under eager_then_compile before a rank-four
        # input arrives; two frames and correct outputs.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True)
        def fn(x):
            return x + 1

        x1 = torch.randn(4)
        x2 = torch.randn(4)

        with torch.compiler.set_stance("eager_then_compile"):
            self.assertEqual(fn(x1), x1 + 1)
            self.assertEqual(fn(x2), x2 + 1)
            self.assertEqual(fn(x2), x2 + 1)
            x3 = torch.randn(2, 3, 4, 5)
            self.assertEqual(fn(x3), x3 + 1)

        self.assertEqual(cnts.frame_count, 2)

    def test_set_stance_force_eager(self):
        # [FUNCTION DOC] force_eager as decorator, context manager, and global setting runs the
        # compiled function eagerly (adds two instead of one). A decorator alone leaves no
        # lasting side effect; the global stance is reset to default at the end.
        @torch.compile(backend="eager")
        def a(x):
            if torch._dynamo.is_compiling():
                return x + 1
            return x + 2

        @torch.compiler.set_stance("force_eager")
        def b(x):
            return a(x)

        def c(x):
            out0 = a(x)
            with torch.compiler.set_stance("force_eager"):
                out1 = a(x)
            return out0, out1, a(x)

        inp = torch.ones(3)
        # test that decorating b has no overall side effect
        self.assertEqual(a(inp), inp + 1)

        self.assertEqual(b(inp), inp + 2)
        self.assertEqual(c(inp), (inp + 1, inp + 2, inp + 1))

        torch.compiler.set_stance("force_eager")
        self.assertEqual(a(inp), inp + 2)
        torch.compiler.set_stance("default")
        self.assertEqual(a(inp), inp + 1)

    def test_set_stance_eager_on_recompile(self):
        # [FUNCTION DOC] After one compiled call, under eager_on_recompile a repeat call with the same
        # argument is served by the compiled code while a call with a new argument value (which
        # would need a recompile) runs the plain Python path. Checks the three results.
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("eager_on_recompile"):
            out2 = a(inp, 1)
            out3 = a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)
        self.assertEqual(out3, inp + 4)

    def test_set_stance_fail_on_recompile(self):
        # [FUNCTION DOC] Under fail_on_recompile a cache hit works but a call needing
        # recompilation raises RuntimeError mentioning the stance.
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            out2 = a(inp, 1)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)

    def test_fail_on_recompile_shows_guard_details(self):
        # [FUNCTION DOC] Compiles twice with different sizes, then under fail_on_recompile calls
        # with a third size and checks the error text lists the function, the line number
        # (normalized) and the failure reasons of the cached guards, via an inline expected
        # string with munging.
        @torch.compile(backend="eager", dynamic=False)
        def f(x):
            return x + 1

        f(torch.ones(4))
        f(torch.ones(5))

        def post_munge(s):
            return re.sub(r"line number: \d+", "line number: N", s)

        with torch.compiler.set_stance("fail_on_recompile"):
            f(torch.ones(4))
            self.assertExpectedInlineMunged(
                RuntimeError,
                lambda: f(torch.ones(7)),
                """\
Detected recompile when torch.compile stance is 'fail_on_recompile'. filename: 'test_decorators.py', function name: 'f', line number: N
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 7
    - 0/1: tensor 'x' size mismatch at index 0. expected 5, actual 7""",
                post_munge=post_munge,
            )

    def test_set_stance_fail_on_recompile_with_disable(self):
        # [FUNCTION DOC] A disabled callee inside a compiled function does not raise under
        # fail_on_recompile.
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))
        # should not raise error
        with torch.compiler.set_stance("fail_on_recompile"):
            f(torch.randn(3, 3))

    def test_set_stance_forbid_in_graph(self):
        # [FUNCTION DOC] set_stance used as decorator or context manager, or through the private
        # set function, in traced code must raise AssertionError about tracing a forbidden
        # callable. When the frame is skipped, using the stance inside raises RuntimeError
        # saying set_stance in a compiled function is invalid.
        @torch.compiler.set_stance("force_eager")
        def a(x):
            return x + 1

        @torch.compile(backend="eager")
        def b(x):
            return a(x)

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            b(torch.ones(3))

        @torch.compile(backend="eager")
        def c(x):
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            c(torch.ones(3))

        @torch.compile(backend="eager")
        @torch.compiler.set_stance("force_eager")
        def d(x):
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            d(torch.ones(3))

        @torch.compile(backend="eager")
        def e(x):
            with torch._dynamo.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            e(torch.ones(3))

        @torch.compile(backend="eager")
        def f(x):
            torch._dynamo.eval_frame._set_stance("force_eager")
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            f(torch.ones(3))

        @torch.compile(backend="eager")
        def g(x):
            torch._dynamo.skip_frame()
            # NOTE: torch._dynamo.is_compiling() will get traced
            # and return true. torch.compiler.is_compiling() is skipped
            # and will return false.
            if torch.compiler.is_compiling():
                raise RuntimeError("Expect this frame to be skipped")
            # should not be traced, but eval frame callback is still set
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(RuntimeError, "set_stance in a torch.compile"):
            g(torch.ones(3))

    def test_set_stance_force_backend(self):
        # [FUNCTION DOC] A stance with force_backend overrides the backend of a compiled
        # function (the counting backend sees one frame). A named backend runs without crashing.
        # Combining force_eager with a forced backend raises RuntimeError.
        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def a(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compiler.set_stance("default", force_backend=cnts)
        def b(x):
            return a(x)

        b(torch.ones(3))

        self.assertEqual(cnts.frame_count, 1)

        @torch.compiler.set_stance("default", force_backend="eager")
        def c(x):
            return a(x)

        # just make sure this doesn't crash
        c(torch.ones(3))

        with self.assertRaisesRegex(RuntimeError, "force_backend"):

            @torch.compiler.set_stance("force_eager", force_backend="eager")
            def d(x):
                pass

    def test_set_stance_force_backend_with_disable(self):
        # [FUNCTION DOC] A forced failing backend is not invoked for a disabled callee: no error
        # raised.
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))

        def fail_backend(gm, ex):
            raise RuntimeError("fail!")

        # should not raise error
        with torch.compiler.set_stance("default", force_backend=fail_backend):
            f(torch.randn(3, 3))

    # also tests a lot of torch._dynamo.patch_dynamo_config functionality
    def test_dont_skip_tracing(self):
        # [FUNCTION DOC] Exercises dont_skip_tracing on functions from a test helper module that
        # trace rules skip: as function wrapper, inside traced code, as decorator, on
        # already-skipped functions, with graph breaks, nested, applied outside a compiled
        # function, as context manager from inside and outside, and across regions. Checks frame
        # counts and numeric results at each step, resetting dynamo once where a skipped frame
        # needs a clean state.
        from torch._dynamo.test_dont_skip_tracing_functions import f1, f3, f4, f5, f6

        cnts = torch._dynamo.testing.CompileCounter()

        # make sure test_dont_skip_tracing_functions is actually skipped by trace rules
        torch.compile(f1, backend=cnts)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 0)

        f1_unskip = torch._dynamo.dont_skip_tracing(f1)

        # basic test
        def g1(x):
            return f1_unskip(x)

        cnts.clear()
        torch.compile(g1, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is traceable
        def g2(x):
            return torch._dynamo.dont_skip_tracing(f1)(x)

        cnts.clear()
        torch.compile(g2, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to non-skipped function
        @torch._dynamo.dont_skip_tracing
        def g3(x):
            return f1(x)

        cnts.clear()
        torch.compile(g3, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to skipped function
        f3_unskip = torch._dynamo.dont_skip_tracing(f3)
        cnts.clear()
        torch.compile(f3_unskip, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test dont_skip_tracing with graph breaks
        inp = torch.ones(3)
        res = torch.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 6)

        @torch.compile(backend=cnts)
        def g4(x):
            x = f5(x, 1)
            x = torch._dynamo.dont_skip_tracing(f6)(x)
            x = f5(x, 8)
            return x

        res = g4(inp)
        self.assertEqual(res, inp + 6)

        # test nested dont_skip_tracing
        # this also happens to test if a previously skipped frame (f4)
        # can actually be compiled if called as a top-level function (in the case of a graph break)
        # TODO the reset is necessary for now since attempting to trace f4 previously
        # resulted in an unconditional skip
        torch._dynamo.reset()
        f4_unskip = torch._dynamo.dont_skip_tracing(f4)
        res = torch.compile(f4_unskip, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test dont_skip_tracing that is activated outside torch.compile
        f4_unskip2 = torch._dynamo.dont_skip_tracing(torch.compile(f4, backend=cnts))
        res = f4_unskip2(inp)
        self.assertEqual(res, inp + 15)

        # test context manager from inside
        @torch.compile(backend=cnts)
        def g5(x):
            x = f5(x, 1)
            with torch._dynamo.dont_skip_tracing():
                x = f5(x, 2)
                torch._dynamo.graph_break()
                x = f5(x, 4)
            x = f5(x, 8)
            return x

        res = g5(inp)
        self.assertEqual(res, inp + 6)

        # test context manager from outside
        with torch._dynamo.dont_skip_tracing():
            res = torch.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test skipped function from different dont_skip_tracing regions
        @torch.compile(backend=cnts)
        def g6(x):
            fn1 = f5
            with torch._dynamo.dont_skip_tracing():
                fn2 = f5
                x = fn1(x, 1)
            x = fn2(x, 2)
            return x

        res = g6(inp)
        self.assertEqual(res, inp + 1)

    def test_patch_dynamo_config_errors(self):
        # [FUNCTION DOC] patch_dynamo_config used in traced code must raise for an unknown key,
        # a dict value that is not a safe constant, a dict passed as the only positional
        # argument, and an argument that cannot be converted to a constant; the messages are
        # matched by regular expressions.
        @torch.compile(backend="eager")
        def f1(x):
            with torch._dynamo.patch_dynamo_config(nonexistent=False):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f1(torch.randn(3))

        @torch.compile(backend="eager")
        def f2(x):
            with torch._dynamo.patch_dynamo_config("verbose", {"a": 1}):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "patch_dynamo_config does not support .* with non-safe-constant"
        ):
            f2(torch.randn(3))

        @torch.compile(backend="eager")
        def f3(x):
            with torch._dynamo.patch_dynamo_config({"recompile_limit": 1}):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f3(torch.randn(3))

        @torch.compile(backend="eager")
        def f4(x):
            with torch._dynamo.patch_dynamo_config(verbose=object()):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "Cannot convert patch_dynamo_config args/kwargs to constants."
        ):
            f4(torch.randn(3))

    def test_error_on_graph_break(self):
        # [FUNCTION DOC] error_on_graph_break as decorator and context manager: disabling inside
        # allows graph breaks (two frames, then three frames with multiple breaks and with a
        # skipped frame), enabling inside raises Unsupported, results are correct otherwise.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f1(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 2

        inp = torch.ones(3)
        self.assertEqual(f1(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

        @torch.compile(backend=cnts)
        def f2(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f2(inp)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f3(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
                x = x + 2
                torch._dynamo.graph_break()
            return x + 4

        cnts.clear()
        self.assertEqual(f3(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner_f4(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f4(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.skip_frame()
                return inner_f4(x)

        cnts.clear()
        self.assertEqual(f4(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

    def test_error_on_graph_break_skips_redundant_set(self):
        # [FUNCTION DOC] With the stored flag already false, entering a context that sets false
        # must not call the setter (the mocked setter raises if called).
        torch._dynamo.utils._set_error_on_graph_break(False)

        with mock.patch(
            "torch._dynamo.decorators._set_error_on_graph_break",
            side_effect=AssertionError("redundant error_on_graph_break set"),
        ):
            with torch._dynamo.error_on_graph_break(False):
                pass

    def test_error_on_graph_break_reentrant_context_restores_outer_state(self):
        # [FUNCTION DOC] Re-entering the same context object nests correctly: the flag stays
        # false inside both levels and returns to true afterwards. The flag is reset to false at
        # the end.
        torch._dynamo.utils._set_error_on_graph_break(True)
        cm = torch._dynamo.error_on_graph_break(False)

        try:
            with cm:
                self.assertFalse(torch._dynamo.utils._get_error_on_graph_break())
                with cm:
                    self.assertFalse(torch._dynamo.utils._get_error_on_graph_break())

            self.assertTrue(torch._dynamo.utils._get_error_on_graph_break())
        finally:
            torch._dynamo.utils._set_error_on_graph_break(False)

    def test_error_on_graph_break_compile_wrapper_skips_redundant_set(self):
        # [FUNCTION DOC] optimize with the flag false and the stored flag false must not call
        # the setter in eval_frame (mocked to raise); the optimized function still returns the
        # right value.
        torch._dynamo.utils._set_error_on_graph_break(False)

        def fn(x):
            return x + 1

        opt_fn = torch._dynamo.optimize("eager", error_on_graph_break=False)(fn)
        x = torch.ones(3)

        with mock.patch(
            "torch._dynamo.eval_frame._set_error_on_graph_break",
            side_effect=AssertionError("redundant error_on_graph_break set"),
        ):
            self.assertEqual(opt_fn(x), fn(x))

    def test_error_on_graph_break_nested(self):
        # [FUNCTION DOC] Checks the flag in nested frames: an inner function decorated false
        # inside an outer true region may break, a context manager inside the callee behaves the
        # same, and inner true inside outer false raises Unsupported.
        # error_on_graph_break in a nested frame
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner_f5(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f5(x):
            x = x + 1
            return inner_f5(x)

        inp = torch.ones(3)
        self.assertEqual(f5(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f6(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f6(x):
            x = x + 1
            return inner_f6(x)

        cnts.clear()
        self.assertEqual(f6(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f7(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f7(x):
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(Unsupported):
            f7(inp)

    def test_error_on_graph_break_nested_with_skip(self):
        # [FUNCTION DOC] Same as the nested test but with a skipped frame between outer and
        # inner (three frames), and a non-recursive disabled middle function case that must
        # raise Unsupported.
        # error_on_graph_break in a nested frame with a skipped frame in between
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner2_f8(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        def inner1_f8(x):
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.skip_frame()
            return inner2_f8(x)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f8(x):
            x = x + 1
            return inner1_f8(x)

        inp = torch.ones(3)
        self.assertEqual(f8(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner2_f9(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.disable(recursive=False)
        def inner1_f9(x):
            return inner2_f9(x)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f9(x):
            x = x + 1
            return inner1_f9(x)

        with self.assertRaises(Unsupported):
            f9(inp)

        # test export with error_on_graph_break(False) still errors

    def test_error_on_graph_break_export(self):
        # [FUNCTION DOC] export of a function whose callee has the flag false and a graph break
        # must still raise Unsupported.
        @torch._dynamo.error_on_graph_break(False)
        def inner(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        def f(x):
            x = x + 1
            return inner(x)

        with self.assertRaises(Unsupported):
            torch._dynamo.export(f)(torch.ones(3))

    def test_error_on_graph_break_nested_deep(self):
        # [FUNCTION DOC] Four levels of nested calls with the flag toggled by a context manager
        # at level three: false inside outer true allows the break (two frames); true inside
        # outer false raises Unsupported.
        cnts = torch._dynamo.testing.CompileCounter()

        def inner1_f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def inner2_f1(x):
            return inner1_f1(x)

        def inner3_f1(x):
            with torch._dynamo.error_on_graph_break(False):
                return inner2_f1(x)

        def inner4_f1(x):
            return inner3_f1(x)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f1(x):
            x = x + 4
            return inner4_f1(x)

        inp = torch.ones(3)
        self.assertEqual(f1(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner1_f2(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def inner2_f2(x):
            return inner1_f2(x)

        def inner3_f2(x):
            with torch._dynamo.error_on_graph_break(True):
                return inner2_f2(x)

        def inner4_f2(x):
            return inner3_f2(x)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f2(x):
            x = x + 4
            return inner4_f2(x)

        with self.assertRaises(Unsupported):
            f2(inp)

    def test_error_on_graph_break_error(self):
        # [FUNCTION DOC] Invalid uses of the error_on_graph_break context manager in compiled
        # code (unknown keyword, no argument, non-bool argument) each raise an exception.
        @torch.compile(backend="eager")
        def f1():
            with torch._dynamo.error_on_graph_break(foo="bar"):
                pass

        @torch.compile(backend="eager")
        def f2():
            with torch._dynamo.error_on_graph_break():
                pass

        @torch.compile(backend="eager")
        def f3():
            with torch._dynamo.error_on_graph_break("foo"):
                pass

        with self.assertRaises(Exception):
            f1()
        with self.assertRaises(Exception):
            f2()
        with self.assertRaises(Exception):
            f3()

    def test_nested_compile_error_on_graph_break(self):
        # [FUNCTION DOC] Between nested compiled functions the innermost setting applies: inner
        # true under outer false raises Unsupported; inner false under outer true returns the
        # right value.
        inp = torch.ones(3)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def inner_f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend="eager")
        def f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            f1(inp)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend="eager")
        def inner_f2(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def f2(x):
            return inner_f2(x)

        self.assertEqual(f2(inp), inp + 3)

    def test_error_on_graph_break_fullgraph(self):
        # [FUNCTION DOC] Setting the flag false inside a fullgraph compile does not override
        # fullgraph: Unsupported is raised.
        # Test that error_on_graph_break=False cannot override fullgraph=True
        inp = torch.ones(3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f(inp)

    def test_error_on_graph_break_empty_graph(self):
        # [FUNCTION DOC] A compiled function with no tensor ops and the flag true runs fine and
        # returns one.
        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def f():
            return 1

        self.assertEqual(f(), 1)

    def test_error_on_graph_break_nonempty_checkpoint(self):
        # [FUNCTION DOC] With three ops compiled before a graph break inside a true context,
        # Unsupported is raised and no frames get compiled.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            x = x + 1
            x = x + 1
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 1

        with self.assertRaises(Unsupported):
            fn(torch.ones(3))

        self.assertEqual(cnts.frame_count, 0)

    def test_nested_compile_fullgraph(self):
        # [FUNCTION DOC] fullgraph true cannot be undone by an enclosing or enclosed compile
        # with fullgraph false: both orderings raise Unsupported.
        # Test that fullgraph=True cannot be toggled back by fullgraph=False
        inp = torch.ones(3)

        @torch.compile(backend="eager", fullgraph=True)
        def inner_f1(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend="eager", fullgraph=False)
        def outer_f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            outer_f1(inp)

        @torch.compile(backend="eager", fullgraph=False)
        def inner_f2(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def outer_f2(x):
            return inner_f2(x)

        with self.assertRaises(Unsupported):
            outer_f2(inp)

    def test_disable_recursive_flags(self):
        # [FUNCTION DOC] For both recursive flag values, disables a model forward and a whole
        # model, checks the recursive-flag query returns the flag, and that the models remain
        # compilable.
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        for recursive_flag in [True, False]:
            model = SimpleModel()
            other_model = SimpleModel()

            model.forward = torch._dynamo.disable(
                model.forward,
                recursive=recursive_flag,
            )
            self.assertEqual(
                torch._dynamo.is_dynamo_disable_recursive(model.forward),
                recursive_flag,
            )

            other_model = torch._dynamo.disable(other_model, recursive=recursive_flag)
            self.assertEqual(
                torch._dynamo.is_dynamo_disable_recursive(
                    other_model.forward
                    if isinstance(other_model, torch.nn.Module)
                    else other_model
                ),
                recursive_flag,
            )

            # check the model is compilable
            torch.compile(model, backend="eager")
            torch.compile(other_model, backend="eager")

    def test_disable_class_and_instance_method(self):
        # [FUNCTION DOC] Re-decorating an instance method already decorated at class level keeps
        # self bound and gives correct results, including double disable. Also checks that the
        # innermost function lookup unwraps unbound methods to the original but stops at bound
        # methods.
        # Test that decorating a method at class definition time and then
        # re-decorating the instance method works correctly. This tests the
        # fix in innermost_fn that stops unwrapping when hitting a bound method.
        from torch._dynamo.eval_frame import innermost_fn

        class Foo:
            def run(self, a, b, c):
                return self.work(a, b, c)

            @torch._dynamo.disable
            def work(self, a, b, c):
                return a + b - c

        foo = Foo()
        # Re-decorate the instance method
        foo.work = torch._dynamo.disable(foo.work)

        a = torch.randint(0, 10, (10,))
        b = torch.randint(0, 10, (10,))
        c = torch.randint(0, 10, (10,))

        # Should work without error - self should be correctly bound
        result = foo.run(a, b, c)
        self.assertEqual(result, a + b - c)

        # Also test nested disable on instance methods
        foo2 = Foo()
        foo2.work = torch._dynamo.disable(torch._dynamo.disable(foo2.work))
        result2 = foo2.run(a, b, c)
        self.assertEqual(result2, a + b - c)

        # Test innermost_fn shortcut behavior for unbound methods
        # disable(disable(Foo.method)) should unwrap to the original function
        class Bar:
            def method(self, x):
                return x + 1

        bar = Bar()
        bound_method = bar.method

        original_method = Bar.method
        disabled_once = torch._dynamo.disable(Bar.method)
        disabled_twice = torch._dynamo.disable(disabled_once)
        # innermost_fn should find the original unbound method
        self.assertIs(innermost_fn(disabled_twice), original_method)
        self.assertIs(innermost_fn(disabled_once), original_method)

        # Test innermost_fn shortcut behavior for bound methods
        # disable(disable(obj.method)) should stop at the bound method
        # innermost_fn should return the bound method itself, not unwrap it
        self.assertIs(innermost_fn(bound_method), bound_method)
        # Wrapping a bound method should also preserve the binding
        disabled_bound = torch._dynamo.disable(bound_method)
        self.assertIs(innermost_fn(disabled_bound), bound_method)

    def test_disable_functools_wraps(self):
        # [FUNCTION DOC] A wrapper created via functools wraps copies the original-callable
        # attribute; the innermost lookup must not follow it (checked through the wrapper id),
        # and disabling the outer wrapper must still give the wrapper result.
        # Test that functools.wraps copying _torchdynamo_orig_callable doesn't
        # cause innermost_fn to bypass the outer wrapper. This tests the fix
        # using _torchdynamo_wrapper_id to verify the attribute was set by our
        # decorator, not copied by functools.wraps.
        from torch._dynamo.eval_frame import innermost_fn

        @torch._dynamo.disable
        def inner_fn(x):
            return x + 1

        # Outer wrapper uses functools.wraps which copies _torchdynamo_orig_callable
        @functools.wraps(inner_fn)
        def outer_wrapper(x):
            return inner_fn(x) * 2

        # innermost_fn should NOT follow the copied _torchdynamo_orig_callable
        # because _torchdynamo_wrapper_id won't match
        self.assertIs(innermost_fn(outer_wrapper), outer_wrapper)

        # Applying disable to outer_wrapper should wrap outer_wrapper, not inner_fn
        disabled_outer = torch._dynamo.disable(outer_wrapper)

        x = torch.tensor([1.0, 2.0, 3.0])
        expected = outer_wrapper(x)  # (x+1)*2 = [4, 6, 8]
        actual = disabled_outer(x)
        self.assertEqual(expected, actual)

    def test_dynamo_disable_annotations(self):
        # [FUNCTION DOC] Exports a module with two disabled methods under preserved node
        # metadata and checks that nodes carry custom metadata naming each disabled method. With
        # forward disabled non-recursively, export raises RuntimeError.
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buffer", torch.rand(2, 2))

            @torch._dynamo.disable()
            def f1(self, x) -> torch.Tensor:
                return x + self.buffer + 1

            @torch._dynamo.disable()
            def f2(self, x) -> torch.Tensor:
                return x + self.buffer + 2

            def forward(self, x) -> torch.Tensor:
                return self.f1(x) + self.f2(x)

        model = SimpleModel()
        inp = torch.rand(2, 2)
        with torch.fx.traceback.preserve_node_meta():
            exported_model = torch.export.export(model, (inp,))
        graph = exported_model.graph_module.graph
        found_f1 = False
        found_f2 = False
        for node in graph.nodes:
            if "custom" in node.meta:
                if "_torchdynamo_disable_method" in node.meta["custom"]:
                    if node.meta["custom"]["_torchdynamo_disable_method"] == "f1":
                        found_f1 = True
                    elif node.meta["custom"]["_torchdynamo_disable_method"] == "f2":
                        found_f2 = True
        self.assertTrue(found_f1)
        self.assertTrue(found_f2)
        model.forward = torch._dynamo.disable(model.forward, recursive=False)
        with self.assertRaises(RuntimeError):
            exported_model = torch.export.export(model, (inp,))

    def test_allow_in_graph_inside_compile_gives_clear_error(self):
        # [FUNCTION DOC] Regression test: calling allow_in_graph inside a compiled region under
        # fullgraph raises Unsupported with a message mentioning allow_in_graph.
        # Regression test for https://github.com/pytorch/pytorch/issues/178511
        # Calling allow_in_graph inside a compiled region is not supported.
        # Verify the error message guides users to annotate before compilation.
        def forward(x):
            wrapped_fn = torch.compiler.allow_in_graph(my_custom_function)
            return wrapped_fn(x)

        compiled = torch.compile(forward, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "allow_in_graph",
        ):
            compiled(torch.randn(4))

    @parametrize("use_decorator", [False, True])
    @parametrize("fwd,bwd", [(True, False), (False, False)])
    def test_override_cudagraphs_annotation_graph_break(self, use_decorator, fwd, bwd):
        # [FUNCTION DOC] Parametrized by decorator versus context manager use and flag values. A
        # callee with a graph break must have both compiled segments carry the cudagraph
        # annotation (fwd and bwd values as requested) on the graph module metadata. Device
        # independent.
        # override_cudagraphs must propagate its annotation onto every compiled
        # subgraph even when the graph break happens inside a callee compiled as
        # a separate frame (it cannot be inlined). The override is established
        # either by a context manager in the caller or by decorating the callee;
        # both must seed every segment, for enable (fwd=True) and disable
        # (fwd=False) alike. Device-independent check on gm.meta (no CUDA).
        annotations = []

        def backend(gm, example_inputs):
            annotations.append(gm.meta.get("cudagraph_annotation"))
            return gm.forward

        if use_decorator:

            @torch._dynamo.override_cudagraphs(fwd=fwd, bwd=bwd)
            def callee(x):
                y = x + 1
                torch._dynamo.graph_break()
                return y * 2

            def model(x):
                return callee(x)

        else:

            def callee(x):
                y = x + 1
                torch._dynamo.graph_break()
                return y * 2

            def model(x):
                with torch._dynamo.override_cudagraphs(fwd=fwd, bwd=bwd):
                    return callee(x)

        torch._dynamo.reset()
        torch.compile(model, backend=backend)(torch.randn(4))
        # Both segments of the callee (before and after the break) carry it.
        self.assertEqual(len(annotations), 2)
        for ann in annotations:
            self.assertIsNotNone(ann)
            self.assertEqual((ann.fwd, ann.bwd), (fwd, bwd))

    def test_override_cudagraphs_annotation_nested_overrides_graph_break(self):
        # [FUNCTION DOC] Nested overrides: the inner wins while active and the outer is restored
        # on exit, so segments of the first callee have the inner values and the later callee
        # has the outer values; no segment lacks an annotation.
        # Nested overrides: the innermost (most recently entered) override wins
        # while active, and on exit the runtime stack restores the outer one so
        # a later separately-compiled callee sees it. Each callee graph-breaks,
        # so it is compiled as a separate frame seeded from the runtime
        # top-of-stack override. inner_callee and outer_callee are distinct code
        # objects (reusing one callee would be a sticky cache hit and would not
        # recompile under the restored override). Device-independent.
        annotations = []

        def backend(gm, example_inputs):
            annotations.append(gm.meta.get("cudagraph_annotation"))
            return gm.forward

        def inner_callee(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        def outer_callee(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        def model(x):
            with torch._dynamo.override_cudagraphs(fwd=True, bwd=False):
                with torch._dynamo.override_cudagraphs(fwd=False, bwd=True):
                    z = inner_callee(x)
                # Inner override popped here; outer (True, False) is restored.
                return outer_callee(z)

        torch._dynamo.reset()
        torch.compile(model, backend=backend)(torch.randn(4))
        for ann in annotations:
            self.assertIsNotNone(ann)
        vals = [(ann.fwd, ann.bwd) for ann in annotations]
        # Every segment carries one of the two overrides, never None or a
        # partial/leaked value (segment counts depend on nested_graph_breaks).
        self.assertTrue(all(v in {(False, True), (True, False)} for v in vals))
        # Inner override wins while active (inner_callee's segments).
        self.assertIn((False, True), vals)
        # After the inner `with` exits, __exit__ restores the outer override so
        # outer_callee sees it. Absent if __exit__ fails to pop/restore.
        self.assertIn((True, False), vals)

    def test_override_cudagraphs_annotation_not_guarded_first_compile_wins(self):
        # [FUNCTION DOC] The override state is not guarded: first compile inside an override
        # bakes the annotation into both segments, and a later call without override does not
        # recompile (backend not called again).
        # We intentionally do NOT guard on the dynamo global cudagraph override
        # state. The annotation is therefore sticky per code object: the first
        # compile wins, and toggling the override on a later call does NOT
        # recompile. This pins that intentional behavior; if a guard is added
        # later this test will flip (the second call would recompile).
        annotations = []

        def backend(gm, example_inputs):
            annotations.append(gm.meta.get("cudagraph_annotation"))
            return gm.forward

        @torch.compile(backend=backend)
        def callee(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        torch._dynamo.reset()
        # First reach callee from inside an override; its two segments compile
        # with the override annotation baked in.
        with torch._dynamo.override_cudagraphs(fwd=True, bwd=True):
            callee(torch.randn(4))
        self.assertEqual(len(annotations), 2)
        for ann in annotations:
            self.assertIsNotNone(ann)
            self.assertEqual((ann.fwd, ann.bwd), (True, True))

        annotations.clear()
        # Reach the same callee with no active override. Because the override
        # state is not guarded, callee is not recompiled: the backend is not
        # invoked again and the first-compile annotation sticks.
        callee(torch.randn(4))
        self.assertEqual(annotations, [])


instantiate_parametrized_tests(DecoratorTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
