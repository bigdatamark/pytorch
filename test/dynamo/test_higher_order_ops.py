# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for dynamo tracing of higher order ops (wrap, cond, map, vmap, grad, jacrev,
# checkpoint and similar). Defines small graph-inspection helpers (count_ops, find_first_node,
# op_count, assert_dict_matches_regex, default_args_generator), test classes HigherOrderOpTests,
# HigherOrderOpVmapGuardTests, FuncTorchHigherOrderOpTests, ActivationCheckpointingTests and
# TestHigherOrderOpsOpInfo, and several module level globals used as captured free variables. Many
# tests compare compiled and eager results and assert exact printed graph text. Import time side
# effects: computes device_type from the current accelerator, creates the global module, object,
# tensor, number and list, and registers device specific test variants at the bottom of the file.
import enum
import functools
import pprint
import re
import unittest
import warnings
from copy import deepcopy

import functorch.experimental.control_flow as control_flow
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import (
    check_dynamic_shape_capture,
    CompileCounter,
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    empty_line_normalizer,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._functorch.apis import _DYNAMO_WRAPPER_ATTRS
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_utils import (
    IS_LINUX,
    munge_exc,
    parametrize,
    skipIfXpu,
    TEST_WITH_SLOW,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch.testing._internal.triton_utils import requires_gpu_and_triton


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def count_ops(gm, args, freq, op):
    # [FUNCTION DOC] Compiler backend callback for use with aot_autograd or custom backends.
    # Counts graph nodes whose target equals the given op and raises an AssertionError if that
    # count differs from freq. Returns the graph module unchanged; the args parameter is unused.
    actual = [node.target for node in gm.graph.nodes].count(op)
    if actual != freq:
        raise AssertionError(f"expected={freq}, actual={actual}")
    return gm


class Obj:
    # [CLASS DOC] HELPER: Empty plain object class; a single instance is created at module level
    # as a global object for tests that capture or mutate global state.
    pass


class MyModule(nn.Module):
    # [CLASS DOC] HELPER: Tiny nn.Module with one scalar parameter named existing (initialized to
    # ones). Forward multiplies the input by that parameter. One instance is kept as a module
    # level global for capture tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the parent constructor and registers a scalar parameter
        # existing initialized to one.
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input multiplied elementwise by the existing
        # parameter.
        return self.existing * x


global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []


def find_first_node(gm, func):
    # [FUNCTION DOC] HELPER: Walks the nodes of the graph of gm in order and returns the first
    # node whose target is the given function object (identity comparison), or None when there is
    # no such node.
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    # [FUNCTION DOC] HELPER: Counts nodes in the graph of gm whose op kind string contains the
    # text call (call_function, call_method, call_module) and returns the count; placeholders,
    # getattr and output nodes are ignored.
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


# Checks that a dict matches a dict with "regex keys". That is,
# the keys are regex expressions.
def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    # [FUNCTION DOC] HELPER: Asserts that dct matches dct_with_regex_keys whose keys are regular
    # expressions. For each regex key, finds the key of dct matched from the start of the string;
    # raises AssertionError if one regex matches several keys or matches none. Builds a new
    # mapping from the matched real keys to the regex dict values and finally compares it with dct
    # using the test case assertEqual. The self argument is the test case.
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(
                        f"Single key regex mapped to multiple keys. Please improve your "
                        f"regex. Got: regex='{regex_key}' "
                        f"keys='{regex_key_to_actual_key[regex_key]}',"
                        f"'{key}'"
                    )
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    for regex_key in regex_keys:
        if regex_key not in regex_key_to_actual_key:
            raise AssertionError(
                f"Got regex '{regex_key}' but could not match any key in dict with "
                f"keys {dct.keys()}"
            )
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    self.assertEqual(dct, new_dct)


def default_args_generator(seed_value):
    # [FUNCTION DOC] HELPER: Generator yielding three variants of a seed argument structure. The
    # seed is flattened with pytree; for iteration i, tensors and floats are shifted by 0.1 times
    # i, ints by i, enums are kept, any other leaf type raises AssertionError. Each variant is
    # unflattened back to the original structure and yielded.
    flat_args, args_spec = pytree.tree_flatten(seed_value)
    for i in range(3):
        new_flat_arg = []
        for val in flat_args:
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i
            elif isinstance(val, int):
                new_val = val + 1 * i
            elif isinstance(val, float):
                new_val = val + 0.1 * i
            elif isinstance(val, enum.Enum):
                new_val = val
            else:
                raise AssertionError("unexpected arg type")

            new_flat_arg.append(new_val)
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        yield new_args


class HigherOrderOpTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main test case for the wrap higher order op and many other dynamo higher order
    # op features (cond, map, side effects, closure capture, pytree arguments, graph breaks). Most
    # tests compile a small function with fullgraph, compare with eager, and check graph shape or
    # exact printed graph text. Has two shared helper methods for the common check patterns.
    def _assert_wrap_fallback(self, func, args, setup=lambda: None):
        # [FUNCTION DOC] HELPER: Checks that a function falls back to graph breaks when wrapped
        # usage cannot be traced. Clears the graph break counters, runs func eagerly (after
        # calling setup), then compiles it without fullgraph using a recording backend and runs
        # again after setup. Asserts at least one graph break was recorded, that no wrap node
        # appears in any captured graph, and that compiled and eager results are equal.
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        setup()
        expected = func(*args)
        setup()
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        num_graph_breaks = len(counters["graph_break"].keys())
        self.assertGreater(num_graph_breaks, 0)

        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(node.target is wrap)

        self.assertEqual(result, expected)

    def _test_wrap_simple(
        self,
        func,
        args_generator,
        expected_num_wrap_args,
        expected_opcount=2,
        return_graph=False,
    ):
        # [FUNCTION DOC] HELPER: Shared checker for a function containing one wrap call. For each
        # argument tuple from args_generator, compiles func with fullgraph, compares to eager, and
        # asserts exactly one frame and one captured graph. On the first run only, also asserts
        # the operator count equals expected_opcount and that the first wrap node has
        # expected_num_wrap_args arguments (later runs may recompile dynamically). Returns the
        # normalized readable text of the first graph when return_graph is true, else None.
        # Given a `func` that has a single call to `wrap`,
        # we check that:
        # - there are no graph breaks
        # - eager vs torch.compile has the same result (correctness)
        # - other compilation metrics, e.g, # of ops in the dynamo captured graph,
        #   the wrap has the expected number of args, etc
        #
        # we have one or multiple runs through with each of the args from args_generator,
        # and we will check:
        # - correctness and no graph breaks for every run
        # - other compilation metrics only for the first run, since automatic_dynamic_shapes
        #   may compile another dynamic version graph for the later runs
        graph = None
        for i, args in enumerate(args_generator):
            backend = EagerAndRecordGraphs()
            cnt = CompileCounterWithBackend(backend)
            expected = func(*args)
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            # check correctness and no graph breaks
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            # check other compilation metrics
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                graph = backend.graphs[0]
                wrap_node = find_first_node(graph, wrap)

                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        # We always return/check the graph from the first run if return_graph = True
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        # [FUNCTION DOC] Verifies that a wrapped body appending to an outer list raises
        # Unsupported with a message about an unsafe side effect (HOP: Unsafe side effect) under
        # fullgraph compile.
        foo = []

        def inner(x):
            # [FUNCTION DOC] HELPER: Wrap body that appends its argument to the captured outer
            # list and returns a clone; the append is the side effect the compiler must reject.
            foo.append(x)
            return x.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled entry point (eager backend, fullgraph) that calls
            # wrap on the side effecting inner function.
            return wrap(inner, x)

        x = torch.randn(3)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "HOP: Unsafe side effect",
        ):
            f(x)

    def test_no_freevars(self):
        # [FUNCTION DOC] Tests wrap of a lambda with no free variables. Checks via the shared
        # helper that the wrap node has 2 args in static mode and 3 with dynamic shapes (extra
        # symbol).
        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda that applies sin to its input.
            return wrap(lambda x: torch.sin(x), x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_enum_arg(self):
        # [FUNCTION DOC] Tests passing an enum value as an extra argument to wrap. Body branches
        # on enum value; checks no graph break and the expected wrap argument count.
        class SomeEnum(enum.Enum):
            # [CLASS DOC] HELPER: Enum with two members A (0) and B (1) used as a constant
            # argument to wrap.
            A = 0
            B = 1

        def g(x, val):
            # [FUNCTION DOC] HELPER: Returns sin of x when val is SomeEnum.A, otherwise cos of x.
            if val == SomeEnum.A:
                return torch.sin(x)
            return torch.cos(x)

        def f(x, val):
            # [FUNCTION DOC] HELPER: Calls wrap on g with the tensor and the enum value.
            return wrap(g, x, val)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x, SomeEnum.A)), arg_count)

    def test_return_captured_var(self):
        # [FUNCTION DOC] Tests that a wrap body returning a captured free tensor (ignoring its
        # input) compiles; with the unused input not lifted, expects 1 op and 2 wrap args in
        # static mode (3 when dynamic).
        freevar = torch.randn(3)

        def test(x):
            # [FUNCTION DOC] HELPER: Wrap body that ignores its argument and returns the captured
            # tensor freevar.
            return freevar

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on test with the input tensor.
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.

        # when testing with dynamic shape, symbols are lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_vars(self):
        # [FUNCTION DOC] Tests a wrap body returning two captured free tensors, one of them twice;
        # expects 1 op and 3 wrap args static (4 dynamic).
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            # [FUNCTION DOC] HELPER: Wrap body returning a tuple of the captured tensors freevar1,
            # freevar2, freevar1.
            return freevar1, freevar2, freevar1

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on test with the input.
            return wrap(test, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 1)

    def test_return_captured_var_used_multiple_times(self):
        # [FUNCTION DOC] Tests a wrap body that adds a captured tensor to its input and also
        # returns the captured tensor; expects 2 ops and 3 wrap args static (4 dynamic).
        freevar = torch.randn(3)

        def test(x):
            # [FUNCTION DOC] HELPER: Wrap body returning the sum of x and the captured freevar
            # together with the freevar itself.
            y = x + freevar
            return y, freevar

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on test with the input.
            return wrap(test, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count, 2)

    def test_capture_untracked_global(self):
        # [FUNCTION DOC] Tests that a lambda capturing an untracked module level global tensor
        # works in wrap; expects 3 wrap args static (4 dynamic).
        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the global tensor global_var to
            # its input.
            return wrap(lambda x: x + global_var, x)

        x = torch.randn(3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_allow_python_side_effects_utility(self):
        # [FUNCTION DOC] Tests the utility that disables side effect safety checks for the current
        # subtracer when used with dynamo_bypassing_wrapper. A wrapped side effecting function
        # (incrementing a nonlocal counter) compiles; a nested HOP inside an unchecked region is
        # still checked and compiling it raises RuntimeError mentioning unsafe side effect.
        from torch._dynamo.utils import (
            _disable_side_effect_safety_checks_for_current_subtracer,
        )
        from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper

        def wrapper(fn):
            # [FUNCTION DOC] HELPER: Identity wrapper used as the wrapper callable of
            # dynamo_bypassing_wrapper; returns the function it receives.
            return fn

        count = 0

        def does_side_effect(x):
            # [FUNCTION DOC] HELPER: Increments the nonlocal counter and returns sin of the input.
            nonlocal count
            count += 1
            return x.sin()

        def does_side_effect_wrapped(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls does_side_effect through the utility that disables side
            # effect safety checks for the current subtracer, forwarding all positional and
            # keyword arguments.
            return _disable_side_effect_safety_checks_for_current_subtracer(
                does_side_effect, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager, fullgraph) function that runs the unchecked
            # side effect function through dynamo_bypassing_wrapper.
            return dynamo_bypassing_wrapper(wrapper, does_side_effect_wrapped, x)

        x = torch.tensor(1.0)
        fn(x)

        def inner_does_side_effect(x):
            # [FUNCTION DOC] HELPER: Increments the nonlocal counter and returns the input
            # unchanged.
            nonlocal count
            count += 1
            return x

        # Test that any nested HOPs are unaffected
        def outer(x):
            # [FUNCTION DOC] HELPER: Calls dynamo_bypassing_wrapper with the inner side effecting
            # function.
            return dynamo_bypassing_wrapper(wrapper, inner_does_side_effect, x)

        def outer_wrapped(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls outer through the utility disabling side effect checks
            # for the current subtracer.
            return _disable_side_effect_safety_checks_for_current_subtracer(
                outer, *args, **kwargs
            )

        @torch.compile(backend="eager", fullgraph=True)
        def fn_nested(x):
            # [FUNCTION DOC] HELPER: Compiled (eager, fullgraph) function running outer_wrapped
            # through dynamo_bypassing_wrapper; the nested HOP inside it is expected to still be
            # rejected.
            return dynamo_bypassing_wrapper(wrapper, outer_wrapped, x)

        x = torch.tensor(1.0)
        with self.assertRaisesRegex(RuntimeError, "HOP: Unsafe side effect"):
            fn_nested(x)

    def test_symint_input(self):
        # [FUNCTION DOC] Tests passing a symbolic size (taken from the input shape) as an argument
        # to wrap; body uses it in a view. Expects 2 wrap args static (3 dynamic) and 2 ops.
        def f(x):
            # [FUNCTION DOC] HELPER: Reads the first dimension size of x and calls wrap on a
            # lambda viewing x with that size.
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)

        x = torch.randn(3, 1)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
        )

    def test_symint_in_slice(self):
        # [FUNCTION DOC] Tests symbolic ints used inside slicing in a wrap lambda; expects wrap
        # arg count 2 static or 7 dynamic (symbols plus compound expressions) and op count 2
        # static or 4 dynamic.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes three sizes from the input shape and calls wrap on a
            # lambda slicing the input using them.
            i = x.size(0) - 2
            j = x.size(1) - 3
            k = x.size(2)
            return wrap(lambda x: x[:i, :j, k:], x)

        x = torch.randn(3, 4, 5)
        self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            # 3 basic symbols and 2 compound symbols
            ifdynstaticdefault(2, 7),
            # 2 more sym expression computation
            expected_opcount=ifdynstaticdefault(2, 4),
        )

    def test_wrap_pytree_args_nested(self):
        # [FUNCTION DOC] Tests wrap with a nested dict/tuple/list pytree argument. Generates three
        # argument variants and asserts the exact printed graph: three tensor inputs lifted and a
        # body doing sin, cos, sin and an add and a subtract.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with the outer dict d (a closure variable
            # defined after f); x, y, z parameters are unused.
            def fn(d):
                # [FUNCTION DOC] HELPER: Wrap body combining entries of a pytree dict: sin of x,
                # cos of first y, and sin of nested third element, adding then subtracting.
                return d["x"].sin() + d["y"][0].cos() - d["y"][1][2].sin()

            return wrap(fn, d)

        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {"x": x, "y": (y, [x, y, z])}

        def my_args_generator(t):
            # [FUNCTION DOC] HELPER: Generator yielding the base argument tuple and then two
            # variants with the first, then second, element shifted by 0.1.
            yield t
            yield t[0] + 0.1, t[1], t[2]
            yield t[0], t[1] + 0.1, t[2]

        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator((x, y, z)),
            4,
            return_graph=True,
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_d_x_: "f32[]", L_d_y_0_: "f32[]", L_d_y_1_2_: "f32[]"):
        l_d_x_ = L_d_x_
        l_d_y_0_ = L_d_y_0_
        l_d_y_1_2_ = L_d_y_1_2_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None
        getitem: "f32[]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_d_x_: "f32[]", l_d_y_0_: "f32[]", l_d_y_1_2_: "f32[]"):
            sin: "f32[]" = l_d_x_.sin();  l_d_x_ = None
            cos: "f32[]" = l_d_y_0_.cos();  l_d_y_0_ = None
            add: "f32[]" = sin + cos;  sin = cos = None
            sin_1: "f32[]" = l_d_y_1_2_.sin();  l_d_y_1_2_ = None
            sub: "f32[]" = add - sin_1;  add = sin_1 = None
            return (sub,)
""",
        )

    def test_wrap_pytree_args_with_symint_constant(self):
        # [FUNCTION DOC] Tests wrap with a tuple containing a tensor, a float constant and a
        # symbolic size. Asserts exact printed graph text, which differs between static (size
        # baked in as a constant) and dynamic (symbol lifted as input) modes.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda taking a tuple, viewing its first
            # element with the size and adding the constant.
            i = x.size(0)
            return wrap(lambda t: t[0].view(t[2]) + t[1], (x, y, i))

        x = torch.randn(3, 1)
        y = 0.5
        actual_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            ifdynstaticdefault(2, 3),
            expected_opcount=2,
            return_graph=True,
        )
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 1]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 1]"):
            view: "f32[3]" = l_x_.view(3);  l_x_ = None
            add: "f32[3]" = view + 0.5;  view = None
            return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                actual_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, 1]", s77: "Sym(s77)"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_);  wrap_body_0 = s77 = l_x_ = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77, 1]"):
            view: "f32[s77]" = l_x_.view(s77);  l_x_ = s77 = None
            add: "f32[s77]" = view + 0.5;  view = None
            return (add,)
""",
            )

    def test_wrap_pytree_kwargs(self):
        # [FUNCTION DOC] Tests wrap called with keyword arguments including a nested tuple of
        # tensors; the body is keyword only. Expects 3 wrap args static (4 dynamic).
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Calls wrap on fn passing x, y, z as keyword arguments.
            def fn(*, x, y, z):
                # [FUNCTION DOC] HELPER: Keyword only wrap body returning x times two plus y plus
                # the first element of z.
                z1, _ = z
                return (x * 2) + y + z1

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        def my_args_generator(t):
            # [FUNCTION DOC] HELPER: Generator yielding the base tuple and two shifted variants
            # (shifts 0.1 and 0.2), each with z being a tuple of the first two entries.
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), arg_count)

    def test_wrap_pytree_args_not_const_symint_tensor(self):
        # [FUNCTION DOC] Tests wrap with a pytree holding a tensor and an opaque user object that
        # wraps a tensor; expects 3 wrap args and the default op count.
        class MyClass:
            # [CLASS DOC] HELPER: Simple container class holding one value in the attribute val.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the attribute val.
                self.val = x

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda multiplying sin of the first tuple
            # element by cos of the val attribute of the second.
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))

        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        self._test_wrap_simple(f, [(x, y)], 3)

    def test_capture_constants(self):
        # [FUNCTION DOC] Tests wrap with constant arguments (float and None). Compiles with
        # fullgraph, checks equality with eager, then changes the float constant and checks it
        # still matches (forcing a recompile).
        x = torch.randn(3, 3)

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Returns x plus y when z is truthy, else x times y.
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with the three arguments.
            return wrap(fn, x, y, z)

        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

        # Ensure that we recompile here
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        # [FUNCTION DOC] Tests nested wraps whose inner lambda captures a global tensor. Asserts
        # correct result, one frame, 2 ops, and that both the outer and inner wrap nodes exist
        # with the body graph containing 2 ops.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function with two nested wrap calls; the innermost
            # lambda adds the global tensor.
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)

        x = torch.randn(3)
        result = f(x)

        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        # [FUNCTION DOC] Tests capturing a tensor from an enclosing function scope (nonlocal)
        # inside wrap via the shared helper, expecting 3 wrap args static (4 dynamic); also runs
        # the function eagerly.
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Defines g and runs the wrap checker on it, then returns g
            # applied to x.
            def g(x):
                # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the captured outer y to its
                # input.
                return wrap(lambda x: x + y, x)

            # when testing with dynamic shape, a symbol is lifted as input
            arg_count = ifdynstaticdefault(3, 4)
            self._test_wrap_simple(g, default_args_generator((x,)), arg_count)
            return g(x)

        f(x, y)

    def test_capture_tracked(self):
        # [FUNCTION DOC] Tests that a tracked tensor argument captured by a wrap lambda is lifted;
        # expects 3 wrap args static (4 dynamic).
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the captured y to x.
            return wrap(lambda x: x + y, x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_capture_tracked_nested(self):
        # [FUNCTION DOC] Tests two nested wrap calls where the inner lambda captures a tracked
        # tensor; expects 3 wrap args static (4 dynamic).
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda that itself calls wrap on a lambda
            # adding y.
            return wrap(lambda x: wrap(lambda x: x + y, x), x)

        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_inlined_functions(self):
        # [FUNCTION DOC] Tests that a wrap lambda calling a helper function g (inlined) with a
        # captured tensor traces correctly; 3 wrap args static (4 dynamic).
        def g(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return x + y

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda that calls g on x and the captured y.
            return wrap(lambda x: g(x, y), x)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_same_freevar_twice(self):
        # [FUNCTION DOC] Tests a wrap body using the same captured tensor twice (sin and cos),
        # returning both; expects 3 ops and 2 wrap args static (3 dynamic).
        free = torch.randn(3)

        def g(x):
            # [FUNCTION DOC] HELPER: Wrap body returning sin and cos of the captured tensor free,
            # ignoring x.
            y = free.sin()
            z = free.cos()
            return y, z

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g with the input.
            return wrap(g, x)

        x = torch.randn(3)

        # Since, `x` is unused, we don't lift it to
        # be the input.
        # when testing with dynamic shape, a symbol is lifted as input
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count, 3)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/177003"
    )
    @torch._dynamo.config.patch(
        capture_scalar_outputs=True,
    )
    def test_unbacked_symbol_closure(self):
        # [FUNCTION DOC] Tests closing over an unbacked symbolic int (from item on a sum) in
        # doubly nested wraps, with scalar output capture enabled. Skipped on Linux or slow runs
        # (see linked issue). Asserts exact printed graph for dynamic and static modes with the
        # item symbol lifted through both bodies.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes the item value of the sum of x then returns wrap of
            # g on x.
            c = x.sum().item()

            def g(x):
                # [FUNCTION DOC] HELPER: Defines inner k and returns wrap on k.
                def k(x):
                    # [FUNCTION DOC] HELPER: Returns the input plus the captured item value c.
                    return x + c

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        out_graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 4, return_graph=True
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77]", s77: "Sym(s77)"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, item);  wrap_body_1 = s77 = l_x_ = item = None
        getitem: "f32[s77]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, item);  wrap_body_0 = s77 = l_x_ = item = None
            getitem: "f32[s77]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", item: "Sym(zuf0)"):
                add: "f32[s77]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        item: "Sym(zuf0)" = sum_1.item();  sum_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, item);  wrap_body_1 = l_x_ = item = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, item);  wrap_body_0 = l_x_ = item = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", item: "Sym(zuf0)"):
                add: "f32[3]" = l_x_ + item;  l_x_ = item = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_with_unbacked_shape_closure(self):
        # [FUNCTION DOC] Tests closing over a tensor with unbacked shape (from nonzero) in doubly
        # nested wraps, with dynamic output shape capture enabled. Expects 4 wrap args static (5
        # dynamic) and 9 ops static (7 dynamic), then asserts exact printed graph text for the
        # dynamic-capture or static case.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes nonzero of x then returns wrap of g on x.
            c = x.nonzero()

            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body that returns wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body returning sin of x and sin of the
                    # captured nonzero result c.
                    return x.sin(), c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(9, 7)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77]", s77: "Sym(s77)"):
        l_x_ = L_x_

        nonzero: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, sym_size_int, nonzero);  wrap_body_1 = s77 = l_x_ = sym_size_int = nonzero = None
        getitem: "f32[s77]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, l_x_, u0, c);  wrap_body_0 = s77 = l_x_ = u0 = c = None
            getitem: "f32[s77]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", l_x_: "f32[s77]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[s77]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )
        else:
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        nonzero: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int, nonzero);  wrap_body_1 = l_x_ = sym_size_int = nonzero = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, u0, c);  wrap_body_0 = l_x_ = u0 = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", u0: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (sin, sin_1)
""",
            )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/184831"
    )
    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
        capture_scalar_outputs=True,
    )
    def test_tensor_to_list_closure(self):
        # [FUNCTION DOC] Tests closing over a Python list produced by tolist inside nested wraps
        # (scalar and dynamic shape output capture on). Skipped on Linux or slow runs (see linked
        # issue). Because tolist specializes on shapes, static and dynamic modes share the same
        # expected graph; asserts 3 wrap args, 4 ops and exact graph text.
        def f(x):
            # [FUNCTION DOC] HELPER: Converts x to a list and returns wrap of g on x.
            li = x.tolist()

            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body returning wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body adding the first list element to x.
                    return li[0] + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.tensor([1, 2, 3], dtype=torch.int16)
        arg_count = ifdynstaticdefault(3, 3)
        out_graph = self._test_wrap_simple(f, ((x,),), arg_count, 4, return_graph=True)

        # tolist will specialize on input shapes, so dynamic and static tests
        # have the same graph
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "i16[3]"):
        l_x_ = L_x_

        getitem = l_x_[0]
        item: "Sym(u0)" = getitem.item();  getitem = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, item, l_x_);  wrap_body_1 = item = l_x_ = None
        getitem_1: "i16[3]" = wrap[0];  wrap = None
        return (getitem_1,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, item, l_x_);  wrap_body_0 = item = l_x_ = None
            getitem: "i16[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, item: "Sym(u0)", l_x_: "i16[3]"):
                add: "i16[3]" = item + l_x_;  item = l_x_ = None
                return (add,)
""",
        )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/184682"
    )
    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_tensor_and_unbacked_symbol_closure(self):
        # [FUNCTION DOC] Tests closing over both a nonzero tensor c and its size symbol sz (the
        # same unbacked symbol u0 appears in both and is de-duplicated when lifted) in nested
        # wraps. Skipped on Linux or slow runs. Checks the wrap arg count, op count, and, only in
        # non-dynamic capture mode, the exact graph text.
        def f(x):
            # [FUNCTION DOC] HELPER: Computes nonzero of x and its first dimension size, then
            # returns wrap of g on x.
            c = x.nonzero()
            sz = c.size(0)

            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body returning wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body returning sin of x plus the captured
                    # size together with sin of the captured nonzero tensor.
                    return x.sin() + sz, c.sin()

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(4, 5)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0
        expected_op_count = ifdynstaticdefault(9, 7)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x,)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        # Note that u0 is accessed from sz and the shape of c
        # We cached via the symbol u0 and de-duplicate them.
        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        nonzero: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, l_x_, sym_size_int, nonzero);  wrap_body_1 = l_x_ = sym_size_int = nonzero = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_1(torch.nn.Module):
        def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_, size, c);  wrap_body_0 = l_x_ = size = c = None
            getitem: "f32[3]" = wrap[0]
            getitem_1: "f32[u0, 1]" = wrap[1];  wrap = None
            return (getitem, getitem_1)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[3]", size: "Sym(u0)", c: "i64[u0, 1]"):
                sin: "f32[3]" = l_x_.sin();  l_x_ = None
                add: "f32[3]" = sin + size;  sin = size = None
                sin_1: "f32[u0, 1]" = c.sin();  c = None
                return (add, sin_1)
""",
            )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/183671"
    )
    @torch._dynamo.config.patch(
        capture_dynamic_output_shape_ops=True,
    )
    def test_concat_unbacked_shape_tensor(self):
        # [FUNCTION DOC] Tests nested wraps closing over a tensor made by concatenating two
        # nonzero results (two unbacked shape symbols). Skipped on Linux or slow runs. Expects 5
        # wrap args static (6 dynamic) and 15 ops static (11 dynamic); asserts graph text only for
        # non-dynamic capture.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Builds the concatenation of the nonzero results of x and y,
            # then returns wrap of g on x.
            c = x.nonzero()
            d = y.nonzero()
            cat = torch.cat((c, d))

            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body returning wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body adding the sum of the captured
                    # concatenated tensor to x.
                    return cat.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(3)
        y = torch.randn(3)
        arg_count = ifdynstaticdefault(5, 6)
        # when compiled with dynamic, we don't have upper bound runtime assertions for u0 and u1
        expected_op_count = ifdynstaticdefault(15, 11)
        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            arg_count,
            expected_op_count,
            return_graph=True,
        )

        if not check_dynamic_shape_capture():
            self.assertExpectedInline(
                out_graph,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]", L_y_: "f32[3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        nonzero: "i64[u0, 1]" = l_x_.nonzero()
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        le: "Sym(u0 <= 3)" = sym_size_int <= 3
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 3 on node 'le'");  le = _assert_scalar_default_1 = None

        nonzero_1: "i64[u1, 1]" = l_y_.nonzero();  l_y_ = None
        sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(nonzero_1, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_2 = None
        le_1: "Sym(u1 <= 3)" = sym_size_int_1 <= 3
        _assert_scalar_default_3 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u1 <= 3 on node 'le_1'");  le_1 = _assert_scalar_default_3 = None

        cat: "i64[u0 + u1, 1]" = torch.cat((nonzero, nonzero_1));  nonzero = nonzero_1 = None

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int, sym_size_int_1, cat, l_x_);  wrap_body_1 = sym_size_int = sym_size_int_1 = cat = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, u1, cat, l_x_);  wrap_body_0 = u0 = u1 = cat = l_x_ = None
            getitem: "f32[3]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", u1: "Sym(u1)", cat: "i64[u0 + u1, 1]", l_x_: "f32[3]"):
                sum_1: "i64[]" = cat.sum();  cat = None
                add: "f32[3]" = sum_1 + l_x_;  sum_1 = l_x_ = None
                return (add,)
""",
            )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
    )
    def test_lift_tensors_with_shared_symbols(self):
        # [FUNCTION DOC] Tests (with dynamic shapes forced on) that tensors sharing symbolic sizes
        # are lifted into nested wraps with the shared symbols passed once; expects 6 wrap args, 2
        # ops and exact graph text.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns wrap of g on x where the captured y is used by the
            # innermost body.
            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body returning wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body returning the matrix product of x and
                    # the captured y.
                    return x @ y

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        out_graph = self._test_wrap_simple(
            f,
            default_args_generator((x, y)),
            6,
            2,
            return_graph=True,
        )
        self.assertExpectedInline(
            out_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, s27]", s77: "Sym(s77)", s27: "Sym(s27)", L_y_: "f32[s27, s94]", s94: "Sym(s94)"):
        l_x_ = L_x_
        l_y_ = L_y_

        wrap_body_1 = self.wrap_body_1
        wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, s27, l_x_, s94, l_y_);  wrap_body_1 = s77 = s27 = l_x_ = s94 = l_y_ = None
        getitem: "f32[s77, s94]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
            wrap_body_0 = self.wrap_body_0
            wrap = torch.ops.higher_order.wrap(wrap_body_0, s77, s27, l_x_, s94, l_y_);  wrap_body_0 = s77 = s27 = l_x_ = s94 = l_y_ = None
            getitem: "f32[s77, s94]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", l_x_: "f32[s77, s27]", s94: "Sym(s94)", l_y_: "f32[s27, s94]"):
                matmul: "f32[s77, s94]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
                return (matmul,)
""",
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        dynamic_shapes=True,
        capture_dynamic_output_shape_ops=True,
    )
    def test_lift_tensors_with_compound_expressions(self):
        # [FUNCTION DOC] Tests (dynamic shapes forced on, nonzero capture enabled) lifting tensors
        # whose shapes are compound symbolic expressions (view then concat with a nonzero result)
        # into nested wraps. Runs the function once eagerly; unless dynamic capture mode is on,
        # checks 6 wrap args, 9 ops and exact graph text.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Reshapes x to two columns, concatenates it with nonzero of y,
            # then returns wrap of g on x.
            x = x.view(-1, 2)
            c = y.nonzero()
            d = torch.concat((x, c))

            def g(x):
                # [FUNCTION DOC] HELPER: Middle wrap body returning wrap of k on x.
                def k(x):
                    # [FUNCTION DOC] HELPER: Innermost body adding the sum of the captured
                    # concatenation to x.
                    return d.sum() + x

                return wrap(k, x)

            return wrap(g, x)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)

        f(x, y)

        if not check_dynamic_shape_capture():
            out_graph = self._test_wrap_simple(
                f,
                default_args_generator((x, y)),
                6,
                9,
                return_graph=True,
            )
            self.assertExpectedInline(
                out_graph,
                """\
    class GraphModule(torch.nn.Module):
        def forward(self, s0: "Sym(s0)", s1: "Sym(s1)", L_x_: "f32[s0, s1]", s2: "Sym(s2)", L_y_: "f32[s1, s2]"):
            l_x_ = L_x_
            l_y_ = L_y_

            x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = l_x_.view(-1, 2);  l_x_ = None

            c: "i64[u0, 2]" = l_y_.nonzero();  l_y_ = None

            sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
            _check_is_size = torch._check_is_size(sym_size_int_1);  _check_is_size = None

            ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

            d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = torch.concat((x, c));  c = None

            wrap_body_1 = self.wrap_body_1
            wrap = torch.ops.higher_order.wrap(wrap_body_1, sym_size_int_1, s1, s0, d, x);  wrap_body_1 = sym_size_int_1 = s1 = s0 = d = x = None
            getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
            return (getitem,)

        class wrap_body_1(torch.nn.Module):
            def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                wrap_body_0 = self.wrap_body_0
                wrap = torch.ops.higher_order.wrap(wrap_body_0, u0, s1, s0, d, x);  wrap_body_0 = u0 = s1 = s0 = d = x = None
                getitem: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = wrap[0];  wrap = None
                return (getitem,)

            class wrap_body_0(torch.nn.Module):
                def forward(self, u0: "Sym(u0)", s1: "Sym(s1)", s0: "Sym(s0)", d: "f32[u0 + ((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]", x: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]"):
                    sum_1: "f32[]" = d.sum();  d = None
                    add: "f32[((s0*s1)//2), ((s0*s1)//(((s0*s1)//2)))]" = sum_1 + x;  sum_1 = x = None
                    return (add,)
    """,
            )

    def test_register_subclass(self):
        # [FUNCTION DOC] Tests that the cond operator rejects a tensor subclass (TwoTensor) until
        # an implementation is registered. Temporarily removes any existing registration and
        # clears the dispatch cache (restored by a cleanup), asserts NotImplementedError about no
        # rule for the subclass, registers a python implementation that applies cond to both inner
        # tensors, and checks result values and that the implementation ran once.
        from torch._higher_order_ops.cond import cond_op
        from torch.testing._internal.two_tensor import TwoTensor

        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.randn(3)
        t = TwoTensor(a, b)

        prev_impl = cond_op.python_key_table.pop(TwoTensor, None)
        cond_op._dispatch_cache.clear()

        def restore_twotensor_impl():
            # [FUNCTION DOC] HELPER: Cleanup that removes the TwoTensor entry from the cond python
            # key table, reinstalls the previously saved implementation if there was one, and
            # clears the cond dispatch cache.
            cond_op.python_key_table.pop(TwoTensor, None)
            if prev_impl is not None:
                cond_op.python_key_table[TwoTensor] = prev_impl
            cond_op._dispatch_cache.clear()

        self.addCleanup(restore_twotensor_impl)

        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HOP cond and subclass .*TwoTensor'>",
        ):
            res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))

        called = 0

        # Using cond.py_impl
        @cond_op.py_impl(TwoTensor)
        def _(pred, true_fn, false_fn, operands):
            # [FUNCTION DOC] HELPER: Registered cond implementation for TwoTensor. Increments the
            # nonlocal call counter, asserts a single operand, runs cond separately on the two
            # component tensors and wraps the results in a new TwoTensor.
            nonlocal called
            called += 1
            assert len(operands) == 1  # noqa: S101
            a = cond_op(pred, true_fn, false_fn, (operands[0].a,))
            b = cond_op(pred, true_fn, false_fn, (operands[0].b,))
            return TwoTensor(a, b)

        res = cond_op(a.sum() > 0, torch.sin, torch.cos, (t,))
        self.assertEqual(res.a, torch.sin(a))
        self.assertEqual(res.b, torch.sin(b))
        self.assertEqual(called, 1)

    def test_register_mode(self):
        # [FUNCTION DOC] Tests that cond rejects a torch dispatch mode until an implementation for
        # the mode class is registered. First asserts NotImplementedError under the mode, then
        # registers a python implementation for the mode and checks cond result equals sin and
        # that the dispatch mode counter variables are used.
        from torch._higher_order_ops.cond import cond_op

        torch_dispatch_called = 0

        class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
            # [CLASS DOC] HELPER: Torch dispatch mode that counts intercepted calls in a nonlocal
            # counter and forwards them unchanged.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Increments the nonlocal dispatch counter and calls the
                # function with the given args and kwargs, returning its result.
                nonlocal torch_dispatch_called
                torch_dispatch_called += 1
                return func(*args, **kwargs)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with self.assertRaisesRegex(
            NotImplementedError,
            "no rule registered for HigherOrderOperator cond and mode .*MyMode",
        ):
            with MyMode():
                res = cond_op(pred, torch.sin, torch.cos, (a,))

        py_impl_called = 0

        # Using cond.py_impl
        @cond_op.py_impl(MyMode)
        def _(mode, pred, true_fn, false_fn, operands):
            # [FUNCTION DOC] HELPER: Registered cond implementation for the mode. Increments the
            # nonlocal counter and calls cond again with the same predicate, branches and
            # operands.
            nonlocal py_impl_called
            py_impl_called += 1
            return cond_op(pred, true_fn, false_fn, operands)

        a = torch.tensor([1.0, 0.1, 1.0])
        pred = a.sum() > 0
        with MyMode():
            res = cond_op(pred, torch.sin, torch.cos, (a,))
        self.assertEqual(res, a.sin())

    def test_capture_value_created_in_subgraph(self):
        # [FUNCTION DOC] Tests that a value created inside an outer wrap body (sum of x and y) and
        # used by an inner nested wrap is lifted correctly. Asserts result, one frame, 2 ops, and
        # walks the nested body graphs checking op counts of 3 and 2.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            # [FUNCTION DOC] HELPER: Computes z as the sum of x and y and returns wrap of a lambda
            # that wraps another lambda adding z.
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function calling wrap on inner.
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)

        # No changes to args of outer wrap
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # z was lifted to arg of inner wrap
        body_function = getattr(gm, wrap_node.args[0].name)
        # addition + wrap + getitem
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

        # Innermost body function: z was also lifted to arg
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_side_effect_set_new_attr_global_obj(self):
        # [FUNCTION DOC] Tests that setting a new attribute on a global object inside a wrap body
        # causes a graph break fallback (checked by the fallback helper).
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook that rebinds the global object to a fresh Obj.
            global global_obj
            global_obj = Obj()

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the nested h function.
            def h(x):
                # [FUNCTION DOC] HELPER: Calls wrap on g and adds the attribute foo set by g to
                # the result.
                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body that sets attribute foo on the global
                    # object to x plus one and returns a clone.
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_obj(self):
        # [FUNCTION DOC] Tests that overwriting an existing attribute on a global object inside
        # wrap triggers a graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook that creates a fresh global object with attribute
            # foo set to a parameter holding 4.0.
            global global_obj
            global_obj = Obj()
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Calls wrap on g and adds the global foo.
                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body assigning x plus one to the global object
                    # attribute foo and returning a clone.
                    global_obj.foo = x + 1
                    return x.clone()

                y = wrap(g, x)
                return y + global_obj.foo

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_obj(self):
        # [FUNCTION DOC] Tests that deleting an existing attribute of a global object inside wrap
        # triggers a graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook creating a fresh global object with attribute foo
            # set to a tensor.
            global global_obj
            global_obj = Obj()
            global_obj.foo = torch.tensor(4.0)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Returns wrap of g on x.
                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body that deletes foo from the global object and
                    # returns a clone.
                    del global_obj.foo
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_new_attr_global_module(self):
        # [FUNCTION DOC] Tests that setting a new attribute on a global module inside wrap causes
        # a graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global module to a fresh MyModule.
            global global_module
            global_module = MyModule()

        def h(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the new attribute foo of the global
            # module.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body that attaches a parameter foo (x plus one) to
                # the global module and returns a clone.
                global_module.foo = nn.Parameter(x + 1)
                return x.clone()

            y = wrap(g, x)
            return y + global_module.foo

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_module(self):
        # [FUNCTION DOC] Tests that replacing an existing parameter of a global module inside wrap
        # causes a graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global module to a fresh MyModule.
            global global_module
            global_module = MyModule()

        def h(x):
            # [FUNCTION DOC] HELPER: Returns wrap of g on x.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body that replaces the parameter existing of the
                # global module and returns the module output for x.
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                return global_module(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_module(self):
        # [FUNCTION DOC] Tests that deleting a parameter of a global module inside wrap causes a
        # graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global module to a fresh MyModule.
            global global_module
            global_module = MyModule()

        def h(x):
            # [FUNCTION DOC] HELPER: Returns wrap of g on x.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body that deletes the parameter existing from the
                # global module and returns a clone.
                del global_module.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_mutate_global_num(self):
        # [FUNCTION DOC] Tests that rebinding a global number inside a wrap body causes a graph
        # break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook resetting the global number to 3.14.
            global global_num
            global_num = 3.14

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the global number to the result.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body that increments the global number by rebinding
                # it and returns x plus it.
                global global_num
                global_num = global_num + 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_num_builtin(self):
        # [FUNCTION DOC] Tests that in-place style augmented assignment to a global number inside
        # wrap causes a graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook resetting the global number to 3.14.
            global global_num
            global_num = 3.14

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the global number.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body using augmented assignment to add one to the
                # global number and returning x plus it.
                global global_num
                global_num += 1
                return x + global_num

            y = wrap(g, x)
            return y + global_num

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor(self):
        # [FUNCTION DOC] Tests that rebinding a global tensor inside wrap causes a graph break
        # fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global tensor to ones of length
            # three.
            global global_var
            global_var = torch.ones(3)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the global tensor.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body that rebinds the global tensor to itself plus
                # one and returns x plus it.
                global global_var
                global_var = global_var + 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor_builtin(self):
        # [FUNCTION DOC] Tests that augmented assignment on a global tensor inside wrap causes a
        # graph break fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global tensor to ones of length
            # three.
            global global_var
            global_var = torch.ones(3)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the global tensor.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body using augmented assignment of one on the global
                # tensor and returning x plus it.
                global global_var
                global_var += 1
                return x + global_var

            y = wrap(g, x)
            return y + global_var

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_list(self):
        # [FUNCTION DOC] Tests that appending to a global list inside wrap causes a graph break
        # fallback.
        def setup():
            # [FUNCTION DOC] HELPER: Setup hook rebinding the global list to a new empty list.
            global global_list
            global_list = []

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on g and adds the last element of the global list.
            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body appending x plus one to the global list and
                # returning the last element.
                val = x + 1
                global_list.append(val)
                return global_list[-1]

            y = wrap(g, x)
            z = y + global_list[-1]
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_nonlocal_num(self):
        # [FUNCTION DOC] Tests that rebinding a nonlocal number inside wrap causes a graph break
        # fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Defines a local counter and runs wrap on g, then adds the
                # counter.
                val = 1

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body rebinding the nonlocal counter to counter
                    # plus one and returning x plus it.
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_obj(self):
        # [FUNCTION DOC] Tests that setting a new attribute on a nonlocal object inside wrap
        # causes a graph break fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates an object, runs wrap on g and adds the attribute
                # val that g set.
                obj = Obj()

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body setting attribute val on the captured
                    # object to the dimension count of x and returning a clone.
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        # [FUNCTION DOC] Tests that overwriting an existing attribute on a nonlocal object inside
        # wrap causes a graph break fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates an object with val equal to three, runs wrap on g
                # and adds val.
                obj = Obj()
                obj.val = 3

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body overwriting val of the captured object with
                    # the dimension count of x and returning a clone.
                    obj.val = x.dim()
                    return x.clone()

                y = wrap(g, x)
                z = y + obj.val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_del_existing_attr_nonlocal_obj(self):
        # [FUNCTION DOC] Tests that deleting an attribute of a nonlocal object inside wrap causes
        # a graph break fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates an object with attribute val and returns wrap of
                # g on x.
                obj = Obj()
                obj.val = 3

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body deleting val from the captured object and
                    # returning a clone.
                    del obj.val
                    return x.clone()

                y = wrap(g, x)
                return y

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_module(self):
        # [FUNCTION DOC] Tests that setting a new attribute on a nonlocal module inside wrap
        # causes a graph break fallback.
        def h(x):
            # [FUNCTION DOC] HELPER: Creates a module, runs wrap on g and adds the attribute val
            # set by g.
            obj = MyModule()

            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body setting val on the captured module to the
                # dimension count of x and returning a clone.
                obj.val = x.dim()
                return x.clone()

            y = wrap(g, x)
            z = y + obj.val
            return z

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_set_existing_attr_nonlocal_module(self):
        # [FUNCTION DOC] Tests that replacing an existing parameter on a nonlocal module inside
        # wrap causes a graph break fallback.
        def h(x):
            # [FUNCTION DOC] HELPER: Creates a module and returns wrap of g on x.
            obj = MyModule()

            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body replacing the module parameter existing with a
                # new one and returning the module output on x.
                obj.existing = nn.Parameter(torch.tensor(3.14))
                return obj(x)

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_del_existing_attr_nonlocal_module(self):
        # [FUNCTION DOC] Tests that deleting a parameter from a nonlocal module inside wrap causes
        # a graph break fallback.
        def h(x):
            # [FUNCTION DOC] HELPER: Creates a module and returns wrap of g on x.
            obj = MyModule()

            def g(x):
                # [FUNCTION DOC] HELPER: Wrap body deleting the parameter existing from the
                # captured module and returning a clone.
                del obj.existing
                return x.clone()

            y = wrap(g, x)
            return y

        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_mutate_nonlocal_tensor(self):
        # [FUNCTION DOC] Tests that rebinding a nonlocal tensor inside wrap causes a graph break
        # fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates a local scalar tensor, runs wrap on g and adds
                # the tensor afterwards.
                val = torch.tensor(1.0)

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body rebinding the nonlocal tensor to itself
                    # plus one and returning x plus it.
                    nonlocal val
                    val = val + 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_num_builtin(self):
        # [FUNCTION DOC] Tests that augmented assignment on a nonlocal number inside wrap causes a
        # graph break fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates a local number, runs wrap on g and adds the
                # number afterwards.
                val = 1

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body adding one to the nonlocal number with
                    # augmented assignment and returning x plus it.
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_tensor_builtin(self):
        # [FUNCTION DOC] Tests that augmented assignment on a nonlocal tensor inside wrap causes a
        # graph break fallback.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates a local scalar tensor, runs wrap on g and adds it
                # afterwards.
                val = torch.tensor(1.0)

                def g(x):
                    # [FUNCTION DOC] HELPER: Wrap body adding one to the nonlocal tensor with
                    # augmented assignment and returning x plus it.
                    nonlocal val
                    val += 1
                    return x + val

                y = wrap(g, x)
                z = y + val
                return z

            return h(x)

        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_nonlocal_list_append_graph_break(self):
        # [FUNCTION DOC] Tests that a wrap body appending to a list owned by the enclosing
        # function graph breaks; result must match eager.
        def g(x):
            # [FUNCTION DOC] HELPER: Creates an empty list, runs wrap on f and returns the first
            # list element.
            y = []

            def f(k):
                # [FUNCTION DOC] HELPER: Wrap body appending k plus one to the outer list and
                # returning k.
                m = k + 1
                y.append(m)
                return k

            wrap(f, x)
            return y[0]

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_nested_nonlocal_list_append_graph_break(self):
        # [FUNCTION DOC] Tests that appending to a list defined two function levels out from the
        # wrap body causes a graph break fallback.
        def g(x):
            # [FUNCTION DOC] HELPER: Calls h.
            def h(x):
                # [FUNCTION DOC] HELPER: Creates an empty list, runs wrap on f and returns the
                # first list element.
                y = []

                def f(k):
                    # [FUNCTION DOC] HELPER: Wrap body appending k plus one to the outer list and
                    # returning k.
                    m = k + 1
                    y.append(m)
                    return k

                wrap(f, x)
                return y[0]

            return h(x)

        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_local_list_append_no_graph_break(self):
        # [FUNCTION DOC] Tests that a list created and appended to entirely inside the wrap body
        # does not graph break; uses the shared helper expecting 2 wrap args static (3 dynamic).
        def g(x):
            # [FUNCTION DOC] HELPER: Returns wrap of f on x.
            def f(k):
                # [FUNCTION DOC] HELPER: Wrap body creating a local list, appending k plus one and
                # returning its first element.
                y = []
                y.append(k + 1)
                return y[0]

            return wrap(f, x)

        x = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(g, default_args_generator((x,)), arg_count)

    def test_wrap_kwarg(self):
        # [FUNCTION DOC] Tests wrap with a keyword tensor argument; expects 3 wrap args static (4
        # dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding two arguments, passing y as a
            # keyword.
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_int(self):
        # [FUNCTION DOC] Tests wrap with a keyword integer argument; the expected wrap arg count
        # gets one extra when dynamic shape capture is active.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding two arguments, passing y as a
            # keyword.
            return wrap(lambda x, y: x + y, x, y=y)

        x = torch.randn(3)
        y = 8

        arg_count = (
            ifdynstaticdefault(2, 3) + 1
            if check_dynamic_shape_capture()
            else ifdynstaticdefault(2, 3)
        )
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_all_kwarg(self):
        # [FUNCTION DOC] Tests wrap called with all arguments passed as keywords; expects 3 wrap
        # args static (4 dynamic).
        def f(y, x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda computing x times two plus y, passing
            # both as keywords.
            return wrap(lambda x, y: (x * 2) + y, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_only(self):
        # [FUNCTION DOC] Tests wrap on a keyword only body function with keyword arguments;
        # expects 3 wrap args static (4 dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with keyword arguments x and y.
            def fn(*, x, y):
                # [FUNCTION DOC] HELPER: Keyword only wrap body returning x times two plus y.
                return (x * 2) + y

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_default(self):
        # [FUNCTION DOC] Tests a keyword only wrap body that has a default keyword z of 8 which
        # the caller does not pass; expects 3 wrap args static (4 dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with keyword arguments x and y only.
            def fn(*, x, y, z=8):
                # [FUNCTION DOC] HELPER: Keyword only body returning x times two plus y plus the
                # keyword z (default 8).
                return (x * 2) + y + z

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_default_if_branch(self):
        # [FUNCTION DOC] Tests a wrap body whose default keyword z is None and branches on it;
        # with z omitted it takes the first branch. Expects 3 wrap args static (4 dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with keyword arguments x and y.
            def fn(*, x, y, z=None):
                # [FUNCTION DOC] HELPER: Keyword only body returning x times two plus y when z is
                # None, otherwise two times x.
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x, y)), arg_count)

    def test_wrap_kwarg_recompile(self):
        # [FUNCTION DOC] Tests that changing whether an optional keyword is passed triggers a
        # recompile. Compiles with the eager backend and fullgraph, checks that the captured call
        # counter stats equal 2 after the first call and stay 2 on a repeat call, then 4 after
        # calling with z set to 8, whose output must equal two times x.
        def f(x, y, z=None):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with keyword arguments x, y, z.
            def fn(*, x, y, z=None):
                # [FUNCTION DOC] HELPER: Keyword only body returning x times two plus y when z is
                # None, otherwise two times x.
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        counters.clear()
        opt = torch.compile(f, backend="eager", fullgraph=True)
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        # verify that we `don't` recompile
        opt(x, y)
        self.assertEqual(counters["stats"]["calls_captured"], 2)

        output = opt(x, y, 8)
        self.assertEqual(counters["stats"]["calls_captured"], 4)
        self.assertEqual(output, 2 * x)

    def test_wrap_kwarg_default_else_branch(self):
        # [FUNCTION DOC] Tests a wrap body whose keyword z is supplied as the constant 8 (so the
        # else branch is taken); expects 2 wrap args static (3 dynamic).
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Calls wrap on fn with keyword arguments x, y, z.
            def fn(*, x, y, z=None):
                # [FUNCTION DOC] HELPER: Keyword only body returning x times two plus y if z is
                # None, else two times x.
                if z is None:
                    return (x * 2) + y
                else:
                    return 2 * x

            return wrap(fn, x=x, y=y, z=z)

        x = torch.randn(3)
        y = torch.randn(3, 3)

        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(f, default_args_generator((x, y, 8)), arg_count)

    def test_map_subgraph_name_is_valid(self):
        # [FUNCTION DOC] Tests that doubly nested control flow map calls get valid, distinct
        # subgraph names; checks the exact printed outer and body graphs (only when not in dynamic
        # capture mode, where the helper returns nothing).
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        def map_f(xs, y):
            # [FUNCTION DOC] HELPER: Calls the control flow map operator over xs with the nested
            # function inner and the extra argument y.
            def inner(x, y):
                # [FUNCTION DOC] HELPER: Calls map over x using inner2 with y.
                def inner2(x, y):
                    # [FUNCTION DOC] HELPER: Returns x plus y.
                    return x + y

                return control_flow.map(inner2, x, y)

            return control_flow.map(inner, xs, y)

        graphs = self._check_map_graph_and_extract(map_f, (xs, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_xs_ : torch.Tensor, L_y_ : torch.Tensor):
    l_xs_ = L_xs_
    l_y_ = L_y_
    map_body_1 = self.map_body_1
    map_impl = torch.ops.higher_order.map_impl(map_body_1, [l_xs_], [l_y_]);  map_body_1 = l_xs_ = l_y_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, l_y_ : torch.Tensor):
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [child], [l_y_]);  map_body_0 = child = l_y_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )

    def test_map_multi_return(self):
        # [FUNCTION DOC] Tests that map with a body returning a tuple of two values produces an
        # outer graph and a body graph matching the expected text.
        def f(x):
            # [FUNCTION DOC] HELPER: Maps a lambda returning sin of x twice over x.
            return control_flow.map(lambda x: (x.sin(), x.sin()), x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem = map_impl[0]
    getitem_1 = map_impl[1];  map_impl = None
    return (getitem, getitem_1)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor):
    child_1 = child.sin()
    child_2 = child.sin();  child = None
    return (child_1, child_2)""",
            )

    def test_map_pytree_return(self):
        # [FUNCTION DOC] Tests map with a body returning a deeply nested pytree (tuple, nested
        # lists, dict) of clones; checks the exact outer and body graph text.
        def _construct_pytree(a):
            # [FUNCTION DOC] HELPER: Builds a nested structure of tuples, lists and a dict, all
            # containing clones of the input.
            return (
                a.clone(),
                [[[a.clone()]]],
                a.clone(),
                (a.clone(), (a.clone(),), a.clone()),
                {"a": a.clone()},
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Maps inner_f over x.
            def inner_f(xs):
                # [FUNCTION DOC] HELPER: Returns the pytree built from the slice.
                return _construct_pytree(xs)

            return control_flow.map(inner_f, x)

        x = torch.randn(3)
        graphs = self._check_map_graph_and_extract(f, (x,))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], []);  map_body_0 = l_x_ = None
    getitem = map_impl[0]
    getitem_1 = map_impl[1]
    getitem_2 = map_impl[2]
    getitem_3 = map_impl[3]
    getitem_4 = map_impl[4]
    getitem_5 = map_impl[5]
    getitem_6 = map_impl[6];  map_impl = None
    return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor):
    child_1 = child.clone()
    child_2 = child.clone()
    child_3 = child.clone()
    child_4 = child.clone()
    child_5 = child.clone()
    child_6 = child.clone()
    child_7 = child.clone();  child = None
    return (child_1, child_2, child_3, child_4, child_5, child_6, child_7)""",
            )

    def test_map_kwargs(self):
        # [FUNCTION DOC] Tests that map called with a keyword argument for the tensor raises
        # TypeError when the compiled function runs, and that no frame was compiled (frame count
        # zero).
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (counting backend) function calling map on a sin
            # lambda passing the tensor as a keyword.
            return control_flow.map(lambda x: x.sin(), x=x)

        x = torch.randn(3)
        self.assertRaises(TypeError, lambda: f(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_map_symint_input(self):
        # [FUNCTION DOC] Tests map where the extra argument is a symbolic size taken from another
        # tensor; checks the exact outer and body graph text.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls map over x with inner and the first dimension size of
            # y.
            def inner(x, y):
                # [FUNCTION DOC] HELPER: Returns sin of x plus y.
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, const_unused : int):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_lowers_to_graph(self):
        # [FUNCTION DOC] Tests that map with a size argument lowers to the expected outer and body
        # graphs (same function shape as the symbolic input test); asserts the exact printed text
        # of both.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls map over x with inner and the first dimension size of
            # y.
            def inner(x, y):
                # [FUNCTION DOC] HELPER: Returns sin of x plus y.
                return torch.sin(x + y)

            return control_flow.map(inner, x, y.size(0))

        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        graphs = self._check_map_graph_and_extract(fn, (x, y))
        if graphs:
            graph, body_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    map_body_0 = self.map_body_0
    map_impl = torch.ops.higher_order.map_impl(map_body_0, [l_x_], [3]);  map_body_0 = l_x_ = None
    getitem = map_impl[0];  map_impl = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                body_graph,
                """\
def forward(self, child : torch.Tensor, const_unused : int):
    add = child + 3;  child = None
    sin = torch.sin(add);  add = None
    return (sin,)""",
            )

    def test_map_example_value_metadata_consistent_with_eager(self):
        # [FUNCTION DOC] Tests that the fake example value stored on the compiled map node has the
        # same sizes, strides and requires_grad as the eager result of map_dense, and that both
        # are contiguous, over many inputs (plain, permuted, detached, narrowed, transposed,
        # strided and reshaped). Each iteration resets dynamo and clears recorded graphs.
        from torch._higher_order_ops.map import map_dense

        backend = EagerAndRecordGraphs()

        def inner(x):
            # [FUNCTION DOC] HELPER: Map body returning sin of x, the transpose of cos of x, and a
            # flattened view of sin of x.
            return x.sin(), x.cos().T, x.sin().view(-1)

        rand_44 = torch.randn(4, 4)
        inps = [
            torch.randn(3),
            torch.randn(3, 4),
            torch.randn(3, 4, 5, requires_grad=True),
            torch.randn(3, 4, 5, requires_grad=True).permute((2, 0, 1)),
            torch.randn(3, 4, 5, requires_grad=True).detach(),
            torch.randn(3, 4, 5, requires_grad=True).narrow(1, 1, 2),
            rand_44.T,
            rand_44[::2],
            rand_44[::2, ::2],
            rand_44[1::3, 1::3],
            rand_44[1::3, 1::2].T,
            rand_44.unsqueeze(1),
            rand_44.squeeze(0),
            rand_44.reshape(2, 8),
        ]
        for x in inps:
            compiled_ret = torch.compile(  # noqa: F841
                control_flow.map, backend=backend, fullgraph=True
            )(inner, x)
            eager_sin, eager_transpose, eager_view = map_dense(inner, (x,), ())

            map_node = next(
                node
                for node in backend.graphs[0].graph.nodes
                if node.op == "call_function" and "map" in node.name
            )

            fake_sin, fake_transpose, fake_view = map_node.meta["example_value"]

            def _check_size_stride_contiguous(x, y):
                # [FUNCTION DOC] HELPER: Asserts that y has the same size, stride and
                # requires_grad flag as x, and that both are contiguous.
                self.assertEqual(y.size(), x.size())
                self.assertEqual(y.stride(), x.stride())
                self.assertEqual(y.requires_grad, x.requires_grad)
                self.assertEqual(x.is_contiguous(), True)
                self.assertEqual(y.is_contiguous(), True)

            _check_size_stride_contiguous(eager_sin, fake_sin)
            _check_size_stride_contiguous(eager_transpose, fake_transpose)
            _check_size_stride_contiguous(eager_view, fake_view)

            torch._dynamo.reset()
            backend.graphs.clear()

    def test_cond_subgraph_name_is_valid(self):
        # [FUNCTION DOC] Tests that nested cond calls get unique valid submodule names; compiles a
        # nested cond and asserts the exact set of named modules in the graph (cond_true_1,
        # cond_false_1 and its two inner branches) and that the result equals xs plus y.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            # [FUNCTION DOC] HELPER: Compiled function calling cond on pred with true_fn and
            # false_fn on a list of pred2, x, y.
            def true_fn(pred2, x, y):
                # [FUNCTION DOC] HELPER: True branch returning x plus y.
                return x + y

            def false_fn(pred2, x, y):
                # [FUNCTION DOC] HELPER: False branch calling cond on pred2 with two inner
                # branches.
                def true_fn2(x, y):
                    # [FUNCTION DOC] HELPER: Inner true branch returning sin of x minus cos of y.
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    # [FUNCTION DOC] HELPER: Inner false branch returning cos of x minus sin of y.
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)

        cond_gm = backend.graphs[0]
        name_set = set()
        name_set.update(name for name, _ in cond_gm.named_modules())
        self.assertEqual(
            name_set,
            {
                "",
                "cond_true_1",
                "cond_false_1",
                "cond_false_1.cond_false_0",
                "cond_false_1.cond_true_0",
            },
        )

    @torch._dynamo.config.patch(
        assume_static_by_default=True,
        dynamic_shapes=True,
    )
    def test_cond_graph_break_in_one_branch(self):
        # [FUNCTION DOC] Tests that a cond whose true branch mutates a module buffer fails both in
        # eager and under compile (dynamic) by raising UncapturedHigherOrderOpError mentioning the
        # cond operator.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer of shape six by four whose forward
            # calls cond with a mutating true branch and a reducing false branch.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and registers the buffer.
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns cond on whether the sum of x exceeds four, with
                # the two branches below.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch that adds one to the buffer in place and
                    # returns buffer sum plus x sum.
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch returning the sum of x minus one.
                    return (x - 1).sum()

                return control_flow.cond(x.sum() > 4, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True)
        mod_for_eager = Foo()

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_eager(torch.ones(6, 4))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_compile(torch.ones(3, 4))

    def test_cond_free_variable_in_both_branches(self):
        # [FUNCTION DOC] Tests cond whose branches both use free variables (module buffer and
        # outer tensor z). Compiles dynamically with fullgraph, checks equality with eager, then
        # checks every cond node has 4 operands and every branch graph has 4 placeholders (buffer,
        # x, z and a symbol).
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = torch.ones(4, 4)

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones buffer; forward calls cond on y with branches
            # that use the buffer and captured z.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and registers the buffer.
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns cond on y with the branches below applied to x.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch returning sum of x plus buffer sum plus z
                    # sum.
                    return x.sum() + self.buffer.sum() + z.sum()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch returning sum of x minus z sum minus
                    # buffer sum.
                    return x.sum() - z.sum() - self.buffer.sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=True
        )
        mod_for_eager = Foo()

        self.assertEqual(
            mod_for_compile(torch.tensor(True), torch.tensor(5)),
            mod_for_eager(torch.tensor(True), torch.tensor(5)),
        )

        for node in backend.graphs[0].graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.higher_order.cond
            ):
                _, _, _, operands = node.args
                # Since we compile with dynamic, each branch takes 4 inputs (buffer, x, z, s1)
                self.assertEqual(len(operands), 4)
            if node.op == "get_attr":
                if str(node.target) in ("cond_true_0, cond_false_0"):
                    num_placeholders = len(
                        [
                            node
                            for node in getattr(
                                backend.graphs[0], str(node.target)
                            ).graph.nodes
                            if node.op == "placeholder"
                        ]
                    )
                    self.assertEqual(num_placeholders, 4)

    def _check_cond_graph_and_extract(self, fn, args):
        # [FUNCTION DOC] HELPER: Compiles fn with fullgraph, asserts output equals eager, one
        # frame and one graph. Returns None in dynamic capture mode, otherwise a tuple of the
        # stripped code text of the outer graph and the true and false branch submodules.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        true_graph = gm.cond_true_0.code.strip()
        false_graph = gm.cond_false_0.code.strip()
        return (graph, true_graph, false_graph)

    def _check_map_graph_and_extract(self, fn, args):
        # [FUNCTION DOC] HELPER: Compiles fn with fullgraph, asserts output equals eager, one
        # frame and one graph. Returns None in dynamic capture mode, otherwise a tuple with the
        # stripped outer graph code followed by the code of each submodule.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        gm = backend.graphs[0]
        graph = gm.code.strip()
        subgraphs = []
        for module_name in gm._modules:
            subgraphs.append(getattr(gm, module_name).code.strip())
        return (graph, *subgraphs)

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/176204"
    )
    def test_cond_branches_no_arguments(self):
        # [FUNCTION DOC] Tests cond with branches taking no arguments that close over x; skipped
        # on Linux or slow runs (see linked issue). Checks the outer, true and false graph text
        # when available.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls cond on whether the sum of x is positive with the two
            # branches and an empty operand tuple.
            def true_fn():
                # [FUNCTION DOC] HELPER: True branch returning sin of the captured x.
                return torch.sin(x)

            def false_fn():
                # [FUNCTION DOC] HELPER: False branch returning cos of the captured x.
                return torch.cos(x)

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum()
    gt = sum_1 > 0;  sum_1 = None
    cond_false_0 = self.cond_false_0
    cond_true_0 = self.cond_true_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (l_x_,));  gt = cond_true_0 = cond_false_0 = l_x_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    sin = torch.sin(l_x__1);  l_x__1 = None
    return (sin,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self, l_x_):
    l_x__1 = l_x_
    cos = torch.cos(l_x__1);  l_x__1 = None
    return (cos,)""",
            )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/183892"
    )
    def test_cond_branches_no_arguments_no_closure(self):
        # [FUNCTION DOC] Tests cond with argument-free branches that return fresh ones tensors
        # without capturing anything; skipped on Linux or slow runs. Calls the checker twice and
        # compares the printed graphs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls cond on whether the sum of x is positive with empty
            # operands.
            def true_fn():
                # [FUNCTION DOC] HELPER: True branch returning ones of shape three by four.
                return torch.ones(3, 4)

            def false_fn():
                # [FUNCTION DOC] HELPER: False branch returning sin of ones of shape three by
                # four.
                return torch.ones(3, 4).sin()

            return control_flow.cond(x.sum() > 0, true_fn, false_fn, ())

        self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            graph, true_graph, false_graph = graphs
            self.assertExpectedInline(
                graph,
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    sum_1 = l_x_.sum();  l_x_ = None
    gt = sum_1 > 0;  sum_1 = None
    cond_false_0 = self.cond_false_0
    cond_true_0 = self.cond_true_0
    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, ());  gt = cond_true_0 = cond_false_0 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
            self.assertExpectedInline(
                true_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    return (ones,)""",
            )
            self.assertExpectedInline(
                false_graph,
                """\
def forward(self):
    ones = torch.ones(3, 4)
    sin = ones.sin();  ones = None
    return (sin,)""",
            )

    def test_cond_side_effect_in_one_branches(self):
        # [FUNCTION DOC] Tests that a cond whose branch mutates an outer list (append, pop) raises
        # UncapturedHigherOrderOpError for eager and compile.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(4, 4)]

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls cond on y with branches that mutate
            # or read the outer list z.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Just calls the parent constructor.
                super().__init__()

            def forward(self, y, x):
                # [FUNCTION DOC] HELPER: Returns cond on y with the branches below applied to x.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch that appends x twice to the outer list,
                    # pops once, and returns x sum plus the last list element sum.
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x.sum() + z[-1].sum()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch returning x sum minus the first outer
                    # list element sum.
                    return x.sum() - z[0].sum()

                return control_flow.cond(y, true_fn, false_fn, [x])

        mod_for_eager = Foo()
        mod_for_compile = torch.compile(
            Foo(), backend=cnt, dynamic=True, fullgraph=False
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_eager(torch.tensor(True), torch.tensor(5))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            mod_for_compile(torch.tensor(True), torch.tensor(5))

    def test_cond_with_constant_pred(self):
        # [FUNCTION DOC] Tests that cond with a Python bool predicate works when compiled with the
        # eager backend, comparing results for true and false with allclose.
        def test(pred, x):
            # [FUNCTION DOC] HELPER: Calls cond on pred with the identity and negation branches.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: True branch returning x.
                return x

            def false_fn(x):
                # [FUNCTION DOC] HELPER: False branch returning the negation of x.
                return -x

            return control_flow.cond(pred, true_fn, false_fn, [x])

        opt_test = torch.compile(test, backend="eager")
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))

    def test_map_graph_break(self):
        # [FUNCTION DOC] Tests that map whose body mutates a module buffer raises
        # UncapturedHigherOrderOpError mentioning map_impl when run under dynamic compile.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer w whose forward maps a body that increments
            # the buffer in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and registers buffer w of ones.
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Returns map of body over xs.
                def body(x):
                    # [FUNCTION DOC] HELPER: Map body that adds one to the buffer in place and
                    # returns its input.
                    self.w += 1
                    return x

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

    def test_map_side_effect(self):
        # [FUNCTION DOC] Tests that map whose body mutates an outer list raises
        # UncapturedHigherOrderOpError mentioning map_impl under dynamic compile.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        z = [torch.ones(6, 4)]

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer w whose forward maps a body that mutates an
            # outer list.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and registers buffer w of ones.
                super().__init__()
                self.w = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Returns map of body over xs.
                def body(x):
                    # [FUNCTION DOC] HELPER: Map body that appends x twice and pops once on the
                    # outer list, returning x plus the sum of the last element.
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x + z[-1].sum()

                return control_flow.map(body, xs)

        mod = Module()

        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))

    def test_wrap_subgraph_name_is_valid(self):
        # [FUNCTION DOC] Tests that three nested wrap calls give valid names in the graph module:
        # the set of named modules is exactly the empty name plus wrap_body_2 and its nested
        # wrap_body_1 and wrap_body_0.
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            # [FUNCTION DOC] HELPER: Computes z as x plus y and returns two nested wraps over
            # lambdas that add z.
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function that wraps inner.
            return wrap(inner, x, y)

        result = f(x, y)

        self.assertEqual(result, x + y + x)
        wrap_gm = backend.graphs[0]
        names = set()
        names.update(mod_name for mod_name, _ in wrap_gm.named_modules())
        self.assertEqual(
            names,
            {
                "",
                "wrap_body_2",
                "wrap_body_2.wrap_body_1",
                "wrap_body_2.wrap_body_1.wrap_body_0",
            },
        )

    def test_wrap_allow_local_assign_in_body_fn(self):
        # [FUNCTION DOC] Tests that a wrap body may assign its arguments to local names, loop over
        # them, and return a list. Uses two argument generators (lists then tuples); expects 3 ops
        # and wrap args 3 static (4 dynamic); unless dynamic capture is on, checks exact graph
        # text.
        def f(arg1, arg2):
            # [FUNCTION DOC] HELPER: Calls wrap on inner_f with both arguments.
            def inner_f(arg1, arg2):
                # [FUNCTION DOC] HELPER: Wrap body that appends x plus one for every element of
                # both inputs to a local list and returns it.
                a = arg1
                b = arg2
                ret = []
                for x in a:
                    ret.append(x + 1)
                for x in b:
                    ret.append(x + 1)
                return ret

            return wrap(inner_f, arg1, arg2)

        x = torch.ones(3)

        def my_args_generator():
            # [FUNCTION DOC] HELPER: Generator yielding two argument pairs: lists holding x and
            # sin of x, then tuples of the same values.
            yield [x], [x.sin()]
            yield (x,), (x.sin(),)

        arg_count = ifdynstaticdefault(3, 4)
        actual_graph = self._test_wrap_simple(
            f,
            my_args_generator(),
            arg_count,
            3,
            return_graph=True,
        )

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_arg1_0_: "f32[3]", L_arg2_0_: "f32[3]"):
        l_arg1_0_ = L_arg1_0_
        l_arg2_0_ = L_arg2_0_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);  wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None
        getitem: "f32[3]" = wrap[0]
        getitem_1: "f32[3]" = wrap[1];  wrap = None
        return (getitem, getitem_1)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_arg1_0_: "f32[3]", l_arg2_0_: "f32[3]"):
            add: "f32[3]" = l_arg1_0_ + 1;  l_arg1_0_ = None

            add_1: "f32[3]" = l_arg2_0_ + 1;  l_arg2_0_ = None
            return (add, add_1)
""",
        )

    def test_capture_global_num(self):
        # [FUNCTION DOC] Tests capturing a module level global number inside a wrap lambda;
        # numbers are not lifted so wrap has 2 args.
        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the global number to its input.
            return wrap(lambda x: x + global_num, x)

        x = torch.zeros([])
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_capture_global_num_adds_guard(self):
        # [FUNCTION DOC] Tests that a global number captured in a wrap lambda adds a guard: after
        # rebinding the global to a new random float, the compiled function must return the new
        # value rather than a stale constant.
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (eager, fullgraph) function calling wrap on a lambda
            # adding the global number.
            return wrap(lambda x: x + global_num, x)

        global global_num
        x = torch.zeros([])
        result = f(x)
        self.assertEqual(result, x + global_num)

        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        # [FUNCTION DOC] Tests capturing a Python float argument inside a wrap lambda; numbers are
        # not lifted so wrap has 2 args.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the captured number y.
            return wrap(lambda x: x + y, x)

        x = torch.zeros([])
        y = 3.14
        # Numbers don't get lifted, so args is still 2.
        self._test_wrap_simple(f, default_args_generator((x, y)), 2)

    def test_side_effect_in_body(self):
        # [FUNCTION DOC] Tests that a wrap body assigning a nonlocal variable is rejected from the
        # graph: after compiling and running, the outer variable is updated as in eager and the
        # graph break counters contain exactly one entry matching unsafe side effect.
        counters.clear()
        backend = EagerAndRecordGraphs()

        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            # [FUNCTION DOC] HELPER: Wrap body that rebinds the nonlocal y to its input x and
            # returns a clone of x.
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=backend)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function (eager recording backend, no fullgraph)
            # that wraps inner.
            return wrap(inner, x)

        f(x)
        self.assertEqual(y, x)
        assert_dict_matches_regex(
            self,
            dict(counters["graph_break"]),
            {"HOP: Unsafe side effect": 1},
        )

    def test_fallback_on_graph_break_simple(self):
        # [FUNCTION DOC] Tests fallback when a wrap body contains an explicit graph break: the
        # compiled result equals eager and no frame was compiled (frame count zero).
        # In the future, there should be a per-HigherOrderOperator switch
        # on whether or not to fallback or raise a loud error.
        # For now we just fallback by default.
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            # [FUNCTION DOC] HELPER: Body that computes sin, triggers a dynamo graph break, then
            # returns sin again.
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function calling wrap on inner.
            return wrap(inner, x)

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_fallback_on_graph_break_complicated(self):
        # [FUNCTION DOC] Tests fallback when a wrap body breaks the graph after using a global
        # tensor: the result equals eager and exactly two frames were compiled (before and after
        # the wrapped call).
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            # [FUNCTION DOC] HELPER: Body that takes sin, multiplies by the global tensor,
            # triggers a graph break, then takes sin again.
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function that clones x, calls wrap on inner, and
            # returns a clone of the result.
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()

        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 2)

    def test_modules(self):
        # [FUNCTION DOC] Tests wrap of a lambda that calls a torch.nn.Linear module. Asserts one
        # frame, one graph, and a body graph with exactly one op, a linear call, whose parameters
        # are not owned as parameters or children of the body module (they are lifted).
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function calling wrap on a lambda that
            # applies the linear module.
            return wrap(lambda x: mod(x), x)

        result = f(x)

        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        # 3 args - 1 for input, and other 2 for the weight and bias
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        linear_node = find_first_node(body_function, torch._C._nn.linear)
        self.assertTrue(linear_node is not None)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_flat_list_output(self):
        # [FUNCTION DOC] Tests a wrap body returning a flat list of two tensors; expects 3 ops and
        # 2 wrap args static (3 dynamic).
        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda returning sin and cos of its input as
            # a list.
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, expected_opcount=3
        )

    def test_support_float_in_output(self):
        # [FUNCTION DOC] Tests that a wrap body can return a list containing Python int and float
        # constants along with a tensor; compiled result must equal the plain list.
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function calling wrap on a lambda
            # returning a one, sin of x and a two point zero.
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)

        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [1, torch.sin(x), 2.0])

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/176969"
    )
    def test_nested_tuple_output(self):
        # [FUNCTION DOC] Tests wrap returning a nested tuple of two tensors, destructured by the
        # caller; skipped on Linux or slow runs. Asserts no graph break, 4 ops, wrap args 2 static
        # (4 dynamic), and exact graph text except in dynamic capture mode.
        def f(x):
            # [FUNCTION DOC] HELPER: Unpacks the nested tuple returned by wrap of a lambda
            # producing sin and cos, and returns their sum.
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            return a + b

        x = torch.randn(2, 3)

        counters.clear()
        arg_count = ifdynstaticdefault(2, 4)
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 4, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[2, 3]" = wrap[0]
        getitem_1: "f32[2, 3]" = wrap[1];  wrap = None

        add: "f32[2, 3]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[2, 3]"):
            sin: "f32[2, 3]" = l_x_.sin()
            cos: "f32[2, 3]" = l_x_.cos();  l_x_ = None
            return (sin, cos)
""",
        )

    def test_output_with_dict(self):
        # [FUNCTION DOC] Tests wrap with a body returning a list containing a dict; asserts no
        # graph break, 2 ops, and the exact graph text (skipped in dynamic capture mode).
        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda returning a list holding a dict with
            # the negated input.
            return wrap(lambda x: [{"a": -x}], x)

        x = torch.randn(3)

        counters.clear()

        arg_count = ifdynstaticdefault(2, 3)
        graph = self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, 2, return_graph=True
        )
        self.assertEqual(len(counters["graph_break"]), 0)

        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3]"):
            neg: "f32[3]" = -l_x_;  l_x_ = None
            return (neg,)
""",
        )

    def test_access_module_attr(self):
        # [FUNCTION DOC] Tests accessing a module attribute (bias) inside a wrap lambda after
        # calling the module outside. Asserts one frame, one graph, a body graph with a single op,
        # and no parameters or children owned by the body module.
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function applying the linear module,
            # then wrap on a lambda subtracting its bias.
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)

        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 1)

        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)

        # Check that the inner function has one op and its a linear op
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)

        # Check that the innermost graph does not have any params
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_make_closure(self):
        # [FUNCTION DOC] Tests wrap of a function that builds and calls an inner closure over its
        # argument y; expects 3 wrap args static (4 dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Defines a closure adding y and returns it applied to x.
            def g(x):
                # [FUNCTION DOC] HELPER: Returns x plus the captured y.
                return x + y

            return g(x)

        def h(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on f with both arguments.
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(h, default_args_generator((x, y)), arg_count)

    def test_internal_nonlocal(self):
        # [FUNCTION DOC] Tests a wrap body using internal nonlocal variables that are mutated by
        # nested helper functions entirely within the body; this is allowed. Expects 3 wrap args
        # static (4 dynamic).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Wrap body keeping a local counter w, updated by g and h, and
            # returning w plus y.
            w = 1

            def g(x):
                # [FUNCTION DOC] HELPER: Sets the nonlocal w to x and returns x.
                nonlocal w
                w = x
                return x

            def h(x):
                # [FUNCTION DOC] HELPER: Increments the nonlocal w by one and returns x.
                nonlocal w
                w = w + 1
                return x

            g(x)
            h(x)
            return w + y

        def h(x, y):
            # [FUNCTION DOC] HELPER: Calls wrap on f with both arguments.
            return wrap(f, x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(h, default_args_generator((x, y)), arg_count)

    def test_capture_numpy_number(self):
        # [FUNCTION DOC] Tests capturing a numpy float32 scalar in a wrap lambda; numpy numbers
        # are lifted to graph inputs so wrap has 3 args static (4 dynamic).
        import numpy as np

        y = np.float32(1.0)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda adding the captured numpy scalar.
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        # np.number are lifted to graph inputs
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_freevars_as_inputs_to_wrap(self):
        # [FUNCTION DOC] Tests passing a tensor explicitly as an extra argument to wrap in
        # addition to the input; expects 3 wrap args static (4 dynamic).
        y = torch.randn(3)

        def f(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda of two arguments adding them, with x
            # and the outer y.
            return wrap(lambda x, y: x + y, x, y)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(f, default_args_generator((x,)), arg_count)

    def test_lift_tensor_constant(self):
        # [FUNCTION DOC] Tests a wrap lambda capturing a tensor constant created inside the
        # function; expects 3 ops and wrap args 3 static (4 dynamic).
        def f(x):
            # [FUNCTION DOC] HELPER: Creates a scalar tensor y and calls wrap on a lambda adding
            # y.
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(3, 4)
        self._test_wrap_simple(
            f, default_args_generator((x,)), arg_count, expected_opcount=3
        )

    def test_nested_wrap(self):
        # [FUNCTION DOC] Tests two levels of wrap where the inner wrap call is a module; expects 4
        # wrap args static (5 dynamic).
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer of ten by ten; forward applies it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to the input.
                return self.linear(x)

        mod = MockModule()

        # Two levels of wrap ops
        def gn(x):
            # [FUNCTION DOC] HELPER: Returns cos of x plus the result of wrapping the module call
            # on x.
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on gn.
            return wrap(gn, x)

        arg_count = ifdynstaticdefault(4, 5)
        self._test_wrap_simple(
            fn, default_args_generator((torch.randn(10, 10),)), arg_count
        )

    def test_fn_with_kwargs_in_torch_ops(self):
        # [FUNCTION DOC] Tests that torch operations called with keyword arguments (cos with input
        # keyword) trace inside wrap; expects 2 wrap args static (3 dynamic).
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on a lambda applying cos with the keyword input.
            return wrap(lambda z: torch.cos(input=z), x)

        x = torch.randn(3)
        arg_count = ifdynstaticdefault(2, 3)
        self._test_wrap_simple(fn, default_args_generator((x,)), arg_count)

    def test_hooks(self):
        # [FUNCTION DOC] Tests that a forward hook registered on a child module still fires on the
        # second compiled iteration (hooks fire during tracing on the first). Runs forward and
        # backward twice and asserts the recorded activations cover exactly the hooked child
        # names.
        class ToyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer named net of ten by ten; forward
            # applies it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and creates the linear layer net.
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the net layer applied to x.
                return self.net(x)

        model = ToyModel()
        forward_handles = {}
        activations = {}

        def save_activations(mod, inp, out):
            # [FUNCTION DOC] HELPER: Forward hook storing the hook input in the activations
            # mapping under the module name of the loop variable at hook time.
            activations[name] = inp

        for name, module in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function calling wrap on a lambda applying
            # the model.
            return wrap(lambda x: model(x), x)

        for _ in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((10, 10))
            pred = fn(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(activations.keys() == forward_handles.keys())

    def _get_source_fn_stack(self, gm, node_names):
        # [FUNCTION DOC] HELPER: Walks all modules of gm and their graph nodes; for each node
        # whose name is in node_names, stores in a result mapping the list of function names taken
        # from its source_fn_stack metadata. Returns the mapping.
        ret = {}
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in node_names:
                    actual_stack = [
                        name for name, _ in node.meta.get("source_fn_stack", [])
                    ]
                    ret[node.name] = actual_stack
        return ret

    def test_wrap_source_fn_stack(self):
        # [FUNCTION DOC] Tests source_fn_stack metadata of nodes inside nested wrap with a module:
        # compiles, gets the stacks for cos, add and linear and compares to the pretty printed
        # expected text.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a four by four linear layer; forward applies it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x.
                return self.linear(x)

        mod = MockModule()

        def gn(x):
            # [FUNCTION DOC] HELPER: Returns cos of x plus wrap of the module on x.
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls wrap on gn.
            return wrap(gn, x)

        backend = EagerAndRecordGraphs()
        inp = torch.randn((4, 4))
        torch.compile(fn, backend=backend, fullgraph=True)(inp)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "linear"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['wrap', 'add'],
 'cos': ['wrap', 'cos'],
 'linear': ['wrap', 'wrap', 'linear']}""",
        )

    def test_cond_source_fn_stack(self):
        # [FUNCTION DOC] Tests source_fn_stack metadata for nodes in nested cond branches (cos,
        # add, sin, sub); compares the pretty printed mapping to the expected text.
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function calling cond on pred with a
            # nested cond inside the false branch.
            def true_fn(pred2, x, y):
                # [FUNCTION DOC] HELPER: True branch returning x plus y.
                return x + y

            def false_fn(pred2, x, y):
                # [FUNCTION DOC] HELPER: False branch calling cond on pred2 with the two inner
                # branches.
                def true_fn2(x, y):
                    # [FUNCTION DOC] HELPER: Inner true branch returning sin of x minus cos of y.
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    # [FUNCTION DOC] HELPER: Inner false branch returning cos of x minus sin of y.
                    return x.cos() - y.sin()

                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])

            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])

        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)
        cond_f(pred, pred2, xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin", "sub"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['cond', 'add'],
 'cos': ['cond', 'cond', 'cos'],
 'sin': ['cond', 'cond', 'sin'],
 'sub': ['cond', 'cond', 'sub']}""",
        )

    def test_map_source_fn_stack(self):
        # [FUNCTION DOC] Tests source_fn_stack metadata for nodes inside nested map calls (cos,
        # add, sin); compares the pretty printed mapping to the expected text.
        backend = EagerAndRecordGraphs()

        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=backend, fullgraph=True)
        def map_f(xs, y):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function applying sin to the result of
            # map over xs with inner and y.
            def inner(x, y):
                # [FUNCTION DOC] HELPER: Map body that maps inner2 over x with y and multiplies
                # the result by cos of y.
                def inner2(x, y):
                    # [FUNCTION DOC] HELPER: Returns x plus y.
                    return x + y

                return control_flow.map(inner2, x, y) * y.cos()

            return control_flow.map(inner, xs, y).sin()

        map_f(xs, y)

        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"cos", "add", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """\
{'add': ['map_impl', 'map_impl', 'add'],
 'cos': ['map_impl', 'cos'],
 'sin': ['sin']}""",
        )

    def test_grad_source_fn_stack(self):
        # [FUNCTION DOC] Tests source_fn_stack for sin and sum nodes under a double application of
        # grad; expects both nodes to have their own name as the only stack entry.
        backend = EagerAndRecordGraphs()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        @torch.compile(backend=backend, fullgraph=False)
        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Compiled (no fullgraph) function applying grad twice to fn at
            # x.
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())

        wrapper_fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {"sum_1", "sin"})
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sin': ['sin'], 'sum_1': ['sum_1']}""",
        )

    def test_vmap_multiply_scalar(self):
        # [FUNCTION DOC] Tests vmap with a Python float or int constant for the unmapped argument
        # under the inductor backend; each compiled result must equal x times the constant.
        @torch.compile(backend="inductor", fullgraph=True)
        def g(x):
            # [FUNCTION DOC] HELPER: Compiled (inductor, fullgraph) function vmapping
            # multiplication over x with the constant 3.14 not mapped.
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 3.14)

        x = torch.randn(3)
        y = g(x)
        self.assertEqual(y, x * 3.14)

        @torch.compile(backend="inductor", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (inductor, fullgraph) function vmapping
            # multiplication over x with the integer constant 314 not mapped.
            return torch.vmap(torch.mul, in_dims=(0, None))(x, 314)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x * 314)

    def test_vmap_source_fn_stack(self):
        # [FUNCTION DOC] Tests source_fn_stack for sum nodes under nested vmap; the batched_output
        # name is absent from the mapping, and sum_1 and sum_2 have their own names as stack.
        backend = EagerAndRecordGraphs()

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap over x of a lambda adding the sums over
            # dimensions zero and one.
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph) function vmapping a lambda that applies
            # cos and inner_fn.
            return torch.func.vmap(lambda x: inner_fn(x.cos()))(x)

        x = torch.randn(3, 3, 3, 3)
        fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(
            gm, {"sum_1", "sum_2", "batched_output"}
        )
        self.assertExpectedInline(
            pprint.pformat(actual_stack),
            """{'sum_1': ['sum_1'], 'sum_2': ['sum_2']}""",
        )

    # https://github.com/pytorch/pytorch/issues/137061
    def test_dynamic_shapes_over_vmap_batch_size(self):
        # [FUNCTION DOC] Tests that four nested vmaps over sizes created by arange (issue 137061)
        # handle dynamic shapes: compares compiled and eager results for two size sets, checks
        # frame counts (second compile only if static), and, in dynamic capture mode, compares the
        # generated graph text.
        def gn(a, b, c, d):
            # [FUNCTION DOC] HELPER: Returns the sum of four arguments.
            return a + b + c + d

        def fn(func, a, b, c, d):
            # [FUNCTION DOC] HELPER: Creates four arange tensors, wraps func in four nested vmaps
            # each mapping one argument, and calls it.
            a = torch.arange(a)
            b = torch.arange(b)
            c = torch.arange(c)
            d = torch.arange(d)
            func = torch.vmap(func, in_dims=(0, None, None, None))
            func = torch.vmap(func, in_dims=(None, 0, None, None))
            func = torch.vmap(func, in_dims=(None, None, 0, None))
            func = torch.vmap(func, in_dims=(None, None, None, 0))
            return func(a, b, c, d)

        cnt = CompileCounterWithBackend("eager")
        # We generate corresponding dynamic shapes test case at
        # `test/dynamo/test_dynamic_shapes.py` automatically.
        compiled_fn = torch.compile(fn, backend=cnt)
        a, b, c, d = 2, 4, 8, 8
        self.assertEqual(fn(gn, a, b, c, d), compiled_fn(gn, a, b, c, d))
        self.assertEqual(cnt.frame_count, 1)

        a, b, c, d = 4, 8, 16, 16
        self.assertEqual(fn(gn, a, b, c, d), compiled_fn(gn, a, b, c, d))
        # Ensure no recompile if dynamic shapes enabled.
        self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))
        graph = cnt.graphs[0]

        # Check dynamic shapes generates correct graph.
        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                graph.code.strip(),
                """\
def forward(self, L_a_ : torch.SymInt, L_b_ : torch.SymInt, L_c_ : torch.SymInt, L_d_ : torch.SymInt):
    l_a_ = L_a_
    l_b_ = L_b_
    l_c_ = L_c_
    l_d_ = L_d_
    arange = torch.arange(l_a_)
    arange_1 = torch.arange(l_b_)
    arange_2 = torch.arange(l_c_)
    arange_3 = torch.arange(l_d_)
    lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None
    _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(l_d_, 'error');  _vmap_increment_nesting = None
    _add_batch_dim = torch._functorch.predispatch._add_batch_dim(arange_3, 0, 1);  arange_3 = None
    lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None
    _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(l_c_, 'error');  _vmap_increment_nesting_1 = None
    _add_batch_dim_1 = torch._functorch.predispatch._add_batch_dim(arange_2, 0, 2);  arange_2 = None
    lazy_load_decompositions_2 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_2 = None
    _vmap_increment_nesting_2 = torch._functorch.predispatch._vmap_increment_nesting(l_b_, 'error');  _vmap_increment_nesting_2 = None
    _add_batch_dim_2 = torch._functorch.predispatch._add_batch_dim(arange_1, 0, 3);  arange_1 = None
    lazy_load_decompositions_3 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_3 = None
    _vmap_increment_nesting_3 = torch._functorch.predispatch._vmap_increment_nesting(l_a_, 'error');  _vmap_increment_nesting_3 = None
    _add_batch_dim_3 = torch._functorch.predispatch._add_batch_dim(arange, 0, 4);  arange = None
    add = _add_batch_dim_3 + _add_batch_dim_2;  _add_batch_dim_3 = _add_batch_dim_2 = None
    add_1 = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None
    add_2 = add_1 + _add_batch_dim;  add_1 = _add_batch_dim = None
    _remove_batch_dim = torch._functorch.predispatch._remove_batch_dim(add_2, 4, l_a_, 0);  add_2 = l_a_ = None
    _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
    _remove_batch_dim_1 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim, 3, l_b_, 0);  _remove_batch_dim = l_b_ = None
    _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
    _remove_batch_dim_2 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_1, 2, l_c_, 0);  _remove_batch_dim_1 = l_c_ = None
    _vmap_decrement_nesting_2 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_2 = None
    _remove_batch_dim_3 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_2, 1, l_d_, 0);  _remove_batch_dim_2 = l_d_ = None
    _vmap_decrement_nesting_3 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_3 = None
    return (_remove_batch_dim_3,)""",
            )

    def test_cond_pytree_operands(self):
        # [FUNCTION DOC] Tests cond with a pytree operand (tensor, nested lists, tuples, dict).
        # Compares compiled and eager outputs and, outside dynamic capture mode, the exact
        # generated code of the graph.
        def _construct_pytree():
            # [FUNCTION DOC] HELPER: Creates seven random three by three tensors and returns them
            # in a nested structure of a tuple, nested lists, a tuple and a dict.
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)
            c = torch.randn(3, 3)
            d = torch.randn(3, 3)
            e = torch.randn(3, 3)
            f = torch.randn(3, 3)
            g = torch.randn(3, 3)
            return (a, [[[b]]], c, (d, (e,), f), {"g": g})

        pred = torch.tensor(True)
        inp = _construct_pytree()

        def _reduce_sum(flattened):
            # [FUNCTION DOC] HELPER: Returns the sum of all items of a flat list, starting from
            # zero.
            init = 0
            for val in flattened:
                init += val
            return init

        def _reduce_max(flattened):
            # [FUNCTION DOC] HELPER: Returns the maximum of all items of a flat list using the
            # first item as the start.
            init = flattened[0]
            for val in flattened:
                init = max(val, init)
            return init

        def true_fn(pytree_in):
            # [FUNCTION DOC] HELPER: Flattens the pytree and returns the sum of its leaves.
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_sum(flattened)

        def false_fn(pytree_in):
            # [FUNCTION DOC] HELPER: Flattens the pytree and returns the max of its leaves.
            flattened, spec = pytree.tree_flatten(pytree_in)
            return _reduce_max(flattened)

        def fn(pred, pytree_in):
            # [FUNCTION DOC] HELPER: Calls cond with the predicate and the pytree wrapped in a
            # list.
            return torch.cond(pred, true_fn, false_fn, [pytree_in])

        backend = EagerAndRecordGraphs()
        compiled_res = torch.compile(fn, backend=backend)(pred, inp)
        eager_res = fn(pred, inp)
        self.assertEqual(compiled_res, eager_res)
        graph = backend.graphs[0]

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, L_pred_ : torch.Tensor, L_pytree_in_0_ : torch.Tensor, L_pytree_in_1_0_0_0_ : torch.Tensor, L_pytree_in_2_ : torch.Tensor, L_pytree_in_3_0_ : torch.Tensor, L_pytree_in_3_1_0_ : torch.Tensor, L_pytree_in_3_2_ : torch.Tensor, L_pytree_in_4_g_ : torch.Tensor):
    l_pred_ = L_pred_
    l_pytree_in_0_ = L_pytree_in_0_
    l_pytree_in_1_0_0_0_ = L_pytree_in_1_0_0_0_
    l_pytree_in_2_ = L_pytree_in_2_
    l_pytree_in_3_0_ = L_pytree_in_3_0_
    l_pytree_in_3_1_0_ = L_pytree_in_3_1_0_
    l_pytree_in_3_2_ = L_pytree_in_3_2_
    l_pytree_in_4_g_ = L_pytree_in_4_g_
    cond_false_0 = self.cond_false_0
    cond_true_0 = self.cond_true_0
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, (l_pytree_in_0_, l_pytree_in_1_0_0_0_, l_pytree_in_2_, l_pytree_in_3_0_, l_pytree_in_3_1_0_, l_pytree_in_3_2_, l_pytree_in_4_g_));  l_pred_ = cond_true_0 = cond_false_0 = l_pytree_in_0_ = l_pytree_in_1_0_0_0_ = l_pytree_in_2_ = l_pytree_in_3_0_ = l_pytree_in_3_1_0_ = l_pytree_in_3_2_ = l_pytree_in_4_g_ = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )

    def test_cond_pytree_operands_with_non_tensor_leaves(self):
        # [FUNCTION DOC] Tests that cond rejects non tensor pytree leaves (a string or a float).
        # Eager raises RuntimeError saying operands must be a tuple of possibly nested dict, list,
        # or tuple, while compiled raises UncapturedHigherOrderOpError.
        def fn(pred, pytree_in):
            # [FUNCTION DOC] HELPER: Calls cond with lambdas that add one to or double the first
            # element of the pytree.
            return torch.cond(
                pred, lambda x: x[0] + 1, lambda x: x[0] * 2, (pytree_in,)
            )

        pred = torch.tensor(True)
        for pytree_in in [("string",), (1.0,)]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"Expect operands to be a tuple of possibly nested dict/list/tuple",
            ):
                fn(pred, pytree_in)

        for pytree_in in [("string",), (1.0,)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn, backend="eager")(pred, pytree_in)

    def test_cond_with_empty_operands(self):
        # [FUNCTION DOC] Tests cond called with no operand list; the branches close over y and z.
        # Checks output two for a false-like predicate and three for a true-like predicate.
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Compiled (fullgraph, eager) function calling cond with
            # argument-free branches.
            def true_fn():
                # [FUNCTION DOC] HELPER: True branch returning y plus two.
                return y + 2

            def false_fn():
                # [FUNCTION DOC] HELPER: False branch returning z plus one.
                return z + 1

            return torch.cond(x, true_fn, false_fn)

        zeros = torch.zeros(1)
        ones = torch.ones(1)
        self.assertEqual(fn(zeros, ones, ones), torch.tensor([2.0]))
        self.assertEqual(fn(ones, ones, ones), torch.tensor([3.0]))

    def test_hopify_generic_wrap(self):
        # [FUNCTION DOC] Tests building custom higher order ops with dynamo_bypassing_wrapper: one
        # adds a constant k to the function output, another applies a user function g to the
        # output. Compiles with the aot_eager_decomp_partition backend and fullgraph and compares
        # to eager, including a composition of both.
        from torch._higher_order_ops.wrap import dynamo_bypassing_wrapper

        def my_hop_fn_impl(fn, *args, k=1, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper factory that returns a function calling fn and adding
            # k to the result (to the first element if the result is a tuple, keeping it a one
            # element tuple).
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls fn and adds k to the output or its first element if
                # a tuple.
                out = fn(*args, **kwargs)
                if isinstance(out, tuple):
                    return (out[0] + k,)
                return out + k

            return wrapper

        def my_hop_fn(fn, *args, k=1, **kwargs):
            # [FUNCTION DOC] HELPER: Calls dynamo_bypassing_wrapper with a partially applied
            # implementation fixed to k.
            return dynamo_bypassing_wrapper(
                functools.partial(my_hop_fn_impl, k=k), fn, *args, **kwargs
            )

        def my_hop_fn_2_impl(fn, *args, g=None):
            # [FUNCTION DOC] HELPER: Wrapper factory returning a function that asserts g is set,
            # calls fn, and applies g to the output (or its first element if a tuple).
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls fn and applies g to the result, asserting g is not
                # None.
                assert g is not None  # noqa: S101
                out = fn(*args)
                if isinstance(out, tuple):
                    return (g(out[0]),)
                return g(out)

            return wrapper

        def my_hop_fn_2(fn, *args, g=None, **kwargs):
            # [FUNCTION DOC] HELPER: Calls dynamo_bypassing_wrapper with the implementation
            # partially applied to g.
            return dynamo_bypassing_wrapper(
                functools.partial(my_hop_fn_2_impl, g=g), fn, *args, **kwargs
            )

        def gn(x, h=1):
            # [FUNCTION DOC] HELPER: Returns sin of x plus h.
            return x.sin() + h

        def fn(x, b):
            # [FUNCTION DOC] HELPER: Calls my_hop_fn on gn with h set to b and k set to two.
            out = my_hop_fn(gn, x, h=b, k=2)
            return out

        a = torch.rand((4, 4), requires_grad=True)
        b = torch.rand((4, 4))
        compiled_fn = torch.compile(
            fn, backend="aot_eager_decomp_partition", fullgraph=True
        )
        self.assertEqual(compiled_fn(a, b), fn(a, b))

        def g(x):
            # [FUNCTION DOC] HELPER: Returns cos of x.
            return x.cos()

        def fn_2(x, b):
            # [FUNCTION DOC] HELPER: Calls my_hop_fn_2 on fn with x, b and g.
            out = my_hop_fn_2(fn, x, b, g=g)
            return out

        a = torch.rand((4, 4), requires_grad=True)
        compiled_fn_2 = torch.compile(
            fn_2, backend="aot_eager_decomp_partition", fullgraph=True
        )
        self.assertEqual(compiled_fn_2(a, b), fn_2(a, b))

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/184537"
    )
    def test_hints_wrapper(self):
        # [FUNCTION DOC] Tests nested hints_wrapper calls with hints dictionaries: eager and
        # compiled results must equal a plain reference; one graph is captured and, outside
        # dynamic capture mode, the printed graph is compared. Skipped on Linux or slow runs.
        def ref_fn(x, y):
            # [FUNCTION DOC] HELPER: Reference implementation: add, relu, add, abs.
            x = x + y
            x = torch.relu(x)
            x = x + y
            return torch.abs(x)

        def fn_with_hints(x, y):
            # [FUNCTION DOC] HELPER: Adds y to x then calls hints_wrapper on outer_body_fn with an
            # outer_body hint.
            x = x + y

            def inner_body_fn(x, y):
                # [FUNCTION DOC] HELPER: Applies relu then adds y.
                x = torch.relu(x)
                x = x + y
                return x

            def outer_body_fn(x, y):
                # [FUNCTION DOC] HELPER: Calls hints_wrapper on inner_body_fn with an inner_body
                # hint then applies abs.
                x = hints_wrapper(inner_body_fn, (x, y), {}, hints={"inner_body": True})
                x = torch.abs(x)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {}, hints={"outer_body": True})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        eager_res = fn_with_hints(x, y)
        compiled_res = torch.compile(fn_with_hints, backend=cnt)(x, y)
        ref_res = ref_fn(x, y)
        self.assertEqual(eager_res, ref_res)
        self.assertEqual(compiled_res, ref_res)
        self.assertEqual(len(cnt.graphs), 1)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        graph = backend.graphs[0]
        self.assertExpectedInline(
            normalize_gm(graph.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4]", L_y_: "f32[4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        add: "f32[2, 4]" = l_x_ + l_y_;  l_x_ = None

        hints_wrapper_body_1 = self.hints_wrapper_body_1
        hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_1, (add, l_y_), {}, hints = {'outer_body': True});  hints_wrapper_body_1 = add = l_y_ = None
        getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None
        return (getitem,)

    class hints_wrapper_body_1(torch.nn.Module):
        def forward(self, x: "f32[2, 4]", l_y_: "f32[4]"):
            hints_wrapper_body_0 = self.hints_wrapper_body_0
            hints_wrapper = torch.ops.higher_order.hints_wrapper(hints_wrapper_body_0, (x, l_y_), {}, hints = {'inner_body': True});  hints_wrapper_body_0 = x = l_y_ = None
            getitem: "f32[2, 4]" = hints_wrapper[0];  hints_wrapper = None

            x_1: "f32[2, 4]" = torch.abs(getitem);  getitem = None
            return (x_1,)

        class hints_wrapper_body_0(torch.nn.Module):
            def forward(self, x: "f32[2, 4]", l_y_: "f32[4]"):
                x_1: "f32[2, 4]" = torch.relu(x);  x = None

                x_2: "f32[2, 4]" = x_1 + l_y_;  x_1 = l_y_ = None
                return (x_2,)
""",
        )

    def test_hints_wrapper_no_hints(self):
        # [FUNCTION DOC] Tests that hints_wrapper without a hints keyword raises RuntimeError
        # mentioning improper args or kwargs when compiled.
        def fn_with_hints(x, y):
            # [FUNCTION DOC] HELPER: Calls hints_wrapper on outer_body_fn without hints.
            def outer_body_fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                x = torch.add(x, y)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = "hints_wrapper: improper args/kwargs"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn_with_hints, backend=cnt)(x, y)

    def test_hints_wrapper_incorrect_type(self):
        # [FUNCTION DOC] Tests that hints_wrapper with a hint whose value is a tuple raises
        # RuntimeError saying hints must be a dict of int, float, bool or str values.
        def fn_with_hints(x, y):
            # [FUNCTION DOC] HELPER: Calls hints_wrapper with the bad hints mapping.
            def outer_body_fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                x = torch.add(x, y)
                return x

            res = hints_wrapper(outer_body_fn, (x, y), {}, hints={"test": (True,)})
            return res

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = r"hints must be a dict containing int, float, bool or str value,"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.compile(fn_with_hints, backend=cnt)(x, y)

    def test_hints_wrapper_pytree_inputs(self):
        # [FUNCTION DOC] Tests that hints_wrapper with a nested pytree (tuple holding a tensor and
        # a dict) as args raises RuntimeError saying args must be a tuple of tensors, ints, floats
        # or bools (run eagerly).
        def fn_with_hints(x, y):
            # [FUNCTION DOC] HELPER: Calls hints_wrapper with a pytree argument and a hint.
            def outer_body_fn(x):
                # [FUNCTION DOC] HELPER: Returns the sum of the first element and the test entry
                # of the second.
                res = torch.add(x[0], x[1]["test"])
                return res

            res = hints_wrapper(
                outer_body_fn, ((x, {"test": y}),), {}, hints={"test": True}
            )
            return res

        x = torch.randn(2, 4)
        y = torch.ones(4)

        msg = r"args must be a tuple of tensors, ints, floats, or bools,"
        with self.assertRaisesRegex(RuntimeError, msg):
            fn_with_hints(x, y)

    @requires_gpu_and_triton
    def test_wrap_inductor_compiled_regions_option(self):
        # [FUNCTION DOC] GPU test (needs triton): compiles a matmul with inductor with the option
        # that wraps compiled regions in a higher order op and without it, under DebugMode.
        # Asserts the debug string contains the compiled code marker only for the wrapped version
        # and both results equal eager.
        """
        Test that wrap_inductor_compiled_regions option wraps compiled regions
        in inductor_compiled_code HOP, making them visible to DebugMode.
        """
        from torch.utils._debug_mode import DebugMode

        # Test with wrapping enabled
        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn_wrapped(x, y):
            # [FUNCTION DOC] HELPER: Compiled (inductor, wrap option enabled, fullgraph) matrix
            # multiply.
            return torch.matmul(x, y)

        # Test with wrapping disabled (default)
        @torch.compile(backend="inductor", fullgraph=True)
        def fn_not_wrapped(x, y):
            # [FUNCTION DOC] HELPER: Compiled (inductor, default options, fullgraph) matrix
            # multiply.
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device=GPU_TYPE)
        y = torch.randn(4, 4, device=GPU_TYPE)

        # Test wrapped version - HOP should be visible in DebugMode
        with DebugMode() as debug_mode_wrapped:
            result_wrapped = fn_wrapped(x, y)

        debug_string_wrapped = debug_mode_wrapped.debug_string()
        self.assertIn("inductor_compiled_code", debug_string_wrapped)

        # Test non-wrapped version - HOP should NOT be visible
        with DebugMode() as debug_mode_not_wrapped:
            result_not_wrapped = fn_not_wrapped(x, y)

        debug_string_not_wrapped = debug_mode_not_wrapped.debug_string()
        self.assertNotIn("inductor_compiled_code", debug_string_not_wrapped)

        # Both should produce correct results
        expected = torch.matmul(x, y)
        self.assertEqual(result_wrapped, expected)
        self.assertEqual(result_not_wrapped, expected)

    @requires_gpu_and_triton
    def test_wrap_inductor_compiled_regions_with_backward(self):
        # [FUNCTION DOC] GPU test (needs triton): compiles a matmul with the wrapped compiled
        # region option, runs forward and backward under DebugMode, asserts the marker appears and
        # results and gradients match eager.
        """
        Test that wrap_inductor_compiled_regions works correctly with autograd.
        """
        from torch.utils._debug_mode import DebugMode

        @torch.compile(
            backend="inductor",
            options={"wrap_inductor_compiled_regions": True},
            fullgraph=True,
        )
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled (inductor, wrap option enabled, fullgraph) matrix
            # multiply.
            return torch.matmul(x, y)

        x = torch.randn(4, 4, device=GPU_TYPE, requires_grad=True)
        y = torch.randn(4, 4, device=GPU_TYPE, requires_grad=True)

        # Clone for eager comparison
        x_eager = x.detach().clone().requires_grad_(True)
        y_eager = y.detach().clone().requires_grad_(True)

        # Compiled forward and backward
        with DebugMode() as debug_mode:
            result = fn(x, y)
            loss = result.sum()
            loss.backward()

        # HOP should be visible in forward pass
        self.assertIn("inductor_compiled_code", debug_mode.debug_string())

        # Eager forward and backward for comparison
        expected = torch.matmul(x_eager, y_eager)
        expected_loss = expected.sum()
        expected_loss.backward()

        # Check correctness
        self.assertEqual(result, expected)
        self.assertEqual(x.grad, x_eager.grad)
        self.assertEqual(y.grad, y_eager.grad)


class HigherOrderOpVmapGuardTests(LoggingTestCase):
    # [CLASS DOC] Logging test case checking that functorch transforms (vmap, grad, jvp, forward
    # AD dual level, linearize) interact correctly with dynamo guards. Tests capture recompile or
    # guard log records via the logging decorator and assert whether a recompile happened and what
    # the guard message says about functorch state.
    @make_logging_test(recompiles=True)
    def test_vmap_grad_guard_ok(self, records):
        # [FUNCTION DOC] Tests that compiling vmap of a function applying vmap of grad of sin
        # produces no recompile records, the result equals cos of the input, and a second call
        # also does not recompile. Receives the captured log records.
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            # [FUNCTION DOC] HELPER: Returns vmap of grad of sin applied to x.
            return vmap(grad(torch.sin))(x)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of g applied to x.
            return vmap(g)(x)

        x = torch.randn(4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.cos(), y)

        # Calling the same function again won't have any effect on guards
        fn(x)
        self.assertEqual(len(records), 0)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_grad_guard_fail(self, records):
        # [FUNCTION DOC] Tests that a function compiled with grad inside does not recompile on
        # repeated calls but does recompile (with a log record whose message includes a functorch
        # state comparison against an empty state) when called under an outer grad. Expected
        # failure under torch dynamo.
        grad = torch.func.grad

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning grad of sin at the sum of
            # x.
            return grad(torch.sin)(x.sum())

        x = torch.randn([])
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call grad should retrigger compilation
        x = torch.randn(3)
        grad(fn)(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_dual_level_guard(self, records):
        # [FUNCTION DOC] Tests that forward mode dual level creation inside a compiled function
        # does not recompile on repeat calls, but calling inside an outer dual level raises
        # InternalTorchDynamoError (nested forward AD unsupported) and emits a record with a guard
        # on the current forward AD level being minus one.
        fwAD = torch.autograd.forward_ad

        @torch.compile(backend="eager", fullgraph=True)
        def fn(foo, tangent):
            # [FUNCTION DOC] HELPER: Compiled (eager, fullgraph) function that enters a dual level
            # and returns a dual tensor made from foo and a slice of tangent.
            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, tangent[1:])
                return dual

        foo = torch.rand(2)
        tangent = torch.rand(3)
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # assertRaises is only here because Nested forward mode AD is not supported
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError):
            with fwAD.dual_level():
                fn(foo, tangent)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "forward_ad")
        self.assertIn(
            """torch.autograd.forward_ad._current_level == -1""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_jvp_guard_fail(self, records):
        # [FUNCTION DOC] Tests that a compiled jvp does not recompile on repeat calls, but
        # recompiles when called under outer jvp of vmap. Asserts records exist and that a
        # functorch or forward AD guard message is present (whichever record exists). Expected
        # failure under torch dynamo.
        jvp = torch.func.jvp
        vmap = torch.func.vmap

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning jvp of sin at x with
            # tangent x.
            return jvp(torch.sin, (x,), (x,))

        x = torch.randn(3, 4)
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call jvp should retrigger compilation
        x = torch.randn(3, 4, 5)
        jvp(vmap(fn), (x,), (x,))

        self.assertGreater(len(records), 0)
        if self.hasRecord(records, "pyfunctorch"):
            record = self.getRecord(records, "pyfunctorch")
            self.assertIn(
                """torch._functorch.pyfunctorch.compare_functorch_state([])""",
                munge_exc(record.getMessage()),
            )
        elif self.hasRecord(records, "forward_ad"):
            record = self.getRecord(records, "forward_ad")
            self.assertIn(
                """torch.autograd.forward_ad._current_level == -1""",
                munge_exc(record.getMessage()),
            )

    @make_logging_test(recompiles=True)
    def test_vmap_guard_ok(self, records):
        # [FUNCTION DOC] Tests that compiled vmap of a sin lambda adds no recompile on repeated
        # calls, nor when called with a different tensor object; also checks values.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda
            # applied to x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.randn(3, 3, 4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), y)

        # Calling the same function again won't have any effect on guards
        z = fn(x)
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), z)

        # calling with a different object will also not affect guards
        w = fn(z)
        self.assertEqual(len(records), 0)
        self.assertEqual(z.sin(), w)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail_different_state(self, records):
        # [FUNCTION DOC] Tests that the vmap randomness mode is part of the functorch guard:
        # calling under vmap with randomness same compiles with no recompile, then under vmap with
        # different randomness recompiles with a message that mentions the Vmap state with level
        # one and mode same. Expected failure under torch dynamo.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda
            # applied to x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 4)
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation
        y = torch.vmap(fn, randomness="different")(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail(self, records):
        # [FUNCTION DOC] Tests that an extra enclosing vmap level triggers a recompile: first call
        # under one vmap is fine, then under two nested vmaps recompiles with a message mentioning
        # Vmap at level one with mode error. Expected failure under torch dynamo.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda
            # applied to x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation as
        # _functorch.current_level() is not the same
        x = torch.zeros(3, 3, 3, 4, 5)
        y = torch.vmap(torch.vmap(fn))(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_grad_vmap_guard_fail(self, records):
        # [FUNCTION DOC] Tests that a function compiled with grad over a vmap (randomness same)
        # invalidates its graph when later called under different randomness and extra vmap
        # nesting; asserts records and the Vmap level one with error mode guard message. Expected
        # failure under torch dynamo.
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            # [FUNCTION DOC] HELPER: Applies vmap (randomness same) of sin to x and returns the
            # sum over dimension zero.
            y = vmap(torch.sin, randomness="same")(x)
            return y.sum(0)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning grad of g at x.
            return grad(g)(x)

        x = torch.randn(3, 3)
        y = vmap(fn, randomness="error")(x)
        self.assertEqual(x.cos(), y)

        # previous FX graph should be invalidated
        x = torch.randn(3, 3, 4)
        y = vmap(vmap(fn, randomness="different"))(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_recompile_different_states(self, records):
        # [FUNCTION DOC] Tests that calling a compiled vmap function under outer vmap with
        # randomness same then different triggers a recompile with a guard message mentioning Vmap
        # level one in mode same. Expected failure under torch dynamo.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda
            # applied to x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        self.assertEqual(len(records), 0)  # sanity check

        torch.vmap(fn, randomness="different")(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(guards=True)
    def test_emit_functorch_guard_if_active(self, records):
        # [FUNCTION DOC] Tests that a functorch guard is emitted only when a functorch transform
        # is active: no functorch record when calling plainly, but a record with Vmap level one
        # and mode error when called under vmap. Uses guard logging.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning sin of x.
            return torch.sin(x)

        x = torch.randn(3, 4)
        _ = fn(x)
        self.assertFalse(self.hasRecord(records, "pyfunctorch"))  # sanity check

        _ = torch.vmap(fn)(x)
        self.assertTrue(self.hasRecord(records, "pyfunctorch"))
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_linearize_recompiles(self, records):
        # [FUNCTION DOC] Tests that compiled linearize does not recompile for different inputs of
        # the same shape but recompiles for a different shape (record count greater than zero).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function that linearizes sin at x and
            # returns the output and the linear function applied to x.
            out, jvp_fn = torch.func.linearize(torch.sin, x)
            return out, jvp_fn(x)

        x = torch.randn(2, 3)
        fn(x)
        self.assertEqual(len(records), 0)

        z = torch.randn(2, 3)
        fn(z)
        self.assertEqual(len(records), 0)

        y = torch.randn(3, 4)
        fn(y)
        self.assertGreater(len(records), 0)


class FuncTorchHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for dynamo tracing of functorch transforms as higher order ops (grad,
    # vmap, jvp, jacrev, hessian and so on). Many tests compile a wrapper function, compare with
    # eager and check the exact printed graph. Teardown unwinds any leaked vmap nesting levels.
    # Contains helper methods for compiling and recording graphs.
    def tearDown(self):
        # [FUNCTION DOC] Test teardown that, in torch dynamo test mode, repeatedly pops vmap
        # interpreters from the functorch stack (stopping at the first non vmap one) and emits a
        # warning if any were left. The parent tearDown always runs in a finally clause.
        # Ensure that in the case of a test failure, the next test won't fail
        # because of a previous call to _vmap_increment_nesting that wasn't undone
        # i.e. test_vmap_free_tensor fails when PYTORCH_TEST_WITH_DYNAMO=1
        # and the call to increment nesting is not undone
        try:
            if TEST_WITH_TORCHDYNAMO:
                warn = False
                while ci := torch._C._functorch.peek_interpreter_stack():
                    if ci.key() == torch._C._functorch.TransformType.Vmap:
                        warn = True
                        torch._C._functorch._vmap_decrement_nesting()
                    else:
                        break

                if warn:
                    msg = (
                        "Interpreter stack is not empty. Test should have called "
                        "'torch._C._functorch._vmap_decrement_nesting()'"
                    )
                    warnings.warn(msg)
        finally:
            super().tearDown()

    def test_teardown_resets_nested_graph_breaks(self):
        # [FUNCTION DOC] Tests that the test base class restores the nested graph breaks config:
        # registers a cleanup asserting the flag equals its expected prior state, and asserts the
        # flag is currently enabled.
        expected_nested_state = getattr(
            self, "_prior_nested_graph_breaks", torch._dynamo.config.nested_graph_breaks
        )

        def _check_flag():
            # [FUNCTION DOC] HELPER: Cleanup asserting the nested graph breaks flag equals the
            # expected prior state.
            self.assertEqual(
                torch._dynamo.config.nested_graph_breaks, expected_nested_state
            )

        self.addCleanup(_check_flag)
        # Sanity check: these tests always run with nested graph breaks enabled.
        self.assertTrue(torch._dynamo.config.nested_graph_breaks)

    def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
        # [FUNCTION DOC] HELPER: Runs fn eagerly and compiled with the recording backend (given
        # fullgraph choice), asserts equal results, and returns the recorded graph module at index
        # graph_idx.
        backend = EagerAndRecordGraphs()
        actual = fn(*inputs)
        expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)

        self.assertEqual(actual, expected)

        wrapped_gm = backend.graphs[graph_idx]
        return wrapped_gm

    def test_hessian(self):
        # [FUNCTION DOC] Tests compiling torch.func.hessian of sin; skipped in dynamic capture
        # mode and otherwise compares the exact printed graph.
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns hessian of sin applied to x.
            return torch.func.hessian(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        new_zeros: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 4, 3]" = new_zeros.view(12, 4, 3);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[4, 3]" = torch._functorch.predispatch._make_dual(l_x_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad_1: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(_make_dual, 3);  _make_dual = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[4, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[4, 3]" = torch.sin(_wrap_for_grad_1)

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(sin, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        new_zeros_1: "f32[12, 12]" = _unwrap_for_grad.new_zeros(12, 12);  _unwrap_for_grad = None

        diagonal_1: "f32[12]" = new_zeros_1.diagonal(0)
        fill__1: "f32[12]" = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        view_1: "f32[12, 4, 3]" = new_zeros_1.view(12, 4, 3);  new_zeros_1 = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view_1, 0, 3);  view_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin], [_wrap_for_grad_1], [_add_batch_dim_1], retain_graph = True, create_graph = True);  sin = _wrap_for_grad_1 = _add_batch_dim_1 = None
        getitem: "f32[4, 3]" = _autograd_grad[0];  _autograd_grad = None

        _remove_batch_dim: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(getitem, 3, 12, 0);  getitem = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = torch.functional.split(_remove_batch_dim, [12], dim = 0);  _remove_batch_dim = None
        getitem_1: "f32[12, 4, 3]" = split[0];  split = None

        view_2: "f32[4, 3, 4, 3]" = getitem_1.view((4, 3, 4, 3));  getitem_1 = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(view_2, level = 0);  view_2 = None
        getitem_2: "f32[4, 3, 4, 3]" = _unpack_dual[0]

        _unwrap_for_grad_1: "f32[4, 3, 4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_2, 2);  getitem_2 = _unwrap_for_grad_1 = None

        getitem_3: "f32[4, 3, 4, 3]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_2: "f32[4, 3, 4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_3, 2);  getitem_3 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim_1: "f32[12, 4, 3, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_2, 1, 12, 0);  _unwrap_for_grad_2 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 4, 3, 12]" = _remove_batch_dim_1.movedim(0, -1);  _remove_batch_dim_1 = None

        split_1 = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_4: "f32[4, 3, 4, 3, 12]" = split_1[0];  split_1 = None

        unflatten: "f32[4, 3, 4, 3, 4, 3]" = getitem_4.unflatten(-1, (4, 3));  getitem_4 = None
        return (unflatten,)
""",
        )

    def test_hessian_with_default_device(self):
        # [FUNCTION DOC] Tests that compiling torch.func.hessian of sin works while the default
        # device context is set to cpu; only compares compiled and eager results.
        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns hessian of sin at x.
            return torch.func.hessian(torch.sin)(x)

        x = torch.randn(4, 3)
        with torch.device("cpu"):
            self._compile_check(wrapper_fn, (x,))

    def test_hessian_argnums(self):
        # [FUNCTION DOC] Tests hessian with argnums selecting the second argument; compares
        # compiled and eager and, outside dynamic capture mode, the printed graph (all lines
        # except the last two) and then separately the final return line.
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function returning sin of x, ignoring y.
            return x.sin()

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns hessian of fn with respect to argument index one,
            # applied to x and y.
            return torch.func.hessian(fn, argnums=(1,))(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            "\n".join(actual.split("\n")[:-2]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        new_zeros: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 3, 4]" = new_zeros.view(12, 3, 4);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 4]" = torch._functorch.predispatch._make_dual(l_y_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad_2: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(_make_dual, 3);  _make_dual = None
        _wrap_for_grad_3: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(_wrap_for_grad, 3);  _wrap_for_grad = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_2);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[4, 3]" = _wrap_for_grad_3.sin();  _wrap_for_grad_3 = None

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(sin, 3)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        new_zeros_1: "f32[12, 12]" = _unwrap_for_grad.new_zeros(12, 12);  _unwrap_for_grad = None

        diagonal_1: "f32[12]" = new_zeros_1.diagonal(0)
        fill__1: "f32[12]" = diagonal_1.fill_(1);  diagonal_1 = fill__1 = None

        view_1: "f32[12, 4, 3]" = new_zeros_1.view(12, 4, 3);  new_zeros_1 = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view_1, 0, 3);  view_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin], [_wrap_for_grad_2], [_add_batch_dim_1], retain_graph = True, create_graph = True);  sin = _wrap_for_grad_2 = _add_batch_dim_1 = None
        getitem: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(getitem, 3, 12, 0);  getitem = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = torch.functional.split(_remove_batch_dim, [12], dim = 0);  _remove_batch_dim = None
        getitem_1: "f32[12, 3, 4]" = split[0];  split = None

        view_2: "f32[4, 3, 3, 4]" = getitem_1.view((4, 3, 3, 4));  getitem_1 = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(view_2, level = 0);  view_2 = None
        getitem_2: "f32[4, 3, 3, 4]" = _unpack_dual[0];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[4, 3, 3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem_2, 2);  _unwrap_for_grad_1 = None

        zeros_like: "f32[4, 3, 3, 4]" = torch.zeros_like(getitem_2);  getitem_2 = None

        _unwrap_for_grad_2: "f32[4, 3, 3, 4]" = torch._functorch.predispatch._unwrap_for_grad(zeros_like, 2);  zeros_like = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim_1: "f32[12, 4, 3, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_2, 1, 12, 0);  _unwrap_for_grad_2 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        movedim: "f32[4, 3, 3, 4, 12]" = _remove_batch_dim_1.movedim(0, -1);  _remove_batch_dim_1 = None

        split_1 = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_3: "f32[4, 3, 3, 4, 12]" = split_1[0];  split_1 = None

        unflatten: "f32[4, 3, 3, 4, 3, 4]" = getitem_3.unflatten(-1, (3, 4));  getitem_3 = None""",
        )

        self.assertExpectedInline(
            actual.split("\n")[-2],
            """        return (unflatten,)""",
        )

    def test_jacrev(self):
        # [FUNCTION DOC] Tests compiling jacrev of sin; checks compiled equals eager and, outside
        # dynamic capture mode, the exact printed graph.
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns jacrev of sin at x.
            return torch.func.jacrev(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[4, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[4, 3]" = torch.sin(_wrap_for_grad)

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(sin, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        new_zeros: "f32[12, 12]" = _unwrap_for_grad.new_zeros(12, 12);  _unwrap_for_grad = None

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 4, 3]" = new_zeros.view(12, 4, 3);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin], [_wrap_for_grad], [_add_batch_dim], retain_graph = True, create_graph = True);  sin = _wrap_for_grad = _add_batch_dim = None
        getitem: "f32[4, 3]" = _autograd_grad[0];  _autograd_grad = None

        _remove_batch_dim: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(getitem, 1, 12, 0);  getitem = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = torch.functional.split(_remove_batch_dim, [12], dim = 0);  _remove_batch_dim = None
        getitem_1: "f32[12, 4, 3]" = split[0];  split = None

        view_1: "f32[4, 3, 4, 3]" = getitem_1.view((4, 3, 4, 3));  getitem_1 = None
        return (view_1,)
""",
        )

    def test_jacrev_two_tensors_argnums(self):
        # [FUNCTION DOC] Tests jacrev with argnums one on a two tensor function; compares eager
        # and compiled and, outside dynamic capture mode, the printed graph after normalizing
        # empty lines.
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns sin of y, ignoring x.
            return y.sin()

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns jacrev of fn with respect to argument one at x and y.
            return torch.func.jacrev(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 4]" = _wrap_for_grad_1.sin()

        _unwrap_for_grad: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(sin, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        new_zeros: "f32[12, 12]" = _unwrap_for_grad.new_zeros(12, 12);  _unwrap_for_grad = None

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 3, 4]" = new_zeros.view(12, 3, 4);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin], [_wrap_for_grad_1], [_add_batch_dim], retain_graph = True, create_graph = True);  sin = _wrap_for_grad_1 = _add_batch_dim = None
        getitem: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(getitem, 1, 12, 0);  getitem = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = torch.functional.split(_remove_batch_dim, [12], dim = 0);  _remove_batch_dim = None
        getitem_1: "f32[12, 3, 4]" = split[0];  split = None

        view_1: "f32[3, 4, 3, 4]" = getitem_1.view((3, 4, 3, 4));  getitem_1 = None
        return (view_1,)
"""
        self.assertExpectedInline(
            empty_line_normalizer(actual),
            empty_line_normalizer(normalize_gm(expected)),
        )

    def test_jacrev_has_aux(self):
        # [FUNCTION DOC] Tests jacrev with argnums one and has_aux set; checks results match eager
        # and, outside dynamic capture mode, the exact printed graph.
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of sin of y and x (the second is the auxiliary
            # output).
            return y.sin(), x

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns jacrev of fn with respect to argument one, with
            # auxiliary output, at x and y.
            return torch.func.jacrev(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 4]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(_wrap_for_grad, 1);  _wrap_for_grad = None

        sin: "f32[3, 4]" = _wrap_for_grad_1.sin()

        _unwrap_for_grad_1: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(sin, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        new_zeros: "f32[12, 12]" = _unwrap_for_grad_1.new_zeros(12, 12);  _unwrap_for_grad_1 = None

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 3, 4]" = new_zeros.view(12, 3, 4);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin], [_wrap_for_grad_1], [_add_batch_dim], retain_graph = True, create_graph = True);  sin = _wrap_for_grad_1 = _add_batch_dim = None
        getitem: "f32[3, 4]" = _autograd_grad[0];  _autograd_grad = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(getitem, 1, 12, 0);  getitem = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        split = torch.functional.split(_remove_batch_dim, [12], dim = 0);  _remove_batch_dim = None
        getitem_1: "f32[12, 3, 4]" = split[0];  split = None

        view_1: "f32[3, 4, 3, 4]" = getitem_1.view((3, 4, 3, 4));  getitem_1 = None
        return (view_1, _unwrap_for_grad)
""",
        )

    def test_vjp(self):
        # [FUNCTION DOC] Tests compiling vjp of a sin then sum function, returning only the primal
        # output; compares with eager and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Calls vjp of fn at x, discards the vjp function and returns
            # the primal output; v is unused.
            (out, vjpfunc) = torch.func.vjp(fn, x)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[5]" = _wrap_for_grad.sin();  _wrap_for_grad = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad,)
""",
        )

    def test_vjp_multiple_outputs(self):
        # [FUNCTION DOC] Tests vjp of a function with two outputs (sin and cos), applying the vjp
        # function to a pair of cotangents; compares with eager and the printed graph (outside
        # dynamic capture mode).
        counters.clear()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Calls vjp of a two output lambda at x, applies the returned
            # vjp function to the pair (v, v), and returns outputs and vjps.
            fn = lambda x: (x.sin(), x.cos())  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc((v, v))
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[5]", L_x_: "f32[5]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        cos: "f32[5]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[5]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1)

        sin: "f32[5]" = _wrap_for_grad.sin();  _wrap_for_grad = None

        _unwrap_for_grad_1: "f32[5]" = torch._functorch.predispatch._unwrap_for_grad(sin, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin, cos], [_set_tensor_requires_grad], [l_v_, l_v_], retain_graph = True, create_graph = True);  sin = cos = _set_tensor_requires_grad = l_v_ = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad_1, _unwrap_for_grad, getitem)
""",
        )

    def test_vjp_multiple_outputs_python_struct(self):
        # [FUNCTION DOC] Tests vjp of a function returning a dict with first and second entries;
        # the vjp function gets a dict of cotangents. Compares with eager and the printed graph
        # (outside dynamic capture mode).
        counters.clear()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Calls vjp of a dict returning lambda at x, applies the vjp
            # function to a dict of cotangents, returns outputs and vjps.
            fn = lambda x: {"first": x.sin(), "second": x.cos()}  # noqa: E731
            (out, vjpfunc) = torch.func.vjp(fn, x)
            vjps = vjpfunc({"first": v, "second": v.sin()})
            return out, vjps

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[5]", L_x_: "f32[5]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad)

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        cos: "f32[5]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[5]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1)

        sin: "f32[5]" = _wrap_for_grad.sin();  _wrap_for_grad = None

        _unwrap_for_grad_1: "f32[5]" = torch._functorch.predispatch._unwrap_for_grad(sin, 1)

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None

        sin_1: "f32[5]" = l_v_.sin()

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad([sin, cos], [_set_tensor_requires_grad], [l_v_, sin_1], retain_graph = True, create_graph = True);  sin = cos = _set_tensor_requires_grad = l_v_ = sin_1 = None
        getitem: "f32[5]" = _autograd_grad[0];  _autograd_grad = None
        return (_unwrap_for_grad_1, _unwrap_for_grad, getitem)
""",
        )

    def test_vjp_has_aux(self):
        # [FUNCTION DOC] Tests vjp with has_aux set; the wrapper returns only the primal output.
        # Compares with eager and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x and x (aux).
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Calls vjp of fn at x with has_aux set and returns the primal
            # output.
            (out, vjpfunc, _) = torch.func.vjp(fn, x, has_aux=True)
            return out

        x = torch.randn([5])
        v = torch.randn(5)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[5]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[5]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        _unwrap_for_grad: "f32[5]" = torch._functorch.predispatch._unwrap_for_grad(_wrap_for_grad, 1);  _unwrap_for_grad = None

        sin: "f32[5]" = _wrap_for_grad.sin();  _wrap_for_grad = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_functional_call(self):
        # [FUNCTION DOC] Tests torch.func.functional_call with a linear model and a parameter
        # mapping, followed by a mean squared error loss; compares eager and compiled and, outside
        # dynamic capture mode, the printed graph.
        def wrapper_fn(model, params, inputs, targets):
            # [FUNCTION DOC] HELPER: Runs the model through functional_call with the given params
            # on the inputs and returns mse loss against the targets.
            prediction = torch.func.functional_call(model, params, (inputs,))
            return torch.nn.functional.mse_loss(prediction, targets)

        model = torch.nn.Linear(3, 3)
        params = dict(model.named_parameters())
        inputs = torch.randn(64, 3)
        targets = torch.randn(64, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (model, params, inputs, targets))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_: "f32[64, 3]", L_model_parameters_bias_: "f32[3]", L_model_parameters_weight_: "f32[3, 3]", L_targets_: "f32[64, 3]"):
        l_inputs_ = L_inputs_
        l_model_parameters_bias_ = L_model_parameters_bias_
        l_model_parameters_weight_ = L_model_parameters_weight_
        l_targets_ = L_targets_

        linear: "f32[64, 3]" = torch._C._nn.linear(l_inputs_, l_model_parameters_weight_, l_model_parameters_bias_);  l_inputs_ = l_model_parameters_weight_ = l_model_parameters_bias_ = None

        mse_loss: "f32[]" = torch.nn.functional.mse_loss(linear, l_targets_);  linear = l_targets_ = None
        return (mse_loss,)
""",
        )

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_SLOW, "https://github.com/pytorch/pytorch/issues/177056"
    )
    def test_functional_call_sequential_params_and_buffers(self):
        # [FUNCTION DOC] Tests functional_call with two separate dicts (params, buffers), compiled
        # without fullgraph. Skipped on Linux or slow runs (see linked issue). Compares to eager
        # and, outside dynamic capture mode, the printed graph after empty line normalization
        # (there are empty line differences between platforms).
        # copied from test/test_stateless.py
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module (copied from the stateless tests) with a one by one
            # linear layer, a registered buffer of ones, and a float attribute foo; forward
            # returns the layer output plus the buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the layer, buffer and the foo attribute.
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.register_buffer("buffer", torch.ones(1))
                self.foo = 0.0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to x plus the buffer.
                return self.l1(x) + self.buffer

        def wrapper_fn(model, params, buffers, inputs):
            # [FUNCTION DOC] HELPER: Calls functional_call with the model, a tuple of the two
            # dictionaries, and the inputs.
            # two separate dictionaries
            return torch.func.functional_call(model, (params, buffers), inputs)

        model = MockModule()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        inputs = torch.tensor([[1.5]])

        wrapped_gm = self._compile_check(
            wrapper_fn, (model, params, buffers, inputs), fullgraph=False
        )
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        expected = """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_: "f32[1, 1]", L_model_buffers_buffer_: "f32[1]", L_model_modules_l1_parameters_bias_: "f32[1]", L_model_modules_l1_parameters_weight_: "f32[1, 1]"):
        l_inputs_ = L_inputs_
        l_model_buffers_buffer_ = L_model_buffers_buffer_
        l_model_modules_l1_parameters_bias_ = L_model_modules_l1_parameters_bias_
        l_model_modules_l1_parameters_weight_ = L_model_modules_l1_parameters_weight_
        linear: "f32[1, 1]" = torch._C._nn.linear(l_inputs_, l_model_modules_l1_parameters_weight_, l_model_modules_l1_parameters_bias_);  l_inputs_ = l_model_modules_l1_parameters_weight_ = l_model_modules_l1_parameters_bias_ = None
        add: "f32[1, 1]" = linear + l_model_buffers_buffer_;  linear = l_model_buffers_buffer_ = None
        return (add,)
"""
        # We found Windows/Linux have some empty line difference, empty_line_normalizer will help fix it.
        self.assertExpectedInline(
            empty_line_normalizer(actual),
            empty_line_normalizer(normalize_gm(expected)),
        )

    def test_grad(self):
        # [FUNCTION DOC] Tests compiling grad of a function computing the sum of sin; compares
        # with eager and, outside dynamic capture mode, the printed graph.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        sum_1: "f32[]" = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_grad_freevar_tensor(self):
        # [FUNCTION DOC] Tests grad of a function that adds a free tensor variable before summing,
        # compiled with the aot_eager backend and fullgraph; result must equal eager.
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus the free tensor y.
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_grad_freevar_python_scalar(self):
        # [FUNCTION DOC] Tests grad of a function using a free Python int variable; compares with
        # eager and, outside dynamic capture mode, the printed graph.
        counters.clear()
        y = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus the free int y.
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + 3;  sin = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_grad_capture_tensor(self):
        # [FUNCTION DOC] Tests grad of a function closing over a tensor created inside the
        # wrapper; compares with eager and, outside dynamic capture mode, the printed graph.
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Creates a random vector y, defines fn adding it, and returns
            # grad of fn at x.
            y = torch.randn(3)

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus y.
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        randn: "f32[3]" = torch.randn(3)

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + randn;  sin = randn = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_grad_closure_scalar(self):
        # [FUNCTION DOC] Tests grad of a function closing over a float created inside the wrapper;
        # compiled without fullgraph because dynamo cannot get the source of the inner function.
        # Compares with eager and, outside dynamic capture mode, the printed graph.
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Sets y to 3.14, defines fn adding it, and returns grad of fn
            # at x.
            y = 3.14

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus y.
                return (x.sin() + y).sum()

            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)

        # Graph break because dynamo is unable to get source `fn` and
        # functools.wraps in `grad` leads to graph-break
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + 3.14;  sin = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_grad_has_aux(self):
        # [FUNCTION DOC] Tests grad with has_aux set on a function returning a scalar loss and an
        # auxiliary cos output; compares with eager and the printed graph (outside dynamic capture
        # mode).
        counters.clear()

        y = 3.14

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x plus the free float y,
            # and cos of x.
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad with auxiliary output of fn at x.
            return torch.func.grad(fn, has_aux=True)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        cos: "f32[3, 3, 3]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1);  cos = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + 3.14;  sin = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad_1 = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_2, _unwrap_for_grad)
""",
        )

    def test_grad_two_tensor_has_aux(self):
        # [FUNCTION DOC] Tests grad with has_aux on a two tensor function differentiating the
        # first argument; compares with eager and the printed graph (outside dynamic capture
        # mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x plus y, and cos of x.
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns grad with auxiliary output of fn at x and y.
            return torch.func.grad(fn, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        cos: "f32[3, 3, 3]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1);  cos = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + _wrap_for_grad_1;  sin = _wrap_for_grad_1 = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad_1 = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_2, _unwrap_for_grad)
""",
        )

    def test_grad_two_tensor_all_grad_has_aux(self):
        # [FUNCTION DOC] Tests grad over both arguments with has_aux, where the argnums are given
        # either as a literal tuple or as a captured tuple variable; both versions are compiled
        # and their printed graphs compared to expected text (outside dynamic capture mode).
        counters.clear()

        nums = (0, 1)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x plus y, and cos of x.
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn_const_var(x, y):
            # [FUNCTION DOC] HELPER: Returns grad of fn over arguments zero and one with auxiliary
            # output, using a literal tuple.
            return torch.func.grad(fn, argnums=(0, 1), has_aux=True)(x, y)

        def wrapper_fn_tuple_var(x, y):
            # [FUNCTION DOC] HELPER: Same as above but argnums comes from the captured tuple nums.
            return torch.func.grad(fn, argnums=nums, has_aux=True)(x, y)

        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm_const_var = self._compile_check(wrapper_fn_const_var, (x, y))
        wrapped_gm_tuple_var = self._compile_check(wrapper_fn_tuple_var, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual_const_var = normalize_gm(
            wrapped_gm_const_var.print_readable(print_output=False)
        )
        actual_tuple_var = normalize_gm(
            wrapped_gm_tuple_var.print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_const_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        cos: "f32[3, 3, 3]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1);  cos = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + _wrap_for_grad_1;  sin = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad, _wrap_for_grad_1], create_graph = True);  _wrap_for_grad = _wrap_for_grad_1 = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad_1 = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0]

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[3, 3, 3]" = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad_3: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_2, _unwrap_for_grad_3, _unwrap_for_grad)
""",
        )
        self.assertExpectedInline(
            actual_tuple_var,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None
        _wrap_for_grad_1: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_y_, 1);  l_y_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        cos: "f32[3, 3, 3]" = _wrap_for_grad.cos()

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(cos, 1);  cos = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        add: "f32[3, 3, 3]" = sin + _wrap_for_grad_1;  sin = None
        sum_1: "f32[]" = add.sum();  add = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad, _wrap_for_grad_1], create_graph = True);  _wrap_for_grad = _wrap_for_grad_1 = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 1);  sum_1 = _unwrap_for_grad_1 = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0]

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[3, 3, 3]" = _autograd_grad[1];  _autograd_grad = None

        _unwrap_for_grad_3: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_2, _unwrap_for_grad_3, _unwrap_for_grad)
""",
        )

    def test_grad_over_grad(self):
        # [FUNCTION DOC] Tests grad applied twice to a scalar function; compiled without
        # fullgraph, compared with eager and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of grad of fn at x.
            return torch.func.grad(torch.func.grad(fn))(x)

        x = torch.randn(())
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)

        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None
        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_1 = None
        _grad_increment_nesting_1 = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting_1 = None

        _wrap_for_grad_1: "f32[]" = torch._C._functorch._wrap_for_grad(_wrap_for_grad, 2)

        set_inplace_requires_grad_allowed_2 = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed_2 = None

        _set_tensor_requires_grad_1: "f32[]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad_1);  _set_tensor_requires_grad_1 = None

        set_inplace_requires_grad_allowed_3 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_3 = None

        sin: "f32[]" = _wrap_for_grad_1.sin()
        sum_1: "f32[]" = sin.sum();  sin = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((sum_1,), [_wrap_for_grad_1], create_graph = True);  _wrap_for_grad_1 = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(sum_1, 2);  sum_1 = _unwrap_for_grad = None

        getitem: "f32[]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable_2 = None

        _autograd_grad_1 = torch._functorch.eager_transforms._autograd_grad((_unwrap_for_grad_1,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad_2: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(_unwrap_for_grad_1, 1);  _unwrap_for_grad_1 = _unwrap_for_grad_2 = None

        getitem_1: "f32[]" = _autograd_grad_1[0];  _autograd_grad_1 = None

        _unwrap_for_grad_3: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _grad_decrement_nesting_1 = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting_1 = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_3,)
""",
        )

    def test_grad_with_graph_break(self):
        # [FUNCTION DOC] Tests that a graph break inside the function differentiated by grad falls
        # back: output equals eager and exactly one graph break was counted.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Triggers a dynamo graph break then returns the sum of sin of
            # x.
            torch._dynamo.graph_break()
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(actual, expected)

    def test_grad_with_side_effect(self):
        # [FUNCTION DOC] Tests that a list append side effect on a captured list inside the grad
        # function does not graph break; output equals eager and zero graph breaks recorded.
        counters.clear()

        foo = [1, 2]

        def fn(x):
            # [FUNCTION DOC] HELPER: Appends three to the captured list foo and returns the sum of
            # sin of x.
            foo.append(3)
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_pytree(self):
        # [FUNCTION DOC] Tests grad over a tuple input (tensor and scalar tensor); output equals
        # eager and zero graph breaks recorded.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Unpacks x into x1 and x2 and returns the sum of sin of x1
            # plus x2.
            x1, x2 = x
            return x1.sin().sum() + x2

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns grad of fn at the tuple x.
            return torch.func.grad(fn)(x)

        x1 = torch.randn(3, 3, 3)
        x2 = torch.randn(())
        actual = wrapper_fn((x1, x2))
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(
            (x1, x2)
        )
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_grad_non_tensor_input(self):
        # [FUNCTION DOC] Tests grad where the second argument is a Python float that is not
        # differentiated; compares with eager and the printed graph (outside dynamic capture
        # mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus y.
            return x.sin().sum() + y

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x and y.
            return torch.func.grad(fn)(x, y)

        x = torch.randn(3, 3, 3)
        y = 3.0
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable("torch.func.{grad, vjp, jacrev, hessian} don't yet support saved tensor hooks. Please open an issue with your use case.");  _saved_tensors_hooks_disable = None
        _grad_increment_nesting = torch._C._functorch._grad_increment_nesting();  _grad_increment_nesting = None

        _wrap_for_grad: "f32[3, 3, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 1);  l_x_ = None

        set_inplace_requires_grad_allowed = torch._C._functorch.set_inplace_requires_grad_allowed(True);  set_inplace_requires_grad_allowed = None

        _set_tensor_requires_grad: "f32[3, 3, 3]" = torch._functorch.eager_transforms._set_tensor_requires_grad(_wrap_for_grad);  _set_tensor_requires_grad = None

        set_inplace_requires_grad_allowed_1 = torch._C._functorch.set_inplace_requires_grad_allowed(False);  set_inplace_requires_grad_allowed_1 = None

        sin: "f32[3, 3, 3]" = _wrap_for_grad.sin()
        sum_1: "f32[]" = sin.sum();  sin = None
        add: "f32[]" = sum_1 + 3.0;  sum_1 = None

        _autograd_grad = torch._functorch.eager_transforms._autograd_grad((add,), [_wrap_for_grad], create_graph = True);  _wrap_for_grad = None

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(add, 1);  add = _unwrap_for_grad = None

        getitem: "f32[3, 3, 3]" = _autograd_grad[0];  _autograd_grad = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        _grad_decrement_nesting = torch._C._functorch._grad_decrement_nesting();  _grad_decrement_nesting = None
        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (_unwrap_for_grad_1,)
""",
        )

    def test_grad_fn_with_kwargs(self):
        # [FUNCTION DOC] Tests grad where the function is called with a keyword argument; output
        # equals eager and zero graph breaks recorded.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x plus y.
            return (x + y).sum()

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns grad of fn at x with y passed as a keyword.
            return torch.func.grad(fn)(x, y=y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_jacfwd(self):
        # [FUNCTION DOC] Tests compiling jacfwd of sin; compares with eager and the printed graph
        # (outside dynamic capture mode).
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns jacfwd of sin at x.
            return torch.func.jacfwd(torch.sin)(x)

        x = torch.randn(4, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]"):
        l_x_ = L_x_

        new_zeros: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 4, 3]" = new_zeros.view(12, 4, 3);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[4, 3]" = torch._functorch.predispatch._make_dual(l_x_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None

        sin: "f32[4, 3]" = torch.sin(_make_dual);  _make_dual = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sin, level = 0);  sin = None
        getitem: "f32[4, 3]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = _unwrap_for_grad = None

        getitem_1: "f32[4, 3]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 2);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_1, 1, 12, 0);  _unwrap_for_grad_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[4, 3, 12]" = _remove_batch_dim.movedim(0, -1);  _remove_batch_dim = None

        split = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_2: "f32[4, 3, 12]" = split[0];  split = None

        unflatten: "f32[4, 3, 4, 3]" = getitem_2.unflatten(-1, (4, 3));  getitem_2 = None
        return (unflatten,)
""",
        )

    def test_jacfwd_two_tensors_argnums(self):
        # [FUNCTION DOC] Tests jacfwd with argnums one on a two tensor function; compares with
        # eager and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns sin of y, ignoring x.
            return y.sin()

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns jacfwd of fn with respect to argument one at x and y.
            return torch.func.jacfwd(fn, argnums=1)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        new_zeros: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 3, 4]" = new_zeros.view(12, 3, 4);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 4]" = torch._functorch.predispatch._make_dual(l_y_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        sin: "f32[3, 4]" = _make_dual.sin();  _make_dual = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sin, level = 0);  sin = None
        getitem: "f32[3, 4]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = _unwrap_for_grad = None

        getitem_1: "f32[3, 4]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 2);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_1, 1, 12, 0);  _unwrap_for_grad_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = _remove_batch_dim.movedim(0, -1);  _remove_batch_dim = None

        split = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_2: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = getitem_2.unflatten(-1, (3, 4));  getitem_2 = None
        return (unflatten,)
""",
        )

    def test_jacfwd_has_aux(self):
        # [FUNCTION DOC] Tests jacfwd with argnums one and has_aux; compares with eager and the
        # printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of sin of y and x.
            return y.sin(), x

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns jacfwd of fn with respect to argument one with
            # auxiliary output.
            return torch.func.jacfwd(fn, argnums=1, has_aux=True)(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        new_zeros: "f32[12, 12]" = l_y_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 3, 4]" = new_zeros.view(12, 3, 4);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 4]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 4]" = torch._functorch.predispatch._make_dual(l_y_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = None

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(_wrap_for_grad, 2);  _wrap_for_grad = None

        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = _wrap_for_grad_1 = None

        sin: "f32[3, 4]" = _make_dual.sin();  _make_dual = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sin, level = 0);  sin = None
        getitem: "f32[3, 4]" = _unpack_dual[0]

        _unwrap_for_grad_1: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = _unwrap_for_grad_1 = None

        getitem_1: "f32[3, 4]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_2: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 2);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_2, 1, 12, 0);  _unwrap_for_grad_2 = None
        _remove_batch_dim_1: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad, 1, 12, 0);  _unwrap_for_grad = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        getitem_2: "f32[4, 3]" = _remove_batch_dim_1[0];  _remove_batch_dim_1 = None

        movedim: "f32[3, 4, 12]" = _remove_batch_dim.movedim(0, -1);  _remove_batch_dim = None

        split = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_3: "f32[3, 4, 12]" = split[0];  split = None

        unflatten: "f32[3, 4, 3, 4]" = getitem_3.unflatten(-1, (3, 4));  getitem_3 = None
        return (unflatten, getitem_2)
""",
        )

    def test_jacfwd_randomness(self):
        # [FUNCTION DOC] Tests jacfwd called with randomness same; compares with eager and the
        # printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of sin of y and x.
            return y.sin(), x

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns jacfwd of fn with randomness same at x and y.
            return torch.func.jacfwd(fn, randomness="same")(x, y)

        x = torch.randn(4, 3)
        y = torch.randn(3, 4)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 3]", L_y_: "f32[3, 4]"):
        l_x_ = L_x_
        l_y_ = L_y_

        new_zeros: "f32[12, 12]" = l_x_.new_zeros(12, 12)

        diagonal: "f32[12]" = new_zeros.diagonal(0)
        fill_: "f32[12]" = diagonal.fill_(1);  diagonal = fill_ = None

        view: "f32[12, 4, 3]" = new_zeros.view(12, 4, 3);  new_zeros = None

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(12, 'same');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(view, 0, 1);  view = None

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[4, 3]" = torch._functorch.predispatch._make_dual(l_x_, _add_batch_dim, level = 0);  _add_batch_dim = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(_make_dual, level = 0);  _make_dual = None

        _wrap_for_grad: "f32[4, 3]" = torch._C._functorch._wrap_for_grad(l_x_, 2);  l_x_ = _wrap_for_grad = None
        _wrap_for_grad_1: "f32[3, 4]" = torch._C._functorch._wrap_for_grad(l_y_, 2);  l_y_ = None

        getitem: "f32[4, 3]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = _unwrap_for_grad = None

        getitem_1: "f32[4, 3]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[4, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 2);  getitem_1 = None

        sin: "f32[3, 4]" = _wrap_for_grad_1.sin();  _wrap_for_grad_1 = None

        _unpack_dual_1 = torch._functorch.predispatch._unpack_dual(sin, level = 0);  sin = None
        getitem_2: "f32[3, 4]" = _unpack_dual_1[0];  _unpack_dual_1 = None

        _unwrap_for_grad_2: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(getitem_2, 2);  _unwrap_for_grad_2 = None

        zeros_like: "f32[3, 4]" = torch.zeros_like(getitem_2);  getitem_2 = None

        _unwrap_for_grad_3: "f32[3, 4]" = torch._functorch.predispatch._unwrap_for_grad(zeros_like, 2);  zeros_like = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _remove_batch_dim: "f32[12, 3, 4]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_3, 1, 12, 0);  _unwrap_for_grad_3 = None
        _remove_batch_dim_1: "f32[12, 4, 3]" = torch._functorch.predispatch._remove_batch_dim(_unwrap_for_grad_1, 1, 12, 0);  _unwrap_for_grad_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        movedim: "f32[3, 4, 12]" = _remove_batch_dim.movedim(0, -1);  _remove_batch_dim = None
        movedim_1: "f32[4, 3, 12]" = _remove_batch_dim_1.movedim(0, -1);  _remove_batch_dim_1 = None

        split = torch.functional.split(movedim, [12], dim = -1);  movedim = None
        getitem_3: "f32[3, 4, 12]" = split[0];  split = None
        split_1 = torch.functional.split(movedim_1, [12], dim = -1);  movedim_1 = None
        getitem_4: "f32[4, 3, 12]" = split_1[0];  split_1 = None

        unflatten: "f32[3, 4, 4, 3]" = getitem_3.unflatten(-1, (4, 3));  getitem_3 = None
        unflatten_1: "f32[4, 3, 4, 3]" = getitem_4.unflatten(-1, (4, 3));  getitem_4 = None
        return (unflatten, unflatten_1)
""",
        )

    def test_jvp_simple(self):
        # [FUNCTION DOC] Tests jvp of a sum of sin function with a tangent; compares with eager
        # and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at primal x and tangent v.
            return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sum_1, level = 0);  sum_1 = None
        getitem: "f32[]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (_unwrap_for_grad, _unwrap_for_grad_1)
""",
        )

    def test_jvp_has_aux(self):
        # [FUNCTION DOC] Tests jvp with has_aux; compares with eager and the printed graph
        # (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x and x.
            return x.sin().sum(), x

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at x with tangent v and has_aux set.
            return torch.func.jvp(fn, (x,), (v,), has_aux=True)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        _unwrap_for_grad: "f32[3, 3]" = torch._functorch.predispatch._unwrap_for_grad(_make_dual, 1)

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sum_1, level = 0);  sum_1 = None
        getitem: "f32[]" = _unpack_dual[0]

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_2: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (_unwrap_for_grad_1, _unwrap_for_grad_2, _unwrap_for_grad)
""",
        )

    def test_jvp_two_tensors_has_aux(self):
        # [FUNCTION DOC] Tests jvp with two primals and tangents and has_aux; compares with eager
        # and the printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum of sin of x plus cos of y, and x.
            return (x.sin().sum() + y.cos()), x

        def wrapper_fn(x, y, v):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at x and y with both tangents being v and
            # has_aux set.
            return torch.func.jvp(fn, (x, y), (v, v), has_aux=True)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[3, 3]", L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_v_ = L_v_
        l_x_ = L_x_
        l_y_ = L_y_

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_v_, level = 0);  l_x_ = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_y_, l_v_, level = 0);  l_y_ = l_v_ = None

        _unwrap_for_grad: "f32[3, 3]" = torch._functorch.predispatch._unwrap_for_grad(_make_dual, 1)

        cos: "f32[3, 3]" = _make_dual_1.cos();  _make_dual_1 = None
        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        sum_1: "f32[]" = sin.sum();  sin = None
        add: "f32[3, 3]" = sum_1 + cos;  sum_1 = cos = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(add, level = 0);  add = None
        getitem: "f32[3, 3]" = _unpack_dual[0]

        _unwrap_for_grad_1: "f32[3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[3, 3]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_2: "f32[3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        return (_unwrap_for_grad_1, _unwrap_for_grad_2, _unwrap_for_grad)
""",
        )

    def test_jvp_two_tensors_disable_grad(self):
        # [FUNCTION DOC] Tests jvp invoked while forward mode gradients are disabled with the
        # private context manager; compares with eager and the printed graph (outside dynamic
        # capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Inside a context disabling forward gradients, returns jvp of
            # fn at x with tangent v.
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):
                return torch.func.jvp(fn, (x,), (v,))

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None
        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sum_1, level = 0);  sum_1 = None
        getitem: "f32[]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        return (_unwrap_for_grad, _unwrap_for_grad_1)
""",
        )

    def test_jvp_two_tensors_disable_enable_disable_grad(self):
        # [FUNCTION DOC] Tests jvp under nested forward gradient mode toggles (disabled, enabled,
        # disabled) to check the mode is restored after tracing; compares with eager and the
        # printed graph (outside dynamic capture mode).
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def wrapper_fn(x, v):
            # [FUNCTION DOC] HELPER: Enters three nested contexts that set forward gradients to
            # false, true, false, and returns jvp of fn at x with tangent v.
            with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (1)
                with torch.autograd.forward_ad._set_fwd_grad_enabled(True):  # (2)
                    with torch.autograd.forward_ad._set_fwd_grad_enabled(False):  # (3)
                        return torch.func.jvp(fn, (x,), (v,))  # (4)

            # Start True
            # False      (1)
            #   True     (2)
            #     False  (3)
            #       True (4)
            #     True   (undo 3)
            #   False    (undo 2)
            # True       (undo 1)

        x = torch.randn(3, 3)
        v = torch.randn(3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, v))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_v_: "f32[3, 3]", L_x_: "f32[3, 3]"):
        l_v_ = L_v_
        l_x_ = L_x_

        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None
        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_2 = None
        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_v_, level = 0);  l_x_ = l_v_ = None

        sin: "f32[3, 3]" = _make_dual.sin();  _make_dual = None
        sum_1: "f32[]" = sin.sum();  sin = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sum_1, level = 0);  sum_1 = None
        getitem: "f32[]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 1);  getitem = None

        getitem_1: "f32[]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 1);  getitem_1 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_4 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_4 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None
        _set_fwd_grad_enabled_5 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_5 = None
        _set_fwd_grad_enabled_6 = torch._C._set_fwd_grad_enabled(False);  _set_fwd_grad_enabled_6 = None
        _set_fwd_grad_enabled_7 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_7 = None
        return (_unwrap_for_grad, _unwrap_for_grad_1)
""",
        )

    def test_jvp_freevar_tensor(self):
        # [FUNCTION DOC] Tests jvp of a function adding a free tensor, compiled with aot_eager and
        # fullgraph; result must equal eager.
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus the free tensor y.
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at x with tangent x.
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_jvp_jvp(self):
        # [FUNCTION DOC] Tests jvp applied twice (nested); skips itself in dynamic shape capture
        # mode. Compares with eager and the printed graph.
        counters.clear()

        if check_dynamic_shape_capture():
            self.skipTest("test fails with dynamic shapes")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of sin at x with tangent x.
            return torch.func.jvp(torch.sin, (x,), (x,))

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at x with tangent x.
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        _jvp_increment_nesting = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting = None
        _set_fwd_grad_enabled = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled = None
        _enter_dual_level = torch._functorch.predispatch._enter_dual_level();  _enter_dual_level = None

        _maybe_load_decompositions = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions = None

        _make_dual: "f32[3, 3, 3]" = torch._functorch.predispatch._make_dual(l_x_, l_x_, level = 0);  l_x_ = None

        _jvp_increment_nesting_1 = torch._functorch.predispatch._jvp_increment_nesting();  _jvp_increment_nesting_1 = None
        _set_fwd_grad_enabled_1 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_1 = None

        _maybe_load_decompositions_1 = torch.autograd.forward_ad._maybe_load_decompositions();  _maybe_load_decompositions_1 = None

        _make_dual_1: "f32[3, 3, 3]" = torch._functorch.predispatch._make_dual(_make_dual, _make_dual, level = 0);  _make_dual = None

        sin: "f32[3, 3, 3]" = torch.sin(_make_dual_1);  _make_dual_1 = None

        _unpack_dual = torch._functorch.predispatch._unpack_dual(sin, level = 0);  sin = None
        getitem: "f32[3, 3, 3]" = _unpack_dual[0]

        _unwrap_for_grad: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem, 2);  getitem = None

        getitem_1: "f32[3, 3, 3]" = _unpack_dual[1];  _unpack_dual = None

        _unwrap_for_grad_1: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_1, 2);  getitem_1 = None

        _set_fwd_grad_enabled_2 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_2 = None
        _jvp_decrement_nesting = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting = None

        _unpack_dual_1 = torch._functorch.predispatch._unpack_dual(_unwrap_for_grad, level = 0);  _unwrap_for_grad = None
        _unpack_dual_2 = torch._functorch.predispatch._unpack_dual(_unwrap_for_grad_1, level = 0);  _unwrap_for_grad_1 = None
        getitem_2: "f32[3, 3, 3]" = _unpack_dual_1[0]

        _unwrap_for_grad_2: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_2, 1);  getitem_2 = None

        getitem_3: "f32[3, 3, 3]" = _unpack_dual_1[1];  _unpack_dual_1 = None

        _unwrap_for_grad_3: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_3, 1);  getitem_3 = None

        getitem_4: "f32[3, 3, 3]" = _unpack_dual_2[0]

        _unwrap_for_grad_4: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_4, 1);  getitem_4 = None

        getitem_5: "f32[3, 3, 3]" = _unpack_dual_2[1];  _unpack_dual_2 = None

        _unwrap_for_grad_5: "f32[3, 3, 3]" = torch._functorch.predispatch._unwrap_for_grad(getitem_5, 1);  getitem_5 = None

        _exit_dual_level = torch._functorch.predispatch._exit_dual_level(level = 0);  _exit_dual_level = None
        _set_fwd_grad_enabled_3 = torch._C._set_fwd_grad_enabled(True);  _set_fwd_grad_enabled_3 = None
        _jvp_decrement_nesting_1 = torch._functorch.predispatch._jvp_decrement_nesting();  _jvp_decrement_nesting_1 = None
        return (_unwrap_for_grad_2, _unwrap_for_grad_4, _unwrap_for_grad_3, _unwrap_for_grad_5)
""",
        )

    def test_jvp_freevar_python_scalar(self):
        # [FUNCTION DOC] Tests jvp of a function adding a free Python int, compiled with aot_eager
        # and fullgraph; result must equal eager.
        counters.clear()
        y = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x plus the free int y.
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns jvp of fn at x with tangent x.
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_linearize_jvp_fn(self):
        # [FUNCTION DOC] Tests linearize of sin followed by calling the returned jvp function,
        # compiled without fullgraph. Checks two graphs (index zero and one) against expected
        # printed text, outside dynamic capture mode.
        counters.clear()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Linearizes sin at x and returns the output and the linear jvp
            # function applied to x.
            output, jvp_fn = torch.func.linearize(torch.sin, x)
            return output, jvp_fn(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=0)

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_buffers_tensor_constant0_: "f32[3, 3, 3]"):
        l_self_buffers_tensor_constant0_ = L_self_buffers_tensor_constant0_

        alias_default: "f32[3, 3, 3]" = torch.ops.aten.alias.default(l_self_buffers_tensor_constant0_);  l_self_buffers_tensor_constant0_ = None

        alias_default_1: "f32[3, 3, 3]" = torch.ops.aten.alias.default(alias_default)

        cos_default: "f32[3, 3, 3]" = torch.ops.aten.cos.default(alias_default_1);  alias_default_1 = None

        sin_default: "f32[3, 3, 3]" = torch.ops.aten.sin.default(alias_default)

        alias_default_2: "f32[3, 3, 3]" = torch.ops.aten.alias.default(sin_default);  alias_default_2 = None
        return (alias_default, cos_default, sin_default)
""",
        )

        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=1)
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_flat_tangents_1_: "f32[3, 3, 3]", L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_: "f32[3, 3, 3]", L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_: "f32[3, 3, 3]"):
        l_flat_tangents_1_ = L_flat_tangents_1_
        l_self_modules_fx_const_folded_attrs_parameters_0_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_0_
        l_self_modules_fx_const_folded_attrs_parameters_1_ = L_self_modules_FX_CONST_FOLDED_ATTRS_parameters_1_

        _new_zeros_with_same_feature_meta_default: "f32[3, 3, 3]" = torch.ops.aten._new_zeros_with_same_feature_meta.default(l_flat_tangents_1_, l_self_modules_fx_const_folded_attrs_parameters_0_);  l_self_modules_fx_const_folded_attrs_parameters_0_ = None

        copy__default: "f32[3, 3, 3]" = torch.ops.aten.copy_.default(_new_zeros_with_same_feature_meta_default, l_flat_tangents_1_);  _new_zeros_with_same_feature_meta_default = l_flat_tangents_1_ = None

        mul_tensor: "f32[3, 3, 3]" = torch.ops.aten.mul.Tensor(copy__default, l_self_modules_fx_const_folded_attrs_parameters_1_);  copy__default = l_self_modules_fx_const_folded_attrs_parameters_1_ = None
        return (mul_tensor,)
""",
        )

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile(self):
        # [FUNCTION DOC] Tests, with error on recompile enabled, that vmap over a compiled
        # function does not recompile on a second identical call (issue 118493).
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda at
            # x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn)(x)
        # should not recompile on second call. See Pytorch issue #118493
        torch.vmap(fn)(x)

    @xfailIfTorchDynamo
    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_different_config(self):
        # [FUNCTION DOC] Tests, with error on recompile enabled (expected failure under torch
        # dynamo), that calling a function compiled under vmap directly without vmap raises
        # RecompileError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda at
            # x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn)(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            fn(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_same_config(self):
        # [FUNCTION DOC] Tests, with error on recompile enabled, that nested vmaps with the same
        # randomness mode do not recompile, but changing the outer randomness to error raises
        # RecompileError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda at
            # x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(torch.vmap(fn, randomness="same"), randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(torch.vmap(fn, randomness="same"), randomness="error")(x)

    @config.patch(error_on_recompile=True)
    def test_vmap_recompile_with_randomness(self):
        # [FUNCTION DOC] Tests, with error on recompile enabled, that calling under vmap with a
        # different randomness mode after same raises RecompileError.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning vmap of a sin lambda at
            # x.
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        with self.assertRaises(torch._dynamo.exc.RecompileError):
            torch.vmap(fn, randomness="different")(x)

    def test_compile_vmap_compiled_fn_jacrev_has_aux(self):
        # [FUNCTION DOC] Tests vmap over a compiled function containing jacrev with has_aux: the
        # vmapped function must not carry dynamo wrapper attributes, and compiling it must produce
        # the same shape and values as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first output of jacrev with auxiliary output of a
            # lambda that duplicates its input.
            grad_res, _ = torch.func.jacrev(lambda x: tuple([x] * 2), has_aux=True)(x)
            return grad_res

        compile_options = dict(backend="eager", fullgraph=True, dynamic=False)
        compiled_fn = torch.compile(fn, **compile_options)  # noqa: UNSPECIFIED_BACKEND
        vmapped_fn = torch.vmap(compiled_fn)
        for attr in _DYNAMO_WRAPPER_ATTRS:
            self.assertFalse(hasattr(vmapped_fn, attr), attr)
        compiled_vmapped_fn = torch.compile(vmapped_fn, **compile_options)  # noqa: UNSPECIFIED_BACKEND

        x = torch.randn(8, dtype=torch.float64)
        expected = vmapped_fn(x)
        actual = compiled_vmapped_fn(x)

        self.assertEqual(expected.shape, torch.Size([8]))
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual, expected)

    def test_compile_function_transform_wrappers_compiled_fn(self):
        # [FUNCTION DOC] Tests that applying function transforms (grad, grad_and_value, jacrev,
        # jacfwd, hessian) to a compiled function yields wrappers without dynamo wrapper
        # attributes, and that compiling the transformed function matches the transform of the
        # uncompiled function (checked per transform with subTest).
        def scalar_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        def vector_fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of x.
            return x.sin()

        compile_options = dict(backend="eager", fullgraph=True, dynamic=False)

        cases = (
            (torch.func.grad, scalar_fn, torch.randn((), dtype=torch.float64)),
            (
                torch.func.grad_and_value,
                scalar_fn,
                torch.randn((), dtype=torch.float64),
            ),
            (torch.func.jacrev, vector_fn, torch.randn(3, dtype=torch.float64)),
            (torch.func.jacfwd, vector_fn, torch.randn(3, dtype=torch.float64)),
            (torch.func.hessian, scalar_fn, torch.randn((), dtype=torch.float64)),
        )
        for transform, fn, x in cases:
            with self.subTest(transform=transform.__name__):
                compiled_fn = torch.compile(fn, **compile_options)  # noqa: UNSPECIFIED_BACKEND
                transformed_fn = transform(compiled_fn)
                for attr in _DYNAMO_WRAPPER_ATTRS:
                    self.assertFalse(hasattr(transformed_fn, attr), attr)

                compiled_transformed_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                    transformed_fn, **compile_options
                )

                self.assertEqual(compiled_transformed_fn(x), transform(fn)(x))

    def test_functionalize_compiled_fn_does_not_copy_dynamo_attrs(self):
        # [FUNCTION DOC] Tests that functionalize of a compiled function does not copy dynamo
        # wrapper attributes and gives the same result as functionalize of the plain function.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clones x, adds one in place and returns sin of the clone.
            y = x.clone()
            y.add_(1)
            return y.sin()

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)
        transformed_fn = torch.func.functionalize(compiled_fn)
        for attr in _DYNAMO_WRAPPER_ATTRS:
            self.assertFalse(hasattr(transformed_fn, attr), attr)
        x = torch.randn(3)
        self.assertEqual(transformed_fn(x), torch.func.functionalize(fn)(x))

    def test_vmap_call_torch_compile_fn(self):
        # [FUNCTION DOC] Tests that calling vmap on a torch.compile'd function from eager mode
        # raises dynamo Unsupported with a message about vmap of a compiled function.
        def wrapped_fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of x.
            return x.sin()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.vmap\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.vmap(fn)(x)

    def test_vmap_call_compiled_backward_fn(self):
        # [FUNCTION DOC] Tests (issue 138422) that calling a compiled backward function inside vjp
        # raises RuntimeError saying it is unsupported within vmap, grad or vjp.
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function returning x squared.
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured output y with respect to
            # x for cotangent v.
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.vjp(get_vjp, x)

    def test_vjp_call_compiled_backward_fn(self):
        # [FUNCTION DOC] Tests (issue 138422) that calling a compiled backward function inside vjp
        # raises RuntimeError saying it is unsupported within vmap, grad or vjp.
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function returning x squared.
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured output y with respect to
            # x for cotangent v.
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.vjp(get_vjp, x)

    def test_grad_call_compiled_backward_fn(self):
        # [FUNCTION DOC] Tests (issue 138422) that calling a compiled backward function inside
        # grad raises RuntimeError saying it is unsupported within vmap, grad or vjp.
        # See PyTorch issue #138422
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function returning x squared.
            return x**2

        x = torch.randn(2, requires_grad=True)
        y = f(x)

        def get_vjp(v):
            # [FUNCTION DOC] HELPER: Returns the gradient of the captured output y with respect to
            # x for cotangent v.
            return torch.autograd.grad(y, x, v)

        with self.assertRaisesRegex(
            RuntimeError,
            "It looks like you're trying to call a compiled backward function within vmap/grad/vjp, which isn't supported",
        ):
            torch.func.grad(get_vjp)(x)

    def test_grad_call_torch_compile_fn(self):
        # [FUNCTION DOC] Tests that calling grad on a torch.compile'd function from eager mode
        # raises dynamo Unsupported with a message about grad of a compiled function.
        def wrapped_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.grad\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.grad(fn)(x)

    def test_jvp_call_torch_compile_fn(self):
        # [FUNCTION DOC] Tests that calling jvp on a torch.compile'd function from eager mode
        # raises dynamo Unsupported with a message about jvp of a compiled function.
        def wrapped_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin of x.
            return x.sin().sum()

        x = torch.randn(3, 4)
        fn = torch.compile(backend="aot_eager", fullgraph=True)(wrapped_fn)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Calling torch.func.jvp\\(compiled_fn\\) function from eager mode is not supported",
        ):
            torch.func.jvp(fn, (x,), (x,))

    @config.patch(error_on_recompile=True)
    def test_grad_recompile(self):
        # [FUNCTION DOC] Tests, with error on recompile enabled, that grad over a function
        # compiled with grad inside does not recompile on a second call.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (eager) function returning grad of sin at x.
            return torch.func.grad(torch.sin)(x)

        x = torch.randn([])
        torch.func.grad(fn)(x)
        # should not recompile on second call
        torch.func.grad(fn)(x)

    def test_vmap_get_wrapped(self):
        # [FUNCTION DOC] Tests that a compiled function can return a vmap of a function, and
        # calling the returned wrapper matches eager vmap.
        counters.clear()

        def g(x):
            # [FUNCTION DOC] HELPER: Returns sin of x.
            return x.sin()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled (aot_eager, fullgraph) function returning vmap of g.
            return torch.vmap(g)

        x = torch.randn(3, 4)
        expected = torch.vmap(g)(x)
        wrapper = fn()
        got = wrapper(x)
        self.assertEqual(expected, got)

    def test_vmap_with_conditional_graph_break(self):
        # [FUNCTION DOC] Tests vmap over a function that graph breaks only for inputs with fewer
        # than two dimensions: first call (one dim per example) records one graph break and gives
        # sin; second call with higher rank records zero graph breaks and gives cos.
        def g(x):
            # [FUNCTION DOC] HELPER: Graph breaks and returns sin of x if x has fewer than two
            # dimensions, otherwise returns cos of x.
            if len(x.shape) < 2:
                torch._dynamo.graph_break()
                return x.sin()
            else:
                return x.cos()

        @torch.compile(backend="aot_eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled (aot_eager) function returning vmap of g at x.
            return torch.vmap(g)(x)

        counters.clear()
        x = torch.randn(2, 3)
        expected = x.sin()
        got = fn(x)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)

        counters.clear()
        y = torch.randn(2, 3, 4)
        expected = y.cos()
        got = fn(y)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 0)

    def test_vmap_with_graph_break(self):
        # [FUNCTION DOC] Tests vmap over a function that prints (causing a graph break); one graph
        # break is recorded and the result matches eager.
        counters.clear()

        def g(x):
            # [FUNCTION DOC] HELPER: Returns sin of cos of x after printing a message.
            y = x.cos()
            print("hi")
            return y.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of g at x.
            return torch.vmap(g)(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_2(self):
        # [FUNCTION DOC] Tests vmap with randomness same over a function calling two helper
        # functions that print; one graph break recorded and results match eager.
        counters.clear()

        def cos(x):
            # [FUNCTION DOC] HELPER: Prints a message and returns cos of x.
            print("cos")
            return x.cos()

        def sin(x):
            # [FUNCTION DOC] HELPER: Prints a message and returns sin of x.
            print("sin")
            return x.sin()

        def g(x):
            # [FUNCTION DOC] HELPER: Returns sin of cos of x via the helpers.
            y = cos(x)
            return sin(y)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap with randomness same of g at x.
            return torch.vmap(g, randomness="same")(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_with_graph_break_lambda(self):
        # [FUNCTION DOC] Tests vmap of a lambda calling a printing helper; one graph break
        # recorded and result matches eager.
        counters.clear()

        def sin(x):
            # [FUNCTION DOC] HELPER: Prints a message and returns sin of x.
            print("sin")
            return x.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda calling sin at x.
            return torch.vmap(lambda x: sin(x))(x)

        x = torch.randn(3, 4)
        opt = torch.compile(fn, backend="aot_eager", fullgraph=False)
        expected = fn(x)
        got = opt(x)
        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(expected, got)

    def test_vmap_scalar_tensor_indexing(self):
        # [FUNCTION DOC] Tests vmap indexing a captured matrix with scalar tensor indices drawn
        # from the mapped argument, compiled with fullgraph; results must equal eager.
        data = torch.arange(20).reshape(2, 10)
        b_indices = torch.arange(2)
        n_indices = torch.arange(10)

        def vmap_index_fn(data_in, b_indices, n_indices):
            # [FUNCTION DOC] HELPER: Returns vmap over the second argument of the indexing
            # function applied to the two index vectors.
            def index_fn(b, n):
                # [FUNCTION DOC] HELPER: Returns the element of the captured data at the row b and
                # column n.
                return data_in[b, n]

            return torch.func.vmap(index_fn, in_dims=(None, 0))(b_indices, n_indices)

        eager_result = vmap_index_fn(data, b_indices, n_indices)

        compiled_result = torch.compile(vmap_index_fn, backend="eager", fullgraph=True)(
            data, b_indices, n_indices
        )

        self.assertEqual(eager_result, compiled_result)

    def test_vmap(self):
        # [FUNCTION DOC] Tests compiling vmap of a lambda adding sums over dimensions zero and
        # one; compares with eager and the printed graph (outside dynamic capture mode).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda summing over both dimensions at x.
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(add, 1, 3, 0);  add = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_const(self):
        # [FUNCTION DOC] Tests vmap of a lambda that also adds a free Python int; compares with
        # eager and the printed graph (outside dynamic capture mode).
        y = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda adding the sums and the free int y.
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        add_1: "f32[3]" = add + 3;  add = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(add_1, 1, 3, 0);  add_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_free_tensor(self):
        # [FUNCTION DOC] Tests vmap of a lambda that adds a free tensor; compares with eager and
        # the printed graph (outside dynamic capture mode).
        y = torch.randn(3, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda adding the sums and the free tensor
            # y.
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)

        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        add_1: "f32[3, 3]" = add + l_y_;  add = l_y_ = None

        _remove_batch_dim: "f32[3, 3, 3]" = torch._functorch.predispatch._remove_batch_dim(add_1, 1, 3, 0);  add_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs(self):
        # [FUNCTION DOC] Tests vmap with two inputs and in_dims given as a tuple (0, 1); compares
        # with eager and the printed graph (outside dynamic capture mode).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap with in_dims zero and one of a lambda over x and
            # y.
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=(0, 1)
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        add_1: "f32[3]" = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(add_1, 1, 3, 0);  add_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_two_inputs_tuple_in_dims(self):
        # [FUNCTION DOC] Tests vmap with in_dims taken from a captured tuple variable; compares
        # with eager and the printed graph (outside dynamic capture mode).
        in_dims = (0, 1)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap with the captured in_dims of a lambda over x and
            # y.
            return torch.func.vmap(
                lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=in_dims
            )(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 1, 1);  l_y_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[3]" = _add_batch_dim.sum(1);  _add_batch_dim = None
        add: "f32[3]" = sum_1 + sum_2;  sum_1 = sum_2 = None
        add_1: "f32[3]" = add + _add_batch_dim_1;  add = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(add_1, 1, 3, 0);  add_1 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim,)
""",
        )

    def test_vmap_over_vmap_two_inputs(self):
        # [FUNCTION DOC] Tests nested vmap of an addition lambda with inner in_dims one; compares
        # with eager and the printed graph (outside dynamic capture mode).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of vmap (inner in_dims one) of an addition
            # lambda at x and y.
            return torch.func.vmap(torch.func.vmap(lambda x, y: x + y, in_dims=1))(x, y)

        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3, 3]", L_y_: "f32[3, 3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None
        _add_batch_dim_1: "f32[3, 3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_2: "f32[3]" = torch._functorch.predispatch._add_batch_dim(_add_batch_dim, 1, 2);  _add_batch_dim = None
        _add_batch_dim_3: "f32[3]" = torch._functorch.predispatch._add_batch_dim(_add_batch_dim_1, 1, 2);  _add_batch_dim_1 = None

        add: "f32[3]" = _add_batch_dim_2 + _add_batch_dim_3;  _add_batch_dim_2 = _add_batch_dim_3 = None

        _remove_batch_dim: "f32[3, 3]" = torch._functorch.predispatch._remove_batch_dim(add, 2, 3, 0);  add = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[3, 3, 3]" = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim, 1, 3, 0);  _remove_batch_dim = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_over_vmap_captured(self):
        # [FUNCTION DOC] Tests nested vmap where the inner lambda captures the outer tensor x;
        # compares with eager and the printed graph (outside dynamic capture mode).
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns nested vmap of a lambda multiplying the captured x by
            # the mapped y values (y taken from the enclosing test scope).
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]", L_y_: "f32[5, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(5, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[3]" = torch._functorch.predispatch._add_batch_dim(l_y_, 0, 1);  l_y_ = None

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(3, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_1: "f32[]" = torch._functorch.predispatch._add_batch_dim(_add_batch_dim, 0, 2);  _add_batch_dim = None

        mul: "f32[2, 3]" = l_x_ * _add_batch_dim_1;  l_x_ = _add_batch_dim_1 = None

        _remove_batch_dim: "f32[3, 2, 3]" = torch._functorch.predispatch._remove_batch_dim(mul, 2, 3, 0);  mul = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        _remove_batch_dim_1: "f32[5, 3, 2, 3]" = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim, 1, 5, 0);  _remove_batch_dim = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
        return (_remove_batch_dim_1,)
""",
        )

    def test_vmap_multiple_outputs(self):
        # [FUNCTION DOC] Tests vmap of a lambda returning two values (sums over dimensions zero
        # and one); compares with eager and the printed graph (outside dynamic capture mode).
        x = torch.ones(2, 4, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda returning a tuple of two sums at x.
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[2, 3]" = torch._functorch.predispatch._remove_batch_dim(sum_1, 1, 2, 0);  sum_1 = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(sum_2, 1, 2, 0);  sum_2 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_diff_dims(self):
        # [FUNCTION DOC] Tests vmap with out_dims given as a literal tuple (1, 0) for a two output
        # lambda; compares with eager and the printed graph (outside dynamic capture mode).
        x = torch.ones(2, 4, 3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap with out_dims one and zero of a lambda returning
            # two sums at x.
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._functorch.predispatch._remove_batch_dim(sum_1, 1, 2, 1);  sum_1 = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(sum_2, 1, 2, 0);  sum_2 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_multiple_outputs_out_dims_tuple(self):
        # [FUNCTION DOC] Tests vmap with out_dims taken from a captured tuple variable; compares
        # with eager and the printed graph (outside dynamic capture mode).
        x = torch.ones(2, 4, 3)
        out_dims = (1, 0)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap with the captured out_dims of a lambda returning
            # two sums at x.
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=out_dims)(x)

        wrapped_gm = self._compile_check(fn, (x,))

        # Dynamic shapes produce a slightly different graph.
        if check_dynamic_shape_capture():
            return

        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 4, 3]"):
        l_x_ = L_x_

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[4, 3]" = torch._functorch.predispatch._add_batch_dim(l_x_, 0, 1);  l_x_ = None

        sum_1: "f32[3]" = _add_batch_dim.sum(0)
        sum_2: "f32[4]" = _add_batch_dim.sum(1);  _add_batch_dim = None

        _remove_batch_dim: "f32[3, 2]" = torch._functorch.predispatch._remove_batch_dim(sum_1, 1, 2, 1);  sum_1 = None
        _remove_batch_dim_1: "f32[2, 4]" = torch._functorch.predispatch._remove_batch_dim(sum_2, 1, 2, 0);  sum_2 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
        return (_remove_batch_dim, _remove_batch_dim_1)
""",
        )

    def test_vmap_kwargs(self):
        # [FUNCTION DOC] Tests vmap called with a keyword argument; compiled without fullgraph,
        # output equals eager and zero graph breaks recorded.
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of an addition lambda at x with y passed as a
            # keyword.
            return torch.func.vmap(lambda x, y: x + y)(x, y=y)

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_pytree_inputs(self):
        # [FUNCTION DOC] Tests vmap over a function taking a dict pytree input; compiled without
        # fullgraph, output equals eager and zero graph breaks recorded.
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def vmap_fn(inps):
            # [FUNCTION DOC] HELPER: Reads the x and y entries of the input dict and returns their
            # sum.
            x = inps["x"]
            y = inps["y"]
            return x + y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of vmap_fn over a dict holding x and y.
            return torch.func.vmap(vmap_fn)({"x": x, "y": y})

        actual = fn(x, y)
        expected = torch.compile(fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_side_effects(self):
        # [FUNCTION DOC] Tests that a list append inside a vmapped function does not graph break;
        # the result equals eager and the list ends with two entries (eager once, compiled once).
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            # [FUNCTION DOC] HELPER: Appends one to the captured list some_list and returns x plus
            # y.
            some_list.append(1)
            return x + y

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of f at x and y.
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)
        self.assertEqual(some_list, [1, 1])

    @unittest.expectedFailure
    def test_vmap_side_effects_append_input(self):
        # [FUNCTION DOC] Expected failure test: appending a mapped input tensor to an outer list
        # inside vmap should not graph break and match eager, but currently fails.
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        some_list = []

        def f(x, y):
            # [FUNCTION DOC] HELPER: Appends x to the captured list and returns x plus y.
            some_list.append(x)
            return x + y

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of f at x and y.
            return torch.func.vmap(f)(x, y)

        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x, y)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_previous_illegal_op_no_graph_break(self):
        # [FUNCTION DOC] Tests that an operation previously illegal under vmap (calling stride on
        # a view) no longer causes a graph break; result equals eager and zero graph breaks
        # recorded.
        counters.clear()

        # calling .stride() would previously graph break
        def bad_fn(x):
            # [FUNCTION DOC] HELPER: Views x as four by three, calls stride on the view (result
            # ignored), and returns the view.
            y = x.view((4, 3))
            y.stride()
            return y

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of bad_fn at x.
            return torch.func.vmap(bad_fn)(x)

        x = torch.randn(2, 3, 4)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=False)(x)
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        # [FUNCTION DOC] Tests repeated vmap calls with different in_dims values (zero, one, two)
        # under dynamic compile: results equal eager, three frames compiled and 18 ops total.
        counters.clear()

        def wrapper_fn(x, in_dims):
            # [FUNCTION DOC] HELPER: Returns vmap of sum with the given in_dims at x.
            return torch.func.vmap(torch.sum, in_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 18)

    def test_vmap_multiple_invocation_out_dims(self):
        # [FUNCTION DOC] Tests repeated vmap calls with different out_dims values (zero, one, two)
        # under dynamic compile: results equal eager, three frames compiled and 18 ops total.
        counters.clear()

        def wrapper_fn(x, out_dims):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda summing dimension zero with the
            # given out_dims at x.
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)

        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2)
        # Third invocation of `opt` makes `in_dims` as SymInt.
        actual = opt(x, 0), opt(x, 1), opt(x, 2)
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 18)

    def test_vmap_out_dims_None(self):
        # [FUNCTION DOC] Tests vmap with in_dims of None and out_dims of None for the first
        # argument (issue 149509); compiled with fullgraph and result equals eager.
        # issue https://github.com/pytorch/pytorch/issues/149509
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x unchanged and y times two.
            return x, y * 2

        def wrapper_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns vmap of fn with in_dims None and zero and out_dims
            # None and zero at x and y.
            return torch.func.vmap(fn, in_dims=(None, 0), out_dims=(None, 0))(x, y)

        x, y = torch.randn(4), torch.randn(3, 4)
        expected = wrapper_fn(x, y)
        got = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)(x, y)
        self.assertEqual(expected, got)

    def test_vmap_new_tensor_in_body(self):
        # [FUNCTION DOC] Tests that creating a new tensor (ones) inside a vmapped function body
        # compiles with fullgraph and equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus a new ones tensor of length three.
            return x + torch.ones(3)

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of fn at x.
            return torch.func.vmap(fn)(x)

        x = torch.randn(
            3,
        )
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        # [FUNCTION DOC] Tests that a vmapped function that ignores its input and returns a new
        # constant tensor compiles with fullgraph and equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a new scalar tensor 0.5, ignoring x.
            return torch.tensor(0.5)

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of fn at x.
            return torch.func.vmap(fn)(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_implicit_via_op(self):
        # [FUNCTION DOC] Tests that adding a Python scalar via torch.add inside a vmapped lambda
        # compiles with fullgraph and equals eager.
        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Returns vmap of a lambda adding 0.5 to its input at x.
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)

        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend="aot_eager", fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)


class ActivationCheckpointingTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for activation checkpointing (torch.utils.checkpoint) under dynamo with
    # aot_autograd or inductor backends, including RNG functionalization, graph break fallback,
    # modules, and nearby tests of higher order op registration (fallthrough keys, cond with
    # keyword arguments) and flop counting for cond. A shared validation helper compares forward
    # results and gradients with eager.
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        # [FUNCTION DOC] HELPER: Detaches and clones every input argument keeping its
        # requires_grad flag; seeds the RNG to zero, runs fn eagerly and backpropagates the sum;
        # seeds again and does the same for the compiled function on cloned inputs. Unless
        # skip_check is true, asserts the results and the gradient of each input are equal.
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.detach().clone().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3361")
    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        # [FUNCTION DOC] GPU test (skipped on XPU, needs triton, RNG functionalization on):
        # checkpoint of a sigmoid of a matrix product with reentrant mode. Uses aot_autograd with
        # counting compilers expecting one matmul in forward and two in backward, then validates
        # against eager.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns sigmoid of the matrix product of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns checkpoint (reentrant) of gn on sin of x and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3361")
    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        # [FUNCTION DOC] GPU test (skipped on XPU, needs triton, RNG functionalization on): same
        # as the basic checkpoint test but checkpoint is called with preserve_rng_state false.
        # Expects one matmul forward and two backward, validated against eager.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns sigmoid of the matrix product of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns checkpoint (reentrant, no RNG state preservation) of
            # gn on sin of x and y.
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=True,
                preserve_rng_state=False,
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3361")
    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        # [FUNCTION DOC] GPU test (skipped on XPU, needs triton, RNG functionalization on):
        # checkpoint of dropout of a matrix product; asserts the philox random op appears once in
        # forward and zero times in backward (it is passed from forward), skipping value checks
        # since the dropout decomposition diverges from eager.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns dropout with probability 0.2 of the matrix product of
            # x and y.
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns checkpoint (reentrant) of gn on sin of x and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device=device_type, requires_grad=True)
        y = torch.randn(4, 4, device=device_type, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        # philox_rand is passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.rngprims.philox_rand.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3361")
    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        # [FUNCTION DOC] GPU test (skipped on XPU, needs triton, RNG functionalization on): same
        # dropout checkpoint function compiled with the inductor backend, validated with value
        # checks skipped.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns dropout with probability 0.2 of the matrix product of
            # x and y.
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns checkpoint (reentrant) of gn on sin of x and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device=device_type, requires_grad=True)
        y = torch.randn(4, 4, device=device_type, requires_grad=True)

        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        # [FUNCTION DOC] GPU test (needs triton, RNG functionalization on): checkpoint whose
        # function contains a graph break falls back; compiled result equals eager and there are
        # two frames, two ops total and two graphs (sin before and cos after).
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Triggers a dynamo graph break and returns sigmoid of the
            # matrix product of x and y.
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns cos of checkpoint (reentrant) of gn on sin of x and
            # y.
            return torch.cos(
                torch.utils.checkpoint.checkpoint(
                    gn, torch.sin(x), y, use_reentrant=True
                ),
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 2)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3361")
    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_module(self):
        # [FUNCTION DOC] GPU test (skipped on XPU, needs triton, RNG functionalization on):
        # checkpoint of a module call. Expects one sigmoid in forward and none in backward (passed
        # from forward), then validates against eager.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ten by ten linear layer; forward returns sigmoid
            # of the layer output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sigmoid of the linear layer applied to x.
                return torch.sigmoid(self.linear(x))

        mod = MockModule()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns checkpoint (reentrant) of the module on sin of x.
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # sigmoid passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    def test_override_fallthrough_dispatch_key(self):
        # [FUNCTION DOC] Tests that registering python implementations for each default
        # fallthrough dispatch key on a new higher order operator marks all of them as non
        # fallthrough and stores the registered kernels. Verifies none are non fallthrough
        # beforehand, registers an index returning function per key, then checks all keys are non
        # fallthrough and the kernels return their indices in order.
        class _FallthroughTestOnly(torch._ops.HigherOrderOperator):
            # [CLASS DOC] HELPER: Minimal higher order operator named _fallthrough_test_only used
            # only to test fallthrough key overriding.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the operator under its name.
                super().__init__("_fallthrough_test_only")

            def __call__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the parent call implementation
                # and returns its result.
                return super().__call__(*args, **kwargs)

        test_op = _FallthroughTestOnly()
        default_keys = torch._ops._HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS
        self.assertTrue(
            not any(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )

        foos = [lambda x=i: x for i, k in enumerate(default_keys)]
        for foo, fallthrough_key in zip(foos, default_keys):
            test_op.py_impl(fallthrough_key)(foo)

        self.assertTrue(
            all(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )
        self.assertEqual(
            list(range(len(default_keys))),
            [test_op.py_kernels[key]() for key in default_keys],
        )

    def test_cond_with_kwargs(self):
        # [FUNCTION DOC] Tests that the cond operator can be called with keyword arguments (pred,
        # true_fn, false_fn, operands) when compiled with fullgraph; results match eager for true
        # and false predicates and only one frame is compiled.
        from torch._higher_order_ops.cond import cond_op

        def test(pred, x):
            # [FUNCTION DOC] HELPER: Calls cond_op with keyword arguments using a clone and a
            # negation branch.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: Returns a clone of x.
                return x.clone()

            def false_fn(x):
                # [FUNCTION DOC] HELPER: Returns the negation of x.
                return -x

            return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt, fullgraph=True)
        inp = torch.ones(3, 3)
        true_pred = torch.Tensor([True])
        false_pred = torch.Tensor([False])
        self.assertTrue(torch.allclose(test(true_pred, inp), opt_test(true_pred, inp)))
        self.assertEqual(cnt.frame_count, 1)
        self.assertTrue(
            torch.allclose(test(false_pred, inp), opt_test(false_pred, inp))
        )
        self.assertEqual(cnt.frame_count, 1)

    def test_cond_with_invalid_kwargs(self):
        # [FUNCTION DOC] Tests cond called with invalid keywords: an unknown keyword argument
        # raises UncapturedHigherOrderOpError when compiled, and passing the predicate both
        # positionally and by keyword raises AssertionError.
        from torch._higher_order_ops.cond import cond_op

        def test(pred, mode, x):
            # [FUNCTION DOC] HELPER: Depending on mode, calls cond_op with an extra unknown
            # keyword or with pred given twice.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: Returns a clone of x.
                return x.clone()

            def false_fn(x):
                # [FUNCTION DOC] HELPER: Returns the negation of x.
                return -x

            if mode:
                return cond_op(
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                    invalid=True,
                )
            else:
                return cond_op(
                    pred,
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                )

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            opt_test(True, True, inp)

        with self.assertRaises(AssertionError):
            opt_test(True, False, inp)

    def test_cond_with_mismatched_output(self):
        # [FUNCTION DOC] Tests that cond whose branches return tensors of different shapes (concat
        # versus sin) runs in eager and also compiles with the eager backend without error.
        def output_mismatch_test(x):
            # [FUNCTION DOC] HELPER: Calls cond on whether the sum of x is positive with
            # argument-free branches.
            def true_fn():
                # [FUNCTION DOC] HELPER: Returns the concatenation of x with itself.
                return torch.concat([x, x])

            def false_fn():
                # [FUNCTION DOC] HELPER: Returns sin of x.
                return x.sin()

            return torch.cond(x.sum() > 0, true_fn, false_fn)

        x = torch.randn(2, 3)
        output_mismatch_test(x)

        torch.compile(output_mismatch_test, backend="eager")(x)

    def test_non_aliasing_util(self):
        # [FUNCTION DOC] Tests the private tensor aliasing assertion utility: passing disjoint
        # nested structures of tensors succeeds, passing the same structure for both arguments
        # raises AssertionError saying inputs to the function body cannot alias outputs.
        from torch._dynamo.variables.higher_order_ops import _assert_tensors_nonaliasing

        a = [torch.tensor(1), {"a": torch.tensor(1)}]
        b = (torch.tensor(1),)
        _assert_tensors_nonaliasing(a, b)

        with self.assertRaisesRegex(
            AssertionError, "inputs to function body cannot alias outputs"
        ):
            _assert_tensors_nonaliasing(a, a)

    def test_flop_counter_for_cond(self):
        # [FUNCTION DOC] Tests that FlopCounterMode counts matmul flops (256 in total for the
        # module) through a cond, attributing them to the Global, module and linear entries.
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a four by four linear layer; forward calls cond with
            # true predicate whose branches apply the layer once or twice.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns cond with a true predicate, a branch applying the
                # layer once, and a branch applying it twice.
                return torch.cond(
                    torch.tensor(True),
                    lambda x: self.linear(x),
                    lambda x: self.linear(self.linear(x)),
                    (x,),
                )

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 256},
                "Mod": {torch.ops.aten.addmm: 256},
                "Mod.linear": {torch.ops.aten.addmm: 256},
            },
        )

    def test_flop_counter_for_nested_cond(self):
        # [FUNCTION DOC] Tests flop counting through nested cond: the totals are 256 for Global
        # and the module and 128 each for the two linear layers.
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two four by four linear layers; forward calls an
            # outer cond whose true branch contains an inner cond.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates both layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Defines two branches and returns the outer cond with a
                # true predicate.
                def true_branch(x):
                    # [FUNCTION DOC] HELPER: True branch returning an inner cond on a true
                    # predicate that applies the first layer or the second layer.
                    # Nested cond inside true branch
                    return torch.cond(
                        torch.tensor(True),
                        lambda x: self.linear1(x),
                        lambda x: self.linear2(x),
                        (x,),
                    )

                def false_branch(x):
                    # [FUNCTION DOC] HELPER: False branch applying the second layer then the
                    # first.
                    return self.linear1(self.linear2(x))

                return torch.cond(torch.tensor(True), true_branch, false_branch, (x,))

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 256},
                "Mod": {torch.ops.aten.addmm: 256},
                "Mod.linear1": {torch.ops.aten.addmm: 128},
                "Mod.linear2": {torch.ops.aten.addmm: 128},
            },
        )

    def test_flop_counter_for_cond_unbalanced_branches(self):
        # [FUNCTION DOC] Tests flop counting through a cond with unbalanced branches (one branch
        # applies the layer, the other clones the input): the counts are 128 for Global, the
        # module and the layer.
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a four by four linear layer; forward calls cond with
            # a true predicate and an unbalanced pair of branches.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module and creates the layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Defines branches and returns cond with a true predicate.
                def true_branch(x):
                    # [FUNCTION DOC] HELPER: True branch applying the linear layer.
                    return self.linear(x)

                def false_branch(x):
                    # [FUNCTION DOC] HELPER: False branch returning a clone of x.
                    return x.clone()

                return torch.cond(torch.tensor(True), true_branch, false_branch, (x,))

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 128},
                "Mod": {torch.ops.aten.addmm: 128},
                "Mod.linear": {torch.ops.aten.addmm: 128},
            },
        )


xfail_hops_compile = {
    # aot_eager
    "map",  # assert type(args[1].realize()) is TensorVariable
    "scan",  # scan is not an OpOverload
    "local_map_hop",  # can't retrace
    "register_hook",  # only generated internally by Dynamo, not callable directly
    # inductor
    "while_loop",  # LoweringException: AssertionError
    "flex_attention",  # LoweringException: AssertionError
    "switch",  # no inductor lowering registered
}


class TestHigherOrderOpsOpInfo(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case that compiles every higher order op from the shared hop_db sample
    # inputs (except those listed as known failures) with aot_eager and inductor backends. Device
    # specific variants are generated for cuda and xpu only.
    @requires_gpu_and_triton
    @parametrize("backend", ("aot_eager", "inductor"))
    @ops(
        list(filter(lambda op: op.name not in xfail_hops_compile, hop_db)),
        allowed_dtypes=(torch.float,),
    )
    def test_hops_compile(self, device, dtype, op, backend):
        # [FUNCTION DOC] GPU test (needs triton), parametrized over the aot_eager and inductor
        # backends and over the HOP database entries excluding the xfail set, float only. For each
        # sample input, builds eager and deep copied compiled arguments, compiles a function
        # calling the op with fullgraph, and asserts equal outputs. Raises SkipTest for the
        # invoke_quant op with aot_eager because the partitioner fails.
        # Ensure HOPs can be compiled

        if backend == "aot_eager" and op.name == "invoke_quant":
            raise unittest.SkipTest(
                "TODO: partitioner fails. migrate canonicalization to aot eager backend"
            )

        sample_inputs_itr = op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        )
        for inp in sample_inputs_itr:
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            eager_args = (*input, *inp.args)
            eager_kwargs = inp.kwargs
            compiled_args = deepcopy(eager_args)
            compiled_kwargs = deepcopy(eager_kwargs)

            def fn(args, kwargs):
                # [FUNCTION DOC] HELPER: Calls the op with the given positional arguments and
                # keyword arguments and returns the result.
                return op.op(*args, **(kwargs))

            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

            eager_out = fn(eager_args, eager_kwargs)
            compiled_out = compiled_fn(compiled_args, compiled_kwargs)
            self.assertEqual(eager_out, compiled_out)


instantiate_device_type_tests(
    TestHigherOrderOpsOpInfo, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
