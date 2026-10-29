# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for the functorch control flow higher-order operators: cond, switch,
#   while_loop, scan, associative_scan and map.
# Defines shared helpers (cudagraph checks, functionalization converters, reference "fake"
#   implementations, combine-function factories, canned while_loop test programs) and a large
#   TestControlFlow suite checking eager, compiled, make_fx, functionalization, autograd and
#   cudagraph behavior, many with golden graph strings.
# Import time: builds the WHILE_LOOP_TESTS table by calling the module-level builder;
#   otherwise only imports.
import contextlib
import functools
import unittest

import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.associative_scan import (
    _fake_associative_scan,
    associative_scan,
)
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    CUDAGraphCaptureControlFlowOpDispatchMode,
)
from torch._higher_order_ops.map import _fake_map
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._higher_order_ops.schema import HopSchemaGenerator
from torch._higher_order_ops.switch import switch
from torch._higher_order_ops.while_loop import while_loop
from torch._subclasses.functional_tensor import (
    CppFunctionalizeAPI,
    FunctionalTensor,
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    requires_cuda,
    run_tests,
    skipIfCrossRef,
    skipIfRocm,
    skipIfTorchDynamo,
    TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


@contextlib.contextmanager
def check_cudagraphs_not_skipped(test_case):
    # [FUNCTION DOC] HELPER: context manager that asserts no inductor cudagraph skips happened
    #   inside its body. Records the dynamo inductor cudagraph_skips counter on entry; in the
    #   finally path restores the saved counter value and then asserts, using the given test case,
    #   that the counter seen at exit equals the one it reads back (the restore happens before the
    #   assert so a failure does not leak state).
    counters = torch._dynamo.utils.counters
    old_cudagraph_skips = counters["inductor"]["cudagraph_skips"]
    try:
        yield
    finally:
        # reset before the assert, because otherwise the reset is
        # skipped in case of assertion error
        new_cudagraph_skips = counters["inductor"]["cudagraph_skips"]
        counters["inductor"]["cudagraph_skips"] = old_cudagraph_skips
        test_case.assertEqual(
            counters["inductor"]["cudagraph_skips"], new_cudagraph_skips
        )


def _check_compile_cudagraph_backend(test_case, fn, args):
    # [FUNCTION DOC] HELPER: compiles fn with the cudagraphs backend and runs it three times
    #   (eager warmup, capture plus replay, replay), each inside the no-skip context manager and
    #   after marking a new cudagraph step. Each output tree is detached and cloned. Then runs fn
    #   eagerly and asserts via test_case that every collected output equals the eager result.
    # test cudagraphs backend
    cudagraphs_compiled_fn = torch.compile(fn, backend="cudagraphs")
    # We run 3 times.
    # This is what cuda graph trees does for the first 3 runs:
    # 1) run in eager mode, for warmup.
    # 2) do stream capture followed by graph replay.
    # 3 and beyond) do graph replay
    # So we need to get to iteration 3 to test all ways of running.
    outputs = []
    for i in range(3):
        with check_cudagraphs_not_skipped(test_case):
            torch.compiler.cudagraph_mark_step_begin()
            outputs.append(
                pytree.tree_map(
                    lambda x: x.detach().clone() if isinstance(x, torch.Tensor) else x,
                    cudagraphs_compiled_fn(*args),
                )
            )
    eager_res = fn(*args)
    for output in outputs:
        test_case.assertEqual(eager_res, output)


def _check_compile_many_backends_with_cudagraph(test_case, fn, args):
    # [FUNCTION DOC] HELPER: runs the cudagraph comparison helper for the backends
    #   eager_no_compile, eager and aot_eager in turn, forwarding test_case, fn and args.
    for backend in ["eager_no_compile", "eager", "aot_eager"]:
        _check_compile_any_backend_with_cudagraph(test_case, fn, args, backend)


def _check_compile_any_backend_with_cudagraph(test_case, fn, args, backend):
    # [FUNCTION DOC] HELPER: checks that fn works under CUDA graph capture for one backend. Uses
    #   fn itself for eager_no_compile, else torch.compile with the backend. Runs a warmup on a
    #   side stream under the control-flow warmup dispatch mode, then captures a CUDAGraph on the
    #   side stream under the capture dispatch mode, runs eager fn on that stream, replays the
    #   graph, and asserts that both the warmup output and the replayed (cloned) captured output
    #   equal the eager result.
    if backend == "eager_no_compile":
        compiled_fn = fn
    else:
        compiled_fn = torch.compile(fn, backend=backend)
    outputs = []

    side_stream = torch.cuda.Stream()

    with torch.cuda.stream(side_stream), ControlFlowOpWarmupDispatchMode():
        warmup_output = compiled_fn(*args)
    outputs.append(warmup_output)

    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph, stream=side_stream),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
    ):
        captured_output = compiled_fn(*args)
    with torch.cuda.stream(side_stream):
        eager_res = fn(*args)
    graph.replay()
    outputs.append(
        pytree.tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
            captured_output,
        )
    )
    for output in outputs:
        test_case.assertEqual(eager_res, output)


# TODO: pull these helpers from AOTAutograd later
def to_fun(t):
    # [FUNCTION DOC] HELPER: converts a tensor into a FunctionalTensor via to_functional; any non-
    #   tensor value is returned unchanged.
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t


def from_fun(t):
    # [FUNCTION DOC] HELPER: inverse of to_fun. Non-FunctionalTensor input is returned as is,
    #   after raising AssertionError if it is a plain tensor that is nevertheless a
    #   functional tensor. For a FunctionalTensor it syncs pending updates and returns the
    #   unwrapped inner tensor.
    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            if torch._is_functional_tensor(t):
                raise AssertionError("Expected tensor to not be a functional tensor")
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t.elem)


def to_fun_old(t):
    # [FUNCTION DOC] HELPER: wraps a plain tensor into a C++ functional tensor and mirrors
    #   its autograd metadata onto the result; non-tensors and already functional tensors
    #   are returned unchanged.
    if isinstance(t, torch.Tensor) and not torch._is_functional_tensor(t):
        out = torch._to_functional_tensor(t)
        torch._mirror_autograd_meta_to(t, out)
        return out
    return t


def from_fun_old(t):
    # [FUNCTION DOC] HELPER: for a tensor input, raises AssertionError if it is not a C++
    #   functional tensor, otherwise syncs and returns the underlying tensor; non-tensor input
    #   is returned unchanged.
    # quick sanity assert
    if isinstance(t, torch.Tensor):
        if not torch._is_functional_tensor(t):
            raise AssertionError("Expected tensor to be a functional tensor")
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t


def _fake_while_loop(cond_fn, body_fn, operands):
    # [FUNCTION DOC] HELPER: reference eager while loop; repeatedly calls body_fn on the operands
    #   while cond_fn is truthy and returns the final operands.
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands


def compile_mode_helper(fct, compile_mode):
    # [FUNCTION DOC] HELPER: wraps fct according to the compile_mode string: compile gives
    #   fullgraph static-shape torch.compile, compile_dynamic_shape gives fullgraph dynamic-shape
    #   compile, eager gives fullgraph compile with the eager backend, and any other value
    #   returns fct untouched.
    if compile_mode == "compile":
        return torch.compile(fct, fullgraph=True, dynamic=False)
    elif compile_mode == "compile_dynamic_shape":
        return torch.compile(fct, fullgraph=True, dynamic=True)
    elif compile_mode == "eager":
        return torch.compile(fct, fullgraph=True, backend="eager")
    else:
        return fct


ALIAS_FN = [
    lambda x: x,
    lambda x: x.view(-1),
    lambda x: x.reshape(-1),
    lambda x: x.squeeze(0),
    lambda x: x.unsqueeze(0),
    lambda x: x.transpose(0, 1),
    lambda x: x.flatten(),
    lambda x: x.expand(1, *x.size()),
]


def get_scan_combine_fn(name, associative=True, parameters=None):
    # [FUNCTION DOC] HELPER: factory returning a combine function for scan or associative_scan
    #   tests, selected by name (add, adds, mul, div, s5_operator, different_input_size_operator,
    #   tuple_fct, complex_pointwise, non_pointwise, RNN, fct_c1_no_grad); raises ValueError for
    #   an unknown name. When associative is False, the chosen function is wrapped so it returns a
    #   pair holding its result twice (carry and output form). The RNN variant reads weights from
    #   the parameters argument.
    def add(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: returns the elementwise sum of the two tensors.
        return x + y

    def adds(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: returns a tuple with x doubled and y doubled (each
        #   added to itself).
        return x + x, y + y

    def mul(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: returns the elementwise product.
        return x * y

    def div(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: returns the elementwise quotient x over y.
        return x / y

    def s5_operator(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: S5-style linear recurrence combine. Unpacks each input into
        #   a pair (A, Bu) and returns the product of the A terms together with A_j times
        #   Bu_i plus Bu_j.
        A_i, Bu_i = x
        A_j, Bu_j = y
        return A_j * A_i, A_j * Bu_i + Bu_j

    def different_input_size_operator(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: combine over 5-tuples (x, dA, dB, C, y). New x is the sum of the
        #   two x entries; new y is an einsum of the new x with the second C; dA, dB and C of the
        #   second element are passed through with zero added. Returns the 5-tuple.
        x_o, dA_o, dB_o, C_o, y_o = x
        x_n, dA_n, dB_n, C_n, y_n = y

        x_new = x_n + x_o
        y_new = torch.einsum("bdn,bn->bd", x_new, C_n)

        return x_new, dA_n + 0.0, dB_n + 0.0, C_n + 0.0, y_new

    def tuple_fct(x, y):
        # [FUNCTION DOC] HELPER: combine for pair inputs; returns a pair holding the sum of the
        #   first entries and the product of the second entries.
        return (x[0] + y[0], x[1] * y[1])

    def complex_pointwise(x, y):
        # [FUNCTION DOC] HELPER: combine over nested dict structures; multiplies the "i" entries,
        #   multiplies the first nested list element under "j" and adds the "o" entries of the
        #   dict in the second list element, rebuilding the same structure.
        return {
            "i": x["i"] * y["i"],
            "j": (
                [x["j"][0][0] * y["j"][0][0]],
                [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
            ),
        }

    def non_pointwise(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: builds a fixed 2x2 matrix from a range on x's device and
        #   returns x and y each matrix-multiplied by it, summed. Not elementwise, so unsuitable
        #   as a pointwise combine.
        W = torch.arange(4, dtype=torch.float, device=x.device).view(2, 2)
        return x @ W + y @ W

    def RNN(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: RNN cell combine using the enclosing parameters list (four
        #   tensors): computes a candidate from y and the first two parameters, then tanh of that
        #   plus x times the third parameter plus the fourth. Returns the new hidden state and a
        #   clone of it.
        c_new = y @ parameters[0] + parameters[1]
        h_new = torch.tanh(c_new + x @ parameters[2] + parameters[3])
        return h_new, h_new.clone()

    def fct_c1_no_grad(x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: combine taking a pair x and a tensor y. Returns a nested pair of
        #   a carry pair (c1, c2) and the output h_new where h_new is tanh of the sum of both x
        #   entries and y, c2 is the second entry of x plus y, and c1 is the first entry of x plus
        #   y computed under no_grad so it carries no gradient history.
        h_new = torch.tanh(x[0] + x[1] + y)
        c2 = x[1] + y
        with torch.no_grad():
            c1 = x[0] + y
        return (c1, c2), h_new

    if name == "add":
        fct = add
    elif name == "adds":
        fct = adds
    elif name == "mul":
        fct = mul
    elif name == "div":
        fct = div
    elif name == "s5_operator":
        fct = s5_operator
    elif name == "different_input_size_operator":
        fct = different_input_size_operator
    elif name == "tuple_fct":
        fct = tuple_fct
    elif name == "complex_pointwise":
        fct = complex_pointwise
    elif name == "non_pointwise":
        fct = non_pointwise
    elif name == "RNN":
        fct = RNN
    elif name == "fct_c1_no_grad":
        fct = fct_c1_no_grad
    else:
        raise ValueError("Combine_fn name unknown!")

    if not associative:
        return lambda x, y: (fct(x, y), fct(x, y))
    else:
        return fct


def _while_loop_tests():
    # [FUNCTION DOC] HELPER: builds the table of while_loop test programs. Defines several
    #   functions and Modules (simple, simple_with_mutation, nested, Nested, SimpleWithLinear,
    #   SimpleWithPytreeCarry, NestedWithLinear, IntCarry, PytreeIntCarry, ConstAndSymIntOutput)
    #   and returns a dict mapping a case name to a pair of (callable or module instance, example
    #   input tuple). Called once at import time to create WHILE_LOOP_TESTS.
    def simple(x):
        # [FUNCTION DOC] HELPER: while_loop that adds one to x until its sum reaches 10; returns
        #   the final carry tuple.
        def cond_fn(x):
            # [FUNCTION DOC] HELPER: predicate: sum of x is less than 10.
            return x.sum() < 10

        def body_fn(x):
            # [FUNCTION DOC] HELPER: returns a one-element tuple with x plus one.
            return (x + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def simple_with_mutation(x):
        # [FUNCTION DOC] HELPER: like simple but both predicate and body perform clone plus in-
        #   place add one and add minus one on a copy, which leaves values unchanged, to exercise
        #   mutation handling inside the loop.
        def cond_fn(x):
            # [FUNCTION DOC] HELPER: predicate on a mutated clone of x (net zero change); true
            #   while its sum is less than 10.
            y = x.clone().add_(1).add_(-1)
            return y.sum() < 10

        def body_fn(x):
            # [FUNCTION DOC] HELPER: mutates a clone of x (net zero change) and returns a tuple
            #   with that plus one.
            y = x.clone().add_(1).add_(-1)
            return (y + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def nested(out_iter, it, y):
        # [FUNCTION DOC] HELPER: nested while_loop test. Outer loop runs while the sum of out_iter
        #   is below 2 and increments out_iter each iteration; each outer body runs an inner
        #   while_loop that runs while the sum of it is below 10, adding y to it and incrementing
        #   y. Returns the outer loop result tuple.
        def cond_fn(out_iter, it, y):
            # [FUNCTION DOC] HELPER: inner predicate, sum of it below 10.
            return it.sum() < 10

        def body_fn(out_iter, it, y):
            # [FUNCTION DOC] HELPER: inner body; returns a clone of out_iter, it plus y, and
            #   y plus one.
            return (out_iter.clone(), it + y, y + 1)

        def outer_cond_fn(out_iter, it, y):
            # [FUNCTION DOC] HELPER: outer predicate, sum of out_iter below 2.
            return out_iter.sum() < 2

        def outer_body_fn(out_iter, it, y):
            # [FUNCTION DOC] HELPER: runs the inner while_loop on the three carries, then returns
            #   out_iter plus one with the inner it and y.
            out_iter, it, y = while_loop(cond_fn, body_fn, (out_iter, it, y))
            return (out_iter + 1, it, y)

        return while_loop(outer_cond_fn, outer_body_fn, (out_iter, it, y))

    class Nested(torch.nn.Module):
        # [CLASS DOC] HELPER: module with two nested while_loops over scalar counters and
        #   two float tensors.
        def forward(self, ci, cj, a, b):
            # [FUNCTION DOC] HELPER: runs an outer while_loop on counter ci while it is positive;
            #   each iteration runs an inner while_loop on counter cj, which adds 3.14 to x and
            #   subtracts 2.71 from y per step, then decrements the outer counter, doubles x and
            #   halves y. Returns the loop result.
            def cond_fn(i1, j1, x1, y1):
                # [FUNCTION DOC] HELPER: outer predicate, first counter greater than zero.
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                # [FUNCTION DOC] HELPER: outer body; runs the inner while_loop then returns
                #   decremented counter, cloned inner counter, x doubled and y halved.
                def cond_fn_nested(i2, j2, x2, y2):
                    # [FUNCTION DOC] HELPER: inner predicate, second counter greater than zero.
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    # [FUNCTION DOC] HELPER: inner body; clones the first counter, decrements the
                    #   second, adds 3.14 to x and subtracts 2.71 from y.
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )
                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class SimpleWithLinear(torch.nn.Module):
        # [CLASS DOC] HELPER: module holding a 2x2 Linear and an integer buffer dec equal to one;
        #   used to test while_loop bodies that use module parameters and buffers.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: creates the linear layer and the buffer dec
            #   holding tensor one.
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            # [FUNCTION DOC] HELPER: runs while_loop on (iter, x); the loop continues while the
            #   counter minus dec is positive, and each step decrements the counter and applies
            #   the linear layer to x.
            def cond_fn(it, x):
                # [FUNCTION DOC] HELPER: true while it minus the module buffer dec is
                #   greater than zero.
                return it - self.dec > 0

            def body_fn(it, x):
                # [FUNCTION DOC] HELPER: returns the decremented counter and the linear layer
                #   applied to x.
                return it - 1, self.linear(x)

            return while_loop(cond_fn, body_fn, (iter, x))

    class SimpleWithPytreeCarry(torch.nn.Module):
        # [CLASS DOC] HELPER: module whose while_loop carry includes a nested pytree of tensors.
        def forward(self, it, pytree_input):
            # [FUNCTION DOC] HELPER: runs while_loop with a counter and a pytree input and
            #   returns the result.
            def cond_fn(it, pytree_input):
                # [FUNCTION DOC] HELPER: true while the counter is positive.
                return it > 0

            def body_fn(it, pytree_input):
                # [FUNCTION DOC] HELPER: rotates the pytree: new first element is the sine of the
                #   "x" entry, new "x" is the cosine of "y", new "y" is the old first element plus
                #   one; counter decremented. Keeps the original container structure.
                x = pytree_input[0][0]
                y = pytree_input[1]["x"]
                z = pytree_input[1]["y"]
                new_x = y.sin()
                new_y = z.cos()
                new_z = x + 1
                return it - 1, ([new_x], {"x": new_y, "y": new_z})

            return while_loop(cond_fn, body_fn, (it, pytree_input))

    class NestedWithLinear(torch.nn.Module):
        # [CLASS DOC] HELPER: module holding a SimpleWithLinear submodule, its own Linear layer
        #   and an integer buffer dec equal to one; nests a module-using loop inside another loop.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: creates the inner module, the outer linear layer and
            #   the dec buffer.
            super().__init__()
            self.mod = SimpleWithLinear()
            self.outer_linear = torch.nn.Linear(2, 2)
            self.dec = torch.nn.Buffer(torch.tensor(1))

        def forward(self, iter, x):
            # [FUNCTION DOC] HELPER: runs while_loop on (iter, x) that continues while the counter
            #   minus dec is positive.
            def cond_fn(it, x):
                # [FUNCTION DOC] HELPER: true while it minus the buffer dec is greater than zero.
                return it - self.dec > 0

            def body_fn(it, x):
                # [FUNCTION DOC] HELPER: decrements the counter, runs the inner module on the
                #   pair, takes its output tensor and applies the outer linear layer.
                return it - 1, self.outer_linear(self.mod(it, x)[1])

            return while_loop(cond_fn, body_fn, (iter, x))

    class PytreeIntCarry(torch.nn.Module):
        # [CLASS DOC] HELPER: module whose while_loop carries Python ints and symbolic shape sizes
        #   inside a pytree beside a tensor.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: carries the two leading sizes of x, a dict of three int
            #   constants and sine of x through a while_loop. Afterwards maps over the int dict to
            #   add one, and to add the final tensor. Returns (final shapes, incremented ints,
            #   ints plus tensor, final tensor).
            a = x.shape[0]
            b = x.shape[1]

            def cond_fn(shapes, const_int_dict, x):
                # [FUNCTION DOC] HELPER: true while the product of the three int constants is
                #   below the product of the two size entries.
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return c1 * c2 * c3 < a * b

            def body_fn(shapes, const_int_dict, x):
                # [FUNCTION DOC] HELPER: increments both sizes and all three int constants and
                #   adds one to x; returns the same pytree structure.
                a, b = shapes
                c1, c2, c3 = const_int_dict["int_carry"]
                return (
                    [a + 1, b + 1],
                    {"int_carry": (c1 + 1, c2 + 1, c3 + 1)},
                    x + 1,
                )

            carry = ([a, b], {"int_carry": (2, 2, 3)}, x.sin())
            out_shapes, out_it, out_x = while_loop(cond_fn, body_fn, carry)
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + out_x, out_it)
            return (out_shapes, out_inc, out_add, out_x)

    class IntCarry(torch.nn.Module):
        # [CLASS DOC] HELPER: module whose while_loop carry is a plain int counter plus a tensor,
        #   invoked through the higher-order op directly.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: calls the while_loop higher-order op directly with initial
            #   counter zero and x. The body writes the counter value added to row it of a clone
            #   of x. Returns four values: final counter plus one, counter plus the tensor,
            #   whether the counter is below the leading size, and a ones tensor of size twice the
            #   counter (guarded by a check that the counter is positive).
            def cond_fn(it, x):
                # [FUNCTION DOC] HELPER: true while the counter is less than the leading
                #   dimension of x.
                return it < x.shape[0]

            def body_fn(it, x):
                # [FUNCTION DOC] HELPER: asserts via torch._check the counter is in range, adds
                #   the counter to the selected row of a clone of x in place, and returns counter
                #   plus one with the clone.
                x_clone = x.clone()
                # Need these checks to select from x
                torch._check(it >= 0)
                torch._check(it < x.shape[0])
                x_clone.select(0, it).copy_(x_clone.select(0, it) + it)
                return it + 1, x_clone

            # We invoke the hop directly to avoid triggering dyanmo tracing
            out_it, out_x = torch.ops.higher_order.while_loop(
                cond_fn, body_fn, (0, x), tuple()
            )
            # We need torch._check to use it in torch.ones call
            torch._check(out_it > 0)
            return (
                out_it + 1,
                out_it + out_x,
                out_it < x.shape[0],
                torch.ones(out_it * 2),
            )

    class ConstAndSymIntOutput(torch.nn.Module):
        # [CLASS DOC] HELPER: module testing while_loop whose carries mix symbolic ints, constant
        #   ints and a data-dependent int.
        def forward(self, t):
            # [FUNCTION DOC] HELPER: builds a carry of the two sizes, three int ones, size0
            #   plus one, the integer sum of t via item, and sine of t; runs the while_loop
            #   higher-order op directly; returns (result plus one, result plus t) mapped over
            #   the output pytree.
            a = t.shape[0]
            b = t.shape[1]

            def cond_fn(a, b, c1, c2, c3, c0, u0, x):
                # [FUNCTION DOC] HELPER: true while the product of c1, c2 and c3 is below the
                #   product of a and b.
                return c1 * c2 * c3 < a * b

            def body_fn(a, b, c1, c2, c3, c0, u0, x):
                # [FUNCTION DOC] HELPER: rotates the carries: returns b, c1, c2, c3, a, constant
                #   zero, u0 plus one, and x plus one.
                return b, c1, c2, c3, a, 0, u0 + 1, x + 1

            carry = (a, b, 1, 1, 1, a + 1, t.sum().to(torch.int64).item(), t.sin())
            out_it = torch.ops.higher_order.while_loop(cond_fn, body_fn, carry, tuple())
            out_inc = pytree.tree_map(lambda x: x + 1, out_it)
            out_add = pytree.tree_map(lambda x: x + t, out_it)
            return out_inc, out_add

    nested2 = Nested()
    simple_with_linear = SimpleWithLinear()
    simple_with_pytree_carry = SimpleWithPytreeCarry()
    nested_with_linear = NestedWithLinear()
    int_carry = IntCarry()
    pytree_int_carry = PytreeIntCarry()
    const_and_symint_output = ConstAndSymIntOutput()

    x = torch.zeros(1)
    y = torch.zeros(1)
    z = torch.zeros(1)
    return {
        "simple": (simple, (x,)),
        "nested": (nested, (x, y, z)),
        "nested2": (
            nested2,
            (torch.tensor(2), torch.tensor(2), torch.ones(2, 2), torch.ones(2, 2)),
        ),
        "simple_with_mutation": (simple_with_mutation, (x,)),
        "simple_with_linear": (
            simple_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "nested_with_linear": (
            nested_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "simple_with_pytree_carry": (
            simple_with_pytree_carry,
            (
                torch.tensor(3),
                ([torch.randn(3, 3)], {"x": torch.randn(3, 3), "y": torch.randn(3, 3)}),
            ),
        ),
        "int_carry": (int_carry, (torch.randn(2, 3),)),
        "pytree_int_carry": (
            pytree_int_carry,
            (torch.randn(2, 3),),
        ),
        "const_and_symint_output": (
            const_and_symint_output,
            (torch.randn(2, 3),),
        ),
    }


WHILE_LOOP_TESTS = _while_loop_tests()


def collect_meta_for_filtered_nodes(
    gm: torch.fx.GraphModule, node_names, meta_field_name
):
    # [FUNCTION DOC] HELPER: walks every submodule of the graph module and, for each graph node
    #   whose name is in node_names, appends the value of each requested meta field name (None
    #   when missing). Returns the flat list.
    ret = []
    for mod in gm.modules():
        for node in mod.graph.nodes:
            if node.name in node_names:
                for field_name in meta_field_name:
                    ret.append(node.meta.get(field_name))
    return ret


def reduce_func(*operands):
    # [FUNCTION DOC] HELPER: returns the sum of all positional operands, starting
    #   from integer zero.
    acc = 0
    for operand in operands:
        acc += operand
    return acc


class ReduceObj:
    # [CLASS DOC] HELPER: callable object wrapping reduce_func, used to
    #   test callable-instance branches.
    def __call__(self, *operands):
        # [FUNCTION DOC] HELPER: sums the operands via reduce_func.
        return reduce_func(*operands)


class ReduceMod(torch.nn.Module):
    # [CLASS DOC] HELPER: nn.Module wrapper around reduce_func, used to test modules
    #   as branch callables.
    def _reduce(self, *operands):
        # [FUNCTION DOC] HELPER: sums the operands via reduce_func.
        return reduce_func(*operands)

    def forward(self, *operands):
        # [FUNCTION DOC] HELPER: delegates to _reduce.
        return self._reduce(*operands)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class TestControlFlow(TestCase):
    # [CLASS DOC] Main test suite for cond, while_loop, scan, associative_scan, map and related
    #   higher-order ops. Skipped on Windows and when dynamo is unsupported. Methods compare
    #   eager, compiled and traced results, check autograd and compare golden graph text. setUp
    #   resets dynamo; check_autograd is a shared gradient comparer.
    def setUp(self):
        # [FUNCTION DOC] Resets torch._dynamo state, then runs the base TestCase setUp.
        torch._dynamo.reset()
        super().setUp()

    def check_autograd(self, result, result_exp, params):
        # [FUNCTION DOC] HELPER: flattens params, result and result_exp as pytrees, computes
        #   gradients of the expected and actual results with respect to params using all-
        #   ones output gradients, and asserts the two gradient lists are equal with atol
        #   6e-05 and rtol 6e-06.
        params_flatten = pytree.tree_leaves(params)
        result_flatten = pytree.tree_leaves(result)
        result_exp_flatten = pytree.tree_leaves(result_exp)
        grad_exp_init = [torch.ones_like(el) for el in result_exp_flatten]
        expected_grads = torch.autograd.grad(
            result_exp_flatten, params_flatten, grad_exp_init
        )
        grad_init = [torch.ones_like(el) for el in result_flatten]
        grads = torch.autograd.grad(result_flatten, params_flatten, grad_init)
        self.assertEqual(grads, expected_grads, atol=6e-05, rtol=6e-06)

    def test_cond_no_trace(self):
        # [FUNCTION DOC] Eager cond with a Python bool False pred: checks the false branch
        #   (cosine) result is returned.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns the sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns the cosine of x.
            return x.cos()

        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_gpu(self):
        # [FUNCTION DOC] CUDA-only test: cond with a CUDA boolean tensor predicate False and CUDA
        #   operand; asserts the result equals the cosine of x.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns the sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns the cosine of x.
            return x.cos()

        x = torch.randn(4, device="cuda")
        pred = torch.tensor(False, device="cuda")
        result = cond(pred, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    def test_cond_autograd_simple(self):
        # [FUNCTION DOC] Eager cond for both predicate values with an input requiring grad: checks
        #   the result and the gradient against calling the selected branch directly. Then traces
        #   a function that runs cond and autograd.grad with make_fx and compares the golden
        #   graph, which contains two cond ops (forward, backward) with their branch subgraphs.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns the sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns the cosine of x.
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond on the predicate and x,
            #   then returns the gradient of the result with respect to x using an all-
            #   ones output gradient.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",
        )

    def test_cond_autograd_complex(self):
        # [FUNCTION DOC] Same as the simple autograd cond test but with more involved branches
        #   (abs of sine of x squared, and cosine of x plus 42): checks forward and gradients for
        #   both predicates and the make_fx golden graph.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns the absolute value of the sine of x squared.
            return torch.abs((x**2).sin())

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns the cosine of x plus 42.
            return (x + 42).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond then returns gradient with respect
            #   to x using ones as the output gradient.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_nested(self):
        # [FUNCTION DOC] Skipped under dynamo (graph break). Defines a module with three levels of
        #   nested torch.cond selected by predicates p0, p1, p2 over three tensors and checks
        #   nested cond forward and gradient behavior.
        class Nested(torch.nn.Module):
            # [CLASS DOC] HELPER: module nesting three levels of cond.
            def forward(self, p0, p1, p2, a, b, c):
                # [FUNCTION DOC] HELPER: selects true_fn or false_fn by p0 and, inside, by p1 and
                #   then p2, each leaf being a small arithmetic expression of the three inputs;
                #   returns the cond result.
                def true_fn(x0, y0, z0):
                    # [FUNCTION DOC] HELPER: when p0 is true, dispatches on p1 between
                    #   true_true_fn and true_false_fn.
                    def true_true_fn(x1, y1, z1):
                        # [FUNCTION DOC] HELPER: returns (x1 minus y1 times z1) times 3.14.
                        return (x1 - y1 * z1) * 3.14

                    def true_false_fn(x1, y1, z1):
                        # [FUNCTION DOC] HELPER: dispatches on p2 between the two
                        #   leaf functions below.
                        def true_false_true_fn(x2, y2, z2):
                            # [FUNCTION DOC] HELPER: returns x2 times y2 times z2 divided by 2.71.
                            return (x2 * y2 * z2) / 2.71

                        def true_false_false_fn(x2, y2, z2):
                            # [FUNCTION DOC] HELPER: returns (x2 plus y2 plus z2) times 1.23.
                            return (x2 + y2 + z2) * 1.23

                        return torch.cond(
                            p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                        )

                    return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

                def false_fn(x0, y0, z0):
                    # [FUNCTION DOC] HELPER: when p0 is false, dispatches on p1 between
                    #   false_true_fn and false_false_fn.
                    def false_true_fn(x1, y1, z1):
                        # [FUNCTION DOC] HELPER: dispatches on p2 between the two
                        #   leaf functions below.
                        def false_true_true_fn(x2, y2, z2):
                            # [FUNCTION DOC] HELPER: returns x2 minus y2 minus z2, plus 1.23.
                            return (x2 - y2 - z2) + 1.23

                        def false_true_false_fn(x2, y2, z2):
                            # [FUNCTION DOC] HELPER: returns x2 divided by y2 divided by
                            #   z2, minus 3.14.
                            return (x2 / y2 / z2) - 3.14

                        return torch.cond(
                            p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                        )

                    def false_false_fn(x1, y1, z1):
                        # [FUNCTION DOC] HELPER: returns (x1 minus y1 times z1) divided by 2.71.
                        return (x1 - y1 * z1) / 2.71

                    return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

                return torch.cond(p0, true_fn, false_fn, [a, b, c])

        nn_module = Nested()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: calls the nested module with predicate tensors false, true,
            #   false and x for all three tensor inputs.
            return nn_module(
                torch.tensor(False), torch.tensor(True), torch.tensor(False), x, x, x
            )

        def false_fn(x):
            # [FUNCTION DOC] HELPER: calls the nested module with predicate tensors true, false,
            #   true and x for all three tensor inputs.
            return nn_module(
                torch.tensor(True), torch.tensor(False), torch.tensor(True), x, x, x
            )

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_mixed_require_grad(self):
        # [FUNCTION DOC] Skipped under dynamo. Runs cond with operands where only some require
        #   grad (x requires grad, y does not) and checks result and gradient with respect to x
        #   for both predicates. Then traces forward plus backward with make_fx and compares the
        #   golden graph, in which the non-grad input is still passed to the branches.
        def true_fn(x, y, z):
            # [FUNCTION DOC] HELPER: returns the product of the three inputs.
            return x * y * z

        def false_fn(x, y, z):
            # [FUNCTION DOC] HELPER: returns the sum of the three inputs.
            return x + y + z

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=False)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x, y, x))
            self.assertEqual(result, fn(x, y, x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x, y, x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x, y, z):
            # [FUNCTION DOC] HELPER: traced function; runs cond over the three operands and
            #   returns the gradient of the result with respect to the first operand only.
            result = cond(pred, true_fn, false_fn, (x, y, z))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x, y, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1, y_1, z_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (z_1, y_1));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (z_1, y_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = z_1 = y_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  cond_1 = getitem_2 = None
    return (getitem_1,)""",
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_grad_through_cond(self):
        # [FUNCTION DOC] Skipped under dynamo. Checks gradients with respect to the weight of a
        #   Linear module captured by the branches, for both predicate values, and traces the
        #   backward with symbolic make_fx (non-fake inputs allowed because module parameters
        #   are not fakified), comparing the golden graph in which parameters are passed as
        #   extra cond operands.
        nn_module = torch.nn.Linear(4, 4)

        def true_fn(x):
            # [FUNCTION DOC] HELPER: applies the captured linear module to x.
            return nn_module(x)

        def false_fn(X):
            # [FUNCTION DOC] HELPER: returns x times the linear module output.
            # KNOWN ISSUE: the parameter is named X (capital) but the body uses x from the
            #   enclosing scope, so it works only through closure capture of the outer x.
            return x * nn_module(x)

        x = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (nn_module.weight,), grad_out)
            expected_grads = torch.autograd.grad(
                fn(
                    x,
                ),
                (nn_module.weight,),
                grad_out,
            )
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond and returns the gradient of the
            #   result with respect to the module weight using ones as output gradient.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (nn_module.weight,), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _tensor_constant0 = self._tensor_constant0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_param_constant0, _param_constant1, x_1, sym_size_int, _tensor_constant0));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _tensor_constant0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _tensor_constant0_1 = self._tensor_constant0
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_param_constant0_1, _param_constant1_1, x_1, sym_size_int, _tensor_constant0_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _param_constant0_1 = _param_constant1_1 = x_1 = sym_size_int = _tensor_constant0_1 = ones_like = None
    getitem_1 = cond_1[0];  getitem_1 = None
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  cond_1 = getitem_4 = None
    return (getitem_2,)""",
        )

    def test_cond_in_forloop(self):
        # [FUNCTION DOC] Unrolls three cond calls in a Python loop (always the true branch because
        #   the predicate is a constant true bool) and compares the result and the input gradient
        #   with a plain loop of squaring plus one. Then traces forward plus backward in symbolic
        #   mode and compares the golden graph, in which cond has been resolved away into
        #   straight-line aten ops.
        def for_loop_fake(x):
            # [FUNCTION DOC] HELPER: reference; three times replaces x by x squared plus one.
            for _ in range(3):
                x = x * x + 1
            return x

        def for_loop_test(x):
            # [FUNCTION DOC] HELPER: three iterations each applying cond with a Python bool
            #   predicate (always true here) choosing between square-plus-one and identity.
            for i in range(3):
                pred = i < 3

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: returns x squared plus one.
                    return x * x + 1

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: returns x unchanged.
                    return x

                x = cond(pred, true_fn, false_fn, (x,))

            return x

        x = torch.ones(4, requires_grad=True)
        x_new = for_loop_test(x)
        x_exp = for_loop_fake(x)

        self.assertEqual(x_new, x_exp)

        grad_out = torch.ones_like(x_new)
        grads = torch.autograd.grad(x_new, (x,), grad_out)
        expected_grads = torch.autograd.grad(x_exp, (x,), grad_out)
        self.assertEqual(expected_grads, grads)

        def f(x):
            # [FUNCTION DOC] HELPER: traced function; runs the cond loop and returns gradient of
            #   its result with respect to x with ones as output gradient.
            x_new = for_loop_test(x)
            grad_out = torch.ones_like(x_new)
            return torch.autograd.grad(x_new, (x,), grad_out)

        gm = make_fx(f, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    mul = torch.ops.aten.mul.Tensor(x_1, x_1)
    add = torch.ops.aten.add.Tensor(mul, 1);  mul = None
    mul_1 = torch.ops.aten.mul.Tensor(add, add)
    add_1 = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(add_1, add_1)
    add_2 = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
    ones_like = torch.ops.aten.ones_like.default(add_2, pin_memory = False);  add_2 = None
    mul_3 = torch.ops.aten.mul.Tensor(ones_like, add_1)
    mul_4 = torch.ops.aten.mul.Tensor(ones_like, add_1);  ones_like = add_1 = None
    add_3 = torch.ops.aten.add.Tensor(mul_4, mul_3);  mul_4 = mul_3 = None
    mul_5 = torch.ops.aten.mul.Tensor(add_3, add)
    mul_6 = torch.ops.aten.mul.Tensor(add_3, add);  add_3 = add = None
    add_4 = torch.ops.aten.add.Tensor(mul_6, mul_5);  mul_6 = mul_5 = None
    mul_7 = torch.ops.aten.mul.Tensor(add_4, x_1)
    mul_8 = torch.ops.aten.mul.Tensor(add_4, x_1);  add_4 = x_1 = None
    add_5 = torch.ops.aten.add.Tensor(mul_8, mul_7);  mul_8 = mul_7 = None
    return (add_5,)""",
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_pytree_not_all_inputs_used(self):
        # [FUNCTION DOC] Skipped under dynamo. Operand is a nested dict/list/tuple pytree where
        #   the true branch ignores the third tensor. For the true predicate it expects taking
        #   gradients with respect to all three leaves to raise an exception (unused input).
        #   Traces the gradient wrt two leaves and compares golden graphs for the outer graph plus
        #   the true branch forward and backward subgraphs.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: adds the first list entry and the "b" entry of the second
            #   entry (ignores the third).
            return x["t"][0] + x["t"][1]["b"]

        def false_fn(x):
            # [FUNCTION DOC] HELPER: first entry times (the third entry's first element
            #   divided by "b").
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            if pred:
                with self.assertRaisesRegex(Exception, r"."):
                    grads = torch.autograd.grad(result, (a, b, c), grad_out)
                    expected_grads = torch.autograd.grad(
                        fn({"t": [a, {"b": b}, (c,)]}), (a, b, c), grad_out
                    )
                    self.assertEqual(expected_grads, grads)

        def f(pred, a, b, c):
            # [FUNCTION DOC] HELPER: traced function; runs cond on the pytree and returns gradient
            #   with respect to a and b.
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            pred, a, b, c
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, a_1, b_1, c_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(b_1, 0)
    sym_size_int_2 = torch.ops.aten.sym_size.int(c_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (a_1, b_1, sym_size_int, sym_size_int_1, c_1, sym_size_int_2, ones_like));  pred_1 = true_graph_1 = false_graph_1 = a_1 = b_1 = sym_size_int = sym_size_int_1 = c_1 = sym_size_int_2 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",
        )
        # Forward
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )
        # Backward
        self.assertExpectedInline(
            gm.true_graph_1.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = add = None
    zeros_like = torch.ops.aten.zeros_like.default(arg4_1, pin_memory = False);  arg4_1 = None
    clone = torch.ops.aten.clone.default(arg6_1)
    clone_1 = torch.ops.aten.clone.default(arg6_1);  arg6_1 = None
    return [clone, clone_1, zeros_like]""",
        )

    def test_cond_autograd_pytree_input(self):
        # [FUNCTION DOC] Checks cond with a pytree operand (dict containing list, dict and tuple
        #   of tensors) for both predicates: result and gradients with respect to a and b match
        #   direct calls. Traces gradients in symbolic mode with captured tensors and compares
        #   the golden graph.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: first entry plus "b" times the third entry's element.
            return x["t"][0] + x["t"][1]["b"] * x["t"][2][0]

        def false_fn(x):
            # [FUNCTION DOC] HELPER: first entry times (third entry's element divided by "b").
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            self.assertEqual(result, fn({"t": [a, {"b": b}, (c,)]}))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (a, b), grad_out)
            expected_grads = torch.autograd.grad(
                fn({"t": [a, {"b": b}, (c,)]}), (a, b), grad_out
            )
            self.assertEqual(expected_grads, grads)

        def f(pred):
            # [FUNCTION DOC] HELPER: traced function; closes over a, b, c; runs cond on the
            #   predicate and returns gradients with respect to a and b.
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (a, b), grad_out)

        # need to set _allow_non_fake_inputs = True because model parameters don't
        # get fakified.
        gm = make_fx(f, tracing_mode="symbolic", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _tensor_constant0_1 = self._tensor_constant0
    _tensor_constant1_1 = self._tensor_constant1
    _tensor_constant2_1 = self._tensor_constant2
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (_tensor_constant0_1, _tensor_constant1_1, _tensor_constant2_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = _tensor_constant0_1 = _tensor_constant1_1 = _tensor_constant2_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1]
    getitem_3 = cond_1[2];  cond_1 = getitem_3 = None
    return (getitem_1, getitem_2)""",
        )

    def test_cond_autograd_different_pytree_output(self):
        # [FUNCTION DOC] Branches return different output pytree structures (tuple with dict and
        #   list versus a dict of list); asserts cond raises UncapturedHigherOrderOpError
        #   mentioning the torch.cond operator for both predicate values.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns a tuple of: first entry, a dict holding third element
            #   divided by b, and a list with the third element.
            return x["t"][0], {"r": x["t"][2][0] / x["t"][1]["b"]}, [x["t"][2][0]]

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a dict with a list holding first times b and
            #   the third element.
            return {"res": [x["t"][0] * x["t"][1]["b"], x["t"][2][0]]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred in [torch.tensor(False), torch.tensor(True)]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_same_pytree_output(self):
        # [FUNCTION DOC] Skipped under dynamo. Branches return the same dict-of-list-and-tuple
        #   structure of cloned tensors; checks values and gradients wrt c for both predicates,
        #   then traces forward only with real tracing and compares the golden graph, which
        #   returns the rebuilt dict.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns a dict with a list of clones of the first entry and a
            #   tuple of a clone of the third element.
            return {"res": [x["t"][0].clone(), (x["t"][2][0].clone(),)]}

        def false_fn(x):
            # [FUNCTION DOC] HELPER: like true_fn but the first list entry is a clone of "b".
            return {"res": [x["t"][1]["b"].clone(), (x["t"][2][0].clone(),)]}

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            result_exp = fn({"t": [a, {"b": b}, (c,)]})
            self.assertEqual(result, result_exp)

            result_flat, _ = pytree.tree_flatten(result)
            result_exp_flat, _ = pytree.tree_flatten(result_exp)

            grad_out = [torch.ones_like(g) for g in result_flat]
            expected_grads = torch.autograd.grad(result_exp_flat, (c,), grad_out)
            grads = torch.autograd.grad(result_flat, (c,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred):
            # [FUNCTION DOC] HELPER: traced function; runs cond on the pytree of closed-over
            #   tensors and returns the result.
            result = cond(pred, true_fn, false_fn, ({"t": [a, {"b": b}, (c,)]},))
            return result

        gm = make_fx(f, tracing_mode="real", _allow_non_fake_inputs=True)(pred)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    _tensor_constant2 = self._tensor_constant2
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (_tensor_constant0, _tensor_constant1, _tensor_constant2));  pred_1 = true_graph_0 = false_graph_0 = _tensor_constant0 = _tensor_constant1 = _tensor_constant2 = None
    getitem = cond[0]
    getitem_1 = cond[1];  cond = None
    return {'res': [getitem, (getitem_1,)]}""",
        )

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_cond_autograd_torch_nn_module(self):
        # [FUNCTION DOC] Skipped under dynamo. Branches use a Linear module and a GRUCell
        #   module; checks results and input gradients for both predicates, then traces
        #   forward and backward and compares a golden graph where all six module parameters
        #   are passed as operands.
        nn_module_true = torch.nn.Linear(4, 4)

        def true_fn(x):
            # [FUNCTION DOC] HELPER: applies the Linear module to abs of sine of x squared.
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.GRUCell(4, 4)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: applies the GRUCell module to cosine of x plus 42.
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond and returns the gradient wrt x.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _param_constant0 = self._param_constant0
    _param_constant1 = self._param_constant1
    _param_constant2 = self._param_constant2
    _param_constant3 = self._param_constant3
    _param_constant4 = self._param_constant4
    _param_constant5 = self._param_constant5
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1, _param_constant0, _param_constant1, _param_constant2, _param_constant3, _param_constant4, _param_constant5));  true_graph_0 = false_graph_0 = _param_constant0 = _param_constant1 = _param_constant2 = _param_constant3 = _param_constant4 = _param_constant5 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    _param_constant0_1 = self._param_constant0
    _param_constant1_1 = self._param_constant1
    _param_constant2_1 = self._param_constant2
    _param_constant3_1 = self._param_constant3
    _param_constant4_1 = self._param_constant4
    _param_constant5_1 = self._param_constant5
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, _param_constant0_1, _param_constant1_1, _param_constant2_1, _param_constant3_1, _param_constant4_1, _param_constant5_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = _param_constant0_1 = _param_constant1_1 = _param_constant2_1 = _param_constant3_1 = _param_constant4_1 = _param_constant5_1 = ones_like = None
    getitem_1 = cond_1[0]
    getitem_2 = cond_1[1];  getitem_2 = None
    getitem_3 = cond_1[2];  getitem_3 = None
    getitem_4 = cond_1[3];  getitem_4 = None
    getitem_5 = cond_1[4];  getitem_5 = None
    getitem_6 = cond_1[5];  getitem_6 = None
    getitem_7 = cond_1[6];  cond_1 = getitem_7 = None
    return (getitem_1,)""",
        )

    def test_cond_autograd_user_nn_module(self):
        # [FUNCTION DOC] Same autograd checks as the nn module test, with a user-defined squaring
        #   module in the true branch and a ReLU module in the false branch; compares the golden
        #   make_fx graph (no parameters, so only x and the gradient are passed).
        class User_nn_module(torch.nn.Module):
            # [CLASS DOC] HELPER: parameterless module that squares its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: only calls the base class initializer.
                super().__init__()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: returns input times input.
                return input * input

        nn_module_true = User_nn_module()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: applies the squaring module to abs of sine of x squared.
            return nn_module_true(torch.abs((x**2).sin()))

        nn_module_false = torch.nn.ReLU(inplace=False)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: applies the ReLU module to cosine of x plus 42.
            return nn_module_false((x + 42).cos())

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond and returns the gradient wrt x.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",
        )

    def test_cond_autograd_inner_fn(self):
        # [FUNCTION DOC] Checks that a branch containing an inner helper function behaves like one
        #   without: forward and gradient for false then true predicate, equality of both results
        #   and gradients, plus the golden make_fx graph.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: abs of sine of x squared.
            return torch.abs((x**2).sin())

        def false_fn(x):
            # [FUNCTION DOC] HELPER: same math as true_fn, computed through an inner function.
            def inner_fn(x):
                # [FUNCTION DOC] HELPER: returns x squared.
                return x**2

            return torch.abs(inner_fn(x).sin())

        x = torch.randn(4, requires_grad=True)
        pred = torch.tensor(False)
        fn = false_fn
        result_false = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_false, fn(x))

        grad_out = torch.ones_like(result_false)
        grads_false = torch.autograd.grad(result_false, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_false)

        pred = torch.tensor(True)
        fn = true_fn
        result_true = cond(pred, true_fn, false_fn, (x,))
        self.assertEqual(result_true, fn(x))
        self.assertEqual(result_false, result_true)

        grad_out = torch.ones_like(result_true)
        grads_true = torch.autograd.grad(result_true, (x,), grad_out)
        expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads_true)
        self.assertEqual(grads_false, grads_true)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond and returns the gradient wrt x.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",
        )

    def test_cond_autograd_inner_tensor(self):
        # [FUNCTION DOC] Checks autograd through a branch that creates its own tensor (ones times
        #   42, no grad) inside the branch; forward and gradient for both predicates plus the
        #   golden make_fx graph.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: abs of sine of x squared.
            return torch.abs((x**2).sin())

        def false_fn(x):
            # [FUNCTION DOC] HELPER: builds a constant tensor of 42s and returns the cosine of
            #   x times it.
            y = torch.ones(4, requires_grad=False) * 42
            return (x * y).cos()

        for pred, fn in zip(
            [torch.tensor(False), torch.tensor(True)], [false_fn, true_fn]
        ):
            x = torch.randn(4, requires_grad=True)
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: traced function; runs cond and returns the gradient wrt x.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        gm = make_fx(f)(pred, x)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    ones_like = torch.ops.aten.ones_like.default(getitem, pin_memory = False);  getitem = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred_1, true_graph_1, false_graph_1, (x_1, ones_like));  pred_1 = true_graph_1 = false_graph_1 = x_1 = ones_like = None
    getitem_1 = cond_1[0];  cond_1 = None
    return (getitem_1,)""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_cond_autograd_gpu(self):
        # [FUNCTION DOC] CUDA-only: cond with CUDA boolean predicates and a CUDA input requiring
        #   grad; checks result and gradient against calling the branch directly.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns cosine of x.
            return x.cos()

        for pred, fn in zip(
            [torch.tensor(False, device="cuda"), torch.tensor(True, device="cuda")],
            [false_fn, true_fn],
        ):
            x = torch.randn(4, requires_grad=True, device="cuda")
            result = cond(pred, true_fn, false_fn, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def _test_cond_autograd(self, cond_fct, pred_fn, true_fn, false_fn, operands):
        # [FUNCTION DOC] HELPER: shared checker. Runs the given cond function on the predicate
        #   computed from operands, compares with the branch executed on grad-enabled detached
        #   clones, then (if any operand requires grad) compares gradients (unused ones
        #   materialized as zeros) and checks that tensor metadata of each gradient equals the
        #   operand metadata ignoring requires_grad. Returns the cond outputs and the operands
        #   that require grad.
        from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata

        # This is a helper function that extracts the metadata from the tensor and
        # sets the requires_grad flag to false. This is needed as we compare the
        # metadata of the operands and the gradients
        def _extract_tensor_metadata_except_requires_grad(arg):
            # [FUNCTION DOC] HELPER: returns the fx tensor metadata of the argument with the
            #   requires_grad field forced to false so operands and gradients can be compared.
            metadata = _extract_tensor_metadata(arg)
            metadata = TensorMetadata(
                metadata.shape,
                metadata.dtype,
                False,
                metadata.stride,
                metadata.memory_format,
                metadata.is_quantized,
                metadata.qparams,
            )
            return metadata

        # Comparison of FWD path
        cond_outputs = cond_fct(pred_fn(*operands), true_fn, false_fn, operands)
        operands_forced_grad = [o.clone().detach() for o in operands]
        for o in operands_forced_grad:
            o.requires_grad = True
        cond_outputs_exp = (
            true_fn(*operands_forced_grad)
            if pred_fn(*operands_forced_grad)
            else false_fn(*operands_forced_grad)
        )
        self.assertEqual(cond_outputs, cond_outputs_exp)

        # Comparison of BWD path
        cond_inputs = [o for o in operands if o.requires_grad]
        cond_inputs_exp = [o for o in operands_forced_grad if o.requires_grad]

        # Check if at least some operators require grads
        if len(cond_inputs) > 0:
            grad_inputs = torch.autograd.grad(
                cond_outputs, cond_inputs, allow_unused=True, retain_graph=True
            )
            grad_inputs_exp = torch.autograd.grad(
                cond_outputs_exp,
                cond_inputs_exp,
                allow_unused=True,
                materialize_grads=True,
            )

            grad_exp_masked = [
                g for g, o in zip(grad_inputs_exp, operands) if o.requires_grad
            ]
            self.assertEqual(grad_exp_masked, grad_inputs)

            # Extraction and comparison of Metadata of operands and gradients
            operands_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in cond_inputs
            ]
            grad_metadata = [
                _extract_tensor_metadata_except_requires_grad(o) for o in grad_inputs
            ]
            self.assertTrue(
                all(op == g for op, g in zip(operands_metadata, grad_metadata))
            )

        return cond_outputs, cond_inputs

    @skipIfTorchDynamo("don't test compile on compile")
    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["compile_dynamic_shape"])
    @parametrize("scalar", [False])
    def test_cond_autograd_zeros_unused_branch_complex_compile_fail(
        self, compile_mode, scalar
    ):
        # [FUNCTION DOC] Skipped under dynamo, without SM70 or CUDA. Parametrized with compile
        #   mode compile_dynamic_shape and scalar False. Builds CUDA operands (non-scalar
        #   shapes in this configuration, labeled as not working), uses a dynamic-shape
        #   compiled cond and runs the shared autograd checker with matrix-multiply branches.
        #   It has no explicit failure assertion; the name suggests the non-scalar case is a
        #   known failing configuration.
        device = torch.device("cuda")
        cond_fct = compile_mode_helper(torch.cond, compile_mode)

        autograd = [False, True, True, True, True]

        if scalar:
            # These operands work
            x = torch.randn((), device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn((), device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn((), device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn((), device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn((), device=device, requires_grad=bool(autograd[4]))
        else:
            # These operands do not work
            x = torch.randn(4, 5, device=device, requires_grad=bool(autograd[0]))
            w1 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[1]))
            b1 = torch.randn(2, 1, device=device, requires_grad=bool(autograd[2]))
            w2 = torch.randn(2, 4, device=device, requires_grad=bool(autograd[3]))
            b2 = torch.randn(1, 5, device=device, requires_grad=bool(autograd[4]))

        operands = [x, w1, b1, w2, b2]

        def true_fn(x, w1, b1, w2, b2):
            # [FUNCTION DOC] HELPER: for scalar mode returns a one-tuple of w1 times x plus b1,
            #   otherwise a one-tuple with the sum of the matrix product of w1 and x plus b1.
            if scalar:
                # This works
                return ((w1 * x + b1),)
            else:
                # This does not work
                return ((w1 @ x + b1).sum(),)

        def false_fn(x, w1, b1, w2, b2):
            # [FUNCTION DOC] HELPER: same as true_fn but using w2 and b2.
            if scalar:
                # This works
                return ((w2 * x + b2),)
            else:
                # This does not work
                return ((w2 @ x + b2).sum(),)

        def pred_fn(x, w1, b1, w2, b2):
            # [FUNCTION DOC] HELPER: predicate; true if the mean of x is greater than zero.
            return x.mean() > 0

        cond_outputs, cond_inputs = self._test_cond_autograd(
            cond_fct, pred_fn, true_fn, false_fn, operands
        )

    def test_switch_basic(self):
        # [FUNCTION DOC] Eager switch over three branches (zeros, ones, twos): integer and single-
        #   element tensor indices, with out-of-range indices clamped to the first or last branch.
        x = torch.tensor([0, 1, 2])
        branches = (
            lambda inp_x: torch.zeros_like(inp_x),
            lambda inp_x: torch.ones_like(inp_x),
            lambda inp_x: 2 * torch.ones_like(inp_x),
        )

        # integer indices (including clamped out-of-range)
        result = switch(-1, branches, (x,))
        self.assertEqual(result, torch.zeros_like(x))

        result = switch(0, branches, (x,))
        self.assertEqual(result, torch.zeros_like(x))

        result = switch(1, branches, (x,))
        self.assertEqual(result, torch.ones_like(x))

        result = switch(2, branches, (x,))
        self.assertEqual(result, 2 * torch.ones_like(x))

        result = switch(3, branches, (x,))
        self.assertEqual(result, 2 * torch.ones_like(x))

        # tensor indices
        result = switch(torch.tensor([-1]), branches, (x,))
        self.assertEqual(result, torch.zeros_like(x))

        result = switch(torch.tensor([0]), branches, (x,))
        self.assertEqual(result, torch.zeros_like(x))

        result = switch(torch.tensor([1]), branches, (x,))
        self.assertEqual(result, torch.ones_like(x))

        result = switch(torch.tensor([2]), branches, (x,))
        self.assertEqual(result, 2 * torch.ones_like(x))

        result = switch(torch.tensor([3]), branches, (x,))
        self.assertEqual(result, 2 * torch.ones_like(x))

    def test_switch_zero_args(self):
        # [FUNCTION DOC] Switch with branches that take no operands and an empty operand list;
        #   checks each index runs the matching branch.
        branches = (
            lambda: torch.zeros(3),
            lambda: torch.ones(3),
            lambda: 2.0 * torch.ones(3),
        )

        for i in range(3):
            self.assertEqual(switch(i, branches, []), branches[i]())

    def test_switch_multiple_args(self):
        # [FUNCTION DOC] Switch with two operands; each index gives the same result as calling the
        #   matching branch directly.
        x = torch.ones(3)
        y = torch.tensor([1.0, 2.0, 3.0])
        branches = (
            lambda inp_x, inp_y: inp_x + inp_y,
            lambda inp_x, inp_y: inp_x * inp_y,
            lambda inp_x, inp_y: inp_x - 0.5 * inp_y,
        )

        for i in range(3):
            self.assertEqual(switch(i, branches, (x, y)), branches[i](x, y))

    def test_switch_invalid_index(self):
        # [FUNCTION DOC] Checks switch raises RuntimeError for a list index ("Expected
        #   index to be") and for a two-element tensor index ("Expected index to be int
        #   or single-element tensor").
        x = torch.ones(3)
        branches = (torch.sin, torch.cos)

        with self.assertRaisesRegex(RuntimeError, "Expected index to be"):
            switch([0, 1], branches, (x,))

        with self.assertRaisesRegex(
            RuntimeError, "Expected index to be int or single-element tensor"
        ):
            switch(torch.tensor([0, 1]), branches, (x,))

    def test_switch_zero_branch(self):
        # [FUNCTION DOC] Checks that an empty branch list raises RuntimeError saying branches must
        #   be a non-empty tuple or list of callables.
        x = torch.ones(3)

        # Empty-branch rejection happens unconditionally at the top of
        # torch.switch, matching jax.lax.switch — both int and tensor
        # indices surface the same RuntimeError.
        idx = 0
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected branches to be a non-empty tuple or list of callables",
        ):
            switch(idx, [], (x,))

    def test_switch_single_branch_shortcut(self):
        # [FUNCTION DOC] With a single branch, switch just calls it whatever the index is (ints 0
        #   and 5, tensors 0 and 7).
        # A single-branch switch degenerates to a plain call regardless of
        # index value (matches jax.lax.switch). The index is ignored since
        # it clamps to 0.
        def only_branch(inp_x):
            # [FUNCTION DOC] HELPER: returns sine of the input.
            return inp_x.sin()

        x = torch.randn(4)
        self.assertEqual(switch(0, (only_branch,), (x,)), x.sin())
        self.assertEqual(switch(5, (only_branch,), (x,)), x.sin())
        self.assertEqual(switch(torch.tensor([0]), (only_branch,), (x,)), x.sin())
        self.assertEqual(switch(torch.tensor([7]), (only_branch,), (x,)), x.sin())

    def test_switch_different_output_shapes(self):
        # [FUNCTION DOC] Eager switch where branches return different shapes (sine keeps shape,
        #   sum returns a scalar); checks both index values.
        x = (torch.pi / 2) * torch.ones(2)
        branches = (torch.sin, torch.sum)

        result = switch(0, branches, (x,))
        self.assertEqual(result, torch.ones_like(x))

        result = switch(1, branches, (x,))
        self.assertEqual(result.item(), torch.pi)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_switch_gpu(self):
        # [FUNCTION DOC] CUDA-only: switch with a CUDA index tensor selecting the cosine branch.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns sine of the input.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine of the input.
            return inp_x.cos()

        x = torch.randn(4, device="cuda")
        index = torch.tensor(1, device="cuda")
        result = switch(index, (branch0, branch1), (x,))
        self.assertEqual(result, torch.cos(x))

    def test_switch_no_trace(self):
        # [FUNCTION DOC] Eager smoke test of switch with two branches using int
        #   and tensor indices.
        # Smoke-test eager execution with 2 branches (cond is a special case of switch).
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: cosine of the input.
            return inp_x.cos()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: sine of the input.
            return inp_x.sin()

        x = torch.randn(4)
        self.assertEqual(switch(0, (branch0, branch1), (x,)), x.cos())
        self.assertEqual(switch(1, (branch0, branch1), (x,)), x.sin())
        # tensor index
        self.assertEqual(switch(torch.tensor([0]), (branch0, branch1), (x,)), x.cos())
        self.assertEqual(switch(torch.tensor([1]), (branch0, branch1), (x,)), x.sin())

    def test_switch_cond_equivalence(self):
        # [FUNCTION DOC] Checks switch with branches ordered false, true and index 1 for true / 0
        #   for false gives the same result as cond with the same predicate.
        # switch(int(pred), [false_fn, true_fn], ops) must equal cond(pred, true_fn, false_fn, ops).
        def true_fn(inp_x):
            # [FUNCTION DOC] HELPER: sine of the input.
            return inp_x.sin()

        def false_fn(inp_x):
            # [FUNCTION DOC] HELPER: cosine of the input.
            return inp_x.cos()

        x = torch.randn(4)
        for pred_val in [True, False]:
            pred = torch.tensor(pred_val)
            cond_result = cond(pred, true_fn, false_fn, (x,))
            idx = torch.tensor([1 if pred_val else 0])
            switch_result = switch(idx, (false_fn, true_fn), (x,))
            self.assertEqual(cond_result, switch_result)

    def test_switch_functionalized(self):
        # [FUNCTION DOC] Checks torch.func.functionalize applied to a function using switch
        #   (one branch mutates a local in place) gives the same results as eager for each
        #   of three indices.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine of the input, then in-place adds 4 to that local,
            #   returns its sum.
            y = inp_x.sin()
            y.add_(4)
            return y.sum()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: sum of cosine of the input.
            return inp_x.cos().sum()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: sum of abs of the input.
            return inp_x.abs().sum()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        x = torch.ones(4, 5)
        functional_f = torch.func.functionalize(f)
        for i in range(3):
            idx = torch.tensor([i])
            self.assertEqual(functional_f(idx, x), f(idx, x))

    @skipIfTorchDynamo("don't test compile on compile")
    def test_switch_functionalized_input_mutation(self):
        # [FUNCTION DOC] Skipped under dynamo. Compiles with the AOT eager recording backend a
        #   function that clones its input and switches between a branch that mutates the input
        #   through a view and a clean branch. Checks the result equals the mutating branch run on
        #   a clone and, unless cross-ref testing is on, compares the golden forward graph
        #   (auto_functionalized_v2 wrapping the switch op with two subgraphs).
        def branch_mutating(inp_x):
            # [FUNCTION DOC] HELPER: makes a view of the input, adds 1 in place, returns the sum
            #   of its sine.
            view_x = inp_x.view(inp_x.shape)
            view_x.add_(1)
            return view_x.sin().sum()

        def branch_clean(inp_x):
            # [FUNCTION DOC] HELPER: returns the sum of cosine of the input.
            return inp_x.cos().sum()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two branches without cloning (defined but not
            #   used in this test).
            return switch(idx, (branch_mutating, branch_clean), (inp_x,))

        x = torch.ones(4, 5)

        def g(idx, inp_x):
            # [FUNCTION DOC] HELPER: clones the input then switches between the mutating and clean
            #   branches; this is the function compiled.
            inp_x = inp_x.clone()
            return switch(idx, (branch_mutating, branch_clean), (inp_x,))

        from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm

        backend = AotEagerAndRecordGraphs()
        torch._dynamo.reset()
        with torch.no_grad():
            result = torch.compile(g, backend=backend, fullgraph=True)(
                torch.tensor([0]), x
            )
        expected = branch_mutating(x.clone())
        self.assertEqual(result, expected)
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 5]", arg1_1: "i64[1]"):
        clone: "f32[4, 5]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        clamp: "i64[1]" = torch.ops.aten.clamp.default(arg1_1, 0, 1);  arg1_1 = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.switch, index = clamp, branch0_fn = auto_functionalized_subgraph_0, branch1_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _all_bases = [clone], _op_schema = _tree_spec_constant0);  clamp = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None
        return (getitem,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 5]"):
            view: "f32[4, 5]" = torch.ops.aten.view.default(arg0_1, [4, 5])
            add: "f32[4, 5]" = torch.ops.aten.add.Tensor(view, 1);  view = None
            view_1: "f32[4, 5]" = torch.ops.aten.view.default(add, [4, 5]);  add = None
            view_2: "f32[4, 5]" = torch.ops.aten.view.default(view_1, [4, 5])
            sin: "f32[4, 5]" = torch.ops.aten.sin.default(view_2);  view_2 = None
            sum_1: "f32[]" = torch.ops.aten.sum.default(sin);  sin = None
            copy_: "f32[4, 5]" = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = copy_ = None
            return (sum_1,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 5]"):
            cos: "f32[4, 5]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
            sum_1: "f32[]" = torch.ops.aten.sum.default(cos);  cos = None
            return (sum_1,)
""",
            )

    def test_switch_with_tensor_closure(self):
        # [FUNCTION DOC] Branches add closed-over tensors a, b, c to the input. Checks eager
        #   results, then traces with make_fx and checks the traced graph computes with a; after
        #   mutating a in place the traced graph output follows the new value, showing closures
        #   are lifted as inputs (not baked as constants).
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1
        c = torch.ones(2, 3) * 2

        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor a.
            return inp_x + a

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor b.
            return inp_x + b

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor c.
            return inp_x + c

        def foo(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three closure branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        inp = torch.randn(2, 3)
        for i, closure_val in enumerate((a, b, c)):
            idx = torch.tensor([i])
            self.assertEqual(foo(idx, inp), inp + closure_val)

        # Closed-over tensors must be lifted as extra inputs in the traced
        # graph. This exercises _merge_graph_inputs, which unions
        # free variables across all branches into a shared placeholder
        # signature.
        gm = make_fx(foo)(torch.tensor([0]), inp)
        self.assertEqual(gm(torch.tensor([0]), inp), inp + a)
        # After mutating the closure, the traced graph should reflect the new value.
        a.add_(-1)
        self.assertEqual(gm(torch.tensor([0]), inp), inp + a)

    def test_switch_branch_returns_none(self):
        # [FUNCTION DOC] Checks switch allows None as an output leaf when all branches return None
        #   in that position: tensor part matches and the None part is returned as None.
        # None is allowed as a branch output leaf when every branch returns
        # None in that position. cond._merge_output already handles None;
        # the dynamo-side gate was over-rejecting it, now fixed.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns sine and None.
            return inp_x.sin(), None

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine and None.
            return inp_x.cos(), None

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two branches.
            return switch(idx, (branch0, branch1), (inp_x,))

        x = torch.randn(4)
        for i, fn in enumerate((branch0, branch1)):
            idx = torch.tensor([i])
            t, n = f(idx, x)
            self.assertEqual(t, fn(x)[0])
            self.assertIsNone(n)

    def test_switch_three_branches_divergent_int_leaves(self):
        # [FUNCTION DOC] Traces switch symbolically with branches whose second output is a
        #   different int constant per branch (0, 1, 2) and then a version where all return 7;
        #   checks the traced graph returns the right tensor and int for each index.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns sine and the int 0.
            return inp_x.sin(), 0

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine and the int 1.
            return inp_x.cos(), 1

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: returns abs and the int 2.
            return inp_x.abs(), 2

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="symbolic")(torch.tensor([0]), x)
        for i, fn in enumerate((branch0, branch1, branch2)):
            t, n = graph(torch.tensor([i]), x)
            self.assertEqual(t, fn(x)[0])
            self.assertEqual(n, fn(x)[1])

        def branch0_eq(inp_x):
            # [FUNCTION DOC] HELPER: returns sine and the int 7.
            return inp_x.sin(), 7

        def branch1_eq(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine and the int 7.
            return inp_x.cos(), 7

        def branch2_eq(inp_x):
            # [FUNCTION DOC] HELPER: returns abs and the int 7.
            return inp_x.abs(), 7

        def g(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three equal-int branches.
            return switch(idx, (branch0_eq, branch1_eq, branch2_eq), (inp_x,))

        graph_eq = make_fx(g, tracing_mode="symbolic")(torch.tensor([0]), x)
        for i, fn in enumerate((branch0_eq, branch1_eq, branch2_eq)):
            t, n = graph_eq(torch.tensor([i]), x)
            self.assertEqual(t, fn(x)[0])
            self.assertEqual(n, 7)

    def test_switch_branch_returns_float_rejected(self):
        # [FUNCTION DOC] Documents current behavior: when branches return a Python float next
        #   to a tensor, switch raises UncapturedHigherOrderOpError saying outputs must
        #   consist of tensors.
        # Python float leaves are blocked by the shared HOP output gate.
        # Captures the current behavior so a future relaxation of
        # validate_subgraph_output_types must update this test.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns sine and the float 3.14.
            return inp_x.sin(), 3.14

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine and the float 3.14.
            return inp_x.cos(), 3.14

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two branches.
            return switch(idx, (branch0, branch1), (inp_x,))

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "HigherOrderOperator body's output must consist of tensors",
        ):
            f(torch.tensor([0]), torch.randn(4))

    def test_switch_mismatched_output_arity(self):
        # [FUNCTION DOC] Branch outputs differ in count (one tensor versus two);
        #   expects RuntimeError or TorchRuntimeError matching a differing-branch-
        #   outputs or unmatched-output-spec message.
        # Branch 0 returns one tensor, branch 1 returns a tuple of two
        # tensors — different number of outputs.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns one tensor (sine).
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns two tensors (cosine, abs).
            return inp_x.cos(), inp_x.abs()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two branches.
            return switch(idx, (branch0, branch1), (inp_x,))

        with self.assertRaisesRegex(
            (RuntimeError, torch._dynamo.exc.TorchRuntimeError),
            r"(differing branch outputs|Unmatched output spec from torch\.switch branches)",
        ):
            f(torch.tensor([0]), torch.randn(4))

    def test_switch_mismatched_output_structure(self):
        # [FUNCTION DOC] Branches return a tuple versus a dict with the same number of tensors;
        #   expects the same output-spec mismatch error.
        # Branch 0 returns a tuple; branch 1 returns a dict. Two tensor
        # leaves each, but the tree specs disagree.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns a tuple of sine and cosine.
            return (inp_x.sin(), inp_x.cos())

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns a dict with sine under a and cosine under b.
            return {"a": inp_x.sin(), "b": inp_x.cos()}

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two branches.
            return switch(idx, (branch0, branch1), (inp_x,))

        with self.assertRaisesRegex(
            (RuntimeError, torch._dynamo.exc.TorchRuntimeError),
            r"(differing branch outputs|Unmatched output spec from torch\.switch branches)",
        ):
            f(torch.tensor([0]), torch.randn(4))

    def test_switch_mismatched_output_structure_three_branches(self):
        # [FUNCTION DOC] Three branches where only the last returns a two-tensor tuple; expects
        #   the output-spec mismatch error.
        # Three-branch mismatch: branches 0 and 1 agree on a single-tensor
        # return; branch 2 diverges by returning a two-tensor tuple.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: returns abs and sum.
            return inp_x.abs(), inp_x.sum()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        with self.assertRaisesRegex(
            (RuntimeError, torch._dynamo.exc.TorchRuntimeError),
            r"(differing branch outputs|Unmatched output spec from torch\.switch branches)",
        ):
            f(torch.tensor([0]), torch.randn(4))

    def test_switch_constant_index_specialization(self):
        # [FUNCTION DOC] Compiled with the eager backend and full graph, a Python constant index
        #   makes dynamo pick the branch directly: indices 0, 1, 2, a negative (clamped to the
        #   first) and a huge one (clamped to the last) each give the matching branch result.
        # When the index is a Python constant, dynamo specializes to that
        # specific branch — the switch HOP never enters the graph.
        # backend="eager" is used because PR 1 has no inductor lowering;
        # specialization is a dynamo-level concern.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        branches = (branch0, branch1, branch2)

        # Each compiled wrapper has its index baked in as a literal so
        # dynamo definitely sees a ConstantVariable (sidestepping any
        # scalar-argument promotion to SymInt).
        @torch.compile(backend="eager", fullgraph=True)
        def f0(inp_x):
            # [FUNCTION DOC] HELPER: compiled; switch with literal index 0.
            return switch(0, branches, (inp_x,))

        @torch.compile(backend="eager", fullgraph=True)
        def f1(inp_x):
            # [FUNCTION DOC] HELPER: compiled; switch with literal index 1.
            return switch(1, branches, (inp_x,))

        @torch.compile(backend="eager", fullgraph=True)
        def f2(inp_x):
            # [FUNCTION DOC] HELPER: compiled; switch with literal index 2.
            return switch(2, branches, (inp_x,))

        @torch.compile(backend="eager", fullgraph=True)
        def f_neg(inp_x):
            # [FUNCTION DOC] HELPER: compiled; switch with literal index minus 5.
            return switch(-5, branches, (inp_x,))

        @torch.compile(backend="eager", fullgraph=True)
        def f_huge(inp_x):
            # [FUNCTION DOC] HELPER: compiled; switch with literal index 99.
            return switch(99, branches, (inp_x,))

        x = torch.randn(4)
        self.assertEqual(f0(x), branch0(x))
        self.assertEqual(f1(x), branch1(x))
        self.assertEqual(f2(x), branch2(x))
        # Clamped negative index picks branch 0.
        self.assertEqual(f_neg(x), branch0(x))
        # Clamped above-range index picks branch N-1.
        self.assertEqual(f_huge(x), branch2(x))

    @skipIfTorchDynamo("mark_dynamic cannot be traced under dynamo_wrapped")
    def test_switch_symint_index_clamped(self):
        # [FUNCTION DOC] Skipped under dynamo. Compiled function derives the index from a dynamic
        #   dimension size minus 5. Size 2 gives a negative index clamped to branch 0; after a
        #   dynamo reset, size 8 gives an index above range clamped to the last branch.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        branches = (branch0, branch1, branch2)

        @torch.compile(backend="eager", fullgraph=True)
        def f(idx_carrier, inp_x):
            # [FUNCTION DOC] HELPER: compiled; index is the leading size of idx_carrier minus 5;
            #   switches over the three branches.
            idx = idx_carrier.shape[0] - 5
            return switch(idx, branches, (inp_x,))

        x = torch.randn(4)

        # shape[0] - 5 is negative -> clamps to branch0.
        idx_carrier = torch.zeros(2)
        torch._dynamo.mark_dynamic(idx_carrier, 0)
        self.assertEqual(f(idx_carrier, x), branch0(x))

        # shape[0] - 5 == 3 (above range) -> clamps to branch2.
        torch._dynamo.reset()
        idx_carrier = torch.zeros(8)
        torch._dynamo.mark_dynamic(idx_carrier, 0)
        self.assertEqual(f(idx_carrier, x), branch2(x))

    def test_switch_nn_module_branches(self):
        # [FUNCTION DOC] Switch where each branch is a different Linear module (parameters
        #   captured as free variables); checks each index gives the matching module output.
        # Each branch is a different nn.Module whose parameters are
        # captured as free variables. Dynamo must lift the parameters of
        # all branches into the HOP's operand tuple and rewrite every
        # branch graph's signature to match.
        linear0 = torch.nn.Linear(4, 3)
        linear1 = torch.nn.Linear(4, 3)

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the two modules.
            return switch(idx, (linear0, linear1), (inp_x,))

        x = torch.randn(2, 4)
        self.assertEqual(f(torch.tensor([0]), x), linear0(x))
        self.assertEqual(f(torch.tensor([1]), x), linear1(x))

    def test_switch_complex_nn_module_branches(self):
        # [FUNCTION DOC] Switch over three MLP modules with different hidden sizes but the same
        #   output shape; checks each index matches calling the module directly.
        # Complex modules: each branch is a different multi-layer MLP with
        # different hidden dimensions. As long as the final output shape
        # matches across branches, dynamo captures and lifts all module
        # parameters correctly.
        class MLP(torch.nn.Module):
            # [CLASS DOC] HELPER: two-layer perceptron with ReLU between linear layers.
            def __init__(self, in_dim, hidden, out_dim):
                # [FUNCTION DOC] HELPER: stores two Linear layers mapping in_dim to hidden and
                #   hidden to out_dim.
                super().__init__()
                self.fc1 = torch.nn.Linear(in_dim, hidden)
                self.fc2 = torch.nn.Linear(hidden, out_dim)

            def forward(self, inp_x):
                # [FUNCTION DOC] HELPER: applies the first layer, ReLU, then the second layer.
                return self.fc2(torch.relu(self.fc1(inp_x)))

        mlp_small = MLP(4, 8, 2)
        mlp_large = MLP(4, 16, 2)
        mlp_tiny = MLP(4, 3, 2)

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three MLPs.
            return switch(idx, (mlp_small, mlp_large, mlp_tiny), (inp_x,))

        x = torch.randn(5, 4)
        for i, mlp in enumerate((mlp_small, mlp_large, mlp_tiny)):
            self.assertEqual(f(torch.tensor([i]), x), mlp(x))

    def test_switch_heterogeneous_lifted_args(self):
        # [FUNCTION DOC] Branches capture different free variables (one tensor, two tensors, a
        #   Linear module); checks the results for each index equal direct calls.
        # Branches with different combinations of lifted free variables:
        # branch0 captures one tensor, branch1 captures two tensors,
        # branch2 captures an nn.Module. dynamo unions all lifted
        # freevars across branches and every branch graph receives the
        # same placeholder signature.
        bias = torch.ones(3)
        offset = torch.tensor([0.5, 0.5, 0.5])
        linear = torch.nn.Linear(4, 3)

        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: linear map with a fixed identity-like weight plus
            #   the bias tensor.
            return torch.nn.functional.linear(inp_x, torch.eye(4, 3).t()) + bias

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: same as branch0 plus an offset tensor.
            return (
                torch.nn.functional.linear(inp_x, torch.eye(4, 3).t()) + bias + offset
            )

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: applies the captured Linear module.
            return linear(inp_x)

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        x = torch.randn(2, 4)
        self.assertEqual(f(torch.tensor([0]), x), branch0(x))
        self.assertEqual(f(torch.tensor([1]), x), branch1(x))
        self.assertEqual(f(torch.tensor([2]), x), branch2(x))

    def test_switch_pytree_operands(self):
        # [FUNCTION DOC] Switch with a nested dict/list/tuple pytree as the single operand; each
        #   of three indices matches calling the branch directly.
        # Test that switch supports pytree operands (nested dict/list/tuple of tensors)
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: first entry plus "b" times third entry's element.
            return inp_x["t"][0] + inp_x["t"][1]["b"] * inp_x["t"][2][0]

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: first entry times (third element divided by "b").
            return inp_x["t"][0] * (inp_x["t"][2][0] / inp_x["t"][1]["b"])

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: first entry minus "b" plus third entry's element.
            return inp_x["t"][0] - inp_x["t"][1]["b"] + inp_x["t"][2][0]

        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.randn(4)
        operands = ({"t": [a, {"b": b}, (c,)]},)

        # Test each branch
        for idx, fn in enumerate([branch0, branch1, branch2]):
            result = switch(torch.tensor(idx), [branch0, branch1, branch2], operands)
            expected = fn(operands[0])
            self.assertEqual(result, expected)

    @parametrize("idx", [0, 1, 2, -5, 99])
    def test_switch_autograd_basic(self, idx):
        # [FUNCTION DOC] Parametrized over indices 0, 1, 2, -5 and 99 (the last two clamp to the
        #   first and last branch). Checks switch result and gradient wrt x against the clamped
        #   branch called directly.
        def branch0(x):
            # [FUNCTION DOC] HELPER: sine.
            return x.sin()

        def branch1(x):
            # [FUNCTION DOC] HELPER: cosine of x plus 42.
            return (x + 42).cos()

        def branch2(x):
            # [FUNCTION DOC] HELPER: tanh of x times 3.14, plus x.
            return (x * 3.14).tanh() + x

        branches = (branch0, branch1, branch2)
        expected_fn = branches[min(max(0, idx), len(branches) - 1)]

        x = torch.randn(4, requires_grad=True)
        result = switch(torch.tensor(idx), branches, (x,))
        self.assertEqual(result, expected_fn(x))

        grad_out = torch.ones_like(result)
        grads = torch.autograd.grad(result, (x,), grad_out)
        expected_grads = torch.autograd.grad(expected_fn(x), (x,), grad_out)
        self.assertEqual(expected_grads, grads)

    def test_switch_autograd_make_fx_two_switches(self):
        # [FUNCTION DOC] Traces switch plus backward with make_fx and asserts the graph has
        #   exactly two switch higher-order op calls (forward and backward), then checks the
        #   gradient value against the branch called directly.
        def branch0(x):
            # [FUNCTION DOC] HELPER: sine.
            return x.sin()

        def branch1(x):
            # [FUNCTION DOC] HELPER: cosine.
            return x.cos()

        branches = (branch0, branch1)

        def f(idx, x):
            # [FUNCTION DOC] HELPER: traced function; switch then gradient wrt x with ones
            #   as output gradient.
            result = switch(idx, branches, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        x = torch.randn(4, requires_grad=True)
        gm = make_fx(f)(torch.tensor(1), x)
        switch_calls = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.higher_order.switch
        ]
        self.assertEqual(len(switch_calls), 2)

        grads = f(torch.tensor(1), x)
        expected_grads = torch.autograd.grad(
            branch1(x), (x,), torch.ones_like(branch1(x))
        )
        self.assertEqual(grads, expected_grads)

    def test_switch_autograd_nested(self):
        # [FUNCTION DOC] Builds branch functions that themselves call switch with constant tensor
        #   indices (nested switch) to check autograd through nested switch.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: nested switch with index 1 over three inner functions.
            def inner0(y):
                # [FUNCTION DOC] HELPER: sine of y times 2.
                return (y * 2).sin()

            def inner1(y):
                # [FUNCTION DOC] HELPER: cosine of y plus 1.
                return (y + 1).cos()

            def inner2(y):
                # [FUNCTION DOC] HELPER: tanh of y.
                return y.tanh()

            return switch(torch.tensor(1), (inner0, inner1, inner2), (x,))

        def mid_fn(x):
            # [FUNCTION DOC] HELPER: nested switch with index 0 over two inner functions.
            def inner0(y):
                # [FUNCTION DOC] HELPER: sine of y plus 3.
                return (y + 3).sin()

            def inner1(y):
                # [FUNCTION DOC] HELPER: cosine of y times 0.5.
                return (y * 0.5).cos()

            return switch(torch.tensor(0), (inner0, inner1), (x,))

        def false_fn(x):
            # [FUNCTION DOC] HELPER: defines three inner functions (relu, sigmoid of y minus 1,
            #   cosine of y squared) and switches among them with index 2.
            def inner0(y):
                # [FUNCTION DOC] HELPER: relu of y.
                return y.relu()

            def inner1(y):
                # [FUNCTION DOC] HELPER: sigmoid of y minus 1.
                return (y - 1).sigmoid()

            def inner2(y):
                # [FUNCTION DOC] HELPER: cosine of y squared.
                return (y**2).cos()

            return switch(torch.tensor(2), (inner0, inner1, inner2), (x,))

        branches = (true_fn, mid_fn, false_fn)
        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True)
            result = switch(torch.tensor(i), branches, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_mixed_require_grad(self):
        # [FUNCTION DOC] Switch with three operands where only x requires grad (y does not); for
        #   each index checks the result and the gradient wrt x equal calling the branch directly.
        def branch0(x, y, z):
            # [FUNCTION DOC] HELPER: product of the three inputs.
            return x * y * z

        def branch1(x, y, z):
            # [FUNCTION DOC] HELPER: sum of the three inputs.
            return x + y + z

        def branch2(x, y, z):
            # [FUNCTION DOC] HELPER: first minus second, times third.
            return (x - y) * z

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=False)

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            result = switch(torch.tensor(i), branches, (x, y, x))
            self.assertEqual(result, fn(x, y, x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x, y, x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_pytree_input(self):
        # [FUNCTION DOC] Switch with a nested dict/list/tuple operand of three grad-requiring
        #   tensors; for each index checks the result and the gradients wrt all three leaves.
        def branch0(x):
            # [FUNCTION DOC] HELPER: first entry plus "b" times third element.
            return x["t"][0] + x["t"][1]["b"] * x["t"][2][0]

        def branch1(x):
            # [FUNCTION DOC] HELPER: first entry times (third element divided by "b").
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        def branch2(x):
            # [FUNCTION DOC] HELPER: first minus "b" plus third element.
            return x["t"][0] - x["t"][1]["b"] + x["t"][2][0]

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            operands = ({"t": [a, {"b": b}, (c,)]},)
            result = switch(torch.tensor(i), branches, operands)
            self.assertEqual(result, fn(operands[0]))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (a, b, c), grad_out)
            expected_grads = torch.autograd.grad(fn(operands[0]), (a, b, c), grad_out)
            self.assertEqual(expected_grads, grads)

    @skipIfTorchDynamo("Skip due to graph break when run with dynamo")
    def test_switch_autograd_pytree_not_all_inputs_used(self):
        # [FUNCTION DOC] Skipped under dynamo. Branches ignore some leaves of the pytree operand.
        #   For each index checks the result and that gradients computed with unused allowed equal
        #   the reference gradients with unused ones materialized as zeros.
        def branch0(x):
            # [FUNCTION DOC] HELPER: first entry plus "b" (ignores the third).
            return x["t"][0] + x["t"][1]["b"]

        def branch1(x):
            # [FUNCTION DOC] HELPER: first times (third divided by "b").
            return x["t"][0] * (x["t"][2][0] / x["t"][1]["b"])

        def branch2(x):
            # [FUNCTION DOC] HELPER: clone of the third element only.
            return x["t"][2][0].clone()

        a = torch.randn(4, requires_grad=True)
        b = torch.randn(4, requires_grad=True)
        c = torch.randn(4, requires_grad=True)

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            operands = ({"t": [a, {"b": b}, (c,)]},)
            result = switch(torch.tensor(i), branches, operands)
            self.assertEqual(result, fn(operands[0]))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (a, b, c), grad_out, allow_unused=True)
            expected_grads = torch.autograd.grad(
                fn(operands[0]),
                (a, b, c),
                grad_out,
                allow_unused=True,
                materialize_grads=True,
            )
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_pytree_output(self):
        # [FUNCTION DOC] Branches return the same dict structure holding a list with a
        #   tensor and a tuple; for each index checks the result and the gradient wrt x of
        #   the flattened outputs.
        def branch0(x):
            # [FUNCTION DOC] HELPER: dict holding sine and a tuple of cosine.
            return {"res": [x.sin(), (x.cos(),)]}

        def branch1(x):
            # [FUNCTION DOC] HELPER: dict holding cosine and a tuple of tanh.
            return {"res": [x.cos(), (x.tanh(),)]}

        def branch2(x):
            # [FUNCTION DOC] HELPER: dict holding tanh and a tuple of sine.
            return {"res": [x.tanh(), (x.sin(),)]}

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True)
            result = switch(torch.tensor(i), branches, (x,))
            expected = fn(x)
            self.assertEqual(result, expected)

            result_flat, _ = pytree.tree_flatten(result)
            expected_flat, _ = pytree.tree_flatten(expected)
            grad_out = [torch.ones_like(g) for g in result_flat]
            grads = torch.autograd.grad(result_flat, (x,), grad_out)
            expected_grads = torch.autograd.grad(expected_flat, (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_grad_through_params(self):
        # [FUNCTION DOC] Checks gradients wrt the weight of a Linear module captured by the
        #   branches, for each of three indices, against direct branch calls (graph retained for
        #   the two gradient calls).
        nn_module = torch.nn.Linear(4, 4)

        def branch0(x):
            # [FUNCTION DOC] HELPER: the module applied to x.
            return nn_module(x)

        def branch1(x):
            # [FUNCTION DOC] HELPER: x times the module applied to x.
            return x * nn_module(x)

        def branch2(x):
            # [FUNCTION DOC] HELPER: relu of the module output.
            return nn_module(x).relu()

        branches = (branch0, branch1, branch2)
        x = torch.randn(4, requires_grad=True)
        for i, fn in enumerate(branches):
            result = switch(torch.tensor(i), branches, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(
                result, (nn_module.weight,), grad_out, retain_graph=True
            )
            expected_grads = torch.autograd.grad(
                fn(x), (nn_module.weight,), grad_out, retain_graph=True
            )
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_inner_tensor(self):
        # [FUNCTION DOC] Branches that create constant tensors internally (ones times 42, a tensor
        #   filled with 0.5); checks results and gradients wrt x for each index.
        def branch0(x):
            # [FUNCTION DOC] HELPER: abs of sine of x squared.
            return torch.abs((x**2).sin())

        def branch1(x):
            # [FUNCTION DOC] HELPER: cosine of x times a constant tensor of 42s.
            y = torch.ones(4, requires_grad=False) * 42
            return (x * y).cos()

        def branch2(x):
            # [FUNCTION DOC] HELPER: tanh of x plus a constant tensor of 0.5s.
            z = torch.full((4,), 0.5, requires_grad=False)
            return (x + z).tanh()

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True)
            result = switch(torch.tensor(i), branches, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_symint_index(self):
        # [FUNCTION DOC] Compiles with the eager backend, full graph and dynamic shapes a function
        #   that computes the switch index as a size taken from an arange of the integer argument;
        #   for each index checks the result and gradient wrt x against the direct branch.
        def branch0(x):
            # [FUNCTION DOC] HELPER: sine.
            return x.sin()

        def branch1(x):
            # [FUNCTION DOC] HELPER: cosine.
            return x.cos()

        def branch2(x):
            # [FUNCTION DOC] HELPER: tanh.
            return x.tanh()

        branches = (branch0, branch1, branch2)

        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def f(i, x):
            # [FUNCTION DOC] HELPER: compiled; index is the size of an arange of length i;
            #   switches over the three branches.
            index = torch.arange(i).size(0)
            return switch(index, branches, (x,))

        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True)
            result = f(i, x)
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    @parametrize("aux", ["none", "int_equal", "int_divergent"])
    def test_switch_autograd_non_tensor_output_leaves(self, aux):
        # [FUNCTION DOC] Parametrized by aux in none, int_equal, int_divergent: each branch
        #   returns a tensor and an auxiliary leaf (None, 7 for all, or 0/1/2 per branch). Checks
        #   tensor value, aux leaf value and the gradient wrt x for each index.
        if aux == "none":
            aux_vals = (None, None, None)
        elif aux == "int_equal":
            aux_vals = (7, 7, 7)
        else:  # int_divergent
            aux_vals = (0, 1, 2)

        def make_branch(op, a):
            # [FUNCTION DOC] HELPER: factory returning a branch that applies op to its input and
            #   returns it along with the fixed auxiliary value a.
            def branch(x):
                # [FUNCTION DOC] HELPER: returns the op applied to x and the
                #   captured auxiliary value.
                return op(x), a

            return branch

        ops = (torch.sin, torch.cos, torch.tanh)
        branches = tuple(make_branch(op, a) for op, a in zip(ops, aux_vals))

        for i, (op, a) in enumerate(zip(ops, aux_vals)):
            x = torch.randn(4, requires_grad=True)
            t, n = switch(torch.tensor(i), branches, (x,))

            expected_t = op(x)
            self.assertEqual(t, expected_t)
            if a is None:
                self.assertIsNone(n)
            else:
                self.assertEqual(n, a)

            grad_out = torch.ones_like(t)
            grads = torch.autograd.grad(t, (x,), grad_out)
            expected_grads = torch.autograd.grad(expected_t, (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    def test_switch_autograd_multiple_tensor_outputs(self):
        # [FUNCTION DOC] Branches return two tensors each; for each index checks the outputs and
        #   the gradient wrt x with ones as output gradients for both outputs.
        def branch0(x):
            # [FUNCTION DOC] HELPER: sine and cosine.
            return x.sin(), x.cos()

        def branch1(x):
            # [FUNCTION DOC] HELPER: tanh and relu.
            return x.tanh(), x.relu()

        def branch2(x):
            # [FUNCTION DOC] HELPER: sigmoid and sine of x squared.
            return x.sigmoid(), (x**2).sin()

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True)
            outs = switch(torch.tensor(i), branches, (x,))
            expected = fn(x)
            self.assertEqual(outs, expected)

            grad_outs = [torch.ones_like(o) for o in outs]
            expected_grad_outs = [torch.ones_like(o) for o in expected]
            grads = torch.autograd.grad(list(outs), (x,), grad_outs)
            expected_grads = torch.autograd.grad(
                list(expected), (x,), expected_grad_outs
            )
            self.assertEqual(expected_grads, grads)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_switch_autograd_gpu(self):
        # [FUNCTION DOC] CUDA-only: switch with a CUDA index tensor and CUDA input requiring grad;
        #   checks result and gradient for each index.
        def branch0(x):
            # [FUNCTION DOC] HELPER: sine.
            return x.sin()

        def branch1(x):
            # [FUNCTION DOC] HELPER: cosine.
            return x.cos()

        def branch2(x):
            # [FUNCTION DOC] HELPER: tanh.
            return x.tanh()

        branches = (branch0, branch1, branch2)
        for i, fn in enumerate(branches):
            x = torch.randn(4, requires_grad=True, device="cuda")
            result = switch(torch.tensor(i, device="cuda"), branches, (x,))
            self.assertEqual(result, fn(x))

            grad_out = torch.ones_like(result)
            grads = torch.autograd.grad(result, (x,), grad_out)
            expected_grads = torch.autograd.grad(fn(x), (x,), grad_out)
            self.assertEqual(expected_grads, grads)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_map_gpu(self):
        # [FUNCTION DOC] CUDA-only: control_flow.map of an add of a per-slice value and a
        #   broadcast value over a CUDA stack; compares with the reference fake map.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        xs = torch.ones(3, 2, 2, device="cuda")
        y = torch.ones(2, device="cuda")
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(expected, res)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    def test_while_loop_gpu(self):
        # [FUNCTION DOC] CUDA-only: while_loop adding one until the sum reaches 10 on a CUDA
        #   tensor; compares with the reference fake while loop.
        def cond_fn(x):
            # [FUNCTION DOC] HELPER: true while the sum of x is below 10.
            return x.sum() < 10

        def body_fn(x):
            # [FUNCTION DOC] HELPER: returns a one-tuple with x plus one.
            return (x + 1,)

        x = torch.zeros(1, device="cuda")
        res = while_loop(cond_fn, body_fn, (x,))
        expected = _fake_while_loop(cond_fn, body_fn, (x,))
        self.assertEqual(expected, res)

    def test_map_illegal_inputs(self):
        # [FUNCTION DOC] Checks control_flow.map raises RuntimeError for: xs containing a non-
        #   tensor (message about mapped xs consisting of tensors), a leading dimension of size 0,
        #   and inconsistent leading dimensions across xs.
        def f(x, y):
            # [FUNCTION DOC] HELPER: adds the two entries of x and y.
            return x[0] + x[1] + y

        with self.assertRaisesRegex(
            RuntimeError,
            r"Mapped xs can only consist of tensors\. Got xs \[3, tensor\(\[1\., 1\.\]\)\]\.",
        ):
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))

        with self.assertRaisesRegex(
            RuntimeError, r"Leading dimensions of mapped xs cannot be 0\."
        ):
            _ = control_flow.map(
                f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Leading dimensions of mapped xs must be consistent\. "
            r"Got shapes \[torch\.Size\(\[3, 4, 5\]\), torch\.Size\(\[4, 4, 5\]\)\]\.",
        ):
            _ = control_flow.map(
                f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5)
            )

    def test_map_illegal_outputs(self):
        # [FUNCTION DOC] map whose function returns a Python scalar via item raises RuntimeError
        #   naming the map_impl higher-order op; returning a size object raises
        #   UncapturedHigherOrderOpError with the same op name; returning None is accepted.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns the item of x (a Python number).
            return x.item()

        def f1(x, y):
            # [FUNCTION DOC] HELPER: returns the size of y (non-tensor).
            return y.size()

        def f2(x, y):
            # [FUNCTION DOC] HELPER: returns None.
            return None

        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        with self.assertRaisesRegex(
            RuntimeError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            control_flow.map(f, x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # "Expected all leaves to be of torch.Tensor type.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            control_flow.map(f1, x, y)

        # return None is OK
        control_flow.map(f2, x, y)

    def test_map_list_in_out(self):
        # [FUNCTION DOC] map with list-of-list structure for both input and output; checks result
        #   structure (one by one nested) and values against the reference fake map.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a nested list with the first nested tensor plus y.
            return [[x[0][0] + y]]

        xs = [[torch.ones(3, 2, 2)]]
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(expected, res)

    def test_map_dict_in_out(self):
        # [FUNCTION DOC] map with nested dict input and a dict output; checks the output has a
        #   single key "c" and matches the reference.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a dict with "c" set to the inner tensor plus y.
            return {"c": x["a"]["b"] + y}

        xs = {"a": {"b": torch.ones(3, 2, 2)}}
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertTrue("c" in res)
        self.assertEqual(expected, res)

    def test_map_autograd_simple(self):
        # [FUNCTION DOC] map with a function of both a mapped input and a broadcast input, both
        #   requiring grad; compares results and gradients wrt both to the reference fake map.
        def f(x, y):
            # [FUNCTION DOC] HELPER: sine then cosine of x times cosine then sine of y.
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=True)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs, y), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs, y), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_simple_partial_grad(self):
        # [FUNCTION DOC] Same as the simple map autograd test but y does not require grad;
        #   compares gradient wrt xs only.
        def f(x, y):
            # [FUNCTION DOC] HELPER: sine then cosine of x times cosine then sine of y.
            return x.sin().cos() * y.cos().sin()

        xs = torch.ones(3, 2, 2, requires_grad=True)
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs,), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs,), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_no_grad_output(self):
        # [FUNCTION DOC] map over a list of xs where only the first requires grad and the function
        #   returns two outputs, the second independent of xs; checks values and gradient of the
        #   first output wrt xs first element.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns first-input sine-cosine plus y, and the sine of
            #   cosine of y.
            return x[0].sin().cos() + y, y.cos().sin()

        xs = [torch.ones(3, 2, 2, requires_grad=True), torch.ones(3, 3)]
        # Disable the gradient computation for y
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res[0])
        grads = torch.autograd.grad(res[0], (xs[0],), grad_out)
        expected_grads = torch.autograd.grad(expected_res[0], (xs[0],), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_nested_list(self):
        # [FUNCTION DOC] Runs map over nested-list inputs with a forward plus backward helper and
        #   compares outputs and gradients with the reference fake map.
        import torch.utils._pytree as pytree

        def f(x, y):
            # [FUNCTION DOC] HELPER: unpacks x into nested parts and returns a nested list with
            #   sine times cosine combinations, one involving y.
            a, b = x
            c, d = a
            return [[b.sin() * c.cos()], d.sin() * y.cos()]

        def fwbw(map_op, f, x, y):
            # [FUNCTION DOC] HELPER: applies the given map operator, flattens inputs and outputs,
            #   and returns the output with gradients of all outputs wrt all flattened inputs
            #   using ones as output gradients.
            z = map_op(f, x, y)
            flat_x = pytree.tree_leaves(x)
            flat_z = pytree.tree_leaves(z)
            grads = torch.autograd.grad(
                flat_z, flat_x, [torch.ones_like(z) for z in flat_z]
            )
            return z, grads

        x = [
            [
                torch.randn(3, 2, 2, requires_grad=True),
                torch.randn(3, 2, 1, requires_grad=True),
            ],
            torch.ones(3, 1, 2, requires_grad=True),
        ]
        y = torch.ones(1, requires_grad=True)
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        self.assertEqual(true_outs, fake_outs)

    def test_map_autograd_higher_order(self):
        # [FUNCTION DOC] Compares jacobian and hessian (from torch.autograd.functional) of
        #   map-based wrappers against the same on the fake map wrappers, for xs requiring
        #   grad and y not.
        from torch.autograd.functional import hessian as hes, jacobian as jac

        def f(x, y):
            # [FUNCTION DOC] HELPER: sine then cosine of x, plus y.
            return x.sin().cos() + y

        def wrapper_jac(x, y):
            # [FUNCTION DOC] HELPER: map of f for jacobian.
            return control_flow.map(f, x, y)

        def wrapper_jac_fake(x, y):
            # [FUNCTION DOC] HELPER: fake map of f for jacobian.
            return _fake_map(f, x, y)

        def wrapper_hes(x, y):
            # [FUNCTION DOC] HELPER: sum of map of f, for hessian.
            return control_flow.map(f, x, y).sum()

        def wrapper_hes_fake(x, y):
            # [FUNCTION DOC] HELPER: sum of fake map of f, for hessian.
            return _fake_map(f, x, y).sum()

        for g_fct, (wrap, wrap_fake) in [
            (jac, [wrapper_jac, wrapper_jac_fake]),
            (hes, [wrapper_hes, wrapper_hes_fake]),
        ]:
            xs = torch.ones(3, 2, 2, requires_grad=True)
            # Disable the gradient computation for y
            y = torch.ones(2, requires_grad=False)
            res = control_flow.map(f, xs, y)
            expected_res = _fake_map(f, xs, y)
            self.assertEqual(expected_res, res)

            expected_grads = g_fct(wrap_fake, (xs, y))
            grads = g_fct(wrap, (xs, y))
            self.assertEqual(expected_res, res)
            self.assertEqual(expected_grads, grads)

    def test_scan_y_less_ndim_then_dim(self):
        # [FUNCTION DOC] Scan along dimension 2 where the per-step output is a scalar (fewer
        #   dimensions than the scan dim) with a matrix-product carry; compares with the
        #   reference fake scan.
        def combine_fn(carry, x):
            # [FUNCTION DOC] HELPER: returns the matrix product of carry and x as the new carry,
            #   and the sum of that product as the per-step output.
            return carry @ x, (carry @ x).sum()

        init = torch.randn(4, 3)
        xs = torch.randn(3, 3, 2)
        dim = 2
        out = scan(combine_fn, init, xs, dim=dim)
        exp_out = _fake_scan(combine_fn, init, xs, dim=dim)
        self.assertEqual(out, exp_out)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_compile(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Skipped under dynamo; needs CUDA; parametrized over reverse, compile mode
        #   (none or eager), device and autograd. Compares scan with the reference fake scan for
        #   cumulative sum and product combine functions (and against torch cumsum and cumprod
        #   when not reversed), integer examples with known expected values, a combine function
        #   with different carry and output math, and a non-associative division; checks gradients
        #   when autograd is on.
        def add2(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns the product as the new carry and the sum
            #   as the output.
            return x * y, x + y

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        scan_fct = compile_mode_helper(scan, compile_mode)

        for op, op_pt, init in [
            (
                get_scan_combine_fn("add", False),
                torch.cumsum,
                torch.zeros(10, 2, device=device, requires_grad=autograd),
            ),
            (
                get_scan_combine_fn("mul", False),
                torch.cumprod,
                torch.ones(10, 2, device=device, requires_grad=autograd),
            ),
        ]:
            result = scan_fct(op, init, x, dim=0, reverse=reverse)
            result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
            self.assertEqual(result, result_exp)
            if not reverse:
                result_exp_PT = op_pt(x, 0)
                self.assertEqual(result[1], result_exp_PT)

            if autograd:
                self.check_autograd(result, result_exp, (init, x))

        # Jax Examples
        x = torch.arange(0, 4, device=device, dtype=torch.int64)
        init = torch.zeros(1, device=device, dtype=torch.int64)
        cumsum1 = scan_fct(
            get_scan_combine_fn("add", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        cumsum_exp = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        if not reverse:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        else:
            self.assertEqual(
                cumsum1[1],
                torch.tensor([[6.0], [6.0], [5.0], [3.0]], dtype=torch.int64),
            )
            self.assertEqual(cumsum1[0], torch.tensor([6.0], dtype=torch.int64))
        self.assertEqual(cumsum1, cumsum_exp)

        # Different carry computation as output computation
        x = torch.arange(1, 5, device=device, dtype=torch.int64)
        init = torch.ones(1, device=device, dtype=torch.int64)
        result = scan_fct(add2, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(add2, init=init, xs=x, dim=0, reverse=reverse)
        if not reverse:
            self.assertEqual(
                result[1],
                torch.tensor([[2.0], [3.0], [5.0], [10.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        else:
            self.assertEqual(
                result[1],
                torch.tensor([[25.0], [14.0], [7.0], [5.0]], dtype=torch.int64),
            )
            self.assertEqual(result[0], torch.tensor([24.0], dtype=torch.int64))
        self.assertEqual(result, result_exp)

        # Non associative operation
        x = torch.arange(
            0, 5, device=device, dtype=torch.float32, requires_grad=autograd
        )
        init = torch.ones(1, device=device, dtype=torch.float32, requires_grad=autograd)
        result = scan_fct(
            get_scan_combine_fn("div", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("div", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize(
        "dtype",
        [
            torch.float16,
            torch.float32,
            torch.int32,
            torch.int64,
            torch.complex64,
        ],
    )
    def test_scan_dtype(self, reverse, compile_mode, device, dtype):
        # [FUNCTION DOC] Skipped under dynamo; needs CUDA; parametrized over reverse, compile
        #   mode, device and dtype (float16, float32, int32, int64, complex64). Checks that scan
        #   outputs match the fake scan and that the device and dtype of carry and outputs are as
        #   expected, including cases where carry dtype differs from output dtype.
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Check all outputs and carries on the correct device and with torch.float32
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.device.type for r in res] for res in result],
            [[device.type for _ in res] for res in result],
        )
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [[dtype for _ in res] for res in result],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.float32 and output.dtype torch.float16
        x = torch.randn(3, 10, 2, device=device).to(dtype=dtype)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=torch.float32),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [torch.float32 for _ in range(len(result[0]))],
                [dtype for _ in range(len(result[1]))],
            ],
        )

        # Check all outputs and carries on the correct device and
        # carry.dtype torch.int64 and output.dtype torch.float32
        x = torch.randn(3, 10, 2, device=device)
        op, init = (
            get_scan_combine_fn("adds"),
            torch.zeros(10, 2, device=device, dtype=dtype),
        )
        result = scan_fct(op, init, x, dim=0, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=0, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(
            [[r.dtype for r in res] for res in result],
            [
                [dtype for _ in range(len(result[0]))],
                [torch.float32 for _ in range(len(result[1]))],
            ],
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_dim(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA. For five random tensor ranks (2 to 5) and random
        #   sizes and scan dimension (no seed, so shapes vary per run), checks scan with add and
        #   mul combine functions against the fake scan and, when not reversed, against torch
        #   cumsum and cumprod; optionally checks gradients.
        import random

        scan_fct = compile_mode_helper(scan, compile_mode)

        num_dims = [random.randint(2, 5) for _ in range(5)]
        for num_dim in num_dims:
            shapes = [random.randint(1, 10) for _ in range(num_dim)]
            rnd_scan_dim = random.randint(0, num_dim - 1)
            x = torch.randn(*shapes, device=device, requires_grad=autograd)
            init_shapes = shapes[:rnd_scan_dim] + shapes[rnd_scan_dim + 1 :]

            for op, op_pt, init in [
                (
                    get_scan_combine_fn("add", False),
                    torch.cumsum,
                    torch.zeros(*init_shapes, device=device, requires_grad=autograd),
                ),
                (
                    get_scan_combine_fn("mul", False),
                    torch.cumprod,
                    torch.ones(*init_shapes, device=device, requires_grad=autograd),
                ),
            ]:
                result = scan_fct(op, init, x, dim=rnd_scan_dim, reverse=reverse)
                result_exp = _fake_scan(
                    op, init=init, xs=x, dim=rnd_scan_dim, reverse=reverse
                )
                self.assertEqual(result, result_exp)
                if not reverse:
                    result_exp_PT = op_pt(x, rnd_scan_dim)
                    self.assertEqual(result[1], result_exp_PT)

                if autograd:
                    self.check_autograd(result, result_exp, (init, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_binary_operator(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA. Runs scan with the S5 operator over list carry
        #   and list inputs built from a repeated vector and random inputs, compares with fake
        #   scan, and with autograd on, compares gradients wrt all inits and elements.
        state_dim = 20
        timesteps = 10
        scan_fct = compile_mode_helper(scan, compile_mode)

        projected_inputs = torch.randn(
            timesteps, state_dim, requires_grad=autograd, device=device
        )
        A = torch.randn(state_dim, requires_grad=autograd, device=device)
        elements = (A.repeat((timesteps, 1)), projected_inputs)
        init = tuple(
            [
                torch.ones_like(
                    torch._ops.ops.aten.slice(elements[0], 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
            + [
                torch.zeros_like(
                    torch._ops.ops.aten.slice(projected_inputs, 0, 0, 1, 1),
                    requires_grad=autograd,
                )
            ]
        )

        init_clone = [i.clone() for i in init]
        init_clone2 = [i.clone() for i in init]
        elements_clone = [ele.clone() for ele in elements]
        elements_clone2 = [ele.clone() for ele in elements]
        result = scan_fct(
            get_scan_combine_fn("s5_operator", False),
            init_clone,
            elements_clone,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("s5_operator", False),
            init_clone2,
            elements_clone2,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            result_flatten, _ = pytree.tree_flatten(result)
            result_exp_flatten, _ = pytree.tree_flatten(expected_result)

            grad_out = [torch.ones_like(el) for el in result_exp_flatten]
            expected_grads = torch.autograd.grad(
                result_exp_flatten, (*init_clone2, *elements_clone2), grad_out
            )
            grads = torch.autograd.grad(
                result_flatten, (*init_clone, *elements_clone), grad_out
            )
            self.assertEqual(grads, expected_grads)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_tuple(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA. Scan with tuple carry and tuple inputs: first
        #   with a combine function whose output equals the carry, then with one whose output is a
        #   different tuple; compares with the fake scan, checks the second output matches the
        #   first run's second output, and checks gradients when autograd is on.
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        scan_fct = compile_mode_helper(scan, compile_mode)

        result_same = scan_fct(
            get_scan_combine_fn("tuple_fct", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("tuple_fct", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result_same, expected_result)

        if autograd:
            self.check_autograd(result_same, expected_result, (init, inp))

        def fct_different_output_tuple(x, y):
            # [FUNCTION DOC] HELPER: returns a pair: the new carry tuple made of the sum
            #   of first entries and the product of second entries and the output (product
            #   of second entries).
            return ((x[0] + y[0], x[1] * y[1]), (x[1] * y[1]))

        inp = (x, y)
        init = tuple(torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp)

        result_diff = scan(
            fct_different_output_tuple, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_different_output_tuple, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result_diff, expected_result)
        self.assertEqual(result_diff[1], result_same[1][1])

        if autograd:
            self.check_autograd(result_diff, expected_result, (init, inp))

    def test_scan_wrong_pytree(self):
        # [FUNCTION DOC] Combine function returns a carry dict with an extra key compared to init;
        #   expects UncapturedHigherOrderOpError complaining init and carry do not have the same
        #   number of outputs.
        # Init and input have same pytree
        def fct_wrong_pytree(x, y):
            # [FUNCTION DOC] HELPER: returns two dicts (carry and output) built from the input
            #   with an extra "k" key, so the carry structure differs from init.
            return (
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": (
                        [x["j"][1][0]["o"].clone()],
                        [{"o": torch.sin(x["i"])}],
                    ),
                },
                {
                    "i": x["i"] * y["j"][0][0],
                    "k": torch.tensor(0.0),
                    "j": ([x["j"][1][0]["o"].clone()], [{"o": torch.sin(x["i"])}]),
                },
            )

        x = torch.randn(3, 2, 2)
        y = torch.randn(3, 2, 2)
        z = torch.randn(3, 2, 2)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # r"The tree structure of the inits and the carries are not identical.*",
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan(fct_wrong_pytree, init, inp, dim=0)

    def test_scan_float_output(self):
        # [FUNCTION DOC] Combine function returns a Python float as the carry; expects
        #   UncapturedHigherOrderOpError naming the scan higher-order op.
        # Init and input have same pytree
        def fct_float_output(x, y):
            # [FUNCTION DOC] HELPER: returns the float 0.0 as carry and x plus y as output.
            return 0.0, x + y

        x = torch.randn(3, 2, 2)
        init = torch._ops.ops.aten.slice(x, 0, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be:
            # torch._dynamo.exc.Unsupported,
            # "HigherOrderOperator body's output must consist of tensors or ints only"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_float_output, init, x, dim=0)

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_complex_pytree(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Needs CUDA. Scan with a nested dict/list pytree carry and input using the
        #   complex pointwise combine function; compares with fake scan across reverse, compile
        #   mode, device, and checks gradients when autograd is on.
        # Init and input have same pytree

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}
        inp_flat, inp_spec = pytree.tree_flatten(inp)
        init_flat = [torch._ops.ops.aten.slice(e, 0, 0, 1, 1) for e in inp_flat]
        init = pytree.tree_unflatten(init_flat, inp_spec)

        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("complex_pointwise", False),
            init=init,
            xs=inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, inp))

    # TODO: Does not work because of the usage of vmap within associative_scan
    # The paT206899919 rameterization is commented out for the moment and the test is marked with expected fail
    # Fails with: AssertionError: scan is not an OpOverload
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_associative_scan(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Scan (compiled) whose body calls associative_scan
        #   (uncompiled, reversed, generic combine mode) over the per-step input; compares with
        #   the fake scan.
        # KNOWN ISSUE: the comment above says this does not work because of vmap use inside
        #   associative_scan and that it should be marked expected fail, but no such marker is
        #   present on the test.
        combine_mode = "generic"
        compile_mode_scan = "compile"
        compile_mode_associative_scan = "none"
        reverse = True
        reverse_associative_scan = True
        device = torch.device("cuda")

        scan_fct = compile_mode_helper(scan, compile_mode_scan)
        associative_scan_fct = compile_mode_helper(
            associative_scan, compile_mode_associative_scan
        )
        init = torch.randn(10, 5, device=device)
        inp = torch.randn(3, 10, 5, device=device)

        def body(x, y):
            # [FUNCTION DOC] HELPER: runs the reversed associative scan with the add combine
            #   function on y, and returns x plus y as carry and x plus the scan result as output.
            val = associative_scan_fct(
                get_scan_combine_fn("add", True),
                y,
                0,
                reverse=reverse_associative_scan,
                combine_mode=combine_mode,
            )
            return x + y, x + val

        result = scan_fct(body, init, inp, dim=0, reverse=reverse)
        expected_result = _fake_scan(
            body,
            init,
            inp,
            0,
            reverse=reverse,
        )

        self.assertEqual(result, expected_result)

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_matmul(self, compile_mode, reverse, device, autograd):
        # [FUNCTION DOC] Skipped under dynamo; needs CUDA. Scans over dim 1 then multiplies the
        #   selected scan result (carry or stacked output, for index 0 and 1) by a ones matrix;
        #   compares the (optionally compiled) chain with the fake scan version and checks
        #   gradients when autograd is on.
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        for ind in range(2):
            # Chain with matmul
            def chain_fct(inp):
                # [FUNCTION DOC] HELPER: runs scan with the add combine function over dim 1 and
                #   returns the selected output (index from the enclosing loop) matrix-multiplied
                #   by a 2 by 5 ones matrix.
                W = torch.ones(2, 5, device=device)
                o = scan(
                    get_scan_combine_fn("add", False),
                    init,
                    inp,
                    dim=1,
                    reverse=reverse,
                )
                return o[ind] @ W

            fct_cmp = compile_mode_helper(chain_fct, compile_mode)

            expected_result = _fake_scan(
                get_scan_combine_fn("add", False),
                init=init,
                xs=inp,
                dim=1,
                reverse=reverse,
            )[ind] @ torch.ones(2, 5, device=device)
            result = fct_cmp(inp)
            self.assertEqual(result, expected_result)

            if autograd:
                self.check_autograd(result, expected_result, (init, inp))

    # TODO: provide an implementation for all compile modes and re-enable all test
    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_downstream_scan_scan_dim(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Skipped under dynamo; needs CUDA. Chains two scans, the second scanning
        #   dim 0 over the stacked output of the first (which scanned dim 1); compares with fake
        #   scans and checks gradients wrt both inits and the input when autograd is on.
        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(3, 2, device=device, requires_grad=autograd)

        # Chain with scan on different dim
        init2 = torch.randn(1, 10, 2, device=device, requires_grad=autograd)

        def chain_fct_different_dim(inp):
            # [FUNCTION DOC] HELPER: first scan over dim 1 with init, then a second scan over dim
            #   0 of the first scan's stacked output with init2; returns the second scan result.
            o1 = scan(
                get_scan_combine_fn("add", False),
                init,
                inp,
                dim=1,
                reverse=reverse,
            )
            o2 = scan(
                get_scan_combine_fn("add", False),
                init2,
                o1[1],
                dim=0,
                reverse=reverse,
            )
            return o2

        fct_cmp = compile_mode_helper(chain_fct_different_dim, compile_mode)

        xs = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init,
            xs=inp,
            dim=1,
            reverse=reverse,
        )[1]
        expected_result = _fake_scan(
            get_scan_combine_fn("add", False),
            init=init2,
            xs=xs,
            dim=0,
            reverse=reverse,
        )
        result = fct_cmp(inp)
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, init2, inp))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_non_pointwise(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA. Scan with the non-pointwise combine function
        #   (matrix multiplication by a fixed matrix) compared to the fake scan, with
        #   optional gradient check.
        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        init = torch.randn(10, 2, device=device, requires_grad=autograd)
        expected_result = _fake_scan(
            get_scan_combine_fn("non_pointwise", False),
            init=init,
            xs=x,
            dim=0,
            reverse=reverse,
        )

        result = scan_fct(
            get_scan_combine_fn("non_pointwise", False),
            init,
            x,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            self.check_autograd(result, expected_result, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_compile_cnt(self, reverse, device):
        # [FUNCTION DOC] Needs CUDA. With automatic dynamic shapes enabled, counts dynamo
        #   frames via a compile counter across a sequence of scan calls varying length, scan
        #   dim and reverse flag; asserts the frame count goes 1, 2, 2, 3, 4, 4, 5, 6, 6, 6,
        #   i.e. recompiles on first size change, dim change and reverse change but not on
        #   further size changes.
        dim = 1

        from torch._dynamo.testing import CompileCounter

        # Tests rely on automatic_dynamic = True
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            cnt = CompileCounter()
            x = torch.randn(3, 2, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # First compilation step
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 1)

            x = torch.randn(3, 20, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # Recompilation due to first different size
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 5, device=device)
            # No recompilation, because of dynamic shape
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 2)

            x = torch.randn(3, 40, 5, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 3)

            x = torch.randn(3, 40, 20, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation due to first different size on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 40, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, because of dynamic shape on new dim
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=2,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 4)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of dim change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 5)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # Recompilation because of reverse change
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 60, 40, device=device)
            init = torch.randn(3, 40, device=device)
            # No recompilation, as nothing changed
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=not reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

            x = torch.randn(3, 120, 80, device=device)
            init = torch.randn(3, 80, device=device)
            # No recompilation, final test
            torch.compile(scan, backend=cnt)(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=1,
                reverse=reverse,
            )
            self.assertEqual(cnt.frame_count, 6)

    def test_scan_operator_call_count(self):
        # [FUNCTION DOC] Uses generic_scan with a combine function that counts its calls: five
        #   steps give five calls, and a single-element scan gives exactly one call.
        from torch._higher_order_ops.scan import generic_scan, wrap_combine_fn_flat

        counter = [0]

        def counting_combine(carry, x):
            # [FUNCTION DOC] HELPER: increments the first slot of the shared counter list and
            #   returns carry plus x and x.
            counter[0] += 1
            return carry + x, x

        init = [torch.zeros(3)]
        xs = [torch.ones(5, 3)]
        combine_flat = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=counting_combine,
            spec_init=pytree.tree_flatten(init)[1],
            spec_xs=pytree.tree_flatten(xs)[1],
            num_init_leaves=len(init),
            num_inp_leaves=len(xs),
        )

        counter[0] = 0
        result = generic_scan(combine_flat, init, xs)  # noqa: F841
        self.assertEqual(counter[0], 5)

        # Single-element scan should call operator exactly once.
        xs_one = [torch.ones(1, 3)]
        combine_flat_one = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=counting_combine,
            spec_init=pytree.tree_flatten(init)[1],
            spec_xs=pytree.tree_flatten(xs_one)[1],
            num_init_leaves=len(init),
            num_inp_leaves=len(xs_one),
        )
        counter[0] = 0
        generic_scan(combine_flat_one, init, xs_one)
        self.assertEqual(counter[0], 1)

    @skipIfTorchDynamo("don't test compile on compile")
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    def test_scan_zero_length(self, reverse, compile_mode):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over reverse and compile modes (none,
        #   eager, compile, compile_dynamic_shape). Scans over inputs with a zero-length scan
        #   dimension for bodies that keep shape, change shape and dtype, or use a tuple input;
        #   compares with the fake scan.
        def body_same(c, x):
            # [FUNCTION DOC] HELPER: carry plus x as new carry and a clone as output.
            y = c + x
            return y, y.clone()

        def body_reduce(c, x):
            # [FUNCTION DOC] HELPER: new carry is carry plus x; output is the sum over the last
            #   dim cast to float64, so output shape and dtype differ from x.
            # y shape and dtype both differ from the xs slice.
            y = c + x
            return y, y.sum(dim=-1).to(torch.float64).clone()

        def body_struct(c, x):
            # [FUNCTION DOC] HELPER: x is a pair; new carry is carry plus both parts; output is a
            #   clone of it.
            # xs is a 2-tuple, but the body emits a single ys leaf.
            x0, x1 = x
            y = c + x0 + x1
            return y, y.clone()

        cases = [
            (body_same, torch.tensor([0.0, 1.0]), torch.zeros(0, 2), 0),
            (body_reduce, torch.arange(6.0).reshape(2, 3), torch.ones(0, 2, 3), 0),
            (body_reduce, torch.arange(6.0).reshape(2, 3), torch.ones(2, 0, 3), 1),
            (body_struct, torch.zeros(2), (torch.ones(0, 2), torch.ones(0, 2)), 0),
        ]
        scan_fct = compile_mode_helper(scan, compile_mode)
        for body, init, xs, dim in cases:
            result = scan_fct(body, init, xs, dim=dim, reverse=reverse)
            expected = _fake_scan(body, init, xs, dim=dim, reverse=reverse)
            self.assertEqual(result, expected)

    @skipIfTorchDynamo("don't test compile on compile")
    @parametrize("reverse", [False, True])
    def test_scan_zero_length_autograd(self, reverse):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over reverse. Scans over a zero-length
        #   xs with init and xs requiring grad, backpropagates the sum of carry and outputs, and
        #   checks the grad of init is all ones (no steps run, carry equals init) and that the
        #   grad of xs has the same shape as xs.
        init = torch.randn(2, 3, requires_grad=True)
        xs = torch.randn(0, 2, 3, requires_grad=True)

        def body(c, x):
            # [FUNCTION DOC] HELPER: new carry is the sine of carry plus x; the output is a
            #   clone of it.
            y = torch.sin(c + x)
            return y, y.clone()

        carry, ys = scan(body, init, xs, reverse=reverse)
        (carry.sum() + ys.sum()).backward()
        # No steps run, so the carry passes through to init (grad of ones) and xs
        # receives a correctly-shaped zero-length gradient.
        self.assertEqual(init.grad, torch.ones_like(init))
        self.assertEqual(xs.grad.shape, xs.shape)

    def test_scan_unequal_scan_dim_size(self):
        # [FUNCTION DOC] Scan with a two-leaf xs whose leading sizes differ (0 and 5) raises
        #   RuntimeError saying all xs leaves must have the same scan dimension size.
        def body(c, x):
            # [FUNCTION DOC] HELPER: xs is a pair; new carry is carry plus both parts; output is a
            #   clone of it.
            x0, x1 = x
            y = c + x0 + x1
            return y, y.clone()

        with self.assertRaisesRegex(
            RuntimeError, "All xs leaves must have the same scan dimension size"
        ):
            scan(body, torch.zeros(2), (torch.ones(0, 2), torch.ones(5, 2)))

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_non_tensor(self):
        # [FUNCTION DOC] Skipped under dynamo. Passing a Python float as init raises RuntimeError
        #   saying all init leaves must be tensors.
        x = torch.randn(3, 1, 2, device=torch.device("cpu"))
        dim = 1

        # Init is a float and not a tensor
        init = 1.0
        with self.assertRaisesRegex(RuntimeError, "All init leaves must be a Tensor.*"):
            scan(get_scan_combine_fn("add", False), init, x, dim=dim, reverse=False)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_shape(self):
        # [FUNCTION DOC] Skipped under dynamo. Init with a shape that does not match the carry
        #   produced by the add combine function raises UncapturedHigherOrderOpError complaining
        #   init and carry metadata differ.
        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong shape (Other dim different)
        init = torch.randn(1, 2)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same metadata.*",
        ):
            scan_fct(
                get_scan_combine_fn("add", False),
                init,
                x,
                dim=dim,
            )

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_longer_carry(self):
        # [FUNCTION DOC] Skipped under dynamo. Init has two entries but the combine function
        #   returns a carry with different arity; expects UncapturedHigherOrderOpError about
        #   differing numbers of init and carry outputs.
        def init_longer_carry(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns a pair of a single tensor plus one and an output of y
            #   plus one; the carry structure does not match the two-entry init.
            return x[0] + 1.0, y + 1.0

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = (
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
            torch._ops.ops.aten.slice(x, dim, 0, 1, 1),
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_longer_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_init_shorter_carry(self):
        # [FUNCTION DOC] Skipped under dynamo. Init is a single tensor but the combine function
        #   returns a two-tensor carry; expects UncapturedHigherOrderOpError about differing
        #   numbers of init and carry outputs.
        def init_shorter_carry(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns a carry tuple of two tensors (x plus 1, x plus 2) and
            #   an output x plus 3.
            return (x + 1, x + 2), x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # The tree structure of the inits and the carries are not identical!
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same number of outputs but got lhs.*",
        ):
            scan_fct(init_shorter_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_init_wrong_pytree_carry_shape(self):
        # [FUNCTION DOC] Skipped under dynamo. The combine function returns a carry that is a
        #   view-like slice of its input with a different shape; expects
        #   UncapturedHigherOrderOpError naming the scan higher-order op.
        def wrong_carry_shape(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns the first row slice of x as carry and x plus
            #   3 as output.
            return x[0, :], x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan_fct(wrong_carry_shape, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    def test_scan_one_return(self):
        # [FUNCTION DOC] Skipped under dynamo. Combine function returns a single tensor instead of
        #   a carry/output pair; expects UncapturedHigherOrderOpError naming the scan op.
        def no_carry(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns only x plus 3.
            return x + 3

        scan_fct = compile_mode_helper(scan, "none")

        # Only init and no input
        x = torch.randn(3, 1, 2)
        dim = 1

        # Init wrong pytree
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # combine_fn needs to produce two pytrees, one for the carries and one for the outputs.
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan_fct(no_carry, init, x, dim=dim)

    @skipIfTorchDynamo("don't test compile on compile")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Skipped under dynamo; needs CUDA; parametrized over reverse, compile
        #   mode, device, autograd. Exercises scan inits: init with no xs (result carry equals
        #   init), scalar init with a carry that grows by one, init of unrelated shape, init
        #   spanning two steps of the scan dim, a tiled init with sliced outputs (checking
        #   result shapes), and the regular case compared with cumsum; checks gradients when
        #   autograd is on.
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Only init and no input
        x = torch.randn(3, 1, 2, device=device, requires_grad=autograd)
        dim = 1
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)

        # Only init given
        init = torch._ops.ops.aten.slice(x, dim, 0, 1, 1)
        result = scan_fct(op, init, [], dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=[], dim=dim, reverse=reverse)
        result_init = scan_fct(op, init, [], dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], init)

        if autograd:
            self.check_autograd(result, result_exp, (init,))

        x = torch.randn(3, 5, 2, device=device, requires_grad=autograd)
        dim = 0

        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        inp = torch._ops.ops.aten.slice(x, dim, 1, None, 1)

        # Init tensor scalar
        init = torch.ones(1, device=device, requires_grad=autograd)

        def add_scalar_carry(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry plus 1.0 as new carry and carry plus input as output.
            return x + 1.0, x + y

        result_init = scan_fct(add_scalar_carry, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0], torch.tensor([3.0], device=device))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Init tensor entirely different shape than inp
        init = torch.randn(7, 8, device=device, requires_grad=autograd)

        def add_scalar_carry2(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry plus 1.0 as new carry; output is the top-left block of
            #   the carry (sized like y) plus y.
            return x + 1.0, x[: y.shape[0], : y.shape[1]] + y

        result_init = scan_fct(add_scalar_carry2, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            add_scalar_carry2, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)

        # Init with two timestep on dim axis. Should work as y has always 1 on dim axis and
        # hence automatic broadcasting should work
        # I.e., the input shape is 2x5x2, but the carry at each iteration is 2x5x2,
        # thus the output of each iteration is 2x5x2, which results in the total output
        # to be 4x5x2
        init = torch._ops.ops.aten.slice(x, dim, 0, 2, 1)
        result_init = scan_fct(op, init, inp, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=inp, dim=dim, reverse=reverse)
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        init = torch.tile(init, (1, 2, 1))

        def add_scalar_carry_sliced_out(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry plus 1.0 as new carry; output is the first slice along
            #   dim 1 of the carry plus y.
            return x + 1.0, x[:, :1, :] + y

        result_init = scan_fct(
            add_scalar_carry_sliced_out, init, inp, dim=dim, reverse=reverse
        )
        result_exp = _fake_scan(
            add_scalar_carry_sliced_out, init=init, xs=inp, dim=dim, reverse=reverse
        )
        self.assertEqual(result_init, result_exp)
        self.assertEqual(result_init[0].shape, torch.Size([2, 10, 2]))
        self.assertEqual(result_init[1].shape, torch.Size([2, 2, 5, 2]))

        if autograd:
            self.check_autograd(result_init, result_exp, (init, inp))

        # Correct case
        op, op_pt = (get_scan_combine_fn("add", False), torch.cumsum)
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        init = torch.zeros(3, 2, device=device, requires_grad=autograd)
        dim = 2

        result = scan_fct(op, init, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(op, init=init, xs=x, dim=dim, reverse=reverse)

        self.assertEqual(result, result_exp)
        if not reverse:
            result_exp_PT = op_pt(x, dim)
            self.assertEqual(result[1], result_exp_PT)

        if autograd:
            self.check_autograd(result, result_exp, (init, x))

    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_init_wrong_pytree_complex(self, reverse, device):
        # [FUNCTION DOC] Needs CUDA; parametrized over reverse and device. Scan with a dict init
        #   whose structure differs from the input/carry structure expected by the complex
        #   pointwise combine function; expects some Exception to be raised.
        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        z = torch.randn(3, 2, 2, device=device)

        # Wrong pytree fed to the function
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, 1, 1),
            "j": (
                {"a": torch._ops.ops.aten.slice(x, 0, 0, 1, 1)},
                [torch._ops.ops.aten.slice(y, 0, 0, 1, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, 1, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, 0, None, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, 0, None, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, 0, None, 1)}],
            ),
        }
        with self.assertRaisesRegex(
            Exception,
            ".*",
        ):
            scan(
                get_scan_combine_fn("complex_pointwise", False),
                init,
                inp,
                dim=0,
                reverse=reverse,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_init_pytree_complex(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA. Scan with dict/list/tuple pytree init and inputs,
        #   using first the regular complex pointwise function, then combine functions whose
        #   output pytree differs, and whose carry pytree differs, from the input; compares with
        #   fake scan and checks gradients on leaves when autograd is on. Slicing of init and
        #   inputs depends on the reverse flag.
        def fct_pointwise_different_output(x, y):
            # [FUNCTION DOC] HELPER: returns a carry dict (products and sums, same structure as
            #   init) and an output with a different structure (a tuple of scaled y and a dict).
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        [x["j"][0][0] * y["j"][0][0]],
                        [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"],
                        "j": (
                            [x["j"][0][0] * y["j"][0][0]],
                            [{"o": x["j"][1][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        def fct_pointwise_different_carry(x, y):
            # [FUNCTION DOC] HELPER: returns a carry dict whose structure differs from the input
            #   (includes an extra tuple member) and an output tuple of a scaled value and a dict.
            return (
                {
                    "i": x["i"] * y["i"],
                    "j": (
                        x["i"] * 2,
                        [x["j"][1][0] * y["j"][0][0]],
                        [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                    ),
                },
                (
                    y["i"] * 2,
                    {
                        "o": x["i"] * y["i"] + x["j"][0][0],
                        "j": (
                            [x["j"][1][0] * y["j"][0][0]],
                            [{"o": x["j"][2][0]["o"] + y["j"][1][0]["o"]}],
                        ),
                    },
                ),
            )

        scan_fct = compile_mode_helper(scan, compile_mode)

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        if reverse:
            init_start, init_end = -1, None
            inp_start, inp_end = 0, -1
        else:
            init_start, init_end = 0, 1
            inp_start, inp_end = 1, None

        # Regular case
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, init_start, init_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, init_start, init_end, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, inp_start, inp_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, inp_start, inp_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, inp_start, inp_end, 1)}],
            ),
        }
        result = scan_fct(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        expected_result = _fake_scan(
            get_scan_combine_fn("complex_pointwise", False),
            init,
            inp,
            dim=0,
            reverse=reverse,
        )
        self.assertEqual(result, expected_result)

        if autograd:
            init_flat = pytree.tree_leaves(init)
            inp_flat = pytree.tree_leaves(inp)
            self.check_autograd(result, expected_result, (*init_flat, *inp_flat))

        # Pytree of output is different
        result = scan_fct(
            fct_pointwise_different_output, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_pointwise_different_output, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result, expected_result)

        # Pytree of carry is different
        init = {
            "i": torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
            "j": (
                torch._ops.ops.aten.slice(x, 0, init_start, init_end, 1),
                [torch._ops.ops.aten.slice(y, 0, init_start, init_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, init_start, init_end, 1)}],
            ),
        }
        inp = {
            "i": torch._ops.ops.aten.slice(x, 0, inp_start, inp_end, 1),
            "j": (
                [torch._ops.ops.aten.slice(y, 0, inp_start, inp_end, 1)],
                [{"o": torch._ops.ops.aten.slice(z, 0, inp_start, inp_end, 1)}],
            ),
        }
        result = scan_fct(
            fct_pointwise_different_carry, init, inp, dim=0, reverse=reverse
        )
        expected_result = _fake_scan(
            fct_pointwise_different_carry, init=init, xs=inp, dim=0, reverse=reverse
        )
        self.assertEqual(result, expected_result)

        if autograd:
            init_flat = pytree.tree_leaves(init)
            inp_flat = pytree.tree_leaves(inp)
            self.check_autograd(result, expected_result, (*init_flat, *inp_flat))

    @skipIfTorchDynamo("don't test compile on compile")
    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_scan_pytree_output(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Compiles a reversed scan with a tuple
        #   carry using the eager recording backend and compares the golden graph: flip of xs,
        #   scan higher-order op with a combine subgraph, and flip of the stacked output.
        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            # [FUNCTION DOC] HELPER: reversed scan along dim 0 of the given function.
            return scan(fct, init, xs, dim=0, reverse=True)

        def combine_fn(init, x):
            # [FUNCTION DOC] HELPER: carry pair becomes (first plus x, second minus x); output is
            #   the difference of the two new carries.
            a, b = (init[0] + x, init[1] - x)
            return (a, b), a - b

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(combine_fn, (init, init.clone()), x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_init_0_: "f32[1, 10, 2]", L_init_1_: "f32[1, 10, 2]", L_xs_: "f32[3, 10, 2]"):
        l_init_0_ = L_init_0_
        l_init_1_ = L_init_1_
        l_xs_ = L_xs_

        flip: "f32[3, 10, 2]" = torch.flip(l_xs_, [0]);  l_xs_ = None
        scan_combine_fn_0 = self.scan_combine_fn_0
        scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_0_, l_init_1_], [flip], []);  scan_combine_fn_0 = l_init_0_ = l_init_1_ = flip = None
        getitem: "f32[1, 10, 2]" = scan[0]
        getitem_1: "f32[1, 10, 2]" = scan[1]
        out: "f32[3, 1, 10, 2]" = scan[2];  scan = None
        out_1: "f32[3, 1, 10, 2]" = out.flip([0]);  out = None
        return (getitem, getitem_1, out_1)

    class scan_combine_fn_0(torch.nn.Module):
        def forward(self, child: "f32[1, 10, 2]", child_1: "f32[1, 10, 2]", child_2: "f32[10, 2]"):
            a: "f32[1, 10, 2]" = child + child_2;  child = None
            b: "f32[1, 10, 2]" = child_1 - child_2;  child_1 = child_2 = None

            child_3: "f32[1, 10, 2]" = a - b
            return [a, b, child_3]
""",
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("autograd", [False, True])
    def test_scan_closure_RNN(self, compile_mode, autograd):
        # [FUNCTION DOC] Skipped under dynamo; requires SM70 and CUDA. Compares scan with a
        #   closure-based RNN cell combine function (weights passed as parameters) to
        #   torch.nn.RNN: final state and stacked outputs; with autograd also compares gradients
        #   wrt h, x and the weights (transposing weight gradients).
        dim = 1
        device = torch.device("cpu")
        scan_fct = compile_mode_helper(scan, compile_mode)

        rnn = torch.nn.RNN(
            input_size=5,
            hidden_size=7,
            batch_first=True,
        )
        rnn = rnn.to(device=device)
        x = torch.randn(3, 10, 5, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)

        W_ih = rnn.weight_ih_l0.T.clone()
        b_ih = rnn.bias_ih_l0.clone()
        W_hh = rnn.weight_hh_l0.T.clone()
        b_hh = rnn.bias_hh_l0.clone()

        if not autograd:
            W_ih = W_ih.detach()
            b_ih = b_ih.detach()
            W_hh = W_hh.detach()
            b_hh = b_hh.detach()

        expected_result = rnn(x, torch.unsqueeze(h, 0))
        expected_result_out = expected_result[0]
        expected_result_state = expected_result[1][0, :]

        result = scan_fct(
            get_scan_combine_fn("RNN", True, parameters=[W_ih, b_ih, W_hh, b_hh]),
            h,
            x,
            dim=dim,
            reverse=False,
        )
        self.assertEqual(result[0], expected_result_state)
        self.assertEqual(result[1], expected_result_out)

        if autograd:
            result_flat = pytree.tree_leaves(result)
            result_exp_flat = [expected_result_state, expected_result_out]

            grad_out_expected = [torch.ones_like(r) for r in result_exp_flat]
            expected_grads = torch.autograd.grad(
                result_exp_flat,
                (
                    h,
                    x,
                    rnn.weight_ih_l0,
                    rnn.bias_ih_l0,
                    rnn.weight_hh_l0,
                    rnn.bias_hh_l0,
                ),
                grad_out_expected,
            )
            expected_add_input_grads = list(expected_grads[2:])
            expected_grads = expected_grads[:2]

            grad_out = [torch.ones_like(r) for r in result]
            grads = torch.autograd.grad(
                result_flat, (h, x, W_ih, b_ih, W_hh, b_hh), grad_out
            )
            add_input_grads = list(grads[2:])
            add_input_grads[0] = add_input_grads[0].T
            add_input_grads[2] = add_input_grads[2].T
            grads = grads[:2]
            self.assertEqual(grads, expected_grads)
            self.assertEqual(add_input_grads, expected_add_input_grads)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize(
        "partial_grad", ["xs", "init", "additional_inputs", "complex", "random"]
    )
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_scan_closure_RNN_partial_autograd(
        self, reverse, compile_mode, partial_grad, device
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over reverse, compile mode, partial-
        #   grad scenario (xs, init, additional_inputs, complex, random) and device. Builds eight
        #   tensors (two xs, two inits, four closure parameters) with various requires_grad masks
        #   (random masks in the random scenario, unseeded), scans with a two-carry RNN, compares
        #   with fake scan, and checks gradients only for outputs requiring grad.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)

        # The first two booleans are the xs
        # The second two are the inits
        # The last four are the additional_inputs
        autograds = []

        if partial_grad == "xs":
            # xs tests
            autograds.append([True, False, True, True, True, True, True, True])
            autograds.append([False, False, True, True, True, True, True, True])
            autograds.append([True, False, False, False, False, False, False, False])
        elif partial_grad == "init":
            # init tests
            autograds.append([True, True, False, True, True, True, True, True])
            autograds.append([True, True, False, False, True, True, True, True])
            autograds.append([False, False, False, True, False, False, False, False])
        elif partial_grad == "additional_inputs":
            # additional input tests
            autograds.append([True, True, True, True, False, True, False, True])
            autograds.append([True, True, True, True, False, False, True, False])
            autograds.append([False, False, False, False, False, True, False, True])
            autograds.append([False, False, False, False, False, False, True, False])
        elif partial_grad == "complex":
            # complex cases
            autograds.append([True, False, False, False, False, False, False, True])
            autograds.append([False, False, True, True, False, False, False, True])
        elif partial_grad == "random":
            # random tests
            import random

            for _ in range(5):
                autograds.append([bool(random.randint(0, 1)) for _ in range(8)])

        for autograd in autograds:
            x = torch.randn(3, 10, 5, device=device, requires_grad=autograd[0])
            x1 = torch.randn(3, 10, 5, device=device, requires_grad=autograd[1])
            h = torch.randn(3, 7, device=device, requires_grad=autograd[2])
            h_1 = torch.randn(3, 7, device=device, requires_grad=autograd[3])
            W_ih = torch.randn(5, 7, device=device, requires_grad=autograd[4])
            b_ih = torch.randn(7, device=device, requires_grad=autograd[5])
            W_hh = torch.randn(7, 7, device=device, requires_grad=autograd[6])
            b_hh = torch.randn(7, device=device, requires_grad=autograd[7])

            params = [
                p
                for p, a in zip([x, x1, h, h_1, W_ih, b_ih, W_hh, b_hh], autograd)
                if a
            ]

            def RNN(x: torch.Tensor, y: torch.Tensor):
                # [FUNCTION DOC] HELPER: combine using closure weights; carries (first entry of x
                #   plus b_hh, second entry of x plus 1) and outputs a tanh/matmul combination.
                c_new_0 = x[0] + b_hh
                c_new_1 = x[1] + 1
                h_new = (
                    torch.tanh(c_new_1 + x[0] @ W_hh + b_hh)
                    + y[0] @ W_ih
                    + y[1] @ W_ih
                    + b_ih
                    + x[1]
                )
                return (c_new_0, c_new_1), h_new

            inits = (h, h_1)
            result = scan_fct(RNN, inits, (x, x1), dim=dim, reverse=reverse)
            result_exp = _fake_scan(RNN, (h, h_1), (x, x1), dim=dim, reverse=reverse)
            self.assertEqual(result, result_exp)

            if autograd:
                result_flat = pytree.tree_leaves(result)
                result_exp_flat = pytree.tree_leaves(result_exp)
                exp_grad_mask = [bool(r.requires_grad) for r in result_exp_flat]
                self.check_autograd(
                    [r for r, m in zip(result_flat, exp_grad_mask) if m],
                    [r for r, m in zip(result_exp_flat, exp_grad_mask) if m],
                    params,
                )

    def test_scan_break_bw_input_output_aliasing(self):
        # [FUNCTION DOC] Unit test of the ScanAutogradImpl pass that clones backward-graph outputs
        #   aliasing placeholders. Builds an fx graph with a computed output, a direct placeholder
        #   output and a view of a placeholder output, runs the pass on a stub instance, and
        #   asserts the computed output is unchanged while the direct and transitive aliases are
        #   replaced by clone nodes of the right inputs.
        # Focused test for ScanAutogradImpl._break_bw_input_output_aliasing.
        # The partitioner naturally produces direct placeholder outputs (covered
        # by test_scan_closure_combine_fn_with_no_grad_additional_inputs_partial),
        # but transitive aliases (a view of a placeholder) don't arise from real
        # fixtures, so we hand-craft a bw_gm that returns both kinds and a
        # non-aliasing computed output, then check the pass clones the right ones.
        from torch._higher_order_ops.scan import ScanAutogradImpl

        graph = torch.fx.Graph()
        ph_tangent = graph.placeholder("tangent")
        ph_tangent.meta["val"] = torch.empty(2, 3)
        ph_zeros = graph.placeholder("zeros_like")
        ph_zeros.meta["val"] = torch.empty(6)
        # Non-aliasing computed gradient.
        computed = graph.call_function(
            torch.ops.aten.mul.Tensor, args=(ph_tangent, 2.0)
        )
        computed.meta["val"] = ph_tangent.meta["val"] * 2.0
        # Transitive alias: view of a placeholder.
        view = graph.call_function(torch.ops.aten.view.default, args=(ph_zeros, [2, 3]))
        view.meta["val"] = ph_zeros.meta["val"].view(2, 3)
        graph.output((computed, ph_zeros, view))
        bw_gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        # Stub instance: only hop_partitioned_graph.bw_gm is read by the pass.
        impl = ScanAutogradImpl.__new__(ScanAutogradImpl)
        impl.hop_partitioned_graph = type("_S", (), {"bw_gm": bw_gm})()

        impl._break_bw_input_output_aliasing()

        outs = next(iter(bw_gm.graph.find_nodes(op="output"))).args[0]
        self.assertIs(outs[0], computed)  # not an alias of any input
        self.assertEqual(outs[1].target, torch.ops.aten.clone.default)  # direct
        self.assertEqual(outs[1].args, (ph_zeros,))
        self.assertEqual(outs[2].target, torch.ops.aten.clone.default)  # transitive
        self.assertEqual(outs[2].args, (view,))

    @requires_cuda
    @skipIfTorchDynamo("not a dynamo test")
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("layers", [1, 2, 3])
    @parametrize("device", ["cpu", "cuda"])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_multiple_layers_gradient(self, layers, device):
        # [FUNCTION DOC] Needs CUDA, SM70, and not dynamo; parametrized over layer count (1, 2, 3)
        #   and device; capture_scalar_outputs enabled. Defines three equivalent RNN-like models
        #   (python loop, scan with list carry, scan with stacked tensor carry), compiles each,
        #   and checks that layer weight gradients and loss equal those of the uncompiled copy
        #   with identical weights.
        import torch.nn as nn

        torch.manual_seed(1)

        LAYERS = layers
        BATCH_SIZE = 2
        SEQ_LEN = 5
        FEATURE_DIM = 10
        DEVICE = device

        class RNNLoop(nn.Module):
            # [CLASS DOC] HELPER: model with a stack of linear layers that unrolls over time in a
            #   Python loop, updating a list of hidden states.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates a ModuleList of Linear layers from twice the
                #   feature dim to the feature dim, and records layer count.
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, inputs_sequence):
                # [FUNCTION DOC] HELPER: for each time step and each layer, concatenates hidden
                #   state and input, applies the layer, adds the update to the hidden state and
                #   feeds it onward; stacks per-step outputs along time.
                B, T, _ = inputs_sequence.shape
                hs_list = initial
                all_out = []
                for t in range(T):
                    input = inputs_sequence[:, t, :]
                    for li, layer in enumerate(self.layers):
                        input_concat = torch.cat((hs_list[li], input), dim=-1)
                        update = layer(input_concat)
                        hs_list[li] = hs_list[li] + update
                        input = hs_list[li]

                    all_out.append(input)

                return torch.stack(all_out, dim=1)

        class RNNScanList(nn.Module):
            # [CLASS DOC] HELPER: same model implemented with scan over a list of hidden-state
            #   tensors as the carry.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the Linear layer stack and records layer count.
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, input_sequence):
                # [FUNCTION DOC] HELPER: runs scan along dim 1 with a step function and returns
                #   the stacked outputs (final carry discarded).
                def step(carry, input):
                    # [FUNCTION DOC] HELPER: updates each layer's hidden state in a copy of
                    #   the carry list and returns cloned hidden states with a clone of the
                    #   last layer output.
                    hs_list = carry[:]
                    for li, layer in enumerate(self.layers):
                        h_prev_li = hs_list[li]
                        input_concat = torch.cat((h_prev_li, input), dim=-1)
                        update = layer(input_concat)
                        h_curr_li = h_prev_li + update
                        hs_list[li] = h_curr_li
                        input = h_curr_li
                    return [t.clone() for t in hs_list], input.clone()

                _, all_outputs_scan = scan(step, initial, input_sequence, dim=1)
                return all_outputs_scan

        class RNNScanTensor(nn.Module):
            # [CLASS DOC] HELPER: same model with the hidden states stacked into one tensor carry.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the Linear layer stack and records layer count.
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(FEATURE_DIM * 2, FEATURE_DIM) for _ in range(LAYERS)]
                )
                self.num_layers = LAYERS

            def forward(self, initial, input_sequence):
                # [FUNCTION DOC] HELPER: stacks initial hidden states along dim 1, scans along dim
                #   1 and returns the stacked outputs.
                def step(carry_tensor, xs_input):
                    # [FUNCTION DOC] HELPER: for each layer updates its slice of a cloned carry
                    #   tensor and returns the cloned carry with a clone of the last layer output.
                    input = xs_input
                    hs_tensor = carry_tensor
                    for li, layer in enumerate(self.layers):
                        current_h_prev_li_slice = hs_tensor[:, li, :]
                        input_concat = torch.cat(
                            (current_h_prev_li_slice, input), dim=-1
                        )
                        update = layer(input_concat)
                        h_curr_li = current_h_prev_li_slice + update
                        hs_tensor = hs_tensor.clone()
                        hs_tensor[:, li, :] = h_curr_li
                        input = h_curr_li
                    return hs_tensor.clone(), input.clone()

                hs_stacked = torch.stack(initial, dim=1)
                _, all_outputs_scan = scan(step, hs_stacked, input_sequence, dim=1)
                return all_outputs_scan

        def run_test_and_get_grads_loss(model, initial_hs, inputs):
            # [FUNCTION DOC] HELPER: zeroes existing parameter grads, runs the model on detached
            #   copies of the initial states and inputs, backpropagates the sum of outputs, and
            #   returns the list of cloned layer weight gradients and the loss.
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            current_initial_hs = [
                h.detach().clone().requires_grad_(h.requires_grad) for h in initial_hs
            ]
            current_inputs = (
                inputs.detach().clone().requires_grad_(inputs.requires_grad)
            )

            out = model(current_initial_hs, current_inputs)
            loss = out.sum()
            loss.backward()

            layer_grads = []
            for layer in model.layers:
                layer_grads.append(layer.weight.grad.clone())

            return layer_grads, loss

        torch.manual_seed(0)

        initial_hs_template = [
            torch.zeros(
                BATCH_SIZE, FEATURE_DIM, requires_grad=True, dtype=torch.float32
            ).to(DEVICE)
            for _ in range(LAYERS)
        ]
        inputs_template = torch.randn(
            BATCH_SIZE, SEQ_LEN, FEATURE_DIM, requires_grad=True, dtype=torch.float32
        ).to(DEVICE)

        # Test 3 models: RNNScanList, RNNScanTensor, RNNLoop
        models = [RNNScanList, RNNScanTensor, RNNLoop]

        for model_class in models:
            # Create uncompiled model
            model_uc = model_class().to(DEVICE)
            uncompiled_grads, uncompiled_loss = run_test_and_get_grads_loss(
                model_uc, initial_hs_template, inputs_template
            )

            # Create compiled model with same weights
            model_to_compile = model_class().to(DEVICE)
            model_to_compile.load_state_dict(model_uc.state_dict())
            compiled_model = torch.compile(model_to_compile)
            compiled_grads, compiled_loss = run_test_and_get_grads_loss(
                compiled_model, initial_hs_template, inputs_template
            )

            # Compare gradients for each layer
            for uncompiled_grad, compiled_grad in zip(uncompiled_grads, compiled_grads):
                self.assertEqual(
                    uncompiled_grad,
                    compiled_grad,
                )

            # Compare losses
            self.assertEqual(
                uncompiled_loss,
                compiled_loss,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_init_carries_unequal_grad(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Combine function computes one carry
        #   under no_grad; both inits (random) may require grad. Compares with fake scan; with
        #   autograd, checks gradients of all result leaves but the first with respect to x and
        #   the second init.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        result = scan_fct(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            # TODO: Ideally we should be able to select the results that require gradients like this
            # [leaf for leaf in pytree.tree_leaves(result) if leaf.requires_grad == True]
            # However, for the scan operator this does not work, as all outputs always have
            # grad_fn=<ScanAutogradOpBackward>
            res_req_grad_flat = pytree.tree_leaves(result)[1:]
            res_exp_req_grad_flat = pytree.tree_leaves(result_exp)[1:]
            self.check_autograd(res_req_grad_flat, res_exp_req_grad_flat, (x, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_init_carries_equal_grad(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Same as the unequal-grad test but
        #   the first init does not require grad; compares with fake scan and checks gradients of
        #   all result leaves except the first.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=False)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        result = scan_fct(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        result_exp = _fake_scan(
            get_scan_combine_fn("fct_c1_no_grad", True),
            (h1, h2),
            x,
            dim=dim,
            reverse=reverse,
        )
        self.assertEqual(result, result_exp)

        if autograd:
            # TODO: Ideally we should be able to select the results that require gradients like this
            # [leaf for leaf in pytree.tree_leaves(result) if leaf.requires_grad == True]
            # However, for the scan operator this does not work, as all outputs always have
            # grad_fn=<ScanAutogradOpBackward>
            res_req_grad_flat = pytree.tree_leaves(result)[1:]
            res_exp_req_grad_flat = pytree.tree_leaves(result_exp)[1:]
            self.check_autograd(res_req_grad_flat, res_exp_req_grad_flat, (x, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_for_out(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. The per-step output is computed
        #   under no_grad while carries are differentiable; compares with fake scan and, with
        #   autograd, checks gradients of the carries wrt x and both inits.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 7, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 7, device=device, requires_grad=autograd)

        def fct_ys_no_grad(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: new carries are the first entry of x plus y and the second
            #   entry of x plus y; the output is tanh of their sum computed under no_grad.
            c1 = x[0] + y
            c2 = x[1] + y
            with torch.no_grad():
                h_new = torch.tanh(x[0] + x[1] + y)
            return (c1, c2), h_new

        result = scan_fct(fct_ys_no_grad, (h1, h2), x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(fct_ys_no_grad, (h1, h2), x, dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[0], result_exp[0], (x, h1, h2))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_additional_inputs_partial(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Combine function uses closure
        #   tensors, with part of the computation (using W_hh and b_hh) under no_grad; compares
        #   with fake scan and checks gradients of the second result wrt h, x, W_ih, b_ih.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry from the closure weights W_ih and b_ih; output is carry
            #   plus one plus a tanh term computed under no_grad.
            c_new = y @ W_ih + b_ih + x

            h_new = c_new + 1
            with torch.no_grad():
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            h_new2 = h_new + h_new_no_grad

            return c_new, h_new2

        result = scan_fct(fct_no_grad_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(fct_no_grad_bhh_Whh, h, x, dim=dim, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x, W_ih, b_ih))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_with_no_grad_additional_inputs_all(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. All closure-parameter terms are
        #   computed under no_grad; compares with fake scan and checks gradients of the second
        #   result wrt h and x only.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bih_Wih_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: computes terms from all four closure parameters under no_grad
            #   and adds them to differentiable carry and output.
            c_new = x + y
            h_new = c_new + x
            with torch.no_grad():
                c_new_no_grad = y @ W_ih + b_ih
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            c_new2 = c_new + c_new_no_grad
            h_new2 = h_new + h_new_no_grad
            return c_new2, h_new2

        result = scan_fct(fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_combine_fn_carries_ys_same_grad(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Carry and output share the
        #   differentiable path plus no_grad closure terms; compares with fake scan and checks
        #   gradients of the second result wrt h and x.
        dim = 1
        scan_fct = compile_mode_helper(scan, compile_mode)
        x = torch.randn(3, 10, 7, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W_ih = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_ih = torch.randn(7, device=device, requires_grad=autograd)
        W_hh = torch.randn(7, 7, device=device, requires_grad=autograd)
        b_hh = torch.randn(7, device=device, requires_grad=autograd)

        def fct_no_grad_bih_Wih_bhh_Whh(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: new carry and output built from a differentiable sum plus
            #   no_grad closure terms.
            c_new = x + y
            h_new = c_new + 1
            with torch.no_grad():
                c_new_no_grad = y @ W_ih + b_ih
                h_new_no_grad = torch.tanh(x @ W_hh + b_hh)
            c_new2 = c_new + c_new_no_grad
            h_new2 = h_new + h_new_no_grad
            return c_new2, h_new2

        result = scan_fct(fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse)
        result_exp = _fake_scan(
            fct_no_grad_bih_Wih_bhh_Whh, h, x, dim=dim, reverse=reverse
        )
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result[1], result_exp[1], (h, x))

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_scan_closure_nested(self, reverse, compile_mode, device, autograd):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Scan with a closure-based cell
        #   (checked against fake scan), then two chained scans of different closure weights
        #   (second scanning the stacked outputs of the first), then a more complex pair reusing
        #   weights across both; checks gradients when autograd is on.
        scan_fct = compile_mode_helper(scan, compile_mode)

        # Simple non-nested case
        x = torch.randn(3, 20, 5, device=device, requires_grad=autograd)
        h = torch.randn(3, 7, device=device, requires_grad=autograd)
        W = torch.randn(5, 7, device=device, requires_grad=autograd)
        b = torch.randn(7, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry y matmul W plus b; output tanh of carry plus x.
            c_new = y @ W + b
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result = scan_fct(f1, h, x, dim=1, reverse=reverse)
        result_exp = _fake_scan(f1, h, x, dim=1, reverse=reverse)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (h, x, W, b))

        # Nested case
        def chain_fct(fct, f_1, f_2, xs, h_1, h_2):
            # [FUNCTION DOC] HELPER: runs the given scan function twice along dim 1 with reverse
            #   from the enclosing test: first f_1 on xs with h_1, then f_2 on the stacked outputs
            #   of the first with h_2; returns the second result.
            o1 = fct(
                f_1,
                h_1,
                xs,
                dim=1,
                reverse=reverse,
            )
            o2 = fct(
                f_2,
                h_2,
                o1[1],
                dim=1,
                reverse=reverse,
            )
            return o2

        x1 = torch.ones(3, 20, 5, device=device, requires_grad=autograd)
        h1 = torch.zeros(3, 7, device=device, requires_grad=autograd)
        h2 = torch.zeros(3, 3, device=device, requires_grad=autograd)
        W_1 = torch.randn(5, 7, device=device, requires_grad=autograd)
        b_1 = torch.randn(7, device=device, requires_grad=autograd)
        W_2 = torch.randn(7, 3, device=device, requires_grad=autograd)
        b_2 = torch.randn(3, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry y matmul W_1 plus b_1; output tanh of carry plus x.
            c_new = y @ W_1 + b_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        def f2(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry y matmul W_2 plus b_2; output tanh of carry plus x.
            c_new = y @ W_2 + b_2
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result1 = chain_fct(scan_fct, f1, f2, x1, h1, h2)
        expected_result = chain_fct(_fake_scan, f1, f2, x1, h1, h2)
        self.assertEqual(result1, expected_result)

        if autograd:
            self.check_autograd(result1, expected_result, (h1, h2, x1, W_1, b_1))

        # Complex case
        x1 = torch.randn(3, 20, 3, device=device, requires_grad=autograd)
        h1 = torch.randn(3, 3, device=device, requires_grad=autograd)
        h2 = torch.randn(3, 3, device=device, requires_grad=autograd)
        W_1 = torch.randn(3, 3, device=device, requires_grad=autograd)
        b_1 = torch.randn(3, device=device, requires_grad=autograd)
        W_2 = torch.randn(3, 3, device=device, requires_grad=autograd)
        b_2 = torch.randn(3, device=device, requires_grad=autograd)

        def f1(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry y matmul W_1 plus b_1; output tanh of carry plus x.
            c_new = y @ W_1 + b_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        def f2(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: carry mixes W_2, b_2 times b_1 and y matmul W_1; output tanh
            #   of carry plus x.
            c_new = y @ W_2 + b_2 * b_1 + y @ W_1
            h_new = torch.tanh(c_new + x)
            return c_new, h_new

        result1 = chain_fct(scan_fct, f1, f2, x1, h1, h2)
        expected_result = chain_fct(_fake_scan, f1, f2, x1, h1, h2)
        self.assertEqual(result1, expected_result)

        if autograd:
            self.check_autograd(
                result1, expected_result, (h1, h2, x1, W_1, b_1, W_2, b_2)
            )

    @skipIfNoDynamoSupport
    def test_scan_simple_graph_wrong_dtype(self):
        # [FUNCTION DOC] Combine function returns a carry of int64 dtype while init is
        #   float; expects UncapturedHigherOrderOpError complaining about differing init
        #   and carry metadata.
        def add_wrong_dtype(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns an int64 ones-like carry and the sum as output.
            return torch.ones_like(x + y, dtype=torch.int64), x + y

        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            # [FUNCTION DOC] HELPER: reversed scan along dim 0.
            return scan(fct, init, xs, dim=0, reverse=True)

        # Wrong dtype
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected init and carry to have same metadata.*",
        ):
            f(add_wrong_dtype, init, x)

    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_scan_simple_graph(self):
        # [FUNCTION DOC] Skipped for cross-ref. Traces a reversed scan with symbolic make_fx and
        #   compares the golden graph (flip, scan op with size arguments, flip of the output),
        #   then compiles with the eager recording backend and compares the dynamo graph code.
        x = torch.randn(3, 10, 2, device=torch.device("cpu"))
        init = torch.randn(1, 10, 2, device=torch.device("cpu"))

        def f(fct, init, xs):
            # [FUNCTION DOC] HELPER: reversed scan along dim 0.
            return scan(fct, init, xs, dim=0, reverse=True)

        # Correct case
        gm = make_fx(f, tracing_mode="symbolic")(
            get_scan_combine_fn("add", False), init, x
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, fct_1, init_1, xs_1):
    flip = torch.ops.aten.flip.default(xs_1, [0])
    sym_size_int = torch.ops.aten.sym_size.int(init_1, 1)
    sym_size_int_1 = torch.ops.aten.sym_size.int(init_1, 2)
    sym_size_int_2 = torch.ops.aten.sym_size.int(xs_1, 1)
    sym_size_int_3 = torch.ops.aten.sym_size.int(xs_1, 2);  xs_1 = None
    scan_combine_graph_0 = self.scan_combine_graph_0
    scan = torch.ops.higher_order.scan(scan_combine_graph_0, [init_1], [flip], (sym_size_int, sym_size_int_1, sym_size_int_2, sym_size_int_3));  scan_combine_graph_0 = init_1 = flip = sym_size_int = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = None
    getitem = scan[0]
    getitem_1 = scan[1];  scan = None
    flip_1 = torch.ops.aten.flip.default(getitem_1, [0]);  getitem_1 = None
    return (getitem, flip_1)""",
        )

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(get_scan_combine_fn("add", False), init, x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    flip = torch.flip(l_xs_, [0]);  l_xs_ = None
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [flip], []);  scan_combine_fn_0 = l_init_ = flip = None
    carry = scan[0]
    out = scan[1];  scan = None
    out_1 = out.flip([0]);  out = None
    return (carry, out_1)""",
        )

    @requires_cuda
    def test_scan_input_mutation(self):
        # [FUNCTION DOC] CUDA-only. Combine function mutates its carry argument in place; scan is
        #   expected to raise UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")

        def fct_input_mutation(x, y):
            # [FUNCTION DOC] HELPER: adds one to x in place then returns two sums.
            x.add_(1)
            return x + y, x + y + 2

        x = torch.randn(3, 2, 2, device=device)
        init = torch.randn(2, 2, device=device)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_mutation, init, x, dim=0)

    @requires_cuda
    def test_scan_additional_input_mutation(self):
        # [FUNCTION DOC] CUDA-only. Combine function mutates a closed-over buffer in place; scan
        #   is expected to raise UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")
        buf = torch.randn(2, 2, device=device)

        def fct_additional_input_mutation(x, y):
            # [FUNCTION DOC] HELPER: adds one to the closure buffer in place and returns two sums.
            buf.add_(1)
            return x + y, x + y + 2

        x = torch.randn(3, 2, 2, device=device)
        init = torch.randn(2, 2, device=device)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_additional_input_mutation, init, x, dim=0)

    @requires_cuda
    def test_scan_input_carry_alias(self):
        # [FUNCTION DOC] CUDA-only. Combine function returns an input leaf directly as a carry;
        #   expects UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            # [FUNCTION DOC] HELPER: returns the first carry entry unchanged as the new carry
            #   (aliasing the input) plus other sums.
            return (x[0], x[1] + y[1]), (x[1] + y[1] + 1, x[1] + y[1] + 2)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_output_alias, init, inp, dim=0)

    @requires_cuda
    def test_scan_input_output_alias(self):
        # [FUNCTION DOC] CUDA-only. Combine function returns a carry input directly as an output;
        #   expects UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            # [FUNCTION DOC] HELPER: returns an input leaf as one of its outputs.
            return (x[0] + 1, x[1] + y[1]), (x[1], x[1] + y[1] + 2)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_input_output_alias, init, inp, dim=0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_carry_carry_alias(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Combine function returns the same tensor for both
        #   carry slots; expects UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")

        def fct_carry_carry_alias(x, y):
            # [FUNCTION DOC] HELPER: returns the same computed tensor twice as the carry
            #   pair, plus outputs.
            c = x[0] + y[1]
            return (c, c), (x[0] + y[1], x[0] + y[1] + 1)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_carry_carry_alias, init, inp, dim=0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_scan_carry_output_alias(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Combine function returns the same tensor as both
        #   a carry and an output; expects UncapturedHigherOrderOpError naming the scan op.
        device = torch.device("cuda")

        def fct_carry_output_alias(x, y):
            # [FUNCTION DOC] HELPER: computes a tensor c and returns it both inside the carry pair
            #   and as the first output.
            c = x[0] + y[1]
            return (x[0] + y[1], c), (c, x[0] + y[1] + 1)

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)
        init = (torch.randn(2, 2, device=device), torch.randn(2, 2, device=device))

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(fct_carry_output_alias, init, inp, dim=0)

    def test_scan_length_validation_pass(self):
        # [FUNCTION DOC] Scan with an explicit length equal to the xs size gives the same result
        #   as the fake scan.
        def add(c, x):
            # [FUNCTION DOC] HELPER: carry plus x as the new carry and a clone of the sum
            #   as the output.
            return c + x, (c + x).clone()

        init = torch.tensor(0.0)
        xs = torch.arange(4, dtype=torch.float32)
        result = scan(add, init, xs, length=4)
        expected = _fake_scan(add, init, xs)
        self.assertEqual(result, expected)

    def test_scan_length_validation_mismatch(self):
        # [FUNCTION DOC] Scan with length 3 on xs of size 4 raises RuntimeError whose message says
        #   the length does not match the xs size.
        def add(c, x):
            # [FUNCTION DOC] HELPER: carry plus x as the new carry and a clone of the
            #   sum as output.
            return c + x, (c + x).clone()

        init = torch.tensor(0.0)
        xs = torch.arange(4, dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, r"length=3.*does not match xs size.*4"
        ):
            scan(add, init, xs, length=3)

    def test_scan_length_negative_raises(self):
        # [FUNCTION DOC] Scan with xs None and length minus one raises RuntimeError saying length
        #   must be a non-negative integer.
        def body(c, x):
            # [FUNCTION DOC] HELPER: increments carry by 1.0 and returns a clone as output.
            return c + 1.0, (c + 1.0).clone()

        init = torch.tensor(0.0)
        with self.assertRaisesRegex(
            RuntimeError, r"length must be a non-negative integer"
        ):
            scan(body, init, None, length=-1)

    def test_scan_xs_empty_tuple_length_eager(self):
        # [FUNCTION DOC] An empty tuple for xs with an explicit length behaves like xs None: the
        #   body receives None for x, four steps run, the carry is 4.0 and the stacked output is
        #   1 to 4.
        # Empty pytree xs (()) is treated the same as xs=None when length is given.
        def body(c, x):
            # [FUNCTION DOC] HELPER: raises RuntimeError if x is not None; otherwise increments
            #   the carry by 1.0 and returns a clone as output.
            if x is not None:
                raise RuntimeError(f"expected x to be None, got {x}")
            return c + 1.0, (c + 1.0).clone()

        init = torch.tensor(0.0)
        carry, ys = scan(body, init, (), length=4)
        self.assertEqual(carry, torch.tensor(4.0))
        self.assertEqual(ys, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    @parametrize("length", [4, 0])
    def test_scan_xs_none_length_body_uses_x_raises(self, length):
        # [FUNCTION DOC] Parametrized over length 4 and 0. A body that adds None x to the carry
        #   makes scan raise RuntimeError or TypeError about an unsupported operand type with
        #   NoneType, whatever the length.
        # combine_fn that dereferences x must raise regardless of length.
        def bad_body(c, x):
            # [FUNCTION DOC] HELPER: adds x (None) to the carry, which fails.
            return c + x, (c + x).clone()

        init = torch.tensor(0.0)
        with self.assertRaisesRegex(
            (RuntimeError, TypeError), r"unsupported operand type.*NoneType"
        ):
            scan(bad_body, init, None, length=length)

    @skipIfTorchDynamo("don't test compile on compile")
    @parametrize("compile_mode", ["none", "eager"])
    @parametrize("output", ["tensor", "none"])
    @parametrize("autograd", [False, True])
    def test_scan_xs_none_length(self, compile_mode, output, autograd):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over compile mode, output kind (tensor
        #   or none) and autograd. Scans with xs None and length 3 and compares with the fake
        #   scan; with a None output and autograd on, asserts that a RuntimeError is raised (a
        #   pinned limitation) and returns early; otherwise checks gradients when autograd is on.
        if output == "tensor":

            def body(c, x):
                # [FUNCTION DOC] HELPER: carry becomes the sine of the carry; output is a
                #   clone of it.
                return torch.sin(c), torch.sin(c).clone()
        else:

            def body(c, x):  # noqa: E306
                # [FUNCTION DOC] HELPER: carry becomes the sine of the carry; output is None.
                return torch.sin(c), None

        init = torch.tensor(0.5, requires_grad=autograd)
        scan_fct = compile_mode_helper(scan, compile_mode)

        if output == "none" and autograd:
            # None ys + autograd is not yet supported; pin the limitation so this
            # starts failing (and prompts an update) once support is added.
            with self.assertRaises(RuntimeError):
                result = scan_fct(body, init, None, length=3)
                torch.autograd.grad(result[0], init, torch.ones_like(result[0]))
            return

        result = scan_fct(body, init, None, length=3)
        result_exp = _fake_scan(body, init, None, length=3)
        self.assertEqual(result, result_exp)

        if autograd:
            self.check_autograd(result, result_exp, (init,))

    @skipIfNoDynamoSupport
    def test_scan_xs_none_length_make_fx_none_output(self):
        # [FUNCTION DOC] Traces scan with xs None, length 3 and a None output through make_fx and
        #   checks the traced graph gives the same result as the fake scan.
        def body(c, x):
            # [FUNCTION DOC] HELPER: carry becomes sine of carry; output None.
            return torch.sin(c), None

        init = torch.tensor(0.5)
        gm = make_fx(lambda i: scan(body, i, None, length=3))(init)
        result = gm(init)
        result_exp = _fake_scan(body, init, None, length=3)
        self.assertEqual(result, result_exp)

    @parametrize("output", ["tensor", "none"])
    def test_scan_xs_none_length_zero(self, output):
        # [FUNCTION DOC] Parametrized over output kind. Scan with xs None and length 0 returns the
        #   init as carry; the tensor-output case gives an output of shape zero by three with the
        #   init dtype, the none case gives None; both match the fake scan.
        if output == "tensor":

            def body(c, x):
                # [FUNCTION DOC] HELPER: returns the carry unchanged and a clone of carry times 2.
                return c, (c * 2).clone()

            init = torch.tensor([1.0, 2.0, 3.0])
            carry, ys = scan(body, init, None, length=0)
            self.assertEqual(carry, init)
            self.assertEqual(ys.shape, torch.Size([0, 3]))
            self.assertEqual(ys.dtype, init.dtype)
        else:

            def body(c, x):  # noqa: E306
                # [FUNCTION DOC] HELPER: carry plus 1.0 and a None output.
                return c + 1.0, None

            init = torch.tensor(0.0)
            carry, ys = scan(body, init, None, length=0)
            self.assertEqual(carry, init)
            self.assertIsNone(ys)
        expected = _fake_scan(body, init, None, length=0)
        self.assertEqual((carry, ys), expected)

    @skipIfNoDynamoSupport
    @parametrize("output", ["tensor", "none"])
    def test_scan_xs_none_length_zero_compile(self, output):
        # [FUNCTION DOC] Parametrized over output kind. Under torch.compile with full
        #   graph, scan with xs None and length 0 raises RuntimeError saying it is not
        #   supported under torch.compile.
        if output == "tensor":

            def body(c, x):
                # [FUNCTION DOC] HELPER: returns the carry unchanged and a clone of carry times 2.
                return c, (c * 2).clone()

            init = torch.tensor([1.0, 2.0, 3.0])
        else:

            def body(c, x):  # noqa: E306
                # [FUNCTION DOC] HELPER: carry plus 1.0 and a None output.
                return c + 1.0, None

            init = torch.tensor(0.0)

        @torch.compile(fullgraph=True)
        def f(i):
            # [FUNCTION DOC] HELPER: compiled; scan with xs None and length 0.
            return scan(body, i, None, length=0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"scan\(\) with length=0 and no xs tensors is not supported under torch\.compile",
        ):
            f(init)

    def test_scan_xs_none_length_vmap(self):
        # [FUNCTION DOC] Applies torch.vmap over a batch of three initial carries to a scan with
        #   xs None and length 4; checks the batched carry is init plus 4 and the stacked output
        #   holds init plus 1 through 4 along dim 1.
        def body(c, x):
            # [FUNCTION DOC] HELPER: carry plus 1.0 and a clone of it as output.
            return c + 1.0, (c + 1.0).clone()

        batch_init = torch.arange(3, dtype=torch.float32)
        batched_carry, batched_ys = torch.vmap(lambda i: scan(body, i, None, length=4))(
            batch_init
        )
        self.assertEqual(batched_carry, batch_init + 4.0)
        expected_ys = torch.stack([batch_init + k for k in range(1, 5)], dim=1)
        self.assertEqual(batched_ys, expected_ys)

    @skipIfTorchDynamo("don't test compile on compile")
    @skipIfNoDynamoSupport
    def test_scan_xs_none_length_dynamic_compile(self):
        # [FUNCTION DOC] Skipped under dynamo. Compiles a scan with xs None and a dynamic length
        #   using a compile counter backend; length 4 and 6 give the expected results and a repeat
        #   of length 4 does not recompile (frame count 1).
        def body(c, x):
            # [FUNCTION DOC] HELPER: carry plus 1.0 and a clone of it as output.
            return c + 1.0, (c + 1.0).clone()

        init = torch.tensor(0.0)
        from torch._dynamo.testing import CompileCounterWithBackend

        cc = CompileCounterWithBackend("eager")

        @torch.compile(backend=cc, fullgraph=True, dynamic=True)
        def f(i, length):
            # [FUNCTION DOC] HELPER: compiled; scan with xs None and the given length.
            return scan(body, i, None, length=length)

        r4 = f(init, 4)
        self.assertEqual(r4, (torch.tensor(4.0), torch.tensor([1.0, 2.0, 3.0, 4.0])))
        r6 = f(init, 6)
        self.assertEqual(
            r6, (torch.tensor(6.0), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        )
        f(init, 4)  # should hit cache, no new compilation
        self.assertEqual(cc.frame_count, 1)


class AssociativeScanModels:
    # [CLASS DOC] HELPER: namespace class with module wrappers used by the associative scan tests:
    #   a factory for scan functions and the modules CombineFn, Simple, ChainFn and NestedFn.
    @staticmethod
    def get_scan_fct(compile_mode, combine_mode):
        # [FUNCTION DOC] HELPER: static factory. For any compile mode except fake, compiles
        #   associative_scan per the mode and returns a function taking combine function,
        #   xs, dim and reverse that fixes the combine mode; for fake returns the reference
        #   fake associative scan.
        # Compile the associative_scan according to the provided compile_mode
        if compile_mode != "fake":
            assoc_scan_comp = compile_mode_helper(associative_scan, compile_mode)

            def scan_fct(combine_fn, xs, dim, reverse):
                # [FUNCTION DOC] HELPER: calls the compiled associative scan with the given
                #   combine function, xs, dim and reverse and the captured combine mode.
                return assoc_scan_comp(combine_fn, xs, dim, reverse, combine_mode)

        else:
            scan_fct = _fake_associative_scan
        return scan_fct

    class CombineFn(torch.nn.Module):
        # [CLASS DOC] HELPER: module that applies one associative scan with a given combine
        #   function, dim, reverse and modes.
        def __init__(self, combine_fn, dim, reverse, combine_mode, compile_mode):
            # [FUNCTION DOC] HELPER: stores the scan function (from get_scan_fct for the compile
            #   and combine modes), the combine function, dim and reverse.
            super().__init__()

            self.scan_fct = AssociativeScanModels.get_scan_fct(
                compile_mode, combine_mode
            )
            self.combine_fn = combine_fn
            self.dim = dim
            self.reverse = reverse

        def forward(self, inputs):
            # [FUNCTION DOC] HELPER: runs the scan function on the inputs and returns the result.
            results = self.scan_fct(self.combine_fn, inputs, self.dim, self.reverse)
            return results

    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: module that runs two associative scans (add and mul) over
        #   the same inputs.
        def __init__(self, dim, reverse, combine_mode, compile_mode):
            # [FUNCTION DOC] HELPER: builds a list of two CombineFn modules (add and mul combine
            #   functions) sharing the given dim, reverse, combine mode and compile mode.
            super().__init__()

            kwargs = {
                "dim": dim,
                "reverse": reverse,
                "combine_mode": combine_mode,
                "compile_mode": compile_mode,
            }
            self.combine_fns = [
                AssociativeScanModels.CombineFn(
                    get_scan_combine_fn("add", True), **kwargs
                ),
                AssociativeScanModels.CombineFn(
                    get_scan_combine_fn("mul", True), **kwargs
                ),
            ]

        def forward(self, inputs):
            # [FUNCTION DOC] HELPER: applies each CombineFn to the inputs and returns the
            #   list of results.
            results = []
            for combine_fn in self.combine_fns:
                results.append(combine_fn(inputs))
            return results

    class ChainFn(torch.nn.Module):
        # [CLASS DOC] HELPER: module that chains several scan stages; per-stage options may be
        #   lists of the chain length, otherwise the single value is shared.
        def __init__(self, combine_fn, dim, reverse, combine_mode, compile_mode):
            # [FUNCTION DOC] HELPER: for each stage builds an options mapping (taking list entries
            #   by position when a value is a list of chain length), creates the scan function for
            #   that stage's combine mode, and stores a tuple of stage function, scan function and
            #   remaining options (dim, reverse).
            super().__init__()

            chain_len = len(combine_fn)
            kwargs = {
                "combine_fn": combine_fn,
                "dim": dim,
                "reverse": reverse,
                "combine_mode": combine_mode,
            }

            # Prepare the kwargs as a list.
            self.nested_tuple = []
            for ind in range(chain_len):
                kwargs_el = {}
                for key, val in kwargs.items():
                    # Check if val is a list and if it has the same length as combine_fn
                    # If so, then use the individual elements.
                    # If not, duplicate the first element.
                    if type(val) is list and len(val) == chain_len:
                        kwargs_el[key] = val[ind]
                    else:
                        kwargs_el[key] = val

                scan_fct = AssociativeScanModels.get_scan_fct(
                    compile_mode, kwargs_el["combine_mode"]
                )
                combine_fn = kwargs_el["combine_fn"]
                del kwargs_el["combine_fn"]
                del kwargs_el["combine_mode"]
                self.nested_tuple.append((combine_fn, scan_fct, kwargs_el))

        def forward(self, inputs):
            # [FUNCTION DOC] HELPER: threads the inputs through each stage, calling the stage
            #   function with the scan function and the stage options.
            results = inputs
            for combine_fn, scan_fct, kwargs in self.nested_tuple:
                results = combine_fn(scan_fct, results, **kwargs)
            return results

    class NestedFn(torch.nn.Module):
        # [CLASS DOC] HELPER: module that runs a scan function with the combine function taken
        #   from the keyword arguments.
        def forward(self, scan_fct, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: takes the combine function out of the keyword
            #   arguments (removing it from the mapping) and calls the scan function with
            #   it and the remaining keywords.
            combine_fn = kwargs["combine_fn"]

            # Remove combine_fn from kwargs
            del kwargs["combine_fn"]

            results = scan_fct(combine_fn, inputs, **kwargs)

            return results


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class AssociativeScanTests(TestCase):
    # [CLASS DOC] Test suite for associative_scan against the reference fake implementation
    #   over compile modes, combine modes (pointwise or generic), directions, devices and
    #   autograd. Skipped on Windows and without dynamo support. Helper methods compare
    #   results and gradients.
    def setUp(self):
        # [FUNCTION DOC] Resets dynamo state then calls the base class setUp.
        torch._dynamo.reset()
        super().setUp()

    def _check_autograd(self, result, result_exp, autograd_param):
        # [FUNCTION DOC] HELPER: keeps only parameters requiring grad and result tensors requiring
        #   grad; raises AssertionError if the number of differentiable results differs between
        #   result and expected; computes gradients of both with ones as output gradients and
        #   asserts equality with atol 6e-05 and rtol 6e-06.
        grad_param = [p for p in autograd_param if p.requires_grad]

        result_flatten, _ = pytree.tree_flatten(result)
        result_exp_flatten, _ = pytree.tree_flatten(result_exp)
        result_flatten = [r for r in result_flatten if r.requires_grad]
        result_exp_flatten = [r for r in result_exp_flatten if r.requires_grad]

        # Check the result and parameter lists
        if len(result_flatten) != len(result_exp_flatten):
            raise AssertionError(
                "The number of elements requiring gradients is different for the results and the expected results"
            )

        grad_exp_init = [torch.ones_like(el) for el in result_exp_flatten]
        expected_grads = torch.autograd.grad(
            result_exp_flatten, grad_param, grad_exp_init
        )
        grad_init = [torch.ones_like(el) for el in result_flatten]
        grads = torch.autograd.grad(result_flatten, grad_param, grad_init)

        self.assertEqual(grads, expected_grads, atol=6e-05, rtol=6e-06)

    def _run_test(self, model, model_fake, inputs, autograd_param=None):
        # [FUNCTION DOC] HELPER: runs model and fake model on the inputs and asserts equal
        #   results; if any given autograd parameter requires grad, compares gradients for the
        #   results that require grad; returns the model result.
        result = model(inputs)
        result_exp = model_fake(inputs)
        self.assertEqual(result, result_exp)

        if autograd_param is not None and any(
            par.requires_grad for par in autograd_param
        ):
            result_flat = pytree.tree_leaves(result)
            result_exp_flat = pytree.tree_leaves(result_exp)
            exp_grad_mask = [bool(r.requires_grad) for r in result_exp_flat]

            self._check_autograd(
                [r for r, m in zip(result_flat, exp_grad_mask) if m],
                [r for r, m in zip(result_exp_flat, exp_grad_mask) if m],
                autograd_param,
            )

        # Return the result of the functions under test for further investigations
        return result

    def _prepare_fake_kwargs(self, original_kwargs):
        # [FUNCTION DOC] HELPER: returns a copy of the keyword mapping with compile_mode
        #   set to fake.
        kwargs_fake = original_kwargs.copy()
        kwargs_fake["compile_mode"] = "fake"
        return kwargs_fake

    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    @parametrize("dim", [0, 1])
    # pointwise only supports CUDA and does not support compile_dynamic_shape
    # (lifted arguments), so skip those combinations as in test_associative_scan_compile.
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_zero_length(
        self, combine_mode, reverse, compile_mode, device, autograd, dim
    ):
        # [FUNCTION DOC] Requires SM70. Parametrized over combine mode, reverse, compile mode,
        #   device, autograd and dim; pointwise on cpu or with dynamic-shape compile is skipped.
        #   Runs the add and mul scan model on an input whose chosen dim has size zero and
        #   compares to the fake model (including gradients).
        shape = [2, 4, 3]
        shape[dim] = 0
        x = torch.randn(*shape, device=device, requires_grad=autograd)
        kwargs = {
            "dim": dim,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.Simple(**kwargs),
            model_fake=AssociativeScanModels.Simple(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_compile(
        self, combine_mode, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). Compares the Simple model with the fake one and, when not reversed,
        #   with cumsum and cumprod; then checks a small float example against known sums for
        #   forward and reversed scans.
        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        results = self._run_test(
            model=AssociativeScanModels.Simple(**kwargs),
            model_fake=AssociativeScanModels.Simple(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

        if not reverse:
            results_torch = []
            for op_pt in [torch.cumsum, torch.cumprod]:
                results_torch.append(op_pt(x, 0))
            self.assertEqual(results, results_torch)

        # Jax Examples
        x = torch.arange(
            0, 4, device=device, dtype=torch.float32, requires_grad=autograd
        )
        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("add", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        result = self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

        if not reverse:
            results_torch = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=torch.float32)
        else:
            results_torch = torch.tensor([6.0, 6.0, 5.0, 3.0], dtype=torch.float32)

        self.assertEqual(result, results_torch)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_dim(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). With a fixed random seed generates four random shapes and scan dims,
        #   resets dynamo each time, compares with the fake model, and when not reversed compares
        #   with cumsum and cumprod.
        # KNOWN ISSUE: the torch reference cumsum/cumprod always use dim 0 instead of the randomly
        #   chosen scan dim, so the comparison is only meaningful when the chosen dim is 0.
        import random

        random.seed(1234)

        num_dims = [random.randint(2, 5) for _ in range(4)]
        for num_dim in num_dims:
            # To avoid triggering automatic dynamic shape
            torch._dynamo.reset()
            shapes = [random.randint(1, 9) for _ in range(num_dim)]
            rnd_scan_dim = random.randint(0, num_dim - 1)
            x = torch.randn(*shapes, device=device, requires_grad=autograd)

            kwargs = {
                "dim": rnd_scan_dim,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            results = self._run_test(
                model=AssociativeScanModels.Simple(**kwargs),
                model_fake=AssociativeScanModels.Simple(**kwargs_fake),
                inputs=x,
                autograd_param=None if not autograd else (x,),
            )

            if not reverse:
                results_torch = []
                for op_pt in [torch.cumsum, torch.cumprod]:
                    results_torch.append(op_pt(x, 0))
                self.assertEqual(results, results_torch)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_dim_shape_failure(self, compile_mode, combine_mode):
        # [FUNCTION DOC] Requires SM70 and CUDA; marked as expected failure. Intended to run
        #   a reversed compiled generic scan on a 9 by 9 input along dim 0 and compare with
        #   the fake model.
        # KNOWN ISSUE: the method declares compile_mode and combine_mode parameters but is not
        #   parametrized, so it fails on a missing-argument error before running; the expected-
        #   failure marker hides this.
        num_dims = [2]
        for num_dim in num_dims:
            shapes = [9 for _ in range(num_dim)]
            rnd_scan_dim = 0
            x = torch.randn(*shapes, device=torch.device("cuda"))

            kwargs = {
                "dim": rnd_scan_dim,
                "reverse": True,
                "compile_mode": "compile",
                "combine_mode": "generic",
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.Simple(**kwargs),
                model_fake=AssociativeScanModels.Simple(**kwargs_fake),
                inputs=x,
            )

    def test_associative_scan_pointwise_cpu_lowering_error(self):
        # [FUNCTION DOC] Compiling a module that calls associative_scan in pointwise mode on a CPU
        #   tensor raises InductorError saying it is not supported on cpu.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module wrapping a pointwise associative scan along dim 0.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: returns associative_scan of the module-level add function
                #   over xs along dim 0 in pointwise mode.
                return associative_scan(combine_fn, xs, dim=0, combine_mode="pointwise")

        from torch._inductor.exc import InductorError

        xs = torch.randn(8, 4)
        with self.assertRaisesRegex(InductorError, "is not supported on cpu"):
            torch.compile(M(), fullgraph=True)(xs)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_pointwise_mixed_device_lowering_error(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Compiling a pointwise associative scan over a
        #   tuple with one CUDA and one CPU tensor raises InductorError saying it is not
        #   supported on cpu.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: element-wise sum of the two tuples, returning a tuple.
            return (x[0] + y[0], x[1] + y[1])

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module wrapping a pointwise associative scan over a
            #   pair of tensors.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: runs associative_scan over the pair (a, b) along dim 0
                #   in pointwise mode.
                return associative_scan(
                    combine_fn, (a, b), dim=0, combine_mode="pointwise"
                )

        from torch._inductor.exc import InductorError

        a = torch.randn(8, 4, device="cuda")
        b = torch.randn(8, 4, device="cpu")
        with self.assertRaisesRegex(InductorError, "is not supported on cpu"):
            torch.compile(M(), fullgraph=True)(a, b)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_tuple(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). Associative scan over a tuple input with the tuple combine function,
        #   compared with the fake model including gradients for both tensors.
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = (x, y)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("tuple_fct", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_expand_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Generic-mode scan whose combine
        #   function expands a reduced tensor back to x's shape; compared with the fake model.
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: multiplies x by the sum of y over the last dim expanded
            #   to x's shape.
            return x * torch.sum(y, -1).expand(x.shape)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_non_contiguous_tensor(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Builds a transposed (non-
        #   contiguous) input, raises AssertionError if it is contiguous, and compares the add
        #   scan with the fake model.
        x = (
            torch.arange(30, device=device, dtype=torch.float32, requires_grad=autograd)
            .view(10, 3)
            .t()
        )
        if x.is_contiguous():
            raise AssertionError("Expected x to not be contiguous")

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("add", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_complex_pytree(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). Scan over a dict/list/tuple pytree input using the complex pointwise
        #   combine function, compared with the fake model including gradients for x, y, z.
        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("complex_pointwise", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (x, y, z),
        )

    @skipIfTorchDynamo("don't test compile on compile")
    @skipIfNoDynamoSupport
    @skipIfCrossRef  # Arg order changes with crossref
    def test_associative_scan_pytree_output(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Compiles a reversed generic
        #   associative scan over a nested pytree of three tensors with the eager recording
        #   backend and compares the golden graph (movedim, flip, vmap-batched combine, interleave
        #   via pad and stack, flip back).
        x = (
            (
                torch.randn(3, 10, 2, device=torch.device("cpu")),
                (torch.randn(3, 10, 2, device=torch.device("cpu")),),
            ),
            torch.randn(3, 10, 2, device=torch.device("cpu")),
        )

        def f(fct, xs):
            # [FUNCTION DOC] HELPER: reversed generic associative scan along dim 0.
            return associative_scan(
                fct, xs, dim=0, reverse=True, combine_mode="generic"
            )

        def combine_fn(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: combine over nested pytree values; returns a nested carry
            #   made of a and a one-tuple holding b, and an output a minus b, where a is the
            #   sum of x's first leaf and y's second and b is the difference of x's inner leaf
            #   and y's second.
            a, b = (x[0][0] + y[1], x[0][1][0] - y[1])
            return (a, (b,)), a - b

        # Check graph
        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(combine_fn, x)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_xs_0_0_: "f32[3, 10, 2]", L_xs_0_1_0_: "f32[3, 10, 2]", L_xs_1_: "f32[3, 10, 2]"):
        l_xs_0_0_ = L_xs_0_0_
        l_xs_0_1_0_ = L_xs_0_1_0_
        l_xs_1_ = L_xs_1_

        elem: "f32[3, 10, 2]" = torch.movedim(l_xs_0_0_, 0, 0);  l_xs_0_0_ = None
        elem_1: "f32[3, 10, 2]" = torch.movedim(l_xs_0_1_0_, 0, 0);  l_xs_0_1_0_ = None
        elem_2: "f32[3, 10, 2]" = torch.movedim(l_xs_1_, 0, 0);  l_xs_1_ = None
        elem_3: "f32[3, 10, 2]" = torch.flip(elem, [0]);  elem = None
        elem_4: "f32[3, 10, 2]" = torch.flip(elem_1, [0]);  elem_1 = None
        elem_5: "f32[3, 10, 2]" = torch.flip(elem_2, [0]);  elem_2 = None
        child: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 0, -1, 2)
        child_1: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 0, -1, 2)
        child_2: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 0, -1, 2)
        child_3: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 1, None, 2)
        child_4: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 1, None, 2)
        child_5: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 1, None, 2)

        lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None

        _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting = None

        _add_batch_dim: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child, 0, 1);  child = None
        _add_batch_dim_1: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_1, 0, 1);  child_1 = None
        _add_batch_dim_2: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_2, 0, 1);  child_2 = _add_batch_dim_2 = None
        _add_batch_dim_3: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_3, 0, 1);  child_3 = _add_batch_dim_3 = None
        _add_batch_dim_4: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_4, 0, 1);  child_4 = _add_batch_dim_4 = None
        _add_batch_dim_5: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_5, 0, 1);  child_5 = None

        a: "f32[10, 2]" = _add_batch_dim + _add_batch_dim_5;  _add_batch_dim = None
        b: "f32[10, 2]" = _add_batch_dim_1 - _add_batch_dim_5;  _add_batch_dim_1 = _add_batch_dim_5 = None

        child_6: "f32[10, 2]" = a - b

        child_7: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(a, 1, 1, 0);  a = None
        child_8: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(b, 1, 1, 0);  b = None
        child_9: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(child_6, 1, 1, 0);  child_6 = None

        _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None

        child_10: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 2, None, 2)
        child_11: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 2, None, 2)
        child_12: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 2, None, 2)

        lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None

        _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting_1 = None

        _add_batch_dim_6: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_7, 0, 1)
        _add_batch_dim_7: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_8, 0, 1)
        _add_batch_dim_8: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_9, 0, 1);  _add_batch_dim_8 = None
        _add_batch_dim_9: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_10, 0, 1);  child_10 = _add_batch_dim_9 = None
        _add_batch_dim_10: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_11, 0, 1);  child_11 = _add_batch_dim_10 = None
        _add_batch_dim_11: "f32[10, 2]" = torch._functorch.predispatch._add_batch_dim(child_12, 0, 1);  child_12 = None

        a_1: "f32[10, 2]" = _add_batch_dim_6 + _add_batch_dim_11;  _add_batch_dim_6 = None
        b_1: "f32[10, 2]" = _add_batch_dim_7 - _add_batch_dim_11;  _add_batch_dim_7 = _add_batch_dim_11 = None

        child_13: "f32[10, 2]" = a_1 - b_1

        child_14: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(a_1, 1, 1, 0);  a_1 = None
        child_15: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(b_1, 1, 1, 0);  b_1 = None
        child_16: "f32[1, 10, 2]" = torch._functorch.predispatch._remove_batch_dim(child_13, 1, 1, 0);  child_13 = None

        _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None

        slice_10: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_3, 0, 0, 1);  elem_3 = None
        cat: "f32[2, 10, 2]" = torch.cat([slice_10, child_14], dim = 0);  slice_10 = child_14 = None
        slice_11: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_4, 0, 0, 1);  elem_4 = None
        cat_1: "f32[2, 10, 2]" = torch.cat([slice_11, child_15], dim = 0);  slice_11 = child_15 = None
        slice_12: "f32[1, 10, 2]" = torch.ops.aten.slice(elem_5, 0, 0, 1);  elem_5 = None
        cat_2: "f32[2, 10, 2]" = torch.cat([slice_12, child_16], dim = 0);  slice_12 = child_16 = None

        b_2: "f32[2, 10, 2]" = torch._C._nn.pad(child_7, [0, 0, 0, 0, 0, 1], 'constant', None);  child_7 = None

        stacked: "f32[2, 2, 10, 2]" = torch.stack([cat, b_2], dim = 1);  cat = b_2 = None
        interleaved: "f32[4, 10, 2]" = torch.flatten(stacked, start_dim = 0, end_dim = 1);  stacked = None
        interleaved_1: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved, 0, 0, 3);  interleaved = None

        b_3: "f32[2, 10, 2]" = torch._C._nn.pad(child_8, [0, 0, 0, 0, 0, 1], 'constant', None);  child_8 = None

        stacked_1: "f32[2, 2, 10, 2]" = torch.stack([cat_1, b_3], dim = 1);  cat_1 = b_3 = None
        interleaved_2: "f32[4, 10, 2]" = torch.flatten(stacked_1, start_dim = 0, end_dim = 1);  stacked_1 = None
        interleaved_3: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved_2, 0, 0, 3);  interleaved_2 = None

        b_4: "f32[2, 10, 2]" = torch._C._nn.pad(child_9, [0, 0, 0, 0, 0, 1], 'constant', None);  child_9 = None

        stacked_2: "f32[2, 2, 10, 2]" = torch.stack([cat_2, b_4], dim = 1);  cat_2 = b_4 = None
        interleaved_4: "f32[4, 10, 2]" = torch.flatten(stacked_2, start_dim = 0, end_dim = 1);  stacked_2 = None
        interleaved_5: "f32[3, 10, 2]" = torch.ops.aten.slice(interleaved_4, 0, 0, 3);  interleaved_4 = None
        flip_3: "f32[3, 10, 2]" = interleaved_1.flip([0]);  interleaved_1 = None
        flip_4: "f32[3, 10, 2]" = interleaved_3.flip([0]);  interleaved_3 = None
        flip_5: "f32[3, 10, 2]" = interleaved_5.flip([0]);  interleaved_5 = None
        movedim_3: "f32[3, 10, 2]" = torch.movedim(flip_3, 0, 0);  flip_3 = None
        movedim_4: "f32[3, 10, 2]" = torch.movedim(flip_4, 0, 0);  flip_4 = None
        movedim_5: "f32[3, 10, 2]" = torch.movedim(flip_5, 0, 0);  flip_5 = None
        return (movedim_3, movedim_4, movedim_5)
""",
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_downstream_scan_matmul(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). Chains an add associative scan along dim 1 with a matrix
        #   multiplication by a ones matrix and compares with the fake chain.
        def first_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: applies the scan function with the add combine function to
            #   the input and returns the result.
            o = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o

        def second_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: ignores the scan function and returns the input matrix-
            #   multiplied by a 2 by 5 ones matrix.
            W = torch.ones(2, 5, device=device)
            return inp @ W

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)
        kwargs = {
            "dim": 1,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_downstream_scan_scan(
        self, combine_mode, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-shape
        #   compile skipped). Chains two add scans along dim 1 and compares with the fake chain.
        def first_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: applies the add scan to the input.
            o1 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o1

        def second_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: applies the add scan to the input.
            o2 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o2

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 1,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse_first", [False, True])
    @parametrize("same_direction", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    # Skipping the autograd=True because
    # associative_scan does currently not support gradients for lifted parameters
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise" and params["autograd"]),
    )
    def test_associative_scan_downstream_scan_scan_different_dim(
        self,
        combine_mode,
        compile_mode,
        reverse_first,
        same_direction,
        device,
        autograd,
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over combine mode, compile mode,
        #   first direction, same_direction and autograd (pointwise on cpu, dynamic-shape compile,
        #   or pointwise with autograd skipped). Chains two add scans over dim 1 then dim 0, the
        #   second direction equal to the first or its opposite, and compares with the fake chain.
        reverse_second = reverse_first if same_direction else not reverse_first

        def first_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: applies the add scan to the input.
            o1 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o1

        def second_chain_fct(scan_fct, inp, **kwargs):
            # [FUNCTION DOC] HELPER: applies the add scan to the input.
            o2 = scan_fct(get_scan_combine_fn("add", True), inp, **kwargs)
            return o2

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": [1, 0],
            "reverse": [reverse_first, reverse_second],
            "compile_mode": compile_mode,
            "combine_fn": [first_chain_fct, second_chain_fct],
            "combine_mode": [combine_mode, combine_mode],
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.ChainFn(**kwargs),
            model_fake=AssociativeScanModels.ChainFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_nested(self):
        # [FUNCTION DOC] Requires SM70 and CUDA; expected failure (comment says vmap inside
        #   associative_scan prevents nested use). Runs a pointwise scan whose combine function
        #   itself calls another scan, compared to a fake nested version.
        combine_mode = "pointwise"
        compile_mode = "eager"
        reverse_first = False
        same_direction = False
        device = torch.device("cuda")

        reverse_second = reverse_first if same_direction else not reverse_first

        def first_nested_fct(x, y):
            # [FUNCTION DOC] HELPER: runs an inner associative scan with the multiply combine
            #   function over y, and returns x plus that.
            y_new = associative_scan(
                second_nested_fct,
                y,
                0,
                reverse=reverse_second,
                combine_mode=combine_mode,
            )
            return x + y_new

        def first_nested_fct_fake(x, y):
            # [FUNCTION DOC] HELPER: same as first_nested_fct using the fake associative scan.
            y_new = _fake_associative_scan(
                second_nested_fct, y, 0, reverse=reverse_second
            )
            return x + y_new

        def second_nested_fct(x, y):
            # [FUNCTION DOC] HELPER: returns x times y.
            return x * y

        inp = torch.randn(3, 10, 2, device=device)

        kwargs = {
            "dim": 0,
            "reverse": reverse_first,
            "compile_mode": compile_mode,
            "combine_fn": first_nested_fct,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        kwargs_fake["combine_fn"] = first_nested_fct_fake
        self._run_test(
            model=AssociativeScanModels.NestedFn(**kwargs),
            model_fake=AssociativeScanModels.NestedFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("loop_type", ["for"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_loop_in_combine_fn(
        self, compile_mode, loop_type, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over compile mode, loop type (only
        #   the Python for loop is enabled), reverse, device and autograd. Generic scan whose
        #   combine function loops; compared with the fake model including gradients.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: accumulates the absolute values of the ten rows of y into a
            #   zero tensor with a loop (a while_loop variant exists for the while loop type), and
            #   returns x times the accumulated value.
            cnt = torch.zeros_like(y[0, :])
            if loop_type == "while":

                def cond_fn(ind, loop_val):
                    # [FUNCTION DOC] HELPER: while-loop predicate, whether the first entry of
                    #   loop_val is below 5.
                    return (loop_val < 5)[0]

                def body_fn(ind, loop_val):
                    # [FUNCTION DOC] HELPER: returns the index plus one and loop_val plus abs
                    #   of the index.
                    return ind + 1, loop_val + torch.abs(ind)

                new_ind, cnt = torch.while_loop(
                    cond_fn=cond_fn,
                    body_fn=body_fn,
                    carried_inputs=(
                        torch.zeros(1, dtype=torch.int32, device=cnt.device),
                        cnt,
                    ),
                )
            else:
                for ind in range(10):
                    cnt += torch.abs(y[ind])
            return x * cnt

        inp = torch.randn(3, 10, 1, device=device, requires_grad=autograd) * 2

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_loop_in_combine_fn_failure(self):
        # [FUNCTION DOC] Requires SM70 and CUDA; expected failure. Generic scan whose combine
        #   function only defines while-loop helpers and returns nothing.
        # KNOWN ISSUE: combine_fn has no return statement, so it returns None, and the loop
        #   helpers are never called; the test passes only because it is expected to fail.
        compile_mode = "none"
        loop_type = "while"
        reverse = False
        device = torch.device("cuda")

        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: creates a zero tensor and defines the while-loop helpers
            #   but returns nothing.
            # KNOWN ISSUE: unfinished body, always returns None.
            _cnt = torch.zeros_like(y[0, :])
            if loop_type == "while":

                def cond_fn(ind, loop_val):
                    # [FUNCTION DOC] HELPER: predicate, whether the first entry of loop_val is
                    #   below 5 (never used).
                    return (loop_val < 5)[0]

                def body_fn(ind, loop_val):
                    # [FUNCTION DOC] HELPER: returns index plus one and loop_val plus abs of the
                    #   index (never used).
                    return ind + 1, loop_val + torch.abs(ind)

        inp = torch.randn(3, 10, 1, device=device) * 2

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["device"] == torch.device("cpu")
            or params["compile_mode"] == "compile_dynamic_shape"
        ),
    )
    def test_associative_scan_cond_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over compile mode, reverse, device
        #   and autograd (cpu and dynamic-shape compile skipped). Generic scan whose combine
        #   function uses cond on the sign of the sum of y; compared with the fake model.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: val is y cloned if sum of y is positive, else one minus y,
            #   chosen by cond; returns x times val.
            val = cond(torch.sum(y) > 0.0, lambda y: y.clone(), lambda y: 1.0 - y, (y,))
            return x * val

        inp = torch.randn(3, 10, 1, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    # TODO: Does not work because of the usage of vmap within associative_scan
    # TODO: Re-enable additional parameters again once this issues has been resolved
    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @unittest.expectedFailure
    def test_associative_scan_map_in_combine_fn(self):
        # [FUNCTION DOC] Requires SM70 and CUDA; expected failure (comment attributes it to vmap
        #   use in associative_scan). Generic scan whose combine function calls control_flow.map.
        compile_mode = "none"
        reverse = False
        device = torch.device("cuda")

        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: maps a body adding the first row of y to each row of y, and
            #   returns x times the mapped result.
            def body(x, y):
                # [FUNCTION DOC] HELPER: returns x plus y.
                return x + y

            y_init = y[0]
            y_new = control_flow.map(body, y, y_init)
            return x * y_new

        inp = torch.randn(3, 10, 1, device=device)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_vmap_in_combine_fn(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized. Generic scan whose combine function
        #   uses torch.vmap; compared with the fake model including gradients.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: squares y through a vmapped function over dim 0 and returns
            #   x plus that.
            def body(x):
                # [FUNCTION DOC] HELPER: returns x squared.
                return x**2

            mapped_body = torch.vmap(body, 0, 0)
            y_new = mapped_body(y)
            return x + y_new

        inp = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": combine_fn,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("reverse", [False, True])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of associative_scan and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    @decorateIf(
        unittest.skip,
        lambda params: (params["device"] == torch.device("cpu")),
    )
    def test_associative_scan_non_pointwise_generic(
        self, reverse, compile_mode, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over reverse, compile mode, device
        #   (cpu skipped) and autograd. Generic-mode scan with the non-pointwise (matrix
        #   multiplication) combine function, compared with the fake model including gradients.
        x = torch.randn(3, 10, 2, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("non_pointwise", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=x,
            autograd_param=None if not autograd else (x,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
            )
        ),
    )
    def test_associative_scan_binary_operator(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise on cpu or dynamic-
        #   shape compile skipped). Scan with the S5 operator over a tuple of a repeated
        #   vector and random projected inputs; compared with the fake model including
        #   gradients of both elements.
        state_dim = 20
        timesteps = 10
        projected_inputs = torch.randn(
            timesteps, state_dim, device=device, requires_grad=autograd
        )
        A = torch.randn(state_dim, device=device, requires_grad=autograd)
        elements = (A.repeat((timesteps, 1)), projected_inputs)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("s5_operator", True),
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=elements,
            autograd_param=None if not autograd else elements,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    def test_associative_scan_different_input_size(self, compile_mode, reverse, device):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over compile mode, reverse and
        #   device. Generic-mode scan along dim 2 over a 5-tuple of tensors with different
        #   trailing shapes using the different-input-size operator; compared with the fake model
        #   (no autograd parameters passed).
        batch = 5
        hidden_dim = 3
        length = 10
        dstate = 7

        deltaA = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        deltaB_u = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        C = torch.randn((batch, dstate, length), requires_grad=True, device=device)
        x = torch.randn(
            (batch, hidden_dim, length, dstate), requires_grad=True, device=device
        )
        y = torch.randn((batch, hidden_dim, length), requires_grad=True, device=device)
        elements = (x, deltaA, deltaB_u, C, y)

        kwargs = {
            "dim": 2,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": get_scan_combine_fn("different_input_size_operator", True),
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=elements,
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_different_input_size_wrong_dim(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. In pointwise mode with scan dim 3, one input
        #   tensor lacks that many dimensions; expects ValueError saying all xs leaves must have
        #   at least dim plus 1 dimensions.
        batch = 5
        hidden_dim = 3
        length = 10
        dstate = 7

        deltaA = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        deltaB_u = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        C = torch.randn((batch, dstate, length), device=torch.device("cuda"))
        x = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        y = torch.randn(
            (batch, hidden_dim, length, dstate), device=torch.device("cuda")
        )
        elements = (x, deltaA, deltaB_u, C, y)

        with self.assertRaisesRegex(
            ValueError,
            "All xs leaves must have at least 'dim \\+ 1' dimensions",
        ):
            associative_scan(
                get_scan_combine_fn("different_input_size_operator", True),
                elements,
                3,
                combine_mode="pointwise",
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_simple(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise mode skipped because
        #   lowering does not support lifted arguments). Runs scans whose combine functions use
        #   closed-over tensors (one tensor used twice, or two different tensors); compared with
        #   the fake model including gradients wrt input and closure tensors.
        H = torch.rand(2, device=device, requires_grad=autograd)

        def fct_freevars1(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: x times closure H plus y times 2.
            return x * H + y * 2

        def fct_freevars2(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: x times closure H plus y times closure H.
            return x * H + y * H

        H1 = torch.rand(1, device=device, requires_grad=autograd)
        H2 = torch.rand(1, device=device, requires_grad=autograd)

        def fct_freevars3(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: x times closure H1 plus y times closure H2.
            return x * H1 + y * H2

        inp = torch.randn(3, 2, 2, device=device, requires_grad=autograd)

        for fct, param in [
            (fct_freevars1, (H,)),
            (fct_freevars2, (H,)),
            (fct_freevars3, (H1, H2)),
        ]:
            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": fct,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=None if not autograd else (inp, *param),
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_nested(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped). Combine
        #   functions that use closures and an inner helper function, either with the closed-over
        #   tensor outside the combine function or a constant created inside it; compared with
        #   matching fake combine functions including gradients.
        H1 = torch.rand(4, 5, device=device, requires_grad=autograd)
        H2 = torch.rand(4, 1, device=device, requires_grad=autograd)

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: applies an inner function multiplying y by closure H2, then
            #   returns x plus that times closure H1.
            def inner(xi):
                # [FUNCTION DOC] HELPER: returns xi times H2.
                return xi * H2

            ret = inner(y)
            return x + ret * H1

        def fct_nested_outside_fake(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: same computation as fct_nested_outside, used with
            #   the fake scan.
            def inner(xi):
                # [FUNCTION DOC] HELPER: returns xi times H2.
                return xi * H2

            ret = inner(y)
            return x + ret * H1

        # TODO: Using random tensors in the `combine_fn` triggers the vmap randomness error:
        # RuntimeError: vmap: called random operation while in randomness error mode.
        # Please either use the 'same' or 'different' randomness flags on vmap or perform the randomness operation out of vmap
        def fct_nested_inside(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: creates a constant tensor of 42s inside the function (random
            #   tensors would trigger the vmap randomness error), applies an inner function
            #   multiplying y by it, and returns x plus that times closure H1.
            H2_i = torch.ones(4, 1, device=device) * 42

            def inner(xi):
                # [FUNCTION DOC] HELPER: returns xi times the constant H2_i.
                return xi * H2_i

            ret = inner(y)
            return x + ret * H1

        def fct_nested_inside_fake(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: same as fct_nested_inside, for the fake scan.
            H2_i = torch.ones(4, 1, device=device) * 42

            def inner(xi):
                # [FUNCTION DOC] HELPER: returns xi times the constant H2_i.
                return xi * H2_i

            ret = inner(y)
            return x + ret * H1

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        for fct, fct_fake, param in [
            (fct_nested_outside, fct_nested_outside_fake, (H1, H2)),
            (fct_nested_inside, fct_nested_inside_fake, ()),
        ]:
            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": fct,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            kwargs_fake["combine_fn"] = fct_fake
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=None if not autograd else (inp, *param),
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_fct(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped). Combine
        #   function that calls a helper function defined outside it; compared with the fake
        #   model including gradients.
        def additional_fct_no_add_inp(x, y):
            # [FUNCTION DOC] HELPER: returns x times y.
            return x * y

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: returns x plus the helper applied to y and y.
            ret = additional_fct_no_add_inp(y, y)
            return x + ret

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_nested_outside,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    def test_associative_scan_freevars_fct_generic(
        self, compile_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized over compile mode, reverse, device
        #   and autograd. Combine function that itself calls a generic-mode associative_scan along
        #   dim 1; compared with a fake version using the fake scan.
        def additional_fct_no_add_inp(x, y):
            # [FUNCTION DOC] HELPER: returns x times y.
            return x * y

        def fct_nested_outside(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: runs a generic associative scan of the multiply function over
            #   y along dim 1 and returns x plus the result.
            ret = associative_scan(
                additional_fct_no_add_inp, y, 1, combine_mode="generic"
            )
            return x + ret

        def fct_nested_outside_fake(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: same using the fake associative scan.
            ret = _fake_associative_scan(additional_fct_no_add_inp, y, 1)
            return x + ret

        inp = torch.randn(3, 4, 5, device=device, requires_grad=autograd)

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_nested_outside,
            "combine_mode": "generic",
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        kwargs_fake["combine_fn"] = fct_nested_outside_fake
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_shape_check(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped). Combine
        #   function multiplies a carry by a closed-over identity matrix (requires grad) and adds
        #   the next element; scan along dim 2; compared with the fake model.
        H = torch.eye(2, device=device, requires_grad=True)

        def fct_freevars(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: x matrix-multiplied by closure H, plus y.
            return x @ H + y

        inp = torch.randn(2, 2, 3, device=device, requires_grad=True)

        kwargs = {
            "dim": 2,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_freevars,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (inp,),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("autograd", [False, True])
    # Skipping the combine_mode=pointwise
    # as the current implementation of associative_scan lowering
    # does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (params["combine_mode"] == "pointwise"),
    )
    def test_associative_scan_freevars_pytree(
        self, compile_mode, combine_mode, reverse, device, autograd
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped). Combine
        #   function over dict/list/tuple pytrees that adds closed-over pytree leaves; compared
        #   with the fake model with gradients wrt all input leaves.
        xf = torch.randn(2, 2, device=device, requires_grad=autograd)
        yf = torch.randn(2, 2, device=device, requires_grad=autograd)
        zf = torch.randn(2, 2, device=device, requires_grad=autograd)
        inpf = {"i": xf, "j": ([yf], [{"o": zf}])}

        def fct_pointwise(x, y):
            # [FUNCTION DOC] HELPER: pointwise combine over the pytree (products of the first two
            #   leaves, sum of the third) with the matching closed-over leaf added to each result.
            return {
                "i": (x["i"] * y["i"]) + inpf["i"],
                "j": (
                    [(x["j"][0][0] * y["j"][0][0]) + inpf["j"][0][0]],
                    [
                        {
                            "o": (x["j"][1][0]["o"] + y["j"][1][0]["o"])
                            + inpf["j"][1][0]["o"]
                        }
                    ],
                ),
            }

        x = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        y = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        z = torch.randn(3, 2, 2, device=device, requires_grad=autograd)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": fct_pointwise,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=None if not autograd else (*pytree.tree_leaves(inp),),
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
                or torch.version.hip
            )
        ),
    )
    def test_associative_scan_partial_grad(
        self, combine_mode, compile_mode, reverse, device
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped on cpu, dynamic-
        #   shape compile and HIP). Scans over a tuple of six tensors with eight requires-grad
        #   masks (all, none, one, five unseeded random masks); compared with the fake model
        #   including gradients of the tensors that require grad.
        import random

        n_params = 6
        autograds = []
        autograds.append([True, True, True, True, True, True])
        autograds.append([False, False, False, False, False, False])
        autograds.append([False, True, False, False, False, False])
        for _ in range(5):
            autograds.append([bool(random.randint(0, 1)) for _ in range(n_params)])

        def mul2(x, y):
            # [FUNCTION DOC] HELPER: returns a tuple of element-wise products of the two tuples.
            return (*[xv * yv for xv, yv in zip(x, y)],)

        for a_grads in autograds:
            inp = tuple(
                [
                    torch.randn(10, 3, 2, device=device, requires_grad=a_grads[n])
                    for n in range(n_params)
                ]
            )

            kwargs = {
                "dim": 0,
                "reverse": reverse,
                "compile_mode": compile_mode,
                "combine_fn": mul2,
                "combine_mode": combine_mode,
            }
            kwargs_fake = self._prepare_fake_kwargs(kwargs)
            self._run_test(
                model=AssociativeScanModels.CombineFn(**kwargs),
                model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
                inputs=inp,
                autograd_param=inp,
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("compile_mode", ["none", "eager", "compile", "compile_dynamic_shape"])
    @parametrize("reverse", [False, True])
    @parametrize("device", [torch.device("cpu"), torch.device("cuda")])
    # Skipping the combination of combine_mode=pointwise and device=cpu
    # as the current implementation of pointwise does only support CUDA device
    # Skipping the combination of combine_mode=pointwise and compile_mode=compile_dynamic_shape
    # as the current implementation does not support lifted arguments
    @decorateIf(
        unittest.skip,
        lambda params: (
            params["combine_mode"] == "pointwise"
            and (
                params["device"] == torch.device("cpu")
                or params["compile_mode"] == "compile_dynamic_shape"
                or torch.version.hip
            )
        ),
    )
    def test_associative_scan_partial_grad_no_grad(
        self, combine_mode, compile_mode, reverse, device
    ):
        # [FUNCTION DOC] Requires SM70 and CUDA; parametrized (pointwise skipped on cpu, dynamic-
        #   shape compile and HIP). Combine function computes the second output under no_grad;
        #   compared with the fake model with gradients only for the first input.
        def mul_single_nograd(x, y):
            # [FUNCTION DOC] HELPER: returns the product of first entries and the product of
            #   second entries computed under no_grad.
            xy1 = x[0] * y[0]
            with torch.no_grad():
                xy2 = x[1] * y[1]
            return xy1, xy2

        inp = tuple(
            [torch.randn(10, 3, 2, device=device, requires_grad=True) for n in range(2)]
        )

        kwargs = {
            "dim": 0,
            "reverse": reverse,
            "compile_mode": compile_mode,
            "combine_fn": mul_single_nograd,
            "combine_mode": combine_mode,
        }
        kwargs_fake = self._prepare_fake_kwargs(kwargs)
        self._run_test(
            model=AssociativeScanModels.CombineFn(**kwargs),
            model_fake=AssociativeScanModels.CombineFn(**kwargs_fake),
            inputs=inp,
            autograd_param=inp[0:1],
        )

    @unittest.skipIf(not SM70OrLater, "triton")
    def test_associative_scan_sparse_tensor(self):
        # [FUNCTION DOC] Requires SM70. A sparse tensor as xs raises ValueError saying xs leaves
        #   must be dense tensors.
        x = torch.tensor(
            [[[0.0, 0], [1.0, 2.0]], [[0.0, 0], [3.0, 4.0]], [[0.0, 0], [5.0, 6.0]]]
        ).to_sparse()

        with self.assertRaisesRegex(
            ValueError,
            "xs leaves must dense Tensors.*",
        ):
            associative_scan(
                get_scan_combine_fn("add", True), x, 0, combine_mode="generic"
            )

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_combine_fn_wrong_meta_in_combine_fn(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Combine functions that change dtype,
        #   device or memory format of the result each make associative_scan raise
        #   UncapturedHigherOrderOpError complaining initial_xs and combine_fn_output
        #   have different metadata.
        device = torch.device("cuda")
        B, N, C, H, W = 3, 3, 2, 3, 3
        x = torch.randn(B, N, C, H, W, device=device)

        def fct_wrong_dtype(x, y):
            # [FUNCTION DOC] HELPER: returns the sum cast to int64.
            return (x + y).to(torch.int64)

        def fct_wrong_device(x, y):
            # [FUNCTION DOC] HELPER: returns the sum moved to the other device type (cpu when the
            #   test device is cuda).
            return (x + y).to(
                torch.device("cpu") if device.type == "cuda" else torch.device("cuda")
            )

        def fct_wrong_stride(x, y):
            # [FUNCTION DOC] HELPER: returns the sum converted to channels_last memory format.
            return (x + y).to(memory_format=torch.channels_last)

        for fct in [fct_wrong_dtype, fct_wrong_device, fct_wrong_stride]:
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                "Expected initial_xs and combine_fn_output to have same metadata.*",
            ):
                associative_scan(fct, x, 0)

    @unittest.skipIf(not SM70OrLater, "triton")
    def test_associative_scan_wrong_pytree(self):
        # [FUNCTION DOC] Requires SM70. A combine function that expects different pytree arguments
        #   makes generic associative_scan raise AssertionError saying combine_fn received the
        #   wrong number of arguments.
        def fct_wrong_pytree(x, y):
            # [FUNCTION DOC] HELPER: returns a dict with an extra "k" key built from the nested
            #   inputs, mismatching the input structure.
            return {
                "i": x["i"] * y["j"][0][0],
                "k": torch.tensor(0.0),
                "j": ([x["j"][1][0]["o"]], [{"o": torch.sin(x["i"])}]),
            }

        x = torch.randn(3, 2, 2)
        y = torch.randn(3, 2, 2)
        z = torch.randn(3, 2, 2)
        inp = {"i": x, "j": ([y], [{"o": z}])}

        with self.assertRaisesRegex(
            AssertionError,
            "Combine_fn received wrong number of arguments.*",
        ):
            associative_scan(fct_wrong_pytree, inp, 0, combine_mode="generic")

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_non_pointwise(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Pointwise combine mode with a non-pointwise
        #   combine function raises RuntimeError saying combine_fn needs to be pointwise.
        device = torch.device("cuda")
        x = torch.randn(3, 10, 2, device=device)
        with self.assertRaisesRegex(
            # Should be:
            RuntimeError,
            r"For combine_mode='pointwise', the combine_fn needs to be pointwise",
        ):
            associative_scan(
                get_scan_combine_fn("non_pointwise", True),
                x,
                0,
                combine_mode="pointwise",
            )

    @requires_cuda
    def test_associative_scan_input_mutation(self):
        # [FUNCTION DOC] CUDA-only. Combine function mutates its first argument in place; expects
        #   UncapturedHigherOrderOpError naming the associative_scan higher-order op.
        device = torch.device("cuda")

        def fct_input_mutation(x, y):
            # [FUNCTION DOC] HELPER: adds one to x in place and returns x plus y.
            x.add_(1)
            return x + y

        x = torch.randn(3, 2, 2, device=device)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_input_mutation, x, 0)

    @requires_cuda
    def test_associative_scan_input_output_alias(self):
        # [FUNCTION DOC] CUDA-only. Combine function returns an input leaf directly; expects
        #   UncapturedHigherOrderOpError naming the associative_scan op.
        device = torch.device("cuda")

        def fct_input_output_alias(x, y):
            # [FUNCTION DOC] HELPER: returns the first input leaf unchanged and the sum of
            #   the second leaves.
            return x[0], x[1] + y[1]

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_input_output_alias, inp, 0)

    @unittest.skipIf(not SM70OrLater, "triton")
    @requires_cuda
    def test_associative_scan_output_output_alias(self):
        # [FUNCTION DOC] Requires SM70 and CUDA. Combine function returns the same tensor for two
        #   outputs; expects UncapturedHigherOrderOpError naming the associative_scan op.
        device = torch.device("cuda")

        def fct_output_output_alias(x, y):
            # [FUNCTION DOC] HELPER: computes a tensor and returns it twice.
            c = x[0] + y[1]
            return c, c

        x = torch.randn(3, 2, 2, device=device)
        y = torch.randn(3, 2, 2, device=device)
        inp = (x, y)

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.associative_scan",
        ):
            associative_scan(fct_output_output_alias, inp, 0)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
class TestControlFlowTraced(TestCase):
    # [CLASS DOC] Tests that control flow higher-order ops can be traced (make_fx in several
    #   modes), compiled, exported and run under CUDA graphs. Skipped on Windows and without
    #   dynamo support. Provides tracing, compile and export check helpers.
    def setUp(self):
        # [FUNCTION DOC] Resets dynamo state then calls the base class setUp.
        torch._dynamo.reset()
        super().setUp()

    def _check_tracing(self, fn, args, allow_non_fake_inputs=False):
        # [FUNCTION DOC] HELPER: computes the eager result, then for each of symbolic, real and
        #   fake tracing modes traces with make_fx, asserts the traced graph returns the eager
        #   result for the same args, and returns a mapping from mode to traced graph.
        graphs = {}
        eager_res = fn(*args)
        for tracing_mode in ["symbolic", "real", "fake"]:
            graph = make_fx(
                fn,
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=allow_non_fake_inputs,
            )(*args)
            graphs[tracing_mode] = graph
            self.assertEqual(graph(*args), eager_res)
        return graphs

    def _check_compile(self, fn, args, *, dynamic=False, backend="eager"):
        # [FUNCTION DOC] HELPER: compiles fn with the given backend and dynamic flag and asserts
        #   it matches the eager result.
        eager_res = fn(*args)
        compiled_fn = torch.compile(fn, backend=backend, dynamic=dynamic)
        self.assertEqual(compiled_fn(*args), eager_res)

    def _check_export(self, fn, args, *, strict=False, dynamic_shapes=None):
        # [FUNCTION DOC] HELPER: exports fn with the new experimental tracer enabled (given
        #   strictness and dynamic shapes), asserts the exported module output matches eager, and
        #   returns the exported program.
        eg_out = fn(*args)
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ep = torch.export.export(
                fn, args, strict=strict, dynamic_shapes=dynamic_shapes
            )
        ep_out = ep.module()(*args)
        self.assertEqual(eg_out, ep_out)
        return ep

    def test_cond_traced_not_nested(self):
        # [FUNCTION DOC] Traces cond with make_fx (the predicate is a runtime tensor input) and
        #   checks the graph selects sine or cosine for true or false predicates and gives
        #   different results; also checks the symbolic-mode trace matches eager.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches and operand list holding x.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))

        graph = make_fx(f, tracing_mode="symbolic")(x, torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True)), f(x, torch.tensor(True)))

    def test_switch_traced_mismatched_output_structure(self):
        # [FUNCTION DOC] Tracing switch_op with branches returning a tuple versus a dict raises
        #   RuntimeError saying the output spec from the branches is unmatched.
        from torch._higher_order_ops.switch import switch_op

        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: returns a tuple of sine and cosine.
            return (inp_x.sin(), inp_x.cos())

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: returns a dict with sine and cosine.
            return {"a": inp_x.sin(), "b": inp_x.cos()}

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: calls switch_op over the two branches.
            return switch_op(idx, (branch0, branch1), (inp_x,))

        x = torch.randn(4)
        with self.assertRaisesRegex(
            RuntimeError, "Unmatched output spec from torch.switch branches"
        ):
            make_fx(f)(torch.tensor([0]), x)

    def test_switch_traced_not_nested(self):
        # [FUNCTION DOC] Traces switch with three branches using make_fx and checks the traced
        #   graph picks the right branch for indices 0, 1, 2 and clamped indices -1 and 99;
        #   repeats under symbolic tracing.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        x = torch.randn(4)
        graph = make_fx(f)(torch.tensor([0]), x)
        # The traced graph must dispatch to each branch when fed the
        # corresponding index at runtime, and each case must match eager.
        self.assertEqual(graph(torch.tensor([0]), x), branch0(x))
        self.assertEqual(graph(torch.tensor([1]), x), branch1(x))
        self.assertEqual(graph(torch.tensor([2]), x), branch2(x))
        # Clamped indices must match the first/last branch.
        self.assertEqual(graph(torch.tensor([-1]), x), branch0(x))
        self.assertEqual(graph(torch.tensor([99]), x), branch2(x))

        # And the same under symbolic tracing.
        graph = make_fx(f, tracing_mode="symbolic")(torch.tensor([0]), x)
        for i, fn in enumerate((branch0, branch1, branch2)):
            self.assertEqual(graph(torch.tensor([i]), x), fn(x))

    def test_switch_tracing_all_modes(self):
        # [FUNCTION DOC] Uses the tracing helper to trace switch under symbolic, real and fake
        #   modes and compare with eager.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        # Exercise the helper, which round-trips through symbolic / real /
        # fake and asserts graph(*args) == eager for each.
        self._check_tracing(f, (torch.tensor([1]), torch.randn(4)))

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_simple_capture_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Compiles switch with three lambda
        #   branches using the eager recording backend, asserts one graph, and compares golden
        #   code of the outer graph (clamp of the index, switch op with three branch submodules)
        #   and each branch submodule.
        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: defines three lambda branches (sine, cosine, abs) and
            #   switches among them.
            branches = (
                lambda inp_x: inp_x.sin(),
                lambda inp_x: inp_x.cos(),
                lambda inp_x: inp_x.abs(),
            )
            return switch(idx, branches, (inp_x,))

        x = torch.randn(4)

        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(torch.tensor([1]), x)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]

        # Pin down the outer graph: each branch is installed as a submodule
        # named switch_branch{i}_0 and the HOP is called with the full list.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_inp_x_ : torch.Tensor, L_idx_ : torch.Tensor):
    l_inp_x_ = L_inp_x_
    l_idx_ = L_idx_
    index = l_idx_.clamp(0, 2);  l_idx_ = None
    switch_branch0_0 = self.switch_branch0_0
    switch_branch1_0 = self.switch_branch1_0
    switch_branch2_0 = self.switch_branch2_0
    switch = torch.ops.higher_order.switch(index, [switch_branch0_0, switch_branch1_0, switch_branch2_0], (l_inp_x_,));  index = switch_branch0_0 = switch_branch1_0 = switch_branch2_0 = l_inp_x_ = None
    getitem = switch[0];  switch = None
    return (getitem,)""",
        )
        # Each branch is a single op applied to the lifted input.
        self.assertExpectedInline(
            gm.switch_branch0_0.code.strip(),
            """\
def forward(self, l_inp_x_):
    l_inp_x__1 = l_inp_x_
    sin = l_inp_x__1.sin();  l_inp_x__1 = None
    return (sin,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch1_0.code.strip(),
            """\
def forward(self, l_inp_x_):
    l_inp_x__1 = l_inp_x_
    cos = l_inp_x__1.cos();  l_inp_x__1 = None
    return (cos,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch2_0.code.strip(),
            """\
def forward(self, l_inp_x_):
    l_inp_x__1 = l_inp_x_
    abs_1 = l_inp_x__1.abs();  l_inp_x__1 = None
    return (abs_1,)""",
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_lifted_args_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Branches close over different
        #   tensors; checks via golden code that the outer graph lifts all closure tensors as
        #   operands and every branch submodule has the same union signature.
        # When different branches close over different tensors, dynamo must
        # union the free variables across branches so every branch graph
        # shares the same placeholder signature and the HOP receives the
        # full union as operands. This pins down _merge_graph_inputs.
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1
        c = torch.ones(2, 3) * 2

        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor a.
            return inp_x + a

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor b.
            return inp_x + b

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor c.
            return inp_x + c

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(torch.tensor([1]), torch.randn(2, 3))
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]

        # The outer graph lifts all three closure cells and passes the full
        # union as switch operands (cells first, then the explicit operand x).
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_inp_x_ : torch.Tensor, L_idx_ : torch.Tensor, L_branch0_closure_0_cell_contents : torch.Tensor, L_branch1_closure_0_cell_contents : torch.Tensor, L_branch2_closure_0_cell_contents : torch.Tensor):
    l_inp_x_ = L_inp_x_
    l_idx_ = L_idx_
    l_branch0_closure_0_cell_contents = L_branch0_closure_0_cell_contents
    l_branch1_closure_0_cell_contents = L_branch1_closure_0_cell_contents
    l_branch2_closure_0_cell_contents = L_branch2_closure_0_cell_contents
    index = l_idx_.clamp(0, 2);  l_idx_ = None
    switch_branch0_0 = self.switch_branch0_0
    switch_branch1_0 = self.switch_branch1_0
    switch_branch2_0 = self.switch_branch2_0
    switch = torch.ops.higher_order.switch(index, [switch_branch0_0, switch_branch1_0, switch_branch2_0], (l_inp_x_, l_branch0_closure_0_cell_contents, l_branch1_closure_0_cell_contents, l_branch2_closure_0_cell_contents));  index = switch_branch0_0 = switch_branch1_0 = switch_branch2_0 = l_inp_x_ = l_branch0_closure_0_cell_contents = l_branch1_closure_0_cell_contents = l_branch2_closure_0_cell_contents = None
    getitem = switch[0];  switch = None
    return (getitem,)""",
        )
        # Every branch submodule exposes the same placeholder signature —
        # the union of all closures plus the operand — even though each
        # branch only reads one of the closure cells.
        self.assertExpectedInline(
            gm.switch_branch0_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch1_closure_0_cell_contents_branch1, l_branch2_closure_0_cell_contents_branch2):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch0_closure_0_cell_contents_branch0;  l_inp_x__1 = l_branch0_closure_0_cell_contents_branch0 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch1_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch1_closure_0_cell_contents_branch1, l_branch2_closure_0_cell_contents_branch2):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch1_closure_0_cell_contents_branch1;  l_inp_x__1 = l_branch1_closure_0_cell_contents_branch1 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch2_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch1_closure_0_cell_contents_branch1, l_branch2_closure_0_cell_contents_branch2):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch2_closure_0_cell_contents_branch2;  l_inp_x__1 = l_branch2_closure_0_cell_contents_branch2 = None
    return (add,)""",
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_lifted_args_dedup_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Branches share some closure tensors;
        #   checks via golden code that each closure tensor appears once in the operand tuple and
        #   every branch submodule has the same deduplicated signature (suffixes show which branch
        #   first lifted a tensor).
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1
        c = torch.ones(2, 3) * 2

        # branch0 closes over {a, b}, branch1 over {a, b, c}, branch2 over {c}.
        # No leaf is in *all* three -> shared block is empty; `a` is lifted by
        # branches 0+1, `b` by branches 0+1, `c` by branches 1+2.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensors a and b.
            return inp_x + a + b

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: input plus a, b and c.
            return inp_x + a + b + c

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: input plus closure tensor c.
            return inp_x + c

        def f(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches.
            return switch(idx, (branch0, branch1, branch2), (inp_x,))

        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend, fullgraph=True)(
            torch.tensor([1]), torch.randn(2, 3)
        )

        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        # The outer operand tuple lists `a`, `b`, `c` exactly once each, with
        # no duplicates from the per-branch unique lists.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_inp_x_ : torch.Tensor, L_idx_ : torch.Tensor, L_branch0_closure_0_cell_contents : torch.Tensor, L_branch0_closure_1_cell_contents : torch.Tensor, L_branch1_closure_2_cell_contents : torch.Tensor):
    l_inp_x_ = L_inp_x_
    l_idx_ = L_idx_
    l_branch0_closure_0_cell_contents = L_branch0_closure_0_cell_contents
    l_branch0_closure_1_cell_contents = L_branch0_closure_1_cell_contents
    l_branch1_closure_2_cell_contents = L_branch1_closure_2_cell_contents
    index = l_idx_.clamp(0, 2);  l_idx_ = None
    switch_branch0_0 = self.switch_branch0_0
    switch_branch1_0 = self.switch_branch1_0
    switch_branch2_0 = self.switch_branch2_0
    switch = torch.ops.higher_order.switch(index, [switch_branch0_0, switch_branch1_0, switch_branch2_0], (l_inp_x_, l_branch0_closure_0_cell_contents, l_branch0_closure_1_cell_contents, l_branch1_closure_2_cell_contents));  index = switch_branch0_0 = switch_branch1_0 = switch_branch2_0 = l_inp_x_ = l_branch0_closure_0_cell_contents = l_branch0_closure_1_cell_contents = l_branch1_closure_2_cell_contents = None
    getitem = switch[0];  switch = None
    return (getitem,)""",
        )
        # Every branch submodule shares the same placeholder signature,
        # listing each of {a, b, c} once. The "_branch0" / "_branch2" suffixes
        # reflect which branch first lifted the leaf.
        self.assertExpectedInline(
            gm.switch_branch0_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch0_closure_1_cell_contents_branch0, l_branch1_closure_2_cell_contents_branch1):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch0_closure_0_cell_contents_branch0;  l_inp_x__1 = l_branch0_closure_0_cell_contents_branch0 = None
    add_1 = add + l_branch0_closure_1_cell_contents_branch0;  add = l_branch0_closure_1_cell_contents_branch0 = None
    return (add_1,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch1_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch0_closure_1_cell_contents_branch0, l_branch1_closure_2_cell_contents_branch1):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch0_closure_0_cell_contents_branch0;  l_inp_x__1 = l_branch0_closure_0_cell_contents_branch0 = None
    add_1 = add + l_branch0_closure_1_cell_contents_branch0;  add = l_branch0_closure_1_cell_contents_branch0 = None
    add_2 = add_1 + l_branch1_closure_2_cell_contents_branch1;  add_1 = l_branch1_closure_2_cell_contents_branch1 = None
    return (add_2,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch2_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_branch0_closure_0_cell_contents_branch0, l_branch0_closure_1_cell_contents_branch0, l_branch1_closure_2_cell_contents_branch1):
    l_inp_x__1 = l_inp_x_
    add = l_inp_x__1 + l_branch1_closure_2_cell_contents_branch1;  l_inp_x__1 = l_branch1_closure_2_cell_contents_branch1 = None
    return (add,)""",
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_constant_index_specialization_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. With a Python constant index 1,
        #   the captured graph contains only the cosine op (no switch higher-order op) and has
        #   no child submodules.
        # When the index is a Python constant, dynamo specializes into the
        # selected branch and the switch HOP is not emitted at all. The
        # captured graph contains only the ops of branches[idx].
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        branches = (branch0, branch1, branch2)

        backend = EagerAndRecordGraphs()
        torch.compile(
            lambda inp_x: switch(1, branches, (inp_x,)),
            backend=backend,
            fullgraph=True,
        )(torch.randn(4))

        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        # No switch HOP in the graph — only branch1's cos op survives.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_inp_x_ : torch.Tensor):
    l_inp_x_ = L_inp_x_
    cos = l_inp_x_.cos();  l_inp_x_ = None
    return (cos,)""",
        )
        # And no branch submodules were installed because the HOP never ran.
        self.assertEqual(list(dict(gm.named_children()).keys()), [])

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_index_type_specialization_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Shows how the index type
        #   affects capture: a Python int specializes (no switch op, no submodules), while a
        #   symbolic int from a dynamic shape and a tensor each emit the switch op with
        #   three branch submodules.
        # Pin down dynamo's specialization behavior by index type.
        # Python int -> specialize: HOP is not emitted, only branches[idx]'s ops survive.
        # SymInt or Tensor -> no specialize: HOP is emitted with all branches as submodules.
        def branch0(inp_x):
            # [FUNCTION DOC] HELPER: sine.
            return inp_x.sin()

        def branch1(inp_x):
            # [FUNCTION DOC] HELPER: cosine.
            return inp_x.cos()

        def branch2(inp_x):
            # [FUNCTION DOC] HELPER: abs.
            return inp_x.abs()

        branches = (branch0, branch1, branch2)

        def fn(idx, inp_x):
            # [FUNCTION DOC] HELPER: switch over the three branches using the given index.
            return switch(idx, branches, (inp_x,))

        x = torch.randn(4)

        # 1. Python int: dynamo specializes into branch1; HOP is not emitted.
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(1, x)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        self.assertNotIn("higher_order.switch", gm.code)
        self.assertEqual(list(dict(gm.named_children()).keys()), [])

        # 2. SymInt: derive the index from a dynamically-shaped tensor's
        # `.shape` so it is genuinely symbolic at trace time. No specialization;
        # HOP captures all branches.
        def fn_symint(idx_carrier, inp_x):
            # [FUNCTION DOC] HELPER: derives the index as the leading size of idx_carrier minus
            #   one and switches over the branches.
            idx = idx_carrier.shape[0] - 1
            return switch(idx, branches, (inp_x,))

        idx_carrier = torch.zeros(2)  # shape[0]-1 == 1 -> branch1 at runtime
        torch._dynamo.mark_dynamic(idx_carrier, 0)

        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        torch.compile(fn_symint, backend=backend, fullgraph=True)(idx_carrier, x)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        self.assertIn("higher_order.switch", gm.code)
        self.assertEqual(
            sorted(dict(gm.named_children()).keys()),
            ["switch_branch0_0", "switch_branch1_0", "switch_branch2_0"],
        )

        # 3. Tensor: no specialization; HOP captures all branches.
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(torch.tensor([1]), x)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        self.assertIn("higher_order.switch", gm.code)
        self.assertEqual(
            sorted(dict(gm.named_children()).keys()),
            ["switch_branch0_0", "switch_branch1_0", "switch_branch2_0"],
        )

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_switch_nn_module_lifted_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Switch over two Linear modules
        #   compiled with the eager recording backend; golden code shows weights and biases
        #   of both modules lifted as operands, and each branch submodule only reading its
        #   own module's parameters.
        # nn.Module branches: every parameter of every branch's module is
        # lifted into the HOP operand tuple, and each branch's submodule
        # signature accepts the full union. Reads of foreign parameters
        # are present in the signature but unused inside the branch body.
        linear0 = torch.nn.Linear(4, 3)
        linear1 = torch.nn.Linear(4, 3)

        backend = EagerAndRecordGraphs()
        torch.compile(
            lambda idx, inp_x: switch(idx, (linear0, linear1), (inp_x,)),
            backend=backend,
            fullgraph=True,
        )(torch.tensor([1]), torch.randn(2, 4))

        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]

        # Both modules' weights AND biases are lifted into the HOP operand
        # tuple, in sorted order of their placeholder names.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_inp_x_ : torch.Tensor, L_idx_ : torch.Tensor, L_linear0_parameters_weight_ : torch.nn.parameter.Parameter, L_linear0_parameters_bias_ : torch.nn.parameter.Parameter, L_linear1_parameters_weight_ : torch.nn.parameter.Parameter, L_linear1_parameters_bias_ : torch.nn.parameter.Parameter):
    l_inp_x_ = L_inp_x_
    l_idx_ = L_idx_
    l_linear0_parameters_weight_ = L_linear0_parameters_weight_
    l_linear0_parameters_bias_ = L_linear0_parameters_bias_
    l_linear1_parameters_weight_ = L_linear1_parameters_weight_
    l_linear1_parameters_bias_ = L_linear1_parameters_bias_
    index = l_idx_.clamp(0, 1);  l_idx_ = None
    switch_branch0_0 = self.switch_branch0_0
    switch_branch1_0 = self.switch_branch1_0
    switch = torch.ops.higher_order.switch(index, [switch_branch0_0, switch_branch1_0], (l_inp_x_, l_linear0_parameters_bias_, l_linear0_parameters_weight_, l_linear1_parameters_bias_, l_linear1_parameters_weight_));  index = switch_branch0_0 = switch_branch1_0 = l_inp_x_ = l_linear0_parameters_bias_ = l_linear0_parameters_weight_ = l_linear1_parameters_bias_ = l_linear1_parameters_weight_ = None
    getitem = switch[0];  switch = None
    return (getitem,)""",
        )
        # Each branch submodule receives the full union but only reads its
        # own module's parameters.
        self.assertExpectedInline(
            gm.switch_branch0_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_linear0_parameters_bias__branch0, l_linear0_parameters_weight__branch0, l_linear1_parameters_bias__branch1, l_linear1_parameters_weight__branch1):
    l_inp_x__1 = l_inp_x_
    linear = torch._C._nn.linear(l_inp_x__1, l_linear0_parameters_weight__branch0, l_linear0_parameters_bias__branch0);  l_inp_x__1 = l_linear0_parameters_weight__branch0 = l_linear0_parameters_bias__branch0 = None
    return (linear,)""",
        )
        self.assertExpectedInline(
            gm.switch_branch1_0.code.strip(),
            """\
def forward(self, l_inp_x_, l_linear0_parameters_bias__branch0, l_linear0_parameters_weight__branch0, l_linear1_parameters_bias__branch1, l_linear1_parameters_weight__branch1):
    l_inp_x__1 = l_inp_x_
    linear = torch._C._nn.linear(l_inp_x__1, l_linear1_parameters_weight__branch1, l_linear1_parameters_bias__branch1);  l_inp_x__1 = l_linear1_parameters_weight__branch1 = l_linear1_parameters_bias__branch1 = None
    return (linear,)""",
        )

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with crossref
    def test_cond_simple_with_linear_compile_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref; with trace_autograd_ops
        #   enabled. Compiles a function doing cond and autograd.grad and checks a single
        #   graph is captured, containing the cond op with true and false submodules and the
        #   autograd grad call.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        x = torch.randn(4, requires_grad=True)

        def f(pred, x):
            # [FUNCTION DOC] HELPER: runs cond then returns the gradient of the result wrt x with
            #   ones as output gradient.
            result = cond(pred, true_fn, false_fn, (x,))
            grad_out = torch.ones_like(result)
            return torch.autograd.grad(result, (x,), grad_out)

        backend = EagerAndRecordGraphs()
        torch.compile(f, backend=backend)(torch.tensor(False), x)
        # With autograd.grad tracing support, the entire function is traced
        # into a single graph instead of breaking into forward + backward graphs
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, L_pred_ : torch.Tensor, L_x_ : torch.Tensor):
    l_pred_ = L_pred_
    l_x_ = L_x_
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, (l_x_,));  l_pred_ = cond_true_0 = cond_false_0 = None
    result = cond[0];  cond = None
    grad_out = torch.ones_like(result)
    grad = torch.autograd.grad(result, (l_x_,), grad_out);  result = l_x_ = grad_out = None
    getitem_1 = grad[0];  grad = None
    return (getitem_1,)""",
        )

    def test_while_loop_op_mismatch_in_meta(self):
        # [FUNCTION DOC] Tracing (fake mode) a module whose while_loop body changes tensor shapes
        #   (via nonzero) raises UncapturedHigherOrderOpError saying carried inputs and body
        #   output have different metadata.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module calling the while_loop higher-order op directly with a
            #   body that returns nonzero results.
            def forward(self, c, a, b):
                # [FUNCTION DOC] HELPER: runs the while_loop higher-order op with condition on the
                #   counter being positive and no additional inputs.
                def cond_fn(c, a, b):
                    # [FUNCTION DOC] HELPER: true while c is greater than zero.
                    return c > 0

                def body_fn(c, a, b):
                    # [FUNCTION DOC] HELPER: decrements c and returns the nonzero index tensors of
                    #   a and b, whose shapes differ from the carries.
                    return c - 1, a.nonzero(), b.nonzero()

                return torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    (c, a, b),
                    tuple(),
                )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected carried_inputs and body_output to have same metadata but found",
        ):
            make_fx(Mod(), tracing_mode="fake")(
                torch.tensor(
                    0,
                ),
                torch.randn(2, 3),
                torch.randn(2, 3),
            )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_not_nested_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Runs cond with true
        #   and false CUDA predicates through the cudagraphs backend helper and the many-backends
        #   capture/replay helper and compares with eager.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y selecting sine or cosine of x.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4).cuda()
        true_pred = torch.tensor(True).cuda()
        false_pred = torch.tensor(False).cuda()

        _check_compile_cudagraph_backend(self, f, [x, true_pred])
        _check_compile_cudagraph_backend(self, f, [x, false_pred])
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, true_pred],
        )
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, false_pred],
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_pin_memory_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Checks pinned host
        #   memory allocated during stream capture inside conditional nodes goes to the private
        #   pool, by running branches that use split_with_sizes_copy (an op using pinned memory
        #   internally) through the cudagraph helpers for true and false predicates.
        # Make sure that pinned host memory allocations get assigned
        # to a private pool correctly during stream capture, even
        # inside of conditional nodes.

        # Ideally, we would call torch.Tensor.pin_memory() directly,
        # but that is not allowed on fake tensors, so instead we call
        # an op whose C++ implementation used pinned memory
        # internally.
        sizes = [3, 4, 3]

        def true_fn(x):
            # [FUNCTION DOC] HELPER: splits x into pieces of sizes 3, 4, 3 with copy.
            return torch.split_with_sizes_copy(x, sizes)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: splits twice x into pieces of the same sizes with copy.
            return torch.split_with_sizes_copy(2 * x, sizes)

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y between the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(10).cuda()
        true_pred = torch.tensor(True).cuda()
        false_pred = torch.tensor(False).cuda()

        _check_compile_cudagraph_backend(self, f, [x, true_pred])
        _check_compile_cudagraph_backend(self, f, [x, false_pred])
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, true_pred],
        )
        _check_compile_many_backends_with_cudagraph(
            self,
            f,
            [x, false_pred],
        )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_triply_nested_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Builds three
        #   levels of nested cond functions and runs two predicate combinations through
        #   the cudagraph helpers.
        def level3_true(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def level3_false(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def level2_true(x, p2):
            # [FUNCTION DOC] HELPER: cond on p2 between the level 3 functions.
            return cond(p2, level3_true, level3_false, [x])

        def level2_false(x, p2):
            # [FUNCTION DOC] HELPER: cond on p2 between x plus 1 and x minus 1 lambdas.
            return cond(p2, lambda x: x + 1, lambda x: x - 1, [x])

        def level1_true(x, p1, p2):
            # [FUNCTION DOC] HELPER: cond on p1 between level2_true and level2_false.
            return cond(p1, level2_true, level2_false, [x, p2])

        def level1_false(x, p1, p2):
            # [FUNCTION DOC] HELPER: cond on p1 between level2_false and level2_true.
            return cond(p1, level2_false, level2_true, [x, p2])

        def f(x, p0, p1, p2):
            # [FUNCTION DOC] HELPER: cond on p0 between the level 1 functions.
            return cond(p0, level1_true, level1_false, [x, p1, p2])

        x = torch.randn(4).cuda()

        test_inputs = [
            [
                x,
                torch.tensor(True).cuda(),
                torch.tensor(True).cuda(),
                torch.tensor(False).cuda(),
            ],
            [
                x,
                torch.tensor(False).cuda(),
                torch.tensor(False).cuda(),
                torch.tensor(True).cuda(),
            ],
        ]

        for args in test_inputs:
            _check_compile_cudagraph_backend(self, f, args)
            _check_compile_many_backends_with_cudagraph(self, f, args)

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_while_loop_traced_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Runs a while_loop
        #   (adding 2 per step until the iteration reaches a tensor limit of 3) through the
        #   cudagraph helpers and compares with eager.
        def f(x, limit):
            # [FUNCTION DOC] HELPER: runs while_loop on a counter starting at zero and x, looping
            #   while the counter is below the limit.
            init_iter = torch.zeros((), dtype=torch.int64, device=x.device)

            def cond_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: true while iteration is less than the closed-over limit.
                return iteration < limit

            def body_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: returns acc plus 2 and iteration plus 1.
                return acc + 2, iteration + 1

            return while_loop(cond_fn, body_fn, (x, init_iter))

        x = torch.randn(4).cuda()
        limit = torch.tensor(3, device="cuda")

        _check_compile_cudagraph_backend(self, f, [x, limit])
        _check_compile_many_backends_with_cudagraph(self, f, [x, limit])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_while_loop_cuda_graph_replay_uses_runtime_condition(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Warms up and
        #   captures a CUDA graph of the while_loop higher-order op (limit passed as an additional
        #   input), then replays it with limits 3, 0 and 5 after resetting the inputs, checking
        #   the output tensor equals one plus twice the limit, the counter output equals the
        #   limit, and the captured input buffers are left unchanged. Shows the loop condition is
        #   evaluated at replay time.
        def cond_fn(acc, iteration, limit):
            # [FUNCTION DOC] HELPER: true while iteration is less than limit.
            return iteration < limit

        def body_fn(acc, iteration, limit):
            # [FUNCTION DOC] HELPER: returns acc plus 2 and iteration plus 1.
            return acc + 2, iteration + 1

        def f(x, iteration, limit):
            # [FUNCTION DOC] HELPER: invokes the while_loop higher-order op on (x, iteration) with
            #   limit as additional input.
            return torch.ops.higher_order.while_loop(
                cond_fn, body_fn, (x, iteration), (limit,)
            )

        x = torch.ones(4, device="cuda")
        iteration = torch.zeros((), dtype=torch.int64, device="cuda")
        limit = torch.tensor(3, device="cuda")

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream), ControlFlowOpWarmupDispatchMode():
            f(x, iteration, limit)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph, stream=side_stream),
            CUDAGraphCaptureControlFlowOpDispatchMode(),
        ):
            out = f(x, iteration, limit)

        torch.cuda.current_stream().wait_stream(side_stream)

        for num_iters in [3, 0, 5]:
            x.fill_(1)
            iteration.zero_()
            limit.fill_(num_iters)
            graph.replay()
            torch.cuda.synchronize()

            self.assertEqual(out[0], torch.full_like(x, 1 + 2 * num_iters))
            self.assertEqual(
                out[1], torch.tensor(num_iters, dtype=torch.int64, device="cuda")
            )
            self.assertEqual(x, torch.ones_like(x))
            self.assertEqual(iteration, torch.zeros_like(iteration))

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_while_loop_cudagraph_kwargs_error(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. For both the capture
        #   dispatch mode and the warmup dispatch mode, calling their dispatch hook for the
        #   while_loop op with an unexpected keyword argument raises RuntimeError saying kwargs
        #   are not supported (each mode checked in its own subtest).
        def cond_fn(iteration):
            # [FUNCTION DOC] HELPER: true while iteration is below 1.
            return iteration < 1

        def body_fn(iteration):
            # [FUNCTION DOC] HELPER: returns a one-tuple with iteration plus one.
            return (iteration + 1,)

        iteration = torch.zeros((), dtype=torch.int64, device="cuda")
        args = (cond_fn, body_fn, (iteration,), ())
        error = "CUDA graph conditional torch.while_loop does not support kwargs"

        for mode_cls in (
            CUDAGraphCaptureControlFlowOpDispatchMode,
            ControlFlowOpWarmupDispatchMode,
        ):
            with (
                self.subTest(mode=mode_cls.__name__),
                self.assertRaisesRegex(RuntimeError, error),
            ):
                mode_cls().__torch_dispatch__(
                    torch.ops.higher_order.while_loop,
                    (),
                    args,
                    {"unexpected": True},
                )

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_while_loop_cudagraph_body_pytree_mismatch_error(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. During CUDA graph
        #   capture with the capture dispatch mode, a while_loop whose body returns a dict instead
        #   of the carry structure raises RuntimeError saying body_fn should return the same
        #   pytree structure as carried_inputs.
        def cond_fn(iteration):
            # [FUNCTION DOC] HELPER: true while iteration is below 1.
            return iteration < 1

        def body_fn(iteration):
            # [FUNCTION DOC] HELPER: returns a dict holding iteration plus one (wrong structure).
            return {"iteration": iteration + 1}

        iteration = torch.zeros((), dtype=torch.int64, device="cuda")
        graph = torch.cuda.CUDAGraph()

        with self.assertRaisesRegex(
            RuntimeError,
            "body_fn should return the same pytree structure as carried_inputs",
        ):
            with (
                torch.cuda.graph(graph),
                CUDAGraphCaptureControlFlowOpDispatchMode(),
            ):
                torch.ops.higher_order.while_loop(cond_fn, body_fn, (iteration,), ())

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_inside_while_loop_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. A while_loop whose
        #   body uses cond on iteration parity (add 3 on even, subtract 1 on odd) is run through
        #   the cudagraph helpers.
        def f(x, limit):
            # [FUNCTION DOC] HELPER: runs while_loop with a zero counter until the
            #   counter reaches limit.
            init_iter = torch.zeros((), dtype=torch.int64, device=x.device)

            def cond_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: true while iteration is less than the closed-over limit.
                return iteration < limit

            def body_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: applies cond on whether iteration is even to add 3 or
                #   subtract 1 from acc, and increments iteration.
                pred = iteration % 2 == 0
                acc = torch.cond(
                    pred,
                    lambda acc: acc + 3,
                    lambda acc: acc - 1,
                    [acc],
                )
                return acc, iteration + 1

            return torch.while_loop(cond_fn, body_fn, (x, init_iter))

        x = torch.randn(4).cuda()
        limit = torch.tensor(4, device="cuda")

        _check_compile_cudagraph_backend(self, f, [x, limit])
        _check_compile_many_backends_with_cudagraph(self, f, [x, limit])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_while_loop_inside_cond_cudagraphs(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. A cond whose
        #   branches are while_loops (add 2 per step versus subtract 1 per step, for limit
        #   iterations) is run through the cudagraph helpers for true and false predicates.
        def add_loop(x, limit):
            # [FUNCTION DOC] HELPER: while_loop adding 2 to x for limit iterations; returns
            #   the accumulated tensor.
            init_iter = torch.zeros((), dtype=torch.int64, device=x.device)

            def cond_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: true while iteration is less than limit.
                return iteration < limit

            def body_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: returns acc plus 2 and iteration plus 1.
                return acc + 2, iteration + 1

            acc, _ = torch.while_loop(cond_fn, body_fn, (x, init_iter))
            return acc

        def sub_loop(x, limit):
            # [FUNCTION DOC] HELPER: while_loop subtracting 1 from x for limit iterations;
            #   returns the tensor.
            init_iter = torch.zeros((), dtype=torch.int64, device=x.device)

            def cond_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: true while iteration is less than limit.
                return iteration < limit

            def body_fn(acc, iteration):
                # [FUNCTION DOC] HELPER: returns acc minus 1 and iteration plus 1.
                return acc - 1, iteration + 1

            acc, _ = torch.while_loop(cond_fn, body_fn, (x, init_iter))
            return acc

        def f(x, pred, limit):
            # [FUNCTION DOC] HELPER: cond on pred between add_loop and sub_loop with operands
            #   x and limit.
            return torch.cond(pred, add_loop, sub_loop, [x, limit])

        x = torch.randn(4).cuda()
        limit = torch.tensor(3, device="cuda")

        for pred in [torch.tensor(True).cuda(), torch.tensor(False).cuda()]:
            _check_compile_cudagraph_backend(self, f, [x, pred, limit])
            _check_compile_many_backends_with_cudagraph(self, f, [x, pred, limit])

    @unittest.skipIf(
        not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
        "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
    )
    def test_cond_traced_record_stream_reuse(self):
        # [FUNCTION DOC] Skipped without CUDA graph conditional-node support. Enables the
        #   allocator setting graph_capture_record_stream_reuse, then expects capturing a cond
        #   (zero-argument branches that allocate tensors) under the capture dispatch mode to
        #   raise RuntimeError mentioning that setting; the finally clause resets the allocator
        #   setting to false.
        torch.cuda.memory._set_allocator_settings(
            "graph_capture_record_stream_reuse:True"
        )
        try:
            predicate = torch.tensor(True, device="cuda")

            def true_fn():
                # [FUNCTION DOC] HELPER: returns two new zero tensors of size 8 on CUDA.
                return torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda")

            def false_fn():
                # [FUNCTION DOC] HELPER: returns two new zero tensors of size 8 on CUDA.
                return torch.zeros(8, device="cuda"), torch.zeros(8, device="cuda")

            g = torch.cuda.CUDAGraph()
            with self.assertRaisesRegex(
                RuntimeError,
                "graph_capture_record_stream_reuse:True",
            ):
                with torch.cuda.graph(g), CUDAGraphCaptureControlFlowOpDispatchMode():
                    torch.cond(predicate, true_fn, false_fn, [])
        finally:
            torch.cuda.memory._set_allocator_settings(
                "graph_capture_record_stream_reuse:False"
            )

    def test_while_loop_nested_traced(self):
        # [FUNCTION DOC] Traces the nested while_loop test program and compares symbolic-mode
        #   golden graphs: the outer graph with the while_loop op, the condition subgraph (sum
        #   less than 2), and the body subgraph (inner while_loop followed by adding one).
        fn, inp = WHILE_LOOP_TESTS["nested"]
        graphs = self._check_tracing(fn, inp)
        self.assertExpectedInline(
            graphs["symbolic"].code.strip("\n"),
            """\
def forward(self, out_iter_1, it_1, y_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (out_iter_1, it_1, y_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = out_iter_1 = it_1 = y_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2];  while_loop = None
    return (getitem, getitem_1, getitem_2)
    """,
        )
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 2);  sum_1 = None
    return lt
    """,
        )
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2];  while_loop = None
    add = torch.ops.aten.add.Tensor(getitem, 1);  getitem = None
    return (add, getitem_1, getitem_2)
    """,
        )

    def test_while_loop_pytree_carry(self):
        # [FUNCTION DOC] Compiles the pytree-carry while_loop program with the eager recording
        #   backend and checks the result matches eager; unless run under dynamo, also checks one
        #   graph was captured and compares golden code of the graph (while_loop op with
        #   flattened pytree carries).
        fn, inp = WHILE_LOOP_TESTS["simple_with_pytree_carry"]
        backend = EagerAndRecordGraphs()
        expected_res = fn(*inp)
        compiled_res = torch.compile(fn, backend=backend)(*inp)
        self.assertEqual(expected_res, compiled_res)
        # When test with torch dynamo, the graph is not captured because
        # it's traced together with the code before torch.compile
        if not TEST_WITH_TORCHDYNAMO:
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_it_ : torch.Tensor, L_pytree_input_0_0_ : torch.Tensor, L_pytree_input_1_x_ : torch.Tensor, L_pytree_input_1_y_ : torch.Tensor):
    l_it_ = L_it_
    l_pytree_input_0_0_ = L_pytree_input_0_0_
    l_pytree_input_1_x_ = L_pytree_input_1_x_
    l_pytree_input_1_y_ = L_pytree_input_1_y_
    cond_fn_0 = self.cond_fn_0
    body_fn_0 = self.body_fn_0
    while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_it_, l_pytree_input_0_0_, l_pytree_input_1_x_, l_pytree_input_1_y_), ());  cond_fn_0 = body_fn_0 = l_it_ = l_pytree_input_0_0_ = l_pytree_input_1_x_ = l_pytree_input_1_y_ = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    value = while_loop[2]
    value_1 = while_loop[3];  while_loop = None
    return (getitem, getitem_1, value, value_1)""",
            )

    def _wrap_with_functionalize(self, fn, func_type):
        # [FUNCTION DOC] HELPER: wraps fn according to func_type: cpp uses the C++ functionalize
        #   API, python uses the Python API and also returns a FunctionalTensorMode, functorch
        #   uses torch.func.functionalize, no leaves fn as is, and any other value raises
        #   AssertionError. Returns the wrapped function and the mode (None when not needed).
        mode = None
        if func_type == "cpp":
            fn = CppFunctionalizeAPI().functionalize(fn)
        elif func_type == "python":
            fn = PythonFunctionalizeAPI().functionalize(fn)
            mode = FunctionalTensorMode()
        elif func_type == "functorch":
            fn = torch.func.functionalize(fn)
        else:
            if func_type != "no":
                raise AssertionError(
                    f"Expected func_type to be 'no', got {func_type!r}"
                )
        return fn, mode

    @parametrize("func_type", ["no", "cpp", "python", "functorch"])
    def test_while_loop_simple_functionalize_check_graph(self, func_type):
        # [FUNCTION DOC] Parametrized over functionalization type (no, cpp, python, functorch).
        #   Traces the simple-with-mutation program in all tracing modes under the chosen
        #   functionalization (entering the mode if given), and compares golden graphs of the
        #   symbolic trace; with no functionalization the cond and body keep in-place adds,
        #   otherwise they are functionalized to out-of-place adds.
        fn, inp = WHILE_LOOP_TESTS["simple_with_mutation"]
        fn, mode = self._wrap_with_functionalize(fn, func_type)
        mode = mode if mode is not None else contextlib.nullcontext()
        with mode:
            graphs = self._check_tracing(fn, inp)
        if func_type == "no":
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, x_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add_ = torch.ops.aten.add_.Tensor(clone, 1);  clone = None
    add__1 = torch.ops.aten.add_.Tensor(add_, -1);  add_ = None
    sum_1 = torch.ops.aten.sum.default(add__1);  add__1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add_ = torch.ops.aten.add_.Tensor(clone, 1);  clone = None
    add__1 = torch.ops.aten.add_.Tensor(add_, -1);  add_ = None
    add = torch.ops.aten.add.Tensor(add__1, 1);  add__1 = None
    return (add,)
    """,
            )
        elif func_type == "python":
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, arg0_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    sum_1 = torch.ops.aten.sum.default(add_1);  add_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 1);  add_1 = None
    return (add_2,)
    """,
            )
        else:
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """\
def forward(self, x_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    getitem = while_loop[0];  while_loop = None
    return (getitem,)
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    sum_1 = torch.ops.aten.sum.default(add_1);  add_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 10);  sum_1 = None
    return lt
    """,
            )
            self.assertExpectedInline(
                graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
                """\
def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(add, -1);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 1);  add_1 = None
    return (add_2,)
    """,
            )

    @parametrize("func_type", ["no", "cpp", "python", "functorch"])
    # - "simple_with_linear" and "nested_with_linear" doesn't work because parameters and buffers
    #   are not inputs so they're not wrapped by functionalization and tracing.
    #
    # - make_fx tracing mode "real" fails for "int_carry", "pytree_int_carry" and "const_and_symint_output"
    #   because tensors are real but we unspecialize the ints with unbacked symints causing
    #   data dependent errors.
    #   Since this is not the common use path, we skip them for now.
    @parametrize(
        "while_loop_test",
        set(WHILE_LOOP_TESTS.keys())
        - {
            "simple_with_linear",
            "nested_with_linear",
            "int_carry",
            "pytree_int_carry",
            "const_and_symint_output",
        },
    )
    def test_while_loop_functionalize(self, func_type, while_loop_test):
        # [FUNCTION DOC] Parametrized over functionalization type and over the while_loop test
        #   programs excluding those with module parameters or unbacked int carries. Traces each
        #   program under the chosen functionalization with the tracing helper (comparing with
        #   eager in all modes).
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        fn, mode = self._wrap_with_functionalize(fn, func_type)
        mode = mode if mode is not None else contextlib.nullcontext()
        with mode:
            self._check_tracing(fn, inp)

    # - make_fx tracing mode "real" fails for "int_carry", "pytree_int_carry" and "const_and_symint_output"
    #   because tensors are real but we unspecialize the ints with unbacked symints causing
    #   data dependent errors.
    #   Since this is not the common use path, we skip them for now.
    @parametrize(
        "while_loop_test",
        set(WHILE_LOOP_TESTS.keys())
        - {"int_carry", "pytree_int_carry", "const_and_symint_output"},
    )
    def test_while_loop_tracing(self, while_loop_test):
        # [FUNCTION DOC] Parametrized over the while_loop test programs except the int-carry ones.
        #   Traces each program with make_fx in all modes, allowing non-fake inputs only for the
        #   programs that use module parameters.
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        allow_non_fake_inputs = while_loop_test in (
            "simple_with_linear",
            "nested_with_linear",
        )
        self._check_tracing(fn, inp, allow_non_fake_inputs)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("while_loop_test", list(WHILE_LOOP_TESTS.keys()))
    def test_while_loop_compile(self, backend, while_loop_test):
        # [FUNCTION DOC] Parametrized over eager and aot_eager backends and every while_loop test
        #   program. Compiles each and compares with eager.
        fn, inp = WHILE_LOOP_TESTS[while_loop_test]
        self._check_compile(fn, inp, backend=backend)

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    @skipIfCrossRef  # Arg order changes with cross ref
    def test_while_loop_simple_with_linear_compile_check_graph(self):
        # [FUNCTION DOC] Skipped under dynamo and cross-ref. Compiles the simple-with-linear
        #   program with the eager recording backend and compares the golden graph, in which the
        #   buffer, weight and bias are passed to the while_loop as additional inputs to the cond
        #   and body subgraphs.
        fn, inp = WHILE_LOOP_TESTS["simple_with_linear"]
        backend = EagerAndRecordGraphs()
        torch.compile(fn, backend=backend)(*inp)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_iter_: "i64[]", L_x_: "f32[2, 2]", L_self_buffers_dec_: "i64[]", L_self_modules_linear_parameters_weight_: "f32[2, 2]", L_self_modules_linear_parameters_bias_: "f32[2]"):
        l_iter_ = L_iter_
        l_x_ = L_x_
        l_self_buffers_dec_ = L_self_buffers_dec_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_iter_, l_x_), (l_self_buffers_dec_, l_self_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_));  cond_fn_0 = body_fn_0 = l_iter_ = l_x_ = l_self_buffers_dec_ = l_self_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = None
        getitem: "i64[]" = while_loop[0]
        getitem_1: "f32[2, 2]" = while_loop[1];  while_loop = None
        return (getitem, getitem_1)

    class cond_fn_0(torch.nn.Module):
        def forward(self, child: "i64[]", child_1: "f32[2, 2]", l_self_buffers_dec__cond_fn: "i64[]", l_self_modules_linear_parameters_bias__body_fn: "f32[2]", l_self_modules_linear_parameters_weight__body_fn: "f32[2, 2]"):
            sub: "i64[]" = child - l_self_buffers_dec__cond_fn;  child = l_self_buffers_dec__cond_fn = None
            gt: "b8[]" = sub > 0;  sub = None
            return gt

    class body_fn_0(torch.nn.Module):
        def forward(self, child_2: "i64[]", child_3: "f32[2, 2]", l_self_buffers_dec__cond_fn: "i64[]", l_self_modules_linear_parameters_bias__body_fn: "f32[2]", l_self_modules_linear_parameters_weight__body_fn: "f32[2, 2]"):
            child: "i64[]" = child_2 - 1;  child_2 = None
            child_4: "f32[2, 2]" = torch._C._nn.linear(child_3, l_self_modules_linear_parameters_weight__body_fn, l_self_modules_linear_parameters_bias__body_fn);  child_3 = l_self_modules_linear_parameters_weight__body_fn = l_self_modules_linear_parameters_bias__body_fn = None
            return (child, child_4)
""",
        )

    def test_while_loop_nested2_traced(self):
        # [FUNCTION DOC] Traces the nested module program in all modes and compares golden code of
        #   the symbolic graph, the outer body, the inner body and the inner condition.
        # KNOWN ISSUE: the golden check for the outer body is written twice with identical
        #   content; the repeated check adds no coverage.
        fn, inp = WHILE_LOOP_TESTS["nested2"]
        graphs = self._check_tracing(fn, inp)
        gm = graphs["symbolic"]
        outer_body = gm.while_loop_body_graph_0
        inner_body = outer_body.while_loop_body_graph_0
        inner_cond = outer_body.while_loop_cond_graph_0
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 1)
    sym_size_int_1 = torch.ops.aten.sym_size.int(arg2_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(arg2_1, 0)
    sym_size_int_3 = torch.ops.aten.sym_size.int(arg3_1, 0)
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (sym_size_int, sym_size_int_1, sym_size_int_2, sym_size_int_3));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    return (getitem, getitem_1, getitem_2, getitem_3)
    """,
        )
        self.assertExpectedInline(
            outer_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (arg7_1, arg7_1, arg7_1, arg7_1));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg7_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1);  getitem = None
    clone = torch.ops.aten.clone.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2);  getitem_2 = None
    div = torch.ops.aten.div.Tensor(getitem_3, 2);  getitem_3 = None
    return (sub, clone, mul, div)
    """,
        )
        self.assertExpectedInline(
            outer_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), (arg7_1, arg7_1, arg7_1, arg7_1));  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg7_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1);  getitem = None
    clone = torch.ops.aten.clone.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2);  getitem_2 = None
    div = torch.ops.aten.div.Tensor(getitem_3, 2);  getitem_3 = None
    return (sub, clone, mul, div)
    """,
        )
        self.assertExpectedInline(
            inner_body.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    sub = torch.ops.aten.sub.Tensor(arg1_1, 1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg2_1, 3.14);  arg2_1 = None
    sub_1 = torch.ops.aten.sub.Tensor(arg3_1, 2.71);  arg3_1 = None
    return (clone, sub, add, sub_1)
    """,
        )
        self.assertExpectedInline(
            inner_cond.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    gt = torch.ops.aten.gt.Scalar(arg1_1, 0);  arg1_1 = None
    return gt
    """,
        )

    def test_cond_nested_traced(self):
        # [FUNCTION DOC] Traces a nested cond with make_fx and checks the traced graph's results
        #   for all four predicate combinations against expected formulas (square plus x, double
        #   plus x, or cosine), then checks the symbolic trace matches eager and, if CUDA graph
        #   conditional nodes are supported, runs the cudagraph helpers.
        def true_nested(y):
            # [FUNCTION DOC] HELPER: returns y times y.
            return y * y

        def false_nested(y):
            # [FUNCTION DOC] HELPER: returns y plus y.
            return y + y

        def true_fn(x, pred2):
            # [FUNCTION DOC] HELPER: applies a nested cond on pred2 between the nested functions
            #   and returns x plus the result.
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            # [FUNCTION DOC] HELPER: returns the cosine of x, ignoring the second argument.
            return x.cos()

        def f(x, pred, pred2):
            # [FUNCTION DOC] HELPER: cond on pred between true_fn and false_fn with operands
            #   x and pred2.
            return cond(pred, true_fn, false_fn, [x, pred2])

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        result_true_true = graph.forward(
            x, torch.tensor(True), torch.tensor(True)
        )  # True + True -> x * x
        result_true_false = graph.forward(
            x, torch.tensor(True), torch.tensor(False)
        )  # True + True -> x + x
        result_false_true = graph.forward(
            x, torch.tensor(False), torch.tensor(True)
        )  # False + either -> cos
        result_false_false = graph.forward(
            x, torch.tensor(False), torch.tensor(False)
        )  # False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertEqual(result_false_true, result_false_false)

        self.assertEqual(result_true_true, (x * x) + x)
        self.assertEqual(result_true_false, x + x + x)

        self.assertEqual(result_false_true, torch.cos(x))

        graph = make_fx(f, tracing_mode="symbolic")(
            x, torch.tensor(False), torch.tensor(False)
        )
        self.assertEqual(
            graph(x, torch.tensor(True), torch.tensor(True)),
            f(x, torch.tensor(True), torch.tensor(True)),
        )

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            x_cuda = x.cuda()
            true_pred = torch.tensor(True).cuda()

            _check_compile_cudagraph_backend(
                self,
                f,
                [x_cuda, true_pred, true_pred],
            )
            _check_compile_many_backends_with_cudagraph(
                self,
                f,
                [x_cuda, true_pred, true_pred],
            )

    def test_cond_functionalized(self):
        # [FUNCTION DOC] Checks cond under torch.func.functionalize gives the same result as
        #   eager, that the symbolic make_fx trace of the functionalized function matches eager
        #   and that none of the ops in the true branch subgraph are mutable; if CUDA graph
        #   conditional nodes are supported, also runs the cudagraph helpers.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x, mutated in place (add 4) in a local; returns max
            #   of sine of x plus sum of the mutated local.
            y = x.sin()
            y.add_(4)
            return x.sin().max() + y.sum()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: min of cosine of x.
            return x.cos().min()

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        def f_(x, y):
            # [FUNCTION DOC] HELPER: cond on a predicate tensor argument.
            return cond(y, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        all_ops_in_true_branch = []
        for node in graph_module.true_graph_0.graph.nodes:
            if node.op == "call_function":
                all_ops_in_true_branch.append(node.target)

        self.assertFalse(any(op._schema.is_mutable for op in all_ops_in_true_branch))

        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            pred = torch.tensor(example_inputs[0].shape[0] == 1, device="cuda")
            _check_compile_cudagraph_backend(self, f_, [torch.ones(4, 5).cuda(), pred])
            _check_compile_many_backends_with_cudagraph(
                self, f_, [torch.ones(4, 5).cuda(), pred]
            )

    def test_cond_accepts_torch_function_as_inputs(self):
        # [FUNCTION DOC] Uses torch.add and torch.mul directly as cond branches with a data-
        #   dependent predicate; traces with all modes and compares golden code of the symbolic
        #   graph and the two branch subgraphs.
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        def f(a, b):
            # [FUNCTION DOC] HELPER: cond on whether the sum of a is positive, with branches
            #   torch.add and torch.mul and operands a and b.
            return cond(a.sum() > 0, torch.add, torch.mul, (a, b))

        gm = self._check_tracing(f, (a, b))["symbolic"]
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, a_1, b_1):
    sum_1 = torch.ops.aten.sum.default(a_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 1)
    sym_size_int_1 = torch.ops.aten.sym_size.int(b_1, 0)
    sym_size_int_2 = torch.ops.aten.sym_size.int(b_1, 1)
    sym_size_int_3 = torch.ops.aten.sym_size.int(a_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (a_1, b_1, sym_size_int, sym_size_int_1, sym_size_int_2, sym_size_int_3));  gt = true_graph_0 = false_graph_0 = a_1 = b_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = sym_size_int_3 = None
    getitem = cond[0];  cond = None
    return getitem""",
        )
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )
        self.assertExpectedInline(
            gm.false_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (mul,)""",
        )

    def test_cond_retrace_functionalized(self):
        # [FUNCTION DOC] Traces cond with make_fx, then retraces the functionalized traced
        #   module and checks the retraced graph gives the same result as the original function
        #   on a zero input.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether all elements of x are nonzero.
            return cond(x.all(), true_fn, false_fn, (x,))

        inp = torch.ones(1, 2)
        gm_non_functional = make_fx(f, tracing_mode="real")(inp)
        gm_functional = make_fx(
            torch.func.functionalize(gm_non_functional), tracing_mode="real"
        )(inp)
        self.assertEqual(gm_functional(torch.zeros(1, 2)), f(torch.zeros(1, 2)))

    def test_cond_subgraph_same_shape_env_as_parent(self):
        # [FUNCTION DOC] Traces cond with symbolic tracing and checks the nodes of the
        #   outer graph and of the true and false subgraphs all share the shape environment
        #   of the outer graph.
        # KNOWN ISSUE: the helper iterating nodes ignores its argument and always walks the outer
        #   graph, so the subgraph loops re-check the outer graph and do not verify the subgraphs.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x plus 10.
            return x.sin() + 10

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x minus 20.
            return x.cos() - 20

        def f(x, pred):
            # [FUNCTION DOC] HELPER: cond on pred and then doubles the result by adding
            #   it to itself.
            y = cond(pred, true_fn, false_fn, [x])
            z = torch.add(y, y)
            return z

        symbolic_traced_graph = self._check_tracing(
            f, (torch.ones(4), torch.Tensor([True]))
        )["symbolic"]
        graph_shape_env = symbolic_traced_graph.shape_env

        def _node_shape_env_iter(gm):
            # [FUNCTION DOC] HELPER: generator yielding the shape environment associated with the
            #   value metadata of each call_function node of the outer symbolic graph (handling
            #   tuples, symbolic ints and tensors).
            # KNOWN ISSUE: the gm parameter is never used; the loop reads
            #   the enclosing symbolic_traced_graph.
            for node in symbolic_traced_graph.graph.nodes:
                if node.op == "call_function":
                    val = node.meta.get("val")
                    if isinstance(val, tuple):
                        for v in val:
                            yield v.fake_mode.shape_env
                    elif isinstance(val, torch.SymInt):
                        yield val.node.shape_env
                    else:
                        yield val.fake_mode.shape_env

        for shape_env in _node_shape_env_iter(symbolic_traced_graph):
            self.assertTrue(shape_env is graph_shape_env)

        for shape_env in _node_shape_env_iter(symbolic_traced_graph.true_graph_0):
            self.assertTrue(shape_env is graph_shape_env)

        for shape_env in _node_shape_env_iter(symbolic_traced_graph.false_graph_0):
            self.assertTrue(shape_env is graph_shape_env)

    def test_cond_functionalized_nested(self):
        # [FUNCTION DOC] Checks nested cond under functionalize matches eager, traces
        #   symbolically, and checks that the innermost true branch subgraph contains
        #   no mutable ops.
        def true_true_fn(x):
            # [FUNCTION DOC] HELPER: mutates a cosine local in place and returns max of sine of x
            #   plus max of sine of the local.
            y = x.cos()
            y.add_(4)
            return x.sin().max() + y.sin().max()

        def true_false_fn(x):
            # [FUNCTION DOC] HELPER: min of cosine of x.
            return x.cos().min()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: nested cond on whether the leading size is 1 between the
            #   two functions above.
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sum of x.
            return x.sum()

        def f(x):
            # [FUNCTION DOC] HELPER: outer cond on whether the leading size is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        gm_true_true_branch = graph_module.true_graph_0.true_graph_0

        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        all_ops = []
        for node in gm_true_true_branch.graph.nodes:
            if node.op == "call_function":
                all_ops.append(node.target)

        self.assertFalse(any(op._schema.is_mutable for op in all_ops))

    def test_cond_functionalized_data_dependent_pred(self):
        # [FUNCTION DOC] Checks cond under functionalize with a data-dependent predicate (count of
        #   nonzero elements) matches eager both eagerly and in a make_fx trace.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sum of sine of x.
            return x.sin().sum()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sum of cosine of x.
            return x.cos().sum()

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the number of nonzero entries of x is 1.
            pred = x.nonzero().shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_functionalized_input_mutation_on_true_branch(self):
        # [FUNCTION DOC] With a constant predicate, make_fx of the functionalized function inlines
        #   the true branch (golden graph with view, add, copy back to the input). With a symbolic
        #   trace, expects TorchRuntimeError saying cond_true might be modifying the input.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: mutates a view of x in place (add one) and returns the sum
            #   of its sine.
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.sin().sum()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sum of cosine of x.
            return x.cos().sum()

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 4.
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [4, 5])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None
    view_1 = torch.ops.aten.view.default(add, [4, 5]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 5])
    sin = torch.ops.aten.sin.default(view_2);  view_2 = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_1);  x_1 = view_1 = copy_ = None
    return sum_1""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_true might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

    def test_cond_functionalized_input_mutation_on_false_branch(self):
        # [FUNCTION DOC] With a constant predicate (5 by 5 input, predicate false) the false
        #   branch is inlined (golden graph). With a symbolic trace expects TorchRuntimeError
        #   saying cond_false might be modifying the input.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sum of sine of x.
            return x.sin().sum()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: mutates a view of x in place (add one) and returns the sum
            #   of its cosine.
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.cos().sum()

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 4.
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 5),)
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [5, 5])
    add = torch.ops.aten.add.Tensor(view, 1);  view = None
    view_1 = torch.ops.aten.view.default(add, [5, 5]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [5, 5])
    cos = torch.ops.aten.cos.default(view_2);  view_2 = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    copy_ = torch.ops.aten.copy_.default(x_1, view_1);  x_1 = view_1 = copy_ = None
    return sum_1""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_false might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

    def test_cond_functionalized_output_alias_input(self):
        # [FUNCTION DOC] With a constant predicate the false branch (which returns a view
        #   of the input) is inlined (golden graph). With a symbolic trace expects
        #   UncapturedHigherOrderOpError naming torch.cond because a branch output
        #   aliases its input.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: clone of x.
            return x.clone()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a view of x with the same shape.
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 4.
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(5, 5),)
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        # torch.cond inlines into one of the branches because the predicate
        # is a constant.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    view = torch.ops.aten.view.default(x_1, [5, 5]);  x_1 = None
    return view""",
        )

        # torch.cond triggers the check of the branches because the predicate
        # is a SymBool.
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

    def test_cond_functionalized_nested_input_mutation(self):
        # [FUNCTION DOC] Nested cond where the innermost true branch mutates its input in place;
        #   symbolic make_fx of the functionalized function raises TorchRuntimeError saying
        #   cond_true might be modifying the input.
        def true_true_fn(x):
            # [FUNCTION DOC] HELPER: adds 4 to x in place and returns the max of sine of x.
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            # [FUNCTION DOC] HELPER: min of cosine of x.
            return x.cos().min()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: nested cond on whether the leading size is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sum of x.
            return x.sum()

        def f(x):
            # [FUNCTION DOC] HELPER: outer cond on whether the leading size is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "cond_true might be modifying the input!",
        ):
            make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
                *example_inputs
            )

    def test_cond_functionalized_nested_input_mutation_with_aot_func(self):
        # [FUNCTION DOC] Uses manual functionalization of the input (old-style functional tensor
        #   with functionalization enabled; the finally clause disables it). Running nested cond
        #   with an input-mutating branch under symbolic make_fx must raise
        #   UncapturedHigherOrderOpError naming torch.cond, both when called directly on the
        #   functional input and through a wrapper that enables functionalization around the call.
        def true_true_fn(x):
            # [FUNCTION DOC] HELPER: adds 4 to x in place and returns max of sine of x.
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            # [FUNCTION DOC] HELPER: min of cosine of x.
            return x.cos().min()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: nested cond on whether the leading size is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sum of x.
            return x.sum()

        def f(x):
            # [FUNCTION DOC] HELPER: outer cond on whether the leading size is 1.
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(4, 5)
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            f(example_input_func)

            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                make_fx(f, tracing_mode="symbolic")(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory; the wrapped function enables
            #   functionalization (without view reapplying) around the call and always
            #   disables it afterwards.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: enables functionalization, calls the wrapped function
                #   with the same arguments, and disables functionalization in a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    return func(*args, **kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input_func)

    def test_cond_functionalized_input_aliasing_with_aot_func(self):
        # [FUNCTION DOC] Branches return the input itself or a view of it; with manual
        #   functionalization enabled, calling cond (directly on a functional input, or
        #   via a wrapper converting tensor args to functional tensors) must raise
        #   UncapturedHigherOrderOpError naming torch.cond (the wrapped call is made
        #   under symbolic make_fx).
        def true_fn(x):
            # [FUNCTION DOC] HELPER: returns x itself.
            return x

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a view of x with the same shape.
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the sum of x is positive.
            pred = x.sum() > 0
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                f(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory; the wrapped function enables
            #   functionalization, converts all tensor positional and keyword arguments to
            #   functional tensors, calls the function, and disables functionalization afterwards.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: enables functionalization, maps tensor arguments to
                #   functional tensors, calls the function, and disables functionalization in
                #   a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        kwargs,
                    )
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input)

    def test_cond_functionalized_aot_func_check_functional(self):
        # [FUNCTION DOC] Traces cond with a wrapper that functionalizes arguments (old-style),
        #   runs the function and converts the outputs back; checks no call_function node in the
        #   true and false subgraphs has a mutable schema, and that the traced module gives the
        #   same result as the original function.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sine of x, mutated in place (add 5) in a local, returned.
            y = x.sin()
            y.add_(5)
            return y

        def f(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 4.
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory; the wrapped function enables
            #   functionalization, maps tensor args and kwargs to functional tensors, runs
            #   the function, converts the outputs back, and disables functionalization in
            #   a finally clause.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: does the functionalize, call and convert-back sequence
                #   described for f_wrapper with cleanup in a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
                finally:
                    torch._disable_functionalization()

            return wrapper

        result_gm = make_fx(f_wrapper(f), tracing_mode="symbolic")(example_input)
        for node in result_gm.true_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        for node in result_gm.false_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        self.assertEqual(result_gm(torch.ones(5, 5)), f(torch.ones(5, 5)))

    def test_cond_nested_traced_other_inputs(self):
        # [FUNCTION DOC] Traces nested cond where the true branch adds a constant tensor to the
        #   nested result; checks the traced graph on two different inputs for the true-true case.
        def true_nested(y):
            # [FUNCTION DOC] HELPER: y times y.
            return y * y

        def false_nested(y):
            # [FUNCTION DOC] HELPER: y plus y.
            return y + y

        def true_fn(k, pred2):
            # [FUNCTION DOC] HELPER: nested cond on pred2 between the nested functions, plus a
            #   constant tensor of 0.25s.
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        def false_fn(k, _):
            # [FUNCTION DOC] HELPER: cosine of k.
            return k.cos()

        def f(k, pred, pred2):
            # [FUNCTION DOC] HELPER: outer cond on pred between true_fn and false_fn with operands
            #   k and pred2.
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi(self):
        # [FUNCTION DOC] Traces two sequential conds (square versus double on one predicate,
        #   add versus multiply on another) and compares the golden outer graph and the
        #   first true subgraph; if CUDA graph conditional nodes are supported, also runs
        #   the cudagraph helpers.
        def true_a(y):
            # [FUNCTION DOC] HELPER: y times y.
            return y * y

        def false_a(y):
            # [FUNCTION DOC] HELPER: y plus y.
            return y + y

        def true_b(y, z):
            # [FUNCTION DOC] HELPER: y plus z.
            return y + z

        def false_b(y, z):
            # [FUNCTION DOC] HELPER: y times z.
            return y * z

        def f(x, pred, pred2):
            # [FUNCTION DOC] HELPER: adds the results of two conds with different predicates.
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out

        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, x_1, pred_1, pred2_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, (x_1,));  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = cond_1[0];  cond_1 = None
    add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return add""",
        )
        self.assertExpectedInline(
            graph.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return (mul,)""",
        )

        if TEST_CUDA_GRAPH_CONDITIONAL_NODES:
            _check_compile_cudagraph_backend(
                self,
                f,
                [x.cuda(), torch.tensor(False).cuda(), torch.tensor(False).cuda()],
            )
            _check_compile_many_backends_with_cudagraph(
                self,
                f,
                [x.cuda(), torch.tensor(False).cuda(), torch.tensor(False).cuda()],
            )

    def test_raise_error_on_mismatch_type_size(self):
        # [FUNCTION DOC] Branches returning a single tensor versus a tuple make tracing cond raise
        #   UncapturedHigherOrderOpError naming torch.cond.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a tuple of x twice.
            return (x, x)

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f)(x, torch.tensor(False))

    def test_raise_error_on_mismatch_tensor_size(self):
        # [FUNCTION DOC] Branches returning tensors of different shapes raise TorchRuntimeError
        #   mentioning merging two branches' output in torch.cond.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a new 10 by 10 zero tensor.
            return torch.zeros([10, 10])

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "When merging two branches' output in torch.cond",
        ):
            make_fx(f)(x, torch.tensor(False))

    def test_cond_traced_not_nested_fake_tensor(self):
        # [FUNCTION DOC] Same as the not-nested traced cond test but with fake tracing mode: the
        #   traced graph selects sine for a true predicate and cosine for a false one.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))

    def test_cond_nested_traced_fake_tensor(self):
        # [FUNCTION DOC] Fake-mode version of the nested cond tracing test; checks all four
        #   predicate combinations against expected values.
        def true_nested(y):
            # [FUNCTION DOC] HELPER: y times y.
            return y * y

        def false_nested(y):
            # [FUNCTION DOC] HELPER: y plus y.
            return y + y

        def true_fn(x, pred2):
            # [FUNCTION DOC] HELPER: nested cond on pred2 and returns x plus the result.
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def f(x, pred, pred2):
            # [FUNCTION DOC] HELPER: outer cond on pred.
            return cond(pred, true_fn, false_fn, [x, pred2])

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        result_true_true = graph.forward(
            x, torch.tensor(True), torch.tensor(True)
        )  # True + True -> x * x
        result_true_false = graph.forward(
            x, torch.tensor(True), torch.tensor(False)
        )  # True + True -> x + x
        result_false_true = graph.forward(
            x, torch.tensor(False), torch.tensor(True)
        )  # False + either -> cos
        result_false_false = graph.forward(
            x, torch.tensor(False), torch.tensor(False)
        )  # False + either -> cos

        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        self.assertEqual(result_false_true, result_false_false)

        self.assertEqual(result_true_true, (x * x) + x)
        self.assertEqual(result_true_false, x + x + x)

        self.assertEqual(result_false_true, torch.cos(x))

    def test_cond_nested_traced_other_inputs_fake_tensor(self):
        # [FUNCTION DOC] Fake-mode version of the nested cond test with a constant tensor added in
        #   the true branch; checks results for two inputs.
        def true_nested(y):
            # [FUNCTION DOC] HELPER: y times y.
            return y * y

        def false_nested(y):
            # [FUNCTION DOC] HELPER: y plus y.
            return y + y

        def true_fn(k, pred2):
            # [FUNCTION DOC] HELPER: nested cond on pred2, plus a constant tensor of 0.25s.
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        def false_fn(k, _):
            # [FUNCTION DOC] HELPER: cosine of k.
            return k.cos()

        def f(k, pred, pred2):
            # [FUNCTION DOC] HELPER: outer cond on pred.
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi_fake_tensor(self):
        # [FUNCTION DOC] Fake-mode version of the sequential conds test; compares golden outer
        #   graph and first true subgraph.
        def true_a(y):
            # [FUNCTION DOC] HELPER: y times y.
            return y * y

        def false_a(y):
            # [FUNCTION DOC] HELPER: y plus y.
            return y + y

        def true_b(y, z):
            # [FUNCTION DOC] HELPER: y plus z.
            return y + z

        def false_b(y, z):
            # [FUNCTION DOC] HELPER: y times z.
            return y * z

        def f(x, pred, pred2):
            # [FUNCTION DOC] HELPER: adds the results of two conds with different predicates.
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out

        x = torch.randn(4)
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        self.assertExpectedInline(
            graph.code.strip(),
            """\
def forward(self, x_1, pred_1, pred2_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, (x_1,));  pred_1 = true_graph_0 = false_graph_0 = None
    getitem = cond[0];  cond = None
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    cond_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, (x_1,));  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    getitem_1 = cond_1[0];  cond_1 = None
    add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    return add""",
        )
        self.assertExpectedInline(
            graph.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return (mul,)""",
        )

    def test_raise_error_on_mismatch_type_size_fake_tensor(self):
        # [FUNCTION DOC] Fake-mode version of the type/size mismatch test; branches
        #   returning a tensor versus a tuple make tracing raise
        #   UncapturedHigherOrderOpError naming torch.cond.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a tuple of x twice.
            return (x, x)

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))

    def test_raise_error_on_mismatch_tensor_size_fake_tensor(self):
        # [FUNCTION DOC] Fake-mode version of the tensor size mismatch test; branches returning
        #   tensors of different shapes raise TorchRuntimeError mentioning merging two branches'
        #   output in torch.cond.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: returns a new 10 by 10 zero tensor.
            return torch.zeros([10, 10])

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on y with the two branches.
            return cond(y, true_fn, false_fn, [x])

        x = torch.randn(4)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "When merging two branches' output in torch.cond",
        ):
            make_fx(f, tracing_mode="fake")(x, torch.tensor(False))

    def check_map_count(self, gm, op_count):
        # [FUNCTION DOC] HELPER: counts call_function nodes targeting the map_impl higher-
        #   order op across the graph module and all of its submodules and asserts the
        #   count equals op_count.
        i = 0
        for m in gm.modules():
            for node in m.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target == torch.ops.higher_order.map_impl
                ):
                    i += 1
        self.assertEqual(i, op_count)

    def test_tracing_map_real(self):
        # [FUNCTION DOC] Traces a map call with make_fx in real mode, checks the traced graph
        #   result equals eager on new inputs and that the graph holds exactly one map op.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        def g(xs, y):
            # [FUNCTION DOC] HELPER: maps f over xs with y as extra argument.
            return control_flow.map(f, xs, y)

        gm = make_fx(g, tracing_mode="real")(torch.ones(3, 2, 2), torch.ones(2))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_simple(self):
        # [FUNCTION DOC] Traces map in symbolic mode and runs the graph on inputs of different
        #   shape; checks it matches eager and has one map op.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        def g(xs, y):
            # [FUNCTION DOC] HELPER: maps f over xs with y.
            return control_flow.map(f, xs, y)

        gm = make_fx(g, tracing_mode="symbolic")(torch.ones(3, 2, 4), torch.ones(4))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_list(self):
        # [FUNCTION DOC] Symbolic tracing of map over a nested list of tensors with a list output,
        #   followed by element-wise ops; checks the graph on different-shaped inputs matches
        #   eager and has one map op.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a list with the nested first input plus y and the
            #   second input times y.
            return [x[0][0] + y, x[1] * y]

        def g(xs, y, z):
            # [FUNCTION DOC] HELPER: maps f then multiplies or adds z to the two outputs.
            out = control_flow.map(f, xs, y)
            return out[0] + z, out[1] * z

        example_x = [[torch.ones(3, 4, 5)], torch.ones(3, 4, 5)]
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
        x = [[torch.randn(4, 5, 6)], torch.ones(4, 5, 6)]
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_dict(self):
        # [FUNCTION DOC] Symbolic tracing of map over a dict input with dict output; checks the
        #   graph on different-shaped inputs matches eager and has one map op.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a dict with the nested "a" entry plus y and the "c"
            #   entry times y.
            return {"d": x["b"]["a"] + y, "e": x["c"] * y}

        def g(xs, y, z):
            # [FUNCTION DOC] HELPER: maps f then adds and multiplies z to the outputs,
            #   returning a dict.
            out = control_flow.map(f, xs, y)
            return {"f": out["d"] + z, "g": out["e"] * z}

        example_x = {"b": {"a": torch.ones(3, 4, 5)}, "c": torch.ones(3, 4, 5)}
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
        x = {"b": {"a": torch.randn(4, 5, 6)}, "c": torch.ones(4, 5, 6)}
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_autograd_symbolic_simple(self):
        # [FUNCTION DOC] Traces map plus autograd.grad in symbolic mode; checks the graph matches
        #   eager gradients and that two map ops exist (forward and backward).
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        def g(xs, y):
            # [FUNCTION DOC] HELPER: maps f and returns the gradients of the output wrt xs and y
            #   with ones as output gradient.
            out = control_flow.map(f, xs, y)
            return torch.autograd.grad(out, (xs, y), torch.ones_like(out))

        gm = make_fx(g, tracing_mode="symbolic")(
            torch.ones(3, 4, 5, requires_grad=True), torch.ones(5, requires_grad=True)
        )
        x = torch.randn(4, 5, 6, requires_grad=True)
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_symbolic_list(self):
        # [FUNCTION DOC] Symbolic trace of map with list input/output and some inputs not
        #   requiring grad; gradients are taken only wrt inputs that require grad; checks the
        #   graph matches eager and has two map ops.
        import torch.utils._pytree as pytree

        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a list with cosine of the first input plus sine of y
            #   and sine of the second input times cosine of y.
            return [x[0].cos() + y.sin(), x[1].sin() * y.cos()]

        def g(xs, y):
            # [FUNCTION DOC] HELPER: maps f, flattens outputs and inputs, and returns gradients
            #   wrt the inputs requiring grad.
            out = control_flow.map(f, xs, y)
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        gm = make_fx(g, tracing_mode="symbolic")(
            [torch.ones(3, 4, 5), torch.ones(3, 4, 5, requires_grad=True)],
            torch.ones(5, requires_grad=True),
        )
        x = [torch.randn(4, 5, 6), torch.ones(4, 5, 6, requires_grad=True)]
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_symbolic_dict(self):
        # [FUNCTION DOC] Same as the list variant but with a dict input; checks the graph matches
        #   eager and has two map ops.
        def f(x, y):
            # [FUNCTION DOC] HELPER: returns a list holding a plus y and b times y.
            return [x["a"] + y, x["b"] * y]

        def g(xs, y):
            # [FUNCTION DOC] HELPER: maps f, flattens outputs and inputs, returns gradients wrt
            #   the inputs requiring grad.
            out = control_flow.map(f, xs, y)
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        traced_x = {
            "a": torch.ones(3, 4, 5, requires_grad=True),
            "b": torch.ones(3, 4, 5, requires_grad=True),
        }
        gm = make_fx(g, tracing_mode="symbolic")(
            traced_x, torch.ones(5, requires_grad=True)
        )
        x = {
            "a": torch.randn(4, 5, 6, requires_grad=True),
            "b": torch.ones(4, 5, 6, requires_grad=True),
        }
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_aot_functionalized(self):
        # [FUNCTION DOC] Traces map plus gradients with and without a functionalizing wrapper;
        #   checks the plain trace matches eager and counts mutable ops (including inside map
        #   bodies): none for the functionalized trace and two for the plain one (forward and
        #   recomputation in backward).
        def inner(x, y):
            # [FUNCTION DOC] HELPER: computes x minus 1, mutates it in place (add 1), and returns
            #   it times y.
            z = x - 1
            z.add_(1)
            return z * y

        def f(xs, y):
            # [FUNCTION DOC] HELPER: maps inner and returns gradients of the result wrt xs and y.
            res = control_flow.map(inner, xs, y)
            grads = torch.autograd.grad(res, (xs, y), torch.ones_like(res))
            return grads

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory enabling functionalization around the
            #   wrapped call and converting outputs back.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: enables functionalization, runs the function, maps
                #   outputs back from functional tensors, and disables functionalization in
                #   a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()

            return wrapper

        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(2, 4, requires_grad=True),
        )
        gm = make_fx(f, tracing_mode="symbolic")(*example_inputs)
        fgm = make_fx(f_wrapper(f), tracing_mode="symbolic")(*example_inputs)
        xs = torch.ones(3, 4, 5, requires_grad=True)
        y = torch.ones(4, 5, requires_grad=True)

        self.assertEqual(gm(xs, y), f(xs, y))

        def count_mutable(gm):
            # [FUNCTION DOC] HELPER: recursively counts call_function nodes in the graph whose
            #   schema is mutable, descending into map_impl body submodules.
            c = 0
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    if node.target == torch.ops.higher_order.map_impl:
                        c += count_mutable(getattr(gm, str(node.args[0])))
                    elif schema := getattr(node.target, "_schema", None):
                        c += int(schema.is_mutable)
            return c

        self.assertEqual(count_mutable(fgm), 0)
        # One for forward, one for recomputation logic in backward
        self.assertEqual(count_mutable(gm), 2)

    def test_map_functionalized(self):
        # [FUNCTION DOC] Checks map under torch.func.functionalize matches eager (including
        #   make_fx real and symbolic traces); verifies no op in the traced body graph is mutable
        #   and that there is one map op.
        def map_fn(x, y):
            # [FUNCTION DOC] HELPER: adds x and y then mutates the result in place (add 4)
            #   and returns it.
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            # [FUNCTION DOC] HELPER: maps map_fn over xs with y.
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

        gm = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

        for node in gm.body_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)
        self.check_map_count(gm, 1)

    def test_map_functionalized_aot_func(self):
        # [FUNCTION DOC] Traces map with a wrapper that enables functionalization; checks the body
        #   graph has no mutable ops and the result matches eager.
        def map_fn(x, y):
            # [FUNCTION DOC] HELPER: adds x and y then mutates the sum in place (add 4)
            #   and returns it.
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            # [FUNCTION DOC] HELPER: maps map_fn over xs with y.
            return control_flow.map(map_fn, xs, y)

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory enabling functionalization around the call.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: enables functionalization, runs the function,
                #   converts outputs back from functional tensors, disables functionalization
                #   in a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()

            return wrapper

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))

        gm = make_fx(f_wrapper(f))(*example_inputs)

        for node in gm.body_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        self.assertEqual(gm(*example_inputs), f(*example_inputs))

    def test_map_functionalized_arg_mutation(self):
        # [FUNCTION DOC] A map function that mutates its extra argument in place under
        #   functionalize raises TorchRuntimeError saying map might be modifying the input.
        def map_fn(x, y):
            # [FUNCTION DOC] HELPER: adds 4 to y in place and returns x plus y.
            y.add_(4)
            return x + y

        def f(xs, y):
            # [FUNCTION DOC] HELPER: maps map_fn over xs with y.
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            "map might be modifying the input!",
        ):
            functional_f(*example_inputs)

    def test_map_functionalized_elem_mutation(self):
        # [FUNCTION DOC] A map function that mutates its mapped element in place under
        #   functionalize raises TorchRuntimeError saying map might be modifying the input.
        def map_fn(x, y):
            # [FUNCTION DOC] HELPER: adds 4 to x in place and returns x plus y.
            x.add_(4)
            return x + y

        def f(xs, y):
            # [FUNCTION DOC] HELPER: maps map_fn over xs with y.
            return control_flow.map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError, "map might be modifying the input!"
        ):
            functional_f(*example_inputs)

    def test_cond_autograd_backward(self):
        # [FUNCTION DOC] Runs cond with a size-based predicate and backward through it; checks no
        #   error is raised when running backward, forward without backward, or compiled, and that
        #   the compiled result equals eager.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: cosine of x.
            return x.cos()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: sine of x.
            return x.sin()

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x exceeds 4,
            #   applied to y.
            return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [y])

        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(4, requires_grad=True),
        )
        f(*example_inputs).sum().backward()

        # Ensure no error is thrown when not running backward
        res = f(*example_inputs)

        # Ensure no error is thrown when not running backward
        res_compiled = torch.compile(f)(*example_inputs)
        self.assertEqual(res, res_compiled)

    @skipIfTorchDynamo("Skip because we're testing export")
    def test_cond_autograd_backward_inp_out_aliasing(self):
        # [FUNCTION DOC] Skipped under dynamo. Runs cond whose branches return the sum of inputs,
        #   with backward, eager and compiled with the AOT recording backend; unless cross-ref,
        #   compares the golden backward graph where the gradient outputs are cloned so they do
        #   not alias the tangent input; also checks compiled equals eager.
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        backend = AotEagerAndRecordGraphs()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on whether the sum of x exceeds 4 with the same function
            #   in both branches.
            return control_flow.cond(x.sum() > 4, fn, fn, (x, y))

        example_inputs = (
            torch.ones(3, 4, requires_grad=True),
            torch.ones(3, 4, requires_grad=True),
        )
        res = f(*example_inputs)
        res.sum().backward()
        res_compiled = torch.compile(f, backend=backend)(*example_inputs)
        res_compiled.sum().backward()
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", gt: "b8[]", tangents_1: "f32[3, 4]"):
        true_graph_1 = self.true_graph_1
        false_graph_1 = self.false_graph_1
        cond_1 = torch.ops.higher_order.cond(gt, true_graph_1, false_graph_1, (primals_1, primals_2, tangents_1));  gt = true_graph_1 = false_graph_1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 4]" = cond_1[0]
        getitem_2: "f32[3, 4]" = cond_1[1];  cond_1 = None
        return (getitem_1, getitem_2)

    class true_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1)
            clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
            return [clone, clone_1]

    class false_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1)
            clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(arg2_1);  arg2_1 = None
            return [clone, clone_1]
""",
            )
        self.assertEqual(res, res_compiled)

    @skipIfTorchDynamo("Skip because we're testing export")
    def test_cond_autograd_backward_out_out_aliasing(self):
        # [FUNCTION DOC] Skipped under dynamo. Like the input-output aliasing test but branches
        #   compute sine of the sum, so the two gradient outputs would alias each other; golden
        #   backward graph shows a clone separating them; also checks compiled equals eager.
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        backend = AotEagerAndRecordGraphs()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: returns sine of x plus y.
            return (x + y).sin()

        def f(x, y):
            # [FUNCTION DOC] HELPER: cond on whether the sum of x exceeds 4 with the same function
            #   in both branches.
            return control_flow.cond(x.sum() > 4, fn, fn, (x, y))

        example_inputs = (
            torch.ones(3, 4, requires_grad=True),
            torch.ones(3, 4, requires_grad=True),
        )
        res = f(*example_inputs)
        res.sum().backward()
        res_compiled = torch.compile(f, backend=backend)(*example_inputs)
        res_compiled.sum().backward()
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", gt: "b8[]", tangents_1: "f32[3, 4]"):
        true_graph_1 = self.true_graph_1
        false_graph_1 = self.false_graph_1
        cond_1 = torch.ops.higher_order.cond(gt, true_graph_1, false_graph_1, (primals_1, primals_2, tangents_1));  gt = true_graph_1 = false_graph_1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 4]" = cond_1[0]
        getitem_2: "f32[3, 4]" = cond_1[1];  cond_1 = None
        return (getitem_1, getitem_2)

    class true_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            cos: "f32[3, 4]" = torch.ops.aten.cos.default(add);  add = None
            mul: "f32[3, 4]" = torch.ops.aten.mul.Tensor(arg2_1, cos);  arg2_1 = cos = None
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(mul)
            return [mul, clone]

    class false_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]", arg2_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            cos: "f32[3, 4]" = torch.ops.aten.cos.default(add);  add = None
            mul: "f32[3, 4]" = torch.ops.aten.mul.Tensor(arg2_1, cos);  arg2_1 = cos = None
            clone: "f32[3, 4]" = torch.ops.aten.clone.default(mul)
            return [mul, clone]
""",
            )
        self.assertEqual(res, res_compiled)

    def test_map_functionalized_elem_alias(self):
        # [FUNCTION DOC] A map function returning its input element itself (aliasing) under
        #   functionalize raises UncapturedHigherOrderOpError naming the map_impl op.
        def map_fn(x):
            # [FUNCTION DOC] HELPER: creates an unused view of x and returns x itself.
            x.view(x.shape)
            return x

        def f(xs):
            # [FUNCTION DOC] HELPER: maps map_fn over xs.
            return control_flow.map(map_fn, xs)

        example_inputs = (torch.ones(3, 2, 4),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # Should be
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing.*"
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            functional_f(*example_inputs)

    def test_nested_map_cond_real(self):
        # [FUNCTION DOC] Traces a map whose function applies cond (multiply or add) in real
        #   tracing mode; checks the graph result against eager and one map op.
        def true_fn(x, y):
            # [FUNCTION DOC] HELPER: x times y.
            return x * y

        def false_fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        def f(x, pred, y):
            # [FUNCTION DOC] HELPER: cond on pred between true_fn and false_fn with operands
            #   x and y.
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            # [FUNCTION DOC] HELPER: maps f over xs with pred and y as extra arguments.
            return control_flow.map(f, xs, pred, y)

        gm = make_fx(g, tracing_mode="real")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 4)
        y = torch.randn(4)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_map_cond_symbolic(self):
        # [FUNCTION DOC] Same as the real nested map/cond test but with symbolic tracing and
        #   differently shaped runtime inputs.
        def true_fn(x, y):
            # [FUNCTION DOC] HELPER: x times y.
            return x * y

        def false_fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        def f(x, pred, y):
            # [FUNCTION DOC] HELPER: cond on pred between true_fn and false_fn with operands
            #   x and y.
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            # [FUNCTION DOC] HELPER: maps f over xs with pred and y.
            return control_flow.map(f, xs, pred, y)

        gm = make_fx(g, tracing_mode="symbolic")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_cond_map_cond_symbolic(self):
        # [FUNCTION DOC] Symbolic tracing of a cond whose branches contain a map whose function
        #   contains a cond; checks the traced graph against eager and that two map ops exist (one
        #   per outer branch).
        def true_fn(x, y):
            # [FUNCTION DOC] HELPER: x times y.
            return x * y

        def false_fn(x, y):
            # [FUNCTION DOC] HELPER: x plus y.
            return x + y

        def f(x, pred, y):
            # [FUNCTION DOC] HELPER: innermost cond on pred.
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            # [FUNCTION DOC] HELPER: maps f over xs.
            return control_flow.map(f, xs, pred, y)

        def main_true_fn(pred, xs, y):
            # [FUNCTION DOC] HELPER: the mapped result times 2.
            return g(pred, xs, y) * 2

        def main_false_fn(pred, xs, y):
            # [FUNCTION DOC] HELPER: the mapped result plus 1.
            return g(pred, xs, y) + 1

        def main(p, pred, xs, y):
            # [FUNCTION DOC] HELPER: outer cond on p between main_true_fn and main_false_fn.
            return cond(p, main_true_fn, main_false_fn, [pred, xs, y])

        gm = make_fx(main, tracing_mode="symbolic")(
            torch.tensor(True), torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        p = torch.tensor(False)
        pred = torch.tensor(False)
        xs = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(p, pred, xs, y)
        self.assertEqual(res, main(p, pred, xs, y))
        self.check_map_count(gm, 2)

    def test_cond_with_sym_pred(self):
        # [FUNCTION DOC] Traces cond with a predicate depending on a symbolic size using
        #   symbolic make_fx; asserts the shape env has no guards (nothing specialized),
        #   compares the golden graph, and checks the traced graph and eager run correctly on an
        #   input of different size.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: x plus x.
            return x + x

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x times x.
            return x * x

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size of x is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 2, 1))
        # The symbols in make_fx's shape_env should not be specialized.
        self.assertEqual(len(gm.shape_env.guards), 0)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, sym_size_int_1, sym_size_int));  eq = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = sym_size_int = None
    getitem = cond[0];  cond = None
    return getitem""",
        )

        # We expect the traced graph module to work even if input size changes.
        x = torch.ones(4, 3, 2)
        self.assertEqual(gm(x), true_fn(x))
        self.assertEqual(foo(x), true_fn(x))

    def test_cond_with_unbacked_sym_pred(self):
        # [FUNCTION DOC] Traces cond whose predicate depends on the size of a nonzero result (an
        #   unbacked symbolic value) and compares the golden graph.
        def foo(x):
            # [FUNCTION DOC] HELPER: defines the branches, computes nonzero indices of x, and runs
            #   cond on whether their count exceeds 3.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: x plus x.
                return x + x

            def false_fn(x):
                # [FUNCTION DOC] HELPER: x times x.
                return x * x

            az = x.nonzero()
            return cond(az.shape[0] > 3, true_fn, false_fn, (x,))

        gm = make_fx(foo, tracing_mode="symbolic")(torch.randn(7))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    nonzero = torch.ops.aten.nonzero.default(x_1)
    sym_size_int = torch.ops.aten.sym_size.int(nonzero, 0);  nonzero = None
    gt = sym_size_int > 3;  sym_size_int = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 0)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x_1, sym_size_int_1));  gt = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = None
    getitem = cond[0];  cond = None
    return getitem""",
        )

    def _check_closure_correctly_lifted(self, f, *, args, exp_res, exp_arg_num):
        # [FUNCTION DOC] HELPER: raises AssertionError unless args is a tuple or list; checks f
        #   gives exp_res eagerly and through a make_fx trace, and that every child submodule of
        #   the traced graph has exactly exp_arg_num placeholder nodes.
        if not isinstance(args, (tuple, list)):
            raise AssertionError(f"Expected args to be tuple or list, got {type(args)}")
        self.assertEqual(f(*args), exp_res)
        gm = make_fx(f)(*args)
        self.assertEqual(gm(*args), exp_res)

        def cnt_placeholder(gm):
            # [FUNCTION DOC] HELPER: returns the number of placeholder nodes in the graph module.
            return len([node for node in gm.graph.nodes if node.op == "placeholder"])

        placeholder_cnts = [cnt_placeholder(mod) for mod in gm.children()]
        self.assertTrue(all(cnt == exp_arg_num for cnt in placeholder_cnts))

    def _check_closure_correctly_lifted_with_mutation(
        self, f, closures_to_be_mutated, *, args, exp_arg_num
    ):
        # [FUNCTION DOC] HELPER: checks closure lifting for the current results, then
        #   calls add with minus one on each closure tensor and checks again with the
        #   recomputed expected value.
        # KNOWN ISSUE: add is out-of-place and its result is discarded, so the closures are never
        #   actually mutated; the second check repeats the first.
        exp_res = f(*args)
        self._check_closure_correctly_lifted(
            f, args=args, exp_res=exp_res, exp_arg_num=exp_arg_num
        )

        for closure in closures_to_be_mutated:
            closure.add(-1)
        new_exp_res = f(*args)

        self._check_closure_correctly_lifted(
            f, args=args, exp_res=new_exp_res, exp_arg_num=exp_arg_num
        )

    def test_cond_with_tensor_closure(self):
        # [FUNCTION DOC] Cond branches close over tensors a and b; checks branches are lifted to
        #   take three arguments (x, a, b) via the closure helper.
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure a.
            return x + a

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure b.
            return x + b

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # expected branches takes [x, a, b] as input
        inp = torch.randn(2, 3)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (a, b), args=(inp,), exp_arg_num=3
        )

    def test_cond_with_tensor_closure_graph_module(self):
        # [FUNCTION DOC] Traces cond with tensor closures using symbolic make_fx allowing non-fake
        #   inputs; compares the golden outer graph (closure tensors as constants passed as
        #   operands) and the true subgraph.
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure a.
            return x + a

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure b.
            return x + b

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # expected branches takes [x, a, b] as input
        inp = torch.randn(2, 3)

        gm = make_fx(foo, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant1 = self._tensor_constant1
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, _tensor_constant0, sym_size_int_1, sym_size_int, _tensor_constant1));  eq = true_graph_0 = false_graph_0 = x_1 = _tensor_constant0 = sym_size_int_1 = sym_size_int = _tensor_constant1 = None
    getitem = cond[0];  cond = None
    return getitem""",
        )
        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    return (add,)""",
        )

    def test_cond_with_module_param_closure(self):
        # [FUNCTION DOC] Cond branches reading a parameter and a buffer from a module object;
        #   checks both branches take (x, param, buffer) via the closure helper.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a non-grad parameter of ones and a buffer of twos.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: registers the parameter and the buffer.
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.ones(2, 3) + 1)

        my_mode = Mod()

        def true_fn(x):
            # [FUNCTION DOC] HELPER: x plus the module parameter.
            return x + my_mode.param

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x plus the module buffer.
            return x + my_mode.buffer

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        # expected both branches takes (x, param, buffer)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (my_mode.param, my_mode.buffer), args=(inp,), exp_arg_num=3
        )

    def test_cond_with_module_python_scalar_closure(self):
        # [FUNCTION DOC] Branches close over a tensor and a Python int; checks that only the
        #   tensor is lifted (both branches take two arguments) and the result is input plus one.
        def foo(x):
            # [FUNCTION DOC] HELPER: builds a tensor a and an int b, defines branches using them,
            #   and runs cond on whether the leading size is 4.
            a = torch.ones(1, 1)
            b = 1

            def true_fn(x):
                # [FUNCTION DOC] HELPER: x plus tensor a.
                return x + a

            def false_fn(x):
                # [FUNCTION DOC] HELPER: x plus integer b.
                return x + b

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        res = inp + 1
        # python scalar b is not lifted as input, so both branches take (x, a)
        self._check_closure_correctly_lifted(
            foo, args=(inp,), exp_res=res, exp_arg_num=2
        )

    def test_cond_nested_with_closure(self):
        # [FUNCTION DOC] Nested cond whose inner branches close over tensors a and b; checks that
        #   the closure arguments are deduplicated so all branches take three arguments.
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure a.
            return x + a

        def inner_false_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure b.
            return x + b

        def foo(x):
            # [FUNCTION DOC] HELPER: outer cond on whether the leading size is 4 between two
            #   functions that run inner conds.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: inner cond on whether the leading size is 2.
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                # [FUNCTION DOC] HELPER: inner cond on whether the leading size exceeds 4.
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inp = torch.ones(2, 3)
        # For top-level cond, it take 3 arguments (x, a, b). Dynamo should
        # realize that the nonlocal variables are same for the true and false
        # branches, so it should de-dupe them.
        # For second-level conds, it takes (x, a, b)
        self._check_closure_correctly_lifted_with_mutation(
            foo, (a, b), args=(inp,), exp_arg_num=3
        )

    def test_cond_nested_with_closure_graph_module(self):
        # [FUNCTION DOC] Defines nested cond functions with closures but never calls them
        #   or asserts anything.
        # KNOWN ISSUE: the test has no assertions and does not execute foo, so it tests nothing.
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure a.
            return x + a

        def inner_false_fn(x):
            # [FUNCTION DOC] HELPER: x plus closure b.
            return x + b

        def foo(x):
            # [FUNCTION DOC] HELPER: outer cond (never executed) over two functions that
            #   run inner conds.
            def true_fn(x):
                # [FUNCTION DOC] HELPER: inner cond on whether the leading size is 2.
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                # [FUNCTION DOC] HELPER: inner cond on whether the leading size exceeds 4.
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

    def test_map_unfunc_boolean_tensor_for_nested_map_cond(self):
        # [FUNCTION DOC] Traces a map whose function runs cond on a boolean tensor, using a
        #   wrapper that functionalizes arguments (old style); compares the golden outer graph
        #   (map_impl with mapped and extra inputs) and the map body graph (cond).
        def map_fn(pred, x):
            # [FUNCTION DOC] HELPER: maps a function that applies cond (double or halve) over x
            #   using pred as extra argument.
            def fn(x, pred):
                # [FUNCTION DOC] HELPER: cond on pred between multiplying x by 2 and dividing
                #   it by 2.
                return control_flow.cond(pred, lambda x: x * 2, lambda x: x / 2, (x,))

            return control_flow.map(fn, x, pred)

        def f_wrapper(func):
            # [FUNCTION DOC] HELPER: decorator factory that functionalizes tensor arguments, runs
            #   the function and converts outputs back with functionalization enabled.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: enables functionalization, maps tensors to functional
                #   tensors in args and kwargs, calls the function, converts outputs back, and
                #   disables functionalization in a finally clause.
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
                finally:
                    torch._disable_functionalization()

            return wrapper

        gm = make_fx(f_wrapper(map_fn))(
            torch.tensor(True), torch.ones([2, 3], requires_grad=False)
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, pred_1, x_1):
    select_copy = torch.ops.aten.select_copy.int(x_1, 0, 0);  select_copy = None
    body_graph_0 = self.body_graph_0
    map_impl = torch.ops.higher_order.map_impl(body_graph_0, [x_1], [pred_1]);  body_graph_0 = x_1 = pred_1 = None
    getitem = map_impl[0];  map_impl = None
    return getitem""",
        )
        self.assertExpectedInline(
            gm.body_graph_0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(arg1_1, true_graph_0, false_graph_0, (arg0_1,));  arg1_1 = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )

    @skipIfCrossRef  # Arg order changes with crossref
    def test_cond_make_fx_preserve_stack_trace_for_nodes_in_subgraph(self):
        # [FUNCTION DOC] Skipped for cross-ref. Exports a cond function, runs the exported module
        #   under an interpreter with node metadata preserved while tracing with make_fx, and
        #   checks that source_fn_stack and stack_trace metadata of the add, mul, sin and cos
        #   nodes match between the exported and retraced modules.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: x plus cosine of x.
            return x + x.cos()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x times sine of x.
            return x * x.sin()

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, (x,))

        inp = torch.randn([4, 3])
        gm, _ = torch._dynamo.export(foo)(inp)

        def run_with_interpreter(*args):
            # [FUNCTION DOC] HELPER: runs the exported graph module with an fx Interpreter inside
            #   a context preserving node meta.
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(gm).run(*args)

        new_gm = make_fx(run_with_interpreter)(inp)

        checked_ops = {"add", "mul", "sin", "cos"}
        checked_meta = ["source_fn_stack", "stack_trace"]
        all_source_fns = collect_meta_for_filtered_nodes(gm, checked_ops, checked_meta)
        new_source_fns = collect_meta_for_filtered_nodes(
            new_gm, checked_ops, checked_meta
        )
        self.assertEqual(all_source_fns, new_source_fns)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "triggers cache limit for foo and changes unique_graphs count.",
    )
    def test_cond_no_dynamo_cache_limit(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Calls a compiled cond function repeatedly with
        #   fresh branch closures (recompile limit plus one times) and checks results stay
        #   correct; asserts the dynamo statistics: captured calls equal twice the iteration count
        #   and unique graphs equal the iteration count.
        torch._dynamo.reset()
        counters = torch._dynamo.utils.counters
        counters.clear()

        def foo(x, true_fn, false_fn):
            # [FUNCTION DOC] HELPER: cond on whether the sum of x is negative, with the
            #   given branch functions.
            return cond(x.sum() < 0, true_fn, false_fn, (x,))

        inp = torch.ones(3, 4)
        exp_out = inp.sin()
        iter_n = torch._dynamo.config.recompile_limit + 1

        # Need functions that cause recompilations
        def get_dummy_fns(str):
            # [FUNCTION DOC] HELPER: returns a cosine-like and a sine-like function whose closures
            #   depend on the given string, so each call fails dynamo guards.
            def dummy_cos(x):
                # [FUNCTION DOC] HELPER: cosine of x with a no-op length adjustment.
                return x.cos() + len(str) - len(str)

            def dummy_sin(x):
                # [FUNCTION DOC] HELPER: sine of x with a no-op length adjustment.
                return x.sin() + len(str) - len(str)

            return dummy_cos, dummy_sin

        for i in range(iter_n):
            # we fail guards each iter because `str(i)` is different
            self.assertEqual(foo(inp, *get_dummy_fns(str(i))), exp_out)

        # each iteration captures a cond and a getitem from the tuple output
        self.assertEqual(counters["stats"]["calls_captured"], iter_n * 2)
        self.assertEqual(counters["stats"]["unique_graphs"], iter_n)

    def test_cond_with_consecutive_make_fx_symbolic(self):
        # [FUNCTION DOC] Traces cond symbolically on four inputs of different shapes and
        #   compares golden graphs for the outer graph and each branch every time (the generated
        #   graph is identical).
        def true_fn(x):
            # [FUNCTION DOC] HELPER: x minus cosine of x.
            return x - x.cos()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: x plus sine of x.
            return x + x.sin()

        def foo(x):
            # [FUNCTION DOC] HELPER: cond on whether the leading size is 4.
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        inps = (torch.ones(3, 4), torch.ones(3, 5), torch.ones(5, 4), torch.ones(5, 3))
        for inp in inps:
            gm = make_fx(foo, tracing_mode="symbolic")(inp)
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    eq = sym_size_int == 4
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, (x_1, sym_size_int_1, sym_size_int));  eq = true_graph_0 = false_graph_0 = x_1 = sym_size_int_1 = sym_size_int = None
    getitem = cond[0];  cond = None
    return getitem""",
            )

            self.assertExpectedInline(
                gm.true_graph_0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    cos = torch.ops.aten.cos.default(arg0_1)
    sub = torch.ops.aten.sub.Tensor(arg0_1, cos);  arg0_1 = cos = None
    return (sub,)""",
            )

            self.assertExpectedInline(
                gm.false_graph_0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    sin = torch.ops.aten.sin.default(arg0_1)
    add = torch.ops.aten.add.Tensor(arg0_1, sin);  arg0_1 = sin = None
    return (add,)""",
            )

    def _create_test_fns_for_cond(
        self, pred, inner_most_fn, operands, closure_list, nested_level
    ):
        # [FUNCTION DOC] HELPER: recursive factory returning operands and a function that runs
        #   cond. At nesting level 0 the branches add or subtract the inner function applied to
        #   the operands and, when closure tensors exist, to the closure list (or just apply it);
        #   the function returns a zeros tensor when there are no operands or closures. At higher
        #   levels, branches add or subtract the next lower level function applied to its own
        #   operands, with the predicate flipped at each level using a comparison, and the empty
        #   case returns ones.
        if nested_level == 0:
            if len(closure_list) > 0:

                def true_fn(*operands):
                    # [FUNCTION DOC] HELPER: inner function of operands plus inner function
                    #   of closure tensors.
                    return inner_most_fn(*operands) + inner_most_fn(*closure_list)

                def false_fn(*operands):
                    # [FUNCTION DOC] HELPER: inner function of operands minus inner function
                    #   of closure tensors.
                    return inner_most_fn(*operands) - inner_most_fn(*closure_list)

            else:

                def true_fn(*operands):
                    # [FUNCTION DOC] HELPER: inner function of operands.
                    return inner_most_fn(*operands)

                def false_fn(*operands):
                    # [FUNCTION DOC] HELPER: inner function of operands.
                    return inner_most_fn(*operands)

            def fn(*operands):
                # [FUNCTION DOC] HELPER: returns a tensor of zeros if there are no operands and no
                #   closures, else cond on pred between the two branches.
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.zeros(1)
                return cond(pred, true_fn, false_fn, operands)

            return operands, fn
        else:
            args, inner_fn = self._create_test_fns_for_cond(
                pred <= 0, inner_most_fn, operands, closure_list, nested_level - 1
            )

            def true_fn(*operands):
                # [FUNCTION DOC] HELPER: inner function of operands plus the nested level function
                #   on its arguments.
                return inner_most_fn(*operands) + inner_fn(*args)

            def false_fn(*operands):
                # [FUNCTION DOC] HELPER: inner function of operands minus the nested level
                #   function on its arguments.
                return inner_most_fn(*operands) - inner_fn(*args)

            def fn(*operands):
                # [FUNCTION DOC] HELPER: returns a tensor of ones if there are no operands and no
                #   closures, else cond on pred between the branches.
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.ones(1)
                return cond(pred, true_fn, false_fn, operands)

            return operands, fn

    def _init_predicate(self, pred_type):
        # [FUNCTION DOC] HELPER: returns a predicate by type name: Python True, integer tensor 1,
        #   float tensor 1.0 or boolean tensor False; raises NotImplementedError for other names.
        if pred_type == "bool":
            return True
        elif pred_type == "intTensor":
            return torch.tensor(1)
        elif pred_type == "floatTensor":
            return torch.tensor(1.0)
        elif pred_type == "boolTensor":
            return torch.tensor(False)
        else:
            raise NotImplementedError

    def _init_fn(self, inner_fn_type):
        # [FUNCTION DOC] HELPER: returns the reduce function, a ReduceMod instance
        #   or a ReduceObj instance for the names function, module and object;
        #   raises NotImplementedError otherwise.
        if inner_fn_type == "function":
            return reduce_func
        elif inner_fn_type == "module":
            return ReduceMod()
        elif inner_fn_type == "object":
            return ReduceObj()
        else:
            raise NotImplementedError

    @parametrize("predType", ["bool", "intTensor", "floatTensor", "boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [0, 1])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0, 2])
    def test_cond_tracing_with_valid_inputs(
        self, predType, innerFnType, nOperands, nClosure, nesting
    ):
        # [FUNCTION DOC] Parametrized over predicate type, inner function type, operand count,
        #   closure count and nesting depth. Builds the cond test function and checks that make_fx
        #   traces in symbolic, fake and real modes (non-fake inputs allowed) reproduce the eager
        #   result, in per-mode subtests.
        pred = self._init_predicate(predType)
        inner_fn = self._init_fn(innerFnType)
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        eager_res = fn(*args)
        for tracing_mode in ["symbolic", "fake", "real"]:
            # set _allow_non_fake_inputs = True to allow fake prop through closures
            with self.subTest(tracing_mode=tracing_mode):
                gm = make_fx(
                    fn, tracing_mode=tracing_mode, _allow_non_fake_inputs=True
                )(*args)
                self.assertEqual(gm(*args), eager_res)

    @parametrize("predType", ["boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [1, 2])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0])
    def test_cond_vmap(self, predType, innerFnType, nOperands, nClosure, nesting):
        # [FUNCTION DOC] Parametrized over boolean-tensor predicates, inner function types,
        #   operand counts, closure counts and nesting 0. Checks that vmap of the cond function
        #   over operands matches eager (comparing each output element when closures exist).
        pred = self._init_predicate(predType)
        inner_fn = self._init_fn(innerFnType)
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        eager_res = fn(*args)
        out = torch.vmap(fn)(*args)
        if nClosure == 0:
            self.assertEqual(eager_res, out)
        else:
            self.assertEqual(eager_res, out[0])
            self.assertEqual(eager_res, out[1])

    def test_cond_vmap_simple(self):
        # [FUNCTION DOC] vmap over a cond whose branches are add 100 or clone and whose predicate
        #   is true; the result has the input shape and equals input plus 100.
        def fn(x):
            # [FUNCTION DOC] HELPER: cond with constant true predicate tensor, adding
            #   100 or cloning.
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + 100,
                false_fn=lambda x: x.clone(),
                operands=(x,),
            )

        a = torch.arange(15).reshape((3, 5))
        res = torch.vmap(fn, in_dims=(0,))(a)
        self.assertEqual(res.shape, (3, 5))
        self.assertEqual(res, a + 100)

    def test_cond_vmap_multiple_inputs(self):
        # [FUNCTION DOC] vmap of cond with a data-dependent predicate over two batched inputs;
        #   checks per-batch selection of x plus 100 or a clone of y against the expected matrix.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: cond on whether sum of x is less than sum of y; true gives x
            #   plus 100, false gives clone of y.
            return torch.cond(
                pred=x.sum() < y.sum(),
                true_fn=lambda x, y: x + 100,
                false_fn=lambda x, y: y.clone(),
                operands=(x, y),
            )

        a = torch.arange(15).reshape(3, 5)
        b = torch.ones_like(a) + 3
        res = torch.vmap(fn, in_dims=(0, 0))(a, b)
        expected = torch.tensor(
            [[100, 101, 102, 103, 104], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]
        )
        self.assertEqual(res.shape, (3, 5))
        self.assertEqual(expected, res)

    def test_cond_vmap_single_input_with_closure(self):
        # [FUNCTION DOC] vmap of cond with a closure tensor; run once and then again with
        #   error_on_recompile set to check no recompilation; result equals the input
        #   plus the closure.
        a = torch.ones((3, 5)) + 3
        c = torch.arange(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: cond with true predicate tensor adding or subtracting
            #   the closure c.
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + c,
                false_fn=lambda x: x - c,
                operands=(x,),
            )

        res = torch.vmap(fn, in_dims=(0,))(
            a,
        )
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            res = torch.vmap(fn, in_dims=(0,))(
                a,
            )
        self.assertEqual(a + c, res)

    def test_cond_vmap_multiple_args_with_closure(self):
        # [FUNCTION DOC] vmap of cond over two batched inputs with a false predicate and closure;
        #   result equals b minus the closure.
        a = torch.ones((3, 5), dtype=torch.int64) + 3
        b = torch.arange(15).reshape(3, 5)
        c = torch.arange(5)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: cond with false predicate tensor; true branch x plus c, false
            #   branch y minus c.
            return torch.cond(
                pred=torch.tensor([False]),
                true_fn=lambda x, y: x + c,
                false_fn=lambda x, y: y - c,
                operands=(x, y),
            )

        res = torch.vmap(fn)(a, b)
        self.assertEqual(b - c, res)

    @parametrize("nClosure", [0, 1])
    def test_cond_vmap_multiple_outputs(self, nClosure):
        # [FUNCTION DOC] Parametrized over closure usage. vmap of cond whose branches return two
        #   tensors; checks two results equal (a plus c, a minus c) or (a plus 1, a minus 1).
        if nClosure:
            c = torch.ones(5, dtype=torch.int64) + 5

            def fn(x):
                # [FUNCTION DOC] HELPER: cond with true predicate; true branch returns x plus c
                #   and x minus c; false branch two clones.
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda x: (x + c, x - c),
                    false_fn=lambda x: (x.clone(), x.clone()),
                    operands=(x,),
                )

        else:

            def fn(x):
                # [FUNCTION DOC] HELPER: cond with true predicate; true branch returns x plus 1
                #   and x minus 1; false branch two clones.
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda x: (x + 1, x - 1),
                    false_fn=lambda x: (x.clone(), x.clone()),
                    operands=(x,),
                )

        a = torch.arange(15).reshape(3, 5)
        res = torch.vmap(fn)(
            a,
        )
        self.assertEqual(len(res), 2)
        if nClosure:
            self.assertEqual(res, (a + c, a - c))
        else:
            self.assertEqual(res, (a + 1, a - 1))

    @parametrize("boolcond", [True, False])
    def test_vmap_vmap(self, boolcond):
        # [FUNCTION DOC] Parametrized over whether the predicate is a Python bool or a tensor.
        #   Nested vmap over a cond function; result equals the input plus one.
        def fn(x):
            # [FUNCTION DOC] HELPER: cond with true predicate (Python bool or tensor as selected),
            #   adding or subtracting 1.
            return torch.cond(
                pred=torch.tensor([True]) if not boolcond else True,
                true_fn=lambda x: x + 1,
                false_fn=lambda x: x - 1,
                operands=(x,),
            )

        def wrapper(x):
            # [FUNCTION DOC] HELPER: vmap of fn over x.
            return torch.vmap(fn)(x)

        a = torch.ones((3, 4, 5))
        res = torch.vmap(wrapper)(a)
        self.assertEqual(res, a + 1)

    def test_cond_trace_set__and_mutate_input(self):
        # [FUNCTION DOC] Cond branches that call set_ on an input and mutate through a view raise
        #   UncapturedHigherOrderOpError naming torch.cond.
        def f(a, tmp):
            # [FUNCTION DOC] HELPER: takes a view of a, replaces a's storage with tmp under
            #   no_grad, multiplies the view by 2 in place, and returns a plus tmp.
            a_view = a.view(-1)
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        inp = torch.ones(3, 3, requires_grad=True)
        tmp = torch.ones(3, 3, requires_grad=True)
        # graph break: torch._dynamo.exc.Unsupported: call_function DelayGraphBreakVariable() [TensorVariable()] {}
        # due to set_
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.cond",
        ):
            torch.cond(inp.sum() > 0, f, f, (inp, tmp))

    @skipIfCrossRef  # Arg order changes with crossref
    def test_cond_trace_set__and_mutate_intermediate(self):
        # [FUNCTION DOC] Skipped for cross-ref. Cond branches using set_ on a cloned intermediate:
        #   compiling with the aot_eager and inductor backends raises RuntimeError about mutating
        #   tensors with frozen storage; with the eager recording backend, compares the golden
        #   true-branch graph and checks the output equals the eager function.
        def f(a, tmp):
            # [FUNCTION DOC] HELPER: clones a and tmp, replaces a's storage with tmp under
            #   no_grad, mutates through a view, and returns a plus tmp.
            a = a.clone()
            a_view = a.view(-1)
            tmp = tmp.clone()
            with torch.no_grad():
                a.set_(tmp)
                a_view.mul_(2)
            return a + tmp

        inp = torch.ones(3, 3, requires_grad=True)
        tmp = torch.ones(3, 3, requires_grad=True)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module calling cond with f in both branches.
            def forward(self, inp: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: cond on whether the sum of inp is positive using f
                #   for both branches.
                return torch.cond(inp.sum() > 0, f, f, (inp, tmp))

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            out = torch.compile(Mod(), backend="aot_eager")(inp, tmp)

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            out = torch.compile(Mod(), backend="inductor")(inp, tmp)

        backend = EagerAndRecordGraphs()
        out = torch.compile(Mod(), backend=backend)(inp, tmp)
        self.assertExpectedInline(
            backend.graphs[0].cond_true_0.code.strip("\n"),
            """\
def forward(self, l_inp_, l_tmp_):
    l_inp__1 = l_inp_
    l_tmp__1 = l_tmp_
    a = l_inp__1.clone();  l_inp__1 = None
    a_view = a.view(-1)
    tmp = l_tmp__1.clone();  l_tmp__1 = None
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    set_ = a.set_(tmp);  set_ = None
    mul_ = a_view.mul_(2);  a_view = mul_ = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
    add = a + tmp;  a = tmp = None
    return (add,)
    """,
        )
        self.assertEqual(out, f(inp, tmp))

    @skipIfCrossRef  # Args get renamed to r in crossref mode
    @parametrize("requires_grad", [True, False])
    def test_cond_symint_operands(self, requires_grad):
        # [FUNCTION DOC] Skipped for cross-ref; parametrized over requires_grad. Compiles a module
        #   whose cond branches use a Python int attribute with dynamic shapes and the eager
        #   recording backend; checks result equality, one graph, and golden graph code in which
        #   the symbolic size is passed as an operand.
        backend = EagerAndRecordGraphs()

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding an int attribute used by the cond branches.
            def __init__(self):
                # [FUNCTION DOC] HELPER: sets the attribute num to 3.
                super().__init__()
                self.num = 3

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: cond with true predicate tensor; true branch a plus b
                #   plus num, false branch a minus b minus num.
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda a, b: a + b + self.num,
                    false_fn=lambda a, b: a - b - self.num,
                    operands=(a, b),
                )

        a = torch.ones(3, 3, requires_grad=requires_grad)
        b = torch.ones(3, 3, requires_grad=requires_grad)
        out = torch.compile(Mod(), backend=backend, dynamic=True)(a, b)
        self.assertEqual(out, Mod()(a, b))
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, s97 : torch.SymInt, L_a_ : torch.Tensor, L_b_ : torch.Tensor):
    l_a_ = L_a_
    l_b_ = L_b_
    tensor = torch.tensor([True])
    cond_true_0 = self.cond_true_0
    cond_false_0 = self.cond_false_0
    cond = torch.ops.higher_order.cond(tensor, cond_true_0, cond_false_0, (l_a_, l_b_, s97));  tensor = cond_true_0 = cond_false_0 = l_a_ = l_b_ = s97 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
        )

    def test_two_hops_not_sharing_code_obj(self):
        # [FUNCTION DOC] With automatic dynamic shapes, uses a compile counter to show that
        #   compiled cond does not recompile on a new tensor of the same shape, that while_loop
        #   causes a recompile (a separate code object), a repeat does not, and a cond with a new
        #   shape triggers a third frame.
        pred, args = torch.tensor(True), (torch.ones(3, 3),)

        def fn1(x):
            # [FUNCTION DOC] HELPER: x plus 1.
            return x + 1

        def fn2(x):
            # [FUNCTION DOC] HELPER: x minus 1.
            return x - 1

        from torch._dynamo.testing import CompileCounter

        # Tests rely on automatic_dynamic = True
        with torch._dynamo.config.patch(automatic_dynamic_shapes=True):
            cnt = CompileCounter()
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, args)
            self.assertEqual(cnt.frame_count, 1)

            args = (torch.randn(3, 3),)
            # No recompilation
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, args)
            self.assertEqual(cnt.frame_count, 1)

            def cond_fn(x):
                # [FUNCTION DOC] HELPER: true if the sum of x is positive.
                return x.sum() > 0

            args = (torch.randn(4, 4),)
            torch.compile(torch.while_loop, backend=cnt)(cond_fn, fn2, args)
            # recompilation
            self.assertEqual(cnt.frame_count, 2)

            args = (torch.randn(4, 4),)
            torch.compile(torch.while_loop, backend=cnt)(cond_fn, fn2, args)
            self.assertEqual(cnt.frame_count, 2)

            # With recompilation due to automatic dynamic
            # This also proves that while_loop doesn't share code obj with cond
            torch.compile(torch.cond, backend=cnt)(pred, fn1, fn2, (torch.randn(4, 4),))
            self.assertEqual(cnt.frame_count, 3)

    def test_hop_raises_if_not_overriding_call(self):
        # [FUNCTION DOC] Instantiating a HigherOrderOperator subclass that does not override its
        #   call hook raises TypeError mentioning the class name.
        class WrongHop(torch._ops.HigherOrderOperator):
            # [CLASS DOC] HELPER: empty HigherOrderOperator subclass.
            pass

        with self.assertRaisesRegex(TypeError, "WrongHop"):
            WrongHop("wrong_hop")

    def test_scan_functionalized(self):
        # [FUNCTION DOC] Checks scan along dim 1 with the add combine function under
        #   torch.func.functionalize equals the eager result.
        def f(init, xs):
            # [FUNCTION DOC] HELPER: scan with the add combine function along dim 1.
            return scan(get_scan_combine_fn("add", False), init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(
            functional_f(example_init, example_inputs), f(example_init, example_inputs)
        )

    def test_scan_functionalized_elem_mutation(self):
        # [FUNCTION DOC] Combine functions that mutate an element in place (first mutating
        #   the carry then the input) under functionalize make the call raise some
        #   exception (message unchecked).
        def add1(x, y):
            # [FUNCTION DOC] HELPER: adds 4 to x in place and returns two sums.
            x.add_(4)
            return x + y, x + y

        def f(init, xs):
            # [FUNCTION DOC] HELPER: scan of add1 along dim 1.
            return scan(add1, init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # RuntimeError,
            # "torch.scan might be modifying the input!",
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.TorchDynamoException,
            # "Unexpected exception when running generated GraphModule.*"
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

        def add2(x, y):
            # [FUNCTION DOC] HELPER: adds 4 to y in place and returns two sums.
            y.add_(4)
            return x + y, x + y

        def f(init, xs):
            # [FUNCTION DOC] HELPER: scan of add2 along dim 1.
            return scan(add2, init, xs, dim=1)

        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # Should be
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # RuntimeError,
            # "torch.scan might be modifying the input!",
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.TorchDynamoException,
            # "Unexpected exception when running generated GraphModule.*"
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

    def test_scan_functionalized_elem_alias(self):
        # [FUNCTION DOC] A combine function returning its carry as both outputs (aliasing) under
        #   functionalize makes scan raise some exception (message unchecked).
        def add(x, y):
            # [FUNCTION DOC] HELPER: returns x twice.
            return x, x

        def f(init, xs):
            # [FUNCTION DOC] HELPER: scan of add along dim 1.
            return scan(add, init, xs, dim=1)

        example_inputs = torch.ones(5, 7, 4)
        example_init = torch.ones(5, 4)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            # TODO: Fix this so that the HOPs show similar errors for functionalization
            # Should be
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=0
            # torch._dynamo.exc.Unsupported,
            # "Encountered aliasing during higher order op tracing for HOP.*"
            # This is the Exception with PYTORCH_TEST_WITH_DYNAMO=1
            # torch._dynamo.exc.UncapturedHigherOrderOpError,
            # r"Higher Order Operator: torch\.ops\.higher_order\.scan",
            Exception,
            ".*",
        ):
            functional_f(example_init, example_inputs)

    @skipIfTorchDynamo("Graph is not captured by backend if test with dynamo")
    def test_scan_pytree_closure(self):
        # [FUNCTION DOC] Skipped under dynamo. Scan whose combine function closes over a pytree
        #   holding a parameter dict and a buffer tuple; compiles with the eager recording
        #   backend, asserts one graph and compares golden code (closure leaves passed to the scan
        #   op as additional inputs; the cross-ref branch has identical expected text), then
        #   checks eager and compiled outputs equal the fake scan result.
        param_buffer = ({"param": torch.randn(3, 3)}, (torch.randn(3),))

        def add(carry, x):
            # [FUNCTION DOC] HELPER: carry matrix-multiplied by the closure parameter and by x
            #   plus the closure buffer; returns the result as carry and its sum as output.
            ret = (carry @ param_buffer[0]["param"]) @ x + param_buffer[1][0]
            return ret, ret.sum()

        def f(init, xs):
            # [FUNCTION DOC] HELPER: scan of add over xs.
            return scan(add, init, xs)

        init = torch.randn(4, 3)
        xs = torch.randn(3, 3, 3)

        backend = EagerAndRecordGraphs()
        eager_out = f(init, xs)
        compiled_out = torch.compile(f, backend=backend)(init, xs)
        exp_out = _fake_scan(add, init, xs)

        self.assertEqual(len(backend.graphs), 1)
        if TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor, L_add_closure_0_cell_contents_0_param_ : torch.Tensor, L_add_closure_0_cell_contents_1_0_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    l_add_closure_0_cell_contents_0_param_ = L_add_closure_0_cell_contents_0_param_
    l_add_closure_0_cell_contents_1_0_ = L_add_closure_0_cell_contents_1_0_
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [l_xs_], [l_add_closure_0_cell_contents_0_param_, l_add_closure_0_cell_contents_1_0_]);  scan_combine_fn_0 = l_init_ = l_xs_ = l_add_closure_0_cell_contents_0_param_ = l_add_closure_0_cell_contents_1_0_ = None
    carry = scan[0]
    out = scan[1];  scan = None
    return (carry, out)""",
            )
        else:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_init_ : torch.Tensor, L_xs_ : torch.Tensor, L_add_closure_0_cell_contents_0_param_ : torch.Tensor, L_add_closure_0_cell_contents_1_0_ : torch.Tensor):
    l_init_ = L_init_
    l_xs_ = L_xs_
    l_add_closure_0_cell_contents_0_param_ = L_add_closure_0_cell_contents_0_param_
    l_add_closure_0_cell_contents_1_0_ = L_add_closure_0_cell_contents_1_0_
    scan_combine_fn_0 = self.scan_combine_fn_0
    scan = torch.ops.higher_order.scan(scan_combine_fn_0, [l_init_], [l_xs_], [l_add_closure_0_cell_contents_0_param_, l_add_closure_0_cell_contents_1_0_]);  scan_combine_fn_0 = l_init_ = l_xs_ = l_add_closure_0_cell_contents_0_param_ = l_add_closure_0_cell_contents_1_0_ = None
    carry = scan[0]
    out = scan[1];  scan = None
    return (carry, out)""",
            )
        self.assertEqual(eager_out, exp_out)
        self.assertEqual(compiled_out, exp_out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_simple(self):
        # [FUNCTION DOC] With scalar output capture enabled, vmap over a function that reshapes
        #   the batched input and runs scan with a clone-carry combine function using a closure
        #   matrix; checks eager scan, compiled, and fake-scan results agree.
        x = torch.randn(3, 4, 4)
        y = torch.randn(4, 2)
        zeros = torch.zeros(2, 3)

        def combine_fn(init, xs):
            # [FUNCTION DOC] HELPER: returns a clone of the carry and the product of xs
            #   with closure y.
            return init.clone(), xs @ y

        def fn(scan_op, x, y):
            # [FUNCTION DOC] HELPER: vmap with batch dims (1, 0, none) of a function that scans
            #   over a reshaped slice.
            def inner_fn(zeros, x, y):
                # [FUNCTION DOC] HELPER: reshapes x to 2 by 2 by 4 and runs the scan op with
                #   combine_fn over it from zeros init.
                x = x.view(2, 2, 4)

                return scan_op(
                    combine_fn,
                    zeros,
                    x,
                )

            return torch.vmap(inner_fn, in_dims=(1, 0, None))(zeros, x, y)

        out = fn(scan, x, y)
        compile_out = torch.compile(fn)(scan, x, y)
        exp = fn(_fake_scan, x, y)
        self.assertEqual(out, exp)
        self.assertEqual(out, compile_out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_complex_ops(self):
        # [FUNCTION DOC] With scalar output capture enabled, vmap over both init and xs of a scan
        #   with a combine function using relu, transpose, matrix multiplication and reductions;
        #   checks eager, compiled and fake scan agree.
        # Test with various operations requiring shape reasoning
        x = torch.randn(4, 5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # [FUNCTION DOC] HELPER: new carry is relu of carry plus the transposed product of
            #   transposed xs and the closure weight; output is sum of sine of carry plus mean of
            #   cosine of xs.
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def fn(scan_op, x, init):
            # [FUNCTION DOC] HELPER: vmap over dim 0 of both arguments of a function that scans.
            def inner_fn(x, init):
                # [FUNCTION DOC] HELPER: runs the scan op with combine_fn on init and x.
                return scan_op(combine_fn, init, x)

            return torch.vmap(inner_fn, in_dims=(0, 0))(x, init)

        out = fn(scan, x, init)
        compile_out = torch.compile(fn)(scan, x, init)
        exp = fn(_fake_scan, x, init)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_unbatched_x(self):
        # [FUNCTION DOC] Same as the complex-ops vmap test but xs is not batched (in-dim none)
        #   while init is; checks eager, compiled and fake scan agree.
        # Test with various operations requiring shape reasoning
        x = torch.randn(5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # [FUNCTION DOC] HELPER: same relu / transpose / matmul combine function as the
            #   complex ops test.
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def fn(scan_op, x, init):
            # [FUNCTION DOC] HELPER: vmap with xs unbatched and init batched over dim 0.
            def inner_fn(x, init):
                # [FUNCTION DOC] HELPER: runs the scan op with combine_fn on init and x.
                return scan_op(combine_fn, init, x)

            return torch.vmap(inner_fn, in_dims=(None, 0))(x, init)

        out = fn(scan, x, init)
        compile_out = torch.compile(fn)(scan, x, init)
        exp = fn(_fake_scan, x, init)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @skipIfTorchDynamo("not a dynamo test")
    def test_scan_in_vmap_unbatched_init_error(self):
        # [FUNCTION DOC] Skipped under dynamo. vmap with xs batched but init unbatched makes scan
        #   fail with RuntimeError about a size mismatch at a non-singleton dimension.
        # Test with various operations requiring shape reasoning
        x = torch.randn(4, 5, 3, 2)
        init = torch.randn(4, 3, 2)
        weight = torch.randn(3, 3)

        def combine_fn(carry, xs):
            # [FUNCTION DOC] HELPER: same relu / transpose / matmul combine function as the
            #   complex ops test.
            # carry: (3, 2), xs: (3, 2)
            intermediate = torch.nn.functional.relu(carry)
            xs_t = xs.transpose(0, 1)  # (2, 3)
            result = xs_t @ weight  # (2, 3)
            new_carry = intermediate + result.transpose(0, 1)  # Back to (3, 2)
            output = torch.sin(carry).sum() + torch.cos(xs).mean()
            return new_carry, output

        def vmap_fn(x, init):
            # [FUNCTION DOC] HELPER: vmap with xs batched and init not batched, scanning inside.
            def fn(x, init):
                # [FUNCTION DOC] HELPER: scans with combine_fn over init and x.
                return scan(combine_fn, init, x)

            return torch.vmap(fn, in_dims=(0, None))(x, init)

        with self.assertRaisesRegex(
            RuntimeError,
            """The size of tensor a \\(4\\) must match the size of tensor b \\(2\\) at non-singleton dimension 4""",
        ):
            vmap_fn(x, init)

    @skipIfTorchDynamo("a vmap test, not a dynamo test")
    def test_vmap_closure_weight_error(self):
        # [FUNCTION DOC] Skipped under dynamo. Shows that a combine function reading a closed-over
        #   batched weight (instead of a vmapped argument) sees the unbatched full weight, so the
        #   vmapped fake scan raises RuntimeError about a shape mismatch.
        init_batched = torch.randn(7, 2, 3)
        xs_batched = torch.randn(7, 5, 4)
        weight = torch.randn(7, 4, 3)

        def combine_fn(carry, xs):
            # [FUNCTION DOC] HELPER: carry plus xs matrix-multiplied by the closure weight; output
            #   is the sum of carry.
            # carry: (2, 3), xs: (4,), weight: (4, 3)
            new_carry = carry + xs @ weight
            output = carry.sum()
            return new_carry, output

        def expected_fn(init, xs, weight):
            # [FUNCTION DOC] HELPER: vmap over dim 0 of init, xs and weight, calling
            #   an inner function.
            def fn(init, xs, weight):
                # [FUNCTION DOC] HELPER: fake scan of combine_fn on init and xs
                #   (weight argument unused).
                return _fake_scan(combine_fn, init, xs)

            return torch.vmap(fn, in_dims=(0, 0, 0))(init, xs, weight)

        # Note that even though weight is vampped but combine_fn is accessing
        # the closure weight instead of the wrapped out weight thus causing
        # a shape mismatch.
        with self.assertRaisesRegex(
            RuntimeError,
            """The size of tensor a \\(2\\) must match the size of tensor b \\(7\\) at non-singleton dimension 1""",
        ):
            expected_fn(init_batched, xs_batched, weight)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_in_vmap_mixed_batch_dims(self):
        # [FUNCTION DOC] With scalar output capture enabled, vmap with init batched on dim 2
        #   and xs on dim 1; combine function uses a scalar closure; checks eager, compiled and
        #   fake scan agree.
        init = torch.randn(8, 5, 6)
        xs_batched = torch.randn(3, 6, 5, 8)
        scale = torch.randn([])

        def combine_fn(carry, xs):
            # [FUNCTION DOC] HELPER: new carry is carry plus the scaled sum of xs; output is xs
            #   matrix-multiplied by carry.
            # carry: 8, 5
            # xs: 5, 8
            # new_carry: 8, 5
            new_carry = carry + (xs * scale).sum()
            output = xs @ carry
            return new_carry, output

        def fn(scan_op, init, xs):
            # [FUNCTION DOC] HELPER: vmap with in dims 2 and 1 of a function that scans.
            def inner_fn(init, xs):
                # [FUNCTION DOC] HELPER: runs the scan op with combine_fn.
                return scan_op(combine_fn, init, xs)

            return torch.vmap(inner_fn, in_dims=(2, 1))(init, xs)

        out = fn(scan, init, xs_batched)
        compile_out = torch.compile(fn)(scan, init, xs_batched)
        exp = fn(_fake_scan, init, xs_batched)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_vmap_scan_vmap_scan_nested(self):
        # [FUNCTION DOC] With scalar output capture enabled, nests vmap, scan, vmap, scan
        #   with batch sizes 3 and 4 and scan lengths 5 and 6; checks eager, compiled and
        #   fake scan agree.
        # Outer batch: 3, inner batch: 4, outer scan: 5, inner scan: 6
        init = torch.randn(3, 4, 2, 8)
        xs_outer = torch.randn(3, 5, 4, 6, 2)

        def fn(scan_op, init, xs):
            # [FUNCTION DOC] HELPER: builds inner and outer combine functions and applies vmap
            #   over a function that runs the outer scan.
            def inner_combine(carry, xs):
                # [FUNCTION DOC] HELPER: new carry adds xs as a column to the carry; output is the
                #   sum of carry over dim 0.
                # carry: (2, 8), xs: (2,)
                new_carry = carry + xs.unsqueeze(-1)
                output = carry.sum(dim=0)  # (8,)
                return new_carry, output

            def outer_combine(init, xs):
                # [FUNCTION DOC] HELPER: uses vmap over an inner function that runs the inner
                #   scan, then returns the carry updated by the mean of results (expanded to carry
                #   size) and the sum of the results.
                # carry: (4, 2, 8,), xs: (4, 6, 2)
                # xs has batch dimension 4 from outer vmap

                def inner_fn(init, xs):
                    # [FUNCTION DOC] HELPER: runs the inner scan and reduces its final carry and
                    #   outputs to a vector.
                    # init: (2, 8)
                    # xs: (6, 2)
                    # final_carry: (2, 8)
                    # outputs: (6, 8)
                    final_carry, outputs = scan_op(inner_combine, init, xs)
                    return (final_carry.sum(0, keepdim=True) + outputs).sum(
                        dim=0
                    )  # (8,)

                inner_results = torch.vmap(inner_fn)(init, xs)  # (4, 8)
                new_carry = init + inner_results.mean(dim=0)  # (8,)
                output = inner_results.sum(dim=0)  # (8,)
                return new_carry.expand(*init.size()), output

            def vmap_inner_fn(init, xs):
                # [FUNCTION DOC] HELPER: runs the outer scan with outer_combine.
                # init: (4, 2, 8)
                # xs: (5, 4, 6, 2)
                return scan_op(outer_combine, init, xs)

            return torch.vmap(vmap_inner_fn)(init, xs)

        out = fn(scan, init, xs_outer)
        compile_out = torch.compile(fn)(scan, init, xs_outer)
        exp = fn(_fake_scan, init, xs_outer)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_scan_vmap_scan_nested(self):
        # [FUNCTION DOC] With scalar output capture enabled, a scan whose combine function uses
        #   vmap over an inner fake scan; checks eager, compiled and fake scan agree.
        xs_outer = torch.randn(5, 3, 4, 2)
        init_outer = torch.randn(3, 8)

        def fn(scan_op, init, xs):
            # [FUNCTION DOC] HELPER: defines the inner and outer combine functions and runs the
            #   given scan op with the outer one.
            def inner_combine_fake(carry, xs):
                # [FUNCTION DOC] HELPER: new carry is carry plus the sum of xs; output is
                #   carry times 2.
                # carry: 8
                # xs: 2
                new_carry = carry + xs.sum()
                output = carry * 2
                return new_carry, output

            def outer_combine_fake(carry, xs):
                # [FUNCTION DOC] HELPER: vmaps an inner function (over carry and xs rows) that
                #   runs a fake scan; returns the next carry (stacked final carries) and the sum
                #   over the batch of summed outputs.
                # carry: 3, 8
                # xs: 3, 4, 2
                def inner_fn(carry_elem, xs_elem):
                    # [FUNCTION DOC] HELPER: runs the fake scan with inner_combine_fake and
                    #   returns the summed outputs and the final carry.
                    # carry_elem: 8
                    # xs: 4, 2
                    # final_carry: 8
                    # outputs.sum(0): 8
                    final_carry, outputs = _fake_scan(
                        inner_combine_fake, carry_elem, xs_elem
                    )
                    return outputs.sum(0), final_carry

                # result: (8,)
                # next_carry, (3, 8))
                result, next_carry = torch.vmap(inner_fn, in_dims=(0, 0))(carry, xs)
                output = result.sum(dim=0)
                return next_carry, output

            return scan_op(outer_combine_fake, init, xs)

        out = fn(scan, init_outer, xs_outer)
        compile_out = torch.compile(fn)(scan, init_outer, xs_outer)
        exp = fn(_fake_scan, init_outer, xs_outer)

        self.assertEqual(out, exp)
        self.assertEqual(compile_out, exp)

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    def test_while_loop_op_int_carry_export(self, strict, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over strict and dynamic. Exports the
        #   int-carry while_loop module (with a dynamic leading dim when dynamic) with the export
        #   check helper; for non-strict dynamic export compares the golden exported graph
        #   (while_loop with int counter and unbacked symbolic outputs).
        m, args = WHILE_LOOP_TESTS["int_carry"]
        dynamic_shapes = {"x": {0: torch.export.Dim("dim_x")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        if not strict and dynamic:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[s77, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None
        sym_size_int_1: "Sym(s77)" = torch.ops.aten.sym_size.int(x, 0)

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (0, x), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = x = None
        getitem_2: "Sym(u1)" = while_loop[0]
        ge: "Sym(u1 >= 1)" = getitem_2 >= 1
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 1 on node 'ge'");  ge = _assert_scalar_default = None
        gt_1: "Sym(u1 > 0)" = getitem_2 > 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u1 on node 'gt_1'");  gt_1 = _assert_scalar_default_1 = None
        getitem_1: "f32[s77, 3]" = while_loop[1];  while_loop = None

        add: "Sym(u1 + 1)" = getitem_2 + 1

        add_1: "f32[s77, 3]" = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = None

        lt: "Sym(u1 < s77)" = getitem_2 < sym_size_int_1;  sym_size_int_1 = None

        mul: "Sym(2*u1)" = getitem_2 * 2;  getitem_2 = None
        ones: "f32[2*u1]" = torch.ops.aten.ones.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
        return pytree.tree_unflatten((add, add_1, lt, ones), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, it_1: "Sym(u0)", x_1: "f32[s77, 3]"):
            sym_size_int_1: "Sym(s77)" = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None

            lt: "Sym(u0 < s77)" = it_1 < sym_size_int_1;  it_1 = sym_size_int_1 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, it_1: "Sym(u0)", x_1: "f32[s77, 3]"):
            clone: "f32[s77, 3]" = torch.ops.aten.clone.default(x_1);  x_1 = None
            select: "f32[3]" = torch.ops.aten.select.int(clone, 0, it_1)
            select_1: "f32[3]" = torch.ops.aten.select.int(clone, 0, it_1)
            add: "f32[3]" = torch.ops.aten.add.Tensor(select_1, it_1);  select_1 = None
            copy_: "f32[3]" = torch.ops.aten.copy_.default(select, add);  select = add = copy_ = None
            add_1: "Sym(u0 + 1)" = it_1 + 1;  it_1 = None
            return (add_1, clone)
""",
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_while_loop_op_int_carry_compile(self, dynamic, backend):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic and backend (eager or
        #   aot_eager). Compiles the int-carry module with the check helper; for the dynamic eager
        #   case without cross-ref checks one graph and compares golden readable code (while_loop
        #   with symbolic additional inputs, runtime asserts and subgraphs).
        m, args = WHILE_LOOP_TESTS["int_carry"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            and dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]"):
        l_x_ = L_x_

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (0, l_x_), (s27, s77));  cond_fn_0 = body_fn_0 = l_x_ = s27 = None
        getitem_4: "Sym(u2)" = while_loop[0]
        ge: "Sym(u2 >= 1)" = getitem_4 >= 1
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u2 >= 1 on node 'ge'");  ge = _assert_scalar_default = None
        gt_1: "Sym(u2 > 0)" = getitem_4 > 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(gt_1, "Runtime assertion failed for expression 0 < u2 on node 'gt_1'");  gt_1 = _assert_scalar_default_1 = None
        out_x: "f32[s77, s27]" = while_loop[1];  while_loop = None

        gt: "Sym(u2 > 0)" = getitem_4 > 0
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(gt, "Runtime assertion failed for expression u2 > 0 on node 'gt'");  gt = _assert_scalar_default_2 = None

        add: "Sym(u2 + 1)" = getitem_4 + 1

        add_1: "f32[s77, s27]" = getitem_4 + out_x;  out_x = None

        lt: "Sym(u2 < s77)" = getitem_4 < s77;  s77 = None

        mul: "Sym(2*u2)" = getitem_4 * 2;  getitem_4 = None
        ones: "f32[2*u2]" = torch.ones(mul);  mul = None
        return (add, add_1, lt, ones)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u0)", child: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            sym_size_int: "Sym(s77)" = torch.ops.aten.sym_size.int(child, 0)

            size = child.size();  child = size = None
            lt: "Sym(u0 < s77)" = unbacked_symint < sym_size_int;  unbacked_symint = sym_size_int = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_0: "Sym(u1)", child_1: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            sym_size_int: "Sym(s77)" = torch.ops.aten.sym_size.int(child_1, 0)

            x_clone: "f32[s77, s27]" = child_1.clone()

            ge: "Sym(u1 >= 0)" = unbacked_symint_0 >= 0
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u1 >= 0 on node 'ge'");  ge = _assert_scalar_default = None

            size = child_1.size();  child_1 = size = None
            lt: "Sym(u1 < s77)" = unbacked_symint_0 < sym_size_int;  sym_size_int = None
            _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(lt, "Runtime assertion failed for expression u1 < s77 on node 'lt'");  lt = _assert_scalar_default_1 = None

            select: "f32[s27]" = x_clone.select(0, unbacked_symint_0)
            select_1: "f32[s27]" = x_clone.select(0, unbacked_symint_0)
            add: "f32[s27]" = select_1 + unbacked_symint_0;  select_1 = None
            copy_: "f32[s27]" = select.copy_(add);  select = add = copy_ = None

            add_1: "Sym(u1 + 1)" = unbacked_symint_0 + 1;  unbacked_symint_0 = None
            return (add_1, x_clone)
""",
            )

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_op_constant_and_symint_output_export(self, strict, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over strict and dynamic; scalar output
        #   capture enabled. Exports the const-and-symint-output module; for non-strict, non-
        #   dynamic export compares the golden exported graph (item call, while_loop with eight
        #   carries and subgraphs).
        m, args = WHILE_LOOP_TESTS["const_and_symint_output"]
        dynamic_shapes = {"t": {0: torch.export.Dim("dim_t")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        # strict or dynamic gives a slightly different graph
        if not strict and not dynamic:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, t):
        t: "f32[2, 3]";

        t, = fx_pytree.tree_flatten_spec(([t], {}), self._in_spec)
        _guards_fn = self._guards_fn(t);  _guards_fn = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(t)
        _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(sum_1, dtype = torch.float32, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default = None
        to: "i64[]" = torch.ops.aten.to.dtype(sum_1, torch.int64);  sum_1 = None
        item: "Sym(u0)" = torch.ops.aten.item.default(to);  to = None
        sin: "f32[2, 3]" = torch.ops.aten.sin.default(t)

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (2, 3, 1, 1, 1, 3, item, sin), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = item = sin = None
        getitem_8: "Sym(u8)" = while_loop[0]
        getitem_9: "Sym(u9)" = while_loop[1]
        getitem_10: "Sym(u10)" = while_loop[2]
        getitem_11: "Sym(u11)" = while_loop[3]
        getitem_12: "Sym(u12)" = while_loop[4]
        getitem_13: "Sym(u13)" = while_loop[5]
        getitem_14: "Sym(u14)" = while_loop[6]
        getitem_7: "f32[2, 3]" = while_loop[7];  while_loop = None

        add: "Sym(u8 + 1)" = getitem_8 + 1
        add_1: "Sym(u9 + 1)" = getitem_9 + 1
        add_2: "Sym(u10 + 1)" = getitem_10 + 1
        add_3: "Sym(u11 + 1)" = getitem_11 + 1
        add_4: "Sym(u12 + 1)" = getitem_12 + 1
        add_5: "Sym(u13 + 1)" = getitem_13 + 1
        add_6: "Sym(u14 + 1)" = getitem_14 + 1
        add_7: "f32[2, 3]" = torch.ops.aten.add.Tensor(getitem_7, 1)

        add_8: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_8);  getitem_8 = None
        add_9: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_9);  getitem_9 = None
        add_10: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_10);  getitem_10 = None
        add_11: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_11);  getitem_11 = None
        add_12: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_12);  getitem_12 = None
        add_13: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_13);  getitem_13 = None
        add_14: "f32[2, 3]" = torch.ops.aten.add.Tensor(t, getitem_14);  getitem_14 = None
        add_15: "f32[2, 3]" = torch.ops.aten.add.Tensor(getitem_7, t);  getitem_7 = t = None
        return pytree.tree_unflatten((add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, a_1: "Sym(u1)", b_1: "Sym(u2)", c1_1: "Sym(u3)", c2_1: "Sym(u4)", c3_1: "Sym(u5)", c0_1: "Sym(u6)", u0_1: "Sym(u7)", x_1: "f32[2, 3]"):
            mul: "Sym(u3*u4)" = c1_1 * c2_1;  c1_1 = c2_1 = None
            mul_1: "Sym(u3*u4*u5)" = mul * c3_1;  mul = c3_1 = None
            mul_2: "Sym(u1*u2)" = a_1 * b_1;  a_1 = b_1 = None
            lt: "Sym(u3*u4*u5 < u1*u2)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, a_1: "Sym(u1)", b_1: "Sym(u2)", c1_1: "Sym(u3)", c2_1: "Sym(u4)", c3_1: "Sym(u5)", c0_1: "Sym(u6)", u0_1: "Sym(u7)", x_1: "f32[2, 3]"):
            add: "Sym(u7 + 1)" = u0_1 + 1;  u0_1 = None
            add_1: "f32[2, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
            return (b_1, c1_1, c2_1, c3_1, a_1, 0, add, add_1)
""",
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_op_constant_and_symint_output_compile(self, dynamic, backend):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic and backend; scalar
        #   output capture enabled. Compiles the const-and-symint-output module; for the static
        #   eager case without cross-ref checks one graph and compares the golden readable code.
        m, args = WHILE_LOOP_TESTS["const_and_symint_output"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            # cross ref or dynamic gives a slightly different graph
            and not dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_t_: "f32[2, 3]"):
        l_t_ = L_t_

        sum_1: "f32[]" = l_t_.sum()
        to: "i64[]" = sum_1.to(torch.int64);  sum_1 = None
        item: "Sym(u0)" = to.item();  to = None
        sin: "f32[2, 3]" = l_t_.sin()

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (2, 3, 1, 1, 1, 3, item, sin), ());  cond_fn_0 = body_fn_0 = item = sin = None
        getitem_8: "Sym(u15)" = while_loop[0]
        getitem_9: "Sym(u16)" = while_loop[1]
        getitem_10: "Sym(u17)" = while_loop[2]
        getitem_11: "Sym(u18)" = while_loop[3]
        getitem_12: "Sym(u19)" = while_loop[4]
        getitem_13: "Sym(u20)" = while_loop[5]
        getitem_14: "Sym(u21)" = while_loop[6]
        getitem_7: "f32[2, 3]" = while_loop[7];  while_loop = None

        add: "Sym(u15 + 1)" = getitem_8 + 1
        add_1: "Sym(u16 + 1)" = getitem_9 + 1
        add_2: "Sym(u17 + 1)" = getitem_10 + 1
        add_3: "Sym(u18 + 1)" = getitem_11 + 1
        add_4: "Sym(u19 + 1)" = getitem_12 + 1
        add_5: "Sym(u20 + 1)" = getitem_13 + 1
        add_6: "Sym(u21 + 1)" = getitem_14 + 1
        add_7: "f32[2, 3]" = getitem_7 + 1

        add_8: "f32[2, 3]" = getitem_8 + l_t_;  getitem_8 = None
        add_9: "f32[2, 3]" = getitem_9 + l_t_;  getitem_9 = None
        add_10: "f32[2, 3]" = getitem_10 + l_t_;  getitem_10 = None
        add_11: "f32[2, 3]" = getitem_11 + l_t_;  getitem_11 = None
        add_12: "f32[2, 3]" = getitem_12 + l_t_;  getitem_12 = None
        add_13: "f32[2, 3]" = getitem_13 + l_t_;  getitem_13 = None
        add_14: "f32[2, 3]" = getitem_14 + l_t_;  getitem_14 = None
        add_15: "f32[2, 3]" = getitem_7 + l_t_;  getitem_7 = l_t_ = None
        return (add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u1)", unbacked_symint_0: "Sym(u2)", unbacked_symint_1: "Sym(u3)", unbacked_symint_2: "Sym(u4)", unbacked_symint_3: "Sym(u5)", unbacked_symint_4: "Sym(u6)", unbacked_symint_5: "Sym(u7)", child: "f32[2, 3]"):
            mul: "Sym(u3*u4)" = unbacked_symint_1 * unbacked_symint_2;  unbacked_symint_1 = unbacked_symint_2 = None
            mul_1: "Sym(u3*u4*u5)" = mul * unbacked_symint_3;  mul = unbacked_symint_3 = None
            mul_2: "Sym(u1*u2)" = unbacked_symint * unbacked_symint_0;  unbacked_symint = unbacked_symint_0 = None
            lt: "Sym(u3*u4*u5 < u1*u2)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_6: "Sym(u8)", unbacked_symint_7: "Sym(u9)", unbacked_symint_8: "Sym(u10)", unbacked_symint_9: "Sym(u11)", unbacked_symint_10: "Sym(u12)", unbacked_symint_11: "Sym(u13)", unbacked_symint_12: "Sym(u14)", child_1: "f32[2, 3]"):
            add: "Sym(u14 + 1)" = unbacked_symint_12 + 1;  unbacked_symint_12 = None
            child: "f32[2, 3]" = child_1 + 1;  child_1 = None
            return (unbacked_symint_7, unbacked_symint_8, unbacked_symint_9, unbacked_symint_10, unbacked_symint_6, 0, add, child)
""",
            )

    @skipIfTorchDynamo("Skip because we're testing export")
    @parametrize("strict", [True, False])
    @parametrize("dynamic", [True, False])
    def test_while_loop_op_pytree_int_carry_export(self, strict, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over strict and dynamic. Exports the
        #   pytree-int-carry module; for strict dynamic export without cross-ref compares the
        #   golden exported graph.
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        dynamic_shapes = {"x": {0: torch.export.Dim("dim_x")}} if dynamic else None
        ep = self._check_export(m, args, strict=strict, dynamic_shapes=dynamic_shapes)
        if strict and dynamic and not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[s6, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None
        sym_size_int_1: "Sym(s6)" = torch.ops.aten.sym_size.int(x, 0)

        sin: "f32[s6, 3]" = torch.ops.aten.sin.default(x);  x = None

        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (sym_size_int_1, 3, 2, 2, 3, sin), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = sym_size_int_1 = sin = None
        getitem_6: "Sym(u10)" = while_loop[0]
        getitem_7: "Sym(u11)" = while_loop[1]
        getitem_8: "Sym(u12)" = while_loop[2]
        getitem_9: "Sym(u13)" = while_loop[3]
        getitem_10: "Sym(u14)" = while_loop[4]
        getitem_5: "f32[s6, 3]" = while_loop[5];  while_loop = None

        add: "Sym(u12 + 1)" = getitem_8 + 1
        add_1: "Sym(u13 + 1)" = getitem_9 + 1
        add_2: "Sym(u14 + 1)" = getitem_10 + 1

        add_3: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_8);  getitem_8 = None
        add_4: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_9);  getitem_9 = None
        add_5: "f32[s6, 3]" = torch.ops.aten.add.Tensor(getitem_5, getitem_10);  getitem_10 = None
        return pytree.tree_unflatten((getitem_6, getitem_7, add, add_1, add_2, add_3, add_4, add_5, getitem_5), self._out_spec)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "Sym(u15)", arg1_1: "Sym(u16)", arg2_1: "Sym(u17)", arg3_1: "Sym(u18)", arg4_1: "Sym(u19)", arg5_1: "f32[s6, 3]"):
            mul: "Sym(u17*u18)" = arg2_1 * arg3_1;  arg2_1 = arg3_1 = None
            mul_1: "Sym(u17*u18*u19)" = mul * arg4_1;  mul = arg4_1 = None
            mul_2: "Sym(u15*u16)" = arg0_1 * arg1_1;  arg0_1 = arg1_1 = None
            lt: "Sym(u17*u18*u19 < u15*u16)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "Sym(u15)", arg1_1: "Sym(u16)", arg2_1: "Sym(u17)", arg3_1: "Sym(u18)", arg4_1: "Sym(u19)", arg5_1: "f32[s6, 3]"):
            add: "Sym(u15 + 1)" = arg0_1 + 1;  arg0_1 = None
            add_1: "Sym(u16 + 1)" = arg1_1 + 1;  arg1_1 = None

            add_2: "Sym(u17 + 1)" = arg2_1 + 1;  arg2_1 = None
            add_3: "Sym(u18 + 1)" = arg3_1 + 1;  arg3_1 = None
            add_4: "Sym(u19 + 1)" = arg4_1 + 1;  arg4_1 = None

            add_5: "f32[s6, 3]" = torch.ops.aten.add.Tensor(arg5_1, 1);  arg5_1 = None
            return (add, add_1, add_2, add_3, add_4, add_5)
""",
            )

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_while_loop_op_pytree_int_carry_compile(self, dynamic, backend):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic and backend. Compiles the
        #   pytree-int-carry module; for the dynamic eager case without cross-ref checks one graph
        #   and compares golden readable code.
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        if backend == "eager":
            backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=dynamic, backend=backend)
        if (
            isinstance(backend, EagerAndRecordGraphs)
            and dynamic
            and not TEST_WITH_CROSSREF
        ):
            self.assertEqual(len(backend.graphs), 1)
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(s27)", L_x_: "f32[s77, s27]"):
        l_x_ = L_x_

        child: "f32[s77, s27]" = l_x_.sin();  l_x_ = None

        cond_fn_0 = self.cond_fn_0
        body_fn_0 = self.body_fn_0
        while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (s77, s27, 2, 2, 3, child), (s27, s77));  cond_fn_0 = body_fn_0 = s77 = s27 = child = None
        getitem_10: "Sym(u10)" = while_loop[0]
        getitem_11: "Sym(u11)" = while_loop[1]
        getitem_12: "Sym(u12)" = while_loop[2]
        getitem_13: "Sym(u13)" = while_loop[3]
        getitem_14: "Sym(u14)" = while_loop[4]
        out_x: "f32[s77, s27]" = while_loop[5];  while_loop = None

        add: "Sym(u12 + 1)" = getitem_12 + 1
        add_1: "Sym(u13 + 1)" = getitem_13 + 1
        add_2: "Sym(u14 + 1)" = getitem_14 + 1

        add_3: "f32[s77, s27]" = getitem_12 + out_x;  getitem_12 = None
        add_4: "f32[s77, s27]" = getitem_13 + out_x;  getitem_13 = None
        add_5: "f32[s77, s27]" = getitem_14 + out_x;  getitem_14 = None
        return (getitem_10, getitem_11, add, add_1, add_2, add_3, add_4, add_5, out_x)

    class cond_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint: "Sym(u0)", unbacked_symint_0: "Sym(u1)", unbacked_symint_1: "Sym(u2)", unbacked_symint_2: "Sym(u3)", unbacked_symint_3: "Sym(u4)", child_1: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            mul: "Sym(u2*u3)" = unbacked_symint_1 * unbacked_symint_2;  unbacked_symint_1 = unbacked_symint_2 = None
            mul_1: "Sym(u2*u3*u4)" = mul * unbacked_symint_3;  mul = unbacked_symint_3 = None
            mul_2: "Sym(u0*u1)" = unbacked_symint * unbacked_symint_0;  unbacked_symint = unbacked_symint_0 = None
            lt: "Sym(u2*u3*u4 < u0*u1)" = mul_1 < mul_2;  mul_1 = mul_2 = None
            return lt

    class body_fn_0(torch.nn.Module):
        def forward(self, unbacked_symint_4: "Sym(u5)", unbacked_symint_5: "Sym(u6)", unbacked_symint_6: "Sym(u7)", unbacked_symint_7: "Sym(u8)", unbacked_symint_8: "Sym(u9)", child_2: "f32[s77, s27]", s27: "Sym(s27)", s77: "Sym(s77)"):
            s27_1 = s27
            s77_1 = s77

            add: "Sym(u5 + 1)" = unbacked_symint_4 + 1;  unbacked_symint_4 = None
            add_1: "Sym(u6 + 1)" = unbacked_symint_5 + 1;  unbacked_symint_5 = None

            add_2: "Sym(u7 + 1)" = unbacked_symint_6 + 1;  unbacked_symint_6 = None
            add_3: "Sym(u8 + 1)" = unbacked_symint_7 + 1;  unbacked_symint_7 = None
            add_4: "Sym(u9 + 1)" = unbacked_symint_8 + 1;  unbacked_symint_8 = None

            child: "f32[s77, s27]" = child_2 + 1;  child_2 = None
            return (add, add_1, add_2, add_3, add_4, child)
""",
            )

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_compile_while_loop_stack_output(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic and backend. Compiles a module that calls the
        #   while_loop_stack_output higher-order op with a linear body and a counter; checks it
        #   returns two tensors stacked over three iterations and matches the eager module.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding a 3 by 3 linear layer, used with the stack-
            #   output while loop.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs the while_loop_stack_output op with a counter below
                #   the leading size and a body applying the linear layer; returns the stacked
                #   counter and stacked tensor outputs.
                c = torch.tensor(0, dtype=torch.int64)

                def cond_fn(c, x):
                    # [FUNCTION DOC] HELPER: true while the counter is less than the leading
                    #   size of x.
                    return c < x.size(0)

                def body_fn(c, x):
                    # [FUNCTION DOC] HELPER: returns counter plus one and the linear layer
                    #   applied to x.
                    return c + 1, self.linear(x)

                stacked_c, stacked_x = torch.ops.higher_order.while_loop_stack_output(
                    cond_fn, body_fn, (c, x), tuple()
                )
                return stacked_c, stacked_x

        x = torch.randn(3, 3)
        mod = Mod()
        compiled_out = torch.compile(mod, backend=backend, dynamic=dynamic)(x)
        self.assertEqual(len(compiled_out), 2)
        self.assertEqual(compiled_out[0].size(0), 3)
        self.assertEqual(compiled_out[1].size(0), 3)
        self.assertEqual(compiled_out, mod(x))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_autograd_simple(self):
        # [FUNCTION DOC] With scalar output capture enabled, compiles with the AOT recording
        #   backend a module doing a while_loop (while the sum is below 2, update x using a square
        #   and a linear layer), runs backward, compares output, parameters and gradients with an
        #   eager Python-loop module with copied weights; checks the forward graph has one stack-
        #   output while_loop node and the backward graph one while_loop node, and (without cross-
        #   ref) compares golden forward and backward graphs.
        backend = torch._dynamo.testing.AotEagerAndRecordGraphs()

        class ModEager(torch.nn.Module):
            # [CLASS DOC] HELPER: eager reference module using a Python while loop with the same
            #   update and a linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: while the sum of x is below 2, replaces x by x squared
                #   plus one plus linear of x; returns x.
                while x.sum() < 2:
                    x = x * x + 1 + self.linear(x)
                return x

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module doing the same computation with the
            #   while_loop higher-order op.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: defines the condition and body and runs the while_loop
                #   op on x.
                def cond_fn(x):
                    # [FUNCTION DOC] HELPER: true while the sum of x is below 2.
                    return x.sum() < 2

                def body_fn(x):
                    # [FUNCTION DOC] HELPER: returns x squared plus one plus linear of x.
                    return x * x + 1 + self.linear(x)

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, (x,))

        x = torch.randn(3, 3, requires_grad=True)
        x_clone = x.clone()
        mod = Mod()
        mod_eager = ModEager()
        # Copy weights from mod to mod_eager
        mod_eager.load_state_dict(mod.state_dict())
        compiled_out = torch.compile(mod, backend=backend, fullgraph=True)(x)
        exp_out = mod_eager(x_clone)
        compiled_out.sum().backward()
        exp_out.sum().backward()
        self.assertEqual(compiled_out, exp_out)
        eager_parameters = dict(mod_eager.named_parameters())
        compiled_parameters = dict(mod.named_parameters())
        for name, param in compiled_parameters.items():
            self.assertEqual(param, eager_parameters[name])
            self.assertEqual(param.grad, eager_parameters[name].grad)

        self.assertEqual(
            len(
                backend.fw_graphs[0].graph.find_nodes(
                    op="call_function",
                    target=torch.ops.higher_order.while_loop_stack_output,
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                backend.bw_graphs[0].graph.find_nodes(
                    op="call_function", target=torch.ops.higher_order.while_loop
                )
            ),
            1,
        )
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]"):
        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop_stack_output = torch.ops.higher_order.while_loop_stack_output(while_loop_cond_graph_0, while_loop_body_graph_0, (primals_1,), (primals_3, primals_2));  while_loop_cond_graph_0 = while_loop_body_graph_0 = None
        getitem: "f32[u2, 3, 3]" = while_loop_stack_output[0];  while_loop_stack_output = None
        select: "f32[3, 3]" = torch.ops.aten.select.int(getitem, 0, -1)

        unsqueeze: "f32[1, 3, 3]" = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        slice_1: "f32[u2 - 1, 3, 3]" = torch.ops.aten.slice.Tensor(getitem, 0, 0, -1);  getitem = None
        cat: "f32[u2, 3, 3]" = torch.ops.aten.cat.default([unsqueeze, slice_1]);  unsqueeze = slice_1 = None
        return (select, primals_2, primals_3, cat)

    class while_loop_cond_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3]", arg2_1: "f32[3, 3]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            lt: "b8[]" = torch.ops.aten.lt.Scalar(sum_1, 2);  sum_1 = None
            return lt

    class while_loop_body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3]", arg2_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, 1);  mul = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg2_1);  arg2_1 = None
            addmm: "f32[3, 3]" = torch.ops.aten.addmm.default(arg1_1, arg0_1, t);  arg1_1 = arg0_1 = t = None
            add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, addmm);  add = addmm = None
            return (add_1,)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[3, 3]", primals_3: "f32[3]", cat: "f32[u2, 3, 3]", tangents_1: "f32[3, 3]"):
        clone: "f32[3, 3]" = torch.ops.aten.clone.default(tangents_1, memory_format = torch.contiguous_format);  tangents_1 = None
        zeros_like: "f32[3]" = torch.ops.aten.zeros_like.default(primals_3, pin_memory = False)
        zeros_like_1: "f32[3, 3]" = torch.ops.aten.zeros_like.default(primals_2, pin_memory = False)
        while_loop_cond_graph_1 = self.while_loop_cond_graph_1
        while_loop_body_graph_1 = self.while_loop_body_graph_1
        zeros: "i64[]" = torch.ops.aten.zeros.default([], dtype = torch.int64, device = device(type='cpu'), pin_memory = False)
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_1, while_loop_body_graph_1, (zeros, clone, zeros_like, zeros_like_1), (cat, primals_3, primals_2));  while_loop_cond_graph_1 = while_loop_body_graph_1 = zeros = clone = zeros_like = zeros_like_1 = cat = primals_3 = primals_2 = None
        getitem_2: "f32[3, 3]" = while_loop[1]
        getitem_3: "f32[3]" = while_loop[2]
        getitem_4: "f32[3, 3]" = while_loop[3];  while_loop = None
        return (getitem_2, getitem_4, getitem_3)

    class while_loop_cond_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[3, 3]", arg2_1: "f32[3]", arg3_1: "f32[3, 3]", arg4_1: "f32[u2, 3, 3]", arg5_1: "f32[3]", arg6_1: "f32[3, 3]"):
            sym_size_int_1: "Sym(u2)" = torch.ops.aten.sym_size.int(arg4_1, 0);  arg4_1 = None

            lt: "b8[]" = torch.ops.aten.lt.Scalar(arg0_1, sym_size_int_1);  arg0_1 = sym_size_int_1 = None
            return lt

    class while_loop_body_graph_1(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[3, 3]", arg2_1: "f32[3]", arg3_1: "f32[3, 3]", arg4_1: "f32[u2, 3, 3]", arg5_1: "f32[3]", arg6_1: "f32[3, 3]"):
            sym_size_int_1: "Sym(u2)" = torch.ops.aten.sym_size.int(arg4_1, 0)

            rsub: "i64[]" = torch.ops.aten.rsub.Scalar(arg0_1, sym_size_int_1);  sym_size_int_1 = None
            sub_1: "i64[]" = torch.ops.aten.sub.Tensor(rsub, 1);  rsub = None
            _local_scalar_dense: "Sym(u7)" = torch.ops.aten._local_scalar_dense.default(sub_1);  sub_1 = None
            select: "f32[3, 3]" = torch.ops.aten.select.int(arg4_1, 0, _local_scalar_dense);  arg4_1 = _local_scalar_dense = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg6_1);  arg6_1 = None
            t_1: "f32[3, 3]" = torch.ops.aten.t.default(t);  t = None
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg1_1, t_1);  t_1 = None
            t_2: "f32[3, 3]" = torch.ops.aten.t.default(arg1_1)
            mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t_2, select);  t_2 = None
            t_3: "f32[3, 3]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
            sum_1: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0], True)
            view: "f32[3]" = torch.ops.aten.view.default(sum_1, [3]);  sum_1 = None
            t_4: "f32[3, 3]" = torch.ops.aten.t.default(t_3);  t_3 = None
            mul_4: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg1_1, select)
            mul_5: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg1_1, select);  arg1_1 = select = None
            add_7: "f32[3, 3]" = torch.ops.aten.add.Tensor(mm, mul_5);  mm = mul_5 = None
            add_8: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_7, mul_4);  add_7 = mul_4 = None
            add_9: "i64[]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            add_10: "f32[3]" = torch.ops.aten.add.Tensor(view, arg2_1);  view = arg2_1 = None
            add_11: "f32[3, 3]" = torch.ops.aten.add.Tensor(t_4, arg3_1);  t_4 = arg3_1 = None
            return (add_9, add_8, add_10, add_11)
""",
            )

    def test_input_output_alias(self):
        # [FUNCTION DOC] For each aliasing function in the shared list, compiling a cond
        #   whose branches return an alias of the input raises
        #   UncapturedHigherOrderOpError naming torch.cond.
        def fn(f, *args):
            # [FUNCTION DOC] HELPER: cond on whether the sum of the first argument is positive,
            #   using the given function in both branches.
            return torch.cond(args[0].sum() > 0, f, f, args)

        x = torch.randn(2, 2)
        for f in ALIAS_FN:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(f, x)

    def test_input_input_alias(self):
        # [FUNCTION DOC] Passes an operand and a view of it to cond (skipping the identity
        #   function, which cond dedupes); compiling raises UncapturedHigherOrderOpError naming
        #   torch.cond for each view function.
        def fn(view_f, arg):
            # [FUNCTION DOC] HELPER: cond on the sign of the sum of arg with operands arg and a
            #   view of arg.
            def f(arg1, arg2):
                # [FUNCTION DOC] HELPER: returns cosine of the first and sine of
                #   the second argument.
                return arg1.cos(), arg2.sin()

            return torch.cond(arg.sum() > 0, f, f, (arg, view_f(arg)))

        x = torch.randn(2, 2)
        # ALIAS_FN[0] is an identical function, cond optimizes the duplication
        # as a result of auto lifting.
        for view_f in ALIAS_FN[1:]:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(view_f, x)

    @parametrize("inference_mode", [True, False])
    def test_input_mutation(self, inference_mode):
        # [FUNCTION DOC] Parametrized over inference_mode. For each aliasing function, a cond
        #   branch that mutates a view of its input in place raises UncapturedHigherOrderOpError
        #   naming torch.cond when compiled; under inference mode compilation succeeds, otherwise
        #   it raises the same error.
        def fn(view_f, *args):
            # [FUNCTION DOC] HELPER: cond on the sign of the sum with the same mutating branch
            #   in both sides.
            def mutate_f(x):
                # [FUNCTION DOC] HELPER: applies the view function to x, adds 1 in place and
                #   returns the sine of it.
                v = view_f(x)
                v.add_(1)
                return v.sin()

            return torch.cond(args[0].sum() > 0, mutate_f, mutate_f, args)

        x = torch.randn(2, 2)
        for f in ALIAS_FN:
            with self.assertRaisesRegex(
                # Should be
                # torch._dynamo.exc.Unsupported,
                # "Encountered aliasing during higher order op tracing for HOP.*"
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.cond",
            ):
                torch.compile(fn)(f, x)

            with torch.inference_mode(inference_mode):
                if not inference_mode:
                    with self.assertRaisesRegex(
                        # Should be
                        # torch._dynamo.exc.Unsupported,
                        # "Encountered aliasing during higher order op tracing for HOP.*"
                        torch._dynamo.exc.UncapturedHigherOrderOpError,
                        r"Higher Order Operator: torch\.cond",
                    ):
                        torch.compile(fn)(f, x)
                else:
                    torch.compile(fn)(f, x)

    @requires_cuda
    @parametrize("device", ["cuda", "cpu"])
    def test_cond_input_mutation(self, device):
        # [FUNCTION DOC] Needs CUDA; parametrized over cuda and cpu. Under no_grad, compiled cond
        #   whose false branch mutates its input in place: with false predicate the output equals
        #   sine plus 2 and is not the input object; with true predicate output is input plus 1.
        predicate_true = torch.tensor(True, device=device)
        predicate_false = torch.tensor(False, device=device)
        org_data = torch.ones(2, 2, device=device)

        def fn(predicate, data):
            # [FUNCTION DOC] HELPER: cond with branches x plus 1 and an in-place sine then add
            #   2 on x.
            return torch.cond(
                predicate, lambda x: x + 1, lambda x: x.sin_().add_(2), [data]
            )

        with torch.no_grad():
            expected = org_data.sin() + 2
            data = org_data.clone()
            output = torch.compile(fn)(predicate_false, data)
            self.assertEqual(output, expected)
            self.assertIsNot(output, data)

            data = org_data.clone()
            output = torch.compile(fn)(predicate_true, data)
            self.assertEqual(output, org_data + 1)

    @skipIfTorchDynamo("Graph is not captured correctly when test with dynamo")
    def test_while_loop_unbacked_bindings(self):
        # [FUNCTION DOC] Skipped under dynamo. Compiles the pytree-int-carry module
        #   dynamically and asserts one graph with a single while_loop node whose metadata
        #   records five unbacked bindings.
        m, args = WHILE_LOOP_TESTS["pytree_int_carry"]
        backend = EagerAndRecordGraphs()
        self._check_compile(m, args, dynamic=True, backend=backend)
        self.assertEqual(len(backend.graphs), 1)
        while_loop_nodes = backend.graphs[0].graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.while_loop
        )
        self.assertEqual(len(while_loop_nodes), 1)
        self.assertEqual(len(while_loop_nodes[0].meta.get("unbacked_bindings")), 5)

    # Return the .module() graph str result of non-strict export
    def _check_export_ret_graph_str(self, fn, args, dynamic_shapes=None) -> str:
        # [FUNCTION DOC] HELPER: exports fn strictly and non-strictly with the given dynamic
        #   shapes, asserts both exported modules give the eager result on the args, and returns
        #   the normalized readable graph of the non-strict export.
        strict_ep = torch.export.export(
            fn, args, dynamic_shapes=dynamic_shapes, strict=True
        )
        non_strict_ep = torch.export.export(
            fn, args, dynamic_shapes=dynamic_shapes, strict=False
        )
        eager_res = fn(*args)
        self.assertEqual(strict_ep.module()(*args), eager_res)
        self.assertEqual(non_strict_ep.module()(*args), eager_res)
        return normalize_gm(non_strict_ep.module().print_readable(print_output=False))

    @skipIfTorchDynamo("Skip because dynamo cannot trace torch.export.")
    def test_cond_eager_run_with_item(self):
        # [FUNCTION DOC] Skipped under dynamo. Exports a module whose cond branches call item on
        #   tensors, then compares the golden non-strict graph with the cond op and branch
        #   subgraphs that call item.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a cond whose branches multiply by item of one
            #   of two tensors.
            def forward(self, a, b1, b2, c):
                # [FUNCTION DOC] HELPER: cond on a, selecting c times item of b1 or
                #   b2; result doubled.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: x times the item of b1.
                    return x * b1.item()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: x times the item of b2.
                    return x * b2.item()

                r = torch.cond(a, true_fn, false_fn, (c,))
                return r * 2

        x = torch.randn(10, requires_grad=True)
        args = (
            torch.tensor(True),
            torch.tensor([3]),
            torch.tensor([4]),
            x,
        )
        model = M()
        torch.export.export(model, args, strict=True)
        graph_str = self._check_export_ret_graph_str(model, args, None)
        self.assertExpectedInline(
            graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, a, b1, b2, c):
        a: "b8[]"; b1: "i64[1]"; b2: "i64[1]"; c: "f32[10]";

        a, b1, b2, c, = fx_pytree.tree_flatten_spec(([a, b1, b2, c], {}), self._in_spec)
        _guards_fn = self._guards_fn(a, b1, b2, c);  _guards_fn = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(a, true_graph_0, false_graph_0, (c, b1, b2));  a = true_graph_0 = false_graph_0 = c = b1 = b2 = None
        getitem: "f32[10]" = cond[0];  cond = None

        mul: "f32[10]" = torch.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return pytree.tree_unflatten((mul,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self, c: "f32[10]", b1: "i64[1]", b2: "i64[1]"):
            item: "Sym(u0)" = torch.ops.aten.item.default(b1);  b1 = None

            mul: "f32[10]" = torch.ops.aten.mul.Tensor(c, item);  c = item = None
            return (mul,)

    class false_graph_0(torch.nn.Module):
        def forward(self, c: "f32[10]", b1: "i64[1]", b2: "i64[1]"):
            item: "Sym(u1)" = torch.ops.aten.item.default(b2);  b2 = None

            mul: "f32[10]" = torch.ops.aten.mul.Tensor(c, item);  c = item = None
            return (mul,)
""",
        )

    def test_cond_merge_graph_preserves_ph_meta(self):
        # [FUNCTION DOC] Compiles a module whose cond branches close over sizes and a
        #   tensor; asserts one graph and that every placeholder of the branch subgraph
        #   has example_value metadata.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward conditions on the sum of x with closures
            #   over sizes of y and z.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: records leading sizes a and b of y and z and runs cond on
                #   whether the sum of x exceeds 5.
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: x plus size a.
                    return x + a

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: x plus size b times tensor z.
                    return x + b * z

                return torch.cond(x.sum() > 5, true_fn, false_fn, (x,))

        backend = EagerAndRecordGraphs()
        _ = torch.compile(M(), backend=backend)(
            torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
        )
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        subgraph_attr = gm.graph.find_nodes(op="get_attr")[0]
        subgm = getattr(gm, subgraph_attr.target)
        for ph in subgm.graph.find_nodes(op="placeholder"):
            self.assertTrue("example_value" in ph.meta)

    @skipIfTorchDynamo("Skip because dynamo cannot trace torch.export.")
    def test_cond_symint_closure(self):
        # [FUNCTION DOC] Skipped under dynamo. Exports a module whose cond branches close over
        #   symbolic sizes of y and z, with dynamic dimensions for x, y and z, using the non-
        #   strict/strict export graph helper, and compares the golden non-strict graph where the
        #   symbolic sizes and z are passed as cond operands.
        from torch.export import Dim

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose cond branches close over sizes of y and z and over
            #   the tensor z.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: records sizes of y and z and runs cond on whether the
                #   leading size of x exceeds 5.
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: x plus size a.
                    return x + a

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: x plus size b times tensor z.
                    return x + b * z

                # When exporting with non-strict: a and b are symints,
                # so torch.compile need to wrap and trace symint inputs.
                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        args = (torch.ones(3, 3), torch.ones(5), torch.ones(3, 3))
        model = M()
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        non_strict_graph_str = self._check_export_ret_graph_str(
            model, args, dynamic_shapes
        )
        self.assertExpectedInline(
            non_strict_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x, y, z):
        x: "f32[s68, 3]"; y: "f32[s17]"; z: "f32[s68, 3]";

        x, y, z, = fx_pytree.tree_flatten_spec(([x, y, z], {}), self._in_spec)
        _guards_fn = self._guards_fn(x, y, z);  _guards_fn = None
        sym_size_int_4: "Sym(s17)" = torch.ops.aten.sym_size.int(y, 0);  y = None
        sym_size_int_5: "Sym(s68)" = torch.ops.aten.sym_size.int(z, 0)

        gt: "Sym(s68 > 5)" = sym_size_int_5 > 5

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, sym_size_int_4, sym_size_int_5, z));  gt = true_graph_0 = false_graph_0 = x = sym_size_int_4 = sym_size_int_5 = z = None
        getitem: "f32[s68, 3]" = cond[0];  cond = None
        return pytree.tree_unflatten((getitem,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self, x: "f32[s68, 3]", sym_size_int_4: "Sym(s17)", sym_size_int_5: "Sym(s68)", z: "f32[s68, 3]"):
            add: "f32[s68, 3]" = torch.ops.aten.add.Tensor(x, sym_size_int_4);  x = sym_size_int_4 = None
            return (add,)

    class false_graph_0(torch.nn.Module):
        def forward(self, x: "f32[s68, 3]", sym_size_int_4: "Sym(s17)", sym_size_int_5: "Sym(s68)", z: "f32[s68, 3]"):
            mul: "f32[s68, 3]" = torch.ops.aten.mul.Tensor(z, sym_size_int_5);  z = sym_size_int_5 = None

            add: "f32[s68, 3]" = torch.ops.aten.add.Tensor(x, mul);  x = mul = None
            return (add,)
""",
        )

    # unbacked symint inputs are created during non-strict export,
    # which causes a graph break
    @unittest.expectedFailure
    def test_cond_unbacked_symint_closure(self):
        # [FUNCTION DOC] Expected failure (per comment, unbacked symbolic ints created during
        #   non-strict export cause a graph break). Exports a module whose cond branches close
        #   over an item-derived unbacked int plus sizes, with dynamic dimensions, through the
        #   export graph helper.
        from torch.export import Dim

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose cond branches close over sizes, a tensor and an
            #   unbacked int taken from the sum of y.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: computes sizes a and b of y and z and an int c from the
                #   sum of y, then runs cond on whether the leading size of x exceeds 5.
                a = y.shape[0]
                b = z.shape[0]
                # c is an unbacked symint in non-strict export
                c = y.sum().item()

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: x plus a plus c.
                    return x + a + c

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: x plus b times z times c.
                    return x + b * z * c

                # When exporting with non-strict: a and b are symints,
                # so torch.compile need to wrap and trace symint inputs.
                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

        args = (torch.ones(3, 3), torch.ones(5, dtype=torch.int32), torch.ones(3, 3))
        model = M()
        dynamic_shapes = {"x": {0: Dim("d")}, "y": {0: Dim("d1")}, "z": {0: Dim("d")}}
        _ = self._check_export_ret_graph_str(model, args, dynamic_shapes)

    @skipIfTorchDynamo(
        "Skip because _merge_output is not intended for dynamo to compile"
    )
    def test_merge_output(self):
        # [FUNCTION DOC] Skipped under dynamo. Tests the cond output-merging helper on pairs of
        #   fake tensors built with given sizes and strides: for valid cases, the merged
        #   tensor's size and stride strings (with unbacked symbols) must equal the expected
        #   strings; for invalid cases (non-contiguous or differing layouts) the helper must
        #   raise an exception.
        from torch._higher_order_ops.cond import _merge_output
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # The shapes and strides are from raondomly generated pairs of tensors then swapaxes
        valid_test_cases = [
            # [(size1, stride1), (size2, stride2), (expected_stride, expected_size)]
            [((3,), (1,)), ((4,), (1,)), ("(u0,)", "(1,)")],
            [((1, 3), (3, 1)), ((3, 2), (2, 1)), ("(u0, u1)", "(u1, 1)")],
            [((2, 1), (1, 1)), ((7, 3), (3, 1)), ("(u0, u1)", "(u1, 1)")],
            [((5, 5), (1, 5)), ((4, 5), (1, 4)), ("(u0, 5)", "(1, u0)")],
            [
                ((7, 3, 1), (1, 7, 1)),
                ((4, 3, 3), (3, 12, 1)),
                ("(u0, 3, u1)", "(u1, u0*u1, 1)"),
            ],
            [
                ((5, 7, 4), (7, 1, 35)),
                ((7, 4, 4), (4, 1, 28)),
                ("(u0, u1, 4)", "(u1, 1, u0*u1)"),
            ],
            [
                ((1, 6, 3, 2), (36, 1, 6, 18)),
                ((4, 2, 2, 6), (24, 1, 2, 4)),
                ("(u0, u1, u2, u3)", "(u1*u2*u3, 1, u1, u1*u2)"),
            ],
            [
                ((6, 1, 6, 3), (18, 1, 1, 6)),
                ((2, 1, 3, 4), (12, 1, 1, 3)),
                ("(u0, 1, u1, u2)", "(u1*u2, 1, 1, u1)"),
            ],
            [
                ((3, 1, 2, 4, 1), (8, 8, 4, 1, 1)),
                ((2, 4, 1, 4, 1), (16, 4, 4, 1, 1)),
                ("(u0, u1, u2, 4, 1)", "(4*u1*u2, 4*u2, 4, 1, 1)"),
            ],
        ]

        def _inner(case):
            # [FUNCTION DOC] HELPER: builds a fake mode with a shape env, creates two empty
            #   strided tensors from the case sizes and strides, merges them with the helper and
            #   asserts the string forms of the merged size and stride equal the expected ones.
            fake_mode = FakeTensorMode(shape_env=ShapeEnv())

            (size1, stride1), (size2, stride2), (merged_size, merged_stride) = case
            with fake_mode:
                t1 = torch.empty_strided(size1, stride1)
                t2 = torch.empty_strided(size2, stride2)
            out = _merge_output(t1, t2, fake_mode)
            self.assertEqual(str(tuple(out.size())), merged_size)
            self.assertEqual(str(tuple(out.stride())), merged_stride)

        for case in valid_test_cases:
            _inner(case)

        # The shapes and strides are from raondomly generated pairs of tensors then swapaxes
        invalid_test_cases = [
            # [(size1, stride1), (size2, stride2)]
            [((1,), (1,)), ((1,), (0,))],
            [
                ((1, 3), (1, 1)),
                ((5, 6), (6, 1)),
            ],  # t1 is not contiguous, t2 is contiguous
            [
                ((2, 1), (1, 1)),
                ((7, 3), (1, 3)),
            ],  # t1 is contiguous, t2 is not contiguous
            [
                ((5, 4), (4, 1)),
                ((5, 5), (1, 5)),
            ],  # t1 is contiguous, t2 is not contiguous
            [((7, 3, 1), (1, 7, 1)), ((4, 3, 3), (9, 1, 3))],  # layout is different
            [((5, 7, 4), (7, 1, 35)), ((7, 4, 4), (4, 28, 1))],  # layout is different
            [
                ((1, 6, 3, 2), (36, 1, 6, 18)),
                ((4, 1, 1, 6), (1, 4, 4, 4)),
            ],  # layout is different
            [
                ((6, 1, 6, 3), (18, 1, 1, 6)),
                ((1, 1, 1, 1), (1, 1, 1, 1)),
            ],  # layout is different
            [
                ((6, 1, 1, 6, 3), (3, 18, 18, 18, 1)),
                ((5, 1, 2, 1, 1), (2, 10, 1, 10, 1)),
            ],  # layout is different
        ]
        for case in invalid_test_cases:
            with self.assertRaisesRegex(Exception, r"."):
                _inner(case)

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_cond_mismatched_branch_output(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic and backend (eager, aot_eager). Cond branches
        #   return tensors of different sizes (slices cloned to share storage offset); compiled
        #   full graph result must equal eager. For the dynamic eager case (without cross-ref) it
        #   also compares the golden graph, in which the output has an unbacked leading size with
        #   a runtime non-negativity assert.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose cond branches return slices of different lengths.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: runs cond on the sign of the sum of x with the two
                #   slicing branches and returns the sum of y minus the cond result.
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: clone of x plus size a, taken from index 2 on.
                    # clone the outputs so branches have the same storage_offset
                    return (x + a)[2:].clone()

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: clone of the first two rows of x plus b times z.
                    # clone the outputs so branches have the same storage_offset
                    return (x + b * z)[:2].clone()

                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x,))
                return y.sum() - ret

        m = M()
        x, y, z = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
        out = m(x, y, z)
        if not (backend == "eager" and dynamic and not TEST_WITH_CROSSREF):
            compiled_out = torch.compile(
                m, backend=backend, dynamic=dynamic, fullgraph=True
            )(x, y, z)
            self.assertEqual(compiled_out, out)
        else:
            bk = EagerAndRecordGraphs()
            compiled_out = torch.compile(
                m, backend=bk, dynamic=dynamic, fullgraph=True
            )(x, y, z)
            self.assertEqual(compiled_out, out)
            self.assertExpectedInline(
                normalize_gm(bk.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s17: "Sym(s17)", s94: "Sym(s94)", L_y_: "f32[s17, s94]", L_z_: "f32[s17, s94]", L_x_: "f32[s17, s94]"):
        l_y_ = L_y_
        l_z_ = L_z_
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum()
        gt: "b8[]" = sum_1 > 0;  sum_1 = None
        cond_true_0 = self.cond_true_0
        cond_false_0 = self.cond_false_0
        cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, (l_x_, s94, s17, s17, l_z_));  gt = cond_true_0 = cond_false_0 = l_x_ = s94 = s17 = l_z_ = None
        getitem_5: "f32[u0, s94]" = cond[0]
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(getitem_5, 0);  getitem_5 = None
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        ret: "f32[u0, s94]" = cond[0];  cond = None

        sum_2: "f32[]" = l_y_.sum();  l_y_ = None
        sub: "f32[u0, s94]" = sum_2 - ret;  sum_2 = ret = None
        return (sub,)

    class cond_true_0(torch.nn.Module):
        def forward(self, l_x_: "f32[s17, s94]", s94: "Sym(s94)", s17_true_branch: "Sym(s17)", getitem_2_false_branch: "Sym(s17)", l_z__false_branch: "f32[s17, s94]"):
            l_x__1 = l_x_
            s94_1 = s94

            add: "f32[s17, s94]" = l_x__1 + s17_true_branch;  l_x__1 = s17_true_branch = None
            getitem: "f32[s17 - 2, s94]" = add[slice(2, None, None)];  add = None
            clone: "f32[s17 - 2, s94]" = getitem.clone();  getitem = None
            return (clone,)

    class cond_false_0(torch.nn.Module):
        def forward(self, l_x_: "f32[s17, s94]", s94: "Sym(s94)", s17_true_branch: "Sym(s17)", getitem_2_false_branch: "Sym(s17)", l_z__false_branch: "f32[s17, s94]"):
            l_x__1 = l_x_
            s94_1 = s94

            mul: "f32[s17, s94]" = getitem_2_false_branch * l_z__false_branch;  getitem_2_false_branch = l_z__false_branch = None
            add: "f32[s17, s94]" = l_x__1 + mul;  l_x__1 = mul = None
            getitem: "f32[2, s94]" = add[slice(None, 2, None)];  add = None
            clone: "f32[2, s94]" = getitem.clone();  getitem = None
            return (clone,)
""",
            )

    @parametrize("dynamic", [True, False])
    @parametrize("backend", ["eager", "aot_eager"])
    def test_cond_mismatched_branch_strided_output(self, dynamic, backend):
        # [FUNCTION DOC] Parametrized over dynamic and backend. Branches return tensors with
        #   different shapes and strides (swapped-axis views expanded differently, plus empty
        #   strided tensors); the compiled full-graph result must equal eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose cond branches return differently strided tuples.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: cond on the sign of the sum of x, returns the sum of y
                #   plus the first output.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: returns the swapped-axes x plus one, unsqueezed and
                    #   expanded to five along a new axis, and an empty strided 3 by 3 tensor.
                    return (
                        (x.swapaxes(-1, 0) + 1)
                        .unsqueeze(1)
                        .expand(-1, 5, -1, -1, -1, -1, -1),
                        torch.empty_strided((3, 3), (0, 1)),
                    )

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: same pattern using y with expansion of four and an
                    #   empty strided 4 by 5 tensor.
                    return (
                        (y.swapaxes(-1, 0) + 1)
                        .unsqueeze(1)
                        .expand(-1, 4, -1, -1, -1, -1, -1),
                        torch.empty_strided((4, 5), (0, 1)),
                    )

                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x, y))
                return y.sum() + ret[0]

        m = M()
        x, y = torch.randn(1, 6, 1, 5, 4, 3), torch.randn(1, 4, 5, 1, 3, 8)
        out = m(x, y)
        compiled_out = torch.compile(
            m, backend=backend, dynamic=dynamic, fullgraph=True
        )(x, y)
        self.assertEqual(compiled_out, out)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_cond_branch_closure_over_int_args(self, backend):
        # [FUNCTION DOC] Parametrized over eager and aot_eager. Compiles (dynamic, full graph)
        #   functions whose zero-argument cond branches create tensors from closed-over int
        #   arguments; first with identical shapes, then (second case, eager backends only because
        #   inductor fails) with branches of different widths; results must equal eager.
        def f(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: builds a true predicate tensor and cond branches returning
            #   ones or zeros of rows by cols; returns the cond result.
            pred = torch.tensor(True)

            def true_fn():
                # [FUNCTION DOC] HELPER: returns ones of size rows by cols.
                return torch.ones(rows, cols)

            def false_fn():
                # [FUNCTION DOC] HELPER: returns zeros of size rows by cols.
                return torch.zeros(rows, cols)

            return torch.cond(pred, true_fn, false_fn)

        compiled = torch.compile(f, backend=backend, dynamic=True, fullgraph=True)
        self.assertEqual(compiled(5, 1000), f(5, 1000))

        # differing sizes still fail on inductor (#189528), hence eager/aot_eager only
        def g(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: like f but the false branch returns zeros with
            #   one extra column.
            pred = torch.tensor(True)

            def true_fn():
                # [FUNCTION DOC] HELPER: returns ones of size rows by cols.
                return torch.ones(rows, cols)

            def false_fn():
                # [FUNCTION DOC] HELPER: returns zeros of size rows by cols plus one.
                return torch.zeros(rows, cols + 1)

            return torch.cond(pred, true_fn, false_fn)

        compiled = torch.compile(g, backend=backend, dynamic=True, fullgraph=True)
        self.assertEqual(compiled(5, 7), g(5, 7))


class TestAutoFunctionalizeControlFlow(TestCase):
    # [CLASS DOC] Tests auto-functionalization of control-flow higher-order ops that mutate
    #   inputs, buffers or captured tensors: runs eager, AOT eager recording and inductor and
    #   compares golden forward graphs (auto_functionalized_v2 wrapping cond, switch, while_loop
    #   or scan). Provides check helpers.
    def check(self, gen_fn, args, device, dynamic) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: moves args to the device; runs a fresh module from the generator
        #   three times (eager under no_grad, AOT eager recording backend, inductor), each on its
        #   own cloned args with dynamo reset between compiles; asserts outputs, mutated args and
        #   all buffers match the eager run for both compiled variants; returns the first
        #   recorded forward graph.
        args = pytree.tree_map(lambda t: t.to(device=device), args)

        def _clone(args):
            # [FUNCTION DOC] HELPER: returns a list with clones of tensor arguments and
            #   other arguments unchanged.
            return [
                arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args
            ]

        def _new_fn():
            # [FUNCTION DOC] HELPER: calls the generator and, if the result is a module, moves it
            #   to the device; returns it.
            mod_or_fn = gen_fn()
            if isinstance(mod_or_fn, torch.nn.Module):
                mod_or_fn.to(device)
            return mod_or_fn

        # Only support input mutation in inference
        cloned_args = [_clone(args) for _ in range(3)]
        with torch.no_grad():
            mod0 = _new_fn()
            exp = mod0(*cloned_args[0])
        backend = AotEagerAndRecordGraphs()
        torch._dynamo.reset()
        with torch.no_grad():
            mod1 = _new_fn()
            eager_out = torch.compile(
                mod1, backend=backend, fullgraph=True, dynamic=dynamic
            )(*cloned_args[1])
        torch._dynamo.reset()
        with torch.no_grad():
            mod2 = _new_fn()
            inductor_out = torch.compile(
                mod2, backend="inductor", fullgraph=True, dynamic=dynamic
            )(*cloned_args[2])

        self.assertEqual(exp, eager_out)
        self.assertEqual(exp, inductor_out)
        self.assertEqual(cloned_args[0], cloned_args[1])
        self.assertEqual(cloned_args[0], cloned_args[2])

        for k in mod0._buffers:
            self.assertTrue(k in mod1._buffers and k in mod2._buffers)
            self.assertEqual(mod0._buffers[k], mod1._buffers[k])
            self.assertEqual(mod0._buffers[k], mod2._buffers[k])
        return backend.fw_graphs[0]

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_input_mutation(self, device, dynamic):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. Cond
        #   whose both branches mutate their cloned input in place; verifies results with
        #   the check helper, and for static CUDA without cross-ref compares the golden
        #   forward graph (auto_functionalized_v2 of the cond op with two subgraphs that
        #   copy the mutation back).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module that clones x, runs a mutating cond and adds y
            #   to the result.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: clones x, conditions on the sign of its sum with the same
                #   mutating function on both branches, and returns y plus the result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and returns sine of x.
                    x.add_(1)
                    return x.sin()

                x = x.clone()
                ret = torch.cond(x.sum() > 0, true_fn, true_fn, (x,))
                return y + ret

        x, y = (
            torch.randn(3, 4, requires_grad=True),
            torch.randn(3, 4, requires_grad=True),
        )
        fw_gm = self.check(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = gt, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _all_bases = [clone], _op_schema = _tree_spec_constant0);  gt = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 4]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None

        add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, getitem);  arg1_1 = getitem = None
        return (add,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)
""",
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_buffer_mutation(self, device, dynamic):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. Cond branches
        #   mutate both their input and a module buffer; verified with the check helper and, for
        #   static CUDA without cross-ref, a golden graph in which the buffer is passed as an
        #   additional base and copied back afterwards.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a buffer of ones whose cond branch
            #   mutates the buffer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer of eight ones on the device.
                super().__init__()
                self.register_buffer(
                    "buf", torch.ones(8, requires_grad=False, device=device)
                )

            def forward(self, p, x):
                # [FUNCTION DOC] HELPER: clones x, runs a mutating cond on predicate p and returns
                #   x plus buffer plus the cond result.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and to the buffer in place;
                    #   returns x plus the buffer.
                    x.add_(1)
                    self.buf.add_(1)
                    return x + self.buf

                x = x.clone()
                out = torch.cond(p, true_fn, true_fn, (x,))
                return x + self.buf + out

        p, x = torch.tensor(True), torch.randn(1, requires_grad=True)
        fw_gm = self.check(M, (p, x), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1]", arg1_1: "b8[]", arg2_1: "f32[8]"):
        clone: "f32[1]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = arg1_1, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg2_1, clone], _op_schema = _tree_spec_constant0);  arg1_1 = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[8]" = auto_functionalized_v2[0]
        getitem_1: "f32[8]" = auto_functionalized_v2[1]
        getitem_2: "f32[1]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem_2, getitem_1);  getitem_2 = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(add, getitem);  add = getitem = None

        copy_: "f32[8]" = torch.ops.aten.copy_.default(arg2_1, getitem_1);  arg2_1 = getitem_1 = copy_ = None
        return (add_1,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)
""",
            )

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/181947")
    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_cond_auto_functionalize_union_input_mutation(self, device, dynamic):
        # [FUNCTION DOC] Skipped on ROCm; needs CUDA and SM70; parametrized over device and
        #   dynamic. True branch mutates its input while the false branch mutates a buffer;
        #   verified via the check helper and, for static CUDA without cross-ref, a golden graph
        #   where the union of mutated inputs is handled.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a 4 by 3 buffer of ones whose branches
            #   mutate different state.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer.
                super().__init__()
                self.register_buffer("buf", torch.ones(4, 3, requires_grad=False))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: clones x, runs cond on the sign of its sum, and returns y
                #   plus result plus the sums of x and the buffer.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and returns sine of x matrix-
                    #   multiplied by the buffer.
                    x.add_(1)
                    return x.sin() @ self.buf

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: adds one to the buffer in place and returns sine of x
                    #   matrix-multiplied by the buffer.
                    self.buf.add_(1)
                    return x.sin() @ self.buf

                x = x.clone()
                ret = torch.cond(x.sum() > 0, true_fn, false_fn, (x,))
                return y + ret + x.sum() + self.buf.sum()

        x, y = (
            torch.randn(3, 4, requires_grad=False),
            torch.randn(1, requires_grad=False),
        )
        fw_gm = self.check(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[4, 3]", arg2_1: "f32[1]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.cond, pred = gt, true_fn = auto_functionalized_subgraph_0, false_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg1_1, clone], _op_schema = _tree_spec_constant0);  gt = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0]
        getitem_1: "f32[4, 3]" = auto_functionalized_v2[1]
        getitem_2: "f32[3, 4]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(getitem_1)
        add_2: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_1, sum_3);  add_1 = sum_3 = None

        copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (add_2,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, arg0_1);  sin = arg0_1 = None
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return (mm,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[4, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, add);  sin = None
            copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (mm,)
""",
            )

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_switch_auto_functionalize_input_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. Switch whose two
        #   identical branches mutate their input; verified on eager and AOT eager only (no
        #   inductor) with a golden forward graph for the static case without cross-ref
        #   (auto_functionalized_v2 of switch with a clamped index).
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module running switch over two mutating branches.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: clones x, computes an index from the sign of its
                #   sum, runs switch with the same mutating function twice, and returns y
                #   plus the result.
                def branch_mut(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and returns sine of x.
                    x.add_(1)
                    return x.sin()

                x = x.clone()
                idx = (x.sum() > 0).long()
                ret = switch(idx, (branch_mut, branch_mut), (x,))
                return y + ret

        x, y = (
            torch.randn(3, 4, requires_grad=False),
            torch.randn(3, 4, requires_grad=False),
        )
        fw_gm = self._check_eager_and_aot_eager_only(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic:
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[3, 4]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        _to_copy: "i64[]" = torch.ops.aten._to_copy.default(gt, dtype = torch.int64);  gt = None

        clamp: "i64[]" = torch.ops.aten.clamp.default(_to_copy, 0, 1);  _to_copy = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.switch, index = clamp, branch0_fn = auto_functionalized_subgraph_0, branch1_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _all_bases = [clone], _op_schema = _tree_spec_constant0);  clamp = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 4]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None

        add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, getitem);  arg1_1 = getitem = None
        return (add,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (sin,)
""",
            )

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_switch_auto_functionalize_buffer_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. Switch whose branches
        #   mutate both their input and a buffer, verified on eager and AOT eager with a golden
        #   graph for the static case without cross-ref.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a buffer of eight ones mutated in switch branches.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer.
                super().__init__()
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            def forward(self, p, x):
                # [FUNCTION DOC] HELPER: clones x, uses the predicate cast to an integer as the
                #   switch index, and returns x plus buffer plus the switch result.
                def branch_mut(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and to the buffer in place;
                    #   returns x plus the buffer.
                    x.add_(1)
                    self.buf.add_(1)
                    return x + self.buf

                x = x.clone()
                idx = p.long()
                out = switch(idx, (branch_mut, branch_mut), (x,))
                return x + self.buf + out

        p, x = torch.tensor(True), torch.randn(1, requires_grad=False)
        fw_gm = self._check_eager_and_aot_eager_only(M, (p, x), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic:
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1]", arg1_1: "b8[]", arg2_1: "f32[8]"):
        clone: "f32[1]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        _to_copy: "i64[]" = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.int64);  arg1_1 = None

        clamp: "i64[]" = torch.ops.aten.clamp.default(_to_copy, 0, 1);  _to_copy = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.switch, index = clamp, branch0_fn = auto_functionalized_subgraph_0, branch1_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg2_1, clone], _op_schema = _tree_spec_constant0);  clamp = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[8]" = auto_functionalized_v2[0]
        getitem_1: "f32[8]" = auto_functionalized_v2[1]
        getitem_2: "f32[1]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem_2, getitem_1);  getitem_2 = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(add, getitem);  add = getitem = None

        copy_: "f32[8]" = torch.ops.aten.copy_.default(arg2_1, getitem_1);  arg2_1 = getitem_1 = copy_ = None
        return (add_1,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[1]"):
            add: "f32[1]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            add_2: "f32[8]" = torch.ops.aten.add.Tensor(add, add_1)
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, add_1);  arg0_1 = add_1 = copy_ = None
            copy__1: "f32[1]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy__1 = None
            return (add_2,)
""",
            )

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_switch_auto_functionalize_union_input_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. Switch where one branch
        #   mutates its input and the other mutates a buffer; verified on eager and AOT eager with
        #   a golden graph for the static case without cross-ref.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a 4 by 3 buffer, switch branches
            #   mutate different state.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer.
                super().__init__()
                self.register_buffer("buf", torch.ones(4, 3, requires_grad=False))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: clones x, computes the index from the sign of its
                #   sum, runs switch over two branches and returns the sum with y, x sum
                #   and buffer sum.
                def branch0(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and returns sine of x matrix-
                    #   multiplied by the buffer.
                    x.add_(1)
                    return x.sin() @ self.buf

                def branch1(x):
                    # [FUNCTION DOC] HELPER: adds one to the buffer in place and returns sine of x
                    #   matrix-multiplied by the buffer.
                    self.buf.add_(1)
                    return x.sin() @ self.buf

                x = x.clone()
                idx = (x.sum() > 0).long()
                ret = switch(idx, (branch0, branch1), (x,))
                return y + ret + x.sum() + self.buf.sum()

        x, y = (
            torch.randn(3, 4, requires_grad=False),
            torch.randn(1, requires_grad=False),
        )
        fw_gm = self._check_eager_and_aot_eager_only(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic:
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 4]", arg1_1: "f32[4, 3]", arg2_1: "f32[1]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(clone)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        _to_copy: "i64[]" = torch.ops.aten._to_copy.default(gt, dtype = torch.int64);  gt = None

        clamp: "i64[]" = torch.ops.aten.clamp.default(_to_copy, 0, 1);  _to_copy = None
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.switch, index = clamp, branch0_fn = auto_functionalized_subgraph_0, branch1_fn = auto_functionalized_subgraph_1, _operand0_base_index = 0, _operand1_base_index = 1, _all_bases = [arg1_1, clone], _op_schema = _tree_spec_constant0);  clamp = auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = clone = _tree_spec_constant0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0]
        getitem_1: "f32[4, 3]" = auto_functionalized_v2[1]
        getitem_2: "f32[3, 4]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None

        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(getitem_1)
        add_2: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_1, sum_3);  add_1 = sum_3 = None

        copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (add_2,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[3, 4]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(add)
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, arg0_1);  sin = arg0_1 = None
            copy_: "f32[3, 4]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return (mm,)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 3]", arg1_1: "f32[3, 4]"):
            add: "f32[4, 3]" = torch.ops.aten.add.Tensor(arg0_1, 1)
            sin: "f32[3, 4]" = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(sin, add);  sin = None
            copy_: "f32[4, 3]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
            return (mm,)
""",
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_while_loop_auto_functionalize_buffer_mutation(self, device, dynamic):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. A while_loop
        #   body mutates a buffer; verified with the check helper and, for static CUDA without
        #   cross-ref, a golden graph (auto_functionalized_v2 of while_loop with the buffer as an
        #   additional input and a copy back).
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a buffer of eight ones whose while_loop
            #   body mutates it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer on the device.
                super().__init__()
                self.register_buffer(
                    "buf", torch.ones(8, requires_grad=False, device=device)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: clones x, runs the while_loop and returns x plus the sum
                #   of the buffer plus the loop result.
                def cond_fn(x):
                    # [FUNCTION DOC] HELPER: true while the sum of x is positive.
                    return x.sum() > 0

                def body_fn(x):
                    # [FUNCTION DOC] HELPER: subtracts one from x, subtracts one from the buffer
                    #   in place, and returns a tuple with the new x plus the buffer sum.
                    x_new = x.add(-1)
                    self.buf.add_(-1)
                    return (x_new + self.buf.sum(),)

                x = x.clone()
                out = while_loop(cond_fn, body_fn, (x,))
                return x + self.buf.sum() + out[0]

        x = torch.tensor([2.0, 1.0], requires_grad=False)
        fw_gm = self.check(M, (x,), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2]", arg1_1: "f32[8]"):
        clone: "f32[2]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None

        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.while_loop, cond_fn = auto_functionalized_subgraph_0, body_fn = auto_functionalized_subgraph_1, carried_input0 = clone, _additional_input0_base_index = 0, _all_bases = [arg1_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = _tree_spec_constant0 = None
        getitem: "f32[2]" = auto_functionalized_v2[0]
        getitem_1: "f32[8]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_1)
        add: "f32[2]" = torch.ops.aten.add.Tensor(clone, sum_1);  clone = sum_1 = None
        add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, getitem);  add = getitem = None

        copy_: "f32[8]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (add_1,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[8]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
            return gt

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[8]"):
            add: "f32[2]" = torch.ops.aten.add.Tensor(arg0_1, -1);  arg0_1 = None
            add_1: "f32[8]" = torch.ops.aten.add.Tensor(arg1_1, -1)
            sum_1: "f32[]" = torch.ops.aten.sum.default(add_1)
            add_2: "f32[2]" = torch.ops.aten.add.Tensor(add, sum_1);  add = sum_1 = None
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg1_1, add_1);  arg1_1 = add_1 = copy_ = None
            return (add_2,)
""",
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_while_loop_auto_functionalize_multiple_buffer_mutation(
        self, device, dynamic
    ):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. A while_loop
        #   body mutates two buffers, one depending on the other; verified with the check helper
        #   and a golden graph for static CUDA without cross-ref.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two buffers (ones and zeros) updated in
            #   the loop body.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the two buffers on the device.
                super().__init__()
                self.register_buffer("buf1", torch.ones(4, device=device))
                self.register_buffer("buf2", torch.zeros(4, device=device))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs the while_loop on x.
                def cond_fn(x):
                    # [FUNCTION DOC] HELPER: true while the sum of x is positive.
                    return x.sum() > 0

                def body_fn(x):
                    # [FUNCTION DOC] HELPER: adds x into the first buffer in place, adds the first
                    #   buffer into the second, and returns x minus one.
                    self.buf1.add_(x)
                    self.buf2.add_(self.buf1)
                    return (x - 1,)

                return while_loop(cond_fn, body_fn, (x,))

        x = torch.tensor([3.0, 2.0, 1.0, 1.0], requires_grad=False)
        fw_gm = self.check(M, (x,), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4]", arg1_1: "f32[4]", arg2_1: "f32[4]"):
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.while_loop, cond_fn = auto_functionalized_subgraph_0, body_fn = auto_functionalized_subgraph_1, carried_input0 = arg0_1, _additional_input0_base_index = 0, _additional_input1_base_index = 1, _all_bases = [arg1_1, arg2_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = arg0_1 = _tree_spec_constant0 = None
        getitem: "f32[4]" = auto_functionalized_v2[0]
        getitem_1: "f32[4]" = auto_functionalized_v2[1]
        getitem_2: "f32[4]" = auto_functionalized_v2[2];  auto_functionalized_v2 = None
        copy_: "f32[4]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        copy__1: "f32[4]" = torch.ops.aten.copy_.default(arg2_1, getitem_2);  arg2_1 = getitem_2 = copy__1 = None
        return (getitem,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4]", arg1_1: "f32[4]", arg2_1: "f32[4]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
            return gt

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4]", arg1_1: "f32[4]", arg2_1: "f32[4]"):
            add: "f32[4]" = torch.ops.aten.add.Tensor(arg1_1, arg0_1)
            add_1: "f32[4]" = torch.ops.aten.add.Tensor(arg2_1, add)
            sub: "f32[4]" = torch.ops.aten.sub.Tensor(arg0_1, 1);  arg0_1 = None
            copy_: "f32[4]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            copy__1: "f32[4]" = torch.ops.aten.copy_.default(arg2_1, add_1);  arg2_1 = add_1 = copy__1 = None
            return (sub,)
""",
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_while_loop_auto_functionalize_buffer_in_cond(self, device, dynamic):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. The while_loop
        #   condition function itself mutates a counter buffer; verified with the check helper and
        #   a golden graph for static CUDA without cross-ref.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with an integer counter buffer incremented in
            #   the loop condition.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the counter buffer as zero.
                super().__init__()
                self.register_buffer("counter", torch.tensor(0, device=device))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs the while_loop on x.
                def cond_fn(x):
                    # [FUNCTION DOC] HELPER: increments the counter buffer in place and returns
                    #   whether it is below 3.
                    self.counter.add_(1)
                    return self.counter < 3

                def body_fn(x):
                    # [FUNCTION DOC] HELPER: returns a tuple with x plus one.
                    return (x + 1,)

                return while_loop(cond_fn, body_fn, (x,))

        x = torch.tensor([0.0, 0.0], requires_grad=False)
        fw_gm = self.check(M, (x,), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2]", arg1_1: "i64[]"):
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.while_loop, cond_fn = auto_functionalized_subgraph_0, body_fn = auto_functionalized_subgraph_1, carried_input0 = arg0_1, _additional_input0_base_index = 0, _all_bases = [arg1_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = arg0_1 = _tree_spec_constant0 = None
        getitem: "f32[2]" = auto_functionalized_v2[0]
        getitem_1: "i64[]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "i64[]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (getitem,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "i64[]"):
            add: "i64[]" = torch.ops.aten.add.Tensor(arg1_1, 1)
            lt: "b8[]" = torch.ops.aten.lt.Scalar(add, 3)
            copy_: "i64[]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return lt

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "i64[]"):
            add: "f32[2]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
            return (add,)
""",
            )

    @requires_cuda
    @unittest.skipIf(not SM70OrLater, "triton")
    @parametrize("device", ["cuda", "cpu"])
    @parametrize("dynamic", [True, False])
    def test_while_loop_auto_functionalize_captured_tensor_mutation(
        self, device, dynamic
    ):
        # [FUNCTION DOC] Needs CUDA and SM70; parametrized over device and dynamic. The while_loop
        #   body mutates a tensor passed to forward (captured in the body); verified with the
        #   check helper; for static CUDA without cross-ref it also prints the graph and compares
        #   the golden graph.
        # KNOWN ISSUE: the test contains a leftover print statement of the graph
        #   before the assertion.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose while_loop body mutates the argument y.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: runs the while_loop on x with a body mutating y, then
                #   returns the result plus the sum of y.
                def cond_fn(x):
                    # [FUNCTION DOC] HELPER: true while the sum of x is positive.
                    return x.sum() > 0

                def body_fn(x):
                    # [FUNCTION DOC] HELPER: subtracts one from y in place and returns a tuple
                    #   with x plus the sum of y.
                    y.add_(-1)
                    return (x + y.sum(),)

                out = while_loop(cond_fn, body_fn, (x,))
                return out[0] + y.sum()

        x = torch.tensor([3.0, 2.0], requires_grad=False)
        y = torch.ones(4, requires_grad=False)

        fw_gm = self.check(M, (x, y), device, dynamic)
        if not TEST_WITH_CROSSREF and not dynamic and device == "cuda":
            print(normalize_gm(fw_gm.print_readable(print_output=False)))
            self.assertExpectedInline(
                normalize_gm(fw_gm.print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2]", arg1_1: "f32[4]"):
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.while_loop, cond_fn = auto_functionalized_subgraph_0, body_fn = auto_functionalized_subgraph_1, carried_input0 = arg0_1, _additional_input0_base_index = 0, _all_bases = [arg1_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = auto_functionalized_subgraph_1 = arg0_1 = _tree_spec_constant0 = None
        getitem: "f32[2]" = auto_functionalized_v2[0]
        getitem_1: "f32[4]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_1)
        add: "f32[2]" = torch.ops.aten.add.Tensor(getitem, sum_1);  getitem = sum_1 = None

        copy_: "f32[4]" = torch.ops.aten.copy_.default(arg1_1, getitem_1);  arg1_1 = getitem_1 = copy_ = None
        return (add,)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[4]"):
            sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
            gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
            return gt

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[4]"):
            add: "f32[4]" = torch.ops.aten.add.Tensor(arg1_1, -1)
            sum_1: "f32[]" = torch.ops.aten.sum.default(add)
            add_1: "f32[2]" = torch.ops.aten.add.Tensor(arg0_1, sum_1);  arg0_1 = sum_1 = None
            copy_: "f32[4]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return (add_1,)
""",
            )

    def _check_eager_and_aot_eager_only(self, gen_fn, args, device, dynamic):
        # [FUNCTION DOC] HELPER: simplified version of the check helper that runs only the eager
        #   reference and the AOT eager recording backend (no inductor), for HOPs whose inductor
        #   mutation support is not finished; asserts equal outputs, args and buffers; returns the
        #   recorded forward graph.
        """Stripped-down version of self.check that runs eager + aot_eager
        but NOT inductor.

        Used by HOP input-mutation tests (scan, map, ...). These HOPs
        currently only have Steps 1-3 (gen_schema, py_functionalize_impl,
        Dynamo) wired for input mutation. Step 4 (Inductor MutationOutput
        emission) is a follow-up; until then the Inductor arm of the full
        check() helper trips on the un-tracked mutation. This helper
        exercises everything that IS in place.
        """
        args = pytree.tree_map(lambda t: t.to(device=device), args)

        def _clone(args):
            # [FUNCTION DOC] HELPER: returns a list with clones of tensor arguments and
            #   other arguments unchanged.
            return [
                arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args
            ]

        def _new_fn():
            # [FUNCTION DOC] HELPER: calls the generator and moves a module result to the device.
            mod_or_fn = gen_fn()
            if isinstance(mod_or_fn, torch.nn.Module):
                mod_or_fn.to(device)
            return mod_or_fn

        cloned_args = [_clone(args) for _ in range(2)]
        with torch.no_grad():
            mod0 = _new_fn()
            exp = mod0(*cloned_args[0])
        backend = AotEagerAndRecordGraphs()
        torch._dynamo.reset()
        with torch.no_grad():
            mod1 = _new_fn()
            eager_out = torch.compile(
                mod1, backend=backend, fullgraph=True, dynamic=dynamic
            )(*cloned_args[1])

        self.assertEqual(exp, eager_out)
        self.assertEqual(cloned_args[0], cloned_args[1])
        for k in mod0._buffers:
            self.assertTrue(k in mod1._buffers)
            self.assertEqual(mod0._buffers[k], mod1._buffers[k])
        return backend.fw_graphs[0]

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_scan_auto_functionalize_buffer_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. A scan combine function
        #   mutates a buffer; verified on eager and AOT eager, and the printed graph must mention
        #   auto_functionalized_v2 and the scan op.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a buffer of four ones updated by the
            #   scan combine function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the buffer.
                super().__init__()
                self.register_buffer(
                    "buf", torch.ones(4, requires_grad=False, device=device)
                )

            def forward(self, init, xs):
                # [FUNCTION DOC] HELPER: scans the combine function over xs along dim 0.
                def combine_fn(carry, x):
                    # [FUNCTION DOC] HELPER: adds x into the buffer in place and returns carry
                    #   plus x and carry times x plus the buffer sum.
                    self.buf.add_(x)
                    return carry + x, carry * x + self.buf.sum()

                return scan(combine_fn, init, xs, dim=0)

        init = torch.zeros(4, requires_grad=False)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        fw_gm = self._check_eager_and_aot_eager_only(M, (init, xs), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)
        self.assertIn("torch.ops.higher_order.scan", graph_str)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_scan_auto_functionalize_multiple_buffer_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. Scan combine function
        #   mutates two buffers; the printed graph must mention auto_functionalized_v2.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two buffers updated by the scan combine function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers the two buffers.
                super().__init__()
                self.register_buffer("buf1", torch.ones(4, device=device))
                self.register_buffer("buf2", torch.zeros(4, device=device))

            def forward(self, init, xs):
                # [FUNCTION DOC] HELPER: scans the combine function over xs along dim 0.
                def combine_fn(carry, x):
                    # [FUNCTION DOC] HELPER: adds x into the first buffer, adds the first buffer
                    #   into the second, and returns carry plus x and carry times x.
                    self.buf1.add_(x)
                    self.buf2.add_(self.buf1)
                    return carry + x, carry * x

                return scan(combine_fn, init, xs, dim=0)

        init = torch.zeros(4, requires_grad=False)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        fw_gm = self._check_eager_and_aot_eager_only(M, (init, xs), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_scan_auto_functionalize_captured_tensor_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. Scan
        #   combine function mutates a captured argument tensor; the printed graph
        #   must mention auto_functionalized_v2.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose scan combine function mutates the
            #   tensor argument y.
            def forward(self, init, xs, y):
                # [FUNCTION DOC] HELPER: scans, then returns the first scan result plus the sum of
                #   y and the second result.
                def combine_fn(carry, x):
                    # [FUNCTION DOC] HELPER: subtracts one from y in place and returns carry plus
                    #   x and carry times x plus the sum of y.
                    y.add_(-1)
                    return carry + x, carry * x + y.sum()

                out = scan(combine_fn, init, xs, dim=0)
                return out[0] + y.sum(), out[1]

        init = torch.zeros(4, requires_grad=False)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        y = torch.ones(4, requires_grad=False)
        fw_gm = self._check_eager_and_aot_eager_only(M, (init, xs, y), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)

    def test_scan_init_mutation_graph_breaks(self):
        # [FUNCTION DOC] A scan combine function that mutates its carry (init) in place is
        #   disallowed: compiling the scan with the eager backend and full graph under no_grad
        #   raises UncapturedHigherOrderOpError naming the scan higher-order op.
        def combine_fn(carry, x):
            # [FUNCTION DOC] HELPER: adds one to carry in place and returns carry plus x and
            #   carry times x.
            carry.add_(1)  # mutating init is disallowed
            return carry + x, carry * x

        init = torch.zeros(4)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        def fn(init, xs):
            # [FUNCTION DOC] HELPER: scan of combine_fn over xs along dim 0.
            return scan(combine_fn, init, xs, dim=0)

        torch._dynamo.reset()
        with torch.no_grad():
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.ops\.higher_order\.scan",
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(init, xs)

    def test_scan_xs_mutation_graph_breaks(self):
        # [FUNCTION DOC] A scan combine function that mutates its per-step xs element in place is
        #   disallowed: compiling with the eager backend and full graph under no_grad raises
        #   UncapturedHigherOrderOpError naming the scan op.
        def combine_fn(carry, x):
            # [FUNCTION DOC] HELPER: adds one to x in place and returns carry plus x and
            #   carry times x.
            x.add_(1)  # mutating xs[t] is disallowed
            return carry + x, carry * x

        init = torch.zeros(4)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        def fn(init, xs):
            # [FUNCTION DOC] HELPER: scan of combine_fn over xs along dim 0.
            return scan(combine_fn, init, xs, dim=0)

        torch._dynamo.reset()
        with torch.no_grad():
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.ops\.higher_order\.scan",
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(init, xs)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_scan_auto_functionalize_partial_buffer_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. A scan combine
        #   function mutates two of three buffers (leaving the middle one only read) so the
        #   mutated set is non-contiguous; verified on eager and AOT eager and the printed graph
        #   must mention auto_functionalized_v2.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module with three zero buffers, of which the first and third are
            #   mutated by the scan combine function.
            def __init__(self):
                # [FUNCTION DOC] HELPER: registers three buffers of four zeros.
                super().__init__()
                self.register_buffer("buf0", torch.zeros(4, device=device))
                self.register_buffer("buf1", torch.zeros(4, device=device))
                self.register_buffer("buf2", torch.zeros(4, device=device))

            def forward(self, init, xs):
                # [FUNCTION DOC] HELPER: scans the combine function over xs along dim 0.
                def combine_fn(carry, x):
                    # [FUNCTION DOC] HELPER: adds x into buffers buf0 and buf2 in place; returns
                    #   carry plus x and carry times x plus the sum of buf1.
                    # Skip buf1 to make non-contiguous mutation set.
                    self.buf0.add_(x)
                    self.buf2.add_(x)
                    return carry + x, carry * x + self.buf1.sum()

                return scan(combine_fn, init, xs, dim=0)

        init = torch.zeros(4, requires_grad=False)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        fw_gm = self._check_eager_and_aot_eager_only(M, (init, xs), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)

    def test_scan_eager_init_mutation_raises(self):
        # [FUNCTION DOC] Calling scan eagerly with a combine function that mutates its carry in
        #   place raises UncapturedHigherOrderOpError naming the scan op.
        def combine_fn(carry, x):
            # [FUNCTION DOC] HELPER: adds one to carry in place and returns carry plus x and
            #   carry times x.
            carry.add_(1)
            return carry + x, carry * x

        init = torch.zeros(4)
        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.scan",
        ):
            scan(combine_fn, init, xs, dim=0)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_map_auto_functionalize_xs_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. A map body mutating its
        #   mapped element in place is auto-functionalized; verified on eager and AOT eager and
        #   the printed graph must mention auto_functionalized_v2 and the map_impl op.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module that maps a body mutating its element.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: maps body_fn over xs.
                def body_fn(x):
                    # [FUNCTION DOC] HELPER: adds one to x in place and returns x times 2 plus the
                    #   sum of x.
                    x.add_(1)
                    return x * 2 + x.sum()

                return control_flow.map(body_fn, xs)

        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        fw_gm = self._check_eager_and_aot_eager_only(M, (xs,), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)
        self.assertIn("torch.ops.higher_order.map_impl", graph_str)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_map_auto_functionalize_multiple_xs_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. A map over a tuple of
        #   two inputs whose body mutates both elements; verified on eager and AOT eager and the
        #   printed graph must mention auto_functionalized_v2.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module that maps a body mutating two mapped elements.
            def forward(self, xs1, xs2):
                # [FUNCTION DOC] HELPER: maps body_fn over the pair of inputs.
                def body_fn(xs):
                    # [FUNCTION DOC] HELPER: unpacks the pair, adds one to the first in place and
                    #   subtracts one from the second in place, and returns their sum and product.
                    x1, x2 = xs
                    x1.add_(1)
                    x2.sub_(1)
                    return x1 + x2, x1 * x2

                return control_flow.map(body_fn, (xs1, xs2))

        xs1 = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        xs2 = torch.arange(20, dtype=torch.float32).reshape(5, 4) + 100
        fw_gm = self._check_eager_and_aot_eager_only(M, (xs1, xs2), device, dynamic)
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)

    def test_map_pos_args_mutation_graph_breaks(self):
        # [FUNCTION DOC] A map body that mutates an additional positional argument is disallowed:
        #   compiling with the eager backend and full graph raises UncapturedHigherOrderOpError
        #   naming the map_impl op.
        def body_fn(x, buf):
            # [FUNCTION DOC] HELPER: adds x into buf in place and returns x times 2.
            buf.add_(x)  # mutating pos_args is disallowed
            return x * 2

        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        buf = torch.zeros(4)

        def fn(xs, buf):
            # [FUNCTION DOC] HELPER: maps body_fn over xs with buf as extra argument.
            return control_flow.map(body_fn, xs, buf)

        torch._dynamo.reset()
        with torch.no_grad():
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(xs, buf)

    def test_map_captured_tensor_mutation_graph_breaks(self):
        # [FUNCTION DOC] A map body that mutates a captured loop-invariant tensor is disallowed:
        #   compiling raises UncapturedHigherOrderOpError naming the map_impl op.
        y = torch.ones(4)

        def body_fn(x):
            # [FUNCTION DOC] HELPER: subtracts one from the captured y in place and returns x
            #   times 2 plus the sum of y.
            y.add_(-1)  # mutating a captured (loop-invariant) tensor is disallowed
            return x * 2 + y.sum()

        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        def fn(xs):
            # [FUNCTION DOC] HELPER: maps body_fn over xs.
            return control_flow.map(body_fn, xs)

        torch._dynamo.reset()
        with torch.no_grad():
            with self.assertRaisesRegex(
                torch._dynamo.exc.UncapturedHigherOrderOpError,
                r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(xs)

    @skipIfTorchDynamo()
    @parametrize("dynamic", [True, False])
    def test_map_auto_functionalize_partial_xs_mutation(self, dynamic):
        # [FUNCTION DOC] Skipped under dynamo; parametrized over dynamic. A map body mutates the
        #   first and third of three mapped inputs, leaving the middle one read-only; verified on
        #   eager and AOT eager and the printed graph must mention auto_functionalized_v2.
        device = "cpu"

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module mapping a body over three inputs.
            def forward(self, xs0, xs1, xs2):
                # [FUNCTION DOC] HELPER: maps body_fn over the triple of inputs.
                def body_fn(xs):
                    # [FUNCTION DOC] HELPER: unpacks three elements, adds one in place to the
                    #   first and third, and returns the first times 2 plus the sums of the
                    #   second and third.
                    x0, x1, x2 = xs
                    # Skip x1 to make a non-contiguous mutation set.
                    x0.add_(1)
                    x2.add_(1)
                    return x0 * 2 + x1.sum() + x2.sum()

                return control_flow.map(body_fn, (xs0, xs1, xs2))

        xs0 = torch.zeros(5, 4)
        xs1 = torch.zeros(5, 4)
        xs2 = torch.zeros(5, 4)
        fw_gm = self._check_eager_and_aot_eager_only(
            M, (xs0, xs1, xs2), device, dynamic
        )
        graph_str = fw_gm.print_readable(print_output=False)
        self.assertIn("auto_functionalized_v2", graph_str)

    def test_map_eager_pos_args_mutation_raises(self):
        # [FUNCTION DOC] Calling map eagerly with a body that mutates its additional positional
        #   argument raises UncapturedHigherOrderOpError naming the map_impl op.
        def body_fn(x, buf):
            # [FUNCTION DOC] HELPER: adds x into buf in place and returns x times 2.
            buf.add_(x)
            return x * 2

        xs = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        buf = torch.zeros(4)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            r"Higher Order Operator: torch\.ops\.higher_order\.map_impl",
        ):
            control_flow.map(body_fn, xs, buf)


_hop_schema_test_schema_types = [
    "bool",
    "int",
    "float",
    "str",
    "Tensor",
    "SymInt",
    "SymBool",
    "GraphModule",
    "ScriptObj",
]


@skipIfTorchDynamo("We don't expect users to torch.compile hop schema generation.")
@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
class TestHopSchema(TestCase):
    # [CLASS DOC] Tests schema generation for higher-order ops: type, list and function
    #   schema generators over example values of several types, the schema tree spec checks,
    #   and the gen_schema methods of cond, switch, while_loop, scan, associative_scan and
    #   map_impl with golden schema strings (including mutation annotations). Skipped under
    #   dynamo and on Windows.
    def _get_example_val(self, ty: str):
        # [FUNCTION DOC] HELPER: returns an example value for a schema type name: True, 1, 1.0,
        #   "foo", a scalar tensor, a symbolic int or bool created from a fresh shape env, a
        #   traced graph module of a sine function, or a TorchBind object (after initializing the
        #   TorchBind test implementations); raises NotImplementedError for other names.
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def create_symtype(cls, pytype, shape_env, val):
            # [FUNCTION DOC] HELPER: creates a symbol in the shape env with a testing constant
            #   source named by the current count of backed variables and wraps it in a SymNode of
            #   the given Python type and hint, then in the given symbolic class.
            from torch._dynamo.source import ConstantSource

            symbol = shape_env.create_symbol(
                val,
                source=ConstantSource(
                    f"__testing_hop_schema{len(shape_env.backed_var_to_val)}"
                ),
            )
            return cls(SymNode(symbol, shape_env, pytype, hint=val))

        if ty == "bool":
            return True
        elif ty == "int":
            return 1
        elif ty == "float":
            return 1.0
        elif ty == "str":
            return "foo"
        elif ty == "Tensor":
            return torch.tensor(1)
        elif ty == "SymInt":
            shape_env = ShapeEnv()
            return create_symtype(torch.SymInt, int, shape_env, 1)
        elif ty == "SymBool":
            shape_env = ShapeEnv()
            return create_symtype(torch.SymBool, bool, shape_env, True)
        elif ty == "GraphModule":

            def f(x):
                # [FUNCTION DOC] HELPER: returns the sine of x.
                return x.sin()

            return make_fx(f)(torch.ones(1))
        elif ty == "ScriptObj":
            from torch.testing._internal.torchbind_impls import (
                init_torchbind_implementations,
            )

            init_torchbind_implementations()
            foo = torch.classes._TorchScriptTesting._Foo(3, 4)
            return foo
        else:
            raise NotImplementedError(ty)

    @parametrize("schema_type", _hop_schema_test_schema_types)
    def test_type_gen(self, schema_type):
        # [FUNCTION DOC] Parametrized over the schema type names. Generates a schema type from an
        #   example value and checks that parsing its string form gives back the same type.
        from torchgen.gen_schema_utils import TypeGen

        example_val = self._get_example_val(schema_type)
        ty = TypeGen.from_example(example_val)
        # Test the generated type can be parsed
        self.assertEqual(ty.parse(str(ty)), ty)

    @parametrize("schema_type", _hop_schema_test_schema_types)
    def test_list_gen(self, schema_type):
        # [FUNCTION DOC] Parametrized over the schema type names. Generates a schema type from a
        #   one-element list of an example value (twice) and checks each string form parses back
        #   to the same type.
        from torchgen.gen_schema_utils import TypeGen

        example_val = self._get_example_val(schema_type)
        li1 = [example_val]
        ty1 = TypeGen.from_example(li1)
        ty2 = TypeGen.from_example(li1)
        self.assertEqual(ty1.parse(str(ty1)), ty1)
        self.assertEqual(ty2.parse(str(ty2)), ty2)

    def test_function_schema_gen(self):
        # [FUNCTION DOC] Generates function schemas from named example inputs of all types and one
        #   tensor, a list of one tensor and a list of two tensors as outputs; compares the golden
        #   schema strings and checks each parses back to an equal schema.
        from torchgen.gen_schema_utils import FunctionSchemaGen

        inps = [
            (schema_type + "_v", self._get_example_val(schema_type))
            for schema_type in _hop_schema_test_schema_types
        ]
        schema1 = FunctionSchemaGen.from_example("test_op1", inps, torch.ones(1))
        schema2 = FunctionSchemaGen.from_example(
            "test_op2",
            inps,
            [
                torch.ones(1),
            ],
        )
        schema3 = FunctionSchemaGen.from_example(
            "test_op3", inps, [torch.ones(1), torch.ones(1)]
        )
        self.assertExpectedInline(
            str(schema1),
            """test_op1(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> Tensor""",
        )
        self.assertExpectedInline(
            str(schema2),
            """test_op2(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> Tensor""",
        )
        self.assertExpectedInline(
            str(schema3),
            """test_op3(bool bool_v, int int_v, float float_v, str str_v, Tensor Tensor_v, SymInt SymInt_v, SymBool SymBool_v, GraphModule GraphModule_v, __torch__.torch.classes._Foo ScriptObj_v) -> (Tensor, Tensor)""",
        )
        self.assertEqual(schema1.parse(str(schema1)), schema1)
        self.assertEqual(schema2.parse(str(schema2)), schema2)
        self.assertEqual(schema3.parse(str(schema3)), schema3)

    def test_schema_tree_spec(self):
        # [FUNCTION DOC] Adding a tuple as one argument to the cond schema generator raises
        #   RuntimeError asking for flattened inputs; adding the flattened tensors and a tree spec
        #   yields a schema with two tensor arguments and no return.
        schema_gen = HopSchemaGenerator(torch.ops.higher_order.cond)
        args = (torch.randn(3, 4), torch.randn(2, 3))
        with self.assertRaisesRegex(
            RuntimeError, "Please only add flattened inputs to the hop schema"
        ):
            schema_gen.add_arg("tuple_args", args)

        for i, arg in enumerate(args):
            schema_gen.add_arg(f"tuple_args{i}", arg)
        schema_gen.add_schema_tree_spec(pytree.tree_flatten(args)[1])
        flat_schema = schema_gen.gen_schema()
        self.assertExpectedInline(
            str(flat_schema), """cond(Tensor tuple_args0, Tensor tuple_args1) -> ()"""
        )

    def test_cond_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the cond schema string for a tensor predicate, two lambda branches
        #   and one tensor operand.
        schema = torch.ops.higher_order.cond.gen_schema(
            torch.tensor(True),
            lambda x: x.sin(),
            lambda x: x.cos(),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """cond(Tensor pred, Any true_fn, Any false_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_cond_gen_schema_symbool_inputs(self):
        # [FUNCTION DOC] Checks the cond schema string uses the symbolic bool type when
        #   the predicate is a symbolic bool derived from the size of a nonzero result
        #   under a fake mode.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode, fake_mode.shape_env.ignore_fresh_unbacked_symbols():
            sym_bool = torch.randn(3, 4).nonzero().size(0) == 0

        schema = torch.ops.higher_order.cond.gen_schema(
            sym_bool,
            lambda x: x.sin(),
            lambda x: x.cos(),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """cond(SymBool pred, Any true_fn, Any false_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_switch_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the switch schema string for a tensor index, three branches
        #   and one operand.
        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(0),
            (lambda x: x.sin(), lambda x: x.cos(), lambda x: x.tan()),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Any branch2_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_switch_gen_schema_symint_inputs(self):
        # [FUNCTION DOC] Checks the switch schema string uses the symbolic int type for an index
        #   derived from a nonzero size under a fake mode.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode, fake_mode.shape_env.ignore_fresh_unbacked_symbols():
            sym_int = torch.randn(3, 4).nonzero().size(0)

        schema = torch.ops.higher_order.switch.gen_schema(
            sym_int,
            (lambda x: x.sin(), lambda x: x.cos()),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(SymInt index, Any branch0_fn, Any branch1_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_switch_gen_schema_multiple_operands(self):
        # [FUNCTION DOC] Checks the switch schema for two operands and branches
        #   returning two tensors.
        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(1),
            (lambda x, y: (x + y, x * y), lambda x, y: (x - y, x / y)),
            (torch.randn(3, 4), torch.randn(3, 4)),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Tensor operand0, Tensor operand1) -> (Tensor, Tensor)""",
        )

    def test_switch_gen_schema_with_input_mutation(self):
        # [FUNCTION DOC] Checks the switch schema marks the operand mutated by one branch as
        #   mutable (alias annotation with a write mark).
        def branch0(x, y):
            # [FUNCTION DOC] HELPER: adds one to x in place and returns x plus y.
            x.add_(1)
            return x + y

        def branch1(x, y):
            # [FUNCTION DOC] HELPER: returns x minus y.
            return x - y

        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(0),
            (branch0, branch1),
            (torch.randn(3, 4), torch.randn(3, 4)),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Tensor(a3!) operand0, Tensor operand1) -> ((Tensor))""",
        )

    def test_switch_gen_schema_with_disjoint_branch_mutation(self):
        # [FUNCTION DOC] Checks that when different branches mutate different operands, the schema
        #   marks both operands as mutable.
        def branch0(x, y):
            # [FUNCTION DOC] HELPER: adds one to x in place and returns x plus y.
            x.add_(1)
            return x + y

        def branch1(x, y):
            # [FUNCTION DOC] HELPER: subtracts one from y in place and returns x minus y.
            y.sub_(1)
            return x - y

        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(0),
            (branch0, branch1),
            (torch.randn(3, 4), torch.randn(3, 4)),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Tensor(a3!) operand0, Tensor(a4!) operand1) -> ((Tensor))""",
        )

    def test_switch_gen_schema_int_index(self):
        # [FUNCTION DOC] Checks the switch schema string uses the int type for a Python int index.
        schema = torch.ops.higher_order.switch.gen_schema(
            0,
            (lambda x: x.sin(), lambda x: x.cos()),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(int index, Any branch0_fn, Any branch1_fn, Tensor operand0) -> ((Tensor))""",
        )

    def test_switch_gen_schema_differing_int_symint_outputs(self):
        # [FUNCTION DOC] Checks the switch schema when branches return a tensor and different
        #   ints: the second return is typed as int.
        def branch0(x):
            # [FUNCTION DOC] HELPER: returns sine and the int 5.
            return x.sin(), 5

        def branch1(x):
            # [FUNCTION DOC] HELPER: returns cosine and the int 7.
            return x.cos(), 7

        def branch2(x):
            # [FUNCTION DOC] HELPER: returns tan and the int 9.
            return x.tan(), 9

        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(0),
            (branch0, branch1, branch2),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Any branch2_fn, Tensor operand0) -> (Tensor, int)""",
        )

    def test_switch_gen_schema_matching_int_int_outputs(self):
        # [FUNCTION DOC] Checks the switch schema when branches return a tensor and the same int:
        #   the second return is typed as int.
        def branch0(x):
            # [FUNCTION DOC] HELPER: returns sine and the int 5.
            return x.sin(), 5

        def branch1(x):
            # [FUNCTION DOC] HELPER: returns cosine and the int 5.
            return x.cos(), 5

        schema = torch.ops.higher_order.switch.gen_schema(
            torch.tensor(0),
            (branch0, branch1),
            (torch.randn(3, 4),),
        )
        self.assertExpectedInline(
            str(schema),
            """switch(Tensor index, Any branch0_fn, Any branch1_fn, Tensor operand0) -> (Tensor, int)""",
        )

    def test_while_loop_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the while_loop schema string for two tensor carries and
        #   no additional inputs.
        def cond_fn(x, y):
            # [FUNCTION DOC] HELPER: true while the sum of x is below 10.
            return x.sum() < 10

        def body_fn(x, y):
            # [FUNCTION DOC] HELPER: returns x plus one and sine of y.
            return x + 1, y.sin()

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor carried_input0, Tensor carried_input1) -> (Tensor, Tensor)""",
        )

    def test_while_loop_gen_schema_with_additional_inputs(self):
        # [FUNCTION DOC] Checks the while_loop schema string for two tensor carries and one
        #   tensor additional input.
        def cond_fn(x, y, z):
            # [FUNCTION DOC] HELPER: true while the sum of x is below z.
            return x.sum() < z

        def body_fn(x, y, z):
            # [FUNCTION DOC] HELPER: returns x plus one and sine of y.
            return x + 1, y.sin()

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (torch.tensor(10),),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor carried_input0, Tensor carried_input1, Tensor additional_input0) -> (Tensor, Tensor)""",
        )

    def test_scan_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the scan schema string for one init, one xs and no additional
        #   inputs; two tensor returns.
        def combine_fn(carry, x):
            # [FUNCTION DOC] HELPER: returns carry plus x and carry times x.
            return carry + x, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0) -> (Tensor, Tensor)""",
        )

    def test_scan_gen_schema_with_additional_inputs(self):
        # [FUNCTION DOC] Checks the scan schema string with one additional tensor input.
        def combine_fn(carry, x, scale):
            # [FUNCTION DOC] HELPER: returns carry plus x times scale and carry times x.
            return carry + x * scale, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (torch.tensor(2.0),),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0, Tensor additional_input0) -> (Tensor, Tensor)""",
        )

    def test_scan_gen_schema_multiple_inputs(self):
        # [FUNCTION DOC] Checks the scan schema string for two inits and two xs with four returns.
        def combine_fn(carry1, carry2, x1, x2):
            # [FUNCTION DOC] HELPER: returns four values built from the two carries and two xs.
            return carry1 + x1, carry2 * x2, carry1 - x1, carry2 + x2

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4), torch.randn(2, 3)),
            (torch.randn(5, 3, 4), torch.randn(5, 2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor init1, Tensor xs0, Tensor xs1) -> (Tensor, Tensor, Tensor, Tensor)""",
        )

    def test_scan_gen_schema_with_mutated_arg_indices_kwarg(self):
        # [FUNCTION DOC] Checks that passing mutated argument indices as a string (index 3) marks
        #   the second additional input as mutable in the scan schema.
        def combine_fn(carry, x, buf0, buf1):
            # [FUNCTION DOC] HELPER: returns carry plus x and carry times x, ignoring the buffers.
            return carry + x, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (torch.randn(3, 4), torch.randn(3, 4)),
            "3",
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0, Tensor additional_input0, Tensor(a4!) additional_input1) -> (Tensor, Tensor)""",
        )

    def test_scan_gen_schema_multiple_additional_mutation_via_kwarg(self):
        # [FUNCTION DOC] Checks that mutated indices 3 and 4 mark the second and third additional
        #   inputs as mutable in the scan schema.
        def combine_fn(carry, x, b0, b1, b2):
            # [FUNCTION DOC] HELPER: adds x in place to b1 and b2 and returns carry plus x and
            #   carry times x.
            b1.add_(x)
            b2.add_(x)
            return carry + x, carry * x

        schema = torch.ops.higher_order.scan.gen_schema(
            combine_fn,
            (torch.randn(3, 4),),
            (torch.randn(5, 3, 4),),
            (torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)),
            "3,4",
        )
        self.assertExpectedInline(
            str(schema),
            """scan(Any combine_fn, Tensor init0, Tensor xs0, Tensor additional_input0, Tensor(a4!) additional_input1, Tensor(a5!) additional_input2) -> (Tensor, Tensor)""",
        )

    def test_associative_scan_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the associative_scan schema string for one xs tensor and
        #   no additional inputs.
        def combine_fn(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4),),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0) -> ((Tensor))""",
        )

    def test_associative_scan_gen_schema_with_additional_inputs(self):
        # [FUNCTION DOC] Checks the associative_scan schema string with one additional tensor.
        def combine_fn(x, y, scale):
            # [FUNCTION DOC] HELPER: returns x times y times scale.
            return x * y * scale

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4),),
            (torch.tensor(2.0),),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0, Tensor additional_input0) -> ((Tensor))""",
        )

    def test_associative_scan_gen_schema_multiple_inputs(self):
        # [FUNCTION DOC] Checks the associative_scan schema string for two xs tensors
        #   with two returns.
        def combine_fn(x1, x2, y1, y2):
            # [FUNCTION DOC] HELPER: returns the sum of the first pair and the product of
            #   the second pair.
            return x1 + y1, x2 * y2

        schema = torch.ops.higher_order.associative_scan.gen_schema(
            combine_fn,
            (torch.randn(5, 3, 4), torch.randn(5, 2, 3)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """associative_scan(Any combine_fn, Tensor xs0, Tensor xs1) -> (Tensor, Tensor)""",
        )

    def test_while_loop_gen_schema_with_int_carries(self):
        # [FUNCTION DOC] Checks the while_loop schema string with two int carries, one tensor
        #   carry and one tensor additional input; the returns include the ints.
        def cond_fn(x, y, z, c):
            # [FUNCTION DOC] HELPER: true while x is less than y.
            return x < y

        def body_fn(x, y, z, c):
            # [FUNCTION DOC] HELPER: returns x plus one, y minus one, sine of z and c plus x.
            return x + 1, y - 1, z.sin(), c + x

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (2, 10, torch.randn(2, 3)),
            (torch.tensor(10),),
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, int carried_input0, int carried_input1, Tensor carried_input2, Tensor additional_input0) -> (int, int, Tensor, Tensor)""",
        )

    def test_while_loop_gen_schema_with_input_mutation(self):
        # [FUNCTION DOC] Checks that mutated indices 0 to 3 mark all three carries and the
        #   additional input as mutable in the while_loop schema.
        def cond_fn(x, y, z, c):
            # [FUNCTION DOC] HELPER: true while x is less than y.
            return x < y

        def body_fn(x, y, z, c):
            # [FUNCTION DOC] HELPER: mutates x, y, z and c in place and returns x, y, z.
            x.add_(1)
            y.sub_(1)
            z.sin_()
            c.add_(x)
            return x, y, z

        c = torch.randn(3, 3)

        schema = torch.ops.higher_order.while_loop.gen_schema(
            cond_fn,
            body_fn,
            (torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)),
            (c,),
            "0,1,2,3",
        )
        self.assertExpectedInline(
            str(schema),
            """while_loop(Any cond_fn, Any body_fn, Tensor(a2!) carried_input0, Tensor(a3!) carried_input1, Tensor(a4!) carried_input2, Tensor(a5!) additional_input0) -> (Tensor, Tensor, Tensor)""",
        )

    def test_map_gen_schema_tensor_inputs(self):
        # [FUNCTION DOC] Checks the map_impl schema string for one xs tensor.
        def body_fn(x):
            # [FUNCTION DOC] HELPER: returns sine of x.
            return x.sin()

        schema = torch.ops.higher_order.map_impl.gen_schema(
            body_fn,
            (torch.randn(5, 3, 4),),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """map_impl(Any f, Tensor xs0) -> ((Tensor))""",
        )

    def test_map_gen_schema_with_additional_inputs(self):
        # [FUNCTION DOC] Checks the map_impl schema string with one additional tensor.
        def body_fn(x, scale):
            # [FUNCTION DOC] HELPER: returns x times scale.
            return x * scale

        schema = torch.ops.higher_order.map_impl.gen_schema(
            body_fn,
            (torch.randn(5, 3, 4),),
            (torch.tensor(2.0),),
        )
        self.assertExpectedInline(
            str(schema),
            """map_impl(Any f, Tensor xs0, Tensor additional_input0) -> ((Tensor))""",
        )

    def test_map_gen_schema_multiple_inputs(self):
        # [FUNCTION DOC] Checks the map_impl schema string for two xs tensors with two returns.
        def body_fn(x1, x2):
            # [FUNCTION DOC] HELPER: returns x1 plus x2 and x1 times x2.
            return x1 + x2, x1 * x2

        schema = torch.ops.higher_order.map_impl.gen_schema(
            body_fn,
            (torch.randn(5, 3, 4), torch.randn(5, 3, 4)),
            (),
        )
        self.assertExpectedInline(
            str(schema),
            """map_impl(Any f, Tensor xs0, Tensor xs1) -> (Tensor, Tensor)""",
        )

    def test_map_gen_schema_with_mutated_arg_indices_kwarg(self):
        # [FUNCTION DOC] Checks that mutated index 0 marks xs0 as mutable in the map_impl schema.
        def body_fn(x):
            # [FUNCTION DOC] HELPER: returns x times 2.
            return x * 2

        schema = torch.ops.higher_order.map_impl.gen_schema(
            body_fn,
            (torch.randn(5, 3, 4),),
            (),
            "0",
        )
        self.assertExpectedInline(
            str(schema),
            """map_impl(Any f, Tensor(a1!) xs0) -> ((Tensor))""",
        )

    def test_map_gen_schema_multiple_xs_mutation_via_kwarg(self):
        # [FUNCTION DOC] Checks that mutated indices 0 and 2 mark xs0 and xs2 as mutable in
        #   the map_impl schema.
        def body_fn(x0, x1, x2):
            # [FUNCTION DOC] HELPER: adds one in place to x0 and x2 and returns the sum
            #   of the three.
            x0.add_(1)
            x2.add_(1)
            return x0 + x1 + x2

        schema = torch.ops.higher_order.map_impl.gen_schema(
            body_fn,
            (torch.randn(5, 3, 4), torch.randn(5, 3, 4), torch.randn(5, 3, 4)),
            (),
            "0,2",
        )
        self.assertExpectedInline(
            str(schema),
            """map_impl(Any f, Tensor(a1!) xs0, Tensor xs1, Tensor(a3!) xs2) -> ((Tensor))""",
        )


class DynamicCondModel(torch.nn.Module):
    # [CLASS DOC] HELPER: small network with a cond whose branches use different linear layers
    #   (one layer versus two layers) followed by ReLU and an output layer; the branch is chosen
    #   by the sign of the sum of the input.
    def __init__(self, input_size=16, hidden_size=64, output_size=10):
        # [FUNCTION DOC] HELPER: creates linear layers fc1_0 (input to hidden), fc1_1 (input to
        #   32), fc1_2 (32 to hidden), a ReLU and the output layer fc2 (hidden to output);
        #   defaults are 16, 64 and 10.
        super().__init__()
        self.fc1_0 = torch.nn.Linear(input_size, hidden_size)
        self.fc1_1 = torch.nn.Linear(input_size, 32)
        self.fc1_2 = torch.nn.Linear(32, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: runs cond on the sign of the sum of x between the two branches,
        #   then ReLU and the output layer; returns the output.
        def true_fn(x):
            # [FUNCTION DOC] HELPER: applies the first linear layer.
            return self.fc1_0(x)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: applies the second then the third linear layer.
            x = self.fc1_1(x)
            return self.fc1_2(x)

        pred = x.sum() > 0
        x = cond(pred, true_fn, false_fn, [x])

        x = self.relu(x)
        x = self.fc2(x)

        return x


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowNN(TestCase):
    # [CLASS DOC] Tests a neural network containing cond with CUDA graph conditional nodes;
    #   skipped without support.
    def test_cond_in_NN(self):
        # [FUNCTION DOC] Runs forward and backward of the dynamic cond model on a CUDA input
        #   through a function returning output, loss and parameter gradients, using the many-
        #   backends capture helper and the cudagraph backend helper.
        model = DynamicCondModel().cuda()

        def autograd_test(x):
            # [FUNCTION DOC] HELPER: clears parameter grads, runs the model, backpropagates the
            #   sum of the output, and returns the output, loss and list of parameter gradients.
            model.zero_grad(set_to_none=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            return (output, loss, grads)

        x = torch.randn(16, device="cuda")

        _check_compile_many_backends_with_cudagraph(self, autograd_test, [x])
        _check_compile_cudagraph_backend(self, autograd_test, [x])


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowAndRNG(TestCase):
    # [CLASS DOC] Tests random number generation interacting with CUDA graph conditional nodes;
    #   skipped without support.
    @parametrize("rng_func", ["custom_generator", "default_generator"])
    def test_rng_with_conditional_nodes_errors(self, rng_func):
        # [FUNCTION DOC] Parametrized over custom_generator and default_generator. Compiling with
        #   the cudagraphs backend a cond whose true branch uses random numbers raises
        #   RuntimeError saying RNG within data-dependent conditional nodes is not supported.
        # KNOWN ISSUE: the custom generator case calls skipTest immediately, so its remaining
        #   setup code never runs.
        pred = torch.tensor(True, device="cuda")
        x = torch.ones(10, dtype=torch.float32, device="cuda")

        if rng_func == "custom_generator":
            self.skipTest(
                "randn() currently does not work with a generator argument in dynamo."
            )
            generator = torch.Generator("cuda")

            def custom_generator(x):
                # [FUNCTION DOC] HELPER: adds random normal noise drawn from the custom generator
                #   to x (unreachable because the test is skipped first).
                return x + torch.randn(
                    *x.shape, generator=generator, dtype=x.dtype, device=x.device
                )

            rng_func = custom_generator
        elif rng_func == "default_generator":

            def default_generator(x):
                # [FUNCTION DOC] HELPER: adds random normal noise from the default generator to x.
                return x + torch.randn(*x.shape, dtype=x.dtype, device=x.device)

            rng_func = default_generator

        def func(pred, x):
            # [FUNCTION DOC] HELPER: cond on pred between the random function and doubling x.
            return torch.cond(pred, rng_func, lambda x: 2 * x, [x])

        compiled_func = torch.compile(func, backend="cudagraphs")
        with self.assertRaisesRegex(
            RuntimeError,
            "RNG within data-dependent conditional nodes is not supported yet",
        ):
            compiled_func(pred, x)

    def test_rng_outside_conditional_nodes_does_not_error(self):
        # [FUNCTION DOC] Cond branches are deterministic and random noise is added outside the
        #   cond; compiled with the cudagraphs backend three times without errors and with output
        #   shape equal to the input.
        pred = torch.tensor(True, device="cuda")
        x = torch.ones(10, dtype=torch.float32, device="cuda")

        def func(pred, x):
            # [FUNCTION DOC] HELPER: cond multiplying by 2 or 3, then adds random normal noise
            #   outside the cond.
            y = torch.cond(pred, lambda t: 2 * t, lambda t: 3 * t, [x])
            return y + torch.randn(*y.shape, dtype=y.dtype, device=y.device)

        compiled_func = torch.compile(func, backend="cudagraphs")
        for _ in range(3):
            out = compiled_func(pred, x)
            self.assertEqual(out.shape, x.shape)


instantiate_parametrized_tests(TestHopSchema)
instantiate_parametrized_tests(TestControlFlowTraced)
instantiate_parametrized_tests(TestAutoFunctionalizeControlFlow)

instantiate_parametrized_tests(TestControlFlow)
instantiate_parametrized_tests(AssociativeScanTests)

instantiate_parametrized_tests(TestControlFlowAndRNG)

if __name__ == "__main__":
    run_tests()
