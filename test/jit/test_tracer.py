# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the TorchScript tracer (torch.jit.trace, trace_module and the
# private traced graph helpers): tracing functions and modules, input and output structures
# (tuples, dicts, lists), in-place ops, hooks, warnings and the trace checker, export and
# import of traced modules, and mixing traced code with scripted code (class TestTracer and
# class TestMixTracingScripting). Importing appends the test directory to sys.path; running
# the file directly is rejected via raise_on_run_directly.

import copy
import io
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import warnings

# Standard library
from collections import namedtuple
from itertools import chain
from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    IS_SANDCASTLE,
    raise_on_run_directly,
    skipIfCompiledWithoutNumpy,
    skipIfCrossRef,
    skipIfTorchDynamo,
    suppress_warnings,
    TemporaryFileName,
)
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,
    _trace,
    enable_cpu_fuser,
    JitTestCase,
    make_global,
    RUN_CUDA,
    RUN_CUDA_MULTI_GPU,
)


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestTracer(JitTestCase):
    # [CLASS DOC] Test case for tracing: each test traces a function or module, then checks
    # results, graph contents, warnings or errors. Skipped under TorchDynamo.
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_large_nbr_kernel_args(self):
        # [FUNCTION DOC] CUDA-only. Traces a recurrent-style module with a long sequence
        # length, so the resulting kernel has many arguments. Moves the module and input to
        # the GPU on device zero (sets the current CUDA device); only checks that tracing
        # succeeds.
        class Recurrence(nn.Module):
            # [CLASS DOC] HELPER: Module that loops over the time dimension of its input,
            # scaling each step, then reassembles the sequence.
            def __init__(self, seq_len):
                # [FUNCTION DOC] HELPER: Stores the sequence length used as the number of
                # loop iterations.
                super().__init__()
                self.seq_len = seq_len

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Transposes the first two dims, doubles each time
                # slice in a loop of seq_len steps, concatenates the pieces, reshapes back
                # and transposes again. Returns the processed tensor.
                input = input.transpose(0, 1)

                # Main loop
                output = []
                for i in range(self.seq_len):
                    b = input[i] * 2
                    output.append(b)

                output = torch.cat(output, 0).view(input.size(0), *output[0].size())
                output = output.transpose(0, 1)
                return output

        input_size = 8
        batch_size = 2
        seq_len = 130

        rec = Recurrence(seq_len)
        input = torch.rand(batch_size, seq_len, input_size)

        torch.cuda.set_device(0)
        rec = rec.cuda()
        input = input.cuda()

        traced_rec = torch.jit.trace(rec, (input))

    def test_trace_legacy_ctor(self):
        # [FUNCTION DOC] Traces a module whose output includes a tensor created with the
        # legacy FloatTensor constructor; only checks that tracing does not fail.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning a tuple of an incremented input and a
            # legacy-constructed tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple: the input plus one, and a
                # one-element float tensor built through the legacy constructor.
                return (x + 1, torch.FloatTensor([0]))

        traced_rec = torch.jit.trace(MyModule(), torch.randn(2, 2))

    def test_simple(self):
        # [FUNCTION DOC] Checks that a simple composition of elementwise ops traces
        # correctly using the checkTrace helper.
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies add, mul, tanh and sigmoid to the inputs.
            return torch.sigmoid(torch.tanh(x * (x + y)))

        self.checkTrace(f, (x, y))

    def test_trace_checking_with_global_name(self):
        # [FUNCTION DOC] Traces a module taking a list of tensors where the example inputs
        # are module-level globals; checks that the trace checker works when input names
        # resolve through globals.
        class MyClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that concatenates a list of tensors along dim zero.
            def forward(self, xs: List[Tensor]):
                # [FUNCTION DOC] HELPER: Returns the concatenation of the input list of
                # tensors.
                y = torch.cat(xs, dim=0)
                return y

        model = MyClass()
        # Simulate these inputs being in the globals, like they would be if,
        # e.g. they were defined outermost scope of a script
        global input1, input2
        input1 = torch.ones(2, 2)
        input2 = torch.ones(2, 2)
        m2 = torch.jit.trace(model, ((input1, input2),))

    def test_trace_aliased_parameter(self):
        # [FUNCTION DOC] Traces a module using its parameter as the example input, so the
        # input aliases a parameter; checks that the traced result matches eager execution
        # on a fresh input.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module holding one Parameter and adding it to its input.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Wraps the given tensor as a Parameter attribute.
                super().__init__()
                self.x = nn.Parameter(x)

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Returns the input plus the parameter.
                return self.x + y

        m = M(torch.rand(3, 4))
        r = torch.jit.trace(m, m.x)
        t2 = torch.rand(3, 4)
        self.assertEqual(r(t2), m.x + t2)

    def test_trace_nested_fn(self):
        # [FUNCTION DOC] Traces a module whose forward defines and calls a scripted
        # function; checks tracing with check_trace succeeds when script functions are
        # created inside the traced call.
        class TracedInlineDecision(torch.nn.Module):
            # [CLASS DOC] HELPER: Traced module defining a scripted helper inside forward.
            def forward(self, x, flag):
                # [FUNCTION DOC] HELPER: Negates the input and passes it to a locally
                # defined scripted function together with a boolean flag.
                @torch.jit.script
                def make_decision(flag, x):
                    # [FUNCTION DOC] HELPER: Scripted function returning x when flag is true
                    # and a zeros tensor shaped like x otherwise.
                    if flag:
                        return x
                    else:
                        return torch.zeros_like(x)

                x = torch.neg(x)
                return make_decision(flag, x)

        decision = TracedInlineDecision()
        torch.jit.trace(
            decision,
            (torch.rand(3, 4), torch.tensor([True], dtype=torch.bool)),
            check_trace=True,
        )

    def test_trace_single_tuple(self):
        # [FUNCTION DOC] Traces a function returning a one-element tuple and raises
        # AssertionError if the traced output differs from eager.
        x = torch.tensor(2.0)

        def f2(x):
            # [FUNCTION DOC] HELPER: Returns a one-element tuple holding the input.
            return (x,)

        jit_f2 = torch.jit.trace(f2, x)
        if f2(x) != jit_f2(x):
            raise AssertionError(f"Traced output {jit_f2(x)} != eager output {f2(x)}")

    def test_trace_out_operator_with_two_output(self):
        # [FUNCTION DOC] Traces a function that calls cummax with out arguments for both
        # outputs; checks only that tracing succeeds.
        example_input = torch.rand(2, 8)
        out_1, out_2 = torch.cummax(example_input, 1)

        def run_cummax(example_input, out_1, out_2):
            # [FUNCTION DOC] HELPER: Calls cummax with preallocated out tensors and returns
            # the values and indices.
            output_1, output_2 = torch.cummax(example_input, 1, out=(out_1, out_2))
            return output_1, output_2

        trace_model = torch.jit.trace(run_cummax, (example_input, out_1, out_2))

    def test_trace_namedtuple(self):
        # [FUNCTION DOC] Traces a function taking a namedtuple; checks the traced module
        # gives the same result as eager on the same input.
        Point = namedtuple("point", ["x", "y"])

        def f(p):
            # [FUNCTION DOC] HELPER: Converts a plain tuple to the point namedtuple if
            # needed and returns the sum of its fields.
            if type(p) is tuple:
                p = Point(*p)
            return p.x + p.y

        p = Point(torch.randn(1), torch.randn(1))
        traced = torch.jit.trace(f, (p,))
        self.assertEqual(f(p), traced(p))

    def test_trace_topk(self):
        # [FUNCTION DOC] Traces topk with a tensor-valued k, then runs the trace on larger
        # and smaller inputs; asserts no slicing warning is emitted and the results equal
        # eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the indices of the top-k entries along
            # dim one.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the second output (indices) of topk with k
                # given as an input tensor.
                return x.topk(y, dim=1)[1]

        mod = M()
        inputs = (torch.randint(0, 10, (20, 20)), torch.tensor(17))
        traced_func = torch.jit.trace(mod, inputs)

        test_inputs = (torch.randint(0, 9, (9, 9)), torch.tensor(8))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

        test_inputs = (torch.randint(0, 50, (50, 50)), torch.tensor(12))
        eager_out = mod(*test_inputs)
        traced_out = traced_func(*test_inputs)
        self.assertNotWarn(
            lambda: traced_func(*test_inputs),
            "Shouldn't throw slicing related warn here",
        )
        self.assertEqual(eager_out, traced_out)

    def test_typeas_trace_check(self):
        # [FUNCTION DOC] Traces type_as with the trace checker enabled; only checks tracing
        # succeeds.
        a = torch.tensor([0.4], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)

        def f(x, y):
            # [FUNCTION DOC] HELPER: Casts the first argument to the type of the second.
            return x.type_as(y)

        trace = torch.jit.trace(f, (a, b))

    def test_trace_index(self):
        # [FUNCTION DOC] Traces a function indexing a tensor with an integer tensor and
        # compares the traced result with eager.
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0], dtype=torch.int64)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Indexes the first tensor with the second.
            return x[y]

        fn_traced = torch.jit.trace(
            fn,
            (
                x,
                y,
            ),
        )

        self.assertEqual(fn(x, y), fn_traced(x, y))

    # Backwards tracing was broken for indexing by a constant,
    # because it's internally implemented using as_strided,
    # and we attempted to trace its derivative (which is not
    # currently supported.)  It currently works because
    # slice() is now not marked as traceable.
    def test_trace_index_constant(self):
        # [FUNCTION DOC] Regression test: backward through a trace of constant-index access
        # works. Compares both result and gradient of the eager function and its traced
        # version.
        x = torch.tensor([0.4], requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first element along dim zero.
            return x[0]

        def run(f):
            # [FUNCTION DOC] HELPER: Calls the given function on the enclosing input and
            # returns the result with its gradient wrt that input.
            y = f(x)
            grad = torch.autograd.grad(y, x)[0].clone()
            return y, grad

        traced_fn = torch.jit.trace(fn, torch.ones(1))
        self.assertEqual(run(fn), run(traced_fn))

    def test_index_put(self):
        # [FUNCTION DOC] Traces a function doing masked in-place assignment and checks the
        # traced function matches eager on new data.
        ten = torch.zeros(3, 3)
        mask = torch.tensor(
            [[True, True, True], [True, False, False], [True, True, False]]
        )

        def test_fn(ten, mask):
            # [FUNCTION DOC] HELPER: Assigns a tensor of ones into the masked positions and
            # returns the tensor.
            ten[mask] = torch.ones(6)
            return ten

        traced_test_fn = torch.jit.trace(test_fn, (ten, mask))

        ten = torch.rand(3, 3)
        self.assertEqual(test_fn(ten, mask), traced_test_fn(ten, mask))

    def test_canonicalize_tensor_iterator(self):
        # [FUNCTION DOC] Traces a chain of four arithmetic ops and checks the graph has
        # exactly five integer constants (four right-hand operands plus the alpha argument
        # shared by add and sub).
        x = torch.randn(4, 4)

        def f(x):
            # [FUNCTION DOC] HELPER: Adds two, subtracts four, multiplies by six and divides
            # by eight.
            x = x + 2
            x = x - 4
            x = x * 6
            x = x / 8
            return x

        traced = torch.jit.trace(f, (x,))
        f(x)
        graph = traced.graph_for(x)
        # There should be 4 int constants for the right sides of operators, plus one
        # for the alpha argument for add and sub
        self.assertTrue(str(traced.graph_for(x)).count(": int = prim::Constant") == 5)

    @suppress_warnings
    def test_constant(self):
        # [FUNCTION DOC] Traces matmul with a freshly constructed constant tensor using
        # checkTrace with different check inputs. Warnings are suppressed.
        x = torch.randn(2, 2, requires_grad=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by a diagonal matrix created
            # inside the function.
            return x.matmul(torch.diag(torch.tensor([2.0, 2.0])))

        self.checkTrace(f, (x,), (torch.ones(2, 2, requires_grad=True),))

    def test_wrapped_number(self):
        # [FUNCTION DOC] Traces a function with no inputs that multiplies a Python float by
        # a ones tensor, with check_trace enabled; guards against wrapped-number promotion
        # issues in the checker.
        # Scalar's get converted to 'wrapped' tensors of default tensor type.
        # Wrapped tensors behave differently in certain promotion operations:
        # float_tensor * double -> float but wrapped_float * double -> double.
        # This can cause issues in check-trace if not handled correctly in
        # `aten::isclose()`.

        def foobar():
            # [FUNCTION DOC] HELPER: Multiplies a Python float scalar by a float ones tensor
            # and returns it.
            x = -10000.0
            result = x * torch.ones(1, dtype=torch.float)
            return result

        scripted = torch.jit.trace(foobar, (), check_trace=True)

    def test_inplace_transplant(self):
        # [FUNCTION DOC] Gets a trace graph for repeated in-place adds on a clone; after
        # dead code elimination checks via FileCheck there is one clone and two in-place
        # adds, then checks export and import round trip.
        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: Clones the input and adds two and three in place.
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y

        g, _ = torch.jit._get_trace_graph(fn, (x,))
        self.run_pass("dce", g)
        FileCheck().check_count("aten::clone", 1, exactly=True).check_count(
            "aten::add_", 2, exactly=True
        ).check_next("return").run(str(g))
        self.assertExportImport(g, (x,))

    def test_inplace_flags(self):
        # [FUNCTION DOC] Traces custom autograd Functions (one in-place, one regular) with
        # forced out-of-place; checks every node has an inplace attribute with values false,
        # true, true, false.
        class InplaceFn(Function):
            # [CLASS DOC] HELPER: Autograd Function that marks its input dirty and adds one
            # in place.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Marks the input as modified and adds one to it in
                # place.
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                # [FUNCTION DOC] HELPER: Passes the gradient through unchanged.
                return go

        class RegularFn(Function):
            # [CLASS DOC] HELPER: Autograd Function returning input plus one out of place.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns input plus one as a new tensor.
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                # [FUNCTION DOC] HELPER: Passes the gradient through unchanged.
                return go

        x = torch.tensor([0.0], requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies regular, in-place, in-place, regular functions
            # in order.
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y

        trace_graph, _ = torch.jit._get_trace_graph(fn, (x,), _force_outplace=True)
        self.run_pass("dce", trace_graph)
        ops = list(trace_graph.nodes())
        for op in ops:
            self.assertTrue(op.hasAttribute("inplace"))
        inplace_flags = [False, True, True, False]
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i("inplace"), is_inplace)

    def test_inplace_check(self):
        # [FUNCTION DOC] Tracing with forced out-of-place and trace checking disabled, then
        # running the result with a custom in-place Function must raise RuntimeError
        # mentioning the inplace Function.
        class MyInplaceFn(Function):
            # [CLASS DOC] HELPER: Autograd Function mutating its input in place.
            @staticmethod
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to the input in place, marks it dirty and
                # returns it.
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                # [FUNCTION DOC] HELPER: Passes the gradient through unchanged.
                return grad

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the in-place Function to the input.
            return MyInplaceFn.apply(x)

        x = torch.randn(5, 5)
        ge = torch.jit.trace(fn, (x,), _force_outplace=True, check_trace=False)
        with self.assertRaisesRegex(RuntimeError, "inplace MyInplaceFn"):
            ge(x)

    def test_force_outplace_check_fill(self):
        # [FUNCTION DOC] Forced out-of-place trace of creating an empty tensor and filling
        # it; traced output must equal eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates an empty tensor of the input shape and fills it
            # with seven in place.
            return torch.empty(x.shape).fill_(7)

        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def test_force_outplace_check_zero(self):
        # [FUNCTION DOC] Forced out-of-place trace of creating an empty tensor and zeroing
        # it; traced output must equal eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates an empty tensor of the input shape and zeros it
            # in place.
            return torch.empty(x.shape).zero_()

        x = torch.randn(10, 15)
        ft = torch.jit.trace(f, x, _force_outplace=True)
        self.assertEqual(f(x), ft(x))

    def do_trace_size(self, requires_grad):
        # [FUNCTION DOC] HELPER: Traces a view whose dims come from the input shape and
        # compares to eager on same and different shapes. Argument requires_grad sets
        # whether inputs need gradients.
        def fn(x):
            # [FUNCTION DOC] HELPER: Views the input with dims derived from its size and
            # shape.
            return x.view(x.shape[1] * 2, x.size(0), 2)

        x = torch.randn(5, 2, 4, requires_grad=requires_grad)
        y = torch.randn(4, 8, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def test_trace_size(self):
        # [FUNCTION DOC] Runs the size-dependent tracing check without gradients.
        self.do_trace_size(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_size_with_grad(self):
        # [FUNCTION DOC] Runs the size-dependent tracing check with gradient-requiring
        # inputs, which takes a different graph executor path.
        self.do_trace_size(True)

    def test_trace_numel(self):
        # [FUNCTION DOC] Traces a function returning the element count and checks it is
        # recomputed for new input sizes.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the number of elements of the input.
            return x.numel()

        x = torch.randn(2, 3, 4)
        y = torch.randn(4, 5, 6)

        traced_fn = torch.jit.trace(fn, x)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

    def do_trace_arange(self, requires_grad):
        # [FUNCTION DOC] HELPER: Traces three arange variants (shape-dependent, constant,
        # start and end from shape) and checks results against eager for original and new
        # shapes. Argument requires_grad sets whether inputs need gradients.
        def arange(x):
            # [FUNCTION DOC] HELPER: Returns arange of the first dimension size.
            return torch.arange(x.shape[0])

        def arange_scalar(x):
            # [FUNCTION DOC] HELPER: Returns arange of a constant length twelve.
            return torch.arange(12)

        def arange_start_end(x):
            # [FUNCTION DOC] HELPER: Returns arange from the first dimension size to that
            # size plus five.
            return torch.arange(start=x.shape[0], end=x.shape[0] + 5)

        x = torch.randn(5, 3, 2, requires_grad=requires_grad)
        y = torch.randn(8, 2, 4, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_arange = torch.jit.trace(arange, x)
        self.assertEqual(traced_arange(y), arange(y))
        self.assertEqual(traced_arange(x), arange(x))

        traced_arange_scalar = torch.jit.trace(arange_scalar, x)
        self.assertEqual(traced_arange_scalar(y), arange_scalar(y))
        self.assertEqual(traced_arange_scalar(x), arange_scalar(x))

        traced_arange_start_end = torch.jit.trace(arange_start_end, x)
        self.assertEqual(traced_arange_start_end(y), arange_start_end(y))
        self.assertEqual(traced_arange_start_end(x), arange_start_end(x))

    def test_trace_arange(self):
        # [FUNCTION DOC] Runs the arange tracing check without gradients.
        self.do_trace_arange(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_arange_with_grad(self):
        # [FUNCTION DOC] Runs the arange tracing check with gradient-requiring inputs.
        self.do_trace_arange(True)

    # Test that a trace of torch.full(x.shape) doesn't store the shape as a constant
    def test_trace_full_dynamic_shape(self):
        # [FUNCTION DOC] Checks tracing of full with a shape taken from the input does not
        # freeze the shape: output shape follows new inputs.
        def full_with_shape_like(x):
            # [FUNCTION DOC] HELPER: Returns a tensor of the input shape filled with two.
            return torch.full(x.shape, 2.0)

        x = torch.randn(3, 4)
        ge = torch.jit.trace(full_with_shape_like, example_inputs=x)
        y = torch.randn(2, 7)
        self.assertEqual(ge(y).shape, y.shape)
        self.assertEqual(ge(x).shape, x.shape)

    # Test that the trace of setitem doesn't store shapes as constants
    # Fix https://github.com/pytorch/pytorch/issues/43548
    def test_trace_slice_setitem_dynamic_shape(self):
        # [FUNCTION DOC] Regression test (issue 43548): tracing slice assignment does not
        # bake shapes in as constants; the trace is run on a larger input and compared to
        # eager.
        def slice_setitem(x, y):
            # [FUNCTION DOC] HELPER: Sets a column of x to y plus one and returns x.
            x[:, 2] = y + 1
            return x

        x = torch.randn(3, 4)
        traced = torch.jit.trace(slice_setitem, (x, x[:, 0]))
        x = torch.randn(10, 5)
        self.assertEqual(traced(x.clone(), x[:, 0]), slice_setitem(x.clone(), x[:, 0]))

    # Suppression: we are intentionally slicing a tensor, we don't care that it
    # will be constantified
    @suppress_warnings
    def do_trace_slice(self, requires_grad):
        # [FUNCTION DOC] HELPER: Traces slicing functions and compares results with eager
        # for original and new inputs; warnings are suppressed since constantified slices
        # are expected. Argument requires_grad sets whether inputs need gradients.
        def slice(x):
            # [FUNCTION DOC] HELPER: Collects four slices of the input with start and stop
            # positions depending on the loop index and returns them as a tuple.
            results = []
            for i in range(4):
                results.append(x[: x.size(0) - i, i : x.size(2), i:3])
            return tuple(results)

        def slice_select(x):
            # [FUNCTION DOC] HELPER: Collects four slices that select a fixed index in the
            # last dim and returns them as a tuple.
            results = []
            for i in range(4):
                results.append(x[:, i:, x.size(2) - 5])
            return tuple(results)

        x = torch.randn(5, 6, 7, requires_grad=requires_grad)
        y = torch.randn(7, 8, 9, requires_grad=requires_grad)

        # Check that it behaves as expected
        traced_slice = torch.jit.trace(slice, x)
        self.assertEqual(traced_slice(y), slice(y))
        self.assertEqual(traced_slice(x), slice(x))

        traced_slice_select = torch.jit.trace(slice_select, x)
        self.assertEqual(traced_slice_select(y), slice_select(y))
        self.assertEqual(traced_slice_select(x), slice_select(x))

    def test_trace_slice(self):
        # [FUNCTION DOC] Runs the slicing tracing check without gradients.
        self.do_trace_slice(False)

    # test the different graph_executor path that happens when
    # gradients are required and sizes are involved
    def test_trace_slice_with_grad(self):
        # [FUNCTION DOC] Runs the slicing tracing check with gradient-requiring inputs.
        self.do_trace_slice(True)

    def test_trace_casts(self):
        # [FUNCTION DOC] Traces several cast spellings; each trace must contain exactly one
        # to node and match eager results. Also traces tensor-to-tensor cast with different
        # shapes.
        casts = [
            lambda x: x.byte(),
            lambda x: x.float(),
            lambda x: x.cpu(),
            lambda x: x.to(device="cpu"),
            lambda x: x.to(dtype=torch.int64),
            lambda x: x.to(device="cpu", dtype=torch.float),
            lambda x: x.to(x),
        ]

        def assertContainsCast(trace):
            # [FUNCTION DOC] HELPER: Asserts the traced graph contains exactly one aten::to
            # node.
            self.assertEqual(
                sum(n.kind() == "aten::to" for n in trace.graph.nodes()), 1
            )

        for cast in casts:
            trace = torch.jit.trace(cast, torch.randn(2, 2))
            assertContainsCast(trace)
            x = torch.randn(2, 2)
            self.assertEqual(trace(x), cast(x))

        def to_tensor(x, y):
            # [FUNCTION DOC] HELPER: Casts the first tensor to the dtype and device of the
            # second.
            return x.to(y)

        to_tensor_trace = torch.jit.trace(
            to_tensor, (torch.randn(2, 2), torch.randn(1, 8))
        )
        assertContainsCast(to_tensor_trace)
        x, y = torch.randn(2, 2), torch.randn(1, 10)
        self.assertEqual(to_tensor_trace(x, y), to_tensor(x, y))

    @skipIfCompiledWithoutNumpy
    @skipIfCrossRef
    def test_trace_warn(self):
        # [FUNCTION DOC] Traces a function that triggers several tracer warnings (int
        # conversion, boolean test, float conversion, list conversion, numpy conversion,
        # iteration over a tensor). Records warnings, keeps only TracerWarning and asserts
        # each message in order.
        def fn(x):
            # [FUNCTION DOC] HELPER: Triggers the six conversions that produce tracer
            # warnings.
            int(x)  # Warning 1.
            y = x * 1
            if y:  # Warning 2.
                pass
            q = [x, x * 4]
            z = q[y]
            float(z)  # Warning 3.
            z.tolist()  # Warning 4.
            z.numpy()  # Warning 5.
            for _ in torch.ones(4, 4):  # Warning 6.
                pass
            return z + 4

        with warnings.catch_warnings(record=True) as warns:
            traced_fn = torch.jit.trace(fn, torch.tensor([1]))
        # torch.jit.trace emits a deprecation FutureWarning; keep only the
        # tracer warnings this test cares about.
        warns = [w for w in warns if w.category is torch.jit.TracerWarning]
        self.assertTrue(warns)
        warns = [str(w.message) for w in warns]
        self.assertIn("a Python integer", warns[0])
        self.assertIn("a Python boolean", warns[1])
        self.assertIn("a Python float", warns[2])
        self.assertIn("a Python list", warns[3])
        self.assertIn("a NumPy array", warns[4])
        self.assertIn("Iterating over", warns[5])

    def test_trace_tuple(self):
        # [FUNCTION DOC] Traces a function returning a nested tuple; checks graph has two
        # tuple constructions just before the return and the export and import round trip
        # works.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the input and a nested tuple of two products.
            return x, (x * y[1], x * y[0])

        x, y = torch.randn(2, 2), (torch.ones(2, 2), torch.randn(2, 2))
        traced_fn = torch.jit.trace(fn, (x, y))
        self.assertEqual(traced_fn(x, y), fn(x, y))
        # should be a tuple nested within another tuple
        FileCheck().check_count("prim::TupleConstruct", 2, exactly=True).check_next(
            "return"
        ).run(str(traced_fn.graph))
        self.assertExportImport(traced_fn.graph, (x, y))

    def test_trace_random(self):
        # [FUNCTION DOC] Traces a random normal op with trace checking off and compares to
        # eager output using a forked RNG state.
        def f(mean, std):
            # [FUNCTION DOC] HELPER: Samples a normal tensor from mean and std.
            return torch.normal(mean, std)

        traced = torch.jit.trace(
            f, (torch.zeros(2, 3), torch.ones(2, 3)), check_trace=False
        )
        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = f(mean, std)
        traced_output = traced(mean, std)
        self.assertEqual(output, traced_output)

    def test_trace_tensor_factory(self):
        # [FUNCTION DOC] Traces a function adding a freshly created ones tensor, for several
        # factory keyword settings (default, int dtype, out tensor, CUDA devices when
        # available) and checks the graph records the ones op rather than a constant.
        def run(**kwargs):
            # [FUNCTION DOC] HELPER: Builds a traced function using the given factory
            # keywords and verifies it with checkTrace and a graph string check.
            inputs_require_grads = kwargs.pop("inputs_require_grads", True)

            def fn(x):
                # [FUNCTION DOC] HELPER: Adds a ones tensor made with the captured keywords
                # to the input.
                return x + torch.ones(2, 3, **kwargs)

            input_kwargs = kwargs.copy()
            if "out" in input_kwargs:
                del input_kwargs["out"]
            input = torch.ones(2, 3, **input_kwargs)
            self.checkTrace(fn, (input,), inputs_require_grads=inputs_require_grads)
            # check we recorded 'ones' and did not just record a constant
            tfn = torch.jit.trace(fn, input)
            self.assertTrue("ones" in str(tfn.graph))

        run()
        run(dtype=torch.int, inputs_require_grads=False)
        run(out=torch.tensor([]))
        if RUN_CUDA:
            run(device="cuda:0")
        if RUN_CUDA_MULTI_GPU:
            run(device="cuda:1")

    def test_trace_indexed_assignment(self):
        # [FUNCTION DOC] Traces indexed assignment into a clone of the input.
        def stuff(x, y):
            # [FUNCTION DOC] HELPER: Clones the input, assigns y to the first row and
            # returns it.
            x = x.clone()
            x[0] = y
            return x

        example = torch.rand(3, 4)
        self.checkTrace(stuff, (example, example[0] + 1))

    # TODO: implement
    @unittest.expectedFailure
    def test_output_unflatten(self):
        # [FUNCTION DOC] Expected failure (marked as to be implemented): outputs of a traced
        # function should keep their nested tuple structure.
        """Check that outputs of traced functions retain the original structure and nesting"""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a nested tuple of tensors.
            return (
                x * 2,
                (
                    x**2,
                    x + 4,
                    (x + 2,),
                ),
                x * 4,
            )

        self.checkTrace(fn, (torch.randn(2, 2),))

    def test_input_flatten(self):
        # [FUNCTION DOC] Checks that a nested tuple input is flattened for tracing.
        """Check that inputs to traced functions are flattened"""

        def fn(x, t):
            # [FUNCTION DOC] HELPER: Unpacks the pair and returns the product of all three
            # tensors.
            y, z = t
            return x * y * z

        inputs = (torch.randn(1), (torch.randn(1), torch.randn(1)))
        self.checkTrace(fn, inputs)

    def test_input_dict_empty(self):
        # [FUNCTION DOC] Tracing with an empty dict input must raise RuntimeError.
        def test(d):
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        with self.assertRaises(RuntimeError):
            self.checkTrace(test, {})

    def test_input_dict_remembers_keys(self):
        # [FUNCTION DOC] The traced module remembers which keys the dict input had: changed
        # values work, extra keys are ignored, missing key x raises RuntimeError.
        """Check that the trace remembers which keys were in a dict input"""

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the x entry of a dict input.
            def forward(self, dict_input):
                # [FUNCTION DOC] HELPER: Returns the value stored under key x.
                return dict_input["x"]

        input_1 = {"x": torch.tensor(1)}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))
        self.assertEqual(m_traced(input_1), torch.tensor(1))

        # should work to change the values and not the keys
        input_same_key_different_value = {"x": torch.tensor(2)}
        self.assertEqual(m_traced(input_same_key_different_value), torch.tensor(2))

        # error to use something that doesn't have `x`
        input_different_key = {"y": torch.tensor(3)}
        with self.assertRaises(RuntimeError):
            m_traced(input_different_key)

        # it's okay to have additional elements in the dictionary, so long as 'x' is there
        input_additional_key = {"x": torch.tensor(4), "y": torch.tensor(3)}
        self.assertEqual(m_traced(input_additional_key), torch.tensor(4))

    def test_input_dict_insertion_order(self):
        # [FUNCTION DOC] Dict input access in a trace is independent of insertion order.
        """Check that dictionary access doesn't care about insertion order"""

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the x and y entries of a dict input.
            def forward(self, dict_input):
                # [FUNCTION DOC] HELPER: Returns the values under keys x and y as a tuple.
                return dict_input["x"], dict_input["y"]

        input_x_then_y = {}
        input_x_then_y["x"] = torch.tensor(1)
        input_x_then_y["y"] = torch.tensor(2)

        m = TestModule()
        m_traced = torch.jit.trace(m, (input_x_then_y,))

        self.assertEqual(m_traced(input_x_then_y), (torch.tensor(1), torch.tensor(2)))

        input_y_then_x = {}
        input_y_then_x["y"] = torch.tensor(4)
        input_y_then_x["x"] = torch.tensor(3)

        self.assertEqual(m_traced(input_y_then_x), (torch.tensor(3), torch.tensor(4)))

    def test_input_dict_recursive(self):
        # [FUNCTION DOC] Traced module accepts nested dict input with new values.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reading a nested value from a dict of dicts.
            def forward(self, dict_input):
                # [FUNCTION DOC] HELPER: Returns the tensor stored under key one inside the
                # dict under key x.
                return dict_input["x"][1]

        input_1 = {"x": {1: torch.tensor(1)}}
        m = TestModule()
        m_traced = torch.jit.trace(m, (input_1,))

        input_2 = {"x": {1: torch.tensor(2)}}
        self.assertEqual(m_traced(input_2), torch.tensor(2))

    def test_input_dict_checkTrace_mut(self):
        # [FUNCTION DOC] checkTrace on a function mutating a dict value in place.
        def test(d):
            # [FUNCTION DOC] HELPER: Applies tanh in place to the x entry and returns it.
            d["x"].tanh_()
            return d["x"]

        inputs = {"x": torch.rand(3, 4), "y": torch.rand(3, 4)}
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_dict_unify(self):
        # [FUNCTION DOC] checkTrace with a dict input holding tensors of different dtypes.
        def test(d):
            # [FUNCTION DOC] HELPER: Returns the int and float entries.
            return d["int"], d["float"]

        inputs = {
            "int": torch.ones((2, 2), dtype=torch.int32),
            "float": torch.ones((2, 2), dtype=torch.float32),
        }
        self.checkTrace(test, (inputs,), inputs_require_grads=False)

    def test_input_tuple_of_dicts(self):
        # [FUNCTION DOC] checkTrace on a tuple input containing dicts of dicts.
        def test(t):
            # [FUNCTION DOC] HELPER: Returns the nested y tensor from the dict in the first
            # tuple slot.
            d = t[0]
            return d["x"]["y"]

        inputs = {"x": {"y": torch.rand(2, 3)}}
        self.checkTrace(test, ((inputs, inputs),), allow_unused=True)

    def test_input_dict_of_dicts(self):
        # [FUNCTION DOC] checkTrace on a dict of dicts with different nested shapes that
        # must be unified.
        def test(d):
            # [FUNCTION DOC] HELPER: Returns the nested y tensor under x.
            return d["x"]["y"]

        nested_input = {"y": torch.rand(2, 3)}
        unified_nested = {"y": torch.rand(3, 2)}
        inputs = {"x": nested_input, "force_unify": unified_nested}
        self.checkTrace(test, (inputs,), allow_unused=True)

    def test_input_dict_of_lists(self):
        # [FUNCTION DOC] checkTrace on a dict whose value is a list of tensors.
        def test(d):
            # [FUNCTION DOC] HELPER: Returns the first tensor of the list under x.
            return d["x"][0]

        inputs = {"x": [torch.rand(3, 2)]}
        self.checkTrace(test, (inputs,))

    def test_input_list_toplevel_flatten(self):
        # [FUNCTION DOC] checkTrace on a function with a top-level list of tensors passed as
        # separate arguments.
        def test(t1, t2):
            # [FUNCTION DOC] HELPER: Adds the two tensors.
            return torch.add(t1, t2)

        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        self.checkTrace(test, inputs)

    def test_input_list_toplevel_flatten_direct(self):
        # [FUNCTION DOC] Traces a module directly with a list of tensors as example inputs.
        class Test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding its two inputs.
            def forward(self, t1, t2):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return torch.add(t1, t2)

        inputs = [torch.ones(2, 2), torch.rand(2, 2)]
        torch.jit.trace(Test(), inputs)

    def test_input_list_of_tuples(self):
        # [FUNCTION DOC] checkTrace on a list of tuples input.
        def test(l):
            # [FUNCTION DOC] HELPER: Returns the first tensor of the first tuple.
            return l[0][0]

        inputs = [(torch.ones(2, 2),)]
        self.checkTrace(test, (inputs,))

    def test_input_dict_empty_list(self):
        # [FUNCTION DOC] A dict input with an empty list value must raise RuntimeError
        # mentioning list tracing.
        def test(d):
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        inputs = {1: []}
        with self.assertRaisesRegex(RuntimeError, "List trace"):
            self.checkTrace(test, (inputs,))

    def test_input_list_mixed_type(self):
        # [FUNCTION DOC] A list input mixing a tensor and a tuple must raise RuntimeError
        # about consistent types.
        def test(d):
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        inputs = [torch.rand(2, 3), (torch.ones(2), torch.ones(2))]
        with self.assertRaisesRegex(RuntimeError, "consistent"):
            self.checkTrace(test, (inputs,))

    def test_conv(self):
        # [FUNCTION DOC] Gets a trace graph of a Conv2d without bias, rebuilds a function
        # from it and compares the result with the recorded outputs.
        x = torch.ones(20, 16, 50, 40)
        g, outputs, inputs = torch.jit._get_trace_graph(
            nn.Conv2d(16, 13, 3, bias=False), x, return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_max_pool(self):
        # [FUNCTION DOC] Traces max pooling plus a constant and checks the graph contains
        # max_pool2d and the result matches eager.
        x = torch.rand(20, 16, 10, 10)

        def max_pool2d(x):
            # [FUNCTION DOC] HELPER: Applies 2x2 max pooling and adds two.
            return F.max_pool2d(x, 2) + 2

        trace = torch.jit.trace(max_pool2d, (x))
        graph = trace.graph_for(x)
        FileCheck().check("aten::max_pool2d(").run(graph)
        self.assertEqual(max_pool2d(x), trace(x))

    def test_nested_inplace(self):
        # [FUNCTION DOC] Trace graph of an in-place functional threshold; checks results, a
        # threshold_ node in the graph and the export and import round trip.
        x = torch.randn(2, 2)
        g, outputs, inputs = torch.jit._get_trace_graph(
            lambda x: F.threshold(x, 0, 0, inplace=True), (x,), return_inputs=True
        )
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        FileCheck().check("threshold_").run(str(g))
        self.assertExportImport(g, (x,))

    def test_repeated_input(self):
        # [FUNCTION DOC] checkTrace with the same tensor passed twice; the resulting graph
        # has three inputs (two plus the module self argument).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of its arguments.
            return a + b

        ge = self.checkTrace(fn, [torch.randn(2, 2)] * 2)
        inputs = set(ge.graph.inputs())
        # three instead of 2 because the export/import in checkTrace adds a
        # `self` module argument
        self.assertTrue(len(inputs) == 3)

    def test_repeated_output(self):
        # [FUNCTION DOC] checkTrace on a function returning the same value twice; both tuple
        # elements must be the same graph value.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum twice in a tuple.
            z = a + b
            return z, z

        ge = self.checkTrace(fn, [torch.randn(2, 2) for _ in range(2)])
        tuple_output = list(ge.graph.outputs())[0]
        tuple_inputs = list(tuple_output.node().inputs())
        self.assertTrue(tuple_inputs[0] == tuple_inputs[1])

    def test_inplace_copy(self):
        # [FUNCTION DOC] Trace graph of a function copying the input into zeros with copy_,
        # with gradients required; checks dead code elimination, execution and export and
        # import.
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Creates zeros of the input size and copies the input
            # into it.
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = torch.jit._get_trace_graph(f, (x,), return_inputs=True)
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))

    def test_inplace_copy_force_outplace(self):
        # [FUNCTION DOC] Same as the in-place copy test but with forced out-of-place; also
        # checks an expand_as appears in the graph.
        x = torch.randn(4, 4, requires_grad=True)

        def f(x):
            # [FUNCTION DOC] HELPER: Creates zeros of the input size and copies the input
            # into it.
            out = torch.zeros(x.size())
            out.copy_(x)
            return out

        g, outputs, inputs = torch.jit._get_trace_graph(
            f, (x,), return_inputs=True, _force_outplace=True
        )
        self.run_pass("dce", g)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))
        self.assertExportImport(g, (x,))
        FileCheck().check("expand_as").run(str(g))

    def test_shared_param(self):
        # [FUNCTION DOC] A module whose two attributes share one Parameter gets a trace
        # graph with only two inputs (self-less parameters deduplicated); checks mul and add
        # appear.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two names for one Parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the same Parameter under two attribute
                # names.
                super().__init__()
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x times a plus b.
                return x * self.a + self.b

        m = MyModule()
        g, _ = torch.jit._get_trace_graph(m, (torch.randn(2, 2),))
        self.run_pass("dce", g)
        self.assertEqual(len(list(g.inputs())), 2)
        FileCheck().check("mul").check("add").run(str(g))

    def run_ge_tests(self, optimize, use_cuda):
        # [FUNCTION DOC] HELPER: Runs checkTrace on a collection of functions (including
        # identity, unused inputs, dropped outputs and autograd fallback) under profiling
        # mode and with the given optimized_execution setting. Arguments: optimize flag and
        # whether to move random inputs to CUDA.
        with enable_profiling_mode_for_profiling_tests():
            with torch.jit.optimized_execution(optimize):

                def rand(*args):
                    # [FUNCTION DOC] HELPER: Returns a random float tensor, moved to CUDA
                    # when requested.
                    t = torch.rand(*args).float()
                    if use_cuda:
                        t = t.cuda()
                    return t

                self.checkTrace(
                    lambda a, b: a * b + b, [rand(1), rand(1)], [rand(2, 3), rand(2, 3)]
                )
                # trivial identity
                self.checkTrace(lambda a, b: (b, a), [rand(1), rand(1)])

                def foo(a):
                    # [FUNCTION DOC] HELPER: Returns the square squared and four times the
                    # square.
                    t = a * a
                    return t * t, 4 * t

                self.checkTrace(foo, [rand(1)])
                # unused input
                self.checkTrace(
                    lambda a, b: a * a, [rand(1), rand(1)], allow_unused=True
                )
                # test outputs that do not get used in grad
                self.checkTrace(foo, [rand(1)], drop=1)
                # test autograd fallback
                self.checkTrace(
                    lambda a, b: a * b / (a - 2 * b) + b, [rand(1), rand(1)]
                )

    def test_ge_unoptimized(self):
        # [FUNCTION DOC] Runs the graph executor tests with optimization off on CPU.
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_SANDCASTLE, "NYI: fuser support for Sandcastle")
    @enable_cpu_fuser
    def test_ge_optimized(self):
        # [FUNCTION DOC] Runs the graph executor tests with optimization on CPU under the
        # CPU fuser; skipped on Sandcastle.
        with enable_profiling_mode_for_profiling_tests():
            self.run_ge_tests(True, False)

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_ge_cuda(self):
        # [FUNCTION DOC] Runs the graph executor tests with optimization on CUDA; skipped
        # without CUDA.
        self.run_ge_tests(True, True)

    # more manual test of graph executor that can be used as a scratchpad
    def test_ge(self):
        # [FUNCTION DOC] Manual graph executor check: traces a function, computes first and
        # second order gradients through the trace and compares them with eager.
        def foo(a, b):
            # [FUNCTION DOC] HELPER: Returns a times b divided by a minus b, plus b.
            return a * b / (a - b) + b

        V = Variable
        a, b = V(torch.rand(1)), V(torch.rand(1))
        ge = torch.jit.trace(foo, (a, b))
        a, b = (
            V(torch.rand(1), requires_grad=True),
            V(torch.rand(1), requires_grad=True),
        )
        (r,) = ge(a, b)
        da, db = torch.autograd.grad(r + 3, [a, b], create_graph=True)

        l2 = da * db + db * db
        g2result = torch.autograd.grad(l2, [da, db])

        r = foo(a, b)
        da2, db2 = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = da2 * db2 + db2 * db2
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def test_trace_annotation(self):
        # [FUNCTION DOC] Checks the private trace decorator: the decorated function is a
        # traced function giving the same result as eager.
        @_trace(torch.rand(1))
        def foo(a):
            # [FUNCTION DOC] HELPER: Returns three times the input as repeated additions.
            return a + a + a

        x = torch.randn(5, 5)
        self.assertEqual(foo(x), x + x + x)

    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    # By default, on Ampere or later GPUs, nn.Linear computes float tensors at TF32 precision.
    # We want float tensors to be computed at full precision in order to use the default precision
    @with_tf32_off
    def test_traced_module_cuda(self):
        # [FUNCTION DOC] CUDA-only. Traces a Sequential-based model and exercises the traced
        # module API: repr, attribute access errors, assigning and deleting parameters,
        # calling submodule errors, cuda, cpu and to conversions with output comparison, and
        # state_dict round trips. Disables TF32.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Module with a Sequential of Linear and Sigmoid layers
            # applied in a loop with a residual add.
            def __init__(self, num_features, num_layers):
                # [FUNCTION DOC] HELPER: Builds a Sequential of alternating Linear and
                # Sigmoid layers and records the layer count.
                super().__init__()
                self.num_layers = num_layers
                layers = [
                    [nn.Linear(num_features, num_features), nn.Sigmoid()]
                    for _ in range(num_layers)
                ]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: For each layer index, applies that layer and adds
                # the result to the running value.
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        model = Model(5, 3)
        x = torch.randn(2, 5)
        traced_model = torch.jit.trace(model, x)

        # We're missing some attributes these modules had initially. Make sure we can
        # still get the __repr__()
        model.__repr__()

        # XXX: indexing sequentials is broken
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # All attributes that aren't parameters should raise
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        linear_submodule.weight = nn.Parameter(
            torch.randn(linear_submodule.weight.shape)
        )
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # Submodules can't be called
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # Type casts
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to("cuda")
        cuda_out = traced_model(x.float().cuda())
        traced_model.to("cpu")
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.to(torch.get_default_dtype())

        # state_dict + load_state_dict
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    @unittest.skipIf(not RUN_CUDA, "uses cuda")
    def test_type_same_device(self):
        # [FUNCTION DOC] CUDA-only. A traced type conversion using a dtype attribute must
        # not mention device in the generated code.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module converting input to a stored dtype.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a dtype attribute.
                super().__init__()
                self.dtype = torch.float16

            def forward(self, x=None):
                # [FUNCTION DOC] HELPER: Converts the input to the stored dtype.
                h = x.type(self.dtype)
                return h

        a = Model()
        b = torch.jit.trace(
            a, example_inputs=(torch.ones([1], device=torch.device("cuda")),)
        )
        FileCheck().check_not("device").run(b.code)

    def test_export_no_reorder(self):
        # [FUNCTION DOC] Traces a function, exports and imports it, and checks outputs and
        # gradients are identical, showing export does not reorder ops.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Returns a times b divided by a minus twice b, plus b.
            return a * b / (a - 2 * b) + b

        recording_inputs = [
            torch.tensor(
                [0.55619788169860839844], dtype=torch.float32, requires_grad=True
            ),
            torch.tensor(
                [0.25947844982147216797], dtype=torch.float32, requires_grad=True
            ),
        ]

        ge1 = torch.jit.trace(func, recording_inputs)
        ge2 = self.getExportImportCopy(ge1)

        outputs_ge1 = ge1(*recording_inputs)
        outputs_ge2 = ge2(*recording_inputs)

        grad_ge1 = torch.autograd.grad(outputs_ge1, recording_inputs)
        grad_ge2 = torch.autograd.grad(outputs_ge2, recording_inputs)
        self.assertTrue(outputs_ge1 == outputs_ge2)
        self.assertTrue(grad_ge1 == grad_ge2)

    def test_python_function(self):
        # [FUNCTION DOC] Traces through a custom autograd Function with a function-style
        # trace decorator, then calls the traced function on different inputs including one
        # requiring grad.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Autograd Function adding one.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns input plus one.
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Passes the gradient through.
                return grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds two, applies the Function and adds three.
            return MyFn.apply(x + 2) + 3

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_python_function_tup(self):
        # [FUNCTION DOC] Same as the previous test but the Function returns a tuple of two
        # tensors.
        class MyFn(Function):
            # [CLASS DOC] HELPER: Autograd Function returning input plus one and input minus
            # one.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns the pair of shifted tensors.
                return x + 1, x - 1

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns the gradient for both outputs.
                return grad_output, grad_output

        @_trace(torch.zeros(2))
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the Function and returns the sum of both
            # outputs plus three.
            a, b = MyFn.apply(x + 2)
            return a + b + 3

        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)

    def test_trace_detach(self):
        # [FUNCTION DOC] Traced detach: graph contains matmul and detach, and the traced
        # result neither requires grad nor has a grad_fn.
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Returns the detached matrix product.
            return torch.matmul(x, w).detach()

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        FileCheck().check("matmul").check("detach").run(str(traced.graph))
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_redispatch(self):
        # [FUNCTION DOC] Tracing must redispatch detach to the lower kernel so the
        # eager-visible requires_grad is false after detach; trace checking is off because
        # the checker would run under no_grad.
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Computes a matmul, asserts it requires grad, detaches
            # it, asserts it no longer does (raising AssertionError otherwise).
            y = torch.matmul(x, w)
            if not y.requires_grad:
                raise AssertionError("Expected y.requires_grad to be True")
            y = y.detach()
            # Make sure trace kernel redispatches to the right lower kernel.
            if y.requires_grad:
                raise AssertionError(
                    "Expected y.requires_grad to be False after detach"
                )
            return y

        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@torch.no_grad()` and break assert.
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_detach_inplace(self):
        # [FUNCTION DOC] Traced in-place detach: graph has matmul and detach and the result
        # has no grad requirement or grad_fn.
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Computes a matmul and detaches it in place.
            y = torch.matmul(x, w)
            y.detach_()
            return y

        traced = torch.jit.trace(foo, (torch.rand(3, 4), torch.rand(4, 5)))

        FileCheck().check("matmul").check("detach(").run(str(traced.graph))
        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        traced_result = traced(x, w)
        self.assertEqual(foo(x, w), traced_result)
        self.assertFalse(traced_result.requires_grad)
        self.assertIsNone(traced_result.grad_fn)

    def test_trace_detach_inplace_redispatch(self):
        # [FUNCTION DOC] Same redispatch check as the detach test but for in-place detach.
        def foo(x, w):
            # [FUNCTION DOC] HELPER: Computes a matmul, asserts it requires grad, detaches
            # in place, asserts it no longer does (raising AssertionError otherwise).
            y = torch.matmul(x, w)
            if not y.requires_grad:
                raise AssertionError("Expected y.requires_grad to be True")
            y.detach_()
            # Make sure trace kernel redispatches to the right lower kernel.
            if y.requires_grad:
                raise AssertionError(
                    "Expected y.requires_grad to be False after detach_"
                )
            return y

        x, w = torch.rand(3, 4), torch.rand(4, 5, requires_grad=True)
        # With `check_trace=True` it will run with `@torch.no_grad()` and break assert.
        torch.jit.trace(foo, (x, w), check_trace=False)

    def test_trace_slice_full_dim(self):
        # [FUNCTION DOC] A trace slicing the full first dim must generalize to an input with
        # different size.
        def foo(x):
            # [FUNCTION DOC] HELPER: Slices the first five rows of column zero and adds one.
            return x[0:5, 0] + 1.0

        traced = torch.jit.trace(foo, (torch.rand(5, 4),))
        test_x = torch.rand(6, 3)
        self.assertEqual(foo(test_x), traced(test_x))

    def test_trace_dict_input(self):
        # [FUNCTION DOC] checkTrace on a model that passes a dict of inputs to a submodule
        # and returns one entry of the dict output.
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Foo submodule and passing it a dict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Foo submodule.
                super().__init__()
                self.foo = Foo()

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Builds a dict of the two inputs, calls the
                # submodule and returns its a entry.
                return self.foo({"a": a, "b": b})["a"]

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a dict and returning a dict of the product.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a dict with key a holding the product of
                # the a and b entries.
                return {"a": x["a"] * x["b"]}

        x = (torch.rand(3), torch.rand(3))
        model = Bar()
        self.checkTrace(model, x)

    def test_trace_dict_output(self):
        # [FUNCTION DOC] Dict outputs: default strict tracing raises RuntimeError about a
        # dict at the output; with strict false the string-keyed and tensor-keyed dict
        # outputs are traced and match eager.
        class TraceDictStrTensor(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dict keyed by strings.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a dict mapping a and b to the inputs.
                return {"a": a, "b": b}

        class TraceDictTensorTensor(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dict keyed by tensors.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a dict mapping each input to the other.
                return {a: b, b: a}

        x = (torch.rand(3), torch.rand(3))
        with self.assertRaisesRegex(RuntimeError, r"Encountering a dict at the output"):
            torch.jit.trace(TraceDictStrTensor(), x)

        traced_dict_str_mod = torch.jit.trace(TraceDictStrTensor(), x, strict=False)
        self.assertEqual(traced_dict_str_mod(*x), {"a": x[0], "b": x[1]})

        traced_dict_tensor_mod = torch.jit.trace(
            TraceDictTensorTensor(), x, strict=False
        )
        self.assertEqual(traced_dict_tensor_mod(*x), {x[0]: x[1], x[1]: x[0]})

    def test_trace_with_tensor_list_output(self):
        # [FUNCTION DOC] A function returning a list of tensors yields a TracerWarning about
        # incorrect traces in strict mode; with strict false it traces and matches eager.
        def f():
            # [FUNCTION DOC] HELPER: Returns a list of two zero tensors.
            return [torch.zeros(1), torch.zeros(5)]

        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            torch.jit.trace(f, [])
        traced_non_strict_f = torch.jit.trace(f, [], strict=False)
        self.assertEqual(traced_non_strict_f(), f())

    def test_trace_with_number_list_output(self):
        # [FUNCTION DOC] Tracing a function returning a list of numbers must raise
        # RuntimeError saying only tensors can be output.
        def f():
            # [FUNCTION DOC] HELPER: Returns a list of two ints.
            return [1, 5]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_tensor_list_output(self):
        # [FUNCTION DOC] Tracing a function returning nested lists of tensors must raise
        # RuntimeError saying only tensors can be output.
        def f():
            # [FUNCTION DOC] HELPER: Returns nested lists of zero tensors.
            return [[torch.zeros(1)], [torch.zeros(5)]]

        with self.assertRaisesRegex(
            RuntimeError, r"Only tensors.+can be output from traced functions"
        ):
            traced_f = torch.jit.trace(f, [])

    def test_trace_with_nested_strided_tensor_output(self):
        # [FUNCTION DOC] Traces a function that builds a nested tensor via a scripted helper
        # and applies cos; the trace must work on other inputs than the example.
        @torch.jit.script
        def nt_construct(values, kv_lengths):
            # [FUNCTION DOC] HELPER: Scripted: splits values by the given lengths and builds
            # a nested tensor from the pieces.
            kv_lengths_list: List[int] = kv_lengths.tolist()
            return torch._nested_tensor_from_tensor_list(
                list(values.split(kv_lengths_list, dim=0)), None, None, None, None
            )

        def f(x, offsets):
            # [FUNCTION DOC] HELPER: Derives lengths from offsets, builds the nested tensor
            # and applies cos.
            kv_lengths = offsets[1:] - offsets[:-1]
            return nt_construct(x, kv_lengths).cos()

        x = torch.rand(5, 4)
        offsets = torch.tensor([0, 2, 5])
        ref = f(x, offsets)
        f_t = torch.jit.trace(f, (x, offsets))
        res = f_t(x, offsets)
        self.assertEqual(ref, res)
        x2 = torch.rand((8, 4))
        offsets2 = torch.tensor([0, 2, 4, 8])
        self.assertEqual(f(x2, offsets2), f_t(x2, offsets2))

    def test_trace_variable_instantiation(self):
        # [FUNCTION DOC] Tracing functions that wrap a tensor in Variable must work and
        # match eager.
        def random_foo(x):
            # [FUNCTION DOC] HELPER: Wraps the input in Variable, adds one and wraps again.
            return Variable(Variable(x) + 1.0)

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        x = torch.rand(5, 6)
        self.assertEqual(random_foo(x), random_foo_traced(x))

    def test_trace_slice_expr_complete_type(self):
        # [FUNCTION DOC] A scripted function slicing the output of a traced function must
        # work (the output type is complete).
        def random_foo(x):
            # [FUNCTION DOC] HELPER: Adds one to the input.
            return x + 1.0

        random_foo_traced = torch.jit.trace(random_foo, (torch.rand(3, 4),))

        @torch.jit.script
        def random_bar(x):
            # [FUNCTION DOC] HELPER: Scripted: calls the traced function and slices the
            # first row.
            return random_foo_traced(x)[0:1]

        x = torch.rand(3, 4)
        self.assertEqual(random_bar(x), (x + 1)[0:1])

    def test_trace_inline_shape(self):
        # [FUNCTION DOC] Checks size peephole behavior: a scripted size-based function
        # traced, still correct on new sizes; and a traced call of a scripted function using
        # the device keeps the device node in the inlined graph.
        # testing peephole optimization of size is turned into a constant
        # in script fn

        @torch.jit.script
        def tensor_size(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Scripted: returns a one-element tensor holding the
            # first dim size.
            return torch.tensor([x.size()[0]])

        self.assertEqual(
            tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )

        traced_tensor_size = torch.jit.trace(
            tensor_size,
            torch.rand(
                7,
            ),
        )

        self.assertEqual(
            traced_tensor_size(
                torch.rand(
                    15,
                )
            ),
            torch.tensor([15]),
        )

        @torch.jit.script
        def use_device(x):
            # [FUNCTION DOC] HELPER: Scripted: zeros shaped like x on the device of x.
            return torch.zeros_like(x, device=x.device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Calls the scripted device function.
            return use_device(x)

        traced_tensor_size = torch.jit.trace(
            foo,
            torch.rand(
                7,
            ),
        )
        self.run_pass("inline", traced_tensor_size.graph)
        FileCheck().check("prim::device").run(traced_tensor_size.graph)

    def test_trace_save(self):
        # [FUNCTION DOC] Saves a traced function to a temporary file, loads it and compares
        # outputs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds two to the input.
            return x + 2

        def check(func):
            # [FUNCTION DOC] HELPER: Saves the given traced function to a temporary file,
            # loads it and asserts equal output for a random input.
            with TemporaryFileName() as fname:
                func.save(fname)
                loaded = torch.jit.load(fname)
                input = torch.randn(2, 2)
                self.assertEqual(func(input), loaded(input))

        out = torch.jit.trace(fn, (torch.ones(2, 2),))
        check(out)

    def test_trace_optioanl_dtype(self):
        # [FUNCTION DOC] Traces a module returning arange of five with no inputs and calls
        # allclose against eager.
        class Test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning arange of five.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns arange of five.
                return torch.arange(5)

        traced = torch.jit.trace(Test(), ())
        torch.allclose(traced(), Test()())

    def test_trace_save_load_copy(self):
        # [FUNCTION DOC] Saves a traced Conv module to bytes, loads it and checks copy and
        # deepcopy of the loaded module work.
        class Test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Conv2d.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution.
                return self.conv(x)

        traced = torch.jit.trace(Test(), torch.rand(1, 3, 224, 224))
        buffer = io.BytesIO()
        torch.jit.save(traced, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        # should work
        copy.copy(loaded)
        copy.deepcopy(loaded)

    def test_trace_export_fns(self):
        # [FUNCTION DOC] Exported getstate and setstate methods on a module must be present
        # in the traced module and in its export and import copy.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with exported getstate and setstate methods.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets integer attribute a to three.
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: Exported: returns a tuple of three and the training
                # flag.
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                # [FUNCTION DOC] HELPER: Exported: restores a and the training flag from the
                # state tuple.
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the attribute to the input.
                return x + self.a

        f = Foo()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            # [FUNCTION DOC] HELPER: Asserts all expected method names exist on the module.
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced)

        imported = self.getExportImportCopy(traced)
        check(imported)

    def test_trace_export_fns_recursive(self):
        # [FUNCTION DOC] Same method presence check but for a submodule inside a wrapper,
        # plus a module with an exported method whose forward is traceable only, to check
        # exported methods are scripted while forwards stay traced.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with exported getstate and setstate methods.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets integer attribute a to three.
                super().__init__()
                self.a = 3

            @torch.jit.export
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: Exported: returns a tuple of three and the training
                # flag.
                return (3, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                # [FUNCTION DOC] HELPER: Exported: restores a and the training flag from the
                # state tuple.
                self.a = state[0]
                self.training = state[1]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the attribute to the input.
                return x + self.a

        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Foo submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo submodule.
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the submodule.
                return self.foo(x)

        f = Wrapper()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["__getstate__", "__setstate__"]

        def check(mod):
            # [FUNCTION DOC] HELPER: Asserts all expected method names exist on the module.
            self.assertTrue(
                all(name in mod._c._method_names() for name in expected_names)
            )

        check(traced.foo)

        imported = self.getExportImportCopy(traced)
        check(imported.foo)

        # Note that Bar's forward can only be traced, but not scripted
        class Bar(nn.Module):
            # [CLASS DOC] HELPER: Module with an exported method and a forward that uses a
            # lambda (not scriptable, only traceable).
            @torch.jit.export
            def addTwo(self, x):
                # [FUNCTION DOC] HELPER: Exported: adds two.
                return x + 2

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies a lambda that adds one.
                return (lambda a: a + 1)(input)  # noqa: PLC3002

        # When tracing Bar as a submodule, we only want to script the
        # exported methods, and we want to keep the forwards still
        # being traced.
        class WrapperExports(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an exported method wrapping a Bar submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Bar submodule.
                super().__init__()
                self.bar = Bar()

            @torch.jit.export
            def addOne(self, x):
                # [FUNCTION DOC] HELPER: Exported: adds one.
                return x + 1

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the submodule.
                return self.bar(x)

        f = WrapperExports()

        traced = torch.jit.trace(f, (torch.rand(3, 4),))
        expected_names = ["addOne"]
        check(traced)

    def test_trace_autograd_function(self):
        # [FUNCTION DOC] Tracing a module that calls a custom autograd Function must
        # succeed.
        class TestFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function negating its input.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Negates the input.
                return torch.neg(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Negates the gradient.
                return torch.neg(grad_output)

        class TracedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu to the Function result.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the Function output.
                return torch.relu(TestFunc.apply(x))

        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a TracedModule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the inner module.
                super().__init__()
                self.tm = TracedModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner module.
                return self.tm(x)

        traced = torch.jit.trace(Wrapper(), (torch.rand(3, 4),))

    def test_trace_multi_output_function(self):
        # [FUNCTION DOC] Traces a custom Function with two outputs that swaps its inputs,
        # prints the graph, checks the expected output schema string is in the graph and
        # that the outputs equal swapped inputs.
        # An autograd.Function with two outputs.
        # It swaps inputs so we can check if shape
        # handling is correct in TorchScript.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function returning its two inputs swapped.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Returns the inputs in swapped order.
                return y, x

            @staticmethod
            def backward(ctx, du, dv):
                # [FUNCTION DOC] HELPER: Returns the gradients swapped.
                return dv, du

        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu to both inputs and passing them to
            # the Function.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies relu to both inputs and returns the
                # Function output pair.
                x = x.relu()
                y = y.relu()
                z = Foo.apply(x, y)
                return z

        x = torch.rand(3, 2, dtype=torch.double)
        y = torch.rand(1, 2, dtype=torch.double)

        # Generate JIT IR.
        traced = torch.jit.trace(Bar(), (x, y))
        print(traced.graph)

        # Expected output schema of the custom autograd.Function.
        schema = (
            "(Double(1, 2, strides=[2, 1], requires_grad=0, device=cpu), "
            "Double(3, 2, strides=[2, 1], requires_grad=0, device=cpu)) "
            "= ^Foo"
        )

        # See if expected schema exists.
        FileCheck().check(schema).run(traced.graph)

        # Also examine if the graph is runnable and produces
        # the right result.
        u, v = traced(x, y)
        self.assertEqual(u, y)
        self.assertEqual(v, x)

    def test_interpolate_trace(self):
        # [FUNCTION DOC] Traces a conv plus bilinear upsampling; the trace must not fail and
        # must generalize to another spatial size, i.e. scale factors are not baked in.
        class test(nn.Module):
            # [CLASS DOC] HELPER: Module applying a convolution then bilinear interpolation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d.
                super().__init__()
                self.conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution then bilinear interpolation
                # with scale factor three.
                y = self.conv(x)
                w = nn.functional.interpolate(
                    y, mode="bilinear", align_corners=False, scale_factor=3
                )
                return w

        f = test()
        # no failure
        g = torch.jit.trace(f, (torch.zeros(1, 1, 28, 28),))
        x = torch.zeros(1, 1, 14, 14)
        # constants not baked in
        self.assertEqual(g(x), f(x))

    @_tmp_donotuse_dont_inline_everything
    def test_trace_optional(self):
        # [FUNCTION DOC] With inlining disabled, traces functions calling a scripted
        # function taking an optional tensor; the none and tensor traces match expected
        # outputs and the graph keeps a call to the scripted function.
        @torch.jit.script
        def test(x: Optional[Tensor]):
            # [FUNCTION DOC] HELPER: Scripted: zeros of size one if x is none, else x.
            if x is None:
                return torch.zeros(1)
            else:
                return x

        def test_none():
            # [FUNCTION DOC] HELPER: Calls the scripted function with none.
            return test(None)

        def test_tensor():
            # [FUNCTION DOC] HELPER: Calls the scripted function with a tensor of zeros of
            # size two.
            return test(torch.zeros(2))

        f_none = torch.jit.trace(test_none, ())
        self.assertEqual(f_none(), torch.zeros(1))

        f_tensor = torch.jit.trace(test_tensor, ())
        self.assertEqual(f_tensor(), torch.zeros(2))

        graph = f_tensor.graph
        FileCheck().check('name="test"').check_next("prim::CallFunction").run(graph)

    def test_trace_nested_datatypes(self):
        # [FUNCTION DOC] A traced function calling a scripted function returning nested
        # lists works on new input.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted: returns nested lists of shifted tensors.
            return [[x + 1, x - 1], [x + 2, x - 2]]

        def bar(x):
            # [FUNCTION DOC] HELPER: Calls the scripted function and returns two of its
            # elements.
            list_stuff = foo(x)
            return list_stuff[0][0], list_stuff[1][1]

        traced = torch.jit.trace(bar, torch.rand(3, 4))
        x = torch.rand(5, 6)
        self.assertEqual(bar(x), traced(x))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_traced_module(self):
        # [FUNCTION DOC] With inlining disabled, a traced module calling a traced function
        # keeps a call to that function in the graph after the mm node.
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # [FUNCTION DOC] HELPER: Traced function negating its input.
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter calling the traced function on a
            # matrix product.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the parameter and calls the traced
                # function.
                return traced_fn(torch.mm(x, self.param))

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        # Note: neg op from the traced function should be properly inlined
        FileCheck().check("aten::mm").check('name="traced_fn"').check_next(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_module_from_traced_module(self):
        # [FUNCTION DOC] With inlining disabled, a traced module calling a traced submodule
        # keeps a method call to its forward in the graph.
        class TracedModule1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter, multiplying input by it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product with the parameter.
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that applies its own parameter then a traced
            # submodule and adds one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and a traced submodule.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = torch.jit.trace(TracedModule1(), torch.rand(3, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the parameter, calls the submodule
                # and adds one.
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))

    def test_index_put_trace_with_view(self):
        # [FUNCTION DOC] Index-put with a view on the right-hand side: the trace graph
        # contains a view node then index_put_.
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(1, 1, 1, 4))
        def test_index_put(target, indices, rhs):
            # [FUNCTION DOC] HELPER: Assigns rhs into target at the given indices and
            # returns target.
            target[indices] = rhs
            return target

        FileCheck().check("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    def test_index_put_trace_without_view(self):
        # [FUNCTION DOC] Index-put with a directly shaped right-hand side: no view node,
        # only index_put_.
        @_trace(torch.rand(100), torch.tensor([1, 2, 3, 4]), torch.rand(4))
        def test_index_put(target, indices, rhs):
            # [FUNCTION DOC] HELPER: Assigns rhs into target at the given indices and
            # returns target.
            target[indices] = rhs
            return target

        FileCheck().check_not("aten::view").check("index_put_").run(
            str(test_index_put.graph)
        )

    @suppress_warnings
    def test_trace_checker_dot_data(self):
        # [FUNCTION DOC] Trace checker detects a tensor constant from data access differing
        # across check inputs: raises TracingCheckError about tensor-valued constants.
        # Warnings suppressed.
        with self.assertRaisesRegex(
            torch.jit.TracingCheckError,
            r"Tensor-valued Constant nodes differed in value across invocations",
        ):

            @_trace(torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)])
            def foo(x):
                # [FUNCTION DOC] HELPER: Reads the data attribute and adds it to the input.
                y = x.data
                return x + y

    @suppress_warnings
    def test_trace_checker_control_flow(self):
        # [FUNCTION DOC] Trace checker detects loop count depending on input size: raises
        # TracingCheckError saying graphs differed. Warnings suppressed.
        def foo(x):
            # [FUNCTION DOC] HELPER: Negates x as many times as its first dimension size.
            for _ in range(x.size(0)):
                x = torch.neg(x)
            return x

        with self.assertRaisesRegex(
            torch.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):
            torch.jit.trace(foo, torch.randn(3, 4), check_inputs=[torch.randn(4, 4)])

    @suppress_warnings
    def test_trace_checker_memoization(self):
        # [FUNCTION DOC] Trace checker detects a function that memoizes in a function
        # attribute: raises TracingCheckError saying graphs differed. Warnings suppressed.
        with self.assertRaisesRegex(
            torch.jit.TracingCheckError, r"Graphs differed across invocations!"
        ):

            def foo(x):
                # [FUNCTION DOC] HELPER: Caches a negated tensor on the first call and adds
                # the cache.
                if not hasattr(foo, "cache"):
                    foo.cache = torch.neg(x)
                return x + foo.cache

            traced = torch.jit.trace(
                foo, torch.rand(3, 4), check_inputs=[(torch.rand(3, 4),)]
            )

    def test_trace_checker_slice_lhs(self):
        # [FUNCTION DOC] checkTrace with loop-based slice assignment.
        def foo(x):
            # [FUNCTION DOC] HELPER: Zeros three rows of the input in place and returns it.
            for i in range(3):
                x[i, :] = torch.zeros(4)
            return x

        self.checkTrace(foo, (torch.rand(3, 4),), inputs_require_grads=False)

    def test_trace_checker_inplace_on_view(self):
        # [FUNCTION DOC] In-place op on a view with check inputs of different shape yields a
        # TracerWarning that output one does not match.
        def foo(x):
            # [FUNCTION DOC] HELPER: Adds minus x to a flat view of x in place.
            x.view(-1).add_(-x.view(-1))
            return x

        with self.assertWarnsRegex(
            torch.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            torch.jit.trace(
                foo,
                torch.rand(3, 4),
                check_inputs=[torch.rand(5, 6)],
                _force_outplace=True,
            )

    def test_lhs_index_fails(self):
        # [FUNCTION DOC] Assigning to an indexed element in a forced out-of-place trace
        # emits a TracerWarning about an incorrect trace.
        def foo(x):
            # [FUNCTION DOC] HELPER: Assigns four to an element and returns the tensor.
            x[0, 1] = 4
            return x

        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    def test_lhs_index_trivial(self):
        # [FUNCTION DOC] Assigning to an ellipsis slice of the first argument traces fine
        # under checkTrace.
        def foo(y, x):
            # [FUNCTION DOC] HELPER: Assigns x into all of y and returns y.
            y[...] = x
            return y

        self.checkTrace(
            foo, (torch.rand(3, 4), torch.rand(4)), inputs_require_grads=False
        )

    def test_inplace_warn(self):
        # [FUNCTION DOC] In-place op on a view emits a TracerWarning about an incorrect
        # trace.
        def foo(x):
            # [FUNCTION DOC] HELPER: Adds minus x to a flat view of x in place.
            x.view(-1).add_(-x.view(-1))
            return x

        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "cause the trace to be incorrect"
        ):
            torch.jit.trace(foo, torch.rand(3, 4), _force_outplace=True)

    @suppress_warnings
    def test_trace_checker_dropout_train(self):
        # [FUNCTION DOC] Dropout in train mode yields TracerWarnings about mismatched output
        # and nondeterministic nodes. Warnings suppressed.
        def foo(x):
            # [FUNCTION DOC] HELPER: Applies dropout with probability one half in train
            # mode.
            return torch.dropout(x, p=0.5, train=True)

        with self.assertWarnsRegex(
            torch.jit.TracerWarning,
            "Output nr 1. of the traced function does not match the "
            "corresponding output of the Python function",
        ):
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])

        with self.assertWarnsRegex(
            torch.jit.TracerWarning, "Trace had nondeterministic nodes"
        ):
            torch.jit.trace(foo, torch.rand(3, 4), check_inputs=[torch.rand(5, 6)])

    def test_trace_checker_dropout_notrain(self):
        # [FUNCTION DOC] Dropout in eval mode traces deterministically: output equals the
        # input.
        input = torch.rand(3, 4)

        @_trace(input)
        def foo(x):
            # [FUNCTION DOC] HELPER: Applies dropout with train mode off.
            return torch.dropout(x, p=0.5, train=False)

        self.assertEqual(foo(input), input)

    def test_trace_contiguous(self):
        # [FUNCTION DOC] A trace of a strided slice followed by contiguous and view must
        # produce a tensor with different storage from the input.
        def foo(x):
            # [FUNCTION DOC] HELPER: Takes every second element of the last dim, makes
            # contiguous, views as twelve.
            return x[:, :, ::2].contiguous().view(12)

        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        y = traced(x)
        self.assertNotEqual(x.storage().data_ptr(), y.storage().data_ptr())

    # This tests the logic in THPVariable_contiguous. There is short-circuiting
    # code that prevents us from even getting to VariableType::contiguous, since
    # it is an optimization that prevents us from acquiring the GIL for touching
    # the device. We needed to add the tracing logic directly into the
    # THPVariable_contiguous function only for the path where we are skipping
    # dispatch into contiguous. We should see an aten::contiguous in this trace!
    def test_trace_contiguous_short_circuit(self):
        # [FUNCTION DOC] Tests that the contiguous call is recorded even when the Python
        # binding short-circuits dispatch: the graph contains a contiguous node.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the contiguous version of the input.
            return x.contiguous()

        x = torch.rand(2, 3, 4)
        traced = torch.jit.trace(foo, (x,))
        FileCheck().check("aten::contiguous").run(str(traced.graph))

    def test_trace_inverse(self):
        # [FUNCTION DOC] Traces bitwise invert on a uint8 tensor and checks it matches eager
        # on another shape.
        def foo(x):
            # [FUNCTION DOC] HELPER: Returns the bitwise inverse.
            return ~x

        foo_traced = torch.jit.trace(foo, torch.zeros(3, 4, dtype=torch.uint8))
        eg = torch.zeros(3, dtype=torch.uint8)
        self.assertEqual(foo_traced(eg), foo(eg))

    def test_trace_modulelist(self):
        # [FUNCTION DOC] Traces a module that loops over a ModuleList of submodules; only
        # checks tracing works.
        class MySubmod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu.
                return self.relu(x)

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ModuleList of two submodules applied in
            # order.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ModuleList.
                super().__init__()
                self.ml = torch.nn.ModuleList([MySubmod(), MySubmod()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each submodule in order.
                for mod in self.ml:
                    x = mod(x)
                return x

        traced = torch.jit.trace(MyMod(), (torch.rand(3, 4),))

    def test_trace_fork_join_and_module(self):
        # [FUNCTION DOC] Traces a module that forks submodule calls and waits on them, then
        # stacks the results; only checks tracing works.
        class MySubmod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a relu and a negation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the input and its negation.
                return self.relu(x), torch.neg(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module forking two submodule calls and stacking the first
            # outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ModuleList of two submodules.
                super().__init__()
                self.ml = torch.nn.ModuleList([MySubmod() for i in range(2)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks each submodule, waits on each future, takes
                # the first output and stacks the results.
                futs = []
                for i in range(2):
                    futs.append(torch.jit._fork(self.ml[i], x))

                results = []
                for i in range(2):
                    results.append(torch.jit._wait(futs[i])[0])

                return torch.stack(results)

        m = Mod()
        traced = torch.jit.trace(m, torch.rand(3, 4))

    def test_trace_invert_module_hierarchy(self):
        # [FUNCTION DOC] Tracing where a submodule is passed as an argument to another
        # module call: the module hierarchy is inverted; only checks tracing works.
        class MySubmod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a relu and a negation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the ReLU.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the input and its negation.
                return self.relu(x), torch.neg(x)

        class MyFunctionalMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling a submodule given as an argument.
            def forward(self, x, submod):
                # [FUNCTION DOC] HELPER: Calls the given submodule on x.
                return submod(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a submodule and a functional module and
            # passing the first to the second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates both submodules.
                super().__init__()
                self.sm = MySubmod()
                self.fm = MyFunctionalMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the functional module with the input and the
                # submodule.
                return self.fm(x, self.sm)

        torch.jit.trace(Mod(), (torch.rand(3, 4),))

    @skipIfCrossRef
    def test_trace_records_names(self):
        # [FUNCTION DOC] Traced graph keeps the argument and local variable names bar, baz
        # and quick_brown_fox; raises AssertionError if one is missing. Skipped under
        # cross-ref.
        def foo(bar, baz):
            # [FUNCTION DOC] HELPER: Computes baz from bar, negates it, then loops
            # subtracting a constant, returning the last result.
            baz = bar + 3
            quick_brown_fox = torch.neg(baz)
            for _ in range(20):
                yeet = quick_brown_fox - 3.14
            return yeet

        traced = torch.jit.trace(foo, (torch.rand(3, 3), torch.rand(3, 3)))
        graph_str = str(traced.graph)
        if "bar" not in graph_str:
            raise AssertionError(f"'bar' not found in graph: {graph_str}")
        if "baz" not in graph_str:
            raise AssertionError(f"'baz' not found in graph: {graph_str}")
        if "quick_brown_fox" not in graph_str:
            raise AssertionError(f"'quick_brown_fox' not found in graph: {graph_str}")

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_tracing_hooks(self):
        # [FUNCTION DOC] Registered forward hooks and pre-hooks (with and without return
        # values and captured variables) are recorded in the trace graph in the expected op
        # order, and traced results equal eager results. Inputs must not leak hook mutations
        # between eager and traced calls.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module doubling its input by adding it to itself.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

        def test_hook(is_post_hook, hook, fc):
            # [FUNCTION DOC] HELPER: Registers a hook (post hook if is_post_hook is true,
            # else pre hook) on a Net, traces it, runs eager and traced modules on fresh
            # inputs, runs the FileCheck on the traced graph and asserts equal input and
            # output.
            n = Net()
            if is_post_hook:
                n.register_forward_hook(hook)
            else:
                n.register_forward_pre_hook(hook)

            module = torch.jit.trace(n, (torch.tensor(1.0),))

            eager_input = torch.tensor(1.0)
            eager_out = n(eager_input)

            fc.run(module.forward.graph)
            input = torch.tensor(1.0)
            output = module(input)

            self.assertEqual(input, eager_input)
            self.assertEqual(output, eager_out)

        def hook_no_return(mod, input, output):
            # [FUNCTION DOC] HELPER: Post hook: adds one to the input in place and subtracts
            # one from the output in place; returns nothing.
            input[0].add_(1)
            output.sub_(1)

        fc = FileCheck().check("add(").check("add_(").check("sub_(")
        test_hook(True, hook_no_return, fc)

        def hook_return(mod, input, output):
            # [FUNCTION DOC] HELPER: Post hook: adds one to the input in place and returns
            # the output minus three.
            input[0].add_(1)
            return output - 3

        fc = FileCheck().check("add(").check("add_(").check("sub(")
        test_hook(True, hook_return, fc)

        b = torch.tensor(3.0)

        def captured_hook(mod, input, output):
            # [FUNCTION DOC] HELPER: Post hook: returns the output minus a captured tensor.
            return output - b

        fc = FileCheck().check("add(").check("sub(")
        test_hook(True, captured_hook, fc)

        def pre_hook_no_ret(mod, input):
            # [FUNCTION DOC] HELPER: Pre hook: adds three to the input in place; returns
            # nothing.
            input[0].add_(3)

        fc = FileCheck().check("add_(").check("add(")
        test_hook(False, pre_hook_no_ret, fc)

        def pre_hook_ret(mod, input):
            # [FUNCTION DOC] HELPER: Pre hook: returns the first input minus four.
            return input[0] - 4

        fc = FileCheck().check("sub(").check("add(")
        test_hook(False, pre_hook_ret, fc)

    def test_tracing_backward_hook_error(self):
        # [FUNCTION DOC] Tracing a module with a registered backward hook (legacy hook API)
        # must raise an exception saying backward hooks are assigned.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module doubling its input by self addition.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

        n = Net()

        def backward_hook(module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Empty backward hook.
            pass

        n.register_backward_hook(backward_hook)
        with self.assertRaisesRegex(Exception, "backward hooks assigned"):
            torch.jit.trace(n, (torch.tensor(1.0),))

    def test_tracing_multiple_methods(self):
        # [FUNCTION DOC] trace_module on several methods (forward and another) with and
        # without check inputs; then tracing bound methods: forward works, a non-forward
        # method raises AttributeError.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution and an extra method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d.
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution.
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                # [FUNCTION DOC] HELPER: Multiplies the given weight by the convolution
                # weight.
                return weight * self.conv.weight

        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)
        inputs = {
            "forward": example_forward_input,
            "weighted_kernel_sum": example_weight,
        }
        n = Net()
        module = torch.jit.trace_module(n, inputs)

        check_inputs = []
        for _ in range(2):
            check_weight = torch.rand(1, 1, 3, 3)
            check_forward_input = torch.rand(1, 1, 3, 3)
            check_inputs.append(
                {"forward": check_forward_input, "weighted_kernel_sum": check_weight}
            )
        module = torch.jit.trace_module(
            n, inputs, check_trace=True, check_inputs=check_inputs
        )
        self.assertTrue(module._c._has_method("forward"))
        self.assertTrue(module._c._has_method("weighted_kernel_sum"))

        module = torch.jit.trace(n.forward, example_forward_input)
        module = torch.jit.trace(
            n.forward,
            example_forward_input,
            check_trace=True,
            check_inputs=[example_forward_input],
        )
        with self.assertRaisesRegex(
            AttributeError,
            "trace doesn't support compiling individual module's functions",
        ):
            module = torch.jit.trace(n.weighted_kernel_sum, inputs)

    def test_tensor_with_grad_as_constant(self):
        # [FUNCTION DOC] Tracing a function capturing a tensor that requires grad must raise
        # RuntimeError about inserting such a tensor as a constant.
        param = torch.randn(3).requires_grad_()
        x = torch.randn(3)

        def f(x):
            # [FUNCTION DOC] HELPER: Adds the captured parameter.
            return x + param

        with self.assertRaisesRegex(
            RuntimeError, "Cannot insert a Tensor that requires grad as a constant"
        ):
            torch.jit.trace(f, x)

    def test_non_tensor_tracing(self):
        # [FUNCTION DOC] Tracing with a Python int tuple as input must raise RuntimeError
        # saying that tuple of int cannot be traced.
        def f(x):
            # [FUNCTION DOC] HELPER: Adds an undefined name to the input; never actually run
            # because tracing fails first.
            return x + param  # noqa: F821

        with self.assertRaisesRegex(
            RuntimeError, r"Type 'Tuple\[int\]' cannot be traced"
        ):
            torch.jit.trace(f, (1,))

    def test_trace_skip_none_submodule(self):
        # [FUNCTION DOC] A submodule attribute reassigned to none is skipped by tracing: the
        # traced module has no such attribute.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose submodule attribute is overwritten with none.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Linear then replaces the attribute with
                # none.
                super().__init__()
                self.submod = torch.nn.Linear(3, 4)
                self.submod = None

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Returns the input.
                return inputs

        m = TestModule()
        tm = torch.jit.trace(m, torch.tensor(1.0))
        self.assertFalse(hasattr(tm, "submod"))

    def test_trace_with_conditional_property(self):
        # [FUNCTION DOC] Tracing a module with a property that reads an optional attribute
        # must not fail.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Module with a property reading an attribute by name.
            def __init__(self, attr=None):
                # [FUNCTION DOC] HELPER: Stores the attribute only when given, and records
                # its name.
                super().__init__()
                if attr is not None:
                    self._attr = attr
                self.attr_name = "_attr"

            @property
            def attr(self):
                # [FUNCTION DOC] HELPER: Property: returns the attribute whose name is
                # recorded.
                return getattr(self, self.attr_name)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        x = torch.ones(1)
        torch.jit.trace(Net(), x)

    def test_trace_func_argument_names_captured(self):
        # [FUNCTION DOC] Argument names of a traced function appear in order in the graph.
        def fn(first_arg: torch.Tensor, second_arg: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds its two tensor arguments.
            return first_arg + second_arg

        traced_fn = torch.jit.trace(fn, (torch.ones(1), torch.ones(1)))
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_fn.graph)
        )

    def test_trace_partial_func_argument_names_captured(self):
        # [FUNCTION DOC] Only provided argument names appear in the graph when a defaulted
        # argument is not traced.
        def fn(first_arg: torch.Tensor, second_arg=1) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds the tensor and the defaulted second argument.
            return first_arg + second_arg

        traced_fn = torch.jit.trace(fn, (torch.ones(1),))
        FileCheck().check("first_arg").check_not("second_arg").run(str(traced_fn.graph))

    def test_trace_module_argument_names_captured(self):
        # [FUNCTION DOC] Argument names are captured in the graph when tracing a module's
        # forward method and when tracing the module.
        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a conv and a two-argument forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d.
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, first_arg: torch.Tensor, second_arg: torch.Tensor):
                # [FUNCTION DOC] HELPER: Convolves the first input and adds the second.
                return self.conv(first_arg) + second_arg

        m = TestModule()
        example_input = (torch.ones(1, 1, 3, 3), torch.ones(1, 1, 3, 3))

        # Explicitly tracing module's forward method
        traced_module_forward = torch.jit.trace(m.forward, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module_forward.graph)
        )

        # Tracing module's directly
        traced_module = torch.jit.trace(m, example_input)
        FileCheck().check("first_arg").check_next("second_arg").run(
            str(traced_module.graph)
        )

    def test_trace_checking_with_deprecated_name(self):
        # [FUNCTION DOC] Tracing a model whose forward takes extra keyword arguments works
        # positionally and with keyword example inputs in non-strict mode.
        class MyClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a catch-all keyword argument in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent initializer.
                super(MyClass, self).__init__()

            def forward(self, x, y, **deprecated_arguments):
                # [FUNCTION DOC] HELPER: Raises RuntimeError if any extra keyword argument
                # is passed; otherwise returns x plus y.
                if len(deprecated_arguments) > 0:
                    raise RuntimeError(
                        f"Got unexpected arguments: {deprecated_arguments}"
                    )
                return x + y

        model = MyClass()
        m2 = torch.jit.trace(model, (torch.ones(1), torch.ones(1)))
        m3 = torch.jit.trace(
            model,
            example_kwarg_inputs={"x": torch.ones(1), "y": torch.ones(1)},
            strict=False,
        )

    def test_trace_with_tuple_tensor(self):
        # [FUNCTION DOC] Traces a module taking a tuple of tensors, using positional and
        # keyword examples; traced outputs equal eager outputs for dict inputs.
        class MyClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module summing a tensor and a pair of tensors.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent initializer.
                super(MyClass, self).__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x plus both elements of y.
                return x + y[0] + y[1]

        model = MyClass()
        traced_model = torch.jit.trace(
            model, (torch.ones(1), (torch.ones(1), torch.ones(1)))
        )
        input_dict = {
            "x": torch.tensor([2, 3]),
            "y": (torch.tensor([5, 6]), torch.tensor([7, 8])),
        }
        self.assertEqual(model(**input_dict), traced_model(**input_dict))
        traced_model = torch.jit.trace(
            model,
            example_kwarg_inputs={
                "x": torch.ones(1),
                "y": (torch.ones(1), torch.ones(1)),
            },
        )
        self.assertEqual(model(**input_dict), traced_model(**input_dict))

    def test_trace_no_duplicated_lifted_input_output(self):
        # [FUNCTION DOC] Traced graph for nested modules with optional arguments must not
        # contain tuple unpack, i.e. lifted inputs and outputs are not duplicated.
        class Normalize(nn.Module):
            # [CLASS DOC] HELPER: Module applying GroupNorm to the second argument when
            # present, else reuses the first; then doubles.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the GroupNorm.
                super().__init__()
                self.norm = nn.GroupNorm(num_groups=32, num_channels=32)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Chooses the first argument if the second is none,
                # else normalizes the second; doubles and returns it.
                if y is None:
                    y = x
                else:
                    y = self.norm(y)
                y = y * 2
                return y

        class G(nn.Module):
            # [CLASS DOC] HELPER: Module calling Normalize with no second input and
            # returning the value and its relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Normalize submodule.
                super().__init__()
                self.norm = Normalize()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair: the normalized value and its relu.
                A = self.norm(x, None)
                B = F.relu(A)
                return A, B

        class Net(nn.Module):
            # [CLASS DOC] HELPER: Top module chaining G and a second Normalize.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates both submodules.
                super().__init__()
                self.g = G()
                self.norm_1 = Normalize()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs G, then feeds its second and first outputs to
                # the second Normalize.
                hs = self.g(x)
                A, B = hs
                h = self.norm_1(B, A)
                return h

        net = Net()
        net = net.eval()
        x = torch.randn(1, 32, 16, 16)
        traced = torch.jit.trace(net, x)
        FileCheck().check_not("prim::TupleUnpack").run(str(traced.graph))


@skipIfTorchDynamo("Not a suitable test for TorchDynamo")
class TestMixTracingScripting(JitTestCase):
    # [CLASS DOC] Test case for combinations of traced and scripted code. Skipped under
    # TorchDynamo.
    def test_trace_script(self):
        # [FUNCTION DOC] checkTrace on scripted functions taking tuple, list and defaulted
        # arguments (interpolate); also defines one with an optional string list that is not
        # run.
        @torch.jit.script
        def func1(x: Tuple[Tensor, Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted: adds the elements of a tuple.
            return x[0] + x[1]

        @torch.jit.script
        def func2(x: List[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted: adds the first two elements of a list.
            return x[0] + x[1]

        a = torch.randn(5)
        b = torch.randn(5)

        self.checkTrace(func1, ((a, b),))
        self.checkTrace(func2, ((a, b),))

        @torch.jit.script
        def func3(
            x: Tensor, method: str = "bilinear", align_corners: bool = True
        ) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted: interpolates to the spatial size with the
            # given mode and align_corners flag.
            hw = x.shape[2:4]
            return F.interpolate(x, hw, mode=method, align_corners=align_corners)

        inp = torch.rand(1, 3, 6, 6)
        self.checkTrace(func3, (inp,))

        @torch.jit.script
        def func4(x: Tensor, a: List[Optional[str]]) -> Tensor:
            # [FUNCTION DOC] HELPER: Scripted: returns x plus two when the list has two
            # elements else x.
            if len(a) == 2:
                return x + 2
            else:
                return x

    def test_trace_mixed_by_script_with_dict_output(self):
        # [FUNCTION DOC] A traced module calling a scripted function returning a dict works
        # and matches expected values.
        @torch.jit.script
        def return_dict(input: torch.Tensor) -> Dict[str, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Scripted: returns a dict with key foo holding the input
            # plus one.
            return {"foo": input + 1}

        class TraceModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module doubling the dict entry from the scripted function.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the scripted function and returns twice its
                # foo entry.
                dict = return_dict(input)
                return dict["foo"] + dict["foo"]

        x = torch.ones(1)
        tm = torch.jit.trace(TraceModule(), x)
        self.assertEqual(tm(x), x + 1 + x + 1)

    def test_trace_of_script(self):
        # [FUNCTION DOC] Traces a function calling a scripted function with control flow;
        # shapes are propagated (no dynamic dims in the graph) and results are checked for
        # two inputs.
        @torch.jit.script
        def foo(a, c):
            # [FUNCTION DOC] HELPER: Scripted: sets b to one when a equals zero, else keeps
            # zero, and returns b plus c.
            b = 0.0
            if bool(a == 0.0):
                b = 1.0
            return b + c

        a = torch.ones(1, dtype=torch.float)

        @_trace(torch.zeros(1, dtype=torch.float))
        def use(b):
            # [FUNCTION DOC] HELPER: Traced: calls the scripted function and adds one.
            return foo(b - 1.0, a) + 1.0

        # test we propagated shapes through the function
        self.assertTrue("Dynamic" not in str(use.graph))

        self.assertEqual(3, use(torch.ones(1, dtype=torch.float)))
        self.assertEqual(2, use(torch.zeros(1, dtype=torch.float)))

    def test_trace_with_size(self):
        # [FUNCTION DOC] A scripted function calling a traced function and using the result
        # as int; the constant branch yields eight.
        @_trace(torch.zeros(1, 1))
        def foo(x):
            # [FUNCTION DOC] HELPER: Traced: adds one.
            return x + 1

        @torch.jit.script
        def bar(x):
            # [FUNCTION DOC] HELPER: Scripted: converts the traced result to int, sets it to
            # seven under an always true branch and returns it plus one.
            y = int(foo(x))
            if 1 == 1:
                y = 7
            return y + 1

        self.assertEqual(8, bar(torch.ones(1, 1)))

    def test_tracing_slicing(self):
        # [FUNCTION DOC] Negative slice indexes: traced result equals scripted and eager on
        # the example shape but differs for another shape.
        @_trace(torch.zeros(10))
        def foo_trace(x):
            # [FUNCTION DOC] HELPER: Traced slice from minus five to minus three.
            return x[-5:-3]

        @torch.jit.script
        def foo_script(x):
            # [FUNCTION DOC] HELPER: Scripted slice from minus five to minus three.
            return x[-5:-3]

        def foo(x):
            # [FUNCTION DOC] HELPER: Eager slice from minus five to minus three.
            return x[-5:-3]

        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_trace(a), foo_script(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_tracing_indexing(self):
        # [FUNCTION DOC] Negative integer index: traced equals scripted and eager on the
        # example shape but differs for another shape.
        @_trace(torch.zeros(10))
        def foo_trace(x):
            # [FUNCTION DOC] HELPER: Traced: returns element minus two.
            return x[-2]

        @torch.jit.script
        def foo_script(x):
            # [FUNCTION DOC] HELPER: Scripted: returns element minus two.
            return x[-2]

        def foo(x):
            # [FUNCTION DOC] HELPER: Eager: returns element minus two.
            return x[-2]

        a = torch.arange(0, 8)
        b = torch.arange(0, 20)
        self.assertEqual(foo_script(a), foo_trace(a))
        self.assertEqual(foo_trace(a), foo(a))
        self.assertNotEqual(foo_trace(a), foo_trace(b))

    def test_trace_hierarchy(self):
        # [FUNCTION DOC] Tracing a module that contains ScriptModules keeps the hierarchy:
        # methods and parameters exist on both the C++ module and the Python wrapper, before
        # and after export and import.
        # Test that we preserve the module hierarchy for a ScriptModule
        # submodule during tracing

        class AnotherScriptMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a parameter and a scripted method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(1, 2, 3))

            @torch.jit.script_method
            def bar(self):
                # [FUNCTION DOC] HELPER: Scripted method returning zeros.
                return torch.zeros(4, 5)

        class SomeScriptMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule containing another ScriptModule and two
            # scripted methods.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the nested ScriptModule.
                super().__init__()
                self.asm = AnotherScriptMod()

            @torch.jit.script_method
            def foo(self):
                # [FUNCTION DOC] HELPER: Scripted method returning zeros.
                return torch.zeros(3, 4)

            @torch.jit.script_method
            def bar(self):
                # [FUNCTION DOC] HELPER: Scripted method returning zeros.
                return torch.zeros(4, 3)

        class TraceMe(torch.nn.Module):
            # [CLASS DOC] HELPER: Traced module that calls a scripted submodule method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the scripted submodule.
                super().__init__()
                self.ssm = SomeScriptMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the output of the scripted method to the
                # input.
                return self.ssm.bar() + x

        orig = TraceMe()
        traced = torch.jit.trace(orig, (torch.rand(4, 3),))
        # for each of these checks, check that *BOTH* the underlying
        # _C.ScriptModule object has the expected method/param, as well as the
        # Python object that wraps it.
        self.assertTrue(traced.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(traced.ssm, "foo"))

        imported = self.getExportImportCopy(traced)

        self.assertTrue(imported.ssm._c._has_method("foo"))
        self.assertTrue(hasattr(imported.ssm, "foo"))

        self.assertTrue(imported.ssm.asm._c._has_method("bar"))
        self.assertTrue(hasattr(imported.ssm.asm, "bar"))

        self.assertTrue(hasattr(imported.ssm.asm, "param"))

    def test_trace_parameter(self):
        # [FUNCTION DOC] Nested trace, script and eager modules containing a Parameter:
        # builds M1 of M2 of M3 of a traced Param and saves it to a buffer with optimized
        # execution off; checks saving works.
        class Param(nn.Module):
            # [CLASS DOC] HELPER: Module with a registered empty bias parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers the parameter.
                super().__init__()
                self.register_parameter("bias", nn.Parameter(torch.empty(4, 4)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        class M3(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule holding a traced module and calling it.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Traces the given model on a random input.
                super().__init__()
                self.traced = torch.jit.trace(model, (torch.rand(3, 3)))

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted: calls the traced module.
                return self.traced(x)

        class M2(nn.Module):
            # [CLASS DOC] HELPER: Eager module wrapping M3.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Creates the M3 submodule.
                super().__init__()
                self.module = M3(model)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the submodule.
                return self.module(x)

        class M1(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule holding a traced M2 and calling it.
            def __init__(self, model):
                # [FUNCTION DOC] HELPER: Traces an M2 module around the given model.
                super().__init__()
                self.traced = torch.jit.trace(M2(model), (torch.rand(3, 3)))

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted: calls the traced module.
                return self.traced(x)

        with torch.jit.optimized_execution(False):
            module = M1(Param())
            f = io.BytesIO()
            torch.jit.save(module, f)

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_traced_module(self):
        # [FUNCTION DOC] With inlining disabled, a traced module calling a scripted function
        # keeps a call to that function in the graph after the mm node.
        @torch.jit.script
        def scripted_fn(x):
            # [FUNCTION DOC] HELPER: Scripted: negates its input.
            return torch.neg(x)

        class TracedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter calling the scripted function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the parameter and calls the scripted
                # function.
                return scripted_fn(torch.mm(x, self.param))

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check("aten::mm").check('name="scripted_fn"').check(
            "prim::CallFunction"
        ).run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_module_from_traced_module(self):
        # [FUNCTION DOC] With inlining disabled, a traced module calling a ScriptModule
        # keeps a method call to its forward in the graph.
        class ScriptMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule multiplying input by a parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param_foo = torch.nn.Parameter(torch.rand(5, 7))

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted method: returns the matrix product with
                # the parameter.
                return torch.mm(x, self.param_foo)

        class TracedModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and a ScriptModule, applying both
            # and adding one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and the ScriptModule.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = ScriptMod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by the parameter, calls the ScriptModule
                # and adds one.
                return self.mod(torch.mm(x, self.param)) + 1.0

        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

        FileCheck().check("aten::mm").check("prim::CallMethod").check_same(
            "forward"
        ).check("aten::add").run(str(tm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_script_fn(self):
        # [FUNCTION DOC] With inlining disabled, a scripted function calling a traced
        # function contains a call and an add in its graph.
        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            # [FUNCTION DOC] HELPER: Traced: negates its input.
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            # [FUNCTION DOC] HELPER: Scripted: calls the traced function and adds one.
            return traced_fn(x) + 1

        FileCheck().check("prim::CallFunction").check("aten::add").run(
            str(script_fn.graph)
        )

    def test_call_traced_mod_from_script_fn(self):
        # [FUNCTION DOC] A scripted function calling a traced module that is not its
        # submodule must raise RuntimeError.
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot call a ScriptModule that is not a submodule of the caller",
        ):

            class TracedModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Module multiplying by a zeros matrix.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the matrix product with a zeros matrix.
                    return torch.mm(x, torch.zeros(4, 3))

            tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))

            @torch.jit.script
            def script_fn(x):
                # [FUNCTION DOC] HELPER: Scripted: calls the traced module and adds one.
                return tm(x) + 1

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_fn_from_script_module(self):
        # [FUNCTION DOC] With inlining disabled, a ScriptModule calling a traced function
        # keeps a call in its forward graph.
        @_trace(torch.rand(3, 3))
        def traced_fn(x):
            # [FUNCTION DOC] HELPER: Traced: negates its input.
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a parameter calling a traced function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted method: multiplies by the parameter and
                # calls the traced function.
                return traced_fn(torch.mm(x, self.param))

        sm = ScriptMod()
        FileCheck().check("aten::mm").check("prim::CallFunction").run(
            str(sm.forward.graph)
        )

    @_tmp_donotuse_dont_inline_everything
    def test_call_tracing_mod_from_script_module(self):
        # [FUNCTION DOC] With inlining disabled, a ScriptModule calling a traced submodule
        # keeps a method call in its graph.
        class TracedMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying input by a parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product with the parameter.
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a parameter and a traced submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter and traces the submodule.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.tm = torch.jit.trace(TracedMod(), torch.rand(3, 3))

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted method: multiplies by the parameter and
                # calls the traced submodule.
                return self.tm(torch.mm(x, self.param))

        sm = ScriptMod()
        FileCheck().check("aten::mm").check("prim::CallMethod").run(str(sm.graph))

    def test_script_inline_trace_multiple_args(self):
        # [FUNCTION DOC] A ScriptModule calling a traced submodule with two arguments runs
        # under optimized execution off.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding two inputs.
            def forward(self, input, input2):
                # [FUNCTION DOC] HELPER: Returns the sum of its inputs.
                return input + input2

        class M2(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule that calls a traced two-input module with the
            # same input twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Traces the adder module.
                super().__init__()
                self.m = torch.jit.trace(M(), (torch.zeros(4, 3), torch.zeros(4, 3)))

            @torch.jit.script_method
            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Scripted method: calls the traced module with the
                # input twice.
                return self.m(inp, inp)

        with torch.jit.optimized_execution(False):
            m2 = M2()
            m2(torch.zeros(4, 3))

    def test_trace_dict_mix_script(self):
        # [FUNCTION DOC] Traces a module holding a scripted module and dict-of-list inputs;
        # the trace on new inputs equals eager.
        class testB(torch.nn.Module):
            # [CLASS DOC] HELPER: Module averaging over a dict of lists via a Linear and
            # stacking.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Linear.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, feature_map: Dict[str, List[Tensor]]) -> Tensor:
                # [FUNCTION DOC] HELPER: Applies Linear to the first tensor of each dict
                # value and stacks the results.
                output = []
                for j in feature_map.values():
                    output.append(self.linear(j[0]))

                return torch.stack(output)

        class testA(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a scripted testB and building a feature map
            # dict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts a testB module.
                super().__init__()
                self.b = torch.jit.script(testB())

            def forward(self, input_map: Dict[str, List[Tensor]]) -> Tensor:
                # [FUNCTION DOC] HELPER: Builds a dict from the first tensor of each list
                # and calls the scripted module.
                feature_map = {}
                for i, j in input_map.items():
                    feature_map[i] = [j[0]]

                return self.b(feature_map)

        input_map = {
            "1": [torch.rand(2, 2), torch.rand(2, 2)],
            "3": [torch.rand(2, 2), torch.rand(2, 2)],
        }
        model = testA()
        traced_model = torch.jit.trace(model, input_map)
        new_input_map = {
            "1": [torch.rand(2, 2), torch.randn(2, 2)],
            "3": [torch.rand(2, 2), torch.rand(2, 2)],
        }
        self.assertEqual(model(new_input_map), traced_model(new_input_map))

    def test_trace_script_returning_complex_dict(self):
        # [FUNCTION DOC] Tracing an eager module over scripted submodules where one returns
        # a nested dict (dict of tuples) and the other consumes it; traced results match
        # eager for two different inputs.
        """Tracing over a script function returning a dictionary should work.
        The dictionary can should be able to contain other containers (like a tuple) recursively.
        """

        class ReturnsDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dict of tuple values.
            def forward(
                self,
                id_score_list: Dict[
                    str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ],
            ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                # [FUNCTION DOC] HELPER: Subtracts a large offset from the second tuple
                # element and returns a dict of the same structure.
                # do some random operations and then return a dict of the same structure
                v = id_score_list["1000"]
                idx_keys = v[1] - 1500000
                weights = v[2]
                result = {"1000": (v[0], idx_keys, weights)}
                return result

        class ChecksDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module consuming the dict and returning a value plus one.
            def forward(
                self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ):
                # [FUNCTION DOC] HELPER: Takes the second tuple element and adds one.
                v = input["1000"]
                return v[1] + 1

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Eager module chaining two scripted modules.
            def __init__(self, checks_dict, returns_dict):
                # [FUNCTION DOC] HELPER: Stores both modules.
                super().__init__()
                self.checks_dict = checks_dict
                self.returns_dict = returns_dict

            def forward(
                self, input: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ):
                # [FUNCTION DOC] HELPER: Feeds the input dict through the dict-returning
                # module, then the consuming one.
                foo = self.returns_dict(input)
                return self.checks_dict(foo)

        input1 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([]),
            )
        }

        input2 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([1500000, 1500004], dtype=torch.int64),
                torch.tensor([2.0, 3.0]),
            )
        }

        checks_dict = torch.jit.script(ChecksDict())
        returns_dict = torch.jit.script(ReturnsDict())
        eager_module = TestModule(checks_dict, returns_dict)
        traced_module = torch.jit.trace(eager_module, input1)
        self.assertEqual(traced_module(input1), eager_module(input1))
        self.assertEqual(traced_module(input2), eager_module(input2))

    def test_trace_returning_dict_with_tensor_tuples(self):
        # [FUNCTION DOC] A traced module returning a dict of tuples of tensors works in
        # non-strict mode and matches expected values; one returning a tuple with a
        # non-tensor raises RuntimeError that the output cannot be understood by the tracer.
        """Tracing over a module returning a dictionary whose values are tuples of tensors
        should work.
        """

        class ReturnsDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dict key mapped to a pair of tensors.
            def forward(
                self, k: torch.Tensor, v: torch.Tensor
            ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
                # [FUNCTION DOC] HELPER: Scales the inputs and returns them in a dict.
                x = 2 * k
                y = 3 * v
                result = {"imakey": (x, y)}
                return result

        class ReturnsBadDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dict key mapped to a tensor and an int.
            def forward(
                self, k: torch.Tensor, v: torch.Tensor
            ) -> Dict[str, Tuple[torch.Tensor, float]]:
                # [FUNCTION DOC] HELPER: Returns a dict with a tensor and an int.
                x = 2 * k
                result = {"imakey": (x, 1)}
                return result

        mod = ReturnsDict()
        traced_module = torch.jit.trace(
            mod, [torch.ones(1), torch.ones(1)], strict=False
        )
        out = traced_module(torch.ones(1), torch.ones(1))
        expected = {"imakey": (torch.tensor([2.0]), torch.tensor([3.0]))}
        self.assertEqual(out, expected)

        with self.assertRaisesRegex(
            RuntimeError, "cannot be understood by the tracer, only outputs matching"
        ):
            mod = ReturnsBadDict()
            traced_module = torch.jit.trace(
                mod, [torch.ones(1), torch.ones(1)], strict=False
            )

    def test_trace_linear(self):
        # [FUNCTION DOC] Traces a Linear module; checkTrace passes and the graph contains a
        # linear node.
        m = torch.nn.Linear(20, 20)
        inp = torch.rand([20, 20])
        self.checkTrace(m, (inp,))
        g = torch.jit.trace(m, (inp,)).graph
        FileCheck().check("aten::linear").run(g)

    def test_traced_module_implements_interface(self):
        # [FUNCTION DOC] Scripts a module that implements an interface and passes it to a
        # function expecting the interface; checkScript compares.
        @torch.jit.interface
        class TestModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Scripted interface declaring a two-tensor forward.
            def forward(
                self, first_arg: torch.Tensor, second_arg: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method with no implementation.
                pass

        make_global(TestModuleInterface)

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module implementing the interface.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Conv2d.
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(
                self, first_arg: torch.Tensor, second_arg: torch.Tensor
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Convolves the first input and adds the second.
                return self.conv(first_arg) + second_arg

        def fn_takes_interface(x: TestModuleInterface):
            # [FUNCTION DOC] HELPER: Calls forward of the interface-typed module on ones
            # tensors.
            ones = torch.ones(1, 1, 3, 3)
            return x.forward(ones, ones)

        scripted_test_module = torch.jit.script(TestModule())
        self.checkScript(fn_takes_interface, (scripted_test_module,))

    def test_traced_module_contains_scripted_interface_types(self):
        # [FUNCTION DOC] Traces a module whose scripted submodule is shared and also wrapped
        # by an eager module, using an interface typed attribute; only checks tracing works.
        class LeafModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a weight parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(19))

            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Adds the weight.
                return input + self.weight

        class LowerModuleImpl(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation of the lower interface wrapping a
            # LeafModule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LeafModule.
                super().__init__()
                self.leaf = LeafModule()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the leaf.
                return self.leaf(input)

        @torch.jit.interface
        class LowerModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Scripted interface declaring a tensor forward.
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method with no implementation.
                pass

        class MiddleModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface typed attribute holding an
            # implementation.
            lower: LowerModuleInterface

            def __init__(self, feature_processor_modules=None):
                # [FUNCTION DOC] HELPER: Creates the implementation; the argument is unused.
                super().__init__()
                self.lower = LowerModuleImpl()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the lower module.
                return self.lower(input)

        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an arbitrary submodule.
            def __init__(self, m):
                # [FUNCTION DOC] HELPER: Stores the submodule.
                super().__init__()
                self.middle = m

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the submodule.
                return self.middle(input)

        class TopModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the scripted MiddleModule and a wrapper of
            # the same module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts a MiddleModule and stores it directly and
                # inside a wrapper.
                super().__init__()
                m = MiddleModule()
                m = torch.jit.script(m)
                self.sub1 = m
                self.sub2 = WrapperModule(m)

            def forward(self, input: torch.Tensor):
                # [FUNCTION DOC] HELPER: Sums the outputs of both submodules.
                return self.sub1(input) + self.sub2(input)

        top = TopModule()
        top_example_input = torch.ones(1)
        torch.jit.trace(top, top_example_input)

    def test_jit_trace_callfunction_return_shapes(self):
        # [FUNCTION DOC] The trace of a function calling a scripted function has a
        # call-function node whose output has complete tensor type, including shape
        # information.
        # a torch.jit.script function gets inserted as a CallFunction node
        @torch.jit.script
        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Scripted: concatenates the input with itself.
            return torch.cat((x, x))

        def outer_fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the scripted function on the sum and applies
            # relu.
            return inner_fn(x + y).relu()

        x, y = [torch.rand((2, 2), dtype=torch.float) for _ in range(2)]
        fn_t = torch.jit.trace(outer_fn, (x, y))

        # expect that the CallFunction node return type has shape information on it.
        FileCheck().check("Float").check("4, 2").check("CallFunction").run(fn_t.graph)
        for n in fn_t.graph.nodes():
            if n.kind() == "prim::CallFunction":
                self.assertTrue(n.output().isCompleteTensor())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
