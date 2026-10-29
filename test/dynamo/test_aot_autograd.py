# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the AOT Autograd integration with Dynamo (aot_eager and inductor
# backends). Covers mutation handling, argument-dupe and storage-overlap guards, donated
# buffers, custom joint passes, sequence numbers, data_ptr access in fake mode, double
# backward, and partitioner behavior with symbolic/unbacked shapes.
# Import-time side effects: defines a global scoped Library named custom with an op
# maybe_dupe_op (CPU and Meta impls), and instantiates the device-generic test class
# AotAutogradFallbackTestsDevice into the module globals; runs the Dynamo test runner when
# executed as a script.
import copy
import operator
import re
import unittest
from textwrap import dedent
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._dynamo.testing import (
    CompileCounter,
    CompileCounterWithBackend,
    expectedFailureDynamic,
    rand_strided,
)
from torch._functorch.aot_autograd import _aot_export_function, create_functional_call
from torch._functorch.partitioners import _extract_fwd_bwd_modules
from torch._guards import CompileContext, StorageOverlap, TracingContext
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental import _config as fx_config
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.profiler import profile
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import compare_equal_outs_and_grads
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import ValueRanges


def maybe_dupe_op(x):
    # [FUNCTION DOC] HELPER: Implementation of the custom op maybe_dupe_op. Computes y as x
    # plus 1 and z as x plus 2. Returns the pair (y, y), the same tensor twice, when x has
    # fewer than 5 elements, otherwise the pair (y, z). Used to produce outputs that alias
    # only for small inputs.
    y = x + 1
    z = x + 2
    if x.numel() < 5:
        return y, y
    else:
        return y, z


def is_dynamic_shape_test(test_name):
    # [FUNCTION DOC] HELPER: Returns True if the test name string ends with the suffix
    # _dynamic_shapes, i.e. the test was generated as a dynamic-shape variant. Used to
    # adjust expected indices.
    return test_name.endswith("_dynamic_shapes")


aten = torch.ops.aten
lib = torch.library.Library("custom", "DEF")  # noqa: SCOPED_LIBRARY
lib.define("maybe_dupe_op(Tensor a) -> (Tensor, Tensor)")
lib.impl("maybe_dupe_op", maybe_dupe_op, "CPU")
lib.impl("maybe_dupe_op", maybe_dupe_op, "Meta")


class AotAutogradFallbackTests(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case (inductor TestCase base) with Dynamo plus AOT Autograd regression
    # tests, mostly using the aot_eager backend. Stateless; each test builds its own modules,
    # inputs and compile counters. Also holds two helper methods for the overlap guard tests.
    def test_LSTM(self):
        # [FUNCTION DOC] Regression test (torchdynamo issue 1147): a bidirectional 2-layer
        # LSTM with a non-contiguous strided input is compiled with aot_eager and its output
        # must match eager.
        # https://github.com/pytorch/torchdynamo/issues/1147
        class Repro(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a 2-layer bidirectional LSTM (64 features);
            # forward returns the LSTM result inside a one-element tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LSTM submodule
                # stored as self_mod_model_lstm_lstm.
                super().__init__()
                self.self_mod_model_lstm_lstm = torch.nn.LSTM(
                    64, 64, num_layers=2, bidirectional=True
                )

            def forward(self, permute: torch.Tensor):
                # [FUNCTION DOC] HELPER: Runs the LSTM on the permuted input and returns its
                # output wrapped in a tuple.
                self_mod_model_lstm_lstm = self.self_mod_model_lstm_lstm(permute)
                return (self_mod_model_lstm_lstm,)

        mod = Repro()

        aot_mod = torch.compile(mod, backend="aot_eager")

        args = [((92, 4, 64), (1, 5888, 92), torch.float32, "cpu", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        eager_result = mod(*args)
        aot_result = aot_mod(*args)
        self.assertTrue(torch._dynamo.testing.same(eager_result, aot_result))

    def test_mutation(self):
        # [FUNCTION DOC] Regression test (torchdynamo issue 1301): compiles a function that
        # mutates a Parameter in place under disabled grad mode (restoring the grad mode in a
        # finally block). Passes if aot_eager compilation and the call do not raise.
        # https://github.com/pytorch/torchdynamo/issues/1301
        def fn(param, y):
            # [FUNCTION DOC] HELPER: Saves grad mode, disables it, adds y into param in place,
            # restores the mode, and returns y.
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)
            return y

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        aot_fn = torch.compile(fn, backend="aot_eager")
        # This should not error: we mutated an autograd leaf under no_grad mode.
        aot_fn(x, y)

    def test_mutation1(self):
        # [FUNCTION DOC] Compiles with aot_eager a function that writes a slice of one input
        # into a slice of another via indexing assignment and returns a view; only checks the
        # call does not raise.
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            # [FUNCTION DOC] HELPER: Slices the second input, assigns it into a slice of
            # the first input in place, and returns a one-element tuple with a reshaped
            # view of the first.
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch.compile(fn, backend="aot_eager")
        aot_fn(x, y)

    def test_negative_testing_mutation(self):
        # [FUNCTION DOC] Like test_mutation1 but the mutated tensor is first the result of
        # sin, so the mutation is on an intermediate rather than an input; only checks the
        # compiled call does not raise. Makes no assertions on values.
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            # [FUNCTION DOC] HELPER: Slices the second input, applies sin to the first,
            # assigns the slice into it in place, and returns a one-element tuple with a view.
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0 = torch.sin(_stack0)
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        aot_fn = torch.compile(fn, backend="aot_eager")
        aot_fn(x, y)

    def test_negative_testing(self):
        # [FUNCTION DOC] Compiles with aot_eager a function that mutates the result of sin in
        # place; only checks the call does not raise.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns sin of x with y added in place.
            return torch.sin(x).add_(y)

        y = torch.randn(4)
        x = torch.randn(4)
        aot_fn = torch.compile(fn, backend="aot_eager")
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_safe(self):
        # [FUNCTION DOC] Exports a module whose forward calls a conv helper using its
        # weight/bias, checks the exported graph matches eager, then compiles the graph with
        # aot_eager and runs it.
        class ModuleSpecialFwd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a 3-to-20 channel 5x5 Conv2d and invoking its
            # _conv_forward with explicit weight and bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution submodule.
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the conv module's internal forward with its own
                # weight and bias.
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Delegates to _conv_forward.
                return self._conv_forward(x)

        # Init mod
        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])

        # Run it for real
        real = mod(rx)

        # Run it in export
        graph, _ = torch._dynamo.export(mod)(rx)

        # Run exported graph with AOT
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

        aot_fn = torch.compile(graph, backend="aot_eager")
        aot_fn(rx)

    def test_call_fn_with_non_const_inputs_aot_unsafe(self):
        # [FUNCTION DOC] Exports a module that mutates a Parameter under no_grad, checks the
        # graph output matches eager, then compiles the exported graph with aot_eager and runs
        # it; must not raise despite mutating an autograd leaf under no_grad.
        class ModuleSpecialFwd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward delegates to a helper that mutates its
            # first argument in place with grad disabled.
            def _some_bad_fwd(self, param, y):
                # [FUNCTION DOC] HELPER: Disables grad, adds y into param in place, restores
                # grad mode, returns y.
                prev_grad = torch.is_grad_enabled()
                try:
                    torch.set_grad_enabled(False)
                    param.add_(y)
                finally:
                    torch.set_grad_enabled(prev_grad)
                return y

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls _some_bad_fwd on the inputs and
                # returns its result.
                return self._some_bad_fwd(x, y)

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn(4))
        y = torch.randn([4])

        # Run it for real
        real = mod(x, y)

        # Run it in export
        graph, _ = torch._dynamo.export(mod)(x, y)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, graph(x, y)))

        # Run exported graph with AOT
        aot_fn = torch.compile(graph, backend="aot_eager")
        # This should not error: we mutated an autograd leaf under no_grad mode.
        aot_fn(x, y)

    def test_call_fn_with_non_const_inputs_aot_unsafe_control_flow(self):
        # [FUNCTION DOC] Compiles a module with data-dependent control flow using a capturing
        # backend that records graphs and counts them. Asserts output equality with eager, 2
        # compiled frames (graph break at the branch) and 7 total ops. Then resets Dynamo and
        # compiles the optimized module again with aot_eager and runs it.
        class ModuleSpecialFwd(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a helper that branches on a tensor value (first
            # element below 3).
            def _some_bad_fwd(self, param, y):
                # [FUNCTION DOC] HELPER: Returns y plus param if the first element of y is
                # below 3, else param times y.
                if y[0][0] < 3:
                    return y + param
                return param * y

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes the product of inputs, passes it through the
                # branching helper as both arguments, and multiplies the result with the sum
                # of the inputs.
                a = x * y
                a = self._some_bad_fwd(a, a)
                b = x + y
                return a * b

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn([2, 2]))
        y = torch.randn([2, 2])

        # Run it for real
        real = mod(x, y)

        # Run it through optimize, with our capturing fn

        gms = []
        counter = CompileCounter()

        def capturing_fn(gm, inputs):
            # [FUNCTION DOC] HELPER: Backend that appends the graph module to the enclosing
            # gms list and delegates compilation to the CompileCounter.
            nonlocal gms
            gms.append(gm)
            return counter(gm, inputs)

        optimized_mod = torch.compile(mod, backend=capturing_fn)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, optimized_mod(x, y)))

        # Uncomment to reproduce commented out graphs below.
        # for gm in gms:
        #     print("GM CODE", gm.code)

        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(counter.op_count, 7)
        # Graph 1
        # def forward(self, L_x_ : torch.nn.parameter.Parameter, L_y_ : torch.Tensor):
        #     l_x_ = L_x_
        #     l_y_ = L_y_
        #     a = l_x_ * l_y_;  l_x_ = l_y_ = None
        #     getitem = a[0]
        #     getitem_1 = getitem[0];  getitem = None
        #     lt = getitem_1 < 3;  getitem_1 = None
        #     return (lt, a)
        # BREAK
        # Graph 2
        # def forward(self, L_nested_frame_values_0_1_ : torch.Tensor, L_x_ : torch.nn.parameter.Parameter, L_y_ : torch.Tensor):
        #     l_nested_frame_values_0_1_ = L_nested_frame_values_0_1_
        #     l_x_ = L_x_
        #     l_y_ = L_y_
        #     a = l_nested_frame_values_0_1_ + l_nested_frame_values_0_1_;  l_nested_frame_values_0_1_ = None
        #     b = l_x_ + l_y_;  l_x_ = l_y_ = None
        #     mul = a * b;  a = b = None
        #     return (mul,)

        # Run fn with AOT
        torch._dynamo.reset()

        aot_fn = torch.compile(optimized_mod, backend="aot_eager")
        aot_fn(x, y)

    # Note: Dynamo recompilation guarding invalid grad
    #
    # This test is a spiritual equivalent to test_invalid_requires_grad_fake in test_autodispatch.py
    # The point of this test is to invoke aot_autograd in a way that would normally trigger an assertion
    # (This is what test_invalid_requires_grad_fake) does. However, the point of this test is to prove
    # that we do not hit this assertion, as dynamo recompiles correctly and protects this condition.
    #
    # Subnote: The reason for us having test_invalid_requires_grad_fake utilizing fake tensors
    # is because dynamo sends fake tensors down to aot_autograd.
    @patch("torch._functorch.config.debug_assert", True)
    def test_requires_grad_fake_via_dynamo_recompiles(self):
        # [FUNCTION DOC] Checks that Dynamo recompiles instead of tripping an AOT Autograd
        # assertion when requires_grad of an input changes. First run with (requires_grad,
        # requires_grad) then (requires_grad, no grad) inputs must give 2 frames and a
        # guard failure mentioning a requires_grad mismatch on y; after a reset, two runs
        # with the same flags compile once with no guard failure. Outputs and grads are
        # compared against eager.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns a tuple containing the sum of
            # its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a one-element tuple with x plus y.
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback: stores the first element of the
            # failure record in the enclosing failure_reason variable.
            nonlocal failure_reason
            failure_reason = failure[0]

        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))
        self.assertIn(
            """tensor 'y' requires_grad mismatch. expected requires_grad=1""",
            failure_reason,
        )

        # Reset failure reason
        failure_reason = None

        self.assertEqual(cc.frame_count, 2)

        torch._dynamo.reset()  # for new backend
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        fxz = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        compare_equal_outs_and_grads(self, F(), fxz, (x, z))
        self.assertEqual(cc.frame_count, 1)
        self.assertTrue(failure_reason is None)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_double_backward_with_compile(self):
        # [FUNCTION DOC] Checks double backward behavior under aot_autograd for grad_output
        # being a tensor requiring grad and for None. (1) autograd.grad with create_graph and
        # a second grad inside the compiled function raises RuntimeError mentioning no double
        # backward support; (2) second grad done outside the compiled function works; (3)
        # fully eager double backward on a compiled output raises the same error. Finally
        # checks the create_graph false case on a compiled function, using the grad_output
        # left from the last loop iteration (None).
        # Remove this test after we get double backward to actually work
        for grad_output in (torch.tensor(1.0, requires_grad=True), None):
            x = torch.tensor(1.0, requires_grad=True)
            err = "torch.compile with aot_autograd does not currently support double backward"

            # (1) double backward entirely inside compiled function
            # Graph break happens at autograd.grad, but y = x.sin().exp() already
            # went through AOT autograd. When autograd.grad(y, x, create_graph=True)
            # runs in eager, it backwards through AOT nodes, creating gx with grad_fn
            # pointing to CompiledFunctionBackward. Second autograd.grad(gx, x) fails.
            def f1(x):
                # [FUNCTION DOC] HELPER: Computes y as exp of sin of x, takes its grad wrt x
                # with create_graph, takes a second grad of that, and returns the first grad.
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                torch.autograd.grad(gx, x)
                return gx

            compiled_f1 = torch.compile(backend="aot_eager")(f1)
            with self.assertRaisesRegex(RuntimeError, err):
                compiled_f1(x)

            # (2) the second half of double backward outside compiled function
            # This works because autograd.grad inside the compiled function is traced,
            # so gx.grad_fn is CompiledFunctionBackward (not BackwardBackward).
            # The second autograd.grad(gx, x) is just first-level backward.
            def f2(x):
                # [FUNCTION DOC] HELPER: Computes exp of sin of x, returns its grad wrt x
                # with create_graph enabled.
                y = x.sin().exp()
                (gx,) = torch.autograd.grad(
                    y, x, create_graph=True, grad_outputs=grad_output
                )
                return gx

            compiled_f2 = torch.compile(backend="aot_eager")(f2)
            gx = compiled_f2(x)
            torch.autograd.grad(gx, x)

            # (3) double backward entirely outside compiled function
            # This still raises RuntimeError because autograd.grad is outside
            # the compiled function, so the graph break detection doesn't apply.
            def f3(x):
                # [FUNCTION DOC] HELPER: Returns exp of sin of x.
                y = x.sin().exp()
                return y

            compiled_f3 = torch.compile(backend="aot_eager")(f3)
            y = compiled_f3(x)
            (gx,) = torch.autograd.grad(
                y, x, create_graph=True, grad_outputs=grad_output
            )
            with self.assertRaisesRegex(RuntimeError, err):
                torch.autograd.grad(gx, x)

        # create_graph=False
        def f4(x):
            # [FUNCTION DOC] HELPER: Returns exp of sin of x.
            y = x.sin().exp()
            return y

        compiled_f4 = torch.compile(backend="aot_eager")(f4)
        x = torch.tensor(1.0, requires_grad=True)
        y = compiled_f4(x)
        (gx,) = torch.autograd.grad(y, x, create_graph=False, grad_outputs=grad_output)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles(self):
        # [FUNCTION DOC] Checks argument-dupe guards: with in-place mutating inputs, two calls
        # with distinct cloned inputs compile once with no guard failure; after a reset,
        # calling with the same tensor for both args then with different tensors compiles
        # twice and the guard failure message contains the identity check on x and y.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that truncates both inputs in place and returns their
            # sum in a tuple.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies in-place trunc to x and y and returns a tuple
                # with x plus y.
                x = x.trunc_()
                y = y.trunc_()
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        x1, x2, x3, x4 = x.clone(), x.clone(), x.clone(), x.clone()
        y = torch.randn(3, 3, requires_grad=True)
        y1, y2, y4 = y.clone(), y.clone(), y.clone()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element of the
            # failure in failure_reason.
            nonlocal failure_reason
            failure_reason = failure[0]

        fxy = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        # Note: to prevent a recompilation between the two calls,
        # we need to clone x and y on each use.
        # fxy mutates the input's metadata, so otherwise dynamo will end up recompiling.
        fxy(x1, y1)
        fxy(x2, y2)

        self.assertTrue(failure_reason is None)

        # Reset failure reason
        failure_reason = None

        self.assertEqual(cc.frame_count, 1)

        torch._dynamo.reset()  # for new backend
        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        fxx = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        fxx(x3, x3)
        fxx(x4, y4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""x is y""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg(self):
        # [FUNCTION DOC] Argument-dupe guard test with a module Parameter and extra non-tensor
        # (int) arguments. Calls with a duplicated tensor then distinct tensors must give 2
        # compiled frames and a failure message with the identity check between a and b;
        # repeated after a Dynamo reset with fresh tensors.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random 3x3 Parameter mean; forward truncates
            # inputs a and b in place and returns their sum plus mean times scalars e and f.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Parameter mean.
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, e, f):
                # [FUNCTION DOC] HELPER: Truncates a and b in place and returns (a + b + mean)
                # times e times f.
                a.trunc_()
                b.trunc_()
                return (a + b + self.mean) * e * f

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        _, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element
            # of the failure.
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, 2, 2)
        f(a2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """a is b""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        _, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(c3, c3, 3, 3)
        f(c4, d4, 3, 3)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""a is b""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_with_global(self):
        # [FUNCTION DOC] Like the non-tensor argument dupe test, but the module also reads a
        # free variable z that is bound to tensor a. Duplicate-then-distinct calls must
        # compile twice and the guard failure mentions the identity check between a and b.
        z = None

        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Parameter mean; forward truncates a and b in
            # place and adds the enclosing-scope tensor z to the sum, times scalars e and f.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Parameter mean.
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b, e, f):
                # [FUNCTION DOC] HELPER: Truncates a and b in place and returns (a + b + z +
                # mean) times e times f, where z is read from the enclosing test scope.
                a.trunc_()
                b.trunc_()
                return (a + b + z + self.mean) * e * f

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        z = a
        a1, a2 = a.clone(), a.clone()
        _, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element
            # of the failure.
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, 2, 2)
        f(a2, b2, 2, 2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """a is b""",
            failure_reason,
        )

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param_non_tensor_arg_list(self):
        # [FUNCTION DOC] Argument-dupe guard test where the non-tensor arguments are lists of
        # ints placed before the tensors. First sequence (same tensor twice, then different)
        # must give 2 frames and an identity guard failure between a and b; the second
        # sequence after a reset only asserts 2 frames.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Parameter mean; forward takes list args e and f
            # and tensors a, b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Parameter mean.
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, e, f, a, b):
                # [FUNCTION DOC] HELPER: Truncates a and b in place and returns (a + b + mean)
                # times the first elements of lists e and f.
                a.trunc_()
                b.trunc_()
                return (a + b + self.mean) * e[0] * f[0]

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        _, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element
            # of the failure.
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], a1, a1)
        f([3, 2, 1], [4, 5, 6], a2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """a is b""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        _, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f([3, 2, 1], [4, 5, 6], c3, c3)
        f([3, 2, 1], [4, 5, 6], c4, d4)
        self.assertEqual(cc.frame_count, 2)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args_param(self):
        # [FUNCTION DOC] Argument-dupe guard test with a Parameter and only tensor inputs.
        # Calls with the same tensor twice then two different tensors must compile twice with
        # an identity failure between a and b; repeated after a reset with fresh tensors.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with Parameter mean; forward truncates a and b in
            # place and returns their sum plus mean.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Parameter mean.
                super().__init__()
                self.mean = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Truncates a and b in place and returns a + b + mean.
                a.trunc_()
                b.trunc_()
                return a + b + self.mean

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2 = a.clone(), a.clone()
        _, b2 = b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element
            # of the failure.
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1)
        f(a2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """a is b""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        _, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(c3, c3)
        f(c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""a is b""", failure_reason)

    @patch("torch._functorch.config.debug_assert", True)
    def test_arg_dupe_via_dynamo_recompiles_many_args(self):
        # [FUNCTION DOC] Argument-dupe guard test with four tensor inputs. Passing the same
        # tensor for all inputs then partly different ones must compile twice with an identity
        # guard failure for a and b; after a reset, duplicating only c and d must yield 2
        # frames and a failure naming c and d.
        class F(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that truncates four inputs in place and returns their
            # sum in a tuple.
            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: Truncates a, b, c and d in place and returns a tuple
                # with their sum.
                a.trunc_()
                b.trunc_()
                c.trunc_()
                d.trunc_()
                return (a + b + c + d,)

        a = torch.randn(3, 3, requires_grad=True)
        b = torch.randn(3, 3, requires_grad=True)
        a1, a2, a3, a4 = a.clone(), a.clone(), a.clone(), a.clone()
        _, b2, b3, b4 = b.clone(), b.clone(), b.clone(), b.clone()

        failure_reason = None

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback storing the first element
            # of the failure.
            nonlocal failure_reason
            failure_reason = failure[0]

        self.assertTrue(failure_reason is None)

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a1, a1, a1, a1)
        f(a2, b2, b2, b2)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn(
            """a is b""",
            failure_reason,
        )

        torch._dynamo.reset()

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        c = torch.randn(3, 3, requires_grad=True)
        d = torch.randn(3, 3, requires_grad=True)
        c3, c4 = c.clone(), c.clone()
        _, d4 = d.clone(), d.clone()

        f = torch._dynamo.optimize(cc, guard_fail_fn=guard_fail_fn)(F())
        f(a3, b3, c3, c3)
        f(a4, b4, c4, d4)
        self.assertEqual(cc.frame_count, 2)
        self.assertIn("""c is d""", failure_reason)

    def test_alias_inputs(self):
        # [FUNCTION DOC] Compares eager and aot_eager results of a function with aliased views
        # of a tensor that is mutated before and after a data-dependent branch (graph break);
        # the compiled output must equal the eager one.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a one-element tensor, takes a view and a
            # squeezed alias, mutates the view, branches on its value, mutates again, and
            # returns the squeezed alias.
            a = torch.tensor([1])
            a = a[0:1]
            b = a.squeeze()
            a[0] = 0
            if a[0] < 1e5:
                pass
            a[0] = 2
            return b

        ref_output = fn()
        aot_fn = torch.compile(fn, backend="aot_eager")
        actual_output = aot_fn()
        self.assertEqual(ref_output, actual_output)

    def test_grad_inputs_alias_inputs(self):
        # [FUNCTION DOC] Compiles a custom autograd Function whose backward returns a saved
        # input as the gradient of an input (a gradient aliasing an input) and runs backward;
        # passes if no error.
        class Test(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function: forward saves x and
            # returns y; backward returns the saved x and the incoming grad as
            # gradients for the two inputs.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Saves x for backward and returns y unchanged.
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Returns the saved x as grad for the first input and
                # the incoming grad for the second.
                (x,) = ctx.saved_tensors
                return x, grad

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the custom Function to x and y.
            return Test.apply(x, y)

        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="aot_eager")
        out = compiled_fn(x, y)
        out.sum().backward()

    def test_joint_custom_pass(self):
        # [FUNCTION DOC] Checks the joint_custom_pass config hook. With an input not requiring
        # grad the pass must not be called; with a grad-requiring input it must be called once
        # with a GraphModule and a tuple of (primals list, tangents list). Then runs backward.
        is_called = False

        def joint_custom_pass(joint_gm: torch.fx.GraphModule, joint_inputs):
            # [FUNCTION DOC] HELPER: Custom joint pass: sets the is_called flag, asserts the
            # argument types (GraphModule, and a tuple whose first two items are lists), and
            # returns the graph module unchanged.
            nonlocal is_called
            is_called = True

            self.assertTrue(isinstance(joint_gm, torch.fx.GraphModule))

            self.assertTrue(isinstance(joint_inputs, tuple))
            # first input is list of primals
            self.assertTrue(isinstance(joint_inputs[0], list))
            # second input is list of tangents
            self.assertTrue(isinstance(joint_inputs[1], list))

            return joint_gm

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sine of its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sin of x.
                return x.sin()

        x = torch.randn(10, requires_grad=False)
        compiled_fn = torch.compile(M(), backend="aot_eager")

        with torch._functorch.config.patch("joint_custom_pass", joint_custom_pass):
            _ = compiled_fn(x)
        # x doesn't require grad, shouldn't trigger joint graph compiler
        self.assertFalse(is_called)

        y = torch.randn(10, requires_grad=True)
        with torch._functorch.config.patch("joint_custom_pass", joint_custom_pass):
            out = compiled_fn(y)
        # y requires grad, should trigger joint graph compiler
        self.assertTrue(is_called)
        out.sum().backward()

    @expectedFailureDynamic  # https://github.com/pytorch/pytorch/issues/103539
    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    @patch("torch._functorch.config.check_custom_op_aliasing", False)
    @patch("torch._functorch.config.debug_assert", True)
    def test_multiple_aot_autograd_calls_dupe_args(self):
        # [FUNCTION DOC] Compiles with a backend that splits the graph into two submodules
        # and wraps each with aot_module_simplified using a no-op compiler, applied to a
        # custom op whose outputs alias for small inputs (4 elements) but not for larger
        # (6). Runs both sizes; passes if no error. Marked as expected failure under
        # dynamic shapes (issue 103539).
        # this is just dealing with the fact that
        # aot_module_simplified expects submods to always return tuples/lists
        class WrapperModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper that calls a submodule and ensures the result is a
            # tuple or list, as aot_module_simplified requires.
            def __init__(self, mod):
                # [FUNCTION DOC] HELPER: Stores the wrapped module.
                super().__init__()
                self.mod = mod

            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Calls the wrapped module; returns lists and tuples as
                # is, otherwise wraps the single output in a tuple.
                out = self.mod(*args)
                if isinstance(out, (list, tuple)):
                    return out
                return (out,)

        def compile_submod(input_mod, args):
            # [FUNCTION DOC] HELPER: Compiles a submodule using aot_module_simplified
            # with a no-op compiler and returns a wrapper module holding the original
            # and the compiled versions.
            from functorch.compile import nop
            from torch._functorch.aot_autograd import aot_module_simplified

            class WrapperModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding the original module and its AOT compiled
                # version; forward runs the compiled one.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Stores input_mod as original and its
                    # aot_module_simplified result as submod.
                    super().__init__()
                    self.original = input_mod
                    self.submod = aot_module_simplified(input_mod, args, nop)

                def forward(self, *args):
                    # [FUNCTION DOC] HELPER: Runs the compiled submodule on
                    # the given arguments.
                    return self.submod(*args)

            return WrapperModule()

        def test_compile(fx_g, example_inps):
            # [FUNCTION DOC] HELPER: Custom backend: splits the graph at nodes containing
            # mul into two submodules, runs the first on the example inputs to obtain
            # inputs for the second, replaces both submodules by AOT compiled wrappers, and
            # returns the split module.
            split_gm = torch.fx.passes.split_module.split_module(
                fx_g, None, lambda node: 1 if "mul" in str(node) else 0
            )
            submod_1_inps = split_gm.submod_0(*example_inps)
            split_gm.submod_0 = compile_submod(
                WrapperModule(split_gm.submod_0), example_inps
            )
            split_gm.submod_1 = compile_submod(
                WrapperModule(split_gm.submod_1), submod_1_inps
            )
            return split_gm

        @torch.compile(backend=test_compile)
        def f(a):
            # [FUNCTION DOC] HELPER: Calls the custom maybe_dupe_op, and returns a tuple with
            # in-place multiplication of its two outputs.
            b, c = torch.ops.custom.maybe_dupe_op(a)
            return (b.mul_(c),)

        f(torch.ones(4))
        f(torch.ones(6))

    def test_nn_parameter_construction(self):
        # [FUNCTION DOC] Regression test (issue 99569): a function constructing an
        # nn.Parameter inside must produce the same result when compiled with aot_eager as
        # in eager mode.
        # https://github.com/pytorch/pytorch/issues/99569
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of x plus a freshly constructed
            # Parameter of ones.
            y = x.sin()
            z = torch.nn.Parameter(torch.ones(1))
            return y + z

        x = torch.rand((4, 4))

        opt_fn = torch.compile(fn, backend="aot_eager")
        self.assertTrue(torch._dynamo.testing.same(fn(x), opt_fn(x)))

    def test_aot_sequence_nr(self):
        # [FUNCTION DOC] Exports a conv, batch norm, relu, linear and L1 loss model, traces it
        # with _aot_export_function (joint, no tangents, preserving node metadata), and builds
        # a table of relative sequence numbers, original aten op, source function and forward
        # source function of each call node having a non-negative seq_nr. The table is
        # compared to a golden string.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Small model with Conv2d, BatchNorm2d, ReLU, Linear
            # and L1Loss layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers conv1, bn1, relu1,
                # fc1 and loss_fn.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                # [FUNCTION DOC] HELPER: Runs conv, batch norm and relu, adds the residual
                # input, flattens, applies the linear layer, and returns a tuple with the L1
                # loss against target.
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)

                return (output,)

        mod = Model()
        mod.train()
        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)

        # Use dynamo export to get the fx graph module
        g_mod, _ = torch._dynamo.export(mod, x, target)

        def _prepare_model_args():
            # [FUNCTION DOC] HELPER: Collects parameters and buffers of the exported module,
            # flattens them with pytree, builds a functional-call wrapper, and returns the
            # flat list and the wrapper.
            named_parameters = dict(g_mod.named_parameters(remove_duplicate=False))
            named_buffers = dict(g_mod.named_buffers(remove_duplicate=False))
            params_and_buffers = {
                **dict(named_parameters),
                **dict(named_buffers),
            }
            params_and_buffers_flat, params_spec = pytree.tree_flatten(
                params_and_buffers
            )
            params_len = len(params_and_buffers_flat)
            functional_call = create_functional_call(g_mod, params_spec, params_len)
            return params_and_buffers_flat, functional_call

        full_args, fn_to_trace = _prepare_model_args()
        param_and_buf_len = len(full_args)
        full_args.extend([x, target])

        # aot_export requires a graph mod input of fwd graph
        # returns the full fwd/bwd graph in graph mod format
        with torch.enable_grad(), fx_traceback.preserve_node_meta():
            fx_g, _, _, _ = _aot_export_function(
                fn_to_trace,
                full_args,
                decompositions=None,
                num_params_buffers=param_and_buf_len,
                no_tangents=True,
            )

        # Walk all the nodes in fx graph.
        # Write the resulting ops to a table
        min_seq_nr = -1
        seq_table = "SeqNr|OrigAten|SrcFn|FwdSrcFn\n"
        for node in fx_g.graph.nodes:
            if "call_" in node.op and "getitem" not in str(node.target):
                seq_nr = node.meta.get("seq_nr", -1)
                if seq_nr < 0:
                    continue
                if min_seq_nr < 0:
                    min_seq_nr = seq_nr
                source_fn_stack = node.meta.get("source_fn_stack", [])
                orig_aten = node.meta.get("original_aten", "")
                mod_name = ""
                if len(source_fn_stack) > 0:
                    mod_name = source_fn_stack[-1][0]
                # Make all seq_nr relative so it starts at 0
                seq_nr = seq_nr - min_seq_nr
                # For backward nodes, also test that metadata from the corresponding
                # forward node is copied over.
                fwd_source_fn_stack = node.meta.get("fwd_source_fn_stack", [])
                fwd_mod_name = ""
                if len(fwd_source_fn_stack):
                    fwd_mod_name = fwd_source_fn_stack[-1][0]
                seq_table = (
                    seq_table + f"{seq_nr}|{orig_aten}|{mod_name}|{fwd_mod_name}\n"
                )

        self.maxDiff = None
        self.assertExpectedInline(
            seq_table,
            dedent(
                """\
SeqNr|OrigAten|SrcFn|FwdSrcFn
0|aten.convolution.default|conv2d|
0|aten.add.Tensor|add_|
1|aten._native_batch_norm_legit_functional.default|batch_norm|
2|aten.relu.default|relu|
2|aten.detach.default|relu|
3|aten.add.Tensor|add|
4|aten.view.default|flatten|
5|aten.view.default|linear|
6|aten.t.default|linear|
7|aten.addmm.default|linear|
8|aten.view.default|linear|
9|aten.sub.Tensor|l1_loss|
10|aten.abs.default|l1_loss|
11|aten.mean.default|l1_loss|
11|aten.ones_like.default||l1_loss
11|aten.expand.default||l1_loss
11|aten.div.Scalar||l1_loss
10|aten.sgn.default||l1_loss
10|aten.mul.Tensor||l1_loss
8|aten.view.default||linear
7|aten.t.default||linear
7|aten.mm.default||linear
7|aten.t.default||linear
7|aten.mm.default||linear
7|aten.t.default||linear
7|aten.sum.dim_IntList||linear
7|aten.view.default||linear
6|aten.t.default||linear
5|aten.view.default||linear
4|aten.view.default||flatten
2|aten.detach.default||relu
2|aten.threshold_backward.default||relu
1|aten.native_batch_norm_backward.default||batch_norm
0|aten.convolution_backward.default||conv2d
11|aten.add.Tensor||
"""
            ),
        )

    def test_split_with_sizes_aot_autograd_cleans_up_traceback_meta(self):
        # [FUNCTION DOC] Checks that after setup_stacktrace_preservation_hooks is applied to
        # the grad_fn of split outputs and backward runs, current_meta contains neither
        # grad_fn_seq_nr nor in_grad_fn (the traceback metadata was cleaned up).
        from torch._functorch.aot_autograd import setup_stacktrace_preservation_hooks

        def fn(result, split_sizes):
            # [FUNCTION DOC] HELPER: Splits result into pieces with the given sizes (as a
            # list) and returns them.
            rs = torch.ops.aten.split_with_sizes(result, split_sizes.tolist())
            return rs

        example_inputs = (
            torch.randn(32, requires_grad=True),
            torch.tensor((7, 16, 9)),
        )
        outs = fn(*example_inputs)
        setup_stacktrace_preservation_hooks([out.grad_fn for out in outs])
        with fx_traceback.preserve_node_meta():
            (outs[0].sum() + outs[1].sum() + outs[2].sum()).backward()

        self.assertNotIn("grad_fn_seq_nr", fx_traceback.current_meta)
        self.assertNotIn("in_grad_fn", fx_traceback.current_meta)

    # https://github.com/pytorch/pytorch/issues/110121
    def test_aot_export_joint_simple_repro(self):
        # [FUNCTION DOC] Regression test (issue 110121): a backend calling
        # aot_export_joint_simple without the joint on a Linear module under fake mode (with
        # non-fake inputs allowed) returns a module whose output must equal eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear layer of 5 inputs and 7 outputs.
            def __init__(self, *args, **kwargs) -> None:
                # [FUNCTION DOC] HELPER: Forwards arguments to the base class and creates
                # the Linear layer.
                super().__init__(*args, **kwargs)
                self.linear = torch.nn.Linear(5, 7)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        def mini_backend(gm, sample_inputs):
            # [FUNCTION DOC] HELPER: Backend: detects the fake mode from sample inputs,
            # enables non-fake inputs, and returns the result of aot_export_joint_simple
            # with trace_joint false.
            from torch._functorch.aot_autograd import aot_export_joint_simple

            fake_mode = torch._dynamo.utils.detect_fake_mode(sample_inputs)

            with patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
                return aot_export_joint_simple(gm, sample_inputs, trace_joint=False)

        sample_inputs = [torch.rand((3, 4, 5))]
        model = Mod()
        m_compiled = torch.compile(model, backend=mini_backend)

        out_ref = model(*sample_inputs)
        out_test = m_compiled(*sample_inputs)
        self.assertEqual(out_ref, out_test)

    # set donated_buffer=False due to create_graph=True
    @torch._functorch.config.patch("donated_buffer", False)
    def test_eager_sequence_nr(self):
        # [FUNCTION DOC] Compiles a double-backward function (grad with create_graph) of a
        # conv model and profiles it. Collects profiler events with a sequence number,
        # building a table and a set of sequence numbers of Backward0/Backward1 events.
        # KNOWN ISSUE: the final assertion passes 13 as the failure message to assertTrue, so
        # it only checks the set is non-empty and not that it has 13 entries. The collected
        # table is not used.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Same model structure as in test_aot_sequence_nr (conv, batch
            # norm, relu, linear, L1 loss).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers conv1, bn1, relu1,
                # fc1 and loss_fn.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(1, 1),
                    stride=1,
                    padding="same",
                    bias=True,
                )
                self.bn1 = torch.nn.BatchNorm2d(num_features=16)
                self.relu1 = torch.nn.ReLU()
                self.fc1 = torch.nn.Linear(in_features=1638400, out_features=1)
                self.loss_fn = torch.nn.L1Loss()

            def forward(self, x, target):
                # [FUNCTION DOC] HELPER: Applies the layers with a residual add and flatten,
                # and returns a tuple with the loss.
                y = x
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = x + y
                x = torch.flatten(x)
                x = self.fc1(x)
                output = self.loss_fn(x, target)

                return (output,)

        def grad_with_create_graph(mod, x, target):
            # [FUNCTION DOC] HELPER: Runs the model and returns the gradient of its
            # first output wrt x with create_graph on, using the outer grad_output
            # tensor as the output gradient.
            y = mod(x, target)
            # Set create_graph=True to ensure that the sequence_nr
            # for backward ops continues to count down.
            (gx,) = torch.autograd.grad(
                y[0], x, create_graph=True, grad_outputs=grad_output
            )
            return gx

        x = torch.rand(100, 16, 32, 32, requires_grad=True)
        target = torch.rand(1)
        mod = Model()
        args = [mod, x, target]
        grad_output = torch.tensor(1.0, requires_grad=True)
        compiled_f1 = torch.compile(backend="aot_eager")(grad_with_create_graph)
        model_instance = compiled_f1
        with profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as kineto_prof:
            model_instance(*args)
        bwd_set = set()
        prof_str = "SeqNr|Thread|FwdThread|Name\n"
        for event in kineto_prof.events():
            if event.sequence_nr >= 0:
                prof_str = (
                    prof_str + f"{event.sequence_nr}|{event.thread}"
                    f"|{event.fwd_thread}|{event.name}|\n"
                )
                if re.search(r"Backward[01]", event.name):
                    bwd_set.add(event.sequence_nr)
        self.assertTrue(len(bwd_set), 13)

    def test_aot_grad_mode_mutation(self):
        # [FUNCTION DOC] For aot_eager and inductor backends, checks that a function disabling
        # grad mode inside compiled code leaves grad mode disabled after the call as in eager,
        # that outputs equal eager, the first output has no grad_fn and the second does, and
        # the backward of the second matches eager for a given tangent.
        for compiler in ["aot_eager", "inductor"]:

            def f(x):
                # [FUNCTION DOC] HELPER: Computes y as x times x, disables grad mode globally,
                # and returns a clone of y and y.
                y = x * x
                torch.set_grad_enabled(False)
                return y.clone(), y

            f_compiled = torch.compile(f, backend=compiler, fullgraph=True)

            torch.set_grad_enabled(True)
            x = torch.ones(3, requires_grad=True) * 3
            y_ref = f(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            y = f_compiled(x)
            self.assertEqual(torch.is_grad_enabled(), False)
            torch.set_grad_enabled(True)
            self.assertEqual(y_ref, y)

            self.assertIsNone(y_ref[0].grad_fn)
            self.assertIsNone(y[0].grad_fn)

            self.assertIsNotNone(y_ref[1].grad_fn)
            self.assertIsNotNone(y[1].grad_fn)

            # Check that the grad computed for the inputs, given the input, is the same
            # The tangent to `y[0]`, which has grad_required=False, is irrelevant
            self.assertEqual(
                sum(y_ref[1].grad_fn(torch.tensor([-1.0, 2.0, 0.0]))),
                sum(
                    x
                    for x in y[1].grad_fn.apply(None, torch.tensor([-1.0, 2.0, 0.0]))
                    if x is not None
                ),
            )

    def test_aot_autograd_raises_invalid_leaf_set(self):
        # [FUNCTION DOC] Checks that calling set_ on a leaf requiring grad inside an aot_eager
        # compiled function raises RuntimeError mentioning an in-place operation.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Replaces the storage of x with a tensor of ones via set_.
            x.set_(torch.ones(2))

        # We still want to make sure that this raises
        x = torch.ones(2, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "is being used in an in-place operation"
        ):
            f(x)

    def test_aot_autograd_expand_mutation_functionalizes(self):
        # [FUNCTION DOC] Checks that mutating an expanded view of an input is functionalized:
        # eager and aot_eager outputs match and the original input is mutated identically.
        def fn(x):
            # [FUNCTION DOC] HELPER: Expands x to a new leading dim of 3, adds 5 in place to
            # one slice, and returns the expanded tensor.
            y = x.expand(3, *x.shape)
            y[0, 0].add_(5)
            return y

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6)
        x_opt = x.detach().clone()
        self.assertEqual(fn(x), opt_fn(x_opt))
        self.assertEqual(x, x_opt)

    def test_aot_autograd_expand_mutation_backwards(self):
        # [FUNCTION DOC] Checks forward and backward correctness for in-place mul on a slice
        # of an expanded input: outputs, mutated inputs and gradient of z must match between
        # eager and aot_eager.
        def fn(x, z):
            # [FUNCTION DOC] HELPER: Expands x, multiplies one element of the view by 5 in
            # place, and returns the product with z.
            y = x.expand(3, *x.shape)
            y[1, 1].mul_(5)
            ret = y * z
            return ret

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6, dtype=torch.float)
        z = x.detach().clone()
        x_opt = x.detach().clone()
        z_opt = x.detach().clone()

        z.requires_grad = True
        z_opt.requires_grad = True

        res = fn(x, z)
        opt_res = opt_fn(x_opt, z_opt)

        self.assertEqual(res, opt_res)

        res.sum().backward()
        opt_res.sum().backward()

        self.assertEqual(x, x_opt)
        self.assertEqual(z.grad, z_opt.grad)

    def test_data_ptr_access_copy(self):
        # [FUNCTION DOC] Checks that copy.copy of a fake tensor works with unsafe data_ptr
        # access disabled; the copy has the same shape as the original.
        import torch._functorch.config as _config

        with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
            with FakeTensorMode():
                x = torch.randn(3)
                y = copy.copy(x)
        self.assertEqual(y.shape, x.shape)

    def test_data_ptr_access_fails_in_forward(self):
        # [FUNCTION DOC] Checks that calling data_ptr inside a custom op implementation raises
        # RuntimeError (Cannot access data pointer) under make_fx fake and symbolic tracing
        # and under fullgraph eager compile, for both graph inputs and intermediates.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_data_ptr_forward", "(Tensor x) -> Tensor", lib=lib
            )

            @torch.library.impl(
                "mylib::foo_data_ptr_forward", "CompositeImplicitAutograd", lib=lib
            )
            def _(x):
                # [FUNCTION DOC] HELPER: Op implementation: reads data_ptr of x (expected to
                # raise under fake tensors) and returns a clone.
                x.data_ptr()
                return x.clone()

            x = torch.randn(3)

            def data_ptr_graph_input(x):
                # [FUNCTION DOC] HELPER: Calls the custom op on the input directly.
                r0 = torch.ops.mylib.foo_data_ptr_forward(x)
                return r0

            def data_ptr_graph_intermediate(x):
                # [FUNCTION DOC] HELPER: Clones x then calls the custom op on the clone.
                y = x.clone()
                r0 = torch.ops.mylib.foo_data_ptr_forward(y)
                return r0

            tests = [data_ptr_graph_input, data_ptr_graph_intermediate]

            def ctx():
                # [FUNCTION DOC] HELPER: Returns an assertRaisesRegex context for RuntimeError
                # matching Cannot access data pointer.
                return self.assertRaisesRegex(
                    RuntimeError, "Cannot access data pointer"
                )

            for f in tests:
                with ctx():
                    make_fx(f, tracing_mode="fake")(x)
                with ctx():
                    make_fx(f, tracing_mode="symbolic")(x)
                with ctx():
                    torch.compile(f, backend="eager", fullgraph=True)(x)

    def test_data_ptr_access_fails_in_backward(self):
        # [FUNCTION DOC] Checks that a data_ptr call inside a custom autograd Function
        # backward raises RuntimeError (Cannot access data pointer) when compiled with
        # aot_eager, and that the backward was in fact called.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo_data_ptr_backward", "(Tensor x) -> Tensor", lib=lib
            )

            backward_called = False

            class Foo(torch.autograd.Function):
                # [CLASS DOC] HELPER: Autograd Function: forward clones x; backward records
                # that it ran, reads data_ptr of grad, and returns a clone.
                @staticmethod
                def forward(ctx, x):
                    # [FUNCTION DOC] HELPER: Returns a clone of x.
                    return x.clone()

                @staticmethod
                def backward(ctx, grad):
                    # [FUNCTION DOC] HELPER: Sets the enclosing backward_called flag, calls
                    # data_ptr on grad, returns a clone of grad.
                    nonlocal backward_called
                    backward_called = True
                    grad.data_ptr()
                    return grad.clone()

            @torch.library.impl(
                "mylib::foo_data_ptr_backward", "CompositeImplicitAutograd", lib=lib
            )
            def _(x):
                # [FUNCTION DOC] HELPER: Op implementation applying the Foo autograd Function.
                return Foo.apply(x)

            def f(x):
                # [FUNCTION DOC] HELPER: Calls the registered custom op on x.
                return torch.ops.mylib.foo_data_ptr_backward(x)

            x = torch.randn(3, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer"):
                torch.compile(f, backend="aot_eager", fullgraph=True)(x)
            self.assertTrue(backward_called)

    # We don't know how to catch multiple mutations to the same memory location
    @unittest.expectedFailure
    def test_aot_autograd_expand_mutation_error(self):
        # [FUNCTION DOC] Expected-failure test: writes via overlapping slices of an expanded
        # tensor are meant to raise in both eager and compiled mode, but multiple mutations of
        # the same memory location are not detected.
        def fn(x):
            # [FUNCTION DOC] HELPER: Expands x and adds 5 in place to a slice of the expanded
            # view, returning the view.
            y = x.expand(3, *x.shape)
            y[0:3, 0].add_(5)
            return y

        opt_fn = torch.compile(fn, backend="aot_eager")

        x = torch.arange(6)
        x_opt = x.detach().clone()
        with self.assertRaises(Exception):
            fn(x)
        with self.assertRaises(Exception):
            opt_fn(x_opt)

    @torch._functorch.config.patch(donated_buffer=True)
    def test_donated_buffer1(self):
        # [FUNCTION DOC] With donated_buffer on, compiles relu and checks the graph_compile
        # log for bw_donated_idxs equal to 1 (dynamic shapes, an extra symint) or 0.
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def relu(x):
            # [FUNCTION DOC] HELPER: Applies the functional relu.
            return torch.nn.functional.relu(x)

        with self.assertLogs(logger_name, level="INFO") as captured:
            relu(torch.rand([3, 3], requires_grad=True)).sum().backward()

        if is_dynamic_shape_test(self._testMethodName):
            # an extra symint exists
            expected_msg = "bw_donated_idxs=[1]"
        else:
            expected_msg = "bw_donated_idxs=[0]"

        # le is a donated buffer from relu
        FileCheck().check(expected_msg).run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer2(self):
        # [FUNCTION DOC] With donated_buffer on, checks the log shows an empty bw_donated_idxs
        # for a compiled matmul whose activation comes from an uncompiled add.
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        # we will reuse the graph for g across f1 and f2
        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def g(activation, param2):
            # [FUNCTION DOC] HELPER: Matrix multiplication of activation and param2.
            return torch.matmul(activation, param2)

        def f(inp, param1, param2):
            # [FUNCTION DOC] HELPER: Adds param1 to inp and calls compiled g with param2.
            # KNOWN ISSUE: the comment above g mentions f1 and f2, which do not exist.
            activation = inp + param1
            return g(activation, param2)

        inp = torch.ones(4, 4)
        param1 = torch.ones(4, 4, requires_grad=True)
        param2 = torch.ones(4, 4, requires_grad=True)

        with self.assertLogs(logger_name, level="INFO") as captured:
            f(inp, param1, param2).sum().backward()

        FileCheck().check("bw_donated_idxs=[]").run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer3(self):
        # [FUNCTION DOC] With donated_buffer on, as test_donated_buffer2 but using
        # exp for the activation (saves its output for backward); the log must show
        # an empty bw_donated_idxs.
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        # we will reuse the graph for g across f1 and f2
        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def g(activation, param2):
            # [FUNCTION DOC] HELPER: Matrix multiplication of activation and param2.
            return torch.matmul(activation, param2)

        def f(inp, param1, param2):
            # [FUNCTION DOC] HELPER: Computes exp of inp plus param1 and calls compiled
            # g with param2.
            # exp saves it output (the activation) for bw
            activation = torch.exp(inp + param1)
            return g(activation, param2)

        inp = torch.ones(4, 4)
        param1 = torch.ones(4, 4, requires_grad=True)
        param2 = torch.ones(4, 4, requires_grad=True)

        with self.assertLogs(logger_name, level="INFO") as captured:
            f(inp, param1, param2).sum().backward()

        FileCheck().check("bw_donated_idxs=[]").run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer4(self):
        # [FUNCTION DOC] With donated_buffer on, compiles a relu plus parameter module
        # and checks the log shows bw_donated_idxs equal to 0 (the relu mask from the
        # forward is donated).
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing relu of input plus a
            # zero-initialized 2x2 Parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the zeros Parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([2, 2]))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns relu of x plus the parameter.
                return torch.nn.functional.relu(x) + self.param

        mod = Mod()
        mod = torch.compile(mod)  # noqa: UNSPECIFIED_BACKEND

        inp = torch.ones([2, 2], requires_grad=True)

        with self.assertLogs(logger_name, level="INFO") as captured:
            mod(inp).sum().backward()

        # Forward graph:
        #   %primals_1 : [num_users=1] = placeholder[target=primals_1]
        #   %primals_2 : [num_users=1] = placeholder[target=primals_2]
        #   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%primals_2,), kwargs = {})
        #   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %primals_1), kwargs = {})
        #   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
        #   return [add, le]
        #
        # `le` is a donated buffer
        FileCheck().check("bw_donated_idxs=[0]").run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer5(self):
        # [FUNCTION DOC] With donated_buffer on, checks that only the relu mask is donated
        # but not an input saved for backward: expected bw_donated_idxs is 2 under dynamic
        # shapes, else 1.
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def f(x, z):
            # [FUNCTION DOC] HELPER: Views x as 2x3, applies relu to z, and returns mm of the
            # view with x plus the relu result.
            y = x.view(2, 3)
            z = torch.nn.functional.relu(z)
            return torch.mm(y, x) + z

        inp = [
            torch.rand([3, 2], requires_grad=True),
            torch.rand([2, 2], requires_grad=True),
        ]

        with self.assertLogs(logger_name, level="INFO") as captured:
            f(*inp).sum().backward()

        # Forward graph:
        #   %primals_1 : [num_users=3] = placeholder[target=primals_1]
        #   %primals_2 : [num_users=1] = placeholder[target=primals_2]
        #   %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%primals_1, [2, 3]), kwargs = {})
        #   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%primals_2,), kwargs = {})
        #   %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %primals_1), kwargs = {})
        #   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %relu), kwargs = {})
        #   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
        #   return [add, primals_1, le]
        #
        if is_dynamic_shape_test(self._testMethodName):
            # an extra symint exists before the saved tensors
            expected_msg = "bw_donated_idxs=[2]"
        else:
            expected_msg = "bw_donated_idxs=[1]"

        # `le` is a donated buffer but primals_1 is not.
        FileCheck().check(expected_msg).run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    @torch._dynamo.config.patch("graph_break_on_nn_param_ctor", False)
    def test_donated_buffer6(self):
        # [FUNCTION DOC] With donated_buffer on, compiles a function returning a Parameter
        # built in the graph and its sine, and checks an empty bw_donated_idxs. Returns early
        # (does nothing) for the dynamic-shape variant.
        if is_dynamic_shape_test(self._testMethodName):
            # parameters should not be dynamic shape
            # torch._dynamo.exc.Unsupported: Parameter not python_constant:
            #    SymNodeVariable() is not a constant
            return

        logger_name = "torch._functorch._aot_autograd.graph_compile"

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a Parameter from x plus 123 and returns it
            # and its sin.
            p = torch.nn.Parameter(x + 123)
            return p, p.sin()

        opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        x = torch.randn(16)

        with self.assertLogs(logger_name, level="INFO") as captured:
            p, r = opt(x)
            r.sum().backward()

        FileCheck().check("bw_donated_idxs=[]").run("\n".join(captured.output))

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer_with_retain_or_create_graph1(self):
        # [FUNCTION DOC] With donated_buffer on, repeated plain backward on fresh forward
        # results must run without error.
        # Gives non-empty bw_donated_idxs
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Module with zero Parameter; forward computes
                # relu(x) plus parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([3, 3]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of x plus the parameter.
                return torch.nn.functional.relu(x) + self.param

        inp = torch.randn(3, 3, requires_grad=True)

        mod = torch.compile(Mod())  # noqa: UNSPECIFIED_BACKEND
        for _ in range(5):
            mod(inp).sum().backward()

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer_with_retain_or_create_graph2(self):
        # [FUNCTION DOC] With donated_buffer on, repeated backward with retain_graph on the
        # same output followed by a final plain backward must run without error.
        # Gives non-empty bw_donated_idxs
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the zeros Parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([3, 3]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of x plus the parameter.
                return torch.nn.functional.relu(x) + self.param

        inp = torch.randn(3, 3, requires_grad=True)

        mod = torch.compile(Mod())  # noqa: UNSPECIFIED_BACKEND
        out = mod(inp).sum()
        for _ in range(5):
            out.backward(retain_graph=True)
        out.backward()

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer_with_retain_or_create_graph3(self):
        # [FUNCTION DOC] With donated_buffer on, a backward with create_graph is done, then a
        # fresh output is backpropagated repeatedly with retain_graph and finally plainly;
        # must run without error.
        # Gives non-empty bw_donated_idxs
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the zeros Parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([3, 3]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of x plus the parameter.
                return torch.nn.functional.relu(x) + self.param

        inp = torch.randn(3, 3, requires_grad=True)

        mod = torch.compile(Mod())  # noqa: UNSPECIFIED_BACKEND
        mod(inp).sum().backward(create_graph=True)
        out = mod(inp).sum()
        for _ in range(5):
            out.backward(retain_graph=True)
        out.backward()

    def test_autograd_function_tangent_mutation(self):
        # [FUNCTION DOC] Compiles with aot_eager (fullgraph) a custom Function whose backward
        # mutates its first tangent in place by copying the second into it. Checks outputs,
        # input and input gradients equal those of eager execution.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function returning two clones; backward copies
            # grad2 into grad1 in place and returns grad1.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns two separate clones of x.
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Copies grad2 into grad1 in place and returns grad1.
                return grad1.copy_(grad2)

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the Foo Function to x.
            return Foo.apply(x)

        x = torch.randn(4, requires_grad=True)
        x_ref = x.clone().detach().requires_grad_()

        out_ref = f(x_ref)
        out = torch.compile(f, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(out_ref, out)
        self.assertEqual(x_ref, x)

        (out[0] + out[1]).sum().backward()
        (out_ref[0] + out_ref[1]).sum().backward()

        self.assertEqual(x_ref.grad, x.grad)

    @torch._functorch.config.patch("donated_buffer", True)
    def test_donated_buffer_with_retain_or_create_graph4(self):
        # [FUNCTION DOC] With donated_buffer on, a plain backward works, but backward with
        # retain_graph on a second output must raise RuntimeError saying the backward was
        # compiled with non-empty donated buffers and requires create_graph and
        # retain_graph both false.
        # Gives non-empty bw_donated_idxs
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the zeros Parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros([3, 3]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of x plus the parameter.
                return torch.nn.functional.relu(x) + self.param

        inp = torch.randn(3, 3, requires_grad=True)

        mod = torch.compile(Mod())  # noqa: UNSPECIFIED_BACKEND
        mod(inp).sum().backward()
        out = mod(inp).sum()
        with self.assertRaisesRegex(
            RuntimeError,
            r"This backward function was compiled with non-empty donated "
            r"buffers which requires create_graph=False and retain_graph=False. "
            r"Please keep backward\(create_graph=False, retain_graph=False\) "
            r"across all backward\(\) function calls, or set "
            r"torch._functorch.config.donated_buffer=False to disable "
            r"donated buffer.",
        ):
            out.backward(retain_graph=True)

    def _get_guard_failure_on_overlapping_view_inputs(self, f, argsfn1, argsfn2):
        # [FUNCTION DOC] HELPER: Compiles f with Dynamo using aot_eager with dynamic shapes,
        # recording guard failures. Runs f eager and compiled with the first argument builder
        # then the second one on a 20-element tensor, asserting equal outputs, and expects
        # exactly one guard failure from the second call. Returns the failure message string.
        # Compile and run f twice, using the arguments generated by argsfn1 and argsfn2.
        #
        # This function expects that the second argument set will trigger a recompilation,
        # which shall be returned in the end.

        guard_failure = []

        def guard_fail_fn(failure):
            # [FUNCTION DOC] HELPER: Appends the first element of the failure to
            # the guard_failure list.
            nonlocal guard_failure
            guard_failure.append(failure[0])

        input = torch.ones(20)
        opt_input = input.clone().detach()

        opt_f = torch._dynamo.optimize(
            "aot_eager", dynamic=True, guard_fail_fn=guard_fail_fn
        )(f)

        out0 = f(*argsfn1(input))
        opt_out0 = opt_f(*argsfn1(opt_input))
        self.assertEqual(out0, opt_out0)

        out1 = f(*argsfn2(input))
        opt_out1 = opt_f(*argsfn2(opt_input))
        self.assertEqual(out1, opt_out1)

        # Check that we only have one instance of guard failure, and that it is due to
        # the overlapping state not matching.
        self.assertEqual(len(guard_failure), 1)
        return guard_failure[0]

    def test_inputs_overlapping_with_mutation_recompile(self):
        # [FUNCTION DOC] Checks that the overlap guard fails when a function mutating three
        # views is first run with two overlapping views and then with non-overlapping views;
        # the failure message must be the expected check_overlapping guard string.
        # Check that the overlap guard actually fails when we run the second time with
        # args that have no storage overlap.

        def f(*args):
            # [FUNCTION DOC] HELPER: Adds 1 in place to every argument and returns the first.
            for a in args:
                a.add_(1)
            return args[0]

        def overlapping_args(x):
            # [FUNCTION DOC] HELPER: Returns three slices of x, the second
            # and third overlapping.
            return x[:5], x[7:13], x[9:]

        def non_overlapping_args(x):
            # [FUNCTION DOC] HELPER: Returns three slices of x, none overlapping.
            return x[:5], x[7:13], x[13:15]

        guard_failure = self._get_guard_failure_on_overlapping_view_inputs(
            f, overlapping_args, non_overlapping_args
        )
        self.assertExpectedInline(
            guard_failure,
            """0/0: check_overlapping(overlapping=[args[1], args[2]], non_overlapping=[args[0]])""",
        )

    def test_different_inputs_overlapping_set_with_mutation(self):
        # [FUNCTION DOC] Checks the overlap guard fails when a second call has a larger set of
        # overlapping views than the first (a,b overlap, then a,b,c); expected guard failure
        # string names overlapping a, b and non-overlapping c, d.
        # Check that the overlap guard actually fails when we run the second time with
        # arguments whose overlapping set is a superset of the set of arguments used in
        # the first time.

        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Multiplies a by 2 in place and returns the sum of
            # the four inputs.
            a.mul_(2)
            return a + b + c + d

        def a_b_overlapping_args(x):
            # [FUNCTION DOC] HELPER: Returns four slices of x where only the
            # first two overlap.
            return x[:5], x[4:9], x[10:15], x[15:]

        def a_b_c_overlapping_args(x):
            # [FUNCTION DOC] HELPER: Returns four slices of x where the first three overlap.
            return x[:5], x[4:9], x[8:13], x[15:]

        guard_failure = self._get_guard_failure_on_overlapping_view_inputs(
            f, a_b_overlapping_args, a_b_c_overlapping_args
        )
        self.assertExpectedInline(
            guard_failure,
            """0/0: check_overlapping(overlapping=[a, b], non_overlapping=[c, d])""",
        )

    def _test_no_storage_overlap_guards(self, f, argsfn):
        # [FUNCTION DOC] HELPER: Compiles f with a recording backend on arguments built by
        # argsfn, with dynamic shapes. Asserts output equals eager, exactly one compiled
        # frame, and that none of the recorded AOT autograd guards is a StorageOverlap guard.
        # Compile f with aot_eager backend, and run it with the argument set returned by
        # argsfn function. Meanwhile, keep track of the aotautograd_gurads, so as to make
        # sure no StorageOverlap guard was added.

        class Compiler:
            # [CLASS DOC] HELPER: Backend wrapper that records the AOT autograd guards of the
            # tracing context and delegates to a CompileCounterWithBackend over aot_eager.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the compile counter with
                # the aot_eager backend.
                self.counter = CompileCounterWithBackend("aot_eager")

            def __call__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Saves the aotautograd guards from the
                # current TracingContext in guards and forwards the call to the
                # counter; returns its result.
                # Instead of checking here, we need to check afterwards, since the
                # StorageOverlap guard is only added later.
                self.guards = TracingContext.get().guards_context.aotautograd_guards
                return self.counter(*args, **kwargs)

        compiler = Compiler()

        input = torch.arange(20)
        opt_input = input.clone().detach()

        out = f(*argsfn(input))
        opt_out = torch.compile(f, backend=compiler, dynamic=True)(*argsfn(opt_input))
        self.assertEqual(out, opt_out)

        self.assertEqual(compiler.counter.frame_count, 1)

        # Check none of the AOTAutograd guards are StorageOverlap guards.
        for g in compiler.guards:
            self.assertNotIsInstance(g, StorageOverlap)

    def test_no_storage_overlap_guards_no_mutation(self):
        # [FUNCTION DOC] Checks no StorageOverlap guard is added when overlapping inputs
        # are not mutated.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        def overlapping_args(input):
            # [FUNCTION DOC] HELPER: Returns two overlapping slices of the input.
            return input[:10], input[5:15]

        self._test_no_storage_overlap_guards(f, overlapping_args)

    def test_no_storage_overlap_guards_no_aliasing(self):
        # [FUNCTION DOC] Checks no StorageOverlap guard is added when mutated inputs do
        # not share storage.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Adds 1 in place to both inputs and returns a.
            a.add_(1)
            b.add_(1)
            return a

        def non_overlapping_args(input):
            # [FUNCTION DOC] HELPER: Returns a slice of input and a slice of a
            # separate new tensor.
            return input[:10], torch.arange(20)[5:15]

        self._test_no_storage_overlap_guards(f, non_overlapping_args)

    def test_inputs_overlapping_with_mutation_stress(self):
        # [FUNCTION DOC] Stress test for the StorageOverlap guard with 101 views (100 disjoint
        # plus one overlapping half of them) mutated in place under dynamic shapes. Asserts
        # output equals eager, no ShapeEnv guard text mentions compute_overlapping_inputs, and
        # fewer than 1000 ShapeEnv guards.
        # KNOWN ISSUE: comments say no more than 500 guards but the assertion allows 1000.
        # Stress test for StorageOverlap guard.
        #
        # Create 100 non-overlapping tensor views, and an extra one that overlaps with
        # the first 50 of them. Then, make sure that none of the produced ShapeEnv
        # guards came from the overlapping computation.

        def f(*args):
            # [FUNCTION DOC] HELPER: Adds 1 in place to every argument and returns the first.
            for a in args:
                a.add_(1)
            return args[0]

        def overlapping_args(input):
            # [FUNCTION DOC] HELPER: Returns 100 disjoint chunks of size 10 plus one slice
            # overlapping half of them.
            return (
                # 100 non-overlapping tensors of size 10.
                *input.split(10),
                # A tensor that overlaps with half of the tensors above.
                input[4:44],
            )

        class Compiler:
            # [CLASS DOC] HELPER: Backend wrapper that records the CompileContext and
            # delegates to a CompileCounterWithBackend over aot_eager.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the compile counter with
                # the aot_eager backend.
                self.counter = CompileCounterWithBackend("aot_eager")

            def __call__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Saves the current CompileContext in compile_context
                # and forwards the call to the counter; returns its result.
                self.compile_context = CompileContext.get()
                return self.counter(*args, **kwargs)

        compiler = Compiler()
        opt_f = torch.compile(f, backend=compiler, dynamic=True)

        input = torch.arange(1_000)
        opt_input = input.clone().detach()

        out0 = f(*overlapping_args(input))
        opt_out0 = opt_f(*overlapping_args(opt_input))
        self.assertEqual(out0, opt_out0)

        # Check that none of the produced ShapeEnv guards came from compute_overlapping_inputs
        # function.
        overlapping_computation_fn = "compute_overlapping_inputs"
        shape_env_guards = compiler.compile_context.shape_env_guards
        for g in shape_env_guards:
            self.assertNotIn(overlapping_computation_fn, g)
        # Check that we have no more than 500 ShapeEnv guards.
        #
        # Note: this is an arbitrary number. So, we might have to change it in the future.
        # However, at the time this change was introduced, it went down from 15154 to 403.
        self.assertLess(len(shape_env_guards), 1000)

    # See # https://github.com/pytorch/pytorch/issues/164814
    def test_aot_autograd_stride_reconstruction_on_zero_dim_dynamic_shaped_tensor(
        self,
    ) -> None:
        # [FUNCTION DOC] Regression test (issue 164814): a function producing a 0-dim
        # dynamic shape tensor via unique and squeeze, multiplied by a grad-requiring
        # sentinel, is compiled with aot_eager and inductor. Eager, aot_eager and inductor
        # values must match and, for inductor, strides must equal eager's, both with and
        # without the squeeze.
        def repro(sentinel: torch.Tensor, skip_squeeze: bool = False) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds a 1-element tensor via unique, reshapes to one
            # element, optionally squeezes to 0-dim, and returns it times the sentinel.
            x = torch.unique(torch.ones(1))
            x = torch.reshape(x, [1])
            if not skip_squeeze:
                x = torch.squeeze(x)  # 0-d tensor
            return x * sentinel

        # Grad required to trigger the issue (need to replay stride)
        sentinel = torch.tensor(1.0, requires_grad=True)
        eager_sq = repro(sentinel)
        comp_aot_sq = torch.compile(repro, backend="aot_eager", fullgraph=True)(
            sentinel
        )
        comp_ind_sq = torch.compile(repro, backend="inductor", fullgraph=True)(sentinel)
        self.assertEqual(eager_sq, comp_aot_sq)
        self.assertEqual(eager_sq, comp_ind_sq)
        self.assertEqual(eager_sq.stride(), comp_ind_sq.stride())

        # Now check semantics preserved when skipping squeeze
        eager_no_sq = repro(sentinel, skip_squeeze=True)
        comp_aot_no_sq = torch.compile(repro, backend="aot_eager", fullgraph=True)(
            sentinel, skip_squeeze=True
        )
        comp_ind_no_sq = torch.compile(repro, backend="inductor", fullgraph=True)(
            sentinel, skip_squeeze=True
        )
        self.assertEqual(eager_no_sq, comp_aot_no_sq)
        self.assertEqual(eager_no_sq, comp_ind_no_sq)
        self.assertEqual(eager_no_sq.stride(), comp_ind_no_sq.stride())

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unbacked_activation_specialized_in_inductor(self):
        # [FUNCTION DOC] Compiles with fullgraph and dynamic shapes a generated program
        # using nonzero (unbacked shape) and checks its result is close to eager's.
        # Resets Dynamo first.
        """Test compilation with unbacked operations like nonzero."""
        torch._dynamo.reset()

        def fuzzed_program(arg_0, sentinel):
            # [FUNCTION DOC] HELPER: Builds integer tensors, combines with add, mul and a
            # nonzero result, divides the input by it, scales by the sentinel, and returns the
            # real part if complex.
            var_node_1 = arg_0
            var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
            var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
            var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)
            var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
            var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)
            var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
            var_node_8 = torch.nonzero(var_node_9)
            var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
            var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True)
        arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)
        args = (arg_0,) + (sentinel,)

        result_original = fuzzed_program(*args)

        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)  # noqa: UNSPECIFIED_BACKEND
        result_compiled = compiled_program(*args)

        self.assertTrue(torch.allclose(result_original, result_compiled))

    def test_layer_norm_double_backward_dynamic_shapes(self):
        # [FUNCTION DOC] Checks that a layer norm model computing a create_graph gradient,
        # traced with make_fx symbolic mode and compiled with dynamic inductor, handles batch
        # sizes 10, 20, 30 and 40 with the right output shapes and a single compiled frame.
        """
        Test that layer_norm with create_graph=True (double backward) works
        with dynamic shapes without recompilations.

        This tests the fix for layer_norm_double_backward to use SymInt properly
        so that dynamic batch sizes don't cause specialization/recompilation.
        """

        class LayerNormGradModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model applying LayerNorm and returning the output and its
            # gradient wrt the input, computed with create_graph.
            def __init__(self, dim=64):
                # [FUNCTION DOC] HELPER: Creates a LayerNorm over dim features (default 64).
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(dim)

            def forward(self, x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Clones x requiring grad, applies layer norm, computes
                # the gradient with ones as output grads and create_graph, and returns the
                # output and gradient.
                x = x.clone().requires_grad_(True)
                y = self.layer_norm(x)
                grad_outputs = torch.ones_like(y)
                # create_graph=True triggers layer_norm_double_backward
                (grad,) = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    allow_unused=True,
                )
                return y, grad

        model = LayerNormGradModel(dim=64)

        # Trace with symbolic shapes
        example_input = torch.randn(10, 64)
        fx_model = make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(example_input)

        # Compile with dynamic=True and a counter to track recompilations
        cnt = CompileCounterWithBackend(backend="inductor")
        compiled_model = torch.compile(
            fx_model, dynamic=True, fullgraph=True, backend=cnt
        )

        batch_sizes = [10, 20, 30, 40]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 64)
            y, grad = compiled_model(x)
            self.assertEqual(y.shape, (batch_size, 64))
            self.assertEqual(grad.shape, (batch_size, 64))

        # Should only compile once regardless of batch size changes
        self.assertEqual(cnt.frame_count, 1)

    @fx_config.patch(translation_validation=False)
    def test_partitioner_saves_unreplaced_input_symbols_for_bw(self):
        # [FUNCTION DOC] Builds by hand a ShapeEnv with a backed symbol s0, an unbacked u0
        # replaced by s0 then renamed to u1, and a derived symbol, and a joint graph using
        # them. Extracts the backward module with the partitioner and checks its placeholders
        # bind s0, u0 and u0 plus 1; then lowers the backward with GraphLowering and checks
        # the graph inputs are s0, u1, u1 plus 1 (equal to s0, s0, s0 plus 1 after renaming).
        shape_env = ShapeEnv(should_record_events=False)
        s0 = make_symbol(SymT.SIZE, 0, integer=True)
        shape_env.var_to_range[s0] = ValueRanges(2, 10)
        shape_env.add_backed_var_to_val(s0, 3)
        s0_sym = shape_env.create_symintnode(s0, hint=3)

        u0_sym = shape_env.create_unbacked_symint()
        u0 = u0_sym.node._expr
        shape_env._set_replacement(u0, s0, "test")
        u1_sym = shape_env.create_unbacked_symint()
        u1 = u1_sym.node._expr
        shape_env._rename_unbacked_to(u0, u1)
        derived_sym = shape_env.create_symintnode(u0 + 1, hint=None)

        graph = torch.fx.Graph()
        s0_node = graph.placeholder("primals_1")
        s0_node.meta["val"] = s0_sym
        u0_node = graph.call_function(operator.add, (s0_node, 0))
        u0_node.meta["val"] = u0_sym
        u0_node.meta["unbacked_bindings"] = {u0: ()}
        derived_node = graph.call_function(operator.add, (u0_node, 1))
        derived_node.meta["val"] = derived_sym
        output = graph.output((s0_node, derived_node))
        output.meta["desc"] = [None, None]
        joint = torch.fx.GraphModule({}, graph)

        _, bw_graph = _extract_fwd_bwd_modules(
            joint, [], [derived_node], num_fwd_outputs=1
        )
        bw_placeholders = list(bw_graph.graph.find_nodes(op="placeholder"))
        # The backward binds the unreplaced symbol u0 (needed by runtime
        # assertions, which preserve raw placeholder expressions) *and* the
        # replacement target s0 (needed by sizevar codegen and FxGraphCache
        # guards, which use ShapeEnv replacements). Binding only u0 leaves s0
        # undefined in the backward, surfacing as a KeyError during backward
        # FxGraphCache guard evaluation.
        self.assertEqual(
            [node.meta["val"].node._expr for node in bw_placeholders],
            [s0, u0, u0 + 1],
        )

        graph_inputs = [node.meta["val"] for node in bw_placeholders]
        lowering = GraphLowering(
            bw_graph, graph_inputs, shape_env=shape_env, is_backward=True
        )
        with V.set_graph_handler(lowering):
            lowering.run(*graph_inputs)

        self.assertEqual(
            [lowering.graph_inputs[name] for name in lowering.graph_input_names],
            [s0, u1, u1 + 1],
        )
        self.assertEqual(
            [
                lowering.graph_inputs[name].xreplace({u1: s0})
                for name in lowering.graph_input_names
            ],
            [s0, s0, s0 + 1],
        )

    def test_batched_matmul_inference_mode(self):
        # [FUNCTION DOC] Regression test: compiling a function doing batched matmul inside
        # inference mode (after a Dynamo reset) must not crash and returns a tensor.
        # Regression test for torch.compile crash with batched matmul in inference_mode
        x = torch.randn(2, 4, 8)
        w = torch.randn(8, 8)

        def f(x, w):
            # [FUNCTION DOC] HELPER: Returns the sum of x matmul w computed
            # under inference mode.
            with torch.inference_mode():
                return (x @ w).sum()

        torch._dynamo.reset()
        result = torch.compile(f)(x, w)  # noqa: UNSPECIFIED_BACKEND
        self.assertIsInstance(result, torch.Tensor)


class AotAutogradFallbackTestsDevice(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Device-generic test class (instantiated per device type at the end of the
    # module). Currently holds one accelerator-only partitioner test.
    @onlyAccelerator
    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    @patch.object(torch._dynamo.config, "capture_dynamic_output_shape_ops", True)
    def test_partitioner_filters_symbool_saved_for_bw(self, device):
        # [FUNCTION DOC] Accelerator-only test: patches the cross-rank decision sync to inject
        # SymBool nodes into saved values, with data dependent shapes. Compiling and running
        # backward must succeed and give a gradient. Then uses aot_function with default
        # partition and dynamic shapes, capturing the forward graph, and asserts exactly one
        # forward graph whose outputs contain no SymBool node. Takes the device argument.
        """When cross-rank sync injects SymBool nodes (whose only consumers are
        _assert_scalar) into saved_values, the partitioner must filter them out
        so they don't become backward graph inputs that Inductor can't lower.

        In multi-GPU MoE training, different EP ranks can produce different
        min-cut partitioner decisions due to data-dependent shapes. The
        cross-rank sync (_sync_decision_cross_ranks) picks one rank's decision
        for all, which may include SymBool assertion nodes that weren't in
        another rank's saved set. We simulate this by patching
        _sync_decision_cross_ranks to inject all SymBool nodes.

        Regression test for https://github.com/pytorch/pytorch/pull/179315
        """
        import torch._functorch.config as functorch_config
        import torch._functorch.partitioners as P

        # Simulate cross-rank sync picking a rank that saved SymBool nodes
        def inject_symbool_nodes(joint_graph, saved_values):
            # [FUNCTION DOC] HELPER: Replacement for the sync function: appends to
            # saved_values every graph node with a SymBool value not already saved, and
            # returns the new list.
            for node in joint_graph.nodes:
                val = node.meta.get("val")
                if isinstance(val, torch.SymBool) and node not in saved_values:
                    saved_values = saved_values + [node]
            return saved_values

        def fn(x, splits_tensor, mask):
            # [FUNCTION DOC] HELPER: Converts splits tensor to a list, selects rows by nonzero
            # mask, splits them by the sizes, doubles the chunks, concatenates and sums.
            splits = splits_tensor.tolist()
            indices = torch.nonzero(mask).squeeze(-1)
            y = x[indices]
            # split verifies sum(splits) == y.shape[0], creating an
            # Equality SymBool node consumed only by _assert_scalar
            chunks = torch.split(y, splits, dim=0)
            return torch.cat([c * 2.0 for c in chunks], dim=0).sum()

        x = torch.randn(20, 16, requires_grad=True, device=device)
        splits_tensor = torch.tensor([3, 5], device=device)
        mask = torch.zeros(20, dtype=torch.bool, device=device)
        mask[:8] = True

        with (
            patch.object(P, "_sync_decision_cross_ranks", inject_symbool_nodes),
            patch.object(functorch_config, "_sync_decision_cross_ranks", True),
        ):
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            loss = compiled_fn(x, splits_tensor, mask)
            # Without the fix, this raises:
            # InductorError: Unsupported inductor graph input type:
            #   <class 'sympy.core.relational.GreaterThan'>
            loss.backward()
        self.assertIsNotNone(x.grad)

        captured_fw_graphs = []

        def fw_compiler(gm, inputs):
            # [FUNCTION DOC] HELPER: Forward compiler that records the graph module and
            # returns it unchanged.
            captured_fw_graphs.append(gm)
            return gm

        from functorch.compile import aot_function, default_partition, nop

        x = torch.randn(20, 16, requires_grad=True, device=device)
        compiled_fn = aot_function(
            fn,
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            partition_fn=default_partition,
            dynamic=True,
        )
        loss = compiled_fn(x, splits_tensor, mask)
        loss.backward()
        self.assertEqual(len(captured_fw_graphs), 1)
        output_node = next(
            node for node in captured_fw_graphs[0].graph.nodes if node.op == "output"
        )
        outputs = output_node.args[0]
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        self.assertFalse(
            any(
                isinstance(node, torch.fx.Node)
                and isinstance(node.meta.get("val"), torch.SymBool)
                for node in outputs
            )
        )


instantiate_device_type_tests(AotAutogradFallbackTestsDevice, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
