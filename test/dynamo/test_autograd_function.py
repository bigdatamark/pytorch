# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo capture of torch.autograd.Function. Defines module-level fixture
# Functions and nn.Modules (custom forward/backward, setup_context, materialized grads, save/mark
# calls) plus two test cases: AutogradFunctionTests (compiled vs eager results, gradients, graph
# breaks, side effects in backward, triton kernels, keyword apply, expected captured graphs) and
# AutogradFunctionFunctorchTests (torch.func vmap/grad interplay). Import-time side effects:
# computes device_type from the current accelerator (cpu fallback) and imports triton and
# add_kernel when a GPU is available. Running the file as a script calls run_tests.
import copy
import math
from dataclasses import dataclass
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch.testing._internal.triton_utils import HAS_GPU, requires_gpu


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)

if HAS_GPU:
    import triton

    from torch.testing._internal.triton_utils import add_kernel


class CustomFunc1(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Custom autograd Function whose forward returns the input
    # doubled (foo + foo) and whose backward passes the incoming gradient through unchanged (not
    # the true derivative, which would be 2x).
    @staticmethod
    def forward(ctx, foo):
        # [FUNCTION DOC] HELPER: Forward: returns foo + foo. ctx is unused.
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: returns grad_output unchanged as the
        # single input gradient.
        return grad_output


class CustomFunc3(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Custom Function whose forward contains an explicit Dynamo
    # graph break, used to check that Dynamo handles breaks inside an autograd Function forward.
    # Saves the result for backward.
    # Test there is graph break in forward function
    @staticmethod
    def forward(ctx, foo):
        # [FUNCTION DOC] HELPER: Forward: computes foo + foo, calls the Dynamo graph_break marker,
        # adds foo again (result is 3 * foo), saves the result on ctx and returns it.
        result = foo + foo
        torch._dynamo.graph_break()
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: retrieves the saved result and scales
        # grad_output by the square root of the saved tensor's element count (an arbitrary
        # but deterministic factor).
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class Module1(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module that applies a freshly instantiated CustomFunc1 via its
    # apply method in forward (the legacy instance-apply call style).
    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns CustomFunc1 applied to foo, called through a new instance
        # of the Function.
        return CustomFunc1().apply(foo)


class Module2(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module that stores CustomFunc1.apply as an attribute and calls it
    # in forward (bound apply stored on the module).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the module and stores CustomFunc1.apply in
        # the attribute fn.
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns the stored apply callable called on foo.
        return self.fn(foo)


class Module3(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module identical in behavior to Module1: applies CustomFunc1
    # through a new instance.
    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns CustomFunc1 applied to foo through a new instance.
        return CustomFunc1().apply(foo)


class Module4(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module identical in behavior to Module2: stores CustomFunc1.apply
    # as attribute fn.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the module and stores CustomFunc1.apply in
        # the attribute fn.
        super().__init__()
        self.fn = CustomFunc1.apply

    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns the stored apply callable called on foo.
        return self.fn(foo)


class Module5(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying CustomFunc3 (forward containing a graph break)
    # through a new instance.
    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns CustomFunc3 applied to foo through a new instance.
        return CustomFunc3().apply(foo)


class Module6(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module that stores CustomFunc3.apply as attribute fn and calls
    # it in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the module and stores CustomFunc3.apply in
        # the attribute fn.
        super().__init__()
        self.fn = CustomFunc3.apply

    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns the stored apply callable called on foo.
        return self.fn(foo)


class LinearFunction(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Linear layer written in the new-style autograd Function
    # form where forward has no ctx argument and a separate setup_context static method saves
    # tensors for backward.
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        # [FUNCTION DOC] HELPER: Forward (no ctx): computes input times the transposed weight;
        # if bias is not None, adds the bias broadcast across rows to the output in place.
        # Returns the output.
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        # [FUNCTION DOC] HELPER: Saves input, weight and bias on ctx for backward. Called by
        # autograd with the forward inputs tuple and the output (output is unused).
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: computes gradients only for inputs that need them (per
        # ctx.needs_input_grad): grad_input = grad_output @ weight, grad_weight = grad_output^T @
        # input, grad_bias = sum over dim 0 (only when bias is not None). Unneeded gradients are
        # None. Returns the three-tuple.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class ModuleLinear(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module wrapping LinearFunction.apply.
    def forward(self, input, weight, bias=None):
        # [FUNCTION DOC] HELPER: Returns LinearFunction applied to input, weight and bias (bias
        # defaults to None).
        return LinearFunction.apply(input, weight, bias)


class MaterializingGradFunction(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Function with two outputs that disables gradient
    # materialization, so undefined output gradients arrive in backward as None instead
    # of zero tensors.
    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Forward: calls set_materialize_grads(False) on ctx and returns
        # two clones of x.
        ctx.set_materialize_grads(False)
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_out1, grad_out2):
        # [FUNCTION DOC] HELPER: Backward: returns the two incoming output gradients unchanged
        # (either may be None since materialization is off).
        return grad_out1, grad_out2


class MaterializingGradModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying MaterializingGradFunction to its input.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the pair of outputs from MaterializingGradFunction
        # applied to x.
        return MaterializingGradFunction.apply(x)


class CustomFuncBwdPrintGraphBreak(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Function whose backward calls print, which Dynamo cannot
    # trace, to exercise graph-break behavior in backward.
    @staticmethod
    def forward(ctx, foo):
        # [FUNCTION DOC] HELPER: Forward: returns torch.add(foo, foo).
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: prints a message (side effect on stdout) and
        # returns grad_output unchanged.
        print("graph break!")
        return grad_output


class CustomFuncBwdPrintModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying CustomFuncBwdPrintGraphBreak.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns CustomFuncBwdPrintGraphBreak applied to x.
        return CustomFuncBwdPrintGraphBreak.apply(x)


class CustomFuncStrideBwd(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Function whose backward multiplies the gradient by the
    # last stride of the gradient tensor, to test that stride queries inside backward are traced.
    @staticmethod
    def forward(ctx, foo):
        # [FUNCTION DOC] HELPER: Forward: returns torch.add(foo, foo).
        return torch.add(foo, foo)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: returns grad_output multiplied by the last element of
        # its stride tuple.
        return grad_output * grad_output.stride()[-1]


class CustomFuncStrideModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying CustomFuncStrideBwd.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns CustomFuncStrideBwd applied to x.
        return CustomFuncStrideBwd.apply(x)


class CustomFuncSaveForBwd(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Function that saves its forward result with
    # save_for_backward and uses it in backward.
    @staticmethod
    def forward(ctx, foo):
        # [FUNCTION DOC] HELPER: Forward: computes foo + foo + foo (3 * foo), saves the result on
        # ctx and returns it.
        result = foo + foo
        result = result + foo
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: reads the saved result and scales grad_output by the
        # square root of its element count.
        (result,) = ctx.saved_tensors
        return grad_output * math.sqrt(result.numel())


class SaveForBwdModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module applying CustomFuncSaveForBwd through a new instance.
    def forward(self, foo):
        # [FUNCTION DOC] HELPER: Returns CustomFuncSaveForBwd applied to foo through
        # a new instance.
        return CustomFuncSaveForBwd().apply(foo)


class ContextSaveAndMark(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture. Function whose forward, under no_grad, first saves
    # the input for backward and then marks the same tensor non-differentiable, returning
    # the input itself.
    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Forward: inside a no_grad block calls save_for_backward(x) then
        # mark_non_differentiable(x) and returns x (the input object is returned as the output).
        with torch.no_grad():
            ctx.save_for_backward(x)
            ctx.mark_non_differentiable(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: returns grad_output unchanged.
        return grad_output


class ContextMarkAndSave(torch.autograd.Function):
    # [CLASS DOC] HELPER: Test fixture, like ContextSaveAndMark but with the two context calls in
    # the opposite order (mark non-differentiable first, then save for backward).
    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Forward: inside a no_grad block calls mark_non_differentiable(x)
        # then save_for_backward(x) and returns x.
        with torch.no_grad():
            ctx.mark_non_differentiable(x)
            ctx.save_for_backward(x)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Backward: returns grad_output unchanged.
        return grad_output


class ModuleWithGradFunc(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module that holds the apply method of a given autograd Function
    # class and calls it in forward.
    def __init__(self, func):
        # [FUNCTION DOC] HELPER: Initializes the module and stores func.apply in the attribute f.
        super().__init__()
        self.f = func.apply

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the stored apply callable called on x.
        return self.f(x)


class AutogradFunctionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for Dynamo's capture of torch.autograd.Function. Tests build Function
    # subclasses (many of them nested in the test body), compile a wrapper with torch.compile
    # using the eager, aot_eager, inductor or a counting backend, and compare results, gradients,
    # graph-break counts and sometimes the printed captured graph against eager execution. Module-
    # level fixtures (CustomFunc*, Module*, etc.) are defined above this class.
    # Sound behaviors, tested for working capture
    def test_function_ctx_does_not_instantiate_function(self):
        # [FUNCTION DOC] Checks that compiling a call to a Function's apply never instantiates the
        # deprecated Function object: Function.__init__ is patched to raise AssertionError and the
        # fullgraph eager-compiled result must equal the eager result.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Function that stores a constant scale (3) on ctx
            # in forward, returns sin of x, and multiplies the gradient by ctx.scale in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: sets ctx.scale to 3 and returns x.sin().
                ctx.scale = 3
                return x.sin()

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output times ctx.scale.
                return grad_output * ctx.scale

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls Foo.apply on x.
            return Foo.apply(x)

        x = torch.randn(3, requires_grad=True)
        ref = fn(x)
        with mock.patch.object(
            torch.autograd.Function,
            "__init__",
            side_effect=AssertionError("deprecated Function instantiated"),
        ):
            res = torch.compile(fn, backend="eager", fullgraph=True)(x)

        self.assertEqual(res, ref)

    def test_autograd_function_equivalence(self):
        # [FUNCTION DOC] Loops over grad on/off and the module fixtures Module1..Module4, resets
        # Dynamo for each, compiles with the eager backend and checks the output on a ones tensor
        # of shape (2, 3) is close to the tensor [2.0] (broadcast compare; forward doubles ones).
        # Fails via assertTrue if not close.
        for grad in [True, False]:
            for i in range(1, 5):
                torch._dynamo.reset()
                model = globals()[f"Module{i}"]()
                opt_model = torch.compile(model, backend="eager")
                self.assertTrue(
                    torch.allclose(
                        opt_model(torch.ones(2, 3, requires_grad=grad)),
                        torch.tensor([2.0], requires_grad=grad),
                    )
                )

    def test_autograd_function_has_graph_break(self):
        # [FUNCTION DOC] Compiles Module5 and Module6 (graph break inside Function forward) with a
        # CompileCounter backend, for requires_grad on and off, runs three iterations comparing
        # against eager with allclose, and asserts exactly 2 frames were compiled (the break
        # splits the function into two graphs).
        for grad in [True, False]:
            x = torch.randn(10, requires_grad=grad)
            for model in [Module5(), Module6()]:
                torch._dynamo.reset()
                cnts = torch._dynamo.testing.CompileCounter()
                opt_model = torch.compile(model, backend=cnts)
                for _ in range(3):
                    ref = model(x)
                    res = opt_model(x)
                    self.assertTrue(torch.allclose(ref, res))
                self.assertEqual(cnts.frame_count, 2)

    def test_linear_setup_context(self):
        # [FUNCTION DOC] Compiles ModuleLinear (Function with setup_context) with the
        # eager backend and fullgraph=True on double tensors and asserts the output equals
        # the eager result.
        model = ModuleLinear()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        input = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        weight = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
        eager_result = model(input, weight)
        optim_result = opt_model(input, weight)
        self.assertEqual(optim_result, eager_result)

    def test_materialize_grad(self):
        # [FUNCTION DOC] Compiles MaterializingGradModule with the eager backend (no fullgraph) on
        # a double tensor requiring grad and asserts the outputs match eager.
        model = MaterializingGradModule()
        opt_model = torch.compile(model, backend="eager")
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        optim_result = opt_model(x)
        eager_result = model(x)
        self.assertEqual(optim_result, eager_result)

    def test_print_in_bwd(self):
        # [FUNCTION DOC] Compiles CustomFuncBwdPrintModule with fullgraph=True and asserts that
        # calling it raises torch._dynamo.exc.Unsupported with a message about Dynamo not knowing
        # how to trace the builtin print operator.
        model = CustomFuncBwdPrintModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Dynamo does not know how to trace builtin operator `print`",
        ):
            opt_model(x)

    def test_stride_in_bwd(self):
        # [FUNCTION DOC] Compiles CustomFuncStrideModule with fullgraph=True and a CompileCounter,
        # runs forward and backward (with a cloned detached input as the gradient) in eager and
        # compiled mode, and asserts outputs and input gradients match and exactly one frame was
        # compiled. Clears Dynamo counters first.
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()
        model = CustomFuncStrideModule()
        opt_model = torch.compile(backend=cnt, fullgraph=True)(model)
        x1 = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x2 = copy.deepcopy(x1)
        ref = model(x1)
        ref.backward(x1.clone().detach())
        res = opt_model(x2)
        res.backward(x2.clone().detach())

        self.assertEqual(ref, res)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(cnt.frame_count, 1)

    def test_enum_arg(self):
        # [FUNCTION DOC] Checks that a Function can receive a Python Enum member as a non-tensor
        # argument under fullgraph eager compile: forward returns sin or cos depending on the enum
        # value; the test calls with the A member and compares to x.sin().
        from enum import Enum

        class SomeEnum(Enum):
            # [CLASS DOC] HELPER: Local Enum with members A (0) and B (1).
            A = 0
            B = 1

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Function taking a tensor and an enum; forward
            # returns sin of x when the enum is SomeEnum.A, otherwise cos; backward returns the
            # gradient unchanged (note: for a second input it returns only one value, but only the
            # tensor needs a gradient here).
            @staticmethod
            def forward(ctx, x, e):
                # [FUNCTION DOC] HELPER: Forward: x.sin() if e is SomeEnum.A else x.cos().
                if e is SomeEnum.A:
                    return x.sin()
                else:
                    return x.cos()

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Backward: returns g unchanged.
                return g

        @torch.compile(backend="eager", fullgraph=True)
        def f(x, enum):
            # [FUNCTION DOC] HELPER: Fullgraph eager-compiled function applying Foo to x and
            # the enum argument.
            output = Foo.apply(
                x,
                enum,
            )
            return output

        x = torch.tensor([[1.0, 2, 3], [4, 5, 6]], requires_grad=True)
        y = f(x, SomeEnum.A)
        self.assertEqual(y, x.sin())

    def test_save_for_bwd(self):
        # [FUNCTION DOC] Compiles SaveForBwdModule with fullgraph=True and the eager backend and
        # runs it once on a double tensor requiring grad. Only checks that tracing and execution
        # do not raise.
        model = SaveForBwdModule()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        opt_model(x)

    def test_allow_in_graph(self):
        # [FUNCTION DOC] Registers a Function class with allow_in_graph even though its forward
        # contains a graph break and then compiles a fullgraph function that applies it; asserts
        # the result equals the eager apply and that only one frame was compiled (the Function is
        # treated as opaque). Clears counters first.
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture registered with torch._dynamo.allow_in_graph.
            # Forward breaks the graph, stores the size of dim 0 on ctx and returns x * 2.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: calls graph_break, stores x.size(0) as ctx.x0
                # and returns x * 2.
                torch._dynamo.graph_break()
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns grad_out times ctx.x0.
                return grad_out * ctx.x0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying AllowInGraphFunc to x.
            return AllowInGraphFunc.apply(x)

        x = torch.rand(2, 3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, AllowInGraphFunc.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_once_differentiable(self):
        # [FUNCTION DOC] Tests that a Function whose backward is wrapped with once_differentiable
        # compiles under fullgraph with the counting backend; compares against eager apply and
        # asserts one frame.
        from torch.autograd.function import once_differentiable

        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class ScaleGradient(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Function whose forward is the identity and whose
            # backward (decorated with once_differentiable) halves the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.
                return x

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad times 0.5.
                return grad * 0.5

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying ScaleGradient to x.
            return ScaleGradient.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = fn(x)

        self.assertEqual(result, ScaleGradient.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_classmethod(self):
        # [FUNCTION DOC] Tests a Function whose forward and backward are classmethods (taking cls
        # as first argument) rather than static methods: it applies it inside an eager-backend
        # compiled function on a 4-D tensor. Only checks that it runs without error.
        class Shake(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Function defining forward and backward as
            # classmethods; forward returns foo + foo and backward returns grad_output.
            @classmethod
            def forward(cls, ctx, foo):
                # [FUNCTION DOC] HELPER: Forward (classmethod): returns foo + foo.
                return foo + foo

            @classmethod
            def backward(cls, ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward (classmethod): returns grad_output unchanged.
                return grad_output

        def f(x):
            # [FUNCTION DOC] HELPER: Applies Shake to x.
            return Shake.apply(x)

        x = torch.randn(4, 4, 4, 4, requires_grad=True)
        opt_m = torch.compile(backend="eager")(f)
        opt_m(x)

    def test_function_context_save_and_mark(self):
        # [FUNCTION DOC] Runs the ContextSaveAndMark fixture (inside ModuleWithGradFunc) in eager,
        # resets Dynamo, runs it compiled with the eager backend and asserts the outputs are
        # equal. Uses a no-grad tensor of shape (1).
        mod = ModuleWithGradFunc(ContextSaveAndMark)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_function_context_mark_and_save(self):
        # [FUNCTION DOC] Same as the save-then-mark test but with the ContextMarkAndSave
        # fixture (mark non-differentiable before save_for_backward): compares eager and eager-
        # backend compiled outputs.
        mod = ModuleWithGradFunc(ContextMarkAndSave)
        args, kwargs = ([torch.rand([1])], {})
        before = mod(*args, **kwargs)

        torch._dynamo.reset()
        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_mark_dirty_discarded_output_grad(self):
        # [FUNCTION DOC] Regression test for mark_dirty on a tensor whose Function output is
        # discarded: an in-place Function (multiply by 3, mark_dirty) is applied to an
        # intermediate and its return value ignored; the loss and weight gradient computed under
        # fullgraph eager compile must equal those from eager execution.
        class TimesThreeInplace(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward multiplies x in place by 3, marks it
            # dirty and returns it; backward multiplies the gradient by 3.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: mutates x in place (mul_ 3), calls
                # ctx.mark_dirty(x) and returns x.
                x.mul_(3)
                ctx.mark_dirty(x)
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad * 3.
                return grad * 3

        def fn(w):
            # [FUNCTION DOC] HELPER: Computes y = ones(2,2) @ w, applies TimesThreeInplace to
            # y while discarding the result, and returns y.sum() (relies on the in-place
            # mutation of y).
            y = torch.ones(2, 2) @ w
            TimesThreeInplace.apply(y)
            return y.sum()

        def grad_for(compiled):
            # [FUNCTION DOC] HELPER: Resets Dynamo, builds w as a 2x2 identity requiring grad,
            # runs fn either compiled (eager backend, fullgraph) or eagerly depending on the flag,
            # calls backward on the loss, and returns the detached loss and w.grad.
            torch._dynamo.reset()
            w = torch.eye(2, requires_grad=True)
            f = torch.compile(fn, backend="eager", fullgraph=True) if compiled else fn
            loss = f(w)
            loss.backward()
            return loss.detach(), w.grad

        eager_loss, eager_grad = grad_for(compiled=False)
        compiled_loss, compiled_grad = grad_for(compiled=True)

        self.assertEqual(eager_loss, compiled_loss)
        self.assertEqual(eager_grad, compiled_grad)

    def test_mark_dirty_on_view_preserves_error(self):
        # [FUNCTION DOC] Checks that mark_dirty on a view of a leaf requiring grad still raises
        # the RuntimeError that eager raises ("a view of a leaf Variable that requires grad"),
        # both in a plain eager-backend compile and under fullgraph (Dynamo reset between).
        class Inplace(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward adds 1 in place to x, marks it dirty and
            # returns it; backward returns the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: x.add_(1) in place, mark_dirty(x), returns x.
                x.add_(1)
                ctx.mark_dirty(x)
                return x

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad unchanged.
                return grad

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Inplace to a view of x (via view_as).
            return Inplace.apply(x.view_as(x))

        x = torch.randn(2, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a leaf Variable that requires grad"
        ):
            torch.compile(fn, backend="eager")(x)

        torch._dynamo.reset()
        x = torch.randn(2, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "a view of a leaf Variable that requires grad"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

    def test_multi_output(self):
        # [FUNCTION DOC] Tests a Function with two outputs (two clones of x) under fullgraph with
        # a counting backend; asserts the result matches eager apply and one frame was compiled.
        # Clears counters first.
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Function returning two clones of x; backward
            # sums the two output gradients.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns (x.clone(), x.clone()).
                return x.clone(), x.clone()

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Backward: returns grad1 + grad2.
                return grad1 + grad2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying Foo to x.
            return Foo.apply(x)

        x = torch.randn(3, requires_grad=True)
        result = f(x)

        self.assertEqual(result, Foo.apply(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_data_in_bwd(self):
        # [FUNCTION DOC] Checks that backward code reading and mutating tensor .data compiles
        # under aot_eager fullgraph: the backward zeroes gradient entries where the saved input
        # is negative; expects gradient [0, 3, 3] for input [-2, 1, 3] and that the forward
        # result matches eager.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves the input and returns input times
            # 3. Backward clones the gradient, zeroes entries (through .data) where the saved
            # input is negative and returns the result times 3.
            @staticmethod
            def forward(ctx, input_tensor):
                # [FUNCTION DOC] HELPER: Forward: saves input_tensor and returns input_tensor * 3.
                ctx.save_for_backward(input_tensor)
                return input_tensor * 3

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: clones grad_output, sets entries to 0
                # through .data where the saved input's data is negative, returns the modified
                # gradient times 3.
                (input_tensor,) = ctx.saved_tensors

                # Modify gradient using .data (Dangerous: Breaks autograd tracking!)
                modified_grad = grad_output.clone()
                modified_grad.data[input_tensor.data < 0] = (
                    0  # Zero-out gradients for negative inputs
                )

                return modified_grad * 3

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to x.
            return Foo.apply(x)

        x = torch.tensor([-2.0, 1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, torch.tensor([0.0, 3.0, 3.0]))

    def test_requires_grad_in_bwd(self):
        # [FUNCTION DOC] Checks that grad_output inside a compiled Function backward does not
        # require grad: the backward returns a deliberately wrong gradient if
        # grad_output.requires_grad is true. Compiled under aot_eager fullgraph, the test expects
        # x.grad to equal cos(x + 1).
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves x and returns sin(x + 1). Backward
            # returns grad times sin(x+1) (wrong on purpose) if grad_output requires grad, else
            # grad times cos(x+1).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: saves x, returns sin(x + 1).
                ctx.save_for_backward(x)
                return torch.sin(x + 1)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: branches on grad_output.requires_grad
                # as described above.
                (x,) = ctx.saved_tensors
                if grad_output.requires_grad:
                    return grad_output * torch.sin(
                        x + 1
                    )  # Wrong gradient, we should never get here.
                else:
                    return grad_output * torch.cos(x + 1)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to x.
            return Foo.apply(x)

        x = torch.tensor([1.0, 3.0], requires_grad=True)
        res = fn(x)
        self.assertEqual(res, Foo.apply(x))
        res.sum().backward()
        self.assertEqual(x.grad, torch.cos(x + 1))

    def test_amp_custom_fwd_bwd(self):
        # [FUNCTION DOC] Tests a Function whose forward and backward are decorated with
        # torch.amp.custom_fwd and custom_bwd (for the module-level device_type). Compiles an
        # application under fullgraph with a counting backend, runs backward, asserts the result
        # matches the eager apply and one frame was compiled. Clears counters first.
        torch._dynamo.utils.counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        class MyMM(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Matrix multiply Function with autocast-aware
            # forward/backward decorators.
            @staticmethod
            @torch.amp.custom_fwd(device_type=device_type)
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Forward (custom_fwd): saves a and b and returns a.mm(b).
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            @torch.amp.custom_bwd(device_type=device_type)
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward (custom_bwd): returns grad times b transposed
                # and a transposed times grad.
                a, b = ctx.saved_tensors
                return grad.mm(b.t()), a.t().mm(grad)

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying MyMM to a and b.
            return MyMM.apply(a, b)

        a = torch.randn([64, 64], dtype=torch.float32, requires_grad=True)
        b = torch.randn([64, 64], dtype=torch.float32, requires_grad=True)
        res = fn(a, b)
        grad = torch.randn_like(res)
        res.backward(grad)

        self.assertEqual(res, MyMM.apply(a, b))
        self.assertEqual(cnt.frame_count, 1)

    def test_set_materialize_grads_no_graph_break(self):
        # [FUNCTION DOC] Checks that calling ctx.set_materialize_grads(True) inside forward does
        # not cause a graph break: compiles under fullgraph with the eager backend, runs backward
        # on a scalar and checks the result equals the eager apply and the gradient is 3.0.
        class MulY(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward enables gradient materialization and
            # returns x * 3; backward returns grad_out * 3.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: ctx.set_materialize_grads(True), returns x * 3.
                ctx.set_materialize_grads(True)
                return x * 3

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns grad_out * 3.
                return grad_out * 3

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Fullgraph eager-compiled function applying MulY to x.
            return MulY.apply(x)

        x = torch.tensor(2.0, requires_grad=True)
        result = f(x)
        result.sum().backward()
        self.assertEqual(result, MulY.apply(x))
        self.assertEqual(x.grad, 3.0)

    def test_user_defined_object_as_input(self):
        # [FUNCTION DOC] Passes a user-defined dataclass object (holding tensors) as a non-tensor
        # input to a Function under fullgraph and the aot_eager counting backend. Verifies the
        # output, the gradients for x (2.5*3.5) and z (2.0), that the tensor inside the dataclass
        # gets no gradient, and compares the captured Dynamo graph text with an expected inline
        # string (autograd_function_apply with forward and backward subgraphs).
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @dataclass
        class Weird:
            # [CLASS DOC] HELPER: Local dataclass with an int field x and tensor fields b and c,
            # used as a non-tensor Function input.
            x: int
            b: torch.Tensor
            c: torch.Tensor

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves weird.b and weird.c and returns
            # their product times a clone of x; backward returns gradients for x and z only (None
            # for the dataclass).
            @staticmethod
            def forward(ctx, x: torch.Tensor, weird: Weird, z: torch.Tensor):
                # [FUNCTION DOC] HELPER: Forward: saves the dataclass tensors b, c and returns b *
                # c * x.clone().
                ctx.save_for_backward(weird.b, weird.c)
                return weird.b * weird.c * x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad*b*c, None, grad*2) for inputs
                # x, weird, z.
                b, c = ctx.saved_tensors
                return grad * b * c, None, grad * 2

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, weird, z):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying Foo to x, weird, z.
            return Foo.apply(x, weird, z)

        x = torch.tensor(2.0, requires_grad=True)
        weird = Weird(1.2, torch.tensor(2.5, requires_grad=True), torch.tensor(3.5))
        z = torch.tensor(3.0, requires_grad=True)

        result = f(x, weird, z)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z))
        self.assertEqual(x.grad, 2.5 * 3.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)

        # check Dynamo captured graph is correct!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_weird_b: "f32[]", L_weird_c: "f32[]", L_x_: "f32[]", L_z_: "f32[]"):
        l_weird_b = L_weird_b
        l_weird_c = L_weird_c
        l_x_ = L_x_
        l_z_ = L_z_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_weird_b, l_weird_c, l_x_, l_z_, non_differentiable_idx = [], saved_for_backward_idx = [0, 1]);  fwd_body_0 = bwd_body_0 = l_weird_b = l_weird_c = l_x_ = l_z_ = None
        getitem: "f32[]" = autograd_function_apply[0];  autograd_function_apply = None
        return (getitem,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, grad: "f32[]", l_weird_b: "f32[]", l_weird_c: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = grad * l_weird_b;  l_weird_b = None
            mul_1: "f32[]" = mul * l_weird_c;  mul = l_weird_c = None
            mul_2: "f32[]" = grad * 2;  grad = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, None, mul_1, mul_2)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_weird_b: "f32[]", l_weird_c: "f32[]", l_x_: "f32[]", l_z_: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[]" = l_weird_b * l_weird_c
            clone: "f32[]" = l_x_.clone();  l_x_ = None
            outs: "f32[]" = mul * clone;  mul = clone = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((outs,), (l_weird_b, l_weird_c))
""",
        )

    def test_tensor_list_as_input(self):
        # [FUNCTION DOC] Passes a list of tensors as a non-tensor-wrapped argument to a Function
        # under fullgraph and aot_eager: forward saves both list tensors and multiplies a clone of
        # x by their sum; checks the output, that x.grad is 7.0 and that the list tensors receive
        # no gradient (backward returns None for that slot).
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves tl[0] and tl[1] and returns
            # x.clone() times their sum; backward returns grad times that sum plus None for
            # the list input.
            @staticmethod
            def forward(ctx, x, tl):
                # [FUNCTION DOC] HELPER: Forward: saves the two list tensors, returns x.clone() *
                # (tl[0] + tl[1]).
                ctx.save_for_backward(tl[0], tl[1])
                return x.clone() * (tl[0] + tl[1])

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad * (tl0 + tl1), None).
                tl0, tl1 = ctx.saved_tensors
                return grad * (tl0 + tl1), None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, tl):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to x and
            # the tensor list.
            return Foo.apply(x, tl)

        x = torch.tensor(2.0, requires_grad=True)
        tl = [
            torch.tensor(3.0, requires_grad=True),
            torch.tensor(4.0, requires_grad=True),
        ]

        result = f(x, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, tl))
        self.assertEqual(x.grad, 7.0)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_multiple_different_non_tensor_inputs(self):
        # [FUNCTION DOC] Like the tensor-list test but with several different non-tensor inputs at
        # once (dataclass with tensors, a tensor z and a tensor list). Compiles under fullgraph
        # aot_eager and checks the result, x.grad (2.5*3.5*0.5), z.grad (2.0), and that dataclass
        # and list tensors get no gradient.
        @dataclass
        class Weird:
            # [CLASS DOC] HELPER: Local dataclass with an int field x and tensor fields b and c.
            x: int
            b: torch.Tensor
            c: torch.Tensor

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves weird.b, weird.c, tl[0], tl[1] and
            # returns x.clone() * b * c * tl[0]; backward returns gradients only for x and z.
            @staticmethod
            def forward(ctx, x, weird, z, tl):
                # [FUNCTION DOC] HELPER: Forward: saves four tensors and returns x.clone() *
                # weird.b * weird.c * tl[0].
                ctx.save_for_backward(weird.b, weird.c, tl[0], tl[1])
                return x.clone() * weird.b * weird.c * tl[0]

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad*b*c*tl0, None, grad*2, None).
                b, c, tl0, _ = ctx.saved_tensors
                return grad * b * c * tl0, None, grad * 2, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, weird, z, tl):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to x,
            # weird, z and tl.
            return Foo.apply(x, weird, z, tl)

        x = torch.tensor(2.0, requires_grad=True)
        weird = Weird(
            1.2,
            torch.tensor(2.5, requires_grad=True),
            torch.tensor(3.5, requires_grad=True),
        )
        z = torch.tensor(3.0, requires_grad=True)
        tl = [
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(0.6, requires_grad=True),
        ]

        result = f(x, weird, z, tl)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, weird, z, tl))
        self.assertEqual(x.grad, 2.5 * 3.5 * 0.5)
        self.assertEqual(z.grad, 2.0)
        self.assertEqual(weird.b.grad, None)
        self.assertEqual(weird.c.grad, None)
        self.assertEqual(tl[0].grad, None)
        self.assertEqual(tl[1].grad, None)

    def test_backward_returns_none_for_tensor_input(self):
        # [FUNCTION DOC] Checks that when backward returns None for a tensor input that requires
        # grad, the compiled (aot_eager, fullgraph) result leaves that input's grad as None while
        # x gets the expected gradient (3.0).
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves y and returns x.clone() * y;
            # backward returns (grad * y, None), deliberately giving no gradient for y.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: saves y and returns x.clone() * y.
                ctx.save_for_backward(y)
                return x.clone() * y

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad * y, None).
                (y,) = ctx.saved_tensors
                return grad * y, None

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to
            # x and y.
            return Foo.apply(x, y)

        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)

        result = f(x, y)
        result.sum().backward()

        self.assertEqual(result, Foo.apply(x, y))
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, None)

    def test_function_with_bound_free_variable(self):
        # [FUNCTION DOC] Tests a Function applied inside an nn.Module to a parameter and a Python
        # int constant bound as a free value (1): forward clamps to the lower bound and backward
        # masks the gradient. Compares eager-backend compiled output with eager on a random input
        # of shape (4, 128, 32, 32).
        class LowerBound(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture implementing a lower-bound clamp. Forward saves
            # inputs and a ones tensor scaled by bound and clamps inputs from below; backward
            # passes the gradient where inputs >= bound and returns None for the bound.
            @staticmethod
            def forward(ctx, inputs, bound):
                # [FUNCTION DOC] HELPER: Forward: saves inputs and inputs.new_ones(1) * bound;
                # returns inputs clamped with minimum bound.
                ctx.save_for_backward(inputs, inputs.new_ones(1) * bound)
                return inputs.clamp(min=bound)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns ((inputs >= bound)
                # * grad_output, None).
                inputs, bound = ctx.saved_tensors
                return (inputs >= bound) * grad_output, None

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Nested module holding a random parameter gamma and applying
            # LowerBound with bound 1 to it, adding the result to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter gamma with random values of shape
                # (4, 128, 32, 32).
                super().__init__()
                self.gamma = torch.nn.Parameter(torch.rand([4, 128, 32, 32]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x + LowerBound applied to the parameter gamma
                # with bound 1.
                gamma = LowerBound.apply(self.gamma, 1)
                return x + gamma

        mod = MyMod()
        args, kwargs = ([torch.rand([4, 128, 32, 32])], {})
        before = mod(*args, **kwargs)

        compiled_model = torch.compile(mod, backend="eager")
        after = compiled_model(*args, **kwargs)
        self.assertEqual(before, after)

    def test_forward_returns_constant(self):
        # [FUNCTION DOC] Checks that a Function whose forward returns a tensor plus a constant
        # Python list ([1, 2, 3]) compiles under fullgraph with aot_eager and backward runs
        # through the tensor output; the result must equal the eager apply output.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns the tensor together with a
            # constant list of ints; backward returns the first output gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns (x, [1, 2, 3]).
                return x, [1, 2, 3]  # Tensor and list of integers

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output1 and ignores the gradient
                # for the list output.
                return grad_output1

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying Foo to x.
            return Foo.apply(x)

        x = torch.tensor(2.0, requires_grad=True)
        result = f(x)
        result[0].sum().backward()

        self.assertEqual(result, Foo.apply(x))

    # I pulled all of these test cases from test_autograd.py
    # In the future, we should make the Dynamo test suite actually
    # run on test_autograd.py (it's disabled right now) and delete these.
    def test_smoke_from_test_autograd(self):
        # [FUNCTION DOC] Smoke test collecting many cases ported from test_autograd.py:
        # defines a number of Function classes (product reduction, square, identity, NaN-
        # producing backward, view, in-place add, in-place mul with jvp, vjp/jvp, view-and-
        # inplace), then a nested test function exercising them once in eager and once under
        # the eager-backend torch.compile. No numerical assertions; it only checks that
        # capture and execution do not raise.
        def mult1(x):
            # [FUNCTION DOC] HELPER: Returns the product of x over the last dimension twice,
            # reducing the last two dims.
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose forward computes the double product reduction via
            # mult1 and saves x and y; backward divides the scaled gradient by x.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: y = mult1(x), saves x and y, returns y.
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * y) expanded with two
                # trailing singleton dims, divided by x.
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        class Double(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function squaring its input; saves x and the square.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: y = x squared, saves x and y, returns y.
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2 * x.
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            # [CLASS DOC] HELPER: Subclass of Double with an equivalent backward expressed with
            # the saved output y: grad_output * 2 * y / x.
            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2 * y / x using
                # the saved tensors.
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        class Identity(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning its first input and the sum of both inputs.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Forward: returns (a, a + b).
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_a + grad_b, grad_b).
                return grad_a + grad_b, grad_b

        class MyFunc2(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose forward clones its input and whose backward
            # returns a 10x10 expanded NaN tensor (to check NaN gradients are not asserted on).
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Forward: returns inp.clone().
                return inp.clone()

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Backward: returns a scalar NaN tensor expanded to
                # shape (10, 10).
                return torch.tensor(float("nan")).expand(10, 10)

        def run_fn(a):
            # [FUNCTION DOC] HELPER: Applies MyFunc2 to a and returns the sum of the output.
            out = MyFunc2.apply(a)
            return out.sum()

        class MyFn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function whose forward returns a view of its input and backward
            # returns the gradient unchanged.
            @staticmethod
            def forward(ctx, inp):
                # [FUNCTION DOC] HELPER: Forward: returns inp.view_as(inp).
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad.
                return grad

        class MyAdder(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function that adds b to a in place, marks a dirty
            # and returns it.
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Forward: a.add_(b), ctx.mark_dirty(a), returns a.
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad, grad).
                return grad, grad

        class InplaceMul(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function multiplying its input in place by 2 and marking it
            # dirty; defines a jvp static method.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: result = x.mul_(2);
                # mark_dirty(result); returns result.
                result = x.mul_(2)
                ctx.mark_dirty(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: does nothing and returns None (never invoked
                # by the test).
                pass

            @staticmethod
            def jvp(ctx, x_t):
                # [FUNCTION DOC] HELPER: Forward-mode rule: refers to the name jvp_err, which is
                # not defined anywhere in this file. Never invoked here.
                # KNOWN ISSUE: jvp_err is an undefined name (suppressed with noqa F821), so
                #   calling this jvp would raise NameError; the test never reaches it.
                if jvp_err:  # noqa: F821
                    return x_t
                else:
                    return x_t.mul_(2)

        class MyFn2(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning (x + y, x), defining vjp and jvp static
            # methods but not backward.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: returns (x + y, x).
                return x + y, x

            @staticmethod
            def vjp(ctx, gO1, gO2):
                # [FUNCTION DOC] HELPER: Vjp rule: returns (gO1 + gO2, gO1).
                return gO1 + gO2, gO1

            @staticmethod
            def jvp(ctx, x_t, y_t):
                # [FUNCTION DOC] HELPER: Jvp rule: returns (x_t + y_t, fn(x_t)) where fn is not
                # defined in this scope.
                # KNOWN ISSUE: fn is an undefined name here (noqa F821); this jvp would raise
                #   NameError if called, which the test never does.
                return x_t + y_t, fn(x_t)  # noqa: F821

        class MyFn3(torch.autograd.Function):
            # [CLASS DOC] HELPER: Function returning a slice view of a clone of its input,
            # optionally mutated in place by adding 2.
            @staticmethod
            def forward(ctx, inp, inplace):
                # [FUNCTION DOC] HELPER: Forward: view = first three elements of inp.clone(); if
                # inplace is true adds 2 to the view in place; returns the view.
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns (grad, None).
                return grad, None

        def test():
            # [FUNCTION DOC] HELPER: Runs each fixture Function once on small random or constant
            # tensors: Mult, Double, Double2, Identity, MyFn, MyAdder (and its backward),
            # InplaceMul, MyFn2 and MyFn3. Results are not checked.
            x = torch.ones(2, 4, 4).requires_grad_()
            mult2(x)

            x = torch.tensor(2).double().requires_grad_()
            double(x)
            double2(x)

            x = torch.randn(5, 5, requires_grad=True)
            y = torch.randn(5, 5, requires_grad=True)
            Identity.apply(x, y)

            a = torch.rand(1, 2)
            b = torch.rand(1, requires_grad=True)
            MyFn.apply(a)

            a = torch.ones(2, requires_grad=True)
            b = torch.ones(2, requires_grad=True)
            c = MyAdder.apply(a.clone(), b)
            c.sum().backward()

            z = torch.tensor(1.0, requires_grad=True)
            x = z.clone()
            y = InplaceMul.apply(x)

            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            c = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)
            MyFn2.apply(a, b)
            MyFn2.apply(c, d)

            base = torch.rand(10, requires_grad=True)
            MyFn3.apply(base, False)

        test()
        opt_test = torch.compile(test, backend="eager")
        opt_test()

    def test_tensor_subclass_intermediary_input(self):
        # [FUNCTION DOC] Tests an autograd Function applied to a wrapper tensor subclass defined
        # in the test (FooTensor): forward reads the subclass's inner data and builds another
        # subclass instance inside the Function. Compares output and input gradient between
        # eager and eager-backend compiled execution. Registers FooTensor with allow_in_graph as
        # a side effect.
        class FooTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass carrying inner data tensor, a config
            # tuple (size, strides, storage offset, dtype, layout, requires_grad) and a scale
            # tensor. Supports flatten/unflatten and a minimal dispatch that handles only
            # clone and view.
            @staticmethod
            def __new__(cls, data, config, scale):
                # [FUNCTION DOC] HELPER: Static constructor: creates a wrapper subclass via
                # _make_wrapper_subclass from the config fields on the data's device, then stores
                # data, config and scale as attributes.
                self = torch.Tensor._make_wrapper_subclass(
                    cls,
                    config[0],
                    strides=config[1],
                    storage_offset=config[2],
                    dtype=config[3],
                    layout=config[4],
                    requires_grad=config[5],
                    device=data.device,
                )
                self._data = data
                self._config = config
                self._scale = scale
                return self

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the fixed string "FooTensor".
                return "FooTensor"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns the inner attribute names (_data, _scale) and a
                # context tuple containing the config.
                return ("_data", "_scale"), (self._config,)

            @staticmethod
            def __tensor_unflatten__(tensors, metadatas, outer_size, outer_stride):
                # [FUNCTION DOC] HELPER: Static: rebuilds a FooTensor from the inner tensors
                # mapping and the first metadata item (outer_size and outer_stride are ignored).
                return FooTensor(tensors["_data"], metadatas[0], tensors["_scale"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch handler: for aten clone returns a new FooTensor
                # with cloned data and the same config and scale; for aten view returns a new
                # FooTensor with the viewed data; any other op raises NotImplementedError. Only
                # meant to satisfy Dynamo fake tensor creation.
                # handling clone and view is so dynamo fakefication passes, it's not
                # intended to be handling user code
                if func == torch.ops.aten.clone.default:
                    return FooTensor(
                        args[0]._data.clone(), args[0]._config, args[0]._scale
                    )
                elif func == torch.ops.aten.view.default:
                    new_data = args[0]._data.view(*args[1:])
                    return FooTensor(new_data, args[0]._config, args[0]._scale)

                raise NotImplementedError

        class foo_autograd_fn(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture (lowercase class name). Function that reads
            # data of its subclass input, creates a FooTensor inside forward and returns the
            # inner data tensor.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: x2 = x._data + 1.0; builds a FooTensor from it
                # and returns its _data.
                # access some data from `x`, where `x` is a tensor subclass
                x2 = x._data + 1.0
                # create and return a tensor subclass from within a torch.autograd.Function
                x3 = FooTensor(x2, x._config, x._scale)
                return x3._data

            @staticmethod
            def backward(ctx, g):
                # [FUNCTION DOC] HELPER: Backward: returns g unchanged.
                return g

        x_ref = torch.randn(4, 4).requires_grad_(True)
        x = copy.deepcopy(x_ref)
        scale = torch.tensor(1.0)
        # Weird that this is needed, but not having this breaks a lot of things
        torch._dynamo.allow_in_graph(FooTensor)

        def foo(x, scale):
            # [FUNCTION DOC] HELPER: Builds a config tuple from x (size, stride, storage offset,
            # dtype, layout, requires_grad), wraps x and scale in a FooTensor and returns the
            # output of foo_autograd_fn applied to it.
            config = (
                x.size(),
                x.stride(),
                x.storage_offset(),
                x.dtype,
                x.layout,
                x.requires_grad,
            )
            x = FooTensor(x, config, scale)
            x = foo_autograd_fn.apply(x)
            return x

        y_ref = foo(x_ref, scale)
        y_ref.sum().backward()

        foo_opt = torch.compile(foo, backend="eager")
        y = foo_opt(x, scale)
        y.sum().backward()

        self.assertEqual(y, y_ref)
        self.assertEqual(x.grad, x_ref.grad)

    def test_assert_is_contiguous_after_matmul(self):
        # [FUNCTION DOC] Checks that Dynamo keeps the assumption that the gradient w.r.t. a matmul
        # output is contiguous: backward asserts grad_x is contiguous, and the test compares eager
        # with a compiled run (counting backend, no fullgraph) of outputs and gradients and checks
        # exactly one frame was compiled.
        class LinearFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Linear Function (x times weight transposed)
            # whose backward asserts that grad_x is contiguous.
            @staticmethod
            def forward(ctx, x, weight):
                # [FUNCTION DOC] HELPER: Forward: saves x and weight,
                # returns x.matmul(weight.t()).
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: computes grad_x = grad_output.matmul(weight),
                # asserts grad_x is contiguous (raises AssertionError otherwise) and computes
                # grad_weight = grad_output transposed matmul x.
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                assert grad_x.is_contiguous()  # noqa: S101
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            # [FUNCTION DOC] HELPER: Applies LinearFunction to x and weight.
            return LinearFunction.apply(x, weight)

        x1 = torch.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = torch.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.sum().backward()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        y2 = opt_fn(x2, W2)
        y2.sum().backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)
        self.assertEqual(cnts.frame_count, 1)

    def test_assert_is_contiguous_on_grad_output_directly(self):
        # [FUNCTION DOC] Checks that Dynamo inserts a contiguous call on grad_output inside
        # backward before the user code asserts contiguity of grad_output. Compiles with the
        # aot_eager counting backend, backpropagates a cloned gradient that requires grad,
        # compares outputs and gradients with eager and compares the captured graph with an
        # expected inline string that contains y.contiguous.
        class LinearFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Linear Function whose backward asserts that
            # grad_output itself is contiguous.
            @staticmethod
            def forward(ctx, x, weight):
                # [FUNCTION DOC] HELPER: Forward: saves x and weight,
                # returns x.matmul(weight.t()).
                ctx.save_for_backward(x, weight)
                y = x.matmul(weight.t())
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: asserts grad_output is contiguous (raises
                # AssertionError otherwise), then returns (grad_output.matmul(weight), grad_output
                # transposed matmul x).
                assert grad_output.is_contiguous()  # noqa: S101
                x, weight = ctx.saved_tensors
                grad_x = grad_output.matmul(weight)
                grad_weight = grad_output.transpose(0, 1).matmul(x)

                return grad_x, grad_weight

        def fn(x, weight):
            # [FUNCTION DOC] HELPER: Applies LinearFunction to x and weight.
            return LinearFunction.apply(x, weight)

        x1 = torch.randn(5, 3, requires_grad=True)
        x2 = copy.deepcopy(x1)
        W1 = torch.randn(4, 3, requires_grad=True)
        W2 = copy.deepcopy(W1)

        y1 = fn(x1, W1)
        y1.backward(y1.clone().detach().requires_grad_(True))

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt)
        y2 = opt_fn(x2, W2)
        y2.backward(y2.clone().detach().requires_grad_(True))

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad)
        self.assertEqual(W1.grad, W2.grad)

        # Check the inserted .contiguous() call is there!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_weight_: "f32[4, 3]", L_x_: "f32[5, 3]"):
        l_weight_ = L_weight_
        l_x_ = L_x_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_weight_, l_x_, non_differentiable_idx = [], saved_for_backward_idx = [0, 1]);  fwd_body_0 = bwd_body_0 = l_weight_ = l_x_ = None
        getitem: "f32[5, 4]" = autograd_function_apply[0];  autograd_function_apply = None
        return (getitem,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, y: "f32[5, 4]", l_weight_: "f32[4, 3]", l_x_: "f32[5, 3]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            contiguous: "f32[5, 4]" = y.contiguous();  y = None

            grad_x: "f32[5, 3]" = contiguous.matmul(l_weight_);  l_weight_ = None

            transpose: "f32[4, 5]" = contiguous.transpose(0, 1);  contiguous = None
            grad_weight: "f32[4, 3]" = transpose.matmul(l_x_);  transpose = l_x_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (grad_weight, grad_x)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_weight_: "f32[4, 3]", l_x_: "f32[5, 3]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            t: "f32[3, 4]" = l_weight_.t()
            y: "f32[5, 4]" = l_x_.matmul(t);  t = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((y,), (l_weight_, l_x_))
""",
        )

    def test_smuggle_symint_issue_111031(self):
        # [FUNCTION DOC] Regression test for issue 111031: a Function that stores a symbolic
        # size (x.size(0)) on ctx must compile with dynamic=True under fullgraph with
        # exactly one frame.
        from torch.autograd import Function

        class Foo(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward stores x.size(0) in ctx.x0 and returns x
            # * 2; backward multiplies the gradient by ctx.x0.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: ctx.x0 = x.size(0); returns x * 2.
                ctx.x0 = x.size(0)
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns grad_out * ctx.x0.
                return grad_out * ctx.x0

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            # [FUNCTION DOC] HELPER: Fullgraph dynamic-shape compiled function applying Foo to x.
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True))
        self.assertEqual(cnts.frame_count, 1)

    def test_needs_input_grad(self):
        # [FUNCTION DOC] Checks that ctx.needs_input_grad is readable inside a compiled backward:
        # the backward is decorated with torch.compile (counting backend, fullgraph). Runs
        # backward on a vector requiring grad and asserts the gradient has the input shape, one
        # frame and two ops were counted.
        cnt = torch._dynamo.testing.CompileCounter()

        class NeedsInputGradFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves foo + foo; backward (itself
            # compiled with torch.compile) returns grad_output times sin of the saved result if
            # the first input needs grad, else None.
            @staticmethod
            def forward(ctx, foo):
                # [FUNCTION DOC] HELPER: Forward: result = foo + foo, saved for
                # backward and returned.
                result = foo + foo
                ctx.save_for_backward(result)
                return result

            @staticmethod
            @torch.compile(backend=cnt, fullgraph=True)
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward (compiled): returns grad_output * result.sin()
                # when ctx.needs_input_grad[0] is true, else None.
                (result,) = ctx.saved_tensors
                if ctx.needs_input_grad[0]:
                    return grad_output * result.sin()
                return None

        x = torch.randn(10, requires_grad=True)
        NeedsInputGradFunc.apply(x).sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_needs_input_grad_setter_roundtrip(self):
        # [FUNCTION DOC] Regression test: when backward is compiled with Dynamo, stores to
        # ctx.needs_input_grad must be visible on later reads. Records the value before and after
        # assigning a sentinel, None and restoring the original into a dict, then checks the
        # original is (True, False) and every later read returns what was stored; the input
        # gradient is all ones.
        # When backward is traced by Dynamo, a store to ctx.needs_input_grad
        # must be honored on subsequent reads. Regression test for the read
        # short-circuiting to the construction-time value and dropping the store.
        sentinel = ([False, True],)
        seen = {}

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns x + y; backward (compiled, eager
            # backend, fullgraph) reads and reassigns ctx.needs_input_grad and records each
            # observation in the enclosing seen dict.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: returns x + y.
                return x + y

            @staticmethod
            @torch.compile(backend="eager", fullgraph=True)
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward (compiled): writes the observed
                # needs_input_grad value into the seen mapping after each of four steps
                # (original, set to sentinel, set to None, restored), mutating
                # ctx.needs_input_grad; returns (grad_output, None).
                seen["original"] = ctx.needs_input_grad
                ctx.needs_input_grad = sentinel
                seen["after_set"] = ctx.needs_input_grad
                ctx.needs_input_grad = None
                seen["after_none"] = ctx.needs_input_grad
                ctx.needs_input_grad = seen["original"]
                seen["after_restore"] = ctx.needs_input_grad
                return grad_output, None

        x = torch.randn(3, requires_grad=True)
        Foo.apply(x, torch.randn(3)).sum().backward()
        self.assertEqual(seen["original"], (True, False))
        self.assertIs(seen["after_set"], sentinel)
        self.assertIsNone(seen["after_none"])
        self.assertIs(seen["after_restore"], seen["original"])
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_repeated_save_for_backward_calls(self):
        # [FUNCTION DOC] Checks that calling save_for_backward twice in forward (the second call
        # overrides the first) compiles: compares eager and compiled (counting backend) outputs
        # and both input gradients, and asserts one frame.
        from torch.autograd import Function

        class Foo(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward calls save_for_backward twice (x then x,
            # y) and returns x * y; backward uses the final saved tensors.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: save_for_backward(x), save_for_backward(x, y),
                # returns x * y.
                ctx.save_for_backward(x)
                ctx.save_for_backward(x, y)
                return x * y

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_out * x, grad_out * y) from
                # the saved tensors.
                x, y = ctx.saved_tensors
                return grad_out * x, grad_out * y

        cnts = torch._dynamo.testing.CompileCounter()

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Applies Foo to x and y.
            return Foo.apply(x, y)

        x_ref = torch.randn(2, requires_grad=True)
        y_ref = torch.randn(2, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_()
        y_test = y_ref.detach().clone().requires_grad_()

        out_ref = foo(x_ref, y_ref)
        out_ref.sum().backward()

        out_test = torch.compile(foo, backend=cnts)(x_test, y_test)
        out_test.sum().backward()

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(y_ref.grad, y_test.grad)

    def test_smuggle_tensor_and_complex_structures(self):
        # [FUNCTION DOC] Checks that a tensor and a list stored on ctx as plain attributes in
        # forward can be used in backward when compiled with dynamic shapes and fullgraph: asserts
        # one frame and runs backward.
        from torch.autograd import Function

        class Foo(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward stores the input tensor and a list [1,
            # 2, 3] on ctx as attributes and returns x * 2; backward multiplies the gradient by
            # the tensor and then repeatedly scales by each list item.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: sets ctx.x0 = x and ctx.x1 = [1, 2, 3]; returns
                # x * 2.
                ctx.x0 = x
                ctx.x1 = [1, 2, 3]
                return x * 2

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: starts with grad_out * ctx.x0 and for each item
                # i in ctx.x1 computes x0mul * i + x0mul; returns the result.
                x0mul = grad_out * ctx.x0
                for i in ctx.x1:
                    x0mul = (x0mul * i) + x0mul
                return x0mul

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True, dynamic=True)
        def foo(x):
            # [FUNCTION DOC] HELPER: Fullgraph dynamic-shape compiled function applying Foo to x.
            return Foo.apply(x)

        foo(torch.randn(2, requires_grad=True)).sum().backward()
        self.assertEqual(cnts.frame_count, 1)

    def test_int_output(self):
        # [FUNCTION DOC] Checks that a Function with an integer-dtype second output works under
        # aot_eager fullgraph: the float output requires grad while the int output does not, in
        # both eager and compiled results, which must be equal.
        from torch.autograd import Function

        class MyFunction(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns sin of x and that result cast to
            # int; backward returns cos of grad1 and a zero tensor.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: out1 = x.sin(), out2 = out1 as int
                # dtype; returns both.
                out1 = x.sin()
                out2 = out1.to(dtype=torch.int)
                return out1, out2

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Backward: returns (grad1.cos(), grad1 * 0.0).
                return grad1.cos(), grad1 * 0.0

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying MyFunction to
            # x and y.
            return MyFunction.apply(x, y)

        x = torch.tensor(10.0, requires_grad=True)
        y = torch.tensor(20.0, requires_grad=True)
        ref1, ref2 = MyFunction.apply(x, y)
        res1, res2 = fn(x, y)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)

    def test_mark_non_differentiable(self):
        # [FUNCTION DOC] Checks mark_non_differentiable on one of two outputs: compiled
        # (aot_eager, counting backend, fullgraph) outputs match eager, the first requires grad
        # and the second does not, backward runs on the first, and the captured graph text is
        # compared with an expected inline string (non_differentiable_idx = [1]).
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        from torch.autograd import Function

        class MyFunction(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns sin of x and y * 2 with
            # the second output marked non-differentiable; backward returns (cos of grad1,
            # grad2 * 0.0).
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: out1 = x.sin(), out2 = y * 2, marks out2 non-
                # differentiable, returns both.
                out1 = x.sin()
                out2 = y * 2
                ctx.mark_non_differentiable(out2)
                return out1, out2

            @staticmethod
            def backward(ctx, grad1, grad2):
                # [FUNCTION DOC] HELPER: Backward: returns (grad1.cos(), grad2 * 0.0).
                return grad1.cos(), grad2 * 0.0

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Fullgraph compiled function applying MyFunction to x and y.
            return MyFunction.apply(x, y)

        x = torch.tensor(10.0, requires_grad=True)
        y = torch.tensor(20.0, requires_grad=True)
        ref1, ref2 = MyFunction.apply(x, y)
        res1, res2 = fn(x, y)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        res1.sum().backward()

        # check Dynamo captured graph is correct!
        actual_graph = torch._dynamo.testing.normalize_gm(
            cnt.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            actual_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]", L_y_: "f32[]"):
        l_x_ = L_x_
        l_y_ = L_y_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_y_, non_differentiable_idx = [1], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = l_y_ = None
        getitem: "f32[]" = autograd_function_apply[0]
        getitem_1: "f32[]" = autograd_function_apply[1];  autograd_function_apply = None
        return (getitem, getitem_1)

    class bwd_body_0(torch.nn.Module):
        def forward(self, grad1: "f32[]", grad2: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            cos: "f32[]" = grad1.cos();  grad1 = None
            mul: "f32[]" = grad2 * 0.0;  grad2 = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (cos, mul)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[]", l_y_: "f32[]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            out1: "f32[]" = l_x_.sin();  l_x_ = None

            out2: "f32[]" = l_y_ * 2;  l_y_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((out1, out2), ())
""",
        )

    def test_mark_multi_output_non_differentiable(self):
        # [FUNCTION DOC] Checks mark_non_differentiable called with two outputs at once: under
        # aot_eager fullgraph the first output requires grad, the second and third do not; results
        # equal eager and backward runs on the first output.
        from torch.autograd import Function

        class MyFunction(Function):
            # [CLASS DOC] HELPER: Nested fixture. Three-output Function (sin of x, y * 2, z + 3)
            # that marks the last two outputs non-differentiable.
            @staticmethod
            def forward(ctx, x, y, z):
                # [FUNCTION DOC] HELPER: Forward: computes three outputs, marks out2 and out3 non-
                # differentiable, returns all.
                out1 = x.sin()
                out2 = y * 2
                out3 = z + 3
                ctx.mark_non_differentiable(out2, out3)
                return out1, out2, out3

            @staticmethod
            def backward(ctx, grad1, grad2, grad3):
                # [FUNCTION DOC] HELPER: Backward: returns (grad1.cos(), grad2, grad3).
                return grad1.cos(), grad2, grad3

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Fullgraph aot_eager-compiled function applying MyFunction to
            # x, y and z.
            return MyFunction.apply(x, y, z)

        x = torch.tensor(10.0, requires_grad=True)
        y = torch.tensor(20.0, requires_grad=True)
        z = torch.tensor(30.0, requires_grad=True)
        ref1, ref2, ref3 = MyFunction.apply(x, y, z)
        res1, res2, res3 = fn(x, y, z)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)
        # Ensure out1 requires gradients, out2 does not.
        self.assertTrue(ref1.requires_grad)
        self.assertTrue(res1.requires_grad)
        self.assertFalse(ref2.requires_grad)
        self.assertFalse(res2.requires_grad)
        self.assertFalse(ref3.requires_grad)
        self.assertFalse(res3.requires_grad)
        res1.sum().backward()

    def test_default_values(self):
        # [FUNCTION DOC] Checks that guard creation for a forward parameter with a default value
        # (alpha=0.99 that apply is not given) does not crash: runs the compiled function twice,
        # once without and once with requires_grad on the input.
        from torch.autograd import Function

        class Foo(Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward takes a default argument alpha and
            # returns x; backward returns the gradient.
            @staticmethod
            def forward(ctx, x, alpha=0.99):
                # [FUNCTION DOC] HELPER: Forward: returns x (alpha unused).
                return x

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns grad_out.
                return grad_out

        @torch.compile(backend="eager")
        def foo(x):
            # [FUNCTION DOC] HELPER: Eager-backend compiled function applying Foo to x using the
            # default for alpha.
            return Foo.apply(x)

        # Make sure guards for default values do not crash
        foo(torch.randn(2))
        foo(torch.randn(2, requires_grad=True))

    def test_fwd_no_grad(self):
        # [FUNCTION DOC] Checks that forward is traced under no_grad mode: the forward calls exp
        # with an out argument on its input, which would raise under grad mode. Compiled under
        # fullgraph eager the result is compared with the eager apply result on a clone.
        # autograd.Function.forward should be traced and called under no_grad mode.
        # torch.exp with out=... arguments don't support automatic differentiation,
        # so can't be traced/called under grad mode (throwing RuntimeError),
        # therefore this unit test ensures fwd is under no_grad mode.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward writes exp of its input into the input
            # in place and returns it; backward returns None.
            @staticmethod
            def forward(ctx, inputs):
                # [FUNCTION DOC] HELPER: Forward: torch.exp(inputs, out=inputs), returns inputs.
                torch.exp(inputs, out=inputs)
                return inputs

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns None.
                return None

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Fullgraph eager-compiled function applying Foo to x.
            return Foo.apply(x)

        x1 = torch.randn(2, 3, requires_grad=True)
        x2 = x1.clone()
        self.assertEqual(f(x1), Foo.apply(x2))

    # https://github.com/pytorch/pytorch/issues/129963
    def test_fwd_propogation_correctness(self):
        # [FUNCTION DOC] Regression test for issue 129963: applies a Function that returns (x
        # cubed, derivative) twice inside a fullgraph aot_eager function. The backward is
        # intentionally wrong (scaled by 0.5) so that gradient 2.25 per element proves the
        # Function backward ran for both applications (the second apply must not be inlined with
        # autograd's own backward).
        class MyCube(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns x cubed and 3 * x squared and
            # saves x and the second; backward combines them as grad_output * dx + grad_dx * 6 * x
            # and returns half of that (intentionally wrong).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: result = x**3, dx = 3 * x**2, saves x and dx,
                # returns (result, dx).
                result = x**3
                dx = 3 * x**2
                ctx.save_for_backward(x, dx)
                return result, dx

            @staticmethod
            def backward(ctx, grad_output, grad_dx):
                # [FUNCTION DOC] HELPER: Backward: computes grad_output * dx + grad_dx * 6 * x and
                # returns it multiplied by 0.5 (deliberately incorrect).
                x, dx = ctx.saved_tensors
                result = grad_output * dx + grad_dx * 6 * x
                # Intentionally return a wrong value to test if the backward is triggered twice.
                # Since if the first MyCube.apply returns values w/o requires_grad=True,
                # this backward would be only triggered once (the first MyCube.apply call),
                # as the second MyCube.apply is inlined by Dynamo and the corresponding backward
                # would be generated by autograd engine.
                return result * 0.5

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MyCube twice, feeding the first output into the
            # second, and returns the final first output.
            x, _ = MyCube.apply(x)
            x, _ = MyCube.apply(x)
            return x

        inp = torch.ones(2, requires_grad=True)
        out = fn(inp)
        out.sum().backward()
        self.assertEqual(out, inp**3)
        self.assertEqual(inp.grad, torch.tensor([2.25, 2.25]))

    def test_tuple_arg(self):
        # [FUNCTION DOC] Passes a Python tuple (10, 10) used as a shape argument, closed over from
        # the enclosing scope, to a Function under fullgraph with a counting backend. Checks
        # output equals x + 1, gradient shape equals the tuple, one frame and two ops.
        cnt = torch._dynamo.testing.CompileCounter()

        class TupleArgFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves a random tensor of the given shape
            # and returns x + 1; backward returns the saved tensor and None.
            @staticmethod
            def forward(ctx, x, shape):
                # [FUNCTION DOC] HELPER: Forward: ctx.save_for_backward(torch.randn(shape));
                # returns x + 1.
                ctx.save_for_backward(torch.randn(shape))
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (saved tensor, None).
                (result,) = ctx.saved_tensors
                return result, None

        @torch.compile(backend=cnt, fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Fullgraph compiled function with no arguments that applies
            # TupleArgFunc to the enclosing x and shape.
            return TupleArgFunc.apply(x, shape)

        shape = (10, 10)
        x = torch.randn(shape, requires_grad=True)
        out = fn()
        out.sum().backward()
        self.assertEqual(out, x + 1)
        self.assertEqual(x.grad.shape, shape)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    @requires_gpu
    def test_triton_kernel_basic(self):
        # [FUNCTION DOC] Triton kernel basic test (requires a GPU): a Function whose forward
        # launches the add_kernel triton kernel and whose backward multiplies by the other input;
        # compiled with inductor fullgraph, backward is run and the output must equal x + y.
        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward launches the add_kernel triton kernel
            # over the elements; backward returns x * grad_output and y * grad_output (not the
            # real derivative of an add).
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: saves x and y, allocates zeros like x,
                # launches add_kernel with grid built from the element count and block size 16,
                # returns the output.
                ctx.save_for_backward(x, y)
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (x * grad_output, y * grad_output).
                x, y = ctx.saved_tensors
                return x * grad_output, y * grad_output

        @torch.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fullgraph inductor-compiled function returning Add applied to
            # x and y.
            z = Add.apply(x, y)
            return z

        x = torch.randn(10, device=device_type, requires_grad=True)
        y = torch.randn(10, device=device_type, requires_grad=True)
        z = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)

    @requires_gpu
    def test_triton_kernel_multiple_out(self):
        # [FUNCTION DOC] Triton kernel test with multiple outputs (requires a GPU): same as the
        # basic triton test but forward also returns the input x as a second output and stores x
        # and y as ctx attributes. Compiled with inductor fullgraph; checks the first output
        # equals x + y.
        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves x, y and stores them as ctx
            # attributes, launches add_kernel and returns (output, x); backward combines saved
            # and attribute tensors.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: saves x, y, stores them on ctx, launches
                # add_kernel with block size 16, returns (output, x).
                ctx.save_for_backward(x, y)
                ctx.t1 = x
                ctx.t2 = y
                output = torch.zeros_like(x)
                n_elements = output.numel()
                grid = lambda meta: (  # noqa: E731
                    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
                )
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
                return output, x

            @staticmethod
            def backward(ctx, grad_output, old_x):
                # [FUNCTION DOC] HELPER: Backward: returns (old_x * x * ctx.t1 * grad_output, y *
                # ctx.t2 * grad_output).
                x, y = ctx.saved_tensors
                x1 = ctx.t1
                y1 = ctx.t2
                return old_x * x * x1 * grad_output, y * y1 * grad_output

        @torch.compile(fullgraph=True, backend="inductor")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Fullgraph inductor-compiled function returning Add applied to
            # x and y.
            z = Add.apply(x, y)
            return z

        x = torch.randn(10, device=device_type, requires_grad=True)
        y = torch.randn(10, device=device_type, requires_grad=True)
        z, _ = f(x, y)
        loss = z.sum()
        loss.backward()
        self.assertEqual(x + y, z)

    @requires_gpu
    def test_triton_kernel_backward_mutation_with_aliased_output(self):
        # [FUNCTION DOC] Triton test (requires a GPU) for a partitioner case: backward launches
        # a triton kernel that mutates a buffer which is also a forward output aliasing the
        # input x. Compiled with aot_eager (no fullgraph); only checks that forward and backward
        # run without error.
        """Test autograd function where backward triton kernel mutates a buffer
        that is also returned as a forward output aliasing an input.

        This triggers a specific partitioner case where a forward output depends
        on backward nodes (the mutation) but should be replaced with the original
        input.
        """
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Triton kernel. Per program block loads grad_out, stores it
        # to grad_x, and adds grad_out into buffer in place (mutating buffer_ptr), with
        # masking against n_elements.
        @triton.jit
        def mul_and_mutate_kernel(
            grad_out_ptr,
            x_ptr,
            grad_x_ptr,
            buffer_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
            tl.store(grad_x_ptr + offsets, grad_out, mask=mask)
            buf = tl.load(buffer_ptr + offsets, mask=mask)
            tl.store(buffer_ptr + offsets, buf + grad_out, mask=mask)

        class TritonAddWithBuffer(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward runs add_kernel and returns the output
            # plus x itself as an aliased buffer; backward runs the mutating kernel on the saved
            # buffer and returns grad_x twice.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: launches add_kernel (block size 1024) into an
                # empty tensor, saves x and the buffer (which is x), stores n_elements on ctx,
                # returns (out, buffer).
                n_elements = x.numel()
                out = torch.empty_like(x)
                buffer = x

                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
                add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)

                ctx.save_for_backward(x, buffer)
                ctx.n_elements = n_elements
                return out, buffer

            @staticmethod
            def backward(ctx, grad_out, grad_buffer):
                # [FUNCTION DOC] HELPER: Backward: allocates grad_x, launches
                # mul_and_mutate_kernel on grad_out, x, grad_x and the saved buffer, and returns
                # (grad_x, clone of grad_x).
                x, buffer = ctx.saved_tensors
                n_elements = ctx.n_elements
                grad_x = torch.empty_like(x)

                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
                mul_and_mutate_kernel[grid](
                    grad_out, x, grad_x, buffer, n_elements, BLOCK_SIZE=1024
                )

                return grad_x, grad_x.clone()

        @torch.compile(backend="aot_eager")
        def compiled_f(x, y):
            # [FUNCTION DOC] HELPER: aot_eager-compiled function applying TritonAddWithBuffer to
            # x and y.
            return TritonAddWithBuffer.apply(x, y)

        x = torch.randn(1024, device=device_type, requires_grad=False)
        y = torch.randn(1024, device=device_type, requires_grad=True)

        out_compiled, buffer_compiled = compiled_f(x, y)
        loss_compiled = out_compiled.sum()
        loss_compiled.backward()

    @requires_gpu
    def test_triton_kernel_backward_readonly_passthrough_output(self):
        # [FUNCTION DOC] Triton test (requires a GPU): backward launches a kernel that reads a
        # read-only tensor (offsets) which is also a forward output. Checks that compilation under
        # aot_eager works and that x receives a gradient.
        """Test autograd function where backward triton kernel takes a read-only
        tensor (e.g. seq_offsets) that is also a forward output.

        The functional wrapper for capture_triton returns all kwargs as outputs
        including read-only ones. When the wrapper is backward-only, a getitem
        extracting the read-only tensor is invalid in the forward graph but still
        listed as a forward output. The partitioner must replace it with the
        original input node.
        """
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Triton kernel. Per block copies grad_out into grad_x and does a
        # read-only load of the first element of offsets_ptr (the loaded value is unused).
        @triton.jit
        def bwd_kernel(
            grad_out_ptr,
            offsets_ptr,
            grad_x_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = block_start < n_elements
            grad_out = tl.load(grad_out_ptr + block_start, mask=mask)
            # Read offsets (read-only, not written back)
            _offsets_val = tl.load(offsets_ptr + 0)
            tl.store(grad_x_ptr + block_start, grad_out, mask=mask)

        class FnWithReadOnlyBwdInput(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves offsets and returns
            # (x.clone(), offsets); backward runs the triton kernel to copy grad_out to grad_x
            # and returns (grad_x, None).
            @staticmethod
            def forward(ctx, x, offsets):
                # [FUNCTION DOC] HELPER: Forward: saves offsets, records x.numel() on ctx and
                # returns (x.clone(), offsets).
                ctx.save_for_backward(offsets)
                ctx.n_elements = x.numel()
                return x.clone(), offsets

            @staticmethod
            def backward(ctx, grad_out, grad_offsets):
                # [FUNCTION DOC] HELPER: Backward: launches bwd_kernel (block size 1024) over
                # grad_out, the saved offsets and a new grad_x; returns (grad_x, None).
                (offsets,) = ctx.saved_tensors
                n_elements = ctx.n_elements
                grad_x = torch.empty_like(grad_out)

                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
                bwd_kernel[grid](grad_out, offsets, grad_x, n_elements, BLOCK_SIZE=1024)

                return grad_x, None

        @torch.compile(backend="aot_eager")
        def compiled_f(x, offsets):
            # [FUNCTION DOC] HELPER: aot_eager-compiled function applying FnWithReadOnlyBwdInput
            # to x and offsets.
            return FnWithReadOnlyBwdInput.apply(x, offsets)

        x = torch.randn(1024, device=device_type, requires_grad=True)
        offsets = torch.tensor([0, 512, 1024], device=device_type)

        out, offsets_out = compiled_f(x, offsets)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_nonlocal_list_mutation_in_autograd_function(self):
        # [FUNCTION DOC] Checks that forward appending a tensor to an enclosing Python list
        # is handled under fullgraph aot_eager: the list is a non-tensor input mutated
        # inside forward, and the function returns the list's first element. Compares both
        # outputs with eager.
        """Test that nonlocal list mutation in autograd.Function forward is handled correctly."""

        class SimpleAutogradFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward does matmul work, appends sin of x to
            # the list z (side effect) and returns two tensors; backward adds the two gradients
            # and returns None for z.
            @staticmethod
            def forward(ctx, x, z):
                # [FUNCTION DOC] HELPER: Forward: o = x @ x @ x; appends x.sin() to z;
                # returns (cos(sin(o)), sin(x)).
                # Simple computation
                o = torch.matmul(x, x) @ x
                out = x.sin()
                # Mutate the nonlocal list
                z.append(out)
                return torch.cos(torch.sin(o)), torch.sin(x)

            @staticmethod
            def backward(ctx, grad_output1, grad_output2):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output1 + grad_output2, None).
                # Simple backward
                return grad_output1 + grad_output2, None

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an empty list, applies SimpleAutogradFunc with x and
            # the list, adds the two outputs and returns the sum plus the first list element.
            z = []

            outs = SimpleAutogradFunc.apply(x, z)
            out1 = outs[0]
            # Check that the extra output pytree handling is done properly
            out2 = outs[-1]

            return out1 + out2, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    def test_rewired_bwd_output(self):
        # [FUNCTION DOC] Checks that saved tensors in backward are correctly rewired when the
        # Function sits after other ops and inputs have a dynamic dimension: saves an
        # intermediate, an input and an output. Compares forward and x.grad with eager and checks
        # the captured graph text against an expected inline string.
        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward computes sin(x) * cos(y) and a + b,
            # saving a, x and the result; backward returns a combination of saved tensors and
            # gradients (not the real derivative).
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: a = sin(x), b = cos(y); saves a, x, result =
                # a*b; returns (result, a + b).
                a = torch.sin(x)
                b = torch.cos(y)
                result = a * b
                # Save input, output and intermediate to test all cases
                ctx.save_for_backward(a, x, result)
                return result, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Backward: returns (a * grad_b * 2 + x, result +
                # grad_a * 3).
                (a, x, result) = ctx.saved_tensors
                return a * grad_b * 2 + x, result + grad_a * 3

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies Add to cos(x) and cos(y) and returns the sum of
            # its two outputs.
            z = Add.apply(torch.cos(x), torch.cos(y))
            return z[0] + z[1]

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, fullgraph=True, backend=backend)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        torch._dynamo.mark_dynamic(x_clone, 0)
        torch._dynamo.mark_dynamic(y_clone, 0)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        self.assertExpectedInline(
            torch._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s17, 8]", s77: "Sym(s17)", L_y_: "f32[s17, 8]", s17: "Sym(s17)"):
        l_x_ = L_x_
        l_y_ = L_y_

        cos: "f32[s17, 8]" = torch.cos(l_x_);  l_x_ = None
        cos_1: "f32[s17, 8]" = torch.cos(l_y_);  l_y_ = None
        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, s77, cos, s17, cos_1, non_differentiable_idx = [], saved_for_backward_idx = [1, 2, 3]);  fwd_body_0 = bwd_body_0 = s77 = cos = s17 = cos_1 = None
        getitem: "f32[s17, 8]" = autograd_function_apply[0]
        getitem_1: "f32[s17, 8]" = autograd_function_apply[1];  autograd_function_apply = None

        add: "f32[s17, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, grad_a: "f32[s17, 8]", grad_b: "f32[s17, 8]", s17: "Sym(s17)", a: "f32[s17, 8]", arg: "f32[s17, 8]", result: "f32[s17, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[s17, 8]" = a * grad_b;  a = grad_b = None
            mul_1: "f32[s17, 8]" = mul * 2;  mul = None
            add: "f32[s17, 8]" = mul_1 + arg;  mul_1 = arg = None
            mul_2: "f32[s17, 8]" = grad_a * 3;  grad_a = None
            add_1: "f32[s17, 8]" = result + mul_2;  result = mul_2 = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, add, None, add_1)

    class fwd_body_0(torch.nn.Module):
        def forward(self, s77: "Sym(s17)", cos: "f32[s17, 8]", s17: "Sym(s17)", cos_1: "f32[s17, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            a: "f32[s17, 8]" = torch.sin(cos)

            b: "f32[s17, 8]" = torch.cos(cos_1);  cos_1 = None

            result: "f32[s17, 8]" = a * b

            out: "f32[s17, 8]" = a + b;  b = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((result, out), (s17, a, cos, result))
""",
        )

    def test_duplicate_input_accumulates_grad(self):
        # [FUNCTION DOC] Checks gradient accumulation when the same tensor is passed twice as
        # inputs to a Function. With fullgraph False Dynamo falls back to eager and
        # results/gradients must match eager; with fullgraph True the compile must raise
        # Unsupported mentioning a duplicate tensor input.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves a and b and returns a * b + a + b;
            # backward returns (grad_output * (b + 1), grad_output * (a + 1)).
            @staticmethod
            def forward(ctx, a, b):
                # [FUNCTION DOC] HELPER: Forward: saves a, b; returns a * b + a + b.
                ctx.save_for_backward(a, b)
                return a * b + a + b

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * (b + 1), grad_output *
                # (a + 1)).
                a, b = ctx.saved_tensors
                return grad_output * (b + 1), grad_output * (a + 1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to (x, x) and returns the sum.
            return Foo.apply(x, x).sum()

        def check_fallback():
            # [FUNCTION DOC] HELPER: Compares eager against an eager-backend non-fullgraph compile
            # on a fixed 5-element tensor: resets Dynamo, runs backward in both cases and asserts
            # that outputs and input gradients are equal.
            x = torch.tensor([0.7, -1.3, 2.1, 0.05, -0.5], requires_grad=True)

            x_ref = x.detach().clone().requires_grad_(True)
            ref = fn(x_ref)
            ref.backward()

            torch._dynamo.reset()
            x_compiled = x.detach().clone().requires_grad_(True)
            res = torch.compile(fn, backend="eager", fullgraph=False)(x_compiled)
            res.backward()

            self.assertEqual(ref, res)
            self.assertEqual(x_ref.grad, x_compiled.grad)

        check_fallback()

        torch._dynamo.reset()
        x = torch.randn(5, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "duplicate tensor input",
        ):
            out = torch.compile(fn, backend="eager", fullgraph=True)(x)
            out.backward()

    def test_udf_output(self):
        # [FUNCTION DOC] Checks that a Function can return a user-defined (non-tensor) object
        # containing tensors: compiled under aot_eager fullgraph, the sum of the object's two
        # fields plus another output equals eager, gradients on x match, and the captured graph
        # text is compared with an expected inline string.
        class Foo:
            # [CLASS DOC] HELPER: Plain Python class with two attributes a and b
            # (no tensor semantics).
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes a and b.
                self.a = a
                self.b = b

        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns a Foo object holding sin(x) and
            # cos(y) plus x * y, saving sin(x); backward returns (grad_b * 2, a * grad_b * 3).
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Forward: a = sin(x), b = cos(y); saves a; returns (Foo(a,
                # b), x * y).
                a = torch.sin(x)
                b = torch.cos(y)
                ctx.save_for_backward(a)
                return Foo(a, b), x * y

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_b * 2, a * grad_b * 3) using
                # the saved a.
                (a,) = ctx.saved_tensors
                return grad_b * 2, a * grad_b * 3

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies Add to x and y and returns z[0].a + z[0].b + z[1].
            z = Add.apply(x, y)
            return z[0].a + z[0].b + z[1]

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, fullgraph=True, backend=backend)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        self.assertExpectedInline(
            torch._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_y_: "f32[8, 8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, l_y_, non_differentiable_idx = [], saved_for_backward_idx = [0]);  fwd_body_0 = bwd_body_0 = l_x_ = l_y_ = None
        getitem: "f32[8, 8]" = autograd_function_apply[0]
        getitem_1: "f32[8, 8]" = autograd_function_apply[1]

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None

        getitem_2: "f32[8, 8]" = autograd_function_apply[2];  autograd_function_apply = None

        add_1: "f32[8, 8]" = add + getitem_2;  add = getitem_2 = None
        return (add_1,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, unused_0, unused_1, grad_b: "f32[8, 8]", a: "f32[8, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[8, 8]" = grad_b * 2
            mul_1: "f32[8, 8]" = a * grad_b;  a = grad_b = None
            mul_2: "f32[8, 8]" = mul_1 * 3;  mul_1 = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (mul, mul_2)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]", l_y_: "f32[8, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            a: "f32[8, 8]" = torch.sin(l_x_)

            b: "f32[8, 8]" = torch.cos(l_y_)

            out: "f32[8, 8]" = l_x_ * l_y_;  l_x_ = l_y_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((a, b, out), (a,))
""",
        )

    def test_aliasing_output(self):
        # [FUNCTION DOC] Checks that a Function returning its input unchanged yields a distinct
        # output under compile: the traced graph must contain a view_as call (so the identity
        # check on the output is false and the sin branch is used). Compares to eager and to an
        # expected inline graph string.
        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns its input object; backward
            # returns the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x itself.
                return x

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Backward: returns grad_out.
                return grad_out

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Add to x and returns cos of the result if it is the
            # same object as x, else sin of the result.
            y = Add.apply(x)
            if y is x:
                return torch.cos(y)
            return torch.sin(y)

        x = torch.randn(8, 8, requires_grad=True)

        ref = fn(x)
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Must have `view_as`
        self.assertTrue(
            "view_as" in backend.graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            torch._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = None
        getitem: "f32[8, 8]" = autograd_function_apply[0];  autograd_function_apply = None

        sin: "f32[8, 8]" = torch.sin(getitem);  getitem = None
        return (sin,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, grad_out: "f32[8, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (grad_out,)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None

            view_as: "f32[8, 8]" = l_x_.view_as(l_x_);  l_x_ = None
            return ((view_as,), ())
""",
        )

    def test_nn_module_dataclasses_as_inputs(self):
        # [FUNCTION DOC] Checks that an nn.Module and a dataclass (with an int and a tensor) can
        # be passed as Function inputs under aot_eager fullgraph: module attributes and dataclass
        # fields are baked as constants or graph inputs. Compares to eager and to an expected
        # inline graph string.
        @dataclass
        class InputData:
            # [CLASS DOC] HELPER: Local dataclass with an int field count and tensor field values.
            count: int
            values: torch.Tensor

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module holding the int attribute a (4) and a random tensor
            # attribute b of shape (4, 4).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module with a = 4 and a random b.
                super().__init__()
                self.a = 4
                self.b = torch.randn(4, 4)

        module = Mod()

        # Create input dataclass
        input_data = InputData(count=5, values=torch.randn(4, 4))

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward combines module attributes with
            # dataclass fields and returns two tensors; backward returns hardcoded
            # multiples of grad_output.
            @staticmethod
            def forward(
                ctx,
                module: torch.nn.Module,
                input_data: InputData,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Forward: output_tensor = module.a + module.b * count *
                # values; returns (output_tensor + x, values + x).
                # Extract inputs
                count = input_data.count
                values = input_data.values

                output_tensor = module.a + module.b * count * values

                return output_tensor + x, values + x

            @staticmethod
            def backward(ctx, grad_output, output_data):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * 4, None,
                # grad_output * 2).
                return grad_output * 4, None, grad_output * 2

        def fn(input_data, x):
            # [FUNCTION DOC] HELPER: Applies Foo to module, input_data and x and returns x + y +
            # y_data + y_data.
            y, y_data = Foo.apply(module, input_data, x)
            return x + y + y_data + y_data

        # Call the function
        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(input_data, x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(input_data, x)
        self.assertEqual(ref, res)

        # Must have `view_as`
        self.assertExpectedInline(
            torch._dynamo.testing.normalize_gm(
                backend.graphs[0].print_readable(print_output=False)
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_input_data_values: "f32[4, 4]", L_module_b: "f32[4, 4]", L_x_: "f32[4, 4]"):
        l_input_data_values = L_input_data_values
        l_module_b = L_module_b
        l_x_ = L_x_

        bwd_body_0 = self.bwd_body_0
        fwd_body_0 = self.fwd_body_0
        autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_module_b, l_input_data_values, l_x_, non_differentiable_idx = [], saved_for_backward_idx = []);  fwd_body_0 = bwd_body_0 = l_module_b = l_input_data_values = None
        getitem: "f32[4, 4]" = autograd_function_apply[0]

        add: "f32[4, 4]" = l_x_ + getitem;  l_x_ = getitem = None

        getitem_1: "f32[4, 4]" = autograd_function_apply[1];  autograd_function_apply = None

        add_1: "f32[4, 4]" = add + getitem_1;  add = None
        add_2: "f32[4, 4]" = add_1 + getitem_1;  add_1 = getitem_1 = None
        return (add_2,)

    class bwd_body_0(torch.nn.Module):
        def forward(self, grad_output: "f32[4, 4]", output_data: "f32[4, 4]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[4, 4]" = grad_output * 4;  mul = None
            mul_1: "f32[4, 4]" = grad_output * 2;  grad_output = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return (None, None, mul_1)

    class fwd_body_0(torch.nn.Module):
        def forward(self, l_module_b: "f32[4, 4]", l_input_data_values: "f32[4, 4]", l_x_: "f32[4, 4]"):
            _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

            mul: "f32[4, 4]" = l_module_b * 5;  l_module_b = None
            mul_1: "f32[4, 4]" = mul * l_input_data_values;  mul = None
            output_tensor: "f32[4, 4]" = 4 + mul_1;  mul_1 = None

            out: "f32[4, 4]" = output_tensor + l_x_;  output_tensor = None
            out_1: "f32[4, 4]" = l_input_data_values + l_x_;  l_input_data_values = l_x_ = None

            _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
            return ((out, out_1), ())
""",
        )

    def test_nn_module_dataclasses_as_io(self):
        # [FUNCTION DOC] Like the module/dataclass input test but the Function also returns a
        # dataclass (OutputData with two tensors) as its second output; compares the aot_eager
        # fullgraph result with eager. No graph text check.
        @dataclass
        class InputData:
            # [CLASS DOC] HELPER: Local dataclass with an int field count and tensor field values.
            count: int
            values: torch.Tensor

        @dataclass
        class OutputData:
            # [CLASS DOC] HELPER: Local dataclass with tensor fields result1 and result2, used as
            # a Function output.
            result1: torch.Tensor
            result2: torch.Tensor

        # Create a simple linear module

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with int attribute a (4) and random tensor b.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the module with a = 4 and a random b.
                super().__init__()
                self.a = 4
                self.b = torch.randn(4, 4)

        module = Mod()

        # Create input dataclass
        input_data = InputData(count=5, values=torch.randn(4, 4))

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns a tensor and an OutputData
            # dataclass; backward returns hardcoded multiples of grad_output.
            @staticmethod
            def forward(
                ctx,
                module: torch.nn.Module,
                input_data: InputData,
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, OutputData]:
                # [FUNCTION DOC] HELPER: Forward: computes output_tensor from module and
                # input_data, builds OutputData(result1, result2) and returns (output_tensor
                # + x, output_data).
                # Extract inputs
                count = input_data.count
                values = input_data.values

                output_tensor = module.a + module.b * count * values

                output_data = OutputData(
                    result1=output_tensor + count, result2=output_tensor * (count + 1)
                )

                return output_tensor + x, output_data

            @staticmethod
            def backward(ctx, grad_output, output_data):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * 4, None,
                # grad_output * 2).
                return grad_output * 4, None, grad_output * 2

        def fn(input_data, x):
            # [FUNCTION DOC] HELPER: Applies Foo and returns x + y +
            # y_data.result1 + y_data.result2.
            y, y_data = Foo.apply(module, input_data, x)
            return x + y + y_data.result1 + y_data.result2

        # Call the function
        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(input_data, x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, fullgraph=True, backend=backend)
        res = opt_fn(input_data, x)
        self.assertEqual(ref, res)

    def test_inplace_op_with_side_effect_wrong_grad(self):
        # [FUNCTION DOC] Regression test for issue 180642: forward appends an intermediate to an
        # enclosing list (side effect) and does an in-place add on a fresh scalar. The compiled
        # fullgraph eager gradient must equal eager (previously the gradient was zero because an
        # alias output overwrote the output slot).
        # Repro for https://github.com/pytorch/pytorch/issues/180642
        # In-place op inside autograd.Function.forward combined with a side
        # effect (list append) causes collect_intermediate_outputs to add a
        # pre-mutation alias as an extra subgraph output. Because the alias
        # shares the same TensorImpl as the real return value,
        # set_gradient_edge overwrites output_nr, routing the backward
        # gradient to the wrong slot and producing zero gradients.
        captured = []

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward appends sin(x) to captured, builds a
            # scalar zero tensor and adds the sum of x in place; backward expands the gradient
            # to size 8.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: appends torch.sin(x) to the captured list,
                # creates a scalar zero loss, adds x.sum() in place and returns loss.
                intermediate = torch.sin(x)
                captured.append(intermediate)
                loss = torch.tensor(0.0, device=x.device)
                loss += x.sum()
                return loss

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad expanded to 8 elements.
                return grad.expand(8)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x.
            return Foo.apply(x)

        x = torch.randn(8, device="cpu", requires_grad=True)

        # Eager reference
        x_ref = x.detach().clone().requires_grad_(True)
        captured.clear()
        out_ref = fn(x_ref)
        out_ref.backward()

        # Compiled
        torch._dynamo.reset()
        captured.clear()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()

        self.assertEqual(x_ref.grad, x_c.grad)

    def test_aliased_intermediate_captured_by_side_effect(self):
        # [FUNCTION DOC] Checks that an intermediate view of the input captured by a side effect
        # (appended to an outer list) and later used outside the Function produces a
        # RuntimeError that tells the user the value aliases an input or output and to clone it,
        # under fullgraph eager.
        # An intermediate that aliases an input is captured via side effect
        # and used by the outer graph. filter_aliased_intermediates drops it
        # from the subgraph outputs, but the outer graph still needs it.
        # We should get a clear error telling the user to clone.
        captured = []

        class Bar(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward appends x.view(-1) to captured and
            # returns x.sum(); backward expands the gradient to 8.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: view = x.view(-1), appended to
                # captured; returns x.sum().
                view = x.view(-1)
                captured.append(view)
                return x.sum()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad.expand(8).
                return grad.expand(8)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Bar to x and adds the sum of the
            # first captured tensor.
            out = Bar.apply(x)
            return out + captured[0].sum()

        x = torch.randn(8, requires_grad=True)
        captured.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            "aliases an input or output.*clone",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

    def test_nullified_ctx_manager_side_effect_in_backward(self):
        # [FUNCTION DOC] Checks that a context manager in backward that sets a boolean attribute
        # and restores it has no net side effect, so a fullgraph eager compile works and its input
        # gradient equals eager. Resets Dynamo between the eager and compiled runs.
        # A context manager that flips a boolean attribute on enter and
        # restores it on exit has no net side effect. Dynamo defers the
        # check and validates nullification after tracing.
        import contextlib

        class State:
            # [CLASS DOC] HELPER: Plain state holder with a boolean attribute
            # enabled, initially False.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets enabled to False.
                self.enabled = False

        state = State()

        @contextlib.contextmanager
        def toggle_flag():
            # [FUNCTION DOC] HELPER: Context manager: saves state.enabled, sets it to True,
            # yields, and restores the old value in a finally block.
            old = state.enabled
            state.enabled = True
            try:
                yield
            finally:
                state.enabled = old

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward clones the gradient
            # inside the toggle_flag context.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: enters toggle_flag and returns grad.clone().
                with toggle_flag():
                    return grad.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x and sums the result.
            return Foo.apply(x).sum()

        x = torch.randn(8, requires_grad=True)

        # Eager reference
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = fn(x_ref)
        out_ref.backward()

        # Compiled
        torch._dynamo.reset()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()

        self.assertEqual(x_ref.grad, x_c.grad)

    def test_non_nullified_side_effect_in_backward_fails(self):
        # [FUNCTION DOC] Checks that a backward which sets an outer object's attribute to True
        # without restoring it still fails under fullgraph eager: calling compile and backward
        # must raise Unsupported.
        # A backward that mutates an outer-scope attribute without restoring
        # it should still fail, even with deferred side-effect checking.
        class State:
            # [CLASS DOC] HELPER: Plain state holder with a boolean attribute
            # enabled, initially False.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets enabled to False.
                self.enabled = False

        state = State()

        class Bad(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward sets state.enabled to
            # True (never restored) and clones the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: sets state.enabled = True
                # and returns grad.clone().
                state.enabled = True
                return grad.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Bad to x and sums the result.
            return Bad.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            out = torch.compile(fn, backend="eager", fullgraph=True)(x)
            out.backward()

    def test_nullified_dict_attr_side_effect_in_backward(self):
        # [FUNCTION DOC] Checks that a backward which swaps a dict-valued attribute and then
        # restores the original is treated as nullified: fullgraph eager compile gives the same
        # input gradient as eager.
        # Attribute that holds a dict is swapped and restored — nullified.
        class Config:
            # [CLASS DOC] HELPER: Plain holder with an options dict (mode fast, level 3).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets options to a dict with mode "fast" and level 3.
                self.options = {"mode": "fast", "level": 3}

        cfg = Config()

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward replaces cfg.options
            # and restores it before returning a clone of the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: saves cfg.options, replaces it with a different
                # dict, restores the saved one and returns grad.clone().
                old = cfg.options
                cfg.options = {"mode": "slow", "level": 1}
                cfg.options = old
                return grad.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x and sums the result.
            return Foo.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = fn(x_ref)
        out_ref.backward()

        torch._dynamo.reset()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()
        self.assertEqual(x_ref.grad, x_c.grad)

    def test_non_nullified_dict_attr_side_effect_in_backward_fails(self):
        # [FUNCTION DOC] Checks that swapping a dict-valued attribute in backward without
        # restoring it fails under fullgraph eager with Unsupported.
        # Attribute that holds a dict is swapped but NOT restored.
        class Config:
            # [CLASS DOC] HELPER: Plain holder with an options dict (mode fast, level 3).
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets options to a dict with mode "fast" and level 3.
                self.options = {"mode": "fast", "level": 3}

        cfg = Config()

        class Bad(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward replaces cfg.options
            # and never restores it.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: sets cfg.options to a new dict
                # and returns grad.clone().
                cfg.options = {"mode": "slow", "level": 1}
                return grad.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Bad to x and sums the result.
            return Bad.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            out = torch.compile(fn, backend="eager", fullgraph=True)(x)
            out.backward()

    def test_nullified_integer_counter_in_backward(self):
        # [FUNCTION DOC] Checks that an integer counter incremented and then decremented inside
        # backward is treated as a nullified mutation, so fullgraph eager gives the same
        # gradient as eager.
        # Integer attribute incremented then decremented — nullified.
        class Counter:
            # [CLASS DOC] HELPER: Plain holder with an integer depth attribute, initially 0.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets depth to 0.
                self.depth = 0

        counter = Counter()

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward increments the
            # counter, clones the gradient, decrements the counter and returns the clone.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: counter.depth += 1; result = grad.clone();
                # counter.depth -= 1; returns result.
                counter.depth += 1
                result = grad.clone()
                counter.depth -= 1
                return result

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x and sums the result.
            return Foo.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = fn(x_ref)
        out_ref.backward()

        torch._dynamo.reset()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()
        self.assertEqual(x_ref.grad, x_c.grad)

    def test_read_after_mutate_in_backward(self):
        # [FUNCTION DOC] Checks that code inside backward sees the value set by a traced context
        # manager: inside the set_mode context, backward reads the state mode and doubles the
        # gradient when it equals "special". Compiled fullgraph eager gradient must equal eager.
        # Code inside the backward reads the flipped value, confirming
        # the traced mutation is visible during tracing.
        import contextlib

        class State:
            # [CLASS DOC] HELPER: Plain holder with a string attribute mode, initially "default".
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets mode to "default".
                self.mode = "default"

        state = State()

        @contextlib.contextmanager
        def set_mode(mode):
            # [FUNCTION DOC] HELPER: Context manager: saves state.mode, sets it to the given mode,
            # yields, restores the old mode in a finally block.
            old = state.mode
            state.mode = mode
            try:
                yield
            finally:
                state.mode = old

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward runs under set_mode
            # special and returns grad times 2.0 if the mode is special, else grad.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: inside set_mode("special"), returns grad * 2.0
                # when state.mode is "special", else grad.
                with set_mode("special"):
                    if state.mode == "special":
                        return grad * 2.0
                    return grad

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x and sums the result.
            return Foo.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = fn(x_ref)
        out_ref.backward()

        torch._dynamo.reset()
        x_c = x.detach().clone().requires_grad_(True)
        out_c = torch.compile(fn, backend="eager", fullgraph=True)(x_c)
        out_c.backward()
        self.assertEqual(x_ref.grad, x_c.grad)

    def test_property_with_side_effect_in_backward_fails(self):
        # [FUNCTION DOC] Checks that a context manager whose property setter has a real side
        # effect (incrementing a counter on every set) is not treated as nullified, even though
        # the value is restored: fullgraph eager compile plus backward must raise Unsupported.
        # A property setter that has a real side effect (incrementing a
        # counter) is NOT a nullified mutation even if the value is restored.
        # Dynamo traces through the setter and sees the non-nullified
        # set_count increment.
        import contextlib

        class State:
            # [CLASS DOC] HELPER: State holder with a private flag, a set counter, and an enabled
            # property whose setter increments the counter before storing the value.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the private _enabled flag to False and
                # set_count to 0.
                self._enabled = False
                self.set_count = 0

            @property
            def enabled(self):
                # [FUNCTION DOC] HELPER: Property getter: returns the private _enabled flag.
                return self._enabled

            @enabled.setter
            def enabled(self, val):
                # [FUNCTION DOC] HELPER: Property setter: increments set_count and stores val in
                # _enabled (the counter increment is the persistent side effect).
                self.set_count += 1
                self._enabled = val

        state = State()

        @contextlib.contextmanager
        def toggle():
            # [FUNCTION DOC] HELPER: Context manager: saves state.enabled, sets it to True through
            # the property setter, yields, restores the old value in a finally block.
            old = state.enabled
            state.enabled = True
            try:
                yield
            finally:
                state.enabled = old

        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward clones x; backward clones the gradient
            # inside the toggle context.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: returns x.clone().
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: enters toggle() and returns grad.clone().
                with toggle():
                    return grad.clone()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies Foo to x and sums the result.
            return Foo.apply(x).sum()

        x = torch.randn(8, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            out = torch.compile(fn, backend="eager", fullgraph=True)(x)
            out.backward()

    def test_apply_kwargs_old_style(self):
        # [FUNCTION DOC] Checks apply called with a keyword argument for an old-style Function
        # (ctx in forward): fullgraph eager compile of a function calling apply with factor=6
        # matches eager in the output, and backward gives the same gradient as eager.
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves input, stores factor on ctx and
            # returns factor times sin(input); backward returns grad times factor times
            # cos(input) and None.
            @staticmethod
            def forward(ctx, input, factor=1):
                # [FUNCTION DOC] HELPER: Forward: saves input, ctx.factor = factor, returns
                # factor * torch.sin(input).
                ctx.save_for_backward(input)
                ctx.factor = factor
                return factor * torch.sin(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * ctx.factor
                # * cos(input), None).
                (input,) = ctx.saved_tensors
                return grad_output * ctx.factor * torch.cos(input), None

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MySin with factor given as a keyword.
            return MySin.apply(x, factor=6)

        x = torch.tensor([0.812], requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        ref.backward()
        grad_ref = x.grad.clone()
        x.grad = None
        res.backward()
        self.assertEqual(x.grad, grad_ref)

    def test_apply_kwargs_setup_context(self):
        # [FUNCTION DOC] Checks apply with a keyword argument for a new-style Function (forward
        # without ctx plus setup_context): fullgraph eager output and input gradient match eager.
        class MyScale(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns factor * x; setup_context saves
            # x and stores factor; backward returns (gO * factor, None).
            @staticmethod
            def forward(x, factor=1):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns factor * x.
                return factor * x

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: Saves x from the inputs tuple and stores factor on
                # ctx; output unused.
                x, factor = inputs
                ctx.save_for_backward(x)
                ctx.factor = factor

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: Backward: returns (gO * ctx.factor, None).
                return gO * ctx.factor, None

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MyScale to x with factor given as a keyword.
            return MyScale.apply(x, factor=3)

        x = torch.tensor([2.0], requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

        ref.backward()
        grad_ref = x.grad.clone()
        x.grad = None
        res.backward()
        self.assertEqual(x.grad, grad_ref)

    def test_apply_kwargs_all_kwargs(self):
        # [FUNCTION DOC] Checks apply called with all arguments as keywords in reversed order (y
        # then x) for a new-style Function: the fullgraph eager output matches eager.
        class Add(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Adds two inputs; setup_context is a no-op;
            # backward returns the gradient for both inputs.
            @staticmethod
            def forward(x, y):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns x + y.
                return x + y

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: setup_context that does nothing.
                pass

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output, grad_output).
                return grad_output, grad_output

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies Add with y and x passed as keyword arguments.
            return Add.apply(y=y, x=x)

        x = torch.tensor([2.0], requires_grad=True)
        y = torch.tensor([3.0], requires_grad=True)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(res, ref)

    def test_apply_kwargs_inference(self):
        # [FUNCTION DOC] Checks apply with a keyword argument when no input requires grad
        # (inference): the fullgraph eager output matches eager.
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns factor times sin(input);
            # backward returns (grad_output, None).
            @staticmethod
            def forward(ctx, input, factor=1):
                # [FUNCTION DOC] HELPER: Forward: returns factor * torch.sin(input).
                return factor * torch.sin(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output, None).
                return grad_output, None

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MySin with factor given as a keyword.
            return MySin.apply(x, factor=6)

        x = torch.tensor([0.812])
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_apply_kwargs_with_custom_signature(self):
        # [FUNCTION DOC] Checks that keyword resolution for apply uses the forward's code object
        # parameter names and ignores any __signature__ attribute: a different signature (x,
        # scale) is attached to forward, yet apply with factor=6 must still compile under
        # fullgraph eager and match eager. Mutates the forward function by assigning __signature__
        # (side effect on the local class only).
        # __signature__ is ignored; resolution uses __code__ (matching C++ behavior)
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward saves input, stores factor on ctx and
            # returns factor times sin(input); backward returns grad times factor times
            # cos(input) and None.
            @staticmethod
            def forward(ctx, input, factor=1):
                # [FUNCTION DOC] HELPER: Forward: saves input, ctx.factor = factor, returns
                # factor * torch.sin(input).
                ctx.save_for_backward(input)
                ctx.factor = factor
                return factor * torch.sin(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * ctx.factor
                # * cos(input), None).
                (input,) = ctx.saved_tensors
                return grad_output * ctx.factor * torch.cos(input), None

        # Set a __signature__ with different param names
        import inspect as _inspect

        MySin.forward.__signature__ = _inspect.Signature(
            [
                _inspect.Parameter("ctx", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
                _inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
                _inspect.Parameter(
                    "scale",
                    _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=1,
                ),
            ]
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MySin with the keyword factor, which matches the code
            # object not the attached signature.
            # "factor" matches __code__, not __signature__
            return MySin.apply(x, factor=6)

        x = torch.tensor([0.812], requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_apply_kwargs_with_wrapped(self):
        # [FUNCTION DOC] Checks that keyword resolution for apply ignores the wrapped attribute
        # created by functools.wraps and uses the wrapper's own code object: forward is decorated
        # with a pass-through wrapper (parameters ctx, input, factor) while the inner function
        # names the parameter multiplier; apply with factor=6 must compile under fullgraph eager
        # and match eager.
        # __wrapped__ is ignored; resolution uses __code__ of the wrapper
        import functools

        def my_decorator(f):
            # [FUNCTION DOC] HELPER: Decorator factory that wraps f in a function with
            # signature (ctx, input, factor=1) preserving metadata via functools.wraps and
            # returns the wrapper.
            @functools.wraps(f)
            def wrapper(ctx, input, factor=1):
                # [FUNCTION DOC] HELPER: Wrapper: calls f with ctx, input and factor positionally
                # and returns its result.
                return f(ctx, input, factor)

            return wrapper

        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward is decorated with my_decorator and saves
            # input, stores multiplier on ctx and returns multiplier times sin(input); backward
            # returns grad times factor times cos(input) and None.
            @staticmethod
            @my_decorator
            def forward(ctx, input, multiplier=1):
                # [FUNCTION DOC] HELPER: Forward (wrapped): saves input, ctx.factor = multiplier,
                # returns multiplier * torch.sin(input).
                ctx.save_for_backward(input)
                ctx.factor = multiplier
                return multiplier * torch.sin(input)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns (grad_output * ctx.factor
                # * cos(input), None).
                (input,) = ctx.saved_tensors
                return grad_output * ctx.factor * torch.cos(input), None

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies MySin with the keyword factor, matching the
            # wrapper's parameter name.
            # "factor" matches the wrapper's __code__, not the wrapped fn
            return MySin.apply(x, factor=6)

        x = torch.tensor([0.812], requires_grad=True)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)


class AutogradFunctionFunctorchTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for autograd Function interactions with torch.func transforms (vmap
    # and grad) in combination with torch.compile (see issue 174067). Tests cover Functions with a
    # generated vmap rule, a broken custom vmap rule, and new-style versus old-style Functions
    # under torch.func.grad, both uncompiled baselines and compiled.
    """Tests for autograd.Function compatibility with torch.func transforms.

    See https://github.com/pytorch/pytorch/issues/174067
    """

    def test_vmap_generate_rule_compiled(self):
        # [FUNCTION DOC] Checks that a Function with generate_vmap_rule enabled can be vmapped
        # inside a fullgraph aot_eager compile: compares forward values and input gradient with
        # eager and asserts one frame.
        class Double(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. New-style Function with generate_vmap_rule set;
            # forward doubles x, setup_context does nothing and backward returns grad * 2.
            generate_vmap_rule = True

            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns x * 2.
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: setup_context that does nothing.
                pass

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Backward: returns grad * 2.
                return grad * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns torch.vmap of Double.apply applied to x.
            return torch.vmap(Double.apply)(x)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)

        x1 = torch.randn(2, 3, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        expected = fn(x1)
        expected.sum().backward()

        actual = opt_fn(x2)
        actual.sum().backward()

        self.assertEqual(actual, expected)
        self.assertEqual(x2.grad, x1.grad)
        self.assertEqual(cnt.frame_count, 1)

    def test_vmap_custom_rule_error_falls_back_to_eager(self):
        # [FUNCTION DOC] Checks that a custom vmap staticmethod that returns an incompatible
        # result raises RuntimeError whose message contains "returned an incompatible" when called
        # through a compiled (eager backend, no fullgraph) function.
        class BadVmap(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Forward returns zeros of x's shape;
            # setup_context does nothing; its vmap rule returns zeros of a smaller shape and a
            # tuple holding None (incompatible with the batched output).
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns zeros with the shape and
                # device of x.
                return torch.zeros(x.shape, device=x.device)

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: setup_context that does nothing.
                pass

            @staticmethod
            def vmap(info, in_dims, x):
                # [FUNCTION DOC] HELPER: vmap rule: ignores info and in_dims and returns (zeros
                # shaped like x without its first dimension, (None,)).
                return torch.zeros(x.shape[1:], device=x.device), (None,)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns torch.vmap of BadVmap.apply applied to x.
            return torch.vmap(BadVmap.apply)(x)

        opt_fn = torch.compile(fn, backend="eager")

        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            opt_fn(torch.randn(2, 3))

    def test_new_style_autograd_function_with_grad_no_compile(self):
        # [FUNCTION DOC] Baseline without compile: a new-style Function with setup_context works
        # under torch.func.grad; gradient of sum of the doubled input is a vector of twos.
        """Baseline: new-style autograd.Function works with torch.func.grad."""

        class NewStyleOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. New-style Function (forward without ctx) that
            # doubles x, saves x in setup_context and doubles the gradient in backward.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns x * 2.
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: setup_context: unpacks the single input and saves
                # it for backward.
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2.
                return grad_output * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of NewStyleOp applied to x.
            return NewStyleOp.apply(x).sum()

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = torch.func.grad(fn)(x)
        self.assertEqual(result, torch.tensor([2.0, 2.0]))

    def test_old_style_autograd_function_with_grad_no_compile(self):
        # [FUNCTION DOC] Baseline without compile: an old-style Function (forward with ctx, no
        # setup_context) fails under torch.func.grad with a RuntimeError saying it must override
        # the setup_context staticmethod.
        """Baseline: old-style autograd.Function fails with torch.func.grad."""

        class OldStyleOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Old-style Function that saves x in forward and
            # doubles it; backward doubles the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: saves x on ctx and returns x * 2.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2.
                return grad_output * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of OldStyleOp applied to x.
            return OldStyleOp.apply(x).sum()

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "must override the setup_context staticmethod"
        ):
            torch.func.grad(fn)(x)

    def test_new_style_autograd_function_with_grad_compiled(self):
        # [FUNCTION DOC] Checks that a new-style Function used inside a function compiled with the
        # eager backend still works under torch.func.grad applied to a loss that calls the
        # compiled function: gradient equals a vector of twos.
        """New-style autograd.Function compiled should work with torch.func.grad.

        This is the main bug from https://github.com/pytorch/pytorch/issues/174067.
        torch.compile wraps autograd.Function in ApplyTemplate which lacks
        setup_context, breaking torch.func compatibility.
        """

        class NewStyleOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. New-style Function that doubles x, saving x in
            # setup_context and doubling the gradient in backward.
            @staticmethod
            def forward(x):
                # [FUNCTION DOC] HELPER: Forward (no ctx): returns x * 2.
                return x * 2

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: setup_context: unpacks the single input and saves
                # it for backward.
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2.
                return grad_output * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies NewStyleOp to x.
            return NewStyleOp.apply(x)

        compiled_fn = torch.compile(fn, backend="eager")

        def loss_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the compiled function applied to x.
            return compiled_fn(x).sum()

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = torch.func.grad(loss_fn)(x)
        self.assertEqual(result, torch.tensor([2.0, 2.0]))

    def test_old_style_autograd_function_with_grad_compiled(self):
        # [FUNCTION DOC] Checks that an old-style Function used inside a function compiled with
        # the eager backend works under torch.func.grad (the compiled path uses traced graphs):
        # gradient equals a vector of twos.
        """Old-style autograd.Function compiled should work with torch.func.grad.

        Even though the original function is old-style, the compiled version
        (ApplyTemplate) can support torch.func transforms because it uses
        traced graphs.
        """

        class OldStyleOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Nested fixture. Old-style Function that saves x in forward and
            # doubles it; backward doubles the gradient.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Forward: saves x on ctx and returns x * 2.
                ctx.save_for_backward(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward: returns grad_output * 2.
                return grad_output * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies OldStyleOp to x.
            return OldStyleOp.apply(x)

        compiled_fn = torch.compile(fn, backend="eager")

        def loss_fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of the compiled function applied to x.
            return compiled_fn(x).sum()

        x = torch.tensor([1.0, 2.0], requires_grad=True)
        result = torch.func.grad(loss_fn)(x)
        self.assertEqual(result, torch.tensor([2.0, 2.0]))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
