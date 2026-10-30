# Owner(s): ["module: onnx"]
# [MODULE DOC] ONNX export tests for modules that call custom torch.autograd.Function subclasses
# (single, multiple, partially used, nested, unsupported-ATen, and symbolic/inline variants). Each
# test builds a small caller module and checks it through the shared run_model_test helper at
# opset 20. Runs the common test runner when executed as a script.

import pytorch_test_common
from onnx_test_common import run_model_test

import torch
from torch.onnx import OperatorExportTypes
from torch.testing._internal import common_utils


class TestAutogradFuns(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export tests for autograd functions. Class attributes select opset 20, no
    # initializers as graph inputs, and ONNX shape inference enabled; these are read by the shared
    # test base and helper.
    opset_version = 20
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_single_output(self):
        # [FUNCTION DOC] Exports a module that wraps an exp-then-log autograd function between two
        # scalar arithmetic ops, on a one-element ones tensor, and compares with run_model_test.
        class SingleOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with one output computing log of exp of the
            # input, saving that result for backward.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Computes exp then log of the input, saves the result for
                # backward, and returns it.
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Unpacks the single saved tensor and returns the incoming
                # gradient multiplied by it (not a mathematically exact gradient; only graph
                # export is exercised).
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies the custom function(s) to the input
            # and returns the result.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Adds 5 to the input, applies the single-output function,
                # adds 3, and returns the result.
                result = input + 5
                return SingleOut.apply(result) + 3

        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_multi_output(self):
        # [FUNCTION DOC] Exports a module calling an autograd function that returns two tensors,
        # on a ones tensor of shape one by five, and compares with run_model_test.
        class MultiOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function returning a pair (exp of the input and log of
            # that).
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Computes exp and then log of that, saves both for
                # backward, and returns both.
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward taking one gradient.
                # KNOWN ISSUE: it unpacks a single saved tensor although two were saved and the
                #   forward has two outputs, so it would fail if actually executed; export tests
                #   never run it.
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies the custom function(s) to the input
            # and returns the result.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the output pair of the two-output function
                # applied to the input.
                return MultiOut.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_partial_output(self):
        # [FUNCTION DOC] Exports a module whose autograd function computes a top-3 of the input
        # and returns only the values, discarding the indices; compares with run_model_test on a
        # one-by-five ones tensor.
        class PartialOut(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function returning only the top-3 values of its input.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Saves the input, computes the top three values along the
                # default dimension, and returns the values only.
                ctx.save_for_backward(input)
                values, _ = torch.topk(input, 3)
                return values

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies the custom function(s) to the input
            # and returns the result.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the partial-output function applied to the input.
                return PartialOut.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_nested_autograd(self):
        # [FUNCTION DOC] Exports a module calling an autograd function whose forward calls another
        # autograd function, checking the nested case through run_model_test on a one-by-five ones
        # tensor.
        class Child(torch.autograd.Function):
            # [CLASS DOC] HELPER: Inner autograd function computing log of log of the input and
            # saving it for backward.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Applies log twice, saves the final result for backward,
                # and returns it.
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Unpacks the single saved tensor and returns the gradient
                # multiplied by it.
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Parent(torch.autograd.Function):
            # [CLASS DOC] HELPER: Outer autograd function that returns exp of the input and the
            # result of applying the inner function to that.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Computes exp of the input, applies the inner function to
                # it, saves both, and returns both.
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return result_exp, result_log

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward unpacking one saved tensor.
                # KNOWN ISSUE: two tensors were saved in forward, so unpacking would fail if
                #   executed; export tests never run it.
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies the custom function(s) to the input
            # and returns the result.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the outer function applied to the input.
                return Parent.apply(input)

        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    # Run export in ONNX_FALLTHROUGH mode as torch.erf() is not supported
    def test_aten_unsupported(self):
        # [FUNCTION DOC] Builds the graph for a module using an erf autograd function through the
        # internal model-to-graph utility twice: with the fallthrough export type the first node
        # must be a Python op node, and with the ATen fallback export type the first node must be
        # an ATen node. Asserts on the kind string of the first graph node in each mode; no model
        # file is written.
        class Erf(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function applying the special erf op and saving the
            # output.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes the special erf of the input, saves the output
                # for backward, and returns it.
                erf_out = torch.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Backward returning the inverse error function of the
                # saved tensors and None.
                # KNOWN ISSUE: the saved tensors tuple is not unpacked, so it would fail if
                #   executed; export tests never run it.
                result = ctx.saved_tensors
                return torch.special.erfinv(result), None

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward applies the custom function(s) to the input
            # and returns the result.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the erf function applied to the input.
                return Erf.apply(input)

        model = Caller()
        input = torch.ones(1, 5)

        # Test ONNX_FALLTHROUGH_MODE
        graph, _, _ = torch.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "prim::PythonOp")

        # Test ATEN_FALLBACK_MODE
        graph, _, _ = torch.onnx.utils._model_to_graph(
            model,
            (input,),
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), "aten::ATen")

    def test_inline_and_symbolic(self):
        # [FUNCTION DOC] Exports a module where one autograd function has a symbolic method
        # (emitting an ONNX Exp node) and another has none (so it is inlined), and compares with
        # run_model_test on a one-element ones tensor.
        class Exp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing exp, with a symbolic method that
            # emits an ONNX Exp node.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Returns exp of the input.
                # KNOWN ISSUE: saves for backward the name input, which refers to the enclosing
                #   test's local tensor (a closure), not the forward argument.
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                # [FUNCTION DOC] HELPER: Export-time symbolic emitting an ONNX Exp node on the
                # input.
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing log of log, with no symbolic method
            # so it is inlined during export.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Returns log of log of the argument.
                # KNOWN ISSUE: saves for backward the enclosing test's local tensor via closure
                #   rather than the forward argument.
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the exp function and then the log-log function.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the exp function to the input and then the
                # log-log function to that, returning the result.
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_inline_with_scoped_tracing(self):
        # [FUNCTION DOC] Same as the inline-and-symbolic test but also sets the global trace
        # module map in torch.jit._trace to map every submodule to its type name before running
        # run_model_test (emulating scoped tracing), then resets the map to None afterwards. Not
        # restored if the helper raises.
        class Exp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing exp, with a symbolic method that
            # emits an ONNX Exp node.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Returns exp of the input.
                # KNOWN ISSUE: saves for backward the enclosing test's local tensor via closure
                #   rather than the forward argument.
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                # [FUNCTION DOC] HELPER: Export-time symbolic emitting an ONNX Exp node on the
                # input.
                return g.op("Exp", input)

        class LogLog(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function computing log of log with no symbolic method.
            @staticmethod
            def forward(ctx, i):
                # [FUNCTION DOC] HELPER: Returns log of log of the argument.
                # KNOWN ISSUE: saves for backward the enclosing test's local tensor via closure
                #   rather than the forward argument.
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the exp function and then the log-log function.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Applies the exp function and then the log-log function to
                # the input and returns the result.
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)

        model = Caller()
        input = torch.ones(1)

        torch.jit._trace._trace_module_map = {
            _m: torch.typename(type(_m)) for _m in model.modules()
        }
        run_model_test(self, model, input_args=(input,))
        torch.jit._trace._trace_module_map = None


if __name__ == "__main__":
    common_utils.run_tests()
