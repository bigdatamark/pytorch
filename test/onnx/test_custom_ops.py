# Owner(s): ["module: onnx"]
# [MODULE DOC] ONNX export tests for custom autograd functions (with a symbolic method or a
# registered custom symbolic for the Python op node) and for exporting a scripted loop model that
# uses a contrib-op symbolic for GELU. Results are checked via onnx_test_common.run_model_test.
# Runs the common test runner when executed as a script.

import onnx_test_common
import pytorch_test_common

import torch
import torch.utils.cpp_extension
from torch.onnx import symbolic_helper
from torch.testing._internal import common_utils


class TestCustomAutogradFunction(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export tests for custom torch.autograd.Function subclasses. Class attributes
    # select opset 9, no initializers as inputs, and ONNX shape inference enabled, which the
    # shared test base/helper reads.
    opset_version = 9
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_symbolic(self):
        # [FUNCTION DOC] Exports a module that applies a custom clip autograd function defining
        # its own symbolic static method (emitting an ONNX Clip node with the scalar as the min
        # attribute) and runs the standard model comparison on a random 2x3x4 input.
        class MyClip(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function clamping the input from below by a scalar,
            # with a symbolic method that emits an ONNX Clip op.
            @staticmethod
            def forward(ctx, input, scalar):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns the input
                # clamped to a minimum of scalar.
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

            @staticmethod
                # [FUNCTION DOC] HELPER: Export-time symbolic: emits an ONNX Clip node on the
                # input with the scalar passed as the float min attribute.
            def symbolic(g, input, scalar):
                return g.op("Clip", input, min_f=scalar)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the apply function of the clip autograd function
            # and applying it with a minimum of 2.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the clip function's apply callable as an
                # attribute.
                super().__init__()
                self.clip = MyClip.apply

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the clip function with minimum 2 and returns the
                # result.
                h = self.clip(x, 2)
                return h

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))

    def test_register_op(self):
        # [FUNCTION DOC] Exports a model using two autograd functions without their own symbolic
        # methods, after registering a custom symbolic handler for the generic Python op node at
        # opset 1 (global registration side effect, never unregistered here). Then runs the
        # standard model comparison.
        class MyClip(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function clamping the input from below by a scalar; has
            # no symbolic method.
            @staticmethod
            def forward(ctx, input, scalar):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns the input
                # clamped to a minimum of scalar.
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function implementing relu via clamping at zero; has no
            # symbolic method.
            @staticmethod
            def forward(ctx, input):
                # [FUNCTION DOC] HELPER: Saves the input for backward and returns the input
                # clamped to a minimum of zero.
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that applies the clip function (minimum 2) followed by
            # the relu function.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the apply callables of the clip and relu autograd
                # functions as attributes.
                super().__init__()
                self.clip = MyClip.apply
                self.relu = MyRelu.apply

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the clip function with minimum 2, then the relu
                # function, and returns the result.
                h = self.clip(x, 2)
                h = self.relu(h)
                return h

        def symbolic_pythonop(g, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Custom symbolic for the Python op node. Dispatches on the op
            # name keyword argument: MyClip yields an ONNX Clip node with the second argument as
            # min attribute, MyRelu yields an ONNX Relu node; any other name reports an
            # unimplemented-symbolic error through symbolic_helper.
            name = kwargs["name"]
            if name == "MyClip":
                return g.op("Clip", args[0], min_f=args[1])
            elif name == "MyRelu":
                return g.op("Relu", args[0])
            else:
                return symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        from torch.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))


class TestExportAsContribOps(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export tests for contrib (non-standard domain) ops. Class attributes select
    # opset 14, no initializers as inputs, and shape inference enabled.
    opset_version = 14
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_contrib_op_with_loop(self):
        # [FUNCTION DOC] Scripts a module containing a loop and a conditional that uses GELU,
        # registers a custom symbolic mapping GELU to the com.microsoft Gelu contrib op (global
        # registration at opset 1), and runs the standard model comparison on a random 3x3x4
        # input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a GELU layer; its forward loops over the first
            # dimension building two lists and returns two stacked tensors.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the exact (non-approximate) GELU layer.
                super().__init__()
                self.gelu = torch.nn.GELU(approximate="none")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: For each row of the input, appends to the second list
                # either the first saved row (when the first list is non-empty) or the GELU of the
                # first row, appends the first row to the first list, and returns both lists
                # stacked into tensors.
                res = []
                res2 = []
                for _ in range(x.size(0)):
                    if len(res) > 0:
                        res2.append(res[0])
                    else:
                        res2.append(self.gelu(x[0]))
                    res.append(x[0])
                return torch.stack(res), torch.stack(res2)

        def symbolic_custom_gelu(g, input, approximate):
            # [FUNCTION DOC] HELPER: Custom symbolic emitting a com.microsoft Gelu node on the
            # input and copying the input's type onto the result; the approximate argument is
            # ignored.
            return g.op("com.microsoft::Gelu", input).setType(input.type())

        from torch.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("::gelu", symbolic_custom_gelu, 1)

        x = torch.randn(3, 3, 4, requires_grad=True)
        model = torch.jit.script(M())
        onnx_test_common.run_model_test(self, model, input_args=(x,))


if __name__ == "__main__":
    common_utils.run_tests()
