# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests that onnxscript functions registered as custom symbolics for the TorchScript
# ONNX exporter end up as local functions in the exported model proto. Needs the onnx and
# onnxscript packages but no runtime. The main guard raises an error instead of running tests
# because the file is not enabled in test discovery.

"""Test the support on onnxscript in PyTorch-ONNX converter."""

import io

import onnx

import onnxscript
from onnxscript.onnx_types import FLOAT

import torch
from torch.onnx._internal.torchscript_exporter import jit_utils
from torch.testing._internal import common_utils


class TestONNXScriptExport(common_utils.TestCase):
    # [CLASS DOC] Test case (plain common_utils test case) with a class attribute opset_version
    # fixed at 15, the first opset allowing local functions. Tests register custom aten symbolics
    # backed by onnxscript functions (global registration side effect) and inspect exported protos.
    # opset version is
    # 1. local function is supported after opset 15
    # 2. onnx-script requires users to determine opset in local function
    opset_version = 15

    def test_onnxscript_registration_with_multiple_models(self):
        # [FUNCTION DOC] Registers an onnxscript Selu for the aten selu op and an onnxscript layer
        # norm for the aten layer_norm op (both at the class opset), exports a SELU module and a
        # LayerNorm module to separate buffers, reloads them, and asserts each proto has exactly
        # one local function, named Selu and layer_norm respectively.
        from onnxscript.onnx_opset import opset15 as op

        # 1. Register Selu onnxscript function as custom Op
        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

        @onnxscript.script(custom_opset)
        def Selu(X):
            # [FUNCTION DOC] HELPER: onnxscript-compiled Selu graph function in a custom opset
            # domain. Computes gamma times the ELU-like value for non-positive inputs (alpha times
            # exp minus alpha) and gamma times X otherwise, casting constants to the type of X.
            # default value is not supported by onnxscript
            alpha = 1.67326  # auto wrapped as Constants
            gamma = 1.0507
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g: jit_utils.GraphContext, X):
            # [FUNCTION DOC] HELPER: Symbolic for selu. Adds a call to the Selu onnxscript function
            # on the graph and copies the input type onto the result so type inference works.
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=self.opset_version,
        )

        # 2. Register layer_norm onnxscript function as custom Op
        @onnxscript.script(custom_opset)
        def layer_norm(
            X, axes: list[int], weight: FLOAT[...], bias: FLOAT[...], eps: float
        ):
            # [FUNCTION DOC] HELPER: onnxscript-compiled layer normalization over the given axes.
            # Takes the input, an axes list, weight, bias and an epsilon; computes mean, variance
            # plus epsilon, reciprocal square root normalization, casts to weight type, scales by
            # weight and adds bias.
            mean = op.ReduceMean(X, axes=axes)
            D = X - mean  # op.Sub(X, mean)
            DD = D * D  # op.Mul(D, D)
            var = op.ReduceMean(DD, axes=axes)
            vareps = var + eps  # op.Add(var, eps)
            stddev = op.Sqrt(vareps)
            invstddev = op.Reciprocal(stddev)
            normalized = D * invstddev  # op.Mul(D, invstddev)
            normalizedw = op.CastLike(
                normalized, weight
            )  # Type issue if missing this Op
            normalizedscaled = normalizedw * weight  # op.Mul(normalized, weight)
            return normalizedscaled + bias

        @torch.onnx.symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
        def custom_layer_norm(
            g, input, normalized_shape, weight, bias, eps, cudnn_enable
        ):
            # [FUNCTION DOC] HELPER: Symbolic for layer_norm (argument parsing decorator applied).
            # Builds the axes list as negative dimensions covering normalized_shape, then adds a
            # call to the layer_norm onnxscript function with weight, bias, axes and eps and copies
            # the input type.
            # comprehension is not supported by onnxscript
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
            return g.onnxscript_op(
                layer_norm, input, weight, bias, axes_i=axes, eps_f=eps
            ).setType(input.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::layer_norm",
            symbolic_fn=custom_layer_norm,
            opset_version=self.opset_version,
        )

        # 3. export two models
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model_selu = torch.nn.SELU()
        selu_onnx = io.BytesIO()
        torch.onnx.export(
            model_selu, x, selu_onnx, opset_version=self.opset_version, dynamo=False
        )

        N, C = 3, 4
        y = torch.randn(N, C)
        model_layer_norm = torch.nn.LayerNorm(C)
        layer_norm_onnx = io.BytesIO()
        torch.onnx.export(
            model_layer_norm,
            y,
            layer_norm_onnx,
            opset_version=self.opset_version,
            dynamo=False,
        )

        # 4. test on models
        selu_proto = onnx.load(io.BytesIO(selu_onnx.getvalue()))
        layer_norm_proto = onnx.load(io.BytesIO(layer_norm_onnx.getvalue()))

        self.assertEqual(len(selu_proto.functions), 1)
        self.assertEqual(len(layer_norm_proto.functions), 1)
        self.assertEqual(selu_proto.functions[0].name, "Selu")
        self.assertEqual(layer_norm_proto.functions[0].name, "layer_norm")

    def test_loop_registration(self):
        # [FUNCTION DOC] Checks that onnxscript custom ops inside nested control flow are found. A
        # scripted module with a loop and a branch calling selu is exported at opset 15 with a
        # custom Selu symbolic (which also prints a message when invoked); asserts
        # one local function.
        # Control flow is tested for _find_onnxscript_op function in torch/onnx/utils.py,
        # which has recursive logic to go through every nodes with subgraph in model proto
        class NestedLoopsModel(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Scripted module holding a SELU layer; its forward loops over the
            # size of dim 3, applying selu on iteration zero and adding the loop index to the
            # running value otherwise.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the script module and create selu.
                super().__init__()
                self.selu = torch.nn.SELU()

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return y, which starts as x, is replaced by selu of x at
                # the first loop iteration and increased in place by the iteration index on
                # later ones.
                y = x
                for i in range(x.size(3)):
                    if i == 0:
                        y = self.selu(x)
                    else:
                        y += i
                return y

        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, 4)

        from onnxscript.onnx_opset import opset15 as op

        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=2)

        @onnxscript.script(custom_opset)
        def Selu(X):
            # [FUNCTION DOC] HELPER: onnxscript Selu graph function (domain onnx-script, version 2)
            # using full-precision alpha and gamma constants cast to the type of X.
            alpha = 1.6732632423543772848170429916717
            gamma = 1.0507009873554804934193349852946
            alphaX = op.CastLike(alpha, X)
            gammaX = op.CastLike(gamma, X)
            neg = gammaX * (alphaX * op.Exp(X) - alphaX)
            pos = gammaX * X
            zero = op.CastLike(0, X)
            return op.Where(X <= zero, neg, pos)

        def custom_selu(g, X):
            # [FUNCTION DOC] HELPER: Symbolic for selu. Prints a marker line, adds a call to the
            # Selu onnxscript function and copies the input type onto the result for
            # shape inference.
            # domain of the Op should be aligned with onnx-script
            # setType API is required for custom Op to support
            # torchscript shape type inference
            print("custom_selu is used!")
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=15,
        )

        saved_model = io.BytesIO()
        torch.onnx.export(
            torch.jit.script(model),
            inputs,
            f=saved_model,
            opset_version=15,
            dynamo=False,
        )
        loop_selu_proto = onnx.load(io.BytesIO(saved_model.getvalue()))
        self.assertEqual(len(loop_selu_proto.functions), 1)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
