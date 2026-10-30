# Owner(s): ["module: onnx"]
# [MODULE DOC] Runtime tests for onnxscript functions used as custom symbolics in the TorchScript
# ONNX exporter: the exported model is run and compared with PyTorch through the shared ONNX
# runtime test base class. Contains one test class with two tests.

"""Test the support on onnxscript in PyTorch-ONNX converter with onnxruntime."""

from typing import Sequence  # noqa: UP035

import onnx_test_common
import onnxscript
from onnxscript.onnx_types import FLOAT

import torch
from torch.onnx._internal.torchscript_exporter import jit_utils
from torch.testing._internal import common_utils


class TestONNXScriptRuntime(onnx_test_common._TestONNXRuntime):
    # [CLASS DOC] Runtime test case derived from the shared ONNX runtime base; opset_version is
    # fixed at 15 (local functions need at least that). Tests register onnxscript-backed custom
    # symbolics (global registration side effect) and verify via run_test.
    # opset version is
    # 1. local function is supported after opset 15
    # 2. onnx-script requires users to determine opset in local function
    opset_version = 15

    def test_selu_from_onnxscript_example(self):
        # [FUNCTION DOC] Defines an onnxscript Selu function, registers it as the symbolic for the
        # aten selu op at the class opset, and verifies a SELU module on a random 4-D
        # input via run_test.
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = torch.nn.SELU()

        from onnxscript.onnx_opset import opset15 as op

        custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

        @onnxscript.script(custom_opset)
        def Selu(
            X,
        ):
            # [FUNCTION DOC] HELPER: onnxscript-compiled Selu graph function: gamma times the
            # scaled exponential branch for non-positive inputs and gamma times X otherwise, with
            # constants cast to the type of X.
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
            # [FUNCTION DOC] HELPER: Symbolic for selu that adds a call to the Selu onnxscript
            # function and copies the input type onto the result.
            return g.onnxscript_op(Selu, X).setType(X.type())

        torch.onnx.register_custom_op_symbolic(
            symbolic_name="aten::selu",
            symbolic_fn=custom_selu,
            opset_version=self.opset_version,
        )
        self.run_test(model, x)

    def test_layer_norm(self):
        # [FUNCTION DOC] Registers an onnxscript layer norm for the aten layer_norm op and verifies
        # a composite module (two CELU activations, three LayerNorm layers applied in sequence,
        # then a dropout submodule) on three random 2x3 inputs via run_test.
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)

        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a dropout layer with the given probability.
            def __init__(self, prob):
                # [FUNCTION DOC] HELPER: Create the dropout layer with probability prob.
                super().__init__()
                self.dropout = torch.nn.Dropout(prob)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x after dropout.
                return self.dropout(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two CELU activations (alpha 1 and 2), a list of
            # num_layers LayerNorm layers (eps equal to the layer index) and a dropout submodule of
            # probability 0.5.
            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Build the layers and store num_layers.
                super().__init__()
                self.num_layers = num_layers
                self.lns = torch.nn.ModuleList(
                    [torch.nn.LayerNorm(3, eps=i) for i in range(num_layers)]
                )
                self.celu1 = torch.nn.CELU(1.0)
                self.celu2 = torch.nn.CELU(2.0)
                self.dropout = N(0.5)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Apply the CELU layers to x and y, run z through all
                # LayerNorm layers in order, and return the sum of the two CELU results plus the
                # dropout applied to z.
                res1 = self.celu1(x)
                res2 = self.celu2(y)
                for ln in self.lns:
                    z = ln(z)
                return res1 + res2, self.dropout(z)

        model = M(3)

        from onnxscript.onnx_opset import opset15 as op

        custom_opset = onnxscript.values.Opset(domain="onnxscript", version=1)

        @onnxscript.script(custom_opset)
        def layer_norm(
            X,
            axes: Sequence[int],
            weight: FLOAT[...],
            bias: FLOAT[...],
            eps: float,
        ):
            # [FUNCTION DOC] HELPER: onnxscript-compiled layer normalization over given axes with
            # weight, bias and epsilon: normalizes by mean and variance, casts to weight type,
            # scales and shifts.
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
            # Builds negative axes covering normalized_shape and adds a call to the layer_norm
            # onnxscript function, copying the input type onto the result.
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

        self.run_test(model, (x, y, z))


if __name__ == "__main__":
    common_utils.run_tests()
