# Owner(s): ["module: onnx"]
# [MODULE DOC] ONNX Runtime export tests that need a CUDA device (fp16, bfloat16, apex, mixed
# devices). One parameterized test class runs against the CUDA execution provider for every opset
# from the minimum to the maximum supported. Import-time effect: the class decorator generates one
# concrete class per opset version.

import unittest

import onnx_test_common
import onnxruntime
import parameterized
from onnx_test_common import MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION
from pytorch_test_common import (
    skipIfNoBFloat16Cuda,
    skipIfNoCuda,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)
from test_pytorch_onnx_onnxruntime import _parameterized_class_attrs_and_values

import torch
from torch.cuda.amp import autocast
from torch.testing._internal import common_utils


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestONNXRuntime_cuda(onnx_test_common._TestONNXRuntime):
    # [CLASS DOC] Opset-parameterized export tests using the CUDA ONNX Runtime provider. The class
    # attribute ort_backend selects the provider; the base class supplies run_test, which exports
    # a model and compares ONNX Runtime outputs with eager outputs.
    ort_backend = "CUDAExecutionProvider"

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class-level setup: runs the base setup, then raises a unittest skip
        # exception for the whole class when the CUDA provider is missing from the list of
        # providers reported by the installed onnxruntime.
        super().setUpClass()
        if cls.ort_backend not in onnxruntime.get_available_providers():
            raise unittest.SkipTest(f"onnxruntime {cls.ort_backend} is not available")

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    def test_gelu_fp16(self):
        # [FUNCTION DOC] Exports a gelu module on a float16 CUDA input and checks ONNX Runtime
        # results against eager with relaxed tolerances (rtol 1e-3, atol 1e-5). Needs opset 9+.
        class GeluModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module whose forward applies the functional gelu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the gelu activation of the input tensor.
                return torch.nn.functional.gelu(x)

        x = torch.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoCuda
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        # [FUNCTION DOC] Exports a LayerNorm module (autocast forward) on a float16 CUDA input of
        # shape 20x5x10x10 and verifies ONNX Runtime output with relaxed tolerances. Skipped for
        # script mode; needs opset 9+ and CUDA.
        class LayerNormModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a LayerNorm over the last two dims (10 x 10); its
            # forward runs under autocast.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the LayerNorm submodule normalizing over a 10
                # x 10 shape.
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer norm to the input, under autocast.
                return self.layer_norm(x)

        x = torch.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        self.run_test(LayerNormModel().cuda(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipIfNoCuda
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        # [FUNCTION DOC] Exports an NLLLoss(no reduction) over LogSoftmax of doubled input
        # under autocast, with float16 CUDA logits and random class targets in which entries
        # equal to class 1 are replaced by the default ignore index (-100). Verifies the fused
        # softmax cross entropy export through run_test. Skipped for script mode; needs opset
        # 12+ and CUDA.
        class FusionModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an unreduced NLL loss and a log-softmax over dim 1,
            # evaluated under autocast to exercise softmax cross entropy fusion.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the NLL loss (reduction none) and log-softmax
                # (dim 1) submodules.
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input, target):
                # [FUNCTION DOC] HELPER: Returns the per-element NLL loss of the log-softmax of
                # twice the input against the targets, under autocast.
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, dtype=torch.float16, device=torch.device("cuda"))
        target = torch.empty(N, dtype=torch.long, device=torch.device("cuda")).random_(
            0, C
        )

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(FusionModel(), (input, target))

    @skipIfNoCuda
    @skipScriptTest()
    def test_apex_o2(self):
        # [FUNCTION DOC] Exports a single linear layer (3 to 5) after wrapping it with apex amp
        # optimization level O2, on a CUDA input. If apex cannot be imported the test is skipped
        # with the import error chained. Skipped for script mode; needs CUDA.
        class LinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping one Linear layer from 3 to 5 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to the input.
                return self.linear(x)

        try:
            from apex import amp
        except Exception as e:
            raise unittest.SkipTest("Apex is not available") from e
        input = torch.randn(3, 3, device=torch.device("cuda"))
        model = amp.initialize(LinearModel(), opt_level="O2")
        self.run_test(model, input)

    # ONNX supports bfloat16 for opsets >= 13
    # Add, Sub and Mul ops don't support bfloat16 cpu in onnxruntime.
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16Cuda
    def test_arithmetic_bfp16(self):
        # [FUNCTION DOC] Exports a module doing bfloat16 add, sub and mul on a float16 CUDA input
        # cast to bfloat16, with the result cast back to float16, and verifies ONNX Runtime output
        # with relaxed tolerances. Needs opset 13+ and bfloat16 CUDA support.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module doing arithmetic in bfloat16 on CUDA.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds a ones tensor of shape 3x4 in bfloat16 on CUDA,
                # casts the input to its type, returns the product of the sum and the
                # difference, as float16.
                y = torch.ones(3, 4, dtype=torch.bfloat16, device=torch.device("cuda"))
                x = x.type_as(y)
                return torch.mul(torch.add(x, y), torch.sub(x, y)).to(
                    dtype=torch.float16
                )

        x = torch.ones(
            3, 4, requires_grad=True, dtype=torch.float16, device=torch.device("cuda")
        )
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    @skipIfNoCuda
    def test_deduplicate_initializers_diff_devices(self):
        # [FUNCTION DOC] Exports a model whose parameters live on different devices (one weight on
        # CPU, one bias on CUDA) with CPU and CUDA inputs, checking that initializer deduplication
        # handles parameters spanning devices. Needs CUDA.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 2x3 ones parameter on CPU and a length-3 ones
            # parameter on CUDA.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the CPU weight parameter and the
                # CUDA bias parameter.
                super().__init__()
                self.w = torch.nn.Parameter(
                    torch.ones(2, 3, device=torch.device("cpu"))
                )
                self.b = torch.nn.Parameter(torch.ones(3, device=torch.device("cuda")))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns a tuple: the matrix product of the weight and the
                # first input, and the second input plus the bias.
                return torch.matmul(self.w, x), y + self.b

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("cuda"))
        self.run_test(Model(), (x, y))


if __name__ == "__main__":
    common_utils.run_tests()
