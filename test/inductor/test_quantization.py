# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the activation quantization post-grad pass in inductor. Defines small
# modules (matmul after relu and tanh, a feedforward net, a module whose intermediate is both an
# output and saved for backward, and a layer norm wrapper) and a TestCase comparing compiled and
# eager predictions, parameters and gradients while checking the pass counters. The tests need a
# GPU; the runner only starts on Linux with a GPU.

import logging

import numpy as np

import torch
import torch._inductor
import torch._inductor.fx_passes.group_batch_fusion
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


log = logging.getLogger(__name__)


class TargetCPModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module computing relu then tanh of the first input and returning
    # its matrix product with the second input.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base initializer; no parameters.
        super().__init__()

    def forward(self, x1, x2):
        # [FUNCTION DOC] HELPER: Applies relu then tanh to x1 and returns the matrix product
        # with x2.
        relued = torch.relu(x1)
        tanhed = torch.tanh(relued)
        tensor = torch.matmul(
            tanhed,
            x2,
        )
        return tensor


class FeedforwardNN(torch.nn.Module):
    # [CLASS DOC] HELPER: Feedforward network of four linear layers (1 to 64, 64 to 64, 64 to
    # 64, 64 to 1) with relu activations and an extra tanh branch feeding the third layer.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the four linear layers.
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs relu of the first layer, takes a tanh of that result, then
        # relu of the second layer; the third layer is applied to the tanh output (not the
        # second layer output), followed by the last layer, and the result is returned.
        x = torch.relu(self.fc1(x))
        tanh_x = torch.tanh(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(tanh_x))
        x = self.fc4(x)
        return x


class SharedOutputAndSavedModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose 3D intermediate is both returned as a user output and
    # saved for backward, used to reproduce a bug where the quantization pass handled both
    # positions of the same tensor and shifted stride mapping. Holds an 8 by 8 scale and a 64 by
    # 32 weight.
    """A module where a 3D intermediate is both a user output and saved for backward.

    This triggers the bug in T264303372: the activation quantization pass would
    quantize both the user output and saved-for-backward positions of the same
    tensor, creating duplicate backward placeholders that shift the stride mapping.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the scale parameter of ones and the random weight
        # parameter.
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(8, 8))
        self.W = torch.nn.Parameter(torch.randn(64, 32))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Reshapes the input to batch by 8 by 8, forms a gated
        # activation, scales it, flattens and multiplies by the weight; returns the matmul
        # result and the gated activation.
        h = x.view(x.shape[0], 8, 8)
        attn = h * torch.sigmoid(h)
        # attn is both returned (user output) and needed by backward (mul saves it)
        scaled = attn * self.scale
        flat = scaled.flatten(1)
        result = flat @ self.W
        return result, attn


class LayernormNN(torch.nn.Module):
    # [CLASS DOC] HELPER: Wrapper module around functional layer norm with a fixed epsilon.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base initializer; no parameters.
        super().__init__()

    def forward(self, input, normalized_shape, weight, bias):
        # [FUNCTION DOC] HELPER: Applies layer norm to the input with the given normalized
        # shape, weight and bias and an epsilon of one thousandth of a percent, and returns it.
        x = torch.nn.functional.layer_norm(
            input=input,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            eps=1e-5,
        )
        return x


class TestQuantization(TestCase):
    # [CLASS DOC] GPU tests that compile modules with the activation quantization pass enabled
    # and compare against eager. Provides comparison helpers for predictions, parameters and
    # gradients.
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Compares two mappings from names to tensors. Returns False if
        # sizes differ or a tensor mismatches within tolerance (logs the gradient mismatch);
        # entries are looked up in the second mapping under a name prefixed by the compile
        # wrapper prefix, and a missing key raises AssertionError. Pairs where both entries are
        # None are skipped. Returns True otherwise.
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            if key2 not in res_dict:
                raise AssertionError(f"{key1} does not exist in traced module")
            # if both of them are None, continue
            if (
                not isinstance(ref_dict[key1], torch.Tensor)
                and not isinstance(res_dict[key2], torch.Tensor)
                and ref_dict[key1] is None
                and res_dict[key2] is None
            ):
                log.info(
                    "None found with key1 and value 1: %s, %s, key2 and value2 %s, %s",
                    key1,
                    ref_dict[key1],
                    key2,
                    res_dict[key2],
                )
                continue
            elif not torch.allclose(
                ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol, equal_nan=True
            ):
                log.info(
                    "gradient mismatch for eager and compiled modules, with eager: %s and compiled: %s",
                    ref_dict[key1],
                    res_dict[key2],
                )
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Runs the eager module and the compiled one on the same inputs
        # and asserts the outputs are equal within the given tolerances.
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Asserts that eager and compiled parameter mappings match, using
        # the dictionary comparison helper with the given tolerances.
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Collects gradients of all parameters of both modules and
        # asserts they match through the dictionary comparison helper.
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {
                "quant_type": "torch.float8_e5m2",
                "use_scaling": True,
                "size_in_mb": 0.0,
                "exclude_primals": True,
                "allowed_dtypes": "torch.bfloat16;torch.float32",
            },
        },
    )
    def test_activation_quantization_aten_with_scaling(self):
        # [FUNCTION DOC] With scaled float8 activation quantization enabled for all sizes,
        # compiles the matmul module and then the feedforward net on GPU, comparing predictions,
        # parameters and gradients with eager and asserting the forward and backward
        # quantization pass counters each equal one for each module (counters cleared in
        # between).
        counters.clear()
        module = TargetCPModule().to(GPU_TYPE)
        input = [
            torch.rand(
                (16, 10), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            torch.rand(
                (10, 16), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

        module = FeedforwardNN().to(GPU_TYPE)
        X = np.linspace(-10, 10, 100).reshape(-1, 1).astype(np.float32)
        input = [
            torch.from_numpy(X).to(GPU_TYPE),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {
                "quant_type": "torch.float8_e5m2",
                "use_scaling": False,
                "size_in_mb": 0.0,
                "exclude_primals": True,
                "allowed_dtypes": "torch.bfloat16;torch.float32",
            },
        },
    )
    def test_activation_quantization_aten_without_scaling(self):
        # [FUNCTION DOC] With unscaled float8 quantization enabled, compiles the layer norm
        # module on bfloat16 GPU inputs, compares predictions, parameters and gradients with
        # eager, and asserts the forward and backward pass counters each equal one.
        counters.clear()

        module = LayernormNN().to(GPU_TYPE)
        normalized_shape = [256]
        input = [
            torch.randn(
                (1, 3, 256), requires_grad=True, device=GPU_TYPE, dtype=torch.bfloat16
            ),
            normalized_shape,
            torch.randn(
                *normalized_shape,
                requires_grad=True,
                device=GPU_TYPE,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                *normalized_shape,
                requires_grad=True,
                device=GPU_TYPE,
                dtype=torch.bfloat16,
            ),
        ]
        traced = torch.compile(module)
        ref = module(*input)
        res = traced(*input)
        self.compare_pred(module, traced, input)
        ref.sum().backward()
        res.sum().backward()
        self.compare_parameters(module, traced)
        self.compare_gradients(module, traced)
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        self.assertTrue(torch.allclose(ref, res))
        counters.clear()

    @requires_gpu()
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "activation_quantization_aten_pass": {
                "quant_type": "torch.float8_e5m2",
                "use_scaling": True,
                "size_in_mb": 0.0,
                "exclude_primals": True,
                "allowed_dtypes": "torch.bfloat16;torch.float32",
            },
        },
    )
    def test_activation_quantization_shared_output_and_saved(self):
        # [FUNCTION DOC] With scaled quantization enabled, compiles the shared output and saved
        # module, runs forward and backward on a combined loss over both outputs, and asserts
        # the forward and backward pass counters each equal one (regression test for a duplicate
        # placeholder bug).
        """Test that activation quantization works when a tensor is both a user
        output and a saved-for-backward activation (T264303372)."""
        counters.clear()
        module = SharedOutputAndSavedModule().to(GPU_TYPE)
        x = torch.randn(8, 64, device=GPU_TYPE, requires_grad=True)
        compiled = torch.compile(module)
        result, attn = compiled(x)
        loss = result.sum() + attn.sum()
        loss.backward()
        self.assertEqual(
            counters["inductor"]["activation_quantization_fwd_aten_pass"], 1
        )
        self.assertEqual(
            counters["inductor"]["activation_quantization_bwd_aten_pass"], 1
        )
        counters.clear()


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
