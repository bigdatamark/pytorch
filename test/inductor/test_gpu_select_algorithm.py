# Owner(s): ["module: inductor"]
# [MODULE DOC] GPU (CUDA and XPU) tests for inductor algorithm selection with int8 weight-only
# quantized matmul patterns. Reuses the CPU select-algorithm test base class and the generic model
# checker by importing the sibling test modules (relative first, then absolute); if they raise a
# skip at import, exits cleanly when run as a script and re-raises otherwise. Defines a decorator
# that forces max-autotune style config and a fake autotune cache. Import time: instantiates
# device-generic test classes for cuda and xpu only.
import functools
import sys
import unittest
from unittest.mock import patch

import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_per_channel_qparams,
)
from torch.testing._internal.common_utils import (
    parametrize,
    TEST_CUDA,
    TEST_WITH_SLOW_GRADCHECK,
    TEST_XPU,
)


try:
    try:
        from . import test_cpu_select_algorithm, test_torchinductor
    except ImportError:
        import test_cpu_select_algorithm  # @manual=fbcode//caffe2/test/inductor:test_cpu_select_algorithm-library
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

check_model = test_torchinductor.check_model
BaseTestSelectAlgorithm = test_cpu_select_algorithm.BaseTestSelectAlgorithm


def patches(fn):
    # [FUNCTION DOC] HELPER: Decorator factory used on tests. Wraps the test function with
    # patchers for verbose dynamo config, inductor config (debug, max_autotune and epilogue_fusion
    # on), a tightened autotune verification tolerance (absolute and relative 1e-4) and a
    # replacement of the algorithm selector cache lookup with a fake that never reads a cache. The
    # returned wrapper clears the dynamo counters and seeds the torch RNG with 12345 before each
    # call, then delegates to the patched test.
    def skip_cache(self, choices, name, key, benchmark, hint_override=None):
        # [FUNCTION DOC] HELPER: Replacement for the selector cache lookup. Returns an empty
        # mapping when no benchmark function is supplied; otherwise runs the benchmark over the
        # choices to get a mapping of choice to timing and multiplies the timing of every extern
        # kernel choice by 1000 so that generated template choices win. Returns the timings.
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        for choice, timing in timings.items():
            if isinstance(choice, select_algorithm.ExternKernelCaller):
                timings[choice] = timing * 1000
        return timings

    for patcher in [
        dynamo_config.patch(verbose=True),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper that clears the global dynamo counters, reseeds the torch
        # RNG with a fixed seed, and calls the patched test function with the received arguments,
        # returning its result.
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


class TestSelectAlgorithmGpu(BaseTestSelectAlgorithm):
    # [CLASS DOC] Device-generic test class (instantiated for cuda and xpu) derived from the CPU
    # algorithm selection base class. The class attribute common is the shared model-checking
    # helper from the inductor torchinductor tests. Holds the int8 weight-only quantization
    # matcher tests.
    common = check_model

    @inductor_config.patch(
        {
            "freezing": True,
            "shape_padding": False,
        }
    )
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1, 17, 32))
    @parametrize("mid_dim", (1, 8))
    @parametrize("in_features", (128, 144, 1024))
    @parametrize("out_features", (64, 65, 1024))
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not available")
    @unittest.skipIf(TEST_WITH_SLOW_GRADCHECK, "Leaking memory")
    def test_int8_woq_mm_gpu(
        self, device, dtype, batch_size, mid_dim, in_features, out_features
    ):
        # [FUNCTION DOC] Builds a bfloat16 module computing a linear layer with an int8-packed
        # weight converted back to the input dtype, multiplied by a per-channel scale, and
        # compiles/checks it with freezing on and shape padding off under the autotune patches.
        # Inputs are random three-dimensional activations (batch, mid dim, in features). Compares
        # to eager through the shared checker and asserts that the weight-only-quant pattern
        # matcher counter equals one. Skipped without CUDA or XPU and under slow gradcheck.
        # Parametrized over batch size, mid dim, in and out features.
        def _convert_weight_to_int8pack(w):
            # [FUNCTION DOC] HELPER: Quantizes a floating weight per output channel to int8. Moves
            # the weight to CPU to compute dynamic per-channel scale and zero point (signed int8),
            # moves them back to the weight device, applies the decomposed per-channel quantize op
            # along axis zero with range minus 128 to 127. Returns the int8 weight and the scale
            # cast to bfloat16.
            # Move to CPU for quantization calculation, then back to original device
            device = w.device
            w_cpu = w.cpu()
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w_cpu.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale).to(device)
            zp = torch.from_numpy(zp).to(device)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module holding one frozen (no gradient) parameter wrapping
            # the int8 weight.
            def __init__(self, w):
                # [FUNCTION DOC] HELPER: Stores the given int8 weight as a parameter that does not
                # require gradients.
                super().__init__()
                self.linear_weight = torch.nn.Parameter(w, requires_grad=False)

            def forward(self, x, scale):
                # [FUNCTION DOC] HELPER: Casts the stored weight to the input dtype, applies a
                # linear layer to the input, and multiplies the result by the given scale; returns
                # the product.
                return (
                    torch.nn.functional.linear(x, self.linear_weight.to(x.dtype))
                    * scale
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype, device=device)
        w = torch.rand((out_features, in_features), dtype=dtype, device=device)
        w_int8pack, w_scales = _convert_weight_to_int8pack(w)
        w_scales = w_scales.to(device)
        mod = M(w_int8pack).eval()
        self.common(mod, (x, w_scales))
        self.assertEqual(counters["inductor"]["woq_matcher_count"], 1)

    @inductor_config.patch({"freezing": True, "cpp.enable_concat_linear": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (1, 32))
    @parametrize("mid_dim", (1, 8))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64,))
    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not available")
    @unittest.skipIf(TEST_WITH_SLOW_GRADCHECK, "Leaking memory")
    def test_int8_woq_mm_concat_gpu(
        self, device, dtype, batch_size, mid_dim, in_features, out_features
    ):
        # [FUNCTION DOC] Same idea as the single-weight test but with three int8 weights sharing
        # the same input, under freezing and the CPP concat-linear option (patched through
        # inductor config). Builds three matmul branches each scaled by its own per-channel scale,
        # compiles and checks against eager via the shared checker, and asserts that the weight-
        # only-quant pattern matcher counter equals three. Skipped without CUDA or XPU and under
        # slow gradcheck. Parametrized over batch size and mid dim, with fixed feature sizes.
        def _convert_weight_to_int8pack(w):
            # [FUNCTION DOC] HELPER: Same as the helper in the previous test: quantizes a weight
            # per output channel to int8 using CPU-computed dynamic scale and zero point, and
            # returns the int8 weight with the bfloat16 scale.
            # Move to CPU for quantization calculation, then back to original device
            device = w.device
            w_cpu = w.cpu()
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w_cpu.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale).to(device)
            zp = torch.from_numpy(zp).to(device)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding three frozen int8 weight parameters used for
            # three parallel matmul branches.
            def __init__(self, w1, w2, w3):
                # [FUNCTION DOC] HELPER: Stores the three int8 weights as parameters that do not
                # require gradients.
                super().__init__()
                self.w1 = torch.nn.Parameter(w1, requires_grad=False)
                self.w2 = torch.nn.Parameter(w2, requires_grad=False)
                self.w3 = torch.nn.Parameter(w3, requires_grad=False)

            def forward(self, x, scale1, scale2, scale3):
                # [FUNCTION DOC] HELPER: Flattens the leading dimensions of the input, multiplies
                # by each transposed weight cast to the input dtype, scales each result by its own
                # scale, and reshapes each back to the original leading dimensions. Returns a
                # tuple of the three outputs.
                # Ref: _linear_fp_act_int8_weight_impl in torchao/dtypes/uintx/plain_layout.py
                y1 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w1.t().to(x.dtype))
                    * scale1
                )
                y2 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w2.t().to(x.dtype))
                    * scale2
                )
                y3 = (
                    torch.mm(x.reshape(-1, x.shape[-1]), self.w3.t().to(x.dtype))
                    * scale3
                )
                return (
                    y1.reshape(*x.shape[:-1], y1.shape[-1]),
                    y2.reshape(*x.shape[:-1], y2.shape[-1]),
                    y3.reshape(*x.shape[:-1], y3.shape[-1]),
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, mid_dim, in_features), dtype=dtype, device=device)
        w1 = torch.rand((out_features, in_features), dtype=dtype, device=device)
        w2 = torch.rand((out_features, in_features), dtype=dtype, device=device)
        w3 = torch.rand((out_features, in_features), dtype=dtype, device=device)
        w1_int8pack, w1_scales = _convert_weight_to_int8pack(w1)
        w2_int8pack, w2_scales = _convert_weight_to_int8pack(w2)
        w3_int8pack, w3_scales = _convert_weight_to_int8pack(w3)
        mod = M(w1_int8pack, w2_int8pack, w3_int8pack).eval()
        self.common(mod, (x, w1_scales, w2_scales, w3_scales))
        self.assertEqual(counters["inductor"]["woq_matcher_count"], 3)


instantiate_device_type_tests(
    TestSelectAlgorithmGpu, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
