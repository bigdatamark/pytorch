# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Composable Kernel (CK / CK-tile) GEMM and conv autotune backends of
# Inductor on ROCm. Each test compiles a matmul, bmm, addmm, scaled matmul or conv2d with max-
# autotune restricted to the CK backend (standalone) or ATen plus CK (fallback), then compares
# against eager. Import time: sets the CUDA allocator to non-expandable segments when CUDA and
# Triton are available, creates a module logger and an empty environment patch mapping. Tests skip
# when not on HIP or when the CK library cannot be imported.
import logging
import os
import unittest


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import try_import_ck_lib
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocmVersionAtLeast,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
)


if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

log = logging.getLogger(__name__)


# patch env for tests if needed
_test_env = {}


@instantiate_parametrized_tests
class TestCKBackend(TestCase):
    # [CLASS DOC] Test case for CK backend autotuning. setUp locates the CK library directory
    # (stored as ck_dir) and skips every test when it is missing. Tests patch inductor config for
    # max autotune with the rocm.ck_dir option and run under the TF32-off context.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Seeds the random generator, looks up the CK library via the
        # inductor import helper and raises SkipTest if absent. Temporarily sets the env var that
        # disables the fresh inductor cache refresh while running the parent setUp, then restores
        # the old value (empty string if it was unset) in a finally block.
        # The new inductor cache refresh mechanism
        # introduced with https://github.com/pytorch/pytorch/pull/122661
        # interacts badly with persistent subprocesses during
        # autotuning. So we need to disable automatic cache refresh
        # before calling setUp() on the parent class.
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )

        torch.random.manual_seed(1234)

        self.ck_dir, _, _, _ = try_import_ck_lib()
        if not self.ck_dir:
            raise unittest.SkipTest("Composable Kernel library is not installed")

        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (
            subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]),
            subtest("CKTILE", decorators=[skipIfRocmVersionAtLeast([7, 14])]),
            "ATen,CK",
        ),
        name_fn=lambda b: {
            "CK": "standalone_ck",
            "CKTILE": "standalone_cktile",
            "ATen,CK": "fallback",
        }[b],
    )
    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("use_aoti", (True, False))
    def test_max_autotune_precompile_matmul(
        self, max_autotune_gemm_backends, autotune_in_subproc, use_aoti
    ):
        # [FUNCTION DOC] Autotunes a bf16 matmul (2240x256 by 256x2048) with CK / CKTILE / ATen+CK
        # backends, in or out of a subprocess, via AOTI runner or torch.compile, and checks the
        # result is close to eager. Raises AssertionError if the rocm config section is missing.
        """
        Make sure autotuning mm doesn't crash.
        """

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Eager reference matmul of two tensors, returned unchanged.
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):
            if use_aoti:
                Y_compiled = AOTIRunnerUtil.run(
                    model=mm,
                    example_inputs=(a, b),
                )
            else:

                @torch.compile(dynamic=False)
                def compiled_mm(x, w):
                    # [FUNCTION DOC] HELPER: Compiled wrapper forwarding both inputs to the
                    # helper.
                    return mm(x, w)

                Y_compiled = compiled_mm(a, b)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize("autotune_in_subproc", (True,))
    def test_max_autotune_precompile_matmul_dynamic(
        self, max_autotune_gemm_backends, autotune_in_subproc
    ):
        # [FUNCTION DOC] Compiles a matmul with a dynamic first dimension and CK backends in a
        # subprocess, checks it against eager, then reruns with a smaller first dimension (1024)
        # to exercise the dynamic shape path. Raises AssertionError if rocm config is missing.
        """
        Test matmul with dynamic shapes
        """

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        torch._dynamo.mark_dynamic(a, 0)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=True)
            def compiled_mm(a, b):
                # [FUNCTION DOC] HELPER: Dynamic-shape compiled matmul of its two inputs.
                return a @ b

            Y_compiled = compiled_mm(a, b)
            Y = a @ b
            torch.testing.assert_close(Y_compiled, Y)

            a1 = torch.randn(1024, 256, **tensor_options)
            Y1_compiled = compiled_mm(a1, b)
            Y1 = a1 @ b
            torch.testing.assert_close(Y1_compiled, Y1)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("num_gemms", (1, 2))
    def test_max_autotune_ck_backend_cpp_wrapper(self, num_gemms):
        # [FUNCTION DOC] Verifies CK GEMM templates work under JIT cpp_wrapper mode with one or
        # two chained GEMMs. Uses non-negative bf16 inputs for stable tolerances, compiles the
        # model with full graph, compares to eager, and checks the generated code references the
        # bare rocm_ symbol and never the AOT-only kernels member. Raises AssertionError if the
        # rocm config section is missing.
        """
        Verify that CK GEMM templates work under JIT cpp_wrapper mode.
        ``num_gemms=2`` chains a second GEMM of a different shape so the
        wrapper has to link against multiple distinct .so files.
        """
        M, N, K = 2240, 2048, 256
        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module computing a matmul of the first two inputs and,
            # when num_gemms from the enclosing test exceeds one, a further matmul with the third
            # input.
            def forward(self, a, b, c):
                # [FUNCTION DOC] HELPER: Returns the matmul of a and b, then of that result and c
                # if num_gemms is greater than 1.
                out = a @ b
                if num_gemms > 1:
                    out = out @ c
                return out

        model = MyModel().cuda()
        # Non-negative inputs: the num_gemms=2 case chains a second GEMM that
        # contracts over K=2048 in bf16. Signed inputs cause cancellation and
        # near-zero outputs whose relative error against eager's separately
        # rounded bf16 result explodes; rand() keeps outputs well-conditioned so
        # the default assert_close tolerance stays meaningful.
        a = torch.rand(M, K, **tensor_options)
        b = torch.rand(K, N, **tensor_options)
        c = torch.rand(N, N // 2, **tensor_options)
        expected = model(a, b, c)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CK",
                    "compile_threads": 2,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_dir": self.ck_dir,
                    "cpp_wrapper": True,
                }
            ),
            tf32_off(),
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, fullgraph=True)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            # JIT path must call the bare extern "C" symbol, not via the
            # AOT-only `kernels.` member.
            FileCheck().check("rocm_").check_not("kernels.rocm_").run(codes[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_precompile_preselected(self, max_autotune_gemm_backends):
        # [FUNCTION DOC] Compiles an fp16 matmul with a transposed second operand using
        # preselected CK instances (rocm.use_preselected_instances) in a subprocess, and checks it
        # against eager.
        """
        End to end test for picking preselected ck instances
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(2048, 256, **tensor_options).transpose(0, 1)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 12,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.use_preselected_instances": True,
                }
            ),
            tf32_off(),
        ):
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("Aten,CK",))
    def test_max_autotune_precompile_non_contiguous(self, max_autotune_gemm_backends):
        # [FUNCTION DOC] Builds large fp16 inputs with unusual strides (first operand column-
        # major-like with padded stride) and checks the compiled matmul falls back and matches
        # eager, treating NaNs as equal. The backend list is ATen plus CK (note the capitalization
        # is Aten in the parameter).
        """
        Make sure the matmul with non-contiguous inputs can fallback
        """

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.empty_strided((50257, 32768), (1, 50304), **tensor_options)
        b = torch.empty_strided((32768, 768), (768, 1), **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def mm(a, b):
                # [FUNCTION DOC] HELPER: Compiled matmul of its two inputs.
                return a @ b

            Y_compiled = mm(a, b)
            Y_eager = a @ b
            torch.testing.assert_close(Y_compiled, Y_eager, equal_nan=True)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize(
        "x_shape",
        ([4096, 2048], [2048], [4096, 1]),
        name_fn=lambda x_shape: f"x_shape_{'x'.join(map(str, x_shape))}",
    )
    @parametrize(
        "dtype",
        (torch.float16, torch.bfloat16),
        name_fn=lambda d: {torch.float16: "float16", torch.bfloat16: "bfloat16"}[d],
    )
    def test_max_autotune_addmm(self, max_autotune_gemm_backends, x_shape, dtype):
        # [FUNCTION DOC] Compiles addmm (alpha and beta both 1.0) with a bias of varying shape
        # (matrix, vector, column) in fp16 or bf16 with CK backends in a subprocess, and compares
        # to eager. Raises AssertionError if the rocm config section is missing.
        m, k, n = 4096, 224, 2048
        alpha, beta = 1.0, 1.0

        tensor_options = {"device": "cuda", "dtype": dtype}
        x = torch.ones(x_shape, **tensor_options)
        a = torch.randn(m, k, **tensor_options)
        b = torch.randn(k, n, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 2,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def addmm(x, a, b, alpha, beta):
                # [FUNCTION DOC] HELPER: Compiled addmm forwarding the bias, matrices, alpha and
                # beta.
                return torch.addmm(x, a, b, alpha=alpha, beta=beta)

            Y_compiled = addmm(x, a, b, alpha, beta)
            Y_eager = torch.addmm(x, a, b, alpha=alpha, beta=beta)

            torch.testing.assert_close(Y_compiled, Y_eager)

    @unittest.skip(
        "FIXME(tenpercent): kernel compilation errors on gfx942 as of 09/01/25"
    )
    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize("quantize_type", ("tensorwise", "rowwise"))
    @parametrize("has_bias", (True, False))
    def test_max_autotune_scaled_mm(
        self, max_autotune_gemm_backends, quantize_type, has_bias
    ):
        # [FUNCTION DOC] Compares compiled versus eager scaled matmul using fp8 inputs quantized
        # tensorwise or rowwise, with or without bias, bf16 output. Skips unless the GPU arch is
        # gfx94x or gfx95x; uses the fnuz fp8 type on gfx94x. Asserts output dtypes and closeness.
        # The test is unconditionally skipped by a decorator (kernel compile errors on gfx942).
        use_fast_accum = False
        runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" not in runtime_arch and "gfx95" not in runtime_arch:
            self.skipTest(f"Unsupported arch {runtime_arch}")
        # output dtype
        dtype = torch.bfloat16
        tensor_options = {"device": "cuda", "dtype": dtype}

        M = 2240
        N = 2048
        K = 256

        x = torch.randn(M, K, **tensor_options)
        w = torch.randn(N, K, **tensor_options)

        bias = None
        if has_bias:
            bias = torch.randn(N, **tensor_options)

        dtype_float8 = (
            torch.float8_e4m3fnuz if "gfx94" in runtime_arch else torch.float8_e4m3fn
        )

        f_quantize = (
            _quantize_tensorwise if quantize_type == "tensorwise" else _quantize_rowwise
        )

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = f_quantize(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale_t = w_inverse_scale.t()

        # quantize input x
        x_fp8, x_inverse_scale = f_quantize(x, dtype_float8)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
        # [FUNCTION DOC] HELPER: Runs scaled matmul on the fp8 operands and scales with optional
        # bias, bf16 output and the enclosing use_fast_accum flag (False); returns the result.
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale_t,
            bias,
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "compile_threads": 24,
                "rocm.ck_max_profiling_configs": 24,
                "rocm.ck_dir": self.ck_dir,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale_t,
                bias,
            )
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)

            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize(
        "max_autotune_conv_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATEN,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_conv2d(self, max_autotune_conv_backends):
        # [FUNCTION DOC] Autotunes a channels-last float32 conv2d (8 to 64 channels, 7x7 kernel)
        # with CK or ATen plus CK conv backends, not in a subprocess, with NHWC suggestion env var
        # set by the decorator. Compares to eager within 2e-4 tolerances.
        tensor_options = {"device": "cuda", "dtype": torch.float32}

        x = torch.randn(1, 8, 224, 224, **tensor_options)
        w = torch.randn(64, 8, 7, 7, **tensor_options)
        x_cl = x.to(memory_format=torch.channels_last)
        w_cl = w.to(memory_format=torch.channels_last)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": False,
                    "max_autotune_conv_backends": max_autotune_conv_backends,
                    "compile_threads": 4,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 4,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def conv2d(x, w):
                # [FUNCTION DOC] HELPER: Compiled 2D convolution of input and weight.
                return torch.conv2d(x, w)

            Y_eager = torch.conv2d(x_cl, w_cl)
            Y_compiled = conv2d(x_cl, w_cl)

            torch.testing.assert_close(Y_compiled, Y_eager, atol=2e-4, rtol=2e-4)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_precompile_bmm(
        self,
        max_autotune_gemm_backends,
    ):
        # [FUNCTION DOC] Autotunes a bf16 batched matmul with a transposed second operand using CK
        # backends and compares the compiled result to eager.
        """
        Test gemm-max-autotune torch.bmm with CK backend
        """

        def bmm(a, b):
            # [FUNCTION DOC] HELPER: Eager reference batched matmul.
            return torch.bmm(a, b)

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(16, 2240, 256, **tensor_options)
        b = torch.randn(16, 2048, 256, **tensor_options).transpose(1, 2)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_bmm(x, w):
                # [FUNCTION DOC] HELPER: Compiled wrapper calling the batched matmul helper.
                return bmm(x, w)

            Y_compiled = compiled_bmm(a, b)

            Y_eager = bmm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y_eager)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA_AND_TRITON and HAS_CPU and is_big_gpu():
        run_tests()
