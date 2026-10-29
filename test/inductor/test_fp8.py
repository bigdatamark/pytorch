# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for FP8 (float8) support on CUDA, ROCm, XPU and CPU.
# Contents: TestFP8Types (casts to and from float8 dtypes, saturated quantization, amax
# and layernorm quantization fusions, eager fallbacks), TestFP8Lowering (scaled matmul
# with tensorwise, rowwise, blockwise and MX scaling recipes, Triton TMA templates, error
# cases), TestE8M0ToFloat (decode of the e8m0 scale dtype), and several test classes for
# the e8m0 round-up conversion primitive and its log2/ceil pattern replacement (hardware
# PTX path vs bit manipulation path). Import-time side effects: some test classes are
# skipped by decorators that query the platform, and three classes are expanded per device
# by instantiate_device_type_tests at module end. The module-level setUpModule raises
# float32 matmul precision to high and tearDownModule restores it. Running the file as a
# script calls run_tests only if CUDA+Triton or CPU is present.

import contextlib
import functools
import unittest
from unittest import mock

import torch
import torch._inductor.lowering as inductor_lowering
from torch import Tensor
from torch._C import FileCheck
from torch._inductor import config, inductor_prims, ir, utils
from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
from torch._inductor.lowering import clone as lowering_clone, register_lowering
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, sympy_index_symbol_with_prefix
from torch.nn.functional import scaled_mm, ScalingType  # type: ignore[attr-defined]
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    IS_SM90,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
    skipCUDAIf,
)
from torch.testing._internal.common_quantized import ceil_div, to_blocked
from torch.testing._internal.common_utils import (
    parametrize,
    random_matrix_with_scaled_reduction_dim,
    skipIfRocm,
    xfailIf,
)
from torch.testing._internal.inductor_utils import (
    _quantize_blockwise,
    _quantize_rowwise,
    _quantize_tensorwise,
    _to_fp8_saturated,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    is_big_gpu,
)
from torch.utils._sympy.symbol import SymT
from torch.utils._triton import has_triton_tma_device


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] HELPER: unittest module fixture. Saves the current float32 matmul
    # precision in the module global _PRIOR_FP32_MATMUL_PRECISION, then sets the precision
    # to "high" for every test in the module.
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    # [FUNCTION DOC] HELPER: unittest module fixture. If a precision was saved by
    # setUpModule, restores it and resets the saved global to None; otherwise
    # does nothing.
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


f8_msg = "FP8 is only supported on H100+, SM 8.9 and MI300+, XPU and CPU devices"


def _is_cuda_device(device) -> bool:
    # [FUNCTION DOC] HELPER: returns True if the device is a CUDA device. Accepts a
    # torch.device (checks its type) or a string (substring match on "cuda", so an indexed
    # name also matches); any other type yields False.
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return "cuda" in device
    return False


def _fix_fp8_dtype_for_rocm(
    dtype: torch.dtype | list[torch.dtype] | tuple[torch.dtype], device
) -> torch.dtype | list[torch.dtype] | tuple[torch.dtype]:
    # [FUNCTION DOC] HELPER: on ROCm MI300 (gfx94x arch of device 0) with a CUDA-type
    # device, maps float8_e4m3fn to float8_e4m3fnuz and float8_e5m2 to float8_e5m2fnuz,
    # recursing through a tuple or list and returning the same container kind. Any other
    # dtype, or any other platform/device, is returned unchanged. This lets one test body
    # serve both CUDA and ROCm (and CPU).
    # This function is used to change FP8 data types
    # with MI300 supported FP8 types if device is GPU:
    #    e4m3fn -> e4m3fnuz
    #    e5m2   -> e5m2fnuz
    # Supports single, tuple and list of dtypes
    # Keeps the same test name for CUDA and ROCm
    # Also it allows to enable FP8 inductor tests for CPU
    if (
        torch.version.hip
        and (_is_cuda_device(device))
        and ("gfx94" in torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    ):
        # MI300 uses different float8 dtypes
        if isinstance(dtype, tuple):
            return tuple(_fix_fp8_dtype_for_rocm(x, device) for x in dtype)
        if isinstance(dtype, list):
            return [_fix_fp8_dtype_for_rocm(x, device) for x in dtype]
        if dtype == torch.float8_e4m3fn:
            return torch.float8_e4m3fnuz
        elif dtype == torch.float8_e5m2:
            return torch.float8_e5m2fnuz
    return dtype


def _prepare_blockwise_scale(
    inverse_scale: torch.Tensor,
    block_outer: int,
    block_inner: int,
    transposed: bool,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: reshapes a blockwise inverse scale into the layout the cuBLAS
    # blockwise kernels expect. For 1x128 blocks (outer 1, inner 128) the scale is made
    # outer-dim-major by a transpose-contiguous-transpose round trip, and one more
    # transpose is applied when the data tensor was passed transposed. For other block
    # sizes (128x128) the last dim is zero padded to a multiple of 4 and the result is
    # transposed. Returns the new scale tensor.
    # The cuBLAS blockwise kernels expect outer-dim-major scales for 1x128 blocks
    # and shape (round_up(K/128, 4), {M,N}/128) for 128x128 blocks (inner dim
    # padded to a multiple of 4 before the transpose). `transposed` indicates
    # whether the corresponding data tensor was transposed (e.g. weight passed
    # as w.t()): if so we apply one additional transpose to keep the scale
    # aligned with the data layout.
    if (block_outer, block_inner) == (1, 128):
        out = inverse_scale.t().contiguous().t()
        return out.t() if transposed else out
    pad_amount = (-inverse_scale.shape[-1]) % 4
    if pad_amount:
        inverse_scale = torch.nn.functional.pad(
            inverse_scale, (0, pad_amount), "constant", 0
        )
    return inverse_scale.t()


@contextlib.contextmanager
def _simulate_float8_e4m3fn_uint8_storage():
    # [FUNCTION DOC] HELPER: context manager that makes Inductor behave as on SM < 89,
    # where float8_e4m3fn inputs are read through uint8 storage. It patches the two
    # codegen modules' predicate for uint8 storage so it is true for e4m3fn when no
    # argument name is given or the name starts with the input pointer prefix, and patches
    # the lowering module's fp8 support check to report e4m3fn as unsupported (other
    # dtypes defer to the original check). The patches are undone on exit.
    # Make float8_e4m3fn behave as it does on SM < 89, where triton lacks native
    # fp8 support and inputs are read through uint8 storage
    import torch._inductor.codegen.triton as triton_codegen
    import torch._inductor.codegen.triton_utils as triton_utils
    import torch._inductor.lowering as inductor_lowering

    real_supported = inductor_lowering.is_triton_fp8_dtype_supported

    def force_uint8_storage(dtype, arg_name=None, *, device=None):
        # [FUNCTION DOC] HELPER: replacement predicate. Returns True only for
        # float8_e4m3fn and only when no argument name is given or the name begins with an
        # input pointer prefix. The device keyword is accepted and ignored.
        return dtype == torch.float8_e4m3fn and (
            arg_name is None or arg_name.startswith("in_ptr")
        )

    def force_unsupported(dtype, device=None, **kwargs):
        # [FUNCTION DOC] HELPER: replacement fp8 support check. Returns False for
        # float8_e4m3fn, otherwise delegates to the real check captured before patching
        # with the same device and extra keyword arguments.
        if dtype == torch.float8_e4m3fn:
            return False
        return real_supported(dtype, device, **kwargs)

    with (
        mock.patch.object(
            triton_utils,
            "use_uint8_triton_storage_for_cuda_float8_e4m3fn",
            force_uint8_storage,
        ),
        mock.patch.object(
            triton_codegen,
            "use_uint8_triton_storage_for_cuda_float8_e4m3fn",
            force_uint8_storage,
        ),
        mock.patch.object(
            inductor_lowering,
            "is_triton_fp8_dtype_supported",
            force_unsupported,
        ),
    ):
        yield


class TestFP8Types(TestCase):
    # [CLASS DOC] Device-generic tests of FP8 dtype handling in Inductor: casts between
    # float and float8 types, saturating conversion, amax and layernorm quantization
    # patterns, eager fallback for unsupported Triton fp8 dtypes, and the benchmark and
    # scaled-mm smoke tests. Instantiated per device (CUDA, XPU, CPU) at module end.
    @onlyCUDA
    @skipIfRocm
    @config.patch({"force_disable_caches": True})
    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/189560"
    )
    def test_float8_e4m3fn_uint8_decode_codegen(self, device):
        # [FUNCTION DOC] Currently skipped (disabled due to CI failures, issue 189560).
        # Intended to check that with uint8 storage simulated, compiling a float
        # conversion of a float8_e4m3fn tensor built from all 256 byte values gives bit-
        # identical float32 results and signs to eager, and that the generated source
        # reads the input as an unsigned 8-bit pointer, uses the fp8e4m3fn-to-float32
        # helper, and does not use the native fp8 pointer type. Captures generated source
        # by patching the GraphLowering method that saves output code.
        from torch._inductor.graph import GraphLowering

        def fn(t):
            # [FUNCTION DOC] HELPER: converts the input tensor to float32.
            return t.float()

        bits = torch.arange(256, device=device, dtype=torch.uint8)
        t = bits.view(torch.float8_e4m3fn)
        expected = fn(t)
        source_codes = []

        def save_output_code(code):
            # [FUNCTION DOC] HELPER: replacement for save_output_code that appends the
            # generated source string to the enclosing list source_codes.
            source_codes.append(code)

        with (
            _simulate_float8_e4m3fn_uint8_storage(),
            mock.patch.object(GraphLowering, "save_output_code", save_output_code),
        ):
            torch._dynamo.reset()
            actual = torch.compile(fn, fullgraph=True)(t)

        self.assertEqual(actual.view(torch.uint32), expected.view(torch.uint32))
        self.assertEqual(torch.signbit(actual), torch.signbit(expected))

        self.assertTrue(source_codes)
        code = "\n".join(source_codes)
        self.assertIn("'in_ptr0': '*u8'", code)
        self.assertIn("triton_helpers.fp8e4m3fn_to_float32", code)
        self.assertNotIn("'in_ptr0': '*fp8e4nv'", code)

    @onlyCUDA
    @skipIfRocm
    @config.patch({"force_disable_caches": True})
    def test_float8_e4m3fn_uint8_storage_arithmetic_falls_back(self, device):
        # [FUNCTION DOC] With uint8 storage simulated, compiles an elementwise self
        # comparison of a float8_e4m3fn tensor (all 256 byte patterns) with fullgraph and
        # checks the result equals eager, i.e. the unsupported arithmetic falls back
        # correctly. CUDA only, skipped on ROCm.
        bits = torch.arange(256, device=device, dtype=torch.uint8)
        t = bits.view(torch.float8_e4m3fn)

        def fn(x):
            # [FUNCTION DOC] HELPER: elementwise equality of the input with itself.
            return x == x

        with _simulate_float8_e4m3fn_uint8_storage():
            torch._dynamo.reset()
            actual = torch.compile(fn, fullgraph=True)(t)
        expected = fn(t)
        self.assertEqual(actual, expected)

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    def test_xblock_for_small_numel(self, float8_dtype: torch.dtype, device: str):
        # [FUNCTION DOC] Regression test: casting a one element float tensor to a float8
        # dtype under torch.compile must not choose a Triton XBLOCK larger than the number
        # of elements (the fp8 conversion raises the minimum elements per thread, which
        # used to break the config sanity check). Compares compiled and eager output after
        # widening to half, with loose tolerance.
        """
        TritonOverrides.to_dtype will set min_elem_per_thread to 2 or 4
        depends on the variant of fp8 type.
        This cause triton_heuristics.triton_config pick a XBLOCK larger
        than numel and fail the config sanity check.

        We should not pick a XBLOCK larger than xnumel
        """
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: casts the input to the float8 dtype of the
            # enclosing test.
            return x.to(dtype=float8_dtype)

        x = torch.randn(1, device=device)
        expected = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(expected.half(), actual.half(), rtol=1e-2, atol=1e-2)

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_eager_fallback(self, dtype: torch.dtype, device: torch.device):
        # [FUNCTION DOC] Compiles, with dynamic shapes, a function that builds an fp8
        # weight, bias and unit scales on the device and calls the scaled matmul with
        # bias, then runs it with two different leading input sizes to exercise the
        # dynamic path. Only checks that it runs without error; the outputs are
        # not compared.
        weight_shape = (32, 16)

        e4m3_type = torch.float8_e4m3fn
        e4m3_type = _fix_fp8_dtype_for_rocm(e4m3_type, device=device)

        def fp8_matmul_unwrapped(x):
            # [FUNCTION DOC] HELPER: creates unit scales, a random bias and a transposed
            # fp8 weight in the test dtype, takes reciprocal scales, and returns the
            # result of the scaled matmul of the input with the weight with the requested
            # output dtype.
            a_scale = torch.Tensor([1.0]).to(device=device)
            b_scale = torch.Tensor([1.0]).to(device=device)
            output_scale = None
            input_bias = torch.rand(32, device=device, dtype=dtype)
            weight = torch.rand(*weight_shape, device=device, dtype=dtype).T.to(
                e4m3_type
            )
            a_inverse_scale = 1 / a_scale
            b_inverse_scale = 1 / b_scale
            output = torch._scaled_mm(
                x,
                weight,
                bias=input_bias,
                out_dtype=dtype,
                scale_a=a_inverse_scale,
                scale_b=b_inverse_scale,
                scale_result=output_scale,
            )
            return output

        compiled_fp8_matmul = torch.compile(
            fp8_matmul_unwrapped, backend="inductor", dynamic=True
        )

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device=device, dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)

        x_shape = (15, 16)
        x = torch.rand(*x_shape, device=device, dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)  # noqa: F841

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    @parametrize("dst_types", [(torch.float8_e4m3fn, torch.float8_e5m2)])
    def test_valid_cast(
        self, dtype: torch.dtype, shape: str, dst_types: tuple, device: torch.device
    ):
        # [FUNCTION DOC] For float16, bfloat16 and float32 inputs of two shapes, compiles
        # with dynamic shapes a function that round-trips a tensor through float8_e4m3 and
        # float8_e5m2 and back, and checks both results are close to the original within
        # 0.5 relative and absolute tolerance. The shape is a comma separated string
        # of dims.
        dst_types = _fix_fp8_dtype_for_rocm(dst_types, device=device)
        e4m3, e5m2 = dst_types

        def fp8_cast(x):
            # [FUNCTION DOC] HELPER: returns a pair of tensors, the input cast to each of
            # the two float8 dtypes and back to the original dtype.
            y0 = x.to(dtype=e4m3).to(dtype)
            y1 = x.to(dtype=e5m2).to(dtype)
            return y0, y1

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device=device, dtype=dtype)
        y0_fp8, y1_fp8 = compiled_fp8_cast(x)

        torch.testing.assert_close(y0_fp8, x, rtol=5e-1, atol=5e-1)
        torch.testing.assert_close(y1_fp8, x, rtol=5e-1, atol=5e-1)

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "Requires CUDA + Triton")
    @skipIfRocm
    @onlyCUDA
    def test_cuda_fp8_cast_fallback_for_unsupported_triton_dtype(self, device):
        # [FUNCTION DOC] For float8_e4m3fnuz and float8_e5m2fnuz (plus float8_e4m3fn when
        # Triton does not support it on the device), compiles a cast of a small float32
        # tensor with fullgraph and checks the result dtype, exact equality with eager,
        # and that the generated source does not mention the Triton type name of the fp8
        # dtype, i.e. the cast fell back to the eager kernel. CUDA only, not ROCm.
        def fp8_cast(x, dtype):
            # [FUNCTION DOC] HELPER: casts the input to the given dtype.
            return x.to(dtype=dtype)

        fp8_dtypes = [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
        if not utils.is_triton_fp8_dtype_supported(torch.float8_e4m3fn, device):
            fp8_dtypes.append(torch.float8_e4m3fn)

        x = torch.tensor(
            [-32.0, -16.0, -1.0, -0.25, 0.0, 0.25, 1.0, 16.0],
            device=device,
            dtype=torch.float32,
        )
        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", fullgraph=True)

        for fp8_dtype in fp8_dtypes:
            torch._dynamo.reset()
            expected = fp8_cast(x, fp8_dtype)
            actual, code = run_and_get_code(compiled_fp8_cast, x, fp8_dtype)

            self.assertEqual(actual.dtype, fp8_dtype)
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
            self.assertNotIn(utils.triton_type(fp8_dtype), "\n".join(code))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @skipIfRocm
    @onlyCUDA
    @parametrize(
        "src_dtype",
        (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64),
    )
    @parametrize("dst_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    def test_int_to_float8_cast(
        self, src_dtype: torch.dtype, dst_dtype: torch.dtype, device: torch.device
    ):
        # [FUNCTION DOC] For each integer or bool source dtype, compiles with fullgraph a
        # cast of a small tensor to a float8 dtype (e4m3fn or e5m2) and checks the result
        # dtype is the target and the values equal eager exactly. Bool input uses
        # alternating values, uint8 uses small non-negative values, other ints include
        # negative values. CUDA only, not ROCm.
        def fp8_cast(x):
            # [FUNCTION DOC] HELPER: casts the input to the enclosing test's float8 dtype.
            return x.to(dtype=dst_dtype)

        if src_dtype == torch.bool:
            x = torch.tensor([False, True, False, True], device=device)
        elif src_dtype == torch.uint8:
            x = torch.tensor([0, 1, 2, 16], dtype=src_dtype, device=device)
        else:
            x = torch.tensor([-16, -2, 0, 16], dtype=src_dtype, device=device)

        expected = fp8_cast(x)
        actual = torch.compile(fp8_cast, backend="inductor", fullgraph=True)(x)

        self.assertEqual(actual.dtype, dst_dtype)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_bad_cast(self, device):
        # [FUNCTION DOC] Checks direct conversion between float8_e4m3fn and float8_e5m2
        # under dynamic compile. On CUDA and XPU both directions must raise
        # BackendCompilerFailed whose message says the conversion is not supported. On
        # other devices (CPU) the conversions succeed and the output has the
        # requested dtype.
        def fp8_cast(x, dtype):
            # [FUNCTION DOC] HELPER: casts the input to the given dtype.
            return x.to(dtype=dtype)

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)
        x_shape = (16, 16, 16)

        if "cuda" in device or "xpu" in device:
            with self.assertRaisesRegex(
                torch._dynamo.exc.BackendCompilerFailed,
                "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
            ):
                x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e4m3fn)
                compiled_fp8_cast(x, torch.float8_e5m2)

            with self.assertRaisesRegex(
                torch._dynamo.exc.BackendCompilerFailed,
                "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
            ):
                x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e5m2)
                compiled_fp8_cast(x, torch.float8_e4m3fn)

        else:
            x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e4m3fn)
            out = compiled_fp8_cast(x, torch.float8_e5m2)
            self.assertEqual(out.dtype, torch.float8_e5m2)

            x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e5m2)
            out = compiled_fp8_cast(x, torch.float8_e4m3fn)
            self.assertEqual(out.dtype, torch.float8_e4m3fn)

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("dst_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("16,16,16", "4,2048,4096"))
    def test_to_fp8_saturated(
        self,
        src_dtype: torch.dtype,
        dst_dtype: torch.dtype,
        shape: str,
        device: torch.device,
    ):
        # [FUNCTION DOC] Compares, for half-precision and float32 sources and both fp8
        # targets, the compiled (dynamic) and eager results of the saturating float8
        # conversion helper on uniform random data, after widening to half, with tolerance
        # 0.5. Shape is a comma separated string.
        dst_dtype = _fix_fp8_dtype_for_rocm(dst_dtype, device=device)

        def fp8_saturated(x, dtype):
            # [FUNCTION DOC] HELPER: applies the saturating float8 conversion helper.
            return _to_fp8_saturated(x, dtype)

        compiled_fp8_cast = torch.compile(
            fp8_saturated, backend="inductor", dynamic=True
        )
        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device=device, dtype=src_dtype)
        y_compiled = compiled_fp8_cast(x, dst_dtype)
        y = fp8_saturated(x, dst_dtype)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=5e-1, atol=5e-1)

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        # [FUNCTION DOC] Compiles a function that takes the absolute maximum of the whole
        # input, scales it by a scalar float scale and converts it with saturation to
        # float8, and compares with eager on random half data (tolerance 1e-2). Covers
        # several batch, sequence and hidden size shapes parsed from a string.
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor):
            # [FUNCTION DOC] HELPER: returns the saturating fp8 conversion of the global
            # amax of the input multiplied by the scale.
            y = torch.amax(torch.abs(x))
            y_scaled = y.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(y_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(0.2, device=device, dtype=torch.float)

        y_compiled = compiled_amax_fp8_quant(x, scale)
        y = amax_fp8(x, scale)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-2, atol=1e-2)

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_along_with_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        # [FUNCTION DOC] Compiles a function that writes the global absolute maximum of
        # the input into a one element buffer (in place side effect) and also returns the
        # scaled, saturated fp8 quantization of the input. Checks both the returned tensor
        # and the mutated amax buffer against eager, over several shapes.
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            # [FUNCTION DOC] HELPER: fills the amax buffer in place with the input's
            # absolute maximum, then returns the input scaled in float32 and converted to
            # fp8 with saturation.
            amax_buffer.fill_(torch.amax(torch.abs(x)))
            x_scaled = x.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(1.0, device=device, dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device=device, dtype=torch.half)
        y_compiled = compiled_amax_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device=device, dtype=torch.half)
        y = amax_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("amax_keep_dim", (True, False))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_layernorm_fp8_quant(
        self,
        float8_dtype: torch.dtype,
        amax_keep_dim: bool,
        shape: str,
        device: torch.device,
    ):
        # [FUNCTION DOC] Same as the amax fusion test but with a float32 layer norm (no
        # affine parameters, eps 1e-5) over the hidden dim before the amax and
        # quantization; amax may keep its reduced dim or not. Compares the quantized
        # output and the amax buffer of compiled and eager runs over several shapes.
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            # [FUNCTION DOC] HELPER: layer norms the input in float32, stores its absolute
            # maximum into the amax buffer in place, then returns the scaled fp8 saturated
            # quantization of the normalized values.
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            amax_buffer.fill_(
                torch.amax(torch.abs(x), keepdim=amax_keep_dim).reshape(-1)[0]
            )
            x_scaled = x * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(0.2, device=device, dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device=device, dtype=torch.half)
        y_compiled = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device=device, dtype=torch.half)
        y = ln_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("4,2048,4096",))
    @parametrize("keepdim", (False, True))
    def test_layernorm_fp8_quant_benchmark(
        self,
        float8_dtype: torch.dtype,
        shape: str,
        keepdim: bool,
    ):
        # [FUNCTION DOC] Benchmark style test (CUDA/XPU only): times compiled and eager
        # layernorm plus fp8 quantization and a compiled layernorm only baseline using the
        # profiling based benchmark helper, then prints the latencies. It asserts nothing
        # about the timings; failures arise only from errors in compile or run.
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=GPU_TYPE)
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def ln(x: Tensor):
            # [FUNCTION DOC] HELPER: float32 layer norm of the input over the hidden dim,
            # no affine parameters.
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            return x

        def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            # [FUNCTION DOC] HELPER: layer norm in float32, copies the absolute maximum
            # (with the test's keepdim choice) into a view of the amax buffer, then
            # returns the scaled saturated fp8 quantization.
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            amax = torch.amax(torch.abs(x), keepdim=keepdim)
            amax_buffer.view_as(amax).copy_(amax)
            x_scaled = x * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=GPU_TYPE, dtype=torch.half)
        scale = torch.tensor(0.2, device=GPU_TYPE, dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device=GPU_TYPE, dtype=torch.half)
        amax_buffer = torch.zeros((1), device=GPU_TYPE, dtype=torch.half)
        _ = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        compiled_latency = utils.do_bench_using_profiling(
            functools.partial(compiled_ln_fp8_quant, x, scale, amax_buffer_compiled)
        )
        eager_latency = utils.do_bench_using_profiling(
            functools.partial(ln_fp8, x, scale, amax_buffer)
        )

        compiled_ln = torch.compile(ln, backend="inductor")
        _ = compiled_ln(x)
        ln_latency = utils.do_bench_using_profiling(functools.partial(compiled_ln, x))

        print(
            f"Config: {float8_dtype=}, {shape=}, {keepdim=}. "
            f"Benchmark results: Inductor: {compiled_latency}ms, Eager: {eager_latency}ms, "
            f"LN only Inductor: {ln_latency}ms."
        )

    @skipCUDAIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    def test_scaled_mm_pdl_handles_none_bias(self, device):
        # [FUNCTION DOC] Regression test: with programmatic dependent launch enabled,
        # tensor descriptors disabled and exhaustive GEMM autotuning over ATen and Triton,
        # compiles a scaled matmul called with no bias and a result scale and checks it
        # matches eager (loose tolerances). Covers the case where the bias argument
        # is None.
        dtype_float8 = _fix_fp8_dtype_for_rocm(torch.float8_e4m3fn, device)
        M, K, N = 32, 64, 32

        # A row-major, B column-major view (transpose of contiguous)
        a = torch.randn(M, K, device=device, dtype=torch.float16).to(dtype_float8)
        b = torch.randn(N, K, device=device, dtype=torch.float16).to(dtype_float8).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        scale_r = torch.tensor(1.0, device=device)

        def linear(a, b, sa, sb, sr):
            # [FUNCTION DOC] HELPER: calls the private scaled matmul with positional
            # scales, no bias, a result scale and bfloat16 output.
            return torch._scaled_mm(a, b, sa, sb, None, sr, out_dtype=torch.bfloat16)

        expected = linear(a, b, scale_a, scale_b, scale_r)

        patch_cfg = {
            "triton.enable_pdl": True,
            "triton.use_tensor_descriptor": False,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
        }

        with config.patch(patch_cfg):
            compiled = torch.compile(linear, mode="max-autotune")
            actual = compiled(a, b, scale_a, scale_b, scale_r)

        self.assertEqual(expected, actual, rtol=5e-2, atol=0.07)


class TestFP8Lowering(TestCase):
    # [CLASS DOC] Device-generic tests of how Inductor lowers and autotunes scaled matmul
    # on fp8 inputs: tensorwise, rowwise, blockwise and MX scaling, scale rank handling,
    # stride preservation, error reporting for bad shapes, and Triton TMA templates.
    # Instantiated per device at module end.
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @skipIfRocm(msg="FP8 scaled_mm tensorwise eager path is not supported by hipBLAS")
    @onlyCUDA
    def test_functional_scaled_mm_fullgraph(self, device):
        # [FUNCTION DOC] Checks that the functional scaled_mm API with tensorwise scales
        # on bfloat16 data quantized to float8_e4m3fn compiles with fullgraph and matches
        # the eager result exactly. CUDA only, skipped on ROCm.
        M, N, K = 128, 128, 128
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        x_fp8, x_scale = _quantize_tensorwise(x, torch.float8_e4m3fn)
        w_fp8, w_scale = _quantize_tensorwise(w, torch.float8_e4m3fn)

        def fn(x_fp8, w_fp8_t, x_scale, w_scale):
            # [FUNCTION DOC] HELPER: calls the functional scaled_mm with tensorwise
            # scaling for both operands and bfloat16 output.
            return scaled_mm(
                x_fp8,
                w_fp8_t,
                x_scale,
                ScalingType.TensorWise,
                w_scale,
                ScalingType.TensorWise,
                output_dtype=torch.bfloat16,
            )

        expected = fn(x_fp8, w_fp8.t(), x_scale, w_scale)
        actual = torch.compile(fn, fullgraph=True)(x_fp8, w_fp8.t(), x_scale, w_scale)
        self.assertEqual(expected, actual)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @skipIfRocm(msg="FP8 scaled_mm tensorwise eager path is not supported by hipBLAS")
    @onlyCUDA
    @parametrize(
        "scale_a_shape,scale_b_shape",
        [
            ((1,), ()),
            ((), (1,)),
            ((1, 1, 1), ()),
            ((1, 1, 1), (1,)),
            ((1, 1, 1), (1, 1)),
        ],
    )
    def test_scaled_mm_mixed_tensorwise_scale_ranks(
        self, scale_a_shape, scale_b_shape, device
    ):
        # [FUNCTION DOC] For several combinations of scale tensor shapes (scalar, one
        # element vector, 3-D with singleton dims), compiles the private scaled matmul on
        # all-ones fp8 operands with fullgraph and checks it equals eager within 1e-2.
        # Verifies mixed ranks of tensorwise scales are accepted. CUDA only, not ROCm.
        M = N = K = 64
        x_fp8 = torch.ones(M, K, device=device, dtype=torch.float8_e4m3fn)
        w_fp8 = torch.ones(N, K, device=device, dtype=torch.float8_e4m3fn)
        scale_a = torch.ones(scale_a_shape, device=device)
        scale_b = torch.ones(scale_b_shape, device=device)

        def fn(x_fp8, w_fp8, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: private scaled matmul of the first operand with the
            # transposed second operand using the given scales, float32 output.
            return torch._scaled_mm(
                x_fp8,
                w_fp8.T,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.float32,
            )

        expected = fn(x_fp8, w_fp8, scale_a, scale_b)
        actual = torch.compile(fn, fullgraph=True)(x_fp8, w_fp8, scale_a, scale_b)
        self.assertEqual(expected, actual, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @skipIfRocm(msg="FP8 scaled_mm tensorwise eager path is not supported by hipBLAS")
    @onlyCUDA
    def test_scaled_mm_rejects_high_rank_scale_b(self, device):
        # [FUNCTION DOC] With two 3-D singleton scales, both the eager call and the
        # compiled fullgraph call of the scaled matmul must raise RuntimeError whose
        # message matches the transpose-expects pattern. CUDA only, not ROCm.
        M = N = K = 64
        x_fp8 = torch.ones(M, K, device=device, dtype=torch.float8_e4m3fn)
        w_fp8 = torch.ones(N, K, device=device, dtype=torch.float8_e4m3fn)
        scale_a = torch.ones(1, 1, 1, device=device)
        scale_b = torch.ones(1, 1, 1, device=device)

        def fn(x_fp8, w_fp8, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: private scaled matmul of the first operand with the
            # transposed second operand, float32 output, scales passed by keyword.
            return torch._scaled_mm(
                x_fp8,
                w_fp8.T,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.float32,
            )

        with self.assertRaisesRegex(RuntimeError, r"t\(\) expects"):
            fn(x_fp8, w_fp8, scale_a, scale_b)
        with self.assertRaisesRegex(RuntimeError, r"t\(\) expects"):
            torch.compile(fn, fullgraph=True)(x_fp8, w_fp8, scale_a, scale_b)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    @onlyOn(["cuda", "xpu", "cpu"])
    def test_tensorwise_scaling(
        self,
        dtype: torch.dtype,
        shape: str,
        has_bias: bool,
        use_fast_accum: bool,
        persistent_matmul: bool,
        device,
    ):
        # [FUNCTION DOC] Tensorwise-scaled fp8 linear (activation by transposed weight,
        # optional bias, optional fast accumulate, optional persistent TMA matmul)
        # compiled in max-autotune mode must match eager, in bfloat16 or float32 output
        # (tolerances differ). Skips float32 output with bias. Inputs are quantized
        # tensorwise to float8_e4m3fn; shape is a comma separated M,K,N string.
        if dtype is torch.float32 and has_bias:
            self.skipTest("bias is not supported when output dtype is float32")
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = random_matrix_with_scaled_reduction_dim(
            M, K, dtype=dtype, device=device, reduction_dim=-1
        )
        w = random_matrix_with_scaled_reduction_dim(
            N, K, dtype=dtype, device=device, reduction_dim=-1
        )
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with tensorwise recipes on both
            # operands, optional bias and fast accumulate flag, output in the test dtype.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.TensorWise,
                w_inverse_scale,
                ScalingType.TensorWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)
            # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
            # autotuning for the compiled case, the results can be different because of
            # the way blocks of results are accumulated (float addition not associative), so
            # setting a small absolute tolerance in these tests
            if dtype == torch.bfloat16:
                self.assertEqual(y_eager, y_compiled, rtol=5e-2, atol=0.07)
            else:
                self.assertEqual(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    def test_scaled_mm_preserves_strides(self, device):
        # [FUNCTION DOC] Checks that a user supplied post-grad custom pass that inserts
        # clone nodes before the scaled matmul operands (changing their strides) still
        # gives a correct result. Verifies the pass ran, the compiled output matches eager
        # (1e-2), and the generated code mentions both the scaled mm and clone. Forces
        # cache disable so the pass runs. Fast accumulate is turned off on XPU.
        """Test that scaled_mm preserves stride ordering through a custom pass."""

        GPU_TYPE = device
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False

        def f(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: casts both inputs to fp8 with the first made row
            # major and the second column major, then calls the functional scaled_mm with
            # tensorwise scales and bfloat16 output.
            # Convert to fp8 with correct strides for scaled_mm
            dtype_float8 = torch.float8_e4m3fn
            dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, GPU_TYPE)
            a_fp8 = a.to(dtype_float8).contiguous()  # row-major
            b_fp8 = b.t().contiguous().t().to(dtype_float8)  # column-major
            return torch.nn.functional.scaled_mm(
                a_fp8,
                b_fp8,
                scale_a,
                ScalingType.TensorWise,
                scale_b,
                ScalingType.TensorWise,
                output_dtype=torch.bfloat16,
                use_fast_accum=use_fast_accum,
            )

        class ScaledMMStridePass(PatternMatcherPass):
            # [CLASS DOC] HELPER: custom post-grad pass derived from the pattern matcher
            # pass class that does not use pattern matching. The attribute called records
            # whether the pass ran.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: calls the parent constructor and sets the called
                # flag to False.
                super().__init__()
                self.called = False

            def __call__(self, g: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: walks the graph; for each call to the v2 scaled
                # mm aten op inserts contiguous clone nodes for its first two operands
                # just before it and rewires the node's arguments to the clones, sets the
                # called flag, lints the graph and returns it.
                # Directly manipulate the graph without using pattern matching
                for node in g.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten._scaled_mm_v2.default
                    ):
                        # Insert clone operations before scaled_mm
                        with g.inserting_before(node):
                            a_fp8, b_fp8 = node.args[0], node.args[1]

                            # Clone the inputs to potentially change stride ordering
                            a_cloned = g.call_function(
                                torch.ops.aten.clone.default,
                                (a_fp8,),
                                {"memory_format": torch.contiguous_format},
                            )
                            b_cloned = g.call_function(
                                torch.ops.aten.clone.default,
                                (b_fp8,),
                                {"memory_format": torch.contiguous_format},
                            )

                            # Replace the arguments in the scaled_mm call
                            node.args = (a_cloned, b_cloned) + node.args[2:]
                            self.called = True

                g.lint()
                return g

        stride_pass = ScaledMMStridePass()

        # Create inputs with correct strides for scaled_mm
        a = torch.randn((64, 128), dtype=torch.bfloat16, device=GPU_TYPE)
        b = torch.randn((128, 64), dtype=torch.bfloat16, device=GPU_TYPE)
        scale_a = torch.tensor(1.0, device=GPU_TYPE)
        scale_b = torch.tensor(1.0, device=GPU_TYPE)

        # First, verify that f works without the pass (baseline)
        expected = f(a, b, scale_a, scale_b)

        from torch._inductor import config

        with config.patch(
            {
                "post_grad_custom_post_pass": stride_pass,
                # Force cache miss so our post-grad pass actually runs
                "force_disable_caches": True,
            }
        ):
            f_compiled = torch.compile(f, dynamic=False)
            result = f_compiled(a, b, scale_a, scale_b)

            # Verify the pattern was called
            self.assertTrue(stride_pass.called, "Stride ordering pass was not called")

            # Verify correctness - the pass should preserve correctness
            # even though it modified strides
            self.assertEqual(expected, result, atol=1e-2, rtol=1e-2)

            # Verify the generated code contains the clones inserted by our pass
            _, (wrapper,) = run_and_get_code(f_compiled, a, b, scale_a, scale_b)
            self.assertIn("scaled_mm", wrapper.lower())
            # The clones should be visible in the generated code
            self.assertIn("clone", wrapper.lower())

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device() or not is_big_gpu(),
        "Need device-side TMA support in Triton and max-autotune",
    )
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_tensorwise_scaling_tma_template(
        self,
        dtype: torch.dtype,
        shape: str,
        use_fast_accum: bool,
        device,
    ):
        # [FUNCTION DOC] On CUDA with device TMA support in Triton and a large GPU,
        # compiles a tensorwise scaled linear in max-autotune with only the device TMA
        # Triton scaled-mm template allowed, then checks the generated code sets both
        # scale recipes to the tensorwise value, output dtypes match, and results are
        # close to eager. Bfloat16 or float32 output; no bias.
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with tensorwise recipes,
            # optional bias, fast accumulate flag and output in the test dtype.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.TensorWise,
                w_inverse_scale,
                ScalingType.TensorWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled, code = run_and_get_code(
                linear_compiled,
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

            FileCheck().check(
                f"SCALE_RECIPE_A : tl.constexpr = {ScalingType.TensorWise.value}"
            ).run(code[0])
            FileCheck().check(
                f"SCALE_RECIPE_B : tl.constexpr = {ScalingType.TensorWise.value}"
            ).run(code[0])
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)
            # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
            # autotuning for the compiled case, the results can be different because of
            # the way blocks of results are accumulated (float addition not associative), so
            # setting a small absolute tolerance in these tests
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling(
        self,
        shape: str,
        has_bias: bool,
        use_fast_accum: bool,
        persistent_matmul: bool,
        device,
    ):
        # [FUNCTION DOC] Rowwise-scaled fp8 linear in bfloat16 output (float32 not
        # supported), optional bias, fast accumulate and persistent TMA flags, compiled in
        # max-autotune and compared against eager (5e-2 relative, 0.07 absolute). Weight
        # scale is transposed to a row vector over output features.
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with rowwise recipes, optional
            # bias, fast accumulate flag, bfloat16 output.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.RowWise,
                w_inverse_scale,
                ScalingType.RowWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=5e-2, atol=0.07)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device() or not is_big_gpu(),
        "Need device-side TMA support in Triton and max-autotune",
    )
    @onlyCUDA
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_rowwise_scaling_tma_template(
        self,
        shape: str,
        use_fast_accum: bool,
        device,
    ):
        # [FUNCTION DOC] Like the rowwise test but restricted to the device TMA Triton
        # template (CUDA, TMA capable and large GPU): compiles, checks the generated code
        # sets both scale recipes to the rowwise value, dtypes are bfloat16 and values
        # close to eager. No bias.
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with rowwise recipes, optional
            # fast accumulate, bfloat16 output.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.RowWise,
                w_inverse_scale,
                ScalingType.RowWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled, code = run_and_get_code(
                linear_compiled,
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

        FileCheck().check(
            f"SCALE_RECIPE_A : tl.constexpr = {ScalingType.RowWise.value}"
        ).run(code[0])
        FileCheck().check(
            f"SCALE_RECIPE_B : tl.constexpr = {ScalingType.RowWise.value}"
        ).run(code[0])
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        _get_torch_cuda_version() < (12, 9),
        "cuBLAS blockwise scaling added in CUDA 12.9",
    )
    @onlyCUDA
    @xfailIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0)
    )  # cuBLAS 128-element blockwise scaling is only supported for CC 9.0
    @parametrize("shape", ((16, 256, 256), (1024, 512, 1024), (32768, 4096, 4096)))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "scaling_block_sizes",
        ((1, 128, 128, 128), (1, 128, 1, 128), (128, 128, 1, 128)),
    )  # (BlockWise1x128, BlockWise128x128), (BlockWise1x128, BlockWise1x128), (BlockWise128x128, BlockWise1x128)
    def test_main_loop_scaling(
        self,
        shape: tuple[int, int, int],
        use_fast_accum: bool,
        scaling_block_sizes: tuple[int, int, int, int],
        device,
    ):
        # [FUNCTION DOC] Blockwise scaling through the device TMA Triton template on CUDA
        # 12.9+ (expected to fail except on compute capability 9.0). Parametrized by
        # shape, fast accumulate and block sizes for activation and weight (1x128 or
        # 128x128 recipes). Some combinations are skipped because of CI failures; XPU
        # skips fast accumulate. Quantizes blockwise, prepares scale layouts, runs eager
        # (expecting a RuntimeError about fast accum on SM90 when enabled), compiles,
        # checks the generated scale recipe constants and, without fast accumulate,
        # compares with eager (1e-2, 0.05).
        # (shape, use_fast_accum, scaling_block_sizes) combos disabled due to CI
        # failures; other combos still run. See the referenced issues.
        _disabled_combos = {
            ((16, 256, 256), False, (1, 128, 128, 128)),
            ((16, 256, 256), False, (1, 128, 1, 128)),
            ((16, 256, 256), False, (128, 128, 1, 128)),
            ((16, 256, 256), True, (1, 128, 128, 128)),
            ((16, 256, 256), True, (1, 128, 1, 128)),
            ((16, 256, 256), True, (128, 128, 1, 128)),
            ((1024, 512, 1024), False, (1, 128, 1, 128)),
            ((1024, 512, 1024), False, (128, 128, 1, 128)),
            ((1024, 512, 1024), True, (1, 128, 1, 128)),
            ((1024, 512, 1024), True, (128, 128, 1, 128)),
            ((32768, 4096, 4096), False, (1, 128, 1, 128)),
            ((32768, 4096, 4096), False, (128, 128, 1, 128)),
            ((32768, 4096, 4096), True, (1, 128, 1, 128)),
            ((32768, 4096, 4096), True, (128, 128, 1, 128)),
        }
        if (shape, use_fast_accum, scaling_block_sizes) in _disabled_combos:
            self.skipTest("disabled due to CI failures; see #190236")
        if "xpu" in device and use_fast_accum:
            self.skipTest("XPU does not support use_fast_accum=True for now")
        # Only bf16 output type is supported for non-tensorwise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        M, N, K = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

        am, ak, bn, bk = scaling_block_sizes

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_blockwise(
            w, dtype_float8, block_outer=bn, block_inner=bk
        )
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = _prepare_blockwise_scale(
            w_inverse_scale, bn, bk, transposed=True
        )

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_blockwise(
            x, dtype_float8, block_outer=am, block_inner=ak
        )
        x_inverse_scale = _prepare_blockwise_scale(
            x_inverse_scale, am, ak, transposed=False
        )

        recipe_x = (
            ScalingType.BlockWise1x128
            if (am, ak) == (1, 128)
            else ScalingType.BlockWise128x128
        )
        recipe_w = (
            ScalingType.BlockWise1x128
            if (bn, bk) == (1, 128)
            else ScalingType.BlockWise128x128
        )

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with the blockwise recipes
            # chosen by the enclosing test for each operand, optional fast accumulate,
            # bfloat16 output.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                recipe_x,
                w_inverse_scale,
                recipe_w,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        # BlockWise1x128 and BlockWise128x128 scaling modes are not compatible with fast_accum
        # Only take this branch on SM90 because other versions xfail everything
        if use_fast_accum and IS_SM90:
            with self.assertRaisesRegex(
                RuntimeError, "scaled_gemm doesn't support fast accum"
            ):
                y_eager = linear(
                    x_fp8,
                    x_inverse_scale,
                    w_t_fp8,
                    w_inverse_scale,
                    bias,
                )
        else:
            y_eager = linear(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

        with config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled, code = run_and_get_code(
                linear_compiled,
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

        # Verify that Inductor chooses the correct scaling recipes
        check_scale_recipe_a = (
            ScalingType.BlockWise1x128.value
            if (am, ak) == (1, 128)
            else ScalingType.BlockWise128x128.value
        )
        FileCheck().check(
            f"SCALE_RECIPE_A : tl.constexpr = {check_scale_recipe_a}"
        ).run(code[0])

        check_scale_recipe_b = (
            ScalingType.BlockWise1x128.value
            if (bn, bk) == (1, 128)
            else ScalingType.BlockWise128x128.value
        )
        FileCheck().check(
            f"SCALE_RECIPE_B : tl.constexpr = {check_scale_recipe_b}"
        ).run(code[0])

        self.assertEqual(y_compiled.dtype, dtype)
        if not use_fast_accum:
            self.assertEqual(y_eager.dtype, dtype)
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_tensorwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool, device
    ):
        # [FUNCTION DOC] Tensorwise-scaled bfloat16 linear over many M, K, N sizes
        # (including M not multiple of 16) with fast accumulate (off on XPU) and optional
        # persistent TMA. Compiles in max-autotune (default mode on ROCm gfx120 with M
        # below 16) and checks dtypes and closeness to eager.
        # alignment requirements: K and N divisible by 16
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        # xpu does not support fast_accum now
        if "xpu" in device:
            use_fast_accum = False
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with tensorwise recipes,
            # optional bias, fast accumulate flag, output in the test dtype.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.TensorWise,
                w_inverse_scale,
                ScalingType.TensorWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )

        # On gfx120x, autotuned kernels have issues with small M
        compile_mode = "max-autotune"
        if (
            torch.version.hip is not None
            and M < 16
            and torch.cuda.is_available()
            and "gfx120" in torch.cuda.get_device_properties(0).gcnArchName
        ):
            compile_mode = "default"

        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode=compile_mode
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=5e-2, atol=0.07)

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @torch._inductor.config.patch("emulate_precision_casts", True)
    def test_mx_fusion(self, device):
        # [FUNCTION DOC] Checks fusion of MX style quantization (amax, exponent
        # extraction, scale layout shuffles) feeding a custom fake scaled mm op registered
        # in a scoped library. Verifies the generated code has two kernel runs before the
        # op call, then for five seeds compares the op inputs recorded with and without
        # the loop index inversion fusion setting; the two runs must record identical
        # operands and scales. Raises AssertionError if the op was not called exactly
        # twice. Runs with precision-emulating casts enabled. CUDA and XPU.
        # use a device key for library registration
        device_type = torch.device(device).type
        device_dispatch_key = "CUDA" if device_type == "cuda" else "XPU"
        # Register fake_scaled_mm custom op scoped to this test
        with torch.library._scoped_library("test_fp8", "FRAGMENT") as lib:
            # Define the op schema
            lib.define(
                "fake_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, "
                "Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, "
                "bool use_fast_accum=False) -> Tensor"
            )
            input_values = []

            # Register CUDA/XPU implementation
            @torch.library.impl(lib, "fake_scaled_mm", device_dispatch_key)
            def fake_scaled_mm_impl(
                # [FUNCTION DOC] HELPER: device implementation of the custom op. Appends
                # operands and scales to the shared list input_values (nonlocal), computes
                # the sum of the two matrices in float32 (not a real matmul), adds bias if
                # given and casts to the output dtype, defaulting to bfloat16.
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                bias=None,
                scale_result=None,
                out_dtype=None,
                use_fast_accum=False,
            ):
                """Software-emulated scaled_mm for testing without CUDA 12.8"""
                out_dtype = out_dtype or torch.bfloat16
                # just using add, because without real dtypes,
                # was seeing overflow/instability
                nonlocal input_values
                input_values.append((mat_a, mat_b, scale_a, scale_b))
                result = mat_a.to(torch.float32) + mat_b.to(torch.float32)
                if bias is not None:
                    result = result + bias.to(torch.float32)
                return result.to(out_dtype)

            # Register fake implementation
            @torch.library.impl(lib, "fake_scaled_mm", "Meta")
            def fake_scaled_mm_meta(
                # [FUNCTION DOC] HELPER: fake/meta implementation. Checks the inner dims
                # of the two matrices agree via a torch check and returns an empty tensor
                # of M by N with the output dtype (default bfloat16) on the first
                # matrix's device.
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                bias=None,
                scale_result=None,
                out_dtype=None,
                use_fast_accum=False,
            ):
                """FakeTensor implementation"""
                out_dtype = out_dtype or torch.bfloat16
                M, K = mat_a.shape
                K2, N = mat_b.shape
                torch._check(
                    K == K2,
                    lambda: f"Incompatible shapes: {mat_a.shape} @ {mat_b.shape}",
                )
                return torch.empty((M, N), dtype=out_dtype, device=mat_a.device)

            def forward(
                # [FUNCTION DOC] HELPER: traced graph body (as dumped from an Inductor
                # graph) that quantizes both float32 inputs in blocks of 32 to
                # float8_e4m3fn with e8m0 power of two scales, rearranges the scales into
                # a blocked layout, and calls the custom fake scaled mm op, returning a
                # one element tuple with its result.
                arg0_1,
                arg1_1,
            ):
                view = torch.ops.aten.reshape.default(arg0_1, [8192, 256, 32])
                abs_1 = torch.ops.aten.abs.default(view)
                amax = torch.ops.aten.amax.default(abs_1, [-1])
                unsqueeze = torch.ops.aten.unsqueeze.default(amax, -1)
                view_1 = torch.ops.aten.view.dtype(unsqueeze, torch.int32)
                bitwise_right_shift = torch.ops.aten.bitwise_right_shift.Tensor_Scalar(
                    view_1, 23
                )
                bitwise_and = torch.ops.aten.bitwise_and.Scalar(
                    bitwise_right_shift, 255
                )
                sub = torch.ops.aten.sub.Tensor(bitwise_and, 127)
                sub_1 = torch.ops.aten.sub.Tensor(sub, 8)
                clamp_min = torch.ops.aten.clamp_min.default(sub_1, -127)
                clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 128)
                add = torch.ops.aten.add.Tensor(clamp_max, 127)
                convert_element_type = torch.ops.prims.convert_element_type.default(
                    add, torch.uint8
                )
                isnan = torch.ops.aten.isnan.default(unsqueeze)
                scalar_tensor = torch.ops.aten.scalar_tensor.default(
                    255, dtype=torch.uint8, layout=torch.strided, device=device
                )
                where = torch.ops.aten.where.self(
                    isnan, scalar_tensor, convert_element_type
                )
                convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                    where, torch.int32
                )
                bitwise_left_shift = torch.ops.aten.bitwise_left_shift.Tensor_Scalar(
                    convert_element_type_1, 23
                )
                view_2 = torch.ops.aten.view.dtype(bitwise_left_shift, torch.float32)
                clamp_min_1 = torch.ops.aten.clamp_min.default(
                    view_2, 1.1754943508222875e-38
                )
                div = torch.ops.aten.div.Tensor(view, clamp_min_1)
                clamp_min_2 = torch.ops.aten.clamp_min.default(div, -448.0)
                clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_2, 448.0)
                convert_element_type_2 = torch.ops.prims.convert_element_type.default(
                    clamp_max_1, torch.float8_e4m3fn
                )
                view_3 = torch.ops.aten.reshape.default(
                    convert_element_type_2, [8192, 8192]
                )
                convert_element_type_2 = None
                view_4 = torch.ops.aten.view.dtype(where, torch.float8_e8m0fnu)
                squeeze = torch.ops.aten.squeeze.dim(view_4, -1)

                view_5 = torch.ops.aten.reshape.default(arg1_1, [8192, 256, 32])
                abs_2 = torch.ops.aten.abs.default(view_5)
                amax_1 = torch.ops.aten.amax.default(abs_2, [-1])
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(amax_1, -1)
                view_6 = torch.ops.aten.view.dtype(unsqueeze_1, torch.int32)
                bitwise_right_shift_1 = (
                    torch.ops.aten.bitwise_right_shift.Tensor_Scalar(view_6, 23)
                )
                bitwise_and_1 = torch.ops.aten.bitwise_and.Scalar(
                    bitwise_right_shift_1, 255
                )
                sub_2 = torch.ops.aten.sub.Tensor(bitwise_and_1, 127)
                sub_3 = torch.ops.aten.sub.Tensor(sub_2, 8)
                clamp_min_3 = torch.ops.aten.clamp_min.default(sub_3, -127)
                clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_3, 128)
                add_1 = torch.ops.aten.add.Tensor(clamp_max_2, 127)
                convert_element_type_3 = torch.ops.prims.convert_element_type.default(
                    add_1, torch.uint8
                )
                isnan_1 = torch.ops.aten.isnan.default(unsqueeze_1)
                unsqueeze_1 = None
                scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
                    255, dtype=torch.uint8, layout=torch.strided, device=device
                )
                where_1 = torch.ops.aten.where.self(
                    isnan_1, scalar_tensor_1, convert_element_type_3
                )
                convert_element_type_4 = torch.ops.prims.convert_element_type.default(
                    where_1, torch.int32
                )
                bitwise_left_shift_1 = torch.ops.aten.bitwise_left_shift.Tensor_Scalar(
                    convert_element_type_4, 23
                )
                convert_element_type_4 = None
                view_7 = torch.ops.aten.view.dtype(bitwise_left_shift_1, torch.float32)
                bitwise_left_shift_1 = None
                clamp_min_4 = torch.ops.aten.clamp_min.default(
                    view_7, 1.1754943508222875e-38
                )
                div_1 = torch.ops.aten.div.Tensor(view_5, clamp_min_4)
                clamp_min_5 = torch.ops.aten.clamp_min.default(div_1, -448.0)
                clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_5, 448.0)
                convert_element_type_5 = torch.ops.prims.convert_element_type.default(
                    clamp_max_3, torch.float8_e4m3fn
                )
                view_8 = torch.ops.aten.reshape.default(
                    convert_element_type_5, [8192, 8192]
                )
                view_9 = torch.ops.aten.view.dtype(where_1, torch.float8_e8m0fnu)
                squeeze_1 = torch.ops.aten.squeeze.dim(view_9, -1)

                permute = torch.ops.aten.permute.default(view_8, [1, 0])

                view_13 = torch.ops.aten.reshape.default(squeeze, [64, 128, 64, 4])
                permute_2 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3])
                clone = torch.ops.aten.clone.default(
                    permute_2, memory_format=torch.contiguous_format
                )
                view_14 = torch.ops.aten.reshape.default(clone, [4096, 4, 32, 4])
                permute_3 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3])
                clone_1 = torch.ops.aten.clone.default(
                    permute_3, memory_format=torch.contiguous_format
                )
                view_15 = torch.ops.aten.reshape.default(clone_1, [4096, 32, 16])

                view_16 = torch.ops.aten.reshape.default(view_15, [2097152])

                view_18 = torch.ops.aten.reshape.default(squeeze_1, [64, 128, 64, 4])
                permute_5 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3])
                clone_2 = torch.ops.aten.clone.default(
                    permute_5, memory_format=torch.contiguous_format
                )
                view_19 = torch.ops.aten.reshape.default(clone_2, [4096, 4, 32, 4])
                permute_6 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3])
                clone_3 = torch.ops.aten.clone.default(
                    permute_6, memory_format=torch.contiguous_format
                )
                view_20 = torch.ops.aten.reshape.default(clone_3, [4096, 32, 16])

                view_21 = torch.ops.aten.reshape.default(view_20, [2097152])

                _scaled_mm = torch.ops.test_fp8.fake_scaled_mm.default(
                    view_3, permute, view_16, view_21, None, None, torch.float32
                )
                return (_scaled_mm,)

            # Run with largest shape
            M, K, N = 8192, 8192, 8192

            A = torch.randn(M, K, dtype=torch.float32, device=device)
            B = torch.randn(K, N, dtype=torch.float32, device=device)
            f_c = torch.compile(fullgraph=True)(forward)

            _, code = run_and_get_code(f_c, A, B)

            FileCheck().check(".run(").check(".run(").check("fake_scaled_mm").run(
                code[0]
            )

            for seed in range(5):
                input_values.clear()
                torch.manual_seed(seed)
                # without dividing, outputs get way too large
                A = torch.randn(M, K, dtype=torch.float32, device=device)
                B = torch.randn(K, N, dtype=torch.float32, device=device)

                # Uses fake_scaled_mm custom op (no CUDA 12.8 needed!)
                torch._dynamo.reset()
                torch.compile(forward)(A, B)

                torch._dynamo.reset()
                with config.patch({"loop_index_inversion_in_fusion": False}):
                    torch.compile(forward)(A, B)

                if len(input_values) != 2:
                    raise AssertionError
                for i in range(4):
                    self.assertEqual(
                        input_values[0][i],
                        input_values[1][i],
                        msg=lambda msg: f"{msg}\nidx {i} seed {seed}",
                    )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyCUDA
    def test_mxfp8_dtype_view_indexer_e2e(self, device):
        # [FUNCTION DOC] End to end check of indexing through a dtype view of e8m0 scales.
        # Registers a custom op (clone on CUDA, empty-like on meta) and a custom lowering
        # that realizes its input, freezes layout, builds an indexer and clones. The
        # registry of lowerings is patched so it is restored afterwards. Compiles an MX
        # scale computation followed by the op and checks the shape, the e8m0 dtype and
        # byte-identical results with eager. CUDA only.
        with (
            torch.library._scoped_library("test_fp8", "FRAGMENT") as lib,
            mock.patch.dict(inductor_lowering.lowerings),
        ):
            lib.define("dtype_view_indexing_consumer(Tensor x) -> Tensor")
            torch.library.impl(lib, "dtype_view_indexing_consumer", "CUDA")(
                lambda x: x.clone()
            )
            torch.library.impl(lib, "dtype_view_indexing_consumer", "Meta")(
                lambda x: torch.empty_like(x)
            )

            def lowering(x):
                # [FUNCTION DOC] HELPER: lowering for the custom op. Realizes the input,
                # freezes its layout, calls its indexer on symbolic index variables (to
                # exercise dtype view indexing), and returns a clone lowering of it.
                x.realize()
                ir.as_storage_and_layout(x.unwrap_view(), freeze=True)
                index = [
                    sympy_index_symbol_with_prefix(SymT.INDEX, i)
                    for i in range(len(x.get_size()))
                ]
                x.make_indexer()(index)
                return lowering_clone(x)

            register_lowering(
                torch.ops.test_fp8.dtype_view_indexing_consumer.default,
                type_promotion_kind=None,
            )(lowering)

            def fn(x):
                # [FUNCTION DOC] HELPER: reshapes input to 128 by 4 by 32, computes block
                # amax, derives a power of two exponent as uint8 (255 for NaN), views it
                # as e8m0 scale, passes it through the custom op and squeezes the
                # last dim.
                x = x.reshape(128, 4, 32)
                amax = torch.amax(torch.abs(x), -1, keepdim=True).float()
                descale = amax / torch.finfo(torch.float8_e4m3fn).max
                exponent = torch.where(
                    torch.isnan(descale),
                    torch.full((), 255, dtype=torch.uint8, device=x.device),
                    (
                        torch.clamp(torch.ceil(torch.log2(descale)), min=-127, max=127)
                        + 127
                    ).to(torch.uint8),
                )
                scale = exponent.view(torch.float8_e8m0fnu)
                return torch.ops.test_fp8.dtype_view_indexing_consumer.default(
                    scale
                ).squeeze(-1)

            x = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
            expected = fn(x)
            actual = torch.compile(fn, fullgraph=True)(x)

        self.assertEqual(actual.shape, (128, 4))
        self.assertEqual(actual.dtype, torch.float8_e8m0fnu)
        self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool, device
    ):
        # [FUNCTION DOC] Rowwise-scaled bfloat16 linear with bias over many M, K, N sizes
        # (fast accumulate off on XPU, optional persistent TMA), compiled in max-autotune
        # (default mode on ROCm gfx120 with M below 16) and compared with eager (1e-2
        # relative, 0.07 absolute) along with dtype checks.
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        # xpu does not support fast_accum now
        if "xpu" in device:
            use_fast_accum = False
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: functional scaled_mm with rowwise recipes, bias, fast
            # accumulate flag and bfloat16 output.
            y = torch.nn.functional.scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                ScalingType.RowWise,
                w_inverse_scale,
                ScalingType.RowWise,
                bias=bias,
                output_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )

        # On gfx120x, autotuned kernels have issues with small M
        compile_mode = "max-autotune"
        if (
            torch.version.hip is not None
            and M < 16
            and torch.cuda.is_available()
            and "gfx120" in torch.cuda.get_device_properties(0).gcnArchName
        ):
            compile_mode = "default"

        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode=compile_mode
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_scaled_mm_v2_no_swizzle(self, device):
        # [FUNCTION DOC] Regression test: the v2 scaled mm must not crash when swizzle
        # arguments are omitted or explicitly set to no swizzle. Runs tensorwise scaled_mm
        # both ways on small fp8 operands, checks output shape and that both outputs are
        # equal. CUDA and XPU.
        """Regression test: _scaled_mm_v2 must not segfault when swizzle is
        omitted (empty) or explicitly set to NO_SWIZZLE."""
        from torch.nn.functional import SwizzleType

        m, k, n = 32, 64, 16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16).to(dtype_float8)
        b = torch.randn(n, k, device=device, dtype=torch.bfloat16).to(dtype_float8).t()
        scale_a = torch.ones(1, device=device)
        scale_b = torch.ones(1, device=device)

        # swizzle omitted (None -> empty list in C++)
        out_no_swizzle = scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TensorWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TensorWise,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(out_no_swizzle.shape, (m, n))

        # swizzle explicitly NO_SWIZZLE
        out_explicit = scaled_mm(
            a,
            b,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TensorWise,
            swizzle_a=SwizzleType.NO_SWIZZLE,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TensorWise,
            swizzle_b=SwizzleType.NO_SWIZZLE,
            output_dtype=torch.bfloat16,
        )
        self.assertEqual(out_no_swizzle, out_explicit)

    @onlyCUDA
    @skipIfRocm  # ROCm MX gemm requires NO_SWIZZLE; swizzle is NVIDIA-only
    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, "Not supported on non B200")
    def test_scaled_mm_v2_swizzle_compile(self, device):
        # [FUNCTION DOC] Checks that MX block scaled (1x32 e8m0 scales swizzled with the
        # 32-4-4 layout) scaled_mm compiles with fullgraph, matches eager, and that the
        # generated code calls the ATen v2 kernel, i.e. no template handles swizzled
        # scales. Uses identity matrices quantized to fp8. CUDA, MX capable GPU, not ROCm.
        """Swizzled scales (MX recipes on NVIDIA require SWIZZLE_32_4_4) are
        not supported by any inductor template; the lowering must fall back
        to the ATen kernel instead of failing compile."""
        from torch.nn.functional import SwizzleType

        M, K, N = 128, 128, 128
        BLOCK_SIZE = 32
        A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
        B_ref = torch.eye(N, device=device, dtype=torch.bfloat16)
        A = A_ref.to(torch.float8_e4m3fn)
        B = B_ref.to(torch.float8_e4m3fn)
        A_scale = torch.full(
            (M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        B_scale = torch.full(
            (N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        A_scale = to_blocked(A_scale)
        B_scale = to_blocked(B_scale)

        def fn(A, B, A_scale, B_scale):
            # [FUNCTION DOC] HELPER: functional scaled_mm with 1x32 blockwise recipes and
            # 32-4-4 swizzles for both operands, bfloat16 output; B is transposed.
            return scaled_mm(
                A,
                B.t(),
                A_scale,
                ScalingType.BlockWise1x32,
                B_scale,
                ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.bfloat16,
            )

        y_eager = fn(A, B, A_scale, B_scale)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        y_compiled, (code,) = run_and_get_code(compiled, A, B, A_scale, B_scale)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)
        # The swizzled path must use the ATen fallback, not a generated kernel
        FileCheck().check("_scaled_mm_v2").run(code)

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, "Not supported on non B200")
    def test_mx_fp8_max_autotune(self, device):
        # [FUNCTION DOC] MX fp8 (e8m0 1x32 scales) max-autotune compile of the private
        # scaled matmul on identity matrices; scales are blocked for CUDA and the weight
        # scale is transposed for XPU. Checks bfloat16 dtypes and closeness to eager
        # (1e-2, 0.07). Needs MX capable GPU; CUDA and XPU.
        # K must match the operands, which are eye(M)/eye(N) (i.e. 128 wide);
        # using a smaller K would size the MX 1x32 scales for fewer K-blocks
        # than the data has (XPU rejects this; CUDA only masks it via the
        # to_blocked() zero-padding below).
        M, K, N = 128, 128, 128
        BLOCK_SIZE = 32
        dtype = torch.bfloat16
        A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
        B_ref = torch.eye(N, device=device, dtype=torch.bfloat16)
        A = A_ref.to(torch.float8_e4m3fn)
        B = B_ref.to(torch.float8_e4m3fn)
        A_scale = torch.full(
            (M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        B_scale = torch.full(
            (N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        if "cuda" in device:
            A_scale = to_blocked(A_scale)
            B_scale = to_blocked(B_scale)
        elif "xpu" in device:
            B_scale = B_scale.t()

        def linear(A, B, A_scale, B_scale):
            # [FUNCTION DOC] HELPER: private scaled matmul of A with transposed B,
            # bfloat16 output, no fast accumulate.
            y = torch._scaled_mm(
                A,
                B.t(),
                A_scale,
                B_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            return y

        y_eager = linear(A, B, A_scale, B_scale)

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        y_compiled = linear_compiled(A, B, A_scale, B_scale)
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_unacceptable_input_dims(self, device):
        # [FUNCTION DOC] Compiling a tensorwise scaled matmul whose K dim (15) is not a
        # multiple of 16 must raise TorchRuntimeError, and the message must contain the
        # divisibility complaint with the actual size. Type checking comes from the meta
        # registrations. CUDA and XPU.
        # for compiled ops, type checking is in torch/_meta_registrations.py
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        # xpu does not support fast_accum now
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False
        M, K, N = 64, 15, 2048  # K needs to be a multiple of 16
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        def linear(x, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: quantizes the activation tensorwise inside the
            # function and calls the private scaled matmul with the bias and fast
            # accumulate flag.
            x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)
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

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        with self.assertRaises(torch._dynamo.exc.TorchRuntimeError) as cm:
            linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue(
            f"Expected self.size(1) to be divisible by 16, but got self.size(1)={K}"
            in str(cm.exception)
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu", "cpu"])
    def test_unacceptable_scale_dims_rowwise_scaling(self, device):
        # [FUNCTION DOC] Compiling a rowwise scaled matmul with the activation and weight
        # scales swapped (weight scale transposed) must raise TorchRuntimeError whose
        # message contains the invalid scaling configuration text.
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        M, K, N = 233, 32, 128
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        # xpu does not support fast_accum now
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False

        def linear(x, w_t_fp8, w_inverse_scale, bias):
            # [FUNCTION DOC] HELPER: quantizes the activation rowwise inside the function
            # and calls the private scaled matmul with the scales deliberately swapped.
            x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                w_inverse_scale.t(),  # testing with w and x scales switched
                x_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        with self.assertRaises(torch._dynamo.exc.TorchRuntimeError) as cm:
            linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue("Invalid scaling configuration." in str(cm.exception))


class TestE8M0ToFloat(TestCase):
    # [CLASS DOC] Tests that converting the e8m0 scale dtype to floating types compiles
    # correctly and generates the expected code. Instantiated for CPU and CUDA only.
    @parametrize(
        "dtype",
        (torch.float64, torch.float32, torch.float16, torch.bfloat16),
    )
    @parametrize("direct_input", (False, True))
    @onlyOn(["cpu", "cuda"])
    def test_conversion(self, device, dtype, direct_input):
        # [FUNCTION DOC] For each target float dtype and for input given either as e8m0
        # directly or as raw uint8 viewed as e8m0, compiles a conversion of eight edge
        # byte values (zero, near bias, max, NaN code) with fullgraph and checks equality
        # with eager, plus bit identity for float32.
        raw = torch.tensor(
            [0, 1, 126, 127, 128, 129, 254, 255],
            device=device,
            dtype=torch.uint8,
        )
        if direct_input:
            inp = raw.view(torch.float8_e8m0fnu)

            def fn(value):
                # [FUNCTION DOC] HELPER: converts the e8m0 input to the target dtype.
                return value.to(dtype)

        else:
            inp = raw

            def fn(value):
                # [FUNCTION DOC] HELPER: views the uint8 input as e8m0 and converts it to
                # the target dtype.
                return value.view(torch.float8_e8m0fnu).to(dtype)

        expected = fn(inp)
        actual = torch.compile(fn, fullgraph=True)(inp)
        self.assertEqual(actual, expected)
        if dtype is torch.float32:
            self.assertEqual(actual.view(torch.int32), expected.view(torch.int32))

    @parametrize("direct_input", (False, True))
    @onlyOn(["cpu", "cuda"])
    def test_codegen(self, device, direct_input):
        # [FUNCTION DOC] Compiles the e8m0-to-float32 conversion on edge byte values,
        # checks bit identical results to eager, and inspects generated source: it must
        # contain two magic constants for the denormal and NaN encodings; on CUDA it must
        # use a bitcast and not the ldexp libdevice function; on CPU it must go through
        # the cpp binding and mention the C++ e8m0 type when the input was e8m0 directly.
        raw = torch.tensor(
            [0, 1, 126, 127, 128, 129, 254, 255],
            device=device,
            dtype=torch.uint8,
        )
        if direct_input:
            inp = raw.view(torch.float8_e8m0fnu)

            def fn(value):
                # [FUNCTION DOC] HELPER: converts the e8m0 input to float32.
                return value.float()

        else:
            inp = raw

            def fn(value):
                # [FUNCTION DOC] HELPER: views the uint8 input as e8m0 and converts
                # to float32.
                return value.view(torch.float8_e8m0fnu).float()

        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), inp)
        expected = fn(inp)
        self.assertEqual(actual.view(torch.int32), expected.view(torch.int32))

        code_str = "\n".join(code)
        self.assertIn("4194304", code_str)
        self.assertIn("2139095041", code_str)
        if torch.device(device).type == "cuda":
            self.assertIn("bitcast=True", code_str)
            self.assertNotIn("libdevice.ldexp", code_str)
        else:
            self.assertIn("async_compile.cpp_pybinding", code_str)
            if direct_input:
                self.assertIn("at::Float8_e8m0fnu", code_str)


class TestCvtE8M0RceilGating(TestCase):
    # [CLASS DOC] Tests the platform gate for the e8m0 round-up conversion; runs on any
    # platform since it mocks the device queries.
    def test_nvidia_sm100_gate_excludes_rocm_gfx1101(self):
        # [FUNCTION DOC] With CUDA reported available at capability 11.0 but a ROCm
        # version string set, the NVIDIA SM100-or-later predicate must return False.
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(11, 0)
            ),
            mock.patch.object(torch.version, "hip", "7.2.26015"),
        ):
            self.assertFalse(utils.is_nvidia_sm100_or_later())


@unittest.skipIf(
    not utils.is_nvidia_sm100_or_later(),
    "Requires NVIDIA SM100+ (Blackwell) for PTX instruction",
)
class TestCvtE8M0Rceil(TestCase):
    # [CLASS DOC] Tests of the e8m0 round-up conversion primitive and the log2-ceil
    # pattern replacement using the PTX instruction. Skipped unless running on NVIDIA
    # SM100 or later. All tests use CUDA tensors.
    """Tests for cvt_e8m0_rceil prim with PTX lowering on NVIDIA Blackwell."""

    def test_correctness(self):
        # [FUNCTION DOC] Compares compiled and eager results of the conversion primitive
        # over symmetric ranges of integers for float32, bfloat16 and float16.
        """Test correctness for various dtypes."""

        def fn(inp):
            # [FUNCTION DOC] HELPER: applies the e8m0 round-up conversion primitive.
            return inductor_prims.cvt_e8m0_rceil(inp)

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            inp = torch.cat(
                [
                    torch.arange(-1024, 0, device="cuda", dtype=dtype),
                    torch.arange(1, 1025, device="cuda", dtype=dtype),
                ]
            )
            eager_result = fn(inp)
            compiled_result = torch.compile(fn)(inp)
            self.assertEqual(compiled_result, eager_result)

    def test_pattern_match(self):
        # [FUNCTION DOC] Initializes the miscellaneous patterns, then checks a log2, ceil,
        # clamp, bias add, uint8 cast sequence compiled matches eager on a few values.
        """Test that the log2+ceil pattern gets matched and replaced."""
        _misc_patterns_init()

        E8M0_BIAS = 127

        def fn_with_log2_pattern(inp):
            # [FUNCTION DOC] HELPER: computes the biased, clamped ceiling of log2
            # as uint8.
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5], device="cuda", dtype=torch.float32
        )

        eager_result = fn_with_log2_pattern(inp)
        compiled_result = torch.compile(fn_with_log2_pattern)(inp)
        self.assertEqual(compiled_result, eager_result)

    def test_ptx_code_generation(self):
        # [FUNCTION DOC] Compiles the conversion primitive and asserts the generated
        # source contains the PTX cvt instruction for round-positive ue8m0x2 from f32.
        """Test that PTX instruction appears in generated code."""

        def fn(inp):
            # [FUNCTION DOC] HELPER: applies the e8m0 round-up conversion primitive.
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(32, device="cuda", dtype=torch.float32)
        compiled_fn = torch.compile(fn)
        _, code = run_and_get_code(compiled_fn, inp)

        code_str = "\n".join(code)
        self.assertIn("cvt.rp.satfinite.ue8m0x2.f32", code_str)

    def test_log2_pattern_near_power_of_two(self):
        # [FUNCTION DOC] Builds values one ULP above powers of two from 1 to 128 and
        # checks the compiled log2-ceil encoding returns the exponent plus one plus bias,
        # in uint8.
        """Values within 1 ULP above a power of 2 must ceil to the next integer.

        Software log2 may round a value like 2^e + 1 ULP down to exactly e
        (the ULP of log2's output at e is larger than the true fractional part),
        so ceil(log2(x)) would incorrectly return e instead of e+1.
        The PTX/bit-manipulation replacement must always return the correct value.
        """
        _misc_patterns_init()
        E8M0_BIAS = 127

        def fn(inp):
            # [FUNCTION DOC] HELPER: biased, clamped ceiling of log2 as uint8.
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        # Build values that are exactly 1 ULP above each power of 2 from 2^0..2^7.
        # For these, ceil(log2(x)) must equal the exponent + 1.
        powers = [float(2**e) for e in range(8)]
        one_ulp_above = [
            torch.nextafter(torch.tensor(p), torch.tensor(float("inf"))).item()
            for p in powers
        ]
        inp = torch.tensor(one_ulp_above, device="cuda", dtype=torch.float32)

        expected = torch.tensor(
            [E8M0_BIAS + e + 1 for e in range(8)], dtype=torch.uint8, device="cuda"
        )
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, expected)


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "Requires CUDA + Triton")
@unittest.skipIf(
    utils.is_nvidia_sm100_or_later(),
    "Pre-NVIDIA-SM100 path: uses bit-manipulation fallback",
)
class TestE8M0Log2PatternBitManip(TestCase):
    # [CLASS DOC] Tests of the log2-ceil pattern replacement on hardware before SM100,
    # where Inductor reads the float exponent bits instead of software log2 (gh-178045).
    # Requires CUDA and Triton and is skipped on SM100 or later.
    """Tests for the e8m0_rceil_log2 pattern on pre-SM100 hardware.

    On hardware without the SM100 PTX instruction, Inductor replaces the
    software log2+ceil sequence with an equivalent IEEE 754 bit-manipulation
    that reads the biased exponent directly.  This avoids 1 ULP errors in
    software log2 near exact powers of 2 (gh-178045).
    """

    def test_pattern_fires_and_is_correct(self):
        # [FUNCTION DOC] Initializes misc patterns and checks the compiled biased ceil-
        # log2 uint8 encoding equals eager on assorted values including powers of two
        # and fractions.
        """The log2+ceil pattern should be matched and produce correct uint8."""
        _misc_patterns_init()
        E8M0_BIAS = 127

        def fn(inp):
            # [FUNCTION DOC] HELPER: biased, clamped ceiling of log2 as uint8.
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5, 0.5, 0.25], device="cuda", dtype=torch.float32
        )
        eager_result = fn(inp)
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, eager_result)

    def test_correct_for_values_one_ulp_above_power_of_two(self):
        # [FUNCTION DOC] Checks the compiled encoding for values one ULP above powers of
        # two from 1 to 128 equals the exponent plus one plus bias.
        """Values 1 ULP above a power of 2 must produce the correct (higher) exponent.

        Software log2 for x = 2^e + 1 ULP may round to exactly e when e is
        large (the ULP of the log2 output exceeds the true fractional part).
        The bit-manipulation replacement reads the exponent field directly and
        is always correct.
        """
        E8M0_BIAS = 127

        def fn(inp):
            # [FUNCTION DOC] HELPER: biased, clamped ceiling of log2 as uint8.
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        powers = [float(2**e) for e in range(8)]
        one_ulp_above = [
            torch.nextafter(torch.tensor(p), torch.tensor(float("inf"))).item()
            for p in powers
        ]
        inp = torch.tensor(one_ulp_above, device="cuda", dtype=torch.float32)

        expected = torch.tensor(
            [E8M0_BIAS + e + 1 for e in range(8)], dtype=torch.uint8, device="cuda"
        )
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, expected)

    def test_regression_gh178045_encoding_correctness(self):
        # [FUNCTION DOC] Regression for gh-178045 using one ULP above powers of two from 1
        # to 64, after resetting dynamo state; the compiled result must equal the exponent
        # plus one plus bias, with a failure message citing the issue.
        """Regression for gh-178045: encoding must be correct for inputs near power-of-2.

        When a compiled kernel (e.g. fused GELU) produces a float value that is
        just above a power of 2 (say 2^e + 1 ULP), the software log2+ceil
        pipeline may give the WRONG result because float32 log2 rounds the
        result down to exactly e, making ceil return e instead of the correct
        e+1.  The bit-manipulation replacement always returns the correct value.

        This test specifically validates that the bit-manipulation codepath gives
        mathematically correct results for such boundary values (i.e. that it
        fixes the correctness bug distinct from the GELU input discrepancy).
        """
        E8M0_BIAS = 127

        # Craft a function that mimics what fused GELU might produce: an
        # output that is exactly 1 ULP above a series of powers of 2.
        def encode_fn(inp):
            # [FUNCTION DOC] HELPER: the log2-ceil encoding to uint8 that the
            # pattern rewrites.
            # This is the e8m0_rceil_log2 pattern that gets rewritten
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        # For each exponent e in [0..6], the value 2^e + 1 ULP has exponent e
        # and a non-zero mantissa, so the correct e8m0 ceiling is e+1.
        # Software log2+ceil may return e for large e (the fractional part of
        # log2 becomes smaller than the ULP of log2's output at that magnitude).
        powers = [float(2**e) for e in range(7)]
        one_ulp_above = [
            torch.nextafter(torch.tensor(p), torch.tensor(float("inf"))).item()
            for p in powers
        ]
        inp = torch.tensor(one_ulp_above, device="cuda", dtype=torch.float32)

        # The correct e8m0 encoding for "1 ULP above 2^e" is E8M0_BIAS + (e+1)
        expected = torch.tensor(
            [E8M0_BIAS + e + 1 for e in range(7)], dtype=torch.uint8, device="cuda"
        )

        torch._dynamo.reset()
        compiled_result = torch.compile(encode_fn)(inp)
        self.assertEqual(
            compiled_result,
            expected,
            msg="Bit-manipulation replacement must return the correct e8m0 ceiling "
            "for values 1 ULP above a power of 2 (gh-178045 regression).",
        )


instantiate_device_type_tests(TestFP8Types, globals(), allow_xpu=True)
instantiate_device_type_tests(TestFP8Lowering, globals(), allow_xpu=True)
instantiate_device_type_tests(TestE8M0ToFloat, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON or HAS_CPU:
        run_tests()
