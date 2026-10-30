# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor's NVIDIA Universal GEMM (NVGEMM, cutlass_api based) backend
# and related flex_gemm higher-order op, scheduling and heuristics. Contains module helpers that
# build test tensors in several memory layouts and a standard autotune config, and test classes
# that compile matmul, bmm, addmm, grouped and scaled (MXFP8, NVFP4) GEMMs with the NVGEMM
# backend on Blackwell GPUs and compare against eager results or inspect generated code. The
# main test class is skipped unless the cutlass_api library is available and the device is a
# datacenter Blackwell part. No significant import-time side effects apart from imports.


import threading
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from torch._higher_order_ops import flex_gemm
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
    EPILOGUE_FN_NAME,
)
from torch._inductor.heuristics.template.nv_universal_gemm import (
    HeuristicConfig,
    NVUniversalGemmHeuristics,
)
from torch._inductor.scheduler import Scheduler
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    ceildiv,
    ensure_nv_universal_gemm_available,
    ensure_nvmatmul_heuristics_available,
    run_and_get_code,
)
from torch.testing._internal.common_utils import (
    dtype_name,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_reduction_ordering


def _round_up(x, multiple):
    # [FUNCTION DOC] HELPER: Rounds the first argument up to the nearest multiple of the second
    # using integer arithmetic and returns it.
    return ((x + multiple - 1) // multiple) * multiple


def _prep_k(K, scale_size):
    # [FUNCTION DOC] HELPER: Number of scale blocks along K padded for the swizzled scale layout:
    # the ceiling division of K by the scale block size, rounded up to a multiple of four.
    """Prepare K dimension for swizzle requirements (round up ceildiv to multiple of 4)."""
    return _round_up(ceildiv(K, scale_size), 4)


def _create_tensor_with_layout(layout, rows, cols, dtype, device="cuda"):
    # [FUNCTION DOC] HELPER: Builds a two dimensional test tensor of the given rows and columns on
    # the given device (default cuda) with the requested dtype, in one of four layouts: contiguous
    # (fresh tensor), aligned_offset (strided view into a larger storage starting at a 16 byte
    # aligned offset), view (a flat random tensor reshaped), padded (strided view with row pitch
    # eight elements wider than the columns). Values are random normal for floating dtypes, random
    # small integers (minus one to one) cast for float8, and random bytes reinterpreted for packed
    # float4. Raises ValueError for an unknown layout.
    """Create a tensor with the specified layout and dtype.

    Supports float16, bfloat16, float8_e4m3fn, and float4_e2m1fn_x2.
    """
    is_fp4 = dtype == torch.float4_e2m1fn_x2
    is_fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    def _make_flat(n):
        # [FUNCTION DOC] HELPER: Returns a random one dimensional tensor of n elements on the
        # enclosing device with the enclosing dtype, generated according to its kind (packed float4
        # from random bytes, float8 from small integers, otherwise normal).
        if is_fp4:
            return torch.randint(0, 256, (n,), device=device, dtype=torch.uint8).view(
                torch.float4_e2m1fn_x2
            )
        elif is_fp8:
            return torch.randint(-1, 2, (n,), device=device).to(dtype)
        else:
            return torch.randn(n, device=device, dtype=dtype)

    if layout == "contiguous":
        if is_fp4:
            return torch.randint(
                0, 256, (rows, cols), device=device, dtype=torch.uint8
            ).view(torch.float4_e2m1fn_x2)
        elif is_fp8:
            return torch.randint(-1, 2, (rows, cols), device=device).to(dtype)
        else:
            return torch.randn(rows, cols, device=device, dtype=dtype)
    elif layout == "aligned_offset":
        storage = _make_flat(rows * cols + 512)
        offset = 16 // storage.element_size()
        return torch.as_strided(storage[offset:], (rows, cols), (cols, 1))
    elif layout == "view":
        return _make_flat(rows * cols).view(rows, cols)
    elif layout == "padded":
        row_pitch = cols + 8
        storage = _make_flat(rows * row_pitch)
        return torch.as_strided(storage, (rows, cols), (row_pitch, 1))
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _nvgemm_config(**overrides):
    # [FUNCTION DOC] HELPER: Returns a config mapping for tests: autotuning enabled, the only GEMM
    # backend NVGEMM, at most three profiled configs, and ATen fallback disabled. Any keyword
    # overrides are merged on top and win. Returns a fresh dictionary each call.
    """Standard NVGEMM test config. Always disables ATen fallback."""
    cfg = {
        "max_autotune": True,
        "max_autotune_gemm_backends": "NVGEMM",
        "nvgemm_max_profiling_configs": 3,
        "autotune_fallback_to_aten": False,
    }
    cfg.update(overrides)
    return cfg


# TODO(nikhilap): Remove Blackwell restriction once cutlass_api includes H100 kernels
@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
@instantiate_parametrized_tests
class TestNVUniversalGemm(TestCase):
    # [CLASS DOC] Main NVGEMM test case, skipped unless cutlass_api is available and the GPU is
    # a datacenter Blackwell. Parametrized tests are instantiated by the class decorator. Each
    # test compiles a small function under an NVGEMM-only autotune config after resetting dynamo
    # and compares to eager output (or inspects code / kernel cache behavior).
    """Test cases for NVIDIA Universal GEMM functionality."""

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize(
        "layout_a,layout_b",
        (
            ("contiguous", "contiguous"),
            ("aligned_offset", "contiguous"),
            ("contiguous", "aligned_offset"),
            ("contiguous", "view"),
            ("aligned_offset", "view"),
            ("padded", "contiguous"),
            ("contiguous", "padded"),
        ),
    )
    def test_matmul(self, dtype, layout_a, layout_b):
        # [FUNCTION DOC] Parametrized over half precision dtype and a pair of layouts for the
        # two operands. Compiles a matmul with M of 513 (not a multiple of 16; only N and K need
        # to be) and compares to eager with default tolerances.
        """Test matmul with various dtypes and tensor layouts.

        M=513 tests that non-divisible M dimension works
        (only N and K must be divisible by 16).
        """
        m, n, k = 513, 512, 512

        def matmul(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product of two tensors.
            return a @ b

        a = _create_tensor_with_layout(layout_a, m, k, dtype)
        b = _create_tensor_with_layout(layout_b, k, n, dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    def test_arch_filter_rejects_min_cc_only_kernels(self):
        # [FUNCTION DOC] Regression test that the arch filter in the kernel cache is exact. Gets
        # the device compute capability and its target, clears the kernel cache, takes the
        # manifest of kernels, and selects those designed for a minimum capability not above the
        # device one, then those among them whose supported targets include the device target.
        # Asserts the first set is non-empty and the second is strictly smaller, i.e. an arch
        # specific kernel is rejected where a minimum-only check would accept it.
        """designed_for_min_cc <= device cc is insufficient: an arch-conditional
        sm90 kernel reports min_cc=90 but only lists a cc=90 target and won't
        compile on sm100. The exact-arch filter (via _device_target) must reject
        kernels a min_cc-only check would wrongly accept.
        """
        from torch._inductor.codegen.nv_universal_gemm import kernel_cache

        major, minor = torch.cuda.get_device_capability()
        cc = major * 10 + minor
        device_target = kernel_cache._device_target(cc)

        kernel_cache.clear_cache()
        manifest = kernel_cache._get_kernel_cache()
        min_cc_ok = [k for k in manifest.values() if k.designed_for_min_cc <= cc]
        arch_ok = [
            k
            for k in min_cc_ok
            if device_target.supports_operators_from(k.metadata.supported_targets)
        ]

        self.assertTrue(min_cc_ok, "no kernels pass the min_cc check")
        self.assertLess(
            len(arch_ok),
            len(min_cc_ok),
            "exact-arch filter should reject kernels that min_cc alone accepts",
        )

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_matmul_swap_ab(self, dtype):
        # [FUNCTION DOC] Parametrized over half precision dtype. With the swap-operands option
        # enabled, compiles a small-M matmul and checks the result equals eager, so swapped
        # operands and transposed output round-trip correctly.
        """swap_ab computes (B^T @ A^T)^T so the large N lands on the M-axis,
        improving tile utilization for small-M shapes. Verify a small-M matmul
        stays numerically correct with swap_ab enabled (the swapped operands and
        transposed output view must round-trip to the original result).
        """
        m, n, k = 16, 512, 512

        def matmul(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product of two tensors.
            return a @ b

        a = _create_tensor_with_layout("contiguous", m, k, dtype)
        b = _create_tensor_with_layout("contiguous", k, n, dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_swap_ab=True)):
            compiled_fn = torch.compile(matmul)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    def test_cudagraphs_intermediate_addmm(self):
        # [FUNCTION DOC] Regression test for CUDA graphs with an NVGEMM addmm whose output is an
        # intermediate consumed by a second addmm. With cudagraphs enabled it runs the compiled
        # chain three times and compares the last result with a float32 eager reference cast to
        # bfloat16 using loose tolerances. Checks that capture does not fail because the cached
        # kernel retains the intermediate.
        """An NVGEMM addmm whose bias-epilogue output is an intermediate consumed
        downstream must not leave that output retained by the cached EFC kernel,
        or CUDA graph capture fails ("tensor(s) in the cudagraph pool not tracked
        as outputs") and the tensor leaks. The cached kernel is built from
        meta-device tensors, so runtime outputs flow only through the per-call
        arguments.
        """
        dtype = torch.bfloat16
        m, k = 512, 512
        # Scaled down so the chained bf16 matmul stays well-conditioned.
        bias = torch.randn(k, dtype=dtype, device="cuda") * 0.1
        x = torch.randn(m, k, dtype=dtype, device="cuda") * 0.1
        w = torch.randn(k, k, dtype=dtype, device="cuda") * 0.1

        def chain(bias, x, w):
            # [FUNCTION DOC] HELPER: Two chained addmm operations that share the bias and weight
            # transpose; the first result is an intermediate feeding the second.
            # h is an NVGEMM addmm output consumed downstream (not the graph's
            # final output), which is what triggers the retention bug.
            h = torch.addmm(bias, x, w.t())
            return torch.addmm(bias, h, w.t())

        expected = chain(bias.float(), x.float(), w.float()).to(dtype)

        torch._dynamo.reset()

        cfg = _nvgemm_config()
        cfg["triton.cudagraphs"] = True
        with config.patch(cfg):
            compiled_fn = torch.compile(chain)
            for _ in range(3):
                result = compiled_fn(bias, x, w)

        torch.testing.assert_close(result, expected, rtol=1.6e-2, atol=1e-1)

    def test_efc_epilogue_lookup_no_deadlock(self):
        # [FUNCTION DOC] Regression test that looking up an epilogue EFC kernel with an unknown
        # name, no pre-resolved base kernel and an empty cache does not deadlock, since the
        # cache lock is reentrant. Runs the lookup in a worker thread, waits up to 120 seconds,
        # asserts the thread finished and that the lookup returned None.
        """get_efc_kernel_with_epilogue holds _cache_lock and, when the base EFC
        kernel is not pre-resolved and misses the args cache, calls
        get_kernel_by_name -> _ensure_caches, which re-acquires _cache_lock on the
        same thread. With a plain (non-reentrant) lock this self-deadlocks; the
        lock is an RLock. Run the reentrant path in a worker thread and assert it
        completes (a plain Lock would hang here).
        """
        from torch._inductor.codegen.nv_universal_gemm import kernel_cache

        # Force the manifest-fallback re-entry: no cached manifest, empty args
        # cache, unknown name, no pre-resolved base_kernel.
        kernel_cache.clear_cache()
        result = []

        def worker():
        # [FUNCTION DOC] HELPER: Thread body that performs the kernel cache lookup of an
        # epilogue kernel for a nonexistent name and appends the returned value to the enclosing
        # result list.
            result.append(
                kernel_cache.get_efc_kernel_with_epilogue(
                    "__nonexistent_kernel__", None, base_kernel=None
                )
            )

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=120)
        self.assertFalse(
            t.is_alive(),
            "get_efc_kernel_with_epilogue deadlocked (reentrant _cache_lock)",
        )
        self.assertIsNone(result[0])

    def test_unaligned_base_pointer_rejected(self):
        # [FUNCTION DOC] Checks that a matmul whose first operand has a base pointer not 16 byte
        # aligned (a view starting 117 bfloat16 elements into a storage) is rejected: running
        # the compiled function raises an error whose message matches no valid choice
        # (NoValidChoicesError), because the NVGEMM backend is the only choice and fallback is
        # off.
        """Test that matmul with unaligned base pointer is rejected.

        cutlass_api requires 16-byte aligned base pointers. Since alignment
        can't be checked at compile time (FakeTensors don't have real pointers),
        Inductor must guard against unaligned buffers.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product of two tensors.
            return a @ b

        # offset=117 elements * 2 bytes = 234 bytes (NOT 16-byte aligned)
        storage = torch.randn(m * k + 512, device=device, dtype=dtype)
        a = torch.as_strided(storage[117:], (m, k), (k, 1))
        b = torch.randn(k, n, device=device, dtype=dtype)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(matmul)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(a, b)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_reinterpret_view_from_slice(self, dtype):
        # [FUNCTION DOC] Parametrized over half precision dtype. Projects an input, splits the
        # result into two column slices (reinterpret views with non-contiguous strides),
        # multiplies one by the transpose of the other, and checks the compiled NVGEMM result
        # equals eager.
        """Test that sliced tensors (creating ReinterpretViews) work correctly.

        When tensors are slices of a shared buffer (e.g., from a fused projection),
        they become ReinterpretViews with non-contiguous strides.
        """
        m, n, k = 512, 512, 512
        device = "cuda"

        def fn(x, weight):
            # [FUNCTION DOC] HELPER: Projects x with the weight, splits columns into two halves
            # and returns the first half times the transpose of the second.
            projected = x @ weight  # (m, 2*n)
            a, b = projected.split(n, dim=1)  # Each is (m, n)
            return a @ b.t()  # (m, m)

        x = torch.randn(m, k, device=device, dtype=dtype)
        weight = torch.randn(k, 2 * n, device=device, dtype=dtype)
        expected = fn(x, weight)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(fn)
            result = compiled_fn(x, weight)

        torch.testing.assert_close(result, expected)

    def test_workspace_allocation(self):
        # [FUNCTION DOC] Mocks the kernel workspace size method of the cutlass_api kernel class
        # to report 1024 bytes (no real kernel needs a workspace), compiles a matmul and checks
        # the generated code passes the workspace argument and the result equals eager.
        """Test that workspace allocation works correctly.

        Since no current CUTLASS kernels require a workspace, we mock the
        kernel.get_workspace_size method to return a non-zero value.
        """
        m, n, k = 512, 512, 512
        dtype = torch.bfloat16
        device = "cuda"

        def matmul(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product of two tensors.
            return a @ b

        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
        expected = matmul(a, b)

        torch._dynamo.reset()

        import cutlass_api

        def patched_get_workspace_size(self, args):
        # [FUNCTION DOC] HELPER: Replacement for the kernel workspace size query that ignores
        # its arguments and always returns 1024.
            return 1024

        with patch.object(
            cutlass_api.Kernel,
            "get_workspace_size",
            patched_get_workspace_size,
        ):
            with config.patch(_nvgemm_config()):
                result, (code,) = run_and_get_code(
                    torch.compile(matmul),
                    a,
                    b,
                )

        self.assertIn("workspace=workspace", code)
        torch.testing.assert_close(result, expected)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_bmm_non_standard_batch_stride(self, dtype):
        # [FUNCTION DOC] Parametrized over half precision dtype. Builds batched operands by
        # transposing tensors so the batch stride is not the largest one, raises AssertionError
        # if the setup somehow produced standard batch strides, then compiles a batched matmul
        # under NVGEMM and compares to eager.
        """Test BMM path with non-standard batch strides."""
        batch, m, n, k = 8, 64, 256, 128
        device = "cuda"

        def bmm(a, b):
            # [FUNCTION DOC] HELPER: Batched matrix product of two tensors.
            return torch.bmm(a, b)

        # Transpose creates non-largest batch stride
        a_base = torch.randn(m, batch, k, device=device, dtype=dtype)
        a = a_base.transpose(0, 1)  # (batch, m, k) with stride (k, batch*k, 1)

        b_base = torch.randn(k, batch, n, device=device, dtype=dtype)
        b = b_base.transpose(0, 1)  # (batch, k, n) with stride (n, batch*n, 1)

        if a.stride()[0] == a.shape[1] * a.shape[2]:
            raise AssertionError(
                "Test setup error: a should have non-standard batch stride"
            )
        if b.stride()[0] == b.shape[1] * b.shape[2]:
            raise AssertionError(
                "Test setup error: b should have non-standard batch stride"
            )

        expected = bmm(a, b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(bmm)
            result = compiled_fn(a, b)

        torch.testing.assert_close(result, expected)

    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view"),
    )
    @parametrize(
        "m,n,k",
        (
            (256, 512, 1024),
            (256, 1024, 512),
            (128, 256, 512),
            (512, 256, 1024),
        ),
    )
    def test_scaled_gemm_mxfp8(self, layout_a, m, n, k):
        # [FUNCTION DOC] Parametrized over the layout of A and shape (m, n, k). Builds float8
        # e4m3 operands (B is a transposed random matrix), block-32 power-of-two scale tensors
        # sized with the padded K block count, and compares the compiled scaled matrix multiply
        # with float32 output against eager.
        """Test MXFP8 scaled GEMM with NVGEMM backend."""
        block_size = 32

        def scaled_mm(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matrix multiply with the given scales and float32
            # output dtype.
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32
            )

        a_fp8 = _create_tensor_with_layout(layout_a, m, k, torch.float8_e4m3fn)
        b_fp8 = torch.randint(-1, 2, (n, k), device="cuda").to(torch.float8_e4m3fn).T

        scale_a = torch.rand(m, _prep_k(k, block_size), device="cuda").to(
            torch.float8_e8m0fnu
        )
        scale_b = torch.rand(_prep_k(k, block_size), n, device="cuda").to(
            torch.float8_e8m0fnu
        )

        expected = scaled_mm(a_fp8, b_fp8, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp8, b_fp8, scale_a, scale_b)

        torch.testing.assert_close(result, expected)

    @parametrize(
        "m,n,k",
        (
            (16, 512, 1024),
            (32, 256, 512),
            (64, 512, 1024),
            (64, 64, 512),
        ),
    )
    def test_scaled_gemm_nvf4_padded_scales(self, m, n, k):
        # [FUNCTION DOC] Parametrized over shape. NVFP4 scaled GEMM with scale tensors padded to
        # 128 rows of the M or N dimension (more elements than the kernel strictly needs); the
        # kernel must accept both padded and unpadded scale sizes. Compiles with the autotune
        # choice restricted by a regex to the vendored choice, and compares to eager allowing
        # NaNs (random fp4 data can decode to NaN).
        """Test NVF4 with padded scales (M or N < 128).

        torch._scaled_mm creates scales with block_size_mn=128 padding,
        producing more elements than the kernel expects. The kernel's
        _supports() must accept both padded and unpadded scale numels.
        """
        packed_k = k // 2

        def scaled_mm(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matrix multiply with the given scales and bfloat16
            # output dtype.
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
            )

        a_fp4 = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b_fp4 = torch.randint(
            0, 256, (n, packed_k), device="cuda", dtype=torch.uint8
        ).view(torch.float4_e2m1fn_x2)
        b_fp4_t = b_fp4.T

        num_k_blocks = ceildiv(k, 16)
        padded_k_blocks = _round_up(num_k_blocks, 4)
        scale_a_numel = _round_up(m, 128) * padded_k_blocks
        scale_b_numel = _round_up(n, 128) * padded_k_blocks

        scale_a = torch.rand(scale_a_numel, device="cuda").to(torch.float8_e4m3fn)
        scale_b = torch.rand(scale_b_numel, device="cuda").to(torch.float8_e4m3fn)

        expected = scaled_mm(a_fp4, b_fp4_t, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(
            _nvgemm_config(
                **{"test_configs.autotune_choice_desc_regex": "inductor_vendored"}
            )
        ):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp4, b_fp4_t, scale_a, scale_b)

        torch.testing.assert_close(result, expected, equal_nan=True)

    def test_scaled_gemm_grouped_n_reduce_provider(self):
        # [FUNCTION DOC] Directly exercises the cutlass operators API for a blockwise scaled
        # NVFP4 GEMM with a local reduce output. Builds fp4 operands, swizzled fp8 scale
        # tensors, a bfloat16 output and a float32 reduce output that gets a sum per group of 32
        # columns along the second axis. Creates the gemm arguments, picks the first candidate
        # EFC-only scaled kernel whose tile width equals N, compiles and runs it on the current
        # stream, and asserts the reduce output equals the group sums of the main output.
        from cutlass import Float32
        from cutlass.operators import ScaleMode, ScaleSwizzleMode

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _scaled_candidates,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_arguments,
        )

        m, n, k = 128, 128, 512
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        group = 32
        reduce_out = torch.empty((m, n // group), device="cuda", dtype=torch.float32)
        mode = ScaleMode.Blockwise1x16
        swizzle = ScaleSwizzleMode.Swizzle32x4x4
        args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
            local_reduce_out=reduce_out,
            local_reduce_group=group,
            local_reduce_axis=1,
        )
        kernel = next(
            candidate
            for candidate in _scaled_candidates(args, 100, efc_only=True)
            if candidate.metadata.design.tile_shape[1] == n
        )
        artifact = kernel.compile(args)
        kernel.run(
            args,
            artifact,
            stream=torch.cuda.current_stream(),
            assume_supported_args=True,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            reduce_out,
            out.float().view(m, -1, group).sum(-1),
        )

    def test_scaled_gemm_unit_dim_epilogue_non_current_stream(self):
        # [FUNCTION DOC] Directly exercises a scaled NVFP4 GEMM with M equal to one and a bias
        # add epilogue function, run on a stream other than the current one. Computes the eager
        # scaled matmul plus bias as reference, builds epilogue arguments, picks a candidate
        # kernel (selection arguments exclude the epilogue) with tile width equal to N, compiles
        # it, synchronizes and sleeps the GPU to make a race visible, runs the kernel on a fresh
        # stream, synchronizes it and asserts the output equals the reference.
        from cutlass import Float32
        from cutlass.operators import ScaleMode, ScaleSwizzleMode
        from cutlass.operators.arguments import EpilogueArguments

        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _scaled_candidates,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _create_gemm_arguments,
        )

        m, n, k = 1, 128, 512
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        bias = torch.randn((1, 1), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        expected = (
            torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            + bias
        )
        epilogue = EpilogueArguments(
            epilogue_fn="def epilogue(accum, bias):\n    D = accum + bias\n    return D\n",
            bias=bias,
            D=out,
        )
        mode = ScaleMode.Blockwise1x16
        swizzle = ScaleSwizzleMode.Swizzle32x4x4
        args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
            epilogue=epilogue,
        )
        selection_args = _create_gemm_arguments(
            "SCALED_GEMM",
            (a, b, scale_a, scale_b),
            out,
            Float32,
            scale_mode_a=mode,
            swizzle_mode_a=swizzle,
            scale_mode_b=mode,
            swizzle_mode_b=swizzle,
        )
        kernel = next(
            candidate
            for candidate in _scaled_candidates(selection_args, 100, efc_only=True)
            if candidate.metadata.design.tile_shape[1] == n
        )
        artifact = kernel.compile(args)
        torch.cuda.synchronize()
        torch.cuda._sleep(10_000_000)
        stream = torch.cuda.Stream()
        kernel.run(args, artifact, stream=stream, assume_supported_args=True)
        stream.synchronize()
        self.assertEqual(out, expected)

    @parametrize("out_dtype", (torch.float32, torch.bfloat16))
    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view"),
    )
    @parametrize(
        "m,n,k",
        (
            (256, 512, 1024),
            (256, 1024, 512),
            (128, 256, 512),
            (512, 256, 1024),
        ),
    )
    def test_scaled_gemm_nvf4(self, out_dtype, layout_a, m, n, k):
        # [FUNCTION DOC] Parametrized over output dtype (float32, bfloat16), A layout and shape.
        # NVF4 scaled GEMM (float4 data with float8 scales, block size 16) with scales padded to
        # 128 rows along M and N and four along K blocks. Compiles under NVGEMM restricted by
        # regex to the vendored choice and compares to eager with NaN equality, since random fp4
        # can produce NaN.
        """Test NVF4 (Float4 + Float8E4M3FN scales, block_size=16) with NVGEMM backend."""
        packed_k = k // 2
        block_size = 16

        def scaled_mm(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matrix multiply with the given scales and the
            # parametrized output dtype.
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype
            )

        a_fp4 = _create_tensor_with_layout(
            layout_a, m, packed_k, torch.float4_e2m1fn_x2
        )
        b_fp4 = torch.randint(
            0, 256, (n, packed_k), device="cuda", dtype=torch.uint8
        ).view(torch.float4_e2m1fn_x2)
        b_fp4_t = b_fp4.T

        num_k_blocks = ceildiv(k, block_size)
        padded_k_blocks = _round_up(num_k_blocks, 4)
        block_size_mn = 128
        scale_a_numel = block_size_mn * ceildiv(m, block_size_mn) * padded_k_blocks
        scale_b_numel = block_size_mn * ceildiv(n, block_size_mn) * padded_k_blocks

        scale_a = torch.rand(scale_a_numel, device="cuda").to(torch.float8_e4m3fn)
        scale_b = torch.rand(scale_b_numel, device="cuda").to(torch.float8_e4m3fn)

        expected = scaled_mm(a_fp4, b_fp4_t, scale_a, scale_b)

        torch._dynamo.reset()

        with config.patch(
            _nvgemm_config(
                **{"test_configs.autotune_choice_desc_regex": "inductor_vendored"}
            )
        ):
            compiled_fn = torch.compile(scaled_mm)
            result = compiled_fn(a_fp4, b_fp4_t, scale_a, scale_b)

        # a_fp4 and b_fp4_t could come with NaNs.
        torch.testing.assert_close(result, expected, equal_nan=True)

    @parametrize(
        "layout_a",
        ("contiguous", "aligned_offset", "view", "padded"),
    )
    def test_grouped_gemm(self, layout_a):
        # [FUNCTION DOC] Parametrized over the layout of A. Grouped GEMM test with four groups
        # of 256 by 256 bfloat16 operands through the NVGEMM backend. Only the TN layout
        # (column-major B) is supported.
        """Test grouped GEMM with NVGEMM backend and various A layouts.

        GroupedGemm currently only supports TN layout (column-major B).
        """
        g, k, n = 4, 256, 256
        dtype = torch.bfloat16
        device = "cuda"

        def grouped_mm(a, b, offsets):
            # [FUNCTION DOC] HELPER: Grouped matrix multiply of A and B using the group offsets.
            return torch._grouped_mm(a, b, offs=offsets)

        b = torch.randn(g, n, k, device=device, dtype=dtype).permute(0, 2, 1)

        m_per_group = [64, 64, 64, 64]
        total_m = sum(m_per_group)
        offsets = torch.tensor(
            [sum(m_per_group[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a = _create_tensor_with_layout(layout_a, total_m, k, dtype, device)

        expected = grouped_mm(a, b, offsets)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(grouped_mm)
            result = compiled_fn(a, b, offsets)

        torch.testing.assert_close(result, expected)

    def test_grouped_gemm_varying_offsets(self):
        # [FUNCTION DOC] Compiles one grouped matmul (four groups, bfloat16, column-major B) and
        # runs it with two offset tensors that have the same total rows but different group
        # sizes (equal groups versus 32, 96, 48, 80). Raises AssertionError if the sizes do not
        # sum equally. Checks each result against eager and that the two results differ, showing
        # offsets are handled at runtime and not baked in.
        """Test that different offset distributions produce correct results.

        Runs the same compiled function with two different offset distributions
        (same total_m) to verify offsets are handled dynamically at runtime.
        """
        g, k, n = 4, 256, 256
        dtype = torch.bfloat16
        device = "cuda"

        def grouped_mm(a, b, offsets):
            # [FUNCTION DOC] HELPER: Grouped matrix multiply of A and B using the group offsets.
            return torch._grouped_mm(a, b, offs=offsets)

        b = torch.randn(g, n, k, device=device, dtype=dtype).permute(0, 2, 1)

        m_per_group_1 = [64, 64, 64, 64]
        total_m = sum(m_per_group_1)
        offsets_1 = torch.tensor(
            [sum(m_per_group_1[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_1 = torch.randn(total_m, k, device=device, dtype=dtype)

        m_per_group_2 = [32, 96, 48, 80]
        if sum(m_per_group_2) != total_m:
            raise AssertionError("Total M must match for cache key test")
        offsets_2 = torch.tensor(
            [sum(m_per_group_2[: i + 1]) for i in range(g)],
            device=device,
            dtype=torch.int32,
        )
        a_2 = torch.randn(total_m, k, device=device, dtype=dtype)

        expected_1 = grouped_mm(a_1, b, offsets_1)
        expected_2 = grouped_mm(a_2, b, offsets_2)

        torch._dynamo.reset()

        with config.patch(_nvgemm_config()):
            compiled_fn = torch.compile(grouped_mm)

            result_1 = compiled_fn(a_1, b, offsets_1)
            torch.testing.assert_close(result_1, expected_1)

            result_2 = compiled_fn(a_2, b, offsets_2)
            torch.testing.assert_close(result_2, expected_2)

            self.assertFalse(torch.allclose(result_1, result_2))


class TestNVUniversalGemmHeuristics(TestCase):
    # [CLASS DOC] Unit tests of the heuristics filter and of the GEMM epilogue IR reduction
    # analysis that run without cutlass_api or a GPU, using mock kernel and input objects. Has
    # two mock factories.
    """Unit tests for NVUniversalGemmHeuristics without requiring actual libraries."""

    def test_grouped_reduction_conversion_contract(self):
        # [FUNCTION DOC] Checks the grouped reduction recognition on loop IR expressions: a sum
        # of squares of the loaded accumulator converted to float32 is classified as sum with
        # square; the same value passing through a float16 conversion first, or through a
        # bitcast conversion, is not recognized and gives None. Uses a store of the reduction
        # and a group size of four with a bfloat16 output.
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_ir,
        )

        load = Expr("load", ("gemm", 0, None))

        def classify(value):
            # [FUNCTION DOC] HELPER: Wraps the given value in a squared sum reduction expression
            # stored at output index zero and returns what the grouped reduction analysis
            # reports for the gemm buffer with group size four and bfloat16 output (a pair of
            # reduction and pre-op kinds, or None).
            square = Expr("mul", (value, value))
            reduction = Expr("reduction", (torch.float32, torch.float32, "sum", square))
            return grouped_reduction_ir(
                GemmEpilogueIRStore(0, reduction), "gemm", 4, torch.bfloat16
            )

        fp32 = Expr("to_dtype", (load, torch.float32))
        self.assertEqual(classify(fp32), ("sum", "square"))
        fp16_then_fp32 = Expr(
            "to_dtype", (Expr("to_dtype", (load, torch.float16)), torch.float32)
        )
        self.assertIsNone(classify(fp16_then_fp32))
        bitcast = Expr("to_dtype_bitcast", (load, torch.bfloat16, torch.bfloat16))
        self.assertIsNone(classify(Expr("to_dtype", (bitcast, torch.float32))))

    def test_grouped_reduction_ir_normalizes_loop_representation(self):
        # [FUNCTION DOC] Checks that the analysis object built from a stored sum reduction of
        # the loaded gemm accumulator reports, for the output, group size four, axis one and
        # float32 dtype, a reduction config equal to one with the sum reduction and the identity
        # pre-op.
        from torch._inductor.kernel.gemm_epilogue import GemmReductionConfig
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRAnalysis,
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
        )

        load = Expr("load", ("gemm", 0, None))
        reduction = Expr("reduction", (torch.float32, torch.float32, "sum", load))
        analysis = GemmEpilogueIRAnalysis({"out": GemmEpilogueIRStore(0, reduction)})

        self.assertEqual(
            analysis.grouped_reduction("out", "gemm", 4, 1, torch.float32),
            GemmReductionConfig("out", 4, 1, "sum", "identity"),
        )

    def test_grouped_reduction_rejects_ambiguous_composite(self):
        # [FUNCTION DOC] A stored value that adds a sum reduction and a max reduction of the
        # same load is an ambiguous composite; the grouped reduction analysis must return None
        # for it.
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression as Expr,
            GemmEpilogueIRStore,
            grouped_reduction_ir,
        )

        summed = Expr(
            "reduction",
            (torch.float32, torch.float32, "sum", Expr("load", ("gemm", 0, None))),
        )
        maximum = Expr(
            "reduction",
            (torch.float32, torch.float32, "max", Expr("load", ("gemm", 0, None))),
        )
        store = GemmEpilogueIRStore(0, Expr("add", (summed, maximum)))
        self.assertIsNone(grouped_reduction_ir(store, "gemm", 4, torch.float32))

    def test_epilogue_ir_preserves_empty_tuple_argument(self):
        # [FUNCTION DOC] Checks that building an epilogue IR expression with an argument tuple
        # containing an empty tuple keeps that argument unchanged and that the keyword arguments
        # are empty.
        from torch._inductor.kernel.loop_ir_epilogue_lowering import (
            GemmEpilogueIRExpression,
        )

        expression = GemmEpilogueIRExpression("reshape", ("value", ()))
        self.assertEqual(expression.args, ("value", ()))
        self.assertEqual(expression.kwargs, ())

    def _create_mock_kernel(self, tile_m, tile_n, tile_k, cluster_m, cluster_n):
        # [FUNCTION DOC] HELPER: Returns a mock kernel whose metadata design has the given tile
        # shape (three values) and cluster shape (two values).
        """Create a mock kernel with the given tile/cluster configuration."""
        kernel = MagicMock()
        kernel.metadata.design.tile_shape = (tile_m, tile_n, tile_k)
        kernel.metadata.design.cluster_shape = (cluster_m, cluster_n)
        return kernel

    def _create_mock_inputs(self, m=512, n=512, k=512, dtype=torch.float16):
        # [FUNCTION DOC] HELPER: Returns a mock GEMM inputs object: hinted M, N, K as given, the
        # given dtype, operand indices zero and one, and hinted strides for row-major operands
        # of those sizes.
        """Create a mock MMKernelInputs."""
        inputs = MagicMock()
        inputs.mnk_hinted.return_value = (m, n, k)
        inputs.dtype.return_value = dtype
        inputs._mat1_idx = 0
        inputs._mat2_idx = 1
        inputs.strides_hinted.return_value = ((k, 1), (n, 1))
        return inputs

    def test_fallback_when_heuristics_unavailable(self):
        # [FUNCTION DOC] When the heuristics report they should not run, filtering ten mock
        # kernels with a count of three must return exactly the first three kernels.
        """Test that filter_kernels returns first N kernels when heuristics unavailable."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = [self._create_mock_kernel(128, 128, 64, 1, 1) for _ in range(10)]
        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=False):
            result = heuristics.filter_kernels(kernels, inputs, count=3)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, kernels[:3])

    def test_fallback_when_no_configs_extracted(self):
        # [FUNCTION DOC] When heuristics should run but the kernels' design objects lack a tile
        # shape (so no configs can be extracted), filtering five kernels with a count of two
        # returns the first two.
        """Test fallback when kernel configs cannot be extracted."""
        heuristics = NVUniversalGemmHeuristics()

        kernels = []
        for _ in range(5):
            kernel = MagicMock()
            kernel.metadata.design = MagicMock(spec=[])  # No tile_shape attr
            kernels.append(kernel)

        inputs = self._create_mock_inputs()

        with patch.object(heuristics, "should_run", return_value=True):
            result = heuristics.filter_kernels(kernels, inputs, count=2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result, kernels[:2])

    def test_filter_kernels_sorts_by_runtime(self):
        # [FUNCTION DOC] With mocked heuristic configs giving estimated runtimes for three
        # kernels, checks the filter returns kernels ordered fastest first (the second, the
        # third, the first), truncates to the requested count of two by dropping the slowest,
        # and returns all three for a count of ten.
        """Test that filter_kernels returns kernels sorted by estimated runtime and respects count."""
        heuristics = NVUniversalGemmHeuristics()

        kernel_a = self._create_mock_kernel(128, 128, 64, 1, 1)
        kernel_b = self._create_mock_kernel(256, 128, 64, 2, 1)
        kernel_c = self._create_mock_kernel(128, 256, 32, 1, 2)
        kernels = [kernel_a, kernel_b, kernel_c]

        inputs = self._create_mock_inputs()

        heuristic_configs = [
            HeuristicConfig(128, 128, 64, 1, 1, 4, 1, 64, 64, 32, 0.003),
            HeuristicConfig(256, 128, 64, 2, 1, 4, 1, 64, 64, 32, 0.001),
            HeuristicConfig(128, 256, 32, 1, 2, 4, 1, 64, 64, 32, 0.002),
        ]

        with patch.object(heuristics, "should_run", return_value=True):
            with patch.object(
                heuristics, "_get_heuristic_configs", return_value=heuristic_configs
            ):
                # Test sorting with count=3 (all kernels)
                result = heuristics.filter_kernels(kernels, inputs, count=3)
                self.assertEqual(len(result), 3)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)
                self.assertIs(result[2], kernel_a)

                # Test count limit with count=2 (should drop slowest)
                result = heuristics.filter_kernels(kernels, inputs, count=2)
                self.assertEqual(len(result), 2)
                self.assertIs(result[0], kernel_b)
                self.assertIs(result[1], kernel_c)

                # Test count > available kernels (should return all 3)
                result = heuristics.filter_kernels(kernels, inputs, count=10)
                self.assertEqual(len(result), 3)


@unittest.skipIf(
    not (
        ensure_nv_universal_gemm_available()
        and is_datacenter_blackwell_arch()
        and ensure_nvmatmul_heuristics_available()
    ),
    "Requires cutlass_api, nvMatmulHeuristics, and Blackwell GPU",
)
class TestNVUniversalGemmHeuristicsIntegration(TestCase):
    # [CLASS DOC] Integration tests that call the real nvMatmulHeuristics library; skipped
    # unless cutlass_api, the heuristics library and a datacenter Blackwell GPU are all
    # available.
    """Integration tests for nvMatmulHeuristics with real library calls."""

    def test_fp4_heuristic_configs(self):
        # [FUNCTION DOC] Asks the heuristics for up to five configs for a 256 by 512 by 1024 FP4
        # block scaled GEMM (row-major A, column-major B, float32 accumulator and output) and
        # asserts at least one config is returned with positive tile M, tile N and estimated
        # runtime.
        """Test that nvMatmulHeuristics returns configs for FP4 blockscaled GEMM."""
        heuristics = NVUniversalGemmHeuristics()

        m, n, k = 256, 512, 1024
        configs = heuristics._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a=torch.float4_e2m1fn_x2,
            layout_a="row",
            layout_b="col",
            count=5,
            valid_configs=OrderedSet(),
            accumulator_type=torch.float32,
            dtype_b=torch.float4_e2m1fn_x2,
            out_dtype=torch.float32,
        )

        self.assertGreater(
            len(configs), 0, "nvMatmulHeuristics returned no FP4 configs"
        )
        for cfg in configs:
            self.assertGreater(cfg.tile_m, 0)
            self.assertGreater(cfg.tile_n, 0)
            self.assertGreater(cfg.estimated_runtime, 0)

    def test_fp8_heuristic_configs(self):
        # [FUNCTION DOC] Same as the FP4 heuristics test but for float8 e4m3 operands: expects
        # at least one config with positive tile sizes and estimated runtime.
        """Test that nvMatmulHeuristics returns configs for FP8 GEMM."""
        heuristics = NVUniversalGemmHeuristics()

        m, n, k = 256, 512, 1024
        configs = heuristics._get_heuristic_configs(
            m,
            n,
            k,
            dtype_a=torch.float8_e4m3fn,
            layout_a="row",
            layout_b="col",
            count=5,
            valid_configs=OrderedSet(),
            accumulator_type=torch.float32,
            dtype_b=torch.float8_e4m3fn,
            out_dtype=torch.float32,
        )

        self.assertGreater(
            len(configs), 0, "nvMatmulHeuristics returned no FP8 configs"
        )
        for cfg in configs:
            self.assertGreater(cfg.tile_m, 0)
            self.assertGreater(cfg.tile_n, 0)
            self.assertGreater(cfg.estimated_runtime, 0)


@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
class TestNVUniversalGemmDynamicShapes(TestCase):
    # [CLASS DOC] NVGEMM tests with dynamic shapes; skipped unless cutlass_api is available and
    # the GPU is datacenter Blackwell.
    """Test cases for NVIDIA Universal GEMM with dynamic shapes."""

    @torch._dynamo.config.patch({"capture_dynamic_output_shape_ops": True})
    def test_unbacked_symint_rejected(self):
        # [FUNCTION DOC] With capture of data dependent output shape ops enabled, compiles
        # dynamically a function whose matrix row count comes from the nonzero count of a tensor
        # (an unbacked symbolic size). Asserts that running it raises an error matching no valid
        # choice, i.e. NVGEMM rejects unbacked symbolic sizes.
        """Test that NVGEMM rejects unbacked symbolic integers."""

        def fn(x, w):
            # [FUNCTION DOC] HELPER: Builds a ones matrix whose row count is the number of
            # nonzero entries of x and returns its product with w.
            nz = torch.nonzero(x)  # Creates unbacked symint for nz.size(0)
            a = torch.ones(nz.size(0), w.size(0), dtype=w.dtype, device=w.device)
            return a @ w

        x = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], device="cuda")
        w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_max_profiling_configs=2)):
            compiled_fn = torch.compile(fn, dynamic=True)
            with self.assertRaisesRegex(
                Exception, "NoValidChoicesError|no valid choice"
            ):
                compiled_fn(x, w)

    def test_dynamic_shapes(self):
        # [FUNCTION DOC] Stress test: one dynamic-shape compiled matmul is called with a
        # sequence of bfloat16 shapes, some tiny shapes flagged as unsupported (must raise the
        # no valid choice error) and the others supported (result must match eager). Unsupported
        # shapes recur after supported ones.
        """Stress test dynamic shapes with extreme variations."""

        def matmul(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product of two tensors.
            return a @ b

        torch._dynamo.reset()

        with config.patch(_nvgemm_config(nvgemm_max_profiling_configs=2)):
            compiled_fn = torch.compile(matmul, dynamic=True)

            shapes = [
                (4, 4, 4, False),
                (16, 16, 16, True),
                (2048, 64, 128, True),
                (4, 4, 4, False),  # Unsupported again
                (64, 2048, 128, True),
                (128, 128, 2048, True),
                (2048, 2048, 512, True),
                (16, 16, 16, True),
            ]

            for m, n, k, supported in shapes:
                a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
                if not supported:
                    with self.assertRaisesRegex(
                        Exception, "NoValidChoicesError|no valid choice"
                    ):
                        compiled_fn(a, b)
                else:
                    result = compiled_fn(a, b)
                    torch.testing.assert_close(result, a @ b)


@unittest.skipIf(
    not (ensure_nv_universal_gemm_available() and is_datacenter_blackwell_arch()),
    "NVIDIA Universal GEMM (cutlass_api) library not available or not on Blackwell",
)
@instantiate_parametrized_tests
class TestNVUniversalGemmEpilogueFusion(TestCase):
    # [CLASS DOC] NVGEMM epilogue fusion tests: each compiles a function with benchmarking
    # mocked out for deterministic fusion decisions and checks both results and generated code
    # markers. Class constants M, N, K are all 512. Parametrized tests are instantiated by the
    # class decorator.
    """Test cases for NVIDIA Universal GEMM epilogue fusion.

    Tests verify both correctness and that fusion actually occurs by examining
    generated code for epilogue markers. Benchmarks are mocked to ensure
    deterministic fusion decisions independent of GPU noise.
    """

    M, N, K = 512, 512, 512

    def _compile_and_check(self, fn, *args):
        # [FUNCTION DOC] HELPER: Resets dynamo, compiles the function with NVGEMM autotuning
        # (two profiled configs, caches force disabled) while the scheduler fused-node and
        # codegen module benchmarks are mocked to fixed times, and runs it with the given args
        # capturing generated code. Returns the result, the joined generated code and a flag
        # that is true when the code contains the epilogue function name and the epilogue
        # arguments class name.
        torch._dynamo.reset()
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_fused_nodes",
                return_value=(1.0, ""),
            ),
            mock.patch.object(
                Scheduler,
                "benchmark_codegened_module",
                return_value=(0.5, ""),
            ),
        ):
            result, code_list = run_and_get_code(torch.compile(fn), *args)
        code = "\n".join(code_list)
        epilogue_fused = EPILOGUE_FN_NAME in code and "EpilogueArguments" in code
        return result, code, epilogue_fused

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_matmul_pointwise_epilogue_fusion(self, dtype):
        # [FUNCTION DOC] Parametrized over half precision dtype. A relu after a matmul must be
        # fused into the GEMM epilogue; checks results close to eager and the fusion flag.
        """Pointwise op (relu) is fused into the GEMM epilogue."""
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Relu of a matrix product.
            return torch.relu(a @ b)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "pointwise op was NOT fused into epilogue")

    def test_matmul_multi_store_epilogue_fusion(self):
        # [FUNCTION DOC] A matmul whose float32 result feeds both a relu output and an add-one
        # output must fuse into the epilogue with two stores: results match eager, the fusion
        # flag is true and the code contains the second output pointer name.
        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a pair: relu of the float32 product, and the same
            # product plus one.
            result = (a @ b).float()
            return torch.relu(result), result + 1.0

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused)
        self.assertIn("out_ptr1", code)

    def test_flex_gemm_pointwise_epilogue_fusion(self):
        # [FUNCTION DOC] flex_gemm with a matrix multiply and a relu epilogue body using the
        # NVGEMM backend option: the result must equal relu of the product within tolerance and
        # the epilogue must be fused.
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls flex_gemm with a matrix multiply, the two inputs and
            # a relu body.
            return flex_gemm(
                torch.mm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, torch.relu(a @ b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "FlexGEMM body was NOT fused into epilogue")

    def test_flex_gemm_preserves_output_for_unfused_reduction(self):
        # [FUNCTION DOC] flex_gemm whose epilogue returns the relu of the accumulator together
        # with a grouped sum (groups of 32 along the last axis) of its float32 view; the
        # reduction output is not fused as a reduction but both outputs must be preserved.
        # Compares both outputs to eager with tolerances, asserts the epilogue is fused and the
        # second output pointer exists in the code.
        dtype = torch.bfloat16
        a = torch.randn(128, 64, device="cuda", dtype=dtype)
        b = torch.randn(64, 128, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Runs flex_gemm with a matrix multiply and the epilogue
            # defined below.
            def epilogue(acc):
                # [FUNCTION DOC] HELPER: Epilogue body returning the relu of the accumulator and
                # the sum over groups of 32 of its float32 reshaped view.
                grouped = acc.float().view(128, -1, 32)
                return acc.relu(), grouped.sum(-1)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "NVGEMM"},
            )

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        torch.testing.assert_close(result[0], expected[0], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(result[1], expected[1], atol=1e-1, rtol=1e-2)
        self.assertTrue(epilogue_fused, "pointwise consumer was NOT fused")
        self.assertIn("out_ptr1", code)

    @parametrize("operation", ("mul", "sigmoid", "gelu"))
    def test_scaled_mm_pointwise_epilogue_fusion(self, operation):
        # [FUNCTION DOC] Parametrized over operation (scale by one half, sigmoid, gelu). NVFP4
        # scaled GEMM with bfloat16 output and padded swizzled fp8 scales, followed by the
        # chosen unary pointwise op; the op must be fused into the scaled epilogue and the
        # result must match eager (NaN equality allowed).
        """Unary pointwise op is fused into an NVFP4 scaled GEMM epilogue."""
        m, n, k = self.M, self.N, self.K
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output followed by the unary op
            # chosen by the enclosing parameter (sigmoid, gelu, or multiplication by one half).
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            if operation == "sigmoid":
                return torch.sigmoid(result)
            if operation == "gelu":
                return torch.nn.functional.gelu(result)
            return result * 0.5

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b
        )
        torch.testing.assert_close(result, fn(a, b, scale_a, scale_b), equal_nan=True)
        self.assertTrue(
            epilogue_fused, f"{operation} was NOT fused into scaled epilogue"
        )

    def test_matmul_single_store_epilogue_chain(self):
        # [FUNCTION DOC] A float32 matmul result followed by relu and an add-one gives a single
        # stored output: results match eager, the epilogue is fused, and no second output
        # pointer is present.
        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Relu of the float32 product plus one.
            return torch.relu((a @ b).float()) + 1.0

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        self.assertEqual(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused)
        self.assertNotIn("out_ptr1", code)

    def test_scaled_mm_multi_store_epilogue_fusion(self):
        # [FUNCTION DOC] NVFP4 scaled GEMM with bfloat16 output whose result is returned along
        # with a float32 relu and a float32 half-scaled copy: three stores. Compares all three
        # outputs to eager, requires the epilogue to be fused and the second and third output
        # pointers to appear in the code.
        m, n, k = self.M, self.N, self.K
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul returning the bfloat16 result, the relu of
            # its float32 view and the float32 view scaled by one half.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            result_fp32 = result.float()
            return result, torch.relu(result_fp32), result_fp32 * 0.5

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b
        )
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result, expected)
        self.assertTrue(epilogue_fused)
        self.assertIn("out_ptr1", code)
        self.assertIn("out_ptr2", code)

    @parametrize(
        "case",
        (
            ((((M, N), torch.bfloat16),), True),
            ((((N,), torch.bfloat16),), True),
            ((((1, N), torch.bfloat16),), True),
            ((((M, 1), torch.bfloat16),), True),
            ((((N,), torch.bfloat16), ((M, 1), torch.bfloat16)), True),
            ((((N,), torch.float32),), True),
        ),
        name_fn=lambda case: "_and_".join(
            f"{'x'.join(map(str, sh))}_{dtype_name(dt)}" for sh, dt in case[0]
        ),
    )
    def test_scaled_mm_broadcast_epilogue_fusion(self, case):
        # [FUNCTION DOC] Parametrized over a case: a tuple of bias specs (shape and dtype) plus
        # whether fusion is expected (all listed cases expect fusion): full, row vector, one-
        # row, one-column, row plus column together, and float32 bias. Builds an NVFP4 scaled
        # GEMM with padded scales, adds each random bias and applies relu. Asserts the fusion
        # flag equals the expectation and the result matches eager with NaN equality (random fp4
        # can give NaNs).
        bias_specs, expected_fused = case
        m, n, k = self.M, self.N, self.K
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        biases = tuple(
            torch.randn(shape, device="cuda", dtype=dtype)
            for shape, dtype in bias_specs
        )

        def fn(a, b, scale_a, scale_b, *biases):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output, then adds every
            # supplied bias in turn and returns the relu.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            for bias in biases:
                result = result + bias
            return torch.relu(result)

        result, code, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b, *biases
        )
        self.assertEqual(epilogue_fused, expected_fused)
        # Random packed FP4 operands can produce NaNs in the reference GEMM.
        torch.testing.assert_close(
            result, fn(a, b, scale_a, scale_b, *biases), equal_nan=True
        )

    @parametrize("bias_shape", ((1,), (1, 1)))
    def test_scaled_mm_unit_dim_broadcast_epilogue_fusion(self, bias_shape):
        # [FUNCTION DOC] Parametrized over a unit bias shape (one element or one by one). NVFP4
        # scaled GEMM with M equal to one, N 128, then a bias add and relu. Asserts the epilogue
        # is fused and the result equals eager.
        m, n, k = 1, 128, self.K
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        bias = torch.randn(bias_shape, device="cuda", dtype=torch.bfloat16)

        def fn(a, b, scale_a, scale_b, bias):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output, plus bias, then relu.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            return torch.relu(result + bias)

        result, _, epilogue_fused = self._compile_and_check(
            fn, a, b, scale_a, scale_b, bias
        )
        self.assertTrue(epilogue_fused)
        self.assertEqual(result, fn(a, b, scale_a, scale_b, bias))

    @parametrize(
        "case",
        (
            (0, "sum", 4),
            (0, "mean", 4),
            (0, "prod", 4),
            (0, "amax", 4),
            (0, "amin", 4),
            (0, "sum", 8),
            (0, "mean", 8),
            (0, "prod", 8),
            (0, "amax", 8),
            (0, "amin", 8),
            (0, "sum", 16),
            (0, "mean", 16),
            (0, "prod", 16),
            (0, "amax", 16),
            (0, "amin", 16),
            (1, "sum", 32),
            (1, "mean", 32),
            (1, "prod", 32),
            (1, "amax", 32),
            (1, "amin", 32),
            (1, "sum", 64),
            (1, "mean", 64),
            (1, "prod", 64),
            (1, "amax", 64),
            (1, "amin", 64),
        ),
        name_fn=lambda case: f"axis_{case[0]}_{case[1]}_group_{case[2]}",
    )
    def test_scaled_mm_grouped_reduce_fusion(self, case):
        # [FUNCTION DOC] Parametrized over (axis, reduction kind, group size) with reductions
        # sum, mean, prod, amax and amin, groups of 4, 8, 16 along axis zero or 32, 64 along
        # axis one. Builds an NVFP4 scaled GEMM (N doubles to 256 for groups above 32) that
        # returns the bfloat16 result and a grouped reduction of its float32 view. Asserts both
        # outputs equal eager and the code uses the local reduce output. For the axis one, sum,
        # group 32 case it also checks, when Triton supports reduction ordering (otherwise
        # skips), that strict numerics disables the local reduce output and the code mentions
        # the inner tree reduction ordering.
        axis, reduction, group = case
        m, n, k = 128, 256 if group > 32 else 128, 512
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output plus a grouped reduction
            # (kind, axis and group from the enclosing parameters) over the float32 view, split
            # in groups along the chosen axis. Returns the result and the reduced tensor.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = (
                result.float().view(-1, group, n)
                if axis == 0
                else result.float().view(m, -1, group)
            )
            dim = 1 if axis == 0 else -1
            if reduction == "sum":
                reduced = grouped.sum(dim)
            elif reduction == "mean":
                reduced = grouped.mean(dim)
            elif reduction == "prod":
                reduced = grouped.prod(dim)
            elif reduction == "amax":
                reduced = grouped.amax(dim)
            else:
                reduced = grouped.amin(dim)
            return result, reduced

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertIn("'local_reduce_out'", code)

        if case == (1, "sum", 32):
            if not has_triton_reduction_ordering():
                self.skipTest("requires INNER_TREE Triton")
            with config.patch({"numerics": "strict"}):
                _, strict_code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
            self.assertNotIn("'local_reduce_out'", strict_code)
            self.assertIn("ReductionOrdering.INNER_TREE", strict_code)

    def test_scaled_mm_grouped_reduce_source_fusion(self):
        # [FUNCTION DOC] NVFP4 scaled GEMM returning relu of the result and the mean of the
        # squared float32 values over groups of 32 on the last axis. Asserts both outputs equal
        # eager and the code uses a local reduce output with the square pre-op source.
        m, n, k, group = 128, 128, 512, 32
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output; returns its relu and
            # the grouped mean of squares of the float32 view.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(m, -1, group)
            return torch.relu(result), grouped.square().mean(-1)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        expected = fn(a, b, scale_a, scale_b)
        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1], expected[1])
        self.assertIn("'local_reduce_out'", code)
        self.assertIn("'local_reduce_source': 'square'", code)

    @config.patch(emulate_precision_casts=True)
    def test_scaled_mm_grouped_reduce_rejects_intermediate_fp16(self):
        # [FUNCTION DOC] With precision-cast emulation enabled, a grouped reduction whose input
        # passes through a float16 intermediate cannot be fused as a local reduce. Asserts the
        # compiled results equal eager and the code does not contain the local reduce output.
        m, n, k, group = 128, 128, 512, 32
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output; returns its relu and
            # the grouped mean of squares of a float32 view obtained after a round trip through
            # float16.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.half().float().view(m, -1, group)
            return torch.relu(result), grouped.square().mean(-1)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertNotIn("'local_reduce_out'", code)

    def test_scaled_mm_grouped_reduce_feeds_main(self):
        # [FUNCTION DOC] Group size four along axis zero: the grouped mean is expanded back and
        # subtracted from the main float32 result, so the reduction feeds the main output.
        # Asserts equality with eager and that the code flags that the local reduce feeds the
        # main output.
        m, n, k, group = 128, 128, 512, 4
        packed_k = k // 2
        a = _create_tensor_with_layout(
            "contiguous", m, packed_k, torch.float4_e2m1fn_x2
        )
        b = torch.randint(0, 256, (n, packed_k), device="cuda", dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        b = b.T
        padded_k_blocks = _round_up(ceildiv(k, 16), 4)
        scale_a = torch.rand(_round_up(m, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )
        scale_b = torch.rand(_round_up(n, 128) * padded_k_blocks, device="cuda").to(
            torch.float8_e4m3fn
        )

        def fn(a, b, scale_a, scale_b):
            # [FUNCTION DOC] HELPER: Scaled matmul with bfloat16 output; returns the float32
            # result minus the group mean broadcast back to full shape.
            result = torch._scaled_mm(
                a,
                b,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            grouped = result.float().view(-1, group, n)
            mean = grouped.mean(1, keepdim=True).expand(-1, group, -1)
            return result.float() - mean.reshape(m, n)

        result, code, _ = self._compile_and_check(fn, a, b, scale_a, scale_b)
        self.assertEqual(result, fn(a, b, scale_a, scale_b))
        self.assertIn("'local_reduce_feeds_main': True", code)

    def test_matmul_add_relu_chained(self):
        # [FUNCTION DOC] A matmul plus bias followed by relu forms a single pointwise chain that
        # is fused as one epilogue; results match eager and the fusion flag is true.
        """Multi-op pointwise chain (a@b + bias → relu) collapses to one
        ComputedBuffer and is fused as a single epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(self.M, self.N, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            # [FUNCTION DOC] HELPER: Relu of matrix product plus bias.
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "bias+relu chain was NOT fused into epilogue")

    def test_matmul_cast_dtype(self):
        # [FUNCTION DOC] A cast of the bfloat16 product to float32 must be fused into the
        # epilogue; results match eager.
        """Output-dtype cast in the epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Product converted to float32.
            return (a @ b).to(torch.float32)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "dtype cast was NOT fused into epilogue")

    def test_plain_matmul_no_epilogue(self):
        # [FUNCTION DOC] A plain matrix product must not have epilogue fusion markers in the
        # code, and the result matches eager.
        """Test that plain matmul does NOT produce epilogue fusion markers."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Plain matrix product.
            return a @ b

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertFalse(
            epilogue_fused, "plain matmul should NOT have epilogue fusion markers"
        )

    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190235"
    )
    def test_reduction_not_fused(self):
        # [FUNCTION DOC] Checks that a sum reduction after a GEMM is not fused into the epilogue
        # (flag false) and the result is close to eager. Currently skipped unconditionally
        # because of CI failures.
        """Test that reductions after GEMM are NOT fused into the epilogue."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Sum over the last dimension of a matrix product.
            return (a @ b).sum(dim=-1)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        expected = fn(a, b)
        torch.testing.assert_close(
            result.float(), expected.float(), atol=5e-2, rtol=5e-2
        )
        self.assertFalse(
            epilogue_fused,
            "reduction after GEMM should NOT be fused into NVGEMM epilogue",
        )

    def test_epilogue_fusion_eliminates_intermediate(self):
        # [FUNCTION DOC] When a relu is fused, no separate intermediate GEMM buffer is
        # allocated. Checks fusion and result, then scans every generated line that defines the
        # main function of an NVGEMM kernel and requires its signature to have exactly the two
        # input pointers, one output pointer and the stream keyword argument.
        """Verify that epilogue fusion eliminates the intermediate GEMM buffer.

        When relu is fused, the GEMM output should not be allocated separately.
        The fused kernel writes directly to the final output. We check that
        only one output buffer is allocated for the fused kernel, not two
        (one for GEMM + one for relu)."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Relu of a matrix product.
            return torch.relu(a @ b)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "relu was NOT fused into epilogue")
        # The fused kernel's main function should take exactly 3 positional
        # params (in_ptr0, in_ptr1, out_ptr0) plus stream. No extra intermediate.
        for line in code.split("\n"):
            if "_main(" in line and "def " in line and "nv_universal_gemm" in line:
                self.assertIn(
                    "in_ptr0, in_ptr1, out_ptr0, stream=None",
                    line,
                    lambda msg: f"{msg}\nUnexpected kernel signature: {line.strip()}",
                )

    def test_epilogue_with_aux_input(self):
        # [FUNCTION DOC] An epilogue of bias add followed by relu, where the bias is a full-size
        # tensor, is fused with the bias passed in as an extra kernel argument; results match
        # eager.
        """Epilogue that reads an auxiliary tensor (bias) gets it as a kernel arg."""
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(self.M, self.N, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            # [FUNCTION DOC] HELPER: Relu of matrix product plus bias.
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "bias+relu was NOT fused into epilogue")

    @parametrize(
        "bias_shape",
        ((N,), (1, N), (M, 1)),
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_epilogue_with_broadcast_aux_input(self, bias_shape):
        # [FUNCTION DOC] Parametrized over broadcastable bias shapes (row vector, one-row, one-
        # column). Same fusion and numerical check as the full bias test.
        dtype = torch.bfloat16
        a = torch.randn(self.M, self.K, device="cuda", dtype=dtype)
        b = torch.randn(self.K, self.N, device="cuda", dtype=dtype)
        bias = torch.randn(bias_shape, device="cuda", dtype=dtype)

        def fn(a, b, bias):
            # [FUNCTION DOC] HELPER: Relu of matrix product plus bias.
            return torch.relu((a @ b) + bias)

        result, code, epilogue_fused = self._compile_and_check(fn, a, b, bias)
        torch.testing.assert_close(result, fn(a, b, bias), atol=1e-2, rtol=1e-2)
        self.assertTrue(epilogue_fused, "broadcast bias+relu was not fused")

    def test_efc_disk_cache_round_trip(self):
        # [FUNCTION DOC] Skips if no cached EFC kernel with bfloat16 inputs and outputs and a
        # ttt layout in its name is found. Otherwise compiles that kernel for a bfloat16 GEMM,
        # unwraps the compiled object to the inner JIT function (checking it is not a plain
        # closure), stores it to the disk cache under a temporary path in the system temp
        # directory and reads it back (must not be None), rewraps it and runs the kernel on the
        # current stream with a fresh output. Asserts the output is close to a float32
        # reference. Side effect: writes a cache file under the temp directory.
        """Verify that EFC kernel compiled artifacts can be serialized to disk
        and reloaded correctly.

        EFC kernels produce a closure-wrapped compiled_obj. The disk cache
        unwraps the inner JIT function for serialization and rewraps it on
        reload. This test compiles an EFC kernel, round-trips through disk
        cache, and verifies the reloaded artifact produces correct results."""
        from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
            _get_kernel_cache,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            _rewrap_efc_compiled_obj,
            _unwrap_efc_compiled_obj,
        )
        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

        cache = _get_kernel_cache()
        efc_kernel = None
        for name, k in cache.items():
            if (
                "EFC" in name
                and "ABFloat16" in name
                and "outBFloat16" in name
                and "ttt" in name
            ):
                efc_kernel = k
                break
        if efc_kernel is None:
            self.skipTest("No matching EFC kernel found in cache")

        import cutlass_api
        from cutlass_api.artifact import CompiledArtifact

        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16)

        args = cutlass_api.arguments.GemmArguments(
            a, b, out, accumulator_type=torch.float32
        )
        artifact = efc_kernel.compile(args)

        # Unwrap, serialize, reload, rewrap
        inner = _unwrap_efc_compiled_obj(artifact.compiled_obj)
        self.assertNotEqual(
            type(inner).__name__,
            "function",
            "unwrap should extract the JIT function, not the closure",
        )

        test_cache: dict = {}
        disk_cache_set(test_cache, "/tmp/test_efc_rt.py", ("efc",), ("key",), inner, 0)
        loaded = disk_cache_get({}, "/tmp/test_efc_rt.py", ("efc",), ("key",), 0)
        self.assertIsNotNone(loaded, "disk_cache_get returned None")

        rewrapped = _rewrap_efc_compiled_obj(loaded, efc_kernel)
        reloaded_artifact = CompiledArtifact(rewrapped, efc_kernel)

        # Run with reloaded artifact and verify correctness
        out2 = torch.empty(self.M, self.N, device="cuda", dtype=torch.bfloat16)
        args2 = cutlass_api.arguments.GemmArguments(
            a, b, out2, accumulator_type=torch.float32
        )
        efc_kernel.run(
            args2,
            reloaded_artifact,
            stream=torch.cuda.current_stream(),
            workspace=None,
            assume_supported_args=True,
        )
        torch.cuda.synchronize()
        expected = a.float() @ b.float()
        torch.testing.assert_close(out2.float(), expected, atol=1e-2, rtol=1e-2)

    @unittest.skip(
        "Disabled due to CI failures; see "
        "https://github.com/pytorch/pytorch/issues/190234"
    )
    def test_workspace_runtime_integration(self):
        # [FUNCTION DOC] End-to-end check that workspace handling works at runtime when kernel
        # workspace size is mocked to 4096. Wraps the NVGEMM module benchmark method to record
        # every returned pair of time and path while compiling a relu matmul with NVGEMM
        # autotuning, and asserts the result matches eager, the benchmark was called, and at
        # least one benchmark time is finite. Currently skipped unconditionally because of CI
        # failures.
        """End-to-end: mock the chosen kernel's workspace_size to non-zero and
        actually let benchmark_codegened_module run, exercising the runtime
        path that consumes the generated get_args()/call() helpers.

        Without the workspace fix, the rendered call would TypeError on the
        missing positional arg, _benchmark_nvgemm_module's broad except would
        return inf, autotune would silently fall back to a non-EFC choice,
        and a naive `assert_close` against eager would still pass. To actually
        catch the regression we intercept _benchmark_nvgemm_module to record
        every (ms, path) it returns and assert at least one finite-ms result —
        i.e., at least one EFC choice with workspace did get benchmarked."""
        import cutlass_api

        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )

        a = torch.randn(self.M, self.K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(self.K, self.N, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Relu of matrix product.
            return torch.relu(a @ b)

        bench_results: list[tuple[float, str]] = []
        orig_bench = CUDACombinedScheduling._benchmark_nvgemm_module

        def capturing_bench(self, module):
            # [FUNCTION DOC] HELPER: Calls the original benchmark method, appends its returned
            # time and path to the enclosing list of benchmark results, and returns them
            # unchanged.
            ms, path = orig_bench(self, module)
            bench_results.append((ms, path))
            return ms, path

        torch._dynamo.reset()
        with (
            patch.object(
                cutlass_api.Kernel, "get_workspace_size", lambda self, args: 4096
            ),
            mock.patch.object(
                CUDACombinedScheduling, "_benchmark_nvgemm_module", capturing_bench
            ),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "NVGEMM",
                    "nvgemm_max_profiling_configs": 2,
                    "force_disable_caches": True,
                }
            ),
        ):
            result = torch.compile(fn)(a, b)
        torch.testing.assert_close(result, fn(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(bench_results, "_benchmark_nvgemm_module never invoked")
        finite = [(ms, p) for ms, p in bench_results if ms != float("inf")]
        self.assertTrue(
            finite,
            lambda msg: f"{msg}\nAll NVGEMM benchmarks returned inf — workspace handling likely "
            f"broken. Results: {bench_results}",
        )


if __name__ == "__main__":
    run_tests()
