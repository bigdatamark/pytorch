# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor GPU source code caches (CUDA, ROCm and XPU/SYCL).
# Defines device-selection helpers, a SAXPY kernel source for CUDA/HIP and one for SYCL, and
# device-generic tests that compile, load, call and cache those kernels. Also a plain XPU
# test class for cache hits and cache_clear. The device-type test class is instantiated for
# cuda and xpu at import time.

import ctypes
import shutil
import unittest
from unittest import mock

import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import CUDACodeCache, ROCmCodeCache, XPUCodeCache
from torch._inductor.codegen.cuda.compile_utils import _cuda_compiler
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.codegen.rocm.compile_command import rocm_compiler
from torch._inductor.codegen.xpu.compile_utils import _sycl_compiler, icpx_exist
from torch._inductor.exc import CUDACompileError, XPUCompileError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.triton_utils import (
    requires_gpu_and_triton,
    requires_xpu_and_triton,
)


def _has_rocm_compiler():
    # [FUNCTION DOC] HELPER: Returns True when a ROCm compiler is configured and found on PATH.
    # Asks rocm_compiler for the compiler name and checks it is not None and resolvable.
    compiler = rocm_compiler()
    return compiler is not None and shutil.which(compiler) is not None


def _has_gpu_codecache_compiler(device_type):
    # [FUNCTION DOC] HELPER: Reports whether the compiler needed for the given device type exists.
    # For xpu checks that the SYCL compiler exists; under a ROCm test build checks the ROCm
    # compiler; otherwise checks that nvcc exists. Returns a bool.
    if device_type == "xpu":
        return icpx_exist(_sycl_compiler())
    if TEST_WITH_ROCM:
        return _has_rocm_compiler()
    return nvcc_exist(_cuda_compiler())


def _gpu_codecache(device_type):
    # [FUNCTION DOC] HELPER: Picks the code cache class for a device type: XPUCodeCache for xpu,
    # ROCmCodeCache when TEST_WITH_ROCM is set, else CUDACodeCache.
    if device_type == "xpu":
        return XPUCodeCache
    return ROCmCodeCache if TEST_WITH_ROCM else CUDACodeCache


def _compile_error(device_type):
    # [FUNCTION DOC] HELPER: Returns the compile error exception class expected for a device type:
    # XPUCompileError for xpu, CUDACompileError otherwise (also used for ROCm).
    return XPUCompileError if device_type == "xpu" else CUDACompileError


# CUDA source doubles as ROCm/HIP source via the __HIPCC__ guard.
_CUDA_SOURCE_CODE = r"""

#include <stdio.h>

#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

__global__
void saxpy_device(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y) {
  // Perform SAXPY
  saxpy_device<<<(n+255)/256, 256>>>(n, a, x, y);
  return 0;
}

}
"""

# The standalone SYCL .so is not linked against libtorch, so it has no ambient
# queue. torch passes the address of its current XPUStream's sycl::queue as the
# trailing argument; the kernel submits to it so the USM pointers (allocated in
# torch's context) are valid, and waits so the result is ready on return.
_XPU_SOURCE_CODE = r"""

#include <sycl/sycl.hpp>

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y, void *queue_ptr) {
  sycl::queue &q = *static_cast<sycl::queue *>(queue_ptr);
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
     y[i] = a * x[i] + y[i];
   }).wait();
  return 0;
}

}
"""


def _source_code(device_type):
    # [FUNCTION DOC] HELPER: Returns the SYCL kernel source for xpu, else the CUDA/HIP source.
    return _XPU_SOURCE_CODE if device_type == "xpu" else _CUDA_SOURCE_CODE


def _break_kernel(source_code, device_type):
    # [FUNCTION DOC] HELPER: Returns a copy of the source with its first occurrence of a marker
    # token renamed by appending a suffix, so compilation fails. The token is the parallel loop
    # call for xpu and the device kernel name otherwise.
    token = "parallel_for" if device_type == "xpu" else "saxpy_device"
    return source_code.replace(token, token + "_wrong", 1)


def _call_saxpy(dll, device_type, n, a, x, y):
    # [FUNCTION DOC] HELPER: Calls the compiled saxpy entry point on the loaded library with
    # element count, scale, and the data pointers of tensors x and y (y is updated in place).
    # For xpu also passes the raw current XPU stream of x's device as an extra trailing pointer.
    # Returns nothing.
    args = [
        ctypes.c_int(n),
        ctypes.c_float(a),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
    ]
    if device_type == "xpu":
        stream = torch._C._xpu_getCurrentRawStream(x.device.index)
        args.append(ctypes.c_void_p(stream))
    dll.saxpy(*args)


class TestGPUCodeCache(InductorTestCase):
    # [CLASS DOC] Device-generic tests for the GPU code caches (CUDA, ROCm, XPU). Each test needs
    # a GPU, Triton and a usable compiler, and runs inside a fresh cache directory.
    @requires_gpu_and_triton
    def test_gpu_load(self):
        # [FUNCTION DOC] Compiles the SAXPY source to an object file and loads it as a shared
        # library through the device code cache inside a fresh cache. Skips if no compiler
        # is available.
        # Checks file suffixes under ROCm, or equal source path and hash key otherwise. Then runs
        # the kernel on random tensors and compares the result with the expected values.
        device = self.device_type
        if not _has_gpu_codecache_compiler(device):
            raise unittest.SkipTest("requires nvcc, ROCm, or SYCL compiler")
        source_code = _source_code(device)
        with fresh_cache():
            codecache = _gpu_codecache(device)
            # Test both .o and .so compilation.
            (
                object_file_path,
                object_hash_key,
                source_code_path0,
            ) = codecache.compile(source_code, "o")
            dll_wrapper, so_hash_key, source_code_path1 = codecache.load(
                source_code, "so"
            )
            if TEST_WITH_ROCM:
                self.assertTrue(object_file_path.endswith(".o"))
                self.assertTrue(source_code_path0.endswith(".cpp"))
                self.assertTrue(source_code_path1.endswith(".cpp"))
            else:
                self.assertEqual(source_code_path0, source_code_path1)
                self.assertEqual(object_hash_key, so_hash_key)

            # Test load and call functions in .so.
            x = torch.rand(10).float().to(device)
            y = torch.rand(10).float().to(device)
            a = 5.0
            expected_y = a * x + y
            _call_saxpy(dll_wrapper, device, 10, a, x, y)
            torch.testing.assert_close(y, expected_y)

    @requires_gpu_and_triton
    def test_compilation_error(self):
        # [FUNCTION DOC] Verifies that compiling a deliberately broken kernel source raises the
        # device-appropriate compile error. Skips if no compiler is available.
        device = self.device_type
        if not _has_gpu_codecache_compiler(device):
            raise unittest.SkipTest("requires nvcc, ROCm, or SYCL compiler")
        with fresh_cache():
            codecache = _gpu_codecache(device)
            error_source_code = _break_kernel(_source_code(device), device)
            with self.assertRaises(_compile_error(device)):
                codecache.compile(error_source_code, "o")

    @requires_gpu_and_triton
    def test_async_compile(self):
        # [FUNCTION DOC] Compiles the SAXPY source to a shared library using the matching
        # AsyncCompile method for the device (xpu, rocm or cuda), waits for pending jobs, then
        # calls the resulting library on random tensors and checks the output matches the
        # expected values.
        device = self.device_type
        if not _has_gpu_codecache_compiler(device):
            raise unittest.SkipTest("requires nvcc, ROCm, or SYCL compiler")
        source_code = _source_code(device)
        with fresh_cache():
            async_compile = AsyncCompile()
            if device == "xpu":
                compiled_res = async_compile.xpu(source_code, "so")
            elif TEST_WITH_ROCM:
                compiled_res = async_compile.rocm(source_code, "so")
            else:
                compiled_res = async_compile.cuda(source_code, "so")
            async_compile.wait(globals())

            # Test load and call functions in .so.
            x = torch.rand(5).float().to(device)
            y = torch.rand(5).float().to(device)
            a = 2.0
            expected_y = a * x + y
            _call_saxpy(compiled_res.result(), device, 5, a, x, y)
            torch.testing.assert_close(y, expected_y)


instantiate_device_type_tests(
    TestGPUCodeCache, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


@requires_xpu_and_triton
class TestXPUCodeCacheClear(InductorTestCase):
    # [CLASS DOC] Tests that repeated XPU compiles and loads hit the cache, and that cache_clear
    # empties both the dll cache and the parent source cache.
    def test_cache_hit(self):
        # [FUNCTION DOC] Compiles and loads the SYCL source twice in a fresh cache. Expects
        # equal hash keys and source paths, and the same library wrapper object on the second load.
        # Skips if no SYCL compiler exists.
        if not icpx_exist(_sycl_compiler()):
            raise unittest.SkipTest("requires SYCL compiler")
        with fresh_cache():
            _, hash_key1, source_path1 = XPUCodeCache.compile(_XPU_SOURCE_CODE, "o")
            _, hash_key2, source_path2 = XPUCodeCache.compile(_XPU_SOURCE_CODE, "o")
            self.assertEqual(hash_key1, hash_key2)
            self.assertEqual(source_path1, source_path2)

            dll_wrapper1, so_hash1, _ = XPUCodeCache.load(_XPU_SOURCE_CODE, "so")
            dll_wrapper2, so_hash2, _ = XPUCodeCache.load(_XPU_SOURCE_CODE, "so")
            self.assertEqual(so_hash1, so_hash2)
            self.assertIs(dll_wrapper1, dll_wrapper2)

    def test_cache_clear_clears_dll_cache(self):
        # [FUNCTION DOC] Inserts a fake entry into the XPU dll cache, calls cache_clear, and
        # asserts the dll cache is empty. Registers cache_clear as cleanup.
        self.addCleanup(XPUCodeCache.cache_clear)
        XPUCodeCache.dll_cache["fake_path.so"] = mock.MagicMock()
        self.assertGreater(len(XPUCodeCache.dll_cache), 0)

        XPUCodeCache.cache_clear()

        self.assertEqual(
            len(XPUCodeCache.dll_cache),
            0,
            "dll_cache should be empty after cache_clear()",
        )

    def test_cache_clear_clears_parent_cache(self):
        # [FUNCTION DOC] Inserts a fake entry into the XPU parent cache mapping, calls cache_clear,
        # and asserts the cache is empty. Registers cache_clear as cleanup.
        self.addCleanup(XPUCodeCache.cache_clear)
        XPUCodeCache.cache["fake_key"] = mock.MagicMock()
        self.assertGreater(len(XPUCodeCache.cache), 0)

        XPUCodeCache.cache_clear()

        self.assertEqual(
            len(XPUCodeCache.cache),
            0,
            "cache should be empty after cache_clear()",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
