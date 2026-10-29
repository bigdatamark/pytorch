# Owner(s): ["module: inductor"]
# [MODULE DOC] Slow end-to-end minifier tests that must run the compiled program in an isolated
# subprocess because the injected bug crashes the process. Defines a decorator-style skip for
# ROCm with Triton older than 3.7 (no debug asserts) and one test class with a CPU and a GPU
# runtime-error case. The main block only runs tests off macOS, off ASAN and below Python 3.11.
import unittest

import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_MACOS,
    skipIfWindows,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu
from torch.utils._triton import get_triton_version


skipIfRocmWithoutDebugAsserts = unittest.skipIf(
    TEST_WITH_ROCM and get_triton_version() < (3, 7),
    "requires Triton 3.7+ on ROCm for debug asserts",
)


# These minifier tests are slow, because they must be run in separate
# subprocesses
class MinifierIsolateTests(MinifierTestBase):
    # [CLASS DOC] Minifier tests (MinifierTestBase subclass) for runtime errors injected after
    # AOT autograd, each run with subprocess isolation. Holds one shared helper plus CPU and GPU
    # test cases.
    def _test_after_aot_runtime_error(self, device, expected_error):
        # [FUNCTION DOC] HELPER: Builds a small compiled program (relu then cos on a 2x2 random
        # tensor moved to the given device) as source text and runs the full minifier flow with
        # the after-AOT repro stage and isolate enabled, expecting the given error text in the
        # subprocess stderr. Isolation is required since the injected bug crashes the process.
        run_code = f"""\
@torch.compile()
def inner(x):
    x = torch.relu(x)
    x = torch.cos(x)
    return x

inner(torch.randn(2, 2).to("{device}"))
"""
        # These must isolate because they crash the process
        self._run_full_test(run_code, "aot", expected_error, isolate=True)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "runtime_error")
    @skipIfWindows(
        msg="Build Failed: fatal error C1083: Cannot open include file: 'Python.h': No such file or directory"
    )
    def test_after_aot_cpu_runtime_error(self):
        # [FUNCTION DOC] With the CPU relu bug injection set to a runtime error, runs the
        # isolated after-AOT minifier flow on cpu with an empty expected error string (any
        # stderr matches). Skipped on Jetson and Windows.
        self._test_after_aot_runtime_error("cpu", "")

    @skipIfRocmWithoutDebugAsserts
    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "runtime_error")
    def test_after_aot_gpu_runtime_error(self):
        # [FUNCTION DOC] With the Triton relu bug injection set to a runtime error, runs the
        # isolated after-AOT minifier flow on the GPU device. The expected text is the injected
        # assert message on XPU or ROCm, otherwise the CUDA device-side assert message. Needs a
        # GPU and is skipped on old-Triton ROCm.
        # CUDA's __assertfail surfaces through PyTorch as "device-side assert";
        # ROCm's Triton AMD lowering prints the injected assertion text before trapping.
        expected_error = (
            "injected assert fail"
            if GPU_TYPE == "xpu" or TEST_WITH_ROCM
            else "device-side assert"
        )
        self._test_after_aot_runtime_error(GPU_TYPE, expected_error)


if __name__ == "__main__":
    import sys

    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    # also skip on Py 3.11+ since unhandled exceptions can cause segfaults
    if not IS_MACOS and not TEST_WITH_ASAN and sys.version_info < (3, 11):
        run_tests()
