# Owner(s): ["module: inductor"]
# [MODULE DOC] Runs the shared inductor test suite against the Triton CPU backend.
# Only active when both HAS_CPU and TRITON_HAS_CPU are true. Defines the tuple
# TRITON_CPU_SLOW_TESTS naming test cases to skip, then (when enabled) defines two test
# classes patched to use the triton cpu_backend with dtype/shape runtime asserts disabled,
# copies all CommonTemplate tests into CpuTritonTests, and wraps the slow ones with a skip.
# The main block runs the tests (needing filelock) only if the CPU backend is available.
import unittest

from torch._inductor import config
from torch._inductor.test_case import run_tests
from torch.testing._internal.inductor_utils import HAS_CPU, TRITON_HAS_CPU


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor

TRITON_CPU_SLOW_TESTS = (
    # ~1000s
    "test_sort_stable_cpu",
    # ~300-400s
    "test_sort_bool_cpu",
    "test_sort_transpose_cpu",
    # ~100-300s
    "test_avg_pool3d_backward2_cpu",
    "test_pattern_matcher_multi_user_cpu",
    "test_split_cumsum_cpu",
    # ~50-150s
    "test_large_strided_reduction_cpu",
    "test_large_block_sizes_cpu",
)

if HAS_CPU and TRITON_HAS_CPU:

    @config.patch(
        {
            "cpu_backend": "triton",
            "test_configs.runtime_triton_dtype_assert": False,
            "test_configs.runtime_triton_shape_assert": False,
        }
    )
    class SweepInputsCpuTritonTest(test_torchinductor.SweepInputsCpuTest):
        # [CLASS DOC] Sweep-inputs CPU test case re-run under the triton CPU backend config
        # (class-level config patch). Adds no tests of its own; inherits SweepInputsCpuTest.
        pass

    @config.patch(
        {
            "cpu_backend": "triton",
            "test_configs.runtime_triton_dtype_assert": False,
            "test_configs.runtime_triton_shape_assert": False,
        }
    )
    class CpuTritonTests(test_torchinductor.TestCase):
        # [CLASS DOC] Test case for CPU tests compiled with the triton CPU backend (config
        # patched at class level). Uses check_model as the common checker and device cpu; the
        # CommonTemplate tests are copied in below, with expected-failure marks read from the
        # _expected_failure_triton_cpu property.
        common = test_torchinductor.check_model
        device = "cpu"

    test_torchinductor.copy_tests(
        test_torchinductor.CommonTemplate,
        CpuTritonTests,
        "cpu",
        xfail_prop="_expected_failure_triton_cpu",
    )

    for name in TRITON_CPU_SLOW_TESTS:
        setattr(
            CpuTritonTests,
            name,
            unittest.skip("Triton CPU: slow test")(getattr(CpuTritonTests, name)),
        )


if __name__ == "__main__":
    if HAS_CPU and TRITON_HAS_CPU:
        run_tests(needs="filelock")
