# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that compiling with inductor and running the attention fusion pass do not
# emit Python user warnings for the float32 matmul precision advisory; the advisory must go to the
# perf_hints log instead. Tests need a CUDA (non-ROCm) device of compute capability 8.0+.

import unittest
import warnings

import torch
import torch._inductor.compile_fx as inductor_compile_fx
import torch._inductor.fx_passes.fuse_attention as fuse_attention
from torch.testing._internal.common_utils import (
    recover_orig_fp32_precision,
    run_tests,
    TestCase,
)
from torch.testing._internal.logging_utils import logs_to_string


TF32_ADVISORY = "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled."


def _has_cuda_sm80() -> bool:
    # [FUNCTION DOC] HELPER: Returns true when CUDA is available, the build is not ROCm, and the
    # current device capability is at least 8.0.
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (8, 0)
    )


class InductorWarningTests(TestCase):
    # [CLASS DOC] Test case for warning behavior of the float32 TF32 advisory in inductor.
    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    @recover_orig_fp32_precision
    def test_trivial_matmul_compile_no_user_warning(self):
        # [FUNCTION DOC] Sets highest float32 matmul precision, clears the compile_fx advisory
        # cache and resets dynamo, then compiles a square matmul on an identity matrix with the
        # inductor backend while capturing user warnings and the compile_fx perf_hints log.
        # Asserts the result equals the input, that no warnings were captured, and that the log
        # contains the TF32 advisory text. A finally block restores the previous matmul precision
        # and resets dynamo again. Skipped without CUDA SM80.
        # recover_orig_fp32_precision restores the per-backend flags; the
        # legacy enum still needs the set_float32_matmul_precision below.
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            inductor_compile_fx._warn_tf32_disabled.cache_clear()
            torch._dynamo.reset()

            x = torch.eye(2, device="cuda")
            log_stream, ctx = logs_to_string("torch._inductor.compile_fx", "perf_hints")
            with ctx(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                actual = torch.compile(
                    lambda y: y @ y, backend="inductor", fullgraph=True
                )(x)
                torch.cuda.synchronize()

            self.assertEqual(actual, x)
            self.assertEqual([str(w.message) for w in caught], [])
            self.assertIn(TF32_ADVISORY, log_stream.getvalue())
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)
            torch._dynamo.reset()

    @unittest.skipIf(not _has_cuda_sm80(), "requires CUDA SM80")
    @recover_orig_fp32_precision
    def test_fuse_attention_tf32_advisory_no_user_warning(self):
        # [FUNCTION DOC] Sets highest float32 matmul precision and clears the fuse_attention
        # advisory cache, then calls the advisory function directly while capturing user warnings
        # and the fuse_attention perf_hints log. Asserts no warnings were captured and the log
        # holds the TF32 advisory text. The previous matmul precision is restored in a finally
        # block. Skipped without CUDA SM80.
        orig_matmul_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            fuse_attention._warn_tf32_disabled.cache_clear()

            log_stream, ctx = logs_to_string(
                "torch._inductor.fx_passes.fuse_attention", "perf_hints"
            )
            with ctx(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore")
                warnings.simplefilter("always", UserWarning)
                fuse_attention._warn_tf32_disabled()

            self.assertEqual([str(w.message) for w in caught], [])
            self.assertIn(TF32_ADVISORY, log_stream.getvalue())
        finally:
            torch.set_float32_matmul_precision(orig_matmul_precision)


if __name__ == "__main__":
    run_tests()
