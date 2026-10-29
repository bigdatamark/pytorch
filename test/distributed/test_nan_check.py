# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for the c10d check_for_nan custom op on CPU and (when available)
# the current accelerator: no error for NaN-free float tensors, integer and empty
# tensors; a RuntimeError mentioning NaN when any element is NaN. Exits at import if
# distributed is unavailable.

import sys
import unittest

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import run_tests, TestCase


HAS_ACCELERATOR = torch.accelerator.is_available()
ACCELERATOR = (
    torch.accelerator.current_accelerator() if HAS_ACCELERATOR else torch.device("cpu")
)


class TestNanCheck(TestCase):
    # [CLASS DOC] Test cases calling torch.ops.c10d.check_for_nan on tensors of various
    # dtypes, sizes and devices. Accelerator variants are skipped when no accelerator exists
    # and synchronize afterwards so asynchronous device errors surface.
    def _check_for_nan(self, tensor):
        # [FUNCTION DOC] HELPER: Invokes the c10d check_for_nan op on the given tensor; the op
        # raises RuntimeError if the tensor holds NaN.
        torch.ops.c10d.check_for_nan(tensor)

    def test_no_nan_float32(self):
        # [FUNCTION DOC] A random float32 CPU tensor passes the check.
        self._check_for_nan(torch.randn(100))

    def test_no_nan_float64(self):
        # [FUNCTION DOC] A random float64 CPU tensor passes the check.
        self._check_for_nan(torch.randn(100, dtype=torch.float64))

    def test_no_nan_float16(self):
        # [FUNCTION DOC] A random float16 CPU tensor passes the check.
        self._check_for_nan(torch.randn(100, dtype=torch.float16))

    def test_no_nan_bfloat16(self):
        # [FUNCTION DOC] A random bfloat16 CPU tensor passes the check.
        self._check_for_nan(torch.randn(100, dtype=torch.bfloat16))

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float32_accelerator(self):
        # [FUNCTION DOC] A random float32 tensor on the accelerator passes the
        # check; then synchronizes.
        self._check_for_nan(torch.randn(100, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float64_accelerator(self):
        # [FUNCTION DOC] A random float64 tensor on the accelerator passes the
        # check; then synchronizes.
        self._check_for_nan(torch.randn(100, dtype=torch.float64, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_float16_accelerator(self):
        # [FUNCTION DOC] A random float16 tensor on the accelerator passes the
        # check; then synchronizes.
        self._check_for_nan(torch.randn(100, dtype=torch.float16, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_no_nan_bfloat16_accelerator(self):
        # [FUNCTION DOC] A random bfloat16 tensor on the accelerator passes the
        # check; then synchronizes.
        self._check_for_nan(torch.randn(100, dtype=torch.bfloat16, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_skips_integer_tensors_accelerator(self):
        # [FUNCTION DOC] An int32 tensor on the accelerator is accepted (no
        # NaN possible); synchronizes.
        self._check_for_nan(
            torch.tensor([1, 2, 3], dtype=torch.int32, device=ACCELERATOR)
        )
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_empty_tensor_accelerator(self):
        # [FUNCTION DOC] An empty float32 tensor on the accelerator is
        # accepted; then synchronizes.
        self._check_for_nan(torch.tensor([], dtype=torch.float32, device=ACCELERATOR))
        torch.accelerator.synchronize()

    @unittest.skipUnless(HAS_ACCELERATOR, "no accelerator available")
    def test_large_tensor_no_nan_accelerator(self):
        # [FUNCTION DOC] A 100000-element random float32 accelerator tensor
        # passes; then synchronizes.
        self._check_for_nan(torch.randn(100000, device=ACCELERATOR))
        torch.accelerator.synchronize()

    def test_nan_float32(self):
        # [FUNCTION DOC] A float32 CPU tensor containing a NaN makes the check raise
        # RuntimeError matching NaN.
        tensor = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_float64(self):
        # [FUNCTION DOC] A float64 CPU tensor containing a NaN raises
        # RuntimeError matching NaN.
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_float16(self):
        # [FUNCTION DOC] A float16 CPU tensor containing a NaN raises
        # RuntimeError matching NaN.
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_bfloat16(self):
        # [FUNCTION DOC] A bfloat16 CPU tensor containing a NaN raises
        # RuntimeError matching NaN.
        tensor = torch.tensor([1.0, float("nan")], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_nan_at_end(self):
        # [FUNCTION DOC] A NaN only in the last element of a 1000-element tensor
        # is still detected.
        tensor = torch.ones(1000)
        tensor[-1] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)

    def test_skips_integer_tensors(self):
        # [FUNCTION DOC] An int32 CPU tensor is accepted without error.
        self._check_for_nan(torch.tensor([1, 2, 3], dtype=torch.int32))

    def test_empty_tensor(self):
        # [FUNCTION DOC] An empty float32 CPU tensor is accepted without error.
        self._check_for_nan(torch.tensor([], dtype=torch.float32))

    def test_all_nan(self):
        # [FUNCTION DOC] A float tensor made entirely of NaN raises RuntimeError matching NaN.
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(torch.full((10,), float("nan")))

    def test_large_tensor_no_nan(self):
        # [FUNCTION DOC] A 100000-element random float32 CPU tensor passes the check.
        self._check_for_nan(torch.randn(100000))

    def test_large_tensor_with_nan(self):
        # [FUNCTION DOC] A 100000-element tensor with NaN set at the final index raises
        # RuntimeError matching NaN.
        tensor = torch.randn(100000)
        tensor[99999] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "NaN"):
            self._check_for_nan(tensor)


if __name__ == "__main__":
    run_tests()
