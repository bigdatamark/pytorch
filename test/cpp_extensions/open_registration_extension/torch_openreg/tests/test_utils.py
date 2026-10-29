# Owner(s): ["module: PrivateUse1"]

# [MODULE DOC] Tests for DLPack export/import of tensors on the out-of-tree "openreg" device.
# Requires the torch_openreg extension; no import-time side effects beyond imports.

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDLPack(TestCase):
    # [CLASS DOC] Test case for DLPack roundtrips of openreg tensors. Stateless.
    def test_open_device_dlpack(self):
        # [FUNCTION DOC] Exports a random openreg tensor to a DLPack capsule, imports it back,
        # and checks the device is preserved and the values match after moving both to CPU.
        """Test DLPack conversion for openreg device"""
        x_in = torch.randn(2, 3).to("openreg")
        capsule = torch.utils.dlpack.to_dlpack(x_in)
        x_out = torch.from_dlpack(capsule)
        self.assertTrue(x_out.device == x_in.device)

        x_in = x_in.to("cpu")
        x_out = x_out.to("cpu")
        self.assertEqual(x_in, x_out)

    def test_dlpack_roundtrip(self):
        # [FUNCTION DOC] DLPack export then import of an openreg tensor preserves its device
        # and values.
        """Test DLPack roundtrip conversion"""
        x = torch.randn(2, 3, device="openreg")
        capsule = torch.utils.dlpack.to_dlpack(x)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x.device, y.device)
        self.assertEqual(x, y)

    def test_dlpack_different_shapes(self):
        # [FUNCTION DOC] Loops over several tensor ranks and shapes, roundtripping each through
        # DLPack and checking shape and values are preserved.
        """Test DLPack with different tensor shapes"""
        shapes = [(1,), (2, 3), (4, 5, 6), (1, 2, 3, 4)]

        for shape in shapes:
            x = torch.randn(*shape, device="openreg")
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x, y)

    @unittest.skip("Abs kernel only supports float type when assertEuqal")
    def test_dlpack_different_dtypes(self):
        # [FUNCTION DOC] Roundtrips tensors of several dtypes through DLPack and checks dtype
        # and values. Currently skipped because the abs kernel only supports float types.
        # KNOWN ISSUE: random generation for integer dtypes via the normal-distribution
        # factory would fail for int32/int64, so this test would break even if unskipped.
        """Test DLPack with different dtypes"""
        dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.randn(2, 3, device="openreg", dtype=dtype)
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x, y)

    def test_dlpack_cross_device(self):
        # [FUNCTION DOC] Copies a CPU tensor to openreg, roundtrips it through DLPack, and
        # checks the result lives on openreg and equals the original CPU values.
        """Test DLPack conversion across devices"""
        x_cpu = torch.randn(2, 3)
        x_openreg = x_cpu.to("openreg")

        capsule = torch.utils.dlpack.to_dlpack(x_openreg)
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(x_cpu, y.cpu())

    def test_dlpack_non_contiguous(self):
        # [FUNCTION DOC] Roundtrips a transposed (non-contiguous) openreg tensor through DLPack
        # and checks shape and values are preserved.
        """Test DLPack with non-contiguous tensors"""
        x = torch.randn(3, 4, device="openreg")
        x_t = x.t()  # Transpose creates non-contiguous tensor

        capsule = torch.utils.dlpack.to_dlpack(x_t)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x_t.shape, y.shape)
        self.assertEqual(x_t, y)


if __name__ == "__main__":
    run_tests()
