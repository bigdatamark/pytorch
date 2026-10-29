# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] Tests for autocast support of the openreg PrivateUse1 backend.
# Checks the default and settable autocast dtype, which ops run in low precision
# (mm) versus float32 (asin), nested and disabled contexts, the unsupported-dtype
# warning and the unsafe binary_cross_entropy error. Uses the "openreg" device
# registered by torch_openreg. No import-time side effects.

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutocast(TestCase):
    # [CLASS DOC] Device-specific autocast tests for openreg; all methods are
    # independent test cases that enter torch autocast contexts on the openreg device.
    def test_autocast_with_unsupported_type(self):
        # [FUNCTION DOC] Entering autocast with float32 on openreg must emit a UserWarning
        # saying the dtype is unsupported (only float16 and bfloat16 are) and disable autocast.
        """Test autocast with unsupported dtype (float32)"""
        with self.assertWarnsRegex(
            UserWarning,
            "In openreg autocast, but the target dtype is not supported. Disabling autocast.\n"
            "openreg Autocast only supports dtypes of torch.float16, torch.bfloat16 currently.",
        ):
            with torch.autocast(device_type="openreg", dtype=torch.float32):
                _ = torch.ones(10)

    def test_autocast_operator_not_supported(self):
        # [FUNCTION DOC] Under float16 autocast, binary_cross_entropy on openreg tensors
        # must raise a RuntimeError stating it is unsafe to autocast.
        """Test that binary_cross_entropy is not supported in autocast"""
        with self.assertRaisesRegex(
            RuntimeError,
            "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.",
        ):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(2, 3, device="openreg")
            with torch.autocast(device_type="openreg", dtype=torch.float16):
                _ = torch.nn.functional.binary_cross_entropy(x, y)

    def test_autocast_low_precision(self):
        # [FUNCTION DOC] Under float16 autocast, a matrix multiply of float32 openreg inputs
        # must produce a float16 result.
        """Test low precision operations (mm) in autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_fp32(self):
        # [FUNCTION DOC] Under default autocast, asin applied to a float16 openreg tensor
        # must produce a float32 result (fp32 op list).
        """Test fp32 operations (asin) in autocast"""
        with torch.amp.autocast(device_type="openreg"):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.asin(x)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_default_dtype(self):
        # [FUNCTION DOC] The default autocast dtype for openreg must be half precision.
        # KNOWN ISSUE: depends on test order, since test_autocast_set_dtype changes the
        # global autocast dtype and never restores it.
        """Test default autocast dtype"""
        openreg_fast_dtype = torch.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, torch.half)

    def test_autocast_set_dtype(self):
        # [FUNCTION DOC] Sets the global autocast dtype for openreg to float16 and then
        # bfloat16 and checks each is read back.
        # KNOWN ISSUE: leaves the process-wide dtype at bfloat16 (no restore), which can
        # leak into later tests.
        """Test setting autocast dtype"""
        for dtype in [torch.float16, torch.bfloat16]:
            torch.set_autocast_dtype("openreg", dtype)
            self.assertEqual(torch.get_autocast_dtype("openreg"), dtype)

    def test_autocast_bfloat16(self):
        # [FUNCTION DOC] Under bfloat16 autocast, matrix multiply of float32 inputs must
        # yield a bfloat16 result.
        """Test autocast with bfloat16 dtype"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float32)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_autocast_low_precision_bfloat16(self):
        # [FUNCTION DOC] Same as the bfloat16 matmul check but with default-dtype random
        # inputs; the result must be bfloat16.
        """Test low precision operations with bfloat16"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_autocast_fp32_with_bfloat16(self):
        # [FUNCTION DOC] Under bfloat16 autocast, asin of a bfloat16 tensor must come
        # out as float32.
        """Test fp32 operations with bfloat16 autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
            x = torch.randn(2, device="openreg", dtype=torch.bfloat16)
            result = torch.asin(x)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_nested_context(self):
        # [FUNCTION DOC] A float16 autocast region containing a bfloat16 region: matmul
        # yields float16 outside, bfloat16 inside, and float16 again after the inner exit.
        """Test nested autocast contexts"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result1 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)

            # Nested autocast context with bfloat16
            with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
                result2 = torch.mm(x, y)
                self.assertEqual(result2.dtype, torch.bfloat16)

            # After exiting nested context, should restore to float16
            result3 = torch.mm(x, y)
            self.assertEqual(result3.dtype, torch.float16)

    def test_autocast_fallthrough_operation(self):
        # [FUNCTION DOC] An op with no autocast rule (add) must fall through and keep the
        # float32 input dtype under float16 autocast.
        """Test fallthrough operations (operations not specially registered)"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            # add operation is not specially registered, should fallthrough
            result = torch.add(x, x)
            # fallthrough operations should preserve input type or use default behavior
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_with_requires_grad(self):
        # [FUNCTION DOC] Autocast matmul on inputs requiring grad must give a float16
        # result that requires grad, and backward must populate both input gradients.
        """Test autocast interaction with requires_grad"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", requires_grad=True)
            y = torch.randn(3, 3, device="openreg", requires_grad=True)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)
            self.assertTrue(result.requires_grad)

            # Test backward propagation
            loss = result.sum()
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertIsNotNone(y.grad)

    def test_autocast_mixed_input_dtypes(self):
        # [FUNCTION DOC] A matmul with one float32 and one float16 input under float16
        # autocast must succeed and return float16.
        """Test combinations of different input dtypes"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float16)
            # mm operation should convert inputs to low precision
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_already_target_dtype(self):
        # [FUNCTION DOC] Matmul of inputs already in float16 under float16 autocast keeps
        # float16.
        """Test when inputs are already in target dtype"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float16)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float16)
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_combination_operations(self):
        # [FUNCTION DOC] Chains ops under float16 autocast: matmul gives float16, asin
        # gives float32, and a second matmul of the float16 result gives float16.
        """Test multiple operations combination under autocast"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            z = torch.randn(2, device="openreg")

            # Low precision operation
            result1 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)

            # fp32 operation
            result2 = torch.asin(z)
            self.assertEqual(result2.dtype, torch.float32)

            # Combined operations
            result3 = torch.mm(result1, y)
            self.assertEqual(result3.dtype, torch.float16)

    def test_autocast_disable(self):
        # [FUNCTION DOC] With enabled=False the matmul of float32 inputs must stay float32.
        """Test disabling autocast"""
        with torch.amp.autocast(
            device_type="openreg", dtype=torch.float16, enabled=False
        ):
            x = torch.randn(2, 3, device="openreg", dtype=torch.float32)
            y = torch.randn(3, 3, device="openreg", dtype=torch.float32)
            result = torch.mm(x, y)
            # When autocast is disabled, should preserve original dtype
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_cache_enabled(self):
        # [FUNCTION DOC] With cache_enabled=True, two identical matmuls both return float16.
        """Test autocast caching"""
        with torch.amp.autocast(
            device_type="openreg", dtype=torch.float16, cache_enabled=True
        ):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result1 = torch.mm(x, y)
            result2 = torch.mm(x, y)
            self.assertEqual(result1.dtype, torch.float16)
            self.assertEqual(result2.dtype, torch.float16)

    def test_autocast_fp32_operation_with_float16_input(self):
        # [FUNCTION DOC] asin on a float16 input under float16 autocast must return float32.
        """Test fp32 operations receiving float16 input"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_fp32_operation_with_float32_input(self):
        # [FUNCTION DOC] asin on a float32 input under float16 autocast must return float32.
        """Test fp32 operations receiving float32 input"""
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, device="openreg", dtype=torch.float32)
            result = torch.asin(x)
            # asin should output float32
            self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
