# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for the Triton outer-product override of batched matmul. Device-generic
# accelerator tests check numerics against the matmul operator for many shapes, dtypes, permuted
# layouts, gradients, device mismatches and copy-on-write inputs; CPU tests check the fallback
# and the shape-detection helper. Instantiates the device tests (CUDA and XPU allowed) at import
# time.

import torch
from torch._native.ops.bmm_outer_product.triton_impl import (
    _bmm_outer_product_cond,
    _is_outer_product,
)
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyAccelerator,
    skipXPUIf,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestBmmOuterProductDevice(TestCase):
    # [CLASS DOC] Accelerator-only tests (via the onlyAccelerator decorator) comparing bmm
    # results with the matmul operator when inputs have an outer-product shape.
    # hw_classification marks it as accelerator hardware.
    hw_classification = HardwareClassification.ACCELERATOR

    def _check_bmm(self, a, b, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts that bmm of the two tensors equals the matmul operator
        # result, forwarding extra keyword arguments (tolerances) to assertEqual.
        self.assertEqual(torch.bmm(a, b), a @ b, **kwargs)

    @onlyAccelerator
    @skipXPUIf(True, "https://github.com/pytorch/pytorch/issues/180318")
    def test_shapes(self, device):
        # [FUNCTION DOC] For several batch/rows/columns shapes, builds random column and row
        # operands (trailing and middle dimension of size one) on the device and checks bmm
        # against matmul. Skipped on XPU because of a known issue; each shape runs in a subTest.
        shapes = [
            (4, 8, 16),
            (32, 8, 256),
            (16, 128, 512),
            (1, 64, 128),
            (8, 1, 1),
            (64, 256, 512),
            (256, 8, 2048),
        ]
        for B, M, N in shapes:
            with self.subTest(B=B, M=M, N=N):
                a = torch.randn(B, M, 1, device=device)
                b = torch.randn(B, 1, N, device=device)
                self._check_bmm(a, b)

    @onlyAccelerator
    def test_basic_dtypes(self, device):
        # [FUNCTION DOC] Checks the outer-product bmm result equals matmul for float32, float16
        # and bfloat16 inputs, one subTest per dtype.
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                a = torch.randn(4, 8, 1, device=device, dtype=dtype)
                b = torch.randn(4, 1, 16, device=device, dtype=dtype)
                self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_permuted_inputs(self, device):
        # [FUNCTION DOC] Builds operands through permute (non-contiguous strides) in three
        # combinations (first, second, both) and checks bmm equals matmul for each.
        B, M, N = 4, 8, 16
        cases = [
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(B, 1, N, device=device),
            ),
            (
                torch.randn(B, M, 1, device=device),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
            (
                torch.randn(M, B, 1, device=device).permute(1, 0, 2),
                torch.randn(N, B, 1, device=device).permute(1, 2, 0),
            ),
        ]
        for a, b in cases:
            self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_fallback_non_outer_product(self, device):
        # [FUNCTION DOC] Uses a genuine (non outer-product) batched matmul shape to confirm the
        # fallback path still gives matmul results, with slightly loosened tolerances.
        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 32, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b, atol=1e-5, rtol=1.3e-6)

    @onlyAccelerator
    def test_batch_one(self, device):
        # [FUNCTION DOC] Edge case of batch size one: bmm must equal matmul for a 64x1 by 1x128
        # product.
        a = torch.randn(1, 64, 1, device=device)
        b = torch.randn(1, 1, 128, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_m_one_n_one(self, device):
        # [FUNCTION DOC] Edge case where both row and column counts are one (8 batches of 1x1
        # products); bmm must equal matmul.
        a = torch.randn(8, 1, 1, device=device)
        b = torch.randn(8, 1, 1, device=device)
        self.assertEqual(torch.bmm(a, b), a @ b)

    @onlyAccelerator
    def test_gradient_flow(self, device):
        # [FUNCTION DOC] Runs bmm on operands requiring gradients, backpropagates the summed
        # result and checks both gradients exist and have the same shapes as the inputs.
        a = torch.randn(4, 8, 1, device=device, requires_grad=True)
        b = torch.randn(4, 1, 16, device=device, requires_grad=True)
        result = torch.bmm(a, b)
        result.sum().backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)

    @onlyAccelerator
    def test_mixed_device_outer_product_fallback(self, device):
        # [FUNCTION DOC] Mixes a CPU tensor with an accelerator tensor and expects bmm to raise
        # a RuntimeError (the override must not claim mismatched devices).
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16, device=device)
        with self.assertRaises(RuntimeError):
            torch.bmm(a, b)

    @onlyAccelerator
    @deviceCountAtLeast(2)
    def test_non_current_device_outer_product(self, devices):
        # [FUNCTION DOC] Requires at least two devices (argument is the list of device strings).
        # Sets the current accelerator index to the first device, runs bmm on tensors of the
        # second device, and checks the current device is unchanged, the output lives on the
        # second device and values equal the elementwise product. Then checks mixing devices
        # raises RuntimeError. The original current device index is restored in a finally block.
        old_device = torch.accelerator.current_device_index()
        try:
            torch.accelerator.set_device_index(devices[0])
            a = torch.randn(4, 8, 1, device=devices[1])
            b = torch.randn(4, 1, 16, device=devices[1])

            out = torch.bmm(a, b)

            self.assertEqual(
                torch.accelerator.current_device_index(), int(devices[0].split(":")[1])
            )
            self.assertEqual(out.device, torch.device(devices[1]))
            self.assertEqual(out, a * b)

            mismatched_a = torch.randn(4, 8, 1, device=devices[0])
            with self.assertRaisesRegex(RuntimeError, "same device|different"):
                torch.bmm(mismatched_a, b)
        finally:
            torch.accelerator.set_device_index(old_device)

    @onlyAccelerator
    def test_cow_inputs_accepted_by_override(self, device):
        # [FUNCTION DOC] Creates lazy-clone (copy-on-write) inputs, asserts they are COW, and
        # asserts the override condition function accepts them (regression guard against
        # excluding COW inputs).
        # The override accepts copy-on-write inputs (it wraps its read-only
        # inputs in ConstTensorWrapper and reads through const_data_ptr()).
        # Assert the cond fires on COW inputs -- it previously excluded them --
        # so a regression back to declining COW would be caught here.
        a = torch.randn(4, 64, 1, device=device)
        b = torch.randn(4, 1, 48, device=device)
        a_cow = a._lazy_clone()
        b_cow = b._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))

        self.assertTrue(_bmm_outer_product_cond(a_cow, b_cow))

    @onlyAccelerator
    def test_cow_inputs_not_materialized(self, device):
        # [FUNCTION DOC] Runs bmm on copy-on-write inputs, checks the result equals the plain
        # product and that both inputs remain copy-on-write afterwards (not materialized).
        # A copy-on-write input routed through the override is read via
        # const_data_ptr() and not materialized. Verify the result is correct
        # and the inputs stay COW across the call.
        a = torch.randn(4, 64, 1, device=device)
        b = torch.randn(4, 1, 48, device=device)
        a_cow = a._lazy_clone()
        b_cow = b._lazy_clone()
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))

        out = torch.bmm(a_cow, b_cow)

        self.assertEqual(out, a @ b)
        self.assertTrue(torch._C._is_cow_tensor(a_cow))
        self.assertTrue(torch._C._is_cow_tensor(b_cow))


class TestBmmOuterProduct(TestCase):
    # [CLASS DOC] CPU-only test case: the override is accelerator-only, so CPU must behave as
    # normal bmm.
    def test_cpu_outer_product_fallback(self):
        # [FUNCTION DOC] Asserts CPU outer-product shaped inputs give the same result as matmul.
        a = torch.randn(4, 8, 1)
        b = torch.randn(4, 1, 16)
        self.assertTrue(a.device.type == "cpu")
        self.assertEqual(torch.bmm(a, b), a @ b)


class TestOuterProductDetection(TestCase):
    # [CLASS DOC] Unit tests for the private shape-detection helper that decides whether two
    # tensors form a batched outer product.
    def test_is_outer_product(self):
        # [FUNCTION DOC] Checks accepted cases (3-D, trailing one in first, middle one in
        # second, also all-one second) and rejected cases: real matmul shapes, 2-D inputs,
        # mismatched inner dimension, inner size greater than one, and complex dtype.
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 16)))
        self.assertTrue(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 1, 1)))
        self.assertFalse(
            _is_outer_product(torch.empty(4, 8, 16), torch.empty(4, 16, 32))
        )
        self.assertFalse(_is_outer_product(torch.empty(8, 1), torch.empty(1, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 1), torch.empty(4, 2, 16)))
        self.assertFalse(_is_outer_product(torch.empty(4, 8, 3), torch.empty(4, 1, 16)))
        self.assertFalse(
            _is_outer_product(
                torch.empty(4, 8, 1, dtype=torch.complex64),
                torch.empty(4, 1, 16, dtype=torch.complex64),
            )
        )


instantiate_device_type_tests(TestBmmOuterProductDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
