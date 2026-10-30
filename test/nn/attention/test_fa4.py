# Owner(s): ["module: sdpa"]
# [MODULE DOC] Tests for the FlashAttention 4 (FA4) backend of scaled dot product attention.
# Defines a CUDA-only test class using the shared FlashAttentionTestMixin for correctness, kernel
# dispatch, compile behavior, and a mocked check that the deterministic flag reaches the FA4
# backward kernel. Skipped unless an SM90 or SM100 GPU and flash_attn.cute.interface exist.
# Import-time side effect: instantiates device-type tests for cuda only.

import importlib
import unittest
from unittest.mock import patch

from _fa_test_common import FlashAttentionTestMixin, SdpaShape

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend
from torch.nn.attention import activate_flash_attention_impl, sdpa_kernel
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


def _fa4_dependencies_available() -> bool:
    # [FUNCTION DOC] HELPER: Report whether FA4 can run here. Returns False without CUDA, when
    # the device major compute capability is not 9 or 10, or when flash_attn.cute.interface
    # cannot be imported; otherwise True.
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major not in (9, 10):
        return False
    try:
        importlib.import_module("flash_attn.cute.interface")
    except ModuleNotFoundError:
        return False
    return True


class TestFlashAttentionFA4(FlashAttentionTestMixin, TestCase):
    # [CLASS DOC] CUDA test case for the FA4 implementation, mixing in shared flash attention
    # tests. Class attributes set impl_name FA4, forward kernel name patterns, and no backward
    # patterns (backward kernels are not profiled).
    hw_classification = HardwareClassification.CUDA

    # Mixin configuration
    impl_name = "FA4"
    fwd_kernel_patterns = ["flash_attncute", "flash_fwd"]
    bwd_kernel_patterns = []

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class fixture. Runs the parent setup, then if FA4 dependencies
        # are available activates the FA4 implementation globally (not undone afterwards; the
        # TODO notes it may pollute other tests).
        super().setUpClass()
        if not _fa4_dependencies_available():
            return
        # This might pollute tests.. TODO
        activate_flash_attention_impl("FA4")

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("batch", [1, 2])
    @parametrize(
        "seq_len",
        [
            512,
            1024,
        ],
    )
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize(
        "is_causal",
        [False, True],
    )
    def test_flash_attention_matches_math(
        self, device, dtype, batch, seq_len, heads, head_dim, is_causal
    ):
        # [FUNCTION DOC] Parametrized over dtype, batch, sequence length, heads, head dim and
        # causality. Checks flash output against the math backend; the backward pass is only
        # tested for head dim 128 with float16, because other cases currently fail (per the
        # inline comments).
        # TODO: Getting bad TMA setup on dO w/ headdim = 64, will take a look
        test_backward = head_dim == 128 and dtype == torch.float16
        shape = SdpaShape(batch, heads, seq_len, head_dim)
        self._assert_flash_matches_math(
            device,
            shape=shape,
            dtype=dtype,
            is_causal=is_causal,
            # Bwd is consistently erroring
            test_backward=test_backward,
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fa4_kernel_called(self, device, dtype):
        # [FUNCTION DOC] Delegates to the mixin check that the FA4 forward kernel shows up in a
        # profile.
        self._test_kernel_called(device, dtype)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_multiple_activate(self):
        # [FUNCTION DOC] Delegates to the mixin check that activating implementations repeatedly
        # behaves correctly.
        self._test_multiple_activate_impl()

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_metadata(self, device, dtype, is_causal):
        # [FUNCTION DOC] Delegates to the mixin check that compiled SDPA preserves output shape,
        # stride, dtype.
        """Test that torch.compile preserves tensor metadata (shape, stride, dtype)."""
        self._test_compiled_sdpa_metadata(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_matches_math(self, device, dtype, is_causal):
        # [FUNCTION DOC] Delegates to the mixin check that compiled FA4 output matches the math
        # backend.
        """Test compiled FA4 numerical correctness against math backend."""
        self._test_compiled_sdpa_matches_math(device, dtype, is_causal)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("dtype", [torch.float16])
    @parametrize("is_causal", [False, True])
    def test_compiled_sdpa_fa4_backward_matches_math(self, device, dtype, is_causal):
        # [FUNCTION DOC] Delegates to the mixin check that compiled FA4 gradients match the math
        # backend for float16 at head dim 128 with default tolerances.
        """Test compiled FA4 backward numerical correctness against math backend."""
        # FA4 uses head_dim=128, no min_atol override
        self._test_compiled_sdpa_backward_matches_math(
            device, dtype, is_causal, head_dim=128
        )

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    def test_attention_preserves_query_layout(self, device):
        # [FUNCTION DOC] Delegates to the mixin check that the attention output keeps the query
        # memory layout.
        """Test that FA4 output has the same layout as the query tensor."""
        self._test_attention_preserves_query_layout(device)

    @unittest.skipUnless(_fa4_dependencies_available(), "FA4 backend unavailable")
    @parametrize("deterministic", [False, True])
    def test_deterministic_flag_passed_to_backward(self, device, deterministic):
        # [FUNCTION DOC] Mocks the FA4 module import so forward and backward kernels return
        # random tensors, sets global deterministic algorithms mode to the parameter, runs SDPA
        # forward and backward under the flash backend, and asserts the backward kernel was
        # called with a deterministic keyword equal to the parameter. The finally block resets
        # deterministic mode to off and clears the import cache (also cleared before the mock is
        # installed).
        """Test that deterministic flag is correctly passed through to FA4 backward kernel."""
        from torch.nn.attention import _fa4

        shape = SdpaShape(2, 4, 512, 128)
        q = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
        k = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
        v = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)

        torch.use_deterministic_algorithms(deterministic)

        try:
            _fa4._fa4_import_module.cache_clear()

            with patch("torch.nn.attention._fa4._fa4_import_module") as mock_import:
                mock_module = mock_import.return_value

                # FA4 uses BSHD layout internally, so mock returns BSHD
                q_transposed = q.transpose(1, 2)
                mock_module._flash_attn_fwd.return_value = (
                    torch.randn_like(q_transposed),
                    torch.randn(
                        q.size(0),
                        q.size(2),
                        q.size(1),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                mock_module._flash_attn_bwd.return_value = (
                    torch.randn_like(q_transposed),
                    torch.randn_like(q_transposed),
                    torch.randn_like(q_transposed),
                )

                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    grad_out = torch.randn_like(out)
                    out.backward(grad_out)

                self.assertTrue(mock_module._flash_attn_bwd.called)
                call_kwargs = mock_module._flash_attn_bwd.call_args.kwargs
                self.assertIn("deterministic", call_kwargs)
                self.assertEqual(
                    call_kwargs["deterministic"],
                    deterministic,
                    lambda msg: f"{msg}\nExpected deterministic={deterministic} but got {call_kwargs['deterministic']}",
                )
        finally:
            torch.use_deterministic_algorithms(False)
            _fa4._fa4_import_module.cache_clear()


instantiate_device_type_tests(TestFlashAttentionFA4, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
