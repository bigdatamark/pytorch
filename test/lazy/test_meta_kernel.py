# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for lazy tensor meta kernels (dtype and device handling of addmm and add on
# the lazy device). Import-time side effect: initializes the TS lazy backend. The file is not run
# by default; running it directly raises an error saying it must be enabled in the test discovery
# list first.

import torch
import torch._lazy
import torch._lazy.ts_backend
from torch import float16, float32
from torch.testing._internal.common_utils import TestCase


torch._lazy.ts_backend.init()


class TestMetaKernel(TestCase):
    # [CLASS DOC] Test case for output dtype and device checks done by lazy meta kernels.
    def test_addmm_invalid_dtype(self):
        # [FUNCTION DOC] Applies a float32 bias-free linear layer to a float16 lazy input and
        # asserts that an exception is raised due to the dtype mismatch.
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertTrue(input.dtype == torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")

        with self.assertRaises(Exception):
            fc_nobias(input)

    def test_addmm(self):
        # [FUNCTION DOC] Applies float16 linear layers, without and with bias, to a float16 lazy
        # input and asserts the output dtype is float16 in both cases.
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertEqual(input.dtype, torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float16).to("lazy")
        out_nobias = fc_nobias(input)
        self.assertEqual(out_nobias.dtype, torch.float16)

        fc_bias = torch.nn.Linear(2, 2, bias=True, dtype=float16).to("lazy")
        out_bias = fc_bias(input)
        self.assertEqual(out_bias.dtype, torch.float16)

    def test_add_invalid_device(self):
        # [FUNCTION DOC] Adds a CPU tensor to a lazy tensor and asserts a runtime error whose
        # message says one operand is not a lazy tensor.
        with self.assertRaisesRegex(RuntimeError, ".*not a lazy tensor.*"):
            _ = torch.tensor([1], device="cpu") + torch.tensor([1], device="lazy")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
