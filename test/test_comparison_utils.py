#!/usr/bin/env python3
# Owner(s): ["module: internals"]
# [MODULE DOC] Tests for the private tensor metadata assertion operator: matching metadata
# passes, and a mismatched dtype, strides, sizes, device or layout raises RuntimeError.

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestComparisonUtils(TestCase):
    # [CLASS DOC] Test case exercising the assert-tensor-metadata operator with size, stride,
    # dtype, device and layout arguments.
    def test_all_equal_no_assert(self):
        # [FUNCTION DOC] Matching sizes, strides and dtype for a one-element float tensor: no
        # error.
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    def test_all_equal_no_assert_nones(self):
        # [FUNCTION DOC] All expectations given as None: nothing is checked and no error is
        # raised.
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self):
        # [FUNCTION DOC] Expects RuntimeError when the expected dtype (int32) differs from the
        # actual float.
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    def test_assert_strides(self):
        # [FUNCTION DOC] Expects RuntimeError when the expected stride (3) differs from the
        # actual one.
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    def test_assert_sizes(self):
        # [FUNCTION DOC] Expects RuntimeError when the expected size (3) differs from the actual
        # one.
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_assert_device(self):
        # [FUNCTION DOC] Skipped without CUDA. Expects RuntimeError when a CPU tensor is checked
        # against a cuda device expectation.
        t = torch.tensor([0.5], device="cpu")

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, device="cuda")

    def test_assert_layout(self):
        # [FUNCTION DOC] Expects RuntimeError when a strided tensor is checked against sparse
        # coo layout.
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, layout=torch.sparse_coo)


if __name__ == "__main__":
    run_tests()
