# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for tensor creation ops (randperm, triu_indices, tril_indices)
# checking the default and explicitly requested result dtype. Importing the module appends the
# test directory to the system path.

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestTensorCreationOps(JitTestCase):
    # [CLASS DOC] JitTestCase suite for tensor-creating ops in TorchScript; each test scripts a
    # function that creates a tensor and asserts its dtype inside the scripted code.
    """
    A suite of tests for ops that create tensors.
    """

    def test_randperm_default_dtype(self):
        # [FUNCTION DOC] Checks that randperm with no dtype yields an int64 tensor in script and
        # eager mode.
        def randperm(x: int):
            # [FUNCTION DOC] HELPER: Creates a random permutation of the given length and
            # asserts int64 dtype.
            perm = torch.randperm(x)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.int64  # noqa: S101

        self.checkScript(randperm, (3,))

    def test_randperm_specifed_dtype(self):
        # [FUNCTION DOC] Checks that randperm honours an explicit float dtype in script and
        # eager mode.
        def randperm(x: int):
            # [FUNCTION DOC] HELPER: Creates a random permutation with float dtype and asserts
            # that dtype.
            perm = torch.randperm(x, dtype=torch.float)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.float  # noqa: S101

        self.checkScript(randperm, (3,))

    def test_triu_indices_default_dtype(self):
        # [FUNCTION DOC] Checks that triu_indices with no dtype yields int64 indices in script
        # and eager mode.
        def triu_indices(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: Creates upper-triangle indices for the given rows and
            # columns and asserts int64.
            indices = torch.triu_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64  # noqa: S101

        self.checkScript(triu_indices, (3, 3))

    def test_triu_indices_specified_dtype(self):
        # [FUNCTION DOC] Checks that triu_indices honours an explicit int32 dtype in script and
        # eager mode.
        def triu_indices(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: Creates upper-triangle indices with int32 dtype and asserts
            # that dtype.
            indices = torch.triu_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32  # noqa: S101

        self.checkScript(triu_indices, (3, 3))

    def test_tril_indices_default_dtype(self):
        # [FUNCTION DOC] Checks that tril_indices with no dtype yields int64 indices in script
        # and eager mode.
        def tril_indices(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: Creates lower-triangle indices for the given rows and
            # columns and asserts int64.
            indices = torch.tril_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64  # noqa: S101

        self.checkScript(tril_indices, (3, 3))

    def test_tril_indices_specified_dtype(self):
        # [FUNCTION DOC] Checks that tril_indices honours an explicit int32 dtype in script and
        # eager mode.
        def tril_indices(rows: int, cols: int):
            # [FUNCTION DOC] HELPER: Creates lower-triangle indices with int32 dtype and asserts
            # that dtype.
            indices = torch.tril_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32  # noqa: S101

        self.checkScript(tril_indices, (3, 3))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
