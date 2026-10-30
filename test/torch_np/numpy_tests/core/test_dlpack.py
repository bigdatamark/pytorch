# Owner(s): ["module: dynamo"]
# [MODULE DOC] NumPy DLPack tests ported to torch._numpy (or real NumPy under dynamo). Checks
# refcounts, dtype passthrough, non-contiguous inputs, many dimensions, device reporting, zero-
# dim arrays and interop with torch tensors through the DLPack protocol.

import functools
import sys
import unittest
from unittest import skipIf as skipif

import numpy
import pytest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_array_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_array_equal


skip = functools.partial(skipif, True)


IS_PYPY = False


@skipif(numpy.__version__ < "1.24", reason="numpy.dlpack is new in numpy 1.23")
@instantiate_parametrized_tests
class TestDLPack(TestCase):
    # [CLASS DOC] Test case for __dlpack__, __dlpack_device__ and from_dlpack; skipped when
    # the installed NumPy is older than 1.24. Parametrized tests are instantiated.
    @xpassIfTorchDynamo_np  # (reason="pytorch seems to handle refcounts differently")
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    def test_dunder_dlpack_refcount(self):
        # [FUNCTION DOC] Export an array via the protocol and check the array refcount
        # rises to 3 while the capsule lives and falls back to 2 after it is deleted.
        # Expected to pass only under dynamo (xpass decorator); skipped on PyPy.
        x = np.arange(5)
        y = x.__dlpack__()
        if sys.getrefcount(x) != 3:
            raise AssertionError(f"Expected refcount 3, got {sys.getrefcount(x)}")
        del y
        if sys.getrefcount(x) != 2:
            raise AssertionError(f"Expected refcount 2, got {sys.getrefcount(x)}")

    @unittest.expectedFailure
    @skipIfTorchDynamo("I can't figure out how to get __dlpack__ into trace_rules.py")
    def test_dunder_dlpack_stream(self):
        # [FUNCTION DOC] Call the export with stream None (should work) and with stream
        # 1 (expected to raise RuntimeError). Marked expected failure because the method
        # is not traceable by dynamo.
        x = np.arange(5)
        x.__dlpack__(stream=None)

        with pytest.raises(RuntimeError):
            x.__dlpack__(stream=1)

    @xpassIfTorchDynamo_np  # (reason="pytorch seems to handle refcounts differently")
    @skipif(IS_PYPY, reason="PyPy can't get refcounts.")
    def test_from_dlpack_refcount(self):
        # [FUNCTION DOC] Same refcount check as the export test but through from_dlpack:
        # refcount of the source array is 3 while the imported result lives and 2 after
        # deleting it.
        x = np.arange(5)
        y = np.from_dlpack(x)
        if sys.getrefcount(x) != 3:
            raise AssertionError(f"Expected refcount 3, got {sys.getrefcount(x)}")
        del y
        if sys.getrefcount(x) != 2:
            raise AssertionError(f"Expected refcount 2, got {sys.getrefcount(x)}")

    @parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ],
    )
    def test_dtype_passthrough(self, dtype):
        # [FUNCTION DOC] For each parametrized dtype, import an arange array through
        # from_dlpack and assert the dtype is preserved and the values are equal.
        x = np.arange(5, dtype=dtype)
        y = np.from_dlpack(x)

        if y.dtype != x.dtype:
            raise AssertionError(
                f"Expected y.dtype == x.dtype, got {y.dtype} vs {x.dtype}"
            )
        assert_array_equal(x, y)

    def test_non_contiguous(self):
        # [FUNCTION DOC] Round trip several non-contiguous or derived views (row,
        # column, a row by index, a copied diagonal) of a 5 by 5 array through
        # from_dlpack and compare values.
        x = np.arange(25).reshape((5, 5))

        y1 = x[0]
        assert_array_equal(y1, np.from_dlpack(y1))

        y2 = x[:, 0]
        assert_array_equal(y2, np.from_dlpack(y2))

        y3 = x[1, :]
        assert_array_equal(y3, np.from_dlpack(y3))

        y4 = x[1]
        assert_array_equal(y4, np.from_dlpack(y4))

        y5 = np.diagonal(x).copy()
        assert_array_equal(y5, np.from_dlpack(y5))

    @parametrize("ndim", range(33))
    def test_higher_dims(self, ndim):
        # [FUNCTION DOC] For each parametrized ndim from 0 to 32, make an all-ones-shape
        # float64 zeros array and check from_dlpack keeps the shape.
        shape = (1,) * ndim
        x = np.zeros(shape, dtype=np.float64)

        if shape != np.from_dlpack(x).shape:
            raise AssertionError(
                f"Expected shape {shape}, got {np.from_dlpack(x).shape}"
            )

    def test_dlpack_device(self):
        # [FUNCTION DOC] Check the device tuple is (1, 0) (CPU) for an original array,
        # for the imported array, and for a strided slice of it.
        x = np.arange(5)
        if x.__dlpack_device__() != (1, 0):
            raise AssertionError(
                f"Expected x.__dlpack_device__() == (1, 0), got {x.__dlpack_device__()}"
            )
        y = np.from_dlpack(x)
        if y.__dlpack_device__() != (1, 0):
            raise AssertionError(
                f"Expected y.__dlpack_device__() == (1, 0), got {y.__dlpack_device__()}"
            )
        z = y[::2]
        if z.__dlpack_device__() != (1, 0):
            raise AssertionError(
                f"Expected z.__dlpack_device__() == (1, 0), got {z.__dlpack_device__()}"
            )

    def dlpack_deleter_exception(self):
        # [FUNCTION DOC] HELPER: Export an array via the protocol, then unconditionally
        # raise RuntimeError, used to check that an error does not break the capsule
        # destructor.
        x = np.arange(5)
        _ = x.__dlpack__()
        raise RuntimeError

    def test_dlpack_destructor_exception(self):
        # [FUNCTION DOC] Check that the helper's RuntimeError propagates to the caller.
        with pytest.raises(RuntimeError):
            self.dlpack_deleter_exception()

    @skip(reason="no readonly arrays in pytorch")
    def test_readonly(self):
        # [FUNCTION DOC] Make an array read-only and expect export to raise BufferError.
        # Skipped because torch has no read-only arrays.
        x = np.arange(5)
        x.flags.writeable = False
        with pytest.raises(BufferError):
            x.__dlpack__()

    def test_ndim0(self):
        # [FUNCTION DOC] A zero-dimensional array survives from_dlpack with equal value.
        x = np.array(1.0)
        y = np.from_dlpack(x)
        assert_array_equal(x, y)

    def test_from_torch(self):
        # [FUNCTION DOC] Import a torch tensor with from_dlpack and compare it to
        # asarray of the tensor.
        t = torch.arange(4)
        a = np.from_dlpack(t)
        assert_array_equal(a, np.asarray(t))

    def test_to_torch(self):
        # [FUNCTION DOC] Convert an array to a torch tensor with torch.from_dlpack and
        # compare values.
        a = np.arange(4)
        t = torch.from_dlpack(a)
        assert_array_equal(np.asarray(t), a)


if __name__ == "__main__":
    run_tests()
