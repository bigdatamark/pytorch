# Owner(s): ["module: dynamo"]
# [MODULE DOC] Port of NumPy core numeric tests (resize, non-array arguments, bool scalars and
# arrays, error state, float exceptions, and more) to the NumPy-compatible layer. Under
# TorchDynamo it uses real numpy, otherwise torch._numpy and its testing helpers. Import time
# pulls in numpy, hypothesis and pytest, defines flags for wasm and refcount support and a skip
# helper that unconditionally skips. Many classes are marked expected-failure or expected-pass-
# under-dynamo.

import functools
import itertools
import math
import platform
import sys
import warnings

import numpy
import pytest


IS_WASM = False
HAS_REFCOUNT = True

import operator
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.random import rand, randint, randn
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )
else:
    import torch._numpy as np
    from torch._numpy.random import rand, randint, randn
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
        assert_warns,  # assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )


skip = functools.partial(skipif, True)


@instantiate_parametrized_tests
class TestResize(TestCase):
    # [CLASS DOC] Test case for resize: cyclic repetition of input data into a new shape, zero-
    # size targets and invalid shapes.
    def test_copies(self):
        # [FUNCTION DOC] Resizes a 2x2 array to shapes 2x4, 4x2 and 4x3 and checks the data is
        # read cyclically in row-major order to fill the new shape.
        A = np.array([[1, 2], [3, 4]])
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_repeats(self):
        # [FUNCTION DOC] Resizes a 1-D array of three elements to 2x4, 4x2 and 4x3 and checks the
        # elements repeat cyclically to fill the result.
        A = np.array([1, 2, 3])
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_zeroresize(self):
        # [FUNCTION DOC] Resizes a 2x2 array to zero-size shapes and checks the result is empty,
        # keeps the dtype for the flat case and has the requested shape for the 0x2 and 2x0 cases.
        A = np.array([[1, 2], [3, 4]])
        Ar = np.resize(A, (0,))
        assert_array_equal(Ar, np.array([]))
        assert_equal(A.dtype, Ar.dtype)

        Ar = np.resize(A, (0, 2))
        assert_equal(Ar.shape, (0, 2))

        Ar = np.resize(A, (2, 0))
        assert_equal(Ar.shape, (2, 0))

    def test_reshape_from_zero(self):
        # [FUNCTION DOC] Resizes an empty float32 array to 2x1 and checks the result is zeros with
        # unchanged dtype.
        # See also gh-6740
        A = np.zeros(0, dtype=np.float32)
        Ar = np.resize(A, (2, 1))
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
        assert_equal(A.dtype, Ar.dtype)

    def test_negative_resize(self):
        # [FUNCTION DOC] Checks that resizing to a shape with negative dimensions raises
        # RuntimeError or ValueError.
        A = np.arange(0, 10, dtype=np.float32)
        new_shape = (-10, -1)
        with pytest.raises((RuntimeError, ValueError)):
            np.resize(A, new_shape=new_shape)


@instantiate_parametrized_tests
class TestNonarrayArgs(TestCase):
    # [CLASS DOC] Test case checking that top-level functions accept plain Python lists and give
    # the same results as for arrays.
    # check that non-array arguments to functions wrap them in arrays
    def test_choose(self):
        # [FUNCTION DOC] Calls choose with a list of indices and a list of choice rows and checks
        # the selected elements.
        choices = [[0, 1, 2], [3, 4, 5], [5, 6, 7]]
        tgt = [5, 1, 5]
        a = [2, 0, 1]

        out = np.choose(a, choices)
        assert_equal(out, tgt)

    def test_clip(self):
        # [FUNCTION DOC] Calls clip on a list with bounds 2 and 7 and checks the clamped values.
        arr = [-1, 5, 2, 3, 10, -4, -9]
        out = np.clip(arr, 2, 7)
        tgt = [2, 5, 2, 3, 7, 2, 2]
        assert_equal(out, tgt)

    @xpassIfTorchDynamo_np  # (reason="TODO implement compress(...)")
    def test_compress(self):
        # [FUNCTION DOC] Calls compress with a boolean-like selector list along axis 0 and checks
        # the second row is kept. Expected pass-or-fail marker under dynamo.
        arr = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

    def test_count_nonzero(self):
        # [FUNCTION DOC] Calls count_nonzero along axis 1 on a nested list and checks per-row
        # counts 2 and 3.
        arr = [[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]]
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        assert_equal(out, tgt)

    def test_cumproduct(self):
        # [FUNCTION DOC] Checks the cumulative product of a 2x3 nested list is flattened to the
        # six running products.
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.all(np.cumprod(A) == np.array([1, 2, 6, 24, 120, 720])))

    def test_diagonal(self):
        # [FUNCTION DOC] Checks the main diagonal of a 3x4 nested list.
        a = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        out = np.diagonal(a)
        tgt = [0, 5, 10]

        assert_equal(out, tgt)

    def test_mean(self):
        # [FUNCTION DOC] Checks the mean over all, axis 0 and axis 1 of a 2x3 list and that the
        # mean of an empty list is NaN. Commented-out warning checks are not run.
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.mean(A) == 3.5)
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        assert_(np.all(np.mean(A, 1) == np.array([2.0, 5.0])))

        #    with warnings.catch_warnings(record=True) as w:
        #        warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.mean([])))

    #        assert_(w[0].category is RuntimeWarning)

    def test_ptp(self):
        # [FUNCTION DOC] Checks the peak-to-peak range along axis 0 of a list of mixed ints and a
        # float equals 15.0.
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.ptp(a, axis=0), 15.0)

    def test_prod(self):
        # [FUNCTION DOC] Checks the product along the last axis of a 3x4 nested list.
        arr = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]]
        tgt = [24, 1890, 600]

        assert_equal(np.prod(arr, axis=-1), tgt)

    def test_ravel(self):
        # [FUNCTION DOC] Checks ravel flattens a 4x3 nested list into row-major order.
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    def test_repeat(self):
        # [FUNCTION DOC] Checks repeat with count 2 duplicates each element of a list in place.
        a = [1, 2, 3]
        tgt = [1, 1, 2, 2, 3, 3]

        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    def test_reshape(self):
        # [FUNCTION DOC] Checks reshape of a 4x3 nested list to 2x6.
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    def test_round(self):
        # [FUNCTION DOC] Checks rounding to one decimal place on a list of floats and that the
        # round method of a double scalar gives 1.0.
        arr = [1.56, 72.54, 6.35, 3.25]
        tgt = [1.6, 72.5, 6.4, 3.2]
        assert_equal(np.around(arr, decimals=1), tgt)
        s = np.float64(1.0)
        assert_equal(s.round(), 1.0)

    def test_round_2(self):
        # [FUNCTION DOC] Checks the round method of a double scalar returns either a double scalar
        # or an array.
        s = np.float64(1.0)
        assert_(isinstance(s.round(), (np.float64, np.ndarray)))

    @xpassIfTorchDynamo_np  # (reason="scalar instances")
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
        ],
    )
    def test_dunder_round(self, dtype):
        # [FUNCTION DOC] Parametrized over several int and float dtypes: for a scalar holding 1,
        # checks the builtin round with no digits, a None positional argument or a None keyword
        # argument returns a Python int equal to 1. Expected pass-or-fail marker under dynamo.
        s = dtype(1)
        assert_(isinstance(round(s), int))
        assert_(isinstance(round(s, None), int))
        assert_(isinstance(round(s, ndigits=None), int))
        assert_equal(round(s), 1)
        assert_equal(round(s, None), 1)
        assert_equal(round(s, ndigits=None), 1)

    @parametrize(
        "val, ndigits",
        [
            # pytest.param(
            #    2**31 - 1, -1, marks=pytest.mark.xfail(reason="Out of range of int32")
            # ),
            subtest((2**31 - 1, -1), decorators=[xpassIfTorchDynamo_np]),
            subtest(
                (2**31 - 1, 1 - math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
            subtest(
                (2**31 - 1, -math.ceil(math.log10(2**31 - 1))),
                decorators=[xpassIfTorchDynamo_np],
            ),
        ],
    )
    def test_dunder_round_edgecases(self, val, ndigits):
        # [FUNCTION DOC] Parametrized over large int32 value and digit count pairs: checks the
        # builtin round on the Python int equals the builtin round on the same value as an int32
        # scalar. Each subtest has an expected pass-or-fail marker under dynamo.
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    @xfail  # (reason="scalar instances")
    def test_dunder_round_accuracy(self):
        # [FUNCTION DOC] Checks builtin round with negative digits on float64 and int64 scalars
        # keeps the scalar type and is within a few ulps of the expected value. Marked expected
        # failure.
        f = np.float64(5.1 * 10**73)
        assert_(isinstance(round(f, -73), np.float64))
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        i = np.int64(501)
        assert_(isinstance(round(i, -2), np.int64))
        assert_array_max_ulp(round(i, -2), 500)
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @xfail  # (raises=AssertionError, reason="gh-15896")
    def test_round_py_consistency(self):
        # [FUNCTION DOC] Checks builtin round of a double scalar with negative digits agrees with
        # the Python float result (numpy issue 15896). Marked expected failure.
        f = 5.1 * 10**73
        assert_equal(round(np.float64(f), -73), round(f, -73))

    def test_searchsorted(self):
        # [FUNCTION DOC] Checks searchsorted of 0 in a sorted list gives insertion index 3.
        arr = [-8, -5, -1, 3, 6, 10]
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    def test_size(self):
        # [FUNCTION DOC] Checks the total size of a 2x3 list is 6 and sizes along axes 0 and 1 are
        # 2 and 3.
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.size(A) == 6)
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)

    def test_squeeze(self):
        # [FUNCTION DOC] Checks squeeze on nested lists and arrays removes all size-one dimensions
        # or only the requested axis, including negative axis, and gives the expected shapes.
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_std(self):
        # [FUNCTION DOC] Checks the standard deviation over all, axis 0 and axis 1 of a 2x3 list
        # against known values and that the standard deviation of an empty list is NaN.
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.std(A), 1.707825127659933)
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.std([])))

    #      assert_(w[0].category is RuntimeWarning)

    def test_swapaxes(self):
        # [FUNCTION DOC] Checks swapaxes of a 2x2x2 nested list exchanging axes 0 and 2.
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        # [FUNCTION DOC] Checks the sum over axis 1 with keepdims gives a column of row sums.
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)

        assert_equal(tgt, out)

    def test_take(self):
        # [FUNCTION DOC] Checks take with a list of indices from a list returns the elements at
        # those positions.
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]

        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        # [FUNCTION DOC] Checks the trace of a 3x2 nested list is 5 (sum of its main diagonal).
        c = [[1, 2], [3, 4], [5, 6]]
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        # [FUNCTION DOC] Checks transpose of a 3x2 nested list with axes order swapped gives a 2x3
        # result.
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        # [FUNCTION DOC] Checks the variance over all, axis 0 and axis 1 of a 2x3 list against
        # known values and that the variance of an empty list is NaN.
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.var(A), 2.9166666666666665)
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))

        #  with warnings.catch_warnings(record=True) as w:
        #      warnings.filterwarnings('always', '', RuntimeWarning)
        assert_(np.isnan(np.var([])))

    #      assert_(w[0].category is RuntimeWarning)


@xfail  # (reason="TODO")
class TestIsscalar(TestCase):
    # [CLASS DOC] Test case for the scalar-detection function. The whole class is marked expected
    # failure (TODO).
    def test_isscalar(self):
        # [FUNCTION DOC] Checks isscalar is true for Python float, int16 scalar, bool, string,
        # Fraction and Number instances, and false for a list and None.
        assert_(np.isscalar(3.1))
        assert_(np.isscalar(np.int16(12345)))
        assert_(np.isscalar(False))
        assert_(np.isscalar("numpy"))
        assert_(not np.isscalar([3.1]))
        assert_(not np.isscalar(None))

        # PEP 3141
        from fractions import Fraction

        assert_(np.isscalar(Fraction(5, 17)))
        from numbers import Number

        assert_(np.isscalar(Number()))


class TestBoolScalar(TestCase):
    # [CLASS DOC] Test case for the boolean scalars: logical and bitwise operators on true and
    # false scalars.
    def test_logical(self):
        # [FUNCTION DOC] Checks the Python and operator on boolean scalars returns the second
        # operand when the first is true and the first operand when it is false.
        f = np.False_
        t = np.True_
        s = "xyz"
        assert_((t and s) is s)
        assert_((f and s) is f)

    def test_bitwise_or_eq(self):
        # [FUNCTION DOC] Checks bitwise or of boolean scalars compares equal to the expected truth
        # table.
        f = np.False_
        t = np.True_
        assert_((t | t) == t)
        assert_((f | t) == t)
        assert_((t | f) == t)
        assert_((f | f) == f)

    def test_bitwise_or_is(self):
        # [FUNCTION DOC] Checks bitwise or of boolean scalars, converted to builtin bool, matches
        # the truth table by identity.
        f = np.False_
        t = np.True_
        assert_(bool(t | t) is bool(t))
        assert_(bool(f | t) is bool(t))
        assert_(bool(t | f) is bool(t))
        assert_(bool(f | f) is bool(f))

    def test_bitwise_and_eq(self):
        # [FUNCTION DOC] Checks bitwise and of boolean scalars compares equal to the expected
        # truth table.
        f = np.False_
        t = np.True_
        assert_((t & t) == t)
        assert_((f & t) == f)
        assert_((t & f) == f)
        assert_((f & f) == f)

    def test_bitwise_and_is(self):
        # [FUNCTION DOC] Checks bitwise and of boolean scalars, converted to builtin bool, matches
        # the truth table by identity.
        f = np.False_
        t = np.True_
        assert_(bool(t & t) is bool(t))
        assert_(bool(f & t) is bool(f))
        assert_(bool(t & f) is bool(f))
        assert_(bool(f & f) is bool(f))

    def test_bitwise_xor_eq(self):
        # [FUNCTION DOC] Checks bitwise xor of boolean scalars compares equal to the expected
        # truth table.
        f = np.False_
        t = np.True_
        assert_((t ^ t) == f)
        assert_((f ^ t) == t)
        assert_((t ^ f) == t)
        assert_((f ^ f) == f)

    def test_bitwise_xor_is(self):
        # [FUNCTION DOC] Checks bitwise xor of boolean scalars, converted to builtin bool, matches
        # the truth table by identity.
        f = np.False_
        t = np.True_
        assert_(bool(t ^ t) is bool(f))
        assert_(bool(f ^ t) is bool(t))
        assert_(bool(t ^ f) is bool(t))
        assert_(bool(f ^ f) is bool(f))


class TestBoolArray(TestCase):
    # [CLASS DOC] Test case for boolean arrays. State set up per test: t all true, f all false, o
    # a scratch output buffer, nm mostly false with two true entries, im mostly true with two
    # false entries; all are offset views of 40 elements to exercise unaligned code paths.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp, then builds the boolean arrays t, f, o,
        # nm and im from sliced longer arrays, setting entries 3 and -2 to true in nm and false in
        # im.
        super().setUp()
        # offset for simd tests
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False

    def test_all_any(self):
        # [FUNCTION DOC] Checks all and any on the prepared arrays, then on 256-long arrays with a
        # single flipped element at every position, and on a 100043-long array with the flipped
        # element at a range of indices including a negative one.
        assert_(self.t.all())
        assert_(self.t.any())
        assert_(not self.f.all())
        assert_(not self.f.any())
        assert_(self.nm.any())
        assert_(self.im.any())
        assert_(not self.nm.all())
        assert_(not self.im.all())
        # check bad element in all positions
        for i in range(256 - 7):
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            assert_(np.any(d))
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            assert_(not np.all(e))
            assert_array_equal(e, ~d)
        # big array test for blocked libc loops
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            assert_(np.any(d), msg=f"{i!r}")
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            assert_(not np.all(e), msg=f"{i!r}")

    def test_logical_not_abs(self):
        # [FUNCTION DOC] Checks logical not and absolute value on the prepared arrays, both
        # returned and written to the output buffer o.
        assert_array_equal(~self.t, self.f)
        assert_array_equal(np.abs(~self.t), self.f)
        assert_array_equal(np.abs(~self.f), self.t)
        assert_array_equal(np.abs(self.f), self.f)
        assert_array_equal(~np.abs(self.f), self.t)
        assert_array_equal(~np.abs(self.t), self.f)
        assert_array_equal(np.abs(~self.nm), self.im)
        np.logical_not(self.t, out=self.o)
        assert_array_equal(self.o, self.f)
        np.abs(self.t, out=self.o)
        assert_array_equal(self.o, self.t)

    def test_logical_and_or_xor(self):
        # [FUNCTION DOC] Checks the or, and and xor operators and the matching logical functions
        # (with output buffer o) on the prepared arrays, plus combinations with plain True and
        # False.
        assert_array_equal(self.t | self.t, self.t)
        assert_array_equal(self.f | self.f, self.f)
        assert_array_equal(self.t | self.f, self.t)
        assert_array_equal(self.f | self.t, self.t)
        np.logical_or(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t & self.t, self.t)
        assert_array_equal(self.f & self.f, self.f)
        assert_array_equal(self.t & self.f, self.f)
        assert_array_equal(self.f & self.t, self.f)
        np.logical_and(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t ^ self.t, self.f)
        assert_array_equal(self.f ^ self.f, self.f)
        assert_array_equal(self.t ^ self.f, self.t)
        assert_array_equal(self.f ^ self.t, self.t)
        np.logical_xor(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.f)

        assert_array_equal(self.nm & self.t, self.nm)
        assert_array_equal(self.im & self.f, False)
        assert_array_equal(self.nm & True, self.nm)
        assert_array_equal(self.im & False, self.f)
        assert_array_equal(self.nm | self.t, self.t)
        assert_array_equal(self.im | self.f, self.im)
        assert_array_equal(self.nm | True, self.t)
        assert_array_equal(self.im | False, self.im)
        assert_array_equal(self.nm ^ self.t, self.im)
        assert_array_equal(self.im ^ self.f, self.im)
        assert_array_equal(self.nm ^ True, self.im)
        assert_array_equal(self.im ^ False, self.im)


@xfailIfTorchDynamo
class TestBoolCmp(TestCase):
    # [CLASS DOC] Test case for comparison and classification functions on float32 and float64
    # arrays of bit-pattern values. The class is expected to fail under dynamo.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Builds float32 array f (256) and float64 array d (128) in blocks
        # where entries are powers of two or zero following the bit pattern of the block index,
        # with matching boolean masks ef and ed. Derives copies containing NaN (nf, nd), mixed
        # infinity and NaN (inff, infd, with masks efnonan, ednonan) and signed values including
        # negative zero, NaN and infinity (signf, signd).
        super().setUp()
        self.f = np.ones(256, dtype=np.float32)
        self.ef = np.ones(self.f.size, dtype=bool)
        self.d = np.ones(128, dtype=np.float64)
        self.ed = np.ones(self.d.size, dtype=bool)
        # generate values for all permutation of 256bit simd vectors
        s = 0
        for i in range(32):
            self.f[s : s + 8] = [i & 2**x for x in range(8)]
            self.ef[s : s + 8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        s = 0
        for i in range(16):
            self.d[s : s + 4] = [i & 2**x for x in range(4)]
            self.ed[s : s + 4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        self.nf = self.f.copy()
        self.nd = self.d.copy()
        self.nf[self.ef] = np.nan
        self.nd[self.ed] = np.nan

        self.inff = self.f.copy()
        self.infd = self.d.copy()
        self.inff[::3][self.ef[::3]] = np.inf
        self.infd[::3][self.ed[::3]] = np.inf
        self.inff[1::3][self.ef[1::3]] = -np.inf
        self.infd[1::3][self.ed[1::3]] = -np.inf
        self.inff[2::3][self.ef[2::3]] = np.nan
        self.infd[2::3][self.ed[2::3]] = np.nan
        self.efnonan = self.ef.copy()
        self.efnonan[2::3] = False
        self.ednonan = self.ed.copy()
        self.ednonan[2::3] = False

        self.signf = self.f.copy()
        self.signd = self.d.copy()
        self.signf[self.ef] *= -1.0
        self.signd[self.ed] *= -1.0
        self.signf[1::6][self.ef[1::6]] = -np.inf
        self.signd[1::6][self.ed[1::6]] = -np.inf
        self.signf[3::6][self.ef[3::6]] = -np.nan
        self.signd[3::6][self.ed[3::6]] = -np.nan
        self.signf[4::6][self.ef[4::6]] = -0.0
        self.signd[4::6][self.ed[4::6]] = -0.0

    def test_float(self):
        # [FUNCTION DOC] For four start offsets, checks comparisons, not-equal against zero, view-
        # as-int8 equals cast to int8, and isnan, isfinite, isinf and signbit on the float32
        # arrays against the expected masks.
        # offset for alignment test
        for i in range(4):
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            r = self.f[i:] != 0
            assert_array_equal(r, self.ef[i:])
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            r3 = 0 != self.f[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    def test_double(self):
        # [FUNCTION DOC] Same checks as the float32 test for two start offsets on the float64
        # arrays.
        # offset for alignment test
        for i in range(2):
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            r = self.d[i:] != 0
            assert_array_equal(r, self.ed[i:])
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            r3 = 0 != self.d[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestSeterr(TestCase):
    # [CLASS DOC] Test case for the floating point error state functions. Marked as expected pass
    # under dynamo (TODO).
    def test_default(self):
        # [FUNCTION DOC] Checks the default error state mapping: divide, invalid and over warn,
        # under ignored.
        err = np.geterr()
        assert_equal(
            err, dict(divide="warn", invalid="warn", over="warn", under="ignore")
        )

    def test_set(self):
        # [FUNCTION DOC] Sets the divide handling to print, checks the previous state is returned
        # and applies, sets over to raise, then restores the old state and checks it.
        err = np.seterr()
        old = np.seterr(divide="print")
        assert_(err == old)
        new = np.seterr()
        assert_(new["divide"] == "print")
        np.seterr(over="raise")
        assert_(np.geterr()["over"] == "raise")
        assert_(new["divide"] == "print")
        np.seterr(**old)
        assert_(np.geterr() == old)

    @xfail
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_divide_err(self):
        # [FUNCTION DOC] Checks that dividing by zero raises FloatingPointError, then after
        # ignoring divide errors that it does not. Marked expected failure; skipped on wasm and
        # armv5tel.
        with assert_raises(FloatingPointError):
            np.array([1.0]) / np.array([0.0])

        np.seterr(divide="ignore")
        np.array([1.0]) / np.array([0.0])

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_errobj(self):
        # [FUNCTION DOC] Installs a custom error object, checks a warning is emitted on divide by
        # zero, then installs a callback handler counting calls, and checks the callback is called
        # once for the operator and once more when passed through the extobj argument. Always
        # restores the old error object and deletes the counter attribute. Skipped on wasm.
        olderrobj = np.geterrobj()
        self.called = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.seterrobj([20000, 1, None])
                np.array([1.0]) / np.array([0.0])
                assert_equal(len(w), 1)

            def log_err(*args):
            # [FUNCTION DOC] HELPER: Error callback that increments the counter on the test case,
            # and asserts it receives exactly two arguments and that the first mentions divide.
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_("divide" in extobj_err[0])

            np.seterrobj([20000, 3, log_err])
            np.array([1.0]) / np.array([0.0])
            assert_equal(self.called, 1)

            np.seterrobj(olderrobj)
            np.divide(1.0, 0.0, extobj=[20000, 3, log_err])
            assert_equal(self.called, 2)
        finally:
            np.seterrobj(olderrobj)
            del self.called


@xfail  # (reason="TODO")
@instantiate_parametrized_tests
class TestFloatExceptions(TestCase):
    # [CLASS DOC] Test case for floating point exceptions raised by scalar operations. The class
    # is marked expected failure.
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        # [FUNCTION DOC] HELPER: Calls flop on x and y; if no exception is raised, fails with an
        # assertion. If FloatingPointError is raised, checks its message contains fpeerr, else
        # fails.
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False, f"Type {ftype} did not raise fpe error '{fpeerr}'.")
        except FloatingPointError as exc:
            assert_(
                str(exc).find(fpeerr) >= 0,
                f"Type {ftype} raised wrong fpe error '{exc}'.",
            )

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # [FUNCTION DOC] HELPER: Checks the operation raises the named floating point error for
        # the two scalars, and also when either or both are replaced by zero-dimensional array
        # views of themselves.
        # Check that fpe exception is raised.
        #
        # Given a floating operation `flop` and two scalar values, check that
        # the operation raises the floating point exception specified by
        # `fpeerr`. Tests all variants with 0-d array scalars as well.

        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    # Test for all real and complex float types
    @skipif(IS_WASM, reason="no wasm fp exception support")
    @parametrize("typecode", np.typecodes["AllFloat"])
    def test_floating_exceptions(self, typecode):
        # [FUNCTION DOC] Parametrized over all float and complex type codes: builds extreme values
        # (tiny, max, eps) for the type and checks each overflow, underflow, divide-by-zero and
        # invalid operation (division, multiplication, add, subtract, power) raises
        # FloatingPointError with the matching message. For complex types the underflow and
        # divide-by-zero messages are empty so any message matches. Skipped on wasm.
        # Test basic arithmetic function errors
        ftype = np.dtype(typecode).type
        if np.dtype(ftype).kind == "f":
            # Get some extreme values for the type
            fi = np.finfo(ftype)
            ft_tiny = fi._machar.tiny
            ft_max = fi.max
            ft_eps = fi.eps
            underflow = "underflow"
            divbyzero = "divide by zero"
        else:
            # 'c', complex, corresponding real dtype
            rtype = type(ftype(0).real)
            fi = np.finfo(rtype)
            ft_tiny = ftype(fi._machar.tiny)
            ft_max = ftype(fi.max)
            ft_eps = ftype(fi.eps)
            # The complex types raise different exceptions
            underflow = ""
            divbyzero = ""
        overflow = "overflow"
        invalid = "invalid"

        # The value of tiny for double double is NaN, so we need to
        # pass the assert
        if not np.isnan(ft_tiny):
            self.assert_raises_fpe(underflow, operator.truediv, ft_tiny, ft_max)
            self.assert_raises_fpe(underflow, operator.mul, ft_tiny, ft_tiny)
        self.assert_raises_fpe(overflow, operator.mul, ft_max, ftype(2))
        self.assert_raises_fpe(overflow, operator.truediv, ft_max, ftype(0.5))
        self.assert_raises_fpe(overflow, operator.add, ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, operator.sub, -ft_max, ft_max * ft_eps)
        self.assert_raises_fpe(overflow, np.power, ftype(2), ftype(2**fi.nexp))
        self.assert_raises_fpe(divbyzero, operator.truediv, ftype(1), ftype(0))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.truediv, ftype(0), ftype(0))
        self.assert_raises_fpe(invalid, operator.sub, ftype(np.inf), ftype(np.inf))
        self.assert_raises_fpe(invalid, operator.add, ftype(np.inf), ftype(-np.inf))
        self.assert_raises_fpe(invalid, operator.mul, ftype(0), ftype(np.inf))

    @skipif(IS_WASM, reason="no wasm fp exception support")
    def test_warnings(self):
        # [FUNCTION DOC] With warnings recorded and always shown, checks that divide by zero,
        # overflow, invalid subtraction of infinities and underflow each emit exactly one new
        # warning with the expected message text. Skipped on wasm.
        # test warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.divide(1, 0.0)
            assert_equal(len(w), 1)
            assert_("divide by zero" in str(w[0].message))
            np.array(1e300) * np.array(1e300)
            assert_equal(len(w), 2)
            assert_("overflow" in str(w[-1].message))
            np.array(np.inf) - np.array(np.inf)
            assert_equal(len(w), 3)
            assert_("invalid value" in str(w[-1].message))
            np.array(1e-300) * np.array(1e-300)
            assert_equal(len(w), 4)
            assert_("underflow" in str(w[-1].message))


class TestTypes(TestCase):
    # [CLASS DOC] Test case for type promotion rules between scalars and arrays, using shared
    # helper checks run through different promotion functions.
    def check_promotion_cases(self, promote_func):
        # [FUNCTION DOC] HELPER: Given a function that returns the promoted dtype of two operands,
        # checks promotion within a kind, between kinds, and between scalars and one-element
        # arrays (including scalars of higher kind than the array) against expected dtypes.
        # tests that the scalars get coerced correctly.
        b = np.bool_(0)
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)
        u8 = np.uint8(0)
        f32, f64 = np.float32(0), np.float64(0)
        c64, c128 = np.complex64(0), np.complex128(0)

        # coercion within the same kind
        assert_equal(promote_func(i8, i16), np.dtype(np.int16))
        assert_equal(promote_func(i32, i8), np.dtype(np.int32))
        assert_equal(promote_func(i16, i64), np.dtype(np.int64))
        assert_equal(promote_func(f32, f64), np.dtype(np.float64))
        assert_equal(promote_func(c128, c64), np.dtype(np.complex128))

        # coercion between kinds
        assert_equal(promote_func(b, i32), np.dtype(np.int32))
        assert_equal(promote_func(b, u8), np.dtype(np.uint8))
        assert_equal(promote_func(i8, u8), np.dtype(np.int16))
        assert_equal(promote_func(u8, i32), np.dtype(np.int32))

        assert_equal(promote_func(f32, i16), np.dtype(np.float32))
        assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
        assert_equal(promote_func(c128, f32), np.dtype(np.complex128))

        # coercion between scalars and 1-D arrays
        assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
        assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
        assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
        assert_equal(promote_func(c64, np.array([f64])), np.dtype(np.complex128))
        assert_equal(
            promote_func(np.complex64(3j), np.array([f64])), np.dtype(np.complex128)
        )

        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
        assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))

    def check_promotion_cases_2(self, promote_func):
        # [FUNCTION DOC] HELPER: Checks promotion cases that depend on the rule that scalars do
        # not upcast arrays (for example int32 with float32 gives float64, int8 array with int64
        # scalar stays int8, float and complex treated as one kind).
        # these are failing because of the "scalars do not upcast arrays" rule
        # Two first tests (i32 + f32 -> f64, and i64+f32 -> f64) xfail
        # until ufuncs implement the proper type promotion (ufunc loops?)
        i8, i32, i64 = np.int8(0), np.int32(0), np.int64(0)
        f32, f64 = np.float32(0), np.float64(0)
        c128 = np.complex128(0)

        assert_equal(promote_func(i32, f32), np.dtype(np.float64))
        assert_equal(promote_func(i64, f32), np.dtype(np.float64))

        assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
        assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))

        # float and complex are treated as the same "kind" for
        # the purposes of array-scalar promotion, so that you can do
        # (0j + float32array) to get a complex64 array instead of
        # a complex128 array.
        assert_equal(promote_func(np.array([f32]), c128), np.dtype(np.complex64))

    def test_coercion(self):
        # [FUNCTION DOC] Runs the first promotion checks with addition as the promotion function,
        # then checks that multiplying a bool or int8 array by Python, float64, float32, float16
        # and complex scalars gives the expected float or complex dtype without narrowing.
        def res_type(a, b):
            # [FUNCTION DOC] HELPER: Returns the dtype of the sum of the two operands.
            return np.add(a, b).dtype

        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype("f8"), f"array type {a.dtype}")
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype("f4"), f"array type {a.dtype}")
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype("f2"), f"array type {a.dtype}")

            b = 1.234j * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype("c16"), f"array type {a.dtype}")
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype("c8"), f"array type {a.dtype}")

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        # Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        #            a float32, shouldn't promote to float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # Probably ~t (bitwise negation) is more proper to use here,
        # but this is arguably less intuitive to understand at a glance, and
        # would fail if 't' is actually an integer array instead of boolean:
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    @xpassIfTorchDynamo_np  # (reason="'Scalars do not upcast arrays' rule")
    def test_coercion_2(self):
        # [FUNCTION DOC] Runs the second set of promotion checks with addition as the promotion
        # function. Expected pass-or-fail marker under dynamo because of the scalars-do-not-upcast
        # rule.
        def res_type(a, b):
            # [FUNCTION DOC] HELPER: Returns the dtype of the sum of the two operands.
            return np.add(a, b).dtype

        self.check_promotion_cases_2(res_type)

    def test_result_type(self):
        # [FUNCTION DOC] Runs the first promotion checks using the result_type function directly.
        self.check_promotion_cases(np.result_type)

    @skip(reason="array(None) not supported")
    def test_tesult_type_2(self):
        # [FUNCTION DOC] Checks result_type of None equals the dtype built from None. Skipped
        # (array of None unsupported).
        # KNOWN ISSUE: The method name misspells test as tesult.
        assert_(np.result_type(None) == np.dtype(None))

    @skip(reason="no endianness in dtypes")
    def test_promote_types_endian(self):
        # [FUNCTION DOC] Checks promote_types of two explicitly little- or big-endian int64 types
        # is the native int64 dtype. Skipped because dtypes have no endianness.
        # promote_types should always return native-endian types
        assert_equal(np.promote_types("<i8", "<i8"), np.dtype("i8"))
        assert_equal(np.promote_types(">i8", ">i8"), np.dtype("i8"))

    def test_can_cast(self):
        # [FUNCTION DOC] Checks can_cast for safe widening (int32 to int64, float to complex),
        # refusal of narrowing (complex to float, int64 to float32) and identical types under the
        # no-casting mode.
        assert_(np.can_cast(np.int32, np.int64))
        assert_(np.can_cast(np.float64, complex))
        assert_(not np.can_cast(complex, float))

        assert_(np.can_cast("i8", "f8"))
        assert_(not np.can_cast("i8", "f4"))

        assert_(np.can_cast("i8", "i8", "no"))

    @skip(reason="no endianness in dtypes")
    def test_can_cast_2(self):
        # [FUNCTION DOC] Checks can_cast under the equiv, safe, same_kind and unsafe modes with
        # explicit byte orders, TypeError for None, and keyword arguments. Skipped because dtypes
        # have no endianness.
        assert_(not np.can_cast("<i8", ">i8", "no"))

        assert_(np.can_cast("<i8", ">i8", "equiv"))
        assert_(not np.can_cast("<i4", ">i8", "equiv"))

        assert_(np.can_cast("<i4", ">i8", "safe"))
        assert_(not np.can_cast("<i8", ">i4", "safe"))

        assert_(np.can_cast("<i8", ">i4", "same_kind"))
        assert_(not np.can_cast("<i8", ">u4", "same_kind"))

        assert_(np.can_cast("<i8", ">u4", "unsafe"))

        assert_raises(TypeError, np.can_cast, "i4", None)
        assert_raises(TypeError, np.can_cast, None, "i4")

        # Also test keyword arguments
        assert_(np.can_cast(from_=np.int32, to=np.int64))

    @xpassIfTorchDynamo_np  # (reason="value-based casting?")
    def test_can_cast_values(self):
        # [FUNCTION DOC] For every integer dtype, checks can_cast accepts the minimum and maximum
        # representable values and rejects values just outside; for float dtypes checks min and
        # max are accepted. Expected pass-or-fail marker under dynamo.
        # gh-5917
        for dt in [np.int8, np.int16, np.int32, np.int64] + [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            ii = np.iinfo(dt)
            assert_(np.can_cast(ii.min, dt))
            assert_(np.can_cast(ii.max, dt))
            assert_(not np.can_cast(ii.min - 1, dt))
            assert_(not np.can_cast(ii.max + 1, dt))

        for dt in [np.float16, np.float32, np.float64, np.longdouble]:
            fi = np.finfo(dt)
            assert_(np.can_cast(fi.min, dt))
            assert_(np.can_cast(fi.max, dt))


# Custom exception class to test exception propagation in fromiter
class NIterError(Exception):
# [CLASS DOC] HELPER: Custom exception raised by the iterator helper to check that exceptions
# propagate through fromiter.
    pass


@skip(reason="NP_VER: fails on CI")
@xpassIfTorchDynamo_np  # (reason="TODO")
@instantiate_parametrized_tests
class TestFromiter(TestCase):
    # [CLASS DOC] Test case for fromiter. The helper makegen supplies a generator of 24 squares
    # and load_data a generator that raises at a chosen index. Skipped on CI and expected pass-or-
    # fail marker under dynamo.
    def makegen(self):
        # [FUNCTION DOC] HELPER: Returns a fresh generator yielding squares of the numbers 0 to
        # 23.
        return (x**2 for x in range(24))

    def test_types(self):
        # [FUNCTION DOC] Builds arrays from the generator with int32, int64 and float dtypes and
        # checks each result has the requested dtype.
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    def test_lengths(self):
        # [FUNCTION DOC] Checks fromiter with no count gives all 24 values, with count 20 gives 20
        # values, and a count larger than the available items raises ValueError.
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        assert_raises(ValueError, np.fromiter, self.makegen(), int, len(expected) + 10)

    def test_values(self):
        # [FUNCTION DOC] Checks the values read from the generator equal the list of squares, in
        # full and when limited to the first 20.
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(np.all(a == expected, axis=0))
        assert_(np.all(a20 == expected[:20], axis=0))

    def load_data(self, n, eindex):
        # [FUNCTION DOC] HELPER: Generator yielding the integers below n in order, but raising the
        # custom iteration error with a message naming the index when it reaches the chosen index.
        # Utility method for the issue 2592 tests.
        # Raise an exception at the desired index in the iterator.
        for e in range(n):
            if e == eindex:
                raise NIterError(f"error at index {eindex}")
            yield e

    @parametrize("dtype", [int])
    @parametrize("count, error_index", [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # [FUNCTION DOC] Parametrized over count and error index pairs: checks fromiter propagates
        # the custom error raised partway through a generator, even when the count was given.
        # Test iteration exceptions are correctly raised. The data/generator
        # has `count` elements but errors at `error_index`
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    @skip(reason="NP_VER: fails on CI")
    def test_empty_result(self):
        # [FUNCTION DOC] Checks that an iterable advertising a length hint of 10 but yielding
        # nothing produces an empty double array of shape zero; raises AssertionError otherwise.
        # Skipped on CI.
        class MyIter:
            # [CLASS DOC] HELPER: Iterable that reports a length hint of 10 while its iterator is
            # actually empty.
            def __length_hint__(self):
                # [FUNCTION DOC] HELPER: Returns the length hint 10.
                return 10

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over an empty list.
                return iter([])  # actual iterator is empty.

        res = np.fromiter(MyIter(), dtype="d")
        if res.shape != (0,):
            raise AssertionError(f"Expected res.shape == (0,), got {res.shape}")
        if res.dtype != "d":
            raise AssertionError(f"Expected res.dtype == 'd', got {res.dtype}")

    def test_too_few_items(self):
        # [FUNCTION DOC] Checks fromiter raises ValueError with a message about the iterator being
        # too short when the count is 10 but only 3 items exist.
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    def test_failed_itemsetting(self):
        # [FUNCTION DOC] Checks fromiter raises TypeError when an item cannot be stored as int,
        # and ValueError when items are tuples but the dtype is a sub-array dtype of length two.
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # The following manages to hit somewhat trickier code paths:
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))


@instantiate_parametrized_tests
class TestNonzeroAndCountNonzero(TestCase):
    # [CLASS DOC] Test case for nonzero and count_nonzero on lists and arrays of different ranks
    # and dtypes. Where this implementation differs from numpy, it returns 0-d arrays instead of
    # Python ints, which the differs tests record.
    def test_count_nonzero_list(self):
        # [FUNCTION DOC] Checks count_nonzero on a nested list gives 5 overall and the expected
        # per-column and per-row counts; raises AssertionError on a wrong total.
        lst = [[0, 1, 2, 3], [1, 0, 0, 6]]
        if np.count_nonzero(lst) != 5:
            raise AssertionError(
                f"Expected count_nonzero == 5, got {np.count_nonzero(lst)}"
            )
        assert_array_equal(np.count_nonzero(lst, axis=0), np.array([1, 1, 1, 2]))
        assert_array_equal(np.count_nonzero(lst, axis=1), np.array([3, 2]))

    def test_nonzero_trivial(self):
        # [FUNCTION DOC] Checks count_nonzero and nonzero for empty, single zero and single non-
        # zero arrays, with default and boolean dtypes.
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype="?")), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype="?")), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_trivial_differs(self):
        # [FUNCTION DOC] Documents a difference from numpy: count_nonzero of an empty list returns
        # an ndarray, not a Python int; raises AssertionError otherwise.
        # numpy returns a python int, we return a 0D array
        if not isinstance(np.count_nonzero([]), np.ndarray):
            raise AssertionError("Expected count_nonzero([]) to return np.ndarray")

    def test_nonzero_zerod(self):
        # [FUNCTION DOC] Checks count_nonzero of zero-dimensional arrays holding 0 or 1, in
        # default and boolean dtypes.
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype="?")), 0)

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype="?")), 1)

    def test_nonzero_zerod_differs(self):
        # [FUNCTION DOC] Documents a difference from numpy: count_nonzero of a zero-dimensional
        # array returns an ndarray.
        # numpy returns a python int, we return a 0D array
        if not isinstance(np.count_nonzero(np.array(1)), np.ndarray):
            raise AssertionError(
                "Expected count_nonzero(np.array(1)) to return np.ndarray"
            )

    def test_nonzero_onedim(self):
        # [FUNCTION DOC] Checks count_nonzero and nonzero indices for a one-dimensional array with
        # four non-zero entries.
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

    def test_nonzero_onedim_differs(self):
        # [FUNCTION DOC] Documents a difference from numpy: count_nonzero of a one-dimensional
        # array returns an ndarray.
        # numpy returns a python int, we return a 0D array
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        if not isinstance(np.count_nonzero(x), np.ndarray):
            raise AssertionError("Expected count_nonzero(x) to return np.ndarray")

    def test_nonzero_twodim(self):
        # [FUNCTION DOC] Checks count_nonzero for a 2x3 array cast to several integer dtypes and
        # nonzero row and column indices, then the same for a 3x3 identity matrix.
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype("i1")), 3)
        assert_equal(np.count_nonzero(x.astype("i2")), 3)
        assert_equal(np.count_nonzero(x.astype("i4")), 3)
        assert_equal(np.count_nonzero(x.astype("i8")), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # [FUNCTION DOC] Builds boolean arrays with sparse true runs at 20 different offsets and
        # checks nonzero returns the expected index arrays (exercises a special sparse path).
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_count_nonzero_axis(self):
        # [FUNCTION DOC] Checks count_nonzero along axis 0 and 1 of a 2x5 array returns an ndarray
        # with the expected counts, and that a repeated axis tuple raises ValueError, a string
        # axis raises TypeError, an out of range axis raises AxisError and an array axis raises
        # TypeError.
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_array_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_array_equal(np.count_nonzero(m, axis=1), expected)

        if not isinstance(np.count_nonzero(m, axis=1), np.ndarray):
            raise AssertionError(
                "Expected count_nonzero(m, axis=1) to return np.ndarray"
            )

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis="foo")
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero, m, axis=np.array([[1], [2]]))

    @parametrize("typecode", "efdFDBbhil?")
    def test_count_nonzero_axis_all_dtypes(self, typecode):
        # [FUNCTION DOC] Parametrized over many dtype codes: places non-zero values in two cells
        # of a 3x3 array and checks counts by axis 0, axis 1 (with intp dtype) and over both axes,
        # None and the default (value 2).
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis

        m = np.zeros((3, 3), dtype=typecode)
        n = np.ones(1, dtype=typecode)

        m[0, 0] = n[0]
        m[1, 0] = n[0]

        expected = np.array([2, 0, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=0)
        assert_array_equal(result, expected)
        if expected.dtype != result.dtype:
            raise AssertionError(f"Expected dtype {expected.dtype}, got {result.dtype}")

        expected = np.array([1, 1, 0], dtype=np.intp)
        result = np.count_nonzero(m, axis=1)
        assert_array_equal(result, expected)
        if expected.dtype != result.dtype:
            raise AssertionError(f"Expected dtype {expected.dtype}, got {result.dtype}")

        expected = np.array(2)
        assert_array_equal(np.count_nonzero(m, axis=(0, 1)), expected)
        assert_array_equal(np.count_nonzero(m, axis=None), expected)
        assert_array_equal(np.count_nonzero(m), expected)

    def test_countnonzero_axis_empty(self):
        # [FUNCTION DOC] Checks count_nonzero with an empty axis tuple returns the elementwise
        # boolean non-zero mask.
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        # [FUNCTION DOC] Checks count_nonzero with keepdims along axis 0, axis 1 and all axes
        # keeps the reduced dimensions and returns an ndarray.
        a = np.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        assert_array_equal(np.count_nonzero(a, axis=0, keepdims=True), [[1, 2, 3, 0]])
        assert_array_equal(np.count_nonzero(a, axis=1, keepdims=True), [[1], [2], [3]])
        assert_array_equal(np.count_nonzero(a, keepdims=True), [[6]])
        if not isinstance(np.count_nonzero(a, axis=1, keepdims=True), np.ndarray):
            raise AssertionError(
                "Expected count_nonzero(a, axis=1, keepdims=True) to return np.ndarray"
            )


class TestIndex(TestCase):
    # [CLASS DOC] Test case for boolean mask indexing.
    def test_boolean(self):
        # [FUNCTION DOC] Builds random 3x5x8 data and a random mask with some entries flipped, and
        # checks that indexing the stacked slice-wise results equals indexing the whole array by
        # the mask along the last two axes.
        a = rand(3, 5, 8)
        V = rand(5, 8)
        g1 = randint(0, 5, size=15)
        g2 = randint(0, 8, size=15)
        V[g1, g2] = -V[g1, g2]
        assert_(
            (np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all()
        )

    def test_boolean_edgecase(self):
        # [FUNCTION DOC] Checks that indexing an empty int32 array with an empty boolean mask
        # gives an empty result of dtype int32.
        a = np.array([], dtype="int32")
        b = np.array([], dtype="bool")
        c = a[b]
        assert_equal(c, [])
        assert_equal(c.dtype, np.dtype("int32"))


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBinaryRepr(TestCase):
    # [CLASS DOC] Test case for binary_repr. Marked expected pass-or-fail under dynamo (TODO).
    def test_zero(self):
        # [FUNCTION DOC] Checks binary_repr of zero is the string 0.
        assert_equal(np.binary_repr(0), "0")

    def test_positive(self):
        # [FUNCTION DOC] Checks binary_repr of three positive numbers against their binary
        # strings.
        assert_equal(np.binary_repr(10), "1010")
        assert_equal(np.binary_repr(12522), "11000011101010")
        assert_equal(np.binary_repr(10736848), "101000111101010011010000")

    def test_negative(self):
        # [FUNCTION DOC] Checks binary_repr of negative numbers gives a leading minus sign and the
        # binary digits of the magnitude.
        assert_equal(np.binary_repr(-1), "-1")
        assert_equal(np.binary_repr(-10), "-1010")
        assert_equal(np.binary_repr(-12522), "-11000011101010")
        assert_equal(np.binary_repr(-10736848), "-101000111101010011010000")

    def test_sufficient_width(self):
        # [FUNCTION DOC] Checks binary_repr with a width pads with leading zeros, and negative
        # numbers use the two's complement form within the width.
        assert_equal(np.binary_repr(0, width=5), "00000")
        assert_equal(np.binary_repr(10, width=7), "0001010")
        assert_equal(np.binary_repr(-5, width=7), "1111011")

    def test_neg_width_boundaries(self):
        # [FUNCTION DOC] Checks the negative boundary values for widths 1 to 10 give a one
        # followed by zeros (numpy issue 8670).
        # see gh-8670

        # Ensure that the example in the issue does not
        # break before proceeding to a more thorough test.
        assert_equal(np.binary_repr(-128, width=8), "10000000")

        for width in range(1, 11):
            num = -(2 ** (width - 1))
            exp = "1" + (width - 1) * "0"
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        # [FUNCTION DOC] Checks a large negative int64 scalar with width 64 gives two ones
        # followed by 62 zeros (numpy issue 14289).
        # See gh-14289.
        assert_equal(np.binary_repr(np.int64(-(2**62)), width=64), "11" + "0" * 62)


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBaseRepr(TestCase):
    # [CLASS DOC] Test case for base_repr. Marked expected pass-or-fail under dynamo (TODO).
    def test_base3(self):
        # [FUNCTION DOC] Checks base_repr of three to the fifth in base 3 is a one followed by
        # five zeros.
        assert_equal(np.base_repr(3**5, 3), "100000")

    def test_positive(self):
        # [FUNCTION DOC] Checks base_repr of positive numbers in several bases, with and without
        # zero padding.
        assert_equal(np.base_repr(12, 10), "12")
        assert_equal(np.base_repr(12, 10, 4), "000012")
        assert_equal(np.base_repr(12, 4), "30")
        assert_equal(np.base_repr(3731624803700888, 36), "10QR0ROFCEW")

    def test_negative(self):
        # [FUNCTION DOC] Checks base_repr of negative numbers gives a minus sign followed by the
        # digits.
        assert_equal(np.base_repr(-12, 10), "-12")
        assert_equal(np.base_repr(-12, 10, 4), "-000012")
        assert_equal(np.base_repr(-12, 4), "-30")

    def test_base_range(self):
        # [FUNCTION DOC] Checks base_repr raises ValueError for bases 1 and 37.
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        with assert_raises(ValueError):
            np.base_repr(1, 37)


class TestArrayComparisons(TestCase):
    # [CLASS DOC] Test case for array_equal and array_equiv, which return builtin bool values.
    def test_array_equal(self):
        # [FUNCTION DOC] Checks array_equal returns a builtin bool, true for equal arrays and
        # false for different shapes or values.
        res = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

    def test_array_equal_equal_nan(self):
        # [FUNCTION DOC] Checks the equal_nan option of array_equal: off by default, on makes NaNs
        # compare equal at the same positions, NaN differs from infinity, and it applies to zero-
        # dimensional, integer, multidimensional and complex arrays.
        # Test array_equal with equal_nan kwarg
        a1 = np.array([1, 2, np.nan])
        a2 = np.array([1, np.nan, 2])
        a3 = np.array([1, 2, np.inf])

        # equal_nan=False by default
        assert_(not np.array_equal(a1, a1))
        assert_(np.array_equal(a1, a1, equal_nan=True))
        assert_(not np.array_equal(a1, a2, equal_nan=True))
        # nan's not conflated with inf's
        assert_(not np.array_equal(a1, a3, equal_nan=True))
        # 0-D arrays
        a = np.array(np.nan)
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Non-float dtype - equal_nan should have no effect
        a = np.array([1, 2, 3], dtype=int)
        assert_(np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Multi-dimensional array
        a = np.array([[0, 1], [np.nan, 1]])
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Complex values
        a, b = [np.array([1 + 1j])] * 2
        a.real, b.imag = np.nan, np.nan
        assert_(not np.array_equal(a, b, equal_nan=False))
        assert_(np.array_equal(a, b, equal_nan=True))

    def test_none_compares_elementwise(self):
        # [FUNCTION DOC] Checks that the equality and inequality methods against None compare
        # elementwise, giving all false and all true arrays.
        a = np.ones(3)
        assert_equal(a.__eq__(None), [False, False, False])
        assert_equal(a.__ne__(None), [True, True, True])

    def test_array_equiv(self):
        # [FUNCTION DOC] Checks array_equiv returns a builtin bool and is true when one array can
        # be broadcast to the other with equal values, false for different values or incompatible
        # shapes.
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(
            np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        assert_(not res)
        assert_(type(res) is bool)


@instantiate_parametrized_tests
class TestClip(TestCase):
    # [CLASS DOC] Test case for clip. State: nr and nc, the number of rows and columns of
    # generated test data. Helper methods generate random data of several dtypes and provide fast
    # and slow reference clip implementations.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp and sets the row count to 5 and column
        # count to 3.
        super().setUp()
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None, casting=None):
        # [FUNCTION DOC] HELPER: Calls the clip method of array a with bounds m and M, forwarding
        # the output array and casting mode only when given; returns the result.
        if out is None:
            if casting is None:
                return a.clip(m, M)
            else:
                return a.clip(m, M, casting=casting)
        else:
            if casting is None:
                return a.clip(m, M, out)
            else:
                return a.clip(m, M, out, casting=casting)

    def clip(self, a, m, M, out=None):
        # [FUNCTION DOC] HELPER: Slow reference clip: builds a selector with 0 where in range, 1
        # where below the lower bound and 2 where above the upper bound, then uses choose among
        # the array and the two bounds, optionally writing into out.
        # use slow-clip
        selector = np.less(a, m) + 2 * np.greater(a, M)
        return selector.choose((a, m, M), out=out)

    # Handy functions
    def _generate_data(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random normal data of shape n by m.
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random complex data of shape n by m (normal real part,
        # uniform imaginary part).
        return randn(n, m) + 1.0j * rand(n, m)

    def _generate_flt_data(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random normal data of shape n by m cast to float32.
        return (randn(n, m)).astype(np.float32)

    def _neg_byteorder(self, a):
        # [FUNCTION DOC] HELPER: Converts the input to an array and casts it to the opposite byte
        # order of the machine.
        a = np.asarray(a)
        if sys.byteorder == "little":
            a = a.astype(a.dtype.newbyteorder(">"))
        else:
            a = a.astype(a.dtype.newbyteorder("<"))
        return a

    def _generate_non_native_data(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random normal data in non-native byte order, asserting
        # its dtype is not native.
        data = randn(n, m)
        data = self._neg_byteorder(data)
        assert_(not data.dtype.isnative)
        return data

    def _generate_int_data(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random values scaled to 0 through 10 as int64 with shape
        # n by m.
        return (10 * rand(n, m)).astype(np.int64)

    def _generate_int32_data(self, n, m):
        # [FUNCTION DOC] HELPER: Returns random values scaled to 0 through 10 as int32 with shape
        # n by m.
        return (10 * rand(n, m)).astype(np.int32)

    # Now the real test cases

    @parametrize("dtype", "?bhilBfd")
    def test_ones_pathological(self, dtype):
        # [FUNCTION DOC] Parametrized over bool, int and float dtype codes: clipping an array of
        # ones with lower bound 1 and upper bound 0 (inverted) must give all zeros, i.e. the upper
        # bound wins.
        # for preservation of behavior described in
        # gh-12519; amin > amax behavior may still change
        # in the future
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        assert_equal(actual, expected)

    @parametrize("dtype", "eFD")
    def test_ones_pathological_2(self, dtype):
        # [FUNCTION DOC] Parametrized over half and complex dtype codes: same inverted-bounds clip
        # of ones as the previous test. Complex dtypes raise SkipTest because clamp is unsupported
        # for them.
        if dtype in "FD":
            # FIXME: make xfail
            raise SkipTest("torch.clamp not implemented for complex types")
        # for preservation of behavior described in
        # gh-12519; amin > amax behavior may still change
        # in the future
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        assert_equal(actual, expected)

    def test_simple_double(self):
        # [FUNCTION DOC] Compares the clip method with scalar bounds against the slow reference on
        # random double data.
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.1
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_int(self):
        # [FUNCTION DOC] Compares the clip method with scalar integer bounds against the slow
        # reference on random int data.
        # Test native int input with scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(int)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_array_double(self):
        # [FUNCTION DOC] Compares the clip method with array bounds (zeros and 0.5) against the
        # slow reference on random double data.
        # Test native double input with array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = np.zeros(a.shape)
        M = m + 0.5
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="byteorder not supported in torch")
    def test_simple_nonnative(self):
        # [FUNCTION DOC] Compares clip against the reference for non-native byte order data, and
        # for native data with a non-native upper bound. Expected pass-or-fail marker under dynamo
        # because byte order is unsupported.
        # Test non native double input with scalar min/max.
        # Test native double input with non native double scalar min/max.
        a = self._generate_non_native_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        # Test native double input with non native double scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = self._neg_byteorder(0.6)
        assert_(not M.dtype.isnative)
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="clamp not supported for complex")
    def test_simple_complex(self):
        # [FUNCTION DOC] Compares clip against the reference for complex data with real scalar
        # bounds, and for real data with complex bounds. Expected pass-or-fail marker under dynamo
        # because clamp does not support complex.
        # Test native complex input with native double scalar min/max.
        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data(self.nr, self.nc)
        m = -0.5 + 1.0j
        M = 1.0 + 2.0j
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    @xfail  # (reason="clamp not supported for complex")
    def test_clip_complex(self):
        # [FUNCTION DOC] Clips a complex array with only a lower bound and only an upper bound
        # (the other None) and checks the data is unchanged (numpy issue 5354). Marked expected
        # failure.
        # Address Issue gh-5354 for clipping complex arrays
        # Test native complex input without explicit min/max
        # ie, either min=None or max=None
        a = np.ones(10, dtype=complex)
        m = a.min()
        M = a.max()
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        assert_array_equal(am, a)
        assert_array_equal(aM, a)

    def test_clip_non_contig(self):
        # [FUNCTION DOC] Compares clip against the reference on a strided view that is neither C
        # nor Fortran contiguous.
        # Test clip for non contiguous native input and native scalar min/max.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags["F_CONTIGUOUS"])
        assert_(not a.flags["C_CONTIGUOUS"])
        ac = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_equal(ac, act)

    def test_simple_out(self):
        # [FUNCTION DOC] Compares clip into a preallocated output array against the slow reference
        # writing into a second output.
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    #   @xpassIfTorchDynamo_np  # (reason="casting not supported")
    @parametrize(
        "casting",
        [
            subtest(None, decorators=[xfail]),
            subtest("unsafe", decorators=[xpassIfTorchDynamo_np]),
        ],
    )
    def test_simple_int32_inout(self, casting):
        # [FUNCTION DOC] Parametrized over casting None (expected failure) and unsafe: clips int32
        # data with double scalar bounds into an int32 output. The casting argument is passed only
        # when not None; with None the output stays zeros while the reference output differs.
        # Test native int32 input with double min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        if casting is not None:
            # explicitly passing "unsafe" will silence warning
            self.fastclip(a, m, M, ac, casting=casting)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_simple_int64_out(self):
        # [FUNCTION DOC] Clips int32 data with int32 scalar bounds into an int64 output and
        # compares with the reference.
        # Test native int32 input with int32 scalar min/max and int64 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int64_inout(self):
        # [FUNCTION DOC] Clips int32 data with a float64 array lower bound and scalar upper bound
        # into an int32 output. Only the slow reference clip is exercised; the clip under test is
        # never called, so the output array stays zeros and the comparison fails, which is why the
        # test is expected to fail.
        # Test native int32 input with double array min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_simple_int32_out(self):
        # [FUNCTION DOC] Clips double data with scalar bounds into an int32 output. Only the slow
        # reference clip is exercised; the clip under test is never called, so the output array
        # stays zeros and the comparison fails, which is why the test is expected to fail.
        # Test native double input with scalar min/max and int out.
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_simple_inplace_01(self):
        # [FUNCTION DOC] Clips in place (the input is also the output) with an array lower bound
        # and scalar upper bound and compares with the reference applied to a copy.
        # Test native double input with array min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_simple_inplace_02(self):
        # [FUNCTION DOC] Clips in place with scalar bounds and compares with the reference applied
        # in place to a copy.
        # Test native double input with scalar min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_noncontig_inplace(self):
        # [FUNCTION DOC] Clips a non-contiguous strided view in place with scalar bounds and
        # compares with the reference applied to a copy.
        # Test non contiguous double input with double scalar min/max in-place.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags["F_CONTIGUOUS"])
        assert_(not a.flags["C_CONTIGUOUS"])
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        # [FUNCTION DOC] Compares clip with double scalar bounds against the reference on double
        # data.
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_02(self):
        # [FUNCTION DOC] Compares clip with Python int bounds against the reference on int32 data.
        # Test native int32 input with int32 scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(np.int32)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_03(self):
        # [FUNCTION DOC] Compares clip with float64 scalar bounds against the reference on int32
        # data.
        # Test native int32 input with float64 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = -2
        M = 4
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        act = self.clip(a, np.float64(m), np.float64(M))
        assert_array_equal(ac, act)

    def test_type_cast_04(self):
        # [FUNCTION DOC] Compares clip with float32 scalar bounds against the reference on int32
        # data.
        # Test native int32 input with float32 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float32(-2)
        M = np.float32(4)
        act = self.fastclip(a, m, M)
        ac = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_type_cast_05(self):
        # [FUNCTION DOC] Compares clip with double array lower bound and scalar upper bound
        # against the reference on int64 data.
        # Test native int32 with double arrays min/max.
        a = self._generate_int_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        act = self.clip(a, m * np.zeros(a.shape), M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_06(self):
        # [FUNCTION DOC] Compares clip against the reference using a non-native byte order scalar
        # lower bound on double data. Expected pass-or-fail marker under dynamo because byte order
        # is unsupported.
        # Test native with NON native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.5
        m_s = self._neg_byteorder(m)
        M = 1.0
        act = self.clip(a, m_s, M)
        ac = self.fastclip(a, m_s, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_07(self):
        # [FUNCTION DOC] Compares clip against the reference for non-native byte order data with a
        # native array lower bound; the reference here is the array method itself. Expected pass-
        # or-fail marker under dynamo.
        # Test NON native with native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        act = a_s.clip(m, M)
        ac = self.fastclip(a_s, m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_08(self):
        # [FUNCTION DOC] Compares clip with native scalar bounds on non-native byte order data
        # against the array method. Expected pass-or-fail marker under dynamo.
        # Test NON native with native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.0
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        ac = self.fastclip(a_s, m, M)
        act = a_s.clip(m, M)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_09(self):
        # [FUNCTION DOC] Compares clip of native data with a non-native byte order array lower
        # bound against the reference. Expected pass-or-fail marker under dynamo.
        # Test native with NON native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.0
        m_s = self._neg_byteorder(m)
        assert_(not m_s.dtype.isnative)
        ac = self.fastclip(a, m_s, M)
        act = self.clip(a, m_s, M)
        assert_array_equal(ac, act)

    def test_type_cast_10(self):
        # [FUNCTION DOC] Compares clip of int64 data with float32 scalar bounds into a float32
        # output against the reference.
        # Test native int32 with float min/max and float out for output argument.
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.float32(-0.5)
        M = np.float32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_equal(ac, act)

    @xpassIfTorchDynamo_np  # (reason="newbyteorder not supported")
    def test_type_cast_11(self):
        # [FUNCTION DOC] Compares clip of non-native byte order data into a non-native output
        # against the reference. Expected pass-or-fail marker under dynamo.
        # Test non native with native scalar, min/max, out non native
        a = self._generate_non_native_data(self.nr, self.nc)
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder(">"))
        bt = b.copy()
        m = -0.5
        M = 1.0
        self.fastclip(a, m, M, out=b)
        self.clip(a, m, M, out=bt)
        assert_array_equal(b, bt)

    def test_type_cast_12(self):
        # [FUNCTION DOC] Compares clip of int64 data with int32 scalar bounds into a float32
        # output against the reference.
        # Test native int32 input and min/max and float out
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.int32(0)
        M = np.int32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_equal(ac, act)

    def test_clip_with_out_simple(self):
        # [FUNCTION DOC] Compares clip of double data with scalar bounds into a preallocated
        # double output against the reference.
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_simple2(self):
        # [FUNCTION DOC] Clips int32 data with float64 scalar bounds into an int32 output. Only
        # the slow reference clip is exercised; the clip under test is never called, so the output
        # array stays zeros and the comparison fails, which is why the test is expected to fail.
        # Test native int32 input with double min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        # [FUNCTION DOC] Clips int32 data with int32 scalar bounds into an int64 output and
        # compares with the reference.
        # Test native int32 input with int32 scalar min/max and int64 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_int32(self):
        # [FUNCTION DOC] Clips int32 data with a float64 array lower bound into an int32 output.
        # Only the slow reference clip is exercised; the clip under test is never called, so the
        # output array stays zeros and the comparison fails, which is why the test is expected to
        # fail.
        # Test native int32 input with double array min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    @xfail  # (reason="FIXME arrays not equal")
    def test_clip_with_out_array_outint32(self):
        # [FUNCTION DOC] Clips double data with scalar bounds into an int32 output. Only the slow
        # reference clip is exercised; the clip under test is never called, so the output array
        # stays zeros and the comparison fails, which is why the test is expected to fail.
        # Test native double input with scalar min/max and int out
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        self.clip(a, m, M, act)
        assert_array_equal(ac, act)

    def test_clip_with_out_transposed(self):
        # [FUNCTION DOC] Checks the clip method writes correctly into a transposed output view and
        # matches the reference.
        # Test that the out argument works when transposed
        a = np.arange(16).reshape(4, 4)
        out = np.empty_like(a).T
        a.clip(4, 10, out=out)
        expected = self.clip(a, 4, 10)
        assert_array_equal(out, expected)

    def test_clip_with_out_memory_overlap(self):
        # [FUNCTION DOC] Checks the clip method works when the output is an overlapping slice of
        # the input, comparing with the reference on a copy.
        # Test that the out argument works when it has memory overlap
        a = np.arange(16).reshape(4, 4)
        ac = a.copy()
        a[:-1].clip(4, 10, out=a[1:])
        expected = self.clip(ac[:-1], 4, 10)
        assert_array_equal(a[1:], expected)

    def test_clip_inplace_array(self):
        # [FUNCTION DOC] Clips in place with an array lower bound and scalar upper bound and
        # compares with the reference on a copy.
        # Test native double input with array min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_clip_inplace_simple(self):
        # [FUNCTION DOC] Clips in place with scalar bounds and compares with the reference on a
        # copy.
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_equal(a, ac)

    def test_clip_func_takes_out(self):
        # [FUNCTION DOC] Checks the top-level clip function accepts an output argument, returns
        # that same object and gives the reference result.
        # Ensure that the clip() function takes an out=argument.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = np.clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_equal(a2, ac)
        assert_(a2 is a)

    @skip(reason="Edge case; Wait until deprecation graduates")
    def test_clip_nan(self):
        # [FUNCTION DOC] Intended to check that clipping with NaN bounds warns with
        # DeprecationWarning and leaves the data unchanged. Marked skipped until the deprecation
        # graduates.
        # KNOWN ISSUE: The bare skip decorator is applied without a reason, so the method is
        # replaced by a decorator function and the body does not run as written.
        d = np.arange(7.0)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=-2, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=10), d)

    @parametrize(
        "amin, amax",
        [
            # two scalars
            (1, 0),
            # mix scalar and array
            (1, np.zeros(10)),
            # two arrays
            (np.ones(10), np.zeros(10)),
        ],
    )
    def test_clip_value_min_max_flip(self, amin, amax):
        # [FUNCTION DOC] Parametrized over scalar and array bounds with lower above upper: checks
        # clip equals the minimum of the maximum with the lower bound and the upper bound, so the
        # upper bound wins.
        a = np.arange(10, dtype=np.int64)
        # requirement from ufunc_docstrings.py
        expected = np.minimum(np.maximum(a, amin), amax)
        actual = np.clip(a, amin, amax)
        assert_equal(actual, expected)

    @parametrize(
        "arr, amin, amax",
        [
            # problematic scalar nan case from hypothesis
            (
                np.zeros(10, dtype=np.int64),
                np.array(np.nan),
                np.zeros(10, dtype=np.int32),
            ),
        ],
    )
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        # [FUNCTION DOC] Parametrized with an int64 array, a NaN zero-dimensional lower bound and
        # an int32 upper bound: checks clip equals the minimum and maximum composition, so NaN
        # propagation matches the comparison functions.
        # enforcement of scalar nan propagation for comparisons
        # called through clip()
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @skip  # hypothesis hynp.from_dtype fails on CI (versions?)
    @given(
        data=st.data(),
        arr=hynp.arrays(
            dtype=hynp.integer_dtypes() | hynp.floating_dtypes(),
            shape=hynp.array_shapes(),
        ),
    )
    def test_clip_property(self, data, arr):
        # [FUNCTION DOC] Property-based test using Hypothesis: draws a random integer or float
        # array and bounds with mutually broadcastable shapes, then checks the clip result has the
        # common result type and equals the minimum and maximum composition in that type. Skipped
        # (Hypothesis dtype strategy fails on CI).
        # KNOWN ISSUE: The bare skip decorator is applied without a reason, so the method is
        # replaced by a decorator function and the body does not run as written.
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        # Generate shapes for the bounds which can be broadcast with each other
        # and with the base shape.  Below, we might decide to use scalar bounds,
        # but it's clearer to generate these shapes unconditionally in advance.
        in_shapes, result_shape = data.draw(
            hynp.mutually_broadcastable_shapes(num_shapes=2, base_shape=arr.shape)
        )
        # Scalar `nan` is deprecated due to the differing behaviour it shows.
        s = numeric_dtypes.flatmap(lambda x: hynp.from_dtype(x, allow_nan=False))
        amin = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[0], elements={"allow_nan": False}
            )
        )
        amax = data.draw(
            s
            | hynp.arrays(
                dtype=numeric_dtypes, shape=in_shapes[1], elements={"allow_nan": False}
            )
        )

        # Then calculate our result and expected result and check that they're
        # equal!  See gh-12519 and gh-19457 for discussion deciding on this
        # property and the result_type argument.
        result = np.clip(arr, amin, amax)
        t = np.result_type(arr, amin, amax)
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        if result.dtype != t:
            raise AssertionError(f"Expected result.dtype == {t}, got {result.dtype}")
        assert_array_equal(result, expected)


class TestAllclose(TestCase):
    # [CLASS DOC] Test case for allclose. Class attributes rtol and atol give the default relative
    # and absolute tolerances used to build near and far test pairs.
    rtol = 1e-5
    atol = 1e-8

    def tst_allclose(self, x, y):
        # [FUNCTION DOC] HELPER: Asserts allclose is true for the two inputs, with a message
        # showing them.
        assert_(np.allclose(x, y), f"{x} and {y} not close")

    def tst_not_allclose(self, x, y):
        # [FUNCTION DOC] HELPER: Asserts allclose is false for the two inputs, with a message
        # showing them.
        assert_(not np.allclose(x, y), f"{x} and {y} shouldn't be close")

    def test_ip_allclose(self):
        # [FUNCTION DOC] Builds pairs within tolerance (scalars, lists, arrays, infinity) and
        # checks allclose is true for each via the helper.
        # Parametric test factory.
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [
            ([1, 0], [1, 0]),
            ([atol], [0]),
            ([1], [1 + rtol + atol]),
            (arr, arr + arr * rtol),
            (arr, arr + arr * rtol + atol * 2),
            (aran, aran + aran * rtol),
            (np.inf, np.inf),
            (np.inf, [np.inf]),
        ]

        for x, y in data:
            self.tst_allclose(x, y)

    def test_ip_not_allclose(self):
        # [FUNCTION DOC] Builds pairs outside tolerance (mismatched or opposite infinities, NaN,
        # values beyond tolerance) and checks allclose is false for each via the helper.
        # Parametric test factory.
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [
            ([np.inf, 0], [1, np.inf]),
            ([np.inf, 0], [1, 0]),
            ([np.inf, np.inf], [1, np.inf]),
            ([np.inf, np.inf], [1, 0]),
            ([-np.inf, 0], [np.inf, 0]),
            ([np.nan, 0], [np.nan, 0]),
            ([atol * 2], [0]),
            ([1], [1 + rtol + atol * 2]),
            (aran, aran + aran * atol + atol * 2),
            (np.array([np.inf, 1]), np.array([0, np.inf])),
        ]

        for x, y in data:
            self.tst_not_allclose(x, y)

    def test_no_parameter_modification(self):
        # [FUNCTION DOC] Checks allclose does not modify its input arrays.
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.allclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_min_int(self):
        # [FUNCTION DOC] Checks allclose of the minimum integer value with itself is true,
        # guarding against absolute value overflow.
        # Could make problems because of abs(min_int) == min_int
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)
        assert_(np.allclose(a, a))

    def test_equalnan(self):
        # [FUNCTION DOC] Checks allclose with equal_nan enabled treats NaNs at the same position
        # as equal.
        x = np.array([1.0, np.nan])
        assert_(np.allclose(x, x, equal_nan=True))


class TestIsclose(TestCase):
    # [CLASS DOC] Test case for isclose. Class attributes rtol and atol are the tolerances; the
    # _setup helper fills lists of input pairs that are all close, none close, or partly close
    # (with expected results for the partly close ones).
    rtol = 1e-5
    atol = 1e-8

    def _setup(self):
        # [FUNCTION DOC] HELPER: Builds and stores on the test case three lists of input pairs
        # (all_close_tests, none_close_tests, some_close_tests) and the expected boolean results
        # for the third list. Called explicitly by tests rather than by the framework.
        atol = self.atol
        rtol = self.rtol
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        self.all_close_tests = [
            ([1, 0], [1, 0]),
            ([atol], [0]),
            ([1], [1 + rtol + atol]),
            (arr, arr + arr * rtol),
            (arr, arr + arr * rtol + atol),
            (aran, aran + aran * rtol),
            (np.inf, np.inf),
            (np.inf, [np.inf]),
            ([np.inf, -np.inf], [np.inf, -np.inf]),
        ]
        self.none_close_tests = [
            ([np.inf, 0], [1, np.inf]),
            ([np.inf, -np.inf], [1, 0]),
            ([np.inf, np.inf], [1, -np.inf]),
            ([np.inf, np.inf], [1, 0]),
            ([np.nan, 0], [np.nan, -np.inf]),
            ([atol * 2], [0]),
            ([1], [1 + rtol + atol * 2]),
            (aran, aran + rtol * 1.1 * aran + atol * 1.1),
            (np.array([np.inf, 1]), np.array([0, np.inf])),
        ]
        self.some_close_tests = [
            ([np.inf, 0], [np.inf, atol * 2]),
            ([atol, 1, 1e6 * (1 + 2 * rtol) + atol], [0, np.nan, 1e6]),
            (np.arange(3), [0, 1, 2.1]),
            (np.nan, [np.nan, np.nan, np.nan]),
            ([0], [atol, np.inf, -np.inf, np.nan]),
            (0, [atol, np.inf, -np.inf, np.nan]),
        ]
        self.some_close_results = [
            [True, False],
            [True, False, False],
            [True, True, False],
            [False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]

    def test_ip_isclose(self):
        # [FUNCTION DOC] Checks isclose gives the expected elementwise results for the partly
        # close pairs.
        self._setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            assert_array_equal(np.isclose(x, y), result)

    def tst_all_isclose(self, x, y):
        # [FUNCTION DOC] HELPER: Asserts every element of isclose of the inputs is true.
        assert_(np.all(np.isclose(x, y)), f"{x} and {y} not close")

    def tst_none_isclose(self, x, y):
        # [FUNCTION DOC] HELPER: Asserts no element of isclose of the inputs is true.
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    def tst_isclose_allclose(self, x, y):
        # [FUNCTION DOC] HELPER: Checks isclose and allclose agree: for two scalars compares the
        # values directly, otherwise compares the all-reduction of isclose with allclose.
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))

    def test_ip_all_isclose(self):
        # [FUNCTION DOC] Runs the all-close helper on every pair in the all-close list.
        self._setup()
        for x, y in self.all_close_tests:
            self.tst_all_isclose(x, y)

    def test_ip_none_isclose(self):
        # [FUNCTION DOC] Runs the none-close helper on every pair in the none-close list.
        self._setup()
        for x, y in self.none_close_tests:
            self.tst_none_isclose(x, y)

    def test_ip_isclose_allclose(self):
        # [FUNCTION DOC] Runs the isclose versus allclose consistency helper on all three lists of
        # pairs.
        self._setup()
        tests = self.all_close_tests + self.none_close_tests + self.some_close_tests
        for x, y in tests:
            self.tst_isclose_allclose(x, y)

    def test_equal_nan(self):
        # [FUNCTION DOC] Checks isclose with equal_nan enabled treats NaN as equal to NaN for a
        # scalar and for arrays.
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        arr = np.array([1.0, np.nan])
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])

    @xfailIfTorchDynamo  # scalars vs 0D
    def test_scalar_return(self):
        # [FUNCTION DOC] Checks the result of isclose on two ints counts as a scalar. Marked
        # expected failure under dynamo because scalars and zero-dimensional arrays differ.
        assert_(np.isscalar(np.isclose(1, 1)))

    def test_no_parameter_modification(self):
        # [FUNCTION DOC] Checks isclose does not modify its input arrays.
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.isclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_non_finite_scalar(self):
        # [FUNCTION DOC] Checks isclose of infinity with negative infinity and of zero with
        # infinity are false when converted with the item method (modified from numpy because
        # results here are array scalars).
        # GH7014, when two scalars are compared the output should also be a
        # scalar
        # XXX: test modified since there are array scalars
        assert_(np.isclose(np.inf, -np.inf).item() is False)
        assert_(np.isclose(0, np.inf).item() is False)


class TestStdVar(TestCase):
    # [CLASS DOC] Test case for std and var on real data. State set in setUp: A is an array
    # alternating 1 and -1, and real_var is its known variance, 1.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp, then stores the sample array and its
        # known variance.
        super().setUp()
        self.A = np.array([1, -1, 1, -1])
        self.real_var = 1

    def test_basic(self):
        # [FUNCTION DOC] Checks variance equals the known value and squared standard deviation
        # equals it too.
        assert_almost_equal(np.var(self.A), self.real_var)
        assert_almost_equal(np.std(self.A) ** 2, self.real_var)

    def test_scalars(self):
        # [FUNCTION DOC] Checks variance and standard deviation of a Python scalar are zero.
        assert_equal(np.var(1), 0)
        assert_equal(np.std(1), 0)

    def test_ddof1(self):
        # [FUNCTION DOC] Checks variance and squared standard deviation with one delta degree of
        # freedom are scaled by length over length minus one.
        assert_almost_equal(
            np.var(self.A, ddof=1), self.real_var * len(self.A) / (len(self.A) - 1)
        )
        assert_almost_equal(
            np.std(self.A, ddof=1) ** 2, self.real_var * len(self.A) / (len(self.A) - 1)
        )

    def test_ddof2(self):
        # [FUNCTION DOC] Checks variance and squared standard deviation with two delta degrees of
        # freedom are scaled by length over length minus two.
        assert_almost_equal(
            np.var(self.A, ddof=2), self.real_var * len(self.A) / (len(self.A) - 2)
        )
        assert_almost_equal(
            np.std(self.A, ddof=2) ** 2, self.real_var * len(self.A) / (len(self.A) - 2)
        )

    def test_out_scalar(self):
        # [FUNCTION DOC] Checks std, var and mean write into the given zero-dimensional output
        # array and return that same object.
        d = np.arange(10)
        out = np.array(0.0)
        r = np.std(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.var(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.mean(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)


class TestStdVarComplex(TestCase):
    # [CLASS DOC] Test case for std and var on complex data.
    def test_basic(self):
        # [FUNCTION DOC] Checks the variance and squared standard deviation of four unit complex
        # values are 1.
        A = np.array([1, 1.0j, -1, -1.0j])
        real_var = 1
        assert_almost_equal(np.var(A), real_var)
        assert_almost_equal(np.std(A) ** 2, real_var)

    def test_scalars(self):
        # [FUNCTION DOC] Checks the variance and standard deviation of a complex Python scalar are
        # zero.
        assert_equal(np.var(1j), 0)
        assert_equal(np.std(1j), 0)


class TestCreationFuncs(TestCase):
    # [CLASS DOC] Test case for the array creation functions ones, zeros, empty and full. State
    # set in setUp: dtypes, the set of dtypes checked, orders, a mapping of memory order to the
    # matching contiguity flag name (only C enabled for now) and ndims, the number of dimensions
    # to test.
    # Test ones, zeros, empty and full.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp and stores the dtype set, the order
        # mapping with only C order, and the maximum dimension count 10.
        super().setUp()
        dtypes = {np.dtype(tp) for tp in "efdFDBbhil?"}
        self.dtypes = dtypes
        self.orders = {
            "C": "c_contiguous"
        }  # XXX: reeenable when implemented, 'F': 'f_contiguous'}
        self.ndims = 10

    def check_function(self, func, fill_value=None):
        # [FUNCTION DOC] HELPER: For sizes 0, 1, 2, each dimension count below ndims, each order
        # and each dtype, calls the creation function with a cubic shape, the order and dtype (and
        # fill_value when given). Checks the result dtype, that the flag matching the order is
        # set, and for filled arrays that all elements equal the fill value converted to the
        # dtype.
        par = ((0, 1, 2), range(self.ndims), self.orders, self.dtypes)
        fill_kwarg = {}
        if fill_value is not None:
            fill_kwarg = {"fill_value": fill_value}

        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]

            arr = func(shape, order=order, dtype=dtype, **fill_kwarg)

            assert_equal(arr.dtype, dtype)
            assert_(getattr(arr.flags, self.orders[order]))

            if fill_value is not None:
                val = fill_value
                assert_equal(arr, dtype.type(val))

    def test_zeros(self):
        # [FUNCTION DOC] Runs the creation check for zeros.
        self.check_function(np.zeros)

    def test_ones(self):
        # [FUNCTION DOC] Runs the creation check for ones.
        self.check_function(np.ones)

    def test_empty(self):
        # [FUNCTION DOC] Runs the creation check for empty.
        self.check_function(np.empty)

    def test_full(self):
        # [FUNCTION DOC] Runs the creation check for full with fill values 0 and 1.
        self.check_function(np.full, 0)
        self.check_function(np.full, 1)

    @skipif(TEST_WITH_TORCHDYNAMO, reason="fails with dynamo")
    @skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_for_reference_leak(self):
        # [FUNCTION DOC] Checks the reference count of the integer used as a dimension is
        # unchanged after calling zeros, ones, empty and full with a ten-dimensional shape.
        # Skipped under dynamo or without refcounts.
        # Make sure we have an object for reference
        dim = 1
        beg = sys.getrefcount(dim)
        np.zeros([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.ones([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.empty([dim] * 10)
        assert_(sys.getrefcount(dim) == beg)
        np.full([dim] * 10, 0)
        assert_(sys.getrefcount(dim) == beg)


@skip(reason="implement order etc")  # FIXME: make xfail
@instantiate_parametrized_tests
class TestLikeFuncs(TestCase):
    # [CLASS DOC] Test case for the like creation functions (ones_like, zeros_like, empty_like,
    # full_like) checking shape, dtype, memory layout, order and shape arguments. State set in
    # setUp: data, pairs of prototype array and optional dtype; shapes, target shapes of ranks 0
    # to 3. The class is skipped (order support missing).
    """Test ones_like, zeros_like, empty_like and full_like"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp, builds prototype arrays (scalar, 1-D, 2-D
        # and 3-D, C, Fortran and neither layout) paired with an optional new dtype, and the list
        # of shapes to test.
        super().setUp()
        self.data = [
            # Array scalars
            (np.array(3.0), None),
            (np.array(3), "f8"),
            # 1D arrays
            (np.arange(6, dtype="f4"), None),
            (np.arange(6), "c16"),
            # 2D C-layout arrays
            (np.arange(6).reshape(2, 3), None),
            (np.arange(6).reshape(3, 2), "i1"),
            # 2D F-layout arrays
            (np.arange(6).reshape((2, 3), order="F"), None),
            (np.arange(6).reshape((3, 2), order="F"), "i1"),
            # 3D C-layout arrays
            (np.arange(24).reshape(2, 3, 4), None),
            (np.arange(24).reshape(4, 3, 2), "f4"),
            # 3D F-layout arrays
            (np.arange(24).reshape((2, 3, 4), order="F"), None),
            (np.arange(24).reshape((4, 3, 2), order="F"), "f4"),
            # 3D non-C/F-layout arrays
            (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None),
            (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), "?"),
        ]
        self.shapes = [
            (),
            (5,),
            (
                5,
                6,
            ),
            (
                5,
                6,
                7,
            ),
        ]

    def compare_array_value(self, dz, value, fill_value):
        # [FUNCTION DOC] HELPER: If a value is given, checks every element of the result equals it
        # (for fill functions, equals the value first converted to the result dtype).
        if value is not None:
            if fill_value:
                # Conversion is close to what np.full_like uses
                # but we  may want to convert directly in the future
                # which may result in errors (where this does not).
                z = np.array(value).astype(dz.dtype)
                assert_(np.all(dz == z))
            else:
                assert_(np.all(dz == value))

    def check_like_function(self, like_function, value, fill_value=False):
        # [FUNCTION DOC] HELPER: For each prototype and dtype, calls the like function with
        # default order, C, F and A order and checks shape, dtype, and contiguity flags (also
        # stride ratios for default order). Then with each target shape and order checks shape,
        # dtype and layout, and for the K order that stride ordering follows the prototype or a
        # default C layout when ranks differ. Values are checked through the compare helper.
        if fill_value:
            fill_kwarg = {"fill_value": value}
        else:
            fill_kwarg = {}
        for d, dtype in self.data:
            # default (K) order, dtype
            dz = like_function(d, dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_equal(
                np.array(dz.strides) * d.dtype.itemsize,
                np.array(d.strides) * dz.dtype.itemsize,
            )
            assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
            assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # C order, default dtype
            dz = like_function(d, order="C", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # F order, default dtype
            dz = like_function(d, order="F", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # A order
            dz = like_function(d, order="A", dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            if d.flags.f_contiguous:
                assert_(dz.flags.f_contiguous)
            else:
                assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # Test the 'shape' parameter
            for s in self.shapes:
                for o in "CFA":
                    sz = like_function(d, dtype=dtype, shape=s, order=o, **fill_kwarg)
                    assert_equal(sz.shape, s)
                    if dtype is None:
                        assert_equal(sz.dtype, d.dtype)
                    else:
                        assert_equal(sz.dtype, np.dtype(dtype))
                    if o == "C" or (o == "A" and d.flags.c_contiguous):
                        assert_(sz.flags.c_contiguous)
                    elif o == "F" or (o == "A" and d.flags.f_contiguous):
                        assert_(sz.flags.f_contiguous)
                    self.compare_array_value(sz, value, fill_value)

                if d.ndim != len(s):
                    assert_equal(
                        np.argsort(
                            like_function(
                                d, dtype=dtype, shape=s, order="K", **fill_kwarg
                            ).strides
                        ),
                        np.argsort(np.empty(s, dtype=dtype, order="C").strides),
                    )
                else:
                    assert_equal(
                        np.argsort(
                            like_function(
                                d, dtype=dtype, shape=s, order="K", **fill_kwarg
                            ).strides
                        ),
                        np.argsort(d.strides),
                    )

    def test_ones_like(self):
        # [FUNCTION DOC] Runs the like-function checks for ones_like with value 1.
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        # [FUNCTION DOC] Runs the like-function checks for zeros_like with value 0.
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        # [FUNCTION DOC] Runs the like-function checks for empty_like without value checking.
        self.check_like_function(np.empty_like, None)

    def test_filled_like(self):
        # [FUNCTION DOC] Runs the like-function checks for full_like with fill values 0, 1, 1000,
        # 123.456 and infinity.
        self.check_like_function(np.full_like, 0, True)
        self.check_like_function(np.full_like, 1, True)
        self.check_like_function(np.full_like, 1000, True)
        self.check_like_function(np.full_like, 123.456, True)
        # Inf to integer casts cause invalid-value errors: ignore them.
        self.check_like_function(np.full_like, np.inf, True)

    @parametrize("likefunc", [np.empty_like, np.full_like, np.zeros_like, np.ones_like])
    @parametrize("dtype", [str, bytes])
    def test_dtype_str_bytes(self, likefunc, dtype):
        # [FUNCTION DOC] Parametrized over the four like functions and str or bytes dtypes:
        # applies the function to a non-contiguous slice and checks the resulting strides equal
        # (16, 4) for str and (4, 1) for bytes (numpy issue 19860); fill value is the empty string
        # for full_like.
        # Regression test for gh-19860
        a = np.arange(16).reshape(2, 8)
        b = a[:, ::2]  # Ensure b is not contiguous.
        kwargs = {"fill_value": ""} if likefunc is np.full_like else {}
        result = likefunc(b, dtype=dtype, **kwargs)
        if dtype is str:
            if result.strides != (16, 4):
                raise AssertionError(f"Expected strides (16, 4), got {result.strides}")
        else:
            # dtype is bytes
            if result.strides != (4, 1):
                raise AssertionError(f"Expected strides (4, 1), got {result.strides}")


class TestCorrelate(TestCase):
    # [CLASS DOC] Test case for correlate. The _setup helper stores input and expected output
    # arrays on the test case.
    def _setup(self, dt):
        # [FUNCTION DOC] HELPER: Stores on the test case input arrays x, xs (strided) and y of the
        # given dtype and the expected full-mode correlation results z1, z1_4, z1r, z2, z2r and
        # zs.
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        self.xs = np.arange(1, 20)[::3]
        self.y = np.array([-1, -2, -3], dtype=dt)
        self.z1 = np.array([-3.0, -8.0, -14.0, -20.0, -26.0, -14.0, -5.0], dtype=dt)
        self.z1_4 = np.array([-2.0, -5.0, -8.0, -11.0, -14.0, -5.0], dtype=dt)
        self.z1r = np.array([-15.0, -22.0, -22.0, -16.0, -10.0, -4.0, -1.0], dtype=dt)
        self.z2 = np.array([-5.0, -14.0, -26.0, -20.0, -14.0, -8.0, -3.0], dtype=dt)
        self.z2r = np.array([-1.0, -4.0, -10.0, -16.0, -22.0, -22.0, -15.0], dtype=dt)
        self.zs = np.array(
            [-3.0, -14.0, -30.0, -48.0, -66.0, -84.0, -102.0, -54.0, -19.0], dtype=dt
        )

    def test_float(self):
        # [FUNCTION DOC] With float arrays, checks full-mode correlation of several operand pairs
        # (different lengths, swapped, flipped, strided) against the expected results.
        self._setup(float)
        z = np.correlate(self.x, self.y, "full")
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.x, self.y[:-1], "full")
        assert_array_almost_equal(z, self.z1_4)
        z = np.correlate(self.y, self.x, "full")
        assert_array_almost_equal(z, self.z2)
        z = np.correlate(np.flip(self.x), self.y, "full")
        assert_array_almost_equal(z, self.z1r)
        z = np.correlate(self.y, np.flip(self.x), "full")
        assert_array_almost_equal(z, self.z2r)
        z = np.correlate(self.xs, self.y, "full")
        assert_array_almost_equal(z, self.zs)

    def test_no_overwrite(self):
        # [FUNCTION DOC] Checks correlate leaves its input arrays unchanged.
        d = np.ones(100)
        k = np.ones(3)
        np.correlate(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    def test_complex(self):
        # [FUNCTION DOC] Checks full-mode correlation of complex inputs equals the conjugate of
        # the flipped expected result.
        x = np.array([1, 2, 3, 4 + 1j], dtype=complex)
        y = np.array([-1, -2j, 3 + 1j], dtype=complex)
        r_z = np.array([3 - 1j, 6, 8 + 1j, 11 + 5j, -5 + 8j, -4 - 1j], dtype=complex)
        r_z = np.flip(r_z).conjugate()
        z = np.correlate(y, x, mode="full")
        assert_array_almost_equal(z, r_z)

    def test_zero_size(self):
        # [FUNCTION DOC] Checks correlate raises ValueError or RuntimeError when either input is
        # empty.
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.array([]), np.ones(1000), mode="full")
        with pytest.raises((ValueError, RuntimeError)):
            np.correlate(np.ones(1000), np.array([]), mode="full")

    @skip(reason="do not implement deprecated behavior")
    def test_mode(self):
        # [FUNCTION DOC] Intended to check mode handling (deprecated short mode name warns,
        # integer modes, None is a TypeError). Skipped because deprecated behavior is not
        # implemented.
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.correlate(d, k, mode="valid")
        with assert_warns(DeprecationWarning):
            valid_mode = np.correlate(d, k, mode="v")
        assert_array_equal(valid_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.correlate(d, k, mode=-1)
        assert_array_equal(np.correlate(d, k, mode=0), valid_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.correlate(d, k, mode=None)


class TestConvolve(TestCase):
    # [CLASS DOC] Test case for convolve.
    def test_object(self):
        # [FUNCTION DOC] Checks convolution of two float lists gives 3 in the interior of the
        # result.
        d = [1.0] * 100
        k = [1.0] * 3
        assert_array_almost_equal(np.convolve(d, k)[2:-2], np.full(98, 3))

    def test_no_overwrite(self):
        # [FUNCTION DOC] Checks convolve leaves its input arrays unchanged.
        d = np.ones(100)
        k = np.ones(3)
        np.convolve(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    @skip(reason="do not implement deprecated behavior")
    def test_mode(self):
        # [FUNCTION DOC] Intended to check mode handling for convolve (deprecated short name,
        # integer modes, None). Skipped because deprecated behavior is not implemented.
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.convolve(d, k, mode="full")
        with assert_warns(DeprecationWarning):
            full_mode = np.convolve(d, k, mode="f")
        assert_array_equal(full_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.convolve(d, k, mode=-1)
        assert_array_equal(np.convolve(d, k, mode=2), full_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.convolve(d, k, mode=None)

    def test_numpy_doc_examples(self):
        # [FUNCTION DOC] Checks the documented convolution examples in full, same and valid modes
        # against known results.
        conv = np.convolve([1, 2, 3], [0, 1, 0.5])
        assert_allclose(conv, [0.0, 1.0, 2.5, 4.0, 1.5], atol=1e-15)

        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "same")
        assert_allclose(conv, [1.0, 2.5, 4.0], atol=1e-15)

        conv = np.convolve([1, 2, 3], [0, 1, 0.5], "valid")
        assert_allclose(conv, [2.5], atol=1e-15)


class TestDtypePositional(TestCase):
    # [CLASS DOC] Test case that dtype can be passed positionally.
    def test_dtype_positional(self):
        # [FUNCTION DOC] Calls the empty function with the dtype given as the second positional
        # argument; passes if no error is raised.
        np.empty((2,), bool)


@instantiate_parametrized_tests
class TestArgwhere(TestCase):
    # [CLASS DOC] Test case for argwhere.
    @parametrize("nd", [0, 1, 2])
    def test_nd(self, nd):
        # [FUNCTION DOC] Parametrized over 0, 1 and 2 dimensions: on a boolean array with all, all
        # but one, one and no true elements, checks the result shape has one row per true element
        # and one column per dimension.
        # get an nd array with multiple elements in every dimension
        x = np.empty((2,) * nd, dtype=bool)

        # none
        x[...] = False
        assert_equal(np.argwhere(x).shape, (0, nd))

        # only one
        x[...] = False
        x.ravel()[0] = True
        assert_equal(np.argwhere(x).shape, (1, nd))

        # all but one
        x[...] = True
        x.ravel()[0] = False
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # all
        x[...] = True
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        # [FUNCTION DOC] Checks argwhere on a condition over a 2x3 array gives the row and column
        # index pairs in row-major order.
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1), [[0, 2], [1, 0], [1, 1], [1, 2]])

    def test_list(self):
        # [FUNCTION DOC] Checks argwhere on a list gives a column of indices of the non-zero
        # entries.
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestStringFunction(TestCase):
    # [CLASS DOC] Test case for the global string function hook. Marked expected pass-or-fail
    # under dynamo (TODO).
    def test_set_string_function(self):
        # [FUNCTION DOC] Installs a custom repr function and checks repr gives its text, restores
        # the default and checks the standard form; repeats for str. Mutates global array printing
        # state.
        a = np.array([1])
        np.set_string_function(lambda x: "FOO", repr=True)
        assert_equal(repr(a), "FOO")
        np.set_string_function(None, repr=True)
        assert_equal(repr(a), "array([1])")

        np.set_string_function(lambda x: "FOO", repr=False)
        assert_equal(str(a), "FOO")
        np.set_string_function(None, repr=False)
        assert_equal(str(a), "[1]")


class TestRoll(TestCase):
    # [CLASS DOC] Test case for roll.
    def test_roll1d(self):
        # [FUNCTION DOC] Checks rolling a 1-D range by 2 moves the last two elements to the front.
        x = np.arange(10)
        xr = np.roll(x, 2)
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))

    def test_roll2d(self):
        # [FUNCTION DOC] Checks rolling a 2x5 array with no axis (flattened), single axes, tuples
        # of shifts and axes, repeated axes, and shifts larger than the axis length.
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r = np.roll(x2, 1)
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        x2r = np.roll(x2, 1, axis=0)
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, 1, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # Roll multiple axes at once.
        x2r = np.roll(x2, 1, axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (0, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, (0, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        x2r = np.roll(x2, (1, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (-1, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]]))

        # Roll the same axis multiple times.
        x2r = np.roll(x2, 1, axis=(0, 0))
        assert_equal(x2r, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

        x2r = np.roll(x2, 1, axis=(1, 1))
        assert_equal(x2r, np.array([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]))

        # Roll more than one turn in either direction.
        x2r = np.roll(x2, 6, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, -4, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

    def test_roll_empty(self):
        # [FUNCTION DOC] Checks rolling an empty array returns an empty array.
        x = np.array([])
        assert_equal(np.roll(x, 1), np.array([]))


class TestRollaxis(TestCase):
    # [CLASS DOC] Test case for rollaxis. The class attribute tgtshape maps pairs of axis and
    # start to the expected shape for an input of shape one, two, three, four.
    # expected shape indexed by (axis, start) for array of
    # shape (1, 2, 3, 4)
    tgtshape = {
        (0, 0): (1, 2, 3, 4),
        (0, 1): (1, 2, 3, 4),
        (0, 2): (2, 1, 3, 4),
        (0, 3): (2, 3, 1, 4),
        (0, 4): (2, 3, 4, 1),
        (1, 0): (2, 1, 3, 4),
        (1, 1): (1, 2, 3, 4),
        (1, 2): (1, 2, 3, 4),
        (1, 3): (1, 3, 2, 4),
        (1, 4): (1, 3, 4, 2),
        (2, 0): (3, 1, 2, 4),
        (2, 1): (1, 3, 2, 4),
        (2, 2): (1, 2, 3, 4),
        (2, 3): (1, 2, 3, 4),
        (2, 4): (1, 2, 4, 3),
        (3, 0): (4, 1, 2, 3),
        (3, 1): (1, 4, 2, 3),
        (3, 2): (1, 2, 4, 3),
        (3, 3): (1, 2, 3, 4),
        (3, 4): (1, 2, 3, 4),
    }

    def test_exceptions(self):
        # [FUNCTION DOC] Checks rollaxis raises AxisError for out-of-range axis or start values
        # (below minus 4 or above the allowed bound).
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        assert_raises(np.AxisError, np.rollaxis, a, -5, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, -5)
        assert_raises(np.AxisError, np.rollaxis, a, 4, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, 5)

    @xfail  # XXX: ndarray.attributes
    def test_results(self):
        # [FUNCTION DOC] For every pair in the expected-shape table, with positive and negative
        # axis and start, checks the result has the expected shape, holds the same data as a view
        # and does not own its data. Marked expected failure because array flag attributes are
        # missing.
        a = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4).copy()
        aind = np.indices(a.shape)
        assert_(a.flags["OWNDATA"])
        for i, j in self.tgtshape:
            # positive axis, positive start
            res = np.rollaxis(a, axis=i, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, j)], str((i, j)))
            assert_(not res.flags["OWNDATA"])

            # negative axis, positive start
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, j)])
            assert_(not res.flags["OWNDATA"])

            # positive axis, negative start
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, 4 - jp)])
            assert_(not res.flags["OWNDATA"])

            # negative axis, negative start
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, 4 - jp)])
            assert_(not res.flags["OWNDATA"])


class TestMoveaxis(TestCase):
    # [CLASS DOC] Test case for moveaxis.
    def test_move_to_end(self):
        # [FUNCTION DOC] Moves each of several source axes of a 5x6x7 array to the last position
        # and computes the shapes. The expected value is passed as the message argument of the
        # truth-assertion helper, so only the truthiness of the actual shape is checked, not
        # equality with the expected shape.
        # KNOWN ISSUE: The expected value is passed as the message argument of the truth-assertion
        # helper, so only the truthiness of the actual shape is checked, not equality with the
        # expected shape.
        x = np.random.randn(5, 6, 7)
        for source, expected in [
            (0, (6, 7, 5)),
            (1, (5, 7, 6)),
            (2, (5, 6, 7)),
            (-1, (5, 6, 7)),
        ]:
            actual = np.moveaxis(x, source, -1).shape
            assert_(actual, expected)

    def test_move_new_position(self):
        # [FUNCTION DOC] Moves axes of a 1x2x3x4 array to new positions, including a negative
        # destination, and computes the shapes. The expected value is passed as the message
        # argument of the truth-assertion helper, so only the truthiness of the actual shape is
        # checked, not equality with the expected shape.
        # KNOWN ISSUE: The expected value is passed as the message argument of the truth-assertion
        # helper, so only the truthiness of the actual shape is checked, not equality with the
        # expected shape.
        x = np.random.randn(1, 2, 3, 4)
        for source, destination, expected in [
            (0, 1, (2, 1, 3, 4)),
            (1, 2, (1, 3, 2, 4)),
            (1, -1, (1, 3, 4, 2)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_preserve_order(self):
        # [FUNCTION DOC] Moves axes onto themselves (single, negative, list and range forms) and
        # computes the shapes. The expected value is passed as the message argument of the truth-
        # assertion helper, so only the truthiness of the actual shape is checked, not equality
        # with the expected shape.
        # KNOWN ISSUE: The expected value is passed as the message argument of the truth-assertion
        # helper, so only the truthiness of the actual shape is checked, not equality with the
        # expected shape.
        x = np.zeros((1, 2, 3, 4))
        for source, destination in [
            (0, 0),
            (3, -1),
            (-1, 3),
            ([0, -1], [0, -1]),
            ([2, 0], [2, 0]),
            (range(4), range(4)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, (1, 2, 3, 4))

    def test_move_multiples(self):
        # [FUNCTION DOC] Moves several axes at once on an array with a zero-length dimension and
        # computes the shapes. The expected value is passed as the message argument of the truth-
        # assertion helper, so only the truthiness of the actual shape is checked, not equality
        # with the expected shape.
        # KNOWN ISSUE: The expected value is passed as the message argument of the truth-assertion
        # helper, so only the truthiness of the actual shape is checked, not equality with the
        # expected shape.
        x = np.zeros((0, 1, 2, 3))
        for source, destination, expected in [
            ([0, 1], [2, 3], (2, 3, 0, 1)),
            ([2, 3], [0, 1], (2, 3, 0, 1)),
            ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
            ([3, 0], [1, 0], (0, 3, 1, 2)),
            ([0, 3], [0, 1], (0, 3, 1, 2)),
        ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_errors(self):
        # [FUNCTION DOC] Checks moveaxis raises AxisError for out of range source or destination,
        # ValueError for repeated axes, ValueError or RuntimeError for mismatched counts, and that
        # a list input comes back as an array.
        # KNOWN ISSUE: The final shape-like check uses the same truth-assertion call with the
        # expected value as the message, so it only checks truthiness.
        x = np.random.randn(1, 2, 3)
        assert_raises(np.AxisError, np.moveaxis, x, 3, 0)  # 'source.*out of bounds',
        assert_raises(np.AxisError, np.moveaxis, x, -4, 0)  # 'source.*out of bounds',
        assert_raises(
            np.AxisError,
            np.moveaxis,
            x,
            0,
            5,  # 'destination.*out of bounds',
        )
        assert_raises(
            ValueError,
            np.moveaxis,
            x,
            [0, 0],
            [0, 1],  # 'repeated axis in `source`',
        )
        assert_raises(
            ValueError,  # 'repeated axis in `destination`',
            np.moveaxis,
            x,
            [0, 1],
            [1, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 'must have the same number',
            np.moveaxis,
            x,
            0,
            [0, 1],
        )
        assert_raises(
            (ValueError, RuntimeError),  # 'must have the same number',
            np.moveaxis,
            x,
            [0, 1],
            [0],
        )

        x = [1, 2, 3]
        result = np.moveaxis(x, 0, 0)
        assert_(x, list(result))
        assert_(isinstance(result, np.ndarray))


class TestCross(TestCase):
    # [CLASS DOC] Test case for cross product.
    def test_2x2(self):
        # [FUNCTION DOC] Checks the two-dimensional cross product gives a scalar and changes sign
        # when operands swap.
        u = [1, 2]
        v = [3, 4]
        z = -2
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_2x3(self):
        # [FUNCTION DOC] Checks the cross product of a 2-vector and a 3-vector gives the expected
        # 3-vector and flips sign when swapped.
        u = [1, 2]
        v = [3, 4, 5]
        z = np.array([10, -5, -2])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_3x3(self):
        # [FUNCTION DOC] Checks the cross product of two 3-vectors gives the expected vector and
        # flips sign when swapped.
        u = [1, 2, 3]
        v = [4, 5, 6]
        z = np.array([-3, 6, -3])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_broadcasting(self):
        # [FUNCTION DOC] Checks broadcasting of cross products with tiled operands and axis
        # arguments for 2x2, 2x3, 3x2 and 3x3 combinations, and that crossing a vector with itself
        # gives zero.
        # Ticket #2624 (Trac #2032)
        u = np.tile([1, 2], (11, 1))
        v = np.tile([3, 4], (11, 1))
        z = -2
        assert_equal(np.cross(u, v), z)
        assert_equal(np.cross(v, u), -z)
        assert_equal(np.cross(u, u), 0)

        u = np.tile([1, 2], (11, 1)).T
        v = np.tile([3, 4, 5], (11, 1))
        z = np.tile([10, -5, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0), z)
        assert_equal(np.cross(v, u.T), -z)
        assert_equal(np.cross(v, v), 0)

        u = np.tile([1, 2, 3], (11, 1)).T
        v = np.tile([3, 4], (11, 1)).T
        z = np.tile([-12, 9, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)
        assert_equal(np.cross(v.T, u.T), -z)
        assert_equal(np.cross(u.T, u.T), 0)

        u = np.tile([1, 2, 3], (5, 1))
        v = np.tile([4, 5, 6], (5, 1)).T
        z = np.tile([-3, 6, -3], (5, 1))
        assert_equal(np.cross(u, v, axisb=0), z)
        assert_equal(np.cross(v.T, u), -z)
        assert_equal(np.cross(u, u), 0)

    def test_broadcasting_shapes(self):
        # [FUNCTION DOC] Checks result shapes of cross with broadcast operands and axis arguments,
        # that out-of-range axes raise AxisError, and that every output axis position gives the
        # expected shape.
        u = np.ones((2, 1, 3))
        v = np.ones((5, 3))
        assert_equal(np.cross(u, v).shape, (2, 5, 3))
        u = np.ones((10, 3, 5))
        v = np.ones((2, 5))
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
        u = np.ones((10, 3, 5, 7))
        v = np.ones((5, 7, 2))
        assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
        assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
        # gh-5885
        u = np.ones((3, 4, 2))
        for axisc in range(-2, 2):
            assert_equal(np.cross(u, u, axisc=axisc).shape, (3, 4))

    @skipif(numpy.__version__ < "1.24", reason="fix landed in NumPy 1.24")
    def test_uint8_int32_mixed_dtypes(self):
        # [FUNCTION DOC] Checks cross of mixed uint8 and int32 vectors gives the int32 result and
        # negated when swapped (numpy issue 19138). Skipped for older numpy versions.
        # KNOWN ISSUE: The version guard compares version strings lexicographically, which
        # misorders some version numbers.
        # regression test for gh-19138
        u = np.array([[195, 8, 9]], np.uint8)
        v = np.array([250, 166, 68], np.int32)
        z = np.array([[950, 11010, -30370]], dtype=np.int32)
        assert_equal(np.cross(v, u), z)
        assert_equal(np.cross(u, v), -z)


class TestOuterMisc(TestCase):
    # [CLASS DOC] Test case for outer product features.
    def test_outer_out_param(self):
        # [FUNCTION DOC] Checks outer writes its result into the supplied output arrays of two
        # different shapes and returns them.
        arr1 = np.ones((5,))
        arr2 = np.ones((2,))
        arr3 = np.linspace(-2, 2, 5)
        out1 = np.empty(shape=(5, 5))
        out2 = np.empty(shape=(2, 5))
        res1 = np.outer(arr1, arr3, out1)
        assert_equal(res1, out1)
        assert_equal(np.outer(arr2, arr3, out2), out2)


@instantiate_parametrized_tests
class TestIndices(TestCase):
    # [CLASS DOC] Test case for indices.
    def test_simple(self):
        # [FUNCTION DOC] Checks the indices of a 4x3 grid give the row index array and the column
        # index array.
        [x, y] = np.indices((4, 3))
        assert_array_equal(x, np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert_array_equal(y, np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]))

    def test_single_input(self):
        # [FUNCTION DOC] Checks indices of a one-element dimension tuple dense and sparse give a
        # single array of 0 through 3.
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))

        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        # [FUNCTION DOC] Checks indices for an empty dimension tuple and for a zero-size
        # dimension, dense and sparse, give empty results.
        assert_array_equal([], np.indices(()))
        assert_array_equal([], np.indices((), sparse=True))
        assert_array_equal([[]], np.indices((0,)))
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        # [FUNCTION DOC] Checks sparse indices of a 4x3 grid give a column vector and a row
        # vector.
        [x, y] = np.indices((4, 3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    @parametrize("dims", [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        # [FUNCTION DOC] Parametrized over four dtypes and three dimension tuples: checks the
        # dense result and each sparse result array have the requested dtype.
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)

        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestRequire(TestCase):
    # [CLASS DOC] Test case for require. The class attribute flag_names lists the flag names and
    # aliases tested. Marked expected pass-or-fail under dynamo (TODO).
    flag_names = [
        "C",
        "C_CONTIGUOUS",
        "CONTIGUOUS",
        "F",
        "F_CONTIGUOUS",
        "FORTRAN",
        "A",
        "ALIGNED",
        "W",
        "WRITEABLE",
        "O",
        "OWNDATA",
    ]

    def generate_all_false(self, dtype):
        # [FUNCTION DOC] HELPER: Builds a read-only structured array with a padding field and
        # returns the named field as a view with C, F, O, W and A flags all false, asserting this.
        arr = np.zeros((2, 2), [("junk", "i1"), ("a", dtype)])
        arr.setflags(write=False)
        a = arr["a"]
        assert_(not a.flags["C"])
        assert_(not a.flags["F"])
        assert_(not a.flags["O"])
        assert_(not a.flags["W"])
        assert_(not a.flags["A"])
        return a

    def set_and_check_flag(self, flag, dtype, arr):
        # [FUNCTION DOC] HELPER: Calls require on the array with the dtype (or its own dtype if
        # None) and the flag; checks the result has the flag and dtype, and that requiring again
        # returns the same object unless the flag is the owndata one, in which case only the flag
        # is checked.
        if dtype is None:
            dtype = arr.dtype
        b = np.require(arr, dtype, [flag])
        assert_(b.flags[flag])
        assert_(b.dtype == dtype)

        # a further call to np.require ought to return the same array
        # unless OWNDATA is specified.
        c = np.require(b, None, [flag])
        if flag[0] != "O":
            assert_(c is b)
        else:
            assert_(c.flags[flag])

    def test_require_each(self):
        # [FUNCTION DOC] For each source dtype, target dtype (including none) and flag name,
        # builds an all-false-flags array and checks require can set the flag.
        id = ["f8", "i4"]
        fd = [None, "f8", "c16"]
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            a = self.generate_all_false(idtype)
            self.set_and_check_flag(flag, fdtype, a)

    def test_unknown_requirement(self):
        # [FUNCTION DOC] Checks require raises KeyError for an unknown requirement letter.
        a = self.generate_all_false("f8")
        assert_raises(KeyError, np.require, a, None, "Q")

    def test_non_array_input(self):
        # [FUNCTION DOC] Checks require on a list with int32 dtype and several flags returns an
        # array with those flags set, the dtype and the values.
        a = np.require([1, 2, 3, 4], "i4", ["C", "A", "O"])
        assert_(a.flags["O"])
        assert_(a.flags["C"])
        assert_(a.flags["A"])
        assert_(a.dtype == "i4")
        assert_equal(a, [1, 2, 3, 4])

    def test_C_and_F_simul(self):
        # [FUNCTION DOC] Checks require raises ValueError when both C and F contiguity are
        # requested.
        a = self.generate_all_false("f8")
        assert_raises(ValueError, np.require, a, None, ["C", "F"])


@xpassIfTorchDynamo_np  # (reason="TODO")
class TestBroadcast(TestCase):
    # [CLASS DOC] Test case for the broadcast object. Marked expected pass-or-fail under dynamo
    # (TODO).
    def test_broadcast_in_args(self):
        # [FUNCTION DOC] Builds broadcast objects from arrays and from nested broadcast objects,
        # and checks shape, number of dimensions, number of iterators and that each iterator's
        # base is the original array (numpy issue 5881).
        # gh-5881
        arrs = [
            np.empty((6, 7)),
            np.empty((5, 6, 1)),
            np.empty((7,)),
            np.empty((5, 1, 7)),
        ]
        mits = [
            np.broadcast(*arrs),
            np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])),
            np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])),
            np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])),
            np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1]),
        ]
        for mit in mits:
            assert_equal(mit.shape, (5, 6, 7))
            assert_equal(mit.ndim, 3)
            assert_equal(mit.nd, 3)
            assert_equal(mit.numiter, 4)
            for a, ia in zip(arrs, mit.iters):
                assert_(a is ia.base)

    def test_broadcast_single_arg(self):
        # [FUNCTION DOC] Checks a broadcast of a single array has the same shape and that its
        # iterator refers to that array (numpy issue 6899).
        # gh-6899
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        assert_equal(mit.shape, (5, 6, 7))
        assert_equal(mit.ndim, 3)
        assert_equal(mit.nd, 3)
        assert_equal(mit.numiter, 1)
        assert_(arrs[0] is mit.iters[0].base)

    def test_number_of_arguments(self):
        # [FUNCTION DOC] Checks broadcast accepts up to 32 arguments with matching iterator count
        # and raises ValueError for 33 or more.
        arr = np.empty((5,))
        for j in range(35):
            arrs = [arr] * j
            if j > 32:
                assert_raises(ValueError, np.broadcast, *arrs)
            else:
                mit = np.broadcast(*arrs)
                assert_equal(mit.numiter, j)

    def test_broadcast_error_kwargs(self):
        # [FUNCTION DOC] Checks broadcast with empty keyword arguments behaves the same as
        # without, and that an unexpected keyword raises ValueError.
        # gh-13455
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        mit2 = np.broadcast(*arrs, **{})  # noqa: PIE804
        assert_equal(mit.shape, mit2.shape)
        assert_equal(mit.ndim, mit2.ndim)
        assert_equal(mit.nd, mit2.nd)
        assert_equal(mit.numiter, mit2.numiter)
        assert_(mit.iters[0].base is mit2.iters[0].base)

        assert_raises(ValueError, np.broadcast, 1, **{"x": 1})  # noqa: PIE804

    @skip(reason="error messages do not match.")
    def test_shape_mismatch_error_message(self):
        # [FUNCTION DOC] Intended to check the shape mismatch error message text. Skipped because
        # error messages do not match.
        with assert_raises(
            ValueError,
            match=r"arg 0 with shape \(1, 3\) and arg 2 with shape \(2,\)",
        ):
            np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])


class TestTensordot(TestCase):
    # [CLASS DOC] Test case for tensordot.
    def test_zero_dimension(self):
        # [FUNCTION DOC] Checks tensordot over a zero-length contracted axis equals the matrix
        # product (numpy issue 5663).
        # Test resolution to issue #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.dot(a, b))

    def test_zero_dimension_einsum(self):
        # [FUNCTION DOC] Checks tensordot over a zero-length contracted axis equals the einsum
        # result.
        # Test resolution to issue #5663
        a = np.zeros((3, 0))
        b = np.zeros((0, 4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.einsum("ij,jk", a, b))

    def test_zero_dimensional(self):
        # [FUNCTION DOC] Checks tensordot of zero-dimensional arrays with no contracted axes
        # returns the original scalar (numpy issue 12130).
        # gh-12130
        arr_0d = np.array(1)
        ret = np.tensordot(
            arr_0d, arr_0d, ([], [])
        )  # contracting no axes is well defined
        assert_array_equal(ret, arr_0d)


if __name__ == "__main__":
    run_tests()
