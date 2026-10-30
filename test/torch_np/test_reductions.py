# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for reduction functions of torch._numpy (flatnonzero, any, all, mean, sum
# and the generic axis, keepdims and out behavior shared by any, all, argmin, argmax, min,
# max, mean, sum, prod, std, var, count_nonzero, plus cumsum and cumprod). Uses real NumPy
# under TorchDynamo and torch._numpy otherwise. Import builds parameter decorators for the
# axis values and the function list, and sets of functions that lack tuple axes, the out
# argument, or float-to-integer casts, which tests skip.

from unittest import skipIf, SkipTest

import numpy
import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    import numpy.core.numeric as _util  # for normalize_axis_tuple
    from numpy.testing import (
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
    )
else:
    import torch._numpy as np
    from torch._numpy import _util
    from torch._numpy.testing import (
        assert_allclose,
        assert_almost_equal,
        assert_array_equal,
        assert_equal,
    )


class TestFlatnonzero(TestCase):
    # [CLASS DOC] Tests for flatnonzero.
    def test_basic(self):
        # [FUNCTION DOC] Checks that flatnonzero of the integers minus two to two returns the
        # flat indices of the non-zero entries.
        x = np.arange(-2, 3)
        assert_equal(np.flatnonzero(x), [0, 1, 3, 4])


class TestAny(TestCase):
    # [CLASS DOC] Tests for the any reduction.
    def test_basic(self):
        # [FUNCTION DOC] Checks any on flat lists: true when some element is non-zero, false
        # for an all-zero list.
        y1 = [0, 0, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 0, 1, 0]
        if not np.any(y1):
            raise AssertionError("Expected np.any(y1) to be True")
        if not np.any(y3):
            raise AssertionError("Expected np.any(y3) to be True")
        if np.any(y2):
            raise AssertionError("Expected np.any(y2) to be False")

    def test_nd(self):
        # [FUNCTION DOC] Checks any on a nested list: overall true, per-column and per-row
        # results along axes zero and one, the default overall result, and that the per-axis
        # result is an ndarray.
        y1 = [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
        if not np.any(y1):
            raise AssertionError("Expected np.any(y1) to be True")
        assert_equal(np.any(y1, axis=0), [1, 1, 0])
        assert_equal(np.any(y1, axis=1), [0, 1, 1])
        assert_equal(np.any(y1), True)
        if not isinstance(np.any(y1, axis=1), np.ndarray):
            raise AssertionError("Expected np.any result to be np.ndarray")

    # YYY: deduplicate
    def test_method_vs_function(self):
        # [FUNCTION DOC] Checks that the any function and the any method agree on a
        # two-dimensional array.
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.any(y), y.any())


class TestAll(TestCase):
    # [CLASS DOC] Tests for the all reduction.
    def test_basic(self):
        # [FUNCTION DOC] Checks all on flat lists: false when any element is zero, true when
        # all are non-zero, and true on the logical negation of an all-zero array.
        y1 = [0, 1, 1, 0]
        y2 = [0, 0, 0, 0]
        y3 = [1, 1, 1, 1]
        if np.all(y1):
            raise AssertionError("Expected np.all(y1) to be False")
        if not np.all(y3):
            raise AssertionError("Expected np.all(y3) to be True")
        if np.all(y2):
            raise AssertionError("Expected np.all(y2) to be False")
        if not np.all(~np.array(y2)):
            raise AssertionError("Expected np.all(~np.array(y2)) to be True")

    def test_nd(self):
        # [FUNCTION DOC] Checks all on a nested list overall (false) and along axes zero and
        # one, and the default overall result.
        y1 = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        if np.all(y1):
            raise AssertionError("Expected np.all(y1) to be False")
        assert_equal(np.all(y1, axis=0), [0, 0, 1])
        assert_equal(np.all(y1, axis=1), [0, 0, 1])
        assert_equal(np.all(y1), False)

    def test_method_vs_function(self):
        # [FUNCTION DOC] Checks that the all function and the all method agree on a
        # two-dimensional array.
        y = np.array([[0, 1, 0, 3], [1, 0, 2, 0]])
        assert_equal(np.all(y), y.all())


class TestMean(TestCase):
    # [CLASS DOC] Tests for the mean reduction.
    def test_mean(self):
        # [FUNCTION DOC] Checks mean of a two-by-three nested list overall (3.5) and along
        # each axis, that the mean of an empty list is not-a-number, and that the function
        # matches the method.
        A = [[1, 2, 3], [4, 5, 6]]
        if np.mean(A) != 3.5:
            raise AssertionError(f"Expected mean 3.5, got {np.mean(A)}")
        if not np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])):
            raise AssertionError("Mean along axis 0 does not match expected")
        if not np.all(np.mean(A, 1) == np.array([2.0, 5.0])):
            raise AssertionError("Mean along axis 1 does not match expected")

        # XXX: numpy emits a warning on empty slice
        if not np.isnan(np.mean([])):
            raise AssertionError("Expected mean of empty array to be nan")

        m = np.asarray(A)
        if np.mean(A) != m.mean():
            raise AssertionError(f"Expected {m.mean()}, got {np.mean(A)}")

    def test_mean_values(self):
        # [FUNCTION DOC] For a real and a complex four-by-five matrix and each axis (and no
        # axis), multiplies the mean by the reduced length and compares to the sum, with
        # warnings turned into errors so any warning fails the test.
        # rmat = np.random.random((4, 5))
        rmat = np.arange(20, dtype=float).reshape((4, 5))
        cmat = rmat + 1j * rmat

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for mat in [rmat, cmat]:
                for axis in [0, 1]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.shape[axis]
                    assert_allclose(res, tgt)

                for axis in [None]:
                    tgt = mat.sum(axis=axis)
                    res = np.mean(mat, axis=axis) * mat.size
                    assert_allclose(res, tgt)

    def test_mean_float16(self):
        # [FUNCTION DOC] Checks that the mean of one hundred thousand float16 ones is exactly
        # one, which fails if the inner sum accumulates in half precision.
        # This fail if the sum inside mean is done in float16 instead
        # of float32.
        if np.mean(np.ones(100000, dtype="float16")) != 1:
            raise AssertionError("Expected mean of ones to be 1")

    @xpassIfTorchDynamo_np  # (reason="XXX: mean(..., where=...) not implemented")
    def test_mean_where(self):
        # [FUNCTION DOC] Marked to pass unexpectedly under the TorchDynamo numpy configuration
        # even though mean with a where mask is not implemented there. Checks mean with
        # boolean masks (full and broadcast partial, scalar true, for two- and
        # three-dimensional arrays) against expected values, via both function and method, and
        # that fully masked slices yield not-a-number with a RuntimeWarning.
        a = np.arange(16).reshape((4, 4))
        wh_full = np.array(
            [
                [False, True, False, True],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        wh_partial = np.array([[False], [True], [True], [False]])
        _cases = [
            (1, True, [1.5, 5.5, 9.5, 13.5]),
            (0, wh_full, [6.0, 5.0, 10.0, 9.0]),
            (1, wh_full, [2.0, 5.0, 8.5, 14.5]),
            (0, wh_partial, [6.0, 7.0, 8.0, 9.0]),
        ]
        for _ax, _wh, _res in _cases:
            assert_allclose(a.mean(axis=_ax, where=_wh), np.array(_res))
            assert_allclose(np.mean(a, axis=_ax, where=_wh), np.array(_res))

        a3d = np.arange(16).reshape((2, 2, 4))
        _wh_partial = np.array([False, True, True, False])
        _res = [[1.5, 5.5], [9.5, 13.5]]
        assert_allclose(a3d.mean(axis=2, where=_wh_partial), np.array(_res))
        assert_allclose(np.mean(a3d, axis=2, where=_wh_partial), np.array(_res))

        with pytest.warns(RuntimeWarning):
            assert_allclose(
                a.mean(axis=1, where=wh_partial), np.array([np.nan, 5.5, 9.5, np.nan])
            )
        with pytest.warns(RuntimeWarning):
            assert_equal(a.mean(where=False), np.nan)
        with pytest.warns(RuntimeWarning):
            assert_equal(np.mean(a, where=False), np.nan)


@instantiate_parametrized_tests
class TestSum(TestCase):
    # [CLASS DOC] Tests for the sum reduction, covering dtypes, numeric stability, initial and
    # where.
    def test_sum(self):
        # [FUNCTION DOC] Checks sum with an axis and keepdims on a three-by-three nested list,
        # and that the function matches the method for the overall sum.
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)
        assert_equal(tgt, out)

        am = np.asarray(m)
        assert_equal(np.sum(m), am.sum())

    def test_sum_stability(self):
        # [FUNCTION DOC] Sums five hundred ones divided by ten in float32 and float64 and
        # checks the error against the exact total stays within a small absolute tolerance.
        a = np.ones(500, dtype=np.float32)
        zero = np.zeros(1, dtype="float32")[0]
        assert_allclose((a / 10.0).sum() - a.size / 10.0, zero, atol=1.5e-4)

        a = np.ones(500, dtype=np.float64)
        assert_allclose((a / 10.0).sum() - a.size / 10.0, 0.0, atol=1.5e-13)

    def test_sum_boolean(self):
        # [FUNCTION DOC] Sums a boolean array as an integer count, and with an explicit
        # float64 dtype checks value and result dtype.
        a = np.arange(7) % 2 == 0
        res = a.sum()
        assert_equal(res, 4)

        res_float = a.sum(dtype=np.float64)
        assert_allclose(res_float, 4.0, atol=1e-15)
        if res_float.dtype != "float64":
            raise AssertionError(f"Expected dtype float64, got {res_float.dtype}")

    @skipIf(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @xpassIfTorchDynamo_np  # (reason="sum: does not warn on overflow")
    def test_sum_dtypes_warnings(self):
        # [FUNCTION DOC] Skipped on NumPy older than 1.24; expected to pass unexpectedly under
        # the TorchDynamo configuration. For several dtypes and sizes, records warnings while
        # summing an increasing range (forward and reversed) and expects a warning count that
        # grows only when the exact total overflows the dtype (float16); torch._numpy does not
        # warn on overflow.
        for dt in (int, np.float16, np.float32, np.float64):
            for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
                # warning if sum overflows, which it does in float16
                import warnings

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)

                    tgt = dt(v * (v + 1) / 2)
                    overflow = not np.isfinite(tgt)
                    assert_equal(len(w), 1 * overflow)

                    d = np.arange(1, v + 1, dtype=dt)

                    assert_almost_equal(np.sum(d), tgt)
                    assert_equal(len(w), 2 * overflow)

                    assert_almost_equal(np.sum(np.flip(d)), tgt)
                    assert_equal(len(w), 3 * overflow)

    def test_sum_dtypes_2(self):
        # [FUNCTION DOC] Sums strided slices (steps two and three, forward and reversed) of
        # five hundred ones for several dtypes, and an in-place doubled single element,
        # checking exact totals.
        for dt in (int, np.float16, np.float32, np.float64):
            d = np.ones(500, dtype=dt)
            assert_almost_equal(np.sum(d[::2]), 250.0)
            assert_almost_equal(np.sum(d[1::2]), 250.0)
            assert_almost_equal(np.sum(d[::3]), 167.0)
            assert_almost_equal(np.sum(d[1::3]), 167.0)
            assert_almost_equal(np.sum(np.flip(d)[::2]), 250.0)

            assert_almost_equal(np.sum(np.flip(d)[1::2]), 250.0)

            assert_almost_equal(np.sum(np.flip(d)[::3]), 167.0)
            assert_almost_equal(np.sum(np.flip(d)[1::3]), 167.0)

            # sum with first reduction entry != 0
            d = np.ones((1,), dtype=dt)
            d += d
            assert_almost_equal(d, 2.0)

    @parametrize("dt", [np.complex64, np.complex128])
    def test_sum_complex_1(self, dt):
        # [FUNCTION DOC] For a complex dtype and many sizes, builds an array with real part
        # one to n and imaginary part its negative, and checks the sum and the reversed sum
        # against the expected complex value within tolerance.
        for v in (0, 1, 2, 7, 8, 9, 15, 16, 19, 127, 128, 1024, 1235):
            tgt = dt(v * (v + 1) / 2) - dt((v * (v + 1) / 2) * 1j)
            d = np.empty(v, dtype=dt)
            d.real = np.arange(1, v + 1)
            d.imag = -np.arange(1, v + 1)
            assert_allclose(np.sum(d), tgt, atol=1.5e-5)
            assert_allclose(np.sum(np.flip(d)), tgt, atol=1.5e-7)

    @parametrize("dt", [np.complex64, np.complex128])
    def test_sum_complex_2(self, dt):
        # [FUNCTION DOC] For a complex dtype, checks sums of strided slices (steps two and
        # three, forward and reversed) of an array of five hundred values equal to one plus
        # one imaginary unit, and an in-place doubled single element.
        d = np.ones(500, dtype=dt) + 1j
        assert_allclose(np.sum(d[::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(d[1::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(d[::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(d[1::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[1::2]), 250.0 + 250j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[::3]), 167.0 + 167j, atol=1.5e-7)
        assert_allclose(np.sum(np.flip(d)[1::3]), 167.0 + 167j, atol=1.5e-7)
        # sum with first reduction entry != 0
        d = np.ones((1,), dtype=dt) + 1j
        d += d
        assert_allclose(d, 2.0 + 2j, atol=1.5e-7)

    @xpassIfTorchDynamo_np  # (reason="initial=... need implementing")
    def test_sum_initial(self):
        # [FUNCTION DOC] Marked to pass unexpectedly under the TorchDynamo configuration.
        # Checks the initial keyword of sum for integer, float and multi-axis cases.
        # Integer, single axis
        assert_equal(np.sum([3], initial=2), 5)

        # Floating point
        assert_almost_equal(np.sum([0.2], initial=0.1), 0.3)

        # Multiple non-adjacent axes
        assert_equal(
            np.sum(np.ones((2, 3, 5), dtype=np.int64), axis=(0, 2), initial=2),
            [12, 12, 12],
        )

    @xpassIfTorchDynamo_np  # (reason="where=... need implementing")
    def test_sum_where(self):
        # [FUNCTION DOC] Marked to pass unexpectedly under the TorchDynamo configuration.
        # Checks sum with a where mask, alone and combined with an axis and an initial value.
        # More extensive tests done in test_reduction_with_where.
        assert_equal(np.sum([[1.0, 2.0], [3.0, 4.0]], where=[True, False]), 4.0)
        assert_equal(
            np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0, initial=5.0, where=[True, False]),
            [9.0, 5.0],
        )


parametrize_axis = parametrize(
    "axis", [0, 1, 2, -1, -2, (0, 1), (1, 0), (0, 1, 2), (1, -1, 0)]
)
parametrize_func = parametrize(
    "func",
    [
        np.any,
        np.all,
        np.argmin,
        np.argmax,
        np.min,
        np.max,
        np.mean,
        np.sum,
        np.prod,
        np.std,
        np.var,
        np.count_nonzero,
    ],
)

fails_axes_tuples = {
    np.any,
    np.all,
    np.argmin,
    np.argmax,
    np.prod,
}

fails_out_arg = {
    np.count_nonzero,
}

restricts_dtype_casts = {np.var, np.std}

fails_empty_tuple = {np.argmin, np.argmax}


@instantiate_parametrized_tests
class TestGenericReductions(TestCase):
    # [CLASS DOC] Generic checks of the axis and keepdims arguments, and of the out argument,
    # for each reduction in the parametrized function list.
    # KNOWN ISSUE: the class docstring says to subclass and define a func attribute, but the
    # tests are parametrized over the module-level function list; only the unused helper
    # method relies on that attribute.
    """Run a set of generic tests to verify that self.func acts like a
    reduction operation.

    Specifically, this class checks axis=... and keepdims=... parameters.
    To check the out=... parameter, see the _GenericHasOutTestMixin class below.

    To use: subclass, define self.func and self.allowed_axes.
    """

    @parametrize_func
    def test_bad_axis(self, func):
        # [FUNCTION DOC] Checks that an invalid axis (a string, an out-of-range integer, a 2-D
        # array, or a float) raises TypeError, or AxisError for the out-of-range integer.
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(TypeError, func, m, axis="foo")
        assert_raises(np.AxisError, func, m, axis=3)
        assert_raises(TypeError, func, m, axis=np.array([[1], [2]]))
        assert_raises(TypeError, func, m, axis=1.5)

        # TODO: add tests with np.int32(3) etc, when implemented

    @parametrize_func
    def test_array_axis(self, func):
        # [FUNCTION DOC] Checks that a zero-dimensional array works as the axis (same as the
        # integer) and that a one-dimensional array axis raises TypeError.
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        assert_equal(func(a, axis=np.array(-1)), func(a, axis=-1))

        with assert_raises(TypeError):
            func(a, axis=np.array([1, 2]))

    @parametrize_func
    def test_axis_empty_generic(self, func):
        # [FUNCTION DOC] Skips argmin and argmax. Checks that an empty tuple axis reduces
        # nothing, equal to reducing a newly added leading axis.
        if func in fails_empty_tuple:
            raise SkipTest("func(..., axis=()) is not valid")

        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_array_equal(func(a, axis=()), func(np.expand_dims(a, axis=0), axis=0))

    @parametrize_func
    def test_axis_bad_tuple(self, func):
        # [FUNCTION DOC] Skips functions that do not accept tuple axes. Checks that a tuple
        # with a repeated axis raises ValueError.
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not allow tuple axis.")

        with assert_raises(ValueError):
            func(m, axis=(1, 1))

    @parametrize_axis
    @parametrize_func
    def test_keepdims_generic(self, axis, func):
        # [FUNCTION DOC] Skips functions without tuple axes. On an array of shape two by three
        # by four, checks the result with keepdims equals the result without it with the
        # reduced axes re-inserted.
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not allow tuple axis.")

        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        with_keepdims = func(a, axis, keepdims=True)
        expanded = np.expand_dims(func(a, axis=axis), axis=axis)
        assert_array_equal(with_keepdims, expanded)

    @skipIf(numpy.__version__ < "1.24", reason="NP_VER: fails on CI w/old numpy")
    @parametrize_func
    def test_keepdims_generic_axis_none(self, func):
        # [FUNCTION DOC] Skipped on NumPy older than 1.24. Checks that reducing over all axes
        # with keepdims gives an array of ones in shape filled with the scalar result.
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        with_keepdims = func(a, axis=None, keepdims=True)
        scalar = func(a, axis=None)
        expanded = np.full((1,) * a.ndim, fill_value=scalar)
        assert_array_equal(with_keepdims, expanded)

    @parametrize_func
    def test_out_scalar(self, func):
        # [FUNCTION DOC] Skips count_nonzero (no out argument). Checks a full reduction with a
        # preallocated output returns the output object and equals the plain result.
        # out no axis: scalar
        if func in fails_out_arg:
            raise SkipTest(f"{func.__name__} does not have out= arg.")

        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))

        result = func(a)
        out = np.empty_like(result)
        result_with_out = func(a, out=out)

        if result_with_out is not out:
            raise AssertionError("Expected result_with_out to be out")
        assert_array_equal(result, result_with_out)

    def _check_out_axis(self, axis, dtype, keepdims):
        # [FUNCTION DOC] HELPER: Out-argument check for a given axis, dtype and keepdims using
        # the func attribute of the instance: asserts the output object is returned, has the
        # dtype, and equals the result.
        # KNOWN ISSUE: never called, and the class defines no func attribute, so calling it
        # would fail.
        # out with axis
        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        result = self.func(a, axis=axis, keepdims=keepdims).astype(dtype)

        out = np.empty_like(result, dtype=dtype)
        result_with_out = self.func(a, axis=axis, keepdims=keepdims, out=out)

        if result_with_out is not out:
            raise AssertionError("Expected result_with_out to be out")
        if result_with_out.dtype != dtype:
            raise AssertionError(f"Expected dtype {dtype}, got {result_with_out.dtype}")
        assert_array_equal(result, result_with_out)

        # TODO: what if result.dtype != out.dtype; does out typecast the result?

        # out of wrong shape (any/out does not broadcast)
        # np.any(m, out=np.empty_like(m)) raises a ValueError (wrong number
        # of dimensions.)
        # pytorch.any emits a warning and resizes the out array.
        # Here we follow pytorch, since the result is a superset
        # of the numpy functionality

    @parametrize("keepdims", [True, False])
    @parametrize("dtype", [bool, "int32", "float64"])
    @parametrize_func
    @parametrize_axis
    def test_out_axis(self, func, axis, dtype, keepdims):
        # [FUNCTION DOC] Skips count_nonzero, tuple-axis functions and var/std (the test casts
        # float results to integer dtypes). Computes the reduction cast to the dtype, then
        # repeats with an output of that dtype and asserts the output object is returned with
        # the right dtype and equal values. Parametrized over function, axis, dtype and
        # keepdims.
        # out with axis
        if func in fails_out_arg:
            raise SkipTest(f"{func.__name__} does not have out= arg.")
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not hangle tuple axis.")
        if func in restricts_dtype_casts:
            raise SkipTest(f"{func.__name__}: test implies float->int casts")

        a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        result = func(a, axis=axis, keepdims=keepdims).astype(dtype)

        out = np.empty_like(result, dtype=dtype)
        result_with_out = func(a, axis=axis, keepdims=keepdims, out=out)

        if result_with_out is not out:
            raise AssertionError("Expected result_with_out to be out")
        if result_with_out.dtype != dtype:
            raise AssertionError(f"Expected dtype {dtype}, got {result_with_out.dtype}")
        assert_array_equal(result, result_with_out)

        # TODO: what if result.dtype != out.dtype; does out typecast the result?

        # out of wrong shape (any/out does not broadcast)
        # np.any(m, out=np.empty_like(m)) raises a ValueError (wrong number
        # of dimensions.)
        # pytorch.any emits a warning and resizes the out array.
        # Here we follow pytorch, since the result is a superset
        # of the numpy functionality

    @parametrize_func
    @parametrize_axis
    def test_keepdims_out(self, func, axis):
        # [FUNCTION DOC] Skips count_nonzero and tuple-axis functions. Builds an output array
        # of the keepdims shape for a four-dimensional array of ones (normalizing the axis
        # with the util helper) and checks the reduction with keepdims and out returns the
        # output with that shape.
        if func in fails_out_arg:
            raise SkipTest(f"{func.__name__} does not have out= arg.")
        if func in fails_axes_tuples:
            raise SkipTest(f"{func.__name__} does not hangle tuple axis.")

        d = np.ones((3, 5, 7, 11))
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = _util.normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim)
            )
        out = np.empty(shape_out)

        result = func(d, axis=axis, keepdims=True, out=out)
        if result is not out:
            raise AssertionError("Expected result to be out")
        assert_equal(result.shape, shape_out)


@instantiate_parametrized_tests
class TestGenericCumSumProd(TestCase):
    # [CLASS DOC] Generic tests for cumsum and cumprod.
    """Run a set of generic tests to verify that cumsum/cumprod are sane."""

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_bad_axis(self, func):
        # [FUNCTION DOC] Checks invalid axis values (string, out-of-range int, 2-D array,
        # float) raise TypeError, or AxisError for the out-of-range integer.
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(TypeError, func, m, axis="foo")
        assert_raises(np.AxisError, func, m, axis=3)
        assert_raises(TypeError, func, m, axis=np.array([[1], [2]]))
        assert_raises(TypeError, func, m, axis=1.5)

        # TODO: add tests with np.int32(3) etc, when implemented

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_array_axis(self, func):
        # [FUNCTION DOC] Checks a zero-dimensional array axis equals the integer axis, and a
        # one-dimensional array axis raises TypeError.
        a = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        assert_equal(func(a, axis=np.array(-1)), func(a, axis=-1))

        with assert_raises(TypeError):
            func(a, axis=np.array([1, 2]))

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_axis_empty_generic(self, func):
        # [FUNCTION DOC] Checks that axis none gives the same result as the operation on the
        # flattened input along axis zero.
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_array_equal(func(a, axis=None), func(a.ravel(), axis=0))

    @parametrize("func", [np.cumsum, np.cumprod])
    def test_axis_bad_tuple(self, func):
        # [FUNCTION DOC] Checks that a tuple axis raises TypeError for the cumulative
        # operations.
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        with assert_raises(TypeError):
            func(m, axis=(1, 1))


if __name__ == "__main__":
    run_tests()
