# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for histogram, histogram_bin_edges and histogramdd in the NumPy-
# compatible layer. Contains a class for 1-D histograms (ranges, density, weights, dtypes,
# error cases, overflow and precision checks), a class for automatic bin-count estimators, and
# a class for N-dimensional histograms. Real NumPy is used under dynamo testing, otherwise
# torch._numpy.
# KNOWN ISSUE: histogram_bin_edges is imported only in the dynamo branch, so tests using that
# bare name can only run as expected failures outside dynamo. Also the module defines the skip
# helper before importing the testing utilities, which is only an import-order quirk.

import functools
from unittest import expectedFailure as xfail, skipIf

from pytest import raises as assert_raises


skip = functools.partial(skipIf, True)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest as slow,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import histogram, histogram_bin_edges, histogramdd
    from numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
    )
else:
    import torch._numpy as np
    from torch._numpy import histogram, histogramdd
    from torch._numpy.testing import (
        assert_,
        assert_allclose,
        assert_almost_equal,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
    )


class TestHistogram(TestCase):
    # [CLASS DOC] Tests for 1-D histogram: counts, edges, density, weights, outliers and
    # invalid input.
    def test_simple(self):
        # [FUNCTION DOC] Checks that bin counts of 100 random samples sum to 100, and that
        # evenly spaced data gives equal counts of 10 in each default bin.
        n = 100
        v = np.random.rand(n)
        (a, b) = histogram(v)
        # check if the sum of the bins equals the number of samples
        assert_equal(np.sum(a, axis=0), n)
        # check that the bin counts are evenly spaced when the data is from
        # a linear function
        (a, b) = histogram(np.linspace(0, 10, 100))
        assert_array_equal(a, 10)

    def test_one_bin(self):
        # [FUNCTION DOC] Regression test (ticket 632): a single bin from explicit edges counts
        # all inside values and returns the edges unchanged; zero bins raises RuntimeError or
        # ValueError; one bin from an integer count spans the data range.
        # Ticket 632
        hist, edges = histogram([1, 2, 3, 4], [1, 2])
        assert_array_equal(
            hist,
            [
                2,
            ],
        )
        assert_array_equal(edges, [1, 2])
        assert_raises((RuntimeError, ValueError), histogram, [1, 2], bins=0)
        h, e = histogram([1, 2], bins=1)
        assert_equal(h, np.array([2]))
        assert_allclose(e, np.array([1.0, 2.0]))

    def test_density(self):
        # [FUNCTION DOC] Checks density histograms integrate to one (uniform and non-constant
        # bin widths, infinite last edge, and a numpy-discussion bug-report case), and that
        # density=False gives counts.
        # Check that the integral of the density equals 1.
        n = 100
        v = np.random.rand(n)
        a, b = histogram(v, density=True)
        area = np.sum(a * np.diff(b))
        assert_almost_equal(area, 1)

        # Check with non-constant bin widths
        v = np.arange(10)
        bins = [0, 1, 3, 6, 10]
        a, b = histogram(v, bins, density=True)
        assert_almost_equal(a, 0.1)
        assert_equal(np.sum(a * np.diff(b)), 1)

        # Test that passing False works too
        a, b = histogram(v, bins, density=False)
        assert_array_equal(a, [1, 2, 3, 4])

        # Variable bin widths are especially useful to deal with
        # infinities.
        v = np.arange(10)
        bins = [0, 1, 3, 6, np.inf]
        a, b = histogram(v, bins, density=True)
        assert_almost_equal(a, [0.1, 0.1, 0.1, 0.0])

        # Taken from a bug report from N. Becker on the numpy-discussion
        # mailing list Aug. 6, 2010.
        counts, dmy = np.histogram([1, 2, 3, 4], [0.5, 1.5, np.inf], density=True)
        assert_equal(counts, [0.25, 0])

    def test_outliers(self):
        # [FUNCTION DOC] Checks values outside the range are not counted (lower and upper),
        # that density normalisation and weights integrate to one, and that weighted counts
        # equal the inner weights.
        # Check that outliers are not tallied
        a = np.arange(10) + 0.5

        # Lower outliers
        h, b = histogram(a, range=[0, 9])
        assert_equal(h.sum(), 9)

        # Upper outliers
        h, b = histogram(a, range=[1, 10])
        assert_equal(h.sum(), 9)

        # Normalization
        h, b = histogram(a, range=[1, 9], density=True)
        assert_almost_equal((h * np.diff(b)).sum(), 1, decimal=15)

        # Weights
        w = np.arange(10) + 0.5
        h, b = histogram(a, range=[1, 9], weights=w, density=True)
        assert_equal((h * np.diff(b)).sum(), 1)

        h, b = histogram(a, bins=8, range=[1, 9], weights=w)
        assert_equal(h, w[1:-1])

    def test_arr_weights_mismatch(self):
        # [FUNCTION DOC] Checks weights of a different length than the data raise RuntimeError
        # or ValueError.
        a = np.arange(10) + 0.5
        w = np.arange(11) + 0.5
        with assert_raises((RuntimeError, ValueError)):  # , "same shape as"):
            h, b = histogram(a, range=[1, 9], weights=w, density=True)

    def test_type(self):
        # [FUNCTION DOC] Checks the dtype category of the counts: integer by default and with
        # integer weights, floating with density or float weights.
        # Check the type of the returned histogram
        a = np.arange(10) + 0.5
        h, b = histogram(a)
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, density=True)
        assert_(np.issubdtype(h.dtype, np.floating))

        h, b = histogram(a, weights=np.ones(10, int))
        assert_(np.issubdtype(h.dtype, np.integer))

        h, b = histogram(a, weights=np.ones(10, float))
        assert_(np.issubdtype(h.dtype, np.floating))

    def test_f32_rounding(self):
        # [FUNCTION DOC] Regression test (gh-4799): a 2-D histogram of float32 data with 100
        # bins counts all three points (sum is three).
        # gh-4799, check that the rounding of the edges works with float32
        x = np.array([276.318359, -69.593948, 21.329449], dtype=np.float32)
        y = np.array([5005.689453, 4481.327637, 6010.369629], dtype=np.float32)
        counts_hist, xedges, yedges = np.histogram2d(x, y, bins=100)
        assert_equal(counts_hist.sum(), 3.0)

    def test_bool_conversion(self):
        # [FUNCTION DOC] Regression test (gh-12107): a histogram of boolean data equals the
        # histogram of the same data as small integers, both counts and edges. The warning
        # check is commented out.
        # gh-12107
        # Reference integer histogram
        a = np.array([1, 1, 0], dtype=np.uint8)
        int_hist, int_edges = np.histogram(a)

        # Should raise an warning on booleans
        # Ensure that the histograms are equivalent, need to suppress
        # the warnings to get the actual outputs
        #      with suppress_warnings() as sup:
        #          rec = sup.record(RuntimeWarning, 'Converting input from .*')
        hist, edges = np.histogram([True, True, False])
        # A warning should be issued
        #            assert_equal(len(rec), 1)
        assert_array_equal(hist, int_hist)
        assert_array_equal(edges, int_edges)

    def test_weights(self):
        # [FUNCTION DOC] Checks weights: constant weights scale counts by the weight (and
        # leave density unchanged), zero/one weights select values, integer weights sum per
        # bin, and non-uniform bin widths with density give the expected normalised values.
        v = np.random.rand(100)
        w = np.ones(100) * 5
        a, b = histogram(v)
        na, nb = histogram(v, density=True)
        wa, wb = histogram(v, weights=w)
        nwa, nwb = histogram(v, weights=w, density=True)
        assert_array_almost_equal(a * 5, wa)
        assert_array_almost_equal(na, nwa)

        # Check weights are properly applied.
        v = np.linspace(0, 10, 10)
        w = np.concatenate((np.zeros(5), np.ones(5)))
        wa, wb = histogram(v, bins=np.arange(11), weights=w)
        assert_array_almost_equal(wa, w)

        # Check with integer weights
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1])
        assert_array_equal(wa, [4, 5, 0, 1])
        wa, wb = histogram([1, 2, 2, 4], bins=4, weights=[4, 3, 2, 1], density=True)
        assert_array_almost_equal(wa, np.array([4, 5, 0, 1]) / 10.0 / 3.0 * 4)

        # Check weights with non-uniform bin widths
        a, b = histogram(
            np.arange(9),
            [0, 1, 3, 6, 10],
            weights=[2, 1, 1, 1, 1, 1, 1, 1, 1],
            density=True,
        )
        assert_almost_equal(a, [0.2, 0.1, 0.1, 0.075])

    @xpassIfTorchDynamo_np  # (reason="histogram complex weights")
    def test_exotic_weights(self):
        # [FUNCTION DOC] Checks complex and Decimal weights give per-bin sums for custom and
        # even bins. Marked an expected failure outside dynamo because such weights are
        # unsupported in torch.
        # Test the use of weights that are not integer or floats, but e.g.
        # complex numbers or object types.

        # Complex weights
        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([1, -1, 2]) + 1j * np.array([2, 1, 2])

        # Check with custom bins
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # Check with even bins
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, np.array([1, 1]) + 1j * np.array([2, 3]))

        # Decimal weights
        from decimal import Decimal

        values = np.array([1.3, 2.5, 2.3])
        weights = np.array([Decimal(1), Decimal(2), Decimal(3)])

        # Check with custom bins
        wa, wb = histogram(values, bins=[0, 2, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

        # Check with even bins
        wa, wb = histogram(values, bins=2, range=[1, 3], weights=weights)
        assert_array_almost_equal(wa, [Decimal(1), Decimal(5)])

    def test_no_side_effects(self):
        # [FUNCTION DOC] Regression test: calling histogram does not modify its input values.
        # This is a regression test that ensures that values passed to
        # ``histogram`` are unchanged.
        values = np.array([1.3, 2.5, 2.3])
        np.histogram(values, range=[-10, 10], bins=100)
        assert_array_almost_equal(values, [1.3, 2.5, 2.3])

    def test_empty(self):
        # [FUNCTION DOC] Checks an empty input with two edges gives a single zero count and
        # those edges.
        a, b = histogram([], bins=([0, 1]))
        assert_array_equal(a, np.array([0]))
        assert_array_equal(b, np.array([0, 1]))

    def test_error_binnum_type(self):
        # [FUNCTION DOC] Checks an integer bin count works and a float bin count raises
        # TypeError.
        # Tests if right Error is raised if bins argument is float
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, 5)
        assert_raises(TypeError, histogram, vals, 2.4)

    def test_finite_range(self):
        # [FUNCTION DOC] Checks a finite range is accepted while a range containing NaN or
        # infinity raises RuntimeError or ValueError.
        # Normal ranges should be fine
        vals = np.linspace(0.0, 1.0, num=100)
        histogram(vals, range=[0.25, 0.75])
        assert_raises((RuntimeError, ValueError), histogram, vals, range=[np.nan, 0.75])
        assert_raises((RuntimeError, ValueError), histogram, vals, range=[0.25, np.inf])

    def test_invalid_range(self):
        # [FUNCTION DOC] Checks a range whose start exceeds its end raises RuntimeError or
        # ValueError.
        # start of range must be < end of range
        vals = np.linspace(0.0, 1.0, num=100)
        with assert_raises((RuntimeError, ValueError)):
            np.histogram(vals, range=[0.1, 0.01])

    @xfail  # (reason="edge cases")
    def test_bin_edge_cases(self):
        # [FUNCTION DOC] Checks that with many bins each sample falls in a bin whose left edge
        # is not above it and whose right edge is above it (floating-point edge placement).
        # Marked an expected failure, since edge cases are not handled exactly.
        # Ensure that floating-point computations correctly place edge cases.
        arr = np.array([337, 404, 739, 806, 1007, 1811, 2012])
        hist, edges = np.histogram(arr, bins=8296, range=(2, 2280))
        mask = hist > 0
        left_edges = edges[:-1][mask]
        right_edges = edges[1:][mask]
        for x, left, right in zip(arr, left_edges, right_edges):
            assert_(x >= left)
            assert_(x < right, msg=f"{x}, {right}")

    def test_last_bin_inclusive_range(self):
        # [FUNCTION DOC] Checks the last bin includes its right edge: the sample equal to five
        # is counted in the final bin when the range ends there.
        arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        hist, edges = np.histogram(arr, bins=30, range=(-0.5, 5))
        assert_equal(hist[-1], 1)

    def test_bin_array_dims(self):
        # [FUNCTION DOC] Checks a 2-D bins array raises RuntimeError or ValueError.
        # gracefully handle bins object > 1 dimension
        vals = np.linspace(0.0, 1.0, num=100)
        bins = np.array([[0, 0.5], [0.6, 1.0]])
        with assert_raises((RuntimeError, ValueError)):
            np.histogram(vals, bins=bins)

    @xpassIfTorchDynamo_np  # (reason="no uint64")
    def test_unsigned_monotonicity_check(self):
        # [FUNCTION DOC] Regression test (9222): non-monotonic unsigned bin edges raise
        # RuntimeError or ValueError. Marked an expected failure outside dynamo (no 64-bit
        # unsigned dtype).
        # Ensures ValueError is raised if bins not increasing monotonically
        # when bins contain unsigned values (see #9222)
        arr = np.array([2])
        bins = np.array([1, 3, 1], dtype="uint64")
        with assert_raises((RuntimeError, ValueError)):
            hist, edges = np.histogram(arr, bins=bins)

    def test_object_array_of_0d(self):
        # [FUNCTION DOC] Regression test (gh-7864): lists of 0-d arrays with an infinite value
        # raise RuntimeError or ValueError, while nearly-constant lists of 0-d arrays do not
        # crash.
        # gh-7864
        assert_raises(
            (RuntimeError, ValueError),
            histogram,
            [np.array(0.4) for i in range(10)] + [-np.inf],
        )
        assert_raises(
            (RuntimeError, ValueError),
            histogram,
            [np.array(0.4) for i in range(10)] + [np.inf],
        )

        # these should not crash
        np.histogram([np.array(0.5) for i in range(10)] + [0.500000000000002])
        np.histogram([np.array(0.5) for i in range(10)] + [0.5])

    @xpassIfTorchDynamo_np  # (reason="bins='auto'")
    def test_some_nan_values(self):
        # [FUNCTION DOC] Regression test (gh-7503): NaN values with automatic bins raise
        # ValueError, but an explicit range or explicit bin edges make histogram work and NaNs
        # are not counted. Marked an expected failure outside dynamo (string bins
        # unsupported).
        # gh-7503
        one_nan = np.array([0, 1, np.nan])
        all_nan = np.array([np.nan, np.nan])

        # the internal comparisons with NaN give warnings
        #      sup = suppress_warnings()
        #      sup.filter(RuntimeWarning)
        #      with sup:
        # can't infer range with nan
        assert_raises(ValueError, histogram, one_nan, bins="auto")
        assert_raises(ValueError, histogram, all_nan, bins="auto")

        # explicit range solves the problem
        h, b = histogram(one_nan, bins="auto", range=(0, 1))
        assert_equal(h.sum(), 2)  # nan is not counted
        h, b = histogram(all_nan, bins="auto", range=(0, 1))
        assert_equal(h.sum(), 0)  # nan is not counted

        # as does an explicit set of bins
        h, b = histogram(one_nan, bins=[0, 1])
        assert_equal(h.sum(), 2)  # nan is not counted
        h, b = histogram(all_nan, bins=[0, 1])
        assert_equal(h.sum(), 0)  # nan is not counted

    def do_signed_overflow_bounds(self, dtype):
        # [FUNCTION DOC] HELPER: For the given signed integer dtype, histograms two values
        # near the minimum and maximum into two bins and checks the edges and the one-count-
        # per-bin result (no overflow).
        exponent = 8 * np.dtype(dtype).itemsize - 1
        arr = np.array([-(2**exponent) + 4, 2**exponent - 4], dtype=dtype)
        hist, e = histogram(arr, bins=2)
        assert_equal(e, [-(2**exponent) + 4, 0, 2**exponent - 4])
        assert_equal(hist, [1, 1])

    def test_signed_overflow_bounds(self):
        # [FUNCTION DOC] Runs the overflow-bounds helper for byte, short and intc dtypes.
        self.do_signed_overflow_bounds(np.byte)
        self.do_signed_overflow_bounds(np.short)
        self.do_signed_overflow_bounds(np.intc)

    @xfail  # (reason="int->float conversion loses precision")
    def test_signed_overflow_bounds_2(self):
        # [FUNCTION DOC] Runs the overflow-bounds helper for the int_ and longlong dtypes.
        # Marked an expected failure because integer to float conversion loses precision.
        self.do_signed_overflow_bounds(np.int_)
        self.do_signed_overflow_bounds(np.longlong)

    def do_precision_lower_bound(self, float_small, float_large):
        # [FUNCTION DOC] HELPER: With one value in the smaller float type and a range in the
        # larger type that starts just above one, checks histogram with one bin counts it and
        # returns edges of the smaller type. Returns early if the range would round to exactly
        # one in the smaller type.
        eps = np.finfo(float_large).eps

        arr = np.array([1.0], float_small)
        range = np.array([1.0 + eps, 2.0], float_large)

        # test is looking for behavior when the bounds change between dtypes
        if range.astype(float_small)[0] != 1:
            return

        # previously crashed
        count, x_loc = np.histogram(arr, bins=1, range=range)
        assert_equal(count, [1])

        # gh-10322 means that the type comes from arr - this may change
        assert_equal(x_loc.dtype, float_small)

    def do_precision_upper_bound(self, float_small, float_large):
        # [FUNCTION DOC] HELPER: Like the lower-bound helper but with a range ending just
        # below one: checks the single count and the edge dtype, returning early when the
        # bound rounds to one.
        eps = np.finfo(float_large).eps

        arr = np.array([1.0], float_small)
        range = np.array([0.0, 1.0 - eps], float_large)

        # test is looking for behavior when the bounds change between dtypes
        if range.astype(float_small)[-1] != 1:
            return

        # previously crashed
        count, x_loc = np.histogram(arr, bins=1, range=range)
        assert_equal(count, [1])

        # gh-10322 means that the type comes from arr - this may change
        assert_equal(x_loc.dtype, float_small)

    def do_precision(self, float_small, float_large):
        # [FUNCTION DOC] HELPER: Runs both the lower-bound and upper-bound precision helpers.
        self.do_precision_lower_bound(float_small, float_large)
        self.do_precision_upper_bound(float_small, float_large)

    @xpassIfTorchDynamo_np  # (reason="mixed dtypes")
    def test_precision(self):
        # [FUNCTION DOC] Runs the precision helper for half/single, half/double and
        # single/double pairs. Marked an expected failure outside dynamo (mixed dtypes
        # unsupported).
        # not looping results in a useful stack trace upon failure
        self.do_precision(np.half, np.single)
        self.do_precision(np.half, np.double)
        self.do_precision(np.single, np.double)

    @xpassIfTorchDynamo_np  # (reason="histogram_bin_edges")
    def test_histogram_bin_edges(self):
        # [FUNCTION DOC] Checks histogram_bin_edges agrees with the edges returned by
        # histogram for explicit edges, an integer bin count with range, and automatic bins.
        # Marked an expected failure outside dynamo.
        hist, e = histogram([1, 2, 3, 4], [1, 2])
        edges = histogram_bin_edges([1, 2, 3, 4], [1, 2])
        assert_allclose(edges, e, atol=2e-15)

        arr = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        hist, e = histogram(arr, bins=30, range=(-0.5, 5))
        edges = histogram_bin_edges(arr, bins=30, range=(-0.5, 5))
        assert_allclose(edges, e, atol=2e-15)

        hist, e = histogram(arr, bins="auto", range=(0, 1))
        edges = histogram_bin_edges(arr, bins="auto", range=(0, 1))
        assert_allclose(edges, e, atol=2e-15)

    # @requires_memory(free_bytes=1e10)
    @xpassIfTorchDynamo_np  # (reason="pytorch does not support bins = [int, int, array]")
    @slow
    def test_big_arrays(self):
        # [FUNCTION DOC] Builds a 100-million by 3 sample of zeros and calls the N-dimensional
        # histogram with mixed integer and array bins, checking a tuple is returned. Marked
        # slow and an expected failure outside dynamo (mixed bin types unsupported); needs a
        # lot of memory.
        sample = np.zeros([100000000, 3])
        xbins = 400
        ybins = 400
        zbins = np.arange(16000)
        hist = np.histogramdd(sample=sample, bins=(xbins, ybins, zbins))
        assert_equal(type(hist), type((1, 2)))


@xpassIfTorchDynamo_np  # (reason="TODO")
@instantiate_parametrized_tests
class TestHistogramOptimBinNums(TestCase):
    # [CLASS DOC] Tests for the string bin estimators (fd, scott, rice, sturges, doane, sqrt,
    # auto, stone). The class is marked an expected failure outside dynamo (estimators not
    # implemented).
    """
    Provide test coverage when using provided estimators for optimal number of
    bins
    """

    def test_empty(self):
        # [FUNCTION DOC] Checks every estimator on empty data gives one zero count and edges
        # zero and one.
        estimator_list = [
            "fd",
            "scott",
            "rice",
            "sturges",
            "doane",
            "sqrt",
            "auto",
            "stone",
        ]
        # check it can deal with empty data
        for estimator in estimator_list:
            a, b = histogram([], bins=estimator)
            assert_array_equal(a, np.array([0]))
            assert_array_equal(b, np.array([0, 1]))

    def test_simple(self):
        # [FUNCTION DOC] For three data sizes of a two-peak linspace mixture, checks the
        # number of bins each estimator picks against precomputed values.
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). All test values have been precomputed and the values
        shouldn't change
        """
        # Some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        basic_test = {
            50: {
                "fd": 4,
                "scott": 4,
                "rice": 8,
                "sturges": 7,
                "doane": 8,
                "sqrt": 8,
                "auto": 7,
                "stone": 2,
            },
            500: {
                "fd": 8,
                "scott": 8,
                "rice": 16,
                "sturges": 10,
                "doane": 12,
                "sqrt": 23,
                "auto": 10,
                "stone": 9,
            },
            5000: {
                "fd": 17,
                "scott": 17,
                "rice": 35,
                "sturges": 14,
                "doane": 17,
                "sqrt": 71,
                "auto": 17,
                "stone": 20,
            },
        }

        for testlen, expectedResults in basic_test.items():
            # Create some sort of non uniform data to test with
            # (2 peak uniform mixture)
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x = np.concatenate((x1, x2))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator)
                assert_equal(
                    len(a),
                    numbins,
                    err_msg=f"For the {estimator} estimator with datasize of {testlen}",
                )

    def test_small(self):
        # [FUNCTION DOC] For data sizes one to three, checks the precomputed number of bins
        # each estimator chooses (estimators can misbehave on small data, especially fd).
        """
        Smaller datasets have the potential to cause issues with the data
        adaptive methods, especially the FD method. All bin numbers have been
        precalculated.
        """
        small_dat = {
            1: {
                "fd": 1,
                "scott": 1,
                "rice": 1,
                "sturges": 1,
                "doane": 1,
                "sqrt": 1,
                "stone": 1,
            },
            2: {
                "fd": 2,
                "scott": 1,
                "rice": 3,
                "sturges": 2,
                "doane": 1,
                "sqrt": 2,
                "stone": 1,
            },
            3: {
                "fd": 2,
                "scott": 2,
                "rice": 3,
                "sturges": 3,
                "doane": 3,
                "sqrt": 2,
                "stone": 1,
            },
        }

        for testlen, expectedResults in small_dat.items():
            testdat = np.arange(testlen)
            for estimator, expbins in expectedResults.items():
                a, b = np.histogram(testdat, estimator)
                assert_equal(
                    len(a),
                    expbins,
                    err_msg=f"For the {estimator} estimator with datasize of {testlen}",
                )

    def test_incorrect_methods(self):
        # [FUNCTION DOC] Checks unknown estimator names raise ValueError.
        """
        Check a Value Error is thrown when an unknown string is passed in
        """
        check_list = ["mad", "freeman", "histograms", "IQR"]
        for estimator in check_list:
            assert_raises(ValueError, histogram, [1, 2, 3], estimator)

    def test_novariance(self):
        # [FUNCTION DOC] Checks all estimators give one bin for constant data (zero spread).
        """
        Check that methods handle no variance in data
        Primarily for Scott and FD as the SD and IQR are both 0 in this case
        """
        novar_dataset = np.ones(100)
        novar_resultdict = {
            "fd": 1,
            "scott": 1,
            "rice": 1,
            "sturges": 1,
            "doane": 1,
            "sqrt": 1,
            "auto": 1,
            "stone": 1,
        }

        for estimator, numbins in novar_resultdict.items():
            a, b = np.histogram(novar_dataset, estimator)
            assert_equal(
                len(a),
                numbins,
                err_msg=f"{estimator} estimator, No Variance test",
            )

    def test_limited_variance(self):
        # [FUNCTION DOC] Checks that when the interquartile range is zero but the variance is
        # not, auto and sturges give twelve-point edges while fd gives a single bin from zero
        # to 100.
        """
        Check when IQR is 0, but variance exists, we return the sturges value
        and not the fd value.
        """
        lim_var_data = np.ones(1000)
        lim_var_data[:3] = 0
        lim_var_data[-4:] = 100

        edges_auto = histogram_bin_edges(lim_var_data, "auto")
        # assert_equal(edges_auto, np.linspace(0, 100, 12))
        assert_allclose(edges_auto, np.linspace(0, 100, 12), rtol=1e-15)

        edges_fd = histogram_bin_edges(lim_var_data, "fd")
        # assert_equal(edges_fd, np.array([0, 100]))
        assert_allclose(edges_fd, np.array([0, 100]), rtol=1e-15)

        edges_sturges = histogram_bin_edges(lim_var_data, "sturges")
        # assert_equal(edges_sturges, np.linspace(0, 100, 12))
        assert_allclose(edges_sturges, np.linspace(0, 100, 12), rtol=1e-15)

    def test_outlier(self):
        # [FUNCTION DOC] Checks fd, scott, doane and stone bin counts on data with a far
        # outlier cluster against precomputed values.
        """
        Check the FD, Scott and Doane with outliers.

        The FD estimates a smaller binwidth since it's less affected by
        outliers. Since the range is so (artificially) large, this means more
        bins, most of which will be empty, but the data of interest usually is
        unaffected. The Scott estimator is more affected and returns fewer bins,
        despite most of the variance being in one area of the data. The Doane
        estimator lies somewhere between the other two.
        """
        xcenter = np.linspace(-10, 10, 50)
        outlier_dataset = np.hstack((np.linspace(-110, -100, 5), xcenter))

        outlier_resultdict = {"fd": 21, "scott": 5, "doane": 11, "stone": 6}

        for estimator, numbins in outlier_resultdict.items():
            a, b = np.histogram(outlier_dataset, estimator)
            assert_equal(len(a), numbins)

    def test_scott_vs_stone(self):
        # [FUNCTION DOC] Checks that for normal samples of growing size, the average gap from
        # half between stone and scott bin counts shrinks, against precomputed values to two
        # decimals.
        """Verify that Scott's rule and Stone's rule converges for normally distributed data"""

        def nbins_ratio(seed, size):
            # [FUNCTION DOC] HELPER: For a seed and size, draws normal samples and returns the
            # stone bin count divided by the sum of the stone and scott bin counts.
            rng = np.random.RandomState(seed)
            x = rng.normal(loc=0, scale=2, size=size)
            a, b = len(np.histogram(x, "stone")[0]), len(np.histogram(x, "scott")[0])
            return a / (a + b)

        ll = [
            [
                nbins_ratio(seed, size)
                for size in np.geomspace(start=10, stop=100, num=4).round().astype(int)
            ]
            for seed in range(10)
        ]

        # the average difference between the two methods decreases as the dataset size increases.
        avg = abs(np.mean(ll, axis=0) - 0.5)
        assert_almost_equal(avg, [0.15, 0.09, 0.08, 0.03], decimal=2)

    def test_simple_range(self):
        # [FUNCTION DOC] Like the simple test but with an extra mixture component outside an
        # explicit range of minus 20 to 20, which must be ignored; checks precomputed bin
        # counts.
        """
        Straightforward testing with a mixture of linspace data (for
        consistency). Adding in a 3rd mixture that will then be
        completely ignored. All test values have been precomputed and
        the shouldn't change.
        """
        # some basic sanity checking, with some fixed data.
        # Checking for the correct number of bins
        basic_test = {
            50: {
                "fd": 8,
                "scott": 8,
                "rice": 15,
                "sturges": 14,
                "auto": 14,
                "stone": 8,
            },
            500: {
                "fd": 15,
                "scott": 16,
                "rice": 32,
                "sturges": 20,
                "auto": 20,
                "stone": 80,
            },
            5000: {
                "fd": 33,
                "scott": 33,
                "rice": 69,
                "sturges": 27,
                "auto": 33,
                "stone": 80,
            },
        }

        for testlen, expectedResults in basic_test.items():
            # create some sort of non uniform data to test with
            # (3 peak uniform mixture)
            x1 = np.linspace(-10, -1, testlen // 5 * 2)
            x2 = np.linspace(1, 10, testlen // 5 * 3)
            x3 = np.linspace(-100, -50, testlen)
            x = np.hstack((x1, x2, x3))
            for estimator, numbins in expectedResults.items():
                a, b = np.histogram(x, estimator, range=(-20, 20))
                msg = f"For the {estimator} estimator"
                msg += f" with datasize of {testlen}"
                assert_equal(len(a), numbins, err_msg=msg)

    @parametrize("bins", ["auto", "fd", "doane", "scott", "stone", "rice", "sturges"])
    def test_signed_integer_data(self, bins):
        # [FUNCTION DOC] Parametrized over estimators. Regression test (gh-14379): int8 data
        # gives the same histogram and edges as the same data as int32.
        # Regression test for gh-14379.
        a = np.array([-2, 0, 127], dtype=np.int8)
        hist, edges = np.histogram(a, bins=bins)
        hist32, edges32 = np.histogram(a.astype(np.int32), bins=bins)
        assert_array_equal(hist, hist32)
        assert_array_equal(edges, edges32)

    def test_simple_weighted(self):
        # [FUNCTION DOC] Checks that using an estimator with weights raises TypeError.
        """
        Check that weighted data raises a TypeError
        """
        estimator_list = ["fd", "scott", "rice", "sturges", "auto"]
        for estimator in estimator_list:
            assert_raises(TypeError, histogram, [1, 2, 3], estimator, weights=[1, 2, 3])


class TestHistogramdd(TestCase):
    # [CLASS DOC] Tests for histogramdd (N-dimensional histogram).
    def test_simple(self):
        # [FUNCTION DOC] Checks counts for 3-D points with explicit bins and ranges,
        # normalisation by total volume, resulting shape, a sequence of 1-D arrays as input,
        # and an integer bin count.
        x = np.array(
            [
                [-0.5, 0.5, 1.5],
                [-0.5, 1.5, 2.5],
                [-0.5, 2.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 1.5, 2.5],
                [0.5, 2.5, 2.5],
            ]
        )
        H, edges = histogramdd(x, (2, 3, 3), range=[[-1, 1], [0, 3], [0, 3]])
        answer = np.array(
            [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 0, 1]]]
        )
        assert_array_equal(H, answer)

        # Check normalization
        ed = [[-2, 0, 2], [0, 1, 2, 3], [0, 1, 2, 3]]
        H, edges = histogramdd(x, bins=ed, density=True)
        assert_(np.all(H == answer / 12.0))

        # Check that H has the correct shape.
        H, edges = histogramdd(
            x, (2, 3, 4), range=[[-1, 1], [0, 3], [0, 4]], density=True
        )
        answer = np.array(
            [
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]],
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
            ]
        )
        assert_array_almost_equal(H, answer / 6.0, 4)
        # Check that a sequence of arrays is accepted and H has the correct
        # shape.
        z = [np.squeeze(y) for y in np.split(x, 3, axis=1)]
        H, edges = histogramdd(z, bins=(4, 3, 2), range=[[-2, 2], [0, 3], [0, 2]])
        answer = np.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 1], [0, 0], [1, 0]],
                [[0, 1], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0]],
            ]
        )
        assert_array_equal(H, answer)

        Z = np.zeros((5, 5, 5))
        Z[list(range(5)), list(range(5)), list(range(5))] = 1.0
        H, edges = histogramdd([np.arange(5), np.arange(5), np.arange(5)], 5)
        assert_array_equal(H, Z)

    def test_shape_3d(self):
        # [FUNCTION DOC] For all orderings of three different bin counts, checks the result
        # shape equals the bin counts.
        # All possible permutations for bins of different lengths in 3D.
        bins = ((5, 4, 6), (6, 4, 5), (5, 6, 4), (4, 6, 5), (6, 5, 4), (4, 5, 6))
        r = np.random.rand(10, 3)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

    def test_shape_4d(self):
        # [FUNCTION DOC] For all orderings of four different bin counts on random points,
        # checks the result shape, and that unit weights give the same counts and edges.
        # All possible permutations for bins of different lengths in 4D.
        bins = (
            (7, 4, 5, 6),
            (4, 5, 7, 6),
            (5, 6, 4, 7),
            (7, 6, 5, 4),
            (5, 7, 6, 4),
            (4, 6, 7, 5),
            (6, 5, 7, 4),
            (7, 5, 4, 6),
            (7, 4, 6, 5),
            (6, 4, 7, 5),
            (6, 7, 5, 4),
            (4, 6, 5, 7),
            (4, 7, 5, 6),
            (5, 4, 6, 7),
            (5, 7, 4, 6),
            (6, 7, 4, 5),
            (6, 5, 4, 7),
            (4, 7, 6, 5),
            (4, 5, 6, 7),
            (7, 6, 4, 5),
            (5, 4, 7, 6),
            (5, 6, 7, 4),
            (6, 4, 5, 7),
            (7, 5, 6, 4),
        )

        r = np.random.rand(10, 4)
        for b in bins:
            H, edges = histogramdd(r, b)
            assert_(H.shape == b)

            h1, e1 = histogramdd(r, b, weights=np.ones(10))
            assert_equal(H, h1)
            for edge, e in zip(edges, e1):
                if not (edge == e).all():
                    raise AssertionError("Expected all edges to be equal")

    def test_weights(self):
        # [FUNCTION DOC] Checks unit weights equal no weights, equal weights with density
        # equal the unweighted density, and integer weights of two double the counts.
        v = np.random.rand(100, 2)
        hist, edges = histogramdd(v)
        n_hist, edges = histogramdd(v, density=True)
        w_hist, edges = histogramdd(v, weights=np.ones(100))
        assert_array_equal(w_hist, hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100) * 2, density=True)
        assert_array_equal(w_hist, n_hist)
        w_hist, edges = histogramdd(v, weights=np.ones(100, int) * 2)
        assert_array_equal(w_hist, 2 * hist)

    def test_identical_samples(self):
        # [FUNCTION DOC] Checks all-identical samples get edges extended by half a unit on
        # each side.
        x = np.zeros((10, 2), int)
        hist, edges = histogramdd(x, bins=2)
        assert_array_equal(edges[0], np.array([-0.5, 0.0, 0.5]))

    def test_empty(self):
        # [FUNCTION DOC] Checks empty samples give zero counts for explicit edges and for an
        # integer bin count in two and three dimensions.
        a, b = histogramdd([[], []], bins=([0, 1], [0, 1]))
        # assert_array_max_ulp(a, np.array([[0.]]))
        assert_allclose(a, np.array([[0.0]]), atol=1e-15)
        a, b = np.histogramdd([[], [], []], bins=2)
        # assert_array_max_ulp(a, np.zeros((2, 2, 2)))
        assert_allclose(a, np.zeros((2, 2, 2)), atol=1e-15)

    def test_bins_errors(self):
        # [FUNCTION DOC] Checks invalid bin specifications (negative, fractional, mismatched
        # dimension or nested negative edges) raise RuntimeError or ValueError.
        # There are two ways to specify bins. Check for the right errors
        # when mixing those.
        x = np.arange(8).reshape(2, 4)
        assert_raises((RuntimeError, ValueError), np.histogramdd, x, bins=[-1, 2, 4, 5])
        assert_raises(
            (RuntimeError, ValueError), np.histogramdd, x, bins=[1, 0.99, 1, 1]
        )
        assert_raises(
            (RuntimeError, ValueError), np.histogramdd, x, bins=[1, 1, 1, [1, 2, 3, -3]]
        )

    @xpassIfTorchDynamo_np  # (reason="pytorch does not support bins = [int, int, array]")
    def test_bins_error_2(self):
        # [FUNCTION DOC] Checks mixing integer bin counts and an explicit edge array is
        # accepted. Marked an expected failure outside dynamo (unsupported in torch).
        # mixing scalar (# of bins) and explicit bin arrays, ugh
        x = np.arange(8).reshape(2, 4)
        assert_(np.histogramdd(x, bins=[1, 1, 1, [1, 2, 3, 4]]))

    @xpassIfTorchDynamo_np  # (reason="pytorch does not support bins = [int, int, array]")
    def test_inf_edges(self):
        # [FUNCTION DOC] Checks infinite bin edges mixed with integer bin counts (issue 1788).
        # Marked an expected failure outside dynamo (unsupported in torch).
        # Test using +/-inf bin edges works. See #1788.
        x = np.arange(6).reshape(3, 2)
        expected = np.array([[1, 0], [0, 1], [0, 1]])
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 2, 10]])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, np.array([-1, 2, np.inf])])
        assert_allclose(h, expected)
        h, e = np.histogramdd(x, bins=[3, [-np.inf, 3, np.inf]])
        assert_allclose(h, expected)

    def test_rightmost_binedge(self):
        # [FUNCTION DOC] Regression test (gh issue 4266): values just below or at the last
        # edge land in the final bin, values just above it are not counted.
        # Test event very close to rightmost binedge. See Github issue #4266
        x = [0.9999999995]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 1.0)
        x = [1.0000000001]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)
        x = [1.0001]
        bins = [[0.0, 0.5, 1.0]]
        hist, _ = histogramdd(x, bins=bins)
        assert_(hist[0] == 0.0)
        assert_(hist[1] == 0.0)

    def test_finite_range(self):
        # [FUNCTION DOC] Checks finite ranges work while a range containing infinity or NaN
        # raises RuntimeError or ValueError.
        vals = np.random.random((100, 3))
        histogramdd(vals, range=[[0.0, 1.0], [0.25, 0.75], [0.25, 0.5]])
        assert_raises(
            (RuntimeError, ValueError),
            histogramdd,
            vals,
            range=[[0.0, 1.0], [0.25, 0.75], [0.25, np.inf]],
        )
        assert_raises(
            (RuntimeError, ValueError),
            histogramdd,
            vals,
            range=[[0.0, 1.0], [np.nan, 0.75], [0.25, 0.5]],
        )

    @xpassIfTorchDynamo_np  # (reason="pytorch does not allow equal entries")
    def test_equal_edges(self):
        # [FUNCTION DOC] Checks repeated adjacent edges are allowed and the value on the last
        # edge is in the final bin. Marked an expected failure outside dynamo (torch rejects
        # equal edges).
        """Test that adjacent entries in an edge array can be equal"""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        x_edges = np.array([0, 2, 2])
        y_edges = 1
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        hist_expected = np.array(
            [
                [2.0],
                [1.0],  # x == 2 falls in the final bin
            ]
        )
        assert_equal(hist, hist_expected)

    def test_edge_dtype(self):
        # [FUNCTION DOC] Checks the dtype of supplied edge arrays is preserved in the returned
        # edges.
        """Test that if an edge array is input, its type is preserved"""
        x = np.array([0, 10, 20])
        y = x / 10
        x_edges = np.array([0, 5, 15, 20])
        y_edges = x_edges / 10
        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(edges[0].dtype, x_edges.dtype)
        assert_equal(edges[1].dtype, y_edges.dtype)

    def test_large_integers(self):
        # [FUNCTION DOC] Checks very large integers (beyond float precision) fall in the right
        # bin.
        big = 2**60  # Too large to represent with a full precision float

        x = np.asarray([0], dtype=np.int64)
        x_edges = np.array([-1, +1], np.int64)
        y = big + x
        y_edges = big + x_edges

        hist, edges = histogramdd((x, y), bins=(x_edges, y_edges))

        assert_equal(hist[0, 0], 1)

    def test_density_non_uniform_2d(self):
        # [FUNCTION DOC] Checks, on a grid of unequal cell sizes, that point counts match the
        # relative cell areas, and that density then gives a uniform value of one over 64.
        # Defines the following grid:
        #
        #    0 2     8
        #   0+-+-----+
        #    + |     +
        #    + |     +
        #   6+-+-----+
        #   8+-+-----+
        x_edges = np.array([0, 2, 8])
        y_edges = np.array([0, 6, 8])
        relative_areas = np.array([[3, 9], [1, 3]])

        # ensure the number of points in each region is proportional to its area
        x = np.array([1] + [1] * 3 + [7] * 3 + [7] * 9)
        y = np.array([7] + [1] * 3 + [7] * 3 + [1] * 9)

        # sanity check that the above worked as intended
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges))
        assert_equal(hist, relative_areas)

        # resulting histogram should be uniform, since counts and areas are proportional
        hist, edges = histogramdd((y, x), bins=(y_edges, x_edges), density=True)
        assert_equal(hist, 1 / (8 * 8))

    def test_density_non_uniform_1d(self):
        # [FUNCTION DOC] Checks the 1-D N-dimensional histogram with non-uniform bins and
        # density equals the 1-D histogram results.
        # compare to histogram to show the results are the same
        v = np.arange(10)
        bins = np.array([0, 1, 3, 6, 10])
        hist, edges = histogram(v, bins, density=True)
        hist_dd, edges_dd = histogramdd((v,), (bins,), density=True)
        assert_equal(hist, hist_dd)
        assert_equal(edges, edges_dd[0])

    def test_bins_array(self):
        # [FUNCTION DOC] Checks the returned edges are all numpy arrays; raises AssertionError
        # otherwise.
        x = np.array(
            [
                [-0.5, 0.5, 1.5],
                [-0.5, 1.5, 2.5],
                [-0.5, 2.5, 0.5],
                [0.5, 0.5, 1.5],
                [0.5, 1.5, 2.5],
                [0.5, 2.5, 2.5],
            ]
        )
        H, edges = histogramdd(x, (2, 3, 3))
        if not all(type(e) is np.ndarray for e in edges):
            raise AssertionError("Expected all edges to be np.ndarray")


if __name__ == "__main__":
    run_tests()
