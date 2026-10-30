# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the matrix-construction helpers of the NumPy-compatible layer: eye,
# diag, fliplr, flipud, histogram2d, tri/tril/triu, tril/triu index helpers and vander. Under
# dynamo testing the real NumPy functions are imported, otherwise torch._numpy ones. Some
# tests are skipped or marked expected failures where features are unimplemented.

"""Test functions for matrix module"""

import functools
from unittest import expectedFailure as xfail, skipIf as skipif

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
    from numpy import (
        arange,
        array,
        diag,
        eye,
        fliplr,
        flipud,
        histogram2d,
        ones,
        tri,
        tril_indices,
        tril_indices_from,
        triu_indices,
        triu_indices_from,
        vander,
        zeros,
    )
    from numpy.testing import (
        assert_allclose,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
    )
else:
    import torch._numpy as np
    from torch._numpy import (
        arange,
        array,
        diag,
        eye,
        fliplr,
        flipud,
        histogram2d,
        ones,
        tri,
        tril_indices,
        tril_indices_from,
        triu_indices,
        triu_indices_from,
        vander,
        zeros,
    )
    from torch._numpy.testing import (
        assert_allclose,
        assert_array_almost_equal,
        assert_array_equal,
        assert_equal,
    )


skip = functools.partial(skipif, True)


def get_mat(n):
# [FUNCTION DOC] HELPER: Build a square integer matrix of size n whose entry at row i, column
# j is i + j, using an outer sum of the range of n integers. Returns that array.
    data = np.arange(n)
    # data = np.add.outer(data, data)
    data = data[:, None] + data[None, :]
    return data


class TestEye(TestCase):
# [CLASS DOC] Tests for eye: identity layout, diagonal offsets, rectangular shapes, dtype and
# order.
    def test_basic(self):
        # [FUNCTION DOC] Checks the default 4x4 identity, a float-typed identity, and that a
        # boolean comparison of a 3x3 identity equals the bool-dtype identity.
        assert_equal(
            eye(4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )

        assert_equal(
            eye(4, dtype="f"),
            array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "f"),
        )

        assert_equal(eye(3) == 1, eye(3, dtype=bool))

    def test_diag(self):
        # [FUNCTION DOC] Checks a 4x4 matrix with a positive and a negative diagonal offset puts
        # the ones on the shifted diagonal and zeros elsewhere.
        assert_equal(
            eye(4, k=1), array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        )

        assert_equal(
            eye(4, k=-1),
            array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
        )

    def test_2d(self):
        # [FUNCTION DOC] Checks non-square (4 rows by 3 columns and 3 by 4) identity matrices.
        assert_equal(eye(4, 3), array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))

        assert_equal(eye(3, 4), array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_diag2d(self):
        # [FUNCTION DOC] Checks non-square matrices combined with positive and negative diagonal
        # offsets.
        assert_equal(eye(3, 4, k=2), array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]))

        assert_equal(
            eye(4, 3, k=-2), array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
        )

    def test_eye_bounds(self):
        # [FUNCTION DOC] Checks offsets at and beyond the matrix edges for square and tall 2 and 3
        # row shapes: out-of-range offsets give all zeros, edge offsets give a single one or none.
        assert_equal(eye(2, 2, 1), [[0, 1], [0, 0]])
        assert_equal(eye(2, 2, -1), [[0, 0], [1, 0]])
        assert_equal(eye(2, 2, 2), [[0, 0], [0, 0]])
        assert_equal(eye(2, 2, -2), [[0, 0], [0, 0]])
        assert_equal(eye(3, 2, 2), [[0, 0], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, 1), [[0, 1], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, -1), [[0, 0], [1, 0], [0, 1]])
        assert_equal(eye(3, 2, -2), [[0, 0], [0, 0], [1, 0]])
        assert_equal(eye(3, 2, -3), [[0, 0], [0, 0], [0, 0]])

    def test_bool(self):
        # [FUNCTION DOC] Checks a boolean-dtype 2x2 identity holds True on the diagonal only.
        assert_equal(eye(2, 2, dtype=bool), [[True, False], [False, True]])

    @xpassIfTorchDynamo_np  # (reason="TODO: implement order=non-default")
    def test_order(self):
        # [FUNCTION DOC] Compares a C-order and a Fortran-order offset identity for equal values
        # and then raises AssertionError unless contiguity flags match the requested order. Marked
        # an expected failure outside dynamo because non-default order is not implemented in
        # torch._numpy; under dynamo it runs against real numpy.
        mat_c = eye(4, 3, k=-1)
        mat_f = eye(4, 3, k=-1, order="F")
        assert_equal(mat_c, mat_f)
        if not mat_c.flags.c_contiguous:
            raise AssertionError("Expected mat_c.flags.c_contiguous to be True")
        if mat_c.flags.f_contiguous:
            raise AssertionError("Expected mat_c.flags.f_contiguous to be False")
        if mat_f.flags.c_contiguous:
            raise AssertionError("Expected mat_f.flags.c_contiguous to be False")
        if not mat_f.flags.f_contiguous:
            raise AssertionError("Expected mat_f.flags.f_contiguous to be True")


class TestDiag(TestCase):
# [CLASS DOC] Tests for diag in both its vector-to-matrix and matrix-to-vector modes.
    def test_vector(self):
        # [FUNCTION DOC] Builds a diagonal matrix from a 1-D integer vector and compares it to a
        # hand-made expected matrix, also for diagonal offsets 2 and -2 in a larger 7x7 result.
        vals = (100 * arange(5)).astype("l")
        b = zeros((5, 5))
        for k in range(5):
            b[k, k] = vals[k]
        assert_equal(diag(vals), b)
        b = zeros((7, 7))
        c = b.copy()
        for k in range(5):
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        assert_equal(diag(vals, k=2), b)
        assert_equal(diag(vals, k=-2), c)

    def test_matrix(self):
        # [FUNCTION DOC] Runs the shared matrix check on a 5x5 integer matrix derived from the
        # helper that builds index-sum matrices.
        self.check_matrix(vals=(100 * get_mat(5) + 1).astype("l"))

    def check_matrix(self, vals):
        # [FUNCTION DOC] HELPER: Given a 5x5 matrix, verifies that extracting the main diagonal
        # and the diagonals at offsets 2 and -2 yields the elements gathered manually by a loop
        # (first three entries for the offset cases). Shared by the C-order and Fortran-order
        # tests.
        b = zeros((5,))
        for k in range(5):
            b[k] = vals[k, k]
        assert_equal(diag(vals), b)
        b = b * 0
        for k in range(3):
            b[k] = vals[k, k + 2]
        assert_equal(diag(vals, 2), b[:3])
        for k in range(3):
            b[k] = vals[k + 2, k]
        assert_equal(diag(vals, -2), b[:3])

    @xpassIfTorchDynamo_np  # (reason="TODO implement orders")
    def test_fortran_order(self):
        # [FUNCTION DOC] Runs the matrix check on a Fortran-ordered array. Marked an expected
        # failure outside dynamo (torch._numpy lacks non-default memory order); runs against numpy
        # under dynamo.
        vals = array((100 * get_mat(5) + 1), order="F", dtype="l")
        self.check_matrix(vals)

    def test_diag_bounds(self):
        # [FUNCTION DOC] Checks diagonal extraction on a 3x2 nested list for offsets from 2 down
        # to -3, including offsets that give an empty result.
        A = [[1, 2], [3, 4], [5, 6]]
        assert_equal(diag(A, k=2), [])
        assert_equal(diag(A, k=1), [2])
        assert_equal(diag(A, k=0), [1, 4])
        assert_equal(diag(A, k=-1), [3, 6])
        assert_equal(diag(A, k=-2), [5])
        assert_equal(diag(A, k=-3), [])

    def test_failure(self):
        # [FUNCTION DOC] Checks that a 3-D input raises ValueError or RuntimeError.
        assert_raises((ValueError, RuntimeError), diag, [[[1]]])


class TestFliplr(TestCase):
# [CLASS DOC] Tests for fliplr (reverse the column order).
    def test_basic(self):
        # [FUNCTION DOC] Checks a 1-D input raises ValueError or RuntimeError, that results match
        # a flip along axis 1 for a 4x4 matrix, and a 2x3 nested-list example.
        assert_raises((ValueError, RuntimeError), fliplr, ones(4))
        a = get_mat(4)
        # b = a[:, ::-1]
        b = np.flip(a, 1)
        assert_equal(fliplr(a), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[2, 1, 0], [5, 4, 3]]
        assert_equal(fliplr(a), b)


class TestFlipud(TestCase):
# [CLASS DOC] Tests for flipud (reverse the row order).
    def test_basic(self):
        # [FUNCTION DOC] Checks results match a flip along axis 0 for a 4x4 matrix and a 2x3
        # nested list.
        a = get_mat(4)
        # b = a[::-1, :]
        b = np.flip(a, 0)
        assert_equal(flipud(a), b)
        a = [[0, 1, 2], [3, 4, 5]]
        b = [[3, 4, 5], [0, 1, 2]]
        assert_equal(flipud(a), b)


@instantiate_parametrized_tests
class TestHistogram2d(TestCase):
# [CLASS DOC] Tests for histogram2d: explicit edges, ranges, density, outliers and empty
# input.
    def test_simple(self):
        # [FUNCTION DOC] Checks counts for five points against explicit 9x9 edges (compared
        # transposed), a single shared edge array, and default bins for 10 consecutive integers,
        # giving a 10x10 identity count matrix with edges evenly spaced from 0 to 9.
        x = array([0.41702200, 0.72032449, 1.1437481e-4, 0.302332573, 0.146755891])
        y = array([0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
        xedges = np.linspace(0, 1, 10)
        yedges = np.linspace(0, 1, 10)
        H = histogram2d(x, y, (xedges, yedges))[0]
        answer = array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        assert_array_equal(H.T, answer)
        H = histogram2d(x, y, xedges)[0]
        assert_array_equal(H.T, answer)
        H, xedges, yedges = histogram2d(list(range(10)), list(range(10)))
        assert_array_equal(H, eye(10, 10))
        assert_array_equal(xedges, np.linspace(0, 9, 11))
        assert_array_equal(yedges, np.linspace(0, 9, 11))

    def test_asym(self):
        # [FUNCTION DOC] Checks a density histogram with differing bin counts per axis and given
        # ranges: counts normalised by 8 samples (3 decimals) and the returned edges are evenly
        # spaced.
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        H, xed, yed = histogram2d(x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
        answer = array(
            [
                [0.0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        assert_array_almost_equal(H, answer / 8.0, 3)
        assert_array_equal(xed, np.linspace(0, 6, 7))
        assert_array_equal(yed, np.linspace(0, 5, 6))

    def test_density(self):
        # [FUNCTION DOC] Checks a density histogram with non-uniform bin edges, where the expected
        # values are count divided by bin area, normalised by the 9 samples.
        x = array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        y = array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        H, xed, yed = histogram2d(x, y, [[1, 2, 3, 5], [1, 2, 3, 5]], density=True)
        answer = array([[1, 1, 0.5], [1, 1, 0.5], [0.5, 0.5, 0.25]]) / 9.0
        assert_array_almost_equal(H, answer, 3)

    def test_all_outliers(self):
        # [FUNCTION DOC] Checks that points all outside the given range produce an all-zero
        # histogram. The offset of 1e6 is there because the reference implementation rounds edges
        # to 6 decimals.
        r = np.random.rand(100) + 1.0 + 1e6  # histogramdd rounds by decimal=6
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0, 1], [0, 1]))
        assert_array_equal(H, 0)

    def test_empty(self):
        # [FUNCTION DOC] Checks empty inputs: with explicit single bin edges the result is a 1x1
        # zero matrix, and with an integer bin count of 4 a 4x4 zero matrix (absolute tolerance
        # 1e-15).
        a, edge1, edge2 = histogram2d([], [], bins=([0, 1], [0, 1]))
        # assert_array_max_ulp(a, array([[0.]]))
        assert_allclose(a, np.array([[0.0]]), atol=1e-15)

        a, edge1, edge2 = histogram2d([], [], bins=4)
        # assert_array_max_ulp(a, np.zeros((4, 4)))
        assert_allclose(a, np.zeros((4, 4)), atol=1e-15)

    @xpassIfTorchDynamo_np  # (reason="pytorch does not support bins = [int, array]")
    def test_binparameter_combination(self):
        # [FUNCTION DOC] Checks mixing an edge sequence on one axis with an integer bin count on
        # the other in both orders, comparing counts and the computed edges. Marked an expected
        # failure outside dynamo, since torch does not support such mixed bins; runs against numpy
        # under dynamo.
        x = array([0, 0.09207008, 0.64575234, 0.12875982, 0.47390599, 0.59944483, 1])
        y = array([0, 0.14344267, 0.48988575, 0.30558665, 0.44700682, 0.15886423, 1])
        edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        H, xe, ye = histogram2d(x, y, (edges, 4))
        answer = array(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert_array_equal(H, answer)
        assert_array_equal(ye, array([0.0, 0.25, 0.5, 0.75, 1]))
        H, xe, ye = histogram2d(x, y, (4, edges))
        answer = array(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert_array_equal(H, answer)
        assert_array_equal(xe, array([0.0, 0.25, 0.5, 0.75, 1]))

    @skip(reason="NP_VER: fails on CI with older NumPy")
    @parametrize("x_len, y_len", [(10, 11), (20, 19)])
    def test_bad_length(self, x_len, y_len):
        # [FUNCTION DOC] Checks that arrays of different lengths (10 vs 11 and 20 vs 19) make
        # histogram2d raise ValueError with the message about equal length. Always skipped via the
        # skip decorator.
        x, y = np.ones(x_len), np.ones(y_len)
        with pytest.raises(ValueError, match="x and y must have the same length."):
            histogram2d(x, y)


class TestTri(TestCase):
# [CLASS DOC] Tests for tri, tril, triu and index helpers built on them.
    def test_dtype(self):
        # [FUNCTION DOC] Checks the 3x3 lower-triangular ones matrix from tri, with default and
        # bool dtype.
        out = array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        assert_array_equal(tri(3), out)
        assert_array_equal(tri(3, dtype=bool), out.astype(bool))

    def test_tril_triu_ndim2(self):
        # [FUNCTION DOC] For every float and integer type code, checks tril/triu of a 2x2 ones
        # matrix give the expected triangles, are transposes of one another, and keep the input
        # dtype.
        for dtype in np.typecodes["AllFloat"] + np.typecodes["AllInteger"]:
            a = np.ones((2, 2), dtype=dtype)
            b = np.tril(a)
            c = np.triu(a)
            assert_array_equal(b, [[1, 0], [1, 1]])
            assert_array_equal(c, b.T)
            # should return the same dtype as the original array
            assert_equal(b.dtype, a.dtype)
            assert_equal(c.dtype, a.dtype)

    def test_tril_triu_ndim3(self):
        # [FUNCTION DOC] For every float and integer type code, checks tril/triu on a 3-D array
        # apply to each trailing 2-D matrix and preserve dtype.
        for dtype in np.typecodes["AllFloat"] + np.typecodes["AllInteger"]:
            a = np.array(
                [
                    [[1, 1], [1, 1]],
                    [[1, 1], [1, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=dtype,
            )
            a_tril_desired = np.array(
                [
                    [[1, 0], [1, 1]],
                    [[1, 0], [1, 0]],
                    [[1, 0], [0, 0]],
                ],
                dtype=dtype,
            )
            a_triu_desired = np.array(
                [
                    [[1, 1], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[1, 1], [0, 0]],
                ],
                dtype=dtype,
            )
            a_triu_observed = np.triu(a)
            a_tril_observed = np.tril(a)
            assert_array_equal(a_triu_observed, a_triu_desired)
            assert_array_equal(a_tril_observed, a_tril_desired)
            assert_equal(a_triu_observed.dtype, a.dtype)
            assert_equal(a_tril_observed.dtype, a.dtype)

    def test_tril_triu_with_inf(self):
        # [FUNCTION DOC] Regression test (issue 4859): infinities in the input are kept in the
        # retained triangle and zeroed outside it, with no NaN produced.
        # Issue 4859
        arr = np.array([[1, 1, np.inf], [1, 1, 1], [np.inf, 1, 1]])
        out_tril = np.array([[1, 0, 0], [1, 1, 0], [np.inf, 1, 1]])
        out_triu = out_tril.T
        assert_array_equal(np.triu(arr), out_triu)
        assert_array_equal(np.tril(arr), out_tril)

    def test_tril_triu_dtype(self):
        # [FUNCTION DOC] Regression test (issue 4916): for a list of dtype codes, tril and triu of
        # a zero matrix keep the input dtype.
        # Issue 4916
        # tril and triu should return the same dtype as input
        for c in "efdFDBbhil?":  # np.typecodes["All"]:
            arr = np.zeros((3, 3), dtype=c)
            assert_equal(np.triu(arr).dtype, arr.dtype)
            assert_equal(np.tril(arr).dtype, arr.dtype)

    @xfail  # (reason="TODO: implement mask_indices")
    def test_mask_indices(self):
        # [FUNCTION DOC] Intended to test mask_indices with and without an offset. Marked expected
        # failure because it is not implemented.
        # KNOWN ISSUE: mask_indices is neither imported nor defined here, so the body would raise
        # NameError.
        # simple test without offset
        iu = mask_indices(3, np.triu)
        a = np.arange(9).reshape(3, 3)
        assert_array_equal(a[iu], array([0, 1, 2, 4, 5, 8]))
        # Now with an offset
        iu1 = mask_indices(3, np.triu, 1)
        assert_array_equal(a[iu1], array([1, 2, 5]))

    @xfail  # (reason="np.tril_indices == our tuple(tril_indices)")
    def test_tril_indices(self):
        # [FUNCTION DOC] Intended to test tril_indices with offset and column count variants,
        # indexing and assignment through the results. Marked expected failure because the torch
        # version returns a tuple while numpy returns an array of indices.
        # indices without and with offset
        il1 = tril_indices(4)
        il2 = tril_indices(4, k=2)
        il3 = tril_indices(4, m=5)
        il4 = tril_indices(4, k=2, m=5)

        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # indexing:
        assert_array_equal(a[il1], array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]))
        assert_array_equal(b[il3], array([1, 6, 7, 11, 12, 13, 16, 17, 18, 19]))

        # And for assigning values:
        a[il1] = -1
        assert_array_equal(
            a,
            array([[-1, 2, 3, 4], [-1, -1, 7, 8], [-1, -1, -1, 12], [-1, -1, -1, -1]]),
        )
        b[il3] = -1
        assert_array_equal(
            b,
            array(
                [
                    [-1, 2, 3, 4, 5],
                    [-1, -1, 8, 9, 10],
                    [-1, -1, -1, 14, 15],
                    [-1, -1, -1, -1, 20],
                ]
            ),
        )
        # These cover almost the whole array (two diagonals right of the main one):
        a[il2] = -10
        assert_array_equal(
            a,
            array(
                [
                    [-10, -10, -10, 4],
                    [-10, -10, -10, -10],
                    [-10, -10, -10, -10],
                    [-10, -10, -10, -10],
                ]
            ),
        )
        b[il4] = -10
        assert_array_equal(
            b,
            array(
                [
                    [-10, -10, -10, 4, 5],
                    [-10, -10, -10, -10, 10],
                    [-10, -10, -10, -10, -10],
                    [-10, -10, -10, -10, -10],
                ]
            ),
        )


@xfail  # (reason="np.triu_indices == our tuple(triu_indices)")
class TestTriuIndices(TestCase):
# [CLASS DOC] Tests for triu_indices; the whole class is marked as an expected failure because
# the torch return type is a tuple, unlike numpy.
    def test_triu_indices(self):
        # [FUNCTION DOC] Checks triu_indices with offset and column count variants, using the
        # results to read from and assign into a 4x4 and a 4x5 array and comparing against
        # expected contents.
        iu1 = triu_indices(4)
        iu2 = triu_indices(4, k=2)
        iu3 = triu_indices(4, m=5)
        iu4 = triu_indices(4, k=2, m=5)

        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # Both for indexing:
        assert_array_equal(a[iu1], array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16]))
        assert_array_equal(
            b[iu3], array([1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 19, 20])
        )

        # And for assigning values:
        a[iu1] = -1
        assert_array_equal(
            a,
            array(
                [[-1, -1, -1, -1], [5, -1, -1, -1], [9, 10, -1, -1], [13, 14, 15, -1]]
            ),
        )
        b[iu3] = -1
        assert_array_equal(
            b,
            array(
                [
                    [-1, -1, -1, -1, -1],
                    [6, -1, -1, -1, -1],
                    [11, 12, -1, -1, -1],
                    [16, 17, 18, -1, -1],
                ]
            ),
        )

        # These cover almost the whole array (two diagonals right of the
        # main one):
        a[iu2] = -10
        assert_array_equal(
            a,
            array(
                [
                    [-1, -1, -10, -10],
                    [5, -1, -1, -10],
                    [9, 10, -1, -1],
                    [13, 14, 15, -1],
                ]
            ),
        )
        b[iu4] = -10
        assert_array_equal(
            b,
            array(
                [
                    [-1, -1, -10, -10, -10],
                    [6, -1, -1, -10, -10],
                    [11, 12, -1, -1, -10],
                    [16, 17, 18, -1, -1],
                ]
            ),
        )


class TestTrilIndicesFrom(TestCase):
# [CLASS DOC] Tests for tril_indices_from argument validation.
    def test_exceptions(self):
        # [FUNCTION DOC] Checks 1-D and 3-D inputs raise ValueError.
        assert_raises(ValueError, tril_indices_from, np.ones((2,)))
        assert_raises(ValueError, tril_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, tril_indices_from, np.ones((2, 3)))


class TestTriuIndicesFrom(TestCase):
# [CLASS DOC] Tests for triu_indices_from argument validation.
    def test_exceptions(self):
        # [FUNCTION DOC] Checks 1-D and 3-D inputs raise ValueError.
        assert_raises(ValueError, triu_indices_from, np.ones((2,)))
        assert_raises(ValueError, triu_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, triu_indices_from, np.ones((2, 3)))


class TestVander(TestCase):
# [CLASS DOC] Tests for vander (Vandermonde matrix).
    def test_basic(self):
        # [FUNCTION DOC] Checks the default number of columns (length of input) and explicit
        # column counts from 0 to 5, comparing with the corresponding slice of a power table.
        c = np.array([0, 1, -2, 3])
        v = vander(c)
        powers = np.array(
            [[0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [16, -8, 4, -2, 1], [81, 27, 9, 3, 1]]
        )
        # Check default value of N:
        assert_array_equal(v, powers[:, 1:])
        # Check a range of N values, including 0 and 5 (greater than default)
        m = powers.shape[1]
        for n in range(6):
            v = vander(c, N=n)
            assert_array_equal(v, powers[:, m - n : m])

    def test_dtypes(self):
        # [FUNCTION DOC] Checks vander on int8 input and on complex input with three columns gives
        # the expected integer-valued results.
        c = array([11, -12, 13], dtype=np.int8)
        v = vander(c)
        expected = np.array([[121, 11, 1], [144, -12, 1], [169, 13, 1]])
        assert_array_equal(v, expected)

        c = array([1.0 + 1j, 1.0 - 1j])
        v = vander(c, N=3)
        expected = np.array([[2j, 1 + 1j, 1], [-2j, 1 - 1j, 1]])
        # The data is floating point, but the values are small integers,
        # so assert_array_equal *should* be safe here (rather than, say,
        # assert_array_almost_equal).
        assert_array_equal(v, expected)


if __name__ == "__main__":
    run_tests()
