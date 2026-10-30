# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for shape-manipulation helpers of the NumPy-compatible layer:
# take_along_axis, put_along_axis, apply_along_axis, apply_over_axes, expand_dims, array_split
# and split, the stack and split families (column_stack, dstack, hsplit, vsplit, dsplit),
# squeeze, kron, tile and may_share_memory. Real NumPy is used under dynamo testing, otherwise
# torch._numpy. Also defines a keepdims wrapper helper and a list-of-arrays comparison helper.
# KNOWN ISSUE: apply_along_axis is imported only in the dynamo branch (the tests call it by
# that bare name), and apply_over_axes is never imported; both classes are expected failures.

import functools
import sys
from unittest import expectedFailure as xfail, skipIf as skipif

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        apply_along_axis,
        array_split,
        column_stack,
        dsplit,
        dstack,
        expand_dims,
        hsplit,
        kron,
        put_along_axis,
        split,
        take_along_axis,
        tile,
        vsplit,
    )
    from numpy.random import rand, randint
    from numpy.testing import assert_, assert_array_equal, assert_equal

else:
    import torch._numpy as np
    from torch._numpy import (
        array_split,
        column_stack,
        dsplit,
        dstack,
        expand_dims,
        hsplit,
        kron,
        put_along_axis,
        split,
        take_along_axis,
        tile,
        vsplit,
    )
    from torch._numpy.random import rand, randint
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal


skip = functools.partial(skipif, True)


IS_64BIT = sys.maxsize > 2**32


def _add_keepdims(func):
    # [FUNCTION DOC] HELPER: Wrap a reduction-like function taking an array and an axis so
    # that the reduced axis is kept as size one, by calling the function and expanding the
    # result along the axis (axis zero when the axis is None, since the result is then a
    # scalar). Returns the wrapper.
    """hack in keepdims behavior into a function taking an axis"""

    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        # [FUNCTION DOC] HELPER: Call the wrapped function with the array, axis and extra
        # keywords, then expand dimensions of the result at that axis and return it.
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # res is now a scalar, so we can insert this anywhere
        return np.expand_dims(res, axis=axis)

    return wrapped


class TestTakeAlongAxis(TestCase):
    # [CLASS DOC] Tests for take_along_axis, including equivalence with sort/min/max via
    # argument indices, invalid indices, empty results and broadcasting.
    def test_argequivalent(self):
        # [FUNCTION DOC] For sort, min and max (the latter two with kept dims) on a random 3-D
        # array and for every axis plus None, checks taking along the axis with the matching
        # arg-function indices reproduces the direct result.
        """Test it translates from arg<func> to <func>"""
        a = rand(3, 4, 5)

        funcs = [
            (np.sort, np.argsort, {}),
            (_add_keepdims(np.min), _add_keepdims(np.argmin), {}),
            (_add_keepdims(np.max), _add_keepdims(np.argmax), {}),
            #  FIXME           (np.partition, np.argpartition, dict(kth=2)),
        ]

        for func, argfunc, kwargs in funcs:
            for axis in list(range(a.ndim)) + [None]:
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))

    def test_invalid(self):
        # [FUNCTION DOC] Checks take_along_axis works for valid 2-D indices, and raises on: a
        # 0-d index array (ValueError or RuntimeError), boolean or float indices (IndexError
        # or RuntimeError), and an out-of-range axis (AxisError).
        """Test it errors when indices has too few dimensions"""
        a = np.ones((10, 10))
        ai = np.ones((10, 2), dtype=np.intp)

        # sanity check
        take_along_axis(a, ai, axis=1)

        # not enough indices
        assert_raises(
            (ValueError, RuntimeError), take_along_axis, a, np.array(1), axis=1
        )
        # bool arrays not allowed
        assert_raises(
            (IndexError, RuntimeError), take_along_axis, a, ai.astype(bool), axis=1
        )
        # float arrays not allowed
        assert_raises(
            (IndexError, RuntimeError), take_along_axis, a, ai.astype(float), axis=1
        )
        # invalid axis
        assert_raises(np.AxisError, take_along_axis, a, ai, axis=10)

    def test_empty(self):
        # [FUNCTION DOC] Checks empty index arrays along the taken axis give a result with the
        # index array's shape.
        """Test everything is ok with empty results, even with inserted dims"""
        a = np.ones((3, 4, 5))
        ai = np.ones((3, 0, 5), dtype=np.intp)

        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, ai.shape)

    def test_broadcast(self):
        # [FUNCTION DOC] Checks non-indexed axes of size one are broadcast against the
        # indices, in both operands, giving shape 3, 2, 5.
        """Test that non-indexing dimensions are broadcast in both directions"""
        a = np.ones((3, 4, 1))
        ai = np.ones((1, 2, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, (3, 2, 5))


class TestPutAlongAxis(TestCase):
    # [CLASS DOC] Tests for put_along_axis, which writes values into an array in place.
    def test_replace_max(self):
        # [FUNCTION DOC] For each axis plus None, writes a very small value at the position of
        # the maximum and checks the minimum afterwards is at the same position as the former
        # maximum.
        a_base = np.array([[10, 30, 20], [60, 40, 50]])

        for axis in list(range(a_base.ndim)) + [None]:
            # we mutate this in the loop
            a = a_base.copy()

            # replace the max with a small value
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            put_along_axis(a, i_max, -99, axis=axis)

            # find the new minimum, which should max
            i_min = _add_keepdims(np.argmin)(a, axis=axis)

            assert_equal(i_min, i_max)

    @xpassIfTorchDynamo_np  # (
    # reason="RuntimeError: Expected index [1, 2, 5] to be smaller than self [3, 4, 1] apart from dimension 1")
    def test_broadcast(self):
        # [FUNCTION DOC] Checks put_along_axis broadcasts non-indexed axes in both directions
        # and that a read back with the same indices gives the written value. Marked expected
        # failure outside dynamo because torch rejects an index larger than the array on non-
        # indexed axes.
        """Test that non-indexing dimensions are broadcast in both directions"""
        a = np.ones((3, 4, 1))
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        put_along_axis(a, ai, 20, axis=1)
        assert_equal(take_along_axis(a, ai, axis=1), 20)


@xpassIfTorchDynamo_np  # (reason="apply_along_axis not implemented")
class TestApplyAlongAxis(TestCase):
    # [CLASS DOC] Tests for apply_along_axis; the class is marked an expected failure outside
    # dynamo because it is not implemented in torch._numpy.
    def test_simple(self):
        # [FUNCTION DOC] Applies the length function along axis zero of a 20x10 ones array and
        # checks each output is 20.
        a = np.ones((20, 10), "d")
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_simple101(self):
        # [FUNCTION DOC] Same as the simple test with a 10x101 array, covering a longer output
        # length.
        a = np.ones((10, 101), "d")
        assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))

    def test_3d(self):
        # [FUNCTION DOC] Sums along axis zero of a 3x3x3 array and checks the 3x3 result.
        a = np.arange(27).reshape((3, 3, 3))
        assert_array_equal(
            apply_along_axis(np.sum, 0, a), [[27, 30, 33], [36, 39, 42], [45, 48, 51]]
        )

    def test_scalar_array(self, cls=np.ndarray):
        # [FUNCTION DOC] Sums along axis zero of a 6x3 ones array viewed as the class given by
        # the cls argument (default ndarray) and checks the result keeps that class and has
        # values 6.
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(np.sum, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        # [FUNCTION DOC] With a function that returns a 0-d array, applies along axis 0 and 1
        # of a 6x3 array viewed as the cls argument class, checking the result type and the
        # summed values.
        def sum_to_0d(x):
            # [FUNCTION DOC] HELPER: Sum a 1-D input (asserting it is 1-D), keeping dims and
            # then squeezing to return a 0-d array.
            """Sum x, returning a 0d array of the same class"""
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))

        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(sum_to_0d, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

        res = apply_along_axis(sum_to_0d, 1, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))

    def test_axis_insertion(self, cls=np.ndarray):
        # [FUNCTION DOC] Checks that when the applied function returns a 2-D result from a 1-D
        # input, the output axes are inserted at the right place for 2-D (first and last axis)
        # and 3-D (middle axis) inputs, by comparing with manually stacked results, including
        # the result type.
        def f1to2(x):
            # [FUNCTION DOC] HELPER: Given a 1-D input (asserted), returns a non-square 2-D
            # array formed by the reversed input multiplied by the column of the input without
            # its first element, viewed as cls.
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            return (x[::-1] * x[1:, None]).view(cls)

        a2d = np.arange(6 * 3).reshape((6, 3))

        # 2d insertion along first axis
        actual = apply_along_axis(f1to2, 0, a2d)
        expected = np.stack(
            [f1to2(a2d[:, i]) for i in range(a2d.shape[1])], axis=-1
        ).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 2d insertion along last axis
        actual = apply_along_axis(f1to2, 1, a2d)
        expected = np.stack(
            [f1to2(a2d[i, :]) for i in range(a2d.shape[0])], axis=0
        ).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 3d insertion along middle axis
        a3d = np.arange(6 * 5 * 3).reshape((6, 5, 3))

        actual = apply_along_axis(f1to2, 1, a3d)
        expected = np.stack(
            [
                np.stack([f1to2(a3d[i, :, j]) for i in range(a3d.shape[0])], axis=0)
                for j in range(a3d.shape[2])
            ],
            axis=-1,
        ).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

    def test_axis_insertion_ma(self):
        # [FUNCTION DOC] Checks apply_along_axis with a function returning masked arrays gives
        # a 3-D masked array whose masks match the per-column results.
        def f1to2(x):
            # [FUNCTION DOC] HELPER: Given a 1-D input (asserted), builds the asymmetric 2-D
            # product and returns it as a masked array with entries divisible by five masked.
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            res = x[::-1] * x[1:, None]
            return np.ma.masked_where(res % 5 == 0, res)

        a = np.arange(6 * 3).reshape((6, 3))
        res = apply_along_axis(f1to2, 0, a)
        assert_(isinstance(res, np.ma.masked_array))
        assert_equal(res.ndim, 3)
        assert_array_equal(res[:, :, 0].mask, f1to2(a[:, 0]).mask)
        assert_array_equal(res[:, :, 1].mask, f1to2(a[:, 1]).mask)
        assert_array_equal(res[:, :, 2].mask, f1to2(a[:, 2]).mask)

    def test_tuple_func1d(self):
        # [FUNCTION DOC] Checks a function returning a tuple of two elements (the pair
        # swapped) is turned into an array along the applied axis.
        def sample_1d(x):
            # [FUNCTION DOC] HELPER: Return the second then the first element of a 1-D input
            # as a tuple.
            return x[1], x[0]

        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):
        # [FUNCTION DOC] Checks empty-input behavior: a 0x0 array raises ValueError for both
        # axes because the function can never be called, whereas a 10x0 array works along axis
        # one (function returns one per row) and raises ValueError along axis zero.
        # can't apply_along_axis when there's no chance to call the function
        def never_call(x):
            # [FUNCTION DOC] HELPER: Fails the test if called; used to show the function is
            # never invoked.
            assert_(False)  # should never be reached

        a = np.empty((0, 0))
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        # but it's sometimes ok with some non-zero dimensions
        def empty_to_1(x):
            # [FUNCTION DOC] HELPER: Asserts its input is empty and returns the constant one.
            assert_(len(x) == 0)
            return 1

        a = np.empty((10, 0))
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    @skip  # TypeError: descriptor 'union' for 'set' objects doesn't apply to a 'numpy.int64' object
    def test_with_iterable_object(self):
        # [FUNCTION DOC] Regression test (issue 5248, 8642): applying a set-union across
        # object arrays of sets, comparing values and element types. Always skipped (cannot
        # work with numpy ints).
        # from issue 5248
        d = np.array([[{1, 11}, {2, 22}, {3, 33}], [{4, 44}, {5, 55}, {6, 66}]])
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])

        assert_equal(actual, expected)

        # issue 8642 - assert_equal doesn't detect this!
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))


@xfail  # (reason="apply_over_axes not implemented")
class TestApplyOverAxes(TestCase):
    # [CLASS DOC] Tests for apply_over_axes; marked an expected failure as it is not
    # implemented.
    def test_simple(self):
        # [FUNCTION DOC] Checks summing a 2x3x4 array over axes zero and two gives shape 1x3x1
        # with the expected sums.
        # KNOWN ISSUE: apply_over_axes is not imported in this module.
        a = np.arange(24).reshape(2, 3, 4)
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))


class TestExpandDims(TestCase):
    # [CLASS DOC] Tests for expand_dims with single axes, axis tuples and invalid axes.
    def test_functionality(self):
        # [FUNCTION DOC] For every axis from minus five to three on a 4-D array, checks the
        # new axis has size one and squeezing recovers the original shape.
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert_(b.shape[axis] == 1)
            assert_(np.squeeze(b).shape == s)

    def test_axis_tuple(self):
        # [FUNCTION DOC] Checks tuples of axes (positive and negative) give the expected
        # output shapes on a 3x3x3 array, raising AssertionError on mismatch.
        a = np.empty((3, 3, 3))
        if np.expand_dims(a, axis=(0, 1, 2)).shape != (1, 1, 1, 3, 3, 3):
            raise AssertionError(
                f"Expected shape (1, 1, 1, 3, 3, 3), got {np.expand_dims(a, axis=(0, 1, 2)).shape}"
            )
        if np.expand_dims(a, axis=(0, -1, -2)).shape != (1, 3, 3, 3, 1, 1):
            raise AssertionError(
                f"Expected shape (1, 3, 3, 3, 1, 1), got {np.expand_dims(a, axis=(0, -1, -2)).shape}"
            )
        if np.expand_dims(a, axis=(0, 3, 5)).shape != (1, 3, 3, 1, 3, 1):
            raise AssertionError(
                f"Expected shape (1, 3, 3, 1, 3, 1), got {np.expand_dims(a, axis=(0, 3, 5)).shape}"
            )
        if np.expand_dims(a, axis=(0, -3, -5)).shape != (1, 1, 3, 1, 3, 3):
            raise AssertionError(
                f"Expected shape (1, 1, 3, 1, 3, 3), got {np.expand_dims(a, axis=(0, -3, -5)).shape}"
            )

    def test_axis_out_of_range(self):
        # [FUNCTION DOC] Checks out-of-range single axes and axis tuples raise AxisError.
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        # [FUNCTION DOC] Checks a repeated axis in a tuple raises ValueError.
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))


class TestArraySplit(TestCase):
    # [CLASS DOC] Tests for array_split with integer section counts and index lists.
    def test_integer_0_split(self):
        # [FUNCTION DOC] Checks splitting into zero sections raises ValueError.
        a = np.arange(10)
        assert_raises(ValueError, array_split, a, 0)

    def test_integer_split(self):
        # [FUNCTION DOC] Splits a length-10 array into 1 to 11 sections and compares with
        # expected pieces: earlier pieces are one longer when the length is not divisible, and
        # extra sections beyond the length are empty.
        a = np.arange(10)
        res = array_split(a, 1)
        desired = [np.arange(10)]
        compare_results(res, desired)

        res = array_split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

        res = array_split(a, 3)
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)

        res = array_split(a, 4)
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)

        res = array_split(a, 5)
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 8),
            np.arange(8, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 6)
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 7)
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 8)
        desired = [
            np.arange(2),
            np.arange(2, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 9)
        desired = [
            np.arange(2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 10)
        desired = [
            np.arange(1),
            np.arange(1, 2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
        ]
        compare_results(res, desired)

        res = array_split(a, 11)
        desired = [
            np.arange(1),
            np.arange(1, 2),
            np.arange(2, 3),
            np.arange(3, 4),
            np.arange(4, 5),
            np.arange(5, 6),
            np.arange(6, 7),
            np.arange(7, 8),
            np.arange(8, 9),
            np.arange(9, 10),
            np.array([]),
        ]
        compare_results(res, desired)

    def test_integer_split_2D_rows(self):
        # [FUNCTION DOC] Splits a 2x10 array into 3 sections along rows (last is empty) and
        # with index list zero, one (first is empty), also checking the empty piece keeps the
        # dtype.
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

        # Same thing for manual splits:
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]), np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    def test_integer_split_2D_cols(self):
        # [FUNCTION DOC] Splits a 2x10 array into 3 sections along the last axis and checks
        # the column chunks (4, 3, 3 wide).
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=-1)
        desired = [
            np.array([np.arange(4), np.arange(4)]),
            np.array([np.arange(4, 7), np.arange(4, 7)]),
            np.array([np.arange(7, 10), np.arange(7, 10)]),
        ]
        compare_results(res, desired)

    def test_integer_split_2D_default(self):
        # [FUNCTION DOC] Checks the default axis is zero by splitting 2 rows into 3 sections,
        # the last being empty with preserved dtype.
        """This will fail if we change default axis"""
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)
        # perhaps should check higher dimensions

    @skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_integer_split_2D_rows_greater_max_int32(self):
        # [FUNCTION DOC] Splits a broadcast array with 2**32 rows into 4 parts and checks
        # every part has the expected shape (a quarter of the rows). Skipped unless the
        # platform is 64-bit.
        a = np.broadcast_to([0], (1 << 32, 2))
        res = array_split(a, 4)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        tgt = [chunk] * 4
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    def test_index_split_simple(self):
        # [FUNCTION DOC] Splits at a list of increasing indices and checks the four pieces.
        a = np.arange(10)
        indices = [1, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_low_bound(self):
        # [FUNCTION DOC] Checks a split index of zero gives an empty first piece.
        a = np.arange(10)
        indices = [0, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_high_bound(self):
        # [FUNCTION DOC] Checks split indices at or beyond the length give empty trailing
        # pieces.
        a = np.arange(10)
        indices = [0, 5, 7, 10, 12]
        res = array_split(a, indices, axis=-1)
        desired = [
            np.array([]),
            np.arange(0, 5),
            np.arange(5, 7),
            np.arange(7, 10),
            np.array([]),
            np.array([]),
        ]
        compare_results(res, desired)


class TestSplit(TestCase):
    # [CLASS DOC] Tests for split, which differs from array_split only by requiring equal
    # sections.
    # The split function is essentially the same as array_split,
    # except that it test if splitting will result in an
    # equal split.  Only test for this case.

    def test_equal_split(self):
        # [FUNCTION DOC] Checks splitting 10 elements in 2 gives two halves.
        a = np.arange(10)
        res = split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

    def test_unequal_split(self):
        # [FUNCTION DOC] Checks splitting 10 elements in 3 raises ValueError.
        a = np.arange(10)
        assert_raises(ValueError, split, a, 3)


class TestColumnStack(TestCase):
    # [CLASS DOC] Tests for column_stack.
    def test_non_iterable(self):
        # [FUNCTION DOC] Checks passing a non-iterable raises TypeError.
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # [FUNCTION DOC] Checks 1-D arrays are stacked as columns into a 3x2 array.
        # example from docstring
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # [FUNCTION DOC] Checks column vectors are concatenated along columns.
        # same as hstack 2D docstring example
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2], [2, 3], [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        # [FUNCTION DOC] Checks that a generator of arrays is accepted without error (no
        # warning is checked, unlike newer numpy).
        # numpy 1.24 emits a warning but we don't
        # with assert_warns(FutureWarning):
        column_stack([np.arange(3) for _ in range(2)])


class TestDstack(TestCase):
    # [CLASS DOC] Tests for dstack (stack along the third axis).
    def test_non_iterable(self):
        # [FUNCTION DOC] Checks passing a non-iterable raises TypeError.
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        # [FUNCTION DOC] Checks two 0-d arrays stack into shape 1x1x2.
        a = np.array(1)
        b = np.array(2)
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] Checks two length-one 1-D arrays stack into shape 1x1x2.
        a = np.array([1])
        b = np.array([2])
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Checks two column vectors of two rows stack into shape 2x1x2.
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        res = dstack([a, b])
        desired = np.array(
            [
                [[1, 1]],
                [
                    [
                        2,
                        2,
                    ]
                ],
            ]
        )
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        # [FUNCTION DOC] Checks two length-two 1-D arrays stack into shape 1x2x2.
        a = np.array([1, 2])
        b = np.array([1, 2])
        res = dstack([a, b])
        desired = np.array([[[1, 1], [2, 2]]])
        assert_array_equal(res, desired)

    def test_generator(self):
        # [FUNCTION DOC] Checks a generator of arrays is accepted without error.
        # numpy 1.24 emits a warning but we don't
        # with assert_warns(FutureWarning):
        dstack([np.arange(3) for _ in range(2)])


# array_split has more comprehensive test of splitting.
# only do simple test on hsplit, vsplit, and dsplit
class TestHsplit(TestCase):
    # [CLASS DOC] Tests for hsplit with integer section counts only.
    """Only testing for integer splits."""

    def test_non_iterable(self):
        # [FUNCTION DOC] Checks a plain scalar input raises ValueError.
        assert_raises(ValueError, hsplit, 1, 1)

    def test_0D_array(self):
        # [FUNCTION DOC] Checks a 0-d array raises ValueError (the test fails if no error is
        # raised).
        a = np.array(1)
        try:
            hsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_1D_array(self):
        # [FUNCTION DOC] Checks a 1-D array of four is split into two halves.
        a = np.array([1, 2, 3, 4])
        res = hsplit(a, 2)
        desired = [np.array([1, 2]), np.array([3, 4])]
        compare_results(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Checks a 2x4 array is split into two 2x2 pieces along columns.
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        res = hsplit(a, 2)
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        compare_results(res, desired)


class TestVsplit(TestCase):
    # [CLASS DOC] Tests for vsplit with integer section counts only.
    """Only testing for integer splits."""

    def test_non_iterable(self):
        # [FUNCTION DOC] Checks a plain scalar input raises ValueError.
        assert_raises(ValueError, vsplit, 1, 1)

    def test_0D_array(self):
        # [FUNCTION DOC] Checks a 0-d array raises ValueError.
        a = np.array(1)
        assert_raises(ValueError, vsplit, a, 2)

    def test_1D_array(self):
        # [FUNCTION DOC] Checks a 1-D array raises ValueError (the test fails if no error is
        # raised).
        a = np.array([1, 2, 3, 4])
        try:
            vsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_2D_array(self):
        # [FUNCTION DOC] Checks a 2x4 array is split into two single-row pieces.
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        res = vsplit(a, 2)
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        compare_results(res, desired)


class TestDsplit(TestCase):
    # [CLASS DOC] Tests for dsplit with integer section counts only.
    # Only testing for integer splits.
    def test_non_iterable(self):
        # [FUNCTION DOC] Checks a plain scalar input raises ValueError.
        assert_raises(ValueError, dsplit, 1, 1)

    def test_0D_array(self):
        # [FUNCTION DOC] Checks a 0-d array raises ValueError.
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    def test_1D_array(self):
        # [FUNCTION DOC] Checks a 1-D array raises ValueError.
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    def test_2D_array(self):
        # [FUNCTION DOC] Checks a 2-D array raises ValueError (the test fails if no error is
        # raised).
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        try:
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        # [FUNCTION DOC] Checks a 2x2x4 array is split into two 2x2x2 pieces along the last
        # axis.
        a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        res = dsplit(a, 2)
        desired = [
            np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
            np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]]),
        ]
        compare_results(res, desired)


class TestSqueeze(TestCase):
    # [CLASS DOC] Tests for squeeze.
    def test_basic(self):
        # [FUNCTION DOC] Checks squeezing random arrays equals reshaping to the non-unit
        # dimensions, and that squeezing a nested list to a single value gives a 0-d array of
        # the array type.
        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        if type(res) is not np.ndarray:
            raise AssertionError(f"Expected type(res) is np.ndarray, got {type(res)}")

    @xfailIfTorchDynamo
    def test_basic_2(self):
        # [FUNCTION DOC] Checks that squeezing returns a view sharing the same base tensor as
        # the original (an implementation detail of torch._numpy). Marked expected failure
        # under dynamo.
        aa = np.ones((3, 1, 4, 1, 1))
        if aa.squeeze().tensor._base is not aa.tensor:
            raise AssertionError("Expected aa.squeeze().tensor._base is aa.tensor")

    def test_squeeze_axis(self):
        # [FUNCTION DOC] Checks squeeze with no axis, an empty axis tuple (no change) and
        # single axes including negative ones, on arrays and nested lists, via the resulting
        # shapes.
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(A, axis=()), A)

        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_squeeze_type(self):
        # [FUNCTION DOC] Regression test (ticket 133): squeezing a length-one array and a 0-d
        # array both return an ndarray.
        # Ticket #133
        a = np.array([3])
        b = np.array(3)
        if type(a.squeeze()) is not np.ndarray:
            raise AssertionError(
                f"Expected type(a.squeeze()) is np.ndarray, got {type(a.squeeze())}"
            )
        if type(b.squeeze()) is not np.ndarray:
            raise AssertionError(
                f"Expected type(b.squeeze()) is np.ndarray, got {type(b.squeeze())}"
            )

    @skip(reason="XXX: order='F' not implemented")
    def test_squeeze_contiguous(self):
        # [FUNCTION DOC] Checks contiguity flags of squeezed results for C and Fortran-ordered
        # inputs (similar to issue 387). Always skipped because Fortran order is not
        # implemented.
        # Similar to GitHub issue #387
        a = np.zeros((1, 2)).squeeze()
        b = np.zeros((2, 2, 2), order="F")[:, :, ::2].squeeze()
        assert_(a.flags.c_contiguous)
        assert_(a.flags.f_contiguous)
        assert_(b.flags.f_contiguous)

    @xpassIfTorchDynamo_np  # (reason="XXX: noop in torch, while numpy raises")
    def test_squeeze_axis_handling(self):
        # [FUNCTION DOC] Checks squeezing an axis of size not one raises ValueError. Marked an
        # expected failure outside dynamo because torch makes it a no-op while numpy raises.
        with assert_raises(ValueError):
            np.squeeze(np.array([[1], [2], [3]]), axis=0)


@instantiate_parametrized_tests
class TestKron(TestCase):
    # [CLASS DOC] Tests for kron (Kronecker product).
    def test_basic(self):
        # [FUNCTION DOC] Checks the product with 0-d, 1-D and 3-D operands on either side
        # against the expected nested results.
        # Using 0-dimensional ndarray
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        assert_array_equal(np.kron(a, b), k)

        # Using 1-dimensional ndarray
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        assert_array_equal(np.kron(a, b), k)

        # Using 3-dimensional ndarray
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)

    @skip(reason="NP_VER: fails on CI")
    @parametrize(
        "shape_a,shape_b",
        [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ],
    )
    def test_kron_shape(self, shape_a, shape_b):
        # [FUNCTION DOC] Parametrized over operand shape pairs (including zero-sized and
        # mismatched rank): the result shape equals the elementwise product of shapes after
        # left-padding the shorter one with ones. Raises AssertionError on mismatch. Always
        # skipped (fails on CI).
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)

        k = np.kron(a, b)
        if not np.array_equal(k.shape, expected_shape):
            raise AssertionError(
                f"Unexpected shape from kron: expected {expected_shape}, got {k.shape}"
            )


class TestTile(TestCase):
    # [CLASS DOC] Tests for tile.
    def test_basic(self):
        # [FUNCTION DOC] Checks repetition counts as an int or tuple on a 1-D array and a
        # nested list.
        a = np.array([0, 1, 2])
        b = [[1, 2], [3, 4]]
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(
            tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
        )

    def test_tile_one_repetition_on_array_gh4679(self):
        # [FUNCTION DOC] Regression test (gh4679): tiling once and modifying the result in
        # place must not change the original array.
        a = np.arange(5)
        b = tile(a, 1)
        b += 2
        assert_equal(a, np.arange(5))

    def test_empty(self):
        # [FUNCTION DOC] Checks tiling empty arrays gives the expected zero-size shapes.
        a = np.array([[[]]])
        b = np.array([[], []])
        c = tile(b, 2).shape
        d = tile(a, (3, 2, 5)).shape
        assert_equal(c, (2, 0))
        assert_equal(d, (3, 2, 0))

    def test_kroncompare(self):
        # [FUNCTION DOC] For several shapes of random integer arrays and repetition tuples,
        # checks tile equals the Kronecker product with an array of ones of the repetition
        # shape.
        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        for s in shape:
            b = randint(0, 10, size=s)
            for r in reps:
                a = np.ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)


@xfail  # Maybe implement one day
class TestMayShareMemory(TestCase):
    # [CLASS DOC] Tests for may_share_memory; marked an expected failure as it is not
    # implemented.
    def test_basic(self):
        # [FUNCTION DOC] Checks views of one array share memory and slices of unrelated arrays
        # do not.
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))

        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res, desired):
    # [FUNCTION DOC] HELPER: Compare two lists of arrays. Raises ValueError if lengths differ,
    # otherwise asserts each pair is array-equal.
    """Compare lists of arrays."""
    if len(res) != len(desired):
        raise ValueError("Iterables have different lengths")
    # See also PEP 618 for Python 3.10
    for x, y in zip(res, desired):
        assert_array_equal(x, y)


if __name__ == "__main__":
    run_tests()
