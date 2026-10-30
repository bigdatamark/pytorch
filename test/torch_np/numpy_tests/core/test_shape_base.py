# Owner(s): ["module: dynamo"]
# [MODULE DOC] Port of numpy shape manipulation tests (atleast_1d/2d/3d, hstack, vstack,
# concatenate, stack, block) to torch._numpy, or real numpy when running under dynamo. Import
# time selects the numpy implementation and the AxisError class and defines a skip helper. The
# block tests are expected to fail since block is not implemented.

import functools
from unittest import expectedFailure as xfail, skipIf as skipif

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
    from numpy import (
        array,
        atleast_1d,
        atleast_2d,
        atleast_3d,
        concatenate,
        hstack,
        newaxis,
        stack,
        vstack,
    )

    if int(numpy.__version__[0]) >= 2:
        from numpy.exceptions import AxisError
    else:
        from numpy import AxisError

    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import (
        array,
        atleast_1d,
        atleast_2d,
        atleast_3d,
        AxisError,
        concatenate,
        hstack,
        newaxis,
        stack,
        vstack,
    )
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal


skip = functools.partial(skipif, True)


IS_PYPY = False


class TestAtleast1d(TestCase):
    # [CLASS DOC] Tests that atleast_1d promotes scalars to one dimension and leaves higher
    # ranks alone.
    def test_0D_array(self):
        # [FUNCTION DOC] Zero dimensional inputs must come back as one element vectors.
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] One dimensional inputs must be returned with unchanged contents.
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Two dimensional inputs must be returned unchanged.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        # [FUNCTION DOC] Three dimensional inputs must be returned unchanged.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r1array(self):
        # [FUNCTION DOC] Checks result shapes for Python int, complex, float and nested list
        # inputs: one element vector for scalars, unchanged two by two shape for the nested
        # list.
        """Test to make sure equivalent Travis O's r1array function"""
        if atleast_1d(3).shape != (1,):
            raise AssertionError(f"shape mismatch: {atleast_1d(3).shape} != (1,)")
        if atleast_1d(3j).shape != (1,):
            raise AssertionError(f"shape mismatch: {atleast_1d(3j).shape} != (1,)")
        if atleast_1d(3.0).shape != (1,):
            raise AssertionError(f"shape mismatch: {atleast_1d(3.0).shape} != (1,)")
        if atleast_1d([[2, 3], [4, 5]]).shape != (2, 2):
            raise AssertionError(
                f"shape mismatch: {atleast_1d([[2, 3], [4, 5]]).shape} != (2, 2)"
            )


class TestAtleast2d(TestCase):
    # [CLASS DOC] Tests that atleast_2d promotes inputs to at least two dimensions.
    def test_0D_array(self):
        # [FUNCTION DOC] Zero dimensional inputs must become one by one arrays.
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] One dimensional inputs must become single row two dimensional arrays.
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Two dimensional inputs must be returned unchanged.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        # [FUNCTION DOC] Three dimensional inputs must be returned unchanged.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r2array(self):
        # [FUNCTION DOC] Checks result shapes for a scalar, a complex list and a nested three
        # dimensional list: one by one, one by two and unchanged two by two by two.
        """Test to make sure equivalent Travis O's r2array function"""
        if atleast_2d(3).shape != (1, 1):
            raise AssertionError(f"shape mismatch: {atleast_2d(3).shape} != (1, 1)")
        if atleast_2d([3j, 1]).shape != (1, 2):
            raise AssertionError(
                f"shape mismatch: {atleast_2d([3j, 1]).shape} != (1, 2)"
            )
        if atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape != (2, 2, 2):
            raise AssertionError(
                f"shape mismatch: {atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape} != (2, 2, 2)"
            )


class TestAtleast3d(TestCase):
    # [CLASS DOC] Tests that atleast_3d promotes inputs to at least three dimensions.
    def test_0D_array(self):
        # [FUNCTION DOC] Zero dimensional inputs must become one by one by one arrays.
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1]]]), array([[[2]]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] One dimensional inputs of length n must become shape one by n by one.
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Two dimensional inputs must gain a trailing axis of length one.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a[:, :, newaxis], b[:, :, newaxis]]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        # [FUNCTION DOC] Three dimensional inputs must be returned unchanged.
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)


class TestHstack(TestCase):
    # [CLASS DOC] Tests for horizontal stacking of arrays.
    def test_non_iterable(self):
        # [FUNCTION DOC] A non iterable argument must raise TypeError.
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        # [FUNCTION DOC] An empty sequence of arrays must raise ValueError.
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        # [FUNCTION DOC] Zero dimensional inputs are stacked into a one dimensional array.
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] One dimensional inputs are joined end to end.
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Two dimensional column inputs are joined along the second axis.
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        # [FUNCTION DOC] Smoke test that lists built from generator style expressions are
        # accepted without raising; numpy warns here but this implementation does not.
        # numpy 1.24 emits warnings but we don't
        # with assert_warns(FutureWarning):
        hstack([np.arange(3) for _ in range(2)])
        # with assert_warns(FutureWarning):
        hstack([x for x in np.ones((3, 2))])  # noqa: C416

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    def test_casting_and_dtype(self):
        # [FUNCTION DOC] Calls hstack with unsafe casting and an int64 dtype on a single
        # appended array and compares with the expected flattened integer result. Skipped on
        # numpy older than 1.24.
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack(np.append(a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)

    def test_casting_and_dtype_type_error(self):
        # [FUNCTION DOC] Safe casting from float to int64 in hstack must raise TypeError.
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)


class TestVstack(TestCase):
    # [CLASS DOC] Tests for vertical stacking of arrays.
    def test_non_iterable(self):
        # [FUNCTION DOC] A non iterable argument must raise TypeError.
        assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        # [FUNCTION DOC] An empty sequence of arrays must raise ValueError.
        assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        # [FUNCTION DOC] Zero dimensional inputs are stacked into a single column.
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        # [FUNCTION DOC] One element vectors are stacked into a column.
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        # [FUNCTION DOC] Two dimensional column inputs are stacked along the first axis.
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        # [FUNCTION DOC] One dimensional inputs become rows of a two dimensional result.
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    @xfail  # (reason="vstack w/generators")
    def test_generator(self):
        # [FUNCTION DOC] A generator argument is expected to raise TypeError mentioning arrays
        # to stack; marked as expected failure since this implementation accepts generators.
        with pytest.raises(TypeError, match="arrays to stack must be"):
            vstack(np.arange(3) for _ in range(2))

    @skipif(numpy.__version__ < "1.24", reason="casting kwarg is new in NumPy 1.24")
    def test_casting_and_dtype(self):
        # [FUNCTION DOC] Stacks an int and a float array with unsafe casting and an int64 dtype
        # and checks the truncated integer result. Skipped on numpy older than 1.24.
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)

    @skipif(numpy.__version__ < "1.24", reason="casting kwarg is new in NumPy 1.24")
    def test_casting_and_dtype_type_error(self):
        # [FUNCTION DOC] Safe casting from float to int64 in vstack must raise TypeError.
        # Skipped on numpy older than 1.24.
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            vstack((a, b), casting="safe", dtype=np.int64)


@instantiate_parametrized_tests
class TestConcatenate(TestCase):
    # [CLASS DOC] Tests for concatenate: argument validation, axis handling, out and dtype
    # options.
    def test_out_and_dtype_simple(self):
        # [FUNCTION DOC] Passing both an output array and a dtype must raise TypeError.
        # numpy raises TypeError on both out=... and dtype=...
        a, b, out = np.ones(3), np.ones(4), np.ones(3 + 4)

        with pytest.raises(TypeError):
            np.concatenate((a, b), out=out, dtype=float)

    def test_returns_copy(self):
        # [FUNCTION DOC] Concatenating one array must return a copy: changing the result must
        # not change the input.
        a = np.eye(3)
        b = np.concatenate([a])
        b[0, 0] = 2
        if b[0, 0] == a[0, 0]:
            raise AssertionError("concatenate should return a copy")

    def test_exceptions(self):
        # [FUNCTION DOC] Error cases: out of range axis (IndexError or AxisError), scalar
        # inputs, mismatched dimensionality, mismatched shapes along non-concatenation axes
        # (rotated through every axis) and an empty sequence (ValueError).
        # test axis must be in bounds
        for ndim in [1, 2, 3]:
            a = np.ones((1,) * ndim)
            np.concatenate((a, a), axis=0)  # OK
            assert_raises((IndexError, np.AxisError), np.concatenate, (a, a), axis=ndim)
            assert_raises(
                (IndexError, np.AxisError), np.concatenate, (a, a), axis=-(ndim + 1)
            )

        # Scalars cannot be concatenated
        assert_raises((RuntimeError, ValueError), concatenate, (0,))
        assert_raises((RuntimeError, ValueError), concatenate, (np.array(0),))

        # dimensionality must match
        assert_raises(
            (RuntimeError, ValueError),
            #        assert_raises_regex(
            #            ValueError,
            #            r"all the input arrays must have same number of dimensions, but "
            #            r"the array at index 0 has 1 dimension\(s\) and the array at "
            #            r"index 1 has 2 dimension\(s\)",
            np.concatenate,
            (np.zeros(1), np.zeros((1, 1))),
        )

        # test shapes must match except for concatenation axis
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for _ in range(3):
            np.concatenate((a, b), axis=axis[0])  # OK
            #            assert_raises_regex(
            assert_raises(
                (RuntimeError, ValueError),
                #                "all the input array dimensions except for the concatenation axis "
                #                "must match exactly, but along dimension {}, the array at "
                #                "index 0 has size 1 and the array at index 1 has size 2"
                #                .format(i),
                np.concatenate,
                (a, b),
                axis=axis[1],
            )
            assert_raises(
                (RuntimeError, ValueError), np.concatenate, (a, b), axis=axis[2]
            )
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # No arrays to concatenate raises ValueError
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        # [FUNCTION DOC] With axis none, inputs are flattened: result is one dimensional with
        # the same dtype and combined size; a given out array is returned as is and holds the
        # same values.
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))

        r = np.concatenate((a, a), axis=None)
        if r.dtype != a.dtype:
            raise AssertionError(f"dtype mismatch: {r.dtype} != {a.dtype}")
        if r.ndim != 1:
            raise AssertionError(f"ndim mismatch: {r.ndim} != 1")

        r = np.concatenate((a, b), axis=None)
        if r.size != a.size + len(b):
            raise AssertionError(f"size mismatch: {r.size} != {a.size + len(b)}")
        if r.dtype != a.dtype:
            raise AssertionError(f"dtype mismatch: {r.dtype} != {a.dtype}")

        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        if out is not rout:
            raise AssertionError("out should be rout")
        if not np.all(r == rout):
            raise AssertionError("r should equal rout")

    @xpassIfTorchDynamo_np  # (reason="concatenate(x, axis=None) relies on x being a sequence")
    def test_large_concatenate_axis_None(self):
        # [FUNCTION DOC] Concatenates the elements of a numpy array treated as a sequence with
        # axis none and with an axis of 100, expecting the flattened input. Expected to pass
        # only under dynamo numpy.
        # When no axis is given, concatenate uses flattened versions.
        # This also had a bug with many arrays (see gh-5979).
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        if not np.all(x == r):
            raise AssertionError("concatenate with axis=None should work")

        # This should probably be deprecated:
        r = np.concatenate(x, 100)  # axis is >= MAXDIMS
        assert_array_equal(x, r)

    def test_concatenate(self):
        # [FUNCTION DOC] Broad checks of concatenate on lists, tuples and arrays in one, two and
        # three dimensions with positive and negative axes, shape mismatch errors and the out
        # argument.
        # Test concatenate function
        # One sequence returns unmodified (but as array)

        # XXX: a single argument; relies on an ndarray being a sequence
        r4 = list(range(4))
        # assert_array_equal(concatenate((r4,)), r4)
        # # Any sequence
        # assert_array_equal(concatenate((tuple(r4),)), r4)
        # assert_array_equal(concatenate((array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # Explicit axis specification
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        assert_raises((RuntimeError, ValueError), concatenate, (a23.T, a13.T), 0)
        # 3D
        res = np.arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @skip(reason="concat, arrays, sequence")
    @skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    def test_operator_concat(self):
        # [FUNCTION DOC] Skipped: Python sequence concatenation operator applied to arrays or
        # mixed with lists or ints must raise TypeError.
        import operator

        a = array([1, 2])
        b = array([3, 4])
        n = [1, 2]
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        # [FUNCTION DOC] An out array of the wrong shape must raise ValueError, while a
        # correctly sized one is accepted.
        a = array([1, 2])
        b = array([3, 4])

        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4, 1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1, 4)))
        concatenate((a, b), out=np.empty(4))

    @parametrize("axis", [None, 0])
    @parametrize(
        "out_dtype", ["c8", "f4", "f8", "i8"]
    )  # torch does not have ">f8", "S4"
    @parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # [FUNCTION DOC] Parametrized over axis, output dtype and casting rule. Compares using
        # an out array against passing its dtype: when the cast is not allowed both must raise
        # TypeError, otherwise they must give equal results, the out array must be returned, and
        # the dtype must match. Giving both out and dtype must always raise TypeError.
        # Compare usage of `out=out` with `dtype=out.dtype`
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype, axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out, axis=axis, casting=casting)
            res_dtype = concatenate(
                to_concat, dtype=out.dtype, axis=axis, casting=casting
            )
            if res_out is not out:
                raise AssertionError("res_out should be out")
            assert_array_equal(out, res_dtype)
            if res_dtype.dtype != out_dtype:
                raise AssertionError(
                    f"dtype mismatch: {res_dtype.dtype} != {out_dtype}"
                )

        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)


@instantiate_parametrized_tests
class TestStackMisc(TestCase):
    # [CLASS DOC] Tests for stack.
    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    def test_stack(self):
        # [FUNCTION DOC] Covers non iterable and zero dimensional inputs, one and two
        # dimensional inputs with all valid axes, invalid axes (AxisError), empty arrays, the
        # out argument, shape and dimension mismatch errors, generator input, and unsafe or safe
        # casting with a dtype. Skipped on numpy older than 1.24.
        # non-iterable input
        assert_raises(TypeError, stack, 1)

        # 0d input
        for input_ in [
            (1, 2, 3),
            [np.int32(1), np.int32(2), np.int32(3)],
            [np.array(1), np.array(2), np.array(3)],
        ]:
            assert_array_equal(stack(input_), [1, 2, 3])
        # 1d input examples
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        r1 = array([[1, 2, 3], [4, 5, 6]])
        assert_array_equal(np.stack((a, b)), r1)
        assert_array_equal(np.stack((a, b), axis=1), r1.T)
        # all input types
        assert_array_equal(np.stack([a, b]), r1)
        assert_array_equal(np.stack(array([a, b])), r1)
        # all shapes for 1d input
        arrays = [np.random.randn(3) for _ in range(10)]
        axes = [0, 1, -1, -2]
        expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
        for axis, expected_shape in zip(axes, expected_shapes):
            assert_equal(np.stack(arrays, axis).shape, expected_shape)

        assert_raises(AxisError, stack, arrays, axis=2)
        assert_raises(AxisError, stack, arrays, axis=-3)

        # all shapes for 2d input
        arrays = [np.random.randn(3, 4) for _ in range(10)]
        axes = [0, 1, 2, -1, -2, -3]
        expected_shapes = [
            (10, 3, 4),
            (3, 10, 4),
            (3, 4, 10),
            (3, 4, 10),
            (3, 10, 4),
            (10, 3, 4),
        ]
        for axis, expected_shape in zip(axes, expected_shapes):
            assert_equal(np.stack(arrays, axis).shape, expected_shape)

        # empty arrays
        if stack([[], [], []]).shape != (3, 0):
            raise AssertionError(
                f"shape mismatch: {stack([[], [], []]).shape} != (3, 0)"
            )
        if stack([[], [], []], axis=1).shape != (0, 3):
            raise AssertionError(
                f"shape mismatch: {stack([[], [], []], axis=1).shape} != (0, 3)"
            )

        # out
        out = np.zeros_like(r1)
        np.stack((a, b), out=out)
        assert_array_equal(out, r1)

        # edge cases
        assert_raises(ValueError, stack, [])
        assert_raises(ValueError, stack, [])
        assert_raises((RuntimeError, ValueError), stack, [1, np.arange(3)])
        assert_raises((RuntimeError, ValueError), stack, [np.arange(3), 1])
        assert_raises((RuntimeError, ValueError), stack, [np.arange(3), 1], axis=1)
        assert_raises(
            (RuntimeError, ValueError), stack, [np.zeros((3, 3)), np.zeros(3)], axis=1
        )
        assert_raises((RuntimeError, ValueError), stack, [np.arange(2), np.arange(3)])

        # generator is deprecated: numpy 1.24 emits a warning but we don't
        # with assert_warns(FutureWarning):
        result = stack(x for x in range(3))

        assert_array_equal(result, np.array([0, 1, 2]))

        # casting and dtype test
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.stack((a, b), axis=1, casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2], [2, 3], [3, 4]])
        assert_array_equal(res, expected_res)

        # casting and dtype with TypeError
        with assert_raises(TypeError):
            stack((a, b), dtype=np.int64, axis=1, casting="safe")

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @parametrize("axis", [0])
    @parametrize("out_dtype", ["c8", "f4", "f8", "i8"])  # torch does not have ">f8",
    @parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
    def test_stack_out_and_dtype(self, axis, out_dtype, casting):
        # [FUNCTION DOC] Parametrized over output dtype and casting rule. When the cast is not
        # allowed a stack with a dtype must raise TypeError; otherwise stacking with out and
        # with dtype must agree, return the out array and have the requested dtype. Giving out
        # and dtype together must raise TypeError. Skipped on numpy older than 1.24.
        to_concat = (array([1, 2]), array([3, 4]))
        res = array([[1, 2], [3, 4]])
        out = np.zeros_like(res)

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
        else:
            res_out = stack(to_concat, out=out, axis=axis, casting=casting)
            res_dtype = stack(to_concat, dtype=out_dtype, axis=axis, casting=casting)
            if res_out is not out:
                raise AssertionError("res_out should be out")
            assert_array_equal(out, res_dtype)
            if res_dtype.dtype != out_dtype:
                raise AssertionError(
                    f"dtype mismatch: {res_dtype.dtype} != {out_dtype}"
                )

        with assert_raises(TypeError):
            stack(to_concat, out=out, dtype=out_dtype, axis=axis)


@xfail  # (reason="TODO: implement block(...)")
@instantiate_parametrized_tests
class TestBlock(TestCase):
    # [CLASS DOC] Tests for the block function, run through a fixture that selects the default,
    # forced concatenation or forced slicing algorithm. The whole class is an expected failure.
    # KNOWN ISSUE: names such as block, _block_setup, _block_concatenate, _block_slicing and
    # assert_raises_regex are not defined or imported in this module, so these tests cannot
    # work.
    @pytest.fixture(params=["block", "force_concatenate", "force_slicing"])
    def block(self, request):
        # [FUNCTION DOC] Pytest style fixture parametrized over three algorithm names; returns a
        # callable that runs the chosen block algorithm, raising ValueError for an unknown
        # parameter.
        # KNOWN ISSUE: the default branch returns the name block, which is not defined in scope.
        # blocking small arrays and large arrays go through different paths.
        # the algorithm is triggered depending on the number of element
        # copies required.
        # We define a test fixture that forces most tests to go through
        # both code paths.
        # Ultimately, this should be removed if a single algorithm is found
        # to be faster for both small and large arrays.
        def _block_force_concatenate(arrays):
            # [FUNCTION DOC] HELPER: sets up the nested list of arrays and then forces the
            # concatenation based algorithm; returns its result.
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            # [FUNCTION DOC] HELPER: sets up the nested list of arrays and then forces the
            # slicing based algorithm; returns its result.
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)

        if request.param == "force_concatenate":
            return _block_force_concatenate
        elif request.param == "force_slicing":
            return _block_force_slicing
        elif request.param == "block":
            return block
        else:
            raise ValueError("Unknown blocking request. There is a typo in the tests.")

    def test_returns_copy(self, block):
        # [FUNCTION DOC] Result of block on a single array must be a copy of the input.
        a = np.eye(3)
        b = block(a)
        b[0, 0] = 2
        if b[0, 0] == a[0, 0]:
            raise AssertionError("block should return a copy")

    def test_block_total_size_estimate(self, block):
        # [FUNCTION DOC] Checks the total element count estimate returned by the setup helper
        # for several nested list layouts.
        _, _, _, total_size = _block_setup([1])
        if total_size != 1:
            raise AssertionError(f"total_size mismatch: {total_size} != 1")

        _, _, _, total_size = _block_setup([[1]])
        if total_size != 1:
            raise AssertionError(f"total_size mismatch: {total_size} != 1")

        _, _, _, total_size = _block_setup([[1, 1]])
        if total_size != 2:
            raise AssertionError(f"total_size mismatch: {total_size} != 2")

        _, _, _, total_size = _block_setup([[1], [1]])
        if total_size != 2:
            raise AssertionError(f"total_size mismatch: {total_size} != 2")

        _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
        if total_size != 4:
            raise AssertionError(f"total_size mismatch: {total_size} != 4")

    def test_block_simple_row_wise(self, block):
        # [FUNCTION DOC] Two two by two arrays in one row are joined side by side.
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        desired = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        result = block([a_2d, b_2d])
        assert_equal(desired, result)

    def test_block_simple_column_wise(self, block):
        # [FUNCTION DOC] Two two by two arrays in separate rows are stacked vertically.
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        expected = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
        result = block([[a_2d], [b_2d]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_row_wise(self, block):
        # [FUNCTION DOC] One dimensional arrays in one row are joined end to end.
        # # # 1-D vectors are treated as row arrays
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([1, 2, 3, 2, 3, 4])
        result = block([a, b])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_multiple_rows(self, block):
        # [FUNCTION DOC] Two rows of one dimensional arrays produce two identical rows.
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3, 2, 3, 4], [1, 2, 3, 2, 3, 4]])
        result = block([[a, b], [a, b]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_column_wise(self, block):
        # [FUNCTION DOC] One dimensional arrays in separate rows are stacked as rows.
        # # # 1-D vectors are treated as row arrays
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3], [2, 3, 4]])
        result = block([[a_1d], [b_1d]])
        assert_equal(expected, result)

    def test_block_mixed_1d_and_2d(self, block):
        # [FUNCTION DOC] A two dimensional array and a one dimensional array in separate rows
        # are combined vertically.
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        result = block([[a_2d], [b_1d]])
        expected = np.array([[1, 1], [1, 1], [2, 2]])
        assert_equal(expected, result)

    def test_block_complicated(self, block):
        # [FUNCTION DOC] Mixed zero, one and two dimensional pieces arranged in five rows form a
        # six by six array.
        # a bit more complicated
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))

        expected = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
                [5, 6, 6, 6, 6, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        result = block(
            [[one_2d, two_2d], [three_2d], [four_1d], [five_0d, six_1d], [zero_2d]]
        )
        assert_equal(result, expected)

    def test_nested(self, block):
        # [FUNCTION DOC] A block call nested inside another block call gives the expected six by
        # six array.
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        result = block([[block([[one], [three], [four]]), two], [five, six], [zero]])
        expected = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 2, 2, 2],
                [4, 4, 4, 2, 2, 2],
                [5, 6, 6, 6, 6, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        assert_equal(result, expected)

    def test_3d(self, block):
        # [FUNCTION DOC] Assembles eight blocks of different shapes in a three level nested list
        # and compares with the expected five by five by five integer array.
        a000 = np.ones((2, 2, 2), int) * 1

        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4

        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7

        a111 = np.ones((3, 3, 3), int) * 8

        result = block(
            [
                [
                    [a000, a001],
                    [a010, a011],
                ],
                [
                    [a100, a101],
                    [a110, a111],
                ],
            ]
        )
        expected = array(
            [
                [
                    [1, 1, 4, 4, 4],
                    [1, 1, 4, 4, 4],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                ],
                [
                    [1, 1, 4, 4, 4],
                    [1, 1, 4, 4, 4],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                    [3, 3, 5, 5, 5],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
                [
                    [2, 2, 6, 6, 6],
                    [2, 2, 6, 6, 6],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                    [7, 7, 8, 8, 8],
                ],
            ]
        )

        assert_array_equal(result, expected)

    def test_block_with_mismatched_shape(self, block):
        # [FUNCTION DOC] Pieces whose shapes cannot be aligned must raise ValueError.
        a = np.array([0, 0])
        b = np.eye(2)
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])

        to_block = [
            [np.ones((2, 3)), np.ones((2, 2))],
            [np.ones((2, 2)), np.ones((2, 2))],
        ]
        assert_raises(ValueError, block, to_block)

    def test_no_lists(self, block):
        # [FUNCTION DOC] A scalar or plain array without lists is returned as an array equal to
        # the input.
        assert_equal(block(1), np.array(1))
        assert_equal(block(np.eye(3)), np.eye(3))

    def test_invalid_nesting(self, block):
        # [FUNCTION DOC] Inconsistently nested lists must raise ValueError mentioning mismatched
        # depths.
        msg = "depths are mismatched"
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(
            ValueError,
            msg,
            block,
            [[[1], [2]], [[3, 4]], [5]],  # missing brackets
        )

    def test_empty_lists(self, block):
        # [FUNCTION DOC] Empty lists at any level must raise ValueError mentioning empty.
        assert_raises_regex(ValueError, "empty", block, [])
        assert_raises_regex(ValueError, "empty", block, [[]])
        assert_raises_regex(ValueError, "empty", block, [[1], []])

    def test_tuple(self, block):
        # [FUNCTION DOC] Tuples used instead of lists must raise TypeError mentioning tuple.
        assert_raises_regex(TypeError, "tuple", block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, "tuple", block, [(1, 2), (3, 4)])

    def test_different_ndims(self, block):
        # [FUNCTION DOC] A float scalar, a two dimensional and a three dimensional array in a
        # flat list are promoted to a common rank and joined into a one by one by six result.
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        result = block([a, b, c])
        expected = np.array([[[1.0, 2.0, 2.0, 3.0, 3.0, 3.0]]])

        assert_equal(result, expected)

    def test_different_ndims_depths(self, block):
        # [FUNCTION DOC] Pieces of differing dimensionality at different list depths are joined
        # into a three row, single plane result.
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        result = block([[a, b], [c]])
        expected = np.array([[[1.0, 2.0, 2.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])

        assert_equal(result, expected)

    def test_block_memory_order(self, block):
        # [FUNCTION DOC] Blocks of C ordered inputs must give a C contiguous result and blocks
        # of Fortran ordered inputs a Fortran contiguous result, in three and two dimensions.
        # 3D
        arr_c = np.zeros((3,) * 3, order="C")
        arr_f = np.zeros((3,) * 3, order="F")

        b_c = [[[arr_c, arr_c], [arr_c, arr_c]], [[arr_c, arr_c], [arr_c, arr_c]]]

        b_f = [[[arr_f, arr_f], [arr_f, arr_f]], [[arr_f, arr_f], [arr_f, arr_f]]]

        if not block(b_c).flags["C_CONTIGUOUS"]:
            raise AssertionError("block(b_c) should be C_CONTIGUOUS")
        if not block(b_f).flags["F_CONTIGUOUS"]:
            raise AssertionError("block(b_f) should be F_CONTIGUOUS")

        arr_c = np.zeros((3, 3), order="C")
        arr_f = np.zeros((3, 3), order="F")
        # 2D
        b_c = [[arr_c, arr_c], [arr_c, arr_c]]

        b_f = [[arr_f, arr_f], [arr_f, arr_f]]

        if not block(b_c).flags["C_CONTIGUOUS"]:
            raise AssertionError("block(b_c) should be C_CONTIGUOUS")
        if not block(b_f).flags["F_CONTIGUOUS"]:
            raise AssertionError("block(b_f) should be F_CONTIGUOUS")


if __name__ == "__main__":
    run_tests()
