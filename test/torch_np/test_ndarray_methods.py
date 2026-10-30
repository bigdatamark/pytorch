# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests of ndarray methods and related functions in torch._numpy: indexing and
# setitem, reshape, transpose, ravel, nonzero, argmax and argmin (keepdims, out, ties, NaN,
# signed-integer extremes), amax and amin, membership tests and iteration. Uses real NumPy
# under TorchDynamo. Several tests inspect the wrapped tensor attribute and are skipped there.
# Import time builds the large module-level data tables of argmax and argmin cases inside the
# test classes.

import itertools
from unittest import expectedFailure as xfail, skipIf as skipif

import numpy
import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


class TestIndexing(TestCase):
    # [CLASS DOC] Basic element indexing and assignment on ndarray.
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attr, type of a[0, 0]")
    def test_indexing_simple(self):
        # [FUNCTION DOC] Skipped under TorchDynamo (uses the tensor attribute). Checks that
        # integer indexing returns an ndarray (zero-dimensional for a full index) and that a
        # row view shares storage with the original (its base is the original tensor).
        a = np.array([[1, 2, 3], [4, 5, 6]])

        if not isinstance(a[0, 0], np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(a[0, 0])}")
        if not isinstance(a[0, :], np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(a[0, :])}")
        if a[0, :].tensor._base is not a.tensor:
            raise AssertionError("Expected tensor._base to be a.tensor")

    def test_setitem(self):
        # [FUNCTION DOC] Assigns to one element of a two-by-three array and checks the array
        # is still an ndarray with the updated contents.
        a = np.array([[1, 2, 3], [4, 5, 6]])
        a[0, 0] = 8
        if not isinstance(a, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(a)}")
        assert_equal(a, [[8, 2, 3], [4, 5, 6]])


class TestReshape(TestCase):
    # [CLASS DOC] Tests for reshape as a function and as a method.
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_reshape_function(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Reshapes a nested list to two by six and
        # checks values; then checks a transposed ndarray is a view of the original tensor.
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        if not np.all(np.reshape(arr, (2, 6)) == tgt):
            raise AssertionError("reshape result does not match expected")

        arr = np.asarray(arr)
        if np.transpose(arr, (1, 0)).tensor._base is not arr.tensor:
            raise AssertionError("Expected transpose tensor._base to be arr.tensor")

    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_reshape_method(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Checks the reshape method with separate
        # integer arguments and with a shape tuple, to two by six and three by four: values
        # match, the result shares its base tensor with the source, and the source shape is
        # unchanged.
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        arr_shape = arr.shape

        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

        # reshape(*shape_tuple)
        if not np.all(arr.reshape(2, 6) == tgt):
            raise AssertionError("reshape(2, 6) result does not match expected")
        if arr.reshape(2, 6).tensor._base is not arr.tensor:  # reshape keeps the base
            raise AssertionError("Expected reshape tensor._base to be arr.tensor")
        if arr.shape != arr_shape:  # arr is intact
            raise AssertionError(f"Expected shape {arr_shape}, got {arr.shape}")

        # XXX: move out to dedicated test(s)
        if arr.reshape(2, 6).tensor._base is not arr.tensor:
            raise AssertionError("Expected reshape tensor._base to be arr.tensor")

        # reshape(shape_tuple)
        if not np.all(arr.reshape((2, 6)) == tgt):
            raise AssertionError("reshape((2, 6)) result does not match expected")
        if arr.reshape((2, 6)).tensor._base is not arr.tensor:
            raise AssertionError("Expected reshape tensor._base to be arr.tensor")
        if arr.shape != arr_shape:
            raise AssertionError(f"Expected shape {arr_shape}, got {arr.shape}")

        tgt = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        if not np.all(arr.reshape(3, 4) == tgt):
            raise AssertionError("reshape(3, 4) result does not match expected")
        if arr.reshape(3, 4).tensor._base is not arr.tensor:
            raise AssertionError("Expected reshape tensor._base to be arr.tensor")
        if arr.shape != arr_shape:
            raise AssertionError(f"Expected shape {arr_shape}, got {arr.shape}")

        if not np.all(arr.reshape((3, 4)) == tgt):
            raise AssertionError("reshape((3, 4)) result does not match expected")
        if arr.reshape((3, 4)).tensor._base is not arr.tensor:
            raise AssertionError("Expected reshape tensor._base to be arr.tensor")
        if arr.shape != arr_shape:
            raise AssertionError(f"Expected shape {arr_shape}, got {arr.shape}")


# XXX : order='C' / 'F'
#        tgt = [[1, 4, 7, 10],
#               [2, 5, 8, 11],
#               [3, 6, 9, 12]]
#        assert np.all(arr.T.reshape((3, 4), order='C') == tgt)
#
#        tgt = [[1, 10, 8, 6], [4, 2, 11, 9], [7, 5, 3, 12]]
#        assert_equal(arr.reshape((3, 4), order='F'), tgt)
#


class TestTranspose(TestCase):
    # [CLASS DOC] Tests for transpose as a function and as a method.
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_transpose_function(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Checks transpose with explicit axes on a
        # list and that for an ndarray the result is a view of the original tensor.
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

        arr = np.asarray(arr)
        if np.transpose(arr, (1, 0)).tensor._base is not arr.tensor:
            raise AssertionError("Expected transpose tensor._base to be arr.tensor")

    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_transpose_method(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Checks the transpose method with no
        # argument or none gives the swapped array, wrong axes arguments raise RuntimeError or
        # ValueError, and the result is a view of the original tensor.
        a = np.array([[1, 2], [3, 4]])
        assert_equal(a.transpose(), [[1, 3], [2, 4]])
        assert_equal(a.transpose(None), [[1, 3], [2, 4]])
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 0))
        assert_raises((RuntimeError, ValueError), lambda: a.transpose(0, 1, 2))

        if a.transpose().tensor._base is not a.tensor:
            raise AssertionError("Expected transpose tensor._base to be a.tensor")


class TestRavel(TestCase):
    # [CLASS DOC] Tests for ravel as a function and as a method.
    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_ravel_function(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Checks ravel of a nested list gives the
        # flat values, and for a contiguous ndarray the result shares its base tensor with the
        # source.
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

        arr = np.asarray(a)
        if np.ravel(arr).tensor._base is not arr.tensor:
            raise AssertionError("Expected ravel tensor._base to be arr.tensor")

    @skipif(TEST_WITH_TORCHDYNAMO, reason=".tensor attribute")
    def test_ravel_method(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Checks the ravel method values and that
        # the result is a view of the source tensor.
        a = np.array([[0, 1], [2, 3]])
        assert_equal(a.ravel(), [0, 1, 2, 3])

        if a.ravel().tensor._base is not a.tensor:
            raise AssertionError("Expected ravel tensor._base to be a.tensor")


class TestNonzero(TestCase):
    # [CLASS DOC] Tests for nonzero as a function and as a method.
    def test_nonzero_trivial(self):
        # [FUNCTION DOC] Checks nonzero (function and method) of empty, single zero and single
        # non-zero arrays, returning a one-tuple of index lists.
        assert_equal(np.nonzero(np.array([])), ([],))
        assert_equal(np.array([]).nonzero(), ([],))

        assert_equal(np.nonzero(np.array([0])), ([],))
        assert_equal(np.array([0]).nonzero(), ([],))

        assert_equal(np.nonzero(np.array([1])), ([0],))
        assert_equal(np.array([1]).nonzero(), ([0],))

    def test_nonzero_onedim(self):
        # [FUNCTION DOC] Checks nonzero indices of a one-dimensional array, by function and
        # method.
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))
        assert_equal(x.nonzero(), ([0, 2, 3, 6],))

    def test_nonzero_twodim(self):
        # [FUNCTION DOC] Checks nonzero of a two-dimensional array and of the three-by-three
        # identity matrix, returning a tuple of row indices and column indices, by function
        # and method.
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))
        assert_equal(x.nonzero(), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))
        assert_equal(x.nonzero(), ([0, 1, 2], [0, 1, 2]))

    def test_sparse(self):
        # [FUNCTION DOC] Checks nonzero of sparse boolean arrays with periodic true entries
        # and runs of true values, for twenty offsets, against computed expected indices.
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))
            assert_equal(c.nonzero()[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i : 20 + i] = True
            c[20 + i * 2] = True
            assert_equal(
                np.nonzero(c)[0],
                np.concatenate((np.arange(10 + i, 20 + i), [20 + i * 2])),
            )

    def test_array_method(self):
        # [FUNCTION DOC] Checks the nonzero method of a two-by-three array gives the expected
        # row and column index lists.
        # Tests that the array method
        # call to nonzero works
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)


@instantiate_parametrized_tests
class TestArgmaxArgminCommon(TestCase):
    # [CLASS DOC] Argmax and argmin tests shared by both methods. A class attribute lists the
    # input shapes (including a scalar shape and several higher-dimensional ones).
    sizes = [
        (),
        (3,),
        (3, 2),
        (2, 3),
        (3, 3),
        (2, 3, 4),
        (4, 3, 2),
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
        (64,),
        (128,),
        (256,),
    ]

    @skipif(numpy.__version__ < "1.22", reason="NP_VER: fails on NumPy 1.21.x")
    @parametrize(
        "size, axis",
        list(
            itertools.chain(
                *[
                    [
                        (size, axis)
                        for axis in list(range(-len(size), len(size))) + [None]
                    ]
                    for size in sizes
                ]
            )
        ),
    )
    @parametrize("method", [np.argmax, np.argmin])
    def test_np_argmin_argmax_keepdims(self, size, axis, method):
        # [FUNCTION DOC] For each shape, axis (including none and negative values) and for
        # both argmax and argmin, checks keepdims on a contiguous and on a transposed input:
        # the result equals the reshaped plain result and has the reduced axis of size one, an
        # out array is returned and filled, and an out array of wrong shape (or of lower
        # dimension) raises ValueError.
        # arr = np.random.normal(size=size)
        arr = np.empty(shape=size)

        # contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        if res.shape != new_shape:
            raise AssertionError(f"Expected shape {new_shape}, got {res.shape}")

        outarray = np.empty(res.shape, dtype=res.dtype)
        res1 = method(arr, axis=axis, out=outarray, keepdims=True)
        if res1 is not outarray:
            raise AssertionError("Expected res1 to be outarray")
        assert_equal(res, outarray)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

        # non-contiguous arrays
        if axis is None:
            new_shape = [1 for _ in range(len(size))]
        else:
            new_shape = list(size)[::-1]
            new_shape[axis] = 1
        new_shape = tuple(new_shape)

        _res_orig = method(arr.T, axis=axis)
        res_orig = _res_orig.reshape(new_shape)
        res = method(arr.T, axis=axis, keepdims=True)
        assert_equal(res, res_orig)
        if res.shape != new_shape:
            raise AssertionError(f"Expected shape {new_shape}, got {res.shape}")
        outarray = np.empty(new_shape[::-1], dtype=res.dtype)
        outarray = outarray.T
        res1 = method(arr.T, axis=axis, out=outarray, keepdims=True)
        if res1 is not outarray:
            raise AssertionError("Expected res1 to be outarray")
        assert_equal(res, outarray)

        if len(size) > 0:
            # one dimension lesser for non-zero sized
            # array should raise an error
            with pytest.raises(ValueError):
                method(arr[0], axis=axis, out=outarray, keepdims=True)

        if len(size) > 0:
            wrong_shape = list(new_shape)
            if axis is not None:
                wrong_shape[axis] = 2
            else:
                wrong_shape[0] = 2
            wrong_outarray = np.empty(wrong_shape, dtype=res.dtype)
            with pytest.raises(ValueError):
                method(arr.T, axis=axis, out=wrong_outarray, keepdims=True)

    @skipif(True, reason="XXX: need ndarray.chooses")
    @parametrize("method", ["max", "min"])
    def test_all(self, method):
        # [FUNCTION DOC] Skipped unconditionally because it needs an ndarray choose method.
        # When enabled it would check that the maximum or minimum along each axis equals
        # picking the argmax or argmin result with choose.
        # a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        a = np.arange(4 * 5 * 6 * 7 * 8).reshape((4, 5, 6, 7, 8))
        arg_method = getattr(a, "arg" + method)
        val_method = getattr(a, method)
        for i in range(a.ndim):
            a_maxmin = val_method(i)
            aarg_maxmin = arg_method(i)
            axes = list(range(a.ndim))
            axes.remove(i)
            if not np.all(a_maxmin == aarg_maxmin.choose(*a.transpose(i, *axes))):
                raise AssertionError("maxmin values do not match choose result")

    @parametrize("method", ["argmax", "argmin"])
    def test_output_shape(self, method):
        # [FUNCTION DOC] Checks the out shape rules of the argmax and argmin methods on a ten
        # by five array reducing the last axis: out arrays of size eleven, shape two by five
        # and shape one by ten raise ValueError, while an array of size ten is filled with the
        # plain result.
        # see also gh-616
        a = np.ones((10, 5))
        arg_method = getattr(a, method)

        # Check some simple shape mismatches
        out = np.ones(11, dtype=np.int_)

        with assert_raises(ValueError):
            arg_method(-1, out=out)

        out = np.ones((2, 5), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        # these could be relaxed possibly (used to allow even the previous)
        out = np.ones((1, 10), dtype=np.int_)
        with assert_raises(ValueError):
            arg_method(-1, out=out)

        out = np.ones(10, dtype=np.int_)
        arg_method(-1, out=out)
        assert_equal(out, arg_method(-1))

    @parametrize("ndim", [0, 1])
    @parametrize("method", ["argmax", "argmin"])
    def test_ret_is_out(self, ndim, method):
        # [FUNCTION DOC] Checks that with the axis and out arguments the method returns the
        # very out object, for a one- and a two-dimensional reduction result of the given
        # method.
        a = np.ones((4,) + (256,) * ndim)
        arg_method = getattr(a, method)
        out = np.empty((256,) * ndim, dtype=np.intp)
        ret = arg_method(axis=0, out=out)
        if ret is not out:
            raise AssertionError("Expected ret to be out")

    @parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    def test_np_vs_ndarray(self, arr_method, np_method):
        # [FUNCTION DOC] Checks that the ndarray method and the module function for argmax or
        # argmin give equal results and fill equal out arrays when axis and out are passed by
        # keyword.
        # make sure both ndarray.argmax/argmin and
        # numpy.argmax/argmin support out/axis args
        # a = np.random.normal(size=(2, 3))
        a = np.arange(6).reshape((2, 3))
        arg_method = getattr(a, arr_method)

        # check keyword args
        out1 = np.zeros(3, dtype=int)
        out2 = np.zeros(3, dtype=int)
        assert_equal(arg_method(out=out1, axis=0), np_method(a, out=out2, axis=0))
        assert_equal(out1, out2)

    @parametrize(
        "arr_method, np_method", [("argmax", np.argmax), ("argmin", np.argmin)]
    )
    def test_np_vs_ndarray_positional(self, arr_method, np_method):
        # [FUNCTION DOC] Same as the keyword variant but passes the axis and out arguments
        # positionally to both the method and the function.
        a = np.arange(6).reshape((2, 3))
        arg_method = getattr(a, arr_method)

        # check positional args
        out1 = np.zeros(2, dtype=int)
        out2 = np.zeros(2, dtype=int)
        assert_equal(arg_method(1, out1), np_method(a, 1, out2))
        assert_equal(out1, out2)


@instantiate_parametrized_tests
class TestArgmax(TestCase):
    # [CLASS DOC] Argmax tests. Class attributes build tables of arrays with expected index of
    # the first maximum: unsigned, signed, float with NaN (NaN counts as maximum), complex
    # with NaN (marked to pass unexpectedly under TorchDynamo) and boolean cases.
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 0),
        ([3, 3, 3, 3, 2, 2, 2, 2], 0),
        ([0, 1, 2, 3, 4, 5, 6, 7], 7),
        ([7, 6, 5, 4, 3, 2, 1, 0], 0),
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 3),
        ([1, 2, 3, 4, -1, -2, -3, -4], 3),
    ]
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widths
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    nan_arr = darr + [
        subtest(
            ([0, 1, 2, 3, complex(0, np.nan)], 4), decorators=[xpassIfTorchDynamo_np]
        ),
        subtest(
            ([0, 1, 2, 3, complex(np.nan, 0)], 4), decorators=[xpassIfTorchDynamo_np]
        ),
        subtest(
            ([0, 1, 2, complex(np.nan, 0), 3], 3), decorators=[xpassIfTorchDynamo_np]
        ),
        subtest(
            ([0, 1, 2, complex(0, np.nan), 3], 3), decorators=[xpassIfTorchDynamo_np]
        ),
        subtest(
            ([complex(0, np.nan), 0, 1, 2, 3], 0), decorators=[xpassIfTorchDynamo_np]
        ),
        subtest(
            ([complex(np.nan, np.nan), 0, 1, 2, 3], 0),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(0, 0), complex(0, 2), complex(0, 1)], 1),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(1, 0), complex(0, 2), complex(0, 1)], 0),
            decorators=[xpassIfTorchDynamo_np],
        ),
        subtest(
            ([complex(1, 0), complex(0, 2), complex(1, 1)], 2),
            decorators=[xpassIfTorchDynamo_np],
        ),
        ([False, False, False, False, True], 4),
        ([False, False, False, True, False], 3),
        ([True, False, False, False, False], 0),
        ([True, False, True, False, False], 0),
    ]

    @parametrize("data", nan_arr)
    def test_combinations(self, data):
        # [FUNCTION DOC] For each array and expected position, checks argmax matches the
        # position and the element there equals the maximum; repeats with every element
        # repeated 129 times and with 513 copies of the minimum appended, to exercise
        # vectorized loops.
        arr, pos = data
        #      with suppress_warnings() as sup:
        #          sup.filter(RuntimeWarning,
        #                      "invalid value encountered in reduce")
        #        if np.asarray(arr).dtype.kind in "c":
        #            pytest.xfail(reason="'max_values_cpu' not implemented for 'ComplexDouble'")

        val = np.max(arr)

        assert_equal(np.argmax(arr), pos)  # , err_msg="%r" % arr)
        assert_equal(arr[np.argmax(arr)], val)  # , err_msg="%r" % arr)

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")

        padding = np.repeat(np.min(arr), 513)
        rarr = np.concatenate((arr, padding))
        rpos = pos
        assert_equal(np.argmax(rarr), rpos, err_msg=f"{rarr!r}")
        assert_equal(rarr[np.argmax(rarr)], val, err_msg=f"{rarr!r}")

    def test_maximum_signed_integers(self):
        # [FUNCTION DOC] Checks that argmax finds the middle element at the maximum for int8,
        # int16, int32 and int64 arrays containing the extreme values of the type.
        a = np.array([1, 2**7 - 1, -(2**7)], dtype=np.int8)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**15 - 1, -(2**15)], dtype=np.int16)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**31 - 1, -(2**31)], dtype=np.int32)
        assert_equal(np.argmax(a), 1)

        a = np.array([1, 2**63 - 1, -(2**63)], dtype=np.int64)
        assert_equal(np.argmax(a), 1)


@instantiate_parametrized_tests
class TestArgmin(TestCase):
    # [CLASS DOC] Argmin tests. Class attributes build tables of arrays with expected index of
    # the first minimum: unsigned, signed, float with NaN, complex with NaN (marked expected
    # failures) and boolean cases.
    usg_data = [
        ([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 8),
        ([3, 3, 3, 3, 2, 2, 2, 2], 4),
        ([0, 1, 2, 3, 4, 5, 6, 7], 0),
        ([7, 6, 5, 4, 3, 2, 1, 0], 7),
    ]
    sg_data = usg_data + [
        ([1, 2, 3, 4, -4, -3, -2, -1], 4),
        ([1, 2, 3, 4, -1, -2, -3, -4], 7),
    ]
    darr = [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (itertools.product(usg_data, (np.uint8,)))
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                sg_data, (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
            )
        )
    ]
    darr += [
        (np.array(d[0], dtype=t), d[1])
        for d, t in (
            itertools.product(
                (
                    ([0, 1, 2, 3, np.nan], 4),
                    ([0, 1, 2, np.nan, 3], 3),
                    ([np.nan, 0, 1, 2, 3], 0),
                    ([np.nan, 0, np.nan, 2, 3], 0),
                    # To hit the tail of SIMD multi-level(x4, x1) inner loops
                    # on variant SIMD widths
                    ([1] * (2 * 5 - 1) + [np.nan], 2 * 5 - 1),
                    ([1] * (4 * 5 - 1) + [np.nan], 4 * 5 - 1),
                    ([1] * (8 * 5 - 1) + [np.nan], 8 * 5 - 1),
                    ([1] * (16 * 5 - 1) + [np.nan], 16 * 5 - 1),
                    ([1] * (32 * 5 - 1) + [np.nan], 32 * 5 - 1),
                ),
                (np.float32, np.float64),
            )
        )
    ]
    nan_arr = darr + [
        subtest(([0, 1, 2, 3, complex(0, np.nan)], 4), decorators=[xfail]),
        subtest(([0, 1, 2, 3, complex(np.nan, 0)], 4), decorators=[xfail]),
        subtest(([0, 1, 2, complex(np.nan, 0), 3], 3), decorators=[xfail]),
        subtest(([0, 1, 2, complex(0, np.nan), 3], 3), decorators=[xfail]),
        subtest(([complex(0, np.nan), 0, 1, 2, 3], 0), decorators=[xfail]),
        subtest(([complex(np.nan, np.nan), 0, 1, 2, 3], 0), decorators=[xfail]),
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xfail],
        ),
        subtest(
            ([complex(np.nan, np.nan), complex(np.nan, 2), complex(np.nan, 1)], 0),
            decorators=[xfail],
        ),
        subtest(
            ([complex(np.nan, 0), complex(np.nan, 2), complex(np.nan, np.nan)], 0),
            decorators=[xfail],
        ),
        subtest(([complex(0, 0), complex(0, 2), complex(0, 1)], 0), decorators=[xfail]),
        subtest(([complex(1, 0), complex(0, 2), complex(0, 1)], 2), decorators=[xfail]),
        subtest(([complex(1, 0), complex(0, 2), complex(1, 1)], 1), decorators=[xfail]),
        ([True, True, True, True, False], 4),
        ([True, True, True, False, True], 3),
        ([False, True, True, True, True], 0),
        ([False, True, False, True, True], 0),
    ]

    @parametrize("data", nan_arr)
    def test_combinations(self, data):
        # [FUNCTION DOC] Same strategy as the argmax variant using argmin, repeated elements
        # and appended maxima. Complex inputs are marked as an expected failure via pytest
        # because minimum is not implemented for complex values.
        arr, pos = data

        if np.asarray(arr).dtype.kind == "c":
            pytest.xfail(reason="'min_values_cpu' not implemented for 'ComplexDouble'")

        #        with suppress_warnings() as sup:
        #            sup.filter(RuntimeWarning, "invalid value encountered in reduce")
        min_val = np.min(arr)

        assert_equal(np.argmin(arr), pos, err_msg=f"{arr!r}")
        assert_equal(arr[np.argmin(arr)], min_val, err_msg=f"{arr!r}")

        # add padding to test SIMD loops
        rarr = np.repeat(arr, 129)
        rpos = pos * 129
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

        padding = np.repeat(np.max(arr), 513)
        rarr = np.concatenate((arr, padding))
        rpos = pos
        assert_equal(np.argmin(rarr), rpos, err_msg=f"{rarr!r}")
        assert_equal(rarr[np.argmin(rarr)], min_val, err_msg=f"{rarr!r}")

    def test_minimum_signed_integers(self):
        # [FUNCTION DOC] Checks that argmin finds the element at the minimum for int8, int16,
        # int32 and int64 arrays containing the extreme values of the type.
        a = np.array([1, -(2**7), -(2**7) + 1, 2**7 - 1], dtype=np.int8)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**15), -(2**15) + 1, 2**15 - 1], dtype=np.int16)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**31), -(2**31) + 1, 2**31 - 1], dtype=np.int32)
        assert_equal(np.argmin(a), 1)

        a = np.array([1, -(2**63), -(2**63) + 1, 2**63 - 1], dtype=np.int64)
        assert_equal(np.argmin(a), 1)


class TestAmax(TestCase):
    # [CLASS DOC] Tests for amax.
    def test_basic(self):
        # [FUNCTION DOC] Checks amax of a flat list and of a nested list along each axis, and
        # that it matches the max method.
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amax(a), 10.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amax(b, axis=0), [8.0, 10.0, 9.0])
        assert_equal(np.amax(b, axis=1), [9.0, 10.0, 8.0])

        arr = np.asarray(a)
        assert_equal(np.amax(arr), arr.max())


class TestAmin(TestCase):
    # [CLASS DOC] Tests for amin.
    def test_basic(self):
        # [FUNCTION DOC] Checks amin of a flat list and of a nested list along each axis, and
        # that it matches the min method.
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.amin(a), -5.0)
        b = [[3, 6.0, 9.0], [4, 10.0, 5.0], [8, 3.0, 2.0]]
        assert_equal(np.amin(b, axis=0), [3.0, 3.0, 2.0])
        assert_equal(np.amin(b, axis=1), [3.0, 4.0, 2.0])

        arr = np.asarray(a)
        assert_equal(np.amin(arr), arr.min())


class TestContains(TestCase):
    # [CLASS DOC] Tests for the membership operator on ndarray.
    def test_contains(self):
        # [FUNCTION DOC] Checks that a present number is found in a three-by-four array and an
        # absent number is not.
        a = np.arange(12).reshape(3, 4)
        if 2 not in a:
            raise AssertionError("Expected 2 to be in array")
        if 42 in a:
            raise AssertionError("Expected 42 to not be in array")


@instantiate_parametrized_tests
class TestNoExtraMethods(TestCase):
    # [CLASS DOC] Checks the ndarray does not expose extra attributes beyond NumPy's.
    # make sure ndarray does not carry extra methods/attributes
    # >>> set(dir(a)) - set(dir(a.tensor.numpy()))
    @parametrize("name", ["fn", "ivar", "method", "name", "plain", "rvar"])
    def test_extra_methods(self, name):
        # [FUNCTION DOC] Parametrized over some leftover implementation names; asserts that
        # reading each attribute of an ndarray raises AttributeError.
        a = np.ones(3)
        with pytest.raises(AttributeError):
            getattr(a, name)


class TestIter(TestCase):
    # [CLASS DOC] Tests for iterating over an ndarray.
    @skipIfTorchDynamo()
    def test_iter_1d(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Iterating a one-dimensional array yields
        # zero-dimensional ndarrays (unlike NumPy's array scalars); asserts the type and ndim
        # of each.
        # numpy generates array scalars, we do 0D arrays
        a = np.arange(5)
        lst = list(a)
        if not all(type(x) is np.ndarray for x in lst):
            raise AssertionError(
                f"Expected all np.ndarray, got {[type(x) for x in lst]}"
            )
        if not all(x.ndim == 0 for x in lst):
            raise AssertionError("Expected all elements to have ndim == 0")

    def test_iter_2d(self):
        # [FUNCTION DOC] Iterating a one-by-five array yields one item along the first axis,
        # which is an ndarray equal to the range of five.
        # numpy iterates over the 0th axis
        a = np.arange(5)[None, :]
        lst = list(a)
        if len(lst) != 1:
            raise AssertionError(f"Expected length 1, got {len(lst)}")
        # FIXME: "is" cannot be used here because dynamo fails
        if type(lst[0]) != np.ndarray:  # noqa: E721
            raise AssertionError(f"Expected np.ndarray, got {type(lst[0])}")
        assert_equal(lst[0], np.arange(5))


if __name__ == "__main__":
    run_tests()
