# Owner(s): ["module: dynamo"]

# [MODULE DOC] Smoke tests for the torch._numpy wrapper (imported as w): checks that groups of
#   functions, categorized by signature shape (one array in, array and axis, array and shape,
#   shape only, sequences of arrays, python scalar args), accept array-likes, torch tensors
#   and w.ndarray and return w.ndarray (or tuples, scalars) as appropriate. Also covers
#   argument normalization errors, copyto, divmod with out arguments, default dtype switching,
#   exported names versus real numpy, and nested construction. Import-time effect: builds the
#   module-level function lists and removes invert and bitwise_not from the shared unary ufunc
#   name list (mutating that list).
import functools
import inspect
from unittest import expectedFailure as xfail, skipIf as skip

import numpy as _np
from pytest import raises as assert_raises

import torch
import torch._numpy as w
import torch._numpy._ufuncs as _ufuncs
import torch._numpy._util as _util
from torch._numpy.testing import assert_allclose, assert_equal
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# These function receive one array_like arg and return one array_like result
one_arg_funcs = [
    w.asarray,
    w.empty_like,
    w.ones_like,
    w.zeros_like,
    functools.partial(w.full_like, fill_value=42),
    w.corrcoef,
    w.squeeze,
    w.argmax,
    # w.bincount,     # XXX: input dtypes
    w.prod,
    w.sum,
    w.real,
    w.imag,
    w.angle,
    w.real_if_close,
    w.isreal,
    w.iscomplex,
    w.isneginf,
    w.isposinf,
    w.i0,
    w.copy,
    w.array,
    w.round,
    w.around,
    w.flip,
    w.vstack,
    w.hstack,
    w.dstack,
    w.column_stack,
    w.row_stack,
    w.flatnonzero,
]

ufunc_names = _ufuncs._unary
ufunc_names.remove("invert")  # torch: bitwise_not_cpu not implemented for 'Float'
ufunc_names.remove("bitwise_not")

one_arg_funcs += [getattr(_ufuncs, name) for name in ufunc_names]


@instantiate_parametrized_tests
class TestOneArr(TestCase):
    # [CLASS DOC] Parametrized smoke tests applying every function in one_arg_funcs to a
    #   tensor, a nested list and a w.ndarray, expecting a w.ndarray result in all cases.
    """Base for smoke tests of one-arg functions: (array_like) -> (array_like)

    Accepts array_likes, torch.Tensors, w.ndarays; returns an ndarray
    """

    @parametrize("func", one_arg_funcs)
    def test_asarray_tensor(self, func):
        # [FUNCTION DOC] Applies the given function to a 2x3 float torch tensor and
        #   raises AssertionError unless the result is a w.ndarray.
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t)

        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_funcs)
    def test_asarray_list(self, func):
        # [FUNCTION DOC] Applies the given function to a nested list of floats and
        #   raises AssertionError unless the result is a w.ndarray.
        lst = [[1.0, 2, 3], [4, 5, 6]]
        la = func(lst)

        if not isinstance(la, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(la)}")

    @parametrize("func", one_arg_funcs)
    def test_asarray_array(self, func):
        # [FUNCTION DOC] Applies the given function to a w.ndarray built from nested
        #   floats and raises AssertionError unless the result is a w.ndarray.
        a = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        la = func(a)

        if not isinstance(la, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(la)}")


one_arg_axis_funcs = [
    w.argmax,
    w.argmin,
    w.prod,
    w.sum,
    w.all,
    w.any,
    w.mean,
    w.argsort,
    w.std,
    w.var,
    w.flip,
]


@instantiate_parametrized_tests
class TestOneArrAndAxis(TestCase):
    # [CLASS DOC] Parametrized smoke tests calling each function in one_arg_axis_funcs
    #   with axis values 0, 1, -1 and None on a tensor, a list and a w.ndarray; result
    #   must be a w.ndarray.
    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_tensor(self, func, axis):
        # [FUNCTION DOC] Calls the function with the axis keyword on a 2x3 tensor;
        #   raises AssertionError unless the result is a w.ndarray.
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_list(self, func, axis):
        # [FUNCTION DOC] Calls the function with the axis keyword on a nested list;
        #   raises AssertionError unless the result is a w.ndarray.
        t = [[1.0, 2, 3], [4, 5, 6]]
        ta = func(t, axis=axis)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_array(self, func, axis):
        # [FUNCTION DOC] Calls the function with the axis keyword on a w.ndarray;
        #   raises AssertionError unless the result is a w.ndarray.
        t = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")


@instantiate_parametrized_tests
class TestOneArrAndAxesTuple(TestCase):
    # [CLASS DOC] Parametrized smoke tests for transpose with explicit axes tuples or
    #   None, on a tensor, list and w.ndarray, also checking the resulting shape where
    #   computed.
    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_tensor(self, func, axes):
        # [FUNCTION DOC] Transposes a 1x2x3 tensor; raises AssertionError if the
        #   result is not a w.ndarray or its shape differs from the shape permuted by
        #   axes (reversed shape when axes is None).
        t = torch.ones((1, 2, 3))
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

        # a np.transpose -specific test
        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(w.ndim(t)))
        if ta.shape != newshape:
            raise AssertionError(f"Expected shape {newshape}, got {ta.shape}")

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_list(self, func, axes):
        # [FUNCTION DOC] Transposes a nested list of shape 1x2x3 with the given axes;
        #   raises AssertionError unless the result is a w.ndarray. No shape check is
        #   made.
        t = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]  # shape = (1, 2, 3)
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_array(self, func, axes):
        # [FUNCTION DOC] Transposes a 1x2x3 w.ndarray; raises AssertionError if the
        #   result is not a w.ndarray or its shape differs from the expected
        #   permutation (reversed when axes is None).
        t = w.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(t.ndim))
        if ta.shape != newshape:
            raise AssertionError(f"Expected shape {newshape}, got {ta.shape}")


arr_shape_funcs = [
    w.reshape,
    w.empty_like,
    w.ones_like,
    functools.partial(w.full_like, fill_value=42),
    w.broadcast_to,
]


@instantiate_parametrized_tests
class TestOneArrAndShape(TestCase):
    # [CLASS DOC] Smoke tests for functions taking an array-like and a shape. State: shape
    #   is set to (2, 3) in setUp, and shape_arg_name maps reshape to the keyword name
    #   newshape (others use shape).
    """Smoke test of functions (array_like, shape_like) -> array_like"""

    def setUp(self):
        # [FUNCTION DOC] Test fixture setup: calls the parent setUp, then stores the
        #   target shape (2, 3) and the mapping from function to its shape keyword
        #   name on the instance.
        super().setUp()
        self.shape = (2, 3)
        self.shape_arg_name = {
            w.reshape: "newshape",
        }  # reshape expects `newshape`

    @parametrize("func", arr_shape_funcs)
    def test_andshape_tensor(self, func):
        # [FUNCTION DOC] Calls the function on a 2x3 tensor with the shape keyword;
        #   raises AssertionError unless the result is a w.ndarray of the requested
        #   shape.
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")

    @parametrize("func", arr_shape_funcs)
    def test_andshape_list(self, func):
        # [FUNCTION DOC] Calls the function on a nested list with the shape keyword;
        #   raises AssertionError unless the result is a w.ndarray of the requested
        #   shape.
        t = [[1, 2, 3], [4, 5, 6]]

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")

    @parametrize("func", arr_shape_funcs)
    def test_andshape_array(self, func):
        # [FUNCTION DOC] Calls the function on a w.ndarray with the shape keyword;
        #   raises AssertionError unless the result is a w.ndarray of the requested
        #   shape.
        t = w.asarray([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")


one_arg_scalar_funcs = [(w.size, _np.size), (w.shape, _np.shape), (w.ndim, _np.ndim)]


@instantiate_parametrized_tests
class TestOneArrToScalar(TestCase):
    # [CLASS DOC] Smoke tests that size, shape and ndim return plain python objects (not
    #   w.ndarray) matching the numpy equivalents, for tensor, list and w.ndarray inputs.
    """Smoke test of functions (array_like) -> scalar or python object."""

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_tensor(self, func, np_func):
        # [FUNCTION DOC] Compares the wrapper function with its numpy counterpart on a
        #   tensor (numpy side receives the converted array); raises AssertionError if
        #   the result is a w.ndarray or differs.
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(_np.asarray(t))

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_list(self, func, np_func):
        # [FUNCTION DOC] Compares the wrapper function with its numpy counterpart on a
        #   nested list; raises AssertionError if the result is a w.ndarray or
        #   differs.
        t = [[1, 2, 3], [4, 5, 6]]
        ta = func(t)
        tn = np_func(t)

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_array(self, func, np_func):
        # [FUNCTION DOC] Compares the wrapper function with its numpy counterpart on a
        #   w.ndarray; raises AssertionError if the result is a w.ndarray or differs.
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(t)

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")


shape_funcs = [w.zeros, w.empty, w.ones, functools.partial(w.full, fill_value=42)]


@instantiate_parametrized_tests
class TestShapeLikeToArray(TestCase):
    # [CLASS DOC] Smoke tests of constructors taking only a shape (zeros, empty, ones,
    #   full with fill value 42). Class attribute shape is (3, 4).
    """Smoke test (shape_like) -> array."""

    shape = (3, 4)

    @parametrize("func", shape_funcs)
    def test_shape(self, func):
        # [FUNCTION DOC] Builds an array from the shape; raises AssertionError unless
        #   it is a w.ndarray of that shape.
        a = func(self.shape)

        if not isinstance(a, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(a)}")
        if a.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {a.shape}")


seq_funcs = [w.atleast_1d, w.atleast_2d, w.atleast_3d, w.broadcast_arrays]


@instantiate_parametrized_tests
class TestSequenceOfArrays(TestCase):
    # [CLASS DOC] Smoke tests of atleast_1d, atleast_2d, atleast_3d and broadcast_arrays
    #   with one or several array inputs.
    """Smoke test (sequence of arrays) -> (sequence of arrays)."""

    @parametrize("func", seq_funcs)
    def test_single_tensor(self, func):
        # [FUNCTION DOC] Calls the function on one tensor. broadcast_arrays returns a
        #   tuple so the first element is taken; otherwise the result is used
        #   directly. It must be a w.ndarray.
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        # for a single argument, broadcast_arrays returns a tuple, while
        # atleast_?d return an array
        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = ta[0] if unpack else ta

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_single_list(self, func):
        # [FUNCTION DOC] Same as the single tensor case but with a nested list input.
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_single_array(self, func):
        # [FUNCTION DOC] Same as the single tensor case but with a w.ndarray input.
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_several(self, func):
        # [FUNCTION DOC] Calls the function with three arguments (tensor, w.ndarray,
        #   list); requires a tuple or list result with one w.ndarray per input, else
        #   raises AssertionError.
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        result = func(*arys)
        if not isinstance(result, (tuple, list)):
            raise AssertionError(f"Expected tuple or list, got {type(result)}")
        if len(result) != len(arys):
            raise AssertionError(f"Expected {len(arys)} results, got {len(result)}")
        if not all(isinstance(_, w.ndarray) for _ in result):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in result]}"
            )


seq_to_single_funcs = [
    w.concatenate,
    w.stack,
    w.vstack,
    w.hstack,
    w.dstack,
    w.column_stack,
    w.row_stack,
]


@instantiate_parametrized_tests
class TestSequenceOfArraysToSingle(TestCase):
    # [CLASS DOC] Smoke tests of functions combining a sequence of arrays into one array
    #   (concatenate, stack, vstack, hstack, dstack, column_stack, row_stack).
    """Smoke test (sequence of arrays) -> (array)."""

    @parametrize("func", seq_to_single_funcs)
    def test_several(self, func):
        # [FUNCTION DOC] Passes a tuple of a tensor, a w.ndarray and a list to the
        #   function; raises AssertionError unless the result is a single w.ndarray.
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        result = func(arys)
        if not isinstance(result, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(result)}")


single_to_seq_funcs = (
    w.nonzero,
    # https://github.com/Quansight-Labs/numpy_pytorch_interop/pull/121#discussion_r1172824545
    # w.tril_indices_from,
    # w.triu_indices_from,
    w.where,
)


@instantiate_parametrized_tests
class TestArrayToSequence(TestCase):
    # [CLASS DOC] Smoke tests of functions returning a tuple of arrays from one array
    #   (nonzero and where).
    """Smoke test array -> (tuple of arrays)."""

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_tensor(self, func):
        # [FUNCTION DOC] Calls the function on a tensor; result must be a tuple of
        #   w.ndarray, else AssertionError.
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        if not isinstance(ta, tuple):
            raise AssertionError(f"Expected tuple, got {type(ta)}")
        if not all(isinstance(x, w.ndarray) for x in ta):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in ta]}"
            )

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_list(self, func):
        # [FUNCTION DOC] Calls the function on a nested list; result must be a tuple
        #   of w.ndarray, else AssertionError.
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        if not isinstance(la, tuple):
            raise AssertionError(f"Expected tuple, got {type(la)}")
        if not all(isinstance(x, w.ndarray) for x in la):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in la]}"
            )

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_array(self, func):
        # [FUNCTION DOC] Calls the function on a w.ndarray; result must be a tuple of
        #   w.ndarray, else AssertionError.
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        if not isinstance(la, tuple):
            raise AssertionError(f"Expected tuple, got {type(la)}")
        if not all(isinstance(x, w.ndarray) for x in la):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in la]}"
            )


funcs_and_args = [
    (w.linspace, (0, 10, 11)),
    (w.logspace, (1, 2, 5)),
    (w.logspace, (1, 2, 5, 11)),
    (w.geomspace, (1, 1000, 5, 11)),
    (w.eye, (5, 6)),
    (w.identity, (3,)),
    (w.arange, (5,)),
    (w.arange, (5, 8)),
    (w.arange, (5, 8, 0.5)),
    (w.tri, (3, 3, -1)),
]


@instantiate_parametrized_tests
class TestPythonArgsToArray(TestCase):
    # [CLASS DOC] Smoke test of constructors taking python scalar arguments (linspace,
    #   logspace, geomspace, eye, identity, arange, tri).
    """Smoke_test (sequence of scalars) -> (array)"""

    @parametrize("func, args", funcs_and_args)
    def test_argstoarray_simple(self, func, args):
        # [FUNCTION DOC] Calls the constructor with the given positional args; raises
        #   AssertionError unless the result is a w.ndarray.
        a = func(*args)
        if not isinstance(a, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(a)}")


class TestNormalizations(TestCase):
    # [CLASS DOC] Smoke tests of the argument normalization layer for decorated functions.
    """Smoke test generic problems with normalizations."""

    def test_unknown_args(self):
        # [FUNCTION DOC] Asserts nonzero raises TypeError for an extra positional
        #   argument and for an unknown keyword argument.
        # Check that unknown args to decorated functions fail
        a = w.arange(7) % 2 == 0

        # unknown positional args
        with assert_raises(TypeError):
            w.nonzero(a, "kaboom")

        # unknown kwarg
        with assert_raises(TypeError):
            w.nonzero(a, oops="ouch")

    def test_too_few_args_positional(self):
        # [FUNCTION DOC] Asserts nonzero called with no arguments raises TypeError.
        with assert_raises(TypeError):
            w.nonzero()

    def test_unknown_args_with_defaults(self):
        # [FUNCTION DOC] Asserts eye works with only the required argument (defaults
        #   fill the rest) and raises TypeError when called with no arguments.
        # check a function 5 arguments and 4 defaults: this should work
        w.eye(3)

        # five arguments, four defaults: this should fail
        with assert_raises(TypeError):
            w.eye()


class TestCopyTo(TestCase):
    # [CLASS DOC] Tests for copyto.
    def test_copyto_basic(self):
        # [FUNCTION DOC] Copies an arange into an empty array of equal length and
        #   asserts they match.
        dst = w.empty(4)
        src = w.arange(4)
        w.copyto(dst, src)
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after copyto")

    def test_copytobcast(self):
        # [FUNCTION DOC] Asserts copyto raises RuntimeError when the source cannot
        #   broadcast to the destination, and succeeds with matching result when it
        #   can.
        dst = w.empty((4, 2))
        src = w.arange(4)

        # cannot broadcast => error out
        with assert_raises(RuntimeError):
            w.copyto(dst, src)

        # broadcast src against dst
        dst = w.empty((2, 4))
        w.copyto(dst, src)
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after broadcast copyto")

    def test_copyto_typecast(self):
        # [FUNCTION DOC] Asserts copyto from float to int destination raises TypeError
        #   with casting no, and copies correctly with casting unsafe.
        dst = w.empty(4, dtype=int)
        src = w.arange(4, dtype=float)

        with assert_raises(TypeError):
            w.copyto(dst, src, casting="no")

        # force the type cast
        w.copyto(dst, src, casting="unsafe")
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after type cast copyto")


class TestDivmod(TestCase):
    # [CLASS DOC] Tests for divmod and its out handling.
    def test_divmod_out(self):
        # [FUNCTION DOC] Calls divmod with a tuple of two output arrays; checks
        #   quotient and remainder equal floor division and modulo, and that the
        #   returned objects are the same objects as the supplied outputs.
        x1 = w.arange(8, 15)
        x2 = w.arange(4, 11)

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

        out1, out2 = out
        if quot is not out1:
            raise AssertionError("Expected quot to be out1")
        if rem is not out2:
            raise AssertionError("Expected rem to be out2")

    def test_divmod_out_list(self):
        # [FUNCTION DOC] Same as the out test but with list inputs; checks the
        #   returned quotient and remainder are the supplied output objects.
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        if quot is not out[0]:
            raise AssertionError("Expected quot to be out[0]")
        if rem is not out[1]:
            raise AssertionError("Expected rem to be out[1]")

    @xfail  # ("out1, out2 not implemented")
    def test_divmod_pos_only(self):
        # [FUNCTION DOC] Expected failure (positional out arguments not implemented).
        #   Passes the two outputs positionally and checks they are returned.
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out1, out2 = w.empty_like(x1), w.empty_like(x1)

        quot, rem = w.divmod(x1, x2, out1, out2)

        if quot is not out1:
            raise AssertionError("Expected quot to be out1")
        if rem is not out2:
            raise AssertionError("Expected rem to be out2")

    def test_divmod_no_out(self):
        # [FUNCTION DOC] Calls divmod without out and checks quotient and remainder
        #   equal floor division and modulo.
        # check that the out= machinery handles no out at all
        x1 = w.array([4, 5, 6])
        x2 = w.array([2, 1, 2])
        quot, rem = w.divmod(x1, x2)

        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

    def test_divmod_out_both_pos_and_kw(self):
        # [FUNCTION DOC] Asserts passing outputs both positionally and by keyword
        #   raises TypeError.
        o = w.empty(1)
        with assert_raises(TypeError):
            w.divmod(1, 2, o, o, out=(o, o))


class TestSmokeNotImpl(TestCase):
    # [CLASS DOC] Smoke test for unimplemented-argument annotations.
    def test_nimpl_basic(self):
        # [FUNCTION DOC] Asserts creating an array with a like argument raises
        #   NotImplementedError.
        # smoke test that the "NotImplemented" annotation is picked up
        with assert_raises(NotImplementedError):
            w.empty(3, like="ooops")


@instantiate_parametrized_tests
class TestDefaultDtype(TestCase):
    # [CLASS DOC] Tests for default dtype handling in the numpy wrapper.
    def test_defaultdtype_defaults(self):
        # [FUNCTION DOC] Checks that by default empty arrays are float64, mixing with
        #   an imaginary multiple gives complex128, and arange gives int64; raises
        #   AssertionError otherwise.
        # by default, both floats and ints 64 bit
        x = w.empty(3)
        z = x + 1j * x

        if x.dtype.torch_dtype != torch.float64:
            raise AssertionError(f"Expected torch.float64, got {x.dtype.torch_dtype}")
        if z.dtype.torch_dtype != torch.complex128:
            raise AssertionError(
                f"Expected torch.complex128, got {z.dtype.torch_dtype}"
            )

        if w.arange(3).dtype.torch_dtype != torch.int64:
            raise AssertionError(
                f"Expected torch.int64, got {w.arange(3).dtype.torch_dtype}"
            )

    @parametrize("dt", ["pytorch", "float32", torch.float32])
    def test_set_default_float(self, dt):
        # [FUNCTION DOC] For each way of naming the pytorch default (string, float32
        #   string, torch dtype), sets the default float dtype and checks floats are
        #   float32 and complex is complex64. Always restores the numpy default dtype
        #   afterward, even on failure.
        try:
            w.set_default_dtype(fp_dtype=dt)

            x = w.empty(3)
            z = x + 1j * x

            if x.dtype.torch_dtype != torch.float32:
                raise AssertionError(
                    f"Expected torch.float32, got {x.dtype.torch_dtype}"
                )
            if z.dtype.torch_dtype != torch.complex64:
                raise AssertionError(
                    f"Expected torch.complex64, got {z.dtype.torch_dtype}"
                )

        finally:
            # restore the
            w.set_default_dtype(fp_dtype="numpy")


@skip(_np.__version__ <= "1.23", reason="from_dlpack is new in NumPy 1.23")
class TestExport(TestCase):
    # [CLASS DOC] Export check; skipped when the installed numpy version string is 1.23 or
    #   older.
    def test_exported_objects(self):
        # [FUNCTION DOC] Collects public function names exported by the wrapper
        #   (excluding set_default_dtype), drops names removed in numpy 2 when that
        #   version is installed, and raises AssertionError listing any names missing
        #   from the numpy namespace.
        exported_fns = {
            x
            for x in dir(w)
            if inspect.isfunction(getattr(w, x))
            and not x.startswith("_")
            and x != "set_default_dtype"
        }
        if _np.__version__ > "2":
            # The following methods are removed in NumPy 2.
            # See https://numpy.org/devdocs/numpy_2_0_migration_guide.html#main-namespace
            exported_fns -= {"product", "round_", "sometrue", "cumproduct", "alltrue"}

        diff = exported_fns.difference(set(dir(_np)))
        if len(diff) != 0:
            raise AssertionError(str(diff))


class TestCtorNested(TestCase):
    # [CLASS DOC] Tests for array construction from nested containers.
    def test_arrays_in_lists(self):
        # [FUNCTION DOC] Builds an array from a nested list containing a 0-d array and
        #   asserts it equals the plain nested list.
        lst = [[1, 2], [3, w.array(4)]]
        assert_equal(w.asarray(lst), [[1, 2], [3, 4]])


class TestMisc(TestCase):
    # [CLASS DOC] Miscellaneous tests of the wrapper.
    def test_ndarrays_to_tensors(self):
        # [FUNCTION DOC] Converts a nested tuple containing a w.ndarray to tensors;
        #   raises AssertionError unless the output has two elements, the first a two-
        #   element tuple whose first item is a torch.Tensor.
        out = _util.ndarrays_to_tensors(((w.asarray(42), 7), 3))
        if len(out) != 2:
            raise AssertionError(f"Expected 2 outputs, got {len(out)}")
        if not isinstance(out[0], tuple) or len(out[0]) != 2:
            raise AssertionError(
                f"Expected out[0] to be tuple of len 2, got {type(out[0])} len {len(out[0])}"
            )
        if not isinstance(out[0][0], torch.Tensor):
            raise AssertionError(
                f"Expected out[0][0] to be torch.Tensor, got {type(out[0][0])}"
            )

    @skip(not TEST_CUDA, reason="requires cuda")
    def test_f16_on_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Checks float16 results of vdot, inner,
        #   matmul, einsum, mean, cov and corrcoef on CUDA tensors match those
        #   computed on CPU.
        # make sure operations with float16 tensors give same results on CUDA and on CPU
        t = torch.arange(5, dtype=torch.float16)
        assert_allclose(w.vdot(t.cuda(), t.cuda()), w.vdot(t, t))
        assert_allclose(w.inner(t.cuda(), t.cuda()), w.inner(t, t))
        assert_allclose(w.matmul(t.cuda(), t.cuda()), w.matmul(t, t))
        assert_allclose(w.einsum("i,i", t.cuda(), t.cuda()), w.einsum("i,i", t, t))

        assert_allclose(w.mean(t.cuda()), w.mean(t))

        assert_allclose(w.cov(t.cuda(), t.cuda()), w.cov(t, t).tensor.cuda())
        assert_allclose(w.corrcoef(t.cuda()), w.corrcoef(t).tensor.cuda())


if __name__ == "__main__":
    run_tests()
