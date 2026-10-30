# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests of NEP 50 (weak scalar promotion) behavior in torch._numpy. It first
# defines a table of example expressions with their old and new expected results, then
# compares torch._numpy against real NumPy running in weak promotion mode over combinations of
# Python scalars and typed values. Import side effects: detects whether NumPy is new enough
# (at least 1.24) and sets a flag used to skip all tests, and builds module-level tables of
# examples, scalar kinds, array kinds, dtypes and per-ufunc corner cases.

"""Test examples for NEP 50."""

import itertools
from unittest import skipIf as skipif, SkipTest


try:
    import numpy as _np

    v = _np.__version__.split(".")
    HAVE_NUMPY = int(v[0]) >= 1 and int(v[1]) >= 24
except ImportError:
    HAVE_NUMPY = False

import torch._numpy as tnp
from torch._numpy import (  # noqa: F401
    array,
    bool_,
    complex128,
    complex64,
    float32,
    float64,
    inf,
    int16,
    int32,
    int64,
    uint8,
)
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


uint16 = uint8  # can be anything here, see below


# from numpy import array, uint8, uint16, int64, float32, float64, inf
# from numpy.testing import assert_allclose
# import numpy as np
# np._set_promotion_state('weak')

from pytest import raises as assert_raises


unchanged = None

# expression    old result   new_result
examples = {
    "uint8(1) + 2": (int64(3), uint8(3)),
    "array([1], uint8) + int64(1)": (array([2], uint8), array([2], int64)),
    "array([1], uint8) + array(1, int64)": (array([2], uint8), array([2], int64)),
    "array([1.], float32) + float64(1.)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1.], float32) + array(1., float64)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1], uint8) + 1": (array([2], uint8), unchanged),
    "array([1], uint8) + 200": (array([201], uint8), unchanged),
    "array([100], uint8) + 200": (array([44], uint8), unchanged),
    "array([1], uint8) + 300": (array([301], uint16), Exception),
    "uint8(1) + 300": (int64(301), Exception),
    "uint8(100) + 200": (int64(301), uint8(44)),  # and RuntimeWarning
    "float32(1) + 3e100": (float64(3e100), float32(inf)),  # and RuntimeWarning [T7]
    "array([1.0], float32) + 1e-14 == 1.0": (array([True]), unchanged),
    "array([0.1], float32) == float64(0.1)": (array([True]), array([False])),
    "array(1.0, float32) + 1e-14 == 1.0": (array(False), array(True)),
    "array([1.], float32) + 3": (array([4.0], float32), unchanged),
    "array([1.], float32) + int64(3)": (array([4.0], float32), array([4.0], float64)),
    "3j + array(3, complex64)": (array(3 + 3j, complex128), array(3 + 3j, complex64)),
    "float32(1) + 1j": (array(1 + 1j, complex128), array(1 + 1j, complex64)),
    "int32(1) + 5j": (array(1 + 5j, complex128), unchanged),
    # additional examples from the NEP text
    "int16(2) + 2": (int64(4), int16(4)),
    "int16(4) + 4j": (complex128(4 + 4j), unchanged),
    "float32(5) + 5j": (complex128(5 + 5j), complex64(5 + 5j)),
    "bool_(True) + 1": (int64(2), unchanged),
    "True + uint8(2)": (uint8(3), unchanged),
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestNEP50Table(TestCase):
    # [CLASS DOC] Evaluates each text expression of the NEP 50 examples table with
    # torch._numpy types and compares to the expected new-behavior result. Skipped if NumPy is
    # unavailable or too old.
    @parametrize("example", examples)
    def test_nep50_exceptions(self, example):
        # [FUNCTION DOC] Looks up the old and new results of the example, evaluates the
        # expression text with eval in the module namespace (where array constructors and
        # scalar types are imported), and checks the outcome. If the new result is the
        # exception marker it expects an OverflowError; otherwise a result equal to the old
        # one is used when the new entry is the unchanged marker, the values must be close
        # (tolerance 1e-16) and the dtypes equal.
        old, new = examples[example]

        if new is Exception:
            with assert_raises(OverflowError):
                eval(example)

        else:
            result = eval(example)

            if new is unchanged:
                new = old

            assert_allclose(result, new, atol=1e-16)
            if result.dtype != new.dtype:
                raise AssertionError(
                    f"Expected result.dtype == {new.dtype}, got {result.dtype}"
                )


# ### Directly compare to numpy ###

weaks = (True, 1, 2.0, 3j)
non_weaks = (
    tnp.asarray(True),
    tnp.uint8(1),
    tnp.int8(1),
    tnp.int32(1),
    tnp.int64(1),
    tnp.float32(1),
    tnp.float64(1),
    tnp.complex64(1),
    tnp.complex128(1),
)
if HAVE_NUMPY:
    dtypes = (
        None,
        _np.bool_,
        _np.uint8,
        _np.int8,
        _np.int32,
        _np.int64,
        _np.float32,
        _np.float64,
        _np.complex64,
        _np.complex128,
    )
else:
    dtypes = (None,)


# ufunc name: [array.dtype]
corners = {
    "true_divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "arctan2": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "copysign": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "heaviside": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "ldexp": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "power": ["uint8"],
    "nextafter": ["float32"],
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestCompareToNumpy(TestCase):
    # [CLASS DOC] Compares torch._numpy scalar-with-array promotion against real NumPy under
    # the weak promotion state, across the module-level tables of scalars, arrays and dtypes.
    # Skipped if NumPy is unavailable or too old.
    @parametrize("scalar, array, dtype", itertools.product(weaks, non_weaks, dtypes))
    def test_direct_compare(self, scalar, array, dtype):
        # [FUNCTION DOC] Adds a Python scalar to a torch._numpy value in both libraries,
        # optionally with an explicit output dtype, and asserts that the result dtype and
        # value agree. The NumPy promotion state is switched to weak and restored afterwards.
        # If the NumPy call raises, the case returns without checking.
        # KNOWN ISSUE: when dtype is None the keyword mapping is never assigned before the
        # NumPy call, so a NameError is raised, caught by the broad exception handler, and the
        # test returns early; the no-dtype cases therefore check nothing.
        # compare to NumPy w/ NEP 50.
        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if dtype is not None:
                kwargs = {"dtype": dtype}
            try:
                result_numpy = _np.add(scalar, array.tensor.numpy(), **kwargs)
            except Exception:
                return

            kwargs = {}
            if dtype is not None:
                kwargs = {"dtype": getattr(tnp, dtype.__name__)}
            result = tnp.add(scalar, array, **kwargs).tensor.numpy()
            if result.dtype != result_numpy.dtype:
                raise AssertionError(
                    f"Expected result.dtype == {result_numpy.dtype}, got {result.dtype}"
                )
            if result != result_numpy:
                raise AssertionError(f"Expected result == {result_numpy}, got {result}")

        finally:
            _np._set_promotion_state(state)

    @parametrize("name", tnp._ufuncs._binary)
    @parametrize("scalar, array", itertools.product(weaks, non_weaks))
    def test_compare_ufuncs(self, name, scalar, array):
        # [FUNCTION DOC] For one binary ufunc name from the torch._numpy list, applies it to a
        # scalar and a typed value in both libraries under the weak promotion state (restored
        # afterwards) and asserts the result dtypes match. Raises SkipTest for names and
        # dtypes listed in the corner-case table; silently returns for matmul, modf, divmod
        # and ldexp. A RuntimeError from torch._numpy or a TypeError from NumPy marks that
        # result as missing and skips the comparison. On Windows the dtype comparison is
        # relaxed for boolean arrays; a mismatch raises AssertionError reporting both dtypes.
        if name in corners and (
            array.dtype.name in corners[name]
            or tnp.asarray(scalar).dtype.name in corners[name]
        ):
            raise SkipTest(f"{name}(..., dtype=array.dtype)")

        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if name in ["matmul", "modf", "divmod", "ldexp"]:
                return
            ufunc = getattr(tnp, name)
            ufunc_numpy = getattr(_np, name)

            try:
                result = ufunc(scalar, array)
            except RuntimeError:
                # RuntimeError: "bitwise_xor_cpu" not implemented for 'ComplexDouble' etc
                result = None

            try:
                result_numpy = ufunc_numpy(scalar, array.tensor.numpy())
            except TypeError:
                # TypeError: ufunc 'hypot' not supported for the input types
                result_numpy = None

            type_mismatch = False
            expected_numpy_dtype = None
            expected_torch_dtype = None

            if result is not None and result_numpy is not None:
                expected_numpy_dtype = result_numpy.dtype
                expected_torch_dtype = result.tensor.numpy().dtype
                if IS_WINDOWS:
                    if (
                        array.tensor.numpy().dtype != _np.bool_
                        and result.tensor.numpy().dtype != result_numpy.dtype
                    ):
                        type_mismatch = True

                    if (
                        array.tensor.numpy().dtype == _np.bool_
                        and result_numpy.dtype == _np.int32
                        and result.tensor.numpy().dtype != _np.int64
                    ):
                        expected_numpy_dtype = _np.int32
                        expected_torch_dtype = tnp.int64
                        type_mismatch = True
                else:
                    if result.tensor.numpy().dtype != result_numpy.dtype:
                        type_mismatch = True

            if type_mismatch:
                raise AssertionError(
                    f"Expected result numpy dtype == {expected_numpy_dtype}, torch dtype == {expected_torch_dtype}"
                )

        finally:
            _np._set_promotion_state(state)


if __name__ == "__main__":
    run_tests()
