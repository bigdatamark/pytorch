# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests documenting why zero-dimensional arrays are used in place of NumPy array
# scalars: shape, size and dtype of such values, conversion to Python int, multiplication by a
# list (which follows 0D-array semantics and yields an ndarray), comparisons, and the lax
# isscalar function. Uses real NumPy under TorchDynamo and torch._numpy otherwise. Import
# builds a module-level parameter decorator over four ways of creating the value 42.

"""
Basic tests to assert and illustrate the  behavior around the decision to use 0D
arrays in place of array scalars.

Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*

Also test the isscalar function (which is deliberately a bit more lax).
"""

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


parametrize_value = parametrize(
    "value",
    [
        subtest(np.int64(42), name="int64"),
        subtest(np.array(42), name="array"),
        subtest(np.asarray(42), name="asarray"),
        subtest(np.asarray(np.int64(42)), name="asarray_int"),
    ],
)


@instantiate_parametrized_tests
class TestArrayScalars(TestCase):
    # [CLASS DOC] Behavior tests for scalar-like values (int64 scalar, 0D arrays),
    # parametrized over the module-level value decorator where noted.
    @parametrize_value
    def test_array_scalar_basic(self, value):
        # [FUNCTION DOC] Asserts the value has zero dimensions, an empty shape, size one and
        # dtype int64; raises AssertionError describing the first mismatch.
        if value.ndim != 0:
            raise AssertionError(f"Expected ndim 0, got {value.ndim}")
        if value.shape != ():
            raise AssertionError(f"Expected shape (), got {value.shape}")
        if value.size != 1:
            raise AssertionError(f"Expected size 1, got {value.size}")
        if value.dtype != np.dtype("int64"):
            raise AssertionError(f"Expected dtype int64, got {value.dtype}")

    @parametrize_value
    def test_conversion_to_int(self, value):
        # [FUNCTION DOC] Converts the value to a Python int and asserts it equals 42, that the
        # result is an int, and that the original value is not itself an int instance.
        py_scalar = int(value)
        if py_scalar != 42:
            raise AssertionError(f"Expected 42, got {py_scalar}")
        if not isinstance(py_scalar, int):
            raise AssertionError(f"Expected int, got {type(py_scalar)}")
        if isinstance(value, int):
            raise AssertionError("Expected value to not be an int")

    @parametrize_value
    def test_decay_to_py_scalar(self, value):
        # [FUNCTION DOC] Multiplies the value by a three-element list, on the left and on the
        # right, and asserts each result is an ndarray of shape three equal to 42 times one,
        # two, three. This shows scalars behave like 0D arrays rather than decaying to Python
        # ints.
        # NumPy distinguishes array scalars and 0D arrays. For instance
        # `scalar * list` is equivalent to `int(scalar) * list`, but
        # `0D array * list` is equivalent to `0D array * np.asarray(list)`.
        # Our scalars follow 0D array behavior (because they are 0D arrays)
        lst = [1, 2, 3]

        product = value * lst
        if not isinstance(product, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(product)}")
        if product.shape != (3,):
            raise AssertionError(f"Expected shape (3,), got {product.shape}")
        assert_equal(product, [42, 42 * 2, 42 * 3])

        # repeat with right-multiply
        product = lst * value
        if not isinstance(product, np.ndarray):
            raise AssertionError(f"Expected np.ndarray, got {type(product)}")
        if product.shape != (3,):
            raise AssertionError(f"Expected shape (3,), got {product.shape}")
        assert_equal(product, [42, 42 * 2, 42 * 3])

    def test_scalar_comparisons(self):
        # [FUNCTION DOC] Asserts that an int64 scalar and a 0D array of 42 compare equal and
        # satisfy greater-or-equal and less-or-equal, and that both compare equal to the
        # Python int 42.
        scalar = np.int64(42)
        arr = np.array(42)

        if not (arr == scalar):
            raise AssertionError("Expected arr == scalar")
        if not (arr >= scalar):
            raise AssertionError("Expected arr >= scalar")
        if not (arr <= scalar):
            raise AssertionError("Expected arr <= scalar")

        if not (scalar == 42):
            raise AssertionError("Expected scalar == 42")
        if not (arr == 42):
            raise AssertionError("Expected arr == 42")


# @xfailIfTorchDynamo
@instantiate_parametrized_tests
class TestIsScalar(TestCase):
    # [CLASS DOC] Tests for isscalar, which is deliberately lax: it accepts any numeric object
    # with exactly one element (including lists and arrays of one element), unlike NumPy.
    # Class attributes hold the parameter lists of scalars and non-scalars; some cases are
    # marked expected failures under TorchDynamo.
    #
    # np.isscalar(...) checks that its argument is a numeric object with exactly one element.
    #
    # This differs from NumPy which also requires that shape == ().
    #
    scalars = [
        subtest(42, "literal"),
        subtest(int(42.0), "int"),
        subtest(np.float32(42), "float32"),
        subtest(np.array(42), "array_0D", decorators=[xfailIfTorchDynamo]),
        subtest([42], "list", decorators=[xfailIfTorchDynamo]),
        subtest([[42]], "list-list", decorators=[xfailIfTorchDynamo]),
        subtest(np.array([42]), "array_1D", decorators=[xfailIfTorchDynamo]),
        subtest(np.array([[42]]), "array_2D", decorators=[xfailIfTorchDynamo]),
    ]

    import math

    not_scalars = [
        int,
        np.float32,
        subtest("s", decorators=[xfailIfTorchDynamo]),
        subtest("string", decorators=[xfailIfTorchDynamo]),
        (),
        [],
        math.sin,
        np,
        np.transpose,
        [1, 2],
        np.asarray([1, 2]),
        np.float32([1, 2]),
    ]

    @parametrize("value", scalars)
    def test_is_scalar(self, value):
        # [FUNCTION DOC] Asserts isscalar returns true for each of the listed scalar-like
        # values.
        if not np.isscalar(value):
            raise AssertionError(f"Expected {value} to be scalar")

    @parametrize("value", not_scalars)
    def test_is_not_scalar(self, value):
        # [FUNCTION DOC] Asserts isscalar returns false for each of the listed non-scalar
        # values (types, strings, empty containers, functions, modules, multi-element arrays).
        if np.isscalar(value):
            raise AssertionError(f"Expected {value} to not be scalar")


if __name__ == "__main__":
    run_tests()
