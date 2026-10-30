# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for scalar type methods (as_integer_ratio, is_integer, class subscripting,
# bit_count) run against torch._numpy, or real numpy when running under dynamo. Every test class
# is unconditionally skipped, so importing only defines the classes and a skip helper.

"""
Test the scalar constructors, which also do type-coercion
"""

import fractions
import functools
import types
from typing import Any
from unittest import skipIf as skipif, SkipTest

import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


skip = functools.partial(skipif, True)


@skip(reason="XXX: scalar.as_integer_ratio not implemented")
@instantiate_parametrized_tests
class TestAsIntegerRatio(TestCase):
    # [CLASS DOC] Skipped tests (scalar as_integer_ratio is not implemented) checking exact
    # fraction results for half, single and double scalars, derived from a CPython float test.
    # derived in part from the cpython test "test_floatasratio"

    @parametrize("ftype", [np.half, np.single, np.double])
    @parametrize(
        "f, ratio",
        [
            (0.875, (7, 8)),
            (-0.875, (-7, 8)),
            (0.0, (0, 1)),
            (11.5, (23, 2)),
        ],
    )
    def test_small(self, ftype, f, ratio):
        # [FUNCTION DOC] Parametrized over float type and (value, expected ratio) pairs: checks
        # that as_integer_ratio of a scalar of that type equals the expected
        # numerator/denominator tuple.
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @parametrize("ftype", [np.half, np.single, np.double])
    def test_simple_fractions(self, ftype):
        # [FUNCTION DOC] For each float type, converts several values through as_integer_ratio
        # and compares the resulting Fraction with the exactly expected Fraction.
        R = fractions.Fraction
        assert_equal(R(0, 1), R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2), R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2), R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1), R(*ftype(-2100.0).as_integer_ratio()))

    @parametrize("ftype", [np.half, np.single, np.double])
    def test_errors(self, ftype):
        # [FUNCTION DOC] Checks that as_integer_ratio raises OverflowError for positive and
        # negative infinity and ValueError for NaN, for each float type.
        assert_raises(OverflowError, ftype("inf").as_integer_ratio)
        assert_raises(OverflowError, ftype("-inf").as_integer_ratio)
        assert_raises(ValueError, ftype("nan").as_integer_ratio)

    def test_against_known_values(self):
        # [FUNCTION DOC] Compares as_integer_ratio of 2.1 (and its negation) in half, single and
        # double precision with hard-coded exact fractions; longdouble is skipped as platform
        # dependent.
        R = fractions.Fraction
        assert_equal(R(1075, 512), R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512), R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152), R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152), R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(
            R(4728779608739021, 2251799813685248), R(*np.double(2.1).as_integer_ratio())
        )
        assert_equal(
            R(-4728779608739021, 2251799813685248),
            R(*np.double(-2.1).as_integer_ratio()),
        )
        # longdouble is platform dependent

    @parametrize(
        "ftype, frac_vals, exp_vals",
        [
            # dtype test cases generated using hypothesis
            # first five generated cases per dtype
            (
                np.half,
                [
                    0.0,
                    0.01154830649280303,
                    0.31082276347447274,
                    0.527350517124794,
                    0.8308562335072596,
                ],
                [0, 1, 0, -8, 12],
            ),
            (
                np.single,
                [
                    0.0,
                    0.09248576989263226,
                    0.8160498218131407,
                    0.17389442853722373,
                    0.7956044195067877,
                ],
                [0, 12, 10, 17, -26],
            ),
            (
                np.double,
                [
                    0.0,
                    0.031066908499895136,
                    0.5214135908877832,
                    0.45780736035689296,
                    0.5906586745934036,
                ],
                [0, -801, 51, 194, -653],
            ),
        ],
    )
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        # [FUNCTION DOC] Round trip check: builds scalars by scaling given fractions with ldexp
        # and the given exponents, asserts the dtype is preserved, then divides numerator by
        # denominator in longdouble and expects the original value. Raises SkipTest if
        # longdouble cannot hold them.
        for frac, exp in zip(frac_vals, exp_vals):
            f = np.ldexp(ftype(frac), exp)
            if f.dtype != ftype:
                raise AssertionError(f"Expected f.dtype == {ftype}, got {f.dtype}")
            n, d = f.as_integer_ratio()

            try:
                nf = np.longdouble(n)
                df = np.longdouble(d)
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                raise SkipTest("longdouble too small on this platform")  # noqa: B904

            assert_equal(nf / df, f, f"{n}/{d}")


@skip(reason="NP_VER: older numpies has problems with .is_integer")
@instantiate_parametrized_tests
class TestIsInteger(TestCase):
    # [CLASS DOC] Skipped tests for the scalar is_integer method (old numpy versions are buggy).
    @parametrize("str_value", ["inf", "nan"])
    @parametrize("code", np.typecodes["Float"])
    def test_special(self, code, str_value):
        # [FUNCTION DOC] For each float type code, infinity and NaN scalars must report
        # is_integer as false; raises AssertionError otherwise.
        cls = np.dtype(code).type
        value = cls(str_value)
        if value.is_integer():
            raise AssertionError(
                f"Expected value.is_integer() to be False for {str_value}"
            )

    @parametrize(
        "code", "efd" + "Bbhil"
    )  # np.typecodes["Float"] + np.typecodes["AllInteger"])
    def test_true(self, code: str) -> None:
        # [FUNCTION DOC] Builds an array of the integers from -5 to 4 cast to the given float or
        # integer type code and asserts every element reports is_integer as true.
        float_array = np.arange(-5, 5).astype(code)
        for value in float_array:
            if not value.is_integer():
                raise AssertionError(
                    f"Expected value.is_integer() to be True for {value}"
                )

    @parametrize("code", "bhil")  # np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        # [FUNCTION DOC] Takes the integers from -5 to 4 cast to an integer type code, scales
        # the array in place by 1.1 and asserts each nonzero element is not an integer.
        # KNOWN ISSUE: the type codes are all integer types, so in-place scaling by a float
        # either # fails to cast or truncates, and the not-integer expectation looks
        # unsatisfiable (skipped).
        float_array = np.arange(-5, 5).astype(code)
        float_array *= 1.1
        for value in float_array:
            if value == 0:
                continue
            if value.is_integer():
                raise AssertionError(
                    f"Expected value.is_integer() to be False for {value}"
                )


@skip(reason="XXX: implementation details of the type system differ")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    # [CLASS DOC] Skipped tests (type system internals differ) for subscripting scalar classes,
    # which should produce a generic alias for abstract numeric classes and raise TypeError
    # otherwise.
    @parametrize(
        "cls",
        [
            np.number,
            np.integer,
            np.inexact,
            np.unsignedinteger,
            np.signedinteger,
            np.floating,
        ],
    )
    def test_abc(self, cls: type[np.number]) -> None:
        # [FUNCTION DOC] Subscripting an abstract numeric class with a type must give a
        # GenericAlias whose origin is that class; raises AssertionError otherwise.
        alias = cls[Any]
        if not isinstance(alias, types.GenericAlias):
            raise AssertionError(
                f"Expected alias to be GenericAlias, got {type(alias)}"
            )
        if alias.__origin__ is not cls:
            raise AssertionError(
                f"Expected alias.__origin__ is {cls}, got {alias.__origin__}"
            )

    def test_abc_complexfloating(self) -> None:
        # [FUNCTION DOC] Same alias check for the complex floating class subscripted with two
        # type arguments: result must be a GenericAlias with that class as origin.
        alias = np.complexfloating[Any, Any]
        if not isinstance(alias, types.GenericAlias):
            raise AssertionError(
                f"Expected alias to be GenericAlias, got {type(alias)}"
            )
        if alias.__origin__ is not np.complexfloating:
            raise AssertionError(
                f"Expected alias.__origin__ is np.complexfloating, got {alias.__origin__}"
            )

    @parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        # [FUNCTION DOC] Subscripts the complex floating class with a tuple of the given length:
        # one or two entries must work, zero or more than two must raise TypeError with a too
        # few or too many arguments message.
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            if not np.complexfloating[arg_tup]:
                raise AssertionError(
                    "Expected np.complexfloating[arg_tup] to be truthy"
                )
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    @parametrize("cls", [np.generic])
    def test_abc_non_numeric(self, cls: type[np.generic]) -> None:
        # [FUNCTION DOC] Subscripting the non-numeric generic scalar base class must raise
        # TypeError.
        with pytest.raises(TypeError):
            cls[Any]

    @parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        # [FUNCTION DOC] Subscripting the concrete scalar class of any type code must raise
        # TypeError.
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    @parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        # [FUNCTION DOC] Subscripts the number class with a tuple of the given length: length
        # one must yield a truthy result, any other length must raise TypeError.
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            if not np.number[arg_tup]:
                raise AssertionError("Expected np.number[arg_tup] to be truthy")
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    def test_subscript_scalar(self) -> None:
        # [FUNCTION DOC] Subscripting the number class with a single type must give a truthy
        # result.
        if not np.number[Any]:
            raise AssertionError("Expected np.number[Any] to be truthy")


@skip(reason="scalartype(...).bit_count() not implemented")
@instantiate_parametrized_tests
class TestBitCount(TestCase):
    # [CLASS DOC] Skipped tests for the scalar bit_count method, derived from a CPython test.
    # derived in part from the cpython test "test_bit_count"

    @parametrize(
        "itype",
        [np.int8, np.int16, np.int32, np.int64]
        + [np.uint8, np.uint16, np.uint32, np.uint64],
    )
    def test_small(self, itype):
        # [FUNCTION DOC] For each integer type, loops over non-negative values below 128 and
        # compares the scalar bit_count with the Python int bit_count; raises AssertionError on
        # a mismatch.
        for a in range(max(np.iinfo(itype).min, 0), 128):
            msg = f"Smoke test for {itype}({a}).bit_count()"
            if itype(a).bit_count() != a.bit_count():
                raise AssertionError(msg)

    def test_bit_count(self):
        # [FUNCTION DOC] Checks bit_count of unsigned 64-bit scalars built from powers of two
        # (exponents 10, 17, 63), one less, and two bit-flipped variants against known counts.
        for exp in [10, 17, 63]:
            a = 2**exp
            if np.uint64(a).bit_count() != 1:
                raise AssertionError(
                    f"Expected bit_count == 1, got {np.uint64(a).bit_count()}"
                )
            if np.uint64(a - 1).bit_count() != exp:
                raise AssertionError(
                    f"Expected bit_count == {exp}, got {np.uint64(a - 1).bit_count()}"
                )
            if np.uint64(a ^ 63).bit_count() != 7:
                raise AssertionError(
                    f"Expected bit_count == 7, got {np.uint64(a ^ 63).bit_count()}"
                )
            if np.uint64((a - 1) ^ 510).bit_count() != exp - 8:
                raise AssertionError(
                    f"Expected bit_count == {exp - 8}, got {np.uint64((a - 1) ^ 510).bit_count()}"
                )


if __name__ == "__main__":
    run_tests()
