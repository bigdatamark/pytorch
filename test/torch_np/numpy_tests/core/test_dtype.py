# Owner(s): ["module: dynamo"]
# [MODULE DOC] NumPy dtype tests ported to torch._numpy (real NumPy under dynamo). Covers dtype
# hashing and equality, invalid type strings, rich comparison with non-dtypes, pickling of dtypes
# and dtype classes, dtype promotion cases (skipped), the dtype attribute protocol and class
# subscripting (skipped). Many classes are skipped because torch._numpy lacks the features.
# Defines two small assertion helpers for dtype equality and inequality.

import functools
import operator
import pickle
import types
from itertools import permutations
from typing import Any
from unittest import skipIf as skipif

import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
    xpassIfTorchDynamo_np,
)


skip = functools.partial(skipif, True)

if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_, assert_equal

import numpy


def assert_dtype_equal(a, b):
    # [FUNCTION DOC] HELPER: Assert two dtypes are equal and that their hashes are equal too.
    assert_equal(a, b)
    assert_equal(
        hash(a), hash(b), "two equivalent types do not hash to the same value !"
    )


def assert_dtype_not_equal(a, b):
    # [FUNCTION DOC] HELPER: Assert two dtypes differ and that their hashes also differ.
    assert_(a != b)
    assert_(hash(a) != hash(b), "two different types hash to the same value !")


@instantiate_parametrized_tests
class TestBuiltin(TestCase):
    # [CLASS DOC] Tests of basic dtype construction, hashing and comparison for builtin
    # types.
    @parametrize("t", [int, float, complex, np.int32])
    def test_run(self, t):
        # [FUNCTION DOC] For each parametrized builtin type, build a dtype and hash it;
        # only checks that these calls do not fail.
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    def test_equivalent_dtype_hashing(self):
        # [FUNCTION DOC] Check the intp dtype equals the integer dtype of the same width
        # (32 or 64 bits by its item size) and both have the same hash.
        # Make sure equivalent dtypes with different type num hash equal
        intp = np.dtype(np.intp)
        if intp.itemsize == 4:
            left = intp
            right = np.dtype(np.int32)
        else:
            left = intp
            right = np.dtype(np.int64)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    @xfailIfTorchDynamo  # TypeError -> InternalTorchDynamoError
    def test_invalid_types(self):
        # [FUNCTION DOC] Check that malformed type strings (object, bool, short,
        # unsigned, half and float with bad sizes, and long codes with the wrong size for
        # the platform) raise TypeError. Expected to fail under dynamo, where TypeError
        # becomes an internal dynamo error.
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, "O3")
        assert_raises(TypeError, np.dtype, "O5")
        assert_raises(TypeError, np.dtype, "O7")
        assert_raises(TypeError, np.dtype, "b3")
        assert_raises(TypeError, np.dtype, "h4")
        assert_raises(TypeError, np.dtype, "I5")
        assert_raises(TypeError, np.dtype, "e3")
        assert_raises(TypeError, np.dtype, "f5")

        if np.dtype("l").itemsize == 8:
            assert_raises(TypeError, np.dtype, "l4")
            assert_raises(TypeError, np.dtype, "L4")
        else:
            assert_raises(TypeError, np.dtype, "l8")
            assert_raises(TypeError, np.dtype, "L8")

    # XXX: what is 'q'? on my 64-bit ubuntu machine it's int64, same as 'l'
    #       if np.dtype('q').itemsize == 8:
    #           assert_raises(TypeError, np.dtype, 'q4')
    #           assert_raises(TypeError, np.dtype, 'Q4')
    #       else:
    #           assert_raises(TypeError, np.dtype, 'q8')
    #           assert_raises(TypeError, np.dtype, 'Q8')

    def test_richcompare_invalid_dtype_equality(self):
        # [FUNCTION DOC] Compare a dtype with the number 7, which is not a valid dtype,
        # and check no exception is raised and the comparison is not equal.
        # KNOWN ISSUE: both checks use the not-equal operator although the first error
        # message says equality, so equality with an invalid object is never actually
        # tested.
        # Make sure objects that cannot be converted to valid
        # dtypes results in False/True when compared to valid dtypes.
        # Here 7 cannot be converted to dtype. No exceptions should be raised

        if not (np.dtype(np.int32) != 7):
            raise AssertionError("dtype richcompare failed for ==")
        if not (np.dtype(np.int32) != 7):
            raise AssertionError("dtype richcompare failed for !=")

    @parametrize("operation", [operator.le, operator.lt, operator.ge, operator.gt])
    def test_richcompare_invalid_dtype_comparison(self, operation):
        # [FUNCTION DOC] For each ordering operator (le, lt, ge, gt), comparing a dtype
        # against the invalid dtype 7 must raise TypeError.
        # Make sure TypeError is raised for comparison operators
        # for invalid dtypes. Here 7 is an invalid dtype.

        with pytest.raises(TypeError):
            operation(np.dtype(np.int32), 7)

    @skipif(
        numpy.__version__ < "1.24",
        reason="older numpies emit DeprecatioWarnings instead",
    )
    @parametrize(
        "dtype",
        [
            "Bool",
            "Bytes0",
            "Complex32",
            "Complex64",
            "Datetime64",
            "Float16",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Object0",
            "Str0",
            "Timedelta64",
            "UInt8",
            "UInt16",
            "Uint32",
            "UInt32",
            "Uint64",
            "UInt64",
            "Void0",
            "Float128",
            "Complex128",
        ],
    )
    def test_numeric_style_types_are_invalid(self, dtype):
        # [FUNCTION DOC] Old style numeric names (such as Int32 or Float64) must raise
        # TypeError when passed to the dtype constructor. Skipped for NumPy older than
        # 1.24.
        with assert_raises(TypeError):
            np.dtype(dtype)


@skip(reason="dtype attributes not yet implemented")
class TestDtypeAttributeDeletion(TestCase):
    # [CLASS DOC] Tests that dtype attributes cannot be deleted; the whole class is skipped
    # because those attributes are not implemented.
    def test_dtype_non_writable_attributes_deletion(self):
        # [FUNCTION DOC] Deleting each read-only dtype attribute in a list must raise
        # AttributeError.
        dt = np.dtype(np.double)
        attr = [
            "subdtype",
            "descr",
            "str",
            "name",
            "base",
            "shape",
            "isbuiltin",
            "isnative",
            "isalignedstruct",
            "fields",
            "metadata",
            "hasobject",
        ]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        # [FUNCTION DOC] Deleting the names attribute must raise AttributeError.
        # KNOWN ISSUE: the name says writable attributes, yet the test expects deletion
        # to fail.
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


@instantiate_parametrized_tests
class TestPickling(TestCase):
    # [CLASS DOC] Tests that dtypes and dtype classes survive pickling round trips.
    def check_pickling(self, dtype):
        # [FUNCTION DOC] HELPER: For every pickle protocol up to the highest, pickle the
        # dtype, check the bytes mention dtype but not the private reconstruct function,
        # load it back, check it is equal, and check arrays of zeros built from the
        # original and the loaded dtype are equal.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            buf = pickle.dumps(dtype, proto)
            # The dtype pickling itself pickles `np.dtype` if it is pickled
            # as a singleton `dtype` should be stored in the buffer:
            if b"_DType_reconstruct" in buf:
                raise AssertionError("_DType_reconstruct should not be in buf")
            if b"dtype" not in buf:
                raise AssertionError("dtype should be in buf")
            pickled = pickle.loads(buf)
            assert_equal(pickled, dtype)

            # XXX: out dtypes do not have .descr
            #         assert_equal(pickled.descr, dtype.descr)
            #         if dtype.metadata is not None:
            #             assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional

            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @parametrize("t", [int, float, complex, np.int32, bool])
    def test_builtin(self, t):
        # [FUNCTION DOC] Run the pickling helper on dtypes of int, float, complex, int32
        # and bool.
        self.check_pickling(np.dtype(t))

    @parametrize(
        "DType",
        [
            subtest(type(np.dtype(t)), name=f"{np.dtype(t).name}_{i}")
            for i, t in enumerate(np.typecodes["All"])
        ]
        + [np.dtype],
    )
    def test_pickle_types(self, DType):
        # [FUNCTION DOC] For each parametrized dtype class (one per type code, plus the
        # base dtype class), pickle round trip under every protocol must return the
        # identical class object.
        # Check that DTypes (the classes/types) roundtrip when pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            roundtrip_DType = pickle.loads(pickle.dumps(DType, proto))
            if roundtrip_DType is not DType:
                raise AssertionError(
                    f"roundtrip_DType is not DType: {roundtrip_DType} vs {DType}"
                )


@skip(reason="XXX: value-based promotions, we don't have.")
@instantiate_parametrized_tests
class TestPromotion(TestCase):
    # [CLASS DOC] Tests of complex dtype promotion rules; skipped because value-based
    # promotion is not implemented in torch._numpy.
    """Test cases related to more complex DType promotions.  Further promotion
    tests are defined in `test_numeric.py`
    """

    @parametrize(
        "other, expected, expected_weak",
        [
            (2**16 - 1, np.complex64, None),
            (2**32 - 1, np.complex128, np.complex64),
            subtest((np.float16(2), np.complex64, None), name="float16_complex64_None"),
            subtest((np.float32(2), np.complex64, None), name="float32_complex64_None"),
            # repeat for complex scalars:
            subtest(
                (np.complex64(2), np.complex64, None), name="complex64_complex64_None"
            ),
        ],
    )
    def test_complex_other_value_based(
        self, weak_promotion, other, expected, expected_weak
    ):
        # [FUNCTION DOC] Check promotion of a complex64 dtype with various scalar values
        # and numpy scalars, giving the expected complex type (or the weak-promotion one
        # when selected), for both the result_type function and the dtype of a minimum
        # ufunc result.
        # KNOWN ISSUE: the weak_promotion argument is not provided by any fixture in this
        # file, so the test could not run if the class were not skipped.
        if weak_promotion and expected_weak is not None:
            expected = expected_weak

        # This would change if we modify the value based promotion
        min_complex = np.dtype(np.complex64)

        res = np.result_type(other, min_complex)
        if res != expected:
            raise AssertionError(f"result_type mismatch: {res} != {expected}")
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(other, np.ones(3, dtype=min_complex)).dtype
        if res != expected:
            raise AssertionError(f"dtype mismatch: {res} != {expected}")

    @parametrize(
        "other, expected",
        [
            (np.bool_, np.complex128),
            (np.int64, np.complex128),
            (np.float16, np.complex64),
            (np.float32, np.complex64),
            (np.float64, np.complex128),
            (np.complex64, np.complex64),
            (np.complex128, np.complex128),
        ],
    )
    def test_complex_scalar_value_based(self, other, expected):
        # [FUNCTION DOC] Check that promoting a given dtype with the Python complex
        # scalar gives the expected complex dtype, using result_type and the dtype of a
        # minimum ufunc result.
        # This would change if we modify the value based promotion
        complex_scalar = 1j

        res = np.result_type(other, complex_scalar)
        if res != expected:
            raise AssertionError(f"result_type mismatch: {res} != {expected}")
        # Check the same for a simple ufunc call that uses the same logic:
        res = np.minimum(np.ones(3, dtype=other), complex_scalar).dtype
        if res != expected:
            raise AssertionError(f"dtype mismatch: {res} != {expected}")

    @parametrize("val", [2, 2**32, 2**63, 2**64, 2 * 100])
    def test_python_integer_promotion(self, val):
        # [FUNCTION DOC] For several Python integer values, result_type of the value with
        # zero (and with a numpy int8 zero) must equal the promotion of the dtypes of the
        # matching arrays.
        # If we only path scalars (mainly python ones!), the result must take
        # into account that the integer may be considered int32, int64, uint64,
        # or object depending on the input value.  So test those paths!
        expected_dtype = np.result_type(np.array(val).dtype, np.array(0).dtype)
        if np.result_type(val, 0) != expected_dtype:
            raise AssertionError(
                f"result_type mismatch: {np.result_type(val, 0)} != {expected_dtype}"
            )
        # For completeness sake, also check with a NumPy scalar as second arg:
        if np.result_type(val, np.int8(0)) != expected_dtype:
            raise AssertionError(
                f"result_type mismatch: {np.result_type(val, np.int8(0))} != {expected_dtype}"
            )

    @parametrize(
        "dtypes, expected",
        [
            # These promotions are not associative/commutative:
            ([np.int16, np.float16], np.float32),
            ([np.int8, np.float16], np.float32),
            ([np.uint8, np.int16, np.float16], np.float32),
            # The following promotions are not ambiguous, but cover code
            # paths of abstract promotion (no particular logic being tested)
            ([1, 1, np.float64], np.float64),
            ([1, 1.0, np.complex128], np.complex128),
            ([1, 1j, np.float64], np.complex128),
            ([1.0, 1.0, np.int64], np.float64),
            ([1.0, 1j, np.float64], np.complex128),
            ([1j, 1j, np.float64], np.complex128),
            ([1, True, np.bool_], np.int_),
        ],
    )
    def test_permutations_do_not_influence_result(self, dtypes, expected):
        # [FUNCTION DOC] For each list of dtypes or scalars, every ordering must give the
        # same expected result type from result_type.
        # Tests that most permutations do not influence the result.  In the
        # above some uint and int combinations promote to a larger integer
        # type, which would then promote to a larger than necessary float.
        for perm in permutations(dtypes):
            if np.result_type(*perm) != expected:
                raise AssertionError(
                    f"result_type mismatch: {np.result_type(*perm)} != {expected}"
                )


class TestMisc(TestCase):
    # [CLASS DOC] Miscellaneous dtype tests.
    def test_dtypes_are_true(self):
        # [FUNCTION DOC] Check float64 and int64 dtype objects are truthy.
        # test for gh-6294
        if not bool(np.dtype("f8")):
            raise AssertionError("np.dtype('f8') should be truthy")
        if not bool(np.dtype("i8")):
            raise AssertionError("np.dtype('i8') should be truthy")

    @xpassIfTorchDynamo_np  # (reason="No keyword arg for dtype ctor.")
    def test_keyword_argument(self):
        # [FUNCTION DOC] Check the dtype constructor accepts a keyword argument and gives
        # a dtype equal to the positional form. Expected to pass only under dynamo.
        # test for https://github.com/numpy/numpy/pull/16574#issuecomment-642660971
        if np.dtype(dtype=np.float64) != np.dtype(np.float64):
            raise AssertionError("dtype keyword argument should work")


class TestFromDTypeAttribute(TestCase):
    # [CLASS DOC] Tests of building a dtype from an object that has a dtype attribute.
    def test_simple(self):
        # [FUNCTION DOC] Define a local class with a float64 dtype class attribute and
        # check that both the class and an instance of it convert to the float64 dtype.
        class dt:
            # [CLASS DOC] HELPER: Local class whose only member is a class attribute
            # holding a float64 dtype.
            dtype = np.dtype("f8")

        if np.dtype(dt) != np.float64:
            raise AssertionError(f"np.dtype(dt) should be float64, got {np.dtype(dt)}")
        if np.dtype(dt()) != np.float64:
            raise AssertionError(
                f"np.dtype(dt()) should be float64, got {np.dtype(dt())}"
            )

    @skip(
        reason="We simply require the .name attribute, so this "
        "fails with an AttributeError."
    )
    def test_recursion(self):
        # [FUNCTION DOC] Make a class whose dtype attribute refers to itself, and check
        # that building a dtype from the class or an instance raises RecursionError.
        # Skipped because torch._numpy fails with AttributeError instead.
        class dt:
            # [CLASS DOC] HELPER: Empty local class; the test sets its dtype
            # attribute to itself.
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)


@skip(reason="Parametric dtypes, our stuff is simpler.")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    # [CLASS DOC] Tests subscripting dtype classes as generic aliases; skipped as
    # torch._numpy has no parametric dtypes.
    def test_dtype(self) -> None:
        # [FUNCTION DOC] Subscripting the dtype class must give a generic alias whose
        # origin is the dtype class.
        alias = np.dtype[Any]
        if not isinstance(alias, types.GenericAlias):
            raise AssertionError(f"alias should be GenericAlias, got {type(alias)}")
        if alias.__origin__ is not np.dtype:
            raise AssertionError(
                f"alias.__origin__ should be np.dtype, got {alias.__origin__}"
            )

    @parametrize("code", np.typecodes["All"])
    def test_dtype_subclass(self, code: str) -> None:
        # [FUNCTION DOC] For each type code, subscripting the dtype subclass must give a
        # generic alias whose origin is that subclass.
        cls = type(np.dtype(code))
        alias = cls[Any]
        if not isinstance(alias, types.GenericAlias):
            raise AssertionError(f"alias should be GenericAlias, got {type(alias)}")
        if alias.__origin__ is not cls:
            raise AssertionError(
                f"alias.__origin__ should be {cls}, got {alias.__origin__}"
            )

    @parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        # [FUNCTION DOC] Subscripting with a tuple of length one must succeed; other
        # lengths from zero to three must raise TypeError.
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            if not np.dtype[arg_tup]:
                raise AssertionError("np.dtype[arg_tup] should be truthy")
        else:
            with pytest.raises(TypeError):
                np.dtype[arg_tup]

    def test_subscript_scalar(self) -> None:
        # [FUNCTION DOC] Subscripting the dtype class with the Any type must give a
        # truthy result.
        if not np.dtype[Any]:
            raise AssertionError("np.dtype[Any] should be truthy")


if __name__ == "__main__":
    run_tests()
