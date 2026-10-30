# Owner(s): ["module: dynamo"]
# [MODULE DOC] Port of numpy scalar math tests to torch._numpy (or real numpy under dynamo):
# type construction and promotion, power, modulus and divmod semantics, complex division,
# conversions, comparison operators, repr, negation, abs, bit shifts, hashing, overflow warnings
# and subclass deferral. Import time defines a skip helper, lists of scalar types, an operator
# table and two small helper functions; several test classes are skipped or expected to fail
# with torch.

import contextlib
import functools
import itertools
import operator
import sys
import warnings
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

import numpy
import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    slowTest as slow,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import (
        _gen_alignment_data,
        assert_,
        assert_almost_equal,
        assert_equal,
    )
else:
    import torch._numpy as np
    from torch._numpy.testing import (
        _gen_alignment_data,
        assert_,
        assert_almost_equal,
        assert_equal,
    )


skip = functools.partial(skipif, True)

IS_PYPY = False

types = [
    np.bool_,
    np.byte,
    np.ubyte,
    np.short,
    np.intc,
    np.int_,
    np.longlong,
    np.single,
    np.double,
    np.csingle,
    np.cdouble,
]

floating_types = np.floating.__subclasses__()
complex_floating_types = np.complexfloating.__subclasses__()

objecty_things = [object(), None]

reasonable_operators_for_scalars = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
    operator.add,
    operator.floordiv,
    operator.mod,
    operator.mul,
    operator.pow,
    operator.sub,
    operator.truediv,
]


# This compares scalarmath against ufuncs.


class TestTypes(TestCase):
    # [CLASS DOC] Tests that scalar construction and scalar arithmetic agree with array
    # behavior.
    def test_types(self):
        # [FUNCTION DOC] For every scalar type, builds a scalar from one and asserts it equals
        # one.
        for atype in types:
            a = atype(1)
            assert_(a == 1, f"error with {atype!r}: got {a!r}")

    def test_type_add(self):
        # [FUNCTION DOC] For every pair of scalar types adds a scalar of each and adds one
        # element arrays of each, then asserts the result dtypes agree (scalar promotion matches
        # ufunc promotion).
        # list of types
        for k, atype in enumerate(types):
            a_scalar = atype(3)
            a_array = np.array([3], dtype=atype)
            for l, btype in enumerate(types):
                b_scalar = btype(1)
                b_array = np.array([1], dtype=btype)
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                # It was comparing the type numbers, but the new ufunc
                # function-finding mechanism finds the lowest function
                # to which both inputs can be cast - which produces 'l'
                # when you do 'q' + 'b'.  The old function finding mechanism
                # skipped ahead based on the first argument, but that
                # does not produce properly symmetric results...
                assert_equal(
                    c_scalar.dtype,
                    c_array.dtype,
                    "error with types "
                    f"({k:d}/'{np.dtype(atype).name}' + {l:d}/'{np.dtype(btype).name}')",
                )

    def test_type_create(self):
        # [FUNCTION DOC] For every scalar type, creates an array from a list with that dtype and
        # calls the scalar type on the same list, then asserts the two results are equal.
        for atype in types:
            a = np.array([1, 2, 3], atype)
            b = atype([1, 2, 3])
            assert_equal(a, b)

    @skipIfTorchDynamo()  # freezes under torch.Dynamo (loop unrolling, huh)
    def test_leak(self):
        # [FUNCTION DOC] Leak smoke test: performs a scalar add two hundred thousand times; a
        # leak would be seen externally (valgrind). Skipped under dynamo because it freezes
        # there.
        # test leak of scalar objects
        # a leak would show up in valgrind as still-reachable of ~2.6MB
        for _ in range(200000):
            np.add(1, 1)


class TestBaseMath(TestCase):
    # [CLASS DOC] Basic elementwise add tests over generated memory alignments.
    def test_blocked(self):
        # [FUNCTION DOC] For float32, float64 and int32, iterates over generated aligned and
        # offset binary operand and output buffers and checks add, square, reciprocal (skipped
        # for int32), scalar operands and the out argument against expected values.
        # test alignments offsets for simd instructions
        # alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 7), (np.int32, 11)]:
            for out, inp1, inp2, msg in _gen_alignment_data(
                dtype=dt, type="binary", max_size=sz
            ):
                exp1 = np.ones_like(inp1)
                inp1[...] = np.ones_like(inp1)
                inp2[...] = np.zeros_like(inp2)
                assert_almost_equal(np.add(inp1, inp2), exp1, err_msg=msg)
                assert_almost_equal(np.add(inp1, 2), exp1 + 2, err_msg=msg)
                assert_almost_equal(np.add(1, inp2), exp1, err_msg=msg)

                np.add(inp1, inp2, out=out)
                assert_almost_equal(out, exp1, err_msg=msg)

                inp2[...] += np.arange(inp2.size, dtype=dt) + 1
                assert_almost_equal(
                    np.square(inp2), np.multiply(inp2, inp2), err_msg=msg
                )
                # skip true divide for ints
                if dt != np.int32:
                    assert_almost_equal(
                        np.reciprocal(inp2), np.divide(1, inp2), err_msg=msg
                    )

                inp1[...] = np.ones_like(inp1)
                np.add(inp1, 2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)
                inp2[...] = np.ones_like(inp2)
                np.add(2, inp2, out=out)
                assert_almost_equal(out, exp1 + 2, err_msg=msg)

    @xpassIfTorchDynamo_np  # (reason="pytorch does not have .view")
    def test_lower_align(self):
        # [FUNCTION DOC] Builds float64 arrays from a byte buffer sliced at an offset of four
        # bytes and viewed as double, then runs add variants including the out argument. Marked
        # as expected to pass only under dynamo numpy because torch has no view with a dtype
        # change.
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        o = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_almost_equal(d + d, d * 2)
        np.add(d, d, out=o)
        np.add(np.ones_like(d), d, out=o)
        np.add(d, np.ones_like(d), out=o)
        np.add(np.ones_like(d), d)
        np.add(d, np.ones_like(d))


class TestPower(TestCase):
    # [CLASS DOC] Tests for the power operator on scalars of several types.
    def test_small_types(self):
        # [FUNCTION DOC] Raises small integer and float16 scalars of value three to the fourth
        # power and asserts the result is 81.
        for t in [np.int8, np.int16, np.float16]:
            a = t(3)
            b = a**4
            assert_(b == 81, f"error with {t!r}: got {b!r}")

    def test_large_types(self):
        # [FUNCTION DOC] Raises fifty-one to the fourth power in wider integer and float types;
        # integers must match exactly 6765201 and floats must match approximately.
        for t in [np.int32, np.int64, np.float32, np.float64]:
            a = t(51)
            b = a**4
            msg = f"error with {t!r}: got {b!r}"
            if np.issubdtype(t, np.integer):
                assert_(b == 6765201, msg)
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

    @skip(reason="NP_VER: fails on CI on older NumPy")
    @xpassIfTorchDynamo_np  # (reason="Value-based casting: (2)**(-2) -> 0 in pytorch.")
    def test_integers_to_negative_integer_power(self):
        # [FUNCTION DOC] Checks integer scalar power with a negative integer exponent:
        # non-uint64 bases must raise ValueError, uint64 bases must give a float64 result.
        # Skipped, and torch value based casting would give a different answer anyway.
        # KNOWN ISSUE: the uint64 branch is dead for the negative one base list (no uint64
        # member there).
        # Note that the combination of uint64 with a signed integer
        # has common type np.float64. The other combinations should all
        # raise a ValueError for integer ** negative integer.
        exp = [np.array(-1, dt)[()] for dt in "bhil"]

        # 1 ** -1 possible special case
        base = [np.array(1, dt)[()] for dt in "bhilB"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 1.0)

        # -1 ** -1 possible special case
        base = [np.array(-1, dt)[()] for dt in "bhil"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, -1.0)

        # 2 ** -1 perhaps generic
        base = [np.array(2, dt)[()] for dt in "bhilB"]
        for i1, i2 in itertools.product(base, exp):
            if i1.dtype != np.uint64:
                assert_raises(ValueError, operator.pow, i1, i2)
            else:
                res = operator.pow(i1, i2)
                assert_(res.dtype.type is np.float64)
                assert_almost_equal(res, 0.5)

    def test_mixed_types(self):
        # [FUNCTION DOC] Raises a scalar of value three to the power of a scalar of value two
        # for all pairs of mixed integer and float types; integer results must equal nine,
        # others approximately.
        typelist = [
            np.int8,
            np.int16,
            np.float16,
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ]
        for t1 in typelist:
            for t2 in typelist:
                a = t1(3)
                b = t2(2)
                result = a**b
                msg = f"error with {t1!r} and {t2!r}:got {result!r}, expected {9!r}"
                if np.issubdtype(np.dtype(result), np.integer):
                    assert_(result == 9, msg)
                else:
                    assert_almost_equal(result, 9, err_msg=msg)

    def test_modular_power(self):
        # [FUNCTION DOC] Three argument (modular) power is not implemented: asserts a TypeError
        # for scalar and zero dimensional array inputs of int32, float32 and complex64.
        # modular power is not implemented, so ensure it errors
        a = 5
        b = 4
        c = 10
        expected = pow(a, b, c)  # noqa: F841
        for t in (np.int32, np.float32, np.complex64):
            # note that 3-operand power only dispatches on the first argument
            assert_raises(TypeError, operator.pow, t(a), b, c)
            assert_raises(TypeError, operator.pow, np.array(t(a)), b, c)


def floordiv_and_mod(x, y):
    # [FUNCTION DOC] HELPER: returns both the floor division and the remainder of two operands
    # as a tuple, so it can be used interchangeably with the divmod builtin.
    return (x // y, x % y)


def _signs(dt):
    # [FUNCTION DOC] HELPER: returns the sign multipliers to test for a dtype code: only
    # positive for unsigned integer codes, both positive and negative otherwise.
    if dt in np.typecodes["UnsignedInteger"]:
        return (+1,)
    else:
        return (+1, -1)


@instantiate_parametrized_tests
class TestModulus(TestCase):
    # [CLASS DOC] Tests for floor division and remainder (modulus) on scalars, checking sign
    # rules.
    def test_modulus_basic(self):
        # [FUNCTION DOC] For both division helpers and all pairs of integer and float codes with
        # all sign combinations, divides seventy-one by nineteen scalars and checks div times b
        # plus rem equals a and that the remainder has the sign of the divisor and magnitude
        # below it.
        # dt = np.typecodes["AllInteger"] + np.typecodes["Float"]
        dt = "Bbhil" + "efd"
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)[()]
                    b = np.array(sg2 * 19, dtype=dt2)[()]
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @slow
    def test_float_modulus_exact(self):
        # [FUNCTION DOC] Slow test: for all dividend and divisor pairs of small integers, checks
        # float floor division and remainder results exactly match Python divmod, including
        # signed zeros.
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = [divmod(*t) for t in arg]

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floordiv_and_mod, divmod]:
            for dt in np.typecodes["Float"]:
                msg = f"op: {op.__name__}, dtype: {dt}"
                fa = a.astype(dt)
                fb = b.astype(dt)
                # use list comprehension so a_ and b_ are scalars
                div, rem = zip(*[op(a_, b_) for a_, b_ in zip(fa, fb)])
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_modulus_roundoff(self):
        # [FUNCTION DOC] Regression test for gh-6127: float scalars with tiny magnitudes in all
        # sign combinations satisfy the division identity and the remainder sign and bound
        # rules.
        # gh-6127
        # dt = np.typecodes["Float"]
        dt = "efd"
        for op in [floordiv_and_mod, divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = "op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s"
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-8, dtype=dt1)[()]
                    b = np.array(sg2 * 6e-8, dtype=dt2)[()]
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @parametrize("dt", "efd")
    def test_float_modulus_corner_cases(self, dt):
        # [FUNCTION DOC] Parametrized over float codes; half precision is skipped by raising
        # SkipTest since nextafter is missing. Checks the remainder of the value just below zero
        # by one stays within bounds, then for each float code checks remainders with zero or
        # NaN or infinity give NaN and that divmod by zero yields infinity and NaN.
        if dt == "e":
            # FIXME: make xfail
            raise SkipTest("RuntimeError: 'nextafter_cpu' not implemented for 'Half'")

        b = np.array(1.0, dtype=dt)
        a = np.nextafter(np.array(0.0, dtype=dt), -b)
        rem = operator.mod(a, b)
        assert_(rem <= b, f"dt: {dt}")
        rem = operator.mod(-a, -b)
        assert_(rem >= -b, f"dt: {dt}")

        # Check nans, inf
        #     with suppress_warnings() as sup:
        #         sup.filter(RuntimeWarning, "invalid value encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in remainder")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in floor_divide")
        #         sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
        #         sup.filter(RuntimeWarning, "invalid value encountered in divmod")
        for dt in "efd":
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            rem = operator.mod(fone, fzer)
            assert_(np.isnan(rem), f"dt: {dt}")
            # MSVC 2008 returns NaN here, so disable the check.
            # rem = operator.mod(fone, finf)
            # assert_(rem == fone, 'dt: %s' % dt)
            rem = operator.mod(fone, fnan)
            assert_(np.isnan(rem), f"dt: {dt}")
            rem = operator.mod(finf, fone)
            assert_(np.isnan(rem), f"dt: {dt}")
            for op in [floordiv_and_mod, divmod]:
                div, mod = op(fone, fzer)
                assert_(np.isinf(div)) and assert_(np.isnan(mod))


class TestComplexDivision(TestCase):
    # [CLASS DOC] Tests for complex scalar division.
    @skip(reason="With pytorch, 1/(0+0j) is nan + nan*j, not inf + nan*j")
    def test_zero_division(self):
        # [FUNCTION DOC] Skipped: checks division by complex zero gives infinity or NaN for
        # various infinite and NaN numerators; torch gives NaN where numpy gives infinity.
        for t in [np.complex64, np.complex128]:
            a = t(0.0)
            b = t(1.0)
            assert_(np.isinf(b / a))
            b = t(complex(np.inf, np.inf))
            assert_(np.isinf(b / a))
            b = t(complex(np.inf, np.nan))
            assert_(np.isinf(b / a))
            b = t(complex(np.nan, np.inf))
            assert_(np.isinf(b / a))
            b = t(complex(np.nan, np.nan))
            assert_(np.isnan(b / a))
            b = t(0.0)
            assert_(np.isnan(b / a))

    def test_signed_zeros(self):
        # [FUNCTION DOC] Divides complex scalars in a table of numerator, denominator and
        # expected tuples and compares real and imaginary parts separately so that signed zeros
        # are checked.
        for t in [np.complex64, np.complex128]:
            # tupled (numerator, denominator, expected)
            # for testing as expected == numerator/denominator
            data = (
                ((0.0, -1.0), (0.0, 1.0), (-1.0, -0.0)),
                ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((0.0, -1.0), (-0.0, -1.0), (1.0, 0.0)),
                ((0.0, -1.0), (-0.0, 1.0), (-1.0, 0.0)),
                ((0.0, 1.0), (0.0, -1.0), (-1.0, 0.0)),
                ((0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((-0.0, -1.0), (0.0, -1.0), (1.0, -0.0)),
                ((-0.0, 1.0), (0.0, -1.0), (-1.0, -0.0)),
            )
            for cases in data:
                n = cases[0]
                d = cases[1]
                ex = cases[2]
                result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                # check real and imag parts separately to avoid comparison
                # in array context, which does not account for signed zeros
                assert_equal(result.real, ex[0])
                assert_equal(result.imag, ex[1])

    def test_branches(self):
        # [FUNCTION DOC] Exercises the two magnitude comparison branches of complex division
        # with cases whose denominators have larger real or larger imaginary parts, expecting
        # one.
        for t in [np.complex64, np.complex128]:
            # tupled (numerator, denominator, expected)
            # for testing as expected == numerator/denominator
            data = []

            # trigger branch: real(fabs(denom)) > imag(fabs(denom))
            # followed by else condition as neither are == 0
            data.append(((2.0, 1.0), (2.0, 1.0), (1.0, 0.0)))

            # trigger branch: real(fabs(denom)) > imag(fabs(denom))
            # followed by if condition as both are == 0
            # is performed in test_zero_division(), so this is skipped

            # trigger else if branch: real(fabs(denom)) < imag(fabs(denom))
            data.append(((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)))

            for cases in data:
                n = cases[0]
                d = cases[1]
                ex = cases[2]
                result = t(complex(n[0], n[1])) / t(complex(d[0], d[1]))
                # check real and imag parts separately to avoid comparison
                # in array context, which does not account for signed zeros
                assert_equal(result.real, ex[0])
                assert_equal(result.imag, ex[1])


class TestConversion(TestCase):
    # [CLASS DOC] Tests for conversions between scalars, arrays, Python numbers and integer
    # limits.
    def test_int_from_long(self):
        # [FUNCTION DOC] Converts large float and integer arrays elementwise to Python int and
        # compares with the exact integers; assumes the default float type is float64.
        # NB: this test assumes that the default fp type is float64
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -(10**6), -(10**12), -(10**18)]
        for T in [None, np.float64, np.int64]:
            a = np.array(l, dtype=T)
            assert_equal([int(_m) for _m in a], li)

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @xpassIfTorchDynamo_np  # (reason="pytorch does not emit this warning.")
    def test_iinfo_long_values_1(self):
        # [FUNCTION DOC] Expects a DeprecationWarning when creating a small integer array from a
        # value one beyond the type maximum, and the result to wrap to the minimum. Skipped on
        # old numpy and expected to fail with torch, which emits no warning.
        for code in "bBh":
            with pytest.warns(DeprecationWarning):
                res = np.array(np.iinfo(code).max + 1, dtype=code)
            tgt = np.iinfo(code).min
            assert_(res == tgt)

    def test_iinfo_long_values_2(self):
        # [FUNCTION DOC] For each integer code, the maximum value round trips through array
        # creation and through the scalar type unchanged.
        for code in np.typecodes["AllInteger"]:
            res = np.array(np.iinfo(code).max, dtype=code)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

        for code in np.typecodes["AllInteger"]:
            res = np.dtype(code).type(np.iinfo(code).max)
            tgt = np.iinfo(code).max
            assert_(res == tgt)

    def test_int_raise_behaviour(self):
        # [FUNCTION DOC] Checks that constructing 64-bit scalar types from a value above their
        # maximum raises OverflowError or RuntimeError.
        def overflow_error_func(dtype):
            # [FUNCTION DOC] HELPER: builds a scalar of the given dtype from its maximum plus
            # one to trigger an overflow error; returns nothing.
            dtype(np.iinfo(dtype).max + 1)

        for code in [np.int_, np.longlong]:
            assert_raises((OverflowError, RuntimeError), overflow_error_func, code)

    def test_numpy_scalar_relational_operators(self):
        # [FUNCTION DOC] Checks greater-than, less-than and equality between Python ints and
        # scalars and between scalars of all integer pairs and all signed integer and float
        # pairs.
        # All integer
        for dt1 in np.typecodes["AllInteger"]:
            assert_(1 > np.array(0, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(0, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in np.typecodes["AllInteger"]:
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(0, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

        # Signed integers and floats
        for dt1 in "bhl" + np.typecodes["Float"]:
            assert_(1 > np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not 1 < np.array(-1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 == np.array(-1, dtype=dt1)[()], f"type {dt1} failed")

            for dt2 in "bhl" + np.typecodes["Float"]:
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    np.array(-1, dtype=dt1)[()] == np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    def test_numpy_scalar_relational_operators_2(self):
        # [FUNCTION DOC] Checks comparisons between unsigned byte scalars and Python ints or
        # signed integer scalars with a negative value.
        # Unsigned integers
        for dt1 in "B":
            assert_(-1 < np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(not -1 > np.array(1, dtype=dt1)[()], f"type {dt1} failed")
            assert_(-1 != np.array(1, dtype=dt1)[()], f"type {dt1} failed")

            # unsigned vs signed
            for dt2 in "bhil":
                assert_(
                    np.array(1, dtype=dt1)[()] > np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    not np.array(1, dtype=dt1)[()] < np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )
                assert_(
                    np.array(1, dtype=dt1)[()] != np.array(-1, dtype=dt2)[()],
                    f"type {dt1} and {dt2} failed",
                )

    def test_scalar_comparison_to_none(self):
        # [FUNCTION DOC] Records warnings while checking a float scalar is not none and asserts
        # none were emitted.
        # Scalars should just return False and not give a warnings.
        # The comparisons are flagged by pep8, ignore that.
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", "", FutureWarning)
            assert_(np.float32(1) is not None)
            assert_(np.float32(1) is not None)
        assert_(len(w) == 0)


# class TestRepr:
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


@xpassIfTorchDynamo_np  # (reason="can delegate repr to pytorch")
class TestRepr(TestCase):
    # [CLASS DOC] Round trip repr tests for float scalars; expected to pass only under dynamo
    # numpy since torch could delegate repr.
    def _test_type_repr(self, t):
        # [FUNCTION DOC] HELPER: for float type t, builds the smallest denormal and smallest
        # normal bit patterns in a byte buffer, views them as t, evaluates the repr text back to
        # a value and asserts equality unless the round trip underflowed to zero.
        finfo = np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize * 8
        # could add some more types to the list below
        for which in ["small denorm", "small norm"]:
            # Values from https://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00] * storage_bytes, dtype=np.uint8)
            if which == "small denorm":
                byte = last_fraction_bit_idx // 8
                bytebit = 7 - (last_fraction_bit_idx % 8)
                constr[byte] = 1 << bytebit
            elif which == "small norm":
                byte = last_exponent_bit_idx // 8
                bytebit = 7 - (last_exponent_bit_idx % 8)
                constr[byte] = 1 << bytebit
            else:
                raise ValueError("hmm")
            val = constr.view(t)[0]
            val_repr = repr(val)
            val2 = t(eval(val_repr))
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # [FUNCTION DOC] Runs the repr round trip for float32 and float64 (long double excluded
        # because the repr text is parsed through a Python float).
        # long double test cannot work, because eval goes through a python
        # float
        for t in [np.float32, np.float64]:
            self._test_type_repr(t)


@skip(reason="Array scalars do not decay to python scalars.")
class TestMultiply(TestCase):
    # [CLASS DOC] Skipped tests (array scalars do not decay to Python scalars) for multiplying
    # numpy scalars with sequences.
    def test_seq_repeat(self):
        # [FUNCTION DOC] Checks list and tuple repeat via integer scalars, a deprecation warning
        # for the bool code, and TypeError for all other scalar codes.
        # KNOWN ISSUE: assert_warns is not imported in this module, so the deprecated branch
        # would fail.
        # Test that basic sequences get repeated when multiplied with
        # numpy integers. And errors are raised when multiplied with others.
        # Some of this behaviour may be controversial and could be open for
        # change.
        accepted_types = set(np.typecodes["AllInteger"])
        deprecated_types = {"?"}
        forbidden_types = set(np.typecodes["All"]) - accepted_types - deprecated_types
        forbidden_types -= {"V"}  # can't default-construct void scalars

        for seq_type in (list, tuple):
            seq = seq_type([1, 2, 3])
            for numpy_type in accepted_types:
                i = np.dtype(numpy_type).type(2)
                assert_equal(seq * i, seq * int(i))
                assert_equal(i * seq, int(i) * seq)

            for numpy_type in deprecated_types:
                i = np.dtype(numpy_type).type()
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, seq, i), seq * int(i)
                )
                assert_equal(
                    assert_warns(DeprecationWarning, operator.mul, i, seq), int(i) * seq
                )

            for numpy_type in forbidden_types:
                i = np.dtype(numpy_type).type()
                assert_raises(TypeError, operator.mul, seq, i)
                assert_raises(TypeError, operator.mul, i, seq)

    def test_no_seq_repeat_basic_array_like(self):
        # [FUNCTION DOC] Checks an array-like object and a memoryview multiplied by float32 or
        # integer scalars multiply elementwise instead of repeating a sequence.
        # KNOWN ISSUE: assert_array_equal is not imported in this module (test is skipped).
        # Test that an array-like which does not know how to be multiplied
        # does not attempt sequence repeat (raise TypeError).
        # See also gh-7428.
        class ArrayLike:
            # [CLASS DOC] HELPER: minimal object wrapping an array and exposing it via the array
            # protocol.
            def __init__(self, arr):
                # [FUNCTION DOC] HELPER: stores the wrapped array in an attribute.
                self.arr = arr

            def __array__(self):
                # [FUNCTION DOC] HELPER: returns the wrapped array.
                return self.arr

        # Test for simple ArrayLike above and memoryviews (original report)
        for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
            assert_array_equal(arr_like * np.float32(3.0), np.full(3, 3.0))
            assert_array_equal(np.float32(3.0) * arr_like, np.full(3, 3.0))
            assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
            assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))


class TestNegative(TestCase):
    # [CLASS DOC] Tests for unary negation of scalars.
    def test_exceptions(self):
        # [FUNCTION DOC] Negating a boolean scalar must raise TypeError (numpy) or RuntimeError
        # (torch).
        a = np.ones((), dtype=np.bool_)[()]
        # XXX: TypeError from numpy, RuntimeError from torch
        assert_raises((TypeError, RuntimeError), operator.neg, a)

    def test_result(self):
        # [FUNCTION DOC] For integer and float codes, negating one gives the type maximum for
        # unsigned codes (wraparound) and otherwise a value that sums with the original to zero.
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        # with suppress_warnings() as sup:
        #     sup.filter(RuntimeWarning)
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            if dt in np.typecodes["UnsignedInteger"]:
                st = np.dtype(dt).type
                max = st(np.iinfo(dt).max)
                assert_equal(operator.neg(a), max)
            else:
                assert_equal(operator.neg(a) + a, 0)


class TestSubtract(TestCase):
    # [CLASS DOC] Tests for scalar subtraction.
    def test_exceptions(self):
        # [FUNCTION DOC] Subtracting boolean scalars must raise TypeError (numpy) or
        # RuntimeError (torch).
        a = np.ones((), dtype=np.bool_)[()]
        with assert_raises((TypeError, RuntimeError)):  # XXX: TypeError from numpy
            operator.sub(a, a)  # RuntimeError from torch

    def test_result(self):
        # [FUNCTION DOC] For integer and float codes, subtracting a scalar one from itself gives
        # zero.
        types = np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
        #        with suppress_warnings() as sup:
        #            sup.filter(RuntimeWarning)
        for dt in types:
            a = np.ones((), dtype=dt)[()]
            assert_equal(operator.sub(a, a), 0)


@instantiate_parametrized_tests
class TestAbs(TestCase):
    # [CLASS DOC] Tests for absolute value on float and complex scalars via two different
    # functions.
    def _test_abs_func(self, absfunc, test_dtype):
        # [FUNCTION DOC] HELPER: applies the given abs function to a scalar of the given type
        # for negative, positive zero, negative zero, maximum, tiniest normal and minimum values
        # and checks the expected magnitudes (zero must come back as positive zero).
        x = test_dtype(-1.5)
        assert_equal(absfunc(x), 1.5)
        x = test_dtype(0.0)
        res = absfunc(x)
        # assert_equal() checks zero signedness
        assert_equal(res, 0.0)
        x = test_dtype(-0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)

        x = test_dtype(np.finfo(test_dtype).max)
        assert_equal(absfunc(x), x.real)

        #      with suppress_warnings() as sup:
        #          sup.filter(UserWarning)
        x = test_dtype(np.finfo(test_dtype).tiny)
        assert_equal(absfunc(x), x.real)

        x = test_dtype(np.finfo(test_dtype).min)
        assert_equal(absfunc(x), -x.real)

    @parametrize("dtype", floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        # [FUNCTION DOC] Runs the shared abs checks using the Python builtin abs for one float
        # or complex type.
        self._test_abs_func(abs, dtype)

    @parametrize("dtype", floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        # [FUNCTION DOC] Runs the shared abs checks using the library abs function for one float
        # or complex type.
        self._test_abs_func(np.abs, dtype)


@instantiate_parametrized_tests
class TestBitShifts(TestCase):
    # [CLASS DOC] Tests for bit shift operators on integer scalars.
    @parametrize("type_code", np.typecodes["AllInteger"])
    @parametrize("op", [operator.rshift, operator.lshift])
    def test_shift_all_bits(self, type_code, op):
        # [FUNCTION DOC] Parametrized over integer code and shift operator. Shifts by the full
        # bit width and by four more: negative values shifted right give minus one, everything
        # else gives zero, and array results must match scalar results. Raises SkipTest for
        # unsigned 16, 32 and 64 bit types and for a known torch issue with full-width shifts of
        # int and long types.
        """Shifts where the shift amount is the width of the type or wider"""
        # gh-2449
        dt = np.dtype(type_code)
        nbits = dt.itemsize * 8
        if dt in (np.dtype(np.uint64), np.dtype(np.uint32), np.dtype(np.uint16)):
            raise SkipTest("NYI: bitshift uint64")

        for val in [5, -5]:
            for shift in [nbits, nbits + 4]:
                val_scl = np.array(val).astype(dt)[()]
                shift_scl = dt.type(shift)

                res_scl = op(val_scl, shift_scl)
                if val_scl < 0 and op is operator.rshift:
                    # sign bit is preserved
                    assert_equal(res_scl, -1)
                else:
                    if type_code in ("i", "l") and shift == np.iinfo(type_code).bits:
                        # FIXME: make xfail
                        raise SkipTest(
                            "https://github.com/pytorch/pytorch/issues/70904"
                        )
                    assert_equal(res_scl, 0)

                # Result on scalars should be the same as on arrays
                val_arr = np.array([val_scl] * 32, dtype=dt)
                shift_arr = np.array([shift] * 32, dtype=dt)
                res_arr = op(val_arr, shift_arr)
                assert_equal(res_arr, res_scl)


@skip(reason="Will rely on pytest for hashing")
@instantiate_parametrized_tests
class TestHash(TestCase):
    # [CLASS DOC] Skipped tests that numpy scalar hashes match Python number hashes.
    @parametrize("type_code", np.typecodes["AllInteger"])
    def test_integer_hashes(self, type_code):
        # [FUNCTION DOC] For each integer code, the hash of scalars built from 0 to 127 must
        # equal the hash of the Python integer; raises AssertionError otherwise.
        scalar = np.dtype(type_code).type
        for i in range(128):
            if hash(i) != hash(scalar(i)):
                raise AssertionError(f"Expected hash({i}) == hash(scalar({i}))")

    @parametrize("type_code", np.typecodes["AllFloat"])
    def test_float_and_complex_hashes(self, type_code):
        # [FUNCTION DOC] For each float code, hashes of scalars built from several values
        # (including pi and infinity) must equal the hash of the same Python value, and distinct
        # NaN hashes are expected if Python distinguishes them.
        scalar = np.dtype(type_code).type
        for val in [np.pi, np.inf, 3, 6.0]:
            numpy_val = scalar(val)
            # Cast back to Python, in case the NumPy scalar has less precision
            if numpy_val.dtype.kind == "c":
                val = complex(numpy_val)
            else:
                val = float(numpy_val)
            if val != numpy_val:
                raise AssertionError(
                    f"Expected val == numpy_val, got {val} vs {numpy_val}"
                )
            if hash(val) != hash(numpy_val):
                raise AssertionError(
                    f"Expected hash(val) == hash(numpy_val), got {hash(val)} vs {hash(numpy_val)}"
                )

        if hash(float(np.nan)) != hash(float(np.nan)):
            # If Python distinguishes different NaNs we do so too (gh-18833)
            if hash(scalar(np.nan)) == hash(scalar(np.nan)):
                raise AssertionError(
                    "Expected hash(scalar(np.nan)) != hash(scalar(np.nan))"
                )

    @parametrize("type_code", np.typecodes["Complex"])
    def test_complex_hashes(self, type_code):
        # [FUNCTION DOC] For each complex code, hashes of scalars built from complex values must
        # equal the hash of the corresponding Python complex.
        # Test some complex valued hashes specifically:
        scalar = np.dtype(type_code).type
        for val in [np.pi + 1j, np.inf - 3j, 3j, 6.0 + 1j]:
            numpy_val = scalar(val)
            if hash(complex(numpy_val)) != hash(numpy_val):
                raise AssertionError(
                    f"Expected hash(complex(numpy_val)) == hash(numpy_val), got {hash(complex(numpy_val))} vs {hash(numpy_val)}"
                )


@contextlib.contextmanager
def recursionlimit(n):
    # [FUNCTION DOC] HELPER: context manager that sets the interpreter recursion limit to the
    # given value for the body and always restores the previous limit afterwards. Not used in
    # this module.
    o = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(n)
        yield
    finally:
        sys.setrecursionlimit(o)


@instantiate_parametrized_tests
class TestScalarOpsMisc(TestCase):
    # [CLASS DOC] Tests for warnings on scalar overflow and division by zero.
    @xfail  # (reason="pytorch does not warn on overflow")
    @parametrize("dtype", "Bbhil")
    @parametrize(
        "operation",
        [
            lambda min, max: max + max,
            lambda min, max: min - max,
            lambda min, max: max * max,
        ],
    )
    def test_scalar_integer_operation_overflow(self, dtype, operation):
        # [FUNCTION DOC] For each integer code and each of three operations (max plus max, min
        # minus max, max times max), expects a RuntimeWarning about overflow. Expected to fail
        # with torch, which does not warn.
        st = np.dtype(dtype).type
        min = st(np.iinfo(dtype).min)
        max = st(np.iinfo(dtype).max)

        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            operation(min, max)

    @skip(reason="integer overflow UB: crashes pytorch under ASAN")
    @parametrize("dtype", "bhil")
    @parametrize(
        "operation",
        [
            lambda min, neg_1: -min,
            lambda min, neg_1: abs(min),
            lambda min, neg_1: min * neg_1,
            subtest(
                lambda min, neg_1: min // neg_1,
                decorators=[skip(reason="broken on some platforms")],
            ),
        ],
    )
    def test_scalar_signed_integer_overflow(self, dtype, operation):
        # [FUNCTION DOC] Signed integer minimum cases (negate, abs, times minus one, floor
        # divide by minus one) expect an overflow RuntimeWarning. Skipped because overflow
        # crashes torch under sanitizers.
        # The minimum signed integer can "overflow" for some additional operations
        st = np.dtype(dtype).type
        min = st(np.iinfo(dtype).min)
        neg_1 = st(-1)

        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            operation(min, neg_1)

    @skipif(numpy.__version__ < "1.24", reason="NP_VER: fails on NumPy 1.23.x")
    @xpassIfTorchDynamo_np  # (reason="pytorch does not warn on overflow")
    @parametrize("dtype", "B")
    def test_scalar_unsigned_integer_overflow(self, dtype):
        # [FUNCTION DOC] Negating an unsigned byte scalar of value eight must warn about
        # overflow, while negating zero must not. Skipped on old numpy; torch emits no warning.
        val = np.dtype(dtype).type(8)
        with pytest.warns(RuntimeWarning, match="overflow encountered"):
            -val

        zero = np.dtype(dtype).type(0)
        -zero  # does not warn

    @xfail  # (reason="pytorch raises RuntimeError on division by zero")
    @parametrize("dtype", np.typecodes["AllInteger"])
    @parametrize(
        "operation",
        [
            lambda val, zero: val // zero,
            lambda val, zero: val % zero,
        ],
    )
    def test_scalar_integer_operation_divbyzero(self, dtype, operation):
        # [FUNCTION DOC] Floor division and modulus of a hundred by zero for each integer code
        # should warn about divide by zero. Expected to fail because torch raises RuntimeError
        # instead.
        st = np.dtype(dtype).type
        val = st(100)
        zero = st(0)

        with pytest.warns(RuntimeWarning, match="divide by zero"):
            operation(val, zero)


ops_with_names = [
    ("__lt__", "__gt__", operator.lt, True),
    ("__le__", "__ge__", operator.le, True),
    ("__eq__", "__eq__", operator.eq, True),
    # Note __op__ and __rop__ may be identical here:
    ("__ne__", "__ne__", operator.ne, True),
    ("__gt__", "__lt__", operator.gt, True),
    ("__ge__", "__le__", operator.ge, True),
    ("__floordiv__", "__rfloordiv__", operator.floordiv, False),
    ("__truediv__", "__rtruediv__", operator.truediv, False),
    ("__add__", "__radd__", operator.add, False),
    ("__mod__", "__rmod__", operator.mod, False),
    ("__mul__", "__rmul__", operator.mul, False),
    ("__pow__", "__rpow__", operator.pow, False),
    ("__sub__", "__rsub__", operator.sub, False),
]


@instantiate_parametrized_tests
class TestScalarSubclassingMisc(TestCase):
    # [CLASS DOC] Skipped tests (scalar subclassing is unsupported) for operator deferral.
    @skip(reason="We do not support subclassing scalars.")
    @parametrize("__op__, __rop__, op, cmp", ops_with_names)
    @parametrize("sctype", [np.float32, np.float64])
    def test_subclass_deferral(self, sctype, __op__, __rop__, op, cmp):
        # [FUNCTION DOC] Skipped. For each operator and float scalar type, checks that two
        # unrelated subclasses fall back to the base operator while a subclass overriding the
        # operator and its reflected form wins.
        """
        This test covers scalar subclass deferral.  Note that this is exceedingly
        complicated, especially since it tends to fall back to the array paths and
        these additionally add the "array priority" mechanism.

        The behaviour was modified subtly in 1.22 (to make it closer to how Python
        scalars work).  Due to its complexity and the fact that subclassing NumPy
        scalars is probably a bad idea to begin with.  There is probably room
        for adjustments here.
        """

        class myf_simple1(sctype):
            # [FUNCTION DOC] HELPER: empty subclass of the scalar type under test.
            pass

        class myf_simple2(sctype):
            # [FUNCTION DOC] HELPER: second empty subclass of the scalar type under test.
            pass

        def op_func(self, other):
            # [FUNCTION DOC] HELPER: stand-in operator that ignores operands and returns the
            # operator name.
            return __op__

        def rop_func(self, other):
            # [FUNCTION DOC] HELPER: stand-in reflected operator that returns the reflected
            # operator name.
            return __rop__

        myf_op = type("myf_op", (sctype,), {__op__: op_func, __rop__: rop_func})

        # inheritance has to override, or this is correctly lost:
        res = op(myf_simple1(1), myf_simple2(2))
        if type(res) is not sctype and type(res) is not np.bool_:
            raise AssertionError(
                f"Expected type(res) is sctype or np.bool_, got {type(res)}"
            )
        if op(myf_simple1(1), myf_simple2(2)) != op(1, 2):
            raise AssertionError("Expected inherited op to match")  # inherited

        # Two independent subclasses do not really define an order.  This could
        # be attempted, but we do not since Python's `int` does neither:
        if op(myf_op(1), myf_simple1(2)) != __op__:
            raise AssertionError(f"Expected op result == {__op__}")
        if op(myf_simple1(1), myf_op(2)) != op(1, 2):
            raise AssertionError("Expected inherited op to match")  # inherited

    @skip(reason="We do not support subclassing scalars.")
    @parametrize("__op__, __rop__, op, cmp", ops_with_names)
    @parametrize("subtype", [float, int, complex, np.float16])
    # @np._no_nep50_warning()
    def test_pyscalar_subclasses(self, subtype, __op__, __rop__, op, cmp):
        # [FUNCTION DOC] Skipped. Checks Python float, int, complex and float16 subclasses that
        # define operators with array ufunc disabled defer correctly against numpy scalars, and
        # without that marker are treated as ordinary subclasses with results of the base type.
        def op_func(self, other):
            # [FUNCTION DOC] HELPER: stand-in operator that ignores operands and returns the
            # operator name.
            return __op__

        def rop_func(self, other):
            # [FUNCTION DOC] HELPER: stand-in reflected operator that returns the reflected
            # operator name.
            return __rop__

        # Check that deferring is indicated using `__array_ufunc__`:
        myt = type(
            "myt",
            (subtype,),
            {__op__: op_func, __rop__: rop_func, "__array_ufunc__": None},
        )

        # Just like normally, we should never presume we can modify the float.
        if op(myt(1), np.float64(2)) != __op__:
            raise AssertionError(f"Expected op result == {__op__}")
        if op(np.float64(1), myt(2)) != __rop__:
            raise AssertionError(f"Expected rop result == {__rop__}")

        if op in {operator.mod, operator.floordiv} and subtype is complex:
            return  # module is not support for complex.  Do not test.

        if __rop__ == __op__:
            return

        # When no deferring is indicated, subclasses are handled normally.
        myt = type("myt", (subtype,), {__rop__: rop_func})

        # Check for float32, as a float subclass float64 may behave differently
        res = op(myt(1), np.float16(2))
        expected = op(subtype(1), np.float16(2))
        if res != expected:
            raise AssertionError(f"Expected res == {expected}, got {res}")
        if type(res) is not type(expected):
            raise AssertionError(
                f"Expected type(res) is {type(expected)}, got {type(res)}"
            )
        res = op(np.float32(2), myt(1))
        expected = op(np.float32(2), subtype(1))
        if res != expected:
            raise AssertionError(f"Expected res == {expected}, got {res}")
        if type(res) is not type(expected):
            raise AssertionError(
                f"Expected type(res) is {type(expected)}, got {type(res)}"
            )


if __name__ == "__main__":
    run_tests()
