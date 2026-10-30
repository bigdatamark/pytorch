# Owner(s): ["module: dynamo"]


# [MODULE DOC] Tests for numpy-style type-checking helpers (real/imag, iscomplex, isreal,
#   isnan, isinf, nan_to_num, real_if_close, common_type) run against torch._numpy, or against
#   real numpy when TEST_WITH_TORCHDYNAMO is set. Several classes are marked expected failure
#   because the functions are not implemented. Import-time effect: picks the numpy backend and
#   defines the skip decorator.
import functools
from unittest import expectedFailure as xfail, skipIf as skipif

from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from numpy.testing import assert_, assert_array_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import (
        common_type,
        iscomplex,
        iscomplexobj,
        isneginf,
        isposinf,
        isreal,
        isrealobj,
        nan_to_num,
        real_if_close,
    )
    from torch._numpy.testing import assert_, assert_array_equal, assert_equal


skip = functools.partial(skipif, True)


def assert_all(x):
    # [FUNCTION DOC] HELPER: Asserts that every element of the given array-like is truthy,
    #   using the numpy-style assert_ and np.all; the array is shown in the failure
    #   message.
    assert_(np.all(x), x)


@xpassIfTorchDynamo_np  # (reason="common_type not implemented")
class TestCommonType(TestCase):
    # [CLASS DOC] Checks common_type picks the widest float or complex type among its
    #   inputs. Marked to pass-or-fail leniently under dynamo via xpassIfTorchDynamo_np.
    def test_basic(self):
        # [FUNCTION DOC] Builds int32, float16, float32, float64, complex64 and
        #   complex128 2x2 arrays and asserts common_type returns float64 for int32
        #   and otherwise the input's own type.
        ai32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        af16 = np.array([[1, 2], [3, 4]], dtype=np.float16)
        af32 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        af64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        acs = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.csingle)
        acd = np.array([[1 + 5j, 2 + 6j], [3 + 7j, 4 + 8j]], dtype=np.cdouble)
        assert_(common_type(ai32) == np.float64)
        assert_(common_type(af16) == np.float16)
        assert_(common_type(af32) == np.float32)
        assert_(common_type(af64) == np.float64)
        assert_(common_type(acs) == np.csingle)
        assert_(common_type(acd) == np.cdouble)


@xfail  # (reason="not implemented")
class TestMintypecode(TestCase):
    # [CLASS DOC] Tests for mintypecode, whose cases are all expected failures.
    # KNOWN ISSUE: mintypecode is never imported in this module, so any run would raise
    #   NameError; the xfail decorator hides this.
    def test_default_1(self):
        # [FUNCTION DOC] Asserts integer-like type codes map to "d", and f, d, F, D
        #   map to themselves.
        for itype in "1bcsuwil":
            assert_equal(mintypecode(itype), "d")
        assert_equal(mintypecode("f"), "f")
        assert_equal(mintypecode("d"), "d")
        assert_equal(mintypecode("F"), "F")
        assert_equal(mintypecode("D"), "D")

    def test_default_2(self):
        # [FUNCTION DOC] Asserts mintypecode of two-character code strings (an integer
        #   code or float/complex pair) resolves to the wider of the float and complex
        #   kinds.
        for itype in "1bcsuwil":
            assert_equal(mintypecode(itype + "f"), "f")
            assert_equal(mintypecode(itype + "d"), "d")
            assert_equal(mintypecode(itype + "F"), "F")
            assert_equal(mintypecode(itype + "D"), "D")
        assert_equal(mintypecode("ff"), "f")
        assert_equal(mintypecode("fd"), "d")
        assert_equal(mintypecode("fF"), "F")
        assert_equal(mintypecode("fD"), "D")
        assert_equal(mintypecode("df"), "d")
        assert_equal(mintypecode("dd"), "d")
        # assert_equal(mintypecode('dF',savespace=1),'F')
        assert_equal(mintypecode("dF"), "D")
        assert_equal(mintypecode("dD"), "D")
        assert_equal(mintypecode("Ff"), "F")
        # assert_equal(mintypecode('Fd',savespace=1),'F')
        assert_equal(mintypecode("Fd"), "D")
        assert_equal(mintypecode("FF"), "F")
        assert_equal(mintypecode("FD"), "D")
        assert_equal(mintypecode("Df"), "D")
        assert_equal(mintypecode("Dd"), "D")
        assert_equal(mintypecode("DF"), "D")
        assert_equal(mintypecode("DD"), "D")

    def test_default_3(self):
        # [FUNCTION DOC] Asserts mintypecode of three-character code strings mixing
        #   float, double and complex codes resolves to the widest, with integer codes
        #   not lowering the result.
        assert_equal(mintypecode("fdF"), "D")
        # assert_equal(mintypecode('fdF',savespace=1),'F')
        assert_equal(mintypecode("fdD"), "D")
        assert_equal(mintypecode("fFD"), "D")
        assert_equal(mintypecode("dFD"), "D")

        assert_equal(mintypecode("ifd"), "d")
        assert_equal(mintypecode("ifF"), "F")
        assert_equal(mintypecode("ifD"), "D")
        assert_equal(mintypecode("idF"), "D")
        # assert_equal(mintypecode('idF',savespace=1),'F')
        assert_equal(mintypecode("idD"), "D")


@xpassIfTorchDynamo_np  # (reason="TODO: decide on if [1] is a scalar or not")
class TestIsscalar(TestCase):
    # [CLASS DOC] Tests np.isscalar; xpass-marked because list-of-one scalar semantics are
    #   undecided.
    def test_basic(self):
        # [FUNCTION DOC] Asserts python numbers (int, complex, float) are scalars and
        #   a list or tuple holding one number is not.
        assert_(np.isscalar(3))
        assert_(not np.isscalar([3]))
        assert_(not np.isscalar((3,)))
        assert_(np.isscalar(3j))
        assert_(np.isscalar(4.0))


class TestReal(TestCase):
    # [CLASS DOC] Tests np.real on real and complex arrays, 0-d arrays and python numbers.
    def test_real(self):
        # [FUNCTION DOC] For a random real array, a 0-d array and a python int,
        #   asserts real returns the same values; array inputs must yield an ndarray
        #   result.
        y = np.random.rand(
            10,
        )
        assert_array_equal(y, np.real(y))

        y = np.array(1)
        out = np.real(y)
        assert_array_equal(y, out)
        assert_(isinstance(out, np.ndarray))

        y = 1
        out = np.real(y)
        assert_equal(y, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar

    def test_cmplx(self):
        # [FUNCTION DOC] For a random complex array, a 0-d complex array and a python
        #   complex, asserts real equals the real part; array inputs must yield an
        #   ndarray result.
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        assert_array_equal(y.real, np.real(y))

        y = np.array(1 + 1j)
        out = np.real(y)
        assert_array_equal(y.real, out)
        assert_(isinstance(out, np.ndarray))

        y = 1 + 1j
        out = np.real(y)
        assert_equal(1.0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar


class TestImag(TestCase):
    # [CLASS DOC] Tests np.imag on real and complex arrays, 0-d arrays and python numbers.
    def test_real(self):
        # [FUNCTION DOC] Asserts imag of real inputs (array, 0-d array, python int) is
        #   zero; array inputs must yield an ndarray result.
        y = np.random.rand(
            10,
        )
        assert_array_equal(0, np.imag(y))

        y = np.array(1)
        out = np.imag(y)
        assert_array_equal(0, out)
        assert_(isinstance(out, np.ndarray))

        y = 1
        out = np.imag(y)
        assert_equal(0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar

    def test_cmplx(self):
        # [FUNCTION DOC] Asserts imag of complex inputs (array, 0-d array, python
        #   complex) equals the imaginary part; array inputs must yield an ndarray
        #   result.
        y = np.random.rand(
            10,
        ) + 1j * np.random.rand(
            10,
        )
        assert_array_equal(y.imag, np.imag(y))

        y = np.array(1 + 1j)
        out = np.imag(y)
        assert_array_equal(y.imag, out)
        assert_(isinstance(out, np.ndarray))

        y = 1 + 1j
        out = np.imag(y)
        assert_equal(1.0, out)
        # assert_(not isinstance(out, np.ndarray))  # XXX: 0D tensor, not scalar


class TestIscomplex(TestCase):
    # [CLASS DOC] Tests iscomplex elementwise detection of nonzero imaginary parts.
    def test_fail(self):
        # [FUNCTION DOC] Asserts iscomplex on an all-real integer array is false
        #   everywhere.
        z = np.array([-1, 0, 1])
        res = iscomplex(z)
        assert_(not np.any(res, axis=0))

    def test_pass(self):
        # [FUNCTION DOC] Asserts iscomplex on a mixed array marks only the entry with
        #   nonzero imaginary part.
        z = np.array([-1j, 1, 0])
        res = iscomplex(z)
        assert_array_equal(res, [1, 0, 0])


class TestIsreal(TestCase):
    # [CLASS DOC] Tests isreal elementwise detection of zero imaginary parts.
    def test_pass(self):
        # [FUNCTION DOC] Asserts isreal marks the real entries of an array with one
        #   imaginary entry.
        z = np.array([-1, 0, 1j])
        res = isreal(z)
        assert_array_equal(res, [1, 1, 0])

    def test_fail(self):
        # [FUNCTION DOC] Asserts isreal is false only where the imaginary part is
        #   nonzero, using a different position than the previous test.
        z = np.array([-1j, 1, 0])
        res = isreal(z)
        assert_array_equal(res, [0, 1, 1])

    def test_isreal_real(self):
        # [FUNCTION DOC] Asserts isreal on an all-real integer array is true
        #   everywhere; raises AssertionError otherwise.
        z = np.array([-1, 0, 1])
        res = isreal(z)
        if not res.all():
            raise AssertionError("Expected res.all() to be True")


class TestIscomplexobj(TestCase):
    # [CLASS DOC] Tests iscomplexobj on arrays, scalars and lists.
    def test_basic(self):
        # [FUNCTION DOC] Asserts an integer array is not complex and an array with a
        #   complex entry is.
        z = np.array([-1, 0, 1])
        assert_(not iscomplexobj(z))
        z = np.array([-1j, 0, -1])
        assert_(iscomplexobj(z))

    def test_scalar(self):
        # [FUNCTION DOC] Asserts a python float is not complex and a python complex
        #   is.
        assert_(not iscomplexobj(1.0))
        assert_(iscomplexobj(1 + 0j))

    def test_list(self):
        # [FUNCTION DOC] Asserts a list containing a complex number is complex, and a
        #   list of int and bool is not.
        assert_(iscomplexobj([3, 1 + 0j, True]))
        assert_(not iscomplexobj([3, 1, True]))


class TestIsrealobj(TestCase):
    # [CLASS DOC] Tests isrealobj, the inverse of iscomplexobj for array inputs.
    def test_basic(self):
        # [FUNCTION DOC] Asserts an integer array is real and a complex-containing
        #   array is not.
        z = np.array([-1, 0, 1])
        assert_(isrealobj(z))
        z = np.array([-1j, 0, -1])
        assert_(not isrealobj(z))


class TestIsnan(TestCase):
    # [CLASS DOC] Tests np.isnan on finite values, infinities, NaN and complex inputs.
    def test_goodvalues(self):
        # [FUNCTION DOC] Asserts isnan is false for ordinary finite values.
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isnan(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # [FUNCTION DOC] Asserts isnan is false for positive infinity produced by
        #   dividing by zero.
        assert_all(np.isnan(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        # [FUNCTION DOC] Asserts isnan is false for negative infinity produced by
        #   dividing by zero.
        assert_all(np.isnan(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        # [FUNCTION DOC] Asserts isnan is true for zero divided by zero.
        assert_all(np.isnan(np.array((0.0,)) / 0.0) == 1)

    def test_integer(self):
        # [FUNCTION DOC] Asserts isnan of a python int is false.
        assert_all(np.isnan(1) == 0)

    def test_complex(self):
        # [FUNCTION DOC] Asserts isnan of a finite python complex is false.
        assert_all(np.isnan(1 + 1j) == 0)

    def test_complex1(self):
        # [FUNCTION DOC] Asserts isnan of a complex 0-d array divided by zero is true.
        assert_all(np.isnan(np.array(0 + 0j) / 0.0) == 1)


class TestIsfinite(TestCase):
    # [CLASS DOC] Tests np.isfinite; a leftover from numpy where it lived in the wrong
    #   module.
    # Fixme, wrong place, isfinite now ufunc

    def test_goodvalues(self):
        # [FUNCTION DOC] Asserts isfinite is true for ordinary finite values.
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isfinite(z) == 1
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # [FUNCTION DOC] Asserts isfinite is false for positive infinity from division
        #   by zero.
        assert_all(np.isfinite(np.array((1.0,)) / 0.0) == 0)

    def test_neginf(self):
        # [FUNCTION DOC] Asserts isfinite is false for negative infinity from division
        #   by zero.
        assert_all(np.isfinite(np.array((-1.0,)) / 0.0) == 0)

    def test_ind(self):
        # [FUNCTION DOC] Asserts isfinite is false for NaN from zero divided by zero.
        assert_all(np.isfinite(np.array((0.0,)) / 0.0) == 0)

    def test_integer(self):
        # [FUNCTION DOC] Asserts isfinite of a python int is true.
        assert_all(np.isfinite(1) == 1)

    def test_complex(self):
        # [FUNCTION DOC] Asserts isfinite of a finite python complex is true.
        assert_all(np.isfinite(1 + 1j) == 1)

    def test_complex1(self):
        # [FUNCTION DOC] Asserts isfinite of a complex 0-d array divided by zero is
        #   false.
        assert_all(np.isfinite(np.array(1 + 1j) / 0.0) == 0)


class TestIsinf(TestCase):
    # [CLASS DOC] Tests np.isinf; a leftover from numpy where it lived in the wrong
    #   module.
    # Fixme, wrong place, isinf now ufunc

    def test_goodvalues(self):
        # [FUNCTION DOC] Asserts isinf is false for ordinary finite values.
        z = np.array((-1.0, 0.0, 1.0))
        res = np.isinf(z) == 0
        assert_all(np.all(res, axis=0))

    def test_posinf(self):
        # [FUNCTION DOC] Asserts isinf is true for positive infinity from division by
        #   zero.
        assert_all(np.isinf(np.array((1.0,)) / 0.0) == 1)

    def test_posinf_scalar(self):
        # [FUNCTION DOC] Asserts isinf is true for a 0-d array holding positive
        #   infinity.
        assert_all(
            np.isinf(
                np.array(
                    1.0,
                )
                / 0.0
            )
            == 1
        )

    def test_neginf(self):
        # [FUNCTION DOC] Asserts isinf is true for negative infinity from division by
        #   zero.
        assert_all(np.isinf(np.array((-1.0,)) / 0.0) == 1)

    def test_neginf_scalar(self):
        # [FUNCTION DOC] Asserts isinf is true for a 0-d array holding negative
        #   infinity.
        assert_all(np.isinf(np.array(-1.0) / 0.0) == 1)

    def test_ind(self):
        # [FUNCTION DOC] Asserts isinf is false for NaN from zero divided by zero.
        assert_all(np.isinf(np.array((0.0,)) / 0.0) == 0)


class TestIsposinf(TestCase):
    # [CLASS DOC] Tests isposinf marks only positive infinity.
    def test_generic(self):
        # [FUNCTION DOC] Divides (-1, 0, 1) by zero and asserts isposinf is false,
        #   false, true.
        vals = isposinf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 0)
        assert_(vals[1] == 0)
        assert_(vals[2] == 1)


class TestIsneginf(TestCase):
    # [CLASS DOC] Tests isneginf marks only negative infinity.
    def test_generic(self):
        # [FUNCTION DOC] Divides (-1, 0, 1) by zero and asserts isneginf is true,
        #   false, false.
        vals = isneginf(np.array((-1.0, 0, 1)) / 0.0)
        assert_(vals[0] == 1)
        assert_(vals[1] == 0)
        assert_(vals[2] == 0)


# @xfail  #(reason="not implemented")
class TestNanToNum(TestCase):
    # [CLASS DOC] Tests nan_to_num replacing NaN and infinities with finite numbers or
    #   custom values.
    def test_generic(self):
        # [FUNCTION DOC] Checks default replacement: negative infinity becomes a very
        #   large negative finite number, NaN becomes 0, positive infinity a very
        #   large positive number. Then checks the nan, posinf and neginf keywords
        #   (10, 20, 30) give 30, 10, 20. Results must be ndarrays.
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0)
        assert_all(vals[0] < -1e10) and assert_all(np.isfinite(vals[0]))
        assert_(vals[1] == 0)
        assert_all(vals[2] > 1e10) and assert_all(np.isfinite(vals[2]))
        if not isinstance(vals, np.ndarray):
            raise AssertionError(f"Expected vals to be np.ndarray, got {type(vals)}")

        # perform the same tests but with nan, posinf and neginf keywords
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=10, posinf=20, neginf=30)
        assert_equal(vals, [30, 10, 20])
        assert_all(np.isfinite(vals[[0, 2]]))
        if not isinstance(vals, np.ndarray):
            raise AssertionError(f"Expected vals to be np.ndarray, got {type(vals)}")

    def test_array(self):
        # [FUNCTION DOC] Asserts nan_to_num of a one-element list returns an ndarray
        #   equal to the integer array, with and without the replacement keywords.
        vals = nan_to_num([1])
        assert_array_equal(vals, np.array([1], int))
        if not isinstance(vals, np.ndarray):
            raise AssertionError(f"Expected vals to be np.ndarray, got {type(vals)}")
        vals = nan_to_num([1], nan=10, posinf=20, neginf=30)
        assert_array_equal(vals, np.array([1], int))
        if not isinstance(vals, np.ndarray):
            raise AssertionError(f"Expected vals to be np.ndarray, got {type(vals)}")

    @skip(reason="we return OD arrays not scalars")
    def test_integer(self):
        # [FUNCTION DOC] Skipped (scalars come back as 0-d arrays). Would check an int
        #   stays unchanged and has numpy int type, with and without keywords.
        vals = nan_to_num(1)
        assert_all(vals == 1)
        if not isinstance(vals, np.int_):
            raise AssertionError(f"Expected vals to be np.int_, got {type(vals)}")
        vals = nan_to_num(1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1)
        if not isinstance(vals, np.int_):
            raise AssertionError(f"Expected vals to be np.int_, got {type(vals)}")

    @skip(reason="we return OD arrays not scalars")
    def test_float(self):
        # [FUNCTION DOC] Skipped (scalars come back as 0-d arrays). Would check python
        #   floats stay unchanged and keep numpy float64 type.
        vals = nan_to_num(1.0)
        assert_all(vals == 1.0)
        assert_equal(type(vals), np.float64)
        vals = nan_to_num(1.1, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1.1)
        assert_equal(type(vals), np.float64)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_good(self):
        # [FUNCTION DOC] Skipped (scalars come back as 0-d arrays). Would check a
        #   finite complex number is unchanged and has numpy complex128 type.
        vals = nan_to_num(1 + 1j)
        assert_all(vals == 1 + 1j)
        if not isinstance(vals, np.complex128):
            raise AssertionError(f"Expected vals to be np.complex128, got {type(vals)}")
        vals = nan_to_num(1 + 1j, nan=10, posinf=20, neginf=30)
        assert_all(vals == 1 + 1j)
        assert_equal(type(vals), np.complex128)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad(self):
        # [FUNCTION DOC] Skipped. Builds a complex value with an infinite part and
        #   asserts nan_to_num makes it finite with complex128 type; the source notes
        #   the result is unexpectedly zero.
        v = 1 + 1j
        v += np.array(0 + 1.0j) / 0.0
        vals = nan_to_num(v)
        # !! This is actually (unexpectedly) zero
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex128)

    @skip(reason="we return OD arrays not scalars")
    def test_complex_bad2(self):
        # [FUNCTION DOC] Skipped. Like the previous test with a negative real part;
        #   asserts finiteness and complex128 type; stricter magnitude checks are left
        #   commented out.
        v = 1 + 1j
        v += np.array(-1 + 1.0j) / 0.0
        vals = nan_to_num(v)
        assert_all(np.isfinite(vals))
        assert_equal(type(vals), np.complex128)
        # Fixme
        # assert_all(vals.imag > 1e10)  and assert_all(np.isfinite(vals))
        # !! This is actually (unexpectedly) positive
        # !! inf.  Comment out for now, and see if it
        # !! changes
        # assert_all(vals.real < -1e10) and assert_all(np.isfinite(vals))

    def test_do_not_rewrite_previous_keyword(self):
        # [FUNCTION DOC] Asserts replacements do not chain: with NaN mapped to
        #   infinity and positive infinity mapped to 999, the NaN slot stays infinite
        #   while the infinite slot becomes 999 and negative infinity becomes a large
        #   finite number. Result must be an ndarray.
        # This is done to test that when, for instance, nan=np.inf then these
        # values are not rewritten by posinf keyword to the posinf value.
        vals = nan_to_num(np.array((-1.0, 0, 1)) / 0.0, nan=np.inf, posinf=999)
        assert_all(np.isfinite(vals[[0, 2]]))
        assert_all(vals[0] < -1e10)
        assert_equal(vals[[1, 2]], [np.inf, 999])
        if not isinstance(vals, np.ndarray):
            raise AssertionError(f"Expected vals to be np.ndarray, got {type(vals)}")


class TestRealIfClose(TestCase):
    # [CLASS DOC] Tests real_if_close tolerance handling.
    def test_basic(self):
        # [FUNCTION DOC] A tiny imaginary part is dropped to a real array equal to the
        #   input; a larger one keeps the result complex; passing a looser tolerance
        #   makes it real again.
        a = np.random.rand(10)
        b = real_if_close(a + 1e-15j)
        assert_all(isrealobj(b))
        assert_array_equal(a, b)
        b = real_if_close(a + 1e-7j)
        assert_all(iscomplexobj(b))
        b = real_if_close(a + 1e-7j, tol=1e-6)
        assert_all(isrealobj(b))


@xfail  # (reason="not implemented")
class TestArrayConversion(TestCase):
    # [CLASS DOC] Tests array conversion helpers; expected failure as not implemented.
    # KNOWN ISSUE: asfarray is never imported in this module, so the test would raise
    #   NameError.
    def test_asfarray(self):
        # [FUNCTION DOC] Asserts asfarray returns a plain ndarray of floating dtype,
        #   and raises TypeError when a dtype given as an array is passed.
        a = asfarray(np.array([1, 2, 3]))
        assert_equal(a.__class__, np.ndarray)
        assert_(np.issubdtype(a.dtype, np.floating))

        # previously this would infer dtypes from arrays, unlike every single
        # other numpy function
        assert_raises(TypeError, asfarray, np.array([1, 2, 3]), dtype=np.array(1.0))


if __name__ == "__main__":
    run_tests()
