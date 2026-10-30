# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the finfo and iinfo machine limit functions of torch._numpy (real NumPy
# under dynamo). Compares attributes of float and integer type info across string and type
# spellings, checks unsigned maximums, checks the repr text, and holds several skipped tests
# (singleton identity, MachAr discovery) that torch._numpy does not support.

"""Test functions for limits module."""

import functools
import warnings
from unittest import expectedFailure as xfail, skipIf

import numpy
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy import double, finfo, half, iinfo, single
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import double, finfo, half, iinfo, single
    from torch._numpy.testing import assert_, assert_equal


skip = functools.partial(skipIf, True)

##################################################


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestPythonFloat(TestCase):
    # [CLASS DOC] Skipped test case that wants finfo of the Python float type to be a
    # singleton.
    def test_singleton(self):
        # [FUNCTION DOC] Call finfo twice and require the two results to be the same
        # object.
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestHalf(TestCase):
    # [CLASS DOC] Skipped test case that wants finfo of the half type to be a singleton.
    def test_singleton(self):
        # [FUNCTION DOC] Call finfo twice for half and require the two results to be the
        # same object.
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestSingle(TestCase):
    # [CLASS DOC] Skipped test case that wants finfo of the single type to be a singleton.
    def test_singleton(self):
        # [FUNCTION DOC] Call finfo twice for single and require identical objects.
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestDouble(TestCase):
    # [CLASS DOC] Skipped test case that wants finfo of the double type to be a singleton.
    def test_singleton(self):
        # [FUNCTION DOC] Call finfo twice for double and require identical objects.
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))


class TestFinfo(TestCase):
    # [CLASS DOC] Tests of the float type info function.
    @skipIf(numpy.__version__ < "1.23", reason=".smallest_normal is new")
    def test_basic(self):
        # [FUNCTION DOC] For float and complex types given as short strings and as types,
        # check that bits, eps, max, min, resolution, tiny and smallest_normal agree
        # between the two spellings, and that finfo of an integer string raises TypeError
        # or ValueError. Skipped for NumPy older than 1.23.
        dts = list(
            zip(
                ["f2", "f4", "f8", "c8", "c16"],
                [np.float16, np.float32, np.float64, np.complex64, np.complex128],
            )
        )
        for dt1, dt2 in dts:
            for attr in (
                "bits",
                "eps",
                "max",
                "min",
                "resolution",
                "tiny",
                "smallest_normal",
            ):
                assert_equal(getattr(finfo(dt1), attr), getattr(finfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            finfo("i4")

    @skip  # (reason="Some of these attributes are not implemented vs NP versions")
    def test_basic_missing(self):
        # [FUNCTION DOC] Read several float info attributes that torch does not
        # implement; skipped.
        dt = np.float32
        for attr in [
            "epsneg",
            "iexp",
            "machep",
            "maxexp",
            "minexp",
            "negep",
            "nexp",
            "nmant",
            "precision",
            "smallest_subnormal",
        ]:
            getattr(finfo(dt), attr)


@instantiate_parametrized_tests
class TestIinfo(TestCase):
    # [CLASS DOC] Tests of the integer type info function.
    def test_basic(self):
        # [FUNCTION DOC] For integer short strings paired with integer types, check bits,
        # min and max agree between spellings, and that iinfo of a float string raises
        # TypeError or ValueError.
        # KNOWN ISSUE: the string list has eight entries but the type list five, and the
        # pairing stops at the shorter list, so the last three strings are never tested.
        dts = list(
            zip(
                ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"],
                [
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                ],
            )
        )
        for dt1, dt2 in dts:
            for attr in ("bits", "min", "max"):
                assert_equal(getattr(iinfo(dt1), attr), getattr(iinfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            iinfo("f4")

    @parametrize(
        "T",
        [
            np.uint8,
            # xfail: unsupported add (uint[16,32,64])
            subtest(np.uint16, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
            subtest(np.uint32, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
            subtest(np.uint64, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
        ],
    )
    def test_unsigned_max(self, T):
        # [FUNCTION DOC] For each unsigned type parametrized, check its maximum equals
        # zero minus one computed in that type (wraparound). The 16, 32 and 64 bit cases
        # are expected failures unless running under dynamo.
        max_calculated = T(0) - T(1)
        assert_equal(iinfo(T).max, max_calculated)


class TestRepr(TestCase):
    # [CLASS DOC] Tests of the text form of info objects.
    def test_iinfo_repr(self):
        # [FUNCTION DOC] Check the repr of iinfo for int16 equals the expected text with
        # min, max and dtype.
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    @skipIf(TEST_WITH_TORCHDYNAMO, reason="repr differs")
    def test_finfo_repr(self):
        # [FUNCTION DOC] Check the repr of finfo for float32 contains the expected
        # resolution and min prefix and the dtype text. Skipped under dynamo where the
        # repr differs.
        repr_f32 = repr(np.finfo(np.float32))
        if "finfo(resolution=1e-06, min=-3.40282e+38," not in repr_f32:
            raise AssertionError(
                f"Expected finfo repr to contain expected prefix, got: {repr_f32}"
            )
        if "dtype=float32" not in repr_f32:
            raise AssertionError(
                f"Expected finfo repr to contain 'dtype=float32', got: {repr_f32}"
            )


def assert_ma_equal(discovered, ma_like):
    # [FUNCTION DOC] HELPER: Compare every attribute of a discovered machine-arithmetic
    # object with the one of a reference object, including shape and dtype of attributes that
    # have them.
    # Check MachAr-like objects same as calculated MachAr instances
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, "shape"):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


class TestMisc(TestCase):
    # [CLASS DOC] Miscellaneous limit tests, mostly skipped.
    @skip(reason="Instantiate {i,f}info from dtypes.")
    def test_instances(self):
        # [FUNCTION DOC] Build integer and float info from a number; skipped, not
        # supported.
        iinfo(10)
        finfo(3.0)

    @skip(reason="MachAr no implemented (does it need to)?")
    def test_known_types(self):
        # [FUNCTION DOC] Compare discovered machine parameters for float16, float32,
        # float64 and longdouble with reference tables; skipped.
        # KNOWN ISSUE: names for the machine discovery function and the reference table
        # are not defined or imported in this file, so the test would fail with NameError
        # if it ran.
        # Test we are correctly compiling parameters for known types
        for ftype, ma_like in (
            (np.float16, _float_ma[16]),
            (np.float32, _float_ma[32]),
            (np.float64, _float_ma[64]),
        ):
            assert_ma_equal(_discovered_machar(ftype), ma_like)
        # Suppress warning for broken discovery of double double on PPC
        ld_ma = _discovered_machar(np.longdouble)
        bytes = np.dtype(np.longdouble).itemsize
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 80-bit extended precision
            assert_ma_equal(ld_ma, _float_ma[80])
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # IEE 754 128-bit
            assert_ma_equal(ld_ma, _float_ma[128])

    @skip(reason="MachAr no implemented (does it need to be)?")
    def test_subnormal_warning(self):
        # [FUNCTION DOC] Check that reading the smallest subnormal of longdouble
        # parameters does not raise warnings; skipped.
        # KNOWN ISSUE: relies on an undefined machine discovery function, so it would
        # raise NameError.
        """Test that the subnormal is zero warning is not being raised."""
        ld_ma = _discovered_machar(np.longdouble)
        bytes = np.dtype(np.longdouble).itemsize
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
                # 80-bit extended precision
                ld_ma.smallest_subnormal
                if len(w) != 0:
                    raise AssertionError(f"Expected no warnings, got {len(w)}")
            elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
                # IEE 754 128-bit
                ld_ma.smallest_subnormal
                if len(w) != 0:
                    raise AssertionError(f"Expected no warnings, got {len(w)}")
            else:
                # Double double
                ld_ma.smallest_subnormal
                # This test may fail on some platforms
                if len(w) != 0:
                    raise AssertionError(f"Expected no warnings, got {len(w)}")

    @xpassIfTorchDynamo_np  # (reason="None of nmant, minexp, maxexp is implemented.")
    def test_plausible_finfo(self):
        # [FUNCTION DOC] For float16, float32, float64, longdouble, complex types and, if
        # present, clongdouble, check finfo gives mantissa size above one, minimum
        # exponent below minus one and maximum exponent above one. Passes only under
        # dynamo (xpass decorator).
        # Assert that finfo returns reasonable results for all types
        for ftype in (
            [np.float16, np.float32, np.float64, np.longdouble]
            + [
                np.complex64,
                np.complex128,
            ]
            # no complex256 in torch._numpy
            + ([np.clongdouble] if hasattr(np, "clongdouble") else [])
        ):
            info = np.finfo(ftype)
            assert_(info.nmant > 1)
            assert_(info.minexp < -1)
            assert_(info.maxexp > 1)


if __name__ == "__main__":
    run_tests()
