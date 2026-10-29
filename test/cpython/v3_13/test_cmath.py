# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_cmath suite for complex math:
# tests the cmath module constants, conversion protocol (__complex__/__float__), agreement
# with math, table-driven special values from a shared test file, polar/rect/phase/abs,
# classification predicates, signed-zero behavior, and cmath.isclose via IsCloseTests.
# Import side effects: installs RedirectImportFinder at the front of sys.meta_path so test.*
# imports of a few helper test modules resolve to top-level modules; defines shared constant
# lists of complex zeros, infinities and NaNs.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_cmath.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase


# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    # [CLASS DOC] HELPER: meta-path finder that redirects imports of selected stdlib test
    # helper modules (listed in redirect_imports) to the same-named top-level modules without
    # the test prefix.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: returns a module spec for a redirected name, else None. If
        # fullname is in redirect_imports, imports the module with the test prefix removed,
        # registers it in sys.modules under the original name (side effect), and returns its
        # spec; an ImportError yields None. All other names yield None so normal finders
        # handle them.
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

from test.support import requires_IEEE_754, cpython_only, import_helper
from test.test_math import parse_testfile, test_file
import test.test_math as test_math
import unittest
import cmath, math
from cmath import phase, polar, rect, pi
import platform
import sys


INF = float('inf')
NAN = float('nan')

complex_zeros = [complex(x, y) for x in [0.0, -0.0] for y in [0.0, -0.0]]
complex_infinities = [complex(x, y) for x, y in [
        (INF, 0.0),  # 1st quadrant
        (INF, 2.3),
        (INF, INF),
        (2.3, INF),
        (0.0, INF),
        (-0.0, INF), # 2nd quadrant
        (-2.3, INF),
        (-INF, INF),
        (-INF, 2.3),
        (-INF, 0.0),
        (-INF, -0.0), # 3rd quadrant
        (-INF, -2.3),
        (-INF, -INF),
        (-2.3, -INF),
        (-0.0, -INF),
        (0.0, -INF), # 4th quadrant
        (2.3, -INF),
        (INF, -INF),
        (INF, -2.3),
        (INF, -0.0)
        ]]
complex_nans = [complex(x, y) for x, y in [
        (NAN, -INF),
        (NAN, -2.3),
        (NAN, -0.0),
        (NAN, 0.0),
        (NAN, 2.3),
        (NAN, INF),
        (-INF, NAN),
        (-2.3, NAN),
        (-0.0, NAN),
        (0.0, NAN),
        (2.3, NAN),
        (INF, NAN)
        ]]

class CMathTests(__TestCase):
    # [CLASS DOC] Tests of the cmath functions. Class attribute test_functions lists every
    # one-argument cmath function plus two lambdas fixing either argument of two-argument log.
    # setUp opens the shared special-values file into test_values; helper assertions compare
    # floats and complex numbers with sign-aware identity or ulp-scale tolerance.
    # list of all functions in cmath
    test_functions = [getattr(cmath, fname) for fname in [
            'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh',
            'cos', 'cosh', 'exp', 'log', 'log10', 'sin', 'sinh',
            'sqrt', 'tan', 'tanh']]
    # test first and second arguments independently for 2-argument log
    test_functions.append(lambda x : cmath.log(x, 1729. + 0j))
    test_functions.append(lambda x : cmath.log(14.-27j, x))

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp, then opens the shared test data file (UTF-8)
        # and keeps the handle in test_values. The handle is closed in tearDown.
        super().setUp()
        self.test_values = open(test_file, encoding="utf-8")

    def tearDown(self):
        # [FUNCTION DOC] Closes the test_values file opened in setUp.
        # KNOWN ISSUE: does not call the parent tearDown, so base-class cleanup is skipped.
        self.test_values.close()

    def assertFloatIdentical(self, x, y):
        # [FUNCTION DOC] HELPER: fails unless floats x and y are identical: both NaN, or equal
        # and, if zero, with the same sign. Fails with a message naming both values (and
        # differing zero signs).
        """Fail unless floats x and y are identical, in the sense that:
        (1) both x and y are nans, or
        (2) both x and y are infinities, with the same sign, or
        (3) both x and y are zeros, with the same sign, or
        (4) x and y are both finite and nonzero, and x == y

        """
        msg = 'floats {!r} and {!r} are not identical'

        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            # both zero; check that signs match
            elif math.copysign(1.0, x) == math.copysign(1.0, y):
                return
            else:
                msg += ': zeros have different signs'
        self.fail(msg.format(x, y))

    def assertComplexesAreIdentical(self, x, y):
        # [FUNCTION DOC] HELPER: fails unless the real parts and the imaginary parts of
        # complex x and y are each identical per assertFloatIdentical, so zero signs
        # must match.
        """Fail unless complex numbers x and y have equal values and signs.

        In particular, if x and y both have real (or imaginary) part
        zero, but the zeros have different signs, this test will fail.

        """
        self.assertFloatIdentical(x.real, y.real)
        self.assertFloatIdentical(x.imag, y.imag)

    def rAssertAlmostEqual(self, a, b, rel_err = 2e-15, abs_err = 5e-323,
                           msg=None):
        # [FUNCTION DOC] HELPER: fails unless floats a and b are close. NaN a requires NaN b;
        # infinite a requires equal b; two zeros must share a sign; otherwise passes if the
        # absolute difference is within the larger of abs_err and rel_err times the magnitude
        # of a. An overflow while computing the difference counts as a failure. Optional msg
        # overrides the failure text.
        """Fail if the two floating-point numbers are not almost equal.

        Determine whether floating-point values a and b are equal to within
        a (small) rounding error.  The default values for rel_err and
        abs_err are chosen to be suitable for platforms where a float is
        represented by an IEEE 754 double.  They allow an error of between
        9 and 19 ulps.
        """

        # special values testing
        if math.isnan(a):
            if math.isnan(b):
                return
            self.fail(msg or '{!r} should be nan'.format(b))

        if math.isinf(a):
            if a == b:
                return
            self.fail(msg or 'finite result where infinity expected: '
                      'expected {!r}, got {!r}'.format(a, b))

        # if both a and b are zero, check whether they have the same sign
        # (in theory there are examples where it would be legitimate for a
        # and b to have opposite signs; in practice these hardly ever
        # occur).
        if not a and not b:
            if math.copysign(1., a) != math.copysign(1., b):
                self.fail(msg or 'zero has wrong sign: expected {!r}, '
                          'got {!r}'.format(a, b))

        # if a-b overflows, or b is infinite, return False.  Again, in
        # theory there are examples where a is within a few ulps of the
        # max representable float, and then b could legitimately be
        # infinite.  In practice these examples are rare.
        try:
            absolute_error = abs(b-a)
        except OverflowError:
            pass
        else:
            # test passes if either the absolute error or the relative
            # error is sufficiently small.  The defaults amount to an
            # error of between 9 ulps and 19 ulps on an IEEE-754 compliant
            # machine.
            if absolute_error <= max(abs_err, rel_err * abs(a)):
                return
        self.fail(msg or
                  '{!r} and {!r} are not sufficiently close'.format(a, b))

    def test_constants(self):
        # [FUNCTION DOC] Checks cmath.pi and cmath.e match the reference values to 9
        # decimal places.
        e_expected = 2.71828182845904523536
        pi_expected = 3.14159265358979323846
        self.assertAlmostEqual(cmath.pi, pi_expected, places=9,
            msg="cmath.pi is {}; should be {}".format(cmath.pi, pi_expected))
        self.assertAlmostEqual(cmath.e, e_expected, places=9,
            msg="cmath.e is {}; should be {}".format(cmath.e, e_expected))

    def test_infinity_and_nan_constants(self):
        # [FUNCTION DOC] Checks cmath.inf, infj, nan and nanj have the expected real and
        # imaginary parts, positive signs, and the expected repr strings.
        self.assertEqual(cmath.inf.real, math.inf)
        self.assertEqual(cmath.inf.imag, 0.0)
        self.assertEqual(cmath.infj.real, 0.0)
        self.assertEqual(cmath.infj.imag, math.inf)

        self.assertTrue(math.isnan(cmath.nan.real))
        self.assertEqual(cmath.nan.imag, 0.0)
        self.assertEqual(cmath.nanj.real, 0.0)
        self.assertTrue(math.isnan(cmath.nanj.imag))
        # Also check that the sign of all of these is positive:
        self.assertEqual(math.copysign(1., cmath.nan.real), 1.)
        self.assertEqual(math.copysign(1., cmath.nan.imag), 1.)
        self.assertEqual(math.copysign(1., cmath.nanj.real), 1.)
        self.assertEqual(math.copysign(1., cmath.nanj.imag), 1.)

        # Check consistency with reprs.
        self.assertEqual(repr(cmath.inf), "inf")
        self.assertEqual(repr(cmath.infj), "infj")
        self.assertEqual(repr(cmath.nan), "nan")
        self.assertEqual(repr(cmath.nanj), "nanj")

    def test_user_object(self):
        # [FUNCTION DOC] Checks every cmath function honors the __complex__ and __float__
        # protocols for user objects: __complex__ results are used, float-only objects work,
        # objects with only __int__ or __index__ raise TypeError, non-complex __complex__
        # results raise TypeError, and exceptions raised in __complex__ propagate.
        # Test automatic calling of __complex__ and __float__ by cmath
        # functions

        # some random values to use as test values; we avoid values
        # for which any of the functions in cmath is undefined
        # (i.e. 0., 1., -1., 1j, -1j) or would cause overflow
        cx_arg = 4.419414439 + 1.497100113j
        flt_arg = -6.131677725

        # a variety of non-complex numbers, used to check that
        # non-complex return values from __complex__ give an error
        non_complexes = ["not complex", 1, 5, 2., None,
                         object(), NotImplemented]

        # Now we introduce a variety of classes whose instances might
        # end up being passed to the cmath functions

        # usual case: new-style class implementing __complex__
        class MyComplex:
            # [CLASS DOC] HELPER: class whose __complex__ returns the value given
            # at construction.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: stores value on the instance.
                self.value = value
            def __complex__(self):
                # [FUNCTION DOC] HELPER: returns the stored value (may be a non-complex object
                # in negative tests).
                return self.value

        # classes for which __complex__ raises an exception
        class SomeException(Exception):
            # [CLASS DOC] HELPER: exception type raised by MyComplexException.
            pass
        class MyComplexException:
            # [CLASS DOC] HELPER: class whose __complex__ raises SomeException.
            def __complex__(self):
                # [FUNCTION DOC] HELPER: always raises SomeException.
                raise SomeException

        # some classes not providing __float__ or __complex__
        class NeitherComplexNorFloat(object):
            # [CLASS DOC] HELPER: class providing neither __float__ nor __complex__.
            pass
        class Index:
            # [CLASS DOC] HELPER: class providing only __int__ and __index__ (both one-line
            # defs returning 2).
            def __int__(self): return 2
            def __index__(self): return 2
        class MyInt:
            # [CLASS DOC] HELPER: class providing only __int__ (a one-line def returning 2).
            def __int__(self): return 2

        # other possible combinations of __float__ and __complex__
        # that should work
        class FloatAndComplex:
            # [CLASS DOC] HELPER: class providing both __float__ and __complex__.
            def __float__(self):
                # [FUNCTION DOC] HELPER: returns the enclosing float argument flt_arg.
                return flt_arg
            def __complex__(self):
                # [FUNCTION DOC] HELPER: returns the enclosing complex argument cx_arg.
                return cx_arg
        class JustFloat:
            # [CLASS DOC] HELPER: class providing only __float__.
            def __float__(self):
                # [FUNCTION DOC] HELPER: returns the enclosing float argument flt_arg.
                return flt_arg

        for f in self.test_functions:
            # usual usage
            self.assertEqual(f(MyComplex(cx_arg)), f(cx_arg))
            # other combinations of __float__ and __complex__
            self.assertEqual(f(FloatAndComplex()), f(cx_arg))
            self.assertEqual(f(JustFloat()), f(flt_arg))
            self.assertEqual(f(Index()), f(int(Index())))
            # TypeError should be raised for classes not providing
            # either __complex__ or __float__, even if they provide
            # __int__ or __index__:
            self.assertRaises(TypeError, f, NeitherComplexNorFloat())
            self.assertRaises(TypeError, f, MyInt())
            # non-complex return value from __complex__ -> TypeError
            for bad_complex in non_complexes:
                self.assertRaises(TypeError, f, MyComplex(bad_complex))
            # exceptions in __complex__ should be propagated correctly
            self.assertRaises(SomeException, f, MyComplexException())

    def test_input_type(self):
        # [FUNCTION DOC] Checks ints and floats are accepted by all cmath functions (same
        # result as the float conversion), while strings raise TypeError.
        # ints should be acceptable inputs to all cmath
        # functions, by virtue of providing a __float__ method
        for f in self.test_functions:
            for arg in [2, 2.]:
                self.assertEqual(f(arg), f(arg.__float__()))

        # but strings should give a TypeError
        for f in self.test_functions:
            for arg in ["a", "long_string", "0", "1j", ""]:
                self.assertRaises(TypeError, f, arg)

    def test_cmath_matches_math(self):
        # [FUNCTION DOC] Checks cmath functions on real arguments in each function's domain
        # agree with the math module (within tolerance) and give zero imaginary part,
        # including log with several bases.
        # check that corresponding cmath and math functions are equal
        # for floats in the appropriate range

        # test_values in (0, 1)
        test_values = [0.01, 0.1, 0.2, 0.5, 0.9, 0.99]

        # test_values for functions defined on [-1., 1.]
        unit_interval = test_values + [-x for x in test_values] + \
            [0., 1., -1.]

        # test_values for log, log10, sqrt
        positive = test_values + [1.] + [1./x for x in test_values]
        nonnegative = [0.] + positive

        # test_values for functions defined on the whole real line
        real_line = [0.] + positive + [-x for x in positive]

        test_functions = {
            'acos' : unit_interval,
            'asin' : unit_interval,
            'atan' : real_line,
            'cos' : real_line,
            'cosh' : real_line,
            'exp' : real_line,
            'log' : positive,
            'log10' : positive,
            'sin' : real_line,
            'sinh' : real_line,
            'sqrt' : nonnegative,
            'tan' : real_line,
            'tanh' : real_line}

        for fn, values in test_functions.items():
            float_fn = getattr(math, fn)
            complex_fn = getattr(cmath, fn)
            for v in values:
                z = complex_fn(v)
                self.rAssertAlmostEqual(float_fn(v), z.real)
                self.assertEqual(0., z.imag)

        # test two-argument version of log with various bases
        for base in [0.5, 2., 10.]:
            for v in positive:
                z = cmath.log(v, base)
                self.rAssertAlmostEqual(math.log(v, base), z.real)
                self.assertEqual(0., z.imag)

    @requires_IEEE_754
    def test_specific_values(self):
        # [FUNCTION DOC] Table-driven test: parses the shared special-values file and for each
        # case calls the named cmath function (with wrappers for rect and polar), expecting
        # ValueError or OverflowError when flagged, otherwise comparing real and imaginary
        # parts within tolerance (looser absolute error for log and log10; sign-ignoring flags
        # honored). Skips a case on old OS X releases.
        # Some tests need to be skipped on ancient OS X versions.
        # See issue #27953.
        SKIP_ON_TIGER = {'tan0064'}

        osx_version = None
        if sys.platform == 'darwin':
            version_txt = platform.mac_ver()[0]
            try:
                osx_version = tuple(map(int, version_txt.split('.')))
            except ValueError:
                pass

        def rect_complex(z):
            # [FUNCTION DOC] HELPER: calls rect with the real and imaginary parts of z and
            # returns its complex result.
            """Wrapped version of rect that accepts a complex number instead of
            two float arguments."""
            return cmath.rect(z.real, z.imag)

        def polar_complex(z):
            # [FUNCTION DOC] HELPER: calls polar on z and returns the radius and angle packed
            # into a complex number.
            """Wrapped version of polar that returns a complex number instead of
            two floats."""
            return complex(*polar(z))

        for id, fn, ar, ai, er, ei, flags in parse_testfile(test_file):
            arg = complex(ar, ai)
            expected = complex(er, ei)

            # Skip certain tests on OS X 10.4.
            if osx_version is not None and osx_version < (10, 5):
                if id in SKIP_ON_TIGER:
                    continue

            if fn == 'rect':
                function = rect_complex
            elif fn == 'polar':
                function = polar_complex
            else:
                function = getattr(cmath, fn)
            if 'divide-by-zero' in flags or 'invalid' in flags:
                try:
                    actual = function(arg)
                except ValueError:
                    continue
                else:
                    self.fail('ValueError not raised in test '
                          '{}: {}(complex({!r}, {!r}))'.format(id, fn, ar, ai))

            if 'overflow' in flags:
                try:
                    actual = function(arg)
                except OverflowError:
                    continue
                else:
                    self.fail('OverflowError not raised in test '
                          '{}: {}(complex({!r}, {!r}))'.format(id, fn, ar, ai))

            actual = function(arg)

            if 'ignore-real-sign' in flags:
                actual = complex(abs(actual.real), actual.imag)
                expected = complex(abs(expected.real), expected.imag)
            if 'ignore-imag-sign' in flags:
                actual = complex(actual.real, abs(actual.imag))
                expected = complex(expected.real, abs(expected.imag))

            # for the real part of the log function, we allow an
            # absolute error of up to 2e-15.
            if fn in ('log', 'log10'):
                real_abs_err = 2e-15
            else:
                real_abs_err = 5e-323

            error_message = (
                '{}: {}(complex({!r}, {!r}))\n'
                'Expected: complex({!r}, {!r})\n'
                'Received: complex({!r}, {!r})\n'
                'Received value insufficiently close to expected value.'
                ).format(id, fn, ar, ai,
                     expected.real, expected.imag,
                     actual.real, actual.imag)
            self.rAssertAlmostEqual(expected.real, actual.real,
                                        abs_err=real_abs_err,
                                        msg=error_message)
            self.rAssertAlmostEqual(expected.imag, actual.imag,
                                        msg=error_message)

    def check_polar(self, func):
        # [FUNCTION DOC] HELPER: checks a polar-like callable on zeros, units, axes,
        # infinities and NaNs against expected (radius, angle) pairs using rAssertAlmostEqual.
        def check(arg, expected):
            # [FUNCTION DOC] HELPER: applies func to arg and compares each element of the
            # result with expected.
            got = func(arg)
            for e, g in zip(expected, got):
                self.rAssertAlmostEqual(e, g)
        check(0, (0., 0.))
        check(1, (1., 0.))
        check(-1, (1., pi))
        check(1j, (1., pi / 2))
        check(-3j, (3., -pi / 2))
        inf = float('inf')
        check(complex(inf, 0), (inf, 0.))
        check(complex(-inf, 0), (inf, pi))
        check(complex(3, inf), (inf, pi / 2))
        check(complex(5, -inf), (inf, -pi / 2))
        check(complex(inf, inf), (inf, pi / 4))
        check(complex(inf, -inf), (inf, -pi / 4))
        check(complex(-inf, inf), (inf, 3 * pi / 4))
        check(complex(-inf, -inf), (inf, -3 * pi / 4))
        nan = float('nan')
        check(complex(nan, 0), (nan, nan))
        check(complex(0, nan), (nan, nan))
        check(complex(nan, nan), (nan, nan))
        check(complex(inf, nan), (inf, nan))
        check(complex(-inf, nan), (inf, nan))
        check(complex(nan, inf), (inf, nan))
        check(complex(nan, -inf), (inf, nan))

    def test_polar(self):
        # [FUNCTION DOC] Runs the polar checks against cmath.polar.
        self.check_polar(polar)

    @cpython_only
    def test_polar_errno(self):
        # [FUNCTION DOC] CPython-only: sets the C errno to 11 around polar calls and checks
        # results are unchanged; errno is reset to zero afterwards. Requires _testcapi.
        # Issue #24489: check a previously set C errno doesn't disturb polar()
        _testcapi = import_helper.import_module('_testcapi')
        def polar_with_errno_set(z):
            # [FUNCTION DOC] HELPER: sets C errno to 11, calls polar on z and returns its
            # result, resetting errno to 0.
            _testcapi.set_errno(11)
            try:
                return polar(z)
            finally:
                _testcapi.set_errno(0)
        self.check_polar(polar_with_errno_set)

    def test_phase(self):
        # [FUNCTION DOC] Checks phase for ordinary values, signed zeros, infinities in all
        # directions, and that any NaN component yields NaN.
        self.assertAlmostEqual(phase(0), 0.)
        self.assertAlmostEqual(phase(1.), 0.)
        self.assertAlmostEqual(phase(-1.), pi)
        self.assertAlmostEqual(phase(-1.+1E-300j), pi)
        self.assertAlmostEqual(phase(-1.-1E-300j), -pi)
        self.assertAlmostEqual(phase(1j), pi/2)
        self.assertAlmostEqual(phase(-1j), -pi/2)

        # zeros
        self.assertEqual(phase(complex(0.0, 0.0)), 0.0)
        self.assertEqual(phase(complex(0.0, -0.0)), -0.0)
        self.assertEqual(phase(complex(-0.0, 0.0)), pi)
        self.assertEqual(phase(complex(-0.0, -0.0)), -pi)

        # infinities
        self.assertAlmostEqual(phase(complex(-INF, -0.0)), -pi)
        self.assertAlmostEqual(phase(complex(-INF, -2.3)), -pi)
        self.assertAlmostEqual(phase(complex(-INF, -INF)), -0.75*pi)
        self.assertAlmostEqual(phase(complex(-2.3, -INF)), -pi/2)
        self.assertAlmostEqual(phase(complex(-0.0, -INF)), -pi/2)
        self.assertAlmostEqual(phase(complex(0.0, -INF)), -pi/2)
        self.assertAlmostEqual(phase(complex(2.3, -INF)), -pi/2)
        self.assertAlmostEqual(phase(complex(INF, -INF)), -pi/4)
        self.assertEqual(phase(complex(INF, -2.3)), -0.0)
        self.assertEqual(phase(complex(INF, -0.0)), -0.0)
        self.assertEqual(phase(complex(INF, 0.0)), 0.0)
        self.assertEqual(phase(complex(INF, 2.3)), 0.0)
        self.assertAlmostEqual(phase(complex(INF, INF)), pi/4)
        self.assertAlmostEqual(phase(complex(2.3, INF)), pi/2)
        self.assertAlmostEqual(phase(complex(0.0, INF)), pi/2)
        self.assertAlmostEqual(phase(complex(-0.0, INF)), pi/2)
        self.assertAlmostEqual(phase(complex(-2.3, INF)), pi/2)
        self.assertAlmostEqual(phase(complex(-INF, INF)), 0.75*pi)
        self.assertAlmostEqual(phase(complex(-INF, 2.3)), pi)
        self.assertAlmostEqual(phase(complex(-INF, 0.0)), pi)

        # real or imaginary part NaN
        for z in complex_nans:
            self.assertTrue(math.isnan(phase(z)))

    def test_abs(self):
        # [FUNCTION DOC] Checks abs of complex zeros is 0, of infinities is infinity, and
        # mixed NaN/infinity cases.
        # zeros
        for z in complex_zeros:
            self.assertEqual(abs(z), 0.0)

        # infinities
        for z in complex_infinities:
            self.assertEqual(abs(z), INF)

        # real or imaginary part NaN
        self.assertEqual(abs(complex(NAN, -INF)), INF)
        self.assertTrue(math.isnan(abs(complex(NAN, -2.3))))
        self.assertTrue(math.isnan(abs(complex(NAN, -0.0))))
        self.assertTrue(math.isnan(abs(complex(NAN, 0.0))))
        self.assertTrue(math.isnan(abs(complex(NAN, 2.3))))
        self.assertEqual(abs(complex(NAN, INF)), INF)
        self.assertEqual(abs(complex(-INF, NAN)), INF)
        self.assertTrue(math.isnan(abs(complex(-2.3, NAN))))
        self.assertTrue(math.isnan(abs(complex(-0.0, NAN))))
        self.assertTrue(math.isnan(abs(complex(0.0, NAN))))
        self.assertTrue(math.isnan(abs(complex(2.3, NAN))))
        self.assertEqual(abs(complex(INF, NAN)), INF)
        self.assertTrue(math.isnan(abs(complex(NAN, NAN))))


    @requires_IEEE_754
    def test_abs_overflows(self):
        # [FUNCTION DOC] Checks abs raises OverflowError when the magnitude is too large
        # to represent.
        # result overflows
        self.assertRaises(OverflowError, abs, complex(1.4e308, 1.4e308))

    def assertCEqual(self, a, b):
        # [FUNCTION DOC] HELPER: fails unless complex a matches the (real, imag) pair b within
        # 1e-7 per component.
        eps = 1E-7
        if abs(a.real - b[0]) > eps or abs(a.imag - b[1]) > eps:
            self.fail((a ,b))

    def test_rect(self):
        # [FUNCTION DOC] Checks rect at several radii and angles against known (real,
        # imag) values.
        self.assertCEqual(rect(0, 0), (0, 0))
        self.assertCEqual(rect(1, 0), (1., 0))
        self.assertCEqual(rect(1, -pi), (-1., 0))
        self.assertCEqual(rect(1, pi/2), (0, 1.))
        self.assertCEqual(rect(1, -pi/2), (0, -1.))

    def test_isfinite(self):
        # [FUNCTION DOC] Checks isfinite over all combinations of special real and imaginary
        # parts matches the math isfinite of both parts.
        real_vals = [float('-inf'), -2.3, -0.0,
                     0.0, 2.3, float('inf'), float('nan')]
        for x in real_vals:
            for y in real_vals:
                z = complex(x, y)
                self.assertEqual(cmath.isfinite(z),
                                  math.isfinite(x) and math.isfinite(y))

    def test_isnan(self):
        # [FUNCTION DOC] Checks isnan is false for finite and infinite numbers and true if any
        # part is NaN.
        self.assertFalse(cmath.isnan(1))
        self.assertFalse(cmath.isnan(1j))
        self.assertFalse(cmath.isnan(INF))
        self.assertTrue(cmath.isnan(NAN))
        self.assertTrue(cmath.isnan(complex(NAN, 0)))
        self.assertTrue(cmath.isnan(complex(0, NAN)))
        self.assertTrue(cmath.isnan(complex(NAN, NAN)))
        self.assertTrue(cmath.isnan(complex(NAN, INF)))
        self.assertTrue(cmath.isnan(complex(INF, NAN)))

    def test_isinf(self):
        # [FUNCTION DOC] Checks isinf is false for finite and NaN-only numbers and true if any
        # part is infinite.
        self.assertFalse(cmath.isinf(1))
        self.assertFalse(cmath.isinf(1j))
        self.assertFalse(cmath.isinf(NAN))
        self.assertTrue(cmath.isinf(INF))
        self.assertTrue(cmath.isinf(complex(INF, 0)))
        self.assertTrue(cmath.isinf(complex(0, INF)))
        self.assertTrue(cmath.isinf(complex(INF, INF)))
        self.assertTrue(cmath.isinf(complex(NAN, INF)))
        self.assertTrue(cmath.isinf(complex(INF, NAN)))

    @requires_IEEE_754
    def testTanhSign(self):
        # [FUNCTION DOC] Checks tanh of each complex zero preserves the signs of the
        # zero parts.
        for z in complex_zeros:
            self.assertComplexesAreIdentical(cmath.tanh(z), z)

    # The algorithm used for atan and atanh makes use of the system
    # log1p function; If that system function doesn't respect the sign
    # of zero, then atan and atanh will also have difficulties with
    # the sign of complex zeros.
    @requires_IEEE_754
    def testAtanSign(self):
        # [FUNCTION DOC] Checks atan of each complex zero preserves signs; depends on a sign-
        # aware log1p.
        for z in complex_zeros:
            self.assertComplexesAreIdentical(cmath.atan(z), z)

    @requires_IEEE_754
    def testAtanhSign(self):
        # [FUNCTION DOC] Checks atanh of each complex zero preserves the signs of the
        # zero parts.
        for z in complex_zeros:
            self.assertComplexesAreIdentical(cmath.atanh(z), z)


class IsCloseTests(test_math.IsCloseTests):
    # [CLASS DOC] isclose tests inherited from test_math.IsCloseTests, rebound to
    # cmath.isclose and extended with complex-specific cases (tolerances, near-zero and
    # special values).
    isclose = cmath.isclose

    def test_reject_complex_tolerances(self):
        # [FUNCTION DOC] Checks complex-valued rel_tol or abs_tol arguments raise TypeError.
        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, rel_tol=1j)

        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, abs_tol=1j)

        with self.assertRaises(TypeError):
            self.isclose(1j, 1j, rel_tol=1j, abs_tol=1j)

    def test_complex_values(self):
        # [FUNCTION DOC] Checks complex values close to 12 decimal places pass at 1e-12
        # relative tolerance and fail at 1e-13.
        # test complex values that are close to within 12 decimal places
        complex_examples = [(1.0+1.0j, 1.000000000001+1.0j),
                            (1.0+1.0j, 1.0+1.000000000001j),
                            (-1.0+1.0j, -1.000000000001+1.0j),
                            (1.0-1.0j, 1.0-0.999999999999j),
                            ]

        self.assertAllClose(complex_examples, rel_tol=1e-12)
        self.assertAllNotClose(complex_examples, rel_tol=1e-13)

    def test_complex_near_zero(self):
        # [FUNCTION DOC] Checks values near zero are close under abs_tol of 1.5e-3 but not
        # 0.5e-3, plus two complex pairs against an absolute tolerance.
        # test values near zero that are near to within three decimal places
        near_zero_examples = [(0.001j, 0),
                              (0.001, 0),
                              (0.001+0.001j, 0),
                              (-0.001+0.001j, 0),
                              (0.001-0.001j, 0),
                              (-0.001-0.001j, 0),
                              ]

        self.assertAllClose(near_zero_examples, abs_tol=1.5e-03)
        self.assertAllNotClose(near_zero_examples, abs_tol=0.5e-03)

        self.assertIsClose(0.001-0.001j, 0.001+0.001j, abs_tol=2e-03)
        self.assertIsNotClose(0.001-0.001j, 0.001+0.001j, abs_tol=1e-03)

    def test_complex_special(self):
        # [FUNCTION DOC] Checks infinity versus imaginary infinity, opposite infinities, and
        # zero versus infinity are never close.
        self.assertIsNotClose(INF, INF*1j)
        self.assertIsNotClose(INF*1j, INF)
        self.assertIsNotClose(INF, -INF)
        self.assertIsNotClose(-INF, INF)
        self.assertIsNotClose(0, INF)
        self.assertIsNotClose(0, INF*1j)


if __name__ == "__main__":
    run_tests()
