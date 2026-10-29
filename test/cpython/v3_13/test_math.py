# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-patched copy of CPython 3.13.5 Lib/test/test_math.py, run through
# CPythonTestCase so the standard-library math module is exercised under
# torch.compile/Dynamo. Main contents: a meta-path finder that redirects 'test.*' imports to
# top-level modules, pure Python reference helpers (ulp conversion, factorial, tolerance
# checks, data-file parsers), small fixture classes, and three test classes: MathTests
# (functions and edge cases of math), IsCloseTests (math.isclose) and FMATests (math.fma).
# Import-time side effects: installs RedirectImportFinder at the front of sys.meta_path,
# computes the double-rounding flag, and derives paths of the mathdata test-case files next
# to this file. Running the file as a script calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_math.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    slowTest,
    run_tests,
    skipIfTorchDynamo,
)

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
    # [CLASS DOC] HELPER: Import hook that maps the names in redirect_imports (for example
    # test.test_math and test.test_iter) onto the same-named top-level module, so the copied
    # CPython tests can import their siblings from this directory. Stateless; installed once
    # at import time.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: Meta-path hook. If the full module name is in redirect_imports,
        # strips the leading 'test.' prefix, imports the bare module, registers it in
        # sys.modules under the original dotted name and returns the bare module's spec. Returns
        # None (let other finders run) when the name is not redirected or the bare import raises
        # ImportError. Mutates sys.modules.
        # KNOWN ISSUE: uses importlib.util without importing it explicitly; it only works
        # because another import already loaded importlib.util.
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

# Python test set -- math module
# XXXX Should not do tests around zero only

from test.support import verbose, requires_IEEE_754
from test import support
import unittest
import fractions
import itertools
import decimal
import math
import os
import platform
import random
import struct
import sys


eps = 1E-05
NAN = float('nan')
INF = float('inf')
NINF = float('-inf')
FLOAT_MAX = sys.float_info.max
FLOAT_MIN = sys.float_info.min

# detect evidence of double-rounding: fsum is not always correctly
# rounded on machines that suffer from double rounding.
x, y = 1e16, 2.9999 # use temporary values to defeat peephole optimizer
HAVE_DOUBLE_ROUNDING = (x + y == 1e16 + 4)

# locate file with test values
if __name__ == '__main__':
    file = sys.argv[0]
else:
    file = __file__
test_dir = os.path.dirname(file) or os.curdir
math_testcases = os.path.join(test_dir, 'mathdata', 'math_testcases.txt')
test_file = os.path.join(test_dir, 'mathdata', 'cmath_testcases.txt')


def to_ulps(x):
    # [FUNCTION DOC] HELPER: Maps a non-NaN float to an integer so that adjacent doubles map
    # to adjacent integers: packs the value as little-endian binary64, reads it back as a
    # signed 64-bit int and, for negative results, flips it to a negative ordering. Positive
    # and negative zero map to 0 and -1. Only meaningful for IEEE 754 binary64 doubles.
    # Returns an int.
    """Convert a non-NaN float x to an integer, in such a way that
    adjacent floats are converted to adjacent integers.  Then
    abs(ulps(x) - ulps(y)) gives the difference in ulps between two
    floats.

    The results from this function will only make sense on platforms
    where native doubles are represented in IEEE 754 binary64 format.

    Note: 0.0 and -0.0 are converted to 0 and -1, respectively.
    """
    n = struct.unpack('<q', struct.pack('<d', x))[0]
    if n < 0:
        n = ~(n+2**63)
    return n


# Here's a pure Python version of the math.factorial algorithm, for
# documentation and comparison purposes.
#
# Formula:
#
#   factorial(n) = factorial_odd_part(n) << (n - count_set_bits(n))
#
# where
#
#   factorial_odd_part(n) = product_{i >= 0} product_{0 < j <= n >> i; j odd} j
#
# The outer product above is an infinite product, but once i >= n.bit_length,
# (n >> i) < 1 and the corresponding term of the product is empty.  So only the
# finitely many terms for 0 <= i < n.bit_length() contribute anything.
#
# We iterate downwards from i == n.bit_length() - 1 to i == 0.  The inner
# product in the formula above starts at 1 for i == n.bit_length(); for each i
# < n.bit_length() we get the inner product for i from that for i + 1 by
# multiplying by all j in {n >> i+1 < j <= n >> i; j odd}.  In Python terms,
# this set is range((n >> i+1) + 1 | 1, (n >> i) + 1 | 1, 2).

def count_set_bits(n):
    # [FUNCTION DOC] HELPER: Recursively counts the 1 bits of a nonnegative integer by
    # clearing the lowest set bit each step. Returns an int; used by py_factorial for
    # the power-of-two exponent.
    """Number of '1' bits in binary expansion of a nonnnegative integer."""
    return 1 + count_set_bits(n & n - 1) if n else 0

def partial_product(start, stop):
    # [FUNCTION DOC] HELPER: Product of the odd integers from start up to (excluding) stop,
    # computed by recursive halving. Both bounds are expected odd with start <= stop; an empty
    # range gives 1, a single factor gives start. Returns an int.
    """Product of integers in range(start, stop, 2), computed recursively.
    start and stop should both be odd, with start <= stop.

    """
    numfactors = (stop - start) >> 1
    if not numfactors:
        return 1
    elif numfactors == 1:
        return start
    else:
        mid = (start + numfactors) | 1
        return partial_product(start, mid) * partial_product(mid, stop)

def py_factorial(n):
    # [FUNCTION DOC] HELPER: Pure Python reference factorial using the binary split formula:
    # iterates the bit positions of n from high to low, accumulating partial products of odd
    # numbers, and shifts the result left by n minus its popcount. Expects a nonnegative int;
    # returns the factorial as an int. Used to cross-check math.factorial.
    """Factorial of nonnegative integer n, via "Binary Split Factorial Formula"
    described at http://www.luschny.de/math/factorial/binarysplitfact.html

    """
    inner = outer = 1
    for i in reversed(range(n.bit_length())):
        inner *= partial_product((n >> i + 1) + 1 | 1, (n >> i) + 1 | 1)
        outer *= inner
    return outer << (n - count_set_bits(n))

def ulp_abs_check(expected, got, ulp_tol, abs_tol):
    # [FUNCTION DOC] HELPER: Compares two finite floats. Succeeds when their distance
    # in ulps (via to_ulps) is within ulp_tol OR their absolute difference is within
    # abs_tol. Returns None on success, otherwise a formatted message with the observed
    # error and permitted tolerances.
    """Given finite floats `expected` and `got`, check that they're
    approximately equal to within the given number of ulps or the
    given absolute tolerance, whichever is bigger.

    Returns None on success and an error message on failure.
    """
    ulp_error = abs(to_ulps(expected) - to_ulps(got))
    abs_error = abs(expected - got)

    # Succeed if either abs_error <= abs_tol or ulp_error <= ulp_tol.
    if abs_error <= abs_tol or ulp_error <= ulp_tol:
        return None
    else:
        fmt = ("error = {:.3g} ({:d} ulps); "
               "permitted error = {:.3g} or {:d} ulps")
        return fmt.format(abs_error, ulp_error, abs_tol, ulp_tol)

def parse_mtestfile(fname):
    # [FUNCTION DOC] HELPER: Generator that parses the math_testcases data file. Strips
    # text after a double dash comment marker, skips blank lines, and splits each line on
    # the arrow into the left side (id, function name, argument) and right side (expected
    # value followed by optional flags). Yields tuples of id, function name, float
    # argument, float expected value and flag list. Opens the file as UTF-8; raises on
    # malformed lines or non-numeric fields.
    """Parse a file with test values

    -- starts a comment
    blank lines, or lines containing only a comment, are ignored
    other lines are expected to have the form
      id fn arg -> expected [flag]*

    """
    with open(fname, encoding="utf-8") as fp:
        for line in fp:
            # strip comments, and skip blank lines
            if '--' in line:
                line = line[:line.index('--')]
            if not line.strip():
                continue

            lhs, rhs = line.split('->')
            id, fn, arg = lhs.split()
            rhs_pieces = rhs.split()
            exp = rhs_pieces[0]
            flags = rhs_pieces[1:]

            yield (id, fn, float(arg), float(exp), flags)


def parse_testfile(fname):
    # [FUNCTION DOC] HELPER: Generator that parses the cmath_testcases data file. Skips lines
    # starting with a double dash and blank lines, splits on the arrow and yields id, function
    # name, real and imaginary argument parts, real and imaginary expected parts (all floats)
    # and a flag list. Opens the file as UTF-8; raises on malformed lines.
    """Parse a file with test values

    Empty lines or lines starting with -- are ignored
    yields id, fn, arg_real, arg_imag, exp_real, exp_imag
    """
    with open(fname, encoding="utf-8") as fp:
        for line in fp:
            # skip comment lines and blank lines
            if line.startswith('--') or not line.strip():
                continue

            lhs, rhs = line.split('->')
            id, fn, arg_real, arg_imag = lhs.split()
            rhs_pieces = rhs.split()
            exp_real, exp_imag = rhs_pieces[0], rhs_pieces[1]
            flags = rhs_pieces[2:]

            yield (id, fn,
                   float(arg_real), float(arg_imag),
                   float(exp_real), float(exp_imag),
                   flags)


def result_check(expected, got, ulp_tol=5, abs_tol=0.0):
    # [FUNCTION DOC] HELPER: Compares an expected and a received value for the table-driven
    # tests. Equal values pass, except that two zeros with different sign bits fail. Mixed
    # int/float pairs are converted to float; two NaNs count as equal; any infinity mismatch
    # fails; two finite floats are compared through ulp_abs_check with the given tolerances.
    # Non-numeric values (such as exception-name strings) must be exactly equal. Returns None on
    # success or a message string of the form expected/got plus the reason.
    # Common logic of MathTests.(ftest, test_testcases, test_mtestcases)
    """Compare arguments expected and got, as floats, if either
    is a float, using a tolerance expressed in multiples of
    ulp(expected) or absolutely (if given and greater).

    As a convenience, when neither argument is a float, and for
    non-finite floats, exact equality is demanded. Also, nan==nan
    as far as this function is concerned.

    Returns None on success and an error message on failure.
    """

    # Check exactly equal (applies also to strings representing exceptions)
    if got == expected:
        if not got and not expected:
            if math.copysign(1, got) != math.copysign(1, expected):
                return f"expected {expected}, got {got} (zero has wrong sign)"
        return None

    failure = "not equal"

    # Turn mixed float and int comparison (e.g. floor()) to all-float
    if isinstance(expected, float) and isinstance(got, int):
        got = float(got)
    elif isinstance(got, float) and isinstance(expected, int):
        expected = float(expected)

    if isinstance(expected, float) and isinstance(got, float):
        if math.isnan(expected) and math.isnan(got):
            # Pass, since both nan
            failure = None
        elif math.isinf(expected) or math.isinf(got):
            # We already know they're not equal, drop through to failure
            pass
        else:
            # Both are finite floats (now). Are they close enough?
            failure = ulp_abs_check(expected, got, ulp_tol, abs_tol)

    # arguments are not equal, and if numeric, are too far apart
    if failure is not None:
        fail_fmt = "expected {!r}, got {!r}"
        fail_msg = fail_fmt.format(expected, got)
        fail_msg += ' ({})'.format(failure)
        return fail_msg
    else:
        return None

class FloatLike:
    # [CLASS DOC] HELPER: Fixture whose instances are convertible to float only through the
    # float conversion protocol. Stores the given value and returns it unchanged from the
    # conversion method; it is not an int or float subclass, so it exercises code paths that
    # call the protocol.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value to be returned by the float conversion.
        self.value = value

    def __float__(self):
        # [FUNCTION DOC] HELPER: Float conversion protocol; returns the stored value as is (no
        # coercion, so it must already be a float for the caller to accept it).
        return self.value

class IntSubclass(int):
    # [CLASS DOC] HELPER: Plain int subclass used to check that math.perm and math.comb accept
    # and return exact ints for int subclasses.
    pass

# Class providing an __index__ method.
class MyIndexable(object):
    # [CLASS DOC] HELPER: Fixture exposing only the integer index protocol (not an int
    # subclass); wraps an int and returns it from the index method. Used for gcd, lcm,
    # perm and comb.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the integer to be returned by the index protocol.
        self.value = value

    def __index__(self):
        # [FUNCTION DOC] HELPER: Index protocol; returns the stored value.
        return self.value

class BadDescr:
    # [CLASS DOC] HELPER: Descriptor whose get method always raises ValueError. Assigned to
    # special method names (ceil, floor, trunc, float conversion) to check that math functions
    # propagate errors raised while looking up those methods.
    def __get__(self, obj, objtype=None):
        # [FUNCTION DOC] HELPER: Descriptor get; unconditionally raises ValueError
        # regardless of arguments.
        raise ValueError

class MathTests(__TestCase):
    # [CLASS DOC] Main test case (CPythonTestCase subclass) covering the functions and
    # constants of the math module: trig/hyperbolic, rounding, logs, exponentials, gcd/lcm,
    # comb/perm, fsum, hypot, dist, sumprod, prod, nextafter, ulp, remainder and special-
    # value (inf, nan, signed zero) behavior, plus table-driven tests from the mathdata
    # files. Has no instance state; ftest and the custom assertions are shared helpers. Some
    # tests are skipped under Dynamo.

    def ftest(self, name, got, expected, ulp_tol=5, abs_tol=0.0):
        # [FUNCTION DOC] HELPER: Assertion helper. Calls result_check with the expected and got
        # values and the ulp/absolute tolerances (defaults 5 ulps, 0.0); fails the test with the
        # given name and the message when they differ. Returns nothing.
        """Compare arguments expected and got, as floats, if either
        is a float, using a tolerance expressed in multiples of
        ulp(expected) or absolutely, whichever is greater.

        As a convenience, when neither argument is a float, and for
        non-finite floats, exact equality is demanded. Also, nan==nan
        in this function.
        """
        failure = result_check(expected, got, ulp_tol, abs_tol)
        if failure is not None:
            self.fail("{}: {}".format(name, failure))

    def testConstants(self):
        # [FUNCTION DOC] Checks math.pi and math.e against high-precision literals with ftest
        # and that math.tau equals twice pi exactly.
        # Ref: Abramowitz & Stegun (Dover, 1965)
        self.ftest('pi', math.pi, 3.141592653589793238462643)
        self.ftest('e', math.e, 2.718281828459045235360287)
        self.assertEqual(math.tau, 2*math.pi)

    def testAcos(self):
        # [FUNCTION DOC] Tests math.acos: wrong arity raises TypeError; values at -1, 0 and 1;
        # inputs outside [-1, 1] (including infinities) raise ValueError; NaN input gives NaN.
        self.assertRaises(TypeError, math.acos)
        self.ftest('acos(-1)', math.acos(-1), math.pi)
        self.ftest('acos(0)', math.acos(0), math.pi/2)
        self.ftest('acos(1)', math.acos(1), 0)
        self.assertRaises(ValueError, math.acos, INF)
        self.assertRaises(ValueError, math.acos, NINF)
        self.assertRaises(ValueError, math.acos, 1 + eps)
        self.assertRaises(ValueError, math.acos, -1 - eps)
        self.assertTrue(math.isnan(math.acos(NAN)))

    def testAcosh(self):
        # [FUNCTION DOC] Tests math.acosh: TypeError on missing argument, known values at 1 and
        # 2, ValueError for inputs below 1 and negative infinity, positive infinity maps to
        # itself, NaN propagates.
        self.assertRaises(TypeError, math.acosh)
        self.ftest('acosh(1)', math.acosh(1), 0)
        self.ftest('acosh(2)', math.acosh(2), 1.3169578969248168)
        self.assertRaises(ValueError, math.acosh, 0)
        self.assertRaises(ValueError, math.acosh, -1)
        self.assertEqual(math.acosh(INF), INF)
        self.assertRaises(ValueError, math.acosh, NINF)
        self.assertTrue(math.isnan(math.acosh(NAN)))

    def testAsin(self):
        # [FUNCTION DOC] Tests math.asin: TypeError on missing argument, values at -1, 0, 1,
        # ValueError for arguments outside [-1, 1] or infinite, NaN propagates.
        self.assertRaises(TypeError, math.asin)
        self.ftest('asin(-1)', math.asin(-1), -math.pi/2)
        self.ftest('asin(0)', math.asin(0), 0)
        self.ftest('asin(1)', math.asin(1), math.pi/2)
        self.assertRaises(ValueError, math.asin, INF)
        self.assertRaises(ValueError, math.asin, NINF)
        self.assertRaises(ValueError, math.asin, 1 + eps)
        self.assertRaises(ValueError, math.asin, -1 - eps)
        self.assertTrue(math.isnan(math.asin(NAN)))

    def testAsinh(self):
        # [FUNCTION DOC] Tests math.asinh: TypeError on missing argument, odd symmetry at plus
        # and minus 1, zero, infinities map to themselves, NaN propagates.
        self.assertRaises(TypeError, math.asinh)
        self.ftest('asinh(0)', math.asinh(0), 0)
        self.ftest('asinh(1)', math.asinh(1), 0.88137358701954305)
        self.ftest('asinh(-1)', math.asinh(-1), -0.88137358701954305)
        self.assertEqual(math.asinh(INF), INF)
        self.assertEqual(math.asinh(NINF), NINF)
        self.assertTrue(math.isnan(math.asinh(NAN)))

    def testAtan(self):
        # [FUNCTION DOC] Tests math.atan: TypeError on missing argument, values at -1, 0, 1,
        # limits of plus and minus infinity (pi/2), NaN propagates.
        self.assertRaises(TypeError, math.atan)
        self.ftest('atan(-1)', math.atan(-1), -math.pi/4)
        self.ftest('atan(0)', math.atan(0), 0)
        self.ftest('atan(1)', math.atan(1), math.pi/4)
        self.ftest('atan(inf)', math.atan(INF), math.pi/2)
        self.ftest('atan(-inf)', math.atan(NINF), -math.pi/2)
        self.assertTrue(math.isnan(math.atan(NAN)))

    def testAtanh(self):
        # [FUNCTION DOC] Tests math.atanh: known values at 0 and plus/minus 0.5; ValueError at
        # plus/minus 1 and infinities; NaN propagates.
        # KNOWN ISSUE: the arity check calls math.atan instead of math.atanh (copied upstream
        # typo), so atanh's own TypeError behavior is not tested.
        self.assertRaises(TypeError, math.atan)
        self.ftest('atanh(0)', math.atanh(0), 0)
        self.ftest('atanh(0.5)', math.atanh(0.5), 0.54930614433405489)
        self.ftest('atanh(-0.5)', math.atanh(-0.5), -0.54930614433405489)
        self.assertRaises(ValueError, math.atanh, 1)
        self.assertRaises(ValueError, math.atanh, -1)
        self.assertRaises(ValueError, math.atanh, INF)
        self.assertRaises(ValueError, math.atanh, NINF)
        self.assertTrue(math.isnan(math.atanh(NAN)))

    def testAtan2(self):
        # [FUNCTION DOC] Tests math.atan2 over the full IEEE 754 special-value table: argument
        # arity errors, quadrant values for simple inputs, and each combination of y in (+0, -0,
        # +inf, -inf, positive finite, negative finite, NaN) with x in (-inf, finite, +/-0,
        # +inf, NaN), including the sign of zero results and NaN propagation.
        self.assertRaises(TypeError, math.atan2)
        self.ftest('atan2(-1, 0)', math.atan2(-1, 0), -math.pi/2)
        self.ftest('atan2(-1, 1)', math.atan2(-1, 1), -math.pi/4)
        self.ftest('atan2(0, 1)', math.atan2(0, 1), 0)
        self.ftest('atan2(1, 1)', math.atan2(1, 1), math.pi/4)
        self.ftest('atan2(1, 0)', math.atan2(1, 0), math.pi/2)
        self.ftest('atan2(1, -1)', math.atan2(1, -1), 3*math.pi/4)

        # math.atan2(0, x)
        self.ftest('atan2(0., -inf)', math.atan2(0., NINF), math.pi)
        self.ftest('atan2(0., -2.3)', math.atan2(0., -2.3), math.pi)
        self.ftest('atan2(0., -0.)', math.atan2(0., -0.), math.pi)
        self.assertEqual(math.atan2(0., 0.), 0.)
        self.assertEqual(math.atan2(0., 2.3), 0.)
        self.assertEqual(math.atan2(0., INF), 0.)
        self.assertTrue(math.isnan(math.atan2(0., NAN)))
        # math.atan2(-0, x)
        self.ftest('atan2(-0., -inf)', math.atan2(-0., NINF), -math.pi)
        self.ftest('atan2(-0., -2.3)', math.atan2(-0., -2.3), -math.pi)
        self.ftest('atan2(-0., -0.)', math.atan2(-0., -0.), -math.pi)
        self.assertEqual(math.atan2(-0., 0.), -0.)
        self.assertEqual(math.atan2(-0., 2.3), -0.)
        self.assertEqual(math.atan2(-0., INF), -0.)
        self.assertTrue(math.isnan(math.atan2(-0., NAN)))
        # math.atan2(INF, x)
        self.ftest('atan2(inf, -inf)', math.atan2(INF, NINF), math.pi*3/4)
        self.ftest('atan2(inf, -2.3)', math.atan2(INF, -2.3), math.pi/2)
        self.ftest('atan2(inf, -0.)', math.atan2(INF, -0.0), math.pi/2)
        self.ftest('atan2(inf, 0.)', math.atan2(INF, 0.0), math.pi/2)
        self.ftest('atan2(inf, 2.3)', math.atan2(INF, 2.3), math.pi/2)
        self.ftest('atan2(inf, inf)', math.atan2(INF, INF), math.pi/4)
        self.assertTrue(math.isnan(math.atan2(INF, NAN)))
        # math.atan2(NINF, x)
        self.ftest('atan2(-inf, -inf)', math.atan2(NINF, NINF), -math.pi*3/4)
        self.ftest('atan2(-inf, -2.3)', math.atan2(NINF, -2.3), -math.pi/2)
        self.ftest('atan2(-inf, -0.)', math.atan2(NINF, -0.0), -math.pi/2)
        self.ftest('atan2(-inf, 0.)', math.atan2(NINF, 0.0), -math.pi/2)
        self.ftest('atan2(-inf, 2.3)', math.atan2(NINF, 2.3), -math.pi/2)
        self.ftest('atan2(-inf, inf)', math.atan2(NINF, INF), -math.pi/4)
        self.assertTrue(math.isnan(math.atan2(NINF, NAN)))
        # math.atan2(+finite, x)
        self.ftest('atan2(2.3, -inf)', math.atan2(2.3, NINF), math.pi)
        self.ftest('atan2(2.3, -0.)', math.atan2(2.3, -0.), math.pi/2)
        self.ftest('atan2(2.3, 0.)', math.atan2(2.3, 0.), math.pi/2)
        self.assertEqual(math.atan2(2.3, INF), 0.)
        self.assertTrue(math.isnan(math.atan2(2.3, NAN)))
        # math.atan2(-finite, x)
        self.ftest('atan2(-2.3, -inf)', math.atan2(-2.3, NINF), -math.pi)
        self.ftest('atan2(-2.3, -0.)', math.atan2(-2.3, -0.), -math.pi/2)
        self.ftest('atan2(-2.3, 0.)', math.atan2(-2.3, 0.), -math.pi/2)
        self.assertEqual(math.atan2(-2.3, INF), -0.)
        self.assertTrue(math.isnan(math.atan2(-2.3, NAN)))
        # math.atan2(NAN, x)
        self.assertTrue(math.isnan(math.atan2(NAN, NINF)))
        self.assertTrue(math.isnan(math.atan2(NAN, -2.3)))
        self.assertTrue(math.isnan(math.atan2(NAN, -0.)))
        self.assertTrue(math.isnan(math.atan2(NAN, 0.)))
        self.assertTrue(math.isnan(math.atan2(NAN, 2.3)))
        self.assertTrue(math.isnan(math.atan2(NAN, INF)))
        self.assertTrue(math.isnan(math.atan2(NAN, NAN)))

    def testCbrt(self):
        # [FUNCTION DOC] Tests math.cbrt: TypeError on missing argument, exact cube roots, sign
        # preservation including negative zero, non-trivial values, infinities map to
        # themselves, NaN propagates.
        self.assertRaises(TypeError, math.cbrt)
        self.ftest('cbrt(0)', math.cbrt(0), 0)
        self.ftest('cbrt(1)', math.cbrt(1), 1)
        self.ftest('cbrt(8)', math.cbrt(8), 2)
        self.ftest('cbrt(0.0)', math.cbrt(0.0), 0.0)
        self.ftest('cbrt(-0.0)', math.cbrt(-0.0), -0.0)
        self.ftest('cbrt(1.2)', math.cbrt(1.2), 1.062658569182611)
        self.ftest('cbrt(-2.6)', math.cbrt(-2.6), -1.375068867074141)
        self.ftest('cbrt(27)', math.cbrt(27), 3)
        self.ftest('cbrt(-1)', math.cbrt(-1), -1)
        self.ftest('cbrt(-27)', math.cbrt(-27), -3)
        self.assertEqual(math.cbrt(INF), INF)
        self.assertEqual(math.cbrt(NINF), NINF)
        self.assertTrue(math.isnan(math.cbrt(NAN)))

    def testCeil(self):
        # [FUNCTION DOC] Tests math.ceil: the result type is int and values round toward
        # positive infinity (negative values toward zero). Also checks the __ceil__ protocol
        # with local fixtures: a class defining it, a float subclass defining it, an object
        # only convertible to float (uses real ceiling), a class without it (TypeError), a
        # class whose lookup raises ValueError (propagated), and that an instance attribute
        # is not used (TypeError).
        self.assertRaises(TypeError, math.ceil)
        self.assertEqual(int, type(math.ceil(0.5)))
        self.assertEqual(math.ceil(0.5), 1)
        self.assertEqual(math.ceil(1.0), 1)
        self.assertEqual(math.ceil(1.5), 2)
        self.assertEqual(math.ceil(-0.5), 0)
        self.assertEqual(math.ceil(-1.0), -1)
        self.assertEqual(math.ceil(-1.5), -1)
        self.assertEqual(math.ceil(0.0), 0)
        self.assertEqual(math.ceil(-0.0), 0)
        #self.assertEqual(math.ceil(INF), INF)
        #self.assertEqual(math.ceil(NINF), NINF)
        #self.assertTrue(math.isnan(math.ceil(NAN)))

        class TestCeil:
            # [CLASS DOC] HELPER: Local fixture defining the ceil special method, returning 42.
            def __ceil__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 42.
                return 42
        class FloatCeil(float):
            # [CLASS DOC] HELPER: Local float subclass whose ceil special method returns 42 (the
            # override must win).
            def __ceil__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 42.
                return 42
        class TestNoCeil:
            # [CLASS DOC] HELPER: Local fixture with no ceil support; math.ceil
            # must raise TypeError.
            pass
        class TestBadCeil:
            # [CLASS DOC] HELPER: Local fixture whose ceil attribute is a descriptor
            # that raises ValueError.
            __ceil__ = BadDescr()
        self.assertEqual(math.ceil(TestCeil()), 42)
        self.assertEqual(math.ceil(FloatCeil()), 42)
        self.assertEqual(math.ceil(FloatLike(42.5)), 43)
        self.assertRaises(TypeError, math.ceil, TestNoCeil())
        self.assertRaises(ValueError, math.ceil, TestBadCeil())

        t = TestNoCeil()
        t.__ceil__ = lambda *args: args
        self.assertRaises(TypeError, math.ceil, t)
        self.assertRaises(TypeError, math.ceil, t, 0)

        self.assertEqual(math.ceil(FloatLike(+1.0)), +1.0)
        self.assertEqual(math.ceil(FloatLike(-1.0)), -1.0)

    @requires_IEEE_754
    def testCopysign(self):
        # [FUNCTION DOC] Tests math.copysign (IEEE 754 only): values with various signs, the
        # ability to distinguish signed zeros and infinities, NaN as first argument stays
        # NaN, and NaN as the sign source (only the magnitude is asserted since the NaN sign
        # bit is platform dependent).
        self.assertEqual(math.copysign(1, 42), 1.0)
        self.assertEqual(math.copysign(0., 42), 0.0)
        self.assertEqual(math.copysign(1., -42), -1.0)
        self.assertEqual(math.copysign(3, 0.), 3.0)
        self.assertEqual(math.copysign(4., -0.), -4.0)

        self.assertRaises(TypeError, math.copysign)
        # copysign should let us distinguish signs of zeros
        self.assertEqual(math.copysign(1., 0.), 1.)
        self.assertEqual(math.copysign(1., -0.), -1.)
        self.assertEqual(math.copysign(INF, 0.), INF)
        self.assertEqual(math.copysign(INF, -0.), NINF)
        self.assertEqual(math.copysign(NINF, 0.), INF)
        self.assertEqual(math.copysign(NINF, -0.), NINF)
        # and of infinities
        self.assertEqual(math.copysign(1., INF), 1.)
        self.assertEqual(math.copysign(1., NINF), -1.)
        self.assertEqual(math.copysign(INF, INF), INF)
        self.assertEqual(math.copysign(INF, NINF), NINF)
        self.assertEqual(math.copysign(NINF, INF), INF)
        self.assertEqual(math.copysign(NINF, NINF), NINF)
        self.assertTrue(math.isnan(math.copysign(NAN, 1.)))
        self.assertTrue(math.isnan(math.copysign(NAN, INF)))
        self.assertTrue(math.isnan(math.copysign(NAN, NINF)))
        self.assertTrue(math.isnan(math.copysign(NAN, NAN)))
        # copysign(INF, NAN) may be INF or it may be NINF, since
        # we don't know whether the sign bit of NAN is set on any
        # given platform.
        self.assertTrue(math.isinf(math.copysign(INF, NAN)))
        # similarly, copysign(2., NAN) could be 2. or -2.
        self.assertEqual(abs(math.copysign(2., NAN)), 2.)

    def testCos(self):
        # [FUNCTION DOC] Tests math.cos: known values with an absolute tolerance of one
        # ulp near zero. Infinite input must give NaN or raise ValueError (either
        # accepted); NaN propagates.
        self.assertRaises(TypeError, math.cos)
        self.ftest('cos(-pi/2)', math.cos(-math.pi/2), 0, abs_tol=math.ulp(1))
        self.ftest('cos(0)', math.cos(0), 1)
        self.ftest('cos(pi/2)', math.cos(math.pi/2), 0, abs_tol=math.ulp(1))
        self.ftest('cos(pi)', math.cos(math.pi), -1)
        try:
            self.assertTrue(math.isnan(math.cos(INF)))
            self.assertTrue(math.isnan(math.cos(NINF)))
        except ValueError:
            self.assertRaises(ValueError, math.cos, INF)
            self.assertRaises(ValueError, math.cos, NINF)
        self.assertTrue(math.isnan(math.cos(NAN)))

    @unittest.skipIf(sys.platform == 'win32' and platform.machine() in ('ARM', 'ARM64'),
                    "Windows UCRT is off by 2 ULP this test requires accuracy within 1 ULP")
    def testCosh(self):
        # [FUNCTION DOC] Tests math.cosh: values at 0 and a Lambert identity check, both
        # infinities give positive infinity, NaN propagates. Skipped on Windows ARM where the C
        # runtime is off by two ulps.
        self.assertRaises(TypeError, math.cosh)
        self.ftest('cosh(0)', math.cosh(0), 1)
        self.ftest('cosh(2)-2*cosh(1)**2', math.cosh(2)-2*math.cosh(1)**2, -1) # Thanks to Lambert
        self.assertEqual(math.cosh(INF), INF)
        self.assertEqual(math.cosh(NINF), INF)
        self.assertTrue(math.isnan(math.cosh(NAN)))

    def testDegrees(self):
        # [FUNCTION DOC] Tests math.degrees at pi, pi/2, -pi/4 and 0, and TypeError
        # without an argument.
        self.assertRaises(TypeError, math.degrees)
        self.ftest('degrees(pi)', math.degrees(math.pi), 180.0)
        self.ftest('degrees(pi/2)', math.degrees(math.pi/2), 90.0)
        self.ftest('degrees(-pi/4)', math.degrees(-math.pi/4), -45.0)
        self.ftest('degrees(0)', math.degrees(0), 0)

    def testExp(self):
        # [FUNCTION DOC] Tests math.exp: known values, +inf gives inf, -inf gives 0, NaN
        # propagates, and a huge finite argument raises OverflowError.
        self.assertRaises(TypeError, math.exp)
        self.ftest('exp(-1)', math.exp(-1), 1/math.e)
        self.ftest('exp(0)', math.exp(0), 1)
        self.ftest('exp(1)', math.exp(1), math.e)
        self.assertEqual(math.exp(INF), INF)
        self.assertEqual(math.exp(NINF), 0.)
        self.assertTrue(math.isnan(math.exp(NAN)))
        self.assertRaises(OverflowError, math.exp, 1000000)

    def testExp2(self):
        # [FUNCTION DOC] Tests math.exp2 analogously to exp: known values, infinities, NaN and
        # OverflowError for a huge argument.
        self.assertRaises(TypeError, math.exp2)
        self.ftest('exp2(-1)', math.exp2(-1), 0.5)
        self.ftest('exp2(0)', math.exp2(0), 1)
        self.ftest('exp2(1)', math.exp2(1), 2)
        self.ftest('exp2(2.3)', math.exp2(2.3), 4.924577653379665)
        self.assertEqual(math.exp2(INF), INF)
        self.assertEqual(math.exp2(NINF), 0.)
        self.assertTrue(math.isnan(math.exp2(NAN)))
        self.assertRaises(OverflowError, math.exp2, 1000000)

    def testFabs(self):
        # [FUNCTION DOC] Tests math.fabs for negative, zero and positive ints and
        # TypeError without arguments.
        self.assertRaises(TypeError, math.fabs)
        self.ftest('fabs(-1)', math.fabs(-1), 1)
        self.ftest('fabs(0)', math.fabs(0), 0)
        self.ftest('fabs(1)', math.fabs(1), 1)

    @skipIfTorchDynamo("infinite loop")
    def testFactorial(self):
        # [FUNCTION DOC] Tests math.factorial for 0 and for 1..999, comparing against a running
        # product and the pure Python py_factorial; negative inputs raise ValueError. Skipped
        # under Dynamo due to an infinite loop.
        self.assertEqual(math.factorial(0), 1)
        total = 1
        for i in range(1, 1000):
            total *= i
            self.assertEqual(math.factorial(i), total)
            self.assertEqual(math.factorial(i), py_factorial(i))
        self.assertRaises(ValueError, math.factorial, -1)
        self.assertRaises(ValueError, math.factorial, -10**100)

    def testFactorialNonIntegers(self):
        # [FUNCTION DOC] Tests that math.factorial raises TypeError for floats (integral or
        # not), Decimal and strings.
        self.assertRaises(TypeError, math.factorial, 5.0)
        self.assertRaises(TypeError, math.factorial, 5.2)
        self.assertRaises(TypeError, math.factorial, -1.0)
        self.assertRaises(TypeError, math.factorial, -1e100)
        self.assertRaises(TypeError, math.factorial, decimal.Decimal('5'))
        self.assertRaises(TypeError, math.factorial, decimal.Decimal('5.2'))
        self.assertRaises(TypeError, math.factorial, "5")

    # Other implementations may place different upper bounds.
    @support.cpython_only
    def testFactorialHugeInputs(self):
        # [FUNCTION DOC] CPython-only test: factorial of an int too large for a C long raises
        # OverflowError, and a huge float still raises TypeError.
        # Currently raises OverflowError for inputs that are too large
        # to fit into a C long.
        self.assertRaises(OverflowError, math.factorial, 10**100)
        self.assertRaises(TypeError, math.factorial, 1e100)

    def testFloor(self):
        # [FUNCTION DOC] Tests math.floor: result type is int, rounding toward negative
        # infinity. Also checks the floor special-method protocol with local fixtures exactly
        # as testCeil does for ceil (custom method used, float-convertible object uses real
        # floor, missing method gives TypeError, raising descriptor propagates ValueError,
        # instance attribute ignored).
        self.assertRaises(TypeError, math.floor)
        self.assertEqual(int, type(math.floor(0.5)))
        self.assertEqual(math.floor(0.5), 0)
        self.assertEqual(math.floor(1.0), 1)
        self.assertEqual(math.floor(1.5), 1)
        self.assertEqual(math.floor(-0.5), -1)
        self.assertEqual(math.floor(-1.0), -1)
        self.assertEqual(math.floor(-1.5), -2)
        #self.assertEqual(math.ceil(INF), INF)
        #self.assertEqual(math.ceil(NINF), NINF)
        #self.assertTrue(math.isnan(math.floor(NAN)))

        class TestFloor:
            # [CLASS DOC] HELPER: Local fixture defining the floor special method, returning 42.
            def __floor__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 42.
                return 42
        class FloatFloor(float):
            # [CLASS DOC] HELPER: Local float subclass whose floor special method returns 42.
            def __floor__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 42.
                return 42
        class TestNoFloor:
            # [CLASS DOC] HELPER: Local fixture with no floor support; math.floor
            # must raise TypeError.
            pass
        class TestBadFloor:
            # [CLASS DOC] HELPER: Local fixture whose floor attribute is a descriptor
            # that raises ValueError.
            __floor__ = BadDescr()
        self.assertEqual(math.floor(TestFloor()), 42)
        self.assertEqual(math.floor(FloatFloor()), 42)
        self.assertEqual(math.floor(FloatLike(41.9)), 41)
        self.assertRaises(TypeError, math.floor, TestNoFloor())
        self.assertRaises(ValueError, math.floor, TestBadFloor())

        t = TestNoFloor()
        t.__floor__ = lambda *args: args
        self.assertRaises(TypeError, math.floor, t)
        self.assertRaises(TypeError, math.floor, t, 0)

        self.assertEqual(math.floor(FloatLike(+1.0)), +1.0)
        self.assertEqual(math.floor(FloatLike(-1.0)), -1.0)

    def testFmod(self):
        # [FUNCTION DOC] Tests math.fmod: sign follows the dividend (including negative zero
        # results), NaN operands propagate, ValueError for zero divisor or infinite dividend,
        # and a finite dividend with an infinite divisor is returned unchanged.
        self.assertRaises(TypeError, math.fmod)
        self.ftest('fmod(10, 1)', math.fmod(10, 1), 0.0)
        self.ftest('fmod(10, 0.5)', math.fmod(10, 0.5), 0.0)
        self.ftest('fmod(10, 1.5)', math.fmod(10, 1.5), 1.0)
        self.ftest('fmod(-10, 1)', math.fmod(-10, 1), -0.0)
        self.ftest('fmod(-10, 0.5)', math.fmod(-10, 0.5), -0.0)
        self.ftest('fmod(-10, 1.5)', math.fmod(-10, 1.5), -1.0)
        self.assertTrue(math.isnan(math.fmod(NAN, 1.)))
        self.assertTrue(math.isnan(math.fmod(1., NAN)))
        self.assertTrue(math.isnan(math.fmod(NAN, NAN)))
        self.assertRaises(ValueError, math.fmod, 1., 0.)
        self.assertRaises(ValueError, math.fmod, INF, 1.)
        self.assertRaises(ValueError, math.fmod, NINF, 1.)
        self.assertRaises(ValueError, math.fmod, INF, 0.)
        self.assertEqual(math.fmod(3.0, INF), 3.0)
        self.assertEqual(math.fmod(-3.0, INF), -3.0)
        self.assertEqual(math.fmod(3.0, NINF), 3.0)
        self.assertEqual(math.fmod(-3.0, NINF), -3.0)
        self.assertEqual(math.fmod(0.0, 3.0), 0.0)
        self.assertEqual(math.fmod(0.0, NINF), 0.0)
        self.assertRaises(ValueError, math.fmod, INF, INF)

    def testFrexp(self):
        # [FUNCTION DOC] Tests math.frexp: TypeError without argument; mantissa/exponent pairs
        # for -1, 0, 1, 2 within a small tolerance via a local checker; infinities keep the
        # mantissa, NaN stays NaN.
        self.assertRaises(TypeError, math.frexp)

        def testfrexp(name, result, expected):
            # [FUNCTION DOC] HELPER: Unpacks the returned and expected (mantissa, exponent)
            # pairs and fails the test with a message naming the case if the mantissas differ by
            # more than eps or exponents differ.
            (mant, exp), (emant, eexp) = result, expected
            if abs(mant-emant) > eps or exp != eexp:
                self.fail('%s returned %r, expected %r'%\
                          (name, result, expected))

        testfrexp('frexp(-1)', math.frexp(-1), (-0.5, 1))
        testfrexp('frexp(0)', math.frexp(0), (0, 0))
        testfrexp('frexp(1)', math.frexp(1), (0.5, 1))
        testfrexp('frexp(2)', math.frexp(2), (0.5, 2))

        self.assertEqual(math.frexp(INF)[0], INF)
        self.assertEqual(math.frexp(NINF)[0], NINF)
        self.assertTrue(math.isnan(math.frexp(NAN)[0]))

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "fsum is not exact on machines with double rounding")
    def testFsum(self):
        # [FUNCTION DOC] Tests math.fsum against exactly known sums (catastrophic cancellation,
        # tie rounding, long series, partial array resizing, a telescoping sum) and 1000
        # shuffled random vectors compared with the pure Python full-precision msum. Also checks
        # infinity/NaN results, FloatLike inputs, OverflowError for overflowing sums or huge
        # ints, ValueError for inf plus -inf, TypeError for bad types, and that iterator errors
        # propagate. Skipped without IEEE 754 or on machines with double rounding.
        # math.fsum relies on exact rounding for correct operation.
        # There's a known problem with IA32 floating-point that causes
        # inexact rounding in some situations, and will cause the
        # math.fsum tests below to fail; see issue #2937.  On non IEEE
        # 754 platforms, and on IEEE 754 platforms that exhibit the
        # problem described in issue #2937, we simply skip the whole
        # test.

        # Python version of math.fsum, for comparison.  Uses a
        # different algorithm based on frexp, ldexp and integer
        # arithmetic.
        from sys import float_info
        mant_dig = float_info.mant_dig
        etiny = float_info.min_exp - mant_dig

        def msum(iterable):
            # [FUNCTION DOC] HELPER: Full-precision summation reference. Splits each value with
            # frexp into a scaled integer mantissa and exponent, accumulates in one big integer
            # at the lowest exponent seen, then rounds the exact total to a float using round-
            # half-even on the tail bits. Returns a float.
            """Full precision summation.  Compute sum(iterable) without any
            intermediate accumulation of error.  Based on the 'lsum' function
            at https://code.activestate.com/recipes/393090-binary-floating-point-summation-accurate-to-full-p/

            """
            tmant, texp = 0, 0
            for x in iterable:
                mant, exp = math.frexp(x)
                mant, exp = int(math.ldexp(mant, mant_dig)), exp - mant_dig
                if texp > exp:
                    tmant <<= texp-exp
                    texp = exp
                else:
                    mant <<= exp-texp
                tmant += mant
            # Round tmant * 2**texp to a float.  The original recipe
            # used float(str(tmant)) * 2.0**texp for this, but that's
            # a little unsafe because str -> float conversion can't be
            # relied upon to do correct rounding on all platforms.
            tail = max(len(bin(abs(tmant)))-2 - mant_dig, etiny - texp)
            if tail > 0:
                h = 1 << (tail-1)
                tmant = tmant // (2*h) + bool(tmant & h and tmant & 3*h-1)
                texp += tail
            return math.ldexp(tmant, texp)

        test_values = [
            ([], 0.0),
            ([0.0], 0.0),
            ([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50], 1e-100),
            ([1e100, 1.0, -1e100, 1e-100, 1e50, -1, -1e50], 1e-100),
            ([2.0**53, -0.5, -2.0**-54], 2.0**53-1.0),
            ([2.0**53, 1.0, 2.0**-100], 2.0**53+2.0),
            ([2.0**53+10.0, 1.0, 2.0**-100], 2.0**53+12.0),
            ([2.0**53-4.0, 0.5, 2.0**-54], 2.0**53-3.0),
            ([1./n for n in range(1, 1001)],
             float.fromhex('0x1.df11f45f4e61ap+2')),
            ([(-1.)**n/n for n in range(1, 1001)],
             float.fromhex('-0x1.62a2af1bd3624p-1')),
            ([1e16, 1., 1e-16], 10000000000000002.0),
            ([1e16-2., 1.-2.**-53, -(1e16-2.), -(1.-2.**-53)], 0.0),
            # exercise code for resizing partials array
            ([2.**n - 2.**(n+50) + 2.**(n+52) for n in range(-1074, 972, 2)] +
             [-2.**1022],
             float.fromhex('0x1.5555555555555p+970')),
            ]

        # Telescoping sum, with exact differences (due to Sterbenz)
        terms = [1.7**i for i in range(1001)]
        test_values.append((
            [terms[i+1] - terms[i] for i in range(1000)] + [-terms[1000]],
            -terms[0]
        ))

        for i, (vals, expected) in enumerate(test_values):
            try:
                actual = math.fsum(vals)
            except OverflowError:
                self.fail("test %d failed: got OverflowError, expected %r "
                          "for math.fsum(%.100r)" % (i, expected, vals))
            except ValueError:
                self.fail("test %d failed: got ValueError, expected %r "
                          "for math.fsum(%.100r)" % (i, expected, vals))
            self.assertEqual(actual, expected)

        from random import random, gauss, shuffle
        for j in range(1000):
            vals = [7, 1e100, -7, -1e100, -9e-20, 8e-20] * 10
            s = 0
            for i in range(200):
                v = gauss(0, random()) ** 7 - s
                s += v
                vals.append(v)
            shuffle(vals)

            s = msum(vals)
            self.assertEqual(msum(vals), math.fsum(vals))

        self.assertEqual(math.fsum([1.0, math.inf]), math.inf)
        self.assertTrue(math.isnan(math.fsum([math.nan, 1.0])))
        self.assertEqual(math.fsum([1e100, FloatLike(1.0), -1e100, 1e-100,
                                    1e50, FloatLike(-1.0), -1e50]), 1e-100)
        self.assertRaises(OverflowError, math.fsum, [1e+308, 1e+308])
        self.assertRaises(ValueError, math.fsum, [math.inf, -math.inf])
        self.assertRaises(TypeError, math.fsum, ['spam'])
        self.assertRaises(TypeError, math.fsum, 1)
        self.assertRaises(OverflowError, math.fsum, [10**1000])

        def bad_iter():
            # [FUNCTION DOC] HELPER: Generator yielding 1.0 then raising ZeroDivisionError, to
            # check fsum propagates iterator exceptions.
            yield 1.0
            raise ZeroDivisionError

        self.assertRaises(ZeroDivisionError, math.fsum, bad_iter())

    def testGcd(self):
        # [FUNCTION DOC] Tests math.gcd: zero and sign handling, small and huge values
        # (including combinations of negated arguments with a known common factor), zero to many
        # arguments, TypeError for floats, and acceptance of objects with the index protocol.
        gcd = math.gcd
        self.assertEqual(gcd(0, 0), 0)
        self.assertEqual(gcd(1, 0), 1)
        self.assertEqual(gcd(-1, 0), 1)
        self.assertEqual(gcd(0, 1), 1)
        self.assertEqual(gcd(0, -1), 1)
        self.assertEqual(gcd(7, 1), 1)
        self.assertEqual(gcd(7, -1), 1)
        self.assertEqual(gcd(-23, 15), 1)
        self.assertEqual(gcd(120, 84), 12)
        self.assertEqual(gcd(84, -120), 12)
        self.assertEqual(gcd(1216342683557601535506311712,
                             436522681849110124616458784), 32)

        x = 434610456570399902378880679233098819019853229470286994367836600566
        y = 1064502245825115327754847244914921553977
        for c in (652560,
                  576559230871654959816130551884856912003141446781646602790216406874):
            a = x * c
            b = y * c
            self.assertEqual(gcd(a, b), c)
            self.assertEqual(gcd(b, a), c)
            self.assertEqual(gcd(-a, b), c)
            self.assertEqual(gcd(b, -a), c)
            self.assertEqual(gcd(a, -b), c)
            self.assertEqual(gcd(-b, a), c)
            self.assertEqual(gcd(-a, -b), c)
            self.assertEqual(gcd(-b, -a), c)

        self.assertEqual(gcd(), 0)
        self.assertEqual(gcd(120), 120)
        self.assertEqual(gcd(-120), 120)
        self.assertEqual(gcd(120, 84, 102), 6)
        self.assertEqual(gcd(120, 1, 84), 1)

        self.assertRaises(TypeError, gcd, 120.0)
        self.assertRaises(TypeError, gcd, 120.0, 84)
        self.assertRaises(TypeError, gcd, 120, 84.0)
        self.assertRaises(TypeError, gcd, 120, 1, 84.0)
        self.assertEqual(gcd(MyIndexable(120), MyIndexable(84)), 12)

    def testHypot(self):
        # [FUNCTION DOC] Tests math.hypot: agreement with a plain square-root-of-sum-of-
        # squares for zero to five arguments, accepted types (int, float, Decimal, Fraction,
        # bool, float-convertible), corner cases (zero, negative zero, order independence),
        # TypeError for keyword arguments or non-numbers, errors for ints too large for
        # float, infinity beating NaN, NaN otherwise propagating, and scaling at extremely
        # large and small magnitudes.
        from decimal import Decimal
        from fractions import Fraction

        hypot = math.hypot

        # Test different numbers of arguments (from zero to five)
        # against a straightforward pure python implementation
        args = math.e, math.pi, math.sqrt(2.0), math.gamma(3.5), math.sin(2.1)
        for i in range(len(args)+1):
            self.assertAlmostEqual(
                hypot(*args[:i]),
                math.sqrt(sum(s**2 for s in args[:i]))
            )

        # Test allowable types (those with __float__)
        self.assertEqual(hypot(12.0, 5.0), 13.0)
        self.assertEqual(hypot(12, 5), 13)
        self.assertEqual(hypot(0.75, -1), 1.25)
        self.assertEqual(hypot(-1, 0.75), 1.25)
        self.assertEqual(hypot(0.75, FloatLike(-1.)), 1.25)
        self.assertEqual(hypot(FloatLike(-1.), 0.75), 1.25)
        self.assertEqual(hypot(Decimal(12), Decimal(5)), 13)
        self.assertEqual(hypot(Fraction(12, 32), Fraction(5, 32)), Fraction(13, 32))
        self.assertEqual(hypot(True, False, True, True, True), 2.0)

        # Test corner cases
        self.assertEqual(hypot(0.0, 0.0), 0.0)     # Max input is zero
        self.assertEqual(hypot(-10.5), 10.5)       # Negative input
        self.assertEqual(hypot(), 0.0)             # Negative input
        self.assertEqual(1.0,
            math.copysign(1.0, hypot(-0.0))        # Convert negative zero to positive zero
        )
        self.assertEqual(                          # Handling of moving max to the end
            hypot(1.5, 1.5, 0.5),
            hypot(1.5, 0.5, 1.5),
        )

        # Test handling of bad arguments
        with self.assertRaises(TypeError):         # Reject keyword args
            hypot(x=1)
        with self.assertRaises(TypeError):         # Reject values without __float__
            hypot(1.1, 'string', 2.2)
        int_too_big_for_float = 10 ** (sys.float_info.max_10_exp + 5)
        with self.assertRaises((ValueError, OverflowError)):
            hypot(1, int_too_big_for_float)

        # Any infinity gives positive infinity.
        self.assertEqual(hypot(INF), INF)
        self.assertEqual(hypot(0, INF), INF)
        self.assertEqual(hypot(10, INF), INF)
        self.assertEqual(hypot(-10, INF), INF)
        self.assertEqual(hypot(NAN, INF), INF)
        self.assertEqual(hypot(INF, NAN), INF)
        self.assertEqual(hypot(NINF, NAN), INF)
        self.assertEqual(hypot(NAN, NINF), INF)
        self.assertEqual(hypot(-INF, INF), INF)
        self.assertEqual(hypot(-INF, -INF), INF)
        self.assertEqual(hypot(10, -INF), INF)

        # If no infinity, any NaN gives a NaN.
        self.assertTrue(math.isnan(hypot(NAN)))
        self.assertTrue(math.isnan(hypot(0, NAN)))
        self.assertTrue(math.isnan(hypot(NAN, 10)))
        self.assertTrue(math.isnan(hypot(10, NAN)))
        self.assertTrue(math.isnan(hypot(NAN, NAN)))
        self.assertTrue(math.isnan(hypot(NAN)))

        # Verify scaling for extremely large values
        fourthmax = FLOAT_MAX / 4.0
        for n in range(32):
            self.assertTrue(math.isclose(hypot(*([fourthmax]*n)),
                                         fourthmax * math.sqrt(n)))

        # Verify scaling for extremely small values
        for exp in range(32):
            scale = FLOAT_MIN / 2.0 ** exp
            self.assertEqual(math.hypot(4*scale, 3*scale), 5*scale)

        self.assertRaises(TypeError, math.hypot, *([1.0]*18), 'spam')

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                     "hypot() loses accuracy on machines with double rounding")
    def testHypotAccuracy(self):
        # [FUNCTION DOC] Tests math.hypot accuracy: for a list of hex-encoded float pairs that
        # once produced 1-2 ulp errors, requires the result to equal the correctly rounded value
        # computed with a 500-digit decimal context. Each pair runs in its own subTest. Skipped
        # without IEEE 754 or with double rounding.
        # Verify improved accuracy in cases that were known to be inaccurate.
        #
        # The new algorithm's accuracy depends on IEEE 754 arithmetic
        # guarantees, on having the usual ROUND HALF EVEN rounding mode, on
        # the system not having double rounding due to extended precision,
        # and on the compiler maintaining the specified order of operations.
        #
        # This test is known to succeed on most of our builds.  If it fails
        # some build, we either need to add another skipIf if the cause is
        # identifiable; otherwise, we can remove this test entirely.

        hypot = math.hypot
        Decimal = decimal.Decimal
        high_precision = decimal.Context(prec=500)

        for hx, hy in [
            # Cases with a 1 ulp error in Python 3.7 compiled with Clang
            ('0x1.10e89518dca48p+29', '0x1.1970f7565b7efp+30'),
            ('0x1.10106eb4b44a2p+29', '0x1.ef0596cdc97f8p+29'),
            ('0x1.459c058e20bb7p+30', '0x1.993ca009b9178p+29'),
            ('0x1.378371ae67c0cp+30', '0x1.fbe6619854b4cp+29'),
            ('0x1.f4cd0574fb97ap+29', '0x1.50fe31669340ep+30'),
            ('0x1.494b2cdd3d446p+29', '0x1.212a5367b4c7cp+29'),
            ('0x1.f84e649f1e46dp+29', '0x1.1fa56bef8eec4p+30'),
            ('0x1.2e817edd3d6fap+30', '0x1.eb0814f1e9602p+29'),
            ('0x1.0d3a6e3d04245p+29', '0x1.32a62fea52352p+30'),
            ('0x1.888e19611bfc5p+29', '0x1.52b8e70b24353p+29'),

            # Cases with 2 ulp error in Python 3.8
            ('0x1.538816d48a13fp+29', '0x1.7967c5ca43e16p+29'),
            ('0x1.57b47b7234530p+29', '0x1.74e2c7040e772p+29'),
            ('0x1.821b685e9b168p+30', '0x1.677dc1c1e3dc6p+29'),
            ('0x1.9e8247f67097bp+29', '0x1.24bd2dc4f4baep+29'),
            ('0x1.b73b59e0cb5f9p+29', '0x1.da899ab784a97p+28'),
            ('0x1.94a8d2842a7cfp+30', '0x1.326a51d4d8d8ap+30'),
            ('0x1.e930b9cd99035p+29', '0x1.5a1030e18dff9p+30'),
            ('0x1.1592bbb0e4690p+29', '0x1.a9c337b33fb9ap+29'),
            ('0x1.1243a50751fd4p+29', '0x1.a5a10175622d9p+29'),
            ('0x1.57a8596e74722p+30', '0x1.42d1af9d04da9p+30'),

            # Cases with 1 ulp error in version fff3c28052e6b0
            ('0x1.ee7dbd9565899p+29', '0x1.7ab4d6fc6e4b4p+29'),
            ('0x1.5c6bfbec5c4dcp+30', '0x1.02511184b4970p+30'),
            ('0x1.59dcebba995cap+30', '0x1.50ca7e7c38854p+29'),
            ('0x1.768cdd94cf5aap+29', '0x1.9cfdc5571d38ep+29'),
            ('0x1.dcf137d60262ep+29', '0x1.1101621990b3ep+30'),
            ('0x1.3a2d006e288b0p+30', '0x1.e9a240914326cp+29'),
            ('0x1.62a32f7f53c61p+29', '0x1.47eb6cd72684fp+29'),
            ('0x1.d3bcb60748ef2p+29', '0x1.3f13c4056312cp+30'),
            ('0x1.282bdb82f17f3p+30', '0x1.640ba4c4eed3ap+30'),
            ('0x1.89d8c423ea0c6p+29', '0x1.d35dcfe902bc3p+29'),
        ]:
            x = float.fromhex(hx)
            y = float.fromhex(hy)
            with self.subTest(hx=hx, hy=hy, x=x, y=y):
                with decimal.localcontext(high_precision):
                    z = float((Decimal(x)**2 + Decimal(y)**2).sqrt())
                self.assertEqual(hypot(x, y), z)

    def testDist(self):
        # [FUNCTION DOC] Tests math.dist: simple exact cases, random vectors up to nine
        # dimensions against a naive formula, non-tuple iterables, accepted element types,
        # corner cases (zero distance, empty vectors, positive zero result, order independence),
        # tuple subclasses, many error cases (keyword arguments, wrong arity, scalars, dimension
        # mismatch, bad element types, overflowing ints, raising float conversion), equivalence
        # to abs in one dimension, a cartesian grid of special values (infinite difference gives
        # inf, else NaN) and scaling for extremely large and small values.
        from decimal import Decimal as D
        from fractions import Fraction as F

        dist = math.dist
        sqrt = math.sqrt

        # Simple exact cases
        self.assertEqual(dist((1.0, 2.0, 3.0), (4.0, 2.0, -1.0)), 5.0)
        self.assertEqual(dist((1, 2, 3), (4, 2, -1)), 5.0)

        # Test different numbers of arguments (from zero to nine)
        # against a straightforward pure python implementation
        for i in range(9):
            for j in range(5):
                p = tuple(random.uniform(-5, 5) for k in range(i))
                q = tuple(random.uniform(-5, 5) for k in range(i))
                self.assertAlmostEqual(
                    dist(p, q),
                    sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
                )

        # Test non-tuple inputs
        self.assertEqual(dist([1.0, 2.0, 3.0], [4.0, 2.0, -1.0]), 5.0)
        self.assertEqual(dist(iter([1.0, 2.0, 3.0]), iter([4.0, 2.0, -1.0])), 5.0)

        # Test allowable types (those with __float__)
        self.assertEqual(dist((14.0, 1.0), (2.0, -4.0)), 13.0)
        self.assertEqual(dist((14, 1), (2, -4)), 13)
        self.assertEqual(dist((FloatLike(14.), 1), (2, -4)), 13)
        self.assertEqual(dist((11, 1), (FloatLike(-1.), -4)), 13)
        self.assertEqual(dist((14, FloatLike(-1.)), (2, -6)), 13)
        self.assertEqual(dist((14, -1), (2, -6)), 13)
        self.assertEqual(dist((D(14), D(1)), (D(2), D(-4))), D(13))
        self.assertEqual(dist((F(14, 32), F(1, 32)), (F(2, 32), F(-4, 32))),
                         F(13, 32))
        self.assertEqual(dist((True, True, False, False, True, True),
                              (True, False, True, False, False, False)),
                         2.0)

        # Test corner cases
        self.assertEqual(dist((13.25, 12.5, -3.25),
                              (13.25, 12.5, -3.25)),
                         0.0)                      # Distance with self is zero
        self.assertEqual(dist((), ()), 0.0)        # Zero-dimensional case
        self.assertEqual(1.0,                      # Convert negative zero to positive zero
            math.copysign(1.0, dist((-0.0,), (0.0,)))
        )
        self.assertEqual(1.0,                      # Convert negative zero to positive zero
            math.copysign(1.0, dist((0.0,), (-0.0,)))
        )
        self.assertEqual(                          # Handling of moving max to the end
            dist((1.5, 1.5, 0.5), (0, 0, 0)),
            dist((1.5, 0.5, 1.5), (0, 0, 0))
        )

        # Verify tuple subclasses are allowed
        class T(tuple):
            # [CLASS DOC] HELPER: Local tuple subclass checking that dist
            # accepts tuple subclasses.
            pass
        self.assertEqual(dist(T((1, 2, 3)), ((4, 2, -1))), 5.0)

        # Test handling of bad arguments
        with self.assertRaises(TypeError):         # Reject keyword args
            dist(p=(1, 2, 3), q=(4, 5, 6))
        with self.assertRaises(TypeError):         # Too few args
            dist((1, 2, 3))
        with self.assertRaises(TypeError):         # Too many args
            dist((1, 2, 3), (4, 5, 6), (7, 8, 9))
        with self.assertRaises(TypeError):         # Scalars not allowed
            dist(1, 2)
        with self.assertRaises(TypeError):         # Reject values without __float__
            dist((1.1, 'string', 2.2), (1, 2, 3))
        with self.assertRaises(ValueError):        # Check dimension agree
            dist((1, 2, 3, 4), (5, 6, 7))
        with self.assertRaises(ValueError):        # Check dimension agree
            dist((1, 2, 3), (4, 5, 6, 7))
        with self.assertRaises(TypeError):
            dist((1,)*17 + ("spam",), (1,)*18)
        with self.assertRaises(TypeError):         # Rejects invalid types
            dist("abc", "xyz")
        int_too_big_for_float = 10 ** (sys.float_info.max_10_exp + 5)
        with self.assertRaises((ValueError, OverflowError)):
            dist((1, int_too_big_for_float), (2, 3))
        with self.assertRaises((ValueError, OverflowError)):
            dist((2, 3), (1, int_too_big_for_float))
        with self.assertRaises(TypeError):
            dist((1,), 2)
        with self.assertRaises(TypeError):
            dist([1], 2)

        class BadFloat:
            # [CLASS DOC] HELPER: Local fixture whose float conversion attribute is a raising
            # descriptor (ValueError); dist must propagate it.
            __float__ = BadDescr()

        with self.assertRaises(ValueError):
            dist([1], [BadFloat()])

        # Verify that the one dimensional case is equivalent to abs()
        for i in range(20):
            p, q = random.random(), random.random()
            self.assertEqual(dist((p,), (q,)), abs(p - q))

        # Test special values
        values = [NINF, -10.5, -0.0, 0.0, 10.5, INF, NAN]
        for p in itertools.product(values, repeat=3):
            for q in itertools.product(values, repeat=3):
                diffs = [px - qx for px, qx in zip(p, q)]
                if any(map(math.isinf, diffs)):
                    # Any infinite difference gives positive infinity.
                    self.assertEqual(dist(p, q), INF)
                elif any(map(math.isnan, diffs)):
                    # If no infinity, any NaN gives a NaN.
                    self.assertTrue(math.isnan(dist(p, q)))

        # Verify scaling for extremely large values
        fourthmax = FLOAT_MAX / 4.0
        for n in range(32):
            p = (fourthmax,) * n
            q = (0.0,) * n
            self.assertTrue(math.isclose(dist(p, q), fourthmax * math.sqrt(n)))
            self.assertTrue(math.isclose(dist(q, p), fourthmax * math.sqrt(n)))

        # Verify scaling for extremely small values
        for exp in range(32):
            scale = FLOAT_MIN / 2.0 ** exp
            p = (4*scale, 3*scale)
            q = (0.0, 0.0)
            self.assertEqual(math.dist(p, q), 5*scale)
            self.assertEqual(math.dist(q, p), 5*scale)

    def test_math_dist_leak(self):
        # [FUNCTION DOC] Regression test for gh-98897: math.dist with vectors of different
        # lengths must raise ValueError (error handling must not leak memory).
        # gh-98897: Check for error handling does not leak memory
        with self.assertRaises(ValueError):
            math.dist([1, 2], [3, 4, 5])

    @slowTest
    def testIsqrt(self):
        # [FUNCTION DOC] Tests math.isqrt: for about thousands of small and huge values, result
        # is an exact int floor square root (s*s <= value < (s+1)*(s+1)); negative input raises
        # ValueError; booleans and index-protocol objects are accepted (negative ones raise
        # ValueError); floats, strings, Decimal and complex raise TypeError. Marked slow.
        # Test a variety of inputs, large and small.
        test_values = (
            list(range(1000))
            + list(range(10**6 - 1000, 10**6 + 1000))
            + [2**e + i for e in range(60, 200) for i in range(-40, 40)]
            + [3**9999, 10**5001]
        )

        for value in test_values:
            with self.subTest(value=value):
                s = math.isqrt(value)
                self.assertIs(type(s), int)
                self.assertLessEqual(s*s, value)
                self.assertLess(value, (s+1)*(s+1))

        # Negative values
        with self.assertRaises(ValueError):
            math.isqrt(-1)

        # Integer-like things
        s = math.isqrt(True)
        self.assertIs(type(s), int)
        self.assertEqual(s, 1)

        s = math.isqrt(False)
        self.assertIs(type(s), int)
        self.assertEqual(s, 0)

        class IntegerLike(object):
            # [CLASS DOC] HELPER: Local fixture exposing only the index protocol
            # for isqrt checks.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the integer value.
                self.value = value

            def __index__(self):
                # [FUNCTION DOC] HELPER: Index protocol; returns the stored value.
                return self.value

        s = math.isqrt(IntegerLike(1729))
        self.assertIs(type(s), int)
        self.assertEqual(s, 41)

        with self.assertRaises(ValueError):
            math.isqrt(IntegerLike(-3))

        # Non-integer-like things
        bad_values = [
            3.5, "a string", decimal.Decimal("3.5"), 3.5j,
            100.0, -4.0,
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    math.isqrt(value)

    def test_lcm(self):
        # [FUNCTION DOC] Tests math.lcm: zero handling, sign handling, small and huge values
        # (with combinations of negated arguments against a computed product), zero to many
        # arguments, TypeError for floats and acceptance of index-protocol objects.
        lcm = math.lcm
        self.assertEqual(lcm(0, 0), 0)
        self.assertEqual(lcm(1, 0), 0)
        self.assertEqual(lcm(-1, 0), 0)
        self.assertEqual(lcm(0, 1), 0)
        self.assertEqual(lcm(0, -1), 0)
        self.assertEqual(lcm(7, 1), 7)
        self.assertEqual(lcm(7, -1), 7)
        self.assertEqual(lcm(-23, 15), 345)
        self.assertEqual(lcm(120, 84), 840)
        self.assertEqual(lcm(84, -120), 840)
        self.assertEqual(lcm(1216342683557601535506311712,
                             436522681849110124616458784),
                             16592536571065866494401400422922201534178938447014944)

        x = 43461045657039990237
        y = 10645022458251153277
        for c in (652560,
                  57655923087165495981):
            a = x * c
            b = y * c
            d = x * y * c
            self.assertEqual(lcm(a, b), d)
            self.assertEqual(lcm(b, a), d)
            self.assertEqual(lcm(-a, b), d)
            self.assertEqual(lcm(b, -a), d)
            self.assertEqual(lcm(a, -b), d)
            self.assertEqual(lcm(-b, a), d)
            self.assertEqual(lcm(-a, -b), d)
            self.assertEqual(lcm(-b, -a), d)

        self.assertEqual(lcm(), 1)
        self.assertEqual(lcm(120), 120)
        self.assertEqual(lcm(-120), 120)
        self.assertEqual(lcm(120, 84, 102), 14280)
        self.assertEqual(lcm(120, 0, 84), 0)

        self.assertRaises(TypeError, lcm, 120.0)
        self.assertRaises(TypeError, lcm, 120.0, 84)
        self.assertRaises(TypeError, lcm, 120, 84.0)
        self.assertRaises(TypeError, lcm, 120, 0, 84.0)
        self.assertEqual(lcm(MyIndexable(120), MyIndexable(84)), 840)

    def testLdexp(self):
        # [FUNCTION DOC] Tests math.ldexp: TypeError for missing or non-int exponent,
        # simple scalings, OverflowError on overflow, underflow to signed zero, infinities
        # and NaN passing through, and exponents of enormous magnitude (up to 10**40)
        # giving the same limiting results.
        self.assertRaises(TypeError, math.ldexp)
        self.assertRaises(TypeError, math.ldexp, 2.0, 1.1)
        self.ftest('ldexp(0,1)', math.ldexp(0,1), 0)
        self.ftest('ldexp(1,1)', math.ldexp(1,1), 2)
        self.ftest('ldexp(1,-1)', math.ldexp(1,-1), 0.5)
        self.ftest('ldexp(-1,1)', math.ldexp(-1,1), -2)
        self.assertRaises(OverflowError, math.ldexp, 1., 1000000)
        self.assertRaises(OverflowError, math.ldexp, -1., 1000000)
        self.assertEqual(math.ldexp(1., -1000000), 0.)
        self.assertEqual(math.ldexp(-1., -1000000), -0.)
        self.assertEqual(math.ldexp(INF, 30), INF)
        self.assertEqual(math.ldexp(NINF, -213), NINF)
        self.assertTrue(math.isnan(math.ldexp(NAN, 0)))

        # large second argument
        for n in [10**5, 10**10, 10**20, 10**40]:
            self.assertEqual(math.ldexp(INF, -n), INF)
            self.assertEqual(math.ldexp(NINF, -n), NINF)
            self.assertEqual(math.ldexp(1., -n), 0.)
            self.assertEqual(math.ldexp(-1., -n), -0.)
            self.assertEqual(math.ldexp(0., -n), 0.)
            self.assertEqual(math.ldexp(-0., -n), -0.)
            self.assertTrue(math.isnan(math.ldexp(NAN, -n)))

            self.assertRaises(OverflowError, math.ldexp, 1., n)
            self.assertRaises(OverflowError, math.ldexp, -1., n)
            self.assertEqual(math.ldexp(0., n), 0.)
            self.assertEqual(math.ldexp(-0., n), -0.)
            self.assertEqual(math.ldexp(INF, n), INF)
            self.assertEqual(math.ldexp(NINF, n), NINF)
            self.assertTrue(math.isnan(math.ldexp(NAN, n)))

    def testLog(self):
        # [FUNCTION DOC] Tests math.log: TypeError for wrong arity, natural log and explicit
        # bases, huge ints, ValueError for negative arguments or negative base, log of -inf
        # raises, +inf gives inf, NaN propagates.
        self.assertRaises(TypeError, math.log)
        self.assertRaises(TypeError, math.log, 1, 2, 3)
        self.ftest('log(1/e)', math.log(1/math.e), -1)
        self.ftest('log(1)', math.log(1), 0)
        self.ftest('log(e)', math.log(math.e), 1)
        self.ftest('log(32,2)', math.log(32,2), 5)
        self.ftest('log(10**40, 10)', math.log(10**40, 10), 40)
        self.ftest('log(10**40, 10**20)', math.log(10**40, 10**20), 2)
        self.ftest('log(10**1000)', math.log(10**1000),
                   2302.5850929940457)
        self.assertRaises(ValueError, math.log, -1.5)
        self.assertRaises(ValueError, math.log, -10**1000)
        self.assertRaises(ValueError, math.log, 10, -10)
        self.assertRaises(ValueError, math.log, NINF)
        self.assertEqual(math.log(INF), INF)
        self.assertTrue(math.isnan(math.log(NAN)))

    def testLog1p(self):
        # [FUNCTION DOC] Tests math.log1p: TypeError without argument, big ints agree with float
        # conversion, ValueError at -1, +inf maps to inf.
        self.assertRaises(TypeError, math.log1p)
        for n in [2, 2**90, 2**300]:
            self.assertAlmostEqual(math.log1p(n), math.log1p(float(n)))
        self.assertRaises(ValueError, math.log1p, -1)
        self.assertEqual(math.log1p(INF), INF)

    @skipIfTorchDynamo("Infinite loop")
    @requires_IEEE_754
    def testLog2(self):
        # [FUNCTION DOC] Tests math.log2: arity error, exact results for 1, 2, 4 and huge powers
        # of two as ints, ValueError for negative or -inf input, NaN propagates. Skipped under
        # Dynamo (infinite loop) and when IEEE 754 is not available.
        self.assertRaises(TypeError, math.log2)

        # Check some integer values
        self.assertEqual(math.log2(1), 0.0)
        self.assertEqual(math.log2(2), 1.0)
        self.assertEqual(math.log2(4), 2.0)

        # Large integer values
        self.assertEqual(math.log2(2**1023), 1023.0)
        self.assertEqual(math.log2(2**1024), 1024.0)
        self.assertEqual(math.log2(2**2000), 2000.0)

        self.assertRaises(ValueError, math.log2, -1.5)
        self.assertRaises(ValueError, math.log2, NINF)
        self.assertTrue(math.isnan(math.log2(NAN)))

    @skipIfTorchDynamo("Infinite loop")
    @requires_IEEE_754
    # log2() is not accurate enough on Mac OS X Tiger (10.4)
    @support.requires_mac_ver(10, 5)
    def testLog2Exact(self):
        # [FUNCTION DOC] Tests that math.log2 of every power of two from 2**-1074 to 2**1023 is
        # exactly the exponent. Skipped under Dynamo and on very old macOS.
        # Check that we get exact equality for log2 of powers of 2.
        actual = [math.log2(math.ldexp(1.0, n)) for n in range(-1074, 1024)]
        expected = [float(n) for n in range(-1074, 1024)]
        self.assertEqual(actual, expected)

    def testLog10(self):
        # [FUNCTION DOC] Tests math.log10: known values, huge int, ValueError for negative or
        # -inf input, NaN propagates.
        # KNOWN ISSUE: the infinity check calls math.log instead of math.log10 (copied upstream
        # typo), so log10 of infinity is not actually tested.
        self.assertRaises(TypeError, math.log10)
        self.ftest('log10(0.1)', math.log10(0.1), -1)
        self.ftest('log10(1)', math.log10(1), 0)
        self.ftest('log10(10)', math.log10(10), 1)
        self.ftest('log10(10**1000)', math.log10(10**1000), 1000.0)
        self.assertRaises(ValueError, math.log10, -1.5)
        self.assertRaises(ValueError, math.log10, -10**1000)
        self.assertRaises(ValueError, math.log10, NINF)
        self.assertEqual(math.log(INF), INF)
        self.assertTrue(math.isnan(math.log10(NAN)))

    def testSumProd(self):
        # [FUNCTION DOC] Tests math.sumprod: core results, result type matching the
        # equivalent plain sum of products for ints, floats, Fraction and Decimal mixes,
        # TypeError for wrong arity or non-iterables or bad elements, ValueError for unequal
        # lengths, large values and OverflowError for ints too large for float, errors from
        # iterators or from multiplication or addition propagating, inf/NaN behavior, and a
        # regression case from development.
        sumprod = math.sumprod
        Decimal = decimal.Decimal
        Fraction = fractions.Fraction

        # Core functionality
        self.assertEqual(sumprod(iter([10, 20, 30]), (1, 2, 3)), 140)
        self.assertEqual(sumprod([1.5, 2.5], [3.5, 4.5]), 16.5)
        self.assertEqual(sumprod([], []), 0)
        self.assertEqual(sumprod([-1], [1.]), -1)
        self.assertEqual(sumprod([1.], [-1]), -1)

        # Type preservation and coercion
        for v in [
            (10, 20, 30),
            (1.5, -2.5),
            (Fraction(3, 5), Fraction(4, 5)),
            (Decimal(3.5), Decimal(4.5)),
            (2.5, 10),             # float/int
            (2.5, Fraction(3, 5)), # float/fraction
            (25, Fraction(3, 5)),  # int/fraction
            (25, Decimal(4.5)),    # int/decimal
        ]:
            for p, q in [(v, v), (v, v[::-1])]:
                with self.subTest(p=p, q=q):
                    expected = sum(p_i * q_i for p_i, q_i in zip(p, q, strict=True))
                    actual = sumprod(p, q)
                    self.assertEqual(expected, actual)
                    self.assertEqual(type(expected), type(actual))

        # Bad arguments
        self.assertRaises(TypeError, sumprod)               # No args
        self.assertRaises(TypeError, sumprod, [])           # One arg
        self.assertRaises(TypeError, sumprod, [], [], [])   # Three args
        self.assertRaises(TypeError, sumprod, None, [10])   # Non-iterable
        self.assertRaises(TypeError, sumprod, [10], None)   # Non-iterable
        self.assertRaises(TypeError, sumprod, ['x'], [1.0])

        # Uneven lengths
        self.assertRaises(ValueError, sumprod, [10, 20], [30])
        self.assertRaises(ValueError, sumprod, [10], [20, 30])

        # Overflows
        self.assertEqual(sumprod([10**20], [1]), 10**20)
        self.assertEqual(sumprod([1], [10**20]), 10**20)
        self.assertEqual(sumprod([10**10], [10**10]), 10**20)
        self.assertEqual(sumprod([10**7]*10**5, [10**7]*10**5), 10**19)
        self.assertRaises(OverflowError, sumprod, [10**1000], [1.0])
        self.assertRaises(OverflowError, sumprod, [1.0], [10**1000])

        # Error in iterator
        def raise_after(n):
            # [FUNCTION DOC] HELPER: Generator yielding 0..n-1 and then raising RuntimeError.
            for i in range(n):
                yield i
            raise RuntimeError
        with self.assertRaises(RuntimeError):
            sumprod(range(10), raise_after(5))
        with self.assertRaises(RuntimeError):
            sumprod(raise_after(5), range(10))

        from test_iter import BasicIterClass

        self.assertEqual(sumprod(BasicIterClass(1), [1]), 0)
        self.assertEqual(sumprod([1], BasicIterClass(1)), 0)

        # Error in multiplication
        class BadMultiply:
            # [CLASS DOC] HELPER: Local fixture whose multiplication (both
            # directions) raises RuntimeError.
            def __mul__(self, other):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError
            def __rmul__(self, other):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError
        with self.assertRaises(RuntimeError):
            sumprod([10, BadMultiply(), 30], [1, 2, 3])
        with self.assertRaises(RuntimeError):
            sumprod([1, 2, 3], [10, BadMultiply(), 30])

        # Error in addition
        with self.assertRaises(TypeError):
            sumprod(['abc', 3], [5, 10])
        with self.assertRaises(TypeError):
            sumprod([5, 10], ['abc', 3])

        # Special values should give the same as the pure python recipe
        self.assertEqual(sumprod([10.1, math.inf], [20.2, 30.3]), math.inf)
        self.assertEqual(sumprod([10.1, math.inf], [math.inf, 30.3]), math.inf)
        self.assertEqual(sumprod([10.1, math.inf], [math.inf, math.inf]), math.inf)
        self.assertEqual(sumprod([10.1, -math.inf], [20.2, 30.3]), -math.inf)
        self.assertTrue(math.isnan(sumprod([10.1, math.inf], [-math.inf, math.inf])))
        self.assertTrue(math.isnan(sumprod([10.1, math.nan], [20.2, 30.3])))
        self.assertTrue(math.isnan(sumprod([10.1, math.inf], [math.nan, 30.3])))
        self.assertTrue(math.isnan(sumprod([10.1, math.inf], [20.3, math.nan])))

        # Error cases that arose during development
        args = ((-5, -5, 10), (1.5, 4611686018427387904, 2305843009213693952))
        self.assertEqual(sumprod(*args), 0.0)


    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "sumprod() accuracy not guaranteed on machines with double rounding")
    @support.cpython_only    # Other implementations may choose a different algorithm
    def test_sumprod_accuracy(self):
        # [FUNCTION DOC] Tests math.sumprod accuracy: results must be correctly rounded for sums
        # such as ten times 0.1 and a cancelling pair of large values. CPython only, IEEE 754
        # only, skipped with double rounding.
        sumprod = math.sumprod
        self.assertEqual(sumprod([0.1] * 10, [1]*10), 1.0)
        self.assertEqual(sumprod([0.1] * 20, [True, False] * 10), 1.0)
        self.assertEqual(sumprod([True, False] * 10, [0.1] * 20), 1.0)
        self.assertEqual(sumprod([1.0, 10E100, 1.0, -10E100], [1.0]*4), 2.0)

    @support.requires_resource('cpu')
    def test_sumprod_stress(self):
        # [FUNCTION DOC] Stress test (needs the cpu resource): builds pools of mixed numeric
        # values (ints, floats, huge magnitudes, inf/NaN, complex, Decimal, Fraction,
        # int/float subclasses) and, for every pair of sequences of length 0 to 3 from each
        # pool, requires sumprod to give the same outcome (exception type, result type and
        # repr) as a naive loop.
        sumprod = math.sumprod
        product = itertools.product
        Decimal = decimal.Decimal
        Fraction = fractions.Fraction

        class Int(int):
            # [CLASS DOC] HELPER: Local int subclass whose add and multiply return Int and
            # whose repr shows the value, so sumprod's handling of subclasses can be compared
            # with the baseline.
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as plain ints and wraps the result in Int.
                return Int(int(self) + int(other))
            def __mul__(self, other):
                # [FUNCTION DOC] HELPER: Multiplies as plain ints and wraps the result in Int.
                return Int(int(self) * int(other))
            __radd__ = __add__
            __rmul__ = __mul__
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a repr of the form Int(value).
                return f'Int({int(self)})'

        class Flt(float):
            # [CLASS DOC] HELPER: Local float subclass whose add and multiply truncate to int
            # and return Int.
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Converts both operands to int, adds, returns Int.
                return Int(int(self) + int(other))
            def __mul__(self, other):
                # [FUNCTION DOC] HELPER: Converts both operands to int, multiplies, returns Int.
                return Int(int(self) * int(other))
            __radd__ = __add__
            __rmul__ = __mul__
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a repr of the form Flt(truncated int value).
                return f'Flt({int(self)})'

        def baseline_sumprod(p, q):
            # [FUNCTION DOC] HELPER: Reference implementation: sums products of strictly zipped
            # elements starting from int zero. Raises ValueError for unequal lengths. Defines
            # the target behavior.
            """This defines the target behavior including exceptions and special values.
            However, it is subject to rounding errors, so float inputs should be exactly
            representable with only a few bits.
            """
            total = 0
            for p_i, q_i in zip(p, q, strict=True):
                total += p_i * q_i
            return total

        def run(func, *args):
            # [FUNCTION DOC] HELPER: Calls func with args and returns a triple of (exception
            # type or None, result type or None, repr of result or the string None).
            # AssertionError and NameError are re-raised so test bugs are not hidden.
            "Make comparing functions easier. Returns error status, type, and result."
            try:
                result = func(*args)
            except (AssertionError, NameError):
                raise
            except Exception as e:
                return type(e), None, 'None'
            return None, type(result), repr(result)

        pools = [
            (-5, 10, -2**20, 2**31, 2**40, 2**61, 2**62, 2**80, 1.5, Int(7)),
            (5.25, -3.5, 4.75, 11.25, 400.5, 0.046875, 0.25, -1.0, -0.078125),
            (-19.0*2**500, 11*2**1000, -3*2**1500, 17*2*333,
               5.25, -3.25, -3.0*2**(-333),  3, 2**513),
            (3.75, 2.5, -1.5, float('inf'), -float('inf'), float('NaN'), 14,
                9, 3+4j, Flt(13), 0.0),
            (13.25, -4.25, Decimal('10.5'), Decimal('-2.25'), Fraction(13, 8),
                 Fraction(-11, 16), 4.75 + 0.125j, 97, -41, Int(3)),
            (Decimal('6.125'), Decimal('12.375'), Decimal('-2.75'), Decimal(0),
                 Decimal('Inf'), -Decimal('Inf'), Decimal('NaN'), 12, 13.5),
            (-2.0 ** -1000, 11*2**1000, 3, 7, -37*2**32, -2*2**-537, -2*2**-538,
                 2*2**-513),
            (-7 * 2.0 ** -510, 5 * 2.0 ** -520, 17, -19.0, -6.25),
            (11.25, -3.75, -0.625, 23.375, True, False, 7, Int(5)),
        ]

        for pool in pools:
            for size in range(4):
                for args1 in product(pool, repeat=size):
                    for args2 in product(pool, repeat=size):
                        args = (args1, args2)
                        self.assertEqual(
                            run(baseline_sumprod, *args),
                            run(sumprod, *args),
                            args,
                        )

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "sumprod() accuracy not guaranteed on machines with double rounding")
    @support.cpython_only    # Other implementations may choose a different algorithm
    @support.requires_resource('cpu')
    def test_sumprod_extended_precision_accuracy(self):
        # [FUNCTION DOC] Tests extended-precision accuracy of math.sumprod: generates ill-
        # conditioned dot products (condition number 1e30) with the GenDot algorithm,
        # computes the exact value with Fractions, and requires the median relative error
        # over 1000 trials to be below 1e-16. Failure suggests an inaccurate C fma. CPython,
        # IEEE 754 and cpu-resource only.
        import operator
        from fractions import Fraction
        from itertools import starmap
        from collections import namedtuple
        from math import log2, exp2, fabs
        from random import choices, uniform, shuffle
        from statistics import median

        DotExample = namedtuple('DotExample', ('x', 'y', 'target_sumprod', 'condition'))

        def DotExact(x, y):
            # [FUNCTION DOC] HELPER: Exact dot product of two sequences
            # using Fraction arithmetic.
            vec1 = map(Fraction, x)
            vec2 = map(Fraction, y)
            return sum(starmap(operator.mul, zip(vec1, vec2, strict=True)))

        def Condition(x, y):
            # [FUNCTION DOC] HELPER: Condition number of the dot product: twice the exact dot of
            # absolute values divided by the absolute exact dot.
            return 2.0 * DotExact(map(abs, x), map(abs, y)) / abs(DotExact(x, y))

        def linspace(lo, hi, n):
            # [FUNCTION DOC] HELPER: Returns n evenly spaced floats from lo to hi inclusive.
            width = (hi - lo) / (n - 1)
            return [lo + width * i for i in range(n)]

        def GenDot(n, c):
            # [FUNCTION DOC] HELPER: Generates two random vectors x and y of length n (at
            # least 6, asserted) whose dot product has roughly condition number c: first half
            # random with large exponents, second half built to cancel, then pairs shuffled.
            # Returns a DotExample with x, y, exact dot and achieved condition number. Uses
            # the random module (non-deterministic).
            """ Algorithm 6.1 (GenDot) works as follows. The condition number (5.7) of
            the dot product xT y is proportional to the degree of cancellation. In
            order to achieve a prescribed cancellation, we generate the first half of
            the vectors x and y randomly within a large exponent range. This range is
            chosen according to the anticipated condition number. The second half of x
            and y is then constructed choosing xi randomly with decreasing exponent,
            and calculating yi such that some cancellation occurs. Finally, we permute
            the vectors x, y randomly and calculate the achieved condition number.
            """

            assert n >= 6
            n2 = n // 2
            x = [0.0] * n
            y = [0.0] * n
            b = log2(c)

            # First half with exponents from 0 to |_b/2_| and random ints in between
            e = choices(range(int(b/2)), k=n2)
            e[0] = int(b / 2) + 1
            e[-1] = 0.0

            x[:n2] = [uniform(-1.0, 1.0) * exp2(p) for p in e]
            y[:n2] = [uniform(-1.0, 1.0) * exp2(p) for p in e]

            # Second half
            e = list(map(round, linspace(b/2, 0.0 , n-n2)))
            for i in range(n2, n):
                x[i] = uniform(-1.0, 1.0) * exp2(e[i - n2])
                y[i] = (uniform(-1.0, 1.0) * exp2(e[i - n2]) - DotExact(x, y)) / x[i]

            # Shuffle
            pairs = list(zip(x, y))
            shuffle(pairs)
            x, y = zip(*pairs)

            return DotExample(x, y, DotExact(x, y), Condition(x, y))

        def RelativeError(res, ex):
            # [FUNCTION DOC] HELPER: Relative error of a computed dot product res, measured by
            # the exact dot of the inputs extended with minus res, divided by the exact target.
            # Returns a float.
            x, y, target_sumprod, condition = ex
            n = DotExact(list(x) + [-res], list(y) + [1])
            return fabs(n / target_sumprod)

        def Trial(dotfunc, c, n):
            # [FUNCTION DOC] HELPER: One trial: generates a length-10 example with condition c,
            # applies dotfunc and returns the relative error. The n parameter is unused.
            ex = GenDot(10, c)
            res = dotfunc(ex.x, ex.y)
            return RelativeError(res, ex)

        times = 1000          # Number of trials
        n = 20                # Length of vectors
        c = 1e30              # Target condition number

        # If the following test fails, it means that the C math library
        # implementation of fma() is not compliant with the C99 standard
        # and is inaccurate.  To solve this problem, make a new build
        # with the symbol UNRELIABLE_FMA defined.  That will enable a
        # slower but accurate code path that avoids the fma() call.
        relative_err = median(Trial(math.sumprod, c, n) for i in range(times))
        self.assertLess(relative_err, 1e-16)

    def testModf(self):
        # [FUNCTION DOC] Tests math.modf: TypeError without argument, fractional and integral
        # parts for positive and negative values via a local checker, infinities give signed
        # zero fraction, NaN gives NaN for both parts.
        self.assertRaises(TypeError, math.modf)

        def testmodf(name, result, expected):
            # [FUNCTION DOC] HELPER: Compares returned and expected (fraction, integer) pairs,
            # failing with a message if they differ by more than eps.
            (v1, v2), (e1, e2) = result, expected
            if abs(v1-e1) > eps or abs(v2-e2):
                self.fail('%s returned %r, expected %r'%\
                          (name, result, expected))

        testmodf('modf(1.5)', math.modf(1.5), (0.5, 1.0))
        testmodf('modf(-1.5)', math.modf(-1.5), (-0.5, -1.0))

        self.assertEqual(math.modf(INF), (0.0, INF))
        self.assertEqual(math.modf(NINF), (-0.0, NINF))

        modf_nan = math.modf(NAN)
        self.assertTrue(math.isnan(modf_nan[0]))
        self.assertTrue(math.isnan(modf_nan[1]))

    def testPow(self):
        # [FUNCTION DOC] Tests math.pow against the C99 special-value table: simple values,
        # OverflowError, results for bases 0, -0, +/-inf, -1 and 1 against many exponents,
        # exponent zero always giving 1, ValueError for negative base with non-integer exponent
        # or zero base with negative exponent, limits for base magnitude relative to 1 with
        # infinite exponents, and negative bases with integer exponents.
        self.assertRaises(TypeError, math.pow)
        self.ftest('pow(0,1)', math.pow(0,1), 0)
        self.ftest('pow(1,0)', math.pow(1,0), 1)
        self.ftest('pow(2,1)', math.pow(2,1), 2)
        self.ftest('pow(2,-1)', math.pow(2,-1), 0.5)
        self.assertEqual(math.pow(INF, 1), INF)
        self.assertEqual(math.pow(NINF, 1), NINF)
        self.assertEqual((math.pow(1, INF)), 1.)
        self.assertEqual((math.pow(1, NINF)), 1.)
        self.assertTrue(math.isnan(math.pow(NAN, 1)))
        self.assertTrue(math.isnan(math.pow(2, NAN)))
        self.assertTrue(math.isnan(math.pow(0, NAN)))
        self.assertEqual(math.pow(1, NAN), 1)
        self.assertRaises(OverflowError, math.pow, 1e+100, 1e+100)

        # pow(0., x)
        self.assertEqual(math.pow(0., INF), 0.)
        self.assertEqual(math.pow(0., 3.), 0.)
        self.assertEqual(math.pow(0., 2.3), 0.)
        self.assertEqual(math.pow(0., 2.), 0.)
        self.assertEqual(math.pow(0., 0.), 1.)
        self.assertEqual(math.pow(0., -0.), 1.)
        self.assertRaises(ValueError, math.pow, 0., -2.)
        self.assertRaises(ValueError, math.pow, 0., -2.3)
        self.assertRaises(ValueError, math.pow, 0., -3.)
        self.assertEqual(math.pow(0., NINF), INF)
        self.assertTrue(math.isnan(math.pow(0., NAN)))

        # pow(INF, x)
        self.assertEqual(math.pow(INF, INF), INF)
        self.assertEqual(math.pow(INF, 3.), INF)
        self.assertEqual(math.pow(INF, 2.3), INF)
        self.assertEqual(math.pow(INF, 2.), INF)
        self.assertEqual(math.pow(INF, 0.), 1.)
        self.assertEqual(math.pow(INF, -0.), 1.)
        self.assertEqual(math.pow(INF, -2.), 0.)
        self.assertEqual(math.pow(INF, -2.3), 0.)
        self.assertEqual(math.pow(INF, -3.), 0.)
        self.assertEqual(math.pow(INF, NINF), 0.)
        self.assertTrue(math.isnan(math.pow(INF, NAN)))

        # pow(-0., x)
        self.assertEqual(math.pow(-0., INF), 0.)
        self.assertEqual(math.pow(-0., 3.), -0.)
        self.assertEqual(math.pow(-0., 2.3), 0.)
        self.assertEqual(math.pow(-0., 2.), 0.)
        self.assertEqual(math.pow(-0., 0.), 1.)
        self.assertEqual(math.pow(-0., -0.), 1.)
        self.assertRaises(ValueError, math.pow, -0., -2.)
        self.assertRaises(ValueError, math.pow, -0., -2.3)
        self.assertRaises(ValueError, math.pow, -0., -3.)
        self.assertEqual(math.pow(-0., NINF), INF)
        self.assertTrue(math.isnan(math.pow(-0., NAN)))

        # pow(NINF, x)
        self.assertEqual(math.pow(NINF, INF), INF)
        self.assertEqual(math.pow(NINF, 3.), NINF)
        self.assertEqual(math.pow(NINF, 2.3), INF)
        self.assertEqual(math.pow(NINF, 2.), INF)
        self.assertEqual(math.pow(NINF, 0.), 1.)
        self.assertEqual(math.pow(NINF, -0.), 1.)
        self.assertEqual(math.pow(NINF, -2.), 0.)
        self.assertEqual(math.pow(NINF, -2.3), 0.)
        self.assertEqual(math.pow(NINF, -3.), -0.)
        self.assertEqual(math.pow(NINF, NINF), 0.)
        self.assertTrue(math.isnan(math.pow(NINF, NAN)))

        # pow(-1, x)
        self.assertEqual(math.pow(-1., INF), 1.)
        self.assertEqual(math.pow(-1., 3.), -1.)
        self.assertRaises(ValueError, math.pow, -1., 2.3)
        self.assertEqual(math.pow(-1., 2.), 1.)
        self.assertEqual(math.pow(-1., 0.), 1.)
        self.assertEqual(math.pow(-1., -0.), 1.)
        self.assertEqual(math.pow(-1., -2.), 1.)
        self.assertRaises(ValueError, math.pow, -1., -2.3)
        self.assertEqual(math.pow(-1., -3.), -1.)
        self.assertEqual(math.pow(-1., NINF), 1.)
        self.assertTrue(math.isnan(math.pow(-1., NAN)))

        # pow(1, x)
        self.assertEqual(math.pow(1., INF), 1.)
        self.assertEqual(math.pow(1., 3.), 1.)
        self.assertEqual(math.pow(1., 2.3), 1.)
        self.assertEqual(math.pow(1., 2.), 1.)
        self.assertEqual(math.pow(1., 0.), 1.)
        self.assertEqual(math.pow(1., -0.), 1.)
        self.assertEqual(math.pow(1., -2.), 1.)
        self.assertEqual(math.pow(1., -2.3), 1.)
        self.assertEqual(math.pow(1., -3.), 1.)
        self.assertEqual(math.pow(1., NINF), 1.)
        self.assertEqual(math.pow(1., NAN), 1.)

        # pow(x, 0) should be 1 for any x
        self.assertEqual(math.pow(2.3, 0.), 1.)
        self.assertEqual(math.pow(-2.3, 0.), 1.)
        self.assertEqual(math.pow(NAN, 0.), 1.)
        self.assertEqual(math.pow(2.3, -0.), 1.)
        self.assertEqual(math.pow(-2.3, -0.), 1.)
        self.assertEqual(math.pow(NAN, -0.), 1.)

        # pow(x, y) is invalid if x is negative and y is not integral
        self.assertRaises(ValueError, math.pow, -1., 2.3)
        self.assertRaises(ValueError, math.pow, -15., -3.1)

        # pow(x, NINF)
        self.assertEqual(math.pow(1.9, NINF), 0.)
        self.assertEqual(math.pow(1.1, NINF), 0.)
        self.assertEqual(math.pow(0.9, NINF), INF)
        self.assertEqual(math.pow(0.1, NINF), INF)
        self.assertEqual(math.pow(-0.1, NINF), INF)
        self.assertEqual(math.pow(-0.9, NINF), INF)
        self.assertEqual(math.pow(-1.1, NINF), 0.)
        self.assertEqual(math.pow(-1.9, NINF), 0.)

        # pow(x, INF)
        self.assertEqual(math.pow(1.9, INF), INF)
        self.assertEqual(math.pow(1.1, INF), INF)
        self.assertEqual(math.pow(0.9, INF), 0.)
        self.assertEqual(math.pow(0.1, INF), 0.)
        self.assertEqual(math.pow(-0.1, INF), 0.)
        self.assertEqual(math.pow(-0.9, INF), 0.)
        self.assertEqual(math.pow(-1.1, INF), INF)
        self.assertEqual(math.pow(-1.9, INF), INF)

        # pow(x, y) should work for x negative, y an integer
        self.ftest('(-2.)**3.', math.pow(-2.0, 3.0), -8.0)
        self.ftest('(-2.)**2.', math.pow(-2.0, 2.0), 4.0)
        self.ftest('(-2.)**1.', math.pow(-2.0, 1.0), -2.0)
        self.ftest('(-2.)**0.', math.pow(-2.0, 0.0), 1.0)
        self.ftest('(-2.)**-0.', math.pow(-2.0, -0.0), 1.0)
        self.ftest('(-2.)**-1.', math.pow(-2.0, -1.0), -0.5)
        self.ftest('(-2.)**-2.', math.pow(-2.0, -2.0), 0.25)
        self.ftest('(-2.)**-3.', math.pow(-2.0, -3.0), -0.125)
        self.assertRaises(ValueError, math.pow, -2.0, -0.5)
        self.assertRaises(ValueError, math.pow, -2.0, 0.5)

        # the following tests have been commented out since they don't
        # really belong here:  the implementation of ** for floats is
        # independent of the implementation of math.pow
        #self.assertEqual(1**NAN, 1)
        #self.assertEqual(1**INF, 1)
        #self.assertEqual(1**NINF, 1)
        #self.assertEqual(1**0, 1)
        #self.assertEqual(1.**NAN, 1)
        #self.assertEqual(1.**INF, 1)
        #self.assertEqual(1.**NINF, 1)
        #self.assertEqual(1.**0, 1)

    def testRadians(self):
        # [FUNCTION DOC] Tests math.radians at 180, 90, -45 and 0 and TypeError
        # without an argument.
        self.assertRaises(TypeError, math.radians)
        self.ftest('radians(180)', math.radians(180), math.pi)
        self.ftest('radians(90)', math.radians(90), math.pi/2)
        self.ftest('radians(-45)', math.radians(-45), -math.pi/4)
        self.ftest('radians(0)', math.radians(0), 0)

    @slowTest
    @requires_IEEE_754
    def testRemainder(self):
        # [FUNCTION DOC] Tests math.remainder (IEEE 754 only, slow): validates a table of
        # hex/decimal triples against the spec (|r| <= |y|/2, x - r a multiple of y, ties to
        # even) and bit-exact results via hex strings; checks a tiny subnormal modulus over
        # many multiples; NaN propagation; infinite modulus returns x; ValueError for zero
        # modulus or infinite dividend.
        from fractions import Fraction

        def validate_spec(x, y, r):
            # [FUNCTION DOC] HELPER: Checks with exact Fractions that r is a valid IEEE
            # remainder of x by y: |r| is at most half of |y|, (x - r)/y is an integer, and that
            # integer is even when |r| is exactly half of |y|. Assumes finite values and nonzero
            # y; failures via test assertions.
            """
            Check that r matches remainder(x, y) according to the IEEE 754
            specification. Assumes that x, y and r are finite and y is nonzero.
            """
            fx, fy, fr = Fraction(x), Fraction(y), Fraction(r)
            # r should not exceed y/2 in absolute value
            self.assertLessEqual(abs(fr), abs(fy/2))
            # x - r should be an exact integer multiple of y
            n = (fx - fr) / fy
            self.assertEqual(n, int(n))
            if abs(fr) == abs(fy/2):
                # If |r| == |y/2|, n should be even.
                self.assertEqual(n/2, int(n/2))

        # triples (x, y, remainder(x, y)) in hexadecimal form.
        testcases = [
            # Remainders modulo 1, showing the ties-to-even behaviour.
            '-4.0 1 -0.0',
            '-3.8 1  0.8',
            '-3.0 1 -0.0',
            '-2.8 1 -0.8',
            '-2.0 1 -0.0',
            '-1.8 1  0.8',
            '-1.0 1 -0.0',
            '-0.8 1 -0.8',
            '-0.0 1 -0.0',
            ' 0.0 1  0.0',
            ' 0.8 1  0.8',
            ' 1.0 1  0.0',
            ' 1.8 1 -0.8',
            ' 2.0 1  0.0',
            ' 2.8 1  0.8',
            ' 3.0 1  0.0',
            ' 3.8 1 -0.8',
            ' 4.0 1  0.0',

            # Reductions modulo 2*pi
            '0x0.0p+0 0x1.921fb54442d18p+2 0x0.0p+0',
            '0x1.921fb54442d18p+0 0x1.921fb54442d18p+2  0x1.921fb54442d18p+0',
            '0x1.921fb54442d17p+1 0x1.921fb54442d18p+2  0x1.921fb54442d17p+1',
            '0x1.921fb54442d18p+1 0x1.921fb54442d18p+2  0x1.921fb54442d18p+1',
            '0x1.921fb54442d19p+1 0x1.921fb54442d18p+2 -0x1.921fb54442d17p+1',
            '0x1.921fb54442d17p+2 0x1.921fb54442d18p+2 -0x0.0000000000001p+2',
            '0x1.921fb54442d18p+2 0x1.921fb54442d18p+2  0x0p0',
            '0x1.921fb54442d19p+2 0x1.921fb54442d18p+2  0x0.0000000000001p+2',
            '0x1.2d97c7f3321d1p+3 0x1.921fb54442d18p+2  0x1.921fb54442d14p+1',
            '0x1.2d97c7f3321d2p+3 0x1.921fb54442d18p+2 -0x1.921fb54442d18p+1',
            '0x1.2d97c7f3321d3p+3 0x1.921fb54442d18p+2 -0x1.921fb54442d14p+1',
            '0x1.921fb54442d17p+3 0x1.921fb54442d18p+2 -0x0.0000000000001p+3',
            '0x1.921fb54442d18p+3 0x1.921fb54442d18p+2  0x0p0',
            '0x1.921fb54442d19p+3 0x1.921fb54442d18p+2  0x0.0000000000001p+3',
            '0x1.f6a7a2955385dp+3 0x1.921fb54442d18p+2  0x1.921fb54442d14p+1',
            '0x1.f6a7a2955385ep+3 0x1.921fb54442d18p+2  0x1.921fb54442d18p+1',
            '0x1.f6a7a2955385fp+3 0x1.921fb54442d18p+2 -0x1.921fb54442d14p+1',
            '0x1.1475cc9eedf00p+5 0x1.921fb54442d18p+2  0x1.921fb54442d10p+1',
            '0x1.1475cc9eedf01p+5 0x1.921fb54442d18p+2 -0x1.921fb54442d10p+1',

            # Symmetry with respect to signs.
            ' 1  0.c  0.4',
            '-1  0.c -0.4',
            ' 1 -0.c  0.4',
            '-1 -0.c -0.4',
            ' 1.4  0.c -0.4',
            '-1.4  0.c  0.4',
            ' 1.4 -0.c -0.4',
            '-1.4 -0.c  0.4',

            # Huge modulus, to check that the underlying algorithm doesn't
            # rely on 2.0 * modulus being representable.
            '0x1.dp+1023 0x1.4p+1023  0x0.9p+1023',
            '0x1.ep+1023 0x1.4p+1023 -0x0.ap+1023',
            '0x1.fp+1023 0x1.4p+1023 -0x0.9p+1023',
        ]

        for case in testcases:
            with self.subTest(case=case):
                x_hex, y_hex, expected_hex = case.split()
                x = float.fromhex(x_hex)
                y = float.fromhex(y_hex)
                expected = float.fromhex(expected_hex)
                validate_spec(x, y, expected)
                actual = math.remainder(x, y)
                # Cheap way of checking that the floats are
                # as identical as we need them to be.
                self.assertEqual(actual.hex(), expected.hex())

        # Test tiny subnormal modulus: there's potential for
        # getting the implementation wrong here (for example,
        # by assuming that modulus/2 is exactly representable).
        tiny = float.fromhex('1p-1074')  # min +ve subnormal
        for n in range(-25, 25):
            if n == 0:
                continue
            y = n * tiny
            for m in range(100):
                x = m * tiny
                actual = math.remainder(x, y)
                validate_spec(x, y, actual)
                actual = math.remainder(-x, y)
                validate_spec(-x, y, actual)

        # Special values.
        # NaNs should propagate as usual.
        for value in [NAN, 0.0, -0.0, 2.0, -2.3, NINF, INF]:
            self.assertIsNaN(math.remainder(NAN, value))
            self.assertIsNaN(math.remainder(value, NAN))

        # remainder(x, inf) is x, for non-nan non-infinite x.
        for value in [-2.3, -0.0, 0.0, 2.3]:
            self.assertEqual(math.remainder(value, INF), value)
            self.assertEqual(math.remainder(value, NINF), value)

        # remainder(x, 0) and remainder(infinity, x) for non-NaN x are invalid
        # operations according to IEEE 754-2008 7.2(f), and should raise.
        for value in [NINF, -2.3, -0.0, 0.0, 2.3, INF]:
            with self.assertRaises(ValueError):
                math.remainder(INF, value)
            with self.assertRaises(ValueError):
                math.remainder(NINF, value)
            with self.assertRaises(ValueError):
                math.remainder(value, 0.0)
            with self.assertRaises(ValueError):
                math.remainder(value, -0.0)

    def testSin(self):
        # [FUNCTION DOC] Tests math.sin: values at 0 and plus/minus pi/2, infinite input gives
        # NaN or ValueError, NaN propagates, TypeError without argument.
        self.assertRaises(TypeError, math.sin)
        self.ftest('sin(0)', math.sin(0), 0)
        self.ftest('sin(pi/2)', math.sin(math.pi/2), 1)
        self.ftest('sin(-pi/2)', math.sin(-math.pi/2), -1)
        try:
            self.assertTrue(math.isnan(math.sin(INF)))
            self.assertTrue(math.isnan(math.sin(NINF)))
        except ValueError:
            self.assertRaises(ValueError, math.sin, INF)
            self.assertRaises(ValueError, math.sin, NINF)
        self.assertTrue(math.isnan(math.sin(NAN)))

    def testSinh(self):
        # [FUNCTION DOC] Tests math.sinh: value at 0, identity with cosh, odd symmetry,
        # infinities map to themselves, NaN propagates.
        self.assertRaises(TypeError, math.sinh)
        self.ftest('sinh(0)', math.sinh(0), 0)
        self.ftest('sinh(1)**2-cosh(1)**2', math.sinh(1)**2-math.cosh(1)**2, -1)
        self.ftest('sinh(1)+sinh(-1)', math.sinh(1)+math.sinh(-1), 0)
        self.assertEqual(math.sinh(INF), INF)
        self.assertEqual(math.sinh(NINF), NINF)
        self.assertTrue(math.isnan(math.sinh(NAN)))

    def testSqrt(self):
        # [FUNCTION DOC] Tests math.sqrt: known values for ints and floats, +inf maps to itself,
        # ValueError for negative input or -inf, NaN propagates.
        self.assertRaises(TypeError, math.sqrt)
        self.ftest('sqrt(0)', math.sqrt(0), 0)
        self.ftest('sqrt(0)', math.sqrt(0.0), 0.0)
        self.ftest('sqrt(2.5)', math.sqrt(2.5), 1.5811388300841898)
        self.ftest('sqrt(0.25)', math.sqrt(0.25), 0.5)
        self.ftest('sqrt(25.25)', math.sqrt(25.25), 5.024937810560445)
        self.ftest('sqrt(1)', math.sqrt(1), 1)
        self.ftest('sqrt(4)', math.sqrt(4), 2)
        self.assertEqual(math.sqrt(INF), INF)
        self.assertRaises(ValueError, math.sqrt, -1)
        self.assertRaises(ValueError, math.sqrt, NINF)
        self.assertTrue(math.isnan(math.sqrt(NAN)))

    def testTan(self):
        # [FUNCTION DOC] Tests math.tan: values at 0 and plus/minus pi/4, infinite input gives
        # NaN or ValueError, NaN propagates.
        self.assertRaises(TypeError, math.tan)
        self.ftest('tan(0)', math.tan(0), 0)
        self.ftest('tan(pi/4)', math.tan(math.pi/4), 1)
        self.ftest('tan(-pi/4)', math.tan(-math.pi/4), -1)
        try:
            self.assertTrue(math.isnan(math.tan(INF)))
            self.assertTrue(math.isnan(math.tan(NINF)))
        except ValueError:
            self.assertRaises(ValueError, math.tan, INF)
            self.assertRaises(ValueError, math.tan, NINF)
        self.assertTrue(math.isnan(math.tan(NAN)))

    def testTanh(self):
        # [FUNCTION DOC] Tests math.tanh: value at 0, odd symmetry within one ulp, limits at
        # plus and minus infinity are plus and minus 1, NaN propagates.
        self.assertRaises(TypeError, math.tanh)
        self.ftest('tanh(0)', math.tanh(0), 0)
        self.ftest('tanh(1)+tanh(-1)', math.tanh(1)+math.tanh(-1), 0,
                   abs_tol=math.ulp(1))
        self.ftest('tanh(inf)', math.tanh(INF), 1)
        self.ftest('tanh(-inf)', math.tanh(NINF), -1)
        self.assertTrue(math.isnan(math.tanh(NAN)))

    @requires_IEEE_754
    def testTanhSign(self):
        # [FUNCTION DOC] Tests that math.tanh of negative zero is negative zero, with the sign
        # verified through copysign (IEEE 754 only).
        # check that tanh(-0.) == -0. on IEEE 754 systems
        self.assertEqual(math.tanh(-0.), -0.)
        self.assertEqual(math.copysign(1., math.tanh(-0.)),
                         math.copysign(1., -0.))

    def test_trunc(self):
        # [FUNCTION DOC] Tests math.trunc: int results truncating toward zero for ints and
        # floats. Checks the trunc special method protocol with local fixtures: custom method
        # used (23), float subclass override used, float-convertible object without the method
        # raises TypeError, missing method TypeError, raising descriptor propagates ValueError,
        # wrong arity TypeError.
        self.assertEqual(math.trunc(1), 1)
        self.assertEqual(math.trunc(-1), -1)
        self.assertEqual(type(math.trunc(1)), int)
        self.assertEqual(type(math.trunc(1.5)), int)
        self.assertEqual(math.trunc(1.5), 1)
        self.assertEqual(math.trunc(-1.5), -1)
        self.assertEqual(math.trunc(1.999999), 1)
        self.assertEqual(math.trunc(-1.999999), -1)
        self.assertEqual(math.trunc(-0.999999), -0)
        self.assertEqual(math.trunc(-100.999), -100)

        class TestTrunc:
            # [CLASS DOC] HELPER: Local fixture defining the trunc special method, returning 23.
            def __trunc__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 23.
                return 23
        class FloatTrunc(float):
            # [CLASS DOC] HELPER: Local float subclass whose trunc special method returns 23.
            def __trunc__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 23.
                return 23
        class TestNoTrunc:
            # [CLASS DOC] HELPER: Local fixture with no trunc support; math.trunc
            # must raise TypeError.
            pass
        class TestBadTrunc:
            # [CLASS DOC] HELPER: Local fixture whose trunc attribute is a descriptor
            # that raises ValueError.
            __trunc__ = BadDescr()

        self.assertEqual(math.trunc(TestTrunc()), 23)
        self.assertEqual(math.trunc(FloatTrunc()), 23)

        self.assertRaises(TypeError, math.trunc)
        self.assertRaises(TypeError, math.trunc, 1, 2)
        self.assertRaises(TypeError, math.trunc, FloatLike(23.5))
        self.assertRaises(TypeError, math.trunc, TestNoTrunc())
        self.assertRaises(ValueError, math.trunc, TestBadTrunc())

    def testIsfinite(self):
        # [FUNCTION DOC] Tests math.isfinite: true for plus and minus zero and finite values,
        # false for NaN and both infinities.
        self.assertTrue(math.isfinite(0.0))
        self.assertTrue(math.isfinite(-0.0))
        self.assertTrue(math.isfinite(1.0))
        self.assertTrue(math.isfinite(-1.0))
        self.assertFalse(math.isfinite(float("nan")))
        self.assertFalse(math.isfinite(float("inf")))
        self.assertFalse(math.isfinite(float("-inf")))

    def testIsnan(self):
        # [FUNCTION DOC] Tests math.isnan: true for NaN (either sign) and for infinity times
        # zero, false for infinity and finite values.
        self.assertTrue(math.isnan(float("nan")))
        self.assertTrue(math.isnan(float("-nan")))
        self.assertTrue(math.isnan(float("inf") * 0.))
        self.assertFalse(math.isnan(float("inf")))
        self.assertFalse(math.isnan(0.))
        self.assertFalse(math.isnan(1.))

    def testIsinf(self):
        # [FUNCTION DOC] Tests math.isinf: true for both infinities and for overflowing
        # literals, false for NaN and finite values.
        self.assertTrue(math.isinf(float("inf")))
        self.assertTrue(math.isinf(float("-inf")))
        self.assertTrue(math.isinf(1E400))
        self.assertTrue(math.isinf(-1E400))
        self.assertFalse(math.isinf(float("nan")))
        self.assertFalse(math.isinf(0.))
        self.assertFalse(math.isinf(1.))

    def test_nan_constant(self):
        # [FUNCTION DOC] Tests that math.nan is a NaN with positive sign bit
        # (checked via copysign).
        # `math.nan` must be a quiet NaN with positive sign bit
        self.assertTrue(math.isnan(math.nan))
        self.assertEqual(math.copysign(1., math.nan), 1.)

    def test_inf_constant(self):
        # [FUNCTION DOC] Tests that math.inf is infinite, positive, and equal to float inf; its
        # negation equals float -inf.
        self.assertTrue(math.isinf(math.inf))
        self.assertGreater(math.inf, 0.0)
        self.assertEqual(math.inf, float("inf"))
        self.assertEqual(-math.inf, float("-inf"))

    # RED_FLAG 16-Oct-2000 Tim
    # While 2.0 is more consistent about exceptions than previous releases, it
    # still fails this part of the test on some platforms.  For now, we only
    # *run* test_exceptions() in verbose mode, so that this isn't normally
    # tested.
    @unittest.skipUnless(verbose, 'requires verbose mode')
    def test_exceptions(self):
        # [FUNCTION DOC] Runs only in verbose mode (known to fail on some platforms):
        # an underflowing exp must return 0 without raising, an overflowing exp must
        # raise OverflowError, and sqrt of -1.0 must raise ValueError. Fails the test
        # with a message otherwise.
        try:
            x = math.exp(-1000000000)
        except:
            # mathmodule.c is failing to weed out underflows from libm, or
            # we've got an fp format with huge dynamic range
            self.fail("underflowing exp() should not have raised "
                        "an exception")
        if x != 0:
            self.fail("underflowing exp() should have returned 0")

        # If this fails, probably using a strict IEEE-754 conforming libm, and x
        # is +Inf afterwards.  But Python wants overflows detected by default.
        try:
            x = math.exp(1000000000)
        except OverflowError:
            pass
        else:
            self.fail("overflowing exp() didn't trigger OverflowError")

        # If this fails, it could be a puzzle.  One odd possibility is that
        # mathmodule.c's macros are getting confused while comparing
        # Inf (HUGE_VAL) to a NaN, and artificially setting errno to ERANGE
        # as a result (and so raising OverflowError instead).
        try:
            x = math.sqrt(-1.0)
        except ValueError:
            pass
        else:
            self.fail("sqrt(-1) didn't raise ValueError")

    @requires_IEEE_754
    def test_testfile(self):
        # [FUNCTION DOC] Table-driven test (IEEE 754 only) using the cmath_testcases file: for
        # each case whose inputs and outputs are purely real, skips rect and polar, optionally
        # skips one case on very old macOS, looks up the function on math, maps invalid/divide-
        # by-zero flags to expected ValueError and overflow to OverflowError, fixes the zero
        # sign for a few sqrt cases, and compares with 5 ulp tolerance using result_check.
        # Collects every failure and fails once with the full list.
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

        fail_fmt = "{}: {}({!r}): {}"

        failures = []
        for id, fn, ar, ai, er, ei, flags in parse_testfile(test_file):
            # Skip if either the input or result is complex
            if ai != 0.0 or ei != 0.0:
                continue
            if fn in ['rect', 'polar']:
                # no real versions of rect, polar
                continue
            # Skip certain tests on OS X 10.4.
            if osx_version is not None and osx_version < (10, 5):
                if id in SKIP_ON_TIGER:
                    continue

            func = getattr(math, fn)

            if 'invalid' in flags or 'divide-by-zero' in flags:
                er = 'ValueError'
            elif 'overflow' in flags:
                er = 'OverflowError'

            try:
                result = func(ar)
            except ValueError:
                result = 'ValueError'
            except OverflowError:
                result = 'OverflowError'

            # C99+ says for math.h's sqrt: If the argument is +∞ or ±0, it is
            # returned, unmodified.  On another hand, for csqrt: If z is ±0+0i,
            # the result is +0+0i.  Lets correct zero sign of er to follow
            # first convention.
            if id in ['sqrt0002', 'sqrt0003', 'sqrt1001', 'sqrt1023']:
                er = math.copysign(er, ar)

            # Default tolerances
            ulp_tol, abs_tol = 5, 0.0

            failure = result_check(er, result, ulp_tol, abs_tol)
            if failure is None:
                continue

            msg = fail_fmt.format(id, fn, ar, failure)
            failures.append(msg)

        if failures:
            self.fail('Failures in test_testfile:\n  ' +
                      '\n  '.join(failures))

    @requires_IEEE_754
    def test_mtestfile(self):
        # [FUNCTION DOC] Table-driven test (IEEE 754 only) using the math_testcases file: calls
        # each named math function on the argument, maps raised ValueError/OverflowError to the
        # expected names via flags, uses default tolerance of 5 ulps with wider tolerances for
        # gamma (20 ulps), lgamma (absolute 1e-15) and erfc for non-negative arguments
        # (10/100/1000 ulps by range), and fails once with all collected mismatches.
        fail_fmt = "{}: {}({!r}): {}"

        failures = []
        for id, fn, arg, expected, flags in parse_mtestfile(math_testcases):
            func = getattr(math, fn)

            if 'invalid' in flags or 'divide-by-zero' in flags:
                expected = 'ValueError'
            elif 'overflow' in flags:
                expected = 'OverflowError'

            try:
                got = func(arg)
            except ValueError:
                got = 'ValueError'
            except OverflowError:
                got = 'OverflowError'

            # Default tolerances
            ulp_tol, abs_tol = 5, 0.0

            # Exceptions to the defaults
            if fn == 'gamma':
                # Experimental results on one platform gave
                # an accuracy of <= 10 ulps across the entire float
                # domain. We weaken that to require 20 ulp accuracy.
                ulp_tol = 20

            elif fn == 'lgamma':
                # we use a weaker accuracy test for lgamma;
                # lgamma only achieves an absolute error of
                # a few multiples of the machine accuracy, in
                # general.
                abs_tol = 1e-15

            elif fn == 'erfc' and arg >= 0.0:
                # erfc has less-than-ideal accuracy for large
                # arguments (x ~ 25 or so), mainly due to the
                # error involved in computing exp(-x*x).
                #
                # Observed between CPython and mpmath at 25 dp:
                #       x <  0 : err <= 2 ulp
                #  0 <= x <  1 : err <= 10 ulp
                #  1 <= x < 10 : err <= 100 ulp
                # 10 <= x < 20 : err <= 300 ulp
                # 20 <= x      : < 600 ulp
                #
                if arg < 1.0:
                    ulp_tol = 10
                elif arg < 10.0:
                    ulp_tol = 100
                else:
                    ulp_tol = 1000

            failure = result_check(expected, got, ulp_tol, abs_tol)
            if failure is None:
                continue

            msg = fail_fmt.format(id, fn, arg, failure)
            failures.append(msg)

        if failures:
            self.fail('Failures in test_mtestfile:\n  ' +
                      '\n  '.join(failures))

    def test_prod(self):
        # [FUNCTION DOC] Tests math.prod: empty and start values, mixed numeric types,
        # integers/floats/Fractions, propagation of multiplication errors, fast-path
        # overflow, TypeError for non-iterables or non-multiplicable elements, odd uses with
        # string/list start, start being keyword-only, zero handling, big ints and floats
        # against a naive loop, NaN and infinity combinations, and result type preservation
        # (int, float, Decimal).
        from fractions import Fraction as F

        prod = math.prod
        self.assertEqual(prod([]), 1)
        self.assertEqual(prod([], start=5), 5)
        self.assertEqual(prod(list(range(2,8))), 5040)
        self.assertEqual(prod(iter(list(range(2,8)))), 5040)
        self.assertEqual(prod(range(1, 10), start=10), 3628800)

        self.assertEqual(prod([1, 2, 3, 4, 5]), 120)
        self.assertEqual(prod([1.0, 2.0, 3.0, 4.0, 5.0]), 120.0)
        self.assertEqual(prod([1, 2, 3, 4.0, 5.0]), 120.0)
        self.assertEqual(prod([1.0, 2.0, 3.0, 4, 5]), 120.0)
        self.assertEqual(prod([1., F(3, 2)]), 1.5)

        # Error in multiplication
        class BadMultiply:
            # [CLASS DOC] HELPER: Local fixture whose reflected
            # multiplication raises RuntimeError.
            def __rmul__(self, other):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError
        with self.assertRaises(RuntimeError):
            prod([10., BadMultiply()])

        # Test overflow in fast-path for integers
        self.assertEqual(prod([1, 1, 2**32, 1, 1]), 2**32)
        # Test overflow in fast-path for floats
        self.assertEqual(prod([1.0, 1.0, 2**32, 1, 1]), float(2**32))

        self.assertRaises(TypeError, prod)
        self.assertRaises(TypeError, prod, 42)
        self.assertRaises(TypeError, prod, ['a', 'b', 'c'])
        self.assertRaises(TypeError, prod, ['a', 'b', 'c'], start='')
        self.assertRaises(TypeError, prod, [b'a', b'c'], start=b'')
        values = [bytearray(b'a'), bytearray(b'b')]
        self.assertRaises(TypeError, prod, values, start=bytearray(b''))
        self.assertRaises(TypeError, prod, [[1], [2], [3]])
        self.assertRaises(TypeError, prod, [{2:3}])
        self.assertRaises(TypeError, prod, [{2:3}]*2, start={2:3})
        self.assertRaises(TypeError, prod, [[1], [2], [3]], start=[])

        # Some odd cases
        self.assertEqual(prod([2, 3], start='ab'), 'abababababab')
        self.assertEqual(prod([2, 3], start=[1, 2]), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        self.assertEqual(prod([], start={2: 3}), {2:3})

        with self.assertRaises(TypeError):
            prod([10, 20], 1)     # start is a keyword-only argument

        self.assertEqual(prod([0, 1, 2, 3]), 0)
        self.assertEqual(prod([1, 0, 2, 3]), 0)
        self.assertEqual(prod([1, 2, 3, 0]), 0)

        def _naive_prod(iterable, start=1):
            # [FUNCTION DOC] HELPER: Reference product: multiplies start by each element in
            # order and returns it.
            for elem in iterable:
                start *= elem
            return start

        # Big integers

        iterable = range(1, 10000)
        self.assertEqual(prod(iterable), _naive_prod(iterable))
        iterable = range(-10000, -1)
        self.assertEqual(prod(iterable), _naive_prod(iterable))
        iterable = range(-1000, 1000)
        self.assertEqual(prod(iterable), 0)

        # Big floats

        iterable = [float(x) for x in range(1, 1000)]
        self.assertEqual(prod(iterable), _naive_prod(iterable))
        iterable = [float(x) for x in range(-1000, -1)]
        self.assertEqual(prod(iterable), _naive_prod(iterable))
        iterable = [float(x) for x in range(-1000, 1000)]
        self.assertIsNaN(prod(iterable))

        # Float tests

        self.assertIsNaN(prod([1, 2, 3, float("nan"), 2, 3]))
        self.assertIsNaN(prod([1, 0, float("nan"), 2, 3]))
        self.assertIsNaN(prod([1, float("nan"), 0, 3]))
        self.assertIsNaN(prod([1, float("inf"), float("nan"),3]))
        self.assertIsNaN(prod([1, float("-inf"), float("nan"),3]))
        self.assertIsNaN(prod([1, float("nan"), float("inf"),3]))
        self.assertIsNaN(prod([1, float("nan"), float("-inf"),3]))

        self.assertEqual(prod([1, 2, 3, float('inf'),-3,4]), float('-inf'))
        self.assertEqual(prod([1, 2, 3, float('-inf'),-3,4]), float('inf'))

        self.assertIsNaN(prod([1,2,0,float('inf'), -3, 4]))
        self.assertIsNaN(prod([1,2,0,float('-inf'), -3, 4]))
        self.assertIsNaN(prod([1, 2, 3, float('inf'), -3, 0, 3]))
        self.assertIsNaN(prod([1, 2, 3, float('-inf'), -3, 0, 2]))

        # Type preservation

        self.assertEqual(type(prod([1, 2, 3, 4, 5, 6])), int)
        self.assertEqual(type(prod([1, 2.0, 3, 4, 5, 6])), float)
        self.assertEqual(type(prod(range(1, 10000))), int)
        self.assertEqual(type(prod(range(1, 10000), start=1.0)), float)
        self.assertEqual(type(prod([1, decimal.Decimal(2.0), 3, 4, 5, 6])),
                         decimal.Decimal)

    @skipIfTorchDynamo("Infinite loop")
    def testPerm(self):
        # [FUNCTION DOC] Tests math.perm: matches the factorial definition for many (n, k),
        # Pascal-like identity, corner cases, one-argument form, TypeError for non-integers or
        # wrong arity, ValueError for negative arguments, zero when k exceeds n, huge n,
        # OverflowError for enormous results on CPython, and int results for bool, int subclass
        # and index-protocol arguments. Skipped under Dynamo (infinite loop).
        perm = math.perm
        factorial = math.factorial
        # Test if factorial definition is satisfied
        for n in range(500):
            for k in (range(n + 1) if n < 100 else range(30) if n < 200 else range(10)):
                self.assertEqual(perm(n, k),
                                 factorial(n) // factorial(n - k))

        # Test for Pascal's identity
        for n in range(1, 100):
            for k in range(1, n):
                self.assertEqual(perm(n, k), perm(n - 1, k - 1) * k + perm(n - 1, k))

        # Test corner cases
        for n in range(1, 100):
            self.assertEqual(perm(n, 0), 1)
            self.assertEqual(perm(n, 1), n)
            self.assertEqual(perm(n, n), factorial(n))

        # Test one argument form
        for n in range(20):
            self.assertEqual(perm(n), factorial(n))
            self.assertEqual(perm(n, None), factorial(n))

        # Raises TypeError if any argument is non-integer or argument count is
        # not 1 or 2
        self.assertRaises(TypeError, perm, 10, 1.0)
        self.assertRaises(TypeError, perm, 10, decimal.Decimal(1.0))
        self.assertRaises(TypeError, perm, 10, "1")
        self.assertRaises(TypeError, perm, 10.0, 1)
        self.assertRaises(TypeError, perm, decimal.Decimal(10.0), 1)
        self.assertRaises(TypeError, perm, "10", 1)

        self.assertRaises(TypeError, perm)
        self.assertRaises(TypeError, perm, 10, 1, 3)
        self.assertRaises(TypeError, perm)

        # Raises Value error if not k or n are negative numbers
        self.assertRaises(ValueError, perm, -1, 1)
        self.assertRaises(ValueError, perm, -2**1000, 1)
        self.assertRaises(ValueError, perm, 1, -1)
        self.assertRaises(ValueError, perm, 1, -2**1000)

        # Returns zero if k is greater than n
        self.assertEqual(perm(1, 2), 0)
        self.assertEqual(perm(1, 2**1000), 0)

        n = 2**1000
        self.assertEqual(perm(n, 0), 1)
        self.assertEqual(perm(n, 1), n)
        self.assertEqual(perm(n, 2), n * (n-1))
        if support.check_impl_detail(cpython=True):
            self.assertRaises(OverflowError, perm, n, n)

        for n, k in (True, True), (True, False), (False, False):
            self.assertEqual(perm(n, k), 1)
            self.assertIs(type(perm(n, k)), int)
        self.assertEqual(perm(IntSubclass(5), IntSubclass(2)), 20)
        self.assertEqual(perm(MyIndexable(5), MyIndexable(2)), 20)
        for k in range(3):
            self.assertIs(type(perm(IntSubclass(5), IntSubclass(k))), int)
            self.assertIs(type(perm(MyIndexable(5), MyIndexable(k))), int)

    @skipIfTorchDynamo("infinite loop")
    def testComb(self):
        # [FUNCTION DOC] Tests math.comb analogously to testPerm: factorial definition,
        # Pascal's identity, corner cases, symmetry, TypeError for non-integers or wrong
        # arity, ValueError for negatives, zero when k exceeds n, huge n, OverflowError on
        # CPython, and int result types for bool, int subclass and index-protocol arguments.
        # Skipped under Dynamo.
        comb = math.comb
        factorial = math.factorial
        # Test if factorial definition is satisfied
        for n in range(500):
            for k in (range(n + 1) if n < 100 else range(30) if n < 200 else range(10)):
                self.assertEqual(comb(n, k), factorial(n)
                    // (factorial(k) * factorial(n - k)))

        # Test for Pascal's identity
        for n in range(1, 100):
            for k in range(1, n):
                self.assertEqual(comb(n, k), comb(n - 1, k - 1) + comb(n - 1, k))

        # Test corner cases
        for n in range(100):
            self.assertEqual(comb(n, 0), 1)
            self.assertEqual(comb(n, n), 1)

        for n in range(1, 100):
            self.assertEqual(comb(n, 1), n)
            self.assertEqual(comb(n, n - 1), n)

        # Test Symmetry
        for n in range(100):
            for k in range(n // 2):
                self.assertEqual(comb(n, k), comb(n, n - k))

        # Raises TypeError if any argument is non-integer or argument count is
        # not 2
        self.assertRaises(TypeError, comb, 10, 1.0)
        self.assertRaises(TypeError, comb, 10, decimal.Decimal(1.0))
        self.assertRaises(TypeError, comb, 10, "1")
        self.assertRaises(TypeError, comb, 10.0, 1)
        self.assertRaises(TypeError, comb, decimal.Decimal(10.0), 1)
        self.assertRaises(TypeError, comb, "10", 1)

        self.assertRaises(TypeError, comb, 10)
        self.assertRaises(TypeError, comb, 10, 1, 3)
        self.assertRaises(TypeError, comb)

        # Raises Value error if not k or n are negative numbers
        self.assertRaises(ValueError, comb, -1, 1)
        self.assertRaises(ValueError, comb, -2**1000, 1)
        self.assertRaises(ValueError, comb, 1, -1)
        self.assertRaises(ValueError, comb, 1, -2**1000)

        # Returns zero if k is greater than n
        self.assertEqual(comb(1, 2), 0)
        self.assertEqual(comb(1, 2**1000), 0)

        n = 2**1000
        self.assertEqual(comb(n, 0), 1)
        self.assertEqual(comb(n, 1), n)
        self.assertEqual(comb(n, 2), n * (n-1) // 2)
        self.assertEqual(comb(n, n), 1)
        self.assertEqual(comb(n, n-1), n)
        self.assertEqual(comb(n, n-2), n * (n-1) // 2)
        if support.check_impl_detail(cpython=True):
            self.assertRaises(OverflowError, comb, n, n//2)

        for n, k in (True, True), (True, False), (False, False):
            self.assertEqual(comb(n, k), 1)
            self.assertIs(type(comb(n, k)), int)
        self.assertEqual(comb(IntSubclass(5), IntSubclass(2)), 10)
        self.assertEqual(comb(MyIndexable(5), MyIndexable(2)), 10)
        for k in range(3):
            self.assertIs(type(comb(IntSubclass(5), IntSubclass(k))), int)
            self.assertIs(type(comb(MyIndexable(5), MyIndexable(k))), int)

    @requires_IEEE_754
    def test_nextafter(self):
        # [FUNCTION DOC] Tests math.nextafter (IEEE 754 only): neighbors around large powers of
        # two, around 1.0 with the steps argument, equal arguments return the target with its
        # sign, around zero (subnormals and signed zero), around infinity, NaN propagation,
        # steps zero returns the start and negative steps raise ValueError.
        # around 2^52 and 2^63
        self.assertEqual(math.nextafter(4503599627370496.0, -INF),
                         4503599627370495.5)
        self.assertEqual(math.nextafter(4503599627370496.0, INF),
                         4503599627370497.0)
        self.assertEqual(math.nextafter(9223372036854775808.0, 0.0),
                         9223372036854774784.0)
        self.assertEqual(math.nextafter(-9223372036854775808.0, 0.0),
                         -9223372036854774784.0)

        # around 1.0
        self.assertEqual(math.nextafter(1.0, -INF),
                         float.fromhex('0x1.fffffffffffffp-1'))
        self.assertEqual(math.nextafter(1.0, INF),
                         float.fromhex('0x1.0000000000001p+0'))
        self.assertEqual(math.nextafter(1.0, -INF, steps=1),
                         float.fromhex('0x1.fffffffffffffp-1'))
        self.assertEqual(math.nextafter(1.0, INF, steps=1),
                         float.fromhex('0x1.0000000000001p+0'))
        self.assertEqual(math.nextafter(1.0, -INF, steps=3),
                         float.fromhex('0x1.ffffffffffffdp-1'))
        self.assertEqual(math.nextafter(1.0, INF, steps=3),
                         float.fromhex('0x1.0000000000003p+0'))

        # x == y: y is returned
        for steps in range(1, 5):
            self.assertEqual(math.nextafter(2.0, 2.0, steps=steps), 2.0)
            self.assertEqualSign(math.nextafter(-0.0, +0.0, steps=steps), +0.0)
            self.assertEqualSign(math.nextafter(+0.0, -0.0, steps=steps), -0.0)

        # around 0.0
        smallest_subnormal = sys.float_info.min * sys.float_info.epsilon
        self.assertEqual(math.nextafter(+0.0, INF), smallest_subnormal)
        self.assertEqual(math.nextafter(-0.0, INF), smallest_subnormal)
        self.assertEqual(math.nextafter(+0.0, -INF), -smallest_subnormal)
        self.assertEqual(math.nextafter(-0.0, -INF), -smallest_subnormal)
        self.assertEqualSign(math.nextafter(smallest_subnormal, +0.0), +0.0)
        self.assertEqualSign(math.nextafter(-smallest_subnormal, +0.0), -0.0)
        self.assertEqualSign(math.nextafter(smallest_subnormal, -0.0), +0.0)
        self.assertEqualSign(math.nextafter(-smallest_subnormal, -0.0), -0.0)

        # around infinity
        largest_normal = sys.float_info.max
        self.assertEqual(math.nextafter(INF, 0.0), largest_normal)
        self.assertEqual(math.nextafter(-INF, 0.0), -largest_normal)
        self.assertEqual(math.nextafter(largest_normal, INF), INF)
        self.assertEqual(math.nextafter(-largest_normal, -INF), -INF)

        # NaN
        self.assertIsNaN(math.nextafter(NAN, 1.0))
        self.assertIsNaN(math.nextafter(1.0, NAN))
        self.assertIsNaN(math.nextafter(NAN, NAN))

        self.assertEqual(1.0, math.nextafter(1.0, INF, steps=0))
        with self.assertRaises(ValueError):
            math.nextafter(1.0, INF, steps=-1)


    @unittest.skip("flaky test under torch dynamo")  # works on pytest and crashes on unittest
    @requires_IEEE_754
    def test_ulp(self):
        # [FUNCTION DOC] Tests math.ulp: ulp of 1.0 is epsilon, powers of two, smallest and
        # largest floats, infinity gives infinity, NaN gives NaN, and the value is symmetric
        # under negation. Currently skipped unconditionally because it is flaky under Dynamo
        # (crashes with unittest).
        self.assertEqual(math.ulp(1.0), sys.float_info.epsilon)
        # use int ** int rather than float ** int to not rely on pow() accuracy
        self.assertEqual(math.ulp(2 ** 52), 1.0)
        self.assertEqual(math.ulp(2 ** 53), 2.0)
        self.assertEqual(math.ulp(2 ** 64), 4096.0)

        # min and max
        self.assertEqual(math.ulp(0.0),
                         sys.float_info.min * sys.float_info.epsilon)
        self.assertEqual(math.ulp(FLOAT_MAX),
                         FLOAT_MAX - math.nextafter(FLOAT_MAX, -INF))

        # special cases
        self.assertEqual(math.ulp(INF), INF)
        self.assertIsNaN(math.ulp(math.nan))

        # negative number: ulp(-x) == ulp(x)
        for x in (0.0, 1.0, 2 ** 52, 2 ** 64, INF):
            with self.subTest(x=x):
                self.assertEqual(math.ulp(-x), math.ulp(x))

    def test_issue39871(self):
        # [FUNCTION DOC] Regression test for issue 39871: atan2, copysign and remainder
        # must raise TypeError (not SystemError) when the first argument is not
        # convertible, and must not try to convert the second argument afterwards
        # (verified via a flag on the fixture).
        # A SystemError should not be raised if the first arg to atan2(),
        # copysign(), or remainder() cannot be converted to a float.
        class F:
            # [CLASS DOC] HELPER: Local fixture whose float conversion records that it was
            # called and then raises ZeroDivisionError.
            def __float__(self):
                # [FUNCTION DOC] HELPER: Sets the converted flag on the instance and divides by
                # zero, raising ZeroDivisionError.
                self.converted = True
                1/0
        for func in math.atan2, math.copysign, math.remainder:
            y = F()
            with self.assertRaises(TypeError):
                func("not a number", y)

            # There should not have been any attempt to convert the second
            # argument to a float.
            self.assertFalse(getattr(y, "converted", False))

    def test_input_exceptions(self):
        # [FUNCTION DOC] Tests that math.exp, erf and atan2 raise TypeError for non-numeric
        # arguments and atan2 raises TypeError for the wrong number of arguments.
        self.assertRaises(TypeError, math.exp, "spam")
        self.assertRaises(TypeError, math.erf, "spam")
        self.assertRaises(TypeError, math.atan2, "spam", 1.0)
        self.assertRaises(TypeError, math.atan2, 1.0, "spam")
        self.assertRaises(TypeError, math.atan2, 1.0)
        self.assertRaises(TypeError, math.atan2, 1.0, 2.0, 3.0)

    # Custom assertions.

    def assertIsNaN(self, value):
        # [FUNCTION DOC] HELPER: Custom assertion; fails the test with a message if the value
        # is not NaN.
        if not math.isnan(value):
            self.fail("Expected a NaN, got {!r}.".format(value))

    def assertEqualSign(self, x, y):
        # [FUNCTION DOC] HELPER: Custom assertion; asserts x equals y and that their sign bits
        # (via copysign) match, to compare signed zeros.
        """Similar to assertEqual(), but compare also the sign with copysign().

        Function useful to compare signed zeros.
        """
        self.assertEqual(x, y)
        self.assertEqual(math.copysign(1.0, x), math.copysign(1.0, y))


class IsCloseTests(__TestCase):
    # [CLASS DOC] Tests for math.isclose (PEP 485). The class attribute isclose selects the
    # function under test so subclasses can reuse the tests with another implementation.
    # Provides assertion helpers that report both values on failure.
    isclose = math.isclose  # subclasses should override this

    def assertIsClose(self, a, b, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts isclose(a, b, ...) is true; message names both values.
        self.assertTrue(self.isclose(a, b, *args, **kwargs),
                        msg="%s and %s should be close!" % (a, b))

    def assertIsNotClose(self, a, b, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts isclose(a, b, ...) is false; message names both values.
        self.assertFalse(self.isclose(a, b, *args, **kwargs),
                         msg="%s and %s should not be close!" % (a, b))

    def assertAllClose(self, examples, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts every pair in examples is close under the
        # given tolerance arguments.
        for a, b in examples:
            self.assertIsClose(a, b, *args, **kwargs)

    def assertAllNotClose(self, examples, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Asserts no pair in examples is close under the
        # given tolerance arguments.
        for a, b in examples:
            self.assertIsNotClose(a, b, *args, **kwargs)

    def test_negative_tolerances(self):
        # [FUNCTION DOC] Checks that isclose raises ValueError when rel_tol or
        # abs_tol is negative.
        # ValueError should be raised if either tolerance is less than zero
        with self.assertRaises(ValueError):
            self.assertIsClose(1, 1, rel_tol=-1e-100)
        with self.assertRaises(ValueError):
            self.assertIsClose(1, 1, rel_tol=1e-100, abs_tol=-1e10)

    def test_identical(self):
        # [FUNCTION DOC] Checks that identical values (including 0.0 and -0.0, int vs float) are
        # close even with both tolerances zero.
        # identical values must test as close
        identical_examples = [(2.0, 2.0),
                              (0.1e200, 0.1e200),
                              (1.123e-300, 1.123e-300),
                              (12345, 12345.0),
                              (0.0, -0.0),
                              (345678, 345678)]
        self.assertAllClose(identical_examples, rel_tol=0.0, abs_tol=0.0)

    def test_eight_decimal_places(self):
        # [FUNCTION DOC] Checks examples within 1e-8 relative tolerance are close and not
        # close at 1e-9.
        # examples that are close to 1e-8, but not 1e-9
        eight_decimal_places_examples = [(1e8, 1e8 + 1),
                                         (-1e-8, -1.000000009e-8),
                                         (1.12345678, 1.12345679)]
        self.assertAllClose(eight_decimal_places_examples, rel_tol=1e-8)
        self.assertAllNotClose(eight_decimal_places_examples, rel_tol=1e-9)

    def test_near_zero(self):
        # [FUNCTION DOC] Checks values near zero are never close by relative tolerance but are
        # close with an absolute tolerance of 1e-8.
        # values close to zero
        near_zero_examples = [(1e-9, 0.0),
                              (-1e-9, 0.0),
                              (-1e-150, 0.0)]
        # these should not be close to any rel_tol
        self.assertAllNotClose(near_zero_examples, rel_tol=0.9)
        # these should be close to abs_tol=1e-8
        self.assertAllClose(near_zero_examples, abs_tol=1e-8)

    def test_identical_infinite(self):
        # [FUNCTION DOC] Checks equal infinities (same sign) are close regardless of tolerance.
        # these are close regardless of tolerance -- i.e. they are equal
        self.assertIsClose(INF, INF)
        self.assertIsClose(INF, INF, abs_tol=0.0)
        self.assertIsClose(NINF, NINF)
        self.assertIsClose(NINF, NINF, abs_tol=0.0)

    def test_inf_ninf_nan(self):
        # [FUNCTION DOC] Checks NaN, mixed infinities, and infinity against finite numbers are
        # never close, even with a very large absolute tolerance.
        # these should never be close (following IEEE 754 rules for equality)
        not_close_examples = [(NAN, NAN),
                              (NAN, 1e-100),
                              (1e-100, NAN),
                              (INF, NAN),
                              (NAN, INF),
                              (INF, NINF),
                              (INF, 1.0),
                              (1.0, INF),
                              (INF, 1e308),
                              (1e308, INF)]
        # use largest reasonable tolerance
        self.assertAllNotClose(not_close_examples, abs_tol=0.999999999999999)

    def test_zero_tolerance(self):
        # [FUNCTION DOC] Checks that with zero relative tolerance equal values are close and
        # slightly different values are not.
        # test with zero tolerance
        zero_tolerance_close_examples = [(1.0, 1.0),
                                         (-3.4, -3.4),
                                         (-1e-300, -1e-300)]
        self.assertAllClose(zero_tolerance_close_examples, rel_tol=0.0)

        zero_tolerance_not_close_examples = [(1.0, 1.000000000000001),
                                             (0.99999999999999, 1.0),
                                             (1.0e200, .999999999999999e200)]
        self.assertAllNotClose(zero_tolerance_not_close_examples, rel_tol=0.0)

    def test_asymmetry(self):
        # [FUNCTION DOC] Checks the PEP 485 symmetry example: 9 and 10 are close with rel_tol
        # 0.1 in both argument orders.
        # test the asymmetry example from PEP 485
        self.assertAllClose([(9, 10), (10, 9)], rel_tol=0.1)

    def test_integers(self):
        # [FUNCTION DOC] Checks isclose with int arguments at 1e-8 versus
        # 1e-9 relative tolerance.
        # test with integer values
        integer_examples = [(100000001, 100000000),
                            (123456789, 123456788)]

        self.assertAllClose(integer_examples, rel_tol=1e-8)
        self.assertAllNotClose(integer_examples, rel_tol=1e-9)

    def test_decimals(self):
        # [FUNCTION DOC] Checks isclose with Decimal arguments at several magnitudes at
        # 1e-8 versus 1e-9.
        # test with Decimal values
        from decimal import Decimal

        decimal_examples = [(Decimal('1.00000001'), Decimal('1.0')),
                            (Decimal('1.00000001e-20'), Decimal('1.0e-20')),
                            (Decimal('1.00000001e-100'), Decimal('1.0e-100')),
                            (Decimal('1.00000001e20'), Decimal('1.0e20'))]
        self.assertAllClose(decimal_examples, rel_tol=1e-8)
        self.assertAllNotClose(decimal_examples, rel_tol=1e-9)

    def test_fractions(self):
        # [FUNCTION DOC] Checks isclose with Fraction arguments at 1e-8 versus
        # 1e-9 relative tolerance.
        # test with Fraction values
        from fractions import Fraction

        fraction_examples = [
            (Fraction(1, 100000000) + 1, Fraction(1)),
            (Fraction(100000001), Fraction(100000000)),
            (Fraction(10**8 + 1, 10**28), Fraction(1, 10**20))]
        self.assertAllClose(fraction_examples, rel_tol=1e-8)
        self.assertAllNotClose(fraction_examples, rel_tol=1e-9)


class FMATests(__TestCase):
    # [CLASS DOC] Tests for math.fma (fused multiply-add): NaN, infinity, signed-zero, overflow
    # and single-rounding behavior, plus random inputs where a naive multiply then add differs.
    # Defines custom assertions for NaN and signed zeros. No instance state.
    """ Tests for math.fma. """

    def test_fma_nan_results(self):
        # [FUNCTION DOC] Checks that fma returns NaN whenever any of the three arguments is NaN,
        # for a set of representative values in the other two positions.
        # Selected representative values.
        values = [
            -math.inf, -1e300, -2.3, -1e-300, -0.0,
            0.0, 1e-300, 2.3, 1e300, math.inf, math.nan
        ]

        # If any input is a NaN, the result should be a NaN, too.
        for a, b in itertools.product(values, repeat=2):
            with self.subTest(a=a, b=b):
                self.assertIsNaN(math.fma(math.nan, a, b))
                self.assertIsNaN(math.fma(a, math.nan, b))
                self.assertIsNaN(math.fma(a, b, math.nan))

    def test_fma_infinities(self):
        # [FUNCTION DOC] Checks fma with infinities: ValueError for infinity times zero or when
        # product and addend are infinities of opposite sign; infinite results when same sign or
        # when only one of product and addend is infinite and the other finite.
        # Cases involving infinite inputs or results.
        positives = [1e-300, 2.3, 1e300, math.inf]
        finites = [-1e300, -2.3, -1e-300, -0.0, 0.0, 1e-300, 2.3, 1e300]
        non_nans = [-math.inf, -2.3, -0.0, 0.0, 2.3, math.inf]

        # ValueError due to inf * 0 computation.
        for c in non_nans:
            for infinity in [math.inf, -math.inf]:
                for zero in [0.0, -0.0]:
                    with self.subTest(c=c, infinity=infinity, zero=zero):
                        with self.assertRaises(ValueError):
                            math.fma(infinity, zero, c)
                        with self.assertRaises(ValueError):
                            math.fma(zero, infinity, c)

        # ValueError when a*b and c both infinite of opposite signs.
        for b in positives:
            with self.subTest(b=b):
                with self.assertRaises(ValueError):
                    math.fma(math.inf, b, -math.inf)
                with self.assertRaises(ValueError):
                    math.fma(math.inf, -b, math.inf)
                with self.assertRaises(ValueError):
                    math.fma(-math.inf, -b, -math.inf)
                with self.assertRaises(ValueError):
                    math.fma(-math.inf, b, math.inf)
                with self.assertRaises(ValueError):
                    math.fma(b, math.inf, -math.inf)
                with self.assertRaises(ValueError):
                    math.fma(-b, math.inf, math.inf)
                with self.assertRaises(ValueError):
                    math.fma(-b, -math.inf, -math.inf)
                with self.assertRaises(ValueError):
                    math.fma(b, -math.inf, math.inf)

        # Infinite result when a*b and c both infinite of the same sign.
        for b in positives:
            with self.subTest(b=b):
                self.assertEqual(math.fma(math.inf, b, math.inf), math.inf)
                self.assertEqual(math.fma(math.inf, -b, -math.inf), -math.inf)
                self.assertEqual(math.fma(-math.inf, -b, math.inf), math.inf)
                self.assertEqual(math.fma(-math.inf, b, -math.inf), -math.inf)
                self.assertEqual(math.fma(b, math.inf, math.inf), math.inf)
                self.assertEqual(math.fma(-b, math.inf, -math.inf), -math.inf)
                self.assertEqual(math.fma(-b, -math.inf, math.inf), math.inf)
                self.assertEqual(math.fma(b, -math.inf, -math.inf), -math.inf)

        # Infinite result when a*b finite, c infinite.
        for a, b in itertools.product(finites, finites):
            with self.subTest(b=b):
                self.assertEqual(math.fma(a, b, math.inf), math.inf)
                self.assertEqual(math.fma(a, b, -math.inf), -math.inf)

        # Infinite result when a*b infinite, c finite.
        for b, c in itertools.product(positives, finites):
            with self.subTest(b=b, c=c):
                self.assertEqual(math.fma(math.inf, b, c), math.inf)
                self.assertEqual(math.fma(-math.inf, b, c), -math.inf)
                self.assertEqual(math.fma(-math.inf, -b, c), math.inf)
                self.assertEqual(math.fma(math.inf, -b, c), -math.inf)

                self.assertEqual(math.fma(b, math.inf, c), math.inf)
                self.assertEqual(math.fma(b, -math.inf, c), -math.inf)
                self.assertEqual(math.fma(-b, -math.inf, c), math.inf)
                self.assertEqual(math.fma(-b, math.inf, c), -math.inf)

    # gh-73468: On some platforms, libc fma() doesn't implement IEE 754-2008
    # properly: it doesn't use the right sign when the result is zero.
    @unittest.skipIf(
        sys.platform.startswith(("freebsd", "wasi", "netbsd", "emscripten"))
        or (sys.platform == "android" and platform.machine() == "x86_64"),
        f"this platform doesn't implement IEE 754-2008 properly")
    def test_fma_zero_result(self):
        # [FUNCTION DOC] Checks the sign of zero results of fma: exact zero inputs, exact
        # cancellation (always positive zero), underflow to signed zero, and a case where
        # rounding the product first would give the wrong sign. Skipped on platforms whose libc
        # fma has wrong zero signs.
        nonnegative_finites = [0.0, 1e-300, 2.3, 1e300]

        # Zero results from exact zero inputs.
        for b in nonnegative_finites:
            with self.subTest(b=b):
                self.assertIsPositiveZero(math.fma(0.0, b, 0.0))
                self.assertIsPositiveZero(math.fma(0.0, b, -0.0))
                self.assertIsNegativeZero(math.fma(0.0, -b, -0.0))
                self.assertIsPositiveZero(math.fma(0.0, -b, 0.0))
                self.assertIsPositiveZero(math.fma(-0.0, -b, 0.0))
                self.assertIsPositiveZero(math.fma(-0.0, -b, -0.0))
                self.assertIsNegativeZero(math.fma(-0.0, b, -0.0))
                self.assertIsPositiveZero(math.fma(-0.0, b, 0.0))

                self.assertIsPositiveZero(math.fma(b, 0.0, 0.0))
                self.assertIsPositiveZero(math.fma(b, 0.0, -0.0))
                self.assertIsNegativeZero(math.fma(-b, 0.0, -0.0))
                self.assertIsPositiveZero(math.fma(-b, 0.0, 0.0))
                self.assertIsPositiveZero(math.fma(-b, -0.0, 0.0))
                self.assertIsPositiveZero(math.fma(-b, -0.0, -0.0))
                self.assertIsNegativeZero(math.fma(b, -0.0, -0.0))
                self.assertIsPositiveZero(math.fma(b, -0.0, 0.0))

        # Exact zero result from nonzero inputs.
        self.assertIsPositiveZero(math.fma(2.0, 2.0, -4.0))
        self.assertIsPositiveZero(math.fma(2.0, -2.0, 4.0))
        self.assertIsPositiveZero(math.fma(-2.0, -2.0, -4.0))
        self.assertIsPositiveZero(math.fma(-2.0, 2.0, 4.0))

        # Underflow to zero.
        tiny = 1e-300
        self.assertIsPositiveZero(math.fma(tiny, tiny, 0.0))
        self.assertIsNegativeZero(math.fma(tiny, -tiny, 0.0))
        self.assertIsPositiveZero(math.fma(-tiny, -tiny, 0.0))
        self.assertIsNegativeZero(math.fma(-tiny, tiny, 0.0))
        self.assertIsPositiveZero(math.fma(tiny, tiny, -0.0))
        self.assertIsNegativeZero(math.fma(tiny, -tiny, -0.0))
        self.assertIsPositiveZero(math.fma(-tiny, -tiny, -0.0))
        self.assertIsNegativeZero(math.fma(-tiny, tiny, -0.0))

        # Corner case where rounding the multiplication would
        # give the wrong result.
        x = float.fromhex('0x1p-500')
        y = float.fromhex('0x1p-550')
        z = float.fromhex('0x1p-1000')
        self.assertIsNegativeZero(math.fma(x-y, x+y, -z))
        self.assertIsPositiveZero(math.fma(y-x, x+y, z))
        self.assertIsNegativeZero(math.fma(y-x, -(x+y), -z))
        self.assertIsPositiveZero(math.fma(x-y, -(x+y), z))

    def test_fma_overflow(self):
        # [FUNCTION DOC] Checks fma overflow behavior: OverflowError when the product or sum
        # overflows, but a finite result when a large product is offset by the addend, including
        # boundary cases where the exact result is just representable.
        a = b = float.fromhex('0x1p512')
        c = float.fromhex('0x1p1023')
        # Overflow from multiplication.
        with self.assertRaises(OverflowError):
            math.fma(a, b, 0.0)
        self.assertEqual(math.fma(a, b/2.0, 0.0), c)
        # Overflow from the addition.
        with self.assertRaises(OverflowError):
            math.fma(a, b/2.0, c)
        # No overflow, even though a*b overflows a float.
        self.assertEqual(math.fma(a, b, -c), c)

        # Extreme case: a * b is exactly at the overflow boundary, so the
        # tiniest offset makes a difference between overflow and a finite
        # result.
        a = float.fromhex('0x1.ffffffc000000p+511')
        b = float.fromhex('0x1.0000002000000p+512')
        c = float.fromhex('0x0.0000000000001p-1022')
        with self.assertRaises(OverflowError):
            math.fma(a, b, 0.0)
        with self.assertRaises(OverflowError):
            math.fma(a, b, c)
        self.assertEqual(math.fma(a, b, -c),
                         float.fromhex('0x1.fffffffffffffp+1023'))

        # Another extreme case: here a*b is about as large as possible subject
        # to math.fma(a, b, c) being finite.
        a = float.fromhex('0x1.ae565943785f9p+512')
        b = float.fromhex('0x1.3094665de9db8p+512')
        c = float.fromhex('0x1.fffffffffffffp+1023')
        self.assertEqual(math.fma(a, b, -c), c)

    def test_fma_single_round(self):
        # [FUNCTION DOC] Checks fma rounds once: fma of (a-1) and (a+1) plus 1 equals a squared
        # for a tiny a.
        a = float.fromhex('0x1p-50')
        self.assertEqual(math.fma(a - 1.0, a + 1.0, 1.0), a*a)

    def test_random(self):
        # [FUNCTION DOC] Checks fma on hand-picked random hex triples where a singly
        # rounded result differs from a double-rounded one; verifies both argument orders
        # of the multiplicands.
        # A collection of randomly generated inputs for which the naive FMA
        # (with two rounds) gives a different result from a singly-rounded FMA.

        # tuples (a, b, c, expected)
        test_values = [
            ('0x1.694adde428b44p-1', '0x1.371b0d64caed7p-1',
             '0x1.f347e7b8deab8p-4', '0x1.19f10da56c8adp-1'),
            ('0x1.605401ccc6ad6p-2', '0x1.ce3a40bf56640p-2',
             '0x1.96e3bf7bf2e20p-2', '0x1.1af6d8aa83101p-1'),
            ('0x1.e5abd653a67d4p-2', '0x1.a2e400209b3e6p-1',
             '0x1.a90051422ce13p-1', '0x1.37d68cc8c0fbbp+0'),
            ('0x1.f94e8efd54700p-2', '0x1.123065c812cebp-1',
             '0x1.458f86fb6ccd0p-1', '0x1.ccdcee26a3ff3p-1'),
            ('0x1.bd926f1eedc96p-1', '0x1.eee9ca68c5740p-1',
             '0x1.960c703eb3298p-2', '0x1.3cdcfb4fdb007p+0'),
            ('0x1.27348350fbccdp-1', '0x1.3b073914a53f1p-1',
             '0x1.e300da5c2b4cbp-1', '0x1.4c51e9a3c4e29p+0'),
            ('0x1.2774f00b3497bp-1', '0x1.7038ec336bff0p-2',
             '0x1.2f6f2ccc3576bp-1', '0x1.99ad9f9c2688bp-1'),
            ('0x1.51d5a99300e5cp-1', '0x1.5cd74abd445a1p-1',
             '0x1.8880ab0bbe530p-1', '0x1.3756f96b91129p+0'),
            ('0x1.73cb965b821b8p-2', '0x1.218fd3d8d5371p-1',
             '0x1.d1ea966a1f758p-2', '0x1.5217b8fd90119p-1'),
            ('0x1.4aa98e890b046p-1', '0x1.954d85dff1041p-1',
             '0x1.122b59317ebdfp-1', '0x1.0bf644b340cc5p+0'),
            ('0x1.e28f29e44750fp-1', '0x1.4bcc4fdcd18fep-1',
             '0x1.fd47f81298259p-1', '0x1.9b000afbc9995p+0'),
            ('0x1.d2e850717fe78p-3', '0x1.1dd7531c303afp-1',
             '0x1.e0869746a2fc2p-2', '0x1.316df6eb26439p-1'),
            ('0x1.cf89c75ee6fbap-2', '0x1.b23decdc66825p-1',
             '0x1.3d1fe76ac6168p-1', '0x1.00d8ea4c12abbp+0'),
            ('0x1.3265ae6f05572p-2', '0x1.16d7ec285f7a2p-1',
             '0x1.0b8405b3827fbp-1', '0x1.5ef33c118a001p-1'),
            ('0x1.c4d1bf55ec1a5p-1', '0x1.bc59618459e12p-2',
             '0x1.ce5b73dc1773dp-1', '0x1.496cf6164f99bp+0'),
            ('0x1.d350026ac3946p-1', '0x1.9a234e149a68cp-2',
             '0x1.f5467b1911fd6p-2', '0x1.b5cee3225caa5p-1'),
        ]
        for a_hex, b_hex, c_hex, expected_hex in test_values:
            with self.subTest(a_hex=a_hex, b_hex=b_hex, c_hex=c_hex,
                              expected_hex=expected_hex):
                a = float.fromhex(a_hex)
                b = float.fromhex(b_hex)
                c = float.fromhex(c_hex)
                expected = float.fromhex(expected_hex)
                self.assertEqual(math.fma(a, b, c), expected)
                self.assertEqual(math.fma(b, a, c), expected)

    # Custom assertions.
    def assertIsNaN(self, value):
        # [FUNCTION DOC] HELPER: Custom assertion; fails if the value is not a NaN.
        self.assertTrue(
            math.isnan(value),
            msg="Expected a NaN, got {!r}".format(value)
        )

    def assertIsPositiveZero(self, value):
        # [FUNCTION DOC] HELPER: Custom assertion; fails unless the value is zero with
        # positive sign bit.
        self.assertTrue(
            value == 0 and math.copysign(1, value) > 0,
            msg="Expected a positive zero, got {!r}".format(value)
        )

    def assertIsNegativeZero(self, value):
        # [FUNCTION DOC] HELPER: Custom assertion; fails unless the value is zero with
        # negative sign bit.
        self.assertTrue(
            value == 0 and math.copysign(1, value) < 0,
            msg="Expected a negative zero, got {!r}".format(value)
        )


if __name__ == "__main__":
    run_tests()
