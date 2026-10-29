# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Tests for binary operators on user-defined numeric types, from CPython 3.13
# Lib/test/test_binop.py, run under CPythonTestCase. Defines a Rat rational-number class
# with arithmetic and equality, tests for it (RatTestCase), comparison-order tests using
# logging classes with an ABC virtual subclass (OperationOrderTests), and tests that
# setting a comparison method to None blocks reflected fallback (FallbackBlockingTests).
# KNOWN ISSUE: the "Test copied from" URL in the header
# names test_iter.py although the content is test_binop.py.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_iter.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    run_tests,
)

# ======= END DYNAMO PATCH =======

"""Tests for binary operators on subtypes of built-in types."""

import unittest
from operator import eq, le, ne
from abc import ABCMeta

def gcd(a, b):
    # [FUNCTION DOC] HELPER: Greatest common divisor by Euclid's algorithm
    # on two integers; the sign follows the second operand's remainder
    # chain, so a negative first argument yields a negative result.
    """Greatest common divisor using Euclid's algorithm."""
    while a:
        a, b = b%a, a
    return b

def isint(x):
    # [FUNCTION DOC] HELPER: Returns True if x is an instance of int.
    """Test whether an object is an instance of int."""
    return isinstance(x, int)

def isnum(x):
    # [FUNCTION DOC] HELPER: Returns 1 if x is an int, float or complex instance, else 0.
    """Test whether an object is an instance of a built-in numeric type."""
    for T in int, float, complex:
        if isinstance(x, T):
            return 1
    return 0

def isRat(x):
    # [FUNCTION DOC] HELPER: Returns True if x is an instance of Rat.
    """Test whether an object is an instance of the Rat class."""
    return isinstance(x, Rat)

class Rat(object):
    # [CLASS DOC] Rational number stored as a normalized pair of ints in slot attributes
    # (numerator, denominator), exposed read-only through the num and den properties. Supports
    # addition, subtraction, multiplication, true and floor division, divmod, modulo (all with
    # reflected forms), float/int conversion and equality; mixing with ints stays exact,
    # mixing with floats or complex gives float results, other types give NotImplemented.

    """Rational number implemented as a normalized pair of ints."""

    __slots__ = ['_Rat__num', '_Rat__den']

    def __init__(self, num=0, den=1):
        # [FUNCTION DOC] Creates a Rat from numerator and denominator (defaults 0 and 1). Raises
        # TypeError if either is not an int and ZeroDivisionError for a zero denominator;
        # reduces by the gcd so the sign ends up on the numerator after normalization.
        """Constructor: Rat([num[, den]]).

        The arguments must be ints, and default to (0, 1)."""
        if not isint(num):
            raise TypeError("Rat numerator must be int (%r)" % num)
        if not isint(den):
            raise TypeError("Rat denominator must be int (%r)" % den)
        # But the zero is always on
        if den == 0:
            raise ZeroDivisionError("zero denominator")
        g = gcd(den, num)
        self.__num = int(num//g)
        self.__den = int(den//g)

    def _get_num(self):
        # [FUNCTION DOC] HELPER: Returns the stored numerator.
        """Accessor function for read-only 'num' attribute of Rat."""
        return self.__num
    num = property(_get_num, None)

    def _get_den(self):
        # [FUNCTION DOC] HELPER: Returns the stored denominator.
        """Accessor function for read-only 'den' attribute of Rat."""
        return self.__den
    den = property(_get_den, None)

    def __repr__(self):
        # [FUNCTION DOC] Returns a string resembling a
        # constructor call with numerator and denominator.
        """Convert a Rat to a string resembling a Rat constructor call."""
        return "Rat(%d, %d)" % (self.__num, self.__den)

    def __str__(self):
        # [FUNCTION DOC] Returns the string form of the float value.
        """Convert a Rat to a string resembling a decimal numeric value."""
        return str(float(self))

    def __float__(self):
        # [FUNCTION DOC] Returns the numerator divided by the denominator as a float.
        """Convert a Rat to a float."""
        return self.__num*1.0/self.__den

    def __int__(self):
        # [FUNCTION DOC] Returns the numerator as an int when the denominator is 1; raises
        # OverflowError if it is too large for int and ValueError if the denominator is not 1.
        """Convert a Rat to an int; self.den must be 1."""
        if self.__den == 1:
            try:
                return int(self.__num)
            except OverflowError:
                raise OverflowError("%s too large to convert to int" %
                                      repr(self))
        raise ValueError("can't convert %s to int" % repr(self))

    def __add__(self, other):
        # [FUNCTION DOC] Adds an int, Rat (exact result) or other number
        # (float result); returns NotImplemented for other types.
        """Add two Rats, or a Rat and a number."""
        if isint(other):
            other = Rat(other)
        if isRat(other):
            return Rat(self.__num*other.__den + other.__num*self.__den,
                       self.__den*other.__den)
        if isnum(other):
            return float(self) + other
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        # [FUNCTION DOC] Subtracts an int, Rat (exact result) or other
        # number (float result) from self; returns NotImplemented otherwise.
        """Subtract two Rats, or a Rat and a number."""
        if isint(other):
            other = Rat(other)
        if isRat(other):
            return Rat(self.__num*other.__den - other.__num*self.__den,
                       self.__den*other.__den)
        if isnum(other):
            return float(self) - other
        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC] Reflected subtraction: computes other minus self
        # for an int, Rat or other number; NotImplemented otherwise.
        """Subtract two Rats, or a Rat and a number (reversed args)."""
        if isint(other):
            other = Rat(other)
        if isRat(other):
            return Rat(other.__num*self.__den - self.__num*other.__den,
                       self.__den*other.__den)
        if isnum(other):
            return other - float(self)
        return NotImplemented

    def __mul__(self, other):
        # [FUNCTION DOC] Multiplies by a Rat or int (exact result)
        # or other number (float result); NotImplemented otherwise.
        """Multiply two Rats, or a Rat and a number."""
        if isRat(other):
            return Rat(self.__num*other.__num, self.__den*other.__den)
        if isint(other):
            return Rat(self.__num*other, self.__den)
        if isnum(other):
            return float(self)*other
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        # [FUNCTION DOC] Divides self by a Rat or int (exact result)
        # or other number (float result); NotImplemented otherwise.
        """Divide two Rats, or a Rat and a number."""
        if isRat(other):
            return Rat(self.__num*other.__den, self.__den*other.__num)
        if isint(other):
            return Rat(self.__num, self.__den*other)
        if isnum(other):
            return float(self) / other
        return NotImplemented

    def __rtruediv__(self, other):
        # [FUNCTION DOC] Reflected true division: other divided by self
        # for a Rat, int or other number; NotImplemented otherwise.
        """Divide two Rats, or a Rat and a number (reversed args)."""
        if isRat(other):
            return Rat(other.__num*self.__den, other.__den*self.__num)
        if isint(other):
            return Rat(other*self.__den, self.__num)
        if isnum(other):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other):
        # [FUNCTION DOC] Floor division by an int or Rat, returning the
        # integer floor of the exact quotient; NotImplemented for other types.
        """Divide two Rats, returning the floored result."""
        if isint(other):
            other = Rat(other)
        elif not isRat(other):
            return NotImplemented
        x = self/other
        return x.__num // x.__den

    def __rfloordiv__(self, other):
        # [FUNCTION DOC] Reflected floor division: the integer floor of other divided by self.
        """Divide two Rats, returning the floored result (reversed args)."""
        x = other/self
        return x.__num // x.__den

    def __divmod__(self, other):
        # [FUNCTION DOC] Divides by an int or Rat and returns a pair of the
        # floored quotient and the remainder; NotImplemented for other types.
        """Divide two Rats, returning quotient and remainder."""
        if isint(other):
            other = Rat(other)
        elif not isRat(other):
            return NotImplemented
        x = self//other
        return (x, self - other * x)

    def __rdivmod__(self, other):
        # [FUNCTION DOC] Reflected divmod: converts an int other to Rat and
        # returns divmod of other by self; NotImplemented for other types.
        """Divide two Rats, returning quotient and remainder (reversed args)."""
        if isint(other):
            other = Rat(other)
        elif not isRat(other):
            return NotImplemented
        return divmod(other, self)

    def __mod__(self, other):
        # [FUNCTION DOC] Returns the remainder part of divmod of self by other.
        """Take one Rat modulo another."""
        return divmod(self, other)[1]

    def __rmod__(self, other):
        # [FUNCTION DOC] Reflected modulo: the remainder part of divmod of other by self.
        """Take one Rat modulo another (reversed args)."""
        return divmod(other, self)[1]

    def __eq__(self, other):
        # [FUNCTION DOC] Compares to an int (equal only when the denominator is 1), a Rat
        # (same numerator and denominator) or another number (float comparison);
        # NotImplemented otherwise. Defining equality without hash makes instances unhashable.
        """Compare two Rats for equality."""
        if isint(other):
            return self.__den == 1 and self.__num == other
        if isRat(other):
            return self.__num == other.__num and self.__den == other.__den
        if isnum(other):
            return float(self) == other
        return NotImplemented

class RatTestCase(CPythonTestCase):
    # [CLASS DOC] Unit tests for Rat and the gcd helper.
    """Unit tests for Rat class and its support utilities."""

    def test_gcd(self):
        # [FUNCTION DOC] Checks gcd results for positive and negative
        # arguments and the sign pattern over a grid of values.
        self.assertEqual(gcd(10, 12), 2)
        self.assertEqual(gcd(10, 15), 5)
        self.assertEqual(gcd(10, 11), 1)
        self.assertEqual(gcd(100, 15), 5)
        self.assertEqual(gcd(-10, 2), -2)
        self.assertEqual(gcd(10, -2), 2)
        self.assertEqual(gcd(-10, -2), -2)
        for i in range(1, 20):
            for j in range(1, 20):
                self.assertTrue(gcd(i, j) > 0)
                self.assertTrue(gcd(-i, j) < 0)
                self.assertTrue(gcd(i, -j) > 0)
                self.assertTrue(gcd(-i, -j) < 0)

    def test_constructor(self):
        # [FUNCTION DOC] Checks normalization of signs and reduction in the
        # constructor, the integer default, ZeroDivisionError for a zero
        # denominator, and TypeError for non-int numerator or denominator.
        a = Rat(10, 15)
        self.assertEqual(a.num, 2)
        self.assertEqual(a.den, 3)
        a = Rat(10, -15)
        self.assertEqual(a.num, -2)
        self.assertEqual(a.den, 3)
        a = Rat(-10, 15)
        self.assertEqual(a.num, -2)
        self.assertEqual(a.den, 3)
        a = Rat(-10, -15)
        self.assertEqual(a.num, 2)
        self.assertEqual(a.den, 3)
        a = Rat(7)
        self.assertEqual(a.num, 7)
        self.assertEqual(a.den, 1)
        try:
            a = Rat(1, 0)
        except ZeroDivisionError:
            pass
        else:
            self.fail("Rat(1, 0) didn't raise ZeroDivisionError")
        for bad in "0", 0.0, 0j, (), [], {}, None, Rat, unittest:
            try:
                a = Rat(bad)
            except TypeError:
                pass
            else:
                self.fail("Rat(%r) didn't raise TypeError" % bad)
            try:
                a = Rat(1, bad)
            except TypeError:
                pass
            else:
                self.fail("Rat(1, %r) didn't raise TypeError" % bad)

    def test_add(self):
        # [FUNCTION DOC] Checks addition with Rat, int and float operands in both orders.
        self.assertEqual(Rat(2, 3) + Rat(1, 3), 1)
        self.assertEqual(Rat(2, 3) + 1, Rat(5, 3))
        self.assertEqual(1 + Rat(2, 3), Rat(5, 3))
        self.assertEqual(1.0 + Rat(1, 2), 1.5)
        self.assertEqual(Rat(1, 2) + 1.0, 1.5)

    def test_sub(self):
        # [FUNCTION DOC] Checks subtraction with Rat, int and float operands in both orders.
        self.assertEqual(Rat(7, 2) - Rat(7, 5), Rat(21, 10))
        self.assertEqual(Rat(7, 5) - 1, Rat(2, 5))
        self.assertEqual(1 - Rat(3, 5), Rat(2, 5))
        self.assertEqual(Rat(3, 2) - 1.0, 0.5)
        self.assertEqual(1.0 - Rat(1, 2), 0.5)

    def test_mul(self):
        # [FUNCTION DOC] Checks multiplication with Rat, int and float operands in both orders.
        self.assertEqual(Rat(2, 3) * Rat(5, 7), Rat(10, 21))
        self.assertEqual(Rat(10, 3) * 3, 10)
        self.assertEqual(3 * Rat(10, 3), 10)
        self.assertEqual(Rat(10, 5) * 0.5, 1.0)
        self.assertEqual(0.5 * Rat(10, 5), 1.0)

    def test_div(self):
        # [FUNCTION DOC] Checks division of Rats and ints and float multiplication results.
        self.assertEqual(Rat(10, 3) / Rat(5, 7), Rat(14, 3))
        self.assertEqual(Rat(10, 3) / 3, Rat(10, 9))
        self.assertEqual(2 / Rat(5), Rat(2, 5))
        self.assertEqual(3.0 * Rat(1, 2), 1.5)
        self.assertEqual(Rat(1, 2) * 3.0, 1.5)

    def test_floordiv(self):
        # [FUNCTION DOC] Checks floor division of Rat by Rat and by int in both orders.
        self.assertEqual(Rat(10) // Rat(4), 2)
        self.assertEqual(Rat(10, 3) // Rat(4, 3), 2)
        self.assertEqual(Rat(10) // 4, 2)
        self.assertEqual(10 // Rat(4), 2)

    def test_eq(self):
        # [FUNCTION DOC] Checks equality of Rat with Rat, int and float in both orders.
        self.assertEqual(Rat(10), Rat(20, 2))
        self.assertEqual(Rat(10), 10)
        self.assertEqual(10, Rat(10))
        self.assertEqual(Rat(10), 10.0)
        self.assertEqual(10.0, Rat(10))

    def test_true_div(self):
        # [FUNCTION DOC] Checks true division results (mostly repeating the
        # division test) and that evaluating a plain division expression gives 0.5.
        self.assertEqual(Rat(10, 3) / Rat(5, 7), Rat(14, 3))
        self.assertEqual(Rat(10, 3) / 3, Rat(10, 9))
        self.assertEqual(2 / Rat(5), Rat(2, 5))
        self.assertEqual(3.0 * Rat(1, 2), 1.5)
        self.assertEqual(Rat(1, 2) * 3.0, 1.5)
        self.assertEqual(eval('1/2'), 0.5)

    # XXX Ran out of steam; TO DO: divmod, div, future division


class OperationLogger:
    # [CLASS DOC] HELPER: Base class for classes that log
    # operations through a callable supplied at construction.
    """Base class for classes with operation logging."""
    def __init__(self, logger):
        # [FUNCTION DOC] HELPER: Stores the logger callable.
        self.logger = logger
    def log_operation(self, *args):
        # [FUNCTION DOC] HELPER: Passes its arguments to the logger.
        self.logger(*args)

def op_sequence(op, *classes):
    # [FUNCTION DOC] HELPER: Instantiates each given class with a logger that
    # appends to a fresh list, applies the operator function to the instances
    # (ignoring TypeError) and returns the list of logged operation names.
    """Return the sequence of operations that results from applying
    the operation `op` to instances of the given classes."""
    log = []
    instances = []
    for c in classes:
        instances.append(c(log.append))

    try:
        op(*instances)
    except TypeError:
        pass
    return log

class A(OperationLogger):
    # [CLASS DOC] HELPER: Logging helper whose equality and
    # ordering hooks log their name and return NotImplemented.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'A.__eq__' and returns NotImplemented.
        self.log_operation('A.__eq__')
        return NotImplemented
    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'A.__le__' and returns NotImplemented.
        self.log_operation('A.__le__')
        return NotImplemented
    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'A.__ge__' and returns NotImplemented.
        self.log_operation('A.__ge__')
        return NotImplemented

class B(OperationLogger, metaclass=ABCMeta):
    # [CLASS DOC] HELPER: Logging helper with an ABCMeta metaclass;
    # its comparison hooks log and return NotImplemented.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'B.__eq__' and returns NotImplemented.
        self.log_operation('B.__eq__')
        return NotImplemented
    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'B.__le__' and returns NotImplemented.
        self.log_operation('B.__le__')
        return NotImplemented
    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'B.__ge__' and returns NotImplemented.
        self.log_operation('B.__ge__')
        return NotImplemented

class C(B):
    # [CLASS DOC] HELPER: Real subclass of B, so its reflected
    # comparison is tried first; hooks log and return NotImplemented.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'C.__eq__' and returns NotImplemented.
        self.log_operation('C.__eq__')
        return NotImplemented
    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'C.__le__' and returns NotImplemented.
        self.log_operation('C.__le__')
        return NotImplemented
    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'C.__ge__' and returns NotImplemented.
        self.log_operation('C.__ge__')
        return NotImplemented

class V(OperationLogger):
    # [CLASS DOC] HELPER: Logging helper registered as a virtual
    # subclass of B; hooks log and return NotImplemented.
    """Virtual subclass of B"""
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'V.__eq__' and returns NotImplemented.
        self.log_operation('V.__eq__')
        return NotImplemented
    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'V.__le__' and returns NotImplemented.
        self.log_operation('V.__le__')
        return NotImplemented
    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Logs 'V.__ge__' and returns NotImplemented.
        self.log_operation('V.__ge__')
        return NotImplemented
B.register(V)


class OperationOrderTests(CPythonTestCase):
    # [CLASS DOC] Tests the order in which comparison hooks of left and right operands are tried.
    def test_comparison_orders(self):
        # [FUNCTION DOC] Checks the logged call order for equality and less-or-equal over
        # pairs of A, B, C and V instances, including that a subclass operand's reflected
        # hook runs first and that the virtual subclass V does not get priority.
        self.assertEqual(op_sequence(eq, A, A), ['A.__eq__', 'A.__eq__'])
        self.assertEqual(op_sequence(eq, A, B), ['A.__eq__', 'B.__eq__'])
        self.assertEqual(op_sequence(eq, B, A), ['B.__eq__', 'A.__eq__'])
        # C is a subclass of B, so C.__eq__ is called first
        self.assertEqual(op_sequence(eq, B, C), ['C.__eq__', 'B.__eq__'])
        self.assertEqual(op_sequence(eq, C, B), ['C.__eq__', 'B.__eq__'])

        self.assertEqual(op_sequence(le, A, A), ['A.__le__', 'A.__ge__'])
        self.assertEqual(op_sequence(le, A, B), ['A.__le__', 'B.__ge__'])
        self.assertEqual(op_sequence(le, B, A), ['B.__le__', 'A.__ge__'])
        self.assertEqual(op_sequence(le, B, C), ['C.__ge__', 'B.__le__'])
        self.assertEqual(op_sequence(le, C, B), ['C.__le__', 'B.__ge__'])

        self.assertTrue(issubclass(V, B))
        self.assertEqual(op_sequence(eq, B, V), ['B.__eq__', 'V.__eq__'])
        self.assertEqual(op_sequence(le, B, V), ['B.__le__', 'V.__ge__'])

class SupEq(object):
    # [CLASS DOC] HELPER: Class whose equality always returns True.
    """Class that can test equality"""
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Always returns True.
        return True

class S(SupEq):
    # [CLASS DOC] HELPER: Subclass of SupEq that sets __eq__ to None to block equality.
    """Subclass of SupEq that should fail"""
    __eq__ = None

class F(object):
    # [CLASS DOC] HELPER: Independent class with no
    # equality hooks; falls back to default comparison.
    """Independent class that should fall back"""

class X(object):
    # [CLASS DOC] HELPER: Independent class that sets __eq__ to None to block equality.
    """Independent class that should fail"""
    __eq__ = None

class SN(SupEq):
    # [CLASS DOC] HELPER: Subclass of SupEq that sets __ne__
    # to None, so equality works but inequality is blocked.
    """Subclass of SupEq that can test equality, but not non-equality"""
    __ne__ = None

class XN:
    # [CLASS DOC] HELPER: Independent class with working
    # equality (always True) and __ne__ set to None.
    """Independent class that can test equality, but not non-equality"""
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Always returns True.
        return True
    __ne__ = None

class FallbackBlockingTests(CPythonTestCase):
    # [CLASS DOC] Tests that setting a comparison method to None
    # blocks the fallback to the other operand's reflected method.
    """Unit tests for None method blocking"""

    def test_fallback_rmethod_blocking(self):
        # [FUNCTION DOC] Checks equality outcomes among SupEq, F, S and X
        # instances: fallbacks succeed, but TypeError is raised when a
        # None-blocked operand is consulted (left operand first; subclass first).
        e, f, s, x = SupEq(), F(), S(), X()
        self.assertEqual(e, e)
        self.assertEqual(e, f)
        self.assertEqual(f, e)
        # left operand is checked first
        self.assertEqual(e, x)
        self.assertRaises(TypeError, eq, x, e)
        # S is a subclass, so it's always checked first
        self.assertRaises(TypeError, eq, e, s)
        self.assertRaises(TypeError, eq, s, e)

    def test_fallback_ne_blocking(self):
        # [FUNCTION DOC] Checks inequality with SN and XN: plain inequality is
        # False, and TypeError is raised when the blocked __ne__ is consulted.
        e, sn, xn = SupEq(), SN(), XN()
        self.assertFalse(e != e)
        self.assertRaises(TypeError, ne, e, sn)
        self.assertRaises(TypeError, ne, sn, e)
        self.assertFalse(e != xn)
        self.assertRaises(TypeError, ne, xn, e)

if __name__ == "__main__":
    run_tests()
