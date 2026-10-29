# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_types suite for built-in types. Test
# classes (all derived from the Dynamo CPythonTestCase) cover: basic truth values, arithmetic
# and string/number formatting (TypesTests), the vertical-bar union type (UnionTests),
# types.MappingProxyType (MappingProxyTests), types.new_class / prepare_class / resolve_bases /
# get_original_bases (ClassCreationTests), types.SimpleNamespace, types.coroutine wrapping
# (CoroutineTests), types.FunctionType defaults (FunctionTests) and static-type slot inheritance
# across subinterpreters (SubinterpreterTests). Import-time side effects: imports the CPython
# test support package, defines a no_rerun stand-in decorator, a module-level TypeVar T and two
# tiny classes (Example, Forward). Running the file as a script calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_types.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# Dummy decorator for CPython test compatibility
def no_rerun(reason):
    # [FUNCTION DOC] HELPER: Stand-in for the CPython no_rerun decorator factory. Takes a reason
    # string (ignored) and returns a decorator that hands back the decorated test unchanged.
    def decorator(func):
        # [FUNCTION DOC] HELPER: Identity decorator; returns the function it receives.
        return func
    return decorator

# ======= END DYNAMO PATCH =======

# Python test set -- part 6, built-in types

from test.support import (
    run_with_locale, is_apple_mobile, cpython_only,
    iter_builtin_types, iter_slot_wrappers,
    MISSING_C_DOCSTRINGS,
)
from test.support.script_helper import assert_python_ok
from test.support.import_helper import import_fresh_module

import collections.abc
from collections import namedtuple, UserDict
import copy
import _datetime
import gc
import inspect
import pickle
import locale
import sys
import textwrap
import types
import unittest.mock
import weakref
import typing


T = typing.TypeVar("T")

class Example:
    # [CLASS DOC] HELPER: Empty plain class used as an arbitrary non-type operand in
    # the union operator tests.
    pass

# [CLASS DOC] HELPER: Empty class whose name is the target of the string forward reference
# "Forward" resolved by the forward-reference union test.
class Forward: ...

def clear_typing_caches():
    # [FUNCTION DOC] HELPER: Runs every cache-clearing callback registered in the
    # typing module (typing._cleanups), so that cached generic/union objects are
    # rebuilt afresh. No return value.
    for f in typing._cleanups:
        f()


class TypesTests(CPythonTestCase):
    # [CLASS DOC] Tests of core built-in value behavior: truthiness, boolean and comparison
    # operators, int/float edge cases, string operations and slicing, and the format mini
    # language for ints and floats, plus type objects of descriptors and special singletons.

    def test_truth_values(self):
        # [FUNCTION DOC] Checks that None, 0, 0.0, empty string and empty dict are false while
        # non-zero numbers, non-empty strings and dicts, functions, classes, modules and plain
        # instances are true. Fails the test with a message on any mismatch.
        if None: self.fail('None is true instead of false')
        if 0: self.fail('0 is true instead of false')
        if 0.0: self.fail('0.0 is true instead of false')
        if '': self.fail('\'\' is true instead of false')
        if not 1: self.fail('1 is false instead of true')
        if not 1.0: self.fail('1.0 is false instead of true')
        if not 'x': self.fail('\'x\' is false instead of true')
        if not {'x': 1}: self.fail('{\'x\': 1} is false instead of true')
        # [FUNCTION DOC] HELPER: Empty function used only as a truthy object.
        def f(): pass
        # [CLASS DOC] HELPER: Empty class used only as a truthy object (class and instance).
        class C: pass
        x = C()
        if not f: self.fail('f is false instead of true')
        if not C: self.fail('C is false instead of true')
        if not sys: self.fail('sys is false instead of true')
        if not x: self.fail('x is false instead of true')

    def test_boolean_ops(self):
        # [FUNCTION DOC] Checks the logical operators or, and, not on small integers, failing
        # with a message if any evaluates to the wrong truth value.
        if 0 or 0: self.fail('0 or 0 is true instead of false')
        if 1 and 1: pass
        else: self.fail('1 and 1 is false instead of true')
        if not 1: self.fail('not 1 is true instead of false')

    def test_comparisons(self):
        # [FUNCTION DOC] Checks chained comparisons for ints, floats and strings (ordering and
        # equality chains) and the identity test for None.
        if 0 < 1 <= 1 == 1 >= 1 > 0 != 1: pass
        else: self.fail('int comparisons failed')
        if 0.0 < 1.0 <= 1.0 == 1.0 >= 1.0 > 0.0 != 1.0: pass
        else: self.fail('float comparisons failed')
        if '' < 'a' <= 'a' == 'a' < 'abc' < 'abd' < 'b': pass
        else: self.fail('string comparisons failed')
        if None is None: pass
        else: self.fail('identity test failed')

    def test_float_constructor(self):
        # [FUNCTION DOC] Checks that the float constructor raises ValueError for an
        # empty string and for strings with an embedded NUL character (also when
        # digit separators are present).
        self.assertRaises(ValueError, float, '')
        self.assertRaises(ValueError, float, '5\0')
        self.assertRaises(ValueError, float, '5_5\0')

    def test_zero_division(self):
        # [FUNCTION DOC] Checks that float and int true division, floor division and modulo by
        # zero each raise ZeroDivisionError.
        try: 5.0 / 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 / 0.0 didn't raise ZeroDivisionError")

        try: 5.0 // 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 // 0.0 didn't raise ZeroDivisionError")

        try: 5.0 % 0.0
        except ZeroDivisionError: pass
        else: self.fail("5.0 % 0.0 didn't raise ZeroDivisionError")

        try: 5 / 0
        except ZeroDivisionError: pass
        else: self.fail("5 / 0 didn't raise ZeroDivisionError")

        try: 5 // 0
        except ZeroDivisionError: pass
        else: self.fail("5 // 0 didn't raise ZeroDivisionError")

        try: 5 % 0
        except ZeroDivisionError: pass
        else: self.fail("5 % 0 didn't raise ZeroDivisionError")

    def test_numeric_types(self):
        # [FUNCTION DOC] Checks int/float cross-type equality, that int() and float()
        # without arguments return zero, and that int() truncates toward zero for
        # positive and negative floats.
        if 0 != 0.0 or 1 != 1.0 or -1 != -1.0:
            self.fail('int/float value not equal')
        # calling built-in types without argument must return 0
        if int() != 0: self.fail('int() does not return 0')
        if float() != 0.0: self.fail('float() does not return 0.0')
        if int(1.9) == 1 == int(1.1) and int(-1.1) == -1 == int(-1.9): pass
        else: self.fail('int() does not round properly')
        if float(1) == 1.0 and float(-1) == -1.0 and float(0) == 0.0: pass
        else: self.fail('float() does not work properly')

    def test_float_to_string(self):
        # [FUNCTION DOC] Checks exponent formatting in scientific notation for floats: every
        # two digit exponent from -99 to 99 and sample three digit exponents, via both the
        # format method and percent formatting, plus the general-format percent codes with
        # and without alternate form.
        def test(f, result):
            # [FUNCTION DOC] HELPER: Asserts that scientific-notation formatting of the
            # float f, through its format method and through percent formatting, equals
            # the expected string result.
            self.assertEqual(f.__format__('e'), result)
            self.assertEqual('%e' % f, result)

        # test all 2 digit exponents, both with __format__ and with
        #  '%' formatting
        for i in range(-99, 100):
            test(float('1.5e'+str(i)), '1.500000e{0:+03d}'.format(i))

        # test some 3 digit exponents
        self.assertEqual(1.5e100.__format__('e'), '1.500000e+100')
        self.assertEqual('%e' % 1.5e100, '1.500000e+100')

        self.assertEqual(1.5e101.__format__('e'), '1.500000e+101')
        self.assertEqual('%e' % 1.5e101, '1.500000e+101')

        self.assertEqual(1.5e-100.__format__('e'), '1.500000e-100')
        self.assertEqual('%e' % 1.5e-100, '1.500000e-100')

        self.assertEqual(1.5e-101.__format__('e'), '1.500000e-101')
        self.assertEqual('%e' % 1.5e-101, '1.500000e-101')

        self.assertEqual('%g' % 1.0, '1')
        self.assertEqual('%#g' % 1.0, '1.00000')

    def test_normal_integers(self):
        # [FUNCTION DOC] Checks integer behavior: small-int caching of 256, basic sums and
        # comparisons, multiplication commutativity, that products near the machine word limits
        # stay the single int type, that sys.maxsize arithmetic yields ints, and that negative
        # shift counts raise ValueError.
        # Ensure the first 256 integers are shared
        a = 256
        b = 128*2
        if a is not b: self.fail('256 is not shared')
        if 12 + 24 != 36: self.fail('int op')
        if 12 + (-24) != -12: self.fail('int op')
        if (-12) + 24 != 12: self.fail('int op')
        if (-12) + (-24) != -36: self.fail('int op')
        if not 12 < 24: self.fail('int op')
        if not -24 < -12: self.fail('int op')
        # Test for a particular bug in integer multiply
        xsize, ysize, zsize = 238, 356, 4
        if not (xsize*ysize*zsize == zsize*xsize*ysize == 338912):
            self.fail('int mul commutativity')
        # And another.
        m = -sys.maxsize - 1
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor
            prod = divisor * j
            if prod != m:
                self.fail("%r * %r == %r != %r" % (divisor, j, prod, m))
            if type(prod) is not int:
                self.fail("expected type(prod) to be int, not %r" %
                                   type(prod))
        # Check for unified integral type
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor - 1
            prod = divisor * j
            if type(prod) is not int:
                self.fail("expected type(%r) to be int, not %r" %
                                   (prod, type(prod)))
        # Check for unified integral type
        m = sys.maxsize
        for divisor in 1, 2, 4, 8, 16, 32:
            j = m // divisor + 1
            prod = divisor * j
            if type(prod) is not int:
                self.fail("expected type(%r) to be int, not %r" %
                                   (prod, type(prod)))

        x = sys.maxsize
        self.assertIsInstance(x + 1, int,
                              "(sys.maxsize + 1) should have returned int")
        self.assertIsInstance(-x - 1, int,
                              "(-sys.maxsize - 1) should have returned int")
        self.assertIsInstance(-x - 2, int,
                              "(-sys.maxsize - 2) should have returned int")

        try: 5 << -5
        except ValueError: pass
        else: self.fail('int negative shift <<')

        try: 5 >> -5
        except ValueError: pass
        else: self.fail('int negative shift >>')

    def test_floats(self):
        # [FUNCTION DOC] Checks basic float addition and ordering comparisons including negative
        # operands, failing the test on a wrong result.
        if 12.0 + 24.0 != 36.0: self.fail('float op')
        if 12.0 + (-24.0) != -12.0: self.fail('float op')
        if (-12.0) + 24.0 != 12.0: self.fail('float op')
        if (-12.0) + (-24.0) != -36.0: self.fail('float op')
        if not 12.0 < 24.0: self.fail('float op')
        if not -24.0 < -12.0: self.fail('float op')

    def test_strings(self):
        # [FUNCTION DOC] Checks string length, concatenation, repetition (including by zero),
        # min, max, membership, a long-string percent formatting case, and extended slicing with
        # positive/negative steps and out-of-range bounds.
        if len('') != 0: self.fail('len(\'\')')
        if len('a') != 1: self.fail('len(\'a\')')
        if len('abcdef') != 6: self.fail('len(\'abcdef\')')
        if 'xyz' + 'abcde' != 'xyzabcde': self.fail('string concatenation')
        if 'xyz'*3 != 'xyzxyzxyz': self.fail('string repetition *3')
        if 0*'abcde' != '': self.fail('string repetition 0*')
        if min('abc') != 'a' or max('abc') != 'c': self.fail('min/max string')
        if 'a' in 'abc' and 'b' in 'abc' and 'c' in 'abc' and 'd' not in 'abc': pass
        else: self.fail('in/not in string')
        x = 'x'*103
        if '%s!'%x != x+'!': self.fail('nasty string formatting bug')

        #extended slices for strings
        a = '0123456789'
        self.assertEqual(a[::], a)
        self.assertEqual(a[::2], '02468')
        self.assertEqual(a[1::2], '13579')
        self.assertEqual(a[::-1],'9876543210')
        self.assertEqual(a[::-2], '97531')
        self.assertEqual(a[3::-2], '31')
        self.assertEqual(a[-100:100:], a)
        self.assertEqual(a[100:-100:-1], a[::-1])
        self.assertEqual(a[-100:100:2], '02468')

    def test_type_function(self):
        # [FUNCTION DOC] Checks that calling type with two or four arguments raises TypeError.
        self.assertRaises(TypeError, type, 1, 2)
        self.assertRaises(TypeError, type, 1, 2, 3, 4)

    def test_int__format__(self):
        # [FUNCTION DOC] Table-driven check of integer format specs: decimal, character, sign
        # and alignment flags, hex/octal/binary with upper case and the alternate prefix form,
        # zero padding, thousands separators, fill/align characters, and very large ints. Also
        # checks the error cases (precision, sign or comma or hash with the character type,
        # non-string spec, unknown type letters raise ValueError or TypeError) and that float
        # type letters convert the int to a float first.
        def test(i, format_spec, result):
            # [FUNCTION DOC] HELPER: Asserts that i is an exact int and format_spec an exact
            # str, and that formatting i with format_spec gives the expected result string.
            # just make sure we have the unified type for integers
            self.assertIs(type(i), int)
            self.assertIs(type(format_spec), str)
            self.assertEqual(i.__format__(format_spec), result)

        test(123456789, 'd', '123456789')
        test(123456789, 'd', '123456789')

        test(1, 'c', '\01')

        # sign and aligning are interdependent
        test(1, "-", '1')
        test(-1, "-", '-1')
        test(1, "-3", '  1')
        test(-1, "-3", ' -1')
        test(1, "+3", ' +1')
        test(-1, "+3", ' -1')
        test(1, " 3", '  1')
        test(-1, " 3", ' -1')
        test(1, " ", ' 1')
        test(-1, " ", '-1')

        # hex
        test(3, "x", "3")
        test(3, "X", "3")
        test(1234, "x", "4d2")
        test(-1234, "x", "-4d2")
        test(1234, "8x", "     4d2")
        test(-1234, "8x", "    -4d2")
        test(1234, "x", "4d2")
        test(-1234, "x", "-4d2")
        test(-3, "x", "-3")
        test(-3, "X", "-3")
        test(int('be', 16), "x", "be")
        test(int('be', 16), "X", "BE")
        test(-int('be', 16), "x", "-be")
        test(-int('be', 16), "X", "-BE")

        # octal
        test(3, "o", "3")
        test(-3, "o", "-3")
        test(65, "o", "101")
        test(-65, "o", "-101")
        test(1234, "o", "2322")
        test(-1234, "o", "-2322")
        test(1234, "-o", "2322")
        test(-1234, "-o", "-2322")
        test(1234, " o", " 2322")
        test(-1234, " o", "-2322")
        test(1234, "+o", "+2322")
        test(-1234, "+o", "-2322")

        # binary
        test(3, "b", "11")
        test(-3, "b", "-11")
        test(1234, "b", "10011010010")
        test(-1234, "b", "-10011010010")
        test(1234, "-b", "10011010010")
        test(-1234, "-b", "-10011010010")
        test(1234, " b", " 10011010010")
        test(-1234, " b", "-10011010010")
        test(1234, "+b", "+10011010010")
        test(-1234, "+b", "-10011010010")

        # alternate (#) formatting
        test(0, "#b", '0b0')
        test(0, "-#b", '0b0')
        test(1, "-#b", '0b1')
        test(-1, "-#b", '-0b1')
        test(-1, "-#5b", ' -0b1')
        test(1, "+#5b", ' +0b1')
        test(100, "+#b", '+0b1100100')
        test(100, "#012b", '0b0001100100')
        test(-100, "#012b", '-0b001100100')

        test(0, "#o", '0o0')
        test(0, "-#o", '0o0')
        test(1, "-#o", '0o1')
        test(-1, "-#o", '-0o1')
        test(-1, "-#5o", ' -0o1')
        test(1, "+#5o", ' +0o1')
        test(100, "+#o", '+0o144')
        test(100, "#012o", '0o0000000144')
        test(-100, "#012o", '-0o000000144')

        test(0, "#x", '0x0')
        test(0, "-#x", '0x0')
        test(1, "-#x", '0x1')
        test(-1, "-#x", '-0x1')
        test(-1, "-#5x", ' -0x1')
        test(1, "+#5x", ' +0x1')
        test(100, "+#x", '+0x64')
        test(100, "#012x", '0x0000000064')
        test(-100, "#012x", '-0x000000064')
        test(123456, "#012x", '0x000001e240')
        test(-123456, "#012x", '-0x00001e240')

        test(0, "#X", '0X0')
        test(0, "-#X", '0X0')
        test(1, "-#X", '0X1')
        test(-1, "-#X", '-0X1')
        test(-1, "-#5X", ' -0X1')
        test(1, "+#5X", ' +0X1')
        test(100, "+#X", '+0X64')
        test(100, "#012X", '0X0000000064')
        test(-100, "#012X", '-0X000000064')
        test(123456, "#012X", '0X000001E240')
        test(-123456, "#012X", '-0X00001E240')

        test(123, ',', '123')
        test(-123, ',', '-123')
        test(1234, ',', '1,234')
        test(-1234, ',', '-1,234')
        test(123456, ',', '123,456')
        test(-123456, ',', '-123,456')
        test(1234567, ',', '1,234,567')
        test(-1234567, ',', '-1,234,567')

        # issue 5782, commas with no specifier type
        test(1234, '010,', '00,001,234')

        # Unified type for integers
        test(10**100, 'd', '1' + '0' * 100)
        test(10**100+100, 'd', '1' + '0' * 97 + '100')

        # make sure these are errors

        # precision disallowed
        self.assertRaises(ValueError, 3 .__format__, "1.3")
        # sign not allowed with 'c'
        self.assertRaises(ValueError, 3 .__format__, "+c")
        # format spec must be string
        self.assertRaises(TypeError, 3 .__format__, None)
        self.assertRaises(TypeError, 3 .__format__, 0)
        # can't have ',' with 'n'
        self.assertRaises(ValueError, 3 .__format__, ",n")
        # can't have ',' with 'c'
        self.assertRaises(ValueError, 3 .__format__, ",c")
        # can't have '#' with 'c'
        self.assertRaises(ValueError, 3 .__format__, "#c")

        # ensure that only int and float type specifiers work
        for format_spec in ([chr(x) for x in range(ord('a'), ord('z')+1)] +
                            [chr(x) for x in range(ord('A'), ord('Z')+1)]):
            if not format_spec in 'bcdoxXeEfFgGn%':
                self.assertRaises(ValueError, 0 .__format__, format_spec)
                self.assertRaises(ValueError, 1 .__format__, format_spec)
                self.assertRaises(ValueError, (-1) .__format__, format_spec)

        # ensure that float type specifiers work; format converts
        #  the int to a float
        for format_spec in 'eEfFgG%':
            for value in [0, 1, -1, 100, -100, 1234567890, -1234567890]:
                self.assertEqual(value.__format__(format_spec),
                                 float(value).__format__(format_spec))

        # Issue 6902
        test(123456, "0<20", '12345600000000000000')
        test(123456, "1<20", '12345611111111111111')
        test(123456, "*<20", '123456**************')
        test(123456, "0>20", '00000000000000123456')
        test(123456, "1>20", '11111111111111123456')
        test(123456, "*>20", '**************123456')
        test(123456, "0=20", '00000000000000123456')
        test(123456, "1=20", '11111111111111123456')
        test(123456, "*=20", '**************123456')

    @run_with_locale('LC_NUMERIC', 'en_US.UTF8', '')
    def test_float__format__locale(self):
        # [FUNCTION DOC] Under the en_US locale (skipped by the decorator if unavailable),
        # checks that the float format code n matches locale-aware grouping from the locale
        # module for a range of magnitudes and precisions.
        # test locale support for __format__ code 'n'

        for i in range(-10, 10):
            x = 1234567890.0 * (10.0 ** i)
            self.assertEqual(locale.format_string('%g', x, grouping=True), format(x, 'n'))
            self.assertEqual(locale.format_string('%.10g', x, grouping=True), format(x, '.10n'))

    @run_with_locale('LC_NUMERIC', 'en_US.UTF8', '')
    def test_int__format__locale(self):
        # [FUNCTION DOC] Under the en_US locale, checks that the integer format code n
        # matches locale-aware grouping for a descending series of large ints, and that
        # width padding with n gives the same length regardless of the value for left,
        # right and centered alignment.
        # test locale support for __format__ code 'n' for integers

        x = 123456789012345678901234567890
        for i in range(0, 30):
            self.assertEqual(locale.format_string('%d', x, grouping=True), format(x, 'n'))

            # move to the next integer to test
            x = x // 10

        rfmt = ">20n"
        lfmt = "<20n"
        cfmt = "^20n"
        for x in (1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890, 12345678900):
            self.assertEqual(len(format(0, rfmt)), len(format(x, rfmt)))
            self.assertEqual(len(format(0, lfmt)), len(format(x, lfmt)))
            self.assertEqual(len(format(0, cfmt)), len(format(x, cfmt)))

    def test_float__format__(self):
        # [FUNCTION DOC] Table-driven check of float format specs: f, e, g, E, G and percent
        # types, default and empty specs, sign flags, zero padding with and without commas, very
        # large values, alternate form (hash), and fill/align characters. Also asserts that
        # non-string specs raise TypeError and integer-only type codes raise ValueError.
        def test(f, format_spec, result):
            # [FUNCTION DOC] HELPER: Asserts that the float f formatted with format_spec, via
            # both the format method and the format builtin, equals the expected string result.
            self.assertEqual(f.__format__(format_spec), result)
            self.assertEqual(format(f, format_spec), result)

        test(0.0, 'f', '0.000000')

        # the default is 'g', except for empty format spec
        test(0.0, '', '0.0')
        test(0.01, '', '0.01')
        test(0.01, 'g', '0.01')

        # test for issue 3411
        test(1.23, '1', '1.23')
        test(-1.23, '1', '-1.23')
        test(1.23, '1g', '1.23')
        test(-1.23, '1g', '-1.23')

        test( 1.0, ' g', ' 1')
        test(-1.0, ' g', '-1')
        test( 1.0, '+g', '+1')
        test(-1.0, '+g', '-1')
        test(1.1234e200, 'g', '1.1234e+200')
        test(1.1234e200, 'G', '1.1234E+200')


        test(1.0, 'f', '1.000000')

        test(-1.0, 'f', '-1.000000')

        test( 1.0, ' f', ' 1.000000')
        test(-1.0, ' f', '-1.000000')
        test( 1.0, '+f', '+1.000000')
        test(-1.0, '+f', '-1.000000')

        # Python versions <= 3.0 switched from 'f' to 'g' formatting for
        # values larger than 1e50.  No longer.
        f = 1.1234e90
        for fmt in 'f', 'F':
            # don't do a direct equality check, since on some
            # platforms only the first few digits of dtoa
            # will be reliable
            result = f.__format__(fmt)
            self.assertEqual(len(result), 98)
            self.assertEqual(result[-7], '.')
            self.assertIn(result[:12], ('112340000000', '112339999999'))
        f = 1.1234e200
        for fmt in 'f', 'F':
            result = f.__format__(fmt)
            self.assertEqual(len(result), 208)
            self.assertEqual(result[-7], '.')
            self.assertIn(result[:12], ('112340000000', '112339999999'))


        test( 1.0, 'e', '1.000000e+00')
        test(-1.0, 'e', '-1.000000e+00')
        test( 1.0, 'E', '1.000000E+00')
        test(-1.0, 'E', '-1.000000E+00')
        test(1.1234e20, 'e', '1.123400e+20')
        test(1.1234e20, 'E', '1.123400E+20')

        # No format code means use g, but must have a decimal
        # and a number after the decimal.  This is tricky, because
        # a totally empty format specifier means something else.
        # So, just use a sign flag
        test(1e200, '+g', '+1e+200')
        test(1e200, '+', '+1e+200')

        test(1.1e200, '+g', '+1.1e+200')
        test(1.1e200, '+', '+1.1e+200')

        # 0 padding
        test(1234., '010f', '1234.000000')
        test(1234., '011f', '1234.000000')
        test(1234., '012f', '01234.000000')
        test(-1234., '011f', '-1234.000000')
        test(-1234., '012f', '-1234.000000')
        test(-1234., '013f', '-01234.000000')
        test(-1234.12341234, '013f', '-01234.123412')
        test(-123456.12341234, '011.2f', '-0123456.12')

        # issue 5782, commas with no specifier type
        test(1.2, '010,.2', '0,000,001.2')

        # 0 padding with commas
        test(1234., '011,f', '1,234.000000')
        test(1234., '012,f', '1,234.000000')
        test(1234., '013,f', '01,234.000000')
        test(-1234., '012,f', '-1,234.000000')
        test(-1234., '013,f', '-1,234.000000')
        test(-1234., '014,f', '-01,234.000000')
        test(-12345., '015,f', '-012,345.000000')
        test(-123456., '016,f', '-0,123,456.000000')
        test(-123456., '017,f', '-0,123,456.000000')
        test(-123456.12341234, '017,f', '-0,123,456.123412')
        test(-123456.12341234, '013,.2f', '-0,123,456.12')

        # % formatting
        test(-1.0, '%', '-100.000000%')

        # format spec must be string
        self.assertRaises(TypeError, 3.0.__format__, None)
        self.assertRaises(TypeError, 3.0.__format__, 0)

        # confirm format options expected to fail on floats, such as integer
        # presentation types
        for format_spec in 'sbcdoxX':
            self.assertRaises(ValueError, format, 0.0, format_spec)
            self.assertRaises(ValueError, format, 1.0, format_spec)
            self.assertRaises(ValueError, format, -1.0, format_spec)
            self.assertRaises(ValueError, format, 1e100, format_spec)
            self.assertRaises(ValueError, format, -1e100, format_spec)
            self.assertRaises(ValueError, format, 1e-100, format_spec)
            self.assertRaises(ValueError, format, -1e-100, format_spec)

        # Alternate float formatting
        test(1.0, '.0e', '1e+00')
        test(1.0, '#.0e', '1.e+00')
        test(1.0, '.0f', '1')
        test(1.0, '#.0f', '1.')
        test(1.1, 'g', '1.1')
        test(1.1, '#g', '1.10000')
        test(1.0, '.0%', '100%')
        test(1.0, '#.0%', '100.%')

        # Issue 7094: Alternate formatting (specified by #)
        test(1.0, '0e',  '1.000000e+00')
        test(1.0, '#0e', '1.000000e+00')
        test(1.0, '0f',  '1.000000' )
        test(1.0, '#0f', '1.000000')
        test(1.0, '.1e',  '1.0e+00')
        test(1.0, '#.1e', '1.0e+00')
        test(1.0, '.1f',  '1.0')
        test(1.0, '#.1f', '1.0')
        test(1.0, '.1%',  '100.0%')
        test(1.0, '#.1%', '100.0%')

        # Issue 6902
        test(12345.6, "0<20", '12345.60000000000000')
        test(12345.6, "1<20", '12345.61111111111111')
        test(12345.6, "*<20", '12345.6*************')
        test(12345.6, "0>20", '000000000000012345.6')
        test(12345.6, "1>20", '111111111111112345.6')
        test(12345.6, "*>20", '*************12345.6')
        test(12345.6, "0=20", '000000000000012345.6')
        test(12345.6, "1=20", '111111111111112345.6')
        test(12345.6, "*=20", '*************12345.6')

    def test_format_spec_errors(self):
        # [FUNCTION DOC] Checks that format raises ValueError for absurdly long width or
        # precision digit strings and for a comma used with the x, X, o, b, n or s type codes.
        # int, float, and string all share the same format spec
        # mini-language parser.

        # Check that we can't ask for too many digits. This is
        # probably a CPython specific test. It tries to put the width
        # into a C long.
        self.assertRaises(ValueError, format, 0, '1'*10000 + 'd')

        # Similar with the precision.
        self.assertRaises(ValueError, format, 0, '.' + '1'*10000 + 'd')

        # And may as well test both.
        self.assertRaises(ValueError, format, 0, '1'*1000 + '.' + '1'*10000 + 'd')

        # Make sure commas aren't allowed with various type codes
        for code in 'xXobns':
            self.assertRaises(ValueError, format, 0, ',' + code)

    def test_internal_sizes(self):
        # [FUNCTION DOC] Checks that object has a positive basic size and
        # tuple a positive item size.
        self.assertGreater(object.__basicsize__, 0)
        self.assertGreater(tuple.__itemsize__, 0)

    def test_slot_wrapper_types(self):
        # [FUNCTION DOC] Checks that slot wrappers of object and int (init, str, lt) are
        # instances of types.WrapperDescriptorType.
        self.assertIsInstance(object.__init__, types.WrapperDescriptorType)
        self.assertIsInstance(object.__str__, types.WrapperDescriptorType)
        self.assertIsInstance(object.__lt__, types.WrapperDescriptorType)
        self.assertIsInstance(int.__lt__, types.WrapperDescriptorType)

    @unittest.skipIf(MISSING_C_DOCSTRINGS,
                     "Signature information for builtins requires docstrings")
    def test_dunder_get_signature(self):
        # [FUNCTION DOC] Skipped when C docstrings are missing. Checks that the
        # signature of the descriptor get method has the parameters instance and owner
        # and that owner defaults to None.
        sig = inspect.signature(object.__init__.__get__)
        self.assertEqual(list(sig.parameters), ["instance", "owner"])
        # gh-93021: Second parameter is optional
        self.assertIs(sig.parameters["owner"].default, None)

    def test_method_wrapper_types(self):
        # [FUNCTION DOC] Checks that bound slot methods of an object and an int are instances
        # of types.MethodWrapperType.
        self.assertIsInstance(object().__init__, types.MethodWrapperType)
        self.assertIsInstance(object().__str__, types.MethodWrapperType)
        self.assertIsInstance(object().__lt__, types.MethodWrapperType)
        self.assertIsInstance((42).__lt__, types.MethodWrapperType)

    def test_method_descriptor_types(self):
        # [FUNCTION DOC] Checks the types of unbound and bound builtin methods and class methods
        # against types.MethodDescriptorType, BuiltinMethodType and ClassMethodDescriptorType.
        self.assertIsInstance(str.join, types.MethodDescriptorType)
        self.assertIsInstance(list.append, types.MethodDescriptorType)
        self.assertIsInstance(''.join, types.BuiltinMethodType)
        self.assertIsInstance([].append, types.BuiltinMethodType)

        self.assertIsInstance(int.__dict__['from_bytes'], types.ClassMethodDescriptorType)
        self.assertIsInstance(int.from_bytes, types.BuiltinMethodType)
        self.assertIsInstance(int.__new__, types.BuiltinMethodType)

    def test_method_descriptor_crash(self):
        # [FUNCTION DOC] Regression test for gh-132747: for several C-implemented methods paired
        # with matching instances, calling the default get with only an instance (owner omitted)
        # must return a builtin method object without crashing. Uses subTest per pair.
        # gh-132747: The default __get__() implementation in C was unable
        # to handle a second argument of None when called from Python
        import _io
        import io
        import _queue

        to_check = [
            # (method, instance)
            (_io._TextIOBase.read, io.StringIO()),
            (_queue.SimpleQueue.put, _queue.SimpleQueue()),
            (str.capitalize, "nobody expects the spanish inquisition")
        ]

        for method, instance in to_check:
            with self.subTest(method=method, instance=instance):
                bound = method.__get__(instance)
                self.assertIsInstance(bound, types.BuiltinMethodType)

    def test_ellipsis_type(self):
        # [FUNCTION DOC] Checks that Ellipsis is an instance of types.EllipsisType.
        self.assertIsInstance(Ellipsis, types.EllipsisType)

    def test_notimplemented_type(self):
        # [FUNCTION DOC] Checks that NotImplemented is an instance of types.NotImplementedType.
        self.assertIsInstance(NotImplemented, types.NotImplementedType)

    def test_none_type(self):
        # [FUNCTION DOC] Checks that None is an instance of types.NoneType.
        self.assertIsInstance(None, types.NoneType)

    def test_traceback_and_frame_types(self):
        # [FUNCTION DOC] Raises and catches an OSError, then checks that its traceback and the
        # traceback's frame are instances of types.TracebackType and types.FrameType.
        try:
            raise OSError
        except OSError as e:
            exc = e
        self.assertIsInstance(exc.__traceback__, types.TracebackType)
        self.assertIsInstance(exc.__traceback__.tb_frame, types.FrameType)

    def test_capsule_type(self):
        # [FUNCTION DOC] Checks that the datetime C API capsule is an
        # instance of types.CapsuleType.
        self.assertIsInstance(_datetime.datetime_CAPI, types.CapsuleType)

    def test_call_unbound_crash(self):
        # [FUNCTION DOC] Regression test for GH-131998: runs a small program in a fresh
        # interpreter (assert_python_ok) that calls a specialized function three times with a
        # list instance and then unbound with the list type; the program must exit successfully
        # (the TypeError is swallowed) instead of crashing.
        # GH-131998: The specialized instruction would get tricked into dereferencing
        # a bound "self" that didn't exist if subsequently called unbound.
        code = """if True:

        def call(part):
            [] + ([] + [])
            part.pop()

        for _ in range(3):
            call(['a'])
        try:
            call(list)
        except TypeError:
            pass
        """
        assert_python_ok("-c", code)


class UnionTests(CPythonTestCase):
    # [CLASS DOC] Tests of the union operator (vertical bar) on types: construction, flattening
    # and de-duplication, hashing, isinstance/issubclass, parameter substitution,
    # pickling/copying, repr, and interplay with typing constructs and generic aliases.

    def test_or_types_operator(self):
        # [FUNCTION DOC] Checks the union operator across builtins, None, typing aliases
        # and generics: equality with typing.Union, flattening of nested unions,
        # de-duplication, symmetry of order, TypeError for non-type operands, unordered
        # comparisons raising TypeError, and tolerance of a typing.Union whose args
        # attribute is a list rather than a tuple.
        self.assertEqual(int | str, typing.Union[int, str])
        self.assertNotEqual(int | list, typing.Union[int, str])
        self.assertEqual(str | int, typing.Union[int, str])
        self.assertEqual(int | None, typing.Union[int, None])
        self.assertEqual(None | int, typing.Union[int, None])
        self.assertEqual(int | type(None), int | None)
        self.assertEqual(type(None) | int, None | int)
        self.assertEqual(int | str | list, typing.Union[int, str, list])
        self.assertEqual(int | (str | list), typing.Union[int, str, list])
        self.assertEqual(str | (int | list), typing.Union[int, str, list])
        self.assertEqual(typing.List | typing.Tuple, typing.Union[typing.List, typing.Tuple])
        self.assertEqual(typing.List[int] | typing.Tuple[int], typing.Union[typing.List[int], typing.Tuple[int]])
        self.assertEqual(typing.List[int] | None, typing.Union[typing.List[int], None])
        self.assertEqual(None | typing.List[int], typing.Union[None, typing.List[int]])
        self.assertEqual(str | float | int | complex | int, (int | str) | (float | complex))
        self.assertEqual(typing.Union[str, int, typing.List[int]], str | int | typing.List[int])
        self.assertIs(int | int, int)
        self.assertEqual(
            BaseException |
            bool |
            bytes |
            complex |
            float |
            int |
            list |
            map |
            set,
            typing.Union[
                BaseException,
                bool,
                bytes,
                complex,
                float,
                int,
                list,
                map,
                set,
            ])
        with self.assertRaises(TypeError):
            int | 3
        with self.assertRaises(TypeError):
            3 | int
        with self.assertRaises(TypeError):
            Example() | int
        x = int | str
        self.assertEqual(x, int | str)
        self.assertEqual(x, str | int)
        self.assertNotEqual(x, {})  # should not raise exception
        with self.assertRaises(TypeError):
            x < x
        with self.assertRaises(TypeError):
            x <= x
        y = typing.Union[str, int]
        with self.assertRaises(TypeError):
            x < y
        y = int | bool
        with self.assertRaises(TypeError):
            x < y
        # Check that we don't crash if typing.Union does not have a tuple in __args__
        y = typing.Union[str, int]
        y.__args__ = [str, int]
        self.assertEqual(x, y)

    def test_hash(self):
        # [FUNCTION DOC] Checks that a union hashes the same regardless of operand order and
        # equals the hash of the equivalent typing.Union.
        self.assertEqual(hash(int | str), hash(str | int))
        self.assertEqual(hash(int | str), hash(typing.Union[int, str]))

    def test_union_of_unhashable(self):
        # [FUNCTION DOC] Checks unions of classes whose metaclass disables hashing: construction
        # works and exposes both args, but hashing any such union raises TypeError.
        class UnhashableMeta(type):
            # [CLASS DOC] HELPER: Metaclass that sets the hash slot to None so
            # its classes are unhashable.
            __hash__ = None

        # [CLASS DOC] HELPER: Empty class with the unhashable metaclass.
        class A(metaclass=UnhashableMeta): ...
        # [CLASS DOC] HELPER: Empty class with the unhashable metaclass.
        class B(metaclass=UnhashableMeta): ...

        self.assertEqual((A | B).__args__, (A, B))
        union1 = A | B
        with self.assertRaises(TypeError):
            hash(union1)

        union2 = int | B
        with self.assertRaises(TypeError):
            hash(union2)

        union3 = A | int
        with self.assertRaises(TypeError):
            hash(union3)

    def test_instancecheck_and_subclasscheck(self):
        # [FUNCTION DOC] For built-in unions and typing.Union objects, checks
        # isinstance and issubclass for member types (including bool as an int
        # subclass), None, and abstract base classes.
        for x in (int | str, typing.Union[int, str]):
            with self.subTest(x=x):
                self.assertIsInstance(1, x)
                self.assertIsInstance(True, x)
                self.assertIsInstance('a', x)
                self.assertNotIsInstance(None, x)
                self.assertTrue(issubclass(int, x))
                self.assertTrue(issubclass(bool, x))
                self.assertTrue(issubclass(str, x))
                self.assertFalse(issubclass(type(None), x))

        for x in (int | None, typing.Union[int, None]):
            with self.subTest(x=x):
                self.assertIsInstance(None, x)
                self.assertTrue(issubclass(type(None), x))

        for x in (
            int | collections.abc.Mapping,
            typing.Union[int, collections.abc.Mapping],
        ):
            with self.subTest(x=x):
                self.assertIsInstance({}, x)
                self.assertNotIsInstance((), x)
                self.assertTrue(issubclass(dict, x))
                self.assertFalse(issubclass(list, x))

    def test_instancecheck_and_subclasscheck_order(self):
        # [FUNCTION DOC] Checks evaluation order with a type variable: unions where a real type
        # comes before the TypeVar answer isinstance/issubclass, while those with the TypeVar
        # first (or for types matching neither) raise TypeError.
        T = typing.TypeVar('T')

        will_resolve = (
            int | T,
            typing.Union[int, T],
        )
        for x in will_resolve:
            with self.subTest(x=x):
                self.assertIsInstance(1, x)
                self.assertTrue(issubclass(int, x))

        wont_resolve = (
            T | int,
            typing.Union[T, int],
        )
        for x in wont_resolve:
            with self.subTest(x=x):
                with self.assertRaises(TypeError):
                    issubclass(int, x)
                with self.assertRaises(TypeError):
                    isinstance(1, x)

        for x in (*will_resolve, *wont_resolve):
            with self.subTest(x=x):
                with self.assertRaises(TypeError):
                    issubclass(object, x)
                with self.assertRaises(TypeError):
                    isinstance(object(), x)

    def test_bad_instancecheck(self):
        # [FUNCTION DOC] Checks that an error raised by a member metaclass instance check
        # propagates, while earlier matching members still succeed.
        class BadMeta(type):
            # [CLASS DOC] HELPER: Metaclass whose instance check always divides by zero.
            def __instancecheck__(cls, inst):
                # [FUNCTION DOC] HELPER: Raises ZeroDivisionError unconditionally.
                1/0
        x = int | BadMeta('A', (), {})
        self.assertTrue(isinstance(1, x))
        self.assertRaises(ZeroDivisionError, isinstance, [], x)

    def test_bad_subclasscheck(self):
        # [FUNCTION DOC] Same as the instance check test but for subclass checks: a
        # member metaclass subclass check error propagates as ZeroDivisionError,
        # earlier matches still succeed.
        class BadMeta(type):
            # [CLASS DOC] HELPER: Metaclass whose subclass check always divides by zero.
            def __subclasscheck__(cls, sub):
                # [FUNCTION DOC] HELPER: Raises ZeroDivisionError unconditionally.
                1/0
        x = int | BadMeta('A', (), {})
        self.assertTrue(issubclass(int, x))
        self.assertRaises(ZeroDivisionError, issubclass, list, x)

    def test_or_type_operator_with_TypeVar(self):
        # [FUNCTION DOC] Checks that a TypeVar combined with a class via the union operator
        # equals the typing.Union form in both orders, and that substituting a type
        # variable yields that type.
        TV = typing.TypeVar('T')
        self.assertEqual(TV | str, typing.Union[TV, str])
        self.assertEqual(str | TV, typing.Union[str, TV])
        self.assertIs((int | TV)[int], int)
        self.assertIs((TV | int)[int], int)

    def test_union_args(self):
        # [FUNCTION DOC] Checks the args tuple of many unions: flattening, order of first
        # appearance, de-duplication, mixing with typing.Union, and None placed on either side
        # of various argument kinds (generic aliases, callables, hashables, TypeVars).
        def check(arg, expected):
            # [FUNCTION DOC] HELPER: Clears typing caches then asserts that
            # arg.__args__ equals expected.
            clear_typing_caches()
            self.assertEqual(arg.__args__, expected)

        check(int | str, (int, str))
        check((int | str) | list, (int, str, list))
        check(int | (str | list), (int, str, list))
        check((int | str) | int, (int, str))
        check(int | (str | int), (int, str))
        check((int | str) | (str | int), (int, str))
        check(typing.Union[int, str] | list, (int, str, list))
        check(int | typing.Union[str, list], (int, str, list))
        check((int | str) | (list | int), (int, str, list))
        check((int | str) | typing.Union[list, int], (int, str, list))
        check(typing.Union[int, str] | (list | int), (int, str, list))
        check((str | int) | (int | list), (str, int, list))
        check((str | int) | typing.Union[int, list], (str, int, list))
        check(typing.Union[str, int] | (int | list), (str, int, list))
        check(int | type(None), (int, type(None)))
        check(type(None) | int, (type(None), int))

        args = (int, list[int], typing.List[int],
                typing.Tuple[int, int], typing.Callable[[int], int],
                typing.Hashable, typing.TypeVar('T'))
        for x in args:
            with self.subTest(x):
                check(x | None, (x, type(None)))
                check(None | x, (type(None), x))

    def test_union_parameter_chaining(self):
        # [FUNCTION DOC] Checks that type parameters propagate through unions and
        # generic aliases: substitution on a union, parameters tuple of nested generics,
        # and partial substitution.
        T = typing.TypeVar("T")
        S = typing.TypeVar("S")

        self.assertEqual((float | list[T])[int], float | list[int])
        self.assertEqual(list[int | list[T]].__parameters__, (T,))
        self.assertEqual(list[int | list[T]][str], list[int | list[str]])
        self.assertEqual((list[T] | list[S]).__parameters__, (T, S))
        self.assertEqual((list[T] | list[S])[int, T], list[int] | list[T])
        self.assertEqual((list[T] | list[S])[int, int], list[int])

    def test_union_parameter_substitution(self):
        # [FUNCTION DOC] Checks substituting a TypeVar inside a three-way union with many kinds
        # of arguments (builtins, typing aliases, callables, literals, other unions, NewType,
        # another TypeVar), comparing result and (when typed) the result type.
        def eq(actual, expected, typed=True):
            # [FUNCTION DOC] HELPER: Asserts actual equals expected and, if typed is true, that
            # both have exactly the same type.
            self.assertEqual(actual, expected)
            if typed:
                self.assertIs(type(actual), type(expected))

        T = typing.TypeVar('T')
        S = typing.TypeVar('S')
        NT = typing.NewType('NT', str)
        x = int | T | bytes

        eq(x[str], int | str | bytes, typed=False)
        eq(x[list[int]], int | list[int] | bytes, typed=False)
        eq(x[typing.List], int | typing.List | bytes)
        eq(x[typing.List[int]], int | typing.List[int] | bytes)
        eq(x[typing.Hashable], int | typing.Hashable | bytes)
        eq(x[collections.abc.Hashable],
           int | collections.abc.Hashable | bytes, typed=False)
        eq(x[typing.Callable[[int], str]],
           int | typing.Callable[[int], str] | bytes)
        eq(x[collections.abc.Callable[[int], str]],
           int | collections.abc.Callable[[int], str] | bytes, typed=False)
        eq(x[typing.Tuple[int, str]], int | typing.Tuple[int, str] | bytes)
        eq(x[typing.Literal['none']], int | typing.Literal['none'] | bytes)
        eq(x[str | list], int | str | list | bytes, typed=False)
        eq(x[typing.Union[str, list]], typing.Union[int, str, list, bytes])
        eq(x[str | int], int | str | bytes, typed=False)
        eq(x[typing.Union[str, int]], typing.Union[int, str, bytes])
        eq(x[NT], int | NT | bytes)
        eq(x[S], int | S | bytes)

    def test_union_pickle(self):
        # [FUNCTION DOC] Checks that a union containing a generic alias over a TypeVar
        # round-trips through pickle at every protocol with equal args and parameters.
        orig = list[T] | int
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            s = pickle.dumps(orig, proto)
            loaded = pickle.loads(s)
            self.assertEqual(loaded, orig)
            self.assertEqual(loaded.__args__, orig.__args__)
            self.assertEqual(loaded.__parameters__, orig.__parameters__)

    def test_union_copy(self):
        # [FUNCTION DOC] Checks that shallow and deep copies of a union keep
        # equality, args and parameters.
        orig = list[T] | int
        for copied in (copy.copy(orig), copy.deepcopy(orig)):
            self.assertEqual(copied, orig)
            self.assertEqual(copied.__args__, orig.__args__)
            self.assertEqual(copied.__parameters__, orig.__parameters__)

    def test_union_parameter_substitution_errors(self):
        # [FUNCTION DOC] Checks that substituting with too many arguments into a
        # one-parameter union raises TypeError.
        T = typing.TypeVar("T")
        x = int | T
        with self.assertRaises(TypeError):
            x[int, str]

    def test_or_type_operator_with_forward(self):
        # [FUNCTION DOC] Checks unions mixing a TypeVar with a string forward reference in both
        # orders: after substitution, the resolved type hints of annotated functions give the
        # substituted type and the real Forward class.
        T = typing.TypeVar('T')
        ForwardAfter = T | 'Forward'
        ForwardBefore = 'Forward' | T
        # [FUNCTION DOC] HELPER: Placeholder function annotated with the
        # forward-after union, no body.
        def forward_after(x: ForwardAfter[int]) -> None: ...
        # [FUNCTION DOC] HELPER: Placeholder function annotated with the
        # forward-before union, no body.
        def forward_before(x: ForwardBefore[int]) -> None: ...
        self.assertEqual(typing.get_args(typing.get_type_hints(forward_after)['x']),
                         (int, Forward))
        self.assertEqual(typing.get_args(typing.get_type_hints(forward_before)['x']),
                         (int, Forward))

    def test_or_type_operator_with_Protocol(self):
        # [FUNCTION DOC] Checks that a Protocol class combined with a class by the union
        # operator equals the typing.Union of the two.
        class Proto(typing.Protocol):
            # [CLASS DOC] HELPER: Minimal Protocol with one method returning int.
            def meth(self) -> int:
                # [FUNCTION DOC] HELPER: Protocol member declaration with no implementation.
                ...
        self.assertEqual(Proto | str, typing.Union[Proto, str])

    def test_or_type_operator_with_Alias(self):
        # [FUNCTION DOC] Checks union with bare and typing alias list
        # types against a plain class.
        self.assertEqual(list | str, typing.Union[list, str])
        self.assertEqual(typing.List | str, typing.Union[typing.List, str])

    def test_or_type_operator_with_NamedTuple(self):
        # [FUNCTION DOC] Checks union of a namedtuple class with a class.
        NT = namedtuple('A', ['B', 'C', 'D'])
        self.assertEqual(NT | str, typing.Union[NT, str])

    def test_or_type_operator_with_TypedDict(self):
        # [FUNCTION DOC] Checks union of a TypedDict class with a class.
        class Point2D(typing.TypedDict):
            # [CLASS DOC] HELPER: TypedDict with integer fields x and y
            # and a string field label.
            x: int
            y: int
            label: str
        self.assertEqual(Point2D | str, typing.Union[Point2D, str])

    def test_or_type_operator_with_NewType(self):
        # [FUNCTION DOC] Checks union of a NewType with a class.
        UserId = typing.NewType('UserId', int)
        self.assertEqual(UserId | str, typing.Union[UserId, str])

    def test_or_type_operator_with_IO(self):
        # [FUNCTION DOC] Checks union of typing.IO with a class.
        self.assertEqual(typing.IO | str, typing.Union[typing.IO, str])

    def test_or_type_operator_with_SpecialForm(self):
        # [FUNCTION DOC] Checks union with special forms: Any, NoReturn, Optional (which
        # flattens to include None) and an existing typing.Union.
        self.assertEqual(typing.Any | str, typing.Union[typing.Any, str])
        self.assertEqual(typing.NoReturn | str, typing.Union[typing.NoReturn, str])
        self.assertEqual(typing.Optional[int] | str, typing.Union[typing.Optional[int], str])
        self.assertEqual(typing.Optional[int] | str, typing.Union[int, str, None])
        self.assertEqual(typing.Union[int, bool] | str, typing.Union[int, bool, str])

    def test_or_type_operator_with_Literal(self):
        # [FUNCTION DOC] Checks union of Literal types: argument tuples preserve order and keep
        # distinct values that compare equal (0 vs False, 1 vs True, int vs IntEnum), while
        # truly identical Literals collapse.
        Literal = typing.Literal
        self.assertEqual((Literal[1] | Literal[2]).__args__,
                         (Literal[1], Literal[2]))

        self.assertEqual((Literal[0] | Literal[False]).__args__,
                         (Literal[0], Literal[False]))
        self.assertEqual((Literal[1] | Literal[True]).__args__,
                         (Literal[1], Literal[True]))

        self.assertEqual(Literal[1] | Literal[1], Literal[1])
        self.assertEqual(Literal['a'] | Literal['a'], Literal['a'])

        import enum
        class Ints(enum.IntEnum):
            # [CLASS DOC] HELPER: Integer enumeration with members A=0 and
            # B=1 for the Literal tests.
            A = 0
            B = 1

        self.assertEqual(Literal[Ints.A] | Literal[Ints.A], Literal[Ints.A])
        self.assertEqual(Literal[Ints.B] | Literal[Ints.B], Literal[Ints.B])

        self.assertEqual((Literal[Ints.B] | Literal[Ints.A]).__args__,
                         (Literal[Ints.B], Literal[Ints.A]))

        self.assertEqual((Literal[0] | Literal[Ints.A]).__args__,
                         (Literal[0], Literal[Ints.A]))
        self.assertEqual((Literal[1] | Literal[Ints.B]).__args__,
                         (Literal[1], Literal[Ints.B]))

    def test_or_type_repr(self):
        # [FUNCTION DOC] Checks the text form of unions: members joined by a vertical
        # bar, None shown as None, nested unions flattened, and generic alias members
        # shown with their arguments.
        self.assertEqual(repr(int | str), "int | str")
        self.assertEqual(repr((int | str) | list), "int | str | list")
        self.assertEqual(repr(int | (str | list)), "int | str | list")
        self.assertEqual(repr(int | None), "int | None")
        self.assertEqual(repr(int | type(None)), "int | None")
        self.assertEqual(repr(int | typing.GenericAlias(list, int)), "int | list[int]")

    def test_or_type_operator_with_genericalias(self):
        # [FUNCTION DOC] Checks unions of generic aliases (including a user GenericAlias
        # subclass): equality with typing.Union, de-duplication, order-insensitivity,
        # repr, propagation of ZeroDivisionError from a member whose equality raises,
        # and TypeError when instance or subclass checks are done against unions
        # containing parameterized generics.
        a = list[int]
        b = list[str]
        c = dict[float, str]
        # [CLASS DOC] HELPER: Empty subclass of types.GenericAlias.
        class SubClass(types.GenericAlias): ...
        d = SubClass(list, float)
        # equivalence with typing.Union
        self.assertEqual(a | b | c | d, typing.Union[a, b, c, d])
        # de-duplicate
        self.assertEqual(a | c | b | b | a | c | d | d, a | b | c | d)
        # order shouldn't matter
        self.assertEqual(a | b | d, b | a | d)
        self.assertEqual(repr(a | b | c | d),
                         "list[int] | list[str] | dict[float, str] | list[float]")

        class BadType(type):
            # [CLASS DOC] HELPER: Metaclass whose equality method always divides by zero.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Raises ZeroDivisionError unconditionally.
                return 1 / 0

        bt = BadType('bt', (), {})
        # Comparison should fail and errors should propagate out for bad types.
        with self.assertRaises(ZeroDivisionError):
            list[int] | list[bt]

        union_ga = (list[str] | int, collections.abc.Callable[..., str] | int,
                    d | int)
        # Raise error when isinstance(type, genericalias | type)
        for type_ in union_ga:
            with self.subTest(f"check isinstance/issubclass is invalid for {type_}"):
                with self.assertRaises(TypeError):
                    isinstance(1, type_)
                with self.assertRaises(TypeError):
                    issubclass(int, type_)

    def test_or_type_operator_with_bad_module(self):
        # [FUNCTION DOC] Regression test for issue 44483: builds metaclass-based fake TypeVar
        # and special-form classes whose module attribute raises ZeroDivisionError, and
        # checks that or-ing a string with their instances raises TypeError or
        # ZeroDivisionError without crashing.
        class BadMeta(type):
            # [CLASS DOC] HELPER: Metaclass named TypeVar whose module property
            # raises ZeroDivisionError.
            __qualname__ = 'TypeVar'
            @property
            def __module__(self):
                # [FUNCTION DOC] HELPER: Property getter that raises ZeroDivisionError.
                1 / 0
        TypeVar = BadMeta('TypeVar', (), {})
        _SpecialForm = BadMeta('_SpecialForm', (), {})
        # Crashes in Issue44483
        with self.assertRaises((TypeError, ZeroDivisionError)):
            str | TypeVar()
        with self.assertRaises((TypeError, ZeroDivisionError)):
            str | _SpecialForm()

    @cpython_only
    def test_or_type_operator_reference_cycle(self):
        # [FUNCTION DOC] CPython-only. Skips unless the interpreter has a total reference count
        # (debug build). Creates 30 unions holding a TypeVar that refers back to the union,
        # deletes them, runs the garbage collector and asserts the reference count grew by at
        # most 15, detecting leaks.
        if not hasattr(sys, 'gettotalrefcount'):
            self.skipTest('Cannot get total reference count.')
        gc.collect()
        before = sys.gettotalrefcount()
        for _ in range(30):
            T = typing.TypeVar('T')
            U = int | list[T]
            T.blah = U
            del T
            del U
        gc.collect()
        leeway = 15
        self.assertLessEqual(sys.gettotalrefcount() - before, leeway,
                             msg='Check for union reference leak.')


class MappingProxyTests(CPythonTestCase):
    # [CLASS DOC] Tests of types.MappingProxyType, a read-only view over a mapping. The class
    # attribute mappingproxy is the type under test, so subclasses can swap in another factory.
    mappingproxy = types.MappingProxyType

    def test_constructor(self):
        # [FUNCTION DOC] Checks that the proxy can wrap dicts, dict subclasses and ChainMap
        # (and compares equal to them), and that wrapping non-mappings such as an int, tuple
        # or list raises TypeError.
        class userdict(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        mapping = {'x': 1, 'y': 2}
        self.assertEqual(self.mappingproxy(mapping), mapping)
        mapping = userdict(x=1, y=2)
        self.assertEqual(self.mappingproxy(mapping), mapping)
        mapping = collections.ChainMap({'x': 1}, {'y': 2})
        self.assertEqual(self.mappingproxy(mapping), mapping)

        self.assertRaises(TypeError, self.mappingproxy, 10)
        self.assertRaises(TypeError, self.mappingproxy, ("a", "tuple"))
        self.assertRaises(TypeError, self.mappingproxy, ["a", "list"])

    def test_methods(self):
        # [FUNCTION DOC] Checks the exact set of public and special attributes the proxy
        # adds over the base object (item access, iteration, length, union operators,
        # reversed, copy, get, views).
        attrs = set(dir(self.mappingproxy({}))) - set(dir(object()))
        self.assertEqual(attrs, {
             '__contains__',
             '__getitem__',
             '__class_getitem__',
             '__ior__',
             '__iter__',
             '__len__',
             '__or__',
             '__reversed__',
             '__ror__',
             'copy',
             'get',
             'items',
             'keys',
             'values',
        })

    def test_get(self):
        # [FUNCTION DOC] Checks item lookup and get with and without defaults, and
        # KeyError for a missing key.
        view = self.mappingproxy({'a': 'A', 'b': 'B'})
        self.assertEqual(view['a'], 'A')
        self.assertEqual(view['b'], 'B')
        self.assertRaises(KeyError, view.__getitem__, 'xxx')
        self.assertEqual(view.get('a'), 'A')
        self.assertIsNone(view.get('xxx'))
        self.assertEqual(view.get('xxx', 42), 42)

    def test_missing(self):
        # [FUNCTION DOC] Checks that the proxy honors a missing-key hook of the wrapped dict
        # subclass for item access but that get and membership do not use it.
        class dictmissing(dict):
            # [CLASS DOC] HELPER: Dict subclass whose missing-key hook
            # returns a text naming the key.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Returns the text "missing=" followed by the key.
                return "missing=%s" % key

        view = self.mappingproxy(dictmissing(x=1))
        self.assertEqual(view['x'], 1)
        self.assertEqual(view['y'], 'missing=y')
        self.assertEqual(view.get('x'), 1)
        self.assertEqual(view.get('y'), None)
        self.assertEqual(view.get('y', 42), 42)
        self.assertTrue('x' in view)
        self.assertFalse('y' in view)

    def test_customdict(self):
        # [FUNCTION DOC] Checks that the proxy delegates every operation to the wrapped object's
        # own overrides (membership, item access, iteration, length, copy, get, keys, items,
        # values) rather than to the underlying dict storage.
        class customdict(dict):
            # [CLASS DOC] HELPER: Dict subclass overriding membership, iteration,
            # length, copy, keys, items, values, item access and get with recognizable
            # constant or tagged results.
            def __contains__(self, key):
                # [FUNCTION DOC] HELPER: True for the key "magic", otherwise defers
                # to the dict implementation.
                if key == 'magic':
                    return True
                else:
                    return dict.__contains__(self, key)

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over a single fixed key.
                return iter(('iter',))

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the constant 500.
                return 500

            def copy(self):
                # [FUNCTION DOC] HELPER: Returns the constant text "copy".
                return 'copy'

            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the constant text "keys".
                return 'keys'

            def items(self):
                # [FUNCTION DOC] HELPER: Returns the constant text "items".
                return 'items'

            def values(self):
                # [FUNCTION DOC] HELPER: Returns the constant text "values".
                return 'values'

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the stored value prefixed with "getitem=".
                return "getitem=%s" % dict.__getitem__(self, key)

            def get(self, key, default=None):
                # [FUNCTION DOC] HELPER: Returns the stored value (or a description of the
                # default when the key is absent) prefixed with "get=".
                return "get=%s" % dict.get(self, key, 'default=%r' % default)

        custom = customdict({'key': 'value'})
        view = self.mappingproxy(custom)
        self.assertTrue('key' in view)
        self.assertTrue('magic' in view)
        self.assertFalse('xxx' in view)
        self.assertEqual(view['key'], 'getitem=value')
        self.assertRaises(KeyError, view.__getitem__, 'xxx')
        self.assertEqual(tuple(view), ('iter',))
        self.assertEqual(len(view), 500)
        self.assertEqual(view.copy(), 'copy')
        self.assertEqual(view.get('key'), 'get=value')
        self.assertEqual(view.get('xxx'), 'get=default=None')
        self.assertEqual(view.items(), 'items')
        self.assertEqual(view.keys(), 'keys')
        self.assertEqual(view.values(), 'values')

    def test_chainmap(self):
        # [FUNCTION DOC] Checks a proxy over a ChainMap: membership, lookup, sorted
        # iteration, length, that copy returns a distinct ChainMap equal to the original,
        # get, and keys/items/values.
        d1 = {'x': 1}
        d2 = {'y': 2}
        mapping = collections.ChainMap(d1, d2)
        view = self.mappingproxy(mapping)
        self.assertTrue('x' in view)
        self.assertTrue('y' in view)
        self.assertFalse('z' in view)
        self.assertEqual(view['x'], 1)
        self.assertEqual(view['y'], 2)
        self.assertRaises(KeyError, view.__getitem__, 'z')
        self.assertEqual(tuple(sorted(view)), ('x', 'y'))
        self.assertEqual(len(view), 2)
        copy = view.copy()
        self.assertIsNot(copy, mapping)
        self.assertIsInstance(copy, collections.ChainMap)
        self.assertEqual(copy, mapping)
        self.assertEqual(view.get('x'), 1)
        self.assertEqual(view.get('y'), 2)
        self.assertIsNone(view.get('z'))
        self.assertEqual(tuple(sorted(view.items())), (('x', 1), ('y', 2)))
        self.assertEqual(tuple(sorted(view.keys())), ('x', 'y'))
        self.assertEqual(tuple(sorted(view.values())), (1, 2))

    def test_contains(self):
        # [FUNCTION DOC] Checks membership tests on a proxy over a dict built from fromkeys.
        view = self.mappingproxy(dict.fromkeys('abc'))
        self.assertTrue('a' in view)
        self.assertTrue('b' in view)
        self.assertTrue('c' in view)
        self.assertFalse('xxx' in view)

    def test_views(self):
        # [FUNCTION DOC] Checks that keys, values and items views taken from a proxy are live:
        # they are empty at first and reflect an item added later to the underlying dict.
        mapping = {}
        view = self.mappingproxy(mapping)
        keys = view.keys()
        values = view.values()
        items = view.items()
        self.assertEqual(list(keys), [])
        self.assertEqual(list(values), [])
        self.assertEqual(list(items), [])
        mapping['key'] = 'value'
        self.assertEqual(list(keys), ['key'])
        self.assertEqual(list(values), ['value'])
        self.assertEqual(list(items), [('key', 'value')])

    def test_len(self):
        # [FUNCTION DOC] Checks that the proxy length equals the length of the wrapped dict
        # for sizes zero through five.
        for expected in range(6):
            data = dict.fromkeys('abcde'[:expected])
            self.assertEqual(len(data), expected)
            view = self.mappingproxy(data)
            self.assertEqual(len(view), expected)

    def test_iterators(self):
        # [FUNCTION DOC] Checks that iterating the proxy and its keys/values/items views yields
        # the same elements as the wrapped dict.
        keys = ('x', 'y')
        values = (1, 2)
        items = tuple(zip(keys, values))
        view = self.mappingproxy(dict(items))
        self.assertEqual(set(view), set(keys))
        self.assertEqual(set(view.keys()), set(keys))
        self.assertEqual(set(view.values()), set(values))
        self.assertEqual(set(view.items()), set(items))

    def test_reversed(self):
        # [FUNCTION DOC] Checks that reversing a proxy yields keys in reverse insertion order
        # using the live dict (a key deleted after creating the proxy is not seen), and that
        # exhausting the reverse iterator raises StopIteration.
        d = {'a': 1, 'b': 2, 'foo': 0, 'c': 3, 'd': 4}
        mp = self.mappingproxy(d)
        del d['foo']
        r = reversed(mp)
        self.assertEqual(list(r), list('dcba'))
        self.assertRaises(StopIteration, next, r)

    def test_copy(self):
        # [FUNCTION DOC] Checks that copying a proxy returns a plain dict equal to the original
        # that is independent of later changes to the original while the proxy still sees them.
        original = {'key1': 27, 'key2': 51, 'key3': 93}
        view = self.mappingproxy(original)
        copy = view.copy()
        self.assertEqual(type(copy), dict)
        self.assertEqual(copy, original)
        original['key1'] = 70
        self.assertEqual(view['key1'], 70)
        self.assertEqual(copy['key1'], 27)

    def test_union(self):
        # [FUNCTION DOC] Checks the union (or) operator on proxies: non-mapping operands and
        # in-place or raise TypeError; with a dict, results are new dicts (proxy on left or
        # right) with the right operand winning conflicts, and neither the proxy source nor
        # the other dict is modified.
        mapping = {'a': 0, 'b': 1, 'c': 2}
        view = self.mappingproxy(mapping)
        with self.assertRaises(TypeError):
            view | [('r', 2), ('d', 2)]
        with self.assertRaises(TypeError):
            [('r', 2), ('d', 2)] | view
        with self.assertRaises(TypeError):
            view |= [('r', 2), ('d', 2)]
        other = {'c': 3, 'p': 0}
        self.assertDictEqual(view | other, {'a': 0, 'b': 1, 'c': 3, 'p': 0})
        self.assertDictEqual(other | view, {'c': 2, 'p': 0, 'a': 0, 'b': 1})
        self.assertEqual(view, {'a': 0, 'b': 1, 'c': 2})
        self.assertDictEqual(mapping, {'a': 0, 'b': 1, 'c': 2})
        self.assertDictEqual(other, {'c': 3, 'p': 0})

    def test_hash(self):
        # [FUNCTION DOC] Checks that a proxy over an unhashable dict raises TypeError
        # when hashed, and a proxy over a dict subclass with a custom hash method
        # hashes the same as that dict.
        class HashableDict(dict):
            # [CLASS DOC] HELPER: Dict subclass with a fixed constant hash value.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the constant integer 3844817361.
                return 3844817361
        view = self.mappingproxy({'a': 1, 'b': 2})
        self.assertRaises(TypeError, hash, view)
        mapping = HashableDict({'a': 1, 'b': 2})
        view = self.mappingproxy(mapping)
        self.assertEqual(hash(view), hash(mapping))


class ClassCreationTests(CPythonTestCase):
    # [CLASS DOC] Tests of the dynamic class creation helpers in the types module: new_class,
    # prepare_class, resolve_bases and get_original_bases, and metaclass resolution rules.
    # Provides the nested Meta metaclass as a shared fixture.

    class Meta(type):
        # [CLASS DOC] HELPER: Metaclass fixture. Its prepare hook returns a namespace
        # holding y = 1 plus any keyword arguments it was given; its init and new
        # drop extra keyword arguments.
        def __init__(cls, name, bases, ns, **kw):
            # [FUNCTION DOC] HELPER: Initializes the class while
            # discarding extra keyword arguments.
            super().__init__(name, bases, ns)
        @staticmethod
        def __new__(mcls, name, bases, ns, **kw):
            # [FUNCTION DOC] HELPER: Creates the class via the parent
            # new, ignoring extra keywords.
            return super().__new__(mcls, name, bases, ns)
        @classmethod
        def __prepare__(mcls, name, bases, **kw):
            # [FUNCTION DOC] HELPER: Builds the namespace through the parent prepare, sets y to
            # 1, adds all keyword arguments, and returns it.
            ns = super().__prepare__(name, bases)
            ns["y"] = 1
            ns.update(kw)
            return ns

    def test_new_class_basics(self):
        # [FUNCTION DOC] Checks that new_class with only a name creates a class with that name
        # whose bases are just object.
        C = types.new_class("C")
        self.assertEqual(C.__name__, "C")
        self.assertEqual(C.__bases__, (object,))

    def test_new_class_subclass(self):
        # [FUNCTION DOC] Checks that new_class with int as a base gives a subclass of int.
        C = types.new_class("C", (int,))
        self.assertTrue(issubclass(C, int))

    def test_new_class_meta(self):
        # [FUNCTION DOC] Checks that new_class with a metaclass and extra keywords creates
        # classes whose type is the metaclass, with the keyword and prepared attributes set, and
        # does not mutate the keyword dict between two calls.
        Meta = self.Meta
        settings = {"metaclass": Meta, "z": 2}
        # We do this twice to make sure the passed in dict isn't mutated
        for i in range(2):
            C = types.new_class("C" + str(i), (), settings)
            self.assertIsInstance(C, Meta)
            self.assertEqual(C.y, 1)
            self.assertEqual(C.z, 2)

    def test_new_class_exec_body(self):
        # [FUNCTION DOC] Checks that the exec_body callback receives the class namespace and can
        # add attributes to it, in combination with a custom metaclass and keywords.
        Meta = self.Meta
        def func(ns):
            # [FUNCTION DOC] HELPER: Callback that stores x = 0 in the given namespace.
            ns["x"] = 0
        C = types.new_class("C", (), {"metaclass": Meta, "z": 2}, func)
        self.assertIsInstance(C, Meta)
        self.assertEqual(C.x, 0)
        self.assertEqual(C.y, 1)
        self.assertEqual(C.z, 2)

    def test_new_class_metaclass_keywords(self):
        # [FUNCTION DOC] Checks that when the metaclass is a plain function, it
        # receives the name, bases, namespace and remaining keyword arguments and its
        # return value is the result.
        #Test that keywords are passed to the metaclass:
        def meta_func(name, bases, ns, **kw):
            # [FUNCTION DOC] HELPER: Metaclass stand-in that returns its arguments as a tuple.
            return name, bases, ns, kw
        res = types.new_class("X",
                              (int, object),
                              dict(metaclass=meta_func, x=0))
        self.assertEqual(res, ("X", (int, object), {}, {"x": 0}))

    def test_new_class_defaults(self):
        # [FUNCTION DOC] Checks new_class with explicitly empty keywords and no body callback
        # gives default name and bases.
        # Test defaults/keywords:
        C = types.new_class("C", (), {}, None)
        self.assertEqual(C.__name__, "C")
        self.assertEqual(C.__bases__, (object,))

    def test_new_class_meta_with_base(self):
        # [FUNCTION DOC] Checks new_class called entirely with keyword arguments (name, bases,
        # kwds, exec_body) combining a base class, a metaclass and a body callback.
        Meta = self.Meta
        def func(ns):
            # [FUNCTION DOC] HELPER: Callback that stores x = 0 in the given namespace.
            ns["x"] = 0
        C = types.new_class(name="C",
                            bases=(int,),
                            kwds=dict(metaclass=Meta, z=2),
                            exec_body=func)
        self.assertTrue(issubclass(C, int))
        self.assertIsInstance(C, Meta)
        self.assertEqual(C.x, 0)
        self.assertEqual(C.y, 1)
        self.assertEqual(C.z, 2)

    def test_new_class_with_mro_entry(self):
        # [FUNCTION DOC] Checks that a non-class base object with a bases-resolution hook is
        # replaced by the classes it returns, and that original bases and MRO reflect that.
        # [CLASS DOC] HELPER: Empty class returned by the bases-resolution hook.
        class A: pass
        class C:
            # [CLASS DOC] HELPER: Non-class object type whose hook replaces it with class A.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns a one-element tuple holding class A.
                return (A,)
        c = C()
        D = types.new_class('D', (c,), {})
        self.assertEqual(D.__bases__, (A,))
        self.assertEqual(D.__orig_bases__, (c,))
        self.assertEqual(D.__mro__, (D, A, object))

    def test_new_class_with_mro_entry_genericalias(self):
        # [FUNCTION DOC] Checks that typing and builtin generic aliases passed as bases are
        # resolved (list with Generic for the typing form, plain list for the builtin form)
        # while the original bases are preserved.
        L1 = types.new_class('L1', (typing.List[int],), {})
        self.assertEqual(L1.__bases__, (list, typing.Generic))
        self.assertEqual(L1.__orig_bases__, (typing.List[int],))
        self.assertEqual(L1.__mro__, (L1, list, typing.Generic, object))

        L2 = types.new_class('L2', (list[int],), {})
        self.assertEqual(L2.__bases__, (list,))
        self.assertEqual(L2.__orig_bases__, (list[int],))
        self.assertEqual(L2.__mro__, (L2, list, object))

    def test_new_class_with_mro_entry_none(self):
        # [FUNCTION DOC] Checks that a hook returning an empty tuple removes
        # its object from the bases.
        # [CLASS DOC] HELPER: Empty class used as a real base.
        class A: pass
        # [CLASS DOC] HELPER: Empty class used as a real base.
        class B: pass
        class C:
            # [CLASS DOC] HELPER: Non-class object type whose hook returns an empty tuple.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns an empty tuple.
                return ()
        c = C()
        D = types.new_class('D', (A, c, B), {})
        self.assertEqual(D.__bases__, (A, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, B, object))

    def test_new_class_with_mro_entry_error(self):
        # [FUNCTION DOC] Checks that a hook returning a non-tuple (a class itself) makes
        # new_class raise TypeError.
        # [CLASS DOC] HELPER: Empty class.
        class A: pass
        class C:
            # [CLASS DOC] HELPER: Object type whose hook wrongly returns a
            # class instead of a tuple.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns class A (not a tuple), which is invalid.
                return A
        c = C()
        with self.assertRaises(TypeError):
            types.new_class('D', (c,), {})

    def test_new_class_with_mro_entry_multiple(self):
        # [FUNCTION DOC] Checks that two hook objects each expanding to two bases are spliced in
        # order, giving the four bases in sequence.
        # [CLASS DOC] HELPER: Empty class.
        class A1: pass
        # [CLASS DOC] HELPER: Empty class.
        class A2: pass
        # [CLASS DOC] HELPER: Empty class.
        class B1: pass
        # [CLASS DOC] HELPER: Empty class.
        class B2: pass
        class A:
            # [CLASS DOC] HELPER: Object type whose hook expands to A1 and A2.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns a tuple of A1 and A2.
                return (A1, A2)
        class B:
            # [CLASS DOC] HELPER: Object type whose hook expands to B1 and B2.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns a tuple of B1 and B2.
                return (B1, B2)
        D = types.new_class('D', (A(), B()), {})
        self.assertEqual(D.__bases__, (A1, A2, B1, B2))

    def test_new_class_with_mro_entry_multiple_2(self):
        # [FUNCTION DOC] Checks expansion of hook objects mixed with a real class between them:
        # the resulting bases keep order, with three expanded, one plain, then two expanded.
        # [CLASS DOC] HELPER: Empty class.
        class A1: pass
        # [CLASS DOC] HELPER: Empty class.
        class A2: pass
        # [CLASS DOC] HELPER: Empty class.
        class A3: pass
        # [CLASS DOC] HELPER: Empty class.
        class B1: pass
        # [CLASS DOC] HELPER: Empty class.
        class B2: pass
        class A:
            # [CLASS DOC] HELPER: Object type whose hook expands to A1, A2, A3.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns a tuple of A1, A2 and A3.
                return (A1, A2, A3)
        class B:
            # [CLASS DOC] HELPER: Object type whose hook expands to B1 and B2.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns a tuple of B1 and B2.
                return (B1, B2)
        # [CLASS DOC] HELPER: Empty class used as a plain middle base.
        class C: pass
        D = types.new_class('D', (A(), C, B()), {})
        self.assertEqual(D.__bases__, (A1, A2, A3, C, B1, B2))

    def test_get_original_bases(self):
        # [FUNCTION DOC] Checks types.get_original_bases: for plain classes it returns the
        # direct bases, for generic subclasses the unresolved generic alias bases, for multiple
        # inheritance the actual bases, for NamedTuple and TypedDict (class and call form) the
        # typing helper as first base, and raises TypeError for a non-class argument.
        T = typing.TypeVar('T')
        # [CLASS DOC] HELPER: Empty class.
        class A: pass
        # [CLASS DOC] HELPER: Generic class over T.
        class B(typing.Generic[T]): pass
        # [CLASS DOC] HELPER: Subclass of B parameterized with int.
        class C(B[int]): pass
        # [CLASS DOC] HELPER: Subclass of B with str plus float.
        class D(B[str], float): pass

        self.assertEqual(types.get_original_bases(A), (object,))
        self.assertEqual(types.get_original_bases(B), (typing.Generic[T],))
        self.assertEqual(types.get_original_bases(C), (B[int],))
        self.assertEqual(types.get_original_bases(int), (object,))
        self.assertEqual(types.get_original_bases(D), (B[str], float))

        # [CLASS DOC] HELPER: Subclass of list parameterized by T.
        class E(list[T]): pass
        # [CLASS DOC] HELPER: Subclass of list parameterized by int.
        class F(list[int]): pass

        self.assertEqual(types.get_original_bases(E), (list[T],))
        self.assertEqual(types.get_original_bases(F), (list[int],))

        # [CLASS DOC] HELPER: Generic base class.
        class FirstBase(typing.Generic[T]): pass
        # [CLASS DOC] HELPER: Generic base class.
        class SecondBase(typing.Generic[T]): pass
        # [CLASS DOC] HELPER: Subclass of FirstBase[int].
        class First(FirstBase[int]): pass
        # [CLASS DOC] HELPER: Subclass of SecondBase[int].
        class Second(SecondBase[int]): pass
        # [CLASS DOC] HELPER: Plain multiple-inheritance subclass of First and Second.
        class G(First, Second): pass
        self.assertEqual(types.get_original_bases(G), (First, Second))

        # [CLASS DOC] HELPER: Generic class over T.
        class First_(typing.Generic[T]): pass
        # [CLASS DOC] HELPER: Generic class over T.
        class Second_(typing.Generic[T]): pass
        # [CLASS DOC] HELPER: Subclass of both unparameterized generic classes.
        class H(First_, Second_): pass
        self.assertEqual(types.get_original_bases(H), (First_, Second_))

        class ClassBasedNamedTuple(typing.NamedTuple):
            # [CLASS DOC] HELPER: Class-syntax NamedTuple with an int field x.
            x: int

        class GenericNamedTuple(typing.NamedTuple, typing.Generic[T]):
            # [CLASS DOC] HELPER: Generic class-syntax NamedTuple with a field x of type T.
            x: T

        CallBasedNamedTuple = typing.NamedTuple("CallBasedNamedTuple", [("x", int)])

        self.assertIs(
            types.get_original_bases(ClassBasedNamedTuple)[0], typing.NamedTuple
        )
        self.assertEqual(
            types.get_original_bases(GenericNamedTuple),
            (typing.NamedTuple, typing.Generic[T])
        )
        self.assertIs(
            types.get_original_bases(CallBasedNamedTuple)[0], typing.NamedTuple
        )

        class ClassBasedTypedDict(typing.TypedDict):
            # [CLASS DOC] HELPER: Class-syntax TypedDict with an int field x.
            x: int

        class GenericTypedDict(typing.TypedDict, typing.Generic[T]):
            # [CLASS DOC] HELPER: Generic class-syntax TypedDict with a field x of type T.
            x: T

        CallBasedTypedDict = typing.TypedDict("CallBasedTypedDict", {"x": int})

        self.assertIs(
            types.get_original_bases(ClassBasedTypedDict)[0],
            typing.TypedDict
        )
        self.assertEqual(
            types.get_original_bases(GenericTypedDict),
            (typing.TypedDict, typing.Generic[T])
        )
        self.assertIs(
            types.get_original_bases(CallBasedTypedDict)[0],
            typing.TypedDict
        )

        with self.assertRaisesRegex(TypeError, "Expected an instance of type"):
            types.get_original_bases(object())

    # Many of the following tests are derived from test_descr.py
    def test_prepare_class(self):
        # [FUNCTION DOC] Checks prepare_class: with bases whose most derived metaclass is
        # the custom A (even though type is passed explicitly), it returns that metaclass,
        # the namespace built by its prepare hook, and an empty keyword dict (the
        # metaclass key is consumed).
        # Basic test of metaclass derivation
        expected_ns = {}
        class A(type):
            # [CLASS DOC] HELPER: Metaclass with a prepare hook returning the
            # shared expected namespace.
            def __new__(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Delegates to the base type
                # constructor with the same arguments.
                return type.__new__(*args, **kwargs)

            def __prepare__(*args):
                # [FUNCTION DOC] HELPER: Returns the shared expected namespace
                # dict, ignoring arguments.
                return expected_ns

        B = types.new_class("B", (object,))
        C = types.new_class("C", (object,), {"metaclass": A})

        # The most derived metaclass of D is A rather than type.
        meta, ns, kwds = types.prepare_class("D", (B, C), {"metaclass": type})
        self.assertIs(meta, A)
        self.assertIs(ns, expected_ns)
        self.assertEqual(len(kwds), 0)

    def test_bad___prepare__(self):
        # [FUNCTION DOC] Checks that a prepare hook returning None makes class creation raise
        # TypeError with a message naming the metaclass (or a placeholder when the metaclass is
        # not a type) and stating a mapping must be returned.
        # __prepare__() must return a mapping.
        class BadMeta(type):
            # [CLASS DOC] HELPER: Metaclass whose prepare hook returns None (invalid).
            @classmethod
            def __prepare__(*args):
                # [FUNCTION DOC] HELPER: Returns None, which is not a mapping.
                return None
        with self.assertRaisesRegex(TypeError,
                                    r'^BadMeta\.__prepare__\(\) must '
                                    r'return a mapping, not NoneType$'):
            class Foo(metaclass=BadMeta):
                # [CLASS DOC] HELPER: Class whose creation is expected to fail;
                # never successfully built.
                pass
        # Also test the case in which the metaclass is not a type.
        class BadMeta:
            # [CLASS DOC] HELPER: Non-type metaclass object type whose
            # prepare hook returns None.
            @classmethod
            def __prepare__(*args):
                # [FUNCTION DOC] HELPER: Returns None, which is not a mapping.
                return None
        with self.assertRaisesRegex(TypeError,
                                    r'^<metaclass>\.__prepare__\(\) must '
                                    r'return a mapping, not NoneType$'):
            class Bar(metaclass=BadMeta()):
                # [CLASS DOC] HELPER: Class whose creation is expected to fail;
                # never successfully built.
                pass

    def test_resolve_bases(self):
        # [FUNCTION DOC] Checks types.resolve_bases: replaces hook objects by their
        # returned classes, honors a hook that returns nothing when its class is already
        # present, leaves real classes alone, and returns the very same tuple object
        # when nothing needs resolving.
        # [CLASS DOC] HELPER: Empty class.
        class A: pass
        # [CLASS DOC] HELPER: Empty class.
        class B: pass
        class C:
            # [CLASS DOC] HELPER: Object type whose hook returns nothing when A is
            # among the bases, otherwise A.
            def __mro_entries__(self, bases):
                # [FUNCTION DOC] HELPER: Returns an empty tuple if A is among
                # bases, else a tuple holding A.
                if A in bases:
                    return ()
                return (A,)
        c = C()
        self.assertEqual(types.resolve_bases(()), ())
        self.assertEqual(types.resolve_bases((c,)), (A,))
        self.assertEqual(types.resolve_bases((C,)), (C,))
        self.assertEqual(types.resolve_bases((A, C)), (A, C))
        self.assertEqual(types.resolve_bases((c, A)), (A,))
        self.assertEqual(types.resolve_bases((A, c)), (A,))
        x = (A,)
        y = (C,)
        z = (A, C)
        t = (A, C, B)
        for bases in [x, y, z, t]:
            self.assertIs(types.resolve_bases(bases), bases)

    def test_resolve_bases_with_mro_entry(self):
        # [FUNCTION DOC] Checks resolve_bases on a typing list alias (gives list plus Generic)
        # and on a builtin list alias (gives list).
        self.assertEqual(types.resolve_bases((typing.List[int],)),
                         (list, typing.Generic))
        self.assertEqual(types.resolve_bases((list[int],)), (list,))

    def test_metaclass_derivation(self):
        # [FUNCTION DOC] Regression test for issue 1294232: with metaclasses AMeta and BMeta (a
        # subclass), checks that the most derived metaclass is chosen regardless of base order
        # or an explicit weaker metaclass, that its prepare hook runs (marker attribute
        # present), and that the call order of new methods is recorded in the list new_calls.
        # issue1294232: correct metaclass calculation
        new_calls = []  # to check the order of __new__ calls
        class AMeta(type):
            # [CLASS DOC] HELPER: Base metaclass recording its call in new_calls; prepare
            # returns an empty namespace.
            def __new__(mcls, name, bases, ns):
                # [FUNCTION DOC] HELPER: Appends the marker AMeta to
                # new_calls then builds the class.
                new_calls.append('AMeta')
                return super().__new__(mcls, name, bases, ns)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: Returns a fresh empty namespace dict.
                return {}

        class BMeta(AMeta):
            # [CLASS DOC] HELPER: Derived metaclass recording its call
            # and tagging the namespace.
            def __new__(mcls, name, bases, ns):
                # [FUNCTION DOC] HELPER: Appends the marker BMeta to new_calls
                # then defers to the parent.
                new_calls.append('BMeta')
                return super().__new__(mcls, name, bases, ns)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: Gets the namespace from the parent prepare, adds a
                # marker key set to true, and returns it.
                ns = super().__prepare__(name, bases)
                ns['BMeta_was_here'] = True
                return ns

        A = types.new_class("A", (), {"metaclass": AMeta})
        self.assertEqual(new_calls, ['AMeta'])
        new_calls.clear()

        B = types.new_class("B", (), {"metaclass": BMeta})
        # BMeta.__new__ calls AMeta.__new__ with super:
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()

        C = types.new_class("C", (A, B))
        # The most derived metaclass is BMeta:
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        # BMeta.__prepare__ should've been called:
        self.assertIn('BMeta_was_here', C.__dict__)

        # The order of the bases shouldn't matter:
        C2 = types.new_class("C2", (B, A))
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', C2.__dict__)

        # Check correct metaclass calculation when a metaclass is declared:
        D = types.new_class("D", (C,), {"metaclass": type})
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', D.__dict__)

        E = types.new_class("E", (C,), {"metaclass": AMeta})
        self.assertEqual(new_calls, ['BMeta', 'AMeta'])
        new_calls.clear()
        self.assertIn('BMeta_was_here', E.__dict__)

    def test_metaclass_override_function(self):
        # [FUNCTION DOC] Checks that when the given metaclass is a plain function there is no
        # metaclass calculation: the function's return value is used for any base combination,
        # including a base with a custom metaclass.
        # Special case: the given metaclass isn't a class,
        # so there is no metaclass calculation.
        class A(metaclass=self.Meta):
            # [CLASS DOC] HELPER: Class created with the shared Meta metaclass fixture.
            pass

        marker = object()
        def func(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns a fixed marker object whatever the arguments.
            return marker

        X = types.new_class("X", (), {"metaclass": func})
        Y = types.new_class("Y", (object,), {"metaclass": func})
        Z = types.new_class("Z", (A,), {"metaclass": func})
        self.assertIs(marker, X)
        self.assertIs(marker, Y)
        self.assertIs(marker, Z)

    def test_metaclass_override_callable(self):
        # [FUNCTION DOC] Checks metaclass calculation when the metaclass is a class not derived
        # from type: tracks prepare and new call order in lists, verifies the most derived such
        # metaclass wins, and that mixing it with type or with plain objects such as int
        # instances raises TypeError.
        # The given metaclass is a class,
        # but not a descendant of type.
        new_calls = []  # to check the order of __new__ calls
        prepare_calls = []  # to track __prepare__ calls
        class ANotMeta:
            # [CLASS DOC] HELPER: Non-type "metaclass" recording its prepare and new calls.
            def __new__(mcls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Records ANotMeta in new_calls
                # and creates a bare instance.
                new_calls.append('ANotMeta')
                return super().__new__(mcls)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: Records ANotMeta in prepare_calls and
                # returns an empty namespace.
                prepare_calls.append('ANotMeta')
                return {}

        class BNotMeta(ANotMeta):
            # [CLASS DOC] HELPER: Derived non-type "metaclass" recording its calls.
            def __new__(mcls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Records BNotMeta in new_calls and
                # defers to the parent new.
                new_calls.append('BNotMeta')
                return super().__new__(mcls)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: Records BNotMeta in prepare_calls and
                # defers to the parent prepare.
                prepare_calls.append('BNotMeta')
                return super().__prepare__(name, bases)

        A = types.new_class("A", (), {"metaclass": ANotMeta})
        self.assertIs(ANotMeta, type(A))
        self.assertEqual(prepare_calls, ['ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['ANotMeta'])
        new_calls.clear()

        B = types.new_class("B", (), {"metaclass": BNotMeta})
        self.assertIs(BNotMeta, type(B))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        C = types.new_class("C", (A, B))
        self.assertIs(BNotMeta, type(C))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        C2 = types.new_class("C2", (B, A))
        self.assertIs(BNotMeta, type(C2))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        # This is a TypeError, because of a metaclass conflict:
        # BNotMeta is neither a subclass, nor a superclass of type
        with self.assertRaises(TypeError):
            D = types.new_class("D", (C,), {"metaclass": type})

        E = types.new_class("E", (C,), {"metaclass": ANotMeta})
        self.assertIs(BNotMeta, type(E))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        F = types.new_class("F", (object(), C))
        self.assertIs(BNotMeta, type(F))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        F2 = types.new_class("F2", (C, object()))
        self.assertIs(BNotMeta, type(F2))
        self.assertEqual(prepare_calls, ['BNotMeta', 'ANotMeta'])
        prepare_calls.clear()
        self.assertEqual(new_calls, ['BNotMeta', 'ANotMeta'])
        new_calls.clear()

        # TypeError: BNotMeta is neither a
        # subclass, nor a superclass of int
        with self.assertRaises(TypeError):
            X = types.new_class("X", (C, int()))
        with self.assertRaises(TypeError):
            X = types.new_class("X", (int(), C))

    def test_one_argument_type(self):
        # [FUNCTION DOC] Checks that only the type builtin itself accepts the
        # single-argument form: calling a metaclass subclass (or a class whose metaclass
        # derives from it) with one argument raises TypeError with the message about exactly
        # 3 arguments. Also checks type of 5 is int.
        expected_message = 'type.__new__() takes exactly 3 arguments (1 given)'

        # Only type itself can use the one-argument form (#27157)
        self.assertIs(type(5), int)

        class M(type):
            # [CLASS DOC] HELPER: Empty metaclass deriving from type.
            pass
        with self.assertRaises(TypeError) as cm:
            M(5)
        self.assertEqual(str(cm.exception), expected_message)

        class N(type, metaclass=M):
            # [CLASS DOC] HELPER: Empty type subclass whose own metaclass is M.
            pass
        with self.assertRaises(TypeError) as cm:
            N(5)
        self.assertEqual(str(cm.exception), expected_message)

    def test_metaclass_new_error(self):
        # [FUNCTION DOC] Regression test for bpo-44232: a metaclass constructor raising an
        # error for any class but Model must have its exception reported properly when type
        # creates a class whose winning metaclass differs from the called one; expects
        # RuntimeWarning to propagate.
        # bpo-44232: The C function type_new() must properly report the
        # exception when a metaclass constructor raises an exception and the
        # winner class is not the metaclass.
        class ModelBase(type):
            # [CLASS DOC] HELPER: Metaclass that creates the class with an empty namespace and
            # raises RuntimeWarning for any class name other than Model.
            def __new__(cls, name, bases, attrs):
                # [FUNCTION DOC] HELPER: Builds the class via the parent new with an empty
                # namespace; raises RuntimeWarning including the name when the name is not
                # Model; otherwise returns the class.
                super_new = super().__new__
                new_class = super_new(cls, name, bases, {})
                if name != "Model":
                    raise RuntimeWarning(f"{name=}")
                return new_class

        class Model(metaclass=ModelBase):
            # [CLASS DOC] HELPER: Class created with the ModelBase metaclass.
            pass

        with self.assertRaises(RuntimeWarning):
            type("SouthPonies", (Model,), {})

    def test_tuple_subclass_as_bases(self):
        # [FUNCTION DOC] Regression test for gh-132176: creating a class with a tuple
        # subclass instance as the bases argument must not crash and the stored bases
        # keep the tuple subclass type.
        # gh-132176: it used to crash on using
        # tuple subclass for as base classes.
        # [CLASS DOC] HELPER: Empty tuple subclass used as the bases container.
        class TupleSubclass(tuple): pass

        typ = type("typ", TupleSubclass((int, object)), {})
        self.assertEqual(typ.__bases__, (int, object))
        self.assertEqual(type(typ.__bases__), TupleSubclass)


class SimpleNamespaceTests(CPythonTestCase):
    # [CLASS DOC] Tests of types.SimpleNamespace: construction, attribute get/set/delete, the
    # underlying dict, repr (including recursive), equality, pickling, copy.replace, subclassing
    # and comparison with look-alike objects.

    def test_constructor(self):
        # [FUNCTION DOC] Checks the constructor with keyword arguments, a mapping, a sequence of
        # pairs, a mapping subclass, or combinations (keywords override), preserving insertion
        # order, and the errors: too many positional arguments, non-iterable or non-pair items
        # (TypeError or ValueError), non-string keys and unhashable keys (TypeError).
        def check(ns, expected):
            # [FUNCTION DOC] HELPER: Asserts the namespace dict has the expected size, content,
            # key order and attribute values.
            self.assertEqual(len(ns.__dict__), len(expected))
            self.assertEqual(vars(ns), expected)
            # check order
            self.assertEqual(list(vars(ns).items()), list(expected.items()))
            for name in expected:
                self.assertEqual(getattr(ns, name), expected[name])

        check(types.SimpleNamespace(), {})
        check(types.SimpleNamespace(x=1, y=2), {'x': 1, 'y': 2})
        check(types.SimpleNamespace(**dict(x=1, y=2)), {'x': 1, 'y': 2})
        check(types.SimpleNamespace({'x': 1, 'y': 2}, x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace([['x', 1], ['y', 2]], x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace(UserDict({'x': 1, 'y': 2}), x=4, z=3),
              {'x': 4, 'y': 2, 'z': 3})
        check(types.SimpleNamespace({'x': 1, 'y': 2}), {'x': 1, 'y': 2})
        check(types.SimpleNamespace([['x', 1], ['y', 2]]), {'x': 1, 'y': 2})
        check(types.SimpleNamespace([], x=4, z=3), {'x': 4, 'z': 3})
        check(types.SimpleNamespace({}, x=4, z=3), {'x': 4, 'z': 3})
        check(types.SimpleNamespace([]), {})
        check(types.SimpleNamespace({}), {})

        with self.assertRaises(TypeError):
            types.SimpleNamespace([], [])  # too many positional arguments
        with self.assertRaises(TypeError):
            types.SimpleNamespace(1)  # not a mapping or iterable
        with self.assertRaises(TypeError):
            types.SimpleNamespace([1])  # non-iterable
        with self.assertRaises(ValueError):
            types.SimpleNamespace([['x']])  # not a pair
        with self.assertRaises(ValueError):
            types.SimpleNamespace([['x', 'y', 'z']])
        with self.assertRaises(TypeError):
            types.SimpleNamespace(**{1: 2})  # non-string key
        with self.assertRaises(TypeError):
            types.SimpleNamespace({1: 2})
        with self.assertRaises(TypeError):
            types.SimpleNamespace([[1, 2]])
        with self.assertRaises(TypeError):
            types.SimpleNamespace(UserDict({1: 2}))
        with self.assertRaises(TypeError):
            types.SimpleNamespace([[[], 2]])  # non-hashable key

    def test_unbound(self):
        # [FUNCTION DOC] Checks the vars of an empty namespace and of one with two attributes.
        ns1 = vars(types.SimpleNamespace())
        ns2 = vars(types.SimpleNamespace(x=1, y=2))

        self.assertEqual(ns1, {})
        self.assertEqual(ns2, {'y': 2, 'x': 1})

    def test_underlying_dict(self):
        # [FUNCTION DOC] Checks that the attribute dict stays valid after its namespace is
        # deleted and equals the original keyword data.
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2)
        ns3 = types.SimpleNamespace(a=True, b=False)
        mapping = ns3.__dict__
        del ns3

        self.assertEqual(ns1.__dict__, {})
        self.assertEqual(ns2.__dict__, {'y': 2, 'x': 1})
        self.assertEqual(mapping, dict(a=True, b=False))

    def test_attrget(self):
        # [FUNCTION DOC] Checks attribute reads and that a missing
        # attribute raises AttributeError.
        ns = types.SimpleNamespace(x=1, y=2, w=3)

        self.assertEqual(ns.x, 1)
        self.assertEqual(ns.y, 2)
        self.assertEqual(ns.w, 3)
        with self.assertRaises(AttributeError):
            ns.z

    def test_attrset(self):
        # [FUNCTION DOC] Checks that assigning new attributes updates the underlying dict.
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2, w=3)
        ns1.a = 'spam'
        ns1.b = 'ham'
        ns2.z = 4
        ns2.theta = None

        self.assertEqual(ns1.__dict__, dict(a='spam', b='ham'))
        self.assertEqual(ns2.__dict__, dict(x=1, y=2, w=3, z=4, theta=None))

    def test_attrdel(self):
        # [FUNCTION DOC] Checks that deleting a missing attribute raises AttributeError, and
        # that existing attributes can be deleted, re-added and removed again.
        ns1 = types.SimpleNamespace()
        ns2 = types.SimpleNamespace(x=1, y=2, w=3)

        with self.assertRaises(AttributeError):
            del ns1.spam
        with self.assertRaises(AttributeError):
            del ns2.spam

        del ns2.y
        self.assertEqual(vars(ns2), dict(w=3, x=1))
        ns2.y = 'spam'
        self.assertEqual(vars(ns2), dict(w=3, x=1, y='spam'))
        del ns2.y
        self.assertEqual(vars(ns2), dict(w=3, x=1))

        ns1.spam = 5
        self.assertEqual(vars(ns1), dict(spam=5))
        del ns1.spam
        self.assertEqual(vars(ns1), {})

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr shows the class name "namespace" with attributes in
        # insertion order, including underscore names and string quoting.
        ns1 = types.SimpleNamespace(x=1, y=2, w=3)
        ns2 = types.SimpleNamespace()
        ns2.x = "spam"
        ns2._y = 5
        name = "namespace"

        self.assertEqual(repr(ns1), "{name}(x=1, y=2, w=3)".format(name=name))
        self.assertEqual(repr(ns2), "{name}(x='spam', _y=5)".format(name=name))

    def test_equal(self):
        # [FUNCTION DOC] Checks that namespaces with equal attributes are equal
        # and differing ones are not.
        ns1 = types.SimpleNamespace(x=1)
        ns2 = types.SimpleNamespace()
        ns2.x = 1

        self.assertEqual(types.SimpleNamespace(), types.SimpleNamespace())
        self.assertEqual(ns1, ns2)
        self.assertNotEqual(ns2, types.SimpleNamespace())

    def test_nested(self):
        # [FUNCTION DOC] Checks that namespaces can hold other namespaces and attribute
        # chains resolve through them.
        ns1 = types.SimpleNamespace(a=1, b=2)
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=ns1)
        ns2.spam = ns1
        ns2.ham = '?'
        ns2.spam = ns3

        self.assertEqual(vars(ns1), dict(a=1, b=2))
        self.assertEqual(vars(ns2), dict(spam=ns3, ham='?'))
        self.assertEqual(ns2.spam, ns3)
        self.assertEqual(vars(ns3), dict(x=ns1))
        self.assertEqual(ns3.x.a, 1)

    def test_recursive(self):
        # [FUNCTION DOC] Checks that self-referential and mutually referential namespaces
        # can be built and traversed.
        ns1 = types.SimpleNamespace(c='cookie')
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=1)
        ns1.spam = ns1
        ns2.spam = ns3
        ns3.spam = ns2

        self.assertEqual(ns1.spam, ns1)
        self.assertEqual(ns1.spam.spam, ns1)
        self.assertEqual(ns1.spam.spam, ns1.spam)
        self.assertEqual(ns2.spam, ns3)
        self.assertEqual(ns3.spam, ns2)
        self.assertEqual(ns2.spam.spam, ns2)

    def test_recursive_repr(self):
        # [FUNCTION DOC] Checks that repr of recursive namespaces abbreviates the repeated
        # namespace with an ellipsis rather than recursing forever.
        ns1 = types.SimpleNamespace(c='cookie')
        ns2 = types.SimpleNamespace()
        ns3 = types.SimpleNamespace(x=1)
        ns1.spam = ns1
        ns2.spam = ns3
        ns3.spam = ns2
        name = "namespace"
        repr1 = "{name}(c='cookie', spam={name}(...))".format(name=name)
        repr2 = "{name}(spam={name}(x=1, spam={name}(...)))".format(name=name)

        self.assertEqual(repr(ns1), repr1)
        self.assertEqual(repr(ns2), repr2)

    def test_as_dict(self):
        # [FUNCTION DOC] Checks that a namespace is not a mapping: length, iteration, membership
        # and item access all raise TypeError.
        ns = types.SimpleNamespace(spam='spamspamspam')

        with self.assertRaises(TypeError):
            len(ns)
        with self.assertRaises(TypeError):
            iter(ns)
        with self.assertRaises(TypeError):
            'spam' in ns
        with self.assertRaises(TypeError):
            ns['spam']

    def test_subclass(self):
        # [FUNCTION DOC] Checks that a subclass of SimpleNamespace constructs instances of its
        # own type with the given attributes.
        class Spam(types.SimpleNamespace):
            # [CLASS DOC] HELPER: Empty SimpleNamespace subclass.
            pass

        spam = Spam(ham=8, eggs=9)

        self.assertIs(type(spam), Spam)
        self.assertEqual(vars(spam), {'ham': 8, 'eggs': 9})

    def test_pickle(self):
        # [FUNCTION DOC] Checks that a namespace pickles and unpickles to an equal object at
        # every pickle protocol; a TypeError during pickling is re-raised naming the protocol.
        ns = types.SimpleNamespace(breakfast="spam", lunch="spam")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pname = "protocol {}".format(protocol)
            try:
                ns_pickled = pickle.dumps(ns, protocol)
            except TypeError as e:
                raise TypeError(pname) from e
            ns_roundtrip = pickle.loads(ns_pickled)

            self.assertEqual(ns, ns_roundtrip, pname)

    def test_replace(self):
        # [FUNCTION DOC] Checks copy.replace on a namespace: result is an equal but
        # distinct SimpleNamespace, independent of the original, and keyword
        # arguments override attributes.
        ns = types.SimpleNamespace(x=11, y=22)

        ns2 = copy.replace(ns)
        self.assertEqual(ns2, ns)
        self.assertIsNot(ns2, ns)
        self.assertIs(type(ns2), types.SimpleNamespace)
        self.assertEqual(vars(ns2), {'x': 11, 'y': 22})
        ns2.x = 3
        self.assertEqual(ns.x, 11)
        ns.x = 4
        self.assertEqual(ns2.x, 3)

        self.assertEqual(vars(copy.replace(ns, x=1)), {'x': 1, 'y': 22})
        self.assertEqual(vars(copy.replace(ns, y=2)), {'x': 4, 'y': 2})
        self.assertEqual(vars(copy.replace(ns, x=1, y=2)), {'x': 1, 'y': 2})

    def test_replace_subclass(self):
        # [FUNCTION DOC] Checks that copy.replace on a subclass instance keeps the subclass type
        # and applies keyword overrides.
        class Spam(types.SimpleNamespace):
            # [CLASS DOC] HELPER: Empty SimpleNamespace subclass.
            pass

        spam = Spam(ham=8, eggs=9)
        spam2 = copy.replace(spam, ham=5)

        self.assertIs(type(spam2), Spam)
        self.assertEqual(vars(spam2), {'ham': 5, 'eggs': 9})

    def test_fake_namespace_compare(self):
        # [FUNCTION DOC] Regression test for issue 24257: a string subclass that claims to be a
        # SimpleNamespace via its class attribute must compare unequal (no SystemError) and
        # ordering comparisons must raise TypeError.
        # Issue #24257: Incorrect use of PyObject_IsInstance() caused
        # SystemError.
        class FakeSimpleNamespace(str):
            # [CLASS DOC] HELPER: String subclass whose class attribute
            # impersonates SimpleNamespace.
            __class__ = types.SimpleNamespace
        self.assertFalse(types.SimpleNamespace() == FakeSimpleNamespace())
        self.assertTrue(types.SimpleNamespace() != FakeSimpleNamespace())
        with self.assertRaises(TypeError):
            types.SimpleNamespace() < FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() <= FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() > FakeSimpleNamespace()
        with self.assertRaises(TypeError):
            types.SimpleNamespace() >= FakeSimpleNamespace()


class CoroutineTests(CPythonTestCase):
    # [CLASS DOC] Tests of the types.coroutine decorator: argument validation, pass-through of
    # async functions and non-generator results, duck-typed generator/coroutine objects, and the
    # generator wrapper object it creates.
    def test_wrong_args(self):
        # [FUNCTION DOC] Checks that decorating a non-callable (None, int, plain object) raises
        # TypeError mentioning that a callable is expected.
        samples = [None, 1, object()]
        for sample in samples:
            with self.assertRaisesRegex(TypeError,
                                        'types.coroutine.*expects a callable'):
                types.coroutine(sample)

    def test_non_gen_values(self):
        # [FUNCTION DOC] Checks that decorated functions returning non-generator values (a
        # string or an awaitable) return them unchanged, also when decorated twice.
        @types.coroutine
        def foo():
            # [FUNCTION DOC] HELPER: Returns the string "spam".
            return 'spam'
        self.assertEqual(foo(), 'spam')

        class Awaitable:
            # [CLASS DOC] HELPER: Awaitable whose await method returns an empty tuple.
            def __await__(self):
                # [FUNCTION DOC] HELPER: Returns an empty tuple (an empty iterator source).
                return ()
        aw = Awaitable()
        @types.coroutine
        def foo():
            # [FUNCTION DOC] HELPER: Returns the outer awaitable instance aw.
            return aw
        self.assertIs(aw, foo())

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(aw, foo())

    def test_async_def(self):
        # [FUNCTION DOC] Checks that decorating an async def function returns the same
        # function with unchanged code flags, and that decorating a plain function returning
        # that coroutine yields the coroutine with the same flags (repeatedly decorated,
        # coroutine closed each time).
        # Test that types.coroutine passes 'async def' coroutines
        # without modification

        # [FUNCTION DOC] HELPER: Empty async function used as a native coroutine.
        async def foo(): pass
        foo_code = foo.__code__
        foo_flags = foo.__code__.co_flags
        decorated_foo = types.coroutine(foo)
        self.assertIs(foo, decorated_foo)
        self.assertEqual(foo.__code__.co_flags, foo_flags)
        self.assertIs(decorated_foo.__code__, foo_code)

        foo_coro = foo()
        # [FUNCTION DOC] HELPER: Returns the pre-built coroutine foo_coro.
        def bar(): return foo_coro
        for _ in range(2):
            bar = types.coroutine(bar)
            coro = bar()
            self.assertIs(foo_coro, coro)
            self.assertEqual(coro.cr_code.co_flags, foo_flags)
            coro.close()

    def test_duck_coro(self):
        # [FUNCTION DOC] Checks that an object with coroutine methods (send, throw, close,
        # await) is returned as-is by the decorated function and is its own awaitable iterator.
        class CoroLike:
            # [CLASS DOC] HELPER: Duck-typed coroutine with send,
            # throw, close and await methods.
            # [FUNCTION DOC] HELPER: Do-nothing send.
            def send(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing throw.
            def throw(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing close.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Returns the object itself.
            def __await__(self): return self

        coro = CoroLike()
        @types.coroutine
        def foo():
            # [FUNCTION DOC] HELPER: Returns the outer duck-typed coroutine object.
            return coro
        self.assertIs(foo(), coro)
        self.assertIs(foo().__await__(), coro)

    def test_duck_corogen(self):
        # [FUNCTION DOC] Same as the duck coroutine test but for an object that also supports
        # iteration (iter and next): returned unchanged and awaits itself.
        class CoroGenLike:
            # [CLASS DOC] HELPER: Duck object that is both coroutine-like and an iterator.
            # [FUNCTION DOC] HELPER: Do-nothing send.
            def send(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing throw.
            def throw(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing close.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Returns the object itself.
            def __await__(self): return self
            # [FUNCTION DOC] HELPER: Returns the object itself.
            def __iter__(self): return self
            # [FUNCTION DOC] HELPER: Do-nothing next.
            def __next__(self): pass

        coro = CoroGenLike()
        @types.coroutine
        def foo():
            # [FUNCTION DOC] HELPER: Returns the outer duck object.
            return coro
        self.assertIs(foo(), coro)
        self.assertIs(foo().__await__(), coro)

    def test_duck_gen(self):
        # [FUNCTION DOC] Uses a mock generator-like object to check the generator wrapper: it is
        # wrapped in types._GeneratorWrapper and is its own await and iterator, proxies name,
        # qualified name and generator/coroutine state attributes (raising AttributeError until
        # the mock has them), forwards close, send, next and throw with all argument forms,
        # propagates exceptions, rejects bad argument counts without calling the mock, is not
        # double wrapped, and supports weak references.
        class GenLike:
            # [CLASS DOC] HELPER: Interface spec for the generator mock (send,
            # throw, close, iter, next).
            # [FUNCTION DOC] HELPER: Do-nothing send.
            def send(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing throw.
            def throw(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing close.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing iter.
            def __iter__(self): pass
            # [FUNCTION DOC] HELPER: Do-nothing next.
            def __next__(self): pass

        # Setup generator mock object
        gen = unittest.mock.MagicMock(GenLike)
        gen.__iter__ = lambda gen: gen
        gen.__name__ = 'gen'
        gen.__qualname__ = 'test.gen'
        self.assertIsInstance(gen, collections.abc.Generator)
        self.assertIs(gen, iter(gen))

        # [FUNCTION DOC] HELPER: Returns the mock generator object gen.
        @types.coroutine
        def foo(): return gen

        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)
        self.assertIs(wrapper.__await__(), wrapper)
        # Wrapper proxies duck generators completely:
        self.assertIs(iter(wrapper), wrapper)

        self.assertIsInstance(wrapper, collections.abc.Coroutine)
        self.assertIsInstance(wrapper, collections.abc.Awaitable)

        self.assertIs(wrapper.__qualname__, gen.__qualname__)
        self.assertIs(wrapper.__name__, gen.__name__)

        # Test AttributeErrors
        for name in {'gi_running', 'gi_frame', 'gi_code', 'gi_yieldfrom',
                     'cr_running', 'cr_frame', 'cr_code', 'cr_await'}:
            with self.assertRaises(AttributeError):
                getattr(wrapper, name)

        # Test attributes pass-through
        gen.gi_running = object()
        gen.gi_frame = object()
        gen.gi_code = object()
        gen.gi_yieldfrom = object()
        self.assertIs(wrapper.gi_running, gen.gi_running)
        self.assertIs(wrapper.gi_frame, gen.gi_frame)
        self.assertIs(wrapper.gi_code, gen.gi_code)
        self.assertIs(wrapper.gi_yieldfrom, gen.gi_yieldfrom)
        self.assertIs(wrapper.cr_running, gen.gi_running)
        self.assertIs(wrapper.cr_frame, gen.gi_frame)
        self.assertIs(wrapper.cr_code, gen.gi_code)
        self.assertIs(wrapper.cr_await, gen.gi_yieldfrom)

        wrapper.close()
        gen.close.assert_called_once_with()

        wrapper.send(1)
        gen.send.assert_called_once_with(1)
        gen.reset_mock()

        next(wrapper)
        gen.__next__.assert_called_once_with()
        gen.reset_mock()

        wrapper.throw(1, 2, 3)
        gen.throw.assert_called_once_with(1, 2, 3)
        gen.reset_mock()

        wrapper.throw(1, 2)
        gen.throw.assert_called_once_with(1, 2)
        gen.reset_mock()

        wrapper.throw(1)
        gen.throw.assert_called_once_with(1)
        gen.reset_mock()

        # Test exceptions propagation
        error = Exception()
        gen.throw.side_effect = error
        try:
            wrapper.throw(1)
        except Exception as ex:
            self.assertIs(ex, error)
        else:
            self.fail('wrapper did not propagate an exception')

        # Test invalid args
        gen.reset_mock()
        with self.assertRaises(TypeError):
            wrapper.throw()
        self.assertFalse(gen.throw.called)
        with self.assertRaises(TypeError):
            wrapper.close(1)
        self.assertFalse(gen.close.called)
        with self.assertRaises(TypeError):
            wrapper.send()
        self.assertFalse(gen.send.called)

        # Test that we do not double wrap
        # [FUNCTION DOC] HELPER: Returns the already wrapped object.
        @types.coroutine
        def bar(): return wrapper
        self.assertIs(wrapper, bar())

        # Test weakrefs support
        ref = weakref.ref(wrapper)
        self.assertIs(ref(), wrapper)

    def test_duck_functional_gen(self):
        # [FUNCTION DOC] Uses a hand-written generator emulation that yields a future-like
        # value and returns twice the sent value, wraps it with the decorator and awaits it
        # from a native coroutine: the first send gives the yielded value, sending 20
        # finishes with result 140.
        class Generator:
            # [CLASS DOC] HELPER: Clumsy iterator emulating a generator that yields
            # a value, receives a number, and returns it doubled. Stores a step
            # counter and the value to yield.
            """Emulates the following generator (very clumsy):

              def gen(fut):
                  result = yield fut
                  return result * 2
            """
            def __init__(self, fut):
                # [FUNCTION DOC] HELPER: Stores the value to yield and resets
                # the step counter to zero.
                self._i = 0
                self._fut = fut
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns the object itself.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Equivalent to sending None.
                return self.send(None)
            def send(self, v):
                # [FUNCTION DOC] HELPER: Step 0 returns the stored value (requires None sent);
                # step 1 raises StopIteration carrying double the sent value; later steps raise
                # StopIteration. Always increments the step counter.
                try:
                    if self._i == 0:
                        assert v is None
                        return self._fut
                    if self._i == 1:
                        raise StopIteration(v * 2)
                    if self._i > 1:
                        raise StopIteration
                finally:
                    self._i += 1
            def throw(self, tp, *exc):
                # [FUNCTION DOC] HELPER: Marks the emulation as finished (step 100) and
                # re-raises the given exception type unless it is GeneratorExit.
                self._i = 100
                if tp is not GeneratorExit:
                    raise tp
            def close(self):
                # [FUNCTION DOC] HELPER: Closes by throwing GeneratorExit.
                self.throw(GeneratorExit)

        # [FUNCTION DOC] HELPER: Returns a new emulated generator for the value "spam".
        @types.coroutine
        def foo(): return Generator('spam')

        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)

        async def corofunc():
            # [FUNCTION DOC] HELPER: Awaits the wrapped generator and
            # returns the result plus 100.
            return await foo() + 100
        coro = corofunc()

        self.assertEqual(coro.send(None), 'spam')
        try:
            coro.send(20)
        except StopIteration as ex:
            self.assertEqual(ex.args[0], 140)
        else:
            self.fail('StopIteration was expected')

    def test_gen(self):
        # [FUNCTION DOC] Checks the wrapper around a real generator: wrapper awaits to the
        # generator itself, name/code/state attributes match, values and return value propagate
        # through next and send (return value arrives in StopIteration), throw works, and
        # double decoration is stable.
        def gen_func():
            # [FUNCTION DOC] HELPER: Generator yielding 1 then 2 and
            # returning the value sent in.
            yield 1
            return (yield 2)
        gen = gen_func()
        # [FUNCTION DOC] HELPER: Returns the shared generator object gen.
        @types.coroutine
        def foo(): return gen
        wrapper = foo()
        self.assertIsInstance(wrapper, types._GeneratorWrapper)
        self.assertIs(wrapper.__await__(), gen)

        for name in ('__name__', '__qualname__', 'gi_code',
                     'gi_running', 'gi_frame'):
            self.assertIs(getattr(foo(), name),
                          getattr(gen, name))
        self.assertIs(foo().cr_code, gen.gi_code)

        self.assertEqual(next(wrapper), 1)
        self.assertEqual(wrapper.send(None), 2)
        with self.assertRaisesRegex(StopIteration, 'spam'):
            wrapper.send('spam')

        gen = gen_func()
        wrapper = foo()
        wrapper.send(None)
        with self.assertRaisesRegex(Exception, 'ham'):
            wrapper.throw(Exception('ham'))

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(foo().__await__(), gen)

    def test_returning_itercoro(self):
        # [FUNCTION DOC] Checks that a decorated generator function returns a
        # coroutine-like object and that a function returning it is passed through
        # unchanged, also when decorated again.
        @types.coroutine
        def gen():
            # [FUNCTION DOC] HELPER: Generator that yields once.
            yield

        gencoro = gen()

        @types.coroutine
        def foo():
            # [FUNCTION DOC] HELPER: Returns the shared generator coroutine gencoro.
            return gencoro

        self.assertIs(foo(), gencoro)

        # decorate foo second time
        foo = types.coroutine(foo)
        self.assertIs(foo(), gencoro)

    def test_genfunc(self):
        # [FUNCTION DOC] Checks that decorating a generator function returns the same function,
        # marks its code with the iterable-coroutine flag (not the coroutine flag), also for
        # created generators, and is idempotent.
        # [FUNCTION DOC] HELPER: One-line generator function yielding once.
        def gen(): yield
        self.assertIs(types.coroutine(gen), gen)
        self.assertIs(types.coroutine(types.coroutine(gen)), gen)

        self.assertTrue(gen.__code__.co_flags & inspect.CO_ITERABLE_COROUTINE)
        self.assertFalse(gen.__code__.co_flags & inspect.CO_COROUTINE)

        g = gen()
        self.assertTrue(g.gi_code.co_flags & inspect.CO_ITERABLE_COROUTINE)
        self.assertFalse(g.gi_code.co_flags & inspect.CO_COROUTINE)

        self.assertIs(types.coroutine(gen), gen)

    def test_wrapper_object(self):
        # [FUNCTION DOC] Checks the wrapper object: its repr mentions GeneratorWrapper, equals
        # str, and its attribute list includes the await, iter, next, coroutine and generator
        # state attributes and send, close, throw.
        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding once.
            yield
        @types.coroutine
        def coro():
            # [FUNCTION DOC] HELPER: Returns a new generator from gen.
            return gen()

        wrapper = coro()
        self.assertIn('GeneratorWrapper', repr(wrapper))
        self.assertEqual(repr(wrapper), str(wrapper))
        self.assertTrue(set(dir(wrapper)).issuperset({
            '__await__', '__iter__', '__next__', 'cr_code', 'cr_running',
            'cr_frame', 'gi_code', 'gi_frame', 'gi_running', 'send',
            'close', 'throw'}))


class FunctionTests(CPythonTestCase):
    # [CLASS DOC] Tests of constructing types.FunctionType with explicit defaults.
    def test_function_type_defaults(self):
        # [FUNCTION DOC] Checks that a function built from a code object with positional
        # defaults and keyword-only defaults has those defaults, can be called with them (result
        # 6), and that passing None for both leaves them None.
        def ex(a, /, b, *, c):
            # [FUNCTION DOC] HELPER: Returns the sum a + b + c.
            return a + b + c

        func = types.FunctionType(
            ex.__code__, {}, "func", (1, 2), None, {'c': 3},
        )

        self.assertEqual(func(), 6)
        self.assertEqual(func.__defaults__, (1, 2))
        self.assertEqual(func.__kwdefaults__, {'c': 3})

        func = types.FunctionType(
            ex.__code__, {}, "func", None, None, None,
        )
        self.assertEqual(func.__defaults__, None)
        self.assertEqual(func.__kwdefaults__, None)

    def test_function_type_wrong_defaults(self):
        # [FUNCTION DOC] Checks that passing a non-tuple for the defaults argument
        # (argument 4) or a non-dict for keyword defaults (argument 6) raises TypeError
        # naming that argument number.
        def ex(a, /, b, *, c):
            # [FUNCTION DOC] HELPER: Returns the sum a + b + c.
            return a + b + c

        with self.assertRaisesRegex(TypeError, 'arg 4'):
            types.FunctionType(
                ex.__code__, {}, "func", 1, None, {'c': 3},
            )
        with self.assertRaisesRegex(TypeError, 'arg 6'):
            types.FunctionType(
                ex.__code__, {}, "func", None, None, 3,
            )


class SubinterpreterTests(CPythonTestCase):
    # [CLASS DOC] Tests that need subinterpreter support: compares the representation of static
    # type slot wrappers between the main interpreter and a subinterpreter.

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class fixture. Binds the module-level name interpreters from the test
        # support package, raising SkipTest if subinterpreter support is unavailable, imports
        # the channels module and then runs the base setup.
        global interpreters
        try:
            from test.support import interpreters
        except ModuleNotFoundError:
            raise unittest.SkipTest('subinterpreters required')
        import test.support.interpreters.channels
        super().setUpClass()

    @cpython_only
    @no_rerun('channels (and queues) might have a refleak; see gh-122199')
    def test_static_types_inherited_slots(self):
        # [FUNCTION DOC] CPython-only, not rerun on refleaks. For every builtin type and slot
        # wrapper builds a script that sends the slot's repr over a channel, runs the script in
        # the main interpreter, records the results, runs the same script in a new
        # subinterpreter, and asserts each result equals the main interpreter's (subTest per
        # type and slot). The assert in the loop verifies the message identity of the main run.
        rch, sch = interpreters.channels.create()

        slots = []
        script = ''
        for cls in iter_builtin_types():
            for slot, own in iter_slot_wrappers(cls):
                slots.append((cls, slot, own))
                script += textwrap.dedent(f"""
                    text = repr({cls.__name__}.{slot})
                    sch.send_nowait(({cls.__name__!r}, {slot!r}, text))
                    """)

        exec(script)
        all_expected = []
        for cls, slot, _ in slots:
            result = rch.recv()
            assert result == (cls.__name__, slot, result[-1]), (cls, slot, result)
            all_expected.append(result)

        interp = interpreters.create()
        interp.exec('from test.support import interpreters')
        interp.prepare_main(sch=sch)
        interp.exec(script)

        for i, (cls, slot, _) in enumerate(slots):
            with self.subTest(cls=cls, slot=slot):
                expected = all_expected[i]
                result = rch.recv()
                self.assertEqual(result, expected)


if __name__ == '__main__':
    run_tests()
