# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_operator adapted to run under Dynamo (CPythonTestCase). Tests
# the operator module: comparison, arithmetic, bitwise, sequence, in-place helpers,
# attrgetter/itemgetter/methodcaller (behavior, errors, signatures, pickling), length_hint and
# call. Each behavior suite runs against both a pure-Python copy and the C accelerated copy of
# operator. Import time side effects: imports CPython test support helpers and creates the two
# fresh operator module copies py_operator (C module blocked) and c_operator (may be None when
# the C module is unavailable); running the file calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_operator.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

import unittest
import inspect
import pickle
import sys
from decimal import Decimal
from fractions import Fraction

from test import support
from test.support import import_helper


py_operator = import_helper.import_fresh_module('operator',
                                                blocked=['_operator'])
c_operator = import_helper.import_fresh_module('operator',
                                               fresh=['_operator'])

class Seq1:
    # [CLASS DOC] HELPER: minimal sequence wrapper around a list, used as a concat/repeat
    # operand. Supports length, item access, addition with another wrapper (list concatenation)
    # and multiplication by a count on either side. Attribute lst holds the list.
    def __init__(self, lst):
        # [FUNCTION DOC] HELPER: store the wrapped list in the attribute lst.
        self.lst = lst
    def __len__(self):
        # [FUNCTION DOC] HELPER: return the length of the wrapped list.
        return len(self.lst)
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: return the wrapped list item or slice for the index.
        return self.lst[i]
    def __add__(self, other):
        # [FUNCTION DOC] HELPER: return a new plain list joining this list and the other
        # wrapper's list.
        return self.lst + other.lst
    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: return the wrapped list repeated by the other operand.
        return self.lst * other
    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: return the wrapped list repeated by the other (left) operand.
        return other * self.lst

class Seq2(object):
    # [CLASS DOC] HELPER: minimal sequence wrapper around a list, used as a concat/repeat
    # operand. Supports length, item access, addition with another wrapper (list concatenation)
    # and multiplication by a count on either side. Attribute lst holds the list.
    def __init__(self, lst):
        # [FUNCTION DOC] HELPER: store the wrapped list in the attribute lst.
        self.lst = lst
    def __len__(self):
        # [FUNCTION DOC] HELPER: return the length of the wrapped list.
        return len(self.lst)
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: return the wrapped list item or slice for the index.
        return self.lst[i]
    def __add__(self, other):
        # [FUNCTION DOC] HELPER: return a new plain list joining this list and the other
        # wrapper's list.
        return self.lst + other.lst
    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: return the wrapped list repeated by the other operand.
        return self.lst * other
    def __rmul__(self, other):
        # [FUNCTION DOC] HELPER: return the wrapped list repeated by the other (left) operand.
        return other * self.lst

class BadIterable:
    # [CLASS DOC] HELPER: object whose iteration always fails; used to check that operator
    # functions propagate errors raised while iterating.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: always raise ZeroDivisionError.
        raise ZeroDivisionError


class OperatorTestCase:
    # [CLASS DOC] Mixin (not a TestCase itself) holding one behavior test per operator function.
    # The concrete subclass must set the attribute module to the operator implementation under
    # test (pure Python or C). No other state.
    def test___all__(self):
        # [FUNCTION DOC] Check the set of public names in __all__ of the module equals the set
        # of non-dunder module attributes whose defining module is operator or _operator.
        operator = self.module
        actual_all = set(operator.__all__)
        computed_all = set()
        for name in vars(operator):
            if name.startswith('__'):
                continue
            value = getattr(operator, name)
            if value.__module__ in ('operator', '_operator'):
                computed_all.add(name)
        self.assertSetEqual(computed_all, actual_all)

    def test_lt(self):
        # [FUNCTION DOC] Check lt raises TypeError with missing arguments or on complex numbers,
        # and returns correct booleans for int/float comparisons.
        operator = self.module
        self.assertRaises(TypeError, operator.lt)
        self.assertRaises(TypeError, operator.lt, 1j, 2j)
        self.assertFalse(operator.lt(1, 0))
        self.assertFalse(operator.lt(1, 0.0))
        self.assertFalse(operator.lt(1, 1))
        self.assertFalse(operator.lt(1, 1.0))
        self.assertTrue(operator.lt(1, 2))
        self.assertTrue(operator.lt(1, 2.0))

    def test_le(self):
        # [FUNCTION DOC] Check le raises TypeError for missing arguments and complex operands,
        # and returns correct booleans for int/float comparisons including equality.
        operator = self.module
        self.assertRaises(TypeError, operator.le)
        self.assertRaises(TypeError, operator.le, 1j, 2j)
        self.assertFalse(operator.le(1, 0))
        self.assertFalse(operator.le(1, 0.0))
        self.assertTrue(operator.le(1, 1))
        self.assertTrue(operator.le(1, 1.0))
        self.assertTrue(operator.le(1, 2))
        self.assertTrue(operator.le(1, 2.0))

    def test_eq(self):
        # [FUNCTION DOC] Check eq raises TypeError without arguments, propagates an exception
        # raised by a custom equality method, and gives correct booleans for numbers.
        operator = self.module
        class C(object):
            # [CLASS DOC] HELPER: object whose equality method always raises SyntaxError.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(TypeError, operator.eq)
        self.assertRaises(SyntaxError, operator.eq, C(), C())
        self.assertFalse(operator.eq(1, 0))
        self.assertFalse(operator.eq(1, 0.0))
        self.assertTrue(operator.eq(1, 1))
        self.assertTrue(operator.eq(1, 1.0))
        self.assertFalse(operator.eq(1, 2))
        self.assertFalse(operator.eq(1, 2.0))

    def test_ne(self):
        # [FUNCTION DOC] Check ne raises TypeError without arguments, propagates an exception
        # from a custom inequality method, and gives correct booleans for numbers.
        operator = self.module
        class C(object):
            # [CLASS DOC] HELPER: object whose inequality method always raises SyntaxError.
            def __ne__(self, other):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(TypeError, operator.ne)
        self.assertRaises(SyntaxError, operator.ne, C(), C())
        self.assertTrue(operator.ne(1, 0))
        self.assertTrue(operator.ne(1, 0.0))
        self.assertFalse(operator.ne(1, 1))
        self.assertFalse(operator.ne(1, 1.0))
        self.assertTrue(operator.ne(1, 2))
        self.assertTrue(operator.ne(1, 2.0))

    def test_ge(self):
        # [FUNCTION DOC] Check ge raises TypeError for missing arguments and complex operands,
        # and gives correct booleans for int/float comparisons.
        operator = self.module
        self.assertRaises(TypeError, operator.ge)
        self.assertRaises(TypeError, operator.ge, 1j, 2j)
        self.assertTrue(operator.ge(1, 0))
        self.assertTrue(operator.ge(1, 0.0))
        self.assertTrue(operator.ge(1, 1))
        self.assertTrue(operator.ge(1, 1.0))
        self.assertFalse(operator.ge(1, 2))
        self.assertFalse(operator.ge(1, 2.0))

    def test_gt(self):
        # [FUNCTION DOC] Check gt raises TypeError for missing arguments and complex operands,
        # and gives correct booleans for int/float comparisons.
        operator = self.module
        self.assertRaises(TypeError, operator.gt)
        self.assertRaises(TypeError, operator.gt, 1j, 2j)
        self.assertTrue(operator.gt(1, 0))
        self.assertTrue(operator.gt(1, 0.0))
        self.assertFalse(operator.gt(1, 1))
        self.assertFalse(operator.gt(1, 1.0))
        self.assertFalse(operator.gt(1, 2))
        self.assertFalse(operator.gt(1, 2.0))

    def test_abs(self):
        # [FUNCTION DOC] Check abs raises TypeError without arguments or for None, and returns
        # the absolute value of integers.
        operator = self.module
        self.assertRaises(TypeError, operator.abs)
        self.assertRaises(TypeError, operator.abs, None)
        self.assertEqual(operator.abs(-1), 1)
        self.assertEqual(operator.abs(1), 1)

    def test_add(self):
        # [FUNCTION DOC] Check add raises TypeError for missing arguments or None operands and
        # adds two integers.
        operator = self.module
        self.assertRaises(TypeError, operator.add)
        self.assertRaises(TypeError, operator.add, None, None)
        self.assertEqual(operator.add(3, 4), 7)

    def test_bitwise_and(self):
        # [FUNCTION DOC] Check and_ raises TypeError for missing arguments or None operands and
        # computes bitwise and of integers.
        operator = self.module
        self.assertRaises(TypeError, operator.and_)
        self.assertRaises(TypeError, operator.and_, None, None)
        self.assertEqual(operator.and_(0xf, 0xa), 0xa)

    def test_concat(self):
        # [FUNCTION DOC] Check concat joins strings, lists and the Seq1/Seq2 wrappers, and
        # raises TypeError for missing arguments, None, or plain integers.
        operator = self.module
        self.assertRaises(TypeError, operator.concat)
        self.assertRaises(TypeError, operator.concat, None, None)
        self.assertEqual(operator.concat('py', 'thon'), 'python')
        self.assertEqual(operator.concat([1, 2], [3, 4]), [1, 2, 3, 4])
        self.assertEqual(operator.concat(Seq1([5, 6]), Seq1([7])), [5, 6, 7])
        self.assertEqual(operator.concat(Seq2([5, 6]), Seq2([7])), [5, 6, 7])
        self.assertRaises(TypeError, operator.concat, 13, 29)

    def test_countOf(self):
        # [FUNCTION DOC] Check countOf counts matching items, raises TypeError for bad
        # arguments, propagates an iteration error, and counts items that match by identity
        # (NaN) or by equality.
        operator = self.module
        self.assertRaises(TypeError, operator.countOf)
        self.assertRaises(TypeError, operator.countOf, None, None)
        self.assertRaises(ZeroDivisionError, operator.countOf, BadIterable(), 1)
        self.assertEqual(operator.countOf([1, 2, 1, 3, 1, 4], 3), 1)
        self.assertEqual(operator.countOf([1, 2, 1, 3, 1, 4], 5), 0)
        # is but not ==
        nan = float("nan")
        self.assertEqual(operator.countOf([nan, nan, 21], nan), 2)
        # == but not is
        self.assertEqual(operator.countOf([{}, 1, {}, 2], {}), 2)

    def test_delitem(self):
        # [FUNCTION DOC] Check delitem raises TypeError for bad arguments, removes the item at
        # an index from the list in place, and returns None.
        operator = self.module
        a = [4, 3, 2, 1]
        self.assertRaises(TypeError, operator.delitem, a)
        self.assertRaises(TypeError, operator.delitem, a, None)
        self.assertIsNone(operator.delitem(a, 1))
        self.assertEqual(a, [4, 2, 1])

    def test_floordiv(self):
        # [FUNCTION DOC] Check floordiv raises TypeError for bad arguments and returns the
        # integer quotient.
        operator = self.module
        self.assertRaises(TypeError, operator.floordiv, 5)
        self.assertRaises(TypeError, operator.floordiv, None, None)
        self.assertEqual(operator.floordiv(5, 2), 2)

    def test_truediv(self):
        # [FUNCTION DOC] Check truediv raises TypeError for bad arguments and returns the float
        # quotient.
        operator = self.module
        self.assertRaises(TypeError, operator.truediv, 5)
        self.assertRaises(TypeError, operator.truediv, None, None)
        self.assertEqual(operator.truediv(5, 2), 2.5)

    def test_getitem(self):
        # [FUNCTION DOC] Check getitem raises TypeError for bad arguments or a None index and
        # returns the element of a range.
        operator = self.module
        a = range(10)
        self.assertRaises(TypeError, operator.getitem)
        self.assertRaises(TypeError, operator.getitem, a, None)
        self.assertEqual(operator.getitem(a, 2), 2)

    def test_indexOf(self):
        # [FUNCTION DOC] Check indexOf returns the first matching position, raises TypeError for
        # bad arguments, propagates iteration errors, raises ValueError when missing, matches by
        # identity or equality, and leaves an iterator positioned right after the match.
        operator = self.module
        self.assertRaises(TypeError, operator.indexOf)
        self.assertRaises(TypeError, operator.indexOf, None, None)
        self.assertRaises(ZeroDivisionError, operator.indexOf, BadIterable(), 1)
        self.assertEqual(operator.indexOf([4, 3, 2, 1], 3), 1)
        self.assertRaises(ValueError, operator.indexOf, [4, 3, 2, 1], 0)
        nan = float("nan")
        self.assertEqual(operator.indexOf([nan, nan, 21], nan), 0)
        self.assertEqual(operator.indexOf([{}, 1, {}, 2], {}), 0)
        it = iter('leave the iterator at exactly the position after the match')
        self.assertEqual(operator.indexOf(it, 'a'), 2)
        self.assertEqual(next(it), 'v')

    def test_invert(self):
        # [FUNCTION DOC] Check invert raises TypeError for missing arguments or None, and that
        # inv is the bitwise complement of an integer.
        operator = self.module
        self.assertRaises(TypeError, operator.invert)
        self.assertRaises(TypeError, operator.invert, None)
        self.assertEqual(operator.inv(4), -5)

    def test_lshift(self):
        # [FUNCTION DOC] Check lshift raises TypeError for bad arguments, shifts integers left,
        # and raises ValueError for a negative shift count.
        operator = self.module
        self.assertRaises(TypeError, operator.lshift)
        self.assertRaises(TypeError, operator.lshift, None, 42)
        self.assertEqual(operator.lshift(5, 1), 10)
        self.assertEqual(operator.lshift(5, 0), 5)
        self.assertRaises(ValueError, operator.lshift, 2, -1)

    def test_mod(self):
        # [FUNCTION DOC] Check mod raises TypeError for bad arguments and returns the remainder.
        operator = self.module
        self.assertRaises(TypeError, operator.mod)
        self.assertRaises(TypeError, operator.mod, None, 42)
        self.assertEqual(operator.mod(5, 2), 1)

    def test_mul(self):
        # [FUNCTION DOC] Check mul raises TypeError for bad arguments and multiplies integers.
        operator = self.module
        self.assertRaises(TypeError, operator.mul)
        self.assertRaises(TypeError, operator.mul, None, None)
        self.assertEqual(operator.mul(5, 2), 10)

    def test_matmul(self):
        # [FUNCTION DOC] Check matmul raises TypeError for missing arguments or ints, and that
        # the at operator dispatches to a custom matrix multiplication method.
        operator = self.module
        self.assertRaises(TypeError, operator.matmul)
        self.assertRaises(TypeError, operator.matmul, 42, 42)
        class M:
            # [CLASS DOC] HELPER: object defining only a matrix multiplication method.
            def __matmul__(self, other):
                # [FUNCTION DOC] HELPER: return the other operand minus one.
                return other - 1
        self.assertEqual(M() @ 42, 41)

    def test_neg(self):
        # [FUNCTION DOC] Check neg raises TypeError for missing arguments or None and negates
        # integers including zero.
        operator = self.module
        self.assertRaises(TypeError, operator.neg)
        self.assertRaises(TypeError, operator.neg, None)
        self.assertEqual(operator.neg(5), -5)
        self.assertEqual(operator.neg(-5), 5)
        self.assertEqual(operator.neg(0), 0)
        self.assertEqual(operator.neg(-0), 0)

    def test_bitwise_or(self):
        # [FUNCTION DOC] Check or_ raises TypeError for bad arguments and computes bitwise or of
        # integers.
        operator = self.module
        self.assertRaises(TypeError, operator.or_)
        self.assertRaises(TypeError, operator.or_, None, None)
        self.assertEqual(operator.or_(0xa, 0x5), 0xf)

    def test_pos(self):
        # [FUNCTION DOC] Check pos raises TypeError for missing arguments or None and returns
        # integers unchanged including zero.
        operator = self.module
        self.assertRaises(TypeError, operator.pos)
        self.assertRaises(TypeError, operator.pos, None)
        self.assertEqual(operator.pos(5), 5)
        self.assertEqual(operator.pos(-5), -5)
        self.assertEqual(operator.pos(0), 0)
        self.assertEqual(operator.pos(-0), 0)

    def test_pow(self):
        # [FUNCTION DOC] Check pow raises TypeError for bad argument counts (zero, one or three)
        # and None operands, and computes integer exponentiation.
        operator = self.module
        self.assertRaises(TypeError, operator.pow, None, None)
        self.assertEqual(operator.pow(3,5), 3**5)
        self.assertRaises(TypeError, operator.pow)
        self.assertRaises(TypeError, operator.pow, 1)
        self.assertRaises(TypeError, operator.pow, 1, 2, 3)

    def test_rshift(self):
        # [FUNCTION DOC] Check rshift raises TypeError for bad arguments, shifts integers right,
        # and raises ValueError for a negative shift count.
        operator = self.module
        self.assertRaises(TypeError, operator.rshift)
        self.assertRaises(TypeError, operator.rshift, None, 42)
        self.assertEqual(operator.rshift(5, 1), 2)
        self.assertEqual(operator.rshift(5, 0), 5)
        self.assertRaises(ValueError, operator.rshift, 2, -1)

    def test_contains(self):
        # [FUNCTION DOC] Check contains raises TypeError for bad arguments, propagates an
        # iteration error, and reports membership in a range.
        operator = self.module
        self.assertRaises(TypeError, operator.contains)
        self.assertRaises(TypeError, operator.contains, None, None)
        self.assertRaises(ZeroDivisionError, operator.contains, BadIterable(), 1)
        self.assertTrue(operator.contains(range(4), 2))
        self.assertFalse(operator.contains(range(4), 5))

    def test_setitem(self):
        # [FUNCTION DOC] Check setitem raises TypeError for bad arguments, assigns the list
        # element in place returning None, and raises IndexError for an out of range index.
        operator = self.module
        a = list(range(3))
        self.assertRaises(TypeError, operator.setitem, a)
        self.assertRaises(TypeError, operator.setitem, a, None, None)
        self.assertIsNone(operator.setitem(a, 0, 2))
        self.assertEqual(a, [2, 1, 2])
        self.assertRaises(IndexError, operator.setitem, a, 4, 2)

    def test_sub(self):
        # [FUNCTION DOC] Check sub raises TypeError for bad arguments and subtracts integers.
        operator = self.module
        self.assertRaises(TypeError, operator.sub)
        self.assertRaises(TypeError, operator.sub, None, None)
        self.assertEqual(operator.sub(5, 2), 3)

    def test_truth(self):
        # [FUNCTION DOC] Check truth raises TypeError without arguments, propagates an exception
        # from a custom truth method, and returns correct booleans for numbers and lists.
        operator = self.module
        class C(object):
            # [CLASS DOC] HELPER: object whose truth-value method always raises SyntaxError.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(TypeError, operator.truth)
        self.assertRaises(SyntaxError, operator.truth, C())
        self.assertTrue(operator.truth(5))
        self.assertTrue(operator.truth([0]))
        self.assertFalse(operator.truth(0))
        self.assertFalse(operator.truth([]))

    def test_bitwise_xor(self):
        # [FUNCTION DOC] Check xor raises TypeError for bad arguments and computes bitwise
        # exclusive or of integers.
        operator = self.module
        self.assertRaises(TypeError, operator.xor)
        self.assertRaises(TypeError, operator.xor, None, None)
        self.assertEqual(operator.xor(0xb, 0xc), 0x7)

    def test_is(self):
        # [FUNCTION DOC] Check is_ is true for the same object and false for an equal but
        # distinct string built by concatenation, and raises TypeError without arguments.
        operator = self.module
        a = b = 'xyzpdq'
        c = a[:3] + b[3:]
        self.assertTrue(operator.is_(a, b))
        self.assertFalse(operator.is_(a,c))
        self.assertRaises(TypeError, operator.is_)

    def test_is_not(self):
        # [FUNCTION DOC] Check is_not is the negation of is_ for the same object and an equal
        # distinct string, and raises TypeError without arguments.
        operator = self.module
        a = b = 'xyzpdq'
        c = a[:3] + b[3:]
        self.assertFalse(operator.is_not(a, b))
        self.assertTrue(operator.is_not(a,c))
        self.assertRaises(TypeError, operator.is_not)

    def test_attrgetter(self):
        # [FUNCTION DOC] Check attrgetter: single, multiple (tuple result) and dotted recursive
        # attribute lookup, AttributeError for missing names (including malformed dotted names),
        # TypeError for bad construction or call arguments, and that exceptions raised by custom
        # attribute lookup propagate.
        operator = self.module
        class A:
            # [CLASS DOC] HELPER: empty class used as a mutable attribute holder.
            pass
        a = A()
        a.name = 'arthur'
        f = operator.attrgetter('name')
        self.assertEqual(f(a), 'arthur')
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 'dent')
        self.assertRaises(TypeError, f, a, surname='dent')
        f = operator.attrgetter('rank')
        self.assertRaises(AttributeError, f, a)
        self.assertRaises(TypeError, operator.attrgetter, 2)
        self.assertRaises(TypeError, operator.attrgetter)

        # multiple gets
        record = A()
        record.x = 'X'
        record.y = 'Y'
        record.z = 'Z'
        self.assertEqual(operator.attrgetter('x','z','y')(record), ('X', 'Z', 'Y'))
        self.assertRaises(TypeError, operator.attrgetter, ('x', (), 'y'))

        class C(object):
            # [CLASS DOC] HELPER: object whose attribute lookup hook always raises SyntaxError.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(SyntaxError, operator.attrgetter('foo'), C())

        # recursive gets
        a = A()
        a.name = 'arthur'
        a.child = A()
        a.child.name = 'thomas'
        f = operator.attrgetter('child.name')
        self.assertEqual(f(a), 'thomas')
        self.assertRaises(AttributeError, f, a.child)
        f = operator.attrgetter('name', 'child.name')
        self.assertEqual(f(a), ('arthur', 'thomas'))
        f = operator.attrgetter('name', 'child.name', 'child.child.name')
        self.assertRaises(AttributeError, f, a)
        f = operator.attrgetter('child.')
        self.assertRaises(AttributeError, f, a)
        f = operator.attrgetter('.child')
        self.assertRaises(AttributeError, f, a)

        a.child.child = A()
        a.child.child.name = 'johnson'
        f = operator.attrgetter('child.child.name')
        self.assertEqual(f(a), 'johnson')
        f = operator.attrgetter('name', 'child.name', 'child.child.name')
        self.assertEqual(f(a), ('arthur', 'thomas', 'johnson'))

    def test_itemgetter(self):
        # [FUNCTION DOC] Check itemgetter: single and multiple item lookup, negative index and
        # slice, strings, dict keys, tuple subclasses, ranges, the documented sort-by-count
        # example, IndexError/KeyError/TypeError for bad lookups or arguments, and propagation
        # of custom lookup exceptions.
        operator = self.module
        a = 'ABCDE'
        f = operator.itemgetter(2)
        self.assertEqual(f(a), 'C')
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 3)
        self.assertRaises(TypeError, f, a, size=3)
        f = operator.itemgetter(10)
        self.assertRaises(IndexError, f, a)

        class C(object):
            # [CLASS DOC] HELPER: object whose item lookup always raises SyntaxError.
            def __getitem__(self, name):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(SyntaxError, operator.itemgetter(42), C())

        f = operator.itemgetter('name')
        self.assertRaises(TypeError, f, a)
        self.assertRaises(TypeError, operator.itemgetter)

        d = dict(key='val')
        f = operator.itemgetter('key')
        self.assertEqual(f(d), 'val')
        f = operator.itemgetter('nonkey')
        self.assertRaises(KeyError, f, d)

        # example used in the docs
        inventory = [('apple', 3), ('banana', 2), ('pear', 5), ('orange', 1)]
        getcount = operator.itemgetter(1)
        self.assertEqual(list(map(getcount, inventory)), [3, 2, 5, 1])
        self.assertEqual(sorted(inventory, key=getcount),
            [('orange', 1), ('banana', 2), ('apple', 3), ('pear', 5)])

        # multiple gets
        data = list(map(str, range(20)))
        self.assertEqual(operator.itemgetter(2,10,5)(data), ('2', '10', '5'))
        self.assertRaises(TypeError, operator.itemgetter(2, 'x', 5), data)

        # interesting indices
        t = tuple('abcde')
        self.assertEqual(operator.itemgetter(-1)(t), 'e')
        self.assertEqual(operator.itemgetter(slice(2, 4))(t), ('c', 'd'))

        # interesting sequences
        class T(tuple):
            # [CLASS DOC] HELPER: plain tuple subclass to check item lookup works on subclasses.
            'Tuple subclass'
            pass
        self.assertEqual(operator.itemgetter(0)(T('abc')), 'a')
        self.assertEqual(operator.itemgetter(0)(['a', 'b', 'c']), 'a')
        self.assertEqual(operator.itemgetter(0)(range(100, 200)), 100)

    def test_methodcaller(self):
        # [FUNCTION DOC] Check methodcaller: construction errors, call with positional and
        # keyword arguments, defaults, wrong call arity raising TypeError, and keyword arguments
        # named self or name being passed through.
        operator = self.module
        self.assertRaises(TypeError, operator.methodcaller)
        self.assertRaises(TypeError, operator.methodcaller, 12)
        class A:
            # [CLASS DOC] HELPER: class with three methods used to test method calling: foo adds
            # its first two arguments, bar returns its argument with default 42, baz returns the
            # name and self keyword values.
            def foo(self, *args, **kwds):
                # [FUNCTION DOC] HELPER: return the sum of the first two positional arguments
                # (IndexError if fewer).
                return args[0] + args[1]
            def bar(self, f=42):
                # [FUNCTION DOC] HELPER: return the argument f (default 42).
                return f
            def baz(*args, **kwds):
                # [FUNCTION DOC] HELPER: return a pair of the keyword arguments name and self.
                return kwds['name'], kwds['self']
        a = A()
        f = operator.methodcaller('foo')
        self.assertRaises(IndexError, f, a)
        f = operator.methodcaller('foo', 1, 2)
        self.assertEqual(f(a), 3)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 3)
        self.assertRaises(TypeError, f, a, spam=3)
        f = operator.methodcaller('bar')
        self.assertEqual(f(a), 42)
        self.assertRaises(TypeError, f, a, a)
        f = operator.methodcaller('bar', f=5)
        self.assertEqual(f(a), 5)
        f = operator.methodcaller('baz', name='spam', self='eggs')
        self.assertEqual(f(a), ('spam', 'eggs'))

    def test_inplace(self):
        # [FUNCTION DOC] Check each in-place operator function (iadd through ixor, plus iconcat)
        # calls the matching in-place special method of the object and returns its result;
        # iconcat uses the in-place add method.
        operator = self.module
        class C(object):
            # [CLASS DOC] HELPER: object whose in-place special methods each return a distinct
            # string naming the operation, and which has item access so it counts as a sequence.
            # (Methods are one-line definitions.)
            def __iadd__     (self, other): return "iadd"
            def __iand__     (self, other): return "iand"
            def __ifloordiv__(self, other): return "ifloordiv"
            def __ilshift__  (self, other): return "ilshift"
            def __imod__     (self, other): return "imod"
            def __imul__     (self, other): return "imul"
            def __imatmul__  (self, other): return "imatmul"
            def __ior__      (self, other): return "ior"
            def __ipow__     (self, other): return "ipow"
            def __irshift__  (self, other): return "irshift"
            def __isub__     (self, other): return "isub"
            def __itruediv__ (self, other): return "itruediv"
            def __ixor__     (self, other): return "ixor"
            def __getitem__(self, other): return 5  # so that C is a sequence
        c = C()
        self.assertEqual(operator.iadd     (c, 5), "iadd")
        self.assertEqual(operator.iand     (c, 5), "iand")
        self.assertEqual(operator.ifloordiv(c, 5), "ifloordiv")
        self.assertEqual(operator.ilshift  (c, 5), "ilshift")
        self.assertEqual(operator.imod     (c, 5), "imod")
        self.assertEqual(operator.imul     (c, 5), "imul")
        self.assertEqual(operator.imatmul  (c, 5), "imatmul")
        self.assertEqual(operator.ior      (c, 5), "ior")
        self.assertEqual(operator.ipow     (c, 5), "ipow")
        self.assertEqual(operator.irshift  (c, 5), "irshift")
        self.assertEqual(operator.isub     (c, 5), "isub")
        self.assertEqual(operator.itruediv (c, 5), "itruediv")
        self.assertEqual(operator.ixor     (c, 5), "ixor")
        self.assertEqual(operator.iconcat  (c, c), "iadd")

    def test_iconcat_without_getitem(self):
        # [FUNCTION DOC] Check iconcat on an int raises TypeError saying the object cannot be
        # concatenated.
        operator = self.module

        msg = "'int' object can't be concatenated"
        with self.assertRaisesRegex(TypeError, msg):
            operator.iconcat(1, 0.5)

    def test_index(self):
        # [FUNCTION DOC] Check index returns integers and objects with an index method, and
        # raises AttributeError or TypeError for float, Fraction, Decimal and None.
        operator = self.module
        class X:
            # [CLASS DOC] HELPER: object whose index method returns 1.
            def __index__(self):
                # [FUNCTION DOC] HELPER: return the integer 1.
                return 1

        self.assertEqual(operator.index(X()), 1)
        self.assertEqual(operator.index(0), 0)
        self.assertEqual(operator.index(1), 1)
        self.assertEqual(operator.index(2), 2)
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(1.5)
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(Fraction(3, 7))
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(Decimal(1))
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(None)

    def test_not_(self):
        # [FUNCTION DOC] Check not_ raises TypeError without arguments, propagates an exception
        # from a custom truth method, and negates truthiness of numbers and lists.
        operator = self.module
        class C:
            # [CLASS DOC] HELPER: object whose truth-value method always raises SyntaxError.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: always raise SyntaxError.
                raise SyntaxError
        self.assertRaises(TypeError, operator.not_)
        self.assertRaises(SyntaxError, operator.not_, C())
        self.assertFalse(operator.not_(5))
        self.assertFalse(operator.not_([0]))
        self.assertTrue(operator.not_(0))
        self.assertTrue(operator.not_([]))

    def test_length_hint(self):
        # [FUNCTION DOC] Check length_hint uses len, iterator hints, and a custom length hint
        # method; falls back to the default for NotImplemented, TypeError, or no hint; raises
        # TypeError for non-int hints or default, ValueError for a negative hint, and propagates
        # LookupError.
        operator = self.module
        class X(object):
            # [CLASS DOC] HELPER: object whose length hint method returns the stored value, or
            # raises it when the value is an exception class. Attribute value holds it.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: store the value in the attribute value.
                self.value = value

            def __length_hint__(self):
                # [FUNCTION DOC] HELPER: raise the stored value if it is a class, else return
                # it.
                if type(self.value) is type:
                    raise self.value
                else:
                    return self.value

        self.assertEqual(operator.length_hint([], 2), 0)
        self.assertEqual(operator.length_hint(iter([1, 2, 3])), 3)

        self.assertEqual(operator.length_hint(X(2)), 2)
        self.assertEqual(operator.length_hint(X(NotImplemented), 4), 4)
        self.assertEqual(operator.length_hint(X(TypeError), 12), 12)
        with self.assertRaises(TypeError):
            operator.length_hint(X("abc"))
        with self.assertRaises(ValueError):
            operator.length_hint(X(-2))
        with self.assertRaises(LookupError):
            operator.length_hint(X(LookupError))

        class Y: pass

        msg = "'str' object cannot be interpreted as an integer"
        with self.assertRaisesRegex(TypeError, msg):
            operator.length_hint(X(2), "abc")
        self.assertEqual(operator.length_hint(Y(), 10), 10)

    def test_call(self):
        # [FUNCTION DOC] Check call invokes a function with positional and keyword arguments and
        # returns its result (a pair of args and kwargs).
        operator = self.module

        def func(*args, **kwargs): return args, kwargs

        self.assertEqual(operator.call(func), ((), {}))
        self.assertEqual(operator.call(func, 0, 1), ((0, 1), {}))
        self.assertEqual(operator.call(func, a=2, obj=3),
                         ((), {"a": 2, "obj": 3}))
        self.assertEqual(operator.call(func, 0, 1, a=2, obj=3),
                         ((0, 1), {"a": 2, "obj": 3}))

    def test_dunder_is_original(self):
        # [FUNCTION DOC] Check each public name of the module that has a matching double
        # underscore alias is the same object as that alias.
        operator = self.module

        names = [name for name in dir(operator) if not name.startswith('_')]
        for name in names:
            orig = getattr(operator, name)
            dunder = getattr(operator, '__' + name.strip('_') + '__', None)
            if dunder:
                self.assertIs(dunder, orig)

    @support.requires_docstrings
    def test_attrgetter_signature(self):
        # [FUNCTION DOC] Check the signature strings of attrgetter and of an attrgetter instance
        # (skipped when docstrings are stripped).
        operator = self.module
        sig = inspect.signature(operator.attrgetter)
        self.assertEqual(str(sig), '(attr, /, *attrs)')
        sig = inspect.signature(operator.attrgetter('x', 'z', 'y'))
        self.assertEqual(str(sig), '(obj, /)')

    @support.requires_docstrings
    def test_itemgetter_signature(self):
        # [FUNCTION DOC] Check the signature strings of itemgetter and of an itemgetter instance
        # (skipped when docstrings are stripped).
        operator = self.module
        sig = inspect.signature(operator.itemgetter)
        self.assertEqual(str(sig), '(item, /, *items)')
        sig = inspect.signature(operator.itemgetter(2, 3, 5))
        self.assertEqual(str(sig), '(obj, /)')

    @support.requires_docstrings
    def test_methodcaller_signature(self):
        # [FUNCTION DOC] Check the signature strings of methodcaller and of a methodcaller
        # instance (skipped when docstrings are stripped).
        operator = self.module
        sig = inspect.signature(operator.methodcaller)
        self.assertEqual(str(sig), '(name, /, *args, **kwargs)')
        sig = inspect.signature(operator.methodcaller('foo', 2, y=3))
        self.assertEqual(str(sig), '(obj, /)')


class PyOperatorTestCase(OperatorTestCase, __TestCase):
    # [CLASS DOC] Runs the operator behavior tests against the pure-Python operator copy.
    module = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class COperatorTestCase(OperatorTestCase, __TestCase):
    # [CLASS DOC] Runs the operator behavior tests against the C operator module; skipped when
    # it is unavailable.
    module = c_operator


class OperatorPickleTestCase:
    # [CLASS DOC] Mixin for pickling round-trip tests of attrgetter, itemgetter and
    # methodcaller. Subclasses set module (used to pickle) and module2 (used to unpickle).
    def copy(self, obj, proto):
        # [FUNCTION DOC] HELPER: pickle obj with the given protocol while the module registry
        # maps operator to the source module, then unpickle it with it mapped to the target
        # module, returning the copy.
        with support.swap_item(sys.modules, 'operator', self.module):
            pickled = pickle.dumps(obj, proto)
        with support.swap_item(sys.modules, 'operator', self.module2):
            return pickle.loads(pickled)

    def test_attrgetter(self):
        # [FUNCTION DOC] For every pickle protocol, round-trip single, multiple and dotted
        # attrgetter objects and check the copy has the same repr and gives the same result.
        attrgetter = self.module.attrgetter
        class A:
            # [CLASS DOC] HELPER: empty class used as a mutable attribute holder.
            pass
        a = A()
        a.x = 'X'
        a.y = 'Y'
        a.z = 'Z'
        a.t = A()
        a.t.u = A()
        a.t.u.v = 'V'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = attrgetter('x')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # multiple gets
                f = attrgetter('x', 'y', 'z')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # recursive gets
                f = attrgetter('t.u.v')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))

    def test_itemgetter(self):
        # [FUNCTION DOC] For every pickle protocol, round-trip single and multiple itemgetter
        # objects and check same repr and same result.
        itemgetter = self.module.itemgetter
        a = 'ABCDE'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = itemgetter(2)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # multiple gets
                f = itemgetter(2, 0, 4)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))

    def test_methodcaller(self):
        # [FUNCTION DOC] For every pickle protocol, round-trip methodcaller objects with no,
        # positional and keyword arguments, checking the repr (except for several keyword
        # arguments) and results.
        methodcaller = self.module.methodcaller
        class A:
            # [CLASS DOC] HELPER: class with methods foo, bar and baz used as the target for
            # method callers.
            def foo(self, *args, **kwds):
                # [FUNCTION DOC] HELPER: return the sum of the first two positional arguments.
                return args[0] + args[1]
            def bar(self, f=42):
                # [FUNCTION DOC] HELPER: return the argument f (default 42).
                return f
            def baz(*args, **kwds):
                # [FUNCTION DOC] HELPER: return a pair of the keyword arguments name and self.
                return kwds['name'], kwds['self']
        a = A()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = methodcaller('bar')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # positional args
                f = methodcaller('foo', 1, 2)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # keyword args
                f = methodcaller('bar', f=5)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                f = methodcaller('baz', self='eggs', name='spam')
                f2 = self.copy(f, proto)
                # Can't test repr consistently with multiple keyword args
                self.assertEqual(f2(a), f(a))

class PyPyOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    # [CLASS DOC] Pickle round trip with the pure-Python module on both sides.
    module = py_operator
    module2 = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class PyCOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    # [CLASS DOC] Pickle with the Python module, unpickle with the C module; skipped without the
    # C module.
    module = py_operator
    module2 = c_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class CPyOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    # [CLASS DOC] Pickle with the C module, unpickle with the Python module; skipped without the
    # C module.
    module = c_operator
    module2 = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class CCOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    # [CLASS DOC] Pickle round trip with the C module on both sides; skipped without the C
    # module.
    module = c_operator
    module2 = c_operator


if __name__ == "__main__":
    run_tests()
