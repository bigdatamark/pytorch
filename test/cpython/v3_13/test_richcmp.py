# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_richcmp suite for rich
# comparison operators (lt, le, eq, ne, gt, ge). Defines Number and Vector helper types
# with custom comparison methods, a table opmap mapping each operator name to three
# equivalent callables (lambda, operator function, dunder function), and test cases for
# vectors, numbers, misbehaving objects, recursion, error messages, dicts and lists.
# No import-time side effects besides class and table creation.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_richcmp.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Tests for rich comparisons

import unittest
from test import support

import operator

class Number:
    # [CLASS DOC] HELPER: Wrapper around a value x whose rich comparison methods compare the
    # wrapped value against the other operand and return that result directly (so the result
    # may be a plain bool). The legacy cmp method must never be called.

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Store the wrapped value in attribute x.
        self.x = x

    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Less-than; returns the wrapped value compared with other.
        return self.x < other

    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Less-or-equal; returns the wrapped value compared with other.
        return self.x <= other

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality; returns the wrapped value compared with other.
        return self.x == other

    def __ne__(self, other):
        # [FUNCTION DOC] HELPER: Inequality; returns the wrapped value compared with other.
        return self.x != other

    def __gt__(self, other):
        # [FUNCTION DOC] HELPER: Greater-than; returns the wrapped value compared with other.
        return self.x > other

    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Greater-or-equal; returns the wrapped value compared with other.
        return self.x >= other

    def __cmp__(self, other):
        # [FUNCTION DOC] HELPER: Legacy comparison hook; always fails the test (TestFailed) because
        # it must not be used.
        raise support.TestFailed("Number.__cmp__() should not be called")

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Return a text form showing the wrapped value.
        return "Number(%r)" % (self.x, )

class Vector:
    # [CLASS DOC] HELPER: Sequence wrapper around data whose comparisons are element-wise and
    # return a new Vector of booleans. Unhashable, and using it as a truth value raises
    # TypeError. Comparing with a sequence of different length raises ValueError.

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Store the data sequence in attribute data.
        self.data = data

    def __len__(self):
        # [FUNCTION DOC] HELPER: Return the number of elements.
        return len(self.data)

    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the element at index i.
        return self.data[i]

    def __setitem__(self, i, v):
        # [FUNCTION DOC] HELPER: Set the element at index i to v.
        self.data[i] = v

    __hash__ = None # Vectors cannot be hashed

    def __bool__(self):
        # [FUNCTION DOC] HELPER: Truth value test; always raises TypeError.
        raise TypeError("Vectors cannot be used in Boolean contexts")

    def __cmp__(self, other):
        # [FUNCTION DOC] HELPER: Legacy comparison hook; always fails the test (TestFailed).
        raise support.TestFailed("Vector.__cmp__() should not be called")

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Return a text form showing the data.
        return "Vector(%r)" % (self.data, )

    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise less-than against a Vector or sequence; returns a
        # Vector of bools. Raises ValueError on length mismatch.
        return Vector([a < b for a, b in zip(self.data, self.__cast(other))])

    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise less-or-equal; returns a Vector of bools.
        return Vector([a <= b for a, b in zip(self.data, self.__cast(other))])

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise equality; returns a Vector of bools.
        return Vector([a == b for a, b in zip(self.data, self.__cast(other))])

    def __ne__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise inequality; returns a Vector of bools.
        return Vector([a != b for a, b in zip(self.data, self.__cast(other))])

    def __gt__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise greater-than; returns a Vector of bools.
        return Vector([a > b for a, b in zip(self.data, self.__cast(other))])

    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Element-wise greater-or-equal; returns a Vector of bools.
        return Vector([a >= b for a, b in zip(self.data, self.__cast(other))])

    def __cast(self, other):
        # [FUNCTION DOC] HELPER: Return the raw data of other (unwrapping a Vector); raises
        # ValueError if its length differs from this vector's data.
        if isinstance(other, Vector):
            other = other.data
        if len(self.data) != len(other):
            raise ValueError("Cannot compare vectors of different length")
        return other

opmap = {
    "lt": (lambda a,b: a< b, operator.lt, operator.__lt__),
    "le": (lambda a,b: a<=b, operator.le, operator.__le__),
    "eq": (lambda a,b: a==b, operator.eq, operator.__eq__),
    "ne": (lambda a,b: a!=b, operator.ne, operator.__ne__),
    "gt": (lambda a,b: a> b, operator.gt, operator.__gt__),
    "ge": (lambda a,b: a>=b, operator.ge, operator.__ge__)
}

class VectorTest(CPythonTestCase):
    # [CLASS DOC] Tests for comparisons whose results are Vector objects (rich results).

    def checkfail(self, error, opname, *args):
        # [FUNCTION DOC] HELPER: Assert that each of the three callables for opname raises the
        # given error type when called with args.
        for op in opmap[opname]:
            self.assertRaises(error, op, *args)

    def checkequal(self, opname, a, b, expres):
        # [FUNCTION DOC] HELPER: Assert that for every callable of opname the result of comparing a
        # and b has the same length as expres and each item is the identical bool.
        for op in opmap[opname]:
            realres = op(a, b)
            # can't use assertEqual(realres, expres) here
            self.assertEqual(len(realres), len(expres))
            for i in range(len(realres)):
                # results are bool, so we can use "is" here
                self.assertTrue(realres[i] is expres[i])

    def test_mixed(self):
        # [FUNCTION DOC] Checks length mismatch raises ValueError for every operator; checks all
        # operators with list/Vector argument combinations give the expected bool vectors; and
        # that converting those results to bool raises TypeError.
        # check that comparisons involving Vector objects
        # which return rich results (i.e. Vectors with itemwise
        # comparison results) work
        a = Vector(range(2))
        b = Vector(range(3))
        # all comparisons should fail for different length
        for opname in opmap:
            self.checkfail(ValueError, opname, a, b)

        a = list(range(5))
        b = 5 * [2]
        # try mixed arguments (but not (a, b) as that won't return a bool vector)
        args = [(a, Vector(b)), (Vector(a), b), (Vector(a), Vector(b))]
        for (a, b) in args:
            self.checkequal("lt", a, b, [True,  True,  False, False, False])
            self.checkequal("le", a, b, [True,  True,  True,  False, False])
            self.checkequal("eq", a, b, [False, False, True,  False, False])
            self.checkequal("ne", a, b, [True,  True,  False, True,  True ])
            self.checkequal("gt", a, b, [False, False, False, True,  True ])
            self.checkequal("ge", a, b, [False, False, True,  True,  True ])

            for ops in opmap.values():
                for op in ops:
                    # calls __bool__, which should fail
                    self.assertRaises(TypeError, bool, op(a, b))

class NumberTest(CPythonTestCase):
    # [CLASS DOC] Tests for comparisons involving Number objects against plain ints.

    def test_basic(self):
        # [FUNCTION DOC] Checks every operator on Number and int combinations (not int with int)
        # gives the same outcome as comparing the plain ints.
        # Check that comparisons involving Number objects
        # give the same results give as comparing the
        # corresponding ints
        for a in range(3):
            for b in range(3):
                for typea in (int, Number):
                    for typeb in (int, Number):
                        if typea==typeb==int:
                            continue # the combination int, int is useless
                        ta = typea(a)
                        tb = typeb(b)
                        for ops in opmap.values():
                            for op in ops:
                                realoutcome = op(a, b)
                                testoutcome = op(ta, tb)
                                self.assertEqual(realoutcome, testoutcome)

    def checkvalue(self, opname, a, b, expres):
        # [FUNCTION DOC] HELPER: For every int/Number combination of a and b and every callable of
        # opname, assert the (unwrapped) result is the identical expected bool.
        for typea in (int, Number):
            for typeb in (int, Number):
                ta = typea(a)
                tb = typeb(b)
                for op in opmap[opname]:
                    realres = op(ta, tb)
                    realres = getattr(realres, "x", realres)
                    self.assertTrue(realres is expres)

    def test_values(self):
        # [FUNCTION DOC] Checks all six operators for the ordered cases equal, less and greater.
        # check all operators and all comparison results
        self.checkvalue("lt", 0, 0, False)
        self.checkvalue("le", 0, 0, True )
        self.checkvalue("eq", 0, 0, True )
        self.checkvalue("ne", 0, 0, False)
        self.checkvalue("gt", 0, 0, False)
        self.checkvalue("ge", 0, 0, True )

        self.checkvalue("lt", 0, 1, True )
        self.checkvalue("le", 0, 1, True )
        self.checkvalue("eq", 0, 1, False)
        self.checkvalue("ne", 0, 1, True )
        self.checkvalue("gt", 0, 1, False)
        self.checkvalue("ge", 0, 1, False)

        self.checkvalue("lt", 1, 0, False)
        self.checkvalue("le", 1, 0, False)
        self.checkvalue("eq", 1, 0, False)
        self.checkvalue("ne", 1, 0, True )
        self.checkvalue("gt", 1, 0, True )
        self.checkvalue("ge", 1, 0, True )

class MiscTest(CPythonTestCase):
    # [CLASS DOC] Miscellaneous rich comparison tests: odd return values, not operator,
    # recursion, error messages.

    def test_misbehavin(self):
        # [FUNCTION DOC] Checks that less, greater and equality methods returning 0 are returned
        # as is, and that the other comparison methods are not invoked.
        class Misb:
            # [CLASS DOC] HELPER: Only lt, gt and eq are usable;
            # the others fail the enclosing test.
            # [FUNCTION DOC] HELPER: Less-than; returns 0. (Single-line def, block sits above.)
            def __lt__(self_, other): return 0
            # [FUNCTION DOC] HELPER: Greater-than; returns 0. (Single-line def, block sits above.)
            def __gt__(self_, other): return 0
            # [FUNCTION DOC] HELPER: Equality; returns 0. (Single-line def, block sits above.)
            def __eq__(self_, other): return 0
            # [FUNCTION DOC] HELPER: Fails the enclosing test if called. (Single-line def.)
            def __le__(self_, other): self.fail("This shouldn't happen")
            # [FUNCTION DOC] HELPER: Fails the enclosing test if called. (Single-line def.)
            def __ge__(self_, other): self.fail("This shouldn't happen")
            # [FUNCTION DOC] HELPER: Fails the enclosing test if called. (Single-line def.)
            def __ne__(self_, other): self.fail("This shouldn't happen")
        a = Misb()
        b = Misb()
        self.assertEqual(a<b, 0)
        self.assertEqual(a==b, 0)
        self.assertEqual(a>b, 0)

    def test_not(self):
        # [FUNCTION DOC] Checks that an exception raised by a truth-value method propagates out of
        # both the not statement and operator.not_.
        # Check that exceptions in __bool__ are properly
        # propagated by the not operator
        import operator
        class Exc(Exception):
            # [CLASS DOC] HELPER: Exception type raised by Bad.
            pass
        class Bad:
            # [CLASS DOC] HELPER: Object whose truth test raises Exc.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc

        def do(bad):
            # [FUNCTION DOC] HELPER: Evaluate the not of the argument, discarding the result.
            not bad

        for func in (do, operator.not_):
            self.assertRaises(Exc, func, Bad())

    @support.no_tracing
    @support.infinite_recursion(25)
    def test_recursion(self):
        # [FUNCTION DOC] Checks comparisons of mutually recursive lists: all six operators raise
        # RecursionError initially; after changing lengths equality/inequality work but ordering
        # still raises; and with differing first elements ordering finally succeeds. Runs with
        # tracing disabled and a recursion limit of 25.
        # Check that comparison for recursive objects fails gracefully
        from collections import UserList
        a = UserList()
        b = UserList()
        a.append(b)
        b.append(a)
        self.assertRaises(RecursionError, operator.eq, a, b)
        self.assertRaises(RecursionError, operator.ne, a, b)
        self.assertRaises(RecursionError, operator.lt, a, b)
        self.assertRaises(RecursionError, operator.le, a, b)
        self.assertRaises(RecursionError, operator.gt, a, b)
        self.assertRaises(RecursionError, operator.ge, a, b)

        b.append(17)
        # Even recursive lists of different lengths are different,
        # but they cannot be ordered
        self.assertTrue(not (a == b))
        self.assertTrue(a != b)
        self.assertRaises(RecursionError, operator.lt, a, b)
        self.assertRaises(RecursionError, operator.le, a, b)
        self.assertRaises(RecursionError, operator.gt, a, b)
        self.assertRaises(RecursionError, operator.ge, a, b)
        a.append(17)
        self.assertRaises(RecursionError, operator.eq, a, b)
        self.assertRaises(RecursionError, operator.ne, a, b)
        a.insert(0, 11)
        b.insert(0, 12)
        self.assertTrue(not (a == b))
        self.assertTrue(a != b)
        self.assertTrue(a < b)

    def test_exception_message(self):
        # [FUNCTION DOC] Checks the TypeError text of unsupported ordering comparisons mentions
        # the operator symbol and both operand type names, using a list of lambdas and patterns.
        class Spam:
            # [CLASS DOC] HELPER: Empty class without comparison support.
            pass

        tests = [
            (lambda: 42 < None, r"'<' .* of 'int' and 'NoneType'"),
            (lambda: None < 42, r"'<' .* of 'NoneType' and 'int'"),
            (lambda: 42 > None, r"'>' .* of 'int' and 'NoneType'"),
            (lambda: "foo" < None, r"'<' .* of 'str' and 'NoneType'"),
            (lambda: "foo" >= 666, r"'>=' .* of 'str' and 'int'"),
            (lambda: 42 <= None, r"'<=' .* of 'int' and 'NoneType'"),
            (lambda: 42 >= None, r"'>=' .* of 'int' and 'NoneType'"),
            (lambda: 42 < [], r"'<' .* of 'int' and 'list'"),
            (lambda: () > [], r"'>' .* of 'tuple' and 'list'"),
            (lambda: None >= None, r"'>=' .* of 'NoneType' and 'NoneType'"),
            (lambda: Spam() < 42, r"'<' .* of 'Spam' and 'int'"),
            (lambda: 42 < Spam(), r"'<' .* of 'int' and 'Spam'"),
            (lambda: Spam() <= Spam(), r"'<=' .* of 'Spam' and 'Spam'"),
        ]
        for i, test in enumerate(tests):
            with self.subTest(test=i):
                with self.assertRaisesRegex(TypeError, test[1]):
                    test[0]()


class DictTest(CPythonTestCase):
    # [CLASS DOC] Tests for dict comparison.

    def test_dicts(self):
        # [FUNCTION DOC] Builds dicts with complex keys and values (random contents): equal dicts
        # compare equal, a modified one is unequal, and ordering operators raise TypeError.
        # Uses the random module so contents vary between runs.
        # Verify that __eq__ and __ne__ work for dicts even if the keys and
        # values don't support anything other than __eq__ and __ne__ (and
        # __hash__).  Complex numbers are a fine example of that.
        import random
        imag1a = {}
        for i in range(50):
            imag1a[random.randrange(100)*1j] = random.randrange(100)*1j
        items = list(imag1a.items())
        random.shuffle(items)
        imag1b = {}
        for k, v in items:
            imag1b[k] = v
        imag2 = imag1b.copy()
        imag2[k] = v + 1.0
        self.assertEqual(imag1a, imag1a)
        self.assertEqual(imag1a, imag1b)
        self.assertEqual(imag2, imag2)
        self.assertTrue(imag1a != imag2)
        for opname in ("lt", "le", "gt", "ge"):
            for op in opmap[opname]:
                self.assertRaises(TypeError, op, imag1a, imag2)

class ListTest(CPythonTestCase):
    # [CLASS DOC] Tests for list comparison.

    def test_coverage(self):
        # [FUNCTION DOC] Checks all six comparisons between one-element and two-element lists.
        # exercise all comparisons for lists
        x = [42]
        self.assertIs(x<x, False)
        self.assertIs(x<=x, True)
        self.assertIs(x==x, True)
        self.assertIs(x!=x, False)
        self.assertIs(x>x, False)
        self.assertIs(x>=x, True)
        y = [42, 42]
        self.assertIs(x<y, True)
        self.assertIs(x<=y, True)
        self.assertIs(x==y, False)
        self.assertIs(x!=y, True)
        self.assertIs(x>y, False)
        self.assertIs(x>=y, False)

    def test_badentry(self):
        # [FUNCTION DOC] Checks an exception raised by an item's equality method propagates out of
        # list equality for all three callables.
        # make sure that exceptions for item comparison are properly
        # propagated in list comparisons
        class Exc(Exception):
            # [CLASS DOC] HELPER: Exception type raised by Bad.
            pass
        class Bad:
            # [CLASS DOC] HELPER: Object whose equality always raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc

        x = [Bad()]
        y = [Bad()]

        for op in opmap["eq"]:
            self.assertRaises(Exc, op, x, y)

    def test_goodentry(self):
        # [FUNCTION DOC] Checks that list less-than falls back to comparing the last items and
        # returns exactly True when the item method returns True.
        # This test exercises the final call to PyObject_RichCompare()
        # in Objects/listobject.c::list_richcompare()
        class Good:
            # [CLASS DOC] HELPER: Object whose less-than always returns True.
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Always returns True.
                return True

        x = [Good()]
        y = [Good()]

        for op in opmap["lt"]:
            self.assertIs(op(x, y), True)


if __name__ == "__main__":
    run_tests()
