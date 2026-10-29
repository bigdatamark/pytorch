# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Common sequence-protocol tests (tuple, list, UserList) copied from CPython
# 3.13 Lib/test/seq_tests.py, with the base test case replaced by CPythonTestCase (kept in
# the private name __TestCase). Provides iterable helper classes for constructor tests and
# the CommonTest base class that concrete test modules subclass after setting type2test.
# Import-time side effect: inserts a RedirectImportFinder at the front of
# sys.meta_path so listed test.* imports resolve to same-named top-level modules.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/seq_tests.py

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
    # [CLASS DOC] Meta path finder that redirects selected "test.X" imports (see redirect_imports)
    # to the top-level module "X". Stateless; installed once at import time into sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] Finder hook. If fullname is in redirect_imports, imports the module
        # named without the "test." prefix, registers it in sys.modules under fullname and
        # returns that module's spec; returns None (defer to other finders) if the name is
        # not listed or the import fails with ImportError. Side effect: mutates sys.modules.
        # KNOWN ISSUE: uses importlib.util without importing it explicitly (only
        # importlib.abc is imported), relying on it being loaded elsewhere.
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

"""
Tests common to tuple, list and UserList.UserList
"""

import unittest
import sys
import pickle
from test import support
from test.support import ALWAYS_EQ, NEVER_EQ

# Various iterables
# This is used for checking the constructor (here and in test_deque.py)
def iterfunc(seqn):
    # [FUNCTION DOC] HELPER: Regular generator that yields each element of the given iterable.
    'Regular generator'
    for i in seqn:
        yield i

class Sequence:
    # [CLASS DOC] HELPER: Sequence-like helper that supports only indexing
    # (old-style iteration via __getitem__). Holds the wrapped data in seqn.
    'Sequence using __getitem__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the wrapped sequence.
        self.seqn = seqn
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Returns the element at index i;
        # IndexError from the wrapped sequence ends iteration.
        return self.seqn[i]

class IterFunc:
    # [CLASS DOC] HELPER: Iterator-protocol helper: object with __iter__
    # returning itself and __next__ stepping through seqn using the index i.
    'Sequence using iterator protocol'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next element, raising
        # StopIteration when the position reaches the length.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class IterGen:
    # [CLASS DOC] HELPER: Iterable helper whose __iter__ is a generator over seqn.
    'Sequence using iterator protocol defined with a generator'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding each element of seqn.
        for val in self.seqn:
            yield val

class IterNextOnly:
    # [CLASS DOC] HELPER: Helper that has __next__ but neither
    # __iter__ nor __getitem__, so it is not iterable.
    'Missing __getitem__ and __iter__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next element, raising StopIteration at the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class IterNoNext:
    # [CLASS DOC] HELPER: Helper whose __iter__ returns itself
    # but which has no __next__, so it is an invalid iterator.
    'Iterator missing __next__()'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self

class IterGenExc:
    # [CLASS DOC] HELPER: Helper iterator whose __next__ always
    # raises ZeroDivisionError, used to test exception propagation.
    'Test propagation of exceptions'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises ZeroDivisionError by dividing by zero.
        3 // 0

class IterFuncStop:
    # [CLASS DOC] HELPER: Helper iterator that stops immediately, yielding nothing.
    'Test immediate stop'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Ignores the sequence argument.
        pass
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises StopIteration.
        raise StopIteration

from itertools import chain
def itermulti(seqn):
    # [FUNCTION DOC] HELPER: Builds a multi-tier iterator: the sequence
    # wrapped by Sequence, then IterGen, then iterfunc, then a map with an
    # identity function, then itertools.chain; returns the chain iterator.
    'Test multiple tiers of iterators'
    return chain(map(lambda x:x, iterfunc(IterGen(Sequence(seqn)))))

class LyingTuple(tuple):
    # [CLASS DOC] HELPER: Tuple subclass whose iterator
    # yields only the value 1, regardless of contents.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding the single value 1.
        yield 1

class LyingList(list):
    # [CLASS DOC] HELPER: List subclass whose iterator
    # yields only the value 1, regardless of contents.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding the single value 1.
        yield 1

class CommonTest(__TestCase):
    # [CLASS DOC] Sequence conformance tests shared by tuple, list and UserList. Subclasses
    # set the class attribute type2test (None here) to the sequence type being tested.
    # The type to be tested
    type2test = None

    def test_constructors(self):
        # [FUNCTION DOC] Checks construction of type2test from nothing, lists, other instances,
        # tuples, arbitrary indexable sequences, strings and many iterable kinds (generators,
        # iterator-protocol objects, nested iterators). Verifies TypeError for non-iterables or
        # bad iterators, ZeroDivisionError propagation, that the iterator (not the overridden
        # contents) is used for tuple/list subclasses, and TypeError for an unknown keyword
        # argument.
        l0 = []
        l1 = [0]
        l2 = [0, 1]

        u = self.type2test()
        u0 = self.type2test(l0)
        u1 = self.type2test(l1)
        u2 = self.type2test(l2)

        uu = self.type2test(u)
        uu0 = self.type2test(u0)
        uu1 = self.type2test(u1)
        uu2 = self.type2test(u2)

        v = self.type2test(tuple(u))
        class OtherSeq:
            # [CLASS DOC] HELPER: Sequence helper exposing __len__ and
            # __getitem__ over initial data, used as constructor input.
            def __init__(self, initseq):
                # [FUNCTION DOC] HELPER: Stores the initial
                # data in a name-mangled private attribute.
                self.__data = initseq
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the length of the stored data.
                return len(self.__data)
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the element at index i of the stored data.
                return self.__data[i]
        s = OtherSeq(u0)
        v0 = self.type2test(s)
        self.assertEqual(len(v0), len(s))

        s = "this is also a sequence"
        vv = self.type2test(s)
        self.assertEqual(len(vv), len(s))

        # Create from various iteratables
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (Sequence, IterFunc, IterGen,
                      itermulti, iterfunc):
                self.assertEqual(self.type2test(g(s)), self.type2test(s))
            self.assertEqual(self.type2test(IterFuncStop(s)), self.type2test())
            self.assertEqual(self.type2test(c for c in "123"), self.type2test("123"))
            self.assertRaises(TypeError, self.type2test, IterNextOnly(s))
            self.assertRaises(TypeError, self.type2test, IterNoNext(s))
            self.assertRaises(ZeroDivisionError, self.type2test, IterGenExc(s))

        # Issue #23757
        self.assertEqual(self.type2test(LyingTuple((2,))), self.type2test((1,)))
        self.assertEqual(self.type2test(LyingList([2])), self.type2test([1]))

        with self.assertRaises(TypeError):
            self.type2test(unsupported_arg=[])

    def test_truth(self):
        # [FUNCTION DOC] Checks truthiness: empty sequence is false, non-empty is true.
        self.assertFalse(self.type2test())
        self.assertTrue(self.type2test([42]))

    def test_getitem(self):
        # [FUNCTION DOC] Checks integer indexing, including negative
        # indices, IndexError for out-of-range indices, ValueError for a
        # zero slice step, and TypeError when called with no argument.
        u = self.type2test([0, 1, 2, 3, 4])
        for i in range(len(u)):
            self.assertEqual(u[i], i)
            self.assertEqual(u[int(i)], i)
        for i in range(-len(u), -1):
            self.assertEqual(u[i], len(u)+i)
            self.assertEqual(u[int(i)], len(u)+i)
        self.assertRaises(IndexError, u.__getitem__, -len(u)-1)
        self.assertRaises(IndexError, u.__getitem__, len(u))
        self.assertRaises(ValueError, u.__getitem__, slice(0,10,0))

        u = self.type2test()
        self.assertRaises(IndexError, u.__getitem__, 0)
        self.assertRaises(IndexError, u.__getitem__, -1)

        self.assertRaises(TypeError, u.__getitem__)

        a = self.type2test([10, 11])
        self.assertEqual(a[0], 10)
        self.assertEqual(a[1], 11)
        self.assertEqual(a[-2], 10)
        self.assertEqual(a[-1], 11)
        self.assertRaises(IndexError, a.__getitem__, -3)
        self.assertRaises(IndexError, a.__getitem__, 3)

    def test_getslice(self):
        # [FUNCTION DOC] Checks slicing: basic, out-of-range, None bounds,
        # extended slices with positive and negative steps, and very large
        # integer bounds; each result must equal the expected sequence.
        l = [0, 1, 2, 3, 4]
        u = self.type2test(l)

        self.assertEqual(u[0:0], self.type2test())
        self.assertEqual(u[1:2], self.type2test([1]))
        self.assertEqual(u[-2:-1], self.type2test([3]))
        self.assertEqual(u[-1000:1000], u)
        self.assertEqual(u[1000:-1000], self.type2test([]))
        self.assertEqual(u[:], u)
        self.assertEqual(u[1:None], self.type2test([1, 2, 3, 4]))
        self.assertEqual(u[None:3], self.type2test([0, 1, 2]))

        # Extended slices
        self.assertEqual(u[::], u)
        self.assertEqual(u[::2], self.type2test([0, 2, 4]))
        self.assertEqual(u[1::2], self.type2test([1, 3]))
        self.assertEqual(u[::-1], self.type2test([4, 3, 2, 1, 0]))
        self.assertEqual(u[::-2], self.type2test([4, 2, 0]))
        self.assertEqual(u[3::-2], self.type2test([3, 1]))
        self.assertEqual(u[3:3:-2], self.type2test([]))
        self.assertEqual(u[3:2:-2], self.type2test([3]))
        self.assertEqual(u[3:1:-2], self.type2test([3]))
        self.assertEqual(u[3:0:-2], self.type2test([3, 1]))
        self.assertEqual(u[::-100], self.type2test([4]))
        self.assertEqual(u[100:-100:], self.type2test([]))
        self.assertEqual(u[-100:100:], u)
        self.assertEqual(u[100:-100:-1], u[::-1])
        self.assertEqual(u[-100:100:-1], self.type2test([]))
        self.assertEqual(u[-100:100:2], self.type2test([0, 2, 4]))

        # Test extreme cases with long ints
        a = self.type2test([0,1,2,3,4])
        self.assertEqual(a[ -pow(2,128): 3 ], self.type2test([0,1,2]))
        self.assertEqual(a[ 3: pow(2,145) ], self.type2test([3,4]))
        self.assertEqual(a[3::sys.maxsize], self.type2test([3]))

    def test_contains(self):
        # [FUNCTION DOC] Checks membership for present and absent
        # values and TypeError when __contains__ has no argument.
        u = self.type2test([0, 1, 2])
        for i in u:
            self.assertIn(i, u)
        for i in min(u)-1, max(u)+1:
            self.assertNotIn(i, u)

        self.assertRaises(TypeError, u.__contains__)

    def test_contains_fake(self):
        # [FUNCTION DOC] Checks that membership uses rich equality: an object that always compares
        # equal is found in any non-empty sequence, and one that never compares equal is not.
        # Sequences must use rich comparison against each item
        # (unless "is" is true, or an earlier item answered)
        # So ALWAYS_EQ must be found in all non-empty sequences.
        self.assertNotIn(ALWAYS_EQ, self.type2test([]))
        self.assertIn(ALWAYS_EQ, self.type2test([1]))
        self.assertIn(1, self.type2test([ALWAYS_EQ]))
        self.assertNotIn(NEVER_EQ, self.type2test([]))
        self.assertNotIn(ALWAYS_EQ, self.type2test([NEVER_EQ]))
        self.assertIn(NEVER_EQ, self.type2test([ALWAYS_EQ]))

    def test_contains_order(self):
        # [FUNCTION DOC] Checks that membership tests elements in order: the
        # match on an earlier item stops the search before a later raising
        # comparison, while a raising item placed first propagates its exception.
        # Sequences must test in-order.  If a rich comparison has side
        # effects, these will be visible to tests against later members.
        # In this test, the "side effect" is a short-circuiting raise.
        class DoNotTestEq(Exception):
            # [CLASS DOC] HELPER: Exception used to detect that a comparison was performed.
            pass
        class StopCompares:
            # [CLASS DOC] HELPER: Element type whose equality comparison raises DoNotTestEq.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises DoNotTestEq.
                raise DoNotTestEq

        checkfirst = self.type2test([1, StopCompares()])
        self.assertIn(1, checkfirst)
        checklast = self.type2test([StopCompares(), 1])
        self.assertRaises(DoNotTestEq, checklast.__contains__, 1)

    def test_len(self):
        # [FUNCTION DOC] Checks length of empty and non-empty sequences.
        self.assertEqual(len(self.type2test()), 0)
        self.assertEqual(len(self.type2test([])), 0)
        self.assertEqual(len(self.type2test([0])), 1)
        self.assertEqual(len(self.type2test([0, 1, 2])), 3)

    def test_minmax(self):
        # [FUNCTION DOC] Checks the builtin min and max over a three-element sequence.
        u = self.type2test([0, 1, 2])
        self.assertEqual(min(u), 0)
        self.assertEqual(max(u), 2)

    def test_addmul(self):
        # [FUNCTION DOC] Checks concatenation, repetition by zero, one, two and three (both
        # operand orders), and that repeating a subclass instance by one returns a new object.
        u1 = self.type2test([0])
        u2 = self.type2test([0, 1])
        self.assertEqual(u1, u1 + self.type2test())
        self.assertEqual(u1, self.type2test() + u1)
        self.assertEqual(u1 + self.type2test([1]), u2)
        self.assertEqual(self.type2test([-1]) + u1, self.type2test([-1, 0]))
        self.assertEqual(self.type2test(), u2*0)
        self.assertEqual(self.type2test(), 0*u2)
        self.assertEqual(self.type2test(), u2*0)
        self.assertEqual(self.type2test(), 0*u2)
        self.assertEqual(u2, u2*1)
        self.assertEqual(u2, 1*u2)
        self.assertEqual(u2, u2*1)
        self.assertEqual(u2, 1*u2)
        self.assertEqual(u2+u2, u2*2)
        self.assertEqual(u2+u2, 2*u2)
        self.assertEqual(u2+u2, u2*2)
        self.assertEqual(u2+u2, 2*u2)
        self.assertEqual(u2+u2+u2, u2*3)
        self.assertEqual(u2+u2+u2, 3*u2)

        class subclass(self.type2test):
            # [CLASS DOC] HELPER: Trivial subclass of type2test,
            # used to check repetition on subclasses.
            pass
        u3 = subclass([0, 1])
        self.assertEqual(u3, u3*1)
        self.assertIsNot(u3, u3*1)

    def test_iadd(self):
        # [FUNCTION DOC] Checks in-place addition with empty, sequence and string operands.
        u = self.type2test([0, 1])
        u += self.type2test()
        self.assertEqual(u, self.type2test([0, 1]))
        u += self.type2test([2, 3])
        self.assertEqual(u, self.type2test([0, 1, 2, 3]))
        u += self.type2test([4, 5])
        self.assertEqual(u, self.type2test([0, 1, 2, 3, 4, 5]))

        u = self.type2test("spam")
        u += self.type2test("eggs")
        self.assertEqual(u, self.type2test("spameggs"))

    def test_imul(self):
        # [FUNCTION DOC] Checks in-place multiplication by three and by zero.
        u = self.type2test([0, 1])
        u *= 3
        self.assertEqual(u, self.type2test([0, 1, 0, 1, 0, 1]))
        u *= 0
        self.assertEqual(u, self.type2test([]))

    def test_getitemoverwriteiter(self):
        # [FUNCTION DOC] Verifies that an overridden __getitem__ does not
        # affect iteration, which still yields the original first element.
        # Verify that __getitem__ overrides are not recognized by __iter__
        class T(self.type2test):
            # [CLASS DOC] HELPER: type2test subclass whose
            # indexing returns a string built from the key.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the key converted
                # to a string followed by three exclamation marks.
                return str(key) + '!!!'
        self.assertEqual(next(iter(T((1,2)))), 1)

    def test_repeat(self):
        # [FUNCTION DOC] Checks repetition by integers from minus three to four
        # against a tuple's repetition, that a negative factor gives an empty
        # sequence, and that multiplying a tuple by one returns the same object.
        for m in range(4):
            s = tuple(range(m))
            for n in range(-3, 5):
                self.assertEqual(self.type2test(s*n), self.type2test(s)*n)
            self.assertEqual(self.type2test(s)*(-4), self.type2test([]))
            self.assertEqual(id(s), id(s*1))

    def test_bigrepeat(self):
        # [FUNCTION DOC] On 32-bit platforms only, checks that overly large repetition raises
        # MemoryError (for both multiplication and in-place); does nothing on 64-bit platforms.
        if sys.maxsize <= 2147483647:
            x = self.type2test([0])
            x *= 2**16
            self.assertRaises(MemoryError, x.__mul__, 2**16)
            if hasattr(x, '__imul__'):
                self.assertRaises(MemoryError, x.__imul__, 2**16)

    def test_subscript(self):
        # [FUNCTION DOC] Checks calling __getitem__ explicitly with indices, slices, out-of-range
        # indices (IndexError), zero slice step (ValueError) and a string (TypeError).
        a = self.type2test([10, 11])
        self.assertEqual(a.__getitem__(0), 10)
        self.assertEqual(a.__getitem__(1), 11)
        self.assertEqual(a.__getitem__(-2), 10)
        self.assertEqual(a.__getitem__(-1), 11)
        self.assertRaises(IndexError, a.__getitem__, -3)
        self.assertRaises(IndexError, a.__getitem__, 3)
        self.assertEqual(a.__getitem__(slice(0,1)), self.type2test([10]))
        self.assertEqual(a.__getitem__(slice(1,2)), self.type2test([11]))
        self.assertEqual(a.__getitem__(slice(0,2)), self.type2test([10, 11]))
        self.assertEqual(a.__getitem__(slice(0,3)), self.type2test([10, 11]))
        self.assertEqual(a.__getitem__(slice(3,5)), self.type2test([]))
        self.assertRaises(ValueError, a.__getitem__, slice(0, 10, 0))
        self.assertRaises(TypeError, a.__getitem__, 'x')

    def test_count(self):
        # [FUNCTION DOC] Checks count with plain values, always-equal
        # and never-equal objects, TypeError without argument, and
        # propagation of an exception raised by an element comparison.
        a = self.type2test([0, 1, 2])*3
        self.assertEqual(a.count(0), 3)
        self.assertEqual(a.count(1), 3)
        self.assertEqual(a.count(3), 0)

        self.assertEqual(a.count(ALWAYS_EQ), 9)
        self.assertEqual(self.type2test([ALWAYS_EQ, ALWAYS_EQ]).count(1), 2)
        self.assertEqual(self.type2test([ALWAYS_EQ, ALWAYS_EQ]).count(NEVER_EQ), 2)
        self.assertEqual(self.type2test([NEVER_EQ, NEVER_EQ]).count(ALWAYS_EQ), 0)

        self.assertRaises(TypeError, a.count)

        class BadExc(Exception):
            # [CLASS DOC] HELPER: Exception raised by the bad comparison helper.
            pass

        class BadCmp:
            # [CLASS DOC] HELPER: Value type whose equality raises
            # BadExc when compared with 2 and otherwise returns false.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Raises BadExc if other equals 2, otherwise returns False.
                if other == 2:
                    raise BadExc()
                return False

        self.assertRaises(BadExc, a.count, BadCmp())

    def test_index(self):
        # [FUNCTION DOC] Checks index lookup with and without start and stop
        # bounds (including negative and huge values), ValueError when the value
        # is absent or the range is empty, always/never-equal objects, TypeError
        # without argument, and propagation of a comparison exception.
        u = self.type2test([0, 1])
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = self.type2test([-2, -1, 0, 0, 1, 2])
        self.assertEqual(u.count(0), 2)
        self.assertEqual(u.index(0), 2)
        self.assertEqual(u.index(0, 2), 2)
        self.assertEqual(u.index(-2, -10), 0)
        self.assertEqual(u.index(0, 3), 3)
        self.assertEqual(u.index(0, 3, 4), 3)
        self.assertRaises(ValueError, u.index, 2, 0, -10)

        self.assertEqual(u.index(ALWAYS_EQ), 0)
        self.assertEqual(self.type2test([ALWAYS_EQ, ALWAYS_EQ]).index(1), 0)
        self.assertEqual(self.type2test([ALWAYS_EQ, ALWAYS_EQ]).index(NEVER_EQ), 0)
        self.assertRaises(ValueError, self.type2test([NEVER_EQ, NEVER_EQ]).index, ALWAYS_EQ)

        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            # [CLASS DOC] HELPER: Exception raised by the bad comparison helper.
            pass

        class BadCmp:
            # [CLASS DOC] HELPER: Value type whose equality raises
            # BadExc when compared with 2 and otherwise returns false.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Raises BadExc if other equals 2, otherwise returns False.
                if other == 2:
                    raise BadExc()
                return False

        a = self.type2test([0, 1, 2, 3])
        self.assertRaises(BadExc, a.index, BadCmp())

        a = self.type2test([-2, -1, 0, 0, 1, 2])
        self.assertEqual(a.index(0), 2)
        self.assertEqual(a.index(0, 2), 2)
        self.assertEqual(a.index(0, -4), 2)
        self.assertEqual(a.index(-2, -10), 0)
        self.assertEqual(a.index(0, 3), 3)
        self.assertEqual(a.index(0, -3), 3)
        self.assertEqual(a.index(0, 3, 4), 3)
        self.assertEqual(a.index(0, -3, -2), 3)
        self.assertEqual(a.index(0, -4*sys.maxsize, 4*sys.maxsize), 2)
        self.assertRaises(ValueError, a.index, 0, 4*sys.maxsize,-4*sys.maxsize)
        self.assertRaises(ValueError, a.index, 2, 0, -10)

    def test_pickle(self):
        # [FUNCTION DOC] Checks that pickling and unpickling the sequence
        # with every protocol yields an equal but distinct object.
        lst = self.type2test([4, 5, 6, 7])
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            lst2 = pickle.loads(pickle.dumps(lst, proto))
            self.assertEqual(lst2, lst)
            self.assertNotEqual(id(lst2), id(lst))

    @support.suppress_immortalization()
    def test_free_after_iterating(self):
        # [FUNCTION DOC] Delegates to the support helper check_free_after_iterating
        # for iteration and reverse iteration of type2test, verifying that the
        # sequence is freed after the iterator is exhausted.
        support.check_free_after_iterating(self, iter, self.type2test)
        support.check_free_after_iterating(self, reversed, self.type2test)
