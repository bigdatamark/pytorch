# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of CPython 3.13 test_index. Checks that the __index__ protocol
# is honored by operator.index, slice.indices, sequence indexing, slicing, repetition, range and
# overflow handling, using CPythonTestCase so the tests run under the Dynamo test harness.
# Importing defines test classes only; run_tests() is called when run as a script.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_index.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
from test import support
import operator
maxsize = support.MAX_Py_ssize_t

class newstyle:
    # [CLASS DOC] HELPER: Minimal new-style class whose __index__ returns the attribute ind set by
    # the test.
    def __index__(self):
        # [FUNCTION DOC] HELPER: Return the ind attribute (a test-assigned value, may be a
        # non-integer to force errors).
        return self.ind

class TrapInt(int):
    # [CLASS DOC] HELPER: int subclass that overrides __index__ to return its own int value.
    def __index__(self):
        # [FUNCTION DOC] HELPER: Return the plain int value of this instance.
        return int(self)

class BaseTestCase(CPythonTestCase):
    # [CLASS DOC] Tests for the __index__ protocol on scalars: operator.index, slice.indices
    # and wrappers. Holds two newstyle fixtures, self.o and self.n, whose ind attribute each
    # test sets.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp and create fresh newstyle fixtures self.o
        # and self.n.
        super().setUp()
        self.o = newstyle()
        self.n = newstyle()

    def test_basic(self):
        # [FUNCTION DOC] operator.index returns the value of __index__ for negative and
        # positive indices.
        self.o.ind = -2
        self.n.ind = 2
        self.assertEqual(operator.index(self.o), -2)
        self.assertEqual(operator.index(self.n), 2)

    def test_slice(self):
        # [FUNCTION DOC] slice built from __index__ objects gives the same indices() result as the
        # plain-int slice.
        self.o.ind = 1
        self.n.ind = 2
        slc = slice(self.o, self.o, self.o)
        check_slc = slice(1, 1, 1)
        self.assertEqual(slc.indices(self.o), check_slc.indices(1))
        slc = slice(self.n, self.n, self.n)
        check_slc = slice(2, 2, 2)
        self.assertEqual(slc.indices(self.n), check_slc.indices(2))

    def test_wrappers(self):
        # [FUNCTION DOC] Direct __index__ calls on ints, bools and newstyle objects return the
        # expected ints.
        self.o.ind = 4
        self.n.ind = 5
        self.assertEqual(6 .__index__(), 6)
        self.assertEqual(-7 .__index__(), -7)
        self.assertEqual(self.o.__index__(), 4)
        self.assertEqual(self.n.__index__(), 5)
        self.assertEqual(True.__index__(), 1)
        self.assertEqual(False.__index__(), 0)

    def test_subclasses(self):
        # [FUNCTION DOC] int subclasses overriding __index__ still work as range slice bounds and
        # slice stops.
        r = list(range(10))
        self.assertEqual(r[TrapInt(5):TrapInt(10)], r[5:10])
        self.assertEqual(slice(TrapInt()).indices(0), (0,0,1))

    def test_error(self):
        # [FUNCTION DOC] Non-integer __index__ results make operator.index and slice.indices
        # raise TypeError.
        self.o.ind = 'dumb'
        self.n.ind = 'bad'
        self.assertRaises(TypeError, operator.index, self.o)
        self.assertRaises(TypeError, operator.index, self.n)
        self.assertRaises(TypeError, slice(self.o).indices, 0)
        self.assertRaises(TypeError, slice(self.n).indices, 0)

    def test_int_subclass_with_index(self):
        # [FUNCTION DOC] For an int subclass, a direct __index__ call returns the override
        # (value plus one, exact int) while operator.index returns the underlying int value (7),
        # not the override.
        # KNOWN ISSUE: the assertion that operator.index returns exact type int is commented out.
        # __index__ should be used when computing indices, even for int
        # subclasses.  See issue #17576.
        class MyInt(int):
            # [CLASS DOC] HELPER: int subclass whose __index__ returns the decimal value plus one.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the int value plus one.
                return int(str(self)) + 1

        my_int = MyInt(7)
        direct_index = my_int.__index__()
        operator_index = operator.index(my_int)
        self.assertEqual(direct_index, 8)
        self.assertEqual(operator_index, 7)
        # Both results should be of exact type int.
        self.assertIs(type(direct_index), int)
        #self.assertIs(type(operator_index), int)

    def test_index_returns_int_subclass(self):
        # [FUNCTION DOC] An __index__ returning a bool subclass of int emits a DeprecationWarning
        # and is converted; for an int subclass whose __index__ returns True, operator.index uses
        # the underlying value (0).
        class BadInt:
            # [CLASS DOC] HELPER: Plain class whose __index__ returns True instead of an
            # exact int.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return True (an int subclass, not exact int).
                return True

        class BadInt2(int):
            # [CLASS DOC] HELPER: int subclass (value 0) whose __index__ returns True.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return True.
                return True

        bad_int = BadInt()
        with self.assertWarns(DeprecationWarning):
            n = operator.index(bad_int)
        self.assertEqual(n, 1)

        bad_int = BadInt2()
        n = operator.index(bad_int)
        self.assertEqual(n, 0)


class SeqTestCase:
    # [CLASS DOC] Mixin of common indexing tests shared by sequence types; subclasses set the seq
    # attribute. Not run directly. Fixtures o, n, o2, n2 are newstyle objects set up in setUp.
    # This test case isn't run directly. It just defines common tests
    # to the different sequence types below
    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp and create four newstyle fixtures (o, n, o2, n2).
        super().setUp()
        self.o = newstyle()
        self.n = newstyle()
        self.o2 = newstyle()
        self.n2 = newstyle()

    def test_index(self):
        # [FUNCTION DOC] Indexing seq with __index__ objects matches indexing with the
        # equivalent ints.
        self.o.ind = -2
        self.n.ind = 2
        self.assertEqual(self.seq[self.n], self.seq[2])
        self.assertEqual(self.seq[self.o], self.seq[-2])

    def test_slice(self):
        # [FUNCTION DOC] Slicing seq with __index__ bounds matches slicing with plain int bounds.
        self.o.ind = 1
        self.o2.ind = 3
        self.n.ind = 2
        self.n2.ind = 4
        self.assertEqual(self.seq[self.o:self.o2], self.seq[1:3])
        self.assertEqual(self.seq[self.n:self.n2], self.seq[2:4])

    def test_slice_bug7532(self):
        # [FUNCTION DOC] Out-of-range positive and negative __index__ slice bounds clamp like ints
        # (issue 7532).
        seqlen = len(self.seq)
        self.o.ind = int(seqlen * 1.5)
        self.n.ind = seqlen + 2
        self.assertEqual(self.seq[self.o:], self.seq[0:0])
        self.assertEqual(self.seq[:self.o], self.seq)
        self.assertEqual(self.seq[self.n:], self.seq[0:0])
        self.assertEqual(self.seq[:self.n], self.seq)
        self.o2.ind = -seqlen - 2
        self.n2.ind = -int(seqlen * 1.5)
        self.assertEqual(self.seq[self.o2:], self.seq)
        self.assertEqual(self.seq[:self.o2], self.seq[0:0])
        self.assertEqual(self.seq[self.n2:], self.seq)
        self.assertEqual(self.seq[:self.n2], self.seq[0:0])

    def test_repeat(self):
        # [FUNCTION DOC] Multiplying seq by __index__ objects (either operand order) equals
        # repeating by plain ints.
        self.o.ind = 3
        self.n.ind = 2
        self.assertEqual(self.seq * self.o, self.seq * 3)
        self.assertEqual(self.seq * self.n, self.seq * 2)
        self.assertEqual(self.o * self.seq, self.seq * 3)
        self.assertEqual(self.n * self.seq, self.seq * 2)

    def test_wrappers(self):
        # [FUNCTION DOC] Explicit __getitem__, __mul__ and __rmul__ calls accept
        # __index__ objects.
        self.o.ind = 4
        self.n.ind = 5
        self.assertEqual(self.seq.__getitem__(self.o), self.seq[4])
        self.assertEqual(self.seq.__mul__(self.o), self.seq * 4)
        self.assertEqual(self.seq.__rmul__(self.o), self.seq * 4)
        self.assertEqual(self.seq.__getitem__(self.n), self.seq[5])
        self.assertEqual(self.seq.__mul__(self.n), self.seq * 5)
        self.assertEqual(self.seq.__rmul__(self.n), self.seq * 5)

    def test_subclasses(self):
        # [FUNCTION DOC] An int subclass with __index__ returning 0 indexes seq at position 0.
        self.assertEqual(self.seq[TrapInt()], self.seq[0])

    def test_error(self):
        # [FUNCTION DOC] Non-integer __index__ results raise TypeError for both indexing and
        # slicing of seq.
        self.o.ind = 'dumb'
        self.n.ind = 'bad'
        indexobj = lambda x, obj: obj.seq[x]
        self.assertRaises(TypeError, indexobj, self.o, self)
        self.assertRaises(TypeError, indexobj, self.n, self)
        sliceobj = lambda x, obj: obj.seq[x:]
        self.assertRaises(TypeError, sliceobj, self.o, self)
        self.assertRaises(TypeError, sliceobj, self.n, self)


class ListTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to a list; adds item assignment, deletion and in-place
    # repeat tests.
    seq = [0,10,20,30,40,50]

    def test_setdelitem(self):
        # [FUNCTION DOC] List item set and delete accept __index__ objects, via syntax and
        # explicit dunder calls.
        self.o.ind = -2
        self.n.ind = 2
        lst = list('ab!cdefghi!j')
        del lst[self.o]
        del lst[self.n]
        lst[self.o] = 'X'
        lst[self.n] = 'Y'
        self.assertEqual(lst, list('abYdefghXj'))

        lst = [5, 6, 7, 8, 9, 10, 11]
        lst.__setitem__(self.n, "here")
        self.assertEqual(lst, [5, 6, "here", 8, 9, 10, 11])
        lst.__delitem__(self.n)
        self.assertEqual(lst, [5, 6, 8, 9, 10, 11])

    def test_inplace_repeat(self):
        # [FUNCTION DOC] In-place list repeat with __index__ counts works and __imul__ returns the
        # same list.
        self.o.ind = 2
        self.n.ind = 3
        lst = [6, 4]
        lst *= self.o
        self.assertEqual(lst, [6, 4, 6, 4])
        lst *= self.n
        self.assertEqual(lst, [6, 4, 6, 4] * 3)

        lst = [5, 6, 7, 8, 9, 11]
        l2 = lst.__imul__(self.n)
        self.assertIs(l2, lst)
        self.assertEqual(lst, [5, 6, 7, 8, 9, 11] * 3)


class NewSeq:
    # [CLASS DOC] HELPER: Minimal sequence wrapping a list; supports repr, equality, len, repeat
    # and getitem.

    def __init__(self, iterable):
        # [FUNCTION DOC] HELPER: Store a list copy of the iterable in _list.
        self._list = list(iterable)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Return the repr of the wrapped list.
        return repr(self._list)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Compare the wrapped list to other with list equality.
        return self._list == other

    def __len__(self):
        # [FUNCTION DOC] HELPER: Return the length of the wrapped list.
        return len(self._list)

    def __mul__(self, n):
        # [FUNCTION DOC] HELPER: Return a new instance of this class holding the list repeated
        # n times.
        return self.__class__(self._list*n)
    __rmul__ = __mul__

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Index or slice the wrapped list; returns the item or list slice.
        return self._list[index]


class TupleTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to a tuple.
    seq = (0,10,20,30,40,50)

class ByteArrayTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to a bytearray.
    seq = bytearray(b"this is a test")

class BytesTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to a bytes object.
    seq = b"this is a test"

class StringTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to a str.
    seq = "this is a test"

class NewSeqTestCase(SeqTestCase, CPythonTestCase):
    # [CLASS DOC] SeqTestCase applied to the custom NewSeq sequence.
    seq = NewSeq((0,10,20,30,40,50))



class RangeTestCase(CPythonTestCase):
    # [CLASS DOC] Tests that range objects accept __index__ objects as indices.

    def test_range(self):
        # [FUNCTION DOC] Indexing a range, directly or via __getitem__, with an __index__ object
        # returns the element.
        n = newstyle()
        n.ind = 5
        self.assertEqual(range(1, 20)[n], 6)
        self.assertEqual(range(1, 20).__getitem__(n), 6)


class OverflowTestCase(CPythonTestCase):
    # [CLASS DOC] Tests __index__ handling of huge ints (beyond machine size); pos is 2**100, neg
    # is minus that.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp and define the huge positive and negative fixtures
        # pos and neg.
        super().setUp()
        self.pos = 2**100
        self.neg = -self.pos

    def test_large_longs(self):
        # [FUNCTION DOC] Huge ints return themselves from __index__.
        self.assertEqual(self.pos.__index__(), self.pos)
        self.assertEqual(self.neg.__index__(), self.neg)

    def test_getitem(self):
        # [FUNCTION DOC] A custom __getitem__ receives huge ints and slices unchanged; len is
        # never called and slice.indices clamps huge bounds to the supplied maximum size.
        class GetItem:
            # [CLASS DOC] HELPER: Class whose __getitem__ echoes the key and whose __len__ must
            # never be called.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Always fails the test if called (length must not
                # be used).
                assert False, "__len__ should not be invoked"
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the key unchanged.
                return key
        x = GetItem()
        self.assertEqual(x[self.pos], self.pos)
        self.assertEqual(x[self.neg], self.neg)
        self.assertEqual(x[self.neg:self.pos].indices(maxsize),
                         (0, maxsize, 1))
        self.assertEqual(x[self.neg:self.pos:1].indices(maxsize),
                         (0, maxsize, 1))

    def test_sequence_repeat(self):
        # [FUNCTION DOC] Repeating a str by a huge positive or negative int raises OverflowError.
        self.assertRaises(OverflowError, lambda: "a" * self.pos)
        self.assertRaises(OverflowError, lambda: "a" * self.neg)


if __name__ == "__main__":
    run_tests()
