# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_slice ported to run under Dynamo's CPythonTestCase. Tests the
# builtin slice type: constructor arity, repr, hash, comparison, members, the indices method
# checked against a pure-Python reference implementation, setitem with a slice key, pickling, copy
# and deepcopy semantics, and reference-cycle collection. Imports test.support.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_slice.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# tests for slice objects; in particular the indices method.

import itertools
import operator
import sys
import unittest
import weakref
import copy

from pickle import loads, dumps
from test import support


def evaluate_slice_index(arg):
    # [FUNCTION DOC] HELPER: converts a slice component to an integer via the index protocol.
    # Returns the integer when the argument defines an index hook; otherwise raises TypeError with
    # a message about slice indices.
    """
    Helper function to convert a slice argument to an integer, and raise
    TypeError with a suitable message on failure.

    """
    if hasattr(arg, '__index__'):
        return operator.index(arg)
    else:
        raise TypeError(
            "slice indices must be integers or "
            "None or have an __index__ method")

def slice_indices(slice, length):
    # [FUNCTION DOC] HELPER: reference implementation of slice.indices. Converts length and step
    # to integers, raises ValueError for negative length or zero step, then clamps start and stop
    # to the bounds that depend on the step sign (negative indices count from the end) and returns
    # the (start, stop, step) tuple.
    """
    Reference implementation for the slice.indices method.

    """
    # Compute step and length as integers.
    length = operator.index(length)
    step = 1 if slice.step is None else evaluate_slice_index(slice.step)

    # Raise ValueError for negative length or zero step.
    if length < 0:
        raise ValueError("length should not be negative")
    if step == 0:
        raise ValueError("slice step cannot be zero")

    # Find lower and upper bounds for start and stop.
    lower = -1 if step < 0 else 0
    upper = length - 1 if step < 0 else length

    # Compute start.
    if slice.start is None:
        start = upper if step < 0 else lower
    else:
        start = evaluate_slice_index(slice.start)
        start = max(start + length, lower) if start < 0 else min(start, upper)

    # Compute stop.
    if slice.stop is None:
        stop = lower if step < 0 else upper
    else:
        stop = evaluate_slice_index(slice.stop)
        stop = max(stop + length, lower) if stop < 0 else min(stop, upper)

    return start, stop, step


# Class providing an __index__ method.  Used for testing slice.indices.

class MyIndexable(object):
    # [CLASS DOC] HELPER: wrapper object whose only feature is an index hook returning the stored
    # integer value; used to check slice.indices accepts custom index-like objects.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: stores the integer value to return from the index hook.
        self.value = value

    def __index__(self):
        # [FUNCTION DOC] HELPER: returns the stored value so the object acts as an integer index.
        return self.value


class SliceTest(CPythonTestCase):
    # [CLASS DOC] Test case for the builtin slice object (construction, comparison, indices,
    # pickling, copying). Stateless; uses module level reference helpers.

    def test_constructor(self):
        # [FUNCTION DOC] Checks slice() with no arguments and slice with four arguments
        # raise TypeError.
        self.assertRaises(TypeError, slice)
        self.assertRaises(TypeError, slice, 1, 2, 3, 4)

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr of a three-argument slice is the expected text.
        self.assertEqual(repr(slice(1, 2, 3)), "slice(1, 2, 3)")

    def test_hash(self):
        # [FUNCTION DOC] Checks hash matches the dunder hash call for 1, 2 and 3 argument slices,
        # different slices compare unequal, and hashing slices holding unhashable parts (list,
        # dict) raises TypeError.
        self.assertEqual(hash(slice(5)), slice(5).__hash__())
        self.assertEqual(hash(slice(1, 2)), slice(1, 2).__hash__())
        self.assertEqual(hash(slice(1, 2, 3)), slice(1, 2, 3).__hash__())
        self.assertNotEqual(slice(5), slice(6))

        with self.assertRaises(TypeError):
            hash(slice(1, 2, []))

        with self.assertRaises(TypeError):
            hash(slice(4, {}))

    def test_cmp(self):
        # [FUNCTION DOC] Checks slice equality: equal parts compare equal, differing step, None
        # and other types compare unequal, and exceptions from element comparison propagate when
        # two distinct slices are compared (identity comparison still succeeds) for start, stop
        # and step positions.
        s1 = slice(1, 2, 3)
        s2 = slice(1, 2, 3)
        s3 = slice(1, 2, 4)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, None)
        self.assertNotEqual(s1, (1, 2, 3))
        self.assertNotEqual(s1, "")

        class Exc(Exception):
            # [CLASS DOC] HELPER: exception type raised by the bad comparison object.
            pass

        class BadCmp(object):
            # [CLASS DOC] HELPER: object whose equality always raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc

        s1 = slice(BadCmp())
        s2 = slice(BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1 == s2)

        s1 = slice(1, BadCmp())
        s2 = slice(1, BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1 == s2)

        s1 = slice(1, 2, BadCmp())
        s2 = slice(1, 2, BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1 == s2)

    def test_members(self):
        # [FUNCTION DOC] Checks start, stop and step members for slices built with one, two and
        # three arguments (single argument sets stop only) and that arbitrary objects are stored
        # as is.
        s = slice(1)
        self.assertEqual(s.start, None)
        self.assertEqual(s.stop, 1)
        self.assertEqual(s.step, None)

        s = slice(1, 2)
        self.assertEqual(s.start, 1)
        self.assertEqual(s.stop, 2)
        self.assertEqual(s.step, None)

        s = slice(1, 2, 3)
        self.assertEqual(s.start, 1)
        self.assertEqual(s.stop, 2)
        self.assertEqual(s.step, 3)

        class AnyClass:
            # [CLASS DOC] HELPER: empty class whose instance is used as an arbitrary slice bound.
            pass

        obj = AnyClass()
        s = slice(obj)
        self.assertTrue(s.stop is obj)

    def check_indices(self, slice, length):
        # [FUNCTION DOC] HELPER: compares slice.indices with the reference slice_indices for a
        # given length, treating ValueError in either as a matching outcome; when valid also
        # checks that a range built from the indices equals slicing a range of that length.
        try:
            actual = slice.indices(length)
        except ValueError:
            actual = "valueerror"
        try:
            expected = slice_indices(slice, length)
        except ValueError:
            expected = "valueerror"
        self.assertEqual(actual, expected)

        if length >= 0 and slice.step != 0:
            actual = range(*slice.indices(length))
            expected = range(length)[slice]
            self.assertEqual(actual, expected)

    def test_indices(self):
        # [FUNCTION DOC] Checks slice.indices on many explicit cases (None, negative, out of range
        # bounds and steps), equality of equivalent slices, a huge step, the helper on a negative
        # length, ValueError for negative length or zero step, TypeError for float arguments, and
        # acceptance of custom objects with an index hook. The large value product loop is
        # commented out in this port.
        self.assertEqual(slice(None           ).indices(10), (0, 10,  1))
        self.assertEqual(slice(None,  None,  2).indices(10), (0, 10,  2))
        self.assertEqual(slice(1,     None,  2).indices(10), (1, 10,  2))
        self.assertEqual(slice(None,  None, -1).indices(10), (9, -1, -1))
        self.assertEqual(slice(None,  None, -2).indices(10), (9, -1, -2))
        self.assertEqual(slice(3,     None, -2).indices(10), (3, -1, -2))
        # issue 3004 tests
        self.assertEqual(slice(None, -9).indices(10), (0, 1, 1))
        self.assertEqual(slice(None, -10).indices(10), (0, 0, 1))
        self.assertEqual(slice(None, -11).indices(10), (0, 0, 1))
        self.assertEqual(slice(None, -10, -1).indices(10), (9, 0, -1))
        self.assertEqual(slice(None, -11, -1).indices(10), (9, -1, -1))
        self.assertEqual(slice(None, -12, -1).indices(10), (9, -1, -1))
        self.assertEqual(slice(None, 9).indices(10), (0, 9, 1))
        self.assertEqual(slice(None, 10).indices(10), (0, 10, 1))
        self.assertEqual(slice(None, 11).indices(10), (0, 10, 1))
        self.assertEqual(slice(None, 8, -1).indices(10), (9, 8, -1))
        self.assertEqual(slice(None, 9, -1).indices(10), (9, 9, -1))
        self.assertEqual(slice(None, 10, -1).indices(10), (9, 9, -1))

        self.assertEqual(
            slice(-100,  100     ).indices(10),
            slice(None).indices(10)
        )
        self.assertEqual(
            slice(100,  -100,  -1).indices(10),
            slice(None, None, -1).indices(10)
        )
        self.assertEqual(slice(-100, 100, 2).indices(10), (0, 10,  2))

        self.assertEqual(list(range(10))[::sys.maxsize - 1], [0])

        # Check a variety of start, stop, step and length values, including
        # values exceeding sys.maxsize (see issue #14794).
        # vals = [None, -2**100, -2**30, -53, -7, -1, 0, 1, 7, 53, 2**30, 2**100]
        # lengths = [0, 1, 7, 53, 2**30, 2**100]
        # for slice_args in itertools.product(vals, repeat=3):
        #     s = slice(*slice_args)
        #     for length in lengths:
        #         self.check_indices(s, length)
        self.check_indices(slice(0, 10, 1), -3)

        # Negative length should raise ValueError
        with self.assertRaises(ValueError):
            slice(None).indices(-1)

        # Zero step should raise ValueError
        with self.assertRaises(ValueError):
            slice(0, 10, 0).indices(5)

        # Using a start, stop or step or length that can't be interpreted as an
        # integer should give a TypeError ...
        with self.assertRaises(TypeError):
            slice(0.0, 10, 1).indices(5)
        with self.assertRaises(TypeError):
            slice(0, 10.0, 1).indices(5)
        with self.assertRaises(TypeError):
            slice(0, 10, 1.0).indices(5)
        with self.assertRaises(TypeError):
            slice(0, 10, 1).indices(5.0)

        # ... but it should be fine to use a custom class that provides index.
        self.assertEqual(slice(0, 10, 1).indices(5), (0, 5, 1))
        self.assertEqual(slice(MyIndexable(0), 10, 1).indices(5), (0, 5, 1))
        self.assertEqual(slice(0, MyIndexable(10), 1).indices(5), (0, 5, 1))
        self.assertEqual(slice(0, 10, MyIndexable(1)).indices(5), (0, 5, 1))
        self.assertEqual(slice(0, 10, 1).indices(MyIndexable(5)), (0, 5, 1))

    def test_setslice_without_getslice(self):
        # [FUNCTION DOC] Checks assigning to a slice key of an object defining only setitem calls
        # it with a slice object and the value (recorded in a list).
        tmp = []
        class X(object):
            # [CLASS DOC] HELPER: class recording every setitem call into a shared list.
            def __setitem__(self, i, k):
                # [FUNCTION DOC] HELPER: appends the index and value pair to the shared list.
                tmp.append((i, k))

        x = X()
        x[1:2] = 42
        self.assertEqual(tmp, [(slice(1, 2), 42)])

    def test_pickle(self):
        # [FUNCTION DOC] Checks pickling round trip for all protocols preserves equality and the
        # indices result while producing a distinct object.
        import pickle

        s = slice(10, 20, 3)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            t = loads(dumps(s, protocol))
            self.assertEqual(s, t)
            self.assertEqual(s.indices(15), t.indices(15))
            self.assertNotEqual(id(s), id(t))

    def test_copy(self):
        # [FUNCTION DOC] Checks copy.copy returns the same slice object (also with mutable parts,
        # whose members stay identical).
        s = slice(1, 10)
        c = copy.copy(s)
        self.assertIs(s, c)

        s = slice(1, 10, 2)
        c = copy.copy(s)
        self.assertIs(s, c)

        # Corner case for mutable indices:
        s = slice([1, 2], [3, 4], [5, 6])
        c = copy.copy(s)
        self.assertIs(s, c)
        self.assertIs(s.start, c.start)
        self.assertIs(s.stop, c.stop)
        self.assertIs(s.step, c.step)

    def test_deepcopy(self):
        # [FUNCTION DOC] Checks copy.deepcopy yields an equal slice, and for mutable parts a new
        # slice object with distinct copies of start, stop and step.
        s = slice(1, 10)
        c = copy.deepcopy(s)
        self.assertEqual(s, c)

        s = slice(1, 10, 2)
        c = copy.deepcopy(s)
        self.assertEqual(s, c)

        # Corner case for mutable indices:
        s = slice([1, 2], [3, 4], [5, 6])
        c = copy.deepcopy(s)
        self.assertIsNot(s, c)
        self.assertEqual(s, c)
        self.assertIsNot(s.start, c.start)
        self.assertIsNot(s.stop, c.stop)
        self.assertIsNot(s.step, c.step)

    def test_cycle(self):
        # [FUNCTION DOC] Checks a slice stored on an object that it references is collected:
        # after dropping the strong reference and forcing garbage collection, the weak
        # reference is dead.
        class myobj(): pass
            # [CLASS DOC] HELPER: empty class for the reference cycle test.
        o = myobj()
        o.s = slice(o)
        w = weakref.ref(o)
        o = None
        support.gc_collect()
        self.assertIsNone(w())

if __name__ == "__main__":
    run_tests()
