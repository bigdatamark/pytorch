# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_iterlen suite. Verifies iterator
# length transparency: the length hint of an iterator equals the number of remaining items as it
# is consumed, for immutable sources, containers that forbid mutation during iteration, and lists
# that may change. Also checks that exceptions from length hooks are not swallowed. Module
# constant n is 10, the item count used by every fixture.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_iterlen.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

""" Test Iterator Length Transparency

Some functions or methods which accept general iterable arguments have
optional, more efficient code paths if they know how many items to expect.
For instance, map(func, iterable), will pre-allocate the exact amount of
space required whenever the iterable can report its length.

The desired invariant is:  len(it)==len(list(it)).

A complication is that an iterable and iterator can be the same object. To
maintain the invariant, an iterator needs to dynamically update its length.
For instance, an iterable such as range(10) always reports its length as ten,
but it=iter(range(10)) starts at ten, and then goes to nine after next(it).
Having this capability means that map() can ignore the distinction between
map(func, iterable) and map(func, iter(iterable)).

When the iterable is immutable, the implementation can straight-forwardly
report the original length minus the cumulative number of calls to next().
This is the case for tuples, range objects, and itertools.repeat().

Some containers become temporarily immutable during iteration.  This includes
dicts, sets, and collections.deque.  Their implementation is equally simple
though they need to permanently set their length to zero whenever there is
an attempt to iterate after a length mutation.

The situation slightly more involved whenever an object allows length mutation
during iteration.  Lists and sequence iterators are dynamically updatable.
So, if a list is extended during iteration, the iterator will continue through
the new items.  If it shrinks to a point before the most recent iteration,
then no further items are available and the length is reported at zero.

Reversed objects can also be wrapped around mutable objects; however, any
appends after the current position are ignored.  Any other approach leads
to confusion and possibly returning the same item more than once.

The iterators not listed above, such as enumerate and the other itertools,
are not length transparent because they have no way to distinguish between
iterables that report static length and iterators whose length changes with
each call (i.e. the difference between enumerate('abc') and
enumerate(iter('abc')).

"""

import unittest
from itertools import repeat
from collections import deque
from operator import length_hint

n = 10


class __TestInvariantWithoutMutations:
    # [CLASS DOC] HELPER: Mixin with the invariant test; subclasses set self.it to an iterator
    # over n items.

    def test_invariant(self):
        # [FUNCTION DOC] Consume self.it one item at a time, checking the length hint counts down
        # from n to 0, that StopIteration is raised at the end and that the hint stays 0.
        it = self.it
        for i in reversed(range(1, n+1)):
            self.assertEqual(length_hint(it), i)
            next(it)
        self.assertEqual(length_hint(it), 0)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(length_hint(it), 0)

class __TestTemporarilyImmutable(__TestInvariantWithoutMutations):
    # [CLASS DOC] HELPER: Mixin for containers that forbid length change while iterating;
    # subclasses set self.mutate to a callable that shrinks the container.

    def test_immutable_during_iteration(self):
        # [FUNCTION DOC] After one next call the length hint is n-1; mutating the container makes
        # the next call raise RuntimeError and the length hint drop to 0.
        # objects such as deques, sets, and dictionaries enforce
        # length immutability  during iteration

        it = self.it
        self.assertEqual(length_hint(it), n)
        next(it)
        self.assertEqual(length_hint(it), n-1)
        self.mutate()
        self.assertRaises(RuntimeError, next, it)
        self.assertEqual(length_hint(it), 0)

## ------- Concrete Type Tests -------

class TestRepeat(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for a repeat iterator of n items.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build a repeat iterator producing None n
        # times and store it in self.it.
        super().setUp()
        self.it = repeat(None, n)

class TestXrange(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for a range iterator.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over a range of n items
        # and store it in self.it.
        super().setUp()
        self.it = iter(range(n))

class TestXrangeCustomReversed(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for a reversed range iterator.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build a reversed iterator over a range of n
        # items and store it in self.it.
        super().setUp()
        self.it = reversed(range(n))

class TestTuple(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for a tuple iterator.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over a tuple of n items
        # and store it in self.it.
        super().setUp()
        self.it = iter(tuple(range(n)))

## ------- Types that should not be mutated during iteration -------

class TestDeque(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a deque iterator; mutate pops from the deque.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over a deque of n items
        # and store it in self.it; self.mutate is set to the deque pop method.
        super().setUp()
        d = deque(range(n))
        self.it = iter(d)
        self.mutate = d.pop

class TestDequeReversed(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a reversed deque iterator; mutate pops from the deque.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build a reversed iterator over a deque of n
        # items and store it in self.it; self.mutate is set to the deque pop method.
        super().setUp()
        d = deque(range(n))
        self.it = reversed(d)
        self.mutate = d.pop

class TestDictKeys(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a dict key iterator; mutate removes an item with popitem.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over the keys of a dict
        # with n items and store it in self.it; self.mutate is set to the dict popitem method.
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d)
        self.mutate = d.popitem

class TestDictItems(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a dict items iterator; mutate removes an item
    # with popitem.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over the items view of a
        # dict with n entries and store it in self.it; self.mutate is set to popitem.
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d.items())
        self.mutate = d.popitem

class TestDictValues(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a dict values iterator; mutate removes an item
    # with popitem.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over the values view of a
        # dict with n entries and store it in self.it; self.mutate is set to popitem.
        super().setUp()
        d = dict.fromkeys(range(n))
        self.it = iter(d.values())
        self.mutate = d.popitem

class TestSet(__TestTemporarilyImmutable, CPythonTestCase):
    # [CLASS DOC] Immutability test for a set iterator; mutate pops an element from the set.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over a set of n items and
        # store it in self.it; self.mutate is set to the set pop method.
        super().setUp()
        d = set(range(n))
        self.it = iter(d)
        self.mutate = d.pop

## ------- Types that can mutate during iteration -------

class TestList(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for iterators over lists, plus a test where the list
    # changes mid-iteration.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build an iterator over a range of n items
        # and store it in self.it(the fixture does not use a list).
        super().setUp()
        self.it = iter(range(n))

    def test_mutation(self):
        # [FUNCTION DOC] A list iterator after two steps has n-2 left; appending raises the hint,
        # truncating to the first item makes the hint 0 and the iterator yields nothing, even if
        # the list is later extended.
        d = list(range(n))
        it = iter(d)
        next(it)
        next(it)
        self.assertEqual(length_hint(it), n - 2)
        d.append(n)
        self.assertEqual(length_hint(it), n - 1)  # grow with append
        d[1:] = []
        self.assertEqual(length_hint(it), 0)
        self.assertEqual(list(it), [])
        d.extend(range(20))
        self.assertEqual(length_hint(it), 0)


class TestListReversed(__TestInvariantWithoutMutations, CPythonTestCase):
    # [CLASS DOC] Invariant test for reversed iterators, plus a mutation test where appends
    # are ignored.

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setUp, then build a reversed iterator over a range of n
        # items and store it in self.it(the fixture does not use a list).
        super().setUp()
        self.it = reversed(range(n))

    def test_mutation(self):
        # [FUNCTION DOC] A reversed list iterator ignores appends after its start, drops to a
        # hint of 0 when the list is truncated before the current position, and stays 0 after
        # later extension.
        d = list(range(n))
        it = reversed(d)
        next(it)
        next(it)
        self.assertEqual(length_hint(it), n - 2)
        d.append(n)
        self.assertEqual(length_hint(it), n - 2)  # ignore append
        d[1:] = []
        self.assertEqual(length_hint(it), 0)
        self.assertEqual(list(it), [])  # confirm invariant
        d.extend(range(20))
        self.assertEqual(length_hint(it), 0)

## -- Check to make sure exceptions are not suppressed by __length_hint__()


class BadLen(object):
    # [CLASS DOC] HELPER: Iterable over ten items whose __len__ raises RuntimeError.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return an iterator over ten integers.
        return iter(range(10))

    def __len__(self):
        # [FUNCTION DOC] HELPER: Raise RuntimeError.
        raise RuntimeError('hello')


class BadLengthHint(object):
    # [CLASS DOC] HELPER: Iterable over ten items whose __length_hint__ raises RuntimeError.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return an iterator over ten integers.
        return iter(range(10))

    def __length_hint__(self):
        # [FUNCTION DOC] HELPER: Raise RuntimeError.
        raise RuntimeError('hello')


class NoneLengthHint(object):
    # [CLASS DOC] HELPER: Iterable over ten items whose __length_hint__ returns NotImplemented.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return an iterator over ten integers.
        return iter(range(10))

    def __length_hint__(self):
        # [FUNCTION DOC] HELPER: Return NotImplemented, an invalid hint value.
        return NotImplemented


class TestLengthHintExceptions(CPythonTestCase):
    # [CLASS DOC] Tests that errors in __len__ or __length_hint__ are not hidden by list, extend
    # or bytearray.

    def test_issue1242657(self):
        # [FUNCTION DOC] list(), list.extend and bytearray.extend propagate the RuntimeError
        # raised by __len__ or __length_hint__ of the argument (issue 1242657).
        self.assertRaises(RuntimeError, list, BadLen())
        self.assertRaises(RuntimeError, list, BadLengthHint())
        self.assertRaises(RuntimeError, [].extend, BadLen())
        self.assertRaises(RuntimeError, [].extend, BadLengthHint())
        b = bytearray(range(10))
        self.assertRaises(RuntimeError, b.extend, BadLen())
        self.assertRaises(RuntimeError, b.extend, BadLengthHint())

    def test_invalid_hint(self):
        # [FUNCTION DOC] A __length_hint__ returning NotImplemented is ignored; list() still
        # yields all ten items.
        # Make sure an invalid result doesn't muck-up the works
        self.assertEqual(list(NoneLengthHint()), list(range(10)))


if __name__ == "__main__":
    run_tests()
