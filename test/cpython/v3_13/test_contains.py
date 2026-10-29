# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_contains.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_contains tests, which check the
# containment operator (membership test and its negation) on builtin and user-defined
# containers. Defines small helper container classes and one CPythonTestCase suite. Running the
# file as a script calls run_tests. Import has no side effects beyond imports.

from collections import deque
import unittest
from test.support import NEVER_EQ


class base_set:
    # [CLASS DOC] HELPER: Base fixture class that stores a single element in the attribute el.
    # It defines no containment, iteration or indexing support of its own.
    def __init__(self, el):
        # [FUNCTION DOC] HELPER: Stores the given element in the attribute el.
        self.el = el

class myset(base_set):
    # [CLASS DOC] HELPER: Fixture container that supports membership through an explicit
    # containment method.
    def __contains__(self, el):
        # [FUNCTION DOC] HELPER: Returns whether the given element equals the stored element.
        return self.el == el

class seq(base_set):
    # [CLASS DOC] HELPER: Fixture container that is only indexable, so membership falls back
    # to iteration.
    def __getitem__(self, n):
        # [FUNCTION DOC] HELPER: Returns the item at position n of a one-element list holding
        # el. Indexing past the end raises an error, which ends the fallback iteration used
        # for membership.
        return [self.el][n]

class TestContains(CPythonTestCase):
    # [CLASS DOC] Test suite for the membership operator, run through the Dynamo CPython test
    # harness. Covers custom containers, builtin sequences, non-reflexive elements and
    # containment blocking.
    def test_common_tests(self):
        # [FUNCTION DOC] Checks membership on the helper classes and strings: a class with a
        # containment method, an indexable class and plain strings behave correctly; a class
        # with neither raises a TypeError; a non-string element tested against a string raises
        # a TypeError.
        a = base_set(1)
        b = myset(1)
        c = seq(1)
        self.assertIn(1, b)
        self.assertNotIn(0, b)
        self.assertIn(1, c)
        self.assertNotIn(0, c)
        self.assertRaises(TypeError, lambda: 1 in a)
        self.assertRaises(TypeError, lambda: 1 not in a)

        # test char in string
        self.assertIn('c', 'abc')
        self.assertNotIn('d', 'abc')

        self.assertIn('', '')
        self.assertIn('', 'abc')

        self.assertRaises(TypeError, lambda: None in 'abc')

    def test_builtin_sequence_types(self):
        # [FUNCTION DOC] Checks membership on range and tuple (every member found, an absent
        # value and the container itself not found), and that a list mutated during comparison
        # does not crash the membership test.
        # a collection of tests on builtin sequence types
        a = range(10)
        for i in a:
            self.assertIn(i, a)
        self.assertNotIn(16, a)
        self.assertNotIn(a, a)

        a = tuple(a)
        for i in a:
            self.assertIn(i, a)
        self.assertNotIn(16, a)
        self.assertNotIn(a, a)

        class Deviant1:
            # [CLASS DOC] HELPER: Element whose equality method mutates the shared class-level
            # list during comparison.
            """Behaves strangely when compared

            This class is designed to make sure that the contains code
            works when the list is modified during the check.
            """
            aList = list(range(15))
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality method that, when compared with 12, removes
                # 12, 13 and 14 from the shared list, and always returns a false value.
                if other == 12:
                    self.aList.remove(12)
                    self.aList.remove(13)
                    self.aList.remove(14)
                return 0

        self.assertNotIn(Deviant1(), Deviant1.aList)

    def test_nonreflexive(self):
        # [FUNCTION DOC] Checks that for several container types holding non-reflexive elements
        # (NaN, an object never equal to anything), each element is still reported as contained
        # and the container equals an equal container and itself.
        # containment and equality tests involving elements that are
        # not necessarily equal to themselves

        values = float('nan'), 1, None, 'abc', NEVER_EQ
        constructors = list, tuple, dict.fromkeys, set, frozenset, deque
        for constructor in constructors:
            container = constructor(values)
            for elem in container:
                self.assertIn(elem, container)
            self.assertTrue(container == constructor(values))
            self.assertTrue(container == container)

    def test_block_fallback(self):
        # [FUNCTION DOC] Checks that setting the containment attribute to None blocks the
        # fallback to iteration: membership on such an object raises a TypeError, while
        # iterating it works.
        # blocking fallback with __contains__ = None
        class ByContains(object):
            # [CLASS DOC] HELPER: Container whose containment method always returns false.
            def __contains__(self, other):
                # [FUNCTION DOC] HELPER: Always returns false regardless of the element.
                return False
        c = ByContains()
        class BlockContains(ByContains):
            # [CLASS DOC] HELPER: Iterable subclass that disables inherited containment by
            # setting it to None.
            """Is not a container

            This class is a perfectly good iterable (as tested by
            list(bc)), as well as inheriting from a perfectly good
            container, but __contains__ = None prevents the usual
            fallback to iteration in the container protocol. That
            is, normally, 0 in bc would fall back to the equivalent
            of any(x==0 for x in bc), but here it's blocked from
            doing so.
            """
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Generator that yields nothing.
                while False:
                    yield None
            __contains__ = None
        bc = BlockContains()
        self.assertFalse(0 in c)
        self.assertFalse(0 in list(bc))
        self.assertRaises(TypeError, lambda: 0 in bc)

if __name__ == '__main__':
    run_tests()
