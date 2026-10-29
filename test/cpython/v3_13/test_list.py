# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_list.py

# [MODULE DOC] Dynamo-patched copy of CPython 3.13 test_list. Installs, at import time,
# a meta path finder on sys.meta_path that redirects imports of selected test.* modules
# to their top-level standalone equivalents, then defines ListTest extending
# list_tests.CommonTest with list-specific behavior checks (construction, subclass
# keywords, overflow, pickling of iterators, mutation during comparison). Running the
# file calls run_tests.

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

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
    # [CLASS DOC] HELPER: Import hook (MetaPathFinder) that maps names in
    # redirect_imports from the test package onto top-level modules so dynamo-run tests
    # find them. Stateless; registered at module import.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a requested module name in redirect_imports,
        # imports the name with the test. prefix removed, stores that module in
        # sys.modules under the original name and returns its spec; returns None for
        # other names or when the import fails, letting normal finders proceed.
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

import sys
import textwrap
import list_tests
from test.support import cpython_only
from test.support.script_helper import assert_python_ok
import pickle
import unittest

class ListTest(list_tests.CommonTest):
    # [CLASS DOC] Tests for the built-in list, inheriting the generic sequence checks
    # from list_tests.CommonTest with type2test set to list. Adds list-specific cases;
    # no instance state.
    type2test = list

    def test_basic(self):
        # [FUNCTION DOC] Checks list construction from empty, list, tuple, string and
        # generator inputs gives equal but distinct lists, and that extending a list
        # from a generator over itself leaves it empty. The 32-bit MemoryError block
        # only runs on 32-bit builds.
        self.assertEqual(list([]), [])
        l0_3 = [0, 1, 2, 3]
        l0_3_bis = list(l0_3)
        self.assertEqual(l0_3, l0_3_bis)
        self.assertTrue(l0_3 is not l0_3_bis)
        self.assertEqual(list(()), [])
        self.assertEqual(list((0, 1, 2, 3)), [0, 1, 2, 3])
        self.assertEqual(list(''), [])
        self.assertEqual(list('spam'), ['s', 'p', 'a', 'm'])
        self.assertEqual(list(x for x in range(10) if x % 2),
                         [1, 3, 5, 7, 9])

        if sys.maxsize == 0x7fffffff:
            # This test can currently only work on 32-bit machines.
            # XXX If/when PySequence_Length() returns a ssize_t, it should be
            # XXX re-enabled.
            # Verify clearing of bug #556025.
            # This assumes that the max data size (sys.maxint) == max
            # address size this also assumes that the address size is at
            # least 4 bytes with 8 byte addresses, the bug is not well
            # tested
            #
            # Note: This test is expected to SEGV under Cygwin 1.3.12 or
            # earlier due to a newlib bug.  See the following mailing list
            # thread for the details:
            self.assertRaises(MemoryError, list, range(sys.maxsize // 2))

        # This code used to segfault in Py2.4a3
        x = []
        x.extend(-y for y in x)
        self.assertEqual(x, [])

    def test_keyword_args(self):
        # [FUNCTION DOC] Checks that passing a keyword argument to list raises TypeError
        # mentioning keyword argument.
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            list(sequence=[])

    def test_keywords_in_subclass(self):
        # [FUNCTION DOC] Checks list subclasses: plain subclass rejects keyword
        # arguments with TypeError, a subclass with custom __init__ accepts an extra
        # keyword and stores it, and one with custom __new__ does the same; types and
        # contents are asserted.
        class subclass(list):
            pass
        u = subclass([1, 2])
        self.assertIs(type(u), subclass)
        self.assertEqual(list(u), [1, 2])
        with self.assertRaises(TypeError):
            subclass(sequence=())

        class subclass_with_init(list):
            def __init__(self, seq, newarg=None):
                super().__init__(seq)
                self.newarg = newarg
        u = subclass_with_init([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

        class subclass_with_new(list):
            def __new__(cls, seq, newarg=None):
                self = super().__new__(cls, seq)
                self.newarg = newarg
                return self
        u = subclass_with_new([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(list(u), [1, 2])
        self.assertEqual(u.newarg, 3)

    def test_truth(self):
        # [FUNCTION DOC] Runs the inherited truth test and then checks an empty list is
        # falsy and a non-empty list is truthy.
        super().test_truth()
        self.assertTrue(not [])
        self.assertTrue([42])

    def test_identity(self):
        # [FUNCTION DOC] Checks two separately created empty list displays are distinct
        # objects.
        self.assertTrue([] is not [])

    def test_len(self):
        # [FUNCTION DOC] Runs the inherited length test and checks lengths of lists with
        # 0, 1 and 3 items.
        super().test_len()
        self.assertEqual(len([]), 0)
        self.assertEqual(len([0]), 1)
        self.assertEqual(len([0, 1, 2]), 3)

    def test_overflow(self):
        # [FUNCTION DOC] Checks that multiplying or in-place multiplying a 4-element
        # list by a count that makes the size exceed twice the machine word range raises
        # MemoryError or OverflowError.
        lst = [4, 5, 6, 7]
        n = int((sys.maxsize*2+2) // len(lst))
        def mul(a, b): return a * b
            # [FUNCTION DOC] HELPER: Returns the product a times b, used so the overflow
            # can be provoked inside assertRaises.
        def imul(a, b): a *= b
            # [FUNCTION DOC] HELPER: Performs in-place multiplication on its local name
            # (no return value); used to provoke overflow errors.
        self.assertRaises((MemoryError, OverflowError), mul, lst, n)
        self.assertRaises((MemoryError, OverflowError), imul, lst, n)

    def test_empty_slice(self):
        # [FUNCTION DOC] Checks assigning an empty list to its own full slice leaves it
        # empty.
        x = []
        x[:] = x
        self.assertEqual(x, [])

    def test_list_resize_overflow(self):
        # [FUNCTION DOC] Regression test for the overflow check in list resizing:
        # shrinks a list to one element, then checks multiplying and in-place
        # multiplying by sys.maxsize raises MemoryError or OverflowError.
        # gh-97616: test new_allocated * sizeof(PyObject*) overflow
        # check in list_resize()
        lst = [0] * 65
        del lst[1:]
        self.assertEqual(len(lst), 1)

        size = sys.maxsize
        with self.assertRaises((MemoryError, OverflowError)):
            lst * size
        with self.assertRaises((MemoryError, OverflowError)):
            lst *= size

    def test_repr_mutate(self):
        # [FUNCTION DOC] Checks repr of a list whose element repr pops from the same
        # list: five elements with each repr removing one yields a repr showing only
        # three items without crashing.
        class Obj:
            @staticmethod
            def __repr__():
                # [FUNCTION DOC] HELPER: Static repr that removes the last item from the
                # enclosing list (ignoring an empty-list IndexError) and returns the
                # text obj.
                try:
                    mylist.pop()
                except IndexError:
                    pass
                return 'obj'

        mylist = [Obj() for _ in range(5)]
        self.assertEqual(repr(mylist), '[obj, obj, obj]')

    def test_repr_large(self):
        # [FUNCTION DOC] Checks repr of a list of 1,000,000 zeros matches the text built
        # by joining, using a local check helper also run on size 10.
        # Check the repr of large list objects
        def check(n):
            # [FUNCTION DOC] HELPER: Builds a list of n zeros and asserts its repr
            # equals brackets around the comma-joined zeros.
            l = [0] * n
            s = repr(l)
            self.assertEqual(s,
                '[' + ', '.join(['0'] * n) + ']')
        check(10)       # check our checking code
        check(1000000)

    def test_iterator_pickle(self):
        # [FUNCTION DOC] For every pickle protocol, pickles a list iterator together
        # with its list at four stages (fresh, advanced, emptied, exhausted), reloads,
        # replaces the reloaded list content with new data and checks the iterator type
        # and that it yields the remaining items of the new data; the exhausted one
        # yields nothing and next raises StopIteration.
        orig = self.type2test([4, 5, 6, 7])
        data = [10, 11, 12, 13, 14, 15]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = iter(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data)

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[1:])

            # empty iterator
            for i in range(1, len(orig)):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig):])

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(list(it), [])

    def test_reversed_pickle(self):
        # [FUNCTION DOC] Same as the iterator pickle test but for reversed iterators:
        # checks type preserved and the remaining items are taken backwards from the
        # replaced data at fresh, advanced, emptied and exhausted stages, for each
        # pickle protocol.
        orig = self.type2test([4, 5, 6, 7])
        data = [10, 11, 12, 13, 14, 15]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = reversed(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig)-1::-1])

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), data[len(orig)-2::-1])

            # empty iterator
            for i in range(1, len(orig)):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(type(it), type(itorig))
            self.assertEqual(list(it), [])

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, a = pickle.loads(d)
            a[:] = data
            self.assertEqual(list(it), [])

    def test_step_overflow(self):
        # [FUNCTION DOC] Checks slice assignment and reading with a step of sys.maxsize
        # only touches the one valid index and does not overflow.
        a = [0, 1, 2, 3, 4]
        a[1::sys.maxsize] = [0]
        self.assertEqual(a[3::sys.maxsize], [3])

    def test_no_comdat_folding(self):
        # [FUNCTION DOC] Regression for issue 8847 (linker function folding): adding a
        # tuple and a list subclass must raise TypeError.
        # Issue 8847: In the PGO build, the MSVC linker's COMDAT folding
        # optimization causes failures in code that relies on distinct
        # function addresses.
        class L(list): pass
        with self.assertRaises(TypeError):
            (3,) + L([1,2])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_equal_operator_modifying_operand(self):
        # [FUNCTION DOC] Skipped under dynamo. Regression test for a segfault: element
        # equality methods clear the other list while comparing two lists; asserts the
        # first comparison is true (one list emptied) and the second false, without
        # crashing.
        # test fix for seg fault reported in bpo-38588 part 2.
        class X:
            # [CLASS DOC] HELPER: Element type whose equality clears list2 and returns
            # NotImplemented.
            def __eq__(self,other) :
                # [FUNCTION DOC] HELPER: Clears list2 as a side effect and returns
                # NotImplemented.
                list2.clear()
                return NotImplemented

        class Y:
            # [CLASS DOC] HELPER: Element type whose equality clears list1 and returns
            # NotImplemented.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Clears list1 as a side effect and returns
                # NotImplemented.
                list1.clear()
                return NotImplemented

        class Z:
            # [CLASS DOC] HELPER: Element type whose equality clears list3 and returns
            # NotImplemented.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Clears list3 as a side effect and returns
                # NotImplemented.
                list3.clear()
                return NotImplemented

        list1 = [X()]
        list2 = [Y()]
        self.assertTrue(list1 == list2)

        list3 = [Z()]
        list4 = [1]
        self.assertFalse(list3 == list4)

    @unittest.skipIf(
        # [FUNCTION DOC] Skipped under dynamo. Regression for gh-120298: an element's
        # less-than clears the other list; comparing nested lists must raise TypeError
        # rather than crash.
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_lt_operator_modifying_operand(self):
        # See gh-120298
        class evil:
            # [CLASS DOC] HELPER: Element type whose less-than operator clears its
            # operand and returns NotImplemented.
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Clears the other operand and returns
                # NotImplemented.
                other.clear()
                return NotImplemented

        a = [[evil()]]
        with self.assertRaises(TypeError):
            a[0] < a

    def test_list_index_modifing_operand(self):
        # [FUNCTION DOC] Regression for gh-120384: extended-slice assignment from an
        # iterable that clears the target list mid-iteration must raise ValueError (size
        # mismatch).
        # See gh-120384
        class evil:
            # [CLASS DOC] HELPER: Iterable wrapping a list that yields all its items and
            # then clears the list.
            def __init__(self, lst):
                # [FUNCTION DOC] HELPER: Stores the wrapped list.
                self.lst = lst
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Generator yielding every item of the wrapped
                # list, then clearing that list.
                yield from self.lst
                self.lst.clear()

        lst = list(range(5))
        operand = evil(lst)
        with self.assertRaises(ValueError):
            lst[::-1] = operand

    @cpython_only
        # [FUNCTION DOC] CPython-only. Checks sys.getsizeof of a list built from a list
        # or a range of 10 items equals that of a list literal of ten items (exact
        # preallocation).
    def test_preallocation(self):
        iterable = [0] * 10
        iter_size = sys.getsizeof(iterable)

        self.assertEqual(iter_size, sys.getsizeof(list([0] * 10)))
        self.assertEqual(iter_size, sys.getsizeof(list(range(10))))

    @unittest.skipIf(
        # [FUNCTION DOC] Skipped under dynamo. Regression for bpo-38610 and bpo-39453:
        # element equality clears the list during index, count, remove and the
        # membership test; checks index/remove raise ValueError, count and membership do
        # not crash.
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_count_index_remove_crashes(self):
        # bpo-38610: The count(), index(), and remove() methods were not
        # holding strong references to list elements while calling
        # PyObject_RichCompareBool().
        class X:
            # [CLASS DOC] HELPER: Element type whose equality clears the module-level-
            # named list lst and returns NotImplemented.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Clears lst as a side effect and returns
                # NotImplemented.
                lst.clear()
                return NotImplemented

        lst = [X()]
        with self.assertRaises(ValueError):
            lst.index(lst)

        class L(list):
            # [CLASS DOC] HELPER: List subclass whose equality stringifies the other
            # operand and returns NotImplemented.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Calls str on other, then returns
                # NotImplemented.
                str(other)
                return NotImplemented

        lst = L([X()])
        lst.count(lst)

        lst = L([X()])
        with self.assertRaises(ValueError):
            lst.remove(lst)

        # bpo-39453: list.__contains__ was not holding strong references
        # to list elements while calling PyObject_RichCompareBool().
        lst = [X(), X()]
        3 in lst
        lst = [X(), X()]
        X() in lst

    def test_tier2_invalidates_iterator(self):
        # [FUNCTION DOC] Regression for GH-121012: repeatedly iterates a list to
        # exhaustion, appends, and checks the exhausted iterator still yields nothing.
        # GH-121012
        for _ in range(100):
            a = [1, 2, 3]
            it = iter(a)
            for _ in it:
                pass
            a.append(4)
            self.assertEqual(list(it), [])

    @unittest.skip("Fails on python <=3.13.2 and passes on >=3.13.3")
    def test_deopt_from_append_list(self):
        # [FUNCTION DOC] Unconditionally skipped test (fails on Python up to 3.13.2).
        # Runs a subprocess with a script that calls a function appending a tuple to its
        # list argument repeatedly then passes the list type to trigger a TypeError,
        # asserting exit code 0 (regression gh-132011).
        # gh-132011: it used to crash, because
        # of `CALL_LIST_APPEND` specialization failure.
        code = textwrap.dedent("""
            l = []
            def lappend(l, x, y):
                l.append((x, y))
            for x in range(3):
                lappend(l, None, None)
            try:
                lappend(list, None, None)
            except TypeError:
                pass
            else:
                raise AssertionError
        """)

        rc, _, _ = assert_python_ok("-c", code)
        self.assertEqual(rc, 0)

if __name__ == "__main__":
    run_tests()
