# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_iter suite. Tests the iterator
# protocol: iter() with one or two arguments, sequence-style and __iter__-style classes, for loops
# and comprehensions, pickling and __reduce__ of iterators, exceptions, the sink state after
# exhaustion, and iteration inside builtins (list, tuple, filter, map, zip, min/max, join, in,
# countOf, indexOf, writelines, unpacking). Import-time side effects: installs a
# RedirectImportFinder at the front of sys.meta_path, and defines helper iterable classes and the
# TRIPLETS table of expected nested loop results.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_iter.py

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
    # [CLASS DOC] HELPER: Meta path finder that redirects selected CPython test.* imports to
    # top-level modules. Installed in sys.meta_path at import time; stateless.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a module name listed in redirect_imports, import the same
        # name without the test. prefix, store the result in sys.modules under the requested name
        # and return that module's spec. Returns None (let other finders proceed) for other names
        # or when the import fails with ImportError.
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

# Test iterators.

import sys
import unittest
from test.support import cpython_only
from test.support.os_helper import TESTFN, unlink
from test.support import check_free_after_iterating, ALWAYS_EQ, NEVER_EQ
from test.support import BrokenIter
import pickle
import collections.abc
import functools
import contextlib
import builtins
import traceback

# Test result of triple loop (too big to inline)
TRIPLETS = [(0, 0, 0), (0, 0, 1), (0, 0, 2),
            (0, 1, 0), (0, 1, 1), (0, 1, 2),
            (0, 2, 0), (0, 2, 1), (0, 2, 2),

            (1, 0, 0), (1, 0, 1), (1, 0, 2),
            (1, 1, 0), (1, 1, 1), (1, 1, 2),
            (1, 2, 0), (1, 2, 1), (1, 2, 2),

            (2, 0, 0), (2, 0, 1), (2, 0, 2),
            (2, 1, 0), (2, 1, 1), (2, 1, 2),
            (2, 2, 0), (2, 2, 1), (2, 2, 2)]

# Helper classes

class BasicIterClass:
    # [CLASS DOC] HELPER: Iterator yielding the integers from 0 up to n minus one, then
    # raising StopIteration.
    def __init__(self, n):
        # [FUNCTION DOC] HELPER: Store the limit n and start the counter i at 0.
        self.n = n
        self.i = 0
    def __next__(self):
        # [FUNCTION DOC] HELPER: Return the next integer, or raise StopIteration once n values
        # were produced.
        res = self.i
        if res >= self.n:
            raise StopIteration
        self.i = res + 1
        return res
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return self (the object is its own iterator).
        return self

class IteratingSequenceClass:
    # [CLASS DOC] HELPER: Iterable (has __iter__, no __getitem__) whose iterator is a
    # BasicIterClass of n items.
    def __init__(self, n):
        # [FUNCTION DOC] HELPER: Store the item count n.
        self.n = n
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return a fresh BasicIterClass over n items.
        return BasicIterClass(self.n)

class IteratorProxyClass:
    # [CLASS DOC] HELPER: Wrapper iterator that forwards next to the wrapped iterator held in i.
    def __init__(self, i):
        # [FUNCTION DOC] HELPER: Store the wrapped iterator in i.
        self.i = i
    def __next__(self):
        # [FUNCTION DOC] HELPER: Return the next item of the wrapped iterator
        # (StopIteration propagates).
        return next(self.i)
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Return self.
        return self

class SequenceClass:
    # [CLASS DOC] HELPER: Old-style sequence with only __getitem__: yields 0..n-1 and raises
    # IndexError beyond.
    def __init__(self, n):
        # [FUNCTION DOC] HELPER: Store the length n (the attribute may be changed by tests).
        self.n = n
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the index itself when it is within range, otherwise
        # raise IndexError.
        if 0 <= i < self.n:
            return i
        else:
            raise IndexError

class SequenceProxyClass:
    # [CLASS DOC] HELPER: Sequence wrapper whose __getitem__ delegates to the wrapped sequence s.
    def __init__(self, s):
        # [FUNCTION DOC] HELPER: Store the wrapped sequence in s.
        self.s = s
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the item of the wrapped sequence at the given index.
        return self.s[i]

class UnlimitedSequenceClass:
    # [CLASS DOC] HELPER: Endless sequence: __getitem__ returns any index unchanged.
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the index itself.
        return i

class DefaultIterClass:
    # [CLASS DOC] HELPER: Class with no iteration support at all; iter() on it raises TypeError.
    pass

class NoIterClass:
    # [CLASS DOC] HELPER: Class with __getitem__ but __iter__ set to None, which
    # disables iteration.
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the index itself.
        return i
    __iter__ = None

class BadIterableClass:
    # [CLASS DOC] HELPER: Iterable whose __iter__ raises ZeroDivisionError.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Raise ZeroDivisionError.
        raise ZeroDivisionError

class CallableIterClass:
    # [CLASS DOC] HELPER: Callable that returns successive integers starting at 0; used with
    # two-argument iter().
    def __init__(self):
        # [FUNCTION DOC] HELPER: Start the counter i at 0.
        self.i = 0
    def __call__(self):
        # [FUNCTION DOC] HELPER: Return the current counter and increment it; raises IndexError
        # once past 100.
        i = self.i
        self.i = i + 1
        if i > 100:
            raise IndexError # Emergency stop
        return i

class EmptyIterClass:
    # [CLASS DOC] HELPER: Empty sequence: length 0 and __getitem__ always raises StopIteration.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Return 0.
        return 0
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Raise StopIteration.
        raise StopIteration

# Main test suite

class TestCase(__TestCase):
    # [CLASS DOC] Main iterator test case; also provides the helpers check_iterator,
    # check_for_loop and check_pickle used by many tests. Uses the scratch file TESTFN in file
    # based tests.

    # Helper to check that an iterator returns a given sequence
    def check_iterator(self, it, seq, pickle=True):
        # [FUNCTION DOC] HELPER: Drain iterator it with next until StopIteration and assert the
        # collected list equals seq.
        # KNOWN ISSUE: the pickle argument is accepted but never used.
        res = []
        while 1:
            try:
                val = next(it)
            except StopIteration:
                break
            res.append(val)
        self.assertEqual(res, seq)

    # Helper to check that a for loop generates a given sequence
    def check_for_loop(self, expr, seq, pickle=True):
        # [FUNCTION DOC] HELPER: Iterate over expr with a for loop, collect the values and assert
        # they equal seq.
        # KNOWN ISSUE: the pickle argument is accepted but never used.
        res = []
        for val in expr:
            res.append(val)
        self.assertEqual(res, seq)

    # Helper to check picklability
    def check_pickle(self, itorg, seq):
        # [FUNCTION DOC] HELPER: For every pickle protocol, round trip iterator itorg: the copy
        # must be an Iterator yielding seq; then advance a copy one step, pickle it again, and
        # check the rest equals seq without its first item.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # Cannot assert type equality because dict iterators unpickle as list
            # iterators.
            # self.assertEqual(type(itorg), type(it))
            self.assertTrue(isinstance(it, collections.abc.Iterator))
            self.assertEqual(list(it), seq)

            it = pickle.loads(d)
            try:
                next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), seq[1:])

    # Test basic use of iter() function
    def test_iter_basic(self):
        # [FUNCTION DOC] iter() over a range yields the full sequence.
        self.check_iterator(iter(range(10)), list(range(10)))

    # Test that iter(iter(x)) is the same as iter(x)
    def test_iter_idempotency(self):
        # [FUNCTION DOC] iter() applied to an iterator returns the same iterator object.
        seq = list(range(10))
        it = iter(seq)
        it2 = iter(it)
        self.assertTrue(it is it2)

    # Test that for loops over iterators work
    def test_iter_for_loop(self):
        # [FUNCTION DOC] A for loop over an iterator yields every item.
        self.check_for_loop(iter(range(10)), list(range(10)))

    # Test several independent iterators over the same list
    def test_iter_independence(self):
        # [FUNCTION DOC] Three nested loops over independent iterators of the same range give the
        # TRIPLETS list.
        seq = range(3)
        res = []
        for i in iter(seq):
            for j in iter(seq):
                for k in iter(seq):
                    res.append((i, j, k))
        self.assertEqual(res, TRIPLETS)

    # Test triple list comprehension using iterators
    def test_nested_comprehensions_iter(self):
        # [FUNCTION DOC] A nested list comprehension over iterators gives the TRIPLETS list.
        seq = range(3)
        res = [(i, j, k)
               for i in iter(seq) for j in iter(seq) for k in iter(seq)]
        self.assertEqual(res, TRIPLETS)

    # Test triple list comprehension without iterators
    def test_nested_comprehensions_for(self):
        # [FUNCTION DOC] A nested list comprehension over the sequence itself gives the
        # TRIPLETS list.
        seq = range(3)
        res = [(i, j, k) for i in seq for j in seq for k in seq]
        self.assertEqual(res, TRIPLETS)

    # Test a class with __iter__ in a for loop
    def test_iter_class_for(self):
        # [FUNCTION DOC] A for loop over a class with __iter__ yields 0..9.
        self.check_for_loop(IteratingSequenceClass(10), list(range(10)))

    # Test a class with __iter__ with explicit iter()
    def test_iter_class_iter(self):
        # [FUNCTION DOC] iter() on a class with __iter__ yields 0..9.
        self.check_iterator(iter(IteratingSequenceClass(10)), list(range(10)))

    # Test for loop on a sequence class without __iter__
    def test_seq_class_for(self):
        # [FUNCTION DOC] A for loop over a __getitem__-only sequence yields 0..9.
        self.check_for_loop(SequenceClass(10), list(range(10)))

    # Test iter() on a sequence class without __iter__
    def test_seq_class_iter(self):
        # [FUNCTION DOC] iter() on a __getitem__-only sequence yields 0..9.
        self.check_iterator(iter(SequenceClass(10)), list(range(10)))

    def test_mutating_seq_class_iter_pickle(self):
        # [FUNCTION DOC] For every pickle protocol, a pickled iterator over a mutable sequence,
        # taken at start, mid and exhaustion, uses the length the sequence has after unpickling
        # (7); an exhausted iterator unpickles as an empty iterator.
        orig = SequenceClass(5)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # initial iterator
            itorig = iter(orig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(7)))

            # running iterator
            next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(1, 7)))

            # empty iterator
            for i in range(1, 5):
                next(itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertIs(type(it), type(itorig))
            self.assertEqual(list(it), list(range(5, 7)))

            # exhausted iterator
            self.assertRaises(StopIteration, next, itorig)
            d = pickle.dumps((itorig, orig), proto)
            it, seq = pickle.loads(d)
            seq.n = 7
            self.assertTrue(isinstance(it, collections.abc.Iterator))
            self.assertEqual(list(it), [])

    def test_mutating_seq_class_exhausted_iter(self):
        # [FUNCTION DOC] After a sequence grows, an exhausted iterator stays exhausted while a not
        # yet finished iterator picks up the new items, and a fresh iterator sees all of them.
        a = SequenceClass(5)
        exhit = iter(a)
        empit = iter(a)
        for x in exhit:  # exhaust the iterator
            next(empit)  # not exhausted
        a.n = 7
        self.assertEqual(list(exhit), [])
        self.assertEqual(list(empit), [5, 6])
        self.assertEqual(list(a), [0, 1, 2, 3, 4, 5, 6])

    def test_reduce_mutating_builtins_iter(self):
        # [FUNCTION DOC] Regression test for issue 101765: __reduce__ of iterators must not
        # crash even when a custom builtins key's equality check exhausts the iterator.
        # Temporarily replaces entries of the builtins dictionary and restores iter and reversed
        # in a finally block.
        # This is a reproducer of issue #101765
        # where iter `__reduce__` calls could lead to a segfault or SystemError
        # depending on the order of C argument evaluation, which is undefined

        # Backup builtins
        builtins_dict = builtins.__dict__
        orig = {"iter": iter, "reversed": reversed}

        def run(builtin_name, item, sentinel=None):
            # [FUNCTION DOC] HELPER: Install a CustomStr key for builtin_name in the builtins dict
            # (removing the original entry), build an iterator over item (with an optional
            # sentinel), and return its __reduce__ result.
            it = iter(item) if sentinel is None else iter(item, sentinel)

            class CustomStr:
                # [CLASS DOC] HELPER: Dictionary key that hashes like a string name and, when
                # compared, exhausts its iterator.
                def __init__(self, name, iterator):
                    # [FUNCTION DOC] HELPER: Store the name and the iterator.
                    self.name = name
                    self.iterator = iterator
                def __hash__(self):
                    # [FUNCTION DOC] HELPER: Return the hash of the stored name.
                    return hash(self.name)
                def __eq__(self, other):
                    # [FUNCTION DOC] HELPER: Exhaust the stored iterator as a side effect, then
                    # compare other to the name.
                    # Here we exhaust our iterator, possibly changing
                    # its `it_seq` pointer to NULL
                    # The `__reduce__` call should correctly get
                    # the pointers after this call
                    list(self.iterator)
                    return other == self.name

            # del is required here
            # to not prematurely call __eq__ from
            # the hash collision with the old key
            del builtins_dict[builtin_name]
            builtins_dict[CustomStr(builtin_name, it)] = orig[builtin_name]

            return it.__reduce__()

        types = [
            (EmptyIterClass(),),
            (bytes(8),),
            (bytearray(8),),
            ((1, 2, 3),),
            (lambda: 0, 0),
            (tuple[int],)  # GenericAlias
        ]

        try:
            run_iter = functools.partial(run, "iter")
            # The returned value of `__reduce__` should not only be valid
            # but also *empty*, as `it` was exhausted during `__eq__`
            # i.e "xyz" returns (iter, ("",))
            self.assertEqual(run_iter("xyz"), (orig["iter"], ("",)))
            self.assertEqual(run_iter([1, 2, 3]), (orig["iter"], ([],)))

            # _PyEval_GetBuiltin is also called for `reversed` in a branch of
            # listiter_reduce_general
            self.assertEqual(
                run("reversed", orig["reversed"](list(range(8)))),
                (reversed, ([],))
            )

            for case in types:
                self.assertEqual(run_iter(*case), (orig["iter"], ((),)))
        finally:
            # Restore original builtins
            for key, func in orig.items():
                # need to suppress KeyErrors in case
                # a failed test deletes the key without setting anything
                with contextlib.suppress(KeyError):
                    # del is required here
                    # to not invoke our custom __eq__ from
                    # the hash collision with the old key
                    del builtins_dict[key]
                builtins_dict[key] = func

    # Test a new_style class with __iter__ but no next() method
    def test_new_style_iter_class(self):
        # [FUNCTION DOC] An object with __iter__ returning itself but no next method makes iter()
        # raise TypeError.
        class IterClass(object):
            # [CLASS DOC] HELPER: Class with __iter__ returning itself and no __next__.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self.
                return self
        self.assertRaises(TypeError, iter, IterClass())

    # Test two-argument iter() with callable instance
    def test_iter_callable(self):
        # [FUNCTION DOC] Two-argument iter() with a callable instance yields 0..9 and stops at the
        # sentinel 10.
        self.check_iterator(iter(CallableIterClass(), 10), list(range(10)), pickle=True)

    # Test two-argument iter() with function
    def test_iter_function(self):
        # [FUNCTION DOC] Two-argument iter() with a plain function yields 0..9 and stops at the
        # sentinel 10.
        def spam(state=[0]):
            # [FUNCTION DOC] HELPER: Return the counter in the mutable default list and
            # increment it.
            i = state[0]
            state[0] = i+1
            return i
        self.check_iterator(iter(spam, 10), list(range(10)), pickle=False)

    # Test two-argument iter() with function that raises StopIteration
    def test_iter_function_stop(self):
        # [FUNCTION DOC] Two-argument iter() ends when the function raises StopIteration.
        def spam(state=[0]):
            # [FUNCTION DOC] HELPER: Return the counter, raising StopIteration at 10 instead of
            # reaching the sentinel.
            i = state[0]
            if i == 10:
                raise StopIteration
            state[0] = i+1
            return i
        self.check_iterator(iter(spam, 20), list(range(10)), pickle=False)

    def test_iter_function_concealing_reentrant_exhaustion(self):
        # [FUNCTION DOC] Regression test for gh-101892: a function that exhausts its
        # own two-argument iterator without returning the sentinel makes the next call
        # raise StopIteration.
        # gh-101892: Test two-argument iter() with a function that
        # exhausts its associated iterator but forgets to either return
        # a sentinel value or raise StopIteration.
        HAS_MORE = 1
        NO_MORE = 2

        def exhaust(iterator):
            # [FUNCTION DOC] HELPER: Consume all of iterator without letting StopIteration escape.
            """Exhaust an iterator without raising StopIteration."""
            list(iterator)

        def spam():
            # [FUNCTION DOC] HELPER: Return the NO_MORE sentinel on a recursive call;
            # otherwise set the recursion flag, exhaust the iterator stored on this function
            # and return HAS_MORE.
            # Touching the iterator with exhaust() below will call
            # spam() once again so protect against recursion.
            if spam.is_recursive_call:
                return NO_MORE
            spam.is_recursive_call = True
            exhaust(spam.iterator)
            return HAS_MORE

        spam.is_recursive_call = False
        spam.iterator = iter(spam, NO_MORE)
        with self.assertRaises(StopIteration):
            next(spam.iterator)

    # Test exception propagation through function iterator
    def test_exception_function(self):
        # [FUNCTION DOC] A RuntimeError raised by the function of a two-argument iterator
        # propagates after 0..9.
        def spam(state=[0]):
            # [FUNCTION DOC] HELPER: Return the counter, raising RuntimeError when it reaches 10.
            i = state[0]
            state[0] = i+1
            if i == 10:
                raise RuntimeError
            return i
        res = []
        try:
            for x in iter(spam, 20):
                res.append(x)
        except RuntimeError:
            self.assertEqual(res, list(range(10)))
        else:
            self.fail("should have raised RuntimeError")

    # Test exception propagation through sequence iterator
    def test_exception_sequence(self):
        # [FUNCTION DOC] A RuntimeError raised in __getitem__ of a sequence propagates out of the
        # for loop after 0..9.
        class MySequenceClass(SequenceClass):
            # [CLASS DOC] HELPER: SequenceClass subclass whose __getitem__ raises RuntimeError at
            # index 10.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Raise RuntimeError at index 10, otherwise defer to the
                # base class.
                if i == 10:
                    raise RuntimeError
                return SequenceClass.__getitem__(self, i)
        res = []
        try:
            for x in MySequenceClass(20):
                res.append(x)
        except RuntimeError:
            self.assertEqual(res, list(range(10)))
        else:
            self.fail("should have raised RuntimeError")

    # Test for StopIteration from __getitem__
    def test_stop_sequence(self):
        # [FUNCTION DOC] A StopIteration raised in __getitem__ ends the iteration cleanly
        # after 0..9.
        class MySequenceClass(SequenceClass):
            # [CLASS DOC] HELPER: SequenceClass subclass whose __getitem__ raises StopIteration at
            # index 10.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Raise StopIteration at index 10, otherwise defer to the
                # base class.
                if i == 10:
                    raise StopIteration
                return SequenceClass.__getitem__(self, i)
        self.check_for_loop(MySequenceClass(20), list(range(10)), pickle=False)

    # Test a big range
    def test_iter_big_range(self):
        # [FUNCTION DOC] A loop over an iterator of ten thousand items yields all of them.
        self.check_for_loop(iter(range(10000)), list(range(10000)))

    # Test an empty list
    def test_iter_empty(self):
        # [FUNCTION DOC] A loop over an empty list iterator yields nothing.
        self.check_for_loop(iter([]), [])

    # Test a tuple
    def test_iter_tuple(self):
        # [FUNCTION DOC] A loop over a tuple iterator yields its ten items.
        self.check_for_loop(iter((0,1,2,3,4,5,6,7,8,9)), list(range(10)))

    # Test a range
    def test_iter_range(self):
        # [FUNCTION DOC] A loop over a range iterator yields its ten items.
        self.check_for_loop(iter(range(10)), list(range(10)))

    # Test a string
    def test_iter_string(self):
        # [FUNCTION DOC] A loop over a string iterator yields its characters.
        self.check_for_loop(iter("abcde"), ["a", "b", "c", "d", "e"])

    # Test a directory
    def test_iter_dict(self):
        # [FUNCTION DOC] A loop over a dict yields its keys.
        dict = {}
        for i in range(10):
            dict[i] = None
        self.check_for_loop(dict, list(dict.keys()))

    # Test a file
    @torch._dynamo.error_on_graph_break(False)
    def test_iter_file(self):
        # [FUNCTION DOC] A loop over a text file yields its five lines and a second pass yields
        # nothing. Writes and reads the scratch file named TESTFN and removes it at the end.
        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.check_for_loop(f, ["0\n", "1\n", "2\n", "3\n", "4\n"], pickle=False)
            self.check_for_loop(f, [], pickle=False)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test list()'s use of iterators.
    def test_builtin_list(self):
        # [FUNCTION DOC] list() accepts sequences, tuples, dicts and files (twice via seek) and
        # raises TypeError for non-iterables. Writes and reads the scratch file named TESTFN and
        # removes it at the end.
        self.assertEqual(list(SequenceClass(5)), list(range(5)))
        self.assertEqual(list(SequenceClass(0)), [])
        self.assertEqual(list(()), [])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(d), list(d.keys()))

        self.assertRaises(TypeError, list, list)
        self.assertRaises(TypeError, list, 42)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(f), ["0\n", "1\n", "2\n", "3\n", "4\n"])
            f.seek(0, 0)
            self.assertEqual(list(f),
                             ["0\n", "1\n", "2\n", "3\n", "4\n"])
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test tuples()'s use of iterators.
    @torch._dynamo.error_on_graph_break(False)
    def test_builtin_tuple(self):
        # [FUNCTION DOC] tuple() accepts sequences, strings, dicts and files and raises
        # TypeError for non-iterables. Writes and reads the scratch file named TESTFN and
        # removes it at the end.
        self.assertEqual(tuple(SequenceClass(5)), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(SequenceClass(0)), ())
        self.assertEqual(tuple([]), ())
        self.assertEqual(tuple(()), ())
        self.assertEqual(tuple("abc"), ("a", "b", "c"))

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(tuple(d), tuple(d.keys()))

        self.assertRaises(TypeError, tuple, list)
        self.assertRaises(TypeError, tuple, 42)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(5):
                f.write("%d\n" % i)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(tuple(f), ("0\n", "1\n", "2\n", "3\n", "4\n"))
            f.seek(0, 0)
            self.assertEqual(tuple(f),
                             ("0\n", "1\n", "2\n", "3\n", "4\n"))
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test filter()'s use of iterators.
    def test_builtin_filter(self):
        # [FUNCTION DOC] filter() works with sequences, strings, dicts and a custom iterable of
        # truthy objects; a non-iterable argument raises TypeError.
        self.assertEqual(list(filter(None, SequenceClass(5))),
                         list(range(1, 5)))
        self.assertEqual(list(filter(None, SequenceClass(0))), [])
        self.assertEqual(list(filter(None, ())), [])
        self.assertEqual(list(filter(None, "abc")), ["a", "b", "c"])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(filter(None, d)), list(d.keys()))

        self.assertRaises(TypeError, filter, None, list)
        self.assertRaises(TypeError, filter, None, 42)

        class Boolean:
            # [CLASS DOC] HELPER: Object whose truth value is the flag passed at construction.
            def __init__(self, truth):
                # [FUNCTION DOC] HELPER: Store the flag in truth.
                self.truth = truth
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Return the stored truth value.
                return self.truth
        bTrue = Boolean(True)
        bFalse = Boolean(False)

        class Seq:
            # [CLASS DOC] HELPER: Iterable over the positional values passed to its constructor,
            # via SeqIter.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Store the arguments in vals.
                self.vals = args
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return a new SeqIter over vals.
                class SeqIter:
                    # [CLASS DOC] HELPER: Iterator over a list of values.
                    def __init__(self, vals):
                        # [FUNCTION DOC] HELPER: Store vals and start index i at 0.
                        self.vals = vals
                        self.i = 0
                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Return self.
                        return self
                    def __next__(self):
                        # [FUNCTION DOC] HELPER: Return the next value or raise StopIteration at
                        # the end.
                        i = self.i
                        self.i = i + 1
                        if i < len(self.vals):
                            return self.vals[i]
                        else:
                            raise StopIteration
                return SeqIter(self.vals)

        seq = Seq(*([bTrue, bFalse] * 25))
        self.assertEqual(list(filter(lambda x: not x, seq)), [bFalse]*25)
        self.assertEqual(list(filter(lambda x: not x, iter(seq))), [bFalse]*25)

    # Test max() and min()'s use of iterators.
    @torch._dynamo.error_on_graph_break(False)
    def test_builtin_max_min(self):
        # [FUNCTION DOC] max and min work on sequences, dicts, value views and file lines. Writes
        # and reads the scratch file named TESTFN and removes it at the end.
        self.assertEqual(max(SequenceClass(5)), 4)
        self.assertEqual(min(SequenceClass(5)), 0)
        self.assertEqual(max(8, -1), 8)
        self.assertEqual(min(8, -1), -1)

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(max(d), "two")
        self.assertEqual(min(d), "one")
        self.assertEqual(max(d.values()), 3)
        self.assertEqual(min(iter(d.values())), 1)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("medium line\n")
            f.write("xtra large line\n")
            f.write("itty-bitty line\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(min(f), "itty-bitty line\n")
            f.seek(0, 0)
            self.assertEqual(max(f), "xtra large line\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test map()'s use of iterators.
    @torch._dynamo.error_on_graph_break(False)
    def test_builtin_map(self):
        # [FUNCTION DOC] map() works with a sequence, a dict and file lines (lengths of lines).
        # Writes and reads the scratch file named TESTFN and removes it at the end.
        # KNOWN ISSUE: the expected list built from the dict is never used.
        self.assertEqual(list(map(lambda x: x+1, SequenceClass(5))),
                         list(range(1, 6)))

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(map(lambda k, d=d: (k, d[k]), d)),
                         list(d.items()))
        dkeys = list(d.keys())
        expected = [(i < len(d) and dkeys[i] or None,
                     i,
                     i < len(d) and dkeys[i] or None)
                    for i in range(3)]

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            for i in range(10):
                f.write("xy" * i + "\n") # line i has len 2*i+1
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(map(len, f)), list(range(1, 21, 2)))
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test zip()'s use of iterators.
    def test_builtin_zip(self):
        # [FUNCTION DOC] zip() with no arguments, iterables, sequences, dicts and files works;
        # non-iterables raise TypeError; classes that misreport length still zip to five items.
        # Writes and reads the scratch file named TESTFN and removes it at the end.
        self.assertEqual(list(zip()), [])
        self.assertEqual(list(zip(*[])), [])
        self.assertEqual(list(zip(*[(1, 2), 'ab'])), [(1, 'a'), (2, 'b')])

        self.assertRaises(TypeError, zip, None)
        self.assertRaises(TypeError, zip, range(10), 42)
        self.assertRaises(TypeError, zip, range(10), zip)

        self.assertEqual(list(zip(IteratingSequenceClass(3))),
                         [(0,), (1,), (2,)])
        self.assertEqual(list(zip(SequenceClass(3))),
                         [(0,), (1,), (2,)])

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(list(d.items()), list(zip(d, d.values())))

        # Generate all ints starting at constructor arg.
        class IntsFrom:
            # [CLASS DOC] HELPER: Endless iterator of consecutive integers starting at the
            # constructor argument.
            def __init__(self, start):
                # [FUNCTION DOC] HELPER: Store the starting value in i.
                self.i = start

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Return the current integer and advance by one.
                i = self.i
                self.i = i+1
                return i

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "bbb\n" "cc\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            self.assertEqual(list(zip(IntsFrom(0), f, IntsFrom(-100))),
                             [(0, "a\n", -100),
                              (1, "bbb\n", -99),
                              (2, "cc\n", -98)])
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        self.assertEqual(list(zip(range(5))), [(i,) for i in range(5)])

        # Classes that lie about their lengths.
        class NoGuessLen5:
            # [CLASS DOC] HELPER: Sequence of five items with no __len__.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the index, or raise IndexError for 5 and above.
                if i >= 5:
                    raise IndexError
                return i

        class Guess3Len5(NoGuessLen5):
            # [CLASS DOC] HELPER: NoGuessLen5 that misreports its length as 3.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return 3.
                return 3

        class Guess30Len5(NoGuessLen5):
            # [CLASS DOC] HELPER: NoGuessLen5 that misreports its length as 30.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return 30.
                return 30

        def lzip(*args):
            # [FUNCTION DOC] HELPER: Return the list produced by zipping the given iterables.
            return list(zip(*args))

        self.assertEqual(len(Guess3Len5()), 3)
        self.assertEqual(len(Guess30Len5()), 30)
        self.assertEqual(lzip(NoGuessLen5()), lzip(range(5)))
        self.assertEqual(lzip(Guess3Len5()), lzip(range(5)))
        self.assertEqual(lzip(Guess30Len5()), lzip(range(5)))

        expected = [(i, i) for i in range(5)]
        for x in NoGuessLen5(), Guess3Len5(), Guess30Len5():
            for y in NoGuessLen5(), Guess3Len5(), Guess30Len5():
                self.assertEqual(lzip(x, y), expected)

    @torch._dynamo.error_on_graph_break(False)
    def test_unicode_join_endcase(self):
        # [FUNCTION DOC] A join over an iterator that inserts a str among file lines (third item)
        # still joins all items. Writes and reads the scratch file named TESTFN and removes it at
        # the end.

        # This class inserts a Unicode object into its argument's natural
        # iteration, in the 3rd position.
        class OhPhooey:
            # [CLASS DOC] HELPER: Iterator over seq that returns a fixed text in the third
            # position instead of the item.
            def __init__(self, seq):
                # [FUNCTION DOC] HELPER: Create the underlying iterator over seq and start the
                # position counter.
                self.it = iter(seq)
                self.i = 0

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Return the replacement text at position 2, otherwise the
                # next underlying item.
                i = self.i
                self.i = i+1
                if i == 2:
                    return "fooled you!"
                return next(self.it)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" + "b\n" + "c\n")
        finally:
            f.close()

        f = open(TESTFN, "r", encoding="utf-8")
        # Nasty:  string.join(s) can't know whether unicode.join() is needed
        # until it's seen all of s's elements.  But in this case, f's
        # iterator cannot be restarted.  So what we're testing here is
        # whether string.join() can manage to remember everything it's seen
        # and pass that on to unicode.join().
        try:
            got = " - ".join(OhPhooey(f))
            self.assertEqual(got, "a\n - b\n - fooled you! - c\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with 'x in y' and 'x not in y'.
    @torch._dynamo.error_on_graph_break(False)
    def test_in_and_not_in(self):
        # [FUNCTION DOC] The in and not in operators work on sequences, iterators, proxies using
        # the ALWAYS_EQ and NEVER_EQ helpers, dict keys, values and items, and file lines;
        # non-iterables raise TypeError and a failing iterable propagates ZeroDivisionError.
        # Writes and reads the scratch file named TESTFN and removes it at the end.
        for sc5 in IteratingSequenceClass(5), SequenceClass(5):
            for i in range(5):
                self.assertIn(i, sc5)
            for i in "abc", -1, 5, 42.42, (3, 4), [], {1: 1}, 3-12j, sc5:
                self.assertNotIn(i, sc5)

        self.assertIn(ALWAYS_EQ, IteratorProxyClass(iter([1])))
        self.assertIn(ALWAYS_EQ, SequenceProxyClass([1]))
        self.assertNotIn(ALWAYS_EQ, IteratorProxyClass(iter([NEVER_EQ])))
        self.assertNotIn(ALWAYS_EQ, SequenceProxyClass([NEVER_EQ]))
        self.assertIn(NEVER_EQ, IteratorProxyClass(iter([ALWAYS_EQ])))
        self.assertIn(NEVER_EQ, SequenceProxyClass([ALWAYS_EQ]))

        self.assertRaises(TypeError, lambda: 3 in 12)
        self.assertRaises(TypeError, lambda: 3 not in map)
        self.assertRaises(ZeroDivisionError, lambda: 3 in BadIterableClass())

        d = {"one": 1, "two": 2, "three": 3, 1j: 2j}
        for k in d:
            self.assertIn(k, d)
            self.assertNotIn(k, d.values())
        for v in d.values():
            self.assertIn(v, d.values())
            self.assertNotIn(v, d)
        for k, v in d.items():
            self.assertIn((k, v), d.items())
            self.assertNotIn((v, k), d.items())

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            for chunk in "abc":
                f.seek(0, 0)
                self.assertNotIn(chunk, f)
                f.seek(0, 0)
                self.assertIn((chunk + "\n"), f)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with operator.countOf (PySequence_Count).
    def test_countOf(self):
        # [FUNCTION DOC] operator.countOf counts matches in lists, tuples, strings, dicts, value
        # views and file lines; non-iterables raise TypeError. Writes and reads the scratch file
        # named TESTFN and removes it at the end.
        from operator import countOf
        self.assertEqual(countOf([1,2,2,3,2,5], 2), 3)
        self.assertEqual(countOf((1,2,2,3,2,5), 2), 3)
        self.assertEqual(countOf("122325", "2"), 3)
        self.assertEqual(countOf("122325", "6"), 0)

        self.assertRaises(TypeError, countOf, 42, 1)
        self.assertRaises(TypeError, countOf, countOf, countOf)

        d = {"one": 3, "two": 3, "three": 3, 1j: 2j}
        for k in d:
            self.assertEqual(countOf(d, k), 1)
        self.assertEqual(countOf(d.values(), 3), 3)
        self.assertEqual(countOf(d.values(), 2j), 1)
        self.assertEqual(countOf(d.values(), 1j), 0)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n" "b\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            for letter, count in ("a", 1), ("b", 2), ("c", 1), ("d", 0):
                f.seek(0, 0)
                self.assertEqual(countOf(f, letter + "\n"), count)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

    # Test iterators with operator.indexOf (PySequence_Index).
    @torch._dynamo.error_on_graph_break(False)
    def test_indexOf(self):
        # [FUNCTION DOC] operator.indexOf returns the first match position, raises ValueError when
        # missing, consumes a file iterator progressively and handles iterable classes. Writes and
        # reads the scratch file named TESTFN and removes it at the end.
        from operator import indexOf
        self.assertEqual(indexOf([1,2,2,3,2,5], 1), 0)
        self.assertEqual(indexOf((1,2,2,3,2,5), 2), 1)
        self.assertEqual(indexOf((1,2,2,3,2,5), 3), 3)
        self.assertEqual(indexOf((1,2,2,3,2,5), 5), 5)
        self.assertRaises(ValueError, indexOf, (1,2,2,3,2,5), 0)
        self.assertRaises(ValueError, indexOf, (1,2,2,3,2,5), 6)

        self.assertEqual(indexOf("122325", "2"), 1)
        self.assertEqual(indexOf("122325", "5"), 5)
        self.assertRaises(ValueError, indexOf, "122325", "6")

        self.assertRaises(TypeError, indexOf, 42, 1)
        self.assertRaises(TypeError, indexOf, indexOf, indexOf)
        self.assertRaises(ZeroDivisionError, indexOf, BadIterableClass(), 1)

        f = open(TESTFN, "w", encoding="utf-8")
        try:
            f.write("a\n" "b\n" "c\n" "d\n" "e\n")
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            fiter = iter(f)
            self.assertEqual(indexOf(fiter, "b\n"), 1)
            self.assertEqual(indexOf(fiter, "d\n"), 1)
            self.assertEqual(indexOf(fiter, "e\n"), 0)
            self.assertRaises(ValueError, indexOf, fiter, "a\n")
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        iclass = IteratingSequenceClass(3)
        for i in range(3):
            self.assertEqual(indexOf(iclass, i), i)
        self.assertRaises(ValueError, indexOf, iclass, -1)

    # Test iterators with file.writelines().
    @torch._dynamo.error_on_graph_break(False)
    def test_writelines(self):
        # [FUNCTION DOC] file.writelines accepts lists, tuples, dicts and a custom iterable of
        # 2000 lines; None and ints raise TypeError. The written file is read back and compared.
        # Writes and reads the scratch file named TESTFN and removes it at the end.
        f = open(TESTFN, "w", encoding="utf-8")

        try:
            self.assertRaises(TypeError, f.writelines, None)
            self.assertRaises(TypeError, f.writelines, 42)

            f.writelines(["1\n", "2\n"])
            f.writelines(("3\n", "4\n"))
            f.writelines({'5\n': None})
            f.writelines({})

            # Try a big chunk too.
            class Iterator:
                # [CLASS DOC] HELPER: Iterator yielding the integers from start up to finish minus
                # one as newline terminated text.
                def __init__(self, start, finish):
                    # [FUNCTION DOC] HELPER: Store start and finish and set the counter i
                    # to start.
                    self.start = start
                    self.finish = finish
                    self.i = self.start

                def __next__(self):
                    # [FUNCTION DOC] HELPER: Return the next number as text with a newline, or
                    # raise StopIteration at finish.
                    if self.i >= self.finish:
                        raise StopIteration
                    result = str(self.i) + '\n'
                    self.i += 1
                    return result

                def __iter__(self):
                    # [FUNCTION DOC] HELPER: Return self.
                    return self

            class Whatever:
                # [CLASS DOC] HELPER: Iterable producing a fresh Iterator over the given range.
                def __init__(self, start, finish):
                    # [FUNCTION DOC] HELPER: Store start and finish.
                    self.start = start
                    self.finish = finish

                def __iter__(self):
                    # [FUNCTION DOC] HELPER: Return a new Iterator over the stored range.
                    return Iterator(self.start, self.finish)

            f.writelines(Whatever(6, 6+2000))
            f.close()

            f = open(TESTFN, encoding="utf-8")
            expected = [str(i) + "\n" for i in range(1, 2006)]
            self.assertEqual(list(f), expected)

        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass


    # Test iterators on RHS of unpacking assignments.
    @torch._dynamo.error_on_graph_break(False)
    def test_unpack_iter(self):
        # [FUNCTION DOC] Unpacking assignments accept iterable classes, dict views and files; the
        # wrong number of values raises ValueError and a non-iterable raises TypeError. Writes and
        # reads the scratch file named TESTFN and removes it at the end.
        a, b = 1, 2
        self.assertEqual((a, b), (1, 2))

        a, b, c = IteratingSequenceClass(3)
        self.assertEqual((a, b, c), (0, 1, 2))

        try:    # too many values
            a, b = IteratingSequenceClass(3)
        except ValueError:
            pass
        else:
            self.fail("should have raised ValueError")

        try:    # not enough values
            a, b, c = IteratingSequenceClass(2)
        except ValueError:
            pass
        else:
            self.fail("should have raised ValueError")

        try:    # not iterable
            a, b, c = len
        except TypeError:
            pass
        else:
            self.fail("should have raised TypeError")

        a, b, c = {1: 42, 2: 42, 3: 42}.values()
        self.assertEqual((a, b, c), (42, 42, 42))

        f = open(TESTFN, "w", encoding="utf-8")
        lines = ("a\n", "bb\n", "ccc\n")
        try:
            for line in lines:
                f.write(line)
        finally:
            f.close()
        f = open(TESTFN, "r", encoding="utf-8")
        try:
            a, b, c = f
            self.assertEqual((a, b, c), lines)
        finally:
            f.close()
            try:
                unlink(TESTFN)
            except OSError:
                pass

        (a, b), (c,) = IteratingSequenceClass(2), {42: 24}
        self.assertEqual((a, b, c), (0, 1, 42))


    @cpython_only
    def test_ref_counting_behavior(self):
        # [FUNCTION DOC] CPython-only: reference counting of items consumed by iteration and
        # partial unpacking, using a class that tracks live instance counts in a class attribute.
        class C(object):
            # [CLASS DOC] HELPER: Class that counts its live instances in the class
            # attribute count.
            count = 0
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Create an instance of cls and increment the class level
                # live count.
                cls.count += 1
                return object.__new__(cls)
            def __del__(self):
                # [FUNCTION DOC] HELPER: Decrement the class level live count after asserting it
                # was positive.
                cls = self.__class__
                assert cls.count > 0
                cls.count -= 1
        x = C()
        self.assertEqual(C.count, 1)
        del x
        self.assertEqual(C.count, 0)
        l = [C(), C(), C()]
        self.assertEqual(C.count, 3)
        try:
            a, b = iter(l)
        except ValueError:
            pass
        del l
        self.assertEqual(C.count, 0)


    # Make sure StopIteration is a "sink state".
    # This tests various things that weren't sink states in Python 2.2.1,
    # plus various things that always were fine.

    def test_sinkstate_list(self):
        # [FUNCTION DOC] A list iterator that raised StopIteration stays exhausted even if the
        # list grows.
        # This used to fail
        a = list(range(5))
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        a.extend(range(5, 10))
        self.assertEqual(list(b), [])

    def test_sinkstate_tuple(self):
        # [FUNCTION DOC] An exhausted tuple iterator yields nothing further.
        a = (0, 1, 2, 3, 4)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_string(self):
        # [FUNCTION DOC] An exhausted string iterator yields nothing further.
        a = "abcde"
        b = iter(a)
        self.assertEqual(list(b), ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(list(b), [])

    def test_sinkstate_sequence(self):
        # [FUNCTION DOC] An exhausted sequence iterator stays exhausted even if the
        # sequence grows.
        # This used to fail
        a = SequenceClass(5)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        a.n = 10
        self.assertEqual(list(b), [])

    def test_sinkstate_callable(self):
        # [FUNCTION DOC] An exhausted two-argument iterator does not call the function again.
        # This used to fail
        def spam(state=[0]):
            # [FUNCTION DOC] HELPER: Return the counter in the mutable default list and increment
            # it; fail the test via an AssertionError if called past 10.
            i = state[0]
            state[0] = i+1
            if i == 10:
                raise AssertionError("shouldn't have gotten this far")
            return i
        b = iter(spam, 5)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_dict(self):
        # [FUNCTION DOC] Exhausted dict iterators yield nothing further.
        # KNOWN ISSUE: the loop variable b is reassigned to iter of the dict on every pass, so the
        #   key, item and value views are never what is tested.
        # XXX For a more thorough test, see towards the end of:
        # http://mail.python.org/pipermail/python-dev/2002-July/026512.html
        a = {1:1, 2:2, 0:0, 4:4, 3:3}
        for b in iter(a), a.keys(), a.items(), a.values():
            b = iter(a)
            self.assertEqual(len(list(b)), 5)
            self.assertEqual(list(b), [])

    def test_sinkstate_yield(self):
        # [FUNCTION DOC] An exhausted generator yields nothing further.
        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding 0..4.
            for i in range(5):
                yield i
        b = gen()
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_range(self):
        # [FUNCTION DOC] An exhausted range iterator yields nothing further.
        a = range(5)
        b = iter(a)
        self.assertEqual(list(b), list(range(5)))
        self.assertEqual(list(b), [])

    def test_sinkstate_enumerate(self):
        # [FUNCTION DOC] An exhausted enumerate iterator yields nothing further.
        a = range(5)
        e = enumerate(a)
        b = iter(e)
        self.assertEqual(list(b), list(zip(range(5), range(5))))
        self.assertEqual(list(b), [])

    def test_3720(self):
        # [FUNCTION DOC] Regression test 3720: an iterator that deletes its own __next__ method
        # must not crash; a TypeError is acceptable.
        # Avoid a crash, when an iterator deletes its next() method.
        class BadIterator(object):
            # [CLASS DOC] HELPER: Iterator whose first __next__ call deletes the method from
            # the class.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Delete the __next__ method from the class and return 1.
                del BadIterator.__next__
                return 1

        try:
            for i in BadIterator() :
                pass
        except TypeError:
            pass

    def test_extending_list_with_iterator_does_not_segfault(self):
        # [FUNCTION DOC] Smoke test: extending a list from a generator after popping elements
        # gives 760 items.
        # The code to extend a list with an iterator has a fair
        # amount of nontrivial logic in terms of guessing how
        # much memory to allocate in advance, "stealing" refs,
        # and then shrinking at the end.  This is a basic smoke
        # test for that scenario.
        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding 0..499.
            for i in range(500):
                yield i
        lst = [0] * 500
        for i in range(240):
            lst.pop(0)
        lst.extend(gen())
        self.assertEqual(len(lst), 760)

    @cpython_only
    def test_iter_overflow(self):
        # [FUNCTION DOC] CPython-only (issue 22939): a sequence iterator at the maximum index
        # raises OverflowError on the step past it, and keeps raising OverflowError.
        # Test for the issue 22939
        it = iter(UnlimitedSequenceClass())
        # Manually set `it_index` to PY_SSIZE_T_MAX-2 without a loop
        it.__setstate__(sys.maxsize - 2)
        self.assertEqual(next(it), sys.maxsize - 2)
        self.assertEqual(next(it), sys.maxsize - 1)
        with self.assertRaises(OverflowError):
            next(it)
        # Check that Overflow error is always raised
        with self.assertRaises(OverflowError):
            next(it)

    def test_iter_neg_setstate(self):
        # [FUNCTION DOC] Setting a negative state on a sequence iterator restarts it from index 0.
        it = iter(UnlimitedSequenceClass())
        it.__setstate__(-42)
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)

    def test_free_after_iterating(self):
        # [FUNCTION DOC] Uses the support helper to check iterator objects are freed after
        # being iterated.
        check_free_after_iterating(self, iter, SequenceClass, (0,))

    def test_error_iter(self):
        # [FUNCTION DOC] iter() on classes without iteration support raises TypeError; a failing
        # __iter__ propagates its ZeroDivisionError.
        for typ in (DefaultIterClass, NoIterClass):
            self.assertRaises(TypeError, iter, typ())
        self.assertRaises(ZeroDivisionError, iter, BadIterableClass())

    def test_exception_locations(self):
        # [FUNCTION DOC] Errors raised in the constructor, iteration or __iter__ of BrokenIter are
        # attributed to the line and columns of the iterator expression in the traceback.
        # The location of an exception raised from __init__ or
        # __next__ should should be the iterator expression

        def init_raises():
            # [FUNCTION DOC] HELPER: Iterate over a BrokenIter whose constructor raises; return
            # the caught exception.
            try:
                for x in BrokenIter(init_raises=True):
                    pass
            except Exception as e:
                return e

        def next_raises():
            # [FUNCTION DOC] HELPER: Iterate over a BrokenIter whose next raises; return the
            # caught exception.
            try:
                for x in BrokenIter(next_raises=True):
                    pass
            except Exception as e:
                return e

        def iter_raises():
            # [FUNCTION DOC] HELPER: Iterate over a BrokenIter whose __iter__ raises; return the
            # caught exception.
            try:
                for x in BrokenIter(iter_raises=True):
                    pass
            except Exception as e:
                return e

        for func, expected in [(init_raises, "BrokenIter(init_raises=True)"),
                               (next_raises, "BrokenIter(next_raises=True)"),
                               (iter_raises, "BrokenIter(iter_raises=True)"),
                              ]:
            with self.subTest(func):
                exc = func()
                f = traceback.extract_tb(exc.__traceback__)[0]
                indent = 16
                co = func.__code__
                self.assertEqual(f.lineno, co.co_firstlineno + 2)
                self.assertEqual(f.end_lineno, co.co_firstlineno + 2)
                self.assertEqual(f.line[f.colno - indent : f.end_colno - indent],
                                 expected)



if __name__ == "__main__":
    run_tests()
