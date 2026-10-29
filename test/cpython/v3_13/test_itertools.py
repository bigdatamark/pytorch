# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_itertools.py

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 itertools test suite. Defines
# shared helpers (test callables, predicates, pickling copiers, sequence wrapper classes
# that misbehave in specific ways) and several CPythonTestCase classes: TestBasicOps
# (per-function behavior, pickling, tuple reuse, overflow), TestExamples (documentation
# examples), TestPurePythonRoughEquivalents (recipes versus built-ins), TestGC
# (reference cycles), TestVariousIteratorArgs (odd iterable arguments),
# LengthTransparency, RegressionTests and SubclassWithKwargsTest. Import time work:
# builds the picklecopiers list and wraps random helpers with torch._dynamo.disable.
# Running the file calls run_tests.

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    slowTest,
)

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

import doctest
import unittest
import itertools
from test import support
from test.support import threading_helper, script_helper
from itertools import *
import weakref
from decimal import Decimal
from fractions import Fraction
import operator
import random
import copy
import pickle
from functools import reduce
import sys
import struct
import threading
import gc
import warnings

def pickle_deprecated(testfunc):
    # [FUNCTION DOC] HELPER: Decorator for pickling tests. Returns a wrapper that runs
    # the test three times: once expecting a DeprecationWarning, once with those
    # warnings ignored, and once with warnings turned into errors where the test must
    # raise DeprecationWarning, AssertionError or SystemError.
    """ Run the test three times.
    First, verify that a Deprecation Warning is raised.
    Second, run normally but with DeprecationWarnings temporarily disabled.
    Third, run with warnings promoted to errors.
    """
    def inner(self):
        # [FUNCTION DOC] HELPER: The wrapper returned by pickle_deprecated; takes the
        # test case instance and drives the wrapped test under the three warning
        # regimes. Raises if the wrapped test misbehaves in any regime.
        with self.assertWarns(DeprecationWarning):
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=DeprecationWarning)
            with self.assertRaises((DeprecationWarning, AssertionError, SystemError)):
                testfunc(self)

    return inner

maxsize = support.MAX_Py_ssize_t
minsize = -maxsize-1

@torch._dynamo.disable
def choice(*args):
    # [FUNCTION DOC] HELPER: Dynamo-disabled wrapper returning random.choice of the
    # given arguments, so random calls are not traced.
    return random.choice(*args)

@torch._dynamo.disable
def randrange(*args):
    # [FUNCTION DOC] HELPER: Dynamo-disabled wrapper returning random.randrange of the
    # given arguments.
    return random.randrange(*args)

def lzip(*args):
    # [FUNCTION DOC] HELPER: Returns the list of tuples produced by zip over the
    # arguments.
    return list(zip(*args))

def onearg(x):
    # [FUNCTION DOC] HELPER: Test function of one argument; returns twice its input.
    'Test function of one argument'
    return 2*x

def errfunc(*args):
    # [FUNCTION DOC] HELPER: Test function that ignores its arguments and always raises
    # ValueError.
    'Test function that raises an error'
    raise ValueError

def gen3():
    # [FUNCTION DOC] HELPER: Generator yielding 0, 1, 2 once; a non-restartable source.
    'Non-restartable source sequence'
    for i in (0, 1, 2):
        yield i

def isEven(x):
    # [FUNCTION DOC] HELPER: Predicate returning true when x is even.
    'Test predicate'
    return x%2==0

def isOdd(x):
    # [FUNCTION DOC] HELPER: Predicate returning true when x is odd, using remainder
    # equal to one (so negative odd numbers give false).
    'Test predicate'
    return x%2==1

def tupleize(*args):
    # [FUNCTION DOC] HELPER: Returns all positional arguments as a tuple.
    return args

def irange(n):
    # [FUNCTION DOC] HELPER: Generator yielding the integers 0 up to n minus one.
    for i in range(n):
        yield i

class StopNow:
    # [CLASS DOC] HELPER: Stateless iterable emulating an empty iterator; its iterator
    # is itself and next always raises StopIteration.
    'Class emulating an empty iterable.'
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises StopIteration.
        raise StopIteration

def take(n, seq):
    # [FUNCTION DOC] HELPER: Returns the first n items of seq as a list, consuming that
    # much of the iterable (works on infinite ones).
    'Convenience function for partially consuming a long of infinite iterable'
    return list(islice(seq, n))

def prod(iterable):
    # [FUNCTION DOC] HELPER: Returns the product of all items of the iterable, starting
    # from 1.
    return reduce(operator.mul, iterable, 1)

def fact(n):
    # [FUNCTION DOC] HELPER: Returns n factorial as the product of the integers 1
    # through n.
    'Factorial'
    return prod(range(1, n+1))

# root level methods for pickling ability
def _testR(r):
    # [FUNCTION DOC] HELPER: Module-level (picklable) function returning the first
    # element of r.
    return r[0]

def _testR2(r):
    # [FUNCTION DOC] HELPER: Module-level (picklable) function returning the third
    # element of r.
    return r[2]

def underten(x):
    # [FUNCTION DOC] HELPER: Picklable predicate, true when x is less than 10.
    return x<10

picklecopiers = [lambda s, proto=proto: pickle.loads(pickle.dumps(s, proto))
                 for proto in range(pickle.HIGHEST_PROTOCOL + 1)]

class TestBasicOps(__TestCase):
    # [CLASS DOC] Main test case for each function in itertools: argument validation,
    # outputs against known results and pure Python reference versions,
    # copy/deepcopy/pickle round trips, tuple-reuse implementation details and overflow
    # behavior. No instance state; uses module-level helpers.

    def pickletest(self, protocol, it, stop=4, take=1, compare=None):
        # [FUNCTION DOC] HELPER: Asserts that pickling an iterator at a given protocol
        # gives an equal one, also after consuming take items. Expands iterators
        # (recursively, up to stop items each) into lists to compare original and copy;
        # optionally compares to an expected sequence compare. Raises a failure on
        # mismatch.
        """Test that an iterator is the same after pickling, also when part-consumed"""
        def expand(it, i=0):
            # [FUNCTION DOC] HELPER: Expands an iterable into a list of at most stop
            # items, recursing into nested iterables (strings and non-iterables are
            # returned as is). Raises RuntimeError if nesting goes deeper than 10.
            # Recursively expand iterables, within sensible bounds
            if i > 10:
                raise RuntimeError("infinite recursion encountered")
            if isinstance(it, str):
                return it
            try:
                l = list(islice(it, stop))
            except TypeError:
                return it # can't expand it
            return [expand(e, i+1) for e in l]

        # Test the initial copy against the original
        dump = pickle.dumps(it, protocol)
        i2 = pickle.loads(dump)
        self.assertEqual(type(it), type(i2))
        a, b = expand(it), expand(i2)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare)
            self.assertEqual(a, c)

        # Take from the copy, and create another copy and compare them.
        i3 = pickle.loads(dump)
        took = 0
        try:
            for i in range(take):
                next(i3)
                took += 1
        except StopIteration:
            pass #in case there is less data than 'take'
        dump = pickle.dumps(i3, protocol)
        i4 = pickle.loads(dump)
        a, b = expand(i3), expand(i4)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare[took:])
            self.assertEqual(a, c);

    @pickle_deprecated
    def test_accumulate(self):
        # [FUNCTION DOC] Tests accumulate: default addition across
        # int/complex/Decimal/Fraction and strings, empty and single inputs, bad
        # argument counts, custom binary functions (min, max, mul), the initial argument
        # including None, and pickling at all protocols. Marked to also run under the
        # deprecation-warning wrapper.
        self.assertEqual(list(accumulate(range(10))),               # one positional arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        self.assertEqual(list(accumulate(iterable=range(10))),      # kw arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        for typ in int, complex, Decimal, Fraction:                 # multiple types
            self.assertEqual(
                list(accumulate(map(typ, range(10)))),
                list(map(typ, [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])))
        self.assertEqual(list(accumulate('abc')), ['a', 'ab', 'abc'])   # works with non-numeric
        self.assertEqual(list(accumulate([])), [])                  # empty iterable
        self.assertEqual(list(accumulate([7])), [7])                # iterable of length one
        self.assertRaises(TypeError, accumulate, range(10), 5, 6)   # too many args
        self.assertRaises(TypeError, accumulate)                    # too few args
        self.assertRaises(TypeError, accumulate, x=range(10))       # unexpected kwd arg
        self.assertRaises(TypeError, list, accumulate([1, []]))     # args that don't add

        s = [2, 8, 9, 5, 7, 0, 3, 4, 1, 6]
        self.assertEqual(list(accumulate(s, min)),
                         [2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
        self.assertEqual(list(accumulate(s, max)),
                         [2, 8, 9, 9, 9, 9, 9, 9, 9, 9])
        self.assertEqual(list(accumulate(s, operator.mul)),
                         [2, 16, 144, 720, 5040, 0, 0, 0, 0, 0])
        with self.assertRaises(TypeError):
            list(accumulate(s, chr))                                # unary-operation
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, accumulate(range(10)))           # test pickling
            self.pickletest(proto, accumulate(range(10), initial=7))
        self.assertEqual(list(accumulate([10, 5, 1], initial=None)), [10, 15, 16])
        self.assertEqual(list(accumulate([10, 5, 1], initial=100)), [100, 110, 115, 116])
        self.assertEqual(list(accumulate([], initial=100)), [100])
        with self.assertRaises(TypeError):
            list(accumulate([10, 20], 100))

    def test_batched(self):
        # [FUNCTION DOC] Tests batched: tuple batches of size n with a short final
        # batch, strict mode raising ValueError on an incomplete batch, TypeError for
        # missing, extra or wrong-type arguments, ValueError for n of zero or negative,
        # and a sweep over inputs checking order, no loss and batch sizes.
        self.assertEqual(list(batched('ABCDEFG', 3)),
                             [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 2)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 1)),
                            [('A',), ('B',), ('C',), ('D',), ('E',), ('F',), ('G',)])
        self.assertEqual(list(batched('ABCDEF', 2, strict=True)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F')])

        with self.assertRaises(ValueError):         # Incomplete batch when strict
            list(batched('ABCDEFG', 3, strict=True))
        with self.assertRaises(TypeError):          # Too few arguments
            list(batched('ABCDEFG'))
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 3, None))       # Too many arguments
        with self.assertRaises(TypeError):
            list(batched(None, 3))                  # Non-iterable input
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 'hello'))       # n is a string
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', 0))             # n is zero
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', -1))            # n is negative

        data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for n in range(1, 6):
            for i in range(len(data)):
                s = data[:i]
                batches = list(batched(s, n))
                with self.subTest(s=s, n=n, batches=batches):
                    # Order is preserved and no data is lost
                    self.assertEqual(''.join(chain(*batches)), s)
                    # Each batch is an exact tuple
                    self.assertTrue(all(type(batch) is tuple for batch in batches))
                    # All but the last batch is of size n
                    if batches:
                        last_batch = batches.pop()
                        self.assertTrue(all(len(batch) == n for batch in batches))
                        self.assertTrue(len(last_batch) <= n)
                        batches.append(last_batch)

    def test_chain(self):
        # [FUNCTION DOC] Tests chain against a pure Python reference: concatenation of
        # strings, empty input, partial consumption of infinite iterables via take, and
        # TypeError for non-iterable arguments.

        def chain2(*iterables):
            # [FUNCTION DOC] HELPER: Pure Python generator version of chain that yields
            # every element of every iterable in order.
            'Pure python version in the docs'
            for it in iterables:
                for element in it:
                    yield element

        for c in (chain, chain2):
            self.assertEqual(list(c('abc', 'def')), list('abcdef'))
            self.assertEqual(list(c('abc')), list('abc'))
            self.assertEqual(list(c('')), [])
            self.assertEqual(take(4, c('abc', 'def')), list('abcd'))
            self.assertRaises(TypeError, list,c(2, 3))

    def test_chain_from_iterable(self):
        # [FUNCTION DOC] Tests chain.from_iterable: results for lists of strings, empty,
        # partial take, TypeError for non-iterable items, and laziness over an infinite
        # repeat of ranges.
        self.assertEqual(list(chain.from_iterable(['abc', 'def'])), list('abcdef'))
        self.assertEqual(list(chain.from_iterable(['abc'])), list('abc'))
        self.assertEqual(list(chain.from_iterable([''])), [])
        self.assertEqual(take(4, chain.from_iterable(['abc', 'def'])), list('abcd'))
        self.assertRaises(TypeError, list, chain.from_iterable([2, 3]))
        self.assertEqual(list(islice(chain.from_iterable(repeat(range(5))), 2)), [0, 1])

    @pickle_deprecated
    def test_chain_reducible(self):
        # [FUNCTION DOC] Tests that chain supports deepcopy and pickle round trips,
        # including after one item was consumed, and still raises TypeError on non-
        # iterables; also runs the pickletest protocol check.
        for oper in [copy.deepcopy] + picklecopiers:
            it = chain('abc', 'def')
            self.assertEqual(list(oper(it)), list('abcdef'))
            self.assertEqual(next(it), 'a')
            self.assertEqual(list(oper(it)), list('bcdef'))

            self.assertEqual(list(oper(chain(''))), [])
            self.assertEqual(take(4, oper(chain('abc', 'def'))), list('abcd'))
            self.assertRaises(TypeError, list, oper(chain(2, 3)))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, chain('abc', 'def'), compare=list('abcdef'))

    @pickle_deprecated
    def test_chain_setstate(self):
        # [FUNCTION DOC] Tests chain.__setstate__: rejects wrong shapes (empty tuple,
        # list, int, tuple with non-iterator, wrong second item) with TypeError, and
        # accepts one or two iterators, the second being yielded first.
        self.assertRaises(TypeError, chain().__setstate__, ())
        self.assertRaises(TypeError, chain().__setstate__, [])
        self.assertRaises(TypeError, chain().__setstate__, 0)
        self.assertRaises(TypeError, chain().__setstate__, ([],))
        self.assertRaises(TypeError, chain().__setstate__, (iter([]), []))
        it = chain()
        it.__setstate__((iter(['abc', 'def']),))
        self.assertEqual(list(it), ['a', 'b', 'c', 'd', 'e', 'f'])
        it = chain()
        it.__setstate__((iter(['abc', 'def']), iter(['ghi'])))
        self.assertEqual(list(it), ['ghi', 'a', 'b', 'c', 'd', 'e', 'f'])

    @pickle_deprecated
    def test_combinations(self):
        # [FUNCTION DOC] Tests combinations: argument errors (missing r, extra args,
        # non-iterable pool, negative r), r larger than n gives empty, exact outputs
        # with copies and pickles also after partial consumption, and a sweep for n up
        # to 6 checking count, uniqueness, lexicographic order and subsequence property,
        # equality with three pure Python implementations, plus pickling.
        self.assertRaises(TypeError, combinations, 'abc')       # missing r argument
        self.assertRaises(TypeError, combinations, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, combinations, None)        # pool is not iterable
        self.assertRaises(ValueError, combinations, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(combinations('abc', 32))), [])     # r > n

            self.assertEqual(list(op(combinations('ABCD', 2))),
                             [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])
            testIntermediate = combinations('ABCD', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])

            self.assertEqual(list(op(combinations(range(4), 3))),
                             [(0,1,2), (0,1,3), (0,2,3), (1,2,3)])
            testIntermediate = combinations(range(4), 3)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [(0,1,3), (0,2,3), (1,2,3)])

        def combinations1(iterable, r):
            # [FUNCTION DOC] HELPER: Pure Python reference generating r-length
            # combinations with an index list incremented in lexicographic order.
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def combinations2(iterable, r):
            # [FUNCTION DOC] HELPER: Reference combinations built by filtering
            # permutations of indices that are already sorted.
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in permutations(range(n), r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def combinations3(iterable, r):
            # [FUNCTION DOC] HELPER: Reference combinations built by filtering
            # combinations_with_replacement of indices to those with no repeats.
            'Pure python version from cwr()'
            pool = tuple(iterable)
            n = len(pool)
            for indices in combinations_with_replacement(range(n), r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(combinations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(r) / fact(n-r)) # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    self.assertEqual(len(set(c)), r)                    # no duplicate elements
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(list(c),
                                     [e for e in values if e in c])      # comb is a subsequence of the input iterable
                self.assertEqual(result, list(combinations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(combinations2(values, r))) # matches second pure python version
                self.assertEqual(result, list(combinations3(values, r))) # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, combinations(values, r))      # test pickling

    @support.bigaddrspacetest
    def test_combinations_overflow(self):
        # [FUNCTION DOC] Decorated for big address space tests only. Checks combinations
        # with a huge r raises OverflowError or MemoryError.
        with self.assertRaises((OverflowError, MemoryError)):
            combinations("AA", 2**29)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_tuple_reuse(self):
        # [FUNCTION DOC] CPython implementation detail: the same tuple object is reused
        # between next calls while it is not retained by the caller; checks one distinct
        # id when iterated and several when listed.
        self.assertEqual(len(set(map(id, combinations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(combinations('abcde', 3))))), 1)

    @pickle_deprecated
    def test_combinations_with_replacement(self):
        # [FUNCTION DOC] Tests combinations_with_replacement analogously to
        # combinations: argument errors, exact outputs with copy/pickle, sweep over
        # sizes checking count formula, uniqueness, order, relation to plain
        # combinations and to two pure Python implementations, and pickling.
        cwr = combinations_with_replacement
        self.assertRaises(TypeError, cwr, 'abc')       # missing r argument
        self.assertRaises(TypeError, cwr, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, cwr, None)        # pool is not iterable
        self.assertRaises(ValueError, cwr, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(cwr('ABC', 2))),
                             [('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])
            testIntermediate = cwr('ABC', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])


        def cwr1(iterable, r):
            # [FUNCTION DOC] HELPER: Reference combinations with replacement using an
            # index list where a trailing run is reset to the incremented value.
            'Pure python version shown in the docs'
            # number items returned:  (n+r-1)! / r! / (n-1)! when n>0
            pool = tuple(iterable)
            n = len(pool)
            if not n and r:
                return
            indices = [0] * r
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != n - 1:
                        break
                else:
                    return
                indices[i:] = [indices[i] + 1] * (r - i)
                yield tuple(pool[i] for i in indices)

        def cwr2(iterable, r):
            # [FUNCTION DOC] HELPER: Reference version filtering the cartesian product
            # of index ranges to non-decreasing index tuples.
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in product(range(n), repeat=r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def numcombs(n, r):
            # [FUNCTION DOC] HELPER: Expected number of combinations with replacement: 1
            # or 0 when the pool is empty (depending on r), otherwise a factorial
            # formula.
            if not n:
                return 0 if r else 1
            return fact(n+r-1) / fact(r)/ fact(n-1)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(cwr(values, r))

                self.assertEqual(len(result), numcombs(n, r))           # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order

                regular_combs = list(combinations(values, r))           # compare to combs without replacement
                if n == 0 or r <= 1:
                    self.assertEqual(result, regular_combs)            # cases that should be identical
                else:
                    self.assertTrue(set(result) >= set(regular_combs))     # rest should be supersets of regular combs

                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    noruns = [k for k,v in groupby(c)]                  # combo without consecutive repeats
                    self.assertEqual(len(noruns), len(set(noruns)))     # no repeats other than consecutive
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(noruns,
                                     [e for e in values if e in c])     # comb is a subsequence of the input iterable
                self.assertEqual(result, list(cwr1(values, r)))         # matches first pure python version
                self.assertEqual(result, list(cwr2(values, r)))         # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, cwr(values,r))               # test pickling

    @support.bigaddrspacetest
    def test_combinations_with_replacement_overflow(self):
        # [FUNCTION DOC] Big address space only. Checks combinations_with_replacement
        # with an enormous r raises OverflowError or MemoryError.
        with self.assertRaises((OverflowError, MemoryError)):
            combinations_with_replacement("AA", 2**30)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_with_replacement_tuple_reuse(self):
        # [FUNCTION DOC] CPython-only check of tuple reuse for
        # combinations_with_replacement: one id while iterating, several once collected
        # in a list.
        cwr = combinations_with_replacement
        self.assertEqual(len(set(map(id, cwr('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(cwr('abcde', 3))))), 1)

    def test_permutations(self):
        # [FUNCTION DOC] Tests permutations: r larger than n empty, exact small output,
        # sweep over n up to 4 checking count, uniqueness, order, element membership and
        # agreement with two pure Python versions, plus r defaulting to the pool size
        # (None or omitted).
        self.assertEqual(list(permutations('abc', 32)), [])     # r > n
        self.assertEqual(list(permutations(range(3), 2)),
                                           [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)])

        def permutations1(iterable, r=None):
            # [FUNCTION DOC] HELPER: Reference permutations using indices and cycle
            # counters; r defaults to the pool size, returns nothing if r exceeds it.
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            if r > n:
                return
            indices = list(range(n))
            cycles = list(range(n-r+1, n+1))[::-1]
            yield tuple(pool[i] for i in indices[:r])
            while n:
                for i in reversed(range(r)):
                    cycles[i] -= 1
                    if cycles[i] == 0:
                        indices[i:] = indices[i+1:] + indices[i:i+1]
                        cycles[i] = n - i
                    else:
                        j = cycles[i]
                        indices[i], indices[-j] = indices[-j], indices[i]
                        yield tuple(pool[i] for i in indices[:r])
                        break
                else:
                    return

        def permutations2(iterable, r=None):
            # [FUNCTION DOC] HELPER: Reference permutations filtering the product of
            # index ranges to tuples with no repeated index.
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            for indices in product(range(n), repeat=r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(5):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(permutations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(n-r))      # right number of perms
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for p in result:
                    self.assertEqual(len(p), r)                         # r-length permutations
                    self.assertEqual(len(set(p)), r)                    # no duplicate elements
                    self.assertTrue(all(e in values for e in p))           # elements taken from input iterable
                self.assertEqual(result, list(permutations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(permutations2(values, r))) # matches second pure python version
                if r == n:
                    self.assertEqual(result, list(permutations(values, None))) # test r as None
                    self.assertEqual(result, list(permutations(values)))       # test default r

    @support.bigaddrspacetest
    def test_permutations_overflow(self):
        # [FUNCTION DOC] Big address space only. Checks permutations with an enormous r
        # raises OverflowError or MemoryError.
        with self.assertRaises((OverflowError, MemoryError)):
            permutations("A", 2**30)

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_permutations_tuple_reuse(self):
        # [FUNCTION DOC] CPython-only check of tuple reuse for permutations (one id
        # while iterating, many when listed).
        self.assertEqual(len(set(map(id, permutations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(permutations('abcde', 3))))), 1)

    def test_combinatorics(self):
        # [FUNCTION DOC] Checks relationships among product,
        # combinations_with_replacement, permutations and combinations for small sizes:
        # output sizes, sorted unique tuples, and each is derivable by filtering another
        # (for example combinations are the sorted permutations).
        # Test relationships between product(), permutations(),
        # combinations() and combinations_with_replacement().

        for n in range(6):
            s = 'ABCDEFG'[:n]
            for r in range(8):
                prod = list(product(s, repeat=r))
                cwr = list(combinations_with_replacement(s, r))
                perm = list(permutations(s, r))
                comb = list(combinations(s, r))

                # Check size
                self.assertEqual(len(prod), n**r)
                self.assertEqual(len(cwr), (fact(n+r-1) / fact(r)/ fact(n-1)) if n else (not r))
                self.assertEqual(len(perm), 0 if r>n else fact(n) / fact(n-r))
                self.assertEqual(len(comb), 0 if r>n else fact(n) / fact(r) / fact(n-r))

                # Check lexicographic order without repeated tuples
                self.assertEqual(prod, sorted(set(prod)))
                self.assertEqual(cwr, sorted(set(cwr)))
                self.assertEqual(perm, sorted(set(perm)))
                self.assertEqual(comb, sorted(set(comb)))

                # Check interrelationships
                self.assertEqual(cwr, [t for t in prod if sorted(t)==list(t)]) # cwr: prods which are sorted
                self.assertEqual(perm, [t for t in prod if len(set(t))==r])    # perm: prods with no dups
                self.assertEqual(comb, [t for t in perm if sorted(t)==list(t)]) # comb: perms that are sorted
                self.assertEqual(comb, [t for t in cwr if len(set(t))==r])      # comb: cwrs without dups
                self.assertEqual(comb, list(filter(set(cwr).__contains__, perm)))     # comb: perm that is a cwr
                self.assertEqual(comb, list(filter(set(perm).__contains__, cwr)))     # comb: cwr that is a perm
                self.assertEqual(comb, sorted(set(cwr) & set(perm)))            # comb: both a cwr and a perm

    @pickle_deprecated
    def test_compress(self):
        # [FUNCTION DOC] Tests compress: keyword and positional use, all-zero and all-
        # one selectors, unequal lengths, a large lazy case, TypeError for bad
        # arguments, and copy, deepcopy and pickle behavior also after one item is
        # consumed.
        # KNOWN ISSUE: the last argument error check repeats the second-argument None
        # call instead of passing too many arguments.
        self.assertEqual(list(compress(data='ABCDEF', selectors=[1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [0,0,0,0,0,0])), list(''))
        self.assertEqual(list(compress('ABCDEF', [1,1,1,1,1,1])), list('ABCDEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1])), list('AC'))
        self.assertEqual(list(compress('ABC', [0,1,1,1,1,1])), list('BC'))
        n = 10000
        data = chain.from_iterable(repeat(range(6), n))
        selectors = chain.from_iterable(repeat((0, 1)))
        self.assertEqual(list(compress(data, selectors)), [1,3,5] * n)
        self.assertRaises(TypeError, compress, None, range(6))      # 1st arg not iterable
        self.assertRaises(TypeError, compress, range(6), None)      # 2nd arg not iterable
        self.assertRaises(TypeError, compress, range(6))            # too few args
        self.assertRaises(TypeError, compress, range(6), None)      # too many args

        # check copy, deepcopy, pickle
        for op in [lambda a:copy.copy(a), lambda a:copy.deepcopy(a)] + picklecopiers:
            for data, selectors, result1, result2 in [
                ('ABCDEF', [1,0,1,0,1,1], 'ACEF', 'CEF'),
                ('ABCDEF', [0,0,0,0,0,0], '', ''),
                ('ABCDEF', [1,1,1,1,1,1], 'ABCDEF', 'BCDEF'),
                ('ABCDEF', [1,0,1], 'AC', 'C'),
                ('ABC', [0,1,1,1,1,1], 'BC', 'C'),
                ]:

                self.assertEqual(list(op(compress(data=data, selectors=selectors))), list(result1))
                self.assertEqual(list(op(compress(data, selectors))), list(result1))
                testIntermediate = compress(data, selectors)
                if result1:
                    next(testIntermediate)
                    self.assertEqual(list(op(testIntermediate)), list(result2))

    @pickle_deprecated
    def test_count(self):
        # [FUNCTION DOC] Tests count with default step: starting values (negative, large
        # near the machine maximum, float, complex, Decimal, Fraction, huge ints), repr
        # text changing after next, type preservation, copy, deepcopy and pickle, and
        # that an enormous step does not leave an error set.
        self.assertEqual(lzip('abc',count()), [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(lzip('abc',count(3)), [('a', 3), ('b', 4), ('c', 5)])
        self.assertEqual(take(2, lzip('abc',count(3))), [('a', 3), ('b', 4)])
        self.assertEqual(take(2, zip('abc',count(-1))), [('a', -1), ('b', 0)])
        self.assertEqual(take(2, zip('abc',count(-3))), [('a', -3), ('b', -2)])
        self.assertRaises(TypeError, count, 2, 3, 4)
        self.assertRaises(TypeError, count, 'a')
        self.assertEqual(take(3, count(maxsize)),
                        [maxsize, maxsize + 1, maxsize + 2])
        self.assertEqual(take(10, count(maxsize-5)),
                         list(range(maxsize-5, maxsize+5)))
        self.assertEqual(take(10, count(-maxsize-5)),
                         list(range(-maxsize-5, -maxsize+5)))
        self.assertEqual(take(3, count(3.25)), [3.25, 4.25, 5.25])
        self.assertEqual(take(3, count(3.25-4j)), [3.25-4j, 4.25-4j, 5.25-4j])
        self.assertEqual(take(3, count(Decimal('1.1'))),
                         [Decimal('1.1'), Decimal('2.1'), Decimal('3.1')])
        self.assertEqual(take(3, count(Fraction(2, 3))),
                         [Fraction(2, 3), Fraction(5, 3), Fraction(8, 3)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(BIGINT)), [BIGINT, BIGINT+1, BIGINT+2])
        c = count(3)
        self.assertEqual(repr(c), 'count(3)')
        next(c)
        self.assertEqual(repr(c), 'count(4)')
        c = count(-9)
        self.assertEqual(repr(c), 'count(-9)')
        next(c)
        self.assertEqual(next(c), -8)
        self.assertEqual(repr(count(10.25)), 'count(10.25)')
        self.assertEqual(repr(count(10.0)), 'count(10.0)')

        self.assertEqual(repr(count(maxsize)), f'count({maxsize})')
        c = count(maxsize - 1)
        self.assertEqual(repr(c), f'count({maxsize - 1})')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize})')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 1})')

        self.assertEqual(type(next(count(10.0))), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            # Test repr
            r1 = repr(count(i))
            r2 = 'count(%r)'.__mod__(i)
            self.assertEqual(r1, r2)

        # check copy, deepcopy, pickle
        for value in -3, 3, maxsize-5, maxsize+5:
            c = count(value)
            self.assertEqual(next(copy.copy(c)), value)
            self.assertEqual(next(copy.deepcopy(c)), value)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, count(value))

        #check proper internal error handling for large "step' sizes
        count(1, maxsize+5); sys.exc_info()

    @pickle_deprecated
    def test_count_with_step(self):
        # [FUNCTION DOC] Tests count with a step: positional and keyword forms, zero,
        # negative and fractional steps, steps beyond the machine range, huge ints, repr
        # (step of integer one is suppressed but float one is shown), type of successive
        # values, and pickling across many start/step combinations.
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(start=2,step=3)),
                         [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(step=-1)),
                         [('a', 0), ('b', -1), ('c', -2)])
        self.assertRaises(TypeError, count, 'a', 'b')
        self.assertEqual(lzip('abc',count(2,0)), [('a', 2), ('b', 2), ('c', 2)])
        self.assertEqual(lzip('abc',count(2,1)), [('a', 2), ('b', 3), ('c', 4)])
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(take(20, count(maxsize-15, 3)), take(20, range(maxsize-15, maxsize+100, 3)))
        self.assertEqual(take(20, count(-maxsize-15, 3)), take(20, range(-maxsize-15,-maxsize+100, 3)))
        self.assertEqual(take(3, count(10, maxsize+5)),
                         list(range(10, 10+3*(maxsize+5), maxsize+5)))
        self.assertEqual(take(3, count(maxsize, 2)),
                         [maxsize, maxsize + 2, maxsize + 4])
        self.assertEqual(take(3, count(maxsize, maxsize)),
                         [maxsize, 2 * maxsize, 3 * maxsize])
        self.assertEqual(take(3, count(-maxsize, maxsize)),
                        [-maxsize, 0, maxsize])
        self.assertEqual(take(3, count(2, 1.25)), [2, 3.25, 4.5])
        self.assertEqual(take(3, count(2, 3.25-4j)), [2, 5.25-4j, 8.5-8j])
        self.assertEqual(take(3, count(Decimal('1.1'), Decimal('.1'))),
                         [Decimal('1.1'), Decimal('1.2'), Decimal('1.3')])
        self.assertEqual(take(3, count(Fraction(2,3), Fraction(1,7))),
                         [Fraction(2,3), Fraction(17,21), Fraction(20,21)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(step=BIGINT)), [0, BIGINT, 2*BIGINT])
        self.assertEqual(repr(take(3, count(10, 2.5))), repr([10, 12.5, 15.0]))
        c = count(3, 5)
        self.assertEqual(repr(c), 'count(3, 5)')
        next(c)
        self.assertEqual(repr(c), 'count(8, 5)')
        c = count(-9, 0)
        self.assertEqual(repr(c), 'count(-9, 0)')
        next(c)
        self.assertEqual(repr(c), 'count(-9, 0)')
        c = count(-9, -3)
        self.assertEqual(repr(c), 'count(-9, -3)')
        next(c)
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(count(10.5, 1.25)), 'count(10.5, 1.25)')
        self.assertEqual(repr(count(10.5, 1)), 'count(10.5)')           # suppress step=1 when it's an int
        self.assertEqual(repr(count(10.5, 1.00)), 'count(10.5, 1.0)')   # do show float values lilke 1.0
        self.assertEqual(repr(count(10, 1.00)), 'count(10, 1.0)')
        c = count(10, 1.0)
        self.assertEqual(type(next(c)), int)
        self.assertEqual(type(next(c)), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            for j in  (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 1, 10, sys.maxsize-5, sys.maxsize+5):
                # Test repr
                r1 = repr(count(i, j))
                if j == 1:
                    r2 = ('count(%r)' % i)
                else:
                    r2 = ('count(%r, %r)' % (i, j))
                self.assertEqual(r1, r2)
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, count(i, j))

        c = count(maxsize -2, 2)
        self.assertEqual(repr(c), f'count({maxsize - 2}, 2)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, 2)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 2}, 2)')

        c = count(maxsize + 1, -1)
        self.assertEqual(repr(c), f'count({maxsize + 1}, -1)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, -1)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize - 1}, -1)')

    @threading_helper.requires_working_threading()
    def test_count_threading(self, step=1):
        # [FUNCTION DOC] Needs working threading. Starts ten threads each calling next
        # 10000 times on one shared count with the given step, joins them, and asserts
        # the next value equals the total number of calls times the step.
        # this test verifies multithreading consistency, which is
        # mostly for testing builds without GIL, but nice to test anyway
        count_to = 10_000
        num_threads = 10
        c = count(step=step)
        def counting_thread():
            # [FUNCTION DOC] HELPER: Thread body calling next on the shared counter
            # count_to times.
            for i in range(count_to):
                next(c)
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=counting_thread)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.assertEqual(next(c), count_to * num_threads * step)

    def test_count_with_step_threading(self):
        # [FUNCTION DOC] Runs the threading count test with a step of 5.
        self.test_count_threading(step=5)

    def test_cycle(self):
        # [FUNCTION DOC] Tests cycle: repeating a string, empty input gives empty, non-
        # iterable raises TypeError, and cycling a one-shot generator caches its items.
        self.assertEqual(take(10, cycle('abc')), list('abcabcabca'))
        self.assertEqual(list(cycle('')), [])
        # self.assertRaises(TypeError, cycle)
        self.assertRaises(TypeError, cycle, 5)
        self.assertEqual(list(islice(cycle(gen3()),10)), [0,1,2,0,1,2,0,1,2,0])

    @pickle_deprecated
    def test_cycle_copy_pickle(self):
        # [FUNCTION DOC] Tests cycle copy and pickle: deepcopy and pickle at every
        # protocol after consuming items continue from the right place, pickletest round
        # trip, and cycles over partially and fully consumed source iterators resume
        # correctly. Shallow copy is deliberately not tested.
        # check copy, deepcopy, pickle
        c = cycle('abc')
        self.assertEqual(next(c), 'a')
        #simple copy currently not supported, because __reduce__ returns
        #an internal iterator
        #self.assertEqual(take(10, copy.copy(c)), list('bcabcabcab'))
        self.assertEqual(take(10, copy.deepcopy(c)), list('bcabcabcab'))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('bcabcabcab'))
            next(c)
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('cabcabcabc'))
            next(c)
            next(c)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, cycle('abc'))

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # test with partial consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(2)]      # consume 2 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

            # test with completely consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(7)]      # consume 7 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

    @pickle_deprecated
    def test_cycle_unpickle_compat(self):
        # [FUNCTION DOC] Loads 20 hard-coded pickle byte strings (protocols 0 to 4,
        # several state encodings) produced by older Python versions for a cycle over a
        # partly consumed list, and checks each unpickles to a cycle whose next ten
        # items are 2, 3, 1, 2, 3, 1, 2, 3, 1, 2. Uses a plain assert for the count of
        # cases.
        testcases = [
            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI0\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aK\x00tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x04\x95=\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI1\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00K\x01tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI00\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aI00\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI01\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00I01\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x04\x95;\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
        ]
        assert len(testcases) == 20
        for t in testcases:
            it = pickle.loads(t)
            self.assertEqual(take(10, it), [2, 3, 1, 2, 3, 1, 2, 3, 1, 2])

    @pickle_deprecated
    def test_cycle_setstate(self):
        # [FUNCTION DOC] Tests cycle.__setstate__ in both modes (state holds only
        # consumed values with flag 0, or all seen values with flag 1) yielding the same
        # infinite sequence, and TypeError for a non-tuple state, a non-list first
        # element, a non-int flag, or malformed tuples.
        # Verify both modes for restoring state

        # Mode 0 is efficient.  It uses an incompletely consumed input
        # iterator to build a cycle object and then passes in state with
        # a list of previously consumed values.  There is no data
        # overlap between the two.
        c = cycle('defg')
        c.__setstate__((list('abc'), 0))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # Mode 1 is inefficient.  It starts with a cycle object built
        # from an iterator over the remaining elements in a partial
        # cycle and then passes in state with all of the previously
        # seen values (this overlaps values included in the iterator).
        c = cycle('defg')
        c.__setstate__((list('abcdefg'), 1))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # The first argument to setstate needs to be a tuple
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__([list('abcdefg'), 0])

        # The first argument in the setstate tuple must be a list
        with self.assertRaises(TypeError):
            c = cycle('defg')
            c.__setstate__((tuple('defg'), 0))
        take(20, c)

        # The second argument in the setstate tuple must be an int
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__((list('abcdefg'), 'x'))

        self.assertRaises(TypeError, cycle('').__setstate__, ())
        self.assertRaises(TypeError, cycle('').__setstate__, ([],))

    @pickle_deprecated
    def test_groupby(self):
        # [FUNCTION DOC] Tests groupby: argument handling (empty input, key, bad types),
        # grouping with a key function, pickled and nested groupby, skipping inner
        # groups, using inner groups after the outer iterator advanced (they are empty),
        # shell-like sort/uniq pipelines on a word, and propagation of exceptions from
        # iteration, equality comparison and the key function on both outer and inner
        # objects.
        # Check whether it accepts arguments correctly
        self.assertEqual([], list(groupby([])))
        self.assertEqual([], list(groupby([], key=id)))
        self.assertRaises(TypeError, list, groupby('abc', []))
        self.assertRaises(TypeError, groupby, None)
        self.assertRaises(TypeError, groupby, 'abc', lambda x:x, 10)

        # Check normal input
        s = [(0, 10, 20), (0, 11,21), (0,12,21), (1,13,21), (1,14,22),
             (2,15,22), (3,16,23), (3,17,23)]
        dup = []
        for k, g in groupby(s, lambda r:r[0]):
            for elem in g:
                self.assertEqual(k, elem[0])
                dup.append(elem)
        self.assertEqual(s, dup)

        # Check normal pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for elem in g:
                    self.assertEqual(k, elem[0])
                    dup.append(elem)
            self.assertEqual(s, dup)

        # Check nested case
        dup = []
        for k, g in groupby(s, _testR):
            for ik, ig in groupby(g, _testR2):
                for elem in ig:
                    self.assertEqual(k, elem[0])
                    self.assertEqual(ik, elem[2])
                    dup.append(elem)
        self.assertEqual(s, dup)

        # Check nested and pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for ik, ig in pickle.loads(pickle.dumps(groupby(g, _testR2), proto)):
                    for elem in ig:
                        self.assertEqual(k, elem[0])
                        self.assertEqual(ik, elem[2])
                        dup.append(elem)
            self.assertEqual(s, dup)


        # Check case where inner iterator is not used
        keys = [k for k, g in groupby(s, _testR)]
        expectedkeys = set([r[0] for r in s])
        self.assertEqual(set(keys), expectedkeys)
        self.assertEqual(len(keys), len(expectedkeys))

        # Check case where inner iterator is used after advancing the groupby
        # iterator
        s = list(zip('AABBBAAAA', range(9)))
        it = groupby(s, _testR)
        _, g1 = next(it)
        _, g2 = next(it)
        _, g3 = next(it)
        self.assertEqual(list(g1), [])
        self.assertEqual(list(g2), [])
        self.assertEqual(next(g3), ('A', 5))
        list(it)  # exhaust the groupby iterator
        self.assertEqual(list(g3), [])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it = groupby(s, _testR)
            _, g = next(it)
            next(it)
            next(it)
            self.assertEqual(list(pickle.loads(pickle.dumps(g, proto))), [])

        # Exercise pipes and filters style
        s = 'abracadabra'
        # sort s | uniq
        r = [k for k, g in groupby(sorted(s))]
        self.assertEqual(r, ['a', 'b', 'c', 'd', 'r'])
        # sort s | uniq -d
        r = [k for k, g in groupby(sorted(s)) if list(islice(g,1,2))]
        self.assertEqual(r, ['a', 'b', 'r'])
        # sort s | uniq -c
        r = [(len(list(g)), k) for k, g in groupby(sorted(s))]
        self.assertEqual(r, [(5, 'a'), (2, 'b'), (1, 'c'), (1, 'd'), (2, 'r')])
        # sort s | uniq -c | sort -rn | head -3
        r = sorted([(len(list(g)) , k) for k, g in groupby(sorted(s))], reverse=True)[:3]
        self.assertEqual(r, [(5, 'a'), (2, 'r'), (2, 'b')])

        # iter.__next__ failure
        class ExpectedError(Exception):
            # [CLASS DOC] HELPER: Marker exception raised deliberately by the fixtures
            # in this test.
            pass
        def delayed_raise(n=0):
            # [FUNCTION DOC] HELPER: Generator yielding the string yo n times and then
            # raising ExpectedError.
            for i in range(n):
                yield 'yo'
            raise ExpectedError
        def gulp(iterable, keyp=None, func=list):
            # [FUNCTION DOC] HELPER: Groups the iterable with the optional key function
            # and returns a list of func applied to each group (list by default),
            # forcing consumption of every group.
            return [func(g) for k, g in groupby(iterable, keyp)]

        # iter.__next__ failure on outer object
        self.assertRaises(ExpectedError, gulp, delayed_raise(0))
        # iter.__next__ failure on inner object
        self.assertRaises(ExpectedError, gulp, delayed_raise(1))

        # __eq__ failure
        class DummyCmp:
            # [CLASS DOC] HELPER: Object whose equality always raises ExpectedError,
            # used to test error propagation from key comparison.
            def __eq__(self, dst):
                # [FUNCTION DOC] HELPER: Always raises ExpectedError.
                raise ExpectedError
        s = [DummyCmp(), DummyCmp(), None]

        # __eq__ failure on outer object
        self.assertRaises(ExpectedError, gulp, s, func=id)
        # __eq__ failure on inner object
        self.assertRaises(ExpectedError, gulp, s)

        # keyfunc failure
        def keyfunc(obj):
            # [FUNCTION DOC] HELPER: Key function that returns the object while the skip
            # counter attribute on the function is positive (decrementing it), then
            # raises ExpectedError.
            if keyfunc.skip > 0:
                keyfunc.skip -= 1
                return obj
            else:
                raise ExpectedError

        # keyfunc failure on outer object
        keyfunc.skip = 0
        self.assertRaises(ExpectedError, gulp, [None], keyfunc)
        keyfunc.skip = 1
        self.assertRaises(ExpectedError, gulp, [None, None], keyfunc)

    def test_filter(self):
        # [FUNCTION DOC] Tests the built-in filter with a predicate, None and bool, and
        # lazily over an infinite count. The TypeError, copy and pickle checks are
        # commented out because dynamo raises its own exceptions or lacks support.
        self.assertEqual(list(filter(isEven, range(6))), [0,2,4])
        self.assertEqual(list(filter(None, [0,1,0,2,0])), [1,2])
        self.assertEqual(list(filter(bool, [0,1,0,2,0])), [1,2])
        self.assertEqual(take(4, filter(isEven, count())), [0,2,4,6])
        # these tests raise dynamo exceptions, not TypeError
        # self.assertRaises(TypeError, filter)
        # self.assertRaises(TypeError, filter, lambda x:x)
        # self.assertRaises(TypeError, filter, lambda x:x, range(6), 7)
        # self.assertRaises(TypeError, filter, isEven, 3)
        # dynamo raises Unsupported in this case
        # self.assertRaises(TypeError, next, filter(range(6), range(6)))

        # check copy, deepcopy, pickle
        # ans = [0,2,4]

        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.copy(c)), ans)
        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.deepcopy(c)), ans)
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans)
        #     next(c)
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans[1:])
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.pickletest(proto, c)

    def test_filterfalse(self):
        # [FUNCTION DOC] Tests filterfalse: complement of filter for predicates, None
        # and bool, laziness over infinite input, TypeError for bad arguments or non-
        # callable predicate when iterated, and pickling at all protocols.
        self.assertEqual(list(filterfalse(isEven, range(6))), [1,3,5])
        self.assertEqual(list(filterfalse(None, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(list(filterfalse(bool, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(take(4, filterfalse(isEven, count())), [1,3,5,7])
        self.assertRaises(TypeError, filterfalse)
        self.assertRaises(TypeError, filterfalse, lambda x:x)
        self.assertRaises(TypeError, filterfalse, lambda x:x, range(6), 7)
        self.assertRaises(TypeError, filterfalse, isEven, 3)
        self.assertRaises(TypeError, next, filterfalse(range(6), range(6)))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, filterfalse(isEven, range(6)))

    def test_zip(self):
        # [FUNCTION DOC] Tests the built-in zip: pairing against a helper, truncation to
        # the shortest input, laziness with an infinite count, no arguments, and tuple
        # conversion of pairs. The TypeError checks are commented out.
        # XXX This is rather silly now that builtin zip() calls zip()...
        ans = [(x,y) for x, y in zip('abc',count())]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(list(zip('abc', range(6))), lzip('abc', range(6)))
        self.assertEqual(list(zip('abcdef', range(3))), lzip('abcdef', range(3)))
        self.assertEqual(take(3,zip('abcdef', count())), lzip('abcdef', range(3)))
        self.assertEqual(list(zip('abcdef')), lzip('abcdef'))
        self.assertEqual(list(zip()), lzip())
        # self.assertRaises(TypeError, zip, 3)
        # self.assertRaises(TypeError, zip, range(3), 3)
        self.assertEqual([tuple(list(pair)) for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))
        self.assertEqual([pair for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))

    @support.impl_detail("tuple reuse is specific to CPython")
    @pickle_deprecated
    def test_zip_tuple_reuse(self):
        # [FUNCTION DOC] CPython implementation detail: zip reuses its result tuple (one
        # id when consumed, distinct ids when listed). Also checks copy, deepcopy and
        # pickle give the same pairs, including after one item consumed, and runs
        # pickletest.
        ids = list(map(id, zip('abc', 'def')))
        self.assertEqual(min(ids), max(ids))
        ids = list(map(id, list(zip('abc', 'def'))))
        self.assertEqual(len(dict.fromkeys(ids)), len(ids))

        # check copy, deepcopy, pickle
        ans = [(x,y) for x, y in copy.copy(zip('abc',count()))]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        ans = [(x,y) for x, y in copy.deepcopy(zip('abc',count()))]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            ans = [(x,y) for x, y in pickle.loads(pickle.dumps(zip('abc',count()), proto))]
            self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            testIntermediate = zip('abc',count())
            next(testIntermediate)
            ans = [(x,y) for x, y in pickle.loads(pickle.dumps(testIntermediate, proto))]
            self.assertEqual(ans, [('b', 1), ('c', 2)])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, zip('abc', count()))

    def test_ziplongest(self):
        # [FUNCTION DOC] Tests zip_longest on mixed-length inputs with default None fill
        # and a fillvalue, with and without an empty keyword mapping, laziness with
        # infinite input, zero or one arguments, and pair conversion. The TypeError
        # checks are commented out.
        for args in [
                ['abc', range(6)],
                [range(6), 'abc'],
                [range(1000), range(2000,2100), range(3000,3050)],
                [range(1000), range(0), range(3000,3050), range(1200), range(1500)],
                [range(1000), range(0), range(3000,3050), range(1200), range(1500), range(0)],
            ]:
            target = [tuple([arg[i] if i < len(arg) else None for arg in args])
                      for i in range(max(map(len, args)))]
            self.assertEqual(list(zip_longest(*args)), target)
            self.assertEqual(list(zip_longest(*args, **{})), target)
            target = [tuple((e is None and 'X' or e) for e in t) for t in target]   # Replace None fills with 'X'
            self.assertEqual(list(zip_longest(*args, **dict(fillvalue='X'))), target)

        self.assertEqual(take(3,zip_longest('abcdef', count())), list(zip('abcdef', range(3)))) # take 3 from infinite input

        self.assertEqual(list(zip_longest()), list(zip()))
        self.assertEqual(list(zip_longest([])), list(zip([])))
        self.assertEqual(list(zip_longest('abcdef')), list(zip('abcdef')))

        self.assertEqual(list(zip_longest('abc', 'defg', **{})),
                         list(zip(list('abc')+[None], 'defg'))) # empty keyword dict
        # self.assertRaises(TypeError, zip_longest, 3)
        # self.assertRaises(TypeError, zip_longest, range(3), 3)

        # for stmt in [
        #     "zip_longest('abc', fv=1)",
        #     "zip_longest('abc', fillvalue=1, bogus_keyword=None)",
        # ]:
        #     try:
        #         eval(stmt, globals(), locals())
        #     except TypeError:
        #         pass
        #     else:
        #         self.fail('Did not raise Type in:  ' + stmt)

        self.assertEqual([tuple(list(pair)) for pair in zip_longest('abc', 'def')],
                         list(zip('abc', 'def')))
        self.assertEqual([pair for pair in zip_longest('abc', 'def')],
                         list(zip('abc', 'def')))

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_zip_longest_tuple_reuse(self):
        # [FUNCTION DOC] CPython implementation detail: zip_longest reuses the result
        # tuple; one id when iterated, distinct ids when collected.
        ids = list(map(id, zip_longest('abc', 'def')))
        self.assertEqual(min(ids), max(ids))
        ids = list(map(id, list(zip_longest('abc', 'def'))))
        self.assertEqual(len(dict.fromkeys(ids)), len(ids))

    @pickle_deprecated
    def test_zip_longest_pickling(self):
        # [FUNCTION DOC] Runs the pickletest round trip for zip_longest with equal and
        # unequal lengths, a fillvalue and an empty first input at every protocol.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, zip_longest("abc", "def"))
            self.pickletest(proto, zip_longest("abc", "defgh"))
            self.pickletest(proto, zip_longest("abc", "defgh", fillvalue=1))
            self.pickletest(proto, zip_longest("", "defgh"))

    def test_zip_longest_bad_iterable(self):
        # [FUNCTION DOC] Checks that an exception raised by an input's __iter__ during
        # zip_longest construction propagates unchanged (the same TypeError object).
        exception = TypeError()

        class BadIterable:
            # [CLASS DOC] HELPER: Iterable whose __iter__ raises a captured exception
            # object.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Raises the exception stored in the enclosing
                # test.
                raise exception

        with self.assertRaises(TypeError) as cm:
            zip_longest(BadIterable())

        self.assertIs(cm.exception, exception)

    def test_bug_7244(self):
        # [FUNCTION DOC] Regression for bug 7244: zip_longest must treat StopIteration
        # from custom iterators normally (terminating, with the fill value used after
        # the shorter one ends) and must not swallow a RuntimeError raised by an
        # iterator; prints are captured.

        class Repeater:
            # [CLASS DOC] HELPER: Iterator similar to repeat: yields o up to t times
            # then raises the exception e it was given. Its iterator is itself.
            # this class is similar to itertools.repeat
            def __init__(self, o, t, e):
                # [FUNCTION DOC] HELPER: Stores the value, converts the count to int and
                # stores the exception to raise.
                self.o = o
                self.t = int(t)
                self.e = e
            def __iter__(self): # its iterator is itself
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Returns the value while the remaining count is
                # positive (decrementing it), otherwise raises the stored exception.
                if self.t > 0:
                    self.t -= 1
                    return self.o
                else:
                    raise self.e

        # Formerly this code in would fail in debug mode
        # with Undetected Error and Stop Iteration
        r1 = Repeater(1, 3, StopIteration)
        r2 = Repeater(2, 4, StopIteration)
        def run(r1, r2):
            # [FUNCTION DOC] HELPER: Collects the pairs from zip_longest of two
            # iterators with fill 0, printing each pair under captured stdout, and
            # returns the list.
            result = []
            for i, j in zip_longest(r1, r2, fillvalue=0):
                with support.captured_output('stdout'):
                    print((i, j))
                result.append((i, j))
            return result
        self.assertEqual(run(r1, r2), [(1,2), (1,2), (1,2), (0,2)])

        # Formerly, the RuntimeError would be lost
        # and StopIteration would stop as expected
        r1 = Repeater(1, 3, RuntimeError)
        r2 = Repeater(2, 4, StopIteration)
        it = zip_longest(r1, r2, fillvalue=0)
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 2))
        self.assertRaises(RuntimeError, next, it)

    def test_pairwise(self):
        # [FUNCTION DOC] Tests pairwise: empty and one-element input, sliding pairs on
        # strings and 10000 ranges, and TypeError for no arguments, extra arguments,
        # keyword use and non-iterables.
        self.assertEqual(list(pairwise('')), [])
        self.assertEqual(list(pairwise('a')), [])
        self.assertEqual(list(pairwise('ab')),
                              [('a', 'b')]),
        self.assertEqual(list(pairwise('abcde')),
                              [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        self.assertEqual(list(pairwise(range(10_000))),
                         list(zip(range(10_000), range(1, 10_000))))

        with self.assertRaises(TypeError):
            pairwise()                                      # too few arguments
        with self.assertRaises(TypeError):
            pairwise('abc', 10)                             # too many arguments
        with self.assertRaises(TypeError):
            pairwise(iterable='abc')                        # keyword arguments
        with self.assertRaises(TypeError):
            pairwise(None)                                  # non-iterable argument

    def test_pairwise_reenter(self):
        # [FUNCTION DOC] Tests that pairwise copes with its input iterator re-entering
        # the pairwise iterator itself during next, for several combinations of call
        # numbers at which the re-entry occurs; the nested results (pairs containing
        # pairs) are asserted exactly.
        def check(reenter_at, expected):
            # [FUNCTION DOC] HELPER: Builds a pairwise over an iterator that re-enters
            # it on the given call numbers and asserts the produced items equal the
            # expected list.
            class I:
                # [CLASS DOC] HELPER: Iterator counting calls to next; on call numbers
                # listed in reenter_at it returns the next item of the outer pairwise
                # iterator, otherwise a fresh single-item list holding the call count.
                count = 0
                def __iter__(self):
                    # [FUNCTION DOC] HELPER: Returns self.
                    return self
                def __next__(self):
                    # [FUNCTION DOC] HELPER: Increments the call count and either re-
                    # enters the outer iterator or returns a one-item list of the count.
                    self.count +=1
                    if self.count in reenter_at:
                        return next(it)
                    return [self.count]  # new object

            it = pairwise(I())
            for item in expected:
                self.assertEqual(next(it), item)

        check({1}, [
            (([2], [3]), [4]),
            ([4], [5]),
        ])
        check({2}, [
            ([1], ([1], [3])),
            (([1], [3]), [4]),
            ([4], [5]),
        ])
        check({3}, [
            ([1], [2]),
            ([2], ([2], [4])),
            (([2], [4]), [5]),
            ([5], [6]),
        ])
        check({1, 2}, [
            ((([3], [4]), [5]), [6]),
            ([6], [7]),
        ])
        check({1, 3}, [
            (([2], ([2], [4])), [5]),
            ([5], [6]),
        ])
        check({1, 4}, [
            (([2], [3]), (([2], [3]), [5])),
            ((([2], [3]), [5]), [6]),
            ([6], [7]),
        ])
        check({2, 3}, [
            ([1], ([1], ([1], [4]))),
            (([1], ([1], [4])), [5]),
            ([5], [6]),
        ])

    def test_pairwise_reenter2(self):
        # [FUNCTION DOC] Tests pairwise when the source re-enters it during the very
        # first call and then ends after a maximum number of items; the list of results
        # for different maximums is asserted (no pairs for up to three items).
        def check(maxcount, expected):
            # [FUNCTION DOC] HELPER: Builds a pairwise over an iterator limited to
            # maxcount calls and asserts the full list equals expected.
            class I:
                # [CLASS DOC] HELPER: Iterator that stops after maxcount items, re-
                # enters the pairwise iterator on its first call (returning its next
                # item or None) and yields one-item lists of the count afterwards.
                count = 0
                def __iter__(self):
                    # [FUNCTION DOC] HELPER: Returns self.
                    return self
                def __next__(self):
                    # [FUNCTION DOC] HELPER: Raises StopIteration past maxcount;
                    # otherwise increments the count and returns the re-entrant result
                    # on call one or a one-item list of the count.
                    if self.count >= maxcount:
                        raise StopIteration
                    self.count +=1
                    if self.count == 1:
                        return next(it, None)
                    return [self.count]  # new object

            it = pairwise(I())
            self.assertEqual(list(it), expected)

        check(1, [])
        check(2, [])
        check(3, [])
        check(4, [(([2], [3]), [4])])

    def test_product(self):
        # [FUNCTION DOC] Tests product: zero, one, many and empty iterables with and
        # without repeat, size of a large product, and agreement with two pure Python
        # implementations for 100 random combinations of argument types (strings,
        # ranges, dict, set, tuple), including when the arguments are one-shot
        # iterators.
        for args, result in [
            ([], [()]),                     # zero iterables
            (['ab'], [('a',), ('b',)]),     # one iterable
            ([range(2), range(3)], [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]),     # two iterables
            ([range(0), range(2), range(3)], []),           # first iterable with zero length
            ([range(2), range(0), range(3)], []),           # middle iterable with zero length
            ([range(2), range(3), range(0)], []),           # last iterable with zero length
            ]:
            self.assertEqual(list(product(*args)), result)
            for r in range(4):
                self.assertEqual(list(product(*(args*r))),
                                 list(product(*args, **dict(repeat=r))))
        self.assertEqual(len(list(product(*[range(7)]*6))), 7**6)

        def product1(*args, **kwds):
            # [FUNCTION DOC] HELPER: Reference cartesian product advancing indices right
            # to left; the keyword repeat multiplies the pools; empty pool gives
            # nothing, no pools gives one empty tuple.
            pools = list(map(tuple, args)) * kwds.get('repeat', 1)
            n = len(pools)
            if n == 0:
                yield ()
                return
            if any(len(pool) == 0 for pool in pools):
                return
            indices = [0] * n
            yield tuple(pool[i] for pool, i in zip(pools, indices))
            while 1:
                for i in reversed(range(n)):  # right to left
                    if indices[i] == len(pools[i]) - 1:
                        continue
                    indices[i] += 1
                    for j in range(i+1, n):
                        indices[j] = 0
                    yield tuple(pool[i] for pool, i in zip(pools, indices))
                    break
                else:
                    return

        def product2(*iterables, repeat=1):
            # [FUNCTION DOC] HELPER: Reference cartesian product from the docs, built by
            # repeatedly extending a list of partial results; raises ValueError for a
            # negative repeat.
            'Pure python version used in docs'
            if repeat < 0:
                raise ValueError('repeat argument cannot be negative')
            pools = [tuple(pool) for pool in iterables] * repeat

            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]

            for prod in result:
                yield tuple(prod)

        argtypes = ['', 'abc', '', range(0), range(4), dict(a=1, b=2, c=3),
                    set('abcdefg'), range(11), tuple(range(13))]
        for i in range(100):
            args = [choice(argtypes) for j in range(randrange(5))]
            expected_len = prod(map(len, args))
            self.assertEqual(len(list(product(*args))), expected_len)
            self.assertEqual(list(product(*args)), list(product1(*args)))
            self.assertEqual(list(product(*args)), list(product2(*args)))
            args = map(iter, args)
            self.assertEqual(len(list(product(*args))), expected_len)

    @support.bigaddrspacetest
    def test_product_overflow(self):
        # [FUNCTION DOC] Big address space only. Checks product with a large number of
        # iterables and an enormous repeat raises OverflowError or MemoryError.
        with self.assertRaises((OverflowError, MemoryError)):
            product(*(['ab']*2**5), repeat=2**25)

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_product_tuple_reuse(self):
        # [FUNCTION DOC] CPython-only check of tuple reuse for product (one id when
        # iterated, many when listed).
        self.assertEqual(len(set(map(id, product('abc', 'def')))), 1)
        self.assertNotEqual(len(set(map(id, list(product('abc', 'def'))))), 1)

    @pickle_deprecated
    def test_product_pickling(self):
        # [FUNCTION DOC] Checks copy and deepcopy of product objects give the same
        # results and that pickletest passes for every protocol, over empty, single and
        # multiple iterables.
        # check copy, deepcopy, pickle
        for args, result in [
            ([], [()]),                     # zero iterables
            (['ab'], [('a',), ('b',)]),     # one iterable
            ([range(2), range(3)], [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]),     # two iterables
            ([range(0), range(2), range(3)], []),           # first iterable with zero length
            ([range(2), range(0), range(3)], []),           # middle iterable with zero length
            ([range(2), range(3), range(0)], []),           # last iterable with zero length
            ]:
            self.assertEqual(list(copy.copy(product(*args))), result)
            self.assertEqual(list(copy.deepcopy(product(*args))), result)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, product(*args))

    @pickle_deprecated
    def test_product_issue_25021(self):
        # [FUNCTION DOC] Regression for issue 25021: restoring product state with an
        # index beyond the tuple length must clamp it (next gives the last element
        # combination), and an empty pool in the state yields immediate StopIteration.
        # test that indices are properly clamped to the length of the tuples
        p = product((1, 2),(3,))
        p.__setstate__((0, 0x1000))  # will access tuple element 1 if not clamped
        self.assertEqual(next(p), (2, 3))
        # test that empty tuple in the list will result in an immediate StopIteration
        p = product((1, 2), (), (3,))
        p.__setstate__((0, 0, 0x1000))  # will access tuple element 1 if not clamped
        self.assertRaises(StopIteration, next, p)

    @pickle_deprecated
    def test_repeat(self):
        # [FUNCTION DOC] Tests repeat (decorated for deprecation-warning runs): keyword
        # arguments, finite and infinite use, zero or negative times give empty,
        # TypeError for wrong arguments, repr with and without a count (a finished
        # repeat shows a count of zero), and copy, deepcopy and pickle round trips of a
        # partly consumed repeat.
        self.assertEqual(list(repeat(object='a', times=3)), ['a', 'a', 'a'])
        self.assertEqual(lzip(range(3),repeat('a')),
                         [(0, 'a'), (1, 'a'), (2, 'a')])
        self.assertEqual(list(repeat('a', 3)), ['a', 'a', 'a'])
        self.assertEqual(take(3, repeat('a')), ['a', 'a', 'a'])
        self.assertEqual(list(repeat('a', 0)), [])
        self.assertEqual(list(repeat('a', -3)), [])
        self.assertRaises(TypeError, repeat)
        self.assertRaises(TypeError, repeat, None, 3, 4)
        self.assertRaises(TypeError, repeat, None, 'a')
        r = repeat(1+0j)
        self.assertEqual(repr(r), 'repeat((1+0j))')
        r = repeat(1+0j, 5)
        self.assertEqual(repr(r), 'repeat((1+0j), 5)')
        list(r)
        self.assertEqual(repr(r), 'repeat((1+0j), 0)')

        # check copy, deepcopy, pickle
        c = repeat(object='a', times=10)
        self.assertEqual(next(c), 'a')
        self.assertEqual(take(2, copy.copy(c)), list('a' * 2))
        self.assertEqual(take(2, copy.deepcopy(c)), list('a' * 2))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, repeat(object='a', times=10))

    def test_repeat_with_negative_times(self):
        # [FUNCTION DOC] Checks that repr of repeat normalizes a negative times count to
        # zero for positional and keyword forms.
        self.assertEqual(repr(repeat('a', -1)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', -2)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', times=-1)), "repeat('a', 0)")
        self.assertEqual(repr(repeat('a', times=-2)), "repeat('a', 0)")

    @pickle_deprecated
    def test_map(self):
        # [FUNCTION DOC] Tests the built-in map with multiple iterables (stops at
        # shortest), infinite inputs, no iterables, TypeError for bad callables or
        # argument counts, ValueError propagation, and copy, deepcopy and pickle round
        # trips.
        self.assertEqual(list(map(operator.pow, range(3), range(1,7))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(list(map(tupleize, 'abc', range(5))),
                         [('a',0),('b',1),('c',2)])
        self.assertEqual(list(map(tupleize, 'abc', count())),
                         [('a',0),('b',1),('c',2)])
        self.assertEqual(take(2,map(tupleize, 'abc', count())),
                         [('a',0),('b',1)])
        self.assertEqual(list(map(operator.pow, [])), [])
        self.assertRaises(TypeError, map)
        self.assertRaises(TypeError, list, map(None, range(3), range(3)))
        self.assertRaises(TypeError, map, operator.neg)
        self.assertRaises(TypeError, next, map(10, range(5)))
        self.assertRaises(ValueError, next, map(errfunc, [4], [5]))
        self.assertRaises(TypeError, next, map(onearg, [4], [5]))

        # check copy, deepcopy, pickle
        ans = [('a',0),('b',1),('c',2)]

        c = map(tupleize, 'abc', count())
        self.assertEqual(list(copy.copy(c)), ans)

        c = map(tupleize, 'abc', count())
        self.assertEqual(list(copy.deepcopy(c)), ans)

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            c = map(tupleize, 'abc', count())
            self.pickletest(proto, c)

    # @pickle_deprecated
    def test_starmap(self):
        # [FUNCTION DOC] Tests starmap: unpacking argument tuples from zip, infinite
        # sources, empty input, an iterator as the argument tuple, and errors from a
        # non-callable or a raising function. Several TypeError, copy and pickle checks
        # are commented out for dynamo.
        self.assertEqual(list(starmap(operator.pow, zip(range(3), range(1,7)))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(take(3, starmap(operator.pow, zip(count(), count(1)))),
                         [0**1, 1**2, 2**3])
        self.assertEqual(list(starmap(operator.pow, [])), [])
        self.assertEqual(list(starmap(operator.pow, [iter([4,5])])), [4**5])
        # self.assertRaises(TypeError, list, starmap(operator.pow, [None]))
        # self.assertRaises(TypeError, starmap)
        # self.assertRaises(TypeError, starmap, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, starmap(10, [(4,5)]))
        self.assertRaises(ValueError, next, starmap(errfunc, [(4,5)]))
        # self.assertRaises(TypeError, next, starmap(onearg, [(4,5)]))

        # # check copy, deepcopy, pickle
        # ans = [0**1, 1**2, 2**3]
        #
        # c = starmap(operator.pow, zip(range(3), range(1,7)))
        # self.assertEqual(list(copy.copy(c)), ans)
        #
        # c = starmap(operator.pow, zip(range(3), range(1,7)))
        # self.assertEqual(list(copy.deepcopy(c)), ans)

        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = starmap(operator.pow, zip(range(3), range(1,7)))
        #       self.pickletest(proto, c)

    @pickle_deprecated
    def test_islice(self):
        # [FUNCTION DOC] Tests islice: agreement with range for many start/stop/step
        # forms, stopping when the source ends, None arguments, exact number of items
        # consumed from the source, invalid arguments (wrong count, negative, zero step,
        # non-integers) raising TypeError or ValueError, copy/deepcopy/pickle, release
        # of the source reference once exhausted (weakref check) and acceptance of
        # objects implementing index.
        for args in [          # islice(args) should agree with range(args)
                (10, 20, 3),
                (10, 3, 20),
                (10, 20),
                (10, 10),
                (10, 3),
                (20,)
                ]:
            self.assertEqual(list(islice(range(100), *args)),
                             list(range(*args)))

        for args, tgtargs in [  # Stop when seqn is exhausted
                ((10, 110, 3), ((10, 100, 3))),
                ((10, 110), ((10, 100))),
                ((110,), (100,))
                ]:
            self.assertEqual(list(islice(range(100), *args)),
                             list(range(*tgtargs)))

        # Test stop=None
        self.assertEqual(list(islice(range(10), None)), list(range(10)))
        self.assertEqual(list(islice(range(10), None, None)), list(range(10)))
        self.assertEqual(list(islice(range(10), None, None, None)), list(range(10)))
        self.assertEqual(list(islice(range(10), 2, None)), list(range(2, 10)))
        self.assertEqual(list(islice(range(10), 1, None, 2)), list(range(1, 10, 2)))

        # Test number of items consumed     SF #1171417
        it = iter(range(10))
        self.assertEqual(list(islice(it, 3)), list(range(3)))
        self.assertEqual(list(it), list(range(3, 10)))

        it = iter(range(10))
        self.assertEqual(list(islice(it, 3, 3)), [])
        self.assertEqual(list(it), list(range(3, 10)))

        # Test invalid arguments
        ra = range(10)
        self.assertRaises(TypeError, islice, ra)
        self.assertRaises(TypeError, islice, ra, 1, 2, 3, 4)
        self.assertRaises(ValueError, islice, ra, -5, 10, 1)
        self.assertRaises(ValueError, islice, ra, 1, -5, -1)
        self.assertRaises(ValueError, islice, ra, 1, 10, -1)
        self.assertRaises(ValueError, islice, ra, 1, 10, 0)
        self.assertRaises(ValueError, islice, ra, 'a')
        self.assertRaises(ValueError, islice, ra, 'a', 1)
        self.assertRaises(ValueError, islice, ra, 1, 'a')
        self.assertRaises(ValueError, islice, ra, 'a', 1, 1)
        self.assertRaises(ValueError, islice, ra, 1, 'a', 1)
        self.assertEqual(len(list(islice(count(), 1, 10, maxsize))), 1)

        # Issue #10323:  Less islice in a predictable state
        c = count()
        self.assertEqual(list(islice(c, 1, 3, 50)), [1])
        self.assertEqual(next(c), 3)

        # check copy, deepcopy, pickle
        for args in [          # islice(args) should agree with range(args)
                (10, 20, 3),
                (10, 3, 20),
                (10, 20),
                (10, 3),
                (20,)
                ]:
            self.assertEqual(list(copy.copy(islice(range(100), *args))),
                             list(range(*args)))
            self.assertEqual(list(copy.deepcopy(islice(range(100), *args))),
                             list(range(*args)))
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, islice(range(100), *args))

        # Issue #21321: check source iterator is not referenced
        # from islice() after the latter has been exhausted
        it = (x for x in (1, 2))
        wr = weakref.ref(it)
        it = islice(it, 1)
        self.assertIsNotNone(wr())
        list(it) # exhaust the iterator
        support.gc_collect()
        self.assertIsNone(wr())

        # Issue #30537: islice can accept integer-like objects as
        # arguments
        class IntLike(object):
            # [CLASS DOC] HELPER: Wrapper around an int exposing it through the integer-
            # conversion protocol, used to verify islice accepts integer-like arguments.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the wrapped integer.
                self.val = val
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the wrapped integer for integer
                # conversion.
                return self.val
        self.assertEqual(list(islice(range(100), IntLike(10))), list(range(10)))
        self.assertEqual(list(islice(range(100), IntLike(10), IntLike(50))),
                         list(range(10, 50)))
        self.assertEqual(list(islice(range(100), IntLike(10), IntLike(50), IntLike(5))),
                         list(range(10,50,5)))

    # @pickle_deprecated
    def test_takewhile(self):
        # [FUNCTION DOC] Tests takewhile: stops at the first false predicate result,
        # empty input, TypeError for bad arguments or non-callable predicate, exhaustion
        # afterward. The ValueError, copy and pickle checks are commented out.
        data = [1, 3, 5, 20, 2, 4, 6, 8]
        self.assertEqual(list(takewhile(underten, data)), [1, 3, 5])
        self.assertEqual(list(takewhile(underten, [])), [])
        self.assertRaises(TypeError, takewhile)
        self.assertRaises(TypeError, takewhile, operator.pow)
        self.assertRaises(TypeError, takewhile, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, takewhile(10, [(4,5)]))
        # self.assertRaises(ValueError, next, takewhile(errfunc, [(4,5)]))
        t = takewhile(bool, [1, 1, 1, 0, 0, 0])
        self.assertEqual(list(t), [1, 1, 1])
        self.assertRaises(StopIteration, next, t)

        # check copy, deepcopy, pickle
        # self.assertEqual(list(copy.copy(takewhile(underten, data))), [1, 3, 5])
        # self.assertEqual(list(copy.deepcopy(takewhile(underten, data))),
        #                 [1, 3, 5])
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     self.pickletest(proto, takewhile(underten, data))

    # @pickle_deprecated
    def test_dropwhile(self):
        # [FUNCTION DOC] Tests dropwhile: skips items while the predicate is true then
        # yields everything, empty input, TypeError for bad arguments or a non-callable
        # predicate. The remaining error, copy and pickle checks are commented out.
        data = [1, 3, 5, 20, 2, 4, 6, 8]
        self.assertEqual(list(dropwhile(underten, data)), [20, 2, 4, 6, 8])
        self.assertEqual(list(dropwhile(underten, [])), [])
        self.assertRaises(TypeError, dropwhile)
        self.assertRaises(TypeError, dropwhile, operator.pow)
        self.assertRaises(TypeError, dropwhile, operator.pow, [(4,5)], 'extra')
        self.assertRaises(TypeError, next, dropwhile(10, [(4,5)]))
        #self.assertRaises(ValueError, next, dropwhile(errfunc, [(4,5)]))
        #
        ## check copy, deepcopy, pickle
        #self.assertEqual(list(copy.copy(dropwhile(underten, data))), [20, 2, 4, 6, 8])
        #self.assertEqual(list(copy.deepcopy(dropwhile(underten, data))),
        #                [20, 2, 4, 6, 8])
        #for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #    self.pickletest(proto, dropwhile(underten, data))

    @pickle_deprecated
    def test_tee(self):
        # [FUNCTION DOC] Tests tee: empty input, fully and non-interleaved consumption,
        # deallocating the leading or trailing copy, random interleaving, argument
        # validation, constructing a tee object from an iterable, multi-way lagged
        # splits, valid and invalid n, independence of tee objects, tee_new, weak
        # reference support, and copy, partially consumed copy, deepcopy and pickle of
        # tee objects.
        n = 200

        a, b = tee([])        # test empty iterator
        self.assertEqual(list(a), [])
        self.assertEqual(list(b), [])

        a, b = tee(irange(n)) # test 100% interleaved
        self.assertEqual(lzip(a,b), lzip(range(n), range(n)))

        a, b = tee(irange(n)) # test 0% interleaved
        self.assertEqual(list(a), list(range(n)))
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of leading iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del a
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of trailing iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del b
        self.assertEqual(list(a), list(range(100, n)))

        for j in range(5):   # test randomly interleaved
            order = [0]*n + [1]*n
            random.shuffle(order)
            lists = ([], [])
            its = tee(irange(n))
            for i in order:
                value = next(its[i])
                lists[i].append(value)
            self.assertEqual(lists[0], list(range(n)))
            self.assertEqual(lists[1], list(range(n)))

        # test argument format checking
        self.assertRaises(TypeError, tee)
        self.assertRaises(TypeError, tee, 3)
        self.assertRaises(TypeError, tee, [1,2], 'x')
        self.assertRaises(TypeError, tee, [1,2], 3, 'x')

        # tee object should be instantiable
        a, b = tee('abc')
        c = type(a)('def')
        self.assertEqual(list(c), list('def'))

        # test long-lagged and multi-way split
        a, b, c = tee(range(2000), 3)
        for i in range(100):
            self.assertEqual(next(a), i)
        self.assertEqual(list(b), list(range(2000)))
        self.assertEqual([next(c), next(c)], list(range(2)))
        self.assertEqual(list(a), list(range(100,2000)))
        self.assertEqual(list(c), list(range(2,2000)))

        # test values of n
        self.assertRaises(TypeError, tee, 'abc', 'invalid')
        self.assertRaises(ValueError, tee, [], -1)
        for n in range(5):
            result = tee('abc', n)
            self.assertEqual(type(result), tuple)
            self.assertEqual(len(result), n)
            self.assertEqual([list(x) for x in result], [list('abc')]*n)

        # tee objects are independent (see bug gh-123884)
        a, b = tee('abc')
        c, d = tee(a)
        e, f = tee(c)
        self.assertTrue(len({a, b, c, d, e, f}) == 6)

        # test tee_new
        t1, t2 = tee('abc')
        tnew = type(t1)
        self.assertRaises(TypeError, tnew)
        self.assertRaises(TypeError, tnew, 10)
        t3 = tnew(t1)
        self.assertTrue(list(t1) == list(t2) == list(t3) == list('abc'))

        # test that tee objects are weak referenceable
        a, b = tee(range(10))
        p = weakref.proxy(a)
        self.assertEqual(getattr(p, '__class__'), type(b))
        del a
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, '__class__')

        ans = list('abc')
        long_ans = list(range(10000))

        # check copy
        a, b = tee('abc')
        self.assertEqual(list(copy.copy(a)), ans)
        self.assertEqual(list(copy.copy(b)), ans)
        a, b = tee(list(range(10000)))
        self.assertEqual(list(copy.copy(a)), long_ans)
        self.assertEqual(list(copy.copy(b)), long_ans)

        # check partially consumed copy
        a, b = tee('abc')
        take(2, a)
        take(1, b)
        self.assertEqual(list(copy.copy(a)), ans[2:])
        self.assertEqual(list(copy.copy(b)), ans[1:])
        self.assertEqual(list(a), ans[2:])
        self.assertEqual(list(b), ans[1:])
        a, b = tee(range(10000))
        take(100, a)
        take(60, b)
        self.assertEqual(list(copy.copy(a)), long_ans[100:])
        self.assertEqual(list(copy.copy(b)), long_ans[60:])
        self.assertEqual(list(a), long_ans[100:])
        self.assertEqual(list(b), long_ans[60:])

        # check deepcopy
        a, b = tee('abc')
        self.assertEqual(list(copy.deepcopy(a)), ans)
        self.assertEqual(list(copy.deepcopy(b)), ans)
        self.assertEqual(list(a), ans)
        self.assertEqual(list(b), ans)
        a, b = tee(range(10000))
        self.assertEqual(list(copy.deepcopy(a)), long_ans)
        self.assertEqual(list(copy.deepcopy(b)), long_ans)
        self.assertEqual(list(a), long_ans)
        self.assertEqual(list(b), long_ans)

        # check partially consumed deepcopy
        a, b = tee('abc')
        take(2, a)
        take(1, b)
        self.assertEqual(list(copy.deepcopy(a)), ans[2:])
        self.assertEqual(list(copy.deepcopy(b)), ans[1:])
        self.assertEqual(list(a), ans[2:])
        self.assertEqual(list(b), ans[1:])
        a, b = tee(range(10000))
        take(100, a)
        take(60, b)
        self.assertEqual(list(copy.deepcopy(a)), long_ans[100:])
        self.assertEqual(list(copy.deepcopy(b)), long_ans[60:])
        self.assertEqual(list(a), long_ans[100:])
        self.assertEqual(list(b), long_ans[60:])

        # check pickle
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, iter(tee('abc')))
            a, b = tee('abc')
            self.pickletest(proto, a, compare=ans)
            self.pickletest(proto, b, compare=ans)

    def test_tee_dealloc_segfault(self):
        # [FUNCTION DOC] Regression for gh-115874: runs a subprocess that stores a tee
        # object in a module at exit to check interpreter shutdown does not segfault;
        # asserts the subprocess exits cleanly.
        # gh-115874: segfaults when accessing module state in tp_dealloc.
        script = (
            "import typing, copyreg, itertools; "
            "copyreg.buggy_tee = itertools.tee(())"
        )
        script_helper.assert_python_ok("-c", script)

    # Issue 13454: Crash when deleting backward iterator from tee()
    @skipIfTorchDynamo("infinite loop in torch dynamo")
    def test_tee_del_backward(self):
        # [FUNCTION DOC] Regression for issue 13454: exhausts the forward tee of a very
        # long repeat and then deletes the backward one (must not crash). Skipped under
        # dynamo because of an infinite loop; cleans up both names if an exception
        # occurs and re-raises.
        forward, backward = tee(repeat(None, 20000000))
        try:
            any(forward)  # exhaust the iterator
            del backward
        except:
            del forward, backward
            raise

    def test_tee_reenter(self):
        # [FUNCTION DOC] Checks that a source iterator whose first next call asks for an
        # item from the sibling tee object makes next raise RuntimeError mentioning tee.
        class I:
            # [CLASS DOC] HELPER: Iterator whose first next call fetches from the second
            # tee object (re-entering tee) and later calls return nothing.
            first = True
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: On the first call returns next of the sibling
                # tee (triggering re-entry); afterwards returns None.
                first = self.first
                self.first = False
                if first:
                    return next(b)

        a, b = tee(I())
        with self.assertRaisesRegex(RuntimeError, "tee"):
            next(a)

    @threading_helper.requires_working_threading()
    def test_tee_concurrent(self):
        # [FUNCTION DOC] Needs working threading. Starts a thread that blocks inside the
        # source iterator while holding one tee, then asserts that using the other tee
        # from the main thread raises RuntimeError mentioning tee; always releases the
        # blocked thread and joins it.
        start = threading.Event()
        finish = threading.Event()
        class I:
            # [CLASS DOC] HELPER: Iterator whose next sets the start event and blocks
            # until the finish event is set.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Signals the start event, then waits on the
                # finish event, returning None.
                start.set()
                finish.wait()

        a, b = tee(I())
        thread = threading.Thread(target=next, args=[a])
        thread.start()
        try:
            start.wait()
            with self.assertRaisesRegex(RuntimeError, "tee"):
                next(b)
        finally:
            finish.set()
            thread.join()

    def test_StopIteration(self):
        # [FUNCTION DOC] Checks that next on a fresh empty iterator raises StopIteration
        # for every itertools type and the filter-like built-ins, including when the
        # source is the always-empty StopNow helper and for tee, islice and a zero-count
        # repeat.
        self.assertRaises(StopIteration, next, zip())

        for f in (chain, cycle, zip, groupby):
            self.assertRaises(StopIteration, next, f([]))
            self.assertRaises(StopIteration, next, f(StopNow()))

        self.assertRaises(StopIteration, next, islice([], None))
        self.assertRaises(StopIteration, next, islice(StopNow(), None))

        p, q = tee([])
        self.assertRaises(StopIteration, next, p)
        self.assertRaises(StopIteration, next, q)
        p, q = tee(StopNow())
        self.assertRaises(StopIteration, next, p)
        self.assertRaises(StopIteration, next, q)

        self.assertRaises(StopIteration, next, repeat(None, 0))

        for f in (filter, filterfalse, map, takewhile, dropwhile, starmap):
            self.assertRaises(StopIteration, next, f(lambda x:x, []))
            self.assertRaises(StopIteration, next, f(lambda x:x, StopNow()))

    @support.cpython_only
    def test_combinations_result_gc(self):
        # [FUNCTION DOC] CPython-only regression for bpo-42536: after garbage collection
        # untracked the reused result tuple of combinations, the tuple returned by the
        # next step must be tracked again.
        # bpo-42536: combinations's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = combinations([None, []], 1)
        next(it)
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which has the value (None,). Make sure it's re-tracked when
        # it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_combinations_with_replacement_result_gc(self):
        # [FUNCTION DOC] CPython-only: same GC re-tracking check for the result tuple of
        # combinations_with_replacement.
        # Ditto for combinations_with_replacement.
        it = combinations_with_replacement([None, []], 1)
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_permutations_result_gc(self):
        # [FUNCTION DOC] CPython-only: same GC re-tracking check for permutations.
        # Ditto for permutations.
        it = permutations([None, []], 1)
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_product_result_gc(self):
        # [FUNCTION DOC] CPython-only: same GC re-tracking check for product.
        # Ditto for product.
        it = product([None, []])
        next(it)
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_zip_longest_result_gc(self):
        # [FUNCTION DOC] CPython-only: same GC re-tracking check for zip_longest.
        # Ditto for zip_longest.
        it = zip_longest([[]])
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_pairwise_result_gc(self):
        # [FUNCTION DOC] CPython-only: same GC re-tracking check for pairwise.
        # Ditto for pairwise.
        it = pairwise([None, None])
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_immutable_types(self):
        # [FUNCTION DOC] CPython-only. For each itertools type (including the private
        # grouper and tee helper types) checks that assigning a new attribute on the
        # type raises TypeError mentioning immutable; reported per type via subTest.
        from itertools import _grouper, _tee, _tee_dataobject
        dataset = (
            accumulate,
            batched,
            chain,
            combinations,
            combinations_with_replacement,
            compress,
            count,
            cycle,
            dropwhile,
            filterfalse,
            groupby,
            _grouper,
            islice,
            pairwise,
            permutations,
            product,
            repeat,
            starmap,
            takewhile,
            _tee,
            _tee_dataobject,
            zip_longest,
        )
        for tp in dataset:
            with self.subTest(tp=tp):
                with self.assertRaisesRegex(TypeError, "immutable"):
                    tp.foobar = 1


class TestExamples(__TestCase):
    # [CLASS DOC] Tests that the examples in the itertools documentation produce the
    # documented results, plus pickle and copy checks for accumulate. No state.

    def test_accumulate(self):
        # [FUNCTION DOC] Checks accumulate of 1 to 5 gives running sums.
        self.assertEqual(list(accumulate([1,2,3,4,5])), [1, 3, 6, 10, 15])

    @pickle_deprecated
    def test_accumulate_reducible(self):
        # [FUNCTION DOC] Checks accumulate can be copied, deep-copied and pickled at
        # every protocol, both fresh and after one item has been consumed.
        # check copy, deepcopy, pickle
        data = [1, 2, 3, 4, 5]
        accumulated = [1, 3, 6, 10, 15]

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it = accumulate(data)
            self.assertEqual(list(pickle.loads(pickle.dumps(it, proto))), accumulated[:])
            self.assertEqual(next(it), 1)
            self.assertEqual(list(pickle.loads(pickle.dumps(it, proto))), accumulated[1:])
        it = accumulate(data)
        self.assertEqual(next(it), 1)
        self.assertEqual(list(copy.deepcopy(it)), accumulated[1:])
        self.assertEqual(list(copy.copy(it)), accumulated[1:])

    @pickle_deprecated
    def test_accumulate_reducible_none(self):
        # [FUNCTION DOC] Regression for issue 25718: accumulate whose running total is
        # None (with the identity-test operator) survives pickling, copy and deepcopy
        # after the first item, yielding true then false.
        # Issue #25718: total is None
        it = accumulate([None, None, None], operator.is_)
        self.assertEqual(next(it), None)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it_copy = pickle.loads(pickle.dumps(it, proto))
            self.assertEqual(list(it_copy), [True, False])
        self.assertEqual(list(copy.deepcopy(it)), [True, False])
        self.assertEqual(list(copy.copy(it)), [True, False])

    def test_chain(self):
        # [FUNCTION DOC] Checks chain of two strings joins to one string.
        self.assertEqual(''.join(chain('ABC', 'DEF')), 'ABCDEF')

    def test_chain_from_iterable(self):
        # [FUNCTION DOC] Checks chain.from_iterable of a list of two strings.
        self.assertEqual(''.join(chain.from_iterable(['ABC', 'DEF'])), 'ABCDEF')

    def test_combinations(self):
        # [FUNCTION DOC] Checks the documented combinations outputs for four letters
        # taken two at a time and for range(4) taken three at a time.
        self.assertEqual(list(combinations('ABCD', 2)),
                         [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])
        self.assertEqual(list(combinations(range(4), 3)),
                         [(0,1,2), (0,1,3), (0,2,3), (1,2,3)])

    def test_combinations_with_replacement(self):
        # [FUNCTION DOC] Checks the documented combinations_with_replacement output for
        # three letters, two at a time.
        self.assertEqual(list(combinations_with_replacement('ABC', 2)),
                         [('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])

    def test_compress(self):
        # [FUNCTION DOC] Checks the documented compress example selects letters A, C, E,
        # F.
        self.assertEqual(list(compress('ABCDEF', [1,0,1,0,1,1])), list('ACEF'))

    def test_count(self):
        # [FUNCTION DOC] Checks count starting at 10 sliced to five items.
        self.assertEqual(list(islice(count(10), 5)), [10, 11, 12, 13, 14])

    def test_cycle(self):
        # [FUNCTION DOC] Checks cycle over four letters sliced to twelve items.
        self.assertEqual(list(islice(cycle('ABCD'), 12)), list('ABCDABCDABCD'))

    def test_dropwhile(self):
        # [FUNCTION DOC] Checks dropwhile with a less-than-five predicate leaves the
        # tail starting at 6.
        self.assertEqual(list(dropwhile(lambda x: x<5, [1,4,6,4,1])), [6,4,1])

    def test_groupby(self):
        # [FUNCTION DOC] Checks groupby keys and group contents for runs of repeated
        # letters.
        self.assertEqual([k for k, g in groupby('AAAABBBCCDAABBB')],
                         list('ABCDAB'))
        self.assertEqual([(list(g)) for k, g in groupby('AAAABBBCCD')],
                         [list('AAAA'), list('BBB'), list('CC'), list('D')])

    def test_filter(self):
        # [FUNCTION DOC] Checks filter keeps the odd numbers below ten.
        self.assertEqual(list(filter(lambda x: x%2, range(10))), [1,3,5,7,9])

    def test_filterfalse(self):
        # [FUNCTION DOC] Checks filterfalse keeps the even numbers below ten.
        self.assertEqual(list(filterfalse(lambda x: x%2, range(10))), [0,2,4,6,8])

    def test_map(self):
        # [FUNCTION DOC] Checks map with pow over two tuples.
        self.assertEqual(list(map(pow, (2,3,10), (5,2,3))), [32, 9, 1000])

    def test_islice(self):
        # [FUNCTION DOC] Checks islice with stop only, start and stop, open stop, and a
        # step of two.
        self.assertEqual(list(islice('ABCDEFG', 2)), list('AB'))
        self.assertEqual(list(islice('ABCDEFG', 2, 4)), list('CD'))
        self.assertEqual(list(islice('ABCDEFG', 2, None)), list('CDEFG'))
        self.assertEqual(list(islice('ABCDEFG', 0, None, 2)), list('ACEG'))

    def test_zip(self):
        # [FUNCTION DOC] Checks zip truncates to the shorter of two inputs.
        self.assertEqual(list(zip('ABCD', 'xy')), [('A', 'x'), ('B', 'y')])

    def test_zip_longest(self):
        # [FUNCTION DOC] Checks zip_longest pads the shorter input with the given fill
        # value.
        self.assertEqual(list(zip_longest('ABCD', 'xy', fillvalue='-')),
                         [('A', 'x'), ('B', 'y'), ('C', '-'), ('D', '-')])

    def test_permutations(self):
        # [FUNCTION DOC] Checks the documented permutations for letters (r=2) and for
        # range(3) with default r.
        self.assertEqual(list(permutations('ABCD', 2)),
                         list(map(tuple, 'AB AC AD BA BC BD CA CB CD DA DB DC'.split())))
        self.assertEqual(list(permutations(range(3))),
                         [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)])

    def test_product(self):
        # [FUNCTION DOC] Checks the documented product examples: two strings, and
        # range(2) with repeat 3.
        self.assertEqual(list(product('ABCD', 'xy')),
                         list(map(tuple, 'Ax Ay Bx By Cx Cy Dx Dy'.split())))
        self.assertEqual(list(product(range(2), repeat=3)),
                        [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                         (1,0,0), (1,0,1), (1,1,0), (1,1,1)])

    def test_repeat(self):
        # [FUNCTION DOC] Checks repeat of 10 three times.
        self.assertEqual(list(repeat(10, 3)), [10, 10, 10])

    def test_stapmap(self):
        # [FUNCTION DOC] Checks starmap with pow over argument tuples. (Name is
        # misspelled as stapmap in the source.)
        self.assertEqual(list(starmap(pow, [(2,5), (3,2), (10,3)])),
                         [32, 9, 1000])

    def test_takewhile(self):
        # [FUNCTION DOC] Checks takewhile with a less-than-five predicate.
        self.assertEqual(list(takewhile(lambda x: x<5, [1,4,6,4,1])), [1,4])


class TestPurePythonRoughEquivalents(__TestCase):
    # [CLASS DOC] Tests that pure Python recipes from the documentation (batched,
    # islice, tee) behave like the corresponding C implementations. Holds no state.

    def test_batched_recipe(self):
        # [FUNCTION DOC] Compares itertools.batched with a pure Python recipe over many
        # strings, None, and batch sizes including invalid ones (negative, zero, None);
        # for each pair both must either give equal lists or raise the same exception
        # type. Reported per case via subTest.
        def batched_recipe(iterable, n):
            # [FUNCTION DOC] HELPER: Pure Python batched: raises ValueError if n is
            # below one, otherwise yields tuples of up to n items taken from an iterator
            # until empty.
            "Batch data into tuples of length n. The last batch may be shorter."
            # batched('ABCDEFG', 3) --> ABC DEF G
            if n < 1:
                raise ValueError('n must be at least one')
            it = iter(iterable)
            while batch := tuple(islice(it, n)):
                yield batch

        for iterable, n in product(
                ['', 'a', 'ab', 'abc', 'abcd', 'abcde', 'abcdef', 'abcdefg', None],
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None]):
            with self.subTest(iterable=iterable, n=n):
                try:
                    e1, r1 = None, list(batched(iterable, n))
                except Exception as e:
                    e1, r1 = type(e), None
                try:
                    e2, r2 = None, list(batched_recipe(iterable, n))
                except Exception as e:
                    e2, r2 = type(e), None
                self.assertEqual(r1, r2)
                self.assertEqual(e1, e2)

    @staticmethod
    def islice(iterable, *args):
        # [FUNCTION DOC] HELPER: Static pure Python islice recipe taking an iterable and
        # slice arguments (start, stop, step with None defaulting to 0, a huge maximum
        # and 1). Yields the selected elements, and consumes the source up to the start
        # or stop position afterwards so the source ends in the same state as the real
        # islice.
        s = slice(*args)
        start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
        it = iter(range(start, stop, step))
        try:
            nexti = next(it)
        except StopIteration:
            # Consume *iterable* up to the *start* position.
            for i, element in zip(range(start), iterable):
                pass
            return
        try:
            for i, element in enumerate(iterable):
                if i == nexti:
                    yield element
                    nexti = next(it)
        except StopIteration:
            # Consume to *stop*.
            for i, element in zip(range(i + 1, stop), iterable):
                pass

    @skipIfTorchDynamo("infinite loop in torch dynamo")
    def test_islice_recipe(self):
        # [FUNCTION DOC] Skipped under dynamo (infinite loop). Checks the pure Python
        # islice recipe on stop-only, start-stop, open-ended and stepped slices, the
        # number of items consumed from the source, and that a stepped slice leaves the
        # source in a predictable position.
        self.assertEqual(list(self.islice('ABCDEFG', 2)), list('AB'))
        self.assertEqual(list(self.islice('ABCDEFG', 2, 4)), list('CD'))
        self.assertEqual(list(self.islice('ABCDEFG', 2, None)), list('CDEFG'))
        self.assertEqual(list(self.islice('ABCDEFG', 0, None, 2)), list('ACEG'))
        # Test items consumed.
        it = iter(range(10))
        self.assertEqual(list(self.islice(it, 3)), list(range(3)))
        self.assertEqual(list(it), list(range(3, 10)))
        it = iter(range(10))
        self.assertEqual(list(self.islice(it, 3, 3)), [])
        self.assertEqual(list(it), list(range(3, 10)))
        # Test that slice finishes in predictable state.
        c = count()
        self.assertEqual(list(self.islice(c, 1, 3, 50)), [1])
        self.assertEqual(next(c), 3)


    def test_tee_recipe(self):
        # [FUNCTION DOC] Runs the tee behavior checks (interleaving, dealloc of
        # leading/trailing copy, random interleaving, argument checks, multi-way split,
        # independence, weak references, and the long repeat exhaustion case) against a
        # pure Python tee recipe defined inline. Copy tests that do not apply to the
        # recipe are disabled behind a false condition.

        # Begin tee() recipe ###########################################

        def tee(iterable, n=2):
            # [FUNCTION DOC] HELPER: Recipe version of tee: raises ValueError for
            # negative n, returns an empty tuple for zero, otherwise returns n linked
            # recipe iterators built from the source.
            if n < 0:
                raise ValueError
            if n == 0:
                return ()
            iterator = _tee(iterable)
            result = [iterator]
            for _ in range(n - 1):
                result.append(_tee(iterator))
            return tuple(result)

        class _tee:
            # [CLASS DOC] HELPER: Pure Python tee iterator. Holds the shared source
            # iterator and a two-slot link cell (value, next link) forming a singly
            # linked chain so several copies read the same data independently.
            # Constructing from another copy shares its iterator and current link.

            def __init__(self, iterable):
                # [FUNCTION DOC] HELPER: Builds an iterator from the argument; if it is
                # already a recipe tee, shares its iterator and link, otherwise starts a
                # fresh empty link cell.
                it = iter(iterable)
                if isinstance(it, _tee):
                    self.iterator = it.iterator
                    self.link = it.link
                else:
                    self.iterator = it
                    self.link = [None, None]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Returns the value stored in the current link,
                # first fetching it from the source (raising StopIteration at the end)
                # and creating the next link if needed, then advances to the next link.
                link = self.link
                if link[1] is None:
                    link[0] = next(self.iterator)
                    link[1] = [None, None]
                value, self.link = link
                return value

        # End tee() recipe #############################################

        n = 200

        a, b = tee([])        # test empty iterator
        self.assertEqual(list(a), [])
        self.assertEqual(list(b), [])

        a, b = tee(irange(n)) # test 100% interleaved
        self.assertEqual(lzip(a,b), lzip(range(n), range(n)))

        a, b = tee(irange(n)) # test 0% interleaved
        self.assertEqual(list(a), list(range(n)))
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of leading iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del a
        self.assertEqual(list(b), list(range(n)))

        a, b = tee(irange(n)) # test dealloc of trailing iterator
        for i in range(100):
            self.assertEqual(next(a), i)
        del b
        self.assertEqual(list(a), list(range(100, n)))

        for j in range(5):   # test randomly interleaved
            order = [0]*n + [1]*n
            random.shuffle(order)
            lists = ([], [])
            its = tee(irange(n))
            for i in order:
                value = next(its[i])
                lists[i].append(value)
            self.assertEqual(lists[0], list(range(n)))
            self.assertEqual(lists[1], list(range(n)))

        # test argument format checking
        self.assertRaises(TypeError, tee)
        self.assertRaises(TypeError, tee, 3)
        self.assertRaises(TypeError, tee, [1,2], 'x')
        self.assertRaises(TypeError, tee, [1,2], 3, 'x')

        # tee object should be instantiable
        a, b = tee('abc')
        c = type(a)('def')
        self.assertEqual(list(c), list('def'))

        # test long-lagged and multi-way split
        a, b, c = tee(range(2000), 3)
        for i in range(100):
            self.assertEqual(next(a), i)
        self.assertEqual(list(b), list(range(2000)))
        self.assertEqual([next(c), next(c)], list(range(2)))
        self.assertEqual(list(a), list(range(100,2000)))
        self.assertEqual(list(c), list(range(2,2000)))

        # test invalid values of n
        self.assertRaises(TypeError, tee, 'abc', 'invalid')
        self.assertRaises(ValueError, tee, [], -1)

        for n in range(5):
            result = tee('abc', n)
            self.assertEqual(type(result), tuple)
            self.assertEqual(len(result), n)
            self.assertEqual([list(x) for x in result], [list('abc')]*n)

        # tee objects are independent (see bug gh-123884)
        a, b = tee('abc')
        c, d = tee(a)
        e, f = tee(c)
        self.assertTrue(len({a, b, c, d, e, f}) == 6)

        # test tee_new
        t1, t2 = tee('abc')
        tnew = type(t1)
        self.assertRaises(TypeError, tnew)
        self.assertRaises(TypeError, tnew, 10)
        t3 = tnew(t1)
        self.assertTrue(list(t1) == list(t2) == list(t3) == list('abc'))

        # test that tee objects are weak referenceable
        a, b = tee(range(10))
        p = weakref.proxy(a)
        self.assertEqual(getattr(p, '__class__'), type(b))
        del a
        gc.collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, '__class__')

        ans = list('abc')
        long_ans = list(range(10000))

        # Tests not applicable to the tee() recipe
        if False:
            # check copy
            a, b = tee('abc')
            self.assertEqual(list(copy.copy(a)), ans)
            self.assertEqual(list(copy.copy(b)), ans)
            a, b = tee(list(range(10000)))
            self.assertEqual(list(copy.copy(a)), long_ans)
            self.assertEqual(list(copy.copy(b)), long_ans)

            # check partially consumed copy
            a, b = tee('abc')
            take(2, a)
            take(1, b)
            self.assertEqual(list(copy.copy(a)), ans[2:])
            self.assertEqual(list(copy.copy(b)), ans[1:])
            self.assertEqual(list(a), ans[2:])
            self.assertEqual(list(b), ans[1:])
            a, b = tee(range(10000))
            take(100, a)
            take(60, b)
            self.assertEqual(list(copy.copy(a)), long_ans[100:])
            self.assertEqual(list(copy.copy(b)), long_ans[60:])
            self.assertEqual(list(a), long_ans[100:])
            self.assertEqual(list(b), long_ans[60:])

        # Issue 13454: Crash when deleting backward iterator from tee()
        forward, backward = tee(repeat(None, 2000)) # 20000000
        try:
            any(forward)  # exhaust the iterator
            del backward
        except:
            del forward, backward
            raise


class TestGC(__TestCase):
    # [CLASS DOC] Checks that itertools iterators participate in cycle garbage
    # collection: each test builds a reference cycle through an iterator and a container
    # and relies on the test harness cycle detection. No state.

    def makecycle(self, iterator, container):
        # [FUNCTION DOC] HELPER: Appends the iterator to the container, advances the
        # iterator once, then deletes local names; used to create a reference cycle that
        # the garbage collector must reclaim.
        container.append(iterator)
        next(iterator)
        del container, iterator

    def test_accumulate(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through accumulate over a
        # list containing the list a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(accumulate([1,2,a,3]), a)

    def test_batched(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through batched (size 2) over
        # a list containing a using makecycle, which must be collectable by the garbage
        # collector.
        a = []
        self.makecycle(batched([1,2,a,3], 2), a)

    def test_chain(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through chain of a itself
        # using makecycle, which must be collectable by the garbage collector.
        a = []
        self.makecycle(chain(a), a)

    def test_chain_from_iterable(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through chain.from_iterable
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(chain.from_iterable([a]), a)

    def test_combinations(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through combinations (r=3)
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(combinations([1,2,a,3], 3), a)

    def test_combinations_with_replacement(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through
        # combinations_with_replacement (r=3) over a list containing a using makecycle,
        # which must be collectable by the garbage collector.
        a = []
        self.makecycle(combinations_with_replacement([1,2,a,3], 3), a)

    def test_compress(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through compress over letters
        # with a selector pattern, data from strings using makecycle, which must be
        # collectable by the garbage collector.
        a = []
        self.makecycle(compress('ABCDEF', [1,0,1,0,1,0]), a)

    def test_count(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through count using an int
        # subclass instance that has a class attribute referencing a, for both start and
        # step using makecycle, which must be collectable by the garbage collector.
        a = []
        Int = type('Int', (int,), dict(x=a))
        self.makecycle(count(Int(0), Int(1)), a)

    def test_cycle(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through cycle over a list
        # holding a twice using makecycle, which must be collectable by the garbage
        # collector.
        a = []
        self.makecycle(cycle([a]*2), a)

    def test_dropwhile(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through dropwhile with bool
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(dropwhile(bool, [0, a, a]), a)

    def test_groupby(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through groupby over a list
        # holding a twice with an identity key using makecycle, which must be
        # collectable by the garbage collector.
        a = []
        self.makecycle(groupby([a]*2, lambda x:x), a)

    def test_issue2246(self):
        # [FUNCTION DOC] Regression for issue 2246: iterates groupby with a key function
        # whose attribute dictionary collects the group objects, creating a cycle
        # through the grouper iterator that must be collectable.
        # Issue 2246 -- the _grouper iterator was not included in GC
        n = 10
        keyfunc = lambda x: x
        for i, j in groupby(range(n), key=keyfunc):
            keyfunc.__dict__.setdefault('x',[]).append(j)

    def test_filter(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through filter with an always
        # true predicate over a list holding a twice using makecycle, which must be
        # collectable by the garbage collector.
        a = []
        self.makecycle(filter(lambda x:True, [a]*2), a)

    def test_filterfalse(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through filterfalse over a
        # itself with an always false predicate using makecycle, which must be
        # collectable by the garbage collector.
        a = []
        self.makecycle(filterfalse(lambda x:False, a), a)

    def test_zip(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through zip of lists holding
        # a using makecycle, which must be collectable by the garbage collector.
        a = []
        self.makecycle(zip([a]*2, [a]*3), a)

    def test_zip_longest(self):
        # [FUNCTION DOC] GC test: zip_longest cycles through lists holding a, both with
        # the default fill and with a fill value list that holds a.
        a = []
        self.makecycle(zip_longest([a]*2, [a]*3), a)
        b = [a, None]
        self.makecycle(zip_longest([a]*2, [a]*3, fillvalue=b), a)

    def test_map(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through map with identity
        # over a list holding a twice using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(map(lambda x:x, [a]*2), a)

    def test_islice(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through islice of a list
        # holding a with stop None using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(islice([a]*2, None), a)

    def test_pairwise(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through pairwise over a list
        # holding a five times using makecycle, which must be collectable by the garbage
        # collector.
        a = []
        self.makecycle(pairwise([a]*5), a)

    def test_permutations(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through permutations (r=3)
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(permutations([1,2,a,3], 3), a)

    def test_product(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through product (repeat=3)
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(product([1,2,a,3], repeat=3), a)

    def test_repeat(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through repeat of a using
        # makecycle, which must be collectable by the garbage collector.
        a = []
        self.makecycle(repeat(a), a)

    def test_starmap(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through starmap with a tuple-
        # returning function over argument tuples holding a using makecycle, which must
        # be collectable by the garbage collector.
        a = []
        self.makecycle(starmap(lambda *t: t, [(a,a)]*2), a)

    def test_takewhile(self):
        # [FUNCTION DOC] GC test: builds a reference cycle through takewhile with bool
        # over a list containing a using makecycle, which must be collectable by the
        # garbage collector.
        a = []
        self.makecycle(takewhile(bool, [1, 0, a, a]), a)

def R(seqn):
    # [FUNCTION DOC] HELPER: Regular generator yielding each item of the given sequence.
    'Regular generator'
    for i in seqn:
        yield i

class G:
    # [CLASS DOC] HELPER: Sequence wrapper supporting only item access by index (old-
    # style iteration protocol); raises the underlying IndexError at the end.
    'Sequence using __getitem__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the wrapped sequence.
        self.seqn = seqn
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Returns the item of the wrapped sequence at index i.
        return self.seqn[i]

class I:
    # [CLASS DOC] HELPER: Sequence wrapper implementing the iterator protocol directly:
    # it is its own iterator and walks the sequence with an index attribute, stopping at
    # the end.
    'Sequence using iterator protocol'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a zero position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item and advances the position; raises
        # StopIteration at the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class Ig:
    # [CLASS DOC] HELPER: Sequence wrapper whose iter method is a generator over the
    # stored sequence (iterator protocol through a generator).
    'Sequence using iterator protocol defined with a generator'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter (unused by
        # iteration).
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding every item of the stored sequence.
        for val in self.seqn:
            yield val

class X:
    # [CLASS DOC] HELPER: Defective iterable that has a next method but neither iter nor
    # getitem, so it cannot be iterated; used to check TypeError.
    'Missing __getitem__ and __iter__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item and advances; raises
        # StopIteration at the end (never reachable through normal iteration since the
        # class lacks an iter method).
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class N:
    # [CLASS DOC] HELPER: Defective iterator: its iter returns itself but it has no next
    # method, so iteration raises TypeError.
    'Iterator missing __next__()'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self

class E:
    # [CLASS DOC] HELPER: Iterator whose next always raises ZeroDivisionError, used to
    # test exception propagation.
    'Test propagation of exceptions'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises ZeroDivisionError by dividing by zero.
        3 // 0

class E2:
    # [CLASS DOC] HELPER: Iterator that yields items normally but raises
    # ZeroDivisionError on the third call (after two items).
    'Test propagation of exceptions after two iterations'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Raises ZeroDivisionError once two items were returned,
        # otherwise returns the next item and advances.
        if self.i == 2:
            raise ZeroDivisionError
        v = self.seqn[self.i]
        self.i += 1
        return v

class S:
    # [CLASS DOC] HELPER: Iterator that stops immediately: next always raises
    # StopIteration, ignoring the sequence.
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

def L(seqn):
    # [FUNCTION DOC] HELPER: Builds a multi-level iterator chain: chain over map
    # identity over generator R over Ig over G wrapping the sequence.
    'Test multiple tiers of iterators'
    return chain(map(lambda x:x, R(Ig(G(seqn)))))


class TestVariousIteratorArgs(__TestCase):
    # [CLASS DOC] Checks each itertools function against the helper iterable classes
    # (index-only, iterator protocol, generator, multi-tier, empty, failing, non-
    # iterable) for correct results and error behavior (TypeError, ZeroDivisionError
    # propagation). No state.

    def test_accumulate(self):
        # [FUNCTION DOC] Checks accumulate on running sums for each good helper class,
        # empty on the immediate-stop class, TypeError for a non-iterable, and that
        # errors from the failing helpers propagate.
        s = [1,2,3,4,5]
        r = [1,3,6,10,15]
        n = len(s)
        for g in (G, I, Ig, L, R):
            self.assertEqual(list(accumulate(g(s))), r)
        self.assertEqual(list(accumulate(S(s))), [])
        self.assertRaises(TypeError, accumulate, X(s))
        self.assertRaises(TypeError, accumulate, N(s))
        self.assertRaises(ZeroDivisionError, list, accumulate(E(s)))

    def test_batched(self):
        # [FUNCTION DOC] Checks batched of a five-letter string with size 2 across the
        # good helper iterables, empty result for the immediate-stop one, TypeError for
        # non-iterable helpers, and ZeroDivisionError propagation from failing
        # iterators.
        s = 'abcde'
        r = [('a', 'b'), ('c', 'd'), ('e',)]
        n = 2
        for g in (G, I, Ig, L, R):
            with self.subTest(g=g):
                self.assertEqual(list(batched(g(s), n)), r)
        self.assertEqual(list(batched(S(s), 2)), [])
        self.assertRaises(TypeError, batched, X(s), 2)
        self.assertRaises(TypeError, batched, N(s), 2)
        self.assertRaises(ZeroDivisionError, list, batched(E(s), 2))
        self.assertRaises(ZeroDivisionError, list, batched(E2(s), 4))

    def test_chain(self):
        # [FUNCTION DOC] Checks chain over several sequences and each good helper class
        # equals direct listing (also doubled), TypeError for the defective classes and
        # ZeroDivisionError propagation.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(chain(g(s))), list(g(s)))
                self.assertEqual(list(chain(g(s), g(s))), list(g(s))+list(g(s)))
            self.assertRaises(TypeError, list, chain(X(s)))
            self.assertRaises(TypeError, list, chain(N(s)))
            self.assertRaises(ZeroDivisionError, list, chain(E(s)))

    def test_compress(self):
        # [FUNCTION DOC] Checks compress with an always-true selector reproduces the
        # data from each good helper class; TypeError for defective ones,
        # ZeroDivisionError propagates.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            n = len(s)
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(compress(g(s), repeat(1))), list(g(s)))
            self.assertRaises(TypeError, compress, X(s), repeat(1))
            self.assertRaises(TypeError, compress, N(s), repeat(1))
            self.assertRaises(ZeroDivisionError, list, compress(E(s), repeat(1)))

    def test_product(self):
        # [FUNCTION DOC] Checks product raises TypeError at construction for defective
        # iterables and ZeroDivisionError for the failing iterator (product consumes its
        # inputs eagerly).
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            self.assertRaises(TypeError, product, X(s))
            self.assertRaises(TypeError, product, N(s))
            self.assertRaises(ZeroDivisionError, product, E(s))

    def test_cycle(self):
        # [FUNCTION DOC] Checks cycle over each helper iterable repeats it three times
        # (first three lengths via islice), TypeError for defective ones,
        # ZeroDivisionError propagation.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgtlen = len(s) * 3
                expected = list(g(s))*3
                actual = list(islice(cycle(g(s)), tgtlen))
                self.assertEqual(actual, expected)
            self.assertRaises(TypeError, cycle, X(s))
            self.assertRaises(TypeError, cycle, N(s))
            self.assertRaises(ZeroDivisionError, list, cycle(E(s)))

    def test_groupby(self):
        # [FUNCTION DOC] Checks the keys of groupby over each good helper class equal
        # the items, TypeError for defective ones, ZeroDivisionError propagation.
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual([k for k, sb in groupby(g(s))], list(g(s)))
            self.assertRaises(TypeError, groupby, X(s))
            self.assertRaises(TypeError, groupby, N(s))
            self.assertRaises(ZeroDivisionError, list, groupby(E(s)))

    def test_filter(self):
        # [FUNCTION DOC] Checks filter with the even predicate matches a list
        # comprehension for every helper class; TypeError for defective ones and
        # ZeroDivisionError propagation.
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(filter(isEven, g(s))),
                                 [x for x in g(s) if isEven(x)])
            self.assertRaises(TypeError, filter, isEven, X(s))
            self.assertRaises(TypeError, filter, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, filter(isEven, E(s)))

    def test_filterfalse(self):
        # [FUNCTION DOC] Checks filterfalse with the even predicate gives the odd items
        # for every helper class; TypeError and ZeroDivisionError behavior as for
        # filter. (Uses the odd predicate which equals the complement only for non-
        # negative inputs.)
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(filterfalse(isEven, g(s))),
                                 [x for x in g(s) if isOdd(x)])
            self.assertRaises(TypeError, filterfalse, isEven, X(s))
            self.assertRaises(TypeError, filterfalse, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, filterfalse(isEven, E(s)))

    def test_zip(self):
        # [FUNCTION DOC] Checks zip over one and two copies of each helper iterable
        # against a list-based helper, and TypeError/ZeroDivisionError behavior.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(zip(g(s))), lzip(g(s)))
                self.assertEqual(list(zip(g(s), g(s))), lzip(g(s), g(s)))
            self.assertRaises(TypeError, zip, X(s))
            self.assertRaises(TypeError, zip, N(s))
            self.assertRaises(ZeroDivisionError, list, zip(E(s)))

    def test_ziplongest(self):
        # [FUNCTION DOC] Checks zip_longest over one and two copies of each helper
        # iterable matches zip (equal lengths), plus TypeError and ZeroDivisionError
        # behavior.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(zip_longest(g(s))), list(zip(g(s))))
                self.assertEqual(list(zip_longest(g(s), g(s))), list(zip(g(s), g(s))))
            self.assertRaises(TypeError, zip_longest, X(s))
            self.assertRaises(TypeError, zip_longest, N(s))
            self.assertRaises(ZeroDivisionError, list, zip_longest(E(s)))

    def test_map(self):
        # [FUNCTION DOC] Checks map with a unary function and with a two-argument power
        # function over each helper class, TypeError for defective inputs,
        # ZeroDivisionError propagation.
        for s in (range(10), range(0), range(100), (7,11), range(20,50,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(map(onearg, g(s))),
                                 [onearg(x) for x in g(s)])
                self.assertEqual(list(map(operator.pow, g(s), g(s))),
                                 [x**x for x in g(s)])
            self.assertRaises(TypeError, map, onearg, X(s))
            self.assertRaises(TypeError, map, onearg, N(s))
            self.assertRaises(ZeroDivisionError, list, map(onearg, E(s)))

    def test_islice(self):
        # [FUNCTION DOC] Checks islice with start 1 and step 2 over each helper class;
        # the TypeError checks for defective inputs are commented out, ZeroDivisionError
        # propagation is still asserted.
        for s in ("12345", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                self.assertEqual(list(islice(g(s),1,None,2)), list(g(s))[1::2])
            # self.assertRaises(TypeError, islice, X(s), 10)
            # self.assertRaises(TypeError, islice, N(s), 10)
            self.assertRaises(ZeroDivisionError, list, islice(E(s), 10))

    def test_pairwise(self):
        # [FUNCTION DOC] Checks pairwise equals zip of a sequence with itself shifted by
        # one for each helper class; TypeError and ZeroDivisionError behavior.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                seq = list(g(s))
                expected = list(zip(seq, seq[1:]))
                actual = list(pairwise(g(s)))
                self.assertEqual(actual, expected)
            self.assertRaises(TypeError, pairwise, X(s))
            self.assertRaises(TypeError, pairwise, N(s))
            self.assertRaises(ZeroDivisionError, list, pairwise(E(s)))

    def test_starmap(self):
        # [FUNCTION DOC] Checks starmap with power over pairs built from each helper
        # class; the TypeError checks are commented out, ZeroDivisionError propagation
        # is asserted using the last pair list from the loop.
        for s in (range(10), range(0), range(100), (7,11), range(20,50,5)):
            for g in (G, I, Ig, S, L, R):
                ss = lzip(s, s)
                self.assertEqual(list(starmap(operator.pow, g(ss))),
                                 [x**x for x in g(s)])
            # self.assertRaises(TypeError, starmap, operator.pow, X(ss))
            # self.assertRaises(TypeError, starmap, operator.pow, N(ss))
            self.assertRaises(ZeroDivisionError, list, starmap(operator.pow, E(ss)))

    def test_takewhile(self):
        # [FUNCTION DOC] Checks takewhile with the even predicate against a manual loop
        # for each helper class; ZeroDivisionError propagation asserted, TypeError
        # checks commented out.
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgt = []
                for elem in g(s):
                    if not isEven(elem): break
                    tgt.append(elem)
                self.assertEqual(list(takewhile(isEven, g(s))), tgt)
            # self.assertRaises(TypeError, takewhile, isEven, X(s))
            # self.assertRaises(TypeError, takewhile, isEven, N(s))
            self.assertRaises(ZeroDivisionError, list, takewhile(isEven, E(s)))

    def test_dropwhile(self):
        # [FUNCTION DOC] Checks dropwhile with the odd predicate against a manual loop
        # for each helper class; ZeroDivisionError propagation asserted, TypeError
        # checks commented out.
        for s in (range(10), range(0), range(1000), (7,11), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                tgt = []
                for elem in g(s):
                    if not tgt and isOdd(elem): continue
                    tgt.append(elem)
                self.assertEqual(list(dropwhile(isOdd, g(s))), tgt)
            # self.assertRaises(TypeError, dropwhile, isOdd, X(s))
            # self.assertRaises(TypeError, dropwhile, isOdd, N(s))
            self.assertRaises(ZeroDivisionError, list, dropwhile(isOdd, E(s)))

    def test_tee(self):
        # [FUNCTION DOC] Checks both copies from tee over each helper class list the
        # full data, TypeError for defective ones, and ZeroDivisionError when the first
        # copy of a failing source is consumed.
        for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
            for g in (G, I, Ig, S, L, R):
                it1, it2 = tee(g(s))
                self.assertEqual(list(it1), list(g(s)))
                self.assertEqual(list(it2), list(g(s)))
            self.assertRaises(TypeError, tee, X(s))
            self.assertRaises(TypeError, tee, N(s))
            self.assertRaises(ZeroDivisionError, list, tee(E(s))[0])

class LengthTransparency(__TestCase):
    # [CLASS DOC] Tests that itertools.repeat reports its remaining length through the
    # length hint protocol. No state.

    def test_repeat(self):
        # [FUNCTION DOC] Checks length_hint of repeat equals the given count (50, 0),
        # and falls back to the supplied default when no count was given.
        self.assertEqual(operator.length_hint(repeat(None, 50)), 50)
        self.assertEqual(operator.length_hint(repeat(None, 0)), 0)
        self.assertEqual(operator.length_hint(repeat(None), 12), 12)

    def test_repeat_with_negative_times(self):
        # [FUNCTION DOC] Checks length_hint of repeat with a negative count (positional
        # or keyword) is zero.
        self.assertEqual(operator.length_hint(repeat(None, -1)), 0)
        self.assertEqual(operator.length_hint(repeat(None, -2)), 0)
        self.assertEqual(operator.length_hint(repeat(None, times=-1)), 0)
        self.assertEqual(operator.length_hint(repeat(None, times=-2)), 0)

class RegressionTests(__TestCase):
    # [CLASS DOC] Regression tests for historical itertools bugs (mutating result
    # tuples, early exception delivery, recursion in chains, groupby crashes). No state.

    def test_sf_793826(self):
        # [FUNCTION DOC] Regression sf 793826: calls a function on a zip result tuple
        # that is later mutated by the zip machinery, and checks the saved copy of the
        # tuple's content matches the tuple object afterwards. Writes the module global
        # T as a side effect.
        # Fix Armin Rigo's successful efforts to wreak havoc

        def mutatingtuple(tuple1, f, tuple2):
            # [FUNCTION DOC] HELPER: Builds a zip over a mapped generator, calls f with
            # the first result tuple while the tuple is still being reused, and then
            # triggers refilling it with the second tuple's values.
            # this builds a tuple t which is a copy of tuple1,
            # then calls f(t), then mutates t to be equal to tuple2
            # (needs len(tuple1) == len(tuple2)).
            def g(value, first=[1]):
                # [FUNCTION DOC] HELPER: Mapped function that on its first call clears a
                # list marker and invokes f on the next zip result, then returns the
                # value unchanged.
                if first:
                    del first[:]
                    f(next(z))
                return value
            items = list(tuple2)
            items[1:1] = list(tuple1)
            gen = map(g, items)
            z = zip(*[gen]*len(tuple1))
            next(z)

        def f(t):
            # [FUNCTION DOC] HELPER: Stores the tuple in the global T and copies its
            # content into the enclosing list first.
            global T
            T = t
            first[:] = list(T)

        first = []
        mutatingtuple((1,2,3), f, (4,5,6))
        second = list(T)
        self.assertEqual(first, second)


    def test_sf_950057(self):
        # [FUNCTION DOC] Regression sf 950057: chain and cycle must raise an exception
        # from an input generator immediately, not when switching sources; checks the
        # history list records only steps 0 and 1 in each case.
        # Make sure that chain() and cycle() catch exceptions immediately
        # rather than when shifting between input sources

        def gen1():
            # [FUNCTION DOC] HELPER: Generator recording steps in the history list,
            # yielding 1, then raising AssertionError.
            hist.append(0)
            yield 1
            hist.append(1)
            raise AssertionError
            hist.append(2)

        def gen2(x):
            # [FUNCTION DOC] HELPER: Generator recording step 3, yielding 2 and
            # recording step 4 on resumption; the argument is unused.
            hist.append(3)
            yield 2
            hist.append(4)

        hist = []
        self.assertRaises(AssertionError, list, chain(gen1(), gen2(False)))
        self.assertEqual(hist, [0,1])

        hist = []
        self.assertRaises(AssertionError, list, chain(gen1(), gen2(True)))
        self.assertEqual(hist, [0,1])

        hist = []
        self.assertRaises(AssertionError, list, cycle(gen1()))
        self.assertEqual(hist, [0,1])

    @support.skip_if_pgo_task
    @support.requires_resource('cpu')
    @slowTest
    def test_long_chain_of_empty_iterables(self):
        # [FUNCTION DOC] Slow, CPU-resource test: chain.from_iterable over ten million
        # empty generators must reach the end with StopIteration without hitting
        # recursion limits.
        # Make sure itertools.chain doesn't run into recursion limits when
        # dealing with long chains of empty iterables. Even with a high
        # number this would probably only fail in Py_DEBUG mode.
        it = chain.from_iterable(() for unused in range(10000000))
        with self.assertRaises(StopIteration):
            next(it)

    def test_issue30347_1(self):
        # [FUNCTION DOC] Regression for issue 30347: consuming a groupby group from
        # inside the key function must not crash.
        def f(n):
            # [FUNCTION DOC] HELPER: Key function that exhausts the current group when n
            # is 5 and returns whether n differs from 6.
            if n == 5:
                list(b)
            return n != 6
        for (k, b) in groupby(range(10), f):
            list(b)  # shouldn't crash

    def test_issue30347_2(self):
        # [FUNCTION DOC] Regression for issue 30347: a key whose equality advances the
        # group iterator during comparison must not crash while consuming groups.
        class K:
            # [CLASS DOC] HELPER: Key wrapper whose equality advances the outer group
            # iterator on its first comparison and always returns true.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Ignores the value.
                pass
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Counts comparisons in the enclosing counter; on
                # the first one calls next on the group iterator, and returns true.
                nonlocal i
                i += 1
                if i == 1:
                    next(g, None)
                return True
        i = 0
        g = next(groupby(range(10), K))[1]
        for j in range(2):
            next(g, None)  # shouldn't crash


class SubclassWithKwargsTest(__TestCase):
    # [CLASS DOC] Tests that itertools classes that can be subclassed accept keyword
    # arguments in subclasses only when the subclass defines them. No state.
    def test_keywords_in_subclass(self):
        # [FUNCTION DOC] For a table of itertools classes with arguments and expected
        # prefix output, checks (a) a plain subclass behaves like the base and rejects
        # an unknown keyword with TypeError, (b) a subclass defining init with a keyword
        # stores it (skipped for repeat, zip, compress which take keywords themselves),
        # (c) a subclass defining new with a keyword stores it. Count is excluded
        # because it cannot be subclassed. Reported via subTest.
        # count is not subclassable...
        testcases = [
            (repeat, (1, 2), [1, 1]),
            (zip, ([1, 2], 'ab'), [(1, 'a'), (2, 'b')]),
            (filter, (None, [0, 1]), [1]),
            (filterfalse, (None, [0, 1]), [0]),
            (chain, ([1, 2], [3, 4]), [1, 2, 3]),
            (map, (str, [1, 2]), ['1', '2']),
            (starmap, (operator.pow, ((2, 3), (3, 2))), [8, 9]),
            (islice, ([1, 2, 3, 4], 1, 3), [2, 3]),
            (takewhile, (isEven, [2, 3, 4]), [2]),
            (dropwhile, (isEven, [2, 3, 4]), [3, 4]),
            (cycle, ([1, 2],), [1, 2, 1]),
            (compress, ('ABC', [1, 0, 1]), ['A', 'C']),
        ]
        for cls, args, result in testcases:
            with self.subTest(cls):
                class subclass(cls):
                    # [CLASS DOC] HELPER: Trivial subclass of the class under test.
                    pass
                u = subclass(*args)
                self.assertIs(type(u), subclass)
                self.assertEqual(list(islice(u, 0, 3)), result)
                with self.assertRaises(TypeError):
                    subclass(*args, newarg=3)

        for cls, args, result in testcases:
            # Constructors of repeat, zip, compress accept keyword arguments.
            # Their subclasses need overriding __new__ to support new
            # keyword arguments.
            if cls in [repeat, zip, compress]:
                continue
            with self.subTest(cls):
                class subclass_with_init(cls):
                    # [CLASS DOC] HELPER: Subclass adding an init that accepts and
                    # stores a keyword argument newarg.
                    def __init__(self, *args, newarg=None):
                        # [FUNCTION DOC] HELPER: Stores the keyword newarg; ignores
                        # other arguments (they are consumed by the base new).
                        self.newarg = newarg
                u = subclass_with_init(*args, newarg=3)
                self.assertIs(type(u), subclass_with_init)
                self.assertEqual(list(islice(u, 0, 3)), result)
                self.assertEqual(u.newarg, 3)

        for cls, args, result in testcases:
            with self.subTest(cls):
                class subclass_with_new(cls):
                    # [CLASS DOC] HELPER: Subclass overriding new to accept a newarg
                    # keyword, forward the remaining arguments to the base new and store
                    # newarg.
                    def __new__(cls, *args, newarg=None):
                        # [FUNCTION DOC] HELPER: Creates the instance via the base new
                        # with positional args only, stores newarg on it, returns it.
                        self = super().__new__(cls, *args)
                        self.newarg = newarg
                        return self
                u = subclass_with_new(*args, newarg=3)
                self.assertIs(type(u), subclass_with_new)
                self.assertEqual(list(islice(u, 0, 3)), result)
                self.assertEqual(u.newarg, 3)


if __name__ == "__main__":
    run_tests()
