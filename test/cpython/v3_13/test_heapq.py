# [MODULE DOC] CPython 3.13 heapq test-suite, adapted to run under Dynamo/PyTorch test infra.
# Builds a pure-Python heapq (C accelerator blocked) and a C-backed heapq (when _heapq is
# available), then runs one shared heap test body and one shared error-handling test body
# against each via mixin classes. Also defines dummy sequence/iterator helper classes used to
# probe argument handling of nlargest/nsmallest. Import-time side effects: imports two fresh
# copies of the heapq module. Several tests are skipped under TEST_WITH_TORCHDYNAMO.
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_collections.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

"""Unittests for heapq."""

import random
import unittest
import doctest

from test.support import import_helper
from unittest import TestCase, skipUnless
from operator import itemgetter

py_heapq = import_helper.import_fresh_module('heapq', blocked=['_heapq'])
c_heapq = import_helper.import_fresh_module('heapq', fresh=['_heapq'])

# _heapq.nlargest/nsmallest are saved in heapq._nlargest/_smallest when
# _heapq is imported, so check them there
func_names = ['heapify', 'heappop', 'heappush', 'heappushpop', 'heapreplace',
              '_heappop_max', '_heapreplace_max', '_heapify_max']

class TestModules(__TestCase):
    # [CLASS DOC] Checks that the heap primitives of each heapq flavor report the expected
    # defining module name (heapq for pure Python, _heapq for C). Stateless.
    def test_py_functions(self):
        # [FUNCTION DOC] For every name in func_names, asserts the pure-Python module's
        # attribute has module name heapq.
        for fname in func_names:
            self.assertEqual(getattr(py_heapq, fname).__module__, 'heapq')

    @skipUnless(c_heapq, 'requires _heapq')
    def test_c_functions(self):
        # [FUNCTION DOC] Same check for the C-backed module: each name must report module
        # _heapq. Skipped when the C extension is unavailable.
        for fname in func_names:
            self.assertEqual(getattr(c_heapq, fname).__module__, '_heapq')


@torch._dynamo.disable
def randrange(*args):
    # [FUNCTION DOC] HELPER: Dynamo-disabled wrapper around random.randrange; forwards all
    # arguments and returns the random integer, keeping RNG calls out of traced code.
    return random.randrange(*args)


class _TestHeap:
    # [CLASS DOC] Mixin holding the behavioral heap tests; not a TestCase itself. Concrete
    # subclasses set the class attribute module to the heapq flavor under test and mix in
    # CPythonTestCase. Tests exercise heappush/heappop/heapify/heapreplace/heappushpop/merge/
    # nsmallest/nlargest with random data and compare against sorted().

    def test_push_pop(self):
        # [FUNCTION DOC] Pushes 256 random floats onto a heap checking the invariant each
        # step, pops all and checks the results equal the sorted input, checks the sorted list
        # is a valid heap, and verifies TypeError for a missing push argument (and for None
        # heaps when the module raises that instead of AttributeError).
        # 1) Push 256 random numbers and pop them off, verifying all's OK.
        heap = []
        data = []
        self.check_invariant(heap)
        for i in range(256):
            item = random.random()
            data.append(item)
            self.module.heappush(heap, item)
            self.check_invariant(heap)
        results = []
        while heap:
            item = self.module.heappop(heap)
            self.check_invariant(heap)
            results.append(item)
        data_sorted = data[:]
        data_sorted.sort()
        self.assertEqual(data_sorted, results)
        # 2) Check that the invariant holds for a sorted array
        self.check_invariant(results)

        self.assertRaises(TypeError, self.module.heappush, [])
        try:
            self.assertRaises(TypeError, self.module.heappush, None, None)
            self.assertRaises(TypeError, self.module.heappop, None)
        except AttributeError:
            pass

    def check_invariant(self, heap):
        # [FUNCTION DOC] HELPER: Asserts every item is >= its parent (parent index is
        # (pos-1)//2) for the given list, i.e. the min-heap invariant holds.
        # Check the heap invariant.
        for pos, item in enumerate(heap):
            if pos: # pos 0 has no parent
                parentpos = (pos-1) >> 1
                self.assertTrue(heap[parentpos] <= item)

    def test_heapify(self):
        # [FUNCTION DOC] Heapifies random float lists of sizes 0..29 and 20000, checks the
        # invariant after each, and that heapify(None) raises TypeError.
        for size in list(range(30)) + [20000]:
            heap = [random.random() for dummy in range(size)]
            self.module.heapify(heap)
            self.check_invariant(heap)

        self.assertRaises(TypeError, self.module.heapify, None)

    def test_naive_nbest(self):
        # [FUNCTION DOC] Naive N-best: pushes 1000 random ints, popping once the heap exceeds
        # 10 items; the remainder must be the 10 largest values.
        data = [randrange(2000) for i in range(1000)]
        heap = []
        for item in data:
            self.module.heappush(heap, item)
            if len(heap) > 10:
                self.module.heappop(heap)
        heap.sort()
        self.assertEqual(heap, sorted(data)[-10:])

    def heapiter(self, heap):
        # [FUNCTION DOC] HELPER: Generator that repeatedly heappops the given heap, yielding
        # items smallest-first until pop raises IndexError (empty), which ends iteration.
        # An iterator returning a heap's elements, smallest-first.
        try:
            while 1:
                yield self.module.heappop(heap)
        except IndexError:
            pass

    def test_nbest(self):
        # [FUNCTION DOC] N-best using a size-10 heap and heapreplace for items larger than the
        # root; draining gives the 10 largest. Also checks heapreplace raises TypeError on bad
        # arguments and IndexError on an empty heap.
        # Less-naive "N-best" algorithm, much faster (if len(data) is big
        # enough <wink>) than sorting all of data.  However, if we had a max
        # heap instead of a min heap, it could go faster still via
        # heapify'ing all of data (linear time), then doing 10 heappops
        # (10 log-time steps).
        data = [randrange(2000) for i in range(1000)]
        heap = data[:10]
        self.module.heapify(heap)
        for item in data[10:]:
            if item > heap[0]:  # this gets rarer the longer we run
                self.module.heapreplace(heap, item)
        self.assertEqual(list(self.heapiter(heap)), sorted(data)[-10:])

        self.assertRaises(TypeError, self.module.heapreplace, None)
        self.assertRaises(TypeError, self.module.heapreplace, None, None)
        self.assertRaises(IndexError, self.module.heapreplace, [], None)

    def test_nbest_with_pushpop(self):
        # [FUNCTION DOC] N-best using heappushpop; the drained heap must hold the 10 largest
        # values. Also checks pushpop on an empty heap returns the pushed item.
        data = [randrange(2000) for i in range(1000)]
        heap = data[:10]
        self.module.heapify(heap)
        for item in data[10:]:
            self.module.heappushpop(heap, item)
        self.assertEqual(list(self.heapiter(heap)), sorted(data)[-10:])
        self.assertEqual(self.module.heappushpop([], 'x'), 'x')

    def test_heappushpop(self):
        # [FUNCTION DOC] Table of heappushpop cases: empty heap returns the item; equal value
        # returns the pushed item and keeps the original (type preserved); smaller item is
        # returned unchanged; larger item replaces the root and the old root is returned.
        h = []
        x = self.module.heappushpop(h, 10)
        self.assertEqual((h, x), ([], 10))

        h = [10]
        x = self.module.heappushpop(h, 10.0)
        self.assertEqual((h, x), ([10], 10.0))
        self.assertEqual(type(h[0]), int)
        self.assertEqual(type(x), float)

        h = [10]
        x = self.module.heappushpop(h, 9)
        self.assertEqual((h, x), ([10], 9))

        h = [10]
        x = self.module.heappushpop(h, 11)
        self.assertEqual((h, x), ([11], 10))

    def test_heappop_max(self):
        # [FUNCTION DOC] Covers the single-remaining-item path of the private _heappop_max by
        # popping a two-item max-heap twice (expects 3 then 2).
        # _heapop_max has an optimization for one-item lists which isn't
        # covered in other tests, so test that case explicitly here
        h = [3, 2]
        self.assertEqual(self.module._heappop_max(h), 3)
        self.assertEqual(self.module._heappop_max(h), 2)

    def test_heapsort(self):
        # [FUNCTION DOC] 100 trials of random size/data; builds the heap with heapify on odd
        # trials and repeated heappush on even trials, then pops everything and expects sorted
        # order.
        # Exercise everything with repeated heapsort checks
        for trial in range(100):
            size = randrange(50)
            data = [randrange(25) for i in range(size)]
            if trial & 1:     # Half of the time, use heapify
                heap = data[:]
                self.module.heapify(heap)
            else:             # The rest of the time, use heappush
                heap = []
                for item in data:
                    self.module.heappush(heap, item)
            heap_sorted = [self.module.heappop(heap) for i in range(size)]
            self.assertEqual(heap_sorted, sorted(data))

    def test_merge(self):
        # [FUNCTION DOC] Builds random rows of (letter, int) tuples, sorts each with each key
        # and reverse combination, and checks merge of the rows equals a sorted of all inputs;
        # merge with no iterables yields nothing.
        inputs = []
        for i in range(randrange(25)):
            row = []
            for j in range(randrange(100)):
                tup = random.choice('ABC'), randrange(-500, 500)
                row.append(tup)
            inputs.append(row)

        for key in [None, itemgetter(0), itemgetter(1), itemgetter(1, 0)]:
            for reverse in [False, True]:
                seqs = []
                for seq in inputs:
                    seqs.append(sorted(seq, key=key, reverse=reverse))
                self.assertEqual(sorted(chain(*inputs), key=key, reverse=reverse),
                                 list(self.module.merge(*seqs, key=key, reverse=reverse)))
                self.assertEqual(list(self.module.merge()), [])

    def test_empty_merges(self):
        # [FUNCTION DOC] Merging two empty lists, with or without a key function, gives no
        # items.
        # Merging two empty lists (with or without a key) should produce
        # another empty list.
        self.assertEqual(list(self.module.merge([], [])), [])
        self.assertEqual(list(self.module.merge([], [], key=lambda: 6)), [])

    def test_merge_does_not_suppress_index_error(self):
        # [FUNCTION DOC] Regression for issue 19018: an IndexError raised inside a user
        # generator during merge must propagate rather than being swallowed.
        # Issue 19018: Heapq.merge suppresses IndexError from user generator
        def iterable():
            # [FUNCTION DOC] HELPER: Generator over range(20) indexing a 10-item list, so it
            # raises IndexError partway through.
            s = list(range(10))
            for i in range(20):
                yield s[i]       # IndexError when i > 10
        with self.assertRaises(IndexError):
            list(self.module.merge(iterable(), iterable()))

    def test_merge_stability(self):
        # [FUNCTION DOC] Merges four sorted streams of int subclass objects carrying (value,
        # stream) pairs and checks the result pairs are sorted, so equal values keep stream
        # order (stable).
        class Int(int):
            # [CLASS DOC] HELPER: Plain int subclass allowing an extra pair attribute to be
            # attached.
            pass
        inputs = [[], [], [], []]
        for i in range(20000):
            stream = randrange(4)
            x = randrange(500)
            obj = Int(x)
            obj.pair = (x, stream)
            inputs[stream].append(obj)
        for stream in inputs:
            stream.sort()
        result = [i.pair for i in self.module.merge(*inputs)]
        self.assertEqual(result, sorted(result))

    def test_nsmallest(self):
        # [FUNCTION DOC] Compares nsmallest to sorted()[:n] for many n (including 0 and above
        # the length), with and without a key function.
        data = [(randrange(2000), i) for i in range(1000)]
        for f in (None, lambda x:  x[0] * 547 % 2000):
            for n in (0, 1, 2, 10, 100, 400, 999, 1000, 1100):
                self.assertEqual(list(self.module.nsmallest(n, data)),
                                sorted(data)[:n])
                self.assertEqual(list(self.module.nsmallest(n, data, key=f)),
                                sorted(data, key=f)[:n])

    def test_nlargest(self):
        # [FUNCTION DOC] Compares nlargest to reverse-sorted prefixes for many n, with and
        # without a key function.
        data = [(randrange(2000), i) for i in range(1000)]
        for f in (None, lambda x:  x[0] * 547 % 2000):
            for n in (0, 1, 2, 10, 100, 400, 999, 1000, 1100):
                self.assertEqual(list(self.module.nlargest(n, data)),
                                sorted(data, reverse=True)[:n])
                self.assertEqual(list(self.module.nlargest(n, data, key=f)),
                                sorted(data, key=f, reverse=True)[:n])

    def test_comparison_operator(self):
        # [FUNCTION DOC] Issue 3051: heap ordering must work with only __lt__ defined.
        # Heapsorts wrapper objects with reversed __lt__ and expects descending order.
        # KNOWN ISSUE: the final assertRaises passes the list data as the callable, so it
        #     checks that calling a list raises TypeError, not that __le__-only objects fail
        #     in heapify.
        # Issue 3051: Make sure heapq works with both __lt__
        # For python 3.0, __le__ alone is not enough
        def hsort(data, comp):
            # [FUNCTION DOC] HELPER: Wraps each item with comp, heapifies, then pops all and
            # returns the wrapped x values in pop order.
            data = [comp(x) for x in data]
            self.module.heapify(data)
            return [self.module.heappop(data).x for i in range(len(data))]
        class LT:
            # [CLASS DOC] HELPER: Wrapper whose __lt__ is inverted (greater-than on x).
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the wrapped value in x.
                self.x = x
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Returns True when this x is greater than the other x.
                return self.x > other.x
        class LE:
            # [CLASS DOC] HELPER: Wrapper defining only an inverted __le__ (no __lt__).
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the wrapped value in x.
                self.x = x
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: Returns True when this x is >= the other x.
                return self.x >= other.x
        data = [random.random() for i in range(100)]
        target = sorted(data, reverse=True)
        self.assertEqual(hsort(data, LT), target)
        self.assertRaises(TypeError, data, LE)


class TestHeapPython(_TestHeap, __TestCase):
    # [CLASS DOC] Runs the shared heap tests against the pure-Python heapq (module attribute).
    module = py_heapq


@skipUnless(c_heapq, 'requires _heapq')
class TestHeapC(_TestHeap, __TestCase):
    # [CLASS DOC] Runs the shared heap tests against the C _heapq-backed heapq; skipped when
    # the extension is not importable.
    module = c_heapq


#==============================================================================

class LenOnly:
    # [CLASS DOC] HELPER: Sequence stub defining only __len__, used to check that heap
    # functions reject objects that are not real lists.
    "Dummy sequence class defining __len__ but not __getitem__."
    def __len__(self):
        # [FUNCTION DOC] HELPER: Always reports length 10.
        return 10

class CmpErr:
    # [CLASS DOC] HELPER: Element whose every comparison raises ZeroDivisionError.
    "Dummy element that always raises an error during comparison"
    def __eq__(self, other):
        raise ZeroDivisionError
    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__

def R(seqn):
    # [FUNCTION DOC] HELPER: Generator yielding each item of the given iterable in order.
    'Regular generator'
    for i in seqn:
        yield i

class G:
    # [CLASS DOC] HELPER: Sequence adapter supporting only indexed access to the wrapped data.
    'Sequence using __getitem__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the wrapped sequence.
        self.seqn = seqn
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Returns the wrapped item at index i (IndexError ends
        # iteration).
        return self.seqn[i]

class I:
    # [CLASS DOC] HELPER: Iterator over a sequence implementing both __iter__ and __next__,
    # keeping its own position counter.
    'Sequence using iterator protocol'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and resets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item and advances; raises StopIteration at
        # the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class Ig:
    # [CLASS DOC] HELPER: Iterable whose __iter__ is a generator over the wrapped sequence.
    'Sequence using iterator protocol defined with a generator'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence (position counter is unused).
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding each wrapped item in order.
        for val in self.seqn:
            yield val

class X:
    # [CLASS DOC] HELPER: Has __next__ but no __iter__ or __getitem__, so it is not iterable.
    'Missing __getitem__ and __iter__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and resets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item or raises StopIteration at the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class N:
    # [CLASS DOC] HELPER: Has __iter__ returning itself but no __next__, so iteration fails.
    'Iterator missing __next__()'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and resets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self

class E:
    # [CLASS DOC] HELPER: Iterator whose __next__ always raises ZeroDivisionError.
    'Test propagation of exceptions'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and resets the position to zero.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises ZeroDivisionError (integer division by zero).
        3 // 0

class S:
    # [CLASS DOC] HELPER: Iterator that is immediately exhausted.
    'Test immediate stop'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Ignores the sequence argument.
        pass
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises StopIteration.
        raise StopIteration

from itertools import chain
def L(seqn):
    # [FUNCTION DOC] HELPER: Builds a multi-tier iterator by wrapping the sequence in G, Ig,
    # R, an identity map and chain, and returns the resulting lazy iterator.
    'Test multiple tiers of iterators'
    return chain(map(lambda x:x, R(Ig(G(seqn)))))


class SideEffectLT:
    # [CLASS DOC] HELPER: Item whose __lt__ empties the shared heap list as a side effect, to
    # test heap functions that must survive mutation during comparison. Holds value and heap.
    def __init__(self, value, heap):
        # [FUNCTION DOC] HELPER: Stores the comparison value and a reference to the heap list.
        self.value = value
        self.heap = heap

    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Clears the shared heap in place, then returns whether this
        # value is less than the other value.
        self.heap[:] = []
        return self.value < other.value


class _TestErrorHandling:
    # [CLASS DOC] Mixin with error-handling tests for heap functions; subclasses set module to
    # the heapq flavor under test. Uses the dummy classes defined above.

    def test_non_sequence(self):
        # [FUNCTION DOC] Heapify/heappop given an int must raise TypeError or AttributeError,
        # as must the two-argument functions (push, replace, nlargest, nsmallest) given two
        # ints.
        for f in (self.module.heapify, self.module.heappop):
            self.assertRaises((TypeError, AttributeError), f, 10)
        for f in (self.module.heappush, self.module.heapreplace,
                  self.module.nlargest, self.module.nsmallest):
            self.assertRaises((TypeError, AttributeError), f, 10, 10)

    def test_len_only(self):
        # [FUNCTION DOC] Objects with only __len__ must be rejected: TypeError or
        # AttributeError for heap mutators, TypeError for nlargest and nsmallest.
        for f in (self.module.heapify, self.module.heappop):
            self.assertRaises((TypeError, AttributeError), f, LenOnly())
        for f in (self.module.heappush, self.module.heapreplace):
            self.assertRaises((TypeError, AttributeError), f, LenOnly(), 10)
        for f in (self.module.nlargest, self.module.nsmallest):
            self.assertRaises(TypeError, f, 2, LenOnly())

    def test_cmp_err(self):
        # [FUNCTION DOC] Heap functions run over elements whose comparisons raise must
        # propagate ZeroDivisionError for heapify, pop, push, replace, nlargest and nsmallest.
        seq = [CmpErr(), CmpErr(), CmpErr()]
        for f in (self.module.heapify, self.module.heappop):
            self.assertRaises(ZeroDivisionError, f, seq)
        for f in (self.module.heappush, self.module.heapreplace):
            self.assertRaises(ZeroDivisionError, f, seq, 10)
        for f in (self.module.nlargest, self.module.nsmallest):
            self.assertRaises(ZeroDivisionError, f, 2, seq)

    def test_arg_parsing(self):
        # [FUNCTION DOC] Calling each heap function with a single int argument must raise
        # TypeError or AttributeError.
        for f in (self.module.heapify, self.module.heappop,
                self.module.heappush, self.module.heapreplace,
                self.module.nlargest, self.module.nsmallest):
            self.assertRaises((TypeError, AttributeError), f, 10)

    def test_iterable_args(self):
        # [FUNCTION DOC] For nlargest and nsmallest over several inputs, results over the
        # iterator wrappers G, I, Ig, L, R match the plain input; S gives an empty result; X
        # and N raise TypeError; E propagates ZeroDivisionError.
        for f in (self.module.nlargest, self.module.nsmallest):
            for s in ("123", "", range(1000), (1, 1.2), range(2000,2200,5)):
                for g in (G, I, Ig, L, R):
                    self.assertEqual(list(f(2, g(s))), list(f(2,s)))
                self.assertEqual(list(f(2, S(s))), [])
                self.assertRaises(TypeError, f, 2, X(s))
                self.assertRaises(TypeError, f, 2, N(s))
                self.assertRaises(ZeroDivisionError, f, 2, E(s))

    # Issue #17278: the heap may change size while it's being walked.

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_heappush_mutating_heap(self):
        # [FUNCTION DOC] Issue 17278: pushing onto a heap whose elements' comparisons empty it
        # must raise IndexError (Python) or RuntimeError (C). Skipped under Dynamo.
        heap = []
        heap.extend(SideEffectLT(i, heap) for i in range(200))
        # Python version raises IndexError, C version RuntimeError
        with self.assertRaises((IndexError, RuntimeError)):
            self.module.heappush(heap, SideEffectLT(5, heap))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_heappop_mutating_heap(self):
        # [FUNCTION DOC] Same for heappop on a heap emptied during comparison; expects
        # IndexError or RuntimeError. Skipped under Dynamo.
        heap = []
        heap.extend(SideEffectLT(i, heap) for i in range(200))
        # Python version raises IndexError, C version RuntimeError
        with self.assertRaises((IndexError, RuntimeError)):
            self.module.heappop(heap)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_comparison_operator_modifiying_heap(self):
        # [FUNCTION DOC] bpo-39421: an element whose __lt__ clears the heap must make
        # heappushpop raise IndexError (strong references must be held during comparison).
        # Skipped under Dynamo.
        # See bpo-39421: Strong references need to be taken
        # when comparing objects as they can alter the heap
        class EvilClass(int):
            # [CLASS DOC] HELPER: int subclass whose __lt__ clears the enclosing heap and
            # returns NotImplemented.
            def __lt__(self, o):
                # [FUNCTION DOC] HELPER: Clears the closed-over heap and returns
                # NotImplemented.
                heap.clear()
                return NotImplemented

        heap = []
        self.module.heappush(heap, EvilClass(0))
        self.assertRaises(IndexError, self.module.heappushpop, heap, 1)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_comparison_operator_modifiying_heap_two_heaps(self):
        # [FUNCTION DOC] Two heaps whose elements' __lt__ clear the other heap; pushing each
        # other's element type must raise IndexError or RuntimeError. Skipped under Dynamo.

        class h(int):
            # [CLASS DOC] HELPER: int subclass whose __lt__ clears list2 and returns
            # NotImplemented.
            def __lt__(self, o):
                # [FUNCTION DOC] HELPER: Clears the closed-over list2 and returns
                # NotImplemented.
                list2.clear()
                return NotImplemented

        class g(int):
            # [CLASS DOC] HELPER: int subclass whose __lt__ clears list1 and returns
            # NotImplemented.
            def __lt__(self, o):
                # [FUNCTION DOC] HELPER: Clears the closed-over list1 and returns
                # NotImplemented.
                list1.clear()
                return NotImplemented

        list1, list2 = [], []

        self.module.heappush(list1, h(0))
        self.module.heappush(list2, g(0))

        self.assertRaises((IndexError, RuntimeError), self.module.heappush, list1, g(1))
        self.assertRaises((IndexError, RuntimeError), self.module.heappush, list2, h(1))

class TestErrorHandlingPython(_TestErrorHandling, __TestCase):
    # [CLASS DOC] Runs the error-handling tests against the pure-Python heapq.
    module = py_heapq

@skipUnless(c_heapq, 'requires _heapq')
class TestErrorHandlingC(_TestErrorHandling, __TestCase):
    # [CLASS DOC] Runs the error-handling tests against the C-backed heapq; skipped if absent.
    module = c_heapq


if __name__ == "__main__":
    run_tests()
