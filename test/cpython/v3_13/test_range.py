# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_range suite for the builtin
# range type: construction and argument errors, huge (beyond machine int) operands,
# len overflow, indexing and slicing, index/count/contains, comparison, repr, pickling
# of ranges and their iterators, iterator __setstate__, and start/stop/step
# attributes. Defines two pure-Python reference generators used for cross-checking.
# No import-time side effects besides class creation.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_range.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

# Python test set -- built-in functions

import unittest
import sys
import pickle
import itertools
from test.support import ALWAYS_EQ

# pure Python implementations (3 args only), for comparison
def pyrange(start, stop, step):
    # [FUNCTION DOC] HELPER: Pure-Python reference generator for range with three arguments.
    # Yields start, start+step, ... while the sequence has not reached stop; the stop is first
    # snapped to the next value congruent to start modulo step. Yields nothing if the
    # range is empty.
    if (start - stop) // step < 0:
        # replace stop with next element in the sequence of integers
        # that are congruent to start modulo step.
        stop += (start - stop) % step
        while start != stop:
            yield start
            start += step

def pyrange_reversed(start, stop, step):
    # [FUNCTION DOC] HELPER: Reference for reversed(range): returns a generator over the same
    # elements in reverse order, built by adjusting stop to the last element and calling
    # pyrange with a negated step.
    stop += (start - stop) % step
    return pyrange(stop - step, start - step, -step)


class RangeTest(__TestCase):
    # [CLASS DOC] Test case for the builtin range: construction, errors, large values,
    # sequence operations, comparison, pickling and attributes.
    def assert_iterators_equal(self, xs, ys, test_id, limit=None):
        # [FUNCTION DOC] HELPER: Assert that iterator xs yields the same items as ys, optionally
        # limited to the first limit items. Fails the test with a message tagged by test_id
        # if xs ends early, has extra items, or differs at some position.
        # check that an iterator xs matches the expected results ys,
        # up to a given limit.
        if limit is not None:
            xs = itertools.islice(xs, limit)
            ys = itertools.islice(ys, limit)
        sentinel = object()
        pairs = itertools.zip_longest(xs, ys, fillvalue=sentinel)
        for i, (x, y) in enumerate(pairs):
            if x == y:
                continue
            elif x == sentinel:
                self.fail('{}: iterator ended unexpectedly '
                          'at position {}; expected {}'.format(test_id, i, y))
            elif y == sentinel:
                self.fail('{}: unexpected excess element {} at '
                          'position {}'.format(test_id, x, i))
            else:
                self.fail('{}: wrong element at position {}; '
                          'expected {}, got {}'.format(test_id, i, y, x))

    def test_range(self):
        # [FUNCTION DOC] Checks list contents of range for 1-3 arguments, negative and empty
        # ranges, membership, lengths, and len for ranges near sys.maxsize.
        self.assertEqual(list(range(3)), [0, 1, 2])
        self.assertEqual(list(range(1, 5)), [1, 2, 3, 4])
        self.assertEqual(list(range(0)), [])
        self.assertEqual(list(range(-3)), [])
        self.assertEqual(list(range(1, 10, 3)), [1, 4, 7])
        self.assertEqual(list(range(5, -5, -3)), [5, 2, -1, -4])

        a = 10
        b = 100
        c = 50

        self.assertEqual(list(range(a, a+2)), [a, a+1])
        self.assertEqual(list(range(a+2, a, -1)), [a+2, a+1])
        self.assertEqual(list(range(a+4, a, -2)), [a+4, a+2])

        seq = list(range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)

        seq = list(range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)

        seq = list(range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)

        self.assertEqual(len(range(0, sys.maxsize, sys.maxsize-1)), 2)

        r = range(-sys.maxsize, sys.maxsize, 2)
        self.assertEqual(len(r), sys.maxsize)

    def test_range_constructor_error_messages(self):
        # [FUNCTION DOC] Checks the TypeError messages of range called with no arguments and with
        # too many (six) arguments.
        with self.assertRaisesRegex(
                TypeError,
                "range expected at least 1 argument, got 0"
        ):
            range()

        with self.assertRaisesRegex(
                TypeError,
                "range expected at most 3 arguments, got 6"
        ):
            range(1, 2, 3, 4, 5, 6)

    def test_large_operands(self):
        # [FUNCTION DOC] Checks range with arguments beyond 64 bits: lengths, emptiness, truth
        # value, negative strides, membership and first/last elements of generated lists.
        x = range(10**20, 10**20+10, 3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)

        x = range(10**20+10, 10**20, 3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        self.assertFalse(x)

        x = range(10**20, 10**20+10, -3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        self.assertFalse(x)

        x = range(10**20+10, 10**20, -3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)
        self.assertTrue(x)

        # Now test range() with longs
        for x in [range(-2**100),
                  range(0, -2**100),
                  range(0, 2**100, -1)]:
            self.assertEqual(list(x), [])
            self.assertFalse(x)

        a = int(10 * sys.maxsize)
        b = int(100 * sys.maxsize)
        c = int(50 * sys.maxsize)

        self.assertEqual(list(range(a, a+2)), [a, a+1])
        self.assertEqual(list(range(a+2, a, -1)), [a+2, a+1])
        self.assertEqual(list(range(a+4, a, -2)), [a+4, a+2])

        seq = list(range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], a)
        self.assertEqual(seq[-1], a+c)

        seq = list(range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], b)
        self.assertEqual(seq[-1], b-c)

        seq = list(range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], -a)
        self.assertEqual(seq[-1], -a-c)

    def test_large_range(self):
        # [FUNCTION DOC] Checks ranges whose length exceeds sys.maxsize: len raises OverflowError,
        # while membership, truthiness, indexing, slicing and out-of-range IndexError work,
        # verified against a computed length.
        # Check long ranges (len > sys.maxsize)
        # len() is expected to fail due to limitations of the __len__ protocol
        def _range_len(x):
            # [FUNCTION DOC] HELPER: Return len of range x, or
            # if that raises OverflowError, compute
            # the length from first element, last element and step.
            try:
                length = len(x)
            except OverflowError:
                step = x[1] - x[0]
                length = 1 + ((x[-1] - x[0]) // step)
            return length

        a = -sys.maxsize
        b = sys.maxsize
        expected_len = b - a
        x = range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+idx)
        self.assertEqual(x[idx:idx+1][0], a+idx)
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = 2 * sys.maxsize
        expected_len = b - a
        x = range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+idx)
        self.assertEqual(x[idx:idx+1][0], a+idx)
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = sys.maxsize**10
        c = 2*sys.maxsize
        expected_len = 1 + (b - a) // c
        x = range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+(idx*c))
        self.assertEqual(x[idx:idx+1][0], a+(idx*c))
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = sys.maxsize**10
        b = 0
        c = -2*sys.maxsize
        expected_len = 1 + (b - a) // c
        x = range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        self.assertTrue(x)
        self.assertEqual(_range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize+1
        self.assertEqual(x[idx], a+(idx*c))
        self.assertEqual(x[idx:idx+1][0], a+(idx*c))
        with self.assertRaises(IndexError):
            x[-expected_len-1]
        with self.assertRaises(IndexError):
            x[expected_len]

    def test_invalid_invocation(self):
        # [FUNCTION DOC] Checks invalid range calls raise errors: wrong argument count and zero
        # step (including big ints) and float or string arguments in every combination.
        self.assertRaises(TypeError, range)
        self.assertRaises(TypeError, range, 1, 2, 3, 4)
        self.assertRaises(ValueError, range, 1, 2, 0)
        a = int(10 * sys.maxsize)
        self.assertRaises(ValueError, range, a, a + 1, int(0))
        self.assertRaises(TypeError, range, 1., 1., 1.)
        self.assertRaises(TypeError, range, 1e100, 1e101, 1e101)
        self.assertRaises(TypeError, range, 0, "spam")
        self.assertRaises(TypeError, range, 0, 42, "spam")
        # Exercise various combinations of bad arguments, to check
        # refcounting logic
        self.assertRaises(TypeError, range, 0.0)
        self.assertRaises(TypeError, range, 0, 0.0)
        self.assertRaises(TypeError, range, 0.0, 0)
        self.assertRaises(TypeError, range, 0.0, 0.0)
        self.assertRaises(TypeError, range, 0, 0, 1.0)
        self.assertRaises(TypeError, range, 0, 0.0, 1)
        self.assertRaises(TypeError, range, 0, 0.0, 1.0)
        self.assertRaises(TypeError, range, 0.0, 0, 1)
        self.assertRaises(TypeError, range, 0.0, 0, 1.0)
        self.assertRaises(TypeError, range, 0.0, 0.0, 1)
        self.assertRaises(TypeError, range, 0.0, 0.0, 1.0)

    def test_index(self):
        # [FUNCTION DOC] Checks range.index and count: results, ValueError for missing values,
        # TypeError without argument, propagation of exceptions from comparisons, big ranges, and
        # an always-equal object.
        u = range(2)
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = range(-2, 3)
        self.assertEqual(u.count(0), 1)
        self.assertEqual(u.index(0), 2)
        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            # [CLASS DOC] HELPER: Exception raised by BadCmp.
            pass

        class BadCmp:
            # [CLASS DOC] HELPER: Object whose equality test raises BadExc when compared with 2.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Raises BadExc if other equals 2, else returns False.
                if other == 2:
                    raise BadExc()
                return False

        a = range(4)
        self.assertRaises(BadExc, a.index, BadCmp())

        a = range(-2, 3)
        self.assertEqual(a.index(0), 2)
        self.assertEqual(range(1, 10, 3).index(4), 1)
        self.assertEqual(range(1, -10, -3).index(-5), 2)

        self.assertEqual(range(10**20).index(1), 1)
        self.assertEqual(range(10**20).index(10**20 - 1), 10**20 - 1)

        self.assertRaises(ValueError, range(1, 2**100, 2).index, 2**87)
        self.assertEqual(range(1, 2**100, 2).index(2**87+1), 2**86)

        self.assertEqual(range(10).index(ALWAYS_EQ), 0)

    def test_user_index_method(self):
        # [FUNCTION DOC] Checks that range accepts user objects defining the index protocol
        # (small and large values), propagates errors from a failing index method, rejects
        # a non-integer return with TypeError, and the same for slice bounds.
        bignum = 2*sys.maxsize
        smallnum = 42

        # User-defined class with an __index__ method
        class I:
            # [CLASS DOC] HELPER: Wraps an integer and exposes it through the index protocol.
            def __init__(self, n):
                # [FUNCTION DOC] HELPER: Store int(n) in attribute n.
                self.n = int(n)
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the stored integer.
                return self.n
        self.assertEqual(list(range(I(bignum), I(bignum + 1))), [bignum])
        self.assertEqual(list(range(I(smallnum), I(smallnum + 1))), [smallnum])

        # User-defined class with a failing __index__ method
        class IX:
            # [CLASS DOC] HELPER: Object whose index method raises RuntimeError.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError
        self.assertRaises(RuntimeError, range, IX())

        # User-defined class with an invalid __index__ method
        class IN:
            # [CLASS DOC] HELPER: Object whose index method returns a string.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns a non-integer string.
                return "not a number"

        self.assertRaises(TypeError, range, IN())

        # Test use of user-defined classes in slice indices.
        self.assertEqual(range(10)[:I(5)], range(5))

        with self.assertRaises(RuntimeError):
            range(0, 10)[:IX()]

        with self.assertRaises(TypeError):
            range(0, 10)[:IN()]

    def test_count(self):
        # [FUNCTION DOC] Checks range.count results and int return type for small, huge and
        # strided ranges, and len of a range just above sys.maxsize.
        self.assertEqual(range(3).count(-1), 0)
        self.assertEqual(range(3).count(0), 1)
        self.assertEqual(range(3).count(1), 1)
        self.assertEqual(range(3).count(2), 1)
        self.assertEqual(range(3).count(3), 0)
        self.assertIs(type(range(3).count(-1)), int)
        self.assertIs(type(range(3).count(1)), int)
        self.assertEqual(range(10**20).count(1), 1)
        self.assertEqual(range(10**20).count(10**20), 0)
        self.assertEqual(range(3).index(1), 1)
        self.assertEqual(range(1, 2**100, 2).count(2**87), 0)
        self.assertEqual(range(1, 2**100, 2).count(2**87+1), 1)

        # self.assertEqual(range(10).count(ALWAYS_EQ), 10)

        self.assertEqual(len(range(sys.maxsize, sys.maxsize+10)), 10)

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of ranges with one, two and three arguments.
        self.assertEqual(repr(range(1)), 'range(0, 1)')
        self.assertEqual(repr(range(1, 2)), 'range(1, 2)')
        self.assertEqual(repr(range(1, 2, 3)), 'range(1, 2, 3)')

    def test_pickling(self):
        # [FUNCTION DOC] Pickles and unpickles ranges (including huge values) with every protocol
        # and checks the elements are unchanged.
        testcases = [(13,), (0, 11), (-22, 10), (20, 3, -1),
                     (13, 21, 3), (-2, 2, 2), (2**65, 2**65+2)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                with self.subTest(proto=proto, test=t):
                    r = range(*t)
                    self.assertEqual(list(pickle.loads(pickle.dumps(r, proto))),
                                     list(r))

    def test_iterator_pickling(self):
        # [FUNCTION DOC] Pickles range iterators at start and after one step, for all protocols and
        # big boundary values, checking type and remaining elements are preserved.
        testcases = [(13,), (0, 11), (-22, 10), (20, 3, -1), (13, 21, 3),
                     (-2, 2, 2)]
        for M in 2**31, 2**63:
            testcases += [
                (M-3, M-1), (4*M, 4*M+2),
                (M-2, M-1, 2), (-M+1, -M, -2),
                (1, 2, M-1), (-1, -2, -M),
                (1, M-1, M-1), (-1, -M, -M),
            ]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                with self.subTest(proto=proto, t=t):
                    it = itorg = iter(range(*t))
                    data = list(range(*t))

                    d = pickle.dumps(it, proto)
                    it = pickle.loads(d)
                    self.assertEqual(type(itorg), type(it))
                    self.assertEqual(list(it), data)

                    it = pickle.loads(d)
                    try:
                        next(it)
                    except StopIteration:
                        continue
                    d = pickle.dumps(it, proto)
                    it = pickle.loads(d)
                    self.assertEqual(list(it), data[1:])

    @skipIfTorchDynamo("infinite loop")
        # [FUNCTION DOC] Pickles an iterator advanced past 32 bits using __setstate__ and checks
        # the next element after unpickling. Skipped under Dynamo (infinite loop).
    def test_iterator_pickling_overflowing_index(self):
        # [FUNCTION DOC] Pickle round trip for an iterator
        # advanced to 2**32+1 with the state setter.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                it = iter(range(2**32 + 2))
                it.__setstate__(2**32 + 1)  # undocumented way to advance an iterator
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(next(it), 2**32 + 1)

    def test_exhausted_iterator_pickling(self):
        # [FUNCTION DOC] Exhausts an iterator over a huge range, pickles it with each protocol and
        # checks both original and copy yield no more items.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            r = range(2**65, 2**65+2)
            i = iter(r)
            while True:
                r = next(i)
                if r == 2**65+1:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    def test_large_exhausted_iterator_pickling(self):
        # [FUNCTION DOC] Same as above for a small range of 20 elements.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            r = range(20)
            i = iter(r)
            while True:
                r = next(i)
                if r == 19:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    def test_iterator_unpickle_compat(self):
        # [FUNCTION DOC] Unpickles byte strings produced by older Python versions (protocols 0-4,
        # small and huge ranges) and checks the iterator yields 14, 16, 18.
        testcases = [
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(I10\nI20\nI2\ntRtRI2\nb.',
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(K\nK\x14K\x02tRtRK\x02b.',
            b'\x80\x02c__builtin__\niter\nc__builtin__\nxrange\nK\nK\x14K\x02\x87R\x85RK\x02b.',
            b'\x80\x03cbuiltins\niter\ncbuiltins\nrange\nK\nK\x14K\x02\x87R\x85RK\x02b.',
            b'\x80\x04\x951\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x8c\x04iter\x93\x8c\x08builtins\x8c\x05range\x93K\nK\x14K\x02\x87R\x85RK\x02b.',

            b'c__builtin__\niter\n(c__builtin__\nxrange\n(L-36893488147419103232L\nI20\nI2\ntRtRL18446744073709551623L\nb.',
            b'c__builtin__\niter\n(c__builtin__\nxrange\n(L-36893488147419103232L\nK\x14K\x02tRtRL18446744073709551623L\nb.',
            b'\x80\x02c__builtin__\niter\nc__builtin__\nxrange\n\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
            b'\x80\x03cbuiltins\niter\ncbuiltins\nrange\n\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
            b'\x80\x04\x95C\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x8c\x04iter\x93\x8c\x08builtins\x8c\x05range\x93\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\xfeK\x14K\x02\x87R\x85R\x8a\t\x07\x00\x00\x00\x00\x00\x00\x00\x01b.',
        ]
        for t in testcases:
            it = pickle.loads(t)
            self.assertEqual(list(it), [14, 16, 18])

    def test_iterator_setstate(self):
        # [FUNCTION DOC] Checks __setstate__ on forward and
        # reversed iterators, for small and 64-bit+
        # ranges, repositions iteration as expected.
        it = iter(range(10, 20, 2))
        it.__setstate__(2)
        self.assertEqual(list(it), [14, 16, 18])
        it = reversed(range(10, 20, 2))
        it.__setstate__(3)
        self.assertEqual(list(it), [12, 10])
        it = iter(range(-2**65, 20, 2))
        it.__setstate__(2**64 + 7)
        self.assertEqual(list(it), [14, 16, 18])
        it = reversed(range(10, 2**65, 2))
        it.__setstate__(2**64 - 7)
        self.assertEqual(list(it), [12, 10])

    def test_odd_bug(self):
        # [FUNCTION DOC] Regression check that a bad argument (a list) to range raises TypeError,
        # not SystemError.
        # This used to raise a "SystemError: NULL result without error"
        # because the range validation step was eating the exception
        # before NULL was returned.
        with self.assertRaises(TypeError):
            range([], 1, -1)

    def test_types(self):
        # [FUNCTION DOC] Checks membership semantics with non-int values: equal floats, bool and
        # complex are contained; objects not equal-comparable are not; int subclasses with custom
        # equality are handled by comparison.
        # Non-integer objects *equal* to any of the range's items are supposed
        # to be contained in the range.
        self.assertIn(1.0, range(3))
        self.assertIn(True, range(3))
        self.assertIn(1+0j, range(3))

        self.assertIn(ALWAYS_EQ, range(3))

        # Objects are never coerced into other types for comparison.
        class C2:
            # [CLASS DOC] HELPER: Object convertible via int and index but not equal to ints.
            # [FUNCTION DOC] HELPER: Conversion to int; returns
            # 1. (Single-line def, block sits above.)
            def __int__(self): return 1
            # [FUNCTION DOC] HELPER: Index conversion; returns 1.
            # (Single-line def, block sits above.)
            def __index__(self): return 1
        self.assertNotIn(C2(), range(3))
        # ..except if explicitly told so.
        self.assertIn(int(C2()), range(3))

        # Check that the range.__contains__ optimization is only
        # used for ints, not for instances of subclasses of int.
        class C3(int):
            # [CLASS DOC] HELPER: int subclass whose equality is always true.
            # [FUNCTION DOC] HELPER: Equality that always returns True. (Single-line def.)
            def __eq__(self, other): return True
        self.assertIn(C3(11), range(10))
        self.assertIn(C3(11), list(range(10)))

    def test_strided_limits(self):
        # [FUNCTION DOC] Checks membership at the boundaries
        # of positive and negative strided ranges.
        r = range(0, 101, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertIn(2, r)
        self.assertNotIn(99, r)
        self.assertIn(100, r)
        self.assertNotIn(101, r)

        r = range(0, -20, -1)
        self.assertIn(0, r)
        self.assertIn(-1, r)
        self.assertIn(-19, r)
        self.assertNotIn(-20, r)

        r = range(0, -20, -2)
        self.assertIn(-18, r)
        self.assertNotIn(-19, r)
        self.assertNotIn(-20, r)

    def test_empty(self):
        # [FUNCTION DOC] Checks empty ranges contain nothing.
        r = range(0)
        self.assertNotIn(0, r)
        self.assertNotIn(1, r)

        r = range(0, -10)
        self.assertNotIn(0, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(1, r)

    @skipIfTorchDynamo("Unrolls ~3M symbolic range-iterator steps; ~597s vs 0.9s eager")
    def test_range_iterators(self):
        # [FUNCTION DOC] Compares range iterators and reversed range iterators with the pure-Python
        # reference over many start/end/step combinations (32 and 64 bit limits), checking the first
        # 100 items. Skipped under Dynamo because it is far too slow.
        # exercise 'fast' iterators, that use a rangeiterobject internally.
        # see issue 7298
        limits = [base + jiggle
                  for M in (2**32, 2**64)
                  for base in (-M, -M//2, 0, M//2, M)
                  for jiggle in (-2, -1, 0, 1, 2)]
        test_ranges = [(start, end, step)
                       for start in limits
                       for end in limits
                       for step in (-2**63, -2**31, -2, -1, 1, 2)]
        test_ranges += [(-2**63, 2**63-2, 1)] # regression test for gh-100810

        for start, end, step in test_ranges:
            iter1 = range(start, end, step)
            iter2 = pyrange(start, end, step)
            test_id = "range({}, {}, {})".format(start, end, step)
            # check first 100 entries
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

            iter1 = reversed(range(start, end, step))
            iter2 = pyrange_reversed(start, end, step)
            test_id = "reversed(range({}, {}, {}))".format(start, end, step)
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

    def test_range_iterators_invocation(self):
        # [FUNCTION DOC] Checks range iterator types cannot be constructed directly: calling the
        # type raises TypeError for both machine-int and long iterators.
        # verify range iterators instances cannot be created by
        # calling their type
        rangeiter_type = type(iter(range(0)))
        self.assertRaises(TypeError, rangeiter_type, 1, 3, 1)
        long_rangeiter_type = type(iter(range(1 << 1000)))
        self.assertRaises(TypeError, long_rangeiter_type, 1, 3, 1)

    def test_slice(self):
        # [FUNCTION DOC] Checks slicing ranges gives the same elements and length as slicing the
        # equivalent list, for various ranges and slice bounds.
        def check(start, stop, step=None):
            # [FUNCTION DOC] HELPER: Build a slice from the
            # arguments and assert slicing the current range
            # r matches slicing list(r) in elements and length.
            i = slice(start, stop, step)
            self.assertEqual(list(r[i]), list(r)[i])
            self.assertEqual(len(r[i]), len(list(r)[i]))
        for r in [range(10),
                  range(0),
                  range(1, 9, 3),
                  range(8, 0, -3),
                  range(sys.maxsize+1, sys.maxsize+10),
                  ]:
            check(0, 2)
            check(0, 20)
            check(1, 2)
            check(20, 30)
            check(-30, -20)
            check(-1, 100, 2)
            check(0, -1)
            check(-1, -3, -1)

    def test_contains(self):
        # [FUNCTION DOC] Checks membership for ascending, descending and strided ranges with ints,
        # equal floats, unequal floats and strings.
        r = range(10)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(9, -1, -1)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(0, 10, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertNotIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = range(9, -1, -2)
        self.assertNotIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)

    def test_reverse_iteration(self):
        # [FUNCTION DOC] Checks reversed iteration matches the reversed list for several ranges.
        for r in [range(10),
                  range(0),
                  range(1, 9, 3),
                  range(8, 0, -3),
                  range(sys.maxsize+1, sys.maxsize+10),
                  ]:
            self.assertEqual(list(reversed(r)), list(r)[::-1])

    def test_issue11845(self):
        # [FUNCTION DOC] Issue 11845: slices of a range using
        # many bounds and steps (including None,
        # negative, beyond length) must not raise errors.
        r = range(*slice(1, 18, 2).indices(20))
        values = {None, 0, 1, -1, 2, -2, 5, -5, 19, -19,
                  20, -20, 21, -21, 30, -30, 99, -99}
        for i in values:
            for j in values:
                for k in values - {0}:
                    r[i:j:k]

    def test_comparison(self):
        # [FUNCTION DOC] Checks range equality matches equality of the corresponding tuples,
        # not-equal is its negation, ranges are unequal to other sequence types, huge values
        # compare properly, and order comparisons raise TypeError.
        test_ranges = [range(0), range(0, -1), range(1, 1, 3),
                       range(1), range(5, 6), range(5, 6, 2),
                       range(5, 7, 2), range(2), range(0, 4, 2),
                       range(0, 5, 2), range(0, 6, 2)]
        test_tuples = list(map(tuple, test_ranges))

        # Check that equality of ranges matches equality of the corresponding
        # tuples for each pair from the test lists above.
        ranges_eq = [a == b for a in test_ranges for b in test_ranges]
        tuples_eq = [a == b for a in test_tuples for b in test_tuples]
        self.assertEqual(ranges_eq, tuples_eq)

        # Check that != correctly gives the logical negation of ==
        ranges_ne = [a != b for a in test_ranges for b in test_ranges]
        self.assertEqual(ranges_ne, [not x for x in ranges_eq])

        # Ranges are unequal to other types (even sequence types)
        self.assertIs(range(0) == (), False)
        # self.assertIs(() == range(0), False)
        self.assertIs(range(2) == [0, 1], False)

        # Huge integers aren't a problem.
        self.assertEqual(range(0, 2**100 - 1, 2),
                         range(0, 2**100, 2))
        self.assertNotEqual(range(0, 2**100, 2),
                            range(0, 2**100 + 1, 2))
        self.assertEqual(range(2**200, 2**201 - 2**99, 2**100),
                         range(2**200, 2**201, 2**100))
        self.assertNotEqual(range(2**200, 2**201, 2**100),
                            range(2**200, 2**201 + 1, 2**100))

        # Order comparisons are not implemented for ranges.
        with self.assertRaises(TypeError):
            range(0) < range(0)
        with self.assertRaises(TypeError):
            range(0) > range(0)
        with self.assertRaises(TypeError):
            range(0) <= range(0)
        with self.assertRaises(TypeError):
            range(0) >= range(0)


    def test_attributes(self):
        # [FUNCTION DOC] Checks start, stop, step attributes for several ranges, including bool
        # arguments.
        # test the start, stop and step attributes of range objects
        self.assert_attrs(range(0), 0, 0, 1)
        self.assert_attrs(range(10), 0, 10, 1)
        self.assert_attrs(range(-10), 0, -10, 1)
        self.assert_attrs(range(0, 10, 1), 0, 10, 1)
        self.assert_attrs(range(0, 10, 3), 0, 10, 3)
        self.assert_attrs(range(10, 0, -1), 10, 0, -1)
        self.assert_attrs(range(10, 0, -3), 10, 0, -3)
        self.assert_attrs(range(True), 0, 1, 1)
        self.assert_attrs(range(False, True), 0, 1, 1)
        self.assert_attrs(range(False, True, True), 0, 1, 1)

    def assert_attrs(self, rangeobj, start, stop, step):
        # [FUNCTION DOC] HELPER: Assert the range has the given
        # start, stop and step, and that each is
        # exactly an int.
        self.assertEqual(rangeobj.start, start)
        self.assertEqual(rangeobj.stop, stop)
        self.assertEqual(rangeobj.step, step)
        self.assertIs(type(rangeobj.start), int)
        self.assertIs(type(rangeobj.stop), int)
        self.assertIs(type(rangeobj.step), int)


if __name__ == "__main__":
    run_tests()
