# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_set ported to run under Dynamo's CPythonTestCase. Exercises set
# and frozenset: construction, set algebra operators and methods (union, intersection, difference,
# symmetric difference, in-place forms), comparison, hashing, pickling, copying, weakrefs,
# subclassing, iteration with mutation, bad hash/eq behavior, GC interaction, set views over
# dicts, and error handling in generator-driven inputs. Test classes are mixins (_Test* with a
# thetype/basetype pair) combined with the test case base to cover set, frozenset and subclasses.
# Import time side effect: inserts RedirectImportFinder at the front of the import meta path so
# some "test." helper modules resolve to standalone modules.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_set.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
)

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
    # [CLASS DOC] HELPER: import finder that is placed first on the import meta path. Module names
    # in the redirect_imports list are imported without the "test." prefix and registered under
    # the original name; all other names fall through to normal finders.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: finder hook. For a name in the redirect list, imports the
        # stripped name, stores it in the module table under fullname and returns its spec;
        # returns None if that import fails or the name is not listed. path and target are unused.
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

import unittest
from test import support
from test.support import warnings_helper
import gc
import weakref
import operator
import copy
import pickle
from random import randrange, shuffle
import warnings
import collections
import collections.abc
import itertools

class PassThru(Exception):
    # [CLASS DOC] HELPER: sentinel exception type that must propagate unchanged through set
    # operations (raised by the generator check_pass_thru).
    pass

def check_pass_thru():
    # [FUNCTION DOC] HELPER: generator function; when iterated, its first step raises
    # PassThru. Used to confirm that errors raised while iterating an argument propagate
    # from set operations. (The yield after the raise is unreachable and only makes the
    # function a generator.)
    raise PassThru
    yield 1

class BadCmp:
    # [CLASS DOC] HELPER: object with a constant hash of 1 whose equality always raises
    # RuntimeError, used to trigger errors during hash collision comparisons.
    def __hash__(self):
        # [FUNCTION DOC] HELPER: returns the constant 1 so all instances collide.
        return 1
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: always raises RuntimeError.
        raise RuntimeError

class ReprWrapper:
    # [CLASS DOC] HELPER: object whose repr is the repr of its value attribute, set by tests to
    # build self-referential containers.
    'Used to test self-referential repr() calls'
    def __repr__(self):
        # [FUNCTION DOC] HELPER: returns the repr of the value attribute (must be assigned
        # before use).
        return repr(self.value)

class HashCountingInt(int):
    # [CLASS DOC] HELPER: int subclass that counts, in hash_count, how many times it is hashed.
    'int-like object that counts the number of times __hash__ is called'
    def __init__(self, *args):
        # [FUNCTION DOC] HELPER: initializes the hash counter to zero (extra args are ignored).
        self.hash_count = 0
    def __hash__(self):
        # [FUNCTION DOC] HELPER: increments the counter and returns the normal int hash.
        self.hash_count += 1
        return int.__hash__(self)

class _TestJointOps:
    # [CLASS DOC] Mixin with tests common to set and frozenset. Subclasses provide thetype (the
    # class under test) and basetype (the type operations are expected to return). setUp creates
    # the sample string data used throughout.
    # Tests common to both set and frozenset

    def setUp(self):
        # [FUNCTION DOC] Creates fixtures: word, otherword, a letters string, self.s (thetype
        # built from word) and self.d (dict whose keys are the characters of word), then calls the
        # parent setUp.
        self.word = word = 'simsalabim'
        self.otherword = 'madagascar'
        self.letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.s = self.thetype(word)
        self.d = dict.fromkeys(word)
        super().setUp()

    def test_new_or_init(self):
        # [FUNCTION DOC] Checks the constructor rejects extra positional arguments and that
        # calling init on a set with an unknown keyword raises TypeError.
        self.assertRaises(TypeError, self.thetype, [], 2)
        self.assertRaises(TypeError, set().__init__, a=1)

    def test_uniquification(self):
        # [FUNCTION DOC] Checks duplicates are removed (sorted contents equal the dict keys),
        # errors from an input generator propagate and unhashable elements raise TypeError.
        actual = sorted(self.s)
        expected = sorted(self.d)
        self.assertEqual(actual, expected)
        self.assertRaises(PassThru, self.thetype, check_pass_thru())
        self.assertRaises(TypeError, self.thetype, [[]])

    def test_len(self):
        # [FUNCTION DOC] Checks len matches the number of unique characters in the fixture.
        self.assertEqual(len(self.s), len(self.d))

    def test_contains(self):
        # [FUNCTION DOC] Checks membership agrees with the reference dict for every letter,
        # that testing an unhashable list raises TypeError, and that a set can be found inside
        # a set of frozensets.
        for c in self.letters:
            self.assertEqual(c in self.s, c in self.d)
        self.assertRaises(TypeError, self.s.__contains__, [[]])
        s = self.thetype([frozenset(self.letters)])
        self.assertIn(self.thetype(self.letters), s)

    def test_union(self):
        # [FUNCTION DOC] Checks union with various iterable types and multiple arguments: element
        # membership, the original unchanged, result type basetype, error propagation from
        # generators and TypeError for unhashable elements.
        u = self.s.union(self.otherword)
        for c in self.letters:
            self.assertEqual(c in u, c in self.d or c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(u), self.basetype)
        self.assertRaises(PassThru, self.s.union, check_pass_thru())
        self.assertRaises(TypeError, self.s.union, [[]])
        for C in set, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype('abcba').union(C('cdc')), set('abcd'))
            self.assertEqual(self.thetype('abcba').union(C('efgfe')), set('abcefg'))
            self.assertEqual(self.thetype('abcba').union(C('ccb')), set('abc'))
            self.assertEqual(self.thetype('abcba').union(C('ef')), set('abcef'))
            self.assertEqual(self.thetype('abcba').union(C('ef'), C('fg')), set('abcefg'))

        # Issue #6573
        x = self.thetype()
        self.assertEqual(x.union(set([1]), x, set([2])), self.thetype([1, 2]))

    def test_or(self):
        # [FUNCTION DOC] Checks the or operator equals union for set and frozenset operands and
        # raises TypeError for other iterables such as strings.
        i = self.s.union(self.otherword)
        self.assertEqual(self.s | set(self.otherword), i)
        self.assertEqual(self.s | frozenset(self.otherword), i)
        try:
            self.s | self.otherword
        except TypeError:
            pass
        else:
            self.fail("s|t did not screen-out general iterables")

    def test_intersection(self):
        # [FUNCTION DOC] Checks intersection against the reference dict for various argument types
        # and multiple arguments, and that calling it with no arguments returns a distinct copy.
        # KNOWN ISSUE: the condition compares thetype with a new empty frozenset instance, which
        #     is never true, so the identity branch for frozensets is never exercised.
        i = self.s.intersection(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.intersection, check_pass_thru())
        for C in set, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype('abcba').intersection(C('cdc')), set('cc'))
            self.assertEqual(self.thetype('abcba').intersection(C('efgfe')), set(''))
            self.assertEqual(self.thetype('abcba').intersection(C('ccb')), set('bc'))
            self.assertEqual(self.thetype('abcba').intersection(C('ef')), set(''))
            self.assertEqual(self.thetype('abcba').intersection(C('cbcf'), C('bag')), set('b'))
        s = self.thetype('abcba')
        z = s.intersection()
        if self.thetype == frozenset():
            self.assertEqual(id(s), id(z))
        else:
            self.assertNotEqual(id(s), id(z))

    def test_isdisjoint(self):
        # [FUNCTION DOC] Checks isdisjoint against a pure Python reference over many argument
        # pairs and iterable types, and that the result is a real boolean.
        def f(s1, s2):
            # [FUNCTION DOC] HELPER: reference isdisjoint: true when the intersection of the two
            # inputs is empty.
            'Pure python equivalent of isdisjoint()'
            return not set(s1).intersection(s2)
        for larg in '', 'a', 'ab', 'abc', 'ababac', 'cdc', 'cc', 'efgfe', 'ccb', 'ef':
            s1 = self.thetype(larg)
            for rarg in '', 'a', 'ab', 'abc', 'ababac', 'cdc', 'cc', 'efgfe', 'ccb', 'ef':
                for C in set, frozenset, dict.fromkeys, str, list, tuple:
                    s2 = C(rarg)
                    actual = s1.isdisjoint(s2)
                    expected = f(s1, s2)
                    self.assertEqual(actual, expected)
                    self.assertTrue(actual is True or actual is False)

    def test_and(self):
        # [FUNCTION DOC] Checks the and operator equals intersection for set and frozenset
        # operands and raises TypeError for other iterables.
        i = self.s.intersection(self.otherword)
        self.assertEqual(self.s & set(self.otherword), i)
        self.assertEqual(self.s & frozenset(self.otherword), i)
        try:
            self.s & self.otherword
        except TypeError:
            pass
        else:
            self.fail("s&t did not screen-out general iterables")

    def test_difference(self):
        # [FUNCTION DOC] Checks difference semantics: membership, result type, original unchanged,
        # error propagation, multiple and zero arguments across iterable types.
        i = self.s.difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c not in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference, [[]])
        for C in set, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype('abcba').difference(C('cdc')), set('ab'))
            self.assertEqual(self.thetype('abcba').difference(C('efgfe')), set('abc'))
            self.assertEqual(self.thetype('abcba').difference(C('ccb')), set('a'))
            self.assertEqual(self.thetype('abcba').difference(C('ef')), set('abc'))
            self.assertEqual(self.thetype('abcba').difference(), set('abc'))
            self.assertEqual(self.thetype('abcba').difference(C('a'), C('b')), set('c'))

    def test_sub(self):
        # [FUNCTION DOC] Checks the subtraction operator equals difference for set and frozenset
        # and raises TypeError for other iterables.
        i = self.s.difference(self.otherword)
        self.assertEqual(self.s - set(self.otherword), i)
        self.assertEqual(self.s - frozenset(self.otherword), i)
        try:
            self.s - self.otherword
        except TypeError:
            pass
        else:
            self.fail("s-t did not screen-out general iterables")

    def test_symmetric_difference(self):
        # [FUNCTION DOC] Checks symmetric_difference: membership, result type, original unchanged,
        # error propagation and results for several argument types.
        i = self.s.symmetric_difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, (c in self.d) ^ (c in self.otherword))
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.symmetric_difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.symmetric_difference, [[]])
        for C in set, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype('abcba').symmetric_difference(C('cdc')), set('abd'))
            self.assertEqual(self.thetype('abcba').symmetric_difference(C('efgfe')), set('abcefg'))
            self.assertEqual(self.thetype('abcba').symmetric_difference(C('ccb')), set('a'))
            self.assertEqual(self.thetype('abcba').symmetric_difference(C('ef')), set('abcef'))

    def test_xor(self):
        # [FUNCTION DOC] Checks the xor operator equals symmetric_difference for sets and raises
        # TypeError for other iterables.
        i = self.s.symmetric_difference(self.otherword)
        self.assertEqual(self.s ^ set(self.otherword), i)
        self.assertEqual(self.s ^ frozenset(self.otherword), i)
        try:
            self.s ^ self.otherword
        except TypeError:
            pass
        else:
            self.fail("s^t did not screen-out general iterables")

    def test_equality(self):
        # [FUNCTION DOC] Checks equality with set and frozenset of the same or different contents
        # and inequality with a plain string.
        self.assertEqual(self.s, set(self.word))
        self.assertEqual(self.s, frozenset(self.word))
        self.assertEqual(self.s == self.word, False)
        self.assertNotEqual(self.s, set(self.otherword))
        self.assertNotEqual(self.s, frozenset(self.otherword))
        self.assertEqual(self.s != self.word, True)

    def test_setOfFrozensets(self):
        # [FUNCTION DOC] Checks that equal frozensets collapse in a set built from a map over
        # strings (three distinct results expected).
        t = map(frozenset, ['abcdef', 'bcd', 'bdcb', 'fed', 'fedccba'])
        s = self.thetype(t)
        self.assertEqual(len(s), 3)

    def test_sub_and_super(self):
        # [FUNCTION DOC] Checks proper subset, subset, proper superset and superset operators for
        # related and unrelated sets, and issubset and issuperset on non-set iterables.
        p, q, r = map(self.thetype, ['ab', 'abcde', 'def'])
        self.assertTrue(p < q)
        self.assertTrue(p <= q)
        self.assertTrue(q <= q)
        self.assertTrue(q > p)
        self.assertTrue(q >= p)
        self.assertFalse(q < r)
        self.assertFalse(q <= r)
        self.assertFalse(q > r)
        self.assertFalse(q >= r)
        self.assertTrue(set('a').issubset('abc'))
        self.assertTrue(set('abc').issuperset('a'))
        self.assertFalse(set('a').issubset('cbs'))
        self.assertFalse(set('cbs').issuperset('a'))

    def test_pickling(self):
        # [FUNCTION DOC] Pickles and unpickles self.s with every protocol and compares; for
        # subclasses it also sets extra instance attributes and checks they round-trip, then
        # removes them.
        for i in range(pickle.HIGHEST_PROTOCOL + 1):
            if type(self.s) not in (set, frozenset):
                self.s.x = ['x']
                self.s.z = ['z']
            p = pickle.dumps(self.s, i)
            dup = pickle.loads(p)
            self.assertEqual(self.s, dup, "%s != %s" % (self.s, dup))
            if type(self.s) not in (set, frozenset):
                self.assertEqual(self.s.x, dup.x)
                self.assertEqual(self.s.z, dup.z)
                self.assertFalse(hasattr(self.s, 'y'))
                del self.s.x, self.s.z

    def test_iterator_pickling(self):
        # [FUNCTION DOC] Pickles iterators of self.s for each protocol, checks they unpickle to
        # iterators that rebuild the set, and that a partly consumed iterator gives the remainder.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            itorg = iter(self.s)
            data = self.thetype(self.s)
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # Set iterators unpickle as list iterators due to the
            # undefined order of set items.
            # self.assertEqual(type(itorg), type(it))
            self.assertIsInstance(it, collections.abc.Iterator)
            self.assertEqual(self.thetype(it), data)

            it = pickle.loads(d)
            try:
                drop = next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(self.thetype(it), data - self.thetype((drop,)))

    def test_deepcopy(self):
        # [FUNCTION DOC] Checks deepcopy of a set creates a new set whose element is the result of
        # the element's own deepcopy hook (value incremented by one).
        class Tracer:
            # [CLASS DOC] HELPER: hashable value holder whose deepcopy returns a new holder with
            # value plus one.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: stores value.
                self.value = value
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns value as hash.
                return self.value
            def __deepcopy__(self, memo=None):
                # [FUNCTION DOC] HELPER: returns a new Tracer with value plus one (memo ignored).
                return Tracer(self.value + 1)
        t = Tracer(10)
        s = self.thetype([t])
        dup = copy.deepcopy(s)
        self.assertNotEqual(id(s), id(dup))
        for elem in dup:
            newt = elem
        self.assertNotEqual(id(t), id(newt))
        self.assertEqual(t.value + 1, newt.value)

    def test_gc(self):
        # [FUNCTION DOC] Builds a set of 1000 objects that reference the set and themselves,
        # creating cycles for the garbage collector. Makes no assertions.
        # Create a nest of cycles to exercise overall ref count check
        class A:
            # [CLASS DOC] HELPER: empty class for cycle objects.
            pass
        s = set(A() for i in range(1000))
        for elem in s:
            elem.cycle = s
            elem.sub = elem
            elem.set = set([elem])

    def test_subclass_with_custom_hash(self):
        # [FUNCTION DOC] Checks a subclass defining a custom hash can be added to, removed from
        # and discarded from a plain set (bug 1257731).
        # Bug #1257731
        class H(self.thetype):
            # [CLASS DOC] HELPER: subclass of thetype hashing by object id.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns the masked object id as hash.
                return int(id(self) & 0x7fffffff)
        s=H()
        f=set()
        f.add(s)
        self.assertIn(s, f)
        f.remove(s)
        f.add(s)
        f.discard(s)

    def test_badcmp(self):
        # [FUNCTION DOC] Checks that errors from equality during insertion, lookup, add, discard
        # and remove propagate as RuntimeError when elements have colliding hashes.
        s = self.thetype([BadCmp()])
        # Detect comparison errors during insertion and lookup
        self.assertRaises(RuntimeError, self.thetype, [BadCmp(), BadCmp()])
        self.assertRaises(RuntimeError, s.__contains__, BadCmp())
        # Detect errors during mutating operations
        if hasattr(s, 'add'):
            self.assertRaises(RuntimeError, s.add, BadCmp())
            self.assertRaises(RuntimeError, s.discard, BadCmp())
            self.assertRaises(RuntimeError, s.remove, BadCmp())

    def test_cyclical_repr(self):
        # [FUNCTION DOC] Checks repr of a set containing an object that contains the set:
        # recursion is abbreviated with the type name and ellipsis.
        w = ReprWrapper()
        s = self.thetype([w])
        w.value = s
        if self.thetype == set:
            self.assertEqual(repr(s), '{set(...)}')
        else:
            name = repr(s).partition('(')[0]    # strip class name
            self.assertEqual(repr(s), '%s({%s(...)})' % (name, name))

    def test_do_not_rehash_dict_keys(self):
        # [FUNCTION DOC] Checks building sets and set operations from a dict (or its set
        # conversions) reuses stored hashes: the total hash call count stays at ten.
        n = 10
        d = dict.fromkeys(map(HashCountingInt, range(n)))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s = self.thetype(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s.difference(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        if hasattr(s, 'symmetric_difference_update'):
            s.symmetric_difference_update(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d2 = dict.fromkeys(set(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d), 123)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        self.assertEqual(d3, dict.fromkeys(d, 123))

    def test_container_iterator(self):
        # [FUNCTION DOC] Checks a cycle through a set iterator stored on an element can be garbage
        # collected (weak reference becomes dead).
        # Bug #3680: tp_traverse was not implemented for set iterator object
        class C(object):
            # [CLASS DOC] HELPER: empty class whose instance is referenced by a cycle.
            pass
        obj = C()
        ref = weakref.ref(obj)
        container = set([obj, 1])
        obj.x = iter(container)
        del obj, container
        gc.collect()
        self.assertTrue(ref() is None, "Cycle was not collected")

    def test_free_after_iterating(self):
        # [FUNCTION DOC] Delegates to the support helper that checks the container is freed after
        # its iterator has been exhausted.
        support.check_free_after_iterating(self, iter, self.thetype)

class TestSet(_TestJointOps, __TestCase):
    # [CLASS DOC] Runs the shared tests and mutable-set tests for the built-in set (thetype and
    # basetype are both set).
    thetype = set
    basetype = set

    def test_init(self):
        # [FUNCTION DOC] Checks re-initializing a set replaces its contents and invalid argument
        # counts raise TypeError.
        s = self.thetype()
        s.__init__(self.word)
        self.assertEqual(s, set(self.word))
        s.__init__(self.otherword)
        self.assertEqual(s, set(self.otherword))
        self.assertRaises(TypeError, s.__init__, s, 2)
        self.assertRaises(TypeError, s.__init__, 1)

    def test_constructor_identity(self):
        # [FUNCTION DOC] Checks constructing a set from a set returns a new object.
        s = self.thetype(range(3))
        t = self.thetype(s)
        self.assertNotEqual(id(s), id(t))

    def test_set_literal(self):
        # [FUNCTION DOC] Checks a literal with braces equals the constructor result.
        s = set([1,2,3])
        t = {1,2,3}
        self.assertEqual(s, t)

    def test_set_literal_insertion_order(self):
        # [FUNCTION DOC] Checks a literal with equal elements 1, 1.0 and True keeps only the first
        # (an int).
        # SF Issue #26020 -- Expect left to right insertion
        s = {1, 1.0, True}
        self.assertEqual(len(s), 1)
        stored_value = s.pop()
        self.assertEqual(type(stored_value), int)

    def test_set_literal_evaluation_order(self):
        # [FUNCTION DOC] Checks elements in a set literal are evaluated left to right.
        # Expect left to right expression evaluation
        events = []
        def record(obj):
            # [FUNCTION DOC] HELPER: records its argument in the enclosing events list.
            events.append(obj)
        s = {record(1), record(2), record(3)}
        self.assertEqual(events, [1, 2, 3])

    def test_hash(self):
        # [FUNCTION DOC] Checks hashing a mutable set raises TypeError.
        self.assertRaises(TypeError, hash, self.s)

    def test_clear(self):
        # [FUNCTION DOC] Checks clear empties the set.
        self.s.clear()
        self.assertEqual(self.s, set())
        self.assertEqual(len(self.s), 0)

    def test_copy(self):
        # [FUNCTION DOC] Checks copy is equal, a distinct object, and of type basetype.
        dup = self.s.copy()
        self.assertEqual(self.s, dup)
        self.assertNotEqual(id(self.s), id(dup))
        self.assertEqual(type(dup), self.basetype)

    def test_add(self):
        # [FUNCTION DOC] Checks add inserts, is idempotent for existing items and raises TypeError
        # for unhashable items.
        self.s.add('Q')
        self.assertIn('Q', self.s)
        dup = self.s.copy()
        self.s.add('Q')
        self.assertEqual(self.s, dup)
        self.assertRaises(TypeError, self.s.add, [])

    def test_remove(self):
        # [FUNCTION DOC] Checks remove deletes the element, raises KeyError for missing and
        # TypeError for unhashable items, and treats a set key like a frozenset.
        self.s.remove('a')
        self.assertNotIn('a', self.s)
        self.assertRaises(KeyError, self.s.remove, 'Q')
        self.assertRaises(TypeError, self.s.remove, [])
        s = self.thetype([frozenset(self.word)])
        self.assertIn(self.thetype(self.word), s)
        s.remove(self.thetype(self.word))
        self.assertNotIn(self.thetype(self.word), s)
        self.assertRaises(KeyError, self.s.remove, self.thetype(self.word))

    def test_remove_keyerror_unpacking(self):
        # [FUNCTION DOC] Checks the KeyError raised by remove carries the original key as its
        # argument (bug 1576657).
        # https://bugs.python.org/issue1576657
        for v1 in ['Q', (1,)]:
            try:
                self.s.remove(v1)
            except KeyError as e:
                v2 = e.args[0]
                self.assertEqual(v1, v2)
            else:
                self.fail()

    def test_remove_keyerror_set(self):
        # [FUNCTION DOC] Checks remove of a missing set-typed key raises KeyError whose argument
        # is the very same key object.
        key = self.thetype([3, 4])
        try:
            self.s.remove(key)
        except KeyError as e:
            self.assertTrue(e.args[0] is key,
                         "KeyError should be {0}, not {1}".format(key,
                                                                  e.args[0]))
        else:
            self.fail()

    def test_discard(self):
        # [FUNCTION DOC] Checks discard removes silently, ignores missing items, raises TypeError
        # for unhashable ones and accepts equivalent set keys.
        self.s.discard('a')
        self.assertNotIn('a', self.s)
        self.s.discard('Q')
        self.assertRaises(TypeError, self.s.discard, [])
        s = self.thetype([frozenset(self.word)])
        self.assertIn(self.thetype(self.word), s)
        s.discard(self.thetype(self.word))
        self.assertNotIn(self.thetype(self.word), s)
        s.discard(self.thetype(self.word))

    def test_pop(self):
        # [FUNCTION DOC] Checks pop removes every element in turn and raises KeyError when empty.
        for i in range(len(self.s)):
            elem = self.s.pop()
            self.assertNotIn(elem, self.s)
        self.assertRaises(KeyError, self.s.pop)

    def test_update(self):
        # [FUNCTION DOC] Checks update returns None, adds elements from various iterables and
        # multiple arguments, and propagates errors from generators or unhashables.
        retval = self.s.update(self.otherword)
        self.assertEqual(retval, None)
        for c in (self.word + self.otherword):
            self.assertIn(c, self.s)
        self.assertRaises(PassThru, self.s.update, check_pass_thru())
        self.assertRaises(TypeError, self.s.update, [[]])
        for p, q in (('cdc', 'abcd'), ('efgfe', 'abcefg'), ('ccb', 'abc'), ('ef', 'abcef')):
            for C in set, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype('abcba')
                self.assertEqual(s.update(C(p)), None)
                self.assertEqual(s, set(q))
        for p in ('cdc', 'efgfe', 'ccb', 'ef', 'abcda'):
            q = 'ahi'
            for C in set, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype('abcba')
                self.assertEqual(s.update(C(p), C(q)), None)
                self.assertEqual(s, set(s) | set(p) | set(q))

    def test_ior(self):
        # [FUNCTION DOC] Checks the in-place or operator adds the other set's elements.
        self.s |= set(self.otherword)
        for c in (self.word + self.otherword):
            self.assertIn(c, self.s)

    def test_intersection_update(self):
        # [FUNCTION DOC] Checks intersection_update returns None, keeps only common elements,
        # accepts multiple arguments and propagates errors.
        retval = self.s.intersection_update(self.otherword)
        self.assertEqual(retval, None)
        for c in (self.word + self.otherword):
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.intersection_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.intersection_update, [[]])
        for p, q in (('cdc', 'c'), ('efgfe', ''), ('ccb', 'bc'), ('ef', '')):
            for C in set, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype('abcba')
                self.assertEqual(s.intersection_update(C(p)), None)
                self.assertEqual(s, set(q))
                ss = 'abcba'
                s = self.thetype(ss)
                t = 'cbc'
                self.assertEqual(s.intersection_update(C(p), C(t)), None)
                self.assertEqual(s, set('abcba')&set(p)&set(t))

    def test_iand(self):
        # [FUNCTION DOC] Checks the in-place and operator keeps only common elements.
        self.s &= set(self.otherword)
        for c in (self.word + self.otherword):
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_difference_update(self):
        # [FUNCTION DOC] Checks difference_update returns None, removes elements of the arguments
        # (none, one or several), and propagates errors for unhashables and generators.
        retval = self.s.difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in (self.word + self.otherword):
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.difference_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference_update, [[]])
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (('cdc', 'ab'), ('efgfe', 'abc'), ('ccb', 'a'), ('ef', 'abc')):
            for C in set, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype('abcba')
                self.assertEqual(s.difference_update(C(p)), None)
                self.assertEqual(s, set(q))

                s = self.thetype('abcdefghih')
                s.difference_update()
                self.assertEqual(s, self.thetype('abcdefghih'))

                s = self.thetype('abcdefghih')
                s.difference_update(C('aba'))
                self.assertEqual(s, self.thetype('cdefghih'))

                s = self.thetype('abcdefghih')
                s.difference_update(C('cdc'), C('aba'))
                self.assertEqual(s, self.thetype('efghih'))

    def test_isub(self):
        # [FUNCTION DOC] Checks the in-place subtraction operator removes the other
        # set's elements.
        self.s -= set(self.otherword)
        for c in (self.word + self.otherword):
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_symmetric_difference_update(self):
        # [FUNCTION DOC] Checks symmetric_difference_update returns None, keeps elements in
        # exactly one operand and propagates errors.
        retval = self.s.symmetric_difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in (self.word + self.otherword):
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.symmetric_difference_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (('cdc', 'abd'), ('efgfe', 'abcefg'), ('ccb', 'a'), ('ef', 'abcef')):
            for C in set, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype('abcba')
                self.assertEqual(s.symmetric_difference_update(C(p)), None)
                self.assertEqual(s, set(q))

    def test_ixor(self):
        # [FUNCTION DOC] Checks the in-place xor operator keeps elements in exactly one operand.
        self.s ^= set(self.otherword)
        for c in (self.word + self.otherword):
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_inplace_on_self(self):
        # [FUNCTION DOC] Checks applying each in-place operator with the set as both operands
        # gives the expected result (unchanged for or and and, empty for subtraction and xor).
        t = self.s.copy()
        t |= t
        self.assertEqual(t, self.s)
        t &= t
        self.assertEqual(t, self.s)
        t -= t
        self.assertEqual(t, self.thetype())
        t = self.s.copy()
        t ^= t
        self.assertEqual(t, self.thetype())

    def test_weakref(self):
        # [FUNCTION DOC] Checks a weak proxy to a set works while alive and raises ReferenceError
        # after the set is released and collected.
        s = self.thetype('gallahad')
        p = weakref.proxy(s)
        self.assertEqual(str(p), str(s))
        s = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, str, p)

    def test_rich_compare(self):
        # [FUNCTION DOC] Checks that comparing a set to an object with custom comparison methods
        # falls back to the reflected comparison on the object for less, greater, less equal and
        # greater equal.
        class TestRichSetCompare:
            # [CLASS DOC] HELPER: records which rich comparison methods were invoked, each
            # returning False.
            def __gt__(self, some_set):
                # [FUNCTION DOC] HELPER: records that the greater-than hook ran; returns False.
                self.gt_called = True
                return False
            def __lt__(self, some_set):
                # [FUNCTION DOC] HELPER: records that the less-than hook ran; returns False.
                self.lt_called = True
                return False
            def __ge__(self, some_set):
                # [FUNCTION DOC] HELPER: records that the greater-equal hook ran; returns False.
                self.ge_called = True
                return False
            def __le__(self, some_set):
                # [FUNCTION DOC] HELPER: records that the less-equal hook ran; returns False.
                self.le_called = True
                return False

        # This first tries the builtin rich set comparison, which doesn't know
        # how to handle the custom object. Upon returning NotImplemented, the
        # corresponding comparison on the right object is invoked.
        myset = {1, 2, 3}

        myobj = TestRichSetCompare()
        myset < myobj
        self.assertTrue(myobj.gt_called)

        myobj = TestRichSetCompare()
        myset > myobj
        self.assertTrue(myobj.lt_called)

        myobj = TestRichSetCompare()
        myset <= myobj
        self.assertTrue(myobj.ge_called)

        myobj = TestRichSetCompare()
        myset >= myobj
        self.assertTrue(myobj.le_called)


class SetSubclass(set):
    # [CLASS DOC] HELPER: trivial set subclass used as thetype for subclass tests.
    pass

class TestSetSubclass(TestSet):
    # [CLASS DOC] Runs the whole TestSet suite with a set subclass as the type under test; adds
    # checks on keyword arguments for subclasses.
    thetype = SetSubclass
    basetype = set

    def test_keywords_in_subclass(self):
        # [FUNCTION DOC] Checks subclasses of set: plain subclass rejects keyword arguments, one
        # with its own init accepts them, one with its own new accepts only positional use and
        # rejects keywords at the call (bug 43413).
        class subclass(set):
            # [CLASS DOC] HELPER: empty set subclass.
            pass
        u = subclass([1, 2])
        self.assertIs(type(u), subclass)
        self.assertEqual(set(u), {1, 2})
        with self.assertRaises(TypeError):
            subclass(sequence=())

        class subclass_with_init(set):
            # [CLASS DOC] HELPER: set subclass whose init takes an extra optional argument stored
            # as newarg.
            def __init__(self, arg, newarg=None):
                # [FUNCTION DOC] HELPER: initializes the set from arg and stores newarg.
                super().__init__(arg)
                self.newarg = newarg
        u = subclass_with_init([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(set(u), {1, 2})
        self.assertEqual(u.newarg, 3)

        class subclass_with_new(set):
            # [CLASS DOC] HELPER: set subclass whose new takes an optional newarg stored on
            # the instance.
            def __new__(cls, arg, newarg=None):
                # [FUNCTION DOC] HELPER: creates the instance from arg, records newarg and
                # returns it.
                self = super().__new__(cls, arg)
                self.newarg = newarg
                return self
        u = subclass_with_new([1, 2])
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(set(u), {1, 2})
        self.assertIsNone(u.newarg)
        # disallow kwargs in __new__ only (https://bugs.python.org/issue43413#msg402000)
        with self.assertRaises(TypeError):
            subclass_with_new([1, 2], newarg=3)


class TestFrozenSet(_TestJointOps, __TestCase):
    # [CLASS DOC] Runs the shared tests for the built-in frozenset (thetype and basetype are both
    # frozenset) and adds frozenset specific checks (hashing, identity of copies).
    thetype = frozenset
    basetype = frozenset

    def test_init(self):
        # [FUNCTION DOC] Checks calling init again on a frozenset does not change its contents.
        s = self.thetype(self.word)
        s.__init__(self.otherword)
        self.assertEqual(s, set(self.word))

    def test_constructor_identity(self):
        # [FUNCTION DOC] Checks constructing a frozenset from a frozenset returns the same object.
        s = self.thetype(range(3))
        t = self.thetype(s)
        self.assertEqual(id(s), id(t))

    def test_hash(self):
        # [FUNCTION DOC] Checks frozenset hash is order independent: equal contents hash equal,
        # and 200 shuffles of a 100 element sequence all give one hash value.
        self.assertEqual(hash(self.thetype('abcdeb')),
                         hash(self.thetype('ebecda')))

        # make sure that all permutations give the same hash value
        n = 100
        seq = [randrange(n) for i in range(n)]
        results = set()
        for i in range(200):
            shuffle(seq)
            results.add(hash(self.thetype(seq)))
        self.assertEqual(len(results), 1)

    def test_copy(self):
        # [FUNCTION DOC] Checks copy of a frozenset returns the very same object.
        dup = self.s.copy()
        self.assertEqual(id(self.s), id(dup))

    def test_frozen_as_dictkey(self):
        # [FUNCTION DOC] Checks equal frozensets built in different orders are distinct objects,
        # equal, and interchangeable as dict keys.
        seq = list(range(10)) + list('abcdefg') + ['apple']
        key1 = self.thetype(seq)
        key2 = self.thetype(reversed(seq))
        self.assertEqual(key1, key2)
        self.assertNotEqual(id(key1), id(key2))
        d = {}
        d[key1] = 42
        self.assertEqual(d[key2], 42)

    def test_hash_caching(self):
        # [FUNCTION DOC] Checks hashing the same frozenset twice gives the same value.
        f = self.thetype('abcdcda')
        self.assertEqual(hash(f), hash(f))

    @skipIfTorchDynamo("slow")
    def test_hash_effectiveness(self):
        # [FUNCTION DOC] Checks frozenset hash spreads values: all 2**n subsets of a tiny universe
        # get distinct hashes, and for powersets of ranges and of nested empty sets the number of
        # distinct low bits exceeds a quarter of the table size. Skipped under Dynamo as slow.
        n = 2
        hashvalues = set()
        addhashvalue = hashvalues.add
        elemmasks = [(i+1, 1<<i) for i in range(n)]
        for i in range(2**n):
            addhashvalue(hash(frozenset([e for e, m in elemmasks if m&i])))
        self.assertEqual(len(hashvalues), 2**n)

        def zf_range(n):
            # [FUNCTION DOC] HELPER: builds a list of n frozensets modeling natural numbers (each
            # is the frozenset of all earlier ones), starting from the empty frozenset.
            # https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers
            nums = [frozenset()]
            for i in range(n-1):
                num = frozenset(nums)
                nums.append(num)
            return nums[:n]

        def powerset(s):
            # [FUNCTION DOC] HELPER: generator yielding every subset of s as a frozenset, ordered
            # by size.
            for i in range(len(s)+1):
                yield from map(frozenset, itertools.combinations(s, i))

        for n in range(18):
            t = 2 ** n
            mask = t - 1
            for nums in (range, zf_range):
                u = len({h & mask for h in map(hash, powerset(nums(n)))})
                self.assertGreater(4*u, t)

class FrozenSetSubclass(frozenset):
    # [CLASS DOC] HELPER: trivial frozenset subclass used as thetype in subclass tests.
    pass

class TestFrozenSetSubclass(TestFrozenSet):
    # [CLASS DOC] Runs the frozenset suite with a subclass as type under test, with checks on
    # keyword arguments and on subclass copies being distinct objects.
    thetype = FrozenSetSubclass
    basetype = frozenset

    def test_keywords_in_subclass(self):
        # [FUNCTION DOC] Checks frozenset subclasses: plain one rejects keyword arguments, one
        # with init accepts them and stores the extra value, one with new accepts them as well.
        class subclass(frozenset):
            # [CLASS DOC] HELPER: empty frozenset subclass.
            pass
        u = subclass([1, 2])
        self.assertIs(type(u), subclass)
        self.assertEqual(set(u), {1, 2})
        with self.assertRaises(TypeError):
            subclass(sequence=())

        class subclass_with_init(frozenset):
            # [CLASS DOC] HELPER: frozenset subclass whose init stores an optional newarg.
            def __init__(self, arg, newarg=None):
                # [FUNCTION DOC] HELPER: stores newarg on the instance (argument arg is
                # ignored here).
                self.newarg = newarg
        u = subclass_with_init([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_init)
        self.assertEqual(set(u), {1, 2})
        self.assertEqual(u.newarg, 3)

        class subclass_with_new(frozenset):
            # [CLASS DOC] HELPER: frozenset subclass whose new stores an optional newarg.
            def __new__(cls, arg, newarg=None):
                # [FUNCTION DOC] HELPER: creates the instance from arg, records newarg and
                # returns it.
                self = super().__new__(cls, arg)
                self.newarg = newarg
                return self
        u = subclass_with_new([1, 2], newarg=3)
        self.assertIs(type(u), subclass_with_new)
        self.assertEqual(set(u), {1, 2})
        self.assertEqual(u.newarg, 3)

    def test_constructor_identity(self):
        # [FUNCTION DOC] Checks that building a subclass instance from another subclass instance
        # gives a distinct object (no identity shortcut).
        s = self.thetype(range(3))
        t = self.thetype(s)
        self.assertNotEqual(id(s), id(t))

    def test_copy(self):
        # [FUNCTION DOC] Checks copy of a subclass instance returns a different object.
        dup = self.s.copy()
        self.assertNotEqual(id(self.s), id(dup))

    def test_nested_empty_constructor(self):
        # [FUNCTION DOC] Checks wrapping an empty instance in another instance gives an equal
        # empty set.
        s = self.thetype()
        t = self.thetype(s)
        self.assertEqual(s, t)

    def test_singleton_empty_frozenset(self):
        # [FUNCTION DOC] Checks that empty instances of the subclass are never shared singletons:
        # all fifteen constructions have different ids.
        Frozenset = self.thetype
        f = frozenset()
        F = Frozenset()
        efs = [Frozenset(), Frozenset([]), Frozenset(()), Frozenset(''),
               Frozenset(), Frozenset([]), Frozenset(()), Frozenset(''),
               Frozenset(range(0)), Frozenset(Frozenset()),
               Frozenset(frozenset()), f, F, Frozenset(f), Frozenset(F)]
        # All empty frozenset subclass instances should have different ids
        self.assertEqual(len(set(map(id, efs))), len(efs))


class SetSubclassWithSlots(set):
    # [CLASS DOC] HELPER: set subclass declaring slots x, y and a dict.
    __slots__ = ('x', 'y', '__dict__')

class TestSetSubclassWithSlots(__TestCase):
    # [CLASS DOC] Runs only the pickling test for a slotted set subclass; defines its own fixtures
    # in setUp and reuses the pickling test from the joint mixin.
    thetype = SetSubclassWithSlots
    test_pickling = _TestJointOps.test_pickling

    def setUp(self):
        # [FUNCTION DOC] Creates the same sample fixtures as the mixin (word, otherword, letters,
        # s, d) and calls the parent setUp.
        self.word = word = 'simsalabim'
        self.otherword = 'madagascar'
        self.letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.s = self.thetype(word)
        self.d = dict.fromkeys(word)
        super().setUp()

class FrozenSetSubclassWithSlots(frozenset):
    # [CLASS DOC] HELPER: frozenset subclass declaring slots x, y and a dict.
    __slots__ = ('x', 'y', '__dict__')

class TestFrozenSetSubclassWithSlots(TestSetSubclassWithSlots):
    # [CLASS DOC] Same pickling check for a slotted frozenset subclass; only thetype differs.
    thetype = FrozenSetSubclassWithSlots

# Tests taken from test_sets.py =============================================

empty_set = set()

#==============================================================================

class _TestBasicOps:
    # [CLASS DOC] Mixin of basic operation tests parameterized by fixtures set in setUp: case,
    # values, set, dup (a duplicate set), length and repr.

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr of the set equals the expected text when one
        # is configured.
        if self.repr is not None:
            self.assertEqual(repr(self.set), self.repr)

    def check_repr_against_values(self):
        # [FUNCTION DOC] HELPER: checks the set repr is a brace delimited list whose items,
        # sorted, equal the sorted reprs of the configured values (order of items is not fixed).
        text = repr(self.set)
        self.assertTrue(text.startswith('{'))
        self.assertTrue(text.endswith('}'))

        result = text[1:-1].split(', ')
        result.sort()
        sorted_repr_values = [repr(value) for value in self.values]
        sorted_repr_values.sort()
        self.assertEqual(result, sorted_repr_values)

    def test_length(self):
        # [FUNCTION DOC] Checks len equals the configured length.
        self.assertEqual(len(self.set), self.length)

    def test_self_equality(self):
        # [FUNCTION DOC] Checks a set equals itself.
        self.assertEqual(self.set, self.set)

    def test_equivalent_equality(self):
        # [FUNCTION DOC] Checks the set equals its duplicate.
        self.assertEqual(self.set, self.dup)

    def test_copy(self):
        # [FUNCTION DOC] Checks copy equals the duplicate.
        self.assertEqual(self.set.copy(), self.dup)

    def test_self_union(self):
        # [FUNCTION DOC] Checks the union of the set with itself equals the duplicate.
        result = self.set | self.set
        self.assertEqual(result, self.dup)

    def test_empty_union(self):
        # [FUNCTION DOC] Checks union with the empty set on the right leaves contents unchanged.
        result = self.set | empty_set
        self.assertEqual(result, self.dup)

    def test_union_empty(self):
        # [FUNCTION DOC] Checks union with the empty set on the left leaves contents unchanged.
        result = empty_set | self.set
        self.assertEqual(result, self.dup)

    def test_self_intersection(self):
        # [FUNCTION DOC] Checks intersection of the set with itself equals the duplicate.
        result = self.set & self.set
        self.assertEqual(result, self.dup)

    def test_empty_intersection(self):
        # [FUNCTION DOC] Checks intersection with the empty set on the right is empty.
        result = self.set & empty_set
        self.assertEqual(result, empty_set)

    def test_intersection_empty(self):
        # [FUNCTION DOC] Checks intersection with the empty set on the left is empty.
        result = empty_set & self.set
        self.assertEqual(result, empty_set)

    def test_self_isdisjoint(self):
        # [FUNCTION DOC] Checks a set is disjoint from itself exactly when it is empty.
        result = self.set.isdisjoint(self.set)
        self.assertEqual(result, not self.set)

    def test_empty_isdisjoint(self):
        # [FUNCTION DOC] Checks any set is disjoint with the empty set.
        result = self.set.isdisjoint(empty_set)
        self.assertEqual(result, True)

    def test_isdisjoint_empty(self):
        # [FUNCTION DOC] Checks the empty set is disjoint with any set.
        result = empty_set.isdisjoint(self.set)
        self.assertEqual(result, True)

    def test_self_symmetric_difference(self):
        # [FUNCTION DOC] Checks symmetric difference of a set with itself is empty.
        result = self.set ^ self.set
        self.assertEqual(result, empty_set)

    def test_empty_symmetric_difference(self):
        # [FUNCTION DOC] Checks symmetric difference with the empty set returns the
        # original contents.
        result = self.set ^ empty_set
        self.assertEqual(result, self.set)

    def test_self_difference(self):
        # [FUNCTION DOC] Checks a set minus itself is empty.
        result = self.set - self.set
        self.assertEqual(result, empty_set)

    def test_empty_difference(self):
        # [FUNCTION DOC] Checks subtracting the empty set leaves contents unchanged.
        result = self.set - empty_set
        self.assertEqual(result, self.dup)

    def test_empty_difference_rev(self):
        # [FUNCTION DOC] Checks the empty set minus any set is empty.
        result = empty_set - self.set
        self.assertEqual(result, empty_set)

    def test_iteration(self):
        # [FUNCTION DOC] Checks every iterated item belongs to the configured values and the
        # iterator's length hint equals the set size.
        for v in self.set:
            self.assertIn(v, self.values)
        setiter = iter(self.set)
        self.assertEqual(setiter.__length_hint__(), len(self.set))

    def test_pickling(self):
        # [FUNCTION DOC] Pickles and unpickles the set with every protocol and checks equality.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            p = pickle.dumps(self.set, proto)
            copy = pickle.loads(p)
            self.assertEqual(self.set, copy,
                             "%s != %s" % (self.set, copy))

    def test_issue_37219(self):
        # [FUNCTION DOC] Checks difference and difference_update given a non iterable (an int)
        # raise TypeError (issue 37219).
        with self.assertRaises(TypeError):
            set().difference(123)
        with self.assertRaises(TypeError):
            set().difference_update(123)

#------------------------------------------------------------------------------

class TestBasicOpsEmpty(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for an empty set (expected repr is set()).
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for the empty case: no values, length zero, repr text of
        # an empty set; then calls the parent setUp.
        self.case   = "empty set"
        self.values = []
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 0
        self.repr   = "set()"
        super().setUp()

#------------------------------------------------------------------------------

class TestBasicOpsSingleton(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a one element set containing a number.
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for a set holding the integer 3 with repr of that set.
        self.case   = "unit set (number)"
        self.values = [3]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 1
        self.repr   = "{3}"
        super().setUp()

    def test_in(self):
        # [FUNCTION DOC] Checks the stored number is a member.
        self.assertIn(3, self.set)

    def test_not_in(self):
        # [FUNCTION DOC] Checks a different number is not a member.
        self.assertNotIn(2, self.set)

#------------------------------------------------------------------------------

class TestBasicOpsTuple(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a one element set containing a tuple.
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for a set holding one two-element tuple with its repr.
        self.case   = "unit set (tuple)"
        self.values = [(0, "zero")]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 1
        self.repr   = "{(0, 'zero')}"
        super().setUp()

    def test_in(self):
        # [FUNCTION DOC] Checks the stored tuple is a member.
        self.assertIn((0, "zero"), self.set)

    def test_not_in(self):
        # [FUNCTION DOC] Checks an unrelated number is not a member.
        self.assertNotIn(9, self.set)

#------------------------------------------------------------------------------

class TestBasicOpsTriple(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a set of mixed items (int, str, function); repr
    # not checked.
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for three mixed values; repr is set to none so repr
        # is skipped.
        self.case   = "triple set"
        self.values = [0, "zero", operator.add]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 3
        self.repr   = None
        super().setUp()

#------------------------------------------------------------------------------

class TestBasicOpsString(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a set of three strings; repr is
    # checked order-insensitively.
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for a set of the strings a, b and c.
        self.case   = "string set"
        self.values = ["a", "b", "c"]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 3
        super().setUp()

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr items against the configured values regardless of order.
        self.check_repr_against_values()

#------------------------------------------------------------------------------

class TestBasicOpsBytes(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a set of three byte strings.
    def setUp(self):
        # [FUNCTION DOC] Fills fixtures for a set of the byte strings a, b and c.
        self.case   = "bytes set"
        self.values = [b"a", b"b", b"c"]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 3
        super().setUp()

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr items against the configured values regardless of order.
        self.check_repr_against_values()

#------------------------------------------------------------------------------

class TestBasicOpsMixedStringBytes(_TestBasicOps, __TestCase):
    # [CLASS DOC] Basic operations for a set mixing str and bytes of the same letters.
    def setUp(self):
        # [FUNCTION DOC] Registers a warnings checker context, ignores BytesWarning, and fills
        # fixtures for a set with two strings and two byte strings.
        self.enterContext(warnings_helper.check_warnings())
        warnings.simplefilter('ignore', BytesWarning)
        self.case   = "string and bytes set"
        self.values = ["a", "b", b"a", b"b"]
        self.set    = set(self.values)
        self.dup    = set(self.values)
        self.length = 4
        super().setUp()

    def test_repr(self):
        # [FUNCTION DOC] Checks the repr items against the configured values regardless of order.
        self.check_repr_against_values()

#==============================================================================

def baditer():
    # [FUNCTION DOC] HELPER: generator function whose first step raises TypeError (the yield
    # is unreachable).
    raise TypeError
    yield True

def gooditer():
    # [FUNCTION DOC] HELPER: generator function yielding the single value True.
    yield True

class TestExceptionPropagation(__TestCase):
    # [CLASS DOC] Tests that errors raised while iterating the argument of the set constructor
    # propagate, and that changing a set while iterating it raises.
    """SF 628246:  Set constructor should not trap iterator TypeErrors"""

    def test_instanceWithException(self):
        # [FUNCTION DOC] Checks the set constructor propagates a TypeError raised by the iterator.
        self.assertRaises(TypeError, set, baditer())

    def test_instancesWithoutException(self):
        # [FUNCTION DOC] Checks the constructor accepts list, tuple, dict, range, string
        # and generator.
        # All of these iterables should load without exception.
        set([1,2,3])
        set((1,2,3))
        set({'one':1, 'two':2, 'three':3})
        set(range(3))
        set('abc')
        set(gooditer())

    def test_changingSizeWhileIterating(self):
        # [FUNCTION DOC] Checks that adding items to a set while iterating over it
        # raises RuntimeError.
        s = set([1,2,3])
        try:
            for i in s:
                s.update([4])
        except RuntimeError:
            pass
        else:
            self.fail("no exception when changing size during iteration")

#==============================================================================

class TestSetOfSets(__TestCase):
    # [CLASS DOC] Tests a set containing frozensets.
    def test_constructor(self):
        # [FUNCTION DOC] Checks a frozenset inside a set can be popped, re-added, removed
        # and discarded.
        inner = frozenset([1])
        outer = set([inner])
        element = outer.pop()
        self.assertEqual(type(element), frozenset)
        outer.add(inner)        # Rebuild set of sets with .add method
        outer.remove(inner)
        self.assertEqual(outer, set())   # Verify that remove worked
        outer.discard(inner)    # Absence of KeyError indicates working fine

#==============================================================================

class TestBinaryOps(__TestCase):
    # [CLASS DOC] Operator tests on the set of 2, 4 and 6: equality, union, intersection,
    # disjointness and symmetric difference with subset, superset, overlapping and non-
    # overlapping operands.
    def setUp(self):
        # [FUNCTION DOC] Creates the base set of 2, 4 and 6 and calls the parent setUp.
        self.set = set((2, 4, 6))
        super().setUp()

    def test_eq(self):              # SF bug 643115
        # [FUNCTION DOC] Checks the set equals a set built from a dict with the same keys
        # (bug 643115).
        self.assertEqual(self.set, set({2:1,4:3,6:5}))

    def test_union_subset(self):
        # [FUNCTION DOC] Checks union with a subset leaves the base set unchanged.
        result = self.set | set([2])
        self.assertEqual(result, set((2, 4, 6)))

    def test_union_superset(self):
        # [FUNCTION DOC] Checks union with a superset gives the superset.
        result = self.set | set([2, 4, 6, 8])
        self.assertEqual(result, set([2, 4, 6, 8]))

    def test_union_overlap(self):
        # [FUNCTION DOC] Checks union with an overlapping set gives the merged members.
        result = self.set | set([3, 4, 5])
        self.assertEqual(result, set([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        # [FUNCTION DOC] Checks union with a disjoint set adds its member.
        result = self.set | set([8])
        self.assertEqual(result, set([2, 4, 6, 8]))

    def test_intersection_subset(self):
        # [FUNCTION DOC] Checks intersection with a subset gives the subset.
        result = self.set & set((2, 4))
        self.assertEqual(result, set((2, 4)))

    def test_intersection_superset(self):
        # [FUNCTION DOC] Checks intersection with a superset gives the base set.
        result = self.set & set([2, 4, 6, 8])
        self.assertEqual(result, set([2, 4, 6]))

    def test_intersection_overlap(self):
        # [FUNCTION DOC] Checks intersection with an overlapping set gives the common member.
        result = self.set & set([3, 4, 5])
        self.assertEqual(result, set([4]))

    def test_intersection_non_overlap(self):
        # [FUNCTION DOC] Checks intersection with a disjoint set is empty.
        result = self.set & set([8])
        self.assertEqual(result, empty_set)

    def test_isdisjoint_subset(self):
        # [FUNCTION DOC] Checks isdisjoint with a subset is false.
        result = self.set.isdisjoint(set((2, 4)))
        self.assertEqual(result, False)

    def test_isdisjoint_superset(self):
        # [FUNCTION DOC] Checks isdisjoint with a superset is false.
        result = self.set.isdisjoint(set([2, 4, 6, 8]))
        self.assertEqual(result, False)

    def test_isdisjoint_overlap(self):
        # [FUNCTION DOC] Checks isdisjoint with an overlapping set is false.
        result = self.set.isdisjoint(set([3, 4, 5]))
        self.assertEqual(result, False)

    def test_isdisjoint_non_overlap(self):
        # [FUNCTION DOC] Checks isdisjoint with a disjoint set is true.
        result = self.set.isdisjoint(set([8]))
        self.assertEqual(result, True)

    def test_sym_difference_subset(self):
        # [FUNCTION DOC] Checks symmetric difference with a subset leaves the remaining member.
        result = self.set ^ set((2, 4))
        self.assertEqual(result, set([6]))

    def test_sym_difference_superset(self):
        # [FUNCTION DOC] Checks symmetric difference with a superset leaves the extra member.
        result = self.set ^ set((2, 4, 6, 8))
        self.assertEqual(result, set([8]))

    def test_sym_difference_overlap(self):
        # [FUNCTION DOC] Checks symmetric difference with an overlapping set.
        result = self.set ^ set((3, 4, 5))
        self.assertEqual(result, set([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        # [FUNCTION DOC] Checks symmetric difference with a disjoint set equals the union.
        result = self.set ^ set([8])
        self.assertEqual(result, set([2, 4, 6, 8]))

#==============================================================================

class TestUpdateOps(__TestCase):
    # [CLASS DOC] In-place operator and method tests on the set of 2, 4 and 6 for union,
    # intersection, symmetric difference and difference, with subset, superset, overlap and non-
    # overlap operands. Each test mutates self.set.
    def setUp(self):
        # [FUNCTION DOC] Creates the base set of 2, 4 and 6 and calls the parent setUp.
        self.set = set((2, 4, 6))
        super().setUp()

    def test_union_subset(self):
        # [FUNCTION DOC] Checks in-place or with a subset leaves the set unchanged.
        self.set |= set([2])
        self.assertEqual(self.set, set((2, 4, 6)))

    def test_union_superset(self):
        # [FUNCTION DOC] Checks in-place or with a superset gives the superset.
        self.set |= set([2, 4, 6, 8])
        self.assertEqual(self.set, set([2, 4, 6, 8]))

    def test_union_overlap(self):
        # [FUNCTION DOC] Checks in-place or with an overlapping set merges members.
        self.set |= set([3, 4, 5])
        self.assertEqual(self.set, set([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        # [FUNCTION DOC] Checks in-place or with a disjoint set adds its member.
        self.set |= set([8])
        self.assertEqual(self.set, set([2, 4, 6, 8]))

    def test_union_method_call(self):
        # [FUNCTION DOC] Checks the update method merges an overlapping set.
        self.set.update(set([3, 4, 5]))
        self.assertEqual(self.set, set([2, 3, 4, 5, 6]))

    def test_intersection_subset(self):
        # [FUNCTION DOC] Checks in-place and with a subset gives the subset.
        self.set &= set((2, 4))
        self.assertEqual(self.set, set((2, 4)))

    def test_intersection_superset(self):
        # [FUNCTION DOC] Checks in-place and with a superset leaves the set unchanged.
        self.set &= set([2, 4, 6, 8])
        self.assertEqual(self.set, set([2, 4, 6]))

    def test_intersection_overlap(self):
        # [FUNCTION DOC] Checks in-place and with an overlapping set keeps the common member.
        self.set &= set([3, 4, 5])
        self.assertEqual(self.set, set([4]))

    def test_intersection_non_overlap(self):
        # [FUNCTION DOC] Checks in-place and with a disjoint set empties the set.
        self.set &= set([8])
        self.assertEqual(self.set, empty_set)

    def test_intersection_method_call(self):
        # [FUNCTION DOC] Checks the intersection_update method keeps only the common member.
        self.set.intersection_update(set([3, 4, 5]))
        self.assertEqual(self.set, set([4]))

    def test_sym_difference_subset(self):
        # [FUNCTION DOC] Checks in-place xor with a subset leaves the remainder.
        self.set ^= set((2, 4))
        self.assertEqual(self.set, set([6]))

    def test_sym_difference_superset(self):
        # [FUNCTION DOC] Checks in-place xor with a superset leaves the extra member.
        self.set ^= set((2, 4, 6, 8))
        self.assertEqual(self.set, set([8]))

    def test_sym_difference_overlap(self):
        # [FUNCTION DOC] Checks in-place xor with an overlapping set.
        self.set ^= set((3, 4, 5))
        self.assertEqual(self.set, set([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        # [FUNCTION DOC] Checks in-place xor with a disjoint set adds its member.
        self.set ^= set([8])
        self.assertEqual(self.set, set([2, 4, 6, 8]))

    def test_sym_difference_method_call(self):
        # [FUNCTION DOC] Checks the symmetric_difference_update method with an overlapping set.
        self.set.symmetric_difference_update(set([3, 4, 5]))
        self.assertEqual(self.set, set([2, 3, 5, 6]))

    def test_difference_subset(self):
        # [FUNCTION DOC] Checks in-place subtraction of a subset leaves the remainder.
        self.set -= set((2, 4))
        self.assertEqual(self.set, set([6]))

    def test_difference_superset(self):
        # [FUNCTION DOC] Checks in-place subtraction of a superset empties the set.
        self.set -= set((2, 4, 6, 8))
        self.assertEqual(self.set, set([]))

    def test_difference_overlap(self):
        # [FUNCTION DOC] Checks in-place subtraction of an overlapping set removes the
        # common members.
        self.set -= set((3, 4, 5))
        self.assertEqual(self.set, set([2, 6]))

    def test_difference_non_overlap(self):
        # [FUNCTION DOC] Checks in-place subtraction of a disjoint set leaves the set unchanged.
        self.set -= set([8])
        self.assertEqual(self.set, set([2, 4, 6]))

    def test_difference_method_call(self):
        # [FUNCTION DOC] Checks the difference_update method removes common members.
        self.set.difference_update(set([3, 4, 5]))
        self.assertEqual(self.set, set([2, 6]))

#==============================================================================

class TestMutate(__TestCase):
    # [CLASS DOC] Mutation tests (add, remove, discard, clear, pop, update) on a set of a, b
    # and c.
    def setUp(self):
        # [FUNCTION DOC] Creates the values list of a, b and c and a set from it; calls the
        # parent setUp.
        self.values = ["a", "b", "c"]
        self.set = set(self.values)
        super().setUp()

    def test_add_present(self):
        # [FUNCTION DOC] Checks adding an existing element changes nothing.
        self.set.add("c")
        self.assertEqual(self.set, set("abc"))

    def test_add_absent(self):
        # [FUNCTION DOC] Checks adding a new element grows the set.
        self.set.add("d")
        self.assertEqual(self.set, set("abcd"))

    def test_add_until_full(self):
        # [FUNCTION DOC] Checks each addition into an empty set increases the length by one and
        # the final set equals the base set.
        tmp = set()
        expected_len = 0
        for v in self.values:
            tmp.add(v)
            expected_len += 1
            self.assertEqual(len(tmp), expected_len)
        self.assertEqual(tmp, self.set)

    def test_remove_present(self):
        # [FUNCTION DOC] Checks remove of a present element.
        self.set.remove("b")
        self.assertEqual(self.set, set("ac"))

    def test_remove_absent(self):
        # [FUNCTION DOC] Checks remove of a missing element raises a LookupError (KeyError).
        try:
            self.set.remove("d")
            self.fail("Removing missing element should have raised LookupError")
        except LookupError:
            pass

    def test_remove_until_empty(self):
        # [FUNCTION DOC] Checks removing every value one by one decrements the length each time.
        expected_len = len(self.set)
        for v in self.values:
            self.set.remove(v)
            expected_len -= 1
            self.assertEqual(len(self.set), expected_len)

    def test_discard_present(self):
        # [FUNCTION DOC] Checks discard of a present element.
        self.set.discard("c")
        self.assertEqual(self.set, set("ab"))

    def test_discard_absent(self):
        # [FUNCTION DOC] Checks discard of a missing element changes nothing and does not raise.
        self.set.discard("d")
        self.assertEqual(self.set, set("abc"))

    def test_clear(self):
        # [FUNCTION DOC] Checks clear empties the set.
        self.set.clear()
        self.assertEqual(len(self.set), 0)

    def test_pop(self):
        # [FUNCTION DOC] Checks repeated pop returns every value exactly once until the set
        # is empty.
        popped = {}
        while self.set:
            popped[self.set.pop()] = None
        self.assertEqual(len(popped), len(self.values))
        for v in self.values:
            self.assertIn(v, popped)

    def test_update_empty_tuple(self):
        # [FUNCTION DOC] Checks update with an empty tuple changes nothing.
        self.set.update(())
        self.assertEqual(self.set, set(self.values))

    def test_update_unit_tuple_overlap(self):
        # [FUNCTION DOC] Checks update with a tuple of existing items changes nothing.
        self.set.update(("a",))
        self.assertEqual(self.set, set(self.values))

    def test_update_unit_tuple_non_overlap(self):
        # [FUNCTION DOC] Checks update with a tuple containing one new item adds it.
        self.set.update(("a", "z"))
        self.assertEqual(self.set, set(self.values + ["z"]))

#==============================================================================

class _TestSubsets:
    # [CLASS DOC] Mixin testing ordering comparisons. Subclasses provide left and right sets, a
    # name, and cases: the comparison operators that must be true. Class level tables map
    # operators to the subset method names and to their reversed operator.

    case2method = {"<=": "issubset",
                   ">=": "issuperset",
                  }

    reverse = {"==": "==",
               "!=": "!=",
               "<":  ">",
               ">":  "<",
               "<=": ">=",
               ">=": "<=",
              }

    def test_issubset(self):
        # [FUNCTION DOC] For every operator of equality, ordering and their negations, evaluates
        # the comparison of left with right (through text evaluation with graph break errors
        # disabled) and, for subset and superset, the method form, checking the result is true
        # exactly when the operator is in cases; then repeats with operands swapped and the
        # reversed operator.
        x = self.left
        y = self.right
        for case in "!=", "==", "<", "<=", ">", ">=":
            expected = case in self.cases
            # Test the binary infix spelling.
            with torch._dynamo.error_on_graph_break(False):
                result = eval("x" + case + "y", locals())
            self.assertEqual(result, expected)
            # Test the "friendly" method-name spelling, if one exists.
            if case in _TestSubsets.case2method:
                method = getattr(x, _TestSubsets.case2method[case])
                result = method(y)
                self.assertEqual(result, expected)

            # Now do the same for the operands reversed.
            rcase = _TestSubsets.reverse[case]
            with torch._dynamo.error_on_graph_break(False):
                result = eval("y" + rcase + "x", locals())
            self.assertEqual(result, expected)
            if rcase in _TestSubsets.case2method:
                method = getattr(y, _TestSubsets.case2method[rcase])
                result = method(x)
                self.assertEqual(result, expected)
#------------------------------------------------------------------------------

class TestSubsetEqualEmpty(_TestSubsets, __TestCase):
    # [CLASS DOC] Comparison fixtures: two empty sets, equal under equality, subset and superset.
    left  = set()
    right = set()
    name  = "both empty"
    cases = "==", "<=", ">="

#------------------------------------------------------------------------------

class TestSubsetEqualNonEmpty(_TestSubsets, __TestCase):
    # [CLASS DOC] Comparison fixtures: two equal non-empty sets.
    left  = set([1, 2])
    right = set([1, 2])
    name  = "equal pair"
    cases = "==", "<=", ">="

#------------------------------------------------------------------------------

class TestSubsetEmptyNonEmpty(_TestSubsets, __TestCase):
    # [CLASS DOC] Comparison fixtures: empty set on the left, non-empty on the right.
    left  = set()
    right = set([1, 2])
    name  = "one empty, one non-empty"
    cases = "!=", "<", "<="

#------------------------------------------------------------------------------

class TestSubsetPartial(_TestSubsets, __TestCase):
    # [CLASS DOC] Comparison fixtures: left is a proper subset of right.
    left  = set([1])
    right = set([1, 2])
    name  = "one a non-empty proper subset of other"
    cases = "!=", "<", "<="

#------------------------------------------------------------------------------

class TestSubsetNonOverlap(_TestSubsets, __TestCase):
    # [CLASS DOC] Comparison fixtures: disjoint non-empty sets, only inequality holds.
    left  = set([1])
    right = set([2])
    name  = "neither empty, neither contains"
    cases = "!="

#==============================================================================

class _TestOnlySetsInBinaryOps:
    # [CLASS DOC] Mixin checking that set operators accept only sets. Subclasses give self.set, an
    # other (non-set operand) and otherIsIterable telling if update-style methods accept it.

    def test_eq_ne(self):
        # [FUNCTION DOC] Checks equality and inequality with a non-set operand are allowed and
        # report unequal in both directions.
        # Unlike the others, this is testing that == and != *are* allowed.
        self.assertEqual(self.other == self.set, False)
        self.assertEqual(self.set == self.other, False)
        self.assertEqual(self.other != self.set, True)
        self.assertEqual(self.set != self.other, True)

    def test_ge_gt_le_lt(self):
        # [FUNCTION DOC] Checks all four ordering operators raise TypeError with a non-set operand
        # in either position.
        self.assertRaises(TypeError, lambda: self.set < self.other)
        self.assertRaises(TypeError, lambda: self.set <= self.other)
        self.assertRaises(TypeError, lambda: self.set > self.other)
        self.assertRaises(TypeError, lambda: self.set >= self.other)

        self.assertRaises(TypeError, lambda: self.other < self.set)
        self.assertRaises(TypeError, lambda: self.other <= self.set)
        self.assertRaises(TypeError, lambda: self.other > self.set)
        self.assertRaises(TypeError, lambda: self.other >= self.set)

    def test_update_operator(self):
        # [FUNCTION DOC] Checks the in-place or operator with a non-set operand raises TypeError.
        try:
            self.set |= self.other
        except TypeError:
            pass
        else:
            self.fail("expected TypeError")

    def test_update(self):
        # [FUNCTION DOC] Checks update accepts the operand when it is iterable, else
        # raises TypeError.
        if self.otherIsIterable:
            self.set.update(self.other)
        else:
            self.assertRaises(TypeError, self.set.update, self.other)

    def test_union(self):
        # [FUNCTION DOC] Checks the or operator raises TypeError in both directions and the union
        # method works for iterable operands; otherwise it raises TypeError.
        self.assertRaises(TypeError, lambda: self.set | self.other)
        self.assertRaises(TypeError, lambda: self.other | self.set)
        if self.otherIsIterable:
            self.set.union(self.other)
        else:
            self.assertRaises(TypeError, self.set.union, self.other)

    def test_intersection_update_operator(self):
        # [FUNCTION DOC] Checks the in-place and operator with a non-set operand raises TypeError.
        try:
            self.set &= self.other
        except TypeError:
            pass
        else:
            self.fail("expected TypeError")

    def test_intersection_update(self):
        # [FUNCTION DOC] Checks intersection_update accepts an iterable operand and raises
        # TypeError for a non-iterable one.
        if self.otherIsIterable:
            self.set.intersection_update(self.other)
        else:
            self.assertRaises(TypeError,
                              self.set.intersection_update,
                              self.other)

    def test_intersection(self):
        # [FUNCTION DOC] Checks the and operator raises TypeError in both directions and the
        # intersection method accepts only iterable operands.
        self.assertRaises(TypeError, lambda: self.set & self.other)
        self.assertRaises(TypeError, lambda: self.other & self.set)
        if self.otherIsIterable:
            self.set.intersection(self.other)
        else:
            self.assertRaises(TypeError, self.set.intersection, self.other)

    def test_sym_difference_update_operator(self):
        # [FUNCTION DOC] Checks the in-place xor operator with a non-set operand raises TypeError.
        try:
            self.set ^= self.other
        except TypeError:
            pass
        else:
            self.fail("expected TypeError")

    def test_sym_difference_update(self):
        # [FUNCTION DOC] Checks symmetric_difference_update accepts only iterable operands.
        if self.otherIsIterable:
            self.set.symmetric_difference_update(self.other)
        else:
            self.assertRaises(TypeError,
                              self.set.symmetric_difference_update,
                              self.other)

    def test_sym_difference(self):
        # [FUNCTION DOC] Checks the xor operator raises TypeError in both directions and the
        # symmetric_difference method accepts only iterable operands.
        self.assertRaises(TypeError, lambda: self.set ^ self.other)
        self.assertRaises(TypeError, lambda: self.other ^ self.set)
        if self.otherIsIterable:
            self.set.symmetric_difference(self.other)
        else:
            self.assertRaises(TypeError, self.set.symmetric_difference, self.other)

    def test_difference_update_operator(self):
        # [FUNCTION DOC] Checks the in-place subtraction operator with a non-set operand
        # raises TypeError.
        try:
            self.set -= self.other
        except TypeError:
            pass
        else:
            self.fail("expected TypeError")

    def test_difference_update(self):
        # [FUNCTION DOC] Checks difference_update accepts only iterable operands.
        if self.otherIsIterable:
            self.set.difference_update(self.other)
        else:
            self.assertRaises(TypeError,
                              self.set.difference_update,
                              self.other)

    def test_difference(self):
        # [FUNCTION DOC] Checks the subtraction operator raises TypeError in both directions and
        # the difference method accepts only iterable operands.
        self.assertRaises(TypeError, lambda: self.set - self.other)
        self.assertRaises(TypeError, lambda: self.other - self.set)
        if self.otherIsIterable:
            self.set.difference(self.other)
        else:
            self.assertRaises(TypeError, self.set.difference, self.other)

#------------------------------------------------------------------------------

class TestOnlySetsNumeric(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with an integer (not iterable).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is 19; not iterable.
        self.set   = set((1, 2, 3))
        self.other = 19
        self.otherIsIterable = False
        super().setUp()

#------------------------------------------------------------------------------

class TestOnlySetsDict(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with a dict (iterable).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is a two item dict; iterable.
        self.set   = set((1, 2, 3))
        self.other = {1:2, 3:4}
        self.otherIsIterable = True
        super().setUp()

#------------------------------------------------------------------------------

class TestOnlySetsOperator(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with a function object (not iterable).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is the operator add function;
        # not iterable.
        self.set   = set((1, 2, 3))
        self.other = operator.add
        self.otherIsIterable = False
        super().setUp()

#------------------------------------------------------------------------------

class TestOnlySetsTuple(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with a tuple (iterable).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is a three number tuple; iterable.
        self.set   = set((1, 2, 3))
        self.other = (2, 4, 6)
        self.otherIsIterable = True
        super().setUp()

#------------------------------------------------------------------------------

class TestOnlySetsString(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with a string (iterable).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is the string abc; iterable.
        self.set   = set((1, 2, 3))
        self.other = 'abc'
        self.otherIsIterable = True
        super().setUp()

#------------------------------------------------------------------------------

class TestOnlySetsGenerator(_TestOnlySetsInBinaryOps, __TestCase):
    # [CLASS DOC] Non-set operand tests with a generator (iterable, single use).
    def setUp(self):
        # [FUNCTION DOC] Sets fixtures: set of 1, 2, 3; other is a fresh generator of even
        # numbers; iterable.
        def gen():
            # [FUNCTION DOC] HELPER: generator yielding 0, 2, 4, 6 and 8.
            for i in range(0, 10, 2):
                yield i
        self.set   = set((1, 2, 3))
        self.other = gen()
        self.otherIsIterable = True
        super().setUp()

#==============================================================================

class _TestCopying:
    # [CLASS DOC] Mixin for copy tests; subclasses assign the set to copy in setUp.

    def test_copy(self):
        # [FUNCTION DOC] Checks a shallow copy has the same size and holds the identical
        # element objects.
        dup = self.set.copy()
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.set, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertTrue(dup_list[i] is set_list[i])

    def test_deep_copy(self):
        # [FUNCTION DOC] Checks a deep copy has the same size and equal (not necessarily
        # identical) elements, comparing both sets in a canonical order sorted by repr.
        dup = copy.deepcopy(self.set)
        ##print type(dup), repr(dup)
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.set, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertEqual(dup_list[i], set_list[i])

#------------------------------------------------------------------------------

class TestCopyingEmpty(_TestCopying, __TestCase):
    # [CLASS DOC] Copy tests on an empty set.
    def setUp(self):
        # [FUNCTION DOC] Sets the set under test to an empty set.
        self.set = set()
        super().setUp()

#------------------------------------------------------------------------------

class TestCopyingSingleton(_TestCopying, __TestCase):
    # [CLASS DOC] Copy tests on a set with one string.
    def setUp(self):
        # [FUNCTION DOC] Sets the set under test to a one string set.
        self.set = set(["hello"])
        super().setUp()

#------------------------------------------------------------------------------

class TestCopyingTriple(_TestCopying, __TestCase):
    # [CLASS DOC] Copy tests on a set with a string, a number and None.
    def setUp(self):
        # [FUNCTION DOC] Sets the set under test to a three item mixed set.
        self.set = set(["zero", 0, None])
        super().setUp()

#------------------------------------------------------------------------------

class TestCopyingTuple(_TestCopying, __TestCase):
    # [CLASS DOC] Copy tests on a set with one tuple.
    def setUp(self):
        # [FUNCTION DOC] Sets the set under test to a one tuple set.
        self.set = set([(1, 2)])
        super().setUp()

#------------------------------------------------------------------------------

class TestCopyingNested(_TestCopying, __TestCase):
    # [CLASS DOC] Copy tests on a set with one nested tuple.
    def setUp(self):
        # [FUNCTION DOC] Sets the set under test to a one nested tuple set.
        self.set = set([((1, 2), (3, 4))])
        super().setUp()

#==============================================================================

class TestIdentities(__TestCase):
    # [CLASS DOC] Algebraic identity tests on two sets built from the letters of two words.
    def setUp(self):
        # [FUNCTION DOC] Creates the two sets a and b used by all tests and calls the
        # parent setUp.
        self.a = set('abracadabra')
        self.b = set('alacazam')
        super().setUp()

    def test_binopsVsSubsets(self):
        # [FUNCTION DOC] Checks difference and intersection are proper subsets of the operands,
        # union is a proper superset, and symmetric difference is a proper subset of the union.
        a, b = self.a, self.b
        self.assertTrue(a - b < a)
        self.assertTrue(b - a < b)
        self.assertTrue(a & b < a)
        self.assertTrue(a & b < b)
        self.assertTrue(a | b > a)
        self.assertTrue(a | b > b)
        self.assertTrue(a ^ b < a | b)

    def test_commutativity(self):
        # [FUNCTION DOC] Checks intersection, union and symmetric difference commute, and
        # difference does not.
        a, b = self.a, self.b
        self.assertEqual(a&b, b&a)
        self.assertEqual(a|b, b|a)
        self.assertEqual(a^b, b^a)
        if a != b:
            self.assertNotEqual(a-b, b-a)

    def test_summations(self):
        # [FUNCTION DOC] Checks that unions of the difference, intersection and symmetric
        # difference parts rebuild the union or the original operands.
        # check that sums of parts equal the whole
        a, b = self.a, self.b
        self.assertEqual((a-b)|(a&b)|(b-a), a|b)
        self.assertEqual((a&b)|(a^b), a|b)
        self.assertEqual(a|(b-a), a|b)
        self.assertEqual((a-b)|b, a|b)
        self.assertEqual((a-b)|(a&b), a)
        self.assertEqual((b-a)|(a&b), b)
        self.assertEqual((a-b)|(b-a), a^b)

    def test_exclusion(self):
        # [FUNCTION DOC] Checks that parts that should not overlap have an empty intersection.
        # check that inverse operations show non-overlap
        a, b, zero = self.a, self.b, set()
        self.assertEqual((a-b)&b, zero)
        self.assertEqual((b-a)&a, zero)
        self.assertEqual((a&b)&(a^b), zero)

# Tests derived from test_itertools.py =======================================

def R(seqn):
    # [FUNCTION DOC] HELPER: plain generator yielding each item of the given sequence.
    'Regular generator'
    for i in seqn:
        yield i

class G:
    # [CLASS DOC] HELPER: sequence wrapper that supports only indexed access, so iteration uses
    # the old index protocol.
    'Sequence using __getitem__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the wrapped sequence.
        self.seqn = seqn
    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: returns the item at the given index (errors propagate).
        return self.seqn[i]

class I:
    # [CLASS DOC] HELPER: iterator class that walks the wrapped sequence with an index counter.
    'Sequence using iterator protocol'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the sequence and resets the position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: returns the object itself as the iterator.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: returns the next item and advances; raises StopIteration at
        # the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class Ig:
    # [CLASS DOC] HELPER: iterable whose iteration hook is a generator over the wrapped sequence.
    'Sequence using iterator protocol defined with a generator'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: generator yielding each item of the sequence.
        for val in self.seqn:
            yield val

class X:
    # [CLASS DOC] HELPER: invalid iterable offering the next hook but neither iteration nor index
    # hooks, so it must be rejected with TypeError.
    'Missing __getitem__ and __iter__'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        # [FUNCTION DOC] HELPER: returns the next item and advances; raises StopIteration at
        # the end.
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v

class N:
    # [CLASS DOC] HELPER: invalid iterable whose iteration hook returns an object lacking a next
    # hook, so it must be rejected with TypeError.
    'Iterator missing __next__()'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: returns self (which has no next hook).
        return self

class E:
    # [CLASS DOC] HELPER: iterator whose next hook always raises ZeroDivisionError.
    'Test propagation of exceptions'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: stores the sequence and a position counter.
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        # [FUNCTION DOC] HELPER: returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: always raises ZeroDivisionError.
        3 // 0

class S:
    # [CLASS DOC] HELPER: iterator that is empty (stops immediately).
    'Test immediate stop'
    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: ignores the sequence argument.
        pass
    def __iter__(self):
        # [FUNCTION DOC] HELPER: returns self.
        return self
    def __next__(self):
        # [FUNCTION DOC] HELPER: always raises StopIteration.
        raise StopIteration

from itertools import chain
def L(seqn):
    # [FUNCTION DOC] HELPER: builds a nested iterator chain over a sequence (indexed wrapper, then
    # generator class, then plain generator, then identity map) to test multiple iterator tiers.
    'Test multiple tiers of iterators'
    return chain(map(lambda x:x, R(Ig(G(seqn)))))

class TestVariousIteratorArgs(__TestCase):
    # [CLASS DOC] Tests set and frozenset operations with many kinds of iterable arguments
    # (indexed, iterator, generator-based, empty, tiered) and bad iterables.

    def test_constructor(self):
        # [FUNCTION DOC] Checks the constructors of set and frozenset give the same items as the
        # source iterables, raise TypeError for invalid iterables and propagate ZeroDivisionError.
        for cons in (set, frozenset):
            for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
                for g in (G, I, Ig, S, L, R):
                    self.assertEqual(sorted(cons(g(s)), key=repr), sorted(g(s), key=repr))
                self.assertRaises(TypeError, cons , X(s))
                self.assertRaises(TypeError, cons , N(s))
                self.assertRaises(ZeroDivisionError, cons , E(s))

    def test_inline_methods(self):
        # [FUNCTION DOC] Checks union, intersection, difference, symmetric_difference and
        # isdisjoint give the same result for a plain sequence as for wrapped iterables, and
        # reject bad ones.
        s = set('november')
        for data in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5), 'december'):
            for meth in (s.union, s.intersection, s.difference, s.symmetric_difference, s.isdisjoint):
                for g in (G, I, Ig, L, R):
                    expected = meth(data)
                    actual = meth(g(data))
                    if isinstance(expected, bool):
                        self.assertEqual(actual, expected)
                    else:
                        self.assertEqual(sorted(actual, key=repr), sorted(expected, key=repr))
                self.assertRaises(TypeError, meth, X(s))
                self.assertRaises(TypeError, meth, N(s))
                self.assertRaises(ZeroDivisionError, meth, E(s))

    def test_inplace_methods(self):
        # [FUNCTION DOC] Checks the four in-place update methods give identical results when fed a
        # list copy or the wrapped iterable, and reject bad iterables with the expected errors.
        for data in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5), 'december'):
            for methname in ('update', 'intersection_update',
                             'difference_update', 'symmetric_difference_update'):
                for g in (G, I, Ig, S, L, R):
                    s = set('january')
                    t = s.copy()
                    getattr(s, methname)(list(g(data)))
                    getattr(t, methname)(g(data))
                    self.assertEqual(sorted(s, key=repr), sorted(t, key=repr))

                self.assertRaises(TypeError, getattr(set('january'), methname), X(data))
                self.assertRaises(TypeError, getattr(set('january'), methname), N(data))
                self.assertRaises(ZeroDivisionError, getattr(set('january'), methname), E(data))

class bad_eq:
    # [CLASS DOC] HELPER: element type whose equality, while the global be_bad flag is set, clears
    # global set2 and raises ZeroDivisionError. Hash is always zero.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: when be_bad is set, clears set2 and raises ZeroDivisionError;
        # otherwise compares identity.
        if be_bad:
            set2.clear()
            raise ZeroDivisionError
        return self is other
    def __hash__(self):
        # [FUNCTION DOC] HELPER: returns the constant 0.
        return 0

class bad_dict_clear:
    # [CLASS DOC] HELPER: element type whose equality, while be_bad is set, clears the global dict
    # dict2. Hash is always zero.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: when be_bad is set, clears dict2; returns identity comparison.
        if be_bad:
            dict2.clear()
        return self is other
    def __hash__(self):
        # [FUNCTION DOC] HELPER: returns the constant 0.
        return 0

class TestWeirdBugs(__TestCase):
    # [CLASS DOC] Regression tests for crashes when sets are mutated during operations.
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_8420_set_merge(self):
        # [FUNCTION DOC] Regression test for issue 8420: update with a set whose element equality
        # clears that set must raise ZeroDivisionError, and symmetric_difference_update with a
        # dict being cleared during comparison must not crash. Sets module level globals be_bad,
        # set2 and dict2. Skipped under Dynamo.
        # This used to segfault
        global be_bad, set2, dict2
        be_bad = False
        set1 = {bad_eq()}
        set2 = {bad_eq() for i in range(75)}
        be_bad = True
        self.assertRaises(ZeroDivisionError, set1.update, set2)

        be_bad = False
        set1 = {bad_dict_clear()}
        dict2 = {bad_dict_clear(): None}
        be_bad = True
        set1.symmetric_difference_update(dict2)

    def test_iter_and_mutate(self):
        # [FUNCTION DOC] Regression test for issue 24581: mutating a set after creating an
        # iterator and then exhausting the iterator must not crash. No assertions.
        # Issue #24581
        s = set(range(100))
        s.clear()
        s.update(range(100))
        si = iter(s)
        s.clear()
        a = list(range(100))
        s.update(range(100))
        list(si)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_merge_and_mutate(self):
        # [FUNCTION DOC] Checks update from a set whose element equality clears that set does not
        # crash. Skipped under Dynamo.
        class X:
            # [CLASS DOC] HELPER: element type with hash zero whose equality clears the
            # variable other.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns the hash of zero.
                return hash(0)
            def __eq__(self, o):
                # [FUNCTION DOC] HELPER: clears the enclosing set other and returns False.
                other.clear()
                return False

        other = set()
        other = {X() for i in range(10)}
        s = {0}
        s.update(other)


class _TestOperationsMutating:
    # [CLASS DOC] Regression mixin for bpo-46615: operations on sets whose elements mutate the
    # sets during comparison must not crash. Subclasses choose constructor1 and constructor2
    # (callables building the two operands).
    """Regression test for bpo-46615"""

    constructor1 = None
    constructor2 = None

    def make_sets_of_bad_objects(self):
        # [FUNCTION DOC] HELPER: builds two sets with up to 49 random elements each, using the
        # class constructors, of a Bad type whose equality randomly clears either set and returns
        # a random boolean, with random hash of 0 or 1. Misbehavior starts only after
        # construction. Returns the pair.
        class Bad:
            # [CLASS DOC] HELPER: element type that, once enabled, randomly clears set1 or set2
            # while being compared.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: when enabled, may clear either set and returns a
                # random boolean.
                if not enabled:
                    return False
                if randrange(20) == 0:
                    set1.clear()
                if randrange(20) == 0:
                    set2.clear()
                return bool(randrange(2))
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns a random 0 or 1.
                return randrange(2)
        # Don't behave poorly during construction.
        enabled = False
        set1 = self.constructor1(Bad() for _ in range(randrange(50)))
        set2 = self.constructor2(Bad() for _ in range(randrange(50)))
        # Now start behaving poorly
        enabled = True
        return set1, set2

    def check_set_op_does_not_crash(self, function):
        # [FUNCTION DOC] HELPER: repeats 100 times: makes two sets and applies the given function
        # to them; a RuntimeError is accepted only if its message says the size changed during
        # iteration. Any other exception propagates.
        for _ in range(100):
            set1, set2 = self.make_sets_of_bad_objects()
            try:
                function(set1, set2)
            except RuntimeError as e:
                # Just make sure we don't crash here.
                self.assertIn("changed size during iteration", str(e))


class _TestBinaryOpsMutating(_TestOperationsMutating):
    # [CLASS DOC] Mixin with operator-level mutation tests over pairs of mutating sets.

    def test_eq_with_mutation(self):
        # [FUNCTION DOC] Checks the equality operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a == b)

    def test_ne_with_mutation(self):
        # [FUNCTION DOC] Checks the inequality operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a != b)

    def test_lt_with_mutation(self):
        # [FUNCTION DOC] Checks the less-than operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a < b)

    def test_le_with_mutation(self):
        # [FUNCTION DOC] Checks the less-or-equal operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a <= b)

    def test_gt_with_mutation(self):
        # [FUNCTION DOC] Checks the greater-than operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a > b)

    def test_ge_with_mutation(self):
        # [FUNCTION DOC] Checks the greater-or-equal operator does not crash with
        # mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a >= b)

    def test_and_with_mutation(self):
        # [FUNCTION DOC] Checks the and operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a & b)

    def test_or_with_mutation(self):
        # [FUNCTION DOC] Checks the or operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a | b)

    def test_sub_with_mutation(self):
        # [FUNCTION DOC] Checks the subtraction operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a - b)

    def test_xor_with_mutation(self):
        # [FUNCTION DOC] Checks the xor operator does not crash with mutating elements.
        self.check_set_op_does_not_crash(lambda a, b: a ^ b)

    def test_iadd_with_mutation(self):
        # [FUNCTION DOC] Checks the in-place and operator does not crash with mutating elements
        # (the test name says iadd but the operator used is and).
        def f(a, b):
            # [FUNCTION DOC] HELPER: applies the in-place and operator to the first set with
            # the second.
            a &= b
        self.check_set_op_does_not_crash(f)

    def test_ior_with_mutation(self):
        # [FUNCTION DOC] Checks the in-place or operator does not crash with mutating elements.
        def f(a, b):
            # [FUNCTION DOC] HELPER: applies the in-place or operator to the first set with
            # the second.
            a |= b
        self.check_set_op_does_not_crash(f)

    def test_isub_with_mutation(self):
        # [FUNCTION DOC] Checks the in-place subtraction operator does not crash with
        # mutating elements.
        def f(a, b):
            # [FUNCTION DOC] HELPER: applies the in-place subtraction to the first set with
            # the second.
            a -= b
        self.check_set_op_does_not_crash(f)

    def test_ixor_with_mutation(self):
        # [FUNCTION DOC] Checks the in-place xor operator does not crash with mutating elements.
        def f(a, b):
            # [FUNCTION DOC] HELPER: applies the in-place xor to the first set with the second.
            a ^= b
        self.check_set_op_does_not_crash(f)

    def test_iteration_with_mutation(self):
        # [FUNCTION DOC] Checks plain iteration over mutating sets (one after the other in both
        # orders, and in lockstep) does not crash.
        def f1(a, b):
            # [FUNCTION DOC] HELPER: iterates fully over a then b.
            for x in a:
                pass
            for y in b:
                pass
        def f2(a, b):
            # [FUNCTION DOC] HELPER: iterates fully over b then a.
            for y in b:
                pass
            for x in a:
                pass
        def f3(a, b):
            # [FUNCTION DOC] HELPER: iterates a and b together in lockstep.
            for x, y in zip(a, b):
                pass
        self.check_set_op_does_not_crash(f1)
        self.check_set_op_does_not_crash(f2)
        self.check_set_op_does_not_crash(f3)


@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestBinaryOpsMutating_Set_Set(_TestBinaryOpsMutating, __TestCase):
    # [CLASS DOC] Operator mutation tests with both operands being built-in sets (skipped
    # under Dynamo).
    constructor1 = set
    constructor2 = set

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestBinaryOpsMutating_Subclass_Subclass(_TestBinaryOpsMutating, __TestCase):
    # [CLASS DOC] Operator mutation tests with both operands being set subclass instances (skipped
    # under Dynamo).
    constructor1 = SetSubclass
    constructor2 = SetSubclass

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestBinaryOpsMutating_Set_Subclass(_TestBinaryOpsMutating, __TestCase):
    # [CLASS DOC] Operator mutation tests with a built-in set on the left and subclass on the
    # right (skipped under Dynamo).
    constructor1 = set
    constructor2 = SetSubclass

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestBinaryOpsMutating_Subclass_Set(_TestBinaryOpsMutating, __TestCase):
    # [CLASS DOC] Operator mutation tests with a subclass on the left and built-in set on the
    # right (skipped under Dynamo).
    constructor1 = SetSubclass
    constructor2 = set


class _TestMethodsMutating(_TestOperationsMutating):
    # [CLASS DOC] Mixin with method-level mutation tests; each test passes an unbound set method
    # to the crash checker.

    def test_issubset_with_mutation(self):
        # [FUNCTION DOC] Checks issubset does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.issubset)

    def test_issuperset_with_mutation(self):
        # [FUNCTION DOC] Checks issuperset does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.issuperset)

    def test_intersection_with_mutation(self):
        # [FUNCTION DOC] Checks intersection does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.intersection)

    def test_union_with_mutation(self):
        # [FUNCTION DOC] Checks union does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.union)

    def test_difference_with_mutation(self):
        # [FUNCTION DOC] Checks difference does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.difference)

    def test_symmetric_difference_with_mutation(self):
        # [FUNCTION DOC] Checks symmetric_difference does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.symmetric_difference)

    def test_isdisjoint_with_mutation(self):
        # [FUNCTION DOC] Checks isdisjoint does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.isdisjoint)

    def test_difference_update_with_mutation(self):
        # [FUNCTION DOC] Checks difference_update does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.difference_update)

    def test_intersection_update_with_mutation(self):
        # [FUNCTION DOC] Checks intersection_update does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.intersection_update)

    def test_symmetric_difference_update_with_mutation(self):
        # [FUNCTION DOC] Checks symmetric_difference_update does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.symmetric_difference_update)

    def test_update_with_mutation(self):
        # [FUNCTION DOC] Checks update does not crash with mutating elements.
        self.check_set_op_does_not_crash(set.update)


@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Set_Set(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with set and set operands (skipped under Dynamo).
    constructor1 = set
    constructor2 = set

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Subclass_Subclass(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with subclass and subclass operands (skipped
    # under Dynamo).
    constructor1 = SetSubclass
    constructor2 = SetSubclass

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Set_Subclass(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with set and subclass operands (skipped under Dynamo).
    constructor1 = set
    constructor2 = SetSubclass

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Subclass_Set(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with subclass and set operands (skipped under Dynamo).
    constructor1 = SetSubclass
    constructor2 = set

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Set_Dict(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with a set and a dict (built with fromkeys) as operands
    # (skipped under Dynamo).
    constructor1 = set
    constructor2 = dict.fromkeys

@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO,
    "__build_class__ with closed over objects not supported",
)
class TestMethodsMutating_Set_List(_TestMethodsMutating, __TestCase):
    # [CLASS DOC] Method mutation tests with a set and a list as operands (skipped under Dynamo).
    constructor1 = set
    constructor2 = list


# Application tests (based on David Eppstein's graph recipes ====================================

def powerset(U):
    # [FUNCTION DOC] HELPER: recursive generator yielding all subsets of U as frozensets (2 to the
    # power of the size of U items), consuming U as an iterator.
    """Generates all subsets of a set or sequence U."""
    U = iter(U)
    try:
        x = frozenset([next(U)])
        for S in powerset(U):
            yield S
            yield S | x
    except StopIteration:
        yield frozenset()

def cube(n):
    # [FUNCTION DOC] HELPER: builds the n-dimensional hypercube graph: a mapping from each
    # vertex (a frozenset subset of range n) to the frozenset of its neighbors, found by
    # toggling one element.
    """Graph of n-dimensional hypercube."""
    singletons = [frozenset([x]) for x in range(n)]
    return dict([(x, frozenset([x^s for s in singletons]))
                 for x in powerset(range(n))])

def linegraph(G):
    # [FUNCTION DOC] HELPER: builds the line graph of G: vertices are frozensets of two endpoints
    # (edges of G) mapped to the frozenset of edges sharing an endpoint.
    """Graph, the vertices of which are edges of G,
    with two vertices being adjacent iff the corresponding
    edges share a vertex."""
    L = {}
    for x in G:
        for y in G[x]:
            nx = [frozenset([x,z]) for z in G[x] if z != y]
            ny = [frozenset([y,z]) for z in G[y] if z != x]
            L[frozenset([x,y])] = frozenset(nx+ny)
    return L

def faces(G):
    # [FUNCTION DOC] HELPER: finds triangles, squares and pentagons in graph G (mapping of vertex
    # to neighbors) by walking up to five vertices back to the start; returns a set of frozensets
    # of vertices.
    'Return a set of faces in G.  Where a face is a set of vertices on that face'
    # currently limited to triangles,squares, and pentagons
    f = set()
    for v1, edges in G.items():
        for v2 in edges:
            for v3 in G[v2]:
                if v1 == v3:
                    continue
                if v1 in G[v3]:
                    f.add(frozenset([v1, v2, v3]))
                else:
                    for v4 in G[v3]:
                        if v4 == v2:
                            continue
                        if v1 in G[v4]:
                            f.add(frozenset([v1, v2, v3, v4]))
                        else:
                            for v5 in G[v4]:
                                if v5 == v3 or v5 == v2:
                                    continue
                                if v1 in G[v5]:
                                    f.add(frozenset([v1, v2, v3, v4, v5]))
    return f


class TestGraphs(__TestCase):
    # [CLASS DOC] Application tests building graphs with sets (hypercube and line graph).

    def test_cube(self):
        # [FUNCTION DOC] Checks the 3-cube has 8 vertices of degree 3 whose neighbors are
        # vertices, and 6 faces that each have 4 vertices.

        g = cube(3)                             # vert --> {v1, v2, v3}
        vertices1 = set(g)
        self.assertEqual(len(vertices1), 8)     # eight vertices
        for edge in g.values():
            self.assertEqual(len(edge), 3)      # each vertex connects to three edges
        vertices2 = set(v for edges in g.values() for v in edges)
        self.assertEqual(vertices1, vertices2)  # edge vertices in original set

        cubefaces = faces(g)
        self.assertEqual(len(cubefaces), 6)     # six faces
        for face in cubefaces:
            self.assertEqual(len(face), 4)      # each face is a square

    def test_cuboctahedron(self):
        # [FUNCTION DOC] Checks the line graph of the 3-cube (cuboctahedron) has 12 vertices
        # of degree 4, 8 triangular and 6 square faces, and that each vertex is a pair of
        # cube vertices.

        # http://en.wikipedia.org/wiki/Cuboctahedron
        # 8 triangular faces and 6 square faces
        # 12 identical vertices each connecting a triangle and square

        g = cube(3)
        cuboctahedron = linegraph(g)            # V( --> {V1, V2, V3, V4}
        self.assertEqual(len(cuboctahedron), 12)# twelve vertices

        vertices = set(cuboctahedron)
        for edges in cuboctahedron.values():
            self.assertEqual(len(edges), 4)     # each vertex connects to four other vertices
        othervertices = set(edge for edges in cuboctahedron.values() for edge in edges)
        self.assertEqual(vertices, othervertices)   # edge vertices in original set

        cubofaces = faces(cuboctahedron)
        facesizes = collections.defaultdict(int)
        for face in cubofaces:
            facesizes[len(face)] += 1
        self.assertEqual(facesizes[3], 8)       # eight triangular faces
        self.assertEqual(facesizes[4], 6)       # six square faces

        for vertex in cuboctahedron:
            edge = vertex                       # Cuboctahedron vertices are edges in Cube
            self.assertEqual(len(edge), 2)      # Two cube vertices define an edge
            for cubevert in edge:
                self.assertIn(cubevert, g)


if __name__ == "__main__":
    run_tests()
