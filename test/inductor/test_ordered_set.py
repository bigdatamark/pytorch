# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Test suite for the OrderedSet container in torch.utils._ordered_set, ported
# from the CPython set and frozenset test suite. Covers construction, set algebra (union,
# intersection, difference, symmetric difference and their operator and in-place forms),
# subset ordering, mutation, copying, pickling, iterator arguments of many kinds, hash-related
# bug regressions and graph-based identity checks. Tests that depend on features OrderedSet
# lacks are skipped. No import-time side effects beyond imports and class definitions.
import collections
import collections.abc
import copy
import gc
import operator
import pickle
import unittest
import warnings
import weakref

from test import support

from torch.testing._internal.common_utils import TestCase
from torch.utils._ordered_set import OrderedSet


class PassThru(Exception):
    # [CLASS DOC] HELPER: Marker exception raised by the generator helper to check that errors
    # from an iterable argument propagate unchanged out of set constructors and methods.
    pass


def check_pass_thru():
    # [FUNCTION DOC] HELPER: Generator function whose body raises the marker exception on the
    # first request for an item, so it never yields. Used as an iterable argument that always
    # fails.
    raise PassThru
    yield 1


class BadCmp:
    # [CLASS DOC] HELPER: Element type whose hash is the constant one and whose equality
    # comparison always raises a runtime error; forces collision-time comparison failures.
    def __hash__(self):
        # [FUNCTION DOC] HELPER: Returns the constant hash value one so all instances collide.
        return 1

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality always raises a runtime error, whatever the other
        # operand.
        raise RuntimeError


class ReprWrapper:
    # [CLASS DOC] HELPER: Element used to build a self-referencing set; its value attribute is
    # assigned by the test after construction and the representation is that value's.
    "Used to test self-referential repr() calls"

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns the representation of the wrapped value attribute,
        # which must have been assigned by the caller beforehand (otherwise an attribute error
        # is raised).
        return repr(self.value)


class HashCountingInt(int):
    # [CLASS DOC] HELPER: Integer subclass that counts how many times its hash is computed, in
    # an instance counter attribute starting at zero.
    "int-like object that counts the number of times __hash__ is called"

    def __init__(self, *args):
        # [FUNCTION DOC] HELPER: Accepts and ignores any arguments and initializes the hash
        # counter attribute to zero.
        self.hash_count = 0

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Increments the hash counter and returns the ordinary integer
        # hash.
        self.hash_count += 1
        return int.__hash__(self)


class TestJointOps(TestCase):
    # [CLASS DOC] Base mixin of tests shared by the set tests; checks behavior of the set type
    # under test (class attribute for the tested type and for the type expected from
    # operations, both OrderedSet here) against a built-in dict and string models. Subclassed
    # by TestSet, which adds mutation tests. setUp builds the shared fixtures.
    # Tests common to both OrderedSet and frozenset
    thetype = OrderedSet
    basetype = OrderedSet

    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, then sets the word and a second
        # word, a string of all ASCII letters, a set of the tested type built from the first
        # word, and a dict with the same keys (used as a reference model).
        super().setUp()
        self.word = word = "simsalabim"
        self.otherword = "madagascar"
        self.letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.s = self.thetype(word)
        self.d = dict.fromkeys(word)

    def test_new_or_init(self):
        # [FUNCTION DOC] Checks the constructor rejects too many positional arguments and that
        # calling the initializer directly with an unexpected keyword argument raises a type
        # error.
        self.assertRaises(TypeError, self.thetype, [], 2)
        self.assertRaises(TypeError, OrderedSet().__init__, a=1)

    def test_uniquification(self):
        # [FUNCTION DOC] Checks that building a set from a word gives the same distinct
        # elements as the dict model, that an error raised inside an iterable propagates, and
        # that an unhashable element raises a type error.
        actual = sorted(self.s)
        expected = sorted(self.d)
        self.assertEqual(actual, expected)
        self.assertRaises(PassThru, self.thetype, check_pass_thru())
        self.assertRaises(TypeError, self.thetype, [[]])

    def test_len(self):
        # [FUNCTION DOC] Checks the length of the set matches the length of the reference
        # dict.
        self.assertEqual(len(self.s), len(self.d))

    def test_contains(self):
        # [FUNCTION DOC] Checks membership for every letter agrees with the dict model, and
        # that testing an unhashable value raises a type error. A commented-out check for
        # nested sets is left unused.
        for c in self.letters:
            self.assertEqual(c in self.s, c in self.d)
        self.assertRaises(TypeError, self.s.__contains__, [[]])
        #
        # s = self.thetype([frozenset(self.letters)])
        # self.assertIn(self.thetype(self.letters), s)

    def test_union(self):
        # [FUNCTION DOC] Checks union: membership is the OR of both inputs, the original is
        # not changed, the result has the expected result type, iterable errors propagate and
        # unhashable elements raise a type error. Also checks the result equals the expected
        # OrderedSet for several iterable types as argument, with one and two arguments, and
        # with an empty set argument repeated (regression for issue 6573).
        u = self.s.union(self.otherword)
        for c in self.letters:
            self.assertEqual(c in u, c in self.d or c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(u), self.basetype)
        self.assertRaises(PassThru, self.s.union, check_pass_thru())
        self.assertRaises(TypeError, self.s.union, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(self.thetype("abcba").union(C("cdc")), OrderedSet("abcd"))
            self.assertEqual(
                self.thetype("abcba").union(C("efgfe")), OrderedSet("abcefg")
            )
            self.assertEqual(self.thetype("abcba").union(C("ccb")), OrderedSet("abc"))
            self.assertEqual(self.thetype("abcba").union(C("ef")), OrderedSet("abcef"))
            self.assertEqual(
                self.thetype("abcba").union(C("ef"), C("fg")), OrderedSet("abcefg")
            )

        # Issue #6573
        x = self.thetype()
        self.assertEqual(
            x.union(OrderedSet([1]), x, OrderedSet([2])), self.thetype([1, 2])
        )

    def test_or(self):
        # [FUNCTION DOC] Checks the bitwise-or operator equals the union method for OrderedSet
        # and frozenset operands. Using a plain string operand is allowed to raise a type
        # error or not; the check that it must raise is commented out.
        i = self.s.union(self.otherword)
        self.assertEqual(self.s | OrderedSet(self.otherword), i)
        self.assertEqual(self.s | frozenset(self.otherword), i)
        try:
            self.s | self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s|t did not screen-out general iterables")

    def test_intersection(self):
        # [FUNCTION DOC] Checks intersection: membership is the AND of both inputs, the
        # original set is unchanged, the result type is as expected, iterable errors
        # propagate, and results for many iterable arguments (including two arguments) equal
        # the expected sets. Finally checks that intersection with no arguments returns a new
        # object rather than the same one.
        i = self.s.intersection(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.intersection, check_pass_thru())
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").intersection(C("cdc")), OrderedSet("cc")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("efgfe")), OrderedSet("")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("ccb")), OrderedSet("bc")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("ef")), OrderedSet("")
            )
            self.assertEqual(
                self.thetype("abcba").intersection(C("cbcf"), C("bag")), OrderedSet("b")
            )
        s = self.thetype("abcba")
        z = s.intersection()
        if self.thetype == frozenset():
            self.assertEqual(id(s), id(z))
        else:
            self.assertNotEqual(id(s), id(z))

    def test_isdisjoint(self):
        # [FUNCTION DOC] Checks the disjointness method returns a real boolean agreeing with a
        # pure-Python reference over all pairs of sample strings and several container types.
        def f(s1, s2):
            # [FUNCTION DOC] HELPER: Reference implementation of disjointness: true when the
            # intersection of the two arguments is empty.
            "Pure python equivalent of isdisjoint()"
            return not OrderedSet(s1).intersection(s2)

        for large in "", "a", "ab", "abc", "ababac", "cdc", "cc", "efgfe", "ccb", "ef":
            s1 = self.thetype(large)
            for rarg in (
                "",
                "a",
                "ab",
                "abc",
                "ababac",
                "cdc",
                "cc",
                "efgfe",
                "ccb",
                "ef",
            ):
                for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                    s2 = C(rarg)
                    actual = s1.isdisjoint(s2)
                    expected = f(s1, s2)
                    self.assertEqual(actual, expected)
                    self.assertTrue(actual is True or actual is False)

    def test_and(self):
        # [FUNCTION DOC] Checks the and operator equals the intersection method for OrderedSet
        # and frozenset operands; a plain string operand may raise a type error, but is not
        # required to.
        i = self.s.intersection(self.otherword)
        self.assertEqual(self.s & OrderedSet(self.otherword), i)
        self.assertEqual(self.s & frozenset(self.otherword), i)
        try:
            self.s & self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s&t did not screen-out general iterables")

    def test_difference(self):
        # [FUNCTION DOC] Checks difference: membership is in the first but not the others, the
        # original set is unchanged, the result type is as expected, iterable errors
        # propagate, unhashable elements raise a type error, and many iterable arguments
        # (including none and two) give the expected sets.
        i = self.s.difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, c in self.d and c not in self.otherword)
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").difference(C("cdc")), OrderedSet("ab")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("efgfe")), OrderedSet("abc")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("ccb")), OrderedSet("a")
            )
            self.assertEqual(
                self.thetype("abcba").difference(C("ef")), OrderedSet("abc")
            )
            self.assertEqual(self.thetype("abcba").difference(), OrderedSet("abc"))
            self.assertEqual(
                self.thetype("abcba").difference(C("a"), C("b")), OrderedSet("c")
            )

    def test_sub(self):
        # [FUNCTION DOC] Checks the minus operator equals the difference method for OrderedSet
        # and frozenset operands; a plain string operand may raise a type error, but is not
        # required to.
        i = self.s.difference(self.otherword)
        self.assertEqual(self.s - OrderedSet(self.otherword), i)
        self.assertEqual(self.s - frozenset(self.otherword), i)
        try:
            self.s - self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s-t did not screen-out general iterables")

    def test_symmetric_difference(self):
        # [FUNCTION DOC] Checks symmetric difference: membership is the exclusive-or of both
        # inputs, the original set is unchanged, the result type is as expected, iterable
        # errors propagate, unhashable elements raise a type error, and several iterable
        # arguments give expected sets.
        i = self.s.symmetric_difference(self.otherword)
        for c in self.letters:
            self.assertEqual(c in i, (c in self.d) ^ (c in self.otherword))
        self.assertEqual(self.s, self.thetype(self.word))
        self.assertEqual(type(i), self.basetype)
        self.assertRaises(PassThru, self.s.symmetric_difference, check_pass_thru())
        self.assertRaises(TypeError, self.s.symmetric_difference, [[]])
        for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("cdc")),
                OrderedSet("abd"),  # codespell:ignore
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("efgfe")),
                OrderedSet("abcefg"),
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("ccb")), OrderedSet("a")
            )
            self.assertEqual(
                self.thetype("abcba").symmetric_difference(C("ef")), OrderedSet("abcef")
            )

    def test_xor(self):
        # [FUNCTION DOC] Checks the caret operator equals the symmetric difference method for
        # OrderedSet and frozenset operands; a plain string operand may raise a type error,
        # but is not required to.
        i = self.s.symmetric_difference(self.otherword)
        self.assertEqual(self.s ^ OrderedSet(self.otherword), i)
        self.assertEqual(self.s ^ frozenset(self.otherword), i)
        try:
            self.s ^ self.otherword
        except TypeError:
            pass
        # else:
        #     self.fail("s^t did not screen-out general iterables")

    def test_equality(self):
        # [FUNCTION DOC] Checks equality and inequality against OrderedSet and frozenset with
        # the same or different contents, and that a set never equals a plain string.
        self.assertEqual(self.s, OrderedSet(self.word))
        self.assertEqual(self.s, frozenset(self.word))
        self.assertEqual(self.s == self.word, False)
        self.assertNotEqual(self.s, OrderedSet(self.otherword))
        self.assertNotEqual(self.s, frozenset(self.otherword))
        self.assertEqual(self.s != self.word, True)

    def test_setOfFrozensets(self):
        # [FUNCTION DOC] Checks a set built from several frozensets collapses equal ones,
        # leaving three distinct elements.
        t = map(frozenset, ["abcdef", "bcd", "bdcb", "fed", "fedccba"])
        s = self.thetype(t)
        self.assertEqual(len(s), 3)

    def test_sub_and_super(self):
        # [FUNCTION DOC] Checks the ordering operators and the subset and superset methods on
        # small sets: proper subset, equal, and unrelated cases, with iterable (string)
        # arguments to the methods.
        p, q, r = map(self.thetype, ["ab", "abcde", "def"])
        self.assertTrue(p < q)
        self.assertTrue(p <= q)
        self.assertTrue(q <= q)
        self.assertTrue(q > p)
        self.assertTrue(q >= p)
        self.assertFalse(q < r)
        self.assertFalse(q <= r)
        self.assertFalse(q > r)
        self.assertFalse(q >= r)
        self.assertTrue(OrderedSet("a").issubset("abc"))
        self.assertTrue(OrderedSet("abc").issuperset("a"))
        self.assertFalse(OrderedSet("a").issubset("cbs"))
        self.assertFalse(OrderedSet("cbs").issuperset("a"))

    def test_pickling(self):
        # [FUNCTION DOC] For every pickle protocol, round-trips the set through serialization
        # and checks equality. For set subclasses other than OrderedSet and frozenset, also
        # sets extra attributes beforehand and checks they survive the round trip and that no
        # unrelated attribute appears, then removes them.
        for i in range(pickle.HIGHEST_PROTOCOL + 1):
            if type(self.s) not in (OrderedSet, frozenset):
                self.s.x = ["x"]
                self.s.z = ["z"]
            p = pickle.dumps(self.s, i)
            dup = pickle.loads(p)
            self.assertEqual(self.s, dup, lambda msg: f"{msg}\n{self.s} != {dup}")
            if type(self.s) not in (OrderedSet, frozenset):
                self.assertEqual(self.s.x, dup.x)
                self.assertEqual(self.s.z, dup.z)
                self.assertFalse(hasattr(self.s, "y"))
                del self.s.x, self.s.z

    @unittest.skip("Pickling nyi")
    def test_iterator_pickling(self):
        # [FUNCTION DOC] Skipped test (pickling not implemented). Would pickle the iterator
        # over the set at each protocol, check it is an iterator and rebuilds the same set,
        # and that a partially consumed iterator rebuilds the set minus the consumed element.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            itorg = iter(self.s)
            data = self.thetype(self.s)
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # Set iterators unpickle as list iterators due to the
            # undefined order of OrderedSet items.
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
        # [FUNCTION DOC] Checks that a deep copy of a set creates a new set object whose
        # element is a deep copy of the original element (the element's custom copy hook
        # increments its value by one), not the same object.
        class Tracer:
            # [CLASS DOC] HELPER: Element with an integer value attribute, hashed by that
            # value, whose deep copy hook returns a new instance with the value increased by
            # one.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value attribute.
                self.value = value

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value as the hash.
                return self.value

            def __deepcopy__(self, memo=None):
                # [FUNCTION DOC] HELPER: Returns a new instance holding the value plus one, so
                # tests can tell a copy from the original. The memo argument is unused.
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
        # [FUNCTION DOC] Builds a set of a thousand fresh objects each referencing the set,
        # itself and a set containing it, creating a nest of reference cycles to exercise
        # garbage collection of the container. Makes no assertions; passes if nothing crashes.
        # Create a nest of cycles to exercise overall ref count check
        class A:
            # [CLASS DOC] HELPER: Empty class; instances get attributes assigned by the test.
            pass

        s = OrderedSet(A() for i in range(1000))
        for elem in s:
            elem.cycle = s
            elem.sub = elem
            elem.OrderedSet = OrderedSet([elem])

    def test_subclass_with_custom_hash(self):
        # [FUNCTION DOC] Regression for bug 1257731: defines a subclass of the tested type
        # with an identity-based hash, then uses an instance as an element of a plain
        # OrderedSet through add, remove, add and discard, checking membership once.
        # Bug #1257731
        class H(self.thetype):
            # [CLASS DOC] HELPER: Subclass of the tested set type that overrides the hash to
            # be derived from the object id, masked to 31 bits.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the object id masked to a non-negative 31-bit
                # integer.
                return int(id(self) & 0x7FFFFFFF)

        s = H()
        f = OrderedSet()
        f.add(s)
        self.assertIn(s, f)
        f.remove(s)
        f.add(s)
        f.discard(s)

    def test_badcmp(self):
        # [FUNCTION DOC] Uses elements whose comparison raises a runtime error: checks
        # construction from two colliding elements, a membership test and (when add exists)
        # add, discard and remove all propagate that runtime error.
        s = self.thetype([BadCmp()])
        # Detect comparison errors during insertion and lookup
        self.assertRaises(RuntimeError, self.thetype, [BadCmp(), BadCmp()])
        self.assertRaises(RuntimeError, s.__contains__, BadCmp())
        # Detect errors during mutating operations
        if hasattr(s, "add"):
            self.assertRaises(RuntimeError, s.add, BadCmp())
            self.assertRaises(RuntimeError, s.discard, BadCmp())
            self.assertRaises(RuntimeError, s.remove, BadCmp())

    @unittest.skip("Different repr")
    def test_cyclical_repr(self):
        # [FUNCTION DOC] Skipped test (different repr). Builds a set containing an object that
        # refers back to the set and checks the textual representation shows the recursion
        # marker, in a form depending on whether the tested type is OrderedSet.
        w = ReprWrapper()
        s = self.thetype([w])
        w.value = s
        if self.thetype == OrderedSet:
            self.assertEqual(repr(s), "{OrderedSet(...)}")
        else:
            name = repr(s).partition("(")[0]  # strip class name
            self.assertEqual(repr(s), "%s({%s(...)})" % (name, name))  # noqa: UP031

    @unittest.skip("Different hashing")
    def test_do_not_rehash_dict_keys(self):
        # [FUNCTION DOC] Skipped test (different hashing). Checks that building sets and
        # running difference, symmetric difference update and dict conversion on a dict keyed
        # by hash-counting integers never recomputes the hashes, so the total count stays at
        # the number of keys.
        n = 10
        d = dict.fromkeys(map(HashCountingInt, range(n)))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s = self.thetype(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        s.difference(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        if hasattr(s, "symmetric_difference_update"):
            s.symmetric_difference_update(d)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d2 = dict.fromkeys(OrderedSet(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d))
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        d3 = dict.fromkeys(frozenset(d), 123)
        self.assertEqual(sum(elem.hash_count for elem in d), n)
        self.assertEqual(d3, dict.fromkeys(d, 123))

    def test_container_iterator(self):
        # [FUNCTION DOC] Regression for bug 3680: stores an iterator over a set that contains
        # an object into that object, deletes the other references, runs garbage collection
        # and asserts the weak reference to the object is cleared, i.e. the cycle is
        # collectable.
        # Bug #3680: tp_traverse was not implemented for OrderedSet iterator object
        class C:
            # [CLASS DOC] HELPER: Empty class; instances get an attribute assigned by the
            # test.
            pass

        obj = C()
        ref = weakref.ref(obj)
        container = OrderedSet([obj, 1])
        obj.x = iter(container)
        del obj, container
        gc.collect()
        self.assertTrue(ref() is None, "Cycle was not collected")

    def test_free_after_iterating(self):
        # [FUNCTION DOC] Delegates to the support helper that checks the container is freed
        # after its iterator is exhausted, using the iter function and the tested type.
        support.check_free_after_iterating(self, iter, self.thetype)


class TestSet(TestJointOps, TestCase):
    # [CLASS DOC] Tests for OrderedSet itself: constructor behavior, set literals, hashing,
    # in-place and mutating operations, and rich comparison. Inherits the shared tests from
    # the joint mixin, with the tested type and result type set to OrderedSet.
    thetype = OrderedSet
    basetype = OrderedSet

    def test_init(self):
        # [FUNCTION DOC] Checks that calling the initializer again on an existing set replaces
        # its contents, leaving a set equal to the new iterable only, and that too many
        # arguments or a non-iterable argument raises a type error.
        s = self.thetype()
        s.__init__(self.word)
        self.assertEqual(s, OrderedSet(self.word))
        s.__init__(self.otherword)
        self.assertEqual(s, OrderedSet(self.otherword))
        self.assertRaises(TypeError, s.__init__, s, 2)
        self.assertRaises(TypeError, s.__init__, 1)

    def test_constructor_identity(self):
        # [FUNCTION DOC] Checks constructing a set from another set creates a distinct object.
        s = self.thetype(range(3))
        t = self.thetype(s)
        self.assertNotEqual(id(s), id(t))

    def test_set_literal(self):
        # [FUNCTION DOC] Checks an OrderedSet built from a list equals the built-in set with
        # the same elements.
        s = OrderedSet([1, 2, 3])
        t = {1, 2, 3}
        self.assertEqual(s, t)

    def test_set_literal_insertion_order(self):
        # [FUNCTION DOC] Checks that a built-in set literal with equal numeric values keeps
        # one element and that the stored value is the integer, i.e. the leftmost one (issue
        # 26020). This tests the built-in set rather than OrderedSet.
        # SF Issue #26020 -- Expect left to right insertion
        s = {1, 1.0, True}  # noqa: B033
        self.assertEqual(len(s), 1)
        stored_value = s.pop()
        self.assertEqual(type(stored_value), int)

    def test_set_literal_evaluation_order(self):
        # [FUNCTION DOC] Checks the built-in set literal evaluates its elements left to right,
        # using a recording helper. This tests the built-in set rather than OrderedSet.
        # Expect left to right expression evaluation
        events = []

        def record(obj):
            # [FUNCTION DOC] HELPER: Appends its argument to the events list of the enclosing
            # test and returns nothing.
            events.append(obj)

        s = {record(1), record(2), record(3)}
        self.assertEqual(events, [1, 2, 3])

    def test_hash(self):
        # [FUNCTION DOC] Checks that hashing a set raises a type error since it is mutable.
        self.assertRaises(TypeError, hash, self.s)

    def test_clear(self):
        # [FUNCTION DOC] Checks that clearing the set leaves an empty set of length zero.
        self.s.clear()
        self.assertEqual(self.s, OrderedSet())
        self.assertEqual(len(self.s), 0)

    def test_copy(self):
        # [FUNCTION DOC] Checks a shallow copy equals the original, is a different object, and
        # has the expected result type.
        dup = self.s.copy()
        self.assertEqual(self.s, dup)
        self.assertNotEqual(id(self.s), id(dup))
        self.assertEqual(type(dup), self.basetype)

    def test_add(self):
        # [FUNCTION DOC] Checks adding an element makes it a member, adding it again leaves
        # the set unchanged, and adding an unhashable element raises a type error.
        self.s.add("Q")
        self.assertIn("Q", self.s)
        dup = self.s.copy()
        self.s.add("Q")
        self.assertEqual(self.s, dup)
        self.assertRaises(TypeError, self.s.add, [])

    def test_remove(self):
        # [FUNCTION DOC] Checks removing a present element works, removing a missing element
        # raises a key error and removing an unhashable one raises a type error. Frozenset-
        # element checks are commented out (not yet implemented).
        self.s.remove("a")
        self.assertNotIn("a", self.s)
        self.assertRaises(KeyError, self.s.remove, "Q")
        self.assertRaises(TypeError, self.s.remove, [])
        # NYI: __as_immutable__
        # s = self.thetype([frozenset(self.word)])
        # self.assertIn(self.thetype(self.word), s)
        # s.remove(self.thetype(self.word))
        # self.assertNotIn(self.thetype(self.word), s)
        # self.assertRaises(KeyError, self.s.remove, self.thetype(self.word))

    def test_remove_keyerror_unpacking(self):
        # [FUNCTION DOC] Checks the key error raised on removing a missing element carries the
        # original key as its first argument, for a string and a tuple key; fails if no error
        # is raised.
        # https://bugs.python.org/issue1576657
        for v1 in ["Q", (1,)]:
            try:
                self.s.remove(v1)
            except KeyError as e:
                v2 = e.args[0]
                self.assertEqual(v1, v2)
            else:
                self.fail()

    def test_remove_keyerror_set(self):
        # [FUNCTION DOC] Checks removing a missing set-valued key raises some exception; fails
        # if nothing is raised. The identity check on the error argument is commented out.
        key = self.thetype([3, 4])
        try:
            self.s.remove(key)
        except Exception:
            pass
            # self.assertTrue(e.args[0] is key,
            #              "KeyError should be {0}, not {1}".format(key,
            #                                                       e.args[0]))
        else:
            self.fail()

    def test_discard(self):
        # [FUNCTION DOC] Checks discard removes a present element, silently ignores a missing
        # one, and raises a type error for an unhashable one. Frozenset-element checks are
        # commented out.
        self.s.discard("a")
        self.assertNotIn("a", self.s)
        self.s.discard("Q")
        self.assertRaises(TypeError, self.s.discard, [])
        # NYI: __as_immutable__
        # s = self.thetype([frozenset(self.word)])
        # self.assertIn(self.thetype(self.word), s)
        # s.discard(self.thetype(self.word))
        # self.assertNotIn(self.thetype(self.word), s)
        # s.discard(self.thetype(self.word))

    def test_pop(self):
        # [FUNCTION DOC] Pops every element from the set, checking each is gone afterwards,
        # then checks popping from the empty set raises a key error.
        for _ in range(len(self.s)):
            elem = self.s.pop()
            self.assertNotIn(elem, self.s)
        self.assertRaises(KeyError, self.s.pop)

    def test_update(self):
        # [FUNCTION DOC] Checks update returns nothing and adds all letters, iterable errors
        # propagate, unhashable elements raise a type error, and updating with one or two
        # iterables of many types gives the expected union.
        retval = self.s.update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            self.assertIn(c, self.s)
        self.assertRaises(PassThru, self.s.update, check_pass_thru())
        self.assertRaises(TypeError, self.s.update, [[]])
        for p, q in (
            ("cdc", "abcd"),
            ("efgfe", "abcefg"),
            ("ccb", "abc"),
            ("ef", "abcef"),
        ):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))
        for p in ("cdc", "efgfe", "ccb", "ef", "abcda"):
            q = "ahi"
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.update(C(p), C(q)), None)
                self.assertEqual(s, OrderedSet(s) | OrderedSet(p) | OrderedSet(q))

    def test_ior(self):
        # [FUNCTION DOC] Checks the in-place or operator adds all elements of the other set.
        self.s |= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            self.assertIn(c, self.s)

    def test_intersection_update(self):
        # [FUNCTION DOC] Checks intersection update returns nothing and keeps only common
        # elements, iterable errors propagate, unhashable elements raise a type error, and
        # results with one or two iterable arguments of many types equal the expected
        # intersections.
        retval = self.s.intersection_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.intersection_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.intersection_update, [[]])
        for p, q in (("cdc", "c"), ("efgfe", ""), ("ccb", "bc"), ("ef", "")):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.intersection_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))
                ss = "abcba"
                s = self.thetype(ss)
                t = "cbc"
                self.assertEqual(s.intersection_update(C(p), C(t)), None)
                self.assertEqual(s, OrderedSet("abcba") & OrderedSet(p) & OrderedSet(t))

    def test_iand(self):
        # [FUNCTION DOC] Checks the in-place and operator keeps only elements present in both
        # sets.
        self.s &= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if c in self.otherword and c in self.word:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_difference_update(self):
        # [FUNCTION DOC] Checks difference update returns nothing and removes elements of the
        # argument, iterable errors propagate, unhashable elements raise type errors (also for
        # symmetric difference update), and that no arguments, one and two iterable arguments
        # give expected sets.
        retval = self.s.difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(PassThru, self.s.difference_update, check_pass_thru())
        self.assertRaises(TypeError, self.s.difference_update, [[]])
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (("cdc", "ab"), ("efgfe", "abc"), ("ccb", "a"), ("ef", "abc")):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.difference_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))

                s = self.thetype("abcdefghih")
                s.difference_update()
                self.assertEqual(s, self.thetype("abcdefghih"))

                s = self.thetype("abcdefghih")
                s.difference_update(C("aba"))
                self.assertEqual(s, self.thetype("cdefghih"))

                s = self.thetype("abcdefghih")
                s.difference_update(C("cdc"), C("aba"))
                self.assertEqual(s, self.thetype("efghih"))

    def test_isub(self):
        # [FUNCTION DOC] Checks the in-place minus operator removes elements of the other set.
        self.s -= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if c in self.word and c not in self.otherword:
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_symmetric_difference_update(self):
        # [FUNCTION DOC] Checks symmetric difference update returns nothing and leaves
        # elements in exactly one of the two inputs, iterable errors propagate, unhashable
        # elements raise a type error, and results for many iterable types equal the expected
        # sets.
        retval = self.s.symmetric_difference_update(self.otherword)
        self.assertEqual(retval, None)
        for c in self.word + self.otherword:
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)
        self.assertRaises(
            PassThru, self.s.symmetric_difference_update, check_pass_thru()
        )
        self.assertRaises(TypeError, self.s.symmetric_difference_update, [[]])
        for p, q in (
            ("cdc", "abd"),  # codespell:ignore
            ("efgfe", "abcefg"),
            ("ccb", "a"),
            ("ef", "abcef"),
        ):
            for C in OrderedSet, frozenset, dict.fromkeys, str, list, tuple:
                s = self.thetype("abcba")
                self.assertEqual(s.symmetric_difference_update(C(p)), None)
                self.assertEqual(s, OrderedSet(q))

    def test_ixor(self):
        # [FUNCTION DOC] Checks the in-place caret operator leaves elements in exactly one of
        # the sets.
        self.s ^= OrderedSet(self.otherword)
        for c in self.word + self.otherword:
            if (c in self.word) ^ (c in self.otherword):
                self.assertIn(c, self.s)
            else:
                self.assertNotIn(c, self.s)

    def test_inplace_on_self(self):
        # [FUNCTION DOC] Checks in-place or and and with a set on itself leave it unchanged,
        # while in-place minus and caret with itself empty it.
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

    @unittest.skip("Slots interferes with weakrefs")
    def test_weakref(self):
        # [FUNCTION DOC] Skipped test (slots interfere with weak references). Would check a
        # weak proxy to a set stringifies like the set and raises a reference error after the
        # set is deleted.
        s = self.thetype("gallahad")
        p = weakref.proxy(s)
        self.assertEqual(str(p), str(s))
        s = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, str, p)

    def test_rich_compare(self):
        # [FUNCTION DOC] Checks that comparing a built-in set with a custom object falls back
        # to the object's reflected comparison methods: each operator on the left set triggers
        # the opposite method on the right object, recorded by a flag. This tests the built-in
        # set.
        class TestRichSetCompare:
            # [CLASS DOC] HELPER: Object defining all four ordering comparisons; each sets a
            # flag attribute recording it was called and returns false.
            def __gt__(self, some_set):
                # [FUNCTION DOC] HELPER: Records that greater-than was called and returns
                # false.
                self.gt_called = True
                return False

            def __lt__(self, some_set):
                # [FUNCTION DOC] HELPER: Records that less-than was called and returns false.
                self.lt_called = True
                return False

            def __ge__(self, some_set):
                # [FUNCTION DOC] HELPER: Records that greater-or-equal was called and returns
                # false.
                self.ge_called = True
                return False

            def __le__(self, some_set):
                # [FUNCTION DOC] HELPER: Records that less-or-equal was called and returns
                # false.
                self.le_called = True
                return False

        # This first tries the builtin rich OrderedSet comparison, which doesn't know
        # how to handle the custom object. Upon returning NotImplemented, the
        # corresponding comparison on the right object is invoked.
        myset = {1, 2, 3}

        myobj = TestRichSetCompare()
        myset < myobj  # noqa: B015
        self.assertTrue(myobj.gt_called)

        myobj = TestRichSetCompare()
        myset > myobj  # noqa: B015
        self.assertTrue(myobj.lt_called)

        myobj = TestRichSetCompare()
        myset <= myobj  # noqa: B015
        self.assertTrue(myobj.ge_called)

        myobj = TestRichSetCompare()
        myset >= myobj  # noqa: B015
        self.assertTrue(myobj.le_called)


# Tests taken from test_sets.py =============================================

empty_set = OrderedSet()

# ==============================================================================


class TestBasicOps(TestCase):
    # [CLASS DOC] Base mixin for the basic set operation tests; subclasses supply fixtures for
    # the set under test, its expected length, a dual set and sample values, through setUp.
    # Tests cover length, equality, copying, union, intersection, disjointness, symmetric
    # difference, difference, iteration and pickling against the empty set and the dual set.
    @unittest.skip("Different repr")
    def test_repr(self):
        # [FUNCTION DOC] Skipped test (different repr). If the subclass provides an expected
        # repr, compares it to the set's representation.
        if self.repr is not None:
            self.assertEqual(repr(self.OrderedSet), self.repr)

    def check_repr_against_values(self):
        # [FUNCTION DOC] HELPER: Checks the set representation is wrapped in braces and that
        # its comma-separated entries, sorted, equal the sorted representations of the
        # subclass' values.
        text = repr(self.OrderedSet)
        self.assertTrue(text.startswith("{"))
        self.assertTrue(text.endswith("}"))

        result = text[1:-1].split(", ")
        result.sort()
        sorted_repr_values = [repr(value) for value in self.values]
        sorted_repr_values.sort()
        self.assertEqual(result, sorted_repr_values)

    def test_length(self):
        # [FUNCTION DOC] Checks the set length equals the expected length from the fixture.
        self.assertEqual(len(self.OrderedSet), self.length)

    def test_self_equality(self):
        # [FUNCTION DOC] Checks the fixture set equals itself.
        self.assertEqual(self.OrderedSet, self.OrderedSet)

    def test_equivalent_equality(self):
        # [FUNCTION DOC] Checks the fixture set equals its duplicate built from the same
        # values.
        self.assertEqual(self.OrderedSet, self.dup)

    def test_copy(self):
        # [FUNCTION DOC] Checks a copy of the fixture set equals the duplicate.
        self.assertEqual(self.OrderedSet.copy(), self.dup)

    def test_self_union(self):
        # [FUNCTION DOC] Checks the union of the fixture set with itself equals the duplicate.
        result = self.OrderedSet | self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_empty_union(self):
        # [FUNCTION DOC] Checks the union of the fixture set with the empty set equals the
        # duplicate.
        result = self.OrderedSet | empty_set
        self.assertEqual(result, self.dup)

    def test_union_empty(self):
        # [FUNCTION DOC] Checks the union of the empty set with the fixture set equals the
        # duplicate.
        result = empty_set | self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_self_intersection(self):
        # [FUNCTION DOC] Checks the intersection of the fixture set with itself equals the
        # duplicate.
        result = self.OrderedSet & self.OrderedSet
        self.assertEqual(result, self.dup)

    def test_empty_intersection(self):
        # [FUNCTION DOC] Checks the intersection of the fixture set with the empty set is
        # empty.
        result = self.OrderedSet & empty_set
        self.assertEqual(result, empty_set)

    def test_intersection_empty(self):
        # [FUNCTION DOC] Checks the intersection of the empty set with the fixture set is
        # empty.
        result = empty_set & self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_self_isdisjoint(self):
        # [FUNCTION DOC] Checks the fixture set is disjoint from itself exactly when it is
        # empty.
        result = self.OrderedSet.isdisjoint(self.OrderedSet)
        self.assertEqual(result, not self.OrderedSet)

    def test_empty_isdisjoint(self):
        # [FUNCTION DOC] Checks the fixture set is disjoint from the empty set.
        result = self.OrderedSet.isdisjoint(empty_set)
        self.assertEqual(result, True)

    def test_isdisjoint_empty(self):
        # [FUNCTION DOC] Checks the empty set is disjoint from the fixture set.
        result = empty_set.isdisjoint(self.OrderedSet)
        self.assertEqual(result, True)

    def test_self_symmetric_difference(self):
        # [FUNCTION DOC] Checks the symmetric difference of the fixture set with itself is
        # empty.
        result = self.OrderedSet ^ self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_empty_symmetric_difference(self):
        # [FUNCTION DOC] Checks the symmetric difference of the fixture set with the empty set
        # equals the fixture set.
        result = self.OrderedSet ^ empty_set
        self.assertEqual(result, self.OrderedSet)

    def test_self_difference(self):
        # [FUNCTION DOC] Checks the difference of the fixture set with itself is empty.
        result = self.OrderedSet - self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_empty_difference(self):
        # [FUNCTION DOC] Checks the difference of the fixture set minus the empty set equals
        # the duplicate.
        result = self.OrderedSet - empty_set
        self.assertEqual(result, self.dup)

    def test_empty_difference_rev(self):
        # [FUNCTION DOC] Checks the difference of the empty set minus the fixture set is
        # empty.
        result = empty_set - self.OrderedSet
        self.assertEqual(result, empty_set)

    def test_iteration(self):
        # [FUNCTION DOC] Checks every element produced by iterating the fixture set is among
        # the fixture values. A length hint check is commented out.
        for v in self.OrderedSet:
            self.assertIn(v, self.values)
        # setiter = iter(self.OrderedSet)
        # self.assertEqual(setiter.__length_hint__(), len(self.OrderedSet))

    def test_pickling(self):
        # [FUNCTION DOC] For every pickle protocol, serializes and deserializes the fixture
        # set and checks equality with the original.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            p = pickle.dumps(self.OrderedSet, proto)
            copy = pickle.loads(p)
            self.assertEqual(
                self.OrderedSet,
                copy,
                lambda msg: f"{msg}\n{self.OrderedSet} != {copy}",
            )

    def test_issue_37219(self):
        # [FUNCTION DOC] Regression for issue 37219: checks that difference and difference
        # update on an empty set with a non-iterable integer argument raise a type error.
        with self.assertRaises(TypeError):
            OrderedSet().difference(123)
        with self.assertRaises(TypeError):
            OrderedSet().difference_update(123)


# ------------------------------------------------------------------------------


class TestBasicOpsEmpty(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with an empty set fixture.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, empty value list, the set and its
        # duplicate built from it, expected length zero and expected repr text for an empty
        # set.
        super().setUp()
        self.case = "empty OrderedSet"
        self.values = []
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 0
        self.repr = "OrderedSet()"


# ------------------------------------------------------------------------------


class TestBasicOpsSingleton(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a one-element numeric set fixture.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, a value list holding the number three,
        # the set and its duplicate, expected length one and the expected repr text.
        super().setUp()
        self.case = "unit OrderedSet (number)"
        self.values = [3]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 1
        self.repr = "{3}"

    def test_in(self):
        # [FUNCTION DOC] Checks the number three is a member of the fixture set.
        self.assertIn(3, self.OrderedSet)

    def test_not_in(self):
        # [FUNCTION DOC] Checks the number two is not a member of the fixture set.
        self.assertNotIn(2, self.OrderedSet)


# ------------------------------------------------------------------------------


class TestBasicOpsTuple(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a one-element set fixture holding a tuple.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, a value list holding one pair of a
        # number and a string, the set and its duplicate, expected length one and the expected
        # repr text.
        super().setUp()
        self.case = "unit OrderedSet (tuple)"
        self.values = [(0, "zero")]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 1
        self.repr = "{(0, 'zero')}"

    def test_in(self):
        # [FUNCTION DOC] Checks the stored tuple is a member of the fixture set.
        self.assertIn((0, "zero"), self.OrderedSet)

    def test_not_in(self):
        # [FUNCTION DOC] Checks an unrelated integer is not a member of the fixture set.
        self.assertNotIn(9, self.OrderedSet)


# ------------------------------------------------------------------------------


class TestBasicOpsTriple(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a three-element set of mixed types including a
    # function object.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, values of a number, a string and the
        # add operator function, the set and its duplicate, expected length three and no
        # expected repr.
        super().setUp()
        self.case = "triple OrderedSet"
        self.values = [0, "zero", operator.add]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3
        self.repr = None


# ------------------------------------------------------------------------------


class TestBasicOpsString(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a set of three single-character strings.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, the three string values, the set and
        # its duplicate, and expected length three.
        super().setUp()
        self.case = "string OrderedSet"
        self.values = ["a", "b", "c"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3

    @unittest.skip("Different repr")
    def test_repr(self):
        # [FUNCTION DOC] Skipped test (different repr). Would run the shared repr-versus-
        # values check.
        self.check_repr_against_values()


# ------------------------------------------------------------------------------


class TestBasicOpsBytes(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a set of three single-byte bytes objects.
    def setUp(self):
        # [FUNCTION DOC] Fixture: sets the case label, the three bytes values, the set and its
        # duplicate, and expected length three.
        super().setUp()
        self.case = "bytes OrderedSet"
        self.values = [b"a", b"b", b"c"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 3

    @unittest.skip("Different repr")
    def test_repr(self):
        # [FUNCTION DOC] Skipped test (different repr). Would run the shared repr-versus-
        # values check.
        self.check_repr_against_values()


# ------------------------------------------------------------------------------


class TestBasicOpsMixedStringBytes(TestBasicOps, TestCase):
    # [CLASS DOC] Basic operation tests with a set mixing two strings and two bytes objects.
    def setUp(self):
        # [FUNCTION DOC] Fixture: globally ignores bytes warnings via the warnings filter (a
        # process-wide side effect), then sets the case label, four mixed values, the set and
        # its duplicate, and expected length four.
        super().setUp()
        warnings.simplefilter("ignore", BytesWarning)
        self.case = "string and bytes OrderedSet"
        self.values = ["a", "b", b"a", b"b"]
        self.OrderedSet = OrderedSet(self.values)
        self.dup = OrderedSet(self.values)
        self.length = 4

    @unittest.skip("Different repr")
    def test_repr(self):
        # [FUNCTION DOC] Skipped test (different repr). Would run the shared repr-versus-
        # values check.
        self.check_repr_against_values()


del TestBasicOps
# ==============================================================================


def baditer():
# [FUNCTION DOC] HELPER: Generator function that raises a type error on first use, before ever
# yielding.
    raise TypeError
    yield True


def gooditer():
# [FUNCTION DOC] HELPER: Generator function yielding a single true value.
    yield True


class TestExceptionPropagation(TestCase):
    # [CLASS DOC] Tests that errors from an iterator argument are not swallowed by the set
    # constructor, and that changing set size during iteration is detected.
    """SF 628246:  Set constructor should not trap iterator TypeErrors"""

    def test_instanceWithException(self):
        # [FUNCTION DOC] Checks constructing a set from the failing generator raises a type
        # error.
        self.assertRaises(TypeError, OrderedSet, baditer())

    def test_instancesWithoutException(self):
        # [FUNCTION DOC] Checks sets can be constructed from a list, tuple, dict, range,
        # string and generator without raising.
        # All of these iterables should load without exception.
        OrderedSet([1, 2, 3])
        OrderedSet((1, 2, 3))
        OrderedSet({"one": 1, "two": 2, "three": 3})
        OrderedSet(range(3))
        OrderedSet("abc")
        OrderedSet(gooditer())

    def test_changingSizeWhileIterating(self):
        # [FUNCTION DOC] Iterates a set while updating it with a new element and expects a
        # runtime error; fails if no error occurs.
        s = OrderedSet([1, 2, 3])
        try:
            for _ in s:
                s.update([4])  # noqa: B909
        except RuntimeError:
            pass
        else:
            self.fail("no exception when changing size during iteration")


# ==============================================================================


class TestSetOfSets(TestCase):
    # [CLASS DOC] Tests holding frozensets as elements of an OrderedSet.
    def test_constructor(self):
        # [FUNCTION DOC] Checks a frozenset element keeps its type when popped, can be re-
        # added, removed leaving the set empty, and discarded when absent without error.
        inner = frozenset([1])
        outer = OrderedSet([inner])
        element = outer.pop()
        self.assertEqual(type(element), frozenset)
        outer.add(inner)  # Rebuild OrderedSet of sets with .add method
        outer.remove(inner)
        self.assertEqual(outer, OrderedSet())  # Verify that remove worked
        outer.discard(inner)  # Absence of KeyError indicates working fine


# ==============================================================================


class TestBinaryOps(TestCase):
    # [CLASS DOC] Tests binary operators between two sets, where one fixture set is a base and
    # the other is a subset, superset, overlapping or non-overlapping set. setUp builds the
    # base.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds the base set of the
        # numbers two, four and six.
        super().setUp()
        self.OrderedSet = OrderedSet((2, 4, 6))

    def test_eq(self):  # SF bug 643115
        # [FUNCTION DOC] Checks the base set equals a set built from the keys of a dict with
        # the same keys (bug 643115).
        self.assertEqual(self.OrderedSet, OrderedSet({2: 1, 4: 3, 6: 5}))

    def test_union_subset(self):
        # [FUNCTION DOC] Checks the or operator: union with a one-element subset gives the
        # base set unchanged.
        result = self.OrderedSet | OrderedSet([2])
        self.assertEqual(result, OrderedSet((2, 4, 6)))

    def test_union_superset(self):
        # [FUNCTION DOC] Checks the or operator: union with a superset adding eight gives the
        # superset.
        result = self.OrderedSet | OrderedSet([2, 4, 6, 8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))

    def test_union_overlap(self):
        # [FUNCTION DOC] Checks the or operator: union with an overlapping set gives the
        # merged elements two through six.
        result = self.OrderedSet | OrderedSet([3, 4, 5])
        self.assertEqual(result, OrderedSet([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        # [FUNCTION DOC] Checks the or operator: union with a disjoint set gives the base
        # elements plus eight.
        result = self.OrderedSet | OrderedSet([8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))

    def test_intersection_subset(self):
        # [FUNCTION DOC] Checks the and operator with a subset gives that subset.
        result = self.OrderedSet & OrderedSet((2, 4))
        self.assertEqual(result, OrderedSet((2, 4)))

    def test_intersection_superset(self):
        # [FUNCTION DOC] Checks the and operator with a superset gives the base set.
        result = self.OrderedSet & OrderedSet([2, 4, 6, 8])
        self.assertEqual(result, OrderedSet([2, 4, 6]))

    def test_intersection_overlap(self):
        # [FUNCTION DOC] Checks the and operator with an overlapping set gives the single
        # shared element four.
        result = self.OrderedSet & OrderedSet([3, 4, 5])
        self.assertEqual(result, OrderedSet([4]))

    def test_intersection_non_overlap(self):
        # [FUNCTION DOC] Checks the and operator with a disjoint set gives the empty set.
        result = self.OrderedSet & OrderedSet([8])
        self.assertEqual(result, empty_set)

    def test_isdisjoint_subset(self):
        # [FUNCTION DOC] Checks disjointness of the base set with a subset is false.
        result = self.OrderedSet.isdisjoint(OrderedSet((2, 4)))
        self.assertEqual(result, False)

    def test_isdisjoint_superset(self):
        # [FUNCTION DOC] Checks disjointness of the base set with a superset is false.
        result = self.OrderedSet.isdisjoint(OrderedSet([2, 4, 6, 8]))
        self.assertEqual(result, False)

    def test_isdisjoint_overlap(self):
        # [FUNCTION DOC] Checks disjointness of the base set with an overlapping set is false.
        result = self.OrderedSet.isdisjoint(OrderedSet([3, 4, 5]))
        self.assertEqual(result, False)

    def test_isdisjoint_non_overlap(self):
        # [FUNCTION DOC] Checks disjointness of the base set with a disjoint set is true.
        result = self.OrderedSet.isdisjoint(OrderedSet([8]))
        self.assertEqual(result, True)

    def test_sym_difference_subset(self):
        # [FUNCTION DOC] Checks the caret operator with a subset gives the remaining element
        # six.
        result = self.OrderedSet ^ OrderedSet((2, 4))
        self.assertEqual(result, OrderedSet([6]))

    def test_sym_difference_superset(self):
        # [FUNCTION DOC] Checks the caret operator with a superset gives only the extra
        # element eight.
        result = self.OrderedSet ^ OrderedSet((2, 4, 6, 8))
        self.assertEqual(result, OrderedSet([8]))

    def test_sym_difference_overlap(self):
        # [FUNCTION DOC] Checks the caret operator with an overlapping set gives the elements
        # in exactly one set.
        result = self.OrderedSet ^ OrderedSet((3, 4, 5))
        self.assertEqual(result, OrderedSet([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        # [FUNCTION DOC] Checks the caret operator with a disjoint set gives the union of
        # both.
        result = self.OrderedSet ^ OrderedSet([8])
        self.assertEqual(result, OrderedSet([2, 4, 6, 8]))


# ==============================================================================


class TestUpdateOps(TestCase):
    # [CLASS DOC] Tests in-place operators and their method forms on a base set of two, four
    # and six, comparing with the expected set after each update.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds the base set of the
        # numbers two, four and six.
        super().setUp()
        self.OrderedSet = OrderedSet((2, 4, 6))

    def test_union_subset(self):
        # [FUNCTION DOC] Checks the in-place or operator updates the base set to the expected
        # union with a subset.
        self.OrderedSet |= OrderedSet([2])
        self.assertEqual(self.OrderedSet, OrderedSet((2, 4, 6)))

    def test_union_superset(self):
        # [FUNCTION DOC] Checks the in-place or operator updates the base set to the expected
        # union with a superset.
        self.OrderedSet |= OrderedSet([2, 4, 6, 8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_union_overlap(self):
        # [FUNCTION DOC] Checks the in-place or operator updates the base set to the expected
        # union with an overlapping set.
        self.OrderedSet |= OrderedSet([3, 4, 5])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 4, 5, 6]))

    def test_union_non_overlap(self):
        # [FUNCTION DOC] Checks the in-place or operator updates the base set to the expected
        # union with a disjoint set.
        self.OrderedSet |= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_union_method_call(self):
        # [FUNCTION DOC] Checks the update method called with an overlapping set updates the
        # base set to the expected union.
        self.OrderedSet.update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 4, 5, 6]))

    def test_intersection_subset(self):
        # [FUNCTION DOC] Checks the in-place and operator updates the base set to the expected
        # intersection with a subset.
        self.OrderedSet &= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet((2, 4)))

    def test_intersection_superset(self):
        # [FUNCTION DOC] Checks the in-place and operator updates the base set to the expected
        # intersection with a superset.
        self.OrderedSet &= OrderedSet([2, 4, 6, 8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6]))

    def test_intersection_overlap(self):
        # [FUNCTION DOC] Checks the in-place and operator updates the base set to the expected
        # intersection with an overlapping set.
        self.OrderedSet &= OrderedSet([3, 4, 5])
        self.assertEqual(self.OrderedSet, OrderedSet([4]))

    def test_intersection_non_overlap(self):
        # [FUNCTION DOC] Checks the in-place and operator updates the base set to the expected
        # intersection with a disjoint set.
        self.OrderedSet &= OrderedSet([8])
        self.assertEqual(self.OrderedSet, empty_set)

    def test_intersection_method_call(self):
        # [FUNCTION DOC] Checks the intersection update method called with an overlapping set
        # updates the base set to the expected intersection.
        self.OrderedSet.intersection_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([4]))

    def test_sym_difference_subset(self):
        # [FUNCTION DOC] Checks the in-place caret operator updates the base set to the
        # expected symmetric difference with a subset.
        self.OrderedSet ^= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet([6]))

    def test_sym_difference_superset(self):
        # [FUNCTION DOC] Checks the in-place caret operator updates the base set to the
        # expected symmetric difference with a superset.
        self.OrderedSet ^= OrderedSet((2, 4, 6, 8))
        self.assertEqual(self.OrderedSet, OrderedSet([8]))

    def test_sym_difference_overlap(self):
        # [FUNCTION DOC] Checks the in-place caret operator updates the base set to the
        # expected symmetric difference with an overlapping set.
        self.OrderedSet ^= OrderedSet((3, 4, 5))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 5, 6]))

    def test_sym_difference_non_overlap(self):
        # [FUNCTION DOC] Checks the in-place caret operator updates the base set to the
        # expected symmetric difference with a disjoint set.
        self.OrderedSet ^= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6, 8]))

    def test_sym_difference_method_call(self):
        # [FUNCTION DOC] Checks the symmetric difference update method called with an
        # overlapping set updates the base set to the expected symmetric difference.
        self.OrderedSet.symmetric_difference_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 3, 5, 6]))

    def test_difference_subset(self):
        # [FUNCTION DOC] Checks the in-place minus operator updates the base set to the
        # expected difference with a subset.
        self.OrderedSet -= OrderedSet((2, 4))
        self.assertEqual(self.OrderedSet, OrderedSet([6]))

    def test_difference_superset(self):
        # [FUNCTION DOC] Checks the in-place minus operator updates the base set to the
        # expected difference with a superset.
        self.OrderedSet -= OrderedSet((2, 4, 6, 8))
        self.assertEqual(self.OrderedSet, OrderedSet([]))

    def test_difference_overlap(self):
        # [FUNCTION DOC] Checks the in-place minus operator updates the base set to the
        # expected difference with an overlapping set.
        self.OrderedSet -= OrderedSet((3, 4, 5))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 6]))

    def test_difference_non_overlap(self):
        # [FUNCTION DOC] Checks the in-place minus operator updates the base set to the
        # expected difference with a disjoint set.
        self.OrderedSet -= OrderedSet([8])
        self.assertEqual(self.OrderedSet, OrderedSet([2, 4, 6]))

    def test_difference_method_call(self):
        # [FUNCTION DOC] Checks the difference update method called with an overlapping set
        # updates the base set to the expected difference.
        self.OrderedSet.difference_update(OrderedSet([3, 4, 5]))
        self.assertEqual(self.OrderedSet, OrderedSet([2, 6]))


# ==============================================================================


class TestMutate(TestCase):
    # [CLASS DOC] Tests mutating methods (add, remove, discard, clear, pop, update) on a
    # three-element string set fixture.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, stores the values a, b, c and builds
        # a set from them.
        super().setUp()
        self.values = ["a", "b", "c"]
        self.OrderedSet = OrderedSet(self.values)

    def test_add_present(self):
        # [FUNCTION DOC] Checks adding an existing element leaves the set unchanged.
        self.OrderedSet.add("c")
        self.assertEqual(self.OrderedSet, OrderedSet("abc"))

    def test_add_absent(self):
        # [FUNCTION DOC] Checks adding a new element appends it to the set.
        self.OrderedSet.add("d")
        self.assertEqual(self.OrderedSet, OrderedSet("abcd"))

    def test_add_until_full(self):
        # [FUNCTION DOC] Adds each value to an empty set checking the length grows by one each
        # time, then that the result equals the fixture.
        tmp = OrderedSet()
        expected_len = 0
        for v in self.values:
            tmp.add(v)
            expected_len += 1  # noqa: SIM113
            self.assertEqual(len(tmp), expected_len)
        self.assertEqual(tmp, self.OrderedSet)

    def test_remove_present(self):
        # [FUNCTION DOC] Checks removing a present element leaves the other elements.
        self.OrderedSet.remove("b")
        self.assertEqual(self.OrderedSet, OrderedSet("ac"))

    def test_remove_absent(self):
        # [FUNCTION DOC] Checks removing a missing element raises a lookup error; fails if
        # nothing is raised.
        try:
            self.OrderedSet.remove("d")
            self.fail("Removing missing element should have raised LookupError")
        except LookupError:
            pass

    def test_remove_until_empty(self):
        # [FUNCTION DOC] Removes every value checking the length shrinks by one each time.
        expected_len = len(self.OrderedSet)
        for v in self.values:
            self.OrderedSet.remove(v)
            expected_len -= 1
            self.assertEqual(len(self.OrderedSet), expected_len)

    def test_discard_present(self):
        # [FUNCTION DOC] Checks discarding a present element removes it.
        self.OrderedSet.discard("c")
        self.assertEqual(self.OrderedSet, OrderedSet("ab"))

    def test_discard_absent(self):
        # [FUNCTION DOC] Checks discarding a missing element leaves the set unchanged.
        self.OrderedSet.discard("d")
        self.assertEqual(self.OrderedSet, OrderedSet("abc"))

    def test_clear(self):
        # [FUNCTION DOC] Checks clear leaves a set of length zero.
        self.OrderedSet.clear()
        self.assertEqual(len(self.OrderedSet), 0)

    def test_pop(self):
        # [FUNCTION DOC] Pops until empty, collecting results in a dict, and checks all
        # original values were popped exactly once.
        popped = {}
        while self.OrderedSet:
            popped[self.OrderedSet.pop()] = None
        self.assertEqual(len(popped), len(self.values))
        for v in self.values:
            self.assertIn(v, popped)

    def test_update_empty_tuple(self):
        # [FUNCTION DOC] Checks updating with an empty tuple leaves the set unchanged.
        self.OrderedSet.update(())
        self.assertEqual(self.OrderedSet, OrderedSet(self.values))

    def test_update_unit_tuple_overlap(self):
        # [FUNCTION DOC] Checks updating with a tuple of an existing element leaves the set
        # unchanged.
        self.OrderedSet.update(("a",))
        self.assertEqual(self.OrderedSet, OrderedSet(self.values))

    def test_update_unit_tuple_non_overlap(self):
        # [FUNCTION DOC] Checks updating with a tuple of an existing and a new element adds
        # only the new one.
        self.OrderedSet.update(("a", "z"))
        self.assertEqual(self.OrderedSet, OrderedSet(self.values + ["z"]))


# ==============================================================================


class TestSubsets(TestCase):
    # [CLASS DOC] Base mixin for ordering tests; subclasses set the left and right operand
    # sets, a name and the tuple of comparison operators expected to be true. Class attributes
    # map the operators to method names and to their reversed forms.
    case2method = {
        "<=": "issubset",
        ">=": "issuperset",
    }

    reverse = {
        "==": "==",
        "!=": "!=",
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
    }

    def test_issubset(self):
        # [FUNCTION DOC] Skipped when run on the base class itself. For each of the six
        # comparison operators, evaluates the comparison dynamically on left and right (and
        # with operands reversed using the reversed operator), checking it is true exactly for
        # the operators in the expected tuple; for the subset and superset operators also
        # checks the named method gives the same answer.
        if type(self) is TestSubsets:
            raise unittest.SkipTest("Only meant to be run as subclass")
        x = self.left
        y = self.right
        for case in "!=", "==", "<", "<=", ">", ">=":
            expected = case in self.cases
            # Test the binary infix spelling.
            result = eval("x" + case + "y", locals())
            self.assertEqual(result, expected)
            # Test the "friendly" method-name spelling, if one exists.
            if case in TestSubsets.case2method:
                method = getattr(x, TestSubsets.case2method[case])
                result = method(y)
                self.assertEqual(result, expected)

            # Now do the same for the operands reversed.
            rcase = TestSubsets.reverse[case]
            result = eval("y" + rcase + "x", locals())
            self.assertEqual(result, expected)
            if rcase in TestSubsets.case2method:
                method = getattr(y, TestSubsets.case2method[rcase])
                result = method(x)
                self.assertEqual(result, expected)


# ------------------------------------------------------------------------------


class TestSubsetEqualEmpty(TestSubsets, TestCase):
    # [CLASS DOC] Ordering cases with two empty sets: equal, subset and superset hold.
    left = OrderedSet()
    right = OrderedSet()
    name = "both empty"
    cases = "==", "<=", ">="


# ------------------------------------------------------------------------------


class TestSubsetEqualNonEmpty(TestSubsets, TestCase):
    # [CLASS DOC] Ordering cases with two equal two-element sets: equal, subset and superset
    # hold.
    left = OrderedSet([1, 2])
    right = OrderedSet([1, 2])
    name = "equal pair"
    cases = "==", "<=", ">="


# ------------------------------------------------------------------------------


class TestSubsetEmptyNonEmpty(TestSubsets, TestCase):
    # [CLASS DOC] Ordering cases with an empty set and a non-empty set: not-equal, proper
    # subset and subset hold.
    left = OrderedSet()
    right = OrderedSet([1, 2])
    name = "one empty, one non-empty"
    cases = "!=", "<", "<="


# ------------------------------------------------------------------------------


class TestSubsetPartial(TestSubsets, TestCase):
    # [CLASS DOC] Ordering cases where left is a proper non-empty subset of right.
    left = OrderedSet([1])
    right = OrderedSet([1, 2])
    name = "one a non-empty proper subset of other"
    cases = "!=", "<", "<="


# ------------------------------------------------------------------------------


class TestSubsetNonOverlap(TestSubsets, TestCase):
    # [CLASS DOC] Ordering cases with two disjoint non-empty sets: only not-equal holds.
    left = OrderedSet([1])
    right = OrderedSet([2])
    name = "neither empty, neither contains"
    cases = "!="


# ==============================================================================


class TestOnlySetsInBinaryOps(TestCase):
    # [CLASS DOC] Base mixin checking that set operators and methods only accept sets as
    # operands. Subclasses provide an other operand of a non-set kind and a set fixture, in
    # setUp.
    def test_eq_ne(self):
        # [FUNCTION DOC] Checks equality and inequality with a non-set operand are allowed and
        # give not-equal results.
        # Unlike the others, this is testing that == and != *are* allowed.
        self.assertEqual(self.other == self.OrderedSet, False)
        self.assertEqual(self.OrderedSet == self.other, False)
        self.assertEqual(self.other != self.OrderedSet, True)
        self.assertEqual(self.OrderedSet != self.other, True)

    def test_ge_gt_le_lt(self):
        # [FUNCTION DOC] Placeholder that does nothing; the ordering-operator type error
        # checks are commented out.
        pass
        # self.assertRaises(TypeError, lambda: self.OrderedSet < self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet <= self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet > self.other)
        # self.assertRaises(TypeError, lambda: self.OrderedSet >= self.other)

        # self.assertRaises(TypeError, lambda: self.other < self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other <= self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other > self.OrderedSet)
        # self.assertRaises(TypeError, lambda: self.other >= self.OrderedSet)

    def test_update_operator(self):
        # [FUNCTION DOC] Applies the in-place or operator with the non-set operand; a type
        # error is tolerated but not required (the strict check is commented out).
        try:
            self.OrderedSet |= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_update(self):
        # [FUNCTION DOC] Calls the update method with the other operand: succeeds when the
        # operand is iterable, otherwise asserts a type error is raised.
        if self.otherIsIterable:
            self.OrderedSet.update(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.update, self.other)

    def test_union(self):
        # [FUNCTION DOC] Calls the union method with the other operand: succeeds when the
        # operand is iterable, otherwise asserts a type error is raised. The operator-form
        # checks are commented out.
        # self.assertRaises(TypeError, lambda: self.OrderedSet | self.other)
        # self.assertRaises(TypeError, lambda: self.other | self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.union(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.union, self.other)

    def test_intersection_update_operator(self):
        # [FUNCTION DOC] Applies the in-place and operator with the non-set operand; a type
        # error is tolerated but not required (the strict check is commented out).
        try:
            self.OrderedSet &= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_intersection_update(self):
        # [FUNCTION DOC] Calls the intersection update method with the other operand: succeeds
        # when the operand is iterable, otherwise asserts a type error is raised.
        if self.otherIsIterable:
            self.OrderedSet.intersection_update(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.intersection_update, self.other
            )

    def test_intersection(self):
        # [FUNCTION DOC] Calls the intersection method with the other operand: succeeds when
        # the operand is iterable, otherwise asserts a type error is raised. The operator-form
        # checks are commented out.
        # self.assertRaises(TypeError, lambda: self.OrderedSet & self.other)
        # self.assertRaises(TypeError, lambda: self.other & self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.intersection(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.intersection, self.other)

    def test_sym_difference_update_operator(self):
        # [FUNCTION DOC] Applies the in-place caret operator with the non-set operand; a type
        # error is tolerated but not required (the strict check is commented out).
        try:
            self.OrderedSet ^= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_sym_difference_update(self):
        # [FUNCTION DOC] Calls the symmetric difference update method with the other operand:
        # succeeds when the operand is iterable, otherwise asserts a type error is raised.
        if self.otherIsIterable:
            self.OrderedSet.symmetric_difference_update(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.symmetric_difference_update, self.other
            )

    def test_sym_difference(self):
        # [FUNCTION DOC] Calls the symmetric difference method with the other operand:
        # succeeds when the operand is iterable, otherwise asserts a type error is raised. The
        # operator-form checks are commented out.
        # self.assertRaises(TypeError, lambda: self.OrderedSet ^ self.other)
        # self.assertRaises(TypeError, lambda: self.other ^ self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.symmetric_difference(self.other)
        else:
            self.assertRaises(
                TypeError, self.OrderedSet.symmetric_difference, self.other
            )

    def test_difference_update_operator(self):
        # [FUNCTION DOC] Applies the in-place minus operator with the non-set operand; a type
        # error is tolerated but not required (the strict check is commented out).
        try:
            self.OrderedSet -= self.other
        except TypeError:
            pass
        # else:
        #     self.fail("expected TypeError")

    def test_difference_update(self):
        # [FUNCTION DOC] Calls the difference update method with the other operand: succeeds
        # when the operand is iterable, otherwise asserts a type error is raised.
        if self.otherIsIterable:
            self.OrderedSet.difference_update(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.difference_update, self.other)

    def test_difference(self):
        # [FUNCTION DOC] Calls the difference method with the other operand: succeeds when the
        # operand is iterable, otherwise asserts a type error is raised. The operator-form
        # checks are commented out.
        # self.assertRaises(TypeError, lambda: self.OrderedSet - self.other)
        # self.assertRaises(TypeError, lambda: self.other - self.OrderedSet)
        if self.otherIsIterable:
            self.OrderedSet.difference(self.other)
        else:
            self.assertRaises(TypeError, self.OrderedSet.difference, self.other)


# ------------------------------------------------------------------------------


class TestOnlySetsNumeric(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is an integer (not
    # iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to an integer and records that it is not iterable.
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = 19
        self.otherIsIterable = False


# ------------------------------------------------------------------------------


class TestOnlySetsDict(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is a dict (iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to a two-entry dict and records that it is iterable.
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = {1: 2, 3: 4}
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsOperator(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is a function object
    # (not iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to the add operator function and records that it is
        # not iterable.
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = operator.add
        self.otherIsIterable = False


# ------------------------------------------------------------------------------


class TestOnlySetsTuple(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is a tuple (iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to a tuple of three numbers and records that it is
        # iterable.
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = (2, 4, 6)
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsString(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is a string (iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to a three-letter string and records that it is
        # iterable.
        super().setUp()
        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = "abc"
        self.otherIsIterable = True


# ------------------------------------------------------------------------------


class TestOnlySetsGenerator(TestOnlySetsInBinaryOps, TestCase):
    # [CLASS DOC] Only-sets operator tests with a non-set operand that is a generator
    # (iterable).
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup, builds the set of one, two and
        # three, sets the other operand to a generator of even numbers below ten and records
        # that it is iterable.
        super().setUp()

        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding the even numbers from zero to eight.
            for i in range(0, 10, 2):  # noqa: UP028
                yield i

        self.OrderedSet = OrderedSet((1, 2, 3))
        self.other = gen()
        self.otherIsIterable = True


del TestOnlySetsInBinaryOps
# ==============================================================================


class TestCopying:
    # [CLASS DOC] Mixin of copy tests; subclasses provide the set fixture. It is not a test
    # case itself and is removed from the module after the subclasses are defined.
    def test_copy(self):
        # [FUNCTION DOC] Checks a shallow copy has the same length and holds the identical
        # element objects, comparing the elements after sorting both by textual form.
        dup = self.OrderedSet.copy()
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.OrderedSet, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertTrue(dup_list[i] is set_list[i])

    def test_deep_copy(self):
        # [FUNCTION DOC] Checks a deep copy has the same length and equal elements, comparing
        # after sorting both by textual form.
        dup = copy.deepcopy(self.OrderedSet)
        # print type(dup), repr(dup)
        dup_list = sorted(dup, key=repr)
        set_list = sorted(self.OrderedSet, key=repr)
        self.assertEqual(len(dup_list), len(set_list))
        for i in range(len(dup_list)):
            self.assertEqual(dup_list[i], set_list[i])


# ------------------------------------------------------------------------------


class TestCopyingEmpty(TestCopying, TestCase):
    # [CLASS DOC] Copy tests using an empty set.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds an empty set.
        super().setUp()
        self.OrderedSet = OrderedSet()


# ------------------------------------------------------------------------------


class TestCopyingSingleton(TestCopying, TestCase):
    # [CLASS DOC] Copy tests using a one-element string set.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds a one-element string set.
        super().setUp()
        self.OrderedSet = OrderedSet(["hello"])


# ------------------------------------------------------------------------------


class TestCopyingTriple(TestCopying, TestCase):
    # [CLASS DOC] Copy tests using a set of a string, a number and none.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds a set of a string, a
        # number and none.
        super().setUp()
        self.OrderedSet = OrderedSet(["zero", 0, None])


# ------------------------------------------------------------------------------


class TestCopyingTuple(TestCopying, TestCase):
    # [CLASS DOC] Copy tests using a set holding one tuple of numbers.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds a set holding one tuple of
        # numbers.
        super().setUp()
        self.OrderedSet = OrderedSet([(1, 2)])


# ------------------------------------------------------------------------------


class TestCopyingNested(TestCopying, TestCase):
    # [CLASS DOC] Copy tests using a set holding a nested tuple of tuples.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds a set holding a nested
        # tuple of tuples.
        super().setUp()
        self.OrderedSet = OrderedSet([((1, 2), (3, 4))])


del TestCopying

# ==============================================================================


class TestIdentities(TestCase):
    # [CLASS DOC] Tests algebraic identities between set operations on two sets built from
    # letters of two words. setUp builds both.
    def setUp(self):
        # [FUNCTION DOC] Fixture: calls the parent setup and builds the two sets from the
        # letters of two words.
        super().setUp()
        self.a = OrderedSet("abracadabra")
        self.b = OrderedSet("alacazam")

    def test_binopsVsSubsets(self):
        # [FUNCTION DOC] Checks difference, intersection and symmetric difference are proper
        # subsets of their operands and the union a proper superset.
        a, b = self.a, self.b
        self.assertTrue(a - b < a)
        self.assertTrue(b - a < b)
        self.assertTrue(a & b < a)
        self.assertTrue(a & b < b)
        self.assertTrue(a | b > a)
        self.assertTrue(a | b > b)
        self.assertTrue(a ^ b < a | b)

    def test_commutativity(self):
        # [FUNCTION DOC] Checks intersection, union and symmetric difference are commutative
        # and difference is not when the sets differ.
        a, b = self.a, self.b
        self.assertEqual(a & b, b & a)
        self.assertEqual(a | b, b | a)
        self.assertEqual(a ^ b, b ^ a)
        if a != b:
            self.assertNotEqual(a - b, b - a)

    def test_summations(self):
        # [FUNCTION DOC] Checks several identities that decompose the union into disjoint
        # parts and recombine them.
        # check that sums of parts equal the whole
        a, b = self.a, self.b
        self.assertEqual((a - b) | (a & b) | (b - a), a | b)
        self.assertEqual((a & b) | (a ^ b), a | b)
        self.assertEqual(a | (b - a), a | b)
        self.assertEqual((a - b) | b, a | b)
        self.assertEqual((a - b) | (a & b), a)
        self.assertEqual((b - a) | (a & b), b)
        self.assertEqual((a - b) | (b - a), a ^ b)

    def test_exclusion(self):
        # [FUNCTION DOC] Checks that the difference and intersection pieces do not overlap,
        # each combination giving an empty set.
        # check that inverse operations show non-overlap
        a, b, zero = self.a, self.b, OrderedSet()
        self.assertEqual((a - b) & b, zero)
        self.assertEqual((b - a) & a, zero)
        self.assertEqual((a & b) & (a ^ b), zero)


# Tests derived from test_itertools.py =======================================


def R(seqn):
    # [FUNCTION DOC] HELPER: Generator that yields each item of the input sequence in order.
    "Regular generator"
    for i in seqn:  # noqa: UP028
        yield i


class G:
    # [CLASS DOC] HELPER: Iterable that supports only item access by index, not the iteration
    # protocol; wraps a given sequence.
    "Sequence using __getitem__"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the wrapped sequence.
        self.seqn = seqn

    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Returns the item at the given index of the wrapped sequence;
        # an index error ends iteration.
        return self.seqn[i]


class I:  # noqa: E742
    # [CLASS DOC] HELPER: Iterable object implementing the iterator protocol itself, with an
    # index counter, over a given sequence.
    "Sequence using iterator protocol"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self

    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item and advances the position; raises stop
        # iteration after the last.
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class Ig:
    # [CLASS DOC] HELPER: Iterable whose iteration method is a generator yielding each item of
    # the wrapped sequence.
    "Sequence using iterator protocol defined with a generator"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and a position counter set to zero (the
        # counter is unused).
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding each item of the wrapped sequence in
        # order.
        for val in self.seqn:  # noqa: UP028
            yield val


class X:
    # [CLASS DOC] HELPER: Object with a next method but neither indexing nor an iteration
    # method, so it is not iterable; used to check type errors.
    "Missing __getitem__ and __iter__"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0

    def __next__(self):
        # [FUNCTION DOC] HELPER: Returns the next item and advances the position; raises stop
        # iteration after the last.
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class N:
    # [CLASS DOC] HELPER: Object whose iteration method returns itself but which lacks a next
    # method, so iterating it fails with a type error.
    "Iterator missing __next__()"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self


class E:
    # [CLASS DOC] HELPER: Iterator whose next method always fails with a division by zero,
    # used to check exception propagation.
    "Test propagation of exceptions"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Stores the sequence and sets the position to zero.
        self.seqn = seqn
        self.i = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self

    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises a zero division error.
        3 // 0


class S:
    # [CLASS DOC] HELPER: Iterator over nothing: its next method immediately stops.
    "Test immediate stop"

    def __init__(self, seqn):
        # [FUNCTION DOC] HELPER: Accepts the sequence and ignores it.
        pass

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns itself as the iterator.
        return self

    def __next__(self):
        # [FUNCTION DOC] HELPER: Always raises stop iteration, yielding no items.
        raise StopIteration


from itertools import chain


def L(seqn):
    # [FUNCTION DOC] HELPER: Builds a multi-level iterator chain: wraps the sequence in the
    # index-only, generator-based and regular generator helpers, then an identity map and a
    # chain; returns that iterator.
    "Test multiple tiers of iterators"
    return chain(map(lambda x: x, R(Ig(G(seqn)))))  # noqa: C417


class TestVariousIteratorArgs(TestCase):
    # [CLASS DOC] Tests that constructors and set methods accept many kinds of iterable
    # arguments and propagate or reject bad ones: index-only sequences, iterator objects,
    # generators, empty iterators, nested iterator tiers, and broken objects.
    def test_constructor(self):
        # [FUNCTION DOC] For OrderedSet and frozenset and a range of sample inputs, checks
        # construction from each well-behaved iterable helper equals the source items
        # (compared sorted by textual form), a non-iterable raises a type error and a failing
        # iterator raises a zero division error.
        for cons in (OrderedSet, frozenset):
            for s in ("123", "", range(1000), ("do", 1.2), range(2000, 2200, 5)):
                for g in (G, I, Ig, S, L, R):
                    self.assertEqual(
                        sorted(cons(g(s)), key=repr), sorted(g(s), key=repr)
                    )
                self.assertRaises(TypeError, cons, X(s))
                self.assertRaises(TypeError, cons, N(s))
                self.assertRaises(ZeroDivisionError, cons, E(s))

    def test_inline_methods(self):
        # [FUNCTION DOC] For sample inputs and the methods union, intersection, difference,
        # symmetric difference and disjointness, checks results with iterator helpers
        # (skipping the index-only one) match results with the plain data; non-iterable
        # helpers raise type errors and the failing one raises zero division.
        s = OrderedSet("november")
        for data in (
            "123",
            "",
            range(1000),
            ("do", 1.2),
            range(2000, 2200, 5),
            "december",
        ):
            for meth in (
                s.union,
                s.intersection,
                s.difference,
                s.symmetric_difference,
                s.isdisjoint,
            ):
                for g in (G, I, Ig, L, R):
                    # Only iterables supported, not sequences
                    if g is G:
                        continue
                    expected = meth(data)
                    actual = meth(g(data))
                    if isinstance(expected, bool):
                        self.assertEqual(actual, expected)
                    else:
                        self.assertEqual(
                            sorted(actual, key=repr), sorted(expected, key=repr)
                        )
                self.assertRaises(TypeError, meth, X(s))
                self.assertRaises(TypeError, meth, N(s))
                self.assertRaises(ZeroDivisionError, meth, E(s))

    def test_inplace_methods(self):
        # [FUNCTION DOC] For sample inputs and the four in-place update methods, checks that
        # passing an iterator helper (skipping the index-only one) gives the same set as
        # passing the materialized list, and that bad helpers raise type or zero division
        # errors.
        for data in (
            "123",
            "",
            range(1000),
            ("do", 1.2),
            range(2000, 2200, 5),
            "december",
        ):
            for methname in (
                "update",
                "intersection_update",
                "difference_update",
                "symmetric_difference_update",
            ):
                for g in (G, I, Ig, S, L, R):
                    # Only Iterables supported, not Sequence
                    if g is G:
                        continue

                    s = OrderedSet("january")
                    t = s.copy()
                    getattr(s, methname)(list(g(data)))
                    getattr(t, methname)(g(data))
                    self.assertEqual(sorted(s, key=repr), sorted(t, key=repr))

                self.assertRaises(
                    TypeError, getattr(OrderedSet("january"), methname), X(data)
                )
                self.assertRaises(
                    TypeError, getattr(OrderedSet("january"), methname), N(data)
                )
                self.assertRaises(
                    ZeroDivisionError, getattr(OrderedSet("january"), methname), E(data)
                )


class bad_eq:
    # [CLASS DOC] HELPER: Element whose equality, when the module global bad flag is on,
    # clears the module global set named set2 and raises a zero division error; otherwise it
    # is identity comparison. Hash is constant zero.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Uses module globals set by the test: if the bad flag is set,
        # clears the global set and raises a zero division error; otherwise returns whether
        # both are the same object.
        if be_bad:
            set2.clear()
            raise ZeroDivisionError
        return self is other

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Returns the constant hash zero.
        return 0


class bad_dict_clear:
    # [CLASS DOC] HELPER: Element whose equality, when the module global bad flag is on,
    # clears the module global dict named dict2; otherwise it is identity comparison. Hash is
    # constant zero.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Uses module globals set by the test: if the bad flag is set,
        # clears the global dict; returns whether both are the same object.
        if be_bad:
            dict2.clear()
        return self is other

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Returns the constant hash zero.
        return 0


class TestWeirdBugs(TestCase):
    # [CLASS DOC] Regression tests for crashes when a set is mutated during an operation on
    # it.
    def test_8420_set_merge(self):
        # [FUNCTION DOC] Regression for bug 8420 (used to crash). Sets module globals and,
        # using built-in sets and a dict, checks that an update whose comparison clears the
        # other set raises a zero division error and that a symmetric difference update with a
        # comparison that clears the dict completes. Mutates module-level globals.
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
        # [FUNCTION DOC] Regression for issue 24581: builds an OrderedSet, creates an
        # iterator, clears and refills the set, then consumes the iterator; passes if nothing
        # crashes.
        # Issue #24581
        s = OrderedSet(range(100))
        s.clear()
        s.update(range(100))
        si = iter(s)
        s.clear()
        a = list(range(100))
        s.update(range(100))
        list(si)

    def test_merge_and_mutate(self):
        # [FUNCTION DOC] Regression: updates a built-in set from a set of elements whose
        # equality clears that source set during comparison; passes if nothing crashes. The
        # initial OrderedSet assignment to the variable is immediately overwritten.
        class X:
        # [CLASS DOC] HELPER: Element with hash of zero whose equality clears the enclosing
        # test's other collection and returns false.
            def __hash__(self):
            # [FUNCTION DOC] HELPER: Returns the hash of zero.
                return hash(0)

            def __eq__(self, o):
            # [FUNCTION DOC] HELPER: Clears the enclosing test's collection named other as a
            # side effect and returns false.
                other.clear()
                return False

        other = OrderedSet()
        other = {X() for i in range(10)}
        s = {0}
        s.update(other)


# Application tests (based on David Eppstein's graph recipes ====================================


def powerset(U):
    # [FUNCTION DOC] HELPER: Generator yielding every subset of the input sequence as a
    # frozenset, built recursively (the empty set is the base case).
    """Generates all subsets of a OrderedSet or sequence U."""
    U = iter(U)
    try:
        x = frozenset([next(U)])
        for S in powerset(U):
            yield S
            yield S | x
    except StopIteration:
        yield frozenset()


def cube(n):
    # [FUNCTION DOC] HELPER: Builds the hypercube graph of the given dimension as a dict
    # mapping each subset of the first n integers (as a frozenset) to the frozenset of its
    # neighbours, those differing by one element.
    """Graph of n-dimensional hypercube."""
    singletons = [frozenset([x]) for x in range(n)]
    return dict(  # noqa: C404
        [(x, frozenset([x ^ s for s in singletons])) for x in powerset(range(n))]
    )


def linegraph(G):
    # [FUNCTION DOC] HELPER: Builds the line graph of a graph given as a dict of adjacency
    # sets: vertices are the edges of the input and each maps to the frozenset of adjacent
    # edges sharing an endpoint.
    """Graph, the vertices of which are edges of G,
    with two vertices being adjacent iff the corresponding
    edges share a vertex."""
    L = {}
    for x in G:
        for y in G[x]:
            nx = [frozenset([x, z]) for z in G[x] if z != y]
            ny = [frozenset([y, z]) for z in G[y] if z != x]
            L[frozenset([x, y])] = frozenset(nx + ny)
    return L


def faces(G):
    # [FUNCTION DOC] HELPER: Finds faces of a graph given as adjacency dict: collects as
    # frozensets of vertices every closed walk of length three, four or five found by nested
    # search from each vertex. Returns an OrderedSet of them.
    "Return a OrderedSet of faces in G.  Where a face is a OrderedSet of vertices on that face"
    # currently limited to triangles,squares, and pentagons
    f = OrderedSet()
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
                                if v5 == v3 or v5 == v2:  # noqa: SIM109
                                    continue
                                if v1 in G[v5]:
                                    f.add(frozenset([v1, v2, v3, v4, v5]))
    return f


class TestGraphs(TestCase):
    # [CLASS DOC] Application tests that build polyhedra graphs and count their vertices and
    # faces with sets.
    def test_cube(self):
        # [FUNCTION DOC] Builds the three-dimensional cube graph and checks eight vertices
        # each with three neighbours, matching vertex sets, and six faces of four vertices
        # each.
        g = cube(3)  # vert --> {v1, v2, v3}
        vertices1 = OrderedSet(g)
        self.assertEqual(len(vertices1), 8)  # eight vertices
        for edge in g.values():
            self.assertEqual(len(edge), 3)  # each vertex connects to three edges
        vertices2 = OrderedSet(v for edges in g.values() for v in edges)
        self.assertEqual(vertices1, vertices2)  # edge vertices in original OrderedSet

        cubefaces = faces(g)
        self.assertEqual(len(cubefaces), 6)  # six faces
        for face in cubefaces:
            self.assertEqual(len(face), 4)  # each face is a square

    def test_cuboctahedron(self):
        # [FUNCTION DOC] Builds the line graph of the cube (cuboctahedron) and checks twelve
        # vertices each with four neighbours, matching vertex sets, eight triangular and six
        # square faces, and that every vertex is an edge of two cube vertices.
        # http://en.wikipedia.org/wiki/Cuboctahedron
        # 8 triangular faces and 6 square faces
        # 12 identical vertices each connecting a triangle and square

        g = cube(3)
        cuboctahedron = linegraph(g)  # V( --> {V1, V2, V3, V4}
        self.assertEqual(len(cuboctahedron), 12)  # twelve vertices

        vertices = OrderedSet(cuboctahedron)
        for edges in cuboctahedron.values():
            self.assertEqual(
                len(edges), 4
            )  # each vertex connects to four other vertices
        othervertices = OrderedSet(
            edge for edges in cuboctahedron.values() for edge in edges
        )
        self.assertEqual(
            vertices, othervertices
        )  # edge vertices in original OrderedSet

        cubofaces = faces(cuboctahedron)
        facesizes = collections.defaultdict(int)
        for face in cubofaces:
            facesizes[len(face)] += 1
        self.assertEqual(facesizes[3], 8)  # eight triangular faces
        self.assertEqual(facesizes[4], 6)  # six square faces

        for vertex in cuboctahedron:
            edge = vertex  # Cuboctahedron vertices are edges in Cube
            self.assertEqual(len(edge), 2)  # Two cube vertices define an edge
            for cubevert in edge:
                self.assertIn(cubevert, g)


# ==============================================================================

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
