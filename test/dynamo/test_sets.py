# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for set, frozenset, user subclasses of both, and the OrderedSet
# utility class. Contains shared base test classes whose cases are written once and instantiated
# for each set type through a thetype class attribute; most cases use the make_dynamo_test
# decorator, which runs the body eagerly and then again under an assert-style compile. Also
# holds tests for guards on set arguments, recompilation when set contents change, hash handling
# (custom hash from object ids, no re-hashing of dict keys), set iterator length hints, and
# generator arguments. Enables the unittest tracing config during tests.

# TODO: move set tests from test_functions.py/test_misc.py to this file

import math
import unittest
from collections.abc import Iterable

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import make_dynamo_test, munge_exc
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class SetSubclass(set):
    # [CLASS DOC] HELPER: Plain subclass of set with no changes, used to test subclass handling.
    pass


class FrozenstSubclass(frozenset):
    # [CLASS DOC] HELPER: Plain subclass of frozenset with no changes, used to test subclass
    # handling.
    pass


class BadCmp:
    # [CLASS DOC] HELPER: Element type whose hash is a constant, so all instances collide and
    # equality is consulted on insert or lookup; its equality always raises a runtime error,
    # which must propagate.
    # Elements collide on hash (so __eq__ runs during set insertion/lookup),
    # and __eq__ raises so the error must propagate.  Mirrors CPython
    # test_set.py BadCmp.
    def __hash__(self):
        # [FUNCTION DOC] HELPER: Always returns the constant one.
        return 1

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Always raises a runtime error, regardless of the other operand.
        raise RuntimeError


class _BaseSetTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] HELPER: Common base for set test cases. Turns on the config option that lets
    # Dynamo trace unittest assertions for the duration of each test, and overrides the equality
    # assertions with boolean forms that are traceable and report both operands on failure.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Saves the current trace-unittest config value, enables it, then
        # runs the parent setup.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the saved trace-unittest config value and returns the
        # parent teardown result.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def assertEqual(self, a, b):
        # [FUNCTION DOC] HELPER: Asserts the two arguments compare equal using the equality
        # operator; the failure message includes both values. Overrides the standard equality
        # assertion so it can be traced.
        return self.assertTrue(a == b, lambda msg: f"{msg}\n{a} != {b}")

    def assertNotEqual(self, a, b):
        # [FUNCTION DOC] HELPER: Asserts the two arguments compare unequal using the inequality
        # operator; the failure message includes both values.
        return self.assertTrue(a != b, lambda msg: f"{msg}\n{a} == {b}")


class CustomSetTests(_BaseSetTests):
    # [CLASS DOC] Tests for a user-defined set subclass that overrides add, using the traced
    # dynamo test mechanism. Holds the subclass as the nested class and exposes it as the
    # thetype attribute.
    class CustomSet(set):
        # [CLASS DOC] HELPER: Set subclass whose add stores the item plus one, and which has an
        # extra method named contains that always answers true (separate from the membership
        # protocol).
        def add(self, item):
            # [FUNCTION DOC] HELPER: Adds item plus one to the set through the parent add and
            # returns the parent result.
            return super().add(item + 1)

        def contains(self, item):
            # [FUNCTION DOC] HELPER: Ignores its argument and always returns true.
            return True

    thetype = CustomSet

    @make_dynamo_test
    def test_custom_add(self):
        # [FUNCTION DOC] Dynamo test: builds the custom set from one and two, calls the
        # overridden add with three, and asserts the set holds one, two and four, proving the
        # override ran.
        s = self.thetype([1, 2])
        s.add(3)
        self.assertTrue(s == {1, 2, 4})

    @make_dynamo_test
    def test_custom_contains(self):
        # [FUNCTION DOC] Dynamo test: asserts the custom set's extra contains method returns
        # true for an element that is not in the set.
        s = self.thetype([1, 2])
        self.assertTrue(s.contains(3))


class MiscTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Miscellaneous standalone set tests: generator arguments, iterator length
    # hints, custom hashes derived from object ids, and hash reuse when building sets from
    # dicts. Each test compiles with the eager backend and full graph.
    def test_isdisjoint_with_generator(self):
        # [FUNCTION DOC] Checks that the set isdisjoint method pulls from a generator argument
        # only until a common element is found. A closed-over counter is updated by the
        # generator; the compiled function picks sine when the counter equals three, which means
        # only the first two items were consumed. Asserts the output equals sine.
        n = 0

        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding one, two and three; before each yield it
            # adds one, two or three to the enclosing counter respectively.
            nonlocal n
            n += 1
            yield 1
            n += 2
            yield 2
            n += 3
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function; builds a set of two, four and five,
            # calls isdisjoint with a fresh generator for its side effect on the counter, then
            # returns sine of x if the counter is three, else cosine.
            nonlocal n
            s = {2, 4, 5}
            s.isdisjoint(gen())
            if n == 3:
                return x.sin()
            return x.cos()

        x = torch.randn(1)
        y = fn(x)
        self.assertEqual(y, x.sin())

    def test_set_iterator_length_hint(self):
        # [FUNCTION DOC] Checks the length hint of iterators over a set, a dict and dict values:
        # before and after taking one element the hint reports the not-yet-consumed count, and
        # an iterator over an empty set reports zero. Compares the collected counts to the
        # expected sequence of counts and the sine output.
        # setiter_len/dictiter_len: __length_hint__ returns the number of
        # not-yet-consumed elements and decreases as the iterator advances.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function; for each of a three element set, a two
            # key dict and a one value dict view, creates an iterator and records its length
            # hint before and after one advance, then records the hint for an empty set
            # iterator. Returns sine of x and the recorded list.
            results = []
            for obj in ({1, 2, 3}, {1: "a", 2: "b"}, {1: "a"}.values()):
                it = iter(obj)
                results.append(it.__length_hint__())
                next(it)
                results.append(it.__length_hint__())
            results.append(iter(set()).__length_hint__())
            return x.sin(), results

        x = torch.randn(1)
        y, results = fn(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(results, [3, 2, 2, 1, 1, 0, 0])

    def test_id_arithmetic_in_custom_hash_fullgraph(self):
        # [FUNCTION DOC] Checks that bitwise and arithmetic operations on the object id of an
        # object created inside the compiled region do not graph break. Builds set, frozenset
        # and set-subclass instances whose hash masks the id, adds one to a set, tests
        # membership, and discards it, for each of three classes in a loop; asserts the output
        # is ones and membership was true.
        # id() of an object created inside the compiled region is a fake,
        # compile-time-only int. Bitwise/arithmetic ops on it (a common way to
        # derive a __hash__, e.g. CPython test_set.test_subclass_with_custom_hash)
        # must not graph break and must stay compile-time. Covers set,
        # frozenset, and subclass bases whose __hash__ masks id(self).
        class HSet(set):
            # [CLASS DOC] HELPER: Set subclass whose hash value is derived from its own id via a
            # bit mask and xor.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns an integer hash computed from masking the
                # object id and xoring with three.
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        class HFrozen(frozenset):
            # [CLASS DOC] HELPER: Frozenset subclass with a hash derived from its own id via a
            # bit mask and xor.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns an integer hash computed from masking the
                # object id and xoring with three.
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        class HSubclass(SetSubclass):
            # [CLASS DOC] HELPER: Subclass of the module's set subclass with a hash derived from
            # its own id via a bit mask and xor.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns an integer hash computed from masking the
                # object id and xoring with three.
                return int((id(self) & 0x7FFFFFFF) ^ 3)

        for cls in (HSet, HFrozen, HSubclass):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled function; creates an instance of the loop's
                # class, adds it to a plain set, records whether it is a member, discards it,
                # and returns x plus one with the membership flag. The class is taken from the
                # enclosing loop variable.
                s = cls()
                f = set()
                f.add(s)
                present = s in f
                f.discard(s)
                return x + 1, present

            res, present = fn(torch.zeros(1))
            self.assertEqual(res, torch.ones(1))
            self.assertTrue(present)

    def test_id_arithmetic_ops_stay_compile_time(self):
        # [FUNCTION DOC] Checks that a mix of integer arithmetic, bitwise and unary operations
        # on a fake object id remain compile-time only when used to compute a hash, so a set
        # holding such an object compiles without a graph break. Asserts output ones and a set
        # length of one.
        # A spread of int arithmetic/bitwise/unary ops on a fake id() value
        # all stay compile-time-only and can serve as a hash without breaking.
        class H(set):
            # [CLASS DOC] HELPER: Set subclass whose hash combines the object id with masks,
            # shifts, negation, multiplication, subtraction and integer division.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Computes an integer from several arithmetic and bitwise
                # operations on the object id, and returns it as an int.
                i = id(self)
                return int((i & 0xFFFF) | (i >> 4) ^ (~i) + (i * 2) - (i // 3))

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function; builds a set containing one instance of
            # the hashed class and returns x plus one with the set length.
            f = {H()}
            return x + 1, len(f)

        res, n = fn(torch.zeros(1))
        self.assertEqual(res, torch.ones(1))
        self.assertEqual(n, 1)

    def test_do_not_rehash_dict_keys(self):
        # [FUNCTION DOC] Checks that building set, frozenset and their subclasses from a dict
        # reuses stored hashes, so keys are hashed exactly once overall, and that eager and
        # compiled runs give the same hash-count history. Loops over four set types with ten
        # keys, compares compiled and eager results, and asserts the first recorded count list
        # has five entries all equal to ten.
        # Building a set/frozenset (or subclass) from a dict must reuse the
        # dict's stored hashes instead of re-invoking __hash__, mirroring
        # CPython's set_update_internal fast path.  Also covers the explicit
        # base-slot call `int.__hash__(self)` inside the custom __hash__.
        class HashCountingInt(int):
            # [CLASS DOC] HELPER: Integer subclass that counts how many times it is hashed; the
            # count lives in the hash_count attribute and its hash delegates to the integer base
            # class.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Initializes the hash counter to zero.
                self.hash_count = 0

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Increments the hash counter and returns the base
                # integer hash of the object.
                self.hash_count += 1
                return int.__hash__(self)

        def run(thetype, n):
            # [FUNCTION DOC] HELPER: Builds a dict whose keys are counting integers, then
            # repeatedly records the total hash count across keys: after construction, after
            # building the given set type from the dict and taking a set difference with the
            # dict, after building a dict from a set of it, and after building a dict from a
            # frozenset of it. Returns the list of totals and the final set's length.
            d = dict.fromkeys(map(HashCountingInt, range(n)))
            counts = [sum(e.hash_count for e in d)]
            s = thetype(d)
            counts.append(sum(e.hash_count for e in d))
            s.difference(d)
            counts.append(sum(e.hash_count for e in d))
            dict.fromkeys(set(d))
            counts.append(sum(e.hash_count for e in d))
            dict.fromkeys(frozenset(d))
            counts.append(sum(e.hash_count for e in d))
            return counts, len(s)

        for thetype in (set, frozenset, SetSubclass, FrozenstSubclass):
            n = 10
            ref = run(thetype, n)
            res = torch.compile(run, backend="eager", fullgraph=True)(thetype, n)
            self.assertEqual(ref, res)
            # Every key hashed exactly once (during the initial fromkeys).
            self.assertEqual(ref[0], [n] * 5)


class TestSetGuards(LoggingTestCase):
    # [CLASS DOC] Logging-enabled test case for guards on set inputs: membership checks
    # specialize and recompile when the set contents change, sets of tensors are rejected, and
    # guard failure is logged.
    def test_set_with_function(self):
        # [FUNCTION DOC] Passes a set containing builtin functions and a string; the compiled
        # function branches on membership of one function. After first run asserts sine with one
        # frame, then removes that function, adds another, and asserts cosine with two frames.
        s = {
            torch._C._set_grad_enabled,
            "hello",
            torch.amp._exit_autocast,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Compiled function; returns sine of x if the autocast exit
            # function is in the set, otherwise cosine.
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove(torch.amp._exit_autocast)
        s.add(torch._C._set_fwd_grad_enabled)
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    @make_logging_test(recompiles=True)
    def test_in_guard(self, records):
        # [FUNCTION DOC] Same shape as the function-set test but with string elements and with
        # recompile logging captured. After removing and adding elements the second call
        # recompiles. Asserts that at least one log record exists and that the record for the
        # set contains check mentions the contains call on the string key.
        s = {
            "Dynamo",
            "Inductor",
            "PyTorch",
            torch.sin,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Compiled function; returns sine of x if the string for
            # PyTorch is in the set, else cosine.
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.randn(2)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        s.add("Cuda")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "set.__contains__")
        self.assertIn(
            """set.__contains__(s, 'PyTorch')""",
            munge_exc(record.getMessage()),
        )

    def test_set_with_tensors(self):
        # [FUNCTION DOC] Passes a set of tensors to a compiled function that iterates it and
        # sums elements. Asserts Dynamo raises the Unsupported error whose message explains that
        # sets of tensors cannot be traced, with hints and a link to the graph break docs; the
        # message is normalized to hide line numbers.
        s = {
            torch.ones(1),
            torch.tensor([1.0]),
            torch.zeros(1),
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Compiled function; iterates the set adding each element to
            # a zero tensor and returns x plus that sum.
            z = torch.zeros(1)
            for i in s:
                z += i
            return x + z

        x = torch.tensor([1.0])
        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(x, s),
            """\
Attempted to wrap a set with tensors
  Explanation: Dynamo cannot trace sets of tensors. To get a stable ordering, Dynamo needs to convert the set into a list and the order might not be stable if the set contains tensors.
  Hint: Use a dictionary where the keys are tensors.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: Python set containing torch.Tensor elements

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0222.html

from user code:
   File "test_sets.py", line N, in fn
    for i in s:""",
        )

    def test_set_multiple_types(self):
        # [FUNCTION DOC] Uses a set mixing a string, float, complex number and nan. Checks sine
        # with one frame, then removes the string and checks cosine with two frames, i.e. the
        # string membership guard triggers a recompile.
        s = {
            "PyTorch",
            3.3,
            1j,
            math.nan,
        }
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Compiled function; returns sine of x if PyTorch is in the
            # set, else cosine.
            if "PyTorch" in s:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x, s)
        self.assertEqual(y, x.sin())
        self.assertEqual(cnts.frame_count, 1)

        s.remove("PyTorch")
        y = fn(x, s)
        self.assertEqual(y, x.cos())
        self.assertEqual(cnts.frame_count, 2)

    def test_set_recompile_on_key_pop(self):
        # [FUNCTION DOC] Compiles a membership-based function over a set of functions, runs
        # twice and asserts a single frame and eager-equal output; removes the tested element
        # and asserts a recompile (two frames) with correct output.
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Returns sine of x if the autocast exit function is in the
            # set, else cosine.
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        # [FUNCTION DOC] Same as the pop test but after removing the tested element a different
        # function is added, keeping the set size; asserts the guard still detects the change:
        # one frame before, two after, outputs matching eager.
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Returns sine of x if the autocast exit function is in the
            # set, else cosine.
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)
        # Add a different value
        s.add(torch._C._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    @unittest.skip("random failures on Python 3.9")
    def test_set_guard_on_keys_change(self):
        # [FUNCTION DOC] Skipped test (the skip reason mentions random failures on one old
        # Python version). Intended to show dict-style guards are not applied to sets: removing
        # and re-adding the same element must not trigger recompilation (frame count stays one)
        # and the result must match eager.
        # This test guarantee that we're not triggering any of the dict guards
        # on sets
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = CompileCounter()

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Multiplies x by the string length of each set element in
            # iteration order and returns it.
            for e in s:
                x = x * len(str(e))
            return x

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(torch.randn(4), s)
        opt_fn(torch.randn(4), s)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # pop and add the same item
        s.remove(torch.amp._exit_autocast)
        # It is not guaranteed that _exit_autocast will be in a specific order
        s.add(torch.amp._exit_autocast)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo don't recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, fn(x, s))


class _FrozensetBase:
    # [CLASS DOC] HELPER: Mixin of test cases for operations shared by frozenset and set: binary
    # operators, comparisons, copy, subset and superset checks, construction, membership,
    # disjointness, intersection, union, differences, and conversions. The including class
    # supplies a thetype attribute naming the type under test. Many cases mirror CPython
    # test_set tests and also check calling methods through the class with explicit self.
    # Frozenset methods
    # + copy
    # + difference
    # + intersection
    # + isdisjoint
    # + issubset
    # + issuperset
    # + symmetric_difference
    # + union
    # BinOps:
    # +, -, |, &, ^, <, >, <=, >=, ==, !=

    @make_dynamo_test
    def test_binop_sub(self):
        # [FUNCTION DOC] Dynamo test: the difference operator between two sets of letters gives
        # empty, the expected remainders, and raises a type error with an integer; the unbound
        # method form gives the expected set.
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p - p, self.thetype())
        self.assertEqual(p - q, self.thetype("ac"))
        self.assertEqual(q - p, self.thetype("ef"))
        self.assertRaises(TypeError, lambda: p - 1)
        self.assertEqual(self.thetype.__sub__(p, q), set("ac"))

    @make_dynamo_test
    def test_binop_or(self):
        # [FUNCTION DOC] Dynamo test: the union operator on sets of letters gives the expected
        # results, also through the unbound method form.
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p | p, self.thetype("abc"))
        self.assertEqual(p | q, self.thetype("abcef"))
        self.assertEqual(self.thetype.__or__(p, q), set("abcef"))

    @make_dynamo_test
    def test_binop_and(self):
        # [FUNCTION DOC] Dynamo test: the intersection operator on sets of letters gives the
        # expected results, also through the unbound method form.
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p & p, self.thetype("abc"))
        self.assertEqual(p & q, self.thetype("b"))
        self.assertEqual(self.thetype.__and__(p, q), set("b"))

    @make_dynamo_test
    def test_binop_xor(self):
        # [FUNCTION DOC] Dynamo test: the symmetric difference operator on sets of letters gives
        # the expected results, also through the unbound method form.
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p ^ p, self.thetype())
        self.assertEqual(p ^ q, self.thetype("acef"))
        self.assertEqual(self.thetype.__xor__(p, q), set("acef"))

    @make_dynamo_test
    def test_badcmp(self):
        # [FUNCTION DOC] Dynamo test: an error raised by the element equality during
        # construction, membership or, where the type has mutators, add, discard and remove,
        # must propagate as a runtime error. Mutator checks are skipped for types lacking an add
        # method.
        # A comparison error during insertion/lookup must propagate as the
        # user exception.  For frozenset types hasattr(s, "add") is False, so
        # the mutating block is skipped (regression: exact frozenset used to
        # report hasattr(set, "add")).  Mirrors CPython test_set.py test_badcmp.
        s = self.thetype([BadCmp()])
        self.assertRaises(RuntimeError, self.thetype, [BadCmp(), BadCmp()])
        self.assertRaises(RuntimeError, s.__contains__, BadCmp())
        if hasattr(s, "add"):
            self.assertRaises(RuntimeError, s.add, BadCmp())
            self.assertRaises(RuntimeError, s.discard, BadCmp())
            self.assertRaises(RuntimeError, s.remove, BadCmp())

    @make_dynamo_test
    def test_cmp_eq(self):
        # [FUNCTION DOC] Dynamo test: a set equals itself and equals sets, frozensets and set-
        # subclass instances built from the same letters, also through the unbound equality
        # method.
        p = self.thetype("abc")
        self.assertEqual(p, p)
        for C in set, frozenset, SetSubclass:
            self.assertEqual(p, C("abc"))
            self.assertEqual(p, C(p))
        self.assertTrue(self.thetype.__eq__(p, p))

    @make_dynamo_test
    def test_cmp_ne(self):
        # [FUNCTION DOC] Dynamo test: a set differs from another set, from other collection
        # types built from different letters, and from an integer, also through the unbound
        # inequality method.
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertNotEqual(p, q)
        self.assertNotEqual(q, p)
        for C in set, frozenset, SetSubclass, dict.fromkeys, str, list, tuple:
            self.assertNotEqual(p, C("abe"))
        self.assertNotEqual(p, 1)
        self.assertTrue(self.thetype.__ne__(p, q))

    @make_dynamo_test
    def test_cmp_less_than(self):
        # [FUNCTION DOC] Dynamo test: strict subset comparison is false for equal or disjoint
        # sets and true for a proper subset, also through the unbound method form.
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p < p)
        self.assertFalse(p < q)
        self.assertTrue(r < p)
        self.assertFalse(r < q)
        self.assertFalse(self.thetype.__lt__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than(self):
        # [FUNCTION DOC] Dynamo test: strict superset comparison, mirroring the subset case,
        # also through the unbound method form.
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p > p)
        self.assertFalse(p > q)
        self.assertTrue(p > r)
        self.assertFalse(q > r)
        self.assertFalse(self.thetype.__gt__(p, p))

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        # [FUNCTION DOC] Dynamo test: subset-or-equal comparison on equal, disjoint and proper
        # subset pairs, also through the unbound method form.
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p <= p)
        self.assertFalse(p <= q)
        self.assertTrue(r <= p)
        self.assertFalse(r <= q)
        self.assertTrue(self.thetype.__le__(p, p))

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        # [FUNCTION DOC] Dynamo test: superset-or-equal comparison on equal, disjoint and proper
        # superset pairs, also through the unbound method form.
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p >= p)
        self.assertFalse(p >= q)
        self.assertTrue(p >= r)
        self.assertFalse(q >= r)
        self.assertTrue(self.thetype.__ge__(p, p))

    @make_dynamo_test
    def test_copy(self):
        # [FUNCTION DOC] Dynamo test: the copy method gives an equal set, raises a type error
        # when given an argument, and the unbound method form returns an equal set.
        p = self.thetype("abc")
        q = p.copy()
        self.assertEqual(p, q)
        self.assertRaises(TypeError, p.copy, 1)
        self.assertEqual(self.thetype.copy(p), p)

    @make_dynamo_test
    def test_issubset(self):
        # [FUNCTION DOC] Dynamo test: issubset gives expected results for sets, raises a type
        # error with no argument, with an integer, and with an unhashable-element iterable, and
        # works through the unbound method form.
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(q.issubset(p))
        self.assertFalse(r.issubset(p))
        self.assertRaises(TypeError, p.issubset)
        self.assertRaises(TypeError, p.issubset, 1)
        self.assertRaises(TypeError, p.issubset, [[]])
        self.assertTrue(self.thetype.issubset(q, p))

    @make_dynamo_test
    def test_issuperset(self):
        # [FUNCTION DOC] Dynamo test: issuperset gives expected results, raises a type error
        # with no argument, with an integer, and with an unhashable-element iterable, and works
        # unbound.
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(p.issuperset(q))
        self.assertFalse(p.issuperset(r))
        self.assertRaises(TypeError, p.issuperset)
        self.assertRaises(TypeError, p.issuperset, 1)
        self.assertRaises(TypeError, p.issuperset, [[]])
        self.assertTrue(self.thetype.issuperset(p, q))

    @make_dynamo_test
    def test_constructor_iterable(self):
        # [FUNCTION DOC] Dynamo test: a set built from a string is an instance of the tested
        # type and of the iterable abstract class.
        p = self.thetype("abc")
        self.assertIsInstance(p, self.thetype)
        self.assertIsInstance(p, Iterable)

    @make_dynamo_test
    def test_new_or_init(self):
        # [FUNCTION DOC] Dynamo test: the plain set and frozenset constructors raise type errors
        # for a second positional argument and for any keyword argument, and calling the set
        # init method with a keyword also raises. Does not depend on the tested type.
        # set/frozenset constructors reject extra positional args and any
        # keyword arguments; set().__init__ rejects keywords even with 0 args.
        self.assertRaises(TypeError, set, [], 2)
        self.assertRaises(TypeError, frozenset, [], 2)
        self.assertRaises(TypeError, set, a=1)
        self.assertRaises(TypeError, frozenset, a=1)
        self.assertRaises(TypeError, set().__init__, a=1)

    @make_dynamo_test
    def test_equality(self):
        # [FUNCTION DOC] Dynamo test: a set equals copies of itself built from the tested type,
        # set and frozenset, via the operator, the bound equality method and the unbound method.
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertEqual(a, typ(a))
            self.assertTrue(a == typ(a))
            self.assertTrue(a.__eq__(typ(a)))
            self.assertTrue(self.thetype.__eq__(a, typ(a)))

    @make_dynamo_test
    def test_in_frozenset(self):
        # [FUNCTION DOC] Dynamo test: a set of letters is found as a member of a container of
        # the tested type holding the frozenset of the same letters.
        item = self.thetype("abc")
        container = self.thetype([frozenset("abc")])
        self.assertIn(item, container)

    @make_dynamo_test
    def test_contains(self):
        # [FUNCTION DOC] Dynamo test: membership operators and the contains method, bound and
        # unbound, report present and absent letters correctly.
        s = self.thetype(["a", "b", "c"])
        self.assertIn("a", s)
        self.assertNotIn("d", s)
        self.assertTrue(s.__contains__("a"))
        self.assertTrue(self.thetype.__contains__(s, "b"))

    @make_dynamo_test
    def test_isdisjoint(self):
        # [FUNCTION DOC] Dynamo test: isdisjoint returns false for overlapping and true for non-
        # overlapping sets, raises type errors for missing, integer and unhashable-element
        # arguments, and works unbound.
        x = self.thetype({"apple", "banana", "cherry"})
        y = self.thetype({"google", "microsoft", "apple"})
        z = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))
        self.assertRaises(TypeError, x.isdisjoint)
        self.assertRaises(TypeError, x.isdisjoint, 1)
        self.assertRaises(TypeError, x.isdisjoint, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertFalse(self.thetype.isdisjoint(p, q))

    @make_dynamo_test
    def test_intersection(self):
        # [FUNCTION DOC] Dynamo test: intersection of three sets gives the single common
        # element, raises type errors for an integer and for unhashable-element arguments, and
        # works unbound.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        intersection_set = set1.intersection(set2, set3)
        self.assertEqual(intersection_set, {"apple"})
        self.assertRaises(TypeError, set1.intersection, 1)
        self.assertRaises(TypeError, set1.intersection, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.intersection(p, q), {"b"})

    @make_dynamo_test
    def test_union(self):
        # [FUNCTION DOC] Dynamo test: union of three sets gives all elements, raises type errors
        # for bad arguments, and the unbound form gives the union of two others.
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        union_set = p.union(q, r)
        self.assertEqual(union_set, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.union, 1)
        self.assertRaises(TypeError, p.union, [[]])
        s = self.thetype.union(q, r)
        self.assertEqual(s, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_difference(self):
        # [FUNCTION DOC] Dynamo test: difference with two sets removes elements of both, raises
        # type errors for bad arguments, and the unbound form gives the expected difference.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        difference_set = set1.difference(set2, set3)
        self.assertEqual(difference_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference, 1)
        self.assertRaises(TypeError, set1.difference, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(self.thetype.difference(p, q), {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference(self):
        # [FUNCTION DOC] Dynamo test for symmetric difference. KNOWN ISSUE: the first check
        # calls the plain difference method rather than the symmetric difference method (and
        # names the result as symmetric), so that part does not test the intended operation; the
        # later checks use the correct method.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        symmetric_diff_set = set1.difference(set2)
        self.assertEqual(symmetric_diff_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.symmetric_difference)
        self.assertRaises(TypeError, set1.symmetric_difference, 1)
        self.assertRaises(TypeError, set1.symmetric_difference, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        symmetric_diff_set = self.thetype.symmetric_difference(p, q)
        self.assertEqual(symmetric_diff_set, {"a", "c", "e", "f"})

    @make_dynamo_test
    def test_to_frozenset(self):
        # [FUNCTION DOC] Dynamo test: converting a set of three strings to a frozenset yields a
        # frozenset of length three. KNOWN ISSUE: the name says conversion from frozenset but
        # behavior only checks the same conversion as the next test.
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)

    @make_dynamo_test
    def test_to_set(self):
        # [FUNCTION DOC] Dynamo test named for conversion to a set. KNOWN ISSUE: the body is
        # identical to the to-frozenset test and still builds a frozenset, so it never converts
        # to a set.
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)


class _SetBase(_FrozensetBase):
    # [CLASS DOC] HELPER: Mixin extending the shared cases with mutating set methods (add,
    # clear, remove, intersection update, difference update, symmetric difference update, pop,
    # update, discard). Non-mutating methods are inherited from the frozenset mixin. The
    # including class supplies thetype.
    # Set Methods
    # + add
    # + clear
    # - copy (inherited from frozenset)
    # - difference (inherited from frozenset)
    # + difference_update
    # + discard
    # - intersection (inherited from frozenset)
    # + intersection_update
    # - isdisjoint (inherited from frozenset)
    # - issubset (inherited from frozenset)
    # - issuperset (inherited from frozenset)
    # + pop
    # + remove
    # - symmetric_difference (inherited from frozenset)
    # + symmetric_difference_update
    # - union (inherited from frozenset)
    # + update

    @make_dynamo_test
    def test_add(self):
        # [FUNCTION DOC] Dynamo test: add inserts new elements, ignores existing ones, raises a
        # type error for an unhashable element and for no argument, and works unbound.
        p = self.thetype("abc")
        p.add("d")
        self.assertEqual(p, {"a", "b", "c", "d"})
        p.add("a")
        self.assertEqual(p, {"a", "b", "c", "d"})
        self.assertRaises(TypeError, p.add, ["ab"])
        self.assertRaises(TypeError, p.add)
        self.thetype.add(p, "e")
        self.assertEqual(p, {"a", "b", "c", "d", "e"})

    @make_dynamo_test
    def test_clear(self):
        # [FUNCTION DOC] Dynamo test: clear empties the set, both bound and via the unbound
        # method.
        p = self.thetype("abc")
        p.clear()
        self.assertEqual(p, set())
        p = self.thetype("abc")
        self.thetype.clear(p)
        self.assertEqual(len(p), 0)

    @make_dynamo_test
    def test_remove(self):
        # [FUNCTION DOC] Dynamo test: remove returns nothing and deletes an element, raises a
        # key error for a missing element, and works unbound.
        p = self.thetype("abc")
        self.assertEqual(p.remove("a"), None)
        self.assertEqual(p, {"b", "c"})
        self.assertRaises(KeyError, p.remove, "a")
        p = self.thetype("abc")
        self.thetype.remove(p, "b")
        self.assertEqual(p, self.thetype({"a", "c"}))

    @make_dynamo_test
    def test_intersection_update(self):
        # [FUNCTION DOC] Dynamo test: intersection update with two sets keeps the common element
        # and returns none, raises a type error for an unhashable-element argument, and works
        # unbound.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        self.assertIsNone(set1.intersection_update(set2, set3))
        self.assertEqual(set1, {"apple"})
        self.assertRaises(TypeError, set1.intersection_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.intersection_update(p, q)
        self.assertEqual(p, {"b"})

    @make_dynamo_test
    def test_difference_update(self):
        # [FUNCTION DOC] Dynamo test: difference update with two sets removes their elements and
        # returns none, raises a type error for bad input, and works unbound.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertIsNone(set1.difference_update(set2, set3))
        self.assertEqual(set1, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.difference_update(p, q)
        self.assertEqual(p, {"a", "c"})

    @make_dynamo_test
    def test_symmetric_difference_update(self):
        # [FUNCTION DOC] Dynamo test: symmetric difference update returns none and leaves
        # elements in exactly one of the sets, raises type errors for missing or bad arguments,
        # and works unbound.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        self.assertIsNone(set1.symmetric_difference_update(set2))
        self.assertEqual(set1, {"banana", "cherry", "google", "microsoft"})
        self.assertRaises(TypeError, set1.symmetric_difference_update)
        self.assertRaises(TypeError, set1.symmetric_difference_update, [[]])
        p, q = map(self.thetype, ["abc", "bef"])
        self.thetype.symmetric_difference_update(p, q)
        self.assertEqual(p, {"a", "c", "e", "f"})

    @make_dynamo_test
    def test_pop(self):
        # [FUNCTION DOC] Dynamo test: pop removes some element so it is no longer present,
        # raises a key error on an empty set, and the unbound form pops the sole element of a
        # one-element set.
        set1 = self.thetype({"apple", "banana", "cherry"})
        e = set1.pop()
        self.assertNotIn(e, set1)
        s = self.thetype()
        self.assertRaises(KeyError, s.pop)
        p = self.thetype("a")
        self.assertEqual(self.thetype.pop(p), "a")

    @make_dynamo_test
    def test_update(self):
        # [FUNCTION DOC] Dynamo test: update with two sets adds all their elements, raises a
        # type error for unhashable elements, and works unbound.
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        p.update(q, r)
        self.assertEqual(p, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.update, [[]])
        self.thetype.update(q, r)
        self.assertEqual(q, {"b", "c", "e", "f"})

    @make_dynamo_test
    def test_discard(self):
        # [FUNCTION DOC] Dynamo test: discard removes present elements, ignores absent ones
        # silently, and works unbound.
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.discard("banana")
        set2.discard("cherry")
        self.assertEqual(set1, {"apple", "cherry"})
        self.assertEqual(set2, {"google", "microsoft", "apple"})
        p = self.thetype("abc")
        self.thetype.discard(p, "a")
        self.assertEqual(p, {"b", "c"})

    @make_dynamo_test
    def test_remove_discard_unhashable(self):
        # [FUNCTION DOC] Dynamo test: remove and discard with an unhashable key raise a type
        # error, because the key is hashed before the membership lookup.
        # remove/discard hash the key before the membership check, so an
        # unhashable element raises TypeError rather than KeyError (remove) or
        # silently succeeding (discard). Mirrors CPython set_discard_key.
        self.assertRaises(TypeError, self.thetype("abc").remove, [])
        self.assertRaises(TypeError, self.thetype("abc").discard, [])


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    # [CLASS DOC] Frozenset instantiation of the shared cases; sets thetype to the builtin
    # frozenset.
    thetype = frozenset

    @make_dynamo_test
    def test_copy_preserves_identity(self):
        # [FUNCTION DOC] Dynamo test: copying a frozenset returns the very same object, both via
        # the method and via the unbound class method (compared by id).
        p = frozenset("abc")
        self.assertTrue(id(p) == id(p.copy()))
        self.assertTrue(id(p) == id(frozenset.copy(p)))


class _SetKeyCoercionMixin:
    # [CLASS DOC] HELPER: Mixin for set types that allow remove and discard with a set key,
    # which is coerced to a frozenset for the lookup. Supplies two dynamo tests; the including
    # class provides thetype.
    # set/frozenset allow an (unhashable) set key for remove/discard by
    # coercing it to a frozenset for the lookup, mirroring the set-key
    # fallback in CPython set_remove_impl / set_discard_impl.
    @make_dynamo_test
    def test_remove_set_key(self):
        # [FUNCTION DOC] Dynamo test: removing a set key matches a stored frozenset of equal
        # contents, leaving the container empty; removing a missing set key raises a key error.
        s = self.thetype([frozenset("ab")])
        s.remove(set("ab"))
        self.assertEqual(len(s), 0)
        self.assertRaises(KeyError, self.thetype([frozenset("ab")]).remove, set("z"))

    @make_dynamo_test
    def test_discard_set_key(self):
        # [FUNCTION DOC] Dynamo test: discarding a set key removes the matching stored
        # frozenset; discarding again is a silent no-op.
        s = self.thetype([frozenset("ab")])
        s.discard(set("ab"))
        self.assertEqual(len(s), 0)
        # A second discard of a missing key is a no-op, not an error.
        s.discard(set("ab"))
        self.assertEqual(len(s), 0)


class SetTests(_SetBase, _SetKeyCoercionMixin, _BaseSetTests):
    # [CLASS DOC] Builtin set instantiation of the shared and mutating cases plus the key
    # coercion cases.
    thetype = set

    def test_in_frozenset(self):
        # [FUNCTION DOC] Overrides the membership-in-container test without the dynamo
        # decorator, only calling the parent version (which is decorated).
        super().test_in_frozenset()


class UserDefinedSetTests(_SetBase, _SetKeyCoercionMixin, _BaseSetTests):
    # [CLASS DOC] Instantiation of the shared, mutating and key coercion cases for a user-
    # defined plain subclass of set, held as the nested class.
    class CustomSet(set):
        # [CLASS DOC] HELPER: Empty subclass of set used as the type under test.
        pass

    thetype = CustomSet

    def test_in_frozenset(self):
        # [FUNCTION DOC] Overrides the container membership test, only delegating to the parent
        # version.
        super().test_in_frozenset()

    def test_equality(self):
        # [FUNCTION DOC] Overrides the equality test, only delegating to the parent version.
        super().test_equality()


class UserDefinedFrozensetTests(_FrozensetBase, _BaseSetTests):
    # [CLASS DOC] Instantiation of the frozenset cases for a user-defined plain subclass of
    # frozenset, held as the nested class, plus an extra test for the copy method's result type.
    class CustomFrozenset(frozenset):
        # [CLASS DOC] HELPER: Empty subclass of frozenset used as the type under test.
        pass

    thetype = CustomFrozenset

    @make_dynamo_test
    def test_copy_returns_base_frozenset(self):
        # [FUNCTION DOC] Dynamo test: copying a subclass instance (bound and unbound through the
        # frozenset class) returns a plain frozenset object of the base type, which is a
        # different object from the original.
        p = self.thetype("abc")
        result = p.copy()
        self.assertTrue(type(result) is frozenset)
        self.assertTrue(id(result) != id(p))

        result = frozenset.copy(p)
        self.assertTrue(type(result) is frozenset)
        self.assertTrue(id(result) != id(p))

    def test_in_frozenset(self):
        # [FUNCTION DOC] Overrides the container membership test, only delegating to the parent
        # version.
        super().test_in_frozenset()


class OrderedSetTests(_SetBase, _BaseSetTests):
    # [CLASS DOC] Instantiation of the shared and mutating cases for the insertion-ordered set
    # utility class, imported inside the class body and used as thetype. Adds order- and type-
    # preservation tests.
    from torch.utils._ordered_set import OrderedSet

    thetype = OrderedSet

    def test_in_frozenset(self):
        # [FUNCTION DOC] Overrides the container membership test with a no-op because ordered
        # sets do not compare equal to plain sets of the same elements.
        # We aren't equal w/ other sets due to ordering
        pass

    def test_equality(self):
        # [FUNCTION DOC] Overrides the equality test, only delegating to the parent version.
        super().test_equality()

    @make_dynamo_test
    def test_maintains_order(self):
        # [FUNCTION DOC] Dynamo test: iterating an ordered set returns the elements in insertion
        # order.
        # Test that OrderedSet maintains insertion order
        s = self.thetype(["c", "b", "a"])
        items = list(s)
        self.assertEqual(items, ["c", "b", "a"])

    @make_dynamo_test
    def test_intersection_maintains_order(self):
        # [FUNCTION DOC] Dynamo test: intersection returns an ordered set of the common elements
        # in the order of the first operand.
        # Test that intersection maintains order from first set
        s1 = self.thetype(["a", "b", "c", "d"])
        s2 = self.thetype(["d", "c", "b"])
        result = s1.intersection(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["b", "c", "d"])

    @make_dynamo_test
    def test_union_maintains_order(self):
        # [FUNCTION DOC] Dynamo test: union returns an ordered set with first-set elements
        # followed by new elements from the second set.
        # Test that union maintains order (first set order, then second set new items)
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["c", "d", "e"])
        result = s1.union(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["a", "b", "c", "d", "e"])

    @make_dynamo_test
    def test_difference_maintains_order(self):
        # [FUNCTION DOC] Dynamo test: difference returns an ordered set with the first operand's
        # remaining elements in their original order.
        # Test that difference maintains order from first set
        s1 = self.thetype(["a", "b", "c", "d"])
        s2 = self.thetype(["b", "d"])
        result = s1.difference(s2)
        self.assertIsInstance(result, self.thetype)
        self.assertEqual(list(result), ["a", "c"])

    @make_dynamo_test
    def test_symmetric_difference_maintains_order(self):
        # [FUNCTION DOC] Dynamo test: symmetric difference returns an ordered set; only set
        # equality of the contents is checked, not their order.
        # Test that symmetric_difference maintains order
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["c", "d", "e"])
        result = s1.symmetric_difference(s2)
        self.assertIsInstance(result, self.thetype)
        # Should have items from s1 not in s2, then items from s2 not in s1
        self.assertEqual(set(result), {"a", "b", "d", "e"})

    @make_dynamo_test
    def test_copy_preserves_type(self):
        # [FUNCTION DOC] Dynamo test: copy returns an ordered set of the same type with the same
        # element order.
        # Test that copy returns an OrderedSet
        s = self.thetype(["a", "b", "c"])
        s_copy = s.copy()
        self.assertIsInstance(s_copy, self.thetype)
        self.assertEqual(list(s_copy), ["a", "b", "c"])

    @make_dynamo_test
    def test_binop_preserves_type(self):
        # [FUNCTION DOC] Dynamo test: the union, intersection, difference and symmetric
        # difference operators each return an ordered set.
        # Test that binary operations preserve OrderedSet type
        s1 = self.thetype(["a", "b", "c"])
        s2 = self.thetype(["b", "c", "d"])

        # Test |
        result = s1 | s2
        self.assertIsInstance(result, self.thetype)

        # Test &
        result = s1 & s2
        self.assertIsInstance(result, self.thetype)

        # Test -
        result = s1 - s2
        self.assertIsInstance(result, self.thetype)

        # Test ^
        result = s1 ^ s2
        self.assertIsInstance(result, self.thetype)

    @make_dynamo_test
    def test_construct_from_generator(self):
        # [FUNCTION DOC] Dynamo test: constructing from a generator expression keeps the
        # generated order, with elements upper-cased.
        s = self.thetype(x.upper() for x in ["a", "b", "c"])
        self.assertEqual(list(s), ["A", "B", "C"])

    @make_dynamo_test
    def test_construct_from_map(self):
        # [FUNCTION DOC] Dynamo test: constructing from a map object keeps order and gives the
        # strings one, two, three digits.
        s = self.thetype(map(str, [1, 2, 3]))
        self.assertEqual(list(s), ["1", "2", "3"])

    @make_dynamo_test
    def test_construct_from_range(self):
        # [FUNCTION DOC] Dynamo test: constructing from a range of four gives the ordered
        # elements zero through three.
        s = self.thetype(range(4))
        self.assertEqual(list(s), [0, 1, 2, 3])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
