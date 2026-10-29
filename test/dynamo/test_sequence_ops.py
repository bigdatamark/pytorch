# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for sequence protocol operations: concatenation, in-place
# concatenation, repetition, indexing and slicing, containment, count and index, and related
# behavior across list, tuple, deque, string, range and user-defined subclasses or sequence-like
# classes. Most cases use the make_dynamo_test decorator (run eagerly, then compiled with an
# assert-style backend) and some are parametrized; several error cases check that the same
# exception type is raised as in eager Python. Defines small helper container classes used by
# the tests below.

"""Tests for sequence protocol operations (sq_*) in PyTorch Dynamo."""

import collections
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


# --- User-defined sequence classes ---


class UserDefinedList(list):
    # [CLASS DOC] HELPER: Empty subclass of list used to check subclass handling.
    """User-defined list subclass."""


class UserDefinedTuple(tuple):
    # [CLASS DOC] HELPER: Empty subclass of tuple with an empty slots declaration, used to check
    # subclass handling.
    """User-defined tuple subclass."""

    __slots__ = []


class UserDefinedDeque(collections.deque):
    # [CLASS DOC] HELPER: Empty subclass of the collections deque, used to check subclass
    # handling.
    """User-defined deque subclass."""


class UserDefinedSequence:
    # [CLASS DOC] HELPER: Minimal user sequence class that wraps a list in the items attribute
    # and implements addition, in-place addition and equality only against its own kind; other
    # operand types yield the not-implemented result for addition, and false for equality.
    """User-defined sequence class with __add__ and __iadd__."""

    def __init__(self, items):
        # [FUNCTION DOC] HELPER: Stores a new list copy of the given iterable as the items
        # attribute.
        self.items = list(items)

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Returns a new instance holding the concatenation of both item
        # lists when the other operand is the same class; otherwise returns the not-implemented
        # marker.
        if isinstance(other, UserDefinedSequence):
            return UserDefinedSequence(self.items + other.items)
        return NotImplemented

    def __iadd__(self, other):
        # [FUNCTION DOC] HELPER: When the other operand is the same class, extends this item
        # list in place and returns self; otherwise returns the not-implemented marker.
        if isinstance(other, UserDefinedSequence):
            self.items.extend(other.items)
            return self
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equal to another instance of the class when the item lists are
        # equal; any other type compares unequal.
        if isinstance(other, UserDefinedSequence):
            return self.items == other.items
        return False

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a string with the class name and the item list.
        return f"UserDefinedSequence({self.items})"


class IndexLike:
    # [CLASS DOC] HELPER: Wrapper holding an integer in the value attribute that can be used
    # wherever an integer index is accepted, via the index protocol.
    """Object usable wherever an integer index is expected (defines __index__)."""

    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the integer value.
        self.value = value

    def __index__(self):
        # [FUNCTION DOC] HELPER: Returns the stored integer, implementing the index protocol.
        return self.value


class TestSqConcat(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for sequence concatenation and in-place concatenation on lists, tuples,
    # strings, deques and user-defined subclasses, plus deque repeat, rotate, re-init, copy and
    # iterator mutation detection. Setup enables the config option for tracing unittest
    # assertions and teardown restores the previous value.
    """Tests for sq_concat (+) and sq_inplace_concat (+=) operators for sequences."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setup, saves the current trace-unittest config
        # value in an attribute and turns the option on.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the parent teardown, then restores the saved trace-
        # unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- List concatenation ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ([1, 2], [3, 4], [1, 2, 3, 4]),
            ([], [1, 2], [1, 2]),
            ([1, 2], [], [1, 2]),
        ],
    )
    @make_dynamo_test
    def test_list_concat(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized dynamo test: adding two lists (normal, empty-first, empty-
        # second) gives the expected list.
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_list_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: adding two empty lists gives an empty list.
        self.assertEqual([] + [], [])

    @make_dynamo_test
    def test_list_concat_chained(self):
        # [FUNCTION DOC] Dynamo test: chained list additions give the expected concatenation.
        self.assertEqual([1] + [2] + [3], [1, 2, 3])
        self.assertEqual([1, 2] + [3] + [4, 5], [1, 2, 3, 4, 5])

    # --- Tuple concatenation ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ((1, 2), (3, 4), (1, 2, 3, 4)),
            ((), (1, 2), (1, 2)),
            ((1, 2), (), (1, 2)),
        ],
    )
    @make_dynamo_test
    def test_tuple_concat(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized dynamo test: adding two tuples (normal, empty-first,
        # empty-second) gives the expected tuple.
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_tuple_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: adding two empty tuples gives an empty tuple.
        self.assertEqual(() + (), ())

    @make_dynamo_test
    def test_tuple_concat_chained(self):
        # [FUNCTION DOC] Dynamo test: chained tuple additions give the expected tuple.
        self.assertEqual((1,) + (2,) + (3,), (1, 2, 3))

    # --- String concatenation (via ConstantVariable) ---

    @make_dynamo_test
    def test_string_concat(self):
        # [FUNCTION DOC] Dynamo test: string concatenation (constant values) with spaces, short
        # strings and an empty string gives the expected strings.
        self.assertEqual("hello" + " " + "world", "hello world")
        self.assertEqual("a" + "b", "ab")
        self.assertEqual("" + "test", "test")

    @make_dynamo_test
    def test_string_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: concatenating two empty strings gives an empty string.
        self.assertEqual("" + "", "")

    # --- Inplace list concatenation (+=) ---

    @make_dynamo_test
    def test_list_inplace_concat(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of a list extends the list variable with
        # the other list's items.
        x = [1, 2]
        x += [3, 4]
        self.assertEqual(x, [1, 2, 3, 4])

    @make_dynamo_test
    def test_list_inplace_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of a non-empty list to an empty list
        # gives that content.
        x = []
        x += [1, 2]
        self.assertEqual(x, [1, 2])

    @make_dynamo_test
    def test_list_inplace_concat_to_empty(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of an empty list leaves a list
        # unchanged.
        x = [1, 2]
        x += []
        self.assertEqual(x, [1, 2])

    # --- Type mismatch errors ---

    @make_dynamo_test
    def test_list_concat_with_tuple_raises(self):
        # [FUNCTION DOC] Dynamo test: adding a tuple to a list raises a type error.
        with self.assertRaises(TypeError):
            [1, 2] + (3, 4)

    @make_dynamo_test
    def test_tuple_concat_with_list_raises(self):
        # [FUNCTION DOC] Dynamo test: adding a list to a tuple raises a type error.
        with self.assertRaises(TypeError):
            (1, 2) + [3, 4]

    @make_dynamo_test
    def test_list_concat_with_string_raises(self):
        # [FUNCTION DOC] Dynamo test: adding a string to a list raises a type error.
        with self.assertRaises(TypeError):
            [1, 2] + "string"

    @make_dynamo_test
    def test_list_concat_with_none_raises(self):
        # [FUNCTION DOC] Dynamo test: adding none to a list raises a type error.
        with self.assertRaises(TypeError):
            [1, 2] + None

    # --- User-defined list subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_list_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two instances of the user list subclass produces a
        # result whose elements are the combined elements.
        a = UserDefinedList([1, 2])
        b = UserDefinedList([3, 4])
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_user_defined_list_concat_with_list(self):
        # [FUNCTION DOC] Dynamo test: adding a user list subclass instance and a plain list
        # produces the combined elements.
        a = UserDefinedList([1, 2])
        b = [3, 4]
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_user_defined_list_concat_returns_plain_list(self):
        # [FUNCTION DOC] Dynamo test: out-of-place addition of two user list subclass instances
        # returns an object whose type is exactly the plain list type, not the subclass.
        # out-of-place C sq_concat constructs a fresh base-type object
        result = UserDefinedList([1]) + UserDefinedList([2])
        self.assertIs(type(result), list)

    @unittest.expectedFailure
    @make_dynamo_test
    def test_user_defined_list_inplace_concat(self):
        # [FUNCTION DOC] Expected failure test: in-place addition of a list to a user list
        # subclass instance should extend the same object, keeping its subclass type and
        # identity. Marked expected failure because Dynamo currently does not satisfy at least
        # one of these checks.
        # in-place C sq_inplace_concat mutates and returns self: subclass
        # type and identity preserved
        a = UserDefinedList([1])
        b = a
        a += [2]
        self.assertEqual(list(a), [1, 2])
        self.assertIs(type(a), UserDefinedList)
        self.assertIs(a, b)

    # --- User-defined tuple subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_tuple_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two user tuple subclass instances gives a tuple
        # equal to the combined elements.
        a = UserDefinedTuple([1, 2])
        b = UserDefinedTuple([3, 4])
        result = a + b
        self.assertEqual(result, (1, 2, 3, 4))

    @make_dynamo_test
    def test_user_defined_tuple_concat_with_tuple(self):
        # [FUNCTION DOC] Dynamo test: adding a user tuple subclass instance and a plain tuple
        # gives the combined tuple.
        a = UserDefinedTuple([1, 2])
        b = (3, 4)
        result = a + b
        self.assertEqual(result, (1, 2, 3, 4))

    # --- User-defined tuple subclass construction ---

    @make_dynamo_test
    def test_user_defined_tuple_construct_empty(self):
        # [FUNCTION DOC] Dynamo test: constructing the user tuple subclass with no argument
        # gives an empty tuple of length zero whose type is the subclass.
        # tuple.__new__(cls) with no iterable arg builds an empty tuple.
        a = UserDefinedTuple()
        self.assertEqual(a, ())
        self.assertEqual(len(a), 0)
        self.assertIs(type(a), UserDefinedTuple)

    @make_dynamo_test
    def test_user_defined_tuple_construct_from_iterables(self):
        # [FUNCTION DOC] Dynamo test: constructing the user tuple subclass from an empty list,
        # an empty set, a list, a generator expression, and in a generator of nested single
        # element instances gives the expected tuples.
        self.assertEqual(UserDefinedTuple([]), ())
        self.assertEqual(UserDefinedTuple(set()), ())
        self.assertEqual(UserDefinedTuple([1, 2, 3]), (1, 2, 3))
        self.assertEqual(UserDefinedTuple(obj for obj in [1, 2, 3]), (1, 2, 3))
        nested = tuple(UserDefinedTuple([obj]) for obj in [1, 2, 3])
        self.assertEqual(nested, ((1,), (2,), (3,)))

    # --- User-defined deque subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_deque_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two user deque subclass instances gives a deque
        # with the combined elements.
        a = UserDefinedDeque([1, 2])
        b = UserDefinedDeque([3, 4])
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_user_defined_deque_maxlen(self):
        # [FUNCTION DOC] Dynamo test: a user deque subclass built with a maximum length of two
        # from three items keeps only the last two and reports a maxlen of two.
        d = UserDefinedDeque([1, 2, 3], maxlen=2)
        self.assertEqual(list(d), [2, 3])
        self.assertEqual(d.maxlen, 2)

    def test_user_defined_deque_graph_input(self):
        # [FUNCTION DOC] Plain test (not the dynamo decorator): a user deque subclass is passed
        # as an input of a fully compiled function that appends a value and returns the length.
        # For no max length and for a max length of three, compares the return value, final
        # contents and maxlen between a compiled run and an eager run on a separate copy,
        # showing the mutation is replayed on the original input.
        # A deque subclass passed in as a graph input is built by builder.py
        # (sourced path), and in-place mutations are replayed onto the original.
        def fn(d):
            # [FUNCTION DOC] HELPER: Appends the value ninety-nine to the deque and returns its
            # length.
            d.append(99)
            return len(d)

        for maxlen in (None, 3):
            d = UserDefinedDeque([1, 2, 3], maxlen=maxlen)
            ref = UserDefinedDeque([1, 2, 3], maxlen=maxlen)
            ref_ret = fn(ref)
            ret = torch.compile(fn, backend="eager", fullgraph=True)(d)
            self.assertEqual(ret, ref_ret)
            self.assertEqual(list(d), list(ref))
            self.assertEqual(d.maxlen, ref.maxlen)

    @make_dynamo_test
    def test_user_defined_deque_inplace_concat(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of two user deque subclass instances
        # extends the first one with the second's items.
        a = UserDefinedDeque([1, 2])
        a += UserDefinedDeque([3, 4])
        self.assertEqual(list(a), [1, 2, 3, 4])

    # --- Deque concatenation ---

    @make_dynamo_test
    def test_deque_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two deques gives a deque with the combined
        # elements.
        d1 = collections.deque([1, 2])
        d2 = collections.deque([3, 4])
        result = d1 + d2
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_deque_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: adding an empty deque to a non-empty deque gives the non-
        # empty content.
        d1 = collections.deque([])
        d2 = collections.deque([1, 2])
        result = d1 + d2
        self.assertEqual(list(result), [1, 2])

    @make_dynamo_test
    def test_deque_concat_with_maxlen(self):
        # [FUNCTION DOC] Dynamo test: adding an unbounded deque to a deque with maximum length
        # three keeps only the last three elements, because the result takes the left operand's
        # bound.
        d1 = collections.deque([1, 2], maxlen=3)
        d2 = collections.deque([3, 4])
        result = d1 + d2
        # Result respects left operand's maxlen of 3
        self.assertEqual(list(result), [2, 3, 4])

    # --- Inplace deque concatenation (+=) ---

    @make_dynamo_test
    def test_deque_inplace_concat(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of two deques extends the first with the
        # second's items.
        d = collections.deque([1, 2])
        d += collections.deque([3, 4])
        self.assertEqual(list(d), [1, 2, 3, 4])

    @make_dynamo_test
    def test_deque_inplace_concat_with_maxlen(self):
        # [FUNCTION DOC] Dynamo test: in-place addition to a deque with maximum length three
        # evicts older items, leaving the last three elements.
        d = collections.deque([1, 2], maxlen=3)
        d += collections.deque([3, 4])
        # Result respects maxlen of 3
        self.assertEqual(list(d), [2, 3, 4])

    # --- Inplace deque repeat (*=) ---

    @make_dynamo_test
    def test_deque_inplace_repeat(self):
        # [FUNCTION DOC] Dynamo test: in-place repetition of a deque by three repeats its
        # contents three times.
        d = collections.deque([1, 2])
        d *= 3
        self.assertEqual(list(d), [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_deque_inplace_repeat_with_maxlen(self):
        # [FUNCTION DOC] Dynamo test: in-place repetition of a bounded deque (maximum length
        # three) keeps only the last three items of the repeated contents.
        d = collections.deque([1, 2], maxlen=3)
        d *= 3
        # A bounded deque keeps the last maxlen items after repeating.
        self.assertEqual(list(d), [2, 1, 2])

    @make_dynamo_test
    def test_deque_inplace_repeat_maxlen_zero(self):
        # [FUNCTION DOC] Dynamo test: in-place repetition of a deque with maximum length zero
        # leaves it empty.
        d = collections.deque([1, 2], maxlen=0)
        d *= 3
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_inplace_repeat_zero(self):
        # [FUNCTION DOC] Dynamo test: in-place repetition by zero empties the deque.
        d = collections.deque([1, 2, 3])
        d *= 0
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_inplace_repeat_negative(self):
        # [FUNCTION DOC] Dynamo test: in-place repetition by a negative number empties the
        # deque.
        d = collections.deque([1, 2, 3])
        d *= -1
        self.assertEqual(list(d), [])

    # --- deque.rotate ---

    @make_dynamo_test
    def test_deque_rotate_default(self):
        # [FUNCTION DOC] Dynamo test: rotating a five element deque with no argument moves the
        # last element to the front (default rotation of one to the right).
        d = collections.deque([1, 2, 3, 4, 5])
        d.rotate()
        self.assertEqual(list(d), [5, 1, 2, 3, 4])

    @make_dynamo_test
    def test_deque_rotate_positive(self):
        # [FUNCTION DOC] Dynamo test: rotating by two moves the last two elements to the front.
        d = collections.deque([1, 2, 3, 4, 5])
        d.rotate(2)
        self.assertEqual(list(d), [4, 5, 1, 2, 3])

    @make_dynamo_test
    def test_deque_rotate_negative(self):
        # [FUNCTION DOC] Dynamo test: rotating by minus two moves the first two elements to the
        # back.
        d = collections.deque([1, 2, 3, 4, 5])
        d.rotate(-2)
        self.assertEqual(list(d), [3, 4, 5, 1, 2])

    @make_dynamo_test
    def test_deque_rotate_wraps(self):
        # [FUNCTION DOC] Dynamo test: rotating by seven on five elements behaves like rotating
        # by two (wraps around).
        d = collections.deque([1, 2, 3, 4, 5])
        d.rotate(7)
        self.assertEqual(list(d), [4, 5, 1, 2, 3])

    @make_dynamo_test
    def test_deque_rotate_index_arg(self):
        # [FUNCTION DOC] Dynamo test: rotation accepts an object implementing the index protocol
        # as the count.
        d = collections.deque([1, 2, 3, 4, 5])
        d.rotate(IndexLike(2))
        self.assertEqual(list(d), [4, 5, 1, 2, 3])

    @make_dynamo_test
    def test_deque_rotate_short(self):
        # [FUNCTION DOC] Dynamo test: rotating a one element deque and an empty deque leaves
        # them unchanged.
        d = collections.deque([1])
        d.rotate(3)
        self.assertEqual(list(d), [1])
        d = collections.deque([])
        d.rotate(2)
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_rotate_with_maxlen(self):
        # [FUNCTION DOC] Dynamo test: rotating a full bounded deque by one and then appending a
        # value drops the front element, giving the expected contents.
        d = collections.deque([1, 2, 3], maxlen=3)
        d.rotate(1)
        d.append(9)
        self.assertEqual(list(d), [1, 2, 9])

    @make_dynamo_test
    def test_deque_rotate_noop_bumps_state(self):
        # [FUNCTION DOC] Dynamo test: rotating inside a loop over a deque raises a runtime error
        # (mutation during iteration) even when the rotation count is zero or equal to the
        # length, so no element moves.
        # rotate() bumps the iteration-state counter whenever len > 1, even when
        # n is a multiple of len (no element actually moves), so mutation during
        # iteration is still detected.
        for n in (0, 4):
            d = collections.deque([1, 2, 3, 4])
            with self.assertRaises(RuntimeError):
                for _ in d:
                    d.rotate(n)

    # --- deque setattr (deque has no __dict__) ---

    @make_dynamo_test
    def test_deque_setattr_maxlen_readonly(self):
        # [FUNCTION DOC] Dynamo test: assigning to the maxlen attribute of a deque raises an
        # attribute error because it is read-only.
        d = collections.deque([1, 2, 3])
        with self.assertRaises(AttributeError):
            d.maxlen = 10

    @make_dynamo_test
    def test_deque_setattr_unknown_attr(self):
        # [FUNCTION DOC] Dynamo test: assigning a new unknown attribute on a deque raises an
        # attribute error because deques have no instance dictionary.
        d = collections.deque([1, 2, 3])
        with self.assertRaises(AttributeError):
            d.foo = 1

    # --- list re-init (list.__init__) ---

    @make_dynamo_test
    def test_list_reinit_clears_and_extends(self):
        # [FUNCTION DOC] Dynamo test: calling the init method again on a list replaces its
        # content with the new iterable.
        x = list(range(-5, 0))
        x.__init__(range(3))
        self.assertEqual(x, [0, 1, 2])

    @make_dynamo_test
    def test_list_reinit_empty(self):
        # [FUNCTION DOC] Dynamo test: calling the init method with no argument empties the list.
        x = [1, 2, 3]
        x.__init__()
        self.assertEqual(x, [])

    @make_dynamo_test
    def test_list_reinit_self_referential(self):
        # [FUNCTION DOC] Dynamo test: re-initializing a list with itself yields an empty list,
        # because the list is cleared before being extended.
        # CPython clears before extending, so a.__init__(a) yields [].
        x = [1, 2, 3]
        x.__init__(x)
        self.assertEqual(x, [])

    @make_dynamo_test
    def test_list_reinit_non_iterable_clears_then_raises(self):
        # [FUNCTION DOC] Dynamo test: re-initializing a list with a non-iterable raises a type
        # error, and the list is left empty because it is cleared first.
        # clear happens before extend, so the TypeError leaves the list empty.
        x = [1, 2, 3]
        with self.assertRaises(TypeError):
            x.__init__(5)
        self.assertEqual(x, [])

    # --- deque re-init (deque.__init__) ---

    @make_dynamo_test
    def test_deque_reinit_clears_and_extends(self):
        # [FUNCTION DOC] Dynamo test: calling the init method on a deque replaces the existing
        # content with the new iterable.
        d = collections.deque(range(-5, 0))
        d.__init__(range(3))
        self.assertEqual(list(d), [0, 1, 2])

    @make_dynamo_test
    def test_deque_reinit_resets_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing a bounded deque without a bound resets the
        # maximum length to none and keeps all new items.
        d = collections.deque([1, 2], maxlen=2)
        d.__init__(range(4))
        # No maxlen passed -> maxlen reset to None, all items kept
        self.assertEqual(d.maxlen, None)
        self.assertEqual(list(d), [0, 1, 2, 3])

    @make_dynamo_test
    def test_deque_reinit_with_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing a deque with an iterable and a maximum
        # length of three keeps the last three items and sets the bound.
        d = collections.deque([1, 2])
        d.__init__(range(5), 3)
        self.assertEqual(d.maxlen, 3)
        self.assertEqual(list(d), [2, 3, 4])

    @make_dynamo_test
    def test_deque_reinit_empty(self):
        # [FUNCTION DOC] Dynamo test: re-initializing with no arguments empties the deque.
        d = collections.deque([1, 2, 3])
        d.__init__()
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_reinit_negative_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing with a negative maximum length raises a
        # value error.
        d = collections.deque([1, 2])
        with self.assertRaises(ValueError):
            d.__init__(range(3), -1)

    @make_dynamo_test
    def test_deque_reinit_float_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing with a float maximum length raises a type
        # error.
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.__init__(maxlen=2.0)

    @make_dynamo_test
    def test_deque_reinit_str_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing with a string maximum length raises a type
        # error.
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.__init__(maxlen="3")

    @make_dynamo_test
    def test_deque_ctor_float_maxlen(self):
        # [FUNCTION DOC] Dynamo test: constructing a deque with a float maximum length raises a
        # type error.
        with self.assertRaises(TypeError):
            collections.deque([1, 2], maxlen=2.0)

    @make_dynamo_test
    def test_deque_reinit_overflow_maxlen(self):
        # [FUNCTION DOC] Dynamo test: re-initializing with a maximum length of two to the power
        # sixty-three raises an overflow error.
        d = collections.deque([1, 2])
        with self.assertRaises(OverflowError):
            d.__init__(maxlen=2**63)

    @make_dynamo_test
    def test_deque_ctor_bool_maxlen(self):
        # [FUNCTION DOC] Dynamo test: a boolean true is accepted as a maximum length (treated as
        # one), so a deque built from five items keeps only the last.
        # bool is an int subclass; CPython accepts it.
        d = collections.deque(range(5), maxlen=True)
        self.assertEqual(list(d), [4])

    # --- deque copy (deque.copy / copy.copy / __copy__) ---

    @make_dynamo_test
    def test_deque_copy_method(self):
        # [FUNCTION DOC] Dynamo test: the deque copy method returns an equal deque with no bound
        # that is independent of later appends to the original.
        d = collections.deque([1, 2, 3])
        e = d.copy()
        self.assertEqual(list(e), [1, 2, 3])
        self.assertIsNone(e.maxlen)
        d.append(4)
        self.assertEqual(list(e), [1, 2, 3])

    @make_dynamo_test
    def test_deque_copy_preserves_maxlen(self):
        # [FUNCTION DOC] Dynamo test: copying a bounded deque preserves both contents and the
        # maximum length.
        d = collections.deque([1, 2, 3], maxlen=3)
        e = d.copy()
        self.assertEqual(list(e), [1, 2, 3])
        self.assertEqual(e.maxlen, 3)

    @make_dynamo_test
    def test_deque_copy_module(self):
        # [FUNCTION DOC] Dynamo test: the copy module's shallow copy of a bounded deque
        # preserves content and maximum length.
        import copy

        d = collections.deque([1, 2], maxlen=5)
        e = copy.copy(d)
        self.assertEqual(list(e), [1, 2])
        self.assertEqual(e.maxlen, 5)

    @make_dynamo_test
    def test_deque_copy_shares_elements(self):
        # [FUNCTION DOC] Dynamo test: a deque copy is shallow: a mutable element shared with the
        # original shows the change made through the original list.
        mut = [10]
        d = collections.deque([mut])
        e = d.copy()
        mut[0] = 11
        self.assertEqual(list(d), list(e))

    @make_dynamo_test
    def test_deque_copy_too_many_args(self):
        # [FUNCTION DOC] Dynamo test: calling the deque copy method with an extra argument
        # raises a type error.
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.copy(1)

    # --- deque iterator mutation detection ---

    @make_dynamo_test
    def test_deque_iter_mutated_pop_raises(self):
        # [FUNCTION DOC] Dynamo test: a deque iterator becomes invalid if the deque is mutated
        # by a pop after creation; advancing it raises a runtime error.
        d = collections.deque("abcdefg")
        it = iter(d)
        d.pop()
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_mutated_append_raises(self):
        # [FUNCTION DOC] Dynamo test: an iterator over an empty deque raises a runtime error
        # when advanced after an append.
        d = collections.deque()
        it = iter(d)
        d.append(10)
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_reversed_iter_mutated_raises(self):
        # [FUNCTION DOC] Dynamo test: a reversed iterator of a deque raises a runtime error when
        # advanced after the deque is mutated by an append on the left.
        d = collections.deque("abc")
        it = reversed(d)
        d.appendleft("z")
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_unmutated_ok(self):
        # [FUNCTION DOC] Dynamo test: iterating forward and in reverse over an unmutated deque
        # gives the expected sequences.
        d = collections.deque("abc")
        self.assertEqual(list(iter(d)), ["a", "b", "c"])
        self.assertEqual(list(reversed(d)), ["c", "b", "a"])

    @make_dynamo_test
    def test_deque_reverse_iterator_type(self):
        # [FUNCTION DOC] Dynamo test: takes the reverse iterator type from an empty deque and
        # uses that type to build an iterator over a four element deque; the produced sequence
        # equals the reversal of the same letters.
        klass = type(reversed(collections.deque()))
        self.assertEqual(list(klass(collections.deque("abcd"))), list(reversed("abcd")))

    @make_dynamo_test
    def test_deque_forward_iterator_type(self):
        # [FUNCTION DOC] Dynamo test: takes the forward iterator type from an empty deque and
        # uses it to build an iterator over a four letter deque; the produced sequence is the
        # letters in order.
        klass = type(iter(collections.deque()))
        self.assertEqual(list(klass(collections.deque("abcd"))), list("abcd"))

    @make_dynamo_test
    def test_deque_reverse_iterator_bad_arg(self):
        # [FUNCTION DOC] Dynamo test: building a deque reverse iterator from a plain list raises
        # a type error.
        klass = type(reversed(collections.deque()))
        with self.assertRaises(TypeError):
            klass([1, 2, 3])

    # Eager parity: only mutations that bump CPython's deque->state invalidate
    # an active iterator. reverse() and setitem do NOT bump; delitem and a
    # non-empty extend do.

    @make_dynamo_test
    def test_deque_iter_reverse_does_not_raise(self):
        # [FUNCTION DOC] Dynamo test: reversing a deque in place does not invalidate an existing
        # iterator, so advancing it does not raise (mirrors CPython, where reverse does not bump
        # the deque state).
        d = collections.deque("abcdefg")
        it = iter(d)
        d.reverse()
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_setitem_does_not_raise(self):
        # [FUNCTION DOC] Dynamo test: assigning an item of a deque by index does not invalidate
        # an existing iterator; advancing it still works.
        d = collections.deque("abcdefg")
        it = iter(d)
        d[0] = "Z"
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_delitem_raises(self):
        # [FUNCTION DOC] Dynamo test: deleting an item by index invalidates an existing deque
        # iterator, so advancing it raises a runtime error.
        d = collections.deque("abcdefg")
        it = iter(d)
        del d[0]
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_extend_empty_does_not_raise(self):
        # [FUNCTION DOC] Dynamo test: extending a deque with an empty list does not invalidate
        # an iterator; advancing still works.
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extend([])
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_extendleft_empty_does_not_raise(self):
        # [FUNCTION DOC] Dynamo test: extending a deque on the left with an empty list does not
        # invalidate an iterator; advancing still works.
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extendleft([])
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_extend_nonempty_raises(self):
        # [FUNCTION DOC] Dynamo test: extending a deque with a non-empty list invalidates an
        # iterator, so advancing raises a runtime error.
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extend([1])
        with self.assertRaises(RuntimeError):
            next(it)

    # --- torch.Size concatenation ---

    @make_dynamo_test
    def test_torch_size_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two torch.Size objects gives a torch.Size with the
        # combined dimensions and the result is an instance of torch.Size.
        s1 = torch.Size([1, 2, 3])
        s2 = torch.Size([4, 5])
        result = s1 + s2
        self.assertEqual(result, torch.Size([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_empty(self):
        # [FUNCTION DOC] Dynamo test: adding a torch.Size of two dimensions to an empty
        # torch.Size gives the equal torch.Size.
        s1 = torch.Size([])
        s2 = torch.Size([1, 2])
        result = s1 + s2
        self.assertEqual(result, torch.Size([1, 2]))

    @make_dynamo_test
    def test_torch_size_inplace_concat(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of two torch.Size objects gives the
        # combined dimensions, still a torch.Size.
        s = torch.Size([1, 2])
        s += torch.Size([3, 4])
        self.assertEqual(s, torch.Size([1, 2, 3, 4]))
        self.assertIsInstance(s, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_with_tuple(self):
        # [FUNCTION DOC] Dynamo test: adding a plain tuple to a torch.Size works (torch.Size
        # subclasses tuple) and the result is a torch.Size with the combined entries.
        # torch.Size + tuple works since torch.Size subclasses tuple
        s = torch.Size([1, 2])
        result = s + (3, 4)
        self.assertEqual(list(result), [1, 2, 3, 4])
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_with_user_defined_tuple(self):
        # [FUNCTION DOC] Dynamo test: adding a user tuple subclass instance to a torch.Size
        # gives a torch.Size with the combined entries.
        # torch.Size + UserDefinedTuple should work
        s = torch.Size([1, 2])
        ud_tuple = UserDefinedTuple([3, 4])
        result = s + ud_tuple
        self.assertEqual(list(result), [1, 2, 3, 4])
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_multiple_tuples(self):
        # [FUNCTION DOC] Dynamo test: chaining the addition of a plain tuple and a user tuple
        # subclass onto a torch.Size gives a torch.Size with all six entries in order.
        # torch.Size + regular tuple + UserDefinedTuple
        s = torch.Size([1, 2])
        result = s + (3, 4) + UserDefinedTuple([5, 6])
        self.assertEqual(list(result), [1, 2, 3, 4, 5, 6])
        self.assertIsInstance(result, torch.Size)

    # --- User-defined sequence class ---

    @make_dynamo_test
    def test_user_defined_sequence_concat(self):
        # [FUNCTION DOC] Dynamo test: adding two instances of the user sequence helper class
        # gives an instance equal to the one holding both item lists.
        a = UserDefinedSequence([1, 2])
        b = UserDefinedSequence([3, 4])
        result = a + b
        self.assertEqual(result, UserDefinedSequence([1, 2, 3, 4]))

    @make_dynamo_test
    def test_user_defined_sequence_inplace_concat(self):
        # [FUNCTION DOC] Dynamo test: in-place addition of the user sequence helper class
        # extends the left instance with the right's items.
        a = UserDefinedSequence([1, 2])
        a += UserDefinedSequence([3, 4])
        self.assertEqual(a, UserDefinedSequence([1, 2, 3, 4]))

    @make_dynamo_test
    def test_user_defined_sequence_empty(self):
        # [FUNCTION DOC] Dynamo test: adding a non-empty user sequence instance to an empty one
        # gives the non-empty items.
        a = UserDefinedSequence([])
        b = UserDefinedSequence([1, 2])
        result = a + b
        self.assertEqual(result, UserDefinedSequence([1, 2]))

    @make_dynamo_test
    def test_user_defined_sequence_chained(self):
        # [FUNCTION DOC] Dynamo test: chaining the addition of three user sequence instances
        # gives all items in order.
        a = UserDefinedSequence([1])
        b = UserDefinedSequence([2])
        c = UserDefinedSequence([3])
        result = a + b + c
        self.assertEqual(result, UserDefinedSequence([1, 2, 3]))


instantiate_parametrized_tests(TestSqConcat)


class TestSqRepeat(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for sequence repetition by multiplication and in-place multiplication on
    # user-defined list and tuple subclasses. Setup enables the trace-unittest config option and
    # teardown restores it.
    """Tests for sq_repeat (*) and sq_inplace_repeat (*=) on sequences."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setup, saves the current trace-unittest config
        # value, and enables it.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the parent teardown, then restores the saved trace-
        # unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- User-defined subclasses: the inherited C sq_repeat slot ---

    @make_dynamo_test
    def test_user_defined_list_repeat(self):
        # [FUNCTION DOC] Dynamo test: multiplying a user list subclass instance by two gives the
        # repeated elements as a plain list, not the subclass type.
        # CPython runs list's inherited C sq_repeat: result is a plain list,
        # not the subclass
        result = UserDefinedList([1, 2]) * 2
        self.assertEqual(result, [1, 2, 1, 2])
        self.assertIs(type(result), list)

    @make_dynamo_test
    def test_user_defined_list_repeat_reflected(self):
        # [FUNCTION DOC] Dynamo test: the reflected form (integer times a user list subclass
        # instance) gives a plain list with repeated elements.
        result = 2 * UserDefinedList([5])
        self.assertEqual(result, [5, 5])
        self.assertIs(type(result), list)

    @make_dynamo_test
    def test_user_defined_tuple_repeat(self):
        # [FUNCTION DOC] Dynamo test: multiplying a user tuple subclass instance by two gives a
        # plain tuple with repeated elements.
        result = UserDefinedTuple([1, 2]) * 2
        self.assertEqual(result, (1, 2, 1, 2))
        self.assertIs(type(result), tuple)

    @make_dynamo_test
    def test_user_defined_list_inplace_repeat(self):
        # [FUNCTION DOC] Dynamo test: in-place multiplication of a user list subclass instance
        # mutates it, so it keeps its subclass type and object identity and holds the element
        # three times.
        # in-place repeat mutates the object: type and identity preserved
        a = UserDefinedList([1])
        b = a
        a *= 3
        self.assertEqual(list(a), [1, 1, 1])
        self.assertIs(type(a), UserDefinedList)
        self.assertIs(a, b)


# ---------------------------------------------------------------------------
# sq_ass_item / mp_ass_subscript on sequences (__setitem__)
# ---------------------------------------------------------------------------


class _WithSetitem:
    # [CLASS DOC] HELPER: Sequence-shaped class that stores a container in the data attribute
    # and forwards item reads and item assignment to it; used as a parametrized container type
    # for assignment tests.
    """Sequence-shaped class with explicit __setitem__ (int keys)."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given container as the data attribute (no copy).
        self.data = data

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Returns the item of the wrapped container for the key.
        return self.data[key]

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Assigns the value to the key of the wrapped container.
        self.data[key] = value


class _MutableSequence:
    # [CLASS DOC] HELPER: Custom mutable sequence that stores a list copy of the input in the
    # data attribute and supports item read, item assignment and length; used as a parametrized
    # container type.
    """Custom sequence with __setitem__ via sq_ass_item."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores a new list built from the iterable as the data
        # attribute.
        self.data = list(data)

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Returns the item of the data list for the key.
        return self.data[key]

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Assigns the value to the key in the data list.
        self.data[key] = value

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of stored items.
        return len(self.data)


# (label, factory) — factory builds an instance from a list.
_SEQUENCE_TYPES = [
    ("list", list),
    ("deque", collections.deque),
    ("WithSetitem", _WithSetitem),
    ("MutableSequence", _MutableSequence),
]


class TestSqAssItem(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for item and slice assignment, and later for deletion, on sequences:
    # parametrized container types, list slices (basic, stepped, extending, shrinking,
    # inserting, full), boolean and symbolic keys, list subclasses, error cases, and visibility
    # of mutation outside the compiled code. Setup enables the trace-unittest and trace-load-
    # build-class config options; teardown restores both.
    """All sequence __setitem__ tests in one class."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setup, saves the current values of the two
        # config options and turns both on.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the parent teardown, then restores both saved config
        # values.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # -- parameterized over sequence container type --

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_basic(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test over list, deque and the two helper classes:
        # builds a container from three numbers, assigns a new value at index zero and checks it
        # is read back.
        c = factory([1, 2, 3])
        c[0] = 100
        self.assertEqual(c[0], 100)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_multiple(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test: assigns two different indexes of a container
        # and checks both new values read back.
        c = factory([1, 2, 3])
        c[0] = 100
        c[1] = 200
        self.assertEqual(c[0], 100)
        self.assertEqual(c[1], 200)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_replaces_value(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test: overwrites the first item of a container with
        # a new value and reads it back.
        c = factory([1, 2, 3])
        c[0] = 999
        self.assertEqual(c[0], 999)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_negative_index(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test: assigns with a negative index (last item) and
        # reads it back with the same index.
        c = factory([1, 2, 3])
        c[-1] = 777
        self.assertEqual(c[-1], 777)

    # -- list slice (mp_ass_subscript_impl slice branch) --

    @make_dynamo_test
    def test_slice_basic(self):
        # [FUNCTION DOC] Dynamo test: assigning a two item list to a slice of two positions in a
        # list replaces those positions.
        lst = [1, 2, 3, 4, 5]
        lst[1:3] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 4, 5])

    @make_dynamo_test
    def test_slice_step(self):
        # [FUNCTION DOC] Dynamo test: assigning three items to every second position of a five
        # item list replaces those positions.
        lst = [1, 2, 3, 4, 5]
        lst[::2] = [10, 30, 50]
        self.assertEqual(lst, [10, 2, 30, 4, 50])

    @make_dynamo_test
    def test_slice_extending(self):
        # [FUNCTION DOC] Dynamo test: assigning to a slice that starts past the end of a list
        # appends the items.
        lst = [1, 2, 3]
        lst[10:] = [4, 5]
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    @make_dynamo_test
    def test_slice_shrinking(self):
        # [FUNCTION DOC] Dynamo test: assigning a one item list to a three position slice
        # shrinks the list.
        lst = [1, 2, 3, 4, 5]
        lst[1:4] = [99]
        self.assertEqual(lst, [1, 99, 5])

    @make_dynamo_test
    def test_slice_empty_insert(self):
        # [FUNCTION DOC] Dynamo test: assigning to an empty slice in the middle of a list
        # inserts the items there.
        lst = [1, 2, 3]
        lst[1:1] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 2, 3])

    @make_dynamo_test
    def test_slice_full_replace(self):
        # [FUNCTION DOC] Dynamo test: assigning to a full slice replaces the whole content, even
        # with a different length.
        lst = [1, 2, 3]
        lst[:] = [10, 20, 30, 40]
        self.assertEqual(lst, [10, 20, 30, 40])

    @make_dynamo_test
    def test_slice_negative(self):
        # [FUNCTION DOC] Dynamo test: assigning to a slice with a negative start replaces the
        # last two items.
        lst = [1, 2, 3, 4, 5]
        lst[-2:] = [99, 100]
        self.assertEqual(lst, [1, 2, 3, 99, 100])

    # -- nb_index path (bool key) --

    @make_dynamo_test
    def test_nb_index_bool_key(self):
        # [FUNCTION DOC] Dynamo test: using the boolean true as a list index (converted through
        # the index protocol) assigns to position one.
        lst = [1, 2, 3]
        lst[True] = 99
        self.assertEqual(lst[1], 99)

    # -- SymNode key --

    @make_dynamo_test
    def test_symnode_list_key(self):
        # [FUNCTION DOC] Dynamo test: a list index computed from a tensor shape dimension
        # (possibly symbolic) minus one assigns to the last position of a five item list.
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 1] = 99
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_symnode_list_key_zero(self):
        # [FUNCTION DOC] Dynamo test: a list index computed as the shape dimension minus five
        # (zero) assigns to the first position.
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 5] = 99
        self.assertEqual(lst[0], 99)

    @make_dynamo_test
    def test_symnode_list_key_then_read(self):
        # [FUNCTION DOC] Dynamo test: stores through a shape-derived list index and then reads
        # back through the same index and through the literal position two.
        t = torch.randn(5)
        lst = [10, 20, 30, 40, 50]
        idx = t.shape[0] - 3
        lst[idx] = 999
        self.assertEqual(lst[idx], 999)
        self.assertEqual(lst[2], 999)

    @make_dynamo_test
    def test_symnode_slice_stop(self):
        # [FUNCTION DOC] Dynamo test: a slice whose stop is a shape dimension (three) is
        # assigned two items, replacing positions one and two.
        t = torch.randn(3)
        lst = [1, 2, 3, 4, 5]
        lst[1 : t.shape[0]] = [88, 99]
        self.assertEqual(lst, [1, 88, 99, 4, 5])

    @make_dynamo_test
    def test_symnode_slice_start_stop(self):
        # [FUNCTION DOC] Dynamo test: a slice whose start and stop are both derived from a shape
        # dimension replaces three items of a five item list.
        t = torch.randn(4)
        lst = [1, 2, 3, 4, 5]
        lst[t.shape[0] - 3 : t.shape[0]] = [77, 88, 99]
        self.assertEqual(lst, [1, 77, 88, 99, 5])

    @make_dynamo_test
    def test_symnode_deque_key(self):
        # [FUNCTION DOC] Dynamo test: a deque item is assigned through a shape-derived index and
        # read back at the last position.
        t = torch.randn(5)
        d = collections.deque([0, 0, 0, 0, 0])
        d[t.shape[0] - 1] = 99
        self.assertEqual(d[4], 99)

    @make_dynamo_test
    def test_symnode_mutable_sequence_key(self):
        # [FUNCTION DOC] Dynamo test: the custom mutable sequence helper is assigned through a
        # shape-derived index minus two and read back at position three.
        t = torch.randn(5)
        seq = _MutableSequence([0, 0, 0, 0, 0])
        seq[t.shape[0] - 2] = 99
        self.assertEqual(seq[3], 99)

    # -- list subclass --

    @make_dynamo_test
    def test_subclass_list_inherited(self):
        # [FUNCTION DOC] Dynamo test: a locally defined list subclass supports item assignment
        # and slice assignment inherited from list, with the expected resulting contents.
        class L(list):
            # [CLASS DOC] HELPER: Empty list subclass defined inside the test.
            pass

        lst = L([1, 2, 3])
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(list(lst), [99, 2, 3])

        # slice
        lst2 = L([1, 2, 3, 4, 5])
        lst2[1:3] = [88, 99]
        self.assertEqual(list(lst2), [1, 88, 99, 4, 5])

        # negative
        lst3 = L([1, 2, 3])
        lst3[-1] = 99
        self.assertEqual(lst3[-1], 99)

    @make_dynamo_test
    def test_subclass_list_extra_state(self):
        # [FUNCTION DOC] Dynamo test: a local list subclass counts item assignments in an extra
        # attribute. After two index assignments the counter is two and values are stored; a
        # second instance with one assignment shows a counter of one, so the overridden
        # assignment ran once per store.
        class L(list):
            # [CLASS DOC] HELPER: List subclass with a write_count attribute (initialized at
            # construction) that is incremented by an overridden item assignment before
            # delegating to the parent.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Runs the parent initializer with the given arguments
                # and sets the write counter to zero.
                super().__init__(*args)
                self.write_count = 0

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Increments the write counter, then performs the parent
                # item assignment.
                self.write_count += 1
                super().__setitem__(key, value)

        lst = L([0, 0, 0])
        lst[0] = 5
        lst[1] = 6
        self.assertEqual(lst[0], 5)
        self.assertEqual(lst[1], 6)
        self.assertEqual(lst.write_count, 2)

        # replace existing
        lst2 = L([10, 20, 30])
        lst2[1] = 99
        self.assertEqual(lst2[1], 99)
        self.assertEqual(lst2.write_count, 1)

    @make_dynamo_test
    def test_subclass_list_slice_override(self):
        # [FUNCTION DOC] Dynamo test: a local list subclass whose item assignment adds one
        # hundred to every value when the key is a slice; assigning ten and twenty to a slice
        # yields one hundred ten and one hundred twenty in the list.
        class L(list):
            # [CLASS DOC] HELPER: List subclass whose item assignment shifts slice values by one
            # hundred and forwards other keys unchanged.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: If the key is a slice, assigns a new list in which each
                # value is increased by one hundred; otherwise assigns the value as given.
                # Delegates to the parent list.
                if isinstance(key, slice):
                    super().__setitem__(key, [v + 100 for v in value])
                else:
                    super().__setitem__(key, value)

        lst = L([1, 2, 3, 4, 5])
        lst[1:3] = [10, 20]
        self.assertEqual(list(lst), [1, 110, 120, 4, 5])

    @make_dynamo_test
    def test_subclass_list_overriding(self):
        # [FUNCTION DOC] Dynamo test: a local list subclass that overrides item assignment to
        # add one thousand; assigning five to the first position reads back as one thousand
        # five.
        class L(list):
            # [CLASS DOC] HELPER: List subclass whose item assignment stores the value plus one
            # thousand.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Assigns the value increased by one thousand through the
                # parent list.
                super().__setitem__(key, value + 1000)

        lst = L([0, 0, 0])
        lst[0] = 5
        self.assertEqual(lst[0], 1005)

    # -- mutation visibility --

    def test_mutation_outer_list_persists(self):
        # [FUNCTION DOC] Plain test: a compiled full-graph function assigns to the first
        # position of a closed-over list; after calling it, the outer list shows the mutation.
        outer = [1, 2, 3]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function; assigns ninety-nine to the first item of
            # the outer list and returns x plus one.
            outer[0] = 99
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, [99, 2, 3])

    @make_dynamo_test
    def test_mutation_setitem_then_read(self):
        # [FUNCTION DOC] Dynamo test: assigns to a list position and reads it back into a
        # variable.
        lst = [1, 2, 3]
        lst[0] = 99
        v = lst[0]
        self.assertEqual(v, 99)

    @make_dynamo_test
    def test_mutation_list_of_dicts(self):
        # [FUNCTION DOC] Dynamo test: mutating a dictionary inside a list through the list index
        # is visible when read back.
        lst = [{"a": 1}, {"b": 2}]
        lst[0]["a"] = 100
        self.assertEqual(lst[0]["a"], 100)

    @make_dynamo_test
    def test_mutation_multiple_sequential(self):
        # [FUNCTION DOC] Dynamo test: five consecutive index assignments fill a list of zeros
        # with one to five.
        lst = [0, 0, 0, 0, 0]
        lst[0] = 1
        lst[1] = 2
        lst[2] = 3
        lst[3] = 4
        lst[4] = 5
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    # -- misc index handling --

    @make_dynamo_test
    def test_list_int_var_index(self):
        # [FUNCTION DOC] Dynamo test: assigns through a variable holding an integer index and
        # checks the value at that position.
        lst = [1, 2, 3]
        idx = 1
        lst[idx] = 42
        self.assertEqual(lst[1], 42)

    @make_dynamo_test
    def test_list_negative_index_middle(self):
        # [FUNCTION DOC] Dynamo test: a negative index of minus two assigns the fourth item,
        # checked with both negative and positive indexes.
        lst = [1, 2, 3, 4, 5]
        lst[-2] = 88
        self.assertEqual(lst[-2], 88)
        self.assertEqual(lst[3], 88)

    @make_dynamo_test
    def test_list_negative_index_last(self):
        # [FUNCTION DOC] Dynamo test: index minus one assigns the last item, checked with both
        # negative and positive indexes.
        lst = [1, 2, 3, 4, 5]
        lst[-1] = 99
        self.assertEqual(lst[-1], 99)
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_mutable_sequence_negative_index(self):
        # [FUNCTION DOC] Dynamo test: the custom mutable sequence accepts a negative index
        # assignment, observed with a negative and a positive index.
        seq = _MutableSequence([1, 2, 3, 4, 5])
        seq[-1] = 99
        self.assertEqual(seq[-1], 99)
        self.assertEqual(seq[4], 99)

    # -- errors --

    @make_dynamo_test
    def test_error_list_out_of_bounds(self):
        # [FUNCTION DOC] Dynamo test: assigning to a list index past the end raises an index
        # error.
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[10] = 100

    @make_dynamo_test
    def test_error_list_negative_oob(self):
        # [FUNCTION DOC] Dynamo test: assigning to a very negative list index raises an index
        # error.
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[-100] = 1

    @make_dynamo_test
    def test_error_list_string_key(self):
        # [FUNCTION DOC] Dynamo test: assigning with a string key to a list raises a type error.
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst["a"] = 1

    @make_dynamo_test
    def test_error_list_float_key(self):
        # [FUNCTION DOC] Dynamo test: assigning with a float key to a list raises a type error.
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst[2.5] = 1

    @make_dynamo_test
    def test_error_tuple_setitem(self):
        # [FUNCTION DOC] Dynamo test: item assignment on a tuple raises a type error.
        t = (1, 2, 3)
        with self.assertRaises(TypeError):
            t[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_error_str_setitem(self):
        # [FUNCTION DOC] Dynamo test: item assignment on a string raises a type error.
        s = "abc"
        with self.assertRaises(TypeError):
            s[0] = "x"  # type: ignore[index]

    @make_dynamo_test
    def test_error_bytes_setitem(self):
        # [FUNCTION DOC] Dynamo test: item assignment on a bytes object raises a type error.
        b = b"abc"
        with self.assertRaises(TypeError):
            b[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_error_frozenset_setitem(self):
        # [FUNCTION DOC] Dynamo test: item assignment on a frozenset raises a type error.
        fs = frozenset([1, 2, 3])
        with self.assertRaises(TypeError):
            fs[0] = 1  # type: ignore[index]

    # -- slice assignment errors --

    @make_dynamo_test
    def test_error_slice_step_zero(self):
        # [FUNCTION DOC] Dynamo test: assigning to a slice with step zero raises a value error
        # whose message says the slice step cannot be zero.
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "slice step cannot be zero"):
            lst[::0] = [10, 20]

    @make_dynamo_test
    def test_error_extended_slice_size_mismatch(self):
        # [FUNCTION DOC] Dynamo test: assigning two items to a stepped slice that selects three
        # positions raises a value error mentioning the sequence size and the extended slice.
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(
            ValueError, "attempt to assign sequence of size .* to extended slice"
        ):
            lst[::2] = [10, 20]

    @make_dynamo_test
    def test_error_slice_non_iterable_simple(self):
        # [FUNCTION DOC] Dynamo test: assigning an integer to a simple slice raises a type
        # error; the accepted message depends on the Python version (older versions say an
        # iterable is required, newer ones mention an extended slice).
        lst = [1, 2, 3]
        # CPython 3.12.5+ reports "must assign iterable to extended slice" even
        # for simple slices; earlier versions say "can only assign an iterable".
        # See https://github.com/pytorch/pytorch/issues/187774.
        with self.assertRaisesRegex(
            TypeError,
            "must assign iterable to extended slice|can only assign an iterable",
        ):
            lst[1:2] = 99  # type: ignore[assignment]

    @make_dynamo_test
    def test_error_slice_non_iterable_extended(self):
        # [FUNCTION DOC] Dynamo test: assigning an integer to an extended (stepped) slice raises
        # a type error whose message says an iterable must be assigned to an extended slice.
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(
            TypeError, "must assign iterable to extended slice"
        ):
            lst[::2] = 99  # type: ignore[assignment]

    # -- slice assignment: alternative RHS types --

    @make_dynamo_test
    def test_slice_assign_tuple_rhs(self):
        # [FUNCTION DOC] Dynamo test: a tuple is accepted as the right side of a slice
        # assignment.
        lst = [1, 2, 3, 4, 5]
        lst[1:3] = (10, 20)
        self.assertEqual(lst, [1, 10, 20, 4, 5])

    @make_dynamo_test
    def test_slice_assign_string_rhs(self):
        # [FUNCTION DOC] Dynamo test: assigning the string ab to a one position slice inserts
        # each character as a separate item.
        lst = [1, 2, 3]
        lst[1:2] = "ab"
        self.assertEqual(lst, [1, "a", "b", 3])

    @make_dynamo_test
    def test_slice_assign_negative_step(self):
        # [FUNCTION DOC] Dynamo test: assigning five items to a full slice with step minus one
        # writes them in reverse position order, giving the reversed list.
        lst = [1, 2, 3, 4, 5]
        lst[::-1] = [50, 40, 30, 20, 10]
        self.assertEqual(lst, [10, 20, 30, 40, 50])

    @make_dynamo_test
    def test_slice_assign_empty_to_simple_slice(self):
        # [FUNCTION DOC] Dynamo test: assigning an empty list to a middle slice deletes those
        # positions.
        lst = [1, 2, 3, 4, 5]
        lst[1:4] = []
        self.assertEqual(lst, [1, 5])

    def test_slice_mutation_visible_outside(self):
        # [FUNCTION DOC] Plain test: slice assignment on a closed-over list inside a compiled
        # full-graph function is visible on the outer list afterwards, checking that the
        # mutation is recorded as a side effect on the slice path.
        # Outer list (captured by closure) mutated via slice assignment inside
        # a compiled function. Verifies side_effects.mutation is registered on
        # the slice path of mp_ass_subscript_impl.
        outer = [1, 2, 3, 4, 5]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function; assigns ten and twenty to positions one
            # and two of the outer list and returns x plus one.
            outer[1:3] = [10, 20]
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, [1, 10, 20, 4, 5])

    # -- __delitem__ (NULL value via mp_ass_subscript / sq_ass_item) --
    # Only list/deque among the parametrized types support __delitem__.

    @parametrize(
        "label,factory",
        [("list", list), ("deque", collections.deque)],
    )
    @make_dynamo_test
    def test_delitem_basic(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test over list and deque: deleting the item at
        # index two from a five item container leaves the other four items.
        c = factory([1, 2, 3, 4, 5])
        del c[2]
        self.assertEqual(list(c), [1, 2, 4, 5])

    @parametrize(
        "label,factory",
        [("list", list), ("deque", collections.deque)],
    )
    @make_dynamo_test
    def test_delitem_negative(self, label, factory):
        # [FUNCTION DOC] Parametrized dynamo test over list and deque: deleting the last item
        # with a negative index leaves the first four.
        c = factory([1, 2, 3, 4, 5])
        del c[-1]
        self.assertEqual(list(c), [1, 2, 3, 4])

    @make_dynamo_test
    def test_delitem_list_first(self):
        # [FUNCTION DOC] Dynamo test: deleting the first item of a list.
        lst = [1, 2, 3]
        del lst[0]
        self.assertEqual(lst, [2, 3])

    @make_dynamo_test
    def test_delitem_list_slice(self):
        # [FUNCTION DOC] Dynamo test: deleting a slice of two positions from a list removes
        # them.
        lst = [1, 2, 3, 4, 5]
        del lst[1:3]
        self.assertEqual(lst, [1, 4, 5])

    @make_dynamo_test
    def test_delitem_list_slice_step(self):
        # [FUNCTION DOC] Dynamo test: deleting every second item through a stepped slice leaves
        # the others.
        lst = [1, 2, 3, 4, 5]
        del lst[::2]
        self.assertEqual(lst, [2, 4])

    @make_dynamo_test
    def test_delitem_list_full_slice(self):
        # [FUNCTION DOC] Dynamo test: deleting a full slice empties the list.
        lst = [1, 2, 3]
        del lst[:]
        self.assertEqual(lst, [])

    @make_dynamo_test
    def test_delitem_list_multiple(self):
        # [FUNCTION DOC] Dynamo test: deleting the first item twice removes two leading items.
        lst = [1, 2, 3, 4, 5]
        del lst[0]
        del lst[0]
        self.assertEqual(lst, [3, 4, 5])

    @make_dynamo_test
    def test_delitem_symnode_key(self):
        # [FUNCTION DOC] Dynamo test: deleting with an index computed from a tensor shape
        # dimension minus one removes the last item.
        t = torch.randn(5)
        lst = [10, 20, 30, 40, 50]
        del lst[t.shape[0] - 1]
        self.assertEqual(lst, [10, 20, 30, 40])

    @make_dynamo_test
    def test_delitem_error_oob(self):
        # [FUNCTION DOC] Dynamo test: deleting a list index past the end raises an index error.
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            del lst[10]

    @make_dynamo_test
    def test_delitem_error_string_key(self):
        # [FUNCTION DOC] Dynamo test: deleting from a list with a string key raises a type
        # error.
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            del lst["a"]

    @make_dynamo_test
    def test_delitem_error_tuple(self):
        # [FUNCTION DOC] Dynamo test: deleting an item of a tuple raises a type error.
        t = (1, 2, 3)
        with self.assertRaises(TypeError):
            del t[0]  # type: ignore[attr-defined]

    @make_dynamo_test
    def test_delitem_error_str(self):
        # [FUNCTION DOC] Dynamo test: deleting a character of a string raises a type error.
        s = "abc"
        with self.assertRaises(TypeError):
            del s[0]  # type: ignore[attr-defined]

    @make_dynamo_test
    def test_delitem_subclass_list_inherited(self):
        # [FUNCTION DOC] Dynamo test: a local plain list subclass supports deleting an item by
        # index through the inherited behavior.
        class L(list):
            # [CLASS DOC] HELPER: Empty list subclass defined inside the test.
            pass

        lst = L([1, 2, 3, 4])
        del lst[1]
        self.assertEqual(list(lst), [1, 3, 4])

    @make_dynamo_test
    def test_delitem_subclass_list_override(self):
        # [FUNCTION DOC] Dynamo test: a local list subclass counts deletions via an overridden
        # delete method; two deletions of the first item leave the last item and a count of two.
        class L(list):
            # [CLASS DOC] HELPER: List subclass with a del_count attribute incremented by its
            # overridden item deletion.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Runs the parent initializer with the given arguments
                # and sets the deletion counter to zero.
                super().__init__(*args)
                self.del_count = 0

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Increments the deletion counter, then performs the
                # parent item deletion for the key.
                self.del_count += 1
                super().__delitem__(key)

        lst = L([1, 2, 3])
        del lst[0]
        del lst[0]
        self.assertEqual(list(lst), [3])
        self.assertEqual(lst.del_count, 2)


instantiate_parametrized_tests(TestSqAssItem)


class _IndexObj:
    # [CLASS DOC] HELPER: Object usable as an integer index; stores an integer converted from
    # its constructor argument in the n attribute.
    def __init__(self, n):
        # [FUNCTION DOC] HELPER: Converts the argument to an integer and stores it as n.
        self.n = int(n)

    def __index__(self):
        # [FUNCTION DOC] HELPER: Returns the stored integer for the index protocol.
        return self.n


class TestRangeUserIndex(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that range construction and range slicing apply the index protocol to
    # their arguments and slice members, including error propagation and the type error for a
    # non-integer index result. All tests compile with the eager backend and full graph.
    # range() and range subscript apply __index__ (PyNumber_Index) to their
    # arguments and slice members.

    def test_range_args_with_index(self):
        # [FUNCTION DOC] Compiles a function building a list from a range whose start, stop and
        # step are index-protocol objects (two, six, two); asserts the list holds two and four.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns the list from a range made with
            # start two, stop six and step two index objects.
            return list(range(_IndexObj(2), _IndexObj(6), _IndexObj(2)))

        self.assertEqual(fn(), [2, 4])

    def test_range_slice_with_index(self):
        # [FUNCTION DOC] Compiles a function that slices range ten with an index-protocol object
        # as the stop (five); asserts the result equals the range of five.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns the first five elements of range
            # ten, with the stop given as an index object.
            return range(10)[: _IndexObj(5)]

        self.assertEqual(fn(), range(5))

    def test_range_index_raises(self):
        # [FUNCTION DOC] Checks that an error raised by an index method propagates out of range
        # construction in compiled code and can be caught by the compiled function. Asserts the
        # function reports the runtime error branch.
        class IX:
            # [CLASS DOC] HELPER: Class whose index method always raises a runtime error.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Always raises a runtime error with the message boom.
                raise RuntimeError("boom")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; builds a range from the failing index
            # object inside a try block and returns a label saying whether a runtime error
            # happened (it should, so the label is runtime_error).
            try:
                range(IX())
                return "no_error"
            except RuntimeError:
                return "runtime_error"

        self.assertEqual(fn(), "runtime_error")

    def test_range_index_non_int(self):
        # [FUNCTION DOC] Checks that an index method returning a string makes range construction
        # raise a type error in compiled code. Asserts the compiled function returns the type
        # error label.
        class IN:
            # [CLASS DOC] HELPER: Class whose index method returns a non-integer value.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns a string instead of an integer, which is
                # invalid for the index protocol.
                return "not a number"  # noqa: PLE0305

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; builds a range from the bad index object
            # inside a try block and returns a label for a type error or for no error.
            try:
                range(IN())
                return "no_error"
            except TypeError:
                return "type_error"

        self.assertEqual(fn(), "type_error")


class TestRangeIteratorSetstate(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for setting the position of range iterators through the setstate method,
    # including reversed ranges, negative and too large positions (clamped to zero and the
    # length), and use after some elements were consumed (the argument then advances relative to
    # the current position). Compiled with eager backend and full graph.
    # range_iterator.__setstate__(k) sets the iterator index, clamped to
    # [0, len], mirroring CPython rangeiter_setstate.

    def test_setstate_resumes_from_index(self):
        # [FUNCTION DOC] Compiles a function that sets a fresh iterator over range ten to twenty
        # step two to position two and lists the rest; asserts fourteen, sixteen, eighteen.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; makes an iterator over range ten to
            # twenty step two, sets its state to two, and returns the remaining items as a list.
            it = iter(range(10, 20, 2))
            it.__setstate__(2)
            return list(it)

        self.assertEqual(fn(), [14, 16, 18])

    def test_setstate_on_reversed(self):
        # [FUNCTION DOC] Compiles a function that sets state three on a reversed iterator over
        # range ten to twenty step two; asserts the remaining items are twelve and ten.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; reverses range ten to twenty step two,
            # sets the iterator state to three, and returns the remaining items.
            it = reversed(range(10, 20, 2))
            it.__setstate__(3)
            return list(it)

        self.assertEqual(fn(), [12, 10])

    def test_setstate_negative_clamps_to_zero(self):
        # [FUNCTION DOC] Compiles a function that sets a hugely negative state on an iterator
        # over range five; asserts the position is clamped to zero so all five items remain.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; sets state minus one hundred on a range
            # five iterator and returns the remaining items.
            it = iter(range(5))
            it.__setstate__(-100)
            return list(it)

        self.assertEqual(fn(), [0, 1, 2, 3, 4])

    def test_setstate_overshoot_clamps_to_len(self):
        # [FUNCTION DOC] Compiles a function that sets a state larger than the length; asserts
        # the position is clamped to the end, so no items remain.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; sets state one hundred on a range five
            # iterator and returns the remaining items.
            it = iter(range(5))
            it.__setstate__(100)
            return list(it)

        self.assertEqual(fn(), [])

    def test_setstate_after_partial_consume(self):
        # [FUNCTION DOC] Checks that setting the state after two items were consumed advances
        # relative to the current position (matching CPython): range ten with state seven leaves
        # only nine.
        # __setstate__ is RELATIVE to the current advanced position: after two
        # next() calls (index 2), __setstate__(7) advances 7 more -> index 9,
        # so only [9] remains. Matches eager CPython.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; consumes two items of a range ten
            # iterator, sets state seven, and returns the remaining items.
            it = iter(range(10))
            next(it)
            next(it)
            it.__setstate__(7)
            return list(it)

        self.assertEqual(fn(), [9])

    def test_length_hint_tracks_index(self):
        # [FUNCTION DOC] Checks that the iterator length hint reflects a relative state change:
        # after consuming one item and setting state four on a range ten iterator the hint is
        # five.
        # After one next() (index 1), __setstate__(4) advances 4 -> index 5,
        # so 5 items remain. Matches eager CPython.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; consumes one item of a range ten
            # iterator, sets state four, and returns the iterator's length hint.
            it = iter(range(10))
            next(it)
            it.__setstate__(4)
            return it.__length_hint__()

        self.assertEqual(fn(), 5)


class TestRangeDynamicBounds(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that a captured range object still works when assume_static_by_default
    # is turned off, so its start, stop and step become symbolic integers; range math must
    # specialize them. Each test patches that config option.
    # With assume_static_by_default=False a captured range object's
    # start/stop/step are wrapped as symbolic ints; range math must specialize
    # them instead of assuming a plain python constant.
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_index_symbolic_bounds(self):
        # [FUNCTION DOC] With the config patched, indexes element zero of a captured range ten
        # inside a compiled full-graph function and asserts zero.
        keys = range(10)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns the first element of the
            # captured range.
            return keys[0]

        self.assertEqual(fn(), 0)

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_slice_symbolic_bounds(self):
        # [FUNCTION DOC] With the config patched, slices positions one to three of a captured
        # range from one to ten step two in compiled code and asserts the result equals the
        # eager slice.
        keys = range(1, 10, 2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns positions one and two of the
            # captured range.
            return keys[1:3]

        self.assertEqual(fn(), range(1, 10, 2)[1:3])

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_iter_symbolic_bounds(self):
        # [FUNCTION DOC] With the config patched, builds a list from a captured range five in
        # compiled code and asserts zero through four.
        keys = range(5)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns the captured range converted to
            # a list.
            return list(keys)

        self.assertEqual(fn(), [0, 1, 2, 3, 4])


class TestRangeContains(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for membership checks on ranges in compiled code: only exact integers
    # and booleans use the arithmetic shortcut, other operand types fall back to a linear scan
    # with equality comparisons, matching CPython.
    # range.__contains__ uses the arithmetic fast path only for exact int/bool
    # operands; everything else falls back to an __eq__ linear scan, matching
    # CPython range_contains / _PySequence_IterSearch.
    def test_non_int_members(self):
        # [FUNCTION DOC] Compiles a function that checks membership of various operands in small
        # ranges: a float with an integer value, true, a complex number, an object that always
        # compares equal, an integer subclass that always compares equal, an out-of-range
        # integer and a non-integral float. Asserts the first five are found and the last two
        # are not.
        class AlwaysEq:
            # [CLASS DOC] HELPER: Class whose equality always answers true and whose hash is
            # zero.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always returns true, whatever the other operand.
                return True

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Always returns zero.
                return 0

        class IntSubclassEq(int):
            # [CLASS DOC] HELPER: Integer subclass whose equality always answers true and whose
            # hash is zero.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always returns true, whatever the other operand.
                return True

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Always returns zero.
                return 0

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns a tuple of seven membership
            # results for the odd operands against small ranges.
            return (
                1.0 in range(3),
                True in range(3),
                (1 + 0j) in range(3),
                AlwaysEq() in range(3),
                IntSubclassEq(11) in range(10),
                5 in range(3),
                2.5 in range(3),
            )

        self.assertEqual(fn(), (True, True, True, True, True, False, False))

    def test_negative_step_and_strided(self):
        # [FUNCTION DOC] Compiles a function testing membership in a negative-step range and in
        # a stepped range of even numbers, for integer and float operands. Asserts the tuple:
        # negative one is in, one point zero is not in the negative range, two and two point
        # zero and one hundred point zero are in the even range, one is not.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Compiled function; returns a tuple of six membership
            # results for integers and floats in a negative-step range and in an even-number
            # range.
            return (
                -1 in range(0, -20, -1),
                1.0 in range(0, -20, -1),
                2 in range(0, 101, 2),
                1 in range(0, 101, 2),
                2.0 in range(0, 101, 2),
                100.0 in range(0, 101, 2),
            )

        self.assertEqual(fn(), (True, False, True, False, True, True))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
