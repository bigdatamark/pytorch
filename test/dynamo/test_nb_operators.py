# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the Python number-protocol slots nb_or, nb_subtract and nb_add.
# Three TestCase classes (TestNbOr, TestNbSub, TestNbAdd) exercise logical and bitwise or,
# subtraction, and addition on builtin values, containers, user-defined classes with forward and
# reflected methods, subclass priority rules, NotImplemented fallbacks and a Rat rational class
# borrowed from CPython test_binop. Most tests use make_dynamo_test so the body is traced by
# Dynamo; a few compile a nested function with fullgraph. Import time defines helper classes and
# parametrization tables only; instantiate_parametrized_tests runs for each class at module level.

"""Tests for nb_or (|, or), nb_subtract (-), nb_add (+) operators in PyTorch Dynamo."""

import collections
import functools

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet


class UserDefinedDict(dict):
    # [CLASS DOC] HELPER: Plain dict subclass with no overrides; used as a user-defined dict type
    # in the dict or combination tables.
    pass


class UserDefinedClassWithOr:
    # [CLASS DOC] HELPER: User-defined class wrapping a value. Its or and reflected or methods
    # build a new instance by or-ing the wrapped value with either the other instance's value or
    # the raw operand, and equality compares wrapped values (False for foreign types).
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped value on the instance attribute value.
        self.value = value

    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: returns a new instance holding the bitwise or of the
        # wrapped value with the other wrapped value (same class) or with the raw operand.
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(self.value | other.value)
        return UserDefinedClassWithOr(self.value | other)

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: returns a new instance with the raw or other value
        # on the left of the or and the wrapped value on the right.
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(other.value | self.value)
        return UserDefinedClassWithOr(other | self.value)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: true only when other is the same class and the wrapped
        # values are equal; otherwise False.
        if isinstance(other, UserDefinedClassWithOr):
            return self.value == other.value
        return False

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string containing the class name and wrapped
        # value.
        return f"UserDefinedClassWithOr({self.value})"


class LeftOrClass:
    # [CLASS DOC] HELPER: Class whose or methods only accept another instance of the same class and
    # otherwise return NotImplemented, so mixed-type or must fall through to the other operand.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped value on the instance attribute value.
        self.value = value

    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: new instance with the or of both values if other is
        # the same class, else NotImplemented.
        if isinstance(other, LeftOrClass):
            return LeftOrClass(self.value | other.value)
        return NotImplemented

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: new instance with the or of other's value and own
        # value if other is the same class, else NotImplemented.
        if isinstance(other, LeftOrClass):
            return LeftOrClass(other.value | self.value)
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: same class and equal wrapped values.
        return isinstance(other, LeftOrClass) and self.value == other.value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string with class name and wrapped value.
        return f"LeftOrClass({self.value})"


class RightOrClass:
    # [CLASS DOC] HELPER: Class whose wrapped value is a string. Forward or only works for the same
    # class and joins strings with a bar; reflected or only handles the left-hand class and returns
    # a formatted string. Used to test fallback from the left operand to the right operand's
    # reflected method.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped string on the instance attribute value.
        self.value = value

    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: joins the two wrapped strings with a bar into a new
        # instance for the same class, else NotImplemented.
        if isinstance(other, RightOrClass):
            return RightOrClass(self.value + "|" + other.value)
        return NotImplemented

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: when other is a LeftOrClass, returns a descriptive
        # string naming both operands; otherwise NotImplemented.
        if isinstance(other, LeftOrClass):
            return f"LeftOrClass({other.value})|RightOrClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: same class and equal wrapped strings.
        return isinstance(other, RightOrClass) and self.value == other.value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string with class name and wrapped value.
        return f"RightOrClass({self.value})"


class MyDict(dict):
    # [CLASS DOC] HELPER: Dict subclass whose or always declines by returning NotImplemented, so
    # Python must use the reflected method of the right operand.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: always returns NotImplemented.
        return NotImplemented


class MySet(set):
    # [CLASS DOC] HELPER: Set subclass whose or always declines by returning NotImplemented,
    # forcing use of the builtin reflected or of the right operand.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: always returns NotImplemented.
        return NotImplemented


class NonDict:
    # [CLASS DOC] HELPER: Plain class (not a dict) whose or returns a fixed wrong-result string;
    # used to check that dict reflected or rejects non-dict operands.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: returns the constant string wrong result, ignoring the
        # operand.
        return "wrong result"


class _IntSubWithOr(int):
    # [CLASS DOC] HELPER: Int subclass overriding both or and reflected or to return marker strings
    # that identify which method ran.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: returns the marker string for the subclass forward
        # method.
        return "_IntSubWithOr.__or__"

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: returns the marker string for the subclass reflected
        # method.
        return "_IntSubWithOr.__ror__"


class _BaseWithOr:
    # [CLASS DOC] HELPER: Base class whose forward and reflected or methods return marker strings
    # naming the base class.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: returns the base-class forward marker string.
        return "_BaseWithOr.__or__"

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: returns the base-class reflected marker string.
        return "_BaseWithOr.__ror__"


class _SubWithOr(_BaseWithOr):
    # [CLASS DOC] HELPER: Subclass of the marker base that overrides both or methods with its own
    # marker strings; used to test that a subclass's reflected method gets priority.
    def __or__(self, other):
        # [FUNCTION DOC] HELPER: Forward or: returns the subclass forward marker string.
        return "_SubWithOr.__or__"

    def __ror__(self, other):
        # [FUNCTION DOC] HELPER: Reflected or: returns the subclass reflected marker string.
        return "_SubWithOr.__ror__"


class _InheritedSub(_BaseWithOr):
    # [CLASS DOC] HELPER: Subclass of the marker base that overrides nothing; used to show that
    # inheriting reflected or gives no priority.
    pass


def _make_dict(d):
    # [FUNCTION DOC] HELPER: Builds a plain dict copy of the mapping d.
    return dict(d)


def _make_defaultdict(d):
    # [FUNCTION DOC] HELPER: Builds a defaultdict with int default factory from the mapping d.
    return collections.defaultdict(int, d)


def _make_ordered(d):
    # [FUNCTION DOC] HELPER: Builds an OrderedDict from the items of d, preserving iteration order.
    return collections.OrderedDict(d.items())


def _make_userdict(d):
    # [FUNCTION DOC] HELPER: Builds a UserDefinedDict (plain dict subclass) from the mapping d.
    return UserDefinedDict(d)


_DICT_COMBOS = {
    "dict_dict": (_make_dict, _make_dict),
    "dict_defaultdict": (_make_dict, _make_defaultdict),
    "defaultdict_dict": (_make_defaultdict, _make_dict),
    "defaultdict_defaultdict": (_make_defaultdict, _make_defaultdict),
    "dict_ordereddict": (_make_dict, _make_ordered),
    "ordereddict_dict": (_make_ordered, _make_dict),
    "ordereddict_ordereddict": (_make_ordered, _make_ordered),
    "dict_userdict": (_make_dict, _make_userdict),
    "userdict_dict": (_make_userdict, _make_dict),
    "userdict_userdict": (_make_userdict, _make_userdict),
}

_INPLACE_COMBOS = {
    "dict": (dict, {"a": 1, "b": 2}, {"b": 20, "c": 3}, {"a": 1, "b": 20, "c": 3}),
    "set": (set, {1, 2}, {2, 3}, {1, 2, 3}),
    "defaultdict": (
        functools.partial(collections.defaultdict, int),
        {"a": 1, "b": 2},
        {"b": 20, "c": 3},
        {"a": 1, "b": 20, "c": 3},
    ),
}


class TestNbOr(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the or operator under Dynamo: logical or short-circuit semantics,
    # bitwise or on ints, bools, sets, frozensets, OrderedSet and dict combinations, in-place or,
    # reflected or, unsupported list/tuple or, user-defined or, cross-type fallback, and subclass
    # reflected priority. Each test body runs through make_dynamo_test.
    def setUp(self):
        # [FUNCTION DOC] Enables the Dynamo unittest-tracing config flag, remembering the previous
        # value in an instance attribute so tearDown can restore it.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Restores the previous value of the unittest-tracing config flag saved in
        # setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Logical or ---

    @make_dynamo_test
    def test_or_with_booleans(self):
        # [FUNCTION DOC] Logical or on all combinations of two booleans returns the expected truth
        # value.
        self.assertEqual(True or True, True)  # noqa: SIM222
        self.assertEqual(True or False, True)  # noqa: SIM222
        self.assertEqual(False or True, True)  # noqa: SIM222
        self.assertEqual(False or False, False)  # noqa: RUF100

    @make_dynamo_test
    def test_or_with_integers(self):
        # [FUNCTION DOC] Logical or on ints returns the first truthy operand, or the last operand
        # when all are falsy.
        self.assertEqual(0 or 5, 5)  # noqa: SIM222
        self.assertEqual(5 or 0, 5)  # noqa: SIM222
        self.assertEqual(3 or 7, 3)  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_strings(self):
        # [FUNCTION DOC] Logical or on strings returns the first non-empty string.
        self.assertEqual("" or "hello", "hello")  # noqa: SIM222
        self.assertEqual("hello" or "world", "hello")  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_containers(self):
        # [FUNCTION DOC] Logical or on lists and None returns the first truthy operand (non-empty
        # list or non-None value).
        self.assertEqual([] or [1, 2], [1, 2])  # noqa: SIM222
        self.assertEqual([1, 2] or [], [1, 2])  # noqa: SIM222
        self.assertEqual(None or 5, 5)  # noqa: SIM222

    @make_dynamo_test
    def test_or_short_circuit(self):
        # [FUNCTION DOC] Checks short-circuit: when the left operand is truthy the right operand,
        # which would raise a division by zero, is never evaluated and the left value is the
        # result.
        x = 5
        result = x or (1 / 0)
        self.assertEqual(result, 5)

    @make_dynamo_test
    def test_or_chained(self):
        # [FUNCTION DOC] Chained logical or returns the first truthy operand in a chain of ints and
        # of bools.
        self.assertEqual(0 or 0 or 3 or 4, 3)  # noqa: SIM222
        self.assertEqual(False or False or True, True)  # noqa: SIM222

    # --- Bitwise | integers ---

    @make_dynamo_test
    def test_bitwise_or_integers(self):
        # [FUNCTION DOC] Bitwise or on non-negative ints gives the expected bit patterns, including
        # identity with zero.
        self.assertEqual(5 | 3, 7)  # 101 | 011 = 111
        self.assertEqual(12 | 10, 14)  # 1100 | 1010 = 1110
        self.assertEqual(5 | 0, 5)
        self.assertEqual(256 | 128, 384)

    @make_dynamo_test
    def test_bitwise_or_negative_integers(self):
        # [FUNCTION DOC] Bitwise or on negative ints follows two's complement semantics, yielding
        # minus one in the tested cases.
        self.assertEqual(-1 | 0, -1)
        self.assertEqual(5 | -1, -1)
        self.assertEqual(-2 | -3, -1)

    @make_dynamo_test
    def test_bitwise_or_integers_chained(self):
        # [FUNCTION DOC] Chained bitwise or of four powers of two accumulates to fifteen.
        self.assertEqual(1 | 2 | 4 | 8, 15)

    # --- Bitwise | booleans ---

    @make_dynamo_test
    def test_bitwise_or_bools(self):
        # [FUNCTION DOC] Bitwise or on bools returns the logical or as a bool.
        self.assertEqual(True | True, True)
        self.assertEqual(True | False, True)
        self.assertEqual(False | False, False)

    @make_dynamo_test
    def test_bitwise_or_int_and_bool(self):
        # [FUNCTION DOC] Bitwise or of an int and True follows int semantics (the bool acts as
        # one).
        self.assertEqual(5 | True, 5)
        self.assertEqual(0 | True, 1)

    # --- Bitwise | set ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ({1, 2}, {2, 3}, {1, 2, 3}),
            ({1}, {2}, {1, 2}),
            ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}),
        ],
    )
    @make_dynamo_test
    def test_set_union(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over operand pairs; checks set union via the or operator
        # equals the expected set.
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_set_union_with_empty(self):
        # [FUNCTION DOC] Union of a set with an empty set, in both operand orders, leaves the set
        # unchanged.
        self.assertEqual({1, 2} | set(), {1, 2})
        self.assertEqual(set() | {1, 2}, {1, 2})

    @make_dynamo_test
    def test_set_union_empty(self):
        # [FUNCTION DOC] Union of two empty sets is an empty set.
        self.assertEqual(set() | set(), set())

    @make_dynamo_test
    def test_set_union_chained(self):
        # [FUNCTION DOC] Chained set unions across three or more sets accumulate all elements.
        self.assertEqual({1} | {2} | {3}, {1, 2, 3})
        self.assertEqual({1, 2} | {2, 3} | {3, 4}, {1, 2, 3, 4})

    # --- Bitwise | frozenset ---

    @parametrize(
        "operand1,operand2,expected",
        [
            (frozenset({1, 2}), frozenset({2, 3}), frozenset({1, 2, 3})),
            (frozenset({1}), frozenset({2}), frozenset({1, 2})),
            (frozenset({1, 2, 3}), frozenset({1, 2, 3}), frozenset({1, 2, 3})),
        ],
    )
    @make_dynamo_test
    def test_frozenset_union(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over frozenset pairs; checks union via or equals the expected
        # frozenset.
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_frozenset_union_with_empty(self):
        # [FUNCTION DOC] Frozenset union with an empty frozenset in both orders returns the
        # nonempty frozenset.
        self.assertEqual(frozenset({1, 2}) | frozenset(), frozenset({1, 2}))
        self.assertEqual(frozenset() | frozenset({1, 2}), frozenset({1, 2}))

    @make_dynamo_test
    def test_frozenset_union_empty(self):
        # [FUNCTION DOC] Union of two empty frozensets is an empty frozenset.
        self.assertEqual(frozenset() | frozenset(), frozenset())

    @make_dynamo_test
    def test_frozenset_union_chained(self):
        # [FUNCTION DOC] Chained frozenset unions of three singletons produce the combined
        # frozenset.
        self.assertEqual(
            frozenset({1}) | frozenset({2}) | frozenset({3}),
            frozenset({1, 2, 3}),
        )

    # --- Bitwise | OrderedSet ---

    @make_dynamo_test
    def test_orderedset_union_basic(self):
        # [FUNCTION DOC] OrderedSet or with an overlapping second set produces the union.
        operand1 = OrderedSet([1, 2])
        operand2 = OrderedSet([2, 3])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_single(self):
        # [FUNCTION DOC] OrderedSet or of two disjoint singleton sets produces a two-element set.
        operand1 = OrderedSet([1])
        operand2 = OrderedSet([2])
        expected = OrderedSet([1, 2])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_duplicate(self):
        # [FUNCTION DOC] OrderedSet or of two identical sets equals the same set.
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([1, 2, 3])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_with_empty(self):
        # [FUNCTION DOC] OrderedSet or with an empty OrderedSet in both orders returns the nonempty
        # set.
        self.assertEqual(OrderedSet([1, 2]) | OrderedSet(), OrderedSet([1, 2]))
        self.assertEqual(OrderedSet() | OrderedSet([1, 2]), OrderedSet([1, 2]))

    @make_dynamo_test
    def test_orderedset_union_empty(self):
        # [FUNCTION DOC] OrderedSet or of two empty sets is empty.
        self.assertEqual(OrderedSet() | OrderedSet(), OrderedSet())

    @make_dynamo_test
    def test_orderedset_union_chained(self):
        # [FUNCTION DOC] Chained or of three single-element OrderedSets gives the combined set.
        self.assertEqual(
            OrderedSet([1]) | OrderedSet([2]) | OrderedSet([3]),
            OrderedSet([1, 2, 3]),
        )

    # --- Bitwise | dict combinations ---

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_operation(self, combo):
        # [FUNCTION DOC] Parametrized over the dict-type combination table; builds left and right
        # dicts with the combo's factory functions and checks that or merges them with right-hand
        # values winning on a shared key.
        left_fn, right_fn = _DICT_COMBOS[combo]
        left = left_fn({"a": 1, "b": 2})
        right = right_fn({"b": 20, "c": 3})
        self.assertEqual(left | right, {"a": 1, "b": 20, "c": 3})

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_empty(self, combo):
        # [FUNCTION DOC] Parametrized over dict-type combinations; or with an empty right-hand dict
        # returns the left contents unchanged.
        left_fn, right_fn = _DICT_COMBOS[combo]
        self.assertEqual(left_fn({"a": 1}) | right_fn({}), {"a": 1})

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_chained(self, combo):
        # [FUNCTION DOC] Parametrized over dict-type combinations; chained or of three dicts yields
        # a merged dict of all keys.
        left_fn, right_fn = _DICT_COMBOS[combo]
        d1 = left_fn({"a": 1})
        d2 = right_fn({"b": 2})
        d3 = left_fn({"c": 3})
        self.assertEqual(d1 | d2 | d3, {"a": 1, "b": 2, "c": 3})

    # --- Inplace |= ---

    @parametrize("combo", list(_INPLACE_COMBOS.keys()))
    @make_dynamo_test
    def test_inplace_or(self, combo):
        # [FUNCTION DOC] Parametrized over container kinds in the in-place table (dict, set,
        # defaultdict); builds both operands from the data in the table, applies in-place or to the
        # left one and compares with the expected result.
        container_type, data1, data2, expected = _INPLACE_COMBOS[combo]
        left = container_type(data1)
        right = container_type(data2)
        left |= right
        self.assertEqual(left, expected)

    # --- Reversed or (__ror__) ---

    @make_dynamo_test
    def test_reversed_or_with_integer(self):
        # [FUNCTION DOC] Int on the left with a user-defined class on the right: the int's or
        # declines, so the reflected method produces an instance holding the or result.
        obj = UserDefinedClassWithOr(3)
        result = 5 | obj
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_with_user_defined_object(self):
        # [FUNCTION DOC] Or between two instances of the user-defined or class gives an instance
        # holding the combined bits.
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        result = obj1 | obj2
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_chained(self):
        # [FUNCTION DOC] Chain starting with the int zero followed by three user-defined instances
        # accumulates the or of all wrapped values (seven).
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        result = 0 | obj1 | obj2 | obj3
        self.assertEqual(result.value, 7)

    @make_dynamo_test
    def test_dict_ror_valid(self):
        # [FUNCTION DOC] Dict subclass whose or declines, on the left of a plain dict: the result
        # falls back to the plain dict's reflected or and equals the right operand contents (right
        # values win).
        d = {"a": 1}
        myd = MyDict({"a": 2})
        result = myd | d
        self.assertEqual(result, {"a": 1})

    @make_dynamo_test
    def test_set_ror_valid(self):
        # [FUNCTION DOC] Set subclass whose or declines, on the left of a plain set: the result is
        # the union via the reflected path.
        s = {1, 2}
        mys = MySet({2, 3})
        result = mys | s
        self.assertEqual(result, {1, 2, 3})

    @make_dynamo_test
    def test_non_dict_or_invalid(self):
        # [FUNCTION DOC] Calls the reflected or of a plain dict directly with a non-dict operand
        # and checks it returns NotImplemented.
        d = {"a": 1}
        non_dict = NonDict()
        result = d.__ror__(non_dict)
        self.assertIs(result, NotImplemented)

    # --- Unsupported types ---

    @make_dynamo_test
    def test_list_or_list_raises_type_error(self):
        # [FUNCTION DOC] Or between two lists raises TypeError.
        with self.assertRaises(TypeError):
            [1, 2] | [3, 4]

    @make_dynamo_test
    def test_tuple_or_tuple_raises_type_error(self):
        # [FUNCTION DOC] Or between two tuples raises TypeError.
        with self.assertRaises(TypeError):
            (1, 2) | (3, 4)

    @make_dynamo_test
    def test_empty_list_or_list_raises_type_error(self):
        # [FUNCTION DOC] Or of an empty list and a nonempty list raises TypeError.
        with self.assertRaises(TypeError):
            [] | [1, 2]

    @make_dynamo_test
    def test_empty_tuple_or_tuple_raises_type_error(self):
        # [FUNCTION DOC] Or of an empty tuple and a nonempty tuple raises TypeError.
        with self.assertRaises(TypeError):
            () | (1, 2)

    # --- User-defined __or__ ---

    @make_dynamo_test
    def test_user_defined_or_basic(self):
        # [FUNCTION DOC] Or between two user-defined or instances gives an instance with the
        # combined value.
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        self.assertEqual(obj1 | obj2, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_with_integer(self):
        # [FUNCTION DOC] User-defined instance or a plain int uses the raw operand path and gives
        # the combined value.
        obj = UserDefinedClassWithOr(5)
        self.assertEqual(obj | 3, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_zero(self):
        # [FUNCTION DOC] User-defined instance or zero returns an equal instance.
        obj = UserDefinedClassWithOr(5)
        self.assertEqual(obj | 0, UserDefinedClassWithOr(5))

    @make_dynamo_test
    def test_user_defined_or_chained(self):
        # [FUNCTION DOC] Chained or of three user-defined instances gives the combined value.
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        self.assertEqual(obj1 | obj2 | obj3, UserDefinedClassWithOr(7))

    # --- Cross-type user-defined or ---

    @make_dynamo_test
    def test_left_or_left_uses_or(self):
        # [FUNCTION DOC] Two LeftOrClass instances use forward or and give the combined instance.
        a = LeftOrClass(5)
        b = LeftOrClass(3)
        self.assertEqual(a | b, LeftOrClass(7))

    @make_dynamo_test
    def test_right_or_right_uses_or(self):
        # [FUNCTION DOC] Two RightOrClass instances use forward or and join their strings with a
        # bar.
        a = RightOrClass("a")
        b = RightOrClass("b")
        self.assertEqual(a | b, RightOrClass("a|b"))

    @make_dynamo_test
    def test_left_or_right_falls_back_to_ror(self):
        # [FUNCTION DOC] LeftOrClass or RightOrClass: left declines, so the right operand's
        # reflected method returns the descriptive string.
        a = LeftOrClass(42)
        b = RightOrClass("x")
        self.assertEqual(a | b, "LeftOrClass(42)|RightOrClass(x)")

    @make_dynamo_test
    def test_right_or_left_raises(self):
        # [FUNCTION DOC] RightOrClass or LeftOrClass: both forward and reflected methods decline,
        # so a TypeError is raised.
        a = RightOrClass("x")
        b = LeftOrClass(42)
        with self.assertRaises(TypeError):
            a | b

    # --- Tensors ---

    @make_dynamo_test
    def test_tensor_to_bool(self):
        # [FUNCTION DOC] Converts a nonzero and a zero tensor to bool and asserts the truth values
        # (true and false respectively).
        t_nonzero = torch.tensor(1)
        t_zero = torch.tensor(0)
        self.assertTrue(bool(t_nonzero))
        self.assertFalse(bool(t_zero))

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Subclass of int overriding or: it handles both the forward case and,
        # taking priority as a subclass, the reflected case with an int on the left.
        self.assertEqual(_IntSubWithOr(1) | 1, "_IntSubWithOr.__or__")
        self.assertEqual(1 | _IntSubWithOr(1), "_IntSubWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline: plain class with forward and reflected or returns the base
        # markers when used with an int on either side.
        self.assertEqual(_BaseWithOr() | 1, "_BaseWithOr.__or__")
        self.assertEqual(1 | _BaseWithOr(), "_BaseWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        # [FUNCTION DOC] When the right operand is a subclass overriding the reflected method, that
        # override runs first even with a base-class instance on the left.
        self.assertEqual(_SubWithOr() | _BaseWithOr(), "_SubWithOr.__or__")
        self.assertEqual(_BaseWithOr() | _SubWithOr(), "_SubWithOr.__ror__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        # [FUNCTION DOC] Subclass inheriting the reflected method (not overriding it) gets no
        # priority: the base forward method is used in all mixed orders; also asserts the reflected
        # method is the same object as the base one.
        self.assertIs(_InheritedSub.__ror__, _BaseWithOr.__ror__)
        self.assertEqual(_InheritedSub() | 1, "_BaseWithOr.__or__")
        self.assertEqual(1 | _InheritedSub(), "_BaseWithOr.__ror__")
        self.assertEqual(_InheritedSub() | _BaseWithOr(), "_BaseWithOr.__or__")
        self.assertEqual(_BaseWithOr() | _InheritedSub(), "_BaseWithOr.__or__")


# --- Helper classes for sub tests ---

# Rat class from CPython test_binop.py for testing tp_as_number.tp_subtract


def _gcd(a, b):
    # [FUNCTION DOC] HELPER: Euclid greatest common divisor of two ints; loops while the first
    # argument is nonzero and returns the final second argument.
    """Greatest common divisor using Euclid's algorithm."""
    while a:
        a, b = b % a, a
    return b


def _isint(x):
    # [FUNCTION DOC] HELPER: Returns whether x is an instance of int.
    """Test whether an object is an instance of int."""
    return isinstance(x, int)


def _isnum(x):
    # [FUNCTION DOC] HELPER: Returns 1 if x is an instance of int, float or complex, else 0.
    """Test whether an object is an instance of a built-in numeric type."""
    for T in int, float, complex:
        if isinstance(x, T):
            return 1
    return 0


def _isRat(x):
    # [FUNCTION DOC] HELPER: Returns whether x is an instance of the Rat class.
    """Test whether an object is an instance of the Rat class."""
    return isinstance(x, Rat)


class Rat:
    # [CLASS DOC] Rational number class from CPython test_binop, stored as a normalized numerator
    # and denominator in two name-mangled slots. Supports arithmetic, floor division, divmod,
    # modulo and equality with Rats, ints and other numbers, and returns NotImplemented for
    # unsupported types. Exposes read-only num and den properties.
    """Rational number implemented as a normalized pair of ints."""

    __slots__ = ["__num", "__den"]

    def __init__(self, num=0, den=1):
        # [FUNCTION DOC] Builds a normalized Rat from integer num and den. Raises TypeError if
        # either is not an int and ZeroDivisionError if den is zero. Divides both by their gcd.
        """Constructor: Rat([num[, den]]).

        The arguments must be ints, and default to (0, 1)."""
        if not _isint(num):
            raise TypeError("Rat numerator must be int ({num})")
        if not _isint(den):
            raise TypeError("Rat denominator must be int ({den})")
        # But the zero is always on
        if den == 0:
            raise ZeroDivisionError("zero denominator")
        g = _gcd(den, num)
        self.__num = int(num // g)
        self.__den = int(den // g)

    def _get_num(self):
        # [FUNCTION DOC] HELPER: Accessor returning the stored numerator.
        """Accessor function for read-only 'num' attribute of Rat."""
        return self.__num

    num = property(_get_num, None)

    def _get_den(self):
        # [FUNCTION DOC] HELPER: Accessor returning the stored denominator.
        """Accessor function for read-only 'den' attribute of Rat."""
        return self.__den

    den = property(_get_den, None)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a constructor-style string with the numerator and
        # denominator.
        """Convert a Rat to a string resembling a Rat constructor call."""
        return f"Rat({self.__num}, {self.__den})"

    def __str__(self):
        # [FUNCTION DOC] HELPER: Returns the decimal string form of the float value.
        """Convert a Rat to a string resembling a decimal numeric value."""
        return str(float(self))

    def __float__(self):
        # [FUNCTION DOC] HELPER: Returns the numerator divided by the denominator as a float.
        """Convert a Rat to a float."""
        return self.__num * 1.0 / self.__den

    def __int__(self):
        # [FUNCTION DOC] Converts to int when the denominator is one; raises OverflowError if the
        # numerator is too large, and ValueError when the denominator is not one.
        """Convert a Rat to an int; self.den must be 1."""
        if self.__den == 1:
            try:
                return int(self.__num)
            except OverflowError:
                raise OverflowError(
                    f"{repr(self)} too large to convert to int"
                ) from None
        raise ValueError(f"can't convert {repr(self)} to int")

    def __add__(self, other):
        # [FUNCTION DOC] Adds a Rat and an int or Rat, returning a Rat with cross-multiplied
        # numerator; for other numeric types returns a float sum; otherwise NotImplemented.
        """Add two Rats, or a Rat and a number."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                self.__num * other.__den + other.__num * self.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return float(self) + other
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        # [FUNCTION DOC] Subtracts an int or Rat from self returning a Rat; for float or complex
        # returns the float difference; otherwise NotImplemented.
        """Subtract two Rats, or a Rat and a number."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                self.__num * other.__den - other.__num * self.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return float(self) - other
        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC] Reflected subtraction: computes other minus self, returning a Rat for int
        # or Rat operands, a float for other numbers, else NotImplemented.
        """Subtract two Rats, or a Rat and a number (reversed args)."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                other.__num * self.__den - self.__num * other.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return other - float(self)
        return NotImplemented

    def __mul__(self, other):
        # [FUNCTION DOC] Multiplies by a Rat or int returning a Rat, or by another number returning
        # a float; otherwise NotImplemented.
        """Multiply two Rats, or a Rat and a number."""
        if _isRat(other):
            return Rat(self.__num * other.__num, self.__den * other.__den)
        if _isint(other):
            return Rat(self.__num * other, self.__den)
        if _isnum(other):
            return float(self) * other
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        # [FUNCTION DOC] True division by a Rat or int returning a Rat; by other numbers returns a
        # float; otherwise NotImplemented.
        """Divide two Rats, or a Rat and a number."""
        if _isRat(other):
            return Rat(self.__num * other.__den, self.__den * other.__num)
        if _isint(other):
            return Rat(self.__num, self.__den * other)
        if _isnum(other):
            return float(self) / other
        return NotImplemented

    def __rtruediv__(self, other):
        # [FUNCTION DOC] Reflected true division of other by self with the same type rules as the
        # forward form; otherwise NotImplemented.
        """Divide two Rats, or a Rat and a number (reversed args)."""
        if _isRat(other):
            return Rat(other.__num * self.__den, other.__den * self.__num)
        if _isint(other):
            return Rat(other * self.__den, self.__num)
        if _isnum(other):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other):
        # [FUNCTION DOC] Floor division by an int or Rat: computes the exact quotient as a Rat and
        # returns the integer floor of its numerator over denominator; other types give
        # NotImplemented.
        """Divide two Rats, returning the floored result."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        x = self / other
        return x.__num // x.__den

    def __rfloordiv__(self, other):
        # [FUNCTION DOC] Reflected floor division: divides other by self and returns the integer
        # floor of the quotient.
        """Divide two Rats, returning the floored result (reversed args)."""
        x = other / self
        return x.__num // x.__den

    def __divmod__(self, other):
        # [FUNCTION DOC] Returns a pair of floor quotient and remainder for an int or Rat divisor;
        # other types give NotImplemented.
        """Divide two Rats, returning quotient and remainder."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        x = self // other
        return (x, self - other * x)

    def __rdivmod__(self, other):
        # [FUNCTION DOC] Reflected divmod for an int or Rat dividend delegating to the builtin
        # divmod with other first; other types give NotImplemented.
        """Divide two Rats, returning quotient and remainder (reversed args)."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        return divmod(other, self)

    def __mod__(self, other):
        # [FUNCTION DOC] Modulo defined as the remainder part of the divmod pair.
        """Take one Rat modulo another."""
        return divmod(self, other)[1]

    def __rmod__(self, other):
        # [FUNCTION DOC] Reflected modulo defined as the remainder part of divmod with other first.
        """Take one Rat modulo another (reversed args)."""
        return divmod(other, self)[1]

    def __eq__(self, other):
        # [FUNCTION DOC] Equality against an int (needs denominator one), a Rat (equal parts), or
        # another number (float comparison); otherwise NotImplemented.
        """Compare two Rats for equality."""
        if _isint(other):
            return self.__den == 1 and self.__num == other
        if _isRat(other):
            return self.__num == other.__num and self.__den == other.__den
        if _isnum(other):
            return float(self) == other
        return NotImplemented


class UserDefinedClassWithSub:
    # [CLASS DOC] HELPER: User-defined class wrapping a value whose subtraction and reflected
    # subtraction return new instances with the difference; equality compares values.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped value on the instance attribute value.
        self.value = value

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: new instance with own value minus the other's
        # value or raw operand.
        if isinstance(other, UserDefinedClassWithSub):
            return UserDefinedClassWithSub(self.value - other.value)
        return UserDefinedClassWithSub(self.value - other)

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: new instance with the other value or raw
        # operand minus own value.
        if isinstance(other, UserDefinedClassWithSub):
            return UserDefinedClassWithSub(other.value - self.value)
        return UserDefinedClassWithSub(other - self.value)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: same class and equal wrapped values, else False.
        if isinstance(other, UserDefinedClassWithSub):
            return self.value == other.value
        return False

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string with class name and wrapped value.
        return f"UserDefinedClassWithSub({self.value})"


class LeftSubClass:
    # [CLASS DOC] HELPER: Class whose subtraction methods accept only the same class and otherwise
    # return NotImplemented.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped value on the instance attribute value.
        self.value = value

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: same-class operand gives a new instance with
        # the difference, else NotImplemented.
        if isinstance(other, LeftSubClass):
            return LeftSubClass(self.value - other.value)
        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: same-class operand gives a new instance
        # with other minus own, else NotImplemented.
        if isinstance(other, LeftSubClass):
            return LeftSubClass(other.value - self.value)
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: same class and equal values.
        return isinstance(other, LeftSubClass) and self.value == other.value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string with class name and wrapped value.
        return f"LeftSubClass({self.value})"


class RightSubClass:
    # [CLASS DOC] HELPER: Class with forward subtraction only for its own class and a reflected
    # subtraction that handles only the LeftSubClass, returning a descriptive string.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the wrapped value on the instance attribute value.
        self.value = value

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: same-class operand gives a new instance with
        # the difference, else NotImplemented.
        if isinstance(other, RightSubClass):
            return RightSubClass(self.value - other.value)
        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: for a LeftSubClass operand returns a string
        # naming both operands, else NotImplemented.
        if isinstance(other, LeftSubClass):
            return f"LeftSubClass({other.value})-RightSubClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality: same class and equal values.
        return isinstance(other, RightSubClass) and self.value == other.value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string with class name and wrapped value.
        return f"RightSubClass({self.value})"


class _IntSubWithSub(int):
    # [CLASS DOC] HELPER: Int subclass overriding subtraction and reflected subtraction to return
    # marker strings.
    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: returns the subclass forward marker string.
        return "_IntSubWithSub.__sub__"

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: returns the subclass reflected marker
        # string.
        return "_IntSubWithSub.__rsub__"


class _BaseWithSub:
    # [CLASS DOC] HELPER: Base class with forward and reflected subtraction returning base-class
    # marker strings.
    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: returns the base forward marker string.
        return "_BaseWithSub.__sub__"

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: returns the base reflected marker string.
        return "_BaseWithSub.__rsub__"


class _SubWithSub(_BaseWithSub):
    # [CLASS DOC] HELPER: Subclass of the marker base overriding both subtraction methods with
    # subclass marker strings.
    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: returns the subclass forward marker string.
        return "_SubWithSub.__sub__"

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: returns the subclass reflected marker
        # string.
        return "_SubWithSub.__rsub__"


class _InheritedSubSub(_BaseWithSub):
    # [CLASS DOC] HELPER: Subclass of the marker base overriding nothing; used for the no-priority
    # check.
    pass


# --- Helper classes for NotImplemented tests ---


class _SubNotImplemented:
    # [CLASS DOC] HELPER: Class whose forward subtraction returns NotImplemented, so subtracting it
    # from itself raises TypeError.
    """Class where __sub__ returns NotImplemented"""

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: always returns NotImplemented.
        return NotImplemented


class _RSubNotImplemented:
    # [CLASS DOC] HELPER: Class whose forward and reflected subtraction both return NotImplemented.
    """Class where __sub__ exists but __rsub__ returns NotImplemented"""

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: always returns NotImplemented.
        return NotImplemented

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: always returns NotImplemented.
        return NotImplemented


class _SubReturnsMarker:
    # [CLASS DOC] HELPER: Class whose forward subtraction returns NotImplemented, so the right
    # operand reflected method is tried.
    # KNOWN ISSUE: the class docstring says the forward method returns a marker string but it
    # returns NotImplemented.
    """Class where __sub__ returns a marker string"""

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Forward subtraction: always returns NotImplemented.
        return NotImplemented


class _RSubReturnsMarker:
    # [CLASS DOC] HELPER: Class defining only a reflected subtraction that returns a marker string.
    """Class where __rsub__ returns a marker string"""

    def __rsub__(self, other):
        # [FUNCTION DOC] HELPER: Reflected subtraction: returns the constant marker string.
        return "_RSubReturnsMarker.__rsub__ called"


class TestNbSub(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the subtraction operator under Dynamo: ints, floats, bools,
    # complex, set, frozenset, dict views, OrderedSet, in-place subtraction, reflected subtraction,
    # user-defined subtraction, subclass priority, Rat arithmetic, NotImplemented handling and
    # symbolic shape values.
    def setUp(self):
        # [FUNCTION DOC] Enables the Dynamo unittest-tracing config flag, saving the previous value
        # on the instance for tearDown.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Restores the unittest-tracing config flag saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Arithmetic sub ---

    @make_dynamo_test
    def test_sub_integers(self):
        # [FUNCTION DOC] Integer subtraction gives the expected positive, negative and zero
        # results.
        self.assertEqual(10 - 3, 7)
        self.assertEqual(0 - 5, -5)
        self.assertEqual(5 - 5, 0)

    @make_dynamo_test
    def test_sub_floats(self):
        # [FUNCTION DOC] Float subtraction is checked with approximate equality.
        self.assertAlmostEqual(1.5 - 0.5, 1.0)
        self.assertAlmostEqual(3.14 - 1.14, 2.0)

    @make_dynamo_test
    def test_sub_negative(self):
        # [FUNCTION DOC] Subtraction with negative operands gives the expected signs.
        self.assertEqual(-3 - (-5), 2)
        self.assertEqual(-3 - 5, -8)

    @make_dynamo_test
    def test_sub_chained(self):
        # [FUNCTION DOC] Chained subtraction of ints evaluates left to right.
        self.assertEqual(10 - 3 - 2 - 1, 4)

    @make_dynamo_test
    def test_sub_zero(self):
        # [FUNCTION DOC] Subtracting zero leaves the number unchanged and zero minus zero is zero.
        self.assertEqual(5 - 0, 5)
        self.assertEqual(0 - 0, 0)

    # --- Sub booleans ---

    @make_dynamo_test
    def test_sub_bools(self):
        # [FUNCTION DOC] Bool subtraction behaves like int subtraction.
        self.assertEqual(True - True, 0)
        self.assertEqual(True - False, 1)
        self.assertEqual(False - False, 0)

    @make_dynamo_test
    def test_sub_int_and_bool(self):
        # [FUNCTION DOC] Subtracting True from an int, and True from zero, treat True as one.
        self.assertEqual(5 - True, 4)
        self.assertEqual(0 - True, -1)

    @make_dynamo_test
    def test_sub_int_mixed(self):
        # [FUNCTION DOC] Mixed int subtraction with literals and a local variable gives the
        # expected values.
        self.assertEqual(100 - 1, 99)
        self.assertEqual(1 - 100, -99)
        self.assertEqual(-10 - (-10), 0)
        x = 42
        self.assertEqual(x - 2, 40)

    @make_dynamo_test
    def test_sub_float_mixed(self):
        # [FUNCTION DOC] Mixed float subtraction using approximate equality, including a local
        # variable.
        self.assertAlmostEqual(0.1 - 0.1, 0.0)
        self.assertAlmostEqual(1.0 - 0.3, 0.7)
        self.assertAlmostEqual(-1.5 - (-2.5), 1.0)
        x = 3.14
        self.assertAlmostEqual(x - 1.0, 2.14)

    @make_dynamo_test
    def test_sub_complex(self):
        # [FUNCTION DOC] Complex number subtraction gives componentwise differences, including zero
        # results.
        self.assertEqual((3 + 4j) - (1 + 2j), 2 + 2j)
        self.assertEqual((1 + 0j) - (0 + 1j), 1 - 1j)
        self.assertEqual((0 + 0j) - (0 + 0j), 0j)
        self.assertEqual((-1 - 1j) - (-1 - 1j), 0j)

    @make_dynamo_test
    def test_sub_complex_with_real(self):
        # [FUNCTION DOC] Subtraction between complex and int or float, in both orders, gives the
        # expected complex results.
        self.assertEqual((5 + 3j) - 2, 3 + 3j)
        self.assertEqual((5 + 3j) - 2.0, 3 + 3j)
        self.assertEqual(10 - (3 + 4j), 7 - 4j)
        self.assertEqual(1.5 - (0.5 + 1j), 1 - 1j)

    # --- Set difference ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ({1, 2, 3}, {2, 3}, {1}),
            ({1, 2}, {2}, {1}),
            ({1, 2, 3}, {4, 5}, {1, 2, 3}),
        ],
    )
    @make_dynamo_test
    def test_set_difference(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over operand pairs; set difference via subtraction equals the
        # expected set.
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_set_difference_with_empty(self):
        # [FUNCTION DOC] Set difference with the empty set, in both orders.
        self.assertEqual({1, 2} - set(), {1, 2})
        self.assertEqual(set() - {1, 2}, set())

    @make_dynamo_test
    def test_set_difference_empty(self):
        # [FUNCTION DOC] Difference of two empty sets is empty.
        self.assertEqual(set() - set(), set())

    @make_dynamo_test
    def test_set_difference_chained(self):
        # [FUNCTION DOC] Chained set differences remove elements successively.
        self.assertEqual({1, 2, 3, 4} - {2} - {3}, {1, 4})

    # --- Frozenset difference ---

    @parametrize(
        "operand1,operand2,expected",
        [
            (frozenset({1, 2, 3}), frozenset({2, 3}), frozenset({1})),
            (frozenset({1, 2}), frozenset({2}), frozenset({1})),
            (frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({1, 2, 3})),
        ],
    )
    @make_dynamo_test
    def test_frozenset_difference(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over frozenset pairs; difference equals the expected
        # frozenset.
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_frozenset_difference_with_empty(self):
        # [FUNCTION DOC] Frozenset difference with an empty frozenset in both orders.
        self.assertEqual(frozenset({1, 2}) - frozenset(), frozenset({1, 2}))
        self.assertEqual(frozenset() - frozenset({1, 2}), frozenset())

    # --- Dict view difference ---

    @make_dynamo_test
    def test_dict_keys_difference(self):
        # [FUNCTION DOC] Difference of dict keys view and a set gives the remaining keys as a set.
        self.assertEqual({"a": 1, "b": 2}.keys() - {"b"}, {"a"})

    @make_dynamo_test
    def test_dict_items_difference(self):
        # [FUNCTION DOC] Difference of dict items view and a set of pairs gives the remaining pairs
        # as a set.
        left = {"a": 1, "b": 2}.items()
        right = {("b", 2)}
        self.assertEqual(left - right, {("a", 1)})

    def test_dict_values_difference_unsupported(self):
        # [FUNCTION DOC] Compiles a function that subtracts two dict values views with fullgraph
        # and eager backend; asserts Dynamo raises its Unsupported exception.
        def fn():
            # [FUNCTION DOC] HELPER: Nested function subtracting two dict values views; the result
            # is discarded and the operation is expected to be unsupported when compiled.
            {"a": 1}.values() - {"b": 2}.values()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn()

    # --- OrderedSet difference ---

    @make_dynamo_test
    def test_orderedset_difference_basic(self):
        # [FUNCTION DOC] OrderedSet difference with an overlapping set.
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([2, 3])
        expected = OrderedSet([1])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_single(self):
        # [FUNCTION DOC] OrderedSet difference removing one element.
        operand1 = OrderedSet([1, 2])
        operand2 = OrderedSet([2])
        expected = OrderedSet([1])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_no_overlap(self):
        # [FUNCTION DOC] OrderedSet difference with a disjoint set returns the original elements.
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([4, 5])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_with_empty(self):
        # [FUNCTION DOC] OrderedSet difference with an empty OrderedSet in both orders.
        self.assertEqual(OrderedSet([1, 2]) - OrderedSet(), OrderedSet([1, 2]))
        self.assertEqual(OrderedSet() - OrderedSet([1, 2]), OrderedSet())

    @make_dynamo_test
    def test_orderedset_difference_empty(self):
        # [FUNCTION DOC] Difference of two empty OrderedSets is empty.
        self.assertEqual(OrderedSet() - OrderedSet(), OrderedSet())

    @make_dynamo_test
    def test_orderedset_difference_chained(self):
        # [FUNCTION DOC] Chained OrderedSet differences across three sets.
        self.assertEqual(
            OrderedSet([1, 2, 3, 4]) - OrderedSet([2]) - OrderedSet([3]),
            OrderedSet([1, 4]),
        )

    # --- Inplace -= ---

    @make_dynamo_test
    def test_inplace_sub_integers(self):
        # [FUNCTION DOC] In-place subtraction on an int variable.
        x = 10
        x -= 3
        self.assertEqual(x, 7)

    @make_dynamo_test
    def test_inplace_sub_set(self):
        # [FUNCTION DOC] In-place subtraction on a set removes elements.
        s = {1, 2, 3}
        s -= {2, 3}
        self.assertEqual(s, {1})

    @make_dynamo_test
    def test_inplace_sub_frozenset(self):
        # [FUNCTION DOC] In-place subtraction on a frozenset rebinds the name to the smaller
        # frozenset.
        f = frozenset({1, 2, 3})
        f -= frozenset({2, 3})
        self.assertEqual(f, frozenset({1}))

    @make_dynamo_test
    def test_inplace_sub_set_with_empty(self):
        # [FUNCTION DOC] In-place subtraction of an empty set leaves the set unchanged.
        s = {1, 2, 3}
        s -= set()
        self.assertEqual(s, {1, 2, 3})

    @make_dynamo_test
    def test_inplace_sub_frozenset_with_empty(self):
        # [FUNCTION DOC] In-place subtraction of an empty frozenset leaves it unchanged.
        f = frozenset({1, 2, 3})
        f -= frozenset()
        self.assertEqual(f, frozenset({1, 2, 3}))

    @make_dynamo_test
    def test_inplace_sub_orderedset_mutates_alias(self):
        # [FUNCTION DOC] In-place subtraction on an OrderedSet mutates the object in place so an
        # alias to it sees the change.
        s = OrderedSet([1, 2, 3])
        alias = s
        s -= OrderedSet([2])
        self.assertEqual(s, OrderedSet([1, 3]))
        self.assertEqual(alias, OrderedSet([1, 3]))

    # --- Reversed sub (__rsub__) ---

    @make_dynamo_test
    def test_reversed_sub_with_integer(self):
        # [FUNCTION DOC] Int minus user-defined instance goes through the reflected method.
        obj = UserDefinedClassWithSub(3)
        result = 10 - obj
        self.assertEqual(result, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_reversed_sub_with_user_defined_object(self):
        # [FUNCTION DOC] Subtraction between two user-defined instances uses the forward method.
        obj1 = UserDefinedClassWithSub(5)
        obj2 = UserDefinedClassWithSub(3)
        result = obj1 - obj2
        self.assertEqual(result, UserDefinedClassWithSub(2))

    @make_dynamo_test
    def test_reversed_sub_chained(self):
        # [FUNCTION DOC] Chained subtraction of three user-defined instances evaluates left to
        # right.
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        obj3 = UserDefinedClassWithSub(2)
        result = obj1 - obj2 - obj3
        self.assertEqual(result, UserDefinedClassWithSub(5))

    # --- User-defined __sub__ ---

    @make_dynamo_test
    def test_user_defined_sub_basic(self):
        # [FUNCTION DOC] User-defined forward subtraction between two instances.
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        self.assertEqual(obj1 - obj2, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_user_defined_sub_with_integer(self):
        # [FUNCTION DOC] User-defined instance minus an int uses the raw operand.
        obj = UserDefinedClassWithSub(10)
        self.assertEqual(obj - 3, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_user_defined_sub_zero(self):
        # [FUNCTION DOC] User-defined instance minus zero returns an equal instance.
        obj = UserDefinedClassWithSub(5)
        self.assertEqual(obj - 0, UserDefinedClassWithSub(5))

    @make_dynamo_test
    def test_user_defined_sub_chained(self):
        # [FUNCTION DOC] Chained user-defined subtraction of three instances.
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        obj3 = UserDefinedClassWithSub(2)
        self.assertEqual(obj1 - obj2 - obj3, UserDefinedClassWithSub(5))

    # --- Cross-type user-defined sub ---

    @make_dynamo_test
    def test_left_sub_left_uses_sub(self):
        # [FUNCTION DOC] Two LeftSubClass instances subtract using forward subtraction.
        a = LeftSubClass(10)
        b = LeftSubClass(3)
        self.assertEqual(a - b, LeftSubClass(7))

    @make_dynamo_test
    def test_right_sub_right_uses_sub(self):
        # [FUNCTION DOC] Two RightSubClass instances subtract using forward subtraction.
        a = RightSubClass(10)
        b = RightSubClass(3)
        self.assertEqual(a - b, RightSubClass(7))

    @make_dynamo_test
    def test_left_sub_right_falls_back_to_rsub(self):
        # [FUNCTION DOC] LeftSubClass minus RightSubClass: left declines so the reflected method of
        # the right gives the descriptive string.
        a = LeftSubClass(10)
        b = RightSubClass(3)
        self.assertEqual(a - b, "LeftSubClass(10)-RightSubClass(3)")

    @make_dynamo_test
    def test_right_sub_left_raises(self):
        # [FUNCTION DOC] RightSubClass minus LeftSubClass: both sides decline so TypeError is
        # raised.
        a = RightSubClass(10)
        b = LeftSubClass(3)
        with self.assertRaises(TypeError):
            a - b

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # [FUNCTION DOC] Int subclass overriding subtraction: both forward and reflected overrides
        # are chosen according to operand position.
        self.assertEqual(_IntSubWithSub(1) - 1, "_IntSubWithSub.__sub__")
        self.assertEqual(1 - _IntSubWithSub(1), "_IntSubWithSub.__rsub__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # [FUNCTION DOC] Baseline: plain base class forward and reflected markers with an int on
        # either side.
        self.assertEqual(_BaseWithSub() - 1, "_BaseWithSub.__sub__")
        self.assertEqual(1 - _BaseWithSub(), "_BaseWithSub.__rsub__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        # [FUNCTION DOC] Right operand subclass overriding the reflected method takes priority over
        # the left base-class instance.
        self.assertEqual(_SubWithSub() - _BaseWithSub(), "_SubWithSub.__sub__")
        self.assertEqual(_BaseWithSub() - _SubWithSub(), "_SubWithSub.__rsub__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        # [FUNCTION DOC] Subclass inheriting the reflected method gets no priority; also asserts
        # the inherited reflected method is the base one.
        self.assertIs(_InheritedSubSub.__rsub__, _BaseWithSub.__rsub__)
        self.assertEqual(_InheritedSubSub() - 1, "_BaseWithSub.__sub__")
        self.assertEqual(1 - _InheritedSubSub(), "_BaseWithSub.__rsub__")
        self.assertEqual(_InheritedSubSub() - _BaseWithSub(), "_BaseWithSub.__sub__")
        self.assertEqual(_BaseWithSub() - _InheritedSubSub(), "_BaseWithSub.__sub__")

    # --- Rat class tests from CPython test_binop.py ---

    @make_dynamo_test
    def test_rat_sub(self):
        # [FUNCTION DOC] Rat subtraction with Rats, ints and floats in both orders; floats compared
        # approximately.
        self.assertEqual(Rat(7, 2) - Rat(7, 5), Rat(21, 10))
        self.assertEqual(Rat(7, 5) - 1, Rat(2, 5))
        self.assertEqual(1 - Rat(3, 5), Rat(2, 5))
        self.assertAlmostEqual(Rat(3, 2) - 1.0, 0.5)
        self.assertAlmostEqual(1.0 - Rat(1, 2), 0.5)

    @make_dynamo_test
    def test_rat_rsub(self):
        # [FUNCTION DOC] Reflected subtraction of a Rat from an int returns the expected Rat.
        # Test reversed subtraction with Rat objects
        obj = Rat(3)
        result = 10 - obj
        self.assertEqual(result, Rat(7))

    @make_dynamo_test
    def test_rat_sub_with_different_denominators(self):
        # [FUNCTION DOC] Rat subtraction with different denominators gives a reduced result.
        # More complex rational subtraction
        a = Rat(5, 6)
        b = Rat(1, 3)
        result = a - b
        self.assertEqual(result, Rat(1, 2))

    @make_dynamo_test
    def test_rat_sub_negative_result(self):
        # [FUNCTION DOC] Rat subtraction yielding a negative result.
        # Subtraction resulting in negative
        a = Rat(1, 4)
        b = Rat(1, 2)
        result = a - b
        self.assertEqual(result, Rat(-1, 4))

    # --- NotImplemented handling from test_descr.py ---

    @make_dynamo_test
    def test_sub_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Forward subtraction returning NotImplemented on both sides raises
        # TypeError.
        # When __sub__ returns NotImplemented, should raise TypeError
        a = _SubNotImplemented()
        with self.assertRaises(TypeError):
            a - a

    @make_dynamo_test
    def test_rsub_not_implemented_returns_type_error(self):
        # [FUNCTION DOC] Class whose forward and reflected subtraction both decline raises
        # TypeError.
        # When __rsub__ returns NotImplemented, should raise TypeError
        b = _RSubNotImplemented()
        with self.assertRaises(TypeError):
            b - b

    @make_dynamo_test
    def test_sub_mixed_not_implemented_fallback(self):
        # [FUNCTION DOC] Forward declines, so the right operand's reflected method runs and its
        # marker string is returned.
        # When left __sub__ returns NotImplemented, try right __rsub__
        a = _SubReturnsMarker()
        b = _RSubReturnsMarker()
        result = a - b
        self.assertEqual(result, "_RSubReturnsMarker.__rsub__ called")

    def test_sub_float_and_symnode(self):
        # [FUNCTION DOC] Compiles a nested function with fullgraph and eager backend over a tensor
        # with a dynamic first dimension and compares with eager; covers float minus symbolic int
        # and the reverse.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function computing a float minus the size and the size
            # minus a float, then subtracting both from the tensor.
            s = x.shape[0]
            a = 2.5 - s
            b = s - 1.25
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_sub_int_and_symnode(self):
        # [FUNCTION DOC] Same as the float symnode test but with int operands mixed with the
        # symbolic size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function with int minus size and size minus int,
            # subtracting both from the input tensor.
            s = x.shape[0]
            a = 3 - s
            b = s - 7
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_sub_bool_and_symnode(self):
        # [FUNCTION DOC] Same as the symnode tests but with bool operands mixed with the symbolic
        # size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function with True minus size and size minus False,
            # subtracting both from the input tensor.
            s = x.shape[0]
            a = True - s
            b = s - False
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    # --- OrderedSet reversed subtraction (__rsub__) ---

    @make_dynamo_test
    def test_orderedset_rsub_basic(self):
        # [FUNCTION DOC] Subtracting an OrderedSet from a plain set gives a plain set difference.
        # Test reversed subtraction: other_set - ordered_set
        os = OrderedSet([2, 3])
        s = {1, 2, 3}
        result = s - os
        self.assertEqual(result, {1})

    @make_dynamo_test
    def test_orderedset_rsub_with_set(self):
        # [FUNCTION DOC] Another plain set minus OrderedSet check; remaining element is the one not
        # in the OrderedSet.
        # Test that set - OrderedSet works correctly
        os = OrderedSet([1, 2])
        s = {1, 2, 3}
        result = s - os
        self.assertEqual(result, {3})

    @make_dynamo_test
    def test_orderedset_rsub_both_orderedsets(self):
        # [FUNCTION DOC] OrderedSet minus OrderedSet gives an OrderedSet difference.
        # Test OrderedSet - OrderedSet (forward subtraction)
        os1 = OrderedSet([1, 2, 3])
        os2 = OrderedSet([2, 3])
        result = os1 - os2
        self.assertEqual(result, OrderedSet([1]))


class _AddNotImplemented:
    # [CLASS DOC] HELPER: Class whose forward addition returns NotImplemented.
    """Class where __add__ returns NotImplemented"""

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Forward addition: always returns NotImplemented.
        return NotImplemented


class _RAddNotImplemented:
    # [CLASS DOC] HELPER: Class whose forward and reflected addition both return NotImplemented.
    """Class where __add__ exists but __radd__ returns NotImplemented"""

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Forward addition: always returns NotImplemented.
        return NotImplemented

    def __radd__(self, other):
        # [FUNCTION DOC] HELPER: Reflected addition: always returns NotImplemented.
        return NotImplemented


class _BaseWithAdd:
    # [CLASS DOC] HELPER: Base class with forward and reflected addition returning base marker
    # strings.
    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Forward addition: returns the base forward marker string.
        return "_BaseWithAdd.__add__"

    def __radd__(self, other):
        # [FUNCTION DOC] HELPER: Reflected addition: returns the base reflected marker string.
        return "_BaseWithAdd.__radd__"


class _SubWithAdd(_BaseWithAdd):
    # [CLASS DOC] HELPER: Subclass overriding only forward addition with its own marker; the
    # reflected method is inherited.
    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Forward addition: returns the subclass forward marker string.
        return "_SubWithAdd.__add__"


class _InheritedSubAdd(_BaseWithAdd):
    # [CLASS DOC] HELPER: Subclass overriding nothing, for the no-priority checks.
    pass


class TestNbAdd(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the addition operator under Dynamo: ints, floats, bools, strings,
    # lists, tuples, in-place addition, reflected addition, subclass method resolution, Rat
    # arithmetic, NotImplemented handling and tensors with scalars and symbolic sizes.
    def setUp(self):
        # [FUNCTION DOC] Enables the Dynamo unittest-tracing config flag, saving the previous value
        # on the instance for tearDown.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Restores the unittest-tracing config flag saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Arithmetic add ---

    @make_dynamo_test
    def test_add_integers(self):
        # [FUNCTION DOC] Integer addition gives the expected sums.
        self.assertEqual(10 + 3, 13)
        self.assertEqual(0 + 5, 5)
        self.assertEqual(5 + 5, 10)

    @make_dynamo_test
    def test_add_floats(self):
        # [FUNCTION DOC] Float addition with approximate equality.
        self.assertAlmostEqual(1.5 + 0.5, 2.0)
        self.assertAlmostEqual(3.14 + 1.14, 4.28)

    @make_dynamo_test
    def test_add_negative(self):
        # [FUNCTION DOC] Addition with negative operands.
        self.assertEqual(-3 + (-5), -8)
        self.assertEqual(-3 + 5, 2)

    @make_dynamo_test
    def test_add_chained(self):
        # [FUNCTION DOC] Chained integer addition.
        self.assertEqual(10 + 3 + 2 + 1, 16)

    @make_dynamo_test
    def test_add_zero(self):
        # [FUNCTION DOC] Adding zero leaves values unchanged.
        self.assertEqual(5 + 0, 5)
        self.assertEqual(0 + 0, 0)

    # --- Add booleans ---

    @make_dynamo_test
    def test_add_bools(self):
        # [FUNCTION DOC] Bool addition behaves like int addition.
        self.assertEqual(True + True, 2)
        self.assertEqual(True + False, 1)
        self.assertEqual(False + False, 0)

    @make_dynamo_test
    def test_add_int_and_bool(self):
        # [FUNCTION DOC] Int plus True treats True as one.
        self.assertEqual(5 + True, 6)
        self.assertEqual(0 + True, 1)

    # --- String concatenation ---

    @make_dynamo_test
    def test_add_strings(self):
        # [FUNCTION DOC] String concatenation with plus.
        self.assertEqual("hello" + " " + "world", "hello world")
        self.assertEqual("a" + "b", "ab")
        self.assertEqual("" + "test", "test")

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
    def test_add_lists(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over list pairs; concatenation via plus equals the expected
        # list.
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_add_lists_empty(self):
        # [FUNCTION DOC] Concatenating two empty lists gives an empty list.
        self.assertEqual([] + [], [])

    @make_dynamo_test
    def test_add_lists_chained(self):
        # [FUNCTION DOC] Chained list concatenation.
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
    def test_add_tuples(self, operand1, operand2, expected):
        # [FUNCTION DOC] Parametrized over tuple pairs; concatenation equals the expected tuple.
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_add_tuples_empty(self):
        # [FUNCTION DOC] Concatenating two empty tuples gives an empty tuple.
        self.assertEqual(() + (), ())

    @make_dynamo_test
    def test_add_tuples_chained(self):
        # [FUNCTION DOC] Chained tuple concatenation of singletons.
        self.assertEqual((1,) + (2,) + (3,), (1, 2, 3))

    # --- Inplace +=  ---

    @make_dynamo_test
    def test_inplace_add_integers(self):
        # [FUNCTION DOC] In-place addition on an int variable.
        x = 5
        x += 3
        self.assertEqual(x, 8)

    @make_dynamo_test
    def test_inplace_add_floats(self):
        # [FUNCTION DOC] In-place addition on a float variable, compared approximately.
        x = 1.5
        x += 2.5
        self.assertAlmostEqual(x, 4.0)

    @make_dynamo_test
    def test_inplace_add_strings(self):
        # [FUNCTION DOC] In-place addition on a string appends text.
        x = "hello"
        x += " world"
        self.assertEqual(x, "hello world")

    @make_dynamo_test
    def test_inplace_add_lists(self):
        # [FUNCTION DOC] In-place addition on a list extends it.
        x = [1, 2]
        x += [3, 4]
        self.assertEqual(x, [1, 2, 3, 4])

    # --- Reversed add (__radd__) ---

    @make_dynamo_test
    def test_reversed_add_with_integer(self):
        # [FUNCTION DOC] Int plus an object that defines the reflected method returns its reflected
        # marker string.
        obj = _BaseWithAdd()
        result = 5 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_radd_priority(self):
        # [FUNCTION DOC] Same check as the previous test: the reflected method is used when the int
        # does not handle the operation.
        # Test that __radd__ is called when left side doesn't implement __add__
        obj = _BaseWithAdd()
        result = 5 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    # --- Subclass method resolution order ---

    @make_dynamo_test
    def test_subclass_add(self):
        # [FUNCTION DOC] Subclass instance plus an int uses the subclass forward override.
        obj = _SubWithAdd()
        result = obj + 1
        self.assertEqual(result, "_SubWithAdd.__add__")

    @make_dynamo_test
    def test_subclass_radd(self):
        # [FUNCTION DOC] Int plus subclass instance uses the inherited base reflected method.
        obj = _SubWithAdd()
        result = 1 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_subclass_and_base_add(self):
        # [FUNCTION DOC] Subclass plus base instance uses the subclass forward method.
        sub = _SubWithAdd()
        base = _BaseWithAdd()
        result = sub + base
        self.assertEqual(result, "_SubWithAdd.__add__")

    @make_dynamo_test
    def test_subclass_priority_in_reverse(self):
        # [FUNCTION DOC] Asserts the subclass reflected method is the base one and checks marker
        # results for four operand combinations of subclass and base objects.
        self.assertIs(_SubWithAdd.__radd__, _BaseWithAdd.__radd__)
        self.assertEqual(_SubWithAdd() + 1, "_SubWithAdd.__add__")
        self.assertEqual(1 + _SubWithAdd(), "_BaseWithAdd.__radd__")
        self.assertEqual(_SubWithAdd() + _BaseWithAdd(), "_SubWithAdd.__add__")
        self.assertEqual(_BaseWithAdd() + _SubWithAdd(), "_BaseWithAdd.__add__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        # [FUNCTION DOC] Subclass that overrides nothing gets no priority in any ordering; forward
        # base method is used with mixed operands and the reflected method is the base one.
        self.assertIs(_InheritedSubAdd.__radd__, _BaseWithAdd.__radd__)
        self.assertEqual(_InheritedSubAdd() + 1, "_BaseWithAdd.__add__")
        self.assertEqual(1 + _InheritedSubAdd(), "_BaseWithAdd.__radd__")
        self.assertEqual(_InheritedSubAdd() + _BaseWithAdd(), "_BaseWithAdd.__add__")
        self.assertEqual(_BaseWithAdd() + _InheritedSubAdd(), "_BaseWithAdd.__add__")

    # --- Rat class tests from CPython test_binop.py ---

    @make_dynamo_test
    def test_rat_add(self):
        # [FUNCTION DOC] Rat addition with Rats, ints and floats in both operand orders; sum of two
        # thirds and one third equals one.
        # Tests from CPython test_binop.py RatTestCase.test_add
        self.assertEqual(Rat(2, 3) + Rat(1, 3), 1)
        self.assertEqual(Rat(2, 3) + 1, Rat(5, 3))
        self.assertEqual(1 + Rat(2, 3), Rat(5, 3))
        self.assertAlmostEqual(1.0 + Rat(1, 2), 1.5)
        self.assertAlmostEqual(Rat(1, 2) + 1.0, 1.5)

    @make_dynamo_test
    def test_rat_add_with_rat(self):
        # [FUNCTION DOC] Adding two Rats gives the exact Rat sum.
        # Addition of two rationals
        self.assertEqual(Rat(7, 2) + Rat(7, 5), Rat(49, 10))

    @make_dynamo_test
    def test_rat_add_rat_and_integer(self):
        # [FUNCTION DOC] Adding an int to a Rat gives a Rat.
        # Addition of rational and integer
        self.assertEqual(Rat(7, 5) + 1, Rat(12, 5))

    @make_dynamo_test
    def test_rat_add_integer_and_rat(self):
        # [FUNCTION DOC] Int plus Rat uses the reflected path and gives a Rat.
        # Addition of integer and rational (tests __radd__)
        self.assertEqual(1 + Rat(3, 5), Rat(8, 5))

    @make_dynamo_test
    def test_rat_add_rat_and_float(self):
        # [FUNCTION DOC] Rat plus float gives a float, compared approximately.
        # Addition of rational and float
        self.assertAlmostEqual(Rat(3, 2) + 1.0, 2.5)

    @make_dynamo_test
    def test_rat_add_float_and_rat(self):
        # [FUNCTION DOC] Float plus Rat gives a float, compared approximately.
        # Addition of float and rational (tests __radd__)
        self.assertAlmostEqual(1.0 + Rat(3, 2), 2.5)

    @make_dynamo_test
    def test_rat_radd(self):
        # [FUNCTION DOC] Int plus Rat through the reflected method.
        # Test reversed addition with Rat objects
        obj = Rat(3)
        result = 10 + obj
        self.assertEqual(result, Rat(13))

    @make_dynamo_test
    def test_rat_add_with_different_denominators(self):
        # [FUNCTION DOC] Rat addition with different denominators.
        # More complex rational addition
        a = Rat(5, 6)
        b = Rat(1, 3)
        result = a + b
        self.assertEqual(result, Rat(7, 6))

    @make_dynamo_test
    def test_rat_add_result_simplification(self):
        # [FUNCTION DOC] Rat addition whose sum reduces to a simpler fraction.
        # Addition resulting in simplifiable fraction
        a = Rat(1, 4)
        b = Rat(1, 4)
        result = a + b
        self.assertEqual(result, Rat(1, 2))

    # --- NotImplemented handling from test_descr.py ---

    @make_dynamo_test
    def test_add_not_implemented_left(self):
        # [FUNCTION DOC] Instance whose forward addition declines, plus an int, raises TypeError
        # since int has no matching reflected method.
        # When left operand returns NotImplemented, should try right's __radd__
        obj = _AddNotImplemented()
        # Left returns NotImplemented, right (int) doesn't have __radd__, raises TypeError
        with self.assertRaises(TypeError):
            obj + 5

    @make_dynamo_test
    def test_add_not_implemented_with_custom_radd(self):
        # [FUNCTION DOC] Forward declines so the right operand's reflected method is used and
        # returns its marker string.
        # When left returns NotImplemented, should use right's __radd__ if available
        obj = _AddNotImplemented()
        custom = _BaseWithAdd()
        # Left (__add__) returns NotImplemented, right (__radd__) is called
        result = obj + custom
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_radd_not_implemented_uses_left(self):
        # [FUNCTION DOC] Right operand's reflected method declines but the left operand's forward
        # method succeeds, so its result is returned.
        # When left has __add__ that works, it should be called even if right's __radd__ returns NotImplemented
        obj = _RAddNotImplemented()
        custom = _BaseWithAdd()
        result = custom + obj
        # Left (__add__) is called and returns successfully, right is not tried
        self.assertEqual(result, "_BaseWithAdd.__add__")

    @make_dynamo_test
    def test_radd_not_implemented_right_used(self):
        # [FUNCTION DOC] Both operands decline in both directions so a TypeError is raised.
        # When both __add__ and __radd__ return NotImplemented, should raise TypeError
        obj1 = _RAddNotImplemented()
        obj2 = _RAddNotImplemented()
        with self.assertRaises(TypeError):
            obj1 + obj2

    @make_dynamo_test
    def test_add_not_implemented_reverse_with_integer(self):
        # [FUNCTION DOC] Int plus an object whose forward addition declines raises TypeError.
        # Right operand with NotImplemented when left is int
        obj = _AddNotImplemented()
        with self.assertRaises(TypeError):
            5 + obj

    @make_dynamo_test
    def test_add_not_implemented_with_int_and_custom(self):
        # [FUNCTION DOC] Left declines and the right base class reflected method supplies the
        # marker string.
        # _AddNotImplemented returns NotImplemented, then _BaseWithAdd.__radd__ is called
        obj_not_impl = _AddNotImplemented()
        obj_impl = _BaseWithAdd()
        # obj_not_impl.__add__ returns NotImplemented, tries obj_impl.__radd__
        result = obj_not_impl + obj_impl
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    def test_add_float_and_symnode(self):
        # [FUNCTION DOC] Compiles a nested function with fullgraph over a tensor with dynamic first
        # dimension, using float plus symbolic size; compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding a float and the size in both orders to
            # the tensor.
            s = x.shape[0]
            a = 2.5 + s
            b = s + 1.25
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_int_and_symnode(self):
        # [FUNCTION DOC] Same pattern with int operands and the symbolic size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding int and size in both orders to the
            # tensor.
            s = x.shape[0]
            a = 3 + s
            b = s + 7
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_bool_and_symnode(self):
        # [FUNCTION DOC] Same pattern with bool operands and the symbolic size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding True and size, and size and False, to
            # the tensor.
            s = x.shape[0]
            a = True + s
            b = s + False
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles tensor plus tensor with fullgraph and compares to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the sum of its two tensor arguments.
            return x + y

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_iadd_tensor_and_tensor(self):
        # [FUNCTION DOC] Compiles an in-place tensor addition with fullgraph and compares with
        # eager using clones of the input.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function applying in-place addition of y to x and
            # returning x.
            x += y
            return x

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x.clone(), y), fn(x.clone(), y))

    def test_iadd_tensor_preserves_identity(self):
        # [FUNCTION DOC] Verifies that in-place tensor addition inside a compiled function keeps
        # the same object identity and returns True.
        def fn():
            # [FUNCTION DOC] HELPER: Nested function creating a ones tensor, adding ones in place
            # and returning whether the name still refers to the original object.
            d = torch.ones(2, 2)
            d_copy = d
            d += torch.ones(2, 2)
            return d is d_copy

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertTrue(opt_fn())

    def test_iadd_tensor_and_scalar(self):
        # [FUNCTION DOC] Compiles in-place addition of a scalar to a tensor and compares with eager
        # using clones.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function adding three to its tensor argument in place
            # and returning it.
            x += 3
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x.clone()), fn(x.clone()))

    def test_add_tensor_and_scalar(self):
        # [FUNCTION DOC] Compiles tensor plus int, float and bool scalars in both orders returning
        # all six results and compares with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning a tuple of tensor-scalar sums in
            # both operand orders.
            return x + 3, 3 + x, x + 2.5, 2.5 + x, x + True, False + x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_tensor_and_symnode(self):
        # [FUNCTION DOC] Compiles tensor plus symbolic size in both orders with a dynamic first
        # dimension.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested function returning the tensor plus size and size plus
            # tensor.
            s = x.shape[0]
            return x + s, s + x

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_tensor_broadcast(self):
        # [FUNCTION DOC] Compiles addition of a matrix and a vector (broadcasting) with fullgraph
        # and compares with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Nested function returning the sum of its two tensor arguments.
            return x + y

        x = torch.randn(4, 4)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


instantiate_parametrized_tests(TestNbOr)
instantiate_parametrized_tests(TestNbSub)
instantiate_parametrized_tests(TestNbAdd)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
