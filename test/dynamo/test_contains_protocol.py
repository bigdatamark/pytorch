# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the `in` / `not in` operators and the __contains__ protocol.
# Defines module-level helper container classes (explicit __contains__, iteration fallback,
# __getitem__ sequence fallback, subclasses with overridden __contains__, error raisers) and
# TestCase classes that run membership checks under make_dynamo_test with unittest tracing
# enabled. Covers builtin containers, range, dict views, operator.contains, and error cases.
# No import-time side effects beyond class definitions.

"""
Tests for the `in` operator and __contains__ protocol in PyTorch Dynamo.

Tests cover:
- sq_contains protocol: list, tuple, str, range, set, frozenset
- mp_contains protocol: dict, dict.keys()
- Fallback iteration: objects with __iter__ but no __contains__
- Sequence-protocol fallback: objects with __getitem__ but no __contains__/__iter__
- operator.contains() — exercises the builtin.py call_contains path
- User-defined classes with __contains__
- User-defined classes without __contains__ (iteration fallback)
- Error handling: unhashable types, non-iterable objects, __contains__ raises,
  non-bool return values, str TypeError, mid-iteration raise
"""

import operator
import types

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test, xfailIfPy314Plus


# ---------------------------------------------------------------------------
# Helper classes defined at module level (not inside compiled functions)
# ---------------------------------------------------------------------------


class WithContains:
    # [CLASS DOC] HELPER: test container whose __contains__ delegates to membership in its
    # stored data.
    """Has explicit __contains__."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return whether item is in the stored data.
        return item in self.data


class WithIterNoContains:
    # [CLASS DOC] HELPER: container with only __iter__ over its data; membership must use
    # iteration fallback.
    """Has __iter__ but no __contains__ — forces iteration fallback."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over the stored data.
        return iter(self.data)


class WithGetitemNoContains:
    # [CLASS DOC] HELPER: container with only __getitem__ over its data; membership uses the
    # legacy sequence protocol.
    """Sequence protocol via __getitem__, no __contains__."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: return the element of the stored data at idx (raises whatever
        # the data raises).
        return self.data[idx]


class WithContainsAndIter:
    # [CLASS DOC] HELPER: container with both __contains__ and __iter__; tracks __iter__ calls
    # in iter_calls.
    """Has both __contains__ and __iter__; __contains__ should be preferred."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store data and set the iter_calls counter to 0.
        self.data = data
        self.iter_calls = 0

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return whether item is in the stored data (iter_calls is
        # not touched).
        return item in self.data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: increment iter_calls and return an iterator over the
        # stored data.
        self.iter_calls += 1
        return iter(self.data)


class ListIterWrapper:
    # [CLASS DOC] HELPER: iteration-only wrapper whose iterator yields each element doubled.
    """List wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over the doubled elements of the data.
        # yields items doubled so membership tests differ from base list
        return iter([x * 2 for x in self.data])


class ListGetitemWrapper:
    # [CLASS DOC] HELPER: getitem-only wrapper that returns each element shifted up by 100.
    """List wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: return the element at idx plus 100.
        # shifts each value up by 100
        return self.data[idx] + 100


class DictIterWrapper:
    # [CLASS DOC] HELPER: iteration-only wrapper whose iterator yields the values (not keys) of
    # a dict.
    """Dict wrapper with custom __iter__ (over values) and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped dict.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over the dict values.
        return iter(self.data.values())


class TupleIterWrapper:
    # [CLASS DOC] HELPER: iteration-only wrapper whose iterator yields each element doubled.
    """Tuple wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over the doubled elements of the data.
        # yields items doubled
        return iter([x * 2 for x in self.data])


class TupleGetitemWrapper:
    # [CLASS DOC] HELPER: getitem-only wrapper that returns each element shifted up by 100.
    """Tuple wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: return the element at idx plus 100.
        # shifts each value up by 100
        return self.data[idx] + 100


class SetIterWrapper:
    # [CLASS DOC] HELPER: iteration-only wrapper whose iterator yields each element doubled.
    """Set wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the wrapped data.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over the doubled elements of the data.
        # yields items doubled
        return iter([x * 2 for x in self.data])


class DictGetitemWrapper:
    # [CLASS DOC] HELPER: getitem-only wrapper over a dict; integer indices select values via
    # the sorted keys.
    """Dict wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: store the dict and a sorted list of its keys.
        self.data = data
        self.sorted_keys = sorted(data.keys())

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: return the dict value for the idx-th sorted key (IndexError
        # when out of range).
        # access values by index via sorted keys
        return self.data[self.sorted_keys[idx]]


class ListSubclassCustomContains(list):
    # [CLASS DOC] HELPER: list subclass whose __contains__ tests membership among
    # the doubled elements.
    """Subclass of list that overrides __contains__."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return whether item is in the list of doubled elements.
        return item in [x * 2 for x in super().__iter__()]


class DictSubclassCustomContains(dict):
    # [CLASS DOC] HELPER: dict subclass whose __contains__ tests membership among the values,
    # not the keys.
    """dict subclass whose __contains__ checks values instead of keys."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return whether item equals one of the dict values.
        return item in self.values()


class SetSubclassCustomContains(set):
    # [CLASS DOC] HELPER: set subclass whose __contains__ returns the negation of the
    # base set result.
    """set subclass whose __contains__ negates the base class result."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return the logical negation of the base
        # class membership result.
        return not super().__contains__(item)


class ContainsRaisesTypeError:
    # [CLASS DOC] HELPER: container whose __contains__ always raises TypeError.
    """__contains__ unconditionally raises TypeError."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: always raise TypeError with message "bad operand".
        raise TypeError("bad operand")


class ContainsReturnsTruthy:
    # [CLASS DOC] HELPER: container whose __contains__ returns the truthy non-bool value 42.
    """__contains__ returns a non-bool truthy value."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return 42 regardless of item.
        return 42


class ContainsReturnsFalsy:
    # [CLASS DOC] HELPER: container whose __contains__ returns the falsy non-bool value 0.
    """__contains__ returns a non-bool falsy value."""

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return 0 regardless of item.
        return 0


class NoIterNoContains:
    # [CLASS DOC] HELPER: object with neither __iter__ nor __contains__; membership
    # tests raise TypeError.
    """Has neither __iter__ nor __contains__ — triggers TypeError on `in`."""


class RaisesDuringIter:
    # [CLASS DOC] HELPER: iterable that yields 1 and 2 then raises ValueError.
    """Iterator that raises ValueError partway through."""

    def __iter__(self):
        # [FUNCTION DOC] HELPER: generator yielding 1, then 2, then raising
        # ValueError("mid-iteration error").
        yield 1
        yield 2
        raise ValueError("mid-iteration error")


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class _ContainsBase:
    # [CLASS DOC] HELPER: mixin of membership tests parameterized by class attributes:
    # thetype, data, empty, item, missing_item and the has_contains/has_iter/has_getitem
    # flags (expected dunder presence). The init method builds the container. Combined with
    # a TestCase in subclasses; setUp/tearDown enable unittest tracing.
    """Base test class for __contains__ protocol with parameterized types."""

    thetype = None  # Override in subclass
    data = [1, 2, 3]
    empty = []
    item = 2
    missing_item = 4
    has_contains = True  # Override in subclass if type doesn't implement __contains__
    has_iter = True  # Override in subclass if type doesn't implement __iter__
    has_getitem = True  # Override in subclass if type doesn't implement __getitem__

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable torch._dynamo.config.enable_trace_unittest,
        # then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def init(self, thetype, data):
        # [FUNCTION DOC] HELPER: build the container by calling thetype with data; overridden
        # for range.
        return thetype(data)

    @make_dynamo_test
    def test_contains(self):
        # [FUNCTION DOC] HELPER: under dynamo, item is in the container and missing_item is not.
        # Basic membership tests for the main type under test
        seq = self.init(self.thetype, self.data)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item in seq)

    @make_dynamo_test
    def test_contains_negation(self):
        # [FUNCTION DOC] HELPER: under dynamo, `not in` is False for item and
        # True for missing_item.
        # Test `not in` operator
        seq = self.init(self.thetype, self.data)
        self.assertFalse(self.item not in seq)
        self.assertTrue(self.missing_item not in seq)

    @make_dynamo_test
    def test_contains_operator_module(self):
        # [FUNCTION DOC] HELPER: under dynamo, operator.contains agrees with `in` for item
        # and missing_item.
        # Test operator.contains()
        seq = self.init(self.thetype, self.data)
        self.assertTrue(operator.contains(seq, self.item))
        self.assertFalse(operator.contains(seq, self.missing_item))

    @make_dynamo_test
    def test_contains_empty(self):
        # [FUNCTION DOC] HELPER: build the container from the empty data; neither item nor
        # missing_item is contained.
        # Test on empty container
        seq = self.init(self.thetype, self.empty)
        self.assertFalse(self.item in seq)
        self.assertTrue(self.missing_item not in seq)

    @make_dynamo_test
    def test_has_contains_method(self):
        # [FUNCTION DOC] HELPER: check hasattr for __contains__ matches the has_contains flag.
        # Verify whether the type has __contains__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__contains__")
        self.assertEqual(
            has_method,
            self.has_contains,
            lambda msg: f"{msg}\n{self.thetype.__name__} __contains__ presence mismatch",
        )

    @make_dynamo_test
    def test_has_iter_method(self):
        # [FUNCTION DOC] HELPER: check hasattr for __iter__ matches the has_iter flag.
        # Verify whether the type has __iter__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__iter__")
        self.assertEqual(
            has_method,
            self.has_iter,
            lambda msg: f"{msg}\n{self.thetype.__name__} __iter__ presence mismatch",
        )

    @make_dynamo_test
    def test_has_getitem_method(self):
        # [FUNCTION DOC] HELPER: check hasattr for __getitem__ matches the has_getitem flag.
        # Verify whether the type has __getitem__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__getitem__")
        self.assertEqual(
            has_method,
            self.has_getitem,
            lambda msg: f"{msg}\n{self.thetype.__name__} __getitem__ presence mismatch",
        )


class ListContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for list.
    thetype = list


class TupleContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for tuple.
    thetype = tuple


class StrContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for str
    # (substring membership).
    thetype = str
    data = "abc"
    item = "b"
    missing_item = "d"


class DictContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for dict (key membership).
    thetype = dict
    data = {"a": 1, "b": 2, "c": 3}
    item = "b"
    missing_item = "d"
    empty = {}


class MappingProxyContainsTest(DictContainsTest):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for types.MappingProxyType
    # over a dict.
    thetype = types.MappingProxyType


class SetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for set.
    thetype = set
    has_getitem = False


class FrozensetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for frozenset.
    thetype = frozenset
    has_getitem = False


class WithContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a class with explicit
    # __contains__ only.
    thetype = WithContains
    has_iter = False
    has_getitem = False


class WithIterNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a class with only
    # __iter__ (iteration fallback).
    thetype = WithIterNoContains
    has_contains = False
    has_getitem = False


class WithGetitemNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a class with only
    # __getitem__ (sequence-protocol fallback).
    thetype = WithGetitemNoContains
    has_contains = False
    has_iter = False


class WithContainsAndIterTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a class with both
    # __contains__ and __iter__.
    thetype = WithContainsAndIter
    has_getitem = False


class ListIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for an iteration-only
    # wrapper yielding doubled elements.
    thetype = ListIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 1
    has_contains = False
    has_getitem = False


class ListGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a getitem-only wrapper
    # shifting values by 100.
    thetype = ListGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class DictIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for an iteration-only
    # wrapper over dict values.
    thetype = DictIterWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will iterate over values
    missing_item = "a"
    has_contains = False
    has_getitem = False


class TupleIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for an iteration-only
    # tuple-style wrapper yielding doubled elements.
    thetype = TupleIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 10
    has_contains = False
    has_getitem = False


class TupleGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a getitem-only
    # tuple-style wrapper shifting values by 100.
    thetype = TupleGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class SetIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for an iteration-only
    # set-style wrapper yielding doubled elements.
    thetype = SetIterWrapper
    data = {1, 2, 3}
    item = 4  # Will be doubled by __iter__
    missing_item = 10
    has_contains = False
    has_getitem = False


class DictGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a getitem-only wrapper
    # over a dict.
    thetype = DictGetitemWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will be retrieved by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class ListSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a list subclass with
    # custom __contains__ (doubled elements).
    thetype = ListSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ checks doubled values
    missing_item = 10


class DictSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a dict subclass with
    # custom __contains__ (checks values).
    thetype = DictSubclassCustomContains
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Custom __contains__ checks values not keys
    missing_item = 10


class SetSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for a set subclass with
    # negating __contains__.
    thetype = SetSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ negates base class result
    missing_item = 1
    has_getitem = False

    @make_dynamo_test
    def test_contains_empty(self):
        # [FUNCTION DOC] With the negating set subclass and empty data, item IS reported
        # contained and `missing_item not in` is False (inverted vs. the base test).
        # Test on empty container
        seq = self.thetype(self.empty)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item not in seq)


class RangeContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared membership tests of _ContainsBase for range, built with the
    # data tuple as range arguments.
    thetype = range
    data = (0, 10, 2)
    item = 2
    missing_item = 10
    empty = (0, 0, 1)

    def init(self, thetype, data):
        # [FUNCTION DOC] build a range by unpacking data as its arguments (start, stop, step).
        return thetype(*data)


class ContainsNonBoolReturnTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that a non-bool __contains__ return value is coerced to a bool by `in`.
    """Test __contains__ that returns non-bool truthy/falsy values."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains_truthy(self):
        # [FUNCTION DOC] a truthy non-bool return (42) from __contains__ makes `in` true for any
        # operand.
        seq = ContainsReturnsTruthy()
        # Non-bool truthy return values should be coerced to True
        result1 = 2 in seq
        result2 = None in seq
        return result1 and result2

    @make_dynamo_test
    def test_contains_falsy(self):
        # [FUNCTION DOC] a falsy non-bool return (0) from __contains__ makes `in` false for
        # any operand.
        seq = ContainsReturnsFalsy()
        # Non-bool falsy return values should be coerced to False
        result1 = 2 in seq
        result2 = None in seq
        return not result1 and not result2


class NoIterNoContainsTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests membership on an object with neither __iter__ nor __contains__.
    """Tests for objects with neither __iter__ nor __contains__"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_no_iter_no_contains_raises_typeerror(self):
        # [FUNCTION DOC] `in` on such an object raises TypeError.
        # Should raise TypeError when neither __iter__ nor __contains__ exists
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 in obj

    @make_dynamo_test
    def test_no_iter_no_contains_not_in_raises_typeerror(self):
        # [FUNCTION DOC] `not in` on such an object raises TypeError.
        # Should raise TypeError for `not in` operator as well
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 not in obj


class RaisesDuringIterTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests membership over an iterator that raises ValueError after yielding 1 and
    # 2.
    """Tests for iterators that raise exceptions during iteration"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_raises_during_iter_found_before_error(self):
        # [FUNCTION DOC] membership of 1 succeeds (found before the error is raised).
        # Item 1 found before error occurs
        obj = RaisesDuringIter()
        result = 1 in obj
        self.assertTrue(result)

    @make_dynamo_test
    def test_raises_during_iter_not_found_raises_error(self):
        # [FUNCTION DOC] membership of 3 propagates the ValueError with message "mid-iteration
        # error".
        # Item not found, error occurs during iteration
        obj = RaisesDuringIter()
        with self.assertRaises(ValueError) as cm:
            _ = 3 in obj
        self.assertEqual(str(cm.exception), "mid-iteration error")


class ContainsRaisesTypeErrorTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that TypeError raised inside __contains__ propagates for `in` and `not
    # in`.
    """Tests for __contains__ that raises TypeError"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains_raises_typeerror(self):
        # [FUNCTION DOC] `in` propagates TypeError "bad operand".
        # __contains__ raises TypeError
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 in obj
        self.assertEqual(str(cm.exception), "bad operand")

    @make_dynamo_test
    def test_contains_raises_typeerror_not_in(self):
        # [FUNCTION DOC] `not in` propagates TypeError "bad operand".
        # __contains__ raises TypeError for `not in` operator
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 not in obj
        self.assertEqual(str(cm.exception), "bad operand")


class RangeContainsMiscTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests range membership: bounds, steps, negative values, empty range,
    # operator.contains.
    """Specific tests for range __contains__ protocol"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_range_basic(self):
        # [FUNCTION DOC] range(5): 2 in, 10 not in.
        # Basic range containment
        seq = range(5)
        self.assertTrue(2 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_with_start_stop(self):
        # [FUNCTION DOC] start/stop range: bounds exclusive of stop.
        # Test range with start and stop
        seq = range(5, 15)
        self.assertTrue(10 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(15 in seq)

    @make_dynamo_test
    def test_range_with_step(self):
        # [FUNCTION DOC] stepped range: only values on the step are members.
        # Test range with step
        seq = range(0, 10, 2)
        self.assertTrue(4 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_negative_step(self):
        # [FUNCTION DOC] negative step range.
        # Test range with negative step
        seq = range(10, 0, -1)
        self.assertTrue(5 in seq)
        self.assertFalse(0 in seq)
        self.assertFalse(11 in seq)

    @make_dynamo_test
    def test_range_empty(self):
        # [FUNCTION DOC] empty range contains nothing.
        # Test empty range
        seq = range(5, 5)
        self.assertFalse(5 in seq)
        self.assertFalse(4 in seq)

    @make_dynamo_test
    def test_range_single_element(self):
        # [FUNCTION DOC] single element range.
        # Test range with single element
        seq = range(5, 6)
        self.assertTrue(5 in seq)
        self.assertFalse(4 in seq)
        self.assertFalse(6 in seq)

    @make_dynamo_test
    def test_range_negative_numbers(self):
        # [FUNCTION DOC] range spanning negative numbers.
        # Test range with negative numbers
        seq = range(-5, 5)
        self.assertTrue(-3 in seq)
        self.assertTrue(0 in seq)
        self.assertFalse(-6 in seq)
        self.assertFalse(5 in seq)

    @make_dynamo_test
    def test_range_negation(self):
        # [FUNCTION DOC] `not in` on a range.
        # Test `not in` operator
        seq = range(5)
        self.assertFalse(2 not in seq)
        self.assertTrue(10 not in seq)

    @make_dynamo_test
    def test_range_operator_module(self):
        # [FUNCTION DOC] operator.contains on a range.
        # Test operator.contains()
        seq = range(10)
        self.assertTrue(operator.contains(seq, 5))
        self.assertFalse(operator.contains(seq, 15))


class SetInSetTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that an unhashable set is looked up as a frozenset in a set `in` check.
    """
    CPython's set.__contains__ converts an unhashable set key to frozenset and
    retries, so `{1, 2} in {frozenset({1, 2})}` returns True even though set is
    not hashable.  Ref: Objects/setobject.c::set_contains.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_set_in_frozenset_set_found(self):
        # [FUNCTION DOC] a set equal to a stored frozenset is found.
        s1 = {1, 2}
        s2 = {frozenset({1, 2}), frozenset({3, 4})}
        self.assertTrue(s1 in s2)

    @make_dynamo_test
    def test_set_in_frozenset_set_not_found(self):
        # [FUNCTION DOC] a set with no equal stored frozenset is not found.
        s1 = {5, 6}
        s2 = {frozenset({1, 2}), frozenset({3, 4})}
        self.assertFalse(s1 in s2)

    @make_dynamo_test
    def test_empty_set_in_frozenset_set(self):
        # [FUNCTION DOC] the empty set matches a stored empty frozenset.
        s2 = {frozenset(), frozenset({1})}
        self.assertTrue(set() in s2)

    @make_dynamo_test
    def test_set_not_in_operator(self):
        # [FUNCTION DOC] `not in` is False when the equivalent frozenset is stored.
        s1 = {1, 2}
        s2 = {frozenset({1, 2})}
        self.assertFalse(s1 not in s2)

    @make_dynamo_test
    def test_set_missing_not_in_operator(self):
        # [FUNCTION DOC] `not in` is True when no equivalent frozenset is stored.
        s1 = {9}
        s2 = {frozenset({1, 2})}
        self.assertTrue(s1 not in s2)


class DoNotTestEq(Exception):
    # [CLASS DOC] HELPER: sentinel exception raised by StopCompares.__eq__.
    pass


class StopCompares:
    # [CLASS DOC] HELPER: object whose equality comparison raises DoNotTestEq, to detect
    # comparisons.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: always raise DoNotTestEq.
        raise DoNotTestEq


class ContainsOrderTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that sequence membership compares in order, short-circuits, and
    # propagates __eq__ errors.
    """
    Sequence __contains__ must compare elements in order via
    PyObject_RichCompareBool(item, search, Py_EQ), short-circuiting on the first
    match.  A custom __eq__ with side effects (here, a short-circuiting raise)
    must be observable.  Ref: Objects/listobject.c::list_contains.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: save and enable enable_trace_unittest, then call base setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restore the saved enable_trace_unittest setting and call
        # base tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_contains_short_circuits_before_raising_eq(self):
        # [FUNCTION DOC] list membership stops at the first match so the raising __eq__ is never
        # called.
        # 1 matches the first element, so StopCompares().__eq__ is never invoked.
        self.assertTrue(1 in [1, StopCompares()])

    @make_dynamo_test
    def test_list_contains_propagates_raising_eq(self):
        # [FUNCTION DOC] list membership reaches the raising __eq__ and propagates DoNotTestEq.
        with self.assertRaises(DoNotTestEq):
            _ = 1 in [StopCompares(), 1]

    @make_dynamo_test
    def test_tuple_contains_propagates_raising_eq(self):
        # [FUNCTION DOC] tuple membership reaches the raising __eq__ and propagates DoNotTestEq.
        with self.assertRaises(DoNotTestEq):
            _ = 1 in (StopCompares(), 1)


@torch._dynamo.config.patch(enable_trace_unittest=True)
class DictViewContainsTest(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests membership on dict, dict.items() and dict.values(), including __eq__
    # errors.
    """
    Membership in dict views: d.keys()/d.items()/d.values().
    - keys: hashed key lookup (dictkeys_contains).
    - items: (key, value) lookup then PyObject_RichCompareBool on the stored
      value (dictitems_contains).
    - values: linear scan comparing each stored value (PySequence_Contains).
    Ref: Objects/dictobject.c.
    """

    @make_dynamo_test
    def test_keys_contains(self):
        # [FUNCTION DOC] key membership is True for an existing key and False otherwise.
        d = {"a": 1, "b": 2}
        self.assertTrue("a" in d)
        self.assertFalse("c" in d)

    @make_dynamo_test
    def test_items_contains(self):
        # [FUNCTION DOC] items membership requires matching key and value; wrong value or
        # missing key is False.
        d = {"a": 1, "b": 2}
        self.assertTrue(("a", 1) in d.items())
        # key present, value mismatch
        self.assertFalse(("a", 2) in d.items())
        # key absent
        self.assertFalse(("c", 1) in d.items())

    @make_dynamo_test
    def test_items_contains_non_pair(self):
        # [FUNCTION DOC] non-pair items (an int, a 3-tuple) are never members of items(); result
        # is False.
        # A hashable item that is not a (key, value) pair can never be in
        # items(); exercises the generic (non-2-tuple) path of
        # DictItemsVariable.sq_contains. CPython returns False.
        d = {"a": 1, "b": 2}
        self.assertFalse(5 in d.items())
        self.assertFalse((1, 2, 3) in d.items())

    @make_dynamo_test
    def test_values_contains(self):
        # [FUNCTION DOC] values membership is True for an existing value and False otherwise.
        d = {"a": 1, "b": 2}
        self.assertTrue(2 in d.values())
        self.assertFalse(3 in d.values())

    @make_dynamo_test
    def test_items_contains_value_eq_raises(self):
        # [FUNCTION DOC] items membership with a value whose __eq__ raises propagates
        # DoNotTestEq.
        # dictitems_contains compares the stored value via
        # PyObject_RichCompareBool(found, value, Py_EQ), so a stored value whose
        # __eq__ raises must propagate.
        d = {1: StopCompares()}
        with self.assertRaises(DoNotTestEq):
            _ = (1, 0) in d.items()

    @make_dynamo_test
    def test_values_contains_eq_raises(self):
        # [FUNCTION DOC] values membership with a value whose __eq__ raises propagates
        # DoNotTestEq.
        # `x in d.values()` scans values comparing each via
        # PyObject_RichCompareBool; a stored value whose __eq__ raises propagates.
        d = {1: StopCompares()}
        with self.assertRaises(DoNotTestEq):
            _ = 0 in d.values()


class ContainsNoneWithIter:
    # [CLASS DOC] HELPER: sets __contains__ to None while defining __iter__, so `in` must raise
    # TypeError.
    """__contains__ = None blocks the slot even though __iter__ exists."""

    __contains__ = None

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over 1, 2, 3.
        return iter([1, 2, 3])


class ContainsPropertyAttrError:
    # [CLASS DOC] HELPER: __contains__ is a property that raises AttributeError when accessed;
    # has __iter__.
    """Pre-3.14 slot_sq_contains PROPAGATES a bind-time AttributeError
    (unlike slot_tp_iter, which blanket-clears lookup errors)."""

    @property
    def __contains__(self):  # noqa: PLE0302
        # [FUNCTION DOC] HELPER: property getter that raises AttributeError("gone") on access.
        raise AttributeError("gone")

    def __iter__(self):
        # [FUNCTION DOC] HELPER: return an iterator over 1, 2, 3.
        return iter([1, 2, 3])


class ContainsNonBool:
    # [CLASS DOC] HELPER: defines __contains__ returning the non-bool value 2.
    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: return 2 regardless of item.
        return 2


@torch._dynamo.config.patch(enable_trace_unittest=True)
class TestSpecialMethodContainsRegressions(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests special-method lookup rules for __contains__ (None, raising property,
    # non-bool).
    """slot_sq_contains lookup semantics (Objects/typeobject.c)."""

    @make_dynamo_test
    def test_contains_none_is_not_a_container(self):
        # [FUNCTION DOC] `in` on an object with __contains__ set to None raises TypeError
        # mentioning "is not a container".
        raised = False
        try:
            _ = 1 in ContainsNoneWithIter()
        except TypeError as e:
            raised = True
            self.assertIn("is not a container", str(e))
        self.assertTrue(raised)

    # 3.14 slot_sq_contains clears a bind-time AttributeError and falls back to
    # iteration; <3.14 propagates it. Dynamo does not model the 3.14 clearing yet.
    @xfailIfPy314Plus
    @make_dynamo_test
    def test_contains_property_attribute_error_propagates(self):
        # [FUNCTION DOC] `in` propagates AttributeError from the __contains__ property.
        # Expected failure on Python 3.14+, where CPython falls back to iteration.
        with self.assertRaises(AttributeError):
            _ = 1 in ContainsPropertyAttrError()

    @make_dynamo_test
    def test_contains_nonbool_return_coerced_to_bool(self):
        # [FUNCTION DOC] a non-bool __contains__ return (2) is coerced to the bool True.
        self.assertIs(1 in ContainsNonBool(), True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
