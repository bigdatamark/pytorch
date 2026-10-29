# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for how Dynamo traces Python iteration protocols: iter and next on
# builtin containers and iterators, user-defined iterables and iterators, reversed dispatch,
# sequence-protocol fallback, enum iteration, special-method lookup edge cases for the iter
# and next slots, and mutation during iteration. Most tests use make_dynamo_test, which runs
# the body eagerly and then compiled with dynamo. Defines helper classes at module level; each
# test class enables the unittest tracing config in setUp.

import enum
import sys
import types
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    make_dynamo_test,
    xfailIfPy313AndEarlier,
)


class CustomIterable:
    # [CLASS DOC] HELPER: Test fixture; iterable that stores a data object and whose __iter__
    # returns a fresh iterator over it.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given data object in the data attribute.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns a new iterator over the stored data.
        return iter(self.data)


class CustomIterator:
    # [CLASS DOC] HELPER: Test fixture; iterator counting from 1 up to max_val, tracking
    # position in the current attribute.
    def __init__(self, max_val):
        # [FUNCTION DOC] HELPER: Stores the upper bound and starts the current counter at
        # zero.
        self.max_val = max_val
        self.current = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self, as an iterator should.
        return self

    def __next__(self):
        # [FUNCTION DOC] HELPER: Raises StopIteration once the counter reaches the bound;
        # otherwise increments the counter and returns the new value.
        if self.current >= self.max_val:
            raise StopIteration
        self.current += 1
        return self.current


class SequenceClass:
    # [CLASS DOC] HELPER: Test fixture with only the sequence protocol (indexing and length,
    # no __iter__), wrapping a data sequence.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given sequence in the data attribute.
        self.data = data

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns the element of the stored data at the given index.
        return self.data[index]

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the stored data.
        return len(self.data)


class CustomList(list):
    # [CLASS DOC] HELPER: Test fixture; list subclass whose __iter__ yields the elements in
    # reverse order.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the reversed list.
        # Return elements in reverse order
        return iter(reversed(self))


class CustomDict(dict):
    # [CLASS DOC] HELPER: Test fixture; dict subclass whose __iter__ yields keys sorted in
    # descending order.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the keys sorted in descending order.
        # Return keys in reverse order
        return iter(sorted(self.keys(), reverse=True))


class CustomSet(set):
    # [CLASS DOC] HELPER: Test fixture; set subclass whose __iter__ yields each element
    # doubled.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator that walks the base set iterator and yields twice
        # each element.
        # Return elements multiplied by 2
        for item in set.__iter__(self):
            yield item * 2


class CustomSetDefaultIter(set):
    # [CLASS DOC] HELPER: Test fixture; set subclass that keeps the inherited __iter__ but
    # whose constructor ignores its argument and always holds ten, twenty and thirty.
    def __init__(self, iterable) -> None:
        # [FUNCTION DOC] HELPER: Ignores the passed iterable and initializes the set with the
        # fixed values ten, twenty and thirty.
        super().__init__([10, 20, 30])


class CustomListDefaultIter(list):
    # [CLASS DOC] HELPER: Test fixture; empty list subclass that inherits the default list
    # iteration.
    pass


class WithReversed:
    # [CLASS DOC] HELPER: Test fixture; class defining __reversed__ over a stored data
    # sequence but no iteration or sequence protocol.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given sequence in the data attribute.
        self.data = data

    def __reversed__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the stored data in reverse order
        # using a reversed slice.
        return iter(self.data[::-1])


class SeqWithReversed:
    # [CLASS DOC] HELPER: Test fixture with both the sequence protocol and a __reversed__ that
    # returns an iterator over the single string custom; used to show __reversed__ takes
    # priority.
    """Has both sequence protocol AND __reversed__; __reversed__ should win."""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given sequence in the data attribute.
        self.data = data

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the stored data.
        return len(self.data)

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: Returns the element of the stored data at the given index.
        return self.data[idx]

    def __reversed__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over a list holding only the string
        # custom, ignoring the data.
        return iter(["custom"])


class NotReversible:
    # [CLASS DOC] HELPER: Test fixture; plain class with no __reversed__ and no sequence
    # protocol.
    pass


class RevDict(dict):
    # [CLASS DOC] HELPER: Test fixture; dict subclass overriding __reversed__ to return an
    # iterator over only the key custom_key.
    def __reversed__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over a one-element list holding the
        # string custom_key.
        return iter(["custom_key"])


class NotReversibleNone:
    # [CLASS DOC] HELPER: Test fixture; sets __reversed__ to None (the opt-out sentinel) while
    # still providing length three and indexing.
    """`__reversed__ = None` is CPython's opt-out sentinel."""

    __reversed__ = None

    def __len__(self):
        # [FUNCTION DOC] HELPER: Always returns three.
        return 3

    def __getitem__(self, idx):
        # [FUNCTION DOC] HELPER: Returns the index itself.
        return idx


class TestIterators(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case covering iteration over builtin containers and iterators under
    # Dynamo, including mutation during enumerate and zip, reversed dispatch, next with
    # defaults, and ConstantVariable tp_iter_impl handling. setUp and tearDown enable and
    # restore the unittest tracing config.
    """Test iterator support in Dynamo"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_list_iteration(self):
        # [FUNCTION DOC] Dynamo test: calls __iter__ on a list and loops, doubling each item;
        # checks the doubled list in order.
        """Test iteration over list"""
        lst = [1, 2, 3, 4, 5]
        result = []
        for item in lst.__iter__():
            result.append(item * 2)
        self.assertEqual(result, [2, 4, 6, 8, 10])

    @make_dynamo_test
    def test_tuple_iteration(self):
        # [FUNCTION DOC] Dynamo test: loops over the iterator of a tuple adding ten to each
        # element; checks the resulting list.
        """Test iteration over tuple"""
        tup = (1, 2, 3, 4, 5)
        result = []
        for item in tup.__iter__():
            result.append(item + 10)
        self.assertEqual(result, [11, 12, 13, 14, 15])

    @make_dynamo_test
    def test_set_iteration(self):
        # [FUNCTION DOC] Dynamo test: loops over the iterator of a set doubling each element;
        # compares the sorted result since set order is unspecified.
        """Test iteration over set"""
        s = {1, 2, 3, 4, 5}
        result = []
        for item in s.__iter__():
            result.append(item * 2)
        self.assertEqual(sorted(result), [2, 4, 6, 8, 10])

    @make_dynamo_test
    def test_mappingproxy_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates a mapping proxy of a three-key dict both via
        # __iter__ and via the iter builtin; each time the sorted keys must be a, b, c.
        m = types.MappingProxyType({"a": 1, "b": 2, "c": 3})
        result = []
        for key in m.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

        result = []
        for key in iter(m):
            result.append(key)
        self.assertEqual(sorted(result), ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_keys_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates the dict directly via __iter__ and collects
        # keys; sorted keys must match.
        """Test iteration over dict keys"""
        d = {"a": 1, "b": 2, "c": 3}
        result = []
        for key in d.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_container_types_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates a dict, its values view and its items view
        # through __iter__; checks sorted keys, doubled values and doubled key-value pairs.
        """Test iteration over dict and dict views"""
        d = {"a": 1, "b": 2, "c": 3}

        # Test dict keys
        result = []
        for key in d.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

        # Test dict values
        result = []
        for val in d.values().__iter__():
            result.append(val * 2)
        self.assertEqual(sorted(result), [2, 4, 6])

        # Test dict items
        result = []
        for key, val in d.items().__iter__():
            result.append((key, val * 2))
        self.assertEqual(sorted(result), [("a", 2), ("b", 4), ("c", 6)])

    @make_dynamo_test
    def test_range_iteration(self):
        # [FUNCTION DOC] Dynamo test: loops over a range of five multiplying each by three;
        # checks the result list.
        """Test iteration over range"""
        result = []
        for i in range(5):
            result.append(i * 3)
        self.assertEqual(result, [0, 3, 6, 9, 12])

    @make_dynamo_test
    def test_enumerate_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates enumerate via __iter__ and checks index/doubled
        # pairs; then appends to the list during iteration and checks the appended element is
        # eventually visited.
        """Test enumerate iterator, then with mutation"""
        lst = [10, 20, 30, 40]
        result = []
        # Explicitly call __iter__() on enumerate
        for idx, val in enumerate(lst).__iter__():
            result.append((idx, val * 2))
        self.assertEqual(result, [(0, 20), (1, 40), (2, 60), (3, 80)])

        # Test with mutation
        lst2 = [10, 20, 30]
        result2 = []
        count = 0
        for idx, val in enumerate(lst2).__iter__():
            result2.append((idx, val))
            count += 1  # noqa: SIM113
            if count == 1:
                lst2.append(40)
        self.assertTrue(any(val == 40 for _, val in result2))

    @make_dynamo_test
    def test_enumerate_with_start(self):
        # [FUNCTION DOC] Dynamo test: enumerate with start one over three strings, iterated
        # via __iter__; checks the (index, value) pairs.
        """Test enumerate with start parameter"""
        lst = ["a", "b", "c"]
        result = []
        # Explicitly call __iter__() on enumerate with start
        for idx, val in enumerate(lst, start=1).__iter__():
            result.append((idx, val))
        self.assertEqual(result, [(1, "a"), (2, "b"), (3, "c")])

    @make_dynamo_test
    def test_zip_iteration(self):
        # [FUNCTION DOC] Dynamo test: zips two lists via __iter__ and checks pairs with second
        # element doubled; then appends to the first list mid-iteration and only checks at
        # least three pairs were produced.
        """Test zip iterator with two lists, then with mutation"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20, 30]
        result = []
        # Explicitly call __iter__() on zip
        for a, b in zip(lst1, lst2).__iter__():
            result.append((a, b * 2))
        self.assertEqual(result, [(1, 20), (2, 40), (3, 60)])

        # Test with mutation
        lst1_mut = [1, 2, 3]
        lst2_mut = [10, 20, 30]
        result2 = []
        count = 0
        for a, b in zip(lst1_mut, lst2_mut).__iter__():
            result2.append((a, b))
            count += 1  # noqa: SIM113
            if count == 1:
                lst1_mut.append(4)
        self.assertTrue(len(result2) >= 3)

    @make_dynamo_test
    def test_zip_three_iterables(self):
        # [FUNCTION DOC] Dynamo test: zips three lists via __iter__ and sums each triple;
        # checks the sums.
        """Test zip with three iterables"""
        lst1 = [1, 2]
        lst2 = [10, 20]
        lst3 = [100, 200]
        result = []
        # Explicitly call __iter__() on zip
        for a, b, c in zip(lst1, lst2, lst3).__iter__():
            result.append(a + b + c)
        self.assertEqual(result, [111, 222])

    @make_dynamo_test
    def test_nested_iteration(self):
        # [FUNCTION DOC] Dynamo test: nested loops over two lists using __iter__ for both;
        # checks the product list in order.
        """Test nested for loops with iterators"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20]
        result = []
        # Explicitly call __iter__() on both lists
        for a in lst1.__iter__():
            for b in lst2.__iter__():
                result.append(a * b)
        self.assertEqual(result, [10, 20, 20, 40, 30, 60])

    @make_dynamo_test
    def test_break_in_iteration(self):
        # [FUNCTION DOC] Dynamo test: breaks out of a loop over a list iterator once an item
        # exceeds three; checks only the earlier items were collected.
        """Test break statement in iteration"""
        lst = [1, 2, 3, 4, 5]
        result = []
        # Explicitly call __iter__()
        for item in lst.__iter__():
            if item > 3:
                break
            result.append(item)
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_continue_in_iteration(self):
        # [FUNCTION DOC] Dynamo test: skips even items with continue while iterating a list
        # iterator; checks only odd items remain.
        """Test continue statement in iteration"""
        lst = [1, 2, 3, 4, 5]
        result = []
        # Explicitly call iter()
        for item in lst.__iter__():
            if item % 2 == 0:
                continue
            result.append(item)
        self.assertEqual(result, [1, 3, 5])

    @make_dynamo_test
    def test_iterator_with_default_value(self):
        # [FUNCTION DOC] Dynamo test: two-argument iter with a callable and a sentinel. The
        # nested get_next increments a counter and returns it; the loop must stop before the
        # sentinel three, giving one and two.
        """Test iter() with callable and default value"""
        counter = {"i": 0}

        def get_next():
            # [FUNCTION DOC] HELPER: Increments the counter stored in the enclosing dict and
            # returns the new value.
            counter["i"] += 1
            return counter["i"]

        result = []
        for val in iter(get_next, 3):  # iter(callable, sentinel) form requires iter()
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [1, 2])

    @make_dynamo_test
    def test_iter_idempotency(self):
        # [FUNCTION DOC] Dynamo test: the iterator of a list iterator is that same object
        # (identity check).
        """Test that iter(iter(x)) is the same as iter(x)"""
        lst = [1, 2, 3]
        it1 = lst.__iter__()
        it2 = it1.__iter__()
        self.assertIs(it1, it2)

    @make_dynamo_test
    def test_multiple_independent_iterators(self):
        # [FUNCTION DOC] Dynamo test: two iterators on one list advance independently (pairs
        # of first and second elements); then with a list appended to after partial
        # consumption, both iterators see the appended element in their remaining items.
        """Test multiple independent iterators on same container, then with mutation"""
        lst = [10, 20, 30, 40]
        it1 = lst.__iter__()
        it2 = lst.__iter__()

        result = []
        val1 = next(it1)
        val2 = next(it2)
        result.append((val1, val2))

        val1 = next(it1)
        val2 = next(it2)
        result.append((val1, val2))

        self.assertEqual(result, [(10, 10), (20, 20)])

        # Test with mutation
        lst_mut = [1, 2, 3]
        it1_mut = lst_mut.__iter__()
        it2_mut = lst_mut.__iter__()

        val1 = next(it1_mut)
        self.assertEqual(val1, 1)
        lst_mut.append(4)

        val1_next = next(it1_mut)
        val2_first = next(it2_mut)
        self.assertEqual(val1_next, 2)
        self.assertEqual(val2_first, 1)

        rest1 = list(it1_mut)
        rest2 = list(it2_mut)
        self.assertIn(4, rest1)
        self.assertIn(4, rest2)

    @make_dynamo_test
    def test_string_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates a string via __iter__ uppercasing characters;
        # joined result must be the uppercased string.
        """Test iteration over string"""
        s = "hello"
        result = []
        for char in s.__iter__():
            result.append(char.upper())
        self.assertEqual("".join(result), "HELLO")

    @make_dynamo_test
    def test_bytes_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates a bytes object via __iter__ doubling each
        # integer byte value; compares with doubled ordinals of the characters.
        """Test iteration over bytes"""
        b = b"hello"
        result = []
        for byte_val in b.__iter__():
            result.append(byte_val * 2)
        # Each byte value is multiplied by 2
        expected = [
            ord("h") * 2,
            ord("e") * 2,
            ord("l") * 2,
            ord("l") * 2,
            ord("o") * 2,
        ]
        self.assertEqual(result, expected)

    def test_constant_variable_list_tp_iter(self):
        # [FUNCTION DOC] Not a compiled test. Builds a ConstantVariable by bypassing its
        # constructor, initializing the base VariableTracker and setting the value to a list
        # of three ints, then calls tp_iter_impl with no tx. Asserts the result is a
        # ListIteratorVariable whose items convert back to the original constants.
        """ConstantVariable wrapping a list value should iterate via tp_iter_impl.

        ConstantVariable.create() normally redirects list to ListVariable. This
        test bypasses __init__ to construct a ConstantVariable holding a list
        directly, exercising the generic tp_iter_impl unpack path.
        """
        from torch._dynamo.variables.base import VariableTracker
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.lists import ListIteratorVariable

        cv = ConstantVariable.__new__(ConstantVariable)
        VariableTracker.__init__(cv)
        cv.value = [1, 2, 3]

        result = cv.tp_iter_impl(None)
        self.assertIsInstance(result, ListIteratorVariable)
        self.assertEqual([v.as_python_constant() for v in result.items], [1, 2, 3])

    def test_constant_variable_tuple_tp_iter(self):
        # [FUNCTION DOC] Same as the list variant but the constant value is a tuple of three
        # ints; tp_iter_impl must return a ListIteratorVariable with the matching constants.
        """ConstantVariable wrapping a tuple value iterates via tp_iter_impl."""
        from torch._dynamo.variables.base import VariableTracker
        from torch._dynamo.variables.constant import ConstantVariable
        from torch._dynamo.variables.lists import ListIteratorVariable

        cv = ConstantVariable.__new__(ConstantVariable)
        VariableTracker.__init__(cv)
        cv.value = (10, 20, 30)

        result = cv.tp_iter_impl(None)
        self.assertIsInstance(result, ListIteratorVariable)
        self.assertEqual([v.as_python_constant() for v in result.items], [10, 20, 30])

    def test_constant_variable_non_iterable_tp_iter(self):
        # [FUNCTION DOC] A ConstantVariable (built without its constructor) holding an int;
        # tp_iter_impl must raise Unsupported, the base fallback.
        """Non-iterable ConstantVariable falls back to base unimplemented."""
        from torch._dynamo.exc import Unsupported
        from torch._dynamo.variables.base import VariableTracker
        from torch._dynamo.variables.constant import ConstantVariable

        cv = ConstantVariable.__new__(ConstantVariable)
        VariableTracker.__init__(cv)
        cv.value = 42

        with self.assertRaises(Unsupported):
            cv.tp_iter_impl(None)

    @make_dynamo_test
    def test_comprehensions_with_iterator(self):
        # [FUNCTION DOC] Dynamo test: list, dict and set comprehensions over a list iterator
        # obtained via __iter__; checks each result.
        """Test different comprehension types with iterators"""
        lst = [1, 2, 3, 4, 5]

        # List comprehension
        result = [x * 2 for x in lst.__iter__()]
        self.assertEqual(result, [2, 4, 6, 8, 10])

        # Dict comprehension
        result = {x: x * 2 for x in lst.__iter__()}
        self.assertEqual(result, {1: 2, 2: 4, 3: 6, 4: 8, 5: 10})

        # Set comprehension
        result = {x % 2 for x in lst.__iter__()}
        self.assertEqual(result, {0, 1})

    @make_dynamo_test
    def test_generator_expression(self):
        # [FUNCTION DOC] Dynamo test: builds a generator expression over a list iterator,
        # consumes it with list, and checks doubled values.
        """Test generator expression (iterator)"""
        lst = [1, 2, 3, 4]
        # Generator expression implicitly calls iter()
        gen = (x * 2 for x in lst.__iter__())
        result = list(gen)
        self.assertEqual(result, [2, 4, 6, 8])

    @make_dynamo_test
    def test_filter_iterator(self):
        # [FUNCTION DOC] Dynamo test: filter with a greater-than-two predicate over a list
        # iterator; checks the surviving elements.
        """Test filter built-in iterator"""
        lst = [1, 2, 3, 4, 5]
        # Explicitly call iter() on filter result
        filtered = filter(lambda x: x > 2, lst.__iter__())
        result = list(filtered)
        self.assertEqual(result, [3, 4, 5])

    @make_dynamo_test
    def test_map_iterator(self):
        # [FUNCTION DOC] Dynamo test: map multiplying by three over a list iterator, consumed
        # via list; checks values.
        """Test map built-in iterator"""
        lst = [1, 2, 3, 4]
        # Explicitly call iter() on map result
        mapped = map(lambda x: x * 3, lst.__iter__())  # noqa: C417
        result = list(mapped)
        self.assertEqual(result, [3, 6, 9, 12])

    @make_dynamo_test
    def test_reversed_iterator(self):
        # [FUNCTION DOC] Dynamo test: reversed of a list, then __iter__ on it; checks the
        # elements come out in reverse.
        """Test reversed iterator"""
        lst = [1, 2, 3, 4, 5]
        # Explicitly call __iter__() on reversed result
        rev = reversed(lst).__iter__()
        result = list(rev)
        self.assertEqual(result, [5, 4, 3, 2, 1])

    @make_dynamo_test
    def test_reversed_user_defined_reversed(self):
        # [FUNCTION DOC] Dynamo test: reversed on an object defining __reversed__ dispatches
        # to it; list result is the reversed data.
        """reversed() dispatches to user-defined __reversed__"""
        self.assertEqual(list(reversed(WithReversed([1, 2, 3, 4]))), [4, 3, 2, 1])

    @make_dynamo_test
    def test_reversed_user_defined_sequence_protocol(self):
        # [FUNCTION DOC] Dynamo test: reversed on a class with only length and indexing falls
        # back to the sequence protocol and yields the data backwards.
        """reversed() on user class with __len__ + __getitem__ but no __reversed__"""
        seq = SequenceClass([10, 20, 30, 40])
        self.assertEqual(list(reversed(seq)), [40, 30, 20, 10])

    @make_dynamo_test
    def test_reversed_user_defined_reversed_overrides_sequence(self):
        # [FUNCTION DOC] Dynamo test: a class with both sequence protocol and __reversed__:
        # the result is the custom iterator, a single string custom.
        """User-defined __reversed__ wins over the sequence protocol fallback"""
        self.assertEqual(list(reversed(SeqWithReversed([1, 2, 3]))), ["custom"])

    @make_dynamo_test
    def test_reversed_user_defined_not_reversible(self):
        # [FUNCTION DOC] Dynamo test: reversed on an object with neither __reversed__ nor
        # sequence protocol must raise TypeError.
        """User class with neither __reversed__ nor sequence protocol -> TypeError"""
        with self.assertRaises(TypeError):
            reversed(NotReversible())

    @make_dynamo_test
    def test_reversed_user_defined_dict_subclass(self):
        # [FUNCTION DOC] Dynamo test: reversed on a dict subclass overriding __reversed__
        # yields the overridden result, only custom_key.
        """dict subclass with overridden __reversed__ is dispatched to"""
        d = RevDict(a=1, b=2)
        self.assertEqual(list(reversed(d)), ["custom_key"])

    @make_dynamo_test
    def test_reversed_range(self):
        # [FUNCTION DOC] Dynamo test: reversed over ranges (simple, with step, and empty)
        # gives the expected descending lists.
        """reversed(range(...)) returns a reverse range_iterator"""
        self.assertEqual(list(reversed(range(5))), [4, 3, 2, 1, 0])
        self.assertEqual(list(reversed(range(2, 10, 3))), [8, 5, 2])
        self.assertEqual(list(reversed(range(0))), [])

    @make_dynamo_test
    def test_reversed_mappingproxy(self):
        # [FUNCTION DOC] Dynamo test: reversed over a mapping proxy of a three-key dict yields
        # the keys in reverse insertion order.
        """reversed() on types.MappingProxyType dispatches to dict.__reversed__"""
        mp = types.MappingProxyType({"a": 1, "b": 2, "c": 3})
        self.assertEqual(list(reversed(mp)), ["c", "b", "a"])

    @make_dynamo_test
    def test_reversed_none_sentinel(self):
        # [FUNCTION DOC] Dynamo test: reversed on a class whose __reversed__ is None must
        # raise TypeError even though it has indexing and length.
        """`__reversed__ = None` opts out, even when sequence protocol exists"""
        with self.assertRaises(TypeError):
            reversed(NotReversibleNone())

    @make_dynamo_test
    def test_next_with_default(self):
        # [FUNCTION DOC] Dynamo test: exhausts a list iterator with next, then the next
        # builtin with a default returns the default string instead of raising.
        """Test next() with default value"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        # Exhaust the iterator
        for _ in lst:
            next(it)
        # Try to get next with default
        result = next(it, "EMPTY")
        self.assertEqual(result, "EMPTY")

    # Corner case tests

    @make_dynamo_test
    def test_iterator_exhaustion(self):
        # [FUNCTION DOC] Dynamo test: the next builtin on an exhausted two-element list
        # iterator raises StopIteration.
        """Test that exhausted iterator raises StopIteration"""
        lst = [1, 2]
        it = lst.__iter__()
        next(it)
        next(it)
        # Next call on exhausted iterator should raise StopIteration
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_empty_and_single_item_containers(self):
        # [FUNCTION DOC] Dynamo test: iterating empty list, dict, set, tuple gives empty
        # lists; single-element containers give a one-element list.
        """Test iteration over empty and single-item containers"""
        # Empty containers
        self.assertEqual(list([].__iter__()), [])
        self.assertEqual(list({}.__iter__()), [])
        self.assertEqual(list(set().__iter__()), [])
        self.assertEqual(list(().__iter__()), [])

        # Single item containers
        self.assertEqual(list([42].__iter__()), [42])
        self.assertEqual(list((42,).__iter__()), [42])
        self.assertEqual(list({42}.__iter__()), [42])

    @make_dynamo_test
    def test_custom_iterator_class(self):
        # [FUNCTION DOC] Dynamo test: iter on a user-defined iterator class counting to three;
        # loop collects one, two, three.
        """Test custom class implementing iterator protocol"""

        it = CustomIterator(3)
        result = []
        for val in iter(it):
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_iterable_class(self):
        # [FUNCTION DOC] Dynamo test: calls __iter__ on a user iterable and materializes it;
        # checks the data values.
        """Test custom class with __iter__ returning different object"""
        obj = CustomIterable([10, 20, 30])
        result = list(obj.__iter__())
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_iter_idempotency_callable(self):
        # [FUNCTION DOC] Dynamo test: same as the idempotency test: __iter__ of a list
        # iterator returns the very same object.
        """Test iter(iter(x)) returns same iterator"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        it2 = it.__iter__()
        # Both should be the same object
        self.assertIs(it, it2)

    @make_dynamo_test
    def test_zip_unequal_lengths(self):
        # [FUNCTION DOC] Dynamo test: zip of a five-element list with the iterator of a two-
        # element list stops at the shorter one.
        """Test zip stops at shortest iterable"""
        lst1 = [1, 2, 3, 4, 5]
        lst2 = [10, 20]
        result = list(zip(lst1, lst2.__iter__()))
        # Should stop at shortest (lst2)
        self.assertEqual(result, [(1, 10), (2, 20)])

    @make_dynamo_test
    def test_enumerate_with_large_start(self):
        # [FUNCTION DOC] Dynamo test: enumerate starting at one hundred, iterated via
        # __iter__; checks indices one hundred and one hundred and one.
        """Test enumerate starting at a large index"""
        lst = ["a", "b"]
        result = list(enumerate(lst, start=100).__iter__())
        self.assertEqual(result, [(100, "a"), (101, "b")])

    @make_dynamo_test
    def test_for_else_with_break(self):
        # [FUNCTION DOC] Dynamo test: for-else: breaking at item three means the else branch
        # must not run; checks collected items and the flag.
        """Test for/else: else does not run with break"""
        lst = [1, 2, 3, 4, 5]
        result = []
        else_executed = False
        for item in lst.__iter__():
            if item == 3:
                break
            result.append(item)
        else:
            else_executed = True
        self.assertEqual(result, [1, 2])
        self.assertFalse(else_executed)

    @make_dynamo_test
    def test_for_else_without_break(self):
        # [FUNCTION DOC] Dynamo test: for-else: a loop that completes without break executes
        # the else branch; checks items and flag.
        """Test for/else: else runs when loop completes naturally"""
        lst = [1, 2, 3]
        result = []
        else_executed = False
        for item in lst.__iter__():
            result.append(item)  # noqa: PERF402
        else:
            else_executed = True
        self.assertEqual(result, [1, 2, 3])
        self.assertTrue(else_executed)

    @make_dynamo_test
    def test_generator_function(self):
        # [FUNCTION DOC] Dynamo test: a nested generator function yields one, two, three and
        # is consumed through __iter__ and list.
        """Test generator function as iterator"""

        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding the integers one, two and three.
            yield 1
            yield 2
            yield 3

        result = list(gen().__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_iterator_unpacking(self):
        # [FUNCTION DOC] Dynamo test: unpacks a three-element list iterator into three
        # variables and checks each value.
        """Test unpacking values from an iterator"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        a, b, c = it
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)

    @make_dynamo_test
    def test_dict_value_mutation_allowed(self):
        # [FUNCTION DOC] Dynamo test: changing a value (not the key set) of a dict during
        # iteration is allowed; the remaining two keys are still produced and exclude the
        # first key.
        """Test that modifying dict values during iteration is allowed"""
        d = {"a": 1, "b": 2, "c": 3}
        it = d.__iter__()
        first_key = next(it)
        # Modify a value (not adding/removing keys)
        d[first_key] = 99
        # Should be able to continue iteration
        rest = list(it)
        self.assertEqual(len(rest), 2)
        self.assertNotIn(first_key, rest)

    @make_dynamo_test
    def test_list_clear_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: clearing a list after taking its first element; the
        # iterator continues without error and the rest must be a list (contents deliberately
        # unchecked).
        """Test that clearing a list during iteration works"""
        lst = [1, 2, 3, 4, 5]
        it = lst.__iter__()
        first = next(it)
        self.assertEqual(first, 1)
        # Clear the list
        lst.clear()
        # Iteration should continue but see the cleared state
        rest = list(it)
        # After clear, remaining elements might be empty or implementation-dependent
        self.assertIsInstance(rest, list)

    @make_dynamo_test
    def test_range_iterator(self):
        # [FUNCTION DOC] Dynamo test: iterates a range from three to eight step two via
        # __iter__; yields three, five, seven.
        """Test range object as iterator"""
        r = range(3, 8, 2)
        result = list(r.__iter__())
        self.assertEqual(result, [3, 5, 7])

    @make_dynamo_test
    def test_reversed_on_list(self):
        # [FUNCTION DOC] Dynamo test: reversed of a list followed by __iter__ gives the
        # elements reversed.
        """Test reversed() returns iterator"""
        lst = [1, 2, 3]
        result = list(reversed(lst).__iter__())
        self.assertEqual(result, [3, 2, 1])

    @make_dynamo_test
    def test_iter_sequence_protocol(self):
        # [FUNCTION DOC] Dynamo test: a class with only indexing and length is iterable by for
        # loop through the legacy sequence protocol.
        """Test iteration using sequence protocol (__getitem__)"""

        seq = SequenceClass([10, 20, 30])
        # Even without __iter__, should be iterable via __getitem__
        result = []
        for val in seq:
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_filter_empty_result(self):
        # [FUNCTION DOC] Dynamo test: filter whose predicate rejects everything produces an
        # empty list.
        """Test filter that produces empty result"""
        lst = [1, 2, 3, 4, 5]
        filtered = filter(lambda x: x > 10, lst.__iter__())
        result = list(filtered)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_map_empty_input(self):
        # [FUNCTION DOC] Dynamo test: map over an empty list iterator produces an empty list.
        """Test map on empty input"""
        lst = []
        mapped = map(lambda x: x * 2, lst.__iter__())  # noqa: C417
        result = list(mapped)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_nested_generators(self):
        # [FUNCTION DOC] Dynamo test: flattening a nested list with a two-level comprehension
        # using __iter__ at both levels.
        """Test nested generator expressions"""
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = [x for row in matrix.__iter__() for x in row.__iter__()]
        self.assertEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    @make_dynamo_test
    def test_iter_two_arg_callable_sentinel(self):
        # [FUNCTION DOC] Dynamo test: two-argument iter stops when the callable returns four;
        # nested get_next increments a counter dict; result is one, two, three.
        """Test iter(callable, sentinel) form"""
        counter = {"val": 0}

        def get_next():
            # [FUNCTION DOC] HELPER: Increments the counter in the enclosing dict and returns
            # the new value.
            counter["val"] += 1
            return counter["val"]

        # Stop when we get 4
        result = list(iter(get_next, 4))  # iter(callable, sentinel) requires iter()
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_zip_with_empty(self):
        # [FUNCTION DOC] Dynamo test: zip with an empty iterator yields nothing.
        """Test zip with one empty iterable"""
        lst1 = [1, 2, 3]
        lst2 = []
        result = list(zip(lst1, lst2.__iter__()))
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_dict_keys_view_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates the keys view via __iter__; sorted keys must
        # match.
        """Test iteration over dict.keys() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.keys()
        keys = d.keys().__iter__()
        result = sorted(keys)
        self.assertEqual(result, ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_values_view_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates the values view via __iter__; sorted values
        # one, two, three.
        """Test iteration over dict.values() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.values()
        values = d.values().__iter__()
        result = sorted(values)
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_dict_items_view_iteration(self):
        # [FUNCTION DOC] Dynamo test: iterates the items view via __iter__; sorted pairs must
        # match.
        """Test iteration over dict.items() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.items()
        items = d.items().__iter__()
        result = sorted(items)
        self.assertEqual(result, [("a", 1), ("b", 2), ("c", 3)])

    @make_dynamo_test
    def test_custom_list_subclass_with_custom_iter(self):
        # [FUNCTION DOC] Dynamo test: a list subclass with reversing __iter__: calling
        # __iter__ explicitly yields reversed elements.
        """Test custom list subclass that overloads __iter__"""

        cl = CustomList([1, 2, 3])
        result = list(cl.__iter__())
        self.assertEqual(result, [3, 2, 1])

    @make_dynamo_test
    def test_custom_dict_subclass_with_custom_iter(self):
        # [FUNCTION DOC] Dynamo test: a dict subclass with descending-key __iter__: explicit
        # iteration yields c, b, a.
        """Test custom dict subclass that overloads __iter__"""

        cd = CustomDict({"a": 1, "b": 2, "c": 3})
        result = list(cd.__iter__())
        self.assertEqual(result, ["c", "b", "a"])

    @make_dynamo_test
    def test_custom_set_subclass_with_custom_iter(self):
        # [FUNCTION DOC] Dynamo test: a set subclass whose __iter__ doubles elements; sorted
        # result is two, four, six.
        """Test custom set subclass that overloads __iter__"""

        cs = CustomSet([1, 2, 3])
        result = sorted(cs.__iter__())
        self.assertEqual(result, [2, 4, 6])

    @make_dynamo_test
    def test_custom_list_subclass_with_default_iter(self):
        # [FUNCTION DOC] Dynamo test: a list subclass without override iterates like a list
        # through both iter and a for loop (second pass doubles values).
        """Test custom list subclass with default __iter__ (inherited from list)"""

        cl = CustomListDefaultIter([1, 2, 3])
        result = list(iter(cl))
        self.assertEqual(result, [1, 2, 3])

        result2 = []
        for v in cl:
            result2.append(v * 2)
        self.assertEqual(result2, [2, 4, 6])

    @make_dynamo_test
    def test_custom_set_subclass_with_default_iter(self):
        # [FUNCTION DOC] Dynamo test: a set subclass whose constructor forces values ten,
        # twenty, thirty iterates with the default set iteration; the passed arguments are
        # ignored.
        """Test custom set subclass with default __iter__"""

        cs = CustomSetDefaultIter([1, 2, 3])
        result = sorted(iter(cs))
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_dict_keys_view_direct_iteration(self):
        # [FUNCTION DOC] Dynamo test: list of a dict gives its keys; compares sorted. Despite
        # the name there is no keys view involved.
        """Test direct for loop over dict.keys() view"""
        d = {"x": 10, "y": 20, "z": 30}
        result = list(d)
        self.assertEqual(sorted(result), ["x", "y", "z"])

    @make_dynamo_test
    def test_dict_values_view_in_comprehension(self):
        # [FUNCTION DOC] Dynamo test: list comprehension doubling dict values; sorted result.
        """Test dict.values() in list comprehension"""
        d = {"a": 1, "b": 2, "c": 3}
        result = [v * 2 for v in d.values()]
        self.assertEqual(sorted(result), [2, 4, 6])

    @make_dynamo_test
    def test_dict_items_view_in_comprehension(self):
        # [FUNCTION DOC] Dynamo test: list comprehension over dict items doubling values;
        # sorted pairs.
        """Test dict.items() in list comprehension"""
        d = {"a": 1, "b": 2, "c": 3}
        result = [(k, v * 2) for k, v in d.items()]
        self.assertEqual(sorted(result), [("a", 2), ("b", 4), ("c", 6)])


# Define custom subclasses outside of test functions
class CustomListWithReverseIter(list):
    # [CLASS DOC] HELPER: Test fixture; list subclass whose __iter__ is a generator yielding
    # the constants one, two, three regardless of content.
    """Custom list that iterates in reverse order"""

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding one, two and three. KNOWN ISSUE: the class
        # docstring and a commented-out line suggest reverse iteration, but it ignores
        # contents.
        # return iter(reversed(self))
        yield 1
        yield 2
        yield 3


class CustomTupleWithDoubleIter(tuple):
    # [CLASS DOC] HELPER: Test fixture; tuple subclass with empty slots whose __iter__ yields
    # every element twice.
    """Custom tuple that yields each element twice"""

    __slots__ = ()

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator that walks the base tuple iterator and yields each
        # item two times.
        for item in tuple.__iter__(self):
            yield item
            yield item


class CustomSetWithFilteredIter(set):
    # [CLASS DOC] HELPER: Test fixture; set subclass whose __iter__ yields only even elements.
    """Custom set that only iterates even numbers"""

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator over the base set iterator that yields only items
        # divisible by two.
        for item in set.__iter__(self):
            if item % 2 == 0:
                yield item


class CustomSetWith123Iter(set):
    # [CLASS DOC] HELPER: Test fixture; set subclass whose __iter__ yields one, two, three
    # regardless of content.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding one, two and three.
        yield 1
        yield 2
        yield 3


class CustomDictWithKeyLengthIter(dict):
    # [CLASS DOC] HELPER: Test fixture; dict subclass whose __iter__ yields only keys whose
    # string form is longer than one character.
    """Custom dict that only iterates keys with length > 1"""

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator over the base dict iterator yielding keys with
        # string length above one.
        for key in dict.__iter__(self):
            if len(str(key)) > 1:
                yield key


class CustomObjectWithCustomIter:
    # [CLASS DOC] HELPER: Test fixture; plain class whose __iter__ is a generator yielding
    # one, two, three.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding one, two and three.
        yield 1
        yield 2
        yield 3


class GeneratorIterIterable:
    # [CLASS DOC] HELPER: Test fixture; iterable whose __iter__ delegates to the stored data
    # with yield from.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the given data in the data attribute.
        self.data = data

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Generator yielding everything from the stored data via
        # delegation.
        yield from self.data


class TestCustomIteratorMethods(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case verifying Dynamo honors user-defined __iter__ overrides on list,
    # tuple, set, dict and plain-object subclasses. setUp and tearDown enable and restore the
    # unittest tracing config.
    """Test custom __iter__ implementations on user-defined subclasses"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_custom_object_with_custom_iter(self):
        # [FUNCTION DOC] Dynamo test: an empty instance of the list subclass with generator
        # __iter__ produces one, two, three.
        obj = CustomListWithReverseIter()
        result = list(obj.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_list_with_reverse_iter(self):
        # [FUNCTION DOC] Dynamo test: a five-element instance of the same list subclass still
        # yields only one, two, three, showing the override ignores contents.
        lst = CustomListWithReverseIter([1, 2, 3, 4, 5])
        result = list(lst.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_tuple_with_double_iter(self):
        # [FUNCTION DOC] Dynamo test: a tuple subclass built from a list; explicit iteration
        # repeats every element twice.
        """Test custom tuple that yields each element twice"""
        tup = CustomTupleWithDoubleIter([10, 20, 30])
        result = list(tup.__iter__())
        self.assertEqual(result, [10, 10, 20, 20, 30, 30])

    @make_dynamo_test
    def test_custom_set_with_filtered_iter(self):
        # [FUNCTION DOC] Dynamo test: set subclass with even-only iteration over one to six;
        # sorted result is two, four, six.
        """Test custom set that only iterates even numbers"""
        s = CustomSetWithFilteredIter([1, 2, 3, 4, 5, 6])
        result = sorted(s.__iter__())
        self.assertEqual(result, [2, 4, 6])

    @make_dynamo_test
    def test_custom_set_with_123_iter(self):
        # [FUNCTION DOC] Dynamo test: set subclass with constant iteration; sorted result is
        # one, two, three. KNOWN ISSUE: the docstring is copied from the even-numbers test and
        # is wrong.
        """Test custom set that only iterates even numbers"""
        s = CustomSetWith123Iter([1, 2, 3, 4, 5, 6])
        result = sorted(s.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_dict_with_key_length_iter(self):
        # [FUNCTION DOC] Dynamo test: dict subclass that iterates only multi-character keys;
        # sorted result is ab and abc.
        """Test custom dict that filters keys by length"""
        d = CustomDictWithKeyLengthIter({"a": 1, "ab": 2, "abc": 3, "b": 4})
        result = sorted(d.__iter__())
        self.assertEqual(result, ["ab", "abc"])

    @make_dynamo_test
    def test_custom_object_yield_from(self):
        # [FUNCTION DOC] Dynamo test: iterating a plain object whose __iter__ delegates with
        # yield from using the iter builtin yields the data.
        """Test custom object whose __iter__ uses yield from self.data"""
        obj = GeneratorIterIterable([10, 20, 30])
        result = list(iter(obj))
        self.assertEqual(result, [10, 20, 30])


class TestIteratorMutationSemantics(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case exploring how Dynamo's iterators behave when the underlying
    # container is changed mid-iteration (dict, set, list insert, pop and append, multiple
    # iterators). setUp and tearDown enable and restore the unittest tracing config.
    """Test iterator behavior when containers are mutated during iteration.

    These tests explore whether Dynamo preserves CPython's iterator semantics
    when the underlying container is modified.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_dict_mutation_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: adds a key to a dict after taking one key from its
        # iterator, then consumes the rest, tolerating a RuntimeError. Per its docstring
        # CPython raises and Dynamo currently allows it; the test accepts both outcomes and
        # asserts nothing on the remainder.
        """Dict mutation during iteration

        In CPython, mutating a dict during iteration raises RuntimeError.
        Dynamo currently allows mutations (known limitation).
        """
        d = {"a": 1, "b": 2}
        it = d.__iter__()

        first = next(it)
        self.assertIn(first, ["a", "b"])

        d["c"] = 3

        try:
            rest = list(it)
            return rest
        except RuntimeError:
            pass

    @make_dynamo_test
    def test_set_mutation_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: same as the dict case for a set: adds an element mid-
        # iteration and accepts either completion or RuntimeError.
        """Set mutation during iteration

        In CPython, mutating a set during iteration raises RuntimeError.
        Dynamo currently allows mutations (known limitation).
        """
        s = {1, 2, 3}
        it = s.__iter__()

        first = next(it)
        self.assertIn(first, {1, 2, 3})

        s.add(4)

        try:
            rest = list(it)
            return rest
        except RuntimeError:
            pass

    @make_dynamo_test
    def test_list_insert_before_iterator_index(self):
        # [FUNCTION DOC] Dynamo test: after taking the first element, inserting at the front
        # of the list means the next element is again one, since the list iterator tracks an
        # absolute index.
        """List iterator uses absolute indices (CPython behavior)"""
        lst = [1, 2, 3]
        it = lst.__iter__()

        val1 = next(it)
        self.assertEqual(val1, 1)

        lst.insert(0, 0)

        val2 = next(it)
        self.assertEqual(val2, 1)

    @make_dynamo_test
    def test_list_remove_before_iterator_index(self):
        # [FUNCTION DOC] Dynamo test: after taking two elements, popping the first item shifts
        # the list so the next value is four (absolute index behavior).
        """List iterator uses absolute indices with removals"""
        lst = [1, 2, 3, 4, 5]
        it = lst.__iter__()

        val1 = next(it)
        val2 = next(it)
        self.assertEqual(val1, 1)
        self.assertEqual(val2, 2)

        lst.pop(0)

        val3 = next(it)
        self.assertEqual(val3, 4)

    @make_dynamo_test
    def test_multiple_iterators_shared_mutation(self):
        # [FUNCTION DOC] Dynamo test: two iterators on one list both observe an element
        # appended later and walk one through four in lockstep.
        """Multiple iterators see shared mutations"""
        lst = [1, 2, 3]
        it1 = lst.__iter__()
        it2 = lst.__iter__()

        self.assertEqual(next(it1), 1)
        self.assertEqual(next(it2), 1)

        lst.append(4)

        self.assertEqual(next(it1), 2)
        self.assertEqual(next(it2), 2)

        self.assertEqual(next(it1), 3)
        self.assertEqual(next(it2), 3)

        self.assertEqual(next(it1), 4)
        self.assertEqual(next(it2), 4)


class Color(enum.Enum):
    # [CLASS DOC] HELPER: Enum fixture with members RED, GREEN, BLUE valued one, two, three.
    RED = 1
    GREEN = 2
    BLUE = 3


class Direction(enum.Enum):
    # [CLASS DOC] HELPER: Enum fixture with string-valued members NORTH, SOUTH, EAST, WEST.
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"


class Priority(enum.IntEnum):
    # [CLASS DOC] HELPER: IntEnum fixture with members LOW, MEDIUM, HIGH valued one, two,
    # three.
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TestEnumIteration(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case checking iteration over enum classes (plain, string-valued and
    # integer enums) under Dynamo. setUp and tearDown enable and restore the unittest tracing
    # config.
    """Test iteration over enum classes"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_enum_iter(self):
        # [FUNCTION DOC] Dynamo test: list of the iter of an enum class gives the members in
        # definition order.
        result = list(iter(Color))
        self.assertEqual(result, [Color.RED, Color.GREEN, Color.BLUE])

    @make_dynamo_test
    def test_enum_for_loop(self):
        # [FUNCTION DOC] Dynamo test: list of the enum class gives members in definition
        # order.
        result = list(Color)
        self.assertEqual(result, [Color.RED, Color.GREEN, Color.BLUE])

    @make_dynamo_test
    def test_enum_iter_values(self):
        # [FUNCTION DOC] Dynamo test: comprehension over the enum collecting each member
        # value; one, two, three.
        result = [m.value for m in Color]
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_enum_iter_names(self):
        # [FUNCTION DOC] Dynamo test: comprehension collecting member names; RED, GREEN, BLUE.
        result = [m.name for m in Color]
        self.assertEqual(result, ["RED", "GREEN", "BLUE"])

    @make_dynamo_test
    def test_enum_string_values_iter(self):
        # [FUNCTION DOC] Dynamo test: iterating the string-valued enum gives NORTH, SOUTH,
        # EAST, WEST in order.
        result = list(iter(Direction))
        self.assertEqual(
            result, [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        )

    @make_dynamo_test
    def test_int_enum_iter(self):
        # [FUNCTION DOC] Dynamo test: iterating the IntEnum gives LOW, MEDIUM, HIGH in order.
        result = list(iter(Priority))
        self.assertEqual(result, [Priority.LOW, Priority.MEDIUM, Priority.HIGH])

    @make_dynamo_test
    def test_enum_iter_unpacking(self):
        # [FUNCTION DOC] Dynamo test: unpacks the three-member enum class into three variables
        # and checks each member.
        a, b, c = Color
        self.assertEqual(a, Color.RED)
        self.assertEqual(b, Color.GREEN)
        self.assertEqual(c, Color.BLUE)


class TestIterWithBuiltins(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the iter builtin applied to builtin iterator types (zip, map,
    # filter, reversed, enumerate) and their chaining. setUp and tearDown enable and restore
    # the unittest tracing config.
    """Test iter() with builtin iterators like zip, map, filter, reversed"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_iter_of_zip(self):
        # [FUNCTION DOC] Dynamo test: iter of a zip object looped to collect pairs.
        """Test iter(zip(a, b)) returns an iterator"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20, 30]
        zip_obj = zip(lst1, lst2)
        it = iter(zip_obj)
        result = []
        for a, b in it:
            result.append((a, b))
        self.assertEqual(result, [(1, 10), (2, 20), (3, 30)])

    @make_dynamo_test
    def test_iter_of_map(self):
        # [FUNCTION DOC] Dynamo test: iter of a map object doubling values, consumed with
        # list.
        """Test iter(map(...)) returns an iterator"""
        lst = [1, 2, 3, 4]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it = iter(map_obj)
        result = list(it)
        self.assertEqual(result, [2, 4, 6, 8])

    @make_dynamo_test
    def test_iter_of_filter(self):
        # [FUNCTION DOC] Dynamo test: iter of a filter object keeping values above three.
        """Test iter(filter(...)) returns an iterator"""
        lst = [1, 2, 3, 4, 5, 6]
        filter_obj = filter(lambda x: x > 3, lst)
        it = iter(filter_obj)
        result = list(it)
        self.assertEqual(result, [4, 5, 6])

    @make_dynamo_test
    def test_iter_of_reversed(self):
        # [FUNCTION DOC] Dynamo test: iter of a reversed list yields descending elements.
        """Test iter(reversed(list)) returns an iterator"""
        lst = [1, 2, 3, 4, 5]
        rev_obj = reversed(lst)
        it = iter(rev_obj)
        result = list(it)
        self.assertEqual(result, [5, 4, 3, 2, 1])

    @make_dynamo_test
    def test_iter_of_enumerate(self):
        # [FUNCTION DOC] Dynamo test: iter of an enumerate object yields index/value pairs
        # from zero.
        """Test iter(enumerate(...)) returns an iterator"""
        lst = ["a", "b", "c"]
        enum_obj = enumerate(lst)
        it = iter(enum_obj)
        result = list(it)
        self.assertEqual(result, [(0, "a"), (1, "b"), (2, "c")])

    @make_dynamo_test
    def test_iter_of_zip_multiple(self):
        # [FUNCTION DOC] Dynamo test: iter of a three-way zip yields triples.
        """Test iter(zip(...)) with multiple iterables"""
        a = [1, 2, 3]
        b = [10, 20, 30]
        c = [100, 200, 300]
        zip_obj = zip(a, b, c)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [(1, 10, 100), (2, 20, 200), (3, 30, 300)])

    @make_dynamo_test
    def test_iter_of_map_with_filter(self):
        # [FUNCTION DOC] Dynamo test: map multiplying by ten then filter above twenty then
        # iter; yields thirty, forty, fifty.
        """Test iter(map(...)) followed by filter"""
        lst = [1, 2, 3, 4, 5]
        mapped = map(lambda x: x * 10, lst)  # noqa: C417
        filtered = filter(lambda x: x > 20, mapped)
        it = iter(filtered)
        result = list(it)
        self.assertEqual(result, [30, 40, 50])

    @make_dynamo_test
    def test_iter_preserves_iterator_state(self):
        # [FUNCTION DOC] Dynamo test: iter on a map iterator returns the same object, and
        # advancing through either alias consumes shared state (two then four).
        """Test that iter() on iterator returns the same iterator"""
        lst = [1, 2, 3, 4]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it1 = iter(map_obj)
        it2 = iter(it1)
        # They should be the same object
        self.assertIs(it1, it2)
        # Advancing one should affect the other
        val1 = next(it1)
        self.assertEqual(val1, 2)
        val2 = next(it2)
        self.assertEqual(val2, 4)

    @make_dynamo_test
    def test_iter_idempotent_on_iterators(self):
        # [FUNCTION DOC] Dynamo test: repeated iter on a list iterator keeps returning the
        # same iterator; values one then two are drawn through different aliases.
        """Test that iter() called multiple times on iterator works"""
        lst = [1, 2, 3]
        it = iter(lst)
        it_again = iter(it)
        it_again_2 = iter(it_again)
        val1 = next(it_again_2)
        self.assertEqual(val1, 1)
        val2 = next(it)
        self.assertEqual(val2, 2)

    @make_dynamo_test
    def test_iter_of_zip_empty(self):
        # [FUNCTION DOC] Dynamo test: iter of a zip of two empty lists yields nothing.
        """Test iter(zip(...)) with empty iterables"""
        lst1 = []
        lst2 = []
        zip_obj = zip(lst1, lst2)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_iter_of_filter_empty_result(self):
        # [FUNCTION DOC] Dynamo test: iter of a filter that rejects everything yields nothing.
        """Test iter(filter(...)) when filter returns empty"""
        lst = [1, 2, 3]
        filter_obj = filter(lambda x: x > 10, lst)
        it = iter(filter_obj)
        result = list(it)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_iter_of_map_with_zip(self):
        # [FUNCTION DOC] Dynamo test: map summing pairs from a zip object and iterated;
        # eleven, twenty-two, thirty-three.
        """Test iter(map(...)) with zip"""
        a = [1, 2, 3]
        b = [10, 20, 30]
        zip_obj = zip(a, b)
        mapped = map(lambda pair: pair[0] + pair[1], zip_obj)  # noqa: C417
        it = iter(mapped)
        result = list(it)
        self.assertEqual(result, [11, 22, 33])

    @make_dynamo_test
    def test_iter_exhaustion_with_builtin_iterators(self):
        # [FUNCTION DOC] Dynamo test: next on an exhausted map iterator raises StopIteration.
        """Test exhausting iter() of builtin iterators"""
        lst = [1, 2]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it = iter(map_obj)
        next(it)
        next(it)
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_iter_next_with_default_on_builtin(self):
        # [FUNCTION DOC] Dynamo test: next on a filter iterator returns two, then the supplied
        # default string when exhausted.
        """Test next() with default on iter() of builtin iterator"""
        lst = [1, 2]
        filter_obj = filter(lambda x: x > 1, lst)
        it = iter(filter_obj)
        val = next(it)
        self.assertEqual(val, 2)
        result = next(it, "EMPTY")
        self.assertEqual(result, "EMPTY")

    @make_dynamo_test
    def test_iter_of_zip_uneven_lengths(self):
        # [FUNCTION DOC] Dynamo test: zip of lists with three and two elements yields two
        # pairs.
        """Test iter(zip(...)) with uneven length iterables"""
        a = [1, 2, 3]
        b = [10, 20]
        zip_obj = zip(a, b)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [(1, 10), (2, 20)])

    @make_dynamo_test
    def test_iter_chaining_builtin_iterators(self):
        # [FUNCTION DOC] Dynamo test: map doubling then filter above four then iter; six,
        # eight, ten.
        """Test chaining multiple iter() calls on builtin iterators"""
        lst = [1, 2, 3, 4, 5]
        # Create a chain of iterators: map -> filter -> iter
        mapped = map(lambda x: x * 2, lst)  # noqa: C417
        filtered = filter(lambda x: x > 4, mapped)
        it = iter(filtered)
        result = list(it)
        self.assertEqual(result, [6, 8, 10])

    @make_dynamo_test
    def test_iter_enumerate_with_start(self):
        # [FUNCTION DOC] Dynamo test: enumerate with start ten then iter yields indices ten to
        # twelve.
        """Test iter(enumerate(...)) with start parameter"""
        lst = ["a", "b", "c"]
        enum_obj = enumerate(lst, start=10)
        it = iter(enum_obj)
        result = list(it)
        self.assertEqual(result, [(10, "a"), (11, "b"), (12, "c")])

    @make_dynamo_test
    def test_iter_zip_with_filter_and_map(self):
        # [FUNCTION DOC] Dynamo test: iter of a zip of two five-element lists; checks length
        # five and the first pair. Despite the name no filter or map is applied.
        """Test iter() with nested builtin iterators"""
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        zipped = zip(a, b)
        # Flatten and just test that iteration works
        it = iter(zipped)
        result = list(it)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], (1, 10))


class NotIterable:
    # [CLASS DOC] HELPER: Test fixture; empty class that is not iterable.
    pass


class BadIterReturnsNonIterator:
    # [CLASS DOC] HELPER: Test fixture; __iter__ returns a non-iterator integer.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns the integer forty-two, which is not an iterator.
        return 42  # not an iterator


class IterWithoutNext:
    # [CLASS DOC] HELPER: Test fixture; __iter__ returns self but no __next__ exists.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self, which lacks __next__ and so is not a valid
        # iterator.
        return self  # no __next__


class NoIterNoGetitem:
    # [CLASS DOC] HELPER: Test fixture; empty class with neither __iter__ nor indexing.
    pass


class BlockedIter:
    # [CLASS DOC] HELPER: Test fixture; sets __iter__ to None to block iteration.
    __iter__ = None


class BlockedGetItem:
    # [CLASS DOC] HELPER: Test fixture; sets __getitem__ to None to block indexing.
    __getitem__ = None


class BlockedLen:
    # [CLASS DOC] HELPER: Test fixture; sets __len__ to None to block length.
    __len__ = None


class TestIterErrors(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case checking Dynamo raises the same exceptions as CPython for invalid
    # iter and next usage, dict mutation during view iteration, and None-blocked special
    # methods. setUp and tearDown enable and restore the unittest tracing config.
    """Test that iter() raises the correct exceptions"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_iter_non_iterable_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on an int raises TypeError.
        with self.assertRaises(TypeError):
            iter(42)

    @make_dynamo_test
    def test_iter_custom_non_iterable_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on an instance of the empty class raises TypeError.
        with self.assertRaises(TypeError):
            iter(NotIterable())

    @make_dynamo_test
    def test_iter_two_arg_non_callable_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: two-argument iter with a non-callable first argument
        # raises TypeError.
        with self.assertRaises(TypeError):
            iter(42, None)

    @make_dynamo_test
    def test_iter_bad_iter_return_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on an object whose __iter__ returns a non-iterator
        # raises TypeError.
        with self.assertRaises(TypeError):
            iter(BadIterReturnsNonIterator())

    @make_dynamo_test
    def test_iter_returns_self_without_next_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on an object whose __iter__ returns itself without
        # __next__ raises TypeError.
        with self.assertRaises(TypeError):
            iter(IterWithoutNext())

    @make_dynamo_test
    def test_iter_no_iter_no_getitem_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on an object with no iteration or indexing support
        # raises TypeError.
        with self.assertRaises(TypeError):
            iter(NoIterNoGetitem())

    @make_dynamo_test
    def test_next_on_exhausted_raises_stop_iteration(self):
        # [FUNCTION DOC] Dynamo test: next on the iterator of an empty list raises
        # StopIteration.
        it = iter([])
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_next_default_on_exhausted(self):
        # [FUNCTION DOC] Dynamo test: next with a default on an empty iterator returns the
        # default.
        it = iter([])
        result = next(it, "default")
        self.assertEqual(result, "default")

    @make_dynamo_test
    def test_next_non_iterator_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: next on a list (iterable, not iterator) raises TypeError
        # with the message that the list object is not an iterator.
        with self.assertRaisesRegex(TypeError, "'list' object is not an iterator"):
            next([1, 2])

    @make_dynamo_test
    def test_next_non_iterator_with_default_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: same TypeError message check, even when a default is
        # supplied.
        with self.assertRaisesRegex(TypeError, "'list' object is not an iterator"):
            next([1, 2], "default")

    @make_dynamo_test
    def test_error_on_dict_keys_mutation_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: after taking a key from a keys-view iterator, adding a
        # new key and calling next must raise RuntimeError.
        """Test that mutating a dict during iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.keys())
        next(it)  # Get first key
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)

    @make_dynamo_test
    def test_error_on_dict_values_mutation_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: same RuntimeError expectation for a values-view
        # iterator.
        """Test that mutating a dict during values() iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.values())
        next(it)  # Get first value
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)

    @make_dynamo_test
    def test_error_on_dict_items_mutation_during_iteration(self):
        # [FUNCTION DOC] Dynamo test: same RuntimeError expectation for an items-view
        # iterator.
        """Test that mutating a dict during items() iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.items())
        next(it)  # Get first item
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)

    @make_dynamo_test
    def test_iter_blocked_slot_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: iter on a class whose __iter__ is None raises TypeError.
        with self.assertRaises(TypeError):
            iter(BlockedIter())

    @make_dynamo_test
    def test_getitem_blocked_slot_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: indexing an instance whose __getitem__ is None raises
        # TypeError.
        with self.assertRaises(TypeError):
            BlockedGetItem()[0]

    @make_dynamo_test
    def test_len_blocked_slot_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: len on an instance whose __len__ is None raises
        # TypeError.
        with self.assertRaises(TypeError):
            len(BlockedLen())


class IterPropAttrErrorWithGetItem:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is a property that raises AttributeError, but
    # indexing over stored data exists for the legacy fallback.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the data in the data attribute.
        self.data = data

    @property
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Property getter that always raises AttributeError.
        raise AttributeError("no __iter__")

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns the stored data element at the index.
        return self.data[index]


class IterPropAttrErrorNoGetItem:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is a property raising AttributeError and no
    # indexing exists.
    @property
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Property getter that always raises AttributeError.
        raise AttributeError("no __iter__")


class IterPropReturnsCallable:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is a property whose value is a callable
    # producing an iterator over the stored data.
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the data in the data attribute.
        self.data = data

    @property
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Property getter returning a zero-argument callable that gives
        # an iterator over the data.
        return lambda: iter(self.data)


class IterPropRaisesValueError:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is a property raising ValueError.
    @property
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Property getter that always raises ValueError.
        raise ValueError("boom")


class BlockedIterWithGetItem:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is None and indexing exists, to check None
    # wins over the indexing fallback.
    __iter__ = None

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns the element of the fixed list one, two, three at the
        # index.
        return [1, 2, 3][index]


class _IterDescriptor:
    # [CLASS DOC] HELPER: Descriptor whose binding yields a callable producing an iterator
    # over the owner instance's data.
    def __get__(self, obj, owner):
        # [FUNCTION DOC] HELPER: Returns a zero-argument callable giving an iterator over the
        # data attribute of the instance.
        return lambda: iter(obj.data)


class _AttrErrorDescriptor:
    # [CLASS DOC] HELPER: Descriptor whose binding always raises AttributeError.
    def __get__(self, obj, owner):
        # [FUNCTION DOC] HELPER: Always raises AttributeError.
        raise AttributeError("no attr")


class IterViaDescriptor:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is provided by the iterating descriptor class
    # attribute.
    __iter__ = _IterDescriptor()

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the data in the data attribute.
        self.data = data


class IterDescriptorAttrErrorWithGetItem:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is the attribute-error descriptor, with
    # indexing available as fallback.
    __iter__ = _AttrErrorDescriptor()

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the data in the data attribute.
        self.data = data

    def __getitem__(self, index):
        # [FUNCTION DOC] HELPER: Returns the stored data element at the index.
        return self.data[index]


class IterNotCallable:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is the integer forty-two, not callable.
    __iter__ = 42


class IterStaticMethod:
    # [CLASS DOC] HELPER: Test fixture; __iter__ is a static method wrapping a function that
    # returns an iterator over one, two, three.
    __iter__ = staticmethod(lambda: iter([1, 2, 3]))


class NextProp:
    # [CLASS DOC] HELPER: Test fixture iterator whose __next__ is a property returning a
    # closure; counts from one to n.
    def __init__(self, n):
        # [FUNCTION DOC] HELPER: Stores the limit n and starts the counter at zero.
        self.n = n
        self.i = 0

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self

    @property
    def __next__(self):
        # [FUNCTION DOC] HELPER: Property getter returning a closure for advancing the
        # iterator.
        def inner():
            # [FUNCTION DOC] HELPER: Raises StopIteration when the counter has reached the
            # limit, else increments and returns the counter.
            if self.i >= self.n:
                raise StopIteration
            self.i += 1
            return self.i

        return inner


class NextNone:
    # [CLASS DOC] HELPER: Test fixture; __iter__ returns self and __next__ is None.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self

    __next__ = None


class NextPropAttrError:
    # [CLASS DOC] HELPER: Test fixture; __iter__ returns self and __next__ is a property
    # raising AttributeError.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns self.
        return self

    @property
    def __next__(self):
        # [FUNCTION DOC] HELPER: Property getter that always raises AttributeError.
        raise AttributeError("gone")


class TestSpecialMethodIterLookup(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the lookup of the __iter__ and __next__ special methods
    # through the type MRO with descriptor binding, as in CPython slot functions: properties,
    # descriptors, None, non-callables, static methods. setUp and tearDown enable and restore
    # the unittest tracing config.
    """slot_tp_iter / slot_tp_iternext lookup semantics: __iter__ / __next__
    resolved via MRO-only special lookup with descriptor binding
    (lookup_maybe_method in Objects/typeobject.c)."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # Pre-3.14 slot_tp_iter blanket-clears lookup errors (PyErr_Clear) before
    # the __getitem__ probe; Dynamo does not encode this yet.
    @unittest.expectedFailure
    @make_dynamo_test
    def test_iter_property_attribute_error_with_getitem(self):
        # [FUNCTION DOC] Dynamo test: expected failure. A property __iter__ raising
        # AttributeError should be swallowed and fall back to the indexing protocol, giving
        # one, two, three. Dynamo does not model the swallowing yet.
        # __iter__ bind raises AttributeError -> swallowed -> seqiter fallback
        self.assertEqual(list(IterPropAttrErrorWithGetItem([1, 2, 3])), [1, 2, 3])

    @unittest.expectedFailure
    @make_dynamo_test
    def test_iter_property_attribute_error_no_getitem(self):
        # [FUNCTION DOC] Dynamo test: expected failure. Same property raising AttributeError
        # without indexing should give TypeError from iter; Dynamo does not yet match.
        with self.assertRaises(TypeError):
            iter(IterPropAttrErrorNoGetItem())

    @make_dynamo_test
    def test_iter_property_returns_callable(self):
        # [FUNCTION DOC] Dynamo test: an __iter__ property returning a callable is bound and
        # called; list gives the data.
        self.assertEqual(list(IterPropReturnsCallable([1, 2, 3])), [1, 2, 3])

    @xfailIfPy313AndEarlier
    @make_dynamo_test
    def test_iter_property_raises_value_error(self):
        # [FUNCTION DOC] Dynamo test: marked as expected failure on Python up to 3.13. A
        # property __iter__ raising ValueError gives TypeError before Python 3.14 (error
        # swallowed, no indexing) and ValueError from 3.14; the test picks the expected
        # exception by version.
        # Pre-3.14 slot_tp_iter swallows any lookup error (blanket
        # PyErr_Clear) and falls through to the absent __getitem__ probe
        # -> TypeError; 3.14 only swallows AttributeError, so ValueError
        # propagates
        expected = ValueError if sys.version_info >= (3, 14) else TypeError
        with self.assertRaises(expected):
            iter(IterPropRaisesValueError())

    @make_dynamo_test
    def test_iter_none_with_getitem(self):
        # [FUNCTION DOC] Dynamo test: __iter__ set to None blocks iteration even though
        # indexing exists; TypeError expected.
        # __iter__ = None wins over the __getitem__ fallback
        with self.assertRaises(TypeError):
            iter(BlockedIterWithGetItem())

    @make_dynamo_test
    def test_iter_via_descriptor(self):
        # [FUNCTION DOC] Dynamo test: __iter__ supplied by a descriptor class attribute is
        # bound and used; list gives the data.
        self.assertEqual(list(IterViaDescriptor([1, 2, 3])), [1, 2, 3])

    @make_dynamo_test
    def test_iter_descriptor_attribute_error_with_getitem(self):
        # [FUNCTION DOC] Dynamo test: a descriptor whose binding raises AttributeError falls
        # back to indexing; list gives four, five, six.
        self.assertEqual(list(IterDescriptorAttrErrorWithGetItem([4, 5, 6])), [4, 5, 6])

    @make_dynamo_test
    def test_iter_not_callable(self):
        # [FUNCTION DOC] Dynamo test: an integer __iter__ makes iter raise TypeError.
        with self.assertRaises(TypeError):
            iter(IterNotCallable())

    @make_dynamo_test
    def test_iter_staticmethod(self):
        # [FUNCTION DOC] Dynamo test: a static method __iter__ is invoked with no instance
        # argument; list gives one, two, three.
        self.assertEqual(list(IterStaticMethod()), [1, 2, 3])

    @make_dynamo_test
    def test_next_property(self):
        # [FUNCTION DOC] Dynamo test: iterator whose __next__ is a property returning a
        # closure yields one, two, three.
        self.assertEqual(list(NextProp(3)), [1, 2, 3])

    @make_dynamo_test
    def test_next_none_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: obtains the iterator of a class whose __next__ is None,
        # then next on it raises TypeError.
        it = iter(NextNone())
        with self.assertRaises(TypeError):
            next(it)

    @make_dynamo_test
    def test_next_missing_raises_type_error(self):
        # [FUNCTION DOC] Dynamo test: next on an object that is not an iterator raises
        # TypeError (builtin next checks iterator support first).
        # builtin next() gates on PyIter_Check
        with self.assertRaises(TypeError):
            next(NoIterNoGetitem())

    @make_dynamo_test
    def test_next_property_attribute_error_propagates(self):
        # [FUNCTION DOC] Dynamo test: when binding the __next__ property raises
        # AttributeError, next propagates AttributeError rather than swallowing it.
        # slot_tp_iternext uses the raising lookup variant: AttributeError
        # from the bind is NOT swallowed
        it = iter(NextPropAttrError())
        with self.assertRaises(AttributeError):
            next(it)


class ColorInt(enum.IntEnum):
    # [CLASS DOC] HELPER: IntEnum fixture with a single member RED valued one.
    RED = 1


class TestSpecialMethodLookupRegressions(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case of regressions for the special-method lookup engine: C-slot
    # dunders must not recurse and errors must match eager. setUp and tearDown enable and
    # restore the unittest tracing config.
    """Regressions for the special-method lookup engine: C-slot dunders must
    never recurse, and error messages must match eager."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, saves the current dynamo
        # enable_trace_unittest config in an instance attribute, then sets that config to
        # True.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Calls the base tearDown and restores the saved
        # enable_trace_unittest config value.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    def test_iter_on_c_iterable_object_graph_breaks(self):
        # [FUNCTION DOC] Compiles a function with the eager backend and full graph that
        # iterates a zip object passed as input; expects Unsupported to be raised (clean graph
        # break) instead of RecursionError.
        # zip object as graph input: its __iter__ is a C wrapper descriptor
        # with no traceable body.  Must graph break cleanly (Unsupported),
        # never RecursionError.
        from torch._dynamo.exc import Unsupported

        def fn(it, x):
            # [FUNCTION DOC] HELPER: Comprehension adding one over the input iterable, short-
            # circuited into returning the tensor plus one.
            return [v + 1 for v in it] and x + 1

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(
                zip([1, 2], [3, 4]), torch.ones(2)
            )

    def test_iter_on_dict_items_object_graph_breaks(self):
        # [FUNCTION DOC] Compiles with full graph a function that calls iter and next on a
        # dict items view passed as input; expects Unsupported.
        from torch._dynamo.exc import Unsupported

        def fn(items, x):
            # [FUNCTION DOC] HELPER: Takes an iterator from the items argument, advances it
            # once, and returns the tensor plus one.
            it = iter(items)
            return next(it) and x + 1

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(
                {"a": 1}.items(), torch.ones(2)
            )

    @make_dynamo_test
    def test_intenum_unary_neg(self):
        # [FUNCTION DOC] Dynamo test: negating an IntEnum member equals the negative integer;
        # must not recurse.
        # int.__neg__ on an IntEnum member: inherited C slot of a modeled
        # constant base -- must fold/delegate, never recurse.
        self.assertEqual(-ColorInt.RED, -1)

    @make_dynamo_test
    def test_iter_none_with_getitem_message(self):
        # [FUNCTION DOC] Dynamo test: iter on a class with __iter__ None and indexing raises
        # TypeError whose message contains not iterable.
        # slot_tp_iter's attr_is_none branch: message is "is not iterable",
        # not "'NoneType' object is not callable"
        raised = False
        try:
            iter(BlockedIterWithGetItem())
        except TypeError as e:
            raised = True
            self.assertIn("is not iterable", str(e))
        self.assertTrue(raised)

    @make_dynamo_test
    def test_next_missing_message(self):
        # [FUNCTION DOC] Dynamo test: next on a non-iterator raises TypeError whose message
        # contains not an iterator.
        # builtin next() gates on PyIter_Check: "is not an iterator"
        raised = False
        try:
            next(NoIterNoGetitem())
        except TypeError as e:
            raised = True
            self.assertIn("is not an iterator", str(e))
        self.assertTrue(raised)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
