# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the native type-slot query used by Dynamo: get_type_slots returns four
# slot tables (sequence, mapping, number, type) and has_slot tests one slot in a table. The
# tests assert which CPython protocol slots are populated for builtins and local classes.
# Imports the native Dynamo extension; no other import-time side effects.

"""Tests for CPython type slot detection in Dynamo.

Tests that get_type_slots correctly identifies which protocol methods
(sequence, mapping, number, type) are implemented by various types.
"""

import collections
import collections.abc
import dataclasses
import enum

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)
from torch._dynamo.test_case import run_tests, TestCase


class TestTypeSlots(TestCase):
    # [CLASS DOC] Test case for the native get_type_slots query and has_slot helper. Each test
    # fetches the four slot tables (sequence, mapping, number, type) for a builtin or locally
    # defined class and asserts which individual CPython protocol slots are populated.
    """Test suite for type slot detection."""

    def _get_slot_info(self, obj_type):
        # [FUNCTION DOC] HELPER: Calls get_type_slots on the given type and returns the four
        # results unchanged in order: sequence slots, mapping slots, number slots, type slots.
        """Helper to get and unpack slot information."""
        seq_slots, map_slots, num_slots, type_slots = get_type_slots(obj_type)
        return seq_slots, map_slots, num_slots, type_slots

    def test_dict_slots(self):
        # [FUNCTION DOC] Checks dict has no sequence length slot but has mapping length, subscript
        # and subscript-assignment slots.
        """Test that dict has mapping protocol but not sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(dict)

        # dict should NOT have sq_length (sequence protocol)
        self.assertFalse(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))

        # dict SHOULD have mapping protocol
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_ASS_SUBSCRIPT))

    def test_list_slots(self):
        # [FUNCTION DOC] Checks list has sequence length, item and contains slots, and also the
        # mapping length and subscript slots.
        """Test that list has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(list)

        # list SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

        # list also has mapping protocol for compatibility (mp_length, mp_subscript, mp_ass_subscript)
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))

    def test_tuple_slots(self):
        # [FUNCTION DOC] Checks tuple has sequence length, item and contains slots.
        """Test that tuple has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(tuple)

        # tuple SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_set_slots(self):
        # [FUNCTION DOC] Checks set has the sequence contains slot and no mapping length slot.
        """Test that set has sequence protocol for contains."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(set)

        # set SHOULD have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

        # set should NOT have mapping protocol
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_dict_subclass_slots(self):
        # [FUNCTION DOC] Checks a plain dict subclass reports both the sequence length slot and
        # the mapping length slot.
        """Test that dict subclasses have both mapping and sequence protocol."""

        class DictSubclass(dict):
            # [CLASS DOC] HELPER: Empty dict subclass used as the queried type.
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(DictSubclass)

        # Dict subclasses expose both protocols for compatibility
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_list_subclass_slots(self):
        # [FUNCTION DOC] Checks a plain list subclass reports sequence length and item slots.
        """Test that list subclasses have sequence protocol."""

        class ListSubclass(list):
            # [CLASS DOC] HELPER: Empty list subclass used as the queried type.
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(ListSubclass)

        # List subclasses should have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_int_slots(self):
        # [FUNCTION DOC] Checks int has the number add, subtract and multiply slots.
        """Test that int has number protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(int)

        # int SHOULD have number protocol operations
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_ADD))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_SUBTRACT))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_MULTIPLY))

    def test_str_slots(self):
        # [FUNCTION DOC] Checks str has sequence length, item and contains slots.
        """Test that str has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(str)

        # str SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_type_has_call_slot(self):
        # [FUNCTION DOC] Checks the type class has the call slot in its type slots.
        """Test that type objects have tp_call slot."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(type)

        # type SHOULD have tp_call (for calling classes/types)
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_CALL))

    def test_type_has_hash_slot(self):
        # [FUNCTION DOC] Checks int has the hash slot in its type slots.
        """Test that most types have tp_hash slot."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(int)

        # int SHOULD have tp_hash
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_custom_class_slots(self):
        # [FUNCTION DOC] Checks a plain user class has neither the sequence length slot nor the
        # mapping length slot.
        """Test that custom user-defined classes have minimal slots."""

        class CustomClass:
            # [CLASS DOC] HELPER: Empty class used as the queried type.
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(CustomClass)

        # Custom classes should not have sequence/mapping/number protocol by default
        self.assertFalse(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_set_subclass_slots(self):
        # [FUNCTION DOC] Checks a set subclass inherits the sequence contains slot.
        """Test that set subclasses inherit set protocol."""

        class SetSubclass(set):
            # [CLASS DOC] HELPER: Empty set subclass used as the queried type.
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(SetSubclass)

        # Set subclasses should have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_tuple_subclass_slots(self):
        # [FUNCTION DOC] Checks a tuple subclass with empty slots inherits sequence length and
        # item slots.
        """Test that tuple subclasses inherit sequence protocol."""

        class TupleSubclass(tuple):
            # [CLASS DOC] HELPER: Tuple subclass declaring no instance slots, used as the
            # queried type.
            __slots__ = ()

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(TupleSubclass)

        # Tuple subclasses should have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_mutable_mapping_slots(self):
        # [FUNCTION DOC] Checks a Python class implementing the MutableMapping abstract base
        # reports mapping length, subscript and subscript-assignment slots (set from its
        # Python-level dunder methods).
        """Test that MutableMapping ABC has mapping protocol."""

        class MyMapping(collections.abc.MutableMapping):
            # [CLASS DOC] HELPER: Minimal MutableMapping implementation backed by a private dict,
            # providing the required item, deletion, iteration and length methods.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the empty backing dict stored on a
                # private attribute.
                self._data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the backing dict entry for the key (raises
                # KeyError if absent).
                return self._data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Stores the value under the key in the backing dict.
                self._data[key] = value

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Removes the key from the backing dict (raises KeyError
                # if absent).
                del self._data[key]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over the backing dict keys.
                return iter(self._data)

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the number of entries in the backing dict.
                return len(self._data)

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(MyMapping)

        # MutableMapping subclasses should have mapping protocol
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_ASS_SUBSCRIPT))

    def test_frozen_dataclass_slots(self):
        # [FUNCTION DOC] Checks a frozen dataclass has the hash slot in its type slots.
        """Test that frozen dataclasses have standard object slots."""

        @dataclasses.dataclass(frozen=True)
        class FrozenData:
            # [CLASS DOC] HELPER: Frozen dataclass with an int field and a string field used as
            # the queried type.
            x: int
            y: str

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(FrozenData)

        # Frozen dataclasses should have basic type slots like hash
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_enum_slots(self):
        # [FUNCTION DOC] Checks an Enum class has the hash slot in its type slots.
        """Test that Enum types have expected slots."""

        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Three-member enum used as the queried type.
            RED = 1
            GREEN = 2
            BLUE = 3

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(Color)

        # Enums should have type protocol slots
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_enum_member_slots(self):
        # [FUNCTION DOC] Checks the hash slot for an Enum class (the enum class itself is queried,
        # not a member instance, despite the test name).
        """Test that individual enum members work correctly."""

        class Status(enum.Enum):
            # [CLASS DOC] HELPER: Two-member string enum used as the queried type.
            PENDING = "pending"
            ACTIVE = "active"

        # Enum members are instances, test the enum class
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(Status)
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_metaclass_slots(self):
        # [FUNCTION DOC] Checks a user-defined metaclass has the call slot.
        # KNOWN ISSUE: the class created with that metaclass is never used and does not affect
        # the assertion.
        """Test that metaclass types have tp_call for instantiation."""

        class MyMeta(type):
            # [CLASS DOC] HELPER: Empty metaclass deriving from type.
            pass

        class MyClass(metaclass=MyMeta):
            # [CLASS DOC] HELPER: Empty class using the metaclass; never queried.
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(MyMeta)

        # Metaclasses should have tp_call for creating instances
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_CALL))

    def test_dict_subclass_with_custom_len(self):
        # [FUNCTION DOC] Regression check: a dict subclass overriding length still reports both
        # the sequence length slot and the mapping length slot.
        """Test dict subclass with custom __len__ (the original bug case)."""

        class DictWithCustomLen(dict):
            # [CLASS DOC] HELPER: Dict subclass whose length is one less than the real size.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the base dict length minus one.
                return super().__len__() - 1

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(
            DictWithCustomLen
        )

        # Should have both protocols (CPython compatibility)
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_frozenset_slots(self):
        # [FUNCTION DOC] Checks frozenset has the sequence contains slot.
        """Test that frozenset has sequence protocol for contains."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(frozenset)

        # frozenset SHOULD have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_float_slots(self):
        # [FUNCTION DOC] Checks float has the number add, subtract and multiply slots.
        """Test that float has number protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(float)

        # float SHOULD have number protocol operations
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_ADD))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_SUBTRACT))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_MULTIPLY))

    def test_deque_sq_item_only(self):
        # [FUNCTION DOC] Checks deque has the sequence item slot but not the mapping
        # subscript slot.
        """deque has sq_item but NOT mp_subscript — generic_getitem dispatches to Branch 2."""
        seq_slots, map_slots, _, _ = self._get_slot_info(collections.deque)
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_nb_index_int_bool(self):
        # [FUNCTION DOC] Checks both int and bool have the number index slot.
        # KNOWN ISSUE: the failure message argument is a function object, not a string, so the
        # intended message is not displayed.
        """int and bool have nb_index (pass _PyIndex_Check)."""
        for t in [int, bool]:
            _, _, num_slots, _ = self._get_slot_info(t)
            self.assertTrue(
                has_slot(num_slots, PyNumberSlots.NB_INDEX),
                lambda msg: f"{msg}\n{t.__name__} should have nb_index",
            )

    def test_nb_index_absent(self):
        # [FUNCTION DOC] Checks float, str and complex lack the number index slot.
        # KNOWN ISSUE: the failure message argument is a function object, not a string, so the
        # intended message is not displayed.
        """float, str, complex do NOT have nb_index (fail _PyIndex_Check)."""
        for t in [float, str, complex]:
            _, _, num_slots, _ = self._get_slot_info(t)
            self.assertFalse(
                has_slot(num_slots, PyNumberSlots.NB_INDEX),
                lambda msg: f"{msg}\n{t.__name__} should NOT have nb_index",
            )

    def test_mp_subscript_on_subscriptable_types(self):
        # [FUNCTION DOC] Checks a list of subscriptable types (list, tuple, dict, str, bytes,
        # range, OrderedDict, defaultdict, Tensor, Size) all have the mapping subscript slot.
        # KNOWN ISSUE: the failure message argument is a function object, not a string, so the
        # intended message is not displayed.
        """All types with mp_subscript_impl overrides must have the C-level slot."""
        import torch

        for t in [
            list,
            tuple,
            dict,
            str,
            bytes,
            range,
            collections.OrderedDict,
            collections.defaultdict,
            torch.Tensor,
            torch.Size,
        ]:
            _, map_slots, _, _ = self._get_slot_info(t)
            self.assertTrue(
                has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT),
                lambda msg: f"{msg}\n{t.__name__} should have mp_subscript",
            )


if __name__ == "__main__":
    run_tests()
