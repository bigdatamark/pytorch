# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for user-defined objects: attribute assignment on slotted classes
#   (own cases and ones adapted from CPython slot tests), reads of a class dict, class attribute
#   assignment, and item assignment/deletion on user classes and metaclasses.
# Module-level helper classes define the slotted fixtures. No import-time side effects beyond
#   defining them.

import dataclasses
import types
import unittest

import torch
import torch._dynamo.testing as dynamo_testing
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class SlotsOnly:
    # [CLASS DOC] HELPER: Fixture with two declared slots (x and y) and no instance dict, so
    #   assigning any other attribute raises AttributeError.
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        # [FUNCTION DOC] HELPER: Store the two arguments into the x and y slots.
        self.x = x
        self.y = y


class SlotsAndDict:
    # [CLASS DOC] HELPER: Fixture with slot x plus an instance dict, so undeclared attributes are
    #   accepted and land in the dict.
    __slots__ = ("x", "__dict__")

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Store the argument into the x slot.
        self.x = x


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenSlots:
    # [CLASS DOC] HELPER: Frozen dataclass with slots and integer fields x and y; its generated
    #   init bypasses the frozen setter through the object-level setattr.
    x: int
    y: int


class SlotsAndSetattr:
    # [CLASS DOC] HELPER: Fixture with slot x and a custom setattr that doubles every assigned
    #   value before storing it in the slot.
    __slots__ = ("x",)

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Assign the argument to x, which goes through the doubling
        #   setattr.
        self.x = x

    def __setattr__(self, name, value):
        # [FUNCTION DOC] HELPER: Custom setattr hook: write the given name with the value
        #   multiplied by two using the object-level setattr. Returns nothing.
        object.__setattr__(self, name, value * 2)


class SlotsAndDictAndSetattr:
    # [CLASS DOC] HELPER: Fixture with slot x, an instance dict, and a custom setattr that
    #   doubles every assigned value.
    __slots__ = ("x", "__dict__")

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Assign the argument to x, which goes through the doubling
        #   setattr.
        self.x = x

    def __setattr__(self, name, value):
        # [FUNCTION DOC] HELPER: Custom setattr hook: write the given name with the value
        #   multiplied by two using the object-level setattr. Returns nothing.
        object.__setattr__(self, name, value * 2)


class SlotsBase:
    # [CLASS DOC] HELPER: Slotted base class declaring slot x, used for slot inheritance tests.
    __slots__ = ("x",)

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize slot x to zero.
        self.x = 0


class SlotsDerived(SlotsBase):
    # [CLASS DOC] HELPER: Slotted subclass of SlotsBase that adds slot y on top of the inherited
    #   x.
    __slots__ = ("y",)

    def __init__(self):
        # [FUNCTION DOC] HELPER: Call the parent initializer (x becomes zero) then set y to zero.
        super().__init__()
        self.y = 0


class Plain:
    # [CLASS DOC] HELPER: Plain class without slots; instances have a dict.
    pass


class SlotsChildOfPlain(Plain):
    # [CLASS DOC] HELPER: Slotted subclass (slot z) of a class without slots, so instances still
    #   have a dict inherited from the parent.
    __slots__ = ("z",)

    def __init__(self):
        # [FUNCTION DOC] HELPER: Set slot z to zero.
        self.z = 0


class Slots:
    # [CLASS DOC] HELPER: Fixture with a single slot x that is never initialized, so the
    #   attribute starts unset.
    __slots__ = ("x",)


class SlotsShadowed(SlotsBase):
    # [CLASS DOC] HELPER: Subclass of SlotsBase whose class attribute x (value 42) shadows the
    #   parent slot descriptor, so reads return the class value.
    x = 42  # class attribute shadows parent's slot descriptor


class SlotsAndProperty:
    # [CLASS DOC] HELPER: Fixture with a private slot and a property x whose setter stores double
    #   the assigned value; the getter returns the stored value.
    __slots__ = ("_x",)

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Assign the argument through the property setter (stored value is
        #   doubled).
        self._x = x

    @property
    def x(self):
        # [FUNCTION DOC] HELPER: Property getter returning the value held in the private slot.
        return self._x

    @x.setter
    def x(self, value):
        # [FUNCTION DOC] HELPER: Property setter storing twice the given value in the private
        #   slot.
        self._x = value * 2


class TestSlotsAttrAssignment(TestCase):
    # [CLASS DOC] Tests that Dynamo-compiled functions match eager behavior for attribute
    #   assignment, deletion, hasattr, getattr and descriptors on slotted objects, including
    #   side-effect materialization on arguments and recompilation behavior.
    """Tests for attribute assignment on objects with __slots__."""

    def test_valid_slot_assignment(self):
        # [FUNCTION DOC] Assigning to a declared slot inside a compiled function must match eager
        #   results (checked with the standard compile-vs-eager helper using one tensor
        #   argument).
        # Case 1: assign to a declared slot — should succeed
        def fn(t):
            # [FUNCTION DOC] HELPER: Create a two-slot object, overwrite slot x, return the
            #   tensor plus x.
            obj = SlotsOnly(1, 2)
            obj.x = 99
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_invalid_slot_assignment_raises(self):
        # [FUNCTION DOC] Assigning an undeclared attribute on a dict-less slotted object must
        #   raise AttributeError in eager and some exception in the compiled full-graph version.
        # Case 2: assign to an undeclared attr on a slotted object (no __dict__)
        # should raise AttributeError in eager; compiled raises an exception too
        def fn(t):
            # [FUNCTION DOC] HELPER: Build a two-slot object, assign the undeclared attribute z
            #   (raises), and would return the tensor plus x.
            obj = SlotsOnly(1, 2)
            obj.z = 99
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertRaises(AttributeError, fn, t)
        self.assertRaises(Exception, compiled_fn, t)

    def test_slots_with_dict_allows_arbitrary_attrs(self):
        # [FUNCTION DOC] When slots include a dict, assigning an arbitrary new attribute works
        #   and compiled output matches eager.
        # Case 3: __slots__ includes __dict__ — arbitrary attr assignment should work
        def fn(t):
            # [FUNCTION DOC] HELPER: Set an extra attribute on the object and return the tensor
            #   plus x plus extra.
            obj = SlotsAndDict(1)
            obj.extra = 42
            return t + obj.x + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_frozen_dataclass_with_slots_construction(self):
        # [FUNCTION DOC] Constructing a frozen slotted dataclass (init bypasses frozen setter via
        #   object setattr) must be traceable and match eager.
        # Case 4: frozen dataclass with slots uses object.__setattr__ in __init__
        # to bypass the frozen __setattr__. Dynamo must allow this for slot descriptors.
        def fn(t):
            # [FUNCTION DOC] HELPER: Build the frozen dataclass and return the tensor plus both
            #   fields.
            obj = FrozenSlots(3, 4)
            return t + obj.x + obj.y

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_setattr_with_slots(self):
        # [FUNCTION DOC] A custom setattr on a slotted class is traced; compiled result must
        #   match eager (values doubled by the hook).
        # Case 5: __slots__ + custom __setattr__ — the custom __setattr__ is traced
        def fn(t):
            # [FUNCTION DOC] HELPER: Create the doubling-setattr object, assign x, return tensor
            #   plus x.
            obj = SlotsAndSetattr(1)
            obj.x = 10
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_dict_valid_slot_assignment(self):
        # [FUNCTION DOC] With slots plus dict, assigning a declared slot still works; compiled
        #   output matches eager.
        # Case 6: __slots__ + __dict__: assigning to a declared slot still works
        def fn(t):
            # [FUNCTION DOC] HELPER: Create the object, overwrite slot x, return tensor plus x.
            obj = SlotsAndDict(1)
            obj.x = 99
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_dict_undeclared_attr_goes_to_dict(self):
        # [FUNCTION DOC] With slots plus dict, an undeclared attribute is stored in the dict
        #   rather than raising; compiled output matches eager.
        # Case 7: __slots__ + __dict__: assigning to an undeclared attr goes to
        # __dict__ instead of raising AttributeError
        def fn(t):
            # [FUNCTION DOC] HELPER: Assign undeclared attribute z and return tensor plus z.
            obj = SlotsAndDict(1)
            obj.z = 42
            return t + obj.z

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_setattr_with_slots_and_dict(self):
        # [FUNCTION DOC] Slots plus dict plus custom setattr: the hook is traced for both slot
        #   and non-slot attributes; compiled output matches eager.
        # Case 8: __slots__ + __dict__ + custom __setattr__ — custom __setattr__
        # is traced for both slot and non-slot attrs
        def fn(t):
            # [FUNCTION DOC] HELPER: Assign x and an extra attribute through the doubling hook
            #   and return tensor plus both.
            obj = SlotsAndDictAndSetattr(1)
            obj.x = 10
            obj.extra = 3
            return t + obj.x + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_inheritance_parent_and_child_slots(self):
        # [FUNCTION DOC] A subclass slot added on top of a parent slot: both attributes are
        #   assignable and readable; compiled output matches eager.
        # Subclass adds its own slot on top of parent's slot — both accessible
        def fn(t):
            # [FUNCTION DOC] HELPER: Create the derived object, set x and y, return tensor plus
            #   both.
            obj = SlotsDerived()
            obj.x = 1
            obj.y = 2
            return t + obj.x + obj.y

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_child_inherits_dict_from_no_slots_parent(self):
        # [FUNCTION DOC] A slotted subclass of a dict-bearing parent accepts arbitrary
        #   attributes; compiled output matches eager.
        # Subclass with __slots__ inheriting from a parent without __slots__
        # gets __dict__ from the parent, so arbitrary attrs are allowed
        def fn(t):
            # [FUNCTION DOC] HELPER: Set the slot z and an undeclared attribute, return tensor
            #   plus both.
            obj = SlotsChildOfPlain()
            obj.z = 1
            obj.extra = 42
            return t + obj.z + obj.extra

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_object_setattr_bypasses_custom_setattr(self):
        # [FUNCTION DOC] The object-level setattr writes the slot directly and skips the custom
        #   (doubling) setattr; compiled output matches eager.
        # object.__setattr__ skips the custom __setattr__ and writes directly to slot
        def fn(t):
            # [FUNCTION DOC] HELPER: Write x through the object-level setattr and return tensor
            #   plus x.
            obj = SlotsAndSetattr(1)
            object.__setattr__(obj, "x", 5)
            return t + obj.x

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_getattr_default_on_unset_slot(self):
        # [FUNCTION DOC] Reading an unset slot with a default through getattr returns the
        #   default; compiled output matches eager.
        # getattr with a default on an unset slot returns the default
        def fn(t):
            # [FUNCTION DOC] HELPER: Fetch the unset slot x with default 99 and add it to the
            #   tensor.
            obj = Slots()
            val = getattr(obj, "x", 99)
            return t + val

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slot_read_after_delete_raises(self):
        # [FUNCTION DOC] Reading a slot after deleting it raises AttributeError in eager and some
        #   exception in the compiled full-graph version.
        # Reading a slot after deletion raises AttributeError in both eager and compiled
        def fn(t):
            # [FUNCTION DOC] HELPER: Set slot x, delete it, then read it (raises).
            obj = Slots()
            obj.x = 1
            del obj.x
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertRaises(AttributeError, fn, t)
        self.assertRaises(Exception, compiled_fn, t)

    def test_slot_shadowed_by_class_attribute(self):
        # [FUNCTION DOC] A class attribute in a subclass shadows the parent slot descriptor, so
        #   reads give the class value (42); compiled output matches eager.
        # Class attribute in subclass shadows parent slot descriptor:
        # reads return the class attribute, writes raise AttributeError
        def fn(t):
            # [FUNCTION DOC] HELPER: Return tensor plus the shadowing class attribute x.
            obj = SlotsShadowed()
            return t + obj.x  # returns class attr 42

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slot_assignment_with_object_as_argument(self):
        # [FUNCTION DOC] A slotted object passed in as an argument can be assigned inside the
        #   compiled function; eager and compiled results are compared on separate tensor clones.
        # Slotted object passed as argument (not created inside fn)
        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Set slot x of the argument object to 10 and return tensor
            #   plus x.
            obj.x = 10
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        obj = Slots()
        self.assertEqual(fn(t.clone(), obj), compiled_fn(t.clone(), obj))

    def test_slot_mutation_materialized_on_argument(self):
        # [FUNCTION DOC] A slot write on an argument object inside a compiled function must be
        #   visible on the real object after the call (side effects are materialized).
        # Slot mutation on an object passed as argument must be visible after
        # the compiled function returns (side effect materialization)
        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Set slot x on the argument object and return the sine of the
            #   tensor.
            obj.x = 10
            return t.sin()

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        obj = Slots()
        compiled_fn(torch.ones(1), obj)
        self.assertEqual(obj.x, 10)

    def test_slot_delete_materialized(self):
        # [FUNCTION DOC] Deleting a slot on an argument object inside a compiled function must be
        #   visible afterwards: the attribute is no longer present.
        # del on a slot inside a compiled fn must be visible after the call returns
        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Delete slot x on the argument object and return the sine of
            #   the tensor.
            del obj.x
            return t.sin()

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        obj = Slots()
        obj.x = 1
        compiled_fn(torch.ones(1), obj)
        self.assertFalse(hasattr(obj, "x"))

    def test_hasattr_on_slotted_object(self):
        # [FUNCTION DOC] hasattr on a slotted object inside compiled code must reflect the
        #   current slot state (unset then set); compiled output matches eager.
        # hasattr inside compiled code reflects actual slot state
        def fn(t):
            # [FUNCTION DOC] HELPER: Record presence of slot x before and after assigning it, add
            #   both booleans to the tensor.
            obj = Slots()
            before = hasattr(obj, "x")  # False — slot not set
            obj.x = 5
            after = hasattr(obj, "x")  # True — slot is now set
            return t + before + after

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_with_property_setter(self):
        # [FUNCTION DOC] A property setter on a slotted class is invoked instead of writing the
        #   slot directly; the getter then returns the doubled value. Compiled output matches
        #   eager.
        # property setter is called instead of writing directly to the slot
        def fn(t):
            # [FUNCTION DOC] HELPER: Assign x through the property and return tensor plus the
            #   property value.
            obj = SlotsAndProperty(1)
            obj.x = 5  # calls setter: _x = 5 * 2 = 10
            return t + obj.x  # calls getter: returns _x = 10

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_direct_dict_write_does_not_shadow_data_descriptor(self):
        # [FUNCTION DOC] Writing into the instance dict directly does not shadow a data
        #   descriptor (property): reading the attribute still gives the property value while the
        #   dict holds the written value. Compiled matches eager.
        class Foo:
            # [CLASS DOC] HELPER: Class with a read-only property x returning 10.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the constant 10.
                return 10

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Write 99 under key x in the instance dict, then return
            #   tensor plus the property value plus the dict entry.
            obj.__dict__["x"] = 99
            return t + obj.x + obj.__dict__["x"]

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        self.assertEqual(fn(t, Foo()), compiled_fn(t, Foo()))

    def test_readonly_property_assignment_raises(self):
        # [FUNCTION DOC] Assigning to a property without a setter raises AttributeError in eager;
        #   the compiled version raises an error whose message mentions that there is no setter.
        class Foo:
            # [CLASS DOC] HELPER: Class with a read-only property x returning 10.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the constant 10.
                return 10

        def fn(obj):
            # [FUNCTION DOC] HELPER: Try to assign x on the argument (raises) then return x.
            obj.x = 99
            return obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertRaises(AttributeError, fn, Foo())
        self.assertRaisesRegex(Exception, "has no setter", compiled_fn, Foo())

    def test_delattr_instance_dict_exposes_non_data_descriptor(self):
        # [FUNCTION DOC] Deleting an instance dict entry that shadowed a non-data descriptor
        #   exposes the descriptor value again. Compiled output matches eager and the compiled
        #   object's dict no longer has the key.
        class Descriptor:
            # [CLASS DOC] HELPER: Non-data descriptor (getter only) that always returns 5.
            def __get__(self, obj, owner):
                # [FUNCTION DOC] HELPER: Descriptor get hook returning 5 regardless of instance
                #   or owner.
                return 5

        class Foo:
            # [CLASS DOC] HELPER: Class with a non-data descriptor x whose instance assigns 7 to
            #   x at construction, placing it in the instance dict.
            x = Descriptor()

            def __init__(self):
                # [FUNCTION DOC] HELPER: Assign 7 to x, which lands in the instance dict because
                #   the descriptor has no setter.
                self.x = 7

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Delete x on the argument object and return tensor plus x
            #   (now the descriptor value).
            del obj.x
            return t + obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertNotIn("x", compiled_obj.__dict__)

    def test_property_deleter(self):
        # [FUNCTION DOC] A property deleter is invoked when the attribute is deleted: it sets a
        #   flag and zeroes the backing value. Compiled output matches eager and the real object
        #   shows the flag and zero afterwards.
        class Foo:
            # [CLASS DOC] HELPER: Class with a property x over a private value (initially 4),
            #   whose deleter sets a deleted flag and zeroes the value.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the deleted flag to false and the private
                #   value to 4.
                self.deleted = False
                self._x = 4

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the private value.
                return self._x

            @x.deleter
            def x(self):
                # [FUNCTION DOC] HELPER: Property deleter marking the instance as deleted and
                #   zeroing the private value.
                self.deleted = True
                self._x = 0

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Delete x on the argument and return tensor plus x plus the
            #   deleted flag.
            del obj.x
            return t + obj.x + obj.deleted

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertTrue(compiled_obj.deleted)
        self.assertEqual(compiled_obj._x, 0)

    def test_property_without_deleter_raises(self):
        # [FUNCTION DOC] Deleting a property that has no deleter raises AttributeError in eager;
        #   the compiled version raises an error mentioning that there is no deleter.
        class Foo:
            # [CLASS DOC] HELPER: Class with a read-only property x returning 10.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the constant 10.
                return 10

        def fn(obj):
            # [FUNCTION DOC] HELPER: Try to delete x on the argument (raises) then return x.
            del obj.x
            return obj.x

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertRaises(AttributeError, fn, Foo())
        self.assertRaisesRegex(Exception, "has no deleter", compiled_fn, Foo())

    def test_slot_and_dict_mutation_same_object(self):
        # [FUNCTION DOC] Mixing slot assignment and direct instance-dict writes on one argument
        #   object: compiled output matches eager and the real object ends with the slot value 2
        #   and dict entry 3.
        class Foo:
            # [CLASS DOC] HELPER: Class with slot x and an instance dict.
            __slots__ = ("x", "__dict__")

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Assign slot x, write key y in the instance dict, return
            #   tensor plus both values.
            obj.x = 2
            obj.__dict__["y"] = 3
            return t + obj.x + obj.__dict__["y"]

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertEqual(compiled_obj.x, 2)
        self.assertEqual(compiled_obj.__dict__["y"], 3)

    def test_dunder_dict_assignment_updates_attribute_lookup(self):
        # [FUNCTION DOC] Replacing the instance dict wholesale must change subsequent attribute
        #   lookup; compiled output matches eager.
        class Foo:
            # [CLASS DOC] HELPER: Class whose only slot is the instance dict.
            __slots__ = ("__dict__",)

        def fn(t):
            # [FUNCTION DOC] HELPER: Replace the dict of a new object with one holding y, then
            #   return tensor plus y read both as attribute and as dict entry.
            obj = Foo()
            obj.__dict__ = {"y": 2}
            return t + obj.y + obj.__dict__["y"]

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_custom_descriptor_shadows_base_slot(self):
        # [FUNCTION DOC] A data descriptor defined in a subclass shadows a slot of the base
        #   class: assignment and reads go through the descriptor (setter stores one more than
        #   the value in y, getter returns double y). Compiled matches eager and y ends as 5.
        class Descriptor:
            # [CLASS DOC] HELPER: Data descriptor: get returns itself for class access else
            #   double the instance y; set stores one more than the given value into y.
            def __get__(self, obj, owner):
                # [FUNCTION DOC] HELPER: Descriptor get hook; returns the descriptor for
                #   class-level access, otherwise twice the instance attribute y.
                if obj is None:
                    return self
                return obj.y * 2

            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Descriptor set hook; stores the value plus one in the
                #   instance attribute y.
                obj.y = value + 1

        class Base:
            # [CLASS DOC] HELPER: Base class with slot x and a dict.
            __slots__ = ("x", "__dict__")

        class Foo(Base):
            # [CLASS DOC] HELPER: Subclass whose class attribute x is the custom descriptor
            #   shadowing the base slot.
            x = Descriptor()

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Assign 4 to x on the argument and return tensor plus x plus
            #   y.
            obj.x = 4
            return t + obj.x + obj.y

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.ones(1)
        compiled_obj = Foo()
        self.assertEqual(fn(t, Foo()), compiled_fn(t, compiled_obj))
        self.assertEqual(compiled_obj.y, 5)

    def test_slot_assignment_no_recompile_same_type(self):
        # [FUNCTION DOC] Calling a compiled function repeatedly with fresh instances of one
        #   slotted type compiles only once (frame count 1).
        # Calling compiled fn repeatedly with the same slotted object type
        # must not trigger recompilation
        cnts = dynamo_testing.CompileCounter()

        def fn(t, obj):
            # [FUNCTION DOC] HELPER: Set slot x to 10 and return tensor plus x.
            obj.x = 10
            return t + obj.x

        compiled_fn = torch.compile(fn, backend=cnts)
        t = torch.ones(1)
        compiled_fn(t, Slots())
        compiled_fn(t, Slots())
        compiled_fn(t, Slots())
        self.assertEqual(cnts.frame_count, 1)

    def test_slot_assignment_recompiles_on_type_change(self):
        # [FUNCTION DOC] Assigning an int and then a float to the slot: same type reuses the
        #   compiled frame (count 1), a float triggers recompilation (count 2), and the result
        #   matches eager.
        # Compiled fn sees slot assigned to int first, then float — guards recompile
        cnts = dynamo_testing.CompileCounter()

        def fn(t, a, obj):
            # [FUNCTION DOC] HELPER: Store the second argument into slot x and return tensor plus
            #   x.
            obj.x = a
            return t + obj.x

        compiled_fn = torch.compile(fn, backend=cnts)
        t = torch.ones(1)

        compiled_fn(t, 1, Slots())
        compiled_fn(t, 1, Slots())
        self.assertEqual(cnts.frame_count, 1)  # same type, no recompile

        x = t.clone()
        res = compiled_fn(x, 1.0, Slots())
        self.assertEqual(cnts.frame_count, 2)  # float instead of int — recompile
        self.assertEqual(res, fn(x, 1.0, Slots()))


class WithGetattribute:
    # [CLASS DOC] HELPER: Fixture defining a getattribute override that records every accessed
    #   attribute name in a set. Instances are initialized with the object-level setattr to avoid
    #   the hook. (Not referenced by tests in this file.)
    # __slots__ = ("x", "_side_effects")

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Write x and an empty set of recorded effects using the
        #   object-level setattr.
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "_side_effects", set())

    def __getattribute__(self, name):
        # [FUNCTION DOC] HELPER: Attribute access hook: add the requested name to the recorded
        #   set, then return the attribute through the object-level getattribute.
        effects = object.__getattribute__(self, "_side_effects")
        effects.add(name)
        return object.__getattribute__(self, name)


class TestSlotsFromCPython(TestCase):
    # [CLASS DOC] Slot behavior tests adapted from the CPython descriptor tests, run through
    #   Dynamo with unittest tracing enabled. Each test compiles a function that uses assertions
    #   on slotted classes.
    """Slot tests extracted from CPython's test_descr.py::test_slots."""

    def setUp(self):
        # [FUNCTION DOC] Save the unittest-tracing config flag and enable it for the test.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Restore the unittest-tracing config flag saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    def test_slots_empty(self):
        # [FUNCTION DOC] An empty slot list gives instances with no dict and no arbitrary
        #   attributes; compiled run matches eager.
        class C:
            # [CLASS DOC] HELPER: Class with an empty slot list.
            __slots__ = []

        def fn(t):
            # [FUNCTION DOC] HELPER: Check that a fresh instance has neither a dict nor a foo
            #   attribute; return sine of tensor.
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "foo"))
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_single(self):
        # [FUNCTION DOC] A single slot named a: unset at first, assignable, can hold None,
        #   deletable; compiled run matches eager.
        class C:
            # [CLASS DOC] HELPER: Class with the single slot a.
            __slots__ = ["a"]

        def fn(t):
            # [FUNCTION DOC] HELPER: Exercise set, read, set to None, delete and hasattr on slot
            #   a with assertions; return sine of tensor.
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "a"))
            x.a = 1
            self.assertEqual(x.a, 1)
            x.a = None
            self.assertEqual(x.a, None)
            del x.a
            self.assertFalse(hasattr(x, "a"))
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_multiple(self):
        # [FUNCTION DOC] Three slots a, b, c are independently unset, assignable and readable;
        #   compiled run matches eager.
        class C:
            # [CLASS DOC] HELPER: Class with slots a, b and c.
            __slots__ = ["a", "b", "c"]

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert all slots unset, assign 1, 2, 3 and verify each;
            #   return sine of tensor.
            x = C()
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "a"))
            self.assertFalse(hasattr(x, "b"))
            self.assertFalse(hasattr(x, "c"))
            x.a = 1
            x.b = 2
            x.c = 3
            self.assertEqual(x.a, 1)
            self.assertEqual(x.b, 2)
            self.assertEqual(x.c, 3)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_name_mangling(self):
        # [FUNCTION DOC] Slot names beginning with double underscore are mangled: the class
        #   method can use them but outside assignment of the unmangled name must raise
        #   AttributeError. Compiled run matches eager.
        class C:
            # [CLASS DOC] HELPER: Class with a name-mangled slot, set in init and read by a
            #   getter method.
            __slots__ = ["__a"]

            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Store the value in the mangled slot.
                self.__a = value

            def get(self):
                # [FUNCTION DOC] HELPER: Return the mangled slot value.
                return self.__a

        def fn(t):
            # [FUNCTION DOC] HELPER: Check getter result, then try assigning the unmangled name
            #   from outside the class; fails the test if no AttributeError. Return sine of
            #   tensor.
            x = C(5)
            self.assertFalse(hasattr(x, "__dict__"))
            self.assertFalse(hasattr(x, "__a"))
            self.assertEqual(x.get(), 5)
            try:
                x.__a = 6
            except AttributeError:
                pass
            else:
                self.fail("Double underscored names not mangled")
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_string_not_expanded(self):
        # [FUNCTION DOC] A slots declaration that is a single string is a one-name slot, not a
        #   sequence of characters; compiled run matches eager.
        # A single string is not expanded as a sequence
        class C:
            # [CLASS DOC] HELPER: Class whose slots declaration is the single string abc.
            __slots__ = "abc"  # noqa: PLC0205

        def fn(t):
            # [FUNCTION DOC] HELPER: Set and read attribute abc with an assertion; return sine of
            #   tensor.
            c = C()
            c.abc = 5
            self.assertEqual(c.abc, 5)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_tuple(self):
        # [FUNCTION DOC] A tuple held in a variable can be used as the slots declaration;
        #   compiled run matches eager.
        slots = ("foo", "bar")

        class C:
            # [CLASS DOC] HELPER: Class using the outer slots tuple of names foo and bar.
            __slots__ = slots

        def fn(t):
            # [FUNCTION DOC] HELPER: Set and verify foo, assert the first slot name has type str;
            #   return sine of tensor.
            x = C()
            x.foo = 5
            self.assertEqual(x.foo, 5)
            self.assertIs(type(slots[0]), str)
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_get_unset_raises(self):
        # [FUNCTION DOC] Reading a declared but unset slot raises AttributeError; compiled run
        #   matches eager.
        class X:
            # [CLASS DOC] HELPER: Class with the single slot a declared by a plain string.
            __slots__ = "a"  # noqa: PLC0205

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert that reading slot a on a fresh instance raises
            #   AttributeError; return sine of tensor.
            with self.assertRaises(AttributeError):
                X().a
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_str_subclass(self):
        # [FUNCTION DOC] A str subclass instance used as a slot name works (regression for a
        #   CPython issue); compiled run matches eager.
        # gh-98783: string subclass in __slots__
        class SubStr(str):  # noqa: SLOT000
            # [CLASS DOC] HELPER: Trivial str subclass used as a slot name.
            pass

        class X:
            # [CLASS DOC] HELPER: Class whose single slot name is an instance of the str
            #   subclass.
            __slots__ = (SubStr("x"),)

        def fn(t):
            # [FUNCTION DOC] HELPER: Assign the slot on a fresh instance and assert that reading
            #   an undeclared name raises AttributeError; return sine of tensor.
            X().x = 1
            with self.assertRaises(AttributeError):
                X().a
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_dict(self):
        # [FUNCTION DOC] Declaring a dict slot gives the instance a dict (but no weakref slot)
        #   and allows arbitrary attributes; compiled run matches eager.
        # __dict__ in __slots__ enables arbitrary attr assignment
        class D:
            # [CLASS DOC] HELPER: Class with only the dict slot.
            __slots__ = ["__dict__"]

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert dict present and weakref absent, set foo, assert the
            #   dict contents; return sine of tensor.
            a = D()
            self.assertTrue(hasattr(a, "__dict__"))
            self.assertFalse(hasattr(a, "__weakref__"))
            a.foo = 42
            self.assertEqual(a.__dict__, {"foo": 42})
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_weakref(self):
        # [FUNCTION DOC] Declaring a weakref slot gives weak reference support without a dict, so
        #   arbitrary attribute assignment raises AttributeError; compiled run matches eager.
        # __weakref__ in __slots__ — no __dict__, arbitrary attr raises
        class W:
            # [CLASS DOC] HELPER: Class with only the weakref slot.
            __slots__ = ["__weakref__"]

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert weakref present and dict absent, and that assigning
            #   foo raises AttributeError; return sine of tensor.
            a = W()
            self.assertTrue(hasattr(a, "__weakref__"))
            self.assertFalse(hasattr(a, "__dict__"))
            with self.assertRaises(AttributeError):
                a.foo = 42
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_special_inherit_dict_weakref(self):
        # [FUNCTION DOC] A class inheriting from one base with a dict slot and one with a weakref
        #   slot has both; compiled run matches eager.
        # Inheriting from both __dict__ and __weakref__ slot classes
        class D:
            # [CLASS DOC] HELPER: Base with only the dict slot.
            __slots__ = ["__dict__"]

        class W:
            # [CLASS DOC] HELPER: Base with only the weakref slot.
            __slots__ = ["__weakref__"]

        class C1(W, D):
            # [CLASS DOC] HELPER: Empty-slot class inheriting from both bases.
            __slots__ = []

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert dict and weakref present, set foo, assert the dict
            #   contents; return sine of tensor.
            a = C1()
            self.assertTrue(hasattr(a, "__dict__"))
            self.assertTrue(hasattr(a, "__weakref__"))
            a.foo = 42
            self.assertEqual(a.__dict__, {"foo": 42})
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    @unittest.expectedFailure
    def test_slots_special2_classcell(self):
        # [FUNCTION DOC] Expected-failure test (decorated as such): classcell handling with a
        #   metaclass that checks the namespace and a class declaring the classcell name as a
        #   slot. Dynamo does not yet support this, so the test is marked as expected to fail.
        # Testing __classcell__ in __slots__
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that asserts the given attribute name is in the class
            #   namespace before creating the class.
            def __new__(metacls, name, bases, namespace, attr):
                # [FUNCTION DOC] HELPER: Metaclass creation hook; asserts the named attribute is
                #   in the namespace using the enclosing test case, then builds the class via the
                #   parent type creation.
                self.assertIn(attr, namespace)
                return super().__new__(metacls, name, bases, namespace)

        class C1:
            # [CLASS DOC] HELPER: Base class whose init sets b to 42.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set b to 42.
                self.b = 42

        class C2(C1, metaclass=Meta, attr="__classcell__"):
            # [CLASS DOC] HELPER: Class using the checking metaclass with a classcell slot.
            __slots__ = ["__classcell__"]

            def __init__(self):
                # [FUNCTION DOC] HELPER: Call the parent initializer.
                super().__init__()

        def fn(t):
            # [FUNCTION DOC] HELPER: Check the classcell slot is a member descriptor, init sets
            #   b, the slot can be set, and defining a class with a conflicting class attribute
            #   and slot raises TypeError; return sine of tensor.
            self.assertIsInstance(
                C2.__dict__["__classcell__"], types.MemberDescriptorType
            )
            c = C2()
            self.assertEqual(c.b, 42)
            self.assertFalse(hasattr(c, "__classcell__"))
            c.__classcell__ = 42
            self.assertEqual(c.__classcell__, 42)
            with self.assertRaises(TypeError):

                class C3:
                    # [CLASS DOC] HELPER: Class intentionally conflicting a class attribute with
                    #   a slot of the same name; its definition is expected to raise TypeError.
                    __classcell__ = 42
                    __slots__ = ["__classcell__"]

            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_slots_multiple_inheritance(self):
        # [FUNCTION DOC] Multiple inheritance of a no-slots class with an empty-slot class still
        #   yields a dict and weakref support; compiled run matches eager.
        # SF bug 575229: multiple inheritance w/ slots dumps core
        class A:
            # [CLASS DOC] HELPER: Class with empty slots.
            __slots__ = ()

        class B:
            # [CLASS DOC] HELPER: Plain class with a dict.
            pass

        class C(A, B):
            # [CLASS DOC] HELPER: Empty-slot subclass of both.
            __slots__ = ()

        def fn(t):
            # [FUNCTION DOC] HELPER: Assert dict and weakref exist on the class, set an arbitrary
            #   attribute on an instance; return sine of tensor.
            self.assertTrue(hasattr(C, "__dict__"))
            self.assertTrue(hasattr(C, "__weakref__"))
            C().x = 2
            return t.sin()

        dynamo_testing.standard_test(self, fn, nargs=1)


class TestUserDefinedClassDict(TestCase):
    # [CLASS DOC] Tests reading, via the class dict view, of user-defined class attributes under
    #   compilation, including guard-driven recompilation when the class is mutated.
    def test_class_dict_read(self):
        # [FUNCTION DOC] Reads from the class dict (item lookup, get with defaults, membership)
        #   in a compiled function match eager.
        class MyClass:
            # [CLASS DOC] HELPER: Class with attribute x equal to 3.
            x = 3

        def fn(t):
            # [FUNCTION DOC] HELPER: Add to the tensor the class dict values: item x, get x, get
            #   missing key with default 99, and 1 flags for presence of x and z.
            t = t + MyClass.__dict__["x"]
            t = t + MyClass.__dict__.get("x", 0)
            t = t + MyClass.__dict__.get("z", 99)
            t = t + (1 if "x" in MyClass.__dict__ else 0)
            t = t + (1 if "z" in MyClass.__dict__ else 0)
            return t

        dynamo_testing.standard_test(self, fn, nargs=1)

    def test_class_dict_via_arg(self):
        # [FUNCTION DOC] Class passed as an argument: the class dict get for x is added to the
        #   tensor and the result equals the value 7.
        class MyClass:
            # [CLASS DOC] HELPER: Class with attribute x equal to 7.
            x = 7

        def fn(t, cls):
            # [FUNCTION DOC] HELPER: Return tensor plus the class dict entry x (default 0).
            return t + cls.__dict__.get("x", 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)
        result = compiled(torch.tensor([0.0]), MyClass)
        self.assertEqual(result, torch.tensor([7.0]))

    def test_class_dict_mutation_recompiles(self):
        # [FUNCTION DOC] Mutating a class attribute between calls triggers recompilation and the
        #   new value is observed: result 1 then 10, frame count 1 then 2.
        # Mutating a class attribute between calls should trigger recompilation,
        # and the compiled function should see the updated value.
        class MyClass:
            # [CLASS DOC] HELPER: Class with attribute x equal to 1 (later mutated by the test).
            x = 1

        def fn(t):
            # [FUNCTION DOC] HELPER: Return tensor plus the class dict entry x.
            return t + MyClass.__dict__["x"]

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([1.0]))
        self.assertEqual(cnt.frame_count, 1)

        MyClass.x = 10
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([10.0]))
        # Should have recompiled due to guard failure
        self.assertEqual(cnt.frame_count, 2)

    def test_class_dict_add_key_recompiles(self):
        # [FUNCTION DOC] Adding a new class attribute changes the result of a compiled membership
        #   check and forces recompilation: result 0 then 1, frame count 1 then 2.
        # Adding a new attribute to the class should trigger recompilation
        # when the compiled code checks for key presence.
        class MyClass:
            # [CLASS DOC] HELPER: Class with attribute x only (y added by the test later).
            x = 1

        def fn(t):
            # [FUNCTION DOC] HELPER: Return tensor plus a flag for whether y is in the class
            #   dict.
            return t + (1 if "y" in MyClass.__dict__ else 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([0.0]))
        self.assertEqual(cnt.frame_count, 1)

        MyClass.y = 99
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([1.0]))
        # Should have recompiled
        self.assertEqual(cnt.frame_count, 2)

    def test_class_dict_delete_key_recompiles(self):
        # [FUNCTION DOC] Deleting a class attribute forces recompilation and changes the result
        #   of a get on the class dict: result 10 then 0, frame count 1 then 2.
        # Deleting a class attribute should trigger recompilation.
        class MyClass:
            # [CLASS DOC] HELPER: Class with attributes x (5) and y (10); y is deleted by the
            #   test.
            x = 5
            y = 10

        def fn(t):
            # [FUNCTION DOC] HELPER: Return tensor plus the class dict entry y (default 0).
            return t + MyClass.__dict__.get("y", 0)

        cnt = dynamo_testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnt)

        result1 = compiled(torch.tensor([0.0]))
        self.assertEqual(result1, torch.tensor([10.0]))
        self.assertEqual(cnt.frame_count, 1)

        del MyClass.y
        result2 = compiled(torch.tensor([0.0]))
        self.assertEqual(result2, torch.tensor([0.0]))
        # Should have recompiled
        self.assertEqual(cnt.frame_count, 2)


class TestClassSetattr(TestCase):
    # [CLASS DOC] Tests assignment to a class attribute from inside a compiled function.
    def test_setattr_class_attribute(self):
        # [FUNCTION DOC] A compiled full-graph function that sets a class attribute to 20 and
        #   reads it back returns 20; the test restores the attribute to 10 afterwards to undo
        #   the side effect.
        class MyModule:
            # [CLASS DOC] HELPER: Class with attribute x equal to 10.
            x = 10

        def fn():
            # [FUNCTION DOC] HELPER: Set the class attribute x to 20 and return it.
            MyModule.x = 20
            return MyModule.x

        opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        result = opt_fn()
        self.assertEqual(result, 20)

        MyModule.x = 10


# ---------------------------------------------------------------------------
# __setitem__ on user-defined classes / metaclasses
# ---------------------------------------------------------------------------


# Metaclasses kept at module level — Dynamo's traced LOAD_BUILD_CLASS does
# not currently propagate the `metaclass=` kwarg.


class _SetitemMetaBasic(type):
    # [CLASS DOC] HELPER: Metaclass supporting item assignment and lookup on classes, backed by a
    #   store mapping on the class.
    def __setitem__(cls, key, value):
        # [FUNCTION DOC] HELPER: Metaclass item assignment hook; stores the value under the key
        #   in the class store mapping.
        cls._store[key] = value

    def __getitem__(cls, key):
        # [FUNCTION DOC] HELPER: Metaclass item lookup hook; returns the value stored under the
        #   key (raises KeyError if absent).
        return cls._store[key]


class _ClassWithBasicMeta(metaclass=_SetitemMetaBasic):
    # [CLASS DOC] HELPER: Class using the basic item metaclass, with a class-level store mapping.
    _store: dict = {}


class _SetitemMetaPerClass(type):
    # [CLASS DOC] HELPER: Metaclass like the basic one but backed by a mapping named entries on
    #   the class.
    def __setitem__(cls, key, value):
        # [FUNCTION DOC] HELPER: Metaclass item assignment hook; stores the value under the key
        #   in the entries mapping.
        cls.entries[key] = value

    def __getitem__(cls, key):
        # [FUNCTION DOC] HELPER: Metaclass item lookup hook; returns the value stored under the
        #   key in entries.
        return cls.entries[key]


class _PerClassEntries(metaclass=_SetitemMetaPerClass):
    # [CLASS DOC] HELPER: Class using the per-class entries metaclass, with a class-level entries
    #   mapping.
    entries: dict = {}


class _SetitemMetaValidating(type):
    # [CLASS DOC] HELPER: Metaclass whose item assignment validates that keys are strings.
    def __setitem__(cls, key, value):
        # [FUNCTION DOC] HELPER: Metaclass item assignment hook; raises TypeError for a
        #   non-string key, otherwise stores the value in the class registry mapping.
        if not isinstance(key, str):
            raise TypeError("class registry expects string keys")
        cls.registry[key] = value


class _ValidatingClass(metaclass=_SetitemMetaValidating):
    # [CLASS DOC] HELPER: Class using the validating metaclass, with a class-level registry
    #   mapping.
    registry: dict = {}


class _SetitemDelitemMeta(type):
    # [CLASS DOC] HELPER: Metaclass supporting item assignment, lookup and deletion on classes,
    #   backed by a store mapping on the class.
    def __setitem__(cls, key, value):
        # [FUNCTION DOC] HELPER: Metaclass item assignment hook; stores the value under the key
        #   in the class store mapping.
        cls._store[key] = value

    def __getitem__(cls, key):
        # [FUNCTION DOC] HELPER: Metaclass item lookup hook; returns the stored value (raises
        #   KeyError if absent).
        return cls._store[key]

    def __delitem__(cls, key):
        # [FUNCTION DOC] HELPER: Metaclass item deletion hook; removes the key from the class
        #   store mapping.
        del cls._store[key]


class _DelClassMeta(metaclass=_SetitemDelitemMeta):
    # [CLASS DOC] HELPER: Class using the delete-capable metaclass, with a class-level store
    #   mapping.
    _store: dict = {}


class TestUserDefinedSetitem(TestCase):
    # [CLASS DOC] Dynamo tests of item assignment and deletion hooks on instances of user classes
    #   and on classes with a metaclass, run as traced bodies. Setup enables unittest tracing and
    #   tracing of class definitions inside test bodies.
    """__setitem__ on user-defined classes (UDOV) and metaclasses (UDCV).

    enable_trace_load_build_class lets us define helper classes inside the
    test body — keeps the helper next to the assertion that exercises it.
    """

    def setUp(self):
        # [FUNCTION DOC] Save the unittest-tracing and class-definition-tracing config flags,
        #   then enable both.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        # [FUNCTION DOC] Restore both config flags saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # -- instance __setitem__ --

    @make_dynamo_test
    def test_validating_ok(self):
        # [FUNCTION DOC] An item-assignment hook that validates key type (TypeError) and value
        #   sign (ValueError) works under tracing; valid assignment is readable back.
        class V:
            # [CLASS DOC] HELPER: Container with a dict, item lookup, and item assignment that
            #   rejects non-string keys and negative values.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Item assignment: raise TypeError for a non-string key,
                #   ValueError for a negative value, otherwise store it.
                if not isinstance(key, str):
                    raise TypeError("only string keys allowed")
                if value < 0:
                    raise ValueError("negative values forbidden")
                self.data[key] = value

        obj = V()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)
        with self.assertRaises(TypeError):
            obj[1] = 5
        with self.assertRaises(ValueError):
            obj["a"] = -1

    @make_dynamo_test
    def test_transforming_value(self):
        # [FUNCTION DOC] An item-assignment hook can transform the stored value (doubling);
        #   reading back gives 10 for an assigned 5.
        class T:
            # [CLASS DOC] HELPER: Container whose item assignment stores double the value.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Item assignment storing twice the value.
                self.data[key] = value * 2

        obj = T()
        obj["a"] = 5
        self.assertEqual(obj["a"], 10)

    @make_dynamo_test
    def test_inherited_method(self):
        # [FUNCTION DOC] A subclass inherits item assignment and lookup from its base without
        #   overriding them.
        class Base:
            # [CLASS DOC] HELPER: Base container storing values unchanged in a dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value unchanged under the key.
                self.data[key] = value

        class Derived(Base):
            # [CLASS DOC] HELPER: Empty subclass inheriting all behavior.
            pass

        obj = Derived()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_overriding_method(self):
        # [FUNCTION DOC] A subclass overriding item assignment (adds 100) takes precedence over
        #   the base; read back gives 105 for an assigned 5.
        class Base:
            # [CLASS DOC] HELPER: Base container storing values unchanged in a dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value unchanged under the key.
                self.data[key] = value

        class Override(Base):
            # [CLASS DOC] HELPER: Subclass whose item assignment adds 100 to the value.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value plus 100 under the key.
                self.data[key] = value + 100

        obj = Override()
        obj["a"] = 5
        self.assertEqual(obj["a"], 105)

    @make_dynamo_test
    def test_return_value_ignored(self):
        # [FUNCTION DOC] The return value of an item assignment hook is ignored; assignment still
        #   stores the value.
        class R:
            # [CLASS DOC] HELPER: Container whose item assignment hook returns a string.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value, then return an ignored string.
                self.data[key] = value
                return "ignored"

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

        obj = R()
        obj["a"] = 5
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_side_effects_in_method(self):
        # [FUNCTION DOC] Side effects performed inside an item assignment hook (last key, last
        #   value, call count) are all visible after two assignments.
        class S:
            # [CLASS DOC] HELPER: Container recording the last key, last value and a call count
            #   on every item assignment.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the data dict, last key and value as none and
                #   count as zero.
                self.data = {}
                self.last_key = None
                self.last_value = None
                self.call_count = 0

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Record key and value, increment the call count and store
                #   the value.
                self.last_key = key
                self.last_value = value
                self.call_count += 1
                self.data[key] = value

        obj = S()
        obj["a"] = 1
        obj["b"] = 2
        self.assertEqual(obj.call_count, 2)
        self.assertEqual(obj.last_key, "b")
        self.assertEqual(obj.last_value, 2)
        self.assertEqual(obj["a"], 1)
        self.assertEqual(obj["b"], 2)

    @make_dynamo_test
    def test_explicit_method_call(self):
        # [FUNCTION DOC] Calling the item assignment method explicitly has the same effect as the
        #   subscript assignment.
        class B:
            # [CLASS DOC] HELPER: Container storing values in a dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value under the key.
                self.data[key] = value

        obj = B()
        obj.__setitem__("a", 5)
        self.assertEqual(obj["a"], 5)

    @make_dynamo_test
    def test_multiple_keys(self):
        # [FUNCTION DOC] Assigning several integer keys in a loop and reading each back works.
        class B:
            # [CLASS DOC] HELPER: Container storing values in a dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with an empty data dict.
                self.data = {}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key.
                return self.data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the value under the key.
                self.data[key] = value

        obj = B()
        for i in range(5):
            obj[i] = i * 10
        for i in range(5):
            self.assertEqual(obj[i], i * 10)

    @make_dynamo_test
    def test_no_setitem_raises_typeerror(self):
        # [FUNCTION DOC] Item assignment on an object whose class lacks the assignment hook
        #   raises TypeError.
        class N:
            # [CLASS DOC] HELPER: Container with only item lookup over given data.
            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Keep the given data.
                self.data = data

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the element for the key.
                return self.data[key]

        obj = N([1, 2, 3])
        with self.assertRaises(TypeError):
            obj[0] = 100

    # -- metaclass __setitem__: Cls[k] = v --

    @make_dynamo_test
    def test_metaclass_basic(self):
        # [FUNCTION DOC] Item assignment on a class whose metaclass defines the hook stores into
        #   the class store and reads back.
        _ClassWithBasicMeta["a"] = 1
        self.assertEqual(_ClassWithBasicMeta["a"], 1)

    @make_dynamo_test
    def test_metaclass_multiple(self):
        # [FUNCTION DOC] Two keys assigned through a metaclass hook that uses an entries mapping
        #   are both retrievable.
        _PerClassEntries["x"] = 10
        _PerClassEntries["y"] = 20
        self.assertEqual(_PerClassEntries["x"], 10)
        self.assertEqual(_PerClassEntries["y"], 20)

    @make_dynamo_test
    def test_metaclass_validating(self):
        # [FUNCTION DOC] A metaclass hook that validates keys stores a string key and raises
        #   TypeError for an integer key.
        _ValidatingClass["k"] = 99
        self.assertEqual(_ValidatingClass.registry["k"], 99)
        with self.assertRaises(TypeError):
            _ValidatingClass[123] = 99

    # -- __delitem__ on user-defined classes --

    @make_dynamo_test
    def test_delitem_basic(self):
        # [FUNCTION DOC] Item deletion on an instance removes the key from its data; remaining
        #   entries are intact.
        class D:
            # [CLASS DOC] HELPER: Container with initial data of two keys, lookup and deletion.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with data holding a and b.
                self.data = {"a": 1, "b": 2}

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the value for the key.
                return self.data[key]

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Delete the key from the data.
                del self.data[key]

        obj = D()
        del obj["a"]
        self.assertNotIn("a", obj.data)
        self.assertEqual(obj.data, {"b": 2})

    @make_dynamo_test
    def test_delitem_tracks(self):
        # [FUNCTION DOC] The deletion hook is called for each deletion and its side effect (list
        #   of deleted keys) is visible.
        class D:
            # [CLASS DOC] HELPER: Container with three initial keys and a list of deleted keys.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with data for a, b, c and an empty deleted list.
                self.data = {"a": 1, "b": 2, "c": 3}
                self.deleted = []

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Append the key to the deleted list and delete it from
                #   the data.
                self.deleted.append(key)
                del self.data[key]

        obj = D()
        del obj["a"]
        del obj["c"]
        self.assertEqual(obj.deleted, ["a", "c"])
        self.assertEqual(obj.data, {"b": 2})

    @make_dynamo_test
    def test_delitem_validating(self):
        # [FUNCTION DOC] A deletion hook can validate the key type: an integer key deletes, a
        #   string key raises TypeError.
        class D:
            # [CLASS DOC] HELPER: Container with one integer key whose deletion hook rejects
            #   non-integer keys.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with data holding one integer key.
                self.data = {1: "a"}

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Raise TypeError for non-integer keys, otherwise delete
                #   the key.
                if not isinstance(key, int):
                    raise TypeError("only int keys")
                del self.data[key]

        obj = D()
        del obj[1]
        self.assertEqual(obj.data, {})

        obj2 = D()
        with self.assertRaises(TypeError):
            del obj2["nope"]

    @make_dynamo_test
    def test_delitem_explicit_method_call(self):
        # [FUNCTION DOC] Calling the deletion method explicitly has the same effect as subscript
        #   deletion.
        class D:
            # [CLASS DOC] HELPER: Container with one key and a deletion hook.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with data holding one key.
                self.data = {"a": 1}

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Delete the key from the data.
                del self.data[key]

        obj = D()
        obj.__delitem__("a")
        self.assertEqual(obj.data, {})

    @make_dynamo_test
    def test_delitem_no_method_typeerror(self):
        # [FUNCTION DOC] Item deletion on an object whose class has no deletion hook raises
        #   TypeError.
        class D:
            # [CLASS DOC] HELPER: Container with list data and lookup only.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Start with a three-element list.
                self.data = [1, 2, 3]

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the element for the index.
                return self.data[key]

        obj = D()
        with self.assertRaises(TypeError):
            del obj[0]

    # -- metaclass __delitem__: del Cls[k] --

    @make_dynamo_test
    def test_metaclass_delitem(self):
        # [FUNCTION DOC] Item deletion on a class with a metaclass hook removes the key from the
        #   class store; the other key remains.
        _DelClassMeta["x"] = 1
        _DelClassMeta["y"] = 2
        del _DelClassMeta["x"]
        self.assertNotIn("x", _DelClassMeta._store)
        self.assertEqual(_DelClassMeta["y"], 2)


class TestObjectConstruction(TestCase):
    # [CLASS DOC] Tests for creation and use of plain object instances under Dynamo.
    @make_dynamo_test
    def test_object_call_identity(self):
        # [FUNCTION DOC] Identity and type checks on two distinct plain object instances behave
        #   as expected under tracing.
        a = object()
        b = object()
        self.assertEqual(a is a, True)
        self.assertEqual(a is b, False)
        self.assertEqual(type(a) is object, True)

    @make_dynamo_test
    def test_object_call_as_sentinel(self):
        # [FUNCTION DOC] A plain object used as a sentinel is unequal to other values and equal
        #   to itself.
        sentinel = object()
        self.assertEqual(sentinel == 1, False)
        self.assertEqual(sentinel == sentinel, True)

    def test_object_call_escapes_graph_breaks(self):
        # [FUNCTION DOC] When a freshly created plain object escapes the compiled region as a
        #   return value, Dynamo graph-breaks instead of failing: the returned value is a real
        #   object instance and the compile counter records no frames (count 0).
        # A bare object() that escapes the compiled region is opaque and
        # sourceless, so reconstruction graph-breaks (runs in eager) rather
        # than failing; the returned value is a real object instance.
        cnt = dynamo_testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input plus one together with a new plain object.
            return x + 1, object()

        _, s = fn(torch.randn(3))
        self.assertIs(type(s), object)
        self.assertEqual(cnt.frame_count, 0)


if __name__ == "__main__":
    run_tests()
