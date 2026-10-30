# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests for TorchBind custom C++ classes exposed under the torch classes test
# namespace (stack string, foo, pickle tester, property, lambda, static method and default
# argument classes). Covers construction and method calls in eager and TorchScript mode,
# argument and return passing, properties, serialization, deepcopy, tracing, profiling and error
# messages. Importing the module appends the test directory to the system path; the test library
# is loaded in setUp, not at import.

import copy
import io
import os
import sys
from typing import Optional

import torch
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib


@skipIfTorchDynamo("skipping as a precaution")
class TestTorchbind(JitTestCase):
    # [CLASS DOC] JitTestCase suite for TorchBind classes, skipped under TorchDynamo. setUp is
    # overridden to avoid the JIT emit hooks of the base class, which crash on TorchBind static
    # methods, and to record the torch function mode stack length and torch function state for
    # the inherited state check on teardown.
    def setUp(self):
        # [FUNCTION DOC] Records the torch function mode stack length and torch function state
        # in instance attributes and loads the TorchBind test library (registering its classes
        # and ops). Intentionally does not call the parent setUp. Side effect: loads a custom
        # library into the process.
        # Don't call super().setUp() — JitTestCase.setUp installs JIT emit
        # hooks that crash on torchbind static methods (can't downcast to
        # GraphFunction). Record state baselines that tearDown checks for.
        self._prev_torch_function_mode_stack_len = torch._C._len_torch_function_stack()
        self._prev_torch_function_state = torch._C._get_torch_function_state()
        load_torchbind_test_lib()

    def test_torchbind(self):
        # [FUNCTION DOC] Checks that TorchBind objects behave the same in eager and scripted
        # functions for Foo and StackString usage, that passing a string to the integer-only
        # increment method raises a RuntimeError mentioning an expected int, and that a module
        # whose prepare-scriptable hook swaps a non-scriptable helper for a real TorchBind Foo
        # can be scripted.
        def test_equality(f, cmp_key):
            # [FUNCTION DOC] HELPER: Runs a function eagerly and scripted, and returns the pair
            # of comparison keys computed by cmp_key for both results.
            obj1 = f()
            obj2 = torch.jit.script(f)()
            return (cmp_key(obj1), cmp_key(obj2))

        def f():
            # [FUNCTION DOC] HELPER: Creates a Foo with arguments 5 and 3, increments it by 1,
            # and returns it.
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment(1)
            return val

        test_equality(f, lambda x: x)

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int'"):
            val = torch.classes._TorchScriptTesting._Foo(5, 3)
            val.increment("foo")

        def f():
            # [FUNCTION DOC] HELPER: Creates a StackString from two strings and returns the
            # popped top item.
            ss = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            return ss.pop()

        test_equality(f, lambda x: x)

        def f():
            # [FUNCTION DOC] HELPER: Creates two StackStrings, pushes the popped item of the
            # second onto the first, and returns the concatenation of one pop from each.
            ss1 = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["111", "222"])
            ss1.push(ss2.pop())
            return ss1.pop() + ss2.pop()

        test_equality(f, lambda x: x)

        # test nn module with prepare_scriptable function
        class NonJitableClass:
            # [CLASS DOC] HELPER: Plain Python class (not scriptable) holding two ints.
            def __init__(self, int1, int2):
                # [FUNCTION DOC] HELPER: Stores the two ints as attributes.
                self.int1 = int1
                self.int2 = int2

            def return_vals(self):
                # [FUNCTION DOC] HELPER: Returns the two stored ints as a tuple.
                return self.int1, self.int2

        class CustomWrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an object with a return_vals method. Its
            # prepare-scriptable hook rebuilds the wrapper around a TorchBind Foo made from
            # those values so scripting works.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores the wrapped object as foo.
                super().__init__()
                self.foo = foo

            def forward(self) -> None:
                # [FUNCTION DOC] HELPER: Increments the wrapped Foo by 1 and returns nothing.
                self.foo.increment(1)
                return

            def __prepare_scriptable__(self):
                # [FUNCTION DOC] HELPER: Reads the two ints from the wrapped object, builds a
                # TorchBind Foo from them, and returns a new wrapper around it.
                int1, int2 = self.foo.return_vals()
                foo = torch.classes._TorchScriptTesting._Foo(int1, int2)
                return CustomWrapper(foo)

        foo = CustomWrapper(NonJitableClass(1, 2))
        jit_foo = torch.jit.script(foo)

    def test_torchbind_take_as_arg(self):
        # [FUNCTION DOC] Publishes StackString as a module global (needed for local name
        # resolution of the annotation), scripts a function taking a StackString argument and
        # pushing an item, and checks the returned object pops that item.
        global StackString  # see [local resolution in python]
        StackString = torch.classes._TorchScriptTesting._StackString

        # [FUNCTION DOC] HELPER: Scripted function taking a StackString by type comment
        # annotation, pushing the string lel onto it and returning it.
        # KNOWN ISSUE: this block sits above the def line because the type comment must stay the
        #     first body line for TorchScript to parse it.
        def foo(stackstring):
            # type: (StackString)
            stackstring.push("lel")
            return stackstring

        script_input = torch.classes._TorchScriptTesting._StackString([])
        scripted = torch.jit.script(foo)
        script_output = scripted(script_input)
        self.assertEqual(script_output.pop(), "lel")

    def test_torchbind_return_instance(self):
        # [FUNCTION DOC] Checks a scripted function can return a new TorchBind object, that the
        # graph creates the object and calls its init method, and that the returned stack pops
        # items in reverse order.
        def foo():
            # [FUNCTION DOC] HELPER: Creates and returns a StackString holding two strings.
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])
            return ss

        scripted = torch.jit.script(foo)
        # Ensure we are creating the object and calling __init__
        # rather than calling the __init__wrapper nonsense
        fc = (
            FileCheck()
            .check("prim::CreateObject()")
            .check('prim::CallMethod[name="__init__"]')
        )
        fc.run(str(scripted.graph))
        out = scripted()
        self.assertEqual(out.pop(), "mom")
        self.assertEqual(out.pop(), "hi")

    def test_torchbind_return_instance_from_method(self):
        # [FUNCTION DOC] Checks that a method returning a new TorchBind instance (clone) works
        # in script, and the clone and original are independent: popping the original does not
        # affect the clone.
        def foo():
            # [FUNCTION DOC] HELPER: Creates a StackString, clones it, pops the original, and
            # returns both.
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])
            clone = ss.clone()
            ss.pop()
            return ss, clone

        scripted = torch.jit.script(foo)
        out = scripted()
        self.assertEqual(out[0].pop(), "hi")
        self.assertEqual(out[1].pop(), "mom")
        self.assertEqual(out[1].pop(), "hi")

    def test_torchbind_def_property_getter_setter(self):
        # [FUNCTION DOC] Checks read-write properties defined with getter and setter, including
        # a lambda-defined property, give the same results in eager and script mode via
        # checkScript.
        def foo_getter_setter_full():
            # [FUNCTION DOC] HELPER: Reads the x property, sets it to the old value plus 4,
            # reads it again, and returns old and new (the getter and setter each adjust x by
            # 2).
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # getX method intentionally adds 2 to x
            old = fooGetterSetter.x
            # setX method intentionally adds 2 to x
            fooGetterSetter.x = old + 4
            new = fooGetterSetter.x
            return old, new

        self.checkScript(foo_getter_setter_full, ())

        def foo_getter_setter_lambda():
            # [FUNCTION DOC] HELPER: Same as the other property function but for the lambda-
            # defined variant.
            foo = torch.classes._TorchScriptTesting._FooGetterSetterLambda(5)
            old = foo.x
            foo.x = old + 4
            new = foo.x
            return old, new

        self.checkScript(foo_getter_setter_lambda, ())

    def test_torchbind_def_property_just_getter(self):
        # [FUNCTION DOC] Checks a getter-only property: reading works in script (result 10),
        # assigning from Python raises a RuntimeError about not being able to set the attribute,
        # and scripting a function that assigns to it raises a RuntimeError saying it tried to
        # set a read-only attribute, highlighting the assignment.
        def foo_just_getter():
            # [FUNCTION DOC] HELPER: Returns a FooGetterSetter object and the value of its read-
            # only y property.
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # getY method intentionally adds 4 to x
            return fooGetterSetter, fooGetterSetter.y

        scripted = torch.jit.script(foo_just_getter)
        out, result = scripted()
        self.assertEqual(result, 10)

        with self.assertRaisesRegex(RuntimeError, "can't set attribute"):
            out.y = 5

        def foo_not_setter():
            # [FUNCTION DOC] HELPER: Tries to assign to the read-only y property; scripting it
            # must fail.
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            old = fooGetterSetter.y
            fooGetterSetter.y = old + 4
            # getY method intentionally adds 4 to x
            return fooGetterSetter.y

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Tried to set read-only attribute: y",
            "fooGetterSetter.y = old + 4",
        ):
            scripted = torch.jit.script(foo_not_setter)

    def test_torchbind_def_property_readwrite(self):
        # [FUNCTION DOC] Checks a read-write property object works in script and that assigning
        # to a read-only property of such an object fails script compilation with a read-only
        # attribute error highlighting the assignment.
        def foo_readwrite():
            # [FUNCTION DOC] HELPER: Updates x through the property and returns the x and y
            # values.
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            old = fooReadWrite.x
            fooReadWrite.x = old + 4
            return fooReadWrite.x, fooReadWrite.y

        self.checkScript(foo_readwrite, ())

        def foo_readwrite_error():
            # [FUNCTION DOC] HELPER: Assigns to the read-only y property of a FooReadWrite;
            # scripting it must fail.
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            fooReadWrite.y = 5
            return fooReadWrite

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set read-only attribute: y", "fooReadWrite.y = 5"
        ):
            scripted = torch.jit.script(foo_readwrite_error)

    def test_torchbind_take_instance_as_method_arg(self):
        # [FUNCTION DOC] Checks a TorchBind object can be passed as a method argument (merge):
        # the merged stack pops the other items first, then the original.
        def foo():
            # [FUNCTION DOC] HELPER: Merges one StackString into another and returns the first.
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = torch.jit.script(foo)
        out = scripted()
        self.assertEqual(out.pop(), "hi")
        self.assertEqual(out.pop(), "mom")

    def test_torchbind_return_tuple(self):
        # [FUNCTION DOC] Checks that a method returning a tuple yields a float and an int with
        # values 1337.0 and 123.
        def f():
            # [FUNCTION DOC] HELPER: Returns the tuple produced by the return_a_tuple method of
            # a StackString.
            val = torch.classes._TorchScriptTesting._StackString(["3", "5"])
            return val.return_a_tuple()

        scripted = torch.jit.script(f)
        tup = scripted()
        self.assertEqual(tup, (1337.0, 123))

    def test_torchbind_save_load(self):
        # [FUNCTION DOC] Checks that a scripted function creating and merging TorchBind objects
        # can be exported and re-imported using the export and import copy helper (no assertions
        # on the result beyond not raising).
        def foo():
            # [FUNCTION DOC] HELPER: Merges one StackString into another and returns the first.
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss

        scripted = torch.jit.script(foo)
        self.getExportImportCopy(scripted)

    def test_torchbind_lambda_method(self):
        # [FUNCTION DOC] Checks that a method defined through a lambda (top) can be called in
        # script and returns the last pushed item.
        def foo():
            # [FUNCTION DOC] HELPER: Returns the top item of a one-item StackString.
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            return ss.top()

        scripted = torch.jit.script(foo)
        self.assertEqual(scripted(), "mom")

    def test_torchbind_class_attr_recursive(self):
        # [FUNCTION DOC] Checks that a module holding a TorchBind Foo attribute can be scripted
        # when built from an instance whose info value feeds a new Foo, and that the scripted
        # forward returns 6.
        class FooBar(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping an object with an info method; to_ivalue
            # builds an equivalent module around a real TorchBind Foo.
            def __init__(self, foo_model):
                # [FUNCTION DOC] HELPER: Stores the wrapped model as foo_mod.
                super().__init__()
                self.foo_mod = foo_model

            def forward(self) -> int:
                # [FUNCTION DOC] HELPER: Returns the info value of the wrapped model.
                return self.foo_mod.info()

            def to_ivalue(self):
                # [FUNCTION DOC] HELPER: Creates a new Foo from the info value and 1 and returns
                # a new FooBar wrapping it.
                torchbind_model = torch.classes._TorchScriptTesting._Foo(
                    self.foo_mod.info(), 1
                )
                return FooBar(torchbind_model)

        inst = FooBar(torch.classes._TorchScriptTesting._Foo(2, 3))
        scripted = torch.jit.script(inst.to_ivalue())
        self.assertEqual(scripted(), 6)

    def test_torchbind_class_attribute(self):
        # [FUNCTION DOC] Scripts a module with a StackString attribute, exports and re-imports
        # it, and checks the copy returns the string deserialized from forward and pops
        # deserialized, was and i from the attribute. Raises AssertionError on mismatch.
        class FooBar1234(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a StackString attribute holding two strings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the StackString attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._StackString(["3", "4"])

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the top of the StackString attribute.
                return self.f.top()

        inst = FooBar1234()
        scripted = torch.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        if eic() != "deserialized":
            raise AssertionError(f"Expected 'deserialized', got {eic()!r}")
        for expected in ["deserialized", "was", "i"]:
            if eic.f.pop() != expected:
                raise AssertionError(f"Expected {expected!r}, got unexpected value")

    def test_torchbind_getstate(self):
        # [FUNCTION DOC] Scripts a module with a PickleTester attribute, exports and re-imports
        # it, and checks forward returns 7 and the pops give 7, 3, 3, 1 because the class state
        # getter returns a fixed value set. Raises AssertionError on mismatch.
        class FooBar4321(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a PickleTester attribute created from two ints.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PickleTester attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the top of the PickleTester attribute.
                return self.f.top()

        inst = FooBar4321()
        scripted = torch.jit.script(inst)
        eic = self.getExportImportCopy(scripted)
        # NB: we expect the values {7, 3, 3, 1} as __getstate__ is defined to
        # return {1, 3, 3, 7}. I tried to make this actually depend on the
        # values at instantiation in the test with some transformation, but
        # because it seems we serialize/deserialize multiple times, that
        # transformation isn't as you would it expect it to be.
        if eic() != 7:
            raise AssertionError(f"Expected 7, got {eic()!r}")
        for expected in [7, 3, 3, 1]:
            if eic.f.pop() != expected:
                raise AssertionError(f"Expected {expected!r}, got unexpected value")

    def test_torchbind_deepcopy(self):
        # [FUNCTION DOC] Checks deepcopy of a scripted module with a PickleTester attribute:
        # copy forward returns 7 and the attribute pops 7, 3, 3, 1. Raises AssertionError on
        # mismatch.
        class FooBar4321(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a PickleTester attribute created from two ints.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PickleTester attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the top of the PickleTester attribute.
                return self.f.top()

        inst = FooBar4321()
        scripted = torch.jit.script(inst)
        copied = copy.deepcopy(scripted)
        if copied.forward() != 7:
            raise AssertionError(f"Expected 7, got {copied.forward()!r}")
        for expected in [7, 3, 3, 1]:
            if copied.f.pop() != expected:
                raise AssertionError(f"Expected {expected!r}, got unexpected value")

    def test_torchbind_python_deepcopy(self):
        # [FUNCTION DOC] Checks deepcopy of an unscripted module with a PickleTester attribute
        # behaves like the scripted copy: forward returns 7 and the attribute pops 7, 3, 3, 1.
        # Raises AssertionError on mismatch.
        class FooBar4321(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a PickleTester attribute created from two ints.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PickleTester attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the top of the PickleTester attribute.
                return self.f.top()

        inst = FooBar4321()
        copied = copy.deepcopy(inst)
        if copied() != 7:
            raise AssertionError(f"Expected 7, got {copied()!r}")
        for expected in [7, 3, 3, 1]:
            if copied.f.pop() != expected:
                raise AssertionError(f"Expected {expected!r}, got unexpected value")

    def test_torchbind_tracing(self):
        # [FUNCTION DOC] Traces a module that passes its TorchBind attribute to a custom op and
        # checks the traced module returns a 4 by 4 zeros tensor.
        class TryTracing(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a PickleTester attribute whose forward calls the
            # take_an_instance custom op on it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PickleTester attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            def forward(self):
                # [FUNCTION DOC] HELPER: Passes the attribute to the take_an_instance op and
                # returns its result.
                return torch.ops._TorchScriptTesting.take_an_instance(self.f)

        traced = torch.jit.trace(TryTracing(), ())
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_pass_wrong_type(self):
        # [FUNCTION DOC] Checks that passing a plain tensor to an op expecting a TorchBind
        # instance raises a RuntimeError mentioning that it found type Tensor.
        with self.assertRaisesRegex(RuntimeError, "but instead found type 'Tensor'"):
            torch.ops._TorchScriptTesting.take_an_instance(torch.rand(3, 4))

    def test_torchbind_tracing_nested(self):
        # [FUNCTION DOC] Same as the tracing test but the TorchBind object is an attribute of a
        # nested submodule; checks the traced module returns a 4 by 4 zeros tensor.
        class TryTracingNest(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module with a PickleTester attribute and no forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the PickleTester attribute.
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class TryTracing123(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module containing the inner module; its forward passes
            # the nested object to the take_an_instance op.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the inner module as the nest attribute.
                super().__init__()
                self.nest = TryTracingNest()

            def forward(self):
                # [FUNCTION DOC] HELPER: Passes the nested PickleTester to the take_an_instance
                # op and returns the result.
                return torch.ops._TorchScriptTesting.take_an_instance(self.nest.f)

        traced = torch.jit.trace(TryTracing123(), ())
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_pickle_serialization(self):
        # [FUNCTION DOC] Saves a PickleTester object with torch save into a buffer, loads it
        # with weights_only disabled, and checks it pops 7, 3, 3, 1.
        nt = torch.classes._TorchScriptTesting._PickleTester([3, 4])
        b = io.BytesIO()
        torch.save(nt, b)
        b.seek(0)
        # weights_only=False as trying to load ScriptObject
        nt_loaded = torch.load(b, weights_only=False)
        for exp in [7, 3, 3, 1]:
            self.assertEqual(nt_loaded.pop(), exp)

    def test_torchbind_instantiate_missing_class(self):
        # [FUNCTION DOC] Checks that instantiating a class that does not exist in a namespace
        # raises a RuntimeError saying it tried to instantiate the class but it does not exist.
        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to instantiate class 'foo.IDontExist', but it does not exist!",
        ):
            torch.classes.foo.IDontExist(3, 4, 5)

    def test_torchbind_optional_explicit_attr(self):
        # [FUNCTION DOC] Checks that a module with an explicit optional TorchBind attribute
        # annotation can be scripted (success if no exception).
        class TorchBindOptionalExplicitAttr(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an annotated optional StackString attribute set to
            # a one-item stack.
            foo: Optional[torch.classes._TorchScriptTesting._StackString]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the StackString attribute.
                super().__init__()
                self.foo = torch.classes._TorchScriptTesting._StackString(["test"])

            def forward(self) -> str:
                # [FUNCTION DOC] HELPER: Returns the popped item of the attribute if it is set,
                # otherwise the text none marker.
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return "<None>"

        mod = TorchBindOptionalExplicitAttr()
        scripted = torch.jit.script(mod)

    def test_torchbind_no_init(self):
        # [FUNCTION DOC] Checks that creating a class registered without a constructor raises a
        # RuntimeError mentioning the missing init registration.
        with self.assertRaisesRegex(RuntimeError, "torch::init"):
            x = torch.classes._TorchScriptTesting._NoInit()

    def test_profiler_custom_op(self):
        # [FUNCTION DOC] Runs a custom op on a TorchBind instance under the autograd profiler
        # and checks the recorded events include one named for that op.
        inst = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        with torch.autograd.profiler.profile() as prof:
            torch.ops._TorchScriptTesting.take_an_instance(inst)

        found_event = False
        for e in prof.function_events:
            if e.name == "_TorchScriptTesting::take_an_instance":
                found_event = True
        self.assertTrue(found_event)

    def test_torchbind_getattr(self):
        # [FUNCTION DOC] Checks that reading a missing attribute with a default on a TorchBind
        # object returns the default (None) instead of raising.
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        self.assertEqual(None, getattr(foo, "bar", None))

    def test_torchbind_attr_exception(self):
        # [FUNCTION DOC] Checks that reading a missing attribute directly on a TorchBind object
        # raises an AttributeError saying it does not have a field.
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        with self.assertRaisesRegex(AttributeError, "does not have a field"):
            foo.bar

    def test_lambda_as_constructor(self):
        # [FUNCTION DOC] Checks a class constructed through a lambda: with the swap flag false
        # the diff is 1, with it true the diff is minus 1 (arguments 4 and 3).
        obj_no_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, False)
        self.assertEqual(obj_no_swap.diff(), 1)

        obj_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, True)
        self.assertEqual(obj_swap.diff(), -1)

    def test_staticmethod(self):
        # [FUNCTION DOC] Checks that a static method of a TorchBind class can be called from a
        # scripted function.
        def fn(inp: int) -> int:
            # [FUNCTION DOC] HELPER: Calls the static method with the integer argument and
            # returns the result.
            return torch.classes._TorchScriptTesting._StaticMethod.staticMethod(inp)

        self.checkScript(fn, (1,))

    def test_staticmethod_default_args(self):
        # [FUNCTION DOC] Checks static method calls using a default argument, both omitted and
        # given explicitly, in script and eager mode.
        def fn(inp: int) -> int:
            # [FUNCTION DOC] HELPER: Calls the static method with a default twice, first
            # omitting and then passing 4, feeding the first result into the second.
            res = (
                torch.classes._TorchScriptTesting._StaticMethod.staticMethodWithDefault(
                    inp
                )
            )
            return (
                torch.classes._TorchScriptTesting._StaticMethod.staticMethodWithDefault(
                    res, 4
                )
            )

        self.checkScript(fn, (1,))

    def test_hasattr(self):
        # [FUNCTION DOC] Checks that the attribute test on a TorchBind object returns false for
        # a name the class does not define.
        ss = torch.classes._TorchScriptTesting._StackString(["foo", "bar"])
        self.assertFalse(hasattr(ss, "baz"))

    def test_default_args(self):
        # [FUNCTION DOC] Checks default arguments of methods and constructors of the DefaultArgs
        # class in two scripted functions compared with eager execution.
        def fn() -> int:
            # [FUNCTION DOC] HELPER: Builds the object with the default constructor and calls
            # increment, decrement, divide and scale_add with and without optional arguments,
            # returning the final increment.
            obj = torch.classes._TorchScriptTesting._DefaultArgs()
            obj.increment(5)
            obj.decrement()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(5)
            obj.scale_add(3, 2)
            obj.divide(3)
            return obj.increment()

        self.checkScript(fn, ())

        def gn() -> int:
            # [FUNCTION DOC] HELPER: Same but constructs with an explicit argument and returns
            # the final decrement.
            obj = torch.classes._TorchScriptTesting._DefaultArgs(5)
            obj.increment(3)
            obj.increment()
            obj.decrement(2)
            obj.divide()
            obj.scale_add(3)
            obj.scale_add(3, 2)
            obj.divide(2)
            return obj.decrement()

        self.checkScript(gn, ())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
