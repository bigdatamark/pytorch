# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

# [MODULE DOC] Tests for user-defined classes compiled by TorchScript as class types:
# reference semantics, attribute errors, save/load, Python interop, sorting, inheritance
# errors, interfaces, operator overloads, cast overloads, default arguments, static and class
# methods, properties, custom delete, and recursive scripting of classes found on modules.
# Single test class TestClassType; most tests define classes and functions locally inside the
# test body. Import-time side effects: extends sys.path with the test directory. Running the
# file directly is rejected.
import io
import os
import sys
import unittest
from typing import Any

import torch
import torch.nn as nn
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from typing import Dict, Iterable, List, Optional, Tuple

import torch.testing._internal.jit_utils
from torch.testing._internal.common_utils import (
    IS_SANDCASTLE,
    raise_on_run_directly,
    skipIfTorchDynamo,
)
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestClassType(JitTestCase):
    # [CLASS DOC] JitTestCase covering TorchScript class types. Each test builds its own local
    # classes, scripts them (or lets recursive scripting find them) and either checks results
    # against eager execution or expects a specific RuntimeError message.
    def test_reference_semantics(self):
        # [FUNCTION DOC] Script a function that mutates a class instance through a method,
        # pass a scripted instance in, and check the change is visible afterwards through both
        # a method and a property, showing class instances have reference semantics across
        # eager and script.
        """
        Test that modifications made to a class instance in TorchScript
        are visible in eager.
        """

        class Foo:
            # [CLASS DOC] HELPER: Class holding an integer, with a setter, a getter and a
            # property that reads the attribute.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the argument as the attribute.
                self.a = a

            def set_a(self, value: int):
                # [FUNCTION DOC] HELPER: Overwrite the attribute with the given value.
                self.a = value

            def get_a(self) -> int:
                # [FUNCTION DOC] HELPER: Return the stored attribute.
                return self.a

            @property
            def attr(self):
                # [FUNCTION DOC] HELPER: Property returning the stored attribute.
                return self.a

        make_global(Foo)  # see [local resolution in python]

        def test_fn(obj: Foo):
            # [FUNCTION DOC] HELPER: Call the setter on the object with the value two.
            obj.set_a(2)

        scripted_fn = torch.jit.script(test_fn)
        obj = torch.jit.script(Foo(1))
        self.assertEqual(obj.get_a(), 1)
        self.assertEqual(obj.attr, 1)

        scripted_fn(obj)

        self.assertEqual(obj.get_a(), 2)
        self.assertEqual(obj.attr, 2)

    def test_get_with_method(self):
        # [FUNCTION DOC] Define a class with a getter method and a function using it, and
        # call the function in plain Python only, so only eager behavior is checked.
        # KNOWN ISSUE: the function is never passed to the script compiler, so no TorchScript
        # behavior is tested.
        class FooTest:
            # [CLASS DOC] HELPER: Class storing a value with a getter method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.foo = x

            def getFooTest(self):
                # [FUNCTION DOC] HELPER: Return the stored value.
                return self.foo

        def fn(x):
            # [FUNCTION DOC] HELPER: Construct the class from the argument and return
            # its getter result.
            foo = FooTest(x)
            return foo.getFooTest()

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_get_attr(self):
        # [FUNCTION DOC] Script a function that constructs a class and reads its attribute
        # directly, and check it returns the input tensor.
        class FooTest:
            # [CLASS DOC] HELPER: Plain class storing its argument in an attribute.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.foo = x

        @torch.jit.script
        def fn(x):
            # [FUNCTION DOC] HELPER: Construct the class and return its attribute.
            foo = FooTest(x)
            return foo.foo

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)

    def test_in(self):
        # [FUNCTION DOC] Script a function that uses the in operator on an instance of a
        # class defining the contains hook, and check it returns true for the matching key
        # and false for another.
        class FooTest:
            # [CLASS DOC] HELPER: Class whose contains hook is true only for one string.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def __contains__(self, key: str) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the key equals the string hi.
                return key == "hi"

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Return membership results for the strings hi and no.
            foo = FooTest()
            return "hi" in foo, "no" in foo

        self.assertEqual(fn(), (True, False))

    def test_set_attr_in_method(self):
        # [FUNCTION DOC] Script a class whose method updates an attribute using its argument,
        # and check a scripted function that calls it yields the expected integer.
        class FooTest:
            # [CLASS DOC] HELPER: Class with an integer attribute and an incrementing method.
            def __init__(self, x: int) -> None:
                # [FUNCTION DOC] HELPER: Store the integer.
                self.foo = x

            def incFooTest(self, y: int) -> None:
                # [FUNCTION DOC] HELPER: Add the argument to the stored integer in place.
                self.foo = self.foo + y

        @torch.jit.script
        def fn(x: int) -> int:
            # [FUNCTION DOC] HELPER: Construct, increment by two, and return the attribute.
            foo = FooTest(x)
            foo.incFooTest(2)
            return foo.foo

        self.assertEqual(fn(1), 3)

    def test_set_attr_type_mismatch(self):
        # [FUNCTION DOC] Scripting a class whose constructor re-assigns an attribute with a
        # value of a different type must raise RuntimeError with the wrong-type attribute
        # assignment message, highlighting the offending assignment.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.foo = 10"
        ):

            @torch.jit.script
            class FooTest:
                # [CLASS DOC] HELPER: Class assigning a tensor-typed argument to an attribute,
                # then an integer to the same attribute, which scripting must reject.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Assign the argument and then an integer to
                    # the same attribute.
                    self.foo = x
                    self.foo = 10  # should error since int != Tensor

    def test_get_attr_not_initialized(self):
        # [FUNCTION DOC] Scripting a class whose method reads an attribute never set must
        # raise RuntimeError saying the object has no such attribute or method,
        # highlighting the access.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "self.asdf"
        ):

            @torch.jit.script
            class FooTest:
                # [CLASS DOC] HELPER: Class with one initialized attribute and a method
                # reading a different nonexistent attribute.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Store the argument as an attribute.
                    self.foo = x

                def get_non_initialized(self):
                    # [FUNCTION DOC] HELPER: Read an attribute that was never defined
                    # (compile error expected).
                    return self.asdf  # asdf isn't an attr

    def test_set_attr_non_initialized(self):
        # [FUNCTION DOC] Scripting a class whose method assigns a new attribute not created in
        # the constructor must raise RuntimeError about setting a nonexistent attribute.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.bar = y"
        ):

            @torch.jit.script
            class FooTest:
                # [CLASS DOC] HELPER: Class whose method tries to create a new attribute
                # outside the constructor.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Store the argument as an attribute.
                    self.foo = x

                def set_non_initialized(self, y):
                    # [FUNCTION DOC] HELPER: Assign a nonexistent attribute; a compile
                    # error is expected.
                    self.bar = y  # can't assign to non-initialized attr

    def test_schema_human_readable(self):
        # [FUNCTION DOC] Scripting a function that calls interpolate with a wrongly typed
        # argument must raise RuntimeError whose message shows the default mode readably as
        # the word nearest rather than as an escaped octal string.
        """
        Make sure that the schema is human readable, ie the mode parameter should read "nearest" instead of being displayed in octal
        aten::__interpolate(Tensor input, int? size=None, float[]? scale_factor=None,
        str mode='\156\145\141\162\145\163\164', bool? align_corners=None) -> (Tensor):
        Expected a value of type 'Optional[int]' for argument 'size' but instead found type 'Tensor'.
        """
        with self.assertRaisesRegexWithHighlight(RuntimeError, "nearest", ""):

            @torch.jit.script
            def FooTest(x):
                # [FUNCTION DOC] HELPER: Call the functional interpolate with a string as the
                # size argument (compile error expected).
                return torch.nn.functional.interpolate(x, "bad")

    def test_type_annotations(self):
        # [FUNCTION DOC] Scripting a function that passes an untyped argument (a tensor) to a
        # class constructor expecting a bool must raise RuntimeError saying a bool was
        # expected. The final call to the function is therefore never reached.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Expected a value of type 'bool", ""
        ):

            @torch.jit.script
            class FooTest:
                # [CLASS DOC] HELPER: Class storing a bool-annotated constructor argument.
                def __init__(self, x: bool) -> None:
                    # [FUNCTION DOC] HELPER: Store the bool argument.
                    self.foo = x

            @torch.jit.script
            def fn(x):
                # [FUNCTION DOC] HELPER: Construct the class from the untyped argument.
                FooTest(x)

            fn(2)

    def test_conditional_set_attr(self):
        # [FUNCTION DOC] Scripting a class that assigns an attribute inside an if block in its
        # constructor must raise RuntimeError saying assignment cannot be in a control-flow
        # block; the error comes from compilation, not from running anything.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "assignment cannot be in a control-flow block", ""
        ):

            @torch.jit.script
            class FooTest:
                # [CLASS DOC] HELPER: Class that creates its attribute only inside
                # a conditional branch.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Create the attribute inside an always-true if
                    # block (compile error expected).
                    if 1 == 1:
                        self.attr = x

    def test_class_type_as_param(self):
        # [FUNCTION DOC] Use a class instance as an argument type of another scripted
        # function, register the class globally, and check the composed functions return
        # the input tensor.
        class FooTest:
            # [CLASS DOC] HELPER: Class storing one attribute.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.attr = x

        make_global(FooTest)  # see [local resolution in python]

        @torch.jit.script
        def fn(foo: FooTest) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Return the attribute of the class instance argument.
            return foo.attr

        @torch.jit.script
        def fn2(x):
            # [FUNCTION DOC] HELPER: Construct the class from the argument and pass it to
            # the first function.
            foo = FooTest(x)
            return fn(foo)

        input = torch.ones(1)
        self.assertEqual(fn2(input), input)

    def test_out_of_order_methods(self):
        # [FUNCTION DOC] Check that a constructor may call a method defined later in the class
        # body by scripting a function that uses it, expecting the input added to itself.
        class FooTest:
            # [CLASS DOC] HELPER: Class whose constructor calls a method declared after it.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument, then replace the attribute with
                # the result of the later method.
                self.x = x
                self.x = self.get_stuff(x)

            def get_stuff(self, y):
                # [FUNCTION DOC] HELPER: Return the stored attribute plus the argument.
                return self.x + y

        @torch.jit.script
        def fn(x):
            # [FUNCTION DOC] HELPER: Construct the class and return its attribute.
            f = FooTest(x)
            return f.x

        input = torch.ones(1)
        self.assertEqual(fn(input), input + input)

    def test_save_load_with_classes(self):
        # [FUNCTION DOC] Save a script module whose forward uses a local class, load it again
        # from an in-memory buffer, and check the output equals the input.
        # KNOWN ISSUE: the surrounding comment says the class registry must be cleared before
        # loading, but the registry clearing helper is not called in this test.
        class FooTest:
            # [CLASS DOC] HELPER: Class storing one value and returning it through a getter.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.x = x

            def get_x(self):
                # [FUNCTION DOC] HELPER: Return the stored value.
                return self.x

        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose forward wraps the input in the class and
            # returns it through the getter.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Construct the class from the input and return
                # its getter result.
                foo = FooTest(a)
                return foo.get_x()

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_returned(self):
        # [FUNCTION DOC] Save a script module whose forward calls a class method that returns
        # a new instance of the class, clear the class registry, load the module from the
        # buffer, and check the output equals the input.
        class FooTest:
            # [CLASS DOC] HELPER: Class storing a value, with a method returning a copy.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.x = x

            def clone(self):
                # [FUNCTION DOC] HELPER: Construct a new instance with the same value
                # and return it.
                clone = FooTest(self.x)
                return clone

        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose forward builds the class, clones it and
            # returns the clone attribute.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Construct, clone and return the cloned value.
                foo = FooTest(a)
                foo_clone = foo.clone()
                return foo_clone.x

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_nested(self):
        # [FUNCTION DOC] Save and load a script module that uses classes containing instances
        # of other classes, clearing the registry before loading, and check the result is
        # twice the input since the outer class sums two nested values.
        class FooNestedTest:
            # [CLASS DOC] HELPER: Innermost class storing one value.
            def __init__(self, y):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.y = y

        class FooNestedTest2:
            # [CLASS DOC] HELPER: Class storing a value and holding an instance of
            # the innermost class.
            def __init__(self, y):
                # [FUNCTION DOC] HELPER: Store the argument and build a nested instance from
                # the same value.
                self.y = y
                self.nested = FooNestedTest(y)

        class FooTest:
            # [CLASS DOC] HELPER: Outer class holding instances of both other classes and the
            # sum of their stored values.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Build both nested instances from the argument and
                # store the sum of their values.
                self.class_attr = FooNestedTest(x)
                self.class_attr2 = FooNestedTest2(x)
                self.x = self.class_attr.y + self.class_attr2.y

        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose forward builds the outer class and
            # returns its sum.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Construct the outer class from the input and
                # return the sum.
                foo = FooTest(a)
                return foo.x

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(2 * input, output)

    def test_python_interop(self):
        # [FUNCTION DOC] Check a Python-created class instance can be passed into and out of a
        # scripted function annotated with the class, with its attributes preserved.
        class Foo:
            # [CLASS DOC] HELPER: Class storing two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Store both arguments.
                self.x = x
                self.y = y

        make_global(Foo)  # see [local resolution in python]

        @torch.jit.script
        def use_foo(foo: Foo) -> Foo:
            # [FUNCTION DOC] HELPER: Return the argument unchanged.
            return foo

        # create from python
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)

        self.assertEqual(x, f.x)
        self.assertEqual(y, f.y)

        # pass in and out of script
        f2 = use_foo(f)

        self.assertEqual(x, f2.x)
        self.assertEqual(y, f2.y)

    def test_class_specialization(self):
        # [FUNCTION DOC] Script a function over several class instances including a tuple of
        # them and check the graph contains exactly four attribute reads.
        class Foo:
            # [CLASS DOC] HELPER: Class storing two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Store both arguments.
                self.x = x
                self.y = y

        make_global(Foo)  # see [local resolution in python]

        def use_foo(foo: Foo, foo2: Foo, tup: Tuple[Foo, Foo]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Unpack the tuple and return the sum of four attributes
            # drawn from the instances.
            a, b = tup
            return foo.x + foo2.y + a.x + b.y

        # create from python
        x = torch.ones(2, 3)
        y = torch.zeros(2, 3)
        f = Foo(x, y)
        f2 = Foo(x * 2, y * 3)
        f3 = Foo(x * 4, y * 4)

        input = (f, f2, (f, f3))
        sfoo = self.checkScript(use_foo, input)
        graphstr = str(sfoo.graph_for(*input))
        FileCheck().check_count("prim::GetAttr", 4).run(graphstr)

    def test_class_sorting(self):
        # [FUNCTION DOC] Check sorting lists of class instances that define the less-than
        # hook: sorted and in-place sort with and without reverse, empty and single lists,
        # instances inside tuples, and the copy semantics of sorted. Also check error
        # messages for a wrong sort argument, a class missing the hook and a class with a
        # wrongly shaped hook.
        class Foo:
            # [CLASS DOC] HELPER: Class with an integer attribute ordered by that integer.
            def __init__(self, x: int) -> None:
                # [FUNCTION DOC] HELPER: Store the integer.
                self.x = x

            def __lt__(self, other) -> bool:
                # [FUNCTION DOC] HELPER: Return whether this value is less than the other one.
                # type: (Foo) -> bool
                return self.x < other.x

            def getVal(self):
                # [FUNCTION DOC] HELPER: Return the stored integer.
                return self.x

        make_global(Foo)  # see [local resolution in python]

        def test(li: List[Foo], reverse: bool = False) -> Tuple[List[int], List[int]]:
            # [FUNCTION DOC] HELPER: Return the stored values of the list sorted by the sorted
            # builtin and then by in-place sort honoring the reverse flag, as a pair of lists.
            li_sorted = sorted(li)
            ret_sorted = torch.jit.annotate(List[int], [])
            for foo in li_sorted:
                ret_sorted.append(foo.getVal())

            li.sort(reverse=reverse)
            ret_sort = torch.jit.annotate(List[int], [])
            for foo in li:
                ret_sort.append(foo.getVal())
            return ret_sorted, ret_sort

        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)],))
        self.checkScript(test, ([Foo(2), Foo(1), Foo(3)], True))
        self.checkScript(test, ([Foo(2)],))
        self.checkScript(test, ([],))

        @torch.jit.script
        def test_list_no_reverse():
            # [FUNCTION DOC] HELPER: Sort a short list in place, return its first value.
            li = [Foo(3), Foo(1)]
            li.sort()
            return li[0].getVal()

        self.assertEqual(test_list_no_reverse(), 1)

        @torch.jit.script
        def test_sorted_copies():
            # [FUNCTION DOC] HELPER: Return the first value of the original list and of its
            # sorted copy, showing sorted does not modify the source.
            li = [Foo(3), Foo(1)]
            li_sorted = sorted(li)
            return li[0].getVal(), li_sorted[0].getVal()

        self.assertEqual(test_sorted_copies(), (3, 1))

        @torch.jit.script
        def test_nested_inside_tuple():
            # [FUNCTION DOC] HELPER: Sort a list of tuples whose second items are class
            # instances and return the sorted values as plain tuples.
            li = [(1, Foo(12)), (1, Foo(11))]
            li.sort()
            return [(li[0][0], li[0][1].getVal()), (li[1][0], li[1][1].getVal())]

        self.assertEqual(test_nested_inside_tuple(), [(1, 11), (1, 12)])

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "bool' for argument 'reverse", ""
        ):

            @torch.jit.script
            def test():
                # [FUNCTION DOC] HELPER: Pass a list as the reverse argument of sort
                # (compile error expected).
                li = [Foo(1)]
                li.sort(li)
                return li

            test()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "must define a __lt__", ""
        ):

            @torch.jit.script
            class NoMethod:
                # [CLASS DOC] HELPER: Class without any ordering hook.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: No-op constructor.
                    pass

            @torch.jit.script
            def test():
                # [FUNCTION DOC] HELPER: Sort a list of the unorderable class; a compile
                # error is expected.
                li = [NoMethod(), NoMethod()]
                li.sort()
                return li

            test()

        @torch.jit.script
        class WrongLt:
            # [CLASS DOC] HELPER: Scripted class whose ordering hook has no type annotations.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            # lt method defined with the wrong signature
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Ordering hook with the wrong signature; does nothing.
                pass

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "must define a __lt__", ""
        ):

            @torch.jit.script
            def test():
                # [FUNCTION DOC] HELPER: Sort a list of the class with the bad ordering hook
                # (compile error expected).
                li = [WrongLt(), WrongLt()]
                li.sort()
                return li

            test()

    def test_class_inheritance(self):
        # [FUNCTION DOC] Scripting a class that derives from another scripted class must raise
        # RuntimeError saying inheritance is not supported.
        @torch.jit.script
        class Base:
            # [CLASS DOC] HELPER: Scripted base class with an attribute and a method using it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Set the attribute to two.
                self.b = 2

            def two(self, x):
                # [FUNCTION DOC] HELPER: Return the argument plus the attribute.
                return x + self.b

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "does not support inheritance", ""
        ):

            @torch.jit.script
            class Derived(Base):
                # [CLASS DOC] HELPER: Subclass of the scripted base class, which
                # scripting must reject.
                def two(self, x):
                    # [FUNCTION DOC] HELPER: Override returning the argument plus the
                    # attribute plus two.
                    return x + self.b + 2

    def test_class_inheritance_implicit(self):
        # [FUNCTION DOC] Check that implicit scripting through type annotation resolution also
        # detects class inheritance: scripting a function that handles either class raises
        # RuntimeError about a missing attribute or method.
        """
        Test that inheritance is detected in
        implicit scripting codepaths (e.g. try_ann_to_type).
        """

        class A:
            # [CLASS DOC] HELPER: Class storing one value, with a static factory.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.t = t

            @staticmethod
            def f(a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Static method returning a new instance built from the
                # argument plus one.
                return A(a + 1)

        class B(A):
            # [CLASS DOC] HELPER: Subclass of the first class that adds ten to the value.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Store the argument plus ten.
                self.t = t + 10

            @staticmethod
            def f(a: torch.Tensor):
                # [FUNCTION DOC] HELPER: Static method returning an instance of the base class
                # built from the argument plus one.
                return A(a + 1)

        x = A(torch.tensor([3]))

        def fun(x: Any):
            # [FUNCTION DOC] HELPER: Dispatch on the instance type of the argument and call
            # the matching static method with the stored value.
            if isinstance(x, A):
                return A.f(x.t)
            else:
                return B.f(x.t)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", ""
        ):
            sc = torch.jit.script(fun)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    @unittest.skipIf(IS_SANDCASTLE, "Importing like this doesn't work in fbcode")
    def test_imported_classes(self):
        # [FUNCTION DOC] Skipped under TorchDynamo and on Sandcastle. Import three test
        # modules that define same-named classes in different packages, script a module using
        # them, save and load it after clearing the class registry, and check the sum is three
        # times the input.
        import jit._imported_class_test.bar
        import jit._imported_class_test.foo
        import jit._imported_class_test.very.very.nested

        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose forward builds one class from each
            # imported module and sums their attributes.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Construct the three imported classes from the input
                # and return the sum of their attributes.
                foo = jit._imported_class_test.foo.FooSameName(a)
                bar = jit._imported_class_test.bar.FooSameName(a)
                three = jit._imported_class_test.very.very.nested.FooUniqueName(a)
                return foo.x + bar.y + three.y

        m = MyMod()

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # classes are globally registered for now, so we need to clear the JIT
        # registry to simulate loading a new model
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(3 * input, output)

    def test_interface(self):
        # [FUNCTION DOC] Test TorchScript interfaces: define classes and interfaces, check
        # classes can be used as interface typed values in lists, check inheritance-like
        # conversion between interfaces, and expect specific errors for missing or
        # incompatible methods. Also check assigning an interface to a module attribute
        # through the annotations mapping survives export and import, and that incompatible or
        # pure Python objects are rejected.
        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted class with two methods adding and doubling.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def one(self, x, y):
                # [FUNCTION DOC] HELPER: Return the sum of the arguments.
                return x + y

            def two(self, x):
                # [FUNCTION DOC] HELPER: Return twice the argument.
                return 2 * x

        @torch.jit.script
        class Bar:
            # [CLASS DOC] HELPER: Scripted class with two methods multiplying and dividing.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def one(self, x, y):
                # [FUNCTION DOC] HELPER: Return the product of the arguments.
                return x * y

            def two(self, x):
                # [FUNCTION DOC] HELPER: Return two divided by the argument.
                return 2 / x

        @torch.jit.interface
        class OneTwo:
            # [CLASS DOC] HELPER: Interface declaring two tensor methods.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

        @torch.jit.interface
        class OneTwoThree:
            # [CLASS DOC] HELPER: Interface declaring three tensor methods, a superset of
            # the first interface.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

            def three(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

        @torch.jit.interface
        class OneTwoWrong:
            # [CLASS DOC] HELPER: Interface whose second method has integer types,
            # incompatible with the classes above.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; body is a placeholder.
                pass

            def two(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Interface method declaration with integer
                # types; placeholder body.
                pass

        @torch.jit.script
        class NotMember:
            # [CLASS DOC] HELPER: Scripted class missing the second interface method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def one(self, x, y):
                # [FUNCTION DOC] HELPER: Return the sum of the arguments.
                return x + y

            # missing two

        @torch.jit.script
        class NotMember2:
            # [CLASS DOC] HELPER: Scripted class whose second method has integer types,
            # mismatching the interface.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def one(self, x, y):
                # [FUNCTION DOC] HELPER: Return the sum of the arguments.
                return x + y

            def two(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Return the constant three for an integer argument.
                return 3

        make_global(Foo, Bar, OneTwo, OneTwoThree, OneTwoWrong, NotMember, NotMember2)

        def use_them(x):
            # [FUNCTION DOC] HELPER: Put one instance of each class in a list of the interface
            # type, apply the first then second method of every element to the value in a loop
            # and return it.
            a = Foo()
            b = Bar()
            c = torch.jit.annotate(List[OneTwo], [a, b])
            for i in range(len(c)):
                x = c[i].one(x, x)
                x = c[i].two(x)
            return x

        self.checkScript(use_them, (torch.rand(3, 4),))

        @torch.jit.script
        def as_interface(x: OneTwo) -> OneTwo:
            # [FUNCTION DOC] HELPER: Return the argument, typed as the interface.
            return x

        @torch.jit.script
        def inherit(x: OneTwoThree) -> OneTwo:
            # [FUNCTION DOC] HELPER: Narrow a larger-interface value to the smaller interface.
            return as_interface(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "does not have method", ""
        ):

            @torch.jit.script
            def wrong1():
                # [FUNCTION DOC] HELPER: Convert a class lacking a method to the interface
                # (compile error expected).
                return as_interface(NotMember())

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not compatible with interface", ""
        ):

            @torch.jit.script
            def wrong2():
                # [FUNCTION DOC] HELPER: Convert a class with an incompatible method to the
                # interface (compile error expected).
                return as_interface(NotMember2())

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "does not have method", ""
        ):

            @torch.jit.script
            def wrong3():
                # [FUNCTION DOC] HELPER: Convert a smaller interface value to the larger
                # interface (compile error expected).
                return inherit(as_interface(Foo()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not compatible with interface", ""
        ):

            @torch.jit.script
            def wrong4(x: OneTwoWrong) -> int:
                # [FUNCTION DOC] HELPER: Convert a value of the incompatible interface
                # (compile error expected).
                return as_interface(x)

        # Test interface/class python assignment
        class TestPyAssign(nn.Module):
            # [CLASS DOC] HELPER: Module holding a scripted class instance in a proxy
            # attribute that is later annotated with the interface type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module and store the instance.
                super().__init__()
                self.proxy_mod = Foo()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call the second method of the proxy.
                return self.proxy_mod.two(x)

        TestPyAssign.__annotations__ = {"proxy_mod": OneTwo}

        input = torch.rand(3, 4)
        scripted_pyassign_mod = torch.jit.script(TestPyAssign())
        imported_mod = self.getExportImportCopy(scripted_pyassign_mod)
        self.assertEqual(scripted_pyassign_mod(input), imported_mod(input))

        class TestPyAssignError(nn.Module):
            # [CLASS DOC] HELPER: Module storing an object passed in by the caller as a proxy
            # attribute and annotated with the larger interface.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Initialize the module and store the given object.
                super().__init__()
                self.proxy_mod = obj

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Call the second method of the proxy.
                return self.proxy_mod.two(x)

        TestPyAssignError.__annotations__ = {"proxy_mod": OneTwoThree}

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not compatible with interface __torch__", ""
        ):
            torch.jit.script(TestPyAssignError(Foo()))

        # test pure python object assignment to interface fails
        class PyClass:
            # [CLASS DOC] HELPER: Plain Python class, not a TorchScript compatible type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "the value is not a TorchScript compatible type", ""
        ):
            torch.jit.script(TestPyAssignError(PyClass()))
        # TODO test: interface-interface class-interface inheritance errors,
        # NamedTuple inheritance errors

    def test_overloaded_fn(self):
        # [FUNCTION DOC] Check operator overload hooks on scripted classes: length,
        # negation and multiplication on one class, then arithmetic, comparison,
        # bitwise, item access and call hooks on another, each compared eager versus
        # script through small helper functions. Also require a missing overload to give
        # an attribute-or-method RuntimeError.
        # KNOWN ISSUE: the modulo hook is defined on the class but no helper function
        # exercises it.
        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted class wrapping a tensor with length, negation and
            # tensor multiplication hooks.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the argument.
                self.x = x

            def __len__(self) -> int:
                # [FUNCTION DOC] HELPER: Return the length of the stored value.
                return len(self.x)

            def __neg__(self):
                # [FUNCTION DOC] HELPER: Negate the stored value in place, return self.
                self.x = -self.x
                return self

            def __mul__(self, other: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Return the stored tensor times the other tensor.
                return self.x * other

        def test_overload():
            # [FUNCTION DOC] HELPER: Return the length of an instance and the product of its
            # negation with a zero tensor.
            a = Foo(torch.ones([3, 3]))
            return len(a), -a * torch.zeros([3, 3])

        make_global(Foo)  # see [local resolution in python]

        self.checkScript(test_overload, ())
        # unary ops tested above

        # TODO - support compiling classes from strings in jit.CompilationUnit
        @torch.jit.script
        class MyClass:
            # [CLASS DOC] HELPER: Scripted class holding an integer with operator hooks for
            # add, subtract, multiply, power, true divide, modulo, all six comparisons, and,
            # or, xor, item get and set, and call.
            def __init__(self, x: int) -> None:
                # [FUNCTION DOC] HELPER: Store the integer.
                self.x = x

            def __add__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Return the stored integer plus the argument.
                return self.x + other

            def __sub__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Return the stored integer minus the argument.
                return self.x - other

            def __mul__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Return the stored integer times the argument.
                return self.x * other

            def __pow__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Return the stored integer raised to the argument,
                # as an integer.
                return int(self.x**other)

            def __truediv__(self, other: int) -> float:
                # [FUNCTION DOC] HELPER: Return the stored integer divided by the argument,
                # as a float.
                return self.x / other

            def __mod__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Return the stored integer modulo the argument.
                return self.x % other

            def __ne__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Compare the stored integer with the
                # argument for inequality.
                return self.x != other

            def __eq__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Compare the stored integer with the
                # argument for equality.
                return self.x == other

            def __lt__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the stored integer is less
                # than the argument.
                return self.x < other

            def __gt__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the stored integer is greater
                # than the argument.
                return self.x > other

            def __le__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the stored integer is at most
                # the argument value.
                return self.x <= other

            def __ge__(self, other: int) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the stored integer is at least
                # the argument value.
                return self.x >= other

            def __and__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Bitwise and of the stored integer with the argument.
                return self.x & other

            def __or__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Bitwise or of the stored integer with the argument.
                return self.x | other

            def __xor__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Bitwise xor of the stored integer with the argument.
                return self.x ^ other

            def __getitem__(self, other: int) -> int:
                # [FUNCTION DOC] HELPER: Item read hook; ignores the instance and returns the
                # index plus one.
                return other + 1

            def __setitem__(self, idx: int, val: int) -> None:
                # [FUNCTION DOC] HELPER: Item write hook; sets the stored integer to the
                # value times index.
                self.x = val * idx

            def __call__(self, val: int) -> int:
                # [FUNCTION DOC] HELPER: Call hook; returns the stored integer times the
                # argument times three.
                return self.x * val * 3

        make_global(Foo)  # see [local resolution in python]

        def add():
            # [FUNCTION DOC] HELPER: Return the sum of an instance holding four and
            # the integer three.
            return MyClass(4) + 3

        def sub():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return an instance holding four minus the integer three.
            return MyClass(4) - 3

        def mul():  # noqa: E306
            # [FUNCTION DOC] HELPER: Multiply an instance holding four by three.
            return MyClass(4) * 3

        def pow():  # noqa: E306
            # [FUNCTION DOC] HELPER: Raise an instance holding four to the power three.
            return MyClass(4) ** 3

        def truediv():  # noqa: E306
            # [FUNCTION DOC] HELPER: Divide an instance holding four by three.
            return MyClass(4) / 3

        def ne():  # noqa: E306
            # [FUNCTION DOC] HELPER: Compare an instance holding four with
            # three for inequality.
            return MyClass(4) != 3

        def eq():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return whether an instance holding four equals three.
            return MyClass(4) == 3

        def lt():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return whether an instance holding four is below three.
            return MyClass(4) < 3

        def gt():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return whether an instance holding four is above three.
            return MyClass(4) > 3

        def le():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return whether an instance holding four is at most three.
            return MyClass(4) <= 3

        def ge():  # noqa: E306
            # [FUNCTION DOC] HELPER: Return whether an instance holding four is
            # at least three.
            return MyClass(4) >= 3

        def _and():  # noqa: E306
            # [FUNCTION DOC] HELPER: Bitwise and of an instance holding four with three.
            return MyClass(4) & 3

        def _or():  # noqa: E306
            # [FUNCTION DOC] HELPER: Bitwise or of an instance holding four with three.
            return MyClass(4) | 3

        def _xor():  # noqa: E306
            # [FUNCTION DOC] HELPER: Bitwise xor of an instance holding four with three.
            return MyClass(4) ^ 3

        def getitem():  # noqa: E306
            # [FUNCTION DOC] HELPER: Use the item read hook of an instance with index one.
            return MyClass(4)[1]

        def setitem():  # noqa: E306
            # [FUNCTION DOC] HELPER: Use the item write hook to set index one to five, then
            # return the stored integer.
            a = MyClass(4)
            a[1] = 5
            return a.x

        def call():  # noqa: E306
            # [FUNCTION DOC] HELPER: Call an instance holding five with the argument two.
            a = MyClass(5)
            return a(2)

        ops = [
            add,
            sub,
            mul,
            pow,
            ne,
            eq,
            lt,
            gt,
            le,
            ge,
            _and,
            _or,
            _xor,
            getitem,
            setitem,
            call,
        ]

        ops.append(truediv)
        for func in ops:
            self.checkScript(func, ())

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", ""
        ):

            @torch.jit.script
            def test():
                # [FUNCTION DOC] HELPER: Add two instances of the first class, which lacks an
                # add hook (compile error expected).
                return Foo(torch.tensor(1)) + Foo(torch.tensor(1))

    def test_cast_overloads(self):
        # [FUNCTION DOC] Check that a scripted class can define integer, float, bool and
        # string conversion hooks used by the builtin conversions and in a condition,
        # comparing scripted output with a plain computation on the float. Also require an
        # error when the bool hook returns a non-bool value.
        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted class holding a float with conversion hooks.
            def __init__(self, val: float) -> None:
                # [FUNCTION DOC] HELPER: Store the float.
                self.val = val

            def __int__(self):
                # [FUNCTION DOC] HELPER: Return the stored float as an integer.
                return int(self.val)

            def __float__(self):
                # [FUNCTION DOC] HELPER: Return the stored float.
                return self.val

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Return the stored float as a bool.
                return bool(self.val)

            def __str__(self):
                # [FUNCTION DOC] HELPER: Return the stored float as a string.
                return str(self.val)

        make_global(Foo)  # see [local resolution in python]

        def test(foo: Foo) -> Tuple[int, float, bool]:
            # [FUNCTION DOC] HELPER: Test the instance as a condition, then return its
            # integer, float and bool conversions as a tuple; also usable in eager mode
            # with plain floats.
            if foo:
                pass
            return int(foo), float(foo), bool(foo)

        fn = torch.jit.script(test)
        self.assertEqual(fn(Foo(0.5)), test(0.5))
        self.assertEqual(fn(Foo(0.0)), test(0.0))
        # str has slightly different formatting
        self.assertTrue("0.5" in (str(Foo(0.5))))
        self.assertTrue("0." in (str(Foo(0.0))))

        @torch.jit.script
        class BadBool:
            # [CLASS DOC] HELPER: Scripted class whose bool hook wrongly returns a tuple.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Return a two-integer tuple instead of a bool (compile
                # error when used as a condition).
                return (1, 2)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected a bool expression for condition", ""
        ):

            @torch.jit.script
            def test():
                # [FUNCTION DOC] HELPER: Use an instance as an if condition;
                # compile error expected.
                if BadBool():
                    print(1)

    def test_init_compiled_first(self):
        # [FUNCTION DOC] Script a class whose first method reads an attribute that only the
        # constructor assigns; scripting must still succeed, showing the constructor is
        # compiled before the other methods.
        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted class with a method that reads an attribute set
            # later in the constructor.
            def __before_init__(self):
                # [FUNCTION DOC] HELPER: Return the attribute assigned by the constructor.
                # accessing this field should not throw, since __init__ should be compiled
                return self.x

            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Store both arguments.
                self.x = x
                self.y = y

    def test_class_constructs_itself(self):
        # [FUNCTION DOC] Script a class whose method constructs another instance of its own
        # class and mutates it; scripting must succeed.
        @torch.jit.script
        class LSTMStateStack:
            # [CLASS DOC] HELPER: Scripted class holding layer sizes, a pair of zero tensors
            # and a stack list, with a copy method.
            def __init__(self, num_layers: int, hidden_size: int) -> None:
                # [FUNCTION DOC] HELPER: Store the sizes, create a pair of zero state tensors
                # and a stack of one entry built from them.
                self.num_layers = num_layers
                self.hidden_size = hidden_size
                self.last_state = (
                    torch.zeros(num_layers, 1, hidden_size),
                    torch.zeros(num_layers, 1, hidden_size),
                )
                self.stack = [(self.last_state[0][-1], self.last_state[0][-1])]

            def copy(self):
                # [FUNCTION DOC] HELPER: Construct a new instance of the same class, copy the
                # stack into it and return it.
                # should be able to construct a class inside its own methods
                other = LSTMStateStack(self.num_layers, self.hidden_size)
                other.stack = list(self.stack)
                return other

    def test_optional_type_promotion(self):
        # [FUNCTION DOC] Script a leaf class and a class declaring an optional leaf
        # attribute that starts as none and is assigned a leaf in a method; scripting must
        # succeed without error.
        @torch.jit.script
        class Leaf:
            # [CLASS DOC] HELPER: Scripted leaf class with one integer attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Set the integer attribute to one.
                self.x = 1

        # should not throw
        @torch.jit.script
        class Tree:
            # [CLASS DOC] HELPER: Scripted class with an optional leaf child.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the child to an annotated optional none.
                self.child = torch.jit.annotate(Optional[Leaf], None)

            def add_child(self, child: Leaf) -> None:
                # [FUNCTION DOC] HELPER: Replace the optional child with the given leaf.
                self.child = child

    def test_recursive_class(self):
        # [FUNCTION DOC] Scripting a class with an attribute annotated as optional of itself
        # must raise RuntimeError, as recursive class types are unsupported.
        """
        Recursive class types not yet supported. We should give a good error message.
        """
        with self.assertRaises(RuntimeError):

            @torch.jit.script
            class Tree:
                # [CLASS DOC] HELPER: Scripted class with a self-referential optional parent.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Initialize the parent attribute to an
                    # annotated optional none.
                    self.parent = torch.jit.annotate(Optional[Tree], None)

    def test_class_constant(self):
        # [FUNCTION DOC] Script, save and reload a module with a class constant for several
        # constant types (int, float, none, bool, string, tuple, list) and check outputs and
        # the constant agree between original and loaded modules.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module declaring one constant attribute and returning it
            # with its input.
            __constants__ = ["w"]

            def __init__(self, w):
                # [FUNCTION DOC] HELPER: Store the constant value.
                super().__init__()
                self.w = w

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the input along with the constant.
                # Make sure class constant is accessible in method
                y = self.w
                return x, y

        # Test serialization/deserialization of class constant
        for c in (2, 1.0, None, True, "str", (2, 3), [5.9, 7.3]):
            m = torch.jit.script(M(c))
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)

            buffer.seek(0)
            m_loaded = torch.jit.load(buffer)
            input = torch.rand(2, 3)
            self.assertEqual(m(input), m_loaded(input))
            # Make sure class constant is accessible from module
            self.assertEqual(m.w, m_loaded.w)

    def test_py_class_to_ivalue_missing_attribute(self):
        # [FUNCTION DOC] Convert a Python instance of an annotated class into a scripted
        # function argument and check it works, then pass a tensor instead and require a
        # RuntimeError saying attribute i is missing.
        class Foo:
            # [CLASS DOC] HELPER: Plain class with an integer and a float attribute
            # declared by annotation.
            i: int
            f: float

            def __init__(self, i: int, f: float):
                # [FUNCTION DOC] HELPER: Store both arguments.
                self.i = i
                self.f = f

        make_global(Foo)  # see [local resolution in python]

        @torch.jit.script
        def test_fn(x: Foo) -> float:
            # [FUNCTION DOC] HELPER: Return the integer attribute plus the float attribute.
            return x.i + x.f

        test_fn(Foo(3, 4.0))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "missing attribute i", ""
        ):
            test_fn(torch.rand(3, 4))

    def test_unused_method(self):
        # [FUNCTION DOC] Check that a method decorated as unused on a scripted class compiles,
        # that a module forward not touching it matches eager, and that calling it directly or
        # through another method raises torch.jit.Error at run time.
        """
        Test unused methods on scripted classes.
        """

        @torch.jit.script
        class Unused:
            # [CLASS DOC] HELPER: Scripted class with a counter, a list, a used method, an
            # unused method and a method that calls the unused one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize a zero counter and an empty integer list.
                self.count: int = 0
                self.items: List[int] = []

            def used(self):
                # [FUNCTION DOC] HELPER: Increment the counter and return it.
                self.count += 1
                return self.count

            @torch.jit.unused
            def unused(self, x: int, y: Iterable[int], **kwargs) -> int:
                # [FUNCTION DOC] HELPER: Body not compiled (decorated as unused); would take
                # the next item of the list attribute.
                a = next(self.items)
                return a

            def uses_unused(self) -> int:
                # [FUNCTION DOC] HELPER: Call the unused method with keyword arguments.
                return self.unused(y="hi", x=3)

        class ModuleWithUnused(nn.Module):
            # [CLASS DOC] HELPER: Module holding an instance of the class with exported
            # methods that call the unused method directly and indirectly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module and create the instance.
                super().__init__()
                self.obj = Unused()

            def forward(self):
                # [FUNCTION DOC] HELPER: Return the result of the used method.
                return self.obj.used()

            @torch.jit.export
            def calls_unused(self):
                # [FUNCTION DOC] HELPER: Exported method calling the unused method of
                # the held object.
                return self.obj.unused(3, "hi")

            @torch.jit.export
            def calls_unused_indirectly(self):
                # [FUNCTION DOC] HELPER: Exported method that calls the method which uses
                # the unused one.
                return self.obj.uses_unused()

        python_module = ModuleWithUnused()
        script_module = torch.jit.script(ModuleWithUnused())

        # Forward should work because it does not used any methods marked unused.
        self.assertEqual(python_module.forward(), script_module.forward())

        # Calling a method marked unused should throw.
        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused()

        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused_indirectly()

    def test_self_referential_method(self):
        # [FUNCTION DOC] Check a scripted class method may use the class name in its
        # annotations, and that a module using it passes checkModule.
        """
        Test that a scripted class can have a method that refers to the class itself
        in its type annotations.
        """

        @torch.jit.script
        class Meta:
            # [CLASS DOC] HELPER: Scripted class with a method taking a list of itself
            # and returning itself.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            def method(self, other: List["Meta"]) -> "Meta":
                # [FUNCTION DOC] HELPER: Return a new instance whose value is the length of
                # the given list.
                return Meta(len(other))

        class ModuleWithMeta(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding an instance of the self-referencing class.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Initialize the module and create the instance.
                super().__init__()
                self.meta = Meta(a)

            def forward(self):
                # [FUNCTION DOC] HELPER: Call the method with a list of the held instance and
                # return the resulting value.
                new_obj = self.meta.method([self.meta])
                return new_obj.a

        self.checkModule(ModuleWithMeta(5), ())

    def test_type_annotation(self):
        # [FUNCTION DOC] Script a class whose attributes are annotated containers, plus
        # a function assigning a new dictionary and list to them; the test only checks
        # that compilation works.
        """
        Test that annotating container attributes with types works correctly
        """

        @torch.jit.script
        class CompetitiveLinkingTokenReplacementUtils:
            # [CLASS DOC] HELPER: Scripted class with annotated list and dict attributes.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize both attributes to empty containers.
                self.my_list: List[Tuple[float, int, int]] = []
                self.my_dict: Dict[int, int] = {}

        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Build an instance and replace its containers
            # with new ones.
            y = CompetitiveLinkingTokenReplacementUtils()
            new_dict: Dict[int, int] = {1: 1, 2: 2}
            y.my_dict = new_dict

            new_list: List[Tuple[float, int, int]] = [(1.0, 1, 1)]
            y.my_list = new_list
            return y

    def test_default_args(self):
        # [FUNCTION DOC] Check default arguments in scripted class constructors and methods:
        # all defaults, some overridden, all overridden, a method default; a constructor with
        # some required arguments; and that a mutable default argument raises RuntimeError.
        """
        Test that methods on class types can have default arguments.
        """

        @torch.jit.script
        class ClassWithDefaultArgs:
            # [CLASS DOC] HELPER: Scripted class with a constructor and method that
            # have default values.
            def __init__(
                self,
                a: int = 1,
                b: Optional[List[int]] = None,
                c: Tuple[int, int, int] = (1, 2, 3),
                d: Optional[Dict[int, int]] = None,
                e: Optional[str] = None,
            ):
                # [FUNCTION DOC] HELPER: Store an integer, tuple and optional string; keep
                # given list and dict if supplied, else use built-in defaults.
                self.int = a
                self.tup = c
                self.str = e

                self.list = [1, 2, 3]
                if b is not None:
                    self.list = b

                self.dict = {1: 2, 3: 4}
                if d is not None:
                    self.dict = d

            def add(self, b: int, scale: float = 1.0) -> float:
                # [FUNCTION DOC] HELPER: Return the integer times the scale plus the argument.
                return self.int * scale + b

        def all_defaults() -> int:
            # [FUNCTION DOC] HELPER: Build with no arguments; return the sum of integer, list
            # item two and tuple item one.
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.int + obj.list[2] + obj.tup[1]

        def some_defaults() -> int:
            # [FUNCTION DOC] HELPER: Build with only a list; return integer plus list item two
            # plus the dict value for key one.
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(b=[5, 6, 7])
            return obj.int + obj.list[2] + obj.dict[1]

        def override_defaults() -> int:
            # [FUNCTION DOC] HELPER: Build with every argument supplied and return the total
            # of the integer, the list items, the tuple items, the dict value stored under key
            # three and the length of the string.
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs(
                3, [9, 10, 11], (12, 13, 14), {3: 4}, "str"
            )
            s: int = obj.int

            for x in obj.list:
                s += x

            for y in obj.tup:
                s += y

            s += obj.dict[3]

            st = obj.str
            if st is not None:
                s += len(st)

            return s

        def method_defaults() -> float:
            # [FUNCTION DOC] HELPER: Return the method result with the default scale plus the
            # result with a scale of one quarter, both with argument three.
            obj: ClassWithDefaultArgs = ClassWithDefaultArgs()
            return obj.add(3) + obj.add(3, 0.25)

        self.checkScript(all_defaults, ())
        self.checkScript(some_defaults, ())
        self.checkScript(override_defaults, ())
        self.checkScript(method_defaults, ())

        # The constructor of this class below has some arguments without default values.
        class ClassWithSomeDefaultArgs:
            # [CLASS DOC] HELPER: Plain class with a required and a
            # defaulted constructor argument.
            def __init__(
                self,
                a: int,
                b: int = 1,
            ):
                # [FUNCTION DOC] HELPER: Store both arguments.
                self.a = a
                self.b = b

        def default_b() -> int:
            # [FUNCTION DOC] HELPER: Build with only the required argument and return the sum
            # of both attributes.
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1)
            return obj.a + obj.b

        def set_b() -> int:
            # [FUNCTION DOC] HELPER: Build with both arguments; return the sum
            # of both attributes.
            obj: ClassWithSomeDefaultArgs = ClassWithSomeDefaultArgs(1, 4)
            return obj.a + obj.b

        self.checkScript(default_b, ())
        self.checkScript(set_b, ())

        # The constructor of this class below has mutable arguments. This should throw
        # an error.
        class ClassWithMutableArgs:
            # [CLASS DOC] HELPER: Plain class with a mutable default
            # list constructor argument.
            def __init__(
                self,
                a: List[int] = [1, 2, 3],  # noqa: B006
            ):
                # [FUNCTION DOC] HELPER: Store the list argument.
                self.a = a

        def should_fail():
            # [FUNCTION DOC] HELPER: Construct with the default list; compile error expected.
            obj: ClassWithMutableArgs = ClassWithMutableArgs()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Mutable default parameters are not supported", ""
        ):
            torch.jit.script(should_fail)

    def test_staticmethod(self):
        # [FUNCTION DOC] Check static methods on a scripted class: one constructs an instance,
        # one calls another static method, and a scripted function calls the second.
        """
        Test static methods on class types.
        """

        @torch.jit.script
        class ClassWithStaticMethod:
            # [CLASS DOC] HELPER: Scripted class with two integers, an equality hook and
            # two static methods.
            def __init__(self, a: int, b: int):
                # [FUNCTION DOC] HELPER: Store both integers.
                self.a: int = a
                self.b: int = b

            def get_a(self):
                # [FUNCTION DOC] HELPER: Return the first integer.
                return self.a

            def get_b(self):
                # [FUNCTION DOC] HELPER: Return the second integer.
                return self.b

            def __eq__(self, other: "ClassWithStaticMethod"):
                # [FUNCTION DOC] HELPER: Return whether both integers match
                # the other instance.
                return self.a == other.a and self.b == other.b

            # staticmethod that calls constructor.
            @staticmethod
            def create(args: List["ClassWithStaticMethod"]) -> "ClassWithStaticMethod":
                # [FUNCTION DOC] HELPER: Static method building a new instance from the
                # first list item.
                return ClassWithStaticMethod(args[0].a, args[0].b)

            # staticmethod that calls another staticmethod.
            @staticmethod
            def create_from(a: int, b: int) -> "ClassWithStaticMethod":
                # [FUNCTION DOC] HELPER: Static method building an instance and passing it
                # through the other static method.
                a = ClassWithStaticMethod(a, b)
                return ClassWithStaticMethod.create([a])

        # Script function that calls staticmethod.
        def test_function(a: int, b: int) -> "ClassWithStaticMethod":
            # [FUNCTION DOC] HELPER: Call the static method that builds from two integers.
            return ClassWithStaticMethod.create_from(a, b)

        make_global(ClassWithStaticMethod)

        self.checkScript(test_function, (1, 2))

    def test_classmethod(self):
        # [FUNCTION DOC] Check a classmethod on a scripted class can be called on an instance
        # in a scripted function.
        """
        Test classmethods on class types.
        """

        @torch.jit.script
        class ClassWithClassMethod:
            # [CLASS DOC] HELPER: Scripted class with an integer, an equality hook and a
            # class method constructor.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a: int = a

            def __eq__(self, other: "ClassWithClassMethod"):
                # [FUNCTION DOC] HELPER: Return whether the integers match.
                return self.a == other.a

            @classmethod
            def create(cls, a: int) -> "ClassWithClassMethod":
                # [FUNCTION DOC] HELPER: Class method returning an instance of the class built
                # from the argument.
                return cls(a)

        make_global(ClassWithClassMethod)

        def test_function(a: int) -> "ClassWithClassMethod":
            # [FUNCTION DOC] HELPER: Build an instance and call the class method through it.
            x = ClassWithClassMethod(a)
            # Support calling classmethod with an instance
            # Calling with the class is not supported.
            return x.create(a)

        self.checkScript(test_function, (1,))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_properties(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Check property support on scripted
        # classes: getter only, getter with setter, a property that is marked unsupported
        # and skipped, and a property used inside a method; the module result is compared
        # eager versus script.
        """
        Test that a scripted class can make use of the @property decorator.
        """

        def free_function(x: int) -> int:
            # [FUNCTION DOC] HELPER: Return the integer plus one.
            return x + 1

        @torch.jit.script
        class Properties:
            # [CLASS DOC] HELPER: Scripted class with a property with setter, an unsupported
            # property listed as unused and a second unused property with setter.
            __jit_unused_properties__ = ["unsupported"]

            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            @property
            def attr(self) -> int:
                # [FUNCTION DOC] HELPER: Property returning the stored integer minus one.
                return self.a - 1

            @property
            def unsupported(self) -> int:
                # [FUNCTION DOC] HELPER: Property excluded from scripting; its body sums
                # a one-item list.
                return sum([self.a])

            @torch.jit.unused
            @property
            def unsupported_2(self) -> int:
                # [FUNCTION DOC] HELPER: Property marked unused; would return the sum of
                # a one-item list.
                return sum([self.a])

            @unsupported_2.setter
            def unsupported_2(self, value):
                # [FUNCTION DOC] HELPER: Setter for the unused property; sets the stored
                # integer to the sum of a one-item list.
                self.a = sum([self.a])

            @attr.setter
            def attr(self, value: int):
                # [FUNCTION DOC] HELPER: Setter storing the value plus three.
                self.a = value + 3

        @torch.jit.script
        class NoSetter:
            # [CLASS DOC] HELPER: Scripted class with a read-only property.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            @property
            def attr(self) -> int:
                # [FUNCTION DOC] HELPER: Property returning the stored integer plus one via
                # the free function.
                return free_function(self.a)

        @torch.jit.script
        class MethodThatUsesProperty:
            # [CLASS DOC] HELPER: Scripted class with a property, setter and a method
            # reading the property.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            @property
            def attr(self) -> int:
                # [FUNCTION DOC] HELPER: Property returning the stored integer minus two.
                return self.a - 2

            @attr.setter
            def attr(self, value: int):
                # [FUNCTION DOC] HELPER: Setter storing the value plus four.
                self.a = value + 4

            def forward(self):
                # [FUNCTION DOC] HELPER: Return the property value.
                return self.attr

        class ModuleWithProperties(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding one properties instance and exercising
            # getters and setters on several instances in forward.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Initialize the module and create the instance.
                super().__init__()
                self.props = Properties(a)

            def forward(self, a: int, b: int, c: int, d: int):
                # [FUNCTION DOC] HELPER: Set properties on the held and on new
                # instances, then return the sum of the held property, the read-only
                # property and the method result.
                self.props.attr = a
                props = Properties(b)
                no_setter = NoSetter(c)
                method_uses_property = MethodThatUsesProperty(a + b)

                props.attr = c
                method_uses_property.attr = d

                return self.props.attr + no_setter.attr + method_uses_property.forward()

        self.checkModule(
            ModuleWithProperties(5),
            (
                5,
                6,
                7,
                8,
            ),
        )

    def test_custom_delete(self):
        # [FUNCTION DOC] Check that deleting an item calls the delete hook of a class and that
        # a class without the hook produces a RuntimeError saying it does not define the
        # delete hook, highlighting the delete target.
        """
        Test that del can be called on an instance of a class that
        overrides __delitem__.
        """

        class Example:
            # [CLASS DOC] HELPER: Plain class with a dictionary of tensors, a membership check
            # and a delete hook.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create a dictionary containing one entry.
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the key is in the dictionary.
                return key in self._data

            def __delitem__(self, key: str):
                # [FUNCTION DOC] HELPER: Remove the key from the dictionary.
                del self._data[key]

        def fn() -> bool:
            # [FUNCTION DOC] HELPER: Delete the entry and return whether it is still there.
            example = Example()
            del example["1"]
            return example.check("1")

        self.checkScript(fn, ())

        # Test the case in which the class does not have __delitem__ defined.
        class NoDelItem:
            # [CLASS DOC] HELPER: Plain class with a dictionary and membership check but
            # no delete hook.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create a dictionary containing one entry.
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                # [FUNCTION DOC] HELPER: Return whether the key is in the dictionary.
                return key in self._data

        def fn() -> bool:
            # [FUNCTION DOC] HELPER: Try to delete an entry from a class without the hook
            # (compile error expected).
            example = NoDelItem()
            key = "1"
            del example[key]
            return example.check(key)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Class does not define __delitem__", "example[key]"
        ):
            self.checkScript(fn, ())

    def test_recursive_script_builtin_type_resolution(self):
        # [FUNCTION DOC] Check implicit recursive compilation resolves torch types used
        # in method annotations through aliases (tensor and device) or string
        # references, by scripting functions using the class, checking them against
        # eager and after export and import.
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled.
        """
        # A will be implicitly compiled because it is not annotated with @torch.jit.script
        # but is used in g() below.
        tensor_t = torch.Tensor
        device_t = torch.device
        device_ty = torch.device

        class A:
            # [CLASS DOC] HELPER: Unannotated plain class compiled implicitly, with methods
            # using torch types through aliases.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            def f(self, x: tensor_t, y: torch.device) -> tensor_t:
                # [FUNCTION DOC] HELPER: Move the tensor to the device.
                return x.to(device=y)

            def g(self, x: device_t) -> device_ty:
                # [FUNCTION DOC] HELPER: Return the device argument.
                return x

            def h(self, a: "A") -> "A":
                # [FUNCTION DOC] HELPER: Return a new instance of the class,
                # ignoring the argument.
                return A()

            def i(self, a: List[int]) -> int:
                # [FUNCTION DOC] HELPER: Return the first element of the integer list.
                return a[0]

            def j(self, l: List[device_t]) -> device_ty:
                # [FUNCTION DOC] HELPER: Return the first element of the device list.
                return l[0]

        def call_f():
            # [FUNCTION DOC] HELPER: Call the tensor-move method with a one-item tensor and
            # the cpu device.
            a = A()
            return a.f(torch.tensor([1]), torch.device("cpu"))

        def call_g():
            # [FUNCTION DOC] HELPER: Call the device-returning method with the cpu device.
            a = A()
            return a.g(torch.device("cpu"))

        def call_i():
            # [FUNCTION DOC] HELPER: Call the first-element method on a one-item integer list.
            a = A()
            return a.i([3])

        def call_j():
            # [FUNCTION DOC] HELPER: Call the device list method with two cpu devices.
            a = A()
            return a.j([torch.device("cpu"), torch.device("cpu")])

        for fn in [call_f, call_g, call_i, call_j]:
            self.checkScript(fn, ())
            s = self.getExportImportCopy(torch.jit.script(fn))
            self.assertEqual(s(), fn())

    def test_recursive_script_module_builtin_type_resolution(self):
        # [FUNCTION DOC] Check that a module returning a plain class, whose method
        # annotation uses a list of torch devices, can be scripted and exported; it only
        # requires that nothing raises.
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled
        when compiling a module.
        """

        class Wrapper:
            # [CLASS DOC] HELPER: Plain class wrapping a tensor with a method that moves it
            # to a device.
            def __init__(self, t):
                # [FUNCTION DOC] HELPER: Store the tensor.
                self.t = t

            def to(self, l: List[torch.device], device: Optional[torch.device] = None):
                # [FUNCTION DOC] HELPER: Move the stored tensor to the device; the device list
                # argument is unused.
                return self.t.to(device=device)

        class A(nn.Module):
            # [CLASS DOC] HELPER: Module returning a wrapper of a random tensor from forward.
            def forward(self):
                # [FUNCTION DOC] HELPER: Return a wrapper holding a random 4 by 4 tensor.
                return Wrapper(torch.rand(4, 4))

        scripted = torch.jit.script(A())
        self.getExportImportCopy(scripted)

    def test_class_attribute_wrong_type(self):
        # [FUNCTION DOC] Script a module holding two class instances whose unannotated
        # attribute holds strings, selected by a flag; scripting must raise RuntimeError
        # saying the attribute could not be cast to Tensor.
        """
        Test that the error message displayed when converting a class type
        to an IValue that has an attribute of the wrong type.
        """

        @torch.jit.script
        class ValHolder:
            # [CLASS DOC] HELPER: Scripted class storing one untyped value (a
            # tensor by default).
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Store the value.
                self.val = val

        class Mod(nn.Module):
            # [CLASS DOC] HELPER: Module holding two instances of the class
            # built from strings.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module and create both
                # instances holding strings.
                super().__init__()
                self.mod1 = ValHolder("1")
                self.mod2 = ValHolder("2")

            def forward(self, cond: bool):
                # [FUNCTION DOC] HELPER: Choose one of the two instances by the flag and
                # return its value.
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Could not cast attribute 'val' to type Tensor", ""
        ):
            torch.jit.script(Mod())

    def test_recursive_scripting(self):
        # [FUNCTION DOC] Check a Python class instance stored as a module attribute is
        # recursively scripted by checkModule.
        """
        Test that class types are recursively scripted when an Python instance of one
        is encountered as a module attribute.
        """

        class Class:
            # [CLASS DOC] HELPER: Plain class holding an integer with a getter.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            def get_a(self) -> int:
                # [FUNCTION DOC] HELPER: Return the integer.
                return self.a

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding any object passed in and using its getter.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Store the object.
                super().__init__()
                self.obj = obj

            def forward(self) -> int:
                # [FUNCTION DOC] HELPER: Return the result of the object getter.
                return self.obj.get_a()

        self.checkModule(M(Class(4)), ())

    def test_recursive_scripting_failed(self):
        # [FUNCTION DOC] Check that an unscriptable class instance stored on a module only
        # causes an error if used in scripted code: using it raises RuntimeError about
        # failing to convert the Python type, while ignoring it in an ignored method allows
        # scripting and checkModule.
        """
        Test that class types module attributes that fail to script
        are added as failed attributes and do not cause compilation itself
        to fail unless they are used in scripted code.
        """

        class UnscriptableClass:
            # [CLASS DOC] HELPER: Plain class whose getter uses something
            # TorchScript cannot compile.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Store the integer.
                self.a = a

            def get_a(self) -> bool:
                # [FUNCTION DOC] HELPER: Use a type-check builtin on the integer; scripting
                # cannot compile it.
                return issubclass(self.a, int)

        # This Module has an attribute of type UnscriptableClass
        # and tries to use it in scripted code. This should fail.
        class ShouldNotCompile(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward uses the unscriptable object,
            # so compilation fails.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Store the object.
                super().__init__()
                self.obj = obj

            def forward(self) -> bool:
                # [FUNCTION DOC] HELPER: Return the result of the object getter.
                return self.obj.get_a()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to convert Python type", ""
        ):
            torch.jit.script(ShouldNotCompile(UnscriptableClass(4)))

        # This Module has an attribute of type UnscriptableClass
        # and does not try to use it in scripted code. This should not fail.
        class ShouldCompile(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing the unscriptable object but using it only in
            # an ignored method, so compilation succeeds.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Store the object.
                super().__init__()
                self.obj = obj

            @torch.jit.ignore
            def ignored_method(self) -> bool:
                # [FUNCTION DOC] HELPER: Ignored method (not compiled) returning the
                # object getter result.
                return self.obj.get_a()

            def forward(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Return the integer added to itself.
                return x + x

        self.checkModule(ShouldCompile(UnscriptableClass(4)), (4,))

    def test_unresolved_class_attributes(self):
        # [FUNCTION DOC] Class-level attributes (tuple-unpacked ones and an annotated one) are
        # not instance attributes: scripting each function that reads one must raise
        # RuntimeError saying the object has no such attribute or method and that it is
        # defined as a class attribute.
        class UnresolvedAttrClass:
            # [CLASS DOC] HELPER: Plain class with class-level attributes defined by unpacking
            # and by annotation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: No-op constructor.
                pass

            (attr_a, attr_b), [attr_c, attr_d] = ("", ""), ["", ""]
            attr_e: int = 0

        def fn_a():
            # [FUNCTION DOC] HELPER: Read the first unpacked class attribute from an instance.
            u = UnresolvedAttrClass()
            return u.attr_a

        def fn_b():
            # [FUNCTION DOC] HELPER: Read the second unpacked class attribute of an instance.
            u = UnresolvedAttrClass()
            return u.attr_b

        def fn_c():
            # [FUNCTION DOC] HELPER: Read the third unpacked class attribute from an instance.
            u = UnresolvedAttrClass()
            return u.attr_c

        def fn_d():
            # [FUNCTION DOC] HELPER: Read the fourth unpacked class attribute of an instance.
            u = UnresolvedAttrClass()
            return u.attr_d

        def fn_e():
            # [FUNCTION DOC] HELPER: Read the annotated class attribute from an instance.
            u = UnresolvedAttrClass()
            return u.attr_e

        error_message_regex = (
            "object has no attribute or method.*is defined as a class attribute"
        )
        for fn in (fn_a, fn_b, fn_c, fn_d, fn_e):
            with self.assertRaisesRegex(RuntimeError, error_message_regex):
                torch.jit.script(fn)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
