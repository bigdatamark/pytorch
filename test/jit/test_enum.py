# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for Python Enum support in TorchScript: supported and unsupported value
# types, comparison, name and value access, constants, closed-over enums, enums as module
# attributes, returning and iterating enums, and explicit scripting. Most tests use FileCheck on
# the scripted graph text. Importing adds the test directory to sys.path; running the file
# directly is refused in favor of test/test_jit.py.
import os
import sys
from enum import Enum
from typing import Any, List

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestEnum(JitTestCase):
    # [CLASS DOC] Test case for enum handling in the TorchScript compiler; each test defines its
    # own small enum classes and, where needed, registers them as global with make_global so
    # scripting can resolve them.
    def test_enum_value_types(self):
        # [FUNCTION DOC] Checks which enum value types script: integer, float and string valued
        # enums used as argument types in a function returning their names compile, and the
        # graph mentions all three enum class names. A tensor-valued enum used as an argument
        # type must fail scripting with RuntimeError about value type Tensor.
        class IntEnum(Enum):
            # [CLASS DOC] HELPER: Enum with integer values.
            FOO = 1
            BAR = 2

        class FloatEnum(Enum):
            # [CLASS DOC] HELPER: Enum with float values.
            FOO = 1.2
            BAR = 2.3

        class StringEnum(Enum):
            # [CLASS DOC] HELPER: Enum with string values.
            FOO = "foo as in foo bar"
            BAR = "bar as in foo bar"

        make_global(IntEnum, FloatEnum, StringEnum)

        @torch.jit.script
        def supported_enum_types(a: IntEnum, b: FloatEnum, c: StringEnum):
            # [FUNCTION DOC] HELPER: Scripted function returning the names of its three enum
            # arguments as a tuple.
            return (a.name, b.name, c.name)

        FileCheck().check("IntEnum").check("FloatEnum").check("StringEnum").run(
            str(supported_enum_types.graph)
        )

        class TensorEnum(Enum):
            # [CLASS DOC] HELPER: Enum whose values are tensors, an unsupported value type.
            FOO = torch.tensor(0)
            BAR = torch.tensor(1)

        make_global(TensorEnum)

        def unsupported_enum_types(a: TensorEnum):
            # [FUNCTION DOC] HELPER: Returns the name of a tensor-valued enum argument; never
            # successfully compiled.
            return a.name

        # TODO: rewrite code so that the highlight is not empty.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Cannot create Enum with value type 'Tensor'", ""
        ):
            torch.jit.script(unsupported_enum_types)

    def test_enum_comp(self):
        # [FUNCTION DOC] Scripts a function comparing two enum values for equality; the graph
        # must contain an equality op, and the result is true for equal members and false for
        # different ones.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members, RED and GREEN.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_comp(x: Color, y: Color) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the two enum arguments are equal.
            return x == y

        FileCheck().check("aten::eq").run(str(enum_comp.graph))

        self.assertEqual(enum_comp(Color.RED, Color.RED), True)
        self.assertEqual(enum_comp(Color.RED, Color.GREEN), False)

    def test_enum_comp_diff_classes(self):
        # [FUNCTION DOC] Checks comparing a value of one enum class to a member of a different
        # class with identical member values: the graph holds a constant for the other class
        # member followed by an equality op, and the result at runtime is false.
        class Foo(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members, same values as the
            # sibling enum.
            ITEM1 = 1
            ITEM2 = 2

        class Bar(Enum):
            # [CLASS DOC] HELPER: Second enum with the same member names and values as the
            # first, used to show enums of different classes are not equal.
            ITEM1 = 1
            ITEM2 = 2

        make_global(Foo, Bar)

        @torch.jit.script
        def enum_comp(x: Foo) -> bool:
            # [FUNCTION DOC] HELPER: Compares the argument with a member of the other
            # enum class.
            return x == Bar.ITEM1

        FileCheck().check("prim::Constant").check_same("Bar.ITEM1").check(
            "aten::eq"
        ).run(str(enum_comp.graph))

        self.assertEqual(enum_comp(Foo.ITEM1), False)

    def test_heterogenous_value_type_enum_error(self):
        # [FUNCTION DOC] Checks that an enum with mixed value types (integer and string) fails
        # scripting of a function using it, raising RuntimeError about being unable to unify the
        # type list.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with an integer member and a string member, which
            # is unsupported.
            RED = 1
            GREEN = "green"

        make_global(Color)

        def enum_comp(x: Color, y: Color) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the two enum arguments are equal; never
            # successfully compiled.
            return x == y

        # TODO: rewrite code so that the highlight is not empty.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Could not unify type list", ""
        ):
            torch.jit.script(enum_comp)

    def test_enum_name(self):
        # [FUNCTION DOC] Scripts a function returning an enum name; the graph must show the enum
        # type followed directly by an enum-name op and a return, and results equal the Python
        # name attribute for both members.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_name(x: Color) -> str:
            # [FUNCTION DOC] HELPER: Returns the name of the enum argument.
            return x.name

        FileCheck().check("Color").check_next("prim::EnumName").check_next(
            "return"
        ).run(str(enum_name.graph))

        self.assertEqual(enum_name(Color.RED), Color.RED.name)
        self.assertEqual(enum_name(Color.GREEN), Color.GREEN.name)

    def test_enum_value(self):
        # [FUNCTION DOC] Scripts a function returning an enum value; the graph must show the
        # enum type followed directly by an enum-value op and a return, and results equal the
        # Python value attribute for both members.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_value(x: Color) -> int:
            # [FUNCTION DOC] HELPER: Returns the value of the enum argument.
            return x.value

        FileCheck().check("Color").check_next("prim::EnumValue").check_next(
            "return"
        ).run(str(enum_value.graph))

        self.assertEqual(enum_value(Color.RED), Color.RED.value)
        self.assertEqual(enum_value(Color.GREEN), Color.GREEN.value)

    def test_enum_as_const(self):
        # [FUNCTION DOC] Scripts a function comparing its argument with an enum member written
        # in the body; the graph must have a constant holding that member, then an equality op
        # and a return. Results are true for the matching member only.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def enum_const(x: Color) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the argument equals the first member.
            return x == Color.RED

        FileCheck().check(
            "prim::Constant[value=__torch__.jit.test_enum.Color.RED]"
        ).check_next("aten::eq").check_next("return").run(str(enum_const.graph))

        self.assertEqual(enum_const(Color.RED), True)
        self.assertEqual(enum_const(Color.GREEN), False)

    def test_non_existent_enum_value(self):
        # [FUNCTION DOC] Checks that referencing a member that does not exist in the enum fails
        # scripting with RuntimeError saying the enum has no such attribute, highlighting the
        # offending expression.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members, none named PURPLE.
            RED = 1
            GREEN = 2

        make_global(Color)

        def enum_const(x: Color) -> bool:
            # [FUNCTION DOC] HELPER: Compares the argument with a nonexistent member; never
            # successfully compiled.
            if x == Color.PURPLE:
                return True
            else:
                return False

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "has no attribute 'PURPLE'", "Color.PURPLE"
        ):
            torch.jit.script(enum_const)

    def test_enum_ivalue_type(self):
        # [FUNCTION DOC] Scripts a function that tests whether an argument of any type is an
        # instance of the enum class; the graph must contain an isinstance op on the enum type
        # followed by a return. Enum members give true and the integer one gives false.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def is_color_enum(x: Any):
            # [FUNCTION DOC] HELPER: Returns whether the untyped argument is an instance of the
            # enum class.
            return isinstance(x, Color)

        FileCheck().check(
            "prim::isinstance[types=[Enum<__torch__.jit.test_enum.Color>]]"
        ).check_next("return").run(str(is_color_enum.graph))

        self.assertEqual(is_color_enum(Color.RED), True)
        self.assertEqual(is_color_enum(Color.GREEN), True)
        self.assertEqual(is_color_enum(1), False)

    def test_closed_over_enum_constant(self):
        # [FUNCTION DOC] Checks scripting a function that reads a closed-over enum. First
        # captures the enum class under an alias and returns a member value through it; the
        # graph must hold an integer constant followed by a return. Then captures a member as a
        # variable and returns its value, with the same graph check. Both calls return the
        # Python value.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        a = Color

        @torch.jit.script
        def closed_over_aliased_type():
            # [FUNCTION DOC] HELPER: Returns the value of a member accessed through a
            # closed-over alias of the enum class.
            return a.RED.value

        FileCheck().check("prim::Constant[value={}]".format(a.RED.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_type.graph))

        self.assertEqual(closed_over_aliased_type(), Color.RED.value)

        b = Color.RED

        @torch.jit.script
        def closed_over_aliased_value():
            # [FUNCTION DOC] HELPER: Returns the value of a closed-over enum member.
            return b.value

        FileCheck().check("prim::Constant[value={}]".format(b.value)).check_next(
            "return"
        ).run(str(closed_over_aliased_value.graph))

        self.assertEqual(closed_over_aliased_value(), Color.RED.value)

    def test_enum_as_module_attribute(self):
        # [FUNCTION DOC] Scripts a module storing an enum member as an attribute and
        # returning its value. The graph must show the module, the enum type, a get-attribute
        # of that field, an enum-value op and a return; calling the scripted module returns
        # the member value.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing an enum member given at construction and
            # returning its value in forward.
            def __init__(self, e: Color):
                # [FUNCTION DOC] HELPER: Stores the enum argument as attribute e.
                super().__init__()
                self.e = e

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the value of the stored enum attribute.
                return self.e.value

        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("prim::EnumValue").check_next("return").run(str(scripted.graph))

        self.assertEqual(scripted(), Color.RED.value)

    def test_string_enum_as_module_attribute(self):
        # [FUNCTION DOC] Scripts a module with a string-valued enum attribute whose forward
        # returns the member name and value as a tuple (the enum is made global before
        # scripting); the scripted result must match the Python name and value.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two string-valued members.
            RED = "red"
            GREEN = "green"

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing an enum member given at construction; forward
            # returns its name and value.
            def __init__(self, e: Color):
                # [FUNCTION DOC] HELPER: Stores the enum argument as attribute e.
                super().__init__()
                self.e = e

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a tuple of the stored member's name and value.
                return (self.e.name, self.e.value)

        make_global(Color)
        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        self.assertEqual(scripted(), (Color.RED.name, Color.RED.value))

    def test_enum_return(self):
        # [FUNCTION DOC] Checks a scripted function can return enum members: true yields RED and
        # false yields GREEN.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        make_global(Color)

        @torch.jit.script
        def return_enum(cond: bool):
            # [FUNCTION DOC] HELPER: Returns the first member if the flag is true, otherwise the
            # second member.
            if cond:
                return Color.RED
            else:
                return Color.GREEN

        self.assertEqual(return_enum(True), Color.RED)
        self.assertEqual(return_enum(False), Color.GREEN)

    def test_enum_module_return(self):
        # [FUNCTION DOC] Scripts a module that stores an enum member as an attribute and returns
        # it itself; the graph must show the module, the enum type, a get-attribute of the field
        # and a return, and the call returns the member.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members.
            RED = 1
            GREEN = 2

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing an enum member given at construction and
            # returning it from forward.
            def __init__(self, e: Color):
                # [FUNCTION DOC] HELPER: Stores the enum argument as attribute e.
                super().__init__()
                self.e = e

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the stored enum member.
                return self.e

        make_global(Color)
        m = TestModule(Color.RED)
        scripted = torch.jit.script(m)

        FileCheck().check("TestModule").check_next("Color").check_same(
            'prim::GetAttr[name="e"]'
        ).check_next("return").run(str(scripted.graph))

        self.assertEqual(scripted(), Color.RED)

    def test_enum_iterate(self):
        # [FUNCTION DOC] Scripts a function that loops over all members of an enum class,
        # collecting values of the members different from the argument. The graph type must list
        # all three members. Results must follow the definition order of the enum, excluding the
        # passed member.
        # KNOWN ISSUE: the in-code comment about ordering names a member that does not exist;
        # the enum has RED, GREEN and BLUE members.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with three integer members in order RED, GREEN, BLUE.
            RED = 1
            GREEN = 2
            BLUE = 3

        def iterate_enum(x: Color):
            # [FUNCTION DOC] HELPER: Loops over the enum class and returns the values of all
            # members not equal to the argument, in definition order.
            res: List[int] = []
            for e in Color:
                if e != x:
                    res.append(e.value)
            return res

        make_global(Color)
        scripted = torch.jit.script(iterate_enum)

        FileCheck().check("Enum<__torch__.jit.test_enum.Color>[]").check_same(
            "Color.RED"
        ).check_same("Color.GREEN").check_same("Color.BLUE").run(str(scripted.graph))

        # PURPLE always appears last because we follow Python's Enum definition order.
        self.assertEqual(scripted(Color.RED), [Color.GREEN.value, Color.BLUE.value])
        self.assertEqual(scripted(Color.GREEN), [Color.RED.value, Color.BLUE.value])

    # Tests that explicitly and/or repeatedly scripting an Enum class is permitted.
    def test_enum_explicit_script(self):
        # [FUNCTION DOC] Checks that an enum class can be scripted with the decorator and then
        # scripted again explicitly without error.
        @torch.jit.script
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with two integer members, scripted by decorator.
            RED = 1
            GREEN = 2

        torch.jit.script(Color)

    # Regression test for https://github.com/pytorch/pytorch/issues/108933
    def test_typed_enum(self):
        # [FUNCTION DOC] Regression test for issue 108933: an enum that also derives from int
        # must be usable as an argument type in a scripted function that compares it with a
        # member. Nothing is asserted beyond scripting succeeding.
        class Color(int, Enum):
            # [CLASS DOC] HELPER: Integer-derived enum with two members.
            RED = 1
            GREEN = 2

        @torch.jit.script
        def is_red(x: Color) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the argument equals the first member.
            return x == Color.RED


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
