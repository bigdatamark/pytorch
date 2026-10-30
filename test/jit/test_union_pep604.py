# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Same Union tests as the typing-spelling file but with unions written using the
# vertical-bar operator in annotations (including template strings), plus two cases marked as
# expected failures for optional-of-union returns and flattening. Importing only extends the
# module search path; running the file directly is rejected.

import io
import os
import sys
import unittest
from enum import Enum
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestUnion(JitTestCase):
    # [CLASS DOC] Test case (written with the vertical-bar union syntax) for TorchScript Union
    # types: annotations (arguments, returns, locals, containers, class and module
    # attributes), flattening and de-duplication of members, equivalence with Optional, type
    # refinement with isinstance, branch typing rules, serialization, and assignment of list
    # and dict values to union annotations. Stateless; checkScript is not always usable
    # because eager and script isinstance semantics differ, so some tests keep separate
    # equivalent code.
    """
    This class tests the functionality of `Union`.

    Note: It's important to be able to refine the type of a `Union` to
    one of its internal types. Currently, there are differences in the
    way Python expects `isinstance` checks and the way TorchScript
    expects `isinstance` checks. This means that we can't use
    `checkScript` in our test cases because either the eager mode or the
    script mode wouldn't run! So, some test cases have separate but
    equivalent functions to emulate `checkScript`.
    """

    def test_check_union_annotation(self):
        # [FUNCTION DOC] Scripts a function with a union argument and an optional argument,
        # then checks that the graph text prints the union in IR style and the generated code
        # prints it in typing style with an optional int. Also runs checkScript and verifies
        # the graph text can be parsed back by the IR parser.
        def test_func(a: int | float, b: Optional[int]):
            # [FUNCTION DOC] HELPER: Function taking a union of int and float and an optional
            # int; returns zero.
            return 0

        scripted_func = torch.jit.script(test_func)
        graph_rep = str(scripted_func.graph)
        code_rep = str(scripted_func.code)
        # TS graph IR for Union should be annotated as Union()
        FileCheck().check("Union(").check("int?").run(graph_rep)
        # Serialized code for Union should be annotated as Union[]
        FileCheck().check("Union[").check("Optional[int]").run(code_rep)
        self.checkScript(test_func, (5, 6))
        # this shouldn't error out
        torch._C.parse_ir(str(scripted_func.graph))

    def test_union_with_scalar_values(self):
        # [FUNCTION DOC] Checks a function accepting a union of int and float works for both
        # via checkScript, and that calling the scripted version with a string raises
        # RuntimeError saying a member of the union was expected but a str was found.
        def fn(x: int | float) -> str:
            # [FUNCTION DOC] HELPER: Accepts an int or float and returns the string foo.
            return "foo"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[float, int\] but "
            "instead found type str",
        ):
            scripted("1")

    def test_union_with_collections(self):
        # [FUNCTION DOC] Checks a union of a str to int dict and an int list accepts both via
        # checkScript, and that scripted calls with a str to str dict, a str list and a plain
        # str each raise RuntimeError naming the expected union members and the offending
        # type.
        def fn(x: Dict[str, int] | List[int]) -> str:
            # [FUNCTION DOC] HELPER: Accepts a dict or list union and returns the string foo.
            return "foo"

        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))
        self.checkScript(fn, ([1, 2, 3],))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"Dict\[str, str\]",
        ):
            scripted({"foo": "bar", "baz": "qux"})

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            "str",
        ):
            scripted("1")

    def test_union_with_enum(self):
        # [FUNCTION DOC] Checks a union of str and an enum: both an enum member and a string
        # pass checkScript, and calling the scripted function with an int raises RuntimeError
        # naming the union (qualified with this module name, test_union_pep604) and the found
        # type int. The enum is made global for resolution.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with members RED and GREEN used as a union member.
            RED = 1
            GREEN = 2

        make_global(Color)

        def fn(x: str | Color) -> str:
            # [FUNCTION DOC] HELPER: Accepts a str or Color and returns the string foo.
            return "foo"

        self.checkScript(fn, (Color.RED,))
        self.checkScript(fn, ("red",))

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[__torch__.jit.test_union_pep604."
            r"Color, str\] but instead found "
            "type int",
        ):
            scripted(1)

    def test_union_in_class_constructor(self):
        # [FUNCTION DOC] Checks a scripted class whose constructor takes a union argument: the
        # eager-callable wrapper function builds instances from a str and an int and the
        # stored values are asserted. Scripting the wrapper and passing a list of strings
        # raises RuntimeError about the expected union members.
        @torch.jit.script
        class A:
            # [CLASS DOC] HELPER: Scripted class storing its union-typed constructor argument
            # in attribute x.
            def __init__(self, x: int | str) -> None:
                # [FUNCTION DOC] HELPER: Stores the union argument in attribute x.
                self.x = x

        def fn(x: str | int) -> A:
            # [FUNCTION DOC] HELPER: Builds an instance of the scripted class from the union
            # argument and returns it.
            return A(x)

        self.assertEqual(fn("foo").x, "foo")
        self.assertEqual(fn(1).x, 1)

        scripted = torch.jit.script(fn)

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[int, str\] but instead "
            r"found type List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

    def test_union_return_type(self):
        # [FUNCTION DOC] Checks a function returning a union of int and str (always a string)
        # via checkScript.
        def fn(x: int) -> int | str:
            # [FUNCTION DOC] HELPER: Returns the string foo, annotated as returning an int or
            # str.
            return "foo"

        self.checkScript(fn, (1,))

    def test_union_as_annotation(self):
        # [FUNCTION DOC] Checks that a local variable annotated as a union of int and str can
        # be assigned a string and returned, via checkScript.
        def fn() -> int | str:
            # [FUNCTION DOC] HELPER: Assigns foo to a union-annotated local and returns it.
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_as_annotation_in_typed_container(self):
        # [FUNCTION DOC] Checks that union-annotated locals can be appended to a list whose
        # element type is a union, via checkScript.
        def fn() -> None:
            # [FUNCTION DOC] HELPER: Creates an empty list of unions and appends a string and
            # an int stored in union locals; returns nothing.
            l: List[int | str] = []
            u1: int | str = "foo"
            u2: int | str = 1
            l.append(u1)
            l.append(u2)

        self.checkScript(fn, ())

    def test_union_as_annotation_py2(self):
        # [FUNCTION DOC] Checks the same union return annotation supplied through a type
        # comment, with a union-annotated local, via checkScript.
        # [FUNCTION DOC] HELPER: Returns a union-annotated local holding foo. Block placed
        # above the definition so the type comment stays the first body line.
        def fn():
            # type: () -> int | str
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    def test_union_as_internal_tuple_type(self):
        # [FUNCTION DOC] Checks that a tuple type annotation whose elements are unions can
        # hold an int and a string, via checkScript.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a tuple annotated as a pair of unions holding an
            # int and a string.
            t: Tuple[int | str, int | str] = (1, "foo")
            return t

        self.checkScript(fn, ())

    def test_union_variable_can_be_reassigned(self):
        # [FUNCTION DOC] Checks that a union-annotated local can be reassigned between member
        # types (str, then int, then str) when each use is type-refined by passing it to
        # helper functions, via checkScript.
        @torch.jit.script
        def aux1(i: int):
            # [FUNCTION DOC] HELPER: Scripted helper returning the square of an int as an int.
            return int(i**2)

        @torch.jit.script
        def aux2(s: str):
            # [FUNCTION DOC] HELPER: Scripted helper returning the string concatenated with
            # itself.
            return s + s

        def fn() -> int | str:
            # [FUNCTION DOC] HELPER: Assigns a string, then an int, then a string to a union
            # local, calling the helpers on converted values, and returns the final value.
            x: int | str = "foo"
            i: int = 1
            x = i
            y: int = aux1(x)
            z: str = aux2(str(y))
            x = z
            return x

        self.checkScript(fn, ())

    def test_union_does_not_replace_existing_annotated_type(self):
        # [FUNCTION DOC] Checks that appending a str to a list annotated as a list of int is
        # not widened to a union: scripting (or running) raises RuntimeError saying it could
        # not match type str.
        def fn():
            # [FUNCTION DOC] HELPER: Builds an annotated int list and appends a string;
            # expected to fail.
            x: List[int] = [1, 2, 3]
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_union_does_not_replace_existing_annotated_type_union(self):
        # [FUNCTION DOC] Checks that appending a float to a list annotated as a list of union
        # of int and str raises RuntimeError saying it could not match type float.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a list of int-or-str values and appends a float;
            # expected to fail.
            x: List[int | str] = [1, "foo", 3]
            x.append(2.0)
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type float"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_union_does_not_replace_existing_annotated_type_empty_container(self):
        # [FUNCTION DOC] Checks that appending a str to an empty list annotated as a list of
        # int raises RuntimeError saying it could not match type str.
        def fn():
            # [FUNCTION DOC] HELPER: Creates an empty int list and appends a string; expected
            # to fail.
            x: List[int] = []
            x.append("foo")
            return x

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)
            scripted()

    def test_unions_of_unions_are_flattened(self):
        # [FUNCTION DOC] Checks that a nested union annotation is flattened: the compiled
        # graph shows the argument typed as a single union of float, int and str.
        @torch.jit.script
        def fn(x: (int | str) | float) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a nested union argument; returns
            # the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, str)").run(s)

    def test_unions_of_a_single_argument_vanish(self):
        # [FUNCTION DOC] Checks that a union with a single member collapses to that member:
        # the compiled graph shows the argument typed as int.
        @torch.jit.script
        def fn(x: Union[int]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a single-member union argument;
            # returns the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : int").run(s)

    def test_union_redundant_arguments_are_skipped(self):
        # [FUNCTION DOC] Checks that a repeated member in a union annotation is dropped: the
        # graph shows the argument typed as a union of int and str.
        @torch.jit.script
        def fn(x: int | str | int) -> str:
            # [FUNCTION DOC] HELPER: Scripted function whose argument repeats int in the
            # union; returns the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(int, str)").run(s)

    def test_union_redundant_arguments_are_skipped_optional(self):
        # [FUNCTION DOC] Checks that optional members inside a union are merged with their
        # plain forms and None: the graph shows a union of float, int and NoneType.
        @torch.jit.script
        def fn(x: int | Optional[float] | Optional[int]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a union of int and two optional
            # types; returns the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, NoneType)").run(s)

    def test_union_redundant_arguments_are_skipped_subtyping(self):
        # [FUNCTION DOC] Checks that a tuple member that is a subtype-equivalent duplicate is
        # removed from the union: the graph shows a union of a tuple of optional int and int,
        # and str.
        @torch.jit.script
        def fn(x: str | Tuple[Optional[int], int] | Tuple[int, int]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a union of str and two tuple
            # types; returns the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union((int?, int), str)").run(s)

    def test_union_redundant_arguments_are_skipped_container(self):
        # [FUNCTION DOC] Checks that duplicate list members are removed: the graph shows a
        # union of float list and str list.
        @torch.jit.script
        def fn(x: List[str] | List[float] | List[str]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a union of list types with a
            # repeat; returns the string foo.
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float[], str[])").run(s)

    def test_union_argument_order_is_ignored(self):
        # [FUNCTION DOC] Checks that member order does not matter: both orderings of a union
        # of int and str produce graphs showing the same canonical union text.
        @torch.jit.script
        def fn1(x: int | str) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with the argument union listed as int
            # then str; returns the string foo.
            return "foo"

        @torch.jit.script
        def fn2(x: str | int) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with the argument union listed as str
            # then int; returns the string foo.
            return "foo"

        for s in (fn1.graph, fn2.graph):
            FileCheck().check("x : Union(int, str)").run(s)

    def test_union_argument_order_is_ignored_container(self):
        # [FUNCTION DOC] Checks member order is ignored for unions of list types: both
        # orderings print the same canonical union of int list and str list.
        @torch.jit.script
        def fn1(x: List[str] | List[int]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a union of str list then int list;
            # returns the string foo.
            return "foo"

        @torch.jit.script
        def fn2(x: List[int] | List[str]) -> str:
            # [FUNCTION DOC] HELPER: Scripted function with a union of int list then str list;
            # returns the string foo.
            return "foo"

        for s in (fn1.graph, fn2.graph):
            FileCheck().check("x : Union(int[], str[])").run(s)

    def test_union_T_None_is_equivalent_to_optional_T(self):
        # [FUNCTION DOC] Checks that a union of int and None is interchangeable with an
        # optional int: helper functions taking either form are called with values declared in
        # the other form, and each combined function returns ten.
        @torch.jit.script
        def inner(x: int | None) -> int:
            # [FUNCTION DOC] HELPER: Scripted function taking an int-or-None argument; returns
            # the value, or five for None.
            if x is not None:
                return x
            else:
                return 5

        @torch.jit.script
        def fn1() -> int:
            # [FUNCTION DOC] HELPER: Scripted function passing an optional int five and an
            # optional None to the first helper and returning the sum.
            a: Optional[int] = 5
            b: Optional[int] = None
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        self.assertEqual(fn1(), 10)

        @torch.jit.script
        def inner2(x: Optional[int]) -> int:
            # [FUNCTION DOC] HELPER: Scripted function taking an optional int; returns the
            # value, or five for None. Not called by the checks (the second caller reuses the
            # first helper).
            if x is not None:
                return x
            else:
                return 5

        @torch.jit.script
        def fn2() -> int:
            # [FUNCTION DOC] HELPER: Scripted function passing int-or-None locals (five and
            # None) to the first helper and returning the sum.
            a: int | None = 5
            b: int | None = None
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        self.assertEqual(fn2(), 10)

    @unittest.expectedFailure
    def test_union_optional_of_union_return(self):
        # [FUNCTION DOC] Expected-failure test (marked as such): scripts a function returning
        # a union spelled with None, str and int, whose body returns a local annotated as an
        # optional of a bar-union. Documents that this combination is not yet supported; the
        # test passes only while scripting fails.
        @torch.jit.script
        def fn() -> None | str | int:
            # [FUNCTION DOC] HELPER: Scripted function returning an optional-of-union local
            # assigned the string foo; compilation is expected to fail.
            y: Optional[int | str] = "foo"
            return y

    @unittest.expectedFailure
    def test_union_optional_of_union_is_flattened(self):
        # [FUNCTION DOC] Expected-failure test (marked as such): same flattening check as the
        # typing-spelling file, using the bar syntax; the final assertions are expected to
        # fail until this form is supported. Checks results for three flags, then save, reload
        # and FileCheck of the flattened union text.
        @torch.jit.script
        def fn(flag: int) -> str | int | None:
            # [FUNCTION DOC] HELPER: Scripted function that assigns a union local and then,
            # depending on the flag, an optional-of-union local to a variable, the other union
            # value or None; returns it.
            y: int | str | None = "foo"
            if flag == 0:
                x: Optional[int | str] = y
            elif flag == 1:
                x: Optional[int | str] = 1
            else:
                x: Optional[int | str] = None
            return x

        # Can't use `checkScript` because it will flag the fact that
        # the original code has `Optional[Union[int, str]]` but the
        # saved/loaded code has `Union[int, NoneType, str]` (even
        # though this is exactly what we want)
        self.assertEqual(fn(0), "foo")
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(2), None)

        buffer = io.BytesIO()
        torch.jit.save(fn, buffer)
        buffer = io.BytesIO(buffer.getvalue())
        l = torch.jit.load(buffer)

        s = l.code

        FileCheck().check("Union[int, NoneType, str]").check(
            "Union[int, NoneType, str]"
        ).run(s)

    def test_union_subclasses_larger_union(self):
        # [FUNCTION DOC] Checks that a value of a smaller union can be returned where a larger
        # union (adding Tensor) is declared, via checkScript.
        def fn() -> int | str | torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns a union-annotated local holding foo with a return
            # type that also admits a tensor.
            x: int | str = "foo"
            return x

        self.checkScript(fn, ())

    # TODO: We would like to eventually support this. The issue is being
    # tracked at https://github.com/pytorch/pytorch/issues/58167
    def test_union_as_dict_key(self):
        # [FUNCTION DOC] Checks that a dict whose key type is a union is currently rejected:
        # scripting raises RuntimeError saying only int, float, complex, Tensor, device and
        # string keys are supported. Marked as a future feature.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a dict with union-typed keys and inserts a str and
            # an int key; expected to fail compiling.
            x: Dict[int | str, str] = {}
            x["foo"] = "bar"
            x[1] = 2
            return x[1]

        with self.assertRaisesRegex(
            RuntimeError,
            "only int, float, complex, Tensor, device and string keys are supported",
        ):
            torch.jit.script(fn)

    def test_union_as_dict_value(self):
        # [FUNCTION DOC] Checks that a dict with union-typed values can store a str and an int
        # and be read back, via checkScript.
        def fn():
            # [FUNCTION DOC] HELPER: Fills a dict with str and int values and returns one
            # value.
            x: Dict[str, int | str] = {}
            x["foo"] = "bar"
            x["baz"] = 2
            return x["baz"]

        self.checkScript(fn, ())

    def test_union_module_with_union_instance_variable(self):
        # [FUNCTION DOC] Checks a module with a union attribute declared on the class and set
        # in the constructor, reassigned in forward, via checkModule for an int and a str
        # instance.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a union-typed attribute x initialized from the
            # constructor; forward stores its union argument in x and returns it.
            x: int | str

            def __init__(self, x: int | str):
                # [FUNCTION DOC] HELPER: Calls the parent constructor and stores the union
                # argument in attribute x.
                super().__init__()
                self.x: int | str = x

            def forward(self, y: int | str):
                # [FUNCTION DOC] HELPER: Assigns the argument to attribute x and returns it.
                self.x = y
                return self.x

        self.checkModule(
            M(
                2,
            ),
            (1,),
        )
        self.checkModule(M("bar"), ("foo",))

    def test_union_module_with_union_class_variable(self):
        # [FUNCTION DOC] Checks a module with a class-level union attribute and a default
        # string, where the constructor and forward use plain local variables named like the
        # attribute, via checkModule.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a class-level union attribute x defaulting to a
            # string; its methods only use local variables that shadow x.
            x: int | str = "foo"

            def __init__(self, y: int):
                # [FUNCTION DOC] HELPER: Calls the parent constructor and assigns the argument
                # to a local variable (no attribute is set).
                super().__init__()
                x = y

            def forward(self, z: str):
                # [FUNCTION DOC] HELPER: Assigns the argument to a local variable and returns
                # it.
                x = z
                return x

        self.checkModule(M(1), ("foo",))

    def test_union_type_refinement(self):
        # [FUNCTION DOC] Checks isinstance refinement of a union: the str branch can use
        # string operations. Run with a str and an int via checkScript.
        def fn(x: int | str) -> str:
            # [FUNCTION DOC] HELPER: Returns the argument if it is a str (after using it as a
            # string), otherwise returns baz.
            if isinstance(x, str):
                z = x + "bar"
                return x
            else:
                return "baz"

        self.checkScript(fn, ("foo",))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_union_rhs(self):
        # [FUNCTION DOC] Checks that torch.jit.isinstance with a union as right-hand side is
        # statically true for an int argument, via checkScript.
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Returns bar if the int argument is an instance of the
            # union of int and str, else baz.
            if torch.jit.isinstance(x, int | str):
                return "bar"
            else:
                return "baz"

        self.checkScript(fn, (1,))

    def test_union_type_refinement_tuple_rhs(self):
        # [FUNCTION DOC] Checks isinstance with a tuple of types on a union of int, float and
        # str list, with nested refinement; runs int, float and list inputs through
        # checkScript.
        def fn(x: int | float | List[str]) -> str:
            # [FUNCTION DOC] HELPER: Returns the integer as a string, foo for floats, and the
            # first element of the list or bar when empty.
            if isinstance(x, (int, float)):
                if isinstance(x, int):
                    return str(x)
                else:
                    return "foo"
            else:
                if len(x):
                    return x[0]
                else:
                    return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))
        self.checkScript(fn, (["a", "b", "c"],))

    def test_union_type_refinement_tuple_rhs_noncontained_type(self):
        # [FUNCTION DOC] Checks isinstance with a tuple containing a type (float) that is not
        # a member of the union; refinement still works. Runs int and list inputs via
        # checkScript.
        def fn(x: int | List[str]) -> str:
            # [FUNCTION DOC] HELPER: Doubles and stringifies the argument for int or float,
            # otherwise returns the first list element or bar.
            if isinstance(x, (int, float)):
                y = x + x
                return str(y)
            else:
                if len(x):
                    return x[0]
                else:
                    return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (["a", "b", "c"],))

    def test_union_type_refinement_tuple_rhs_union(self):
        # [FUNCTION DOC] Checks torch.jit.isinstance with a tuple containing a union on an int
        # argument and that the scripted function returns the string 2. Uses assertEqual
        # instead of checkScript due to a known bug with tuple literals.
        @torch.jit.script
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Scripted function returning the doubled int as a string
            # when the type test matches, else foo.
            if torch.jit.isinstance(x, (int | str, float)):
                y = x + x
                return str(y)
            else:
                return "foo"

        # TODO: There's currently an unrelated bug in
        # `torch.jit.isinstance` that makes it fail for tuple literals.
        # Posted here: https://github.com/pytorch/pytorch/issues/60095
        # Change `assertEqual` to `checkScript` when the bug is fixed
        self.assertEqual(fn(1), "2")

    def test_union_type_refinement_statically_false(self):
        # [FUNCTION DOC] Checks that a type test known to be false at compile time (int
        # against str, list, float union) is folded away: the graph contains no conditional
        # blocks.
        @torch.jit.script
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Scripted function that would add a string to an int in
            # the dead branch and otherwise returns bar.
            if torch.jit.isinstance(x, (str | float, List[str], str)):
                z = x + "foo"
                return z
            else:
                return "bar"

        s = fn.graph

        # Check that we don't have any branching statements
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_statically_true(self):
        # [FUNCTION DOC] Checks that a negated type test known at compile time to be false for
        # a union of list and int is folded away: the graph contains no conditional blocks.
        @torch.jit.script
        def fn(x: List[int] | int) -> List[int] | int:
            # [FUNCTION DOC] HELPER: Scripted function returning the argument in the dead
            # branch, otherwise building a list and returning it through a union local.
            if not torch.jit.isinstance(x, (int, List[int])):
                return x
            else:
                l = [1, 2, 3]
                y: List[int] | int = l
                return y

        s = fn.graph

        # Check that we don't have any branching statements
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_partial_static_refinement_tuple_rhs(self):
        # [FUNCTION DOC] Checks partial static refinement with a tuple of types where only int
        # overlaps the union: inside the branch the argument is known to be an int. Runs list
        # and int inputs via checkScript.
        def fn(x: List[int] | int) -> int:
            # [FUNCTION DOC] HELPER: Returns the argument plus one when it is an int (in the
            # type-tested branch), else one hundred.
            if torch.jit.isinstance(x, (int, float, str)):
                # We should know that `x` is an `int` here
                z = x + 1
                return z
            else:
                return 100

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_partial_static_refinement_union_rhs(self):
        # [FUNCTION DOC] Same as the tuple case but with a union as the right-hand side of the
        # type test; runs list and int inputs via checkScript.
        def fn(x: List[int] | int) -> int:
            # [FUNCTION DOC] HELPER: Returns the argument plus one when it is an int (in the
            # type-tested branch), else one hundred.
            if torch.jit.isinstance(x, int | float | str):
                # We should know that `x` is an `int` here
                z = x + 1
                return z
            else:
                return 100

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_internal_declaration(self):
        # [FUNCTION DOC] Checks that a local declared with a union annotation holding None can
        # be tested with isinstance even though branches assign different types to another
        # variable. Runs True and False via checkScript.
        def fn(flag: bool) -> str:
            # [FUNCTION DOC] HELPER: Declares a union local set to None, assigns an unused
            # variable different types per branch, and returns the local if it is a str, else
            # bar.
            x: int | str | None = None
            if flag:
                y = "foo"
            else:
                y = 1
            if isinstance(x, str):
                return x
            else:
                return "bar"

        self.checkScript(fn, (True,))
        self.checkScript(fn, (False,))

    def test_union_branching_with_union_return_and_homogenous_types(self):
        # [FUNCTION DOC] Checks branches returning different strings from a function declared
        # to return a union, via checkScript with an odd and an even input.
        def fn(x: int) -> int | str:
            # [FUNCTION DOC] HELPER: Returns foo for odd input and bar for even input.
            if x % 2:
                return "foo"
            else:
                return "bar"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_branching_does_not_autoinfer_undeclared_union(self):
        # [FUNCTION DOC] Checks that a variable assigned a str in one branch and an int in the
        # other is not inferred as a union: scripting raises RuntimeError naming the two
        # branch types.
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Assigns a string or the int input to a variable by branch
            # and tests it; expected to fail compiling.
            if x % 2:
                y = "foo"
            else:
                y = x
            if isinstance(y, str):
                return y
            else:
                return "bar"

        with self.assertRaisesRegex(
            RuntimeError,
            "y is set to type str in the true branch and type int in the false branch",
        ):
            torch.jit.script(fn)

    def test_union_branching_does_not_widen_existing_inferred_type(self):
        # [FUNCTION DOC] Checks that reassigning a variable that was inferred as str to an int
        # is an error: scripting raises RuntimeError saying the variable previously had type
        # str.
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Assigns a string, then a string or the int input
            # depending on a branch; expected to fail compiling.
            y = "foo"
            if x % 2:
                y = "bar"
            else:
                y = x
            if isinstance(y, str):
                return y
            else:
                return "baz"

        with self.assertRaisesRegex(
            RuntimeError,
            "previously had type str but is now being assigned to a value of type int",
        ):
            torch.jit.script(fn)

    def test_union_schema_matching_on_internal_type(self):
        # [FUNCTION DOC] Checks that after refining a union of list and dict to the list type,
        # indexing works, and that the dict branch can use dict methods. Runs a list and a
        # dict via checkScript.
        def fn(x: List[int] | Dict[str, int]) -> int:
            # [FUNCTION DOC] HELPER: Returns the first list element, or the first dict value
            # for a dict.
            if torch.jit.isinstance(x, List[int]):
                return x[0]
            else:
                return list(x.values())[0]

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))

    def test_union_subtractive_refinement(self):
        # [FUNCTION DOC] Checks that the else-side of a negated int test refines a union of
        # list and int to the list, allowing list operations. Runs an int and a list via
        # checkScript.
        def fn(x: List[int] | int) -> int:
            # [FUNCTION DOC] HELPER: For non-int input appends one and returns the first
            # element; for an int returns it.
            if not isinstance(x, int):
                x.append(1)
                return x[0]
            else:
                return x

        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    def test_union_subtractive_refinement_with_container(self):
        # [FUNCTION DOC] Same subtractive refinement using torch.jit.isinstance with a list
        # type; runs an int and a list via checkScript.
        def fn(x: List[int] | int) -> int:
            # [FUNCTION DOC] HELPER: Returns an int unchanged; for a list appends one and
            # returns the first element.
            if not torch.jit.isinstance(x, List[int]):
                return x
            else:
                x.append(1)
                return x[0]

        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    def test_union_memory_aliasing(self):
        # [FUNCTION DOC] Checks aliasing is preserved through a union-typed (optional list)
        # container element: appending via the alias after a type test is visible in the
        # original list. Runs via checkScript.
        def fn():
            # [FUNCTION DOC] HELPER: Stores a tensor list in a list of optional tensor lists,
            # retrieves it, appends a tensor when the type test passes, and returns the
            # original list.
            x: List[torch.Tensor] = []
            z: List[Optional[List[torch.Tensor]]] = []
            z.append(x)
            x_alias = z[0]
            if torch.jit.isinstance(x_alias, List[torch.Tensor]):
                x_alias.append(torch.tensor(3))
            return x

        self.checkScript(fn, ())

    def test_union_serialization_preserves_type_annotations(self):
        # [FUNCTION DOC] Checks via checkScript (which includes save and load) that union
        # annotations on locals survive serialization; without them the variable would be
        # typed differently per branch.
        # This function will fail after being torch.jit.save'd and
        # torch.jit.load'd if the type annotations aren't preserved
        # for Union during serialization. We need the `Union[str, int]`
        # annotation to make sure that `y` is typed as a Union instead
        # of as a str in one branch and an int in the other
        def fn(x: int) -> str:
            # [FUNCTION DOC] HELPER: Assigns a str or the int input to a union-annotated local
            # by branch, and returns it if it is a str, else baz.
            if x % 2:
                y: str | int = "bar"
            else:
                y: str | int = x
            if isinstance(y, str):
                return y
            else:
                return "baz"

        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def _assert_passes(self, template: str, ann: str, lhs: str):
        # [FUNCTION DOC] HELPER: Formats the template with the annotation and the
        # right-hand-side snippet and checks the resulting function source compiles and
        # behaves identically in eager and script via checkScript with no inputs, under the
        # name fn.
        code = template.format(ann=ann, lhs=lhs)
        self.checkScript(code, (), name="fn")

    def _assert_raises(self, template: str, ann: str, lhs: str, msg: str):
        # [FUNCTION DOC] HELPER: Formats the template with the annotation and right-hand side,
        # then builds a compilation unit from it, expecting RuntimeError whose message matches
        # the given regex, then fetches fn from the unit.
        code = template.format(ann=ann, lhs=lhs)
        with self.assertRaisesRegex(RuntimeError, msg):
            cu = torch.jit.CompilationUnit(code, _frames_up=1)
            string_frontend = getattr(cu, "fn")  # noqa: B009

    def test_union_with_list_assignment(self):
        # [FUNCTION DOC] Checks assigning list literals and list comprehensions (empty,
        # tensors, strings, mixed) to locals annotated with various unions: ambiguous list
        # candidates, unions without lists, and unions with a single list type. Uses a source
        # template and two helpers to check which combinations compile and which raise the
        # expected messages.
        template = dedent(
            """
            def fn():
                x: {ann} = {lhs}
                if torch.jit.isinstance(x, List[torch.Tensor]):
                    x.append(torch.tensor(3))
                return x
        """
        )

        lhs = {
            "list_literal_empty": "[]",
            "list_literal_of_tensor": "[torch.arange(3), torch.arange(5)]",
            "list_literal_of_str": '["foo", "bar", "baz"]',
            "list_literal_of_mixed": "[torch.arange(5), 1]",
            "list_comprehension_of_tensor": "[torch.add(x, 1) for x in [torch.arange(3), torch.arange(5)]]",
            "list_comprehension_of_str": '[x + "!" for x in ["foo", "bar", "baz"]]',
            "list_comprehension_of_mixed": "[torch.add(1, x) for x in [torch.arange(5), 1]]",
        }

        """
        List[str] | List[torch.Tensor]
        """
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_literal_empty"],
            "there are multiple possible List type candidates in the Union annotation",
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_literal_of_tensor"]
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_literal_of_str"]
        )

        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_literal_of_mixed"],
            "none of those types match the types of the given list elements",
        )

        self._assert_passes(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_comprehension_of_tensor"],
        )

        self._assert_passes(
            template, "List[str] | List[torch.Tensor]", lhs["list_comprehension_of_str"]
        )

        # TODO: Support mixed list comprehensions
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["list_comprehension_of_mixed"],
            "Arguments for call are not valid",
        )

        """
        int | torch.Tensor
        """
        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_literal_empty"],
            "Expected a Union type annotation with an inner List type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_literal_of_tensor"],
            "Expected a Union type annotation with an inner List type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["list_comprehension_of_tensor"],
            "Expected a Union type annotation with an inner List type",
        )

        """
        List[torch.Tensor] | int
        """
        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_literal_empty"]
        )

        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_literal_of_tensor"]
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_literal_of_str"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_literal_of_mixed"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        self._assert_passes(
            template, "List[torch.Tensor] | int", lhs["list_comprehension_of_tensor"]
        )

        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_comprehension_of_str"],
            r"List type annotation `List\[Tensor\]` did "
            "not match the types of the given list "
            "elements",
        )

        # TODO(@ansley): Support mixed list comprehensions
        self._assert_raises(
            template,
            "List[torch.Tensor] | int",
            lhs["list_comprehension_of_mixed"],
            "Arguments for call are not valid",
        )

    def test_union_with_dict_assignment(self):
        # [FUNCTION DOC] Same idea for dict literals, dict comprehensions (disabled because
        # the string frontend lacks tuple unpacking) and dict keyword construction assigned to
        # union-annotated locals: ambiguous dict types, unions without dicts, and unions with
        # a single dict type. Uses a source template and two helpers.
        # KNOWN ISSUE: the first check of the union of two dict types passes a union of lists
        # as the annotation while expecting the inner-Dict message, so it does not test the
        # dict union it is commented as.
        # KNOWN ISSUE: the dict comprehension snippet for string to int and the
        # aggregate-function dict keyword snippet are each missing a closing parenthesis, but
        # they are only used by commented-out checks.
        template = dedent(
            """
            def fn():
                x: {ann} = {lhs}
                if torch.jit.isinstance(x, Dict[str, torch.Tensor]):
                    x["foo"] = torch.tensor(3)
                return x
        """
        )

        lhs = {
            "dict_literal_empty": "{}",
            "dict_literal_of_str_tensor": '{"foo" : torch.arange(3), "bar" : torch.arange(5)}',
            "dict_literal_of_str_int": '{"foo" : 1, "bar" : 2}',
            "dict_literal_of_mixed": '{"foo" : torch.arange(3), "bar" : 2}',
            "dict_comprehension_of_str_tensor": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [torch.arange(3), torch.arange(5)])}',
            "dict_comprehension_of_str_int": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [1, 2]}',
            "dict_comprehension_of_mixed": '{x : torch.add(y, 1) for x, y in \
                    zip(["foo", "bar"], [torch.arange(3), 2])}',
            "dict_keyword": "dict(foo=torch.arange(3), baz=torch.arange(5))",
            "dict_keyword_with_iterable": 'dict([("foo", torch.arange(3)), ("bar", torch.arange(5))])',
            "dict_keyword_with_empty_iterable": "dict([])",
            "dict_keyword_with_internal_aggregate_function": 'dict(zip(["foo", "bar"], [torch.arange(3), torch.arange(5)])',
            "dict_keyword_with_mapping": 'dict({"foo" : torch.arange(3), "bar" : torch.arange(5)})',
            "dict_keyword_with_mapping_and_kwargs": 'dict({"foo" : torch.arange(3), "bar" : torch.arange(5)}, baz=torch.arange(7))',
        }

        """
        Dict[str, torch.Tensor] | Dict[str, int]
        """
        self._assert_raises(
            template,
            "List[str] | List[torch.Tensor]",
            lhs["dict_literal_empty"],
            "Expected a Union type annotation with an inner Dict type",
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_str_tensor"],
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_str_int"],
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_literal_of_mixed"],
            "none of those dict types can hold the types of the given keys and values",
        )

        # TODO: String frontend does not support tuple unpacking
        # https://github.com/pytorch/pytorch/issues/64096
        # self._assert_passes(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_str_tensor"])

        # self._assert_passes(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_str_int"])

        # self._assert_raises(template, "Dict[str, torch.Tensor] | Dict[str, int]",
        #              lhs["dict_comprehension_of_mixed"],
        #              "foobar")

        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | Dict[str, int]",
        #                    lhs["dict_keyword_with_internal_aggregate_function"])

        # TODO(@ansley): Follow-up project needed for full type
        # inference with dict keyword (supported for dict comprehension
        # and dict literal already; should not be a blocker for anyone)
        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_iterable"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_empty_iterable"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_mapping"],
            "full type inference is not yet supported",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | Dict[str, int]",
            lhs["dict_keyword_with_mapping_and_kwargs"],
            "full type inference is not yet supported",
        )

        """
        int | torch.Tensor
        """
        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["dict_literal_empty"],
            "Expected a Union type annotation with an inner Dict type",
        )

        self._assert_raises(
            template,
            "int | torch.Tensor",
            lhs["dict_literal_of_str_tensor"],
            "Expected a Union type annotation with an inner Dict type",
        )

        # See above--string frontend does not support tuple unpacking
        # self._assert_raises(template, "int | torch.Tensor",
        #              lhs["dict_comprehension_of_tensor"],
        #              "foobar")

        """
        Dict[str, torch.Tensor] | int
        """
        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_literal_empty"]
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_literal_of_str_tensor"]
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_literal_of_str_int"],
            "Type annotation was inferred to be "
            r"`Dict\[str, Tensor\]`, but the type of "
            "values given by the dict literal is",
        )

        self._assert_raises(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_literal_of_mixed"],
            "Type annotation was inferred to be "
            r"`Dict\[str, Tensor\]`, but the type of "
            "values given by the dict literal is",
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword"]
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword_with_iterable"]
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_keyword_with_empty_iterable"],
        )

        self._assert_passes(
            template, "Dict[str, torch.Tensor] | int", lhs["dict_keyword_with_mapping"]
        )

        self._assert_passes(
            template,
            "Dict[str, torch.Tensor] | int",
            lhs["dict_keyword_with_mapping_and_kwargs"],
        )

        # See above--string frontend does not support tuple unpacking
        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_keyword_with_internal_aggregate_function"])
        #
        # self._assert_passes(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_str_tensor"])

        # self._assert_raises(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_str_int"],
        #                    "foobar")

        # self._assert_raises(template,
        #                    "Dict[str, torch.Tensor] | int",
        #                    lhs["dict_comprehension_of_mixed"],
        #                    "foobar")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
