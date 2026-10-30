# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import inspect
import os
import sys
from collections import namedtuple
from textwrap import dedent
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.testing._internal.jit_utils
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestTypesAndAnnotation(JitTestCase):
    # [CLASS DOC] Test case for TorchScript handling of type annotations and types used as
    # values: PEP 585 builtin generics, ignored functions with typed signatures, bad or bare
    # Optional/Tuple annotations, ignored module attributes, annotation placement rules, and
    # InferredType errors. Holds no state; each test builds its own functions and modules.
    def test_pep585_type(self):
        # [FUNCTION DOC] Checks that lowercase builtin generics (list, tuple, dict) are
        # accepted in annotations, both in the signature and for local variables. Two nested
        # functions (one with typing-style and one with builtin-style return types) append to
        # a list, set a dict key and return the popped element plus the dict. Each is run
        # through checkScript and also compared between eager and scripted calls with
        # assertEqual.
        def fn(x: torch.Tensor) -> Tuple[Tuple[torch.Tensor], Dict[str, int]]:
            # [FUNCTION DOC] HELPER: Scripted body using typing-style signature types and
            # builtin generic local annotations; appends a one-tuple holding x to a list,
            # stores one in a dict, returns the popped tuple and the dict.
            xl: list[tuple[torch.Tensor]] = []
            xd: dict[str, int] = {}
            xl.append((x,))
            xd["foo"] = 1
            return xl.pop(), xd

        self.checkScript(fn, [torch.randn(2, 2)])

        x = torch.randn(2, 2)
        expected = fn(x)
        scripted = torch.jit.script(fn)(x)

        self.assertEqual(expected, scripted)

        def fn2(x: torch.Tensor) -> tuple[tuple[torch.Tensor], dict[str, int]]:
            # [FUNCTION DOC] HELPER: Same as the first nested function but the return type
            # also uses builtin generics; returns the popped tuple and the dict.
            xl: list[tuple[torch.Tensor]] = []
            xd: dict[str, int] = {}
            xl.append((x,))
            xd["foo"] = 1
            return xl.pop(), xd

        self.checkScript(fn2, [torch.randn(2, 2)])

        expected = fn2(x)
        scripted = torch.jit.script(fn2)(x)

        self.assertEqual(expected, scripted)

    def test_types_as_values(self):
        # [FUNCTION DOC] Checks that types can appear as values and in signatures: scripts a
        # function returning a tensor's device, then scripts two modules whose ignored method
        # returns a namedtuple pair (one annotated with a tuple of two namedtuples, the other
        # annotated that way while returning a single namedtuple). Only checks that scripting
        # and calling succeed; results are not asserted.
        def fn(m: torch.Tensor) -> torch.device:
            # [FUNCTION DOC] HELPER: Returns the device of the input tensor.
            return m.device

        self.checkScript(fn, [torch.randn(2, 2)])

        GG = namedtuple("GG", ["f", "g"])

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose ignored method foo builds a pair of GG
            # namedtuples from its two tensor inputs; forward delegates to it.
            @torch.jit.ignore
            def foo(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[GG, GG]:
                # [FUNCTION DOC] HELPER: Ignored method returning a tuple of two GG
                # namedtuples built from x and z.
                return GG(x, z), GG(x, z)

            def forward(self, x, z):
                # [FUNCTION DOC] HELPER: Forward delegating to the ignored method foo.
                return self.foo(x, z)

        foo = torch.jit.script(Foo())
        y = foo(torch.randn(2, 2), torch.randn(2, 2))

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Same shape as the previous module but the ignored method foo
            # has untyped arguments and returns a single GG, although annotated as returning a
            # pair; forward delegates to it.
            @torch.jit.ignore
            def foo(self, x, z) -> Tuple[GG, GG]:
                # [FUNCTION DOC] HELPER: Ignored method that returns one GG built from x and z
                # despite its pair return annotation (never checked since ignored methods run
                # in Python).
                return GG(x, z)

            def forward(self, x, z):
                # [FUNCTION DOC] HELPER: Forward delegating to the ignored method foo.
                return self.foo(x, z)

        foo = torch.jit.script(Foo())
        y = foo(torch.randn(2, 2), torch.randn(2, 2))

    def test_ignore_with_types(self):
        # [FUNCTION DOC] Checks that ignored functions and methods with Optional-in-Dict
        # annotated parameters can be called from a scripted module. Scripts a module and
        # verifies via FileCheck that the printed graph mentions dropout_modality and
        # in_batch.
        @torch.jit.ignore
        def fn(x: Dict[str, Optional[torch.Tensor]]):
            # [FUNCTION DOC] HELPER: Ignored free function; adds 10 to its dict argument
            # (would fail if actually run; only compiled as a call target).
            return x + 10

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a forward taking a dict of optional tensors,
            # which calls an ignored method and an ignored function and returns a tensor of
            # one.
            def forward(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the ignored method and ignored function on the
                # input dict, ignoring results, and returns a constant tensor holding one.
                self.dropout_modality(in_batch)
                fn(in_batch)
                return torch.tensor(1)

            @torch.jit.ignore
            def dropout_modality(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> Dict[str, Optional[torch.Tensor]]:
                # [FUNCTION DOC] HELPER: Ignored method returning the input dict unchanged.
                return in_batch

        sm = torch.jit.script(M())
        FileCheck().check("dropout_modality").check("in_batch").run(str(sm.graph))

    def test_python_callable(self):
        # [FUNCTION DOC] Checks that an instance of a plain Python class whose ignored call
        # method returns a string of the type of its first argument can be called from a
        # scripted function. Compares the scripted result with the direct call using
        # assertEqual.
        class MyPythonClass:
            # [CLASS DOC] HELPER: Plain Python class whose ignored call method returns the
            # string form of the type of its first positional argument.
            @torch.jit.ignore
            def __call__(self, *args) -> str:
                # [FUNCTION DOC] HELPER: Ignored call method; returns a string naming the type
                # of the first argument.
                return str(type(args[0]))

        the_class = MyPythonClass()

        @torch.jit.script
            # [FUNCTION DOC] HELPER: Scripted function that calls the captured class instance
            # on x and returns the result.
        def fn(x):
            return the_class(x)

        # This doesn't involve the string frontend, so don't use checkScript
        x = torch.ones(2)
        self.assertEqual(fn(x), the_class(x))

    def test_bad_types(self):
        # [FUNCTION DOC] Checks that calling an ignored function with an argument that does
        # not match its inferred parameter type fails at compile time: expects a RuntimeError
        # mentioning the argument name my_arg when scripting a function that passes a string.
        @torch.jit.ignore
        def fn(my_arg):
            # [FUNCTION DOC] HELPER: Ignored function adding 10 to its argument.
            return my_arg + 10

        with self.assertRaisesRegex(RuntimeError, "argument 'my_arg'"):

            @torch.jit.script
            def other_fn(x):
                # [FUNCTION DOC] HELPER: Scripted function (expected to fail compiling) that
                # calls the ignored function with a string.
                return fn("2")

    def test_type_annotate_py3(self):
        # [FUNCTION DOC] Checks local variable annotations in scripted code. First a function
        # using typed lists, tensors and optionals mutated in a loop passes checkScript. Then
        # a function annotating a list of int but assigned a float element must raise
        # RuntimeError with the message that the list type annotation did not match the
        # element types.
        def fn():
            # [FUNCTION DOC] HELPER: Builds annotated local variables (list of int, tensor,
            # two optional tensors), updates them in a ten-iteration loop and returns all
            # four.
            a: List[int] = []
            b: torch.Tensor = torch.ones(2, 2)
            c: Optional[torch.Tensor] = None
            d: Optional[torch.Tensor] = torch.ones(3, 4)
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
                d = None
            return a, b, c, d

        self.checkScript(fn, ())

        # [FUNCTION DOC] HELPER: Annotates a list of int but assigns a list holding a float;
        # scripting it must fail. Comment placed above the definition.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def wrong_type():
            wrong: List[int] = [0.5]
            return wrong

        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            torch.jit.script(wrong_type)

    def test_optional_no_element_type_annotation(self):
        # [FUNCTION DOC] Checks that a bare Optional annotation without a contained type is
        # rejected. Both nested functions are compiled through a compilation unit from their
        # dedented source, and through torch.jit.script; each of the four attempts must raise
        # RuntimeError with the message about using Optional without a contained type.
        """
        Test that using an optional with no contained types produces an error.
        """

        # [FUNCTION DOC] HELPER: Function (returning a pair) with a bare Optional return
        # annotation; used for scripting and source-based define. Block placed above the
        # definition because its source is passed to define.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def fn_with_comment(x: torch.Tensor) -> Optional:
            return (x, x)

        # [FUNCTION DOC] HELPER: Same as the previous function, another bare Optional return
        # annotation case. Block placed above the definition because its source is passed to
        # define.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def annotated_fn(x: torch.Tensor) -> Optional:
            return (x, x)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_tuple_no_element_type_annotation(self):
        # [FUNCTION DOC] Same as the Optional variant but for a bare Tuple return annotation:
        # four attempts (define from source and script, for both functions) must each raise
        # RuntimeError about using Tuple without a contained type.
        """
        Test that using a tuple with no contained types produces an error.
        """

        # [FUNCTION DOC] HELPER: Function with a bare Tuple return annotation. Block placed
        # above the definition because its source is passed to define.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def fn_with_comment(x: torch.Tensor) -> Tuple:
            return (x, x)

        # [FUNCTION DOC] HELPER: Second function with a bare Tuple return annotation. Block
        # placed above the definition because its source is passed to define.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def annotated_fn(x: torch.Tensor) -> Tuple:
            return (x, x)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_ignoring_module_attributes(self):
        # [FUNCTION DOC] Checks the ignored-attributes mechanism for modules. A module
        # declaring two attributes as ignored is scripted and its forward and ignored method
        # give the expected values (4 and 1). A second module whose forward uses an ignored
        # attribute must fail scripting with RuntimeError saying the attribute was ignored
        # during compilation, with the highlighted source text being the attribute access.
        """
        Test that module attributes can be ignored.
        """

        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Small submodule whose forward returns the sum of a
            # one-element list holding its integer argument.
            def forward(self, a: int) -> int:
                # [FUNCTION DOC] HELPER: Returns the sum of a list containing a.
                return sum([a])

        class ModuleWithIgnoredAttr(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that lists attributes a and sub as ignored for
            # scripting; stores two ints and a Sub submodule; forward returns b; an ignored
            # method forwards a through the submodule.
            __jit_ignored_attributes__ = ["a", "sub"]

            def __init__(self, a: int, b: int):
                # [FUNCTION DOC] HELPER: Constructor storing the two ints a and b and a Sub
                # instance.
                super().__init__()
                self.a = a
                self.b = b
                self.sub = Sub()

            def forward(self) -> int:
                # [FUNCTION DOC] HELPER: Returns the b attribute.
                return self.b

            @torch.jit.ignore
            def ignored_fn(self) -> int:
                # [FUNCTION DOC] HELPER: Ignored method calling the Sub forward with attribute
                # a and returning the integer result.
                return self.sub.forward(self.a)

        mod = ModuleWithIgnoredAttr(1, 4)
        scripted_mod = torch.jit.script(mod)
        self.assertEqual(scripted_mod(), 4)
        self.assertEqual(scripted_mod.ignored_fn(), 1)

        # Test the error message for ignored attributes.
        class ModuleUsesIgnoredAttr(torch.nn.Module):
            # [CLASS DOC] HELPER: Module listing a and sub as ignored while its forward uses
            # the ignored sub attribute, so scripting it is expected to fail.
            __jit_ignored_attributes__ = ["a", "sub"]

            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Constructor storing int a and a Sub instance.
                super().__init__()
                self.a = a
                self.sub = Sub()

            # [FUNCTION DOC] HELPER: Forward calling the ignored submodule with a nonexistent
            # attribute b. Block placed above the definition.
            # KNOWN ISSUE: block placed above the definition (not inside the body) because
            # tests assert on the compiled source text or error highlights.
            def forward(self) -> int:
                return self.sub(self.b)

        mod = ModuleUsesIgnoredAttr(1)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"attribute was ignored during compilation", "self.sub"
        ):
            scripted_mod = torch.jit.script(mod)

    def test_ignoring_fn_with_nonscriptable_types(self):
        # [FUNCTION DOC] Checks that a plain class whose dropped methods have annotations that
        # cannot be scripted (an iterator type and torch.fx types) can still be scripted.
        # Calls torch.jit.script on the class; no assertions beyond not raising.
        class CFX:
            # [CLASS DOC] HELPER: Plain class holding a list of tensors, a scriptable forward
            # computing sin, and two methods marked with the internal drop decorator whose
            # annotations are not scriptable.
            def __init__(self, a: List[torch.Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Stores the list of tensors in attribute a.
                self.a = a

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the elementwise sine of x.
                return torch.sin(x)

            @torch.jit._drop
            def __iter__(self) -> Iterator[torch.Tensor]:
                # [FUNCTION DOC] HELPER: Dropped method returning an iterator over the stored
                # list.
                return iter(self.a)

            @torch.jit._drop
            def __fx_create_arg__(
                self, tracer: torch.fx.Tracer
            ) -> torch.fx.node.Argument:
                # [FUNCTION DOC] HELPER: Dropped method creating an FX call_function node for
                # this class with the argument built from a features attribute; never
                # executed.
                # torch.fx classes are not scriptable
                return tracer.create_node(
                    "call_function",
                    CFX,
                    args=(tracer.create_arg(self.features),),
                    kwargs={},
                )

        torch.jit.script(CFX)

    def test_unimported_type_resolution(self):
        # [FUNCTION DOC] Checks that a type written only in a type comment (the number
        # pseudo-type) is resolved by falling back from the Python resolver to the C++
        # resolver. FileCheck confirms the compiled graph mentions Scalar.
        # verify fallback from the python resolver to the c++ resolver

        # [FUNCTION DOC] HELPER: Scripted function using a type comment as first body line;
        # adds one to x. Block placed above the decorator so the type comment remains the
        # first body line.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        @torch.jit.script
        def fn(x):
            # type: (number) -> number
            return x + 1

        FileCheck().check("Scalar").run(fn.graph)

    def test_parser_bug(self):
        # [FUNCTION DOC] Regression stub for a parser bug: only defines a nested function with
        # an Optional tensor parameter and an empty body; nothing is scripted or asserted.
        def parser_bug(o: Optional[torch.Tensor]):
            # [FUNCTION DOC] HELPER: Does nothing; takes an optional tensor.
            pass

    def test_mismatched_annotation(self):
        # [FUNCTION DOC] Checks that assigning an int to a local annotated as str fails
        # compilation with RuntimeError mentioning that the variable was annotated with a
        # type.
        with self.assertRaisesRegex(RuntimeError, "annotated with type"):

            @torch.jit.script
            def foo():
                # [FUNCTION DOC] HELPER: Scripted function (expected to fail) annotating a
                # local as str and assigning four.
                x: str = 4
                return x

    def test_reannotate(self):
        # [FUNCTION DOC] Checks that annotating a variable after it was already declared fails
        # compilation with RuntimeError containing declare and annotate.
        with self.assertRaisesRegex(RuntimeError, "declare and annotate"):

            @torch.jit.script
            def foo():
                # [FUNCTION DOC] HELPER: Scripted function (expected to fail) that assigns an
                # int then re-annotates the variable as optional inside a conditional.
                x = 5
                if 1 == 1:
                    x: Optional[int] = 7

    def test_annotate_outside_init(self):
        # [FUNCTION DOC] Checks that annotating instance attributes outside the constructor of
        # a scripted class raises ValueError with the message that annotations must be
        # declared in init, with the highlighted text being the annotated attribute
        # assignment. Covers an annotation in a plain method and inside a loop; a class
        # annotating in init and assigning later must script fine.
        msg = "annotations on instance attributes must be declared in __init__"
        highlight = "self.x: int"

        # Simple case
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            # [CLASS DOC] HELPER: Scripted class (expected to fail) whose set method annotates
            # an attribute outside init.
            # KNOWN ISSUE: block placed above the definition (not inside the body) because
            # tests assert on the compiled source text or error highlights.
            @torch.jit.script
            class BadModule:
                def __init__(self, x: int):
                    # [FUNCTION DOC] HELPER: Constructor storing x.
                    self.x = x

                # [FUNCTION DOC] HELPER: Method annotating the attribute assignment outside
                # init (the error trigger). Block placed above the definition.
                # KNOWN ISSUE: block placed above the definition (not inside the body) because
                # tests assert on the compiled source text or error highlights.
                def set(self, val: int):
                    self.x: int = val

        # Type annotation in a loop
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            # [CLASS DOC] HELPER: Scripted class (expected to fail) whose set method annotates
            # the attribute inside a loop.
            # KNOWN ISSUE: block placed above the definition (not inside the body) because
            # tests assert on the compiled source text or error highlights.
            @torch.jit.script
            class BadModuleLoop:
                def __init__(self, x: int):
                    # [FUNCTION DOC] HELPER: Constructor storing x.
                    self.x = x

                # [FUNCTION DOC] HELPER: Method annotating the attribute inside a
                # three-iteration loop (the error trigger). Block placed above the definition.
                # KNOWN ISSUE: block placed above the definition (not inside the body) because
                # tests assert on the compiled source text or error highlights.
                def set(self, val: int):
                    for _ in range(3):
                        self.x: int = val

        # Type annotation in __init__, should not fail
        @torch.jit.script
        class GoodModule:
            # [CLASS DOC] HELPER: Scripted class that annotates its attribute in init and
            # reassigns it in a method; must script successfully.
            def __init__(self, x: int):
                # [FUNCTION DOC] HELPER: Constructor storing annotated attribute x.
                self.x: int = x

            def set(self, val: int):
                # [FUNCTION DOC] HELPER: Reassigns the attribute x.
                self.x = val

    def test_inferred_type_error_message(self):
        # [FUNCTION DOC] Checks that asking an InferredType built with only an error reason
        # for its type raises RuntimeError twice: once with the message that the type is null
        # and once containing the original reason text.
        inferred_type = torch._C.InferredType("ErrorReason")

        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to get the type from an InferredType but the type is null.",
        ):
            t = inferred_type.type()

        with self.assertRaisesRegex(RuntimeError, "ErrorReason"):
            t = inferred_type.type()


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
