# Owner(s): ["module: dynamo"]
# [MODULE DOC]
# Tests that Dynamo reproduces Python string-conversion semantics when tracing: the built-in
# string conversion of constants, containers, exceptions, user-defined classes, metaclasses,
# functions and mappings, the explicit dunder string method (bound and unbound), the object
# base-class version that always uses the representation, and formatted-string literals that
# must observe mutations made between evaluations. Tests decorated with the dynamo-test
# helper run their body under Dynamo directly; the others compile a function with the eager
# backend and compare against eager execution. No import-time side effects beyond defining
# a small helper class and the test cases.
"""Tests for tp_str / generic_str behavior in Dynamo."""

import collections
import typing
import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class _OpaqueStrDescriptorObject:
    # [CLASS DOC] HELPER: Class whose dunder string attribute is the unbound upper-casing
    # method of the built-in string type. Converting an instance to a string in eager mode
    # raises a type error because the descriptor does not apply to this class.
    __str__ = str.upper


class TpStrTests(TestCase):
    # [CLASS DOC] Tests of string conversion for built-in values, run directly under
    # Dynamo through the dynamo-test decorator. Each test uses plain assertions inside.
    @make_dynamo_test
    def test_str_int(self):
        # [FUNCTION DOC] Checks string conversion of positive, negative and zero integers
        # yields their decimal text.
        assert str(42) == "42"  # noqa: S101
        assert str(-1) == "-1"  # noqa: S101
        assert str(0) == "0"  # noqa: S101

    @make_dynamo_test
    def test_str_float(self):
        # [FUNCTION DOC] Checks string conversion of positive, zero and negative floats
        # yields the usual shortest decimal text.
        assert str(3.14) == "3.14"  # noqa: S101
        assert str(0.0) == "0.0"  # noqa: S101
        assert str(-2.5) == "-2.5"  # noqa: S101

    @make_dynamo_test
    def test_str_bool(self):
        # [FUNCTION DOC] Checks string conversion of both boolean values gives the
        # capitalized words True and False.
        assert str(True) == "True"  # noqa: S101
        assert str(False) == "False"  # noqa: S101

    @make_dynamo_test
    def test_str_none(self):
        # [FUNCTION DOC] Checks string conversion of the none value gives the word None.
        assert str(None) == "None"  # noqa: S101

    @make_dynamo_test
    def test_str_string_identity(self):
        # [FUNCTION DOC] Checks string conversion of a non-empty and an empty string
        # returns equal strings.
        s = "hello"
        empty = ""
        assert str(s) == "hello"  # noqa: S101
        assert str(empty) == ""  # noqa: S101

    @make_dynamo_test
    def test_str_dunder_constant(self):
        # [FUNCTION DOC] Checks calling the bound dunder string method directly on an
        # integer, a float and a boolean constant gives the same text as conversion.
        assert (42).__str__() == "42"  # noqa: S101
        assert (3.14).__str__() == "3.14"  # noqa: S101
        assert True.__str__() == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_constant(self):
        # [FUNCTION DOC] Checks calling the unbound dunder string method of the int, float
        # and bool types with a matching constant gives the expected text.
        assert int.__str__(42) == "42"  # noqa: S101
        assert float.__str__(3.14) == "3.14"  # noqa: S101
        assert bool.__str__(True) == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_unbound_dunder_string(self):
        # [FUNCTION DOC] Checks the unbound dunder string method of the string type returns
        # the same text for a non-empty and an empty string.
        assert str.__str__("hello") == "hello"  # noqa: S101
        assert str.__str__("") == ""  # noqa: S101

    @make_dynamo_test
    def test_str_list_falls_back_to_repr(self):
        # [FUNCTION DOC] Checks string conversion of a list of integers equals its
        # bracketed comma-separated representation text.
        assert str([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_object_dunder_str_on_string_uses_repr(self):
        # [FUNCTION DOC] Checks the object base-class dunder string method applied to a
        # string returns its quoted representation, for non-empty and empty strings.
        assert object.__str__("hello") == "'hello'"  # noqa: S101
        assert object.__str__("") == "''"  # noqa: S101

    @make_dynamo_test
    def test_object_dunder_str_on_list_uses_repr(self):
        # [FUNCTION DOC] Checks the object base-class dunder string method applied to a
        # list returns the list representation text.
        assert object.__str__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101


class TpStrUserDefinedTests(TestCase):
    # [CLASS DOC] Tests of string conversion for user-defined and library objects. Most
    # tests compile a small function with the eager backend and full-graph mode and compare
    # its output with eager execution of the same function.
    def test_counter_str(self):
        # [FUNCTION DOC] Compiles a function returning the string form of a counter built
        # from a short string and checks it matches eager output.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its tensor argument and returns the string
            # conversion of a freshly built counter.
            return str(collections.Counter("aba"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_str(self):
        # [FUNCTION DOC] Checks that string conversion of an instance whose class defines
        # a dunder string method returns that method's result under compilation.
        class MyObj:
            # [CLASS DOC] HELPER: Stores one value and defines a custom string form.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.value = value

            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the repr of
                # the stored value.
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the object argument.
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_dunder_str(self):
        # [FUNCTION DOC] Same as the previous test but the compiled function calls the
        # dunder string method directly on the object; results must match eager.
        class MyObj:
            # [CLASS DOC] HELPER: Stores one value and defines a custom string form.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.value = value

            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the repr of
                # the stored value.
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the result of calling the object's dunder
            # string method directly.
            return obj.__str__()

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_default_object_str(self):
        # [FUNCTION DOC] Checks string conversion of an instance of a plain class with no
        # custom methods (default address-based text) is identical between eager and
        # compiled runs, using the same instance for both.
        class Plain:
            # [CLASS DOC] HELPER: Empty class relying on default string conversion.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the object argument.
            return str(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_repr_fallback_for_str(self):
        # [FUNCTION DOC] Checks that for a class defining only a representation method,
        # string conversion falls back to it, and compiled output equals eager output.
        class MyObj:
            # [CLASS DOC] HELPER: Stores one value and defines only a custom repr.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.value = value

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the repr of
                # the stored value.
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the object argument.
            return str(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_on_plain_instance(self):
        # [FUNCTION DOC] Checks the object base-class dunder string method applied to a
        # plain instance gives the same text under compilation as in eager mode.
        class Plain:
            # [CLASS DOC] HELPER: Empty class using default representation.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the object base-class dunder string result
            # for the given object.
            return object.__str__(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_str_ignores_user_defined_str(self):
        # [FUNCTION DOC] Checks that the object base-class dunder string method bypasses
        # a user-defined dunder string and yields the repr text instead: compiled output
        # equals eager output, equals the repr, and differs from the normal string form.
        class MyObj:
            # [CLASS DOC] HELPER: Class with a custom string method returning a fixed text.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the fixed text MyObj.
                return "MyObj"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the object base-class dunder string result
            # for the given object.
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_object_dunder_str_uses_user_defined_repr(self):
        # [FUNCTION DOC] Checks that the object base-class dunder string method uses a
        # user-defined repr method (not the user-defined string method): compiled output
        # equals eager, equals the repr, and differs from the normal string form.
        class MyObj:
            # [CLASS DOC] HELPER: Class with distinct fixed texts for repr and string.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the fixed text MyObjRepr.
                return "MyObjRepr"

            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the fixed text MyObjStr.
                return "MyObjStr"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the object base-class dunder string result
            # for the given object.
            return object.__str__(obj)

        obj = MyObj()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, repr(obj))
        self.assertNotEqual(out, str(obj))

    def test_str_returning_non_string_raises(self):
        # [FUNCTION DOC] Checks that a dunder string method returning a non-string makes
        # string conversion raise a type error under compilation; the compiled function
        # catches it and returns the message, which must mention the dunder string method
        # name and equal the eager message.
        class BadStr:
            # [CLASS DOC] HELPER: Class whose string method wrongly returns an integer.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the integer three instead of a string.
                return 3  # noqa: PLE0307

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Converts the object to a string; on a type error
            # returns the error message text instead.
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        obj = BadStr()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertIn("__str__", out)
        self.assertEqual(fn(x, obj), out)

    @unittest.expectedFailure
    def test_user_defined_opaque_str_descriptor_raises_type_error(self):
        # [FUNCTION DOC] Expected-failure test. In eager mode, converting an instance of
        # the helper class (whose string method is the unbound upper-casing method of the
        # string type) raises a type error whose message says the descriptor does not apply;
        # the test asserts that message and that the compiled run returns the same message.
        # It is marked as expected to fail, so Dynamo does not yet match this behavior.
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Converts the object to a string; on a type error
            # returns the error message text instead.
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueStrDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply",
            eager_result,
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(eager_result, compiled(x, _OpaqueStrDescriptorObject()))

    def test_metaclass_str(self):
        # [FUNCTION DOC] Checks string conversion of a class whose metaclass defines both
        # repr and string methods uses the metaclass string method; compiled equals eager.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass with distinct repr and string texts that
            # embed the class name.
            def __repr__(cls):
                # [FUNCTION DOC] HELPER: Returns repr-style text with the class name.
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                # [FUNCTION DOC] HELPER: Returns string-style text with the class name.
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Empty class created with the custom metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the class object.
            return str(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_type_dunder_str_on_class(self):
        # [FUNCTION DOC] Checks that the type base-class dunder string method, applied to
        # a class with a custom metaclass, gives matching results in eager and compiled
        # runs, and that the compiled result equals the same call made outside compilation.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass with distinct repr and string texts that
            # embed the class name.
            def __repr__(cls):
                # [FUNCTION DOC] HELPER: Returns repr-style text with the class name.
                return f"<MetaRepr {cls.__name__}>"

            def __str__(cls):
                # [FUNCTION DOC] HELPER: Returns string-style text with the class name.
                return f"<MetaStr {cls.__name__}>"

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Empty class created with the custom metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the type base-class dunder string result for
            # the class object.
            return type.__str__(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))
        self.assertEqual(compiled(x), type.__str__(MyClass))

    def test_user_function_str(self):
        # [FUNCTION DOC] Checks string conversion of a locally defined function object
        # matches between eager and compiled runs.
        def helper(y):
            # [FUNCTION DOC] HELPER: Returns its argument plus one; only its string form
            # is used by the test.
            return y + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the helper function.
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_lambda_str(self):
        # [FUNCTION DOC] Checks string conversion of an anonymous function object held in
        # a closure variable matches between eager and compiled runs.
        helper = lambda: None  # noqa: E731

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the captured
            # anonymous function.
            return str(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str(self):
        # [FUNCTION DOC] Checks string conversion of a default-dict with the int factory
        # and one entry matches between eager and compiled runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Ignores its tensor argument and returns the string
            # conversion of a freshly built default-dict.
            return str(collections.defaultdict(int, {"a": 1}))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_str_with_nested_function_factory_unsupported(self):
        # [FUNCTION DOC] Checks that converting to a string a default-dict whose factory is
        # a nested function closing over a traced tensor makes full-graph compilation raise
        # the Dynamo unsupported error, with a message about repr on a nested function with
        # a non-constructible closure.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines a nested factory closing over the tensor
            # argument and returns the string conversion of a default-dict using it.
            def factory():
                # [FUNCTION DOC] HELPER: Returns the captured tensor argument.
                return x

            return str(collections.defaultdict(factory, {"a": 1}))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_ordereddict_and_namedtuple_str_track_nested_repr(self):
        # [FUNCTION DOC] Checks that string forms of an ordered dict and a named tuple that
        # both hold an object with a custom repr reflect a mutation (appending to a list
        # attribute) made between two conversions. Compares the tensor output and the two
        # string tuples from eager and compiled runs, using separate fresh objects.
        class Obj:
            # [CLASS DOC] HELPER: Holds a value and renders it through a custom repr.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the value.
                return f"Obj({self.val})"

        class Named(typing.NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with a single field holding any object.
            obj: object

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Wraps the object in an ordered dict and a named tuple,
            # takes their string forms, appends zero to the object's list value, takes the
            # string forms again, and returns the incremented tensor plus both pairs.
            ordered = collections.OrderedDict([("obj", obj)])
            named = Named(obj)
            y = x + 1
            s1 = (str(ordered), str(named))
            obj.val.append(0)
            s2 = (str(ordered), str(named))
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_structseq_str_with_tensor_graph_breaks(self):
        # [FUNCTION DOC] Checks that string conversion of a max-reduction result (a struct
        # sequence holding tensors) is unsupported in full-graph mode, raising the Dynamo
        # unsupported error, while the non-full-graph compiled function falls back and
        # returns text equal to eager's.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of the max reduction of
            # the input along the first dimension.
            return str(torch.max(x, dim=0))

        x = torch.randn(3, 2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), str(torch.max(x, dim=0)))


class TpStrExceptionTests(TestCase):
    # [CLASS DOC] Tests of string conversion for exception instances, both built-in
    # (run directly under Dynamo) and user-defined subclasses (compiled and compared).
    @make_dynamo_test
    def test_exception_no_args(self):
        # [FUNCTION DOC] Checks an exception built with no arguments converts to an empty
        # string.
        assert str(ValueError()) == ""  # noqa: S101

    @make_dynamo_test
    def test_exception_one_arg(self):
        # [FUNCTION DOC] Checks an exception with one string argument converts to that
        # string.
        assert str(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_exception_one_int_arg(self):
        # [FUNCTION DOC] Checks an exception with one integer argument converts to the
        # integer's decimal text.
        assert str(ValueError(42)) == "42"  # noqa: S101

    @make_dynamo_test
    def test_exception_multiple_args(self):
        # [FUNCTION DOC] Checks an exception with two arguments converts to the text of
        # the argument tuple.
        assert str(ValueError("error", 42)) == "('error', 42)"  # noqa: S101

    @make_dynamo_test
    def test_exception_dunder(self):
        # [FUNCTION DOC] Checks calling the dunder string method directly on a type error
        # instance returns its message.
        assert TypeError("bad type").__str__() == "bad type"  # noqa: S101

    @make_dynamo_test
    def test_exception_unbound_dunder(self):
        # [FUNCTION DOC] Checks calling the unbound dunder string method of the value error
        # class with an instance returns its message.
        assert ValueError.__str__(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_runtime_error(self):
        # [FUNCTION DOC] Checks a runtime error instance converts to its message text.
        assert str(RuntimeError("runtime failure")) == "runtime failure"  # noqa: S101

    def test_user_defined_exception_subclass_str(self):
        # [FUNCTION DOC] Checks string conversion of an instance of a user-defined value
        # error subclass with no overrides matches between eager and compiled runs.
        class MyError(ValueError):
            # [CLASS DOC] HELPER: Empty subclass of the value error exception.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of a new instance of
            # the subclass carrying one message argument.
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_defined_exception_subclass_custom_str(self):
        # [FUNCTION DOC] Checks string conversion of an exception subclass overriding the
        # dunder string method uses the override; eager and compiled results must match.
        class MyError(ValueError):
            # [CLASS DOC] HELPER: Value error subclass with a custom string form built from
            # its first argument.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the repr of
                # the first stored argument.
                return f"MyError({self.args[0]!r})"

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the string conversion of a new instance of
            # the subclass carrying one message argument.
            return str(MyError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


class FStringMutationTests(TestCase):
    # [CLASS DOC] Test case for formatted-string literals and explicit string conversion:
    # each test builds a function that formats an object, mutates it, formats it again, and
    # verifies via the shared _check helper that compiled output equals eager output.
    """Tests for f-string mutation ordering (issue #177582).

    Dynamo must evaluate f-string formatting at the correct bytecode point
    so that mutations between two f-strings are reflected in the output.
    """

    def _check(self, fn, *args_factory):
        # [FUNCTION DOC] HELPER: Runs the function eagerly on a deep copy of the given
        # arguments, then compiles it with a compile-counting backend and runs it on
        # another deep copy. Asserts the two results are equal and that exactly one frame
        # was compiled (no graph break). Imports torch modules locally.
        import copy

        import torch
        import torch._dynamo.testing

        eager_result = fn(*copy.deepcopy(args_factory))
        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt)
        compiled_result = compiled_fn(*copy.deepcopy(args_factory))
        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(cnt.frame_count, 1)

    def test_fstring_tracks_user_defined_object_mutations(self):
        # [FUNCTION DOC] Checks two f-string renderings of an object with a custom repr,
        # separated by an in-place list mutation, show the mutation in the second string.
        import torch

        class Obj:
            # [CLASS DOC] HELPER: Holds a value and renders it through a custom repr.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the value.
                return f"Obj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the incremented tensor and two f-string
            # renderings of the object, with a zero appended to its list between them.
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_tracks_frozen_dataclass_field_mutations(self):
        # [FUNCTION DOC] Same mutation-ordering check using a frozen dataclass whose list
        # field is mutated in place (the field binding is frozen, the list is not) and
        # which has a custom repr.
        from dataclasses import dataclass

        import torch

        @dataclass(frozen=True)
        class FrozenObj:
            # [CLASS DOC] HELPER: Frozen dataclass with one list field and a custom repr.
            val: list

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the field.
                return f"FrozenObj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the incremented tensor and two f-string
            # renderings of the object, with a zero appended to its list between them.
            x = x + 1
            s1 = f"obj = {obj}"
            obj.val.append(0)
            s2 = f"obj = {obj}"
            return x, s1, s2

        self._check(fn, torch.randn(3), FrozenObj([1, 2]))

    def test_fstring_str_conversion_tracks_mutations(self):
        # [FUNCTION DOC] Same mutation-ordering check using the explicit string conversion
        # flag inside the f-string on an object that only defines a repr.
        import torch

        class Obj:
            # [CLASS DOC] HELPER: Holds a value and renders it through a custom repr.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the value.
                return f"Obj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the incremented tensor and two string-flag
            # f-string renderings, with a zero appended to the list between them.
            x = x + 1
            s1 = f"{obj!s}"
            obj.val.append(0)
            s2 = f"{obj!s}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_fstring_repr_conversion_tracks_mutations(self):
        # [FUNCTION DOC] Same mutation-ordering check using the explicit repr conversion
        # flag inside the f-string.
        import torch

        class Obj:
            # [CLASS DOC] HELPER: Holds a value and renders it through a custom repr.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the value.
                return f"Obj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the incremented tensor and two repr-flag
            # f-string renderings, with a zero appended to the list between them.
            x = x + 1
            s1 = f"{obj!r}"
            obj.val.append(0)
            s2 = f"{obj!r}"
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))

    def test_explicit_str_tracks_mutations(self):
        # [FUNCTION DOC] Same mutation-ordering check but using the built-in string
        # conversion function instead of an f-string.
        import torch

        class Obj:
            # [CLASS DOC] HELPER: Holds a value and renders it through a custom repr.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Saves the given value as an attribute.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text naming the class and the value.
                return f"Obj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the incremented tensor and two explicit string
            # conversions of the object, with a zero appended to its list between them.
            x = x + 1
            s1 = str(obj)
            obj.val.append(0)
            s2 = str(obj)
            return x, s1, s2

        self._check(fn, torch.randn(3), Obj([1, 2]))


if __name__ == "__main__":
    run_tests()
