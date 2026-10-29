# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that calling repr on various objects inside Dynamo-compiled
# code gives the same result as eager: builtins, containers, enums, user classes,
# metaclasses, exceptions, functions, modules, defaultdicts, structseqs and
# compile-time identity values. Also covers cases that must graph break or raise
# Unsupported (tensors in reprs, closures). Defines small fixture classes.
"""Tests for tp_repr / generic_repr: repr() via PyObject_Repr in Dynamo."""

import collections
import enum
import typing
import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class _Color(enum.Enum):
    # [CLASS DOC] HELPER: Enum with two members used to test member repr.
    RED = 1
    BLUE = 2


class _OpaqueReprDescriptorObject:
    # [CLASS DOC] HELPER: Fixture whose repr attribute is a C string method descriptor
    # that cannot be applied to it, so calling repr fails in eager mode.
    __repr__ = str.upper


class TpReprTests(TestCase):
    # [CLASS DOC] Tests for Dynamo support of repr (type slot repr and generic repr).
    # Many use the make_dynamo_test decorator, which runs the body eagerly and compiled.
    @make_dynamo_test
    def test_int_repr(self):
        # [FUNCTION DOC] Repr of an integer equals its decimal string.
        assert repr(3) == "3"  # noqa: S101

    @make_dynamo_test
    def test_int_dunder_repr(self):
        # [FUNCTION DOC] Calling the repr method of an integer returns its decimal string.
        assert (3).__repr__() == "3"  # noqa: S101

    @make_dynamo_test
    def test_list_repr(self):
        # [FUNCTION DOC] Repr of a list of three integers has the usual bracket format.
        assert repr([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_list_dunder_repr(self):
        # [FUNCTION DOC] Calling the list class repr method on a list gives the usual format.
        assert list.__repr__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_set_repr(self):
        # [FUNCTION DOC] Repr of a set of three integers has the usual brace format.
        assert repr({1, 2, 3}) == "{1, 2, 3}"  # noqa: S101

    def test_counter_repr(self):
        # [FUNCTION DOC] Repr of a Counter built from a string matches eager under full graph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of a Counter of the letters of a string.
            return repr(collections.Counter("aba"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    @make_dynamo_test
    def test_enum_member_repr(self):
        # [FUNCTION DOC] Repr of an enum member shows class, member name and value.
        assert repr(_Color.RED) == "<_Color.RED: 1>"  # noqa: S101

    def test_user_defined_repr(self):
        # [FUNCTION DOC] Repr of an instance of a user class defining its own repr method
        # matches eager.
        class MyObj:
            # [CLASS DOC] HELPER: Holds a value and has a repr showing the class name and value.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Store the value on the instance.
                self.value = value

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the class name with the repr of the value.
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the repr of the object argument.
            return repr(obj)

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_dunder_repr(self):
        # [FUNCTION DOC] Calling the repr method directly on a user instance with a custom
        # repr matches eager.
        class MyObj:
            # [CLASS DOC] HELPER: Holds a value and has a repr showing the class name and value.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Store the value on the instance.
                self.value = value

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the class name with the repr of the value.
                return f"MyObj({self.value!r})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the result of the object's repr method.
            return obj.__repr__()

        obj = MyObj("value")
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_user_defined_default_object_repr(self):
        # [FUNCTION DOC] Repr of a plain user instance
        # with the default object repr matches eager.
        class Plain:
            # [CLASS DOC] HELPER: Empty class using the default object repr.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the repr of the object argument.
            return repr(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_object_dunder_repr_on_plain_instance(self):
        # [FUNCTION DOC] The object base class repr method
        # applied to a plain instance matches eager.
        class Plain:
            # [CLASS DOC] HELPER: Empty class using the default object repr.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the result of
            # the object base class repr on the argument.
            return object.__repr__(obj)

        obj = Plain()
        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_repr_returning_non_string_raises(self):
        # [FUNCTION DOC] A repr method returning a non-string makes repr raise TypeError;
        # the compiled function catches it and returns the message, which must be a string
        # equal to eager's.
        class BadRepr:
            # [CLASS DOC] HELPER: Fixture whose repr method returns an integer.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the integer three instead of a string.
                return 3

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the repr of the
            # object, or the message of a TypeError raised.
            try:
                return repr(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        obj = BadRepr()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertTrue(type(out) is str)
        self.assertEqual(fn(x, obj), out)

    def test_dunder_repr_returning_non_string_raises(self):
        # [FUNCTION DOC] Calling a repr method that returns a non-string directly does not
        # raise; compiled and eager both return the integer three.
        class BadRepr:
            # [CLASS DOC] HELPER: Fixture whose repr method returns an integer.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the integer three instead of a string.
                return 3

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the result of the object's repr method.
            return obj.__repr__()

        x = torch.randn(4)
        obj = BadRepr()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, obj)
        self.assertEqual(fn(x, obj), out)
        self.assertEqual(out, 3)

    # __repr__ = str.upper is an opaque C method descriptor with no traceable
    # body, so Dynamo graph breaks instead of reproducing eager's TypeError.
    @unittest.expectedFailure
    def test_user_defined_opaque_repr_descriptor_raises_type_error(self):
        # [FUNCTION DOC] Expected failure: eager repr of the fixture raises TypeError about
        # the descriptor not applying, and the compiled version should match, but Dynamo
        # graph breaks on the opaque descriptor so full graph compilation fails.
        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the repr of the
            # object, or the message of a TypeError raised.
            try:
                return repr(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueReprDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply",
            eager_result,
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(eager_result, compiled(x, _OpaqueReprDescriptorObject()))

    def test_metaclass_repr(self):
        # [FUNCTION DOC] Repr of a class whose metaclass defines repr matches eager.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose repr shows the class name in angle brackets.
            def __repr__(cls):
                # [FUNCTION DOC] HELPER: Return the angle-bracket string with the class name.
                return f"<Meta {cls.__name__}>"

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Empty class using the custom metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the class defined by the test.
            return repr(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_type_dunder_repr_on_class(self):
        # [FUNCTION DOC] The type class repr method applied to a class with a custom
        # metaclass repr yields the default type repr, matching eager.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose repr shows the class name in angle brackets.
            def __repr__(cls):
                # [FUNCTION DOC] HELPER: Return the angle-bracket string with the class name.
                return f"<Meta {cls.__name__}>"

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Empty class using the custom metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the type base repr of the class defined by the test.
            return type.__repr__(MyClass)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))
        self.assertEqual(compiled(x), type.__repr__(MyClass))

    def test_exception_repr(self):
        # [FUNCTION DOC] Repr of a ValueError instance matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of a ValueError created with a message.
            return repr(ValueError("oops"))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_user_function_repr(self):
        # [FUNCTION DOC] Repr of a plain function matches eager.
        def helper(y):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return y + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the helper function.
            return repr(helper)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_tensor_repr_graph_breaks(self):
        # [FUNCTION DOC] Repr of a tensor raises Unsupported under full graph and works
        # (equal to eager) with graph breaks allowed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the tensor.
            return repr(x)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), repr(x))

    def test_tensor_dunder_repr_graph_breaks(self):
        # [FUNCTION DOC] Calling the repr method on a tensor raises Unsupported under full
        # graph and works with graph breaks allowed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the result of the tensor's repr method.
            return x.__repr__()

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), repr(x))

    def test_list_repr_with_tensor_graph_breaks(self):
        # [FUNCTION DOC] Repr of a list holding a tensor raises Unsupported under full
        # graph and works with graph breaks allowed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of a one-element list holding the tensor.
            return repr([x])

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), repr([x]))

    def test_list_repr_tracks_user_defined_object_mutations(self):
        # [FUNCTION DOC] Repr of a list holding a user object is computed twice around a
        # mutation of the object's list attribute; both
        # strings and the tensor result must match eager.
        class Obj:
            # [CLASS DOC] HELPER: Holds a value and has a repr showing the class name and value.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Store the value on the instance.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the class name with the value.
                return f"Obj({self.val})"

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Compute a tensor, take the repr of a list of the object,
            # mutate the object's value, take the repr again, and return all three.
            y = x + 1
            s1 = repr([obj])
            obj.val.append(0)
            s2 = repr([obj])
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_dict_repr_with_tensor_graph_breaks(self):
        # [FUNCTION DOC] Repr of a dictionary holding a tensor raises Unsupported under
        # full graph and works with graph breaks allowed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of a dictionary holding the tensor.
            return repr({"x": x})

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), repr({"x": x}))

    def test_nn_module_repr(self):
        # [FUNCTION DOC] Repr of a module closed over by the function (a sequential of two
        # layers) matches eager.
        mod = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 4))

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the enclosing module.
            return repr(mod)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_dict_view_repr(self):
        # [FUNCTION DOC] Repr of the keys view of a small dictionary matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of
            # the keys view of a dictionary of constants.
            return repr({"a": 1, "b": 2}.keys())

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_dict_view_repr_with_tensor(self):
        # [FUNCTION DOC] Repr of the keys view of a dictionary containing a tensor compiles
        # fully and gives the expected keys text.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the
            # keys view of a dictionary holding the tensor.
            return repr({"x": x}.keys())

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), "dict_keys(['x'])")

    def test_defaultdict_repr(self):
        # [FUNCTION DOC] Repr of a defaultdict with an int factory matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of a defaultdict with one entry.
            return repr(collections.defaultdict(int, {"a": 1}))

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_defaultdict_repr_with_nested_function_factory_unsupported(self):
        # [FUNCTION DOC] Repr of a defaultdict whose factory is a nested function closing
        # over a tensor raises Unsupported about a non-constructible closure.
        def fn(x):
            # [FUNCTION DOC] HELPER: Define a closure factory
            # and return the repr of a defaultdict using it.
            def factory():
                # [FUNCTION DOC] HELPER: Return the closed-over input.
                return x

            return repr(collections.defaultdict(factory, {"a": 1}))

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_defaultdict_repr_uses_factory_repr(self):
        # [FUNCTION DOC] Repr of a defaultdict uses the factory's own repr; matches eager.
        class Factory:
            # [CLASS DOC] HELPER: Callable factory with a fixed repr string.
            def __call__(self):
                # [FUNCTION DOC] HELPER: Return zero.
                return 0

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a fixed string naming the class.
                return "Factory()"

        def fn(x, factory):
            # [FUNCTION DOC] HELPER: Return the repr of
            # a defaultdict built with the given factory.
            return repr(collections.defaultdict(factory, {"a": 1}))

        x = torch.randn(4)
        factory = Factory()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, factory), compiled(x, factory))

    def test_user_defined_dict_subclass_repr(self):
        # [FUNCTION DOC] Repr of an instance of a dict subclass matches eager.
        class MyDict(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Return the repr of the object argument.
            return repr(obj)

        x = torch.randn(4)
        obj = MyDict({"a": 1})
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, obj), compiled(x, obj))

    def test_ordereddict_and_namedtuple_repr_track_nested_repr(self):
        # [FUNCTION DOC] Reprs of an ordered dictionary and a named tuple that hold a user
        # object are taken before and after the object's list attribute is mutated; the
        # strings and tensor must match eager, showing nested reprs are not stale.
        class Obj:
            # [CLASS DOC] HELPER: Holds a value and has a repr showing the class name and value.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Store the value on the instance.
                self.val = val

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return the class name with the value.
                return f"Obj({self.val})"

        class Named(typing.NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one field holding any object.
            obj: object

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Build the containers, take both reprs, mutate the object,
            # take both again and return the tensor with both pairs of strings.
            ordered = collections.OrderedDict([("obj", obj)])
            named = Named(obj)
            y = x + 1
            s1 = (repr(ordered), repr(named))
            obj.val.append(0)
            s2 = (repr(ordered), repr(named))
            return y, s1, s2

        x = torch.randn(4)
        eager_result = fn(x, Obj([1, 2]))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            x, Obj([1, 2])
        )
        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1:], compiled_result[1:])

    def test_structseq_repr_with_tensor_graph_breaks(self):
        # [FUNCTION DOC] Repr of a structseq result (max with dim) raises Unsupported
        # under full graph and works with graph breaks allowed.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the repr of the max values and indices over dim zero.
            return repr(torch.max(x, dim=0))

        x = torch.randn(3, 2)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)

        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), repr(torch.max(x, dim=0)))

    def test_nested_function_repr_with_tensor_closure_unsupported(self):
        # [FUNCTION DOC] Repr of a nested function closing over a tensor raises Unsupported
        # about a non-constructible closure.
        def fn(x):
            # [FUNCTION DOC] HELPER: Define a closure and return its repr.
            def inner():
                # [FUNCTION DOC] HELPER: Return the closed-over input plus one.
                return x + 1

            return repr(inner)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"repr\(\) on nested function with non-constructible closure",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    def test_self_ref_list_repr(self):
        # [FUNCTION DOC] Repr of a list that contains itself matches eager (recursion
        # placeholder); graph breaks are allowed.
        def fn():
            # [FUNCTION DOC] HELPER: Build a self-containing list and return its repr.
            l = [1, 2, 3]
            l[0] = l
            return repr(l)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())

    def test_self_ref_dict_repr(self):
        # [FUNCTION DOC] Repr of a dictionary that contains itself matches eager; graph
        # breaks are allowed.
        def fn():
            # [FUNCTION DOC] HELPER: Build a self-containing dictionary and return its repr.
            d = {}
            d["self"] = d
            return repr(d)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())

    def test_mutual_ref_repr(self):
        # [FUNCTION DOC] Repr of two lists containing each other matches eager; graph
        # breaks are allowed.
        def fn():
            # [FUNCTION DOC] HELPER: Build two mutually containing
            # lists and return the repr of one.
            a = [1]
            b = [2]
            a.append(b)
            b.append(a)
            return repr(a)

        compiled = torch.compile(fn, backend="eager", fullgraph=False)
        self.assertEqual(compiled(), fn())

    def test_str_of_id_of_compile_time_object(self):
        # [FUNCTION DOC] The string of the identity number of objects created inside the
        # compiled region must be digits, equal for the same object and different for
        # distinct objects. Full graph compilation.
        # id() on a sourceless object minted inside the region yields a
        # FakeIdVariable (int-typed); str()/repr() must mirror int.__repr__
        # and produce a decimal string with stable compile-time identity.
        class Obj:
            # [CLASS DOC] HELPER: Empty class whose instances are created inside the region.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Create two objects, convert identities to strings and
            # return the tensor plus whether the string is all digits, repeats and differs.
            a = Obj()
            b = Obj()
            sa = str(id(a))
            same = sa == str(id(a))
            distinct = sa != str(id(b))
            return x + 1, sa.isdigit(), same, distinct

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        _, is_digit, same, distinct = compiled(torch.randn(4))
        self.assertTrue(is_digit)
        self.assertTrue(same)
        self.assertTrue(distinct)

    def test_repr_of_hash_of_compile_time_object(self):
        # [FUNCTION DOC] The repr of the hash of an object whose hash method returns its
        # identity must be a decimal digit string. Full graph compilation.
        # hash() returning id(self) on a sourceless object yields a
        # FakeIdVariable (HASH kind); repr() must route through its
        # tp_repr_impl and mirror int.__repr__ (a decimal string).
        class Obj:
            # [CLASS DOC] HELPER: Class whose hash method returns the object's identity number.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Return the identity number of the instance.
                return id(self)

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the tensor plus one and whether the repr of an
            # object's hash is all digits.
            s = repr(hash(Obj()))
            return x + 1, s.isdigit()

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        _, is_digit = compiled(torch.randn(4))
        self.assertTrue(is_digit)


if __name__ == "__main__":
    run_tests()
