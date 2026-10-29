# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo's unified attribute-access protocol (the tp_getattro
# implementation): getattr and hasattr builtins, tensor attributes, user-defined objects and
# classes, descriptors, builtin and constant variable types, bound method handling, descriptor type
# checks, and attribute mutation (setattr/delattr) on functions, modules and classes. Almost every
# test compiles a small function with the eager backend and compares the result with the expected
# Python value. No import-time side effects.
"""Tests for tp_getattro_impl: unified attribute access protocol in Dynamo."""

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TpGetattroTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Single test class holding all attribute-access tests. Each test defines a small
    # target function (usually a nested one), compiles it with the eager backend (mostly requiring
    # a full graph with no breaks), calls it and asserts on the result, sometimes also on the
    # compile counter frame count or on a raised error.
    # --- getattr() builtin ---

    def test_getattr_constant(self):
        # [FUNCTION DOC] Compiled function reads the class attribute of an integer constant; result
        # must be the int type.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the class attribute of the integer constant forty-two.
            return (42).__class__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, int)

    def test_getattr_with_default_exists(self):
        # [FUNCTION DOC] The getattr builtin with a default on an existing attribute returns the
        # real attribute (str class), not the default.
        def fn():
            # [FUNCTION DOC] HELPER: Calls the getattr builtin on a string for the class attribute
            # with a None default.
            return getattr("hello", "__class__", None)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, str)

    def test_getattr_with_default_missing(self):
        # [FUNCTION DOC] The getattr builtin with a default on a missing attribute returns the
        # default (42).
        def fn():
            # [FUNCTION DOC] HELPER: Calls the getattr builtin on a string for a nonexistent
            # attribute with default 42.
            return getattr("hello", "nonexistent", 42)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_getattr_with_none_default(self):
        # [FUNCTION DOC] The getattr builtin with None as default on a missing attribute
        # returns None.
        def fn():
            # [FUNCTION DOC] HELPER: Calls the getattr builtin on a string for a nonexistent
            # attribute with a None default.
            return getattr("hello", "nonexistent", None)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIsNone(result)

    # --- hasattr() builtin ---

    def test_hasattr_true(self):
        # [FUNCTION DOC] The hasattr builtin on a tensor for the shape attribute is true, so the
        # compiled function takes the branch that adds one; result equals the input plus one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one if it has a shape attribute, else
            # the input unchanged.
            if hasattr(x, "shape"):
                return x + 1
            return x

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x + 1)

    def test_hasattr_false(self):
        # [FUNCTION DOC] The hasattr builtin on an integer for a nonexistent attribute
        # returns False.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the integer 42 has a nonexistent attribute.
            return hasattr(42, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_constant_true(self):
        # [FUNCTION DOC] The hasattr builtin on a string for an existing method name returns True.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether a string constant has the upper method.
            return hasattr("hello", "upper")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_false_then_access(self):
        # [FUNCTION DOC] After a failing hasattr check, no pending exception state leaks: later
        # reading the shape of the argument still works and returns 5.
        """hasattr returning False must not leak exception state."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Discards a failing hasattr result, then returns the first
            # dimension of the input.
            _ = hasattr(42, "nonexistent")
            return x.shape[0]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 5)

    def test_hasattr_sequence(self):
        # [FUNCTION DOC] Several hasattr calls (two true, two false) in one function each restore
        # exception state; result must be the tuple true, false, true, false.
        """Multiple hasattr calls must each restore exception state."""

        def fn():
            # [FUNCTION DOC] HELPER: Evaluates hasattr on an int and a string for existing and
            # missing attributes and returns the four booleans as a tuple.
            a = hasattr(42, "__add__")
            b = hasattr(42, "nonexistent")
            c = hasattr("hi", "upper")
            d = hasattr("hi", "nonexistent")
            return (a, b, c, d)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (True, False, True, False))

    def test_hasattr_false_in_except(self):
        # [FUNCTION DOC] A failing hasattr inside an except block must keep the active exception
        # visible: the function returns the input plus one only if the current exception type is
        # still ValueError.
        """hasattr inside an except block must preserve the active exception."""
        import sys

        def fn(x):
            # [FUNCTION DOC] HELPER: Raises and catches a ValueError, runs a failing hasattr inside
            # the handler, reads the current exception type, and returns the input plus one if the
            # check was false and the type is ValueError, else the input.
            try:
                raise ValueError("test")
            except ValueError:
                has = hasattr(42, "nonexistent")
                exc_type = sys.exc_info()[0]
                if not has and exc_type is ValueError:
                    return x + 1
            return x

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x + 1)

    def test_hasattr_user_function_true(self):
        # [FUNCTION DOC] The hasattr builtin on a user-defined function for its name attribute
        # is True.
        def bar():
            # [FUNCTION DOC] HELPER: Empty user function whose attribute is queried.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the outer user function has a name attribute.
            return hasattr(bar, "__name__")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_user_function_false(self):
        # [FUNCTION DOC] The hasattr builtin on a user function for a nonexistent attribute
        # is False.
        def bar():
            # [FUNCTION DOC] HELPER: Empty user function whose attribute is queried.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the outer user function has a
            # nonexistent attribute.
            return hasattr(bar, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_skip_function_true(self):
        # [FUNCTION DOC] The hasattr builtin on the print builtin (a skipped function) for its name
        # attribute is True.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the print builtin has a name attribute.
            return hasattr(print, "__name__")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_skip_function_false(self):
        # [FUNCTION DOC] The hasattr builtin on the print builtin for a nonexistent attribute
        # is False.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the print builtin has a nonexistent attribute.
            return hasattr(print, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_hasattr_python_module_true(self):
        # [FUNCTION DOC] The hasattr builtin on the math module for an existing function name
        # is True.
        import math

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the math module has a square root function.
            return hasattr(math, "sqrt")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_hasattr_python_module_false(self):
        # [FUNCTION DOC] The hasattr builtin on the math module for a nonexistent name is False.
        import math

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the math module has a nonexistent attribute.
            return hasattr(math, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    # --- Tensor attributes ---

    def test_tensor_shape(self):
        # [FUNCTION DOC] Reading the shape of a tensor and indexing it gives 5 for a 5 by 3 input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first dimension of the input tensor.
            return x.shape[0]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5, 3))
        self.assertEqual(result, 5)

    def test_tensor_dtype(self):
        # [FUNCTION DOC] Reading the dtype of a float tensor gives the float32 dtype.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the dtype of the input tensor.
            return x.dtype

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3))
        self.assertEqual(result, torch.float32)

    def test_tensor_device(self):
        # [FUNCTION DOC] Reading the device of a tensor gives the same device as the
        # original tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the device of the input tensor.
            return x.device

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, x.device)

    def test_tensor_grad_alias(self):
        # [FUNCTION DOC] Reading the underscore-prefixed gradient attribute of a tensor that has a
        # gradient set returns the same gradient as the public gradient attribute; compiled with a
        # compile counter backend.
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the private gradient attribute of the input tensor.
            return x._grad

        x = torch.randn(3, requires_grad=True)
        x.grad = torch.ones(3)
        result = torch.compile(fn, backend=cnt)(x)
        self.assertEqual(result, x.grad)

    # --- User-defined objects ---

    def test_udov_instance_attr(self):
        # [FUNCTION DOC] Instance attribute on a user-defined object is read inside the compiled
        # function (not full graph); result is 42.
        class MyObj:
            # [CLASS DOC] HELPER: Small class whose instance stores one integer attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute val to 42.
                self.val = 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the val attribute of the argument.
            return obj.val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_udov_class_attr(self):
        # [FUNCTION DOC] Class-level attribute of a user-defined object read through an instance
        # returns 99.
        class MyObj:
            # [CLASS DOC] HELPER: Small class with an integer class attribute.
            class_val = 99

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the class attribute of the argument.
            return obj.class_val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 99)

    def test_udov_property(self):
        # [FUNCTION DOC] A property on a user-defined class evaluated in compiled code returns the
        # property's constant 42.
        class MyObj:
            # [CLASS DOC] HELPER: Small class that exposes one read-only property.
            @property
            def val(self):
                # [FUNCTION DOC] HELPER: Property getter returning the constant 42.
                return 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the property of the argument.
            return obj.val

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_udov_getattr_fallback(self):
        # [FUNCTION DOC] A fallback attribute hook on the class supplies an attribute that is
        # missing normally; reading it in compiled code returns 123.
        class MyObj:
            # [CLASS DOC] HELPER: Class with an attribute fallback hook.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Returns 123 for the name dynamic and raises AttributeError
                # for any other name.
                if name == "dynamic":
                    return 123
                raise AttributeError(name)

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the dynamic attribute of the argument.
            return obj.dynamic

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 123)

    def test_udov_non_function_getattr_graph_breaks(self):
        # [FUNCTION DOC] When the fallback hook of a class is a callable instance rather than a
        # function, the compiled call still returns 42 but the frame cannot be compiled: counter
        # frame count stays 0 (graph break).
        """Non-function __getattr__ (callable instance) triggers a graph break."""

        class CallableGetattr:
            # [CLASS DOC] HELPER: Callable class used as a non-function attribute fallback hook.
            def __call__(self, name):
                # [FUNCTION DOC] HELPER: Ignores the name and returns 42.
                return 42

        class MyObj:
            # [CLASS DOC] HELPER: Class whose attribute fallback hook is set to an instance of the
            # callable class.
            __getattr__ = CallableGetattr()

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the dynamic attribute of the argument.
            return obj.dynamic

        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnt)(MyObj())
        self.assertEqual(result, 42)
        self.assertEqual(cnt.frame_count, 0)

    def test_udov_getattribute_override(self):
        # [FUNCTION DOC] A class that overrides the attribute-lookup hook to return 999 for the
        # name special is read in compiled code; result is 999.
        class MyObj:
            # [CLASS DOC] HELPER: Class overriding the attribute-lookup hook.
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Returns 999 for the name special, otherwise delegates to
                # the parent lookup.
                if name == "special":
                    return 999
                return super().__getattribute__(name)

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the special attribute of the argument.
            return obj.special

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 999)

    # --- User-defined classes (type_getattro) ---

    def test_class_attr(self):
        # [FUNCTION DOC] Reading a class attribute directly on a user-defined class gives 42 in
        # full-graph mode.
        class MyClass:
            # [CLASS DOC] HELPER: Small class with an integer class attribute.
            x = 42

        def fn():
            # [FUNCTION DOC] HELPER: Returns the class attribute of the class.
            return MyClass.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_class_bases(self):
        # [FUNCTION DOC] Reading the bases attribute of a derived user class returns a tuple
        # holding the parent class.
        class A:
            # [CLASS DOC] HELPER: Base class with no members.
            pass

        class B(A):
            # [CLASS DOC] HELPER: Derived class from the base.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns the bases tuple of the derived class.
            return B.__bases__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (A,))

    def test_class_base(self):
        # [FUNCTION DOC] Reading the single base attribute of a derived user class returns the
        # parent class itself.
        class A:
            # [CLASS DOC] HELPER: Base class with no members.
            pass

        class B(A):
            # [CLASS DOC] HELPER: Derived class from the base.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns the single base of the derived class.
            return B.__base__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, A)

    def test_builtin_type_bases(self):
        # [FUNCTION DOC] The bases attribute of a builtin exception type returns a proper tuple
        # equal to the real one.
        """__bases__ on a BuiltinVariable type returns a proper TupleVariable."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns the bases tuple of the ArithmeticError builtin type.
            return ArithmeticError.__bases__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, ArithmeticError.__bases__)

    def test_builtin_type_bases_len(self):
        # [FUNCTION DOC] The length of the bases tuple of a builtin type works under compilation
        # and equals the real length.
        """len() on __bases__ of a builtin type works (regression for test_c_classes)."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns the length of the bases tuple of the ArithmeticError
            # builtin type.
            return len(ArithmeticError.__bases__)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, len(ArithmeticError.__bases__))

    def test_builtin_type_bases_reversed(self):
        # [FUNCTION DOC] Reversing the bases tuple of a builtin type works under compilation;
        # returned list equals the eager reversed list.
        """reversed() over __bases__ works (the original test_c_classes failure path)."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns a list of the reversed bases of the ArithmeticError
            # builtin type.
            return list(reversed(ArithmeticError.__bases__))

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, list(reversed(ArithmeticError.__bases__)))

    def test_builtin_type_bases_slicing(self):
        # [FUNCTION DOC] Slicing the bases of a builtin type works under compilation; returned pair
        # of slices equals the eager slices.
        """Slicing __bases__ works (used by functools._c3_mro)."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns the bases of OSError sliced into first-element and
            # remainder tuples.
            bases = OSError.__bases__
            return bases[:1], bases[1:]

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        expected = OSError.__bases__[:1], OSError.__bases__[1:]
        self.assertEqual(result, expected)

    # --- Module attributes ---

    def test_nn_module_forward(self):
        # [FUNCTION DOC] Calls a linear module through a compiled function that takes the module as
        # argument; result shape is of size 4 and one frame is compiled.
        m = torch.nn.Linear(3, 4)
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x, m):
            # [FUNCTION DOC] HELPER: Calls the module on the input.
            return m(x)

        result = fn(torch.randn(3), m)
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(cnt.frame_count, 1)

    # --- Dunder method dispatch ---

    def test_dunder_getattribute(self):
        # [FUNCTION DOC] Calling the lookup-hook method explicitly on a user object by name returns
        # the stored attribute, 42.
        class MyObj:
            # [CLASS DOC] HELPER: Small class storing one integer attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute val to 42.
                self.val = 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the attribute-lookup method of the argument for the
            # name val.
            return obj.__getattribute__("val")

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_dunder_getattribute_skips_getattr(self):
        # [FUNCTION DOC] Explicit call of the attribute-lookup method for a missing name raises
        # AttributeError even though the class defines a fallback hook, matching CPython.
        """obj.__getattribute__("nonexistent") raises AttributeError even
        when __getattr__ is defined.  This matches CPython: the bytecode
        path (LOAD_ATTR + CALL) resolves __getattribute__ as a bound
        WrapperDescriptor that calls object.__getattribute__ directly,
        which does not invoke __getattr__.
        """

        class MyObj:
            # [CLASS DOC] HELPER: Class whose fallback hook always returns 42.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Fallback hook returning 42 for any name.
                return 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the attribute-lookup method of the argument for a
            # nonexistent name.
            return obj.__getattribute__("nonexistent")

        with self.assertRaises(AttributeError):
            torch.compile(fn, backend="eager")(MyObj())

    # --- Sparse tensor blocking ---

    def test_sparse_tensor_attr_access_graph_breaks(self):
        # [FUNCTION DOC] Reading the shape attribute of a sparse COO tensor graph breaks: the
        # result densifies equal to the input and the compile counter frame count is 0.
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the shape of the input and returns the input.
            _ = x.shape
            return x

        x = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([4.0, 5.0]),
            size=(4, 4),
        )
        result = torch.compile(fn, backend=cnt)(x)
        self.assertEqual(result.to_dense(), x.to_dense())
        # Sparse tensor attribute access triggers graph break
        self.assertEqual(cnt.frame_count, 0)

    # --- TorchInGraphFunctionVariable ---

    def test_torch_in_graph_function_getattro(self):
        # [FUNCTION DOC] A torch in-graph function (the sine function) is accessed and called in
        # compiled code; result equals the eager sine.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of the input.
            return torch.sin(x)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, torch.sin(x))

    # --- Descriptor protocol (tp_descr_get through tp_getattro_impl) ---

    def test_data_descriptor_priority_over_instance_dict(self):
        # [FUNCTION DOC] A data descriptor (property) takes precedence over an entry of the same
        # name stored in the instance dictionary: result is 99, not 1.
        """Data descriptors (property) take precedence over instance __dict__."""

        class MyObj:
            # [CLASS DOC] HELPER: Class with a read-only property returning 99.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the constant 99.
                return 99

        obj = MyObj()
        obj.__dict__["x"] = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the x attribute of the argument.
            return obj.x

        result = torch.compile(fn, backend="eager")(obj)
        self.assertEqual(result, 99)

    def test_non_data_descriptor_shadowed_by_instance_dict(self):
        # [FUNCTION DOC] A non-data descriptor is shadowed by an entry of the same name in the
        # instance dictionary: result is 1, not 99.
        """Instance __dict__ takes precedence over non-data descriptors."""

        class Desc:
            # [CLASS DOC] HELPER: Non-data descriptor class that only defines the get hook.
            def __get__(self, obj, objtype=None):
                # [FUNCTION DOC] HELPER: Get hook returning 99 regardless of instance or
                # owner type.
                return 99

        class MyObj:
            # [CLASS DOC] HELPER: Class with the descriptor as a class attribute.
            x = Desc()

        obj = MyObj()
        obj.__dict__["x"] = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the x attribute of the argument.
            return obj.x

        result = torch.compile(fn, backend="eager")(obj)
        self.assertEqual(result, 1)

    def test_staticmethod_descriptor(self):
        # [FUNCTION DOC] A static method accessed through an instance can be called in full-graph
        # mode and returns 42.
        class MyObj:
            # [CLASS DOC] HELPER: Class with one static method.
            @staticmethod
            def greet():
                # [FUNCTION DOC] HELPER: Static method returning 42.
                return 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the greet static method on the argument.
            return obj.greet()

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, 42)

    def test_classmethod_descriptor(self):
        # [FUNCTION DOC] A class method accessed through an instance is called in full-graph mode
        # and returns the class attribute 10.
        class MyObj:
            # [CLASS DOC] HELPER: Class with a class attribute and a class method.
            val = 10

            @classmethod
            def get_val(cls):
                # [FUNCTION DOC] HELPER: Class method returning the class attribute val.
                return cls.val

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the get_val class method on the argument.
            return obj.get_val()

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, 10)

    def test_classmethod_descriptor_on_class(self):
        # [FUNCTION DOC] A class method accessed directly through the class is called in full-graph
        # mode and returns 10.
        class MyObj:
            # [CLASS DOC] HELPER: Class with a class attribute and a class method.
            val = 10

            @classmethod
            def get_val(cls):
                # [FUNCTION DOC] HELPER: Class method returning the class attribute val.
                return cls.val

        def fn():
            # [FUNCTION DOC] HELPER: Calls the get_val class method on the class.
            return MyObj.get_val()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 10)

    def test_property_setter(self):
        # [FUNCTION DOC] Property with a setter: assigning 5 doubles the stored value, so reading
        # back gives 10.
        class MyObj:
            # [CLASS DOC] HELPER: Class with a private stored value and a property with getter
            # and setter.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the private value to 0.
                self._x = 0

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the private value.
                return self._x

            @x.setter
            def x(self, val):
                # [FUNCTION DOC] HELPER: Property setter storing twice the assigned value.
                self._x = val * 2

        def fn(obj):
            # [FUNCTION DOC] HELPER: Assigns 5 to the x property and returns it.
            obj.x = 5
            return obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 10)

    def test_slots_member_descriptor(self):
        # [FUNCTION DOC] Slot-based class with member descriptors: the sum of the two slot fields
        # is 3.
        class MyObj:
            # [CLASS DOC] HELPER: Class using two slots named x and y.
            __slots__ = ("x", "y")

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the slots x to 1 and y to 2.
                self.x = 1
                self.y = 2

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the sum of the x and y attributes of the argument.
            return obj.x + obj.y

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 3)

    def test_namedtuple_field_access(self):
        # [FUNCTION DOC] Named-tuple field attributes are read in full-graph mode; sum of the two
        # fields of a point (3, 4) is 7.
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        def fn():
            # [FUNCTION DOC] HELPER: Builds a point with fields 3 and 4 and returns the sum of
            # its fields.
            p = Point(3, 4)
            return p.x + p.y

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 7)

    def test_wrapper_descriptor_binding(self):
        # [FUNCTION DOC] A wrapper descriptor (list addition slot) is bound and called on a list;
        # result is the concatenated list.
        """list.__add__ is a wrapper_descriptor; [1].__add__ binds it."""

        def fn():
            # [FUNCTION DOC] HELPER: Concatenates two lists through the explicit add method.
            x = [1, 2]
            y = [3, 4]
            return x.__add__(y)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, [1, 2, 3, 4])

    def test_method_descriptor_binding(self):
        # [FUNCTION DOC] A method descriptor (dict keys) is bound and called; sorted keys equal a
        # and b.
        """dict.keys is a method_descriptor; {}.keys() binds and calls it."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns a list of the keys of a two-entry dictionary.
            d = {"a": 1, "b": 2}
            return list(d.keys())

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(sorted(result), ["a", "b"])

    def test_classmethod_descriptor_dict_fromkeys(self):
        # [FUNCTION DOC] The class-method descriptor of the dict type for building from keys works;
        # result maps a and b to 0.
        """dict.fromkeys is a classmethod_descriptor."""

        def fn():
            # [FUNCTION DOC] HELPER: Builds a dictionary from two keys with value 0 through the
            # class method.
            return dict.fromkeys(["a", "b"], 0)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, {"a": 0, "b": 0})

    # --- Consistency ---

    def test_getattr_matches_dot_access(self):
        # [FUNCTION DOC] Attribute access matches the descriptor result when the same attribute is
        # read twice; comparison is true.
        class MyObj:
            # [CLASS DOC] HELPER: Class with an integer class attribute.
            x = 42

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns whether the x attribute of the argument equals itself
            # when read twice.
            return obj.x == obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertTrue(result)

    # --- generic_getattr dispatch ---

    def test_generic_getattr_side_effects(self):
        # [FUNCTION DOC] Setting then reading an instance attribute inside compiled code returns
        # the new value 42.
        class MyObj:
            # [CLASS DOC] HELPER: Class whose instance starts with x equal to 1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute x to 1.
                self.x = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Sets x to 42 on the argument and returns it.
            obj.x = 42
            return obj.x

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 42)

    def test_delattr_exposes_class_attr(self):
        # [FUNCTION DOC] Deleting an instance attribute exposes the class-level attribute with the
        # same name; result is the class value.
        """Deleting an instance attr exposes the class attr underneath."""

        class MyObj:
            # [CLASS DOC] HELPER: Class with a class attribute x whose instance shadows it in
            # the initializer.
            x = "class"

            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute x to the string instance.
                self.x = "instance"

        def fn(obj):
            # [FUNCTION DOC] HELPER: Deletes the x attribute of the argument and returns the x
            # attribute again.
            del obj.x
            return obj.x

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, "class")

    def test_delattr_then_hasattr_false(self):
        # [FUNCTION DOC] Deleting the only attribute of an object makes the hasattr builtin
        # return False.
        """Deleting the only attr makes hasattr return False."""

        class MyObj:
            # [CLASS DOC] HELPER: Class storing one instance attribute x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute x to 1.
                self.x = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Deletes x from the argument and returns whether it still
            # has x.
            del obj.x
            return hasattr(obj, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertFalse(result)

    def test_dict_replacement_attr_found(self):
        # [FUNCTION DOC] Replacing the instance dictionary wholesale with new content makes later
        # lookups find the new entry; result is 42.
        """Replacing __dict__ wholesale; lookup finds the attr in new dict."""

        class MyObj:
            # [CLASS DOC] HELPER: Class storing one instance attribute x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute x to 1.
                self.x = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Replaces the instance dictionary of the argument with one
            # containing x and y, then returns x.
            obj.__dict__ = {"x": 42, "y": 99}
            return obj.x

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertEqual(result, 42)

    def test_dict_replacement_attr_not_found(self):
        # [FUNCTION DOC] Replacing the instance dictionary with one lacking x makes hasattr
        # return False.
        """Replacing __dict__ wholesale; attr not in new dict."""

        class MyObj:
            # [CLASS DOC] HELPER: Class storing one instance attribute x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the instance attribute x to 1.
                self.x = 1

        def fn(obj):
            # [FUNCTION DOC] HELPER: Replaces the instance dictionary of the argument with one
            # containing only y, then returns whether x exists.
            obj.__dict__ = {"y": 99}
            return hasattr(obj, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)(MyObj())
        self.assertFalse(result)

    # --- UnspecializedNNModule pending mutation ---

    def test_unspecialized_nn_module_pending_mutation_graph_breaks(self):
        # [FUNCTION DOC] A module that sets a new attribute and lists its parameters inside forward
        # is compiled with a counter; result shape is of size 4 and one frame is compiled (pending
        # mutation does not break it).
        # KNOWN ISSUE: the test name says graph breaks but the assertion only checks that one frame
        # was compiled, so a break is not actually verified.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a linear layer whose forward sets an
            # extra attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent initializer and creates a linear layer
                # from 3 to 4.
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets an extra attribute to 1, enumerates the parameters
                # into a list, and returns the linear layer applied to the input.
                self.extra_val = 1
                _params = list(self.parameters())
                return self.linear(x)

        m = MyModule()
        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(m, backend=cnt)(torch.randn(3))
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(cnt.frame_count, 1)

    # --- object_generic_getattr on converted VTs ---

    def test_constant_method_via_generic_getattr(self):
        # [FUNCTION DOC] A method of a string constant is resolved through the descriptor protocol;
        # upper-casing hello gives HELLO.
        """ConstantVariable now resolves methods through the descriptor protocol
        via object_generic_getattr, instead of falling back to GetAttrVariable.
        """

        def fn():
            # [FUNCTION DOC] HELPER: Returns the upper-cased string hello.
            return "hello".upper()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "HELLO")

    def test_constant_class_attr_via_generic_getattr(self):
        # [FUNCTION DOC] The class attribute of an integer constant resolves through a getset
        # descriptor on the object type; result is int.
        """(42).__class__ resolves through getset_descriptor on object."""

        def fn():
            # [FUNCTION DOC] HELPER: Stores 42 in a local and returns its class attribute.
            x = 42
            return x.__class__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertIs(result, int)

    def test_range_method_via_generic_getattr(self):
        # [FUNCTION DOC] A method of a range object resolves through the descriptor protocol; count
        # of 5 in range(10) is 1.
        """RangeVariable now resolves methods through the descriptor protocol."""

        def fn():
            # [FUNCTION DOC] HELPER: Builds a range of 10 and returns the count of the value 5.
            r = range(10)
            return r.count(5)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 1)

    def test_range_index_via_generic_getattr(self):
        # [FUNCTION DOC] The index method of a range resolves; position of 7 in range(10) is 7.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a range of 10 and returns the index of the value 7.
            r = range(10)
            return r.index(7)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 7)

    # --- object_generic_getattr edge cases ---

    def test_constant_nonexistent_attr_raises(self):
        # [FUNCTION DOC] A nonexistent attribute on an integer constant raises AttributeError (no
        # matching entry in the type hierarchy).
        """Step 7: mro_lookup returns NO_SUCH_SUBOBJ -> AttributeError."""

        def fn():
            # [FUNCTION DOC] HELPER: Reads a nonexistent attribute from the integer 42.
            x = 42
            return x.nonexistent

        with self.assertRaises(AttributeError):
            torch.compile(fn, backend="eager")()

    def test_range_start_stop_step(self):
        # [FUNCTION DOC] The start, stop and step attributes of a range are read directly; result
        # is the tuple 2, 10, 3.
        """RangeVariable.tp_getattro_impl fast path for start/stop/step."""

        def fn():
            # [FUNCTION DOC] HELPER: Builds a range of 2 to 10 step 3 and returns its start, stop
            # and step.
            r = range(2, 10, 3)
            return r.start, r.stop, r.step

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (2, 10, 3))

    def test_class_flags(self):
        # [FUNCTION DOC] The flags attribute of a user class is read in full-graph mode and equals
        # the eager value.
        class A:
            # [CLASS DOC] HELPER: Empty class whose flags are queried.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Returns the flags attribute of the class.
            return A.__flags__

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, A.__flags__)

    # --- Dunder __getattr__ explicit call ---

    def test_dunder_getattr_explicit_call(self):
        # [FUNCTION DOC] Calling the fallback hook explicitly by name returns its value for
        # dynamic: 123.
        class MyObj:
            # [CLASS DOC] HELPER: Class with a fallback hook returning 123 for the name dynamic.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: Returns 123 for the name dynamic and raises AttributeError
                # for anything else.
                if name == "dynamic":
                    return 123
                raise AttributeError(name)

        def fn(obj):
            # [FUNCTION DOC] HELPER: Calls the fallback hook of the argument for the name dynamic.
            return obj.__getattr__("dynamic")

        result = torch.compile(fn, backend="eager")(MyObj())
        self.assertEqual(result, 123)

    # --- BoundBuiltinMethodVariable slots ---

    def test_bound_builtin_method_hash(self):
        # [FUNCTION DOC] The hash builtin on a bound builtin method works under compilation; the
        # result is an integer (checked by returning a boolean).
        """hash() on a bound builtin method produced by object_generic_getattr."""

        def fn():
            # [FUNCTION DOC] HELPER: Takes the bound upper method of a string, hashes it and
            # returns whether the hash is an integer.
            s = "hello"
            h = hash(s.upper)
            return isinstance(h, int)

        result = torch.compile(fn, backend="eager")()
        self.assertTrue(result)

    def test_bound_builtin_method_identity_comparison(self):
        # [FUNCTION DOC] Two reads of the same bound builtin method yield distinct objects; the
        # identity inequality is true.
        """Bound builtin methods use identity comparison."""

        def fn():
            # [FUNCTION DOC] HELPER: Reads the upper method of a string twice and returns whether
            # the two bound methods are different objects.
            s = "hello"
            m1 = s.upper
            m2 = s.upper
            return m1 is not m2

        result = torch.compile(fn, backend="eager")()
        self.assertTrue(result)

    def test_bmv_load_then_call(self):
        # [FUNCTION DOC] A bound method of a range is saved in a variable and then called later;
        # count of 5 is 1.
        """Load a method into a variable, then call it through CMV."""

        def fn():
            # [FUNCTION DOC] HELPER: Builds a range of 10, loads its count method into a local, and
            # calls it with 5.
            r = range(10)
            m = r.count
            return m(5)

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 1)

    def test_bmv_defers_graph_break_to_call_time(self):
        # [FUNCTION DOC] Loading a method that Dynamo cannot call (the reduce method of a range)
        # succeeds without a graph break, but calling it raises Unsupported in full-graph mode.
        # Resets Dynamo between the two parts.
        """CallMethodVariable defers graph breaks from LOAD_ATTR to CALL.

        When a method exists on the type (MRO walk finds it) but the VT's
        call_method doesn't handle it, CMV is returned at load time and
        the graph break happens at call time, not at attribute access time.
        """

        # Loading the method succeeds (CMV returned, no graph break).
        @torch.compile(backend="eager", fullgraph=True)
        def fn_load(x):
            # [FUNCTION DOC] HELPER: Builds a range, reads its reduce method without calling it,
            # and returns the input plus one.
            r = range(10)
            r.__reduce__
            return x + 1

        x = torch.randn(3)
        self.assertEqual(fn_load(x), x + 1)

        torch._dynamo.reset()

        # Calling it graph-breaks (call_method doesn't handle __reduce__).
        def fn_call(x):
            # [FUNCTION DOC] HELPER: Builds a range, calls its reduce method, and returns the input
            # plus one.
            r = range(10)
            r.__reduce__()
            return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn_call, backend="eager", fullgraph=True)(x)

    # --- ConstantVariable: CallMethodVariable (format/join have call_method) ---

    def test_str_format_via_bound_method(self):
        # [FUNCTION DOC] The format method of a string constant works; result is hello world.
        def fn():
            # [FUNCTION DOC] HELPER: Formats a template string with one argument.
            return "hello {}".format("world")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "hello world")

    def test_str_join_via_bound_method(self):
        # [FUNCTION DOC] The join method of a string constant works; result is a, b, c.
        def fn():
            # [FUNCTION DOC] HELPER: Joins a three-element list with a comma-space separator.
            return ", ".join(["a", "b", "c"])

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "a, b, c")

    # --- ConstantVariable: other constant types ---

    def test_float_method_via_generic_getattr(self):
        # [FUNCTION DOC] A float constant's is-integer method works; 3.14 gives False.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the is-integer method result for 3.14.
            return (3.14).is_integer()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_int_method_via_generic_getattr(self):
        # [FUNCTION DOC] An integer constant's bit-length method works; 255 gives 8.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the bit length of 255.
            return (255).bit_length()

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 8)

    def test_complex_real_imag(self):
        # [FUNCTION DOC] The real and imaginary attributes of a complex constant work; result is
        # the pair 3.0 and 4.0.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a complex number and returns its real and
            # imaginary parts.
            c = 3 + 4j
            return c.real, c.imag

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, (3.0, 4.0))

    def test_bytes_method_via_generic_getattr(self):
        # [FUNCTION DOC] The decode method of a bytes constant works; result is hello.
        def fn():
            # [FUNCTION DOC] HELPER: Decodes the bytes constant hello as utf-8.
            return b"hello".decode("utf-8")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, "hello")

    def test_dict_method_as_standalone_attr(self):
        # [FUNCTION DOC] Regression test: reading the keys method of a dictionary argument as a
        # standalone value (not called) must not build a source chain through the dict guard
        # manager; returns the method's name keys.
        # Regression test: accessing d.keys as a standalone value (not
        # immediately called) must not create an AttrSource chain through
        # the DictGuardManager, which doesn't support getattr_manager.
        def fn(d):
            # [FUNCTION DOC] HELPER: Reads the keys method of the dictionary into a local and
            # returns its name attribute.
            m = d.keys
            return m.__name__

        result = torch.compile(fn, backend="eager", fullgraph=True)({"a": 1})
        self.assertEqual(result, "keys")

    def test_pybind11_class_attr_constant_fold(self):
        # [FUNCTION DOC] An attribute read on a constant class with a pybind11-style metaclass
        # overriding the lookup hook is folded to a constant instead of graph breaking; compiled
        # output equals eager for adding the SDPA math backend value.
        """LOAD_ATTR on a constant class with a pybind11 metaclass (which
        overrides __getattribute__) should constant-fold rather than
        graph-break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the value of the math member of the
            # SDPBackend enumeration.
            return x + torch.nn.attention.SDPBackend.MATH.value

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_constant_fold_fallback_recompiles_on_change(self):
        # [FUNCTION DOC] When constant folding is used because the metaclass overrides the lookup
        # hook, changing the class attribute afterwards must trigger a recompile: first call gives
        # the input plus 10 with one frame, then after setting the value to 20 gives plus 20 and
        # two frames.
        """When _load_attr constant-folds because the metaclass overrides
        __getattribute__, the guard must trigger recompilation if the
        attribute value changes."""

        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that overrides the class attribute lookup hook.
            def __getattribute__(cls, name):
                # [FUNCTION DOC] HELPER: Delegates to the standard type lookup for the attribute.
                return type.__getattribute__(cls, name)

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the overriding metaclass with a class attribute value
            # equal to 10.
            value = 10

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the class attribute value.
            return x + MyClass.value

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 10)
        self.assertEqual(cnt.frame_count, 1)

        MyClass.value = 20
        self.assertEqual(fn(x), x + 20)
        self.assertEqual(cnt.frame_count, 2)

    # --- C descriptor type check (descr_check equivalent) ---

    def test_method_descriptor_incompatible_type(self):
        # [FUNCTION DOC] A method descriptor from list borrowed onto an unrelated class raises
        # TypeError in eager when read from an instance, and Dynamo must raise its Unsupported
        # error in full-graph mode rather than silently succeeding.
        class Borrower:
            # [CLASS DOC] HELPER: Class that borrows the list append method descriptor as a
            # class attribute.
            append = list.append

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Reads the append attribute on the object and returns the input
            # plus one.
            obj.append
            return x + 1

        x = torch.randn(4)
        b = Borrower()
        with self.assertRaises(TypeError):
            fn(x, b)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x, b)

    def test_wrapper_descriptor_incompatible_type(self):
        # [FUNCTION DOC] Same as the previous test for a wrapper descriptor (list addition slot):
        # TypeError in eager, Unsupported under full-graph compile.
        class Borrower:
            # [CLASS DOC] HELPER: Class that borrows the list addition wrapper descriptor.
            add = list.__add__

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Reads the add attribute on the object and returns the input
            # plus one.
            obj.add
            return x + 1

        x = torch.randn(4)
        b = Borrower()
        with self.assertRaises(TypeError):
            fn(x, b)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x, b)

    def test_member_descriptor_incompatible_type(self):
        # [FUNCTION DOC] Same check for a slot member descriptor taken from another class:
        # TypeError in eager, Unsupported under full-graph compile.
        class Alien:
            # [CLASS DOC] HELPER: Class defining one slot x.
            __slots__ = ("x",)

        class Borrower:
            # [CLASS DOC] HELPER: Class borrowing the slot member descriptor of the other class.
            x = Alien.x

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Reads the x attribute on the object and returns the input
            # plus one.
            obj.x
            return x + 1

        x = torch.randn(4)
        b = Borrower()
        with self.assertRaises(TypeError):
            fn(x, b)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x, b)

    def test_getset_descriptor_incompatible_type(self):
        # [FUNCTION DOC] Same check for a getset descriptor taken from the int type: TypeError in
        # eager, Unsupported under full-graph compile.
        class Borrower:
            # [CLASS DOC] HELPER: Class borrowing the denominator getset descriptor of the
            # int type.
            denominator = int.denominator

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Reads the denominator attribute on the object and returns the
            # input plus one.
            obj.denominator
            return x + 1

        x = torch.randn(4)
        b = Borrower()
        with self.assertRaises(TypeError):
            fn(x, b)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x, b)

    def test_method_descriptor_compatible_type(self):
        # [FUNCTION DOC] Positive case: calling a list method on a real list works; the compiled
        # output sum equals the eager sum.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a three-element list, appends 4 to it and returns
            # the input.
            l = [1, 2, 3]
            l.append(4)
            return x

        x = torch.randn(4)
        eager = fn(x).sum()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)(x).sum()
        self.assertEqual(eager, compiled)

    # --- UserFunctionVariable attribute mutation ---

    def test_function_setattr_then_getattr(self):
        # [FUNCTION DOC] Setting an attribute on a user function and reading it back inside
        # compiled code returns 42.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets the attribute x of the target function to 42 and
            # returns it.
            target.x = 42
            return target.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_function_hasattr_after_setattr(self):
        # [FUNCTION DOC] The hasattr builtin sees an attribute that was just set on a user function
        # in compiled code.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x on the target function and returns whether
            # it exists.
            target.x = 42
            return hasattr(target, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_function_preexisting_attr(self):
        # [FUNCTION DOC] An attribute set on a function before compilation is readable in compiled
        # code; value is 10.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        target.x = 10

        def fn():
            # [FUNCTION DOC] HELPER: Returns the attribute x of the target function.
            return target.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 10)

    def test_function_setattr_persists(self):
        # [FUNCTION DOC] An attribute set on a function inside compiled code persists in the real
        # function after the call, and the function returns the input plus one.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets attribute my_attr to 99 on the target function and
            # returns the input plus one.
            target.my_attr = 99
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt_fn(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertEqual(target.my_attr, 99)

    def test_function_setattr_tensor_value(self):
        # [FUNCTION DOC] A tensor value stored on a function attribute inside compiled code is
        # readable inside and persists afterwards with value 6.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn(x):
            # [FUNCTION DOC] HELPER: Stores the input plus one on the data attribute of the target
            # and returns that attribute.
            target.data = x + 1
            return target.data

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, torch.tensor(6))
        self.assertEqual(target.data, torch.tensor(6))

    def test_function_setattr_cross_compilation(self):
        # [FUNCTION DOC] Attribute set in one compiled function is visible to a second separately
        # compiled function and to eager code.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn1():
            # [FUNCTION DOC] HELPER: Sets attribute x to 42 on the target function.
            target.x = 42

        def fn2():
            # [FUNCTION DOC] HELPER: Returns attribute x of the target function.
            return target.x

        torch.compile(fn1, backend="eager", fullgraph=True)()
        self.assertEqual(target.x, 42)
        result = torch.compile(fn2, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_function_overwrite_preexisting_attr(self):
        # [FUNCTION DOC] Overwriting an attribute that already exists on a function inside compiled
        # code gives 42 both inside the call and on the real function afterwards.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        target.x = 10

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x to 42 on the target function and returns it.
            target.x = 42
            return target.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)
        self.assertEqual(target.x, 42)

    # --- PythonModuleVariable attribute mutation ---

    def test_module_setattr_then_getattr(self):
        # [FUNCTION DOC] Setting and reading an attribute on a freshly created module object inside
        # compiled code returns 42.
        import types

        mod = types.ModuleType("test_mod")

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x to 42 on a module object and returns it.
            mod.x = 42
            return mod.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_module_hasattr_after_setattr(self):
        # [FUNCTION DOC] The hasattr builtin sees an attribute just set on a module object in
        # compiled code.
        import types

        mod = types.ModuleType("test_mod")

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x on the module object and returns whether
            # it exists.
            mod.x = 42
            return hasattr(mod, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_module_setattr_persists(self):
        # [FUNCTION DOC] An attribute set on a module object in compiled code persists after the
        # call; function returns the input plus one.
        import types

        mod = types.ModuleType("test_mod")

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets attribute my_attr to 99 on the module object and returns
            # the input plus one.
            mod.my_attr = 99
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt_fn(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertEqual(mod.my_attr, 99)

    # --- UserDefinedClassVariable attribute mutation ---

    def test_class_setattr_then_getattr(self):
        # [FUNCTION DOC] Setting then reading an attribute on a user-defined class in compiled code
        # returns 42.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class that receives an attribute.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Sets the class attribute x to 42 and returns it.
            MyClass.x = 42
            return MyClass.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_class_hasattr_after_setattr(self):
        # [FUNCTION DOC] The hasattr builtin sees an attribute just set on a user class in
        # compiled code.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class that receives an attribute.
            pass

        def fn():
            # [FUNCTION DOC] HELPER: Sets class attribute x and returns whether it exists.
            MyClass.x = 42
            return hasattr(MyClass, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_class_setattr_persists(self):
        # [FUNCTION DOC] An attribute set on a user class in compiled code persists after the call;
        # function returns the input plus one.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class that receives an attribute.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets class attribute my_attr to 99 and returns the input
            # plus one.
            MyClass.my_attr = 99
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt_fn(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertEqual(MyClass.my_attr, 99)

    # --- SkipFunctionVariable attribute mutation ---

    def test_skip_function_setattr_then_getattr(self):
        # [FUNCTION DOC] Setting then reading an attribute on a function decorated as disabled for
        # Dynamo works in compiled code; result is 42.
        @torch._dynamo.disable
        def skipped():
            # [FUNCTION DOC] HELPER: Function decorated to be skipped by Dynamo; returns 0.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x on the skipped function to 42 and returns it.
            skipped.x = 42
            return skipped.x

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_skip_function_setattr_persists(self):
        # [FUNCTION DOC] An attribute set on a Dynamo-disabled function inside compiled code
        # persists after the call.
        @torch._dynamo.disable
        def skipped():
            # [FUNCTION DOC] HELPER: Function decorated to be skipped by Dynamo; returns 0.
            return 0

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets attribute my_attr on the skipped function and returns the
            # input plus one.
            skipped.my_attr = 99
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt_fn(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertEqual(skipped.my_attr, 99)

    def test_skip_function_hasattr_after_setattr(self):
        # [FUNCTION DOC] The hasattr builtin sees an attribute just set on a Dynamo-disabled
        # function in compiled code.
        @torch._dynamo.disable
        def skipped():
            # [FUNCTION DOC] HELPER: Function decorated to be skipped by Dynamo; returns 0.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x on the skipped function and returns whether
            # it exists.
            skipped.x = 42
            return hasattr(skipped, "x")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertTrue(result)

    def test_polyfilled_function_hasattr_after_setattr(self):
        # [FUNCTION DOC] The hasattr builtin sees an attribute set on a function registered as a
        # polyfill substitute. Registers the pair with the compiler substitution API and, in a
        # finally block, deletes the attribute from the original function to avoid leaking state.
        def my_polyfill(x):
            # [FUNCTION DOC] HELPER: Identity function registered as the function to
            # be substituted.
            return x

        def my_replacement(x):
            # [FUNCTION DOC] HELPER: Identity function used as the replacement implementation.
            return x

        torch.compiler.substitute_in_graph(my_polyfill)(my_replacement)
        try:

            def fn():
                # [FUNCTION DOC] HELPER: Sets attribute x on the polyfilled function and returns
                # whether it exists.
                my_polyfill.x = 42
                return hasattr(my_polyfill, "x")

            result = torch.compile(fn, backend="eager", fullgraph=True)()
            self.assertTrue(result)
        finally:
            if hasattr(my_polyfill, "x"):
                del my_polyfill.x

    # --- Additional coverage for setattr mutation paths ---

    def test_two_setattrs_on_same_object(self):
        # [FUNCTION DOC] Two attributes set on the same function in compiled code can both be read;
        # sum is 141.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives attributes.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attributes x and y to 42 and 99 and returns their sum.
            target.x = 42
            target.y = 99
            return target.x + target.y

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 141)

    def test_hasattr_nonexistent_on_opted_in_vt(self):
        # [FUNCTION DOC] The hasattr builtin on a function that had a different attribute set
        # returns False for a nonexistent name.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x and returns whether a nonexistent
            # attribute exists.
            target.x = 42
            return hasattr(target, "nonexistent")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertFalse(result)

    def test_getattr_with_default_after_setattr(self):
        # [FUNCTION DOC] The getattr builtin with a default returns the real value after setting
        # the attribute, not the default sentinel.
        def target():
            # [FUNCTION DOC] HELPER: Empty function that receives an attribute.
            return 0

        def fn():
            # [FUNCTION DOC] HELPER: Sets attribute x to 42 and reads it through the getattr
            # builtin with a string default.
            target.x = 42
            return getattr(target, "x", "sentinel")

        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, 42)

    def test_polyfilled_function_setattr(self):
        # [FUNCTION DOC] Setting an attribute on a polyfill-registered function inside compiled
        # code persists on the function. Registers the substitution pair and in a finally block
        # deletes the attribute to clean up.
        def my_polyfill(x):
            # [FUNCTION DOC] HELPER: Identity function registered as the function to
            # be substituted.
            return x

        def my_replacement(x):
            # [FUNCTION DOC] HELPER: Identity function used as the replacement implementation.
            return x

        torch.compiler.substitute_in_graph(my_polyfill)(my_replacement)
        try:

            def fn(x):
                # [FUNCTION DOC] HELPER: Sets a private attribute on the polyfilled function to a
                # string and returns the input plus one.
                my_polyfill._my_attr = "mutated"
                return x + 1

            x = torch.randn(4)
            result = torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(my_polyfill._my_attr, "mutated")
            self.assertEqual(result, x + 1)
        finally:
            if hasattr(my_polyfill, "_my_attr"):
                del my_polyfill._my_attr

    # --- Mutation opt-in guard tests ---

    def test_torch_internal_class_setattr_graph_breaks(self):
        # [FUNCTION DOC] Setting an attribute on a class inside the torch package (the
        # Linear module class) is not allowed to be traced; compiling in full-graph mode
        # raises Unsupported.
        """setattr on torch-internal classes should graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets an attribute on the Linear module class and returns the
            # input plus one.
            torch.nn.Linear.my_attr = 42
            return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(1))

    def test_torch_toplevel_class_setattr_graph_breaks(self):
        # [FUNCTION DOC] Same for a class whose module is the torch top level (the Size class):
        # full-graph compile raises Unsupported.
        """setattr on classes with __module__=='torch' should graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets an attribute on the torch Size class and returns the
            # input plus one.
            torch.Size.my_attr = 42
            return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(1))

    def test_metaclass_setattr_works(self):
        # [FUNCTION DOC] Setting an attribute on a class that has a custom metaclass setter works
        # because side effects are replayed through that setter; the attribute is 42 afterwards and
        # the call returns the input plus one.
        """setattr on class with custom metaclass __setattr__ works because
        side-effect replay calls setattr() which goes through the metaclass.
        """

        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass overriding the class attribute set hook.
            def __setattr__(cls, name, value):
                # [FUNCTION DOC] HELPER: Delegates to the parent set hook.
                super().__setattr__(name, value)

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Empty class using the custom metaclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets attribute my_attr on the class to 42 and returns the
            # input plus one.
            MyClass.my_attr = 42
            return x + 1

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertEqual(MyClass.my_attr, 42)

    def test_class_delattr(self):
        # [FUNCTION DOC] Deleting an attribute from a class works (the graph breaks at the delete
        # but replays); afterwards the class no longer has the attribute and the call returns the
        # input plus one.
        """delattr on a class works (graph breaks at the del, but replays)."""

        class MyClass:
            # [CLASS DOC] HELPER: Empty class that gets and loses an attribute.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Sets class attribute y, deletes it, and returns the input
            # plus one.
            MyClass.y = 42
            del MyClass.y
            return x + 1

        result = torch.compile(fn, backend="eager")(torch.tensor(1))
        self.assertEqual(result, torch.tensor(2))
        self.assertFalse(hasattr(MyClass, "y"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
