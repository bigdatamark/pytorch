# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for TorchScript module type sharing: checks that scripting
# nn.Module instances reuses one concrete type when structure matches (attributes,
# constants, submodules, functions) and creates distinct types otherwise. Importing
# appends the test directory to sys.path; running the file directly is rejected via
# raise_on_run_directly.

import io
import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    suppress_warnings,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestTypeSharing(JitTestCase):
    # [CLASS DOC] Test case verifying when scripted modules share a C++ module type. Each
    # test builds module instances, scripts them and compares the types via the two
    # assertion helpers.
    def assertSameType(self, m1, m2):
        # [FUNCTION DOC] HELPER: Scripts m1 and m2 if they are not already ScriptModules,
        # then asserts that both have equal underlying C++ types. Fails the test otherwise.
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertEqual(m1._c._type(), m2._c._type())

    def assertDifferentType(self, m1, m2):
        # [FUNCTION DOC] HELPER: Scripts m1 and m2 if they are not already ScriptModules,
        # then asserts that their underlying C++ types differ. Fails the test if the types
        # are equal.
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertNotEqual(m1._c._type(), m2._c._type())

    def test_basic(self):
        # [FUNCTION DOC] Two modules of the same class holding identical tensor attributes
        # must share a type.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing three constructor arguments as attributes.
            def __init__(self, a, b, c):
                # [FUNCTION DOC] HELPER: Stores the three arguments a, b, c as plain
                # attributes.
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a, b, c)
        self.assertSameType(m1, m2)

    def test_diff_attr_values(self):
        # [FUNCTION DOC] Modules of one class whose tensor attributes hold different values
        # (same shapes and types) must still share a type.
        """
        Types should be shared even if attribute values differ
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing three constructor arguments as attributes.
            def __init__(self, a, b, c):
                # [FUNCTION DOC] HELPER: Stores the three arguments a, b, c as plain
                # attributes.
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        m1 = M(a, b, c)
        m2 = M(a * 2, b * 3, c * 4)
        self.assertSameType(m1, m2)

    def test_constants(self):
        # [FUNCTION DOC] Modules with equal declared constants share a type; changing the
        # constant value yields a different type.
        """
        Types should be shared for identical constant values, and different for different constant values
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a declared constant and a tensor attribute.
            __constants__ = ["const"]

            def __init__(self, attr, const):
                # [FUNCTION DOC] HELPER: Stores attr as an attribute and const as a declared
                # constant.
                super().__init__()
                self.attr = attr
                self.const = const

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the constant value.
                return self.const

        attr = torch.rand(2, 3)
        m1 = M(attr, 1)
        m2 = M(attr, 1)
        self.assertSameType(m1, m2)

        # a different constant value
        m3 = M(attr, 2)
        self.assertDifferentType(m1, m3)

    def test_linear(self):
        # [FUNCTION DOC] Scripted Linear modules with equal sizes share a type; a
        # differently sized Linear gets a different type.
        """
        Simple example with a real nn Module
        """
        a = torch.nn.Linear(5, 5)
        b = torch.nn.Linear(5, 5)
        c = torch.nn.Linear(10, 10)
        a = torch.jit.script(a)
        b = torch.jit.script(b)
        c = torch.jit.script(c)

        self.assertSameType(a, b)
        self.assertDifferentType(a, c)

    def test_submodules(self):
        # [FUNCTION DOC] Parent modules share a type only if their submodule types match;
        # checks both parent and submodule type comparisons across instances with different
        # Linear sizes.
        """
        If submodules differ, the types should differ.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear submodules whose sizes come from
            # constructor arguments.
            def __init__(self, in1, out1, in2, out2):
                # [FUNCTION DOC] HELPER: Creates submod1 and submod2 as Linear layers with
                # the given feature sizes.
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies submod1 then submod2 to the input and
                # returns the result.
                x = self.submod1(x)
                x = self.submod2(x)
                return x

        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        self.assertSameType(a, b)
        self.assertSameType(a.submod1, b.submod1)
        c = M(2, 2, 2, 2)
        self.assertDifferentType(a, c)

        self.assertSameType(b.submod2, c.submod1)
        self.assertDifferentType(a.submod1, b.submod2)

    def test_param_vs_attribute(self):
        # [FUNCTION DOC] The same attribute stored as a Parameter in one module and as a
        # plain tensor in another must produce different types.
        """
        The same module with an `foo` as a parameter vs. attribute shouldn't
        share types
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing one value as the attribute foo.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores foo as an attribute (a Parameter or a tensor
                # depending on the caller).
                super().__init__()
                self.foo = foo

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus the foo attribute.
                return x + self.foo

        as_param = torch.nn.Parameter(torch.ones(2, 2))
        as_attr = torch.ones(2, 2)
        param_mod = M(as_param)
        attr_mod = M(as_attr)
        self.assertDifferentType(attr_mod, param_mod)

    def test_same_but_different_classes(self):
        # [FUNCTION DOC] Two structurally identical modules defined by separate classes A
        # and B must not share a type.
        """
        Even if everything about the module is the same, different originating
        classes should prevent type sharing.
        """

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear submodules and a constant;
            # structurally identical to class B.
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                # [FUNCTION DOC] HELPER: Creates two Linear submodules and sets the constant
                # to 5.
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both submodules in sequence and scales the
                # result by the constant.
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear submodules and a constant;
            # structurally identical to class A.
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                # [FUNCTION DOC] HELPER: Creates two Linear submodules and sets the constant
                # to 5.
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both submodules in sequence and scales the
                # result by the constant.
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        a = A(1, 1, 2, 2)
        b = B(1, 1, 2, 2)
        self.assertDifferentType(a, b)

    def test_mutate_attr_value(self):
        # [FUNCTION DOC] Replacing tensor attribute values on already-created modules
        # (different values) must not change type sharing; the modules are scripted
        # afterwards and compared.
        """
        Mutating the value of an attribute should not change type sharing
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear submodules and a tensor attribute
            # foo.
            def __init__(self, in1, out1, in2, out2):
                # [FUNCTION DOC] HELPER: Creates two Linear submodules and a ones tensor
                # attribute foo.
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both submodules in sequence and adds foo to
                # the result.
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo

        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.foo = torch.ones(2, 2)
        b.foo = torch.rand(2, 2)
        self.assertSameType(a, b)

    def test_assign_python_attr(self):
        # [FUNCTION DOC] Adding a new Python attribute after scripting leaves the types
        # equal, but adding it before scripting makes the attribute part of the type, so the
        # types differ.
        """
        Assigning a new (python-only) attribute should not change type sharing
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear submodules and a tensor attribute
            # foo.
            def __init__(self, in1, out1, in2, out2):
                # [FUNCTION DOC] HELPER: Creates two Linear submodules and a ones tensor
                # attribute foo.
                super().__init__()
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies both submodules in sequence and adds foo to
                # the result.
                x = self.submod1(x)
                x = self.submod2(x)
                return x + self.foo

        # explicitly call script() to freeze the type
        a = torch.jit.script(M(1, 1, 2, 2))
        b = torch.jit.script(M(1, 1, 2, 2))
        a.new_attr = "foo bar baz"
        self.assertSameType(a, b)

        # but if we assign attributes *before* calling script(), the types
        # should be different, since `new_attr` should be turned into a Script
        # attribute
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        a.new_attr = "foo bar baz"
        self.assertDifferentType(a, b)

    def test_failed_attribute_compilation(self):
        # [FUNCTION DOC] Scripting a module whose attribute holds an object that cannot be
        # converted raises RuntimeError about failing to convert the Python type, with the
        # highlighted source text being the attribute access.
        """
        Attributes whose type cannot be inferred should fail cleanly with nice hints
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute foo is a Python builtin that
            # TorchScript cannot type.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a non-convertible object as the attribute
                # foo.
                super().__init__()
                # assign a type we know can't be converted to TorchScript
                self.foo = object

            # [FUNCTION DOC] HELPER: Returns the unconvertible attribute foo.
            # KNOWN ISSUE: placed above the def (not inside the body) because the test
            # asserts on a highlighted source line of this scripted forward.
            def forward(self):
                # try to use it in forward
                return self.foo

        m = M()
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to convert Python type", "self.foo"
        ):
            torch.jit.script(m)

    def test_script_function_attribute_different(self):
        # [FUNCTION DOC] Two modules holding different scripted functions as an attribute
        # must have different types.
        """
        Different functions passed in should lead to different types
        """

        @torch.jit.script
        def fn1(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the input added to itself.
            return x + x

        @torch.jit.script
        def fn2(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the input minus itself.
            return x - x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a callable attribute fn and calling it in
            # forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function on the input and returns
                # the result.
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_builtin_function_same(self):
        # [FUNCTION DOC] Two modules holding the same builtin function (add) as attribute
        # must share a type.
        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a builtin callable fn and applying it to
            # the input twice as arguments.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the stored function with the input as both
                # arguments.
                return self.fn(input, input)

        c1 = Caller(torch.add)
        c2 = Caller(torch.add)

        self.assertSameType(c1, c2)

    def test_builtin_function_different(self):
        # [FUNCTION DOC] Two modules holding different builtin functions (add vs sub) as
        # attribute must have different types.
        class Caller(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a builtin callable fn and applying it to
            # the input twice as arguments.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the stored function with the input as both
                # arguments.
                return self.fn(input, input)

        c1 = Caller(torch.add)
        c2 = Caller(torch.sub)

        self.assertDifferentType(c1, c2)

    def test_script_function_attribute_same(self):
        # [FUNCTION DOC] Two modules holding the same scripted function as attribute must
        # share a type.
        """
        Same functions passed in should lead to same types
        """

        @torch.jit.script
        def fn(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the input added to itself.
            return x + x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a callable attribute fn and calling it in
            # forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function on the input and returns
                # the result.
                return self.fn(x)

        fn1_mod = M(fn)
        fn2_mod = M(fn)

        self.assertSameType(fn1_mod, fn2_mod)

    def test_python_function_attribute_different(self):
        # [FUNCTION DOC] Two modules holding different plain Python functions as attribute
        # must have different types.
        """
        Different functions passed in should lead to different types
        """

        def fn1(x):
            # [FUNCTION DOC] HELPER: Plain function returning the input added to itself.
            return x + x

        def fn2(x):
            # [FUNCTION DOC] HELPER: Plain function returning the input minus itself.
            return x - x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a callable attribute fn and calling it in
            # forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function on the input and returns
                # the result.
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_python_function_attribute_same(self):
        # [FUNCTION DOC] Two modules holding the same plain Python function as attribute
        # must share a type.
        """
        Same functions passed in should lead to same types
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Plain function returning the input added to itself.
            return x + x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a callable attribute fn and calling it in
            # forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the given function as the attribute fn.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function on the input and returns
                # the result.
                return self.fn(x)

        fn1_mod = M(fn)
        fn2_mod = M(fn)

        self.assertSameType(fn1_mod, fn2_mod)

    @suppress_warnings
    def test_tracing_gives_different_types(self):
        # [FUNCTION DOC] Two traces of the same module class (with different example inputs)
        # must always yield different types. Warnings are suppressed.
        """
        Since we can't guarantee that methods are the same between different
        trace runs, tracing must always generate a unique type.
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with data-dependent control flow in forward, which
            # tracing records per run.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Fixture module forward; returns x if its sum
                # exceeds the sum of y, otherwise y.
                if x.sum() > y.sum():
                    return x
                else:
                    return y

        a = torch.jit.trace(M(), (torch.zeros(1, 1), torch.ones(1, 1)))
        b = torch.jit.trace(M(), (torch.ones(1, 1), torch.zeros(1, 1)))
        self.assertDifferentType(a, b)

    def test_ignored_fns(self):
        # [FUNCTION DOC] Modules whose forward calls a method marked ignore share a type
        # even though the attribute used by the ignored method differs in shape; the calls
        # still return their own values.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an attribute foo exposed through an ignored
            # method.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: Stores foo as an attribute.
                super().__init__()
                self.foo = foo

            @torch.jit.ignore
            def ignored(self):
                # [FUNCTION DOC] HELPER: Ignored method returning the attribute foo.
                return self.foo

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the result of the ignored method.
                return self.ignored()

        a = torch.jit.script(M(torch.ones(1)))
        b = torch.jit.script(M(torch.ones(2)))
        self.assertSameType(a, b)
        self.assertNotEqual(a(), b())

    @suppress_warnings
    def test_script_module_containing_traced_module(self):
        # [FUNCTION DOC] Modules containing traced submodules that were traced with
        # different example inputs must have different types. Warnings are suppressed.
        class Traced(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with data-dependent branching, traced by the
            # enclosing module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fixture module forward; returns x if its sum is
                # positive, otherwise x doubled.
                if x.sum() > 0:
                    return x
                else:
                    return x + x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a traced version of the Traced module.
            def __init__(self, input):
                # [FUNCTION DOC] HELPER: Traces a new Traced module with the given example
                # input and stores it as the attribute traced.
                super().__init__()
                self.traced = torch.jit.trace(Traced(), input)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the traced submodule on the input and returns
                # the result.
                return self.traced(x)

        a = M((torch.ones(1),))
        b = M((torch.zeros(1),))
        self.assertDifferentType(a, b)

    def test_loaded_modules_work(self):
        # [FUNCTION DOC] Saves scripted modules to an in-memory buffer, loads them back,
        # wraps the loaded module in another scripted module and runs it, verifying loaded
        # modules with different attributes can be wrapped and called.
        class AB(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two integer attributes a and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets integer attributes a and b to one.
                super().__init__()
                self.a = 1
                self.b = 1

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the sum of a and b.
                return self.a + self.b

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single integer attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets integer attribute a to one.
                super().__init__()
                self.a = 1

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the attribute a.
                return self.a

        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that stores a submodule and calls it.
            def __init__(self, sub):
                # [FUNCTION DOC] HELPER: Stores the given submodule as the attribute sub.
                super().__init__()
                self.sub = sub

            def forward(self):
                # [FUNCTION DOC] HELPER: Calls the submodule and returns its result.
                return self.sub()

        def package(x):
            # [FUNCTION DOC] HELPER: Scripts a module, saves it to an in-memory bytes
            # buffer, rewinds the buffer, loads it back with jit load and returns a scripted
            # Wrapper around the loaded module.
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(x), buffer)
            buffer.seek(0)
            return torch.jit.script(Wrapper(torch.jit.load(buffer)))

        a = package(AB())
        a()
        b = package(A())
        b()

    def test_module_dict_same_type_different_name(self):
        # [FUNCTION DOC] Two ModuleDict holders with different key names but the same value
        # types must have different types, while holders with identical keys share a type.
        """
        We should be able to differentiate between two ModuleDict instances
        that have different keys but the same value types.
        """

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Trivial module used as ModuleDict value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleDict built from a mapping.
            def __init__(self, s):
                # [FUNCTION DOC] HELPER: Creates the ModuleDict attribute from the given
                # mapping.
                super().__init__()
                self.dict = torch.nn.ModuleDict(s)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        a = Foo({"foo": A()})
        b = Foo({"bar": A()})
        c = Foo({"bar": A()})
        self.assertDifferentType(a, b)
        self.assertSameType(b, c)

    def test_type_sharing_define_in_init(self):
        # [FUNCTION DOC] ScriptModule subclass instances that define methods through the
        # define call in their constructor must not share types; each instance returns its
        # own constant.
        """
        Tests that types between instances of a ScriptModule
        subclass that defines methods in its __init__ are not
        shared.
        """

        class A(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule whose forward is defined from source text in
            # the constructor, embedding the given integer.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Calls define with generated source for a forward
                # returning the given value.
                super().__init__()
                self.define(
                    f"""
                def forward(self) -> int:
                    return {val}
                """
                )

        one = A(1)
        two = A(2)

        self.assertEqual(one(), 1)
        self.assertEqual(two(), 2)

    def test_type_sharing_disabled(self):
        # [FUNCTION DOC] With the share_types option off in the recursive script module
        # creation, instances, and nested submodules of the same class, all get different
        # types.
        """
        Test that type sharing can be disabled.
        """

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a submodule.
            def __init__(self, sub):
                # [FUNCTION DOC] HELPER: Stores the given submodule as the attribute sub.
                super().__init__()
                self.sub = sub

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Leaf module with no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        top1 = A(A(B()))
        top2 = A(A(B()))

        top1_s = torch.jit._recursive.create_script_module(
            top1,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )
        top2_s = torch.jit._recursive.create_script_module(
            top2,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )

        self.assertDifferentType(top1_s, top2_s)
        self.assertDifferentType(top1_s, top1_s.sub)
        self.assertDifferentType(top1_s, top2_s.sub)
        self.assertDifferentType(top2_s, top2_s.sub)
        self.assertDifferentType(top2_s, top1_s.sub)

    def test_type_shared_ignored_attributes(self):
        # [FUNCTION DOC] Modules whose differing attribute is listed in the ignored
        # attributes list share a type, since the remaining attribute types match.
        """
        Test that types are shared if the exclusion of their
        ignored attributes makes them equal.
        """

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two attributes where the first is declared as
            # an ignored attribute.
            __jit_ignored_attributes__ = ["a"]

            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores a and b as attributes.
                super().__init__()
                self.a = a
                self.b = b

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        a_with_linear = A(torch.nn.Linear(5, 5), 5)
        a_with_string = A("string", 10)

        # Both should have the same type because the attribute
        # that differs in type is ignored and the common attribute
        # has the same type.
        self.assertSameType(a_with_linear, a_with_string)

    def test_type_not_shared_ignored_attributes(self):
        # [FUNCTION DOC] Scripting the same module twice, with the ignored attributes list
        # changed on the class between the calls, must produce different types. Mutates the
        # class attribute.
        """
        Test that types are not shared if the exclusion of their
        ignored attributes makes them not equal.
        """

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with three attributes; ignored attributes list
            # initially only the first.
            __jit_ignored_attributes__ = ["a"]

            def __init__(self, a, b, c):
                # [FUNCTION DOC] HELPER: Stores a, b, c as attributes.
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Identity: returns the input unchanged.
                return x

        mod = A(torch.nn.Linear(5, 5), 5, "string")
        s1 = torch.jit.script(mod)
        A.__jit_ignored_attributes__ = ["a", "b"]
        s2 = torch.jit.script(mod)

        # The types of s1 and s2 should differ. Although they are instances
        # of A, __jit_ignored_attributes__ was modified before scripting s2,
        # so the set of ignored attributes is different between s1 and s2.
        self.assertDifferentType(s1, s2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
