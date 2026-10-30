# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] JIT tests for recursive scripting of nn.Module hierarchies and plain
# functions: attribute handling (functions, Final constants, typed and untyped values,
# None submodules), method calls, error stacks naming the chain of compiled callees,
# ignored classes and methods, the prepare-scriptable hook, container modules, and a
# thread-safety check for call stacks. Importing appends the test directory to
# sys.path; direct execution redirects to test/test_jit.py.

import os
import re
import sys
import threading
import types
import typing
import typing_extensions
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.jit.frontend
import torch.nn as nn
from torch import Tensor
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import (
    _tmp_donotuse_dont_inline_everything,
    JitTestCase,
)


class TestRecursiveScript(JitTestCase):
    # [CLASS DOC] Test case holding all recursive-scripting tests. Most tests define small
    # nn.Module classes inline and call checkModule to compare eager and scripted behavior; error
    # tests inspect exception text with FileCheck or highlight assertions.
    def test_inferred_nonetype(self):
        # [FUNCTION DOC] Module whose attribute is None, with forward asserting it is None;
        # scripts it and runs checkModule.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a None attribute that forward asserts is None.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the attribute to None.
                super().__init__()
                self.x = None

            def forward(self):
                # [FUNCTION DOC] HELPER: Asserts the attribute is None.
                assert self.x is None  # noqa: S101

        m = torch.jit.script(M())
        self.checkModule(M(), ())

    def test_script_function_attribute(self):
        # [FUNCTION DOC] Scripted free functions stored as module attributes and called from
        # forward; checkModule on two modules with different functions.
        @torch.jit.script
        def fn1(x):
            # [FUNCTION DOC] HELPER: Scripted function returning x plus x.
            return x + x

        @torch.jit.script
        def fn2(x):
            # [FUNCTION DOC] HELPER: Scripted function returning x minus x.
            return x - x

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a function given at construction and calling it
            # in forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the function argument as an attribute.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function on the input.
                return self.fn(x)

        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        self.checkModule(fn1_mod, (torch.randn(2, 2),))
        self.checkModule(fn2_mod, (torch.randn(2, 2),))

    def test_python_function_attribute(self):
        # [FUNCTION DOC] A plain Python builtin (sigmoid) stored as a module attribute and called
        # in forward; checkModule compares eager and scripted results.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a callable and applying it in forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the callable argument as an attribute.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the stored callable to the input.
                return self.fn(x)

        mod = M(torch.sigmoid)

        self.checkModule(mod, (torch.randn(2, 2),))

    def test_failed_function_compilation(self):
        # [FUNCTION DOC] A module attribute function referencing an
        # undefined name must fail scripting with a RuntimeError saying
        # compilation failed, highlighting the undefined name.
        # [FUNCTION DOC] HELPER: Python function that returns an
        # undefined name, so it cannot be compiled.
        # KNOWN ISSUE: placed above the def because the test asserts highlighted source text of
        # the failing body.
        def fn(x):
            return i_dont_exist  # noqa: F821

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a function and calling it in forward.
            def __init__(self, fn):
                # [FUNCTION DOC] HELPER: Stores the function argument as an attribute.
                super().__init__()
                self.fn = fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the stored function.
                return self.fn(x)

        m = M(fn)
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to compile", "i_dont_exist"
        ):
            torch.jit.script(m)

    def test_init_error(self):
        # [FUNCTION DOC] Scripting a module whose init never calls the base initializer must raise
        # a RuntimeError containing the text has not been initialized.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module whose init skips the base class initializer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets an attribute without initializing the base class.
                self.x = 2

            def forward(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        with self.assertRaisesRegex(RuntimeError, "has not been initialized"):
            torch.jit.script(M())

    def test_script_after_eval(self):
        # [FUNCTION DOC] Scripts a module before and after eval mode is set. The first scripted
        # copy keeps training true and returns 2, while the copy made after eval has training
        # false and returns 0; the original module stays in eval. Also checks the Python training
        # flag matches the stored attribute on the underlying script module.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns 2 in training mode and 0 otherwise.
            def forward(self):
                # [FUNCTION DOC] HELPER: Branches on the training flag.
                if self.training:
                    return 2
                else:
                    return 0

        m = M()
        sm1 = torch.jit.script(m)
        m.eval()
        sm2 = torch.jit.script(m)

        # m is in eval mode, training should be False
        self.assertFalse(m.training)

        # sm1 was created while m had training = True
        self.assertTrue(sm1.training)
        self.assertEqual(sm1.training, sm1._c.getattr("training"))
        self.assertEqual(sm1(), 2)

        # sm2 was created after m was eval'ed
        self.assertFalse(sm2.training)
        self.assertEqual(sm2.training, sm2._c.getattr("training"))
        self.assertEqual(sm2(), 0)

    def test_module_name(self):
        # [FUNCTION DOC] Scripts a module and checks its graph
        # text contains the original class name.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding a stored integer to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets an integer attribute to two.
                super().__init__()
                self.x = 2

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the attribute to the input.
                return t + self.x

        m = torch.jit.script(MyModule())
        FileCheck().check("MyModule").run(m.graph)

    def test_repeated_error_stack(self):
        # [FUNCTION DOC] Scripts a function a that calls b, which calls c, which calls d, which
        # does invalid arithmetic on a string, twice in a row. Each failure message must contain
        # two occurrences of the phrase is being compiled, showing the stack is not polluted by
        # the earlier failure.
        # KNOWN ISSUE: the assertions run only inside the except branch, so the test passes
        # silently if scripting does not raise.
        def d(x):
            # [FUNCTION DOC] HELPER: Returns a string minus an integer, an invalid operation.
            return "a" - 2

        def c(x):
            # [FUNCTION DOC] HELPER: Calls d.
            return d(x)

        def b(x):
            # [FUNCTION DOC] HELPER: Calls c.
            return c(x)

        def a(x):
            # [FUNCTION DOC] HELPER: Calls b.
            return b(x)

        try:
            torch.jit.script(a)
        except Exception as e:
            FileCheck().check_count("is being compiled", 2).run(str(e))

        try:
            torch.jit.script(a)
        except Exception as e:
            # Make sure that no entries are left over from the previous failure
            FileCheck().check_count("is being compiled", 2).run(str(e))

    def test_constants_with_final(self):
        # [FUNCTION DOC] Checks the Final annotation from torch.jit, typing_extensions and typing:
        # each module with an integer constant declared Final is run through checkModule.
        class M1(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an int attribute annotated with torch.jit.Final.
            x: torch.jit.Final[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the constant to two.
                super().__init__()
                self.x = 2

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the constant to the input.
                return t + self.x

        self.checkModule(M1(), (torch.randn(2, 2),))

        class M2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an int attribute
            # annotated with typing_extensions Final.
            x: typing_extensions.Final[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the constant to two.
                super().__init__()
                self.x = 2

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the constant to the input.
                return t + self.x

        self.checkModule(M2(), (torch.randn(2, 2),))

        class M3(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an int attribute annotated with typing Final.
            x: typing.Final[int]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the constant to two.
                super().__init__()
                self.x = 2

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the constant to the input.
                return t + self.x

        self.checkModule(M3(), (torch.randn(2, 2),))

    def test_ignore_class(self):
        # [FUNCTION DOC] A class decorated with torch.jit.ignore cannot be instantiated in
        # scripted code: scripting a module that constructs it must raise a FrontendError with the
        # text Cannot instantiate class, highlighting the class name.
        @torch.jit.ignore
        class MyScriptClass:
            # [CLASS DOC] HELPER: Class marked ignored by TorchScript; its method adds a string
            # and an int.
            def unscriptable(self):
                # [FUNCTION DOC] HELPER: Method that would fail if run (string plus int); it is
                # never compiled.
                return "a" + 200

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward instantiates the ignored class.
            # [FUNCTION DOC] HELPER: Returns a new instance of the ignored class.
            # KNOWN ISSUE: placed above the def because the test asserts highlighted source
            # text of the scripted body.
            def forward(self, x):
                return MyScriptClass()

        with self.assertRaisesRegexWithHighlight(
            torch.jit.frontend.FrontendError,
            "Cannot instantiate class",
            "MyScriptClass",
        ):
            t = torch.jit.script(TestModule())

    def test_method_call(self):
        # [FUNCTION DOC] Module forward calling another instance method; checkModule compares
        # eager and scripted.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a helper method and a forward that uses it.
            def test(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument unchanged.
                return x

            def forward(self, z):
                # [FUNCTION DOC] HELPER: Calls the helper and returns the input plus twenty plus
                # the helper result.
                y = self.test(z)
                return z + 20 + y

        self.checkModule(M(), (torch.randn(2, 2),))

    def test_module_repr(self):
        # [FUNCTION DOC] Prints a scripted module with convolution, linear and nested submodules,
        # captures stdout, and checks the printed text names the root and each submodule class;
        # also checks original_name equals the source class name.
        class Submodule(nn.Module):
            # [CLASS DOC] HELPER: Identity submodule.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d, a Linear and a Submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the three submodules.
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)
                self.sub = Submodule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the outputs of the linear layer, the submodule and
                # the convolution.
                return self.lin(x) + self.sub(x) + self.conv(x)

        m = torch.jit.script(MyModule())

        with self.capture_stdout() as out:
            print(m)

        f = FileCheck()
        f.check("MyModule")
        f.check("Conv2d")
        f.check("Linear")
        f.check("Submodule")
        f.run(out[0])

        self.assertEqual(m.original_name, "MyModule")

    def test_dir(self):
        # [FUNCTION DOC] Checks that a scripted module exposes the same names in dir as the eager
        # module, except an ignore list of container methods and training. Run for a custom
        # module, a Sequential and a ModuleDict.
        def test_module_dir(mod):
            # [FUNCTION DOC] HELPER: Compares the dir of an eager module with the dir of its
            # scripted version, skipping ignore-listed names and asserting every other eager name
            # is present in the scripted listing.
            dir_set = dir(mod)
            scripted_mod = torch.jit.script(mod)
            dir_scripted = set(dir(scripted_mod))
            # set not currently copied over
            ignore_set = [
                "training",
                "__delitem__",
                "__setitem__",
                "clear",
                "items",
                "keys",
                "pop",
                "update",
                "values",
            ]
            for attr in dir_set:
                if attr in ignore_set:
                    continue
                self.assertTrue(attr in dir_scripted, attr)

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with a Conv2d and a Linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two layers.
                super().__init__()
                self.conv = nn.Conv2d(10, 10, 3)
                self.lin = nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the linear and convolution outputs.
                return self.lin(x) + self.conv(x)

        test_module_dir(MyModule())

        # test custom __dir__ for containers
        conv = nn.Conv2d(10, 10, 3)
        linear = nn.Linear(10, 10)

        test_module_dir(nn.Sequential(conv, linear))
        test_module_dir(
            nn.ModuleDict(OrderedDict([("conv", conv), ("linear", linear)]))
        )

    def test_class_compile(self):
        # [FUNCTION DOC] Scripts a module that constructs a plain Python class whose method calls
        # a module-level function; checkModule compares results.
        def other_fn(a: int, b: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Multiplies an int and a tensor.
            return a * b

        class B:
            # [CLASS DOC] HELPER: Plain class that stores a constant and has a helper method.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Ignores its argument and sets an attribute to two.
                self.x = 2

            def helper(self, a):
                # [FUNCTION DOC] HELPER: Returns the attribute plus the argument plus other_fn
                # applied to them.
                return self.x + a + other_fn(self.x, a)

        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that builds the plain class and calls its helper.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Constructs the helper object from the input and returns
                # its helper result.
                b = B(x)
                return b.helper(x)

        self.checkModule(N(), (torch.randn(2, 2),))

    def test_error_stack(self):
        # [FUNCTION DOC] Scripts function a through a chain of callees where the innermost call
        # passes a string to an int parameter. If scripting raises RuntimeError, checks the
        # message has the type mismatch text then the source definitions of c, b and a in order.
        # KNOWN ISSUE: the assertions run only inside the except branch, so the test passes
        # silently if scripting does not raise.
        def d(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns its int argument plus ten.
            return x + 10

        # [FUNCTION DOC] HELPER: Calls d with a string (invalid) and with the argument.
        # KNOWN ISSUE: placed above the def because the test checks the quoted definition line in
        # the error text.
        def c(x):
            return d("hello") + d(x)

        # [FUNCTION DOC] HELPER: Calls c.
        # KNOWN ISSUE: placed above the def because the test checks the quoted definition line in
        # the error text.
        def b(x):
            return c(x)

        # [FUNCTION DOC] HELPER: Calls b.
        # KNOWN ISSUE: placed above the def because the test checks the quoted definition line in
        # the error text.
        def a(x):
            return b(x)

        try:
            scripted = torch.jit.script(a)
        except RuntimeError as e:
            checker = FileCheck()
            checker.check("Expected a value of type 'int'")
            checker.check("def c(x)")
            checker.check("def b(x)")
            checker.check("def a(x)")
            checker.run(str(e))

    def test_error_stack_module(self):
        # [FUNCTION DOC] Same as the function error stack test but through modules: M calls a
        # method that calls a submodule whose forward calls b, then c, then d with a bad argument.
        # On a RuntimeError, checks for the type mismatch text and the compile-chain notes for c
        # from b and b from a caller.
        # KNOWN ISSUE: the assertions run only inside the except branch, so the test passes
        # silently if scripting does not raise.
        def d(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns its int argument plus ten.
            return x + 10

        def c(x):
            # [FUNCTION DOC] HELPER: Calls d with a string and with the argument.
            return d("hello") + d(x)

        def b(x):
            # [FUNCTION DOC] HELPER: Calls c.
            return c(x)

        class Submodule(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule whose forward calls the free function b.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls b.
                return b(x)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the submodule and a method that adds its output
            # to the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule.
                super().__init__()
                self.submodule = Submodule()

            def some_method(self, y):
                # [FUNCTION DOC] HELPER: Returns the input plus the submodule output.
                return y + self.submodule(y)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the helper method.
                return self.some_method(x)

        try:
            scripted = torch.jit.script(M())
        except RuntimeError as e:
            checker = FileCheck()
            checker.check("Expected a value of type 'int'")
            checker.check("'c' is being compiled since it was called from 'b'")
            checker.check("'b' is being compiled since it was called from")
            checker.run(str(e))

    @_tmp_donotuse_dont_inline_everything
    def test_script_basic(self):
        # [FUNCTION DOC] With function inlining disabled by the decorator, scripts a function
        # calling a Python function and checks the graph contains a function call rather than an
        # inlined copy, and that results equal tensor addition.
        def a_python_fn(a, b, c):
            # [FUNCTION DOC] HELPER: Adds three values.
            return a + b + c

        @torch.jit.script
        def a_script_fn(d, e, f):
            # [FUNCTION DOC] HELPER: Scripted function calling the Python function.
            return a_python_fn(d, e, f)

        graph = str(a_script_fn.graph)
        FileCheck().check("prim::CallFunction").run(graph)
        FileCheck().check_not("^a_python_fn").run(graph)
        t = torch.ones(2, 2)
        self.assertEqual(a_script_fn(t, t, t), t + t + t)

    def test_error_stack_class(self):
        # [FUNCTION DOC] Scripts a function returning an instance of a class whose method contains
        # an import statement. If scripting raises, checks the message mentions import statements
        # and the chain of callers.
        # KNOWN ISSUE: the assertions run only inside the except branch, so the test passes
        # silently if scripting does not raise.
        class X:
            # [CLASS DOC] HELPER: Class whose method contains an unsupported import statement.
            def bad_fn(self):
                # [FUNCTION DOC] HELPER: Imports pdb, which TorchScript cannot compile.
                import pdb  # noqa: F401

        def fn(x) -> X:
            # [FUNCTION DOC] HELPER: Returns a new instance of the class.
            return X(10)

        try:
            torch.jit.script(fn)
        except Exception as e:
            checker = FileCheck()
            checker.check("import statements")
            checker.check("is being compiled since it was called from")
            checker.run(str(e))

    def test_error_stack_annotation(self):
        # [FUNCTION DOC] Like the previous class error-stack test, but the class appears as the
        # return annotation. If scripting raises, checks for the import statement text, the caller
        # chain, and the annotation text.
        # KNOWN ISSUE: the assertions run only inside the except branch, so the test passes
        # silently if scripting does not raise.
        class X:
            # [CLASS DOC] HELPER: Class whose method contains an unsupported import statement.
            def bad_fn(self):
                # [FUNCTION DOC] HELPER: Imports pdb, which TorchScript cannot compile.
                import pdb  # noqa: F401

        def fn(x) -> X:
            # [FUNCTION DOC] HELPER: Returns a new instance of the class.
            return X(10)

        try:
            torch.jit.script(fn)
        except Exception as e:
            checker = FileCheck()
            checker.check("import statements")
            checker.check("is being compiled since it was called from")
            checker.check("-> X")
            checker.run(str(e))

    def test_module_basic(self):
        # [FUNCTION DOC] Scripts a module whose submodule has a constant, a parameter and a method
        # that rebinds a variable to a different type; checkModule compares with eager. The
        # unusable method is not called so it is never compiled.
        class Other(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a declared constant, a parameter, and an
            # uncompilable helper method.
            __constants__ = ["x"]

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the constant and creates a parameter of ones.
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            def some_unscriptable_method(self):
                # [FUNCTION DOC] HELPER: Reassigns a name from int
                # to list; not scriptable and unused.
                a = 2
                a = [2]
                return a

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the constant and the parameter to the input.
                return t + self.x + self.param

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding the submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule with value 200.
                super().__init__()
                self.other = Other(200)

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Doubles the submodule output.
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_module_function_export(self):
        # [FUNCTION DOC] Like the basic module test, but the submodule also has an exported entry
        # point method; checkModule verifies scripting.
        class Other(torch.nn.Module):
            # [CLASS DOC] HELPER: Submodule with a constant, a
            # parameter and an exported extra method.
            __constants__ = ["x"]

            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the constant and creates a parameter of ones.
                super().__init__()
                self.x = x
                self.param = torch.nn.Parameter(torch.ones(2, 2))

            @torch.jit.export
            def some_entry_point(self, y):
                # [FUNCTION DOC] HELPER: Exported method returning input plus twenty.
                return y + 20

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Adds the constant and the parameter to the input.
                return t + self.x + self.param

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Parent module holding the submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodule with value 200.
                super().__init__()
                self.other = Other(200)

            def forward(self, t):
                # [FUNCTION DOC] HELPER: Doubles the submodule output.
                return self.other(t) * 2

        self.checkModule(M(), (torch.ones(2, 2),))

    def test_iterable_modules(self):
        # [FUNCTION DOC] Scripts a module that iterates over a ModuleList and also calls a nested
        # Sequential, accumulating in place; checkModule compares with eager.
        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding ten.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus ten.
                return x + 10

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a nested Sequential
            # and a ModuleList of the inner module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the container modules.
                super().__init__()
                self.sequential = nn.Sequential(
                    Inner(), Inner(), nn.Sequential(Inner(), Inner())
                )
                self.module_list = nn.ModuleList([Inner(), Inner()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds each list module output to the input in place, then
                # adds the Sequential output, and returns it.
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        self.checkModule(M(), (torch.randn(5, 5),))

    def test_prepare_scriptable_basic(self):
        # [FUNCTION DOC] A module whose prepare-scriptable hook returns a ReLU is scripted; its
        # scripted output must differ from the eager SELU output.
        class SeluButReluWhenScripted(torch.nn.SELU):
            # [CLASS DOC] HELPER: SELU subclass that is replaced by ReLU when scripted.
            def __prepare_scriptable__(self):
                # [FUNCTION DOC] HELPER: Returns a ReLU module to be scripted instead.
                return nn.ReLU()

        t = torch.randn(5, 5)
        m = SeluButReluWhenScripted()
        sm = torch.jit.script(m)
        eager_out = m(t)
        script_out = sm(t)
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_iterable_modules(self):
        # [FUNCTION DOC] Nested Sequential and ModuleList containing replaceable modules,
        # including one shared instance; scripted output must differ from eager output.
        class SeluButReluWhenScripted(torch.nn.SELU):
            # [CLASS DOC] HELPER: SELU subclass that is replaced by ReLU when scripted.
            def __prepare_scriptable__(self):
                # [FUNCTION DOC] HELPER: Returns a ReLU module to be scripted instead.
                return nn.ReLU()

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Sequential and a ModuleList containing replaceable
            # modules and a shared one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the containers, reusing one shared instance in
                # several places.
                super().__init__()
                shared = SeluButReluWhenScripted()
                self.sequential = nn.Sequential(
                    SeluButReluWhenScripted(),
                    SeluButReluWhenScripted(),
                    nn.Sequential(
                        SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()
                    ),
                    shared,
                )
                self.module_list = nn.ModuleList(
                    [SeluButReluWhenScripted(), shared, SeluButReluWhenScripted()]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds each list module output in place, then the
                # Sequential output, and returns it.
                for mod in self.module_list:
                    x += mod(x)
                x += self.sequential(x)
                return x

        t = torch.randn(5, 5)
        m = M()
        eager_out = m(t.clone())
        sm = torch.jit.script(m)
        script_out = sm(t.clone())
        self.assertNotEqual(eager_out, script_out)

    def test_prepare_scriptable_cycle(self):
        # [FUNCTION DOC] Builds two modules referencing each other through
        # their instance dictionaries and checks scripting the pair terminates
        # without error. No assertion beyond completion.
        t = torch.randn(5, 5)
        c = torch.nn.Module()
        p = torch.nn.Module()
        c.__dict__["_p"] = p
        p.__dict__["_c"] = c

        sm = torch.jit.script(p)

    def test_prepare_scriptable_escape_hatch(self):
        # [FUNCTION DOC] Scripting an object whose call takes a variable number of arguments must
        # raise NotSupportedError. A subclass whose prepare-scriptable hook returns a plain
        # two-int function is scriptable, accepts two arguments, and raises RuntimeError when
        # called with four.
        class NonJitableClass:
            # [CLASS DOC] HELPER: Callable object that sums two ints and any extra arguments.
            def __call__(self, int1, int2, *args):
                # [FUNCTION DOC] HELPER: Returns the sum of the two ints and all extra arguments.
                total = int1 + int2
                for arg in args:
                    total += arg
                return total

        obj = NonJitableClass()

        self.assertEqual(obj(1, 2), 3)
        self.assertEqual(obj(1, 2, 3, 4), 10)
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError,
            expected_regex="can't take variable number of arguments",
        ):
            torch.jit.script(obj)

        def escape_hatch(int1: int, int2: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of two ints.
            return int1 + int2

        class NonJitableClassWithEscapeHatch(NonJitableClass):
            # [CLASS DOC] HELPER: Subclass whose hook substitutes the simple function.
            def __prepare_scriptable__(self):
                # [FUNCTION DOC] HELPER: Returns the escape-hatch function.
                return escape_hatch

        jit_obj = torch.jit.script(NonJitableClassWithEscapeHatch())

        self.assertEqual(jit_obj(1, 2), 3)
        with self.assertRaisesRegex(
            RuntimeError,
            expected_regex=re.escape(
                "expected at most 2 argument(s) but received 4 argument(s)"
            ),
        ):
            jit_obj(1, 2, 3, 4)

    def test_prepare_scriptable_already_scripted_with_ignored_child(self):
        # [FUNCTION DOC] Regression test: scripting a wrapper around an
        # already scripted child that has an ignored attribute submodule
        # must succeed and the wrapper output must be zeros.
        # Scripting a wrapper around an already-scripted child used to re-walk that
        # child's _modules and re-assign its __jit_ignored_attributes__ submodule (a
        # non-scripted nn.Module), raising "Cannot re-assign modules in a ScriptModule
        # with non-scripted module". The _modules loop now skips already-scripted
        # children, so re-scripting the wrapper succeeds.
        class IgnoredChild(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that declares an attribute as ignored by scripting, with
            # a scripted-only fast path.
            __jit_ignored_attributes__ = ["sub"]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a Linear layer as the ignored attribute.
                super().__init__()
                self.sub = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns zeros when scripting,
                # otherwise calls the ignored method.
                if torch.jit.is_scripting():
                    return torch.zeros_like(x)
                return self._real(x)

            @torch.jit.ignore
            def _real(self, x):
                # [FUNCTION DOC] HELPER: Ignored method applying the Linear layer.
                return self.sub(x)

        class Wrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module holding an inner module.
            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Stores the inner module.
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner module forward.
                return self.inner.forward(x)

        inner = torch.jit.script(IgnoredChild())
        self.assertIn("sub", dict(inner.named_children()))
        scripted = torch.jit.script(Wrapper(inner))
        t = torch.randn(2, 4)
        self.assertEqual(scripted(t), torch.zeros_like(t))

    def test_attributes(self):
        # [FUNCTION DOC] Sets many attribute kinds on a module instance (dict, float, int, bool,
        # tuple, lists, empty containers, None and scripted class objects), declares types for the
        # empty and object ones via the annotations mapping, and checks forward returns them
        # through checkModule. Contains commented-out cases kept as TODOs.
        @torch.jit.script
        class Inner2:
            # [CLASS DOC] HELPER: Scripted class with a string attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the string attribute.
                self.b = "a string"

        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted class with an int attribute and an inner object.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the int and creates the inner object.
                self.a = 4
                self.inner = Inner2()

        @torch.jit.script
        class SFoo:
            # [CLASS DOC] HELPER: Scripted class with state hooks for save/load.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the int and creates the inner object.
                self.a = 4
                self.inner = Inner2()

            def __setstate__(self, obj: Tuple[int, Inner2]) -> None:
                # [FUNCTION DOC] HELPER: Restores the int and inner object from a pair.
                a, inner = obj
                self.a = a
                self.inner = inner

            def __getstate__(self):
                # [FUNCTION DOC] HELPER: Returns a pair of the int and inner object.
                return (self.a, self.inner)

        untyped_values = (
            ("my_dict", {"I": "am", "a test": "test"}),
            ("my_float", 2.3),
            ("my_int", 99),
            ("my_bool", False),
            ("my_tuple", (1, 2, 3, 4)),
            ("my_list", [(1, 2), (3, 4)]),
            # ('my_tensor', torch.randn(2, 2)),
            ("my_int_list", [1, 2, 3, 4]),
            # ('my_tensor_list', [torch.ones(2, 2) + i for i in range(4)]),
            ("my_bool_list", [True, True, False, True]),
            ("my_float_list", [1.0, 2.0, 3.0, 4.0]),
            ("my_str_list", ["hello", "bye"]),
        )
        typed_values = (
            ("my_empty_list", []),
            ("my_empty_dict", {}),
            ("my_none", None),
            ("my_object", Foo()),
            ("my_object2", SFoo()),
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attributes are set from outside and typed via the
            # annotations mapping.
            # TODO: re-enable this once this test is in a Python 3-only syntax
            # file
            # my_empty_list : List[int]
            # my_empty_dict : Dict[str, int]
            # my_none : Optional[int]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a tuple of most of the attributes, including
                # attributes of the scripted class objects.
                return (
                    self.my_dict,
                    self.my_float,
                    self.my_int,
                    self.my_bool,
                    # self.my_tensor,
                    self.my_int_list,
                    # self.my_tensor_list,
                    self.my_bool_list,
                    self.my_float_list,
                    self.my_str_list,
                    self.my_empty_list,
                    self.my_empty_dict,
                    self.my_none,
                    self.my_object.a,
                    self.my_object.inner.b,
                    self.my_object.a,
                    self.my_object2.inner.b,
                )

        # TODO: as a followup, fix this test
        # We can't define class attributes like we should be doing:
        #   class M(torch.nn.Module):
        #       my_empty_list : List[int]
        #       my_empty_dict : Dict[str, int]
        #       my_none : Optional[int]
        #       my_out_of_line_attribute: List[int] = [1, 2, 3]
        # since there's no string frontend for Python classes (so the `define`)
        # trick doesn't work.
        M.__annotations__ = {
            "my_empty_list": List[int],
            "my_empty_dict": Dict[str, int],
            "my_none": Optional[int],
            "my_object": Foo,
            "my_object2": SFoo,
        }

        m = M()
        for name, value in untyped_values + typed_values:
            setattr(m, name, value)

        self.checkModule(m, (torch.randn(5, 5),))

    def test_function_attribute_in_submodule(self):
        # [FUNCTION DOC] Module whose submodule stores a function attribute and applies a norm
        # module in forward; checkModule compares with eager.
        class N(nn.Module):
            # [CLASS DOC] HELPER: Submodule storing a function and a norm module.
            def __init__(self, norm):
                # [FUNCTION DOC] HELPER: Stores the relu function and the norm module.
                super().__init__()
                self.activation = torch.nn.functional.relu
                self.norm = norm

            def forward(self, src):
                # [FUNCTION DOC] HELPER: Applies the norm module to the input.
                output = src
                output = self.norm(output)
                return output

        class M(nn.Module):
            # [CLASS DOC] HELPER: Parent module holding the submodule.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a ReLU norm and the submodule.
                super().__init__()
                encoder_norm = nn.ReLU()
                self.encoder = N(encoder_norm)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the submodule.
                return self.encoder(x)

        m = M()
        self.checkModule(m, (torch.randn(5, 5),))

    def test_inner_traced_module(self):
        # [FUNCTION DOC] Puts a traced module in a ModuleList held by a scripted model that loops
        # over them; checkModule compares with eager.
        class Dummy(nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Model holding a list of modules and applying each in forward.
            def __init__(self, dummies):
                # [FUNCTION DOC] HELPER: Stores the module list.
                super().__init__()
                self._dummies = dummies

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Collects the output of each module into a list.
                out = []
                for dummy in self._dummies:
                    out.append(dummy(x))
                return out

        dummy = torch.jit.trace(Dummy(), torch.randn(1, 2))
        dummies = nn.ModuleList([dummy])
        model = Model(dummies)
        self.checkModule(model, (torch.rand(5, 5),))

    def test_script_loaded_module(self):
        # [FUNCTION DOC] Exports and re-imports a scripted module, then holds the loaded copy as a
        # submodule of a new module that is checked with checkModule.
        """
        Test that we can hold a loaded ScriptModule as a submodule.
        """

        class Dummy(nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        dummy = torch.jit.script(Dummy())
        dummy = self.getExportImportCopy(dummy)

        class ContainsLoaded(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the loaded script module.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the loaded module.
                super().__init__()
                self.encoder = dummy

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the loaded module.
                return self.encoder(input)

        self.checkModule(ContainsLoaded(), (torch.rand(2, 3),))

    def test_optional_module(self):
        # [FUNCTION DOC] Runs checkModule on a module with a Linear submodule, then sets the
        # submodule to None and runs it again to check scripting handles an optional submodule.
        class Dummy(nn.Module):
            # [CLASS DOC] HELPER: Module with a Linear submodule
            # that forward uses only if present.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Linear layer.
                super().__init__()
                self.foo = nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer if it is not None, otherwise returns
                # the input.
                if self.foo is not None:
                    return self.foo(x)
                return x

        mod = Dummy()
        self.checkModule(mod, (torch.rand(2, 2),))
        mod.foo = None
        self.checkModule(mod, (torch.rand(2, 2),))

    def test_thread_safe_error_stacks(self):
        # [FUNCTION DOC] Regression test for thread-safe call stacks: a worker thread creates a
        # call stack object from a source range factory and stores it; after the thread finishes,
        # the stored object is deleted from the main thread and the list is asserted empty. The
        # comment says earlier code crashed.
        # prior to #160386, this causes a segfault. See [Note: Thread-safe CallStack]
        callstacks = []

        def callstack_creator():
            # [FUNCTION DOC] HELPER: Creates a source range factory and call stack, appends the
            # call stack to the shared list, and deletes the local reference.
            factory = torch._C._jit_tree_views.SourceRangeFactory(
                "source code", "a.py", 1, 0
            )
            x = torch._C.CallStack("a", factory.make_range(1, 0, 1))
            callstacks.append(x)
            del x

        t = threading.Thread(target=callstack_creator)
        t.start()
        t.join()
        del t
        del callstacks[0]
        self.assertTrue(len(callstacks) == 0)

    def test_override_instance_method_ignore(self):
        # [FUNCTION DOC] A per-instance replacement of an ignored method (bound with MethodType)
        # must be what the scripted module calls: the scripted module returns the new value.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an ignored method returning old.
            @torch.jit.ignore
            def i_am_ignored(self):
                # [FUNCTION DOC] HELPER: Ignored method returning old.
                return "old"

        m = M()

        # Override the ignored method by binding a new method to this instance.
        @torch.jit.ignore
        def i_am_ignored(self):
            # [FUNCTION DOC] HELPER: Ignored function returning new, bound to the instance later.
            return "new"

        m.i_am_ignored = types.MethodType(i_am_ignored, m)
        self.assertEqual(m.i_am_ignored(), "new")

        # ScriptModule should correctly reflect the override.
        s = torch.jit.script(m)
        self.assertEqual(s.i_am_ignored(), "new")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
