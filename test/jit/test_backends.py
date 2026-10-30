# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the JIT to-backend lowering machinery using two test backends: a plain
# test backend (library libjitbackend_test) and a demo backend with a compiler (library
# libbackend_with_compiler). Defines small modules, lowering helper functions, base test cases
# that compare Python, scripted, lowered (and mobile) outputs, and wrapper test cases that drive
# the subclass tests so test_jit need not import each. Import time side effect: extends sys.path.
# Most classes are skipped on Windows, macOS, Sandcastle and fbcode.

import io
import os
import sys
import unittest

import torch
import torch._C
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    raise_on_run_directly,
)
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


def to_test_backend(module, method_compile_spec):
    # [FUNCTION DOC] HELPER: Lowers a module to the test backend, using the given compile spec as
    # the spec for the forward method only; returns the lowered module.
    return torch._C._jit_to_backend(
        "test_backend", module, {"forward": method_compile_spec}
    )


def to_test_backend_multi(module, method_compile_spec):
    # [FUNCTION DOC] HELPER: Lowers a module to the test backend with a full mapping from method
    # name to compile spec; returns the lowered module.
    return torch._C._jit_to_backend("test_backend", module, method_compile_spec)


def to_test_backend_selective(module, method_compile_spec, submodules):
    # [FUNCTION DOC] HELPER: Selectively lowers the named submodules of a scripted module to the
    # test backend, using a per-submodule lowering callback built from the given forward compile
    # spec; returns the resulting module.
    def _to_test_backend(module):
        # [FUNCTION DOC] HELPER: Callback that lowers one submodule with the enclosing
        # compile spec.
        return to_test_backend(module, method_compile_spec)

    return torch._C._jit_to_backend_selective(module, _to_test_backend, submodules)


class BasicModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Tiny module with forward, accum (addition) and sub_accum (subtraction)
    # methods; forward returns both results as a tuple.
    """
    A simple Module used to test to_backend lowering machinery.
    """

    def forward(self, x, h):
        # [FUNCTION DOC] HELPER: Returns a tuple of the accumulate result and the subtract result.
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
        return x + h

    def sub_accum(self, x, h):
        # [FUNCTION DOC] HELPER: Returns the difference of the two inputs.
        return x - h


# This is ignored in IS_WINDOWS or IS_MACOS cases. Hence we need the one in TestBackends.
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class JitBackendTestCase(JitTestCase):
    # [CLASS DOC] HELPER: Base test case for plain test backend tests. setUp loads the backend
    # library; subclasses must set a Python module, a scripted module and a lowered module.
    # Provides output comparison and save/load helpers and empty stub tests.
    """
    A common base class for JIT backend tests that contains common utility
    functions for output comparison and serialization/deserialization.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp and loads the shared library of the test
        # backend found by the library location helper into the torch ops registry.
        super().setUp()
        lib_file_path = find_library_location("libjitbackend_test.so")
        torch.ops.load_library(str(lib_file_path))
        # Subclasses are expected to set up three variables in their setUp methods:
        # module - a regular, Python version of the module being tested
        # scripted_module - a scripted version of module
        # lowered_module - a version of module lowered to a backend

    def check_function(self, function_name, input):
        # [FUNCTION DOC] HELPER: Runs the named method on the Python, scripted and lowered modules
        # with the same inputs and asserts that all three outputs are equal.
        """
        Check that the function named 'function_name' produces the same output using
        Python, regular JIT and the backend for the given 'input'.
        """
        # Get handles for Python, JIT and backend methods.
        python_method = self.module.__getattribute__(function_name)
        jit_method = self.scripted_module.__getattr__(function_name)
        backend_method = self.lowered_module.__getattr__(function_name)

        # Run methods.
        python_output = python_method(*input)
        jit_output = jit_method(*input)
        backend_output = backend_method(*input)

        # The answers returned by Python, JIT and to_backend should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)

    def save_load(self):
        # [FUNCTION DOC] HELPER: Replaces the lowered module with a copy produced by exporting and
        # re-importing it.
        """
        Save and load the lowered module.
        """
        self.lowered_module = self.getExportImportCopy(self.lowered_module)

    def test_execution(self):
        # [FUNCTION DOC] Stub for correctness tests in subclasses; does nothing here.
        """
        Stub for correctness tests.
        """

    def test_save_load(self):
        # [FUNCTION DOC] Stub for serialization tests in subclasses; does nothing here.
        """
        Stub for serialization tests.
        """

    def test_errors(self):
        # [FUNCTION DOC] Stub for error checking tests in subclasses; does nothing here.
        """
        Stub for testing error checking.
        """


class BasicModuleTest(JitBackendTestCase):
    # [CLASS DOC] Tests lowering of the basic module with all three methods lowered.
    """
    Tests for BasicModule.
    """

    def setUp(self):
        # [FUNCTION DOC] Creates the Python, scripted and lowered versions of the basic module,
        # lowering accum, sub_accum and forward with empty compile specs.
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()
        self.scripted_module = torch.jit.script(BasicModule())
        self.lowered_module = to_test_backend_multi(
            self.scripted_module,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )

    def test_execution(self):
        # [FUNCTION DOC] Compares Python, scripted and lowered outputs of all three methods on one
        # random vector input.
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.check_function("accum", (input, input))
        self.check_function("sub_accum", (input, input))
        self.check_function("forward", (input, input))

    def test_save_load(self):
        # [FUNCTION DOC] Runs execution, saves the stored compile spec, round trips the lowered
        # module through save and load, asserts the compile spec is unchanged and execution
        # still matches.
        # Lowered module should produce the same outputs.
        self.test_execution()

        # Save the compile spec to compare against the version retrieved after loading.
        pre_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")

        # Save and load the lowered module.
        self.save_load()

        # Get the compile spec after loading.
        post_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")

        # Compile specs should match.
        self.assertEqual(pre_compile_spec, post_compile_spec)

        # Loaded module should produce the same outputs.
        self.test_execution()


class BasicModuleUnavailableTest(JitBackendTestCase):
    # [CLASS DOC] Tests lowering to a backend that is unavailable: lowering succeeds, running
    # raises, saving works and loading raises.
    """
    Tests for BasicModule with a backend that is not available.
    Fundamentally:
      * _jit_to_backend is successful.
      * Execution fails with an exception.
      * Saving is successful.
      * Loading fails with an exception.
    """

    def setUp(self):
        # [FUNCTION DOC] Creates the Python and scripted modules and lowers the scripted one to
        # the unavailable test backend with an empty forward compile spec.
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()
        self.scripted_module = torch.jit.script(BasicModule())
        self.lowered_module = torch._C._jit_to_backend(
            "test_backend_unavailable",
            self.scripted_module,
            {"forward": {"": ""}},
        )

    def test_execution(self):
        # [FUNCTION DOC] Asserts that calling forward on the lowered module raises an exception
        # whose message says the backend is not available.
        # Test execution with backend fails because the backend that is not available.
        input = torch.randn(5)

        # Test exception is thrown.
        with self.assertRaisesRegexWithHighlight(
            Exception,
            r"Backend is not available.",
            'raise Exception("Backend is not available."',
        ):
            backend_method = self.lowered_module.__getattr__("forward")
            backend_method(*(input, input))

    def test_save_load(self):
        # [FUNCTION DOC] Saves the lowered module to an in-memory buffer (succeeds) and asserts
        # that loading it back raises an exception saying the backend is not available.
        # Test that saving the lowered module is OK but loading fails because the backend is not available.
        buffer = io.BytesIO()
        torch.jit.save(self.lowered_module, buffer)
        buffer.seek(0)
        with self.assertRaisesRegexWithHighlight(
            Exception,
            r"Backend is not available.",
            'raise Exception("Backend is not available."',
        ):
            torch.jit.load(buffer)


class NestedModuleTest(JitBackendTestCase):
    # [CLASS DOC] Tests that a lowered module can be used as the submodule of another module.
    """
    Tests for NestedModule that check that a module lowered to a backend can be used
    as a submodule.
    """

    class NestedModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with one submodule whose forward delegates to the submodule.
        """
        A Module with one submodule that is used to test that lowered Modules
        can be used as submodules.
        """

        def __init__(self, submodule):
            # [FUNCTION DOC] HELPER: Stores the submodule.
            super().__init__()
            self.submodule = submodule

        def forward(self, x, h):
            # [FUNCTION DOC] HELPER: Returns the submodule forward result for the inputs.
            return self.submodule.forward(x, h)

    def setUp(self):
        # [FUNCTION DOC] Builds Python and scripted nested modules, lowers a separate basic module
        # with all methods, and scripts a nested module around it as the lowered module.
        super().setUp()
        # Create Python, JIT and backend versions of NestedModule.
        # Both modules in self.module are regular Python modules.
        self.module = NestedModuleTest.NestedModule(BasicModule())
        # Both modules in self.scripted_module are ScriptModules.
        self.scripted_module = torch.jit.script(
            NestedModuleTest.NestedModule(BasicModule())
        )

        # First, script another instance of NestedModule with share_types=False so that it can be
        # selectively lowered without modifying the type of self.scripted_module.
        lowered_module = to_test_backend_multi(
            torch.jit.script(BasicModule()),
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )
        # self.lowered_module is a ScriptModule, but its submodule is a lowered module.
        self.lowered_module = torch.jit.script(
            NestedModuleTest.NestedModule(lowered_module)
        )

    def test_execution(self):
        # [FUNCTION DOC] Compares the forward outputs of the Python, scripted and lowered modules.
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test forward.
        self.check_function("forward", (input, input))

    def test_save_load(self):
        # [FUNCTION DOC] Checks execution, round trips the lowered module through save and load,
        # and checks execution again.
        # Lowered module should produce the same outputs.
        self.test_execution()

        # Save and load the lowered module.
        self.save_load()

        # Loaded module should produce the same outputs.
        self.test_execution()


class SelectiveLoweringTest(JitBackendTestCase):
    # [CLASS DOC] Tests the selective lowering API on a three-way module hierarchy where two basic
    # modules are lowered and one is left alone.
    """
    Tests for the selective lowering API.
    """

    class OuterModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Top module with three submodules (sub1, sub2, other).
        def __init__(self, sub1, sub2, other):
            # [FUNCTION DOC] HELPER: Stores the three submodules.
            super().__init__()
            self.sub1 = sub1
            self.sub2 = sub2
            self.other = other

        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Calls the first submodule's inner module directly, then the
            # second and third submodules, and returns the elementwise sums of the paired results.
            # Call the module that will be lowered directly to test
            # type remapping in modules that are not its parent.
            a, b = self.sub1.submodule.forward(x, y)
            c, d = self.sub2.forward(x, y)
            e, f = self.other.forward(x, y)
            return a + c + e, b + d + f

    class MiddleModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Wrapper module holding a single submodule.
        def __init__(self, submodule):
            # [FUNCTION DOC] HELPER: Stores the submodule.
            super().__init__()
            self.submodule = submodule

        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Returns the submodule forward result.
            return self.submodule.forward(x, y)

    def setUp(self):
        # [FUNCTION DOC] Builds the hierarchy three times (Python, scripted and lowered).
        # Scripting is done without type sharing so lowering one instance does not alter the
        # others; then selectively lowers the inner modules of sub1 and sub2 on the lowered copy.
        super().setUp()
        OuterModule = SelectiveLoweringTest.OuterModule
        MiddleModule = SelectiveLoweringTest.MiddleModule

        def script_without_type_sharing(mod):
            # [FUNCTION DOC] HELPER: Scripts a module without sharing types between instances.
            return torch.jit._recursive.create_script_module(
                mod, torch.jit._recursive.infer_methods_to_compile, share_types=False
            )

        # Create Python, JIT and backend versions of a hierarchy that looks like this:
        #                 --------- OuterModule --------
        #                 |              |              |
        #           MiddleModule    MiddleModule   MiddleModule
        #                |               |              |
        #           BasicModule     BasicModule    BasicModule
        #
        # Two BasicModules will be lowered and the third will not.
        self.module = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )
        self.scripted_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        self.lowered_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        self.lowered_module = to_test_backend_selective(
            self.lowered_module, {"forward": ""}, ["sub1.submodule", "sub2.submodule"]
        )

    def test_execution(self):
        # [FUNCTION DOC] Compares forward outputs across the three versions and then runs the type
        # remap check.
        input = torch.randn(5)
        self.check_function("forward", (input, input))

        self.test_selective_lowering_type_remap()

    def test_save_load(self):
        # [FUNCTION DOC] Runs execution, round trips through save and load, runs execution again,
        # and reruns the type remap check.
        self.test_execution()
        self.save_load()
        self.test_execution()

        self.test_selective_lowering_type_remap()

    def test_selective_lowering_type_remap(self):
        # [FUNCTION DOC] Inspects graphs with file checks: the unlowered scripted module keeps
        # basic modules; the lowered module's outer and middle graphs refer to the lowered
        # wrapper; the lowered inner modules of sub1 and sub2 mention the backend class; the other
        # branch is left untouched.
        """
        Check that type remapping and replacement occurred during selective lowering.
        """
        # Check that self.lowered_module was not lowered, but that it does contain test_backendLoweredModule due to it
        # calling the lowered module directly.
        FileCheck().check("OuterModule").check("BasicModule").run(
            self.scripted_module.graph
        )
        FileCheck().check("OuterModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.graph)

        # Check that self.lowered_module.sub1/sub2 were not lowered but that BasicModule has been replaced in their graphs.
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub1.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub1.graph)

        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub2.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub2.graph)

        # Check that self.lowered_module.sub1/sub2.submodule were lowered. They should have a new attribute
        # __loweredModule__ whose graph should mention __torch__.torch.classes.__backends__.test_backend,
        # the TorchBind class for executing functions on the test JIT backend.
        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub1.submodule.__loweredModule__.graph)

        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub2.submodule.__loweredModule__.graph)

        # Check that self.other and self.other.submodule have been left untouched by the selective lowering process.
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredWrapper.test_backend").run(self.scripted_module.other.graph)
        FileCheck().check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredModule.test_backend").run(
            self.scripted_module.other.submodule.graph
        )

    def test_errors(self):
        # [FUNCTION DOC] Asserts selective lowering raises RuntimeError for a non script module,
        # for a named attribute that is not a module, and for hierarchies without unique types.
        """
        Check errors associated with selective lowering.
        """
        # Check error messages thrown when attempting to lower something that is not a ScriptModule.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Object .* is not a ScriptModule", ""
        ):
            to_test_backend_selective(torch.nn.ReLU(), {"forward": ""}, ["submodule"])

        MiddleModule = SelectiveLoweringTest.MiddleModule
        mod = MiddleModule(BasicModule())
        mod.new_attr = 3

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute named new_attr is not a Module", ""
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["new_attr"]
            )

        # Check error message thrown when module hierarchy doesn't have unique types.
        OuterModule = SelectiveLoweringTest.OuterModule
        mod = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Selective lowering is only supported for module hierarchies with unique types",
            "",
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["sub1.submodule"]
            )


# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class TestBackends(JitTestCase):
    # [CLASS DOC] Wrapper test case that owns instances of the plain backend test classes and
    # calls their tests so test_jit need not import each one; skipped on unsupported platforms.
    """
    This class wraps and invokes all subclasses of JitBackendTestCase so that each one
    does not have to be individually imported in test_jit.py.
    """

    def __init__(self, name):
        # [FUNCTION DOC] Creates the four wrapped test instances with the same test name.
        super().__init__(name)
        self.basic_module_test = BasicModuleTest(name)
        self.basic_module_unavailable_test = BasicModuleUnavailableTest(name)
        self.nested_module_test = NestedModuleTest(name)
        self.selective_lowering_test = SelectiveLoweringTest(name)

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and each wrapped test's setUp.
        super().setUp()
        self.basic_module_test.setUp()
        self.basic_module_unavailable_test.setUp()
        self.nested_module_test.setUp()
        self.selective_lowering_test.setUp()

    def test_execution(self):
        # [FUNCTION DOC] Runs the execution test of each of the four wrapped test cases.
        self.basic_module_test.test_execution()
        self.basic_module_unavailable_test.test_execution()
        self.nested_module_test.test_execution()
        self.selective_lowering_test.test_execution()

    def test_save_load(self):
        # [FUNCTION DOC] Runs the save/load test of each of the four wrapped test cases.
        self.basic_module_test.test_save_load()
        self.basic_module_unavailable_test.test_save_load()
        self.nested_module_test.test_save_load()
        self.selective_lowering_test.test_save_load()

    def test_errors(self):
        # [FUNCTION DOC] Runs only the error test of the selective lowering test case.
        self.selective_lowering_test.test_errors()


"""
Unit Tests for backend with compiler
This test case and the existing TestBackends are separate because they cover different aspects.
The actual backend implementation in this test is different.
It has a simple demo compiler to test the end-to-end flow in mobile.
However, this test cannot cover the selective_lowering for now, which is covered in TestBackends.
"""


class BasicModuleAdd(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward adds its two inputs.
    """
    A simple add Module used to test to_backend lowering machinery.
    """

    def forward(self, x, h):
        # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
        return x + h


# This is ignored in IS_WINDOWS or IS_MACOS cases. Hence we need the one in TestBackends.
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class JitBackendTestCaseWithCompiler(JitTestCase):
    # [CLASS DOC] HELPER: Base test case for the compiler backend. setUp loads the backend
    # library; subclasses must set Python, scripted, lowered and mobile modules.
    """
    A common base class for JIT backend tests with compilers that contains common utility
    functions for output comparison.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setUp and loads the shared library of the
        # compiler backend into the torch ops registry.
        super().setUp()
        lib_file_path = find_library_location("libbackend_with_compiler.so")
        torch.ops.load_library(str(lib_file_path))
        # Subclasses are expected to set up four variables in their setUp methods:
        # module - a regular, Python version of the module being tested
        # scripted_module - a scripted version of module
        # lowered_module - a version of module lowered to a backend
        # mobile_module - a module with a format that Pytorch Mobile can execute

    def check_forward(self, input):
        # [FUNCTION DOC] HELPER: Runs forward on the Python, scripted, lowered and mobile modules
        # for the same input tuple and asserts all outputs match.
        """
        Check that the forward function produces the same output using
        Python, regular JIT, the backend, and mobile for the given 'input'.
        """

        # Get outputs from forward.
        python_output = self.module.forward(*input)
        jit_output = self.scripted_module.forward(*input)
        backend_output = self.lowered_module(*input)
        mobile_output = self.mobile_module(*input)

        # The answers returned by Python, JIT, to_backend, and mobile should all match.
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)
        self.assertEqual(mobile_output, backend_output)

    def test_execution(self):
        # [FUNCTION DOC] Stub for correctness tests; does nothing here.
        """
        Stub for correctness tests.
        """

    def test_errors(self):
        # [FUNCTION DOC] Stub for error tests; does nothing here.
        """
        Stub for testing error checking.
        """


class BasicModuleTestWithCompiler(JitBackendTestCaseWithCompiler):
    # [CLASS DOC] Tests lowering the add module with the compiler backend and comparing
    # with mobile.
    """
    Tests for BasicModuleAdd.
    """

    def setUp(self):
        # [FUNCTION DOC] Creates Python and scripted adds, lowers with a compile spec of input
        # shapes and an extra option, serializes the lowered module for the lite interpreter into
        # a buffer and loads it as the mobile module.
        super().setUp()
        # Create Python, JIT and backend versions of BasicModuleAdd.
        self.module = BasicModuleAdd()
        self.scripted_module = torch.jit.script(BasicModuleAdd())
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        self.lowered_module = torch._C._jit_to_backend(
            "backend_with_compiler_demo", self.scripted_module, compile_spec
        )
        # Create mobile version of BasicModuleAdd
        buffer = io.BytesIO(self.lowered_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # [FUNCTION DOC] Runs forward comparison on a single ones input used for both arguments.
        # Test execution with backend against Python and JIT.
        input = torch.ones(1, dtype=torch.float)
        self.check_forward((input, input))


class ErrorMessagesWithCompiler(JitBackendTestCase):
    # [CLASS DOC] Tests error messages from the compiler backend when an operator is unsupported.
    """
    Tests for errors that occur with compiler, specifically:
        * an operator is not supported by the backend
    """

    class ModuleNotSupported(torch.nn.Module):
        # [CLASS DOC] HELPER: Module whose forward multiplies, an operator the demo
        # compiler rejects.
        """
        A module with an operator that is not supported.
        """

        def forward(self, x, h):
            # [FUNCTION DOC] HELPER: Returns the product of the inputs (a following line
            # is unreachable).
            # KNOWN ISSUE: the expected error text in test_errors embeds this method's source
            # lines, so any comment added in this body changes the source shown and makes that
            # expectation fail.
            return x * h
            self._loweredmodule.forward()

    def test_errors(self):
        # [FUNCTION DOC] Asserts lowering the multiply module raises RuntimeError whose
        # message names the unsupported operator and shows the source context with the
        # highlighted multiply.
        scripted_module_n = torch.jit.script(
            ErrorMessagesWithCompiler.ModuleNotSupported()
        )
        # Test exception is thrown when lowering a module with an unsupported operator
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            # Special escape characters are replaced with '.'
            r"""The node of aten::mul is not supported in this compiler. .*
        def forward.self, x, h.:
            return x . h
                   ~~~~~ <--- HERE
            self._loweredmodule.forward..
""",
            "",
        ):
            torch._C._jit_to_backend(
                "backend_with_compiler_demo", scripted_module_n, {"forward": {"": ""}}
            )


class CompModuleTestWithCompiler(JitBackendTestCase):
    # [CLASS DOC] Tests a module with two lowered submodules (addition and subtraction).
    """
    Tests for CompModule, which is a module with two lowered submodules
    """

    class BasicModuleSub(torch.nn.Module):
        # [CLASS DOC] HELPER: Module whose forward subtracts the second input from the first.
        """
        A simple subtraction Module to be used in CompModule.
        """

        def forward(self, x, h):
            # [FUNCTION DOC] HELPER: Returns the difference of the inputs.
            return x - h

    class CompModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module holding a lowered add and a lowered subtract submodule.
        """
        A module with two lowered submodules.
        """

        def __init__(self, addmodule, submodule):
            # [FUNCTION DOC] HELPER: Stores the two lowered submodules.
            super().__init__()
            self.lowered_add = addmodule
            self.lowered_sub = submodule

        def forward(self, a, b, s):
            # [FUNCTION DOC] HELPER: Calls both lowered submodules and returns the third argument
            # times the product of their results.
            c = self.lowered_add.forward(a, b)
            d = self.lowered_sub.forward(a, b)
            y = s * (c * d)
            return y

    def setUp(self):
        # [FUNCTION DOC] Lowers an add module (with a shape compile spec) and a subtract module,
        # builds Python and scripted composite modules, uses the scripted module as the stand-in
        # lowered module, and creates a mobile module from its lite interpreter serialization.
        super().setUp()
        # Create Python and JIT versions of CompModule with lowered submodules.
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        lowered_add = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(BasicModuleAdd()),
            compile_spec,
        )
        lowered_sub = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(CompModuleTestWithCompiler.BasicModuleSub()),
            {"forward": {"": ""}},
        )
        self.module = CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)
        self.scripted_module = torch.jit.script(
            CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)
        )
        # No backend version of CompModule currently, so this is filler.
        self.lowered_module = self.scripted_module
        # Create a mobile version of CompModule from JIT version
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # [FUNCTION DOC] Compares forward outputs on ones inputs across the Python, scripted and
        # stand-in lowered modules.
        # Test execution with backend against Python and JIT.
        input1 = torch.ones(1, dtype=torch.float)
        input2 = torch.ones(1, dtype=torch.float)

        # Test forward.
        self.check_function("forward", (input1, input2, input2))


# This is needed for IS_WINDOWS or IS_MACOS to skip the tests.
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class TestBackendsWithCompiler(JitTestCase):
    # [CLASS DOC] Wrapper test case that owns and invokes the compiler backend test classes;
    # skipped on unsupported platforms.
    """
    This class wraps and invokes all subclasses of JitBackendTestCaseWithCompiler
    so that each one does not have to be individually imported in test_jit.py.
    """

    def __init__(self, name):
        # [FUNCTION DOC] Creates the basic, error and composite compiler test instances with the
        # same name.
        super().__init__(name)
        self.basic_module_compiler_test = BasicModuleTestWithCompiler(name)
        self.error_module_compiler_test = ErrorMessagesWithCompiler(name)
        self.comp_module_compiler_test = CompModuleTestWithCompiler(name)

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and then each wrapped compiler test's setUp.
        super().setUp()
        self.basic_module_compiler_test.setUp()
        self.error_module_compiler_test.setUp()
        self.comp_module_compiler_test.setUp()

    def test_execution(self):
        # [FUNCTION DOC] Runs the execution test of the basic and composite compiler test cases.
        self.basic_module_compiler_test.test_execution()
        self.comp_module_compiler_test.test_execution()

    def test_errors(self):
        # [FUNCTION DOC] Runs the error test of the error message test case.
        self.error_module_compiler_test.test_errors()


class CompModuleTestSameNameWithCompiler(JitBackendTestCase):
    # [CLASS DOC] Intended to test a composite module whose two lowered submodules share the same
    # module name.
    # KNOWN ISSUE: the nested module classes are referenced by bare names that are not defined at
    # module scope (suppressed with F821), so setUp would raise a name error, and lowered_module
    # is never set although the execution test needs it.
    """
    Tests for CompModule, which is a module with two lowered submodules with same module name
    """

    class ModuleAdd(torch.nn.Module):
        # [CLASS DOC] HELPER: Module whose forward adds its inputs.
        """
        A simple Module used to test to_backend lowering machinery.
        """

        def forward(self, x, h):
            # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
            return x + h

    class CompModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with two lowered copies of the add module, named add and sub.
        """
        A module with two lowered submodules.
        """

        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Lowers two scripted add modules with the compiler backend
            # using one compile spec and stores them.
            # KNOWN ISSUE: refers to an undefined global name for the add module.
            super().__init__()
            compile_spec = {
                "forward": {
                    "some_other_option": "True",
                },
            }
            self.add = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )
            self.sub = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )

        def forward(self, a, b, s: int):
            # [FUNCTION DOC] HELPER: Calls both lowered submodules and returns the integer
            # argument times the product of their results.
            c = self.add.forward(a, b)
            d = self.sub.forward(a, b)
            y = s * (c * d)
            return y

    def setUp(self):
        # [FUNCTION DOC] Builds the composite module (by an undefined global name, see class
        # note), scripts it, and loads a mobile module from its lite interpreter serialization.
        super().setUp()

        self.module = CompModule()  # noqa: F821
        self.scripted_module = torch.jit.script(self.module)
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # [FUNCTION DOC] Compares forward outputs on tensor inputs and an integer scale.
        a = torch.ones(1)
        b = 3 * torch.ones(1)
        s = 3
        # Test forward.
        self.check_function("forward", (a, b, s))


class AddedAttributesTest(JitBackendTestCase):
    # [CLASS DOC] Tests adding attributes (bundled inputs) to a lowered model.
    """
    Tests for adding attributes to a model after lowering.
    """

    def setUp(self):
        # [FUNCTION DOC] Creates Python, scripted and lowered versions of the basic module,
        # lowering all three methods.
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()
        self.scripted_module = torch.jit.script(BasicModule())
        self.lowered_module = to_test_backend_multi(
            self.scripted_module,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )

    def test_attribute(self):
        # [FUNCTION DOC] Runs the lowered module, augments it with bundled inputs, runs it on the
        # bundled inputs, round trips through save and load, and asserts outputs are preserved.
        # KNOWN ISSUE: the augment call passes a bare name lowered_module (undefined, suppressed
        # with F821) instead of the instance attribute, so it would raise a name error.
        input = [(torch.ones(5),)]
        pre_bundled = self.lowered_module(*input[0])
        # Attach bundled inputs which adds several attributes and functions to the model
        self.lowered_module = (
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                lowered_module,  # noqa: F821
                input,
            )
        )
        post_bundled = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        # Save and load the lowered module.
        self.save_load()
        # Use bundled after save and load to prove its preserved
        post_load = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        self.assertEqual(pre_bundled, post_bundled)
        self.assertEqual(post_bundled, post_load)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
