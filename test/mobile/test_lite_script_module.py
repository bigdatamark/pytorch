# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Lite interpreter tests for TorchScript modules: saving to the mobile format
# (to buffer or file) and loading back, method lookup and calling, optional arguments, error
# messages for unsupported class types, operator list export, mobile debug info and source
# range reporting in runtime errors, interface call stack traces, and quantized and bundled
# input models. No import-time side effects besides imports.
# KNOWN ISSUE: the module imports TemporaryFileName from the standard tempfile module, which
# does not define it, so importing this file raises ImportError as written.

import inspect
import io
from tempfile import TemporaryFileName

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter
from torch.testing import FileCheck
from torch.testing._internal.common_quantization import (
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    QuantizationLiteTestCase,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptModule(TestCase):
    # [CLASS DOC] Main TestCase for the script to lite interpreter round trip. Holds one helper
    # and tests; several tests assert on source line numbers or traceback text of scripted code.
    def getScriptExportImportCopy(
        self, m, save_mobile_debug_info=True, also_test_file=False
    ):
        # [FUNCTION DOC] HELPER: Scripts module m, serializes it in lite interpreter format
        # (optionally with mobile debug info) and loads it back. By default goes through an
        # in-memory buffer; with also_test_file it saves to a temporary file and loads from the
        # file path instead (the buffer path is then not used). Returns the mobile module.
        m_scripted = torch.jit.script(m)

        if not also_test_file:
            buffer = io.BytesIO(
                m_scripted._save_to_buffer_for_lite_interpreter(
                    _save_mobile_debug_info=save_mobile_debug_info
                )
            )
            buffer.seek(0)
            mobile_module = _load_for_lite_interpreter(buffer)
            return mobile_module

        with TemporaryFileName() as fname:
            m_scripted._save_for_lite_interpreter(
                fname, _save_mobile_debug_info=save_mobile_debug_info
            )
            mobile_module = _load_for_lite_interpreter(fname)
            return mobile_module

    def test_load_mobile_module(self):
        # [FUNCTION DOC] Scripts a module adding ten, round trips it to the lite interpreter and
        # asserts calling the mobile module, its forward method and run_method by name all
        # match the scripted result.
        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds ten to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus ten.
                return x + 10

        input = torch.tensor([1])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )

    def test_save_mobile_module_with_debug_info_with_trace(self):
        # [FUNCTION DOC] For both tracing and scripting a module with two child modules that
        # multiply tensors, exports with debug info, asserts the call stack debug map entry is
        # present in the archive, then runs the mobile module with mismatched shapes and
        # asserts the error message names the module hierarchy of the failing child (first A0,
        # then A1).
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module multiplying its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the element-wise product of x and y.
                return x * y

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two child modules of type A, A0 and A1.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two child modules.
                super().__init__()
                self.A0 = A()
                self.A1 = A()

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the sum of A0 applied to x and y and A1
                # applied to y and z.
                return self.A0(x, y) + self.A1(y, z)

        for export_method in ["trace", "script"]:
            x = torch.rand((2, 3))
            y = torch.rand((2, 3))
            z = torch.rand((2, 3))
            if export_method == "trace":
                trace_module = torch.jit.trace(B(), [x, y, z])
            else:
                trace_module = torch.jit.script(B())
            exported_module = trace_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
            buffer = io.BytesIO(exported_module)
            buffer.seek(0)

            if b"callstack_debug_map.pkl" not in exported_module:
                raise AssertionError(
                    "Expected callstack_debug_map.pkl in exported module"
                )

            mobile_module = _load_for_lite_interpreter(buffer)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A0\(A\)::forward.aten::mul",
            ):
                x = torch.rand((2, 3))
                y = torch.rand((8, 10))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A1\(A\)::forward.aten::mul",
            ):
                x = torch.rand((2, 3))
                y = torch.rand((2, 3))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)

    def test_load_mobile_module_with_debug_info(self):
        # [FUNCTION DOC] Same as the basic load test (add five instead of ten) but saves the
        # module with mobile debug info enabled; checks call, forward and run_method results.
        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds five to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus five.
                return x + 5

        input = torch.tensor([3])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )

    def test_find_and_run_method(self):
        # [FUNCTION DOC] Checks method lookup on a mobile module. An identity module has no
        # bundled input method; after augmenting the script module with bundled inputs and
        # re-exporting, find_method reports the bundled input getter, and running forward on
        # the bundled input equals the scripted result.
        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Identity module.
            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Returns the argument unchanged.
                return arg

        input = (torch.tensor([1]),)

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(*input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertFalse(has_bundled_inputs)

        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            script_module, [input], []
        )

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        self.assertTrue(has_bundled_inputs)

        bundled_inputs = mobile_module.run_method("get_all_bundled_inputs")
        mobile_module_result = mobile_module.forward(*bundled_inputs[0])
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_method_calls_with_optional_arg(self):
        # [FUNCTION DOC] Verifies default argument values survive export. The scripted outer
        # module and mobile module agree when called with defaults; passing an explicit value
        # only to the scripted module makes results differ; passing it to both matches again.
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding an optional integer argument (default two).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Plain constructor that only calls the parent one.
                super().__init__()

            # opt arg in script-to-script invocation
            def forward(self, x, two: int = 2):
                # [FUNCTION DOC] HELPER: Returns x plus the argument two.
                return x + two

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning an A child; its forward adds an optional one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the child A module stored as A0.
                super().__init__()
                self.A0 = A()

            # opt arg in Python-to-script invocation
            def forward(self, x, one: int = 1):
                # [FUNCTION DOC] HELPER: Returns the child module applied to x plus the argument.
                return self.A0(x) + one

        script_module = torch.jit.script(B())
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        mobile_module = _load_for_lite_interpreter(buffer)

        input = torch.tensor([5])
        script_module_forward_result = script_module.forward(input)
        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )

        # change ref only
        script_module_forward_result = script_module.forward(input, 2)
        self.assertFalse(
            (script_module_forward_result == mobile_module_forward_result).all().item()
        )

        # now both match again
        mobile_module_forward_result = mobile_module.forward(input, 2)
        torch.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )

    def test_unsupported_classtype(self):
        # [FUNCTION DOC] Scripts a module whose forward instantiates a plain Python class and
        # calls its method; asserts that exporting to the lite format raises RuntimeError
        # with a message recommending a torch.nn.Module based class instead.
        class Foo:
            # [CLASS DOC] HELPER: Ordinary (non-module) class with a method adding two ints.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Empty constructor.
                return

            def func(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: Returns the sum of two integers.
                return x + y

        class MyTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward creates the plain class and calls it.
            def forward(self, arg):
                # [FUNCTION DOC] HELPER: Returns the result of the plain class method for 1 and 2.
                f = Foo()
                return f.func(1, 2)

        script_module = torch.jit.script(MyTestModule())
        with self.assertRaisesRegex(
            RuntimeError,
            r"Workaround: instead of using arbitrary class type \(class Foo\(\)\), "
            r"define a pytorch class \(class Foo\(torch\.nn\.Module\)\)\. "
            r"The problematic type is: ",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_list_with_module_class(self):
        # [FUNCTION DOC] Scripts a module returning a list of a module class; asserts that
        # exporting to the lite format raises RuntimeError with the specific message about
        # lists or dictionaries of pytorch class types being unsupported.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty module subclass used as a list element type.
            pass

        class MyTestModuleForListWithModuleClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo child, whose forward returns it in a list.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo child stored as foo.
                super().__init__()
                self.foo = Foo()

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a list containing the Foo child.
                my_list: list[Foo] = [self.foo]
                return my_list

        script_module = torch.jit.script(MyTestModuleForListWithModuleClass())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with pytorch class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
            r"Workaround\: instead of using pytorch class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_unsupported_return_dict_with_module_class(self):
        # [FUNCTION DOC] Dictionary variant of the previous test: the scripted module returns a
        # dict of int to a module class; exporting raises RuntimeError with the same message.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty module subclass used as a dict value type.
            pass

        class MyTestModuleForDictWithModuleClass(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo child, whose forward returns it in a dict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo child stored as foo.
                super().__init__()
                self.foo = Foo()

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a dict mapping 1 to the Foo child.
                my_dict: dict[int, Foo] = {1: self.foo}
                return my_dict

        script_module = torch.jit.script(MyTestModuleForDictWithModuleClass())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with pytorch class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
            r"Workaround\: instead of using pytorch class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            script_module._save_to_buffer_for_lite_interpreter()

    def test_module_export_operator_list(self):
        # [FUNCTION DOC] Exports a scripted module that creates zeros, an empty_like tensor and
        # a convolution call, loads it, and asserts the exported operator list equals the
        # expected set of four operator names.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with constant weight and bias tensors and no parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a 20x1x5x5 ones weight and 20 ones bias.
                super().__init__()
                self.weight = torch.ones((20, 1, 5, 5))
                self.bias = torch.ones(20)

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Builds zeros and an empty_like tensor and runs a
                # fixed-argument convolution of the input with the stored weight and bias.
                # Returns a tuple of the zeros, the empty_like tensor and the convolution output.
                x1 = torch.zeros(2, 2)
                x2 = torch.empty_like(torch.empty(2, 2))
                x3 = torch._convolution(
                    input,
                    self.weight,
                    self.bias,
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    False,
                    [0, 0],
                    1,
                    False,
                    False,
                    True,
                    True,
                )
                return (x1, x2, x3)

        m = torch.jit.script(Foo())

        buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)

        expected_ops = {
            "aten::_convolution",
            "aten::empty.memory_format",
            "aten::empty_like",
            "aten::zeros",
        }
        actual_ops = _export_operator_list(mobile_module)
        self.assertEqual(actual_ops, expected_ops)

    def test_source_range_simple(self):
        # [FUNCTION DOC] Round trips a ScriptModule with a matrix multiply and runs it with
        # incompatible shapes, asserting the error message cites this test file and the line
        # number computed as three lines after the class definition line.
        # KNOWN ISSUE: depends on source line offsets inside FooTest; any comment inserted
        # between the class line and the forward body would shift it, so docs for the nested
        # class and its forward method are kept above the class line instead.
        # [CLASS DOC] HELPER: ScriptModule whose forward multiplies x by the transpose of w.
        # Its forward method (decorated as script method) returns the matrix product; its
        # documentation lives here for the line number reason noted in the test doc above.
        class FooTest(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, w):
                return torch.mm(x, w.t())

        ft = FooTest()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest)

        with self.assertRaisesRegex(
            RuntimeError, f'test_lite_script_module.py", line {lineno + 3}'
        ):
            loaded(torch.rand(3, 4), torch.rand(30, 40))

    def test_source_range_raise_exception(self):
        # [FUNCTION DOC] Scripts a module whose forward raises RuntimeError with a fixed text,
        # exports and loads it, and asserts calling it raises the JIT error type carrying that
        # text. Does not check a source location (see the comment in the code).
        class FooTest2(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule whose forward always raises RuntimeError.
            @torch.jit.script_method
            def forward(self):
                # [FUNCTION DOC] HELPER: Raises RuntimeError with a fixed message.
                raise RuntimeError("foo")

        _, _ = inspect.getsourcelines(FooTest2)

        # In C++ code, the type of exception thrown is torch::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occurred w/o first changing
        # torch::jit::JITException to extend c10::Error.
        with self.assertRaisesRegex(torch.jit.Error, "foo"):
            ft = FooTest2()
            loaded = self.getScriptExportImportCopy(ft)
            loaded()

    def test_source_range_function_call(self):
        # [FUNCTION DOC] Round trips a ScriptModule where forward calls a script method, runs it
        # with a bad shape for the final add, and asserts the error text mentions this file at
        # the computed lines three and nine after the class line, and the top module name.
        # KNOWN ISSUE: depends on source line offsets inside FooTest3; comments between the
        # class line and the forward body would shift them, so the class and method docs are
        # kept above the class line instead.
        # [CLASS DOC] HELPER: ScriptModule with a script method add_method (adds two tensors)
        # and a forward (multiplies x and y, adds two, then calls add_method with w). Method
        # documentation lives here: add_method returns the sum; forward returns the result of
        # the helper call.
        class FooTest3(torch.jit.ScriptModule):
            @torch.jit.script_method
            def add_method(self, x, w):
                return x + w

            @torch.jit.script_method
            def forward(self, x, y, w):
                x = x * y
                x = x + 2
                return self.add_method(x, w)

        ft = FooTest3()
        loaded = self.getScriptExportImportCopy(ft)
        _, lineno = inspect.getsourcelines(FooTest3)

        try:
            loaded(torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 3}' in error_message
        )
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 9}' in error_message
        )
        self.assertTrue("top(FooTest3)" in error_message)

    def test_source_range_no_debug_info(self):
        # [FUNCTION DOC] Same as the matrix multiply failure test but exports without mobile
        # debug info and asserts the error message does not mention this test file.
        class FooTest4(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule whose forward multiplies x by the transpose of w.
            @torch.jit.script_method
            def forward(self, x, w):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x and the transpose of w.
                return torch.mm(x, w.t())

        ft = FooTest4()
        loaded = self.getScriptExportImportCopy(ft, save_mobile_debug_info=False)

        try:
            loaded(torch.rand(3, 4), torch.rand(30, 40))
        except RuntimeError as e:
            error_message = f"{e}"
        self.assertTrue("test_lite_script_module.py" not in error_message)

    def test_source_range_raise_exc(self):
        # [FUNCTION DOC] Exports a ScriptModule whose helper script method raises when its
        # integer argument equals the stored value, calls it with that value and asserts the
        # JIT error message contains the raise text.
        class FooTest5(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule storing an integer val, with script methods
            # add_method (raises when the argument equals val, else adds) and forward.
            def __init__(self, val: int):
                # [FUNCTION DOC] HELPER: Stores the integer val and initializes the parent.
                super().__init__()
                self.val = val

            @torch.jit.script_method
            def add_method(self, val: int, x, w):
                # [FUNCTION DOC] HELPER: Raises RuntimeError if val equals the stored value,
                # otherwise returns x plus w.
                if val == self.val:
                    raise RuntimeError("self.val and val are same")
                return x + w

            @torch.jit.script_method
            def forward(self, val: int, x, y, w):
                # [FUNCTION DOC] HELPER: Multiplies x and y, adds two, and returns the result of
                # add_method with val, the new x and w.
                x = x * y
                x = x + 2
                return self.add_method(val, x, w)

        ft = FooTest5(42)
        loaded = self.getScriptExportImportCopy(ft)
        _, _ = inspect.getsourcelines(FooTest5)

        try:
            loaded(42, torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))
        except torch.jit.Error as e:
            error_message = f"{e}"

        # In C++ code, the type of exception thrown is torch::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occurred w/o first changing
        # torch::jit::JITException to extend c10::Error.
        self.assertTrue("self.val and val are same" in error_message)

    def test_stacktrace_interface_call(self):
        # [FUNCTION DOC] Enables mobile export of interface calls, scripts a module that calls an
        # interface typed child's forward and forwardError, exports with debug info, loads it,
        # and runs it expecting RuntimeError. Uses FileCheck on the message to verify the
        # negative dimension error, the TorchScript traceback and the highlighted source lines
        # for the call chain. Enabling interface call export is a process-wide side effect.
        # KNOWN ISSUE: FileCheck matches quoted source text; doc blocks of the scripted nested
        # methods are kept above their def lines rather than inside the bodies.
        @torch.jit.interface
        class Forward(torch.nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface with forward and forwardError methods
            # that both take a tensor and return a tensor. Bodies are empty placeholders.
            # [FUNCTION DOC] HELPER: Interface method declaration for forward (empty body).
            def forward(self, x) -> torch.Tensor:
                pass

            # [FUNCTION DOC] HELPER: Interface method declaration for forwardError (empty body).
            def forwardError(self, x) -> torch.Tensor:
                pass

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Implementation of the interface; forwardError fails through call.
            # [FUNCTION DOC] HELPER: Returns the input unchanged.
            def forward(self, x):
                return x

            # [FUNCTION DOC] HELPER: Returns the call result plus x; the call raises at runtime.
            def forwardError(self, x):
                return self.call() + x

            # [FUNCTION DOC] HELPER: Tries to create a tensor with a negative size, which fails.
            def call(self):
                return torch.ones(-1)

        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a child typed as the interface, holding a B.
            b: Forward

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a B instance as the interface typed child b.
                super().__init__()
                self.b = B()

            # [FUNCTION DOC] HELPER: Calls the child forward and then its forwardError; the
            # second call fails at runtime. Returns nothing.
            def forward(self):
                self.b.forward(torch.ones(1))
                self.b.forwardError(torch.ones(1))

        a = torch.jit.script(A())
        torch._C._enable_mobile_interface_call_export()
        buffer = io.BytesIO(
            a._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        try:
            mobile_module()
            self.assertTrue(False)
        except RuntimeError as exp:
            FileCheck().check("Trying to create tensor with negative dimension").check(
                "Traceback of TorchScript"
            ).check("self.b.forwardError").check_next(
                "~~~~~~~~~~~~~~~~~~~ <--- HERE"
            ).check("return self.call").check_next("~~~~~~~~~ <--- HERE").check(
                "return torch.ones"
            ).check_next("~~~~~~~~~~ <--- HERE").run(str(exp))


class TestLiteScriptQuantizedModule(QuantizationLiteTestCase):
    # [CLASS DOC] Quantized model lite interpreter tests; compare scripted and mobile outputs.
    def test_single_layer(self):
        # [FUNCTION DOC] Builds a quantized single layer linear model under qnnpack and compares
        # scripted and mobile results on a random 2x5 input.
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedSingleLayerLinearModel, qengine="qnnpack"
        )
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_two_layer(self):
        # [FUNCTION DOC] Same comparison for the two layer linear model using the default engine.
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(model_class=TwoLayerLinearModel)
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_annotated_nested(self):
        # [FUNCTION DOC] Same comparison for the annotated nested model quantized with qnnpack.
        input = torch.rand(2, 5, dtype=torch.float)
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedNestedModel, qengine="qnnpack"
        )
        self._compare_script_and_mobile(model=quantized_model, input=input)

    def test_quantization_example(self):
        # [FUNCTION DOC] Follows the static quantization example: builds a quant stub, conv,
        # relu, dequant model, fuses conv and relu, prepares with random calibration data,
        # converts to int8 and compares scripted and mobile results.
        # From the example in Static Quantization section of https://pytorch.org/docs/stable/quantization.html
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Float model with quant stub, conv, relu and dequant stub.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the four submodules.
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies quant stub, conv, relu, dequant stub in order.
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x

        model_fp32 = M()

        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_fused = torch.ao.quantization.fuse_modules(
            model_fp32, [["conv", "relu"]]
        )
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        input_fp32 = torch.randn(4, 1, 4, 4)
        model_fp32_prepared(input_fp32)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        input = torch.randn(4, 1, 4, 4)
        self._compare_script_and_mobile(model=model_int8, input=input)

    def test_bundled_input_with_dynamic_type(self):
        # [FUNCTION DOC] Scripts a module taking three int-to-tensor dicts, bundles one sample
        # input for forward, exports to the lite format, loads it and checks that the bundled
        # input getter returns the same three dicts.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes three dicts and returns the first.
            def forward(
                self,
                x: dict[int, torch.Tensor],
                y: dict[int, torch.Tensor],
                z: dict[int, torch.Tensor],
            ):
                # [FUNCTION DOC] HELPER: Returns the first dict x; y and z are unused.
                return x

        model = Model()
        script_module = torch.jit.script(model)

        sample_input = {
            script_module.forward: [
                (
                    {0: torch.ones(1)},
                    {1: torch.ones(1)},
                    {2: torch.ones(1)},
                )
            ]
        }

        bundled_model = torch.utils.bundled_inputs.bundle_inputs(
            script_module, sample_input
        )

        buf = bundled_model._save_to_buffer_for_lite_interpreter()
        mobile_module = _load_for_lite_interpreter(io.BytesIO(buf))

        i = mobile_module.run_method("get_all_bundled_inputs")

        self.assertEqual(
            i[0],
            (
                {0: torch.ones(1)},
                {1: torch.ones(1)},
                {2: torch.ones(1)},
            ),
        )


if __name__ == "__main__":
    run_tests()
