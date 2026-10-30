# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests of exception handling in TorchScript: error message propagation from script
# methods, builtin and unknown-op compile errors, raise statements and their inferred return
# types, assert statements, ignored Python ops that fail, dict expansion rejection, and custom
# Python exception classes. Uses the plain test case base. Running the file directly raises
# RuntimeError saying it is not in use and must be enabled in discover_tests.py first.
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase


r"""
Test TorchScript exception handling.
"""


class TestException(TestCase):
    # [CLASS DOC] Test case for TorchScript raise, assert and error reporting behavior. Some
    # tests compile source from strings, whose inner functions are not Python definitions and so
    # are not documented separately.
    def test_pyop_exception_message(self):
        # [FUNCTION DOC] Builds a script module wrapping a 2d convolution and calls it with a
        # one-dimensional tensor; the RuntimeError raised must carry the original convolution
        # message about expecting 3D unbatched or 4D batched input.
        class Foo(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module holding one convolution layer (1 input channel,
            # 10 output, kernel 5) applied in a script method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base module and creates the convolution
                # layer as attribute conv.
                super().__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method applying the convolution to the input.
                return self.conv(x)

        foo = Foo()
        # testing that the correct error message propagates
        with self.assertRaisesRegex(
            RuntimeError, r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d"
        ):
            foo(torch.ones([123]))  # wrong size

    def test_builtin_error_messsage(self):
        # [FUNCTION DOC] Checks compile-time error messages: scripting a function that calls
        # masked_fill with too few arguments raises RuntimeError saying call arguments are not
        # valid, and scripting one that calls an op not supported by TorchScript (the anomaly
        # mode setter) raises RuntimeError saying the op may not exist or be supported.
        with self.assertRaisesRegex(RuntimeError, "Arguments for call are not valid"):

            @torch.jit.script
            def close_match(x):
                # [FUNCTION DOC] HELPER: Calls masked_fill with only one argument; never
                # successfully compiled.
                return x.masked_fill(True)

        with self.assertRaisesRegex(
            RuntimeError,
            "This op may not exist or may not be currently supported in TorchScript",
        ):

            @torch.jit.script
            def unknown_op(x):
                # [FUNCTION DOC] HELPER: Calls an op unsupported in TorchScript and returns the
                # input; never successfully compiled.
                torch.set_anomaly_enabled(True)
                return x

    def test_exceptions(self):
        # [FUNCTION DOC] Broad test of raise handling. Covers a compilation unit function that
        # raises on a true condition (raises torch.jit.Error containing the argument text); an
        # undefined exception name and use of an exception as a value (both RuntimeError at
        # compile time); output type inference for always-throwing functions (none type when
        # undeclared, declared type otherwise); raising a non-exception value rejected; a
        # function whose variable escapes an if whose other branch always raises; and raises
        # with several arguments or no message, which appear in the Error text. Inner functions
        # are scripted or compiled and not run unless noted.
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                if bool(cond):
                    raise ValueError(3)
                return 1
        """
        )

        cu.foo(torch.tensor(0))
        with self.assertRaisesRegex(torch.jit.Error, "3"):
            cu.foo(torch.tensor(1))

        def foo(cond):
            # [FUNCTION DOC] HELPER: Raises an undefined exception name inside a conditional;
            # scripting it must fail.
            a = 3
            if bool(cond):
                raise ArbitraryError(a, "hi")  # noqa: F821
                if 1 == 2:
                    raise ArbitraryError  # noqa: F821
            return a

        with self.assertRaisesRegex(RuntimeError, "undefined value ArbitraryError"):
            torch.jit.script(foo)

        def exception_as_value():
            # [FUNCTION DOC] HELPER: Constructs a bare Exception object and prints it, which is
            # not allowed as a value; scripting must fail.
            a = Exception()
            print(a)

        with self.assertRaisesRegex(RuntimeError, "cannot be used as a value"):
            torch.jit.script(exception_as_value)

        @torch.jit.script
        def foo_no_decl_always_throws():
            # [FUNCTION DOC] HELPER: Scripted function with no declared return type that always
            # raises RuntimeError; its output type is inferred as none.
            raise RuntimeError("Hi")

        # function that has no declared type but always throws set to None
        output_type = next(foo_no_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "NoneType")

        @torch.jit.script
        def foo_decl_always_throws():
            # [FUNCTION DOC] HELPER: Scripted function declared by type comment to return a
            # tensor that always raises an exception; its output type remains tensor.
            # type: () -> Tensor
            raise Exception("Hi")  # noqa: TRY002

        output_type = next(foo_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "Tensor")

        def foo():
            # [FUNCTION DOC] HELPER: Raises the result of an arithmetic expression, which is
            # not an exception; scripting must fail with RuntimeError saying it must derive
            # from BaseException.
            raise 3 + 4

        with self.assertRaisesRegex(RuntimeError, "must derive from BaseException"):
            torch.jit.script(foo)

        # a escapes scope
        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Scripted function where one branch assigns a variable and
            # the other raises in all nested paths; returns the variable (1), showing the
            # variable remains usable after the conditional.
            if 1 == 1:
                a = 1
            else:
                if 1 == 1:
                    raise Exception("Hi")  # noqa: TRY002
                else:
                    raise Exception("Hi")  # noqa: TRY002
            return a

        self.assertEqual(foo(), 1)

        @torch.jit.script
        def tuple_fn():
            # [FUNCTION DOC] HELPER: Scripted function raising RuntimeError with two message
            # arguments; the resulting error text contains both joined by a comma.
            raise RuntimeError("hello", "goodbye")

        with self.assertRaisesRegex(torch.jit.Error, "hello, goodbye"):
            tuple_fn()

        @torch.jit.script
        def no_message():
            # [FUNCTION DOC] HELPER: Scripted function raising RuntimeError without a message;
            # the error text contains the class name.
            raise RuntimeError

        with self.assertRaisesRegex(torch.jit.Error, "RuntimeError"):
            no_message()

    def test_assertions(self):
        # [FUNCTION DOC] Checks assert statements: a compilation unit function asserting a
        # condition with a message succeeds when true and raises torch.jit.Error containing
        # AssertionError and the message when false. A scripted function with the same assertion
        # behaves identically (the exception name is not validated beyond the message text).
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                assert bool(cond), "hi"
                return 0
        """
        )

        cu.foo(torch.tensor(1))
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            cu.foo(torch.tensor(0))

        @torch.jit.script
        def foo(cond):
            # [FUNCTION DOC] HELPER: Scripted function asserting its condition with the message
            # hi; returns nothing.
            assert bool(cond), "hi"  # noqa: S101

        foo(torch.tensor(1))
        # we don't currently validate the name of the exception
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            foo(torch.tensor(0))

    def test_python_op_exception(self):
        # [FUNCTION DOC] Checks that an exception raised inside an ignored Python function
        # called from a script function surfaces as RuntimeError saying the operation failed in
        # the TorchScript interpreter.
        @torch.jit.ignore
        def python_op(x):
            # [FUNCTION DOC] HELPER: Ignored Python function that always raises a generic
            # exception with a message.
            raise Exception("bad!")  # noqa: TRY002

        @torch.jit.script
        def fn(x):
            # [FUNCTION DOC] HELPER: Scripted function calling the ignored Python function on
            # its input.
            return python_op(x)

        with self.assertRaisesRegex(
            RuntimeError, "operation failed in the TorchScript interpreter"
        ):
            fn(torch.tensor(4))

    def test_dict_expansion_raises_error(self):
        # [FUNCTION DOC] Checks scripting a function that expands a dictionary into a new
        # dictionary literal raises the frontend NotSupportedError mentioning dict expansion.
        def fn(self):
            # [FUNCTION DOC] HELPER: Builds a three-entry dictionary and returns a dictionary
            # expanded from it; never successfully compiled.
            d = {"foo": 1, "bar": 2, "baz": 3}
            return {**d}

        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError, "Dict expansion "
        ):
            torch.jit.script(fn)

    def test_custom_python_exception(self):
        # [FUNCTION DOC] Defines a ValueError subclass inside the test, scripts a function that
        # raises it with a message, and checks calling it raises torch.jit.Error whose text
        # names the class (qualified with this test module) and the message.
        class MyValueError(ValueError):
            # [CLASS DOC] HELPER: Custom exception class deriving from ValueError with no
            # extra behavior.
            pass

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function raising the custom exception with a
            # fixed message.
            raise MyValueError("test custom exception")

        with self.assertRaisesRegex(
            torch.jit.Error, "jit.test_exception.MyValueError: test custom exception"
        ):
            fn()

    def test_custom_python_exception_defined_elsewhere(self):
        # [FUNCTION DOC] Imports a key-error subclass defined in the jit.myexception helper
        # module, scripts a function raising it, and checks the torch.jit.Error text names the
        # class with its module path and the message.
        from jit.myexception import MyKeyError

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function raising the imported custom key error
            # with a fixed message.
            raise MyKeyError("This is a user defined key error")

        with self.assertRaisesRegex(
            torch.jit.Error,
            "jit.myexception.MyKeyError: This is a user defined key error",
        ):
            fn()


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
