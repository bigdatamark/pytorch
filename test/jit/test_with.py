# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] JIT tests for with-statement support in TorchScript. Defines one test case whose
# tests define scripted context-manager classes (counter kept in a double tensor, mutated in place
# by enter and exit) and verify that with-statements compile and match eager behavior with and
# without the as clause, with early return, break, continue, nesting and multiple items; that
# exceptions propagate while exit still restores state; that malformed context managers are
# rejected with specific errors; and that no_grad and record_function work in scripts.

import os
import sys
from typing import Any, List

import torch
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
)
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestWith(JitTestCase):
    # [CLASS DOC] Test case for TorchScript with-statements. Tests build their scripted helper
    # classes and functions locally (registering the class globally with make_global) and run them
    # through checkScript or direct scripted calls.
    """
    A suite of tests for with statements.
    """

    def test_with_as(self):
        # [FUNCTION DOC] Checks with-statements that bind the entered value with as. Defines a
        # context manager and several scripted functions (basic, pass body, early return,
        # conditional early return, break, continue, serial, nested, multiple items), runs each
        # through checkScript against eager execution, and asserts the shared context's count
        # returns to 2 afterwards.
        """
        Check that with statements that use the 'as' keyword to bind expressions
        to targets work as expected.
        """

        @torch.jit.script
        class Context:
            # [CLASS DOC] HELPER: Scripted context manager. Its state is a one-element double
            # tensor holding the start value; entering adds 0.3 in place and returns the
            # tensor, exiting subtracts 0.3 in place, so in-place effects are visible outside
            # the interpreter.
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                # [FUNCTION DOC] HELPER: Stores a one-element double tensor initialised with start
                # as the count.
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Adds 0.3 to the count tensor in place and returns
                # that tensor.
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any) -> bool:
                # [FUNCTION DOC] HELPER: Subtracts 0.3 from the count in place and returns True
                # (exception handled flag).
                self.count.sub_(0.3)
                return True

        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds the entered value to the input inside a with-statement,
            # then scales by the count.
            """Basic test with one with-statement."""

            c = Context(1)

            with c as mult:
                y = x + mult

            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Runs a with-statement whose body is pass; enter and exit must
            # still run.
            """
            Test with a pass statement inside a with-statement. Although
            the body of the with is empty, __enter__ and __exit__ should
            still be called.
            """
            c = Context(1)

            with c as mult:
                pass

            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns from inside the with-statement; code after it is
            # never reached.
            """
            Test that returning early from inside a with-statement works
            as expected.
            """
            with c as mult:
                y = x + mult
                return y

            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns early from the with-statement only when the entered
            # value is positive.
            """
            Test that conditionally returning early from inside a with-statement works
            as expected.
            """
            with c as mult:
                y = x + mult
                if mult > 0:
                    return y

            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Breaks out of a loop inside the with-statement when an element
            # is zero, otherwise accumulates.
            """
            Test that breaking early from inside a with-statement works
            as expected.
            """
            with c as mult:
                for a in l:
                    if a == 0:
                        break
                    x += a * mult

            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Skips zero elements with continue inside a loop within
            # the with-statement.
            """
            Test that using continue inside a with-statement works
            as expected.
            """
            with c as mult:
                for a in l:
                    if a == 0:
                        continue
                    x += a * mult

            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses two with-statements one after another on the
            # same context.
            """
            Test two with-statements in a row.
            """
            c = Context(1)

            with c as mult:
                y = x + mult

            with c as mult:
                y *= mult

            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses a with-statement nested inside another with-statement.
            """
            Test nested with-statements.
            """
            c = Context(1)

            with c as m:
                with c as n:
                    y = x + n

                y *= m

            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses one with-statement with two context managers
            # entered together.
            """
            Test a with-statement with multiple with items.
            """
            c = Context(1)
            d = Context(2)

            with c as m, d as n:
                y = x + (m + n)

            return y

        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]

        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_no_as(self):
        # [FUNCTION DOC] Same scenarios as the with-as test but without binding the entered value;
        # bodies read the context's count attribute instead. Uses a context manager whose exit
        # returns nothing, checks each function with checkScript, and asserts the shared count ends
        # at 2.
        """
        Check that with statements that do not use the 'as' keyword to bind expressions
        to targets work as expected.
        """

        @torch.jit.script
        class Context:
            # [CLASS DOC] HELPER: Scripted context manager. Its state is a one-element double
            # tensor holding the start value; entering adds 0.3 in place and returns the
            # tensor, exiting subtracts 0.3 in place, so in-place effects are visible outside
            # the interpreter.
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                # [FUNCTION DOC] HELPER: Stores a one-element double tensor initialised with start
                # as the count.
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Adds 0.3 to the count tensor in place and returns
                # that tensor.
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                # [FUNCTION DOC] HELPER: Subtracts 0.3 from the count in place; returns nothing.
                self.count.sub_(0.3)

        make_global(Context)

        def test_basic(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds the context count to the input inside a with-statement
            # without an as clause.
            """Basic test with one with-statement."""

            c = Context(1)

            with c:
                y = x + c.count

            y *= c.count
            return y

        def test_pass(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Runs a with-statement whose body is pass; enter and exit must
            # still run.
            """
            Test with a pass statement inside a with-statement. Although
            the body of the with is empty, __enter__ and __exit__ should
            still be called.
            """
            c = Context(1)

            with c:
                pass

            x *= c.count
            return x

        def test_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns from inside the with-statement.
            """
            Test that returning early from inside a with-statement works
            as expected.
            """
            with c:
                y = x + c.count
                return y

            x = y + y
            return x

        def test_conditional_early_return(x: torch.Tensor, c: Context) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns early only when the context count is positive.
            """
            Test that conditionally returning early from inside a with-statement works
            as expected.
            """
            with c:
                y = x + c.count
                if c.count > 0:
                    return y

            x = y + y
            return x

        def test_break(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Breaks out of a loop inside the with-statement when an element
            # is zero.
            """
            Test that breaking early from inside a with-statement works
            as expected.
            """
            with c:
                for a in l:
                    if a == 0:
                        break
                    x += a * c.count

            return x

        def test_continue(x: torch.Tensor, c: Context, l: List[int]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Skips zero elements with continue inside a loop within
            # the with-statement.
            """
            Test that using continue inside a with-statement works
            as expected.
            """
            with c:
                for a in l:
                    if a == 0:
                        continue
                    x += a * c.count

            return x

        def test_serial(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses two with-statements one after another on the
            # same context.
            """
            Test two with-statements in a row.
            """
            c = Context(1)

            with c:
                y = x + c.count

            with c:
                y *= c.count

            return y

        def test_nested(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses a with-statement nested inside another with-statement on
            # the same context.
            """
            Test nested with-statements.
            """
            c = Context(1)

            with c:
                with c:
                    y = x + c.count

                y *= c.count

            return y

        def test_combined(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Uses one with-statement with two context managers.
            """
            Test a with-statement with multiple with items.
            """
            c = Context(1)
            d = Context(2)

            with c, d:
                y = x + (c.count + d.count)

            return y

        test_input = torch.randn(2, 2)
        test_context = Context(2)
        test_list = [2, 0, 1, 3, 0, 2]

        self.checkScript(test_basic, (test_input,))
        self.checkScript(test_pass, (test_input,))
        self.checkScript(test_early_return, (test_input, test_context))
        self.checkScript(test_break, (test_input, test_context, test_list))
        self.checkScript(test_continue, (test_input, test_context, test_list))
        self.assertEqual(test_context.count, 2)
        self.checkScript(test_serial, (test_input,))
        self.checkScript(test_nested, (test_input,))
        self.checkScript(test_combined, (test_input,))

    def test_with_exceptions(self):
        # [FUNCTION DOC] Checks exception handling inside with-statements. Defines a context
        # manager and scripted functions that raise from the body (plain, nested, and via a call to
        # another scripted function that also holds a with-statement). Calls each and asserts a
        # generic exception with the raised message is thrown (highlighting the raise site) and
        # that the context's count is back to 1, i.e. exit ran each time. checkScript is not used
        # because the string frontend cannot compile class types.
        """
        Check that exceptions thrown in the bodies of with-statements are
        handled correctly.
        """

        @torch.jit.script
        class Context:
            # [CLASS DOC] HELPER: Scripted context manager. Its state is a one-element double
            # tensor holding the start value; entering adds 0.3 in place and returns the
            # tensor, exiting subtracts 0.3 in place, so in-place effects are visible outside
            # the interpreter.
            """
            This class implements a basic context manager interface for use in
            the unit tests. Unlike Context, the stateful part of this class
            is a Tensor that is mutated in-place so that modifications made in the
            JIT interpreter are visible outside of it.
            """

            def __init__(self, start: int):
                # [FUNCTION DOC] HELPER: Stores a one-element double tensor initialised with start
                # as the count.
                self.count = torch.tensor([start], dtype=torch.double)

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Adds 0.3 to the count tensor in place and returns
                # that tensor.
                self.count.add_(0.3)
                return self.count

            def __exit__(self, type: Any, value: Any, tb: Any):
                # [FUNCTION DOC] HELPER: Subtracts 0.3 from the count in place; returns nothing.
                self.count.sub_(0.3)

        make_global(Context)

        # [FUNCTION DOC] HELPER: Scripted function that always raises a generic exception with a
        # fixed message.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        @torch.jit.script
        def method_that_raises() -> torch.Tensor:
            raise Exception("raised exception")  # noqa: TRY002

        # [FUNCTION DOC] HELPER: Scripted function that calls the raising function inside
        # a with-statement.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        @torch.jit.script
        def test_exception(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while executing the body of a with-statement.
            """
            with c as _:
                x += method_that_raises()

            return x

        # [FUNCTION DOC] HELPER: Scripted function that calls the raising function inside two
        # nested with-statements.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        @torch.jit.script
        def test_exception_nested(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while executing the body of a nested with-statement.
            """
            with c as _:
                with c as _:
                    x += method_that_raises()

            return x

        # [FUNCTION DOC] HELPER: Scripted function that enters a with-statement and calls the
        # raising function in its body.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        @torch.jit.script
        def with_that_raises(c: Context) -> torch.Tensor:
            a = torch.tensor([1])

            with c as _:
                a += method_that_raises()

            return a

        # [FUNCTION DOC] HELPER: Scripted function holding a with-statement whose body calls the
        # function that also has one, so two frames have active contexts when the error is raised.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        @torch.jit.script
        def test_exception_fn_call(x: torch.Tensor, c: Context) -> torch.Tensor:
            """
            Test the case in which an exception is thrown while there are active with-statements in two different
            frames.
            """
            with c as _:
                x += with_that_raises(c)

            return x

        c = Context(1)

        # checkScript and checkScriptRaisesRegex cannot be used because the string frontend will
        # not compile class types (of which Context, the context manager being used for this test
        # is one).
        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception(torch.randn(2), c)
        self.assertEqual(c.count, 1)

        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception_nested(torch.randn(2), c)
        self.assertEqual(c.count, 1)

        with self.assertRaisesRegexWithHighlight(
            Exception, r"raised exception", 'raise Exception("raised exception'
        ):
            test_exception_fn_call(torch.randn(2), c)
        self.assertEqual(c.count, 1)

    def test_with_errors(self):
        # [FUNCTION DOC] Checks compile-time errors for invalid context managers. Defines scripted
        # classes lacking enter and exit, with a bad enter signature, a bad exit signature, and
        # exit arguments of the wrong types, plus a function entering a string. Each is run through
        # checkScript inside assertRaisesRegexWithHighlight and must raise a runtime error with the
        # expected message and highlight (the with item expression).
        """
        Check that errors related to with-statements are detected and reported correctly.
        """

        @torch.jit.script
        class NoEnterNoExit:
            # [CLASS DOC] HELPER: Scripted class with no enter or exit method; used to trigger the
            # missing-methods error.
            """
            This class is missing __enter__ and __exit__ methods.
            """

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets count to 1.
                self.count = 1

        @torch.jit.script
        class BadEnter:
            # [CLASS DOC] HELPER: Scripted class whose enter method takes an extra argument, which
            # is rejected.
            """
            This class has an __enter__ method with an incorrect signature.
            """

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets count to 1.
                self.count = 1

            def __enter__(self, incr: int):  # noqa: PLE0302
                # [FUNCTION DOC] HELPER: Invalid enter taking an extra integer argument and adding
                # it to count.
                self.count += incr

            def __exit__(self, type: Any, value: Any, tb: Any):
                # [FUNCTION DOC] HELPER: Valid exit signature with three Any arguments;
                # does nothing.
                pass

        @torch.jit.script
        class BadExit:
            # [CLASS DOC] HELPER: Scripted class whose exit method has too few arguments, which
            # is rejected.
            """
            This class has an __exit__ method with an incorrect signature.
            """

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets count to 1.
                self.count = 1

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Valid enter without extra arguments; increments count.
                self.count += 1

            def __exit__(self, type: Any, value: Any):  # noqa: PLE0302
                # [FUNCTION DOC] HELPER: Invalid exit taking only two arguments besides self;
                # does nothing.
                pass

        @torch.jit.script
        class ExitIncorrectTypes:
            # [CLASS DOC] HELPER: Scripted class whose exit method types its second and third
            # arguments as int instead of Any, which is rejected.
            """
            This class has an __exit__ method with unsupported argument types.
            """

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets count to 1.
                self.count = 1

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Valid enter without extra arguments; increments count.
                self.count += 1

            def __exit__(self, type: Any, value: int, tb: int):
                # [FUNCTION DOC] HELPER: Invalid exit whose second and third arguments are typed as
                # int; does nothing.
                pass

        # [FUNCTION DOC] HELPER: Enters a with-statement on an object lacking enter and exit;
        # compiling it must fail.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        def test_no_enter_no_exit(x: torch.Tensor, cm: NoEnterNoExit) -> torch.Tensor:
            with cm as _:
                pass

            return x

        # [FUNCTION DOC] HELPER: Enters a with-statement on an object whose enter has a bad
        # signature; compiling it must fail.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        def test_bad_enter(x: torch.Tensor, cm: BadEnter) -> torch.Tensor:
            with cm as _:
                pass

            return x

        # [FUNCTION DOC] HELPER: Enters a with-statement on an object whose exit has a bad
        # signature; compiling it must fail.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        def test_bad_exit(x: torch.Tensor, cm: BadExit) -> torch.Tensor:
            with cm as _:
                pass

            return x

        # [FUNCTION DOC] HELPER: Enters a with-statement on an object whose exit has non-Any
        # argument types; compiling it must fail.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        def test_exit_incorrect_types(
            x: torch.Tensor, cm: ExitIncorrectTypes
        ) -> torch.Tensor:
            with cm as _:
                pass

            return x

        # [FUNCTION DOC] HELPER: Uses a string literal as the with item, which is not an object
        # with enter and exit; compiling it must fail.
        # KNOWN ISSUE: doc placed above the decorator, not inside the body, because tests assert on
        #     the highlighted source text of this scripted function.
        def test_enter_without_object():
            with "not_object" as obj:
                pass

        test_tensor = torch.randn(5, dtype=torch.double)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"does not define __enter__ and __exit__ methods", "cm"
        ):
            self.checkScript(test_no_enter_no_exit, (test_tensor, NoEnterNoExit()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"__enter__ must have only one argument and one return value",
            "cm",
        ):
            self.checkScript(test_bad_enter, (test_tensor, BadEnter()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"__exit__ must have four arguments", "cm"
        ):
            self.checkScript(test_bad_exit, (test_tensor, BadExit()))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"argument 2 of __exit__ must have Any type", "cm"
        ):
            self.checkScript(
                test_exit_incorrect_types, (test_tensor, ExitIncorrectTypes())
            )

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"must return an object", '"not_object"'
        ):
            self.checkScript(test_enter_without_object, ())

    def test_with_no_grad(self):
        # [FUNCTION DOC] Checks torch.no_grad as a context manager in scripts: a scripted function
        # computing under no_grad returns a result without requires_grad or grad_fn; assigning a
        # grad-less tensor into a grad-requiring tensor under no_grad keeps requires_grad on the
        # target; and an ignored Python method called from a scripted module under no_grad also
        # produces a result without requires_grad.
        """
        Check that torch.no_grad() works. Most of these are adapted from
        corresponding tests for eager-mode no_grad.
        """

        # Basic no_grad test.
        def test_no_grad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds two tensors inside a no_grad with-statement and returns
            # the sum.
            with torch.no_grad():
                w = x + y

            return w

        s = torch.jit.script(test_no_grad)
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        w = s(x, y)

        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)

        # Test assignment of a grad-less Tensor to a Tensor with gradients
        # in a no_grad block.
        def test_no_grad_assignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Assigns into the first row of the first argument from the
            # second under no_grad and returns it.
            with torch.no_grad():
                x[0] = y

            return x

        s = torch.jit.script(test_no_grad_assignment)
        z = torch.randn(5)
        w = s(x, z)
        self.assertTrue(w.requires_grad)
        self.assertIsNone(w.grad_fn)

        # Check that @torch.jit.ignored functions respect no_grad when it is
        # called in JIT mode.
        class NoGradModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose scripted forward calls an ignored Python method
            # inside a no_grad with-statement.
            @torch.jit.ignore
            def adder(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Ignored (Python-executed) method returning the sum of its
                # two tensors.
                w = x + y
                return w

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls the ignored adder method under no_grad and returns
                # its result.
                with torch.no_grad():
                    w = self.adder(x, y)

                return w

        s = torch.jit.script(NoGradModule())
        w = s(x, y)

        self.assertFalse(w.requires_grad)

    @skipIfTorchDynamo("Torchdynamo cannot correctly handle profiler.profile calls")
    def test_with_record_function(self):
        # [FUNCTION DOC] Checks that the record_function context manager is scriptable. Scripts a
        # function with nested record_function contexts, runs it under the profiler, and asserts
        # one event named foo exists, its CPU children include the event named nested, and that
        # nested event has an aten add child. Skipped under TorchDynamo.
        """
        Check that torch.autograd.profiler.record_function context manager is
        torchscriptable.
        """

        def with_rf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Adds two tensors inside nested record_function contexts named
            # foo and nested.
            with torch.autograd.profiler.record_function("foo"):
                # Nested record_function.
                with torch.autograd.profiler.record_function("nested"):
                    a = x + y
            return a

        scripted = torch.jit.script(with_rf)
        x, y = torch.ones(2), torch.ones(2)
        with torch.autograd.profiler.profile() as p:
            scripted(x, y)

        # Need to call below to populate CPU children.
        p.key_averages()
        function_events = p.function_events
        # Event with name "foo" should be recorded.
        rf_events = [evt for evt in function_events if evt.name == "foo"]
        self.assertEqual(len(rf_events), 1)
        rf_event = rf_events[0]
        child_events = rf_event.cpu_children
        # Ensure we find nested record_function event
        self.assertTrue("nested" in (child.name for child in child_events))
        nested_function_event = [
            evt for evt in function_events if evt.name == "nested"
        ][0]
        # Nested record function should have child "aten::add"
        nested_child_events = nested_function_event.cpu_children
        self.assertTrue("aten::add" in (child.name for child in nested_child_events))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
