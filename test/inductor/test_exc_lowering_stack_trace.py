# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that the Inductor lowering exception carries the user stack trace (a Found
# from section) when available and still formats correctly when the trace is missing or empty.
# Uses an override of the lowering for the aten ceil op to force a failure.

import torch
from torch._inductor.exc import InductorError, LoweringException
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import override_lowering


class TestLoweringExceptionStackTrace(InductorTestCase):
    # [CLASS DOC] Test case with two tests on the message content of the lowering exception.
    """Tests that LoweringException includes user stack traces and remains backward-compatible, with diagnostics."""

    def test_lowering_exception_includes_stack_trace(self):
        # [FUNCTION DOC] Builds a chain of five nested user functions ending in the aten ceil
        # op, compiled with the inductor backend and full graph. Overrides the ceil lowering
        # with one that records being called and raises a RuntimeError. Expects the compile to
        # raise an InductorError whose message contains the wrapped lowering exception text, a
        # Found from section, and the names of the innermost and outermost user frames. Also
        # prints the message.
        def frame5(x):
            # [FUNCTION DOC] HELPER: Innermost user frame; applies the aten ceil op, the target
            # of the injected failure.
            return torch.ops.aten.ceil.default(x)

        def frame4(x):
            # [FUNCTION DOC] HELPER: Forwards to the next-inner frame helper.
            return frame5(x)

        def frame3(x):
            # [FUNCTION DOC] HELPER: Forwards to the next-inner frame helper.
            return frame4(x)

        def frame2(x):
            # [FUNCTION DOC] HELPER: Forwards to the next-inner frame helper.
            return frame3(x)

        def frame1(x):
            # [FUNCTION DOC] HELPER: Outermost user frame helper; forwards to the next-inner
            # frame helper.
            return frame2(x)

        @torch.compile(backend="inductor", fullgraph=True)
        def test_function(x):
            # [FUNCTION DOC] HELPER: Compiled entry point that calls the outermost frame helper.
            return frame1(x)

        invoked = {"flag": False}

        def failing_lowering(orig_fn, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement lowering: sets the invoked flag in the
            # surrounding mapping and raises a RuntimeError with a fixed message. The flag is
            # never asserted by the test.
            invoked["flag"] = True
            raise RuntimeError(
                "Intentional lowering failure for testing  user stack traces"
            )

        ceil_op = torch.ops.aten.ceil.default

        with override_lowering(ceil_op, failing_lowering):
            try:
                test_function(torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32))
                self.fail("Expected InductorError, but got a result")
            except InductorError as e:
                error_msg = str(e)

                self.assertIn(
                    "LoweringException: RuntimeError: Intentional lowering failure for testing  user stack traces",
                    error_msg,
                    "InductorError should contain the wrapped LoweringException with our message",
                )

                self.assertIn(
                    "Found from :",
                    error_msg,
                    "Error should include 'Found from :' with user stack trace",
                )
                print(error_msg)

                self.assertIn("frame5", error_msg, "Stack trace should include frame5")
                self.assertIn("frame1", error_msg, "Stack trace should include frame1")

    def test_lowering_exception_without_stack_trace(self):
        # [FUNCTION DOC] Constructs the lowering exception directly, with no stack trace and
        # then with an empty-string stack trace, for a RuntimeError and the ceil target with
        # empty args. Checks the text contains the wrapped error message and the op name and
        # never contains the Found from section.
        """
        LoweringException works correctly when stack_trace is not present.
        It should not include the 'Found from' section. Verifies backward compatibility.
        """
        test_exception = RuntimeError("Test exception without stack trace")

        test_target = torch.ops.aten.ceil.default

        test_args = []
        test_kwargs = {}

        exc = LoweringException(
            test_exception,
            test_target,
            test_args,
            test_kwargs,
            stack_trace=None,  # type: ignore[call-arg]
        )
        msg = str(exc)
        self.assertIn("RuntimeError: Test exception without stack trace", msg)
        self.assertIn("ceil", msg)
        self.assertNotIn("Found from", msg)

        # Also test with stack_trace explicitly set to empty string
        exc_empty = LoweringException(
            test_exception, test_target, test_args, test_kwargs, stack_trace=""
        )
        msg_empty = str(exc_empty)
        self.assertNotIn("Found from", msg_empty)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
