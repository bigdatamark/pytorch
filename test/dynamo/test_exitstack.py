# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tracing contextlib.ExitStack (entering a context manager and
# callbacks that raise) under torch.compile. Defines a default-dtype context manager helper and
# the TestExitStack case. No import-time side effects.
import contextlib
import sys

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@contextlib.contextmanager
def set_default_dtype(dtype):
    # [FUNCTION DOC] HELPER: context manager that sets torch's global default dtype to the given
    # dtype for the with-body and restores the previous default on exit, even on exceptions.
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class TestExitStack(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case with unittest tracing enabled per test; stores the previous
    # config value in _prev during setUp and restores it in tearDown.
    def setUp(self):
        # [FUNCTION DOC] HELPER: runs base setUp, saves enable_trace_unittest config in _prev
        # and turns it on.
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: restores enable_trace_unittest to the value saved in setUp.
        # Does not call the base tearDown.
        torch._dynamo.config.enable_trace_unittest = self._prev

    def test_exitstack(self):
        # [FUNCTION DOC] Compiles (fullgraph, eager backend) a function that enters the
        # default-dtype context manager via an ExitStack and returns sin of the input; checks
        # the result equals eager sin and has float64 dtype.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: sets default dtype to float64 through an ExitStack, then
            # returns the elementwise sine of the argument.
            with contextlib.ExitStack() as stack:
                stack.enter_context(set_default_dtype(torch.float64))
                return t.sin()

        t = torch.randn(2, dtype=torch.float64)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(y.dtype, torch.float64)

    @make_dynamo_test
    def test_ensure_exc_is_active_in_two_contexts(self):
        # [FUNCTION DOC] Inside an ExitStack with two callbacks that each raise (IndexError,
        # KeyError) while a ZeroDivisionError propagates, checks the final IndexError has the
        # KeyError as its context, i.e. the in-flight exception is active in each callback.
        # Fails the test if no IndexError escapes.
        def raise_exc(exc):
            # [FUNCTION DOC] HELPER: asserts an exception is currently being handled, then
            # raises the given exception.
            self.assertIsNotNone(sys.exc_info()[1])
            raise exc

        try:
            with contextlib.ExitStack() as stack:
                stack.callback(raise_exc, IndexError)
                stack.callback(raise_exc, KeyError)
                raise ZeroDivisionError
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
        else:
            self.fail("Expected IndexError, but no exception was raised")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
