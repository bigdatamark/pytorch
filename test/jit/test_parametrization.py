# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for parametrized modules (torch.nn.utils.parametrize) under tracing and
# scripting. Tracing is expected to work, including with the caching context; scripting is
# currently expected to fail with a JIT error (see the TODO in the test).

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestParametrization(JitTestCase):
    # [CLASS DOC] JitTestCase with tests for tracing and scripting a Linear whose weight is
    # parametrized by a symmetric parametrization.
    # Define some parametrization
    class Symmetric(nn.Module):
        # [CLASS DOC] HELPER: Parametrization module that maps a square matrix to a
        # symmetric one.
        def forward(self, X):
            # [FUNCTION DOC] HELPER: Returns the upper triangle of the input plus the transposed
            # strict upper triangle, giving a symmetric matrix built from the upper triangle.
            return X.triu() + X.triu(1).mT

    def test_traceable(self):
        # [FUNCTION DOC] Registers the symmetric parametrization on a 5x5 Linear weight, traces
        # it with trace_module and checks traced output equals eager output, also inside the
        # caching context. Checks that tracing while caching raises RuntimeError mentioning that
        # a model cannot be traced while caching.
        r"""Test the jit scripting and tracing of a parametrized model."""
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        # Check the tracing works. Because traced functions cannot be called
        # directly, we run the comparison on the activations.
        traced_model = torch.jit.trace_module(model, {"forward": x})
        y_hat = traced_model(x)
        self.assertEqual(y, y_hat)

        # Check traced model works with caching
        with parametrize.cached():
            y_hat = traced_model(x)
            self.assertEqual(y, y_hat)

        # Check the tracing throws an error when caching
        with self.assertRaisesRegex(RuntimeError, "Cannot trace a model while caching"):
            with parametrize.cached():
                traced_model = torch.jit.trace_module(model, {"forward": x})

    def test_scriptable(self):
        # [FUNCTION DOC] Registers the symmetric parametrization on a Linear weight and expects
        # the whole scripting block to raise torch.jit.Error. The comparisons and caching checks
        # inside that block are therefore never reached.
        # KNOWN ISSUE: the test only documents that scripting of parametrized modules is broken.
        # TODO: Need to fix the scripting in parametrizations
        #       Currently, all the tests below will throw torch.jit.Error
        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", self.Symmetric())

        x = torch.randn(3, 5)
        y = model(x)

        with self.assertRaises(torch.jit.Error):
            # Check scripting works
            scripted_model = torch.jit.script(model)
            y_hat = scripted_model(x)
            self.assertEqual(y, y_hat)

            with parametrize.cached():
                # Check scripted model works when caching
                y_hat = scripted_model(x)
                self.assertEqual(y, y_hat)

                # Check the scripting process throws an error when caching
                with self.assertRaisesRegex(RuntimeError, "Caching is not implemented"):
                    scripted_model = torch.jit.trace_module(model)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
