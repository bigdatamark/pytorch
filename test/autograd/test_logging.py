# Owner(s): ["module: autograd"]

# [MODULE DOC] Test module for autograd debug logging. Defines one test case that runs a small
# backward pass with the autograd logger at DEBUG level and checks the logged node names.
# Run directly, it uses the dynamo test runner. No import-time side effects.

import logging

import torch
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestAutogradLogging(LoggingTestCase):
    # [CLASS DOC] Logging test case for the autograd engine; relies on LoggingTestCase to capture
    # log records emitted by the autograd logger during each decorated test.
    @make_logging_test(autograd=logging.DEBUG)
    def test_logging(self, records):
        # [FUNCTION DOC] Builds a graph of multiply, divide, sum and clone, then runs backward on
        # both the sum and its clone. Expects exactly five debug records, in execution order:
        # CloneBackward0, SumBackward0, DivBackward0, MulBackward0, AccumulateGrad, each name
        # appearing in the corresponding record message. Raises on mismatch via assertions.
        a = torch.rand(10, requires_grad=True)
        b = a.mul(2).div(3).sum()
        c = b.clone()
        torch.autograd.backward((b, c))

        self.assertEqual(len(records), 5)
        expected = [
            "CloneBackward0",
            "SumBackward0",
            "DivBackward0",
            "MulBackward0",
            "AccumulateGrad",
        ]

        for i, record in enumerate(records):
            self.assertIn(expected[i], record.getMessage())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
