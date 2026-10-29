# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log.debug_grad_log."""
# [MODULE DOC] Tests for the torch.utils.debug_log.debug_grad_log helper.
# The single test class installs a logging handler on the torch.utils.debug_log
# logger and checks which backward-pass log lines appear for various tensors.
# Imports debug_grad_log; has no import-time side effects beyond imports.

import logging

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.debug_log import debug_grad_log


class TestDebugGradLog(TestCase):
    # [CLASS DOC] Test case that captures log output from the torch.utils.debug_log
    # logger. State: _log_records (list of formatted log strings) and _handler (the
    # capturing handler). The bwd_logs property filters to backward-pass records.
    # Tests check log content for single tensors, multiple tensors, gradient norm
    # values, tensors without grad tracking, and the forward pass (no logging).
    def setUp(self):
        # [FUNCTION DOC] Per-test fixture. Calls the base setUp, creates an empty list
        # for captured records, attaches a capturing handler to the torch.utils.debug_log
        # logger, sets that logger to INFO level, and registers a cleanup that removes
        # the handler again. Side effect: the logger level stays INFO after the test.
        super().setUp()
        self._log_records: list[str] = []

        class _Handler(logging.Handler):
            # [CLASS DOC] HELPER: logging handler that stores each formatted record into a
            # destination list supplied at construction (attribute dest).
            def __init__(self, dest):
                # [FUNCTION DOC] HELPER: initializes the base Handler and remembers the list
                # to which emitted records will be appended.
                super().__init__()
                self.dest = dest

            def emit(self, record):
                # [FUNCTION DOC] HELPER: formats the record with the handler formatter and
                # appends the resulting string to the destination list.
                self.dest.append(self.format(record))

        self._handler = _Handler(self._log_records)
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(self._handler)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, self._handler)

    @property
    def bwd_logs(self) -> list[str]:
        # [FUNCTION DOC] HELPER: property returning the captured log lines that contain
        # the backward-pass marker, in capture order.
        return [r for r in self._log_records if "[bwd]" in r]

    def test_single_tensor(self):
        # [FUNCTION DOC] One tracked tensor derived from a random input is passed to
        # debug_grad_log, then backward is run on its sum. Expects exactly one backward
        # log line, which mentions the grad norm entry for tensor index zero.
        x = torch.randn(4, requires_grad=True)
        y = x * 2
        debug_grad_log(y)
        y.sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=", self.bwd_logs[0])

    def test_multi_tensor(self):
        # [FUNCTION DOC] Two grad-tracking inputs are registered together, then a
        # combined expression is summed and back-propagated. Expects exactly one
        # backward log line containing grad norm entries for tensor index zero and one.
        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        debug_grad_log(x, y)
        (x * 2 + y * 3).sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=", self.bwd_logs[0])
        self.assertIn("t1_grad_norm=", self.bwd_logs[0])

    def test_gradient_values(self):
        # [FUNCTION DOC] Checks logged values. With scalar-like inputs, the backward of
        # a sum of two scaled terms gives gradients 2 and 3, so the single backward log
        # line must contain the norms printed with four decimals as 2.0000 and 3.0000.
        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([1.0], requires_grad=True)
        debug_grad_log(x, y)
        (x * 2 + y * 3).sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=2.0000", self.bwd_logs[0])
        self.assertIn("t1_grad_norm=3.0000", self.bwd_logs[0])

    def test_no_requires_grad_no_log(self):
        # [FUNCTION DOC] A tensor that does not require grad is registered; no backward
        # pass is run. Expects no backward log lines.
        x = torch.randn(3, requires_grad=False)
        debug_grad_log(x)
        self.assertEqual(len(self.bwd_logs), 0)

    def test_forward_is_noop(self):
        # [FUNCTION DOC] Registering a grad-tracking tensor must not log anything during
        # the forward pass: the total captured record list must remain empty.
        x = torch.randn(3, requires_grad=True)
        debug_grad_log(x)
        self.assertEqual(len(self._log_records), 0)


if __name__ == "__main__":
    run_tests()
