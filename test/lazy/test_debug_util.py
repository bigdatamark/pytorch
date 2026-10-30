# Owner(s): ["oncall: jit"]
# [MODULE DOC] Test for lazy tensor debug utilities. Import-time side effect: initializes the
# TorchScript lazy backend. The test runs a linear layer on the lazy device with the environment
# variable naming a tensors-save file set, then checks the saved graph dump contains a Python
# stacktrace section with the expected frames. Skipped on Windows.

import os
import re
import tempfile
import unittest

import torch._lazy
import torch._lazy.ts_backend
import torch.nn as nn
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch._lazy.ts_backend.init()


@unittest.skipIf(IS_WINDOWS, "To be fixed")
class DebugUtilTest(TestCase):
    # [CLASS DOC] Test case for the lazy debug-dump feature; skipped on Windows by the decorator.
    def _run_linear(self):
        # [FUNCTION DOC] HELPER: Moves a 5 by 5 linear layer to the lazy device, applies it to a
        # random lazy input, and calls the lazy step marker so the pending graph is executed (and
        # dumped if the save-tensors environment variable is set). The output is unused.
        device = "lazy"
        model = nn.Linear(5, 5).to(device)
        output = model(torch.randn(1, 5).to(device))  # noqa: F841
        torch._lazy.mark_step()

    def test_get_python_frames(self):
        # [FUNCTION DOC] Creates a temporary file, points the lazy tensors-save environment
        # variable (left set in the process afterwards) at it, runs the linear helper, reads the
        # file, and checks with a dot-matches-newline regular expression that it contains a Python
        # stacktrace section with frames for mark_step, the helper and this test in order. On
        # mismatch it prints the file content and fails the test.
        # We only care about the first "Python Stacktrace" part of the saved
        # graph. However, we cannot save the whole stack for comparison given
        # it depends on a lot of things.
        partial_graph = (
            r"Python Stacktrace:.*"
            r"mark_step \(.*/_lazy/__init__.py:[0-9]+\).*"
            r"_run_linear \(.*lazy/test_debug_util.py:[0-9]+\).*"
            r"test_get_python_frames \(.*lazy/test_debug_util.py:[0-9]+\)"
        )

        with tempfile.NamedTemporaryFile(mode="r+", encoding="utf-8") as graph_file:
            os.environ["LTC_SAVE_TENSORS_FILE"] = graph_file.name
            self._run_linear()
            file = graph_file.read()
            if re.search(partial_graph, file, re.DOTALL) is None:
                print(file)
                self.assertTrue(False)


if __name__ == "__main__":
    run_tests()
