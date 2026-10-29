# Owner(s): ["module: dynamo"]
# [MODULE DOC] Test module checking that AOT autograd emits debug-level log records when compiling
# a function. Contains one logging test case; running the file as a script starts the PyTorch test
# runner. No import-time side effects beyond imports.
import logging

import torch
from torch._functorch.aot_autograd import aot_function
from torch._functorch.compilers import nop
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class TestAOTLogging(LoggingTestCase):
    # [CLASS DOC] Logging test case (based on LoggingTestCase) holding a single test that checks
    # AOT autograd logging output. It keeps no state of its own.
    @make_logging_test(aot=logging.DEBUG)
    def test_logging(self, records):
        # [FUNCTION DOC] Compiles a simple sine function with aot_function using the no-op compiler
        # for forward and backward, runs it once on a random 3-element tensor, and asserts that at
        # least one log record was captured. The records argument is supplied by the logging-test
        # decorator, which enables debug logging for the AOT logger and collects emitted records.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns the elementwise sine of its input.
            return torch.sin(x)

        compiled_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
        compiled_f(torch.randn(3))
        self.assertGreater(len(records), 0)


if __name__ == "__main__":
    run_tests()
