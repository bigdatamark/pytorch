# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for TorchScript logging hooks in torch.jit._logging: numeric counters,
# timers, aggregation types and the global logging option string. Adds the test directory to
# sys.path at import time; running the file directly is rejected via raise_on_run_directly.

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestLogging(JitTestCase):
    # [CLASS DOC] JitTestCase suite checking that stat values added from scripted, traced and
    # eager code reach the currently installed LockingLogger. Each test installs a fresh logger
    # and restores the previous one in a finally block.
    def test_bump_numeric_counter(self):
        # [FUNCTION DOC] Scripts a module whose forward loops over the first dimension adding a
        # counter named foo each iteration and bumps either positive or negative afterwards.
        # Runs it five times on random 3x4x5 inputs and asserts foo totals 15 and positive
        # totals 5.
        class ModuleThatLogs(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a script_method forward that updates tensor x
            # in place and logs stat values; used only by the counter test above.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted forward. Adds one to x per leading-dim row,
                # logging foo each time, then logs positive or negative depending on the sign of
                # the sum. Returns x.
                for _ in range(x.size(0)):
                    x += 1.0
                    torch.jit._logging.add_stat_value("foo", 1)

                if bool(x.sum() > 0.0):
                    torch.jit._logging.add_stat_value("positive", 1)
                else:
                    torch.jit._logging.add_stat_value("negative", 1)
                return x

        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtl = ModuleThatLogs()
            for _ in range(5):
                mtl(torch.rand(3, 4, 5))

            self.assertEqual(logger.get_counter_val("foo"), 15)
            self.assertEqual(logger.get_counter_val("positive"), 5)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_trace_numeric_counter(self):
        # [FUNCTION DOC] Traces a function that logs a stat value, then installs a logger and
        # calls the traced function once; asserts the counter is 1, showing logging calls
        # survive tracing.
        def foo(x):
            # [FUNCTION DOC] HELPER: Traced function; logs counter foo with value one and returns
            # x + 1.
            torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter(self):
        # [FUNCTION DOC] Runs an eager-mode (non-scripted) module that records time points around
        # a loop and logs the difference; asserts the mytimer counter is greater than zero.
        class ModuleThatTimes(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule subclass whose forward is not decorated, so it runs
            # as plain Python and calls the logging functions eagerly.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Eager forward; takes start and end time points around 30
                # in-place additions, logs the elapsed value as mytimer, and returns x.
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter_script(self):
        # [FUNCTION DOC] Same as the eager timer test but forward is compiled via script_method;
        # asserts the mytimer counter logged during the call is greater than zero.
        class ModuleThatTimes(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with a script_method forward that measures elapsed
            # time of a loop and logs it.
            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Scripted forward; records time points around 30 in-place
                # additions, logs the difference as mytimer, returns x.
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_counter_aggregation(self):
        # [FUNCTION DOC] Traces a function that logs foo three times, sets the foo aggregation
        # type to AVG on the logger, runs the traced function once and asserts the counter value
        # is 1 (the average of three ones).
        def foo(x):
            # [FUNCTION DOC] HELPER: Traced function; logs foo with value one three times, returns
            # x + 1.
            for _ in range(3):
                torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        logger.set_aggregation_type("foo", torch.jit._logging.AggregationType.AVG)
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_logging_levels_set(self):
        # [FUNCTION DOC] Sets the C++ JIT logging option to the string foo and asserts the getter
        # returns the same string. Does not reset the option afterwards (global side effect).
        torch._C._jit_set_logging_option("foo")
        self.assertEqual("foo", torch._C._jit_get_logging_option())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
