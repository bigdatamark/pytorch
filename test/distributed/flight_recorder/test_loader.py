# [MODULE DOC] Tests that the flight recorder loader function read_dir leaves the Python garbage
# collector in its original enabled or disabled state, on success and on failure. Uses temporary
# directories and pickle dump files; no import-time side effects.
# Owner(s): ["oncall: distributed"]

import argparse
import gc
import os
import pickle
import tempfile

from torch.distributed.flight_recorder.components.loader import read_dir
from torch.testing._internal.common_utils import run_tests, TestCase


class FlightRecorderLoaderTest(TestCase):
    # [CLASS DOC] Tests read_dir's handling of the global garbage collector state, which it must
    # restore after reading.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and registers a cleanup that restores the garbage
        # collector to the enabled or disabled state it had before the test.
        super().setUp()
        restore_gc = gc.enable if gc.isenabled() else gc.disable
        self.addCleanup(restore_gc)

    def _write_dump(self, trace_dir):
        # [FUNCTION DOC] HELPER: Writes a pickle file named dump0 in the given directory containing
        # an empty entries list, version 2.8 and an empty pg_config.
        path = os.path.join(trace_dir, "dump0")
        with open(path, "wb") as output:
            pickle.dump(
                {
                    "entries": [],
                    "version": "2.8",
                    "pg_config": {},
                },
                output,
            )

    def test_read_dir_restores_enabled_gc_after_success(self):
        # [FUNCTION DOC] Writes a dump, enables the garbage collector, runs read_dir for prefix
        # dump and checks the collector is still enabled.
        with tempfile.TemporaryDirectory() as trace_dir:
            self._write_dump(trace_dir)
            gc.enable()

            read_dir(argparse.Namespace(trace_dir=trace_dir, prefix="dump"))

            self.assertTrue(gc.isenabled())

    def test_read_dir_restores_enabled_gc_after_failure(self):
        # [FUNCTION DOC] Enables the garbage collector, checks read_dir on a missing directory
        # raises AssertionError mentioning it does not exist, and checks the collector is
        # enabled afterwards.
        gc.enable()

        with self.assertRaisesRegex(AssertionError, "does not exist"):
            read_dir(
                argparse.Namespace(
                    trace_dir="missing-flight-recorder-dir",
                    prefix=None,
                )
            )

        self.assertTrue(gc.isenabled())

    def test_read_dir_preserves_disabled_gc(self):
        # [FUNCTION DOC] Writes a dump, disables the garbage collector, runs read_dir and checks it
        # is still disabled.
        with tempfile.TemporaryDirectory() as trace_dir:
            self._write_dump(trace_dir)
            gc.disable()

            read_dir(argparse.Namespace(trace_dir=trace_dir, prefix="dump"))

            self.assertFalse(gc.isenabled())


if __name__ == "__main__":
    run_tests()
