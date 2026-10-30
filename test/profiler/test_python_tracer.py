# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests for the Python call tracer used by the profiler when stack recording is on.
# Covers a C function wrapped as a classmethod, the sys.monitoring tool registration on Python
# 3.12, and a subprocess regression check for unexpected C-return events across threads. Runs
# through the standard test runner when executed as a script; no import-time side effects.

import json
import subprocess
import sys
import time

from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfPythonVersionMismatch,
    TemporaryFileName,
    TestCase,
)


class TestPythonTracer(TestCase):
    # [CLASS DOC] Test case holding the Python-tracer profiler tests. Stateless; each test builds
    # its own profiler session. Two tests only run on Python 3.12 exactly.
    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_method_with_c_function(self):
        # [FUNCTION DOC] Profiles a sort whose key function calls a C builtin bound as a
        # classmethod, and sleeps 1.2 seconds per key call. Exports a chrome trace to a temporary
        # file, loads the trace events and asserts a python_function event named for the builtin
        # sorted has duration of at least 2 seconds (microseconds in the trace). Only runs on
        # Python 3.12.
        class A:
            # [CLASS DOC] HELPER: Local class whose class attribute is a classmethod wrapping the
            # builtin repr, so calling it goes through a C function bound as a method.
            method_with_c_function = classmethod(repr)

        def get_key(x):
            # [FUNCTION DOC] HELPER: Sort key function. Invokes the classmethod-wrapped C
            # function on a new instance of the local class, sleeps 1.2 seconds, and returns the
            # length of the argument.
            A().method_with_c_function()
            time.sleep(1.2)
            return len(x)

        names = ["Alice", "Bob"]

        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            sorted(names, key=get_key)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
                found = False
                for event in events:
                    if (
                        event.get("cat", "") == "python_function"
                        and event.get("name", "") == "<built-in function sorted>"
                    ):
                        duration = event.get("dur", 0)
                        if duration >= 2000000:
                            found = True
                            break
                self.assertTrue(found)

    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_monitoring_callback(self):
        # [FUNCTION DOC] Checks the profiler's sys.monitoring tool slot 2 behavior on Python
        # 3.12. While profiling with stack recording, the tool name in slot 2 must be "PyTorch
        # Profiler" when the micro version is below 5, otherwise None; after the profiler exits
        # the slot must be None again. Only runs on Python 3.12.
        vi = sys.version_info
        from sys import monitoring

        with profile(activities=[ProfilerActivity.CPU], with_stack=True):
            name = monitoring.get_tool(2)
            if vi.micro < 5:
                self.assertEqual(name, "PyTorch Profiler")
            else:
                self.assertEqual(name, None)
        name = monitoring.get_tool(2)
        self.assertEqual(name, None)

    def test_unexpected_c_return_events(self):
        # [FUNCTION DOC] Regression test: runs a script in a fresh interpreter subprocess in
        # which a worker thread blocks on events and a lock while the main thread profiles twice
        # with stack recording, so C-return events arrive without matching calls. The subprocess
        # must exit cleanly (check=True) and its stderr must not contain the message about an
        # empty Python replay stack during a pop operation.
        code = """
import threading
import time
import torch

from threading import Event, Lock

lock = Lock()
lock.acquire()
event1 = Event()
event2 = Event()
event3 = Event()

def run():
    event1.set()
    event2.wait()
    lock.acquire()
    event3.set()

threading.Thread(target=run).start()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    event1.wait()
    event2.set()
    time.sleep(1)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    lock.release()
    event3.wait()
    """

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        self.assertFalse(
            "Python replay stack is empty during pop operation" in result.stderr
        )


if __name__ == "__main__":
    run_tests()
