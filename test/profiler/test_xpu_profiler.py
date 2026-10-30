# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Intel XPU profiler tests. Defines a module-level verbosity flag (off by default)
# that makes the tests print collected counts, and one test class whose tests need an XPU device
# and are skipped otherwise. Checks device event collection, the overhead category in exported
# chrome traces, and the structure of an exported trace. Runs through the standard test runner.


import json
import os
import tempfile
import unittest
from collections import defaultdict

import torch
from torch.profiler import DeviceType
from torch.testing._internal.common_utils import run_tests, TEST_XPU, TestCase


Verbose = False


class XpuProfilerTest(TestCase):
    # [CLASS DOC] XPU profiler test case. Stateless apart from the module verbosity flag; all
    # tests except the JSON helper are skipped when no XPU is present.
    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler(self):
        # [FUNCTION DOC] Zeroes an integer tensor on the XPU ten times under a profiler enabled
        # for XPU activity only, counts profiler events per device type, and asserts exactly two
        # device types were seen: CPU and XPU.
        t = torch.empty(1000, dtype=torch.int, device="xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as p:
            for _ in range(10):
                t.zero_()

        events = defaultdict(int)
        for event in p.events():
            events[event.device_type] += 1

        if Verbose:
            print(f"{events=}")

        self.assertEqual(len(events), 2)
        self.assertTrue(DeviceType.CPU in events)
        self.assertTrue(DeviceType.XPU in events)

    def gen_and_check_json(self, p, json_file):
        # [FUNCTION DOC] HELPER: Exports the profile as a chrome trace to the given path, reloads
        # it, and asserts: a trace events list exists and is non-empty; every event has a phase
        # and name; complete events (phase X) have a category. Counts names and categories of
        # complete events and asserts the categories include xpu_runtime, xpu_driver and kernel.
        # Prints counts if verbose.
        p.export_chrome_trace(json_file)

        with open(json_file) as f:
            data = json.load(f)
            self.assertTrue("traceEvents" in data)

            trace_events = data["traceEvents"]
            self.assertTrue(isinstance(trace_events, list))
            self.assertTrue(len(trace_events) > 0)

            count_names = defaultdict(int)
            count_cats = defaultdict(int)
            for event in trace_events:
                self.assertTrue("ph" in event)
                self.assertTrue("name" in event)

                if event["ph"] == "X":
                    self.assertTrue("cat" in event)
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

            if Verbose:
                print(f"{count_names=}")
                print(f"{count_cats=}")

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("xpu_driver" in count_cats)
            self.assertTrue("kernel" in count_cats)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_overhead(self):
        # [FUNCTION DOC] Profiles ten matrix multiplications on the XPU with CPU and XPU
        # activities, exports a chrome trace to a temporary file and loads it. Asserts at least
        # one complete event in the overhead category exists and that each carries overhead cost
        # and overhead count arguments.
        # The OVERHEAD activity type surfaces the profiler's own collection cost
        # on a dedicated track, matching CUDA behaviour. It is enabled together
        # with ProfilerActivity.XPU via kXpuTypes. PTI instruments the device
        # operations it traces, so a workload that issues device ops is expected
        # to produce overhead records.
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        ) as p:
            for _ in range(10):
                r = torch.matmul(a, b)

        self.assertTrue(r.numel() > 0)

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            p.export_chrome_trace(tmp.name)
            with open(tmp.name) as f:
                data = json.load(f)

            overhead_events = [
                e
                for e in data["traceEvents"]
                if e.get("ph") == "X" and e.get("cat") == "overhead"
            ]

            if Verbose:
                print(f"{len(overhead_events) = }")

            self.assertGreater(len(overhead_events), 0)
            for e in overhead_events:
                args = e.get("args", {})
                self.assertIn("overhead cost", args)
                self.assertIn("overhead count", args)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_xpu_driver(self):
        # [FUNCTION DOC] Profiles a matmul, add and abs chain on the XPU with CPU and XPU
        # activities. Validates the exported trace through the JSON helper, writing to the path
        # in the JSON_FILE environment variable if set, otherwise to a temporary file. Prints the
        # key averages table when verbose.
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
        ) as p:
            r1 = torch.matmul(a, b)
            r2 = torch.add(r1, 1.0)
            result = torch.abs(r2)

        self.assertTrue(result.numel() > 0)

        json_file = os.environ.get("JSON_FILE")

        if json_file:
            self.gen_and_check_json(p, json_file)
        else:
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
                self.gen_and_check_json(p, tmp.name)

        if Verbose:
            print(p.key_averages().table())


if __name__ == "__main__":
    run_tests()
