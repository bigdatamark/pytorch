# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Test for the XPU scope profiler (per-kernel hardware metrics via XPUPTI).
# Defines one TestCase that profiles a few XPU ops with metric names requested through the
# fine-grained activity dict, exports a chrome trace, and validates the JSON structure and
# metric counts. Also checks that activity-type-only filters do not enable the scope profiler.
# Import-time side effects: none beyond imports and a module-level Verbose debug flag.

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import torch
from torch.testing._internal.common_utils import run_tests, TEST_XPU, TestCase


Verbose = False


class XpuScopeProfilerTest(TestCase):
    # [CLASS DOC] Test case holding the XPU scope profiler checks. The attribute metrics_names
    # lists the six hardware metric names that are requested and later expected in the trace.
    # Helper methods validate trace events and environment prerequisites (Level Zero metrics
    # flag and perf paranoid sysfs settings); the test methods need an XPU device.
    def __init__(self, arg):
        # [FUNCTION DOC] HELPER: Standard TestCase constructor taking the test method name;
        # forwards it to the parent and sets metrics_names to the six metric names
        # (GpuTime, GpuCoreClocks, AvgGpuCoreFrequencyMHz, an ALU utilization counter,
        # XVE_ACTIVE and XVE_STALL).
        super().__init__(arg)

        self.metrics_names = [
            "GpuTime",
            "GpuCoreClocks",
            "AvgGpuCoreFrequencyMHz",
            "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION",
            "XVE_ACTIVE",
            "XVE_STALL",
        ]

    def count_metrics(self, event, counter):
        # [FUNCTION DOC] HELPER: Validate one trace event and tally its metric names.
        # Asserts the event has an args entry that is a mapping, then increments, for each key
        # of that mapping, the count stored under that key in the supplied counter mapping
        # (mutated in place). Returns nothing.
        self.assertTrue("args" in event)
        self.assertTrue(isinstance(event["args"], dict))

        for arg in event["args"]:
            counter[arg] += 1

    def check_metrics(self, counter):
        # [FUNCTION DOC] HELPER: Check that every expected metric appears equally often.
        # Walks the counter mapping of metric key to occurrence count; a key matches a metric
        # name if equal to it or prefixed by the name plus a double colon or a space and
        # bracket (unit suffix forms). Asserts each key matches at most one name, groups
        # matching keys by their count, then asserts only one distinct count exists and that
        # exactly as many keys matched as there are metric names. Prints debug info if Verbose.
        rev_counter = defaultdict(int)

        for metric_name_in_json in counter:
            # Kineto emits the metric either bare (counter "C" events) or with a
            # unit suffix. The scope profiler's per-kernel metadata uses
            # "<name>::<unit>" (e.g. "GpuTime::ns"); some loggers use "<name> [..]".
            metric_name_in_json_valid = sum(
                metric_name_in_json == metric_name
                or metric_name_in_json.startswith(
                    (metric_name + "::", metric_name + " [")
                )
                for metric_name in self.metrics_names
            )
            self.assertTrue(metric_name_in_json_valid <= 1)
            if metric_name_in_json_valid == 1:
                rev_counter[counter[metric_name_in_json]] += 1

        if Verbose:
            print(f"{rev_counter=}")

        # keys are amounts of certain metrics
        # only one dict element means all metrics have the same amount
        self.assertEqual(len(rev_counter), 1)
        first_value = next(iter(rev_counter.values()))
        self.assertEqual(first_value, len(self.metrics_names))

    def gen_and_check_json(self, p, json_file):
        # [FUNCTION DOC] HELPER: Export the profiler trace to a file and validate it.
        # Args: p is a finished profiler, json_file the output path. Loads the JSON and asserts
        # a non-empty list of trace events, each with phase and name. Complete ("X") events
        # must have a category; names and categories are counted, and events named with the
        # metrics prefix feed the X-metric counter, while counter ("C") events feed the
        # C-metric counter. Asserts at least one metrics event exists, that the xpu_runtime and
        # kernel categories are present, and runs check_metrics on both counters.
        p.export_chrome_trace(json_file)

        with open(json_file) as f:
            data = json.load(f)
            self.assertTrue("traceEvents" in data)

            trace_events = data["traceEvents"]
            self.assertTrue(isinstance(trace_events, list))
            self.assertTrue(len(trace_events) > 0)

            count_names = defaultdict(int)
            count_cats = defaultdict(int)
            count_x_metrics = defaultdict(int)
            count_c_metrics = defaultdict(int)
            for event in trace_events:
                self.assertTrue("ph" in event)
                self.assertTrue("name" in event)

                if event["ph"] == "X":
                    self.assertTrue("cat" in event)
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

                    if event["name"].startswith("metrics: "):
                        self.count_metrics(event, count_x_metrics)

                elif event["ph"] == "C":
                    self.count_metrics(event, count_c_metrics)

            if Verbose:
                print(f"{count_names=}")
                print(f"{count_cats=}")
                print(f"{count_x_metrics=}")
                print(f"{count_c_metrics=}")

            metric_name_in_json_valid = sum(
                name.startswith("metrics: ") for name in count_names
            )
            self.assertTrue(metric_name_in_json_valid > 0)

            self.assertTrue("xpu_runtime" in count_cats)
            self.assertTrue("kernel" in count_cats)

            self.check_metrics(count_x_metrics)
            self.check_metrics(count_c_metrics)

    def required_env_setting(self):
        # [FUNCTION DOC] HELPER: Return True only if the ZET_ENABLE_METRICS environment
        # variable equals the string 1 (Level Zero metrics enabled). Prints it if Verbose.
        zet_enable_metrics = os.environ.get("ZET_ENABLE_METRICS", "unset")
        if Verbose:
            print(f"ZET_ENABLE_METRICS={zet_enable_metrics}")
        return zet_enable_metrics == "1"

    def required_paranoid_setting(self, paranoid_path):
        # [FUNCTION DOC] HELPER: Return True if the given sysfs paranoid file exists and its
        # stripped content is the string 0. A missing file yields False. Prints debug info if
        # Verbose.
        try:
            paranoid_content = Path(paranoid_path).read_text().strip()
            if Verbose:
                print(f"{paranoid_path} contains: {paranoid_content}")
            return paranoid_content == "0"

        except FileNotFoundError:
            if Verbose:
                print(f"{paranoid_path} doesn't exists")
            return False

    def required_paranoid_settings(self):
        # [FUNCTION DOC] HELPER: Return True if either the xe observation paranoid file or the
        # i915 perf stream paranoid file permits profiling (content 0), checking both paths.
        return any(
            self.required_paranoid_setting(paranoid)
            for paranoid in [
                "/proc/sys/dev/xe/observation_paranoid",
                "/proc/sys/dev/i915/perf_stream_paranoid",
            ]
        )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_scope_profiler(self):
        # [FUNCTION DOC] End-to-end scope profiler test; skipped unless an XPU is available,
        # ZET_ENABLE_METRICS is 1 and a paranoid setting is 0. Runs matmul, add and abs on XPU
        # under a profiler with CPU activity plus an XPU entry listing the metric names, then
        # exports the trace (to the JSON_FILE environment path if set, else a temp file) and
        # validates it. Finally asserts the key averages contain at least one metrics event.
        if not self.required_env_setting():
            self.skipTest("ZET_ENABLE_METRICS not set")

        if not self.required_paranoid_settings():
            self.skipTest("paranoid settings missing")

        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        # Hardware metrics are requested through the fine-grained activity dict:
        # {ProfilerActivity.XPU: [<metric names>]} enables the XPUPTI scope
        # profiler and forwards the metric names to it.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                {torch.profiler.ProfilerActivity.XPU: self.metrics_names},
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

        count_metrics = sum(
            event.key.startswith("metrics: ") for event in p.key_averages()
        )
        self.assertTrue(count_metrics > 0)

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_activity_type_filter_does_not_enable_scope_profiler(self):
        # [FUNCTION DOC] Verifies that an XPU activity dict listing only an activity type name
        # (a kernel type, not a hardware metric) does not enable the scope profiler. Profiles a
        # matmul on XPU, then asserts the result is non-empty and that zero key-average events
        # carry the metrics prefix. Needs XPU but no metrics hardware; skipped without XPU.
        """A fine-grained XPU dict that lists only activity type names (no
        hardware metric names) filters activities as in #176351 and must not
        turn on the per-kernel scope profiler. Needs no metrics hardware."""
        a = torch.rand([100, 200]).to("xpu")
        b = torch.rand([200, 300]).to("xpu")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                {torch.profiler.ProfilerActivity.XPU: ["CONCURRENT_KERNEL"]},
            ],
        ) as p:
            result = torch.matmul(a, b)

        self.assertTrue(result.numel() > 0)
        count_metrics = sum(
            event.key.startswith("metrics: ") for event in p.key_averages()
        )
        self.assertEqual(count_metrics, 0)


if __name__ == "__main__":
    run_tests()
