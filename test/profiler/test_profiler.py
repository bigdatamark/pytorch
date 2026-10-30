# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the profiler stack: the autograd profiler, torch.profiler profile with
# schedules and record_function, Kineto chrome-trace export, the events list API, memory and FLOPs
# profiling, Python tracing and the experimental utilities. Test classes cover CUDA, ITT, the
# general CPU profiler, a per-device class instantiated for each backend, tree utilities,
# private-use backend state, the device-stopped schedule action, events/trace parity, the Python
# trace exporter and kernel metadata format. At import time it sets the Kineto log level
# environment variable if unset, turns off the tqdm monitor thread when tqdm exists, probes for
# psutil, and instantiates the per-device test class. Running it directly calls run_tests.

import collections
import copy
import gc
import json
import mmap
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from typing import TYPE_CHECKING
from unittest.mock import patch


# Suppress libkineto USDT profiler_start/profiler_stop logs in this verbose
# profiler test file. USDT is the highest libkineto log type, so use one level
# above it.
os.environ.setdefault("KINETO_LOG_LEVEL", "6")

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch._inductor.utils import is_big_gpu
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyOn,
    skipIf,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_JETSON,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    run_tests,
    serialTest,
    skipIfRocm,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
    TestCase,
    xfailIfNoAcceleratorTriton,
)


if TYPE_CHECKING:
    from torch.autograd.profiler_util import FunctionEvent


def get_profiler_activities(device_type):
    # [FUNCTION DOC] Returns the profiler activity list for a device type string: always CPU, plus
    # the matching device activity (looked up by upper-cased device name) when the device is not
    # cpu or meta and that activity is reported by supported_activities.
    activities = [ProfilerActivity.CPU]
    if device_type not in ("cpu", "meta"):
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        if device_activity and device_activity in supported_activities():
            activities.append(device_activity)
    return activities


def setUpModule():
    # [FUNCTION DOC] HELPER: unittest module-level fixture. When Kineto is available, CUDA exists
    # and the CUDA activity is supported, runs one tiny add on a CUDA tensor under a CPU+CUDA
    # profile and synchronizes, so Kineto initializes with CUDA support before CPU-only tests run.
    if (
        kineto_available()
        and torch.cuda.is_available()
        and ProfilerActivity.CUDA in supported_activities()
    ):
        # Kineto's process-global profiler cannot currently upgrade from a
        # CPU-only first initialization to CUDA-capable profiling. Prime it with
        # CUDA so CPU-only tests do not poison later CUDA profiler tests.
        x = torch.ones(1, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
            x + x
            torch.cuda.synchronize()


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerCUDA(TestCase):
    # [CLASS DOC] CUDA-only profiler tests (skipped without psutil, on Windows, or without CUDA).
    # Covers memory-leak checking, CUDA graph workaround in a subprocess, multi-GPU events, OOM
    # tracing, CUDA sync events, external-correlation disabling and mixed activity filters.
    def payload(self, device="cpu", tensor_size=10):
        # [FUNCTION DOC] HELPER: workload used by profiling tests. Creates two random square
        # matrices of the given size on the given device, multiplies them, and adds one back.
        # Returns nothing.
        x = torch.randn(tensor_size, tensor_size).to(device)
        y = torch.randn(tensor_size, tensor_size).to(device)
        z = torch.mm(x, y)
        z = z + y

    def _check_stats(self, profiler_stats):
        # [FUNCTION DOC] HELPER: asserts that every timing and count field of a profiler stats
        # object (window duration, event count, prepare/enable/disable/parse/build-tree durations)
        # is positive.
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/78457")
    def test_mem_leak(self):
        # [FUNCTION DOC] Runs ten profiled sessions of repeated CUDA matmuls, collecting the
        # process RSS after each (last five kept). Fails only if RSS grew every time and a single
        # step grew by more than 100 KiB. Skipped on ROCm.
        """Checks that there's no memory leak when using profiler with CUDA"""
        t = torch.rand(1, 1).cuda()
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            with _profile(use_device="cuda"):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.cuda.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))
        )
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        self.assertTrue(
            not (is_increasing and max_diff > 100 * 1024),
            msg=lambda msg: f"{msg}\nmemory usage is increasing, {str(last_rss)}",
        )

    def test_custom_module_input_op_ids(self):
        # [FUNCTION DOC] Smoke test: runs a custom autograd Function forward and backward under
        # emit_nvtx with record_shapes enabled; makes no assertions beyond not raising.
        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: identity autograd Function that saves its input; its
            # backward returns the saved input rather than a real gradient (fine here
            # since values are unchecked).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: saves x for backward and returns it unchanged.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: unpacks the single saved tensor and returns
                # it as the gradient.
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            # [FUNCTION DOC] HELPER: applies the custom Function to the given tensor and returns
            # the result.
            return MyFunc.apply(input_ten)

        # Only testing that emit_nvtx runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_nvtx(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

    def test_cudagraph_profiling_workaround(self):
        # [FUNCTION DOC] Launches a fresh Python subprocess (60 second timeout) that
        # initializes the profiler CUDA-graph workaround, builds a graphed callable,
        # checks its output, and profiles it under CPU and then CUDA activities. The
        # check call raises if the child exits non-zero.
        import subprocess

        # repro taken from #75504
        # Launch in a separate process to catch hanging/illegal memory errors
        # and to make sure CUPTI isn't already initialized.
        p = subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
import os
import torch
from torch.profiler import ProfilerActivity, profile

def add_one(in_: torch.Tensor):
    return in_ + 1

sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)

# add this before cuda graphs are created
torch.profiler._utils._init_for_cuda_graphs()

add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))
zeros = torch.zeros(10, device="cuda")
out = add_one_graphed(zeros)
if out[0] != 1:
    raise AssertionError(f"Expected out[0] == 1, got {out[0]}")

with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)
""",
            ],
            text=True,
            timeout=60,
        )

        # ^ this will throw an exception if the script fails.

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    def test_kineto_multigpu(self):
        # [FUNCTION DOC] Profiles matmuls on GPUs 0 and 1, then checks that gemm kernels were seen
        # on two distinct CUDA devices (count only on ROCm, exact indices zero and one on CUDA)
        # and that a CPU event mentioning the device runtime exists. Also validates profiler
        # stats. Needs 2+ GPUs.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for gpu_id in [0, 1]:
                x = torch.randn(10, 10).cuda(gpu_id)
                y = torch.randn(10, 10).cuda(gpu_id)
                z = x.matmul(y)
                torch.cuda.synchronize(gpu_id)

        is_rocm = torch.version.hip is not None
        # on ROCm, Gemm shader is hipblaslt Shader, so we use UserArgs_MT to match.
        gemm_string = "userargs_mt" if is_rocm else "gemm"
        device_string = "hip" if is_rocm else "cuda"

        device_indices = set()
        found_cuda = False
        for evt in prof.events():
            if gemm_string in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                device_indices.add(evt.device_index)
            if device_string in evt.name.lower() and evt.device_type == DeviceType.CPU:
                found_cuda = True

        if is_rocm:
            # Note: On ROCm, device_indices (Node IDs) may start from values other than 0 (e.g. {2, 3})
            # because systems can contain additional (non-GPU) devices detected by the kernel,
            # resulting in offset indexing. Therefore, we validate the count of unique devices,
            # not their specific indices.
            self.assertEqual(len(device_indices), 2)
        else:
            # CUDA correctly reports logical device indices
            self.assertEqual(device_indices, {0, 1})

        self.assertTrue(found_cuda)
        self._check_stats(prof._stats())

    @unittest.skipIf(
        IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared"
    )
    def test_oom_tracing(self):
        # [FUNCTION DOC] Profiles an oversized CUDA allocation that must raise an out-of-memory
        # RuntimeError, exports a chrome trace, and checks it has an instantaneous OutOfMemory
        # event with device type, device id and bytes args. Skipped on Jetson.
        def run_profiler(tensor_creation_fn):
            # [FUNCTION DOC] HELPER: runs the given tensor-creating callable under a
            # memory/shape profile, expecting a RuntimeError about failed allocation,
            # and returns the profiler object.
            with _profile(profile_memory=True, record_shapes=True) as prof:
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        def create_cuda_tensor_oom():
            # [FUNCTION DOC] HELPER: tries to allocate a huge float32 tensor on cuda:0
            # to trigger OOM.
            device = torch.device("cuda:0")
            return torch.empty(
                1024, 1024, 1024, 1024, dtype=torch.float32, device=device
            )

        def check_trace(fname):
            # [FUNCTION DOC] HELPER: exports the enclosing test's profiler (captured by closure)
            # to the given file, reloads the JSON, and asserts an OutOfMemory event exists with
            # the required args and no duration or category fields.
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                self.assertTrue("traceEvents" in trace)
                events = trace["traceEvents"]
                found_out_of_memory_events = False
                for evt in events:
                    self.assertTrue("name" in evt)
                    if evt["name"] == "[OutOfMemory]":
                        found_out_of_memory_events = True
                        self.assertTrue("args" in evt)
                        self.assertTrue("Device Type" in evt["args"])
                        self.assertTrue("Device Id" in evt["args"])
                        self.assertTrue("Bytes" in evt["args"])

                        # Memory should be an instantaneous event.
                        self.assertTrue("dur" not in evt["args"])
                        self.assertTrue("cat" not in evt["args"])
                self.assertTrue(found_out_of_memory_events)

        with TemporaryFileName(mode="w+") as fname:
            prof = run_profiler(create_cuda_tensor_oom)
            check_trace(fname)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_cuda_sync_events(self):
        # [FUNCTION DOC] Checks that CUDA synchronize calls appear in the exported trace when sync
        # events are enabled, both through the experimental config flag and through the global
        # setter (which is reset in a finally block). Prints a progress line for each variant.
        device = torch.device("cuda:0")
        t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)

        def workload() -> None:
            # [FUNCTION DOC] HELPER: adds two tensors, synchronizes CUDA, adds again.
            torch.add(t1, t2)
            torch.cuda.synchronize()
            torch.add(t1, t2)

        def trace_and_check(exp_config: _ExperimentalConfig | None) -> None:
            # [FUNCTION DOC] HELPER: profiles the workload with the given experimental config,
            # exports a chrome trace to a temporary file, collects event categories (or names on
            # ROCm) and asserts the sync event name is present.
            with _profile(
                use_kineto=True,
                use_device="cuda",
                experimental_config=exp_config,
            ) as prof:
                workload()

            with TemporaryFileName(mode="w+") as fname:
                # fname = "/tmp/kineto_out.json"
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    cat_or_name = "cat" if torch.version.cuda else "name"
                    cats = {e.get(cat_or_name, None) for e in j["traceEvents"]}
            event_name = "cuda_sync" if torch.version.cuda else "hipDeviceSynchronize"
            self.assertTrue(
                event_name in cats,
                lambda msg: f"{msg}\nExpected to find {event_name} event found = {cats}",
            )

        print("Testing enable_cuda_sync_events in _ExperimentalConfig")
        trace_and_check(exp_config=_ExperimentalConfig(enable_cuda_sync_events=True))

        print("Testing _profiler._set_cuda_sync_enabled_val()")
        try:
            torch._C._profiler._set_cuda_sync_enabled_val(True)
            trace_and_check(exp_config=None)
        finally:
            torch._C._profiler._set_cuda_sync_enabled_val(False)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_disable_external_correlation(self):
        # [FUNCTION DOC] Profiles a CUDA workload with external correlation disabled
        # and enabled, exports traces, and checks that CUDA runtime/memcpy/kernel
        # events carry an external id only when correlation is enabled (device
        # synchronize events excluded). Skipped under dynamo.
        cuda_external_id_events = {"cuda_runtime", "gpu_memcpy", "kernel"}
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        def check_correlations(event, disable_external_correlation):
            # [FUNCTION DOC] HELPER: for events in the CUDA-related categories,
            # asserts the external id arg is absent when disabled, otherwise present
            # and positive (except sync calls).
            if "cat" in event and event["cat"] in cuda_external_id_events:
                if disable_external_correlation:
                    self.assertTrue("External id" not in event["args"])
                else:
                    excluded_events = (
                        {"hipDeviceSynchronize"}
                        if TEST_WITH_ROCM
                        else {"cudaDeviceSynchronize"}
                    )
                    if event["name"] not in excluded_events:
                        self.assertTrue("External id" in event["args"])
                        self.assertTrue(event["args"]["External id"] > 0)

        def validate_json(prof, disable_external_correlation):
            # [FUNCTION DOC] HELPER: exports the profile to a temp JSON file, checks each event,
            # records seen categories, and asserts all CUDA-related categories were seen.
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    seen_event_types = set()
                    for event in events:
                        check_correlations(event, disable_external_correlation)
                        if "cat" in event:
                            seen_event_types.add(event["cat"])
                    self.assertTrue(cuda_external_id_events.issubset(seen_event_types))

        # Run with External Id for CUDA events on and off
        for disable_external_correlation in [False, True]:
            with profile(
                activities=activities,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    disable_external_correlation=disable_external_correlation
                ),
            ) as prof:
                self.payload(device="cuda", tensor_size=256)
            validate_json(prof, disable_external_correlation)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_mixed_syntax(self):
        # [FUNCTION DOC] Verifies that a mix of a plain activity enum and a mapping from CUDA
        # activity to a list of names is accepted by profile, and that events are still recorded.
        """Enum and dict entries can coexist for different activity groups."""
        activities = [ProfilerActivity.CPU, {ProfilerActivity.CUDA: ["GPU_MEMCPY"]}]
        with profile(activities=activities) as p:
            with record_function("test_annotation"):
                x = torch.randn(10, 10).to("cuda")
                y = torch.mm(x, x)
        self.assertGreater(len(p.events()), 0)


@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestProfilerITT(TestCase):
    # [CLASS DOC] Tests for the ITT (Intel) profiler hooks; skipped when ITT is unavailable.
    def test_custom_module_input_op_ids(self):
        # [FUNCTION DOC] Smoke test: runs a custom autograd Function forward and backward under
        # emit_itt with record_shapes enabled; makes no assertions beyond not raising.
        class MyFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: identity autograd Function that saves its input; backward
            # returns the saved input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: saves x for backward and returns it unchanged.
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                # [FUNCTION DOC] HELPER: unpacks the single saved tensor and returns
                # it as the gradient.
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            # [FUNCTION DOC] HELPER: applies the custom Function to the given tensor and returns
            # the result.
            return MyFunc.apply(input_ten)

        # Only testing that emit_itt runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_itt(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()


@instantiate_parametrized_tests
class TestProfiler(TestCase):
    # [CLASS DOC] Main CPU-oriented profiler test case (parametrized). Covers source attribution,
    # threading, Python tracing, record_function, schedules, memory profiling, export formats,
    # tensor-subclass and other feature interactions.
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    def test_source(self):
        # [FUNCTION DOC] Profiles eager, TorchScript and autograd code with stack recording and
        # checks that add events have stack entries naming this test file and the scripted or test
        # functions. With Kineto, also exports a trace and checks the DummyModule event's Python
        # parent id equals the wrapper function event's Python id. Temporarily turns off graph
        # executor optimization and restores it at the end (not in a finally block).
        """Checks that source code attribution works for eager, TS and autograd mode"""
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            # [FUNCTION DOC] HELPER: scripted matmul of two tensors.
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            # [FUNCTION DOC] HELPER: scripted function adding x and z, calling the scripted
            # matmul, adding and returning the sum of the result.
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            # [CLASS DOC] HELPER: small module with a single strided 2D convolution, used to check
            # module event naming and parent linkage.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: builds the convolution layer (3 in, 2 out channels, 1x1
                # kernel, stride 2, padding 3, no bias).
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the convolution to the input.
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            # [FUNCTION DOC] HELPER: calls the module on x; named so the trace shows a Python
            # wrapper frame.
            return mod(x)

        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = call_module(a)
            c = b.sum()
            c.backward()

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any("test_profiler" in entry for entry in e.stack))
                self.assertTrue(
                    any(
                        (
                            "test_source" in entry
                            or "ts_method_1" in entry
                            or "ts_method_2" in entry
                        )
                        for entry in e.stack
                    )
                )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    # [FUNCTION DOC] HELPER: returns the single trace event whose name matches the
                    # regex; asserts that exactly one matches.
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    self.assertEqual(
                        len(matches), 1, repr([e["name"] for e in matches])
                    )
                    return matches[0]

                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                self.assertEqual(
                    module_event["args"]["Python parent id"],
                    wrapper_event["args"]["Python id"],
                )

        torch._C._set_graph_executor_optimize(prev_opt)

    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            "multiple_preexisting": ((False, False),) * 2,
            "open_in_scope": ((True, False),),
            "close_in_scope": ((False, True),),
            "complex": (
                # Large number of background threads
                (False, False),
                (False, False),
                (False, False),
                (False, False),
                # some of which finish during profiling
                (False, True),
                (False, True),
                # And the profiled section is also multithreaded
                (True, False),
                (True, True),
            ),
        }.items(),
        name_fn=lambda name, thread_spec: name,
    )
    @serialTest()
    @parametrize("work_in_main_thread", [True, False])
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_source_multithreaded(self, name, thread_spec, work_in_main_thread):
        # [FUNCTION DOC] Parametrized threading test. Starts background worker threads before,
        # during or join/finish them inside the profile per thread_spec, optionally running the
        # work on the main thread, then checks the Python call tree: one thread id per
        # pre-existing thread plus main, no placeholder thread id, and that the pre-existing and
        # in-run functions were observed once per worker thread. Always aborts barriers and joins
        # threads in a finally block.
        """Test various threading configurations.

        `thread_spec` is a Tuple[Tuple[bool, bool], ...] where each pair is a
        thread. The first bool indicates if the thread should be started under
        the profiler context and the second is if it should be joined under the
        profiler context.
        """

        timeout = 15
        num_threads = len(thread_spec) + 1  # Main thread
        start_barrier = threading.Barrier(num_threads, timeout=timeout)
        end_barrier = threading.Barrier(num_threads, timeout=timeout)

        class Task(threading.Thread):
            # [CLASS DOC] HELPER: daemon thread that starts itself on construction and runs the
            # shared worker body, optionally waiting on an end gate before exiting.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the end-gate event, starts the daemon thread
                # immediately, and sets finished to false afterwards.
                self._end_gate = threading.Event()
                super().__init__(daemon=True)
                self.start()
                self.finished = False

            def run(self):
                # [FUNCTION DOC] HELPER: thread entry point; runs the worker body with this
                # thread's end gate.
                self._run(self._end_gate)

            def release(self):
                # [FUNCTION DOC] HELPER: opens the end gate so the thread may exit.
                self._end_gate.set()

            @staticmethod
            def _run(end_gate=None):
                # [FUNCTION DOC] HELPER: worker body. Waits at the start barrier inside a known
                # function, builds and runs a small linear+relu model on random input, waits at
                # the end barrier, then waits on the optional end gate with the timeout.
                def known_preexisting_function():
                    # [FUNCTION DOC] HELPER: blocks on the start barrier; a fixed function that is
                    # already running when profiling begins.
                    start_barrier.wait()

                # Fixed point that we can use to test capture of functions
                # which are already running when profiling is enabled.
                known_preexisting_function()

                model = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )

                def invoked_during_run():
                    # [FUNCTION DOC] HELPER: empty function whose call is expected to appear in
                    # the traced call tree.
                    pass

                invoked_during_run()

                _ = model(torch.rand(4, 10))
                end_barrier.wait()

                if end_gate is not None:
                    end_gate.wait(timeout=timeout)

        threads = {}

        def add_threads(context: bool):
            # [FUNCTION DOC] HELPER: creates a Task for every spec entry whose
            # start-under-profiler flag equals the context flag, storing it by index; raises
            # AssertionError on a duplicate index.
            for idx, (start_under_profiler, _) in enumerate(thread_spec):
                if start_under_profiler == context:
                    if idx in threads:
                        raise AssertionError(f"Thread index {idx} already exists")
                    threads[idx] = Task()

        def join_threads(context: bool):
            # [FUNCTION DOC] HELPER: releases then joins (with timeout) the threads whose
            # end-under-profiler flag equals the context flag.
            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                if end_under_profiler == context:
                    threads[idx].release()

            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                t = threads[idx]
                if end_under_profiler == context:
                    t.join(timeout=timeout)

        try:
            add_threads(False)
            with torch.profiler.profile(with_stack=True) as prof:
                # Threads added while the profiler are running will not be observed
                # since there is no way to hook into Python's thread start call to
                # register the observer. These are here purely to verify safety.
                add_threads(True)

                if work_in_main_thread:
                    Task._run()
                else:
                    start_barrier.wait()
                    end_barrier.wait()

                join_threads(True)
            join_threads(False)

        finally:
            # It is very important that we clean up everything because the
            # Python tracer will detect ALL active threads. (Even orphans from
            # prior failed tests.) If we don't clean up properly we can
            # contaminate subsequent tests.
            start_barrier.abort()
            end_barrier.abort()
            for t in threads.values():
                t.release()

            for t in threads.values():
                t.join(timeout=timeout)

            for t in threads.values():
                self.assertFalse(t.is_alive())

        roots = prof.profiler.kineto_results.experimental_event_tree()
        nodes = [
            node
            for node in _utils.traverse_dfs(roots)
            if isinstance(node.extra_fields, _ExtraFields_PyCall)
        ]
        tid_counts = collections.Counter([node.start_tid for node in nodes])

        prior_threads = sum(
            not start_under_profiler for start_under_profiler, _ in thread_spec
        )
        expected_threads = prior_threads + 1
        self.assertEqual(
            len(tid_counts),
            expected_threads,
            lambda msg: f"{msg}\n{expected_threads}, {tid_counts}",
        )
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2**64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [
            node.start_tid
            for node in nodes
            if "known_preexisting_function" in node.name
        ]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [
            node.start_tid for node in nodes if "invoked_during_run" in node.name
        ]
        self.assertEqual(len(observed_during_run), worker_threads)
        self.assertEqual(len(observed_during_run), len(set(observed_during_run)))

    def payload(self, device="cpu", tensor_size=10):
        # [FUNCTION DOC] HELPER: small CPU workload: two random matrices on the device, a matmul
        # and an add; returns nothing.
        x = torch.randn(tensor_size, tensor_size).to(device)
        y = torch.randn(tensor_size, tensor_size).to(device)
        z = torch.mm(x, y)
        z = z + y

    def _check_stats(self, profiler_stats):
        # [FUNCTION DOC] HELPER: asserts that every timing and count field of a profiler stats
        # object (window duration, event count, prepare/enable/disable/parse/build-tree durations)
        # is positive.
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_module_hierarchy(self):
        # [FUNCTION DOC] Scripts a small module tree (C containing A and B), profiles it on CPU
        # with module recording (FutureWarning suppressed), exports a chrome trace and checks that
        # the module-hierarchy argument of sub, mul and add events is one of the expected
        # hierarchy strings. Asserts at least one event was checked so the comparison is not
        # vacuous. Needs Kineto.
        class A(nn.Module):
            # [CLASS DOC] HELPER: module whose forward subtracts 2 from y then delegates to an
            # implementation method that triples x via another method and adds y; used to build a
            # nested hierarchy.
            def my_new_method(self, x):
                # [FUNCTION DOC] HELPER: returns x multiplied by 3.
                return x * 3

            def forward_impl_(self, x, y):
                # [FUNCTION DOC] HELPER: returns x tripled via the helper method plus y.
                return self.my_new_method(x) + y

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: subtracts 2 from y and returns the implementation
                # method's result.
                y = y - 2
                return self.forward_impl_(x, y)

        class B(nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds 2 to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns x plus 2.
                return x + 2

        class C(nn.Module):
            # [CLASS DOC] HELPER: top-level module owning submodules A0 (type A) and B0 (type B).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the A0 and B0 submodules.
                super().__init__()
                self.A0 = A()
                self.B0 = B()

            def call_b(self, x):
                # [FUNCTION DOC] HELPER: calls the forward of B0 on x and returns it.
                return self.B0.forward(x)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: returns the sum of A0 applied to x and y and the call_b
                # result on x.
                return self.A0.forward(x, y) + self.call_b(x)

        model = C()
        model = torch.jit.script(model)
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)
        op_to_module_hierarchy = {}
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."
        ]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.",
            "TOP(C)::forward.",
        ]
        with TemporaryFileName(mode="w+") as fname:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                with profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    with_modules=True,
                ) as prof:
                    model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                if "traceEvents" not in trace:
                    raise AssertionError("Expected 'traceEvents' in trace")
                events = trace["traceEvents"]
                checked = 0
                for evt in events:
                    if "name" not in evt:
                        raise AssertionError("Expected 'name' in event")
                    if "args" in evt:
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            hierarchy = evt["args"]["Module Hierarchy"]
                            if op_name in op_to_module_hierarchy:
                                checked += 1
                                if hierarchy not in op_to_module_hierarchy[op_name]:
                                    raise AssertionError(
                                        f"Expected hierarchy '{hierarchy}' in {op_to_module_hierarchy[op_name]}"
                                    )
                # Without this the comparisons above are vacuous: a trace with no
                # module hierarchy at all satisfies every branch.
                self.assertGreater(checked, 0)

    def test_high_level_trace(self):
        # [FUNCTION DOC] Checks that Python-side high-level events (DataLoader iteration and
        # Optimizer step/zero_grad) are recorded with the expected counts: runs a training loop
        # with no profiler first, then under profile, after pickling the optimizer, and with a
        # subclassed optimizer. Creating extra optimizer and dataloader instances checks hooks are
        # not patched repeatedly.
        """Checks that python side high level events are recorded."""

        class RepeatedDataset(torch.utils.data.Dataset):
            # [CLASS DOC] HELPER: map-style dataset of a fixed length that returns the same random
            # input and target tensors for every index.
            def __init__(self, N, D_in, D_out):
                # [FUNCTION DOC] HELPER: stores length N and random input (N x D_in) and target (N
                # x D_out).
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                # [FUNCTION DOC] HELPER: returns the dataset length.
                return self.N

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: ignores idx and returns the stored input and
                # target tensors.
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            # [CLASS DOC] HELPER: two linear layers with a clamp-at-zero activation between them.
            def __init__(self, D_in, H, D_out):
                # [FUNCTION DOC] HELPER: creates the two linear layers (D_in to H, H to D_out).
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the first layer, clamps at zero, applies the
                # second, returns it.
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            # [CLASS DOC] HELPER: trivial SGD subclass, used to check the step/zero_grad hooks
            # work for subclasses (event names carry the subclass name).
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: forwards all arguments to the SGD constructor.
                super().__init__(*args, **kwargs)

        def train():
            # [FUNCTION DOC] HELPER: one pass over the dataloader, with forward, loss, zero_grad,
            # backward and optimizer step per batch; uses the enclosing test's model, criterion,
            # optimizer, dataloader.
            for data in dataloader:
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            # [FUNCTION DOC] HELPER: counts profiler events whose names contain the marker
            # character and are in the expected mapping, then asserts each expected key was seen
            # with exactly the expected count.
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count:
                        actual_event_count[key] = (
                            actual_event_count.setdefault(key, 0) + 1
                        )
            for key, count in expected_event_count.items():
                self.assertTrue(
                    (key in actual_event_count) and (count == actual_event_count[key])
                )

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

    def test_override_time_units(self):
        # [FUNCTION DOC] Profiles a small conv/linear model and checks the key-averages table for
        # each time unit (seconds, milliseconds, microseconds): only that unit's suffix pattern
        # appears, and every event's cpu time and total cpu time, formatted to three decimals in
        # that unit, is present.
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0

        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        with _profile() as prof:
            model(inputs)

        profiler_output = prof.key_averages().table(time_unit="s")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_s = f"{event.cpu_time / US_IN_SECOND:.3f}s"
            cpu_time_total_str_s = f"{event.cpu_time_total / US_IN_SECOND:.3f}s"
            self.assertTrue(cpu_time_str_s in profiler_output)
            self.assertTrue(cpu_time_total_str_s in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="ms")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_ms = f"{event.cpu_time / US_IN_MS:.3f}ms"
            cpu_time_total_str_ms = f"{event.cpu_time_total / US_IN_MS:.3f}ms"
            self.assertTrue(cpu_time_str_ms in profiler_output)
            self.assertTrue(cpu_time_total_str_ms in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="us")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_us = f"{event.cpu_time:.3f}us"
            cpu_time_total_str_us = f"{event.cpu_time_total:.3f}us"
            self.assertTrue(cpu_time_str_us in profiler_output)
            self.assertTrue(cpu_time_total_str_us in profiler_output)

    def test_export_stacks(self):
        # [FUNCTION DOC] Profiles a small matmul with stack recording, exports stacks
        # to a temporary file, and asserts the file is non-empty and each line ends in
        # a positive integer.
        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with open(fname) as f:
                lines = f.readlines()
            if len(lines) <= 0:
                raise AssertionError("Empty stacks file")
            for line in lines:
                is_int = False
                try:
                    if int(line.split(" ")[-1]) <= 0:
                        raise AssertionError("Invalid stacks record")
                    is_int = True
                except ValueError:
                    pass
                if not is_int:
                    raise AssertionError("Invalid stacks record")

    def test_export_chrome_trace_escapes_names(self):
        # [FUNCTION DOC] Builds an event list holding one synthetic function event whose name
        # contains backslashes and quotes, exports it as a chrome trace, and checks the JSON
        # parses and the name round-trips unchanged.
        from torch.autograd.profiler_util import EventList, FunctionEvent

        tricky_name = 'aten::mm C:\\src\\"foo".cpp'
        evt = FunctionEvent(
            id=0,
            name=tricky_name,
            trace_name=tricky_name,
            thread=0,
            start_us=0,
            end_us=10,
        )
        event_list = EventList([evt])

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
        self.assertEqual(trace[0]["name"], tricky_name)

    def test_experimental_config_pickle(self):
        # [FUNCTION DOC] Checks the experimental config survives pickle (default and
        # fully populated options) and a deep copy, only asserting the resulting type;
        # BytesWarning is ignored.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BytesWarning)

            # Test with default values
            config = _ExperimentalConfig()
            pickled = pickle.dumps(config)
            unpickled = pickle.loads(pickled)
            self.assertIsInstance(unpickled, _ExperimentalConfig)

            # Test with non-default values
            config = _ExperimentalConfig(
                profiler_metrics=["metric1", "metric2"],
                profiler_measure_per_kernel=True,
                verbose=True,
                performance_events=["event1", "event2"],
                enable_cuda_sync_events=True,
                adjust_profiler_step=True,
                disable_external_correlation=True,
                profile_all_threads=True,
                capture_overload_names=True,
                record_python_gc_info=True,
                expose_kineto_event_metadata=True,
                custom_profiler_config="custom_config",
            )
            pickled = pickle.dumps(config)
            unpickled = pickle.loads(pickled)
            self.assertIsInstance(unpickled, _ExperimentalConfig)

            # Test deepcopy (which uses pickle internally)
            copied = copy.deepcopy(config)
            self.assertIsInstance(copied, _ExperimentalConfig)

    def test_profiler_range_metrics_deprecated(self):
        # [FUNCTION DOC] For configs that set the deprecated metrics or per-kernel
        # measurement options, asserts that starting an empty profile emits a
        # FutureWarning mentioning profiler_metrics.
        # profiler_metrics and profiler_measure_per_kernel are deprecated
        # no-ops: passing either must warn with FutureWarning and not error.
        for cfg in (
            _ExperimentalConfig(profiler_metrics=["m1", "m2"]),
            _ExperimentalConfig(profiler_measure_per_kernel=True),
        ):
            with self.assertWarnsRegex(FutureWarning, "profiler_metrics"):
                with profile(
                    activities=[ProfilerActivity.CPU],
                    experimental_config=cfg,
                ):
                    pass

    def test_with_modules_deprecated(self):
        # [FUNCTION DOC] Asserts that using the module-recording option emits a FutureWarning
        # saying it is deprecated.
        # with_modules only collects data for TorchScript models and is on its
        # way out: passing it must warn with FutureWarning and not error.
        with self.assertWarnsRegex(FutureWarning, "with_modules is deprecated"):
            with profile(activities=[ProfilerActivity.CPU], with_modules=True):
                torch.ones(1)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @parametrize("use_cuda", [False, True])
    def test_trace_only_export_matches_default(self, use_cuda):
        # [FUNCTION DOC] Parametrized on CUDA use (skipped without CUDA). Profiles a payload with
        # and without the trace-only experimental option, exports chrome traces, and checks cpu_op
        # names match and trace-only loses no argument keys except the call stack; with CUDA also
        # compares kernel and ac2g counts. Then checks that events() raises RuntimeError in
        # trace-only mode, and that combining trace-only with stack recording warns and falls back
        # to the normal path.
        """trace_only=True must produce the same chrome trace metadata as the default path."""
        if use_cuda and ProfilerActivity.CUDA not in supported_activities():
            self.skipTest("CUDA is required")

        device = "cuda" if use_cuda else "cpu"
        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)

        def profile_and_export(trace_only):
            # [FUNCTION DOC] HELPER: profiles one payload run with the given trace-only flag,
            # exports a chrome trace to a temporary file and returns the parsed JSON.
            with profile(
                activities=activities,
                record_shapes=True,
                experimental_config=_ExperimentalConfig(trace_only=trace_only),
            ) as prof:
                self.payload(device=device)
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    return json.load(f)

        # Warmup: ensures dynamo compilation and CUDA init don't skew comparison
        profile_and_export(trace_only=False)

        default_trace = profile_and_export(trace_only=False)
        trace_only_trace = profile_and_export(trace_only=True)

        def get_cpu_op_metadata(trace_data):
            # [FUNCTION DOC] HELPER: maps each cpu_op event name in a trace to the union of its
            # argument keys.
            result = {}
            for ev in trace_data.get("traceEvents", []):
                if ev.get("cat") == "cpu_op":
                    name = ev.get("name", "")
                    args = ev.get("args", {})
                    if name not in result:
                        result[name] = set(args.keys())
                    else:
                        result[name] |= set(args.keys())
            return result

        default_meta = get_cpu_op_metadata(default_trace)
        trace_only_meta = get_cpu_op_metadata(trace_only_trace)

        self.assertEqual(
            sorted(default_meta.keys()),
            sorted(trace_only_meta.keys()),
            "cpu_op event names should match",
        )
        for op in default_meta:
            missing = default_meta[op] - trace_only_meta.get(op, set()) - {"Call stack"}
            self.assertEqual(
                missing,
                set(),
                lambda msg: f"{msg}\n{op}: metadata keys missing in trace_only: {missing}",
            )

        if use_cuda:

            def count_by_cat(trace_data):
                # [FUNCTION DOC] HELPER: returns a counter of trace events keyed by category.
                return collections.Counter(
                    ev.get("cat", "") for ev in trace_data.get("traceEvents", [])
                )

            default_counts = count_by_cat(default_trace)
            trace_only_counts = count_by_cat(trace_only_trace)
            for cat in ("kernel", "ac2g"):
                self.assertGreater(
                    default_counts[cat],
                    0,
                    lambda msg: f"{msg}\nexpected {cat} events in default trace",
                )
                self.assertEqual(
                    default_counts[cat],
                    trace_only_counts[cat],
                    lambda msg: f"{msg}\n{cat} event count mismatch",
                )

        # events() must raise in trace_only mode
        with profile(
            activities=[ProfilerActivity.CPU],
            experimental_config=_ExperimentalConfig(trace_only=True),
        ) as prof:
            self.payload()
        with self.assertRaises(RuntimeError):
            prof.events()

        # trace_only + with_stack should warn and disable trace_only
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with profile(
                activities=[ProfilerActivity.CPU],
                with_stack=True,
                experimental_config=_ExperimentalConfig(trace_only=True),
            ) as prof:
                self.payload()
            self.assertTrue(any("trace_only" in str(x.message) for x in w))
            # Should fall back to normal path
            prof.events()

    def test_profiler_metadata(self):
        # [FUNCTION DOC] Adds string and JSON metadata to a profile, exports a chrome
        # trace and asserts the string value is stored as given and the JSON value is
        # stored as a parsed list.
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                if "test_key1" not in trace:
                    raise AssertionError("Expected 'test_key1' in trace")
                if trace["test_key1"] != "test_value1":
                    raise AssertionError(
                        f"Expected trace['test_key1'] == 'test_value1', got {trace['test_key1']}"
                    )
                if "test_key2" not in trace:
                    raise AssertionError("Expected 'test_key2' in trace")
                if trace["test_key2"] != [1, 2, 3]:
                    raise AssertionError(
                        f"Expected trace['test_key2'] == [1, 2, 3], got {trace['test_key2']}"
                    )

    def _test_profiler_tracing(self, use_kineto):
        # [FUNCTION DOC] HELPER: shared body for tracing tests. Exports a valid chrome
        # trace for a simple add, then for an empty profile (with Kineto, a warning
        # about no valid events is expected if a warnings section exists), and on
        # CUDA-capable builds repeats the add on cuda:0.
        with _profile(use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1), torch.ones(1)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # read the trace and expect valid json
            # if the JSON generated by export_chrome_trace is not valid, this will throw and fail the test.
            with open(fname) as f:
                json.load(f)

        # test empty trace
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # saving an empty trace
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            if use_kineto:
                with open(fname) as f:
                    contents = json.load(f)
                    # Some builds may not have logger observer
                    # so skip if not
                    if "WARNING" in contents:
                        found_empty_warning = False
                        for warning in contents["WARNING"]:
                            if "No Valid Trace Events" in warning:
                                found_empty_warning = True
                        self.assertTrue(found_empty_warning)

        # Same test but for cuda.
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        device = torch.device("cuda:0")
        with _profile(use_device="cuda", use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # Now validate the json
            with open(fname) as f:
                json.load(f)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_trace_sanitizes_python_function_names(self):
        # [FUNCTION DOC] Builds a Python function with a file name and function name
        # containing double quotes, calls it under a stack-recording profile, exports
        # a trace and checks a python_function event exists in which the quotes were
        # replaced by single quotes. Needs Kineto.
        def template():
            # [FUNCTION DOC] HELPER: function returning None, whose code object is
            # cloned with bad names.
            return None

        bad_code = template.__code__.replace(
            co_filename='profiler_bad"file.py',
            co_name='bad"name',
        )
        bad_fn = types.FunctionType(bad_code, {})

        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            bad_fn()

        del bad_fn, bad_code
        gc.collect()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        python_function_names = [
            event.get("name", "")
            for event in trace["traceEvents"]
            if event.get("cat") == "python_function"
        ]
        self.assertGreater(len(python_function_names), 0)
        self.assertTrue(
            any(
                "profiler_bad'file.py" in name and "bad'name" in name
                for name in python_function_names
            )
        )

    def test_profiler_tracing(self):
        # [FUNCTION DOC] Runs the shared tracing test without Kineto, and with Kineto
        # if available.
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    def test_profiler_op_event_args(self):
        # [FUNCTION DOC] Enables recording of concrete inputs (a global setting, left enabled),
        # profiles an ones and cat call with shapes recorded, and checks the ones event's input
        # types and concrete inputs, the cat event's dims and types, and that every cpu_op event
        # has a nonnegative record function id.
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        with _profile(record_shapes=True) as prof:
            a = torch.ones((64, 32), dtype=torch.float32)
            c = torch.cat([a, a]).sin()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::ones":
                        self.assertEqual(
                            args["Input type"],
                            ["ScalarList", "Scalar", "", "", "Scalar"],
                        )
                        self.assertEqual(
                            args["Concrete Inputs"], ["[64, 32]", "6", "", "", "False"]
                        )

                    if e["name"] == "aten::cat":
                        self.assertEqual(args["Input Dims"], [[[64, 32], [64, 32]], []])
                        self.assertEqual(args["Input type"], ["TensorList", "Scalar"])

                    # check that each op has record function id
                    self.assertGreaterEqual(
                        args.get("Record function id", -1),
                        0,
                        lambda msg: f"{msg}\nFailed finding record funciont for op = {e}",
                    )

    def test_profiler_strides(self):
        # [FUNCTION DOC] With concrete input recording enabled, profiles an add of two strided
        # views and checks the add event's recorded input strides.
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        with _profile(record_shapes=True) as prof:
            c = torch.add(a, b)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::add":
                        self.assertEqual(args["Input Strides"], [[17, 1], [25, 2], []])

    def test_profiler_strides_without_concrete_inputs(self):
        # [FUNCTION DOC] Same as the stride test but with concrete input recording
        # disabled (restored to enabled in a finally block): strides must still be
        # exported for the add event.
        torch._C._profiler._set_record_concrete_inputs_enabled_val(False)
        try:
            base_tensor = torch.randn(1024, dtype=torch.float32)
            a = base_tensor.as_strided((16, 16), (17, 1), 0)
            b = base_tensor.as_strided((16, 16), (25, 2), 272)
            with _profile(record_shapes=True) as prof:
                c = torch.add(a, b)

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    op_events = [
                        e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                    ]
                    for e in op_events:
                        args = e["args"]
                        if e["name"] == "aten::add":
                            self.assertIn("Input Strides", args)
                            self.assertEqual(
                                args["Input Strides"], [[17, 1], [25, 2], []]
                            )
        finally:
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)

    def test_profiler_fwd_bwd_link(self):
        # [FUNCTION DOC] Profiles an add and a loss backward, exports a trace, and checks there
        # are exactly two forward-backward flow start and finish events, with ids 1 and 2, whose
        # start timestamps map to the loss op and the add op.
        with _profile(use_kineto=True) as prof:
            t1, t2 = (
                torch.ones(1, requires_grad=True),
                torch.ones(1, requires_grad=True),
            )
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                events = j["traceEvents"]
                ts_to_name = {}
                flow_s_to_ts = {}
                flow_f_to_ts = {}
                for e in events:
                    if e["ph"] == "X":
                        ts_to_name[e["ts"]] = e["name"]
                    if (
                        "cat" in e
                        and "name" in e
                        and e["cat"] == "fwdbwd"
                        and e["name"] == "fwdbwd"
                    ):
                        if e["ph"] == "s":
                            flow_s_to_ts[e["id"]] = e["ts"]
                        elif e["ph"] == "f":
                            flow_f_to_ts[e["id"]] = e["ts"]

                self.assertEqual(len(flow_s_to_ts), 2)
                self.assertEqual(len(flow_f_to_ts), 2)
                self.assertIn(1, flow_s_to_ts)
                self.assertIn(1, flow_f_to_ts)
                self.assertIn(2, flow_s_to_ts)
                self.assertIn(2, flow_f_to_ts)
                s_ts_1 = flow_s_to_ts[1]
                f_ts_1 = flow_f_to_ts[1]
                s_ts_2 = flow_s_to_ts[2]
                f_ts_2 = flow_f_to_ts[2]
                self.assertTrue(
                    all(ts in ts_to_name for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2])
                )
                self.assertTrue(
                    ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits"
                )
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")

    def test_profiler_disable_fwd_bwd_link(self):
        # [FUNCTION DOC] Disables forward-backward linking (restored in a finally block), profiles
        # an add and a loss backward, and asserts no trace event has the fwdbwd category.
        try:
            torch._C._profiler._set_fwd_bwd_enabled_val(False)

            with _profile(use_kineto=True) as prof:
                t1, t2 = (
                    torch.ones(1, requires_grad=True),
                    torch.ones(1, requires_grad=True),
                )
                z = torch.add(t1, t2)
                y = torch.ones(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                loss.backward()

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    events = j["traceEvents"]

                    for e in events:
                        self.assertNotEqual(e.get("cat", None), "fwdbwd")
        finally:
            torch._C._profiler._set_fwd_bwd_enabled_val(True)

    def test_profiler_type(self):
        # [FUNCTION DOC] Checks the active profiler type is none outside a profiler, legacy inside
        # the legacy profiler, and Kineto inside the standard profiler.
        profiler_type = torch._C._autograd._profiler_type
        ActiveProfilerType = torch._C._profiler.ActiveProfilerType
        self.assertEqual(profiler_type(), ActiveProfilerType.NONE)

        # Autograd profiler
        with _profile_legacy():
            self.assertEqual(profiler_type(), ActiveProfilerType.LEGACY)

        # Kineto profiler
        with profile():
            self.assertEqual(profiler_type(), ActiveProfilerType.KINETO)

    def test_profiler_correlation_id(self):
        # [FUNCTION DOC] Profiles a model five times and asserts cpu_op Kineto events have
        # correlation ids that are unique across all runs and fit in 32 bits.
        """
        We expect the correlation_id of CPU operator events to be unique across
        multiple invocations of the profiler, so we will reuse id_uniqueness_set.
        Only cpu_op events are checked: with CUDA available, profile() also
        collects CUDA activities, and CUPTI's device-enumeration calls at startup
        (cudaGetDeviceCount, etc.) are recorded as cuda_runtime events whose
        correlation ids come from a separate counter (also starting at 1) and
        would otherwise collide with the operator ids.
        """
        id_uniqueness_set = set()
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        uint32_max = 2**32 - 1
        for _ in range(5):
            with profile() as prof:
                model(inputs)
            for event in prof.profiler.kineto_results.events():
                corr_id = event.correlation_id()
                is_cpu = event.device_type() == DeviceType.CPU
                if corr_id and is_cpu and event.activity_type() == "cpu_op":
                    self.assertTrue(corr_id not in id_uniqueness_set)
                    id_uniqueness_set.add(corr_id)
                    self.assertTrue(corr_id < uint32_max)

    def test_nested_tensor_with_shapes(self):
        # [FUNCTION DOC] Profiles a linear op on a nested tensor with shapes recorded and checks
        # that mm or addmm events report non-empty input shapes.
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        inp = torch.nested.nested_tensor([a, b])
        with torch.profiler.profile(record_shapes=True) as prof:
            torch.nn.functional.linear(inp, c, None)
        for e in prof.events():
            if e.name in ("aten::mm", "aten::addmm"):
                # intentionally vague tests to protect against possible future changes
                # of mm to addmm or other impl, or changing internal order of args
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.input_shapes[0]) > 0)

    def test_custom_autograd_function_input_shapes(self):
        # [FUNCTION DOC] Applies a custom autograd Function with a tensor, a Python scalar and a
        # tensor slice under shape recording, and checks the one resulting event reports the
        # shapes of the tensors with an empty entry for the scalar.
        class CustomRecordShapes(torch.autograd.Function):
            # [CLASS DOC] HELPER: custom autograd Function computing x plus y times scale.
            @staticmethod
            def forward(ctx, x, scale, y):
                # [FUNCTION DOC] HELPER: returns x plus y scaled by the scale argument.
                return x + y * scale

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: returns the gradient for x and y and no
                # gradient for the scale.
                return grad, None, grad

        x = torch.ones(2, 3, requires_grad=True)
        y = torch.ones(4, requires_grad=True)
        with profile(record_shapes=True) as prof:
            CustomRecordShapes.apply(x, 2.0, y[:3])

        events = [
            event for event in prof.events() if event.name == "CustomRecordShapes"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].input_shapes, [[2, 3], [], [3]])

    def test_concrete_inputs_profiling(self):
        # [FUNCTION DOC] Profiles an as_strided call with shapes recorded and checks
        # that the event has input shapes and concrete inputs matching the original
        # tensor shape, size and stride.
        x = torch.rand(2, 6)
        with profile(record_shapes=True) as p:
            y = x.as_strided([4, 3], [1, 4])

        found = False
        for e in p.events():
            if e.name in ("aten::as_strided"):
                found = True
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.concrete_inputs) > 0)
                self.assertEqual([2, 6], e.input_shapes[0])
                self.assertEqual([4, 3], e.concrete_inputs[1])
                self.assertEqual([1, 4], e.concrete_inputs[2])

        self.assertTrue(found, "Expected to find aten::as_strided but did not")

    def test_concrete_inputs_profiling_toggling(self):
        # [FUNCTION DOC] Toggles the concrete-input recording flag in the middle of a profile (on
        # to off, then off to on) and checks the as_strided event is still recorded with input
        # shapes. Sets the flag back to enabled in a finally block.
        try:
            for before, after in [(True, False), (False, True)]:
                x = torch.rand(2, 6)
                torch._C._profiler._set_record_concrete_inputs_enabled_val(before)
                with profile(record_shapes=True) as p:
                    y = x.as_strided([4, 3], [1, 4])
                    torch._C._profiler._set_record_concrete_inputs_enabled_val(after)

                found = False
                for e in p.events():
                    if e.name in ("aten::as_strided"):
                        found = True
                        self.assertTrue(len(e.input_shapes))

                self.assertTrue(found, "Expected to find aten::as_strided but did not")
        finally:
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)

    def test_record_function_fast(self):
        # [FUNCTION DOC] Exercises the fast record-function context manager in several variants:
        # no extra arguments (empty shapes and keyword inputs), tensor inputs plus keyword inputs,
        # non-tensor inputs (shape reported as a single empty entry) with an exception raised
        # inside the block (code after the raise must not be recorded), nesting, shapes not
        # recorded when disabled, and a tuple of inputs. Each variant checks at least four events
        # with the expected name were recorded.
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            for _ in range(4):
                # Test first with no optional args
                with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf1"]), 4
        )
        for e in p.events():
            if e.name == "add_test_fast_rf1":
                self.assertTrue(e.input_shapes == [])
                self.assertTrue(e.kwinputs == {})
        with profile(record_shapes=True) as p:
            # add optional args
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf2", [x, y], {"stream": 0, "grid": "lambda x : x + 1"}
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf2"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf2":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])
                self.assertTrue(e.kwinputs == {"stream": 0, "grid": "lambda x : x + 1"})

        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf3", input_values=["hi"], keyword_values={"hi": "hello"}
            )
            for _ in range(4):
                try:
                    with cm:
                        x.add(y)
                        raise ValueError
                        x.relu()
                except ValueError:
                    pass

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf3"]), 4
        )
        self.assertFalse(any((e.name and "relu" in e.name) for e in p.events()))

        for e in p.events():
            if e.name == "add_test_fast_rf3":
                self.assertTrue(e.input_shapes == [[]])

        with profile() as p:
            for _ in range(4):
                with torch._C._profiler._RecordFunctionFast(
                    "add_test_fast_rf4", [x, y]
                ):
                    x.add(y)
                    with torch._C._profiler._RecordFunctionFast("add_test_fast_rf5"):
                        x.relu()

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf4"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf4":
                self.assertTrue(e.input_shapes == [])

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf5"]), 4
        )

        with profile(record_shapes=True) as p:
            # test optional args with tuple
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf6",
                (
                    x,
                    y,
                ),
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf6"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf6":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs(self):
        # [FUNCTION DOC] Records fast-record-function events carrying keyword inputs, exports a
        # trace and checks the args hold stream, grid and boolean (a string with a quote shows up
        # as None). A second profile with a user scope entry in the keyword inputs checks the
        # event category becomes user_annotation and the stream and grid args are dropped. Skipped
        # under dynamo.
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {
                    "stream": 0,
                    "grid": "lambda x : x + 1",
                    "debug": 'debug"',
                    "boolean": True,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" in args)
                    self.assertTrue("grid" in args)
                    self.assertTrue("boolean" in args)
                    self.assertTrue(args["stream"] == 0)
                    self.assertTrue(args["grid"] == "lambda x : x + 1")
                    self.assertTrue(args["debug"] == "None")
                    self.assertTrue(args["boolean"])
                    self.assertTrue(e["cat"] == "cpu_op")

        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {"stream": "test", "grid": [1, 2], "scope": "user_scope"},
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" not in args)
                    self.assertTrue("grid" not in args)
                    self.assertTrue(e["cat"] == "user_annotation")

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs_list_of_strings(self):
        # [FUNCTION DOC] Checks keyword inputs that are lists of strings are exported
        # as lists, together with int and string values, and that lists with mixed or
        # non-string element types are left out of the exported args while valid
        # entries remain. Skipped under dynamo.
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list",
                [x, y],
                {
                    "string_list": ["hello", "world", "test"],
                    "int_param": 42,
                    "string_param": "single_string",
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("string_list" in args)
                    self.assertTrue("int_param" in args)
                    self.assertTrue("string_param" in args)
                    # Check that the list of strings is properly serialized
                    # The list should be formatted as a JSON array by ivalueListToStr
                    self.assertEqual(args["string_list"], ["hello", "world", "test"])
                    self.assertEqual(args["int_param"], 42)
                    self.assertEqual(args["string_param"], "single_string")
                    self.assertTrue(e["cat"] == "cpu_op")

        # Test mixed types that should be filtered out
        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list_filtered",
                [x, y],
                {
                    "valid_string_list": ["valid1", "valid2"],
                    "mixed_list": ["string", 123],  # Should be filtered out
                    "non_string_list": [1, 2, 3],  # Should be filtered out
                    "valid_int": 100,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list_filtered"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    # Only valid types should be present
                    self.assertTrue("valid_string_list" in args)
                    self.assertTrue("valid_int" in args)
                    # Invalid lists should be filtered out
                    self.assertTrue("mixed_list" not in args)
                    self.assertTrue("non_string_list" not in args)
                    # Check values
                    self.assertEqual(args["valid_string_list"], ["valid1", "valid2"])
                    self.assertEqual(args["valid_int"], 100)
                    self.assertTrue(e["cat"] == "cpu_op")

    def test_is_profiler_enabled(self):
        # [FUNCTION DOC] Checks the autograd profiler enabled flag is false outside a
        # profiler and true inside both the torch.profiler and the autograd profiler
        # contexts, then false again.
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with torch.autograd.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

    def test_guarded_record_function_fast(self):
        # [FUNCTION DOC] Uses a fast record-function guarded by the profiler-enabled
        # flag inside a loop of four adds under a profile and checks at least four
        # events with that name were recorded.
        x, y = (torch.rand((4, 4)) for _ in range(2))

        with profile() as p:
            cm = torch._C._profiler._RecordFunctionFast("guarded_rff")
            for _ in range(4):
                if torch.autograd.profiler._is_profiler_enabled:
                    with cm:
                        x.add(y)
                else:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "guarded_rff"]), 4
        )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skip("This test is known to pollute other tests on a fresh build")
    def test_profiler_time_scale(self):
        # [FUNCTION DOC] Skipped test (known to pollute other tests). Profiles a five second loop
        # inside a record_function span and checks the event duration, both in function events and
        # in the exported trace, is within half a second of five seconds.
        MARGIN_ERROR = 0.5
        SEC_TO_US = 1000 * 1000
        # IMPORTANT: For reasons that are not yet understood, having a long idle
        # profiling window will make later CUPTI activity records (in other tests)
        # arrive with invalid timestamps, causing them to be dropped as out-of-range.
        # Empirically, WAIT_TIME should be kept <= 6.
        WAIT_TIME = 5
        with profile() as p:
            with torch.profiler.record_function("test_span"):
                for _ in range(WAIT_TIME):
                    torch.rand(4, 4)
                    time.sleep(1)
        events = p.events()

        # make sure function events are scaled appropriately
        self.assertTrue(events[0].name == "test_span")
        test_span = events[0]
        self.assertGreaterEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME - MARGIN_ERROR,
            "event out of range",
        )
        self.assertLessEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME + MARGIN_ERROR,
            "event out of range",
        )

        # make sure tracing is scaled appropriately
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
            events = report["traceEvents"]
            for event in events:
                if event["name"] == "test_span":
                    self.assertGreaterEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME - MARGIN_ERROR,
                        "profiling out of range",
                    )
                    self.assertLessEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME + MARGIN_ERROR,
                        "profiling out of range",
                    )

    def _schedule_helper(self, warmup, active, repeat, acc_events=True):
        # [FUNCTION DOC] HELPER: profiles 100 add calls stepping the profiler after each, using a
        # schedule with the given warmup, active and repeat counts (no wait or skip), and the
        # given accumulate events flag. Returns the recorded count of add events from key
        # averages, or zero if absent.
        with profile(
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=0,
                warmup=warmup,
                active=active,
                repeat=repeat,
            ),
            acc_events=acc_events,
        ) as prof:
            for _ in range(100):
                torch.add(1, 2)
                prof.step()
        # print(prof.key_averages())
        for ev in prof.key_averages():
            if ev.key == "aten::add":
                return ev.count
        return 0

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_schedule_function_count(self):
        # [FUNCTION DOC] Checks the number of add events recorded by the schedule helper for
        # several combinations of warmup, active, repeat and event accumulation (for example 83
        # events for warmup 1, active 5, unlimited repeat). Skipped under dynamo.
        self.assertEqual(self._schedule_helper(warmup=0, active=1, repeat=1), 1)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=0), 100)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=10), 50)
        self.assertEqual(self._schedule_helper(warmup=1, active=5, repeat=0), 83)
        self.assertEqual(self._schedule_helper(warmup=10, active=10, repeat=4), 40)
        self.assertEqual(self._schedule_helper(warmup=50, active=1, repeat=0), 1)
        self.assertEqual(
            self._schedule_helper(warmup=0, active=5, repeat=0, acc_events=False), 0
        )
        self.assertEqual(
            self._schedule_helper(warmup=10, active=10, repeat=4, acc_events=False), 10
        )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_lazy_build_tree(self):
        # [FUNCTION DOC] Checks the function-event tree is built lazily: right after profiling the
        # stats show zero tree build time and zero events, and after asking for events both are
        # positive. Skipped under dynamo.
        with profile() as p:
            self.payload()

        stats = p._stats()
        # Test that the tree is not built
        self.assertEqual(stats.function_events_build_tree_call_duration_us, 0)
        self.assertEqual(stats.number_of_events, 0)

        # Test that the tree is built on demand
        p.events()
        self.assertGreater(stats.function_events_build_tree_call_duration_us, 0)
        self.assertGreater(stats.number_of_events, 0)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_skip_first_wait(self):
        # [FUNCTION DOC] Builds a schedule with skip_first_wait set and compares its action for
        # steps zero through nineteen against a hand-written list (none, warmup, record,
        # record-and-save sequences across two repeats). Skipped under dynamo.
        # Other tests test when skip_first_wait is false (default) so just test the true case
        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=5, warmup=1, active=2, repeat=2, skip_first_wait=1
        )
        test_schedule_expected_outputs = [
            # repeat No. 1 begin
            # skip first 3
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 1 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 1 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 5
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_python_gc_event(self):
        # [FUNCTION DOC] With and without the Python GC recording option, profiles a payload that
        # forces a garbage collection between two named spans, exports a trace, and checks Python
        # GC events exist between the two spans when enabled and are absent when disabled. Needs
        # Kineto; skipped under dynamo.
        activities = [ProfilerActivity.CPU]

        def payload():
            # [FUNCTION DOC] HELPER: two matmuls separated by a forced garbage collection, each
            # inside a named record_function span (pre_gc and post_gc).
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            with record_function("pre_gc"):
                torch.mm(x, y)
            gc.collect()
            with record_function("post_gc"):
                torch.mm(x, y)

        def validate_json(prof, gc_collection_on):
            # [FUNCTION DOC] HELPER: exports the profile to a temporary trace file. When GC
            # recording is on, requires the pre_gc, post_gc and at least one Python GC event
            # strictly between them; when off, requires no Python GC events.
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    # Find required events
                    if gc_collection_on:
                        pre_gc = next(
                            (e for e in events if e["name"] == "pre_gc"), None
                        )
                        post_gc = next(
                            (e for e in events if e["name"] == "post_gc"), None
                        )
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        # Assert all required events are present
                        self.assertIsNotNone(pre_gc, "pre_gc event is missing")
                        self.assertIsNotNone(post_gc, "post_gc event is missing")
                        self.assertTrue(
                            len(python_gc_events) > 0, "No Python GC events found"
                        )
                        # Calculate boundaries
                        pre_gc_end = pre_gc["ts"] + pre_gc.get("dur", 0)
                        post_gc_start = post_gc["ts"]
                        # Assert at least one Python GC event is correctly placed.
                        # Other automatic GC events can happen while the profiler is
                        # active, especially during first-run initialization.
                        python_gc_events_between = [
                            e
                            for e in python_gc_events
                            if e["ts"] > pre_gc_end
                            and e["ts"] + e.get("dur", 0) < post_gc_start
                        ]
                        self.assertTrue(
                            len(python_gc_events_between) > 0,
                            "No Python GC events found between pre_gc and post_gc",
                        )
                    else:
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        self.assertTrue(
                            len(python_gc_events) == 0,
                            "Python GC event found when flag off",
                        )

        for gc_flag in [True, False]:
            with profile(
                activities=activities,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    record_python_gc_info=gc_flag
                ),
                with_stack=True,
            ) as prof:
                payload()
            validate_json(prof, gc_flag)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_none(self):
        # [FUNCTION DOC] Checks that a profile with a default (no) timeout parsing step yields
        # function events including an aten::mm event. Needs Kineto.
        """Test that _parse_kineto_results works normally without timeout."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        # Access function_events to trigger parsing
        events = p.function_events
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_large(self):
        # [FUNCTION DOC] Calls the Kineto result parser directly with a 60 second timeout and with
        # no timeout, and asserts both return the same number of events.
        """Test that _parse_kineto_results with a large timeout processes all events."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        # Call _parse_kineto_results directly with a large timeout
        events_with_timeout = p._parse_kineto_results(p.kineto_results, timeout_s=60.0)
        events_without_timeout = p._parse_kineto_results(
            p.kineto_results, timeout_s=None
        )

        # Both should return the same number of events
        self.assertEqual(len(events_with_timeout), len(events_without_timeout))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_zero(self):
        # [FUNCTION DOC] Parses Kineto results with a zero timeout and expects a warning log from
        # the autograd profiler logger about timing out with partial results, and strictly fewer
        # events than an untimed parse of the same data.
        """Test that _parse_kineto_results with zero timeout returns partial results and logs."""
        with _profile(use_kineto=True) as p:
            # Generate some events
            for _ in range(10):
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

        # Get baseline count without timeout
        events_no_timeout = p._parse_kineto_results(p.kineto_results, timeout_s=None)
        baseline_count = len(events_no_timeout)

        # With a zero timeout, we should get fewer events (or possibly zero)
        # and a warning should be logged
        import logging

        with self.assertLogs("torch.autograd.profiler", level=logging.WARNING) as cm:
            events_with_timeout = p._parse_kineto_results(
                p.kineto_results, timeout_s=0.0
            )

        # Check that we got a warning about timeout
        self.assertTrue(
            any("timed out" in msg and "partial results" in msg for msg in cm.output)
        )

        # With zero timeout, we should have fewer events than baseline
        self.assertLess(len(events_with_timeout), baseline_count)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_fails(self):
        # [FUNCTION DOC] Asserts that the Kineto result parser raises ValueError saying timeout_s
        # must be non-negative when given a negative timeout.
        """Test that _parse_kineto_results fails with a negative timeout."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        with self.assertRaisesRegex(ValueError, "timeout_s must be non-negative"):
            events = p._parse_kineto_results(p.kineto_results, timeout_s=-60.0)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_none(self):
        # [FUNCTION DOC] Checks the public profile with no post-processing timeout returns events
        # that include aten::mm.
        """Test that torch.profiler.profile works normally without timeout."""
        with profile() as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        events = p.events()
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_large(self):
        # [FUNCTION DOC] Checks the public profile with a 60 second post-processing timeout still
        # returns events including aten::mm.
        """Test that torch.profiler.profile with a large timeout processes all events."""
        with profile(post_processing_timeout_s=60.0) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        events = p.events()
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_zero(self):
        # [FUNCTION DOC] Profiles with a zero post-processing timeout and expects the
        # events call to log a warning from the autograd profiler logger about timing
        # out with partial results.
        """Test that torch.profiler.profile with zero timeout returns partial results."""
        import logging

        with profile(post_processing_timeout_s=0.0) as p:
            for _ in range(10):
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

        with self.assertLogs("torch.autograd.profiler", level=logging.WARNING) as cm:
            events = p.events()

        self.assertTrue(
            any("timed out" in msg and "partial results" in msg for msg in cm.output)
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_fails(self):
        # [FUNCTION DOC] Asserts that constructing a profile with a negative post-processing
        # timeout raises ValueError saying it must be non-negative.
        """Test that torch.profiler.profile with a negative timeout fails correctly."""
        with self.assertRaisesRegex(
            ValueError, "post_processing_timeout_s must be non-negative"
        ):
            with profile(post_processing_timeout_s=-1.0) as p:
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_duplicate_raises(self):
        # [FUNCTION DOC] Asserts a ValueError when the same activity appears both as a plain entry
        # and as a key in a filter mapping.
        """Same activity appearing more than once raises ValueError."""
        with self.assertRaises(ValueError):
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    {ProfilerActivity.CPU: ["CPU_OP"]},
                ],
            ) as p:
                pass

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_invalid_type_name(self):
        # [FUNCTION DOC] Asserts a RuntimeError when an activity filter names a nonexistent
        # activity type.
        """Invalid activity type name raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            with profile(
                activities=[{ProfilerActivity.CPU: ["NONEXISTENT_TYPE"]}],
            ) as p:
                x = torch.randn(10, 10)
                y = torch.mm(x, x)


class SimpleNet(nn.Module):
    # [CLASS DOC] HELPER: tiny two-layer linear network (10 to 5 to 2) used as a model fixture.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: creates the two linear layers.
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: applies the first then the second linear layer.
        return self.fc2(self.fc1(x))


class MockNode:
    # [CLASS DOC] HELPER: fake tree node with a name and a list of child nodes, built recursively
    # from a mapping of child names to child mappings; used to test tree-walking utilities.
    def __init__(self, name, children) -> None:
        # [FUNCTION DOC] HELPER: stores the name and recursively creates a child node for each
        # item of the children mapping (child names come from the mapping keys).
        self.name = name
        self.children = [MockNode(name, i) for name, i in children.items()]


class TestProfilerDevice(TestCase):
    # [CLASS DOC] Profiler tests meant to be instantiated per device type (CPU, CUDA, XPU and
    # others): Kineto basics, memory profiling, chrome trace validation and related helpers.
    # Methods take a device argument supplied by the device-type test generator.
    """Tests that should run on multiple backends (CPU, CUDA, XPU, etc.)."""

    def payload(self, device="cpu", tensor_size=10):
        # [FUNCTION DOC] HELPER: small workload: two random matrices moved to the
        # device, matmul, add.
        x = torch.randn(tensor_size, tensor_size).to(device)
        y = torch.randn(tensor_size, tensor_size).to(device)
        z = torch.mm(x, y)
        z = z + y

    def _check_stats(self, profiler_stats):
        # [FUNCTION DOC] HELPER: asserts that every timing and count field of a profiler stats
        # object (window duration, event count, prepare/enable/disable/parse/build-tree durations)
        # is positive.
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    def _step_helper_func(self, prof):
        # [FUNCTION DOC] HELPER: sleeps 0.1 seconds, creates a random image-sized tensor, and
        # advances the profiler by one step.
        time.sleep(0.1)
        torch.randn(1, 3, 224, 224)
        prof.step()

    def _partial_overlap(self, prof_step, step_helper_func):
        # [FUNCTION DOC] HELPER: given two trace events with start and duration, returns true when
        # one starts inside the other and ends after it (partial overlap in either direction),
        # else false.
        p_start = prof_step["ts"]
        p_end = prof_step["ts"] + prof_step["dur"]
        h_start = step_helper_func["ts"]
        h_end = step_helper_func["ts"] + step_helper_func["dur"]

        if p_start < h_start and p_end < h_end and p_end > h_start:
            return True
        if p_start > h_start and p_start < h_end and p_end > h_end:
            return True
        return False

    def _check_all_gpu_present(self, gpu_dict, max_gpu_count):
        # [FUNCTION DOC] HELPER: asserts the counter of GPU labels has the value one for every
        # label GPU 0 through the one before max_gpu_count.
        for i in range(max_gpu_count):
            self.assertEqual(gpu_dict["GPU " + str(i)], 1)

    def _is_secondary_profiler_event(self, traceEvent):
        # [FUNCTION DOC] HELPER: returns true if the event name begins with the XPU
        # profiler instance name or its iteration start marker, which lie outside the
        # PyTorch profiler window.
        # On an XPU build the trace contains, in addition to the
        # "PyTorch Profiler (0)" instance, the XPU profiler instance
        # "__xpu_profiler__ (N)" plus its own "Iteration Start: __xpu_profiler__"
        # marker. These are emitted regardless of which activities are requested
        # and are not bounded by the PyTorch profiler window, so they must be
        # ignored when validating the trace.
        name = traceEvent.get("name", "")
        return name.startswith(
            ("__xpu_profiler__", "Iteration Start: __xpu_profiler__")
        )

    def _validate_basic_json(self, traceEvents, device_available=False):
        # [FUNCTION DOC] HELPER: validates a list of trace events. Requires the main profiler
        # event, the iteration start event and the record window end event, that the profiler span
        # lies inside them, and that every other event (ignoring XPU secondary profiler events)
        # starts at or after the profiler start and ends before the window end. For events
        # labelled with a GPU, counts the labels and checks the next event's sort index equals a
        # fixed large offset plus the GPU number. The device availability argument is unused.
        MAX_GPU_COUNT = 8

        def _find_event(name):
            # [FUNCTION DOC] HELPER: returns the first trace event with the given name, or None.
            for event in traceEvents:
                if event.get("name") == name:
                    return event
            return None

        traceEventProfiler = _find_event("PyTorch Profiler (0)")
        recordStart = _find_event("Iteration Start: PyTorch Profiler")
        recordEnd = _find_event("Record Window End")

        self.assertIsNotNone(
            traceEventProfiler, "missing 'PyTorch Profiler (0)' trace event"
        )
        self.assertIsNotNone(
            recordStart, "missing 'Iteration Start: PyTorch Profiler' trace event"
        )
        self.assertIsNotNone(recordEnd, "missing 'Record Window End' trace event")

        self.assertGreaterEqual(
            traceEventProfiler["ts"],
            recordStart["ts"],
            "Profiler starts before record!",
        )
        self.assertLessEqual(
            traceEventProfiler["ts"] + traceEventProfiler["dur"],
            recordEnd["ts"],
            "Profiler ends after record end!",
        )

        gpu_dict = collections.defaultdict(int)
        for i, traceEvent in enumerate(traceEvents):
            if traceEvent is recordStart or traceEvent is recordEnd:
                continue
            # Skip the secondary device profiler instance, which lives outside the
            # PyTorch profiler window and would otherwise trip the bounds checks.
            if self._is_secondary_profiler_event(traceEvent):
                continue
            if "ts" in traceEvent:
                self.assertGreaterEqual(
                    traceEvent["ts"],
                    traceEventProfiler["ts"],
                    "Trace event is out of bounds",
                )
            if "dur" in traceEvent:
                self.assertLessEqual(
                    traceEvent["ts"] + traceEvent["dur"],
                    recordEnd["ts"],
                    "Trace event ends too late!",
                )
            gpu_value = traceEvent.get("args", {}).get("labels", None)
            if gpu_value and "GPU" in gpu_value:
                gpu_dict[gpu_value] += 1
                kExceedMaxPid = 5000000
                self.assertTrue(
                    traceEvents[i + 1]["args"]["sort_index"]
                    == kExceedMaxPid + int(gpu_value.split()[1])
                )

    def _test_chrome_trace_basic_helper(self, device):
        # [FUNCTION DOC] HELPER: profiles an add on the device with stack recording, exports a
        # chrome trace, and runs the basic JSON validation on its events.
        x, y = (torch.rand(4, 4).to(device) for _ in range(2))
        device_type = device.split(":")[0]
        device_available = device_type != "cpu"

        with profile(with_stack=True) as p:
            torch.add(x, y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
                self._validate_basic_json(report["traceEvents"], device_available)

    @onlyOn(["cpu", "cuda"])
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self, device):
        # [FUNCTION DOC] Device test (CPU and CUDA, needs Kineto). Profiles a matmul payload with
        # the autograd profiler (Kineto enabled), checks the printed profile mentions aten::mm,
        # builds a key averages table, and checks gemm and memcpy device events exist on CUDA or
        # the mm event on CPU. Also checks the stats via the helper.
        device_type = device.split(":")[0]
        with _profile(use_device=device_type, use_kineto=True):
            self.payload(device=device)

        with _profile(use_device=device_type, use_kineto=True) as p:
            self.payload(device=device)

        self.assertTrue("aten::mm" in str(p))

        output = p.key_averages().table(
            sort_by="self_device_time_total"
            if device_type != "cpu"
            else "self_cpu_time_total",
            row_limit=-1,
        )
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name.lower() or "Cijk" in e.name:
                found_gemm = True
            if "memcpy" in e.name.lower() or "__amd_rocclr_copyBuffer" in e.name:
                found_memcpy = True
        if device_type in ("cuda",):
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        self._check_stats(p._stats)

    def test_memory_profiler(self, device):
        # [FUNCTION DOC] Device test of memory profiling. Profiles tensor creation and
        # deletion inside named user scopes and checks per-event CPU memory usage
        # (positive for allocating ops and the alloc scope, negative for the dealloc
        # scope), repeats for device tensors (device memory) and mkldnn tensors when
        # available, and with Kineto exports a trace checking [memory] events carry
        # address, device type, device id and bytes and are instantaneous. Finally
        # checks a garbage collected tensor shows up as a negative [memory] entry.
        device_type = device.split(":")[0]

        def run_profiler(tensor_creation_fn):
            # [FUNCTION DOC] HELPER: profiles the given tensor-creating callable inside an alloc
            # scope, then deletes the tensor inside a dealloc scope, with memory and shape
            # profiling; returns key averages grouped by input shape.
            with _profile(
                profile_memory=True,
                record_shapes=True,
                use_kineto=kineto_available(),
            ) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            # [FUNCTION DOC] HELPER: maps event key to the named metric attribute; for each name
            # in allocs requires presence and a positive value, for each name in deallocs requires
            # presence and a negative value.
            stat_metrics = {}
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertGreater(
                        stat_metrics[alloc_fn],
                        0,
                        lambda msg: f"{msg}\nalloc_fn = {alloc_fn}",
                    )
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertLess(
                        stat_metrics[dealloc_fn],
                        0,
                        lambda msg: f"{msg}\nalloc_fn = {dealloc_fn}",
                    )

        def create_cpu_tensor():
            # [FUNCTION DOC] HELPER: returns a random 10 by 10 CPU tensor.
            return torch.rand(10, 10)

        def create_device_tensor():
            # [FUNCTION DOC] HELPER: returns a random 10 by 10 tensor moved to the test device.
            return torch.rand(10, 10).to(device)

        def create_mkldnn_tensor():
            # [FUNCTION DOC] HELPER: returns a random 10 by 10 float32 tensor
            # converted to mkldnn layout.
            return torch.rand(10, 10, dtype=torch.float32).to_mkldnn()

        stats = run_profiler(create_cpu_tensor)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "test_user_scope_alloc",
            ],
            deallocs=[
                "test_user_scope_dealloc",
            ],
        )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                with profile(profile_memory=True) as prof:
                    x = None
                    with record_function("test_user_scope_alloc"):
                        x = create_cpu_tensor()
                    with record_function("test_user_scope_dealloc"):
                        del x
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    trace = json.load(f)
                    if "traceEvents" not in trace:
                        raise AssertionError("Expected 'traceEvents' in trace")
                    events = trace["traceEvents"]
                    found_memory_events = False
                    for evt in events:
                        if "name" not in evt:
                            raise AssertionError("Expected 'name' in event")
                        if evt["name"] == "[memory]":
                            found_memory_events = True
                            if "args" not in evt:
                                raise AssertionError("Expected 'args' in memory event")
                            if "Addr" not in evt["args"]:
                                raise AssertionError("Expected 'Addr' in event args")
                            if "Device Type" not in evt["args"]:
                                raise AssertionError(
                                    "Expected 'Device Type' in event args"
                                )
                            if "Device Id" not in evt["args"]:
                                raise AssertionError(
                                    "Expected 'Device Id' in event args"
                                )
                            if "Bytes" not in evt["args"]:
                                raise AssertionError("Expected 'Bytes' in event args")

                            if "dur" in evt["args"]:
                                raise AssertionError("Unexpected 'dur' in event args")
                            if "cat" in evt["args"]:
                                raise AssertionError("Unexpected 'cat' in event args")
                    if not found_memory_events:
                        raise AssertionError("Expected to find memory events")

        if device_type != "cpu":
            create_device_tensor()
            stats = run_profiler(create_device_tensor)
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.backends.mkldnn.is_available():
            create_mkldnn_tensor()
            stats = run_profiler(create_mkldnn_tensor)
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::rand",
                    "aten::empty",
                    "aten::to_mkldnn",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )

        with _profile(profile_memory=True, use_kineto=kineto_available()) as prof:
            x = torch.rand(10, 10)
            del x
            if device_type != "cpu":
                y = torch.rand(10, 10).to(device)
                del y
            gc.collect()
        stats = prof.key_averages(group_by_input_shape=True)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=["aten::rand", "aten::empty"],
            deallocs=["[memory]"],
        )
        if device_type != "cpu":
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])

    def test_flops(self, device):
        # [FUNCTION DOC] Profiles a conv/linear model plus a jagged nested tensor add with FLOPs
        # counting and checks the key averages table has a total FLOPs column. On non-CPU devices
        # with Kineto it repeats with the device activities and checks the same column.
        device_type = device.split(":")[0]
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        nested_tensor = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 5))], layout=torch.jagged
        )
        with _profile(
            record_shapes=True, with_flops=True, use_kineto=kineto_available()
        ) as prof:
            model(inputs)
            nested_tensor = nested_tensor + nested_tensor
        profiler_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")
        if not kineto_available() or device_type == "cpu":
            return

        with profile(
            activities=get_profiler_activities(device_type),
            record_shapes=True,
            with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_device_time_total", row_limit=-1
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_api(self, device):
        # [FUNCTION DOC] Runs with the Kineto daemon environment variables patched for the test.
        # Checks a scheduled profile (wait 1, warmup 1, active 2) over 8 steps calls the trace
        # handler twice and advances the global Kineto step tracker by 8; checks a plain profile
        # yields a table; and checks a schedule with skip, wait, warmup, active and repeat returns
        # the expected actions for 21 steps.
        device_type = device.split(":")[0]
        called_num = [0]

        use_device = device_type != "cpu"
        with profile(activities=supported_activities()):
            self.payload(device=device)

        def trace_handler(p):
            # [FUNCTION DOC] HELPER: trace-ready callback that builds a key averages table and
            # increments the shared call counter.
            output = p.key_averages().table(
                sort_by="self_device_time_total"
                if use_device
                else "self_cpu_time_total",
                row_limit=-1,
            )
            called_num[0] += 1

        initial_step = KinetoStepTracker.current_step()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(8):
                self.payload(device=device)
                p.step()

        self.assertEqual(called_num[0], 2)
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        with profile(activities=supported_activities()) as p:
            self.payload(device=device)
            self.payload(device=device)
        output = p.key_averages().table(
            sort_by="self_device_time_total" if use_device else "self_cpu_time_total",
            row_limit=-1,
        )

        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=2, warmup=1, active=4, repeat=2
        )
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_multiple_steppers(self, device):
        # [FUNCTION DOC] Runs with the Kineto daemon environment variables patched. Simulates
        # several step sources: an optimizer step plus a hand-incremented named step counter, run
        # for 8 iterations without profiling, then 8 more under a scheduled profile that steps
        # itself. Checks the global step tracker advanced by two counts per iteration over both
        # phases (16 in total) and always erases the extra named counter afterwards.
        device_type = device.split(":")[0]
        niters = 8
        net = SimpleNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        opt.zero_grad()
        inputs = torch.rand(10)

        with profile(activities=supported_activities()):
            self.payload(device=device)

        def optimizer_step():
            # [FUNCTION DOC] HELPER: bumps the global step tracker under the name yet_another_step
            # to mimic a step hook in an optimizer.
            """This simulates a step() hook in the optimizer"""
            KinetoStepTracker.increment_step("yet_another_step")

        initial_step = KinetoStepTracker.current_step()

        def run_batch():
            # [FUNCTION DOC] HELPER: one training iteration of the test net with cross entropy
            # loss, backward, optimizer step and the extra step increment.
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
            optimizer_step()

        try:
            for _ in range(niters):
                run_batch()

            with profile(
                activities=supported_activities(),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            ) as p:
                for _ in range(niters):
                    run_batch()
                    p.step()

            self.assertEqual(
                KinetoStepTracker.current_step(), initial_step + 2 * niters
            )
        finally:
            # KinetoStepTracker is global across device-specialized test runs.
            KinetoStepTracker.erase_step_count("yet_another_step")

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_kineto_profiler_with_environment_variable(self, device):
        # [FUNCTION DOC] Runs a script in a subprocess (with the daemon environment variables
        # patched in) that profiles a small training loop under a schedule and verifies the step
        # tracker count. A failing exit status makes the test fail with a message about the
        # Dynolog environment variable. Skipped under dynamo.
        device_type = device.split(":")[0]
        script = """
import torch
import torch.nn as nn
from torch.profiler import supported_activities, profile, ProfilerActivity
from torch.autograd.profiler import KinetoStepTracker

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def payload(device="cpu"):
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = torch.mm(x, y)
    z = z + y

niters = 8
device = "cpu"
for act in supported_activities():
    if act != ProfilerActivity.CPU:
        device = act.name.lower()
        break
net = SimpleNet()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
opt.zero_grad()
inputs = torch.rand(10)

with profile(activities=supported_activities()):
    payload(device=device)

initial_step = KinetoStepTracker.current_step()

def run_batch():
    out = net(inputs)
    loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
    loss.backward()
    opt.step()

for _ in range(niters):
    run_batch()

with profile(
    activities=supported_activities(),
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
) as p:
    for _ in range(niters):
        run_batch()
        p.step()
if KinetoStepTracker.current_step() != initial_step + 2 * niters:
    raise AssertionError(f"Expected step {initial_step + 2 * niters}, got {KinetoStepTracker.current_step()}")
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )

    @onlyAccelerator
    def test_event_list(self, device):
        # [FUNCTION DOC] Accelerator-only test: profiles forward and backward of a matmul/relu/sum
        # with stack recording, wraps the events in an event list, checks that the chrome trace
        # export is valid JSON and that the table method runs.
        device_type = device.split(":")[0]
        x, y = (torch.rand((4, 4), requires_grad=True, device=device) for _ in range(2))
        with profile(with_stack=True) as p:
            z = (x @ y).relu().sum()
            z.backward()

        event_list = torch.autograd.profiler_util.EventList(p.events())

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                json.load(f)

        event_list.table()

    @skipIf(
        True,
        "XPU Trace event ends too late! Refer https://github.com/intel/torch-xpu-ops/issues/2263",
        device_type="xpu",
    )
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_chrome_trace(self, device):
        # [FUNCTION DOC] Runs the shared chrome trace validation helper for the device. Skipped on
        # XPU (known issue with trace events ending too late), without Kineto, and under dynamo.
        device_type = device.split(":")[0]
        self._test_chrome_trace_basic_helper(device)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_cpu_annotation_overlap(self, device):
        # [FUNCTION DOC] Profiles five steps of a helper under a scheduled profile with adjusted
        # profiler step enabled, exports a trace and checks there are five profiler step events
        # and five helper events, with no partial overlap between any profiler step and any helper
        # event. Skipped under dynamo.
        device_type = device.split(":")[0]
        with torch.profiler.profile(
            activities=get_profiler_activities(device_type),
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            experimental_config=torch._C._profiler._ExperimentalConfig(
                adjust_profiler_step=True
            ),
        ) as prof:
            for _ in range(5):
                self._step_helper_func(prof)
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            prof_steps = []
            step_helper_funcs = []
            with open(fname) as f:
                report = json.load(f)
                for event in report["traceEvents"]:
                    if "ProfilerStep" in event["name"]:
                        prof_steps.append(event)
                    if "step_helper_func" in event["name"]:
                        step_helper_funcs.append(event)
            self.assertEqual(len(prof_steps), 5)
            self.assertEqual(len(step_helper_funcs), 5)
            for i in range(len(step_helper_funcs)):
                for j in range(len(step_helper_funcs)):
                    self.assertTrue(
                        not self._partial_overlap(prof_steps[i], step_helper_funcs[j])
                    )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_user_annotation(self, device):
        # [FUNCTION DOC] Profiles a payload inside a user record_function scope and checks only
        # the key averages entry with that scope name is flagged as a user annotation. Skipped
        # under dynamo.
        device_type = device.split(":")[0]
        with profile(activities=supported_activities()) as p:
            with torch.profiler.record_function("test_user_annotation"):
                self.payload(device=device)

        for evt in p.key_averages():
            if evt.key == "test_user_annotation":
                self.assertTrue(evt.is_user_annotation)
            else:
                self.assertFalse(evt.is_user_annotation)

    @onlyAccelerator
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_profile(self, device):
        # [FUNCTION DOC] Accelerator-only: profiles an in-place multiply of a device tensor and
        # checks that the only aten event recorded is the in-place multiply. Skipped under dynamo.
        device_type = device.split(":")[0]
        x = torch.randn(4, device=device)
        with torch.profiler.profile(with_stack=True) as p:
            x *= 2
        names = [e.name for e in p.events()]
        for name in names:
            if name.startswith("aten") and name != "aten::mul_":
                self.assertTrue(False, "Found unexpected event: " + name)
        self.assertTrue("aten::mul_" in names)

    @onlyAccelerator
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_dynamic_toggle(self, device):
        # [FUNCTION DOC] Accelerator-only: checks that device activity collection can be toggled
        # during a profile. Without toggling there are aten, device-named and kernel events;
        # toggling off the device activity leaves aten events but no device or kernel events;
        # toggling off all activities leaves no events. Skipped under dynamo.
        device_type = device.split(":")[0]
        gpu_activity = getattr(ProfilerActivity, device_type.upper(), None)
        activities = get_profiler_activities(device_type)
        with profile(activities=activities) as p:
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)

        self.assertTrue(any("aten" in e.name for e in p.events()))

        self.assertTrue(any(device_type in e.name.lower() for e in p.events()))

        self.assertTrue(any("kernel" in e.name.lower() for e in p.events()))

        with profile(activities=activities) as p1:
            p1.toggle_collection_dynamic(False, [gpu_activity])
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)

        self.assertTrue(any("aten" in e.name for e in p1.events()))

        self.assertTrue(all(device_type not in e.name for e in p1.events()))

        self.assertTrue(all("kernel" not in e.name.lower() for e in p1.events()))

        with profile(activities=activities) as p2:
            p2.toggle_collection_dynamic(False, activities)
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)
        self.assertTrue(len(p2.events()) == 0)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @onlyOn("cpu")
    @unittest.skipIf(IS_WINDOWS, "can't use os.fork() on Windows")
    def test_forked_process(self, device):
        # [FUNCTION DOC] CPU-only (not Windows). Profiles in the parent, then forks; the child
        # profiles again, exports a trace and requires that its CPU op events carry the child
        # process id and a thread id different from the parent's, then exits without returning to
        # the test runner; the parent waits for the child. Skipped under dynamo.
        device_type = device.split(":")[0]

        def validate_forked_json(profiler):
            # [FUNCTION DOC] HELPER: exports the profiler trace and, for every cpu_op event,
            # checks the process id equals the child pid and the thread id differs from the parent
            # tid; records that a cpu_op was found in the enclosing scope variable.
            nonlocal cpu_op_found, parent_tid, child_pid
            with TemporaryFileName(mode="w+") as fname:
                profiler.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    for event in events:
                        if "cat" in event and event["cat"] == "cpu_op":
                            self.assertEqual(event["pid"], child_pid)
                            self.assertNotEqual(event["tid"], parent_tid)
                            cpu_op_found = True

        cpu_op_found = False
        parent_tid = threading.current_thread().ident
        with profile(activities=[ProfilerActivity.CPU]) as p:
            self.payload()
        pid = os.fork()
        if pid == 0:
            child_pid = os.getpid()
            with profile(activities=[ProfilerActivity.CPU]) as p:
                self.payload()
            validate_forked_json(p)
            self.assertTrue(cpu_op_found)
            os._exit(0)
        else:
            os.waitpid(pid, 0)

    @onlyAccelerator
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_profile_all_threads(self, device):
        # [FUNCTION DOC] Accelerator-only. Profiles matmuls with and without profiling of all
        # threads, once with the profile entered in one thread while another thread runs work, and
        # once with the side thread started inside the profile. Checks matmul events carry the
        # right shapes, there is one event thread (or two when all threads are profiled) and each
        # has five events.
        profiling_started = threading.Event()
        profiling_ended = threading.Event()
        n_rep = 5

        def prep_inputs():
            # [FUNCTION DOC] HELPER: returns two random 1024 by 1024 tensors on the test device.
            return [torch.randn(1024, 1024, device=device) for _ in range(2)]

        def main_thread_fn(profile_all_threads, returned_events):
            # [FUNCTION DOC] HELPER: in the profiled thread, runs matmuls inside a profile that
            # signals the start event, waits for the end event, then appends the events to the
            # given list.
            x, y = prep_inputs()
            experimental_config = torch._C._profiler._ExperimentalConfig(
                profile_all_threads=profile_all_threads
            )
            with torch.profiler.profile(
                experimental_config=experimental_config, record_shapes=True
            ) as p:
                profiling_started.set()
                for _ in range(n_rep):
                    _ = x @ y
                profiling_ended.wait()
            returned_events.append(p.events())

        def side_thread_fn():
            # [FUNCTION DOC] HELPER: waits for the start event, runs the matmuls on a separate
            # thread, then sets the end event.
            x, y = prep_inputs()
            profiling_started.wait()
            for _ in range(n_rep):
                _ = x @ y
            profiling_ended.set()

        def main_with_thread_fn(profile_all_threads):
            # [FUNCTION DOC] HELPER: profiles matmuls on the current thread while a side thread
            # runs the same work, joins it, and returns the events.
            x, y = prep_inputs()
            experimental_config = torch._C._profiler._ExperimentalConfig(
                profile_all_threads=profile_all_threads
            )
            with torch.profiler.profile(
                experimental_config=experimental_config, record_shapes=True
            ) as p:
                side_thread = threading.Thread(target=side_thread_fn)
                side_thread.start()
                for _ in range(n_rep):
                    _ = x @ y
                side_thread.join()
            return p.events()

        for profile_all_threads in (True, False):
            returned_events = []
            main_thread = threading.Thread(
                target=main_thread_fn, args=(profile_all_threads, returned_events)
            )
            side_thread = threading.Thread(target=side_thread_fn)
            main_thread.start()
            side_thread.start()
            main_thread.join()
            side_thread.join()

            def verify_events(events):
                # [FUNCTION DOC] HELPER: counts mm events per thread, checking their input shapes,
                # requires one thread (plus one when profiling all threads) and the expected
                # repetition count on each.
                mm_events = collections.defaultdict(int)
                for e in events:
                    if e.name == "aten::mm":
                        mm_events[e.thread] += 1
                        self.assertEqual(e.input_shapes, [[1024, 1024], [1024, 1024]])
                self.assertEqual(len(mm_events), 1 + int(profile_all_threads))
                for v in mm_events.values():
                    self.assertEqual(v, n_rep)

            verify_events(returned_events[0])
            events = main_with_thread_fn(profile_all_threads)
            verify_events(events)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler(self, device):
        # [FUNCTION DOC] Basic profile API test: profiles a payload with the device activities and
        # requires events, an mm event, and (for non-CPU devices, including the private backend)
        # events of that device type. Needs Kineto.
        """Basic test for torch.profiler.profile API."""
        device_type = device.split(":")[0]
        use_device = device_type != "cpu"
        activities = get_profiler_activities(device_type)
        with profile(activities=activities) as p:
            self.payload(device=device)
        events = p.events()
        self.assertGreater(len(events), 0)
        found_mm = False
        for e in events:
            if "aten::mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)
        if use_device:
            from torch._C import _get_privateuse1_backend_name

            device_type_enum = getattr(DeviceType, device_type.upper(), None)
            if (
                device_type_enum is None
                and device_type == _get_privateuse1_backend_name()
            ):
                device_type_enum = DeviceType.PrivateUse1
            gpu_events = [e for e in events if e.device_type == device_type_enum]
            self.assertGreater(len(gpu_events), 0, "No GPU events captured by profiler")

    @onlyOn(["cuda"])
    @unittest.skipIf(TEST_WITH_ROCM, "not supported on ROCm")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_dict_syntax(self, device):
        # [FUNCTION DOC] CUDA-only (not ROCm). Profiles with a mapping filter keeping only memcpy
        # and runtime activities and checks memcpy and CUDA runtime events appear and no
        # lazy-loading overhead events are present. Prints the events.
        """Dict syntax collects only the requested activity types."""
        device_type = device.split(":")[0]
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        self.assertIsNotNone(device_activity)
        with profile(
            activities=[{device_activity: ["GPU_MEMCPY", "CUDA_RUNTIME"]}],
        ) as p:
            x = torch.randn(10, 10).to(device)
            y = torch.mm(x, x)
        events = p.events()
        self.assertGreater(len(events), 0)
        print(events)
        has_memcpy = any("Memcpy" in e.name for e in events)
        self.assertTrue(has_memcpy, "Expected GPU_MEMCPY events")
        has_runtime = any("cuda" in e.name for e in events)
        self.assertTrue(has_runtime, "Expected CUDA_RUNTIME events")
        has_overhead = any("Lazy Function Loading" in e.name for e in events)
        self.assertFalse(has_overhead)

    @onlyAccelerator
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_nonmember_type_name(self, device):
        # [FUNCTION DOC] Accelerator-only: asserts RuntimeError when the filter lists a CPU
        # activity name under the device activity group.
        """Activity type name that is not a member of the requested activity group raises RuntimeError."""
        device_type = device.split(":")[0]
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        self.assertIsNotNone(device_activity)
        with self.assertRaises(RuntimeError):
            with profile(
                activities=[{device_activity: ["CPU_OP"]}],
            ) as p:
                x = torch.randn(10, 10)
                y = torch.mm(x, x)

    @onlyAccelerator
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_empty_list(self, device):
        # [FUNCTION DOC] Accelerator-only: asserts an empty name list for the device activity
        # collects no events at all.
        """Passing an empty list to activities means not collecting for the specified activity."""
        device_type = device.split(":")[0]
        device_activity = getattr(ProfilerActivity, device_type.upper(), None)
        self.assertIsNotNone(device_activity)
        with profile(
            activities=[{device_activity: []}],
        ) as p:
            x = torch.randn(10, 10).to(device)
            y = torch.mm(x, x)
        self.assertEqual(len(p.events()), 0)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180072")
    @onlyAccelerator
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto_kernel_metadata_in_trace(self, device):
        # [FUNCTION DOC] Accelerator-only (not ROCm). Exports a trace and checks kernel events
        # exist, all carry device, stream and correlation args, grid and block appear together,
        # and at least one kernel has launch grid and block metadata.
        device_type = device.split(":")[0]
        with profile(activities=get_profiler_activities(device_type)) as prof:
            self.payload(device=device)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
            events = trace["traceEvents"]
            kernel_events = [e for e in events if e.get("cat", "") == "kernel"]
            self.assertGreater(
                len(kernel_events), 0, "Error: No kernel events in trace"
            )
            has_kernel_launch_metadata = False
            for ke in kernel_events:
                args = ke.get("args", {})
                name = ke.get("name", "<unknown>")
                for key in ["device", "stream", "correlation"]:
                    self.assertIn(
                        key, args, lambda msg: f"{msg}\nkernel '{name}' missing '{key}'"
                    )
                has_grid = "grid" in args
                has_block = "block" in args
                self.assertEqual(
                    has_grid,
                    has_block,
                    lambda msg: f"{msg}\nkernel '{name}' should provide grid and block together",
                )
                has_kernel_launch_metadata |= has_grid
            self.assertTrue(
                has_kernel_launch_metadata,
                "Error: No kernel events in trace contained grid/block metadata",
            )

    @onlyAccelerator
    @skipIfRocm(msg="ROCm does not emit OVERHEAD activity records")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_overhead_activities_own_no_device_time(self, device):
        # [FUNCTION DOC] Accelerator-only (not ROCm). Checks profiler overhead events exist and
        # report zero self and total device time, and that no event's self device time exceeds the
        # sum of all non-CPU device time.
        """
        OVERHEAD events (profiler-internal host cost) must report zero device time
        """
        device_type = device.split(":")[0]
        # Skip warm-up: the first traced launches are the ones that emit overhead records.
        with profile(activities=get_profiler_activities(device_type)) as prof:
            self.payload(device=device)

        events = prof.events()
        total_device_time = sum(
            e.self_device_time_total for e in events if e.device_type != DeviceType.CPU
        )

        overhead_events = [e for e in events if e.activity_type == "overhead"]
        self.assertGreater(
            len(overhead_events), 0, "Expected at least one overhead event"
        )
        for e in overhead_events:
            self.assertEqual(e.self_device_time_total, 0)
            self.assertEqual(e.device_time_total, 0)

        # No single row may own more device time than the whole trace spent on device.
        for e in events:
            self.assertLessEqual(
                e.self_device_time_total,
                total_device_time,
            )


instantiate_device_type_tests(TestProfilerDevice, globals())


class TestExperimentalUtils(TestCase):
    # [CLASS DOC] Tests for experimental profiler utilities: tree traversal, symbolization
    # fuzzing, and the other helpers in the experimental utils module.
    def make_tree(self) -> list[MockNode]:
        # [FUNCTION DOC] HELPER: builds two root mock nodes (root_0 and root_1) from a fixed
        # nested mapping and returns them as a list.
        tree = {
            "root_0": {
                "1": {"2": {}},
                "3": {
                    "4": {},
                    "5": {},
                },
            },
            "root_1": {
                "6": {},
                "7": {},
                "8": {
                    "9": {"10": {}},
                },
            },
        }
        return [MockNode(name, i) for name, i in tree.items()]

    def test_dfs(self) -> None:
        # [FUNCTION DOC] Checks depth-first traversal visits the fixed tree in pre-order.
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_dfs(self.make_tree())),
            "root_0 1 2 3 4 5 root_1 6 7 8 9 10",
        )

    def test_bfs(self) -> None:
        # [FUNCTION DOC] Checks breadth-first traversal visits the fixed tree level by level.
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_bfs(self.make_tree())),
            "root_0 root_1 1 3 6 7 8 2 4 5 9 10",
        )

    @unittest.skipIf(
        not IS_LINUX or not (IS_X86 or IS_ARM64), "linux x86/aarch64 only cpp unwinding"
    )
    def test_fuzz_symbolize(self):
        # [FUNCTION DOC] Linux x86/aarch64 only. Finds the executable code sections of loaded
        # shared libraries, picks 200 random addresses with a fixed seed, and runs the three
        # address symbolizers (fast, dladdr, addr2line) on them. Only checks that each returns one
        # result per address (the fast and addr2line lengths are compared; dladdr is merely run
        # without crashing).
        # generate some random addresses in the text section and make sure the
        # symbolizers do not throw exceptions/crash
        def get_text_sections():
            # [FUNCTION DOC] HELPER: walks the memory-mapped files of this process, and for each
            # distinct shared library that is an ELF file parses the section header table by hand
            # to find the text section. Returns a list of (absolute start address, size) pairs,
            # one per library. Closes each mapping.
            text_sections = []
            seen = set()
            for filename in os.listdir("/proc/self/map_files"):
                library = os.readlink("/proc/self/map_files/" + filename)
                if ".so" not in library or library in seen:
                    continue
                seen.add(library)
                with open(os.path.join("/proc/self/map_files", library), "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

                    def unpack(fmt, offset):
                        # [FUNCTION DOC] HELPER: reads a binary value of the given
                        # struct format from the mapped file at the given byte offset
                        # and returns the unpacked tuple.
                        return struct.unpack(
                            fmt, mm[offset : offset + struct.calcsize(fmt)]
                        )

                    if mm[:4] != b"\x7fELF":
                        continue
                    (section_headers_start,) = unpack("Q", 40)
                    (section_header_size,) = unpack("H", 58)
                    (num_section_headers,) = unpack("H", 60)
                    (shstrndx,) = unpack("H", 62)
                    (shstrtab_offset,) = unpack(
                        "Q", section_headers_start + shstrndx * section_header_size + 24
                    )
                    for i in range(num_section_headers):
                        (section_name_offset,) = unpack(
                            "I", section_headers_start + i * section_header_size
                        )
                        name_start = shstrtab_offset + section_name_offset
                        section_name = mm[name_start : name_start + 6]
                        if section_name != b".text\0":
                            continue
                        (section_offset,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 24
                        )
                        (section_size,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 32
                        )
                        start = int(filename.split("-")[0], 16) + section_offset
                        text_sections.append((start, section_size))
                        break
                    mm.close()
            return text_sections

        r = random.Random()
        r.seed(1)
        text_sections = get_text_sections()
        addrs = []
        for _ in range(200):
            s = r.randrange(0, len(text_sections))
            start, size = text_sections[s]
            addr = r.randrange(start, start + size)
            addrs.append(addr)
        fast = torch._C._profiler.symbolize_addresses(addrs, "fast")
        dladdr = torch._C._profiler.symbolize_addresses(addrs, "dladdr")
        addr2line = torch._C._profiler.symbolize_addresses(addrs, "addr2line")
        self.assertEqual(len(fast), len(addrs))
        self.assertEqual(len(addr2line), len(fast))

    def test_profiler_overload_names(self):
        # [FUNCTION DOC] With a CPU fallthrough kernel registered for the add overload in a scoped
        # library, profiles an add with overload name capture on. Checks that exactly one event
        # has two children and overload name Tensor, that one child has overload name out, that
        # key averages have two entries (three when grouping by overload name) and the table shows
        # an Overload Name column only when grouping, and that the exported trace has events for
        # both add overloads.
        from torch.library import _scoped_library, fallthrough_kernel

        def validate_json(prof):
            # [FUNCTION DOC] HELPER: prints a blank line, exports a trace and asserts events for
            # the Tensor and out overloads of add exist by name.
            print()
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    self.assertTrue(
                        any("aten::add.Tensor" in e["name"] for e in events)
                    )
                    self.assertTrue(any("aten::add.out" in e["name"] for e in events))

        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("add.Tensor", fallthrough_kernel, "CPU")
            experimental_config = torch._C._profiler._ExperimentalConfig(
                capture_overload_names=True
            )
            with profile(
                experimental_config=experimental_config,
                activities=[ProfilerActivity.CPU],
            ) as prof:
                torch.add(1, 5)

            # The following execution trace is expected
            #
            # Dispatch trace:
            # [call] op=[aten::add.Tensor], key=[AutogradCPU]
            #   [redispatch] op=[aten::add.Tensor], key=[Undefined]
            #     [call] op=[aten::empty.memory_format], key=[BackendSelect]
            #       [redispatch] op=[aten::empty.memory_format], key=[CPU]
            #     [call] op=[aten::add.out], key=[CPU]
            #
            # prof.table()
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------
            #            Name    Overload Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------
            #       aten::add           Tensor        71.97%     130.887us       100.00%     181.873us     181.873us             1
            #     aten::empty    memory_format         8.52%      15.489us         8.52%      15.489us      15.489us             1
            #       aten::add              out        19.52%      35.497us        19.52%      35.497us      35.497us             1
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------

            # aten::add.out and aten::empty.memory_format are children of aten::add.Tensor
            aten_add_parent: list[FunctionEvent] = [
                event for event in prof.events() if len(event.cpu_children) == 2
            ]
            if len(aten_add_parent) != 1:
                raise AssertionError(
                    f"Expected 1 parent event, got {len(aten_add_parent)}"
                )
            aten_add_parent = aten_add_parent[0]
            if aten_add_parent.overload_name != "Tensor":
                raise AssertionError(
                    f"Expected overload_name 'Tensor', got '{aten_add_parent.overload_name}'"
                )

            aten_add_out_event = [
                c for c in aten_add_parent.cpu_children if c.overload_name == "out"
            ]
            if len(aten_add_out_event) != 1:
                raise AssertionError(
                    f"Expected 1 out event, got {len(aten_add_out_event)}"
                )

            # Without group_by_overload_name, the overload name is ignored in the key averages
            key_averages = prof.key_averages()
            if len(key_averages) != 2:
                raise AssertionError(
                    f"Expected 2 key averages, got {len(key_averages)}"
                )
            if "Overload Name" in key_averages.table():
                raise AssertionError("Overload Name should not be in table")

            key_averages = prof.key_averages(group_by_overload_name=True)
            if len(key_averages) != 3:
                raise AssertionError(
                    f"Expected 3 key averages with group_by_overload_name, got {len(key_averages)}"
                )
            if "Overload Name" not in key_averages.table():
                raise AssertionError("Overload Name should be in table")
            validate_json(prof)

    def test_expose_kineto_event_metadata(self):
        # [FUNCTION DOC] Profiles an add with Kineto event metadata exposure on and checks, via a
        # nested helper, that the metadata key Ev Idx appears for aten::add both in the exported
        # trace and in the function events.
        def check_metadata(prof, op_name, metadata_key):
            # [FUNCTION DOC] HELPER: exports a trace, requires the named op to be present with the
            # metadata key in its args, and requires the matching function events to hold the key
            # in their metadata text; raises AssertionError otherwise.
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    found_op = False
                    for e in events:
                        if "name" in e and "args" in e and e["name"] == op_name:
                            if metadata_key not in e["args"]:
                                raise AssertionError(
                                    f"Metadata for '{op_name}' in Chrome trace did not contain '{metadata_key}'."
                                )
                            found_op = True
                    if not found_op:
                        raise AssertionError(
                            f"Could not find op '{op_name}' in Chrome trace."
                        )
                found_op = False
                for event in prof.events():
                    if event.name == op_name:
                        if metadata_key not in event.metadata_json:
                            raise AssertionError(
                                f"Metadata for '{op_name}' in FunctionEvent did not contain '{metadata_key}'."
                            )
                        found_op = True
                if not found_op:
                    raise AssertionError(
                        f"Could not find op '{op_name}' in prof.events()."
                    )

        experimental_config = torch._C._profiler._ExperimentalConfig(
            expose_kineto_event_metadata=True
        )
        with profile(
            experimental_config=experimental_config,
            activities=[ProfilerActivity.CPU],
        ) as prof:
            torch.add(1, 5)

        check_metadata(prof, op_name="aten::add", metadata_key="Ev Idx")

    def test_function_event_metadata(self):
        # [FUNCTION DOC] Builds a function event with string extra metadata and typed metadata,
        # and checks the metadata property returns the typed mapping and the grid is exposed as an
        # attribute of the event metadata object.
        from torch.autograd.profiler_util import FunctionEvent

        extra_meta = {"grid": "[1, 2, 3]"}
        typed_metadata = {"grid": [1, 2, 3], "new metadata": {"nested": True}}
        event = FunctionEvent(
            1,
            "test",
            1,
            0,
            1,
            extra_meta=extra_meta,
            typed_metadata=typed_metadata,
        )

        self.assertEqual(event.metadata, typed_metadata)
        self.assertEqual(event.event_metadata.grid, [1, 2, 3])

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/158727",
    )
    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_profiler_debug_autotuner(self):
        # [FUNCTION DOC] Compiles a matmul with max-autotune and Triton, with and without
        # bandwidth profiling, runs both under a profile, and asserts the sets of event names
        # containing mm or triton are equal. Skipped when the large GPU requirement is not met.
        # KNOWN ISSUE: the skip decorator skips on Linux (and ROCm or slow mode), so it
        # effectively never runs on the usual Linux CUDA hosts; it references an open issue.
        """
        This test makes sure that profiling events will be present when the kernel is run using the DebugAutotuner.
        """
        if not is_big_gpu():
            raise unittest.SkipTest("requires large gpu to max-autotune")
        in1 = torch.randn((256, 512), device="cuda", dtype=torch.float16)
        in2 = torch.randn((512, 768), device="cuda", dtype=torch.float16)

        def mm():
            # [FUNCTION DOC] HELPER: matmul of the two captured CUDA half-precision tensors.
            return torch.mm(in1, in2)

        pb_mm = torch.compile(
            mm,
            options={
                "benchmark_kernel": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "profile_bandwidth": True,
            },
        )
        comp_mm = torch.compile(
            mm,
            options={
                "benchmark_kernel": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

        with profile() as prof1:
            pb_mm()
        with profile() as prof2:
            comp_mm()

        def names(prof):
            # [FUNCTION DOC] HELPER: returns the set of event names from a profile containing mm
            # or triton.
            return {
                ev.name
                for ev in prof.events()
                if "mm" in ev.name or "triton" in ev.name
            }

        n1 = names(prof1)
        n2 = names(prof2)
        self.assertEqual(n1, n2)


class TestPrivateUse1ProfilerState(TestCase):
    # [CLASS DOC] Tests of which profiler state is selected for a private-use backend, with the
    # backend name lookup patched to return a made-up backend.
    """Tests for PrivateUse1 profiler state selection logic."""

    def test_kineto_privateuse1_state_with_use_kineto_true(self):
        # [FUNCTION DOC] Patches the backend name lookup and checks that building the autograd
        # profiler with Kineto enabled selects the Kineto private-use state.
        """Test that KINETO_PRIVATEUSE1 state is selected when use_kineto=True."""
        from unittest.mock import patch

        from torch._C._profiler import ProfilerState

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            prof = _profile(
                use_cpu=True,
                use_device="custom_backend",
                use_kineto=True,
            )
            self.assertEqual(prof.profiler_kind, ProfilerState.KINETO_PRIVATEUSE1)

    def test_kineto_privateuse1_fallback_state_with_use_kineto_false(self):
        # [FUNCTION DOC] Same setup with Kineto disabled; checks the Kineto private-use fallback
        # state is selected.
        """Test that KINETO_PRIVATEUSE1_FALLBACK is selected when use_kineto=False."""
        from unittest.mock import patch

        from torch._C._profiler import ProfilerState

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            prof = _profile(
                use_cpu=True,
                use_device="custom_backend",
                use_kineto=False,
            )
            self.assertEqual(
                prof.profiler_kind, ProfilerState.KINETO_PRIVATEUSE1_FALLBACK
            )

    def test_privateuse1_fallback_requires_use_cpu(self):
        # [FUNCTION DOC] Checks that building the profiler without Kineto and without CPU
        # profiling for a private-use device raises AssertionError.
        """Test that KINETO_PRIVATEUSE1_FALLBACK requires use_cpu=True."""
        from unittest.mock import patch

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            # When use_kineto=False and use_cpu=False, should raise AssertionError
            with self.assertRaises(AssertionError):
                _profile(
                    use_cpu=False,
                    use_device="custom_backend",
                    use_kineto=False,
                )


@instantiate_parametrized_tests
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerDeviceStopped(TestCase):
    # [CLASS DOC] CUDA-only tests of the device-stopped profiler action. The Kineto stopped signal
    # is patched (target stored in PATCH_TARGET), so each test drives the profiler through start,
    # step and stop with scripted signal values and checks current_action and callback counts. The
    # class is parametrized.
    """Tests for the DEVICE_STOPPED transition: when Kineto signals that
    device collection has stopped (e.g. CUPTI buffer overflow), the profiler
    saves any collected trace, sits in DEVICE_STOPPED for the rest of the
    cycle, and resumes normal scheduling on the next cycle. The Kineto signal
    is mocked so no actual overflow is needed.

    Note that we don't actually depend on any CUDA specific behavior. But the
    DEVICE_STOPPED logic does behave different if the user has only requested
    CPU-only profiling. Explicitly specifying a GPU device seems cleaner than
    patching the `use_device` attributes in the profiler."""

    PATCH_TARGET = "torch.autograd._is_kineto_stopped"

    def _make_profiler(self, **schedule_kwargs):
        # [FUNCTION DOC] HELPER: builds a CPU+CUDA profiler whose schedule is built from the given
        # keyword arguments.
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(**schedule_kwargs),
        )

    def test_enters_device_stopped_from_warmup(self):
        # [FUNCTION DOC] With the stopped signal true, a warmup phase step is overridden into the
        # device-stopped action.
        p = self._make_profiler(wait=0, warmup=2, active=2)
        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            # step 0 -> 1: WARMUP -> WARMUP per schedule, but _is_kineto_stopped
            # returns True so we override to DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()

    def test_enters_device_stopped_from_record(self):
        # [FUNCTION DOC] Records normally for a step, then with the signal true a step in the
        # record phase becomes device-stopped.
        p = self._make_profiler(wait=0, warmup=1, active=3)
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            # step 0 -> 1: WARMUP -> RECORD (normal)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 1 -> 2: RECORD -> RECORD per schedule, but _is_kineto_stopped
            # -> True overrides to DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()

    def test_does_not_enter_device_stopped_from_record_and_save(self):
        # [FUNCTION DOC] Checks that a stopped signal on the step after record-and-save does not
        # produce device-stopped: the natural transition to warmup fires the trace-ready callback
        # once, and after the signal clears the next cycle records and fires the callback a second
        # time on stop.
        """When prev_action is RECORD_AND_SAVE, the natural action_map
        transition (e.g. R&S -> WARMUP) already fires stop_trace +
        _trace_ready + prepare_trace, which resets Kineto. We do NOT
        override to DEVICE_STOPPED — doing so would skip the reset and
        waste the next cycle in DEVICE_STOPPED. If Kineto recovers,
        profiling continues normally; if not, the entry guard fires on
        the next step (prev=WARMUP or RECORD)."""
        callback_count = [0]

        def handler(prof):
            # [FUNCTION DOC] HELPER: trace-ready callback that increments the shared counter.
            callback_count[0] += 1

        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=2),
            on_trace_ready=handler,
        )

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 2 -> 3: prev=R&S, schedule=WARMUP. Despite kineto_stopped,
            # we do NOT override; the natural (R&S, WARMUP) transition fires
            # stop_trace + _trace_ready + prepare_trace.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(
                callback_count[0],
                1,
                "(R&S, WARMUP) natural transition must fire on_trace_ready",
            )

        # Simulate Kineto recovering after prepare_trace: cycle 2 records
        # normally rather than being wasted in DEVICE_STOPPED.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 3 -> 4: WARMUP -> RECORD
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()  # step 4 -> 5: RECORD -> RECORD_AND_SAVE
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            p.stop()
            # cycle 2 R&S -> stop fires on_trace_ready a second time
            self.assertEqual(callback_count[0], 2)

    def test_device_stopped_persists_through_cycle(self):
        # [FUNCTION DOC] Once in device-stopped, successive steps stay there while the
        # schedule says record and record-and-save, and leave to warmup when the
        # schedule starts a new cycle.
        """Once in DEVICE_STOPPED, the profiler stays there until the schedule
        moves to WARMUP (new cycle) or NONE (end)."""
        p = self._make_profiler(wait=0, warmup=1, active=4, repeat=2)
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 1 -> 2: RECORD -> DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: schedule says RECORD, stays DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 3 -> 4: schedule says RECORD_AND_SAVE, stays DEVICE_STOPPED
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 4 -> 5: schedule rolls over to WARMUP (new cycle), exits
            # DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.stop()

    def test_device_stopped_recovers_without_warmup(self):
        # [FUNCTION DOC] With no warmup or wait, device-stopped must still exit at the cycle
        # boundary (after the record-and-save step) back into record.
        """With warmup=0 (and wait=0), the schedule never produces WARMUP or
        NONE between cycles. DEVICE_STOPPED must still exit at cycle
        boundaries (RECORD_AND_SAVE -> next), otherwise the profiler would
        be stuck in DEVICE_STOPPED forever."""
        # The normal schedule for wait=0, warmup=0, active=3, repeat=2:
        # step 0..1: RECORD
        # step 2:    RECORD_AND_SAVE
        # step 3..4: RECORD
        # step 5:    RECORD_AND_SAVE
        # step 6+:   NONE
        p = self._make_profiler(wait=0, warmup=0, active=3, repeat=2)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: prev raw schedule output was RECORD_AND_SAVE, so
            # this step is a cycle boundary. Exit DEVICE_STOPPED into
            # current_action (RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.stop()

    def test_device_stopped_exits_via_wait_phase(self):
        # [FUNCTION DOC] With a wait phase, device-stopped is left when the schedule moves to the
        # idle phase, and normal warmup and record follow.
        """With wait > 0, the NONE phase between cycles is the natural
        exit path out of DEVICE_STOPPED. After the wait, profiling resumes
        normally through WARMUP and RECORD."""
        # The normal schedule for wait=2, warmup=1, active=2, repeat=2 (cycle length 5):
        # step 0..1: NONE
        # step 2:    WARMUP
        # step 3:    RECORD
        # step 4:    R&S
        # step 5..6: NONE
        # step 7:    WARMUP
        # step 8:    RECORD
        # step 9:    R&S
        # step 10+:  NONE
        p = self._make_profiler(wait=2, warmup=1, active=2, repeat=2)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        with patch(self.PATCH_TARGET, return_value=False):
            # step 4 -> 5: schedule rolls into cycle 2's wait phase. NONE is
            # in the exit set, so DEVICE_STOPPED exits.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            # step 5 -> 6: still in wait phase
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            # step 6 -> 7: WARMUP (cycle 2 begins)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            # step 7 -> 8: RECORD (cycle 2 active begins)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.stop()

    def test_device_stopped_persists_through_multistep_warmup(self):
        # [FUNCTION DOC] With a multi-step warmup, device-stopped persists through
        # warmup, record and record-and-save steps and only exits to none at the end
        # of the cycle (single repeat).
        """With warmup >= 2, consecutive WARMUP steps are mid-cycle, not
        cycle boundaries. DEVICE_STOPPED must persist through the entire
        warmup phase rather than oscillating WARMUP -> DEVICE_STOPPED ->
        WARMUP -> DEVICE_STOPPED on every step."""
        # The normal schedule for wait=0, warmup=3, active=2, repeat=1 (cycle length 5):
        # step 0..2: WARMUP
        # step 3:    RECORD
        # step 4:    R&S
        # step 5+:   NONE
        p = self._make_profiler(wait=0, warmup=3, active=2, repeat=1)

        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 1 -> 2: schedule still WARMUP (same warmup phase, NOT a
            # cycle boundary). Must stay in DEVICE_STOPPED.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 2 -> 3: schedule moves to RECORD. Still mid-cycle
            # (prev_schedule_action is WARMUP, not RECORD_AND_SAVE). Stay.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 3 -> 4: schedule moves to R&S. Still mid-cycle. Stay.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            # step 4 -> 5: prev_schedule_action == R&S marks the cycle
            # boundary (here also end-of-profiling since repeat=1). Exit
            # DEVICE_STOPPED into NONE.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.stop()

    def test_warmup_to_device_stopped_fires_trace_ready(self):
        # [FUNCTION DOC] Checks entering device-stopped from warmup fires the trace-ready callback
        # exactly once per cycle for an active=1 schedule, and exiting it back to warmup does not
        # fire it again.
        """When DEVICE_STOPPED is entered from WARMUP (e.g. active=1
        schedule where every R&S step gets converted to DS), the
        (WARMUP, DEVICE_STOPPED) transition must fire on_trace_ready so
        the user's callback is invoked exactly once per cycle. Without
        this, active=1 schedules with persistent kineto_stopped would
        silently drop all callbacks; with double-firing we'd produce
        spurious callbacks."""
        callback_count = [0]

        def handler(prof):
            # [FUNCTION DOC] HELPER: trace-ready callback that increments the shared counter.
            callback_count[0] += 1

        # Schedule active=1, warmup=1, repeat=3 cycles W, R&S; with kineto_stopped
        # on every step the entry guard converts each cycle's R&S to DS via
        # (W, DS), and DS exits at cycle boundary via (DS, W).
        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=3),
            on_trace_ready=handler,
        )

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()

        with patch(self.PATCH_TARGET, return_value=True):
            # Cycle 1: step 0 -> 1: (W, DS) fires _trace_ready. callback=1.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            self.assertEqual(callback_count[0], 1)
            # step 1 -> 2: (DS, W) fires prepare_trace, no callback. callback=1.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            self.assertEqual(callback_count[0], 1)
            # Cycle 2: step 2 -> 3: (W, DS) fires _trace_ready. callback=2.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            self.assertEqual(callback_count[0], 2)
            p.stop()

    def test_start_after_stop_in_device_stopped(self):
        # [FUNCTION DOC] Stops while in device-stopped, then starts again: the second start
        # re-derives the action from the schedule (record-and-save), actually records, and on stop
        # fires the callback again (two callbacks in total).
        """A user may call start() after a stop() that left current_action
        in DEVICE_STOPPED. The second start() must re-derive state from
        the schedule (not trust the stale DS), so a real (NONE, X)
        transition fires prepare_trace and the second run actually records."""
        callback_count = [0]

        def handler(prof):
            # [FUNCTION DOC] HELPER: trace-ready callback that increments the shared counter.
            callback_count[0] += 1

        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
            on_trace_ready=handler,
        )

        # Run 1: enter DEVICE_STOPPED, stop while in it.
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
            p.stop()
        # (W, DS) fired _trace_ready during run 1.
        self.assertEqual(callback_count[0], 1)

        # Run 2: must actually profile, not be a silent no-op. step_num is
        # carried over, so schedule(step_num=1) is the active step (R&S).
        # start() re-derives current_action from the schedule, so we get
        # (NONE, R&S) -> prepare_trace + start_trace.
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            p.stop()
        # (R&S, None) fires _trace_ready -> callback_count increments.
        self.assertEqual(
            callback_count[0],
            2,
            "second run must produce a trace, not silently no-op",
        )

    def test_user_schedule_returning_device_stopped_raises_at_init(self):
        # [FUNCTION DOC] Asserts that constructing a profiler whose schedule returns
        # device-stopped at step zero raises ValueError saying the action is set internally.
        """ProfilerAction.DEVICE_STOPPED is internal; user-provided
        schedules must not return it. __init__ catches schedules that
        return it at step 0 — without this check, start() would silently
        no-op the (NONE, DS) transition and leave the profiler dead."""
        with self.assertRaisesRegex(ValueError, "DEVICE_STOPPED is set internally"):
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=lambda step: ProfilerAction.DEVICE_STOPPED,
            )

    def test_user_schedule_returning_device_stopped_raises_at_step(self):
        # [FUNCTION DOC] Uses a schedule that returns device-stopped from step one on; checks step
        # raises ValueError, step number and current action are unchanged, the step record
        # function is cleared, and stop then completes without error.
        """If the user schedule returns DEVICE_STOPPED on a later step,
        step() raises ValueError and leaves the profiler state untouched
        so the user can stop() cleanly after catching the error."""

        def bad_schedule(step):
            # [FUNCTION DOC] HELPER: schedule returning warmup at step zero and the internal
            # device-stopped action afterwards.
            if step == 0:
                return ProfilerAction.WARMUP
            return ProfilerAction.DEVICE_STOPPED

        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=bad_schedule,
        )
        p.start()
        step_num_before = p.step_num
        current_action_before = p.current_action
        with self.assertRaisesRegex(ValueError, "DEVICE_STOPPED is set internally"):
            p.step()
        # Profiler state must be unchanged so stop() can finalize cleanly.
        self.assertEqual(p.step_num, step_num_before)
        self.assertEqual(p.current_action, current_action_before)
        # step_rec_fn was exited at the top of step() before the raise; it
        # must have been cleared so stop() doesn't double-exit it.
        self.assertIsNone(p.step_rec_fn)
        # stop() runs without re-exiting an already-exited record_function.
        p.stop()

    def test_device_stopped_recovers_across_infinite_cycles(self):
        # [FUNCTION DOC] With unlimited repeat, enters device-stopped at the record-and-save step
        # of one cycle and recovers to warmup at the next cycle boundary, twice in a row.
        """With repeat=0 (infinite cycles), DEVICE_STOPPED entry and
        recovery must work across multiple cycles, not just one."""
        # Normal schedule for wait=0, warmup=1, active=2, repeat=0 (cycle length 3, infinite):
        # step 0: WARMUP
        # step 1: RECORD
        # step 2: RECORD_AND_SAVE
        # step 3: WARMUP
        # step 4: RECORD
        # step 5: RECORD_AND_SAVE
        # step 6: WARMUP
        # ...
        p = self._make_profiler(wait=0, warmup=1, active=2, repeat=0)

        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        # Cycle 1: enter DS at the R&S step.
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()  # step 2: prev=R, schedule=R&S -> DS
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        # Cycle 1 -> Cycle 2: exit DS at the R&S boundary.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 3: prev=DS, prev_schedule=R&S -> WARMUP
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()  # step 4: WARMUP -> RECORD
            self.assertEqual(p.current_action, ProfilerAction.RECORD)

        # Cycle 2: enter DS again.
        with patch(self.PATCH_TARGET, return_value=True):
            p.step()  # step 5: prev=R, schedule=R&S -> DS
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        # Cycle 2 -> Cycle 3: exit again. Confirms recovery is not a
        # one-shot — it works across arbitrary cycles.
        with patch(self.PATCH_TARGET, return_value=False):
            p.step()  # step 6: prev=DS, prev_schedule=R&S -> WARMUP
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.stop()

    def test_skips_device_stopped_when_prev_is_none(self):
        # [FUNCTION DOC] During the wait phase (previous action none) a true stopped signal does
        # not move the profiler into device-stopped.
        p = self._make_profiler(wait=2, warmup=1, active=1)
        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            # prev_action is NONE so the entry guard for DEVICE_STOPPED does not
            # fire.
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.stop()

    def test_skips_device_stopped_when_already_stopping(self):
        # [FUNCTION DOC] When the schedule ends the cycle (record-and-save then none), a true
        # stopped signal does not cause device-stopped; the action is none.
        p = self._make_profiler(wait=0, warmup=1, active=1, repeat=1)
        with patch(self.PATCH_TARGET, return_value=False):
            p.start()
            self.assertEqual(p.current_action, ProfilerAction.WARMUP)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)

        with patch(self.PATCH_TARGET, return_value=True):
            # step 1 -> 2: RECORD_AND_SAVE -> NONE. _is_kineto_stopped is True,
            # but the schedule already transitions to NONE, so the entry guard
            # for DEVICE_STOPPED does not fire (current_action != NONE).
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.NONE)
            p.stop()

    @parametrize(
        "prev",
        [ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE],
    )
    def test_unreachable_transition_raises(self, prev):
        # [FUNCTION DOC] Parametrized on previous action none or record-and-save: calling the
        # internal transition routine into device-stopped must raise RuntimeError saying that
        # transition is unreachable.
        """Transitions into DEVICE_STOPPED that step() / start() / __init__
        all exclude should raise if invoked directly via _transit_action.

        - prev=NONE: step()'s entry guard excludes it, and start() /
          __init__ reject DEVICE_STOPPED from user schedules, so
          current_action is never DEVICE_STOPPED at _transit_action(NONE, ...).
        - prev=RECORD_AND_SAVE: entry guard excludes it because the
          natural (R&S, *) transitions handle cycle-boundary cleanup.
        """
        p = self._make_profiler(wait=0, warmup=1, active=1)
        with self.assertRaisesRegex(
            RuntimeError,
            f"Profiler internal error: {prev.name} -> DEVICE_STOPPED should be unreachable",
        ):
            p._transit_action(prev, ProfilerAction.DEVICE_STOPPED)

    def test_cpu_only_profiler_ignores_stale_is_kineto_stopped(self):
        # [FUNCTION DOC] A CPU-only profiler ignores a true stopped signal and follows the
        # schedule (record, then record-and-save).
        """A CPU-only profiler should not enter DEVICE_STOPPED even if
        _is_kineto_stopped returns True (e.g. stale flag from a previous
        device profiler)."""
        p = profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
        )
        with patch(self.PATCH_TARGET, return_value=True):
            p.start()
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD)
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)
            p.stop()

    def test_device_stopped_with_real_kineto(self):
        # [FUNCTION DOC] Integration test with a live Kineto session (CPU+CUDA, warmup 1, active
        # 4, repeat 2). Records two steps, then with the stopped signal patched to true takes one
        # more step and checks the action becomes device-stopped and the trace-ready callback ran
        # once with a non-empty event list. Further steps stay device-stopped until warmup of
        # cycle two, whose four active steps end in record-and-save; stop fires the callback again
        # with events.
        """Integration test: when _is_kineto_stopped flips True mid-run
        against a live Kineto session, the DEVICE_STOPPED transition must
        actually fire stop_trace + _trace_ready (not just flip the action
        enum) — and produce a non-empty partial trace. After the schedule
        rolls over, the next cycle must run cleanly."""
        event_counts: list[int] = []

        def handler(prof):
            # [FUNCTION DOC] HELPER: trace-ready callback that appends the number of
            # profiled events.
            event_counts.append(len(list(prof.events())))

        p = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=4, repeat=2),
            on_trace_ready=handler,
        )

        p.start()
        self.assertEqual(p.current_action, ProfilerAction.WARMUP)

        # Cycle 1: two RECORD steps with real workload (steps 1 and 2).
        for _ in range(2):
            torch.add(torch.ones(2, 2), torch.ones(2, 2))
            p.step()
        self.assertEqual(p.current_action, ProfilerAction.RECORD)

        # Step 3: under the patch, RECORD -> DEVICE_STOPPED fires stop_trace +
        # _trace_ready against the live Kineto session, so handler() runs with
        # whatever Kineto collected during steps 1-2.
        with patch(self.PATCH_TARGET, return_value=True):
            torch.add(torch.ones(2, 2), torch.ones(2, 2))
            p.step()
            self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)

        self.assertEqual(len(event_counts), 1)
        self.assertGreater(
            event_counts[0], 0, "Partial cycle 1 trace should contain events"
        )

        # Walk through DEVICE_STOPPED until the schedule rolls over to WARMUP.
        # Patch is removed; persistence guard alone keeps us in DEVICE_STOPPED
        # because it only checks prev_action, not _is_kineto_stopped.
        p.step()  # step 4: schedule says RECORD_AND_SAVE, persists DEVICE_STOPPED
        self.assertEqual(p.current_action, ProfilerAction.DEVICE_STOPPED)
        p.step()  # step 5: schedule says WARMUP, exits DEVICE_STOPPED via prepare_trace
        self.assertEqual(p.current_action, ProfilerAction.WARMUP)

        # Cycle 2: full active phase against a freshly prepared Kineto session.
        for _ in range(4):
            torch.add(torch.ones(2, 2), torch.ones(2, 2))
            p.step()
        self.assertEqual(p.current_action, ProfilerAction.RECORD_AND_SAVE)

        # p.stop() transits RECORD_AND_SAVE -> None, firing _trace_ready #2.
        p.stop()

        self.assertEqual(len(event_counts), 2)
        self.assertGreater(event_counts[1], 0, "Cycle 2 trace should contain events")


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestProfilerEventsParity(TestCase):
    # [CLASS DOC] Tests (Kineto and CUDA required) that the events list API reports the same
    # information as the exported chrome trace JSON: Python functions, flows, timestamps, op args,
    # external ids, activity types and device metadata.
    """Tests validating parity between events() and export_chrome_trace() JSON."""

    def test_python_function_events_in_events(self):
        # [FUNCTION DOC] Profiles a module call with stack recording and checks events contain
        # Python function events with positive durations whose count equals the python_function
        # complete events in the trace, and that the module event's Python id, parent id and
        # module id equal the trace args.
        class DummyModule(nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns x plus 1.
                return x + 1

        mod = DummyModule()
        with profile(
            activities=[ProfilerActivity.CPU],
            with_stack=True,
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as prof:
            mod(torch.randn(4, 4))

        events = prof.events()
        python_events = [e for e in events if e.is_python_function]
        self.assertGreater(len(python_events), 0)
        for e in python_events:
            self.assertIsInstance(e.name, str)
            self.assertGreater(e.time_range.end - e.time_range.start, 0)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

            json_py = [
                e
                for e in trace["traceEvents"]
                if e.get("cat") == "python_function" and e.get("ph") == "X"
            ]
            self.assertEqual(len(python_events), len(json_py))

            # Verify python_id/parent_id/module_id parity with JSON args
            fe_mod = next((e for e in events if "DummyModule" in e.name), None)
            self.assertIsNotNone(fe_mod)
            self.assertGreater(fe_mod.python_id, 0)
            self.assertGreaterEqual(fe_mod.python_module_id, 0)

            json_mod = next(
                (e for e in json_py if "DummyModule" in e.get("name", "")),
                None,
            )
            self.assertIsNotNone(json_mod)
            args = json_mod["args"]
            self.assertEqual(fe_mod.python_id, args["Python id"])
            self.assertEqual(fe_mod.python_parent_id, args["Python parent id"])
            self.assertEqual(fe_mod.python_module_id, args["Python module id"])

    def test_key_averages_excludes_python_functions_by_default(self):
        # [FUNCTION DOC] Checks that key averages leave out Python function events (a thread wait
        # from the threading module) by default, and include them when asked with the
        # include-python-functions option.
        """key_averages() must not include Python function events (e.g. threading.py: wait)
        by default; they can be opted in with include_python_functions=True."""
        t = threading.Thread(target=lambda: time.sleep(0.05))
        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            t.start()
            t.join()

        avgs = prof.key_averages()
        threading_entries = [e for e in avgs if "threading" in e.key]
        self.assertEqual(
            len(threading_entries),
            0,
            f"key_averages() should not include threading.py events by default, got: {[e.key for e in threading_entries]}",
        )

        avgs_with_py = prof.key_averages(include_python_functions=True)
        threading_entries_with_py = [e for e in avgs_with_py if "threading" in e.key]
        self.assertGreater(
            len(threading_entries_with_py),
            0,
            "key_averages(include_python_functions=True) should include threading.py events",
        )

    def test_profiler_flow_events_parity(self):
        # [FUNCTION DOC] Profiles a CUDA matmul and compares the ids of asynchronous CPU to GPU
        # flow start and end events (flow type 2) from the events list with the ac2g flow events
        # in the exported trace; also checks the flow field types.
        """Verify that async CPU->GPU flow fields on events() match Chrome trace JSON."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device="cuda")
            torch.mm(x, x)

        # Collect async CPU->GPU flow info from events()
        events_with_flow = [
            e for e in prof.events() if e.flow_id is not None and e.flow_id != 0
        ]
        self.assertGreater(
            len(events_with_flow), 0, "No flow events found via events()"
        )

        for e in events_with_flow:
            self.assertIsInstance(e.flow_id, int)
            self.assertIsInstance(e.flow_type, int)
            self.assertIsInstance(e.flow_start, bool)

        # Verify parity with Chrome trace JSON for async CPU->GPU flow
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "ac2g"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            # kLinkAsyncCpuGpu = 2
            ac2g_events = [e for e in events_with_flow if e.flow_type == 2]
            events_flow_starts = {e.flow_id for e in ac2g_events if e.flow_start}
            events_flow_ends = {e.flow_id for e in ac2g_events if not e.flow_start}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "Async CPU->GPU flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "Async CPU->GPU flow end IDs differ between events() and Chrome trace",
            )

    def test_profiler_fwdbwd_flow_events_parity(self):
        # [FUNCTION DOC] Profiles a forward and a backward on CPU and compares forward-backward
        # flow start and end ids (flow type 1) from the events list with the fwdbwd flow events in
        # the trace.
        """Verify that fwd->bwd flow fields on events() match Chrome trace JSON."""
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            t1 = torch.ones(1, requires_grad=True)
            t2 = torch.ones(1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()

        fwdbwd_events = [
            e for e in prof.events() if e.flow_type == 1 and e.flow_id != 0
        ]
        self.assertGreater(
            len(fwdbwd_events), 0, "No fwdbwd flow events found via events()"
        )

        events_flow_starts = {e.flow_id for e in fwdbwd_events if e.flow_start}
        events_flow_ends = {e.flow_id for e in fwdbwd_events if not e.flow_start}

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "fwdbwd"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "fwdbwd flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "fwdbwd flow end IDs differ between events() and Chrome trace",
            )

    def test_profiler_timestamp_consistency(self):
        # [FUNCTION DOC] Checks that the start time of the matmul event, converted
        # from microseconds to nanoseconds, plus the trace start time, minus the trace
        # base time and scaled back to microseconds, equals the timestamp of the same
        # event in the exported trace.
        """Verify that FunctionEvent timestamps can reconstruct Chrome trace ts values."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device="cuda")
            torch.mm(x, x)

        trace_start_ns = prof.profiler.kineto_results.trace_start_ns()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            # Chrome trace is relative to a different base time which is not exposed in Python.
            # It's probably not important to do so as we still have the relative differences
            # in duration.
            base_time_ns = j.get("baseTimeNanoseconds", 0)

            # Grab mm timestamp from events() and json
            fe_mm = next((e for e in prof.events() if e.name == "aten::mm"), None)
            json_mm = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::mm" and e.get("ph") == "X"
                ),
                None,
            )

            # Reconstruct Chrome trace ts from events():
            # absolute_ns = mm_op_start_us * 1000 + trace_start_ns
            # chrome_ts = (absolute_ns - base_time_ns) / 1000 -> realign with json timeframe
            absolute_ns = int(fe_mm.time_range.start * 1000) + trace_start_ns
            recovered_ts = (absolute_ns - base_time_ns) / 1000
            self.assertEqual(
                recovered_ts,
                json_mm["ts"],
                msg="Recovered Chrome trace ts doesn't match JSON for aten::mm",
            )

    def test_profiler_op_args_events_parity(self):
        # [FUNCTION DOC] Profiles an add of strided views and a cat of a tensor list with shapes
        # recorded, and checks the structured input shapes, structured input strides and input
        # dtypes of the events equal the Input Dims, Input Strides and Input type args of the
        # trace events.
        """Verify that cpu_op args on events() match Chrome trace JSON args."""
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        t1 = torch.ones((64, 32))
        t2 = torch.ones((64, 32))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            torch.add(a, b)
            torch.cat([t1, t2])

        fe_add = next((e for e in prof.events() if e.name == "aten::add"), None)
        self.assertIsNotNone(fe_add)
        fe_cat = next((e for e in prof.events() if e.name == "aten::cat"), None)
        self.assertIsNotNone(fe_cat)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_add = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::add" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_add)
            args = json_add["args"]
            self.assertEqual(fe_add.structured_input_shapes, args["Input Dims"])
            self.assertEqual(fe_add.structured_input_strides, args["Input Strides"])
            self.assertEqual(fe_add.input_dtypes, args["Input type"])

            # Test a case with TensorList inputs -- structured_input_shapes
            # should handle TensorList nesting correctly.
            json_cat = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::cat" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_cat)
            args_cat = json_cat["args"]
            self.assertEqual(fe_cat.structured_input_shapes, args_cat["Input Dims"])
            self.assertEqual(fe_cat.structured_input_strides, args_cat["Input Strides"])
            self.assertEqual(fe_cat.input_dtypes, args_cat["Input type"])

    def test_profiler_external_id_parity(self):
        # [FUNCTION DOC] Profiles CUDA work inside a named region and checks that the multiset of
        # (name, external id) pairs from the events list, ignoring zero ids, equals that from the
        # exported trace args.
        """Verify that FunctionEvent.external_id matches External id in Chrome trace JSON."""
        from collections import Counter

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("test_region"):
                x = torch.randn(32, 32, device="cuda")
                y = torch.mm(x, x)
                z = y + x
                z.cpu()
                torch.cuda.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_ext = Counter(
                (e["name"], e["args"]["External id"])
                for e in j["traceEvents"]
                if e.get("args", {}).get("External id") is not None
            )
            events_name_ext = Counter(
                (ev.name, ev.external_id) for ev in prof.events() if ev.external_id != 0
            )

            self.assertEqual(
                events_name_ext,
                json_name_ext,
                "(name, external_id) pairs differ between events() and Chrome trace JSON",
            )

    def test_profiler_activity_type_parity(self):
        # [FUNCTION DOC] Checks every event has a non-empty string activity type, the matmul event
        # is a cpu_op, and each event's (name, activity type) pair exists as a (name, category)
        # pair among the complete events of the exported trace.
        """Verify activity_type on events() matches Chrome trace cat field."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device="cuda")
            torch.mm(x, x)

        events = prof.events()
        for e in events:
            self.assertIsInstance(e.activity_type, str)
            self.assertGreater(len(e.activity_type), 0)

        mm_event = next((e for e in events if e.name == "aten::mm"), None)
        self.assertIsNotNone(mm_event)
        self.assertEqual(mm_event.activity_type, "cpu_op")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_cats = {
                (e["name"], e["cat"])
                for e in j["traceEvents"]
                if e.get("ph") == "X" and "cat" in e
            }
            for e in events:
                self.assertIn(
                    (e.name, e.activity_type),
                    json_name_cats,
                    lambda msg: f"{msg}\nactivity_type mismatch for {e.name}",
                )

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/179944")
    def test_structured_metadata_matches_chrome_trace(self):
        # [FUNCTION DOC] Profiles CUDA work with Kineto metadata exposure and compares
        # per-event structured metadata fields between the events list and the trace,
        # keyed by name, activity type, external id and correlation or id. Fails with
        # a long instructional message if the trace has argument keys not known to the
        # metadata key table, and also checks event identities and metadata
        # dictionaries are equal. Skipped on ROCm (issue linked).
        # Compare metadata fields between events() and Chrome trace JSON to make sure they stay in parity
        # 1. Run a dummy workload with profiling enabled and collect the json/events() outputs
        # 2. Parse each event instance in the json and events() to create a key->value mapping
        #      - The key is a tuple of metadata fields that should be unique for each event
        #      - The value is a dict of metadata fields for that event
        # 3. Ensure that the keys and values match between the json and events() outputs

        from torch.autograd.profiler_util import _EVENT_METADATA_KEYS

        target_cats = ("cuda_runtime", "gpu_memcpy", "kernel")
        allowed_non_structured_trace_keys = {
            "External id",
            "correlation",
            "cbid",
            "cid",
            "device",
            "kind",
            "kernel",
            "ptr",
            "src",
            "dst",
        }
        supported_trace_keys = set(_EVENT_METADATA_KEYS).union(
            allowed_non_structured_trace_keys
        )

        def metadata_dict_from_trace_args(args):
            # [FUNCTION DOC] HELPER: converts trace args to a mapping of structured field names to
            # values using the metadata key table, applying each key's converter to string values
            # and keeping other values.
            out = {}
            for kineto_key, (field_name, convert) in _EVENT_METADATA_KEYS.items():
                if kineto_key in args:
                    raw_value = args[kineto_key]
                    out[field_name] = (
                        convert(raw_value) if isinstance(raw_value, str) else raw_value
                    )
            return out

        def metadata_dict_from_function_event(fe):
            # [FUNCTION DOC] HELPER: returns the non-None fields of a function event's
            # metadata object by the field names of the key table, or an empty mapping
            # if the event has no metadata.
            if fe.event_metadata is None:
                return {}

            out = {}
            for field_name, _ in _EVENT_METADATA_KEYS.values():
                val = getattr(fe.event_metadata, field_name)
                if val is not None:
                    out[field_name] = val
            return out

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                expose_kineto_event_metadata=True
            ),
        ) as prof:
            x = torch.randn(10, 10, device="cuda")
            y = torch.mm(x, x)
            z = x + y
            z.cpu()

        # Build a mapping from key to events() FunctionEvent metadata
        event_records = {}
        for fe in prof.events():
            if fe.external_id == 0 or fe.id == 0 or fe.activity_type not in target_cats:
                continue
            # Using just one of these keys could result in collisions, so try to uniquely identify the event with all of them
            key = (fe.name, fe.activity_type, fe.external_id, fe.id)
            self.assertNotIn(
                key,
                event_records,
                lambda msg: f"{msg}\nDuplicate FunctionEvent record key encountered: {key}",
            )
            event_records[key] = metadata_dict_from_function_event(fe)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        json_records = {}
        # Track unexpected (event_name, cat, key) combos, deduplicated
        unexpected_combos: set[tuple[str, str, str]] = set()

        # Loop through the trace events to perform a comparison
        for te in trace["traceEvents"]:
            cat = te.get("cat", "")
            args = te.get("args", {})
            ext_id = args.get("External id")
            correlation = args.get("correlation")

            if ext_id is None or correlation is None:
                continue
            if cat not in target_cats:
                continue

            # Any metadata keys that show up in JSON should show up in events()
            for k in set(args) - supported_trace_keys:
                unexpected_combos.add((te["name"][:100], cat, k))

            # Build the same key from JSON to try to match with a FunctionEvent
            key = (te["name"], te["cat"], ext_id, correlation)
            self.assertNotIn(
                key,
                json_records,
                lambda msg: f"{msg}\nDuplicate Chrome trace record key encountered: {key}",
            )
            json_records[key] = metadata_dict_from_trace_args(args)

        failure_msg = """\
====================================================================================
IMPORTANT: Are you making a Kineto change or bumping the third_party/kineto
submodule hash and seeing this message?

New metadata keys (see below) were found in the Chrome trace JSON that are not
yet exposed through the profiler's events() API (i.e. EventMetadata in
torch/autograd/profiler_util.py).

To fix this properly, you need to make sure the new Kineto data makes its way
to the events() property. The steps are:

1. Add the new key(s) to _EVENT_METADATA_KEYS in torch/autograd/profiler_util.py
   with the appropriate field name and type converter.
2. Add corresponding field(s) to the EventMetadata dataclass in the same file.
3. If the key should NOT be mapped (e.g. it duplicates an existing FunctionEvent
   attribute), add it to allowed_non_structured_trace_keys in this test instead.

For a model PR to follow, see: https://github.com/pytorch/pytorch/pull/180100
===================================================================================="""
        if unexpected_combos:
            summary = "\n".join(
                f"  {name} ({cat}): {key!r}"
                for name, cat, key in sorted(unexpected_combos)
            )
            raise AssertionError(f"\n{failure_msg}\n\nUnmapped keys:\n{summary}")

        self.assertGreater(len(json_records), 0, "No device-side records were compared")
        self.assertEqual(
            set(event_records),
            set(json_records),
            "Device event identities differ between events() and Chrome trace JSON",
        )

        for key in json_records:
            expected_meta = json_records[key]
            actual_meta = event_records[key]
            self.assertEqual(
                actual_meta,
                expected_meta,
                lambda msg: f"{msg}\n{key}: structured metadata differs between events() and Chrome trace JSON",
            )

    def test_typed_metadata_matches_chrome_trace(self):
        # [FUNCTION DOC] Profiles CUDA work with metadata exposure, keys device-side events and
        # trace records by name, category, external id and correlation, and checks the sets of
        # keys match and every typed metadata field of each event equals the trace arg value (and
        # none are extra).
        # Match events to Chrome trace records by identity (via name, cat, external/correlation id),
        # then verify every metadata field has the same value in the matching trace record.
        target_cats = ("cuda_runtime", "gpu_memcpy", "kernel")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                expose_kineto_event_metadata=True
            ),
        ) as prof:
            x = torch.randn(10, 10, device="cuda")
            y = torch.mm(x, x)
            z = x + y
            z.cpu()

        event_records = {}
        for event in prof.events():
            if (
                event.external_id == 0
                or event.id == 0
                or event.activity_type not in target_cats
            ):
                continue
            key = (event.name, event.activity_type, event.external_id, event.id)
            self.assertNotIn(key, event_records)
            event_records[key] = dict(event.metadata or {})

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        json_records = {}
        for trace_event in trace["traceEvents"]:
            category = trace_event.get("cat", "")
            args = trace_event.get("args", {})
            external_id = args.get("External id")
            correlation = args.get("correlation")
            if (
                external_id is None
                or correlation is None
                or category not in target_cats
            ):
                continue
            key = (trace_event["name"], category, external_id, correlation)
            self.assertNotIn(key, json_records)
            json_records[key] = args

        self.assertGreater(len(json_records), 0)
        self.assertEqual(set(event_records), set(json_records))

        for key, trace_metadata in json_records.items():
            typed_metadata = event_records[key]
            self.assertGreater(len(typed_metadata), 0)
            self.assertEqual(set(typed_metadata) - set(trace_metadata), set())
            for field, value in typed_metadata.items():
                self.assertEqual(value, trace_metadata[field])


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestPythonChromeTraceExport(TestCase):
    # [CLASS DOC] Tests (Kineto and CUDA required) of the Python streaming chrome trace exporter
    # versus the native Kineto save, plus gzip and plain JSON output through the export function.
    """Verify that the Python streaming exporter produces traces equivalent
    to the C++ Kineto save() path."""

    def _profile_workload(self):
        # [FUNCTION DOC] HELPER: profiles twenty iterations of CUDA matmul and add on a 64 by 64
        # tensor, synchronizes, and returns the profiler.
        x = torch.randn(64, 64, device="cuda")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                torch.mm(x, x)
                torch.add(x, x)
            torch.cuda.synchronize()
        return prof

    def test_python_export_matches_kineto(self):
        # [FUNCTION DOC] Saves a trace using the native Kineto path and using the Python exporter,
        # loads both, and checks the complete (non-Trace-category) events agree in count, in the
        # multiset of category and name, event by event after sorting (ignoring the Ev Idx arg),
        # and in flow start and end events.
        prof = self._profile_workload()

        with tempfile.TemporaryDirectory() as tmpdir:
            kineto_path = os.path.join(tmpdir, "kineto.json")
            prof.profiler.kineto_results.save(kineto_path)

            py_path = os.path.join(tmpdir, "python.json")
            from torch.profiler._chrome_trace_export import (
                export_chrome_trace as _export,
            )

            _export(prof.profiler.kineto_results, py_path)

            with open(kineto_path) as f:
                kineto_trace = json.load(f)
            with open(py_path) as f:
                py_trace = json.load(f)

        kineto_events = kineto_trace["traceEvents"]
        py_events = py_trace["traceEvents"]

        def _activity_events(events):
            # [FUNCTION DOC] HELPER: returns the complete-phase events whose category
            # is not Trace.
            return [
                e
                for e in events
                if e.get("ph") == "X" and e.get("cat") not in ("Trace",)
            ]

        kineto_acts = _activity_events(kineto_events)
        py_acts = _activity_events(py_events)
        self.assertEqual(
            len(kineto_acts),
            len(py_acts),
            lambda msg: f"{msg}\nActivity event count mismatch: kineto={len(kineto_acts)}, python={len(py_acts)}",
        )

        from collections import Counter

        kineto_names = Counter((e["cat"], e["name"]) for e in kineto_acts)
        py_names = Counter((e["cat"], e["name"]) for e in py_acts)
        self.assertEqual(
            kineto_names, py_names, "Event (cat, name) distribution differs"
        )

        def _strip_ev_idx(event):
            # [FUNCTION DOC] HELPER: returns a shallow copy of the event with the Ev Idx argument
            # removed from a copied args mapping.
            """Return a copy with the internal 'Ev Idx' arg removed."""
            e = dict(event)
            args = dict(e.get("args", {}))
            args.pop("Ev Idx", None)
            e["args"] = args
            return e

        kineto_sorted = sorted(
            [_strip_ev_idx(e) for e in kineto_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        py_sorted = sorted(
            [_strip_ev_idx(e) for e in py_acts],
            key=lambda e: (e["cat"], e["name"], e["ts"]),
        )
        # Compare every key of every activity event.
        for ke, pe in zip(kineto_sorted, py_sorted):
            self.assertEqual(
                ke,
                pe,
                lambda msg: f"{msg}\nEvent mismatch for {ke['cat']}::{ke['name']} at ts={ke['ts']}",
            )

        # Flow event parity (ph, id, cat).
        kineto_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in kineto_events
                if e.get("ph") in ("s", "f")
            ],
        )
        py_flow = sorted(
            [
                (e["ph"], e["id"], e.get("cat", ""))
                for e in py_events
                if e.get("ph") in ("s", "f")
            ],
        )
        self.assertEqual(kineto_flow, py_flow, "Flow events differ")

    def test_python_export_gzip(self):
        # [FUNCTION DOC] Exports a trace to a gzip path with the Python exporter option and checks
        # the decompressed JSON has trace events including complete events.
        import gzip as gzip_mod

        prof = self._profile_workload()

        with tempfile.TemporaryDirectory() as tmpdir:
            gz_path = os.path.join(tmpdir, "trace.json.gz")
            prof.export_chrome_trace(gz_path, use_python_export=True)

            with gzip_mod.open(gz_path, "rt") as f:
                trace = json.load(f)

        self.assertIn("traceEvents", trace)
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertGreater(len(x_events), 0)

    def test_python_export_plain_json(self):
        # [FUNCTION DOC] Exports a trace to a plain JSON path with the Python exporter option and
        # checks the JSON has trace events including complete events.
        prof = self._profile_workload()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "trace.json")
            prof.export_chrome_trace(json_path, use_python_export=True)

            with open(json_path) as f:
                trace = json.load(f)

        self.assertIn("traceEvents", trace)
        x_events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
        self.assertGreater(len(x_events), 0)


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestMetadataJsonFormat(TestCase):
    # [CLASS DOC] Tests (Kineto and CUDA required) guarding the format of the metadata
    # text of kernel activities: the Python exporter splices it into JSON verbatim and
    # searches it by string.
    """Guard the format of ITraceActivity.metadataJson() for kernel events.

    The Python-side chrome trace exporter splices metadataJson() verbatim
    into the output JSON and extracts specific fields (like graph node id)
    via string matching. These tests ensure the format stays stable so that
    downstream consumers don't silently break.
    """

    def _get_kernel_metadata(self):
        # [FUNCTION DOC] HELPER: profiles a CUDA matmul up to three times until a kernel activity
        # appears and returns its metadata text; fails the test with diagnostic type counts if
        # none ever appears.
        x = torch.randn(64, 64, device="cuda")
        # The first profiled iteration(s) can drop CUDA kernels while CUPTI
        # warms up, so retry a few times before giving up.
        activities = []
        for _ in range(3):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                torch.mm(x, x)
                torch.cuda.synchronize()
            activities = prof.profiler.kineto_results.trace_activities()
            for act in activities:
                if act.type() == "kernel":
                    return act.metadata_json()

        raw_types = collections.Counter(act.type() for act in activities)
        event_cats = collections.Counter(
            e.activity_type for e in prof.events() if getattr(e, "activity_type", None)
        )
        self.fail(
            f"No kernel activity found. cuda={torch.version.cuda} "
            f"supported={supported_activities()} n={len(activities)} "
            f"trace_activities_types={dict(raw_types)} events_cats={dict(event_cats)}"
        )

    def test_metadata_json_is_valid_json_fragment(self):
        # [FUNCTION DOC] Checks the kernel metadata text, wrapped in braces, parses to
        # a non-empty mapping.
        md = self._get_kernel_metadata()
        parsed = json.loads("{" + md + "}")
        self.assertIsInstance(parsed, dict)
        self.assertGreater(len(parsed), 0)

    def test_kernel_metadata_has_expected_fields(self):
        # [FUNCTION DOC] Checks the parsed kernel metadata has device, stream, correlation, grid
        # and block, plus on CUDA builds the queue, context, register, shared memory, occupancy
        # and graph node fields.
        md = self._get_kernel_metadata()
        parsed = json.loads("{" + md + "}")
        common_keys = ["device", "stream", "correlation", "grid", "block"]
        cuda_only_keys = [
            "queued",
            "context",
            "registers per thread",
            "shared memory",
            "blocks per SM",
            "warps per SM",
            "graph node id",
        ]
        expected = common_keys if TEST_WITH_ROCM else common_keys + cuda_only_keys
        for key in expected:
            self.assertIn(
                key,
                parsed,
                lambda msg: f"{msg}\nMissing field '{key}' in kernel metadataJson",
            )

    def test_kernel_metadata_field_types(self):
        # [FUNCTION DOC] Checks integer-valued kernel metadata fields are integers, and grid and
        # block are lists of length three.
        md = self._get_kernel_metadata()
        parsed = json.loads("{" + md + "}")
        common_int_keys = ["device", "stream", "correlation"]
        cuda_only_int_keys = [
            "queued",
            "context",
            "registers per thread",
            "shared memory",
            "graph node id",
        ]
        int_keys = (
            common_int_keys if TEST_WITH_ROCM else common_int_keys + cuda_only_int_keys
        )
        for key in int_keys:
            self.assertIsInstance(
                parsed[key],
                int,
                lambda msg: f"{msg}\nExpected int for '{key}', got {type(parsed[key]).__name__}",
            )
        for key in ["grid", "block"]:
            self.assertIsInstance(parsed[key], list)
            self.assertEqual(len(parsed[key]), 3)

    def test_metadata_json_key_value_format(self):
        # [FUNCTION DOC] Checks the raw metadata text uses a colon followed by a space after the
        # key for the graph node id (not on ROCm), correlation and stream keys.
        md = self._get_kernel_metadata()
        # Verify the ": " (colon-space) separator convention that string
        # splicing in the export path relies on for field extraction.
        if not TEST_WITH_ROCM:
            self.assertIn('"graph node id": ', md)
        self.assertIn('"correlation": ', md)
        self.assertIn('"stream": ', md)


if __name__ == "__main__":
    run_tests()
