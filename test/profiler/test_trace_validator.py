# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests for the profiler chrome-trace validator rules. Provides device-type to
# profiler-activity mapping, a small ResNet50 model builder, and payload functions that profile a
# workload and return the exported trace events. Contains synthetic unit tests that feed
# hand-built event mappings to each rule, plus end-to-end test classes (agnostic and CUDA) that
# profile real workloads; both end-to-end classes are currently skipped at class level pending a
# kineto clock-skew fix. Import-time side effect: registers device-type instantiations of the
# agnostic class for all device types except cpu.

import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler._trace_validator import (
    _check_backward_seq_id_uniqueness,
    _check_gpu_kernel_causality,
    _check_nccl_metadata,
    _check_stream_sync_overlap,
    _check_stream_wait_corr_id_in_past,
    _check_stream_wait_corr_id_populated,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TestCase,
)


# ---------------------------------------------------------------------------
# Device-agnostic accelerator helpers
# ---------------------------------------------------------------------------

_DEVICE_TYPE_TO_ACTIVITY = {
    "cuda": ProfilerActivity.CUDA,
    "xpu": ProfilerActivity.XPU,
    "hpu": ProfilerActivity.HPU,
    "mtia": ProfilerActivity.MTIA,
    "privateuse1": ProfilerActivity.PrivateUse1,
}


def _activity_for_device_type(device_type):
    # [FUNCTION DOC] HELPER: Looks up the profiler activity for a device type string in the
    # module mapping. Raises ValueError telling the caller to extend the mapping if the type
    # is unknown.
    """Map a device type string to its ProfilerActivity enum."""
    if device_type not in _DEVICE_TYPE_TO_ACTIVITY:
        raise ValueError(
            f"No ProfilerActivity mapping for device type {device_type!r}. "
            f"Add it to _DEVICE_TYPE_TO_ACTIVITY."
        )
    return _DEVICE_TYPE_TO_ACTIVITY[device_type]


def _sync_device(device):
    # [FUNCTION DOC] HELPER: Synchronizes the device's backend module in the torch namespace if
    # it has a synchronize function. Does nothing for no device, cpu, or a backend without one.
    """Synchronize the given device's backend."""
    if device is None or device.type == "cpu":
        return
    mod = getattr(torch, device.type, None)
    if mod and hasattr(mod, "synchronize"):
        mod.synchronize()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    # [CLASS DOC] HELPER: ResNet bottleneck block module with three convolution and batch norm
    # stages (1x1, 3x3 with the given stride, 1x1) and an in-place ReLU. A projection shortcut of
    # convolution plus batch norm is created only when channel counts differ or stride is not
    # one; otherwise the shortcut is the identity.
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        # [FUNCTION DOC] HELPER: Builds the three conv and batch norm layers, the ReLU, and
        # the optional downsample projection from the input, middle and output channel counts
        # and stride.
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else None
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the three conv/batch-norm stages with ReLU after the first
        # two, computes the shortcut (projected input if a downsample exists, else the input),
        # and returns the ReLU of the sum.
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_ch, mid_ch, out_ch, n, stride=1):
    # [FUNCTION DOC] HELPER: Builds a sequential stage of the given number of bottleneck blocks.
    # The first block changes channels from the input to the output count with the given stride;
    # the rest map output channels to output channels with stride one.
    layers = [_Bottleneck(in_ch, mid_ch, out_ch, stride=stride)]
    for _ in range(n - 1):
        layers.append(_Bottleneck(out_ch, mid_ch, out_ch))
    return nn.Sequential(*layers)


def _resnet50():
    # [FUNCTION DOC] HELPER: Builds an untrained ResNet50-style sequential network: strided 7x7
    # conv, batch norm, ReLU, max pool, four stages of 3, 4, 6 and 3 blocks, adaptive average
    # pool, flatten, and a linear layer to 1000 outputs.
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        _make_layer(64, 64, 256, n=3),
        _make_layer(256, 128, 512, n=4, stride=2),
        _make_layer(512, 256, 1024, n=6, stride=2),
        _make_layer(1024, 512, 2048, n=3, stride=2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 1000),
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _profile_resnet_payload(trace_path):
    # [FUNCTION DOC] HELPER: CUDA-only payload. Builds a ResNet50 and random input on the first
    # CUDA device, then profiles three iterations of forward, backward and optimizer step (each
    # in a named record_function scope) with CPU and CUDA activities and CUDA sync events
    # enabled. Exports a chrome trace to the given path and returns the loaded events.
    """Profile a ResNet50 training loop and return events (CUDA-specific)."""
    device = torch.device("cuda:0")
    model = _resnet50().to(device)
    inputs = torch.randn(4, 3, 224, 224, device=device)
    outputs = torch.rand_like(model(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
    ) as prof:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with record_function("## forward ##"):
                pred = model(inputs)
            with record_function("## backward ##"):
                loss_fn(pred, outputs).backward()
            with record_function("## optimizer ##"):
                optimizer.step()

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return _load_events(trace_path)


def _profile_complex_payload(trace_path):
    # [FUNCTION DOC] HELPER: CUDA-only payload. Under a profiler with CUDA sync events, runs a
    # matrix multiply with backward, then multiplies on one stream, records a CUDA event, makes a
    # second stream wait on it and multiplies there, and synchronizes the second stream. Exports
    # the trace to the given path and returns the loaded events.
    """Profile multi-stream + CUDA events + forward/backward (CUDA-specific)."""
    device = torch.device("cuda:0")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
    ) as prof:
        x = torch.randn(32, 32, device=device, requires_grad=True)
        y = torch.mm(x, x)
        loss = y.sum()
        loss.backward()

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(s1):
            a = torch.randn(64, 64, device=device)
            _b = torch.mm(a, a)
            event.record(s1)
        s2.wait_event(event)
        with torch.cuda.stream(s2):
            _c = torch.mm(a, a)
        s2.synchronize()

    torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    return _load_events(trace_path)


def _profile_training_payload(trace_path, device, activity):
    # [FUNCTION DOC] HELPER: Accelerator-agnostic variant of the resnet payload: same training
    # loop of three iterations on the given device, profiled with CPU and the given activity but
    # without CUDA-specific configuration. Syncs the device before and after, exports the trace
    # to the given path and returns the loaded events.
    """Profile a ResNet50 training loop on any accelerator and return events.

    Unlike _profile_resnet_payload, this avoids CUDA-specific APIs
    (_ExperimentalConfig, torch.cuda.Stream, torch.cuda.Event).
    """
    model = _resnet50().to(device)
    inputs = torch.randn(4, 3, 224, 224, device=device)
    outputs = torch.rand_like(model(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    _sync_device(device)

    with profile(activities=[ProfilerActivity.CPU, activity]) as prof:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with record_function("## forward ##"):
                pred = model(inputs)
            with record_function("## backward ##"):
                loss_fn(pred, outputs).backward()
            with record_function("## optimizer ##"):
                optimizer.step()

    _sync_device(device)
    prof.export_chrome_trace(trace_path)
    return _load_events(trace_path)


def _load_events(trace_path):
    # [FUNCTION DOC] HELPER: Reads a trace JSON file and returns its trace events list if the top
    # level is an object with that key; if the key is missing, returns the whole loaded data.
    with open(trace_path) as f:
        data = json.load(f)
    return data.get("traceEvents", data)


# ---------------------------------------------------------------------------
# E2E mixin — shared helpers for E2E test classes
# ---------------------------------------------------------------------------


class _TraceValidatorE2EMixin:
    # [CLASS DOC] HELPER: Mixin for the end-to-end test classes. Expects the host class to
    # define a payloads mapping from payload name to event list. Provides event lookup and
    # violation formatting.
    """Shared helpers for E2E trace validator test classes."""

    def _events(self, payload):
        # [FUNCTION DOC] HELPER: Returns the stored event list for the named payload; raises
        # KeyError if missing.
        return self._payloads[payload]

    @staticmethod
    def _fmt(violations, limit=5):
        # [FUNCTION DOC] HELPER: Formats up to limit violations (default five) as indented
        # lines joined by newlines, adding a final line with the count of omitted ones if
        # there are more.
        lines = [f"  {v}" for v in violations[:limit]]
        if len(violations) > limit:
            lines.append(f"  ... and {len(violations) - limit} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accelerator-UNRELATED tests (CPU, synthetic event dicts)
# ---------------------------------------------------------------------------


class TestTraceValidatorRules(TestCase):
    # [CLASS DOC] Synthetic unit tests of each validation rule using hand-built event mappings;
    # no profiling or devices involved. For every rule there is a passing case returning an empty
    # list and a failing case checking the violation count, rule name and message text.
    """Synthetic unit tests for validation rules.

    Category: Accelerator-UNRELATED

    Each validator is a pure list[dict] -> list[Violation] function.
    These tests verify rule logic using hand-crafted event dictionaries.
    No profiling or device operations — runs on any machine.
    """

    # --- _check_nccl_metadata ---

    def test_nccl_metadata_pass(self):
        # [FUNCTION DOC] A communication event carrying all five required metadata arguments
        # yields no NCCL metadata violations.
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {
                    "Collective name": "all_reduce",
                    "dtype": "float32",
                    "In msg nelems": 1024,
                    "Out msg nelems": 1024,
                    "Group size": 8,
                },
            },
        ]
        self.assertEqual(_check_nccl_metadata(events), [])

    def test_nccl_metadata_fail(self):
        # [FUNCTION DOC] A communication event with only the collective name argument yields
        # exactly one violation attributed to the NCCL metadata rule.
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {"Collective name": "all_reduce"},
            },
        ]
        v = _check_nccl_metadata(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_nccl_metadata")

    def test_nccl_metadata_partial_fields(self):
        # [FUNCTION DOC] A communication event lacking the in and out message element counts
        # yields one violation whose message mentions both missing field names.
        events = [
            {
                "ph": "X",
                "name": "record_param_comms",
                "ts": 100,
                "dur": 10,
                "args": {
                    "Collective name": "all_reduce",
                    "dtype": "float32",
                    "Group size": 8,
                },
            },
        ]
        v = _check_nccl_metadata(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_nccl_metadata")
        self.assertIn("In msg nelems", v[0].message)
        self.assertIn("Out msg nelems", v[0].message)

    # --- _check_gpu_kernel_causality ---

    def test_gpu_kernel_causality_pass(self):
        # [FUNCTION DOC] A launch at time 100 and a kernel at time 110 sharing an External id
        # gives no causality violations.
        events = [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 100,
                "dur": 5,
                "args": {"External id": 1, "correlation": 10},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "volta_sgemm",
                "ts": 110,
                "dur": 50,
                "args": {"External id": 1, "correlation": 10},
            },
        ]
        self.assertEqual(_check_gpu_kernel_causality(events), [])

    def test_gpu_kernel_causality_fail(self):
        # [FUNCTION DOC] A kernel starting at time 100, before its launch at time 200 (same
        # External id), gives one violation from the causality rule whose message says the kernel
        # starts before its launch.
        events = [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 200,
                "dur": 5,
                "args": {"External id": 1, "correlation": 10},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "volta_sgemm",
                "ts": 100,
                "dur": 50,
                "args": {"External id": 1, "correlation": 10},
            },
        ]
        v = _check_gpu_kernel_causality(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_gpu_kernel_causality")
        self.assertIn("before its cudaLaunchKernel", v[0].message)

    # --- _check_stream_wait_corr_id_populated ---

    def test_stream_wait_corr_id_populated_pass(self):
        # [FUNCTION DOC] A stream wait event with a correlation id of 42 gives no
        # populated-id violations.
        events = [
            {
                "ph": "X",
                "name": "cudaStreamWaitEvent",
                "ts": 100,
                "dur": 1,
                "args": {
                    "cuda_sync_kind": "Stream Wait Event",
                    "wait_on_cuda_event_record_corr_id": 42,
                    "device": 0,
                    "stream": 7,
                },
            },
        ]
        self.assertEqual(_check_stream_wait_corr_id_populated(events), [])

    def test_stream_wait_corr_id_populated_fail(self):
        # [FUNCTION DOC] A stream wait event whose correlation id is negative one gives exactly
        # one violation from the populated-id rule.
        events = [
            {
                "ph": "X",
                "name": "cudaStreamWaitEvent",
                "ts": 100,
                "dur": 1,
                "args": {
                    "cuda_sync_kind": "Stream Wait Event",
                    "wait_on_cuda_event_record_corr_id": -1,
                    "device": 0,
                    "stream": 7,
                },
            },
        ]
        v = _check_stream_wait_corr_id_populated(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_stream_wait_corr_id_populated")

    # --- _check_stream_sync_overlap ---

    def test_stream_sync_overlap_pass(self):
        # [FUNCTION DOC] A kernel running 100 to 150 and a stream sync starting at 200 on the
        # same device and stream gives no overlap violations.
        events = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "ts": 100,
                "dur": 50,
                "args": {"device": 0, "stream": 7},
            },
            {
                "ph": "X",
                "cat": "cuda_sync",
                "name": "cudaStreamSynchronize",
                "ts": 200,
                "dur": 10,
                "args": {"cuda_sync_kind": "Stream Sync", "device": 0, "stream": 7},
            },
        ]
        self.assertEqual(_check_stream_sync_overlap(events), [])

    def test_stream_sync_overlap_fail(self):
        # [FUNCTION DOC] A kernel running 100 to 300 with a stream sync starting at 150 on the
        # same device and stream gives one overlap violation whose message says the kernel is
        # still running.
        events = [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm_kernel",
                "ts": 100,
                "dur": 200,
                "args": {"device": 0, "stream": 7},
            },
            {
                "ph": "X",
                "cat": "cuda_sync",
                "name": "cudaStreamSynchronize",
                "ts": 150,
                "dur": 10,
                "args": {"cuda_sync_kind": "Stream Sync", "device": 0, "stream": 7},
            },
        ]
        v = _check_stream_sync_overlap(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_stream_sync_overlap")
        self.assertIn("still running", v[0].message)

    # --- _check_stream_wait_corr_id_in_past ---

    def test_stream_wait_corr_id_in_past_pass(self):
        # [FUNCTION DOC] An event record at time 100 referenced by a stream wait at time 200
        # gives no in-the-past violations.
        events = [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaEventRecord",
                "ts": 100,
                "dur": 1,
                "args": {"External id": 5},
            },
            {
                "ph": "X",
                "name": "cudaStreamWaitEvent",
                "ts": 200,
                "dur": 1,
                "args": {
                    "cuda_sync_kind": "Stream Wait Event",
                    "wait_on_cuda_event_record_corr_id": 5,
                },
            },
        ]
        self.assertEqual(_check_stream_wait_corr_id_in_past(events), [])

    def test_stream_wait_corr_id_in_past_fail(self):
        # [FUNCTION DOC] An event record at time 300 referenced by a stream wait at time 200
        # gives one violation whose message says the record is in the future.
        events = [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaEventRecord",
                "ts": 300,
                "dur": 1,
                "args": {"External id": 5},
            },
            {
                "ph": "X",
                "name": "cudaStreamWaitEvent",
                "ts": 200,
                "dur": 1,
                "args": {
                    "cuda_sync_kind": "Stream Wait Event",
                    "wait_on_cuda_event_record_corr_id": 5,
                },
            },
        ]
        v = _check_stream_wait_corr_id_in_past(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_stream_wait_corr_id_in_past")
        self.assertIn("in the future", v[0].message)

    def test_stream_wait_corr_id_in_past_missing_record(self):
        # [FUNCTION DOC] A stream wait referencing correlation id 999 with no event record in the
        # list gives one violation whose message says there is no matching event record.
        events = [
            {
                "ph": "X",
                "name": "cudaStreamWaitEvent",
                "ts": 200,
                "dur": 1,
                "args": {
                    "cuda_sync_kind": "Stream Wait Event",
                    "wait_on_cuda_event_record_corr_id": 999,
                },
            },
        ]
        v = _check_stream_wait_corr_id_in_past(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_stream_wait_corr_id_in_past")
        self.assertIn("no matching cudaEventRecord", v[0].message)

    # --- _check_backward_seq_id_uniqueness ---

    def test_backward_seq_id_uniqueness_pass(self):
        # [FUNCTION DOC] Two backward evaluate-function events of the same op sharing sequence
        # number one give no uniqueness violations.
        events = [
            {
                "ph": "X",
                "name": "autograd::engine::evaluate_function: AddmmBackward0",
                "ts": 100,
                "dur": 10,
                "args": {"Sequence number": 1},
            },
            {
                "ph": "X",
                "name": "autograd::engine::evaluate_function: AddmmBackward0",
                "ts": 200,
                "dur": 10,
                "args": {"Sequence number": 1},
            },
        ]
        self.assertEqual(_check_backward_seq_id_uniqueness(events), [])

    def test_backward_seq_id_uniqueness_fail(self):
        # [FUNCTION DOC] Two backward evaluate-function events of different ops sharing sequence
        # number one give one violation whose message says the number is shared by several ops.
        events = [
            {
                "ph": "X",
                "name": "autograd::engine::evaluate_function: AddmmBackward0",
                "ts": 100,
                "dur": 10,
                "args": {"Sequence number": 1},
            },
            {
                "ph": "X",
                "name": "autograd::engine::evaluate_function: MulBackward0",
                "ts": 200,
                "dur": 10,
                "args": {"Sequence number": 1},
            },
        ]
        v = _check_backward_seq_id_uniqueness(events)
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].rule_name, "_check_backward_seq_id_uniqueness")
        self.assertIn("shared by", v[0].message)


# ---------------------------------------------------------------------------
# Accelerator-AGNOSTIC E2E tests (any accelerator)
# ---------------------------------------------------------------------------


# Class-level skip so setUpClass (which profiles a ResNet50 on GPU) is bypassed
# while all E2E tests are disabled. Re-enable alongside the per-test skips once
# kineto's CPU/GPU timestamp clock-skew issue is fixed.
@unittest.skip("E2E tests disabled pending kineto clock-skew fix; see per-test skips")
@skipIfTorchDynamo("profiler tests do not work with dynamo")
class TestTraceValidatorE2EAgnostic(_TraceValidatorE2EMixin, TestCase):
    # [CLASS DOC] End-to-end test class for hardware-independent rules (backward sequence id
    # uniqueness). Class-level state: a temporary trace directory path and a payload mapping,
    # both filled in setUpClass by profiling a training loop on the class's device type. Skipped
    # at class level pending a kineto clock-skew fix and also skipped under dynamo; instantiated
    # per device type except cpu.
    """E2E tests for validation rules that are not tied to any specific accelerator.

    Category: Accelerator-AGNOSTIC

    These tests profile a real workload on whatever accelerator is available
    and validate trace properties that are hardware-independent (autograd
    sequence IDs). Any backend that can run a ResNet50 training loop will
    exercise these tests.

    Instantiated per device type via ``instantiate_device_type_tests``.
    """

    _trace_dir: str = ""
    _payloads: dict = {}

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Creates a temporary directory, profiles the generic training payload on
        # the first device of the class's device type with the matching activity, and stores the
        # events under the training key. Not run while the class is skipped.
        super().setUpClass()
        device = torch.device(cls.device_type, 0)
        activity = _activity_for_device_type(cls.device_type)
        cls._trace_dir = tempfile.mkdtemp(prefix="profiler_e2e_trace_agnostic_")
        cls._payloads = {
            "training": _profile_training_payload(
                os.path.join(cls._trace_dir, "training.json"),
                device,
                activity,
            ),
        }

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Removes the temporary trace directory, ignoring errors if it exists,
        # then calls the base class teardown.
        if cls._trace_dir and os.path.isdir(cls._trace_dir):
            shutil.rmtree(cls._trace_dir, ignore_errors=True)
        super().tearDownClass()

    @unittest.skip(
        "kineto backward sequence ID uniqueness not yet verified in kineto integration testing"
    )
    def test_backward_seq_id_uniqueness(self, device):
        # [FUNCTION DOC] Would assert the backward sequence id rule finds no violations in the
        # training payload, showing formatted violations on failure. Currently skipped because
        # the kineto behaviour is not yet verified. The device argument is unused.
        v = _check_backward_seq_id_uniqueness(self._events("training"))
        self.assertEqual(len(v), 0, self._fmt(v))


instantiate_device_type_tests(
    TestTraceValidatorE2EAgnostic, globals(), except_for=("cpu",)
)


# ---------------------------------------------------------------------------
# Accelerator-SPECIFIC E2E tests (CUDA only)
# ---------------------------------------------------------------------------


# Class-level skip so setUpClass (which profiles a ResNet50 on GPU) is bypassed
# while all E2E tests are disabled. Re-enable alongside the per-test skips once
# kineto's CPU/GPU timestamp clock-skew issue is fixed.
@unittest.skip("E2E tests disabled pending kineto clock-skew fix; see per-test skips")
@unittest.skipUnless(TEST_CUDA, "CUDA not available")
@skipIfTorchDynamo("profiler tests do not work with dynamo")
@instantiate_parametrized_tests
class TestTraceValidatorE2ECUDA(_TraceValidatorE2EMixin, TestCase):
    # [CLASS DOC] End-to-end test class for CUDA-specific rules. Only runs when CUDA is
    # available and is currently skipped at class level pending a kineto clock-skew fix; also
    # skipped under dynamo. Class-level state: a temporary trace directory and a mapping of
    # payload names (resnet and complex) to events, filled in setUpClass. Each test is
    # parametrized over both payloads.
    """E2E tests for CUDA-specific profiler trace validation.

    Category: Accelerator-SPECIFIC (CUDA only)

    These tests exercise validators that parse CUDA-specific trace events
    (cuda_runtime, cuda_sync, cudaLaunchKernel, cudaEventRecord, NCCL).
    They use CUDA-specific profiling payloads including multi-stream
    and CUDA event synchronization.
    """

    _trace_dir: str = ""
    _payloads: dict = {}

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Creates a temporary directory and profiles the CUDA ResNet and the
        # complex multi-stream payloads, storing events under the resnet and complex keys. Not
        # run while the class is skipped.
        super().setUpClass()
        cls._trace_dir = tempfile.mkdtemp(prefix="profiler_e2e_trace_cuda_")
        cls._payloads = {
            "resnet": _profile_resnet_payload(
                os.path.join(cls._trace_dir, "resnet.json")
            ),
            "complex": _profile_complex_payload(
                os.path.join(cls._trace_dir, "complex.json")
            ),
        }

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Removes the temporary trace directory, ignoring errors if it exists,
        # then calls the base class teardown.
        if cls._trace_dir and os.path.isdir(cls._trace_dir):
            shutil.rmtree(cls._trace_dir, ignore_errors=True)
        super().tearDownClass()

    @unittest.skip(
        "kineto reports GPU kernel timestamps before cudaLaunchKernel due to clock skew"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_gpu_kernel_causality(self, payload):
        # [FUNCTION DOC] Asserts the kernel causality rule finds no violations in the given
        # payload. Currently skipped because kineto reports GPU timestamps before launches
        # (clock skew).
        v = _check_gpu_kernel_causality(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    @unittest.skip(
        "kineto does not populate wait_on_cuda_event_record_corr_id for stream wait events (returns -1)"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_wait_corr_id_populated(self, payload):
        # [FUNCTION DOC] Asserts the populated-id rule finds no violations in the given
        # payload. Currently skipped because kineto leaves the correlation id at negative one
        # for stream waits.
        v = _check_stream_wait_corr_id_populated(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    @unittest.skip(
        "kineto stream sync overlap detection not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_sync_overlap(self, payload):
        # [FUNCTION DOC] Asserts the stream sync overlap rule finds no violations in the given
        # payload. Currently skipped because kineto behaviour is not yet verified.
        v = _check_stream_sync_overlap(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    @unittest.skip(
        "kineto wait_on_cuda_event_record_corr_id temporal ordering not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_stream_wait_corr_id_in_past(self, payload):
        # [FUNCTION DOC] Asserts the in-the-past rule finds no violations in the given payload.
        # Currently skipped because temporal ordering is not yet verified in kineto.
        v = _check_stream_wait_corr_id_in_past(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))

    @unittest.skip(
        "kineto NCCL collective metadata not yet verified in kineto integration testing"
    )
    @parametrize("payload", ["resnet", "complex"])
    def test_nccl_metadata(self, payload):
        # [FUNCTION DOC] Asserts the NCCL metadata rule finds no violations in the given payload.
        # Currently skipped because NCCL metadata is not yet verified in kineto.
        v = _check_nccl_metadata(self._events(payload))
        self.assertEqual(len(v), 0, self._fmt(v))


if __name__ == "__main__":
    run_tests()
