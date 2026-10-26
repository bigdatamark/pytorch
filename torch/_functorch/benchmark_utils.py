# [MODULE DOC] Utilities to benchmark GPU utilization of a callable using the PyTorch profiler.
# Runs a function under an optional optimization context, dumps a chrome trace, and parses the
# trace to compute the fraction of wall time the GPU was busy and the fraction spent in
# matmul/convolution kernels. Main entry points: dump_chrome_trace, compute_utilization,
# benchmark_utilization. Keeps module-level state: gpu_pids and a replaceable synchronize hook.
# No import-time side effects beyond defining these globals.
from __future__ import annotations

import contextlib
import json
import operator
import os
import time
from contextlib import AbstractContextManager
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVar

import torch
from torch.profiler import profile, ProfilerActivity


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_R = TypeVar("_R")


def synchronize() -> None:
    # [FUNCTION DOC] HELPER: Default no-op device synchronization hook. dump_chrome_trace rebinds
    # this module global to torch.accelerator.synchronize when an accelerator is available.
    pass


def dump_chrome_trace(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_filename: str,
    optimize_ctx: AbstractContextManager[Any],
    activities: Sequence[ProfilerActivity],
    num_runs: int = 1,
    devices: list[str] | None = None,
    kwargs_for_f: dict[str, Any] | None = None,
    kwargs_for_profiler: dict[str, Any] | None = None,
) -> float:
    # [FUNCTION DOC] Times and profiles repeated calls of f on input_ inside optimize_ctx.
    # Steps: pick the synchronize hook (accelerator sync unless devices is exactly cpu), run 5
    # warmup calls, time num_runs calls with a fixed seed, then re-run num_runs calls under the
    # profiler (kwargs_for_profiler forwarded) and export the chrome trace to trace_filename.
    # Returns the un-profiled elapsed seconds for the timed runs. Side effects: rebinds the
    # global synchronize, seeds the RNG, writes the trace file.
    # KNOWN ISSUE: the docstring's "total runtime" excludes warmup; devices defaults to cuda.
    """
    Output the chrome trace of running f(input_, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename
    """

    if devices is None:
        devices = ["cuda"]

    global synchronize
    if devices != ["cpu"] and torch.accelerator.is_available():
        synchronize = torch.accelerator.synchronize

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5):  # warmup runs
            f(input_, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input_, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0

    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input_, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)

    return timing


def get_chrome_trace_events(filename: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Loads a chrome trace JSON file and returns its list of trace events
    # (the traceEvents entry). Raises on missing file, bad JSON, or missing key.
    with open(filename) as f:
        data = json.load(f)
    events = data["traceEvents"]
    return events


def is_gpu_compute_event(event: dict[str, Any]) -> bool:
    # [FUNCTION DOC] Returns True if the event has a pid that is in the module global gpu_pids
    # and has phase "X" (a complete duration event). Depends on compute_utilization having
    # populated gpu_pids first.
    global gpu_pids
    return (
        "pid" in event
        and event["pid"] in gpu_pids
        and "ph" in event
        and event["ph"] == "X"
    )


def get_sorted_gpu_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Filters events to GPU compute events and returns them sorted by start
    # timestamp. Uses global gpu_pids via is_gpu_compute_event.
    sorted_gpu_events: list[dict[str, Any]] = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=operator.itemgetter("ts"))


def get_duration(sorted_gpu_events: list[dict[str, Any]]) -> int:
    # [FUNCTION DOC] Computes total busy time of events sorted by start time, counting
    # overlapping intervals only once (union of intervals). Returns 0 for an empty list.
    # Each event needs ts (start) and dur keys; result is in trace time units (microseconds).
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


def get_sorted_gpu_mm_conv_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Returns the time-sorted GPU compute events whose kernel name looks like a
    # matmul or convolution (name contains gemm, conv, cutlass, or wgrad).
    def is_mm_conv_event(event: dict[str, Any]) -> bool:
        # [FUNCTION DOC] HELPER: True if the event has a name containing one of the matmul or
        # convolution kernel substrings.
        return "name" in event and (
            "gemm" in event["name"]
            or "conv" in event["name"]
            or "cutlass" in event["name"]
            or "wgrad" in event["name"]
        )

    gpu_events = get_sorted_gpu_events(events)
    sorted_events: list[dict[str, Any]] = []
    for event in gpu_events:
        if not is_mm_conv_event(event):
            continue
        sorted_events.append(event)
    return sorted_events


gpu_pids: list[Any] = []


def compute_utilization(filename: str, total_length: float) -> tuple[float, float]:
    # [FUNCTION DOC] Parses a chrome trace file and returns (GPU busy fraction, matmul/conv busy
    # fraction) relative to total_length seconds (converted to microseconds). GPU process ids
    # are found from process_labels events whose labels mention GPU; the global gpu_pids is
    # reset and repopulated as a side effect. Raises if a process_labels event lacks args.
    # KNOWN ISSUE: the docstring says "percent" but values are fractions, and can exceed 1.
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
    events = get_chrome_trace_events(filename)

    # get pids of GPU events
    global gpu_pids
    gpu_pids = []
    for event in events:
        if "name" not in event:
            continue
        if event["name"] == "process_labels" and "GPU" in event["args"]["labels"]:
            gpu_pids.append(event["pid"])

    total_length = total_length * 1e6
    sorted_gpu_events = get_sorted_gpu_events(events)
    utilization = get_duration(sorted_gpu_events) / total_length

    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

    return utilization, mm_conv_utilization


def benchmark_utilization(
    f: Callable[[tuple[Any, ...]], _R],
    input_: tuple[Any, ...],
    trace_folder: str,
    optimize_ctx: AbstractContextManager[Any] | None = None,
    trace_file_name: str = "tmp_chrome_trace",
    num_runs: int = 1,
) -> tuple[float, float]:
    # [FUNCTION DOC] End-to-end helper: creates trace_folder if missing (prints a message),
    # defaults optimize_ctx to a null context, dumps a CUDA-profiled chrome trace to
    # trace_folder/trace_file_name.json via dump_chrome_trace, then returns the
    # (GPU utilization, matmul/conv utilization) from compute_utilization. Writes files.
    """
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input_, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()


    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(
        f, a, "tmp", trace_file_name="tmp_chrome_trace"
    )
    ```

    Args:
        f: function to benchmark

        input_: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    """
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print("create folder " + trace_folder)

    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()

    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + ".json")
    total_length = dump_chrome_trace(
        f,
        input_,
        chrome_trace_file_name,
        optimize_ctx,
        [ProfilerActivity.CUDA],
        num_runs=num_runs,
        devices=["cuda"],
    )
    utilization, mm_conv_utilization = compute_utilization(
        chrome_trace_file_name, total_length
    )

    return utilization, mm_conv_utilization
