# [MODULE DOC] Benchmark and profiling helpers embedded in Inductor-generated compiled modules. Provides per-kernel
# benchmarking (benchmark_all_kernels), profile report parsing (parse_profile_event_list), a torch profiler runner
# (perf_profile), an NCU runner (ncu_analyzer), a CUDA memory snapshot collector, and compiled_module_main, the
# command-line entry point called from a compiled module's main block. Import-time side effects: computes
# PROFILE_DIR and PROFILE_PATH under the system temp directory.
import argparse
import datetime
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet

from .runtime.benchmarking import benchmarker
from .runtime.runtime_utils import create_bandwidth_info_str, get_num_bytes


class BenchmarkCallableType(Protocol):
    # [CLASS DOC] HELPER: Protocol for the compiled module's benchmark function, called with times and repeat keyword
    # arguments and returning the measured wall time (the one-line __call__ stub declares only this signature).
    def __call__(self, times: int, repeat: int) -> float: ...


_kernel_category_choices = [
    "foreach",
    "persistent_reduction",
    "pointwise",
    "reduction",
    "split_scan",
    "template",
]


def get_kernel_category_by_source_code(src_code: str) -> str:
    # [FUNCTION DOC] Returns the kernel category (one of the known decorator names) whose triton heuristics decorator text
    # appears in the source code; returns unknown if zero or several categories match.
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
    choices = [
        ch for ch in _kernel_category_choices if f"@triton_heuristics.{ch}" in src_code
    ]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod: ModuleType) -> str:
    # [FUNCTION DOC] Returns the category of a compiled kernel module by checking which category names appear among the
    # module's globals (imported decorators); returns unknown unless exactly one matches.
    # KNOWN ISSUE: the docstring lists only three categories, but foreach, split_scan and template are also possible.
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Category can be one of:
    - pointwise
    - reduction
    - persistent_reduction

    Currently we simply decide the category depending on what decorator is imported
    by the kernel.
    """
    choices = [ch for ch in _kernel_category_choices if ch in kernel_mod.__dict__]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_triton_kernel(mod: ModuleType):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Returns the single autotuner object in the module whose name starts with the triton prefix.
    # Raises AssertionError unless exactly one such object exists.
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cand_list = [
        v
        for k, v in mod.__dict__.items()
        if k.startswith("triton_") and isinstance(v, CachingAutotuner)
    ]
    if len(cand_list) != 1:
        raise AssertionError(
            f"expected exactly one triton kernel, got {len(cand_list)}"
        )
    return cand_list[0]


def benchmark_all_kernels(
    benchmark_name: str, benchmark_all_configs: dict[Any, Any] | None
) -> None:
    # [FUNCTION DOC] Benchmarks every kernel module cached in PyCodeCache that has get_args and call, printing results.
    # Per kernel: gets its triton kernel and arguments, computes data size in GB (from metadata or argument sizes), then
    # either benchmarks all autotune configs (if benchmark_all_configs is truthy; asserts the module supports it) or
    # times the selected config (asserts exactly one launcher). Prints bandwidth/register info lines, or a hint message if
    # no kernel was found. Experimental; relevant only with config.benchmark_kernel.
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    nfound = 0
    for kernel_mod in PyCodeCache.modules:
        kernel_key = kernel_mod.key
        if not hasattr(kernel_mod, "get_args") or not hasattr(kernel_mod, "call"):
            continue

        triton_kernel = get_triton_kernel(kernel_mod)
        device_type = triton_kernel.device_props.type
        kernel_category = get_kernel_category(kernel_mod)
        args = kernel_mod.get_args()
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in triton_kernel.fn.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        num_gb = triton_kernel.inductor_meta.get("kernel_num_gb", None)
        if num_gb is None:
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        def get_info_str(
            ms: float,
            n_regs: Any | None,
            n_spills: Any | None,
            shared: Any | None,
            prefix: str = "",
        ) -> str:
            # [FUNCTION DOC] HELPER: Builds the printable info line for a timing: bandwidth from kernel size and time, with
            # register, spill and shared memory details appended if all are known.
            if not any(x is None for x in [n_regs, n_spills, shared]):
                kernel_detail_str = (
                    f"  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem"
                )
            else:
                kernel_detail_str = ""

            gb_per_s = num_gb / (ms / 1e3)
            return create_bandwidth_info_str(
                ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str
            )

        kernel_desc = (
            f"{benchmark_name:20} {kernel_category[:3].upper()} {kernel_key[:10]}"
        )
        if benchmark_all_configs:
            if not hasattr(kernel_mod, "benchmark_all_configs"):
                raise AssertionError("kernel_mod is missing benchmark_all_configs")
            bench_result = kernel_mod.benchmark_all_configs(args)
            print(kernel_desc)
            for launcher, ms in bench_result.items():
                print(
                    f"  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}"
                )
        else:
            ms = benchmarker.benchmark(
                lambda: kernel_mod.call(args),
                device=device_type,
                rep=40,
            )
            if len(triton_kernel.launchers) != 1:
                raise AssertionError("Autotuner should have selected the best config")
            launcher = triton_kernel.launchers[0]
            print(
                get_info_str(
                    ms,
                    launcher.n_regs,
                    launcher.n_spills,
                    launcher.shared,
                    prefix=f"{kernel_desc} ",
                )
            )

        nfound += 1
    if nfound == 0:
        print(
            "No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True"
        )


@dataclass
class ProfileEvent:
    # [CLASS DOC] Dataclass for one aggregated profile row: category, kernel key, self device time in ms, and average
    # count per run (float).
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    # [FUNCTION DOC] Aggregates profiler events (non-CPU only) by category inferred from the kernel name prefix
    # (pointwise, reduction, persistent reduction, unknown triton, unknown), averaged over nruns, and prints per-category
    # tables, device busy percent, total wall time, and a summary line for later tabulation. Raises AssertionError for
    # legacy profiler events. If device_name is empty, category tables are skipped and no device is reported.
    """
    Parse and generate a report for an event_list.
    """

    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        # [FUNCTION DOC] HELPER: Converts an event's total self device time from microseconds to milliseconds per run.
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        # [FUNCTION DOC] HELPER: Records a ProfileEvent (key, per-run time, per-run count) under the given category.
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        if ev.is_legacy:
            raise AssertionError("Don't support the legacy profiler")
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        # [FUNCTION DOC] HELPER: Prints a table of the category's events sorted by descending time with percent of wall time
        # and a total row, and returns the total time in ms (0.0 and no output if no device).
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        # [FUNCTION DOC] HELPER: Prints all category reports in a fixed order, the busy percent, wall time, and the
        # tabulate summary line. Raises AssertionError if an unexpected category was recorded.
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        if not OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)):
            raise AssertionError(f"{list(all_events.keys())}")

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"


def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    # [FUNCTION DOC] Runs the benchmark function under the torch profiler, exports a Chrome trace to PROFILE_PATH, prints
    # the trace location and a top-10 table of averaged events, then prints the parsed per-category report. Side effects:
    # writes a file, prints.
    with torch.profiler.profile(record_shapes=True) as p:
        benchmark_compiled_module_fn(times=times, repeat=repeat)

    path = PROFILE_PATH
    p.export_chrome_trace(path)
    print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
    print(f"Chrome trace for the profile is written to {path}")
    event_list = p.key_averages(group_by_input_shape=True)
    print(event_list.table(sort_by="self_device_time_total", row_limit=10))
    parse_profile_event_list(
        benchmark_name, event_list, wall_time_ms, times * repeat, p.use_device or ""
    )


def ncu_analyzer(
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
    args: argparse.Namespace,
) -> None:
    # [FUNCTION DOC] Runs NVIDIA Nsight Compute on a subprocess that imports the benchmark module and runs it once.
    # Uses args.ncu_kernel_regex to filter kernels and args.ncu_metrics (else the full set) to select metrics, writing the
    # report to a timestamped file in the temp directory. Prints the output path, or the error if the ncu command fails
    # (the failure is not re-raised).
    import inspect
    import os
    import subprocess

    kernel_regex = args.ncu_kernel_regex
    metrics = args.ncu_metrics

    module_file = inspect.getfile(benchmark_compiled_module_fn)
    module_dir = os.path.dirname(module_file)
    module_name = os.path.splitext(os.path.basename(module_file))[0]

    ncu_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ncu_output = os.path.join(ncu_dir, f"ncu_output_{timestamp}.ncu-rep")
    python_cmd = (
        f"""import sys; sys.path.insert(0, '{module_dir}'); """
        f"""from {module_name} import benchmark_compiled_module; """
        """benchmark_compiled_module(times=1, repeat=1)"""
    )

    ncu_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--kernel-name-base",
        "function",
        "--print-units",
        "base",
        "--import-source",
        "yes",
        "--force-overwrite",
        "--export",
        ncu_output,
    ]

    if kernel_regex:
        ncu_cmd.extend(["--kernel-name", f"regex:{kernel_regex}"])

    if metrics:
        ncu_cmd.extend(["--metrics", metrics])
    else:
        ncu_cmd.extend(["--set", "full"])

    ncu_cmd.extend(
        [
            "python",
            "-c",
            python_cmd,
        ]
    )

    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"\nNCU profiling results for benchmark {benchmark_name}:")
        print(f"NCU report has been written to {ncu_output}")

    except subprocess.CalledProcessError as e:
        print(f"NCU profiling failed with error: {e}")
        return


def collect_memory_snapshot(
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    # [FUNCTION DOC] Records CUDA memory history while running the benchmark 10 times, dumps a snapshot to a pickle file
    # in the temp directory, stops recording, and prints the path. Raises AssertionError if CUDA is unavailable.
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available")

    torch.cuda.memory._record_memory_history(max_entries=100000)
    benchmark_compiled_module_fn(times=10, repeat=1)  # run 10 times
    snapshot_path = f"{tempfile.gettempdir()}/memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"The collect memory snapshot has been written to {snapshot_path}")


# With AOTAutograd cache, we directly call the compiled module. So prevent
# Dynamo from reentering
@torch.compiler.disable  # type: ignore[misc]
def compiled_module_main(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None:
    # [FUNCTION DOC] Entry point for a compiled module's main block (excluded from Dynamo). Parses command-line options
    # (kernel benchmarking, all configs, profile, CUDA memory snapshot, NCU options, times, repeat). If kernel benchmarking
    # is requested, runs it. Otherwise runs the whole-module benchmark, reports peak GPU memory if CUDA exists, and
    # optionally collects a memory snapshot, profiles, and runs NCU analysis.
    """
    This is the function called in __main__ block of a compiled module.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-kernels",
        "-k",
        action="store_true",
        help="Whether to benchmark each individual kernels",
    )
    parser.add_argument(
        "--benchmark-all-configs",
        "-c",
        action="store_true",
        help="Whether to benchmark each individual config for a kernel",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the compiled module",
    )
    parser.add_argument(
        "--cuda-memory-snapshot",
        action="store_true",
        help="""
            Whether to collect CUDA memory snapshot. Refer to
            "https://pytorch.org/blog/understanding-gpu-memory-1/
            for details about how to visualize the collected snapshot
        """,
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Whether to run ncu analysis",
    )
    parser.add_argument(
        "--ncu-kernel-regex",
        type=str,
        default=None,
        help=(
            "Filter kernels profiled by NCU using a regex (e.g., '^triton_.*'). "
            "Maps to '--kernel-name regex:<regex>'. "
            "If None, NCU will profile all kernels."
        ),
    )
    parser.add_argument(
        "--ncu-metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated list of NCU metrics to collect (e.g., 'dram__bytes.sum.per_second'). "
            "If None, NCU will use '--set full'."
        ),
    )
    parser.add_argument(
        "--times",
        type=int,
        default=10,
        help="Number of times to run each benchmark iteration",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of repetitions of each benchmark run",
    )

    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = args.times
        repeat = args.repeat

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) * 1000

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak GPU memory usage {peak_mem / 1e6:.3f} MB")

        if torch.cuda.is_available() and args.cuda_memory_snapshot:
            collect_memory_snapshot(benchmark_compiled_module_fn)

        if args.profile:
            perf_profile(
                wall_time_ms,
                times,
                repeat,
                benchmark_name,
                benchmark_compiled_module_fn,
            )
        if args.ncu:
            ncu_analyzer(
                benchmark_name,
                benchmark_compiled_module_fn,
                args=args,
            )
