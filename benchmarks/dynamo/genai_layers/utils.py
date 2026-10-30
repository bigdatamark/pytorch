# [MODULE DOC] Shared infrastructure for the genai layer kernel benchmarks: a GPU timing helper,
# a Performance record, the BenchmarkKernel base class (backend dispatch, input cloning,
# accuracy checking, per-shape benchmarking, geomean speedup report) and plotting helpers that
# draw memory bandwidth per shape per backend. Imports matplotlib, scipy and the inductor
# benchmarker at import time; plotting writes files into a pics directory.
import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from scipy.stats import gmean

import torch
from torch._inductor.runtime.benchmarking import benchmarker


def benchmark_kernel_in_milliseconds(func: Callable, *args, **kwargs) -> float:
    # [FUNCTION DOC] Time a callable on the GPU and return the latency in milliseconds. Calls
    # it 5 times as warmup, then measures with the inductor GPU benchmarker while the compiler
    # stance is set to fail on any recompile (so a recompile raises instead of being timed).
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    with torch.compiler.set_stance("fail_on_recompile"):
        return benchmarker.benchmark_gpu(lambda: func(*args, **kwargs))


@dataclass
class Performance:
    # [CLASS DOC] Dataclass holding one benchmark measurement: setting label (usually the input
    # shape), latency in milliseconds, memory traffic in bytes, memory bandwidth in GB/s
    # (computed on construction, overriding any given value) and a compute intensity field
    # that defaults to zero and is never computed here.
    # Benchmark setting usually the shape of the input tensor
    setting: str

    # Latency in milliseconds
    latency: float

    # Number of  memory access in bytes
    memory_bytes: float

    # Memory bandwidth in GB/s
    memory_bandwidth: float = 0.0

    # Compute intensity in FLOPs/byte
    compute_intensity: float = 0.0

    def __post_init__(self):
        # [FUNCTION DOC] Derive memory bandwidth in GB/s from memory bytes and latency in ms.
        # Divides by the latency, so a zero latency raises ZeroDivisionError.
        self.memory_bandwidth = self.memory_bytes / (self.latency / 1000) / 1e9

    def __str__(self):
        # [FUNCTION DOC] Return a one-line summary with the setting, latency and bandwidth.
        return f"setting: {self.setting}, latency: {self.latency} ms, memory bandwidth: {self.memory_bandwidth} GB/s"


class BenchmarkKernel:
    # [CLASS DOC] Base class for a kernel benchmark. Subclasses fill in the list of available
    # backend names and provide shapes, a memory traffic model, one builder method per backend
    # (each returns a zero-argument callable to time) and a benchmark loop over shapes.
    # State: script_args (parsed command line), name (the class name), available_backends,
    # compile_mode (taken from script_args) and profiling_results (mapping from backend name to
    # a list of Performance records, created on demand).
    def __init__(self, script_args):
        # [FUNCTION DOC] Store the script arguments, derive the kernel name from the class name,
        # start with no available backends, and read the compile mode from the arguments.
        self.script_args = script_args
        self.name = self.__class__.__name__
        self.available_backends: list[str] = []
        self.compile_mode: str = script_args.compile_mode

        # mapping from backend to list of performance results
        self.profiling_results: defaultdict[str, list[Performance]] = defaultdict(list)

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Abstract: return the bytes of memory traffic for the given inputs.
        # Subclasses must override; raises NotImplementedError here.
        # Get the necessary memory access in bytes for the kernelßß
        raise NotImplementedError

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Abstract: return the input shapes to benchmark. Raises
        # NotImplementedError here.
        # Get a list of input shapes to benchmark the kernel
        raise NotImplementedError

    def eager(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the eager implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def compiled(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the compiled implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def helion(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the helion implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def quack(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the quack implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def liger(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the liger implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def triton(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Abstract backend builder for the triton implementation. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def benchmark(self):
        # [FUNCTION DOC] Abstract: run the whole benchmark over all shapes. Raises
        # NotImplementedError here.
        raise NotImplementedError

    def clone_inputs(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Return a pair of cloned inputs: the positional arguments as a list of
        # detached copies keeping each tensor's requires-grad flag, and the keyword arguments as
        # a mapping with tensors copied the same way and other values passed through. If the
        # keyword arguments are empty or None they are returned unchanged. Positional entries
        # must all be tensors.
        args_ref = [
            arg.clone().detach().requires_grad_(arg.requires_grad) for arg in args
        ]

        kwargs_ref = (
            {
                k: (
                    v.clone().detach().requires_grad_(v.requires_grad)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in kwargs.items()
            }
            if kwargs
            else kwargs
        )

        return args_ref, kwargs_ref

    def check_accuracy(self, args, kwargs) -> None:
        # [FUNCTION DOC] Run every available backend once on cloned inputs and compare to the
        # eager result. If a compiled backend exists and custom compile options were given,
        # also runs a recompiled version under patched inductor config, stored under the custom
        # name. Compares with assert_close (optional tolerance used for both atol and rtol), and
        # also compares gradients of result elements that require grad. Prints a success or
        # failure line per backend; on failure exits the process with status 1 if the exit-on
        # failure option is set.
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)()

        if (
            "compiled" in self.available_backends
            and self.script_args.custom_compile_options
        ):
            torch._dynamo.reset()  # cause recompile
            with torch._inductor.config.patch(self.script_args.custom_compile_options):
                args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
                res[self.script_args.custom_compile_name] = self.compiled(
                    args_ref, kwargs_ref
                )()

        gold = res["eager"]

        tol = {}
        if self.script_args.tolerance:
            tol = {
                "atol": self.script_args.tolerance,
                "rtol": self.script_args.tolerance,
            }
        for backend in res:
            if backend == "eager":
                continue
            try:
                torch.testing.assert_close(res[backend], gold, **tol)
                for t, gold_t in zip(res[backend], gold):
                    if t.requires_grad:
                        torch.testing.assert_close(t.grad, gold_t.grad, **tol)
                print(
                    f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel"
                )
            except Exception as e:
                print(
                    f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel. Error {e}"
                )
                if self.script_args.exit_on_accuracy_failure:
                    print("Exit right away since --exit-on-accuracy-failure is set")
                    sys.exit(1)

    def benchmark_single_shape_for_backend(
        self, backend, args, kwargs, setting, fn=None
    ) -> bool:
        # [FUNCTION DOC] Time one backend on the given inputs. Uses the given builder or else the
        # method named by the backend, builds the callable on cloned inputs and times it. If
        # that raises, prints the failure, removes the backend from the available list and
        # returns False. Otherwise records a Performance (with the memory traffic model) under
        # the backend name, prints it and returns True.
        if fn is None:
            fn = getattr(self, backend)
        args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
        try:
            avg_time = benchmark_kernel_in_milliseconds(fn(args_ref, kwargs_ref))
        except Exception as e:
            print(
                f"Failed to run {backend} backend on {self.name} kernel for {setting} due to {e}"
            )
            self.available_backends.remove(backend)  # noqa: B909
            return False
        mem_bytes = self.get_memory_bytes(args_ref, kwargs_ref)
        perf = Performance(setting, avg_time, mem_bytes)
        print(f"{self.name} kernel on {backend} backend. {perf}")
        self.profiling_results[backend].append(perf)
        return True

    def benchmark_single_shape(
        self, args, kwargs=None, should_check_accuracy=True, setting: str = ""
    ):
        # [FUNCTION DOC] Benchmark all available backends on one input, then optionally the
        # compiled backend again with custom inductor options (after resetting dynamo); if that
        # run fails the custom options are cleared so it is not retried. Then runs the accuracy
        # check unless disabled. The setting string labels the results.
        # KNOWN ISSUE: a failing backend is removed from the available list while it is being
        # iterated, so the backend after it is skipped for that shape.
        for backend in self.available_backends:
            self.benchmark_single_shape_for_backend(backend, args, kwargs, setting)
        if (
            "compiled" in self.available_backends
            and self.script_args.custom_compile_options
        ):
            torch._dynamo.reset()  # cause recompile
            with torch._inductor.config.patch(self.script_args.custom_compile_options):
                status = self.benchmark_single_shape_for_backend(
                    self.script_args.custom_compile_name,
                    args,
                    kwargs,
                    setting,
                    fn=self.compiled,
                )
            if not status:
                self.script_args.custom_compile_options = (
                    None  # once fail, don't run again
                )

        if should_check_accuracy:
            self.check_accuracy(args, kwargs)

    def visualize(self) -> None:
        # [FUNCTION DOC] Plot memory bandwidth of all recorded backends, titled with the kernel
        # name and the name of GPU 0, and save it as an image named after the kernel. Returns
        # None.
        device_name = torch.cuda.get_device_name(0)
        visualize_comparison(
            self.profiling_results,
            title=f"{self.name} ({device_name})",
            output_path=f"{self.name}_bench",
        )
        return

    def report_geomean_speedup(self) -> None:
        # [FUNCTION DOC] Print, for each non-eager backend, the geometric mean of the ratio of
        # eager latency to backend latency across its recorded settings (zero for zero latency),
        # plus the number of data points. Prints the eager data point count first.
        # Raises KeyError if a backend has a setting that eager did not record.
        print(f"Geomean speedup for benchmark {self.name}")
        eager_result = {
            result.setting: result for result in self.profiling_results["eager"]
        }
        print(f"  eager {len(eager_result)} data points")
        for backend, backend_result in self.profiling_results.items():
            if backend == "eager":
                continue
            speeduplist = []
            for result in backend_result:
                eager_latency = eager_result[result.setting].latency
                backend_latency = result.latency
                speeduplist.append(
                    eager_latency / backend_latency if backend_latency != 0 else 0.0
                )

            if len(speeduplist) > 0:
                print(
                    f"  {backend} {len(speeduplist)} data points, {gmean(speeduplist):.2f}x speedup"
                )


def get_backend_colors() -> dict[str, str]:
    # [FUNCTION DOC] Return a mapping from backend name to a hex color used in plots, with a
    # default black entry for unknown backends.
    """Get consistent color scheme for different backends."""
    return {
        "eager": "#1f77b4",  # blue
        "compiled": "#ff7f0e",  # orange
        "quack": "#2ca02c",  # green
        "liger": "#d62728",  # red
        "helion": "#9467bd",  # purple
        "triton": "#8c564b",  # brown
        "cutlass": "#e377c2",  # pink
        "flash_attn": "#7f7f7f",  # gray
        "default": "#000000",  # black
    }


def visualize_comparison(
    profiling_results: dict[str, list[Performance]],
    title: str | None = None,
    output_path: str | None = None,
) -> None:
    # [FUNCTION DOC] Draw one line plot of memory bandwidth versus input setting for each
    # backend in the results. The x axis ticks come from the settings recorded for eager;
    # backends are plotted only at settings they share with eager, and skipped if none.
    # If an output path is given, creates a pics directory if needed, saves a PNG there and
    # prints the path. Always closes the figure. The optional title defaults to a generic one.
    # Looks up the eager entry directly, so a plain mapping without it raises KeyError.
    """
    Create a single memory_bandwidth comparison plot from profiling results.

    Args:
        profiling_results: Dict mapping backend names to lists of Performance objects
        output_path: Path to save the plot (optional)
    """
    # Get backend colors
    backend_colors = get_backend_colors()

    # Extract settings from eager backend which runs all settings
    all_settings = []
    for perf in profiling_results["eager"]:
        all_settings.append(perf.setting)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for backend in profiling_results:
        backend_perfs = profiling_results[backend]
        perf_dict = {perf.setting: perf for perf in backend_perfs}

        x_vals = []
        y_vals = []
        for i, setting in enumerate(all_settings):
            if setting in perf_dict:
                x_vals.append(i)
                y_vals.append(perf_dict[setting].memory_bandwidth)

        if x_vals:  # Only plot if we have data
            color = backend_colors.get(backend, backend_colors["default"])
            ax.plot(
                x_vals,
                y_vals,
                "o-",
                label=backend,
                color=color,
                linewidth=2,
                markersize=8,
                alpha=0.8,
            )

    # Configure the plot
    ax.set_title(title or "Memory Bandwidth Comparison", fontsize=16)
    ax.set_xlabel("Shape", fontsize=12)
    ax.set_ylabel("memory bandwidth (GB/s)", fontsize=12)
    ax.set_xticks(range(len(all_settings)))
    ax.set_xticklabels(
        [
            s.replace("shape: ", "").replace("[", "").replace("]", "")
            for s in all_settings
        ],
        rotation=45,
        ha="right",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot if output path is provided
    if output_path:
        # Save as PNG
        os.makedirs("pics", exist_ok=True)
        full_path = os.path.join("pics", output_path + ".png")
        plt.savefig(full_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to {full_path}")

    plt.close()
