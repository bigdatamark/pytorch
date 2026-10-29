# mypy: allow-untyped-defs
# [MODULE DOC]
# Utilities to benchmark torch.compile backends: benchmark_compile times compilation and running
# of one backend/mode, and bench_all tries eager plus every registered backend (and inductor
# modes) and returns a markdown table. Import-time side effects: records the current float32
# matmul precision, tries to import the optional tabulate package and, if missing, prints a
# message and skips defining the benchmarking functions (the exported names then do not exist).
from typing import Any, cast
from collections.abc import Callable

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer


__all__ = ["bench_all", "benchmark_compile"]


_warned_tensor_cores = False
_default_float_32_precision = torch.get_float32_matmul_precision()

try:

    from tabulate import tabulate

    HAS_TABULATE = True
except ModuleNotFoundError:
    HAS_TABULATE = False
    tabulate = None  # type: ignore[assignment]
    print("tabulate is not installed, please pip install tabulate to use this utility")

if HAS_TABULATE:
    def _enable_tensor_cores() -> None:
        # [FUNCTION DOC] HELPER: On a CUDA machine whose device capability is at least 8.0 and
        # where TF32 matmul is off, sets the float32 matmul precision to high and prints a
        # notice once per process. Mutates global torch settings.
        global _warned_tensor_cores

        if torch.cuda.is_available():
            if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
                if not _warned_tensor_cores:
                    print("Your GPU supports tensor cores")
                    print("we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
                    _warned_tensor_cores = True

    def _disable_tensor_cores() -> None:
        # [FUNCTION DOC] HELPER: Restores the float32 matmul precision recorded at import time.
        torch.set_float32_matmul_precision(_default_float_32_precision)

    def bench_loop(
        model: torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters: int = 5,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: Callable | None = None,
    ):
        # [FUNCTION DOC] HELPER: Times a model for num_iters iterations with a Timer and returns
        # the mean milliseconds per iteration rounded to two decimals. Uses a training step
        # statement (forward, loss, backward, optimizer step, zero grad) when both optimizer and
        # loss function are given, otherwise inference only.
        # KNOWN ISSUE: with an optimizer but no loss function the inference statement is used, so
        # no training happens; and the optimizer-only case in bench_all is passed no loss function
        # for the eager run.
        # Define the statement and setup for the benchmark
        if optimizer and loss_fn:
            # Training mode
            stmt = """
    output = model(sample_input)
    loss = loss_fn(output) if loss_fn else output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            """
        else:
            # Inference mode
            stmt = "model(sample_input)"

        # Create the Timer object
        timer = Timer(
            stmt=stmt,
            globals={"model": model, "sample_input": sample_input, "optimizer": optimizer, "loss_fn": loss_fn},
        )


        result = timer.timeit(number=num_iters)

        # Get the average time per iteration in milliseconds
        avg_time = result.mean * 1000
        return round(avg_time, 2)

    def benchmark_compile(
        model: torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters: int = 5,
        backend: str | None = None,
        mode: str | None = "default",
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn : torch.nn.Module | Callable | None = None,
    ):
        # [FUNCTION DOC] Benchmarks one compile backend and mode. With a backend, resets dynamo,
        # compiles the model with a frame-counting wrapper, times one iteration (compilation) and
        # then num_iters iterations, raises internally if no or more than one compilation
        # occurred, and on any failure prints the error and returns a pair of nones. Without a
        # backend runs the model eagerly. Returns (compilation ms or none, running ms or none),
        # each rounded to two decimals (a zero value becomes none).
        """
        Use this utility to benchmark torch.compile
        """
        if backend:
            try:
                torch._dynamo.reset()
                compile_counter_with_backend = CompileCounterWithBackend(backend)
                opt_model = torch.compile(model, backend=compile_counter_with_backend, mode=mode)

                # Compilation only happens after the first inference
                compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)

                running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

                if compile_counter_with_backend.frame_count == 0:
                    raise RuntimeError("No compilation occurred during benchmarking.")

                if compile_counter_with_backend.frame_count > 1:
                    raise RuntimeError("Recompilation occurred during benchmarking.")

            except Exception as e:
                print(e)
                print(f"Failed to compile {backend} with mode {mode}")
                return None, None
        else:
            opt_model = model
            compilation_time = None
            running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

        compilation_time = round(compilation_time, 2) if compilation_time else None
        running_time = round(running_time, 2) if running_time else None


        return compilation_time, running_time


    def bench_all(
        model : torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters : int = 5,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn : torch.nn.Module | Callable | None = None,
    ):
        # [FUNCTION DOC] Benchmarks eager mode and every registered dynamo backend (all inductor
        # modes plus the unset mode, skipping the default mode) and returns a github-style table
        # string with columns for training or inference, backend, mode, compilation time and
        # average running time. Enables tensor cores around inductor runs on CUDA. Prints
        # progress notices and failures; resets dynamo repeatedly.
        # KNOWN ISSUE: the eager run is not passed the loss function, and the non-inductor row
        # builds its compilation text from a formatted string, so a missing time prints as
        # "None ms" instead of a dash.
        """
        This is a simple utility that can be used to benchmark torch.compile
        In particular it ensures that your GPU is setup to use tensor cores if it supports its
        It also tries out all the main backends and prints a table of results so you can easily compare them all
        Many of the backends have their own optional dependencies so please pip install them separately

        You will get one table for inference and another for training
        If you'd like to leverage this utility for training make sure to pass in a torch.optim.Optimizer

        The important warnings are
        Your GPU supports tensor cores
        we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

        If a compilation fails for any reason including the dependency not being included
        then we will print Failed to compile {backend} with mode {mode}
        """
        field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time"]
        table = []


        eager_time = None
        torch._dynamo.reset()
        _, eager_time = benchmark_compile(model, sample_input, num_iters, None, None, optimizer)
        table.append(
            [("Training" if optimizer else "Inference"), "Eager", "-", "-", f"{eager_time} ms"]
        )

        for backend in torch._dynamo.list_backends():

            if backend == "inductor":
                mode_options = cast(list[str | None], list(torch._inductor.list_mode_options().keys())) + [None]
                for mode in mode_options:
                    if mode == "default":
                        continue
                    torch._dynamo.reset()
                    try:
                        if torch.cuda.is_available():
                            _enable_tensor_cores()
                        compilation_time, running_time = benchmark_compile(
                            model, sample_input, num_iters, backend, mode, optimizer, loss_fn)
                    finally:
                        if torch.cuda.is_available():
                            _disable_tensor_cores()
                    table.append([
                        ("Training" if optimizer else "Inference"),
                        backend if backend else "-",
                        mode if mode is not None else "-",
                        f"{compilation_time} ms " if compilation_time else "-",
                        f"{running_time} ms " if running_time else "-",
                    ])

            else:
                torch._dynamo.reset()
                compilation_time, running_time = benchmark_compile(
                    model, sample_input, num_iters, backend, None, optimizer, loss_fn)

                if running_time is not None:
                    table.append([
                        ("Training" if optimizer else "Inference"),
                        backend, "-",
                        f"{compilation_time} ms " or "-",
                        f"{running_time} ms ",
                    ])


        # pyrefly: ignore [not-callable]
        return tabulate(table, headers=field_names, tablefmt="github")
