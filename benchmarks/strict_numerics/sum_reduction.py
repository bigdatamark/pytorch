from __future__ import annotations
# [MODULE DOC] Command-line benchmark of row-sum reduction bandwidth on cuda for three
# providers: plain eager, eager with the inner-tree summation environment switch enabled, and
# inductor-compiled. For a grid of row counts and column counts it measures GB/s (optionally
# with cuda graph replay over a rotating set of input buffers), prints one summary line per row
# count, and saves a plot of bandwidth against column count. Requires triton and a cuda device.
# Importing only defines constants and functions; the data type is float32.

import argparse
import os
from itertools import cycle
from typing import TYPE_CHECKING
from unittest.mock import patch

from triton.testing import do_bench, do_bench_cudagraph

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache


if TYPE_CHECKING:
    from collections.abc import Callable


PROVIDERS = ["eager_aten", "eager_inner_tree", "inductor_default"]
M_VALUES = [512, 1024, 2048, 4096, 8192, 16384]
N_VALUES = [2**exponent for exponent in range(9, 18)]
MEMORY_BUDGET = 40 * (1 << 30)
CUDA_GRAPH_WORKING_SET_BYTES = 256 * (1 << 20)
COLORS = {
    "eager_aten": "C0",
    "inductor_default": "C1",
    "eager_inner_tree": "C2",
}
DTYPE = torch.float32


def _gbps(m: int, n: int, milliseconds: float) -> float:
    # [FUNCTION DOC] HELPER: Converts a time in milliseconds for reading an m by n float32
    # matrix once into effective bandwidth in decimal GB/s.
    return m * n * DTYPE.itemsize / (milliseconds * 1e-3) / 1e9


def _sum_rows(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: The benchmarked operation: sums the input over its second dimension
    # and returns the row sums.
    return tensor.sum(1)


def _circular_runner(
    function: Callable[[torch.Tensor], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
) -> Callable[[], torch.Tensor]:
    # [FUNCTION DOC] HELPER: Returns a zero-argument callable that, on each call, applies the
    # function to the next tensor of the given tuple, cycling through the tuple endlessly. Used
    # to rotate inputs so successive launches do not reuse cached data.
    tensor_iterator = cycle(tensors)
    return lambda: function(next(tensor_iterator))


def measure(m: int, n: int, provider: str, cudagraph: bool) -> float:
    # [FUNCTION DOC] Measures the achieved bandwidth in GB/s of the row sum for an m by n
    # float32 cuda tensor with the named provider. Returns NaN without running if the tensor
    # exceeds the memory budget constant. Seeds the random generator, creates the input (in
    # cuda graph mode, enough copies to cover the working-set byte constant, otherwise one),
    # and picks the cuda graph or the plain timing function. The summation environment variable
    # is patched to enable the inner-tree path only for the inner-tree provider (and to disable
    # it otherwise) for the duration of the measurement. Eager providers run once as warmup
    # then are timed; any other provider resets dynamo, uses a fresh inductor cache, compiles
    # the row sum, warms up and times it. Always frees the buffers and empties the cuda cache
    # before returning.
    tensor_bytes = m * n * DTYPE.itemsize
    if tensor_bytes > MEMORY_BUDGET:
        return float("nan")

    torch.manual_seed(0)
    input_count = (
        max(
            1,
            (CUDA_GRAPH_WORKING_SET_BYTES + tensor_bytes - 1) // tensor_bytes,
        )
        if cudagraph
        else 1
    )
    input_buffer = torch.randn(input_count, m, n, device="cuda", dtype=DTYPE)
    tensors = input_buffer.unbind()
    runner = _circular_runner(_sum_rows, tensors)
    benchmark = do_bench_cudagraph if cudagraph else do_bench
    inner_tree = provider == "eager_inner_tree"
    try:
        with patch.dict(
            os.environ,
            {"PYTORCH_SUM_INNER_TREE": "1" if inner_tree else "0"},
        ):
            if provider.startswith("eager"):
                runner()
                milliseconds = benchmark(runner)
            else:
                torch._dynamo.reset()
                with fresh_inductor_cache():
                    compiled = torch.compile(_sum_rows)
                    runner = _circular_runner(compiled, tensors)
                    runner()
                    milliseconds = benchmark(runner)
        return _gbps(m, n, milliseconds)
    finally:
        del runner
        del tensors
        del input_buffer
        torch.cuda.empty_cache()


def _assert_inner_tree_route() -> None:
    # [FUNCTION DOC] HELPER: Sanity check that the inner-tree switch changes the numerical
    # result: sums a random 1024 by 300 cuda tensor with the switch on and off, and fails with
    # an assertion error if the two results are exactly equal (meaning the inner-tree path was
    # not taken). Frees the tensor and empties the cuda cache afterwards.
    tensor = torch.randn(1024, 300, device="cuda", dtype=DTYPE)
    try:
        with patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "1"}):
            inner_tree = torch.sum(tensor, 1)
        with patch.dict(os.environ, {"PYTORCH_SUM_INNER_TREE": "0"}):
            aten = torch.sum(tensor, 1)
        assert not torch.equal(inner_tree, aten), (  # noqa: S101
            "eager_inner_tree did not route to CuTeDSL; refusing to label ATen "
            "results as inner-tree results"
        )
    finally:
        del tensor
        torch.cuda.empty_cache()


def _plot(data: dict[str, dict[int, list[float]]], cudagraph: bool) -> str:
    # [FUNCTION DOC] HELPER: Draws a grid of line plots, one panel for each row count in the
    # module list, showing bandwidth per provider against column count on a base-2 log axis,
    # with legend, grid and a title that mentions whether cuda graph mode was used. Selects the
    # non-interactive plotting backend, saves a png file in the working directory whose name
    # depends on the cuda graph flag, and returns that file name. The data argument is a mapping
    # from provider to a mapping from row count to the list of bandwidths in column-count order.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    flat_axes = axes.flatten()
    for index, m in enumerate(M_VALUES):
        axis = flat_axes[index]
        for provider in PROVIDERS:
            axis.plot(
                N_VALUES,
                data[provider][m],
                marker="o",
                label=provider,
                color=COLORS[provider],
            )
        axis.set_xscale("log", base=2)
        axis.set_title(f"M={m}")
        axis.set_xlabel("N")
        axis.set_ylabel("GB/s")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    mode = "CUDA Graph" if cudagraph else "standard"
    figure.suptitle(f"Inner-tree SUM reduction {mode} GB/s vs N", y=1.0)
    figure.tight_layout()
    suffix = "_cudagraph" if cudagraph else ""
    output_path = f"strict_numerics_row_reduction{suffix}_gbps_vs_n.png"
    figure.savefig(output_path)
    return output_path


def main() -> None:
    # [FUNCTION DOC] Script entry point. Parses the optional cuda graph flag, requires cuda
    # (assertion error otherwise), runs the inner-tree sanity check, prints the providers and
    # the measurement mode, then measures every provider for every row and column count with
    # inductor caches disabled, printing a summary line per row count with the last column
    # result for each provider. Finally saves the plot and prints its file name.
    parser = argparse.ArgumentParser()
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"  # noqa: S101
    _assert_inner_tree_route()
    print("providers:", PROVIDERS)
    measurement = (
        "CUDA Graph replay with a "
        f"{CUDA_GRAPH_WORKING_SET_BYTES // (1 << 20)} MiB circular input buffer"
        if args.cudagraph
        else "standard"
    )
    print("measurement:", measurement)

    data = {provider: {m: [] for m in M_VALUES} for provider in PROVIDERS}
    with inductor_config.patch({"force_disable_caches": True}):
        for m in M_VALUES:
            for n in N_VALUES:
                for provider in PROVIDERS:
                    data[provider][m].append(measure(m, n, provider, args.cudagraph))
            row = "  ".join(
                f"{provider}={data[provider][m][-1]:.0f}" for provider in PROVIDERS
            )
            print(f"M={m:>6} (N={N_VALUES[-1]}): {row}")

    output_path = _plot(data, args.cudagraph)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
