# [MODULE DOC] CUDA benchmark of torch.nn.functional.scaled_dot_product_attention forward and
# backward passes over a grid of batch sizes, sequence lengths and causal flags, optionally for
# the FA3 flash attention implementation. Defines frozen dataclasses for configs and results,
# timing and TFLOPS helpers, config generators, table printing and CSV writing. Run as a script
# to print a table and write a CSV file into a results directory. No import-time side effects
# beyond imports.
import itertools
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass

from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
    sdpa_kernel,
    SDPBackend,
)
from torch.nn.functional import scaled_dot_product_attention


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # [FUNCTION DOC] Time a callable with the inductor profiler-based benchmarking helper and
    # return the time in microseconds (the helper reports milliseconds, converted by
    # scaling). Extra arguments are forwarded to the callable on every invocation.
    """Thin wrapper around do_bench_using_profiling"""

    def no_args():
        # [FUNCTION DOC] HELPER: Zero-argument adapter that calls the wrapped function
        # with the captured arguments, as required by the benchmarking helper. Return
        # value is discarded.
        func(*args, **kwargs)

    time = do_bench_using_profiling(no_args)
    return time * 1e3


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # [FUNCTION DOC] Time a callable with torch.utils.benchmark: run it five times as warmup,
    # then adaptive autorange with a minimum run time of 0.1 seconds, and return the median
    # in microseconds. Extra arguments are forwarded to the callable. Not used by the main
    # flow.
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@dataclass(frozen=True)
class ExperimentConfig:
    # [CLASS DOC] Immutable description of one SDPA experiment: batch size, head count, query
    # and key/value sequence lengths, total embedding size, causal flag, dtype, backend (None
    # means all backends allowed) and an optional flash implementation name (None or FA2 for
    # the default, FA3 or FA4 to switch implementation). The device defaults to CUDA. The
    # per-head size is derived.
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    flash_impl: str | None = None  # None/"FA2" for default, "FA3", or "FA4"
    device: torch.device = torch.device("cuda")

    @property
    def head_dim(self) -> int:
        # [FUNCTION DOC] Property giving the per-head dimension: the embedding size
        # divided by the number of heads (integer division).
        return self.embed_dim // self.num_heads

    def asdict(self):
        # [FUNCTION DOC] Return the dataclass fields as a dict plus an extra entry for
        # the derived head dimension.
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    # [CLASS DOC] Immutable measurements for one experiment: forward and backward times in
    # microseconds and the corresponding achieved TFLOPS values.
    forward_time: float  # microseconds
    backward_time: float  # microseconds
    forward_tflops: float
    backward_tflops: float

    def asdict(self):
        # [FUNCTION DOC] Return the result fields as a plain dict.
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    # [CLASS DOC] Immutable pairing of an ExperimentConfig and its ExperimentResults.
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        # [FUNCTION DOC] Return one flat dict merging the config fields (with head
        # dimension) and the result fields.
        dict1 = self.config.asdict()
        dict2 = self.results.asdict()
        return {**dict1, **dict2}


def calculate_tflops(
    config: ExperimentConfig,
    time_us: float,
    is_backward: bool = False,
    sparsity: float = 0.0,
) -> float:
    # [FUNCTION DOC] Estimate achieved TFLOPS for an attention call from the config shape and
    # a time in microseconds. FLOPs per batch element and head are twice the product of query
    # length, key length and head dim for each of the two matmuls, plus twice query length
    # times key length for softmax; the total is scaled by the density (one minus sparsity),
    # multiplied by 2.5 for the backward pass, and divided by time. Returns the float TFLOPS
    # value.
    """
    Calculate TFLOPS for scaled dot product attention.

    Parameters:
    - config: The experiment configuration
    - time_us: The execution time in microseconds
    - is_backward: Whether to calculate for backward pass (includes gradient computation)
    - sparsity: Sparsity factor between 0.0 and 1.0, where 0.0 means no sparsity and 1.0 means fully sparse

    Returns:
    - TFLOPS value
    """
    B = config.batch_size
    H = config.num_heads
    M = config.q_seq_len
    N = config.kv_seq_len
    D = config.head_dim

    # Calculate density factor (1.0 - sparsity)
    density = 1.0 - sparsity

    # Forward pass FLOPs
    qk_flops = (
        M * N * D * 2
    )  # Q*K^T matmul: (M,D) @ (D,N) with 2 FLOPs per multiply-add
    softmax_flops = M * N * 2  # Softmax operations (exp and div)
    av_flops = (
        M * N * D * 2
    )  # Attention @ V: (M,N) @ (N,D) with 2 FLOPs per multiply-add

    total_flops = B * H * (qk_flops + softmax_flops + av_flops)

    # Apply density factor to account for sparsity
    total_flops *= density

    # For backward pass flash uses 2.5x more flops will use this
    if is_backward:
        total_flops *= 2.5

    # Convert to TFLOPS: flops / (time_us * 1e-6) / 1e12
    tflops = total_flops / (time_us * 1e-6) / 1e12

    return tflops


def get_input(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Create random query, key and value tensors with shapes batch, heads,
    # sequence length, head dim (key and value use the key/value length), using the config
    # dtype and device, with gradient tracking enabled. Returns the (q, k, v) tuple.
    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # [FUNCTION DOC] Benchmark forward and backward of one config. Generates inputs, selects
    # the config backend through sdpa_kernel (no restriction if the backend is None) and, for
    # FA3 or FA4, activates that flash implementation first (an ImportError becomes
    # RuntimeError with an install hint). Times the forward call and the backward call of the
    # produced output with a random gradient, restores the default flash implementation
    # afterwards, and converts times to TFLOPS assuming sparsity 0.5 for causal attention.
    # Returns an ExperimentResults.
    q, k, v = get_input(config)
    is_causal = config.is_causal
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )

    # Activate flash attention implementation if specified
    if config.backend is SDPBackend.FLASH_ATTENTION and config.flash_impl in (
        "FA3",
        "FA4",
    ):
        try:
            activate_flash_attention_impl(config.flash_impl)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to activate {config.flash_impl}: {e}\n"
                f"Please install the required flash attention library or run with default configuration (without --flash_test)."
            ) from e

    try:
        with context:
            forward_time = benchmark_cuda_function_in_microseconds(
                scaled_dot_product_attention,
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=None,
            )
            out_torch = scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=None
            )
            d_out = torch.randn_like(out_torch)
            backward_time = benchmark_cuda_function_in_microseconds(
                out_torch.backward, d_out, retain_graph=True
            )
    finally:
        # Restore default FA2 implementation if we activated a different one
        if config.backend is SDPBackend.FLASH_ATTENTION and config.flash_impl in (
            "FA3",
            "FA4",
        ):
            restore_flash_attention_impl()

    # Calculate TFLOPS for forward and backward passes
    sparsity = 0.5 if is_causal else 0.0
    forward_tflops = calculate_tflops(config, forward_time, sparsity=sparsity)
    backward_tflops = calculate_tflops(
        config, backward_time, is_backward=True, sparsity=sparsity
    )

    return ExperimentResults(
        forward_time=forward_time,
        backward_time=backward_time,
        forward_tflops=forward_tflops,
        backward_tflops=backward_tflops,
    )


def print_results(experiments: list[Experiment]):
    # [FUNCTION DOC] Print the experiments as one pretty-formatted table with a column per
    # field. Drops the device column, and drops the backend and flash_impl columns when the
    # first experiment has none for them. Output goes to stdout; floats are shown with three
    # decimals.
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    if table_data["flash_impl"][0] is None:
        del table_data["flash_impl"]
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def write_results_to_csv(
    experiments: list[Experiment], output_dir: str = "benchmark_results"
):
    # [FUNCTION DOC] Write experiment results to a timestamped CSV file inside output_dir,
    # creating the directory if needed. Does nothing (returns early) when the list is empty,
    # otherwise writes a header and one row per experiment without the device column, and
    # prints the path.
    """
    Write experiment results to a CSV file in the specified directory.
    The filename includes a timestamp for uniqueness.
    """
    import csv
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")

    # Get all fields from the first experiment
    if not experiments:
        return

    fieldnames = list(experiments[0].asdict().keys())
    if "device" in fieldnames:
        fieldnames.remove("device")  # Remove device field as it's always cuda

    # Write results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            row = experiment.asdict()
            if "device" in row:
                del row["device"]  # Remove device field
            writer.writerow(row)

    print(f"Results written to: {filename}")


def generate_experiment_configs() -> list[ExperimentConfig]:
    # [FUNCTION DOC] Build the default grid of configs: batch sizes 1, 8, 16, 16 heads, five
    # equal query/key lengths from 128 to 8192, embedding 2048, bfloat16, causal and
    # non-causal, with the backend left as None so all backends stay enabled. Returns the
    # list.
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [None]  # If set to None, all backends are enabled
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, is_causal, dtypes, backends
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
            )
        )

    return all_configs


def generate_experiment_configs_with_flash() -> list[ExperimentConfig]:
    # [FUNCTION DOC] Build the same grid as the default generator but restricted to the flash
    # attention backend and crossing it with the flash implementation names FA2 and FA3.
    # Returns the list of configs.
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [
        SDPBackend.FLASH_ATTENTION,
    ]
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    # FA2 (default), "FA3", "FA4" for the alternative implementations
    flash_impls = ["FA2", "FA3"]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
        flash_impl,
    ) in itertools.product(
        batch_sizes,
        num_heads,
        q_kv_seq_lens,
        embed_dims,
        is_causal,
        dtypes,
        backends,
        flash_impls,
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
                flash_impl=flash_impl,
            )
        )

    return all_configs


def main():
    # [FUNCTION DOC] Command line entry point. Accepts a flash test option to choose the
    # flash config grid instead of the default grid, seeds torch with 123, runs every config
    # with a progress bar, prints the table and writes the CSV into the relative directory
    # benchmark_results one level above the working directory.
    import argparse

    parser = argparse.ArgumentParser(description="SDPA benchmark runner")
    parser.add_argument(
        "--flash_test",
        action="store_true",
        help="Use flash attention configs (generate_experiment_configs_with_flash)",
    )
    args = parser.parse_args()

    seed = 123
    torch.manual_seed(seed)
    results = []
    if args.flash_test:
        configs = generate_experiment_configs_with_flash()
    else:
        configs = generate_experiment_configs()
    for config in tqdm(configs):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)
    write_results_to_csv(results, "../benchmark_results")


if __name__ == "__main__":
    main()
