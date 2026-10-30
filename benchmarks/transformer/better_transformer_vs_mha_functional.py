# [MODULE DOC] Benchmark script for the fast path of the built-in multi-head attention (nested
# tensor inputs, flash attention) versus the slow path with padded inputs and a key padding
# mask, on CUDA with half precision. Sweeps batch size, sequence length, embedding size, head
# count and a beta distribution parameter that controls padding, prints a table and optionally
# writes a CSV file of results and a JSON file of mismatch counts. Import-time side effects:
# suppresses all warnings and creates the global mismatch counter.
"""
Tests the performance of torch.nn.MultiheadAttention's fast path (BetterTransformer)
vs the slow path (torch.nn.functional.multi_head_attention)

To run this script install these dependencies:

pip install tqdm
pip install prettytable
"""

import argparse
import itertools
import json
import random
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from pprint import pprint

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

import torch


warnings.filterwarnings("ignore")

error_dict = defaultdict(int)


def benchmark_torch_function(iters, f, *args, **kwargs):
    # [FUNCTION DOC] Times a callable on the GPU: calls it twice as warmup, synchronizes,
    # records CUDA events around the given number of calls, synchronizes again and computes
    # the average time per call in microseconds. Returns a tuple of that average followed by
    # the outputs of one more call (which must be iterable).
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) * 1000 / iters, *f(*args, **kwargs)


def run(
    a: int,
    b: int,
    iters: int,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    num_heads: int,
    device: str,
    dtype: str,
    block_size: int,
    seed,
):
    # [FUNCTION DOC] Runs one comparison. Seeds the random generators, draws per-row sequence
    # lengths from a beta distribution rounded to multiples of the block size (at least one
    # block), forces one random row to the full length, builds a nested random query and a
    # native multi-head attention module with shared projection weights, and builds a boolean
    # query mask on the GPU. Returns None if the mask has zero length. Times the nested fast
    # path (inference mode, flash enabled) and the padded path with a key padding mask,
    # converts both outputs to masked padded form, checks they are close (mismatches are
    # counted in the global error counter and printed), and returns an ordered mapping of
    # dtype, sizes, both times, the speedup of slow over fast and the padding fraction.
    # KNOWN ISSUE: the mask is always moved to the default CUDA device regardless of the
    # device argument.
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from scipy.stats import beta

    lengths = (
        beta.rvs(a, b, size=batch_size)
        * (sequence_length + block_size - 1)
        // block_size
    )
    lengths = list(map(int, list(lengths)))
    lengths = [l * block_size for l in lengths]
    lengths = [max(l, block_size) for l in lengths]

    # Used to enforce no padding
    # lengths = [sequence_length] * batch_size

    # Ensure one row in the batch of ele has the max_sequence_length
    lengths[random.randint(0, batch_size - 1)] = sequence_length

    q = [torch.randn(l, embed_dim, device=device, dtype=dtype) for l in lengths]
    q = torch.nested.nested_tensor(q, device=device, dtype=dtype)
    k, v = q, q

    qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
    proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    native_mha = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True, device=device, dtype=dtype
    ).eval()
    native_mha.in_proj_weight = qkv.weight
    native_mha.in_proj_bias = qkv.bias
    native_mha.out_proj.weight = proj.weight
    native_mha.out_proj.bias = proj.bias

    # Create query mask
    q_mask = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [torch.tensor([True] * length, dtype=torch.bool) for length in lengths]
        ),
        0,
    )
    q_mask = q_mask.cuda()

    if q_mask.size(1) == 0:
        return None

    # Benchmark the native MHA in core
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        with torch.inference_mode():
            time_native_mha_fast, y_native_mha_fast, _ = benchmark_torch_function(
                iters, native_mha, q, k, v, need_weights=False
            )
    q = q.to_padded_tensor(0)
    k = q
    v = q
    # Internal Flash Attention
    time_native_mha_slow, y_native_mha_slow, _ = benchmark_torch_function(
        iters, native_mha, q, k, v, key_padding_mask=~q_mask, need_weights=False
    )

    # Convert to padded for comparison
    if y_native_mha_fast.is_nested:
        y_native_mha_fast = torch.nested.to_padded_tensor(y_native_mha_fast, 0)
    y_native_mha_fast = y_native_mha_fast * q_mask.unsqueeze(-1)

    if y_native_mha_slow.is_nested:
        y_native_mha_slow = torch.nested.to_padded_tensor(y_native_mha_slow, 0)
    y_native_mha_slow = y_native_mha_slow * q_mask.unsqueeze(-1)

    # Correctness check
    entry_name = f"batch:{batch_size}_seq_len:{sequence_length}_n_heads:{num_heads}_embed_dim:{embed_dim}"
    try:
        torch.testing.assert_close(
            y_native_mha_fast, y_native_mha_slow, atol=1e-3, rtol=1e-3
        )
    except AssertionError:
        error_dict[entry_name] += 1
        pprint(error_dict)

    # Calculate amount of padding
    padding = 1 - q_mask.float().mean().item()

    # Calculate the speedup for flash attention
    speedup_fast_internal = time_native_mha_slow / time_native_mha_fast

    result_entry = OrderedDict()
    result_entry["dtype"] = dtype
    result_entry["batch_size"] = batch_size
    result_entry["sequence_length"] = sequence_length
    result_entry["n_heads"] = num_heads
    result_entry["embed_dim"] = embed_dim
    result_entry["time_native_mha_slow(\u00b5s)"] = f"{time_native_mha_slow:.3f}"
    result_entry["time_native_mha_fast (\u00b5s)"] = f"{time_native_mha_fast:.3f}"
    result_entry["speedup flash_mha v native_mha"] = f"{speedup_fast_internal:.3f}"
    result_entry["padding"] = f"{padding:.3f}"
    return result_entry


def main(save_path: Path | None, error_path: Path | None):
    # [FUNCTION DOC] Runs the full sweep over batch sizes, sequence lengths, embedding sizes,
    # head counts and beta parameters with a progress bar, calling the single comparison with
    # fixed settings (CUDA, float16, fixed seed). Skips combinations that return nothing,
    # builds a results table with column names from the first entry, prints it and the
    # mismatch counter, writes the table as CSV to the save path when given, prints the total
    # error count and writes the mismatch counter as JSON to the error path when given.
    # Requires CUDA.
    table = PrettyTable()
    entries = defaultdict(list)

    print("CUDA device: ", torch.cuda.get_device_name(0))
    iters = 100
    header = None
    batch_sizes = [16, 32, 64, 128, 256]
    sequence_lengths = [64, 128, 256, 512]
    embed_dims = [512, 1024]
    num_heads_list = [8, 16]
    betas = range(1, 64, 4)

    for batch_size, sequence_length, embed_dim, num_heads, block_size, b in tqdm(
        list(
            itertools.product(
                batch_sizes, sequence_lengths, embed_dims, num_heads_list, [2], betas
            )
        )
    ):
        seed = 26214  # Magic number that works well for higher b values
        entry = run(
            1,
            b * 0.05,
            iters,
            batch_size,
            sequence_length,
            embed_dim,
            num_heads,
            "cuda",
            torch.float16,
            block_size,
            seed,
        )
        if entry is None:
            continue
        if header is None:
            table.field_names = list(entry.keys())
            header = list(entry.keys())
        row = []
        for k, v in entry.items():
            row.append(v)
            entries[k].append(v)
        table.add_row(row)

    # Print the full table to console
    print(table)
    pprint(error_dict)

    csv_string = table.get_csv_string()
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)

    print(f"Total errors: {sum(error_dict.values())}")
    if error_path is not None:
        with open(error_path, "w") as file:
            file.write(json.dumps(error_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", "--save_path", type=str, help="Path to save the results"
    )
    parser.add_argument(
        "--error-save-path",
        "--error_save_path",
        type=str,
        help="Path to save the errors",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path) if args.save_path else None
    error_path = Path(args.error_save_path) if args.error_save_path else None

    main(save_path, error_path)
