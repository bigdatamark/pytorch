# [MODULE DOC] Benchmark script comparing scaled dot product attention with a causal bias
# subclass versus the same causal mask materialized as a dense tensor, inside a small
# composite multi-head attention module on CUDA. Defines config and result records, input
# generation, the module, the sweep and a printed summary table of speedups. Nothing runs at
# import time; main runs under the script entry guard.
import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.nn.attention.bias import CausalBias, CausalVariant
from torch.nn.parameter import Parameter


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # [FUNCTION DOC] Times a callable: runs it five times as warmup, then uses the torch
    # benchmark timer with adaptive autorange (minimum run time of 0.1 seconds) on a call with
    # the given positional and keyword arguments, and returns the median time in microseconds.
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
    # [CLASS DOC] Immutable record of one experiment: batch size, head count, query and key
    # sequence lengths, embedding dimension and dtype. Provides the derived per-head dimension
    # and a dictionary export.
    batch_size: int
    num_heads: int
    q_seq_len: int
    k_seq_len: int
    embed_dim: int
    dtype: torch.dtype

    @property
    def head_dim(self) -> int:
        # [FUNCTION DOC] Returns the per-head dimension as the integer quotient of embedding
        # dimension and head count.
        return self.embed_dim // self.num_heads

    def asdict(self):
        # [FUNCTION DOC] Returns the dataclass fields as a dictionary with the derived head
        # dimension added.
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    # [CLASS DOC] Immutable record of the two measured times in microseconds: with the dense
    # materialized mask and with the causal bias subclass.
    materialized_mask_time: float
    attn_mask_subclass_time: float

    def get_entries(self) -> list:
        # [FUNCTION DOC] Returns the two times formatted as strings, materialized mask time
        # first, in a list.
        return [
            f"{self.materialized_mask_time:2f}",
            f"{self.attn_mask_subclass_time:2f}",
        ]


@dataclass(frozen=True)
class Experiment:
    # [CLASS DOC] Immutable pairing of an experiment configuration with its measured results.
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> list:
        # [FUNCTION DOC] Returns the concatenation of the config entries and the result
        # entries as a list.
        # KNOWN ISSUE: the configuration class defines no such entries method, so calling this
        # raises an AttributeError; nothing in the script calls it.
        return self.config.get_entries() + self.results.get_entries()


def generate_inputs(
    batch_size, q_sequence_length, kv_sequence_length, embed_dim, dtype, device
):
    # [FUNCTION DOC] Creates random uniform query, key and value tensors on the given device
    # and dtype: the query has the query sequence length and the key and value share the key-
    # value length, all with the embedding dimension. Returns the three tensors as a tuple.
    q_shape = (batch_size, q_sequence_length, embed_dim)
    kv_shape = (batch_size, kv_sequence_length, embed_dim)

    make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
    make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
    return make_q(), make_kv(), make_kv()


class CompositeMHA(torch.nn.Module):
    # [CLASS DOC] Minimal multi-head attention module built from four raw weight parameters
    # (query, key, value projections and output projection) and scaled dot product attention
    # that takes a dense mask or a causal bias. Parameters are created uninitialized and must
    # be initialized with the reset method.
    def __init__(self, num_heads, embed_dim, device=None, dtype=None):
        # [FUNCTION DOC] Creates the four square weight parameters with uninitialized memory
        # on the given device and dtype, and records the head count, head dimension and
        # embedding dimension. Raises AssertionError if the embedding dimension is not
        # divisible by the head count.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        if self.head_dim * num_heads != self.embed_dim:
            raise AssertionError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.q_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.k_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.v_proj_weight = Parameter(
            torch.empty((embed_dim, embed_dim), **factory_kwargs)
        )
        self.out_proj = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.num_heads = num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | CausalBias,
    ):
        # [FUNCTION DOC] Projects query, key and value with linear layers, reshapes the inputs
        # into batch by heads by sequence by head dimension, runs scaled dot product attention
        # with the mask and no dropout, merges the heads and applies the output projection.
        # Returns the attended tensor.
        # KNOWN ISSUE: the projected tensors are used only for their batch size; the reshape
        # and attention operate on the original unprojected inputs, so the query, key and
        # value projection weights have no effect on the result.
        query_projected = F.linear(query, self.q_proj_weight)
        key_projected = F.linear(key, self.k_proj_weight)
        value_projected = F.linear(value, self.v_proj_weight)

        query = query.view(
            query_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            key_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            value_projected.size(0), -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )

        attn = attn.transpose(1, 2).reshape(query.size(0), -1, self.embed_dim)
        # Match return signature of nn.MHA
        return F.linear(attn, self.out_proj)

    def reset_parameters(self):
        # [FUNCTION DOC] Initializes the query, key and value weights with Xavier uniform and
        # sets the output projection weights to zero, in place.
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.constant_(self.out_proj, 0.0)


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # [FUNCTION DOC] Runs one experiment on a CUDA device: builds the module, resets its
    # parameters, generates inputs, creates a lower-right causal bias and its dense
    # materialization, times the module with each mask, checks that the two outputs are close
    # with the testing assert helper, and returns the two timings. Raises an error if the
    # outputs differ or CUDA is unavailable.
    device = torch.device("cuda")
    composite_mha = CompositeMHA(
        config.num_heads, config.embed_dim, device, config.dtype
    )
    composite_mha.reset_parameters()
    query, key, value = generate_inputs(
        config.batch_size,
        config.q_seq_len,
        config.k_seq_len,
        config.embed_dim,
        config.dtype,
        device,
    )
    attn_mask = CausalBias(
        CausalVariant.LOWER_RIGHT, config.q_seq_len, config.k_seq_len
    )
    attn_mask_tensor = attn_mask._materialize(device)

    materialized_mask_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, attn_mask_tensor
    )
    attn_mask_subclass_time = benchmark_torch_function_in_microseconds(
        composite_mha, query, key, value, attn_mask
    )
    torch.testing.assert_close(
        composite_mha(query, key, value, attn_mask_tensor),
        composite_mha(query, key, value, attn_mask),
    )

    return ExperimentResults(
        materialized_mask_time=materialized_mask_time,
        attn_mask_subclass_time=attn_mask_subclass_time,
    )


def generate_experiment_configs() -> list[ExperimentConfig]:
    # [FUNCTION DOC] Builds the experiment sweep as the product of batch sizes, head counts,
    # query/key length pairs, embedding dimensions and dtypes (bfloat16 only), returning a
    # list of configuration records.
    batch_sizes = [1, 8, 16, 128]
    num_heads = [16, 32]
    q_kv_seq_lens = [(128, 256), (256, 416), (512, 4097), (1024, 2048), (1, 2048)]
    embed_dims = [2048, 4096]
    dtypes = [
        torch.bfloat16,
    ]
    all_configs = []
    for bsz, heads, (q_seq_len, kv_seq_len), embed_dim, dtype in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, dtypes
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                k_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                dtype=dtype,
            )
        )

    return all_configs


def calculate_speedup(results: ExperimentResults) -> float:
    # [FUNCTION DOC] Returns the speedup as the dense mask time divided by the causal bias
    # time.
    return results.materialized_mask_time / results.attn_mask_subclass_time


def print_results(results: list[Experiment]):
    # [FUNCTION DOC] Prints a summary table: computes the speedup for every experiment, finds
    # the largest and smallest, and prints one row each for the average speedup and the max
    # and min cases (with their configuration fields) using the tabulate library. Average row
    # leaves configuration columns empty.
    # Calculate speedups
    speedups = [calculate_speedup(r.results) for r in results]

    # Find indices of max and min speedups
    max_speedup_index = np.argmax(speedups)
    min_speedup_index = np.argmin(speedups)

    # Get the config dictionaries
    max_config_dict = results[max_speedup_index].config.asdict()
    min_config_dict = results[min_speedup_index].config.asdict()

    # Create table data
    table_data = [
        {
            "Type": "Average",
            "Speedup": np.mean(speedups),
            **dict.fromkeys(max_config_dict),
        },
        {"Type": "Max", "Speedup": speedups[max_speedup_index], **max_config_dict},
        {"Type": "Min", "Speedup": speedups[min_speedup_index], **min_config_dict},
    ]

    # Print table
    print(tabulate(table_data, headers="keys", tablefmt="pretty"))


def main():
    # [FUNCTION DOC] Entry point: seeds numpy and torch with a fixed seed, runs every
    # configuration with a progress bar, collects experiments and prints the summary table.
    # Requires CUDA.
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    results = []
    # Run one timing experiment comparing nn_mha vs composite_mha
    for config in tqdm(generate_experiment_configs()):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)


if __name__ == "__main__":
    main()
