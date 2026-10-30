# [MODULE DOC] Benchmark script comparing torch.nn.MultiheadAttention against a composite
# implementation (fused input projection, scaled_dot_product_attention, output projection) on
# CUDA, eager and torch.compile, for dense and padded (nested tensor) inputs. Defines the frozen
# dataclasses for configs and results, CompositeMHA, input generation, timing and reporting
# helpers. Import-time side effect: all Python warnings are ignored. Run as a script with an
# optional save path to print a table and write it as CSV.
import argparse
import itertools
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch.backends.cuda import sdp_kernel


warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class ExperimentConfig:
    # [CLASS DOC] Immutable description of one experiment: batch size, head count, maximum
    # sequence length, embedding size, dtype, optional padding fraction (None means dense
    # input) and four flags selecting which scaled dot product attention backends are enabled
    # (math, flash, memory efficient, cudnn). Provides the field values and names as table
    # columns.
    batch_size: int
    num_heads: int
    max_sequence_len: int
    embed_dimension: int
    dtype: torch.dtype
    pad_percentage: float | None
    enable_math: bool
    enable_flash: bool
    enable_mem_efficient: bool
    enable_cudnn: bool

    def get_entries(self) -> list:
        # [FUNCTION DOC] Return the field values in declaration order as a list, matching
        # the order of get_entry_names.
        return [
            self.batch_size,
            self.num_heads,
            self.max_sequence_len,
            self.embed_dimension,
            self.dtype,
            self.pad_percentage,
            self.enable_math,
            self.enable_flash,
            self.enable_mem_efficient,
            self.enable_cudnn,
        ]

    @classmethod
    def get_entry_names(cls) -> list[str]:
        # [FUNCTION DOC] Return the list of field names in declaration order, used as
        # table column headers.
        return [
            "batch_size",
            "num_heads",
            "max_sequence_len",
            "embed_dimension",
            "dtype",
            "pad_percentage",
            "enable_math",
            "enable_flash",
            "enable_mem_efficient",
            "enable_cudnn",
        ]


@dataclass(frozen=True)
class ExperimentResults:
    # [CLASS DOC] Immutable timings in microseconds for one experiment: eager
    # nn.MultiheadAttention, its compiled version (None if not measured), the composite
    # implementation, and its compiled version (None if not measured).
    nn_mha_time: float
    compiled_nn_mha_time: float | None
    composite_mha_time: float
    compiled_composite_mha_time: float | None

    def get_entries(self) -> list:
        # [FUNCTION DOC] Return the four timings formatted as strings for table rows. The
        # compiled entries become None when the value is missing or zero; otherwise
        # values are formatted as floating point text.
        return [
            f"{self.nn_mha_time:2f}",
            f"{self.compiled_nn_mha_time:2f}" if self.compiled_nn_mha_time else None,
            f"{self.composite_mha_time:2f}",
            f"{self.compiled_composite_mha_time:2f}"
            if self.compiled_composite_mha_time
            else None,
        ]

    @classmethod
    def get_entry_names(cls) -> list[str]:
        # [FUNCTION DOC] Return the four column header strings for the timings, each
        # labelled with the microsecond unit.
        return [
            "nn_mha_time (\u00b5s)",
            "compiled_nn_mha_time (\u00b5s)",
            "composite_mha_time (\u00b5s)",
            "compiled_composite_mha_time (\u00b5s)",
        ]


@dataclass(frozen=True)
class Experiment:
    # [CLASS DOC] Immutable pairing of an ExperimentConfig and its ExperimentResults.
    config: ExperimentConfig
    results: ExperimentResults

    def get_entries(self) -> list:
        # [FUNCTION DOC] Return the config entries followed by the results entries as one
        # table row.
        return self.config.get_entries() + self.results.get_entries()


class CompositeMHA(torch.nn.Module):
    # [CLASS DOC] Minimal multi-head self attention module built from the weights of an
    # existing torch.nn.MultiheadAttention. Holds the packed input projection weight and
    # bias, the output projection module and the head count, and uses
    # scaled_dot_product_attention for the core. Its call signature and return value (output,
    # None) mimic the nn module.
    def __init__(self, num_heads, in_proj_weight, in_proj_bias, out_proj):
        # [FUNCTION DOC] Store the head count, the packed query/key/value projection
        # weight and bias, and the output projection module as attributes (the tensors
        # are shared, not copied).
        super().__init__()
        self.in_proj_weight = in_proj_weight
        self.in_proj_bias = in_proj_bias
        self.out_proj = out_proj
        self.num_heads = num_heads

    def forward(self, query, key, value, mask):
        # [FUNCTION DOC] Self attention forward pass. Requires query, key and value to be
        # the very same tensor object and mask to be None, otherwise raises
        # NotImplementedError. Projects the input with the packed weight, splits the
        # result into three chunks along the last dim, reshapes each to batch, heads,
        # sequence, head dim, runs scaled dot product attention without mask, dropout or
        # causality, merges the heads, applies the output projection and returns (output,
        # None).
        if not (query is key and key is value):
            raise NotImplementedError(
                "query, key and value must be the same Tensor for now."
            )
        if mask is not None:
            raise NotImplementedError("mask is currently not supported.")

        query_projected = torch.nn.functional.linear(
            query, self.in_proj_weight, self.in_proj_bias
        )

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)

        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        # Match return signature of nn.MHA
        return self.out_proj(attn), None


def build_composite_mha_from_nn_mha(pt):
    # [FUNCTION DOC] Create a CompositeMHA that shares the weights of the given
    # torch.nn.MultiheadAttention. Raises AssertionError unless the module uses one embedding
    # dimension for query, key and value, has a packed input projection weight, and was
    # created with batch_first set.
    if not pt._qkv_same_embed_dim:
        raise AssertionError("pt._qkv_same_embed_dim must be True")
    in_proj_weight = pt.in_proj_weight
    if in_proj_weight is None:
        raise AssertionError("pt.in_proj_weight must not be None")
    if not pt.batch_first:
        raise AssertionError("pt.batch_first must be True")
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


def generate_rand_batch(
    batch_size,
    max_sequence_len,
    embed_dimension,
    pad_percentage=None,
    dtype=torch.float16,
    device="cuda",
):
    # [FUNCTION DOC] Create a random input batch. With no padding fraction, return a pair of
    # one dense normally distributed tensor of shape batch, sequence, embedding and None.
    # Otherwise draw a per-sample length from a gaussian around the padding fraction, force
    # one random sample to the maximum length, and return a nested tensor of per-sample
    # random tensors plus the list of lengths. Uses the global random state and the given
    # dtype and device.
    if not pad_percentage:
        return (
            torch.randn(
                batch_size,
                max_sequence_len,
                embed_dimension,
                dtype=dtype,
                device=device,
            ),
            None,
        )
    # Really slow but should work
    seq_len_list = [
        int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01)))
        for _ in range(batch_size)
    ]
    # Make random ele max length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    # print(f"Theoretical padding: {pad_percentage} actual: {1 - (sum(seq_len_list) / (batch_size * max_sequence_len))}")
    return (
        torch.nested.nested_tensor(
            [
                torch.randn(seq_len, embed_dimension, dtype=dtype, device=device)
                for seq_len in seq_len_list
            ]
        ),
        seq_len_list,
    )


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    # [FUNCTION DOC] Time a callable with torch.utils.benchmark blocked autorange and return
    # the mean run time in microseconds. Extra positional and keyword arguments are forwarded
    # to it.
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def assert_close_tensors(tensor_a, tensor_b):
    # [FUNCTION DOC] Rough correctness check of two tensors. If both are nested, compares
    # each unbound component with absolute and relative tolerance of 1e-2; otherwise compares
    # the tensors with tolerance 1e-3. Raises AssertionError reporting the maximum difference
    # when values are not close.
    # First order sanity check. Not a replacement for rigorous tests.
    if tensor_a.is_nested and tensor_b.is_nested:
        for a, b in zip(tensor_a.unbind(), tensor_b.unbind()):
            if not torch.allclose(a, b, atol=1e-2, rtol=1e-2):
                raise AssertionError(
                    f"Nested tensors not close: max diff = {(a - b).abs().max()}"
                )
    else:
        if not torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3):
            raise AssertionError(
                f"Tensors not close: max diff = {(tensor_a - tensor_b).abs().max()}"
            )


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    # [FUNCTION DOC] Run one experiment on CUDA inside the sdp_kernel context built from the
    # config flags: build an nn MultiheadAttention (eval mode, config dtype, no dropout) and
    # its composite counterpart, generate an input, check that outputs are close, time both
    # eagerly and, only when there is no padding (nested tensors cannot be compiled), also
    # with torch.compile. Returns an Experiment holding the config and timings.
    # KNOWN ISSUE: the return annotation says ExperimentResults but the function returns an
    # Experiment that wraps the config and the results.
    with sdp_kernel(
        enable_math=config.enable_math,
        enable_flash=config.enable_flash,
        enable_mem_efficient=config.enable_mem_efficient,
        enable_cudnn=config.enable_cudnn,
    ):
        dropout_p = 0.0
        mask = None

        nn_mha = torch.nn.MultiheadAttention(
            embed_dim=config.embed_dimension,
            num_heads=config.num_heads,
            batch_first=True,
            dropout=dropout_p,
        )
        nn_mha = nn_mha.eval().to("cuda", config.dtype)
        composite_mha = build_composite_mha_from_nn_mha(nn_mha)
        qkv, lengths = generate_rand_batch(
            config.batch_size,
            config.max_sequence_len,
            config.embed_dimension,
            config.pad_percentage,
            config.dtype,
        )
        nn_mha_output, _ = nn_mha(qkv, qkv, qkv, mask)
        composite_mha_output, _ = composite_mha(qkv, qkv, qkv, mask)

        # First order sanity check
        assert_close_tensors(nn_mha_output, composite_mha_output)

        nn_mha_time = benchmark_torch_function_in_microseconds(
            nn_mha, qkv, qkv, qkv, mask
        )
        composite_mha_time = benchmark_torch_function_in_microseconds(
            composite_mha, qkv, qkv, qkv, mask
        )

        # TorchDynamo will error on NestedTensors
        if config.pad_percentage is None:
            compiled_nn_mha = torch.compile(nn_mha)
            compiled_composite_mha = torch.compile(composite_mha)

            compiled_nn_mha_time = benchmark_torch_function_in_microseconds(
                compiled_nn_mha, qkv, qkv, qkv, mask
            )

            compiled_composite_mha_time = benchmark_torch_function_in_microseconds(
                compiled_composite_mha,
                qkv,
                qkv,
                qkv,
                mask,
            )
        else:
            compiled_nn_mha_time = None
            compiled_composite_mha_time = None

        results = ExperimentResults(
            nn_mha_time,
            compiled_nn_mha_time,
            composite_mha_time,
            compiled_composite_mha_time,
        )
        return Experiment(config, results)


# Could return generator
def generate_experiments(
    batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
) -> list[ExperimentConfig]:
    # [FUNCTION DOC] Build the list of ExperimentConfig objects for the cartesian product of
    # the given batch sizes, head counts, maximum sequence lengths, embedding sizes, dtypes
    # and padding fractions. Every config enables the flash, memory efficient and cudnn
    # backends and disables the math backend.
    configs = []
    for bsz, n_heads, seq_len, embed_dim, dtype, padding in itertools.product(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    ):
        configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=n_heads,
                max_sequence_len=seq_len,
                embed_dimension=embed_dim,
                dtype=dtype,
                pad_percentage=padding,
                enable_math=False,
                enable_flash=True,
                enable_mem_efficient=True,
                enable_cudnn=True,
            )
        )
    return configs


def main(save_path: Path | None):
    # [FUNCTION DOC] Entry point. Seeds numpy and torch (123), runs and prints one fixed
    # experiment, then runs a grid of experiments (batch 256, 32 heads, sequence 256,
    # embedding 512, three dtypes, with and without padding) with a progress bar, prints them
    # in a PrettyTable and, when save_path is not None, writes the table as CSV text to that
    # path.
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Run one timing experiment comparing nn_mha vs composite_mha
    config = ExperimentConfig(
        batch_size=128,
        num_heads=8,
        max_sequence_len=512,
        embed_dimension=512,
        dtype=torch.float16,
        pad_percentage=None,
        enable_math=False,
        enable_flash=True,
        enable_mem_efficient=True,
        enable_cudnn=True,
    )

    experiment = run_single_experiment(config)
    pprint(experiment)

    table = PrettyTable()
    table.float_format = ".3"
    table.field_names = (
        ExperimentConfig.get_entry_names() + ExperimentResults.get_entry_names()
    )

    # Run a bunch of experiments
    batch_sizes = [256]
    num_heads = [32]
    max_seq_lens = [256]
    embed_dims = [512]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    pad_percentages = [None, 0.9]

    experiment_configs = generate_experiments(
        batch_sizes, num_heads, max_seq_lens, embed_dims, dtypes, pad_percentages
    )

    experiments: list[Experiment] = []
    for experiment_config in tqdm(experiment_configs):
        experiment = run_single_experiment(experiment_config)
        experiments.append(experiment)
        table.add_row(experiment.get_entries())

    print(table)

    csv_string = table.get_csv_string()
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", "--save_path", type=str, help="Path to save the results"
    )

    args = parser.parse_args()
    save_path = Path(args.save_path) if args.save_path else None
    main(save_path)
