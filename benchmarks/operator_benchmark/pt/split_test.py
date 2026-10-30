# [MODULE DOC] Microbenchmark for the split operator on 2D inputs with short and long shape
# configs on cpu and cuda. Registers the benchmark at import time; running as a script starts
# the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for Split operator"""


# Configs for PT Split operator
split_configs_short = op_bench.config_list(
    attr_names=["M", "N", "parts"],
    attrs=[
        [8, 8, 2],
        [256, 512, 2],
        [512, 512, 2],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

split_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], parts=[2, 4], device=["cpu", "cuda"], tags=["long"]
)


class SplitBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of torch split on a random 2D tensor with a precomputed chunk size.
    def init(self, M, N, parts, device):
        # [FUNCTION DOC] Args: M, N (shape), parts, device. Stores a random M x N tensor on the
        # device and a split size equal to the integer part of M times N divided by parts in
        # self.inputs, and sets the module name.
        # KNOWN ISSUE: the split size is an element count, but split works along the first
        # dimension of length M. The size is at least M for these configs, so the op returns a
        # single chunk and the parts config does not produce multiple pieces.
        self.inputs = {
            "input": torch.rand(M, N, device=device),
            "split_size": int(M * N / parts),
        }
        self.set_module_name("split")

    def forward(self, input, split_size: int):
        # [FUNCTION DOC] Splits the input along its first dimension into chunks of the given
        # size and returns the resulting tuple of tensors.
        return torch.split(input, split_size)


op_bench.generate_pt_test(split_configs_short + split_configs_long, SplitBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
