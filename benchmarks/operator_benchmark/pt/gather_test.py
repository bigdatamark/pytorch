# [MODULE DOC] Operator microbenchmark for torch.gather on a random MxN matrix along dimension 0 or
# 1, on CPU and CUDA, with short and long configs. Registers tests at import time. Running as a
# script starts the op_bench runner.
import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for gather operator."""

# An example input from this configuration is M=4, N=4, dim=0.
gather_configs_short = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [256, 512, 0],
        [512, 512, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


gather_configs_long = op_bench.cross_product_configs(
    M=[128, 1024], N=[128, 1024], dim=[0, 1], device=["cpu", "cuda"], tags=["long"]
)


class GatherBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for gather with a random integer index tensor of the same shape as the
    # input. State in the inputs mapping: input_one, dim and index.
    def init(self, M, N, dim, device):
        # [FUNCTION DOC] Seed numpy with a fixed value, create a random MxN input on the device and
        # an MxN integer index tensor with values below the input size along dim (M for dim 0, else
        # N). Stores input, dim and index in the inputs mapping and sets the module name to gather.
        min_val = M if dim == 0 else N
        numpy.random.seed((1 << 32) - 1)
        self.inputs = {
            "input_one": torch.rand(M, N, device=device),
            "dim": dim,
            "index": torch.tensor(
                numpy.random.randint(0, min_val, (M, N)), device=device
            ),
        }
        self.set_module_name("gather")

    def forward(self, input_one, dim: int, index):
        # [FUNCTION DOC] Gather values of input_one along dim at the positions given by index and
        # return the result, which has the shape of index.
        return torch.gather(input_one, dim, index)


op_bench.generate_pt_test(gather_configs_short + gather_configs_long, GatherBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
