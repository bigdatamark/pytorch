# [MODULE DOC] Operator microbenchmark for functional group_norm on CPU tensors of two shapes (3-D
# and 4-D) with group counts 2 and 4. Registers short-config tests at import time. Running as a
# script starts the op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.nn.functional as F


"""Microbenchmarks for groupnorm operator."""

groupnorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    num_groups=(2, 4),
    tags=["short"],
)


class GroupNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for group_norm with random affine parameters. State in the inputs
    # mapping: input, num_groups, weight, bias and eps.
    def init(self, dims, num_groups):
        # [FUNCTION DOC] Create a random input of the given dims scaled to roughly minus 128 to
        # 128, random float weight and bias vectors sized to the channel count (second dimension),
        # and an epsilon of 1e-5. Stores them with num_groups in the inputs mapping. No device
        # argument, so CPU only.
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims) - 0.5) * 256,
            "num_groups": num_groups,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
        }

    def forward(self, input, num_groups: int, weight, bias, eps: float):
        # [FUNCTION DOC] Return group normalization of input using num_groups groups, weight, bias
        # and eps.
        return F.group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(groupnorm_configs_short, GroupNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
