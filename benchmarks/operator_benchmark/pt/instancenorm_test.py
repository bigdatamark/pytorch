# [MODULE DOC] Operator microbenchmark for functional instance_norm on CPU tensors of two shapes
# (3-D and 4-D). Registers short-config tests at import time. Running as a script starts the
# op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.nn.functional as F


"""Microbenchmarks for instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    tags=["short"],
)


class InstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for instance_norm with random affine parameters. State in the inputs
    # mapping: input, weight, bias and eps.
    def init(self, dims):
        # [FUNCTION DOC] Create a random input of the given dims scaled to roughly minus 128 to
        # 128, random float weight and bias vectors sized to the channel count (second dimension),
        # and an epsilon of 1e-5. Stores them in the inputs mapping. No device argument, so
        # CPU only.
        num_channels = dims[1]
        self.inputs = {
            "input": (torch.rand(*dims) - 0.5) * 256,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        # [FUNCTION DOC] Return instance normalization of input using weight, bias and eps.
        return F.instance_norm(input, weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(instancenorm_configs_short, InstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
