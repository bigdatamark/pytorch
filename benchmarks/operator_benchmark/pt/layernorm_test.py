# [MODULE DOC] Operator microbenchmark for functional layer_norm on CPU tensors of four shapes
# (three 3-D and one 4-D). Registers short-config tests at import time. Running as a script starts
# the op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.nn.functional as F


"""Microbenchmarks for layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    tags=["short"],
)


class LayerNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for layer_norm normalizing over all non-batch dimensions with random
    # affine parameters. State in the inputs mapping: input, weight, bias and eps.
    def init(self, dims):
        # [FUNCTION DOC] Create a random input of the given dims scaled to roughly minus 128 to
        # 128, and random float weight and bias tensors shaped like the input without its first
        # dimension, plus an epsilon of 1e-5. Stores them in the inputs mapping. No device
        # argument, so CPU only.
        input = (torch.rand(*dims) - 0.5) * 256
        self.inputs = {
            "input": input,
            "weight": torch.rand(*input.size()[1:], dtype=torch.float),
            "bias": torch.rand(*input.size()[1:], dtype=torch.float),
            "eps": 1e-5,
        }

    def forward(self, input, weight, bias, eps: float):
        # [FUNCTION DOC] Return layer normalization of input over every dimension after the first,
        # using weight, bias and eps.
        return F.layer_norm(input, input.size()[1:], weight=weight, bias=bias, eps=eps)


op_bench.generate_pt_test(layernorm_configs_short, LayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
