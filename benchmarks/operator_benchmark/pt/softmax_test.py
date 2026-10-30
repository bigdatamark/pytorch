# [MODULE DOC] Microbenchmarks for softmax style modules. One group applies Softmax, Softmax2d
# and LogSoftmax to 4D inputs; another applies Softmax and LogSoftmax to 2D inputs along a
# chosen dimension, including large shapes. Config sets and op lists are built and registered
# at import time. Running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""
Microbenchmarks for the softmax operators.
"""


# Configs for softmax ops
softmax_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


softmax_configs_long = op_bench.cross_product_configs(
    N=[8, 16],
    C=[3],
    H=[256, 512],
    W=[256, 512],
    device=["cpu", "cuda"],
    tags=["long"],
)


softmax_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Softmax", nn.Softmax],
        ["Softmax2d", nn.Softmax2d],
        ["LogSoftmax", nn.LogSoftmax],
    ],
)

softmax_two_dims_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Softmax", nn.Softmax],
        ["LogSoftmax", nn.LogSoftmax],
    ],
)


softmax_two_dims_configs = op_bench.config_list(
    attr_names=["M", "N", "dim"],
    attrs=[
        [700, 23258, 0],
        [700, 23258, 1],
        [1024, 23258, 1],
        [128, 128, 1],
        [48, 128, 1],
        [16, 1024, 1],
        [32, 1024, 1],
        [48, 1024, 1],
        [16, 512, 1],
        [32, 512, 1],
        [48, 512, 1],
        [16, 256, 1],
        [32, 256, 1],
        [48, 256, 1],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)


class SoftmaxBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a softmax module (constructed with default arguments) on a
    # random 4D input.
    def init(self, N, C, H, W, device, op_func):
        # [FUNCTION DOC] Args: N, C, H, W (input shape), device, op_func (module class). Stores
        # a random tensor of that shape on the device in self.inputs and instantiates the
        # module class with no arguments.
        self.inputs = {"input": torch.rand(N, C, H, W, device=device)}
        self.op_func = op_func()

    def forward(self, input):
        # [FUNCTION DOC] Applies the softmax module to the input and returns the result.
        return self.op_func(input)


class Softmax2DimsBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a softmax module constructed with an explicit dim on a random 2D
    # input.
    def init(self, M, N, dim, device, op_func):
        # [FUNCTION DOC] Args: M, N (input shape), dim, device, op_func (module class). Stores
        # a random M x N tensor on the device in self.inputs and instantiates the module class
        # with the given dim.
        self.inputs = {"input": torch.rand(M, N, device=device)}
        self.op_func = op_func(dim=dim)

    def forward(self, input):
        # [FUNCTION DOC] Applies the softmax module to the input and returns the result.
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(
    softmax_ops_list, softmax_configs_short + softmax_configs_long, SoftmaxBenchmark
)


op_bench.generate_pt_tests_from_op_list(
    softmax_two_dims_ops_list, softmax_two_dims_configs, Softmax2DimsBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
