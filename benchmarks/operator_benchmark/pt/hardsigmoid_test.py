# [MODULE DOC] Operator microbenchmark for the Hardsigmoid module on random NCHW CPU tensors. The
# op list holds the module class, which is instantiated per benchmark. Registers tests from the op
# list with short and long configs at import time. Running as a script starts the runner.
import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""
Microbenchmarks for the hardsigmoid operator.
"""


# Configs for hardsigmoid ops
hardsigmoid_configs_short = op_bench.config_list(
    attr_names=["N", "C", "H", "W"],
    attrs=[
        [1, 3, 256, 256],
        [4, 3, 256, 256],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)


hardsigmoid_configs_long = op_bench.cross_product_configs(
    N=[8, 16], C=[3], H=[256, 512], W=[256, 512], device=["cpu"], tags=["long"]
)


hardsigmoid_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["Hardsigmoid", nn.Hardsigmoid],
    ],
)


class HardsigmoidBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for an activation module class applied to a random NxCxHxW tensor.
    def init(self, N, C, H, W, device, op_func):
        # [FUNCTION DOC] Create a random NxCxHxW tensor on the device, store it in the inputs
        # mapping and instantiate op_func (a module class) with no arguments, keeping the instance.
        # Returns nothing.
        self.inputs = {"input_one": torch.rand(N, C, H, W, device=device)}
        self.op_func = op_func()

    def forward(self, input_one):
        # [FUNCTION DOC] Apply the activation module instance to input_one and return the result.
        return self.op_func(input_one)


op_bench.generate_pt_tests_from_op_list(
    hardsigmoid_ops_list,
    hardsigmoid_configs_short + hardsigmoid_configs_long,
    HardsigmoidBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
