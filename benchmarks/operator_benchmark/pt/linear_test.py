# [MODULE DOC] Operator microbenchmark for the nn.Linear module using the shared linear short and
# long configs from the pt configs module. Registers tests at import time. Running as a script
# starts the op_bench runner.
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""Microbenchmarks for Linear operator."""


class LinearBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a Linear layer applied to a random batch. State: the linear module
    # and an input in the inputs mapping.
    def init(self, N, IN, OUT, device):
        # [FUNCTION DOC] Create a random input of shape N x IN on the device, a Linear module from
        # IN to OUT features moved to the device, and set the module name to linear.
        self.inputs = {"input_one": torch.rand(N, IN, device=device)}
        self.linear = nn.Linear(IN, OUT).to(device=device)
        self.set_module_name("linear")

    def forward(self, input_one):
        # [FUNCTION DOC] Apply the linear layer to input_one and return the output of shape N
        # x OUT.
        return self.linear(input_one)


op_bench.generate_pt_test(
    configs.linear_configs_short + configs.linear_configs_long, LinearBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
