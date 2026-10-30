# [MODULE DOC] Operator benchmarks for activation modules (GELU, SiLU, ReLU, LeakyReLU). Defines
# the op list, a short config list and a long config list (one-dimensional sizes from 1 up to
# about one million elements on cuda) and a benchmark class. Import-time side effect: registers
# forward and backward tests for the long configs only (the short configs are defined but not
# registered). Runs the benchmark runner when executed as a script.
import operator_benchmark as op_bench

import torch
import torch.nn as nn


"""Microbenchmarks for activation operators."""


activation_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["gelu", nn.GELU],
        ["silu", nn.SiLU],
        ["relu", nn.ReLU],
        ["leaky_relu", nn.LeakyReLU],
    ],
)

activation_short_configs = op_bench.config_list(
    attr_names=["shape"],
    attrs=[
        [(1,)],
        [(64,)],
        [(4096,)],
        [(8192,)],
    ],
    cross_product_configs={
        "device": ["cuda"],
    },
    tags=["short"],
)

activation_long_configs = op_bench.cross_product_configs(
    shape=[(1,), (64,), (4096,), (8192,), (131072,), (262144,), (524288,), (1048576,)],
    device=["cuda"],
    tags=["long"],
)


class ActivationBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for an activation module class chosen by the op list
    # entry. The inputs mapping holds one random tensor whose requires_grad comes
    # from auto_set; the module name is the class name of the activation.
    def init(self, op_func, device, shape):
        # [FUNCTION DOC] Builds the inputs mapping with one random tensor of the
        # given shape on the given device, instantiates the activation module from
        # its class, and sets the module name to that class's name.
        self.inputs = {
            "input": torch.rand(shape, device=device, requires_grad=self.auto_set())
        }
        self.op_func = op_func()
        self.set_module_name(op_func.__name__)

    def forward(self, input):
        # [FUNCTION DOC] Applies the activation module to the input tensor and returns the result.
        return self.op_func(input)


op_bench.generate_pt_tests_from_op_list(
    activation_list,
    activation_long_configs,
    ActivationBenchmark,
)

op_bench.generate_pt_gradient_tests_from_op_list(
    activation_list,
    activation_long_configs,
    ActivationBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
