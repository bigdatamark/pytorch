# [MODULE DOC] Microbenchmarks for quantized Linear and dynamic quantized Linear modules.
# Defines a shared base class and two concrete benchmarks, registered at import time over the
# short and long linear configs from pt.configs with CUDA configs removed. Running as a script
# starts the benchmark runner.
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd


"""
Microbenchmarks for Quantized Linear operators.
"""


class _QLinearBenchmarkBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] HELPER: Shared base for linear benchmarks. It prepares quantized and float
    # inputs and installs a quantized weight and quantization parameters on the module supplied
    # by a subclass.
    def init(self, N, IN, OUT, linear_under_test):
        # [FUNCTION DOC] HELPER: Args: N (batch), IN, OUT (feature sizes), linear_under_test
        # (module built by the subclass). Creates scale 1/255 and zero point 0 as tensors, a
        # random float input self.X and its quint8 quantization self.qX, and a random qint8
        # weight of shape OUT x IN. Keeps the module as self.qlinear and assigns its weight,
        # scale and zero_point attributes. Does not set self.inputs; subclasses do.
        scale = torch.tensor(1.0 / 255)
        zero_point = torch.tensor(0)
        self.X = torch.randn(N, IN, dtype=torch.float32)
        self.qX = torch.quantize_per_tensor(
            self.X, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        W = torch.randn(OUT, IN, dtype=torch.float32)
        qW = torch.quantize_per_tensor(W, scale=scale, zero_point=0, dtype=torch.qint8)

        # Assume that the `self.qlinear` is set in the child
        self.qlinear = linear_under_test
        self.qlinear.weight = qW
        self.qlinear.scale = scale
        self.qlinear.zero_point = zero_point

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Applies the module under test to the given input and returns
        # the output.
        # Assume that the `self.input` is set in the child
        return self.qlinear(input)


class QLinearBenchmark(_QLinearBenchmarkBase):
    # [CLASS DOC] Benchmark of the statically quantized Linear module with a quint8 input.
    def init(self, N, IN, OUT, device):
        # [FUNCTION DOC] Args: N, IN, OUT, device (unused). Builds a quantized Linear module
        # and runs the base setup, then uses the quantized input as the only input and sets the
        # module name.
        super().init(N, IN, OUT, nnq.Linear(IN, OUT))
        self.inputs = {"input": self.qX}
        self.set_module_name("QLinear")


class QDynamicLinearBenchmark(_QLinearBenchmarkBase):
    # [CLASS DOC] Benchmark of the dynamic quantized Linear module, which takes a float input.
    def init(self, N, IN, OUT, device):
        # [FUNCTION DOC] Args: N, IN, OUT, device (unused). Builds a dynamic quantized Linear
        # module and runs the base setup, then uses the float input as the only input and sets
        # the module name.
        super().init(N, IN, OUT, nnqd.Linear(IN, OUT))
        self.inputs = {"input": self.X}
        self.set_module_name("QDynamicLinear")


op_bench.generate_pt_test(
    configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long),
    QLinearBenchmark,
)
op_bench.generate_pt_test(
    configs.remove_cuda(configs.linear_configs_short + configs.linear_configs_long),
    QDynamicLinearBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
