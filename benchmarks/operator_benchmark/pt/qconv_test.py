# [MODULE DOC] Microbenchmarks for quantized Conv1d and Conv2d modules (torch.ao.nn.quantized).
# Defines two benchmark classes and registers them at import time using the short and long
# convolution configs from pt.configs with CUDA configs removed. Running as a script starts the
# benchmark runner.
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized as nnq


"""
Microbenchmarks for qConv operators.
"""


class QConv1dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a quantized 1D convolution module on a random quint8 input.
    # Groups are fixed to 1 and padding to 0. Holds the quantized weight, scale and zero point.
    # def init(self, N, IC, OC, L, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, L, device):
        # [FUNCTION DOC] Builds the benchmark. Args: IC/OC (input/output channels), kernel,
        # stride, N (batch), L (length), device (unused, only CPU is benchmarked). Quantizes a
        # random float input of shape N x IC x L to quint8 and a random weight to qint8, both
        # with scale 1/255 and zero point 0. Creates the quantized Conv1d, installs the weight
        # without bias, overrides its scale and zero_point tensors and sets the module name.
        # The input goes in self.inputs.
        G = 1
        pad = 0
        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, L, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv1d = nnq.Conv1d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv1d.set_weight_bias(self.qW, None)
        self.qconv1d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv1d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv1d")

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized conv module to the given quantized input and
        # returns the output.
        return self.qconv1d(input)


class QConv2dBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of a quantized 2D convolution module on a random quint8 input.
    # Holds the quantized weight, scale and zero point; groups and padding come from the
    # config.
    # def init(self, N, IC, OC, H, W, G, kernel, stride, pad):
    def init(self, IC, OC, kernel, stride, N, H, W, G, pad, device):
        # [FUNCTION DOC] Builds the benchmark. Args: IC/OC (channels), kernel (square size),
        # stride, N (batch), H/W (spatial size), G (groups), pad (padding), device (unused).
        # Quantizes a random float input and a random weight (OC x IC/G x kernel x kernel) to
        # quint8 and qint8 with scale 1/255 and zero point 0. Creates the quantized Conv2d,
        # installs the weight without bias, overrides its scale and zero_point tensors and sets
        # the module name. Note the local name for the weight shadows the width argument after
        # the input is created.
        # super().init(N, IC, OC, (H, W), G, (kernel, kernel), stride, pad)

        self.scale = 1.0 / 255
        self.zero_point = 0
        X = torch.randn(N, IC, H, W, dtype=torch.float32)
        qX = torch.quantize_per_tensor(
            X, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8
        )
        # Convert the tensor to NHWC format
        W = torch.randn(OC, IC // G, kernel, kernel, dtype=torch.float32)
        self.qW = torch.quantize_per_tensor(
            W, scale=self.scale, zero_point=0, dtype=torch.qint8
        )

        self.inputs = {"input": qX}

        self.qconv2d = nnq.Conv2d(IC, OC, kernel, stride=stride, padding=pad, groups=G)
        self.qconv2d.set_weight_bias(self.qW, None)
        self.qconv2d.scale = torch.tensor(self.scale, dtype=torch.double)
        self.qconv2d.zero_point = torch.tensor(self.zero_point, dtype=torch.int)
        self.set_module_name("QConv2d")

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized conv module to the given quantized input and
        # returns the output.
        return self.qconv2d(input)


op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_1d_configs_short + configs.conv_1d_configs_long),
    QConv1dBenchmark,
)
op_bench.generate_pt_test(
    configs.remove_cuda(configs.conv_2d_configs_short + configs.conv_2d_configs_long),
    QConv2dBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
