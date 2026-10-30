# [MODULE DOC] Operator microbenchmark for the quantized linear_unpack_fp16 op, which recovers a
# weight from its fp16 packed form. CPU only, with long and short configs. Registers tests at
# import time. Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for linear_unpack_fp16_ operator. Supports both Caffe2/PyTorch."""

# Configs for PT linear_unpack_fp16 operator
linear_unpack_fp16_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu"], tags=["long"]
)

linear_unpack_fp16_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu"],
    },
    tags=["short"],
)


class LinearUnpackFP16Benchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for unpacking an fp16 prepacked weight. The input is created by running
    # the prepack op on a random tensor.
    def init(self, M, N, K, device):
        # [FUNCTION DOC] Create a random float32 tensor of shape MxNxK on the device, run the
        # quantized prepack op on it (the unpack op needs the prepack output as input), store the
        # packed object in the inputs mapping and set the module name to linear_unpack_fp16.
        # input to unpack operator must be what the output is for prepack operator
        self.inputs = {
            "input_one": torch.ops.quantized.linear_prepack_fp16(
                torch.rand(
                    M, N, K, device=device, requires_grad=False, dtype=torch.float32
                )
            )
        }
        self.set_module_name("linear_unpack_fp16")

    def forward(self, input_one):
        # [FUNCTION DOC] Run the quantized unpack op on the packed input_one and return the result.
        return torch.ops.quantized.linear_unpack_fp16(input_one)


# The generated test names based on linear_unpack_fp16_short_configs will be in the following pattern:
# linear_unpack_fp16_M8_N16_K32_devicecpu

op_bench.generate_pt_test(
    linear_unpack_fp16_long_configs + linear_unpack_fp16_short_configs,
    LinearUnpackFP16Benchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
