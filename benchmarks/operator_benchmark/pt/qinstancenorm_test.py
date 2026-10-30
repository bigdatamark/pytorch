# [MODULE DOC] Microbenchmark for the quantized instance_norm operator. Defines a short config
# set (3D and 4D shapes, qint8) and one benchmark class registered at import time. Running as a
# script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized instancenorm operator."""

instancenorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QInstanceNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of the quantized instance normalization op on a random quantized
    # tensor.
    def init(self, dims, dtype):
        # [FUNCTION DOC] Args: dims (input shape, second entry is the channel count), dtype
        # (quantized dtype). Builds a random float input in roughly [-128, 128), quantizes it
        # with scale 1.0 and zero point 0, and fills self.inputs with the quantized tensor,
        # random per-channel float weight and bias, eps 1e-5, and output scale 0.1 and zero
        # point 0.
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        self.inputs = {
            "qX": torch.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        # [FUNCTION DOC] Calls the quantized instance_norm op with the input, weight, bias, eps
        # and the requested output scale and zero point. Returns the quantized output tensor.
        return torch.ops.quantized.instance_norm(
            qX,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


op_bench.generate_pt_test(instancenorm_configs_short, QInstanceNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
