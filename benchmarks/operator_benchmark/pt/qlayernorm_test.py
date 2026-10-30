# [MODULE DOC] Microbenchmark for the quantized layer_norm operator. Defines a short config set
# (several 3D and 4D shapes, qint8) and one benchmark class registered at import time. Running
# as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized layernorm operator."""

layernorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (1, 8, 16),
        (8, 8, 16),
        (32, 8, 16),
        (64, 128, 56, 56),
    ),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QLayerNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of the quantized layer normalization op on a random quantized
    # tensor.
    def init(self, dims, dtype):
        # [FUNCTION DOC] Args: dims (input shape), dtype (quantized dtype). Builds a random
        # float input in roughly [-128, 128) and quantizes it with scale 1.0 and zero point 0.
        # The normalized shape is every dimension after the first, and the random float weight
        # and bias use that shape. Fills self.inputs with the quantized tensor, weight, bias,
        # eps 1e-5, and output scale 0.1 and zero point 0.
        X = (torch.rand(*dims) - 0.5) * 256
        scale = 1.0
        zero_point = 0
        self.qX = torch.quantize_per_tensor(
            X, scale=scale, zero_point=zero_point, dtype=dtype
        )

        self.inputs = {
            "qX": self.qX,
            "weight": torch.rand(*self.qX.size()[1:], dtype=torch.float),
            "bias": torch.rand(*self.qX.size()[1:], dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def forward(self, qX, weight, bias, eps: float, Y_scale: float, Y_zero_point: int):
        # [FUNCTION DOC] Calls the quantized layer_norm op using all dimensions after the first
        # as the normalized shape, plus weight, bias, eps and the requested output scale and
        # zero point. Returns the quantized output tensor.
        return torch.ops.quantized.layer_norm(
            qX,
            qX.size()[1:],
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


op_bench.generate_pt_test(layernorm_configs_short, QLayerNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
