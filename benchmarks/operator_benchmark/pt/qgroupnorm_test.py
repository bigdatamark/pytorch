# [MODULE DOC] Microbenchmark for the quantized group_norm operator. Defines a short config set
# (3D and 4D shapes, group counts 2 and 4, qint8) and one benchmark class registered at import
# time. Running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized groupnorm operator."""

groupnorm_configs_short = op_bench.cross_product_configs(
    dims=(
        (32, 8, 16),
        (32, 8, 56, 56),
    ),
    num_groups=(2, 4),
    dtype=(torch.qint8,),
    tags=["short"],
)


class QGroupNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of the quantized group normalization op on a random quantized
    # tensor.
    def init(self, dims, num_groups, dtype):
        # [FUNCTION DOC] Args: dims (input shape, second entry is the channel count),
        # num_groups, dtype (quantized dtype). Builds a random float input in roughly [-128,
        # 128), quantizes it with scale 1.0 and zero point 0, and fills self.inputs with the
        # quantized tensor, the group count, random per-channel float weight and bias, eps
        # 1e-5, and output scale 0.1 and zero point 0.
        X = (torch.rand(*dims) - 0.5) * 256
        num_channels = dims[1]
        scale = 1.0
        zero_point = 0

        self.inputs = {
            "qX": torch.quantize_per_tensor(
                X, scale=scale, zero_point=zero_point, dtype=dtype
            ),
            "num_groups": num_groups,
            "weight": torch.rand(num_channels, dtype=torch.float),
            "bias": torch.rand(num_channels, dtype=torch.float),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def forward(
        self,
        qX,
        num_groups: int,
        weight,
        bias,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        # [FUNCTION DOC] Calls the quantized group_norm op with the input, group count, weight,
        # bias, eps and the requested output scale and zero point. Returns the quantized output
        # tensor.
        return torch.ops.quantized.group_norm(
            qX,
            num_groups,
            weight=weight,
            bias=bias,
            eps=eps,
            output_scale=Y_scale,
            output_zero_point=Y_zero_point,
        )


op_bench.generate_pt_test(groupnorm_configs_short, QGroupNormBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
