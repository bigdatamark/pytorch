# [MODULE DOC] Microbenchmark for interpolate applied to quantized tensors. Defines a long
# config set (cross product of dtype, mode and scale on a 512 cube) and a short set (explicit
# shape, mode and scale cases covering downsample and upsample), then registers the benchmark
# at import time. Running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for the quantized interpolate op.

Note: We are not benchmarking `upsample` as it is being deprecated, and calls
the `interpolate` anyway.
"""

qinterpolate_long_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [512, 512, 512],
    ],
    cross_product_configs={
        "dtype": [torch.quint8, torch.qint8, torch.qint32],
        "mode": ["nearest", "bilinear"],
        "scale": [0.5, 1.0, 2.0],
        "contig": [True],  # TODO: Add `False` after #29435
    },
    tags=["long"],
)


qinterpolate_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "dtype", "mode", "scale", "contig"],
    attrs=[
        [32, 32, 32, torch.quint8, "nearest", 0.5, True],  # Downsample
        [32, 32, 32, torch.quint8, "bilinear", 0.5, True],  # Downsample
        [32, 32, 32, torch.quint8, "nearest", 2.0, True],  # Upsample
        [32, 32, 32, torch.quint8, "bilinear", 2.0, True],  # Upsample
        [3, 720, 1280, torch.quint8, "bilinear", 0.83333, True],  # Downsample
    ],
    tags=["short"],
)


class QInterpolateBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of the functional interpolate on a quantized 4D tensor.
    def init(self, M, N, K, dtype, mode, scale, contig):
        # [FUNCTION DOC] Args: M, N, K (the input is shaped 1 x M x N x K), dtype, mode, scale,
        # contig. Builds a random float input, quantizes it with scale 0.1 and zero point 42,
        # and permutes it with reversed dimensions when contig is false. Stores the quantized
        # tensor, the scale factor and the mode in self.inputs and sets the module name.
        # KNOWN ISSUE: the scale argument is overwritten with 0.1 before use, so the configured
        # scale values are ignored and the same value is used both as the quantization scale
        # and as the interpolation scale factor.
        f_input = (torch.rand(1, M, N, K) - 0.5) * 256
        scale = 0.1
        zero_point = 42
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        if not contig:
            permute_dims = list(range(self.q_input.ndim))[::-1]
            self.q_input = self.q_input.permute(permute_dims)

        self.inputs = {"q_input": self.q_input, "scale_factor": scale, "mode": mode}
        self.set_module_name("q_interpolate")

    def forward(self, q_input, scale_factor: float, mode: str):
        # [FUNCTION DOC] Runs the functional interpolate on the quantized input with the given
        # scale factor and mode and returns the resized tensor.
        return torch.nn.functional.interpolate(
            q_input, scale_factor=scale_factor, mode=mode
        )


op_bench.generate_pt_test(
    qinterpolate_short_configs + qinterpolate_long_configs, QInterpolateBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
