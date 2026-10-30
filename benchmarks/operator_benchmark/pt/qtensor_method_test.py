# [MODULE DOC] Benchmark for quantized tensor methods, currently only an in-place copy. Defines
# short and long config sets (shape, dtype, contiguity), a base class and one concrete
# benchmark, registered at import time. Running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


# Configs for pointwise and reduction unary ops
qmethods_configs_short = op_bench.config_list(
    attr_names=["M", "N"],
    attrs=[
        [32, 32],
    ],
    cross_product_configs={
        "dtype": [torch.quint8],
        "contig": [False, True],
    },
    tags=["short"],
)

qmethods_configs_long = op_bench.cross_product_configs(
    M=[256, 1024],
    N=[256, 1024],
    dtype=[torch.qint8, torch.qint32],
    contig=[False, True],
    tags=["long"],
)


class _QMethodBenchmarkBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] HELPER: Base class that prepares a quantized input tensor. It defines no
    # forward, which concrete subclasses supply.
    def init(self, M, N, dtype, contig):
        # [FUNCTION DOC] HELPER: Args: M, N (shape), dtype, contig. Quantizes a random float
        # tensor with scale 1.0 and zero point 0. When contig is false the dimensions are
        # reversed by a permute so the tensor is non-contiguous. Stores it in self.q_input and
        # self.inputs.
        f_input = torch.rand(M, N)
        scale = 1.0
        zero_point = 0
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        if not contig:
            permute_dims = list(range(self.q_input.ndim))[::-1]
            self.q_input = self.q_input.permute(permute_dims)

        self.inputs = {
            "q_input": self.q_input,
        }


class QMethodTensorInputCopyBenchmark(_QMethodBenchmarkBase):
    # [CLASS DOC] Benchmark of the in-place copy method of a quantized tensor.
    def forward(self, q_input):
        # [FUNCTION DOC] Copies the quantized input onto itself in place and returns the same
        # tensor.
        return q_input.copy_(q_input)


op_bench.generate_pt_test(
    qmethods_configs_short + qmethods_configs_long, QMethodTensorInputCopyBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
