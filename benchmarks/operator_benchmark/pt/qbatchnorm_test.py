# [MODULE DOC] Operator microbenchmarks for the quantized batch_norm1d and batch_norm2d ops on a
# qint8 per-tensor quantized input, CPU only. A shared base class quantizes the input and random
# statistics; subclasses choose the input rank and the op. Tests are registered at import time.
# Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for quantized batchnorm operator."""

batchnorm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 256, 3136],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": (torch.qint8,),
    },
    tags=["short"],
)


class QBatchNormBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Base class for the quantized batch norm benchmarks. Subclasses implement the
    # input creation hook and the forward. State in the inputs mapping: quantized input, mean, var,
    # weight, bias, eps and output scale and zero point.
    def init(self, M, N, K, device, dtype):
        # [FUNCTION DOC] Call the subclass hook to create the float input as input_one, quantize it
        # per tensor with scale 0.1 and zero point 0 in the dtype, and store it in the inputs
        # mapping with random mean, var, weight and bias vectors of length N (on CPU), epsilon
        # 1e-5, and output scale 0.1 with zero point 0.
        self._init(M, N, K, device)
        x_scale = 0.1
        x_zero_point = 0
        self.inputs = {
            "q_input_one": torch.quantize_per_tensor(
                self.input_one, scale=x_scale, zero_point=x_zero_point, dtype=dtype
            ),
            "mean": torch.rand(N),
            "var": torch.rand(N),
            "weight": torch.rand(N),
            "bias": torch.rand(N),
            "eps": 1e-5,
            "Y_scale": 0.1,
            "Y_zero_point": 0,
        }

    def _init(self, M, N, K, device):
        # [FUNCTION DOC] HELPER: Hook for subclasses to set the module name and create the float
        # input as input_one. The base version does nothing.
        pass

    def forward(self):
        # [FUNCTION DOC] Placeholder forward that does nothing; subclasses override it.
        pass


class QBatchNorm1dBenchmark(QBatchNormBenchmark):
    # [CLASS DOC] Benchmark for quantized batch_norm1d on an MxNxK input (N channels).
    def _init(self, M, N, K, device):
        # [FUNCTION DOC] HELPER: Set the module name to QBatchNorm1d and create a random MxNxK
        # float input as input_one, with requires_grad from auto_set.
        self.set_module_name("QBatchNorm1d")
        self.input_one = torch.rand(
            M, N, K, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        # [FUNCTION DOC] Run the quantized batch_norm1d op on the quantized input with the given
        # weight, bias, running mean and variance, epsilon, and output scale and zero point.
        # Returns the result.
        return torch.ops.quantized.batch_norm1d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


class QBatchNorm2dBenchmark(QBatchNormBenchmark):
    # [CLASS DOC] Benchmark for quantized batch_norm2d on an MxNxKx1 input (N channels).
    def _init(self, M, N, K, device):
        # [FUNCTION DOC] HELPER: Set the module name to QBatchNorm2d and create a random float
        # input of shape MxNxKx1 (the quantized op needs rank 4) as input_one, with requires_grad
        # from auto_set.
        self.set_module_name("QBatchNorm2d")
        # Note: quantized implementation requires rank 4, which is why we
        # add a 1 as the last dimension
        self.input_one = torch.rand(
            M, N, K, 1, device=device, requires_grad=self.auto_set()
        )

    def forward(
        self,
        q_input_one,
        weight,
        bias,
        mean,
        var,
        eps: float,
        Y_scale: float,
        Y_zero_point: int,
    ):
        # [FUNCTION DOC] Run the quantized batch_norm2d op on the quantized input with the given
        # weight, bias, running mean and variance, epsilon, and output scale and zero point.
        # Returns the result.
        return torch.ops.quantized.batch_norm2d(
            q_input_one, weight, bias, mean, var, eps, Y_scale, Y_zero_point
        )


op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm1dBenchmark)
op_bench.generate_pt_test(batchnorm_configs_short, QBatchNorm2dBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
