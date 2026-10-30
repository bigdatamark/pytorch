# [MODULE DOC] Operator microbenchmarks for quantized binary arithmetic: add, add_relu and mul on
# two tensors, and add_scalar and mul_scalar with an integer scalar, using ops from the quantized
# namespace. Square inputs of several sizes in quint8, qint8 and qint32, contiguous or not. Tests
# are registered at import time. Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch
from torch._ops import ops


qarithmetic_binary_configs = op_bench.cross_product_configs(
    N=(2, 8, 64, 512),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    contig=(False, True),
    tags=("short",),
)


qarithmetic_binary_ops = op_bench.op_list(
    attrs=(
        ("add", ops.quantized.add),
        ("add_relu", ops.quantized.add_relu),
        ("mul", ops.quantized.mul),
    ),
    attr_names=("op_name", "op_func"),
)

qarithmetic_binary_scalar_ops = op_bench.op_list(
    attrs=(
        ("add_scalar", ops.quantized.add_scalar),
        ("mul_scalar", ops.quantized.mul_scalar),
    ),
    attr_names=("op_name", "op_func"),
)


class _QFunctionalBinaryArithmeticBenchmarkBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] HELPER: Base class that prepares a quantized square input shared by the
    # arithmetic benchmark classes. Attributes set by setup: qfunctional, scale, zero_point
    # and q_input_a.
    def setup(self, N, dtype, contig):
        # [FUNCTION DOC] HELPER: Create a QFunctional module (not used afterwards), a random NxN
        # float tensor scaled to roughly minus 128 to 128, and quantize it per tensor with scale
        # 1.0 and zero point 0 in the dtype. If contig is false, reverse the dimension order with a
        # permute so the result is non-contiguous. Stores the quantized tensor in q_input_a.
        self.qfunctional = torch.ao.nn.quantized.QFunctional()

        # TODO: Consider more diverse shapes
        f_input = (torch.rand(N, N) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0
        self.q_input_a = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )

        if not contig:
            permute_dims = list(range(f_input.ndim))[::-1]
            self.q_input_a = self.q_input_a.permute(permute_dims)


class QFunctionalBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    # [CLASS DOC] Benchmark for a quantized two-tensor arithmetic op; both operands are the same
    # quantized tensor. State in the inputs mapping: q_input_a, q_input_b, scale and zero_point.
    def init(self, N, dtype, contig, op_func):
        # [FUNCTION DOC] Prepare the shared input through the base setup, store both operands with
        # the output scale and zero point in the inputs mapping and keep op_func.
        super().setup(N, dtype, contig)
        self.inputs = {
            "q_input_a": self.q_input_a,
            "q_input_b": self.q_input_a,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        self.op_func = op_func

    def forward(self, q_input_a, q_input_b, scale: float, zero_point: int):
        # [FUNCTION DOC] Run the op on the two quantized tensors with the given output scale and
        # zero point and return the quantized result.
        return self.op_func(q_input_a, q_input_b, scale=scale, zero_point=zero_point)


op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_ops, qarithmetic_binary_configs, QFunctionalBenchmark
)


class QFunctionalScalarBenchmark(_QFunctionalBinaryArithmeticBenchmarkBase):
    # [CLASS DOC] Benchmark for a quantized tensor-plus-scalar arithmetic op. State in the inputs
    # mapping: q_input and a fixed integer scalar.
    def init(self, N, dtype, contig, op_func):
        # [FUNCTION DOC] Prepare the shared input through the base setup, store it with the scalar
        # 42 in the inputs mapping and keep op_func.
        super().setup(N, dtype, contig)
        self.inputs = {"q_input": self.q_input_a, "scalar_input": 42}
        self.op_func = op_func

    def forward(self, q_input, scalar_input: int):
        # [FUNCTION DOC] Run the scalar op on q_input and scalar_input and return the
        # quantized result.
        return self.op_func(q_input, scalar_input)


op_bench.generate_pt_tests_from_op_list(
    qarithmetic_binary_scalar_ops,
    qarithmetic_binary_configs,
    QFunctionalScalarBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
