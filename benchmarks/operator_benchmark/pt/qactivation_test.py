# [MODULE DOC] Operator microbenchmarks for quantized activations. Long configs use VGG-16 ReLU
# shapes (with and without a batch of 16) in quint8; short configs use small shapes in quint8,
# qint8 and qint32. One benchmark class calls activations taking only the tensor (relu, relu6,
# hardtanh, hardsigmoid, leaky_relu, sigmoid, tanh); another calls ones needing scale and zero
# point (hardswish, elu, celu). Tests are registered at import time. Running as a script starts the
# op_bench runner.
import operator_benchmark as op_bench

import torch
import torch.ao.nn.quantized.functional as qF


r"""Microbenchmarks for the quantized activations."""

qactivation_long_configs = op_bench.cross_product_configs(
    dims=(
        # VGG-16 relu's with original shape: (-1, 3, 224, 224)
        (64, 224, 224),  # ReLU-1   # noqa: E201
        (128, 112, 112),  # ReLU-6
        (256, 56, 56),  # ReLU-11  # noqa: E241
        (512, 28, 28),  # ReLU-18  # noqa: E241
        (512, 14, 14),  # ReLU-25  # noqa: E241
        # Batch = 16
        (16, 64, 224, 224),  # ReLU-1   # noqa: E241
        (16, 128, 112, 112),  # ReLU-6
        (16, 256, 56, 56),  # ReLU-11  # noqa: E241
        (16, 512, 28, 28),  # ReLU-18  # noqa: E241
        (16, 512, 14, 14),  # ReLU-25  # noqa: E241
    ),
    contig=(False, True),
    inplace=(False, True),
    dtype=(torch.quint8,),
    tags=("long",),
)

qactivation_short_configs = op_bench.cross_product_configs(
    dims=(
        (3, 4, 5),  # Rank=3
        (2, 3, 4, 5),  # Rank=4,
        # Dimensions from the floating point benchmarks
        (512, 512),
        (256, 1024),
    ),
    contig=(False,),
    inplace=(False,),
    dtype=(torch.quint8, torch.qint8, torch.qint32),
    tags=("short",),
)

qactivation_ops = op_bench.op_list(
    attrs=(
        ("relu", torch.nn.ReLU()),
        ("relu6", torch.ops.quantized.relu6),
        ("functional.hardtanh", qF.hardtanh),
        ("functional.hardsigmoid", qF.hardsigmoid),
        ("functional.leaky_relu", qF.leaky_relu),
        ("functional.sigmoid", torch.nn.functional.sigmoid),
        ("functional.tanh", torch.nn.functional.tanh),
    ),
    attr_names=("op_name", "op_func"),
)


class QActivationBenchmarkBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Base class for the quantized activation benchmarks. It builds a per-tensor
    # quantized input and stores the activation callable in qop; subclasses define forward.
    r"""Base class for all the activations."""

    def _setup(self, dims, contig, dtype):
        # [FUNCTION DOC] HELPER: Create a random float tensor of the given dims scaled to roughly
        # minus 128 to 128, quantize it per tensor with scale 1.0 and zero point 0 in the given
        # dtype (both kept as attributes), and if contig is false reverse the dimension order with
        # a permute so the tensor is non-contiguous. Stores it as q_input in the inputs mapping.
        # Input
        f_input = (torch.rand(*dims) - 0.5) * 256
        self.scale = 1.0
        self.zero_point = 0

        # Quantize the tensor
        q_input = torch.quantize_per_tensor(
            f_input, scale=self.scale, zero_point=self.zero_point, dtype=dtype
        )
        if not contig:
            # Make non-contiguous
            new_shape = list(range(q_input.ndim))[::-1]
            q_input = q_input.permute(new_shape)

        self.inputs = {"q_input": q_input}

    def init(self, dims, contig, inplace, dtype, op_func):
        # [FUNCTION DOC] Build the quantized input through the setup helper and keep op_func as the
        # activation callable.
        # KNOWN ISSUE: the inplace config value is accepted but never used, so inplace true and
        # false configs produce identical benchmarks.
        self._setup(dims, contig, dtype)
        self.qop = op_func


class QActivationBenchmark(QActivationBenchmarkBase):
    # [CLASS DOC] Benchmark for activations that take only the quantized tensor.
    def forward(self, q_input):
        # [FUNCTION DOC] Apply the activation to q_input and return the quantized result.
        return self.qop(q_input)


op_bench.generate_pt_tests_from_op_list(
    qactivation_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationBenchmark,
)


qactivation_scale_zero_point_ops = op_bench.op_list(
    attrs=(
        ("functional.hardswish", qF.hardswish),
        ("functional.elu", qF.elu),
        ("functional.celu", qF.celu),
    ),
    attr_names=("op_name", "op_func"),
)


class QActivationScaleZeroPointBenchmark(QActivationBenchmarkBase):
    # [CLASS DOC] Benchmark for activations that also take explicit output scale and zero point,
    # reusing the values chosen when the input was quantized.
    def forward(self, q_input):
        # [FUNCTION DOC] Apply the activation to q_input with the stored scale and zero point and
        # return the quantized result.
        return self.qop(q_input, scale=self.scale, zero_point=self.zero_point)


op_bench.generate_pt_tests_from_op_list(
    qactivation_scale_zero_point_ops,
    qactivation_short_configs + qactivation_long_configs,
    QActivationScaleZeroPointBenchmark,
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
