# [MODULE DOC] Benchmarks for 2D pooling (max, average, adaptive average) on quantized tensors.
# Defines long and short config sets (a rank-3 input is selected by a batch size of zero), a
# shared base class and three concrete benchmarks, registered at import time. Running as a
# script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


# 2D pooling will have input matrix of rank 3 or 4
qpool2d_long_configs = op_bench.config_list(
    attrs=(
        #  C    H    W   k       s       p
        (1, 3, 3, (3, 3), (1, 1), (0, 0)),  # dummy        # noqa: E201,E241
        (3, 64, 64, (3, 3), (2, 2), (1, 1)),  # dummy        # noqa: E201,E241
        # VGG16 pools with original input shape: (-1, 3, 224, 224)
        (64, 224, 224, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-4  # noqa: E201
        (256, 56, 56, (2, 2), (2, 2), (0, 0)),  # MaxPool2d-16 # noqa: E241
    ),
    attr_names=("C", "H", "W", "k", "s", "p"),  # Input layout  # Pooling parameters
    cross_product_configs={
        "N": (1, 4),
        "contig": (False, True),
        "dtype": (torch.quint8,),
    },
    tags=("long",),
)

qpool2d_short_configs = op_bench.config_list(
    attrs=((1, 3, 3, (3, 3), (1, 1), (0, 0)),),  # dummy
    attr_names=("C", "H", "W", "k", "s", "p"),  # Input layout  # Pooling parameters
    cross_product_configs={
        "N": (2,),
        "contig": (True,),
        "dtype": (torch.qint32, torch.qint8, torch.quint8),
    },
    tags=("short",),
)

qadaptive_avgpool2d_long_configs = op_bench.cross_product_configs(
    input_size=(
        # VGG16 pools with original input shape: (-1, 3, 224, 224)
        (112, 112),  # MaxPool2d-9
    ),
    output_size=(
        (448, 448),
        # VGG16 pools with original input shape: (-1, 3, 224, 224)
        (224, 224),  # MaxPool2d-4
        (112, 112),  # MaxPool2d-9
        (56, 56),  # MaxPool2d-16 # noqa: E201,E241
        (14, 14),  # MaxPool2d-30 # noqa: E201,E241
    ),
    N=(1, 4),
    C=(1, 3, 64, 128),
    contig=(False, True),
    dtype=(torch.quint8,),
    tags=("long",),
)

qadaptive_avgpool2d_short_configs = op_bench.config_list(
    attrs=((4, 3, (224, 224), (112, 112), True),),
    attr_names=("N", "C", "input_size", "output_size", "contig"),
    cross_product_configs={
        "dtype": (torch.qint32, torch.qint8, torch.quint8),
    },
    tags=("short",),
)


class _QPool2dBenchmarkBase(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] HELPER: Base class for quantized pooling benchmarks. Subclasses set the
    # pool_op module in init and call setup to prepare the quantized input; forward applies
    # pool_op.
    def setup(self, N, C, H, W, dtype, contig):
        # [FUNCTION DOC] HELPER: Args: N (batch, zero means no batch dimension), C, H, W,
        # dtype, contig. Builds a random float tensor in roughly [-128, 128) of shape N x C x H
        # x W (or C x H x W when N is zero) and quantizes it with scale 1.0 and zero point 0.
        # When contig is false the tensor is permuted to channels-last, made contiguous, and
        # permuted back so that it is logically the same but non-contiguous. Stores the result
        # in self.q_input and in self.inputs.
        # Input
        if N == 0:
            f_input = (torch.rand(C, H, W) - 0.5) * 256
        else:
            f_input = (torch.rand(N, C, H, W) - 0.5) * 256

        scale = 1.0
        zero_point = 0

        # Quantize the tensor
        self.q_input = torch.quantize_per_tensor(
            f_input, scale=scale, zero_point=zero_point, dtype=dtype
        )
        if not contig:
            # Permute into NHWC and back to make it non-contiguous
            if N == 0:
                self.q_input = self.q_input.permute(1, 2, 0).contiguous()
                self.q_input = self.q_input.permute(2, 0, 1)
            else:
                self.q_input = self.q_input.permute(0, 2, 3, 1).contiguous()
                self.q_input = self.q_input.permute(0, 3, 1, 2)

        self.inputs = {"q_input": self.q_input}

    def forward(self, q_input):
        # [FUNCTION DOC] HELPER: Applies the pooling module to the quantized input and returns
        # the result.
        return self.pool_op(q_input)


class QMaxPool2dBenchmark(_QPool2dBenchmarkBase):
    # [CLASS DOC] Benchmark of MaxPool2d on a quantized tensor.
    def init(self, N, C, H, W, k, s, p, contig, dtype):
        # [FUNCTION DOC] Args: N, C, H, W, k (kernel), s (stride), p (padding), contig, dtype.
        # Creates a MaxPool2d module with the given kernel, stride and padding, unit dilation,
        # no ceil mode and no indices, then calls the base setup to build the input.
        self.pool_op = torch.nn.MaxPool2d(
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=(1, 1),
            ceil_mode=False,
            return_indices=False,
        )
        super().setup(N, C, H, W, dtype, contig)


class QAvgPool2dBenchmark(_QPool2dBenchmarkBase):
    # [CLASS DOC] Benchmark of AvgPool2d on a quantized tensor.
    def init(self, N, C, H, W, k, s, p, contig, dtype):
        # [FUNCTION DOC] Args: N, C, H, W, k (kernel), s (stride), p (padding), contig, dtype.
        # Creates an AvgPool2d module with the given kernel, stride and padding and no ceil
        # mode, then calls the base setup to build the input.
        self.pool_op = torch.nn.AvgPool2d(
            kernel_size=k, stride=s, padding=p, ceil_mode=False
        )
        super().setup(N, C, H, W, dtype, contig)


class QAdaptiveAvgPool2dBenchmark(_QPool2dBenchmarkBase):
    # [CLASS DOC] Benchmark of AdaptiveAvgPool2d on a quantized tensor.
    def init(self, N, C, input_size, output_size, contig, dtype):
        # [FUNCTION DOC] Args: N, C, input_size (height and width pair), output_size, contig,
        # dtype. Creates an AdaptiveAvgPool2d module with the target output size, then calls
        # the base setup with the unpacked input height and width to build the input.
        self.pool_op = torch.nn.AdaptiveAvgPool2d(output_size=output_size)
        super().setup(N, C, *input_size, dtype=dtype, contig=contig)


op_bench.generate_pt_test(
    qadaptive_avgpool2d_short_configs + qadaptive_avgpool2d_long_configs,
    QAdaptiveAvgPool2dBenchmark,
)
op_bench.generate_pt_test(
    qpool2d_short_configs + qpool2d_long_configs, QAvgPool2dBenchmark
)
op_bench.generate_pt_test(
    qpool2d_short_configs + qpool2d_long_configs, QMaxPool2dBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
