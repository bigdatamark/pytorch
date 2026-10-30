# [MODULE DOC] Microbenchmark for dtype conversion of tensors through the to method. Config
# sets cross ten source dtypes with ten target dtypes (bool, integer, half, bfloat16, float,
# double) on cpu and cuda, in a short and a long shape group. Registers the benchmarks at
# import time; running as a script starts the benchmark runner.
import operator_benchmark as op_bench

import torch


tensor_conversion_short_configs = op_bench.cross_product_configs(
    M=[32],
    N=[128],
    device=["cpu", "cuda"],
    dtype_one=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    dtype_two=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    tags=["short"],
)

tensor_conversion_long_configs = op_bench.cross_product_configs(
    M=[1024],
    N=[1024],
    device=["cpu", "cuda"],
    dtype_one=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    dtype_two=[
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.bfloat16,
        torch.float,
        torch.double,
    ],
    tags=["long"],
)


class TensorConversionBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark of converting a 2D tensor from one dtype to another.
    def init(self, M, N, dtype_one, dtype_two, device):
        # [FUNCTION DOC] Args: M, N (shape), dtype_one (source), dtype_two (target), device.
        # Creates a random float tensor with no gradient on the device and casts it to the
        # source dtype. Stores it in self.inputs and keeps both dtypes.
        self.inputs = {
            "input": torch.rand(
                M, N, device=device, requires_grad=False, dtype=torch.float
            ).to(dtype=dtype_one)
        }
        self.dtype_one = dtype_one
        self.dtype_two = dtype_two

    def forward(self, input):
        # [FUNCTION DOC] Converts the input to the target dtype and returns the new tensor.
        return input.to(dtype=self.dtype_two)


op_bench.generate_pt_test(tensor_conversion_short_configs, TensorConversionBenchmark)
op_bench.generate_pt_test(tensor_conversion_long_configs, TensorConversionBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
