# [MODULE DOC] Operator microbenchmarks for batched matrix multiply via einsum, and elementwise
# multiply via einsum or torch.mul, on random 3-D float tensors on CPU and CUDA. Two benchmark
# classes with their own op lists and short and long config sets are registered at import time
# through the op list helper. Running as a script starts the op_bench runner.
import operator_benchmark as op_bench

import torch


"""
Microbenchmarks for batch matrix mult with einsum and torch.bmm.
"""

batch_mm_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [4, 5, 3, 2],
        [32, 25, 20, 30],
        [128, 100, 120, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

batch_mm_configs_long = op_bench.config_list(
    attr_names=["B", "M", "N", "K"],
    attrs=[
        [128, 256, 128, 256],
        [512, 1024, 1024, 512],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["long"],
)

batch_mm_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_bmm", torch.einsum],
        # ["bmm", torch.bmm],
    ],
)


class BatchMatrixMultBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for batched matrix multiply of a BxMxN and a BxNxK random tensor.
    def init(self, B, M, N, K, device, op_func):
        # [FUNCTION DOC] Create random tensors BxMxN and BxNxK on the device, store them in the
        # inputs mapping and keep the op callable in op_func. Returns nothing.
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, N, K, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] If the op callable is einsum, contract the two inputs with a batched
        # matrix multiply subscript spec; otherwise call bmm. Returns the BxMxK result.
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bjk->bik", input_one, input_two)
        else:
            return torch.bmm(input_one, input_two)


"""
Microbenchmarks for element-wise matrix mult with einsum and torch.mul.
"""

batch_elementwise_configs_short = op_bench.config_list(
    attr_names=["B", "M", "N"],
    attrs=[
        [4, 5, 3],
        [32, 25, 20],
        [100, 90, 110],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


batch_elementwise_configs_long = op_bench.cross_product_configs(
    B=[128, 512, 1024],
    M=[128, 512, 1024],
    N=[128, 512, 1024],
    device=["cpu", "cuda"],
    tags=["long"],
)

batch_elementwise_op_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["einsum_elementwise", torch.einsum],
        ["mul", torch.mul],
    ],
)


class BatchElementWiseBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for elementwise multiply of two BxMxN random tensors, using either
    # einsum or torch.mul.
    def init(self, B, M, N, device, op_func):
        # [FUNCTION DOC] Create two random BxMxN tensors on the device, store them in the inputs
        # mapping and keep the op callable in op_func. Returns nothing.
        self.inputs = {
            "input_one": torch.rand(B, M, N, device=device),
            "input_two": torch.rand(B, M, N, device=device),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] If the op callable is einsum, compute the elementwise product via an
        # einsum subscript spec; otherwise call torch.mul. Returns the BxMxN result.
        if self.op_func.__name__ == "einsum":
            return torch.einsum("bij,bij->bij", input_one, input_two)
        else:
            return torch.mul(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    batch_mm_op_list,
    batch_mm_configs_short + batch_mm_configs_long,
    BatchMatrixMultBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    batch_elementwise_op_list,
    batch_elementwise_configs_short + batch_elementwise_configs_long,
    BatchElementWiseBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
