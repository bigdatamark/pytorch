# [MODULE DOC] Operator benchmarks for in-place binary operators (add, subtract, multiply,
# copy, and divide). Defines small wrapper functions around the in-place tensor methods,
# and builds op lists and config lists twice: once for integer dtypes for add, sub, mul
# and copy, and once for float dtypes for div (division needs float data). The benchmark
# class is defined twice with identical code. Registers forward tests at import time and
# runs the benchmark runner when executed as a script.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for inplace binary operators."""


def add_(in1, in2):
    # [FUNCTION DOC] HELPER: Adds the second tensor into the
    # first in place and returns the first tensor.
    return in1.add_(in2)


def sub_(in1, in2):
    # [FUNCTION DOC] HELPER: Subtracts the second tensor from
    # the first in place and returns the first tensor.
    return in1.sub_(in2)


def div_(in1, in2):
    # [FUNCTION DOC] HELPER: Divides the first tensor by the
    # second in place and returns the first tensor.
    return in1.div_(in2)


def mul_(in1, in2):
    # [FUNCTION DOC] HELPER: Multiplies the first tensor by the
    # second in place and returns the first tensor.
    return in1.mul_(in2)


def copy_(in1, in2):
    # [FUNCTION DOC] HELPER: Copies the second tensor into the first in
    # place (with dtype conversion) and returns the first tensor.
    return in1.copy_(in2)


######
# Benchmark ops performance for inplace add + sub + mul + copy
######
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add_", add_],
        ["sub_", sub_],
        # ["div_",  div_ ], # done separately below because of data type
        ["mul_", mul_],
        ["copy_", copy_],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [torch.int32],
        "dtype_two": [torch.int32],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32],
    dtype_two=[torch.int8, torch.int32],
    tags=["long"],
)


class InpBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark applying an in-place binary function, given by the op list entry, to
    # two random tensors of possibly different dtypes. The first definition is used for the
    # integer-typed configs. Because the op is in place, the first input is modified on every
    # iteration.
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        # [FUNCTION DOC] Creates two random normal tensors of the given sizes
        # on the device, converts them to the two requested dtypes, stores
        # them in the inputs mapping and keeps the op function.
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Applies the stored in-place function
        # to the two inputs and returns its result.
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, InpBinaryOpBenchmark
)


######
# Benchmark ops performance for inplace div
######
# Performing division inplace benchmarks separately, as data needs to be float
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["div_", div_],
    ],
)

binary_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype_one": [torch.float],
        "dtype_two": [torch.float],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.float, torch.float],
    dtype_two=[torch.float, torch.float],
    tags=["long"],
)


class InpBinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Duplicate of the earlier in-place binary benchmark class with identical
    # behavior, redefined for the division registration so the name now refers to this second
    # definition.
    # KNOWN ISSUE: the long config for the division case lists the float dtype twice per
    # dtype argument, which produces duplicate configs and duplicate test names.
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        # [FUNCTION DOC] Same as the earlier definition: creates two random normal
        # tensors of the given sizes on the device, converts them to the requested
        # dtypes, stores them in the inputs mapping and keeps the op function.
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Applies the stored in-place function
        # to the two inputs and returns its result.
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, InpBinaryOpBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
