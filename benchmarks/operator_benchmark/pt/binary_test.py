# [MODULE DOC] Operator microbenchmarks for elementwise binary torch operators. Defines operator
# lists and config lists, then registers benchmark tests through op_bench helpers at import time.
# Four groups: broadcasting numeric ops (add, sub, div, mul on shape 64x1x64 against 1x64x1, CPU,
# float/bfloat16/float64), non-broadcast numeric and bitwise ops on MxNxK int tensors (CPU/CUDA,
# short and long configs), and the same two layouts for the boolean logical_and op using random
# Bernoulli booleans. The class names BinaryOpBcastBenchmark and BinaryOpBenchmark are each defined
# twice (numeric then boolean); the later definitions shadow the earlier ones, but tests were
# already generated from the first. Running the file as a script starts the op_bench
# benchmark runner.
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for binary operators."""


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
    ],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.float, torch.bfloat16, torch.float64],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a binary op applied to two tensors that broadcast against each
    # other. Holds the input tensors in the inputs mapping and the op callable in op_func.
    def init(self, in_one, in_two, dtype, device, op_func):
        # [FUNCTION DOC] Build two random normal tensors of the shapes in_one and in_two on the
        # given device, cast to dtype, store them in the inputs mapping and keep op_func.
        # Returns nothing.
        self.inputs = {
            "in_one": torch.randn(in_one, device=device).to(dtype=dtype),
            "in_two": torch.randn(in_two, device=device).to(dtype=dtype),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        # [FUNCTION DOC] Run the benchmarked op on the two tensors and return the result.
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["add", torch.add],
        ["sub", torch.sub],
        ["div", torch.div],
        ["mul", torch.mul],
        ["asr", torch.bitwise_right_shift],
        ["lsl", torch.bitwise_left_shift],
        ["xor", torch.bitwise_xor],
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
        "dtype_one": [torch.int32, torch.uint8],
        "dtype_two": [torch.int32, torch.uint8],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.int8, torch.int32, torch.uint8],
    dtype_two=[torch.int8, torch.int32, torch.uint8],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for a binary op on two same-shaped MxNxK tensors with independent
    # dtypes. Holds the input tensors in the inputs mapping and the op callable in op_func.
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        # [FUNCTION DOC] Build two random normal MxNxK tensors on the device, cast to dtype_one and
        # dtype_two respectively, store them in the inputs mapping and keep op_func.
        # Returns nothing.
        self.inputs = {
            "input_one": torch.randn(M, N, K, device=device).to(dtype=dtype_one),
            "input_two": torch.randn(M, N, K, device=device).to(dtype=dtype_two),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Run the benchmarked op on input_one and input_two and return the result.
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


######
# Benchmark ops performance for boolean dtype
######


# Benchmark ops performance with broadcast
binary_ops_bcast_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
)

# Configs with broadcast
binary_configs_broadcast = op_bench.config_list(
    attr_names=["in_one", "in_two"],
    attrs=[
        [[64, 1, 64], [1, 64, 1]],
    ],
    cross_product_configs={
        "device": ["cpu"],
        "dtype": [torch.bool],
    },
    tags=["short"],
)


class BinaryOpBcastBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Boolean-dtype variant of the broadcast benchmark; redefines the earlier class of
    # the same name. Inputs are random booleans rather than normal samples.
    def init(self, in_one, in_two, dtype, device, op_func):
        # [FUNCTION DOC] Build two tensors of shapes in_one and in_two by sampling Bernoulli with
        # probability one half on the device and casting to dtype (bool in the configs). Stores
        # them in the inputs mapping and keeps op_func. Returns nothing.
        self.inputs = {
            "in_one": torch.bernoulli(0.5 * torch.ones(in_one, device=device)).to(
                dtype=dtype
            ),
            "in_two": torch.bernoulli(0.5 * torch.ones(in_two, device=device)).to(
                dtype=dtype
            ),
        }
        self.op_func = op_func

    def forward(self, in_one, in_two):
        # [FUNCTION DOC] Run the benchmarked op on the two boolean tensors and return the result.
        return self.op_func(in_one, in_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_bcast_list, binary_configs_broadcast, BinaryOpBcastBenchmark
)


# Benchmark ops performance without broadcast
binary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[["logical_and", torch.logical_and]],
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
        "dtype_one": [torch.bool],
        "dtype_two": [torch.bool],
    },
    tags=["short"],
)

binary_long_configs = op_bench.cross_product_configs(
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype_one=[torch.bool, torch.bool],
    dtype_two=[torch.bool, torch.bool],
    tags=["long"],
)


class BinaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Boolean-dtype variant of the same-shape benchmark; redefines the earlier class of
    # the same name. Inputs are random booleans of shape MxNxK.
    def init(self, M, N, K, device, dtype_one, dtype_two, op_func):
        # [FUNCTION DOC] Build two MxNxK tensors by sampling Bernoulli with probability 0.5 on the
        # device and casting to dtype_one and dtype_two. Stores them in the inputs mapping and
        # keeps op_func. Returns nothing.
        self.inputs = {
            "input_one": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_one
            ),
            "input_two": torch.bernoulli(0.5 * torch.ones(M, N, K, device=device)).to(
                dtype=dtype_two
            ),
        }
        self.op_func = op_func

    def forward(self, input_one, input_two):
        # [FUNCTION DOC] Run the benchmarked op on input_one and input_two and return the result.
        return self.op_func(input_one, input_two)


op_bench.generate_pt_tests_from_op_list(
    binary_ops_list, binary_short_configs + binary_long_configs, BinaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
