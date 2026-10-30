# [MODULE DOC] Operator microbenchmark for nan_to_num and its in-place variant nan_to_num_ on
# random CPU matrices with one NaN planted, over float and double and with or without keeping
# infinities. Tests are registered from the op list at import time. Running as a script starts the
# op_bench runner.
import math

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for torch.nan_to_num / nan_to_num_ operators"""

# Configs for PT torch.nan_to_num / nan_to_num_ operators

nan_to_num_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["nan_to_num", torch.nan_to_num],
        ["nan_to_num_", torch.nan_to_num_],
    ],
)

nan_to_num_long_configs = op_bench.cross_product_configs(
    M=[32, 64, 128],
    N=range(32, 128, 32),
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["long"],
)


nan_to_num_short_configs = op_bench.cross_product_configs(
    M=[16, 64],
    N=[64, 64],
    dtype=[torch.float, torch.double],
    replace_inf=[True, False],
    tags=["short"],
)


class ReplaceNaNBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark for replacing NaN (and optionally infinities) in a random matrix. State
    # in the inputs mapping: input and the replace_inf flag; op_func holds the chosen variant.
    def init(self, M, N, dtype, replace_inf, op_func):
        # [FUNCTION DOC] Create a random MxN tensor of the given dtype and set its first element to
        # NaN. Stores it with replace_inf in the inputs mapping, keeps op_func, and sets the module
        # name to nan_to_num, or to nan_to_num_ for the in-place variant so test names stay unique.
        input = torch.randn(M, N, dtype=dtype)
        input[0][0] = float("nan")
        self.inputs = {"input": input, "replace_inf": replace_inf}
        self.op_func = op_func
        self.set_module_name("nan_to_num")

        #  To make casename unique as nan_to_num and nan_to_num_ are two different functions.
        if op_func is torch.nan_to_num_:
            self.set_module_name("nan_to_num_")

    def forward(self, input, replace_inf: bool):
        # [FUNCTION DOC] Run the op with NaN replaced by 1.0. If replace_inf is true, infinities
        # take the default replacements; otherwise posinf and neginf are passed as the infinities
        # themselves, so they are kept. The in-place variant mutates input. Returns the result.
        # compare inplace
        if replace_inf:
            return self.op_func(input, nan=1.0)
        else:
            return self.op_func(input, nan=1.0, posinf=math.inf, neginf=-math.inf)


op_bench.generate_pt_tests_from_op_list(
    nan_to_num_ops_list,
    nan_to_num_long_configs + nan_to_num_short_configs,
    ReplaceNaNBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
