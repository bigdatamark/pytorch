# [MODULE DOC] Example benchmark showing how to register several operators from an op list:
# pointwise unary ops abs and acos on a 128 by 128 random tensor under the short tag. Registers
# at import time and, when run as a script, calls the benchmark runner main function.
import operator_benchmark as op_bench

import torch


# Configs for pointwise unary ops
unary_ops_configs = op_bench.config_list(
    attrs=[
        [128, 128],
    ],
    attr_names=["M", "N"],
    tags=["short"],
)


unary_ops_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["abs", torch.abs],
        ["acos", torch.acos],
    ],
)


class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark op that applies a pointwise function, supplied by the op list entry,
    # to one random matrix. The framework sets the displayed module name from the op list entry.
    def init(self, M, N, op_func):
        # [FUNCTION DOC] Creates the input as a random matrix of the given rows
        # and columns and stores the unary function to benchmark.
        self.input_one = torch.rand(M, N)
        self.op_func = op_func

    def forward(self):
        # [FUNCTION DOC] Applies the stored function to the stored input and returns the result.
        return self.op_func(self.input_one)


op_bench.generate_pt_tests_from_op_list(
    unary_ops_list, unary_ops_configs, UnaryOpBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
