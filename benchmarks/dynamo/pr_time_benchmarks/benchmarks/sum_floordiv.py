# [MODULE DOC] PR-time regression benchmark (linked GitHub issue) measuring compile-time
# instruction count of exporting a module that sums 100 scalar items and floor-divides the
# total, using the new experimental export tracer. Run as a script with one argument, the CSV
# result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch._dynamo.utils import CompileTimeInstructionCounter


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with category sum_floordiv, backend label export and cpu
    # device. Class constant N sets the input length to 100. State: the module and a list
    # of scalar tensors.
    N = 100

    def __init__(self):
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend label
        # and device.
        super().__init__(category="sum_floordiv", backend="export", device="cpu")

    def name(self):
        # [FUNCTION DOC] Returns the result name: category followed by the suffix regression.
        return f"{self.category()}_regression"

    def description(self):
        # [FUNCTION DOC] Returns a description pointing at the tracking issue URL.
        return "information at https://github.com/pytorch/pytorch/issues/134133"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: defines a module whose forward sums the Python scalars
        # of every tensor in the input list and returns the floor division of that sum by 2,
        # instantiates it, and builds a list of 100 scalar integer tensors with values starting
        # at 2.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that sums the scalar items of the input list and
            # floor-divides by 2.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Converts each tensor to a Python number, sums them, and
                # returns the total floor-divided by 2.
                total = sum(t.item() for t in x)
                return total // 2

        self.m = M()
        self.input = [torch.tensor(i + 2) for i in range(self.N)]

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: under patched configs (C++ symbolic shape guards off,
        # new experimental export tracer on) and inside the compile-time instruction counter
        # record context, runs strict export of the module on the input list.
        # enable_cpp_symbolic_shape_guards has impact on this benchmark
        # Keep using False value for consistency.
        with (
            torch._dynamo.config.patch("enable_cpp_symbolic_shape_guards", False),
            torch._export.config.patch(use_new_tracer_experimental=True),
            CompileTimeInstructionCounter.record(),
        ):
            torch.export.export(self.m, (self.input,), strict=True)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, runs
    # the benchmark with compile-time instruction counting and appends results.
    result_path = sys.argv[1]
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
