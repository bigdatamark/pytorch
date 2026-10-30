# [MODULE DOC] PR-time benchmark measuring compile-time instruction count when a compiled
# function is called with eight different Python float arguments (exercising recompilation or
# float specialization) using inductor on cpu. Run as a script with one argument, the CSV result
# path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_cache


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with fixed category float_args, inductor backend and cpu device. No
    # instance state besides the base class.
    def __init__(self):
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend and device.
        super().__init__(
            category="float_args",
            backend="inductor",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name, equal to the category.
        return f"{self.category()}"

    def description(self):
        # [FUNCTION DOC] Returns a fixed description string.
        return "Benchmark to measure recompilations with float arguments."

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: seeds the global random number generator with 0.
        torch.manual_seed(0)

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: compiles an add function with inductor and, inside a
        # fresh inductor cache, calls it eight times with a small integer range tensor and a
        # float that changes each call (multiples of 2.5).
        @torch.compile(backend="inductor")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Compiled workload; returns the tensor plus the scalar.
            return x + y

        with fresh_cache():
            for i in range(8):
                f(torch.arange(3), i * 2.5)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, runs
    # the benchmark with compile-time instruction counting and appends results.
    result_path = sys.argv[1]
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
