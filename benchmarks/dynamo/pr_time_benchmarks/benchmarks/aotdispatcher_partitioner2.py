# [MODULE DOC] Second PR-time partitioner benchmark under aot_eager_decomp_partition, with many
# parallel use chains over one small input (see the GitHub issue referenced in the description).
# Run as a script with one argument, the CSV result path. No import-time side effects
# beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with the same category as the first partitioner benchmark
    # (aotdispatcher_partitioner) but a distinct result name ending in 2. Backend
    # aot_eager_decomp_partition, cpu device. State: one 4 by 4 input created in the
    # one-time setup.
    def __init__(self):
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend and device;
        # takes no arguments.
        super().__init__(
            category="aotdispatcher_partitioner",
            backend="aot_eager_decomp_partition",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category and device joined by underscore,
        # followed by the digit 2.
        return f"{self.category()}_{self.device()}2"

    def description(self):
        # [FUNCTION DOC] Returns a multi-line description string naming the workload and linking
        # an issue.
        return """
Partitioner benchmark with many parallel use chains.
See https://github.com/pytorch/pytorch/issues/145081"""

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: creates a random 4 by 4 tensor requiring grad as
        # the input.
        self.x = torch.randn(4, 4, requires_grad=True)

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: compiles a function with the benchmark backend in full
        # graph mode and calls it once on the stored input.
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled workload. Builds 16 tensors by adding the loop
            # index to x, then adds x to each. For each index except the last four it replaces
            # entry i by its sine times itself and subtracts that from the next three entries in
            # place. Returns the sum of all entries.
            tmps = [x + i for i in range(16)]
            tmps = [x + tmp for tmp in tmps]
            for i in range(len(tmps) - 4):
                tmps[i] = tmps[i].sin().mul(tmps[i])
                tmps[i + 1] -= tmps[i]
                tmps[i + 2] -= tmps[i]
                tmps[i + 3] -= tmps[i]
            return sum(tmps)

        f(self.x)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, builds
    # the single benchmark, collects compile-time instruction counts and appends results to
    # the path.
    result_path = sys.argv[1]
    all = [
        Benchmark(),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
