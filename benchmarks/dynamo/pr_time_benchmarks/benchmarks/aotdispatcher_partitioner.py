# [MODULE DOC] PR-time benchmark for the min-cut partitioner under the
# aot_eager_decomp_partition backend: one input and 100 weights with a chain of matmul followed
# by two sin ops, mixing recompute and non-recompute ops. Run as a script with one argument, the
# CSV result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with fixed category aotdispatcher_partitioner, backend
    # aot_eager_decomp_partition and cpu device. State: the weight tensors and input created in
    # the one-time setup.
    def __init__(self):
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend and device;
        # takes no arguments.
        super().__init__(
            category="aotdispatcher_partitioner",
            backend="aot_eager_decomp_partition",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category and device joined by underscore.
        return f"{self.category()}_{self.device()}"

    def description(self):
        # [FUNCTION DOC] Returns a fixed description string of the partitioner workload.
        return "partitioner benchmark 1 input and 100 weights, mix of recompute and non-recompute ops"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: creates 100 random 16 by 16 weight tensors requiring
        # grad and one random 16 by 16 input without grad.
        self.weights = [torch.randn(16, 16, requires_grad=True) for _ in range(100)]
        self.inp = torch.randn(16, 16)

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: compiles a function with the benchmark backend in full
        # graph mode and calls it once with the stored input and weights.
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(inp, *weights):
            # [FUNCTION DOC] HELPER: Compiled workload; threads the input through each weight in
            # order, multiplying the weight with the running value and applying sin twice, and
            # returns the final value.
            x = inp
            for w in weights:
                x = torch.matmul(w, x).sin().sin()
            return x

        f(self.inp, *self.weights)


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
