# [MODULE DOC] PR-time benchmark for the update_hint regression (see the linked PR in the
# description): compiles a function that converts a tensor to a list, bounds each entry with
# checks and splits on the list, and measures compile-time instruction count with the inductor
# backend. main appends results to a file given on the command line.
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark in category update_hint using the inductor backend on cpu. The class
    # attribute N is the number of split sizes. After _prepare_once it holds splits and input.
    N = 20

    def __init__(self):
        # [FUNCTION DOC] Initializes the base class with the update_hint category, inductor
        # backend and cpu device.
        super().__init__(
            category="update_hint",
            backend="inductor",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the benchmark name: the category followed by a regression
        # suffix.
        return f"{self.category()}_regression"

    def description(self):
        # [FUNCTION DOC] Returns a pointer to the pull request that describes the regression.
        return "information at https://github.com/pytorch/pytorch/pull/129893"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup. Turns on scalar output capture in dynamo (global config
        # change) and seeds the random generator. Creates N random integer split sizes below ten
        # and a random float input vector whose length is the sum of those sizes.
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(0)

        self.splits = torch.randint(10, (self.N,))
        sz = self.splits.sum().item()
        self.input = torch.randn(sz)

    def _prepare(self):
        # [FUNCTION DOC] Per-measurement setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] The measured work: defines and calls a fullgraph compiled function on
        # the prepared input and splits.
        @torch.compile(fullgraph=True)
        def f(a, b):
            # [FUNCTION DOC] HELPER: Compiled body. Converts the sizes tensor b to a Python
            # list, checks each entry is non-negative and at most N, and returns the input a
            # split into pieces of those sizes.
            xs = b.tolist()
            for x in xs:
                torch._check(x >= 0)
                torch._check(x <= self.N)
            return a.split(xs)

        f(self.input, self.splits)


def main():
    # [FUNCTION DOC] Entry point. Takes the output path from the first command line argument,
    # runs the benchmark with compile-time instruction counting enabled, and appends results
    # there.
    result_path = sys.argv[1]
    Benchmark().enable_compile_time_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
