# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of a compiled function
# that converts a tensor of 200 integers to a Python list and sums it, either with the builtin
# sum or an explicit loop. Run as a script with one argument, the CSV result path. No
# import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with category symint_sum, inductor backend label and cpu device.
    # Class constant N sets the list length to 200. State: the loop flag and the random
    # integer tensor.
    N = 200

    def __init__(self, use_loop=False):
        # [FUNCTION DOC] Stores the loop flag and calls the base constructor with the fixed
        # category, backend and device.
        self.use_loop = use_loop
        super().__init__(
            category="symint_sum",
            backend="inductor",
            device="cpu",
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category, with a loop suffix when the loop
        # variant is used.
        if self.use_loop:
            return f"{self.category()}_loop"

        return self.category()

    def description(self):
        # [FUNCTION DOC] Returns a description that is a link to a design document.
        return "see https://docs.google.com/document/d/11xJXl1etSmefUxPiVyk885e0Dl-4o7QwxYcPiMIo2iY/edit"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: turns on the global dynamo option to capture scalar
        # outputs (never restored), seeds the random generator with 0, and creates 200 random
        # integers in the range zero to nine.
        torch._dynamo.config.capture_scalar_outputs = True
        torch.manual_seed(0)

        self.splits = torch.randint(10, (self.N,))

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: compiles a function in full graph mode (default backend)
        # and calls it once on the stored tensor.
        @torch.compile(fullgraph=True)
        def f(a):
            # [FUNCTION DOC] HELPER: Compiled workload; converts the tensor to a list,
            # accumulates its sum either with a loop or the builtin sum depending on the loop
            # flag, and returns the result as a tensor.
            xs = a.tolist()
            y = 0
            if self.use_loop:
                for i in xs:
                    y += i
            else:
                y = sum(xs)
            return torch.tensor(y)

        f(self.splits)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument
    # and runs the builtin-sum and loop variants with compile-time instruction counting,
    # appending results.
    result_path = sys.argv[1]
    Benchmark(
        use_loop=False
    ).enable_compile_time_instruction_count().collect_all().append_results(result_path)
    Benchmark(
        use_loop=True
    ).enable_compile_time_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
