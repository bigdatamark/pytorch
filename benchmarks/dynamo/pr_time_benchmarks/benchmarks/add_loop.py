# [MODULE DOC] PR-time benchmark that measures compile-time instruction count of torch.compile
# on a function made of a long unrolled loop of add, scaled add and sin operations. Run as a
# script with one argument, the CSV result path; it runs eager, inductor (cpu and gpu) variants,
# static and dynamic shapes, and appends results via the benchmark base class. Requires
# benchmark_base to be importable from the script directory. No import-time side effects
# beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_cache


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark of compiling a loop of elementwise ops on two 1000-element tensors.
    # Configured by backend name, dynamic-shapes flag and whether to use cuda; the category is
    # fixed to add_loop. State: two input tensors created in the one-time setup.
    def __init__(self, backend, dynamic=False, is_gpu=False):
        # [FUNCTION DOC] Builds the benchmark with category add_loop, the given backend, device
        # cuda if is_gpu else cpu, and the dynamic flag. Only forwards to the base
        # class constructor.
        super().__init__(
            category="add_loop",
            backend=backend,
            device="cuda" if is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        # [FUNCTION DOC] Returns the unique result name: category and backend joined by
        # underscore, plus a dynamic suffix when dynamic shapes are on and a gpu suffix when the
        # device is cuda.
        prefix = f"{self.category()}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self.device() == "cuda":
            prefix += "_gpu"
        return prefix

    def description(self):
        # [FUNCTION DOC] Returns a fixed human-readable description string.
        # KNOWN ISSUE: the text says 100 add nodes but the compiled function loops 1000 times.
        return "a loop over 100 add node"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: creates two float tensors of ones with 1000 elements
        # each on the benchmark device and stores them as the inputs.
        self.a = torch.ones(1000, device=self.device())
        self.b = torch.torch.ones(1000, device=self.device())

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state so each measured compile
        # starts cold.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: defines a function compiled with the chosen backend,
        # full graph mode and the dynamic setting, then calls it once on the stored inputs
        # inside a fresh inductor cache context. The function clones the first input and runs
        # 1000 steps cycling through add of b, add of 8 times b, and sin.
        @torch.compile(
            backend=self.backend(),
            fullgraph=True,
            dynamic=self.is_dynamic(),
        )
        def f(a, b):
            # [FUNCTION DOC] HELPER: Compiled workload. Clones a, then for each of 1000 steps
            # applies one of three updates depending on the step index modulo 3 (add b, add
            # scaled b, or sin); returns the final tensor.
            result = a.clone()
            for i in range(1000):
                if i % 3 == 0:
                    result = result + b
                elif i % 3 == 1:
                    result = result + 8 * b
                else:
                    result = result.sin()
            return result

        with fresh_cache():
            f(self.a, self.b)


def main():
    # [FUNCTION DOC] Script entry point. Takes the output CSV path from the first command line
    # argument, builds five benchmark variants (eager static, eager dynamic, inductor cpu, inductor
    # gpu, inductor gpu dynamic), and for each enables compile-time instruction counting, collects,
    # and appends results to the path.
    result_path = sys.argv[1]
    all = [
        Benchmark("eager"),
        Benchmark("eager", dynamic=True),
        Benchmark("inductor"),
        Benchmark("inductor", is_gpu=True),
        Benchmark("inductor", is_gpu=True, dynamic=True),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
