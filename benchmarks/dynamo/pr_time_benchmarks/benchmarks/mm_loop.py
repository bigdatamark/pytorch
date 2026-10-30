# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of inductor with max
# autotune on a chain of matrix multiplications, static and dynamic shape variants. Requires
# cuda. Run as a script with one argument, the CSV result path. No import-time side effects
# beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch._inductor.utils import fresh_cache


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with fixed category mm_loop, inductor backend and cuda device,
    # parameterized by the dynamic flag. State: two 10 by 10 input tensors.
    def __init__(self, is_dynamic: bool) -> None:
        # [FUNCTION DOC] Calls the base constructor with the fixed category, backend and device,
        # and the dynamic flag.
        super().__init__(
            category="mm_loop",
            backend="inductor",
            device="cuda",
            dynamic=is_dynamic,
        )

    def name(self) -> str:
        # [FUNCTION DOC] Returns the result name: category and backend joined by underscore,
        # with a dynamic suffix when dynamic and a gpu suffix when the device is cuda (always
        # the case here).
        prefix = f"{self.category()}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self.device() == "cuda":
            prefix += "_gpu"
        return prefix

    def description(self) -> str:
        # [FUNCTION DOC] Returns a fixed description string.
        # KNOWN ISSUE: the text says 100 matmuls but the compiled function performs 201 (one
        # plus a loop of 200).
        return "a mm 100 times in a loop with max auto tune on"

    def _prepare_once(self) -> None:
        # [FUNCTION DOC] One-time setup: creates two 10 by 10 ones tensors on the device.
        self.a = torch.ones(10, 10, device=self.device())
        self.b = torch.torch.ones(10, 10, device=self.device())

    def _prepare(self) -> None:
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: compiles the function with inductor in full graph mode
        # and the dynamic setting, then calls it once on the inputs inside a fresh inductor
        # cache with the max_autotune config enabled.
        @torch.compile(
            backend="inductor",
            fullgraph=True,
            dynamic=self._dynamic,
        )
        def f(a, b):
            # [FUNCTION DOC] HELPER: Compiled workload; multiplies a and b, then multiplies the
            # running result by b 200 more times and returns it.
            z = torch.mm(a, b)
            for i in range(200):
                z = torch.mm(z, b)
            return z

        with fresh_cache(), torch._inductor.config.patch(max_autotune=True):
            f(self.a, self.b)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument
    # and runs the static and dynamic benchmarks with compile-time instruction counting,
    # appending results.
    result_path = sys.argv[1]
    all_benchamrks = [Benchmark(False), Benchmark(True)]
    for b in all_benchamrks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
