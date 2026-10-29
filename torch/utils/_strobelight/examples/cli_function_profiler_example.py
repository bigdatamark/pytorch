# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Example script showing the strobelight decorator, used either with
# inline profiler options or with an explicit profiler instance. When run as a main program
# it profiles two toy workloads (requires meta-internal strobeclient); no effect on import.
import torch
from torch.utils._strobelight.cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)


if __name__ == "__main__":

    def fn(x, y, z):
        # [FUNCTION DOC] HELPER: Elementwise multiply x by y and add z; returns the result.
        return x * y + z

    # use decorator with default profiler or optional profile arguments.
    @strobelight(sample_each=10000, stop_at_error=False)
    @torch.compile()
    def work() -> None:
        # [FUNCTION DOC] HELPER: Profiled workload: repeatedly resets dynamo and calls fn on
        # random square matrices of sizes 0..4 (compiled), forcing recompilation. Returns None.
        for _ in range(10):
            torch._dynamo.reset()
            for j in range(5):
                torch._dynamo.reset()
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    work()

    # or pass a profiler instance.
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    @strobelight(profiler, sample_tags=["something", "another"])
    def work2() -> None:
        # [FUNCTION DOC] HELPER: Profiled workload: a pure Python busy loop of 100 million
        # additions. Returns None.
        sum = 0
        for _ in range(100000000):
            sum += 1  # noqa: SIM113

    work2()
