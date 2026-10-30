# mypy: allow-untyped-defs
# [MODULE DOC] Example script showing the Strobelight command line function profiler. Only when
# run as a program it profiles a compiled function using the decorator with explicit arguments and
# a busy loop function using an explicitly constructed profiler instance with sample tags.
# Requires Strobelight tooling to be available; nothing happens on import.
import torch
from torch._strobelight.cli_function_profiler import (
    strobelight,
    StrobelightCLIFunctionProfiler,
)


if __name__ == "__main__":

    def fn(x, y, z):
        # [FUNCTION DOC] HELPER: Elementwise multiply of the first two arguments plus the third;
        # the workload compiled inside the profiled function.
        return x * y + z

    # use decorator with default profiler or optional profile arguments.
    @strobelight(sample_each=10000, stop_at_error=False)
    @torch.compile()
    def work():
        # [FUNCTION DOC] Profiled workload: ten times resets dynamo, and inside each, five times
        # resets dynamo again and calls the helper on three random square tensors of growing size.
        # It is compiled and wrapped by the profiler decorator with a sample interval and
        # stop-at-error disabled. Returns nothing.
        for _ in range(10):
            torch._dynamo.reset()
            for j in range(5):
                torch._dynamo.reset()
                fn(torch.rand(j, j), torch.rand(j, j), torch.rand(j, j))

    work()

    # or pass a profiler instance.
    profiler = StrobelightCLIFunctionProfiler(stop_at_error=False)

    @strobelight(profiler, sample_tags=["something", "another"])
    def work2():
        # [FUNCTION DOC] Profiled workload that spins in a loop of one hundred million increments
        # of a local counter to give the profiler something to sample. Returns nothing. Decorated
        # with a shared profiler instance and sample tags.
        sum = 0
        for _ in range(100000000):
            sum += 1  # noqa: SIM113

    work2()
