# mypy: allow-untyped-defs
# [MODULE DOC] Example script showing the Strobelight compile time profiler. Only when run as a
# program it enables the profiler, sets a frame id filter selecting frames whose id starts with
# one, compiles a function repeatedly and a second function once. Nothing happens on import.
import torch
from torch._strobelight.compile_time_profiler import StrobelightCompileTimeProfiler


if __name__ == "__main__":
    # You can pass TORCH_COMPILE_STROBELIGHT=True instead.
    StrobelightCompileTimeProfiler.enable()

    # You can use the code below to filter what frames to be profiled.
    StrobelightCompileTimeProfiler.frame_id_filter = "1/.*"
    # StrobelightCompileTimeProfiler.frame_id_filter='0/.*'
    # StrobelightCompileTimeProfiler.frame_id_filter='.*'
    # You can set env variable COMPILE_STROBELIGHT_FRAME_FILTER to set the filter also.

    def fn(x, y, z):
        # [FUNCTION DOC] HELPER: Elementwise multiply of the first two arguments plus the third;
        # the workload run inside the compiled function.
        return x * y + z

    @torch.compile()
    def work(n):
        # [FUNCTION DOC] Compiled workload: runs nested loops (three by five) calling the helper
        # on three random square tensors of the given size. Each distinct size triggers a
        # recompile, which the example uses to produce several frames to profile.
        for _ in range(3):
            for _ in range(5):
                fn(torch.rand(n, n), torch.rand(n, n), torch.rand(n, n))

    # Strobelight will be called only 3 times because dynamo will be disabled after
    # 3rd iteration.
    # Frame 0/0
    for i in range(3):
        torch._dynamo.reset()
        work(i)

    @torch.compile(fullgraph=True)
    def func4(x):
        # [FUNCTION DOC] Compiled with the full graph requirement; returns the elementwise square
        # of the input. Called once so it appears as a separate profiled frame.
        return x * x

    # Frame 1/0
    func4(torch.rand(10))
