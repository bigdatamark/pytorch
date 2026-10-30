# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for torch.utils.ThroughputBenchmark. Defines a scripted and an eager two-
# input two-layer network, and a TestCase that runs the benchmark on them (plain, with
# profiling output, and with torch.compile plus inductor cpp_wrapper/freezing in several
# dtypes). No import-time effects.

import torch
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase
from torch.utils import ThroughputBenchmark


class TwoLayerNet(torch.jit.ScriptModule):
    # [CLASS DOC] HELPER: Two-layer network as a torch.jit.ScriptModule; its forward is a
    # script method. Holds linear1 (D_in -> H) and linear2 (2*H -> D_out). Used as a benchmark
    # target.
    def __init__(self, D_in, H, D_out):
        # [FUNCTION DOC] HELPER: Builds the two Linear layers; linear2 takes the concatenation
        # of two hidden vectors, so its input width is twice H. Args: D_in, H, D_out layer
        # sizes.
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    @torch.jit.script_method
    def forward(self, x1, x2):
        # [FUNCTION DOC] HELPER: Applies linear1 with a ReLU-style clamp to both inputs,
        # concatenates the results along dim 1, and returns linear2 applied to that. Args: x1,
        # x2 batches of width D_in.
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TwoLayerNetModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Eager nn.Module version of the same two-input network as
    # TwoLayerNet, with identical layers and forward computation.
    def __init__(self, D_in, H, D_out):
        # [FUNCTION DOC] HELPER: Builds linear1 (D_in -> H) and linear2 (2*H -> D_out) layers.
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    def forward(self, x1, x2):
        # [FUNCTION DOC] HELPER: Runs linear1 plus clamp on both inputs, concatenates on dim
        # 1, and returns linear2 of the result.
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TestThroughputBenchmark(TestCase):
    # [CLASS DOC] Test case exercising ThroughputBenchmark on scripted, eager, profiled and
    # compiled modules. Only checks that benchmark results match direct module calls and that
    # it runs.
    def linear_test(self, Module, profiler_output_path=""):
        # [FUNCTION DOC] HELPER: Shared body for the benchmark tests. Builds the module with
        # sizes 10, 5, 15, batch 8, registers 2 random input pairs (second passed by keyword
        # x2), checks run_once output equals the direct module output with assert_close, then
        # runs a 4-thread benchmark (100 warmup, 1000 iterations, optional profiler output
        # path) and prints the stats. Args: Module class, profiler_output_path (empty string
        # disables profiling).
        D_in = 10
        H = 5
        D_out = 15
        B = 8
        NUM_INPUTS = 2

        module = Module(D_in, H, D_out)

        inputs = []

        for _ in range(NUM_INPUTS):
            inputs.append([torch.randn(B, D_in), torch.randn(B, D_in)])
        bench = ThroughputBenchmark(module)

        for input in inputs:
            # can do both args and kwargs here
            bench.add_input(input[0], x2=input[1])

        for i in range(NUM_INPUTS):
            # or just unpack the list of inputs
            module_result = module(*inputs[i])
            bench_result = bench.run_once(*inputs[i])
            torch.testing.assert_close(bench_result, module_result)

        stats = bench.benchmark(
            num_calling_threads=4,
            num_warmup_iters=100,
            num_iters=1000,
            profiler_output_path=profiler_output_path,
        )

        print(stats)

    def test_script_module(self):
        # [FUNCTION DOC] Runs the shared benchmark check on the scripted module.
        self.linear_test(TwoLayerNet)

    def test_module(self):
        # [FUNCTION DOC] Runs the shared benchmark check on the eager module.
        self.linear_test(TwoLayerNetModule)

    def test_profiling(self):
        # [FUNCTION DOC] Runs the shared benchmark check on the eager module writing profiler
        # output to a temporary file that is cleaned up afterwards.
        with TemporaryFileName() as fname:
            self.linear_test(TwoLayerNetModule, profiler_output_path=fname)

    def linear_with_compile_test(self, Module, dtype):
        # [FUNCTION DOC] HELPER: Benchmark check with torch.compile. Sets global dynamo
        # error_on_recompile and inductor cpp_wrapper and freezing to true (never restored).
        # Runs and compiles the module under no_grad (with CPU autocast when dtype is not
        # float32), then, under autocast for half dtypes, adds an input, compares run_once to
        # the compiled module with assert_close, benchmarks with 4 threads and prints stats.
        # Args: Module class, dtype.
        from contextlib import nullcontext

        from torch._dynamo import config
        from torch._inductor import config as inductor_config

        config.error_on_recompile = True
        inductor_config.cpp_wrapper = True
        inductor_config.freezing = True
        D_in = 10
        H = 5
        D_out = 15
        B = 8

        autocast = dtype != torch.float32
        module = Module(D_in, H, D_out)

        input = (torch.randn(B, D_in), torch.randn(B, D_in))

        with torch.no_grad(), torch.amp.autocast("cpu", enabled=autocast, dtype=dtype):
            torch._dynamo.reset()
            module(*input)
            module = torch.compile(module)
            module(*input)
            module(*input)

        ctx = nullcontext()
        if dtype == torch.float16 or dtype == torch.bfloat16:
            ctx = torch.amp.autocast("cpu", enabled=autocast, dtype=dtype)
        with torch.no_grad(), ctx:
            bench = ThroughputBenchmark(module)
            bench.add_input(*input)

            module_result = module(*input)
            bench_result = bench.run_once(*input)
            torch.testing.assert_close(bench_result, module_result)

            stats = bench.benchmark(
                num_calling_threads=4, num_warmup_iters=100, num_iters=1000
            )

            print(stats)

    def test_compile(self):
        # [FUNCTION DOC] Runs the compile benchmark check for float32, float16 and bfloat16 in
        # turn.
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            self.linear_with_compile_test(TwoLayerNetModule, dtype)


if __name__ == "__main__":
    run_tests()
