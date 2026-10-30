# [MODULE DOC] Example benchmark that uses a TorchScript function in a legacy JIT
# forward method to repeatedly sum a tensor. Defines one tiny config (8 by 16, short
# tag), a scripted summation loop and a benchmark class, and registers it at import
# time. When run as a script it calls the benchmark runner main function.
import operator_benchmark as op_bench

import torch


intraop_bench_configs = op_bench.config_list(
    attrs=[
        [8, 16],
    ],
    attr_names=["M", "N"],
    tags=["short"],
)


@torch.jit.script
# [FUNCTION DOC] Scripted with TorchScript. Loops for the given number of
# iterations; each time it adds the sum of the tensor, converted to a float, to a
# running total and then increments the first element of the tensor in place by
# 0.01. Returns the float total. Side effect: mutates its tensor argument.
# KNOWN ISSUE: this block is placed above the def line instead of inside the body because
# the TorchScript type comment must stay on the line right after the def line.
def torch_sumall(a, iterations):
    # type: (Tensor, int)
    result = 0.0
    for _ in range(iterations):
        result += float(torch.sum(a))
        a[0][0] += 0.01
    return result


class TorchSumBenchmark(op_bench.TorchBenchmarkBase):
    # [CLASS DOC] Benchmark op for the sum example with module name sum. It
    # defines a legacy JIT forward method but no forward method.
    # KNOWN ISSUE: because no forward method is defined, the current framework paths
    # (eager, JIT scripting of forward, compile) call the Module default forward and
    # would fail; the legacy method is not called by the current runner.
    def init(self, M, N):
        # [FUNCTION DOC] Creates a random matrix of the given size
        # as the input and sets the module name to sum.
        self.input_one = torch.rand(M, N)
        self.set_module_name("sum")

    # This is a very temporary method and will be removed soon, so
    # don't use this method in your benchmark
    # TODO(mingzhe): use one forward method for both JIT and Eager
    def jit_forward(self, iters):
        # [FUNCTION DOC] Legacy temporary method (per its comment, to be removed)
        # that runs the scripted summation loop on the stored input for the given
        # number of iterations and returns the float total.
        return torch_sumall(self.input_one, iters)


op_bench.generate_pt_test(intraop_bench_configs, TorchSumBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
