# [MODULE DOC] PR-time benchmark measuring raw instruction count (not compile time) of calling a
# custom autograd Function with 13 inputs, 2 outputs and no saved tensors. Requires cuda.
# Defines a module-level output holder shared with the Function, the Function class, and the
# benchmark. Run as a script with one argument, the CSV result path.
import sys

from benchmark_base import BenchmarkBase

import torch


_OUTPUTS = None


class NoSave(torch.autograd.Function):
    # [CLASS DOC] Custom autograd Function whose forward ignores its inputs and returns the
    # pre-built module-level tuple of two empty tensors; backward always raises. Saves nothing
    # on the context.
    @staticmethod
    def forward(ctx, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13):
        # [FUNCTION DOC] Returns the module-level output tuple regardless of the 13 input
        # tensors. Relies on that global having been set by the benchmark setup.
        return _OUTPUTS

    @staticmethod
    def backward(ctx, *grad_outputs):
        # [FUNCTION DOC] Always raises RuntimeError stating this is a forward-only benchmark.
        raise RuntimeError("forward-only benchmark")


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark with category autograd_function_apply on cuda. State: a tuple of 13
    # empty tensors requiring grad, built in the one-time setup. Measures plain
    # instruction count.
    def __init__(self):
        # [FUNCTION DOC] Calls the base constructor with the fixed category and the cuda device.
        super().__init__(category="autograd_function_apply", device="cuda")

    def name(self):
        # [FUNCTION DOC] Returns the result name: category plus a fixed suffix describing 13
        # inputs, 2 outputs, no saved tensors.
        return f"{self.category()}_13_in_2_out_no_save"

    def description(self):
        # [FUNCTION DOC] Returns a fixed description string.
        return "autograd.Function.apply with 13 inputs, 2 outputs, and no saved tensors"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: sets the module-level output tuple to two empty tensors
        # on the device, creates 13 empty tensors requiring grad as arguments, warms up by
        # applying the Function 10000 times, then synchronizes cuda.
        global _OUTPUTS
        _OUTPUTS = (
            torch.empty((0,), device=self.device()),
            torch.empty((0,), device=self.device()),
        )
        self._args = tuple(
            torch.empty((0,), device=self.device(), requires_grad=True)
            for _ in range(13)
        )
        for _ in range(10000):
            NoSave.apply(*self._args)
        torch.cuda.synchronize()

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: does nothing.
        pass

    def _work(self):
        # [FUNCTION DOC] Measured work: one call to the Function apply with the 13
        # stored tensors.
        NoSave.apply(*self._args)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument and runs
    # the benchmark with plain instruction counting, appending results to the path.
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
