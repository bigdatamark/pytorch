# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of dynamo inlining a
# chain of ten nested trivial functions called 1000 times inside a module forward. Run as a
# script with one argument, the CSV result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_cache


# Create a chain of artificial nesting
def fn(x):
    # [FUNCTION DOC] HELPER: Innermost function of the nesting chain; returns its input
    # plus one.
    return x + 1


def fn1(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn.
    return fn(x)


def fn2(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn1.
    return fn1(x)


def fn3(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn2.
    return fn2(x)


def fn4(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn3.
    return fn3(x)


def fn5(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn4.
    return fn4(x)


def fn6(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn5.
    return fn5(x)


def fn7(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn6.
    return fn6(x)


def fn8(x):
    # [FUNCTION DOC] HELPER: Chain link that forwards to fn7.
    return fn7(x)


def fn9(x):
    # [FUNCTION DOC] HELPER: Outermost chain link that forwards to fn8, so each call nests ten
    # function frames.
    return fn8(x)


class InlineMod(nn.Module):
    # [CLASS DOC] Module whose forward calls the ten-deep function chain a fixed number of
    # times. State: the repeat count of 1000.
    def __init__(self):
        # [FUNCTION DOC] Initializes the module and sets the repeat count to 1000.
        super().__init__()
        self._n = 1000

    def forward(self, x):
        # [FUNCTION DOC] Applies the nested function chain to the input 1000 times in a loop and
        # returns the result.
        for _ in range(self._n):
            x = fn9(x)
        return x


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark of compiling a given module class with a backend (default
    # eager). Category is basic. State: module class and name, gpu flag, module instance and
    # a ones input.
    def __init__(
        self,
        ModuleClass,
        backend="eager",
        is_gpu=False,
        dynamic=False,
    ):
        # [FUNCTION DOC] Stores the module class, its name and the gpu flag, then calls the
        # base constructor with category basic, the backend, cuda or cpu device, and the
        # dynamic flag.
        self.ModuleClass = ModuleClass
        self._name = ModuleClass.__name__
        self._is_gpu = is_gpu

        super().__init__(
            category="basic",
            backend=backend,
            device="cuda" if self._is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category, module class name and backend joined
        # by underscore. Dynamic and gpu are not reflected in the name.
        prefix = f"{self.category()}_{self._name}_{self.backend()}"
        return prefix

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: instantiates the module, sets float32 matmul precision
        # to high (global side effect) and creates a ones input of length 10 on the device.
        self.m = self.ModuleClass()
        torch.set_float32_matmul_precision("high")
        self.input = torch.ones(10, device=self.device())

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: inside a fresh inductor cache, compiles the module
        # (moved to cuda in place when gpu) with the backend and dynamic setting and calls it
        # once. The comment about C++ symbolic shape guards is stale here: the base class
        # collect step applies that setting.
        # enable_cpp_symbolic_shape_guards has impact on this benchmark
        # Keep using False value for consistency.
        with (
            fresh_cache(),
        ):
            opt_m = torch.compile(backend=self.backend(), dynamic=self.is_dynamic())(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument,
    # runs the single inline-module benchmark with compile-time instruction counting and
    # appends results.
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(InlineMod),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
