# [MODULE DOC] PR-time benchmark measuring the raw instruction count of one call through a
# wrapper such as the dynamo disable decorator around a trivial identity function. Run as a
# script with one argument, the CSV result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch


def _trivial(x):
    # [FUNCTION DOC] HELPER: Trivial callee that returns its argument unchanged, so measurements
    # reflect only the wrapper.
    return x


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark of per-call wrapper overhead, parameterized by category name,
    # description text and a wrapper callable applied to the trivial function. Runs on cpu.
    # State: the wrapper, the wrapped function and a one-element input created in setup.
    # Per-call overhead of a disable-style wrapper around a trivial callee, so
    # the instruction count reflects the wrapper, not the callee body. `wrap` is
    # the decorator under test, applied to _trivial.
    def __init__(self, category, description, wrap):
        # [FUNCTION DOC] Stores the description and wrapper, then calls the base constructor
        # with the given category and the cpu device.
        self._description = description
        self._wrap = wrap
        super().__init__(category=category, device="cpu")

    def name(self):
        # [FUNCTION DOC] Returns the result name, which is just the category.
        return self.category()

    def description(self):
        # [FUNCTION DOC] Returns the description text given at construction.
        return self._description

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: resets dynamo state, wraps the trivial function with
        # the wrapper, creates a one-element ones tensor, and warms up by running the work
        # ten times.
        torch._dynamo.reset()
        self._fn = self._wrap(_trivial)
        self.a = torch.ones(1)
        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: does nothing.
        pass

    def _work(self):
        # [FUNCTION DOC] Measured work: one call of the wrapped function on the stored tensor.
        self._fn(self.a)


def _benchmarks():
    # [FUNCTION DOC] HELPER: Returns the list of benchmarks to run: a single disable-overhead
    # benchmark wrapping with the dynamo disable function.
    return [
        Benchmark(
            "disable_overhead",
            "per-call overhead of torch._dynamo.disable with a trivial callee",
            torch._dynamo.disable,
        ),
    ]


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument and,
    # for each benchmark, collects plain instruction counts and appends results to the path.
    result_path = sys.argv[1]
    for benchmark in _benchmarks():
        benchmark.enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
