# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of compiling a deeply
# nested module tree (depth 3, width 4 by default) built from relu layers and sequential
# containers. Run as a script with one argument, the CSV result path. No import-time side
# effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_cache


class NestedModule(nn.Module):
    # [CLASS DOC] Recursive module: each level holds two relu layers, a sequential container of
    # width children (nested modules while depth is positive, plain relu layers at depth zero)
    # and an integer attribute equal to 2. State: depth, width, relu_a, relu_b, sub_mods, a.
    def __init__(self, depth=3, width=4):
        # [FUNCTION DOC] Builds the module tree recursively: stores depth and width, creates two
        # relu modules, builds width children (a nested module of depth one less, or a relu at
        # depth zero), wraps them in a sequential container and sets the integer attribute to 2.
        super().__init__()
        self.depth = depth
        self.width = width

        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()

        sub_mods = []
        if depth > 0:
            for i in range(width):
                sub_mods.append(NestedModule(depth - 1, width))
        else:
            for i in range(width):
                sub_mods.append(nn.ReLU())
        self.sub_mods = nn.Sequential(*sub_mods)
        self.a = 2

    def forward(self, x):
        # [FUNCTION DOC] Applies the first relu, adds the output of the sequential children
        # to it, then returns that value plus the second relu applied to it plus the
        # integer attribute.
        x = self.relu_a(x)
        x = x + self.sub_mods(x)
        return x + self.relu_b(x) + self.a


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark of compiling a given module class with a backend (default eager).
    # Category is basic. State: module class and name, gpu flag, module instance, ones input.
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
        # once on the input. The comment about C++ symbolic shape guards is stale here: the base
        # class collect step applies that setting.
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
    # runs the single nested-module benchmark with compile-time instruction counting and
    # appends results.
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(NestedModule),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
