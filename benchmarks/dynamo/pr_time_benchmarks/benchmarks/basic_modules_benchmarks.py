# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of compiling small nn
# modules (currently a list of linear layers) with eager and inductor backends, on cpu or gpu,
# optionally forcing shape padding. Run as a script with one argument, the CSV result path.
import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._inductor.utils import fresh_cache


class ListOfLinears(nn.Module):
    # [CLASS DOC] Module holding a ModuleList of 20 linear layers (10 to 10 features).
    def __init__(self):
        # [FUNCTION DOC] Creates the list of 20 linear layers.
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        # [FUNCTION DOC] Applies each layer to the running value, adding the output of the layer
        # at half the loop index to the output of the current layer on the same input; shows
        # ModuleList used both as an iterable and indexed by int. Returns the final value.
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark parameterized by a module class, backend, gpu flag, dynamic flag and
    # force-shape-pad flag. Category is basic_modules. State: the module class, its instance and
    # a ones input created in the one-time setup.
    def __init__(
        self, ModuleClass, backend, is_gpu=False, dynamic=False, force_shape_pad=False
    ):
        # [FUNCTION DOC] Stores the module class, its name and the gpu and shape-pad flags, then
        # calls the base constructor with category basic_modules, the backend, cuda or cpu
        # device, and the dynamic flag.
        self.ModuleClass = ModuleClass
        self._name = ModuleClass.__name__
        self._is_gpu = is_gpu
        self._force_shape_pad = force_shape_pad

        super().__init__(
            category="basic_modules",
            backend=backend,
            device="cuda" if self._is_gpu else "cpu",
            dynamic=dynamic,
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category, module class name and backend joined
        # by underscore, with suffixes for dynamic, gpu and force_shape_pad when enabled.
        prefix = f"{self.category()}_{self._name}_{self.backend()}"
        if self.is_dynamic():
            prefix += "_dynamic"
        if self._is_gpu:
            prefix += "_gpu"
        if self._force_shape_pad:
            prefix += "_force_shape_pad"
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
        # [FUNCTION DOC] Measured work: inside a fresh inductor cache and with the inductor
        # force_shape_pad option patched, compiles the module (moved to cuda in place when using
        # gpu) with the backend and dynamic setting and calls it once on the input.
        with (
            fresh_cache(),
            torch._inductor.config.patch(force_shape_pad=self._force_shape_pad),
        ):
            opt_m = torch.compile(backend=self.backend(), dynamic=self.is_dynamic())(
                self.m.cuda() if self._is_gpu else self.m
            )
            opt_m(self.input)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, builds
    # four variants (eager, inductor, inductor gpu, inductor gpu with shape padding), and for each
    # collects compile-time instruction counts and appends results.
    result_path = sys.argv[1]
    benchmarks = [
        Benchmark(ListOfLinears, "eager"),
        Benchmark(ListOfLinears, "inductor"),
        Benchmark(ListOfLinears, "inductor", is_gpu=True),
        Benchmark(ListOfLinears, "inductor", is_gpu=True, force_shape_pad=True),
    ]
    for b in benchmarks:
        b.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
