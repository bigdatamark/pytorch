# [MODULE DOC] PR-time benchmark measuring raw instruction count of calling a compiled add-one
# function at runtime across variants (requires grad, inference mode, backward, dynamic shapes,
# plus a cpu eager variant). Defaults to inductor on cuda. Run as a script with one argument,
# the CSV result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch.autograd.grad_mode import inference_mode


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark of per-call runtime overhead of a compiled small function. State:
    # requires-grad, inference-mode and backward flags, a two-element input, the compiled
    # function, and (in backward mode) the last forward value.
    def __init__(
        self,
        requires_grad,
        inference_mode,
        backward,
        dynamic,
        backend="inductor",
        device="cuda",
    ):
        # [FUNCTION DOC] Raises AssertionError if both inference_mode and backward are
        # requested. Stores the three flags and calls the base constructor with category
        # runtime_overhead, the backend, the device and the dynamic flag.
        if inference_mode and backward:
            raise AssertionError("inference_mode and backward cannot be both True")

        self._requires_grad = requires_grad
        self._inference_mode = inference_mode
        self._backward = backward

        super().__init__(
            category="runtime_overhead",
            backend=backend,
            device=device,
            dynamic=dynamic,
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category and backend, then device (only when
        # not cuda), then suffixes for requires_grad, inference_mode, backward and dynamic
        # when set.
        prefix = f"{self.category()}_{self.backend()}"
        # Disambiguate non-cuda variants; keep existing cuda names unchanged.
        if self.device() != "cuda":
            prefix += f"_{self.device()}"
        if self._requires_grad:
            prefix += "_requires_grad"
        if self._inference_mode:
            prefix += "_inference_mode"
        if self._backward:
            prefix += "_backward"
        if self.is_dynamic():
            prefix += "_dynamic"
        return prefix

    def description(self):
        # [FUNCTION DOC] Returns a fixed description string.
        return "runtime of a compiled add1 op small input"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: resets dynamo, creates a ones tensor of two elements on
        # the device with the requires-grad flag, compiles an add-one function with the backend
        # (full graph, dynamic setting), and runs ten warm-up iterations. Warm-up in backward
        # mode runs forward, sums and calls backward; otherwise it runs the measured work.
        torch._dynamo.reset()
        self.a = torch.ones(2, device=self.device(), requires_grad=self._requires_grad)

        @torch.compile(
            backend=self.backend(),
            fullgraph=True,
            dynamic=self.is_dynamic(),
        )
        def add1(a):
            # [FUNCTION DOC] HELPER: Compiled workload; returns the input plus one.
            return a + 1

        self._add1 = add1

        # warmup
        for _ in range(10):
            if self._backward:
                self.forward_val = self._add1(self.a).sum()
                self.forward_val.backward()
            else:
                self._work()

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: in backward mode runs the forward pass and stores
        # the summed output so the measured step is only backward; otherwise does nothing.
        if self._backward:
            self.forward_val = self._add1(self.a).sum()

    def _work(self):
        # [FUNCTION DOC] Measured work: in inference mode runs the compiled function under the
        # inference mode context; else in backward mode calls backward on the stored forward
        # value; else calls the compiled function once.
        if self._inference_mode:
            with inference_mode():
                self._add1(self.a)
        elif self._backward:
            self.forward_val.backward()
        else:
            self._add1(self.a)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument and
    # builds nine variants: eight cuda inductor ones covering requires_grad, inference_mode and
    # backward combinations in static and dynamic form, and one cpu eager variant that isolates
    # the compile wrapper prologue. Each is collected with plain instruction counting and
    # appended to the path.
    result_path = sys.argv[1]
    all = [
        Benchmark(
            requires_grad=False, inference_mode=False, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=False, inference_mode=True, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=False, dynamic=False
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=True, dynamic=False
        ),
        Benchmark(
            requires_grad=False, inference_mode=False, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=False, inference_mode=True, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=False, dynamic=True
        ),
        Benchmark(
            requires_grad=True, inference_mode=False, backward=True, dynamic=True
        ),
        # cpu + eager isolates the compile_wrapper per-call prologue (dispatch
        # key set save/restore) with no CUDA launch or inductor runtime noise.
        Benchmark(
            requires_grad=False,
            inference_mode=False,
            backward=False,
            dynamic=False,
            backend="eager",
            device="cpu",
        ),
    ]

    for benchmark in all:
        benchmark.enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
