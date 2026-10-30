# [MODULE DOC] PR-time benchmark measuring compile-time instruction count of the AOT dispatcher
# path (aot_eager_decomp_partition backend) on a function with 100 inputs each added to itself,
# in training or inference mode, with plain tensors or TwoTensor subclass inputs. Run as a
# script with one argument, the CSV result path. No import-time side effects beyond imports.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch.testing._internal.two_tensor import TwoTensor


class Benchmark(BenchmarkBase):
    # [CLASS DOC] Benchmark parameterized by keyword-only training and subclass flags. Category
    # is aotdispatcher, backend aot_eager_decomp_partition, device cpu. State: the flags and the
    # list of 100 input arguments built in the one-time setup.
    def __init__(self, *, training, subclass):
        # [FUNCTION DOC] Stores the training and subclass flags, then calls the base constructor
        # with mode set to training or inference accordingly.
        self._training = training
        self._subclass = subclass
        super().__init__(
            category="aotdispatcher",
            backend="aot_eager_decomp_partition",
            device="cpu",
            mode="training" if self._training else "inference",
        )

    def name(self):
        # [FUNCTION DOC] Returns the result name: category and mode, then subclass or
        # nosubclass, and a cpu suffix when the device is cpu.
        prefix = f"{self.category()}_{self.mode()}"
        if self._subclass:
            prefix += "_subclass"
        else:
            prefix += "_nosubclass"
        if self.device() == "cpu":
            prefix += "_cpu"
        return prefix

    def description(self):
        # [FUNCTION DOC] Returns a fixed description of the workload shape (100 inputs, 100
        # outputs, each input added once).
        return "100 inputs, 100 outputs, each input is added once"

    def _prepare_once(self):
        # [FUNCTION DOC] One-time setup: builds 100 ones tensors of length 100 on the device,
        # requiring grad when in training mode. In subclass mode each is wrapped into a
        # TwoTensor pairing it with a detached clone that has requires_grad set to the training
        # flag. Stores the list as the call arguments.
        _args = [
            torch.ones(100, requires_grad=self._training, device=self.device())
            for _ in range(100)
        ]
        if self._subclass:
            _args = [
                TwoTensor(x, x.clone().detach().requires_grad_(self._training))
                for x in _args
            ]
        self._args = _args

    def _prepare(self):
        # [FUNCTION DOC] Per-iteration setup: resets dynamo state.
        torch._dynamo.reset()

    def _work(self):
        # [FUNCTION DOC] Measured work: compiles a function with the benchmark backend in full
        # graph mode and calls it once on the stored inputs.
        @torch.compile(backend=self.backend(), fullgraph=True)
        def f(*args):
            # [FUNCTION DOC] HELPER: Compiled workload; returns a list with the sum of each
            # input with itself (one output per input).
            outs = [torch.add(x, x) for x in args]
            return outs

        f(*self._args)


def main():
    # [FUNCTION DOC] Script entry point. Reads the output CSV path from the first argument, builds
    # the four combinations of training and subclass flags, and for each collects compile-time
    # instruction counts and appends results to the path.
    result_path = sys.argv[1]
    all = [
        Benchmark(training=False, subclass=False),
        Benchmark(training=True, subclass=False),
        Benchmark(training=False, subclass=True),
        Benchmark(training=True, subclass=True),
    ]

    for benchmark in all:
        benchmark.enable_compile_time_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
