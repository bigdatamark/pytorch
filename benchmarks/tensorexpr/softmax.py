# [MODULE DOC] Tensor-expression benchmark for softmax over the last dimension of a 2D input.
# Defines one benchmark class and registers it with the benchmark registry at import time.
import scipy.special

from . import benchmark


class SoftmaxBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark that adds a small constant to a random 2D input and applies
    # softmax along the last dimension. Holds the two extents, the dtype and the single input
    # in the inputs list.
    def __init__(self, mode, device, dtype, M, N):
        # [FUNCTION DOC] Stores the extents and dtype and creates one random normal M by N
        # tensor (gradient-tracking in the both mode) as the only entry of the inputs list.
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.dtype = dtype
        self.inputs = [
            self.randn(
                [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        ]

    def forward(self, inputs):
        # [FUNCTION DOC] Adds a small constant to the input, applies softmax over the last
        # dimension with the stored dtype and returns the result.
        x = self.add(inputs, 0.001)
        y = self.softmax(x, dim=-1, dtype=self.dtype)
        return y

    def reference(self):
        # [FUNCTION DOC] Computes the expected softmax over the last axis with scipy on the
        # numpy copy of the input.
        # KNOWN ISSUE: the inputs attribute is a list, and the numpy helper calls a tensor
        # method on its argument, so this raises an error; it is never reached because the
        # benchmark is not marked deterministic.
        return scipy.special.softmax(self.numpy(self.inputs), axis=-1)

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: M and N.
        return [self.M, self.N]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for this benchmark.
        return "softmax"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, from M times N times a per-mode count that is larger in the both mode.
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.M * self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: three entries of M and N covering
        # a tall, a skinny and a wide shape.
        return [
            [480, 20],
            [1 << 15, 32],
            [128, 1 << 16],
        ]


benchmark.register_benchmark_class(SoftmaxBench)
