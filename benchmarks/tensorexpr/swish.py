# [MODULE DOC] Tensor-expression benchmark of a hard-swish style activation built from relu,
# minimum and constant multiplications. Defines one class and registers it with the benchmark
# registry at import time.
import torch

from . import benchmark


class SwishBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark of the activation: input times (minimum of relu of the input and
    # six, plus three) times one sixth. Holds the random input and preallocated constant
    # tensors of the same shape.
    def __init__(self, mode, device, dtype, M, N):
        # [FUNCTION DOC] Stores the extents, creates the random M by N input (gradient-
        # tracking in the both mode), sets the inputs list, and builds constant tensors for
        # six, three and one sixth on the device by offsetting a zero tensor (created without
        # the dtype argument, so in the default dtype).
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.data = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.data]
        self.zeros = torch.zeros(M, N, device=device)
        self.six = self.zeros + 6.0
        self.three = self.zeros + 3.0
        self.sixth = self.zeros + 1.0 / 6.0

    def forward(self, inp):
        # [FUNCTION DOC] Computes the input times the sum of the minimum of relu of the input
        # and the six tensor and the three tensor, times the one-sixth tensor, and returns it.
        y = inp * (torch.min(torch.relu(inp), self.six) + self.three) * self.sixth
        return y

    def reference(self):
        # [FUNCTION DOC] Runs forward on the stored input and converts the result to a numpy
        # array.
        return self.numpy(self.forward(self.data))

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: M and N.
        return [self.M, self.N]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for this benchmark.
        return "swish"

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
        # [FUNCTION DOC] Returns the default configurations: one entry of M equal to 128 and N
        # equal to two to the power 16.
        return [[128, 1 << 16]]


benchmark.register_benchmark_class(SwishBench)
