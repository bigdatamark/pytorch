# [MODULE DOC] Tensor-expression benchmark for batched matrix multiplication. Defines one
# benchmark class and registers it with the benchmark registry at import time.
import numpy as np

from . import benchmark


class MatMulBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark of a batched matmul between a tensor of shape batch by M by N and
    # one of shape batch by N by K. Keeps the four extents and both random operands as the
    # inputs list.
    def __init__(self, mode, device, dtype, B, M, N, K):
        # [FUNCTION DOC] Stores the batch and matrix extents, creates the two random operands
        # on the given device and dtype (gradient-tracking in the both mode) and sets the
        # inputs list.
        super().__init__(mode, device, dtype)
        self.B = B
        self.M = M
        self.N = N
        self.K = K
        self.d1 = self.rand(
            [B, M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [B, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.d1, self.d2]

    def forward(self, d1, d2):
        # [FUNCTION DOC] Multiplies the two operands with the engine matmul and returns the
        # product.
        y = self.matmul(d1, d2)
        return y

    def reference(self):
        # [FUNCTION DOC] Computes the expected product with numpy from the numpy copies of
        # both operands.
        return np.matmul(self.numpy(self.d1), self.numpy(self.d2))

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: batch, M, N and K.
        return [self.B, self.M, self.N, self.K]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for this benchmark.
        return "batch_matmul"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, from the sizes of the operands plus the first operand size again as the
        # output stand-in, times a per-mode count that is larger in the both mode.
        # KNOWN ISSUE: the buffer size sums the first operand twice and the second once
        # instead of using the output size, which has batch by M by K elements; the two only
        # agree when N equals K.
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = 1 + 1
            algorithmic_count = 1 + (1 + 1)

        buffer_size = (
            self.B * self.M * self.N
            + self.B * self.M * self.N
            + self.B * self.N * self.K
        )
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    def compute_workload(self):
        # [FUNCTION DOC] Returns the floating point operation count: two times batch times M
        # times N times K, multiplied by one for fwd or three otherwise.
        if self.mode == "fwd":
            count = 1
        else:
            count = 1 + (1 + 1)

        op_count = 2 * self.B * self.M * self.N * self.K

        return op_count * count

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry of batch, M, N and K.
        return [[128, 64, 128, 256]]


benchmark.register_benchmark_class(MatMulBench)
