# [MODULE DOC] Tensor-expression benchmarks for 2D max pooling and average pooling. One shared
# class parameterized by a case name, plus two thin subclasses, registered with the benchmark
# registry at import time.
from . import benchmark


class PoolingBench(benchmark.Benchmark):
    # [CLASS DOC] Shared pooling benchmark that holds the case (maxpool or avgpool), kernel
    # size, input extents and a random NCHW input. Always uses stride one.
    def __init__(self, case, mode, device, dtype, kernel_size, N, C, H, W):
        # [FUNCTION DOC] Stores the case, kernel size and extents and creates the random input
        # on the device with the given dtype, tracking gradients in the both mode.
        # KNOWN ISSUE: the base initializer is called with only the mode and device and
        # without the dtype argument it requires, so construction raises a TypeError.
        super().__init__(mode, device)
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.data = self.rand(
            [N, C, H, W], device=device, dtype=dtype, requires_grad=self.requires_grad
        )

    def forward(self):
        # [FUNCTION DOC] Applies max pooling or average pooling with stride one to the stored
        # input depending on the case, and returns the output.
        # KNOWN ISSUE: if the case is neither maxpool nor avgpool the output variable is never
        # assigned and an error is raised on return.
        if self.case == "maxpool":
            y = self.max_pool2d(self.data, self.kernel_size, stride=1)
        elif self.case == "avgpool":
            y = self.avg_pool2d(self.data, self.kernel_size, stride=1)
        return y

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: kernel size, N, C, H and W.
        return [self.kernel_size, self.N, self.C, self.H, self.W]

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, from the input size times a per-mode access count that is larger in the
        # both mode.
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 1 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (1 + 1) + (2 + 1)

        buffer_size = self.N * self.C * self.H * self.W
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry of kernel size, N, C, H
        # and W.
        return [[3, 16, 32, 256, 256]]


class MaxPoolBench(PoolingBench):
    # [CLASS DOC] Max pooling benchmark; a thin subclass that selects the maxpool case.
    def __init__(self, *args):
        # [FUNCTION DOC] Forwards all arguments to the base class with the maxpool case
        # prepended.
        super().__init__("maxpool", *args)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for max pooling.
        return "maxpool"


class AvgPoolBench(PoolingBench):
    # [CLASS DOC] Average pooling benchmark; a thin subclass that selects the avgpool case.
    def __init__(self, *args):
        # [FUNCTION DOC] Forwards all arguments to the base class with the avgpool case
        # prepended.
        super().__init__("avgpool", *args)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for average pooling.
        return "avgpool"


benchmark.register_benchmark_class(MaxPoolBench)
benchmark.register_benchmark_class(AvgPoolBench)
