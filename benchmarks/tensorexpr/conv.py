# [MODULE DOC] Tensor-expression benchmarks for 2D convolution, covering a regular convolution
# and a depthwise one that share one implementation class. Registers both classes with the
# benchmark registry at import time.
from . import benchmark


class ConvImplBench(benchmark.Benchmark):
    # [CLASS DOC] Shared implementation of the convolution benchmarks. Holds the case name,
    # kernel size, input extents, output channel count, group count, the random input data and
    # the conv layer. Subclasses fix the case.
    def __init__(self, case, mode, device, dtype, kernel_size, N, iC, H, W, oC):
        # [FUNCTION DOC] Creates the random NCHW input, picks groups as one for a regular conv
        # or the input channel count for depthwise conv, builds the conv layer from the engine
        # and moves it to the device when the device is not cpu. Raises ValueError for an
        # unknown case.
        super().__init__(mode, device, dtype)
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.iC = iC
        self.H = H
        self.W = W
        self.oC = oC
        self.data = self.rand(
            [N, iC, H, W], device=device, requires_grad=self.requires_grad
        )
        if case == "conv":
            self.groups = 1
        elif case == "depthwise_conv":
            self.groups = iC
        else:
            raise ValueError(f"invalid case: {case}")

        self.conv = self.conv2d_layer(iC, oC, kernel_size, groups=self.groups)
        if device != "cpu":
            self.to_device(self.conv, device)

    def forward(self):
        # [FUNCTION DOC] Runs the conv layer on the stored input and returns the output. Takes
        # no arguments because the data lives on the instance.
        y = self.conv(self.data)
        return y

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: kernel size, batch size, input
        # channels, height, width and output channels.
        return [self.kernel_size, self.N, self.iC, self.H, self.W, self.oC]

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, summed over input, output and kernel buffers, each scaled by a per-mode
        # access count (larger in the both mode). Output is assumed to have the same spatial
        # size as the input, and the kernel size accounts for groups.
        if self.mode == "fwd":
            sol_count = {"i": 1, "o": 1, "k": 1}
            algorithmic_count = {"i": 1, "o": 1, "k": 1}
        else:
            sol_count = {"i": 1 + 1, "o": 1 + 1, "k": 1 + 1}
            algorithmic_count = {"i": 1 + (1 + 1), "o": 1 + (1 + 1), "k": 1 + (1 + 1)}

        buffer_size = {
            "i": self.N * self.iC * self.H * self.W,
            "o": self.N * self.oC * self.H * self.W,
            "k": self.oC
            * (self.iC / self.groups)
            * self.kernel_size
            * self.kernel_size,
        }
        sol_size = 0
        algorithmic_size = 0
        for key in sol_count:
            sol_size += buffer_size[key] * sol_count[key]
            algorithmic_size += buffer_size[key] * algorithmic_count[key]
        return {"sol": sol_size, "algorithmic": algorithmic_size}

    def compute_workload(self):
        # [FUNCTION DOC] Returns the floating point operation count: two times the multiply-
        # accumulates of the convolution, multiplied by one for fwd or by three for both.
        # Raises ValueError for any other mode.
        if self.mode == "fwd":
            count = 1
        elif self.mode == "both":
            count = 1 + (1 + 1)
        else:
            raise ValueError(f"invalid mode: {self.mode}")

        op_count = (
            self.N
            * self.iC
            / self.groups
            * self.oC
            * self.kernel_size
            * self.kernel_size
            * self.H
            * self.W
        )
        op_count *= 2

        return op_count * count

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configuration list holding one entry of kernel
        # size, batch, input channels, height, width and output channels.
        return [
            [3, 64, 32, 128, 128, 64],
        ]


class ConvBench(ConvImplBench):
    # [CLASS DOC] Regular (single group) convolution benchmark, a thin subclass that selects
    # the conv case.
    def __init__(self, *args):
        # [FUNCTION DOC] Forwards all arguments to the base class with the conv case
        # prepended.
        super().__init__("conv", *args)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the regular convolution benchmark.
        return "conv"


class DepthwiseConvBench(ConvImplBench):
    # [CLASS DOC] Depthwise convolution benchmark, a thin subclass that selects the depthwise
    # case where groups equal the input channels.
    def __init__(self, *args):
        # [FUNCTION DOC] Forwards all arguments to the base class with the depthwise_conv case
        # prepended.
        super().__init__("depthwise_conv", *args)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the depthwise convolution
        # benchmark.
        return "depthwise_conv"


benchmark.register_benchmark_class(ConvBench)
benchmark.register_benchmark_class(DepthwiseConvBench)
