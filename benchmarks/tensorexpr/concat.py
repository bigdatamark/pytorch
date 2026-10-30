# [MODULE DOC] Tensor-expression benchmarks for concatenating two 2D inputs along a chosen
# dimension. Defines a plain concat benchmark and a variant that adds a relu and enables
# graph-level concat optimization, and registers both with the benchmark registry at import
# time.
import numpy as np

import torch

from . import benchmark


class Concat2D2InputBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark that adds a small constant to two 2D inputs and concatenates the
    # results along a configurable dimension. Keeps the four input extents, the concat
    # dimension, two random input tensors, and the inputs list used by the harness.
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        # [FUNCTION DOC] Stores the two input shapes and the concat dimension, then creates
        # two random normal tensors of those shapes on the given device and dtype, tracking
        # gradients when the mode is both. Sets the inputs list to the two tensors.
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]

    def forward(self, input1, input2):
        # [FUNCTION DOC] Adds a small constant to each input, concatenates the two results
        # along the configured dimension with the engine concat, and returns the joined
        # tensor.
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        return y

    def reference(self):
        # [FUNCTION DOC] Builds the expected result with numpy by concatenating the numpy
        # copies of both inputs along the configured dimension.
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        # [FUNCTION DOC] Returns the configuration as a list: the four input extents followed
        # by the concat dimension.
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string used to select this benchmark.
        return "concat2d2input"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic counted in
        # elements: the combined size of both inputs times a small per-mode multiplier, larger
        # in the both mode because the backward pass is included. Element byte size is not
        # applied.
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations, each being the four input extents
        # plus the concat dimension, from tiny to very large shapes.
        return [
            [1, 160, 1, 14, 1],
            [1, 580, 1, 174, 1],
            [20, 160, 20, 14, 1],
            [20, 580, 20, 174, 1],
            [8, 512, 8, 512, 1],
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(Concat2D2InputBench)


class ConcatGraphOptBench(benchmark.Benchmark):
    # [CLASS DOC] Variant of the concat benchmark that also applies a relu to the concatenated
    # result. Construction flips global JIT switches so the CPU fuser and the concat-without-
    # conditionals lowering are active.
    def __init__(self, mode, device, dtype, I1_D1, I1_D2, I2_D1, I2_D2, concat_dim):
        # [FUNCTION DOC] Stores shapes and the concat dimension, creates two random normal
        # inputs (gradient-tracking in the both mode) and sets the inputs list. Side effect:
        # globally enables CPU fusion override and the concat-without-conditionals JIT option,
        # which stay on for the process.
        super().__init__(mode, device, dtype)
        self.I1_D1 = I1_D1
        self.I1_D2 = I1_D2
        self.I2_D1 = I2_D1
        self.I2_D2 = I2_D2
        self.concat_dim = concat_dim
        self.input1 = self.randn(
            [I1_D1, I1_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.input2 = self.randn(
            [I2_D1, I2_D2], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.input1, self.input2]
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_cat_wo_conditionals(True)

    def forward(self, input1, input2):
        # [FUNCTION DOC] Adds a small constant to each input, concatenates along the
        # configured dimension, applies relu to the joined tensor and returns it.
        x1 = self.add(input1, 0.00001)
        x2 = self.add(input2, 0.00001)
        y = self.cat((x1, x2), dim=self.concat_dim)
        z = self.relu(y)
        return z

    def reference(self):
        # [FUNCTION DOC] Builds the numpy concatenation of the two inputs along the configured
        # dimension.
        # KNOWN ISSUE: the reference omits the relu that forward applies, so it would differ
        # from forward for negative values; it is only compared if the benchmark were marked
        # deterministic, which it is not.
        return np.concatenate(
            (self.numpy(self.input1), self.numpy(self.input2)),
            axis=self.concat_dim,
        )

    def config(self):
        # [FUNCTION DOC] Returns the configuration as a list: the four input extents followed
        # by the concat dimension.
        return [self.I1_D1, self.I1_D2, self.I2_D1, self.I2_D2, self.concat_dim]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string used to select this benchmark.
        return "concatGraphOpt"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, using the same counting as the plain concat benchmark; the relu is not
        # counted.
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (3 + 1) + (3 + 1)

        buffer_size = self.I1_D1 * self.I1_D2 + self.I2_D1 * self.I2_D2
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: only the four largest shapes with
        # concat dimension one.
        return [
            [1 << 13, 1060, 1 << 13, 1040, 1],
            [1 << 13, 2000, 1 << 13, 1074, 1],
            [1 << 15, 1060, 1 << 15, 2670, 1],
            [1 << 15, 5120, 1 << 15, 2512, 1],
        ]


benchmark.register_benchmark_class(ConcatGraphOptBench)
