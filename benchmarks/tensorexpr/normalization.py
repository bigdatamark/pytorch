# [MODULE DOC] Tensor-expression benchmarks for batch norm, instance norm and layer norm on
# NCHW inputs. Defines a shared base class and three subclasses, and registers the three
# subclasses with the benchmark registry at import time.
from . import benchmark, tensor_engine


class NormalizationBench(benchmark.Benchmark):
    # [CLASS DOC] Shared base of the normalization benchmarks. Holds the NCHW extents, the
    # random input data, random running mean and variance vectors per channel, and whether the
    # op runs in training mode (only in the both mode). Subclasses supply forward and the
    # module name.
    def __init__(self, mode, device, dtype, N, C, H, W):
        # [FUNCTION DOC] Stores the extents and creates the NCHW input via the NCHW random
        # helper plus per-channel running mean and variance vectors; sets the training flag to
        # true only in the both mode.
        # KNOWN ISSUE: the NCHW random helper is called with a dtype keyword, but neither the
        # benchmark base method nor the torch engine method accepts one, so construction
        # raises a TypeError with the torch engine.
        super().__init__(mode, device, dtype)
        self.N = N
        self.C = C
        self.H = H
        self.W = W

        self.data = self.nchw_rand(
            [self.N, self.C, self.H, self.W],
            device=device,
            dtype=dtype,
            requires_grad=self.requires_grad,
        )
        self.running_mean = self.rand([self.C], device=device, dtype=dtype)
        self.running_var = self.rand([self.C], device=device, dtype=dtype)
        self.training = self.mode == "both"

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: N, C, H and W.
        return [self.N, self.C, self.H, self.W]

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic, using the
        # input element count times four (a bytes-per-element assumption) times a per-mode
        # access count that is larger in the both mode.
        if self.mode == "fwd":
            sol_count = 1 + 1
            algorithmic_count = 2 + 1
        else:
            sol_count = (1 + 1) + (1 + 1)
            algorithmic_count = (2 + 1) + (3 + 1)

        buffer_size = self.N * self.C * self.H * self.W * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: a single entry of N, C, H and W.
        return [[128, 32, 128, 128]]


class BatchNormBench(NormalizationBench):
    # [CLASS DOC] Batch normalization benchmark using the running statistics and the training
    # flag from the base class.
    def forward(self):
        # [FUNCTION DOC] Runs batch norm on the input with the stored running mean and
        # variance, training as set by the mode, and returns the output.
        y = self.batch_norm(
            self.data, self.running_mean, self.running_var, training=self.training
        )
        return y

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for batch norm.
        return "batchnorm"


class InstanceNormBench(NormalizationBench):
    # [CLASS DOC] Instance normalization benchmark; can be marked unsupported by engines that
    # lack the op.
    def forward(self):
        # [FUNCTION DOC] Runs instance norm on the input and returns the output.
        y = self.instance_norm(self.data)
        return y

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for instance norm.
        return "instance_norm"

    def is_supported(self):
        # [FUNCTION DOC] Reports whether the active engine supports instance norm, by checking
        # the engine method for an unsupported marker.
        return tensor_engine.is_supported(self.instance_norm)


class LayerNormBench(NormalizationBench):
    # [CLASS DOC] Layer normalization benchmark that normalizes over the last two dimensions
    # (height and width).
    def forward(self):
        # [FUNCTION DOC] Runs layer norm on the input over the trailing height and width
        # dimensions and returns the output.
        y = self.layer_norm(self.data, [self.H, self.W])
        return y

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for layer norm.
        return "layernorm"


benchmark.register_benchmark_class(BatchNormBench)
benchmark.register_benchmark_class(InstanceNormBench)
benchmark.register_benchmark_class(LayerNormBench)
