# [MODULE DOC] Tensor-expression benchmarks for sum reductions. Defines a 3D reduction family
# (row, mid, col, full) with an optional input-transform skip, a 2D reduction family with
# inner and outer variants, and dynamic shape versions of the 2D ones. Registers the concrete
# classes with the benchmark registry at import time.
from . import benchmark


class ReduceBench(benchmark.Benchmark):
    # [CLASS DOC] Base benchmark for sum reductions over a 3D random input of shape M by N by
    # K. The case selects which dimensions are reduced; the skip flag decides whether a
    # constant is added to the input before the sum. Subclasses fix the case.
    def __init__(self, mode, device, dtype, case, M, N, K, skip_input_transform):
        # [FUNCTION DOC] Stores the case and extents, parses the skip option string, creates
        # the random normal 3D input as the single entry of the inputs list, and chooses the
        # reduced dimensions from the case: row reduces the last two, mid the first and last,
        # col the first two, full all three. Raises ValueError for an unknown case or option.
        super().__init__(mode, device, dtype)
        self.case = case
        self.M = M
        self.N = N
        self.K = K
        self._set_skip_input_transform(skip_input_transform)

        self.inputs = [
            self.randn(
                [M, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        ]
        if case == "row":
            self.dims = [1, 2]
        elif case == "mid":
            self.dims = [0, 2]
        elif case == "col":
            self.dims = [0, 1]
        elif case == "full":
            self.dims = [0, 1, 2]
        else:
            raise ValueError(f"invalid case: {case}")

    def forward(self, inputs):
        # [FUNCTION DOC] Optionally adds a small constant to the input (not when the skip flag
        # is set), sums over the chosen dimensions and returns the reduced tensor.
        if self.skip_input_transform:
            x = inputs
        else:
            x = self.add(inputs, 0.001)
        y = self.sum(x, self.dims)
        return y

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: for the full case the total element
        # count and the skip option string, otherwise the three extents and the option string.
        if self.case == "full":
            return [self.M * self.N * self.K, self._skip_input_transform_str()]
        return [self.M, self.N, self.K, self._skip_input_transform_str()]

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry of three extents and
        # the skip option string for no skip.
        return [
            # [512, 512, 512],
            [512, 64, 512, "s0"],
        ]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the generic reduction benchmark.
        return "reduce"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements: the input size times a per-mode count, which is one for fwd and two
        # otherwise.
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + 1

        buffer_size = self.M * self.N * self.K
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    def _set_skip_input_transform(self, input_str):
        # [FUNCTION DOC] HELPER: Parses the option string: s0 means keep the input transform,
        # s1 means skip it. Sets the boolean skip attribute, and raises ValueError for any
        # other string.
        # In the test setting, s1 will skip the input transformation, and s0 will not.
        if input_str == "s0":
            self.skip_input_transform = False
        elif input_str == "s1":
            self.skip_input_transform = True
        else:
            raise ValueError(f"invalid skip_input_transform: {input_str}")

    def _skip_input_transform_str(self):
        # [FUNCTION DOC] HELPER: Converts the boolean skip attribute back to its option
        # string, s1 when skipping and s0 otherwise.
        if self.skip_input_transform:
            return "s1"
        else:
            return "s0"


class ReduceRowBench(ReduceBench):
    # [CLASS DOC] Reduction over the last two dimensions of the 3D input.
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        # [FUNCTION DOC] Forwards to the base class with the row case.
        super().__init__(mode, device, dtype, "row", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the row reduction.
        return "reduce_row"


class ReduceMidBench(ReduceBench):
    # [CLASS DOC] Reduction over the first and last dimensions of the 3D input.
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        # [FUNCTION DOC] Forwards to the base class with the mid case.
        super().__init__(mode, device, dtype, "mid", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the mid reduction.
        return "reduce_mid"


class ReduceColBench(ReduceBench):
    # [CLASS DOC] Reduction over the first two dimensions of the 3D input.
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        # [FUNCTION DOC] Forwards to the base class with the col case.
        super().__init__(mode, device, dtype, "col", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the col reduction.
        return "reduce_col"


class ReduceFullBench(ReduceBench):
    # [CLASS DOC] Reduction over all three dimensions, built as a long vector by passing the
    # length as the first extent and one for the other two.
    def __init__(self, mode, device, dtype, M, skip_input_transform):
        # [FUNCTION DOC] Forwards to the base class with the full case, the given length and
        # unit other extents.
        super().__init__(mode, device, dtype, "full", M, 1, 1, skip_input_transform)

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: the total element count and the skip
        # option string.
        return [self.M * self.N * self.K, self._skip_input_transform_str()]

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry with a length of two to
        # the power 24 and the skip option for skipping the transform.
        return [
            [1 << 24, "s1"],
        ]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the full reduction.
        return "reduce_full"


class Reduce2DBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark of a sum reduction over one dimension of a 2D random input after
    # adding a small constant, so that the fuser has something to fuse. Holds the reduced
    # dimension and both extents.
    """
    A benchmark class to validate 2 dimensional reduction performance.
    Only a simple add is fused to induce the fuser and isolate reduction perf.
    """

    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        # [FUNCTION DOC] Stores the reduction dimension and extents and creates one random
        # normal 2D input as the single entry of the inputs list. Raises ValueError if the
        # dimension is neither zero nor one (checked after the input is created).
        super().__init__(mode, device, dtype)
        self.red_dim = red_dim
        self.dim0 = dim0
        self.dim1 = dim1

        self.inputs = [
            self.randn(
                [dim0, dim1],
                device=device,
                dtype=dtype,
                requires_grad=self.requires_grad,
            )
        ]

        if red_dim != 0 and red_dim != 1:
            raise ValueError(f"invalid reduction dimension: {red_dim}")

    def forward(self, inputs):
        # [FUNCTION DOC] Adds a small constant to the input, sums over the configured
        # dimension and returns the result.
        x = self.add(inputs, 0.001)
        y = self.sum(x, [self.red_dim])
        return y

    def config(self):
        # [FUNCTION DOC] Returns the configuration list: reduction dimension and the two
        # extents.
        return [self.red_dim, self.dim0, self.dim1]

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry of reduction dimension
        # one and a 640 by 524288 shape.
        return [
            [1, 640, 524288],
        ]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the 2D reduction.
        return "reduce2d"

    @staticmethod
    def input_iterable():
        # [FUNCTION DOC] Returns True so the harness treats the input as iterable.
        return True

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, equal to the input size plus the output size (the extent of the non-
        # reduced dimension). Raises AssertionError for any mode other than fwd, because only
        # the forward pass is modeled.
        if self.mode != "fwd":
            raise AssertionError(
                f"Only the forward operation is modeled, but got mode='{self.mode}'"
            )

        buffer_size = self.dim0 * self.dim1
        if self.red_dim == 0:
            buffer_size += self.dim1
        else:
            buffer_size += self.dim0
        return {
            "sol": buffer_size,
            "algorithmic": buffer_size,
        }


class Reduce2DInnerBench(Reduce2DBench):
    # [CLASS DOC] 2D reduction fixed to the inner dimension (reduction dimension one); the
    # configuration omits the dimension.
    def __init__(self, mode, device, dtype, dim0, dim1):
        # [FUNCTION DOC] Forwards to the base class with reduction dimension one.
        super().__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the parent default configuration with its first item (the
        # reduction dimension) dropped.
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # [FUNCTION DOC] Returns the parent configuration with its first item (the reduction
        # dimension) dropped.
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the inner reduction.
        return "reduce2d_inner"


class Reduce2DOuterBench(Reduce2DBench):
    # [CLASS DOC] 2D reduction fixed to the outer dimension (reduction dimension zero); the
    # configuration omits the dimension.
    def __init__(self, mode, device, dtype, dim0, dim1):
        # [FUNCTION DOC] Forwards to the base class with reduction dimension zero.
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the parent default configuration with its first item (the
        # reduction dimension) dropped.
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # [FUNCTION DOC] Returns the parent configuration with its first item (the reduction
        # dimension) dropped.
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the outer reduction.
        return "reduce2d_outer"


benchmark.register_benchmark_class(ReduceRowBench)
benchmark.register_benchmark_class(ReduceMidBench)
benchmark.register_benchmark_class(ReduceColBench)
benchmark.register_benchmark_class(Reduce2DInnerBench)
benchmark.register_benchmark_class(Reduce2DOuterBench)
benchmark.register_benchmark_class(ReduceFullBench)


class DynamicReduce2DBench(benchmark.DynamicShape, Reduce2DBench):
    # [CLASS DOC] Dynamic shape version of the 2D reduction benchmark; many inputs with
    # randomized shapes are precomputed and cycled by the dynamic shape mixin.
    """
    A benchmark class to validate 2 dimensional reduction performance.
    Only a simple add is fused to induce the fuser and isolate reduction perf.
    """

    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        # [FUNCTION DOC] Initializes the dynamic shape state, then the 2D reduction base with
        # the given mode, device, dtype, reduction dimension and extents.
        benchmark.DynamicShape.__init__(self)
        Reduce2DBench.__init__(self, mode, device, dtype, red_dim, dim0, dim1)

    def instantiate_input(self):
        # [FUNCTION DOC] Draws randomized extents near the configured ones, creates a random
        # normal 2D tensor of that shape and assigns the inputs list to a one element list,
        # which the mixin stores as a new sample.
        dim0, dim1 = self.rand_shape([self.dim0, self.dim1])

        self.inputs = [
            self.randn(
                [dim0, dim1],
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
        ]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the dynamic 2D reduction.
        return "dynamicreduce2d"


class DynamicReduce2DInnerBench(DynamicReduce2DBench):
    # [CLASS DOC] Dynamic shape 2D reduction over the inner dimension.
    def __init__(self, mode, device, dtype, dim0, dim1):
        # [FUNCTION DOC] Forwards to the dynamic base class with reduction dimension one.
        super().__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the dynamic base default configuration with its first item
        # dropped.
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # [FUNCTION DOC] Returns the parent configuration with its first item (the reduction
        # dimension) dropped.
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the dynamic inner reduction.
        return "reduce2d_dynamic_inner"


class DynamicReduce2DOuterBench(DynamicReduce2DBench):
    # [CLASS DOC] Dynamic shape 2D reduction over the outer dimension.
    def __init__(self, mode, device, dtype, dim0, dim1):
        # [FUNCTION DOC] Forwards to the dynamic base class with reduction dimension zero.
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the dynamic base default configuration with its first item
        # dropped.
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        # [FUNCTION DOC] Returns the parent configuration with its first item (the reduction
        # dimension) dropped.
        parent_config = super().config()
        return parent_config[1:]

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name string for the dynamic outer reduction.
        return "reduce2d_dynamic_outer"


benchmark.register_benchmark_class(DynamicReduce2DInnerBench)
benchmark.register_benchmark_class(DynamicReduce2DOuterBench)
