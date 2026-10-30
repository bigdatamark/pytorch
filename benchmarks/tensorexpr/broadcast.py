# [MODULE DOC] Broadcasting-add benchmarks for the tensor expression harness. Defines three
# row/mid/col multiply benchmarks and a three-argument benchmark (their registration is
# commented out), plus a template class for broadcasting elementwise binary and unary operations
# of which many subclasses are generated and registered when the module is imported.
# KNOWN ISSUE: the generated operator list references a numpy erf function, which numpy does not
#     provide, so importing this module may fail with an attribute error.
import itertools
import operator

import numpy as np

import torch

from . import benchmark


class BroadcastMulBench(benchmark.Benchmark):
    # [CLASS DOC] Benchmark adding two broadcast tensors whose shapes depend on the case: row,
    # mid or col, giving an output of M by N by K. Holds the two inputs and the sizes;
    # subclasses fix the case.
    def __init__(self, mode, device, dtype, case, M, N, K):
        # [FUNCTION DOC] Stores the sizes and creates two random inputs whose broadcast
        # dimensions depend on the case (row: M by N by 1 and M by 1 by K; mid: M by N by 1 and
        # 1 by N by K; col: M by 1 by K and 1 by N by K). Raises a value error for any other
        # case. Collects them in the inputs list.
        super().__init__(mode, device, dtype)
        self.case = case
        self.M = M
        self.N = N
        self.K = K

        if case == "row":
            self.d1 = self.rand(
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "mid":
            self.d1 = self.rand(
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "col":
            self.d1 = self.rand(
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        else:
            raise ValueError(f"invalid case: {case}")

        self.inputs = [self.d1, self.d2]

    def forward(self, d1, d2):
        # [FUNCTION DOC] Returns the broadcast sum of the two inputs.
        y = d1 + d2
        return y

    def reference(self):
        # [FUNCTION DOC] Returns the numpy sum of the two stored inputs as the reference result.
        return self.numpy(self.d1) + self.numpy(self.d2)

    def config(self):
        # [FUNCTION DOC] Returns the sizes M, N and K as a list.
        return [self.M, self.N, self.K]

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns one default config: M 128, N 256, K 128.
        return [[128, 256, 128]]

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with entries sol and algorithmic in element counts:
        # the buffer is M times N times K elements, multiplied by 1 and 1 in forward mode or 2
        # and 3 in both mode.
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1)

        buffer_size = self.M * self.N * self.K
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }


class BroadcastRowBench(BroadcastMulBench):
    # [CLASS DOC] Row case of the broadcast sum benchmark.
    def __init__(self, mode, device, dtype, M, N, K):
        # [FUNCTION DOC] Creates the base benchmark with the case fixed to row.
        super().__init__(mode, device, dtype, "row", M, N, K)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name used for selection.
        return "broadcast_row"


class BroadcastMidBench(BroadcastMulBench):
    # [CLASS DOC] Mid case of the broadcast sum benchmark.
    def __init__(self, mode, device, dtype, M, N, K):
        # [FUNCTION DOC] Creates the base benchmark with the case fixed to mid.
        super().__init__(mode, device, dtype, "mid", M, N, K)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name used for selection.
        return "broadcast_mid"


class BroadcastColBench(BroadcastMulBench):
    # [CLASS DOC] Col case of the broadcast sum benchmark.
    def __init__(self, mode, device, dtype, M, N, K):
        # [FUNCTION DOC] Creates the base benchmark with the case fixed to col.
        super().__init__(mode, device, dtype, "col", M, N, K)

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name used for selection.
        return "broadcast_col"


class BroadcastThreeArgs(benchmark.Benchmark):
    # [CLASS DOC] Benchmark adding three tensors of different ranks that broadcast to a
    # four-dimensional output of L by K by M by N.
    def __init__(self, mode, device, dtype, M, N, K, L):
        # [FUNCTION DOC] Stores the sizes and creates three random inputs of shapes M by N, K by
        # M by 1 and L by K by 1 by 1, collected in the inputs list.
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.K = K
        self.L = L

        self.d1 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d3 = self.rand(
            [L, K, 1, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )

        self.inputs = [self.d1, self.d2, self.d3]

    def forward(self, d1, d2, d3):
        # [FUNCTION DOC] Returns the broadcast sum of the three inputs.
        y = d1 + d2 + d3
        return y

    def reference(self):
        # [FUNCTION DOC] Returns the numpy sum of the three stored inputs as the
        # reference result.
        return self.numpy(self.d1) + self.numpy(self.d2) + self.numpy(self.d3)

    def config(self):
        # [FUNCTION DOC] Returns the sizes M, N, K and L as a list.
        return [self.M, self.N, self.K, self.L]

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns one default config: M 32, N 16, K 64, L 128.
        return [[32, 16, 64, 128]]

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with entries sol and algorithmic: a buffer of M
        # times N times K times L times 4, multiplied by 1 and 1 in forward mode or 2 and 4 in
        # both mode.
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1 + 1)

        buffer_size = self.M * self.N * self.K * self.L * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def module():
        # [FUNCTION DOC] Returns the module name used for selection.
        return "broadcast_3args"


# benchmark.register_benchmark_class(BroadcastRowBench)
# benchmark.register_benchmark_class(BroadcastMidBench)
# benchmark.register_benchmark_class(BroadcastColBench)
# benchmark.register_benchmark_class(BroadcastThreeArgs)


# TODO: merge this with elementwise bench
# A template class for elementwise operations.
# A derived class will override the class instance to customize its behavior.
class BroadcastBench(benchmark.Benchmark):
    # [CLASS DOC] Template for broadcasting elementwise benchmarks. Customization class
    # attributes: a name suffix, binary and unary functions for torch and numpy, and a flag
    # saying whether each input gets its own unary application or the third input is derived
    # from the first. Generated subclasses set these attributes. Holds four random inputs.
    # List of customization class variables.
    op_str = None
    binary_op_pt_func = None
    binary_op_np_func = None
    unary_op_pt_func = None
    unary_op_np_func = None
    split_input = True

    def __init__(self, mode, device, dtype, M, N, K):
        # [FUNCTION DOC] Stores the sizes and creates four random inputs of shapes M by N, K by
        # 1 by N, M by N and K by M by 1, collected in the inputs list.
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.K = K
        self.d1 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [K, 1, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d3 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d4 = self.rand(
            [K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.d1, self.d2, self.d3, self.d4]

    def _eval(self, d1, d2, d3, d4, binary_op, unary_op):
        # [FUNCTION DOC] HELPER: Shared computation for torch and numpy. Uses addition as the
        # binary function and identity as the unary function when none is given. Applies the
        # unary function to the inputs (in shared mode the third input is the unary of the first
        # plus a small constant), combines the first with the second and the third with the
        # fourth by the binary function, and returns the sum of those two results.
        if not binary_op:

            def binary_op(x, y):
                # [FUNCTION DOC] HELPER: Default binary function: returns the sum of its
                # two arguments.
                return x + y

        if not unary_op:

            def unary_op(x):
                # [FUNCTION DOC] HELPER: Default unary function: returns its argument unchanged.
                return x

        if self.split_input:
            d1 = unary_op(d1)
            d2 = unary_op(d2)
            d3 = unary_op(d3)
            d4 = unary_op(d4)
        else:
            d1, d2, d3, d4 = (
                unary_op(d1),
                unary_op(d2),
                unary_op(d1 + 0.001),
                unary_op(d4),
            )
        a = binary_op(d1, d2)
        b = binary_op(d3, d4)
        c = a + b
        return c

    def forward(self, d1, d2, d3, d4):
        # [FUNCTION DOC] Runs the shared computation with the torch functions configured on
        # the class.
        binary_op = self.__class__.binary_op_pt_func
        unary_op = self.__class__.unary_op_pt_func
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def reference(self):
        # [FUNCTION DOC] Runs the shared computation on numpy copies of the inputs with the
        # numpy functions configured on the class; used as the reference.
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        # [FUNCTION DOC] Returns the sizes M, N and K as a list.
        return [self.M, self.N, self.K]

    @classmethod
    def module(cls):
        # [FUNCTION DOC] Class method: returns the module name, a fixed prefix followed by the
        # class operation name.
        return "broadcast_" + cls.op_str

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with entries sol and algorithmic: a buffer of M times
        # N times K times 4, with sol factor 1 in all modes and algorithmic factor 1 in forward
        # mode or the number of inputs in both mode.
        input_count = len(self.inputs)
        if self.mode == "fwd":
            if self.split_input:
                sol_count = 1
                algorithmic_count = 1
            else:
                sol_count = 1
                algorithmic_count = 1
        else:
            if self.split_input:
                sol_count = 1
                algorithmic_count = input_count
            else:
                sol_count = 1
                algorithmic_count = input_count

        buffer_size = self.M * self.N * self.K * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns one default config: M 256, N 128, K 512.
        return [[1 << 8, 1 << 7, 1 << 9]]


def register_broadcast_ops():
    # [FUNCTION DOC] Generates and registers the benchmark subclasses. For every combination of
    # the split flag (separate or shared third input) and each entry of the binary operator
    # list, and likewise for the unary list, creates a subclass of the template named after the
    # operation with the torch and numpy functions set and registers it with the harness.
    # Entries with two items use the same function for torch and numpy. Name strings are
    # prefixed with split or shared. Called once at import.
    binary_op_list = [
        ["mul", operator.mul],
        ["add", operator.add],
        ["sub", operator.sub],
        ["div", lambda a, b: a / (b + 1e-4)],
        [
            "pow",
            torch.pow,
            np.power,
        ],  # no fuson triggered
        ["max", torch.max, np.maximum],
        ["min", torch.min, np.minimum],
    ]

    unary_op_list = [
        ["erf", torch.erf, np.erf],
        ["exp", torch.exp, np.exp],
        ["sin", torch.sin, np.sin],
        ["cos", torch.cos, np.cos],
    ]

    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # Make a copy of BroadcastBench
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # Make a copy of BroadcastBench
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)


register_broadcast_ops()
