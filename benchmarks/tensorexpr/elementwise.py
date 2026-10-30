# [MODULE DOC] Elementwise tensor-expression benchmarks. Defines a templated base benchmark
# customized per operator, a function that generates and registers one subclass per binary or
# unary operator in split and shared input variants, and two simple add benchmarks (static and
# dynamic shape). Importing the module registers all of them.
import itertools
import operator

import numpy as np
import scipy.special

import torch

from . import benchmark


# A template class for elementwise operations.
# A derived class will override the class instance to customize its behavior.
class ElementBench(benchmark.Benchmark):
    # [CLASS DOC] Template benchmark for elementwise operators. Subclasses are generated at
    # import time and set the class variables for the operator name, torch and numpy binary
    # and unary functions and whether the four inputs are independent or derived from one.
    # Holds the length N and four random input vectors.
    # List of customization class variables.
    op_str = None
    binary_op_pt_func = None
    binary_op_np_func = None
    unary_op_pt_func = None
    unary_op_np_func = None
    split_input = True

    def __init__(self, mode, device, dtype, N):
        # [FUNCTION DOC] Stores the length and creates four random vectors on the given device
        # and dtype (gradient-tracking in the both mode), sets the inputs list, and marks the
        # benchmark deterministic unless the operator name contains rand.
        super().__init__(mode, device, dtype)
        self.N = N
        self.d1 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d3 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d4 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.d1, self.d2, self.d3, self.d4]
        self.deterministic = "rand" not in self.op_str

    def _eval(self, d1, d2, d3, d4, binary_op, unary_op):
        # [FUNCTION DOC] HELPER: Shared computation used by both forward and reference.
        # Applies the unary op to the inputs (independent inputs, or three shifted copies of
        # the first when inputs are shared), combines pairs with the binary op, and returns
        # the sum of the two results. Missing ops default to addition and identity.
        if not binary_op:

            def binary_op(x, y):
                # [FUNCTION DOC] HELPER: Default binary op: returns the sum of its two
                # arguments.
                return x + y

        if not unary_op:

            def unary_op(x):
                # [FUNCTION DOC] HELPER: Default unary op: returns its argument unchanged.
                return x

        if self.split_input:
            d1 = unary_op(d1)
            d2 = unary_op(d2)
            d3 = unary_op(d3)
            d4 = unary_op(d4)
        else:
            d2 = unary_op(d1 + 0.001)
            d3 = unary_op(d1 + 0.002)
            d4 = unary_op(d1 + 0.003)
            d1 = unary_op(d1)
        a = binary_op(d1, d2)
        b = binary_op(d3, d4)
        c = a + b
        return c

    def forward(self, d1, d2, d3, d4):
        # [FUNCTION DOC] Runs the shared computation with the torch binary and unary functions
        # of the concrete subclass and returns the result.
        binary_op = self.__class__.binary_op_pt_func
        unary_op = self.__class__.unary_op_pt_func
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def reference(self):
        # [FUNCTION DOC] Runs the shared computation on numpy copies of the inputs with the
        # numpy binary and unary functions of the subclass, giving the expected result.
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        # [FUNCTION DOC] Returns the configuration list holding only the vector length.
        return [self.N]

    @classmethod
    def module(cls):
        # [FUNCTION DOC] Returns the module name, built by prefixing the operator name with a
        # fixed element prefix.
        return "element_" + cls.op_str

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements, computed from the vector length and the number of inputs with different
        # counts for fwd versus both, split versus shared inputs. Random-generating operators
        # are counted as a single access.
        input_count = len(self.inputs)
        if self.mode == "fwd":
            if self.split_input:
                sol_count = input_count + 1
                algorithmic_count = input_count + 1
            else:
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1
        else:
            if self.split_input:
                sol_count = (input_count + 1) + (1 + input_count)
                algorithmic_count = (input_count + 1) + ((2 + 1) * input_count)
            else:
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1

        buffer_size = self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry with a vector length of
        # two to the power 25.
        return [[1 << 25]]


def register_element_ops():
    # [FUNCTION DOC] Generates the elementwise benchmark subclasses and registers them. For
    # every combination of split or shared inputs with each binary operator, and then each
    # unary operator, it makes a new subclass of the template named after the operator, sets
    # its class variables (names, torch and numpy functions, split flag), and registers it. A
    # two item entry reuses the torch function as the numpy one.
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
        ["erf", torch.erf, scipy.special.erf],
        ["exp", torch.exp, np.exp],
        ["sin", torch.sin, np.sin],
        ["cos", torch.cos, np.cos],
        ["rand_like", torch.rand_like, lambda x: np.random.rand(*x.shape)],
    ]

    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # Make a copy of ElementBench
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # Make a copy of ElementBench
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)


# benchmark.register_benchmark_class(ElementMulBench)
register_element_ops()


class SimpleElementBench(benchmark.Benchmark):
    # [CLASS DOC] Minimal benchmark that adds two constants in sequence to one random vector,
    # used to study fusion of simple elementwise chains. Holds the length and the single
    # input.
    def __init__(self, mode, device, dtype, N):
        # [FUNCTION DOC] Stores the length, creates one random vector (gradient-tracking in
        # the both mode) and sets the inputs list to it.
        super().__init__(mode, device, dtype)
        self.N = N
        self.data = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.data]

    def forward(self, data):
        # [FUNCTION DOC] Adds one small constant to the input, then another, and returns the
        # result.
        a = data + 0.001
        b = a + 0.002
        return b

    def reference(self):
        # [FUNCTION DOC] Intended to compute the expected result.
        # KNOWN ISSUE: this method is copied from the template class and reads class
        # attributes and instance fields that do not exist on this class, so calling it raises
        # an error; it is never reached because the benchmark is not deterministic.
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        # [FUNCTION DOC] Returns the configuration list holding only the vector length.
        return [self.N]

    @staticmethod
    def input_iterable():
        # [FUNCTION DOC] Returns True to tell the harness that this benchmark's input can be
        # iterated over by the dynamic shape machinery.
        return True

    @classmethod
    def module(cls):
        # [FUNCTION DOC] Returns the module name string for this benchmark.
        return "simple_element"

    def memory_workload(self):
        # [FUNCTION DOC] Returns a mapping with sol and algorithmic memory traffic in
        # elements: twice the vector length for either mode.
        if self.mode == "fwd":
            sol_count = 2
            algorithmic_count = 2
        else:
            sol_count = 2
            algorithmic_count = 2

        buffer_size = self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Returns the default configurations: one entry with a vector length of
        # two to the power 25.
        return [[1 << 25]]


benchmark.register_benchmark_class(SimpleElementBench)


class DynamicSimpleElementBench(benchmark.DynamicShape, SimpleElementBench):
    # [CLASS DOC] Dynamic shape version of the simple elementwise benchmark. Mixes in the
    # dynamic shape helper so that many inputs of randomized length are precomputed and cycled
    # during the run.
    def __init__(self, mode, device, dtype, N):
        # [FUNCTION DOC] Initializes the dynamic shape state first, then the simple benchmark
        # base with the given mode, device, dtype and length.
        benchmark.DynamicShape.__init__(self)
        SimpleElementBench.__init__(self, mode, device, dtype, N)

    @classmethod
    def module(cls):
        # [FUNCTION DOC] Returns the module name string for the dynamic variant.
        return "simple_dynamic_element"

    def instantiate_input(self):
        # [FUNCTION DOC] Draws a randomized length near the configured one, creates a random
        # vector of that length and assigns the inputs list to a one element list holding it.
        # The dynamic shape helper turns each assignment into a new stored sample.
        (N,) = self.rand_shape([self.N])
        data = self.rand(
            [N], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.inputs = [data]


benchmark.register_benchmark_class(DynamicSimpleElementBench)
