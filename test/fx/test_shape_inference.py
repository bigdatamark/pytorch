# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for the FX shape inference helpers: infer_symbol_values (solves symbolic
# dimension values from error message constraint strings) and infer_shape (runs shape inference
# over a traced graph module). Running the file directly raises a RuntimeError since it is not
# used by test discovery.

import copy
import unittest
from collections import defaultdict

import torch
import torch.fx as fx
from torch._dynamo.source import LocalSource
from torch.fx.experimental.shape_inference.infer_shape import infer_shape
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv


class TestShapeInference(unittest.TestCase):
    # [CLASS DOC] unittest based test case with one test for infer_symbol_values and one smoke
    # test for infer_shape.
    def test_infer_symbol_values(self):
        # [FUNCTION DOC] Creates a ShapeEnv with eight dynamic symbolic ints named s0 to s7,
        # each with hint two, and keeps a deep copy as the initial values. Feeds a series of
        # runtime error message strings (size mismatches, invalid reshapes, matmul reduction
        # dimension mismatches, split size sums) to infer_symbol_values one by one, which
        # updates the working symint list in place. Finally asserts symbols one through seven
        # were resolved to 1773, 216, 640, 4160, 40, 160 and 4258.
        def mksym(shape_env, value, source, dynamic_dim) -> None:
            # [FUNCTION DOC] HELPER: Builds a symbolic integer node in the given shape env:
            # creates a symbol with the given value, source and dynamic dim, and wraps it with
            # the same value as hint and the same source.
            return shape_env.create_symintnode(
                shape_env.create_symbol(
                    value,
                    source=source,
                    dynamic_dim=dynamic_dim,
                ),
                hint=value,
                source=source,
            )

        shape_env = ShapeEnv()
        N = 8
        sample = {f"s{i}": 2 for i in range(N)}
        init_symints = [
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        symints = copy.deepcopy(init_symints)
        symbol_to_idx_dict = {f"s{i}": i for i in range(N)}
        padding_constraints = defaultdict(list)

        # prepare constraints strings
        constraints = []
        constraints.append(
            "The size of tensor a (s1) must match the size of tensor b (1773) at non-singleton dimension 1)"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, (s2//2) + 12] but got: [s0, 120]."
        )
        constraints.append("shape '[s0, -1, 32]' is invalid for input of size s0*s3")
        constraints.append(
            "a and b must have same reduction dim, but got [32*s0, s3] X [20, 15]."
        )
        constraints.append(
            "a and b must have same reduction dim, but got [s0, s4 + 1568] X [5728, 1024]."
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 40] but got: [s0, s5]."
        )
        constraints.append(
            "shape '[s0, -1, 32]' is invalid for input of size s0*s6 + 1344*s0"
        )
        constraints.append(
            "shape '[-1, 47]' is invalid for input of size 32*s0*s6 + 1344*s0"
        )
        constraints.append(
            "Expected size for first two dimensions of batch2 tensor to be: [s0, 47*s6] but got: [s0*s6, 47]."
        )
        constraints.append("Split sizes add up to 4258 but got the tensor's size of s7")

        for constraint in constraints:
            infer_symbol_values(
                symints,
                init_symints,
                symbol_to_idx_dict,
                padding_constraints,
                constraint,
            )

        self.assertEqual(symints[1], 1773)
        self.assertEqual(symints[2], 216)
        self.assertEqual(symints[3], 640)
        self.assertEqual(symints[4], 4160)
        self.assertEqual(symints[5], 40)
        self.assertEqual(symints[6], 160)
        self.assertEqual(symints[7], 4258)

    def test_infer_shape(self):
        # [FUNCTION DOC] Smoke test: traces a small two linear layer model with sigmoid and tanh
        # into a graph module and calls infer_shape with a single input tensor of shape 1 by 1.
        # Only checks that no exception is raised.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear style weights and biases (256 by 328
            # and 328 by 256) stored as uninitialized tensors, applying linear, sigmoid,
            # linear and tanh.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Allocates uninitialized tensors for the first and
                # second weights and biases as plain attributes.
                super().__init__()
                self.w_1 = torch.empty([256, 328])
                self.b_1 = torch.empty([256])
                self.w_2 = torch.empty([328, 256])
                self.b_2 = torch.empty([328])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies a linear layer with w_1 and b_1, sigmoid, a
                # linear layer with w_2 and b_2, then tanh, and returns the result.
                l_1 = torch.nn.functional.linear(x, self.w_1, bias=self.b_1)
                s_1 = torch.sigmoid(l_1)
                l_2 = torch.nn.functional.linear(s_1, self.w_2, bias=self.b_2)
                t_1 = torch.tanh(l_2)
                return t_1

        def generate_graph_module(model):
            # [FUNCTION DOC] HELPER: Symbolically traces the given model and returns
            # the resulting GraphModule.
            gm = fx.symbolic_trace(model)
            return gm

        m = TestModule()
        gm = generate_graph_module(m)
        input_tensors = [torch.randn(1, 1)]
        infer_shape(gm, input_tensors)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
