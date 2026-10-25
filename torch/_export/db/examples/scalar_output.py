# [MODULE DOC] ExportDB example: returning a scalar derived from a dynamic
# dimension. Defines example_args, tags, dynamic_shapes and model, with a
# module-level tensor x and Dim dim1_x.
# mypy: allow-untyped-defs
import torch

from torch.export import Dim

x = torch.randn(3, 2)
dim1_x = Dim("dim1_x")

class ScalarOutput(torch.nn.Module):
    # [CLASS DOC] Stateless example module whose output is a shape-derived integer.
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    def __init__(self) -> None:
        # [FUNCTION DOC] Plain initializer; adds no state.
        super().__init__()

    def forward(self, x):
        # [FUNCTION DOC] Returns the size of the second dimension of x plus one
        # as a scalar (symbolic integer when that dimension is dynamic).
        return x.shape[1] + 1

example_args = (x,)
tags = {"torch.dynamic-shape"}
dynamic_shapes = {"x": {1: dim1_x}}
model = ScalarOutput()
