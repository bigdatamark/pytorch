# [MODULE DOC] ExportDB example (marked not supported yet): calling round on a
# dynamic shape. Defines example_args, tags, support_level, dynamic_shapes and
# model, with a module-level tensor x and Dim dim0_x.
# mypy: allow-untyped-defs
import torch

from torch._export.db.case import SupportLevel
from torch.export import Dim

class DynamicShapeRound(torch.nn.Module):
    # [CLASS DOC] Stateless example module slicing by a rounded half of a dimension.
    """
    Calling round on dynamic shapes is not supported.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns the leading rows of x, with the count being the
        # first dimension divided by two and rounded.
        return x[: round(x.shape[0] / 2)]

x = torch.randn(3, 2)
dim0_x = Dim("dim0_x")
example_args = (x,)
tags = {"torch.dynamic-shape", "python.builtin"}
support_level = SupportLevel.NOT_SUPPORTED_YET
dynamic_shapes = {"x": {0: dim0_x}}
model = DynamicShapeRound()
