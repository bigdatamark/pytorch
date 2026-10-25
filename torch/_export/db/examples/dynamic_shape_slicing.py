# [MODULE DOC] ExportDB example: slicing with bounds derived from the input
# shape. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class DynamicShapeSlicing(torch.nn.Module):
    # [CLASS DOC] Stateless example module slicing a 2-D input by shape-based bounds.
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns rows up to (first dimension minus 2) and columns
        # starting at (second dimension minus 1) with step 2.
        return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeSlicing()
