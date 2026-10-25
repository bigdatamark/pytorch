# [MODULE DOC] ExportDB example: functorch map over the first tensor dimension.
# Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import map

class DynamicShapeMap(torch.nn.Module):
    # [CLASS DOC] Stateless example module mapping an add over rows of xs.
    """
    functorch map() maps a function over the first tensor dimension.
    """

    def forward(self, xs, y):
        # [FUNCTION DOC] Maps a body that adds y over each slice of xs along the
        # first dimension and returns the stacked results.
        def body(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y for one slice.
            return x + y

        return map(body, xs, y)

example_args = (torch.randn(3, 2), torch.randn(2))
tags = {"torch.dynamic-shape", "torch.map"}
model = DynamicShapeMap()
