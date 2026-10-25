# [MODULE DOC] ExportDB example: building and reading a local dictionary during
# tracing. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class Dictionary(torch.nn.Module):
    # [CLASS DOC] Stateless example module that uses a temporary dict.
    """
    Dictionary structures are inlined and flattened along tracing.
    """

    def forward(self, x, y):
        # [FUNCTION DOC] Stores the square of x in a local dict, multiplies y by
        # it, and returns a dict with the single key y holding the product.
        elements = {}
        elements["x2"] = x * x
        y = y * elements["x2"]
        return {"y": y}

example_args = (torch.randn(3, 2), torch.tensor(4))
tags = {"python.data-structure"}
model = Dictionary()
