# [MODULE DOC] ExportDB example: an optional input with a tensor default.
# Defines example_args, tags, support_level and model.
# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class OptionalInput(torch.nn.Module):
    # [CLASS DOC] Stateless example module with a default-valued second input.
    # KNOWN ISSUE: the docstring says this is not supported yet, but the case
    # is registered with support level SUPPORTED.
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=torch.randn(2, 3)):
        # [FUNCTION DOC] Returns x plus y if y is not None, otherwise x. The
        # default y is one random tensor created once at definition time.
        if y is not None:
            return x + y
        return x


example_args = (torch.randn(2, 3),)
tags = {"python.object-model"}
support_level = SupportLevel.SUPPORTED
model = OptionalInput()
