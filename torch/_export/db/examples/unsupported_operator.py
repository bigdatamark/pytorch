# [MODULE DOC] ExportDB example: use of the sym_min operator on a shape. The
# file name says unsupported but the case is marked SUPPORTED. Defines
# example_args, tags, support_level and model.
# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class TorchSymMin(torch.nn.Module):
    # [CLASS DOC] Stateless example module combining a sum with a symbolic minimum.
    """
    torch.sym_min operator is supported in export.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns the sum of all elements of x plus the minimum of
        # the first dimension of x and 100.
        return x.sum() + torch.sym_min(x.size(0), 100)


example_args = (torch.randn(3, 2),)
tags = {"torch.operator"}
support_level = SupportLevel.SUPPORTED
model = TorchSymMin()
