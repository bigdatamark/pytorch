# [MODULE DOC] ExportDB example: unpacking a list input with star syntax.
# Defines example_args, tags and model.
# mypy: allow-untyped-defs

import torch

class ListUnpack(torch.nn.Module):
    # [CLASS DOC] Stateless example module unpacking a list of tensors.
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """

    def forward(self, args: list[torch.Tensor]):
        # [FUNCTION DOC] Splits the list into its first element and the rest and
        # returns the first element plus the second element of the list.
        """
        Lists are treated as static construct, therefore unpacking should be
        erased after tracing.
        """
        x, *y = args
        return x + y[0]

example_args = ([torch.randn(3, 2), torch.tensor(4), torch.tensor(5)],)
tags = {"python.control-flow", "python.data-structure"}
model = ListUnpack()
