# [MODULE DOC] ExportDB example: bounding a data-dependent value with torch
# checks so it can be used as a tensor size. Defines example_args, tags, model.
# mypy: allow-untyped-defs
import torch


class ConstrainAsSizeExample(torch.nn.Module):
    # [CLASS DOC] Stateless example module that builds a tensor whose size comes
    # from an input scalar bounded between 0 and 5.
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check APIs.
    """

    def forward(self, x):
        # [FUNCTION DOC] Reads the scalar a from x, asserts 0 <= a <= 5 through
        # torch._check, and returns a zero tensor of shape (a, 5).
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)
        return torch.zeros((a, 5))


example_args = (torch.tensor(4),)
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsSizeExample()
