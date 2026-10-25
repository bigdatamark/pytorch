# [MODULE DOC] ExportDB example: bounding a data-dependent value with torch
# checks so a branch on it can be resolved. Defines example_args, tags, model.
# mypy: allow-untyped-defs
import torch


class ConstrainAsValueExample(torch.nn.Module):
    # [CLASS DOC] Stateless example module that branches on an input scalar
    # bounded between 0 and 5.
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check API.
    """

    def forward(self, x, y):
        # [FUNCTION DOC] Reads scalar a from x, asserts 0 <= a <= 5 through
        # torch._check, and since that implies a < 6 always returns sin of y
        # (the cosine branch is unreachable).
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)

        if a < 6:
            return y.sin()
        return y.cos()


example_args = (torch.tensor(4), torch.randn(5, 5))
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsValueExample()
