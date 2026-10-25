# [MODULE DOC] ExportDB example: an if statement on a static property (rank).
# Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class StaticIf(torch.nn.Module):
    # [CLASS DOC] Stateless example module branching on the rank of its input.
    """
    `if` statement with static predicate value should be traced through with the
    taken branch.
    """

    def forward(self, x):
        # [FUNCTION DOC] If x has exactly three dimensions, returns x plus a
        # ones tensor of shape 1x1x1 (broadcast); otherwise returns x. Only the
        # taken branch appears in the trace.
        if len(x.shape) == 3:
            return x + torch.ones(1, 1, 1)

        return x

example_args = (torch.randn(3, 2, 2),)
tags = {"python.control-flow"}
model = StaticIf()
