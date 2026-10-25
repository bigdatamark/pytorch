# [MODULE DOC] ExportDB example: cond branches that close over a variable of
# the enclosing function. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondClosedOverVariable(torch.nn.Module):
    # [CLASS DOC] Stateless example module whose cond branches capture x.
    """
    torch.cond() supports branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        # [FUNCTION DOC] Runs cond on pred with branches that use the captured x
        # and ignore their operand (x plus 1). Returns x times 2 or x minus 2.
        def true_fn(val):
            # [FUNCTION DOC] HELPER: returns the captured x times 2 (val unused).
            return x * 2

        def false_fn(val):
            # [FUNCTION DOC] HELPER: returns the captured x minus 2 (val unused).
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1])

example_args = (torch.tensor(True), torch.randn(3, 2))
tags = {"torch.cond", "python.closure"}
model = CondClosedOverVariable()
