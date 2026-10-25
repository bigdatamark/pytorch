# [MODULE DOC] ExportDB example: list membership tests on shapes and constants
# inside assertions. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class ListContains(torch.nn.Module):
    # [CLASS DOC] Stateless example module asserting membership conditions.
    """
    List containment relation can be checked on a dynamic shape or constants.
    """

    def forward(self, x):
        # [FUNCTION DOC] Asserts the last dimension of x is 6 or 2, the first
        # dimension is not 4, 5 or 6, and a constant string is not in a list
        # (raises AssertionError on failure). Returns x plus x.
        assert x.size(-1) in [6, 2]  # noqa: S101
        assert x.size(0) not in [4, 5, 6]  # noqa: S101
        assert "monkey" not in ["cow", "pig"]  # noqa: S101
        return x + x

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape", "python.data-structure", "python.assert"}
model = ListContains()
