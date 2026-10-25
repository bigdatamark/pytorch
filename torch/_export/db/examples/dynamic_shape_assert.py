# [MODULE DOC] ExportDB example: Python assert statements on a dynamic shape.
# Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class DynamicShapeAssert(torch.nn.Module):
    # [CLASS DOC] Stateless example module asserting facts about the input shape.
    """
    A basic usage of python assertion.
    """

    def forward(self, x):
        # [FUNCTION DOC] Asserts the first dimension of x is greater than 2 (with
        # a message) and greater than 1 (without); raises AssertionError
        # otherwise. Returns x unchanged.
        # assertion with error message
        assert x.shape[0] > 2, f"{x.shape[0]} is greater than 2"  # noqa: S101
        # assertion without error message
        assert x.shape[0] > 1  # noqa: S101
        return x

example_args = (torch.randn(3, 2),)
tags = {"python.assert"}
model = DynamicShapeAssert()
