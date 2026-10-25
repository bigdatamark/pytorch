# [MODULE DOC] ExportDB example: a nested function that mutates a captured
# local via nonlocal. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class NestedFunction(torch.nn.Module):
    # [CLASS DOC] Stateless example module defining and calling a closure.
    """
    Nested functions are traced through. Side effects on global captures
    are not supported though.
    """

    def forward(self, a, b):
        # [FUNCTION DOC] Computes x as a plus b and z as a minus b, then returns
        # the closure applied to x.
        x = a + b
        z = a - b

        def closure(y):
            # [FUNCTION DOC] HELPER: increments the captured x by one (rebinding
            # it) and returns x times y plus the captured z.
            nonlocal x
            x += 1
            return x * y + z

        return closure(x)

example_args = (torch.randn(3, 2), torch.randn(2))
tags = {"python.closure"}
model = NestedFunction()
