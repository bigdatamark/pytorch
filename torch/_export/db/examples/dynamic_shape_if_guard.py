# [MODULE DOC] ExportDB example: a Python if on a backed dynamic shape, which
# specializes to one branch with a guard. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class DynamicShapeIfGuard(torch.nn.Module):
    # [CLASS DOC] Stateless example module branching on the first dimension of its input.
    """
    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    dimension is marked as dynamic shape from higher level API.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns cosine of x if its first dimension equals 3,
        # otherwise sine of x. Export specializes on the observed dimension.
        if x.shape[0] == 3:
            return x.cos()

        return x.sin()

example_args = (torch.randn(3, 2, 2),)
tags = {"torch.dynamic-shape", "python.control-flow"}
model = DynamicShapeIfGuard()
