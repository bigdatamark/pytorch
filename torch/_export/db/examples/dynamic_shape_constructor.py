# [MODULE DOC] ExportDB example: a tensor constructor sized from an input
# dimension. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class DynamicShapeConstructor(torch.nn.Module):
    # [CLASS DOC] Stateless example module building a tensor from a shape expression.
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """

    def forward(self, x):
        # [FUNCTION DOC] Returns a one-dimensional zero tensor whose length is
        # twice the first dimension of x.
        return torch.zeros(x.shape[0] * 2)

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeConstructor()
