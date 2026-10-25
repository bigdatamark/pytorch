# [MODULE DOC] ExportDB example: a view whose target shape is computed from the
# input shape. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

class DynamicShapeView(torch.nn.Module):
    # [CLASS DOC] Stateless example module reshaping and permuting its input.
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    def forward(self, x):
        # [FUNCTION DOC] Reshapes x so its last dimension becomes two dimensions
        # of size 2 and 5 (requires the last dimension to equal 10), then swaps
        # the last two dimensions and returns the permuted tensor.
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

example_args = (torch.randn(10, 10),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeView()
