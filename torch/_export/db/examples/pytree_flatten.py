# [MODULE DOC] ExportDB example: using the pytree flatten utility inside
# forward. Defines example_args and model.
# KNOWN ISSUE: example_args ends with a stray trailing comma, so it is a tuple
# wrapping the intended one-element args tuple, i.e. an extra nesting level.
# mypy: allow-untyped-defs
import torch

from torch.utils import _pytree as pytree

class PytreeFlatten(torch.nn.Module):
    # [CLASS DOC] Stateless example module flattening a dict input.
    """
    Pytree from PyTorch can be captured by TorchDynamo.
    """

    def forward(self, x):
        # [FUNCTION DOC] Flattens the nested structure x into a list of leaves
        # (discarding the spec) and returns the first leaf plus one.
        y, _spec = pytree.tree_flatten(x)
        return y[0] + 1

example_args = ({1: torch.randn(3, 2), 2: torch.randn(3, 2)},),
model = PytreeFlatten()
