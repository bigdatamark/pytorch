# [MODULE DOC] ExportDB example: a boolean expression on shapes used as the
# cond predicate. Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondPredicate(torch.nn.Module):
    # [CLASS DOC] Stateless example module using a shape-based predicate.
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        # [FUNCTION DOC] Builds a predicate that is true when x has more than 2
        # dimensions and its third dimension exceeds 10, then runs cond to
        # return cosine of x (true) or sine of x (false).
        pred = x.dim() > 2 and x.shape[2] > 10

        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

example_args = (torch.randn(6, 4, 3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondPredicate()
