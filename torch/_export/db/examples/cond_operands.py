# [MODULE DOC] ExportDB example: passing a list of tensors as cond operands with
# a dynamic first dimension on x. Defines example_args, tags, extra_inputs,
# dynamic_shapes and model (module-level tensors x, y and Dim dim0_x).
# mypy: allow-untyped-defs
import torch

from torch.export import Dim

x = torch.randn(3, 2)
y = torch.randn(2)
dim0_x = Dim("dim0_x")

class CondOperands(torch.nn.Module):
    # [CLASS DOC] Stateless example module running cond with operands x and y.
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x, y):
        # [FUNCTION DOC] Runs torch.cond on whether the first dimension of x
        # exceeds 2, with operands x and y. Returns x plus y or x minus y.
        def true_fn(x, y):
            # [FUNCTION DOC] HELPER: returns x plus y.
            return x + y

        def false_fn(x, y):
            # [FUNCTION DOC] HELPER: returns x minus y.
            return x - y

        return torch.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

example_args = (x, y)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
extra_inputs = (torch.randn(2, 2), torch.randn(2))
dynamic_shapes = {"x": {0: dim0_x}, "y": None}
model = CondOperands()
