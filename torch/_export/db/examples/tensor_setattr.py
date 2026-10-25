# [MODULE DOC] ExportDB example: calling setattr on a tensor with a runtime
# attribute name. Defines example_args (tensor and name string), tags, model.
# mypy: allow-untyped-defs
import torch


class TensorSetattr(torch.nn.Module):
    # [CLASS DOC] Stateless example module that mutates an input tensor object.
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr):
        # [FUNCTION DOC] Side effect: sets the attribute named by the string attr
        # on tensor x to a new random 3x2 tensor. Returns x plus 4.
        setattr(x, attr, torch.randn(3, 2))
        return x + 4

example_args = (torch.randn(3, 2), "attr")
tags = {"python.builtin"}
model = TensorSetattr()
