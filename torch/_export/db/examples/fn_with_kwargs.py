# [MODULE DOC] ExportDB example: forward with positional, tuple, variadic
# positional, keyword-only and variadic keyword parameters. Defines
# example_args, example_kwargs, tags and model.
# mypy: allow-untyped-defs
import torch

class FnWithKwargs(torch.nn.Module):
    # [CLASS DOC] Stateless example module multiplying all of its inputs together.
    # KNOWN ISSUE: the docstring says keyword arguments are unsupported, yet the
    # case is registered with the default supported level and example kwargs.
    """
    Keyword arguments are not supported at the moment.
    """

    def forward(self, pos0, tuple0, *myargs, mykw0, **mykwargs):
        # [FUNCTION DOC] Multiplies pos0 by every tensor in tuple0, every extra
        # positional tensor, the keyword-only tensor mykw0, and the extra
        # keyword tensors input0 and input1 (KeyError if absent). Returns the product.
        out = pos0
        for arg in tuple0:
            out = out * arg
        for arg in myargs:
            out = out * arg
        out = out * mykw0
        out = out * mykwargs["input0"] * mykwargs["input1"]
        return out

example_args = (
    torch.randn(4),
    (torch.randn(4), torch.randn(4)),
    *[torch.randn(4), torch.randn(4)]
)
example_kwargs = {
    "mykw0": torch.randn(4),
    "input0": torch.randn(4),
    "input1": torch.randn(4),
}
tags = {"python.data-structure"}
model = FnWithKwargs()
