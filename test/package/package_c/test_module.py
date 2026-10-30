# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Fixture that defines a module wrapping a torchvision resnet18 only if torchvision
# can be imported (the import error is swallowed otherwise), plus a plain leaf
# addition function.

import torch


try:
    from torchvision.models import resnet18

    class TorchVisionTest(torch.nn.Module):
        # [CLASS DOC] HELPER: Defined only when torchvision is available. Holds a resnet18
        # instance as tvmod, but forward does not call it; it applies the leaf function and a
        # relu instead.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Initializes the base module and creates a resnet18 model
            # as tvmod.
            super().__init__()
            self.tvmod = resnet18()

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Calls a_non_torch_leaf on the input with itself, adds 3.0
            # and applies relu, returning the result.
            x = a_non_torch_leaf(x, x)
            return torch.relu(x + 3.0)

except ImportError:
    pass


def a_non_torch_leaf(a, b):
    # [FUNCTION DOC] HELPER: Returns the sum of the two arguments.
    return a + b
