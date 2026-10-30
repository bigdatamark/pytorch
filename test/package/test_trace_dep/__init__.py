# [PACKAGE DOC] Fixture package for the dependency analysis test in the package tests. It
# imports the third party yaml library (unused) and torch, and defines one small module class.
# The analysis test checks that tracing the class dependencies reports this package but not
# yaml, since yaml is imported yet never used by the traced object.
import yaml

import torch


class SumMod(torch.nn.Module):
    # [CLASS DOC] HELPER: Parameterless module whose forward reduces the input to its total
    # sum.
    def forward(self, inp):
        # [FUNCTION DOC] HELPER: Returns the sum of all elements of the input tensor as a
        # scalar tensor.
        return torch.sum(inp)
