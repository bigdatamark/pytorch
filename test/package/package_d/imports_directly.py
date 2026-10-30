# [MODULE DOC] Fixture module that imports a string constant directly from the nested
# subsubpackage via a relative import, and uses it as a class attribute.
import torch

from .subpackage_0.subsubpackage_0 import important_string


class ImportsDirectlyFromSubSubPackage(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a class attribute key holding the imported string; forward
    # returns the sum of all elements of the input.
    key = important_string

    def forward(self, inp):
        # [FUNCTION DOC] HELPER: Returns the sum of all elements of the input tensor.
        return torch.sum(inp)
