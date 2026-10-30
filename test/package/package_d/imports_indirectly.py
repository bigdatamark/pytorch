# [MODULE DOC] Fixture module that imports a string constant from the intermediate subpackage
# (which re-exports it from the nested subsubpackage) and uses it as a class attribute.
import torch

from .subpackage_0 import important_string


class ImportsIndirectlyFromSubPackage(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a class attribute key holding the imported string; forward
    # returns the sum of all elements of the input.
    key = important_string

    def forward(self, inp):
        # [FUNCTION DOC] HELPER: Returns the sum of all elements of the input tensor.
        return torch.sum(inp)
