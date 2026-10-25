# [MODULE DOC] ExportDB example: module attributes (a string and an int) that
# export specializes on. Defines example_args and model.
# mypy: allow-untyped-defs
from enum import Enum

import torch

class Animal(Enum):
    # [CLASS DOC] HELPER: enum with the single member COW whose value is the string moo.
    COW = "moo"

class SpecializedAttribute(torch.nn.Module):
    # [CLASS DOC] Example module with attributes a (string) and b (integer)
    # that are baked into the traced graph.
    """
    Model attributes are specialized.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Sets attribute a to the string moo and attribute b to 4.
        super().__init__()
        self.a = "moo"
        self.b = 4

    def forward(self, x):
        # [FUNCTION DOC] If attribute a equals the COW enum value, returns x
        # squared plus b; otherwise raises ValueError.
        if self.a == Animal.COW.value:
            return x * x + self.b
        else:
            raise ValueError("bad")

example_args = (torch.randn(3, 2),)
model = SpecializedAttribute()
