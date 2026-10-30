# [MODULE DOC] Fixture module with TorchScript-compiled artifacts for package tests: a scripted
# class, a scripted function using it, and eager helper classes taking part in isinstance
# checks. Import-time side effect: the jit script decorator compiles the class and the function
# when the module is imported.
from typing import Any

import torch


@torch.jit.script
class MyScriptClass:
    # [CLASS DOC] HELPER: TorchScript class (compiled by the script decorator at import) holding
    # one attribute named foo that is set at construction and can be replaced with set_foo.
    """Intended to be scripted."""

    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the argument as the foo attribute.
        self.foo = x

    def set_foo(self, x):
        # [FUNCTION DOC] HELPER: Replaces the foo attribute with the argument.
        self.foo = x


@torch.jit.script
def uses_script_class(x):
    # [FUNCTION DOC] HELPER: Scripted function; builds a MyScriptClass from the argument and
    # returns the stored foo attribute, i.e. the argument itself.
    """Intended to be scripted."""
    foo = MyScriptClass(x)
    return foo.foo


class IdListFeature:
    # [CLASS DOC] HELPER: Plain eager class holding a one-by-one tensor of ones as id_list.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets id_list to a one-by-one tensor filled with ones.
        self.id_list = torch.ones(1, 1)

    def returns_self(self) -> "IdListFeature":
        # [FUNCTION DOC] HELPER: Returns a new IdListFeature instance (not the same object).
        return IdListFeature()


class UsesIdListFeature(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward branches on the type of its input.
    def forward(self, feature: Any):
        # [FUNCTION DOC] HELPER: If the argument is an IdListFeature returns its id_list tensor;
        # otherwise returns the argument unchanged.
        if isinstance(feature, IdListFeature):
            return feature.id_list
        else:
            return feature
