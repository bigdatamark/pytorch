# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Fixture module collection for FX and package tests: several small nn.Module
# wrappers that compose submodules and tensors, a module calling a custom function, and that
# function. Import-time side effect: registers the function named a_non_torch_leaf as a wrapped
# (leaf) function with FX symbolic tracing.

import torch
from torch.fx import wrap


wrap("a_non_torch_leaf")


class ModWithSubmod(torch.nn.Module):
    # [CLASS DOC] HELPER: Wraps one callable submodule and forwards the input through it.
    def __init__(self, script_mod):
        # [FUNCTION DOC] HELPER: Stores the given module as the script_mod submodule.
        super().__init__()
        self.script_mod = script_mod

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the submodule applied to the input.
        return self.script_mod(x)


class ModWithTensor(torch.nn.Module):
    # [CLASS DOC] HELPER: Holds a tensor as a plain attribute and multiplies the input by it.
    def __init__(self, tensor):
        # [FUNCTION DOC] HELPER: Stores the tensor as an ordinary attribute (not a registered
        # parameter or buffer).
        super().__init__()
        self.tensor = tensor

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the elementwise product of the stored tensor and
        # the input.
        return self.tensor * x


class ModWithSubmodAndTensor(torch.nn.Module):
    # [CLASS DOC] HELPER: Holds a tensor and a submodule; the output is the submodule result
    # plus the tensor.
    def __init__(self, tensor, sub_mod):
        # [FUNCTION DOC] HELPER: Stores the tensor attribute and registers the given module
        # as sub_mod.
        super().__init__()
        self.tensor = tensor
        self.sub_mod = sub_mod

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the submodule applied to the input plus the
        # stored tensor.
        return self.sub_mod(x) + self.tensor


class ModWithTwoSubmodsAndTensor(torch.nn.Module):
    # [CLASS DOC] HELPER: Holds a tensor and two submodules; the output is the sum of both
    # submodule results and the tensor.
    def __init__(self, tensor, sub_mod_0, sub_mod_1):
        # [FUNCTION DOC] HELPER: Stores the tensor attribute and registers the two given modules
        # as sub_mod_0 and sub_mod_1.
        super().__init__()
        self.tensor = tensor
        self.sub_mod_0 = sub_mod_0
        self.sub_mod_1 = sub_mod_1

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum of both submodule outputs on the input and the
        # stored tensor.
        return self.sub_mod_0(x) + self.sub_mod_1(x) + self.tensor


class ModWithMultipleSubmods(torch.nn.Module):
    # [CLASS DOC] HELPER: Holds two submodules and adds their outputs.
    def __init__(self, mod1, mod2):
        # [FUNCTION DOC] HELPER: Registers the two given modules as mod1 and mod2.
        super().__init__()
        self.mod1 = mod1
        self.mod2 = mod2

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum of both submodule outputs on the input.
        return self.mod1(x) + self.mod2(x)


class SimpleTest(torch.nn.Module):
    # [CLASS DOC] HELPER: Parameterless module that applies the wrapped leaf function and then
    # a relu.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls a_non_torch_leaf on the input with itself, adds 3.0 and
        # applies relu, returning the result.
        x = a_non_torch_leaf(x, x)
        return torch.relu(x + 3.0)


def a_non_torch_leaf(a, b):
    # [FUNCTION DOC] HELPER: Returns the sum of the two arguments. Registered as a wrapped
    # function at module top so that FX tracing records it as a single call node.
    return a + b
