# [MODULE DOC] Exposes the functions of the C extension variable functions module as
# attributes of the torch._VF module without static typing support. Import-time side
# effect: replaces the module entry in the system module table with a VFModule instance.

"""
This makes the functions in torch._C._VariableFunctions available as
    torch._VF.<funcname>
without mypy being able to find them.

A subset of those functions are mapped to ATen functions in
torch/jit/_builtins.py

See https://github.com/pytorch/pytorch/issues/21478 for the reason for
introducing torch._VF

"""

import sys
import types

import torch


class VFModule(types.ModuleType):
    # [CLASS DOC] Module type forwarding attribute lookup to the C variable functions module
    # stored in the attribute vf.
    vf: types.ModuleType

    def __init__(self, name: str):
        # [FUNCTION DOC] Creates the module with the given name and stores the C variable
        # functions module as vf.
        super().__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, name: str) -> object:
        # [FUNCTION DOC] Called for attributes not found on the module; returns the attribute of
        # the C variable functions module, raising AttributeError if it does not exist.
        return getattr(self.vf, name)


sys.modules[__name__] = VFModule(__name__)
