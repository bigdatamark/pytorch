# [MODULE DOC] Test helper defining a TorchScript class with a unique name (FooUniqueName), placed
# in a deeply nested package to test importing script classes by nested module path. Compiling the
# class happens at import time.
import torch


# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script
class FooUniqueName:
    # [CLASS DOC] HELPER: Script class holding a single attribute y; compiled when this module
    # loads.
    def __init__(self, y):
        # [FUNCTION DOC] HELPER: Constructor of the script class; stores the argument as attribute
        # y.
        self.y = y
