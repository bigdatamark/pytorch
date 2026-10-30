# [MODULE DOC] Test helper defining a TorchScript class named FooSameName which wraps a second
# FooSameName imported from the sibling module bar. Used to test importing script classes whose
# names collide across modules. Compiling the class happens at import time.
import torch

from . import bar


# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script
class FooSameName:
    # [CLASS DOC] HELPER: Script class with attributes x and nested; nested is a FooSameName
    # instance from the bar module built from the same x. Compiled at import time.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Constructor; stores x and builds the nested object from the bar
        # module's class with the same argument.
        self.x = x
        self.nested = bar.FooSameName(x)
