# [MODULE DOC] Test helper defining a TorchScript class named FooSameName (stores one attribute).
# The same class name also exists in foo.py; the tests use the pair to check that script classes
# with identical names in different modules are imported without clashing. Defining the class runs
# the script compiler at import time.
import torch


# This file contains definitions of script classes.
# They are used by test_jit.py to test ScriptClass imports


@torch.jit.script
class FooSameName:
    # [CLASS DOC] HELPER: Script class holding a single attribute y; compiled when this module
    # loads. Counterpart of the same-named class in foo.py.
    def __init__(self, y):
        # [FUNCTION DOC] HELPER: Constructor of the script class; stores the argument as attribute
        # y.
        self.y = y
