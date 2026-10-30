# [MODULE DOC] Variant of the std_sys_module_hacks fixture for Python 3.13, where the typing.io
# and typing.re pseudo-submodules no longer exist; it imports only os, os.path and typing.
import os
import os.path
import typing  # noqa: F401

import torch


class Module(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward takes no tensor and returns an absolute path
    # computed from a constant relative path using os.path.
    def forward(self):
        # [FUNCTION DOC] HELPER: Returns the absolute version of the fixed relative path string
        # for the current working directory.
        return os.path.abspath("test")
