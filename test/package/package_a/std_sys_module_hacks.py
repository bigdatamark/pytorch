# [MODULE DOC] Fixture whose imports include the typing.io and typing.re pseudo-submodules plus
# os.path, to exercise packaging of standard-library submodule imports. The sibling file for
# Python 3.13 omits the typing pseudo-submodules.
import os
import os.path
import typing
import typing.io
import typing.re  # noqa: F401

import torch


class Module(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward takes no tensor and returns an absolute path
    # computed from a constant relative path using os.path.
    def forward(self):
        # [FUNCTION DOC] HELPER: Returns the absolute version of the fixed relative path string
        # for the current working directory.
        return os.path.abspath("test")
