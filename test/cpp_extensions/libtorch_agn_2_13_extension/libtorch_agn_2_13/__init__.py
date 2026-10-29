# [PACKAGE DOC] Python package for the libtorch_agn_2_13 test extension. Importing it locates
# the single compiled _C shared library next to this file (raises an error if there is not
# exactly one), loads it with torch.ops.load_library so its stable ops get registered, and
# imports the ops submodule, and registers a fake (meta) implementation for the
# identity_with_fake_module op at import time. Exports: ops.
import sys
from pathlib import Path

import torch


so_files = list(
    Path(__file__).parent.glob("_C*" + (".pyd" if sys.platform == "win32" else ".so"))
)
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.{{so,pyd}} file, found {len(so_files)}")

torch.ops.load_library(str(so_files[0]))

from . import ops


# Fake impl for the op the C++ side declared with
# m.set_python_module("libtorch_agn_2_13"). Registering here means the fake
# impl is available as soon as the extension is imported.
@torch.library.register_fake("libtorch_agn_2_13::identity_with_fake_module")
def _(t):
    # [FUNCTION DOC] Fake implementation of identity_with_fake_module: returns an empty tensor with
    # the same shape, dtype and device as t, without touching data. Registered at import time.
    return torch.empty_like(t)


__all__ = [
    "ops",
]
