# [PACKAGE DOC] Python package for the libtorch_agn_2_14 test extension. Importing it locates
# the single compiled _C shared library next to this file (raises an error if there is not
# exactly one), loads it with torch.ops.load_library so its stable ops get registered, and
# imports the ops and _interop submodules (_interop is a separate importable extension module
# with PyObject/Tensor conversion helpers). Exports: _interop, ops.
import sys
from pathlib import Path

import torch


# The op extension (_C) registers this version's and the inherited 2.9-2.13
# STABLE_TORCH_LIBRARY ops; load it via load_library like the other libtorch_agn
# extensions.
so_files = list(
    Path(__file__).parent.glob("_C*" + (".pyd" if sys.platform == "win32" else ".so"))
)
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.{{so,pyd}} file, found {len(so_files)}")
torch.ops.load_library(str(so_files[0]))

# _interop is a separate importable module exposing the PyObject<->Tensor helpers
# used by the interop tests.
from . import _interop, ops


__all__ = [
    "_interop",
    "ops",
]
