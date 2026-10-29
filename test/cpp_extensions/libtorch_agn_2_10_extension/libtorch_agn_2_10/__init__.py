# [PACKAGE DOC] Python package for the libtorch_agn_2_10 test extension. Importing it locates
# the single compiled _C shared library next to this file (raises an error if there is not
# exactly one), loads it with torch.ops.load_library so its stable ops get registered, and
# imports the ops submodule. Exports: ops.
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


__all__ = [
    "ops",
]
