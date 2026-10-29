# [PACKAGE DOC] python_agnostic: example extension built against the limited Python API.
# Importing it loads the compiled _C shared library next to this file (exactly one must
# exist), imports the ops submodule that defines the ultra_norm operator, and then runs a
# self-check on a CUDA or XPU device comparing ultra_norm against a plain norm. Raises
# AssertionError if the library is missing/ambiguous, no CUDA/XPU device exists, or the
# result differs. Exports ops (ultra_norm).
from pathlib import Path

import torch


so_files = list(Path(__file__).parent.glob("_C*.so"))
if len(so_files) != 1:
    raise AssertionError(f"Expected one _C*.so file, found {len(so_files)}")
torch.ops.load_library(so_files[0])

from . import ops


# ----------------------------------------------------------------------------- #
# We've reached the end of what is normal in __init__ files.
# The following is used to assert the ultra_norm op is properly loaded and
# calculates correct results upon import of this extension.

if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    raise AssertionError("Expected CUDA or XPU device backend, found none")

inputs = [
    torch.tensor([1.0, 2.0, 3.0], device=device),
    torch.tensor([-4.0, -5.0, -6.0], device=device),
]

if not torch.equal(
    ops.ultra_norm(inputs),
    torch.norm(torch.tensor([1.0, 2.0, 3.0, -4.0, -5.0, -6.0], device=device)),
):
    raise AssertionError("ultra_norm op did not produce expected result")
