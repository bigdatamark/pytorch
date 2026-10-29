# flake8: noqa: F401
# [MODULE DOC] Deprecated compatibility shim; re-exports quantized Conv and ConvTranspose
# 1d/2d/3d (plus the private helper _reverse_repeat_padding, not in __all__) from
# torch.ao.nn.quantized.modules.conv.
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.modules.conv import (
    _reverse_repeat_padding,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]
