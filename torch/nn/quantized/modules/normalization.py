# [MODULE DOC] Deprecated compatibility shim; re-exports quantized LayerNorm, GroupNorm and
# InstanceNorm1d/2d/3d from torch.ao.nn.quantized.modules.normalization.
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.modules.normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)


__all__ = [
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]
