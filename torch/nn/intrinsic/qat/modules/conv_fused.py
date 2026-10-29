# [MODULE DOC]
# Compatibility shim re-exporting the fused conv QAT modules (ConvBn, ConvBnReLU, ConvReLU for 1d/2d/3d) and the helper functions
# freeze_bn_stats and update_bn_stats from torch.ao.nn.intrinsic.qat. No logic, no side effects.
r"""Intrinsic QAT Modules.

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

from torch.ao.nn.intrinsic.qat import (
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    freeze_bn_stats,
    update_bn_stats,
)


__all__ = [
    # Modules
    "ConvBn1d",
    "ConvBnReLU1d",
    "ConvReLU1d",
    "ConvBn2d",
    "ConvBnReLU2d",
    "ConvReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "ConvReLU3d",
    # Utilities
    "freeze_bn_stats",
    "update_bn_stats",
]
