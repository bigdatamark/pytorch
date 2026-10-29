# [PACKAGE DOC]
# Compatibility package for quantized fused modules. Imports the dynamic and modules subpackages and star-imports modules, exporting
# BNReLU2d/3d, ConvReLU1d/2d/3d and LinearReLU (re-exported from torch.ao.nn.intrinsic.quantized).
# to ensure customers can use the module below
# without importing it directly
from torch.nn.intrinsic.quantized import dynamic, modules
from torch.nn.intrinsic.quantized.modules import *  # noqa: F403


__all__ = [
    "BNReLU2d",
    "BNReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
]
