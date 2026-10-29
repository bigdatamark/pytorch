# [PACKAGE DOC]
# Compatibility package exporting quantized fused modules (LinearReLU, ConvReLU1d/2d/3d, BNReLU2d/3d) from the per-file shim modules.
from torch.nn.intrinsic.quantized.modules.bn_relu import BNReLU2d, BNReLU3d
from torch.nn.intrinsic.quantized.modules.conv_relu import (
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
)
from torch.nn.intrinsic.quantized.modules.linear_relu import LinearReLU


__all__ = [
    "LinearReLU",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "BNReLU2d",
    "BNReLU3d",
]
