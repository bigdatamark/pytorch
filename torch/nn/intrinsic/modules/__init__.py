# [PACKAGE DOC]
# Compatibility package re-exporting the fused module classes (and the private _FusedModule base) from torch.nn.intrinsic.modules.fused.
# Exports: ConvBn/ConvBnReLU/ConvReLU 1d-3d, BNReLU2d/3d, LinearBn1d, LinearReLU.
from torch.nn.intrinsic.modules.fused import (
    _FusedModule,
    BNReLU2d,
    BNReLU3d,
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    LinearBn1d,
    LinearReLU,
)


__all__ = [
    "BNReLU2d",
    "BNReLU3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearBn1d",
    "LinearReLU",
]
