# [PACKAGE DOC]
# Backward-compatibility package torch.nn.intrinsic: re-exports the fused (intrinsic) module classes from torch.ao.nn.intrinsic
# (ConvBn/ConvBnReLU/ConvReLU 1d-3d, LinearReLU, LinearBn1d, BNReLU2d/3d). Also imports the modules, qat and quantized subpackages so
# they are reachable as attributes. The private _FusedModule base is imported but not listed in __all__.
from torch.ao.nn.intrinsic import (
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
from torch.ao.nn.intrinsic.modules.fused import _FusedModule

# Include the subpackages in case user imports from it directly
from torch.nn.intrinsic import modules, qat, quantized


__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
]
