# [PACKAGE DOC]
# Compatibility package for intrinsic QAT modules. Re-exports ConvBn/ConvBnReLU/ConvReLU 1d-3d, LinearBn1d, LinearReLU and the
# utilities freeze_bn_stats and update_bn_stats from the per-file shim modules (which forward to torch.ao.nn.intrinsic.qat).
from torch.nn.intrinsic.qat.modules.conv_fused import (
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
from torch.nn.intrinsic.qat.modules.linear_fused import LinearBn1d
from torch.nn.intrinsic.qat.modules.linear_relu import LinearReLU


__all__ = [
    "LinearReLU",
    "LinearBn1d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "update_bn_stats",
    "freeze_bn_stats",
]
