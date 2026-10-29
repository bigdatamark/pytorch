# [MODULE DOC] Compatibility shim re-exporting the quantized ConvReLU1d/2d/3d from torch.ao.nn.intrinsic.quantized.
from torch.ao.nn.intrinsic.quantized import ConvReLU1d, ConvReLU2d, ConvReLU3d


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
]
