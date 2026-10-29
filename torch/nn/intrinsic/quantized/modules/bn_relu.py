# [MODULE DOC] Compatibility shim re-exporting the quantized BNReLU2d and BNReLU3d from torch.ao.nn.intrinsic.quantized.
from torch.ao.nn.intrinsic.quantized import BNReLU2d, BNReLU3d


__all__ = [
    "BNReLU2d",
    "BNReLU3d",
]
