# [PACKAGE DOC] Quantized sparse modules. Re-exports the dynamic subpackage plus
# Linear and LinearPackedParams (sparse quantized linear layer and its packed-
# parameter holder).
from torch.ao.nn.sparse.quantized import dynamic

from .linear import Linear, LinearPackedParams


__all__ = [
    "dynamic",
    "Linear",
    "LinearPackedParams",
]
