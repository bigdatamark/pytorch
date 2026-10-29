# [PACKAGE DOC] Triton-based native op overrides package; exports ConstTensorWrapper, a read-only tensor argument wrapper for Triton kernels.
from .const_tensor_wrapper import ConstTensorWrapper


__all__ = ["ConstTensorWrapper"]
