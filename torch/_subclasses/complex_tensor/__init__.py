# [PACKAGE DOC] Public entry point for the ComplexTensor subclass package. Re-exports ComplexTensor, ComplexTensorMode,
# is_complex_tensor and WrapComplexMode, and rewrites __module__ of the first three so they appear to
# live in this package.
from ._core import ComplexTensor
from ._ops import ComplexTensorMode, is_complex_tensor, WrapComplexMode


__all__ = ["ComplexTensor", "ComplexTensorMode", "is_complex_tensor", "WrapComplexMode"]

ComplexTensor.__module__ = __name__
ComplexTensorMode.__module__ = __name__
is_complex_tensor.__module__ = __name__
