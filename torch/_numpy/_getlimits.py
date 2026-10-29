# mypy: ignore-errors

# [MODULE DOC] torch._numpy equivalents of numpy.finfo and numpy.iinfo.
# Accepts anything convertible to a torch._numpy dtype and returns the corresponding torch.finfo / torch.iinfo.
import torch

from . import _dtypes


def finfo(dtyp):
    # [FUNCTION DOC] Return torch.finfo for the dtype-like argument (converted via _dtypes.dtype). Errors from dtype conversion propagate; torch raises for non-float dtypes.
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.finfo(torch_dtype)


def iinfo(dtyp):
    # [FUNCTION DOC] Return torch.iinfo for the dtype-like argument (converted via _dtypes.dtype). Errors from dtype conversion propagate; torch raises for non-integer dtypes.
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.iinfo(torch_dtype)
