# mypy: allow-untyped-defs
# [MODULE DOC] Small helpers mapping Python scalar types and torch symbolic types to torch dtypes.
# Contains a symbolic-type predicate, a scalar-type-to-dtype lookup, and three module-level
# mapping tables (python types, symbolic types, and their merge). No import-time side effects
# beyond building the tables.
"""Utilities for converting and operating on ONNX and torch types."""

from __future__ import annotations

from typing import Any
from typing_extensions import TypeIs

import torch


def is_torch_symbolic_type(
    value: Any,
) -> TypeIs[torch.SymBool | torch.SymInt | torch.SymFloat]:
    # [FUNCTION DOC] Type guard: returns true if the value is a symbolic bool, int or float.
    # Args: value, any object. Returns a bool; no side effects, never raises.
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))


def from_scalar_type_to_torch_dtype(scalar_type: type) -> torch.dtype | None:
    # [FUNCTION DOC] Looks up the torch dtype for a python scalar type or symbolic type class.
    # Args: scalar_type, a type object. Returns the dtype (bool->bool, int->int64,
    # float->float32, complex->complex64, SymInt/SymFloat/SymBool likewise) or None if unknown.
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


_PYTHON_TYPE_TO_TORCH_DTYPE = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex64,
}

_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

_SCALAR_TYPE_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,  # type: ignore[dict-item]
}
