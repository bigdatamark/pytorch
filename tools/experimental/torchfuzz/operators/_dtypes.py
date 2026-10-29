# pyre-strict
# [MODULE DOC] HELPER: Shared dtype constants and small codegen/shape utilities used by the
# torchfuzz operator modules: the tuple of floating dtypes, a float-dtype test, contiguous
# stride computation, scalar formatting for generated code and random broadcast-shape
# generation. No import-time side effects beyond imports.
"""Shared dtype constants and utilities for torchfuzz operator modules."""

from __future__ import annotations

import math
import random

import torch


FLOAT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
)


def is_float_dtype(dt: torch.dtype) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the dtype is one of the four floating dtypes in
    # FLOAT_DTYPES (half, float, double, bfloat16); false otherwise (including complex).
    return dt in FLOAT_DTYPES


def contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Computes row-major contiguous strides for a size tuple by a
    # running product from the last dim backwards. Empty size gives an empty tuple. Pure.
    """Row-major contiguous strides for ``size``."""
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def scalar_repr(value: object) -> str:
    # [FUNCTION DOC] HELPER: Returns source text for a scalar to embed in generated code.
    # Infinite floats become a float-constructor call with "inf" or "-inf", NaN becomes a
    # float-constructor call with "nan", everything else uses its normal repr. Pure.
    """Format a scalar for embedding in codegen, handling inf/nan."""
    if isinstance(value, float):
        if math.isinf(value):
            return f"float('{'-' if value < 0 else ''}inf')"
        if math.isnan(value):
            return "float('nan')"
    return repr(value)


def random_broadcast_shape(output_size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Picks a random input shape that broadcasts to output_size: each
    # dim becomes 1 with probability 0.3, then leading size-1 dims are dropped with
    # probability 0.5 each (keeping at least one dim). Returns the original size when nothing
    # shrank, and an empty tuple for empty input. Uses the global random module (side effect).
    """Return a shape that broadcasts to ``output_size``.

    Each dimension has a 30% chance of shrinking to 1. Leading size-1
    dims may then be dropped (50% chance each) to produce a lower-rank
    input. Returns ``output_size`` unchanged if no dims were shrunk.
    """
    if not output_size:
        return ()
    result = list(output_size)
    changed = False
    for i in range(len(result)):
        if random.random() < 0.3:
            result[i] = 1
            changed = True
    if not changed:
        return output_size
    while len(result) > 1 and result[0] == 1 and random.random() < 0.5:
        result.pop(0)
    return tuple(result)
