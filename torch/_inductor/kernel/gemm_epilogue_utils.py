# mypy: allow-untyped-defs
# [MODULE DOC] Small helpers for handling possibly symbolic shapes and integers in GEMM epilogue
# frontends. Provides shape canonicalization, guarded integer extraction, and "statically known"
# comparisons that avoid adding guards. No import-time side effects.
"""Symbolic shape helpers shared by GEMM epilogue frontends."""

from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    guard_int,
    has_guarding_hint,
    statically_known_true as fx_statically_known_true,
)


def normalize_shape(shape: Any) -> Any:
    # [FUNCTION DOC] Convert a list, tuple or torch.Size into a plain tuple; any other value is
    # returned unchanged. Pure, no errors raised.
    """Canonicalize sequence-like shapes to tuples."""
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


def guarded_int(value: Any) -> int | None:
    # [FUNCTION DOC] Extract a concrete int from an fx Node (using its "val" metadata), a SymInt,
    # or a plain int. A SymInt with no guarding hint yields None; a hinted SymInt is guarded
    # (side effect: adds a shape guard) and its int value returned. Other types yield None.
    """Return an integer after guarding backed symbolic values."""
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        if not has_guarding_hint(value):
            return None
        # TODO: Defer speculative guards if they become a meaningful source of
        # recompilation.
        return guard_int(value)
    return value if isinstance(value, int) else None


def statically_known(expr: Any) -> bool:
    # [FUNCTION DOC] Decide whether a predicate is provably true without adding guards. Python bools
    # are returned as-is; sympy expressions use the current graph's sizevars; anything else goes
    # through the fx symbolic-shapes helper.
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    # [FUNCTION DOC] True only if the two values are statically known to be equal (no guards added).
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    # [FUNCTION DOC] True if both shapes have the same length and every pair of dimensions is
    # statically known equal; no guards are added. Returns False when equality is unproven.
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )
