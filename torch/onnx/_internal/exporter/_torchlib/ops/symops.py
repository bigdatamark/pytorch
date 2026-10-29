"""Implementation for torch.sym* ops."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC003
# [MODULE DOC]
# torchlib ONNX implementations of torch symbolic-number ops: sym_float, sym_max, sym_min,
# sym_not, sym_sum and sym_ite. Each is registered trace-only through onnx_impl, so
# importing this module registers them in the torchlib registry.

from __future__ import annotations

from collections.abc import Sequence

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import (
    BOOL,
    FLOAT,
    IntType,
    TensorType,
    TTensor,
)
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


@onnx_impl(torch.sym_float, trace_only=True)
def sym_float(self: TensorType) -> FLOAT:
    # [FUNCTION DOC]
    # Casts the input to ONNX float32 with a Cast node and returns the result.
    """sym_float(SymInt self) -> SymFloat"""
    return op.Cast(self, to=FLOAT.dtype)


@onnx_impl(torch.sym_max, trace_only=True)
def sym_max(x: IntType, y: IntType) -> IntType:
    # [FUNCTION DOC]
    # Elementwise maximum of two integer values via the Max node; returns the result.
    """sym_max(SymInt x, SymInt y) -> SymInt"""
    return op.Max(x, y)


@onnx_impl(torch.sym_min, trace_only=True)
def sym_min(x: IntType, y: IntType) -> IntType:
    # [FUNCTION DOC]
    # Elementwise minimum of two integer values via the Min node; returns the result.
    """sym_min(SymInt x, SymInt y) -> SymInt"""
    return op.Min(x, y)


@onnx_impl(torch.sym_not, trace_only=True)
def sym_not(self: BOOL) -> BOOL:
    # [FUNCTION DOC]
    # Logical negation of a boolean value via the Not node; returns the result.
    """sym_not(SymBool self) -> SymBool"""
    return op.Not(self)


@onnx_impl(torch.sym_sum, trace_only=True)
def sym_sum(args: Sequence[IntType]) -> IntType:
    # [FUNCTION DOC]
    # Sums a sequence of integer values at trace time. Empty input yields a constant zero,
    # a single element is returned unchanged, otherwise a chain of Add nodes is built
    # left to right and the final node output is returned.
    """sym_sum(SymInt[] args) -> SymInt"""
    if len(args) == 0:
        return op.Constant(value_int=0)
    if len(args) == 1:
        return args[0]
    result = op.Add(args[0], args[1])
    for i in range(2, len(args)):
        result = op.Add(result, args[i])
    return result


@onnx_impl(torch.sym_ite, trace_only=True)
def sym_ite(b: BOOL, t: TTensor, f: TTensor) -> TTensor:
    # [FUNCTION DOC]
    # Conditional select: emits a Where node choosing t where b is true, else f.
    """sym_ite(SymBool b, Tensor t, Tensor f) -> Tensor"""
    return op.Where(b, t, f)
