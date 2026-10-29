"""torch.ops.aten operators under the `core` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001
# [MODULE DOC]
# torchlib ONNX implementations of core aten ops (currently abs, abs for complex, add, add
# for complex). Each is registered via onnx_impl as trace-only, so importing this module
# registers them in the torchlib registry. The aten namespace alias is defined here.

from __future__ import annotations

import operator

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal, TRealOrUInt8
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


@onnx_impl((aten.abs.default, operator.abs), trace_only=True)
def aten_abs(self: TRealOrUInt8) -> TRealOrUInt8:
    # [FUNCTION DOC]
    # Elementwise absolute value via the ONNX Abs node; returns the output value.
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(aten.abs.default, complex=True, trace_only=True)
def aten_abs_complex(self: TRealOrUInt8) -> TRealOrUInt8:
    # [FUNCTION DOC]
    # Magnitude of a complex input stored as real pairs in a trailing dimension of size 2:
    # L2 norm over the last axis without keeping it. Returns the reduced value.
    """abs(Tensor self) -> Tensor"""

    return op.ReduceL2(self, [-1], keepdims=False)


@onnx_impl((aten.add.Tensor, aten.add.Scalar, operator.add), trace_only=True)
def aten_add(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    # [FUNCTION DOC]
    # Elementwise self plus alpha times other. When alpha is not 1, it is cast to other's
    # type and multiplied into other first. Returns the Add output.
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""
    if alpha != 1.0:
        alpha = op.CastLike(alpha, other)
        other = op.Mul(other, alpha)
    return op.Add(self, other)


@onnx_impl((aten.add.Tensor, aten.add.Scalar), trace_only=True, complex=True)
def aten_add_complex(self: TReal, other: TReal, alpha: float = 1.0) -> TReal:
    # [FUNCTION DOC]
    # Complex-registered add; delegates to the real add implementation with the same
    # arguments (real/imag pairs add elementwise in the same way).
    """add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"""

    return aten_add(self, other, alpha=alpha)
