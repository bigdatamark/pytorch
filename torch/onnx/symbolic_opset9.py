# [MODULE DOC] Backward-compatibility shim for the legacy TorchScript exporter's ONNX opset 9
# symbolic functions. Re-exports everything from the internal torchscript_exporter module of the
# same name, so old import paths keep working. Also re-imports a few private helpers (padding,
# reshape, slice, var_mean) for old callers.
# The public __all__ is empty. No other logic or side effects besides the imports.
"""Backward compatibility module for torch.onnx.symbolic_opset9."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import *  # noqa: F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset9 import (  # noqa: F401
    _prepare_onnx_paddings,
    _reshape_from_tensor,
    _slice,
    _var_mean,
)
