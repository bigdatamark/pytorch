# [MODULE DOC] Backward-compatibility shim for the legacy TorchScript exporter's ONNX opset 10
# symbolic functions. Re-exports everything from the internal torchscript_exporter module of the
# same name, so old import paths keep working. Also re-imports the private slice helper for old
# callers. The public __all__ is empty. No other logic or side effects besides the imports.
"""Backward compatibility module for torch.onnx.symbolic_opset10."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import *  # noqa: F403
from torch.onnx._internal.torchscript_exporter.symbolic_opset10 import (  # noqa: F401
    _slice,
)
