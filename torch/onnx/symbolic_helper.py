# [MODULE DOC] Backward-compatibility shim for the legacy TorchScript exporter's symbolic helper
# utilities. Re-exports everything from the internal torchscript_exporter symbolic_helper module so
# old import paths keep working. The public __all__ is empty. No side effects besides the imports.
"""Backward compatibility module for torch.onnx.symbolic_helper."""

from __future__ import annotations


__all__: list[str] = []

from torch.onnx._internal.torchscript_exporter.symbolic_helper import *  # noqa: F403
