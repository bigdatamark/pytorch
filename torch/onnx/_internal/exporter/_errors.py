# [MODULE DOC] Defines the exporter-specific exception hierarchy. All classes derive from
# torch.onnx.errors.OnnxExporterError; ConversionError is the parent of the dispatch and graph
# construction errors. No import-time side effects beyond importing torch.onnx.errors.
"""Error classes for the ONNX exporter."""

from __future__ import annotations

import torch.onnx.errors


class TorchExportError(torch.onnx.errors.OnnxExporterError):
    # [CLASS DOC] Raised when capturing a graph with torch.export fails. Marker type only.
    """Error during graph capturing using torch.export."""


class ConversionError(torch.onnx.errors.OnnxExporterError):
    # [CLASS DOC] Raised when converting an ExportedProgram to ONNX fails. Base class of the
    # more specific conversion errors below. Marker type, no extra state.
    """Error during ExportedProgram to ONNX conversion."""


class DispatchError(ConversionError):
    # [CLASS DOC] Raised when no ONNX function can be selected for a node (dispatching failure).
    """Error during ONNX Function dispatching."""


class GraphConstructionError(ConversionError):
    # [CLASS DOC] Raised when building the ONNX graph from the selected functions fails.
    """Error during ONNX graph construction."""
