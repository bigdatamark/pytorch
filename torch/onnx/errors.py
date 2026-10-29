# [MODULE DOC] Exception and warning types raised by the ONNX exporter.
# Defines OnnxExporterWarning (a UserWarning), OnnxExporterError (base RuntimeError),
# UnsupportedOperatorError and SymbolicValueError (both legacy, TorchScript exporter only).
# Only the warning and the two specific errors are listed in the public export list;
# the base error class is not exported. No import-time side effects.
"""ONNX exporter exceptions."""

from __future__ import annotations


__all__ = [
    "OnnxExporterWarning",
    "SymbolicValueError",
    "UnsupportedOperatorError",
]

import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import _C


class OnnxExporterWarning(UserWarning):
    # [CLASS DOC] Warning category used for all non-fatal diagnostics from the exporter.
    # Carries no extra state; exists so users can filter exporter warnings by category.
    """Warnings in the ONNX exporter."""


class OnnxExporterError(RuntimeError):
    # [CLASS DOC] Base class of all exporter errors (a RuntimeError subclass).
    # Carries no extra state; catch it to handle any exporter failure.
    """Errors raised by the ONNX exporter. This is the base class for all exporter errors."""


class UnsupportedOperatorError(OnnxExporterError):
    # [CLASS DOC] Error signalling that an operator cannot be exported to the requested opset.
    # Only the formatted message is stored. Legacy: used by the TorchScript exporter only.
    """Raised when an operator is unsupported by the exporter."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, name: str, version: int, supported_version: int | None) -> None:
        # [FUNCTION DOC] Build the error message and initialize the base error with it.
        # Args: name (operator name), version (requested opset), supported_version (opset
        # that added support, or None). Message variants: suggest the newer opset when known;
        # plain "not supported" for aten/prim/quantized namespaces; otherwise a hint about
        # registering custom operators with the right domain and version. Raises nothing.
        if supported_version is not None:
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported. Support for this operator was added in version "
                f"{supported_version}, try exporting with this version"
            )
        elif name.startswith(("aten::", "prim::", "quantized::")):
            msg = (
                f"Exporting the operator '{name}' to ONNX opset version {version} "
                "is not supported"
            )
        else:
            msg = (
                f"ONNX export failed on an operator with unrecognized namespace {name}. "
                "If you are trying to export a custom operator, make sure you registered it with "
                "the right domain and version."
            )

        super().__init__(msg)


class SymbolicValueError(OnnxExporterError):
    # [CLASS DOC] Error tied to a specific TorchScript graph value, with debugging context.
    # Only the formatted message is stored. Legacy: used by the TorchScript exporter only.
    """Errors around TorchScript values and nodes."""

    # NOTE: This is legacy and is only used by the torchscript exporter
    # Clean up when the torchscript exporter is removed
    def __init__(self, msg: str, value: _C.Value) -> None:
        # [FUNCTION DOC] Compose a detailed message about a graph value and initialize the base.
        # Args: msg (user message), value (the offending TorchScript value). The message
        # appends the value, its type, the kind of its node and the source location when
        # available, then lists the node's inputs and outputs with types (or "Empty").
        # If the node lacks those accessors (attribute error), a fallback note is added.
        # No side effects.
        message = (
            f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the "
            f"TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        )

        code_location = value.node().sourceRange()
        if code_location:
            message += f"\n    (node defined in {code_location})"

        try:
            # Add its input and output to the message.
            message += "\n\n"
            message += textwrap.indent(
                (
                    "Inputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {input_}  (type '{input_.type()}')"
                            for i, input_ in enumerate(value.node().inputs())
                        )
                        or "    Empty"
                    )
                    + "\n"
                    + "Outputs:\n"
                    + (
                        "\n".join(
                            f"    #{i}: {output}  (type '{output.type()}')"
                            for i, output in enumerate(value.node().outputs())
                        )
                        or "    Empty"
                    )
                ),
                "    ",
            )
        except AttributeError:
            message += (
                " Failed to obtain its input and output for debugging. "
                "Please refer to the TorchScript graph for debugging information."
            )

        super().__init__(message)
