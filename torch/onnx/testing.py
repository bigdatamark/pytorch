# [MODULE DOC] Public testing utilities for exported ONNX models. Re-exports assert_onnx_program
# (compares an ONNX program's outputs to eager PyTorch) from the internal exporter testing module.
# Import-time side effect: rewrites its __module__ attribute to torch.onnx.testing.
"""Utilities to aid in testing exported ONNX models."""

__all__ = ["assert_onnx_program"]

from torch.onnx._internal.exporter._testing import assert_onnx_program


assert_onnx_program.__module__ = "torch.onnx.testing"
