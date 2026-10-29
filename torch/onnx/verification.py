# [MODULE DOC] Public verification tools for ONNX models. Re-exports VerificationInfo (a record of
# comparison results) and verify_onnx_program (runs an ONNX program and compares it against eager
# PyTorch) from the internal exporter module. Import-time side effect: rewrites the __module__
# attribute of both objects to torch.onnx.verification.
"""A set of tools to verify the correctness of ONNX models."""

__all__ = ["VerificationInfo", "verify_onnx_program"]

from torch.onnx._internal.exporter._verification import (
    VerificationInfo,
    verify_onnx_program,
)


VerificationInfo.__module__ = "torch.onnx.verification"
verify_onnx_program.__module__ = "torch.onnx.verification"
