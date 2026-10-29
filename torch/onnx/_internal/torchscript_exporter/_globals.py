# [MODULE DOC] Holds the single process-wide mutable exporter state object. Defines the
# _InternalGlobals class and creates the GLOBALS instance at import time. Depends only on the
# onnx constants module to keep the dependency direction clean.
"""Globals used internally by the ONNX exporter.

Do not use this module outside of `torch.onnx` and its tests.

Be very judicious when adding any new global variables. Do not create new global
variables unless they are absolutely necessary.
"""

import torch._C._onnx as _C_onnx

# This module should only depend on _constants and nothing else in torch.onnx to keep
# dependency direction clean.
from torch.onnx import _constants


class _InternalGlobals:
    # [CLASS DOC] HELPER: Container for exporter-wide state: opset version (default from
    # constants), training mode, in-export flag, export_training flag, operator export type,
    # shape inference flag and autograd inlining flag. Properties validate types on assignment.
    """Globals used internally by ONNX exporter.

    NOTE: Be very judicious when adding any new variables. Do not create new
    global variables unless they are absolutely necessary.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes defaults: default opset, EVAL training mode, not in
        # export, export_training off, ONNX operator export type, shape inference on, inlining on.
        self._export_onnx_opset_version = _constants.ONNX_DEFAULT_OPSET
        self._training_mode: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL
        self._in_onnx_export: bool = False
        # Whether the user's model is training during export
        self.export_training: bool = False
        self.operator_export_type: _C_onnx.OperatorExportTypes = (
            _C_onnx.OperatorExportTypes.ONNX
        )
        self.onnx_shape_inference: bool = True
        self._autograd_inlining: bool = True

    @property
    def training_mode(self) -> _C_onnx.TrainingMode:
        # [FUNCTION DOC] HELPER: Getter returning the current exporter training mode.
        """The training mode for the exporter."""
        return self._training_mode

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode) -> None:
        # [FUNCTION DOC] HELPER: Setter storing the training mode. Raises a type error if the
        # value is not a TrainingMode enum member.
        if not isinstance(training_mode, _C_onnx.TrainingMode):
            raise TypeError(
                "training_mode must be of type 'torch.onnx.TrainingMode'. This is "
                "likely a bug in torch.onnx."
            )
        self._training_mode = training_mode

    @property
    def export_onnx_opset_version(self) -> int:
        # [FUNCTION DOC] HELPER: Getter returning the opset version used during export.
        """Opset version used during export."""
        return self._export_onnx_opset_version

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int) -> None:
        # [FUNCTION DOC] HELPER: Setter storing the opset version; no validation.
        self._export_onnx_opset_version = value

    @property
    def in_onnx_export(self) -> bool:
        # [FUNCTION DOC] HELPER: Getter: true while an ONNX export is in progress.
        """Whether it is in the middle of ONNX export."""
        return self._in_onnx_export

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool) -> None:
        # [FUNCTION DOC] HELPER: Setter storing the in-export flag. Raises a type error unless the
        # value is exactly a bool.
        if type(value) is not bool:
            raise TypeError("in_onnx_export must be a boolean")
        self._in_onnx_export = value

    @property
    def autograd_inlining(self) -> bool:
        # [FUNCTION DOC] HELPER: Getter: whether autograd functions must be inlined.
        """Whether Autograd must be inlined."""
        return self._autograd_inlining

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool) -> None:
        # [FUNCTION DOC] HELPER: Setter storing the inlining flag. Raises a type error unless the
        # value is exactly a bool.
        if type(value) is not bool:
            raise TypeError("autograd_inlining must be a boolean")
        self._autograd_inlining = value


GLOBALS = _InternalGlobals()
