# [MODULE DOC]
# Test helper for the ONNX exporter. Exports one public function, assert_onnx_program, that
# compares the outputs of an ONNXProgram with those of its source ExportedProgram.
# No import-time side effects beyond imports.
"""Test utilities for ONNX export."""

from __future__ import annotations


__all__ = ["assert_onnx_program"]

from typing import Any, Literal, TYPE_CHECKING

import torch
from torch.utils import _pytree


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


def assert_onnx_program(
    program: _onnx_program.ONNXProgram,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    strategy: str | None = "TorchExportNonStrictStrategy",
    backend: Literal["onnxruntime", "reference"] = "onnxruntime",
) -> None:
    # [FUNCTION DOC]
    # Verifies numerical agreement between an ONNXProgram and its ExportedProgram.
    # Optionally checks the recorded capture strategy name first. Runs the torch module on
    # the given or default example inputs, flattens outputs, drops None outputs, wraps
    # non-tensor outputs as tensors and views complex outputs as real. Then evaluates the
    # ONNX program with the onnxruntime or reference backend and compares the output
    # tuples with assert_close (NaNs equal, device ignored, given rtol and atol).
    # Raises ValueError for a strategy mismatch, a missing ExportedProgram, missing example
    # inputs, or an unknown backend; assert_close raises on numerical mismatch.
    # Side effect: executes the model on both runtimes.
    """Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The ``ONNXProgram`` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        strategy: Assert the capture strategy used to export the program. Values can be
            class names like "TorchExportNonStrictStrategy".
            If None, the strategy is not asserted.
        backend: The backend to use for evaluating the ONNX program.
            Supported values are "onnxruntime" and "reference".
    """
    if strategy is not None:
        if program._capture_strategy != strategy:
            raise ValueError(
                f"Expected strategy '{strategy}' is used to capture the exported program, "
                f"but got '{program._capture_strategy}'."
            )
    exported_program = program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNXProgram does not contain an ExportedProgram. "
            "To verify the ONNX program, initialize ONNXProgram with an ExportedProgram, "
            "or assign the ExportedProgram to the ONNXProgram.exported_program attribute."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    # ONNX outputs are always real, so we need to convert torch complex outputs to real representations
    torch_outputs_adapted = []
    for output in torch_outputs:
        # ONNX graph does not support None outputs, so we skip them
        if output is None:
            continue
        if not isinstance(output, torch.Tensor):
            torch_outputs_adapted.append(torch.tensor(output))
        elif torch.is_complex(output):
            torch_outputs_adapted.append(torch.view_as_real(output))
        else:
            torch_outputs_adapted.append(output)

    # Obtain the ONNX outputs using the specified backend
    if backend == "onnxruntime":
        onnx_outputs = program(*args, **kwargs)
    elif backend == "reference":
        onnx_outputs = program.call_reference(*args, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends are 'onnxruntime' and 'reference'."
        )

    # TODO(justinchuby): Include output names in the error message
    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(torch_outputs_adapted),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        check_device=False,
    )
