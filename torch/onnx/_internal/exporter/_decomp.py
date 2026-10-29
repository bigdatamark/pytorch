# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that relate the ONNX registry to PyTorch decompositions: listing the
# torch ops the registry implements, and building a decomposition table of ops that lack an
# ONNX implementation. No import-time side effects beyond imports.
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch._ops


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.onnx._internal.exporter import _registration


def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[_registration.TorchOp]:
    # [FUNCTION DOC] Returns a list with one torch op target per entry of the registry's
    # function mapping, taking the fx target of the first decomposition metadata of each entry.
    # Args: registry. Raises AssertionError if any entry has an empty metadata list.
    # No side effects.
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[_registration.TorchOp] = []
    for onnx_decomp_meta in registry.functions.values():
        if len(onnx_decomp_meta) == 0:
            raise AssertionError("onnx_decomp_meta must not be empty")
        # Different OnnxDecompMeta for the same TorchOp should
        # have the same fx_target.
        fx_target = onnx_decomp_meta[0].fx_target
        registered_ops.append(fx_target)
    return registered_ops


def create_onnx_friendly_decomposition_table(
    onnx_registered_ops: set[_registration.TorchOp],
) -> dict[_registration.TorchOp, Callable]:
    # [FUNCTION DOC] Builds a mapping from op overload to decomposition function, drawn first
    # from the torch.export default decompositions and then from the general decomposition
    # table (first occurrence wins). Skips ops already implemented in ONNX (the given set) and
    # entries without a schema (higher-order ops). Returns the new mapping; no side effects.
    # KNOWN ISSUE: the docstring says ops are excluded when they have an "ONNX symbolic
    # function", but the check is membership in the registered-ops set argument.
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        onnx_registered_ops: All ops that have an ONNX decomposition implemented.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: dict[_registration.TorchOp, Callable] = {}

    for op_overload, decomp_fn in itertools.chain(
        torch.export.default_decompositions().items(),  # type: ignore[attr-defined]
        torch._decomp.decomposition_table.items(),  # type: ignore[attr-defined]
    ):
        # Skip decomposition for op_overload as long as that op_overload has a corresponding ONNX
        # symbolic function.
        # NOTE: Do not skip torch._refs decomps. They are fine because otherwise the model is
        # not exportable anyways.
        if op_overload in onnx_registered_ops:
            continue
        # If it is HOP, we filter those out as well.
        if not hasattr(op_overload, "_schema"):
            continue
        # NOTE: torch._decomp.decomposition_table covers more ops
        # than torch.export.default_decompositions, but the latter is
        # more critical to torch.onnx.export.
        if op_overload in decomposition_table:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table
