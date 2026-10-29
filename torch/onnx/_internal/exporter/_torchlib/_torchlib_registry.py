# [MODULE DOC]
# Registry of ONNX implementations of torch ops. Provides the onnx_impl decorator, which
# optionally compiles or wraps a function with onnxscript and appends an OnnxDecompMeta
# record to a module-level list, and get_torchlib_ops, which imports the ops subpackage
# to trigger registration and returns all records. Import-time side effects: creates the
# empty registry list and a logger.
# KNOWN ISSUE: the logger is created with the literal string "__name__" rather than the
# module name, so log records are attributed to a logger named "__name__".
"""Registry for aten functions."""

from __future__ import annotations


__all__ = ["onnx_impl", "get_torchlib_ops"]

import logging
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import onnxscript

import torch
from torch.onnx._internal.exporter import _constants, _registration


# Use ParamSpec for better type preservation instead of bound Callable TypeVar
_P = ParamSpec("_P")
_R = TypeVar("_R")

logger = logging.getLogger("__name__")


_registry: list[_registration.OnnxDecompMeta] = []


def onnx_impl(
    target: _registration.TorchOp | tuple[_registration.TorchOp, ...],
    *,
    trace_only: bool = False,
    complex: bool = False,
    opset_introduced: int = 18,
    no_compile: bool = False,
    private: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # [FUNCTION DOC]
    # Decorator factory. target is one torch op overload or a tuple of them; a packet
    # raises TypeError immediately. The returned decorator leaves the function as is when
    # no_compile is set, otherwise compiles it with onnxscript (script) or wraps it as a
    # traced function when trace_only is set, both against the torchlib domain opset v1.
    # Unless private is set, appends one OnnxDecompMeta per target to the global registry
    # (carrying is_complex, opset_introduced, and skip_signature_inference from no_compile).
    # Returns the processed function, so decorated names stay usable.
    """Register an ONNX implementation of a torch op."""

    if isinstance(target, torch._ops.OpOverloadPacket):
        raise TypeError(
            f"Target '{target}' should be provided as an OpOverload instead of an "
            "OpOverloadPacket. You can get the default overload with "
            "<op>.default"
        )

    def wrapper(
        func: Callable[_P, _R],
    ) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: The decorator returned by onnx_impl; compiles or wraps func
        # as configured, registers it for each target unless private, and returns the
        # processed function. Mutates the module-level registry list.
        processed_func: Any
        if no_compile:
            processed_func = func
        else:
            torchlib_opset = onnxscript.values.Opset(
                domain=_constants.TORCHLIB_DOMAIN, version=1
            )

            if not trace_only:
                # Compile the function
                processed_func = onnxscript.script(opset=torchlib_opset)(func)
            else:
                processed_func = onnxscript.TracedOnnxFunction(torchlib_opset, func)

        if not private:
            # TODO(justinchuby): Simplify the logic and remove the private attribute
            # Skip registration if private
            if not isinstance(target, Sequence):
                targets = (target,)
            else:
                targets = target  # type: ignore[assignment]

            for t in targets:
                _registry.append(
                    _registration.OnnxDecompMeta(
                        onnx_function=processed_func,
                        fx_target=t,
                        signature=None,
                        is_complex=complex,
                        opset_introduced=opset_introduced,
                        skip_signature_inference=no_compile,
                    )
                )
        return processed_func  # type: ignore[return-value]

    return wrapper


def get_torchlib_ops() -> tuple[_registration.OnnxDecompMeta, ...]:
    # [FUNCTION DOC]
    # Imports the ops subpackage so all op modules run their registration decorators, then
    # returns a snapshot tuple of every registered OnnxDecompMeta. Raises AssertionError
    # if the registry is still empty. Side effect: first call imports the op modules.
    # Trigger op registration
    from torch.onnx._internal.exporter._torchlib import ops

    del ops
    if len(_registry) == 0:
        raise AssertionError("_registry must not be empty")
    return tuple(_registry)
