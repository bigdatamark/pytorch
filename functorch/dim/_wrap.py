# [MODULE DOC]
# Wraps torch functions that take a dim argument (reductions etc.) so they accept first-class
# Dim objects: WrappedOperator holds per-op settings, patched_dim_method translates dims to
# integer indices on the underlying tensor and re-wraps results, _wrap builds the wrapper.
# No import-time side effects beyond imports.
"""
Python implementation of function wrapping functionality for functorch.dim.
"""

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._pytree import tree_map

from ._dim_entry import DimEntry
from ._enable_all_layers import EnableAllLayers
from ._tensor_info import TensorInfo


if TYPE_CHECKING:
    from collections.abc import Callable


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Identity hook; returns the given tensor unchanged.
    """Handle tensor conversion for torch function integration."""
    return tensor


class WrappedOperator:
    # [CLASS DOC] Describes one wrapped torch operation: the original callable, the wrapper
    # implementation to dispatch to, name, doc (extended with a note about the dim argument),
    # the dim argument name, and tuning flags (is_pointwise, dim_offset, keepdim_offset,
    # single_dim, reduce). The function method produces the user-facing wrapper function.
    """
    This class wraps PyTorch operations to support first-class dimensions.
    """

    def __init__(
        self, orig: Callable, wrapper_implementation: Callable, dim_name: str = "dim"
    ):
        # [FUNCTION DOC] Stores orig and wrapper_implementation, copies name and doc from
        # orig, appends a note about dim_name to the doc when both exist, and sets defaults
        # (dim offset 0, keepdim offset 1, not single-dim, reducing, not pointwise).
        self.orig = orig
        self.wrapper_implementation = wrapper_implementation
        self.name = getattr(orig, "__name__", "")
        self.doc = getattr(orig, "__doc__", None)
        self.dim_name = dim_name

        self.is_pointwise = False
        self.dim_offset = 0
        self.keepdim_offset = 1
        self.single_dim = False
        self.reduce = True

        # Update docstring if we have a dim_name
        if self.doc and self.dim_name:
            self.doc = f"{self.doc}\nArgument '{self.dim_name}' can be either an integer or a torchdim.Dim object.\n"

    def function(self) -> Callable:
        # [FUNCTION DOC] Returns a plain function that forwards all arguments to the wrapper
        # implementation together with this operator; it carries orig's name and the
        # extended doc.
        """Create a wrapped function that calls our wrapper implementation."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Calls the wrapper implementation with the operator
            # and all given arguments and returns its result.
            return self.wrapper_implementation(self, *args, **kwargs)

        # Copy metadata using functools.update_wrapper for just __name__ and __doc__
        functools.update_wrapper(
            wrapped_func, self.orig, assigned=("__name__",), updated=()
        )
        wrapped_func.__doc__ = self.doc

        return wrapped_func


def _wrap_dim(dim: Any, ndim: int, keepdim: bool = False) -> DimEntry:
    # [FUNCTION DOC] HELPER: Converts one dim spec to a DimEntry: a Dim gives a dim entry
    # (raises ValueError if keepdim is true), an int gives a negative positional entry
    # counted from the right, anything else gives an empty entry.
    """Convert single dimension specification to DimEntry object."""
    from . import Dim

    if isinstance(dim, Dim):
        if keepdim:
            raise ValueError("cannot preserve first-class dimensions with keepdim=True")
        return DimEntry(dim)
    elif isinstance(dim, int):
        i = dim
        while i >= 0:
            i -= ndim
        return DimEntry(i)
    else:
        return DimEntry()


def _wrap_dims(dim: Any, ndim: int, keepdim: bool = False) -> list[DimEntry]:
    # [FUNCTION DOC] HELPER: Converts a dim spec (single dim/int, or an iterable of them) into
    # a list of DimEntry. Raises ValueError for a Dim with keepdim, and a TypeError if the
    # non-single spec is not iterable.
    """Convert dimension specification to list of DimEntry objects."""
    de = _wrap_dim(dim, ndim, keepdim)
    result = []
    if not de.is_none():
        result.append(de)
    else:
        for d in dim:
            result.append(_wrap_dim(d, ndim, keepdim))
    return result


def patched_dim_method(wrapper: WrappedOperator, *args: Any, **kwargs: Any) -> Any:
    # [FUNCTION DOC] Runs the wrapped operation with first-class dim support. The first
    # argument is the tensor; the dim argument is read by keyword or position. Without a dim
    # argument the op runs on the batched tensor with all layers enabled. With one, dims are
    # resolved to integer indices into the levels, the op runs on the plain tensor, and
    # tensor results are re-wrapped with levels minus reduced dims (kept if keepdim or not a
    # reduction). Raises ValueError when no args, or a dim is not among the tensor levels, or
    # a Dim is combined with keepdim.
    # KNOWN ISSUE: keepdim_offset is compared to the arg count separately from dim_offset,
    # so positional keepdim detection assumes self precedes both.
    """
    This is the core method that handles dimension-aware operations.
    """
    if not args:
        raise ValueError("Expected at least one argument (self)")

    # Get dimension argument
    dim_arg = kwargs.get(wrapper.dim_name)
    if dim_arg is None and wrapper.dim_offset < len(args):
        # Try to get dim from positional args (accounting for self at index 0)
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(args):
            dim_arg = args[dim_idx]

    # If no dimension argument provided, fall back to standard functorch handling
    if dim_arg is None:
        info = TensorInfo.create(args[0], ensure_batched=True, ensure_present=False)
        if not info:
            return wrapper.orig(*args, **kwargs)

        with EnableAllLayers(info.levels) as guard:
            if info.batchedtensor is None:
                raise AssertionError("Expected batchedtensor to be non-None")
            guard.inplace_update_layers(info.batchedtensor, info.levels)
            new_args = list(args)
            new_args[0] = handle_from_tensor(info.batchedtensor)
            result = wrapper.orig(*new_args, **kwargs)
            return guard.from_batched(result, info.has_device)

    # Handle dimension-aware operation
    info = TensorInfo.create(args[0])
    if not info:
        return wrapper.orig(*args, **kwargs)

    # Check for keepdim parameter
    keepdim = False
    if wrapper.reduce:
        keepdim_arg = kwargs.get("keepdim")
        if keepdim_arg is None and wrapper.keepdim_offset < len(args):
            keepdim_idx = wrapper.keepdim_offset + 1
            if keepdim_idx < len(args):
                keepdim_arg = args[keepdim_idx]
        if keepdim_arg is not None:
            keepdim = bool(keepdim_arg)

    # Wrap dimensions
    ndim = info.ndim()
    dims = _wrap_dims(dim_arg, ndim, keepdim)

    # Convert dimensions to indices and validate
    dim_indices: list[int] = []
    seen = [False] * len(info.levels)

    for d in dims:
        midx = None
        for i, level in enumerate(info.levels):
            if level == d:
                midx = i
                break

        if midx is None:
            # Try to match by position/name more flexibly
            for i, level in enumerate(info.levels):
                if hasattr(level, "matches") and level.matches(d):
                    midx = i
                    break

            if midx is None:
                level_strs = [str(level) for level in info.levels]
                raise ValueError(
                    f"Tensor with dimensions {level_strs} does not contain {d}"
                )

        seen[midx] = True
        dim_indices.append(midx)

    # Determine new levels after reduction
    new_levels = []
    if wrapper.reduce and not keepdim:
        for i, level in enumerate(info.levels):
            if not seen[i]:
                new_levels.append(level)
    else:
        new_levels = info.levels[:]

    # Create dimension indices for the original function
    if len(dim_indices) == 1:
        py_indices: Any = dim_indices[0]
    else:
        py_indices = tuple(dim_indices)

    # Update arguments
    new_args = list(args)
    new_kwargs = kwargs.copy()
    if info.tensor is None:
        raise AssertionError("Expected tensor to be non-None")
    new_args[0] = handle_from_tensor(info.tensor)

    # Update dimension argument
    if wrapper.dim_name in new_kwargs:
        new_kwargs[wrapper.dim_name] = py_indices
    else:
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(new_args):
            new_args = list(new_args)
            new_args[dim_idx] = py_indices

    # Call original function
    result = wrapper.orig(*new_args, **new_kwargs)

    # Wrap results
    def wrap_result(obj: Any) -> Any:
        # [FUNCTION DOC] HELPER: Wraps a torch tensor into a first-class Tensor using the
        # computed new levels; other objects pass through unchanged.
        if isinstance(obj, torch.Tensor):
            from . import Tensor

            return Tensor.from_positional(obj, new_levels, info.has_device)
        return obj

    return tree_map(wrap_result, result)


def _wrap(
    orig: Callable,
    dim_offset: int | None = None,
    keepdim_offset: int | None = None,
    dim_name: str | None = None,
    single_dim: bool | None = None,
    reduce: bool | None = None,
) -> Callable:
    # [FUNCTION DOC] Builds a WrappedOperator around orig with patched_dim_method as the
    # implementation, applies any non-None option overrides (dim_name defaults to "dim"),
    # and returns the resulting wrapper function.
    """
    Wrap a PyTorch function to support first-class dimensions.

    Args:
        orig: Original function to wrap
        dim_offset: Offset for dimension argument (default: 0)
        keepdim_offset: Offset for keepdim argument (default: 1)
        dim_name: Name of dimension parameter (default: "dim")
        single_dim: Whether function takes single dimension (default: False)
        reduce: Whether function reduces dimensions (default: True)
    """
    dim_name = dim_name or "dim"

    wrapper = WrappedOperator(orig, patched_dim_method, dim_name)

    if dim_offset is not None:
        wrapper.dim_offset = dim_offset
    if keepdim_offset is not None:
        wrapper.keepdim_offset = keepdim_offset
    if single_dim is not None:
        wrapper.single_dim = single_dim
    if reduce is not None:
        wrapper.reduce = reduce

    return wrapper.function()


def call_torch_function(
    wrapper: WrappedOperator,
    func: Callable,
    types: tuple,
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    # [FUNCTION DOC] Forwards a torch function call to the first-class tensor class's
    # torch-function handler with func, types, args and kwargs (empty mapping if None).
    # The wrapper argument is unused.
    """
    Handle __torch_function__ calls for wrapped operators.
    """
    if kwargs is None:
        kwargs = {}

    # Import here to avoid circular imports
    from . import _Tensor

    # Use the torch function mechanism from _Tensor
    return _Tensor.__torch_function__(func, types, args, kwargs)
