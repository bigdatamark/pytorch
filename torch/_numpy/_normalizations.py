# [MODULE DOC] Argument normalization layer for torch._numpy.
# Defines annotation marker aliases (ArrayLike, DTypeLike, AxisLike, OutArray, KeepDims, ...), per-annotation
# normalizer functions registered in the normalizers mapping keyed by the annotation string, and the
# normalizer decorator that wraps implementer functions: it converts inputs, handles keepdims and out=,
# and wraps tensor results back into ndarray. No import-time side effects beyond defining these names.
""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on."""

from __future__ import annotations

import functools
import inspect
import operator
import typing
from typing import TYPE_CHECKING

import torch

from . import _dtypes, _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._ndarray import ndarray


# These names are used both as static type annotations and as runtime markers:
# the `normalizer` decorator dispatches on the *string* form of each parameter's
# annotation (e.g. "ArrayLike", "OutArray | None"). Because `from __future__
# import annotations` is active everywhere these are used, the annotations are
# never evaluated, so the static type below is what each parameter resolves to
# *after* normalization runs (e.g. an ArrayLike argument is a torch.Tensor by the
# time the implementer body sees it).
ArrayLike: typing.TypeAlias = torch.Tensor
Scalar = int | float | complex | bool
ArrayLikeOrScalar = ArrayLike | Scalar

# DTypeLike normalizes to a torch.dtype (or None when no dtype is given).
DTypeLike: typing.TypeAlias = "torch.dtype | None"
# AxisLike is an axis specification: a single axis, a tuple of axes, or None.
AxisLike: typing.TypeAlias = "int | tuple[int, ...] | None"
# NDArray normalizes to the wrapped tensor.
NDArray: typing.TypeAlias = torch.Tensor
# CastingModes is one of the numpy casting strings.
CastingModes: typing.TypeAlias = str
# KeepDims is the keepdims flag; handled entirely by the normalizer.
KeepDims: typing.TypeAlias = bool

# OutArray is to annotate the out= array argument.
#
# This one is special is several respects:
# First, It needs to be an NDArray, and we need to preserve the `result is out`
# semantics. Therefore, we cannot just extract the Tensor from the out array.
# So we never pass the out array to implementer functions and handle it in the
# `normalizer` below.
# Second, the out= argument can be either keyword or positional argument, and
# as a positional arg, it can be anywhere in the signature.
# To handle all this, we define a special `OutArray` annotation and dispatch on it.
#
OutArray: typing.TypeAlias = "ndarray"

# NotImplementedType marks unsupported parameters; the normalizer raises unless
# the argument equals the (always None) default, so the value is never used.
NotImplementedType: typing.TypeAlias = None


def normalize_array_like(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor:
    # [FUNCTION DOC] Convert any array-like (ndarray, list, scalar, tensor) to a torch.Tensor via asarray; parm is unused. Errors from asarray propagate.
    from ._ndarray import asarray

    return asarray(x).tensor


def normalize_array_like_or_scalar(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | Scalar:
    # [FUNCTION DOC] Pass Python scalars and symbolic numbers through unchanged; otherwise convert to a tensor as normalize_array_like does.
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x  # pyrefly: ignore[bad-return]
    return normalize_array_like(x, parm)  # codespell:ignore


def normalize_optional_array_like_or_scalar(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | Scalar | None:
    # [FUNCTION DOC] Return None for None, else normalize as an array-like-or-scalar.
    if x is None:
        return None
    return normalize_array_like_or_scalar(x, parm)  # codespell:ignore


def normalize_optional_array_like(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | None:
    # [FUNCTION DOC] Return None for None, else convert to a tensor. Needed because the plain array-like normalizer is not triggered for Optional annotations.
    # This explicit normalizer is needed because otherwise normalize_array_like
    # does not run for a parameter annotated as Optional[ArrayLike]
    return None if x is None else normalize_array_like(x, parm)  # codespell:ignore


def normalize_seq_array_like(
    x: typing.Iterable[object],
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Convert each element of an iterable of array-likes to a tensor; returns a tuple of tensors.
    return tuple(normalize_array_like(value) for value in x)


def normalize_dtype(
    dtype: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.dtype | None:
    # [FUNCTION DOC] Convert a dtype-like (numpy dtype, string, type, torch._numpy dtype) to a torch.dtype; None stays None.
    # cf _decorators.dtype_to_torch
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
    return torch_dtype


def normalize_not_implemented(
    arg: object,
    parm: inspect.Parameter,  # codespell:ignore
) -> None:
    # [FUNCTION DOC] Validate an argument for an unsupported parameter: raises NotImplementedError naming the parameter unless the value equals the parameter default. Returns None.
    if arg != parm.default:  # codespell:ignore
        raise NotImplementedError(
            f"'{parm.name}' parameter is not supported."  # codespell:ignore
        )


def normalize_axis_like(
    arg: AxisLike | ndarray,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> AxisLike:
    # [FUNCTION DOC] Convert an ndarray axis argument to a Python int via the index protocol; other values (int, tuple, None) pass through unchanged.
    from ._ndarray import ndarray

    if isinstance(arg, ndarray):
        return operator.index(arg)
    return arg


def normalize_ndarray(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | None:
    # [FUNCTION DOC] Extract the tensor from an ndarray argument; None passes through. Raises TypeError if the argument is not an ndarray.
    # check the arg is an ndarray, extract its tensor attribute
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        name = parm.name if parm is not None else "argument"  # codespell:ignore
        raise TypeError(f"'{name}' must be an array")
    return arg.tensor


def normalize_outarray(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> ndarray | None:
    # [FUNCTION DOC] Validate an out= argument and return the ndarray itself (not its tensor) so identity is preserved; None passes through.
    # Raw torch tensors (as passed by Dynamo) are wrapped into an ndarray. Raises TypeError for anything else.
    # almost normalize_ndarray, only return the array, not its tensor
    if arg is None:
        return arg
    from ._ndarray import ndarray

    # Dynamo can pass torch tensors as out arguments,
    # wrap it in an ndarray before processing
    if isinstance(arg, torch.Tensor):
        arg = ndarray(arg)

    if not isinstance(arg, ndarray):
        name = parm.name if parm is not None else "argument"  # codespell:ignore
        raise TypeError(f"'{name}' must be an array")
    return arg


def normalize_casting(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> CastingModes:
    # [FUNCTION DOC] Validate a casting mode: must be one of the strings no, equiv, safe, same_kind, unsafe; returns it unchanged, else raises ValueError.
    # KNOWN ISSUE: the registry maps "Optional[CastingModes]" to this function, which rejects None, so an explicit None casting is an error.
    valid = ("no", "equiv", "safe", "same_kind", "unsafe")
    if not isinstance(arg, str) or arg not in valid:
        raise ValueError(
            f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')"
        )
    return arg


# The registered normalizers have intentionally heterogeneous signatures
# (e.g. normalize_seq_array_like takes an iterable), so the common callable
# type erases the parameter list.
normalizers: dict[str, Callable[..., object]] = {
    "ArrayLike": normalize_array_like,
    "ArrayLikeOrScalar": normalize_array_like_or_scalar,
    "Optional[ArrayLike]": normalize_optional_array_like,
    "ArrayLike | None": normalize_optional_array_like,
    "Sequence[ArrayLike]": normalize_seq_array_like,
    "Optional[ArrayLikeOrScalar]": normalize_optional_array_like_or_scalar,
    "ArrayLikeOrScalar | None": normalize_optional_array_like_or_scalar,
    "Optional[NDArray]": normalize_ndarray,
    "NDArray | None": normalize_ndarray,
    "Optional[OutArray]": normalize_outarray,
    "OutArray | None": normalize_outarray,
    "NDArray": normalize_ndarray,
    "Optional[DTypeLike]": normalize_dtype,
    "DTypeLike | None": normalize_dtype,
    "AxisLike": normalize_axis_like,
    "NotImplementedType": normalize_not_implemented,
    "Optional[CastingModes]": normalize_casting,
    "CastingModes | None": normalize_casting,
}


def maybe_normalize(arg: object, parm: inspect.Parameter) -> object:  # codespell:ignore
    # [FUNCTION DOC] Apply the registered normalizer for the parameter's annotation string (looked up in normalizers); return the argument unchanged if none is registered.
    """Normalize arg if a normalizer is registered."""
    normalizer = normalizers.get(parm.annotation)  # codespell:ignore
    return normalizer(arg, parm) if normalizer else arg  # codespell:ignore


# ### Return value helpers ###


def maybe_copy_to(
    out: ndarray | None, result: object, promote_scalar_result: bool = False
) -> object:
    # [FUNCTION DOC] Implement out= semantics on a result. With no out, return the result. For a tensor result, copy it into out.tensor
    # (shape must match; a single-element result may be squeezed to fit a 0-d out only if promote_scalar_result) and return out.
    # Raises ValueError on shape mismatch. For tuple or list results, recurse pairwise with out and return the same container type. Any other result type raises AssertionError.
    # NB: here out is either an ndarray or None
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out.shape}"
                    f" while result.shape = {result.shape}."
                )
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        return type(result)(
            maybe_copy_to(o, r, promote_scalar_result) for o, r in zip(out, result)
        )
    else:
        raise AssertionError  # We should never hit this path


def wrap_tensors(result: object) -> object:
    # [FUNCTION DOC] Recursively wrap torch tensors in ndarray, preserving tuple/list container types; other values are returned unchanged.
    from ._ndarray import ndarray

    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)(wrap_tensors(x) for x in result)
    return result


def array_or_scalar(
    values: torch.Tensor,
    py_type: type[int | float | complex | bool] = float,
    return_scalar: bool = False,
) -> int | float | complex | bool | ndarray:
    # [FUNCTION DOC] Return either a Python scalar of py_type (from the single item of values) when return_scalar is true, or an ndarray wrapping values.
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray

        return ndarray(values)


# ### The main decorator to normalize arguments / postprocess the output ###

_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


@typing.overload
def normalizer(_func: Callable[_P, _R]) -> Callable[_P, _R]: ...


@typing.overload
def normalizer(
    _func: None = None, *, promote_scalar_result: bool = False
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def normalizer(
    _func: Callable[_P, _R] | None = None, *, promote_scalar_result: bool = False
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    # [FUNCTION DOC] Decorator factory/entry point. Used directly (decorating a function) or with promote_scalar_result as a keyword to get a decorator.
    # Returns a wrapper that normalizes arguments according to parameter annotations, runs the function, applies keepdims and out= handling, and wraps tensor results into ndarray.
    def normalizer_inner(func: Callable[_P, _R]) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: Build the normalizing wrapper around func and return it (functools.wraps preserves metadata).
        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwds: _P.kwargs) -> _R:
            # [FUNCTION DOC] HELPER: Normalize positional and keyword arguments using the signature of func (positional args are matched by position; a leading variadic parameter normalizes all positionals; unknown kwargs pass through),
            # call func, then if a keepdims-annotated parameter is true re-insert reduced axes into the result, then apply out= copy semantics, and wrap tensors in ndarray.
            # Raises AssertionError if keepdims is used with a non-tensor first argument; other errors come from normalizers or func.
            sig = inspect.signature(func)
            params = sig.parameters
            first_param = next(iter(params.values()))

            # NumPy's API does not have positional args before variadic positional args
            norm_args: tuple[object, ...]
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                norm_args = tuple(maybe_normalize(arg, first_param) for arg in args)
            else:
                # NB: extra unknown arguments: pass through, will raise in func(*args) below
                norm_args = (
                    tuple(
                        maybe_normalize(arg, parm)  # codespell:ignore
                        for arg, parm in zip(args, params.values())  # codespell:ignore
                    )
                    + args[len(params.values()) :]
                )

            norm_kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }

            # The normalizer rebuilds args/kwds, so they are no longer the
            # ParamSpec-tracked *args/**kwargs; the call is correct at runtime.
            # pyrefly: ignore[invalid-param-spec]
            result: object = func(*norm_args, **norm_kwds)

            # keepdims
            bound_args = None
            if "keepdims" in params and params["keepdims"].annotation == "KeepDims":
                # keepdims can be in any position so we need sig.bind
                bound_args = sig.bind(*norm_args, **norm_kwds).arguments
                if bound_args.get("keepdims", False):
                    # In this case the first arg is the initial tensor and
                    # the second arg is (optionally) the axis
                    tensor = norm_args[0]
                    axis = bound_args.get("axis")
                    if not isinstance(tensor, torch.Tensor):
                        raise AssertionError("keepdims requires a tensor first arg")
                    # pyrefly: ignore[bad-argument-type]
                    result = _util.apply_keepdims(result, axis, tensor.ndim)

            # out
            if "out" in params:
                # out can be in any position so we need sig.bind
                if bound_args is None:
                    bound_args = sig.bind(*norm_args, **norm_kwds).arguments
                out = bound_args.get("out")
                result = maybe_copy_to(out, result, promote_scalar_result)
            result = wrap_tensors(result)

            return result  # pyrefly: ignore[bad-return]

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
