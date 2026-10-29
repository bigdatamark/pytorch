# [MODULE DOC] Torch-level implementations of numpy-style reductions (sum, prod, mean,
# std, var, min/max, any/all, argmax/argmin, cumsum/cumprod, average, quantile,
# percentile, median, ptp, count_nonzero). Functions take and return torch tensors and
# torch dtypes only; the public layer wraps them with array/dtype normalization.
# Aliases max, min, product and cumproduct are defined. Many numpy arguments (out,
# keepdims, where, initial) are accepted for signature compatibility and handled or
# ignored by the wrapping layer. No import-time side effects besides defining
# functions.
"""Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._normalizations import (
        ArrayLike,
        AxisLike,
        DTypeLike,
        KeepDims,
        NotImplementedType,
        OutArray,
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _deco_axis_expand(
    func: Callable[Concatenate[torch.Tensor, AxisLike, _P], _R],
) -> Callable[Concatenate[torch.Tensor, AxisLike, _P], _R]:
    # [FUNCTION DOC] HELPER: Decorator for reductions whose second positional argument is
    # axis. Normalizes a non-None axis into a tuple of non-negative ints for the tensor
    # rank. An empty axis tuple is handled by reshaping the input with an extra length-one
    # leading dimension and reducing over it, so the wrapped function's own checks still
    # run. Returns the wrapped function (functools.wraps applied).
    """
    Generically handle axis arguments in reductions.
    axis is *always* the 2nd arg in the function so no need to have a look at its signature
    """

    @functools.wraps(func)
    def wrapped(
        a: torch.Tensor, axis: AxisLike = None, *args: _P.args, **kwds: _P.kwargs
    ) -> _R:
        # [FUNCTION DOC] HELPER: Wrapper produced by the axis-expansion decorator; normalizes
        # axis, rewrites the empty-tuple case, then calls the original function with the
        # tensor, the axis and the remaining arguments. Raises whatever axis normalization
        # raises for out-of-range or repeated axes.
        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)

        if axis == ():
            # So we insert a length-one axis and run the reduction along it.
            # We cannot return a.clone() as this would sidestep the checks inside the function
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)

        return func(a, axis, *args, **kwds)

    return wrapped


def _atleast_float(dtype: torch.dtype | None, other_dtype: torch.dtype) -> torch.dtype:
    # [FUNCTION DOC] HELPER: Pick a floating or complex dtype: uses the given dtype, or
    # other_dtype when none given. Bool and integer dtypes map to the default float dtype;
    # float and complex dtypes pass through unchanged.
    """Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    """
    if dtype is None:
        dtype = other_dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        return _dtypes_impl.default_dtypes().float_dtype
    return dtype


@_deco_axis_expand
def count_nonzero(
    a: ArrayLike, axis: AxisLike = None, *, keepdims: KeepDims = False
) -> torch.Tensor:
    # [FUNCTION DOC] Count nonzero elements of a tensor, over axis (already normalized by
    # the decorator) or over everything when axis is None. keepdims is accepted but not
    # used here.
    return a.count_nonzero(axis)


@_deco_axis_expand
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    *,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Index of the maximum value along a single axis (or of the flattened
    # tensor when axis is None). Bool input is cast to uint8 since torch lacks a bool
    # kernel. Raises NotImplementedError for complex input and an error from
    # allow_only_single_axis when multiple axes are given. out and keepdims are ignored
    # here.
    if a.is_complex():
        raise NotImplementedError(f"argmax with dtype={a.dtype}.")

    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmax_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmax(a, axis)


@_deco_axis_expand
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    *,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Index of the minimum value along a single axis (or of the flattened
    # tensor when axis is None). Bool input is cast to uint8. Raises NotImplementedError
    # for complex input; multi-axis tuples are rejected. out and keepdims are ignored
    # here.
    if a.is_complex():
        raise NotImplementedError(f"argmin with dtype={a.dtype}.")

    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmin_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmin(a, axis)


@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Logical OR reduction over a single axis, or over all elements when
    # axis is None. Multi-axis tuples are rejected. out, keepdims and where are not used
    # in this layer.
    axis = _util.allow_only_single_axis(axis)
    return torch.any(a) if axis is None else torch.any(a, dim=axis)


@_deco_axis_expand
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Logical AND reduction over a single axis, or over all elements when
    # axis is None. Multi-axis tuples are rejected. out, keepdims and where are not used
    # in this layer.
    axis = _util.allow_only_single_axis(axis)
    return torch.all(a) if axis is None else torch.all(a, dim=axis)


@_deco_axis_expand
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Maximum reduction over axis (tuple allowed) or over all elements when
    # axis is None. Raises NotImplementedError for complex input. out, keepdims, initial
    # and where are not used here. Also exported as max.
    if a.is_complex():
        raise NotImplementedError(f"amax with dtype={a.dtype}")

    return a.amax() if axis is None else a.amax(axis)


max = amax


@_deco_axis_expand
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Minimum reduction over axis (tuple allowed) or over all elements when
    # axis is None. Raises NotImplementedError for complex input. out, keepdims, initial
    # and where are not used here. Also exported as min.
    if a.is_complex():
        raise NotImplementedError(f"amin with dtype={a.dtype}")

    return a.amin() if axis is None else a.amin(axis)


min = amin


@_deco_axis_expand
def ptp(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Peak-to-peak range: maximum minus minimum along axis, or over all
    # elements when axis is None. No complex check is made. out and keepdims are not used
    # here.
    amax = a.amax() if axis is None else a.amax(axis)
    amin = a.amin() if axis is None else a.amin(axis)
    return amax - amin


@_deco_axis_expand
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Sum over axis (tuple allowed) or all elements. dtype must be None or
    # a torch dtype (otherwise raises AssertionError); a bool dtype is replaced by the
    # default integer dtype. Returns the reduced tensor. out, keepdims, initial and where
    # are not used here.
    # KNOWN ISSUE: only an explicitly requested bool dtype is promoted; bool input with
    #   dtype None is passed to torch sum, which itself promotes to int64.
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise AssertionError(
            f"dtype must be None or a torch.dtype, got {type(dtype).__name__}"
        )

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    if axis is None:
        return a.sum(dtype=dtype)
    return a.sum(dim=axis, dtype=dtype)


@_deco_axis_expand
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Product over a single axis or all elements when axis is None (multi-
    # axis tuples rejected). A bool dtype is replaced by the default integer dtype. Other
    # arguments (out, keepdims, initial, where) are unused here. Also exported as product.
    axis = _util.allow_only_single_axis(axis)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    if axis is None:
        return a.prod(dtype=dtype)
    return a.prod(dim=axis, dtype=dtype)


product = prod


@_deco_axis_expand
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Mean over axis or all elements. The computation dtype is the
    # requested dtype, or the input dtype made floating (default float for bool and ints).
    # Returns a tensor of that dtype. out, keepdims and where are unused here.
    dtype = _atleast_float(dtype, a.dtype)

    if axis is None:
        return a.mean(dtype=dtype)
    return a.mean(dim=axis, dtype=dtype)


@_deco_axis_expand
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    ddof: float = 0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Standard deviation with delta degrees of freedom ddof. Casts the
    # input to a floating dtype, reduces over axis, then casts the result to the
    # originally requested dtype (unchanged if None). out, keepdims and where are unused
    # here.
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.std(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)


@_deco_axis_expand
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    ddof: float = 0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Variance with delta degrees of freedom ddof. Casts the input to a
    # floating dtype, reduces over axis, then casts the result to the originally requested
    # dtype (unchanged if None). out, keepdims and where are unused here.
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.var(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)


# cumsum / cumprod are almost reductions:
#   1. no keepdims
#   2. axis=None flattens


def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Cumulative sum. dtype defaults to the input dtype, with bool mapped
    # to the default integer dtype. When axis is None the input is flattened; otherwise
    # axis is normalized to a valid index. Returns a tensor of the same size as the
    # (flattened) input. out is unused here.
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype

    # pyrefly: ignore[bad-argument-type]
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    # torch accepts the numpy-compat `axis=` alias, but its stub only lists `dim`.
    # pyrefly: ignore[missing-argument, unexpected-keyword]
    return a.cumsum(axis=axis, dtype=dtype)


def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Cumulative product. dtype defaults to the input dtype, with bool
    # mapped to the default integer dtype. When axis is None the input is flattened;
    # otherwise axis is normalized to a valid index. Returns a tensor of the same size as
    # the (flattened) input. out is unused here. Also exported as cumproduct.
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype

    # pyrefly: ignore[bad-argument-type]
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    # torch accepts the numpy-compat `axis=` alias, but its stub only lists `dim`.
    # pyrefly: ignore[missing-argument, unexpected-keyword]
    return a.cumprod(axis=axis, dtype=dtype)


cumproduct = cumprod


def average(
    a: ArrayLike,
    axis: int | None = None,
    weights: ArrayLike | None = None,
    returned: bool = False,
    *,
    keepdims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Weighted or plain average along a single axis. Without weights this
    # is the mean, and the weight sum is the element count per output. With weights, non-
    # float input becomes double; if shapes differ, axis is required and weights must be
    # 1D with length matching that axis (TypeError or ValueError otherwise), and weights
    # are broadcast along the axis. Result is the weighted sum divided by the weight sum.
    # keepdims re-expands the reduced axis. When returned is true, returns a pair of
    # result and the weight sum broadcast to the result shape.
    if weights is None:
        result = mean(a, axis)
        wsum = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    else:
        if not a.dtype.is_floating_point:
            a = a.double()

        # axis & weights
        if a.shape != weights.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            if weights.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ."
                )
            if weights.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            # setup weight to broadcast along axis
            weights = torch.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)

        # do the work
        result_dtype = _dtypes_impl.result_type_impl(a, weights)
        numerator = sum(a * weights, axis, dtype=result_dtype)
        wsum = sum(weights, axis, dtype=result_dtype)
        result = numerator / wsum

    # We process keepdims manually because the decorator does not deal with variadic returns
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)

    if returned:
        if wsum.shape != result.shape:
            wsum = torch.broadcast_to(wsum, result.shape).clone()
        return result, wsum
    else:
        return result


# Not using deco_axis_expand as it assumes that axis is the second arg
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Quantile of a tensor with q in [0, 1], along one axis or the
    # flattened input when axis is None. Non-float input is cast to the default float
    # dtype and float16 to float32; q is cast to the input dtype. method is passed to
    # torch as the interpolation mode. overwrite_input is ignored (always out-of-place).
    # Multi-axis tuples are rejected. out, keepdims and interpolation are unused here.
    # KNOWN ISSUE: keepdims is accepted but never applied.
    if overwrite_input:
        # raise NotImplementedError("overwrite_input in quantile not implemented.")
        # NumPy documents that `overwrite_input` MAY modify inputs:
        # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile
        # Here we choose to work out-of-place because why not.
        pass

    if not a.dtype.is_floating_point:
        dtype = _dtypes_impl.default_dtypes().float_dtype
        a = a.to(dtype)

    # edge case: torch.quantile only supports float32 and float64
    if a.dtype == torch.float16:
        a = a.to(torch.float32)

    if axis is None:
        a = a.flatten()
        q = q.flatten()
        axis = (0,)
    else:
        axis = _util.normalize_axis_tuple(axis, a.ndim)

    # FIXME(Mario) Doesn't np.quantile accept a tuple?
    # torch.quantile does accept a number. If we don't want to implement the tuple behaviour
    # (it's deffo low prio) change `normalize_axis_tuple` into a normalize_axis index above.
    axis = _util.allow_only_single_axis(axis)

    q = _util.cast_if_needed(q, a.dtype)

    return torch.quantile(a, q, axis, interpolation=method)


def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Percentile: converts q (0-100) to a fraction (integer q first becomes
    # default float) and delegates to quantile with the same axis, method and other
    # arguments. out is not forwarded.
    # np.percentile(float_tensor, 30) : q.dtype is int64 => q / 100.0 is float32
    if _dtypes_impl.python_type_for_torch(q.dtype) is int:
        q = q.to(_dtypes_impl.default_dtypes().float_dtype)
    qq = q / 100.0

    return quantile(
        a,
        qq,
        axis=axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )


def median(
    a: ArrayLike,
    axis: int | None = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    # [FUNCTION DOC] Median, implemented as the 0.5 quantile along axis (or of the
    # flattened input). Delegates to quantile and forwards out, overwrite_input and
    # keepdims.
    return quantile(
        a,
        torch.as_tensor(0.5),
        axis=axis,
        overwrite_input=overwrite_input,
        out=out,
        keepdims=keepdims,
    )
