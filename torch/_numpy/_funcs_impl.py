# mypy: ignore-errors

# [MODULE DOC] Implementations of NumPy-compatible functions operating directly on torch tensors and torch dtypes.
# Functions here take tensors (and scalars) with NumPy-style signatures; type annotations such as ArrayLike, DTypeLike, OutArray and
# NotImplementedType are markers consumed by the normalizer decorator when _funcs.py re-exports every public function here
# (argument conversion, out= handling, wrapping results in ndarray). Covers array joining and splitting, creation routines,
# cov/corrcoef/convolve/correlate, element selection, shape manipulation, sorting, math, reductions helpers, indexing and
# more. Underscore-prefixed names are internal and not exported. No import-time side effects.
"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""

# Contents of this module ends up in the main namespace via _funcs.py
# where type annotations are used in conjunction with the @normalizer decorator.
from __future__ import annotations

import builtins
import itertools
import operator
from typing import TYPE_CHECKING

import torch

from . import _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._normalizations import (
        ArrayLike,
        ArrayLikeOrScalar,
        CastingModes,
        DTypeLike,
        NDArray,
        NotImplementedType,
        OutArray,
    )


def copy(
    a: ArrayLike, order: NotImplementedType = "K", subok: NotImplementedType = False
):
    # [FUNCTION DOC] Return a copy of the tensor (clone). order and subok are unsupported markers.
    return a.clone()


def copyto(
    dst: NDArray,
    src: ArrayLike,
    casting: CastingModes | None = "same_kind",
    where: NotImplementedType = None,
):
    # [FUNCTION DOC] Copy src into dst in place after casting src to dst's dtype under the casting rule (default same_kind).
    # Mutates dst; returns None. Raises a casting error if the cast is not allowed. The where argument is unsupported.
    (src,) = _util.typecast_tensors((src,), dst.dtype, casting=casting)
    dst.copy_(src)


def atleast_1d(*arys: ArrayLike):
    # [FUNCTION DOC] Return each input viewed as at least 1-d: a single tensor for one input, a list of tensors for several.
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_2d(*arys: ArrayLike):
    # [FUNCTION DOC] Return each input viewed as at least 2-d: a single tensor for one input, a list of tensors for several.
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_3d(*arys: ArrayLike):
    # [FUNCTION DOC] Return each input viewed as at least 3-d: a single tensor for one input, a list of tensors for several.
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def _concat_check(tup, dtype, out):
    # [FUNCTION DOC] HELPER: Validate arguments of concatenation-like functions. Raises ValueError if the sequence is empty and TypeError if both out and dtype are given.
    if tup == ():
        raise ValueError("need at least one array to concatenate")

    """Check inputs in concatenate et al."""
    if out is not None and dtype is not None:
        # mimic numpy
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an "
            "argument, but both were provided."
        )


def _concat_cast_helper(tensors, out=None, dtype=None, casting="same_kind"):
    # [FUNCTION DOC] HELPER: Pick the target dtype (explicit dtype, the out array dtype, or the promoted dtype of all inputs) and cast every tensor to it under the casting rule. Returns the cast tensors.
    """Figure out dtypes, cast if necessary."""

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)

    # cast input arrays if necessary; do not broadcast them against `out`
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)

    return tensors


def _concatenate(
    tensors, axis=0, out=None, dtype=None, casting: CastingModes | None = "same_kind"
):
    # [FUNCTION DOC] HELPER: Pure-torch concatenate: optionally flattens inputs when axis is None, casts to a common dtype, then concatenates along axis. Does not copy into out.
    # pure torch implementation, used below and in cov/corrcoef below
    tensors, axis = _util.axis_none_flatten(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)
    return torch.cat(tensors, axis)


def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis=0,
    out: OutArray | None = None,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Join a sequence of arrays along an existing axis (None flattens them first). Validates arguments, promotes or casts dtypes, and returns the concatenated tensor (out handling is done by the normalizer).
    # Raises ValueError for an empty sequence and TypeError if both out and dtype are given.
    _concat_check(ar_tuple, dtype, out=out)
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    return result


def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Stack arrays vertically (row-wise) after casting to a common dtype; raises ValueError for an empty sequence.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.vstack(tensors)


row_stack = vstack


def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Stack arrays horizontally (column-wise) after casting to a common dtype; raises ValueError for an empty sequence.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.hstack(tensors)


def dstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Stack arrays depth-wise (along the third axis) after casting to a common dtype; raises ValueError for an empty sequence.
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.dstack(tensors)


def column_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Stack 1-d arrays as columns of a 2-d array (2-d arrays are stacked as is) after casting to a common dtype; raises ValueError for an empty sequence.
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.column_stack(tensors)


def stack(
    arrays: Sequence[ArrayLike],
    axis=0,
    out: OutArray | None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # [FUNCTION DOC] Join arrays along a new axis (normalized against the result rank) after casting to a common dtype. Raises ValueError for an empty sequence, TypeError if both out and dtype are given, and an axis error when out of range.
    _concat_check(arrays, dtype, out=out)

    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    return torch.stack(tensors, axis=axis)


def append(arr: ArrayLike, values: ArrayLike, axis=None):
    # [FUNCTION DOC] Append values to arr. With axis None both are flattened (arr only if not already 1-d) and joined along the last axis; otherwise they are concatenated along axis. Returns a new tensor.
    if axis is None:
        if arr.ndim != 1:
            arr = arr.flatten()
        values = values.flatten()
        axis = arr.ndim - 1
    return _concatenate((arr, values), axis=axis)


# ### split ###


def _split_helper(tensor, indices_or_sections, axis, strict=False):
    # [FUNCTION DOC] HELPER: Dispatch splitting: an int means equal sections, a list or tuple means split points. Any other type raises TypeError. The strict flag only affects the int case.
    if isinstance(indices_or_sections, int):
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # NB: drop split=..., it only applies to split_helper_int
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError(f"split_helper: {type(indices_or_sections)}")


def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    # [FUNCTION DOC] HELPER: Split into n sections along axis like NumPy: when the length is not divisible, the first (length mod n) sections get one extra element.
    # Raises NotImplementedError if the argument is not an int, ValueError if n is not positive or (when strict) the division is uneven. Returns a tuple of tensor views.
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("split: indices_or_sections")

    axis = _util.normalize_axis_index(axis, tensor.ndim)

    # numpy: l%n chunks of size (l//n + 1), the rest are sized l//n
    l, n = tensor.shape[axis], indices_or_sections

    if n <= 0:
        raise ValueError

    if l % n == 0:
        num, sz = n, l // n
        lst = [sz] * num
    else:
        if strict:
            raise ValueError("array split does not result in an equal division")

        num, sz = l % n, l // n + 1
        lst = [sz] * num

    lst += [sz - 1] * (n - num)

    return torch.split(tensor, lst, axis)


def _split_helper_list(tensor, indices_or_sections, axis):
    # [FUNCTION DOC] HELPER: Split at the given index list along axis, converting NumPy split indices to torch section lengths. Indices beyond the axis length yield trailing empty pieces.
    # Raises NotImplementedError if the argument is not a list. Returns a tuple of tensors.
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError("split: indices_or_sections: list")
    # numpy expects indices, while torch expects lengths of sections
    # also, numpy appends zero-size arrays for indices above the shape[axis]
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)

    lst.append(tensor.shape[axis])
    lst = [
        lst[0],
    ] + [a - b for a, b in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra

    return torch.split(tensor, lst, axis)


def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    # [FUNCTION DOC] Split into sections or at indices along axis, allowing unequal sections.
    return _split_helper(ary, indices_or_sections, axis)


def split(ary: ArrayLike, indices_or_sections, axis=0):
    # [FUNCTION DOC] Split into sections or at indices along axis; an integer count must divide the length evenly (else ValueError).
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def hsplit(ary: ArrayLike, indices_or_sections):
    # [FUNCTION DOC] Split horizontally: along axis 1 for arrays with two or more dimensions, else axis 0. Raises ValueError for 0-d input or uneven division.
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if ary.ndim > 1 else 0
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def vsplit(ary: ArrayLike, indices_or_sections):
    # [FUNCTION DOC] Split along the first axis. Raises ValueError if the array has fewer than 2 dimensions or the division is uneven.
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return _split_helper(ary, indices_or_sections, 0, strict=True)


def dsplit(ary: ArrayLike, indices_or_sections):
    # [FUNCTION DOC] Split along the third axis. Raises ValueError if the array has fewer than 3 dimensions or the division is uneven.
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return _split_helper(ary, indices_or_sections, 2, strict=True)


def kron(a: ArrayLike, b: ArrayLike):
    # [FUNCTION DOC] Kronecker product of two arrays.
    return torch.kron(a, b)


def vander(x: ArrayLike, N=None, increasing=False):
    # [FUNCTION DOC] Vandermonde matrix of x with N columns (default len(x)); powers increase left to right if increasing is true.
    return torch.vander(x, N, increasing)


# ### linspace, geomspace, logspace and arange ###


def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    retstep=False,
    dtype: DTypeLike | None = None,
    axis=0,
):
    # [FUNCTION DOC] Evenly spaced numbers over [start, stop]. Only axis 0, retstep false and endpoint true are supported (else NotImplementedError).
    # Dtype defaults to the default float dtype. start and stop must be scalars (torch raises TypeError otherwise).
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype: DTypeLike | None = None,
    axis=0,
):
    # [FUNCTION DOC] Numbers spaced evenly on a log scale between start and stop, implemented through logspace with a derived base. Only axis 0 and endpoint true are supported (else NotImplementedError).
    # KNOWN ISSUE: the dtype argument is ignored.
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    return torch.logspace(
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )


def logspace(
    start,
    stop,
    num=50,
    endpoint=True,
    base=10.0,
    dtype: DTypeLike | None = None,
    axis=0,
):
    # [FUNCTION DOC] Numbers spaced evenly on a log scale: base raised to a linear range from start to stop. Only axis 0 and endpoint true are supported (else NotImplementedError).
    # KNOWN ISSUE: dtype None is passed straight to torch rather than defaulting to the default float dtype.
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)


def arange(
    start: ArrayLikeOrScalar | None = None,
    stop: ArrayLikeOrScalar | None = None,
    step: ArrayLikeOrScalar | None = 1,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Evenly spaced values in a half-open interval. A single argument is treated as stop with start 0. Result dtype defaults to the default float dtype if any argument is floating, else the default int dtype.
    # Raises ZeroDivisionError for step 0, TypeError if neither start nor stop given, NotImplementedError for complex arguments. Returns an empty tensor when the range is empty.
    # Complex dtypes are computed in float64 and cast.
    if step == 0:
        raise ZeroDivisionError
    if stop is None and start is None:
        raise TypeError
    if stop is None:
        # XXX: this breaks if start is passed as a kwarg:
        # arange(start=4) should raise (no stop) but doesn't
        start, stop = 0, start
    if start is None:
        start = 0

    # the dtype of the result
    if dtype is None:
        dtype = (
            _dtypes_impl.default_dtypes().float_dtype
            if any(_dtypes_impl.is_float_or_fp_tensor(x) for x in (start, stop, step))
            else _dtypes_impl.default_dtypes().int_dtype
        )
    work_dtype = torch.float64 if dtype.is_complex else dtype

    # RuntimeError: "lt_cpu" not implemented for 'ComplexFloat'. Fall back to eager.
    if any(_dtypes_impl.is_complex_or_complex_tensor(x) for x in (start, stop, step)):
        raise NotImplementedError

    if (step > 0 and start > stop) or (step < 0 and start < stop):
        # empty range
        return torch.empty(0, dtype=dtype)

    result = torch.arange(start, stop, step, dtype=work_dtype)
    result = _util.cast_if_needed(result, dtype)
    return result


# ### zeros/ones/empty/full ###


def empty(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Uninitialized array of the given shape; dtype defaults to the default float dtype.
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.empty(shape, dtype=dtype)


# NB: *_like functions deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.


def empty_like(
    prototype: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # [FUNCTION DOC] Uninitialized array shaped like prototype (optionally reshaped to shape) with optional dtype.
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(
    shape,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Array of the given shape filled with fill_value; an int shape becomes a 1-tuple, dtype defaults to the dtype of fill_value.
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)


def full_like(
    a: ArrayLike,
    fill_value,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # [FUNCTION DOC] Array shaped like a filled with fill_value, with optional dtype and optional reshape to shape.
    # XXX: fill_value broadcasts
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Array of ones of the given shape; dtype defaults to the default float dtype.
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.ones(shape, dtype=dtype)


def ones_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # [FUNCTION DOC] Array of ones shaped like a, with optional dtype and optional reshape to shape.
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def zeros(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Array of zeros of the given shape; dtype defaults to the default float dtype.
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.zeros(shape, dtype=dtype)


def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # [FUNCTION DOC] Array of zeros shaped like a, with optional dtype and optional reshape to shape.
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# ### cov & corrcoef ###


def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    # [FUNCTION DOC] HELPER: Prepare cov/corrcoef input. Without y returns x unchanged. With y, both are promoted to at least 2-d (views), transposed if not rowvar (and not a single row), cloned, and concatenated along axis 0.
    """Prepare inputs for cov and corrcoef."""

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y_tensor is not None:
        # make sure x and y are at least 2D
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()

        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


def corrcoef(
    x: ArrayLike,
    y: ArrayLike | None = None,
    rowvar=True,
    bias=None,
    ddof=None,
    *,
    dtype: DTypeLike | None = None,
):
    # [FUNCTION DOC] Pearson correlation coefficient matrix of the variables in x (and y). The deprecated bias and ddof raise NotImplementedError.
    # Half precision CPU input is computed in float32 and cast back.
    # KNOWN ISSUE: for float16 CPU input a user-supplied dtype is overwritten with float32.
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)

    is_half = (xy_tensor.dtype == torch.float16) and xy_tensor.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)

    if is_half:
        result = result.to(torch.float16)

    return result


def cov(
    m: ArrayLike,
    y: ArrayLike | None = None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights: ArrayLike | None = None,
    aweights: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
):
    # [FUNCTION DOC] Covariance matrix of the variables in m (and y). ddof defaults to 1 unless bias is set (then 0); fweights and aweights are forwarded to torch. Half precision CPU input is computed in float32 and cast back.
    # KNOWN ISSUE: for float16 CPU input a user-supplied dtype is overwritten with float32.
    m = _xy_helper_corrcoef(m, y, rowvar)

    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = (m.dtype == torch.float16) and m.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    m = _util.cast_if_needed(m, dtype)
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)

    if is_half:
        result = result.to(torch.float16)

    return result


def _conv_corr_impl(a, v, mode):
    # [FUNCTION DOC] HELPER: 1-d cross-correlation core shared by convolve and correlate. Promotes both inputs to a common dtype and runs a 1-d convolution with padding derived from mode (full, valid, same).
    # Raises NotImplementedError for mode same with an even-length second input. Returns a 1-d tensor.
    dt = _dtypes_impl.result_type_impl(a, v)
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)

    padding = v.shape[0] - 1 if mode == "full" else mode

    if padding == "same" and v.shape[0] % 2 == 0:
        # UserWarning: Using padding='same' with even kernel lengths and odd
        # dilation may require a zero-padded copy of the input be created
        # (Triggered internally at pytorch/aten/src/ATen/native/Convolution.cpp:1010.)
        raise NotImplementedError("mode='same' and even-length weights")

    # NumPy only accepts 1D arrays; PyTorch requires 2D inputs and 3D weights
    aa = a[None, :]
    vv = v[None, None, :]

    result = torch.nn.functional.conv1d(aa, vv, padding=padding)

    # torch returns a 2D result, numpy returns a 1D array
    return result[0, :]


def convolve(a: ArrayLike, v: ArrayLike, mode="full"):
    # [FUNCTION DOC] Discrete convolution of two 1-d arrays. Swaps the inputs if the second is longer, flips the kernel (torch computes correlation), then calls the shared implementation. mode is full, valid or same.
    # NumPy: if v is longer than a, the arrays are swapped before computation
    if a.shape[0] < v.shape[0]:
        a, v = v, a

    # flip the weights since numpy does and torch does not
    v = torch.flip(v, (0,))

    return _conv_corr_impl(a, v, mode)


def correlate(a: ArrayLike, v: ArrayLike, mode="valid"):
    # [FUNCTION DOC] Cross-correlation of two 1-d arrays: conjugates the second input and applies the shared implementation (default mode valid).
    v = torch.conj_physical(v)
    return _conv_corr_impl(a, v, mode)


# ### logic & element selection ###


def bincount(x: ArrayLike, /, weights: ArrayLike | None = None, minlength=0):
    # [FUNCTION DOC] Count occurrences of each non-negative int in x, optionally weighted, with at least minlength bins. Empty input is allowed. x is safely cast to the default int dtype (raises on unsafe cast).
    if x.numel() == 0:
        # edge case allowed by numpy
        x = x.new_empty(0, dtype=int)

    int_dtype = _dtypes_impl.default_dtypes().int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting="safe")

    return torch.bincount(x, weights, minlength)


def where(
    condition: ArrayLike,
    x: ArrayLikeOrScalar | None = None,
    y: ArrayLikeOrScalar | None = None,
    /,
):
    # [FUNCTION DOC] Choose elements from x or y by condition; with only the condition returns the indices of nonzero elements (tuple of tensors). Raises ValueError if exactly one of x and y is given. Non-bool condition is converted to bool.
    if (x is None) != (y is None):
        raise ValueError("either both or neither of x and y should be given")

    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)

    if x is None and y is None:
        result = torch.where(condition)
    else:
        result = torch.where(condition, x, y)
    return result


# ###### module-level queries of object properties


def ndim(a: ArrayLike):
    # [FUNCTION DOC] Number of dimensions.
    return a.ndim


def shape(a: ArrayLike):
    # [FUNCTION DOC] Shape as a tuple.
    return tuple(a.shape)


def size(a: ArrayLike, axis=None):
    # [FUNCTION DOC] Total number of elements, or the length along axis if given.
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]


# ###### shape manipulations and indexing


def expand_dims(a: ArrayLike, axis):
    # [FUNCTION DOC] Insert length-1 axes at the given positions; returns a view (never copies).
    shape = _util.expand_shape(a.shape, axis)
    return a.view(shape)  # never copies


def flip(m: ArrayLike, axis=None):
    # [FUNCTION DOC] Reverse the order of elements along the given axis or axes (default all). Returns a copy, unlike NumPy which returns a view.
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


def flipud(m: ArrayLike):
    # [FUNCTION DOC] Reverse the order of rows (flip along the first axis).
    return torch.flipud(m)


def fliplr(m: ArrayLike):
    # [FUNCTION DOC] Reverse the order of columns (flip along the second axis); requires at least 2 dimensions.
    return torch.fliplr(m)


def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    # [FUNCTION DOC] Rotate an array by 90 degrees k times in the plane given by axes (normalized against the rank). Returns a new tensor.
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    return torch.rot90(m, k, axes)


# ### broadcasting and indices ###


def broadcast_to(array: ArrayLike, shape, subok: NotImplementedType = False):
    # [FUNCTION DOC] Broadcast the array to the given shape (returns a view); subok is an unsupported marker.
    return torch.broadcast_to(array, size=shape)


# This is a function from tuples to tuples, so we just reuse it.  However,
# dynamo expects its __module__ to be torch._numpy
def broadcast_shapes(*args):
    # [FUNCTION DOC] Compute the broadcast of several shapes (tuples) and return the resulting shape tuple. Kept in this module so dynamo sees it as part of torch._numpy.
    return torch.broadcast_shapes(*args)


def broadcast_arrays(*args: ArrayLike, subok: NotImplementedType = False):
    # [FUNCTION DOC] Broadcast several arrays against each other; returns a tuple of tensors (views).
    return torch.broadcast_tensors(*args)


def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing="xy"):
    # [FUNCTION DOC] Coordinate matrices from coordinate vectors. indexing must be xy or ij (else ValueError); xy swaps the first two axes. sparse keeps broadcastable vectors, otherwise full matrices; copy clones the outputs. Returns a list of tensors.
    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = torch.broadcast_tensors(*output)

    if copy:
        output = [x.clone() for x in output]

    return list(output)  # match numpy, return a list


def indices(dimensions, dtype: DTypeLike | None = int, sparse=False):
    # [FUNCTION DOC] Grid of indices for the given dimensions. Dense output is a tensor of shape (N, *dimensions) with dtype dtype; sparse output is a tuple of N broadcastable index tensors. Default dtype is the Python int type.
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1691-L1791
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = ()
    else:
        res = torch.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = torch.arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1 :]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


# ### tri*-something ###


def tril(m: ArrayLike, k=0):
    # [FUNCTION DOC] Lower triangle of the matrix (elements above the k-th diagonal zeroed).
    return torch.tril(m, k)


def triu(m: ArrayLike, k=0):
    # [FUNCTION DOC] Upper triangle of the matrix (elements below the k-th diagonal zeroed).
    return torch.triu(m, k)


def tril_indices(n, k=0, m=None):
    # [FUNCTION DOC] Row and column indices of the lower triangle of an n-by-m matrix (m defaults to n) with diagonal offset k. Returns a 2-row tensor, not a tuple.
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)


def triu_indices(n, k=0, m=None):
    # [FUNCTION DOC] Row and column indices of the upper triangle of an n-by-m matrix (m defaults to n) with diagonal offset k. Returns a 2-row tensor, not a tuple.
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)


def tril_indices_from(arr: ArrayLike, k=0):
    # [FUNCTION DOC] Lower-triangle indices for the shape of a 2-d array. Raises ValueError if not 2-d. Returns a tensor to avoid a graph break.
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)


def triu_indices_from(arr: ArrayLike, k=0):
    # [FUNCTION DOC] Upper-triangle indices for the shape of a 2-d array. Raises ValueError if not 2-d. Returns a tensor to avoid a graph break.
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)


def tri(
    N,
    M=None,
    k=0,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] Array with ones at and below the k-th diagonal and zeros elsewhere, of shape (N, M) with M defaulting to N.
    # KNOWN ISSUE: dtype None is passed to torch directly, giving torch's default dtype rather than the configured default float dtype.
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    return torch.tril(tensor, diagonal=k)


# ### equality, equivalence, allclose ###


def isclose(a: ArrayLike, b: ArrayLike, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    # [FUNCTION DOC] Elementwise closeness test after promoting both inputs to a common dtype; returns a boolean tensor.
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    # [FUNCTION DOC] True if all elements are close after promoting both inputs to a common dtype; returns a bool.
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _tensor_equal(a1, a2, equal_nan=False):
    # [FUNCTION DOC] HELPER: Shared implementation of array_equal/array_equiv: False on shape mismatch, otherwise whether all elements are equal (optionally treating NaNs as equal). Returns a Python bool.
    # Implementation of array_equal/array_equiv.
    if a1.shape != a2.shape:
        return False
    cond = a1 == a2
    if equal_nan:
        cond = cond | (torch.isnan(a1) & torch.isnan(a2))
    return cond.all().item()


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    # [FUNCTION DOC] True if both arrays have the same shape and elements; NaNs compare equal only if equal_nan.
    return _tensor_equal(a1, a2, equal_nan=equal_nan)


def array_equiv(a1: ArrayLike, a2: ArrayLike):
    # [FUNCTION DOC] True if the arrays are broadcastable and equal after broadcasting; False if broadcasting fails.
    # *almost* the same as array_equal: _equiv tries to broadcast, _equal does not
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return _tensor_equal(a1_t, a2_t)


def nan_to_num(
    x: ArrayLike, copy: NotImplementedType = True, nan=0.0, posinf=None, neginf=None
):
    # [FUNCTION DOC] Replace NaN with nan and infinities with posinf/neginf (defaults: largest/smallest finite values). Complex input is processed on real and imaginary parts separately. copy is an unsupported marker.
    # work around RuntimeError: "nan_to_num" not implemented for 'ComplexDouble'
    if x.is_complex():
        re = torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf)
        im = torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
        return re + 1j * im
    else:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ### put/take_along_axis ###


def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis=None,
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
):
    # [FUNCTION DOC] Take elements along an axis (None flattens first) using the index array, via indexing; mode and out are not handled here.
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result


def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    # [FUNCTION DOC] Gather values from arr along axis using indices of matching dimensionality; axis None flattens arr first.
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    return torch.take_along_dim(arr, indices, axis)


def put(
    a: NDArray,
    indices: ArrayLike,
    values: ArrayLike,
    mode: NotImplementedType = "raise",
):
    # [FUNCTION DOC] Replace elements of a (flat positions given by indices) with values, in place. values is cast to a's dtype, repeated or truncated to match the number of indices. mode is unsupported. Returns None.
    v = values.type(a.dtype)
    # If indices is larger than v, expand v to at least the size of indices. Any
    # unnecessary trailing elements are then trimmed.
    if indices.numel() > v.numel():
        ratio = (indices.numel() + v.numel() - 1) // v.numel()
        v = v.unsqueeze(0).expand((ratio,) + v.shape)
    # Trim unnecessary elements, regardless if v was expanded or not. Note
    # np.put() trims v to match indices by default too.
    if indices.numel() < v.numel():
        v = v.flatten()
        v = v[: indices.numel()]
    a.put_(indices, v)
    return None


def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    # [FUNCTION DOC] Write values into arr in place at indices along axis (None flattens first; indices and values are broadcast together and values cast to arr's dtype). Returns None.
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)

    indices, values = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None


def choose(
    a: ArrayLike,
    choices: Sequence[ArrayLike],
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
):
    # [FUNCTION DOC] Construct an array by picking, per element of a, from the list of choice arrays (broadcast together) using a as the index into the choice list. mode and out are not handled here.
    # First, broadcast elements of `choices`
    choices = torch.stack(torch.broadcast_tensors(*choices))

    # Use an analog of `gather(choices, 0, a)` which broadcasts `choices` vs `a`:
    # (taken from https://github.com/pytorch/pytorch/issues/9407#issuecomment-1427907939)
    idx_list = [
        torch.arange(dim).view((1,) * i + (dim,) + (1,) * (choices.ndim - i - 1))
        for i, dim in enumerate(choices.shape)
    ]

    idx_list[0] = a
    return choices[tuple(idx_list)].squeeze(0)


# ### unique et al. ###


def unique(
    ar: ArrayLike,
    return_index: NotImplementedType = False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan: NotImplementedType = True,
):
    # [FUNCTION DOC] Unique elements (sorted) of ar along axis (None flattens). Returns the unique tensor, or a tuple including inverse indices and/or counts if requested. return_index and equal_nan are unsupported markers.
    (ar,), axis = _util.axis_none_flatten(ar, axis=axis)
    axis = _util.normalize_axis_index(axis, ar.ndim)

    result = torch.unique(
        ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis
    )

    return result


def nonzero(a: ArrayLike):
    # [FUNCTION DOC] Indices of nonzero elements as a tuple of tensors, one per dimension.
    return torch.nonzero(a, as_tuple=True)


def argwhere(a: ArrayLike):
    # [FUNCTION DOC] Indices of nonzero elements as a 2-d tensor with one row per element.
    return torch.argwhere(a)


def flatnonzero(a: ArrayLike):
    # [FUNCTION DOC] Indices of nonzero elements of the flattened array.
    return torch.flatten(a).nonzero(as_tuple=True)[0]


def clip(
    a: ArrayLike,
    min: ArrayLike | None = None,
    max: ArrayLike | None = None,
    out: OutArray | None = None,
):
    # [FUNCTION DOC] Limit values to the interval [min, max] (either may be None). out is handled by the normalizer.
    return torch.clamp(a, min, max)


def repeat(a: ArrayLike, repeats: ArrayLikeOrScalar, axis=None):
    # [FUNCTION DOC] Repeat each element repeats times, along axis (None flattens first).
    return torch.repeat_interleave(a, repeats, axis)


def tile(A: ArrayLike, reps):
    # [FUNCTION DOC] Tile the array by reps (an int is treated as a 1-tuple).
    if isinstance(reps, int):
        reps = (reps,)
    return torch.tile(A, reps)


def resize(a: ArrayLike, new_shape=None):
    # [FUNCTION DOC] Return a new array of new_shape filled by repeating the flattened input (zeros if input or new size is empty), unlike ndarray.resize. new_shape None returns the input unchanged; an int is a 1-tuple.
    # Raises ValueError for negative dimensions.
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/fromnumeric.py#L1420-L1497
    if new_shape is None:
        return a

    if isinstance(new_shape, int):
        new_shape = (new_shape,)

    a = a.flatten()

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")

    if a.numel() == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return torch.zeros(new_shape, dtype=a.dtype)

    repeats = -(-new_size // a.numel())  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


# ### diag et al. ###


def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    # [FUNCTION DOC] Diagonal of the array for the plane (axis1, axis2) with offset (axes normalized); returns a view.
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.diagonal(a, offset, axis1, axis2)


def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
):
    # [FUNCTION DOC] Sum along the offset diagonal in the plane (axis1, axis2), optionally with dtype. out is handled by the normalizer.
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


def eye(
    N,
    M=None,
    k=0,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    # [FUNCTION DOC] 2-d identity-like array of shape (N, M) (M defaults to N) with ones on the k-th diagonal; dtype defaults to the default float dtype.
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def identity(n, dtype: DTypeLike | None = None, *, like: NotImplementedType = None):
    # [FUNCTION DOC] Square identity matrix of size n.
    # KNOWN ISSUE: dtype None is passed to torch directly, giving torch's default dtype rather than the configured default float dtype.
    return torch.eye(n, dtype=dtype)


def diag(v: ArrayLike, k=0):
    # [FUNCTION DOC] Extract the k-th diagonal of a 2-d array or build a 2-d array with v on the k-th diagonal.
    return torch.diag(v, k)


def diagflat(v: ArrayLike, k=0):
    # [FUNCTION DOC] Build a 2-d array with the flattened input on the k-th diagonal.
    return torch.diagflat(v, k)


def diag_indices(n, ndim=2):
    # [FUNCTION DOC] Index arrays for the main diagonal of an array with ndim dimensions and size n each: a tuple of ndim copies of the same range.
    idx = torch.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr: ArrayLike):
    # [FUNCTION DOC] Diagonal indices for an array. Raises ValueError if it has fewer than 2 dimensions or its dimensions are not all equal.
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(s[0], arr.ndim)


def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    # [FUNCTION DOC] Fill the main diagonal of a (at least 2-d) in place with val (scalar or array, trimmed to fit); for tall 2-d matrices wrap restarts the diagonal after the matrix width plus one rows. Returns a. Raises ValueError for fewer than 2 dimensions.
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if val.numel() == 0 and not wrap:
        a.fill_diagonal_(val)
        return a

    if val.ndim == 0:
        val = val.unsqueeze(0)

    # torch.Tensor.fill_diagonal_ only accepts scalars
    # If the size of val is too large, then val is trimmed
    if a.ndim == 2:
        tall = a.shape[0] > a.shape[1]
        # wrap does nothing for wide matrices...
        if not wrap or not tall:
            # Never wraps
            diag = a.diagonal()
            diag.copy_(val[: diag.numel()])
        else:
            # wraps and tall... leaving one empty line between diagonals?!
            max_, min_ = a.shape
            idx = torch.arange(max_ - max_ // (min_ + 1))
            mod = idx % min_
            div = idx // min_
            a[(div * (min_ + 1) + mod, mod)] = val[: idx.numel()]
    else:
        idx = diag_indices_from(a)
        # a.shape = (n, n, ..., n)
        a[idx] = val[: a.shape[0]]

    return a


def vdot(a: ArrayLike, b: ArrayLike, /):
    # [FUNCTION DOC] Dot product of two flattened arrays (first argument conjugated), after promotion to a common dtype; half is computed in float32 and bool as uint8, then converted back.
    # 1. torch only accepts 1D arrays, numpy flattens
    # 2. torch requires matching dtype, while numpy casts (?)
    t_a, t_b = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.flatten()
    if t_b.ndim > 1:
        t_b = t_b.flatten()

    dtype = _dtypes_impl.result_type_impl(t_a, t_b)
    is_half = dtype == torch.float16 and (t_a.is_cpu or t_b.is_cpu)
    is_bool = dtype == torch.bool

    # work around torch's "dot" not implemented for 'Half', 'Bool'
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)

    result = torch.vdot(t_a, t_b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)

    return result


def tensordot(a: ArrayLike, b: ArrayLike, axes=2):
    # [FUNCTION DOC] Tensor dot product over the given axes (int, or pair of axis lists; bare ints in the pair become lists), after promoting both inputs to a common dtype.
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]

    target_dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, target_dtype)
    b = _util.cast_if_needed(b, target_dtype)

    return torch.tensordot(a, b, dims=axes)


def dot(a: ArrayLike, b: ArrayLike, out: OutArray | None = None):
    # [FUNCTION DOC] Dot product: scalar multiplication if either input is 0-d, otherwise matrix multiplication; bool is computed as uint8 and converted back. out is handled by the normalizer.
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_bool = dtype == torch.bool
    if is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    else:
        result = torch.matmul(a, b)

    if is_bool:
        result = result.to(torch.bool)

    return result


def inner(a: ArrayLike, b: ArrayLike, /):
    # [FUNCTION DOC] Inner product over the last axes; half is computed in float32 and bool as uint8, results converted back.
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_half = dtype == torch.float16 and (a.is_cpu or b.is_cpu)
    is_bool = dtype == torch.bool

    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    result = torch.inner(a, b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result


def outer(a: ArrayLike, b: ArrayLike, out: OutArray | None = None):
    # [FUNCTION DOC] Outer product of two vectors. out is handled by the normalizer.
    return torch.outer(a, b)


def cross(a: ArrayLike, b: ArrayLike, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # [FUNCTION DOC] Cross product of 2- or 3-element vectors along axisa/axisb (axis overrides all three), vendored from NumPy; result axis placed at axisc. Inputs are promoted to a common dtype, shapes broadcast.
    # Raises ValueError if a vector dimension is not 2 or 3. The 2-by-2 case returns the scalar z-component.
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1486-L1685
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # Check axisa and axisb are within bounds
    axisa = _util.normalize_axis_index(axisa, a.ndim)
    axisb = _util.normalize_axis_index(axisb, b.ndim)

    # Move working axis to the end of the shape
    a = torch.moveaxis(a, axisa, -1)
    b = torch.moveaxis(b, axisb, -1)
    msg = "incompatible dimensions for cross product\n(dimension must be 2 or 3)"
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

    # Create the output array
    shape = broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = _util.normalize_axis_index(axisc, len(shape))
    dtype = _dtypes_impl.result_type_impl(a, b)
    cp = torch.empty(shape, dtype=dtype)

    # recast arrays as dtype
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp[...] = a0 * b1 - a1 * b0
            return cp
        else:
            if b.shape[-1] != 3:
                raise AssertionError(f"b.shape[-1] must be 3, got {b.shape[-1]}")
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp0[...] = a1 * b2
            cp1[...] = -a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
    else:
        if a.shape[-1] != 3:
            raise AssertionError(f"a.shape[-1] must be 3, got {a.shape[-1]}")
        if b.shape[-1] == 3:
            cp0[...] = a1 * b2 - a2 * b1
            cp1[...] = a2 * b0 - a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
        else:
            if b.shape[-1] != 2:
                raise AssertionError(f"b.shape[-1] must be 2, got {b.shape[-1]}")
            cp0[...] = -a2 * b1
            cp1[...] = a2 * b0
            cp2[...] = a0 * b1 - a1 * b0

    return torch.moveaxis(cp, -1, axisc)


def einsum(*operands, out=None, dtype=None, order="K", casting="safe", optimize=False):
    # [FUNCTION DOC] Einstein summation, normalized manually (exported without the normalizer). Accepts either a subscripts string followed by operands or the alternating operand/sublist format.
    # Operands are converted to tensors, promoted to a common dtype (half on CPU computed in float32, small ints in int64; casting rule enforced), and torch einsum is run. The optimize argument temporarily sets the opt_einsum backend
    # strategy/enabled state (True maps to auto, False disables) and is restored afterwards, even on error. out is copied into via maybe_copy_to; the result is wrapped in an ndarray.
    # Raises TypeError if out is not an ndarray, NotImplementedError if order is not K, ValueError for a bad casting mode.
    # KNOWN ISSUE: the half/short-int upcast is not reverted, so results for those input dtypes have the wider dtype.
    # Have to manually normalize *operands and **kwargs, following the NumPy signature
    # We have a local import to avoid polluting the global space, as it will be then
    # exported in funcs.py
    from ._ndarray import ndarray
    from ._normalizations import (
        maybe_copy_to,
        normalize_array_like,
        normalize_casting,
        normalize_dtype,
        wrap_tensors,
    )

    dtype = normalize_dtype(dtype)
    casting = normalize_casting(casting)
    if out is not None and not isinstance(out, ndarray):
        raise TypeError("'out' must be an array")
    if order != "K":
        raise NotImplementedError("'order' parameter is not supported.")

    # parse arrays and normalize them
    sublist_format = not isinstance(operands[0], str)
    if sublist_format:
        # op, str, op, str ... [sublistout] format: normalize every other argument

        # - if sublistout is not given, the length of operands is even, and we pick
        #   odd-numbered elements, which are arrays.
        # - if sublistout is given, the length of operands is odd, we peel off
        #   the last one, and pick odd-numbered elements, which are arrays.
        #   Without [:-1], we would have picked sublistout, too.
        array_operands = operands[:-1][::2]
    else:
        # ("ij->", arrays) format
        subscripts, array_operands = operands[0], operands[1:]

    tensors = [normalize_array_like(op) for op in array_operands]
    target_dtype = _dtypes_impl.result_type_impl(*tensors) if dtype is None else dtype

    # work around 'bmm' not implemented for 'Half' etc
    is_half = target_dtype == torch.float16 and all(t.is_cpu for t in tensors)
    if is_half:
        target_dtype = torch.float32

    is_short_int = target_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]
    if is_short_int:
        target_dtype = torch.int64

    tensors = _util.typecast_tensors(tensors, target_dtype, casting)

    from torch.backends import opt_einsum

    try:
        # set the global state to handle the optimize=... argument, restore on exit
        if opt_einsum.is_available():
            old_strategy = torch.backends.opt_einsum.strategy
            old_enabled = torch.backends.opt_einsum.enabled

            # torch.einsum calls opt_einsum.contract_path, which runs into
            # https://github.com/dgasmith/opt_einsum/issues/219
            # for strategy={True, False}
            if optimize is True:
                optimize = "auto"
            elif optimize is False:
                torch.backends.opt_einsum.enabled = False

            torch.backends.opt_einsum.strategy = optimize

        if sublist_format:
            # recombine operands
            sublists = operands[1::2]
            has_sublistout = len(operands) % 2 == 1
            if has_sublistout:
                sublistout = operands[-1]
            operands = list(itertools.chain.from_iterable(zip(tensors, sublists)))
            if has_sublistout:
                operands.append(sublistout)

            result = torch.einsum(*operands)
        else:
            result = torch.einsum(subscripts, *tensors)

    finally:
        if opt_einsum.is_available():
            torch.backends.opt_einsum.strategy = old_strategy
            torch.backends.opt_einsum.enabled = old_enabled

    result = maybe_copy_to(out, result)
    return wrap_tensors(result)


# ### sort and partition ###


def _sort_helper(tensor, axis, kind, order):
    # [FUNCTION DOC] HELPER: Shared preparation for sort/argsort. Raises NotImplementedError for complex dtypes. Flattens if axis is None, normalizes axis, and returns (tensor, axis, stable) where stable is true iff kind is stable.
    if tensor.dtype.is_complex:
        raise NotImplementedError(f"sorting {tensor.dtype} is not supported")
    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    stable = kind == "stable"

    return tensor, axis, stable


def sort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    # [FUNCTION DOC] Sorted copy of the array along axis (default last; None flattens). kind stable requests a stable sort; order is unsupported. Complex dtypes raise NotImplementedError.
    # `order` keyword arg is only relevant for structured dtypes; so not supported here.
    a, axis, stable = _sort_helper(a, axis, kind, order)
    result = torch.sort(a, dim=axis, stable=stable)
    return result.values


def argsort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    # [FUNCTION DOC] Indices that would sort the array along axis (default last; None flattens). kind stable requests a stable sort. Complex dtypes raise NotImplementedError.
    a, axis, stable = _sort_helper(a, axis, kind, order)
    return torch.argsort(a, dim=axis, stable=stable)


def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: ArrayLike | None = None
):
    # [FUNCTION DOC] Indices at which values v would be inserted into sorted a to keep order (side left or right, optional sorter). Complex dtypes raise NotImplementedError.
    if a.dtype.is_complex:
        raise NotImplementedError(f"searchsorted with dtype={a.dtype}")

    return torch.searchsorted(a, v, side=side, sorter=sorter)


# ### swap/move/roll axis ###


def moveaxis(a: ArrayLike, source, destination):
    # [FUNCTION DOC] Move axes from source positions to destination positions (both normalized to tuples); returns a view.
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    return torch.moveaxis(a, source, destination)


def swapaxes(a: ArrayLike, axis1, axis2):
    # [FUNCTION DOC] Swap two axes (normalized against the rank); returns a view.
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.swapaxes(a, axis1, axis2)


def rollaxis(a: ArrayLike, axis, start=0):
    # [FUNCTION DOC] Roll the given axis backwards until it lies before position start (vendored from NumPy). Raises an axis error if start is out of range. Returns the input itself when no movement is needed, else a permuted view.
    # Straight vendor from:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
    #
    # Also note this function in NumPy is mostly retained for backwards compat
    # (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
    # so let's not touch it unless hard pressed.
    n = a.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        # numpy returns a view, here we try returning the tensor itself
        # return tensor[...]
        return a
    axes = list(range(n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


def roll(a: ArrayLike, shift, axis=None):
    # [FUNCTION DOC] Roll elements along the given axis or axes (None flattens); an int shift is repeated for every axis given.
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    return torch.roll(a, shift, axis)


# ### shape manipulations ###


def squeeze(a: ArrayLike, axis=None):
    # [FUNCTION DOC] Remove length-1 dimensions: all of them for axis None, none for an empty tuple, else the specified axis or axes.
    # KNOWN ISSUE: for a tuple of axes the loop squeezes the original array each iteration instead of the running result, so only the last axis takes effect.
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    else:
        if isinstance(axis, tuple):
            result = a
            for ax in axis:
                result = a.squeeze(ax)
        else:
            result = a.squeeze(axis)
    return result


def reshape(a: ArrayLike, newshape, order: NotImplementedType = "C"):
    # [FUNCTION DOC] Reshape; newshape is the tuple of the shape (a single-element tuple is unwrapped so that both call styles work). order is unsupported.
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    newshape = newshape[0] if len(newshape) == 1 else newshape
    return a.reshape(newshape)


# NB: cannot use torch.reshape(a, newshape) above, because of
# (Pdb) torch.reshape(torch.as_tensor([1]), 1)
# *** TypeError: reshape(): argument 'shape' (position 2) must be tuple of SymInts, not int


def transpose(a: ArrayLike, axes=None):
    # [FUNCTION DOC] Permute dimensions. No axes (or empty, or None) reverses them; a single-element axes tuple is unwrapped so that both call styles work.
    # numpy allows both .transpose(sh) and .transpose(*sh)
    # also older code uses axes being a list
    if axes in [(), None, (None,)]:
        axes = tuple(reversed(range(a.ndim)))
    elif len(axes) == 1:
        axes = axes[0]
    return a.permute(axes)


def ravel(a: ArrayLike, order: NotImplementedType = "C"):
    # [FUNCTION DOC] Flatten to one dimension; order is an unsupported marker.
    return torch.flatten(a)


def diff(
    a: ArrayLike,
    n=1,
    axis=-1,
    prepend: ArrayLike | None = None,
    append: ArrayLike | None = None,
):
    # [FUNCTION DOC] n-th discrete difference along axis. n must be non-negative (ValueError); n of 0 returns the input. prepend and append are broadcast to the array shape except along axis (scalars get length 1 there) before differencing.
    axis = _util.normalize_axis_index(axis, a.ndim)

    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    if n == 0:
        # match numpy and return the input immediately
        return a

    if prepend is not None:
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)

    if append is not None:
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)

    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)


# ### math functions ###


def angle(z: ArrayLike, deg=False):
    # [FUNCTION DOC] Argument (phase angle) of complex numbers; in degrees if deg is true.
    result = torch.angle(z)
    if deg:
        result = result * (180 / torch.pi)
    return result


def sinc(x: ArrayLike):
    # [FUNCTION DOC] Normalized sinc function.
    return torch.sinc(x)


# NB: have to normalize *varargs manually
def gradient(f: ArrayLike, *varargs, axis=None, edge_order=1):
    # [FUNCTION DOC] Numerical gradient of an N-d array using central differences inside and first- or second-order one-sided differences at edges (vendored from NumPy). Spacing is given in varargs (none, one scalar for all axes, or one scalar or 1-d coordinate array per axis).
    # Integer input is promoted to float64. Returns a single tensor for one axis, else a list per axis. Raises ValueError for bad spacing shapes or lengths, edge_order above 2, or axes too short; TypeError for a wrong number of spacing arguments.
    N = f.ndim  # number of dimensions

    varargs = _util.ndarrays_to_tensors(varargs)

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _util.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and (_dtypes_impl.is_scalar(varargs[0]) or varargs[0].ndim == 0):
        # single scalar or 0D tensor for all axes (np.ndim(varargs[0]) == 0)
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = torch.as_tensor(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            if not (distances.dtype.is_floating_point or distances.dtype.is_complex):
                distances = distances.double()

            diffx = torch.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype
    if _dtypes_impl.python_type_for_torch(otype) in (int, bool):
        # Convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        f = f.double()
        otype = torch.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = torch.empty_like(f, dtype=otype)

        # spacing for the current axis (NB: np.ndim(ax_dx) == 0)
        uniform_spacing = _dtypes_impl.is_scalar(ax_dx) or ax_dx.ndim == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = [1] * N
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


# ### Type/shape etc queries ###


def round(a: ArrayLike, decimals=0, out: OutArray | None = None):
    # [FUNCTION DOC] Round to the given number of decimals. Complex values round real and imaginary parts separately; integer input is returned unchanged. out is handled by the normalizer.
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = torch.complex(
            torch.round(a.real, decimals=decimals),
            torch.round(a.imag, decimals=decimals),
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


around = round
round_ = round


def real_if_close(a: ArrayLike, tol=100):
    # [FUNCTION DOC] Return the real part if all imaginary parts are below tol (relative to dtype epsilon when tol is above 1, else absolute); otherwise the array unchanged. Non-complex input is returned unchanged.
    if not torch.is_complex(a):
        return a
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(a.dtype).eps

    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a


def real(a: ArrayLike):
    # [FUNCTION DOC] Real part of the array.
    return torch.real(a)


def imag(a: ArrayLike):
    # [FUNCTION DOC] Imaginary part of the array (zeros for non-complex input).
    if a.is_complex():
        return a.imag
    return torch.zeros_like(a)


def iscomplex(x: ArrayLike):
    # [FUNCTION DOC] Boolean mask of elements with nonzero imaginary part (all False for non-complex input).
    if torch.is_complex(x):
        return x.imag != 0
    return torch.zeros_like(x, dtype=torch.bool)


def isreal(x: ArrayLike):
    # [FUNCTION DOC] Boolean mask of elements with zero imaginary part (all True for non-complex input).
    if torch.is_complex(x):
        return x.imag == 0
    return torch.ones_like(x, dtype=torch.bool)


def iscomplexobj(x: ArrayLike):
    # [FUNCTION DOC] True if the array has a complex dtype.
    return torch.is_complex(x)


def isrealobj(x: ArrayLike):
    # [FUNCTION DOC] True if the array does not have a complex dtype.
    return not torch.is_complex(x)


def isneginf(x: ArrayLike, out: OutArray | None = None):
    # [FUNCTION DOC] Boolean mask of negative infinities. out is handled by the normalizer.
    return torch.isneginf(x)


def isposinf(x: ArrayLike, out: OutArray | None = None):
    # [FUNCTION DOC] Boolean mask of positive infinities. out is handled by the normalizer.
    return torch.isposinf(x)


def i0(x: ArrayLike):
    # [FUNCTION DOC] Modified Bessel function of the first kind, order 0.
    return torch.special.i0(x)


def isscalar(a):
    # [FUNCTION DOC] True if the argument converts to an array with exactly one element; False if conversion fails. Imports the array normalizer locally to avoid exporting it.
    # KNOWN ISSUE: differs from NumPy, which tests for scalar types; any single-element array-like (e.g. a one-item list) returns True.
    # We need to use normalize_array_like, but we don't want to export it in funcs.py
    from ._normalizations import normalize_array_like

    try:
        t = normalize_array_like(a)
        return t.numel() == 1
    except Exception:
        return False


# ### Filter windows ###


def hamming(M):
    # [FUNCTION DOC] Hamming window of M points (non-periodic) in the default float dtype.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)


def hanning(M):
    # [FUNCTION DOC] Hann window of M points (non-periodic) in the default float dtype.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)


def kaiser(M, beta):
    # [FUNCTION DOC] Kaiser window of M points with shape parameter beta (non-periodic) in the default float dtype.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)


def blackman(M):
    # [FUNCTION DOC] Blackman window of M points (non-periodic) in the default float dtype.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)


def bartlett(M):
    # [FUNCTION DOC] Bartlett window of M points (non-periodic) in the default float dtype.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)


# ### Dtype routines ###

# vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L666


array_type = [
    [torch.float16, torch.float32, torch.float64],
    [None, torch.complex64, torch.complex128],
]
array_precision = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.complex64: 1,
    torch.complex128: 2,
}


def common_type(*tensors: ArrayLike):
    # [FUNCTION DOC] Common floating dtype of the inputs, vendored from NumPy: integer inputs count as double precision, complex if any input is complex. Returns a torch dtype from the array_type table.
    # Raises TypeError for dtypes not in the precision table (e.g. bfloat16).
    is_complex = False
    precision = 0
    for a in tensors:
        t = a.dtype
        if iscomplexobj(a):
            is_complex = True
        if not (t.is_floating_point or t.is_complex):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = builtins.max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]


# ### histograms ###


def histogram(
    a: ArrayLike,
    bins: ArrayLike = 10,
    range=None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    # [FUNCTION DOC] Histogram of a over bins (int count or edges), optional range, weights and density. The deprecated normed raises ValueError; complex weights raise NotImplementedError. Integer input is computed in double.
    # Returns (counts, bin edges); counts are int64 when unweighted-or-integer-weighted and not density, and edges are int64 for integer input.
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    if weights is not None and weights.dtype.is_complex:
        raise NotImplementedError("complex weights histogram.")

    is_a_int = not (a.dtype.is_floating_point or a.dtype.is_complex)
    is_w_int = weights is None or not weights.dtype.is_floating_point
    if is_a_int:
        a = a.double()

    if weights is not None:
        weights = _util.cast_if_needed(weights, a.dtype)

    if isinstance(bins, torch.Tensor):
        if bins.ndim == 0:
            # bins was a single int
            bins = operator.index(bins)
        else:
            bins = _util.cast_if_needed(bins, a.dtype)

    if range is None:
        h, b = torch.histogram(a, bins, weight=weights, density=bool(density))
    else:
        h, b = torch.histogram(
            a, bins, range=range, weight=weights, density=bool(density)
        )

    if not density and is_w_int:
        h = h.long()
    if is_a_int:
        b = b.long()

    return h, b


def histogram2d(
    x,
    y,
    bins=10,
    range: ArrayLike | None = None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    # [FUNCTION DOC] 2-d histogram of x and y (must have equal length, else ValueError) via histogramdd. Returns (counts, x edges, y edges).
    # vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/twodim_base.py#L655-L821
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        bins = [bins, bins]

    h, e = histogramdd((x, y), bins, range, normed, weights, density)

    return h, e[0], e[1]


def histogramdd(
    sample,
    bins=10,
    range: ArrayLike | None = None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    # [FUNCTION DOC] Multi-dimensional histogram of sample (a list/tuple is treated as columns, a 2-d array as rows of points), normalized manually. bins is an int, a sequence of ints, or a sequence of edge arrays (cast to the sample dtype and cast back on output).
    # The deprecated normed raises ValueError. Returns (counts, list of edge tensors).
    # KNOWN ISSUE: when weights are given the range is recomputed from the sample min/max, silently overriding a user-supplied range.
    # have to normalize manually because `sample` interpretation differs
    # for a list of lists and a 2D array
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    from ._normalizations import normalize_array_like, normalize_seq_array_like

    if isinstance(sample, (list, tuple)):
        sample = normalize_array_like(sample).T
    else:
        sample = normalize_array_like(sample)

    sample = torch.atleast_2d(sample)

    if not (sample.dtype.is_floating_point or sample.dtype.is_complex):
        sample = sample.double()

    # bins is either an int, or a sequence of ints or a sequence of arrays
    bins_is_array = not (
        isinstance(bins, int) or builtins.all(isinstance(b, int) for b in bins)
    )
    if bins_is_array:
        bins = normalize_seq_array_like(bins)
        bins_dtypes = [b.dtype for b in bins]
        bins = [_util.cast_if_needed(b, sample.dtype) for b in bins]

    if range is not None:
        range = range.flatten().tolist()

    if weights is not None:
        # range=... is required : interleave min and max values per dimension
        mm = sample.aminmax(dim=0)
        range = torch.cat(mm).reshape(2, -1).T.flatten()
        range = tuple(range.tolist())
        weights = _util.cast_if_needed(weights, sample.dtype)
        w_kwd = {"weight": weights}
    else:
        w_kwd = {}

    h, b = torch.histogramdd(sample, bins, range, density=bool(density), **w_kwd)

    if bins_is_array:
        b = [_util.cast_if_needed(bb, dtyp) for bb, dtyp in zip(b, bins_dtypes)]

    return h, b


# ### odds and ends


def min_scalar_type(a: ArrayLike, /):
    # [FUNCTION DOC] Smallest dtype that can hold the single value in a (like NumPy min_scalar_type); arrays with more than one element return their own dtype. Returns a torch._numpy DType wrapper.
    # Bool stays bool; ints prefer unsigned; floats and complex check ranges.
    # KNOWN ISSUE: a float NaN or inf value fits no range so the chosen dtype is never assigned and an unbound-variable error results.
    # https://github.com/numpy/numpy/blob/maintenance/1.24.x/numpy/core/src/multiarray/convert_datatype.c#L1288

    from ._dtypes import DType

    if a.numel() > 1:
        # numpy docs: "For non-scalar array a, returns the vector's dtype unmodified."
        return DType(a.dtype)

    if a.dtype == torch.bool:
        dtype = torch.bool

    elif a.dtype.is_complex:
        fi = torch.finfo(torch.float32)
        fits_in_single = a.dtype == torch.complex64 or (
            fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max
        )
        dtype = torch.complex64 if fits_in_single else torch.complex128

    elif a.dtype.is_floating_point:
        for dt in [torch.float16, torch.float32, torch.float64]:
            fi = torch.finfo(dt)
            if fi.min <= a <= fi.max:
                dtype = dt
                break
    else:
        # must be integer
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # Prefer unsigned int where possible, as numpy does.
            ii = torch.iinfo(dt)
            if ii.min <= a <= ii.max:
                dtype = dt
                break

    return DType(dtype)


def pad(array: ArrayLike, pad_width: ArrayLike, mode="constant", **kwargs):
    # [FUNCTION DOC] Constant padding of an array; pad_width is broadcast to one (before, after) pair per axis. Only mode constant is supported (else NotImplementedError); constant_values is read from kwargs (default 0) and converted to the array's Python scalar kind.
    if mode != "constant":
        raise NotImplementedError
    value = kwargs.get("constant_values", 0)
    # `value` must be a python scalar for torch.nn.functional.pad
    typ = _dtypes_impl.python_type_for_torch(array.dtype)
    value = typ(value)

    pad_width = torch.broadcast_to(pad_width, (array.ndim, 2))
    pad_width = torch.flip(pad_width, (0,)).flatten()

    return torch.nn.functional.pad(array, tuple(pad_width), value=value)
