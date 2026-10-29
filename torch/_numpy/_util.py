# [MODULE DOC] Assorted helpers for the torch-backed numpy layer that depend only on
# torch and the standard library: sequence test, numpy-compatible exception classes,
# dtype casting helpers, axis normalization (index, tuple, single-axis), shape
# expansion for keepdims and expand_dims, flattening for axis=None, checked
# typecasting, and coercion of arbitrary objects or ndarray wrappers into tensors. No
# import-time side effects.
"""Assorted utilities, which do not need anything other than torch and stdlib."""

import operator
from collections.abc import Iterable, Sequence, Sized
from typing import TypeGuard

import torch

from . import _dtypes_impl


# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq: object) -> TypeGuard[Sized]:
    # [FUNCTION DOC] True if seq is a sized, non-string object whose length can be taken;
    # False for strings, unsized objects, or if taking the length raises.
    if isinstance(seq, str):
        return False
    if not isinstance(seq, Sized):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


class AxisError(ValueError, IndexError):
    # [CLASS DOC] Error for an out-of-range axis; inherits ValueError and IndexError so
    # either handler catches it, mirroring numpy's AxisError. No added state.
    pass


class UFuncTypeError(TypeError, RuntimeError):
    # [CLASS DOC] Error for ufunc type problems; inherits TypeError and RuntimeError to
    # mirror numpy's UFuncTypeError. No added state.
    pass


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    # [FUNCTION DOC] Return tensor converted to dtype only when dtype is not None and
    # differs from the tensor's dtype; otherwise return the same tensor object.
    # NB: no casting if dtype=None
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def cast_int_to_float(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] Convert bool and integer tensors to the default float dtype; floating
    # and complex tensors are returned unchanged.
    # cast integers and bools to the default float dtype
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x


# a replica of the version in ./numpy/numpy/core/src/multiarray/common.h
def normalize_axis_index(ax: int, ndim: int, argname: str | None = None) -> int:
    # [FUNCTION DOC] Validate a single axis against ndim and return it as a non-negative
    # index (negative values wrap). Raises AxisError if out of range. argname is unused.
    if not (-ndim <= ax < ndim):
        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
    if ax < 0:
        ax += ndim
    return ax


# from https://github.com/numpy/numpy/blob/main/numpy/core/numeric.py#L1378
def normalize_axis_tuple(
    axis: int | Iterable[int],
    ndim: int,
    argname: str | None = None,
    allow_duplicate: bool = False,
) -> tuple[int, ...]:
    # [FUNCTION DOC] Convert an axis argument (int, index-like, or iterable of ints) to a
    # tuple of non-negative axes using normalize_axis_index. Unless allow_duplicate is
    # set, raises ValueError on repeated axes (message mentions argname when given).
    # Raises AxisError for out-of-range values.
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    """
    # Optimization to speed-up the most common cases.
    axes: Iterable[int]
    if isinstance(axis, (tuple, list)):
        axes = axis
    else:
        # operator.index handles the scalar shorthand (int and __index__-like
        # objects); a TypeError means axis is already an iterable of ints.
        try:
            axes = [operator.index(axis)]  # pyrefly: ignore[bad-argument-type]
        except TypeError:
            axes = axis  # pyrefly: ignore[bad-assignment]
    # Going via an iterator directly is slower than via list comprehension.
    result = tuple(normalize_axis_index(ax, ndim, argname) for ax in axes)
    if not allow_duplicate and len(set(map(int, result))) != len(result):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return result


def allow_only_single_axis(axis: int | tuple[int, ...] | None) -> int | None:
    # [FUNCTION DOC] Reduce an axis argument to a single int or None. None and ints pass
    # through; a sequence must have exactly one entry, which is returned; otherwise raises
    # NotImplementedError.
    if axis is None:
        return axis
    if isinstance(axis, int):
        return axis
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    return axis[0]


def expand_shape(
    arr_shape: Sequence[int], axis: int | tuple[int, ...] | list[int]
) -> list[int]:
    # [FUNCTION DOC] Compute the shape obtained by inserting length-one dimensions at the
    # given axis or axes (as numpy expand_dims). Axes are normalized against the output
    # rank; the remaining positions are filled from arr_shape in order. Returns a list of
    # ints; raises on out-of-range or repeated axes.
    # taken from numpy 1.23.x, expand_dims function
    axis_seq = axis if isinstance(axis, (list, tuple)) else (axis,)
    out_ndim = len(axis_seq) + len(arr_shape)
    normalized = normalize_axis_tuple(axis_seq, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in normalized else next(shape_it) for ax in range(out_ndim)]
    return shape


def apply_keepdims(
    tensor: torch.Tensor, axis: int | tuple[int, ...] | None, ndim: int
) -> torch.Tensor:
    # [FUNCTION DOC] Restore reduced dimensions after a reduction. With axis None the
    # (scalar) result is expanded to ndim length-one dimensions and made contiguous;
    # otherwise the result is reshaped to include length-one dimensions at the given axes.
    # Returns the reshaped tensor.
    if axis is None:
        # tensor was a scalar
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


def axis_none_flatten(
    *tensors: torch.Tensor, axis: int | None = None
) -> tuple[tuple[torch.Tensor, ...], int]:
    # [FUNCTION DOC] When axis is None, flatten every given tensor and return them with
    # axis 0; otherwise return the tensors unchanged with the given axis. Returns a pair
    # of (tuple of tensors, axis).
    """Flatten the arrays if axis is None."""
    if axis is None:
        tensors = tuple(ar.flatten() for ar in tensors)
        return tensors, 0
    else:
        return tensors, axis


def typecast_tensor(
    t: torch.Tensor, target_dtype: torch.dtype, casting: str
) -> torch.Tensor:
    # [FUNCTION DOC] Cast a tensor to target_dtype if the casting rule allows it, else
    # raise TypeError describing the disallowed cast. Returns the (possibly same) tensor.
    # KNOWN ISSUE: the docstring says ValueError is raised but the code raises TypeError.
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    can_cast = _dtypes_impl.can_cast_impl

    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array data from {t.dtype} to"
            f" {target_dtype} according to the rule '{casting}'"
        )
    return cast_if_needed(t, target_dtype)


def typecast_tensors(
    tensors: Sequence[torch.Tensor], target_dtype: torch.dtype, casting: str
) -> tuple[torch.Tensor, ...]:
    # [FUNCTION DOC] Apply typecast_tensor to every tensor in a sequence with the same
    # target dtype and casting rule; returns a tuple.
    return tuple(typecast_tensor(t, target_dtype, casting) for t in tensors)


def _try_convert_to_tensor(obj: object) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Convert an object to a tensor via torch.as_tensor; any
    # failure is re-raised as NotImplementedError carrying the original error text.
    try:
        tensor = torch.as_tensor(obj)
    except Exception as e:
        mesg = f"failed to convert {obj} to ndarray. \nInternal error is: {str(e)}."
        raise NotImplementedError(mesg)  # noqa: B904
    return tensor


def _coerce_to_tensor(
    obj: object,
    dtype: torch.dtype | None = None,
    copy: bool = False,
    ndmin: int = 0,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Core of the array-creation function. Tensors pass through;
    # other objects are converted with the default dtype temporarily set to the numpy-
    # compatible float default (restored afterwards even on error). Then casts to dtype if
    # given, prepends length-one dimensions until the rank reaches ndmin, and clones when
    # copy is truthy (a non-boolean copy mode that cannot be converted is treated as
    # false). Raises NotImplementedError if conversion fails. Side effect: temporarily
    # mutates the global default dtype.
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coercive function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        # tensor.dtype is the pytorch default, typically float32. If obj's elements
        # are not exactly representable in float32, we've lost precision:
        # >>> torch.as_tensor(1e12).item() - 1e12
        # -4096.0
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            tensor = _try_convert_to_tensor(obj)
        finally:
            torch.set_default_dtype(default_dtype)

    # type cast if requested
    tensor = cast_if_needed(tensor, dtype)

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)

    # special handling for np._CopyMode
    try:
        copy = bool(copy)
    except ValueError:
        # TODO handle _CopyMode.IF_NEEDED correctly
        copy = False
    # copy if requested
    if copy:
        tensor = tensor.clone()

    return tensor


def ndarrays_to_tensors(*inputs: object) -> object:
    # [FUNCTION DOC] Replace ndarray wrapper objects with their underlying tensors; other
    # values are untouched. A single ndarray gives its tensor; a single tuple is converted
    # element-wise (recursively) into a new tuple; several arguments are treated as one
    # tuple. Imports the ndarray class lazily.
    # KNOWN ISSUE: with no arguments it returns a ValueError instance instead of raising
    #   it.
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    from ._ndarray import ndarray

    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.tensor
        elif isinstance(input_, tuple):
            result = []
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            return tuple(result)
        else:
            return input_
    else:
        if not isinstance(inputs, tuple):
            raise AssertionError(
                f"Expected inputs to be a tuple, got {type(inputs).__name__}"
            )
        return ndarrays_to_tensors(inputs)
