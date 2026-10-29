# mypy: ignore-errors

# [MODULE DOC] Defines the ndarray wrapper class around torch.Tensor and the core array constructors for torch._numpy.
# Contents: Flags (read-only flag view), the ndarray class (methods and operator dunders generated at class creation from
# the methods, dunder and ri_dunder tables via _funcs and _ufuncs), NumPy-compatible indexing helpers, and array,
# asarray, ascontiguousarray, from_dlpack, can_cast, result_type. Import-time side effect: builds ndarray methods by
# looking up functions in _funcs and _ufuncs (so those modules must be fully loaded first).
from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Sequence

import torch

from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
    ArrayLike,
    normalize_array_like,
    normalizer,
    NotImplementedType,
)


newaxis = None

FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}


class Flags:
    # [CLASS DOC] Minimal stand-in for ndarray.flags: a read-only mapping of flag names to values.
    # State: _flag_to_value dict holding only the flags the owner computed. Flags may be read by upper-case name,
    # one-letter or two-letter shorthand, or lower-case attribute; modification is not supported.
    def __init__(self, flag_to_value: dict):
        # [FUNCTION DOC] Store the flag-value mapping. Raises AssertionError if any key is not a known flag name.
        invalid_keys = [k for k in flag_to_value if k not in FLAGS]
        if invalid_keys:
            raise AssertionError(f"Invalid flag keys: {invalid_keys}")
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        # [FUNCTION DOC] Attribute access for lower-case flag names (e.g. c_contiguous): returns the value for the upper-cased flag. Raises AttributeError for anything else.
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        # [FUNCTION DOC] Look up a flag by name or shorthand. Raises KeyError for an unknown key and NotImplementedError for a known flag that was not provided.
        if key in SHORTHAND_TO_FLAGS:
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr, value):
        # [FUNCTION DOC] Setting a lower-case flag attribute is routed to item assignment (which always raises); other attributes are set normally.
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, value):
        # [FUNCTION DOC] Flags cannot be modified: raises NotImplementedError for known flag keys and KeyError for unknown ones.
        if key in FLAGS or key in SHORTHAND_TO_FLAGS:
            raise NotImplementedError("Modifying flags is not implemented")
        else:
            raise KeyError(f"No flag key '{key}'")


def create_method(fn, name=None):
    # [FUNCTION DOC] HELPER: Wrap fn in a plain function named name (default the function's own name) with qualified name ndarray.<name>, so it can be installed as an ndarray method.
    name = name or fn.__name__

    def f(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Forward all positional and keyword arguments to fn.
        return fn(*args, **kwargs)

    f.__name__ = name
    f.__qualname__ = f"ndarray.{name}"
    return f


# Map ndarray.name_method -> np.name_func
# If name_func == None, it means that name_method == name_func
methods = {
    "clip": None,
    "nonzero": None,
    "repeat": None,
    "round": None,
    "squeeze": None,
    "swapaxes": None,
    "ravel": None,
    # linalg
    "diagonal": None,
    "dot": None,
    "trace": None,
    # sorting
    "argsort": None,
    "searchsorted": None,
    # reductions
    "argmax": None,
    "argmin": None,
    "any": None,
    "all": None,
    "max": None,
    "min": None,
    "ptp": None,
    "sum": None,
    "prod": None,
    "mean": None,
    "var": None,
    "std": None,
    # scans
    "cumsum": None,
    "cumprod": None,
    # advanced indexing
    "take": None,
    "choose": None,
}

dunder = {
    "abs": "absolute",
    "invert": None,
    "pos": "positive",
    "neg": "negative",
    "gt": "greater",
    "lt": "less",
    "ge": "greater_equal",
    "le": "less_equal",
}

# dunder methods with right-looking and in-place variants
ri_dunder = {
    "add": None,
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "pow": "power",
    "mod": "remainder",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "left_shift",
    "rshift": "right_shift",
    "matmul": None,
}


def _upcast_int_indices(index):
    # [FUNCTION DOC] HELPER: Recursively convert small integer index tensors (int8, int16, int32, uint8) to int64, descending into tuples; everything else is returned unchanged.
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple(_upcast_int_indices(i) for i in index)
    return index


def _has_advanced_indexing(index):
    # [FUNCTION DOC] HELPER: True if any element of the index tuple is a sequence, a bool, a boolean tensor or a tensor with at least one dimension.
    """Check if there's any advanced indexing"""
    return any(
        isinstance(idx, (Sequence, bool))
        or (isinstance(idx, torch.Tensor) and (idx.dtype == torch.bool or idx.ndim > 0))
        for idx in index
    )


def _numpy_compatible_indexing(index):
    # [FUNCTION DOC] HELPER: Make an index NumPy-compatible when advanced indexing is present. Non-tuple indices are wrapped in a tuple; if any element is advanced, Python ints
    # (not bools) and 0-d integer tensors are turned into one-element lists so they combine like NumPy advanced indices. Returns a tuple.
    """Convert scalar indices to lists when advanced indexing is present for NumPy compatibility."""
    if not isinstance(index, tuple):
        index = (index,)

    # Check if there's any advanced indexing (sequences, booleans, or tensors)
    has_advanced = _has_advanced_indexing(index)

    if not has_advanced:
        return index

    # Convert integer scalar indices to single-element lists when advanced indexing is present
    # Note: Do NOT convert boolean scalars (True/False) as they have special meaning in NumPy
    converted = []
    for idx in index:
        if isinstance(idx, int) and not isinstance(idx, bool):
            # Integer scalars should be converted to lists
            converted.append([idx])
        elif (
            isinstance(idx, torch.Tensor)
            and idx.ndim == 0
            and not torch.is_floating_point(idx)
            and idx.dtype != torch.bool
        ):
            # Zero-dimensional tensors holding integers should be treated the same as integer scalars
            converted.append([idx])
        else:
            # Everything else (booleans, lists, slices, etc.) stays as is
            converted.append(idx)

    return tuple(converted)


def _get_bool_depth(s):
    # [FUNCTION DOC] HELPER: Return a pair (is_bool, depth): whether the value is a bool, boolean tensor or nested sequence of bools, and how many dimensions of the indexed array it consumes.
    """Returns the depth of a boolean sequence/tensor"""
    if isinstance(s, bool):
        return True, 0
    if isinstance(s, torch.Tensor) and s.dtype == torch.bool:
        return True, s.ndim
    if not (isinstance(s, Sequence) and s and s[0] != s):
        return False, 0
    is_bool, depth = _get_bool_depth(s[0])
    return is_bool, depth + 1


def _numpy_empty_ellipsis_patch(index, tensor_ndim):
    # [FUNCTION DOC] HELPER: Work around a NumPy/PyTorch difference for an ellipsis matching zero dimensions between advanced indices.
    # Returns a triple (index, squeeze_fn, unsqueeze_fn). If the index has an ellipsis that consumes no dimensions and advanced indexing on both sides, the ellipsis is replaced
    # by a new axis and the two functions remove/add that axis on the result/value; otherwise the index is returned (as a tuple) with identity functions.
    """
    Patch for NumPy-compatible ellipsis behavior when ellipsis doesn't match any dimensions.

    In NumPy, when an ellipsis (...) doesn't actually match any dimensions of the input array,
    it still acts as a separator between advanced indices. PyTorch doesn't have this behavior.

    This function detects when we have:
    1. Advanced indexing on both sides of an ellipsis
    2. The ellipsis doesn't actually match any dimensions
    """
    if not isinstance(index, tuple):
        index = (index,)

    # Find ellipsis position
    ellipsis_pos = None
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            ellipsis_pos = i
            break

    # If no ellipsis, no patch needed
    if ellipsis_pos is None:
        return index, lambda x: x, lambda x: x

    # Count non-ellipsis dimensions consumed by the index
    consumed_dims = 0
    for idx in index:
        is_bool, depth = _get_bool_depth(idx)
        if is_bool:
            consumed_dims += depth
        elif idx is Ellipsis or idx is None:
            continue
        else:
            consumed_dims += 1

    # Calculate how many dimensions the ellipsis should match
    ellipsis_dims = tensor_ndim - consumed_dims

    # Check if ellipsis doesn't match any dimensions
    if ellipsis_dims == 0:
        # Check if we have advanced indexing on both sides of ellipsis
        left_advanced = _has_advanced_indexing(index[:ellipsis_pos])
        right_advanced = _has_advanced_indexing(index[ellipsis_pos + 1 :])

        if left_advanced and right_advanced:
            # This is the case where NumPy and PyTorch differ
            # We need to ensure the advanced indices are treated as separated
            new_index = index[:ellipsis_pos] + (None,) + index[ellipsis_pos + 1 :]
            end_ndims = 1 + sum(
                1 for idx in index[ellipsis_pos + 1 :] if isinstance(idx, slice)
            )

            def squeeze_fn(x):
                # [FUNCTION DOC] HELPER: Remove the inserted axis (counted from the end) from the result.
                return x.squeeze(-end_ndims)

            def unsqueeze_fn(x):
                # [FUNCTION DOC] HELPER: Insert the axis (counted from the end) into a tensor value having enough dimensions; other values are returned unchanged.
                if isinstance(x, torch.Tensor) and x.ndim >= end_ndims:
                    return x.unsqueeze(-end_ndims)
                return x

            return new_index, squeeze_fn, unsqueeze_fn

    return index, lambda x: x, lambda x: x


# Used to indicate that a parameter is unspecified (as opposed to explicitly
# `None`)
class _Unspecified:
    # [CLASS DOC] HELPER: Sentinel type; a single instance (the unspecified attribute) marks a parameter that was not passed, as distinct from an explicit None.
    pass


_Unspecified.unspecified = _Unspecified()

###############################################################
#                      ndarray class                          #
###############################################################


class ndarray:
    # [CLASS DOC] NumPy-like array wrapping a torch.Tensor.
    # State: the tensor attribute holds the data. Methods are mostly thin forwards: at class creation, the functions named in methods are installed from _funcs,
    # dunder and ri_dunder operators (plain, reflected and in-place variants; in-place uses out=self) and conj/conjugate/divmod from _ufuncs.
    # Also defines properties (shape, size, ndim, dtype, strides, itemsize, flags, data, nbytes, T, real, imag, flat), conversion methods, NumPy-compatible indexing, iteration and DLPack support.
    def __init__(self, t=None):
        # [FUNCTION DOC] Create an ndarray from a tensor (default: an empty tensor). Raises ValueError for any other argument type; use array or asarray instead.
        if t is None:
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            self.tensor = t
        else:
            raise ValueError(
                "ndarray constructor is not recommended; prefer"
                "either array(...) or zeros/empty(...)"
            )

    # Register NumPy functions as methods
    for method, name in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)

    # Regular methods but coming from ufuncs
    conj = create_method(_ufuncs.conjugate, "conj")
    conjugate = create_method(_ufuncs.conjugate)

    for method, name in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f"__{method}__"
        vars()[method] = create_method(fn, method)

    for method, name in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f"__{method}__"
        vars()[plain] = create_method(fn, plain)
        rvar = f"__r{method}__"
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f"__i{method}__"
        vars()[ivar] = create_method(
            lambda self, other, fn=fn: fn(self, other, out=self), ivar
        )

    # There's no __idivmod__
    __divmod__ = create_method(_ufuncs.divmod, "__divmod__")
    __rdivmod__ = create_method(
        lambda self, other: _ufuncs.divmod(other, self), "__rdivmod__"
    )

    # prevent loop variables leaking into the ndarray class namespace
    del ivar, rvar, name, plain, fn, method

    @property
    def shape(self):
        # [FUNCTION DOC] Property: shape as a tuple of ints.
        return tuple(self.tensor.shape)

    @property
    def size(self):
        # [FUNCTION DOC] Property: total number of elements.
        return self.tensor.numel()

    @property
    def ndim(self):
        # [FUNCTION DOC] Property: number of dimensions.
        return self.tensor.ndim

    @property
    def dtype(self):
        # [FUNCTION DOC] Property: the dtype as a torch._numpy dtype wrapper.
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self):
        # [FUNCTION DOC] Property: strides in bytes (torch element strides times element size).
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self):
        # [FUNCTION DOC] Property: size in bytes of one element.
        return self.tensor.element_size()

    @property
    def flags(self):
        # [FUNCTION DOC] Property: a Flags object with C_CONTIGUOUS, F_CONTIGUOUS (contiguity of the transpose), OWNDATA (the tensor is not a view) and WRITEABLE (always true).
        # Note contiguous in torch is assumed C-style
        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": self.T.tensor.is_contiguous(),
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # pytorch does not have readonly tensors
            }
        )

    @property
    def data(self):
        # [FUNCTION DOC] Property: the address of the underlying data (an int), not a buffer object as in NumPy.
        return self.tensor.data_ptr()

    @property
    def nbytes(self):
        # [FUNCTION DOC] Property: byte size of the tensor's whole underlying storage.
        # KNOWN ISSUE: for views this is the size of the entire storage, not of the viewed elements as NumPy reports.
        return self.tensor.storage().nbytes()

    @property
    def T(self):
        # [FUNCTION DOC] Property: the transposed array (all axes reversed).
        return self.transpose()

    @property
    def real(self):
        # [FUNCTION DOC] Property: real part as an ndarray.
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        # [FUNCTION DOC] Setter: assign the array-like value to the real part of the tensor in place.
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        # [FUNCTION DOC] Property: imaginary part as an ndarray.
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        # [FUNCTION DOC] Setter: assign the array-like value to the imaginary part of the tensor in place.
        self.tensor.imag = asarray(value).tensor

    @property
    def flat(self):
        # [FUNCTION DOC] Property: the flattened array (same as ravel; NumPy returns an iterator object).
        return self.ravel()

    # ctors
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        # [FUNCTION DOC] Return a copy converted to dtype. Only the defaults order K, casting unsafe, subok true and copy true are supported; any other value raises NotImplementedError.
        if order != "K":
            raise NotImplementedError(f"astype(..., order={order} is not implemented.")
        if casting != "unsafe":
            raise NotImplementedError(
                f"astype(..., casting={casting} is not implemented."
            )
        if not subok:
            raise NotImplementedError(f"astype(..., subok={subok} is not implemented.")
        if not copy:
            raise NotImplementedError(f"astype(..., copy={copy} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType = "C"):
        # [FUNCTION DOC] Return a copy of the array (normalized wrapper; order is unsupported unless default).
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType = "C"):
        # [FUNCTION DOC] Return the array flattened to one dimension (order unsupported unless default).
        return torch.flatten(self)

    def resize(self, *new_shape, refcheck=False):
        # [FUNCTION DOC] Resize the array in place to new_shape (accepts a tuple or separate ints; no-op for no or None shape). Unlike NumPy, new elements are zero-filled, not repeated copies.
        # Raises NotImplementedError if refcheck is true, ValueError for negative sizes. Mutates self.tensor via resize_; growing requires a contiguous tensor.
        # NB: differs from np.resize: fills with zeros instead of making repeated copies of input.
        if refcheck:
            raise NotImplementedError(
                f"resize(..., refcheck={refcheck} is not implemented."
            )
        if new_shape in [(), (None,)]:
            return

        # support both x.resize((2, 2)) and x.resize(2, 2)
        if len(new_shape) == 1:
            new_shape = new_shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)

        if builtins.any(x < 0 for x in new_shape):
            raise ValueError("all elements of `new_shape` must be non-negative")

        new_numel, old_numel = math.prod(new_shape), self.tensor.numel()

        self.tensor.resize_(new_shape)

        if new_numel >= old_numel:
            # zero-fill new elements
            if not self.tensor.is_contiguous():
                raise AssertionError("tensor must be contiguous for resize with growth")
            b = self.tensor.flatten()  # does not copy
            b[old_numel:].zero_()

    def view(self, dtype=_Unspecified.unspecified, type=_Unspecified.unspecified):
        # [FUNCTION DOC] Return a view with the given dtype (default: current dtype) as a new ndarray. The type argument is unsupported and raises NotImplementedError.
        if dtype is _Unspecified.unspecified:
            dtype = self.dtype
        if type is not _Unspecified.unspecified:
            raise NotImplementedError(f"view(..., type={type} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        tview = self.tensor.view(torch_dtype)
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike):
        # [FUNCTION DOC] Fill the array in place with a scalar or 0-d array value. Returns None.
        # Both Pytorch and NumPy accept 0D arrays/tensors and scalars, and
        # error out on D > 0 arrays
        self.tensor.fill_(value)

    def tolist(self):
        # [FUNCTION DOC] Return the data as nested Python lists.
        return self.tensor.tolist()

    def __iter__(self):
        # [FUNCTION DOC] Iterate over the first axis, yielding each sub-tensor wrapped in an ndarray.
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self):
        # [FUNCTION DOC] String form: the tensor's string with "tensor" replaced by torch.ndarray and the dtype prefix torch. removed.
        return (
            str(self.tensor)
            .replace("tensor", "torch.ndarray")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = create_method(__str__)

    def __eq__(self, other):
        # [FUNCTION DOC] Elementwise equality via the equal ufunc. If other cannot be converted (RuntimeError or TypeError), return an all-False boolean array of the same shape.
        try:
            return _ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        # [FUNCTION DOC] Elementwise inequality, computed as the logical inversion of equality.
        return ~(self == other)

    def __index__(self):
        # [FUNCTION DOC] Convert a single-element integer array to a Python int for use as an index. Raises TypeError for anything not convertible.
        try:
            return operator.index(self.tensor.item())
        except Exception as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            ) from exc

    def __bool__(self):
        # [FUNCTION DOC] Truth value of a single-element array (delegates to tensor bool conversion).
        return bool(self.tensor)

    def __int__(self):
        # [FUNCTION DOC] Convert a single-element array to a Python int.
        return int(self.tensor)

    def __float__(self):
        # [FUNCTION DOC] Convert a single-element array to a Python float.
        return float(self.tensor)

    def __complex__(self):
        # [FUNCTION DOC] Convert a single-element array to a Python complex.
        return complex(self.tensor)

    def is_integer(self):
        # [FUNCTION DOC] True if the array holds a single element whose value is integral; False on any failure (including multi-element arrays).
        try:
            v = self.tensor.item()
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self):
        # [FUNCTION DOC] Size of the first axis. Raises an error (IndexError) for 0-d arrays.
        return self.tensor.shape[0]

    def __contains__(self, x):
        # [FUNCTION DOC] Membership test delegated to the tensor.
        return self.tensor.__contains__(x)

    def transpose(self, *axes):
        # [FUNCTION DOC] Transpose with the axes given as separate arguments (no axes reverses all axes).
        # np.transpose(arr, axis=None) but arr.transpose(*axes)
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order="C"):
        # [FUNCTION DOC] Reshape with the shape given as a tuple or separate ints; order is passed through to the reshape function.
        # arr.reshape(shape) and arr.reshape(*shape)
        return _funcs.reshape(self, shape, order=order)

    def sort(self, axis=-1, kind=None, order=None):
        # [FUNCTION DOC] Sort in place along axis: computes a sorted copy and copies it into self. Returns None.
        # ndarray.sort works in-place
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args):
        # [FUNCTION DOC] Return the element: with no arguments the Python scalar of a one-element array; with one argument a flat index; with several a multi-index.
        # KNOWN ISSUE: the flat-index and multi-index forms return an ndarray (via indexing), not a Python scalar as NumPy does.
        # Mimic NumPy's implementation with three special cases (no arguments,
        # a flat index and a multi-index):
        # https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/methods.c#L702
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            # int argument
            return self.ravel()[args[0]]
        else:
            return self.__getitem__(args)

    def __getitem__(self, index):
        # [FUNCTION DOC] Index the array with NumPy semantics and return an ndarray. Negative-step slices are emulated by flipping the tensor along that axis
        # and adjusting the slice; ndarray indices become tensors, small integer index tensors are upcast to int64, scalars are adapted when advanced indexing is present,
        # and the empty-ellipsis difference is patched on the result.
        # KNOWN ISSUE: the negative-step slice adjustment tests truthiness of start and stop, so a zero or negative bound is mishandled (zero is treated as unspecified).
        tensor = self.tensor

        def neg_step(i, s):
            # [FUNCTION DOC] HELPER: For a negative-step slice at axis i, flip the enclosing tensor along that axis (rebinding it) and return an equivalent positive-direction slice; other index items are returned unchanged.
            # Raises AssertionError if start or stop is not an int or None.
            if not (isinstance(s, slice) and s.step is not None and s.step < 0):
                return s

            nonlocal tensor
            tensor = torch.flip(tensor, (i,))

            # Account for the fact that a slice includes the start but not the end
            if not (isinstance(s.start, int) or s.start is None):
                raise AssertionError(
                    f"slice start must be int or None, got {type(s.start).__name__}"
                )
            if not (isinstance(s.stop, int) or s.stop is None):
                raise AssertionError(
                    f"slice stop must be int or None, got {type(s.stop).__name__}"
                )
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None

            return slice(start, stop, -s.step)

        if isinstance(index, Sequence):
            index = type(index)(neg_step(i, s) for i, s in enumerate(index))
        else:
            index = neg_step(0, index)
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, maybe_squeeze, _ = _numpy_empty_ellipsis_patch(index, tensor.ndim)
        return maybe_squeeze(ndarray(tensor.__getitem__(index)))

    def __setitem__(self, index, value):
        # [FUNCTION DOC] Assign value into the indexed region in place. Index is normalized as in __getitem__; non-scalar values are converted to a tensor and cast to the array dtype.
        # Returns whatever the tensor assignment returns (None).
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, _, maybe_unsqueeze = _numpy_empty_ellipsis_patch(index, self.tensor.ndim)

        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            value = _util.cast_if_needed(value, self.tensor.dtype)

        return self.tensor.__setitem__(index, maybe_unsqueeze(value))

    take = _funcs.take
    put = _funcs.put

    def __dlpack__(self, *, stream=None):
        # [FUNCTION DOC] DLPack export of the underlying tensor; stream is forwarded.
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        # [FUNCTION DOC] DLPack device tuple of the underlying tensor.
        return self.tensor.__dlpack_device__()


def _tolist(obj):
    # [FUNCTION DOC] HELPER: Recursively convert nested lists/tuples containing ndarrays into nested lists, replacing each ndarray by its tensor's list form.
    """Recursively convert tensors into lists."""
    a1 = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    # [FUNCTION DOC] Create an ndarray from obj (list, tuple, scalar, tensor or ndarray), optionally with dtype, copy flag and minimum number of dimensions.
    # Unsupported: subok true, like given, order other than K (raise NotImplementedError). If obj is already an ndarray and copy is False with no dtype and enough dimensions, it is returned as is.
    # Lists/tuples of tensors are stacked; other sequences are converted via nested lists. Returns a new ndarray.
    if subok is not False:
        raise NotImplementedError("'subok' parameter is not supported.")
    if like is not None:
        raise NotImplementedError("'like' parameter is not supported.")
    if order != "K":
        raise NotImplementedError

    # a happy path
    if (
        isinstance(obj, ndarray)
        and copy is False
        and dtype is None
        and ndmin <= obj.ndim
    ):
        return obj

    if isinstance(obj, (list, tuple)):
        # FIXME and they have the same dtype, device, etc
        if obj and all(isinstance(x, torch.Tensor) for x in obj):
            # list of arrays: *under torch.Dynamo* these are FakeTensors
            obj = torch.stack(obj)
        else:
            # XXX: remove tolist
            # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
            obj = _tolist(obj)

    # is obj an ndarray already?
    if isinstance(obj, ndarray):
        obj = obj.tensor

    # is a specific dtype requested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)


def asarray(a, dtype=None, order="K", *, like=None):
    # [FUNCTION DOC] Convert a to an ndarray without copying when possible (array with copy False and ndmin 0).
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


def ascontiguousarray(a, dtype=None, *, like=None):
    # [FUNCTION DOC] Convert a to an ndarray and ensure it is contiguous; non-contiguous arrays get a contiguous tensor assigned in place on the (possibly shared) result.
    arr = asarray(a, dtype=dtype, like=like)
    if not arr.tensor.is_contiguous():
        arr.tensor = arr.tensor.contiguous()
    return arr


def from_dlpack(x, /):
    # [FUNCTION DOC] Create an ndarray from an object supporting DLPack (shares memory).
    t = torch.from_dlpack(x)
    return ndarray(t)


def _extract_dtype(entry):
    # [FUNCTION DOC] HELPER: Get a dtype wrapper from a dtype-like; if that fails, treat the entry as an array-like and use its dtype.
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty


def can_cast(from_, to, casting="safe"):
    # [FUNCTION DOC] Whether from_ (dtype or array-like) can be cast to to (dtype or array-like) under the casting rule (default safe). Returns a bool.
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)

    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)


def result_type(*arrays_and_dtypes):
    # [FUNCTION DOC] Compute the promoted dtype of arrays and dtypes. Each entry is converted to a tensor if possible, otherwise treated as a dtype and represented by an empty one-element tensor. Returns a torch._numpy dtype wrapper.
    tensors = []
    for entry in arrays_and_dtypes:
        try:
            t = asarray(entry).tensor
        except (RuntimeError, ValueError, TypeError):
            dty = _dtypes.dtype(entry)
            t = torch.empty(1, dtype=dty.torch_dtype)
        tensors.append(t)

    torch_dtype = _dtypes_impl.result_type_impl(*tensors)
    return _dtypes.dtype(torch_dtype)
