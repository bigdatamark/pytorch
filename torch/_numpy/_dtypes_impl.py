# mypy: ignore-errors

# [MODULE DOC] Dtype and scalar-type helpers for torch._numpy, operating purely on torch dtypes.
# Provides the lazily initialised NumPy-style default dtypes, cast/promotion lookups
# backed by _casting_dicts, and NEP 50 weak-scalar promotion for binary operations.
# Import-time side effect: imports _casting_dicts midway through the module.
"""Dtypes/scalar type implementations with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""

from collections import namedtuple

import torch


# defaults : mimic NumPy, allow user control
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# a global state
# We set it the first time we call default_dtypes() to avoid importing
# torch._dynamo.config and create a circular reference
_default_dtypes = None


def default_dtypes():
    # [FUNCTION DOC] Return the cached DefaultDTypes (float, complex, int torch dtypes).
    # On first call reads torch._dynamo.config numpy_default_float/complex/int, resolves each
    # name as an attribute of torch, validates they are torch dtypes, and stores them in a
    # module-global cache. Raises AssertionError if a configured name is not a torch dtype
    # and AttributeError if the name does not exist in torch. Later config changes are ignored.
    global _default_dtypes
    if _default_dtypes is None:
        import torch._dynamo.config as config

        _default_dtypes = DefaultDTypes(
            float_dtype=getattr(torch, config.numpy_default_float),
            complex_dtype=getattr(torch, config.numpy_default_complex),
            int_dtype=getattr(torch, config.numpy_default_int),
        )
        if not isinstance(_default_dtypes.float_dtype, torch.dtype):
            raise AssertionError(
                f"float_dtype must be a torch.dtype, got {type(_default_dtypes.float_dtype)}"
            )
        if not isinstance(_default_dtypes.complex_dtype, torch.dtype):
            raise AssertionError(
                f"complex_dtype must be a torch.dtype, got {type(_default_dtypes.complex_dtype)}"
            )
        if not isinstance(_default_dtypes.int_dtype, torch.dtype):
            raise AssertionError(
                f"int_dtype must be a torch.dtype, got {type(_default_dtypes.int_dtype)}"
            )
    return _default_dtypes


def get_default_dtype_for(dtype):
    # [FUNCTION DOC] Map a torch dtype to the default dtype of its category: bool stays bool,
    # complex -> default complex, floating -> default float, anything else -> default int.
    """Default scalar type given sctype category."""
    if dtype == torch.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes().complex_dtype
    if dtype.is_floating_point:
        return default_dtypes().float_dtype
    # else, it must be (some) integer
    return default_dtypes().int_dtype


from . import _casting_dicts as _cd


def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    # [FUNCTION DOC] Look up whether a cast from one torch dtype to another is allowed under the
    # given casting rule (key into the cast table, e.g. no/equiv/safe/same_kind/unsafe). Returns a bool;
    # raises KeyError for unknown rule or dtype.
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(*tensors):
    # [FUNCTION DOC] Compute the promoted torch dtype of one or more tensors by folding the
    # pairwise result-type table over their dtypes. A single tensor returns its own dtype.
    # Raises IndexError if called with no tensors and KeyError for dtypes missing in the table.
    # NB: torch dtypes here
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp

    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


def python_type_for_torch(dtyp):
    # [FUNCTION DOC] Return the Python scalar type (float, complex, bool or int) matching a torch dtype.
    """Get a python scalar type a torch dtype"""
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == torch.bool:
        typ = bool
    else:
        typ = int
    return typ


# ### NEP 50 helpers ###

_SCALAR_TYPES = (int, bool, float, complex)

_SCALAR_AND_SYMBOLIC_TYPES = (
    *_SCALAR_TYPES,
    torch.SymInt,
    torch.SymFloat,
    torch.SymBool,
)

_NEP50_FUNCS_TENSOR_ONLY = (
    "minimum",
    "maximum",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "gcd",
    "hypot",
    "heaviside",
    "fmod",
    "fmin",
    "fmax",
    "copysign",
    "arctan2",
)


def is_scalar(x):
    # [FUNCTION DOC] HELPER: True if x is a Python int, bool, float or complex.
    return isinstance(x, _SCALAR_TYPES)


def is_scalar_or_symbolic(x):
    # [FUNCTION DOC] HELPER: True if x is a Python scalar or a torch symbolic int/float/bool.
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)


def _dtype_for_scalar(py_type):
    # [FUNCTION DOC] HELPER: Map a Python (or symbolic) scalar type to its NumPy-like torch dtype
    # (bool, int64, float64, complex128). Raises KeyError for other types.
    return {
        bool: torch.bool,
        torch.SymBool: torch.bool,
        int: torch.int64,
        torch.SymInt: torch.int64,
        float: torch.float64,
        torch.SymFloat: torch.float64,
        complex: torch.complex128,
    }[py_type]


def _dtype_for_scalar_or_tensor(x):
    # [FUNCTION DOC] HELPER: Dtype of a tensor, or the dtype implied by the type of a Python scalar.
    return x.dtype if isinstance(x, torch.Tensor) else _dtype_for_scalar(type(x))


def is_float_or_fp_tensor(x):
    # [FUNCTION DOC] True if the tensor dtype, or the dtype implied by the scalar type, is floating point.
    return _dtype_for_scalar_or_tensor(x).is_floating_point


def is_complex_or_complex_tensor(x):
    # [FUNCTION DOC] True if the tensor dtype, or the dtype implied by the scalar type, is complex.
    return _dtype_for_scalar_or_tensor(x).is_complex


def _category(dtype):
    # [FUNCTION DOC] HELPER: Rank of a dtype's kind: 0 bool, 1 integer, 2 floating, 3 complex.
    # Raises KeyError for dtypes not in the table (e.g. bfloat16).
    return {
        torch.bool: 0,
        torch.SymBool: 0,
        # int
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 1,
        torch.int32: 1,
        torch.int64: 1,
        torch.SymInt: 1,
        # float
        torch.float16: 2,
        torch.float32: 2,
        torch.float64: 2,
        torch.SymFloat: 2,
        # complex
        torch.complex64: 3,
        torch.complex128: 3,
    }[dtype]


def nep50_to_tensors(x1, x2, handle_weaks, function_name):
    # [FUNCTION DOC] Convert two binary-op operands into tensors following NEP 50 weak-scalar promotion.
    # If handle_weaks is false or both are non-tensors, scalars are converted to 0-d tensors of the
    # default dtype for their kind. Otherwise exactly one operand is a Python scalar ("weak"): it adopts
    # the tensor's dtype when its kind is not higher than the tensor's, else the scalar's default dtype
    # (float scalar with float32 tensor and complex scalar gives complex64). A Python int out of range of
    # an integer tensor dtype raises OverflowError. The weak scalar stays a plain Python value unless its
    # dtype changes or function_name is in the tensor-only list. Returns the pair in original order.
    """If either of inputs is a python scalar, type-promote with NEP 50."""

    def to_tensor(scalar, dtype=None):
        # [FUNCTION DOC] HELPER: Make a 0-d tensor from a scalar, using the given dtype or else the default
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return torch.as_tensor(scalar, dtype=dtype)

    x1_is_weak = not isinstance(x1, torch.Tensor)
    x2_is_weak = not isinstance(x2, torch.Tensor)
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # scalar <op> tensor: NEP 50
    if x1_is_weak == x2_is_weak:
        raise AssertionError(
            f"Expected exactly one weak type, got x1_is_weak={x1_is_weak}, x2_is_weak={x2_is_weak}"
        )

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

    # find the dtype for the weak's type
    weak_dtype = _dtype_for_scalar(type(weak))

    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # special-case complex + float32
    if weak_dtype.is_complex and not_weak.dtype == torch.float32:
        dt = torch.complex64

    # detect overflows: in PyTorch, uint8(-1) wraps around to 255,
    # while NEP50 mandates an exception.
    #
    # Note that we only check if each element of the binop overflows,
    # not the result. Consider, e.g. `uint8(100) + 200`. Operands are OK
    # in uint8, but the result overflows and wrap around 255.
    # Numpy emits a RuntimeWarning, PyTorch does not, and we do not either.
    if cat_weak == 1 and cat_not_weak == 1:
        # integers
        iinfo = torch.iinfo(not_weak.dtype)
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )
    if weak_dtype != dt or function_name in _NEP50_FUNCS_TENSOR_ONLY:
        # finally, can make `weak` into a 0D tensor, if both parameters are required to be tensor.
        weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
