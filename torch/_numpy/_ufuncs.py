# mypy: ignore-errors
# [MODULE DOC] Public-layer numpy ufuncs built on torch. Wraps the plain-torch
# implementations in the binary and unary ufunc implementation modules with argument
# normalization, dtype/casting handling and the out argument, and special-cases
# matmul, ldexp, divmod and modf. Import-time side effects: for every public name in
# the implementation modules a wrapped function is created and attached to this
# module's namespace; also builds the lists of binary and unary names and sets the
# export list to their concatenation.

from __future__ import annotations

import torch

from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util
from ._normalizations import (
    ArrayLike,
    ArrayLikeOrScalar,
    CastingModes,
    DTypeLike,
    normalizer,
    NotImplementedType,
    OutArray,
)


def _ufunc_postprocess(result, out, casting):
    # [FUNCTION DOC] HELPER: Finish a ufunc call. If out is given, casts the result to
    # out's torch dtype under the casting rule and broadcasts it to out's shape; otherwise
    # returns the result unchanged. Does not copy into out (the normalizer handles that).
    if out is not None:
        result = _util.typecast_tensor(result, out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result


# ############# Binary ufuncs ######################

_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul", "divmod", "ldexp"]
]


NEP50_FUNCS = (
    "add",
    "subtract",
    "multiply",
    "floor_divide",
    "true_divide",
    "divide",
    "remainder",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "hypot",
    "arctan2",
    "logaddexp",
    "logaddexp2",
    "heaviside",
    "copysign",
    "fmax",
    "minimum",
    "fmin",
    "maximum",
    "fmod",
    "gcd",
    "lcm",
    "pow",
)


def deco_binary_ufunc(torch_func):
    # [FUNCTION DOC] Build a numpy-style binary ufunc around a torch function. The wrapped
    # function (normalized arguments) takes two array-or-scalar operands plus out,
    # casting, dtype and several accepted-but-unsupported numpy options. Returns the
    # wrapped function, renamed after the torch function.
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x1: ArrayLikeOrScalar,
        x2: ArrayLikeOrScalar,
        /,
        out: OutArray | None = None,
        *,
        where: NotImplementedType = True,
        casting: CastingModes | None = "same_kind",
        order: NotImplementedType = "K",
        dtype: DTypeLike | None = None,
        subok: NotImplementedType = False,
        signature: NotImplementedType = None,
        extobj: NotImplementedType = None,
    ):
        # [FUNCTION DOC] HELPER: The ufunc entry point. With dtype given, both operands are
        # cast to it; with two tensors, they are cast to their common result type under the
        # casting rule; otherwise scalars and tensors are converted following NEP 50 promotion
        # (applied only to names in the NEP50 list). Calls the torch function and post-
        # processes with out and casting. Raises errors from casting violations.
        if dtype is not None:

            def cast(x, dtype):
            # [FUNCTION DOC] HELPER: Cast one operand to dtype: tensors via checked typecast under
            # the casting rule, other values by conversion to a tensor of that dtype.
                if isinstance(x, torch.Tensor):
                    return _util.typecast_tensor(x, dtype, casting)
                else:
                    return torch.as_tensor(x, dtype=dtype)

            x1 = cast(x1, dtype)
            x2 = cast(x2, dtype)
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            dtype = _dtypes_impl.result_type_impl(x1, x2)
            x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)
        else:
            x1, x2 = _dtypes_impl.nep50_to_tensors(
                x1, x2, torch_func.__name__ in NEP50_FUNCS, torch_func.__name__
            )

        result = torch_func(x1, x2)

        return _ufunc_postprocess(result, out, casting)

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


# matmul's signature is _slightly_ different from other ufuncs:
# - no where=...
# - additional axis=..., axes=...
# - no NEP50 scalars in or out
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: OutArray | None = None,
    *,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
    axes: NotImplementedType = None,
    axis: NotImplementedType = None,
):
    # [FUNCTION DOC] Matrix product ufunc. Unlike other ufuncs it has no where argument
    # and no NEP 50 scalar handling. Computes the common result type (or uses dtype),
    # casts both operands under the casting rule, multiplies, and post-processes with out.
    # axes, axis and similar options are accepted but unsupported.
    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    result = _binary_ufuncs_impl.matmul(x1, x2)

    result = _ufunc_postprocess(result, out, casting)
    return result


# ldexp casting is special : the dtype of the result == dtype of the 1st arg
@normalizer
def ldexp(
    x1: ArrayLikeOrScalar,
    x2: ArrayLikeOrScalar,
    /,
    out: OutArray | None = None,
    *,
    where: NotImplementedType = True,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
):
    # [FUNCTION DOC] x1 * 2**x2 ufunc. Result dtype follows the first argument: with dtype
    # given the first operand is cast to it, otherwise a non-tensor first operand is
    # converted and integers become float. Raises ValueError if the second argument is not
    # integer typed. A float16 result, which torch upcasts, is converted back to float16.
    if dtype is not None:
        if isinstance(x1, torch.Tensor):
            x1 = _util.typecast_tensor(x1, dtype, casting)
        else:
            x1 = torch.as_tensor(x1, dtype=dtype)
    else:
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1)
            x1 = _util.cast_int_to_float(x1)

    x2 = torch.as_tensor(x2)
    # the second arg must be integer
    if _dtypes_impl._category(x2.dtype) != 1:
        raise ValueError("ldexp 2nd arg must be integer")

    result = _binary_ufuncs_impl.ldexp(x1, x2)

    if x1.dtype == torch.float16:
        # torch.ldexp(f16, int) -> f32, undo it
        result = result.to(torch.float16)

    return _ufunc_postprocess(result, out, casting)


# nin=2, nout=2
@normalizer
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: OutArray | None = None,
    out2: OutArray | None = None,
    /,
    out: tuple[OutArray | None, OutArray | None] = (None, None),
    *,
    where: NotImplementedType = True,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
):
    # [FUNCTION DOC] Floor division and modulus returning the pair (quotient, remainder).
    # Output arrays may be given as out1 and out2 positionally or as an out pair, but not
    # mixed: raises ValueError if only one positional out is supplied, and TypeError if
    # both positional and keyword forms are used. Operands are cast to the common result
    # type (or dtype) under casting; each result is post-processed against its out.
    # make sure we either have no out arrays at all, or there is either
    # out1, out2, or out=tuple, but not both
    num_outs = sum(x is not None for x in [out1, out2])
    if num_outs == 1:
        raise ValueError("both out1 and out2 need to be provided")
    elif num_outs == 2:
        o1, o2 = out
        if o1 is not None or o2 is not None:
            raise TypeError(
                "cannot specify 'out' as both a positional and keyword argument"
            )
    else:
        out1, out2 = out

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    quot, rem = _binary_ufuncs_impl.divmod(x1, x2)

    quot = _ufunc_postprocess(quot, out1, casting)
    rem = _ufunc_postprocess(rem, out2, casting)
    return quot, rem


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    vars()[name] = deco_binary_ufunc(ufunc)


def modf(x, /, *args, **kwds):
    # [FUNCTION DOC] Split x into fractional and integral parts, implemented via divmod by
    # one. Returns the pair (remainder, quotient), i.e. fractional part first. Extra
    # arguments are forwarded to divmod.
    quot, rem = divmod(x, 1, *args, **kwds)
    return rem, quot


_binary = _binary + ["divmod", "modf", "matmul", "ldexp"]


# ############# Unary ufuncs ######################


_unary = [
    name
    for name in dir(_unary_ufuncs_impl)
    if not name.startswith("_") and name != "torch"
]


# these are ufunc(int) -> float
_fp_unary = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log1p",
    "log2",
    "rad2deg",
    "radians",
    "reciprocal",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]


def deco_unary_ufunc(torch_func):
    # [FUNCTION DOC] Build a numpy-style unary ufunc around a torch function. The wrapped
    # function takes one array plus out, casting, dtype and accepted-but-unsupported numpy
    # options. Returns the wrapped function renamed after the torch function.
    """Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x: ArrayLike,
        /,
        out: OutArray | None = None,
        *,
        where=True,
        casting: CastingModes | None = "same_kind",
        order="K",
        dtype: DTypeLike | None = None,
        subok: NotImplementedType = False,
        signature=None,
        extobj=None,
    ):
        # [FUNCTION DOC] HELPER: The unary ufunc entry point. Optionally casts the input to
        # dtype under the casting rule; for functions in the integer-to-float list, integer
        # input is converted to float first. Calls the torch function and post-processes with
        # out and casting.
        if dtype is not None:
            x = _util.typecast_tensor(x, dtype, casting)

        if torch_func.__name__ in _fp_unary:
            x = _util.cast_int_to_float(x)

        result = torch_func(x)
        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    vars()[name] = deco_unary_ufunc(ufunc)


__all__ = _binary + _unary  # noqa: PLE0605
