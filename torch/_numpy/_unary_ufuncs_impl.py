# mypy: ignore-errors
# [MODULE DOC] Plain-torch work functions for numpy unary ufuncs. Re-exports torch
# functions under numpy names (aliases such as invert, fix, rint, conjugate, fabs,
# radians, degrees) and defines the few that torch lacks (cbrt, positive, absolute),
# plus aliases abs and conj. The ufunc wrapper module enumerates the public names
# here. No import-time side effects besides defining names.

"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""

import torch
from torch import (  # noqa: F401
    absolute as fabs,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    bitwise_not,
    bitwise_not as invert,
    ceil,
    conj_physical as conjugate,
    cos,
    cosh,
    deg2rad,
    deg2rad as radians,
    exp,
    exp2,
    expm1,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    log10,
    log1p,
    log2,
    logical_not,
    negative,
    rad2deg,
    rad2deg as degrees,
    reciprocal,
    round as fix,
    round as rint,
    sign,
    signbit,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)


# special cases: torch does not export these names
def cbrt(x):
    # [FUNCTION DOC] Cube root computed as the elementwise power one third. KNOWN ISSUE: a
    # power of one third yields NaN for negative inputs, unlike a true cube root.
    return torch.pow(x, 1 / 3)


def positive(x):
    # [FUNCTION DOC] Unary plus: returns the operand's unary-plus result (same values).
    return +x


def absolute(x):
    # [FUNCTION DOC] Elementwise absolute value; bool tensors are returned unchanged
    # because torch absolute is not implemented for bool.
    # work around torch.absolute not impl for bools
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)


# TODO set __name__ and __qualname__
abs = absolute
conj = conjugate
