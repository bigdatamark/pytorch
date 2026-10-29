# mypy: allow-untyped-defs
# [PACKAGE DOC]
# Reference implementations of torch.special functions (Bessel functions, entr,
# erfcx, expit, i0e/i1/i1e, log_ndtr, logit, ndtr, ndtri, softmax variants,
# xlog1py, multigammaln, zeta), built from prims and other refs with type
# promotion wrappers; many are registered as aten decompositions on import.
import math
from typing import Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
    _make_alias,
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)


__all__ = [
    "bessel_j0",
    "bessel_j1",
    "entr",
    "erfcx",
    "expit",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "logit",
    "log_softmax",
    "multigammaln",
    "ndtr",
    "ndtri",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "zeta",
]
aten = torch._ops.ops.aten


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j0(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise Bessel function of the first kind, order 0, via prim; ints promote to float.
    return prims.bessel_j0(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j1(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise Bessel function of the first kind, order 1, via prim; ints promote to float.
    return prims.bessel_j1(a)


@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def entr(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise entropy term: NaN stays NaN, negative x gives negative infinity,
    # zero gives 0, positive gives minus x times log x.
    return torch.where(
        torch.isnan(a),
        a,
        torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)),
    )


@register_decomposition(aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def erfcx(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise scaled complementary error function via prim.
    return prims.erfcx(a)


# alias for sigmoid
expit = _make_alias(torch.sigmoid, "expit")


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i0e(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise exponentially scaled modified Bessel function, order 0, via prim.
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise modified Bessel function of the first kind, order 1, via prim.
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1e(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise exponentially scaled modified Bessel function, order 1, via prim.
    return prims.bessel_i1e(a)


@register_decomposition(aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def log_ndtr(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise log of the standard normal CDF. Uses erfcx for inputs below 1
    # (numerically stable for large negatives) and log1p of minus half erfc otherwise.
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    t = a * M_SQRT1_2
    return torch.where(
        a < 1.0,
        torch.log(torch.special.erfcx(-t) / 2) - t * t,
        torch.log1p(-torch.erfc(t) / 2),
    )


@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: float | None = None) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise log of x/(1-x). If eps is given, input is first clamped to
    # [eps, 1-eps]; if eps is None, eps is -1.0, which makes the clamp bounds -1 and 2
    # (effectively unclamped for typical inputs).
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.where(self < lo, lo, torch.where(self > hi, hi, self))
    return torch.log(torch.true_divide(self, torch.sub(1, self)))


@register_decomposition(aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlog1py(a: TensorLikeType | NumberType, b: TensorLikeType | NumberType):
    # [FUNCTION DOC] Elementwise a times log1p(b), defined as 0 where a is 0, and NaN where b is NaN.
    # At least one argument must be a tensor (error otherwise); a scalar partner is
    # converted to a scalar tensor of the other's dtype and device.
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(a, TensorLike) and isinstance(b, Number):
        # pyrefly: ignore [bad-argument-type]
        b = refs.scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        # pyrefly: ignore [bad-argument-type]
        a = refs.scalar_tensor(a, dtype=b.dtype, device=b.device)

    # mypy: expected "Tensor"
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log1p(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


@register_decomposition(aten.mvlgamma)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType:
    # [FUNCTION DOC] Multivariate log-gamma of dimension p: sum of lgamma over a shifted by
    # half-integer offsets plus the constant p(p-1)/4 times log pi. Does not validate
    # that a exceeds the (p-1)/2 domain bound.
    c = 0.25 * p * (p - 1) * math.log(math.pi)
    b = 0.5 * torch.arange(start=(1 - p), end=1, step=1, dtype=a.dtype, device=a.device)
    return torch.sum(torch.lgamma(a.unsqueeze(-1) + b), dim=-1) + c


@register_decomposition(aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtr(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise standard normal CDF: half of one plus erf(x / sqrt 2).
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    a_sqrt_2 = a * M_SQRT1_2
    return (1 + torch.erf(a_sqrt_2)) * 0.5


@register_decomposition(aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtri(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise inverse of the standard normal CDF via prim.
    return prims.ndtri(a)


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # [FUNCTION DOC] Forwarding alias of torch.log_softmax over dim with optional dtype; no out argument.
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: torch.dtype | None = None,
) -> TensorLikeType:
    # [FUNCTION DOC] Forwarding alias of torch.softmax over dim with optional dtype; no out argument.
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise spherical Bessel function of the first kind, order 0, via prim.
    return prims.spherical_bessel_j0(a)


# TODO: add docstring
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def zeta(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # [FUNCTION DOC] Elementwise Hurwitz zeta function of a and b via prim; binary op with int to float promotion.
    return prims.zeta(a, b)
