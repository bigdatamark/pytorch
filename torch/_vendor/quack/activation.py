# [MODULE DOC] Device-side (CuTe DSL) activation functions and their backward passes for float32 values or packed pairs of
# float32 values: sigmoid, relu, relu squared, gelu (tanh approximation), softplus, silu, glu, reglu, geglu, swiglu and the
# gpt-oss swiglu variant, with tanh-based fast variants. Backward functions generally return the input gradient(s)
# together with the recomputed forward output. Ends with four lookup maps from activation name to function
# (act_fn_map, dact_fn_map, gate_fn_map, dgate_fn_map). Import-time side effects: defines the packed subtract helper.
# Copyright (c) 2025, Tri Dao.

import math
from typing import Tuple
from functools import partial

import cutlass.cute as cute
from cutlass import Float32, Boolean, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm


F32_or_F32x2 = Float32 | Tuple[Float32, Float32]


sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
)


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    # [FUNCTION DOC] Hardware approximate tanh of a float32 scalar via inline PTX assembly (no side effects). Returns Float32.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def sigmoid_tanh(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] Sigmoid computed as 0.5 plus 0.5 times tanh of half the input, so only one tanh is needed.
    # Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    if const_expr(not isinstance(x, tuple)):
        # return 0.5 + 0.5 * cute.math.tanh(0.5 * x, fastmath=True)
        return 0.5 + 0.5 * tanh(0.5 * x)
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return cute.arch.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))


@dsl_user_op
def sigmoid(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] Sigmoid computed as the approximate reciprocal of one plus exp of the negated input (the packed path uses exp2 with a
    # log2(e) scale). Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.rcp_approx(1.0 + cute.math.exp(-x, fastmath=True), loc=loc, ip=ip)
    else:
        log2_e = math.log2(math.e)
        neg_x = cute.arch.mul_packed_f32x2(x, (-log2_e, -log2_e))
        exp_neg_x = (
            cute.math.exp2(neg_x[0], fastmath=True),
            cute.math.exp2(neg_x[1], fastmath=True),
        )
        denom = cute.arch.add_packed_f32x2(exp_neg_x, (1.0, 1.0))
        return cute.arch.rcp_approx(denom[0]), cute.arch.rcp_approx(denom[1])


@dsl_user_op
def dsigmoid_from_output(out: Float32, dout: Float32, *, loc=None, ip=None) -> Float32:
    # [FUNCTION DOC] Sigmoid gradient from the forward output: dout times out times (1 minus out), computed as dout times (out minus out squared).
    # return dout * out * (1.0 - out)
    return dout * (out - out * out)


@dsl_user_op
def relu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] max(x, 0). Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0))
    else:
        return cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0))


@dsl_user_op
@cute.jit
def drelu(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] ReLU backward. Returns (dx, relu(x)) where dx equals dout where x is positive and zero elsewhere. Operates on scalars or
    # pairs; inputs x and dout must have the same form.
    if const_expr(not isinstance(x, tuple)):
        x_pos = Boolean(x > 0)
        return dout if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))
    else:
        x0_pos = Boolean(x[0] > 0)
        x1_pos = Boolean(x[1] > 0)
        dx = (dout[0] if x0_pos else Float32(0.0), dout[1] if x1_pos else Float32(0.0))
        return dx, relu(x)


@dsl_user_op
def relu_sq(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] Relu squared forward: max(x, 0) times x. Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * x
    else:
        relu_x = (cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0)))
        return cute.arch.mul_packed_f32x2(relu_x, x)


@dsl_user_op
@cute.jit
def drelu_sq(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] Relu squared backward. Returns (dx, relu_sq_out) where dx is 2 times dout times relu(x) (zero for non-positive x).
    # Scalar or pair form.
    """
    ReLU squared backward pass: computes gradient w.r.t. x and recomputes forward
    Given: relu_sq_out = max(x, 0) * x, and dout = grad w.r.t. relu_sq_out
    Returns: (dx, relu_sq_out) where:
    - dx = dout * 2 * x if x > 0, else 0
    - relu_sq_out = max(x, 0) * x
    """
    if const_expr(not isinstance(x, tuple)):
        relu_x = relu(x)
        relu_sq_out = relu_x * x
        # Derivative: d/dx[max(x,0) * x] = 2*x if x > 0, else 0
        dx = 2.0 * (dout * relu_x)
        return dx, relu_sq_out
    else:
        relu_x = relu(x)
        relu_sq_out = cute.arch.mul_packed_f32x2(relu_x, x)
        dx = cute.arch.mul_packed_f32x2((2.0, 2.0), cute.arch.mul_packed_f32x2(dout, relu_x))
        return dx, relu_sq_out


@dsl_user_op
def gelu_tanh_approx(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] GELU with the tanh approximation, using a factored polynomial form of the inner argument and the fast tanh.
    # Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x * x)))
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
    if const_expr(not isinstance(x, tuple)):
        return 0.5 * (
            x
            # Currently cute.math.tanh(x, fastmath=True) generates very slow code
            # * (1 + cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)), fastmath=True))
            * (1.0 + tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x))))
        )
    else:
        x_sq = cute.arch.mul_packed_f32x2(x, x)
        x_sq_scaled = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        z = cute.arch.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        x_tanh_z = cute.arch.fma_packed_f32x2(tanh_z, x, x)
        return cute.arch.mul_packed_f32x2((0.5, 0.5), x_tanh_z)


@dsl_user_op
def dgelu_tanh_approx(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] GELU (tanh approximation) backward. Returns (dx, gelu_out): the derivative via the chain rule through the tanh
    # argument, multiplied by dout, plus the recomputed forward value. Scalar or pair form.
    """
    GELU tanh approximation backward pass: computes gradient w.r.t. x and recomputes forward
    Given: gelu_out = 0.5 * x * (1 + tanh(x * (c1 + c2 * x^2))), and dout = grad w.r.t. gelu_out
    Returns: (dx, gelu_out)

    Derivative uses the chain rule:
    d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    where z = x * (c1 + c2 * x^2), dz/dx = c1 + 3 * c2 * x^2
    and sech^2(z) = 1 - tanh^2(z)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # c1 ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # c2 ~0.0356774
    sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff  # c3 ~0.01070322

    if const_expr(not isinstance(x, tuple)):
        # Compute z = x * (c1 + c2 * x^2)
        x_sq = x * x
        # tanh_z = cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq), fastmath=True)
        tanh_z = tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq))
        half_tanh_z_plus_one = 0.5 + 0.5 * tanh_z
        gelu_out = x * half_tanh_z_plus_one

        # Compute gradient
        # sech^2(z) = 1 - tanh^2(z)
        sech2_z = 1 - tanh_z * tanh_z
        # dz/dx = c1 + 3 * c2 * x^2
        dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
        # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
        dgelu = half_tanh_z_plus_one + x * (0.5 * (sech2_z * dz_dx))

        dx = dout * dgelu
        return dx, gelu_out
    else:
        # Compute z = x * (c1 + c2 * x^2)
        x_sq = cute.arch.mul_packed_f32x2(x, x)
        x_sq_scaled = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        z = cute.arch.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        half_tanh_z_plus_one = cute.arch.fma_packed_f32x2(tanh_z, (0.5, 0.5), (0.5, 0.5))
        gelu_out = cute.arch.mul_packed_f32x2(x, half_tanh_z_plus_one)

        # Compute gradient
        # sech^2(z) = 1 - tanh^2(z)
        sech2_z = cute.arch.fma_packed_f32x2(tanh_z, (-tanh_z[0], -tanh_z[1]), (1.0, 1.0))
        # dz/dx = c1 + 3 * c2 * x^2
        dz_dx = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff_3, sqrt_2_over_pi_coeff_3), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
        sech2_dz_dx = cute.arch.mul_packed_f32x2(sech2_z, dz_dx)
        x_sech2_dz_dx = cute.arch.mul_packed_f32x2(x, sech2_dz_dx)
        dgelu = cute.arch.fma_packed_f32x2(x_sech2_dz_dx, (0.5, 0.5), half_tanh_z_plus_one)

        dx = cute.arch.mul_packed_f32x2(dout, dgelu)
        return dx, gelu_out


@dsl_user_op
@cute.jit
def softplus(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] Softplus log(1 + exp(x)) using fast math, returning x unchanged where x exceeds 20 to avoid overflow. Scalar or pair form.
    if const_expr(not isinstance(x, tuple)):
        use_linear = Boolean(x > 20.0)
        return (
            cute.math.log(Float32(cute.math.exp(x, fastmath=True)) + 1.0, fastmath=True)
            if not use_linear
            else x
        )
    else:
        log2_e = math.log2(math.e)
        x_log2e = cute.arch.mul_packed_f32x2(x, (log2_e, log2_e))
        x_exp = (cute.math.exp(x_log2e[0], fastmath=True), cute.math.exp(x_log2e[1], fastmath=True))
        x_exp_p1 = cute.arch.add_packed_f32x2(x_exp, (1.0, 1.0))
        log_x_exp_p1 = (
            cute.math.log2(x_exp_p1[0], fastmath=True),
            cute.math.log2(x_exp_p1[1], fastmath=True),
        )
        ln2 = math.log(2.0)
        softplus_x = cute.arch.mul_packed_f32x2(log_x_exp_p1, (ln2, ln2))
        use_linear_0 = Boolean(x[0] > 20.0)
        use_linear_1 = Boolean(x[1] > 20.0)
        return (
            softplus_x[0] if not use_linear_0 else x[0],
            softplus_x[1] if not use_linear_1 else x[1],
        )


@dsl_user_op
@cute.jit
def dsoftplus_from_output(out: Float32, dout: Float32, *, loc=None, ip=None) -> Float32:
    # [FUNCTION DOC] Softplus backward from the forward output: dout times (1 minus exp(-out)), or dout unchanged where out exceeds 20.
    use_linear = Boolean(out > 20.0)
    # dx = dout * (1.0 - cute.math.exp(-out, fastmath=True)) if not use_linear else dout
    dx = dout - dout * cute.math.exp(-out, fastmath=True)
    return dx if not use_linear else dout


@dsl_user_op
def silu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] SiLU: x times sigmoid(x). Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    """
    silu(x) = x * sigmoid(x) = x * rcp(1 + exp(-x)).
    """
    if const_expr(not isinstance(x, tuple)):
        return x * sigmoid(x)
    else:
        return cute.arch.mul_packed_f32x2(x, sigmoid(x))


@dsl_user_op
def silu_tanh(x: F32_or_F32x2, *, already_halved: bool = False, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] SiLU via the tanh identity (half-x times tanh of half-x plus half-x). If already_halved is true the input is taken to be
    # already multiplied by 0.5. Operates on a single float32 value or a pair (tuple of two) for packed two-lane arithmetic.
    """
    silu(x) = x * sigmoid(x) = x * (1 + tanh(x / 2)) / 2 = (0.5 * x) * tanh(0.5 * x) + (0.5 * x)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * x, MUFU.TANH, and FFMA.
    """
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x if const_expr(not already_halved) else x
        # return x_half * cute.math.tanh(x_half, fastmath=True) + x_half
        return x_half * tanh(x_half) + x_half
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x) if const_expr(not already_halved) else x
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return cute.arch.fma_packed_f32x2(x_half, tanh_x_half, x_half)


@dsl_user_op
def dsilu(
    x: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] SiLU backward. Returns (d_silu(x) times dout, silu(x)) computed with fused multiply-add friendly expansion. Scalar or pair form.
    """
    SiLU backward pass: computes d_silu(x) * dout and recomputes silu(x).

    d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x + silu_x) * dout
        return d_silu_x_dout, silu_x
    else:
        sigmoid_x = sigmoid(x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x = cute.arch.add_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, silu_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x, dout
        )
        return d_silu_x_dout, silu_x


@dsl_user_op
def dsilu_tanh(
    x: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    already_halved: bool = False,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] SiLU backward using the tanh-based sigmoid; already_halved indicates x is pre-multiplied by 0.5.
    # Returns (d_silu(x) times dout, silu(x)). Scalar or pair form.
    """
    SiLU backward using sigmoid(x) = 0.5 * (1 + tanh(0.5 * x)).
    """
    if const_expr(not isinstance(x, tuple)):
        if const_expr(not already_halved):
            x_half = 0.5 * x
            tanh_x_half = tanh(x_half)
            sigmoid_x = 0.5 * tanh_x_half + 0.5
            silu_x = x_half * tanh_x_half + x_half
        else:
            tanh_x = tanh(x)
            sigmoid_x = 0.5 * tanh_x + 0.5
            silu_x = x * tanh_x + x
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x + silu_x) * dout
        return d_silu_x_dout, silu_x
    else:
        if const_expr(not already_halved):
            x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
            tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x_half, tanh_x_half, x_half)
        else:
            tanh_x = (tanh(x[0]), tanh(x[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x, tanh_x, x)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x = cute.arch.add_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, silu_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x, dout
        )
        return d_silu_x_dout, silu_x


@dsl_user_op
def swiglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] SwiGLU forward: silu(x) times y (x is the gate, y the up projection). Scalar or pair form.
    if const_expr(not isinstance(x, tuple)):
        return silu(x) * y
    else:
        return cute.arch.mul_packed_f32x2(silu(x), y)


@dsl_user_op
def swiglu_tanh(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] SwiGLU forward using the tanh-based silu. Scalar or pair form.
    if const_expr(not isinstance(x, tuple)):
        return silu_tanh(x) * y
    else:
        return cute.arch.mul_packed_f32x2(silu_tanh(x), y)


@dsl_user_op
def dswiglu(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] SwiGLU backward. Returns (dx, dy, swiglu_out) with dx = dout times y times d_silu(x), dy = dout times silu(x),
    # and swiglu_out = silu(x) times y, arranged to minimize instruction count. Scalar or pair form.
    """
    SwiGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: swiglu_out = silu(x) * y, and dout = grad w.r.t. swiglu_out
    Returns: (dx, dy, swiglu_out) where dx = dout * y * d_silu(x), dy = dout * silu(x)

    d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    This has been optimized to use fewer instructions (i.e. we expand things out
    to use FFMA instead of FADD and FMUL).
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x  # FMUL
        silu_x_dout = silu_x * dout  # FMUL
        #   d_silu(x) * dout
        # = sigmoid_x * (1 + x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + sigmoid_x * x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + silu_x * (1 - sigmoid_x)) * dout
        # = (sigmoid_x + silu_x - silu_x * sigmoid_x) * dout
        # = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout  # FFMA, FFMA
        dx = d_silu_x_dout * y  # FMUL
        dy = silu_x_dout
        swiglu_out = silu_x * y  # FMUL
        return dx, dy, swiglu_out
    else:
        # Compute sigmoid(x) and silu(x)
        sigmoid_x = sigmoid(x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        # d_silu(x) * dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        d_silu_x_dout = cute.arch.fma_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, dout, silu_x_dout
        )
        dx = cute.arch.mul_packed_f32x2(d_silu_x_dout, y)
        dy = silu_x_dout
        swiglu_out = cute.arch.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out


@dsl_user_op
def dswiglu_tanh(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    already_halved: bool = False,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] SwiGLU backward using the tanh-based sigmoid; already_halved indicates x is pre-multiplied by 0.5.
    # Returns (dx, dy, swiglu_out). Scalar or pair form.
    """
    SwiGLU backward using sigmoid(x) = 0.5 * (1 + tanh(0.5 * x)).
    """
    if const_expr(not isinstance(x, tuple)):
        if const_expr(not already_halved):
            sigmoid_x = sigmoid_tanh(x)
            silu_x = x * sigmoid_x  # FMUL
        else:
            tanh_x = tanh(x)
            sigmoid_x = 0.5 * tanh_x + 0.5
            silu_x = x * tanh_x + x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout
        dx = d_silu_x_dout * y
        dy = silu_x_dout
        swiglu_out = silu_x * y
        # Overall it's 1 MUFU.TANH, 5 FMUL, 3 FFMA
        return dx, dy, swiglu_out
    else:
        if const_expr(not already_halved):
            sigmoid_x = sigmoid_tanh(x)
            silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        else:
            tanh_x = (tanh(x[0]), tanh(x[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x, tanh_x, x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        # d_silu(x) * dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        d_silu_x_dout = cute.arch.fma_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, dout, silu_x_dout
        )
        dx = cute.arch.mul_packed_f32x2(d_silu_x_dout, y)
        dy = silu_x_dout
        swiglu_out = cute.arch.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out


@dsl_user_op
def swiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    # [FUNCTION DOC] gpt-oss SwiGLU variant: x times sigmoid(alpha times x) times (y + 1), default alpha 1.702. Scalar or pair form.
    """The swiglu variant used in gpt-oss, which has a scaling factor on x and bias of 1 to y.
    https://github.com/openai/gpt-oss/blob/7be9334950053a888e24887a57dac797a17d6e00/gpt_oss/torch/model.py#L249
    x * sigmoid(alpha * x) * (y + 1)
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_alpha_x = sigmoid(alpha * x)
        silu_x = x * sigmoid_alpha_x
        return silu_x * y + silu_x
    else:
        alpha_x = cute.arch.mul_packed_f32x2((alpha, alpha), x)
        sigmoid_alpha_x = sigmoid(alpha_x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        return cute.arch.fma_packed_f32x2(silu_x, y, silu_x)


@dsl_user_op
def swiglu_oai_tanh(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    # [FUNCTION DOC] Tanh-based version of swiglu_oai kept for instruction-level and accuracy comparison. Scalar or pair form.
    """Tanh-based swiglu_oai kept for SASS/accuracy comparison."""
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        silu_x = x_half * tanh(alpha * x_half) + x_half
        return silu_x * y + silu_x
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
        alpha_x_half = cute.arch.mul_packed_f32x2((alpha, alpha), x_half)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        silu_x = cute.arch.fma_packed_f32x2(x_half, tanh_alpha_x_half, x_half)
        return cute.arch.fma_packed_f32x2(silu_x, y, silu_x)


@dsl_user_op
def dswiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] Backward of swiglu_oai. Returns (dx, dy, swiglu_oai_out) given x, y, dout and alpha. Scalar or pair form.
    """
    Swiglu OAI backward pass: computes gradients w.r.t. x and y
    Given: swiglu_oai_out = x * sigmoid(alpha * x) * (y + 1), and dout = grad w.r.t. swiglu_oai_out
    Returns: (dx, dy, swiglu_oai_out)

    Derivative of x * sigmoid(alpha * x) w.r.t. x:
    d/dx[x * sigmoid(alpha * x)] = sigmoid(alpha * x) + alpha * x * sigmoid(alpha * x) * (1 - sigmoid(alpha * x))
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_alpha_x = sigmoid(alpha * x)
        silu_x = x * sigmoid_alpha_x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * dout
        dx = d_silu_x_dout * y + d_silu_x_dout
        dy = silu_x_dout
        swiglu_out = silu_x * y + silu_x
        return dx, dy, swiglu_out
    else:
        alpha_x = cute.arch.mul_packed_f32x2((alpha, alpha), x)
        sigmoid_alpha_x = sigmoid(alpha_x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        silu_x_minus_product = cute.arch.fma_packed_f32x2(
            silu_x, (-sigmoid_alpha_x[0], -sigmoid_alpha_x[1]), silu_x
        )
        sigmoid_plus_alpha_diff = cute.arch.fma_packed_f32x2(
            (alpha, alpha), silu_x_minus_product, sigmoid_alpha_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(sigmoid_plus_alpha_diff, dout)
        dx = cute.arch.fma_packed_f32x2(d_silu_x_dout, y, d_silu_x_dout)
        dy = silu_x_dout
        swiglu_out = cute.arch.fma_packed_f32x2(silu_x, y, silu_x)
        return dx, dy, swiglu_out


@dsl_user_op
def dswiglu_oai_tanh(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] Tanh-based backward of swiglu_oai kept for comparison. Returns (dx, dy, swiglu_oai_out). Scalar or pair form.
    """Tanh-based dswiglu_oai kept for SASS/accuracy comparison."""
    if const_expr(not isinstance(x, tuple)):
        alpha_x_half = (0.5 * alpha) * x
        sigmoid_alpha_x = 0.5 + 0.5 * tanh(alpha_x_half)
        silu_x = x * sigmoid_alpha_x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * dout
        dx = d_silu_x_dout * y + d_silu_x_dout
        dy = silu_x_dout
        swiglu_out = silu_x * y + silu_x
        return dx, dy, swiglu_out
    else:
        alpha_x_half = cute.arch.mul_packed_f32x2(((0.5 * alpha), (0.5 * alpha)), x)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        sigmoid_alpha_x = cute.arch.fma_packed_f32x2(tanh_alpha_x_half, (0.5, 0.5), (0.5, 0.5))
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        silu_x_minus_product = cute.arch.fma_packed_f32x2(
            silu_x, (-sigmoid_alpha_x[0], -sigmoid_alpha_x[1]), silu_x
        )
        sigmoid_plus_alpha_diff = cute.arch.fma_packed_f32x2(
            (alpha, alpha), silu_x_minus_product, sigmoid_alpha_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(sigmoid_plus_alpha_diff, dout)
        dx = cute.arch.fma_packed_f32x2(d_silu_x_dout, y, d_silu_x_dout)
        dy = silu_x_dout
        swiglu_out = cute.arch.fma_packed_f32x2(silu_x, y, silu_x)
        return dx, dy, swiglu_out


@dsl_user_op
def glu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] Gated linear unit: sigmoid(x) times y. Scalar or pair form.
    """GLU: Gated Linear Unit
    glu(x, y) = sigmoid(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        return sigmoid_x * y  # FMUL
    else:
        sigmoid_x = sigmoid(x)
        return cute.arch.mul_packed_f32x2(sigmoid_x, y)


@dsl_user_op
def dglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] GLU backward. Returns (dx, dy, glu_out) with dy = dout times sigmoid(x) and dx = (y minus glu_out) times dy.
    # Scalar or pair form (the pair path uses the packed subtract).
    # KNOWN ISSUE: an inline comment mentions a tanh instruction in the total instruction count, but sigmoid here uses exp and reciprocal.
    """
    GLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: glu_out = sigmoid(x) * y, and dout = grad w.r.t. glu_out
    Returns: (dx, dy, glu_out) where:
    - dx = dout * y * sigmoid(x) * (1 - sigmoid(x))
    - dy = dout * sigmoid(x)
    - glu_out = sigmoid(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        sigmoid_x_dout = sigmoid_x * dout  # FMUL
        glu_out = sigmoid_x * y  # FMUL
        # dx = y * sigmoid(x) * (1 - sigmoid(x)) * dout
        #    = y * (1 - sigmoid(x)) * sigmoid_x_dout
        #    = (y - y * sigmoid(x)) * sigmoid_x_dout
        #    = (y - glu_out) * sigmoid_x_dout
        dx = (y - glu_out) * sigmoid_x_dout  # FADD, FMUL
        dy = sigmoid_x_dout
        # Total: 1 MUFU.TANH, 4 FMUL, 1 FADD, 1 FFMA
        return dx, dy, glu_out
    else:
        sigmoid_x = sigmoid(x)
        sigmoid_x_dout = cute.arch.mul_packed_f32x2(sigmoid_x, dout)
        glu_out = cute.arch.mul_packed_f32x2(sigmoid_x, y)
        # dx = (y - glu_out) * sigmoid_x_dout
        y_minus_glu_out = sub_packed_f32x2(y, glu_out)
        dx = cute.arch.mul_packed_f32x2(y_minus_glu_out, sigmoid_x_dout)
        dy = sigmoid_x_dout
        return dx, dy, glu_out


@dsl_user_op
def reglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] ReGLU forward: max(x, 0) times y. Scalar or pair form.
    """ReGLU: ReLU Gated Linear Unit
    reglu(x, y) = relu(x) * y = max(x, 0) * y
    """
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * y
    else:
        relu_x = relu(x)
        return cute.arch.mul_packed_f32x2(relu_x, y)


@dsl_user_op
@cute.jit
def dreglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] ReGLU backward. Returns (dx, dy, reglu_out): dx is dout times y where x is positive else zero, dy is dout times relu(x),
    # reglu_out is relu(x) times y. Scalar or pair form.
    """
    ReGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: reglu_out = relu(x) * y, and dout = grad w.r.t. reglu_out
    Returns: (dx, dy, reglu_out) where:
    - dx = dout * y if x > 0, else 0
    - dy = dout * relu(x)
    - reglu_out = relu(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        x_pos = Boolean(x > 0)
        relu_x = cute.arch.fmax(x, Float32(0.0))
        dx = (dout * y) if x_pos else Float32(0.0)
        dy = dout * relu_x
        reglu_out = relu_x * y
        return dx, dy, reglu_out
    else:
        x0_pos = Boolean(x[0] > 0)
        x1_pos = Boolean(x[1] > 0)
        relu_x = relu(x)
        dout_y = cute.arch.mul_packed_f32x2(dout, y)
        dx = ((dout_y[0] if x0_pos else Float32(0.0)), (dout_y[1] if x1_pos else Float32(0.0)))
        dy = cute.arch.mul_packed_f32x2(dout, relu_x)
        reglu_out = cute.arch.mul_packed_f32x2(relu_x, y)
        return dx, dy, reglu_out


@dsl_user_op
def geglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    # [FUNCTION DOC] GeGLU forward: tanh-approximate gelu(x) times y. Scalar or pair form.
    """GeGLU: GELU Gated Linear Unit
    geglu(x, y) = gelu(x) * y
    Uses the tanh approximation of GELU
    """
    if const_expr(not isinstance(x, tuple)):
        return gelu_tanh_approx(x) * y
    else:
        return cute.arch.mul_packed_f32x2(gelu_tanh_approx(x), y)


@dsl_user_op
def dgeglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    # [FUNCTION DOC] GeGLU backward built on dgelu_tanh_approx. Returns (dx, dy, geglu_out) with dx = d_gelu(x)*dout*y, dy = gelu(x)*dout,
    # geglu_out = gelu(x)*y. Scalar or pair form.
    """
    GeGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: geglu_out = gelu(x) * y, and dout = grad w.r.t. geglu_out
    Returns: (dx, dy, geglu_out) where:
    - dx = dout * y * d_gelu(x)
    - dy = dout * gelu(x)
    - geglu_out = gelu(x) * y
    """
    if const_expr(not isinstance(x, tuple)):
        # Reuse dgelu_tanh_approx to compute d_gelu(x) * dout and gelu(x)
        dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
        # Compute gradients for geglu
        dx = dgelu_x_dout * y
        dy = gelu_x * dout
        geglu_out = gelu_x * y
        return dx, dy, geglu_out
    else:
        # Reuse dgelu_tanh_approx to compute d_gelu(x) * dout and gelu(x)
        dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
        # Compute gradients for geglu
        dx = cute.arch.mul_packed_f32x2(dgelu_x_dout, y)
        dy = cute.arch.mul_packed_f32x2(gelu_x, dout)
        geglu_out = cute.arch.mul_packed_f32x2(gelu_x, y)
        return dx, dy, geglu_out


# ============================================================================
# Activation name -> function maps
# ============================================================================

act_fn_map = {
    None: None,
    "silu": silu,
    "silu-tanh": silu_tanh,
    "relu": relu,
    "relu_sq": relu_sq,
    "gelu_tanh_approx": gelu_tanh_approx,
}

dact_fn_map = {
    None: None,
    "silu": dsilu,
    "silu-tanh": dsilu_tanh,
    "relu": drelu,
    "relu_sq": drelu_sq,
    "gelu_tanh_approx": dgelu_tanh_approx,
}

gate_fn_map = {
    "swiglu": swiglu,
    "swiglu-tanh": swiglu_tanh,
    "swiglu_oai": swiglu_oai,
    "swiglu_oai-tanh": swiglu_oai_tanh,
    "reglu": reglu,
    "geglu": geglu,
    "glu": glu,
}

dgate_fn_map = {
    "swiglu": dswiglu,
    "swiglu-tanh": dswiglu_tanh,
    "swiglu_oai": dswiglu_oai,
    "swiglu_oai-tanh": dswiglu_oai_tanh,
    "reglu": dreglu,
    "geglu": dgeglu,
    "glu": dglu,
}
