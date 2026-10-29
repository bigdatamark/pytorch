# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC] Helper library imported by generated Inductor Triton kernels and by the Triton launcher code.
# Contains (a) host-side utilities for selecting the Triton driver and backend options, and (b) many @triton.jit
# device functions: integer div/mod/pow, NaN-aware min/max (plain, strict and with index), online softmax combine,
# Welford combine, Philox-based random helpers, bucketize binary search, decoupled-lookback scans, frexp, bitonic
# sort with index, a grid barrier, and inline-asm pack/unpack helpers. Import-time side effects: defines a module
# ContextVar controlling whether driver setup is skipped and a constexpr log2(e) constant.
import math as pymath
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from .triton_compat import (
    _log2,
    builtins_use_semantic_kwarg,
    JITFunction,
    libdevice,
    math,
    tl,
    triton,
)


_T = TypeVar("_T")
_LOG_2_E: tl.constexpr = tl.constexpr(pymath.log2(pymath.e))
_skip_gpu_driver_setup: ContextVar[bool] = ContextVar(
    "_skip_gpu_driver_setup", default=False
)


@contextmanager
def skip_gpu_driver_setup():
    # [FUNCTION DOC] Context manager that makes set_driver_to_gpu a no-op inside its scope.
    # Sets a ContextVar flag to true on entry and resets it to the prior value on exit (even on error); isolated per thread/context.
    # Scoped no-op for set_driver_to_gpu(). ContextVar keeps nested/thread-local
    # uses isolated.
    token = _skip_gpu_driver_setup.set(True)
    try:
        yield
    finally:
        _skip_gpu_driver_setup.reset(token)


def set_driver_to_cpu():
    # [FUNCTION DOC] Make the Triton CPU backend the active driver.
    # No-op if the CPU driver is already active. If no 'cpu' backend is registered, only emits a warning (kernels will not be executable).
    # Side effect: mutates the global triton runtime driver.
    driver = triton.runtime.driver
    if backend := triton.backends.backends.get("cpu", None):
        if isinstance(driver.active, backend.driver):
            # Don't re-initialize backend if it is already active
            return
        driver.set_active(backend.driver())
        return
    # This can be a hard error once triton-cpu is merged into fbcode
    warnings.warn(
        "Could not find an active CPU backend. Generated kernels will not be executable!"
    )


def _is_backend_active(name, backend):
    # [FUNCTION DOC] HELPER: Return whether the given Triton backend (by name) is usable.
    # True if its driver reports active; otherwise for 'nvidia'/'amd' falls back to torch.cuda availability and the HIP/non-HIP build flag
    # (works around Triton failing GPU detection in subprocess workers). False for other backends.
    if backend.driver.is_active():
        return True
    # Triton may fail to detect the GPU in subprocess workers when using
    # ctypes-based driver detection (triton-lang/triton#9578). Fall back
    # to torch's own device checks which are more reliable in these environments.
    if name == "nvidia":
        import torch

        return torch.cuda.is_available() and torch.version.hip is None
    if name == "amd":
        import torch

        return torch.cuda.is_available() and torch.version.hip is not None
    return False


def set_driver_to_gpu():
    # [FUNCTION DOC] Make a non-CPU Triton backend (first active one found) the active driver.
    # Returns immediately if driver setup is currently skipped. Skips re-initialization if the matching driver is already active
    # (including when wrapped in a lazy proxy). Raises RuntimeError if no active GPU backend exists. Side effect: mutates global driver.
    if _skip_gpu_driver_setup.get():
        return

    driver = triton.runtime.driver
    for name, backend in triton.backends.backends.items():
        if _is_backend_active(name, backend) and name != "cpu":
            # After https://github.com/triton-lang/triton/commit/b844d519bc5e86edf00fe6b3c6c2d1badcd509a4,
            # `driver.active` can be of `LazyProxy` type and the sign of this - `_obj` attribute.
            if (
                isinstance(driver.active, backend.driver)
                or hasattr(driver.active, "_obj")
                and isinstance(driver.active._obj, backend.driver)
            ):
                # Don't re-initialize backend if it is already active
                return
            driver.set_active(backend.driver())
            return
    raise RuntimeError("Could not find an active GPU backend")


def get_backend_options_for_target(target, options=None):
    # [FUNCTION DOC] Parse backend options for a compile target.
    # Args: target; options (optional mapping of overrides, copied). Returns the parsed backend options as a plain dict.
    options = {} if options is None else dict(options)
    backend = triton.compiler.compiler.make_backend(target)
    return backend.parse_options(options).__dict__


def get_backend_options():
    # [FUNCTION DOC] Return the parsed backend options for the currently active driver target.
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return get_backend_options_for_target(target)


def _is_concrete_backend_option_value(value: Any) -> bool:
    # [FUNCTION DOC] HELPER: Return False if the value is (or recursively contains, in sequences or dict keys/values) a tensor, symbolic int/float/bool or sympy expression; otherwise True.
    import sympy

    import torch

    if isinstance(
        value,
        (
            torch.Tensor,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            sympy.Expr,
        ),
    ):
        return False
    if isinstance(value, (tuple, list)):
        return all(_is_concrete_backend_option_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            _is_concrete_backend_option_value(key)
            and _is_concrete_backend_option_value(item)
            for key, item in value.items()
        )
    return True


def try_filter_backend_options_for_target(target, options, kernel_arg_names=()):
    # [FUNCTION DOC] Select the entries of options that are valid Triton backend options for the target.
    # Args: target; options mapping; kernel_arg_names (names that are legal kernel parameters).
    # Returns a dict of the options recognized by the backend. Raises RuntimeError if an option is neither a backend option nor a kernel
    # parameter, or if a retained backend option is not a concrete (non-symbolic, non-tensor) value.
    parsed_options = get_backend_options_for_target(target)
    kernel_arg_names = tuple(kernel_arg_names)
    filtered_options = {
        name: value for name, value in options.items() if name in parsed_options
    }
    invalid_options = [
        name
        for name in options
        if name not in parsed_options and name not in kernel_arg_names
    ]
    if invalid_options:
        raise RuntimeError(
            "Triton launch kwargs must be kernel parameters or valid backend options: "
            f"{sorted(invalid_options)!r}."
        )
    dynamic_options = [
        name
        for name, value in filtered_options.items()
        if not _is_concrete_backend_option_value(value)
    ]
    if dynamic_options:
        raise RuntimeError(
            "Triton backend options must be concrete values: "
            f"{sorted(dynamic_options)!r}."
        )
    return filtered_options


def get_constexprs(kernel: JITFunction) -> list[int]:
    # [FUNCTION DOC] Return the positional indices of the constexpr parameters of a JIT kernel.
    return [p.num for p in kernel.params if p.is_constexpr]


@triton.jit
def promote_to_tensor(x):
    # [FUNCTION DOC] Device function: promote a scalar to a tensor by adding a one-element zero int1 tensor (tensors pass through).
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def fp8e4m3fn_to_float32(x):
    # [FUNCTION DOC] Device function: decode float8 e4m3fn bit patterns (input 8-bit values) to float32.
    # Handles sign, normals (exponent rebias), subnormals (exponent zero) and NaN (all exponent and mantissa bits set); there is no infinity.
    x_u32 = x.to(tl.uint32)
    sign = (x_u32 & 0x80) << 24
    exp = (x_u32 >> 3) & 0xF
    mant = x_u32 & 0x7

    normal_bits = sign | ((exp + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    subnormal_abs = mant.to(tl.float32) * 0.001953125
    subnormal_bits = subnormal_abs.to(tl.uint32, bitcast=True) | sign
    subnormal = subnormal_bits.to(tl.float32, bitcast=True)

    nan = (sign | 0x7FF00000).to(tl.float32, bitcast=True)
    result = tl.where(exp == 0, subnormal, normal)
    return tl.where((exp == 0xF) & (mant == 0x7), nan, result)


@triton.jit
def div_floor_integer(a, b):
    # [FUNCTION DOC] Device function: integer floor division of a by b.
    # Corrects the truncating quotient by one when the remainder is nonzero and operand signs differ.
    # NOTE: a // b is C division, but we want floor division
    # Based on c10::div_floor_integer
    quot = a // b
    remainder = a % b
    fixed = tl.where(remainder != 0, quot - 1, quot)
    return tl.where((a < 0) != (b < 0), fixed, quot)


@triton.jit
def remainder_integer(a, b):
    # [FUNCTION DOC] Device function: integer remainder with floor-division (Python) semantics; result takes the sign of the divisor.
    # NOTE: a % b matches C division, not floor division
    remainder = a % b
    return tl.where((remainder != 0) & ((a < 0) != (b < 0)), remainder + b, remainder)


@triton.jit
def pow_integer(base, exponent):
    # [FUNCTION DOC] Device function: exact integer power by repeated squaring over all bits of the exponent (treated as unsigned).
    # Returns a tensor of base dtype; avoids floating-point pow rounding. Negative exponents are not meaningful.
    # Triton has no exact integer pow primitive; use repeated squaring for
    # nonnegative integer exponents so integral scalar pow does not round
    # through libdevice.pow before casting back to int.
    exponent_dtype: tl.constexpr = tl.core.get_int_dtype(
        exponent.dtype.primitive_bitwidth, signed=False
    )
    exp = exponent.to(exponent_dtype)
    result = tl.full(base.shape, 1, base.dtype)
    for _ in tl.static_range(exponent_dtype.primitive_bitwidth):
        result = tl.where((exp & 1) != 0, result * base, result)
        exp = exp >> 1
        base = base * base
    return result


@triton.jit
def is_floating(x):
    # [FUNCTION DOC] Device function: return whether the input (promoted to tensor) has a floating-point dtype (compile-time).
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def _prod_accumulate(a, b):
    # [FUNCTION DOC] HELPER: Device function: binary combine for product reductions (multiply).
    return a * b


@triton.jit
def prod(input, axis):
    # [FUNCTION DOC] Device function: product reduction of input along the given axis.
    return tl.reduce(input, axis, _prod_accumulate)


@triton.jit
def minimum(a, b):
    # [FUNCTION DOC] Device function: elementwise minimum that propagates NaN.
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def maximum(a, b):
    # [FUNCTION DOC] Device function: elementwise maximum that propagates NaN.
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _minimum_reduce(a, b):
    # [FUNCTION DOC] HELPER: Device function: NaN-propagating minimum combine for reductions; for floats, equal inputs return the second argument
    # (so signed zeros resolve deterministically).
    value = minimum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def _maximum_reduce(a, b):
    # [FUNCTION DOC] HELPER: Device function: NaN-propagating maximum combine for reductions; for floats, equal inputs return the second argument
    # (so signed zeros resolve deterministically).
    value = maximum(a, b)
    if is_floating(a):
        value = tl.where(a == b, b, value)
    return value


@triton.jit
def fmaximum(a, b):
    # [FUNCTION DOC] Device function: elementwise maximum using Triton default NaN handling (does not force NaN propagation).
    return tl.maximum(a, b)


@triton.jit
def min2(a, dim):
    # [FUNCTION DOC] Device function: NaN-propagating minimum reduction of a along dim.
    return tl.reduce(a, dim, minimum)


@triton.jit
def max2(a, dim):
    # [FUNCTION DOC] Device function: NaN-propagating maximum reduction of a along dim.
    return tl.reduce(a, dim, maximum)


@triton.jit
def min2_strict(a, dim):
    # [FUNCTION DOC] Device function: NaN-propagating minimum reduction along dim with signed-zero-aware combine.
    return tl.reduce(a, dim, _minimum_reduce)


@triton.jit
def max2_strict(a, dim):
    # [FUNCTION DOC] Device function: NaN-propagating maximum reduction along dim with signed-zero-aware combine.
    return tl.reduce(a, dim, _maximum_reduce)


@triton.jit
def fmax2(a, dim):
    # [FUNCTION DOC] Device function: maximum reduction along dim using the non-NaN-propagating maximum.
    return tl.reduce(a, dim, fmaximum)


@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index):
    # [FUNCTION DOC] Device function: combine two (value, index) pairs keeping the smaller value.
    # NaN is treated as smaller than any non-NaN value and equal to other NaNs; ties are broken by the lower index.
    # Returns the selected (value, index) pair.
    mask = a_value < b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    # [FUNCTION DOC] Device function: combine two (value, index) pairs keeping the larger value.
    # NaN is treated as larger than any non-NaN value and equal to other NaNs; ties go to the lower index.
    # Returns the selected (value, index) pair.
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan & (not b_isnan)
        # Consider NaNs as equal
        equal |= a_isnan & b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def min_with_index(value, index, dim):
    # [FUNCTION DOC] Device function: reduce (value, index) along dim, returning the minimum value and its index (lowest index on ties).
    return tl.reduce((value, index), dim, minimum_with_index)


@triton.jit
def max_with_index(value, index, dim):
    # [FUNCTION DOC] Device function: reduce (value, index) along dim, returning the maximum value and its index (lowest index on ties).
    return tl.reduce((value, index), dim, maximum_with_index)


@triton.jit
def exp(x, use_fast_math: tl.constexpr):
    # [FUNCTION DOC] Device function: exponential; uses fast math exp when the constexpr flag is set, otherwise libdevice exp.
    if use_fast_math:
        return math.exp(x)
    else:
        return libdevice.exp(x)


@triton.jit
def online_softmax_reduce(
    lhs_max,
    lhs_sum,
    dim,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    # [FUNCTION DOC] Device function: reduce partial (max, sum) softmax statistics along dim.
    # Computes the max (optionally signed-zero strict), then sums each partial sum rescaled by exp of the difference to the max;
    # -inf maxima contribute a zero shift to avoid NaN. Returns (max, sum).
    if strict_signed_zero:
        out_max = max2_strict(lhs_max, dim)
    else:
        out_max = max2(lhs_max, dim)
    out_max_keepdim = tl.expand_dims(out_max, dim)
    delta = tl.where(out_max_keepdim == float("-inf"), 0, lhs_max - out_max_keepdim)
    out_sum = tl.sum(lhs_sum * exp(delta, use_fast_math), dim)
    return out_max, out_sum


@triton.jit
def online_softmax_combine(
    lhs_max,
    lhs_sum,
    rhs_max,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    # [FUNCTION DOC] Device function: online softmax combine of an accumulator (lhs max, sum) with a new block's max.
    # Assumes the new block's sum is exactly 1, which saves registers and work. Rescales both sides by exp(max difference)
    # (scale 1 when the new max is -inf). Returns (new max, new sum).
    """
    When we do combine, we assume lhs is the accumulator and rhs is the next
    block of data.
    Then rhs_sum is always 1. With that assumption, we can save some registers
    and computation.
    """
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    # Should be
    #   out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    # but since rhs_sum is all 1, we can simplify it.
    out_sum = lhs_sum * lhs_scale + rhs_scale
    return out_max, out_sum


@triton.jit
def online_softmax_combine_with_sum(
    lhs_max,
    lhs_sum,
    rhs_max,
    rhs_sum,
    use_fast_math: tl.constexpr,
    strict_signed_zero: tl.constexpr,
):
    # [FUNCTION DOC] Device function: general online softmax combine of two (max, sum) pairs, rescaling each by exp(max difference)
    # (scale 1 when the combined max is -inf). Returns (new max, new sum).
    if strict_signed_zero:
        out_max = _maximum_reduce(lhs_max, rhs_max)
    else:
        out_max = maximum(lhs_max, rhs_max)

    lhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(lhs_max - out_max, use_fast_math)
    )
    rhs_scale = tl.where(
        out_max == float("-inf"), 1.0, exp(rhs_max - out_max, use_fast_math)
    )

    out_sum = lhs_sum * lhs_scale + rhs_sum * rhs_scale
    return out_max, out_sum


@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration):
    # [FUNCTION DOC] Device function: one Welford update step. On the first iteration the mean is the value, weight is 1 and m2 is zero;
    # otherwise it updates mean, m2 and weight with the new value. Returns (mean, m2, weight).
    if first_iteration:
        new_weight = tl.full(weight.shape, 1, weight.dtype)
        new_mean = value
        new_m2 = tl.zeros_like(m2)
    else:
        delta = value - mean
        new_weight = weight + 1
        new_mean = mean + delta / new_weight
        new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2, new_weight


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    # [FUNCTION DOC] Device function: merge two Welford partial states (mean, m2, weight).
    # Guards against infinity minus infinity producing NaN and against zero total weight. Returns (mean, m2, weight).
    # Guard against inf - inf = NaN when both means are infinite and equal.
    # This occurs during FP16/BF16 LayerNorm when inputs overflow to inf.
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def welford(mean, m2, weight, dim):
    # [FUNCTION DOC] Device function: reduce Welford states (mean, m2, weight) along dim using welford_combine.
    return tl.reduce((mean, m2, weight), dim, welford_combine)


@triton.jit
def device_assert_then(cond, msg, r):
    # [FUNCTION DOC] Device function: device-side assert on cond with message msg, then return r unchanged.
    tl.device_assert(cond, msg)
    return r


@triton.jit
def rand_eager_kernel(seed, offset_blocks, tid: tl.tensor, VEC: tl.constexpr):
    # [FUNCTION DOC] Device function: eager-mode-compatible Philox random uniform generator.
    # Args: seed; offset_blocks; tid (thread index tensor); VEC (constexpr vector width). Derives subsequence and block offset from tid,
    # runs Philox to get four 32-bit values, selects the one for the lane, and returns a float32 in (0, 1].
    inv = 1.0 / 4294967296.0
    half = inv * 0.5

    tid_u64 = tid.to(tl.uint64)

    subseq = tid_u64 // VEC
    which4 = (tid_u64 % VEC) // 4
    lane = tid_u64 % 4

    offblk = offset_blocks.to(tl.uint64) + which4

    u0, u1, u2, u3 = tl.philox(
        seed,
        (offblk & 0xFFFFFFFF).to(tl.uint32),
        ((offblk >> 32) & 0xFFFFFFFF).to(tl.uint32),
        (subseq & 0xFFFFFFFF).to(tl.uint32),
        ((subseq >> 32) & 0xFFFFFFFF).to(tl.uint32),
    )

    v01 = tl.where(lane == 0, u0, u1)
    v23 = tl.where(lane == 2, u2, u3)
    rand_int = tl.where((lane == 0) | (lane == 1), v01, v23)

    return 1.0 - (rand_int.to(tl.float32) * inv + half)


@triton.jit
def _random_4x_to_block(r0, r1, r2, r3):
    # [FUNCTION DOC] HELPER: Device function: interleave four equal-size random tensors into one flat tensor 4x as large, ordered by logical offset
    # so the stream does not depend on block size.
    # Pack lanes by logical offset so the random stream is independent of XBLOCK.
    size: tl.constexpr = r0.numel
    return tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), [size * 4])


@triton.jit
def rand4x(seed, offsets, BLOCK: tl.constexpr):
    # [FUNCTION DOC] Device function: uniform random numbers for the given offsets using the 4-wide Philox generator when BLOCK is a multiple of 4;
    # otherwise falls back to the plain rand. Seed is reduced to a scalar. Returns a tensor of BLOCK floats.
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.rand4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.rand(seed, offsets)


@triton.jit
def randn4x(seed, offsets, BLOCK: tl.constexpr):
    # [FUNCTION DOC] Device function: normal random numbers for the given offsets; 4-wide Philox path when BLOCK is a multiple of 4,
    # otherwise the plain randn. Returns a tensor of BLOCK floats.
    offsets = offsets.to(tl.uint32)
    seed = tl.min(seed + offsets * 0, axis=0)
    if BLOCK >= 4 and BLOCK % 4 == 0:
        base = tl.min(offsets, axis=0)
        reduced_offsets = base // 4 + tl.arange(0, BLOCK // 4)
        r0, r1, r2, r3 = tl.randn4x(seed, reduced_offsets)
        return _random_4x_to_block(r0, r1, r2, r3)
    return tl.randn(seed, offsets)


@triton.jit
def randint64(seed, offset, low, high):
    # [FUNCTION DOC] Device function: random int64 in [low, high). Combines two 32-bit random words into 64 bits and reduces modulo (high - low)
    # (slight modulo bias). Returns int64 values.
    r0, r1, _r2, _r3 = tl.randint4x(seed, offset)
    r0 = r0.to(tl.uint64)
    r1 = r1.to(tl.uint64)
    result = r0 | (r1 << 32)
    size = high - low
    result = result % size.to(tl.uint64)
    result = result.to(tl.int64) + low
    return result


@triton.jit
def _any_combine(a, b):
    # [FUNCTION DOC] HELPER: Device function: bitwise-or combine for the any reduction.
    return a | b


@triton.jit
def any(a, dim):
    # [FUNCTION DOC] Device function: logical any (bitwise-or reduction) of a along dim.
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def bucketize_binary_search(
    values: tl.tensor,
    boundaries_ptr: tl.tensor,
    BOUNDARIES_SIZE: int,
    BOUNDARIES_UNDERLYING_NUMEL: int,
    BOUNDARIES_STRIDE: int,
    boundary_indices: tl.tensor,
    indexing_dtype: tl.dtype,
    right: "bool",  # triton can't handle the unquoted bool annotation
    sorter_ptr: tl.tensor,
    SORTER_STRIDE: int,
    sorter_indices: tl.tensor,
):
    # [FUNCTION DOC] Device function: vectorized binary search for bucketize / searchsorted.
    # For each value, searches its set of boundaries (addressed via boundary_indices, size, stride, with an optional sorter for unsorted
    # boundaries) and returns the insertion index in the indexing dtype. The right flag selects >= versus > comparison; NaN values
    # sort after everything. Loads are masked to stay inside the boundaries tensor.
    """
    See [Note: Inductor bucketize op]

    Inputs:
    -------
    values: the values to bucketize.
    boundaries_ptr: a pointer to the beginning of the boundaries tensor, in 1-D.
    BOUNDARIES_SIZE: the length of the last dimension of the boundaries tensor (i.e. one
    individual set of boundaries).
    BOUNDARIES_UNDERLYING_NUMEL: the length of the boundaries tensor, in 1-D, ignoring
    any striding.
    BOUNDARIES_STRIDE: the stride of the last dimension of the boundaries tensor
    boundary_indices: a tensor of the same size as "values"; each element is an index
    into a 1-D, un-strided boundaries tensor, pointing to the first element in the set
    of boundaries used for that value.
    indexing_dtype: the dtype used for indexing into the boundaries tensor, and the
    return dtype.
    right: if true, use boundary intervals closed on the left; otherwise use intervals
    closed on the right.
    sorter_ptr: an optional pointer to a sorter tensor of the same shape as boundaries,
    but potentially different striding.  If present, this allows us to treat boundaries
    as sorted even if the elements of boundaries are unsorted.
    SORTER_STRIDE: must be present if sorter_ptr is non-None; the stride of the last
    dimension of the sorter tensor.
    sorter_indices: must be present if sorter_ptr is non-None; see "boundary_indices".
    BLOCK_SHAPE: the shape of the data block being processed.
    """

    low = tl.zeros(values.shape, dtype=indexing_dtype)
    high = tl.full(values.shape, BOUNDARIES_SIZE, dtype=indexing_dtype)

    full_range = BOUNDARIES_SIZE + 1
    while full_range > 1:
        mid = (high + low) // 2
        mask = (
            (mid * BOUNDARIES_STRIDE + boundary_indices) < BOUNDARIES_UNDERLYING_NUMEL
        ).logical_and(mid < BOUNDARIES_SIZE)
        mid_indices = (
            mid
            if sorter_ptr is None or SORTER_STRIDE is None
            else tl.load(
                sorter_ptr + sorter_indices + SORTER_STRIDE * mid,
                mask=mask,
                other=0,
            )
        )

        bucket_upper_bound = tl.load(
            boundaries_ptr + boundary_indices + BOUNDARIES_STRIDE * mid_indices,
            mask=mask,
            other=0,
        )
        if right:
            is_above = values >= bucket_upper_bound
        else:
            is_above = values > bucket_upper_bound

        if is_floating(values):
            is_above = is_above | (values != values)

        low = tl.where(is_above & mask, mid + 1, low)
        high = tl.where(is_above, high, mid)

        full_range = (full_range + 1) // 2

    return low


@triton.jit
def pack_value_flag(
    value,
    flag,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # [FUNCTION DOC] Device function: pack a value (bitcast to unsigned) and a flag into one wider unsigned integer; flag in low bits, value in high bits.
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    uv = value.to(DTYPE_VALUE_AS_UINT, bitcast=True).to(DTYPE_PACK)
    return flag.to(DTYPE_PACK) | (uv << bitwidth)


@triton.jit
def unpack_value(
    pack,
    DTYPE_VALUE,
    DTYPE_VALUE_AS_UINT,
):
    # [FUNCTION DOC] Device function: extract the value from a packed (value, flag) integer and bitcast it back to the value dtype.
    # Workaround for triton bug, tensor.to doesn't unwrap constexpr values
    DTYPE_VALUE = tl.core._unwrap_if_constexpr(DTYPE_VALUE)
    DTYPE_VALUE_AS_UINT = tl.core._unwrap_if_constexpr(DTYPE_VALUE_AS_UINT)
    bitwidth = DTYPE_VALUE_AS_UINT.primitive_bitwidth
    value_uint = (pack >> bitwidth).to(DTYPE_VALUE_AS_UINT)
    return value_uint.to(DTYPE_VALUE, bitcast=True)


@triton.jit
def unpack_flag(pack, DTYPE_FLAG):
    # [FUNCTION DOC] Device function: extract the flag from a packed (value, flag) integer.
    return pack.to(DTYPE_FLAG)


@triton.jit
def exclusive_scan_decoupled_lookback(
    scratch_base,
    block_value,
    index,
    combine_fn,
    DTYPE_VALUE_AS_UINT: tl.constexpr,
    DTYPE_PACK: tl.constexpr,
):
    # [FUNCTION DOC] Device function: single-pass decoupled-lookback exclusive scan across thread blocks for values of 32 bits or fewer.
    # Each block publishes its packed value with flag 1 (partial), walks back over predecessors spinning until their flag is set,
    # combining values until it sees an inclusive result (flag 2), then publishes its inclusive prefix with flag 2.
    # Args: scratch_base pointer; block_value; index of block; combine_fn; packing dtypes. Returns the exclusive prefix (untouched for block 0).
    # Side effects: atomic global-memory writes to scratch; relies on all earlier blocks making progress.
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    DTYPE_VALUE_AS_UINT: A tl.uint{n} type equal in size to ``block_value``
    DTYPE_PACK: Unsigned type twice the width of block_value

    NOTE: This function is limited to values which are 32-bits or less because
    we need to pack (value, flag) into a single unsigned int.
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    DTYPE_VALUE = block_value.dtype
    pack = pack_value_flag(
        block_value,
        tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        # tl.atomic_load
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        while flag == 0:
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = unpack_flag(pack, DTYPE_VALUE_AS_UINT)

        value = unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = -1
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    pack = pack_value_flag(
        inclusive_prefix,
        tl.full([], 2, DTYPE_VALUE_AS_UINT),
        DTYPE_VALUE_AS_UINT,
        DTYPE_PACK,
    )
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    return exclusive_prefix


@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    # [FUNCTION DOC] Device function: decoupled-lookback exclusive scan for 64-bit values using three 64-bit scratch slots per block
    # (flag, partial value, inclusive value) with release/acquire atomics and barriers. Returns the exclusive prefix.
    # KNOWN ISSUE: docstring mentions an init argument that does not exist in the signature.
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identity of combine_fn
    """
    # Publish block sum so subsequent blocks don't get stuck waiting for us
    if index > 0:
        block_value_u64 = block_value.to(tl.uint64, bitcast=True)
        tl.store(scratch_base + 3 * index + 1, block_value_u64)
        tl.debug_barrier()
        flag_one = tl.full([], 1, tl.uint64)
        tl.atomic_xchg(scratch_base + 3 * index + 0, flag_one, sem="release")

    # Calculate exclusive prefix scan
    exclusive_prefix = tl.zeros([], block_value.dtype)
    prefix_valid = False
    test_target = index - 1
    while test_target >= 0:
        flag = tl.full([], 0, tl.uint64)
        while flag == 0:
            flag = tl.atomic_add(scratch_base + 3 * test_target + 0, 0, sem="acquire")

        value_u64 = tl.load(scratch_base + 3 * test_target + flag.to(tl.int32))
        value = value_u64.to(block_value.dtype, bitcast=True)
        if prefix_valid:
            exclusive_prefix = combine_fn(value, exclusive_prefix)
        else:
            exclusive_prefix = value
            prefix_valid = True

        if flag == 2:
            test_target = tl.full([], -1, index.dtype)  # Match the original type
        else:
            test_target = test_target - 1

    # Make inclusive block sum visible to other blocks
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    inclusive_prefix_u64 = inclusive_prefix.to(tl.uint64, bitcast=True)
    tl.store(scratch_base + 3 * index + 2, inclusive_prefix_u64)
    tl.debug_barrier()
    flag_two = tl.full([], 2, tl.uint64)
    tl.atomic_xchg(scratch_base + 3 * index + 0, flag_two, sem="release")

    return exclusive_prefix


@triton.jit
def frexp(x):
    # [FUNCTION DOC] Device function: split x into (mantissa, exponent) like C frexp; zero, inf and NaN map to exponent 0 with mantissa 0 or x itself.
    # TODO(isuruf): use inline_asm_elementwise here
    zero = x == 0
    not_finite = libdevice.isinf(x).to(tl.int1) | libdevice.isnan(x).to(tl.int1)
    special = zero | not_finite
    safe_x = tl.where(special, 1.0, x)
    y = libdevice.ilogb(safe_x) + 1
    exponent = tl.where(special, 0, y)
    mantissa = tl.where(zero, 0, tl.where(not_finite, x, libdevice.ldexp(safe_x, -y)))
    return mantissa, exponent


@triton.jit
def _compare_and_swap_with_index(
    x,
    idxs,
    rnumel,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Device function: one compare-and-swap stage of a bitonic network over a flattened power-of-two block with paired indices.
    # Splits elements into left/right partners at the stride for step i, decides swap with NaN-greater-than-all ordering,
    # index tie-break when stable, ordering of invalid (index >= rnumel) elements last, and a flip mask for direction.
    # Swaps via bitwise xor on integer bitcasts. Returns (values, indices).
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)

    y = tl.reshape(x, shape)
    iy = y.to(idtype, bitcast=True)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    right_mask = tl.arange(0, 2)[None, :, None].to(idtype)
    left_mask = (1 - right_mask).to(idtype)
    ileft = tl.broadcast_to(tl.sum(iy * left_mask, 1).to(idtype)[:, None, :], shape)
    iright = tl.broadcast_to(tl.sum(iy * right_mask, 1).to(idtype)[:, None, :], shape)
    ileft = tl.reshape(ileft, x.shape)
    iright = tl.reshape(iright, x.shape)
    left = ileft.to(x.dtype, bitcast=True)
    right = iright.to(x.dtype, bitcast=True)

    # idx
    y_idx = tl.reshape(idxs, shape)
    left_idx = tl.broadcast_to(
        tl.sum(y_idx * left_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    right_idx = tl.broadcast_to(
        tl.sum(y_idx * right_mask.to(y_idx.dtype), 1)[:, None, :], shape
    )
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # valid
    if rnumel is None:
        left_valid_mask = tl.full(x.shape, True, tl.int1)
        right_valid_mask = tl.full(x.shape, True, tl.int1)
    else:
        left_valid_mask = left_idx < rnumel
        right_valid_mask = right_idx < rnumel

    # actual compare-and-swap
    ix = x.to(idtype, bitcast=True)

    # sort treats nan as having the higher value. comparisons with nan always return False.
    # to align with sort semantics, we need to update descending to check if right_isnan,
    # and ascending to check if left_isnan.
    left_isnan = left != left
    right_isnan = right != right

    if descending:
        cond = left < right
        if is_floating(left):
            if not stable:
                cond = cond | right_isnan
            else:
                cond = cond | (right_isnan & (~left_isnan))

    else:
        cond = left > right
        if is_floating(left):
            if not stable:
                cond = cond | left_isnan
            else:
                cond = cond | (left_isnan & (~right_isnan))

    if stable:
        # When stable sorting, tie break by index
        eq = left == right
        if is_floating(left):
            eq = eq | (left_isnan & right_isnan)
        cond = cond | (eq & (left_idx > right_idx))

    cond = (right_valid_mask > left_valid_mask) | (
        (right_valid_mask == left_valid_mask) & cond
    )
    cond = (cond ^ flip).to(tl.int1)
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_idxs = idxs ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(idxs))

    return ret.to(x.dtype, bitcast=True), new_idxs


@triton.jit
def _bitonic_merge_with_index(
    x,
    idxs,
    rnumel,
    stage: tl.constexpr,
    alternating: tl.constexpr,
    n_dims: tl.constexpr,
    stable: tl.constexpr,
    descending: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Device function: one bitonic merge stage (stage rounds of compare-and-swap); alternates direction between sub-sequences
    # when alternating is set. Returns (values, indices).
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if alternating:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = False
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, idxs = _compare_and_swap_with_index(
            x, idxs, rnumel, flip, i + (n_dims - stage), n_dims, stable, descending
        )
    return x, idxs


@triton.jit
def sort_with_index(
    x,  # value
    idxs,  # index
    rnumel,  # number of elements
    dim: tl.constexpr = None,
    stable: tl.constexpr = tl.constexpr(False),
    descending: tl.constexpr = tl.constexpr(False),
):
    # [FUNCTION DOC] Device function: sort x along its minor dimension carrying idxs, using bitonic merge sort over a power-of-two block.
    # Args: x; idxs; rnumel (valid element count or None); dim (must be minor dim); stable; descending. NaN sorts as largest.
    # Fails at compile time if dim is not the last dimension. Returns (sorted values, permuted indices).
    x, idxs = tl.broadcast(x, idxs)
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, idxs = _bitonic_merge_with_index(
            x,
            idxs,
            rnumel,
            i,
            alternating=i < n_dims,
            n_dims=n_dims,
            stable=stable,
            descending=descending,
        )
    return x, idxs


@triton.jit
def select_one(x, mask, dim, keep_dims=False):
    # [FUNCTION DOC] Device function: select the single element of x where mask is nonzero along dim, via a masked sum on the integer bitcast.
    # Returns a value of x's dtype (assumes exactly one selected element per reduction).
    idtype = tl.core.get_int_dtype(x.dtype.primitive_bitwidth, signed=False)
    ix = x.to(idtype, bitcast=True)
    iy = tl.sum(ix * mask, dim, keep_dims=keep_dims)
    return iy.to(idtype).to(x.dtype, bitcast=True)


@triton.jit
def x_grid_barrier(sem):
    # [FUNCTION DOC] Device function: grid-wide barrier across all programs sharing the same y/z program id.
    # Uses a uint32 semaphore (zero or 0x80000000 initialized, unique per y/z id): program 0 adds a large offset, others add one,
    # then spins with acquire atomics until the top bit flips. Side effects: atomics on sem and debug barriers.
    """
    Wait for all other thread blocks in grid sharing same y/z program_id
    to reach this barrier before returning.

    Args:
        sem: an uint32 semaphores, zero or 0x80000000 initialized.  Must be unique to each y/z program ID.
    """
    # ensure stores before this are visible
    tl.debug_barrier()

    one_i32 = 1
    one_u32 = one_i32.to(tl.uint32)  # type: ignore[attr-defined]
    expected = tl.num_programs(0).to(tl.uint32)
    if tl.program_id(0) == 0:
        nb = 0x80000000 - (expected - one_u32)
    else:
        nb = one_u32

    old_arrive = tl.atomic_add(sem, nb, sem="release")

    bar_flipped = False
    while not bar_flipped:
        # want a `ld.acquire.gpu.u32 $0,[$1];` but Triton doesn't have it
        current_arrive = tl.atomic_add(sem, 0, sem="acquire")
        # current_arrive = tl.load(sem, volatile=True)
        bar_flipped = ((old_arrive ^ current_arrive) & 0x80000000) != 0

    # TODO(jansel): is this needed?
    tl.debug_barrier()


def triton_builtin(f: Callable[..., _T]) -> Callable[..., _T]:
    # [FUNCTION DOC] Decorator marking a function as a Triton builtin evaluated at compile time. Returns a wrapper that maps the semantic keyword to
    # the builder keyword on Triton versions that need it (otherwise the function itself) and sets a marker attribute on it.
    """
    Decorator to mark a function as a Triton built-in function.  These functions
    are evaluated at compile time.

    Args:
        f (function): The function to be marked as a Triton built-in.

    Returns:
        function: The same function, marked as a Triton built-in.
    """
    if builtins_use_semantic_kwarg:
        # support Triton before and after https://github.com/triton-lang/triton/pull/7054
        # and after https://github.com/triton-lang/triton/pull/7239
        def wrapper(*args, _semantic, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper adapting the newer _semantic keyword to the _builder keyword expected by the wrapped function.
            kwargs["_builder"] = _semantic
            return f(*args, **kwargs)
    else:
        wrapper = f  # type: ignore[assignment]

    wrapper.__triton_builtin__ = True  # type: ignore[attr-defined]
    return wrapper


@triton_builtin
def constexpr_next_power_of_2(
    n: tl.constexpr, *, _builder: object = None
) -> tl.constexpr:
    # [FUNCTION DOC] Builtin: next power of two of a constexpr integer, returned as a constexpr. Raises AssertionError if n is not a constexpr.
    """
    A version triton.next_power_of_two that can be used within a kernel on constants.
    """
    if not isinstance(n, tl.constexpr):
        raise AssertionError(f"Expected tl.constexpr, got {type(n)}")
    return tl.constexpr(triton.next_power_of_2(n.value))


@triton_builtin
def if_mask(mask: Any, val, *, _builder: object = None) -> tl.constexpr:
    # [FUNCTION DOC] Builtin: returns constexpr None when mask is a constexpr None, otherwise returns val (avoids Triton error when 'other' is given without mask).
    """
    Work around triton compile error: `ValueError: `other` cannot be provided without `mask``
    A compile-time to check to return either `val` or `None` depending on the value of mask.
    """
    if isinstance(mask, tl.constexpr) and mask.value is None:
        return tl.constexpr(None)
    return val


@triton.jit
def inline_asm_pack(x, pack: tl.constexpr):
    # [FUNCTION DOC] Device function: flatten x to 1D and pad with zeros (by repeated join) so the length is divisible by pack. Returns the 1D tensor.
    """Ravel to 1D and pad (via join with zeros) so numel is divisible by pack."""
    result = x.ravel()
    # Only pad when the block size is smaller than pack. When block >= pack
    # the numel is already divisible by pack (both are powers of 2).
    n_pad: tl.constexpr = _log2(pack) - _log2(result.numel)
    for _ in tl.static_range(n_pad):
        result = tl.reshape(
            tl.join(result, tl.zeros_like(result)), (result.shape[0] * 2,)
        )
    return result


@triton.jit
def inline_asm_unpack(x, orig, pack: tl.constexpr):
    # [FUNCTION DOC] Device function: inverse of inline_asm_pack; strips the padding and reshapes to the shape of orig. Returns the reshaped tensor.
    """Unpad and reshape back to orig's shape."""
    result = x
    n_pad: tl.constexpr = _log2(pack) - _log2(orig.numel)
    for _ in tl.static_range(n_pad):
        result, _ = tl.split(tl.reshape(result, (result.shape[0] // 2, 2)))
    return tl.reshape(result, orig.shape)
