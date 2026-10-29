# mypy: allow-untyped-defs
# [MODULE DOC] Halide-expression helpers for Philox-based random number generation used by
# Inductor's Halide backend: counter-based Philox rounds, uniform/normal/integer samplers.
# Tries to import halide as hl at import time; if absent, hl and the Philox u32 constants are None.
# All functions build Halide expressions (not concrete values) and require hl to be available.
import math


try:
    import halide as hl  # type: ignore[import-untyped, import-not-found]
except ImportError:
    hl = None

PHILOX_N_ROUNDS_DEFAULT = 10  # Default number of rounds for philox

if hl is not None:
    PHILOX_KEY_A_U32 = hl.u32(0x9E3779B9)
    PHILOX_KEY_B_U32 = hl.u32(0xBB67AE85)
    PHILOX_ROUND_A_U32 = hl.u32(0xD2511F53)
    PHILOX_ROUND_B_U32 = hl.u32(0xCD9E8D57)
else:
    PHILOX_KEY_A_U32 = None
    PHILOX_KEY_B_U32 = None
    PHILOX_ROUND_A_U32 = None
    PHILOX_ROUND_B_U32 = None


def _pair_uniform_to_normal(u1, u2):
    # [FUNCTION DOC] HELPER: Box-Muller transform. Takes two uniform float expressions in [0, 1),
    # clamps the first to at least 1e-7 to avoid log of zero, and returns a pair of
    # independent standard-normal float32 expressions (radius*cos, radius*sin).
    """Box-Muller transform"""
    u1 = hl.max(hl.f32(1.0e-7), u1)
    th = hl.f32(math.tau) * u2
    r = hl.sqrt(hl.f32(-2.0) * hl.log(u1))
    return r * hl.cos(th), r * hl.sin(th)


def _uint_to_uniform_float(x):
    # [FUNCTION DOC] HELPER: Convert a 32-bit (signed or unsigned) integer expression to a float64
    # expression in [0, 1). Reinterprets as Int32, maps negatives to their bitwise complement
    # (so values are non-negative), then multiplies by about 2^-31.
    # Raises AssertionError if the input type is not UInt(32) or Int(32).
    """
    Numerically stable function to convert a random uint into a random float uniformly sampled in [0, 1).
    """

    # TODO:
    # conditions can be simplified
    # scale is ((2**23 - 1) / 2**23) * 2**(N_BITS - 1)
    # https://github.com/triton-lang/triton/blob/e4a0d93ff1a367c7d4eeebbcd7079ed267e6b06f/python/triton/language/random.py#L116-L132.
    if x.type() != hl.UInt(32) and x.type() != hl.Int(32):
        raise AssertionError(f"Expected UInt(32) or Int(32), got {x.type()}")
    x = hl.cast(hl.Int(32), x)
    scale = hl.f64(4.6566127342e-10)
    x = hl.select(x < 0, -x - 1, x)
    return x * scale


def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds):
    # [FUNCTION DOC] Run the Philox4x32 counter-based generator for n_rounds rounds (a Python-level
    # unrolled loop) on four 32-bit counter expressions and two 32-bit key expressions.
    # Each round mixes counters with high/low products of the round constants and then bumps
    # the keys by the key constants. Returns the four transformed counter expressions.
    def umulhi(a, b):
        # [FUNCTION DOC] HELPER: High 32 bits of the 64-bit product of two 32-bit expressions, as UInt(32).
        a = hl.cast(hl.UInt(64), a)
        b = hl.cast(hl.UInt(64), b)
        return hl.cast(hl.UInt(32), ((a * b) >> 32) & hl.u64(0xFFFFFFFF))

    for _ in range(n_rounds):
        _c0, _c2 = c0, c2

        c0 = umulhi(PHILOX_ROUND_B_U32, _c2) ^ c1 ^ k0
        c2 = umulhi(PHILOX_ROUND_A_U32, _c0) ^ c3 ^ k1
        c1 = PHILOX_ROUND_B_U32 * _c2
        c3 = PHILOX_ROUND_A_U32 * _c0
        # raise key
        k0 = k0 + PHILOX_KEY_A_U32
        k1 = k1 + PHILOX_KEY_B_U32

    return c0, c1, c2, c3


def halide_philox(seed, c0, c1, c2, c3, n_rounds):
    # [FUNCTION DOC] Split a 64-bit seed into low and high 32-bit keys and run philox_impl with the
    # given 32-bit counters. Returns four 32-bit random expressions.
    # Raises AssertionError if c0 is not a 32-bit type.
    seed = hl.cast(hl.UInt(64), seed)

    if c0.type().bits() != 32:
        raise AssertionError(f"Expected 32-bit type, got {c0.type().bits()}-bit")

    seed_hi = hl.cast(hl.UInt(32), (seed >> 32) & hl.u64(0xFFFFFFFF))
    seed_lo = hl.cast(hl.UInt(32), seed & hl.u64(0xFFFFFFFF))

    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)


def randint4x(seed, offset, n_rounds):
    # [FUNCTION DOC] Generate four random 32-bit integer expressions for a seed and offset
    # (offset cast to UInt32 as the first counter, other counters zero).
    offset = hl.cast(hl.UInt(32), offset)
    _0 = hl.u32(0)
    return halide_philox(seed, offset, _0, _0, _0, n_rounds)


def rand4x(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    # [FUNCTION DOC] Like randint4x but converts each of the four outputs to a uniform float in [0, 1).
    # Returns a 4-tuple of float expressions.
    i1, i2, i3, i4 = randint4x(seed, offset, n_rounds)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    u3 = _uint_to_uniform_float(i3)
    u4 = _uint_to_uniform_float(i4)
    return u1, u2, u3, u4


def randint(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    # [FUNCTION DOC] Return the first of the four Philox random 32-bit integer expressions.
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret


def rand(seed, offset, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    # [FUNCTION DOC] Return a uniform float expression in [0, 1) derived from the first Philox output.
    source = randint(seed, offset, n_rounds)
    return _uint_to_uniform_float(source)


def rand_eager_kernel(seed, offset_blocks, tid, VEC, n_rounds=PHILOX_N_ROUNDS_DEFAULT):
    # [FUNCTION DOC] Emulate eager-mode (CUDA-style) uniform float generation per thread id.
    # The thread id is split into a subsequence (tid // VEC), a block-of-4 index and a lane (tid % 4);
    # the 64-bit offset block plus block index and the subsequence become the four 32-bit counters.
    # Philox outputs are selected by lane and mapped to a float32 in (0, 1] as one minus
    # (value * 2^-32 + half step). Returns a float32 expression.
    inv = hl.cast(hl.Float(32), 1.0 / 4294967296.0)  # 2^-32
    half = hl.cast(hl.Float(32), 0.5) * inv

    tid_u64 = hl.cast(hl.UInt(64), tid)
    VEC_u64 = hl.cast(hl.UInt(64), VEC)
    subseq = tid_u64 // VEC_u64
    which4 = (tid_u64 % VEC_u64) // hl.cast(hl.UInt(64), 4)
    lane = tid_u64 % hl.cast(hl.UInt(64), 4)

    offblk = hl.cast(hl.UInt(64), offset_blocks) + which4

    c0 = hl.cast(hl.UInt(32), offblk & hl.cast(hl.UInt(64), 0xFFFFFFFF))
    c1 = hl.cast(
        hl.UInt(32),
        (offblk >> hl.cast(hl.UInt(64), 32)) & hl.cast(hl.UInt(64), 0xFFFFFFFF),
    )
    c2 = hl.cast(hl.UInt(32), subseq & hl.cast(hl.UInt(64), 0xFFFFFFFF))
    c3 = hl.cast(
        hl.UInt(32),
        (subseq >> hl.cast(hl.UInt(64), 32)) & hl.cast(hl.UInt(64), 0xFFFFFFFF),
    )

    u0, u1, u2, u3 = halide_philox(seed, c0, c1, c2, c3, n_rounds)

    v01 = hl.select(lane == hl.cast(hl.UInt(64), 0), u0, u1)
    v23 = hl.select(lane == hl.cast(hl.UInt(64), 2), u2, u3)
    rand_int = hl.select(
        (lane == hl.cast(hl.UInt(64), 0)) | (lane == hl.cast(hl.UInt(64), 1)), v01, v23
    )

    return hl.cast(hl.Float(32), 1.0) - (hl.cast(hl.Float(32), rand_int) * inv + half)


def randn(seed, offset):
    # [FUNCTION DOC] Standard-normal float expression: uses the first two Philox outputs as uniforms
    # and returns only the first Box-Muller value (the second is discarded).
    i1, i2, _, _ = randint4x(seed, offset, PHILOX_N_ROUNDS_DEFAULT)
    u1 = _uint_to_uniform_float(i1)
    u2 = _uint_to_uniform_float(i2)
    n1, _ = _pair_uniform_to_normal(u1, u2)
    return n1


def randint64(seed, offset, low, high):
    # [FUNCTION DOC] Random Int64 expression in [low, high): combines the first two Philox outputs into
    # a 64-bit value, takes it modulo (high - low) as unsigned, and adds low.
    # KNOWN ISSUE: modulo reduction introduces slight bias for ranges not dividing 2^64.
    r0, r1, _r2, _r3 = randint4x(seed, offset, PHILOX_N_ROUNDS_DEFAULT)
    r0 = hl.cast(hl.UInt(64), r0)
    r1 = hl.cast(hl.UInt(64), r1)

    result = r0 | (r1 << 32)
    size = high - low
    result = result % hl.cast(hl.UInt(64), size)
    result = hl.cast(hl.Int(64), result) + low
    return result
