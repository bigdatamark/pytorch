# [MODULE DOC] Pure-PyTorch quantization helpers for MX (MXFP8, MXFP4) and NVFP4 block-scaled
# formats, ported from torchao, plus a swizzle of scale tensors into the blocked layout that
# cuBLAS expects. Defines numeric constants for FP8 E4M3, E8M0 and FP4 E2M1 formats, the
# quantizers, and a scale-blocking function. Import-time side effects: creates three
# torch.compile wrappers (dynamic shapes) of the quantizers; compilation itself happens
# lazily on first call. Only floor-mode scaling is supported.
"""Minimal MX / NVFP4 quantization + scale swizzling utilities.

Ported from torchao (BSD-3) to avoid the runtime dependency:
  torchao/prototype/mx_formats/{mx_tensor, nvfp4_tensor, utils, constants}.py
  torchao/prototype/custom_fp_utils.py
  torchao/prototype/mx_formats/kernels.py

All quantizers are pure-PyTorch. Use the `to_mx_compiled` / `to_mxfp4_compiled` /
`to_nvfp4_compiled` module-level handles if you want torch.compile-generated
Triton kernels (much faster on big tensors; one-time compile overhead).

Only the FLOOR scaling mode is ported (torchao's default for MX formats).
"""

import torch

F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E4M3_MAX_POW2 = 8
E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255
F32_EXP_BIAS = 127
F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)  # 2**-126
MBITS_F32 = 23
EBITS_F32 = 8

# FP4 E2M1 constants
F4_E2M1_MAX = 6.0
F4_E2M1_MAX_POW2 = 2
F4_E2M1_MAX_INT = 7  # 3-bit magnitude mask
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1

E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny


def _n_ones(n: int) -> int:
    # [FUNCTION DOC] HELPER: Return an integer whose low n bits are all set (a mask of n ones).
    return (1 << n) - 1


def to_mx(data_hp: torch.Tensor, block_size: int = 32):
    # [FUNCTION DOC] Quantize a contiguous bf16 or fp32 tensor to MXFP8 (E4M3 data) with one E8M0
    # scale per block of block_size values along the last dimension. Returns the quantized data
    # (same shape as the input) and the scales (last dimension divided by the block size).
    # Per block: the maximum magnitude is found, its float exponent minus the E4M3 maximum
    # exponent gives the floor-mode scale exponent, clamped to the E8M0 range and biased; NaN
    # blocks get the NaN code. Values are divided by the reconstructed scale (floored at the
    # smallest normal float so tiny scales are not flushed to zero) and, when not under dynamo
    # tracing, clamped to the E4M3 range before the cast. Fails an assertion for other dtypes, a
    # last dimension not divisible by block size, or non-contiguous input.
    """MXFP8-e4m3 quantization with FLOOR scaling.

    Args:
        data_hp: (..., K) bf16 or fp32 tensor, contiguous, K % block_size == 0.
    Returns:
        qdata: (..., K) float8_e4m3fn
        scale: (..., K // block_size) float8_e8m0fnu
    """
    assert data_hp.dtype in (torch.bfloat16, torch.float32)
    assert data_hp.shape[-1] % block_size == 0
    assert data_hp.is_contiguous()

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    # FLOOR scaling: extract biased exponent of max_abs via bit-shift
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((torch.bitwise_right_shift(max_abs_int32, MBITS_F32)) & 0xFF) - F32_EXP_BIAS
    scale_e8m0_unbiased = extracted_pow2 - F8E4M3_MAX_POW2
    scale_e8m0_unbiased = torch.clamp(
        scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1
    )
    scale_e8m0_biased = (scale_e8m0_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)
    # restore NaN sentinel (uint8 cast drops NaN)
    scale_e8m0_biased = torch.where(torch.isnan(max_abs), E8M0_EXPONENT_NAN_VAL, scale_e8m0_biased)

    # reconstruct fp32 scale from biased exponent
    scale_fp32 = (torch.bitwise_left_shift(scale_e8m0_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    # avoid 2**-127 being flushed to 0 (pytorch #125557)
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = data_hp / scale_fp32
    # eager fp8 cast is unsaturated; clamp explicitly
    if not torch._dynamo.is_compiling():
        data_lp = torch.clamp(data_lp, min=-F8E4M3_MAX, max=F8E4M3_MAX)

    qdata = data_lp.to(torch.float8_e4m3fn).reshape(orig_shape)
    scale = scale_e8m0_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return qdata, scale


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """FP32 -> sub-byte float (uint8, code in low bits). Verbatim from torchao.
    # [FUNCTION DOC] HELPER: Convert fp32 values to a small float format with the given
    # exponent and mantissa bit counts, returning the codes in the low bits of uint8 (sign in
    # the top used bit). Handles three cases with masks: saturation to the largest code for
    # values at or above the largest normal, denormals via an addition trick, and normals via
    # an exponent rebias with round-to-nearest-even. NaN is not handled. Asserts fp32 input
    # and a total width of at most 8 bits.

    Round-to-nearest-even via magic-adder; saturation on overflow; no NaN.
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp
    return x.to(torch.uint8)


def _pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit uint8 values in pairs: pair (a,b) -> byte (b<<4 | a)."""
    # [FUNCTION DOC] HELPER: Pack pairs of 4-bit values stored in uint8 into bytes: the first of
    # each pair goes in the low nibble and the second in the high nibble. The last dimension
    # must be even (asserted). Returns a tensor with that dimension halved.
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] | uint8_data[1::2] << 4).view(*shape[:-1], shape[-1] // 2)


def _compute_e8m0_scale_floor(max_abs: torch.Tensor, target_max_pow2: int) -> torch.Tensor:
    """Return biased E8M0 scale (uint8) for FLOOR-mode MX quantization."""
    # [FUNCTION DOC] HELPER: Compute the biased E8M0 scale (uint8) for floor-mode MX quantization
    # from per-block maximum magnitudes in fp32. The exponent of the maximum minus the target
    # maximum exponent is clamped to the E8M0 range, biased, and entries where the maximum is NaN
    # are set to the NaN code.
    max_abs_int32 = max_abs.view(torch.int32)
    extracted_pow2 = ((torch.bitwise_right_shift(max_abs_int32, MBITS_F32)) & 0xFF) - F32_EXP_BIAS
    scale_unbiased = extracted_pow2 - target_max_pow2
    scale_unbiased = torch.clamp(
        scale_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1
    )
    scale_biased = (scale_unbiased + E8M0_EXPONENT_BIAS).to(torch.uint8)
    scale_biased = torch.where(torch.isnan(max_abs), E8M0_EXPONENT_NAN_VAL, scale_biased)
    return scale_biased


def to_mxfp4(x: torch.Tensor, block_size: int = 32):
    """MXFP4 quantization: E2M1 data + E8M0 per-block scales, FLOOR scaling.
    # [FUNCTION DOC] Quantize a contiguous bf16, fp16 or fp32 tensor to MXFP4 (E2M1 data with
    # E8M0 block scales, floor mode). Returns packed uint8 data whose last dimension is halved
    # (first value in the low nibble) and E8M0 scales whose last dimension is divided by the
    # block size. Fails an assertion for other dtypes, a last dimension not divisible by the
    # block size, or non-contiguous input. Does not clamp values after scaling; the converter
    # saturates large magnitudes instead.

    Args:
        x: (..., K) bf16/fp16/fp32, contiguous, K % block_size == 0.
    Returns:
        qdata_packed: uint8, shape (..., K // 2). Two FP4 values per byte
                      (first -> low nibble, second -> high nibble).
        scale: float8_e8m0fnu, shape (..., K // block_size).
    """
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert x.shape[-1] % block_size == 0
    assert x.is_contiguous()

    orig_shape = x.shape
    data_hp = x.reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)
    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    scale_biased = _compute_e8m0_scale_floor(max_abs, F4_E2M1_MAX_POW2)
    scale_fp32 = (torch.bitwise_left_shift(scale_biased.to(torch.int32), MBITS_F32)).view(
        torch.float32
    )
    scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

    data_lp = data_hp / scale_fp32
    data_lp = data_lp.reshape(orig_shape)
    data_lp = _f32_to_floatx_unpacked(data_lp.float(), EBITS_F4_E2M1, MBITS_F4_E2M1)
    data_lp = _pack_uint4(data_lp)

    scale = scale_biased.view(torch.float8_e8m0fnu).squeeze(-1)
    return data_lp, scale


def nvfp4_per_tensor_scale(amax: torch.Tensor) -> torch.Tensor:
    """NVFP4 per-tensor scale: amax / (F8E4M3_MAX * F4_E2M1_MAX) = amax / 2688."""
    # [FUNCTION DOC] Compute the NVFP4 per-tensor scale from an absolute-maximum tensor: convert to
    # fp32 and divide by the product of the E4M3 maximum and the E2M1 maximum (2688).
    return amax.to(torch.float32) / (F8E4M3_MAX * F4_E2M1_MAX)


def to_nvfp4(x: torch.Tensor, block_size: int = 16, per_tensor_scale=None):
    """NVFP4 quantization: E2M1 data + E4M3 per-block scales + optional fp32 per-tensor scale.
    # [FUNCTION DOC] Quantize a contiguous bf16 or fp32 tensor to NVFP4: E2M1 data with E4M3
    # per-block scales (block size must be 16) and an optional fp32 per-tensor scale. The block
    # scale is the block maximum divided by the E2M1 maximum (and by the per-tensor scale if
    # given), clamped to the E4M3 range and cast. Data is multiplied by the reciprocal of the
    # effective scale, clamped to the E2M1 range, converted and packed two per byte. Returns packed
    # data, block scales, and the per-tensor scale (a new tensor holding one when none was given).
    # Fails an assertion for other dtypes, a bad last dimension, non-contiguous input, or a
    # block size other than 16.

    Args:
        x: (..., K) bf16/fp32, contiguous, K % 16 == 0.
        block_size: must be 16.
        per_tensor_scale: scalar fp32 tensor, or None (uses 1.0 / returns unit).
    Returns:
        qdata_packed: uint8, shape (..., K // 2)
        scale: float8_e4m3fn, shape (..., K // 16)
        per_tensor_scale: scalar fp32 tensor (1.0 if None was passed)
    """
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert x.shape[-1] % block_size == 0
    assert x.is_contiguous()
    assert block_size == 16, "NVFP4 requires block_size=16"

    orig_shape = x.shape
    data_hp = x.float().reshape(*orig_shape[:-1], orig_shape[-1] // block_size, block_size)
    max_abs = torch.amax(torch.abs(data_hp), dim=-1)
    block_scale = max_abs / F4_E2M1_MAX

    if per_tensor_scale is None:
        block_scale_fp8 = torch.clamp(block_scale, min=E4M3_EPS, max=F8E4M3_MAX).to(
            torch.float8_e4m3fn
        )
        recip = 1.0 / block_scale_fp8.to(torch.float32)
        returned_pts = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    else:
        scaled = block_scale.to(torch.float32) / per_tensor_scale
        block_scale_fp8 = torch.clamp(scaled, min=E4M3_EPS, max=F8E4M3_MAX).to(torch.float8_e4m3fn)
        recip = (1.0 / per_tensor_scale) / block_scale_fp8.to(torch.float32)
        returned_pts = per_tensor_scale.to(torch.float32)

    data_scaled = data_hp * recip.unsqueeze(-1)
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)
    data_lp = _f32_to_floatx_unpacked(data_scaled.float(), EBITS_F4_E2M1, MBITS_F4_E2M1)
    data_lp = _pack_uint4(data_lp)
    return data_lp, block_scale_fp8, returned_pts


# ---------------------------------------------------------------------------
# torch.compile-wrapped fast paths. Generates fused Triton quant kernels via
# Inductor. dynamic=True avoids recompilation on shape changes.
# ---------------------------------------------------------------------------
to_mx_compiled = torch.compile(to_mx, dynamic=True)
to_mxfp4_compiled = torch.compile(to_mxfp4, dynamic=True)
to_nvfp4_compiled = torch.compile(to_nvfp4, dynamic=True)


def _ceil_div(a, b):
    return (a + b - 1) // b
    # [FUNCTION DOC] HELPER: Integer ceiling division of a by b.


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Swizzle a (H, W) e8m0 scale tensor into the 128x4 blocked layout
    # [FUNCTION DOC] Rearrange a 2D scale tensor (rows by columns) into the 128 by 4 blocked,
    # swizzled layout used by cuBLAS for block-scaled matmul, returning a flat 1D tensor.
    # Rows and columns are padded with zeros up to multiples of 128 and 4 (always copied
    # when compiling; otherwise only when padding is needed), then viewed as tiles, permuted and
    # interleaved by reshaping and transposing so each 512-element tile holds a 32 by 16 pattern.
    # Output size is the padded element count. The input is not modified.
    cuBLAS expects for MXFP8 _scaled_mm. Returns a 1-D flat tensor of size
    32*ceil(H/128) * 16*ceil(W/4)."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if torch.compiler.is_compiling() or (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()
