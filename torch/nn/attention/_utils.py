# mypy: allow-untyped-defs
# [MODULE DOC]
# HELPER: small internal utilities for scaled dot product attention wrappers: grad check,
# unpadding of flash output, default scale computation, and input validation. Exports nothing.
"""Defines utilities for interacting with scaled_dot_product_attention"""

import math

import torch


__all__: list[str] = []


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: return true if any given tensor requires grad, else false.
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: if the last dimension differs from og_size, return a slice
    # (a view) keeping the first og_size entries of it; otherwise return the input unchanged.
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


def _calculate_scale(head_dim_size: int, scale: float | None) -> float:
    # [FUNCTION DOC] HELPER: return the given scale if not None, else the default
    # softmax scale, one over the square root of head_dim_size (the unpadded head size).
    """
    For FlashAttention we pad the head dimension to be a multiple of 8 so we need to scale the output
    by the original head size and not the padded.
    """
    if scale is not None:
        return scale
    return 1.0 / math.sqrt(head_dim_size)


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    allow_lowp_kv=False,
) -> None:
    # [FUNCTION DOC] HELPER: validate query/key/value for attention. Raises ValueError if
    # dtypes differ (skipped when allow_lowp_kv is true), if devices differ, or if any of
    # them has fewer than 2 dimensions. attn_mask, dropout_p, is_causal and scale are
    # accepted but not checked. Returns nothing.
    if not allow_lowp_kv:
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(
                f"Expected query, key, and value to have the same dtype, "
                f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
                f"and value.dtype: {value.dtype} instead."
            )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
