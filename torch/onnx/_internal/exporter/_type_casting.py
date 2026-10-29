# [MODULE DOC]
# Helpers for converting the packed torch float4_e2m1fn_x2 dtype to unpacked numpy data
# for ONNX: unpack_float4x2_as_uint8 and get_float4_shape. No import-time side effects.
import numpy as np

import torch


def unpack_float4x2_as_uint8(tensor: torch.Tensor) -> np.ndarray:
    # [FUNCTION DOC]
    # Takes a float4_e2m1fn_x2 tensor (two 4-bit values per byte) and returns a flat-then-
    # reshaped uint8 numpy array with one fp4 code per element. Low nibble becomes the
    # even output index, high nibble the odd index; the result is resized to the unpacked
    # shape (last dimension doubled). Raises AssertionError for any other dtype.
    # Copies data to host (force) without mutating the input tensor.
    """Convert a float4x2 tensor to unpacked uint8 np array."""
    if tensor.dtype != torch.float4_e2m1fn_x2:
        raise AssertionError(f"Expected float4_e2m1fn_x2, got {tensor.dtype}")
    data = tensor.view(torch.uint8).numpy(force=True).flatten()
    result_size = tensor.numel() * 2
    result = np.empty([result_size], dtype=np.uint8)
    array_low = data & np.uint8(0x0F)
    array_high = data & np.uint8(0xF0)
    array_high >>= np.uint8(4)
    result[0::2] = array_low
    result[1::2] = array_high
    result.resize(get_float4_shape(tensor), refcheck=False)
    return result


def get_float4_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    # [FUNCTION DOC]
    # Returns the logical shape of the unpacked fp4 data: the packed tensor shape with the
    # last dimension doubled. Raises AssertionError for a non float4_e2m1fn_x2 dtype.
    """Get the shape of an unpacked float4 tensor.

    The float4_e2m1fn_x2 type is a shell type described in
    https://github.com/pytorch/pytorch/issues/146414.

    the shell dtype is takes up 1 byte per element and semantically represents
    two fp4 values packed into 1 byte. Semantically it represents (*tensor.shape[:-1], tensor.shape[-1]*2)
    fp4 elements.
    """
    if tensor.dtype != torch.float4_e2m1fn_x2:
        raise AssertionError(f"Expected float4_e2m1fn_x2, got {tensor.dtype}")
    return (*tensor.shape[:-1], tensor.shape[-1] * 2)
