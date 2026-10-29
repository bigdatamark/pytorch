# [MODULE DOC] Conversion of tensors and scalars into numpy arrays for TensorBoard writers.
# Contains make_np (public) and _prepare_pytorch (helper). No import-time side effects.
"""This module converts objects into numpy array."""

import numpy as np

import torch


def make_np(x: torch.Tensor) -> np.ndarray:
    # [FUNCTION DOC] Convert a numpy array, scalar or torch tensor into a numpy array.
    # Arrays are returned unchanged, scalars become a one-element array, meta tensors yield a
    # random one-element array, other tensors go through _prepare_pytorch. Raises
    # NotImplementedError for any other input type. No side effects.
    """
    Convert an object into numpy array.

    Args:
      x: An instance of torch tensor

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        if x.device.type == "meta":
            return np.random.randn(1)
        return _prepare_pytorch(x)
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or torch tensor are expected."
    )


def _prepare_pytorch(x: torch.Tensor) -> np.ndarray:
    # [FUNCTION DOC] HELPER: Convert a tensor to numpy; bfloat16 is first cast to float16
    # (numpy lacks bfloat16), then the tensor is detached and moved to the CPU. Returns the array.
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)
    # pyrefly: ignore [bad-assignment]
    x = x.detach().cpu().numpy()
    # pyrefly: ignore [bad-return]
    return x
