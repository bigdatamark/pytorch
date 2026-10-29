# [MODULE DOC] Small helpers for the caching system: a fault-tolerant LRU cache decorator and a tensor
# metadata encoder with its typed mapping.
"""Utility functions

This module provides helper functions for LRU caching decorators used
throughout the caching system.
"""

from collections.abc import Callable
from functools import lru_cache, wraps
from typing_extensions import ParamSpec, TypedDict, TypeVar

from torch import Tensor


# Type specification for function parameters
P = ParamSpec("P")
# Type variable for function return values
R = TypeVar("R")


def _lru_cache(fn: Callable[P, R]) -> Callable[P, R]:
    # [FUNCTION DOC] HELPER: Decorator wrapping a function in an LRU cache (size 64, typed). If the cached call raises a
    # TypeError (e.g. unhashable arguments) the original function is called uncached.
    # KNOWN ISSUE: a TypeError raised inside the function itself also triggers a second, uncached call.
    """LRU cache decorator with TypeError fallback.

    Provides LRU caching with a fallback mechanism that calls the original
    function if caching fails due to unhashable arguments. Uses a cache
    size of 64 with typed comparison.

    Args:
        fn: The function to be cached.

    Returns:
        A wrapper function that attempts caching with fallback to original function.
    """
    cached_fn = lru_cache(maxsize=64, typed=True)(fn)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # [FUNCTION DOC] HELPER: Calls the cached function; on TypeError falls back to calling the original.
        try:
            return cached_fn(*args, **kwargs)
        except TypeError:
            return fn(*args, **kwargs)

    return wrapper


class EncodedTensor(TypedDict):
    # [CLASS DOC] Typed mapping of tensor metadata: shape, stride and dtype string.
    """TypedDict for encoded tensor metadata."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


def _encode_tensor(t: Tensor) -> EncodedTensor:
    # [FUNCTION DOC] HELPER: Returns an EncodedTensor with the tensor's shape and stride as tuples and dtype as a string.
    """Encode a tensor's metadata into a JSON-serializable dict.

    Args:
        t: PyTorch tensor to encode

    Returns:
        Dict containing shape, stride, and dtype information
    """
    return EncodedTensor(
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        dtype=str(t.dtype),
    )
