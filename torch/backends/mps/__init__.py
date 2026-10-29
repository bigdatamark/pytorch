# [MODULE DOC] Exposes torch.backends.mps queries: whether the build supports MPS (Apple Metal), whether it is usable now, macOS version checks, device name and GPU core count.
# Most queries are cached after first call. No import-time side effects beyond imports.
from functools import lru_cache as _lru_cache
from typing import Optional, TYPE_CHECKING

import torch
from torch.library import Library as _Library


__all__ = [
    "get_core_count",
    "get_name",
    "is_built",
    "is_available",
    "is_macos13_or_newer",
    "is_macos_or_newer",
]


def is_built() -> bool:
    # [FUNCTION DOC] Returns whether the build includes MPS support (not whether a working device exists).
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run on a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps


@_lru_cache
def is_available() -> bool:
    # [FUNCTION DOC] Returns whether MPS is currently usable on this machine; result is cached.
    r"""Return a bool indicating if MPS is currently available."""
    return torch._C._mps_is_available()


@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    # [FUNCTION DOC] Returns whether the running macOS version is at least the given major and minor; result is cached per argument pair.
    r"""Return a bool indicating whether MPS is running on given MacOS or newer."""
    return torch._C._mps_is_on_macos_or_newer(major, minor)


@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    # [FUNCTION DOC] Returns whether macOS is version 13 (with the given minor, default 0) or newer; cached.
    r"""Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_or_newer(13, minor)


@_lru_cache
def get_name() -> str:
    # [FUNCTION DOC] Returns the Metal device name string; cached.
    r"""Return Metal device name"""
    return torch._C._mps_get_name()


@_lru_cache
def get_core_count() -> int:
    # [FUNCTION DOC] Returns the GPU core count as an integer; cached.
    r"""Return GPU core count.

    According to the documentation, one core is comprised of 16 Execution Units.
    One execution Unit has 8 ALUs.
    And one ALU can run 24 threads, i.e. one core is capable of executing 3072 threads concurrently.
    """
    return torch._C._mps_get_core_count()
