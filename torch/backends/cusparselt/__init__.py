# [MODULE DOC] Exposes cuSPARSELt (NVIDIA sparse library) queries: version, availability, and maximum algorithm id.
# Import-time side effects: tries to import the compiled _cusparselt extension (falls back to None) and defines module-level caches for the version and max algorithm id.
# Exports version, is_available, get_max_alg_id.
from typing import Optional

import torch


__all__ = [
    "version",
    "is_available",
    "get_max_alg_id",
]

try:
    from torch._C import _cusparselt
except ImportError:
    _cusparselt = None  # type: ignore[assignment]

__cusparselt_version: int | None = None
__MAX_ALG_ID: int | None = None

if _cusparselt is not None:

    def _init() -> bool:
        # [FUNCTION DOC] HELPER: Lazily initializes cached cuSPARSELt version and max algorithm id (mapping known versions 400, 502, 602 to 4, 5, 37; other versions leave it unset) and returns True. Mutates module globals.
        global __cusparselt_version
        global __MAX_ALG_ID
        if __cusparselt_version is None:
            # pyrefly: ignore [missing-attribute]
            __cusparselt_version = _cusparselt.getVersionInt()
            if __cusparselt_version == 400:
                __MAX_ALG_ID = 4
            elif __cusparselt_version == 502:
                __MAX_ALG_ID = 5
            elif __cusparselt_version == 602:
                __MAX_ALG_ID = 37
        return True

else:

    def _init() -> bool:
        # [FUNCTION DOC] HELPER: Fallback used when the extension is missing; always returns False.
        return False


def version() -> int | None:
    # [FUNCTION DOC] Returns the cuSPARSELt version integer, or None if the extension is unavailable.
    """Return the version of cuSPARSELt"""
    if not _init():
        return None
    return __cusparselt_version


def is_available() -> bool:
    # [FUNCTION DOC] Returns the compiled-in flag telling whether cuSPARSELt support is present.
    r"""Return a bool indicating if cuSPARSELt is currently available."""
    return torch._C._has_cusparselt


def get_max_alg_id() -> int | None:
    # [FUNCTION DOC] Returns the maximum algorithm id for the detected cuSPARSELt version, or None if the extension is unavailable or the version is not a known one.
    if not _init():
        return None
    return __MAX_ALG_ID
