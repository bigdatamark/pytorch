# [MODULE DOC] Helpers to detect whether the Meta Triton automatic warp
# specialization (autoWS) feature exists and is switched on. No import-time side
# effects; triton is imported lazily inside the functions.

"""Utils for Meta Triton autoWS."""

from __future__ import annotations

import functools


@functools.cache
def has_meta_ws() -> bool:
    # [FUNCTION DOC] Returns True if triton can be imported and its nvidia knobs expose
    # the use_meta_ws attribute; False on import failure. Result is cached.
    """Whether Meta Triton autoWS is available."""
    try:
        from triton import knobs
    except ImportError:
        return False
    return hasattr(getattr(knobs, "nvidia", None), "use_meta_ws")


def meta_ws_enabled() -> bool:
    # [FUNCTION DOC] Returns False if autoWS is unavailable; otherwise returns the
    # boolean value of the triton nvidia use_meta_ws knob (not cached).
    """Whether Meta Triton autoWS is available and enabled via its knob."""
    if not has_meta_ws():
        return False
    from triton import knobs

    return bool(knobs.nvidia.use_meta_ws)
