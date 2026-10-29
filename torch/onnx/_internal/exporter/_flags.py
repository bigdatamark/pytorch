# [MODULE DOC] Holds the process-global flag that tells other code the ONNX exporter is running,
# and a decorator that sets it for the duration of a call. The flag starts out false.
"""Internal flags for ONNX export."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable


_is_onnx_exporting = False

# Use ParamSpec to preserve parameter types instead of erasing to Any
_P = ParamSpec("_P")
_R = TypeVar("_R")


def set_onnx_exporting_flag(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator. Wraps func so the module-level exporting flag is true while it
    # runs. Args: func, the callable to wrap. Returns: the wrapper, with func's metadata kept.
    # The wrapper forwards all arguments and the return value; exceptions propagate unchanged.
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Sets the global flag, calls func, and always resets the flag
        # to false afterwards (even on error). Side effect: mutates global state.
        # KNOWN ISSUE: the flag is reset to false, not restored to its previous value, so a
        # nested decorated call clears it early; it is also not thread safe.
        global _is_onnx_exporting
        _is_onnx_exporting = True
        try:
            return func(*args, **kwargs)
        finally:
            # Ensure it resets even if an exception occurs
            _is_onnx_exporting = False

    return wrapper
