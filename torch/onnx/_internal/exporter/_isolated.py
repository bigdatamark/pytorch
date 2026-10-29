# [MODULE DOC] Runs functions in a forked subprocess so a crash (segfault) in native code does
# not kill the main process. Provides safe_call; on Windows it falls back to a direct call.
"""Isolated calls to methods that may segfault."""

from __future__ import annotations

import multiprocessing
import os
import warnings
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack


if TYPE_CHECKING:
    from collections.abc import Callable


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Ts = TypeVarTuple("_Ts")

_IS_WINDOWS = os.name == "nt"


def _call_function_and_return_exception(
    func: Callable[[Unpack[_Ts]], _R], args: tuple[Unpack[_Ts]], kwargs: dict[str, Any]
) -> _R | Exception:
    # [FUNCTION DOC] HELPER: Pool worker entry point. Calls func with args and kwargs and returns
    # the result, or returns (not raises) any Exception so it can be sent back across processes.
    """Call function and return an exception if there is one."""

    try:
        return func(*args, **kwargs)
    except Exception as e:
        return e


def safe_call(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    # [FUNCTION DOC] Calls func in a one-worker forked process pool and returns its result.
    # Re-raises, in the caller, an exception returned by the worker. On Windows (no fork) it warns
    # and calls func directly in-process. Raises multiprocessing.TimeoutError if no result in 5s.
    # KNOWN ISSUE: the 5 second timeout is hard coded, so slow functions fail; a function whose
    # legitimate return value is an Exception instance is raised instead of returned.
    # A segfault in the child leaves the wait to time out rather than raising a clear error.
    """Call a function in a separate process.

    Args:
        func: The function to call.
        args: The positional arguments to pass to the function.
        kwargs: The keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Raises:
        Exception: If the function raised an exception.
    """
    if _IS_WINDOWS:
        # On Windows, we cannot create a new process with fork.
        warnings.warn(
            f"A new process is not created for {func} on Windows.", stacklevel=1
        )
        return func(*args, **kwargs)

    with multiprocessing.get_context("fork").Pool(1) as pool:
        # It is important to fork a process here to prevent the main logic from
        # running again when the user does not place it under a `if __name__ == "__main__":`
        # block.
        result = pool.apply_async(
            _call_function_and_return_exception, (func, args, kwargs)
        )
        result = result.get(timeout=5)
    if isinstance(result, Exception):
        raise result
    return result
