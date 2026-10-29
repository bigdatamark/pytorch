# [MODULE DOC] MPS profiling utilities: start/stop OS Signpost tracing (and a profile context
# manager for it) plus Metal GPU capture helpers (enabled check, in-progress check, capture
# context manager). Exports the names in __all__. No import-time side effects.
import contextlib
from collections.abc import Iterator
from typing import Literal

import torch


__all__ = [
    "start",
    "stop",
    "profile",
    "metal_capture",
    "is_metal_capture_enabled",
    "is_capturing_metal",
]


ProfilerMode = Literal["interval", "event", "interval,event"]


def start(mode: ProfilerMode = "interval", wait_until_completed: bool = False) -> None:
    # [FUNCTION DOC] Starts OS Signpost tracing in the MPS backend. The mode string is lowercased
    # with spaces removed and must be interval, event or interval,event (validated natively).
    # If wait_until_completed is true, each GPU operation is awaited, hurting performance.
    r"""Start OS Signpost tracing from MPS backend.

    The generated OS Signposts could be recorded and viewed in
    XCode Instruments Logging tool.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    mode_normalized = mode.lower().replace(" ", "")
    torch._C._mps_profilerStartTrace(  # type: ignore[attr-defined]
        mode_normalized, wait_until_completed
    )


def stop() -> None:
    # [FUNCTION DOC] Stops OS Signpost tracing in the MPS backend.
    r"""Stops generating OS Signpost tracing from MPS backend."""
    torch._C._mps_profilerStopTrace()  # type: ignore[attr-defined]


@contextlib.contextmanager
def profile(
    mode: ProfilerMode = "interval", wait_until_completed: bool = False
) -> Iterator[None]:
    # [FUNCTION DOC] Context manager: starts tracing with the given options, yields, and always
    # stops tracing on exit, even if the body raises.
    # KNOWN ISSUE: start is inside the try block, so if starting fails, stop is still called.
    r"""Context Manager to enabling generating OS Signpost tracing from MPS backend.

    Args:
        mode(str): OS Signpost tracing mode could be "interval", "event",
            or both "interval,event".
            The interval mode traces the duration of execution of the operations,
            whereas event mode marks the completion of executions.
            See document `Recording Performance Data`_ for more info.
        wait_until_completed(bool): Waits until the MPS Stream complete
            executing each encoded GPU operation. This helps generating single
            dispatches on the trace's timeline.
            Note that enabling this option would affect the performance negatively.

    .. _Recording Performance Data:
       https://developer.apple.com/documentation/os/logging/recording_performance_data
    """
    try:
        start(mode, wait_until_completed)
        yield
    finally:
        stop()


def is_metal_capture_enabled() -> bool:
    # [FUNCTION DOC] Returns True if Metal capture is usable (the capture-enabled environment
    # variable was set for the process).
    """Checks if `metal_capture` context manager is usable
    To enable metal capture, set MTL_CAPTURE_ENABLED envvar
    """
    return torch._C._mps_isCaptureEnabled()  # type: ignore[attr-defined, no-any-return]


def is_capturing_metal() -> bool:
    # [FUNCTION DOC] Returns True if a Metal capture is currently in progress.
    """Checks if metal capture is in progress"""
    return torch._C._mps_isCapturing()  # type: ignore[attr-defined, no-any-return]


@contextlib.contextmanager
def metal_capture(fname: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that starts a Metal capture written to the gputrace file
    # named fname, yields, and always stops the capture on exit (after pending MPS work is
    # waited on). Errors (e.g. capture not enabled) come from the native layer.
    """Context manager that enables capturing of Metal calls into gputrace"""
    try:
        torch._C._mps_startCapture(fname)  # type: ignore[attr-defined]
        yield
    finally:
        # _mps_stopCapture waits for work enqueued on MPS streams before
        # stopping the capture, even if the body raised
        torch._C._mps_stopCapture()  # type: ignore[attr-defined]
