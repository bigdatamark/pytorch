# mypy: allow-untyped-defs
# [MODULE DOC] NVTX profiling-range helpers (push/pop, start/end, mark, and a range
# context manager) over torch._C._nvtx. If the extension lacks NVTX, a stub is
# installed at import time whose functions raise RuntimeError when called.
r"""This package adds support for NVIDIA Tools Extension (NVTX) used in profiling."""

from contextlib import contextmanager


try:
    from torch._C import _nvtx
except ImportError:

    class _NVTXStub:
        # [CLASS DOC] HELPER: fallback used when NVTX is not built in; rangePushA,
        # rangePop and markA all map to _fail.
        @staticmethod
        def _fail(*args, **kwargs):
            # [FUNCTION DOC] HELPER: always raises RuntimeError saying NVTX is not installed.
            raise RuntimeError(
                "NVTX functions not installed. Are you sure you have a CUDA build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _nvtx = _NVTXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    # [FUNCTION DOC] Pushes a named NVTX range on the nested stack; returns its zero-based depth.
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _nvtx.rangePushA(msg)


def range_pop():
    # [FUNCTION DOC] Pops the innermost NVTX range; returns the depth of the range ended.
    """Pop a range off of a stack of nested range spans.  Returns the  zero-based depth of the range that is ended."""
    return _nvtx.rangePop()


def range_start(msg) -> int:
    # [FUNCTION DOC] Starts a named NVTX range and returns a handle for range_end; unlike
    # push/pop it may be ended on a different thread.
    """
    Mark the start of a range with string message. It returns a unique handle
    for this range to pass to the corresponding call to rangeEnd().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.rangeStartA(msg)


def range_end(range_id) -> None:
    # [FUNCTION DOC] Ends the range identified by the handle returned from range_start.
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): a unique handle for the start range.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.rangeEnd(range_id)


def _device_range_start(msg: str, stream: int = 0) -> object:
    # [FUNCTION DOC] HELPER: starts a range that is timestamped once prior work on the
    # given CUDA stream id completes; returns an opaque handle for _device_range_end.
    """
    Marks the start of a range with string message.
    It returns an opaque heap-allocated handle for this range
    to pass to the corresponding call to device_range_end().

    A key difference between this and range_start is that the
    range_start marks the range right away, while _device_range_start
    marks the start of the range as soon as all the tasks on the
    CUDA stream are completed.

    Returns: An opaque heap-allocated handle that should be passed to _device_range_end().

    Args:
        msg (str): ASCII message to associate with the range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    return _nvtx.deviceRangeStart(msg, stream)


def _device_range_end(range_handle: object, stream: int = 0) -> None:
    # [FUNCTION DOC] HELPER: ends a device range once prior work on the stream completes.
    """
    Mark the end of a range for a given range_handle as soon as all the tasks
    on the CUDA stream are completed.

    Args:
        range_handle: a unique handle for the start range.
        stream (int): CUDA stream id.
    """
    # pyrefly: ignore [missing-attribute]
    _nvtx.deviceRangeEnd(range_handle, stream)


def mark(msg):
    # [FUNCTION DOC] Records an instantaneous NVTX event with the given message.
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _nvtx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    # [FUNCTION DOC] Context manager: formats msg with the extra arguments, pushes an
    # NVTX range, yields, and always pops the range on exit (even on exception).
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
