# mypy: allow-untyped-defs
# [MODULE DOC] Python bindings for Intel ITT (Instrumentation and Tracing Technology) ranges:
# is_available, range_push, range_pop, mark and a range context manager. Import-time side
# effect: if the C ITT module is missing, installs a stub whose functions raise RuntimeError
# (except is_available, which returns False).
from contextlib import contextmanager
from typing import NoReturn


try:
    from torch._C import _itt
except ImportError:

    class _ITTStub:
        # [CLASS DOC] HELPER: Fallback used when the C ITT module is unavailable. Its
        # rangePush, rangePop and mark all raise RuntimeError; is_available returns False.
        @staticmethod
        def _fail(*args, **kwargs) -> NoReturn:
            # [FUNCTION DOC] HELPER: Ignores arguments and always raises RuntimeError saying
            # ITT is not installed.
            raise RuntimeError(
                "ITT functions not installed. Are you sure you have a ITT build?"
            )

        @staticmethod
        def is_available() -> bool:
            # [FUNCTION DOC] HELPER: Stub availability check; always returns False.
            return False

        rangePush = _fail
        rangePop = _fail
        mark = _fail

    _itt = _ITTStub()  # type: ignore[assignment]


__all__ = ["is_available", "range_push", "range_pop", "mark", "range"]


def is_available():
    # [FUNCTION DOC] Returns True if ITT support is built in, by delegating to the C module
    # (or the stub, which returns False).
    """
    Check if ITT feature is available or not
    """
    return _itt.is_available()


def range_push(msg):
    # [FUNCTION DOC] Pushes a named range onto ITT's nested range stack and returns the
    # zero-based depth of the new range. Raises RuntimeError if ITT is not built in.
    """
    Pushes a range onto a stack of nested range span.  Returns zero-based
    depth of the range that is started.

    Arguments:
        msg (str): ASCII message to associate with range
    """
    return _itt.rangePush(msg)


def range_pop():
    # [FUNCTION DOC] Pops the innermost ITT range and returns the zero-based depth of the
    # range that ended. Raises RuntimeError if ITT is not built in.
    """
    Pops a range off of a stack of nested range spans. Returns the
    zero-based depth of the range that is ended.
    """
    return _itt.rangePop()


def mark(msg):
    # [FUNCTION DOC] Records an instantaneous ITT event labeled with msg. Raises
    # RuntimeError if ITT is not built in.
    """
    Describe an instantaneous event that occurred at some point.

    Arguments:
        msg (str): ASCII message to associate with the event.
    """
    return _itt.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    # [FUNCTION DOC] Context manager: formats msg with the extra positional and keyword
    # arguments, pushes it as an ITT range, runs the body, and always pops the range on exit
    # (including on error). Shadows the builtin range within this module.
    """
    Context manager / decorator that pushes an ITT range at the beginning
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
