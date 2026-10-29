# [MODULE DOC] Exception types and helpers for DCP failures. Exports CheckpointException, which aggregates
# per-rank exceptions with their stack summaries; also defines the WRAPPED_EXCEPTION alias and
# helpers to wrap an exception for cross-rank gathering and to recognize a wrapped pair. No import-time side effects.
import traceback as tb
from typing import Any


WRAPPED_EXCEPTION = tuple[BaseException, tb.StackSummary]

__all__ = ["CheckpointException"]


def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    # [FUNCTION DOC] HELPER: Pairs an exception with its extracted traceback summary. Clears the private code object
    # of each frame summary (present on newer Python versions) by mutating it in place so the pair can be pickled
    # for gathering across ranks. Returns the tuple.
    summary = tb.extract_tb(exc.__traceback__)
    # Python 3.13+ stores bytecode objects in FrameSummary._code,
    # which cannot be pickled. Clear them so gather_object succeeds
    # and the real exception is reported instead of a misleading
    # "cannot pickle code objects" TypeError.
    for frame in summary:
        if hasattr(frame, "_code"):
            object.__setattr__(frame, "_code", None)
    return (exc, summary)


def _is_wrapped_exception(obj: Any) -> bool:
    # [FUNCTION DOC] HELPER: Returns true only for a two-element tuple of an exception and a traceback StackSummary.
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:
        return False
    return isinstance(obj[0], BaseException) and isinstance(obj[1], tb.StackSummary)


class CheckpointException(BaseException):
    # [CLASS DOC] Exception (derives from BaseException, not Exception) raised when a checkpoint save or load fails
    # on one or more ranks. State: a mapping from rank to a wrapped (exception, stack summary) pair.
    """Exception raised if failure was detected as part of a checkpoint load or save."""

    def __init__(self, msg: str, failures: dict[int, WRAPPED_EXCEPTION]):
        # [FUNCTION DOC] Stores the failures mapping and passes the message and mapping to the base exception.
        super().__init__(msg, failures)
        self._failures = failures

    @property
    def failures(self) -> dict[int, WRAPPED_EXCEPTION]:
        # [FUNCTION DOC] Property returning the mapping from rank to wrapped exception pair.
        """Return a dictionary mapping node ranks to their associated exceptions in case of failure."""
        return self._failures

    def __str__(self) -> str:
        # [FUNCTION DOC] Builds a text listing the failing ranks, and for each rank its formatted traceback (when present)
        # and the exception type and message. Uses a local variable named str that shadows the builtin.
        str = f"CheckpointException ranks:{self._failures.keys()}\n"
        for rank, exc_pair in self._failures.items():
            exc, trace = exc_pair
            str += f"Traceback (most recent call last): (RANK {rank})\n"
            if trace is not None:
                str += "".join(tb.format_list(trace))
            str += "".join(tb.format_exception_only(type(exc), value=exc))
        return str
