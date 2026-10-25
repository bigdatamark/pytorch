# [MODULE DOC] Dynamo polyfills for traceback.extract_tb and traceback.clear_frames,
# registered via substitute_in_graph with constant folding allowed. Exports extract_tb and
# clear_frames. Import-time side effect: registers the substitutions.
"""
Python polyfills for traceback
"""

import traceback
from traceback import StackSummary
from types import TracebackType

from ..decorators import substitute_in_graph


__all__ = ["extract_tb", "clear_frames"]


@substitute_in_graph(traceback.extract_tb, can_constant_fold_through=True)
def extract_tb(tb: TracebackType | None, limit: int | None = None) -> StackSummary:
    # [FUNCTION DOC] Polyfill of traceback.extract_tb. Walks the traceback chain via
    # tb_next, collecting each entry's frame_summary (a Dynamo-provided attribute), up to
    # limit entries when limit is truthy (a limit of zero means unlimited, unlike the
    # stdlib). Returns a StackSummary built from the list; empty for a None traceback.
    # KNOWN ISSUE: limit of 0 or negative is treated as no limit, differing from stdlib.
    if tb is None:
        return traceback.StackSummary.from_list([])
    # pyrefly: ignore [implicit-any]
    frame_summary = []
    while tb is not None:
        if limit:
            if len(frame_summary) < limit:
                frame_summary.append(
                    # pyrefly: ignore[missing-attribute]
                    tb.frame_summary
                )
            else:
                break
        else:
            frame_summary.append(tb.frame_summary)  # pyrefly: ignore[missing-attribute]
        tb = tb.tb_next
    return traceback.StackSummary.from_list(frame_summary)


@substitute_in_graph(traceback.clear_frames, can_constant_fold_through=True)
def clear_frames(tb: TracebackType | None) -> None:
    # [FUNCTION DOC] Polyfill of traceback.clear_frames: a no-op that returns None.
    # no-op
    return None
