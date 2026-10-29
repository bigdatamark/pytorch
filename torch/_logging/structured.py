# [MODULE DOC] Helpers for emitting structured-trace payloads: string interning, dumping generated
# FX source files, and converting tracebacks into lists of JSON-ready frame dicts.
# Module state: INTERN_TABLE (string to integer id) and DUMPED_FILES (filenames already dumped).
# No import-time side effects beyond defining these globals.
"""
Utilities for converting data types into structured JSON for dumping.
"""

import inspect
import os
import traceback
from collections.abc import Sequence
from typing import Any

import torch._logging._internal


INTERN_TABLE: dict[str, int] = {}


DUMPED_FILES: set[str] = set()


def intern_string(s: str | None) -> int:
    # [FUNCTION DOC] Maps a string to a stable small integer id. None yields -1. The first time a
    # string is seen it gets the next id (current table size), is stored in INTERN_TABLE, and a
    # "str" structured trace event carrying (string, id) is emitted without context.
    # Returns the integer id. Side effects: mutates INTERN_TABLE, emits a trace event.
    if s is None:
        return -1

    r = INTERN_TABLE.get(s)
    if r is None:
        r = len(INTERN_TABLE)
        INTERN_TABLE[s] = r
        torch._logging._internal.trace_structured(
            "str", lambda: (s, r), suppress_context=True
        )
    return r


def dump_file(filename: str) -> None:
    # [FUNCTION DOC] Emits a "dump_file" structured trace whose payload is the source of a
    # generated FX module file. Only filenames containing "eval_with_key" are handled, and each
    # filename is dumped at most once (tracked in DUMPED_FILES). Source is fetched lazily from the
    # FX graph module loader. Returns None.
    if "eval_with_key" not in filename:
        return
    if filename in DUMPED_FILES:
        return
    DUMPED_FILES.add(filename)
    from torch.fx.graph_module import _loader

    torch._logging._internal.trace_structured(
        "dump_file",
        metadata_fn=lambda: {
            "name": filename,
        },
        payload_fn=lambda: _loader.get_source(filename),
    )


def from_traceback(tb: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Converts a sequence of frame summaries into a list of dicts with keys line
    # (line number), name (function name), filename (interned integer id, see intern_string) and
    # loc (source line text). Key names match the C++ combined traceback format. Interning may
    # emit trace events as a side effect.
    # dict naming convention here coincides with
    # python/combined_traceback.cpp
    r = [
        {
            "line": frame.lineno,
            "name": frame.name,
            "filename": intern_string(frame.filename),
            "loc": frame.line,
        }
        for frame in tb
    ]
    return r


def get_user_stack(num_frames: int) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Returns up to num_frames frames of the user's stack as frame dicts. Prefers
    # the stack recorded in the current tracing context (last num_frames entries). Otherwise
    # captures the current stack, walks it from the innermost frame outward to the first frame
    # outside the torch package directory, and returns the num_frames frames ending there; if all
    # frames are inside torch, returns the last num_frames frames.
    # KNOWN ISSUE: the slice start can go negative when fewer than num_frames frames remain,
    # which would wrap around and return a wrong/empty slice.
    from torch._guards import TracingContext
    from torch.utils._traceback import CapturedTraceback

    user_tb = TracingContext.extract_stack()
    if user_tb:
        return from_traceback(user_tb[-1 * num_frames :])

    tb = CapturedTraceback.extract().summary()

    # Filter out frames that are within the torch/ codebase
    torch_filepath = os.path.dirname(inspect.getfile(torch)) + os.path.sep
    for i, frame in enumerate(reversed(tb)):
        if torch_filepath not in frame.filename:
            # Only display `num_frames` frames in the traceback
            filtered_tb = tb[len(tb) - i - num_frames : len(tb) - i]
            return from_traceback(filtered_tb)

    return from_traceback(tb[-1 * num_frames :])


def get_framework_stack(
    num_frames: int = 25, cpp: bool = False
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Captures the current stack (optionally with C++ frames when cpp is true),
    # keeps Python frames from files not in the "uninteresting" set plus C++ frames whose names
    # mention at:: or torch::, and returns the last num_frames of them as frame dicts.
    # KNOWN ISSUE: docstring mentions the user stack too, but only the framework stack is returned.
    """
    Returns the traceback for the user stack and the framework stack
    """
    from torch.fx.experimental.symbolic_shapes import uninteresting_files
    from torch.utils._traceback import CapturedTraceback

    tb = CapturedTraceback.extract(cpp=cpp).summary()
    tb = [
        frame
        for frame in tb
        if (
            (
                frame.filename.endswith(".py")
                and frame.filename not in uninteresting_files()
            )
            or ("at::" in frame.name or "torch::" in frame.name)
        )
    ]

    return from_traceback(tb[-1 * num_frames :])
