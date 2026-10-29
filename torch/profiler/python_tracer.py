# [MODULE DOC] HELPER: Utility for the profiler's Python tracer that computes the list of
# filesystem path prefixes (site-packages, sys.path entries, user base and site dirs, and the
# directory containing the torch package) used to shorten file paths in traces.
import os
import site
import sys

import torch


def _prefix_regex() -> list[str]:
    # [FUNCTION DOC] HELPER: Collects site-packages, sys.path, user base/site dirs and the
    # parent of the torch install dir, makes them absolute and unique, sorts them in reverse
    # order, and returns each with a trailing path separator. Raises an error if any
    # entry is not a string. Reads interpreter state only.
    # KNOWN ISSUE: name says regex but returns plain path prefix strings.
    raw_paths = (
        site.getsitepackages()
        + sys.path
        + [site.getuserbase()]
        + [site.getusersitepackages()]
        + [os.path.dirname(os.path.dirname(torch.__file__))]
    )

    path_prefixes = sorted({os.path.abspath(i) for i in raw_paths}, reverse=True)
    if not all(isinstance(i, str) for i in path_prefixes):
        raise AssertionError("all path_prefixes must be strings")
    return [i + os.sep for i in path_prefixes]
