# [MODULE DOC]
# Provides strtobool, a local replacement for the removed distutils helper that
# converts yes/no style strings to booleans. Defines the accepted true and false
# value sets at import time; no other side effects.

"""Local replacement for ``distutils.util.strtobool`` (removed in Python 3.12)."""

from __future__ import annotations


_TRUE_VALUES = frozenset({"y", "yes", "t", "true", "on", "1"})
_FALSE_VALUES = frozenset({"n", "no", "f", "false", "off", "0"})


def strtobool(val: str) -> bool:
    # [FUNCTION DOC]
    # Case-insensitively maps y, yes, t, true, on, 1 to True and n, no, f, false,
    # off, 0 to False. Raises ValueError for any other input.
    # KNOWN ISSUE: the docstring says it returns 1 or 0 but it returns a bool.
    """Convert a string representation of truth to true (1) or false (0).

    True values are "y", "yes", "t", "true", "on", and "1"; false values
    are "n", "no", "f", "false", "off", and "0".  Raises ValueError if
    "val" is anything else.
    """
    val = val.lower()
    if val in _TRUE_VALUES:
        return True
    if val in _FALSE_VALUES:
        return False
    raise ValueError(f"invalid truth value {val!r}")
