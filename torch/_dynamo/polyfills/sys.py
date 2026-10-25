# [MODULE DOC] Dynamo polyfills for sys.intern, sys.getrecursionlimit and (Python 3.11+
# only) sys.get_int_max_str_digits / sys.set_int_max_str_digits, registered via
# substitute_in_graph with constant folding allowed. The 3.11 pair is appended to
# __all__ only on supported versions. Import-time side effect: registers substitutions.
"""
Python polyfills for sys
"""

from __future__ import annotations

import sys

from ..decorators import substitute_in_graph


__all__ = [
    "intern",
    "getrecursionlimit",
]


# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(sys.intern, can_constant_fold_through=True)
def intern(string: str, /) -> str:
    # [FUNCTION DOC] Polyfill of sys.intern: returns the given string unchanged (no real
    # interning is performed).
    return string


@substitute_in_graph(sys.getrecursionlimit, can_constant_fold_through=True)
def getrecursionlimit() -> int:
    # [FUNCTION DOC] Polyfill of sys.getrecursionlimit: returns the interpreter's current
    # recursion limit.
    return sys.getrecursionlimit()


if sys.version_info >= (3, 11):

    @substitute_in_graph(sys.get_int_max_str_digits, can_constant_fold_through=True)
    def get_int_max_str_digits() -> int:
        # [FUNCTION DOC] Polyfill of sys.get_int_max_str_digits: returns the current
        # limit on digits for int/str conversion.
        return sys.get_int_max_str_digits()

    @substitute_in_graph(sys.set_int_max_str_digits, can_constant_fold_through=True)
    def set_int_max_str_digits(maxdigits: int) -> None:
        # [FUNCTION DOC] Polyfill of sys.set_int_max_str_digits: sets the process-wide
        # int/str conversion digit limit (global side effect). Raises ValueError for
        # invalid values. Returns nothing.
        sys.set_int_max_str_digits(maxdigits)

    __all__ += ["get_int_max_str_digits", "set_int_max_str_digits"]
