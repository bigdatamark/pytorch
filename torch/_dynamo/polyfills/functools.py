# [MODULE DOC]
# Polyfills for functools.cmp_to_key and functools.reduce, registered with
# substitute_in_graph at import time so Dynamo can trace through them.
# Also defines the helper key-wrapper class used by cmp_to_key.
"""
Python polyfills for functools
"""

import functools
from collections.abc import Callable, Iterable
from functools import _initial_missing  # type: ignore[attr-defined]
from typing import Any, TypeVar

from ..decorators import substitute_in_graph


__all__ = ["cmp_to_key", "reduce"]


_T = TypeVar("_T")
_U = TypeVar("_U")


class _KeyWrapper:
    # [CLASS DOC] HELPER: Wraps one object with a two-argument comparison
    # function; attributes mycmp and obj. Rich comparisons call mycmp on the
    # wrapped objects and compare the result with zero. Unhashable.
    def __init__(self, mycmp: Callable[[Any, Any], int], obj: Any) -> None:
        # [FUNCTION DOC] HELPER: Store the comparison function and the object.
        self.mycmp = mycmp
        self.obj = obj

    def __lt__(self, other: "_KeyWrapper") -> bool:
        # [FUNCTION DOC] HELPER: True if mycmp of the two wrapped objects is negative.
        return self.mycmp(self.obj, other.obj) < 0

    def __gt__(self, other: "_KeyWrapper") -> bool:
        # [FUNCTION DOC] HELPER: True if mycmp of the two wrapped objects is positive.
        return self.mycmp(self.obj, other.obj) > 0

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: True if mycmp gives zero; raises AttributeError
        # if other has no obj attribute.
        return self.mycmp(self.obj, other.obj) == 0  # type: ignore[attr-defined]

    def __le__(self, other: "_KeyWrapper") -> bool:
        # [FUNCTION DOC] HELPER: True if mycmp of the two wrapped objects is <= 0.
        return self.mycmp(self.obj, other.obj) <= 0

    def __ge__(self, other: "_KeyWrapper") -> bool:
        # [FUNCTION DOC] HELPER: True if mycmp of the two wrapped objects is >= 0.
        return self.mycmp(self.obj, other.obj) >= 0

    __hash__ = None  # type: ignore[assignment]


# Reference: https://docs.python.org/3/library/functools.html#functools.cmp_to_key
@substitute_in_graph(functools.cmp_to_key, skip_signature_check=True)
def cmp_to_key(mycmp: Callable[[Any, Any], int]) -> Callable[[Any], _KeyWrapper]:
    # [FUNCTION DOC] Given an old-style comparison function, return a callable
    # (a partial of the key wrapper class) that maps an object to a sortable key.
    return functools.partial(_KeyWrapper, mycmp)


# Reference: https://docs.python.org/3/library/functools.html#functools.reduce
@substitute_in_graph(functools.reduce)
def reduce(
    function: Callable[[_U, _T], _U],
    iterable: Iterable[_T],
    initial: _U = _initial_missing,  # type: ignore[assignment]
    /,
) -> _U:
    # [FUNCTION DOC] Left-fold the iterable with a two-argument function. If no
    # initial value is given, the first element seeds the fold; an empty
    # iterable then raises TypeError. Returns the accumulated value (the initial
    # value for an empty iterable when one is given).
    it = iter(iterable)

    value: _U
    if initial is _initial_missing:
        try:
            value = next(it)  # type: ignore[assignment]
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value",
            ) from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value
