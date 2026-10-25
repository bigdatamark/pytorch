# [MODULE DOC]
# Polyfill for the C helper _collections._count_elements (used by Counter).
# Registers a pure-Python, traceable substitute via substitute_in_graph, only
# when the _collections C module is importable. Import-time side effect: the
# substitution registration; __all__ lists the name only if registration ran.
"""
Python polyfills for builtins
"""

from collections.abc import Iterable, MutableMapping
from typing import TypeVar

from ..decorators import substitute_in_graph


__all__ = []


T = TypeVar("T")


try:
    import _collections  # type: ignore[import-not-found]

    @substitute_in_graph(_collections._count_elements)
    def _count_elements(
        mapping: MutableMapping[T, int],
        iterable: Iterable[T],
    ) -> None:
        # [FUNCTION DOC] Increment, in the given mutable mapping, the count of
        # each element of the iterable (missing keys start at zero). Mutates the
        # mapping in place; returns nothing. Errors from hashing or iteration
        # propagate.
        "Tally elements from the iterable."
        mapping_get = mapping.get
        for elem in iterable:
            mapping[elem] = mapping_get(elem, 0) + 1

    __all__.append("_count_elements")

except ImportError:
    pass
