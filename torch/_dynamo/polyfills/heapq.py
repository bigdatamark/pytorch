# [MODULE DOC]
# Polyfills for heapq. Imports a fresh pure-Python copy of heapq (C accelerator
# blocked) at import time as py_heapq, and registers traceable substitutes that
# delegate to it. Import-time side effect: temporarily manipulates sys.modules.
"""
Python polyfills for heapq
"""

from __future__ import annotations

import heapq
import importlib
import sys
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from types import ModuleType


_T = TypeVar("_T")


# Partially copied from CPython test/support/import_helper.py
# https://github.com/python/cpython/blob/bb8791c0b75b5970d109e5557bfcca8a578a02af/Lib/test/support/import_helper.py
def _save_and_remove_modules(names: set[str]) -> dict[str, ModuleType]:
    # [FUNCTION DOC] HELPER: Remove the named modules and their submodules from
    # sys.modules and return the removed mapping (name to module). Mutates sys.modules.
    orig_modules = {}
    prefixes = tuple(name + "." for name in names)
    for modname in list(sys.modules):
        if modname in names or modname.startswith(prefixes):
            orig_modules[modname] = sys.modules.pop(modname)
    return orig_modules


def import_fresh_module(name: str, blocked: list[str]) -> ModuleType:
    # [FUNCTION DOC] HELPER: Import a fresh copy of module name while the
    # blocked modules are made unimportable (set to None in sys.modules), then
    # restore the original sys.modules entries. Returns the fresh module; import
    # errors propagate.
    # Keep track of modules saved for later restoration as well
    # as those which just need a blocking entry removed
    names = {name, *blocked}
    orig_modules = _save_and_remove_modules(names)
    for modname in blocked:
        sys.modules[modname] = None  # type: ignore[assignment]

    try:
        return importlib.import_module(name)
    finally:
        _save_and_remove_modules(names)
        sys.modules.update(orig_modules)


# Import the pure Python heapq module, blocking the C extension
py_heapq = import_fresh_module("heapq", blocked=["_heapq"])


__all__ = [
    "_heapify_max",
    "_heappop_max",
    "_heapreplace_max",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
    "merge",
    "nlargest",
    "nsmallest",
]


@substitute_in_graph(heapq._heapify_max)
def _heapify_max(heap: list[_T], /) -> None:
    # [FUNCTION DOC] Turn the list into a max-heap in place via the pure-Python heapq.
    return py_heapq._heapify_max(heap)


@substitute_in_graph(heapq._heappop_max)  # type: ignore[attr-defined]
def _heappop_max(heap: list[_T]) -> _T:
    # [FUNCTION DOC] Pop and return the largest item of a max-heap (mutates the list).
    return py_heapq._heappop_max(heap)


@substitute_in_graph(heapq._heapreplace_max)  # type: ignore[attr-defined]
def _heapreplace_max(heap: list[_T], item: _T) -> _T:
    # [FUNCTION DOC] Pop and return the largest item of a max-heap, then push item (mutates the list).
    return py_heapq._heapreplace_max(heap, item)


@substitute_in_graph(heapq.heapify)
def heapify(heap: list[_T], /) -> None:
    # [FUNCTION DOC] Turn the list into a min-heap in place.
    return py_heapq.heapify(heap)


@substitute_in_graph(heapq.heappop)
def heappop(heap: list[_T], /) -> _T:
    # [FUNCTION DOC] Pop and return the smallest item of a min-heap (mutates the list; raises on empty).
    return py_heapq.heappop(heap)


@substitute_in_graph(heapq.heappush)
def heappush(heap: list[_T], item: _T) -> None:
    # [FUNCTION DOC] Push item onto a min-heap, keeping the heap invariant (mutates the list).
    return py_heapq.heappush(heap, item)


@substitute_in_graph(heapq.heappushpop)
def heappushpop(heap: list[_T], item: _T) -> _T:
    # [FUNCTION DOC] Push item then pop and return the smallest item, more efficiently than separate calls.
    return py_heapq.heappushpop(heap, item)


@substitute_in_graph(heapq.heapreplace)
def heapreplace(heap: list[_T], item: _T) -> _T:
    # [FUNCTION DOC] Pop and return the smallest item, then push item (raises on empty heap).
    return py_heapq.heapreplace(heap, item)


@substitute_in_graph(heapq.merge)  # type: ignore[arg-type]
def merge(*iterables, key=None, reverse=False):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Lazily merge several sorted inputs into one sorted iterator;
    # optional key function and reverse flag.
    return py_heapq.merge(*iterables, key=key, reverse=reverse)


@substitute_in_graph(heapq.nlargest)  # type: ignore[arg-type]
def nlargest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Return a list of the n largest elements of the iterable, in descending order; optional key.
    return py_heapq.nlargest(n, iterable, key=key)


@substitute_in_graph(heapq.nsmallest)  # type: ignore[arg-type]
def nsmallest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Return a list of the n smallest elements of the iterable, in ascending order; optional key.
    return py_heapq.nsmallest(n, iterable, key=key)
