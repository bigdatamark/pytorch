# [MODULE DOC]
# Polyfills for builtins (all, any, enumerate, sum, typing.cast) registered via
# substitute_in_graph, plus iterator helper classes and factory functions
# (sequence, reversed-sequence and callable iterators) that mirror CPython
# iterator semantics for tracing.
"""
Python polyfills for builtins
"""

from __future__ import annotations

import builtins
import functools
import operator
import sys
import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "all",
    "any",
    "cast",
    "enumerate",
    "sum",
]


_T = TypeVar("_T")


@substitute_in_graph(builtins.all, can_constant_fold_through=True)
def all(iterable: Iterable[object], /) -> bool:
    # [FUNCTION DOC] True if every element is truthy (short-circuits on the first falsy one); True for empty.
    for elem in iterable:
        if not elem:
            return False
    return True


@substitute_in_graph(builtins.any, can_constant_fold_through=True)
def any(iterable: Iterable[object], /) -> bool:
    # [FUNCTION DOC] True if any element is truthy (short-circuits); False for empty.
    for elem in iterable:
        if elem:
            return True
    return False


@substitute_in_graph(builtins.enumerate, is_embedded_type=True)  # type: ignore[arg-type]
def enumerate(iterable: Iterable[_T], start: int = 0) -> Iterable[tuple[int, _T]]:
    # [FUNCTION DOC] Generator yielding (index, element) pairs starting from
    # start. Raises TypeError if start is not an int.
    if not isinstance(start, int):
        raise TypeError(
            f"{type(start).__name__!r} object cannot be interpreted as an integer"
        )

    for x in iterable:
        yield start, x
        start += 1


@substitute_in_graph(builtins.sum, can_constant_fold_through=True)  # type: ignore[arg-type]
def sum(iterable: Iterable[_T], /, start: _T = 0) -> _T:  # type: ignore[assignment]
    # [FUNCTION DOC] Sum the iterable by folding with addition from start (default 0); returns the total.
    return functools.reduce(operator.add, iterable, start)


# TODO(guilhermeleobas): Implement this iterator as a VariableTracker to see if
# it is faster than tracing through it
class _CallableIterator:
    # [CLASS DOC] HELPER: Iterator for the two-argument form of iter: calls fn
    # with no arguments on each step and stops when the result equals the
    # sentinel. State: fn, sentinel, exhausted flag (once set, always stops).
    def __init__(self, fn, sentinel):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Store fn and sentinel; start not exhausted.
        self.fn = fn
        self.sentinel = sentinel
        self.exhausted = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Return self.
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Call fn and return its result; raises
        # StopIteration if already exhausted (including re-entrantly during the
        # call) or if the result equals the sentinel (which marks exhausted).
        if self.exhausted:
            raise StopIteration

        # The iterator created in this case will call object with no arguments
        # for each call to its __next__() method;
        r = self.fn()

        # CPython calliter_iternext (Objects/iterobject.c) re-reads it_callable /
        # it_sentinel after the call returns. If a reentrant next() exhausted
        # this iterator during fn() (gh-101892), it_sentinel is now NULL, so the
        # sentinel comparison is skipped and StopIteration is raised regardless
        # of the returned value.
        if self.exhausted:
            raise StopIteration

        # If the value returned is equal to sentinel, StopIteration will be raised
        if r == self.sentinel:
            self.exhausted = True
            raise StopIteration

        # otherwise the value will be returned.
        return r


class _SequenceIterator:
    # [CLASS DOC] HELPER: Iterator over an object that only defines item access,
    # mirroring CPython's sequence iterator. State: iterable, index, exhausted.
    def __init__(self, iterable) -> None:
        # [FUNCTION DOC] HELPER: Store the sequence; index starts at 0, not exhausted.
        self.iterable = iterable
        self.index = 0
        self.exhausted = False

    def __iter__(self) -> _SequenceIterator:
        # [FUNCTION DOC] HELPER: Return self.
        return self

    def __next__(self) -> object:
        # [FUNCTION DOC] HELPER: Return the item at the current index and
        # advance. IndexError/StopIteration from item access marks exhausted and
        # raises StopIteration. Raises OverflowError if the index reached the
        # maximum machine size.
        if self.exhausted:
            raise StopIteration

        # CPython iter_iternext (Objects/iterobject.c): the index counter is a
        # Py_ssize_t, so once it reaches PY_SSIZE_T_MAX (sys.maxsize) it cannot
        # advance and every subsequent next() raises OverflowError.
        if self.index == sys.maxsize:
            raise OverflowError("iter index too large")

        try:
            result = self.iterable.__getitem__(self.index)
            self.index += 1
            return result
        except (IndexError, StopIteration):
            self.exhausted = True
            raise StopIteration from None

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC] HELPER: Restore the index (negative clamped to 0), only if not exhausted.
        # CPython iter_setstate (Objects/iterobject.c): a negative index is
        # clamped to 0, and the state is only applied while the iterator is
        # live (it_seq != NULL), i.e. not yet exhausted.
        if not self.exhausted:
            self.index = max(state, 0)


def sequence_iterator(iterable) -> Iterable[object]:
    # [FUNCTION DOC] Build a sequence iterator for an object with item access;
    # raises TypeError if the object has no item-access method.
    if hasattr(iterable, "__getitem__"):
        return _SequenceIterator(iterable)
    raise TypeError(f"'{type(iterable)}' object is not iterable")


class _ReversedSequenceIterator:
    # [CLASS DOC] HELPER: Iterator walking a sequence backwards by item access,
    # starting at length minus one. State: seq, index (negative means done).
    # Mirrors CPython's reversed object (Objects/enumobject.c). Walks the
    # sequence backwards via __getitem__, starting from len(seq) - 1.
    def __init__(self, seq) -> None:
        # [FUNCTION DOC] HELPER: Store the sequence and set index to its last position (calls len).
        self.seq = seq
        self.index = len(seq) - 1

    def __iter__(self) -> _ReversedSequenceIterator:
        # [FUNCTION DOC] HELPER: Return self.
        return self

    def __next__(self) -> object:
        # [FUNCTION DOC] HELPER: Return the item at the current index and move
        # back one; raises StopIteration when past the start or when item access
        # raises IndexError/StopIteration (which also ends iteration permanently).
        if self.index < 0:
            raise StopIteration
        try:
            result = self.seq[self.index]
        except (IndexError, StopIteration):
            self.index = -1
            raise StopIteration from None
        self.index -= 1
        return result


def reversed_sequence_iterator(seq) -> Iterable[object]:
    # [FUNCTION DOC] Return a backwards iterator over the sequence.
    return _ReversedSequenceIterator(seq)


def callable_iterator(fn, sentinel, /):
    # [FUNCTION DOC] Implements the two-argument form of iter: raises TypeError
    # if fn is not callable, else returns an iterator calling fn until it yields the sentinel.
    # If the second argument, sentinel, is given, then object must be a
    # callable object.
    if not isinstance(fn, Callable):  # type: ignore[arg-type]
        raise TypeError("iter(v, w): v must be a callable")

    return _CallableIterator(fn, sentinel)


@substitute_in_graph(typing.cast, can_constant_fold_through=True)
def cast(typ: type, val: _T) -> _T:  # type: ignore[type-var]
    # [FUNCTION DOC] Type-checking no-op: returns val unchanged.
    return val
