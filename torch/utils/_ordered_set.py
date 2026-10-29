# [MODULE DOC] Defines OrderedSet, a mutable set that remembers insertion order, plus the
# type variables used for its annotations. Exports only OrderedSet. No import-time side effects.
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    MutableSet,
    Reversible,
    Set as AbstractSet,
)
from typing import Any, cast, TypeVar


T = TypeVar("T", bound=Hashable)
T_co = TypeVar("T_co", bound=Hashable, covariant=True)

__all__ = ["OrderedSet"]


class OrderedSet(MutableSet[T], Reversible[T]):
    # [CLASS DOC] Set of hashable elements iterated in insertion order. State: _dict, a mapping
    # whose keys are the elements (values unused), which gives ordering and fast membership.
    # Re-adding an existing element keeps its original position. Implements the abstract set
    # methods directly and overrides several operators for speed and for returning OrderedSet.
    # Pickles as a list of elements.
    """
    Insertion ordered set, similar to OrderedDict.
    """

    __slots__ = ("_dict",)

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        # [FUNCTION DOC] Builds the set from an optional iterable, preserving first-seen order
        # and dropping duplicates. Raises TypeError for unhashable elements.
        self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}

    @staticmethod
    def _from_dict(dict_inp: dict[T, None]) -> OrderedSet[T]:
        # [FUNCTION DOC] HELPER: Wraps the given mapping (not copied) as an OrderedSet.
        s: OrderedSet[T] = OrderedSet()
        s._dict = dict_inp
        return s

    #
    # Required overridden abstract methods
    #
    def __contains__(self, elem: object) -> bool:
        # [FUNCTION DOC] Membership test; raises TypeError for unhashable input.
        return elem in self._dict

    def __iter__(self) -> Iterator[T]:
        # [FUNCTION DOC] Iterates elements in insertion order.
        return iter(self._dict)

    def __len__(self) -> int:
        # [FUNCTION DOC] Number of elements.
        return len(self._dict)

    def __reversed__(self) -> Iterator[T]:
        # [FUNCTION DOC] Iterates elements from newest to oldest.
        return reversed(self._dict)

    def add(self, elem: T) -> None:
        # [FUNCTION DOC] Adds the element at the end if absent; existing ones keep position.
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        # [FUNCTION DOC] Removes the element if present; silently does nothing otherwise.
        self._dict.pop(elem, None)

    def clear(self) -> None:
        # [FUNCTION DOC] Removes all elements.
        # overridden because MutableSet impl is slow
        self._dict.clear()

    # Unimplemented set() methods in _collections_abc.MutableSet

    @classmethod
    def _wrap_iter_in_set(cls, other: Any) -> Any:
        # [FUNCTION DOC] HELPER: Returns other converted to an instance of this class if it is a
        # non-set iterable; any other object is returned unchanged.
        """
        Wrap non-Set Iterables in OrderedSets

        Some of the magic methods are more strict on input types than
        the public apis, so we need to wrap inputs in sets.
        """

        if not isinstance(other, AbstractSet) and isinstance(other, Iterable):
            return cls(other)
        else:
            return other

    def pop(self) -> T:
        # [FUNCTION DOC] Removes and returns the most recently added element (last in order).
        # Raises KeyError if the set is empty. KNOWN ISSUE: the built-in set pops an arbitrary
        # element; this removes from the end, which may surprise callers expecting queue order.
        if not self:
            raise KeyError("pop from an empty set")
        return self._dict.popitem()[0]

    def copy(self) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a shallow copy with the same order.
        return OrderedSet._from_dict(self._dict.copy())

    def difference(self, *others: Iterable[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a new set of the elements not present in any of the others.
        res = self.copy()
        res.difference_update(*others)
        return res

    def difference_update(self, *others: Iterable[T]) -> None:
        # [FUNCTION DOC] Removes, in place, all elements found in any of the others.
        for other in others:
            self -= other  # type: ignore[arg-type]

    def update(self, *others: Iterable[T]) -> None:
        # [FUNCTION DOC] Adds, in place, all elements from each of the others.
        for other in others:
            self |= other

    def intersection(self, *others: Iterable[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a new set of elements present in this set and in every other
        # (an argument identical to this set is skipped). Order follows this set.
        res = self.copy()
        for other in others:
            if other is not self:
                res &= other  # type: ignore[arg-type]
        return res

    def intersection_update(self, *others: Iterable[T]) -> None:
        # [FUNCTION DOC] Keeps, in place, only elements also present in every other.
        for other in others:
            self &= other  # type: ignore[arg-type]

    def issubset(self, other: Iterable[T]) -> bool:
        # [FUNCTION DOC] True if every element is in other (a non-set iterable is converted first).
        return self <= self._wrap_iter_in_set(other)

    def issuperset(self, other: Iterable[T]) -> bool:
        # [FUNCTION DOC] True if this set contains every element of other (converted if needed).
        return self >= self._wrap_iter_in_set(other)

    def symmetric_difference(self, other: Iterable[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a new set of elements in exactly one of the two operands.
        return self ^ other  # type: ignore[operator]

    def symmetric_difference_update(self, other: Iterable[T]) -> None:
        # [FUNCTION DOC] Replaces this set, in place, with the symmetric difference with other.
        self ^= other  # type: ignore[arg-type]

    def union(self, *others: Iterable[T]) -> OrderedSet[T]:
        # [FUNCTION DOC] Returns a new set with this set's elements followed by new elements
        # from each of the others in order (an argument identical to this set is skipped).
        res = self.copy()
        for other in others:
            if other is self:
                continue
            res |= other
        return res

    # Specify here for correct type inference, otherwise would
    # return AbstractSet[T]
    def __sub__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # [FUNCTION DOC] Set difference returning an OrderedSet. When other is an OrderedSet that
        # is not much smaller, copies this set and removes other's elements; otherwise defers to
        # the generic set implementation.
        # following cpython set impl optimization
        if isinstance(other, OrderedSet) and (len(self) * 4) > len(other):
            out = self.copy()
            out -= other
            return out
        return cast(OrderedSet[T], super().__sub__(other))

    def __ior__(self, other: Iterable[T]) -> OrderedSet[T]:  # type: ignore[misc, override]   # noqa: PYI034
        # [FUNCTION DOC] In-place union. Fast path merges the other's mapping when it is an
        # OrderedSet; otherwise uses the generic implementation. Returns self.
        if isinstance(other, OrderedSet):
            self._dict.update(other._dict)
            return self
        return super().__ior__(other)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equality by elements, ignoring order (mapping equality). Other set
        # types use the generic set comparison. KNOWN ISSUE: class defines equality without a
        # hash method, so instances are unhashable (normal for mutable sets).
        if isinstance(other, OrderedSet):
            return self._dict == other._dict
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC] Negation of equality, with the same order-insensitive semantics.
        if isinstance(other, OrderedSet):
            return self._dict != other._dict
        return super().__ne__(other)

    def __or__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # [FUNCTION DOC] Union operator returning an OrderedSet; delegates to the generic one.
        return cast(OrderedSet[T], super().__or__(other))

    def __and__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # [FUNCTION DOC] Intersection operator returning an OrderedSet. If other is a larger
        # OrderedSet the roles are swapped for speed, so the result order follows the larger
        # operand's iteration of the smaller one rather than this set's order.
        # MutableSet impl will iterate over other, iter over smaller of two sets
        if isinstance(other, OrderedSet) and len(self) < len(other):
            # pyrefly: ignore [unsupported-operation, bad-return]
            return other & self
        return cast(OrderedSet[T], super().__and__(other))

    def __xor__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # [FUNCTION DOC] Symmetric difference operator returning an OrderedSet.
        return cast(OrderedSet[T], super().__xor__(other))

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form showing the class name and the elements as a list.
        return f"{self.__class__.__name__}({list(self)})"

    def __getstate__(self) -> list[T]:
        # [FUNCTION DOC] Pickle state: the elements in order as a list.
        return list(self._dict.keys())

    def __setstate__(self, state: list[T]) -> None:
        # [FUNCTION DOC] Restores the set from a list of elements.
        self._dict = dict.fromkeys(state, None)

    def __reduce__(self) -> tuple[type[OrderedSet[T]], tuple[list[T]]]:
        # [FUNCTION DOC] Pickle recipe: rebuild with the base class from the element list
        # (subclass identity is not preserved).
        return (OrderedSet, (list(self),))
