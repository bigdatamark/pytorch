# [MODULE DOC]
# Defines immutable_list and immutable_dict, list/dict subclasses used for the args and kwargs
# of fx Nodes so they cannot be mutated in place. Mutating methods raise TypeError with a hint
# to copy and reassign. Both are hashable and picklable via __reduce__.
# Import-time side effect: registers both types with the pytree registry (flatten/unflatten
# with keys, and a serialized type name).
from collections.abc import Iterable
from typing import Any, NoReturn, TypeVar
from typing_extensions import Self

from torch.utils._pytree import (
    _dict_flatten,
    _dict_flatten_with_keys,
    _dict_unflatten,
    _list_flatten,
    _list_flatten_with_keys,
    _list_unflatten,
    Context,
    register_pytree_node,
)

from ._compatibility import compatibility


__all__ = ["immutable_list", "immutable_dict"]


_help_mutation = """
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:

    new_args = ...  # copy and mutate args
    node.args = new_args
""".strip()


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self: Any, *args: Any, **kwargs: Any) -> NoReturn:
    # [FUNCTION DOC] HELPER: Placeholder installed for every mutating method of the immutable
    # collections. Ignores its arguments and always raises TypeError naming the type and
    # advising to copy and reassign the node's args or kwargs.
    raise TypeError(
        f"{type(self).__name__!r} object does not support mutation. {_help_mutation}",
    )


@compatibility(is_backward_compatible=True)
class immutable_list(list[_T]):
    # [CLASS DOC] A list subclass whose mutating operations (item set/delete, in-place add and
    # multiply, append, clear, extend, insert, pop, remove, reverse, sort) all raise TypeError.
    # Hashable (by tuple contents) and pickles by reconstructing from a tuple.
    """An immutable version of :class:`list`."""

    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation

    def __hash__(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC] Returns a hash computed from the elements as a tuple; elements must be
        # hashable or the hash computation raises TypeError.
        return hash(tuple(self))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[_T, ...]]]:
        # [FUNCTION DOC] Pickle/copy support: returns the type and a single tuple of the
        # elements so the object is rebuilt through the constructor, not by item mutation.
        return (type(self), (tuple(self),))


@compatibility(is_backward_compatible=True)
class immutable_dict(dict[_KT, _VT]):
    # [CLASS DOC] A dict subclass whose mutating operations (item set/delete, in-place or,
    # clear, pop, popitem, setdefault, update) all raise TypeError. Hashable (by a frozenset of
    # its items) and pickles by reconstructing from a tuple of items.
    """An immutable version of :class:`dict`."""

    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        # [FUNCTION DOC] Returns a hash of the frozen set of key/value pairs; values must be
        # hashable or TypeError is raised.
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[tuple[_KT, _VT], ...]]]:
        # [FUNCTION DOC] Pickle/copy support: returns the type and a tuple of key/value pairs
        # used to rebuild the mapping through the constructor.
        return (type(self), (tuple(self.items()),))


# Register immutable collections for PyTree operations
def _immutable_list_flatten(d: immutable_list[_T]) -> tuple[list[_T], Context]:
    # [FUNCTION DOC] HELPER: Pytree flatten for immutable_list; delegates to the plain list
    # flatten and returns (child values, context).
    return _list_flatten(d)


def _immutable_list_unflatten(
    values: Iterable[_T],
    context: Context,
) -> immutable_list[_T]:
    # [FUNCTION DOC] HELPER: Pytree unflatten for immutable_list; rebuilds a plain list from the
    # values and context, then wraps it in an immutable_list.
    return immutable_list(_list_unflatten(values, context))


def _immutable_dict_flatten(d: immutable_dict[Any, _VT]) -> tuple[list[_VT], Context]:
    # [FUNCTION DOC] HELPER: Pytree flatten for immutable_dict; delegates to the plain dict
    # flatten and returns (values, context holding the keys).
    return _dict_flatten(d)


def _immutable_dict_unflatten(
    values: Iterable[_VT],
    context: Context,
) -> immutable_dict[Any, _VT]:
    # [FUNCTION DOC] HELPER: Pytree unflatten for immutable_dict; rebuilds a plain dict from the
    # values and context keys, then wraps it in an immutable_dict.
    return immutable_dict(_dict_unflatten(values, context))


register_pytree_node(
    immutable_list,
    _immutable_list_flatten,
    _immutable_list_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
register_pytree_node(
    immutable_dict,
    _immutable_dict_flatten,
    _immutable_dict_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
