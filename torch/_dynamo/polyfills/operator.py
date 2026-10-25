# [MODULE DOC] Dynamo polyfills for operator.attrgetter, concat, countOf, iconcat, itemgetter
# and methodcaller (simple unary/binary operators are handled by BuiltinVariable instead).
# Registers pure-Python substitutes via substitute_in_graph; getters are embedded types.
# Import-time side effect: registers the substitutions.
"""
Python polyfills for operator
"""

from __future__ import annotations

import operator
from typing import Any, overload, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


# Most unary and binary operators are handled by BuiltinVariable (e.g., `pos`, `add`)
__all__ = ["attrgetter", "concat", "countOf", "iconcat", "itemgetter", "methodcaller"]


_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_Ts = TypeVarTuple("_Ts")
_U = TypeVar("_U")
_U1 = TypeVar("_U1")
_U2 = TypeVar("_U2")
_Us = TypeVarTuple("_Us")


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(attr: str, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def attrgetter(
    attr1: str, attr2: str, /, *attrs: str
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.attrgetter
@substitute_in_graph(operator.attrgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def attrgetter(*attrs: str) -> Callable[[Any], Any | tuple[Any, ...]]:
    # [FUNCTION DOC] Polyfill of operator.attrgetter. Takes one or more attribute names
    # (dotted names allowed) and returns a callable that resolves each name on its argument
    # by following the dotted path; returns a single value for one name, else a tuple.
    # Raises TypeError if no names are given or any name is not a string.
    if len(attrs) == 0:
        raise TypeError("attrgetter expected 1 argument, got 0")

    if any(not isinstance(attr, str) for attr in attrs):
        raise TypeError("attribute name must be a string")

    def resolve_attr(obj: Any, attr: str) -> Any:
        # [FUNCTION DOC] HELPER: follows the dot-separated attribute path on obj.
        for name in attr.split("."):
            obj = getattr(obj, name)
        return obj

    if len(attrs) == 1:
        attr = attrs[0]

        def getter(obj: Any) -> Any:
            # [FUNCTION DOC] HELPER: returns the single resolved attribute of obj.
            return resolve_attr(obj, attr)

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            # [FUNCTION DOC] HELPER: returns a tuple of all resolved attributes of obj.
            return tuple(resolve_attr(obj, attr) for attr in attrs)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.concat
@substitute_in_graph(operator.concat, can_constant_fold_through=True)  # type: ignore[arg-type]
def concat(a: Sequence[_T], b: Sequence[_T2], /) -> Sequence[_T | _T2]:
    # [FUNCTION DOC] Polyfill of operator.concat: returns the concatenation of two
    # sequences (their sum). Raises whatever the addition raises (e.g. TypeError).
    return a + b  # type: ignore[operator]


# Reference: https://docs.python.org/3/library/operator.html#operator.countOf
@substitute_in_graph(operator.countOf, can_constant_fold_through=True)  # type: ignore[arg-type,misc]
def countOf(a: Iterable[_T], b: _T, /) -> int:
    # [FUNCTION DOC] Polyfill of operator.countOf: counts items of the iterable that are
    # identical or equal to b and returns the count.
    return sum(it is b or it == b for it in a)


# Reference: https://docs.python.org/3/library/operator.html#operator.iconcat
@substitute_in_graph(operator.iconcat)  # type: ignore[arg-type]
def iconcat(a: Sequence[_T], b: Sequence[_T2], /) -> Sequence[_T | _T2]:
    # [FUNCTION DOC] Polyfill of operator.iconcat: performs in-place addition of b onto a
    # (mutating a if it supports it) and returns the result.
    a += b  # type: ignore[operator]
    return a  # type: ignore[return-value]


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(item: _T, /) -> Callable[[Any], _U]: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def itemgetter(
    item1: _T1, item2: _T2, /, *items: Unpack[_Ts]
) -> Callable[[Any], tuple[_U1, _U2, Unpack[_Us]]]: ...


# Reference: https://docs.python.org/3/library/operator.html#operator.itemgetter
@substitute_in_graph(operator.itemgetter, is_embedded_type=True)  # type: ignore[arg-type,misc]
def itemgetter(*items: Any) -> Callable[[Any], Any | tuple[Any, ...]]:
    # [FUNCTION DOC] Polyfill of operator.itemgetter. Takes one or more keys and returns a
    # callable that indexes its argument with them: a single value for one key, else a
    # tuple. Raises TypeError if no keys are given.
    if len(items) == 0:
        raise TypeError("itemgetter expected 1 argument, got 0")

    if len(items) == 1:
        item = items[0]

        def getter(obj: Any) -> Any:
            # [FUNCTION DOC] HELPER: returns the argument indexed by the single key.
            return obj[item]

    else:

        def getter(obj: Any) -> tuple[Any, ...]:  # type: ignore[misc]
            # [FUNCTION DOC] HELPER: returns a tuple indexing the argument with every key.
            return tuple(obj[item] for item in items)

    return getter


# Reference: https://docs.python.org/3/library/operator.html#operator.methodcaller
@substitute_in_graph(operator.methodcaller, is_embedded_type=True)  # type: ignore[arg-type]
def methodcaller(name: str, /, *args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    # [FUNCTION DOC] Polyfill of operator.methodcaller. Returns a callable that looks up the
    # named method on its argument and calls it with the stored positional and keyword
    # arguments. Raises TypeError if name is not a string.
    if not isinstance(name, str):
        raise TypeError("method name must be a string")

    def caller(obj: Any) -> Any:
        # [FUNCTION DOC] HELPER: calls the named method on obj with the stored arguments.
        return getattr(obj, name)(*args, **kwargs)

    return caller
