# [MODULE DOC] Core unification algorithm: reify substitutes variables in a term
# using a substitution mapping; unify finds a substitution making two terms equal.
# Both dispatch on argument types through the shared multipledispatch namespace
# from .dispatch (sequences, dicts, and a catch-all). Import-time side effect:
# registers the _reify, _unify and unify implementations in that namespace.
from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from .dispatch import dispatch
from .unification_tools import assoc  # type: ignore[import]
from .utils import transitive_get as walk
from .variable import isvar


if TYPE_CHECKING:
    from .variable import Var


_Ts = TypeVarTuple("_Ts")

__all__ = ["reify", "unify"]

###############
# Reification #
###############


@dispatch(Iterator, dict)
def _reify(t: Iterator[object], s: dict[Var, object]) -> Iterator[object]:
    # [FUNCTION DOC] HELPER: Dispatch case for iterators. Lazily returns a map
    # iterator that reifies each element with substitution s. No mutation.
    return map(partial(reify, s=s), t)
    # return (reify(arg, s) for arg in t)


@dispatch(tuple, dict)  # type: ignore[no-redef]
def _reify(t: tuple[Unpack[_Ts]], s: dict[Var, object]) -> tuple[Unpack[_Ts]]:
    # [FUNCTION DOC] HELPER: Dispatch case for tuples. Returns a new tuple with
    # every element reified under substitution s.
    return tuple(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type, bad-return]


@dispatch(list, dict)  # type: ignore[no-redef]
def _reify(t: list[object], s: dict[Var, object]) -> list[object]:
    # [FUNCTION DOC] HELPER: Dispatch case for lists. Returns a new list with
    # every element reified under substitution s.
    return list(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type]


@dispatch(dict, dict)  # type: ignore[no-redef]
def _reify(d: dict[object, object], s: dict[Var, object]) -> dict[object, object]:
    # [FUNCTION DOC] HELPER: Dispatch case for dicts. Returns a new dict with the
    # same keys (keys are not reified) and reified values.
    return {k: reify(v, s) for k, v in d.items()}


@dispatch(object, dict)  # type: ignore[no-redef]
def _reify(o: object, s: dict[Var, object]) -> object:
    # [FUNCTION DOC] HELPER: Catch-all dispatch case. Returns the object unchanged.
    return o  # catch all, just return the object


def reify(e: object, s: dict[Var, object]) -> object:
    # [FUNCTION DOC] Replace logic variables in term e using substitution s.
    # If e is a variable bound in s, the bound value is itself reified
    # recursively (following chains); an unbound variable is returned as is.
    # Non-variables go through the type-dispatched _reify. Returns the new term;
    # inputs are not mutated. Raises NotImplementedError only if dispatch fails.
    """Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
    if isvar(e):
        return reify(s[e], s) if e in s else e
    return _reify(e, s)


###############
# Unification #
###############

seq = tuple, list, Iterator


@dispatch(seq, seq, dict)  # type: ignore[arg-type]
def _unify(
    u: Sequence[object], v: Sequence[object], s: dict[Var, object]
) -> dict[Var, object] | bool:
    # [FUNCTION DOC] HELPER: Dispatch case unifying two sequences (tuple, list or
    # iterator). Returns False on length mismatch; otherwise unifies elements
    # pairwise, threading the substitution through, and returns the final
    # substitution or False at the first failing pair.
    # KNOWN ISSUE: iterators are passed to len, which fails for them, so the
    # Iterator case is effectively unsupported here.
    if len(u) != len(v):
        return False
    for uu, vv in zip(u, v):  # avoiding recursion
        s = unify(uu, vv, s)  # pyrefly: ignore[bad-assignment]
        if s is False:
            return False
    return s


#
# @dispatch((set, frozenset), (set, frozenset), dict)
# def _unify(u, v, s):
#     i = u & v
#     u = u - i
#     v = v - i
#     return _unify(sorted(u), sorted(v), s)
#
#
# @dispatch(dict, dict, dict)
# def _unify(u, v, s):
#     if len(u) != len(v):
#         return False
#     for key, uval in iteritems(u):
#         if key not in v:
#             return False
#         s = unify(uval, v[key], s)
#         if s is False:
#             return False
#     return s
#
#
# @dispatch(object, object, dict)
# def _unify(u, v, s):
#     return False  # catch all


@dispatch(object, object, dict)
def unify(
    u: object, v: object, s: dict[Var, object]
) -> dict[Var, object] | bool:  # no check at the moment
    # [FUNCTION DOC] Unify terms u and v under existing substitution s. Both are
    # first resolved through s (transitive lookup). If equal, returns s
    # unchanged. If either is a variable, returns a new substitution extending s
    # with that binding (no occurs check). Otherwise defers to _unify by type.
    # Returns the substitution mapping or False when the terms cannot unify.
    """Find substitution so that u == v while satisfying s
    >>> x = var("x")
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    u = walk(u, s)
    v = walk(v, s)
    if u == v:
        return s
    if isvar(u):
        return assoc(s, u, v)
    if isvar(v):
        return assoc(s, v, u)
    return _unify(u, v, s)


@dispatch(object, object)  # type: ignore[no-redef]
# The noqa suppresses F811 for this multipledispatch overload redefinition
# (it replaces the bare-name-statement hack that previously silenced it).
def unify(u: object, v: object) -> dict[Var, object] | bool:  # noqa: F811
    # [FUNCTION DOC] Two-argument overload: unify u and v starting from an empty
    # substitution. Returns the substitution or False.
    return unify(u, v, {})
