# [MODULE DOC] Small utilities for the unification package: hashability test,
# transitive dict lookup, exception-expectation helpers (raises, xfail),
# topological sort (_toposort, reverse_dict), and freeze for making containers
# hashable. No import-time side effects.
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .variable import Var

_T = TypeVar("_T")


__all__ = ["hashable", "transitive_get", "raises", "reverse_dict", "xfail", "freeze"]


def hashable(x: object) -> bool:
    # [FUNCTION DOC] True if x can be hashed, False if hashing raises TypeError.
    try:
        hash(x)
        return True
    except TypeError:
        return False


def transitive_get(key: object, d: dict[Var, object]) -> object:
    # [FUNCTION DOC] Repeatedly look up key in d while the current value is
    # hashable and present as a key; return the final value. Loops forever if d
    # contains a cycle (for example a variable bound to itself indirectly).
    """Transitive dict.get
    >>> d = {1: 2, 2: 3, 3: 4}
    >>> d.get(1)
    2
    >>> transitive_get(1, d)
    4
    """
    while hashable(key) and key in d:
        key = d[key]
    return key


def raises(
    err: type[BaseException],
    lamda: Callable[[], object],  # codespell:ignore lamda
) -> bool:
    # [FUNCTION DOC] HELPER: Call the zero-argument callable; return True if it
    # raises err (or a subclass), False if it returns normally. Other exceptions
    # propagate.
    try:
        lamda()  # codespell:ignore lamda
        return False
    except err:
        return True


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges: dict[_T, Iterable[_T]]) -> list[_T]:
    # [FUNCTION DOC] HELPER: Kahn topological sort. edges maps a node to the nodes
    # that depend on it; returns nodes ordered so each precedes its dependents.
    # Starts from nodes with no incoming edges (picked in set order, so ties are
    # not deterministic). Raises ValueError("Input has cycles") on cycles.
    # KNOWN ISSUE: nodes appearing only as dependents (not keys of edges) are
    # never emitted unless they also have a key.
    """Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> # xdoctest: +SKIP
    >>> _toposort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    Closely follows the wikipedia page [2]
    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = {v for v in edges if v not in incoming_edges}
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            if n not in incoming_edges[m]:
                raise AssertionError(f"Expected {n} in incoming_edges[{m}]")
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d: dict[_T, Iterable[_T]]) -> dict[_T, tuple[_T, ...]]:
    # [FUNCTION DOC] Invert a dependency mapping: for each value found in any
    # key's iterable, map it to a tuple of the keys that listed it. Keys with
    # empty iterables do not appear in the result. Order follows dict iteration.
    """Reverses direction of dependence dict.

    >>> d = {"a": (1, 2), "b": (2, 3), "c": ()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    .. note::
        dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    """
    result = {}  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            # pyrefly: ignore [unsupported-operation]
            result[val] = result.get(val, ()) + (key,)
    return result  # pyrefly: ignore[bad-return]


def xfail(func: Callable[[], object]) -> None:
    # [FUNCTION DOC] HELPER: Intended to assert that func fails. Calls func and
    # always returns None.
    # KNOWN ISSUE: the "XFailed test passed" exception is raised inside the try
    # block and is swallowed by the broad except, so this never raises.
    try:
        func()
        raise Exception("XFailed test passed")  # pragma:nocover  # noqa: TRY002
    except Exception:
        pass


def freeze(d: object) -> object:
    # [FUNCTION DOC] Recursively convert containers to hashable form: dict to
    # frozenset of frozen items, set to frozenset, tuple/list to tuple; other
    # values are returned unchanged. Pure.
    """Freeze container to hashable form
    >>> freeze(1)
    1
    >>> freeze([1, 2])
    (1, 2)
    >>> freeze({1: 2})  # doctest: +SKIP
    frozenset([(1, 2)])
    """
    if isinstance(d, dict):
        return frozenset(map(freeze, d.items()))
    if isinstance(d, set):
        return frozenset(map(freeze, d))
    if isinstance(d, (tuple, list)):
        return tuple(map(freeze, d))
    return d
