# [MODULE DOC] Pattern-matching function dispatch built on unification. A
# Dispatcher keeps signatures (possibly containing logic variables) ordered from
# most to least specific and calls the first function whose signature unifies
# with the arguments. Contents: Dispatcher, VarDispatcher, match decorator,
# and ordering helpers (supercedes, edge, ordering). Holds a module-level
# global_namespace of dispatchers keyed by function name.
from __future__ import annotations

from typing import TYPE_CHECKING

from .core import reify, unify  # type: ignore[attr-defined]


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .variable import Var

from .unification_tools import first, groupby  # type: ignore[import]
from .utils import _toposort, freeze
from .variable import isvar


class Dispatcher:
    # [CLASS DOC] Callable that selects an implementation by unifying the call
    # arguments against registered signatures. State: name; funcs, mapping a
    # frozen signature to its function; ordering, the signatures sorted so more
    # specific ones are tried first (recomputed on every add).
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Store the dispatcher name and start with no functions
        # and an empty ordering.
        self.name = name
        self.funcs: dict[object, Callable[..., object]] = {}
        self.ordering: list[object] = []

    def add(self, signature: tuple[object, ...], func: Callable[..., object]) -> None:
        # [FUNCTION DOC] Register func under the frozen form of signature
        # (overwriting an equal signature) and recompute the ordering.
        self.funcs[freeze(signature)] = func
        self.ordering = ordering(self.funcs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        # [FUNCTION DOC] Resolve the matching function for args and call it with
        # the original args and kwargs. Raises NotImplementedError if no match.
        func, _ = self.resolve(args)
        return func(*args, **kwargs)

    def resolve(
        self, args: tuple[object, ...]
    ) -> tuple[Callable[..., object], dict[Var, object]]:
        # [FUNCTION DOC] Walk the ordered signatures of matching length and unify
        # the frozen args with each. Returns the first matching function and the
        # resulting substitution. Raises NotImplementedError listing known
        # signatures and the input if nothing matches.
        n = len(args)
        for signature in self.ordering:
            if len(signature) != n:  # pyrefly: ignore[bad-argument-type]
                continue
            s = unify(freeze(args), signature)
            if s is not False:
                result = self.funcs[signature]
                return result, s  # pyrefly: ignore[bad-return]
        raise NotImplementedError(
            "No match found. \nKnown matches: "
            + str(self.ordering)
            + "\nInput: "
            + str(args)
        )

    def register(self, *signature: object) -> Callable[..., object]:
        # [FUNCTION DOC] Return a decorator that registers the decorated function
        # under signature on this dispatcher.
        def _(func: Callable[..., object]) -> Dispatcher:
            # [FUNCTION DOC] HELPER: Add func to the dispatcher and return the
            # dispatcher (so the decorated name becomes the dispatcher).
            self.add(signature, func)
            return self

        return _


class VarDispatcher(Dispatcher):
    # [CLASS DOC] Dispatcher subclass that passes the unified variable bindings
    # to the chosen function as keyword arguments named after the variables.
    """A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher("d")
    >>> x = var("x")
    >>> @d.register("inc", x)
    ... def f(x):
    ...     return x + 1
    >>> @d.register("double", x)
    ... def f(x):
    ...     return x * 2
    >>> d("inc", 10)
    11
    >>> d("double", 10)
    20
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        # [FUNCTION DOC] Resolve the match, then call the function with keyword
        # arguments built from the substitution, keyed by each variable's token
        # name. The kwargs parameter is ignored. Raises NotImplementedError if
        # no signature matches.
        func, s = self.resolve(args)
        d = {k.token: v for k, v in s.items()}  # pyrefly: ignore[missing-attribute]
        return func(**d)


global_namespace: dict[str, Dispatcher] = {}


def match(*signature: object, **kwargs: object) -> Callable[..., object]:
    # [FUNCTION DOC] Decorator factory. Optional kwargs: namespace (mapping of
    # name to dispatcher, default global_namespace) and Dispatcher (class to
    # instantiate, default Dispatcher). The decorator looks up or creates the
    # dispatcher named after the function, registers the function under
    # signature, and returns the dispatcher. Mutates the namespace.
    namespace: dict[str, Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "namespace", global_namespace
    )
    dispatcher_cls: type[Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "Dispatcher", Dispatcher
    )

    def _(func: Callable[..., object]) -> Dispatcher:
        # [FUNCTION DOC] HELPER: Register func with the dispatcher named after it,
        # creating that dispatcher in the namespace if absent; return it.
        name = func.__name__

        if name not in namespace:
            namespace[name] = dispatcher_cls(name)
        d = namespace[name]

        d.add(signature, func)

        return d

    return _


def supercedes(a: object, b: object) -> bool:
    # [FUNCTION DOC] True if pattern a is at least as specific as pattern b. A
    # variable b is superseded by any non-variable a. Otherwise unify the two;
    # if that fails return False; drop variable-to-variable bindings and return
    # True when reifying a with the rest leaves a unchanged, else False.
    # KNOWN ISSUE: the second reify check (on b) also returns False, so it is
    # redundant and the final branches are equivalent.
    """``a`` is a more specific match than ``b``"""
    if isvar(b) and not isvar(a):
        return True
    s = unify(a, b)
    if s is False:
        return False
    s = {
        k: v
        for k, v in s.items()  # pyrefly: ignore[missing-attribute]
        if not isvar(k) or not isvar(v)
    }
    if reify(a, s) == a:
        return True
    if reify(b, s) == b:
        return False
    return False


# Taken from multipledispatch
def edge(a: object, b: object, tie_breaker: Callable[[object], int] = hash) -> bool:
    # [FUNCTION DOC] True if a must be checked before b: a supercedes b, and if
    # b also supercedes a (equivalent patterns) the tie_breaker values decide
    # (a greater than b). Pure function.
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


# Taken from multipledispatch
def ordering(signatures: Iterable[object]) -> list[object]:
    # [FUNCTION DOC] Build the precedence graph over all signature pairs using
    # edge, make sure every signature appears as a node, and return a
    # topological sort (most specific first). Quadratic in signature count.
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))  # pyrefly: ignore[bad-argument-type]
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(first, edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[attr-defined, assignment]
    return _toposort(edges)  # pyrefly: ignore[bad-argument-type]
