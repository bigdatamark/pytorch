# [MODULE DOC] Core of the multiple-dispatch machinery. Dispatcher stores
# implementations keyed by tuples of types (with optional trailing variadic),
# orders them from most to least specific, caches lookups per argument-type
# tuple, and calls the best match (falling back to less specific ones when an
# implementation raises MDNotImplementedError). Also contains MethodDispatcher
# (descriptor for methods), deprecated ordering toggles, variadic matching
# helpers, and signature/warning text formatting. No import-time side effects.
from __future__ import annotations

import inspect
import itertools as itl
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated
from warnings import warn


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

from .conflict import ambiguities, AmbiguityWarning, ordering, super_signature
from .utils import expand_tuples
from .variadic import isvariadic, Variadic


_T = TypeVar("_T")


__all__ = [
    "MDNotImplementedError",
    "ambiguity_warn",
    "halt_ordering",
    "restart_ordering",
    "variadic_signature_matches_iter",
    "variadic_signature_matches",
    "Dispatcher",
    "source",
    "MethodDispatcher",
    "str_signature",
    "warning_text",
]


class MDNotImplementedError(NotImplementedError):
    # [CLASS DOC] Raised by an implementation to signal that the dispatcher
    # should try the next less specific matching implementation.
    """A NotImplementedError for multiple dispatch"""


def ambiguity_warn(
    dispatcher: Dispatcher, ambiguities: set[tuple[tuple[type, ...], tuple[type, ...]]]
) -> None:
    # [FUNCTION DOC] Emit an AmbiguityWarning whose text lists the ambiguous
    # signature pairs of the dispatcher and suggested disambiguating signatures.
    """Raise warning when ambiguity is detected.

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also
    --------
    Dispatcher.add
    warning_text
    """
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


@deprecated(
    "`halt_ordering` is deprecated, you can safely remove this call.",
    category=FutureWarning,
)
def halt_ordering() -> None:
    # [FUNCTION DOC] Deprecated no-op kept for compatibility; emits a
    # FutureWarning when called (via the deprecated decorator).
    """Deprecated interface to temporarily disable ordering."""


@deprecated(
    "`restart_ordering` is deprecated, if you would like to eagerly order the dispatchers, "
    "you should call the `reorder()` method on each dispatcher.",
    category=FutureWarning,
)
def restart_ordering(on_ambiguity: Callable[..., None] = ambiguity_warn) -> None:
    # [FUNCTION DOC] Deprecated no-op kept for compatibility; the on_ambiguity
    # argument is ignored. Emits a FutureWarning when called.
    """Deprecated interface to temporarily resume ordering."""


def variadic_signature_matches_iter(
    types: tuple[type, ...], full_signature: tuple[type, ...]
) -> Generator[bool, None, None]:
    # [FUNCTION DOC] HELPER: Generator yielding one boolean per input type saying
    # whether it matches the corresponding signature element (a variadic element
    # is reused for all remaining types), then a final boolean: True if the
    # signature ended on a variadic, False if signature elements were left over.
    # Raises AssertionError if the signature runs out on a non-variadic element.
    # Raises StopIteration-derived errors (RuntimeError in a generator) if types
    # outnumber a signature that has no trailing variadic.
    """Check if a set of input types matches a variadic signature.

    Notes
    -----
    The algorithm is as follows:

    Initialize the current signature to the first in the sequence.
    For each type in ``types``:

    - If the current signature is variadic

      - If the type matches the signature, yield True
      - Else, try to get the next signature.
        If no signatures are left we can't possibly have a match,
        so yield False.

    - Else, yield True if the type matches the current signature.
      Get the next signature.
    """
    sigiter = iter(full_signature)
    sig = next(sigiter)
    for typ in types:
        matches = issubclass(typ, sig)
        yield matches
        if not isvariadic(sig):
            # we're not matching a variadic argument, so move to the next
            # element in the signature
            sig = next(sigiter)
    else:
        try:
            sig = next(sigiter)
        except StopIteration:
            if not isvariadic(sig):
                raise AssertionError("Expected variadic signature") from None
            yield True
        else:
            # We have signature items left over, so all of our arguments
            # haven't matched
            yield False


def variadic_signature_matches(
    types: tuple[type, ...], full_signature: tuple[type, ...]
) -> bool:
    # [FUNCTION DOC] True if every check from variadic_signature_matches_iter
    # passes. Raises AssertionError if full_signature is empty.
    # No arguments always matches a variadic signature
    if not full_signature:
        raise AssertionError("full_signature is empty")
    return all(variadic_signature_matches_iter(types, full_signature))


class Dispatcher:
    # [CLASS DOC] Callable holding implementations keyed by tuples of types.
    # State (slots): name; funcs, mapping signature to function; _cache, mapping
    # a tuple of runtime argument types to the resolved function (cleared on
    # add); _ordering, lazily computed specificity order (deleted on add); doc,
    # optional extra documentation. Calling it picks the most specific matching
    # implementation by argument types.
    """Dispatch methods based on type signature
    Use ``dispatch`` to add implementations
    Examples
    --------
    >>> # xdoctest: +SKIP("bad import name")
    >>> from multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1
    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1
    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """

    __slots__ = "__name__", "name", "funcs", "_ordering", "_cache", "doc"

    def __init__(self, name: str, doc: str | None = None) -> None:
        # [FUNCTION DOC] Initialize name, empty implementation table, optional doc
        # text and empty lookup cache. The ordering is computed lazily.
        self.name = self.__name__ = name
        self.funcs: dict[tuple[type, ...], Callable[..., object]] = {}
        self.doc = doc

        self._cache: dict[tuple[type, ...], Callable[..., object]] = {}

    def register(
        self, *types: type, **kwargs: object
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
        # [FUNCTION DOC] Return a decorator that adds the decorated function to
        # this dispatcher for the given types (extra kwargs are forwarded to add)
        # and returns the function itself, so decorators can be stacked.
        """register dispatcher with new implementation
        >>> # xdoctest: +SKIP
        >>> f = Dispatcher("f")
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1
        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1
        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]
        >>> f(1)
        2
        >>> f(1.0)
        0.0
        >>> f([1, 2, 3])
        [3, 2, 1]
        """

        def _df(func: Callable[..., _T]) -> Callable[..., _T]:
            # [FUNCTION DOC] HELPER: Add func under the captured types; return func.
            # KNOWN ISSUE: add accepts no keyword arguments, so non-empty kwargs
            # given to register would raise TypeError here.
            self.add(types, func, **kwargs)  # type: ignore[call-arg]
            return func

        return _df

    @classmethod
    def get_func_params(
        cls, func: Callable[..., object]
    ) -> Iterable[inspect.Parameter] | None:
        # [FUNCTION DOC] Return the parameters of func from its signature (all
        # of them). Returns None if inspect lacks signature support.
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(
        cls, func: Callable[..., object]
    ) -> tuple[type, ...] | None:
        # [FUNCTION DOC] Return the annotations of func's positional parameters as
        # a tuple, or None if there are no parameters or any is unannotated.
        """get annotations of function positional parameters"""
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            params = (
                param
                for param in params
                if param.kind
                in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            )

            annotations = tuple(param.annotation for param in params)

            if all(ann is not Parameter.empty for ann in annotations):
                return annotations

    def add(self, signature: tuple[type, ...], func: Callable[..., object]) -> None:
        # [FUNCTION DOC] Register func for signature. An empty signature falls back
        # to func's parameter annotations. Tuple entries (unions) are expanded
        # into several signatures. A one-element list entry in last position
        # becomes a variadic type. Stores the function, clears the lookup cache
        # and invalidates the cached ordering. Raises TypeError for non-type
        # entries, a non-last variadic, or a variadic list without exactly one item.
        """Add new types/method pair to dispatcher
        >>> # xdoctest: +SKIP
        >>> D = Dispatcher("add")
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)
        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>
        >>> # When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        >>> # with a dispatcher/itself, and a set of ambiguous type signature pairs
        >>> # as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func)
            return

        new_signature = []

        for index, typ in enumerate(signature, start=1):
            if not isinstance(typ, (type, list)):
                str_sig = ", ".join(
                    c.__name__ if isinstance(c, type) else str(c) for c in signature
                )
                raise TypeError(
                    f"Tried to dispatch on non-type: {typ}\n"
                    f"In signature: <{str_sig}>\n"
                    f"In function: {self.name}"
                )

            # handle variadic signatures
            if isinstance(typ, list):
                if index != len(signature):
                    raise TypeError("Variadic signature must be the last element")

                if len(typ) != 1:
                    raise TypeError(
                        "Variadic signature must contain exactly one element. "
                        "To use a variadic union type place the desired types "
                        "inside of a tuple, e.g., [(int, str)]"
                    )
                # pyrefly: ignore [bad-specialization]
                new_signature.append(Variadic[typ[0]])
            else:
                new_signature.append(typ)  # pyrefly: ignore[bad-argument-type]

        self.funcs[tuple(new_signature)] = func
        self._cache.clear()

        try:
            del self._ordering
        except AttributeError:
            pass

    @property
    def ordering(self) -> list[tuple[type, ...]]:
        # [FUNCTION DOC] Property: the cached specificity ordering of signatures,
        # computed (with ambiguity warnings) on first access after a change.
        try:
            return self._ordering
        except AttributeError:
            return self.reorder()

    def reorder(
        self, on_ambiguity: Callable[..., None] = ambiguity_warn
    ) -> list[tuple[type, ...]]:
        # [FUNCTION DOC] Recompute and store the signature ordering. If ambiguous
        # signature pairs exist, call on_ambiguity with the dispatcher and the
        # pairs (default emits a warning). Returns the ordering.
        self._ordering = od = ordering(self.funcs)
        amb = ambiguities(self.funcs)
        if amb:
            on_ambiguity(self, amb)
        return od

    def __call__(self, *args: object, **kwargs: object) -> object:
        # [FUNCTION DOC] Look up (and cache) the implementation for the runtime
        # types of args and call it with args and kwargs. If it raises
        # MDNotImplementedError, try the remaining matches in order. Raises
        # NotImplementedError if there is no match or all matches decline.
        types = tuple(type(arg) for arg in args)
        try:
            func = self._cache[types]
        except KeyError as e:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    f"Could not find signature for {self.name}: <{str_signature(types)}>"
                ) from e
            self._cache[types] = func
        try:
            return func(*args, **kwargs)

        except MDNotImplementedError as e:
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass

            raise NotImplementedError(
                "Matching functions for "
                f"{self.name}: <{str_signature(types)}> found, but none completed successfully",
            ) from e

    def __str__(self) -> str:
        # [FUNCTION DOC] Short text naming the dispatcher (also used as repr).
        return f"<dispatched {self.name}>"

    __repr__ = __str__

    def dispatch(self, *types: type) -> Callable[..., object] | None:
        # [FUNCTION DOC] Return the implementation for the given argument types:
        # an exact signature match if present, else the first entry of
        # dispatch_iter; None if nothing matches.
        """Determine appropriate implementation for this type signature
        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.
        >>> # xdoctest: +SKIP
        >>> from multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1
        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4
        >>> print(inc.dispatch(float))
        None
        See Also:
          ``multipledispatch.conflict`` - module to determine resolution order
        """

        if types in self.funcs:
            return self.funcs[types]

        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(
        self, *types: type
    ) -> Generator[Callable[..., object], None, None]:
        # [FUNCTION DOC] Generator over all implementations whose signature
        # matches the given types, in specificity order. Fixed-length signatures
        # need equal length and per-position subclass relations; signatures ending
        # in a variadic use the variadic matching rules.
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    yield result

    @deprecated(
        "`resolve()` is deprecated, use `dispatch(*types)`", category=FutureWarning
    )
    def resolve(self, types: tuple[type, ...]) -> Callable[..., object] | None:
        # [FUNCTION DOC] Deprecated alias for dispatch taking the types as one
        # tuple; emits a FutureWarning.
        """Determine appropriate implementation for this type signature
        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        return self.dispatch(*types)

    def __getstate__(self) -> dict[str, Any]:
        # [FUNCTION DOC] Pickle support: state is only the name and the
        # implementation table (the cache, ordering and doc are not saved).
        return {"name": self.name, "funcs": self.funcs}

    def __setstate__(self, d: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore name and implementations from d, recompute the
        # ordering (without ambiguity warnings) and reset the cache.
        # KNOWN ISSUE: the doc and __name__ slots are not restored, so accessing
        # them on an unpickled dispatcher raises AttributeError.
        self.name = d["name"]
        self.funcs = d["funcs"]
        self._ordering = ordering(self.funcs)
        self._cache = {}

    @property
    def __doc__(self):  # type: ignore[override]
        # [FUNCTION DOC] Property building documentation text: a header with the
        # name, the optional doc, then each implementation's docstring under an
        # "Inputs" heading (least specific first), then a list of signatures
        # whose functions lack docstrings. Returns the joined string.
        docs = [f"Multiply dispatched method: {self.name}"]

        if self.doc:
            docs.append(self.doc)

        other: list[str] = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = f"Inputs: <{str_signature(sig)}>\n"
                s += "-" * len(s) + "\n"
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append("Other signatures:\n    " + "\n    ".join(other))

        return "\n\n".join(docs)

    def _help(self, *args: object) -> str | None:
        # [FUNCTION DOC] HELPER: Docstring of the implementation chosen for the
        # types of args.
        # KNOWN ISSUE: if no implementation matches, dispatch returns None and
        # the result is the generic docstring of None rather than an error.
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args: object, **kwargs: object) -> None:
        # [FUNCTION DOC] Print (to stdout) the docstring of the implementation
        # that would handle args. kwargs are ignored.
        """Print docstring for the function corresponding to inputs"""
        print(self._help(*args))

    def _source(self, *args: object) -> str:
        # [FUNCTION DOC] HELPER: Source text of the implementation chosen for the
        # types of args. Raises TypeError if none matches.
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    def source(self, *args: object, **kwargs: object) -> None:
        # [FUNCTION DOC] Print (to stdout) the source of the implementation that
        # would handle args. kwargs are ignored. Raises TypeError if none matches.
        """Print source code for the function corresponding to inputs"""
        print(self._source(*args))


def source(func: Callable[..., object]) -> str:
    # [FUNCTION DOC] Return a string with the file path of func followed by its
    # source code. May raise OSError or TypeError for objects without source.
    s = f"File: {inspect.getsourcefile(func)}\n\n"
    s = s + inspect.getsource(func)
    return s


class MethodDispatcher(Dispatcher):
    # [CLASS DOC] Dispatcher usable as a class attribute (descriptor). Extra
    # slots obj and cls record the instance and owner from the last attribute
    # access; the instance is passed as the first argument to the implementation.
    # Signatures exclude the self parameter when reading annotations.
    # KNOWN ISSUE: obj and cls are stored on the shared dispatcher, so this is
    # not safe across instances used concurrently or interleaved.
    """Dispatch methods based on type signature
    See Also:
        Dispatcher
    """

    __slots__ = ("obj", "cls")

    @classmethod
    def get_func_params(
        cls, func: Callable[..., object]
    ) -> Iterator[inspect.Parameter] | None:
        # [FUNCTION DOC] Like the base version but skips the first parameter
        # (self). Returns None if inspect lacks signature support.
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    def __get__(self, instance: object | None, owner: type) -> MethodDispatcher:
        # [FUNCTION DOC] Descriptor hook: remember the accessing instance and
        # owner class on this dispatcher (mutation) and return the dispatcher.
        self.obj = instance
        self.cls = owner
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        # [FUNCTION DOC] Find the implementation for the runtime types of args
        # and call it with the remembered instance first, then args and kwargs.
        # Raises NotImplementedError if no signature matches. Unlike the base
        # class it does not cache or fall back on MDNotImplementedError.
        types = tuple(type(arg) for arg in args)
        func = self.dispatch(*types)
        if not func:
            raise NotImplementedError(
                f"Could not find signature for {self.name}: <{str_signature(types)}>"
            )
        return func(self.obj, *args, **kwargs)


def str_signature(sig: Iterable[type]) -> str:
    # [FUNCTION DOC] Comma-separated names of the types in sig.
    """String representation of type signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ", ".join(cls.__name__ for cls in sig)


def warning_text(name: str, amb: set[tuple[tuple[type, ...], tuple[type, ...]]]) -> str:
    # [FUNCTION DOC] Build the ambiguity warning message for function name: list
    # each ambiguous signature pair, then suggest a disambiguating dispatch
    # declaration per pair using super_signature.
    """The text for ambiguity warnings"""
    text = f"\nAmbiguities exist in dispatched function {name}\n\n"
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + ", ".join("[" + str_signature(s) + "]" for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += "\n\n".join(
        [
            "@dispatch(" + str_signature(super_signature(s)) + f")\ndef {name}(...)"
            for s in amb
        ]
    )
    return text
