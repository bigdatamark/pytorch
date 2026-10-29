# [MODULE DOC] Public entry point of the multipledispatch implementation: the
# dispatch decorator factory and the ismethod heuristic. Holds a module-level
# global_namespace mapping function names to Dispatcher objects (used when no
# namespace is given). Exports dispatch and ismethod.
from __future__ import annotations

import inspect
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack


if TYPE_CHECKING:
    from collections.abc import Callable

from .dispatcher import Dispatcher, MethodDispatcher


global_namespace: dict[str, Dispatcher] = {}

__all__ = ["dispatch", "ismethod"]

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts")


def dispatch(
    *types: Unpack[_Ts], **kwargs: Any
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    # [FUNCTION DOC] Return a decorator that registers the decorated function for
    # the given positional argument types. Only the namespace keyword is used
    # (default global_namespace); other keywords are ignored. The decorator
    # finds or creates a Dispatcher named after the function (in the namespace,
    # or, for methods, in the caller's local scope), adds the implementation, and
    # returns the dispatcher. Mutates the namespace.
    """Dispatch function on the types of the inputs
    Supports dispatch on all non-keyword arguments.
    Collects implementations based on the function name.  Ignores namespaces.
    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Example:
        >>> # xdoctest: +SKIP
        >>> @dispatch(int)
        ... def f(x):
        ...     return x + 1
        >>> @dispatch(float)
        ... def f(x):
        ...     return x - 1
        >>> # xdoctest: +SKIP
        >>> f(3)
        4
        >>> f(3.0)
        2.0
        >>> # Specify an isolated namespace with the namespace keyword argument
        >>> my_namespace = {}
        >>> @dispatch(int, namespace=my_namespace)
        ... def foo(x):
        ...     return x + 1
        >>> # Dispatch on instance methods within classes
        >>> class MyClass(object):
        ...     @dispatch(list)
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     @dispatch(int)
        ...     def __init__(self, datum):
        ...         self.data = [datum]
        >>> MyClass([1, 2, 3]).data
        [1, 2, 3]
        >>> MyClass(3).data
        [3]
    """
    namespace = kwargs.get("namespace", global_namespace)

    types_tuple: tuple[type, ...] = tuple(types)  # type: ignore[arg-type]

    def _df(func: Callable[..., _T]) -> Callable[..., _T]:
        # [FUNCTION DOC] HELPER: Register func with its dispatcher and return the
        # dispatcher. Methods (first parameter named self) look the dispatcher up
        # in the defining frame's locals via frame inspection, creating a
        # MethodDispatcher if absent; plain functions use the namespace mapping.
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(  # type: ignore[union-attr]
                name,  # type: ignore[union-attr]
                MethodDispatcher(name),
            )
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types_tuple, func)
        return dispatcher

    return _df  # type: ignore[return-value]


def ismethod(func: Callable[..., object]) -> bool:
    # [FUNCTION DOC] Heuristic: True if func has a parameter named self. Needed
    # because methods look like plain functions while the class body runs.
    """Is func a method?
    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get("self", None) is not None
    else:
        spec = inspect.getfullargspec(func)  # type: ignore[union-attr, assignment]
        return bool(spec and spec.args and spec.args[0] == "self")
