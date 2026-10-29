# [MODULE DOC] Provides Thunk, a tiny lazy-evaluation wrapper around a zero-argument callable.
# No import-time side effects.
from collections.abc import Callable
from typing import Generic, TypeVar


R = TypeVar("R")


class Thunk(Generic[R]):
    # [CLASS DOC] Lazy value holder. State: f is the pending callable (None once forced) and r
    # is the cached result. Uses slots. Calling force evaluates at most once and then drops
    # the callable. Not thread safe.
    """
    A simple lazy evaluation implementation that lets you delay
    execution of a function.  It properly handles releasing the
    function once it is forced.
    """

    f: Callable[[], R] | None
    r: R | None

    __slots__ = ["f", "r"]

    def __init__(self, f: Callable[[], R]) -> None:
        # [FUNCTION DOC] Store the callable f and an empty result.
        self.f = f
        self.r = None

    def force(self) -> R:
        # [FUNCTION DOC] Return the value, calling the stored function on first use, caching
        # the result and releasing the function. If the function raises, nothing is cached and
        # the next call retries.
        if self.f is None:
            return self.r  # type: ignore[return-value]
        self.r = self.f()
        self.f = None
        return self.r
