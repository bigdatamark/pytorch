# [MODULE DOC] Caching helpers: prefetchable_cache (a cache-once decorator with background
# prefetch, set and clear) and cache_method (a per-instance cache for methods that does not keep
# self alive). Creates at import time a module-level thread pool of two workers shared by all
# prefetches, and a sentinel object for cache misses.
from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Concatenate, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future


_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")

# Sentinel used to indicate that cache lookup failed.
_cache_sentinel = object()

_prefetch_executor = ThreadPoolExecutor(max_workers=2)


def prefetchable_cache(func: Callable[[], _T]) -> Callable[[], _T]:
    # [FUNCTION DOC] Decorator for a zero-argument function. Returns a wrapper that computes the
    # value once and caches it, guarded by a lock; the wrapper also gets set, clear and prefetch
    # attributes. The cache, pending future and lock are closure state.
    """
    Like functools.cache but with a prefetch() method that starts computing
    the value in a background thread, and a set() method for prepopulating.
    """
    _cache: _T | object = _cache_sentinel
    _lock = threading.Lock()
    _future: Future[_T] | None = None

    def wrapper() -> _T:
        # [FUNCTION DOC] Returns the cached value; else waits for a pending prefetch and caches
        # its result; else calls the function and caches it. Holds the lock throughout, so
        # exceptions from the function or future propagate and nothing is cached.
        nonlocal _cache, _future
        with _lock:
            if _cache is not _cache_sentinel:
                return _cache  # type: ignore[return-value]
            if _future is not None:
                _cache = _future.result()
                _future = None
                return _cache  # type: ignore[return-value]
            _cache = func()
            return _cache  # type: ignore[return-value]

    def set_val(val: _T) -> None:
        # [FUNCTION DOC] HELPER: Stores val as the cached value. Raises RuntimeError if a value is
        # already cached. A pending prefetch is not cancelled (it is simply ignored later).
        nonlocal _cache
        with _lock:
            if _cache is not _cache_sentinel:
                raise RuntimeError("prefetchable_cache value already set")
            _cache = val

    def clear() -> None:
        # [FUNCTION DOC] HELPER: Discards the cached value and any pending prefetch future (a
        # running computation is not cancelled), so the next call recomputes.
        nonlocal _cache, _future
        with _lock:
            _cache = _cache_sentinel
            _future = None

    def prefetch() -> None:
        # [FUNCTION DOC] HELPER: If nothing is cached and no prefetch is pending, submits the
        # function to the shared thread pool and records the future. Otherwise does nothing.
        nonlocal _future
        with _lock:
            if _cache is not _cache_sentinel or _future is not None:
                return
            _future = _prefetch_executor.submit(func)

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    wrapper.prefetch = prefetch  # type: ignore[attr-defined]
    return wrapper


def cache_method(
    f: Callable[Concatenate[_C, _P], _T],
) -> Callable[Concatenate[_C, _P], _T]:
    # [FUNCTION DOC] Decorator that memoizes a method per instance, keyed by the positional
    # arguments only (not self). The cache mapping is stored on the instance under an attribute
    # named after the method. Returns the wrapped method; the wrapper raises AssertionError when
    # called with keyword arguments.
    """
    Like `@functools.cache` but for methods.

    `@functools.cache` (and similarly `@functools.lru_cache`) shouldn't be used
    on methods because it caches `self`, keeping it alive
    forever. `@cache_method` ignores `self` so won't keep `self` alive (assuming
    no cycles with `self` in the parameters).

    Footgun warning: This decorator completely ignores self's properties so only
    use it when you know that self is frozen or won't change in a meaningful
    way (such as the wrapped function being pure).
    """
    cache_name = "_cache_method_" + f.__name__

    @functools.wraps(f)
    def wrap(self: _C, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: Looks up the positional args in the instance cache (creating the
        # cache on first use) and returns the stored value, else calls the method, stores and
        # returns the result. Raises AssertionError on keyword arguments. Arguments must be
        # hashable. KNOWN ISSUE: an existing but empty cache mapping is falsy, so it is replaced
        # by a new one, harmless but redundant.
        if kwargs:
            raise AssertionError("cache_method does not accept keyword arguments")
        if not (cache := getattr(self, cache_name, None)):
            cache = {}
            setattr(self, cache_name, cache)
        cached_value = cache.get(args, _cache_sentinel)
        if cached_value is not _cache_sentinel:
            return cached_value
        value = f(self, *args, **kwargs)
        cache[args] = value
        return value

    return wrap
