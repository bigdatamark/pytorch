# [MODULE DOC] Small torch.compile-related helpers that import torch._dynamo
# lazily to avoid circular imports. Main content: the _disable_dynamo decorator
# (with typed overloads) for internal use. No import-time side effects.
"""
APIs related to torch.compile which lazily import torch._dynamo to avoid
circular dependencies.
"""

import functools
from collections.abc import Callable
from typing import overload, TypeVar
from typing_extensions import ParamSpec


_T = TypeVar("_T")
_P = ParamSpec("_P")


@overload
def _disable_dynamo(
    fn: Callable[_P, _T], recursive: bool = True
) -> Callable[_P, _T]: ...


@overload
def _disable_dynamo(
    fn: None = None, recursive: bool = True
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


def _disable_dynamo(
    fn: Callable[_P, _T] | None = None, recursive: bool = True
) -> Callable[_P, _T] | Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] HELPER: Internal decorator that makes a function run with
    # dynamo disabled. With a function, returns a wrapper that on first call
    # lazily imports torch._dynamo, builds the disabled version (recursive flag
    # passed through), caches it as an attribute on the original function
    # (mutation), and calls it. Without a function, returns a partial usable as
    # a decorator with the given recursive setting.
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            # [FUNCTION DOC] HELPER: Wrapper that fetches or builds (and caches
            # on the wrapped function) the dynamo-disabled callable, then invokes
            # it with the given arguments and returns its result.
            # cache this on the first invocation to avoid adding too much overhead.
            disable_fn = getattr(fn, "__dynamo_disable", None)
            if disable_fn is None:
                import torch._dynamo

                # We can safely turn off functools.wraps here because the inner
                # already wraps fn in the outer scope.
                disable_fn = torch._dynamo.disable(fn, recursive, wrapping=False)
                fn.__dynamo_disable = disable_fn  # type: ignore[attr-defined]

            return disable_fn(*args, **kwargs)

        return inner
    else:
        # decorator usage like @_disable_dynamo(recursive=False). The resulting
        # object expects the original decorated function as the arg.
        return functools.partial(_disable_dynamo, recursive=recursive)
