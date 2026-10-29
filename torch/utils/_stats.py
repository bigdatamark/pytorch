# [MODULE DOC] Torch-free global call counters used for lightweight statistics.
# Holds the mapping simple_call_counter (label to count, insertion ordered), a function
# to bump a named label, and a decorator that counts calls by qualified function name.
# No import-time side effects other than creating the empty counter mapping.
# NOTE! PLEASE KEEP THIS FILE *FREE* OF TORCH DEPS! IT SHOULD BE IMPORTABLE ANYWHERE.
# IF YOU FEEL AN OVERWHELMING URGE TO ADD A TORCH DEP, MAKE A TRAMPOLINE FILE A LA torch._dynamo.utils
# AND SCRUB AWAY TORCH NOTIONS THERE.
import collections
import functools
from collections import OrderedDict
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec


simple_call_counter: OrderedDict[str, int] = collections.OrderedDict()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def count_label(label: str) -> None:
    # [FUNCTION DOC] Increment the global counter for label by one, creating it at zero
    # first if absent. Mutates simple_call_counter; returns nothing.
    prev = simple_call_counter.setdefault(label, 0)
    simple_call_counter[label] = prev + 1


def count(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] Decorator: returns a wrapper around fn that increments the global
    # counter keyed by fn's qualified name on every call, then calls fn and returns its
    # result. Exceptions from fn propagate (the count is bumped before the call).
    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Bump the counter for the wrapped function's qualified
        # name (initializing to zero if missing), then forward all arguments to it.
        if fn.__qualname__ not in simple_call_counter:
            simple_call_counter[fn.__qualname__] = 0
        simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
        return fn(*args, **kwargs)

    return wrapper
