# [MODULE DOC] Provides the exposed_in decorator factory, which rewrites an object's
# __module__ so a function defined in a private submodule appears as part of a public module.
# No import-time side effects.
from collections.abc import Callable
from typing import TypeVar


F = TypeVar("F")


# Allows one to expose an API in a private submodule publicly as per the definition
# in PyTorch's public api policy.
#
# It is a temporary solution while we figure out if it should be the long-term solution
# or if we should amend PyTorch's public api policy. The concern is that this approach
# may not be very robust because it's not clear what __module__ is used for.
# However, both numpy and jax overwrite the __module__ attribute of their APIs
# without problem, so it seems fine.
def exposed_in(module: str) -> Callable[[F], F]:
    # [FUNCTION DOC] Takes a dotted module name and returns a decorator that sets the decorated
    # object's module attribute to that name and returns the same object (mutation in place).
    def wrapper(fn: F) -> F:
        # [FUNCTION DOC] HELPER: Overwrites the module attribute of fn with the captured name
        # and returns fn. Raises if the object does not allow setting that attribute.
        fn.__module__ = module
        return fn

    return wrapper
