# [MODULE DOC] Small decorator helpers for the autograd code generators. They wrap a generator
# function that takes a native function paired with differentiability info so that it runs inside
# the torchgen native function context. No import-time side effects beyond imports.
import functools
from collections.abc import Callable

from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo as NFWDI
from torchgen.context import native_function_manager
from torchgen.utils import T


# Like tools.api.context.with_native_function, but for
# NativeFunctionWithDifferentiabilityInfo.
def with_native_function_with_differentiability_info(
    func: Callable[[NFWDI], T],
) -> Callable[[NFWDI], T]:
    # [FUNCTION DOC] Decorator factory for a generator taking one
    # NativeFunctionWithDifferentiabilityInfo. Returns a wrapper (preserving the wrapped name and
    # docstring) that enters the native function context for the info's underlying function, calls
    # the wrapped function with the info and returns its result. Errors raised inside are
    # annotated with the function location by the context.
    @functools.wraps(func)
    def wrapper(f: NFWDI) -> T:
        # [FUNCTION DOC] HELPER: Wrapper produced by the decorator. Enters the native function
        # context for the function inside the given info object, calls the original function with
        # that info and returns the result.
        with native_function_manager(f.func):
            return func(f)

    return wrapper


# Like the above but with an additional dispatch key string argument
def with_native_function_with_differentiability_info_and_key(
    func: Callable[[NFWDI, str], T],
) -> Callable[[NFWDI, str], T]:
    # [FUNCTION DOC] Same as the decorator above but for generators that take an extra dispatch
    # key string. Returns a wrapper that enters the native function context for the info's
    # function and calls the wrapped function with the info and the key, returning its result.
    @functools.wraps(func)
    def wrapper(f: NFWDI, key: str) -> T:
        # [FUNCTION DOC] HELPER: Wrapper produced by the key-taking decorator. Enters the native
        # function context for the info's function and forwards both the info and the dispatch key
        # string to the original function, returning its result.
        with native_function_manager(f.func):
            return func(f, key)

    return wrapper
