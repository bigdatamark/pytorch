# [MODULE DOC] Small functorch utilities: re-exports exposed_in and unwrap_dead_wrappers,
# defines the argnums_t type alias (an int or tuple of ints) and a context manager that
# temporarily allows single-level autograd functions. No import-time side effects.
import contextlib
from collections.abc import Generator

from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_dead_wrappers,
)
from torch.utils._exposed_in import exposed_in


__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]


@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that sets the global "single level autograd function
    # allowed" flag to true inside the block and restores the previous value on exit, even on error.
    # KNOWN ISSUE: the previous state is read inside the try block; if reading it fails, the
    # finally clause would reference an unbound variable.
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)


argnums_t = int | tuple[int, ...]
