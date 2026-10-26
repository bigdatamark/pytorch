# [MODULE DOC] Registry of intermediate-value hooks for debugging: a global ordered list of callbacks invoked with a buffer name and tensor value.
# Provides a context manager to register a hook temporarily and a function to run all registered hooks.
# mypy: allow-untyped-defs
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

# Executed in the order they're registered
INTERMEDIATE_HOOKS: list[Callable[[str, "torch.Tensor"], None]] = []


@contextlib.contextmanager
def intermediate_hook(fn):
    # [FUNCTION DOC] Context manager that appends fn to the global hook list on entry and removes the last hook on exit (even on error).
    # Mutates global state; assumes properly nested use since it pops the last entry.
    INTERMEDIATE_HOOKS.append(fn)
    try:
        yield
    finally:
        INTERMEDIATE_HOOKS.pop()


def run_intermediate_hooks(name, val):
    # [FUNCTION DOC] Run all registered hooks in registration order with the given name and value.
    # Temporarily clears the global hook list while running, so hooks that trigger further hook runs do not recurse; restores the list afterwards, even if a hook raises.
    global INTERMEDIATE_HOOKS
    hooks = INTERMEDIATE_HOOKS
    INTERMEDIATE_HOOKS = []
    try:
        for hook in hooks:
            hook(name, val)
    finally:
        INTERMEDIATE_HOOKS = hooks
