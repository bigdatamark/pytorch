# [MODULE DOC] Thread-local nesting counter identifying the active SubgraphTracer scope.
# Exposes current_scope_id (getter, default 1) and enter_new_scope (context manager
# that increments the counter and restores it on exit). Import-time side effect:
# creates the module-level thread-local storage object.
"""
Provides thread-local scope identification for SubgraphTracer instances.

This module implements a thread-safe mechanism for tracking nested tracing contexts,
which is essential when multiple SubgraphTracer instances are active. The scope ID
helps identify which tracer context is currently active when direct access to the
InstructionTranslator is difficult.

Key components:
- Thread-local scope ID storage (_current_scope_id)
- Getter function (current_scope_id) to safely access the current scope
- Context manager (enter_new_scope) for managing nested scope transitions

The scope ID increments when entering a new context and decrements when exiting,
allowing proper tracking of nested tracing operations across different threads.
"""

import contextlib
import threading
from collections.abc import Generator


# Global variable to identify which SubgraphTracer we are in.
# It is sometimes difficult to find an InstructionTranslator to use.
_current_scope_id = threading.local()


def current_scope_id() -> int:
    # [FUNCTION DOC] Returns the current thread's scope id, initializing it to 1 on
    # first access in that thread (side effect: sets the thread-local value).
    global _current_scope_id
    if not hasattr(_current_scope_id, "value"):
        _current_scope_id.value = 1
    return _current_scope_id.value


@contextlib.contextmanager
def enter_new_scope() -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that increments the thread-local scope id for
    # the duration of its body and decrements it on exit (also on error).
    global _current_scope_id
    try:
        _current_scope_id.value = current_scope_id() + 1
        yield
    finally:
        _current_scope_id.value = current_scope_id() - 1
