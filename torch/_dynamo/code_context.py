# [MODULE DOC]
# Per-code-object context storage for Dynamo. Defines CodeContextDict, a thin wrapper
# over a weak-key dictionary keyed by code objects, and a module-level singleton
# named code_context. Entries vanish automatically when a code object is collected.
# Import-time side effect: creates the singleton instance.
"""
This module provides thread-safe code context management for TorchDynamo using weak references.

The CodeContextDict class maintains a mapping between Python code objects and their associated
context data, using weak references to automatically clean up entries when code objects are
garbage collected. This prevents memory leaks while allowing context data to be associated
with code objects throughout their lifecycle.

Key features:
- Thread-safe context storage and retrieval
- Automatic cleanup using weak references
- Safe context management for Python code objects
- Memory-leak prevention

Example usage:
    code_obj = compile('x = 1', '<string>', 'exec')

    # Store context
    context = code_context.get_context(code_obj)
    context['metadata'] = {'optimized': True}

    # Retrieve context
    if code_context.has_context(code_obj):
        ctx = code_context.get_context(code_obj)
        # Use context data...

    # Remove context
    ctx = code_context.pop_context(code_obj)
"""

import types
from typing import Any

from .utils import ExactWeakKeyDictionary


class CodeContextDict:
    # [CLASS DOC]
    # Maps code objects to per-code context mappings (string keys to arbitrary values).
    # State: code_context, an exact-identity weak-key dictionary. Entries are created
    # lazily by get_context and dropped by pop_context, clear, or code object collection.
    # Despite the module docstring, no explicit locking is performed here.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Creates the empty underlying weak-key dictionary.
        self.code_context: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def has_context(self, code: types.CodeType) -> bool:
        # [FUNCTION DOC]
        # Returns True if a context entry currently exists for the given code object.
        return code in self.code_context

    def get_context(self, code: types.CodeType) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the context mapping for the code object, creating and storing an
        # empty one if absent. Side effect: may insert a new entry.
        ctx = self.code_context.get(code)
        if ctx is None:
            # pyrefly: ignore [implicit-any]
            ctx = {}
            self.code_context[code] = ctx
        return ctx

    def pop_context(self, code: types.CodeType) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Removes and returns the context for the code object. If none existed, an
        # empty one is created first and then removed, so an empty mapping is returned.
        ctx = self.get_context(code)
        self.code_context._remove_id(id(code))
        return ctx

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Drops all stored contexts for every code object.
        self.code_context.clear()


code_context: CodeContextDict = CodeContextDict()
