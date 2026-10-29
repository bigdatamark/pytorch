# mypy: allow-untyped-defs
# [MODULE DOC]
# Exposes the fill_uninitialized_memory setting of deterministic mode as a module-level
# property. Import-time side effect: replaces the class of this module object in the module
# registry with a ModuleType subclass that carries the property.
import sys
import types

import torch


class _Deterministic(types.ModuleType):
    # [CLASS DOC] HELPER: Module subclass used to give the torch.utils.deterministic module a
    # property backed by the C++ global flag for filling uninitialized memory.
    @property
    def fill_uninitialized_memory(self):
        # [FUNCTION DOC] Getter: returns the current global flag (bool) controlling whether
        # uninitialized memory is filled with a known value in deterministic mode.
        """
        Whether to fill uninitialized memory with a known value when
        :meth:`torch.use_deterministic_algorithms()` is set to ``True``.
        """
        return torch._C._get_deterministic_fill_uninitialized_memory()

    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode):
        # [FUNCTION DOC] Setter: sets the global flag to mode (side effect on process-wide state).
        return torch._C._set_deterministic_fill_uninitialized_memory(mode)


sys.modules[__name__].__class__ = _Deterministic
