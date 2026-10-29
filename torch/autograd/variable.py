# mypy: allow-untyped-defs
# [MODULE DOC] Legacy compatibility shim: exposes Variable, a thin alias-like class whose isinstance check
# is true for any torch.Tensor, and re-exports the C++ ImperativeEngine. No import-time side effects
# beyond creating one engine instance on the Variable class.
import torch
from torch._C import _ImperativeEngine as ImperativeEngine


__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    # [CLASS DOC] Metaclass making isinstance checks against Variable succeed for any torch.Tensor.
    def __instancecheck__(cls, other):
        # [FUNCTION DOC] Returns True if other is a torch.Tensor, regardless of cls.
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    # [CLASS DOC] Deprecated legacy Variable type; construction is handled by the C++ legacy base.
    # Holds the shared _execution_engine (an ImperativeEngine instance) as a class attribute.
    _execution_engine = ImperativeEngine()
