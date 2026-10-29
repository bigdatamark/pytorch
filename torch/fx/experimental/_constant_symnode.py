# [MODULE DOC] Python counterpart of the C++ constant symbolic-node implementation, providing
# ConstantIntNode, an always-constant integer symbolic node. Star-imports typing names.
from typing import *  # noqa: F403


# Python version of c10/core/ConstantSymNodeImpl.cpp
# This needs to exist because the Python version of nested int is not compatible
# with the C++ version of constant symnode.
class ConstantIntNode:
    # [CLASS DOC] A symbolic-node object that wraps a fixed Python integer in the val attribute.
    # It is never symbolic, is not a float, bool or nested int, and its arithmetic and
    # comparison methods delegate to the other operand with the roles swapped so that the
    # other node decides the result. Immutable by convention.
    def __init__(self, val: int) -> None:
        # [FUNCTION DOC] Store the wrapped integer in the val attribute.
        self.val = val

    def is_constant(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def maybe_as_int(self) -> int:
        # [FUNCTION DOC] Return the wrapped integer (never None).
        return self.val

    def is_int(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def is_float(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def is_bool(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def is_nested_int(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def clone(self) -> "ConstantIntNode":
        # [FUNCTION DOC] Return this same object (no copy), valid because it is immutable.
        return self

    def _str(self) -> str:
        # [FUNCTION DOC] HELPER: Decimal text of the wrapped integer.
        return str(self.val)

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Same text as the private string helper.
        return self._str()

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Same text as the private string helper.
        return self._str()

    def _graph_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Text used when printing graphs; the integer's decimal text.
        return self._str()

    def add(self, other: Any) -> Any:
        # [FUNCTION DOC] Sum with another node: asks other to add this node (commutative swap).
        return other.add(self)

    def sub(self, other: Any) -> Any:
        # [FUNCTION DOC] Compute this minus other, as the negation of other plus the wrapped value.
        # KNOWN ISSUE: passes the raw integer, not this node, to the other node's add, unlike
        # the other methods; this only works if the other node accepts plain integers.
        return other.neg().add(self.val)

    def mul(self, other: Any) -> Any:
        # [FUNCTION DOC] Product with another node: asks other to multiply by this node.
        return other.mul(self)

    def eq(self, other: Any) -> Any:
        # [FUNCTION DOC] Equality: delegates to the other node's equality with this node.
        return other.eq(self)

    def ne(self, other: Any) -> Any:
        # [FUNCTION DOC] Inequality: delegates to the other node's inequality with this node.
        return other.ne(self)

    def gt(self, other: Any) -> Any:
        # [FUNCTION DOC] This greater than other, evaluated as other less than this node.
        return other.lt(self)

    def lt(self, other: Any) -> Any:
        # [FUNCTION DOC] This less than other, evaluated as other greater than this node.
        return other.gt(self)

    def le(self, other: Any) -> Any:
        # [FUNCTION DOC] This at most other, evaluated as other at least this node.
        return other.ge(self)

    def ge(self, other: Any) -> Any:
        # [FUNCTION DOC] This at least other, evaluated as other at most this node.
        return other.le(self)

    def is_symbolic(self) -> bool:
        # [FUNCTION DOC] Always False, since the value is a known constant.
        return False

    def constant_int(self) -> int:
        # [FUNCTION DOC] Return the wrapped integer.
        return self.val

    def guard_int(self, file: str, line: int) -> int:
        # [FUNCTION DOC] Return the wrapped integer; no guard is recorded and the file and line
        # arguments are ignored.
        return self.val
