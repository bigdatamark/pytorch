# [MODULE DOC] Python implementation of the symbolic node behind nested ints (mirrors the C++ nested-int
# node). A nested int is identified by a tensor id and a multiplier coefficient, used for ragged dimensions.
# Exports NestedIntNode. Comparisons are only partly defined and raise a value error when indeterminate.
from typing import *  # noqa: F403

import torch
from torch.fx.experimental._constant_symnode import ConstantIntNode


__all__ = ["NestedIntNode"]


# Python version of aten/src/ATen/core/NestedIntSymNodeImpl.cpp
def _eq(lhs: Any, rhs: Any) -> bool:
    # [FUNCTION DOC] HELPER: True only if both are nested int nodes with the same id and coefficient.
    return (
        isinstance(lhs, NestedIntNode)
        and isinstance(rhs, NestedIntNode)
        and lhs.t_id == rhs.t_id
        and lhs.coeff == rhs.coeff
    )


def _ge(lhs: Any, rhs: Any) -> bool:
    # [FUNCTION DOC] HELPER: Greater-or-equal between nodes. Same id: compares coefficients. Nested vs plain constant:
    # true if the constant is at most 2 (nested on the left), false if the constant is below 2 (nested on the right).
    # Raises a value error when the relation is indeterminate or inputs are unsupported.
    if isinstance(rhs, NestedIntNode) and isinstance(lhs, NestedIntNode):
        if lhs.t_id == rhs.t_id:
            return lhs.coeff >= rhs.coeff
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(lhs, NestedIntNode):
        if rhs.is_constant() and rhs.constant_int() <= 2:
            return True
        raise ValueError("ge: relation is indeterminate")
    elif isinstance(rhs, NestedIntNode):
        if lhs.is_constant() and lhs.constant_int() < 2:
            return False
        raise ValueError("ge: relation is indeterminate")
    else:
        raise ValueError("inputs unsupported")


class NestedIntNode:
    # [CLASS DOC] Symbolic node for a nested int: fields t_id (identifier of the source tensor) and coeff
    # (integer multiplier). Reports itself as an int and nested int, never symbolic or constant. Supports
    # multiplication by constants and comparisons via the helpers; string form is j followed by id, with coefficient prefix.
    def __init__(self, t_id: int, coeff: int) -> None:
        # [FUNCTION DOC] Stores the tensor id and coefficient.
        self.t_id = t_id
        self.coeff = coeff

    def nested_int_coeff(self) -> int:
        # [FUNCTION DOC] Returns the coefficient.
        return self.coeff

    def maybe_as_int(self) -> Optional[int]:
        # [FUNCTION DOC] Always None: a nested int has no concrete integer value.
        return None

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
        # [FUNCTION DOC] Always True.
        return True

    def clone(self) -> "NestedIntNode":
        # [FUNCTION DOC] Returns the same node (immutable).
        return self

    def _str(self) -> Any:
        # [FUNCTION DOC] HELPER: Text form: j plus id, prefixed by coefficient and a star when coefficient is not 1.
        if self.coeff == 1:
            return f"j{self.t_id}"
        return f"{self.coeff}*j{self.t_id}"

    def str(self) -> Any:
        # [FUNCTION DOC] Returns the text form.
        return self._str()

    def __str__(self) -> Any:
        # [FUNCTION DOC] Returns the text form.
        return self._str()

    def __repr__(self) -> Any:
        # [FUNCTION DOC] Returns the text form.
        return self._str()

    def _graph_repr(self) -> Any:
        # [FUNCTION DOC] HELPER: Returns the text form for graph printing.
        return self._str()

    def mul(self, other: Any) -> "NestedIntNode":
        # [FUNCTION DOC] Multiplies by a constant node, returning a new node with the same id and scaled coefficient.
        # Raises a value error if other is not a constant.
        if other.is_constant():
            other = other.constant_int()
        else:
            raise ValueError(f"unsupported: {type(other)}")
        return NestedIntNode(self.t_id, self.coeff * other)

    def eq(self, other: Any) -> Any:
        # [FUNCTION DOC] Returns a constant boolean symbolic node for equality.
        return torch._C._get_constant_bool_symnode(_eq(self, other))

    def ne(self, other: Any) -> Any:
        # [FUNCTION DOC] Returns a constant boolean symbolic node for inequality.
        return torch._C._get_constant_bool_symnode(not _eq(self, other))

    def gt(self, other: Any) -> Any:
        # [FUNCTION DOC] Greater-than as negation of other being greater-or-equal; may raise a value error if indeterminate.
        return torch._C._get_constant_bool_symnode(not _ge(other, self))

    def lt(self, other: Any) -> Any:
        # [FUNCTION DOC] Less-than as negation of self being greater-or-equal to other; may raise a value error if indeterminate.
        return torch._C._get_constant_bool_symnode(not _ge(self, other))

    def le(self, other: Any) -> Any:
        # [FUNCTION DOC] Less-or-equal as other being greater-or-equal to self; may raise a value error if indeterminate.
        return torch._C._get_constant_bool_symnode(_ge(other, self))

    def ge(self, other: Any) -> Any:
        # [FUNCTION DOC] Greater-or-equal; may raise a value error if indeterminate.
        return torch._C._get_constant_bool_symnode(_ge(self, other))

    def is_symbolic(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def nested_int(self) -> int:
        # [FUNCTION DOC] Returns the tensor id.
        return self.t_id

    def is_constant(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def wrap_int(self, num: int) -> ConstantIntNode:
        # [FUNCTION DOC] Wraps a plain integer into a constant int node. Raises an assertion error if the value is not exactly an int.
        if type(num) is not int:
            raise AssertionError(
                f"Expected num to be int, but got {type(num).__name__}"
            )
        return ConstantIntNode(num)
