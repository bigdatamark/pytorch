# [MODULE DOC] Lightweight gradual-type vocabulary for torch.fx type annotations:
#   TensorType (a tensor type described by a sequence of dimensions), the Dyn singleton
#   meaning no type information, and the relations is_consistent and is_more_precise.
#   Imports the unification Var type; no other import-time side effects.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

from ._compatibility import compatibility


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx.experimental.migrate_gradual_types.constraint import DVar


__all__ = ["Dyn", "TensorType", "is_consistent", "is_more_precise"]


@compatibility(is_backward_compatible=False)
class TensorType:
    # [CLASS DOC] Type for tensors holding a sequence of dimensions, each an int, Dyn,
    #   or a constraint variable. Stores the sequence in __args__ and itself in
    #   __origin__ so typing-style introspection works. Subscripting the class builds an
    #   instance. Equality compares dimension lists.
    # KNOWN ISSUE: __eq__ is defined without __hash__, so instances are unhashable.
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim: Sequence[Any]) -> None:
        # [FUNCTION DOC] Store the given dimension sequence as __args__ and set
        #   __origin__ to the TensorType class. No validation.
        self.__origin__ = TensorType
        self.__args__ = dim

    @property
    def dims(self) -> Sequence[DVar | int | _DynType]:
        # [FUNCTION DOC] Property returning the stored dimension sequence (__args__)
        #   unchanged.
        return self.__args__

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return a string of the form TensorType[dims].
        return f"TensorType[{self.dims}]"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal only if other is an instance of this class and the
        #   dimension lists compare equal element by element; any other object compares
        #   unequal.
        if isinstance(other, self.__class__):
            return list(self.dims) == list(other.dims)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args: object) -> TensorType:
        # [FUNCTION DOC] Implements subscripting of the class: accepts the subscript
        #   argument(s); a single tuple argument is unpacked, then a TensorType is built
        #   from the tuple of arguments and returned.
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))


class _DynType:
    # [CLASS DOC] HELPER: Type of the Dyn singleton, standing for absence of type
    #   information. All instances compare equal to each other.
    # KNOWN ISSUE: __eq__ without __hash__ makes instances unhashable.
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Set the __name__ attribute to the string _DynType.
        self.__name__ = "_DynType"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equal to any instance of this class, unequal to
        #   everything else.
        return isinstance(other, self.__class__)

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Return the string Dyn.
        return "Dyn"

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Return the string Dyn.
        return "Dyn"


Dyn = _DynType()


@compatibility(is_backward_compatible=False)
def is_consistent(t1: object, t2: object) -> bool:
    # [FUNCTION DOC] Gradual-typing consistency check between two types; returns True or
    #   False. True if equal, if either side is Dyn or a unification Var, or if both are
    #   TensorTypes of the same rank whose dimensions are pairwise consistent; otherwise
    #   False. Reflexive and symmetric, not transitive. No exceptions or side effects.
    """
    A binary relation denoted by ~ that determines if t1 is consistent with t2.
    The relation is reflexive, symmetric but not transitive.
    returns True if t1 and t2 are consistent and False otherwise.
    Example:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """

    if t1 == t2:
        return True

    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_consistent(elem1, elem2) for elem1, elem2 in zip(t1.dims, t2.dims)
        )
    else:
        return False


@compatibility(is_backward_compatible=False)
def is_more_precise(t1: object, t2: object) -> bool:
    # [FUNCTION DOC] Check whether t1 is at least as precise as t2; returns True or
    #   False. True if equal, if t2 is Dyn, or if both are TensorTypes of equal rank
    #   whose dimensions are pairwise more precise; otherwise False (a Dyn t1 against a
    #   concrete t2 gives False). Note that Var is not treated specially, unlike
    #   is_consistent.
    # KNOWN ISSUE: the docstring examples mix >= and <= notation and are hard to
    #   reconcile with the code, which only implements the t1 as precise as t2
    #   direction.
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    if t1 == t2:
        return True

    if isinstance(t2, _DynType):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.dims, t2.dims)
        )

    else:
        return False
