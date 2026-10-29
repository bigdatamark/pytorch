# [MODULE DOC]
# Defines Equality, a tiny value class representing an equality constraint between two
# objects (used by the experimental graph type checker). No import-time side effects.
class Equality:
    # [CLASS DOC]
    # Holds a left-hand side (lhs) and right-hand side (rhs) of an equality constraint.
    # Equal to another Equality when both sides compare equal. Defines __eq__ without
    # __hash__, so instances are unhashable.
    def __init__(self, lhs: object, rhs: object) -> None:
        # [FUNCTION DOC] Stores lhs and rhs as attributes.
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the text "lhs = rhs".
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the same text as __str__.
        return f"{self.lhs} = {self.rhs}"

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] True only if other is an Equality with equal lhs and rhs; False for any other type.
        if isinstance(other, Equality):
            return self.lhs == other.lhs and self.rhs == other.rhs
        else:
            return False
