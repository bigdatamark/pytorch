# [MODULE DOC] Defines SingletonInt, a sympy atom representing a nested (jagged) integer
# symbol with a value id and an integer coefficient, mirroring the C++ NestedIntSymNodeImpl.
# Also registers multiple-dispatch comparison rules for it. Exports SingletonInt.
from typing import Any

import sympy
from sympy.multipledispatch import dispatch


__all__ = ["SingletonInt"]


class SingletonInt(sympy.AtomicExpr):
    # [CLASS DOC] Sympy atomic expression for a nested int. State: _val identifies the
    # singleton and _coeff is a multiplier. Equality requires equal value and coefficient;
    # only multiplication by an integer is supported, other arithmetic raises
    # NotImplementedError. Has no free symbols.
    # This is probably not super important unless we are in multiple dispatch
    # situations with other more exotic Expr types.
    _op_priority = 99999
    _val: int
    _coeff: int

    def __new__(
        cls, *args: Any, coeff: int | None = None, **kwargs: Any
    ) -> "SingletonInt":
        # [FUNCTION DOC] Construct the sympy atom, forwarding positional and keyword args to
        # the base class.
        # KNOWN ISSUE: the coeff argument is accepted but ignored here; the real value is set
        #   in __init__.
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    # The semantics of this class should match that of NestedIntSymNodeImpl in
    # c10/core/NestedIntSymNodeImpl.h
    def __init__(self, val: int, *, coeff: int = 1) -> None:
        # [FUNCTION DOC] Store val and coeff (default 1) on the instance and run the base
        # initializer.
        self._val = val
        self._coeff = coeff
        super().__init__()

    # See NOTE [ Inequalities with nested int ]
    def _eval_Eq(self, other: sympy.Basic) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Equality hook: returns sympy true only when other is a SingletonInt
        # with the same value and coefficient, otherwise sympy false.
        if (
            isinstance(other, SingletonInt)
            and other._val == self._val
            and self._coeff == other._coeff
        ):
            return sympy.true
        else:
            return sympy.false

    # This is necessary so that calling expr.free_symbols on exprs that contain
    # this Singleton does not error
    @property
    def free_symbols(self) -> set[sympy.Symbol]:
        # [FUNCTION DOC] Property returning an empty set so free-symbol queries on expressions
        # containing this atom do not fail.
        return set()

    def __mul__(self, other: int) -> "SingletonInt":
        # [FUNCTION DOC] Multiply by an integer: returns a new SingletonInt with the
        # coefficient scaled. Raises ValueError if other is a SingletonInt.
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other: int) -> "SingletonInt":
        # [FUNCTION DOC] Reflected multiply; same behavior as __mul__ (new scaled
        # SingletonInt, ValueError for a SingletonInt other).
        if isinstance(other, SingletonInt):
            raise ValueError(
                "SingletonInt cannot be multiplied by another SingletonInt"
            )
        return SingletonInt(self._val, coeff=self._coeff * other)

    # Make sure we promptly raise an error instead of falling back to building
    # an expression tree. There are probably more ops, how can we be exhaustive?
    def __add__(self, other: object) -> "SingletonInt":
        # [FUNCTION DOC] Always raises NotImplementedError so addition fails promptly instead
        # of building an expression tree.
        raise NotImplementedError("NYI")

    def __sub__(self, other: object) -> "SingletonInt":
        # [FUNCTION DOC] Always raises NotImplementedError so subtraction fails promptly.
        raise NotImplementedError("NYI")

    def __truediv__(self, other: object) -> "SingletonInt":
        # [FUNCTION DOC] Always raises NotImplementedError so true division fails promptly.
        raise NotImplementedError("NYI")

    def __floordiv__(self, other: object) -> "SingletonInt":
        # [FUNCTION DOC] Always raises NotImplementedError so floor division fails promptly.
        raise NotImplementedError("NYI")

    def __mod__(self, other: object) -> "SingletonInt":
        # [FUNCTION DOC] Always raises NotImplementedError so modulus fails promptly.
        raise NotImplementedError("NYI")


# See NOTE [ Inequalities with nested int ]
@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a: sympy.Integer, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    # [FUNCTION DOC] Dispatch rule for a >= b where a is a sympy Integer and b a SingletonInt:
    # returns sympy false if a is below 2 (a nested int is assumed at least 2), else raises
    # ValueError (indeterminate).
    if a < 2:
        return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, sympy.Integer)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: sympy.Integer) -> sympy.logic.boolalg.BooleanAtom:
    # [FUNCTION DOC] Dispatch rule for a >= b where a is a SingletonInt and b a sympy Integer:
    # returns sympy true if b is at most 2, else raises ValueError (indeterminate).
    if b <= 2:
        return sympy.true
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")


@dispatch(SingletonInt, SingletonInt)  # type: ignore[no-redef]
def _eval_is_ge(a: SingletonInt, b: SingletonInt) -> sympy.logic.boolalg.BooleanAtom:
    # [FUNCTION DOC] Dispatch rule for two SingletonInts: if the values match, compares
    # coefficients and returns sympy true or false; otherwise raises ValueError
    # (indeterminate).
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError("Symbolic SingletonInt: Relation is indeterminate")
