# [MODULE DOC] Defines sympy singleton numbers for integer-valued infinities: IntInfinity
# (int_oo) and NegativeIntInfinity (-int_oo). Unlike sympy.oo they report is_integer, so they
# can bound integer ranges (e.g. in value range analysis). Also defines the int_oo alias and
# the is_infinite predicate. Importing creates the singleton S.IntInfinity (via int_oo).
from typing import Any

import mpmath.libmp as mlib  # type: ignore[import-untyped]
import sympy
from sympy import Expr
from sympy.core.decorators import _sympifyit
from sympy.core.expr import AtomicExpr
from sympy.core.numbers import Number
from sympy.core.parameters import global_parameters
from sympy.core.singleton import S, Singleton


# pyrefly: ignore [invalid-inheritance]
class IntInfinity(Number, metaclass=Singleton):
    # [CLASS DOC] Sympy singleton Number for positive integer infinity. Stateless (no slots).
    # Arithmetic with other Numbers follows infinity rules (NaN for indeterminate forms, sympy
    # infinities absorb it), comparisons treat it as below sympy.oo but above all else, and
    # equality is identity with the singleton. Has a high _op_priority so its operators win.
    r"""Positive integer infinite quantity.

    Integer infinity is a value in an extended integers which
    is greater than all other integers.  We distinguish it from
    sympy's existing notion of infinity in that it reports that
    it is_integer.

    Infinity is a singleton, and can be accessed by ``S.IntInfinity``,
    or can be imported as ``int_oo``.
    """

    # NB: We can't actually mark this as infinite, as integer and infinite are
    # inconsistent assumptions in sympy.  We also report that we are complex,
    # different from sympy.oo

    is_integer = True
    is_commutative = True
    is_number = True
    is_extended_real = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    __slots__: tuple[str, ...] = ()

    def __new__(cls) -> "IntInfinity":
        # [FUNCTION DOC] Creates the singleton instance without any arguments or state.
        return AtomicExpr.__new__(cls)

    def _sympystr(self, printer: Any) -> str:
        # [FUNCTION DOC] HELPER: Printer hook; returns the text int_oo.
        return "int_oo"

    def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic | None:
        # [FUNCTION DOC] HELPER: Substitution hook; returns new if old equals this singleton, else
        # None (meaning no change).
        if self == old:
            return new
        return None

    # We could do these, not sure about it
    """
    def _eval_evalf(self, prec=None):
        return Float('inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Addition. With a Number (and evaluation enabled): sympy +/- infinity
        # wins, -int_oo or NaN gives NaN, anything else gives int_oo. Otherwise defers to
        # Number addition.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NegativeInfinity):
                return other
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Subtraction. With a Number (and evaluation enabled): minus sympy oo gives
        # -oo, minus -oo gives oo, minus int_oo or NaN gives NaN, otherwise int_oo. Else defers to
        # Number subtraction.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.NegativeInfinity
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Reflected subtraction (other minus int_oo), computed as the sum of
        # other and the negation of this value.
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Multiplication. With a Number (and evaluation enabled): zero or NaN gives
        # NaN, positive gives int_oo, otherwise -int_oo. Else defers to Number multiplication.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeIntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] True division. With a Number (and evaluation enabled): any infinity or NaN
        # gives NaN; nonnegative divisor gives sympy oo, else -oo (float results, not int_oo).
        # Else defers to Number division.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return S.Infinity  # truediv produces float
            return S.NegativeInfinity  # truediv produces float
        return Number.__truediv__(self, other)

    def __abs__(self) -> sympy.Expr:
        # [FUNCTION DOC] Absolute value; returns int_oo.
        return S.IntInfinity

    def __neg__(self) -> sympy.Expr:
        # [FUNCTION DOC] Negation; returns -int_oo.
        return S.NegativeIntInfinity

    def _eval_power(self, expt: sympy.Expr) -> sympy.Expr | None:
        # [FUNCTION DOC] Power hook. Positive exponent gives int_oo, negative gives zero, NaN or
        # complex infinity exponent gives NaN. For non-real numeric exponents, branches on the sign
        # of the real part (complex infinity, zero, or NaN) and otherwise evaluates numerically.
        # Returns None if the exponent is not decidable.
        if expt.is_extended_positive:
            return S.IntInfinity
        if expt.is_extended_negative:
            return S.Zero
        if expt is S.NaN:
            return S.NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re

            expt_real = re(expt)
            if expt_real.is_positive:
                return S.ComplexInfinity
            if expt_real.is_negative:
                return S.Zero
            if expt_real.is_zero:
                return S.NaN

            return self ** expt.evalf()
        return None

    def _as_mpf_val(self, prec: int) -> Any:
        # [FUNCTION DOC] HELPER: Float conversion hook; returns the mpmath positive infinity.
        return mlib.finf

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash; delegates to the base class.
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equality; true only for the IntInfinity singleton itself.
        return other is S.IntInfinity

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC] Inequality; true unless other is the IntInfinity singleton.
        return other is not S.IntInfinity

    def __gt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Greater-than; false versus sympy oo or int_oo, else true.
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __ge__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Greater-or-equal; false versus sympy oo, true versus int_oo and all else.
        if other is S.Infinity:
            return sympy.false  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    def __lt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Less-than; true versus sympy oo, false versus int_oo and all else.
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __le__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Less-or-equal; true versus sympy oo or int_oo, false otherwise.
        if other is S.Infinity:
            return sympy.true  # sympy.oo > int_oo
        elif other is S.IntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Modulo; returns NaN for any sympy expression, NotImplemented otherwise.
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self) -> "IntInfinity":
        # [FUNCTION DOC] Floor; returns itself.
        return self

    def ceiling(self) -> "IntInfinity":
        # [FUNCTION DOC] Ceiling; returns itself.
        return self


int_oo = S.IntInfinity


def is_infinite(expr: sympy.Basic) -> bool:
    # [FUNCTION DOC] Returns whether expr is one of sympy oo, -oo, int_oo or -int_oo (by
    # membership test). Needed because the is_finite assumption is true for the integer
    # infinities.
    """Check if an expression is any type of infinity (positive or negative).

    This handles both sympy's built-in infinities (oo, -oo) and PyTorch's
    integer infinities (int_oo, -int_oo).

    Note: We cannot rely on sympy's is_finite property because IntInfinity
    and NegativeIntInfinity have is_integer=True, which implies is_finite=True
    in sympy's assumption system.
    """
    return expr in (
        S.Infinity,
        S.NegativeInfinity,
        S.IntInfinity,
        S.NegativeIntInfinity,
    )


# pyrefly: ignore [invalid-inheritance]
class NegativeIntInfinity(Number, metaclass=Singleton):
    # [CLASS DOC] Sympy singleton Number for negative integer infinity (mirror of IntInfinity).
    # Stateless (no slots), integer-valued, with infinity arithmetic rules, comparisons treating
    # it as above -oo and below all else, and identity-based equality.
    """Negative integer infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    IntInfinity
    """

    # Ensure we get dispatched to before plain numbers
    _op_priority = 100.0

    is_integer = True
    is_extended_real = True
    is_commutative = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__: tuple[str, ...] = ()

    def __new__(cls) -> "NegativeIntInfinity":
        # [FUNCTION DOC] Creates the singleton instance without any arguments or state.
        return AtomicExpr.__new__(cls)

    def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic | None:
        # [FUNCTION DOC] HELPER: Substitution hook; returns new if old equals this singleton, else
        # None (meaning no change).
        if self == old:
            return new
        return None

    def _sympystr(self, printer: Any) -> str:
        # [FUNCTION DOC] HELPER: Printer hook; returns the text -int_oo.
        return "-int_oo"

    """
    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)
    """

    @_sympifyit("other", NotImplemented)
    def __add__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Addition. With a Number (and evaluation enabled): sympy oo gives oo,
        # int_oo or NaN gives NaN, anything else gives -int_oo. Else defers to Number addition.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity:
                return S.Infinity
            if other in (S.IntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)

    __radd__ = __add__

    @_sympifyit("other", NotImplemented)
    def __sub__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Subtraction. With a Number (and evaluation enabled): minus -oo gives oo,
        # minus -int_oo or NaN gives NaN, otherwise -int_oo. Else defers to Number subtraction.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NegativeInfinity:
                return S.Infinity
            if other in (S.NegativeIntInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit("other", NotImplemented)
    def __rsub__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Reflected subtraction (other minus -int_oo), computed as the sum of
        # other and the negation of this value.
        return (-self).__add__(other)

    @_sympifyit("other", NotImplemented)
    def __mul__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Multiplication. With a Number (and evaluation enabled): zero or NaN gives
        # NaN, positive gives -int_oo, otherwise int_oo. Else defers to Number multiplication.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.IntInfinity
        return Number.__mul__(self, other)

    __rmul__ = __mul__

    @_sympifyit("other", NotImplemented)
    def __truediv__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] True division. With a Number (and evaluation enabled): any infinity or NaN
        # gives NaN; nonnegative divisor gives -int_oo, else sympy oo (float result).
        # Else defers to Number division.
        # KNOWN ISSUE: the nonnegative case returns -int_oo while the mirror IntInfinity
        # code returns a sympy float infinity for the same case.
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (
                S.Infinity,
                S.IntInfinity,
                S.NegativeInfinity,
                S.NegativeIntInfinity,
                S.NaN,
            ):
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity  # truediv returns float
        return Number.__truediv__(self, other)

    def __abs__(self) -> sympy.Expr:
        # [FUNCTION DOC] Absolute value; returns int_oo.
        return S.IntInfinity

    def __neg__(self) -> sympy.Expr:
        # [FUNCTION DOC] Negation; returns int_oo.
        return S.IntInfinity

    def _eval_power(self, expt: sympy.Expr) -> sympy.Expr | None:
        # [FUNCTION DOC] Power hook for numeric exponents. Any infinity or NaN exponent gives
        # NaN; a positive integer exponent gives -int_oo if odd, else int_oo. Otherwise
        # combines the powers of int_oo and minus one, handling zero and complex-infinity
        # cases. Returns None for non-numeric exponents.
        if expt.is_number:
            if expt in (
                S.NaN,
                S.Infinity,
                S.NegativeInfinity,
                S.IntInfinity,
                S.NegativeIntInfinity,
            ):
                return S.NaN

            if isinstance(expt, sympy.Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeIntInfinity
                else:
                    return S.IntInfinity

            inf_part = S.IntInfinity**expt
            s_part = S.NegativeOne**expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            if (
                inf_part is S.ComplexInfinity
                and s_part.is_finite
                and not s_part.is_zero
            ):
                return S.ComplexInfinity
            return s_part * inf_part
        return None

    def _as_mpf_val(self, prec: int) -> Any:
        # [FUNCTION DOC] HELPER: Float conversion hook; returns the mpmath negative infinity.
        return mlib.fninf

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash; delegates to the base class.
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equality; true only for the NegativeIntInfinity singleton itself.
        return other is S.NegativeIntInfinity

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC] Inequality; true unless other is the NegativeIntInfinity singleton.
        return other is not S.NegativeIntInfinity

    def __gt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Greater-than; true versus sympy -oo, false versus -int_oo and all else.
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.false

    def __ge__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Greater-or-equal; true versus sympy -oo or -int_oo, false otherwise.
        if other is S.NegativeInfinity:
            return sympy.true  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.false

    def __lt__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Less-than; false versus sympy -oo or -int_oo, true otherwise.
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.false  # consistency with sympy.oo
        else:
            return sympy.true

    def __le__(self, other: sympy.Expr) -> sympy.logic.boolalg.BooleanAtom:
        # [FUNCTION DOC] Less-or-equal; false versus sympy -oo, true versus -int_oo and all else.
        if other is S.NegativeInfinity:
            return sympy.false  # -sympy.oo < -int_oo
        elif other is S.NegativeIntInfinity:
            return sympy.true  # consistency with sympy.oo
        else:
            return sympy.true

    @_sympifyit("other", NotImplemented)
    def __mod__(self, other: sympy.Expr) -> sympy.Expr:
        # [FUNCTION DOC] Modulo; returns NaN for any sympy expression, NotImplemented otherwise.
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self) -> "NegativeIntInfinity":
        # [FUNCTION DOC] Floor; returns itself.
        return self

    def ceiling(self) -> "NegativeIntInfinity":
        # [FUNCTION DOC] Ceiling; returns itself.
        return self

    def as_powers_dict(self) -> dict[sympy.Expr, int]:
        # [FUNCTION DOC] Returns the factor mapping with minus one and int_oo each to power one, so
        # this value is treated as the product of -1 and int_oo.
        return {S.NegativeOne: 1, S.IntInfinity: 1}
