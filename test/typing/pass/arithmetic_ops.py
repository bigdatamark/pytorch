# [MODULE DOC] Static typing sample for arithmetic operators between a tensor, an int, a float
# and a bool. It defines one constant of each kind, then asserts the deduced type of unary,
# comparison, arithmetic, matrix-multiplication and integer-only bitwise/shift operators in both
# operand orders; the expected result is always Tensor, and lines carrying type-ignore markers
# or a should-fail note record places where the checker currently disagrees. It also defines a
# helper class overriding every binary magic method to return itself, to show what a fully typed
# operand would produce. The companion runtime test test_python_operators reads the assertion
# lines mentioning the tensor constant to verify coverage, so such lines must keep the exact
# shape of assert_type calls.
from typing import TypeAlias
from typing_extensions import assert_type

from torch import randn, Tensor


# Test deduced types of arithmetic operations between tensors, ints, floats and bools
# The expected type should always be `Tensor`, but isn't.
# See https://github.com/pytorch/pytorch/issues/145838

TENSOR, INT, FLOAT, BOOL = randn(3), 2, 1.5, True

#
# Unary ops
#

assert_type(+TENSOR, Tensor)
assert_type(-TENSOR, Tensor)
assert_type(~TENSOR, Tensor)

#
# Binary ops that return a boolean
#

# Operator ==
assert_type(TENSOR == TENSOR, Tensor)
assert_type(TENSOR == BOOL, Tensor)
assert_type(BOOL == TENSOR, bool)  # Should be Tensor
assert_type(TENSOR == INT, Tensor)
assert_type(INT == TENSOR, bool)  # Should be Tensor
assert_type(TENSOR == FLOAT, Tensor)
assert_type(FLOAT == TENSOR, bool)  # Should be Tensor

# Operator !=
assert_type(TENSOR != TENSOR, Tensor)
assert_type(TENSOR != BOOL, Tensor)
assert_type(BOOL != TENSOR, bool)  # Should be Tensor
assert_type(TENSOR != INT, Tensor)
assert_type(INT != TENSOR, bool)  # Should be Tensor
assert_type(TENSOR != FLOAT, Tensor)
assert_type(FLOAT != TENSOR, bool)  # Should be Tensor

# Operator <
assert_type(TENSOR < TENSOR, Tensor)
assert_type(TENSOR < BOOL, Tensor)
assert_type(BOOL < TENSOR, Tensor)
assert_type(TENSOR < INT, Tensor)
assert_type(INT < TENSOR, Tensor)
assert_type(TENSOR < FLOAT, Tensor)
assert_type(FLOAT < TENSOR, Tensor)

# Operator >
assert_type(TENSOR > TENSOR, Tensor)
assert_type(TENSOR > BOOL, Tensor)
assert_type(BOOL > TENSOR, Tensor)
assert_type(TENSOR > INT, Tensor)
assert_type(INT > TENSOR, Tensor)
assert_type(TENSOR > FLOAT, Tensor)
assert_type(FLOAT > TENSOR, Tensor)

# Operator <=
assert_type(TENSOR <= TENSOR, Tensor)
assert_type(TENSOR <= BOOL, Tensor)
assert_type(BOOL <= TENSOR, Tensor)
assert_type(TENSOR <= INT, Tensor)
assert_type(INT <= TENSOR, Tensor)
assert_type(TENSOR <= FLOAT, Tensor)
assert_type(FLOAT <= TENSOR, Tensor)

# Operator >=
assert_type(TENSOR >= TENSOR, Tensor)
assert_type(TENSOR >= BOOL, Tensor)
assert_type(BOOL >= TENSOR, Tensor)
assert_type(TENSOR >= INT, Tensor)
assert_type(INT >= TENSOR, Tensor)
assert_type(TENSOR >= FLOAT, Tensor)
assert_type(FLOAT >= TENSOR, Tensor)

#
# Binary ops that take and return ints or floats
#

# Operator +
assert_type(TENSOR + TENSOR, Tensor)
assert_type(TENSOR + BOOL, Tensor)
assert_type(BOOL + TENSOR, Tensor)
assert_type(TENSOR + INT, Tensor)
assert_type(INT + TENSOR, Tensor)
assert_type(TENSOR + FLOAT, Tensor)
assert_type(FLOAT + TENSOR, Tensor)

# Operator -
assert_type(TENSOR - TENSOR, Tensor)
assert_type(TENSOR - BOOL, Tensor)
assert_type(BOOL - TENSOR, Tensor)
assert_type(TENSOR - INT, Tensor)
assert_type(INT - TENSOR, Tensor)
assert_type(TENSOR - FLOAT, Tensor)
assert_type(FLOAT - TENSOR, Tensor)

# Operator *
assert_type(TENSOR * TENSOR, Tensor)
assert_type(TENSOR * BOOL, Tensor)
assert_type(BOOL * TENSOR, Tensor)
assert_type(TENSOR * INT, Tensor)
assert_type(INT * TENSOR, Tensor)
assert_type(TENSOR * FLOAT, Tensor)
assert_type(FLOAT * TENSOR, Tensor)

# Operator //
assert_type(TENSOR // TENSOR, Tensor)
assert_type(TENSOR // BOOL, Tensor)
assert_type(BOOL // TENSOR, Tensor)
assert_type(TENSOR // INT, Tensor)
assert_type(INT // TENSOR, Tensor)
assert_type(TENSOR // FLOAT, Tensor)
assert_type(FLOAT // TENSOR, Tensor)

# Operator /
assert_type(TENSOR / TENSOR, Tensor)
assert_type(TENSOR / BOOL, Tensor)
assert_type(BOOL / TENSOR, Tensor)
assert_type(TENSOR / INT, Tensor)
assert_type(INT / TENSOR, Tensor)
assert_type(TENSOR / FLOAT, Tensor)
assert_type(FLOAT / TENSOR, Tensor)

# Operator %
assert_type(TENSOR % TENSOR, Tensor)
assert_type(TENSOR % BOOL, Tensor)
assert_type(BOOL % TENSOR, Tensor)
assert_type(TENSOR % INT, Tensor)
assert_type(INT % TENSOR, Tensor)
assert_type(TENSOR % FLOAT, Tensor)
assert_type(FLOAT % TENSOR, Tensor)

# Operator **
assert_type(TENSOR**TENSOR, Tensor)
assert_type(TENSOR**BOOL, Tensor)
assert_type(BOOL**TENSOR, Tensor)
assert_type(TENSOR**INT, Tensor)
assert_type(INT**TENSOR, Tensor)
assert_type(TENSOR**FLOAT, Tensor)
assert_type(FLOAT**TENSOR, Tensor)

#
# Matrix multiplication
#

# Operator @
assert_type(TENSOR @ TENSOR, Tensor)
assert_type(TENSOR @ BOOL, Tensor)  # Should fail type checking
assert_type(BOOL @ TENSOR, Tensor)  # type: ignore[operator]
assert_type(TENSOR @ INT, Tensor)  # Should fail type checking
assert_type(INT @ TENSOR, Tensor)  # type: ignore[operator]
assert_type(TENSOR @ FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT @ TENSOR, Tensor)  # type: ignore[operator]

#
# Binary ops that take and return ints only
#

# Operator <<
assert_type(TENSOR << TENSOR, Tensor)
assert_type(TENSOR << BOOL, Tensor)
assert_type(BOOL << TENSOR, Tensor)
assert_type(TENSOR << INT, Tensor)
assert_type(INT << TENSOR, Tensor)
assert_type(TENSOR << FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT << TENSOR, Tensor)  # Should fail type checking

# Operator >>
assert_type(TENSOR >> TENSOR, Tensor)
assert_type(TENSOR >> BOOL, Tensor)
assert_type(BOOL >> TENSOR, Tensor)
assert_type(TENSOR >> INT, Tensor)
assert_type(INT >> TENSOR, Tensor)
assert_type(TENSOR >> FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT >> TENSOR, Tensor)  # Should fail type checking

# Operator &
assert_type(TENSOR & TENSOR, Tensor)
assert_type(TENSOR & BOOL, Tensor)
assert_type(BOOL & TENSOR, Tensor)
assert_type(TENSOR & INT, Tensor)
assert_type(INT & TENSOR, Tensor)
assert_type(TENSOR & FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT & TENSOR, Tensor)  # type: ignore[operator]

# Operator |
assert_type(TENSOR | TENSOR, Tensor)
assert_type(TENSOR | BOOL, Tensor)
assert_type(BOOL | TENSOR, Tensor)
assert_type(TENSOR | INT, Tensor)
assert_type(INT | TENSOR, Tensor)
assert_type(TENSOR | FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT | TENSOR, Tensor)  # type: ignore[operator]

# Operator ^
assert_type(TENSOR ^ TENSOR, Tensor)
assert_type(TENSOR ^ BOOL, Tensor)
assert_type(BOOL ^ TENSOR, Tensor)
assert_type(TENSOR ^ INT, Tensor)
assert_type(INT ^ TENSOR, Tensor)
assert_type(TENSOR ^ FLOAT, Tensor)  # Should fail type checking
assert_type(FLOAT ^ TENSOR, Tensor)  # type: ignore[operator]


NUMBER: TypeAlias = int | float | bool


class Binary:
    # [CLASS DOC] HELPER: Demonstration type whose binary magic methods all accept a number
    # (int, float or bool) and return the same instance, annotated as returning this class. It
    # has no state. Used to check what result type a checker deduces for each binary operator
    # and its reflected form when only this class defines the method. Equality and inequality
    # are overridden to return this class instead of bool, which requires override ignores.
    """
    This class demonstrates what is possible by overriding every magic method
    relating to binary operations.
    """

    def __add__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for addition with the instance on the left. Takes
        # a number, ignores it and returns the same instance; no side effects.
        return self

    def __and__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for bitwise and with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __div__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for the legacy name for division; Python 3 true
        # division uses the other division method, so this one is not picked for the slash
        # operator. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __eq__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for equality comparison; returns an instance
        # rather than a bool. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __floordiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for floor division with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __ge__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for greater-or-equal comparison; returns an
        # instance rather than a bool. Takes a number, ignores it and returns the same instance;
        # no side effects.
        return self

    def __gt__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for greater-than comparison; returns an instance
        # rather than a bool. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __le__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for less-or-equal comparison; returns an instance
        # rather than a bool. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __lshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for left shift with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __lt__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for less-than comparison; returns an instance
        # rather than a bool. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __mod__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for remainder with the instance on the left. Takes
        # a number, ignores it and returns the same instance; no side effects.
        return self

    def __mul__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for multiplication with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __ne__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for inequality comparison; returns an instance
        # rather than a bool. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __or__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for bitwise or with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __pow__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for power with the instance as base. Takes a
        # number, ignores it and returns the same instance; no side effects.
        return self

    def __radd__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected addition, used when a number is on
        # the left. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rand__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected bitwise and, used when a number is
        # on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __rdiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected legacy division name; not used for
        # the slash operator in Python 3. Takes a number, ignores it and returns the same
        # instance; no side effects.
        return self

    def __rfloordiv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected floor division, used when a number
        # is on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __rlshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected left shift, used when a number is on
        # the left. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rmod__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected remainder, used when a number is on
        # the left. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rmul__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected multiplication, used when a number
        # is on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __ror__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected bitwise or, used when a number is on
        # the left. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rpow__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected power, used when a number is the
        # base. Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rrshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected right shift, used when a number is
        # on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __rshift__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for right shift with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __rsub__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected subtraction, used when a number is
        # on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __rtruediv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected true division, used when a number is
        # on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __rxor__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for reflected bitwise xor, used when a number is
        # on the left. Takes a number, ignores it and returns the same instance; no side
        # effects.
        return self

    def __sub__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for subtraction with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self

    def __truediv__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for true division (slash operator) with the
        # instance on the left. Takes a number, ignores it and returns the same instance; no
        # side effects.
        return self

    def __xor__(self, other: NUMBER) -> "Binary":  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Magic method for bitwise xor with the instance on the left.
        # Takes a number, ignores it and returns the same instance; no side effects.
        return self


BINARY = Binary()

assert_type(BINARY + INT, Binary)
assert_type(BINARY & INT, Binary)
assert_type(BINARY / INT, Binary)
assert_type(BINARY == INT, Binary)
assert_type(BINARY // INT, Binary)
assert_type(BINARY >= INT, Binary)
assert_type(BINARY > INT, Binary)
assert_type(BINARY <= INT, Binary)
assert_type(BINARY << INT, Binary)
assert_type(BINARY < INT, Binary)
assert_type(BINARY % INT, Binary)
assert_type(BINARY * INT, Binary)
assert_type(BINARY != INT, Binary)
assert_type(BINARY | INT, Binary)
assert_type(BINARY**INT, Binary)
assert_type(BINARY >> INT, Binary)
assert_type(BINARY - INT, Binary)
assert_type(BINARY ^ INT, Binary)

assert_type(INT + BINARY, Binary)
assert_type(INT & BINARY, Binary)
assert_type(INT / BINARY, Binary)
assert_type(INT == BINARY, bool)
assert_type(INT // BINARY, Binary)
assert_type(INT >= BINARY, Binary)
assert_type(INT > BINARY, Binary)
assert_type(INT <= BINARY, Binary)
assert_type(INT << BINARY, Binary)
assert_type(INT < BINARY, Binary)
assert_type(INT % BINARY, Binary)
assert_type(INT * BINARY, Binary)
assert_type(INT != BINARY, bool)
assert_type(INT | BINARY, Binary)
assert_type(INT**BINARY, Binary)
assert_type(INT >> BINARY, Binary)
assert_type(INT - BINARY, Binary)
assert_type(INT ^ BINARY, Binary)

assert_type(BINARY + FLOAT, Binary)
assert_type(BINARY & FLOAT, Binary)
assert_type(BINARY / FLOAT, Binary)
assert_type(BINARY == FLOAT, Binary)
assert_type(BINARY // FLOAT, Binary)
assert_type(BINARY >= FLOAT, Binary)
assert_type(BINARY > FLOAT, Binary)
assert_type(BINARY <= FLOAT, Binary)
assert_type(BINARY << FLOAT, Binary)
assert_type(BINARY < FLOAT, Binary)
assert_type(BINARY % FLOAT, Binary)
assert_type(BINARY * FLOAT, Binary)
assert_type(BINARY != FLOAT, Binary)
assert_type(BINARY | FLOAT, Binary)
assert_type(BINARY**FLOAT, Binary)
assert_type(BINARY >> FLOAT, Binary)
assert_type(BINARY - FLOAT, Binary)
assert_type(BINARY ^ FLOAT, Binary)

assert_type(FLOAT + BINARY, Binary)
assert_type(FLOAT & BINARY, Binary)
assert_type(FLOAT / BINARY, Binary)
assert_type(FLOAT == BINARY, bool)
assert_type(FLOAT // BINARY, Binary)
assert_type(FLOAT >= BINARY, Binary)
assert_type(FLOAT > BINARY, Binary)
assert_type(FLOAT <= BINARY, Binary)
assert_type(FLOAT << BINARY, Binary)
assert_type(FLOAT < BINARY, Binary)
assert_type(FLOAT % BINARY, Binary)
assert_type(FLOAT * BINARY, Binary)
assert_type(FLOAT != BINARY, bool)
assert_type(FLOAT | BINARY, Binary)
assert_type(FLOAT**BINARY, Binary)
assert_type(FLOAT >> BINARY, Binary)
assert_type(FLOAT - BINARY, Binary)
assert_type(FLOAT ^ BINARY, Binary)

assert_type(BINARY + BOOL, Binary)
assert_type(BINARY & BOOL, Binary)
assert_type(BINARY / BOOL, Binary)
assert_type(BINARY == BOOL, Binary)
assert_type(BINARY // BOOL, Binary)
assert_type(BINARY >= BOOL, Binary)
assert_type(BINARY > BOOL, Binary)
assert_type(BINARY <= BOOL, Binary)
assert_type(BINARY << BOOL, Binary)
assert_type(BINARY < BOOL, Binary)
assert_type(BINARY % BOOL, Binary)
assert_type(BINARY * BOOL, Binary)
assert_type(BINARY != BOOL, Binary)
assert_type(BINARY | BOOL, Binary)
assert_type(BINARY**BOOL, Binary)
assert_type(BINARY >> BOOL, Binary)
assert_type(BINARY - BOOL, Binary)
assert_type(BINARY ^ BOOL, Binary)

assert_type(BOOL + BINARY, Binary)
assert_type(BOOL & BINARY, Binary)
assert_type(BOOL / BINARY, Binary)
assert_type(BOOL == BINARY, bool)
assert_type(BOOL // BINARY, Binary)
assert_type(BOOL >= BINARY, Binary)
assert_type(BOOL > BINARY, Binary)
assert_type(BOOL <= BINARY, Binary)
assert_type(BOOL << BINARY, Binary)
assert_type(BOOL < BINARY, Binary)
assert_type(BOOL % BINARY, Binary)
assert_type(BOOL * BINARY, Binary)
assert_type(BOOL != BINARY, bool)
assert_type(BOOL | BINARY, Binary)
assert_type(BOOL**BINARY, Binary)
assert_type(BOOL >> BINARY, Binary)
assert_type(BOOL - BINARY, Binary)
assert_type(BOOL ^ BINARY, Binary)
