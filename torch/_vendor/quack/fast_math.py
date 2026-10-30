# Copyright (c) 2025, Tri Dao.
# [MODULE DOC] Defines FastDivmod, a variant of the cute fast divmod divisor that also keeps the
# original divisor so it survives host to device MLIR value transfer.

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Integer
from cutlass.cutlass_dsl import dsl_user_op


class FastDivmod(cute.FastDivmodDivisor):
    # [CLASS DOC] Fast division/modulo divisor object. Extends the cute divisor with a public
    # divisor attribute (the raw divisor value, possibly a runtime value) alongside the
    # precomputed magic-number state of the base class, and round-trips both through MLIR values.
    """We store the divisor along with the FastDivmodDivisor."""

    @dsl_user_op
    def __init__(
        self,
        divisor: Integer,
        is_power_of_2: bool = None,
        *,
        loc=None,
        ip=None,
    ):
        # [FUNCTION DOC] Initialize the base fast divmod divisor with the given divisor and
        # optional power-of-two hint, then remember the raw divisor in the divisor attribute.
        super().__init__(divisor, is_power_of_2=is_power_of_2, loc=loc, ip=ip)
        self.divisor = divisor

    def __extract_mlir_values__(self):
        # [FUNCTION DOC] Return the list of MLIR values for the base divisor state followed by the
        # values of the raw divisor, used when passing the object from host to device.
        """Extract MLIR values for Host->Device transfer."""
        return [self._divisor] + cutlass.extract_mlir_values(self.divisor)

    def __new_from_mlir_values__(self, values):
        # [FUNCTION DOC] Build a new FastDivmod without calling the constructor: the first value
        # becomes the base divisor state and the remaining values rebuild the raw divisor
        # attribute. Returns the new object; self is not modified.
        """Reconstruct FastDivmodDivisor from MLIR values."""
        new_obj = object.__new__(FastDivmod)
        new_obj._divisor = values[0]
        new_obj.divisor = cutlass.new_from_mlir_values(self.divisor, values[1:])
        return new_obj
