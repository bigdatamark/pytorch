# [MODULE DOC] Fuzzer operators for Python-level scalar arithmetic. Defines a base class that emits
# a binary infix expression plus add, mul, sub and div subclasses (div guards the denominator).
# No import-time side effects; subclasses are registered by registry introspection.
"""Scalar pointwise operator implementation."""

import random

import torch

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec


class ScalarPointwiseOperatorBase(Operator):
    # [CLASS DOC] Base for binary scalar operators. Attribute symbol is the infix operator text
    # used in generated code; the registry name is passed to the base class. Its name ends in
    # Base so the registry skips it. Produces only non-bool scalar specs from two scalar inputs.
    """Base class for scalar pointwise operations."""

    def __init__(self, name: str, symbol: str):
        # [FUNCTION DOC] Store the registry name via the base class and the infix symbol text.
        super().__init__(name)
        self.symbol = symbol

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Property returning None: no torch op corresponds to these operators.
        """Scalar operations don't have specific torch ops, they use Python operators."""
        return None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for scalar specs whose dtype is not bool. Pure check.
        """Scalar pointwise operations can only produce scalars."""
        if output_spec.dtype == torch.bool:
            return False
        return isinstance(output_spec, ScalarSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] Pick, at random, a pair of dtypes from the type-promotion helper's list of
        # pairs that promote to the output dtype, and return two scalar specs of those dtypes.
        # num_inputs is ignored. Raises ValueError if output_spec is not a scalar spec. Consumes
        # global random state.
        """Decompose scalar into input scalars for pointwise operation with type promotion."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce ScalarSpec outputs"
            )

        # Use shared type promotion utility
        from torchfuzz.type_promotion import get_scalar_promotion_pairs

        supported_types = get_scalar_promotion_pairs(output_spec.dtype)
        dtypes = random.choice(supported_types)

        return [ScalarSpec(dtype=dtypes[0]), ScalarSpec(dtype=dtypes[1])]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return one line of source assigning output_name the first input, the
        # infix symbol, then the second input. Raises ValueError unless exactly two input names.
        """Generate code for scalar pointwise operation."""
        if len(input_names) != 2:
            raise ValueError(f"{self.__class__.__name__} requires exactly two inputs")

        return f"{output_name} = {input_names[0]} {self.symbol} {input_names[1]}"


class ScalarAddOperator(ScalarPointwiseOperatorBase):
    # [CLASS DOC] Scalar addition operator, registry name scalar_add, symbol plus.
    """Operator for scalar addition."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name scalar_add with the plus symbol.
        super().__init__("scalar_add", "+")


class ScalarMulOperator(ScalarPointwiseOperatorBase):
    # [CLASS DOC] Scalar multiplication operator, registry name scalar_mul, symbol star.
    """Operator for scalar multiplication."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name scalar_mul with the multiplication symbol.
        super().__init__("scalar_mul", "*")


class ScalarSubOperator(ScalarPointwiseOperatorBase):
    # [CLASS DOC] Scalar subtraction operator, registry name scalar_sub, symbol minus.
    """Operator for scalar subtraction."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name scalar_sub with the minus symbol.
        super().__init__("scalar_sub", "-")


class ScalarDivOperator(ScalarPointwiseOperatorBase):
    # [CLASS DOC] Scalar true-division operator, registry name scalar_div, symbol slash. Overrides
    # codegen to clamp the denominator so generated code cannot divide by zero.
    """Operator for scalar division."""

    def __init__(self):
        # [FUNCTION DOC] Register under the name scalar_div with the division symbol.
        super().__init__("scalar_div", "/")

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return one line of source dividing the first input by the larger of the
        # second input and a floor: 1 when the output dtype is an int8/16/32/64 scalar, else 1e-6.
        # Raises ValueError unless exactly two input names are given.
        # KNOWN ISSUE: the max clamp only guards zero; negative denominators are replaced by the
        # floor, which changes the division semantics for them. Also true division of ints yields
        # a float even when the output dtype is integer.
        """Generate code for scalar division with zero-denominator guard."""
        if len(input_names) != 2:
            raise ValueError(f"{self.__class__.__name__} requires exactly two inputs")

        # Prevent ZeroDivisionError at runtime by clamping the denominator.
        # Clamp denominator to at least 1 (for ints) or 1e-6 (for floats).
        if isinstance(output_spec, ScalarSpec) and output_spec.dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            return f"{output_name} = {input_names[0]} / max({input_names[1]}, 1)"
        else:
            return f"{output_name} = {input_names[0]} / max({input_names[1]}, 1e-6)"
