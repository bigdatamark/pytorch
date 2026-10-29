# pyre-strict
# [MODULE DOC] Bitwise operators for torchfuzz: base classes for binary bitwise ops and
# shift ops, concrete and/or/xor/left-shift/right-shift subclasses, and a unary bitwise-not
# operator. Module constants hold the allowed integer dtype tuples, bit widths, and an opt-in
# flag (default off) that clamps shift counts at codegen time. No import-time side effects.
"""Bitwise ops: and, or, xor, not, left_shift, right_shift.

and/or/xor/not accept integer + bool dtypes (output dtype = input dtype).
left_shift / right_shift accept integer dtypes only (no bool).

Shift-count values are NOT constrained by default -- out-of-range shifts
at runtime are tolerated as 'fuzzer-fuzzy' behavior. An opt-in clamp is
available via _CLAMP_SHIFT_COUNTS to filter UB-flavored noise during
triage.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import contiguous_stride, random_broadcast_shape
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


_BITWISE_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.bool,
)

_SHIFT_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)

# Opt-in noise filter: when True, shift codegen wraps the shift-count
# input in torch.clamp(0, bit_width-1) so out-of-range UB does not flood
# triage. Default False preserves the fuzzer-fuzzy behavior intended by
# the design discussion.
_CLAMP_SHIFT_COUNTS: bool = False


_BIT_WIDTHS: dict[torch.dtype, int] = {
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.uint8: 8,
    torch.uint16: 16,
    torch.uint32: 32,
    torch.uint64: 64,
}


# ---------------------------------------------------------------------------
# Base classes (excluded from introspection by the ``Base`` suffix)
# ---------------------------------------------------------------------------


class BitwiseBinaryOperatorBase(Operator):
    # [CLASS DOC] Base for two-operand bitwise ops (and/or/xor). The operator name passed by
    # the subclass doubles as the torch function path. Output must be a tensor of an integer
    # or bool dtype; both inputs share that dtype, possibly with broadcast or scalar operands.
    """Two-tensor bitwise. Inputs match output dtype/shape/stride.

    Allowed dtypes: int8/16/32/64, uint8/16/32/64, bool.
    """

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec whose dtype is an integer or bool dtype.
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _BITWISE_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns two input specs matching the output size, stride and dtype.
        # With probability 0.3 one random operand is replaced by a contiguous broadcastable
        # smaller shape (if a smaller shape was found); with the next 0.2 one random operand
        # becomes a scalar spec of the same dtype. Uses the global random module.
        # Raises AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        r = random.random()
        if r < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        elif r < 0.5:
            # scalar operand
            idx = random.randint(0, 1)
            specs[idx] = ScalarSpec(dtype=output_spec.dtype)
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line assigning the torch function applied to the first
        # two input names to output_name. output_spec is unused.
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        )


class BitwiseShiftOperatorBase(Operator):
    # [CLASS DOC] Base for bitwise shift ops. Output must be a tensor of a non-bool integer
    # dtype; inputs (value, shift count) share that dtype. Shift counts are unconstrained
    # unless the module clamp flag is on, in which case codegen clamps them to the bit width.
    """Two-tensor bitwise shift. Inputs match output dtype/shape/stride.

    Allowed dtypes: int8/16/32/64, uint8/16/32/64 (no bool).

    The second input (shift count) is a same-dtype, same-shape tensor.
    By default values are NOT constrained -- out-of-range shifts at
    runtime are accepted as fuzzer-fuzzy behavior. When
    _CLAMP_SHIFT_COUNTS is True (a one-line module toggle), the shift
    count is wrapped in torch.clamp(0, bit_width - 1) at codegen time
    to filter UB-flavored noise during triage.
    """

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec whose dtype is a signed or unsigned
        # integer dtype (bool excluded).
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _SHIFT_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Same strategy as the binary bitwise base: two inputs matching the
        # output; with probability 0.3 one random operand gets a smaller contiguous broadcast
        # shape, with the next 0.2 one random operand becomes a scalar spec. Uses the global
        # random module. Raises AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        r = random.random()
        if r < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        elif r < 0.5:
            # scalar shift count
            idx = random.randint(0, 1)
            specs[idx] = ScalarSpec(dtype=output_spec.dtype)
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line assigning the torch shift function of the first
        # input and the second input (the shift count) to output_name. If the module clamp
        # flag is set, the count is wrapped in a clamp from zero to bit width minus one.
        # Raises AssertionError for a non-tensor spec; a dtype missing from the bit-width table
        # would raise a lookup error when clamping.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        if _CLAMP_SHIFT_COUNTS:
            bw = _BIT_WIDTHS[output_spec.dtype]
            shift_expr = f"torch.clamp({input_names[1]}, 0, {bw - 1})"
        else:
            shift_expr = input_names[1]
        return f"{output_name} = {self.torch_op_name}({input_names[0]}, {shift_expr})"


# ---------------------------------------------------------------------------
# Bitwise binary subclasses
# ---------------------------------------------------------------------------


class BitwiseAndOperator(BitwiseBinaryOperatorBase):
    # [CLASS DOC] Binary bitwise AND operator (torch.bitwise_and).
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_and.
        super().__init__("torch.bitwise_and")


class BitwiseOrOperator(BitwiseBinaryOperatorBase):
    # [CLASS DOC] Binary bitwise OR operator (torch.bitwise_or).
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_or.
        super().__init__("torch.bitwise_or")


class BitwiseXorOperator(BitwiseBinaryOperatorBase):
    # [CLASS DOC] Binary bitwise XOR operator (torch.bitwise_xor).
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_xor.
        super().__init__("torch.bitwise_xor")


# ---------------------------------------------------------------------------
# Bitwise shift subclasses
# ---------------------------------------------------------------------------


class BitwiseLeftShiftOperator(BitwiseShiftOperatorBase):
    # [CLASS DOC] Bitwise left shift operator (torch.bitwise_left_shift).
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_left_shift.
        super().__init__("torch.bitwise_left_shift")


class BitwiseRightShiftOperator(BitwiseShiftOperatorBase):
    # [CLASS DOC] Bitwise right shift operator (torch.bitwise_right_shift).
    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_right_shift.
        super().__init__("torch.bitwise_right_shift")


# ---------------------------------------------------------------------------
# Bitwise unary
# ---------------------------------------------------------------------------


class BitwiseNotOperator(Operator):
    # [CLASS DOC] Unary bitwise NOT operator (torch.bitwise_not) over integer or bool tensors;
    # the single input has the same size, stride and dtype as the output.
    """Single-tensor bitwise NOT. Same dtype set as bitwise binary."""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property; returns the operator name, which is the torch function path.
        return self.name

    def __init__(self) -> None:
        # [FUNCTION DOC] Registers the operator under the name torch.bitwise_not.
        super().__init__("torch.bitwise_not")

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec whose dtype is an integer or bool dtype.
        return (
            isinstance(output_spec, TensorSpec) and output_spec.dtype in _BITWISE_DTYPES
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns a one-element list with a tensor spec copying the output
        # size, stride and dtype. Raises AssertionError if output_spec is not a TensorSpec.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Returns one line assigning torch.bitwise_not of the first input to
        # output_name. output_spec is unused.
        return f"{output_name} = torch.bitwise_not({input_names[0]})"
