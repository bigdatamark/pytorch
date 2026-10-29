# pyre-strict
# [MODULE DOC] Fuzzer operators for torch logical ops: binary logical_and/or/xor and unary
# logical_not. Outputs are always bool tensors; inputs are random numeric or bool dtypes.
# Defines a module-level tuple of allowed input dtypes; no other import-time side effects.
"""Logical ops: and, or, xor (binary) + not (unary).

Inputs may be any numeric dtype OR bool -- semantics treat non-zero as
True. Output is always bool.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import contiguous_stride, random_broadcast_shape
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_LOGICAL_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
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


# ---------------------------------------------------------------------------
# Base class for binary logical ops
# ---------------------------------------------------------------------------


class LogicalBinaryOperatorBase(Operator):
    # [CLASS DOC] Base for two-input logical ops producing bool tensors. Holds no extra state
    # beyond the operator name (the full torch function name). Both inputs share one randomly
    # chosen dtype.
    """Two-tensor logical -> bool. Inputs share a randomly-chosen dtype."""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the operator name (the full torch function name) used as the
        # callable in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True only for tensor specs whose dtype is bool.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns two input specs sharing a random dtype from the allowed list
        # and the output's size and stride. With probability 0.3, one randomly chosen input is
        # replaced by a contiguous spec of a random broadcastable shape (only if that shape
        # differs from the output size). Raises AssertionError if output_spec is not a
        # TensorSpec. Uses the random module.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        dtype = random.choice(_LOGICAL_INPUT_DTYPES)
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=dtype,
        )
        specs: list[Spec] = [spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=dtype,
                )
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line calling the logical function on the two inputs and
        # assigning to output_name.
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, {input_names[1]})"
        )


# ---------------------------------------------------------------------------
# Logical binary subclasses
# ---------------------------------------------------------------------------


class LogicalAndOperator(LogicalBinaryOperatorBase):
    # [CLASS DOC] Operator for elementwise logical AND (torch.logical_and).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.logical_and.
        super().__init__("torch.logical_and")


class LogicalOrOperator(LogicalBinaryOperatorBase):
    # [CLASS DOC] Operator for elementwise logical OR (torch.logical_or).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.logical_or.
        super().__init__("torch.logical_or")


class LogicalXorOperator(LogicalBinaryOperatorBase):
    # [CLASS DOC] Operator for elementwise logical XOR (torch.logical_xor).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.logical_xor.
        super().__init__("torch.logical_xor")


# ---------------------------------------------------------------------------
# Logical unary (only one op, so a base class is not justified)
# ---------------------------------------------------------------------------


class LogicalNotOperator(Operator):
    # [CLASS DOC] Operator for logical NOT (torch.logical_not): one input of random dtype,
    # bool output.
    """Single-tensor logical -> bool."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Register under the name torch.logical_not.
        super().__init__("torch.logical_not")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Returns the operator name (the full torch function name) used as the
        # callable in generated code.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True only for tensor specs whose dtype is bool.
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns one input spec with the output's size and stride and a random
        # allowed dtype. Raises AssertionError if output_spec is not a TensorSpec. Uses the
        # random module.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        dtype = random.choice(_LOGICAL_INPUT_DTYPES)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Emits a line applying logical_not to the single input and assigning
        # to output_name.
        return f"{output_name} = torch.logical_not({input_names[0]})"
