# [MODULE DOC] Fuzzer operator for torch.masked_select producing a 1-D tensor. Defines
# MaskedSelectOperator; no import-time side effects.
"""Masked select operator implementation."""

import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class MaskedSelectOperator(Operator):
    # [CLASS DOC] Operator producing a 1-D tensor via masked_select. Stateless; the generated
    # code builds its own source tensor and mask so exactly the requested number of elements
    # is selected.
    """Operator for selecting elements from a tensor based on a mask."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register under the name masked_select.
        super().__init__("masked_select")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns the fully qualified torch function name for this operator.
        """Return the torch operation name."""
        return "torch.masked_select"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True only for tensor specs with exactly one dimension.
        """Masked select produces a 1D tensor; we'll synthesize inputs to match size."""
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) == 1

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] Returns fixed input specs: a contiguous 2 by 3 tensor of the output
        # dtype and a contiguous 2 by 3 bool mask. Raises ValueError if output_spec is not a
        # TensorSpec. num_inputs is unused.
        # KNOWN ISSUE: these inputs are not used for their values; codegen only reads the first
        # input's device and dtype.
        """Generate input specs for masked_select operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("MaskedSelectOperator can only produce TensorSpec outputs")

        # Input tensor - can be any shape and type
        input_tensor_spec = TensorSpec(
            size=(2, 3),  # Fixed size for consistency
            stride=(3, 1),  # Contiguous
            dtype=output_spec.dtype,  # Match output dtype
        )

        # Mask tensor - must be boolean and broadcastable to input
        mask_spec = TensorSpec(
            size=(2, 3),  # Same size as input for simplicity
            stride=(3, 1),  # Contiguous
            dtype=torch.bool,
        )

        return [input_tensor_spec, mask_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits code that builds an arange source tensor of length max(k, 1) on
        # the first input's device and dtype, a mask whose first k entries are true, and
        # selects with them, so exactly k elements come out (k is the requested output
        # length). Raises ValueError if there are not exactly two input names or the output is
        # not a 1-D TensorSpec.
        # KNOWN ISSUE: the generated code uses fixed temp names, so repeated use in one program
        # would overwrite earlier temporaries (harmless unless values are reused).
        """Generate code for masked_select with synthesized inputs to match size.

        Constructs an input tensor and mask so that exactly k elements are selected,
        where k = output_spec.size[0]. No data-dependent guards.
        """
        if len(input_names) != 2:
            raise ValueError("MaskedSelectOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec) or len(output_spec.size) != 1:
            raise ValueError("MaskedSelectOperator requires 1D TensorSpec output")
        k = output_spec.size[0]
        # Build a 1D input of length >= k and a mask with first k positions True
        # Use input's device and output dtype to avoid mismatches
        return (
            f"_x_ms = torch.arange(max({k}, 1), device={input_names[0]}.device).to({input_names[0]}.dtype)\n"
            f"_mask_ms = torch.zeros_like(_x_ms, dtype=torch.bool)\n"
            f"_mask_ms[:{k}] = True\n"
            f"{output_name} = torch.masked_select(_x_ms, _mask_ms)"
        )
