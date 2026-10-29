# [MODULE DOC] Fuzzer operator that converts a 0-d tensor into a Python scalar via the item
# method. Defines ItemOperator; no import-time side effects.
"""Item operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import ScalarSpec, Spec, TensorSpec


class ItemOperator(Operator):
    # [CLASS DOC] Operator producing scalar values (ScalarSpec) from a single 0-d tensor input.
    # Stateless; implements can_produce, fuzz_inputs_specs and codegen.
    """Operator for converting 0-d tensor to scalar."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Register the operator under the name item.
        super().__init__("item")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Returns None because item is a tensor method, not a torch function.
        """Item is a tensor method, not a direct torch operation."""
        return None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Returns True only when the requested output spec is a ScalarSpec.
        """Item produces scalars from 0-d tensors."""
        return isinstance(output_spec, ScalarSpec)

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        # [FUNCTION DOC] Returns a single input spec: a 0-d tensor (empty size and stride) with the
        # scalar's dtype. Raises ValueError if output_spec is not a ScalarSpec. num_inputs ignored.
        """Decompose scalar into a single-element tensor for item operation."""
        if not isinstance(output_spec, ScalarSpec):
            raise ValueError("ItemOperator can only produce ScalarSpec outputs")

        # Create a tensor spec that can produce a scalar via .item()
        # Use a 0-D tensor (scalar tensor) to ensure .item() works reliably
        tensor_spec = TensorSpec(size=(), stride=(), dtype=output_spec.dtype)

        return [tensor_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Emits a line assigning the item result of the single input to output_name.
        # Raises ValueError if the number of input names is not exactly one.
        """Generate code for item operation."""
        if len(input_names) != 1:
            raise ValueError("ItemOperator requires exactly one input")

        return f"{output_name} = {input_names[0]}.item()"
