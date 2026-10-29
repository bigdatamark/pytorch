# [MODULE DOC] Defines ArgOperator, the leaf operator representing a function argument in a
# generated program. Imports Operator and Spec only; no import-time side effects.
"""Arg operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec


class ArgOperator(Operator):
    # [CLASS DOC] Leaf operator (registry name "arg") standing for a program input. It has no
    # torch op, accepts any output spec and needs no inputs; the real argument binding is done
    # elsewhere by the program code generator. Stateless beyond the base name and weight.
    """Operator for function arguments/parameters."""

    def __init__(self):
        # [FUNCTION DOC] Initializes the base operator with the name "arg" and default weight.
        super().__init__("arg")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Property; always returns None because arg is not a torch operation.
        """Arg is not a torch operation, it represents function arguments."""
        return None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Always returns true: any spec (tensor or scalar) can be an argument.
        """Arg can produce any type of output."""
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an empty list: a leaf operator has no input specs.
        """Arg requires no inputs for fuzzing."""
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns only a placeholder comment line naming output_name; the real
        # argument assignment is handled by the program code generator. Inputs are ignored.
        """Generate code for arg operation."""
        # The actual argument name assignment will be handled separately
        # in the codegen.py when processing arg operations
        return f"# {output_name} will be assigned an argument value"
