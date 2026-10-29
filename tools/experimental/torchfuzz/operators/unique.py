# [MODULE DOC] Fuzzer operator for torch.unique. Defines UniqueOperator, which produces 1D outputs
# of a requested length by generating code that uniquifies a range of distinct integers. No
# import-time side effects; discovered by the operator registry via introspection.
"""Unique operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class UniqueOperator(Operator):
    # [CLASS DOC] Operator producing 1D tensors of unique values of any dtype. Stateless apart from
    # the registry name "unique". Key methods: can_produce, fuzz_inputs_specs, codegen.
    """Operator for finding unique elements in a tensor."""

    def __init__(self):
        # [FUNCTION DOC] Initialize the operator with the registry name "unique".
        super().__init__("unique")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Property returning the string "torch.unique".
        """Return the torch operation name."""
        return "torch.unique"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True if output_spec is a tensor spec with exactly one dimension,
        # regardless of dtype or length. Pure check.
        """Unique can produce 1D tensor outputs of arbitrary length without guards.

        We will synthesize an input with exactly the desired number of unique
        elements so that torch.unique returns the target size deterministically.
        """
        return isinstance(output_spec, TensorSpec) and len(output_spec.size) == 1

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 1) -> list[Spec]:
        # [FUNCTION DOC] Return a single contiguous input spec of fixed size (2, 3) with the same
        # dtype as the output. num_inputs is ignored. Raises ValueError if output_spec is not a
        # tensor spec.
        """Generate input spec for unique operation."""
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("UniqueOperator can only produce TensorSpec outputs")

        # Input can be any tensor - unique will flatten and find unique values
        input_spec = TensorSpec(
            size=(2, 3),  # Fixed size for consistency
            stride=(3, 1),  # Contiguous
            dtype=output_spec.dtype,  # Match output dtype
        )

        return [input_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source text that builds an int64 range of the desired output length
        # (zero if the spec is not a tensor spec), applies unique, and casts the result to the
        # dtype of the input variable. Raises ValueError unless exactly one input name is given.
        # KNOWN ISSUE: the cast uses the input's dtype rather than the output spec dtype; they
        # match only because the input spec copies the output dtype.
        """Generate code for unique with deterministic target size input (no guards)."""
        if len(input_names) != 1:
            raise ValueError("UniqueOperator requires exactly one input")
        # Desired output length and target dtype
        desired_len = output_spec.size[0] if isinstance(output_spec, TensorSpec) else 0
        # Synthesize in a wide dtype (int64) to guarantee desired_len distinct values,
        # apply unique, then cast to the target dtype. No conditionals or guards.
        return (
            f"_inp_unique_wide = torch.arange({desired_len}, device={input_names[0]}.device, dtype=torch.int64)\n"
            f"_uniq_wide = torch.unique(_inp_unique_wide)\n"
            f"{output_name} = _uniq_wide.to({input_names[0]}.dtype)"
        )
