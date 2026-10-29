# [MODULE DOC] Defines GatherOperator, which emits torch.gather along dimension 0 with an
# index tensor remapped into range. No import-time side effects.
import torch
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


class GatherOperator(Operator):
    # [CLASS DOC] Operator (registry name "gather") for torch.gather. Output is a tensor with
    # at least one dim; inputs are a source tensor (dim 0 enlarged by 2) and a long index tensor
    # with the output's shape. Always gathers along dimension 0. Holds no extra state.
    """Operator for gathering values along an axis specified by dim using indices."""

    def __init__(self):
        # [FUNCTION DOC] Initializes the base operator with the name "gather".
        super().__init__("gather")

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Property; returns the string "torch.gather".
        """Return the torch operation name."""
        return "torch.gather"

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] True only for a TensorSpec with at least one dimension.
        """Gather can produce tensors of various shapes, but not 0-dimensional tensors."""
        if not isinstance(output_spec, TensorSpec):
            return False
        # gather requires at least one dimension
        return len(output_spec.size) > 0

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        # [FUNCTION DOC] Returns [source spec, index spec]. The source has the output's dtype
        # and size with dimension 0 increased by 2, contiguous strides. The index is a long
        # tensor with exactly the output's size and contiguous strides. Raises ValueError if
        # output_spec is not a TensorSpec.
        # KNOWN ISSUE: the num_inputs parameter is unused.
        """Generate input specs for gather operation.

        torch.gather(input, dim, index) returns a tensor with:
        - output.shape == index.shape
        - output[i][j][k] = input[i][j][index[i][j][k]] (for dim=2 example)
        """
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator can only produce TensorSpec outputs")

        # The output shape matches the index shape
        output_size = output_spec.size
        dim = 0  # Gather along dimension 0 for simplicity

        # Input tensor - create a shape that matches output except for the gather dimension
        # which can be any size >= max(indices) + 1
        # For simplicity, make input larger in the gather dimension
        if len(output_size) == 1:
            # Output is 1D
            input_size = (output_size[0] + 2,)
            input_stride = (1,)
        elif len(output_size) == 2:
            # Output is 2D, make input 2D with first dim larger
            input_size = (output_size[0] + 2, output_size[1])
            input_stride = (output_size[1], 1)  # Contiguous
        else:
            # For higher dimensions
            input_size = tuple(
                s + 2 if i == dim else s for i, s in enumerate(output_size)
            )
            # Contiguous stride
            input_stride = tuple(
                int(torch.tensor(input_size[i + 1 :]).prod().item())
                if i < len(input_size) - 1
                else 1
                for i in range(len(input_size))
            )

        input_tensor_spec = TensorSpec(
            size=input_size,
            stride=input_stride,
            dtype=output_spec.dtype,
        )

        # Index tensor - same shape as output, long dtype
        index_spec = TensorSpec(
            size=output_size,
            stride=tuple(
                int(torch.tensor(output_size[i + 1 :]).prod().item())
                if i < len(output_size) - 1
                else 1
                for i in range(len(output_size))
            ),
            dtype=torch.long,
        )

        return [input_tensor_spec, index_spec]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns three generated lines: read the source size along dim 0, map
        # the index input into range with a remainder by that size (so every index is valid and
        # identical across runs), then gather along dim 0 into output_name.
        # Raises ValueError unless there are exactly two inputs and a TensorSpec output.
        """Generate code for gather.

        Creates appropriate indices to gather from the input tensor.
        """
        if len(input_names) != 2:
            raise ValueError("GatherOperator requires exactly two inputs")
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("GatherOperator requires TensorSpec output")

        # Determine dimension
        dim = 0  # Gather along dimension 0 for simplicity

        # Map the pre-generated index input (a fixed graph arg, generated once and
        # shared across the reference and test runs) into the valid range with
        # remainder. Do NOT synthesize a fresh in-program torch.randint: an inline
        # randint draws from whatever device RNG is active, so a cross-device or
        # eager-vs-compile comparison would use different indices and diverge even
        # when the gather kernel is correct. remainder keeps every index in
        # [0, input_size).
        return (
            f"_input_size_{output_name} = {input_names[0]}.size({dim})\n"
            f"_index_{output_name} = torch.remainder({input_names[1]}, _input_size_{output_name})\n"
            f"{output_name} = torch.gather({input_names[0]}, {dim}, _index_{output_name})"
        )
