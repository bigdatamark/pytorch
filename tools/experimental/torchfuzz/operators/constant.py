# [MODULE DOC] Defines ConstantOperator, the leaf operator that emits literal constants
# (scalars or filled tensors) into generated programs, using seeded fuzzing helpers.
# No import-time side effects.
"""Constant operator implementation."""

from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import (
    fuzz_scalar,
    fuzz_tensor_simple,
    ScalarSpec,
    Spec,
    TensorSpec,
)


class ConstantOperator(Operator):
    # [CLASS DOC] Leaf operator (registry name "constant") producing any spec with no inputs.
    # State: template, an optional fuzz template used to wrap tensor constants for the target
    # device; None until set_template is called. Constant values are deterministic per
    # variable name because the fuzz seed is derived from a hash of the name.
    """Operator for generating constants."""

    def __init__(self):
        # [FUNCTION DOC] Initializes the base operator with the name "constant" and sets the
        # template attribute to None.
        super().__init__("constant")
        # Active FuzzTemplate instance (set via set_template) used to delegate
        # device-specific tensor wrapping in codegen.  None until set_template
        # is called.
        self.template = None

    @property
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Property; always None because constants are not torch operations.
        """Constant is not a torch operation, it generates constant values."""
        return None

    def set_template(self, template):
        # [FUNCTION DOC] Stores the active fuzz template so codegen can delegate device-specific
        # wrapping of tensor constants to it. Mutates operator state; returns nothing.
        """Set the active :class:`FuzzTemplate` instance for codegen.

        ``ops_fuzzer._get_template_filtered_operators`` calls this with the
        instance that ``convert_graph_to_python_code`` will subsequently use,
        so the operator's ``codegen`` can defer device-specific wrapping
        (``DTensor.from_local`` etc.) to ``template.codegen_constant``.
        """
        self.template = template

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Always returns true; a constant can be any scalar or tensor spec.
        """Constant can produce any type of output."""
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Returns an empty list: constants take no inputs.
        """Constant requires no inputs for fuzzing."""
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Returns source text defining output_name as a constant. The seed is an
        # MD5-derived number of the name (deterministic across processes). Scalar specs: one
        # fuzzed scalar value embedded as a literal (complex values via a complex constructor).
        # Tensor specs: a fuzzed tensor is built only to pick a fill value (its first element;
        # integer values clamped to the range 0..3; empty tensors use 1 or True), and the
        # emitted line is a full-tensor creation of the spec size and dtype, passed through the
        # template's constant wrapper when a template is set. Other spec types yield a comment
        # line. Input names are unused.
        # KNOWN ISSUE: the tensor stride is ignored (always contiguous), and only the first
        # element is used so the constant is uniform; unsigned ints are not clamped.
        """Generate code for constant creation."""
        # Create constant by calling fuzzing functions during codegen with deterministic seed
        # Use a deterministic hash based on the variable name to ensure reproducibility across processes
        import hashlib

        var_seed = int(hashlib.md5(output_name.encode()).hexdigest()[:8], 16) % (2**31)  # noqa: S324

        if isinstance(output_spec, ScalarSpec):
            # Call fuzz_scalar during codegen and embed the result
            actual_value = fuzz_scalar(output_spec, seed=var_seed)

            # Format the value for embedding in code
            if isinstance(actual_value, bool):
                value_str = str(actual_value)
            elif isinstance(actual_value, (int, float)):
                value_str = repr(actual_value)
            elif isinstance(actual_value, complex):
                value_str = f"complex({actual_value.real}, {actual_value.imag})"
            else:
                value_str = repr(actual_value)

            return f"{output_name} = {value_str}"

        elif isinstance(output_spec, TensorSpec):
            # Call fuzz_tensor_simple during codegen and embed the result
            actual_tensor = fuzz_tensor_simple(
                output_spec.size, output_spec.stride, output_spec.dtype, seed=var_seed
            )

            # Convert tensor to code representation
            size_str = str(output_spec.size)
            dtype_str = f"torch.{output_spec.dtype}".replace("torch.torch.", "torch.")

            # Handle empty tensors (with 0 elements)
            if actual_tensor.numel() == 0:
                # For empty tensors, use a default fill value based on dtype
                import torch

                default_values = {
                    torch.float16: 1.0,
                    torch.float32: 1.0,
                    torch.float64: 1.0,
                    torch.bfloat16: 1.0,
                    torch.int8: 1,
                    torch.int16: 1,
                    torch.int32: 1,
                    torch.int64: 1,
                    torch.bool: True,
                    torch.complex64: 1.0,
                    torch.complex128: 1.0,
                }

                fill_value = default_values.get(output_spec.dtype, 1)
                tensor_creation = (
                    f"torch.full({size_str}, {fill_value}, dtype={dtype_str})"
                )
            else:
                # For non-empty tensors, use the first element as fill value
                fill_value = actual_tensor.flatten()[0].item()

                # For integer types, clamp the value to a smaller range to avoid
                # issues when used in arithmetic with embedding indices
                import torch

                if output_spec.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    # Clamp integer values to [0, 3] to avoid index overflow in multiplication
                    # Even with multiplication, indices should stay in reasonable range
                    fill_value = max(0, min(3, abs(fill_value)))

                tensor_creation = (
                    f"torch.full({size_str}, {fill_value}, dtype={dtype_str})"
                )

            # Delegate device-specific wrapping (cuda placement, DTensor, etc.) to the
            # active template; falls back to a plain assignment when no template is set.
            if self.template is not None:
                return self.template.codegen_constant(output_name, tensor_creation)
            return f"{output_name} = {tensor_creation}"

        else:
            return f"# Unknown output spec type for constant: {type(output_spec)}"
