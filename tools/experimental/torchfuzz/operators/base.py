# [MODULE DOC] Defines the abstract Operator base class that every torchfuzz operator
# extends: name, selection weight, and the interface for spec production, input spec
# generation and code generation. No import-time side effects.
"""Base operator implementation."""

from abc import ABC, abstractmethod

from torchfuzz.tensor_fuzzer import Spec


class Operator(ABC):
    # [CLASS DOC] Abstract base for operators. Attributes: name (unique registry name) and
    # weight (float used when sampling among compatible operators). Subclasses must implement
    # torch_op_name, can_produce and codegen, and normally fuzz_inputs_specs (leaf operators
    # return an empty list). get_weight may be overridden for context-sensitive weighting.
    """Base class for all operators in torchfuzz."""

    def __init__(self, name: str, weight: float = 1.0):
        # [FUNCTION DOC] Stores the operator name and the weight (coerced to float).
        """Initialize operator with name and optional selection weight.

        Args:
            name: Unique operator name used in the registry
            weight: Relative selection weight when sampling among compatible operators
                    (default 1.0). Higher values increase selection likelihood.
        """
        self.name = name
        self.weight: float = float(weight)

    @property
    @abstractmethod
    def torch_op_name(self) -> str | None:
        # [FUNCTION DOC] Abstract property; subclasses return the torch op name string or None
        # for non-torch operators. The base body raises NotImplementedError.
        """
        Return the torch operation name this operator represents.

        Returns:
            Optional[str]: The torch operation name (e.g., "torch.ops.aten.add", "torch.nonzero").
                          Returns None for non-torch operations like "arg" and "constant".
        """
        raise NotImplementedError("Subclasses must implement torch_op_name")

    @abstractmethod
    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Abstract; subclasses return whether they can yield the given spec.
        # The base body raises NotImplementedError.
        """Check if this operator can produce the given output spec."""
        raise NotImplementedError("Subclasses must implement can_produce()")

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Subclass hook returning the list of input specs that can produce the
        # given output spec. The base body raises NotImplementedError.
        # KNOWN ISSUE: not marked abstract, so subclasses that forget it fail only at call time.
        """
        Get input specifications for fuzzing.

        Subclasses must implement this to return a list of input Specs that,
        when used with this operator, can produce the given output_spec. Leaf
        operators should return an empty list.
        """
        raise NotImplementedError("Subclasses must implement fuzz_inputs_specs()")

    @abstractmethod
    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Abstract; subclasses return source text for the operation assigning
        # output_name from input_names. The base body raises NotImplementedError.
        """Generate code for this operation."""
        raise NotImplementedError("Subclasses must implement codegen()")

    def get_weight(
        self,
        *,
        target_spec: Spec | None = None,
        depth: int | None = None,
        stack_size: int | None = None,
        template: str | None = None,
    ) -> float:
        # [FUNCTION DOC] Returns the selection weight; the default ignores the keyword-only
        # context arguments (target spec, depth, stack size, template) and returns the
        # static weight attribute.
        """
        Return the selection weight for this operator.

        Subclasses may override to implement context-sensitive weighting.
        The default implementation returns the static attribute `self.weight`.
        """
        return self.weight

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the class name followed by the operator name in parentheses.
        """String representation of the operator."""
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the same text as the string conversion.
        """Repr representation of the operator."""
        return self.__str__()
