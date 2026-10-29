# [MODULE DOC] Registry of fuzzer operators. Holds the list of operator module names, a discovery
# helper, the OperatorRegistry class, and module-level functions to get, register, list and
# re-weight operators. Import-time side effect: creates the global registry, which imports every
# listed operator module and instantiates all concrete Operator subclasses found there.
"""Operator registry for mapping operation names to operator instances."""

import importlib
import inspect
from types import ModuleType

from torchfuzz.operators.base import Operator


_OPERATOR_MODULES: list[str] = [
    "activations",
    "arg",
    "argsort",
    "bitwise",
    "comparison",
    "constant",
    "elementwise_math",
    "gather",
    "index_select",
    "item",
    "layout",
    "logical",
    "loss_functions",
    "manipulation_indexing",
    "masked_select",
    "matrix_multiply",
    "nn_functional",
    "nonzero",
    "reduction",
    "scalar_pointwise",
    "special_functions",
    "tensor_creation",
    "tensor_pointwise",
    "unique",
]


def _concrete_operator_classes(module: ModuleType) -> list[type[Operator]]:
    # [FUNCTION DOC] HELPER: Return the list of classes defined in the given module that are
    # concrete Operator subclasses: excludes Operator itself, classes imported from elsewhere,
    # abstract classes and names ending with Base. Order follows alphabetical member order.
    """Discover all concrete Operator subclasses defined in a module.

    Filters out:
    - Non-Operator classes
    - The Operator ABC itself
    - Re-exported classes (whose __module__ doesn't match)
    - Abstract classes
    - Classes with names ending in "Base" (reserved for abstract bases)
    """
    out: list[type[Operator]] = []
    for _name, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, Operator):
            continue
        if cls is Operator:
            continue
        if cls.__module__ != module.__name__:
            continue
        if inspect.isabstract(cls):
            continue
        if cls.__name__.endswith("Base"):
            continue
        out.append(cls)
    return out


class OperatorRegistry:
    # [CLASS DOC] Mapping from operator name to Operator instance. State: the private dictionary of
    # operators, filled with all discovered default operators at construction. Registering a
    # name twice replaces the earlier entry. Key methods: register, get, list_operators.
    """Registry for managing operator instances."""

    def __init__(self):
        # [FUNCTION DOC] Create the empty mapping and populate it with the default operators.
        """Initialize the registry with default operators."""
        self._operators: dict[str, Operator] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        # [FUNCTION DOC] HELPER: For each module name in the module list, import it from the
        # operators package, find its concrete operator classes, instantiate each with no
        # arguments and register it. Import errors propagate.
        """Register the default set of operators via introspection."""
        for module_name in _OPERATOR_MODULES:
            module = importlib.import_module(
                f".{module_name}", package="torchfuzz.operators"
            )
            for cls in _concrete_operator_classes(module):
                self.register(cls())  # pyrefly: ignore[missing-argument]

    def register(self, operator: Operator):
        # [FUNCTION DOC] Store the operator under its name attribute, replacing any existing entry.
        """Register an operator in the registry."""
        self._operators[operator.name] = operator

    def get(self, op_name: str) -> Operator | None:
        # [FUNCTION DOC] Return the operator registered under op_name, or None if absent. Any name
        # starting with the prefix arg_ resolves to the operator named arg.
        """Get an operator by name."""
        # Handle special arg_ operations by mapping them to the ArgOperator
        if op_name.startswith("arg_"):
            return self._operators.get("arg")
        return self._operators.get(op_name)

    def list_operators(self) -> dict[str, Operator]:
        # [FUNCTION DOC] Return a shallow copy of the name-to-operator mapping.
        """List all registered operators."""
        return self._operators.copy()


# Global registry instance
_global_registry = OperatorRegistry()


def get_operator(op_name: str) -> Operator | None:
    # [FUNCTION DOC] Look up op_name in the global registry; returns the operator or None.
    """Get an operator from the global registry."""
    return _global_registry.get(op_name)


def register_operator(operator: Operator):
    # [FUNCTION DOC] Add or replace the operator in the global registry (mutates global state).
    """Register an operator in the global registry."""
    _global_registry.register(operator)


def list_operators() -> dict[str, Operator]:
    # [FUNCTION DOC] Return a copy of the global name-to-operator mapping.
    """List all operators in the global registry."""
    return _global_registry.list_operators()


def set_operator_weight(op_name: str, weight: float) -> None:
    # [FUNCTION DOC] Set the weight attribute of one operator in the global registry, found first
    # by registry name (including the arg_ prefix mapping) and otherwise by matching torch op
    # name. The weight is stored as a float. Raises ValueError if weight is not positive and
    # KeyError if no operator matches. Mutates global operator state.
    """Set the selection weight for a specific operator.

    Args:
        op_name: The registered operator name (e.g., "add", "arg") OR fully-qualified torch op
                 (e.g., "torch.nn.functional.relu", "torch.matmul")
        weight: New relative selection weight (must be > 0)
    """
    if weight <= 0:
        raise ValueError("Operator weight must be > 0")

    # Try by registry key
    op = _global_registry.get(op_name)
    if op is not None:
        op.weight = float(weight)
        return

    # Fallback: try to locate by fully-qualified torch op name
    for candidate in _global_registry.list_operators().values():
        if getattr(candidate, "torch_op_name", None) == op_name:
            candidate.weight = float(weight)
            return

    raise KeyError(f"Operator '{op_name}' not found by registry name or torch op name")


def set_operator_weights(weights: dict[str, float]) -> None:
    # [FUNCTION DOC] Apply the single-operator weight update to every entry of the mapping; the
    # first failure raises and earlier updates stay applied.
    """Bulk-update operator weights from a mapping of name -> weight."""
    for name, w in weights.items():
        set_operator_weight(name, w)


def set_operator_weight_by_torch_op(torch_op_name: str, weight: float) -> None:
    # [FUNCTION DOC] Set the weight of the first operator whose torch op name equals the given
    # name. Raises ValueError if weight is not positive, KeyError if none matches. Mutates the
    # operator.
    """Set operator weight by fully-qualified torch op name."""
    if weight <= 0:
        raise ValueError("Operator weight must be > 0")
    for candidate in _global_registry.list_operators().values():
        if getattr(candidate, "torch_op_name", None) == torch_op_name:
            candidate.weight = float(weight)
            return
    raise KeyError(f"Torch op '{torch_op_name}' not found in registry")


def set_operator_weights_by_torch_op(weights: dict[str, float]) -> None:
    # [FUNCTION DOC] Apply the by-torch-op weight update to each entry of the mapping; the first
    # failure raises and earlier updates stay applied.
    """Bulk-update weights by fully-qualified torch op names."""
    for name, w in weights.items():
        set_operator_weight_by_torch_op(name, w)
