# mypy: allow-untyped-defs
# [MODULE DOC]
# Small utility used by the TorchScript decomposition machinery: registers a
# graph as the decomposition of a specific operator overload with the C++ JIT.
# No import-time side effects.
import torch
from torch._ops import OpOverload, OpOverloadPacket


def _register_decomposition(op: OpOverload, graph: torch._C.Graph) -> None:
    # [FUNCTION DOC] HELPER: Registers graph as the JIT decomposition for the
    # schema of the given operator overload by calling into the C++ registry.
    # Raises AssertionError if op is an overload packet or not an OpOverload.
    # Side effect: mutates the global C++ decomposition registry. Returns None.
    if isinstance(op, OpOverloadPacket):
        raise AssertionError(
            f"Must pass specific op overload, not overload packet, found {op}"
        )
    if not isinstance(op, OpOverload):
        raise AssertionError(f"Expected OpOverload, got {type(op)}")

    torch._C._jit_register_decomposition_for_schema(op._schema, graph)
