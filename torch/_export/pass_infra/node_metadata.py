# [MODULE DOC]
# Defines NodeMetadata, a small wrapper over an FX node meta dict used by export
# passes, and PROTECTED_KEYS, the keys that may not be reassigned through it.
# No import-time side effects.
from typing import Any


NodeMetadataValue = Any


PROTECTED_KEYS: set[str] = {
    "val",
    "stack_trace",
    "nn_module_stack",
    "debug_handle",
    "tensor_meta",
}


class NodeMetadata:
    # [CLASS DOC] Thin copy-on-construct wrapper around a dict of FX node metadata.
    # State: data (a private copy of the given mapping). Keys listed in PROTECTED_KEYS
    # (val, stack_trace, nn_module_stack, debug_handle, tensor_meta) cannot be
    # overwritten after construction.
    def __init__(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC] Store a shallow copy of the given metadata mapping in data.
        self.data: dict[str, Any] = data.copy()

    def __getitem__(self, key: str) -> NodeMetadataValue:
        # [FUNCTION DOC] Return the value for key; raises KeyError if missing.
        return self.data[key]

    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        # [FUNCTION DOC] Set a metadata entry. Raises RuntimeError if key is one of the
        # protected keys (even when not yet present). Mutates data in place.
        # KNOWN ISSUE: return annotation claims a value but nothing is returned.
        if key in PROTECTED_KEYS:
            raise RuntimeError(f"Could not override node key: {key}")
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        # [FUNCTION DOC] Return whether key is present in the metadata.
        return key in self.data

    def copy(self) -> "NodeMetadata":
        # [FUNCTION DOC] Return a new NodeMetadata holding a shallow copy of the data.
        return NodeMetadata(self.data.copy())
