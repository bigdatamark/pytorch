# mypy: allow-untyped-defs
# [PACKAGE DOC] Package for DebugMode, a TorchDispatchMode that records operator calls, module
# annotations, DTensor redistributions and Triton kernel launches into a log. Re-exports the call
# record types, DebugMode, DebugInterpreter, get_active_debug_mode and the tensor hashing helpers.
# Public API (__all__) is only DebugMode and get_active_debug_mode. Importing has no side effects.
"""
DebugMode: a debugging TorchDispatchMode that intercepts and logs runtime calls.

See torch.utils._debug_mode._mode for the full implementation and docstring.
"""

from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _DebugCall,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _TritonKernelCall,
)
from torch.utils._debug_mode._mode import (
    DebugInterpreter,
    DebugMode,
    get_active_debug_mode,
)
from torch.utils._debug_mode._utils import (
    _stringify_shape,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)


__all__ = ["DebugMode", "get_active_debug_mode"]
