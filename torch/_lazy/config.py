# [MODULE DOC] Python accessors for lazy tensor core configuration (force-fallback setting, IR node reuse).
import torch._C._lazy


def get_force_fallback() -> str:
    # [FUNCTION DOC] Returns the current force-fallback config string from the C++ lazy runtime.
    """Get the config used to force LTC fallback"""
    return torch._C._lazy._get_force_fallback()


def set_force_fallback(configval: str) -> None:
    # [FUNCTION DOC] Sets the force-fallback config string in the C++ lazy runtime. Global side effect.
    """Set the config used to force LTC fallback"""
    torch._C._lazy._set_force_fallback(configval)


def set_reuse_ir(val: bool) -> None:
    # [FUNCTION DOC] Enables or disables reuse of IR nodes during tracing. Global side effect.
    """Set the config to reuse IR nodes for faster tracing"""
    torch._C._lazy._set_reuse_ir(val)
