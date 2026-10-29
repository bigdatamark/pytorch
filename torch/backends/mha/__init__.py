# [MODULE DOC] Holds the global switch that enables or disables the C++ fast path used by multi-head attention and the transformer encoder.
# State is a module-level boolean (default enabled) with a getter and a setter. No import-time side effects beyond defining it.
# Config options to enable/disable C++ kernel for nn.functional.MHA
# and nn.TransformerEncoder
import torch


_is_fastpath_enabled: bool = True


def get_fastpath_enabled() -> bool:
    # [FUNCTION DOC] Returns the global fast-path flag; when running under the JIT scripting compiler it always returns True.
    # The fast path may still be skipped at run time if input conditions are not met.
    """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    .. note::
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True


def set_fastpath_enabled(value: bool) -> None:
    # [FUNCTION DOC] Sets the global fast-path flag to the given boolean (mutates module state, returns nothing).
    """Sets whether fast path is enabled"""
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
