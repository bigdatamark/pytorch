# [PACKAGE DOC] torch.amp: automatic mixed precision. Re-exports autocast, custom_fwd, custom_bwd, is_autocast_available (from autocast_mode) and GradScaler (from grad_scaler); also imports the private _enter_autocast/_exit_autocast helpers for internal use.
from .autocast_mode import (
    _enter_autocast,
    _exit_autocast,
    autocast,
    custom_bwd,
    custom_fwd,
    is_autocast_available,
)
from .grad_scaler import GradScaler


__all__ = [
    "autocast",
    "custom_bwd",
    "custom_fwd",
    "is_autocast_available",
    "GradScaler",
]
