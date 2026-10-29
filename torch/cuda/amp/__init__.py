# [PACKAGE DOC] Deprecated torch.cuda.amp package kept for backward compatibility. Re-exports autocast, custom_fwd,
# custom_bwd (autocast_mode), amp_definitely_not_available (common) and GradScaler (grad_scaler); all
# forward to the device-generic torch.amp equivalents with device type cuda.
# pyrefly: ignore [deprecated]
from .autocast_mode import autocast, custom_bwd, custom_fwd
from .common import amp_definitely_not_available
from .grad_scaler import GradScaler


__all__ = [
    "amp_definitely_not_available",
    "autocast",
    "custom_bwd",
    "custom_fwd",
    "GradScaler",
]
