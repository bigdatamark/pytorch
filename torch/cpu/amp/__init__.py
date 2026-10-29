# [PACKAGE DOC] Deprecated torch.cpu.amp namespace kept for backward compatibility.
# Re-exports autocast (context manager/decorator for CPU mixed precision) from autocast_mode
# and GradScaler (gradient scaler for CPU) from grad_scaler. Both emit FutureWarning and
# forward to the device-generic torch.amp equivalents with the device fixed to cpu.
# pyrefly: ignore [deprecated]
from .autocast_mode import autocast
from .grad_scaler import GradScaler
