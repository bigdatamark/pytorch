# [MODULE DOC] Entry point for the lazy TorchScript backend.
import torch._C._lazy_ts_backend


def init() -> None:
    # [FUNCTION DOC] Initializes the lazy TorchScript backend in the C++ runtime (global side effect).
    """Initializes the lazy Torchscript backend"""
    torch._C._lazy_ts_backend._init()
