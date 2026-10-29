# [MODULE DOC] Deprecated CPU-specific GradScaler wrapper. Defines GradScaler, a thin
# subclass of torch.amp.GradScaler that pins the device to cpu. Exports only GradScaler.
from typing_extensions import deprecated

import torch


__all__ = ["GradScaler"]


class GradScaler(torch.amp.GradScaler):
    # [CLASS DOC] Deprecated alias-style subclass of torch.amp.GradScaler for CPU. Holds no extra
    # state; all scaling behavior is inherited. Constructing it emits a FutureWarning.
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cpu.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cpu", args...)`` instead.
    """

    @deprecated(
        "`torch.cpu.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cpu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        # [FUNCTION DOC] Initializes the scaler with init_scale, growth_factor, backoff_factor,
        # growth_interval and enabled, forwarding them to the base class with device cpu.
        # Decorated so a FutureWarning is emitted on construction.
        super().__init__(
            "cpu",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
