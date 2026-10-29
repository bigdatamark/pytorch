# [MODULE DOC]
# Deprecated backward-compatibility aliases _StreamBase and _EventBase that
# subclass torch.Stream and torch.Event and emit a FutureWarning when used.
from typing_extensions import deprecated

import torch


# Preserved only for BC reasons
@deprecated(
    "`torch._streambase._StreamBase` is deprecated. Please use `torch.Stream` instead.",
    category=FutureWarning,
)
class _StreamBase(torch.Stream):
    # [CLASS DOC] HELPER: Deprecated empty subclass of torch.Stream kept for BC.
    pass


@deprecated(
    "`torch._streambase._EventBase` is deprecated. Please use `torch.Event` instead.",
    category=FutureWarning,
)
class _EventBase(torch.Event):
    # [CLASS DOC] HELPER: Deprecated empty subclass of torch.Event kept for BC.
    pass
