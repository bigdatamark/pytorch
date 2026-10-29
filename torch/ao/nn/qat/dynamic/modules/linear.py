# [MODULE DOC] Defines the dynamic QAT Linear module. No import-time side effects.
from typing import Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfig


__all__ = ["Linear"]


class Linear(torch.ao.nn.qat.Linear):
    # [CLASS DOC] Linear module with a weight FakeQuantize for dynamic quantization
    # aware training. Same interface as the QAT Linear; requires the activation observer
    # in the qconfig to be memoryless.
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for dynamic quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: Optional["QConfig"] = None,
        device: int | str | torch.device | None = None,
        dtype: str | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the base QAT Linear, then validates the qconfig.
        # Raises ValueError if the qconfig activation observer is not memoryless (a
        # moving-average observer with averaging constant 1).
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        if not torch.ao.quantization.qconfig._activation_is_memoryless(qconfig):  # type: ignore[arg-type]
            raise ValueError(
                "Dynamic QAT requires a memoryless observer."
                + "This means a MovingAverage observer with averaging constant equal to 1"
            )
