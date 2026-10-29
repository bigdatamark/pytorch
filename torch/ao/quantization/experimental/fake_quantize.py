# [MODULE DOC] Fake quantization module for APoT quantization: APoTFakeQuantize wraps an APoTObserver and applies the custom autograd function.
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.ao.quantization.experimental.fake_quantize_function import (
    fake_quantize_function,
)
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.fake_quantize import FakeQuantizeBase


class APoTFakeQuantize(FakeQuantizeBase):
    # [CLASS DOC] Fake quantizer for APoT. Holds an observer in activation_post_process, the dtype from it, and the qparams alpha, gamma,
    # quantization_levels and level_indices (set when the observer runs; not initialised in the constructor).
    # Behavior is controlled by the observer_enabled and fake_quant_enabled flags of FakeQuantizeBase.
    alpha: Tensor
    gamma: Tensor
    quantization_levels: Tensor
    level_indices: Tensor

    def __init__(self, observer: Callable = APoTObserver, **observer_kwargs: Any):
        # [FUNCTION DOC] Create the observer from the given class (default APoTObserver) and keyword arguments and copy its dtype.
        super().__init__()
        self.activation_post_process = observer(**observer_kwargs)
        self.dtype = self.activation_post_process.dtype

    def calculate_qparams(  # type: ignore[override]
        self, signed: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # [FUNCTION DOC] Delegate to the observer's calculate_qparams with the signed flag and return its (alpha, gamma, quantization_levels, level_indices) tuple.
        return self.activation_post_process.calculate_qparams(signed=signed)

    def forward(self, X: torch.Tensor) -> Tensor:  # type: ignore[override]
        # [FUNCTION DOC] Forward pass. If the observer is enabled: feed X to the observer and store the four qparams from calculate_qparams (unsigned).
        # If fake quant is enabled: raises AssertionError if any qparam is None, then returns X passed through fake_quantize_function
        # (quantize and dequantize with masked gradient). Otherwise returns X as is.
        # KNOWN ISSUE: the qparams are never initialised, so using fake quant before observing raises AttributeError instead of the intended AssertionError.
        if self.observer_enabled[0] == 1:
            self.activation_post_process.forward(X)
            result = self.activation_post_process.calculate_qparams(signed=False)
            self.alpha = result[0]
            self.gamma = result[1]
            self.quantization_levels = result[2]
            self.level_indices = result[3]

        if self.fake_quant_enabled[0] == 1:
            if (
                self.alpha is None
                or self.gamma is None
                or self.quantization_levels is None
                or self.level_indices is None
            ):
                raise AssertionError("Must set qparams for fake quant")
            X = fake_quantize_function.apply(
                X, self.alpha, self.gamma, self.quantization_levels, self.level_indices
            )
        return X
