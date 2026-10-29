# [MODULE DOC] Reference quantized Linear module for FX graph mode quantization.
# Defines Linear, which keeps a float weight and runs float linear after a
# quantize-dequantize round trip of the weight, so backends can pattern-match
# and lower it. No import-time side effects.
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ReferenceQuantizedModule


__all__ = ["Linear"]


class Linear(nn.Linear, ReferenceQuantizedModule):
    # [CLASS DOC] Reference quantized linear layer: subclass of the float Linear
    # and ReferenceQuantizedModule. Stores the float weight and bias plus weight
    # quantization parameters (set by _init_weight_qparams); activations stay
    # float. _IS_REFERENCE is true.
    """A reference quantized linear module that fits into the FX
    Graph Mode Quantization workflow
    activation will be floating point Tensor, we will store floating
    point weight as well in the module, but in forward we'll quantize
    and dequantize the weight before running the floating point functional
    linear operator.
    """

    _IS_REFERENCE = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight_qparams: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float Linear base with the given sizes,
        # bias flag, device and dtype, then sets up weight quantization
        # parameters from weight_qparams (a mapping, may be None) via the
        # reference-module helper.
        super().__init__(in_features, out_features, bias_, device, dtype)
        self._init_weight_qparams(weight_qparams, device)

    def _get_name(self) -> str:
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedLinear(Reference).
        return "QuantizedLinear(Reference)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Gets the weight after a quantize-dequantize round trip
        # (get_weight) and applies float linear to input x with the module's
        # bias; returns the float result. No quantization of activations happens
        # here.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.linear ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.linear --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized linear
        """
        weight_quant_dequant = self.get_weight()
        result = F.linear(x, weight_quant_dequant, self.bias)
        return result

    @classmethod
    def from_float(
        cls, float_linear: nn.Linear, weight_qparams: dict[str, Any]
    ) -> "Linear":
        # [FUNCTION DOC] Builds a reference Linear from a float Linear and a
        # weight_qparams mapping, copying sizes, device, dtype and presence of
        # bias. The weight and bias are detached copies wrapped as new
        # parameters (they share storage with the originals). Returns the new
        # module.
        qref_linear = Linear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            qref_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return qref_linear
