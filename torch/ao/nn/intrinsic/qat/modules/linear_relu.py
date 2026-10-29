# [MODULE DOC] Defines the QAT fused LinearReLU module (fake-quantized weight, linear
# then relu). No import-time side effects beyond imports.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.ao.nn.intrinsic.modules.fused import _FusedModule


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny


__all__ = ["LinearReLU"]


class LinearReLU(nnqat.Linear, _FusedModule):
    # [CLASS DOC] QAT fused Linear+ReLU module. Subclass of the QAT Linear and
    # _FusedModule; holds the weight fake-quant module inherited from the base class.
    # _FLOAT_MODULE is the float fused LinearReLU. Methods: forward, from_float,
    # to_float.
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # pyrefly: ignore [bad-override]
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: QConfigAny = None,
    ) -> None:
        # [FUNCTION DOC] Constructs the QAT Linear with the given sizes, bias flag and
        # qconfig (forwarded to the base class).
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Fake-quantizes the weight, applies a linear op with the bias,
        # then ReLU. Returns the output tensor.
        return F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(
        cls,
        mod: torch.nn.Module,
        use_precomputed_fake_quant: bool = False,
    ) -> LinearReLU:
        # [FUNCTION DOC] Classmethod: builds a QAT module from a float fused module by
        # delegating to the parent class from_float; use_precomputed_fake_quant is
        # passed through.
        return super().from_float(mod, use_precomputed_fake_quant)  # type: ignore[no-untyped-call,no-any-return]

    def to_float(self) -> nni.LinearReLU:
        # [FUNCTION DOC] Converts back to a float fused LinearReLU: creates a float
        # Linear with detached copies of weight (and bias if present) as new Parameters,
        # pairs it with a ReLU. Returns the fused module.
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.ao.nn.intrinsic.LinearReLU(linear, relu)  # type: ignore[no-untyped-call]
