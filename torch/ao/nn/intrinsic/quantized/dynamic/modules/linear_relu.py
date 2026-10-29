# [MODULE DOC] Defines the dynamic-quantized fused LinearReLU module. No import-time
# side effects.
from typing import Any
from typing_extensions import Self

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd


__all__ = ["LinearReLU"]


class LinearReLU(nnqd.Linear):
    # [CLASS DOC] Dynamically quantized fused Linear+ReLU; subclass of the dynamic
    # quantized Linear supporting qint8 and float16 weights. _FLOAT_MODULE is the float
    # fused LinearReLU.
    r"""
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
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
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        # [FUNCTION DOC] Constructs the module; forwards sizes, bias flag and weight
        # dtype to the base class.
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Runs the dynamic quantized linear+relu op matching the packed
        # weight dtype (qint8 uses reduce_range; float16 uses the fp16 op). Raises
        # RuntimeError for any other dtype. Returns the result cast to the input dtype.
        if self._packed_params.dtype == torch.qint8:
            # TODO check if we should set reduce_range = True by default here
            Y = torch.ops.quantized.linear_relu_dynamic(
                x, self._packed_params._packed_params, reduce_range=True
            )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_relu_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear relu!")
        return Y.to(x.dtype)

    def _get_name(self) -> str:
        # [FUNCTION DOC] HELPER: returns the display name DynamicQuantizedLinearReLU.
        return "DynamicQuantizedLinearReLU"

    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = False
    ) -> Self:
        # [FUNCTION DOC] Classmethod: converts a float fused module to this dynamic
        # quantized module via the parent from_float.
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qlinear_relu: Any) -> Self:  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds the module from a reference quantized fused
        # module (a sequence); uses its first element (the reference linear) with the
        # parent from_reference.
        return super().from_reference(ref_qlinear_relu[0])
