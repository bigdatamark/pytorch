# mypy: allow-untyped-defs
# [MODULE DOC] QAT Linear module (fake-quantized weight). No import-time side effects.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.intrinsic import LinearReLU
from torch.nn.utils.parametrize import (
    is_parametrized,
    transfer_parametrizations_and_params,
    type_before_parametrizations,
)


__all__ = ["Linear"]


class Linear(nn.Linear):
    # [CLASS DOC] Linear module with a weight FakeQuantize for quantization aware
    # training. State: qconfig and weight_fake_quant (built from qconfig.weight);
    # _FLOAT_MODULE is nn.Linear. Methods: forward, from_float, to_float.
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the Linear with optional device/dtype and creates
        # the weight fake-quant from qconfig. Raises AssertionError if qconfig is falsy.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        # [FUNCTION DOC] Applies a linear op using the fake-quantized weight and the
        # float bias.
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Classmethod: builds a QAT module from a float module of type
        # _FLOAT_MODULE with a valid qconfig, sharing (or transferring parametrizations
        # of) weight and bias. Raises AssertionError on wrong type or missing/falsy
        # qconfig. use_precomputed_fake_quant is unused. KNOWN ISSUE: the LinearReLU
        # unwrapping branch is unreachable here because the earlier type check requires
        # exactly _FLOAT_MODULE; it only matters for subclasses overriding
        # _FLOAT_MODULE.
        r"""Create a qat module from a float module or qparams_dict
        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        if type_before_parametrizations(mod) != cls._FLOAT_MODULE:
            raise AssertionError(
                f"qat.{cls.__name__}.from_float only works for "
                f"{cls._FLOAT_MODULE.__name__}, got {type_before_parametrizations(mod).__name__}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            qconfig=qconfig,
        )

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        # [FUNCTION DOC] Converts to a float Linear with detached copies of weight and
        # bias, preserving the training flag. Returns it.
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear
