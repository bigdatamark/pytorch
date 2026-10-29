# mypy: allow-untyped-defs
# [MODULE DOC] Quantized fused BatchNorm+ReLU modules (BNReLU2d, BNReLU3d) wrapping the
# quantized batch_norm_relu ops. No import-time side effects.

import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq


__all__ = ["BNReLU2d", "BNReLU3d"]


class BNReLU2d(nnq.BatchNorm2d):
    # [CLASS DOC] Quantized fused BatchNorm2d+ReLU module; subclass of the quantized
    # BatchNorm2d with the same state (scale, zero_point, affine params, running stats).
    # _FLOAT_MODULE is the float fused BNReLU2d.
    r"""
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        # [FUNCTION DOC] Forwards num_features, eps, momentum, device and dtype to the
        # base quantized batch norm constructor.
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        # [FUNCTION DOC] Applies fused quantized batch norm and ReLU via the quantized
        # batch_norm2d_relu op using stored weight, bias, running stats, eps, scale and
        # zero_point. Raises ValueError if the input is not 4-dimensional (N, C, H, W).
        r"""Applies fused BatchNorm2d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return torch.ops.quantized.batch_norm2d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedBNReLU2d.
        return "QuantizedBNReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: converts a float fused module via the parent
        # from_float (no QAT-specific handling yet).
        r"""Creates a quantized module from a float module."""
        # TODO: Add qat support for BNReLU2d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        # [FUNCTION DOC] Classmethod: builds the quantized module from a reference fused
        # module, using its first element (the batch norm) plus the given output scale
        # and zero point.
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)


class BNReLU3d(nnq.BatchNorm3d):
    # [CLASS DOC] Quantized fused BatchNorm3d+ReLU module; subclass of the quantized
    # BatchNorm3d with the same state (scale, zero_point, affine params, running stats).
    # _FLOAT_MODULE is the float fused BNReLU3d.
    r"""
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        # [FUNCTION DOC] Forwards num_features, eps, momentum, device and dtype to the
        # base quantized batch norm constructor.
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        # [FUNCTION DOC] Applies fused quantized batch norm and ReLU via the quantized
        # batch_norm3d_relu op using stored weight, bias, running stats, eps, scale and
        # zero_point. Raises ValueError if the input is not 5-dimensional (N, C, D, H,
        # W).
        r"""Applies fused BatchNorm3d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return torch.ops.quantized.batch_norm3d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedBNReLU3d.
        return "QuantizedBNReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: converts a float fused module via the parent
        # from_float (no QAT-specific handling yet).
        r"""Creates a quantized module from a float module."""
        # TODO: Add qat support for BNReLU3d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        # [FUNCTION DOC] Classmethod: builds the quantized module from a reference fused
        # module, using its first element (the batch norm) plus the given output scale
        # and zero point.
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)
