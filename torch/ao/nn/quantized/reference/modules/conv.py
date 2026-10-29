# mypy: allow-untyped-defs
# [MODULE DOC] Reference quantized convolution modules: Conv1d/2d/3d and
# ConvTranspose1d/2d/3d plus private base classes. They keep float weights (no
# backend packing) and, in forward, run the float functional convolution on a
# quantize-dequantize round trip of the weight so backends can lower them.
# Provides from_float constructors that copy a float conv's configuration and
# detached weight/bias. No import-time side effects.
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t

from .utils import ReferenceQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]


class _ConvNd(torch.nn.modules.conv._ConvNd, ReferenceQuantizedModule):
    # [CLASS DOC] HELPER: Base for the reference convolution modules (combines the
    # float conv base with ReferenceQuantizedModule). Does not pack parameters.
    # Provides the shared from_float builder used by the Conv1d/2d/3d classes;
    # _IS_REFERENCE is true.
    """A reference version of nn.quantized.Conv2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    __annotations__ = {"bias": torch.Tensor | None}
    _IS_REFERENCE = True

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
    # [FUNCTION DOC] HELPER: Static builder (the target class is passed
    # explicitly as cls): creates an instance of cls from a float conv, copying
    # channels, kernel size, stride, padding, dilation, groups, bias presence,
    # padding mode, device and dtype, and passes weight_qparams; then replaces
    # weight and bias with detached copies wrapped as new parameters. Returns
    # the new module.
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class Conv1d(_ConvNd, nn.Conv1d):
    # [CLASS DOC] Reference quantized 1D convolution: float Conv1d plus weight
    # quantization parameters. Activations stay float; the weight is fake-
    # quantized in forward. Built directly or via from_float.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float Conv1d base with the given
        # arguments (channels, kernel, stride, padding, dilation, groups, bias,
        # padding mode, device, dtype), then sets up weight quantization
        # parameters from weight_qparams (mapping, may be None).
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Gets the quantize-dequantized weight and applies float
        # conv1d to x with the module's bias, stride, padding, dilation and
        # groups; returns the float result.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv1d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConv1d(Reference).
        return "QuantizedConv1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference Conv1d from a float
        # conv and weight_qparams through the shared base builder. Returns the
        # new module.
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv2d(_ConvNd, nn.Conv2d):
    # [CLASS DOC] Reference quantized 2D convolution: float Conv2d plus weight
    # quantization parameters. Activations stay float; the weight is fake-
    # quantized in forward. Built directly or via from_float.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float Conv2d base with the given
        # arguments (channels, kernel, stride, padding, dilation, groups, bias,
        # padding mode, device, dtype), then sets up weight quantization
        # parameters from weight_qparams (mapping, may be None).
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Gets the quantize-dequantized weight and applies float
        # conv2d to x with the module's bias, stride, padding, dilation and
        # groups; returns the float result.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv2d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConv2d(Reference).
        return "QuantizedConv2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference Conv2d from a float
        # conv and weight_qparams through the shared base builder. Returns the
        # new module.
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class Conv3d(_ConvNd, nn.Conv3d):
    # [CLASS DOC] Reference quantized 3D convolution: float Conv3d plus weight
    # quantization parameters. Activations stay float; the weight is fake-
    # quantized in forward. Built directly or via from_float.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float Conv3d base with the given
        # arguments (channels, kernel, stride, padding, dilation, groups, bias,
        # padding mode, device, dtype), then sets up weight quantization
        # parameters from weight_qparams (mapping, may be None).
        nn.Conv3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Gets the quantize-dequantized weight and applies float
        # conv3d to x with the module's bias, stride, padding, dilation and
        # groups; returns the float result.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.conv3d ---

        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.conv3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """
        weight_quant_dequant = self.get_weight()

        result = F.conv3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConv3d(Reference).
        return "QuantizedConv3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference Conv3d from a float
        # conv and weight_qparams through the shared base builder. Returns the
        # new module.
        return _ConvNd.from_float(cls, float_conv, weight_qparams)


class _ConvTransposeNd(_ConvNd, torch.nn.modules.conv._ConvTransposeNd):
    # [CLASS DOC] HELPER: Base for the reference transposed convolution modules;
    # same role as the non-transposed base but its from_float passes constructor
    # arguments in the transposed-conv order (output_padding before bias, dilation
    # after bias).
    """A reference version of nn.quantized.ConvTranspose2d
    we will not pack the parameters in this module, since weight packing is an
    optimization for quantized backends supported in PyTorch (fbgemm/qnnpack),
    this is useful when user want to use this module in other backends like Glow.
    """

    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
    # [FUNCTION DOC] HELPER: Static builder (cls passed explicitly): creates an
    # instance from a float transposed conv, copying channels, kernel size,
    # stride, padding, output padding, groups, bias presence, dilation, padding
    # mode, device and dtype, plus weight_qparams; replaces weight and bias with
    # detached parameter copies. Returns the new module.
        qref_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.output_padding,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        qref_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            qref_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return qref_conv


class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    # [CLASS DOC] Reference quantized 1D transposed convolution: float
    # ConvTranspose1d plus weight quantization parameters. Activations stay
    # float; the weight is fake-quantized in forward. Built directly or via
    # from_float.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float ConvTranspose1d base with the
        # given arguments (channels, kernel, stride, padding, output padding,
        # groups, bias, dilation, padding mode, device, dtype), then sets up
        # weight quantization parameters from weight_qparams (mapping, may be
        # None).
        nn.ConvTranspose1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        # [FUNCTION DOC] Computes the output padding for the optional
        # output_size, gets the quantize-dequantized weight and applies float
        # conv_transpose1d to x; returns the float result. Raises AssertionError
        # if padding is not a tuple.
        # KNOWN ISSUE: the output padding computation is passed the builtin name
        # input rather than the argument x, so the call with a real output_size
        # (or even without) is likely to fail or use the wrong object.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose1d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose1d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv1d
        """

        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConvTranspose1d(Reference).
        return "QuantizedConvTranspose1d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference ConvTranspose1d from a
        # float transposed conv and weight_qparams through the shared base
        # builder. Returns the new module.
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    # [CLASS DOC] Reference quantized 2D transposed convolution: float
    # ConvTranspose2d plus weight quantization parameters. Activations stay
    # float; the weight is fake-quantized in forward. Built directly or via
    # from_float.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float ConvTranspose2d base with the
        # given arguments (channels, kernel, stride, padding, output padding,
        # groups, bias, dilation, padding mode, device, dtype), then sets up
        # weight quantization parameters from weight_qparams (mapping, may be
        # None).
        nn.ConvTranspose2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        # [FUNCTION DOC] Computes the output padding for the optional
        # output_size, gets the quantize-dequantized weight and applies float
        # conv_transpose2d to x; returns the float result. Raises AssertionError
        # if padding is not a tuple.
        # KNOWN ISSUE: the output padding computation is passed the builtin name
        # input rather than the argument x, so the call with a real output_size
        # (or even without) is likely to fail or use the wrong object.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose2d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose2d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv2d
        """
        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.

        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConvTranspose2d(Reference).
        return "QuantizedConvTranspose2d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference ConvTranspose2d from a
        # float transposed conv and weight_qparams through the shared base
        # builder. Returns the new module.
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)


class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    # [CLASS DOC] Reference quantized 3D transposed convolution: float
    # ConvTranspose3d plus weight quantization parameters. Activations stay
    # float; the weight is fake-quantized in forward. Built directly or via
    # from_float.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Initializes the float ConvTranspose3d base with the
        # given arguments (channels, kernel, stride, padding, output padding,
        # groups, bias, dilation, padding mode, device, dtype), then sets up
        # weight quantization parameters from weight_qparams (mapping, may be
        # None).
        nn.ConvTranspose3d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            # pyrefly: ignore [bad-argument-type]
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(
        self, x: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        # [FUNCTION DOC] Computes the output padding for the optional
        # output_size, gets the quantize-dequantized weight and applies float
        # conv_transpose3d to x; returns the float result. Raises AssertionError
        # if padding is not a tuple.
        # KNOWN ISSUE: the output padding computation is passed the builtin name
        # input rather than the argument x, so the call with a real output_size
        # (or even without) is likely to fail or use the wrong object.
        """
        we have:
        w(float) -- quant - dequant \
        x(float) ------------- F.convTranspose3d ---
        In the full model, we will see
        w(float) -- quant - *dequant \
        x -- quant --- *dequant --  *F.convTranspose3d --- *quant - dequant
        and the backend should be able to fuse the ops with `*` into a quantized conv3d
        """

        if not isinstance(self.padding, tuple):
            raise AssertionError(
                f"Expected self.padding to be a tuple, got {type(self.padding)}"
            )
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            input,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose3d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedConvTranspose3d(Reference).
        return "QuantizedConvTranspose3d(Reference)"

    @classmethod
    def from_float(cls, float_conv, weight_qparams):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds a reference ConvTranspose3d from a
        # float transposed conv and weight_qparams through the shared base
        # builder. Returns the new module.
        return _ConvTransposeNd.from_float(cls, float_conv, weight_qparams)
