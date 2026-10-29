# mypy: allow-untyped-defs
# [MODULE DOC] QAT convolution modules (Conv1d, Conv2d, Conv3d) built on a private
# shared base that fake-quantizes weights. No import-time side effects.
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from torch.ao.nn.intrinsic import _FusedModule
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple


__all__ = ["Conv1d", "Conv2d", "Conv3d"]


class _ConvNd(nn.modules.conv._ConvNd):
    # [CLASS DOC] HELPER: shared base of the QAT convolutions. Adds qconfig and a weight
    # FakeQuantize (weight_fake_quant) to the float conv base; subclasses set
    # _FLOAT_MODULE and _FLOAT_CONV_MODULE. Provides forward, a from_float builder and
    # to_float (also handles fused conv-relu subclasses).
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] HELPER: initializes the float conv base with the given
        # geometry, then stores qconfig and builds weight_fake_quant. Raises
        # AssertionError if qconfig is falsy.
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        # [FUNCTION DOC] Convolves the input with the fake-quantized weight and the
        # float bias.
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] HELPER: declared as a static method whose first argument is the
        # target QAT class (subclasses call it passing cls explicitly). Builds a QAT
        # conv from a float module of exactly type _FLOAT_MODULE that has a valid
        # qconfig, unwrapping fused modules to their first element, and shares weight
        # and bias with the float module. Raises AssertionError on wrong type or
        # missing/falsy qconfig. use_precomputed_fake_quant is unused.
        r"""Create a qat module from a float module

        Args:
           `mod`: a float module, either produced by torch.ao.quantization utilities
           or directly from user
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                f"qat.{cls.__name__}.from_float only works for "
                f"{cls._FLOAT_MODULE.__name__}, got {type(mod).__name__}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        if issubclass(type(mod), _FusedModule):
            mod = mod[0]
        qconfig = mod.qconfig
        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        # [FUNCTION DOC] Converts to a float conv (_FLOAT_CONV_MODULE) with detached
        # copies of weight and bias. For fused subclasses also builds the float relu
        # from _FLOAT_RELU_MODULE and returns the fused float module with the same
        # training flag. Raises AssertionError if a fused class lacks
        # _FLOAT_RELU_MODULE.
        """This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, _FusedModule):
            modules = [conv]
            if not hasattr(cls, "_FLOAT_RELU_MODULE"):
                raise AssertionError(
                    f"{cls.__name__} must have _FLOAT_RELU_MODULE attribute"
                )
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)

            # pyrefly: ignore [missing-attribute]
            fused = cls._FLOAT_MODULE(*modules)
            fused.train(self.training)
            return fused
        else:
            return conv


class Conv1d(_ConvNd, nn.Conv1d):
    # [CLASS DOC] QAT Conv1d: nn.Conv1d with a weight FakeQuantize. _FLOAT_MODULE and
    # _FLOAT_CONV_MODULE are nn.Conv1d. Attribute weight_fake_quant fake-quantizes the
    # weight in forward.
    r"""
    A Conv1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as :class:`~torch.nn.Conv1d`

    Similar to :class:`~torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Normalizes kernel_size, stride, dilation (and padding unless it
        # is a string) to 1d tuples with _single, then initializes the base with
        # transposed=False and zero output padding. qconfig is required by the base
        # (AssertionError if falsy).
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds this QAT conv from a float conv by calling
        # the shared base builder with this class.
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv2d(_ConvNd, nn.Conv2d):
    # [CLASS DOC] QAT Conv2d: nn.Conv2d with a weight FakeQuantize. _FLOAT_MODULE and
    # _FLOAT_CONV_MODULE are nn.Conv2d. Attribute weight_fake_quant fake-quantizes the
    # weight in forward.
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Normalizes kernel_size, stride, dilation (and padding unless it
        # is a string) to 2d tuples with _pair, then initializes the base with
        # transposed=False and zero output padding. qconfig is required by the base
        # (AssertionError if falsy).
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # [FUNCTION DOC] Convolves with the fake-quantized weight and float bias (same
        # as the base).
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds this QAT conv from a float conv by calling
        # the shared base builder with this class.
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv3d(_ConvNd, nn.Conv3d):
    # [CLASS DOC] QAT Conv3d: nn.Conv3d with a weight FakeQuantize. _FLOAT_MODULE and
    # _FLOAT_CONV_MODULE are nn.Conv3d. Attribute weight_fake_quant fake-quantizes the
    # weight in forward.
    r"""
    A Conv3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv3d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv3d#torch.nn.Conv3d
    for documentation.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Normalizes kernel_size, stride, dilation (and padding unless it
        # is a string) to 3d tuples with _triple, then initializes the base with
        # transposed=False and zero output padding. qconfig is required by the base
        # (AssertionError if falsy).
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        # [FUNCTION DOC] Convolves with the fake-quantized weight and float bias (same
        # as the base).
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Classmethod: builds this QAT conv from a float conv by calling
        # the shared base builder with this class.
        return super().from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )
