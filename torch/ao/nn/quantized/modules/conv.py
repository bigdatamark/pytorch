# mypy: allow-untyped-defs
# [MODULE DOC] Quantized convolution and transposed-convolution modules (Conv1d/2d/3d, ConvTranspose1d/2d/3d)
# built on a shared private base _ConvNd. Weights are stored prepacked in backend-specific packed params
# and converted to and from quantized tensors for state dict, pickling and copying. Modules store output
# scale/zero_point and are created from float, QAT or reference modules. No import-time side effects
# beyond defining the supported padding mode set (zeros, reflect).
r"""Quantized convolution modules."""

from typing import ClassVar, Literal

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn as nn
import torch.nn.functional as F
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.utils import fuse_conv_bn_weights

from .utils import _quantize_weight, WeightedQuantizedModule


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
]

_SUPPORTED_PADDING = {"zeros", "reflect"}


def _reverse_repeat_padding(padding: list[int]) -> list[int]:
# [FUNCTION DOC] HELPER: Converts per-dimension padding amounts into the flat list expected by pad:
# dimensions are reversed and every amount is repeated twice (once per side). Returns a new list.
    _reversed_padding_repeated_twice: list[int] = []
    N = len(padding)
    for idx in range(N):
        _reversed_padding_repeated_twice.extend(padding[N - idx - 1] for _ in range(2))
    return _reversed_padding_repeated_twice


class _ConvNd(WeightedQuantizedModule):
    # [CLASS DOC] HELPER: Shared base of the quantized conv modules (derived from WeightedQuantizedModule).
    # State: in/out channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups,
    # padding_mode, plus scale and zero_point for the output and a prepacked weight/bias stored by
    # subclasses via set_weight_bias. Handles repr, state dict (unpacked weight, bias, scale, zero_point),
    # pickling, copying and the conversion classmethods. Subclasses must call _init instead of __init__.
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
    ):
        # [FUNCTION DOC] HELPER: Placeholder constructor that always raises NotImplementedError; subclasses define
        # their own constructors and call _init (see the referenced PR note).
        # All subclasses have this signature - See PR #49702s
        raise NotImplementedError

    def _init(
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
        padding_mode="zeros",
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Real initializer for subclasses. Validates that out_channels is positive and
        # that in_channels and out_channels are divisible by groups, and that padding_mode is supported, raising
        # ValueError otherwise. Stores the hyperparameters, builds a default qint8 quantized weight (scale 1,
        # zero point 0, shape depends on transposed) and a zero float bias if requested, packs them through
        # set_weight_bias and sets scale 1.0 and zero_point 0. Device is applied to the weight and bias.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if out_channels <= 0:
            raise ValueError(f"out_channels must be greater than 0, got {out_channels}")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(
                f"'padding_mode' {padding_mode} is not supported by quantized convolution"
            )
        self.padding_mode = padding_mode
        # Initialize as NCHW. set_weight will internally transpose to NHWC.
        if self.transposed:
            weight_shape = [in_channels, out_channels // self.groups]
        else:
            weight_shape = [out_channels, in_channels // self.groups]
        qweight = torch._empty_affine_quantized(
            weight_shape + list(kernel_size),
            scale=1,
            zero_point=0,
            dtype=torch.qint8,
            **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
        )
        bias_float = (
            torch.zeros(
                out_channels,
                dtype=torch.float,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            )
            if bias
            else None
        )

        self.set_weight_bias(qweight, bias_float)
        self.scale = 1.0
        self.zero_point = 0

    def set_weight_bias(self, qweight, bias_float):
        # [FUNCTION DOC] HELPER: Abstract hook for subclasses to pack a quantized weight and float bias; raises NotImplementedError.
        raise NotImplementedError

    def bias(self):
        # [FUNCTION DOC] HELPER: Abstract accessor for the bias; raises NotImplementedError.
        raise NotImplementedError

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Abstract accessor returning unpacked (weight, bias); raises NotImplementedError.
        raise NotImplementedError

    def extra_repr(self):
        # [FUNCTION DOC] HELPER: Builds the repr string with channels, kernel size, stride, scale and zero point,
        # adding padding, dilation, output_padding, groups only when non-default and noting bias=False when
        # no bias exists. Formats using the module attribute mapping.
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, scale={scale}, zero_point={zero_point}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias() is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    # ===== Serialization methods =====
    # The special consideration here is that we have to unpack the weights into
    # their regular QTensor form for serialization. Packed weights should not
    # live outside the process in which they were created, rather they should be
    # derived from the QTensor weight.
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # TODO: maybe change to this when https://github.com/pytorch/pytorch/pull/32958 is landed
    #   self
    #   |--- _packed_params : Conv2dPackedParamsBase or Conv3dPackedParamsBase
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # [FUNCTION DOC] HELPER: Extends state dict saving with the unpacked weight and bias plus scale and
        # zero_point as tensors under prefixed keys (packed weights are never serialized). Mutates destination.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        (w, b) = self._weight_bias()
        destination[prefix + "weight"] = w
        destination[prefix + "bias"] = b
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    @torch.jit.export
    def __getstate__(self):
        # [FUNCTION DOC] HELPER: Returns a tuple of the hyperparameters, unpacked weight and bias, scale,
        # zero_point and training flag, used for pickling and copying. Exported for TorchScript.
        (w, b) = self._weight_bias()
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.transposed,
            self.output_padding,
            self.groups,
            self.padding_mode,
            w,
            b,
            self.scale,
            self.zero_point,
            self.training,
        )

    # ===== Deserialization methods =====
    # Counterpart to the serialization methods, we must pack the serialized
    # QTensor weight into its packed format for use by the FBGEMM ops.
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # [FUNCTION DOC] HELPER: Reads and removes the prefixed weight, bias, scale and zero_point from state_dict,
        # repacks the weight and bias, then runs the standard loading with strict checking disabled.
        # Raises KeyError if an entry is missing.
        self.set_weight_bias(state_dict[prefix + "weight"], state_dict[prefix + "bias"])
        state_dict.pop(prefix + "weight")
        state_dict.pop(prefix + "bias")
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")
        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def __setstate__(self, state):
        # [FUNCTION DOC] HELPER: Restores attributes from the tuple produced by __getstate__ and repacks the
        # weight and bias. Exported for TorchScript.
        self.in_channels = state[0]
        self.out_channels = state[1]
        self.kernel_size = state[2]
        self.stride = state[3]
        self.padding = state[4]
        self.dilation = state[5]
        self.transposed = state[6]
        self.output_padding = state[7]
        self.groups = state[8]
        self.padding_mode = state[9]
        self.set_weight_bias(state[10], state[11])
        self.scale = state[12]
        self.zero_point = state[13]
        self.training = state[14]

    def __deepcopy__(self, memo):
        # [FUNCTION DOC] HELPER: Deep copy support: creates a bare instance, initializes the Module base and restores
        # the state obtained from __getstate__ (memo is unused). Returns the new instance.
        new_instance = type(self).__new__(type(self))
        torch.nn.Module.__init__(new_instance)
        state = self.__getstate__()
        new_instance.__setstate__(state)
        return new_instance

    def __copy__(self):
        # [FUNCTION DOC] HELPER: Shallow copy is implemented as a full deep copy; returns the new instance.
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(cls, mod, activation_post_process, weight_post_process=None):
        # [FUNCTION DOC] Class method: builds a quantized conv of class cls from a float or QAT module mod. Runs the
        # weight observer (taken from the module qconfig if not given) on mod's weight and requires it to have
        # dtype qint8 (AssertionError otherwise), quantizes the weight, constructs the module using mod's
        # hyperparameters, and packs the weight and bias. If an activation observer with non-float dtype is given
        # its scale and zero point are stored as output qparams; otherwise (dynamic quantization) they stay default.
        r"""Creates a qconv object and returns it."""
        if weight_post_process is None:
            weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        if weight_post_process.dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have a dtype of qint8, got {weight_post_process.dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvNd
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            mod.bias is not None,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            activation_post_process is None
            or activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = activation_post_process.calculate_qparams()
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static helper (takes the target class explicitly) that converts a float or QAT conv. For a
        # QAT module (has weight_fake_quant) it folds batch norm into the weight and bias when the module is the
        # conv-bn type, requires an attached activation_post_process and uses the fake quant as weight observer.
        # For a float module it requires the exact float type and a qconfig, unwraps fused relu/add modules to the
        # inner conv and takes the weight observer from qconfig. Raises AssertionError on violations. Delegates to get_qconv.
        if hasattr(mod, "weight_fake_quant"):
            # assert type(mod) is cls.__QAT_MODULE, " nnq." + cls.__name__ + \
            # ".from_float only works for " + cls.__QAT_MODULE.__name__
            if type(mod) is cls._NNIQAT_CONV_BN_MODULE:
                mod.weight, mod.bias = fuse_conv_bn_weights(
                    mod.weight,
                    mod.bias,
                    mod.bn.running_mean,
                    mod.bn.running_var,
                    mod.bn.eps,
                    mod.bn.weight,
                    mod.bn.bias,
                )
            if not hasattr(mod, "activation_post_process"):
                raise AssertionError("Input QAT module must have observer attached")
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            if type(mod) is not cls._FLOAT_MODULE:
                raise AssertionError(
                    f"nnq.{cls.__name__}.from_float only works for "
                    f"{cls._FLOAT_MODULE.__name__} but got: {type(mod).__name__}"
                )
            if not hasattr(mod, "qconfig"):
                raise AssertionError("Input float module must have qconfig defined.")
            activation_post_process = (
                None
                if not hasattr(mod, "activation_post_process")
                else mod.activation_post_process
            )
            if type(mod) in [
                cls._NNI_CONV_RELU_MODULE,
                cls._NNI_CONV_ADD_MODULE,
                cls._NNI_CONV_ADD_RELU_MODULE,
            ]:
                mod = mod[0]
            weight_post_process = mod.qconfig.weight()
        return cls.get_qconv(mod, activation_post_process, weight_post_process)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        # [FUNCTION DOC] Class method: builds a quantized conv from a reference quantized conv ref_qconv, copying its
        # hyperparameters, device and dtype, taking its quantized weight and bias, and setting the given
        # output_scale and output_zero_point. Returns the new module.
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconv (Module): a reference quantized  module, either produced by torch.ao.quantization
                                utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconv.in_channels,
            ref_qconv.out_channels,
            ref_qconv.kernel_size,  # type: ignore[arg-type]
            ref_qconv.stride,  # type: ignore[arg-type]
            ref_qconv.padding,  # type: ignore[arg-type]
            ref_qconv.dilation,  # type: ignore[arg-type]
            ref_qconv.groups,
            ref_qconv.bias is not None,  # type: ignore[arg-type]
            ref_qconv.padding_mode,
            device=ref_qconv.weight.device,
            dtype=ref_qconv.weight.dtype,
        )
        qweight = ref_qconv.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconv.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class Conv1d(_ConvNd):
    # [CLASS DOC] Quantized 1D convolution (input quint8, weight packed qint8). Class attributes name the
    # float module and the fused/QAT module types it can be converted from. Output qparams in scale and
    # zero_point; the weight is held in _packed_params. Padding modes: zeros natively, reflect by explicit padding.
    r"""Applies a 1D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv1d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv1d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0,
        ...                                     dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn1d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU1d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding (unless a string) and dilation to 1-tuples and
        # calls the shared _init with transposed False and zero output padding.
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)

        padding = padding if isinstance(padding, str) else _single(padding)
        dilation = _single(dilation)

        # Subclasses of _ConvNd needs to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConv1d".
        return "QuantizedConv1d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks quantized weight w and optional float bias b with the conv1d prepack kernel and
        # stores the result in _packed_params. For non-zero padding modes the packed padding is zero because
        # padding is applied explicitly in forward.
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv1d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Unpacks _packed_params and returns the (weight, bias) pair.
        w, b = torch.ops.quantized.conv1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        return self._weight_bias()[0]

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        return self._weight_bias()[1]

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized 1D convolution. Raises ValueError if the input is not 3-dimensional.
        # For non-zero padding modes the input is first padded explicitly using the first padding value.
        # Returns a quantized tensor with the stored scale and zero_point.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv1d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts a float or QAT module via the shared _ConvNd conversion helper.
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv2d(_ConvNd):
    # [CLASS DOC] Quantized 2D convolution (input quint8, weight packed qint8); class attributes name the
    # float, QAT conv-bn, fused relu and add module types it converts from. Stores output scale/zero_point and _packed_params.
    r"""Applies a 2D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv2d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn2d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU2d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nni.ConvAdd2d]] = nni.ConvAdd2d
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nni.ConvAddReLU2d]] = nni.ConvAddReLU2d

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding and dilation to pairs and calls the shared _init
        # with transposed False and zero output padding.
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConv2d".
        return "QuantizedConv2d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks quantized weight w and optional float bias b with the conv2d prepack kernel and
        # stores the result in _packed_params. For non-zero padding modes the packed padding is zero.
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv2d_prepack(
                w, b, self.stride, _pair(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the (weight, bias) pair unpacked from the packed params.
        return self._packed_params.unpack()

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        return self._weight_bias()[0]

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        return self._weight_bias()[1]

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized 2D convolution. Raises ValueError if the input is not 4-dimensional.
        # For non-zero padding modes the input is explicitly padded first. Returns a quantized tensor with the stored qparams.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv2d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts a float or QAT module via the shared _ConvNd conversion helper.
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class Conv3d(_ConvNd):
    # [CLASS DOC] Quantized 3D convolution (input quint8, weight packed qint8); class attributes name the
    # float, QAT conv-bn and fused relu module types it converts from. Reflect padding is not supported.
    r"""Applies a 3D convolution over a quantized input signal composed of
    several quantized input planes.

    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.Conv3d`.

    .. note::
        Only `zeros` is supported for the :attr:`padding_mode` argument.

    .. note::
        Only `torch.quint8` is supported for the input data type.


    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point

    See :class:`~torch.nn.Conv3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # With square kernels and equal stride
        >>> m = nn.quantized.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.quantized.Conv3d(16, 33, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), dilation=(1, 2, 2))
        >>> input = torch.randn(20, 16, 56, 56, 56)
        >>> # quantize input to quint8
        >>> # xdoctest: +SKIP
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)

    """

    _FLOAT_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _NNIQAT_CONV_BN_MODULE: ClassVar[type[nn.Module] | None] = nniqat.ConvBn3d
    _NNI_CONV_RELU_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU3d
    _NNI_CONV_ADD_MODULE: ClassVar[type[nn.Module] | None] = None
    _NNI_CONV_ADD_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding and dilation to triples and calls the shared _init
        # with transposed False and zero output padding. Raises AssertionError if padding_mode is reflect.
        if padding_mode == "reflect":
            raise AssertionError("Conv3d does not support reflection padding")
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConv3d".
        return "QuantizedConv3d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks quantized weight w and optional float bias b with the conv3d prepack kernel and
        # stores the result in _packed_params. For non-zero padding modes the packed padding is zero.
        if self.padding_mode == "zeros":
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            self._packed_params = torch.ops.quantized.conv3d_prepack(
                w, b, self.stride, _triple(0), self.dilation, self.groups
            )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the (weight, bias) pair unpacked from the packed params.
        return self._packed_params.unpack()

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        return self._weight_bias()[0]

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        return self._weight_bias()[1]

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized 3D convolution. Raises ValueError if the input is not 5-dimensional.
        # For non-zero padding modes the input is explicitly padded first. Returns a quantized tensor with the stored qparams.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return ops.quantized.conv3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts a float or QAT module via the shared _ConvNd conversion helper.
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


# === Transposed Convolutions ===


class _ConvTransposeNd(_ConvNd):
    # [CLASS DOC] HELPER: Shared base of the quantized transposed convolutions. Only zeros padding is allowed.
    # Provides float conversion and reference conversion; subclasses supply the dimension specific packing,
    # unpacking and forward, and name their float module type in _FLOAT_MODULE.
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
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
        device=None,
        dtype=None,
    ):
        # [FUNCTION DOC] HELPER: Raises ValueError if padding_mode is not zeros; otherwise forwards all arguments
        # to the shared _init of the conv base.
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Subclasses of _ConvNd need to call _init rather than __init__. See
        # discussion on PR #49702
        super()._init(
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

    def _input_padding(
        self, kernel_size: list[int], dilation: list[int], padding: list[int]
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Computes per-dimension input padding as dilation times (kernel size minus one)
        # minus padding for each dimension and returns it as a list. Not used by the code in this file.
        res = torch.jit.annotate(list[int], [])
        for kdx in range(len(kernel_size)):
            pad = dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx]
            res.append(pad)
        return res

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts a float transposed conv mod (exact type must equal the class's
        # _FLOAT_MODULE and it needs a qconfig, else AssertionError). Runs the qconfig weight observer (must be
        # qint8, else AssertionError), quantizes the weight, constructs the module from mod's hyperparameters and
        # packs weight and bias. Activation scale and zero point are copied from the module's
        # activation_post_process observer unless it is missing or float (dynamic quantization).
        r"""Creates a quantized module from a float module or qparams_dict.
        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
              utilities or provided by the user
        """
        # derived classes override cls._FLOAT_MODULE attribute
        msg = (
            " nnq."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        )
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(msg)
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined.")
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]
        weight_post_process(mod.weight)
        if weight_post_process.dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have a dtype of qint8, got {weight_post_process.dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # the __init__ call used is the one from derived classes and not the one from _ConvTransposeNd
        # pyrefly: ignore [missing-argument]
        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,  # type: ignore[call-arg]
            mod.stride,
            mod.padding,
            mod.output_padding,
            mod.groups,
            mod.bias is not None,
            mod.dilation,
            mod.padding_mode,
        )
        qconv.set_weight_bias(qweight, mod.bias)
        if (
            not hasattr(mod, "activation_post_process")
            or mod.activation_post_process.dtype == torch.float
        ):
            return qconv  # dynamic quantization doesn't need scale/zero_point
        else:
            act_scale, act_zp = mod.activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
            qconv.scale = float(act_scale)
            qconv.zero_point = int(act_zp)
            return qconv

    @staticmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        # [FUNCTION DOC] Static helper (takes the target class explicitly) that builds a quantized transposed conv
        # from a reference module: copies hyperparameters, device and dtype, sets the reference's quantized
        # weight and bias, and sets output_scale and output_zero_point. Returns the new module.
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module
        Args:
            ref_qconvt (Module): a reference quantized  module, either produced by torch.ao.quantization
                                 utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        qconv = cls(
            ref_qconvt.in_channels,
            ref_qconvt.out_channels,
            ref_qconvt.kernel_size,  # type: ignore[arg-type]
            ref_qconvt.stride,  # type: ignore[arg-type]
            ref_qconvt.padding,  # type: ignore[arg-type]
            ref_qconvt.output_padding,  # type: ignore[arg-type]
            ref_qconvt.groups,
            ref_qconvt.bias is not None,  # type: ignore[arg-type]
            ref_qconvt.dilation,  # type: ignore[arg-type]
            ref_qconvt.padding_mode,
            device=ref_qconvt.weight.device,
            dtype=ref_qconvt.weight.dtype,
        )
        qweight = ref_qconvt.get_quantized_weight()
        qconv.set_weight_bias(qweight, ref_qconvt.bias)
        qconv.scale = float(output_scale)
        qconv.zero_point = int(output_zero_point)
        return qconv


class ConvTranspose1d(_ConvTransposeNd):
    # [CLASS DOC] Quantized 1D transposed convolution (documented as QNNPACK-engine only). Created directly
    # or via from_reference; from_float is inherited from the shared base.
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose1d`.

    .. note:: Currently only the QNNPACK engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'qnnpack'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv1d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With square kernels and equal stride
        >>> m = nnq.ConvTranspose1d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose1d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv1d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose1d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose1d]] = nn.ConvTranspose1d

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding, dilation and output_padding to 1-tuples and
        # calls the transposed base initializer with transposed set to true.
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConvTranspose1d".
        return "QuantizedConvTranspose1d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks weight w and optional bias b with the transposed conv1d prepack kernel (using
        # stride, padding, output_padding, dilation and groups) and stores the packed params.
        self._packed_params = torch.ops.quantized.conv_transpose1d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the (weight, bias) pair unpacked by the transposed conv1d unpack kernel.
        w, b = torch.ops.quantized.conv_transpose1d_unpack(self._packed_params)
        return w, b

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized transposed 1D convolution with the stored scale and zero_point.
        # Raises ValueError if the input is not 3-dimensional.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        return torch.ops.quantized.conv_transpose1d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        # [FUNCTION DOC] Class method: builds the module from a reference quantized module via the shared
        # transposed-conv helper, with the given output scale and zero point.
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose2d(_ConvTransposeNd):
    # [CLASS DOC] Quantized 2D transposed convolution (QNNPACK or FBGEMM). Created directly or via from_reference.
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose2d`.

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv2d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose2d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> # QNNPACK or FBGEMM as backend
        >>> torch.backends.quantized.engine = 'qnnpack'
        >>> # With square kernels and equal stride
        >>> import torch.ao.nn.quantized as nnq
        >>> m = nnq.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose2d]] = nn.ConvTranspose2d

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding, dilation and output_padding to pairs and calls
        # the transposed base initializer with transposed set to true.
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConvTranspose2d".
        return "QuantizedConvTranspose2d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks weight w and optional bias b with the transposed conv2d prepack kernel and stores the packed params.
        self._packed_params = torch.ops.quantized.conv_transpose2d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the (weight, bias) pair unpacked by the conv2d unpack kernel.
        w, b = torch.ops.quantized.conv2d_unpack(self._packed_params)
        return w, b

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized transposed 2D convolution with the stored qparams. Raises ValueError if
        # the input is not 4-dimensional.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.conv_transpose2d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        # [FUNCTION DOC] Class method: builds the module from a reference quantized module via the shared
        # transposed-conv helper, with the given output scale and zero point.
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )


class ConvTranspose3d(_ConvTransposeNd):
    # [CLASS DOC] Quantized 3D transposed convolution (documented as FBGEMM-engine only). Created directly or via from_reference.
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes.
    For details on input arguments, parameters, and implementation see
    :class:`~torch.nn.ConvTranspose3d`.

    .. note:: Currently only the FBGEMM engine is implemented.
        Please, set the `torch.backends.quantized.engine = 'fbgemm'`

    For special notes, please, see :class:`~torch.ao.nn.quantized.Conv3d`

    Attributes:
        weight (Tensor):     packed tensor derived from the learnable weight
                             parameter.
        scale (Tensor):      scalar for the output scale
        zero_point (Tensor): scalar for the output zero point
    See :class:`~torch.nn.ConvTranspose3d` for other attributes.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> torch.backends.quantized.engine = 'fbgemm'
        >>> from torch.ao.nn import quantized as nnq
        >>> # With cubic kernels and equal stride
        >>> m = nnq.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> m = nnq.ConvTranspose3d(16, 33, (3, 3, 5), stride=(2, 1, 1), padding=(4, 2, 2))
        >>> input = torch.randn(20, 16, 50, 100, 100)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> output = m(q_input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12, 12)
        >>> q_input = torch.quantize_per_tensor(input, scale=1.0, zero_point=0, dtype=torch.quint8)
        >>> downsample = nnq.Conv3d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nnq.ConvTranspose3d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(q_input)
        >>> h.size()
        torch.Size([1, 16, 6, 6, 6])
        >>> # xdoctest: +SKIP("FIXME: output_size is not a parameter)
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12, 12])
    """

    _FLOAT_MODULE: ClassVar[type[nn.ConvTranspose3d]] = nn.ConvTranspose3d

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
    ):
        # [FUNCTION DOC] Normalizes kernel_size, stride, padding, dilation and output_padding to triples and calls
        # the transposed base initializer with transposed set to true.
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedConvTranspose3d".
        return "QuantizedConvTranspose3d"

    def set_weight_bias(self, w: torch.Tensor, b: torch.Tensor | None) -> None:
        # [FUNCTION DOC] Prepacks weight w and optional bias b with the transposed conv3d prepack kernel and stores the packed params.
        self._packed_params = torch.ops.quantized.conv_transpose3d_prepack(
            w,
            b,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )

    def _weight_bias(self):
        # [FUNCTION DOC] HELPER: Returns the (weight, bias) pair unpacked by the conv3d unpack kernel.
        w, b = torch.ops.quantized.conv3d_unpack(self._packed_params)
        return w, b

    def weight(self):
        # [FUNCTION DOC] Returns the unpacked quantized weight tensor.
        (w, _) = self._weight_bias()
        return w

    def bias(self):
        # [FUNCTION DOC] Returns the unpacked bias tensor (or None).
        (_, b) = self._weight_bias()
        return b

    def forward(self, input):
        # [FUNCTION DOC] Runs the quantized transposed 3D convolution with the stored qparams. Raises ValueError if
        # the input is not 5-dimensional.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, T, H, W)`!")
        return ops.quantized.conv_transpose3d(
            input, self._packed_params, self.scale, self.zero_point
        )

    @classmethod
    def from_reference(cls, ref_qconvt, output_scale, output_zero_point):  # type: ignore[override]
        # [FUNCTION DOC] Class method: builds the module from a reference quantized module via the shared
        # transposed-conv helper, with the given output scale and zero point.
        return _ConvTransposeNd.from_reference(
            cls, ref_qconvt, output_scale, output_zero_point
        )
