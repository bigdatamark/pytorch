# mypy: allow-untyped-defs
# [MODULE DOC] Quantized activation modules: ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax,
# MultiheadAttention and PReLU. Each wraps a quantized op kernel and carries output quantization
# parameters (scale, zero_point); most can be created from an observed float module (from_float) or a
# reference module (from_reference). No import-time side effects beyond imports.
from warnings import warn

import torch


__all__ = [
    "ReLU6",
    "Hardswish",
    "ELU",
    "LeakyReLU",
    "Sigmoid",
    "Softmax",
    "MultiheadAttention",
    "PReLU",
]


class ReLU6(torch.nn.ReLU):
    # [CLASS DOC] Quantized ReLU6 (clamp between the zero point and the quantized value of 6). Stateful only
    # through the inplace flag inherited from ReLU; keeps the input's quantization parameters.
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        # [FUNCTION DOC] Initializes the ReLU base with the inplace flag and stores it as an attribute.
        super().__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized relu6 kernel to the input, honoring the inplace flag; returns the result.
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedReLU6".
        return "QuantizedReLU6"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: builds a ReLU6 using the inplace setting of float module mod. The flag is ignored.
        return ReLU6(mod.inplace)


class Hardswish(torch.nn.Hardswish):
    # [CLASS DOC] Quantized Hardswish. Holds buffers scale and zero_point describing the output quantization.
    r"""This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point, device=None, dtype=None):
        # [FUNCTION DOC] Registers scale and zero_point buffers from the given numbers (created with the
        # requested device and dtype). Args: scale, zero_point, device, dtype.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized hardswish kernel using the stored scale and zero_point; returns a quantized tensor.
        return torch.ops.quantized.hardswish(input, self.scale, self.zero_point)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedHardswish".
        return "QuantizedHardswish"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: reads output qparams from the float module's activation_post_process
        # observer and builds a Hardswish with them (converted to float and int).
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Hardswish(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: builds the module from the given output scale and zero_point (converted to
        # float and int); mod is unused.
        return cls(float(scale), int(zero_point))


class ELU(torch.nn.ELU):
    # [CLASS DOC] Quantized ELU. Stores output scale and zero_point as plain attributes plus the alpha constant
    # inherited from ELU.
    r"""This is the quantized equivalent of :class:`~torch.nn.ELU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """

    def __init__(self, scale, zero_point, alpha=1.0):
        # [FUNCTION DOC] Initializes the ELU base with alpha and stores the output scale and zero_point.
        super().__init__(alpha)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized functional elu with the stored scale, zero_point and alpha; returns a quantized tensor.
        return torch.ao.nn.quantized.functional.elu(
            input, self.scale, self.zero_point, self.alpha
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedELU".
        return "QuantizedELU"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: takes output qparams from the observer of float module mod and builds an ELU with mod's alpha.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return ELU(float(scale), int(zero_point), mod.alpha)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: builds an ELU from the given scale, zero_point and alpha of the reference module mod.
        return cls(float(scale), int(zero_point), mod.alpha)


class LeakyReLU(torch.nn.LeakyReLU):
    # [CLASS DOC] Quantized LeakyReLU. Holds scale and zero_point buffers (output quantization) plus
    # negative_slope and inplace inherited from LeakyReLU.
    r"""This is the quantized equivalent of :class:`~torch.nn.LeakyReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
    """

    def __init__(
        self,
        scale: float,
        zero_point: int,
        negative_slope: float = 1e-2,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the LeakyReLU base with negative_slope and inplace, and registers scale and
        # zero_point buffers created with the requested device and dtype.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(negative_slope, inplace)
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized leaky_relu kernel with the module's slope, inplace flag, scale and zero_point.
        return torch.ops.quantized.leaky_relu(
            input, self.negative_slope, self.inplace, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedLeakyReLU".
        return "QuantizedLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: gets output qparams from the observer on float module mod and builds the
        # module with mod's negative_slope and inplace.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: builds the module from scale, zero_point and the slope/inplace of reference module mod.
        return cls(float(scale), int(zero_point), mod.negative_slope, mod.inplace)


class Sigmoid(torch.nn.Sigmoid):
    # [CLASS DOC] Quantized Sigmoid. Stores output_scale and output_zero_point as plain attributes.
    r"""This is the quantized equivalent of :class:`~torch.nn.Sigmoid`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, output_scale: float, output_zero_point: int):
        # [FUNCTION DOC] Stores the output scale and zero point; no other state.
        super().__init__()
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized sigmoid kernel with the stored output scale and zero point.
        return torch.ops.quantized.sigmoid(
            input, self.output_scale, self.output_zero_point
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: reads output qparams from the observer of float module mod and builds the
        # module with them. There is no from_reference counterpart.
        (
            output_scale,
            output_zero_point,
        ) = mod.activation_post_process.calculate_qparams()
        return cls(float(output_scale), int(output_zero_point))


class Softmax(torch.nn.Softmax):
    # [CLASS DOC] Quantized Softmax. Stores dim (may be None), and output scale and zero_point as plain attributes.
    r"""This is the quantized version of :class:`~torch.nn.Softmax`.

    Args:
        dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, dim=None, scale=1.0, zero_point=0):
        # [FUNCTION DOC] Initializes the Softmax base and stores dim, scale and zero_point (defaults: no dim, 1.0, 0).
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized softmax kernel. When dim is None, an implicit dimension is chosen from
        # the input's rank using the private helper in torch.nn.functional (which may emit a deprecation warning).
        # Returns a quantized tensor.
        dim = self.dim
        if dim is None:
            stacklevel = 3
            # Note: adding the mypy ignore on _get_softmax_dim seems less bad
            # than making `_get_softmax_dim` an official API.
            dim = torch.nn.functional._get_softmax_dim(  # type: ignore[attr-defined]
                "softmax", input.dim(), stacklevel
            )
        return torch.ops.quantized.softmax(input, dim, self.scale, self.zero_point)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedSoftmax".
        return "QuantizedSoftmax"

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: builds a Softmax with float module mod's dim and output qparams from its observer.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Softmax(mod.dim, float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: builds a Softmax with reference module mod's dim and the given scale and zero_point.
        return cls(mod.dim, float(scale), int(zero_point))


class MultiheadAttention(torch.ao.nn.quantizable.MultiheadAttention):
    # [CLASS DOC] Quantized MultiheadAttention, a subclass of the quantizable (observed) MultiheadAttention.
    # It only converts an observed module in place of class into this class; _FLOAT_MODULE names the
    # observed source type. Direct conversion from a plain float module is not supported.
    _FLOAT_MODULE = torch.ao.nn.quantizable.MultiheadAttention

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedMultiheadAttention".
        return "QuantizedMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        # [FUNCTION DOC] Class method: always raises NotImplementedError, because conversion must go from float to
        # observed to quantized and this class handles only the last step.
        # The whole flow is float -> observed -> quantized
        # This class does observed -> quantized only
        raise NotImplementedError(
            "It looks like you are trying to convert a "
            "non-observed MHA module. Please, see "
            "the examples on quantizable MHAs."
        )

    @classmethod
    def from_observed(cls, other):
        # [FUNCTION DOC] Class method: converts an observed MHA module to quantized form. Runs the generic quantization
        # convert step (not in place, qconfig removed), then switches the converted object's class to this class.
        # If bias_k / bias_v exist they are removed from the parameters and stored as quint8 per-tensor quantized
        # tensors; in_proj_weight and in_proj_bias attributes are deleted. Returns the converted module.
        # KNOWN ISSUE: the bias_v quantization parameters are computed from bias_k (already quantized), not from
        # bias_v, and fail if bias_v exists while bias_k is None.
        converted = torch.ao.quantization.convert(
            other,
            mapping=None,
            inplace=False,
            remove_qconfig=True,
            convert_custom_config_dict=None,
        )
        converted.__class__ = cls
        # Remove the parameters for the bias_k and bias_v to quantize them
        # TODO: This is a potential source of accuracy drop.
        #       quantized cat takes the scale and zp of the first
        #       element, which might lose the precision in the bias_k
        #       and the bias_v (which are cat'ed with k/v being first).
        if converted.bias_k is not None:
            bias_k = converted._parameters.pop("bias_k")
            sc, zp = torch._choose_qparams_per_tensor(bias_k, reduce_range=False)
            bias_k = torch.quantize_per_tensor(bias_k, sc, zp, torch.quint8)
            setattr(converted, "bias_k", bias_k)  # noqa: B010

        if converted.bias_v is not None:
            bias_v = converted._parameters.pop("bias_v")
            sc, zp = torch._choose_qparams_per_tensor(
                bias_k,  # type: ignore[possibly-undefined]
                reduce_range=False,
            )
            bias_v = torch.quantize_per_tensor(bias_v, sc, zp, torch.quint8)
            setattr(converted, "bias_v", bias_v)  # noqa: B010

        del converted.in_proj_weight
        del converted.in_proj_bias

        return converted


class PReLU(torch.nn.Module):
    # [CLASS DOC] Quantized PReLU. State: num_parameters, output scale and zero_point (plain attributes) and a
    # quantized weight tensor set through set_weight (initially a random tensor quantized as quint8 with
    # scale 1 and zero point 0).
    r"""This is the quantized equivalent of :class:`~torch.nn.PReLU`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        num_parameters: number of parameters: 1, or the number of channels at input. Default: 1
    """

    def __init__(
        self, output_scale: float, output_zero_point: int, num_parameters: int = 1
    ) -> None:
        # [FUNCTION DOC] Stores output qparams and num_parameters, then creates a random float weight quantized to
        # quint8 (scale 1.0, zero point 0) as a placeholder weight. Args: output_scale, output_zero_point, num_parameters.
        super().__init__()
        self.num_parameters = num_parameters
        self.scale = output_scale
        self.zero_point = output_zero_point
        w = torch.randn(num_parameters, dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.quint8)
        self.set_weight(qw)

    def set_weight(self, w: torch.Tensor) -> None:
        # [FUNCTION DOC] Replaces the stored weight with the given already-quantized tensor w.
        self.weight = w

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies the quantized prelu kernel with the stored weight, scale and zero_point; returns a quantized tensor.
        return torch.ops.quantized.prelu(
            input, self.weight, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedPReLU".
        return "QuantizedPReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Class method: builds the module using output qparams from the observer on float module mod,
        # then quantizes mod's weight to quint8 with the qconfig's weight observer (qparams from that observer)
        # and stores it. Emits a warning if the weight observer dtype is not quint8. Returns the new module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}",
                stacklevel=2,
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Class method: same weight handling as from_float (weight observer from the module's qconfig,
        # quint8 quantization, warning on other observer dtypes) but takes the output scale and zero_point as arguments.
        qprelu = cls(float(scale), int(zero_point), mod.num_parameters)
        float_wt = mod.weight.float()
        observer = mod.qconfig.weight()
        observer(float_wt)
        if observer.dtype != torch.quint8:
            warn(
                f"PReLU's weight observer should have dtype quint8 but got {observer.dtype}",
                stacklevel=2,
            )
        wt_scale, wt_zp = observer.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            float_wt, float(wt_scale), int(wt_zp), torch.quint8
        )
        qprelu.set_weight(qweight)
        return qprelu
