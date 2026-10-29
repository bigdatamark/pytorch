# mypy: allow-untyped-defs
# [MODULE DOC] Quantized fused linear modules: LinearReLU, LinearLeakyReLU (onednn only)
# and LinearTanh. No import-time side effects.
import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight


__all__ = [
    "LinearReLU",
    "LinearLeakyReLU",
    "LinearTanh",
]


class LinearReLU(nnq.Linear):
    # [CLASS DOC] Quantized fused Linear+ReLU module; subclass of the quantized Linear
    # with identical state. _FLOAT_MODULE is the float fused LinearReLU.
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        # [FUNCTION DOC] Forwards sizes, bias flag and weight dtype to the quantized
        # Linear constructor.
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Calls the quantized linear_relu op with the packed params,
        # scale and zero_point. Returns the quantized output.
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedLinearReLU.
        return "QuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Classmethod: converts a float fused module via the parent
        # from_float.
        return super().from_float(mod, use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        # [FUNCTION DOC] Classmethod: builds the module from a reference fused module,
        # using its first element (reference linear) with the given output scale and
        # zero point.
        return super().from_reference(
            ref_linear_relu[0], output_scale, output_zero_point
        )


class LinearLeakyReLU(nnq.Linear):
    # [CLASS DOC] Quantized fused Linear+LeakyReLU module (onednn backend only). Extends
    # the quantized Linear with a negative_slope attribute used by the quantized op.
    r"""
    For onednn backend only
    A LinearLeakyReLU module fused from Linear and LeakyReLU modules
    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.
    Attributes:
        Same as torch.ao.nn.quantized.Linear
        + negative_slope
    Examples::
        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearLeakyReLU(20, 30, 0.01)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearLeakyReLU  # type: ignore[assignment]

    def __init__(
        self, in_features, out_features, negative_slope, bias=True, dtype=torch.qint8
    ):
        # [FUNCTION DOC] Initializes the base Linear with sizes, bias and dtype and
        # stores negative_slope.
        super().__init__(in_features, out_features, bias, dtype)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Calls the quantized linear_leaky_relu op with packed params,
        # scale, zero_point and negative_slope. Returns the quantized output.
        return torch.ops.quantized.linear_leaky_relu(
            x,
            self._packed_params._packed_params,
            self.scale,
            self.zero_point,
            self.negative_slope,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedLinearLeakyReLU.
        return "QuantizedLinearLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Classmethod: converts a float fused LinearLeakyReLU (must have
        # a qconfig and activation observer) to the quantized module. Runs the weight
        # observer, computes activation qparams, quantizes the weight, copies the
        # negative slope, and sets scale and zero_point. Raises AssertionError if the
        # type is wrong, qconfig is missing, or the weight dtype is not qint8.
        # use_precomputed_fake_quant is unused.
        if type(mod) is not nni.LinearLeakyReLU:
            raise AssertionError("Input float module should be LinearLeakyReLU")
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        leaky_relu = mod[1]
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_leaky_relu = cls(
            mod.in_features, mod.out_features, leaky_relu.negative_slope, dtype=dtype
        )
        qlinear_leaky_relu.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_leaky_relu.scale = float(act_scale)
        qlinear_leaky_relu.zero_point = int(act_zp)
        return qlinear_leaky_relu

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        # [FUNCTION DOC] Classmethod: builds the module from a reference fused module
        # (linear, leaky relu): copies the quantized weight, bias and negative slope,
        # and sets scale and zero_point from the given output qparams.
        linear = ref_mod[0]
        leaky_relu = ref_mod[1]
        qlinear_leaky_relu = cls(
            linear.in_features, linear.out_features, leaky_relu.negative_slope
        )
        qweight = linear.get_quantized_weight()
        qlinear_leaky_relu.set_weight_bias(qweight, linear.bias)
        qlinear_leaky_relu.scale = float(output_scale)
        qlinear_leaky_relu.zero_point = int(output_zero_point)
        return qlinear_leaky_relu


class LinearTanh(nnq.Linear):
    # [CLASS DOC] Quantized fused Linear+Tanh module; subclass of the quantized Linear
    # with identical state.
    r"""
    A LinearTanh module fused from Linear and Tanh modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearTanh(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearTanh  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        # [FUNCTION DOC] Forwards sizes, bias flag and weight dtype to the quantized
        # Linear constructor.
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Calls the quantized linear_tanh op with packed params, scale
        # and zero_point. Returns the quantized output.
        return torch.ops.quantized.linear_tanh(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedLinearTanh.
        return "QuantizedLinearTanh"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Classmethod: converts a float fused LinearTanh (needs qconfig
        # and activation observer) to the quantized module: observes and quantizes the
        # weight, computes activation qparams, sets scale and zero_point. Raises
        # AssertionError if the type is wrong, qconfig is missing, or the weight dtype
        # is not qint8. use_precomputed_fake_quant is unused.
        if type(mod) is not nni.LinearTanh:
            raise AssertionError("Input float module should be LinearTanh")
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr,operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_tanh = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear_tanh.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_tanh.scale = float(act_scale)
        qlinear_tanh.zero_point = int(act_zp)
        return qlinear_tanh

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        # [FUNCTION DOC] Classmethod: builds the module from a reference fused module
        # (first element is the reference linear): copies the quantized weight and bias
        # and sets scale and zero_point from the output qparams.
        linear = ref_mod[0]
        qlinear_tanh = cls(linear.in_features, linear.out_features)
        qweight = linear.get_quantized_weight()
        qlinear_tanh.set_weight_bias(qweight, linear.bias)
        qlinear_tanh.scale = float(output_scale)
        qlinear_tanh.zero_point = int(output_zero_point)
        return qlinear_tanh
