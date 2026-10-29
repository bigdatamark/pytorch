# mypy: allow-untyped-defs
# [MODULE DOC] Quantized normalization modules: LayerNorm, GroupNorm,
# InstanceNorm1d, InstanceNorm2d and InstanceNorm3d. Each subclasses the float
# module, stores weight and bias as given plus output scale and zero_point
# buffers, and forwards through the corresponding quantized op. Can be built
# from observed float modules (from_float) or reference modules (from_reference,
# except GroupNorm). No import-time side effects.
import torch


__all__ = [
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]


class LayerNorm(torch.nn.LayerNorm):
    # [CLASS DOC] Quantized version of the float LayerNorm. State: inherited
    # float module attributes, weight and bias (assigned directly from the
    # arguments, not copied), and buffers scale and zero_point for the output
    # quantization. Forward calls the quantized.layer_norm op.
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        normalized_shape,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float base class (normalized_shape and
        # eps/elementwise_affine) with device/dtype factory kwargs, then sets
        # weight and bias to the given tensors and registers scale and
        # zero_point as buffers created from the given numbers.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            **factory_kwargs,
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized layer norm op to the quantized
        # input using the stored weight, bias, eps and output scale/zero point;
        # returns the quantized output.
        return torch.ops.quantized.layer_norm(
            input,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            output_scale=self.scale,
            output_zero_point=self.zero_point,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedLayerNorm.
        return "QuantizedLayerNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a quantized module from an observed float
        # module: reads scale and zero point from its activation_post_process
        # observer, and reuses the float module's weight, bias and
        # configuration. use_precomputed_fake_quant is unused. Returns the new
        # module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Builds a quantized module from a reference module,
        # reusing its weight, bias and configuration, with the given scale (as
        # float) and zero point (as int). Returns the new module.
        return cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )


class GroupNorm(torch.nn.GroupNorm):
    # [CLASS DOC] Quantized version of the float GroupNorm. State: inherited
    # float module attributes, weight and bias (assigned directly from the
    # arguments, not copied), and buffers scale and zero_point for the output
    # quantization. Forward calls the quantized.group_norm op.
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]

    def __init__(
        self,
        num_groups,
        num_channels,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float base class (group and channel
        # counts, eps and affine) with device/dtype factory kwargs, then sets
        # weight and bias to the given tensors and registers scale and
        # zero_point as buffers created from the given numbers.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_groups, num_channels, eps, affine, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized group norm op to the quantized
        # input using the stored weight, bias, eps and output scale/zero point;
        # returns the quantized output.
        return torch.ops.quantized.group_norm(
            input,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name QuantizedGroupNorm.
        return "QuantizedGroupNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a quantized module from an observed float
        # module: reads scale and zero point from its activation_post_process
        # observer, and reuses the float module's weight, bias and
        # configuration. use_precomputed_fake_quant is unused. Returns the new
        # module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_groups,
            mod.num_channels,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod


class InstanceNorm1d(torch.nn.InstanceNorm1d):
    # [CLASS DOC] Quantized version of the float InstanceNorm1d. State:
    # inherited float module attributes, weight and bias (assigned directly from
    # the arguments, not copied), and buffers scale and zero_point for the
    # output quantization. Forward calls the quantized.instance_norm op.
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float base class (num_features, eps,
        # momentum, affine and track_running_stats) with device/dtype factory
        # kwargs, then sets weight and bias to the given tensors and registers
        # scale and zero_point as buffers created from the given numbers.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized instance norm op to the quantized
        # input using the stored weight, bias, eps and output scale/zero point;
        # returns the quantized output.
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedInstanceNorm1d.
        return "QuantizedInstanceNorm1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a quantized module from an observed float
        # module: reads scale and zero point from its activation_post_process
        # observer, and reuses the float module's weight, bias and
        # configuration. use_precomputed_fake_quant is unused. Returns the new
        # module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Builds a quantized module from a reference module,
        # reusing its weight, bias and configuration, with the given scale (as
        # float) and zero point (as int). Returns the new module.
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm2d(torch.nn.InstanceNorm2d):
    # [CLASS DOC] Quantized version of the float InstanceNorm2d. State:
    # inherited float module attributes, weight and bias (assigned directly from
    # the arguments, not copied), and buffers scale and zero_point for the
    # output quantization. Forward calls the quantized.instance_norm op.
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float base class (num_features, eps,
        # momentum, affine and track_running_stats) with device/dtype factory
        # kwargs, then sets weight and bias to the given tensors and registers
        # scale and zero_point as buffers created from the given numbers.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized instance norm op to the quantized
        # input using the stored weight, bias, eps and output scale/zero point;
        # returns the quantized output.
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedInstanceNorm2d.
        return "QuantizedInstanceNorm2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a quantized module from an observed float
        # module: reads scale and zero point from its activation_post_process
        # observer, and reuses the float module's weight, bias and
        # configuration. use_precomputed_fake_quant is unused. Returns the new
        # module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Builds a quantized module from a reference module,
        # reusing its weight, bias and configuration, with the given scale (as
        # float) and zero point (as int). Returns the new module.
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm3d(torch.nn.InstanceNorm3d):
    # [CLASS DOC] Quantized version of the float InstanceNorm3d. State:
    # inherited float module attributes, weight and bias (assigned directly from
    # the arguments, not copied), and buffers scale and zero_point for the
    # output quantization. Forward calls the quantized.instance_norm op.
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Initializes the float base class (num_features, eps,
        # momentum, affine and track_running_stats) with device/dtype factory
        # kwargs, then sets weight and bias to the given tensors and registers
        # scale and zero_point as buffers created from the given numbers.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        # [FUNCTION DOC] Applies the quantized instance norm op to the quantized
        # input using the stored weight, bias, eps and output scale/zero point;
        # returns the quantized output.
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        # [FUNCTION DOC] HELPER: returns the display name
        # QuantizedInstanceNorm3d.
        return "QuantizedInstanceNorm3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Builds a quantized module from an observed float
        # module: reads scale and zero point from its activation_post_process
        # observer, and reuses the float module's weight, bias and
        # configuration. use_precomputed_fake_quant is unused. Returns the new
        # module.
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        # [FUNCTION DOC] Builds a quantized module from a reference module,
        # reusing its weight, bias and configuration, with the given scale (as
        # float) and zero point (as int). Returns the new module.
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
