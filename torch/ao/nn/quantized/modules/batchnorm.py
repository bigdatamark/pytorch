# mypy: allow-untyped-defs
# [MODULE DOC] Quantized BatchNorm2d and BatchNorm3d modules, which run batch normalization on
# quantized tensors via the quantized batch_norm kernels using stored output scale and zero_point.
# Includes a shared private base class and conversion from float (observed) or reference modules.
import torch
import torch.ao.nn.intrinsic as nni


__all__ = ["BatchNorm2d", "BatchNorm3d"]


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    # [CLASS DOC] HELPER: Shared base for the quantized batch norm modules, derived from the float
    # _BatchNorm. Adds buffers scale (output scale) and zero_point (output zero point); weight, bias and
    # running statistics are kept as in the float module. Subclasses supply _NNI_BN_RELU_MODULE.
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        # [FUNCTION DOC] Initializes the float base with affine and running stats enabled, then registers
        # scale (initial 1.0) and zero_point (initial 0) buffers. Args: num_features, eps, momentum, device, dtype.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        self.register_buffer("scale", torch.tensor(1.0, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] HELPER: Static helper (takes the concrete class explicitly) that converts an observed
        # float module. Unwraps a fused BN+ReLU module when mod is of the class's _NNI_BN_RELU_MODULE type,
        # computes scale and zero_point from the activation_post_process observer, and creates the new module
        # sharing weight, bias and running statistics with the float one. Returns the new module.
        activation_post_process = mod.activation_post_process
        if type(mod) is cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        # [FUNCTION DOC] Class method: builds the quantized module from a reference batch norm bn, reusing its
        # num_features, eps, momentum, device, dtype, weight, bias and running statistics, and setting the
        # given output_scale and output_zero_point. Returns the new module.
        qbn = cls(
            bn.num_features,
            bn.eps,
            bn.momentum,
            device=bn.weight.device,
            dtype=bn.weight.dtype,
        )
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn


class BatchNorm2d(_BatchNorm):
    # [CLASS DOC] Quantized 2D batch norm; expects 4D input (N, C, H, W) conceptually. Uses BNReLU2d as the
    # fused float type it can convert from.
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm2d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU2d

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None
    ) -> None:
        # [FUNCTION DOC] Forwards the arguments to the base class initializer.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedBatchNorm2d".
        return "QuantizedBatchNorm2d"

    def _check_input_dim(self, input):
        # [FUNCTION DOC] HELPER: Raises ValueError if the input is not 4-dimensional. Not called from forward
        # (disabled because it is not symbolically traceable).
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies the quantized 2D batch norm kernel to the input with the stored weight, bias,
        # running statistics, eps, scale and zero_point; returns the quantized output tensor.
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm2d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts an observed float module via the shared base conversion helper.
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class BatchNorm3d(_BatchNorm):
    # [CLASS DOC] Quantized 3D batch norm; expects 5D input. Uses BNReLU3d as the fused float type it can
    # convert from.
    r"""This is the quantized version of :class:`~torch.nn.BatchNorm3d`."""

    _NNI_BN_RELU_MODULE = nni.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        # [FUNCTION DOC] Forwards the arguments to the base class initializer.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_features, eps, momentum, **factory_kwargs)

    def _get_name(self):
        # [FUNCTION DOC] HELPER: Returns the display name "QuantizedBatchNorm3d".
        return "QuantizedBatchNorm3d"

    def _check_input_dim(self, input):
        # [FUNCTION DOC] HELPER: Raises ValueError if the input is not 5-dimensional. Not called from forward.
        # KNOWN ISSUE: the error message mentions a 4D shape (N, C, H, W) although 5D input is required.
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, H, W)`!")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies the quantized 3D batch norm kernel to the input with the stored weight, bias,
        # running statistics, eps, scale and zero_point; returns the quantized output tensor.
        # disabling this since this is not symbolically traceable
        # self._check_input_dim(input)
        return torch.ops.quantized.batch_norm3d(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        # [FUNCTION DOC] Class method: converts an observed float module via the shared base conversion helper.
        return _BatchNorm.from_float(
            cls, mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )
