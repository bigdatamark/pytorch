# mypy: allow-untyped-defs
# [PACKAGE DOC] Package of quantized nn modules (Conv, Linear, BatchNorm, activations, embeddings, RNN,
# normalization, dropout, functional wrappers). Imports torch.ao.nn.quantizable first to resolve its
# lazy imports, re-exports the module classes listed in __all__, and itself defines Quantize and
# DeQuantize, the modules that convert tensors between float and quantized representations.
import torch

# The quantized modules use `torch.nn` and `torch.ao.nn.quantizable`
# packages. However, the `quantizable` package uses "lazy imports"
# to avoid circular dependency.
# Hence we need to include it here to make sure it is resolved before
# they are used in the modules.
import torch.ao.nn.quantizable
from torch.nn.modules.pooling import MaxPool2d

from .activation import (
    ELU,
    Hardswish,
    LeakyReLU,
    MultiheadAttention,
    PReLU,
    ReLU6,
    Sigmoid,
    Softmax,
)
from .batchnorm import BatchNorm2d, BatchNorm3d
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .dropout import Dropout
from .embedding_ops import Embedding, EmbeddingBag
from .functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from .linear import Linear
from .normalization import (
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
)
from .rnn import LSTM


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]


class Quantize(torch.nn.Module):
    # [CLASS DOC] Module that quantizes a float tensor per-tensor with fixed scale, zero_point and dtype.
    # State: buffers scale (float) and zero_point (long), both one-element tensors, plus a plain dtype
    # attribute. Built directly or from a float module with an observer via from_float.
    r"""Quantizes an incoming tensor

    Args:
     `scale`: scale of the output Quantized Tensor
     `zero_point`: zero_point of output Quantized Tensor
     `dtype`: data type of output Quantized Tensor
     `factory_kwargs`: Dictionary of kwargs used for configuring initialization
         of internal buffers. Currently, `device` and `dtype` are supported.
         Example: `factory_kwargs={'device': 'cuda', 'dtype': torch.float64}`
         will initialize internal buffers as type `torch.float64` on the current CUDA device.
         Note that `dtype` only applies to floating-point buffers.

    Examples::
        >>> t = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> qt = qm(t)
        >>> print(qt)
        tensor([[ 1., -1.],
                [ 1., -1.]], size=(2, 2), dtype=torch.qint8, scale=1.0, zero_point=2)
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, scale, zero_point, dtype, factory_kwargs=None):
        # [FUNCTION DOC] Stores scale and zero_point as one-element buffers (zero_point forced to long dtype,
        # other factory kwargs such as device are shared) and records the target dtype. factory_kwargs is an
        # optional mapping validated by torch.nn.factory_kwargs.
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super().__init__()
        self.register_buffer("scale", torch.tensor([scale], **factory_kwargs))
        self.register_buffer(
            "zero_point",
            torch.tensor(
                [zero_point],
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            ),
        )
        self.dtype = dtype

    def forward(self, X):
        # [FUNCTION DOC] Quantizes tensor X per-tensor using the stored scale, zero_point and dtype; returns
        # the quantized tensor.
        return torch.quantize_per_tensor(
            X, float(self.scale), int(self.zero_point), self.dtype
        )

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: builds a Quantize from a float module mod that was observed. Reads
        # scale and zero_point from the module's activation_post_process observer and uses its dtype.
        # Raises AssertionError if mod has no activation_post_process attribute.
        if not hasattr(mod, "activation_post_process"):
            raise AssertionError(
                f"Module {type(mod).__name__} must have activation_post_process attribute"
            )
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Quantize(
            scale.float().item(),
            zero_point.long().item(),
            mod.activation_post_process.dtype,
        )

    def extra_repr(self):
        # [FUNCTION DOC] HELPER: Returns a string describing scale, zero_point and dtype for the module repr.
        return f"scale={self.scale}, zero_point={self.zero_point}, dtype={self.dtype}"


class DeQuantize(torch.nn.Module):
    # [CLASS DOC] Stateless module that converts a quantized tensor back to a float tensor.
    r"""Dequantizes an incoming tensor

    Examples::
        >>> input = torch.tensor([[1., -1.], [1., -1.]])
        >>> scale, zero_point, dtype = 1.0, 2, torch.qint8
        >>> qm = Quantize(scale, zero_point, dtype)
        >>> # xdoctest: +SKIP
        >>> quantized_input = qm(input)
        >>> dqm = DeQuantize()
        >>> dequantized = dqm(quantized_input)
        >>> print(dequantized)
        tensor([[ 1., -1.],
                [ 1., -1.]], dtype=torch.float32)
    """

    def forward(self, Xq):
        # [FUNCTION DOC] Returns the dequantized (float) version of the quantized input tensor Xq.
        return Xq.dequantize()

    @staticmethod
    def from_float(mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Static method: returns a new DeQuantize; mod and the flag are ignored.
        return DeQuantize()
