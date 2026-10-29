# mypy: allow-untyped-defs
# [MODULE DOC]
# Linear-family layers: Identity (pass-through), Linear (affine map), a private
# NonDynamicallyQuantizableLinear subclass (scripting workaround), Bilinear (two-input
# bilinear map) and LazyLinear (Linear whose in_features is inferred on first call).
# Exports Bilinear, Identity, LazyLinear and Linear via __all__. No import-time side effects.
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedParameter

from .lazy import LazyModuleMixin
from .module import Module


__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]


class Identity(Module):
    # [CLASS DOC]
    # Placeholder module with no parameters that returns its input unchanged and ignores
    # any constructor arguments. Useful to stand in for an optional layer.
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # Accepts and discards any positional and keyword arguments; only initializes the
        # base Module.
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Returns the very same input tensor object, unmodified.
        """
        Runs the forward pass.
        """
        return input


class Linear(Module):
    # [CLASS DOC]
    # Fully connected layer computing input times transposed weight plus optional bias,
    # applied over the last dimension of an input with any number of leading dimensions.
    # State: integer in_features and out_features (in __constants__), a weight parameter of
    # shape out_features by in_features, and a bias parameter of length out_features or
    # None when bias is disabled. Key methods: reset_parameters, forward, extra_repr.
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores in_features and out_features, allocates an uninitialized weight parameter
        # (out_features by in_features) and, if bias is true, a bias parameter; otherwise
        # registers bias as None. Device and dtype apply to new parameters. Finishes by
        # calling reset_parameters, which fills them with random values.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [FUNCTION DOC]
        # Re-initializes weight with Kaiming uniform (equivalent to uniform within plus or
        # minus one over the square root of in_features) and, if present, bias uniformly within
        # plus or minus one over the square root of fan-in (zero bound when fan-in is zero).
        # Mutates parameters in place; no return value.
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the affine transform of input using the stored weight and bias; the last
        # input dimension must equal in_features and becomes out_features. Shape errors are
        # raised by the functional op.
        """
        Runs the forward pass.
        """
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Returns a string with in_features, out_features and whether a bias exists.
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# This class exists solely to avoid triggering an obscure error when scripting
# an improperly quantized attention layer. See this issue for details:
# https://github.com/pytorch/pytorch/issues/58969
# TODO: fail fast on quantization API usage error, then remove this class
# and replace uses of it with plain Linear
class NonDynamicallyQuantizableLinear(Linear):
    # [CLASS DOC] HELPER: Plain Linear subclass with identical behavior, existing only so
    # that dynamic quantization passes do not match it by exact type (avoids a scripting
    # error on improperly quantized attention layers). Not exported in __all__.
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Forwards all arguments unchanged to the Linear initializer.
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )


class Bilinear(Module):
    # [CLASS DOC]
    # Layer combining two inputs through a bilinear form per output feature, plus optional bias.
    # State: in1_features, in2_features, out_features (in __constants__), a weight parameter
    # of shape out_features by in1_features by in2_features, and a bias parameter of length
    # out_features or None. Key methods: reset_parameters, forward (two inputs), extra_repr.
    r"""Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.

    Args:
        in1_features: size of each first input sample, must be > 0
        in2_features: size of each second input sample, must be > 0
        out_features: size of each output sample, must be > 0
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_\text{in1})` where :math:`H_\text{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_\text{in2})` where :math:`H_\text{in2}=\text{in2\_features}`.
        - Output: :math:`(*, H_\text{out})` where :math:`H_\text{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """

    __constants__ = ["in1_features", "in2_features", "out_features"]
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the three feature sizes, allocates an uninitialized weight parameter and an
        # optional bias parameter (registered as None when bias is false), then calls
        # reset_parameters. Raises ValueError (from reset_parameters) if in1_features is not positive.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in1_features, in2_features), **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [FUNCTION DOC]
        # Raises ValueError if in1_features is not positive. Otherwise draws weight (and bias
        # if present) uniformly within plus or minus one over the square root of in1_features
        # (taken from the second weight dimension). Mutates parameters in place.
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if self.in1_features <= 0:
            raise ValueError(
                f"in1_features must be > 0, but got (in1_features={self.in1_features})"
            )
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the bilinear transform of the two inputs using weight and bias; leading
        # dimensions must match and last dimensions must equal in1_features and in2_features.
        # Errors come from the functional op.
        """
        Runs the forward pass.
        """
        return F.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Returns a string with the three feature sizes and whether a bias exists.
        """
        Return the extra representation of the module.
        """
        return (
            f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class LazyLinear(LazyModuleMixin, Linear):
    # [CLASS DOC]
    # Linear layer whose in_features is inferred from the last dimension of the first input.
    # Until then weight (and bias, if enabled) are uninitialized parameters and in_features is 0.
    # After the first forward the instance becomes a regular Linear (cls_to_become).
    # Key methods: initialize_parameters (materializes and initializes), reset_parameters
    # (no-op while uninitialized).
    r"""A :class:`torch.nn.Linear` module where `in_features` is inferred.

    In this module, the `weight` and `bias` are of :class:`torch.nn.UninitializedParameter`
    class. They will be initialized after the first call to ``forward`` is done and the
    module will become a regular :class:`torch.nn.Linear` module. The ``in_features`` argument
    of the :class:`Linear` is inferred from the ``input.shape[-1]``.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`


    """

    cls_to_become = Linear  # type: ignore[assignment]
    # pyrefly: ignore [bad-override]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        # [FUNCTION DOC]
        # Builds a zero-sized Linear without bias, then replaces weight with an uninitialized
        # parameter and, if bias is true, adds an uninitialized bias parameter (otherwise bias
        # stays None). Sets out_features; in_features stays 0 until the first forward.
        factory_kwargs = {"device": device, "dtype": dtype}
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        # pyrefly: ignore [bad-argument-type]
        super().__init__(0, 0, False)
        # pyrefly: ignore [unexpected-keyword]
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            # pyrefly: ignore [unexpected-keyword]
            self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        # [FUNCTION DOC]
        # Re-initializes via the Linear implementation only when no parameter is
        # uninitialized and in_features is nonzero; otherwise does nothing.
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # pyrefly: ignore [bad-argument-type]
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        # [FUNCTION DOC]
        # If parameters are uninitialized: sets in_features from the input's last dimension,
        # materializes weight (and bias) with the resulting shapes and initializes them, all
        # without gradient tracking. Then, if in_features is still 0 (parameters were loaded
        # earlier from a checkpoint), checks the input's last dimension against the weight's
        # and raises AssertionError on mismatch, otherwise records in_features.
        # Side effects: mutates parameters and in_features.
        """
        Infers ``in_features`` based on ``input`` and initializes parameters.
        """
        # pyrefly: ignore [bad-argument-type]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
        if self.in_features == 0:
            if input.shape[-1] != self.weight.shape[-1]:
                raise AssertionError(
                    f"The in_features inferred from input: {input.shape[-1]} "
                    f"is not equal to in_features from self.weight: "
                    f"{self.weight.shape[-1]}"
                )
            self.in_features = input.shape[-1]


# TODO: PartialLinear - maybe in sparse?
