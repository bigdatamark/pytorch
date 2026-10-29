# mypy: allow-untyped-defs
# [MODULE DOC] Element-wise and softmax-family activation layers as torch.nn Modules.
# Most classes are thin stateful wrappers that store hyperparameters (and an inplace flag) and
# delegate forward to the matching function in torch.nn.functional or torch. Also defines
# MultiheadAttention (learnable projections plus an inference fast path) and PReLU (learnable
# slope). Exports the names listed in __all__. No import-time side effects beyond defining classes.
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter

from .linear import NonDynamicallyQuantizableLinear
from .module import Module


__all__ = [
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]


class Threshold(Module):
    # [CLASS DOC] Module that replaces input elements not greater than a threshold with a fixed value.
    # State: threshold, value, inplace (all listed in __constants__ for TorchScript). No parameters.
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Threshold.png

    Examples::

        >>> m = nn.Threshold(0, 0.5)
        >>> input = torch.arange(-3, 3)
        >>> output = m(input)
    """

    __constants__ = ["threshold", "value", "inplace"]

    threshold: float
    value: float
    inplace: bool

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the threshold, replacement value and inplace flag.
        # Args: threshold (float), value (float), inplace (bool, default False). No validation is done
        # (a TODO comment notes a missing check that value must not exceed threshold when inplace).
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply the threshold function element-wise to the input tensor.
        # Returns a tensor of the same shape; modifies the input in place when inplace is set.
        """
        Runs the forward pass.
        """
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix string with threshold, value and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"threshold={self.threshold}, value={self.value}{inplace_str}"


class ReLU(Module):
    # [CLASS DOC] Module applying the rectified linear unit, max(0, x), element-wise.
    # State: inplace flag only. No parameters.
    r"""Applies the rectified linear unit function element-wise.

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply ReLU to the input via the functional API; returns a same-shaped tensor
        # (the input itself modified when inplace is set).
        """
        Runs the forward pass.
        """
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return "inplace=True" when the inplace flag is set, otherwise an empty string.
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class RReLU(Module):
    # [CLASS DOC] Module for the randomized leaky ReLU activation.
    # Negative inputs are scaled by a slope drawn uniformly from [lower, upper] in training mode and
    # by the fixed midpoint (lower + upper) / 2 in eval mode. State: lower, upper, inplace.
    r"""Applies the randomized leaky rectified linear unit function, element-wise.

    Method described in the paper:
    `Empirical Evaluation of Rectified Activations in Convolutional Network <https://arxiv.org/abs/1505.00853>`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})` during training while during
    evaluation :math:`a` is fixed with :math:`a = \frac{\text{lower} + \text{upper}}{2}`.

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/RReLU.png

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    __constants__ = ["lower", "upper", "inplace"]

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self, lower: float = 1.0 / 8, upper: float = 1.0 / 3, inplace: bool = False
    ) -> None:
        # [FUNCTION DOC] Store the slope bounds and the inplace flag.
        # Args: lower (float, default 1/8), upper (float, default 1/3), inplace (bool, default False).
        # No validation that lower <= upper is performed here.
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply randomized leaky ReLU, passing the module's training flag so the slope is
        # sampled in training mode and fixed in eval mode. Returns a same-shaped tensor.
        """
        Runs the forward pass.
        """
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with lower, upper and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"lower={self.lower}, upper={self.upper}{inplace_str}"


class Hardtanh(Module):
    # [CLASS DOC] Module that clamps every element to the closed range [min_val, max_val].
    # State: min_val, max_val, inplace. Invariant: max_val is strictly greater than min_val.
    # Subclassed by ReLU6.
    r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["min_val", "max_val", "inplace"]

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        # [FUNCTION DOC] Store the clamp bounds and inplace flag.
        # Args: min_val (float, default -1), max_val (float, default 1), inplace (bool),
        # min_value / max_value (deprecated aliases that override min_val / max_val and emit a
        # FutureWarning). Raises AssertionError if max_val is not greater than min_val.
        super().__init__()
        if min_value is not None:
            warnings.warn(
                "keyword argument `min_value` is deprecated and renamed to `min_val`",
                FutureWarning,
                stacklevel=2,
            )
            min_val = min_value
        if max_value is not None:
            warnings.warn(
                "keyword argument `max_value` is deprecated and renamed to `max_val`",
                FutureWarning,
                stacklevel=2,
            )
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        if self.max_val <= self.min_val:
            raise AssertionError(
                f"max_val ({self.max_val}) must be greater than min_val ({self.min_val})"
            )

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Clamp the input to [min_val, max_val] element-wise via the functional API.
        """
        Runs the forward pass.
        """
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with min_val, max_val and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"min_val={self.min_val}, max_val={self.max_val}{inplace_str}"


class ReLU6(Hardtanh):
    # [CLASS DOC] Hardtanh specialization with fixed bounds 0 and 6, i.e. min(max(0, x), 6).
    # Inherits state and forward from Hardtanh; only the inplace flag is configurable.
    r"""Applies the ReLU6 function element-wise.

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Initialize the parent Hardtanh with bounds 0.0 and 6.0. Args: inplace (bool).
        super().__init__(0.0, 6.0, inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return "inplace=True" when inplace is set, else an empty string (the bounds are
        # not shown, unlike the parent class).
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Sigmoid(Module):
    # [CLASS DOC] Module applying the logistic sigmoid, 1 / (1 + exp(-x)), element-wise. Stateless.
    r"""Applies the Sigmoid function element-wise.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the element-wise sigmoid of the input as a same-shaped tensor.
        """
        Runs the forward pass.
        """
        return torch.sigmoid(input)


class Hardsigmoid(Module):
    # [CLASS DOC] Module applying the piecewise-linear Hardsigmoid approximation of the sigmoid:
    # 0 for x <= -3, 1 for x >= 3, x / 6 + 1/2 between. State: inplace flag.
    r"""Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardsigmoid.png

    Examples::

        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply Hardsigmoid to the input via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.hardsigmoid(input, self.inplace)


class Tanh(Module):
    # [CLASS DOC] Module applying the hyperbolic tangent element-wise. Stateless.
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        >>> m(input)
        tensor([-0.9640, -0.4621,  0.0000,  0.4621,  0.9640])
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the element-wise tanh of the input as a same-shaped tensor.
        """
        Runs the forward pass.
        """
        return torch.tanh(input)


class SiLU(Module):
    # [CLASS DOC] Module applying SiLU (swish), x * sigmoid(x), element-wise. State: inplace flag.
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply SiLU to the input via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return "inplace=True" when inplace is set, else an empty string.
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Mish(Module):
    # [CLASS DOC] Module applying Mish, x * tanh(softplus(x)), element-wise. State: inplace flag.
    r"""Applies the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Mish.png

    Examples::

        >>> m = nn.Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply Mish to the input via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return "inplace=True" when inplace is set, else an empty string.
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class Hardswish(Module):
    # [CLASS DOC] Module applying Hardswish (from MobileNetV3): 0 for x <= -3, x for x >= 3, and
    # x * (x + 3) / 6 between. State: inplace flag.
    r"""Applies the Hardswish function, element-wise.

    Method described in the paper: `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardswish.png

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace"]

    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply Hardswish to the input via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.hardswish(input, self.inplace)


class ELU(Module):
    # [CLASS DOC] Module applying the Exponential Linear Unit: x for positive inputs and
    # alpha * (exp(x) - 1) otherwise. State: alpha, inplace.
    r"""Applies the Exponential Linear Unit (ELU) function, element-wise.

    Method described in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
    Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

    ELU is defined as:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ELU.png

    Examples::

        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store alpha and the inplace flag. Args: alpha (float, default 1.0), inplace (bool).
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply ELU with the stored alpha via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with alpha and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{inplace_str}"


class CELU(Module):
    # [CLASS DOC] Module applying CELU: max(0, x) + min(0, alpha * (exp(x / alpha) - 1)), element-wise.
    # State: alpha, inplace.
    r"""Applies the CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/CELU.png

    Examples::

        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """

    __constants__ = ["alpha", "inplace"]
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store alpha and the inplace flag. Args: alpha (float, default 1.0), inplace (bool).
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply CELU with the stored alpha via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with alpha and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"alpha={self.alpha}{inplace_str}"


class SELU(Module):
    # [CLASS DOC] Module applying the self-normalizing SELU activation: a fixed scale times an ELU
    # with a fixed alpha. State: inplace flag only (alpha and scale are constants in the functional).
    r"""Applies the SELU function element-wise.

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    .. warning::
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SELU.png

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the inplace flag. Args: inplace (bool, default False).
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply SELU to the input via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return "inplace=True" when inplace is set, else an empty string.
        """
        Return the extra representation of the module.
        """
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class GLU(Module):
    # [CLASS DOC] Module applying the gated linear unit: the input is split in two halves along dim,
    # and the first half is multiplied by the sigmoid of the second. State: dim.
    r"""Applies the gated linear unit function.

    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.GLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = -1) -> None:
        # [FUNCTION DOC] Store the split dimension. Args: dim (int, default -1).
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply GLU along dim; the output has half the size of the input along that
        # dimension. The functional call raises an error if that size is odd.
        """
        Runs the forward pass.
        """
        return F.glu(input, self.dim)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing dim.
        """
        Return the extra representation of the module.
        """
        return f"dim={self.dim}"


class GELU(Module):
    # [CLASS DOC] Module applying the Gaussian Error Linear Unit, x * Phi(x), element-wise, either
    # exact or with the tanh approximation. State: approximate (string, "none" or "tanh").
    r"""Applies the Gaussian Error Linear Units function.

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["approximate"]
    approximate: str

    def __init__(self, approximate: str = "none") -> None:
        # [FUNCTION DOC] Store the approximation mode. Args: approximate (str, default "none").
        # Validation of the value is deferred to the functional call.
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply GELU using the stored approximation mode; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing the approximate mode as a quoted string.
        """
        Return the extra representation of the module.
        """
        return f"approximate={repr(self.approximate)}"


class Hardshrink(Module):
    # [CLASS DOC] Module applying hard shrinkage: elements with magnitude above lambd are kept and
    # all others are set to zero. State: lambd.
    r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        # [FUNCTION DOC] Store the shrinkage threshold. Args: lambd (float, default 0.5).
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply hard shrinkage via the functional API; same-shaped output.
        """
        Run forward pass.
        """
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix: just the lambd value, without a name.
        """
        Return the extra representation of the module.
        """
        return f"{self.lambd}"


class LeakyReLU(Module):
    # [CLASS DOC] Module applying LeakyReLU: x for non-negative inputs and negative_slope * x
    # otherwise. State: negative_slope, inplace.
    r"""Applies the LeakyReLU function element-wise.

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    .. image:: ../scripts/activation_images/LeakyReLU.png

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        # [FUNCTION DOC] Store the slope and inplace flag.
        # Args: negative_slope (float, default 0.01), inplace (bool, default False).
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply leaky ReLU via the functional API; same-shaped output.
        """
        Run forward pass.
        """
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with negative_slope and, if enabled, inplace=True.
        """
        Return the extra representation of the module.
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope}{inplace_str}"


class LogSigmoid(Module):
    # [CLASS DOC] Module applying log(sigmoid(x)) element-wise in a numerically stable way. Stateless.
    r"""Applies the Logsigmoid function element-wise.

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/LogSigmoid.png

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the element-wise log-sigmoid of the input as a same-shaped tensor.
        """
        Run forward pass.
        """
        return F.logsigmoid(input)


class Softplus(Module):
    # [CLASS DOC] Module applying Softplus, (1 / beta) * log(1 + exp(beta * x)), a smooth ReLU.
    # Reverts to the identity where beta * x exceeds threshold, for numerical stability.
    # State: beta, threshold.
    r"""Applies the Softplus function element-wise.

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softplus.png

    Examples::

        >>> m = nn.Softplus()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["beta", "threshold"]
    beta: float
    threshold: float

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        # [FUNCTION DOC] Store beta and threshold. Args: beta (float, default 1.0), threshold (float, default 20.0).
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply softplus with the stored beta and threshold; same-shaped output.
        """
        Run forward pass.
        """
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix with beta and threshold.
        """
        Return the extra representation of the module.
        """
        return f"beta={self.beta}, threshold={self.threshold}"


class Softshrink(Module):
    # [CLASS DOC] Module applying soft shrinkage: moves elements toward zero by lambd and sets those
    # with magnitude at most lambd to zero. State: lambd (should be non-negative).
    r"""Applies the soft shrinkage function element-wise.

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["lambd"]
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        # [FUNCTION DOC] Store the shrinkage amount. Args: lambd (float, default 0.5). Not validated here.
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply soft shrinkage via the functional API; same-shaped output.
        """
        Run forward pass.
        """
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix: just the lambd value as a string.
        """
        Return the extra representation of the module.
        """
        return str(self.lambd)


def _check_arg_device(x: torch.Tensor | None) -> bool:
# [FUNCTION DOC] HELPER: Report whether an optional tensor lives on a device type that the
# MultiheadAttention fast path supports (cpu, cuda or the registered privateuse1 backend).
# Returns True when the tensor is None.
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True


def _arg_requires_grad(x: torch.Tensor | None) -> bool:
# [FUNCTION DOC] HELPER: Return the requires_grad flag of the tensor, or False when it is None.
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
# [FUNCTION DOC] HELPER: Report whether the code is being traced by make_fx or exported, i.e.
# a proxy-tensor dispatch mode is on the current dispatch mode stack or the compiler is exporting.
# Always False while TorchScript is scripting. Reads global dispatch-mode state, no mutation.
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        # this can be triggered when dynamo inlining the module too.
        return (
            any(
                type(x) is torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
                for x in torch_dispatch_mode_stack
            )
            or torch.compiler.is_exporting()
        )
    else:
        return False


class MultiheadAttention(Module):
    # [CLASS DOC] Multi-head scaled dot-product attention layer (Attention Is All You Need).
    # State: embed_dim, kdim, vdim, num_heads, head_dim, dropout, batch_first, add_zero_attn,
    # parameters in_proj_weight (when q/k/v dims equal embed_dim) or separate q/k/v_proj_weight,
    # optional in_proj_bias, optional bias_k / bias_v, and an out_proj linear layer.
    # Unused parameter slots are registered as None. forward either takes an optimized native
    # inference fast path (self-attention, eval, no grad, batch_first, etc.) or delegates to the
    # functional multi_head_attention_forward.
    r"""Allows the model to jointly attend to information from different representation subspaces.

    This MultiheadAttention layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O

    where :math:`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiheadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ["batch_first"]
    bias_k: torch.Tensor | None
    bias_v: torch.Tensor | None

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        # [FUNCTION DOC] Build the attention layer and initialize all parameters.
        # Args: embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first,
        # device, dtype. kdim / vdim default to embed_dim. Raises ValueError if embed_dim or num_heads
        # is not positive and AssertionError if embed_dim is not divisible by num_heads.
        # Creates the input projection(s), output projection and optional bias_k / bias_v, then calls
        # the private parameter reset.
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize parameters: xavier-uniform for the input projection weight(s),
        # zero for the input and output projection biases when present, xavier-normal for bias_k and
        # bias_v when present. Mutates parameters in place.
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # [FUNCTION DOC] Support unpickling: old checkpoints (from v1.1.0) lack the flag saying q/k/v
        # share one embedding dimension, so it is defaulted to True before delegating to the parent.
        # Mutates the passed state mapping.
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # [FUNCTION DOC] Compute attention outputs for query, key, value tensors.
        # Args: query, key, value, optional key_padding_mask, need_weights (default True), optional
        # attn_mask, average_attn_weights (default True), is_causal (hint only).
        # Returns a pair: attention output and attention weights (None if need_weights is False,
        # except on the native fast path which follows need_weights).
        # Flow: canonicalize masks to the query dtype, work out the first reason (if any) the fast path
        # cannot be used (non-batched, non-self attention, training, grad needed, nested/mask conflicts,
        # tracing, autocast, odd num_heads, ...); if none, merge masks and call the native fused kernel.
        # Otherwise transposes batch-first input, calls the functional multi_head_attention_forward
        # (with separate projection weights when q/k/v dims differ) and transposes the output back.
        # Raises AssertionError if nested tensors are used outside the fast path.
        # KNOWN ISSUE: the fast path is also disabled when num_heads is odd, which the class
        # docstring does not list among the conditions.
        # KNOWN ISSUE: the fast path only returns when both in_proj_bias and in_proj_weight are set;
        # with bias=False the work falls through to the slow path (which rejects nested inputs).
        # KNOWN ISSUE: is_causal is forwarded to the functional call but not used by the fast path.
        r"""Compute attention outputs using query, key, and value embeddings.

            Supports optional parameters for padding, masks and attention weights.

        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and float masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
                and achieve the best performance for MHA.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
                heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
                effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
            is_causal: If specified, applies a causal mask as attention mask.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``attn_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
        why_not_fast_path = ""
        if (
            (attn_mask is not None and torch.is_floating_point(attn_mask))
            or (key_padding_mask is not None)
            and torch.is_floating_point(key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not is_batched:
            why_not_fast_path = (
                f"input not batched; expected query.dim() of 3 but got {query.dim()}"
            )
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (
            key_padding_mask is not None or attn_mask is not None
        ):
            why_not_fast_path = (
                "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
            )
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        fast_path_blocked_by_tracing = False
        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}"
                )
            elif torch.is_grad_enabled() and any(
                _arg_requires_grad(x) for x in tensor_args
            ):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
                fast_path_blocked_by_tracing = True
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query
                )

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type,
                    )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        if any_nested:
            raise AssertionError(
                "MultiheadAttention does not support NestedTensor outside of its fast path. "
                + f"The fast path was not hit because {why_not_fast_path}"
            )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
            if fast_path_blocked_by_tracing:
                # Keep the traced slowpath layout aligned with eager fastpath.
                attn_output = attn_output.contiguous()
            return attn_output, attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        query: Tensor,
    ) -> tuple[Tensor | None, int | None]:
        # [FUNCTION DOC] Combine attention and key-padding masks for the native fast path.
        # Args: attn_mask (2D or 3D or None), key_padding_mask (or None), query (batch, seq, embed).
        # Returns (merged_mask, mask_type): (None, None) if no masks; key padding mask with type 1 if
        # only that is given; otherwise the attention mask is viewed/expanded to 4D (batch, heads, seq,
        # seq), the padding mask (if any) is expanded and added, and type 2 is returned.
        # KNOWN ISSUE: the docstring says masks are combined with logical or and labels attn_mask as
        # type 0, but the code adds them and returns type 2 whenever attn_mask is given.
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: int | None = None
        merged_mask: Tensor | None = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


class PReLU(Module):
    # [CLASS DOC] Module applying PReLU: max(0, x) + a * min(0, x) with a learnable slope a.
    # State: num_parameters, init, and the learnable weight parameter of that length (one shared
    # slope, or one slope per channel where channel is the second input dimension).
    r"""Applies the element-wise PReLU function.

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \ge 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. image:: ../scripts/activation_images/PReLU.png

    Examples::

        >>> m = nn.PReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        # [FUNCTION DOC] Create the learnable slope parameter and fill it with the initial value.
        # Args: num_parameters (int, default 1), init (float, default 0.25), device, dtype.
        # Note num_parameters is assigned before the parent constructor runs.
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.weight = Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [FUNCTION DOC] Reset the weight parameter in place to the constant init value.
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply PReLU with the learnable weight via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing num_parameters.
        """
        Return the extra representation of the module.
        """
        return f"num_parameters={self.num_parameters}"


class Softsign(Module):
    # [CLASS DOC] Module applying Softsign, x / (1 + |x|), element-wise. Stateless.
    r"""Applies the element-wise Softsign function.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softsign.png

    Examples::

        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the element-wise softsign of the input as a same-shaped tensor.
        """
        Runs the forward pass.
        """
        return F.softsign(input)


class Tanhshrink(Module):
    # [CLASS DOC] Module applying Tanhshrink, x - tanh(x), element-wise. Stateless.
    r"""Applies the element-wise Tanhshrink function.

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Return the element-wise tanhshrink of the input as a same-shaped tensor.
        """
        Runs the forward pass.
        """
        return F.tanhshrink(input)


class Softmin(Module):
    # [CLASS DOC] Module applying softmin (softmax of the negated input) along one dimension so the
    # slice sums to 1. State: dim (None means implicit dimension choice with a deprecation warning).
    r"""Applies the Softmin function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    __constants__ = ["dim"]
    dim: int | None

    def __init__(self, dim: int | None = None) -> None:
        # [FUNCTION DOC] Store the dimension. Args: dim (int or None, default None).
        # KNOWN ISSUE: unlike Softmax and LogSoftmax, the type of dim is not validated here.
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        # [FUNCTION DOC] Support unpickling: after parent restore, set dim to None if the attribute is
        # missing (old checkpoints). Mutates the module.
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply softmin along dim via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.softmin(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing dim.
        """
        Return the extra representation of the module.
        """
        return f"dim={self.dim}"


class Softmax(Module):
    # [CLASS DOC] Module applying softmax along one dimension, producing values in [0, 1] that sum to
    # 1 along that dimension. State: dim (None means implicit dimension choice with a warning).
    r"""Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecified
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)

    """

    __constants__ = ["dim"]
    dim: int | None

    def __init__(self, dim: int | None = None) -> None:
        # [FUNCTION DOC] Store the dimension. Args: dim (int or None, default None).
        # Raises TypeError if dim is neither None nor a plain int (bool is rejected).
        if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool)):
            raise TypeError(f"dim must be an int or None, got {type(dim)}")
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        # [FUNCTION DOC] Support unpickling: after parent restore, set dim to None if missing.
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply softmax along dim via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing dim.
        """
        Return the extra representation of the module.
        """
        return f"dim={self.dim}"


class Softmax2d(Module):
    # [CLASS DOC] Module applying softmax over the channel dimension at each spatial location of a
    # 3D (C, H, W) or 4D (N, C, H, W) input. Stateless.
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply softmax over dimension -3 of the input.
        # Raises ValueError if the input is not 3D or 4D.
        """
        Runs the forward pass.
        """
        if input.dim() not in (3, 4):
            raise ValueError(
                f"Softmax2d: expected input to be 3D or 4D, got {input.dim()}D instead"
            )
        return F.softmax(input, -3, _stacklevel=5)


class LogSoftmax(Module):
    # [CLASS DOC] Module applying log(softmax(x)) along one dimension in a numerically stable way.
    # State: dim (None means implicit dimension choice with a warning).
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    __constants__ = ["dim"]
    dim: int | None

    def __init__(self, dim: int | None = None) -> None:
        # [FUNCTION DOC] Store the dimension. Args: dim (int or None, default None).
        # Raises TypeError if dim is neither None nor a plain int (bool is rejected).
        if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool)):
            raise TypeError(f"dim must be an int or None, got {type(dim)}")
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        # [FUNCTION DOC] Support unpickling: after parent restore, set dim to None if missing.
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Apply log-softmax along dim via the functional API; same-shaped output.
        """
        Runs the forward pass.
        """
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing dim.
        """
        Return the extra representation of the module.
        """
        return f"dim={self.dim}"
