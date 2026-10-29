# [MODULE DOC]
# Dropout regularization modules for torch.nn: Dropout, Dropout1d/2d/3d (channel-wise),
# AlphaDropout and FeatureAlphaDropout (SELU-preserving variants), all built on the shared
# private base _DropoutNd which validates and stores the probability. Each forward delegates
# to the matching torch.nn.functional function using the module training flag.
# No import-time side effects beyond imports.
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = [
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
]


class _DropoutNd(Module):
    # [CLASS DOC] HELPER: Shared base for dropout modules.
    # State: p (drop probability, validated to lie in the closed range zero to one) and inplace flag.
    # Subclasses implement forward by calling a functional dropout; extra_repr reports p and inplace.
    __constants__ = ["p", "inplace"]
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Validates that p is within zero and one inclusive and stores p and inplace.
        # Raises ValueError with a message naming the offending value when p is out of range.
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a string with the p and inplace values for the module repr.
        return f"p={self.p}, inplace={self.inplace}"


class Dropout(_DropoutNd):
    # [CLASS DOC]
    # Element-wise dropout: during training zeroes each element independently with probability p
    # and scales survivors by one over one minus p; in eval mode it is the identity.
    # State inherited from _DropoutNd (p, inplace). Accepts tensors of any shape.
    r"""During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p`.

    The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.

    Each channel will be zeroed out independently on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional dropout to the input using p, the module training flag and inplace.
        # Returns a tensor of the same shape (the input itself when inplace). Random mask drawn from the global RNG.
        """
        Runs the forward pass.
        """
        return F.dropout(input, self.p, self.training, self.inplace)


class Dropout1d(_DropoutNd):
    # [CLASS DOC]
    # Channel-wise dropout for one-dimensional feature maps: each whole channel is zeroed
    # independently with probability p during training. Expects (N, C, L) or (C, L) input.
    # State inherited from _DropoutNd (p, inplace).
    r"""Randomly zero out entire channels.

    A channel is a 1D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 1D tensor :math:`\text{input}[i, j]`.

    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv1d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout1d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, L)` or :math:`(C, L)`.
        - Output: :math:`(N, C, L)` or :math:`(C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout1d(p=0.2)
        >>> input = torch.randn(20, 16, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional 1D channel dropout using p, the training flag and inplace.
        # Returns a tensor of the same shape. Random mask drawn from the global RNG.
        """
        Runs the forward pass.
        """
        return F.dropout1d(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    # [CLASS DOC]
    # Channel-wise dropout for two-dimensional feature maps: each whole channel is zeroed
    # independently with probability p during training. Expects (N, C, H, W); a three-dimensional
    # input is treated as batched 1D channels for historical reasons, per the docstring.
    # State inherited from _DropoutNd (p, inplace).
    r"""Randomly zero out entire channels.

    A channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`.

    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout2d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    .. warning ::
        Due to historical reasons, this class will perform 1D channel-wise dropout
        for 3D inputs (as done by :class:`nn.Dropout1d`). Thus, it currently does NOT
        support inputs without a batch dimension of shape :math:`(C, H, W)`. This
        behavior will change in a future release to interpret 3D inputs as no-batch-dim
        inputs. To maintain the old behavior, switch to :class:`nn.Dropout1d`.

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(N, C, L)`.
        - Output: :math:`(N, C, H, W)` or :math:`(N, C, L)` (same shape as input).

    Examples::

        >>> m = nn.Dropout2d(p=0.2)
        >>> input = torch.randn(20, 16, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional 2D channel dropout using p, the training flag and inplace.
        # Returns a tensor of the same shape. Random mask drawn from the global RNG.
        """
        Runs the forward pass.
        """
        return F.dropout2d(input, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):
    # [CLASS DOC]
    # Channel-wise dropout for three-dimensional feature maps: each whole channel is zeroed
    # independently with probability p during training. Expects (N, C, D, H, W) or (C, D, H, W).
    # State inherited from _DropoutNd (p, inplace).
    r"""Randomly zero out entire channels.

    A channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`.

    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv3d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional 3D channel dropout using p, the training flag and inplace.
        # Returns a tensor of the same shape. Random mask drawn from the global RNG.
        """
        Runs the forward pass.
        """
        return F.dropout3d(input, self.p, self.training, self.inplace)


class AlphaDropout(_DropoutNd):
    # [CLASS DOC]
    # Alpha dropout that keeps zero mean and unit variance for SELU-activated inputs: dropped
    # elements are set to the negative saturation value, then the result is scaled and shifted.
    # Identity in eval mode. Accepts any shape. State inherited from _DropoutNd (p, inplace).
    # KNOWN ISSUE: the inplace flag is stored but forward does not pass it to the functional call.
    r"""Applies Alpha Dropout over the input.

    Alpha Dropout is a type of Dropout that maintains the self-normalizing
    property.
    For an input with zero mean and unit standard deviation, the output of
    Alpha Dropout maintains the original mean and standard deviation of the
    input.
    Alpha Dropout goes hand-in-hand with SELU activation function, which ensures
    that the outputs have zero mean and unit standard deviation.

    During training, it randomly masks some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit standard deviation.

    During evaluation the module simply computes an identity function.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        p (float): probability of an element to be dropped. Default: 0.5
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.AlphaDropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional alpha dropout using p and the training flag; returns a tensor of the same shape.
        # KNOWN ISSUE: the inplace attribute is ignored here.
        """
        Runs the forward pass.
        """
        return F.alpha_dropout(input, self.p, self.training)


class FeatureAlphaDropout(_DropoutNd):
    # [CLASS DOC]
    # Alpha dropout applied to entire channels (feature maps) instead of single elements,
    # keeping SELU self-normalizing statistics. Expects (N, C, D, H, W) or (C, D, H, W) per the docstring.
    # State inherited from _DropoutNd (p, inplace).
    # KNOWN ISSUE: the inplace flag is stored but forward does not pass it to the functional call.
    r"""Randomly masks out entire channels.

    A channel is a feature map,
    e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input
    is a tensor :math:`\text{input}[i, j]` of the input tensor). Instead of
    setting activations to zero, as in regular Dropout, the activations are set
    to the negative saturation value of the SELU activation function. More details
    can be found in the paper `Self-Normalizing Neural Networks`_ .

    Each element will be masked independently for each sample on every forward
    call with probability :attr:`p` using samples from a Bernoulli distribution.
    The elements to be masked are randomized on every forward call, and scaled
    and shifted to maintain zero mean and unit variance.

    Usually the input comes from :class:`nn.AlphaDropout` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.AlphaDropout` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed. Default: 0.5
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input).

    Examples::

        >>> m = nn.FeatureAlphaDropout(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies functional feature alpha dropout using p and the training flag; returns a tensor of the same shape.
        # KNOWN ISSUE: the inplace attribute is ignored here.
        """
        Runs the forward pass.
        """
        return F.feature_alpha_dropout(input, self.p, self.training)
