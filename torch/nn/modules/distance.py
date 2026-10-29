# [MODULE DOC]
# Distance modules for torch.nn: PairwiseDistance (p-norm distance between paired vectors)
# and CosineSimilarity (cosine of the angle between paired vectors). Each is a thin
# Module wrapper around the matching torch.nn.functional call. Exports both names via
# __all__. No import-time side effects beyond imports.
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["PairwiseDistance", "CosineSimilarity"]


class PairwiseDistance(Module):
    # [CLASS DOC]
    # Module computing the p-norm distance between corresponding vectors of two inputs.
    # State: norm (the norm degree, constructor argument p), eps (small value added to the
    # difference for numerical stability), keepdim (whether the reduced vector dimension is kept).
    # No parameters or buffers. forward delegates to the functional pairwise distance.
    r"""
    Computes the pairwise distance between input vectors, or between columns of input matrices.

    Distances are computed using ``p``-norm, with constant ``eps`` added to avoid division by zero
    if ``p`` is negative, i.e.:

    .. math ::
        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + \epsilon e \right\Vert_p,

    where :math:`e` is the vector of ones and the ``p``-norm is given by.

    .. math ::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    Args:
        p (real, optional): the norm degree. Can be negative. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension.
            Default: False
    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)` where `N = batch dimension` and `D = vector dimension`
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the Input1
        - Output: :math:`(N)` or :math:`()` based on input dimension.
          If :attr:`keepdim` is ``True``, then :math:`(N, 1)` or :math:`(1)` based on input dimension.

    Examples:
        >>> pdist = nn.PairwiseDistance(p=2)
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> output = pdist(input1, input2)
    """

    __constants__ = ["norm", "eps", "keepdim"]
    norm: float
    eps: float
    keepdim: bool

    def __init__(
        self, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # Stores p as the attribute norm, plus eps and keepdim, after initializing the Module base.
        # Args: p (norm degree, default 2.0), eps (default 1e-6), keepdim (default False). No validation is done.
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the functional pairwise distance of x1 and x2 using the stored norm, eps and keepdim.
        # Inputs are same-shaped (or broadcastable) tensors; output reduces the last dimension
        # (kept as size one when keepdim is true). Shape errors come from the functional call.
        """
        Runs the forward pass.
        """
        return F.pairwise_distance(x1, x2, self.norm, self.eps, self.keepdim)


class CosineSimilarity(Module):
    # [CLASS DOC]
    # Module computing cosine similarity between two tensors along a given dimension.
    # State: dim (dimension reduced over, default 1), eps (lower bound on the norm product
    # to avoid division by zero, default 1e-8). No parameters or buffers.
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
          and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`

    Examples:
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        # [FUNCTION DOC]
        # Stores dim and eps after initializing the Module base. No validation is done.
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the functional cosine similarity of x1 and x2 along the stored dim with the stored eps.
        # The output has the input shape with dimension dim removed. Shape errors come from the functional call.
        """
        Runs the forward pass.
        """
        return F.cosine_similarity(x1, x2, self.dim, self.eps)
