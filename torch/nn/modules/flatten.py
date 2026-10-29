# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Flatten and Unflatten modules, thin Module wrappers around the tensor
# flatten and unflatten operations for use inside Sequential containers.
# Exports Flatten and Unflatten via __all__. No import-time side effects.

from torch import Tensor
from torch.types import _size

from .module import Module


__all__ = ["Flatten", "Unflatten"]


class Flatten(Module):
    # [CLASS DOC]
    # Parameter-free module that merges a contiguous range of dimensions
    # (start_dim through end_dim, inclusive) of its input into one dimension.
    # State: integer attributes start_dim and end_dim, both listed in __constants__.
    # Key methods: forward delegates to the tensor flatten method; extra_repr
    # reports both dims.
    r"""
    Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`, see :meth:`torch.flatten` for details.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = nn.Flatten()
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 25])
        >>> # With non-default parameters
        >>> m = nn.Flatten(0, 2)
        >>> output = m(input)
        >>> output.size()
        torch.Size([160, 5])
    """

    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        # [FUNCTION DOC]
        # Stores start_dim (default 1) and end_dim (default -1) after base Module
        # initialization. No validation is performed here.
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the input with dimensions start_dim..end_dim merged into one.
        # Errors for out-of-range dims are raised by the underlying tensor operation.
        """
        Runs the forward pass.
        """
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Returns a string listing start_dim and end_dim, used in the module repr.
        """
        Returns the extra representation of the module.
        """
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    # [CLASS DOC]
    # Parameter-free module that expands one dimension (dim) of its input into
    # several dimensions given by unflattened_size.
    # State: integer dim and a tuple/list of ints unflattened_size, both in
    # __constants__. The constructor validates that unflattened_size is a
    # tuple or list of ints. Key methods: forward, extra_repr.
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        dim (int): Dimension to be unflattened
        unflattened_size (Union[torch.Size, Tuple, List]): New shape of the unflattened dimension

    Examples:
        >>> input = torch.randn(2, 50)
        >>> # With tuple of ints
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
        >>> # With torch.Size
        >>> m = nn.Sequential(
        >>>     nn.Linear(50, 50),
        >>>     nn.Unflatten(1, torch.Size([2, 5, 5]))
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([2, 2, 5, 5])
    """

    __constants__ = ["dim", "unflattened_size"]
    dim: int
    unflattened_size: _size

    def __init__(self, dim: int, unflattened_size: _size) -> None:
        # [FUNCTION DOC]
        # Initializes the base Module, validates unflattened_size via the helper
        # (raises TypeError if not a tuple/list of ints), then stores dim and
        # unflattened_size.
        super().__init__()
        self._require_tuple_int(unflattened_size)
        self.dim = dim
        self.unflattened_size = unflattened_size

    def _require_tuple_int(self, input) -> None:
        # [FUNCTION DOC] HELPER: Validation of the unflattened_size argument.
        # Returns None if input is a tuple or list whose elements are all ints.
        # Raises TypeError naming the offending element type and position, or the
        # overall type if input is neither tuple nor list.
        if isinstance(input, (tuple, list)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError(
                        "unflattened_size must be tuple of ints, "
                        + f"but found element of type {type(elem).__name__} at pos {idx}"
                    )
            return
        raise TypeError(
            f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}"
        )

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns the input with dimension dim split into the sizes in
        # unflattened_size. Size mismatches are reported by the tensor operation.
        """
        Runs the forward pass.
        """
        return input.unflatten(self.dim, self.unflattened_size)

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Returns a string listing dim and unflattened_size, used in the module repr.
        """
        Returns the extra representation of the module.
        """
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"
