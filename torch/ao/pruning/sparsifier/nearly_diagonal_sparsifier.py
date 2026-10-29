# mypy: allow-untyped-defs
# [MODULE DOC] Defines NearlyDiagonalSparsifier, which keeps a band of diagonals around
# the main diagonal of a 2D weight. No import-time side effects.
import torch

from . import base_sparsifier


class NearlyDiagonalSparsifier(base_sparsifier.BaseSparsifier):
    # [CLASS DOC] Sparsifier whose mask keeps only entries within a distance of half
    # (nearliness minus one) of the main diagonal; configured by the odd integer nearliness.
    r"""Nearly Diagonal Sparsifier

    This sparsifier creates a nearly diagonal mask to be applied to the weight matrix.
    Nearly Diagonal Matrix is a matrix that contains non-zero elements near the diagonal and the rest are zero.
    An example of a nearly diagonal matrix with degree (or nearliness) 3 and 5 are follows respectively.
    1 1 0 0       1 1 1 0
    1 1 1 0       1 1 1 1
    0 1 1 1       1 1 1 1
    0 0 1 1       0 1 1 1
    Note that a nearly diagonal matrix with degree 1 is just a matrix with main diagonal populated

    This sparsifier is controlled by one variable:
    1. `nearliness` defines the number of non-zero diagonal lines that are closest to the main diagonal.
        Currently - supports only odd number

    Note:
        This can be accelerated (vectorized) once the Spdiagonal feature (PR: #78439) is landed or the banded matrix
        feature is landed: https://stackoverflow.com/questions/52463972/generating-banded-matrices-using-numpy

    Args:
        nearliness: The degree of nearliness (default = 1)

    """

    def __init__(self, nearliness: int = 1):
        # [FUNCTION DOC] Stores nearliness as the default config and initializes the base class.
        defaults = {"nearliness": nearliness}
        super().__init__(defaults=defaults)

    def update_mask(  # type:ignore[override]
        self, module, tensor_name, nearliness, **kwargs
    ):
        # [FUNCTION DOC] Resets the mask of module.tensor_name to zeros and, unless nearliness
        # is not positive (then all zeros remain), sets to one the entries within half of
        # nearliness (integer division) of the diagonal in every row. Raises ValueError if
        # nearliness is even or its half is not smaller than the smaller tensor dimension.
        # The weight must be 2D. Mutates the mask in place.
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        mask.data = torch.zeros_like(mask)
        if nearliness <= 0:
            return

        tensor = getattr(module, tensor_name)
        height, width = tensor.shape

        if nearliness % 2 == 0:
            raise ValueError("nearliness can only be an odd number")
        dist_to_diagonal = nearliness // 2
        # check
        if dist_to_diagonal >= min(height, width):
            raise ValueError(
                "nearliness cannot be larger than the dimensions of tensor."
            )

        for row in range(height):
            # Bounds of entries that needs to be set to 1
            low = max(0, row - dist_to_diagonal)
            high = min(width, row + dist_to_diagonal + 1)
            mask[row, low:high].fill_(1)
