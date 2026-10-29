# [MODULE DOC] DimEntry, a small wrapper identifying one tensor dimension ("level") in
# first-class dimensions: either a first-class Dim object or a negative integer giving a
# positional dimension counted from the right. Also provides ndim_of_levels (count positional
# levels) and _match_levels (restride a tensor to a target level order). No import-time side
# effects beyond imports; Dim is imported lazily to avoid a circular import.
from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from . import Dim

import torch  # noqa: TC002


# NB: The old code represented dimension was from as negative number, so we
# follow this convention even though it shouldn't be necessary now
class DimEntry:
    # [CLASS DOC] Identifies one level of a first-class-dimension tensor. State: data, either a
    # Dim object (a named first-class dimension), a negative int (positional dimension counted
    # from the right), or the int 0 meaning "none". Equality compares ints by value and Dims by
    # identity (to avoid triggering tensor-like comparison); a positional entry never equals a
    # Dim entry. Key methods: is_positional, is_none, position, dim.
    # The dimension this is from the rhs, or a FCD
    data: Dim | int

    def __init__(self, data: Dim | int | None = None) -> None:
        # [FUNCTION DOC] Create an entry from a Dim, a negative int, or None (stored as 0,
        # the "none" entry). Raises an error if given a non-negative int (including an explicit
        # 0) or any value that is neither int nor Dim.
        from . import Dim

        if type(data) is int:
            if data >= 0:
                raise AssertionError(f"Expected negative int, got {data}")
        elif data is None:
            data = 0
        else:
            if not isinstance(data, Dim):
                raise AssertionError(f"Expected Dim, got {type(data)}")
        self.data = data

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] True only if other is a DimEntry and both are positional with equal
        # ints, or both hold the very same Dim object. Mixed or non-DimEntry compare as False.
        if not isinstance(other, DimEntry):
            return False
        # Use 'is' for Dim objects to avoid triggering __torch_function__
        # Use '==' only for positional (int) comparisons
        if self.is_positional() and other.is_positional():
            # Both are positional (ints)
            return self.data == other.data
        elif not self.is_positional() and not other.is_positional():
            # Both are Dim objects - use 'is' to avoid __eq__
            return self.data is other.data
        else:
            # One is positional, one is Dim - they can't be equal
            return False

    def is_positional(self) -> bool:
        # [FUNCTION DOC] True if data is an int that is negative (a positional dimension).
        return type(self.data) is int and self.data < 0

    def is_none(self) -> bool:
        # [FUNCTION DOC] True if this is the "none" entry, i.e. data is not a Dim and equals 0.
        # Use isinstance to check for Dim objects, avoid triggering __torch_function__
        from . import Dim

        if isinstance(self.data, Dim):
            # This is a Dim object, it can't be "none" (which is represented by 0)
            return False
        else:
            # This is an int or other type
            return self.data == 0

    def position(self) -> int:
        # [FUNCTION DOC] Return the stored int. Raises an error if data is not an int.
        if not isinstance(self.data, int):
            raise AssertionError(f"Expected int, got {type(self.data)}")
        return self.data

    def dim(self) -> Dim:
        # [FUNCTION DOC] Return the stored Dim. Raises an error if data is an int.
        if isinstance(self.data, int):
            raise AssertionError("Expected Dim, got int")
        return self.data

    def __repr__(self) -> str:
        # [FUNCTION DOC] String form is the repr of the wrapped data (Dim or int).
        return repr(self.data)


def ndim_of_levels(levels: Sequence[DimEntry]) -> int:
    # [FUNCTION DOC] Return how many entries in levels are positional (not first-class Dims).
    r = 0
    for l in levels:
        if l.is_positional():
            r += 1
    return r


def _match_levels(
    tensor: torch.Tensor,
    from_levels: list[DimEntry],
    to_levels: list[DimEntry],
    drop_levels: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Return tensor viewed with the dimension order of to_levels. For each
    # target level, size and stride come from the matching entry in from_levels; a level absent
    # there gets stride 0 and size 1 (positional) or the Dim's size (first-class). Returns the
    # tensor itself if the levels are equal. Raises an error if from_levels is longer than
    # to_levels and drop_levels is False. The parameter drop_levels only gates that check.
    """
    Reshape a tensor to match target levels using as_strided.

    Args:
        tensor: Input tensor to reshape
        from_levels: Current levels of the tensor
        to_levels: Target levels to match
        drop_levels: If True, missing dimensions are assumed to have stride 0

    Returns:
        Reshaped tensor
    """
    if from_levels == to_levels:
        return tensor

    sizes = tensor.size()
    strides = tensor.stride()

    if not drop_levels:
        if len(from_levels) > len(to_levels):
            raise AssertionError("Cannot expand dimensions without drop_levels")

    new_sizes = []
    new_strides = []

    for level in to_levels:
        # Find index of this level in from_levels
        try:
            idx = from_levels.index(level)
        except ValueError:
            # Level not found in from_levels
            if level.is_positional():
                new_sizes.append(1)
            else:
                new_sizes.append(level.dim().size)
            new_strides.append(0)
        else:
            new_sizes.append(sizes[idx])
            new_strides.append(strides[idx])

    return tensor.as_strided(new_sizes, new_strides, tensor.storage_offset())
