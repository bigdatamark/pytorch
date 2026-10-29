# [MODULE DOC] EnableAllLayers: a context manager used by first-class dimensions that pushes one
# functorch vmap nesting level per bound Dim so batching rules apply, and converts batched
# results back into first-class-dim Tensors. Depends on private torch._C._functorch bindings.
# No import-time side effects.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from ._dim_entry import DimEntry


if TYPE_CHECKING:
    from . import Dim, Tensor


class EnableAllLayers:
    # [CLASS DOC] Context manager that, on entry, enters one functorch vmap level per first-class
    # Dim found in the given levels (sorted by each Dim's level) and, on exit, pops them in
    # reverse. State: levels_to_dim (the Dims, ordered) and levels_start (the functorch level of
    # the first pushed layer, 0 until entered). Invariant: layers popped on exit must match the
    # ones pushed, otherwise an error is raised. Also maps batched tensors back to Tensor objects.
    """
    RAII-style context manager for enabling functorch vmap layers.
    It manages the creation and cleanup of functorch dynamic layers.

    This is probably one of the more algorithmically important parts of first
    class dims. Intuitively, FCD can be thought of as another way of using
    vmap, where you don't actually have to vmap at the top level, instead the
    vmaps are implicitly determined by inspecting the bound dimensions on the
    FCD tensors involved in a compute (this is similar to our concept of
    non-lexical modes that we spent a long time talking about years ago). But
    under the hood you still need to actually enable the vmap mode. So once
    FCD has determined all of the dims we are batching over, it needs to
    enable all those layers so functorch can actually apply the batching
    rules. Therefore enable all layers!
    """

    levels_start: int
    levels_to_dim: list[Dim]

    def __init__(self, levels: list[DimEntry]):
        # [FUNCTION DOC] Collect the Dim of every non-positional entry in levels into
        # levels_to_dim, sorted by the Dim's _level. Raises an error if an entry holds a
        # non-Dim. Does not touch functorch state; layers are only pushed on entry.
        """
        Initialize and push dynamic layers for all first-class dimensions.

        Args:
            levels: List of dimension entries to create layers for
        """

        from . import Dim

        self.levels_start = 0
        self.levels_to_dim = []

        for l in levels:
            if not l.is_positional():
                d = l.dim()
                if not isinstance(d, Dim):
                    raise AssertionError(f"Expected Dim, got {type(d)}")
                self.levels_to_dim.append(d)

        # Sort by level for stable ordering
        self.levels_to_dim.sort(key=lambda d: d._level)

    def __enter__(self) -> EnableAllLayers:  # noqa: PYI034
        # [FUNCTION DOC] Push one vmap nesting level (randomness mode "different") per Dim, with
        # that Dim's size as batch size, and record the first level in levels_start. Global
        # side effect on the functorch dynamic layer stack. Returns self.
        # Create functorch dynamic layers
        for i, dim in enumerate(self.levels_to_dim):
            batch_size = dim.size
            level = torch._C._functorch._vmap_increment_nesting(batch_size, "different")
            if i == 0:
                self.levels_start = level
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # [FUNCTION DOC] Pop one vmap nesting level per Dim, checking each popped level equals
        # the expected one (highest first); raises an error on mismatch. Exception arguments are
        # ignored and exceptions are not suppressed.
        """Clean up dynamic layers in reverse order."""
        to_remove = self.levels_start + len(self.levels_to_dim) - 1
        for i in range(len(self.levels_to_dim)):
            popped = torch._C._functorch._vmap_decrement_nesting()
            if popped != to_remove - i:
                raise AssertionError(f"Expected layer {to_remove - i}, got {popped}")

    def from_batched(self, batchedtensor: torch.Tensor, has_device: bool) -> Tensor:
        # [FUNCTION DOC] Unwrap the functorch batched-tensor layers of batchedtensor. Starts with
        # a positional level for each base dimension, then for each wrapper layer inserts the
        # matching first-class Dim at the layer's batch dimension. Returns a new Tensor holding
        # the innermost tensor, the original batched tensor, has_device, and the levels. Raises
        # an error if a layer level lies outside this manager's range or lacks level/bdim info.
        """
        Create a Tensor from a batched tensor by unwrapping functorch layers.

        Args:
            batchedtensor: Batched tensor from functorch operation
            has_device: Whether tensor has device info

        Returns:
            Tensor with appropriate levels
        """
        # Create positional levels for base dimensions
        levels: list[DimEntry] = []
        for i in range(-batchedtensor.dim(), 0):
            levels.append(DimEntry(i))

        tensor = batchedtensor

        while torch._C._functorch.is_batchedtensor(tensor):
            level = torch._C._functorch.maybe_get_level(tensor)
            if level is None:
                raise AssertionError("Expected level to be non-None")
            if not (
                level >= self.levels_start
                and level < self.levels_start + len(self.levels_to_dim)
            ):
                raise AssertionError(f"Level {level} out of range")
            dim = DimEntry(self.levels_to_dim[level - self.levels_start])
            bdim = torch._C._functorch.maybe_get_bdim(tensor)
            if bdim is None:
                raise AssertionError("Expected bdim to be non-None")
            levels.insert(bdim, dim)
            tensor = torch._C._functorch.get_unwrapped(tensor)

        from . import Tensor

        result = Tensor()
        result._tensor = tensor
        result._batchtensor = batchedtensor
        result._has_device = has_device
        result._levels = levels
        return result

    def inplace_update_layers(
        self, batchtensor: torch.Tensor, levels: list[DimEntry]
    ) -> None:
        # [FUNCTION DOC] For a functorch batched tensor, walk its wrapper layers from the
        # innermost Dim outward and, for each managed Dim that appears in levels, overwrite the
        # layer's level with the matching level of this manager (unsafe in-place mutation of the
        # tensor). Does nothing if the tensor is not batched. Returns nothing.
        """
        Update the levels of a batched tensor in place.

        This requires the _maybe_unsafe_set_level binding that we'll add to functorch.

        Args:
            batchtensor: Batched tensor to update
            levels: New levels to set
        """
        # Check if tensor is batched
        if not torch._C._functorch.is_batchedtensor(batchtensor):
            return

        impl = batchtensor

        for i in reversed(range(len(self.levels_to_dim))):
            if impl is None:
                break

            if any(l == DimEntry(self.levels_to_dim[i]) for l in levels):
                # This is very interesting!  The level on batch tensor is
                # meaningless!  We set it RIGHT before we go into vmap
                torch._C._functorch._maybe_unsafe_set_level(impl, self.levels_start + i)
                impl = torch._C._functorch.get_unwrapped(impl)
