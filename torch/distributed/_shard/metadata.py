# mypy: allow-untyped-defs
# [MODULE DOC] Defines ShardMetadata, the description of one shard of a tensor (offsets, sizes, placement).
from dataclasses import dataclass
from functools import reduce

from torch.distributed.remote_device import _remote_device


@dataclass
class ShardMetadata:
    # [CLASS DOC] Dataclass with slots shard_offsets (list of ints), shard_sizes (list of ints) and placement (optional remote device).
    # Invariant enforced at construction: offsets and sizes have equal length and are non-negative. Hashable via a combined hash of all fields.
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
    """

    __slots__ = ["shard_offsets", "shard_sizes", "placement"]

    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: _remote_device | None

    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: str | _remote_device | None = None,
    ):
        # [FUNCTION DOC] Store the offsets and sizes; a string placement is parsed into a remote device. Raises ValueError if the lengths differ or any offset or size is negative.
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")

    def __hash__(self):
        # [FUNCTION DOC] Hash combining offsets, sizes and placement via a shift-and-add fold starting from a fixed seed.
        def _hash_reduce(a, b):
            # [FUNCTION DOC] HELPER: Fold step: shift accumulator left by 8 bits and add the hash of the next item.
            return (a << 8) + hash(b)

        res = reduce(_hash_reduce, self.shard_offsets, 37)
        res = reduce(_hash_reduce, self.shard_sizes, res)
        res = _hash_reduce(res, self.placement)
        return res
