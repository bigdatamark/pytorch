# [MODULE DOC] Defines Shard, a dataclass pairing a local tensor with the ShardMetadata describing its
# offsets, sizes and placement. No import-time side effects.
from dataclasses import dataclass

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.remote_device import _remote_device


@dataclass
class Shard:
    # [CLASS DOC] Container for one shard of a ShardedTensor: the local tensor plus its ShardMetadata. Uses slots.
    # Construction validates that the tensor size equals the metadata shard sizes and, if a placement is set, that
    # the tensor device matches the placement device; otherwise ValueError. Alternate constructor:
    # from_tensor_and_offsets.
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(torch.Tensor): Local tensor for the shard.
        metadata(:class `torch.distributed._shard.sharded_tensor.ShardMetadata`):
            The metadata for the shard, including offsets, lengths and device placement.
    """

    __slots__ = ["tensor", "metadata"]
    tensor: torch.Tensor
    metadata: ShardMetadata

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Post-init validation: raises ValueError if the tensor's sizes differ from metadata.shard_sizes,
        # or if metadata has a placement whose device differs from the tensor's device.
        # verification between local tensor and metadata
        if list(self.tensor.size()) != self.metadata.shard_sizes:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_sizes}, "
            )
        placement_device = self.metadata.placement
        if (
            placement_device is not None
            and placement_device.device() != self.tensor.device
        ):
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {self.tensor.device}, "
                f"local shard metadata placement device: {placement_device.device()}"
            )

    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: torch.Tensor, shard_offsets: list[int], rank: int
    ) -> "Shard":
        # [FUNCTION DOC] Class method: builds a Shard from a local tensor, a list of per-dimension shard offsets and a
        # rank. Shard sizes come from the tensor, placement is a remote device "rank:<rank>/<tensor device>".
        # Returns the new Shard (validated by post-init).
        """
        Creates a Shard of a ShardedTensor from a local torch.Tensor, shard_offsets and rank.

        Args:
            tensor(torch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        """
        shard_sizes = list(tensor.size())
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets, shard_sizes=shard_sizes, placement=placement
        )
        return Shard(tensor, shard_meta)
