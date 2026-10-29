# [MODULE DOC] Small helpers for the sharding package: the deprecation message constant (recommending DTensor) and functions that narrow a tensor to a shard
# given offsets and sizes or a ShardMetadata.
from collections.abc import Sequence

import torch
from torch.distributed._shard.metadata import ShardMetadata


DEPRECATE_MSG = "Please use DTensor instead and we are deprecating ShardedTensor."


def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    # [FUNCTION DOC] Narrow tensor along each dimension idx to the range [offset, offset + size) for the paired entries of offsets and sizes; dimensions where size is not smaller than the tensor
    # size are left untouched. Returns the (possibly same) tensor view. Errors from the narrow op (out of range) propagate.
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
    # [FUNCTION DOC] Narrow tensor using the shard offsets and sizes from the given ShardMetadata; returns the resulting view.
    """
    Narrow the tensor according to the metadata
    """
    return narrow_tensor_by_index(tensor, metadata.shard_offsets, metadata.shard_sizes)
