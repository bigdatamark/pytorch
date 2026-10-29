# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Internal helpers for sharding specs: overlap detection between shards (pairwise, 1-D
# interval and N-D sweep-line), validation that shards do not overlap and cover a tensor, and chunking
# arithmetic that mirrors torch.chunk. No import-time side effects.
import math
import sys
from bisect import bisect_right, insort

from torch.distributed._shard.metadata import ShardMetadata


def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    # [FUNCTION DOC] HELPER: Returns True if two shards overlap, i.e. their extents intersect in every dimension,
    # and False as soon as they are disjoint in one dimension. Uses the dimension count of the first shard.
    """
    Checks if two shards overlap.
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_sizes[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_sizes[i]:
            return False

    return True


def _find_nd_overlapping_shards(
    shards: list[ShardMetadata], sharded_dims: list[int]
) -> tuple[int, int] | None:
    # [FUNCTION DOC] HELPER: Finds an overlapping pair among shards partitioned over several dimensions using a sweep
    # over the sharded dimension. The sweep dimension is the one with the largest extent of the first shard; shards
    # are visited in offset order while an ordered list of active shards (by end position) is pruned and each new
    # shard is checked against actives. Returns a pair of indices (earlier active, current) or None. Returns None for
    # fewer than two shards or no sharded dims.
    """Find overlapping shards using sweep-line algorithm."""
    if len(shards) <= 1:
        return None

    dims = len(sharded_dims)
    if dims == 0:
        return None

    sweep_dim_idx = 0
    if dims > 1:
        max_size = 0
        for i, dim in enumerate(sharded_dims):
            dim_size = shards[0].shard_offsets[dim] + shards[0].shard_sizes[dim]
            if dim_size > max_size:
                max_size = dim_size
                sweep_dim_idx = i
    sweep_dim = sharded_dims[sweep_dim_idx]

    sorted_indices = sorted(
        range(len(shards)),
        key=lambda idx: (
            shards[idx].shard_offsets[sweep_dim],
            *(shards[idx].shard_offsets[d] for d in sharded_dims if d != sweep_dim),
        ),
    )
    active: list[tuple[int, int]] = []

    for idx in sorted_indices:
        current = shards[idx]
        start = current.shard_offsets[sweep_dim]
        end = start + current.shard_sizes[sweep_dim]

        cutoff = bisect_right(active, (start, sys.maxsize))
        if cutoff:
            del active[:cutoff]

        for _, other_idx in active:
            other = shards[other_idx]

            if _check_shard_metadata_pair_overlap(current, other):
                return (other_idx, idx)
        insort(active, (end, idx))
    return None


def _find_1d_overlapping_shards(
    shards: list[ShardMetadata], dim: int
) -> tuple[int, int] | None:
    # [FUNCTION DOC] HELPER: Finds an overlapping pair of shards when only one dimension is partitioned: sorts
    # inclusive intervals along dim and compares neighbours. Returns a pair of shard indices or None.
    # (begin, end, index_in_shards). Begin and end are inclusive.
    intervals = [
        (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i)
        for i, s in enumerate(shards)
    ]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None


def validate_non_overlapping_shards_metadata(shards: list[ShardMetadata]):
    # [FUNCTION DOC] Raises ValueError if any two shards overlap. Determines which dimensions differ among shards;
    # no differing dimension means all shards are identical, which is accepted only if every shard has zero offsets
    # and an empty (zero volume) size, otherwise shards 0 and 1 are reported; one differing dimension uses the
    # 1-D interval check; more use the sweep-line check. Returns None when there are fewer than two shards.
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
    if not shards or len(shards) == 1:
        return

    sharded_dims: list[int] = []
    for dim in range(len(shards[0].shard_offsets)):
        for i in range(1, len(shards)):
            if (
                shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim]
                or shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]
            ):
                sharded_dims.append(dim)
                break

    pair: tuple[int, int] | None = None
    if len(sharded_dims) == 0:
        # if shard is all zeros, we should consider as pass
        all_zeros: bool = all(
            # strictly limited all offsets to be 0 to pass
            # could loosen it later on
            shard.shard_offsets == [0] * len(shards[0].shard_offsets)
            and math.prod(shard.shard_sizes) == 0  # one dimension is 0
            for shard in shards
        )
        if all_zeros:
            return
        # All shards are the same, all dims are not partitioned. Choose any 2.
        pair = (0, 1)
    elif len(sharded_dims) == 1:
        # Shards are partitioned over only one dimension. Overlap can be found
        # using a O(nlogn) overlapping interval algorithm.
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    else:
        # Shards are partitioned over more than one dimension.
        # Use sweep-line algorithm for O(n log n) complexity.
        pair = _find_nd_overlapping_shards(shards, sharded_dims)

    if pair:
        raise ValueError(f"Shards {shards[pair[0]]} and {shards[pair[1]]} overlap")


def check_tensor(shards_metadata, tensor_dims) -> None:
    # [FUNCTION DOC] Checks that shards fit a tensor of the given dims: the number of dimensions must match, each
    # shard must lie inside the tensor on every dimension, and total shard volume must equal tensor volume (which,
    # given non-overlap, means full coverage). Raises ValueError otherwise. Assumes at least one shard.
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """

    # If the tensor's volume matches the total volume of all shards and
    # all shard boundaries are within tensor dims, we have a compatible
    # sharding spec for this tensor. Note that we have already verified
    # we don't have overlapping shards.
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(
            f"Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}"
        )

    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for i, shard_length in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(
                    f"Shard offset {shard.shard_offsets[i]} and length "
                    f"{shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}"
                )
        total_shard_volume += shard_volume

    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f"Total volume of shards: {total_shard_volume} "
            f"does not match tensor volume: {tensor_volume}, in other words "
            f"all the individual shards do not cover the entire tensor"
        )


def get_split_size(dim_size, chunks):
    # [FUNCTION DOC] Returns the chunk size torch.chunk would use: ceiling division of dim_size by chunks.
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    return (dim_size + chunks - 1) // chunks


def get_chunked_dim_size(dim_size, split_size, idx):
    # [FUNCTION DOC] Returns the size of chunk number idx for a dimension of size dim_size split with split_size,
    # clamped to zero for chunks past the end.
    """
    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``
    and ``split_size``.

    Args:
        dim_size(int): Size of the dimension being chunked.
        split_size(int): The chunk size for each chunk of ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
    return max(min(dim_size, split_size * (idx + 1)) - split_size * idx, 0)


def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    # [FUNCTION DOC] Returns (start offset, chunk size) for the given rank under chunk sharding. Walks the spec's
    # placements in order accumulating offsets using the chunk sizes of a split across world_size, and stops at the
    # placement whose rank equals rank. KNOWN ISSUE: if the rank is not in the placements, the start is 0 and the
    # size is that of the last chunk; with no placements the size is undefined and an error would occur.
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.

    Args:
        sharding_dim_size(int): The dimension length which we shard on.
        world_size(int): number of ranks.
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            sharding spec.
        rank(int): # of cuda process.

    Returns:
        start_pos(int): start position of sharded tensor on the given rank.
        chunk_size(int): chunk size of sharded tensor on the given rank.
    """
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return start_pos, chunk_size  # type: ignore[possibly-undefined]
