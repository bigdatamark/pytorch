# [MODULE DOC] Geometry helpers for checkpoint resharding: test whether two chunks overlap and
# compute the overlapping region of a saved chunk and a current chunk. Exports nothing publicly.
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata


__all__: list[str] = []


def _check_shard_metadata_pair_overlap(
    shard1: ChunkStorageMetadata, shard2: ChunkStorageMetadata
) -> bool:
    # [FUNCTION DOC] HELPER: Return True if the two chunks overlap in every dimension (uses the
    # first chunk's rank; no overlap if in any dimension one starts at or past the other's end).
    """Check if two shards overlap."""
    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.offsets)
    for i in range(ndims):
        if shard1.offsets[i] >= shard2.offsets[i] + shard2.sizes[i]:
            return False
        if shard2.offsets[i] >= shard1.offsets[i] + shard1.sizes[i]:
            return False

    return True


def _shards_get_overlap_region_wrt_saved_tensor(
    saved_shard: ChunkStorageMetadata, current_shard: ChunkStorageMetadata
) -> list[tuple[int, int, int, int]]:
    # [FUNCTION DOC] HELPER: For each dimension return (dim, offset within saved shard, offset within
    # current shard, overlap length). Assumes the shards overlap; with no overlap the length can be
    # zero or negative (not checked).
    """
    Return the overlapping region between saved_shard and current_shard.

    The returned list has the same number of elements as the tensor's dimension.
    For each element, we produce a tuple with the following contents:
        (dimension, `saved_shard` offset, `current_shard` offset, length)

    Offsets are relative to each shard.
    """
    narrows = []
    for dim, (
        saved_shard_offset,
        current_shard_offset,
        saved_shard_size,
        current_shard_size,
    ) in enumerate(
        zip(
            saved_shard.offsets,
            current_shard.offsets,
            saved_shard.sizes,
            current_shard.sizes,
        )
    ):
        min_range_end = min(
            saved_shard_offset + saved_shard_size,
            current_shard_offset + current_shard_size,
        )

        length = min_range_end - max(current_shard_offset, saved_shard_offset)

        if saved_shard_offset > current_shard_offset:
            offset_for_saved_tensor = 0
            offset_for_current_tensor = saved_shard_offset - current_shard_offset
        else:
            offset_for_saved_tensor = current_shard_offset - saved_shard_offset
            offset_for_current_tensor = 0

        narrows.append(
            (dim, offset_for_saved_tensor, offset_for_current_tensor, length)
        )

    return narrows
