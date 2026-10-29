# [PACKAGE DOC] Package for sharding specs. Re-exports ShardMetadata, the spec classes from the api module
# (DevicePlacementSpec, EnumerableShardingSpec, PlacementSpec, ShardingSpec, and the helper that infers a spec
# from shard metadata) and ChunkShardingSpec.
from torch.distributed._shard.metadata import ShardMetadata

from .api import (
    _infer_sharding_spec_from_shards_metadata,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    PlacementSpec,
    ShardingSpec,
)
from .chunk_sharding_spec import ChunkShardingSpec as ChunkShardingSpec
