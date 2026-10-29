# [PACKAGE DOC] Imports (for their registration side effects) all ShardedTensor op handlers: misc ops, tensor ops, the ChunkShardingSpec embedding and embedding_bag ops,
# and re-exports allclose, equal and the init ops (constant_, kaiming_uniform_, normal_, uniform_).
import torch.distributed._shard.sharded_tensor._ops.misc_ops
import torch.distributed._shard.sharded_tensor._ops.tensor_ops

# Import all ChunkShardingSpec ops
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import (
    sharded_embedding,
)
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import (
    sharded_embedding_bag,
)

from .binary_cmp import allclose, equal
from .init import constant_, kaiming_uniform_, normal_, uniform_
