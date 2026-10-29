# [PACKAGE DOC] Package for the (deprecated in favor of DTensor) sharding APIs; re-exports shard_parameter, shard_module, load_with_process_group and _shard_tensor from api.
from .api import _shard_tensor, load_with_process_group, shard_module, shard_parameter
