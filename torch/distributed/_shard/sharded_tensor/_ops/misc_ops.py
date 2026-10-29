# mypy: allow-untyped-defs
# [MODULE DOC] Registers a ShardedTensor handler for the internal compatible-shallow-copy-type check. Registration happens at import.
import torch
from torch.distributed._shard.sharded_tensor import _sharded_op_impl


# This is used by `_apply()` within module.py to set new
# parameters after apply a certain method, we should follow
# the future behavior of overwriting the existing tensor
# instead of doing in-place change using `.data = `.
@_sharded_op_impl(torch._has_compatible_shallow_copy_type)
def tensor_has_compatible_shallow_copy_type(types, args=(), kwargs=None, pg=None):
    # [FUNCTION DOC] Always returns False so module parameter application replaces tensors instead of modifying them in place.
    return False
