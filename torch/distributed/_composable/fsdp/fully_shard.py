# [MODULE DOC] Backward-compatibility shim: re-exports FSDPModule, fully_shard,
# register_fsdp_forward_method and UnshardHandle from torch.distributed.fsdp. No logic.
# TODO: For backward compatibility, we are importing the public objects
# originally from this file.
from torch.distributed.fsdp import (  # noqa: F401
    FSDPModule,
    fully_shard,
    register_fsdp_forward_method,
    UnshardHandle,
)
