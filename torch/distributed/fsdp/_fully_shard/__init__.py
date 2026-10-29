# [PACKAGE DOC] Public entry point of the per-parameter-sharding FSDP (fully_shard) package.
# Re-exports the policy dataclasses (MixedPrecisionPolicy, OffloadPolicy, CPUOffloadPolicy,
# DataParallelMeshDims) from _fsdp_api and the main API (fully_shard, FSDPModule,
# register_fsdp_forward_method, share_comm_ctx, UnshardHandle) from _fully_shard.
# The names are listed in the package export list.
from ._fsdp_api import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from ._fully_shard import (
    FSDPModule,
    fully_shard,
    register_fsdp_forward_method,
    share_comm_ctx,
    UnshardHandle,
)


__all__ = [
    "CPUOffloadPolicy",
    "DataParallelMeshDims",
    "FSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
    "share_comm_ctx",
]
