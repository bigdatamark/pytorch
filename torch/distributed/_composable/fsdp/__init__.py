# [PACKAGE DOC] Backward-compatibility package re-exporting the FSDP policies (CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy) and fully_shard API (FSDPModule, fully_shard, register_fsdp_forward_method) from torch.distributed.fsdp.
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy

from .fully_shard import FSDPModule, fully_shard, register_fsdp_forward_method
