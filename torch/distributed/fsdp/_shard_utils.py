# mypy: allow-untyped-defs
# [MODULE DOC] Helpers for FSDP state-dict sharding: build ShardedTensor or DTensor
# objects by chunking a tensor along dim 0 across ranks, and all-gather a DTensor
# back into a local tensor. No import-time side effects beyond imports.
import copy
import itertools
import math

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard as DShard


def _get_remote_device_str(rank, device_type, num_devices_per_node):
    # [FUNCTION DOC] HELPER: Build a placement string "rank:R/device" for a rank. CPU has no
    # index; HPU uses the current device index of the HPU module; other devices use rank
    # modulo num_devices_per_node as the index. Returns the string.
    if device_type.lower() == "cpu":
        return f"rank:{rank}/{device_type}"
    elif device_type.lower() == "hpu":
        return f"rank:{rank}/{device_type}:{_get_device_module(device_type).current_device()}"
    else:
        return f"rank:{rank}/{device_type}:{rank % num_devices_per_node}"


def _create_chunk_sharded_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    device: torch.device | None = None,
) -> ShardedTensor:
    # [FUNCTION DOC] HELPER: Chunk a tensor along dim 0 into world_size pieces and build a
    # ShardedTensor whose metadata is computed locally (no communication). The local rank
    # gets a cloned chunk as its only shard (none if there are fewer chunks than ranks).
    # Placements come from the global ranks in pg and the device type (default device of
    # pg unless device is given). Raises AssertionError on inconsistent metadata lengths.
    """
    Shard a tensor to chunks along the first dimension. The local rank will get its
    corresponding chunk as the local shard to create a ShardedTensor.
    """
    chunks = tensor.chunk(world_size, dim=0)
    if len(chunks) > rank:
        local_shard = chunks[rank].clone()
        offsets = [0 for _ in tensor.size()]
        offsets[0] = math.ceil(tensor.size()[0] / world_size) * rank
        local_shards = [Shard.from_tensor_and_offsets(local_shard, offsets, rank)]
    else:
        local_shards = []

    # Create a ShardedTensor without invoking communication.
    chunk_sizes = [list(chunk.size()) for chunk in chunks]
    dim0_offsets = [0] + list(
        itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes])
    )[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    device_type = (
        distributed_c10d._get_pg_default_device(pg).type
        if device is None
        else device.type
    )
    placements = [
        _get_remote_device_str(
            dist.get_global_rank(pg, r),
            device_type,
            num_devices_per_node,
        )
        for r in range(len(chunk_sizes))
    ]
    if len(chunk_sizes) != len(chunk_offsets) or len(chunk_sizes) != len(placements):
        raise AssertionError(
            f"Expected chunk_sizes, chunk_offsets, and placements to have the same length, "
            f"got {len(chunk_sizes)}, {len(chunk_offsets)}, {len(placements)}"
        )
    shard_metadata = [
        ShardMetadata(offset, size, placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=pg
    )


def _create_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    # [FUNCTION DOC] HELPER: Make a DTensor from a full tensor by detaching and cloning it,
    # wrapping it as replicated on every mesh dim, then redistributing so the last mesh dim
    # is sharded on tensor dim 0 (FSDP: shard; HSDP: replicate then shard). Involves
    # a redistribute collective-free local chunking. Returns the DTensor.
    """
    Shard a tensor to chunks along the first dimension. The local rank will get its
    corresponding chunk as the local tensor to create a DTensor.
    """
    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    tensor = tensor.detach().clone()

    # FSDP placements: [Shard(0)]
    # HSDP placements: [Replicate(), Shard(0)]
    replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
    shard_placements[-1] = DShard(0)  # type: ignore[call-overload]

    return DTensor.from_local(
        tensor, device_mesh, replicate_placements, run_check=False
    ).redistribute(
        placements=shard_placements,
    )


def _all_gather_dtensor(
    tensor: DTensor,
    root_mesh: DeviceMesh | None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: All-gather a DTensor along its last mesh dim by redistributing that
    # placement to replicate, then return the resulting local tensor. Raises AssertionError
    # if the tensor's device mesh is not root_mesh. Runs a collective.
    """
    All gather a DTensor in its sharded dimension and return the local tensor.
    """
    if root_mesh != tensor.device_mesh:
        raise AssertionError("The device mesh of a tensor should be a root mesh.")

    placements = list(copy.deepcopy(tensor.placements))
    # FSDP placements: [Shard(0)] -> [Replicate()]
    # HSDP placements: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
    placements[-1] = Replicate()
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    return tensor.to_local()
