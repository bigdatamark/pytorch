# mypy: allow-untyped-defs
# [MODULE DOC] FSDP extension for 2-D FSDP + Tensor Parallel. Provides DTensorExtensions, an
# FSDPExtensions implementation that flattens/unflattens DTensor parameters, chunks tensors
# into ShardedTensor or DTensor form, and all-gathers along the FSDP mesh dimension, plus
# private helpers that derive ShardedTensor metadata from a DTensor. Only 1-D device meshes
# are handled by the sharded-tensor helpers. No import-time side effects.
import copy
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)


__all__ = ["DTensorExtensions"]


def _get_box(tensor: DTensor) -> tuple[torch.Size, torch.Size]:
    # [FUNCTION DOC] HELPER: Returns (per-chunk offset step, local tensor size) for a DTensor on
    # a 1-D mesh. The offset step is zero everywhere except the shard dim, where it is the
    # global size divided (integer division) by the number of chunks. Raises AssertionError for
    # non-1-D meshes.
    # KNOWN ISSUE: integer division assumes even sharding; uneven last chunks get wrong offsets.
    device_mesh = tensor.device_mesh
    if device_mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(mesh_dim=0)

    # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
    if tensor.placements[0].is_shard():
        shard_dim = cast(DShard, placement).dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())


def _get_box_for(tensor: DTensor, idx: int) -> tuple[torch.Size, torch.Size]:
    # [FUNCTION DOC] HELPER: Returns (offsets, size) of chunk number idx, i.e. the per-chunk
    # offset step from the box helper multiplied by idx, with the local size unchanged.
    offsets, size = _get_box(tensor)
    return (torch.Size([val * idx for val in offsets]), size)


def _get_local_box(tensor: DTensor) -> tuple[torch.Size, torch.Size]:
    # [FUNCTION DOC] HELPER: Returns (offsets, size) of the calling rank's chunk, using its mesh
    # coordinate along dimension zero. Raises AssertionError if this rank is not in the mesh.
    device_mesh = tensor.device_mesh
    coord = device_mesh.get_coordinate()
    if coord is None:
        raise AssertionError
    return _get_box_for(tensor, coord[0])


def _create_shard_md_from_dt(dt: DTensor, current_rank: int) -> ShardMetadata:
    # [FUNCTION DOC] HELPER: Builds ShardMetadata for the local shard of a DTensor: offsets and
    # sizes of the local box and a placement string naming current_rank and the local tensor's
    # device. Raises AssertionError for non-1-D meshes.
    mesh = dt.device_mesh
    if mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")

    offsets, sizes = _get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}",
    )


def _create_sharded_tensor_md_from_dt(
    dt: DTensor, dt_pg: c10d.ProcessGroup
) -> ShardedTensorMetadata:
    # [FUNCTION DOC] HELPER: Builds global ShardedTensorMetadata for a DTensor with full
    # coverage but only one real shard. Shard count is the group size if dim zero placement is
    # a shard, else one. Chunk zero is assigned to the current rank; all other chunks are
    # assigned to a "scapegoat" rank (rank 0, or rank 1 when the current rank is 0). Size and
    # tensor properties (dtype, layout, requires_grad) come from the DTensor.
    # This is where it gets tricky, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1

    for i in range(shard_count):
        offsets, sizes = _get_box_for(dt, i)
        shards_md.append(
            ShardMetadata(
                shard_offsets=list(offsets),
                shard_sizes=list(sizes),
                placement=(
                    f"rank:{scapegoat_rank if i > 0 else my_rank}/{dt._local_tensor.device}"
                ),
            )
        )

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            # ignore memory_format and pin_memory as those are not supported by DT
        ),
    )


def _get_dt_pg(dt: DTensor) -> c10d.ProcessGroup:
    # [FUNCTION DOC] HELPER: Returns the process group of the DTensor's 1-D device mesh.
    # Raises AssertionError for non-1-D meshes.
    mesh = dt.device_mesh
    if mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")
    return mesh.get_group()


def _rewrite_spec_if_needed(
    spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int
) -> shard_spec.ShardingSpec:
    # [FUNCTION DOC] HELPER: Returns spec unchanged unless it is a ChunkShardingSpec having a
    # placement for the given rank whose device differs from the tensor's device; in that case
    # returns a deep copy with those placements replaced by the rank on the tensor's device.
    """
    Rewrite ``spec`` to match the device of ``tensor``.

    FSDP.sharded_optim_state_dict sneakily ships optimizer state to CPU so if the original ShardingSpec
    produces CUDA metadata, ST construction bombs.
    """
    if not isinstance(spec, ChunkShardingSpec):
        return spec

    # let's see if we need
    rewrite = False
    for p in spec.placements:
        p = cast(_remote_device, p)
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        # pyrefly: ignore [missing-attribute]
        for i, placement in enumerate(spec.placements):
            placement = cast(_remote_device, placement)
            if placement.rank() == rank and placement.device() != tensor.device:
                # pyrefly: ignore [missing-attribute]
                spec.placements[i] = _remote_device(f"rank:{rank}/{tensor.device}")

    return spec


def _chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Chunks a tensor for FSDP and returns a ShardedTensor. For a
    # ShardedTensor input (must have exactly one local shard, else AssertionError) it chunks
    # the local tensor into an inner ShardedTensor and wraps it as the single shard of an outer
    # ShardedTensor with requires_grad off. For a DTensor on a 1-D mesh it does the same using
    # metadata derived from the DTensor and the accelerator device count, on the DTensor's group.
    # Any other tensor is chunked directly by the shared FSDP shard utility.
    # KNOWN ISSUE: the DTensor branch ignores the num_devices_per_node argument and uses the
    # accelerator device count instead.
    if type(tensor) is ShardedTensor:
        if len(tensor.local_shards()) != 1:
            raise AssertionError

        inner_param = tensor.local_tensor()
        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )

        outer_local_shard = tensor.local_shards()[0]
        shards: list[Shard] = [
            Shard(inner_st, copy.deepcopy(outer_local_shard.metadata))
        ]
        st_meta = copy.deepcopy(tensor.metadata())
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=tensor._process_group,
            init_rrefs=False,
        )
        return st_outer
    elif type(tensor) is DTensor:
        device_mesh = tensor.device_mesh
        if device_mesh.ndim != 1:
            raise AssertionError("Only 1D DeviceMeshes currently handled")

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.accelerator.device_count(),
            pg,
        )

        dt_pg = _get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = [
            Shard(inner_st, _create_shard_md_from_dt(tensor, dist.get_rank(dt_pg)))
        ]

        st_meta = _create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False,
        )

        return st_outer
    else:
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )


def _chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    # [FUNCTION DOC] HELPER: Shards a (possibly TP-sharded) tensor over the FSDP mesh dimension.
    # Uses the root mesh of device_mesh; raises RuntimeError if there is none or it has fewer
    # than two dims. Detaches and clones the tensor. A plain tensor becomes a DTensor with
    # first-dim shard on mesh dim zero and replication elsewhere. A DTensor keeps its TP
    # placement on the last mesh dim and gets a first-dim shard on the second-to-last dim
    # (other dims replicated). Returns the redistributed DTensor. The rank argument is unused.
    """
    Shard a tensor to chunks along the first dimension.

    The local rank will gets its corresponding chunk as the local tensor to create a DTensor.
    """
    root_mesh = device_mesh._get_root_mesh() if device_mesh is not None else None
    if root_mesh is None:
        raise RuntimeError("No parent device_mesh is found for FSDP device_mesh.")
    if root_mesh.ndim < 2:
        raise RuntimeError(
            f"Found parent device_mesh of ndim={root_mesh.ndim},",
            "but meshes must be at least 2D.",
        )

    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    tensor = tensor.detach().clone()

    # When a layer is not involved in TP, then the tensor will not be a DTensor.
    # e.g. When a layer is not specified in the parallelize_plan, TP will have no effect on the layer.
    # e.g. When you do PairwiseParallel on a 3 layer model, TP will have no effect on the third layer.
    if isinstance(tensor, torch.Tensor) and not isinstance(tensor, DTensor):
        # For tensors, it is replicated across tp dimension and sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), Replicate()).
        replicate_placements = [Replicate() for _ in range(root_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(root_mesh.ndim)]
        shard_placements[0] = DShard(0)  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, root_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=root_mesh,
            placements=shard_placements,
        )

    else:
        tp_placements = tensor.placements
        tp_placement = tp_placements[0]

        tensor = tensor.to_local()

        # For DTensors, it is sharded across tp dimension first and then sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), tp_placement).
        # For higher dimensional meshes, it is replicated across other dimensions. For example, with
        # HSDP the shard placements for tensor is (Replicate, Shard(0), tp_placement).
        replicate_placements = [Replicate() for _ in range(root_mesh.ndim)]
        replicate_placements[-1] = tp_placement  # type: ignore[call-overload]
        shard_placements = [Replicate() for i in range(root_mesh.ndim)]  # type: ignore[misc]
        shard_placements[-2] = DShard(0)  # type: ignore[call-overload]
        shard_placements[-1] = tp_placement  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, root_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=root_mesh,
            placements=shard_placements,
        )


def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, list[Shard]]:
    # [FUNCTION DOC] HELPER: Given a ShardedTensor, returns (tensor, local shards). If it has
    # exactly one local shard that is itself a ShardedTensor (nested FSDP+TP form), unwraps to
    # that inner tensor and its local shards. Returns an empty list when there are no shards.
    shards = cast(ShardedTensor, tensor).local_shards()
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        inner_tensor = shards[0].tensor
        shards = inner_tensor.local_shards()  # pyre-ignore[16]
        tensor = inner_tensor

    return (tensor, shards if len(shards) > 0 else [])


def _all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: DeviceMesh | None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Redistributes the DTensor so every mesh dim except the last (the TP
    # dim) is replicated, then returns the local tensor. Raises AssertionError if parent_mesh
    # differs from the tensor's mesh. Performs collective communication.
    """All gather a DTensor in its FSDP dimension and return the local tensor."""
    if parent_mesh != tensor.device_mesh:
        raise AssertionError

    placements = list(copy.deepcopy(tensor.placements))
    # FSDP + TP: [Shard(0), tp_placement] -> [Replicate(), tp_placement]
    # HSDP + TP: [Replicate(), Shard(0), tp_placement] -> [Replicate(), Replicate(), tp_placement]
    for i in range(len(placements) - 1):
        placements[i] = Replicate()
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    return tensor.to_local()


class DTensorExtensions(FSDPExtensions):
    # [CLASS DOC] FSDPExtensions implementation that lets FSDP handle DTensor (TP) parameters.
    # State: device_handle (device module used for streams) and compute_stream (optional, None
    # by default). Methods delegate to module-level helpers: flatten/unflatten DTensors,
    # chunk to ShardedTensor or DTensor, prepare state-dict loading, and all-gather over FSDP dims.
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """

    def __init__(self, device_handle) -> None:
        # [FUNCTION DOC] Stores device_handle, sets compute_stream to None, and replaces the
        # instance's post-unflatten method with a version wrapped to disable Dynamo tracing.
        super().__init__()
        self.compute_stream = None
        self.device_handle = device_handle
        # we have to use the dynamo disable this way to disable dynamo as the decorator way would
        # trigger build failure with torch deploy...
        self.post_unflatten_transform = torch._dynamo.disable(  # type: ignore[method-assign]
            self.post_unflatten_transform
        )

    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        # [FUNCTION DOC] Flattens a parameter: a DTensor becomes (local tensor, spec) with the
        # local tensor marked as requiring grad; a plain tensor becomes (tensor, None).
        return _flatten_tensor(tensor)

    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor:
        # [FUNCTION DOC] Rebuilds a DTensor from the unflattened local tensor and the spec saved
        # at flatten time, under the compute stream (or current stream if unset). Registers a
        # grad hook via the unflatten helper, marks the result as FSDP-flattened, and returns it.
        stream = self.compute_stream or self.device_handle.current_stream()
        with self.device_handle.stream(stream):
            # runtime we put the unflattened tensor call on the compute stream since
            # the unflattened tensor might contain computations in fwd/bwd where we
            # need to sync properly.
            # TODO: this is a short term fix and we should make the get_unflat_views
            # directly happen in the compute stream.
            result = _unflatten_tensor(
                tensor,
                param_extension,
                device_handle=self.device_handle,
                compute_stream=self.compute_stream,
            )
            _set_fsdp_flattened(result)
            return result

    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Delegates to the tensor-chunking helper; the device argument is unused.
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Delegates to the DTensor-chunking helper; returns a DTensor sharded
        # over the FSDP mesh dimension.
        return _chunk_dtensor(tensor, rank, device_mesh)

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Shard]]:
        # [FUNCTION DOC] Delegates to the helper returning (tensor, local shards), unwrapping
        # nested ShardedTensors.
        return _pre_load_state_dict(tensor)

    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Delegates to the helper that replicates all but the last mesh dim and
        # returns the local tensor; raises AssertionError if parent_mesh mismatches.
        return _all_gather_dtensor(tensor, parent_mesh)
