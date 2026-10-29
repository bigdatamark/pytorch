# mypy: allow-untyped-defs
# [MODULE DOC] Shared helpers and small data types for the FSDP fully_shard implementation.
# Contains mesh-info dataclasses (FSDP, DDP, HSDP), the TrainingState enum, dtype-cast and
# chunking utilities, DTensor construction without autograd, and shard-placement resolution.
# No import-time side effects beyond defining names.
import functools
import math
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import _get_registry
from torch.distributed.tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec

from ._fsdp_api import DataParallelMeshDims


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _dynamo_disable(func: Callable[_P, _R]) -> Callable[_P, _R]:
    # [FUNCTION DOC] HELPER: Decorator that wraps a function so each call runs under dynamo disable
    # (recursive, reason "skipping FSDP hooks"), keeping FSDP hooks out of compiled graphs. Returns the wrapper.
    """Disable dynamo tracing for FSDP hooks."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped function under dynamo disable (recursive) and returns its result.
        return torch._dynamo.disable(
            func, recursive=True, reason="skipping FSDP hooks"
        )(*args, **kwargs)

    return wrapper


    # [FUNCTION DOC] HELPER: Decorator that runs the wrapped function with functorch transforms
    # disabled when any are active; otherwise calls it directly. Returns the wrapped function.
def _disable_functorch_if_active(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: If functorch transforms are active, calls the wrapped function with functorch
        # disabled; otherwise calls it directly. Returns its result.
        if torch._C._are_functorch_transforms_active():
            with torch._C._DisableFuncTorch():
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@dataclass
    # [CLASS DOC] Holds the device mesh and which mesh dims are used for sharding and/or replication.
    # Attributes: mesh, shard_mesh_dim, replicate_mesh_dim, optional dp_mesh_dims, optional spmd_mesh
    # (must contain all non-PP SPMD dims), and is_spmd_mesh (derived: true when dp_mesh_dims is given).
    # At least one of shard/replicate mesh dim must be set.
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: int | None = None
    replicate_mesh_dim: int | None = None
    dp_mesh_dims: DataParallelMeshDims | None = None
    # The full SPMD mesh (excluding PP dims) that params are distributed on.
    # Must include all non-PP SPMD dims (e.g. DP + TP); passing a submesh
    # that omits dims like TP will lead to incorrect behavior.
    spmd_mesh: DeviceMesh | None = field(default=None, repr=False)
    is_spmd_mesh: bool = field(default=False, init=False, repr=False)

        # [FUNCTION DOC] Validates that at least one of shard_mesh_dim / replicate_mesh_dim is set (raises
        # AssertionError otherwise) and derives is_spmd_mesh from whether dp_mesh_dims was provided.
    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )
        self.is_spmd_mesh = self.dp_mesh_dims is not None


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    # [CLASS DOC] Mesh info for sharding: requires shard_mesh_dim and exposes the shard mesh size,
    # shard process group, local rank in it, and an optional separate reduce-scatter process group
    # (None by default, meaning reduce-scatter reuses the shard group).
    def __post_init__(self):
        # [FUNCTION DOC] Runs the base validation, requires a non-None shard_mesh_dim (raises AssertionError),
        # and computes shard_mesh_size, shard_process_group, shard_mesh_rank; sets reduce_scatter_process_group to None.
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = self.shard_process_group.rank()
        # Reduce-scatter shares the shard PG (one NCCL communicator) by default;
        # FSDPModule.set_separate_reduce_scatter_group assigns a dedicated PG
        # here so reduce-scatter can overlap all-gather in backward.
        self.reduce_scatter_process_group: dist.ProcessGroup | None = None


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    # [CLASS DOC] Mesh info for replication: requires replicate_mesh_dim and exposes the replicate mesh
    # size, replicate process group, and local rank in that group.
    def __post_init__(self):
        # [FUNCTION DOC] Runs the base validation, requires a non-None replicate_mesh_dim (raises AssertionError),
        # and computes replicate_mesh_size, replicate_process_group, replicate_mesh_rank.
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.size(self.replicate_mesh_dim)
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = self.replicate_process_group.rank()


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    # [CLASS DOC] Mesh info for hybrid sharding: combines FSDPMeshInfo and DDPMeshInfo so both a shard
    # dim and a replicate dim are required and both sets of attributes are populated.
    def __post_init__(self):  # pylint:disable=useless-parent-delegation
        # [FUNCTION DOC] HELPER: Delegates to the parent initializers through the method resolution order
        # (FSDP, then DDP, then base), so both dims are validated and both groups populated.
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()


class TrainingState(Enum):
    # [CLASS DOC] Enum of the per-state / per-parameter-group phases: FORWARD, PRE_BACKWARD,
    # POST_BACKWARD, IDLE.
    """Describes the training state of one FSDP state / parameter group."""

    # Transition to forward starting pre-forward until post-forward
    FORWARD = auto()
    # Transition to pre-backward when unsharding in backward
    PRE_BACKWARD = auto()
    # Transition to post-backward when resharding and reducing gradients
    POST_BACKWARD = auto()
    # Idle before/after forward or before pre-backward/after post-backward
    IDLE = auto()


def _raise_assert_with_print(*args: Any, **kwargs: Any):
    # [FUNCTION DOC] HELPER: Prints the rank and given arguments, prints the current stack, then raises
    # AssertionError with the same arguments (so failures are visible in multi-rank logs).
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)


def _is_composable_with_fsdp(module: nn.Module) -> bool:
    # [FUNCTION DOC] HELPER: Returns False if the module already has the replicate composable API applied
    # (per the contract registry), else True; modules with no registry are composable.
    registry = _get_registry(module)
    if registry is None:
        return True
    # Registry keys by function name
    return "replicate" not in registry


def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    # [FUNCTION DOC] HELPER: Returns the tensor size with dim 0 rounded up to a multiple of dim0_factor,
    # other dims unchanged.
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    return torch.Size([padded_dim0]) + tensor_size[1:]


def _chunk_with_empty(
    # [FUNCTION DOC] HELPER: Like torch.chunk along dim but always returns exactly num_chunks tensors;
    # missing trailing chunks are padded with empty 1-D tensors of the same dtype/device as the first chunk.
    tensor: torch.Tensor, num_chunks: int, dim: int
) -> list[torch.Tensor]:
    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
    while len(chunks) < num_chunks:
        chunks.append(chunks[0].new_empty(0))
    return chunks


def _get_dim_chunked_size(
    chunk: torch.Tensor, unchunked_size: torch.Size, dim: int
) -> torch.Size:
    # [FUNCTION DOC] HELPER: Returns the chunk's own size if it is non-empty; for an empty chunk returns
    # the unchunked size with the chunked dim set to 0 (keeps other dims for DTensor APIs).
    if chunk.numel() > 0:
        return chunk.size()
    # For 0 numel, we need to preserve nonzero-sized dims for DTensor APIs
    # pyrefly: ignore [bad-return]
    return unchunked_size[:dim] + torch.Size([0]) + unchunked_size[dim + 1 :]


def _from_local_no_grad(
    local_tensor: torch.Tensor,
    sharding_spec: DTensorSpec,
) -> DTensor:
    # [FUNCTION DOC] HELPER: Builds a DTensor directly from a local tensor and DTensorSpec, with
    # requires_grad copied from the local tensor. Not differentiable; avoids from_local CPU overhead.
    """
    This method is similar to ``DTensor.from_local()`` except that in eager mode
    it avoids some CPU overhead by avoiding default args and not being differentiable.
    """
    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # Use the local tensor directly instead of constructing a new tensor
        # variable, e.g. with `view_as()`, since this is not differentiable
        # pyrefly: ignore [bad-argument-count]
        local_tensor,
        sharding_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=local_tensor.requires_grad,
    )


def _to_dtype_if_needed(
    tensor: torch.Tensor, dtype: torch.dtype | None
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the tensor cast to dtype when dtype is given and differs; otherwise
    # returns the tensor unchanged.
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Casts x to dtype only if x is a floating-point tensor of a different dtype;
    # non-tensors, non-floating tensors, and matching dtypes are returned unchanged.
    if (
        not isinstance(x, torch.Tensor)
        or not torch.is_floating_point(x)
        or x.dtype == dtype
    ):
        return x
    return x.to(dtype)


def is_bw() -> bool:
    # [FUNCTION DOC] Returns True when called during an autograd backward pass (a graph task is current).
    return torch._C._current_graph_task_id() != -1


@dataclass
class ShardPlacementResult:
    # [CLASS DOC] Result of resolving a shard placement: optional Shard placement (None means default
    # dim-0 sharding) and the FSDPMeshInfo to shard on.
    placement: Shard | None
    mesh_info: FSDPMeshInfo


ShardPlacementFnResult = Shard | ShardPlacementResult | None


def resolve_shard_placement(
    result: ShardPlacementFnResult,
    default_mesh_info: FSDPMeshInfo,
) -> ShardPlacementResult:
    # [FUNCTION DOC] Normalizes a shard_placement_fn result: None gives default placement (None) on the
    # default mesh info; a Shard gives that placement on the default mesh; a ShardPlacementResult is
    # returned as-is. Raises ValueError for any other type.
    """Resolve the shard_placement_fn result to a ShardPlacementResult.

    Handles different input types and applies defaults:
    - None: Use default sharding (Shard(0)) on default mesh
    - Shard: Use specified shard dimension on default mesh
    - ShardPlacementResult: Use as-is

    Args:
        result: The return value from shard_placement_fn, or None if no fn provided.
        default_mesh_info: The default FSDPMeshInfo to use if not specified.

    Returns:
        A ShardPlacementResult with placement and mesh_info.
    """
    if result is None:
        return ShardPlacementResult(placement=None, mesh_info=default_mesh_info)
    if isinstance(result, Shard):
        return ShardPlacementResult(placement=result, mesh_info=default_mesh_info)
    if isinstance(result, ShardPlacementResult):
        return result
    raise ValueError(f"Invalid shard_placement_fn result: {result}")
