# [MODULE DOC] Hooks letting FSDP compose with tensor parallelism and other tensor subclasses.
# Defines the abstract FSDPExtensions interface, a module-level global holding the active
# extension (None by default), a setter, and _ext_* dispatch helpers that call the extension
# if present or fall back to default sharded-tensor/DTensor utilities from _shard_utils.
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed.fsdp._shard_utils import (
    _all_gather_dtensor,
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)
from torch.distributed.tensor import DeviceMesh, DTensor


class FSDPExtensions(ABC):
    # [CLASS DOC] Abstract interface of customizable FSDP hooks for tensor parallelism.
    # Implementers provide pre/post flatten transforms, tensor and DTensor chunking, state dict
    # pre-load transform and DTensor all-gather. Activated via _set_fsdp_extensions.
    """
    This enables some customizable hooks to enable composability with tensor
    parallelism. To activate these hooks, use :func:`_set_fsdp_extensions` to
    set a custom :class:`FSDPExtensions` that implements the hooks.
    """

    @abstractmethod
    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        # [FUNCTION DOC] Abstract: transform a tensor before it is flattened; returns the new tensor and
        # an optional extension object needed later to undo the transform.
        """E.g. converting ``DistributedTensor`` to local tensor."""
        ...

    @abstractmethod
    def post_unflatten_transform(
        self,
        tensor: torch.Tensor,
        param_extension: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Abstract: inverse of the pre-flatten transform; rebuild the original tensor kind
        # from a local tensor and the saved param_extension.
        """E.g. converting local tensor to ``DistributedTensor``."""
        ...

    @abstractmethod
    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Abstract: shard a tensor into chunks across the process group and return the
        # local chunk for this rank.
        """Shards a tensor to chunks and returns the local chunk."""
        ...

    @abstractmethod
    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Abstract: shard a tensor or DTensor over the device mesh and return the local
        # DTensor for this rank.
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
        ...

    @abstractmethod
    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Shard]]:
        # [FUNCTION DOC] Abstract: called before loading a sharded state dict; returns the tensor and the
        # list of local shards to load data from.
        """
        This is to be called before loading a *sharded* model state dict and
        should return the tensor and list of shards from which to load data.
        """
        ...

    @abstractmethod
    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Abstract: all-gather a DTensor along the FSDP mesh dimension and return the local
        # tensor of the remaining (tensor parallel) DTensor.
        """
        This is to be called before loading a *sharded* DTensor state dict.
        This gathers tensor in FSDP dimension and returns local tensor of
        TP DTensor.
        """
        ...


_extensions: FSDPExtensions | None = None


def _set_fsdp_extensions(flattener: FSDPExtensions) -> None:
# [FUNCTION DOC] HELPER: install the given FSDPExtensions object as the module-level global
# extension (mutates global state; no validation).
    global _extensions
    _extensions = flattener


def _ext_pre_flatten_transform(
    tensor: torch.Tensor,
    fsdp_extension: FSDPExtensions | None = None,
) -> tuple[torch.Tensor, Any | None]:
    # [FUNCTION DOC] HELPER: apply the extension's pre-flatten transform if an extension is given.
    # Returns (new tensor, extension data) only when the extension returned non-None data;
    # otherwise returns the original tensor and None.
    if fsdp_extension is not None:
        new_tensor, param_extension = fsdp_extension.pre_flatten_transform(tensor)
        if param_extension is not None:
            return new_tensor, param_extension
    return tensor, None


def _ext_post_unflatten_transform(
    tensor: torch.Tensor,
    param_extension: Any,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: apply the extension's post-unflatten transform if both an extension and
    # param_extension are non-None; otherwise return the tensor unchanged.
    if fsdp_extension is not None and param_extension is not None:
        return fsdp_extension.post_unflatten_transform(tensor, param_extension)
    return tensor


def _ext_chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: chunk a tensor using the extension's chunk_tensor, or the default
    # sharded-tensor chunking when no extension is given. Returns the local chunk.
    chunk_tensor_fn = (
        fsdp_extension.chunk_tensor
        if fsdp_extension is not None
        else _create_chunk_sharded_tensor
    )
    return chunk_tensor_fn(
        tensor,
        rank,
        world_size,
        num_devices_per_node,
        pg,
    )


def _ext_chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: chunk to a DTensor using the extension's chunk_dtensor, or the default
    # DTensor chunking when no extension is given.
    chunk_dtensor_fn = (
        fsdp_extension.chunk_dtensor
        if fsdp_extension is not None
        else _create_chunk_dtensor
    )
    return chunk_dtensor_fn(
        tensor,
        rank,
        device_mesh,
    )


def _ext_pre_load_state_dict_transform(
    tensor: torch.Tensor,
    fsdp_extension: FSDPExtensions | None = None,
) -> tuple[torch.Tensor, list[Shard]]:
    # [FUNCTION DOC] HELPER: return (tensor, local shards) for state dict loading. Delegates to the
    # extension if given; otherwise requires an exact ShardedTensor and returns its local shards.
    # Raises AssertionError if the tensor is not exactly a ShardedTensor.
    if fsdp_extension is not None:
        return fsdp_extension.pre_load_state_dict_transform(tensor)

    if type(tensor) is not ShardedTensor:
        raise AssertionError(f"Expected ShardedTensor, got {type(tensor)}")
    shards = tensor.local_shards()
    return (tensor, shards)


def _ext_all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: DeviceMesh | None,
    fsdp_extension: FSDPExtensions | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: all-gather a DTensor via the extension, or the default utility when no
    # extension is given. Returns the gathered local tensor.
    all_gather_dtensor_fn = (
        fsdp_extension.all_gather_dtensor
        if fsdp_extension is not None
        else _all_gather_dtensor
    )
    return all_gather_dtensor_fn(tensor, parent_mesh)
