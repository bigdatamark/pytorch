# mypy: allow-untyped-defs
# [MODULE DOC] Core sharding spec abstractions: PlacementSpec, DevicePlacementSpec, the abstract ShardingSpec,
# a registry of per-spec custom op overrides (with registration decorator and dispatch helpers),
# EnumerableShardingSpec (explicit list of shards), and a helper inferring a spec from shard metadata.
# Import-time side effect: creates the empty module-level custom-op registry. ShardedTensor is imported only
# for type checking to avoid a circular import.
import functools
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.op_registry_utils import _decorator_func

from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata,
)


if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor


class PlacementSpec(ABC):  # noqa: B024
    # [CLASS DOC] Abstract base class for describing the placement of an entity; has no members. Subclass for
    # custom placements.
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """


@dataclass
class DevicePlacementSpec(PlacementSpec):
    # [CLASS DOC] Dataclass placing an entity on one device. Field device is a remote device; non-remote values
    # (such as strings) are converted on construction.
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """

    device: torch.distributed._remote_device

    def __post_init__(self):
        # [FUNCTION DOC] Post-init: if device is not already a remote device object, converts it to one (may raise if the
        # value is not a valid remote device description).
        if not isinstance(self.device, torch.distributed._remote_device):
            self.device = torch.distributed._remote_device(self.device)


class ShardingSpec(ABC):
    # [CLASS DOC] Abstract base class for sharding specifications. Subclasses must implement build_metadata (compute
    # ShardedTensorMetadata for a global size) and shard (scatter a global tensor into a ShardedTensor).
    """
    Base class representing sharding specifications.
    """

    @abstractmethod
    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        # [FUNCTION DOC] Abstract: given a global tensor size and tensor properties, returns the ShardedTensorMetadata
        # describing how the tensor is laid out across ranks.
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """

    @abstractmethod
    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> "ShardedTensor":
        """
        # [FUNCTION DOC] Abstract: shards a full tensor held on src_rank (default 0) across the ranks of process_group
        # (default group when unset) and returns a ShardedTensor.
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        Keyword args:
            src_rank (int, optional): The source rank which is used as the ground truth of
                the data for the parameter that would be sharded and scattered
                across the rest of the ranks.
                Default: 0.
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
        Returns:
            A :class:`ShardedTensor` sharded from the given tensor.
        """


# Ops customized for a particular ShardingSpec.
_CUSTOM_SHARDING_SPEC_OPS: dict[str, dict[Callable, Callable]] = {}


def _has_custom_op(sharding_spec, op):
    """
    # [FUNCTION DOC] HELPER: Returns True if a custom op override is registered for the spec's class name and the
    # given op in the module-level registry; False otherwise.
    Returns whether or not the ShardingSpec has a custom op implementation.
    """
    class_name = type(sharding_spec).__qualname__
    return (
        class_name in _CUSTOM_SHARDING_SPEC_OPS
        and op in _CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


def _dispatch_custom_op(
    sharding_spec, op: Callable, types, args, kwargs, process_group
):
    # [FUNCTION DOC] HELPER: Calls the registered custom op for the spec's class with (types, args, kwargs,
    # process_group) and returns its result. Raises RuntimeError if no override is registered.
    """
    Calls the custom op for this ShardingSpec if it exists.
    """
    class_name = type(sharding_spec).__qualname__
    if not _has_custom_op(sharding_spec, op):
        raise RuntimeError(f"Custom op: {op} not registered for {class_name}")
    func = _CUSTOM_SHARDING_SPEC_OPS[class_name][op]
    return func(types, args, kwargs, process_group)


def custom_sharding_spec_op(sharding_spec_class, func):
    # [FUNCTION DOC] Returns a decorator factory for registering a custom implementation of op func for the given
    # sharding spec class. Side effect: creates the registry entry for the class name if absent. The returned
    # partial applies the shared op registration helper to that class's op table. Note the registry is keyed by
    # class qualified name.
    """
    Decorator to allow custom registration of ops.
    Args:
        sharding_spec_class(type): The ShardingSpec for which we need to add this custom op.
        func(Callable): The op to override (ex: torch.bmm)
    """
    class_name = sharding_spec_class.__qualname__
    if class_name not in _CUSTOM_SHARDING_SPEC_OPS:
        _CUSTOM_SHARDING_SPEC_OPS[class_name] = {}
    return functools.partial(
        _decorator_func, op=func, op_table=_CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    # [CLASS DOC] Dataclass sharding spec that lists every shard explicitly via a list of ShardMetadata. Validated on
    # construction: the list must be non-empty, all shards must have the same number of dimensions, and shards must
    # not overlap (ValueError). build_metadata checks coverage of the tensor; shard is not implemented.
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """

    shards: list[ShardMetadata]

    def __post_init__(self):
        # [FUNCTION DOC] Post-init validation: raises ValueError for an empty shard list, for shards with differing
        # numbers of dimensions, or when shards overlap.
        if len(self.shards) == 0:
            raise ValueError(f"Empty shard list provided: {self.shards}")

        # Validate each shard has same rank.
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(
                    f"Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}"
                )
            rank = len(shard.shard_offsets)

        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        # [FUNCTION DOC] Verifies the shards fit exactly in tensor_sizes (ValueError otherwise) and returns a
        # ShardedTensorMetadata built from the shards, tensor_sizes and tensor_properties.
        # check if shards form a valid tensor
        check_tensor(self.shards, tensor_sizes)
        return sharded_tensor_meta.ShardedTensorMetadata(
            self.shards, tensor_sizes, tensor_properties
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> "ShardedTensor":
        # [FUNCTION DOC] Not implemented; always raises NotImplementedError.
        # TODO: figure out a generic and efficient way to scatter the shards for EnumerableShardingSpec
        raise NotImplementedError("EnumerableShardingSpec.shard not implemented yet!")


def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    # [FUNCTION DOC] HELPER: Infers a spec from the list of shard metadata. If all non-zero offsets lie in one common
    # dimension, builds a candidate ChunkShardingSpec with placements ordered by offset and compares the sorted shard
    # sizes and offsets along that dimension with a true chunk split of the total length; returns the chunk spec when
    # they match. Otherwise (multi-dimension sharding, mismatch) returns an EnumerableShardingSpec of the metadata.
    # KNOWN ISSUE: the dimension check uses truthiness, so a sharding dimension of 0 is treated as unset and a later
    # shard in another dimension may not be detected as a mismatch.
    Infer the sharding spec from the metadata of each shard of a ShardedTensor.
    If the tensor is sharded only on one dimension, we can then verify whether it's
    a ChunkShardingSpec or not. The way to verify it is to first get the total length
    and perform a chunk sharding with the given placements to see if we can have the
    same chunk size as the given shards_metadata. If not, we assume it's enum sharded.

    Args:
        shards_metadata (List[ShardMetadata]): List of Metadata of local shards.

    Returns:
        A :class:`torch.distributed._shard.sharding_spec.ShardingSpec` object of sharding
            spec for one sharded tensor.
    """
    placements = []
    chunk_sharding_dim = None
    chunk_offset_list = []
    shard_size_list = []
    shard_offset_list = []
    # collect local shard metadata from the global sharded_tensor_metadata
    for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
        placements.append(shard_metadata.placement)
        local_offsets = shard_metadata.shard_offsets
        chunk_offset_list.append(sum(local_offsets))
        shard_size_list.append(shard_metadata.shard_sizes)
        shard_offset_list.append(shard_metadata.shard_offsets)
        shard_dims = [idx for idx, e in enumerate(local_offsets) if e != 0]
        # If the offset is [0, 0, ..., 0] (all zeros),
        # we cannot decide whether how the tensor is sharded.
        if len(shard_dims) == 0:
            continue
        # If the offset is [0, N, .,0, M, 0, .., 0],
        # we are sure it's sharded by more than one dimension.
        if len(shard_dims) != 1:
            chunk_sharding_dim = None
            break
        # If the offset is [0, 0, .,0, M, 0, .., 0], aka, it's sharded by just
        # one dimension, we need to make sure all ranks share the same dimension.
        if not chunk_sharding_dim:
            chunk_sharding_dim = shard_dims[0]
        elif chunk_sharding_dim != shard_dims[0]:
            chunk_sharding_dim = None
            break

    if chunk_sharding_dim is not None:
        # Ensure we infer the correct placement order from offsets
        placements = [
            x
            for _, x in sorted(
                zip(chunk_offset_list, placements), key=operator.itemgetter(0)
            )
        ]

        from .chunk_sharding_spec import ChunkShardingSpec

        chunk_spec = ChunkShardingSpec(
            dim=chunk_sharding_dim,
            placements=placements,
        )

        shard_sizes = sorted([x[chunk_sharding_dim] for x in shard_size_list])
        shard_total_length = sum(shard_sizes)
        shard_offsets = sorted([x[chunk_sharding_dim] for x in shard_offset_list])

        chunks = len(placements)
        split_size = get_split_size(shard_total_length, chunks)
        chunk_shard_sizes = sorted(
            [
                get_chunked_dim_size(shard_total_length, split_size, idx)
                for idx in range(chunks)
            ]
        )
        # Should match ChunkShardingSpec offsets calculation
        chunk_shard_offsets = [split_size * idx for idx in range(chunks)]
        if shard_sizes == chunk_shard_sizes and shard_offsets == chunk_shard_offsets:
            return chunk_spec
    return EnumerableShardingSpec(shards_metadata)
