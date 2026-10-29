# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC] HELPER: Flattening of nested ShardedTensor entries (as produced by FSDP sharded state dicts) for checkpointing.
# Contains _flatten_sharded_tensors. No import-time side effects.

import copy
from typing import TYPE_CHECKING

import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensor, ShardMetadata
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.remote_device import _remote_device

from ._traverse import OBJ_PATH, set_element, STATE_DICT_ITEM, traverse_state_dict
from .utils import _element_wise_add, _normalize_device_info


if TYPE_CHECKING:
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata


# TODO: We need to refactor this code.
def _flatten_sharded_tensors(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    # [FUNCTION DOC] HELPER: Returns a new state dict in which any ShardedTensor whose single local shard is itself a ShardedTensor
    # is replaced by a flat ShardedTensor. Offsets of the inner shard are added to the outer shard offsets, global metadata
    # is copied with the outer shard removed and the other shards attributed to another rank, and the result is built from
    # the local shard plus that metadata. The resulting tensors are only valid for checkpointing. Other values are copied by
    # reference; tensors with no local shards are dropped; multi-shard tensors pass through. Raises ValueError if the inner tensor has more than one local shard.
    # Requires an initialized process group (uses the rank).
    r"""
    Transform ``state_dict`` by flattening all nested ShardedTensor instances found.

    The resulting ShardedTensor instances are only correct regarding the local shard and
    MUST not be used for any other purpose but checkpointing, as no operator will work with them.

    This function should be used in conjunction with a state_dict produced by FSDP's
    StateDictType.SHARDED_STATE_DICT methods.
    """
    new_state_dict: STATE_DICT_TYPE = {}

    def rewrite_dict(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        # [FUNCTION DOC] HELPER: Traversal visitor that writes either the original value or the flattened ShardedTensor into the new state dict at the given path.
        if not isinstance(value, ShardedTensor):
            set_element(new_state_dict, path, value)
            return
        shards = value.local_shards()

        if len(shards) == 0:
            return
        if len(shards) != 1:
            set_element(new_state_dict, path, value)
            return

        outer_shard = shards[0]

        inner_st = outer_shard.tensor
        if not isinstance(inner_st, ShardedTensor):
            set_element(new_state_dict, path, value)
            return

        if len(inner_st.local_shards()) != 1:
            raise ValueError("Cannot handle inner tensor with more than 1 shard")
        inner_shard = inner_st.local_shards()[0]

        local_shards = [
            Shard(
                tensor=inner_shard.tensor,
                metadata=ShardMetadata(
                    shard_offsets=_element_wise_add(
                        outer_shard.metadata.shard_offsets,
                        inner_shard.metadata.shard_offsets,
                    ),
                    shard_sizes=inner_shard.metadata.shard_sizes,
                    placement=f"rank:{dist.get_rank()}/{inner_shard.tensor.device}",
                ),
            )
        ]

        st_meta: ShardedTensorMetadata = copy.deepcopy(value.metadata())
        other_rank = 0 if dist.get_rank() > 0 else 1
        device_info = _normalize_device_info(inner_shard.tensor.device.type, 0)

        # Remove the outer ST shard the inner ST covers
        for i, shard_md in enumerate(st_meta.shards_metadata):
            if shard_md.shard_offsets == outer_shard.metadata.shard_offsets:
                st_meta.shards_metadata.pop(i)
                break

        # Attribute other rank for the other shards
        for shard_md in st_meta.shards_metadata:
            shard_md.placement = _remote_device(f"rank:{other_rank}/{device_info}")

        # Add other inner shards from the inner tensor
        for inner_md in inner_st.metadata().shards_metadata:
            if inner_md.shard_offsets != inner_shard.metadata.shard_offsets:
                st_meta.shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=_element_wise_add(
                            outer_shard.metadata.shard_offsets,
                            inner_md.shard_offsets,
                        ),
                        shard_sizes=inner_md.shard_sizes,
                        placement=f"rank:{other_rank}/{device_info}",
                    )
                )

        # Finally add this shard
        st_meta.shards_metadata.append(local_shards[0].metadata)

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=local_shards,
            sharded_tensor_metadata=st_meta,
        )
        set_element(new_state_dict, path, st)

    traverse_state_dict(state_dict, rewrite_dict)
    return new_state_dict
