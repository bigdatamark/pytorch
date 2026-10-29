# Copyright (c) Meta Platforms, Inc. and affiliates

# [MODULE DOC] Helpers to load FSDP/TP-sharded optimizer state from a distributed checkpoint.
# Exports load_sharded_optimizer_state_dict, which allocates empty sharded/plain tensors for every
# optimizer entry found in checkpoint metadata and fills them via load_state_dict.
# Private helpers compute per-rank device placement, a column-wise ChunkShardingSpec, the 2D (TP slice)
# layout of a model state dict, and a load planner that applies TP offsets.
# No import-time side effects besides defining STATE_DICT_2D_LAYOUT.

import dataclasses
from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch._utils import _get_device_module
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as ShardTensorProperties,
)
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.checkpoint._nested_dict import unflatten_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import (
    _create_read_items,
    create_read_items_for_chunk_list,
)

# pyrefly: ignore [deprecated]
from torch.distributed.checkpoint.state_dict_loader import load_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.checkpoint.utils import (
    _element_wise_add,
    _element_wise_sub,
    _normalize_device_info,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DTensor


STATE_DICT_2D_LAYOUT = dict[str, tuple[Sequence[int] | None, Sequence[int]]]


# TODO: Update docstrings for optimizer.py
__all__ = [
    "load_sharded_optimizer_state_dict",
]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    # [FUNCTION DOC] HELPER: Map a global rank to a device string. Returns "cpu" for cpu device type or
    # when the device module is unavailable; otherwise a normalized device string using the rank
    # modulo the local device count.
    if device_type == "cpu":
        return "cpu"
    device_module = _get_device_module(device_type)
    if device_module.is_available():
        return _normalize_device_info(
            device_type, global_rank % device_module.device_count()
        )
    return "cpu"


def _create_colwise_spec(
    pg: dist.ProcessGroup | None = None,
) -> ChunkShardingSpec:
    # [FUNCTION DOC] HELPER: Build a ChunkShardingSpec that shards dim 0 across all ranks of the given
    # process group (or the default world when None). Each placement is "rank:<idx>/<device>" where
    # the device is derived from the group's default device type and the rank's global rank.
    pg_device_type = dist.distributed_c10d._get_pg_default_device(pg).type
    if pg is None:
        placements = [
            f"rank:{idx}/{_gen_rank_device(idx, pg_device_type)}"
            for idx in range(dist.get_world_size())
        ]
    else:
        placements = [
            f"rank:{idx}/{_gen_rank_device(dist.get_global_rank(pg, idx), pg_device_type)}"
            for idx in range(pg.size())
        ]
    return ChunkShardingSpec(
        dim=0,
        placements=cast(list[_remote_device | str], placements),
    )


def _is_nested_tensor(val: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: True if val is a ShardedTensor whose first local shard is itself a
    # ShardedTensor (2D FSDP+TP layout). False for empty or plain values.
    # Raises ValueError for a DTensor nested in a ShardedTensor, or a DTensor wrapping a DTensor or
    # ShardedTensor.
    if type(val) is ShardedTensor:
        if len(val.local_shards()) == 0:
            return False
        if type(val.local_shards()[0].tensor) is ShardedTensor:
            return True
        if type(val.local_shards()[0].tensor) is DTensor:
            raise ValueError("Cannot handle DTensor nested inside ShardedTensor")
    elif type(val) is DTensor and (
        type(val._local_tensor) is DTensor or type(val._local_tensor) is ShardedTensor
    ):
        raise ValueError("Cannot handle nested DTensor")
    return False


def _alloc_tensor(
    props: TensorProperties, size: Sequence[int], device_type: str = "cuda"
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Allocate an uninitialized tensor of the given size on the current device of
    # device_type, copying dtype, layout, requires_grad and pin_memory from props.
    # KNOWN ISSUE: for cpu the "device" is whatever the cpu module's current_device returns, cast
    # to a device without conversion.
    if device_type == "cpu":
        device = cast(torch.device, _get_device_module(device_type).current_device())
    else:
        device = torch.device(
            device_type, _get_device_module(device_type).current_device()
        )

    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device=device,
    )


def _get_state_dict_2d_layout(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_2D_LAYOUT, dist.ProcessGroup | None]:
    # [FUNCTION DOC] HELPER: Inspect a model state dict and return a mapping from key to (offset, size)
    # of the local TP slice (offset None for non-nested entries, where size is the tensor size),
    # plus the data-parallel process group taken from the last nested ShardedTensor (None if none).
    # Raises AssertionError if a nested tensor has several local shards or is not a ShardedTensor.
    """
    Load the right TP slice of the optimizer state.

    This is not easy since the per-tensor slicing can't be inferred from checkpoint metadata.
    We take advantage of the model state_dict producing a sliced ST to figure out what we need to load.
    This is pretty fragile and it might be easier for FSDP to compute this info for us.
    Returns a dictionary where keys are the same of the state_dict and the value is a tuple of
    (offset, size) for the current rank TP slice.
    N.B. The state_dict *MUST* come from FSDP.sharded_state_dict.
    """
    specs: STATE_DICT_2D_LAYOUT = {}
    dp_pg: dist.ProcessGroup | None = None
    for key, value in state_dict.items():
        specs[key] = (None, value.size())
        if _is_nested_tensor(value):
            if not len(value.local_shards()) == 1:
                raise AssertionError("Cannot handle ST with multiple shards")
            if not isinstance(value, ShardedTensor):
                raise AssertionError("Can only handle nested ShardedTensor")
            shard = value.local_shards()[0]
            specs[key] = (
                shard.metadata.shard_offsets,
                shard.metadata.shard_sizes,
            )
            dp_pg = shard.tensor._process_group  # type: ignore[attr-defined]

    return (
        specs,
        dp_pg,
    )


class _ReaderWithOffset(DefaultLoadPlanner):
    # [CLASS DOC] HELPER: DefaultLoadPlanner variant that reads each ShardedTensor shard from a
    # displaced region of the checkpoint tensor given by fqn_to_offset (the TP slice offset).
    # State: fqn_to_offset (input), metadata and state_dict (set by the base planner), and
    # translation, a mapping from displaced MetadataIndex back to the original index, rebuilt on
    # every create_local_plan and consulted in lookup_tensor.
    translation: dict[MetadataIndex, MetadataIndex]
    state_dict: STATE_DICT_TYPE
    # pyrefly: ignore [bad-override]
    metadata: Metadata

    def __init__(self, fqn_to_offset: dict[str, Sequence[int]]) -> None:
        # [FUNCTION DOC] Store the fqn-to-offset mapping and initialize empty metadata, state_dict
        # and translation.
        super().__init__()
        self.fqn_to_offset = fqn_to_offset
        self.metadata = Metadata({})
        self.state_dict = {}
        self.translation = {}

    def create_local_plan(self) -> LoadPlan:
        # [FUNCTION DOC] Build the LoadPlan. Entries that are not ShardedTensors or have no offset get
        # normal read items. For offset entries, the single local shard's chunk is shifted by the
        # offset to create read items, and each displaced dest index is recorded in translation
        # mapped to the un-displaced index. Raises AssertionError if the shard count is not one or
        # a dest offset is missing. Resets translation as a side effect.
        requests = []
        self.translation = {}
        for fqn, obj in self.state_dict.items():
            md = self.metadata.state_dict_metadata[fqn]
            if not isinstance(obj, ShardedTensor):
                requests += _create_read_items(fqn, md, obj)
                continue

            if fqn not in self.fqn_to_offset:
                requests += _create_read_items(fqn, md, obj)
                continue

            offset = self.fqn_to_offset[fqn]

            if not len(obj.local_shards()) == 1:
                raise AssertionError("Expected exactly one local shard")
            original_shard = obj.local_shards()[0]
            local_chunks = [
                ChunkStorageMetadata(
                    offsets=torch.Size(
                        _element_wise_add(original_shard.metadata.shard_offsets, offset)
                    ),
                    sizes=torch.Size(original_shard.metadata.shard_sizes),
                )
            ]

            reqs = create_read_items_for_chunk_list(
                fqn, cast(TensorStorageMetadata, md), local_chunks
            )
            # TODO: The ReadItems will have a displaced MetadataIndex, fix it.
            # TODO: we should change _create_sharded_read_items to have more ergonomic API
            for ri in reqs:
                if ri.dest_index.offset is None:
                    raise AssertionError("dest_index.offset must not be None")
                original_offset = _element_wise_sub(ri.dest_index.offset, offset)
                original_index = dataclasses.replace(
                    ri.dest_index, offset=torch.Size(original_offset)
                )
                self.translation[ri.dest_index] = original_index

            requests += reqs
        return LoadPlan(requests)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        # [FUNCTION DOC] Resolve the destination tensor, first translating a displaced index back to
        # the original one when present in translation.
        return super().lookup_tensor(self.translation.get(index, index))


def load_sharded_optimizer_state_dict(
    model_state_dict: STATE_DICT_TYPE,
    optimizer_key: str,
    storage_reader: StorageReader,
    planner: LoadPlanner | None = None,
) -> STATE_DICT_TYPE:
    # [FUNCTION DOC] Read checkpoint metadata, and for every entry whose planner path starts with
    # optimizer_key allocate a destination: a placeholder string for bytes, a plain tensor for
    # single-element tensors, a chunk-sharded tensor when no TP process group exists, or a
    # ShardedTensor built from the TP layout (recording TP offsets) otherwise. Then calls
    # load_state_dict (using _ReaderWithOffset when a TP group exists, else the given planner),
    # unflattens the result with the saved planner data and returns it.
    # Args: model_state_dict (must come from FSDP sharded state dict), optimizer_key,
    # storage_reader, optional planner. Side effects: allocates tensors, performs collective loads.
    """
    Load a state_dict in conjunction with FSDP sharded optimizer state.

    This is the current recommended way to checkpoint FSDP.

    Examples::

    >>> # xdoctest: +SKIP
    >>> import torch.distributed.checkpoint as dist_cp
    >>> # Save
    >>> model: torch.nn.Model
    >>> optim_params = model.parameters()
    >>> optim = torch.optim.SGD(optim_params, lr=0.01)
    >>> # Save
    >>> with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    >>>     state_dict = {
    >>>         "optimizer": FSDP.optim_state_dict(model, optim),
    >>>         "model": model.state_dict()
    >>>     }
    >>>     dist_cp.save_state_dict(
    >>>         state_dict=optim_state,
    >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
    >>>         planner=dist_cp.DefaultSavePlanner(),
    >>>     )
    >>>
    >>> # Load
    >>> with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
    >>>     model_state_dict = model_tp.state_dict()
    >>>     checkpoint = {
    >>>         "model": model_state_dict
    >>>     }
    >>>     dist_cp.load_state_dict(
    >>>         state_dict=checkpoint,
    >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
    >>>         planner=dist_cp.DefaultLoadPlanner(),
    >>>     )
    >>>     model.load_state_dict(checkpoint["model_state"])
    >>>
    >>>     optim_state = dist_cp.load_sharded_optimizer_state_dict(
    >>>         model_state_dict,
    >>>         optimizer_key="optimizer",
    >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
    >>>     )
    >>>
    >>>     flattened_osd = FSDP.optim_state_dict_to_load(
    >>>        model, optim, optim_state["optimizer"]
    >>>     )
    >>>
    >>>     optim.load_state_dict(flattened_osd)
    """
    metadata = storage_reader.read_metadata()

    layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
    dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
    device_module = _get_device_module(dp_pg_device_type)

    if dp_pg is None:
        placements = []
        for i in range(dist.get_world_size()):
            device_info = _normalize_device_info(
                dp_pg_device_type, i % device_module.device_count()
            )
            placements.append(f"rank:{i}/{device_info}")
        sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
    else:
        sharding_spec = _create_colwise_spec(dp_pg)

    # Create a state_dict for optimizer state
    state_dict: STATE_DICT_TYPE = {}

    fqn_to_offset: dict[str, Sequence[int]] = {}
    for key, value in metadata.state_dict_metadata.items():
        key_path = metadata.planner_data[key]
        if key_path[0] != optimizer_key:
            continue

        if isinstance(value, BytesStorageMetadata):
            state_dict[key] = "<bytes_io>"
            continue

        # value: TensorStorageMetadata
        if value.size.numel() == 1:
            state_dict[key] = _alloc_tensor(
                value.properties, value.size, dp_pg_device_type
            )
        elif dp_pg is None:
            state_dict[key] = _create_chunk_sharded_tensor(
                _alloc_tensor(value.properties, value.size, dp_pg_device_type),
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                num_devices_per_node=device_module.device_count(),
                pg=_get_default_group(),
            )
        else:
            spec_key = key_path[2]
            alloc_size = layout_specs.get(spec_key, (None, value.size))[1]

            properties = ShardTensorProperties(
                dtype=value.properties.dtype,
                layout=value.properties.layout,
                requires_grad=value.properties.requires_grad,
                memory_format=value.properties.memory_format,
                pin_memory=value.properties.pin_memory,
            )

            st_md = sharding_spec.build_metadata(torch.Size(alloc_size), properties)
            local_shards = []
            current_rank = dist.get_rank(dp_pg)
            for shard_md in st_md.shards_metadata:
                if cast(_remote_device, shard_md.placement).rank() != current_rank:
                    continue
                local_shards.append(
                    Shard(
                        tensor=_alloc_tensor(
                            value.properties, shard_md.shard_sizes, dp_pg_device_type
                        ),
                        metadata=shard_md,
                    )
                )

            st = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = cast(Sequence[int], layout_specs[spec_key][0])

            state_dict[key] = st

    # Whether we unflatten before or after doesn't matter
    load_state_dict(
        state_dict=state_dict,
        storage_reader=storage_reader,
        # FIXME the type of planner is wrong in load_state_dict
        planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
    )

    state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

    return state_dict
