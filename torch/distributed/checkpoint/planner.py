# [MODULE DOC] Core data types and abstract planner interfaces for distributed checkpointing.
# Defines enums (WriteItemType, LoadItemType), frozen dataclasses describing work items
# (BytesIOWriteData, TensorWriteData, WriteItem, ReadItem, SavePlan, LoadPlan), and the
# SavePlanner (abstract) and LoadPlanner interfaces that customize save/load planning.
# No import-time side effects.
import abc
import io
import operator
from dataclasses import dataclass
from enum import auto, Enum
from functools import reduce
from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
    TensorProperties,
)


__all__ = [
    "WriteItemType",
    "LoadItemType",
    "BytesIOWriteData",
    "TensorWriteData",
    "WriteItem",
    "ReadItem",
    "SavePlan",
    "LoadPlan",
    "SavePlanner",
    "LoadPlanner",
]


class WriteItemType(Enum):
    # [CLASS DOC] Kind of a write: TENSOR (whole tensor), SHARD (a shard of a sharded tensor),
    # BYTE_IO (serialized non-tensor bytes).
    TENSOR = auto()
    SHARD = auto()
    BYTE_IO = auto()


class LoadItemType(Enum):
    # [CLASS DOC] Kind of a read: TENSOR or BYTE_IO.
    TENSOR = auto()
    BYTE_IO = auto()


@dataclass(frozen=True)
class BytesIOWriteData:
    # [CLASS DOC] Immutable description of a bytes write: nbytes is the number of bytes to write.
    nbytes: int


@dataclass(frozen=True)
class TensorWriteData:
    # [CLASS DOC] Immutable description of a tensor write: the chunk (offsets and sizes) being
    # written, the tensor properties (dtype, layout, ...), and the size of the full global tensor.
    chunk: ChunkStorageMetadata
    properties: TensorProperties
    size: torch.Size


@dataclass(frozen=True)
class WriteItem:
    # [CLASS DOC] Immutable unit of write work: index identifies the entry in the state dict, type is
    # a WriteItemType, and exactly one of bytes_io_data (bytes writes) or tensor_data (tensor and
    # shard writes) is normally set; the other is None.
    """Dataclass which holds information about what needs to be written to storage."""

    index: MetadataIndex
    type: WriteItemType

    # Size of bytesIO data to be written.
    bytes_io_data: BytesIOWriteData | None = None

    # Value present if it's a tensor write
    tensor_data: TensorWriteData | None = None

    def tensor_storage_size(self) -> int | None:
        # [FUNCTION DOC] Return the byte size of the full tensor (product of tensor_data.size times
        # the dtype element size), or None when this item has no tensor data.
        # KNOWN ISSUE: uses the global tensor size, not the chunk size.
        """
        Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

        Returns:
            Optional[int] storage size, in bytes of underlying tensor if any.
        """
        if self.tensor_data is None:
            return None

        numels = reduce(operator.mul, self.tensor_data.size, 1)
        dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)
        return numels * dtype_size


@dataclass(frozen=True)
class ReadItem:
    # [CLASS DOC] Immutable unit of read work: copy a hypercube of size lengths from the checkpoint
    # entry storage_index (starting at storage_offsets) into the state dict entry dest_index
    # (starting at dest_offsets). type is a LoadItemType.
    # Read Item
    type: LoadItemType

    # Index into the state_dict
    dest_index: MetadataIndex
    # Offsets into destination tensor
    dest_offsets: torch.Size

    # Index into the checkpoint
    storage_index: MetadataIndex
    # Offset into the checkpoint data
    storage_offsets: torch.Size

    # Size of the hypercube to copy
    lengths: torch.Size


@dataclass(frozen=True)
class SavePlan:
    # [CLASS DOC] Immutable per-rank save plan: items to write, opaque storage_data and
    # planner_data contributed by the storage writer/planner, and usable, which when False tells
    # ranks not to reuse cached plans.
    items: list[WriteItem]
    storage_data: Any = None
    planner_data: Any = None
    # This is used to indicate that the ranks should
    # use the cached plans to write data instead.
    usable: bool = True


@dataclass
class LoadPlan:
    # [CLASS DOC] Mutable per-rank load plan: list of ReadItem plus opaque storage_data and
    # planner_data.
    items: list[ReadItem]
    storage_data: Any = None
    planner_data: Any = None


class SavePlanner(abc.ABC):
    # [CLASS DOC] Abstract interface for customizing checkpoint saving. Call order: set_up_planner
    # (all ranks), create_local_plan (all ranks), create_global_plan (coordinator only),
    # finish_plan (all ranks), resolve_data (many times per rank). Stateful; acts as a proxy to the
    # state dict. The underscore class attributes are per-key plan caches (local plan, final plan,
    # all plans, global plan, metadata) used for plan caching; they are class-level shared dicts.
    """
    Abstract class defining the protocol used by save_state_dict to plan the save process.

    SavePlanners are stateful objects that can be used to customize the whole save process.

    SavePlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during save_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of a checkpoint save.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `SavePlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the SavePlan from all ranks and make any global decision.

    4) finish_plan - called on all ranks.
        This gives each rank a chance to adjust to global planning decisions.

    5) resolve_data - called multiple times on each rank
        Lookups a value on the `state_dict` for the storage layer to write.

    Users are recommended to extend DefaultSavePlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are 3 usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the save process as it
    doesn't require understanding the intricacies of how SavePlan works:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultSavePlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         storage_meta: Optional[StorageMeta],
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>> # prefix all keys with `foo_``
    >>>         super().set_up_planner({"foo_" + k: v for k, v in state_dict.items()}, storage_meta, is_coordinator)

    Modifying local plan and lookup in tandem. This is useful when fine control of how data is persisted

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class FP16Planner(DefaultSavePlanner):
    >>>     def create_local_plan(self):
    >>>         plan = super().create_local_plan()
    >>>         for p in plan:
    >>>             if p.tensor_data is not None:
    >>>                 p.tensor_data.properties.dtype = torch.float16
    >>>         return plan
    >>>
    >>>     def resolve_data(self, write_item):
    >>>         item = super().resolve_data(write_item)
    >>>         return item if write_item.type == WriteItemType.BYTE_IO else item.to(torch.float16)

    Using the global planning step to make central decisions that can't be made individually by each rank

    >>> # xdoctest: +SKIP("undefined vars")
    >>> from itertools import zip_longest
    >>> from dataclasses import replace
    >>> class DDPLoadBalancingPlanner(DefaultSavePlanner):
    >>> # This uses the default local plan behavior of having all non-sharded writes in rank 0
    >>> # This sample doesn't handle ShardedTensors
    >>>     def create_global_plan(self, all_plans):
    >>>         iters = [iter(all_plans[0].items)] * len(all_plans)
    >>>         items_per_rank = [
    >>>             [item for item in items if item is not None]
    >>>             for items in zip(*zip_longest(*iters), strict=True)
    >>>         ]
    >>>         all_plans = [
    >>>             replace(plan, items=items)
    >>>             for plan, items in zip(all_plans, items_per_rank, strict=True)
    >>>         ]
    >>>         return super().create_global_plan(all_plans)

    Finally, some planners need to save additional metadata in the checkpoint, this is
    accomplished by having each rank contribute their data items in the local plan and
    the global planner aggregate them:

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class SaveExtraDataPlanner(DefaultSavePlanner):
    >>>     def create_local_plan(self) -> SavePlan:
    >>>         plan = super().create_local_plan()
    >>>         return replace(plan, planner_data="per-rank-data")
    >>>
    >>>     def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
    >>>         global_plan, metadata = super().create_global_plan(all_plans)
    >>>         merged_data = [p.planner_data for p in global_plan]
    >>>         metadata = replace(metadata, planner_data=merged_data)
    >>>         return global_plan, metadata
    """

    # Save plan for the current rank as computed by `create_local_plan` API
    # Cached on the local rank.
    _cached_save_plan: dict[str, SavePlan] = {}
    # Final save plan for the current rank.
    # This is created by merging the plan created by `create_local_plan` API
    # and the result of `create_global_plan` for the given rank.
    # This is the final plan computed by the `finish_plan` API that gets
    # sent to the `write_data`.
    # Cached on the local rank.
    _cached_final_save_plan: dict[str, SavePlan] = {}
    # Collection of all the local plans from all the ranks.
    # This is the input to the `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_all_plans: dict[str, list[SavePlan]] = {}
    # Global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_global_plan: dict[str, list[SavePlan]] = {}
    # Metadata for the global checkpoint plan as computed by `create_global_plan` API.
    # Cached on the coordinator rank.
    _cached_metadata: dict[str, Metadata] = {}

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] Abstract. Initialize the planner for a save of state_dict, keeping the given
        # state dict, optional storage_meta and coordinator flag. Called on all ranks.
        """
        Initialize this planner to save ``state_dict``.

        Implementations should save those values as they won't be provided later in the save process.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> SavePlan:
        # [FUNCTION DOC] Abstract. Compute this rank's SavePlan from the state dict; planner-specific
        # data may go in planner_data. Called on all ranks.
        """
        Compute the save plan for the current rank.

        This will be aggregated and passed to create_global_plan.
        Planner specific data can be passed through SavePlan::planner_data.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def create_global_plan(
        self, all_plans: list[SavePlan]
    ) -> tuple[list[SavePlan], Metadata]:
        # [FUNCTION DOC] Abstract. Given every rank's local plan, make global decisions and return
        # the per-rank plans plus checkpoint Metadata. Called on the coordinator rank only.
        """
        Compute the global checkpoint plan and return the local plan of each rank.

        This is called on the coordinator rank only.
        """

    @abc.abstractmethod
    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        # [FUNCTION DOC] Abstract. Merge the local plan with the global planning result for this rank
        # and return the final SavePlan passed to the writer. Called on all ranks.
        """
        Merge the plan created by `create_local_plan` and the result of `create_global_plan`.

        This is called on all ranks.
        """

    @abc.abstractmethod
    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO:
        # [FUNCTION DOC] Abstract. Look up the state dict object for write_item and apply any
        # transformation (e.g. serialization), returning a tensor or BytesIO. Must be idempotent and
        # thread-safe; called at least once per item, lazily allocating memory.
        """
        Transform and prepare ``write_item`` from ``state_dict`` for storage, ensuring idempotency and thread-safety.

        Lookup the object associated with ``write_item`` in ``state_dict`` and apply any
        transformation (such as serialization) prior to the storage layer consuming it.

        Called on each rank multiple times, at least once per WriteItem in the final SavePlan.

        This method should be idempotent and thread-save. StorageWriter implementations
        are free to call it as frequently as they need.

        Any transformation that allocates memory should be lazily done when his method
        is called in order to reduce peak memory required by checkpointing.

        When returning tensors, they can be on any device or format, they can be views too.
        It's the storage layer responsibility to figure out how to save them.
        """


class LoadPlanner:
    # [CLASS DOC] Interface for customizing checkpoint loading (not declared an ABC though methods
    # are marked abstract). Call order: set_up_planner, create_local_plan (all ranks),
    # create_global_plan (coordinator), finish_plan, then load_bytes per non-tensor value and
    # resolve_tensor/commit_tensor pairs per tensor. Stateful proxy to the state dict, which is
    # modified in place.
    """
    Abstract class defining the protocol used by load_state_dict to plan the load process.

    LoadPlanner are stateful objects that can be used to customize the whole load process.

    LoadPlanner acts as an access proxy to the state_dict, so any transformation done to it
    will be visible to the whole process.

    A planner subclass can expect the following sequence of calls during load_state_dict:

    1) set_up_planner - called on all ranks.
        Signals the start of loading a checkpoint.

    2) create_local_plan - called on all ranks.
        Process the state_dict and produces a `LoadPlan` that will be sent for global planning.

    3) create_global_plan - called on the coordinator rank only.
        Takes the LoadPlan from all ranks and make any global decision.

    4) load_bytes - called multiple times on each rank
        This is called once per non-tensor value in state_dict.

    5) resolve_tensor and commit_tensor - called multiple times on each rank
        They are called in pair for each Tensor value in state_dict.

    Users are recommended to extend DefaultLoadPlanner instead of this interface directly as
    most changes can be expressed by changes in a single method.

    There are two usual patterns of extension:

    Rewriting state_dict. This is the simplest way to extend the load process as it
    doesn't require understanding the intricacies of how LoadPlan works. We need
    to keep a reference to the original state_dict as load happens in place so
    we need to be able to perform it in place

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class RenamePlanner(DefaultLoadPlanner):
    >>>     def set_up_planner(
    >>>         self,
    >>>         state_dict: STATE_DICT_TYPE,
    >>>         metadata: Metadata,
    >>>         is_coordinator: bool,
    >>>     ) -> None:
    >>>         self.original_state_dict = state_dict
    >>>         state_dict = {"foo_" + k: v for k, v in state_dict.items()}
    >>>
    >>>         if self.flatten_sharded_tensors:
    >>>             state_dict = _flatten_sharded_tensors(state_dict)
    >>>
    >>>         if self.flatten_state_dict:
    >>>             state_dict, self.mappings = flatten_state_dict(state_dict)
    >>>
    >>>         self.state_dict = state_dict
    >>>         self.metadata = metadata
    >>>         self.is_coordinator = is_coordinator
    >>>
    >>>     def load_bytes(self, read_item, value):
    >>> # Remove the "foo_" prefix
    >>>         self.original_state_dict[read_item.dest_index.fqn[4:]] = torch.load(value, weights_only=False)


    Modifying resolve_tensor and commit_tensor to handle load time transformation.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> class MetaModelMaterialize(DefaultSavePlanner):
    >>>     def resolve_tensor(self, read_item):
    >>>         tensor = super().resolve_tensor(read_item)
    >>>         return torch.empty_like(tensor, device="cpu")
    >>>
    >>>     def commit_tensor(self, read_item, tensor):
    >>>         self.state_dict[read_item.dest_index.fqn] = tensor
    """

    @abc.abstractmethod
    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] Abstract. Initialize to load into state_dict using optional checkpoint
        # metadata and coordinator flag. Called on every rank.
        """
        Initialize this instance to load data into ``state_dict``.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_local_plan(self) -> LoadPlan:
        # [FUNCTION DOC] Abstract. Build this rank's LoadPlan from the state dict and metadata given
        # to set_up_planner. Called on every rank.
        """
        Create a LoadPlan based on state_dict and metadata provided by set_up_planner.

        . N.B. This is called on every rank.
        """

    @abc.abstractmethod
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        # [FUNCTION DOC] Abstract. Take all ranks' plans and return one plan per rank. Coordinator only.
        """
        Compute the global load plan and return plans for each rank.

        . N.B. This is called on the coordinator rank only
        """

    @abc.abstractmethod
    def finish_plan(self, central_plan: LoadPlan) -> LoadPlan:
        # [FUNCTION DOC] Abstract. Accept the coordinator's plan for this rank and return the final plan.
        """Accept the plan from coordinator and return final LoadPlan."""

    @abc.abstractmethod
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        # [FUNCTION DOC] Abstract. Apply the bytes in value (format defined by the SavePlanner) to the
        # state dict entry described by read_item, modifying the state dict in place.
        """
        Load the item described by ``read_item``and ``value``.

        This method is expected to modify in-place the underlying state_dict.

        The contents of ``value`` are defined by the SavePlanner used to produce
        the checkpoint being loaded.
        """

    def resolve_bytes(self, read_item: ReadItem) -> io.BytesIO:
        # [FUNCTION DOC] Return the BytesIO the reader should fill for read_item; should alias the one
        # in the state dict. Base implementation always raises NotImplementedError.
        """
        Return the BytesIO to be used by the StorageReader to load `read_item`.

        The BytesIO should alias with one on the underlying state_dict as StorageReader will replace its contents.
        """
        raise NotImplementedError("LoadPlanner.resolve_bytes is not implemented")

    @abc.abstractmethod
    def resolve_tensor(self, read_item: ReadItem) -> torch.Tensor:
        # [FUNCTION DOC] Abstract. Return the destination tensor for read_item that the reader fills;
        # should alias the state dict tensor, otherwise commit_tensor must copy it back.
        """
        Return the tensor described by ``read_item`` to be used by the StorageReader to load `read_item`.

        The tensor should alias with one on the underlying state_dict as StorageReader will replace its contents.
        If, for any reason, that's not possible, the planner can use the ``commit_tensor`` method to copy the data
        back to the one in state_dict.
        """

    @abc.abstractmethod
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        # [FUNCTION DOC] Abstract. Called after the reader filled the tensor returned by resolve_tensor;
        # lets the planner post-process it and copy it into the state dict.
        """
        Call once the StorageReader finished loading data into ``tensor``.

        The provided tensor is the same one returned by the call to ``resolve_tensor``.
        This method is only needed if this LoadPlanner needs to post process ``tensor`` prior to
        copying it back to the one in the state_dict.

        The contents of tensor will follow its device synchronization model.
        """
