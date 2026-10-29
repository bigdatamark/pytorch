# [MODULE DOC] Abstract storage interfaces for distributed checkpointing: the WriteResult record,
# the StorageWriter ABC (used by save) and the StorageReader ABC (used by load).
# Concrete backends (filesystem etc.) subclass these. No import-time side effects.
import abc
import os
from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    SavePlan,
    SavePlanner,
)
from torch.futures import Future


__all__ = ["WriteResult", "StorageWriter", "StorageReader"]


@dataclass(frozen=True)
class WriteResult:
    # [CLASS DOC] Frozen record produced by a storage writer per written item: the metadata index of the
    # item, the number of bytes written, and opaque storage-specific data.
    index: MetadataIndex

    size_in_bytes: int
    storage_data: Any


class StorageWriter(abc.ABC):
    # [CLASS DOC] Abstract interface used by save to write checkpoint data. One instance serves as both
    # coordinator and follower depending on the role given at setup. Call order: reset (optional
    # checkpoint id), set_up_storage_writer, prepare_local_plan, prepare_global_plan (coordinator only),
    # write_data, finish (coordinator only). Subclasses must implement all abstract methods.
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] Abstract: signal that a brand new checkpoint write begins, optionally with a
        # checkpoint id whose meaning (path, key, etc.) is storage-dependent. Returns nothing.
        """
        Calls to indicates a brand new checkpoint write is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint write. The meaning of the checkpoint_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def set_up_storage_writer(
        self, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Abstract: initialize this writer, telling it whether it is the coordinator. Extra
        # positional and keyword arguments are implementation-defined. Returns nothing.
        """
        Initialize this instance.

        Args:
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        # [FUNCTION DOC] Abstract: apply storage-specific local planning to the given save plan and return
        # the (possibly transformed) plan; preferably only attaches storage data to it.
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plan (SavePlan): The local plan from the ``SavePlanner`` in use.

        Returns:
            A transformed ``SavePlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # [FUNCTION DOC] Abstract: coordinator-only centralized planning. Takes the list of per-rank save
        # plans and returns a list of transformed plans.
        """
        Perform centralized planning of storage.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in SavePlan::storage_data.

        Args:
            plans: A list of ``SavePlan`` instances, one for each rank.

        Returns:
            A list of transformed ``SavePlan`` after storage global planning
        """

    @abc.abstractmethod
    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[list[WriteResult]]:
        # [FUNCTION DOC] Abstract: write all items of the plan, using the planner to resolve each item to
        # its data lazily (tensors may be on any device and may be non-contiguous views). Returns a future
        # resolving to a list of WriteResult.
        """
        Write all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``SavePlanner::resolve_data`` on each item
        from the plan to get access to the underlying object to write.

        Subclasses should lazily call `resolve_data` as it can allocate memory.
        In case of tensors, make following assumptions:

        - They might be on any device, including not matching the one on ``WriteItem::tensor_data``
        - They might be views or not contiguous. Only the projection needs to be saved.

        Args:
            plan (SavePlan): The save plan to execute.
            planner (SavePlanner): Planner object to be used to resolve items to data.

        Returns:
            A future that completes to a list of WriteResult
        """

    @abc.abstractmethod
    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        # [FUNCTION DOC] Abstract: coordinator-only. Persist the checkpoint metadata and mark the checkpoint
        # successful, given the metadata and the per-rank lists of WriteResult. Returns nothing.
        """
        Write the metadata and marks the current checkpoint as successful.

        The actual format/schema used for serializing `metadata` is an
        implementation detail. The only requirement is that it's recoverable
        in to the same object graph.

        Args:
            metadata (Metadata): metadata for the new checkpoint
            results: A list of WriteResults from all ranks.

        Returns:
            None
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Abstract class method: return whether the given checkpoint id is supported by this
        # storage; used for automatic storage selection.
        """
        Check if the given checkpoint_id is supported by the storage. This allows
        us to enable automatic storage selection.
        """
        ...

    def storage_meta(self) -> StorageMeta | None:
        # [FUNCTION DOC] Return storage-specific metadata (StorageMeta) for observability, passed to the save
        # planner; the default implementation returns None.
        """
        Return the storage-specific metadata. This is used to store additional information
        in a checkpoint that can be useful for providing request-level observability. StorageMeta
        is passed to the ``SavePlanner`` during save calls. Returns None by default.

        TODO: provide an example
        """
        return None


class StorageReader(abc.ABC):
    # [CLASS DOC] Abstract interface used by load to read checkpoint data. Call order: reset (optional
    # checkpoint id), read_metadata, set_up_storage_reader, prepare_local_plan, prepare_global_plan
    # (coordinator only), read_data. Subclasses must implement all abstract methods.
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls by ``load_state_dict``:

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) read_metadata()
    2) (all ranks) set_up_storage_reader()
    3) (all ranks) prepare_local_plan()
    4) (coordinator) prepare_global_plan()
    5) (all ranks) read_data()
    """

    @abc.abstractmethod
    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        # [FUNCTION DOC] Abstract: signal that a brand new checkpoint read begins, optionally with a
        # checkpoint id whose meaning is storage-dependent. Returns nothing.
        """
        Calls to indicates a brand new checkpoint read is going to happen.
        A checkpoint_id may be present if users set the checkpoint_id for
        this checkpoint read. The meaning of the checkpoint_id is
        storage-dependent. It can be a path to a folder/file or a key for
        a key-value storage.

        Args:
            checkpoint_id (Union[str, os.PathLike, None]):
                The ID of this checkpoint instance. The meaning of the checkpoint_id
                depends on the storage. It can be a path to a folder or to a file.
                It can also be a key if the storage is more like a key-value store.
                (Default: ``None``)
        """
        ...

    @abc.abstractmethod
    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        # [FUNCTION DOC] Abstract: read and return the Metadata of the checkpoint being loaded. Extra
        # arguments are implementation-defined.
        """
        Read the checkpoint metadata.

        Returns:
            The metadata object associated with the checkpoint being loaded.

        """

    @abc.abstractmethod
    def set_up_storage_reader(
        self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Abstract: initialize this reader with the checkpoint metadata and whether it is the
        # coordinator. Extra arguments are implementation-defined. Returns nothing.
        """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """

    @abc.abstractmethod
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        # [FUNCTION DOC] Abstract: apply storage-specific local planning to the given load plan and return
        # the (possibly transformed) plan.
        """
        Perform storage-specific local planning.

        While this method can produce a completely different plan, the recommended
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plan (LoadPlan): The local plan from the ``LoadPlan`` in use.

        Returns:
            A transformed ``LoadPlan`` after storage local planning
        """

    @abc.abstractmethod
    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
        # [FUNCTION DOC] Abstract: coordinator-only centralized planning. Takes the list of per-rank load
        # plans and returns a list of transformed plans.
        """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """

    @abc.abstractmethod
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # [FUNCTION DOC] Abstract: read all items of the plan, using the planner to resolve target tensors
        # and to deserialize bytes into place; the storage layer schedules any cross-device copies.
        # Returns a future that completes when all reads finish.
        """
        Read all items from ``plan`` using ``planner`` to resolve the data.

        A subclass should call ``LoadPlanner::load_bytes`` to deserialize a BytesIO
        object into the right place.

        A subclass should call ``LoadPlanner::resolve_tensor`` to get access to the
        tensors that in should load data into.

        It's the StorageLayer responsibility to properly schedule any cross device copies
        required.

        Args:
            plan (LoadPlan): The local plan to execute on
            planner (LoadPlanner): The planner object to use to resolve items.

        Returns:
            A future that completes once all reads are finished.
        """

    @classmethod
    @abc.abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        # [FUNCTION DOC] Abstract class method: return whether the given checkpoint id is supported by this
        # storage; used for automatic storage selection.
        """
        Check if the given checkpoint_id is supported by the storage. This allows
        us to enable automatic storage selection.
        """
        ...
