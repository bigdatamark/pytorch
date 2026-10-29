# [MODULE DOC]
# Experimental high level checkpointer wrapper. Defines _Checkpointer, which binds a storage
# writer, reader and optional planners and delegates to the state dict saver and loader
# functions. Exports nothing publicly (empty __all__). No import-time side effects.
from concurrent.futures import Future
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint.state_dict_loader as loader
import torch.distributed.checkpoint.state_dict_saver as saver
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.storage import (
    LoadPlanner,
    SavePlanner,
    StorageReader,
    StorageWriter,
)


__all__: list[str] = []


class _Checkpointer:
    # [CLASS DOC] HELPER: convenience object remembering storage_writer, storage_reader,
    # process_group, coordinator_rank, no_dist, load_planner and save_planner, and exposing
    # save, async_save and load that call the module-level functions with those settings.
    # Experimental.
    """This base class specifies a high level API for saving and loading
    distributed `state_dict` 's. It provides an abstraction over the low-level APIs
    provided by :py:mod:`torch.distributed.checkpoint.storage`, essentially calling
    :py:meth: `torch.distributed.state_dict_saver.save` and
    :py:meth: `torch.distributed.state_dict_loader.load` with the provided storage
    readers and writers.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    def __init__(
        self,
        storage_writer: StorageWriter,
        storage_reader: StorageReader,
        *,
        process_group: dist.ProcessGroup | None = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: LoadPlanner | None = None,
        save_planner: SavePlanner | None = None,
    ):
        # [FUNCTION DOC] HELPER: stores the given writer, reader and options as attributes. No validation or I/O.
        """Initializes the Checkpointer instance.

        Args:
            storage_writer: Instance of StorageWrite used to perform writes.
            storage_reader: StorageReader used to load data from.
            process_group: ProcessGroup to be used for cross-rank synchronization.
            coordinator_rank: Rank to use to coordinate the checkpoint. rank0 is used by default.
            no_dist: If ``True``, distributed checkpoint will not load in SPMD style. (Default: ``False``)
            loader_planner: Instance of LoadPlanner to use when loading.
            save_planner: Instance of SavePlanner to use when saving.
        """
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner

    def save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Metadata:
        # [FUNCTION DOC] HELPER: synchronously saves the state dict through the saver using the
        # stored writer, process group, coordinator rank, no_dist flag and save planner.
        # Returns the checkpoint metadata. Writes to storage; collective across ranks.
        """Calls :py:meth: `torch.distributed.state_dict_saver.save`. Utilizing values passed during initialization."""
        return saver.save(
            state_dict,
            self.storage_writer,
            process_group=self.process_group,
            coordinator_rank=self.coordinator_rank,
            no_dist=self.no_dist,
            planner=self.save_planner,
        )

    def async_save(
        self,
        state_dict: STATE_DICT_TYPE,
    ) -> Future:
        # [FUNCTION DOC] HELPER: starts an asynchronous save via the saver using the stored writer,
        # process group and save planner, and returns the resulting future. Raises AssertionError if
        # the saver does not return a future.
        # KNOWN ISSUE: coordinator rank and no_dist settings are not forwarded to the async saver.
        """
        Calls :py:meth: `torch.distributed.state_dict_saver._async_save`. Utilizing values passed during initialization.

        Returns:
            Future: A future holding the resultant Metadata object from `save`.
        """
        response = saver.async_save(
            state_dict,
            storage_writer=self.storage_writer,
            process_group=self.process_group,
            planner=self.save_planner,
        )
        if not isinstance(response, Future):
            raise AssertionError("response should be a Future instance")
        return response

    def load(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] HELPER: loads checkpoint data in place into the given state dict using the
        # stored reader, process group and load planner. Returns nothing; collective across ranks.
        # KNOWN ISSUE: coordinator rank and no_dist settings are not forwarded to the loader.
        """Calls :py:meth: `torch.distributed.state_dict_loader.load`. Utilizing values passed during initialization."""
        loader.load(
            state_dict,
            storage_reader=self.storage_reader,
            process_group=self.process_group,
            planner=self.load_planner,
        )
