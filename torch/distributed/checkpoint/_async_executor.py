# pyre-strict
# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the abstract interface for asynchronous checkpoint save executors (thread based
# or process based implementations live in sibling modules). No import-time side effects.
import abc
import os
from concurrent.futures import Future

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


class _AsyncCheckpointExecutor(abc.ABC):
    # [CLASS DOC] HELPER: abstract interface for running a checkpoint save in the background.
    # Subclasses implement execute_save and decide whether work runs in a thread or process.
    @abc.abstractmethod
    def execute_save(
        self,
        staging_future_or_state_dict: STATE_DICT_TYPE | Future[STATE_DICT_TYPE],
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        # [FUNCTION DOC] HELPER: abstract. Starts a save of a state dict (or of a future that
        # resolves to one, from staging) using the given checkpoint id, storage writer, planner,
        # process group, and the no_dist and use_collectives switches. Returns a future that
        # completes when the save finishes. The base body has no implementation (returns None).
        """
        Execute the checkpoint save request asynchronously.

        This method is intended to be used as an abstraction for
        implementing async checkpointing. The actual checkpoint save
        operation is executed in a separate thread or process depending
        on the implementation of this interface.
        """
