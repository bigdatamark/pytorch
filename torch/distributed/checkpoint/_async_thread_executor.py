# pyre-strict
# mypy: allow-untyped-defs
# [MODULE DOC]
# Thread based async checkpoint executor: runs the synchronous save in a single worker
# thread. Contains the worker function save_wrapper and the executor class. No import-time
# side effects.
import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch.distributed as dist
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


def save_wrapper(
    staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
    *,
    checkpoint_id: str | os.PathLike | None = None,
    storage_writer: StorageWriter | None = None,
    planner: SavePlanner | None = None,
    process_group: dist.ProcessGroup | None = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future:
    # [FUNCTION DOC] HELPER: worker run in the executor thread. If given a future, waits for
    # its result (the staged state dict), then calls the synchronous save with all other
    # arguments passed through and returns its result. Exceptions from the future or save propagate.
    # KNOWN ISSUE: the return annotation says Future but save returns checkpoint metadata.
    from torch.distributed.checkpoint.state_dict_saver import save

    staged_dict = (
        staging_future_or_state_dict.result()
        if isinstance(staging_future_or_state_dict, Future)
        else staging_future_or_state_dict
    )
    return save(
        staged_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    # [CLASS DOC] HELPER: executor that submits saves to a private single-worker thread pool.
    # The pool is shut down (without waiting) once a submitted save completes, so each instance
    # is effectively usable for one save.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: creates the one-thread pool with a name prefix for its thread.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AsyncCheckpointExecutor"
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        # [FUNCTION DOC] HELPER: submits the save wrapper to the thread pool with the given
        # arguments, registers a callback that shuts the pool down when the future finishes,
        # and returns the future (resolving to the save result).
        f: Future = self._executor.submit(
            save_wrapper,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
