# pyre-strict
# mypy: allow-untyped-defs
# [MODULE DOC]
# Process based async checkpoint executor. A long-lived daemon subprocess (spawned per rank, with
# its own gloo process group) performs checkpoint saves; the trainer talks to it over a pipe
# using request/response objects. Contains the control enum, request and init-info dataclasses,
# the subprocess manager class, a module-level singleton (_CHECKPOINT_PROCESS) and the executor class.
# Environment switches read at runtime: DCP_USE_PREFIX_STORE, DCP_DISABLE_AUTOMATIC_GC,
# DCP_DISABLE_MANUAL_GC. Import-time side effect: none beyond creating the root-level logger reference.
import gc
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import PrefixStore, TCPStore
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.logger import _dcp_method_logger, _init_logger
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.elastic.agent.server.api import _get_fq_hostname
from torch.distributed.elastic.utils.distributed import get_free_port


logger = logging.getLogger()


class _CheckpointSaveProcessControlOpts(Enum):
    # [CLASS DOC] HELPER: control messages exchanged with the background process: INIT_COMPLETE
    # (child to parent, initialization done) and TERMINATE (parent to child, shut down).
    INIT_COMPLETE = "init_complete"
    TERMINATE = "terminate"


@dataclass(init=False, unsafe_hash=True)
class _CheckpointRequestIdentifier:
    # [CLASS DOC] HELPER: identifies one save request: the checkpoint id plus a random unique
    # string. Hashable (unsafe hash) so it can key request tracking.
    checkpoint_id: str | os.PathLike | None
    uuid: str

    def __init__(self, checkpoint_id: str | os.PathLike | None):
        # [FUNCTION DOC] HELPER: stores the checkpoint id and generates a fresh random unique string.
        self.checkpoint_id = checkpoint_id
        self.uuid = str(uuid4())


@dataclass
class _AsyncCheckpointRequest:
    # [CLASS DOC] HELPER: message sent to the background process: the staged state dict, request
    # identifier, optional storage writer and planner, and the no_dist and use_collectives flags.
    staged_state_dict: STATE_DICT_TYPE
    checkpoint_request_id: _CheckpointRequestIdentifier
    storage_writer: StorageWriter | None = None
    planner: SavePlanner | None = None
    no_dist: bool = False
    use_collectives: bool = True


@dataclass(init=False)
class _ProcessGroupInitInfo:
    # [CLASS DOC] HELPER: picklable bundle of what the background process needs to create its own
    # process group: node-local rank, global rank, world size, rendezvous address and port, and
    # flags for prefix store use and for disabling automatic or manual garbage collection.
    local_rank: int
    global_rank: int
    world_size: int
    tcp_store_master_addr: str
    tcp_store_master_port: int
    use_prefix_store: bool
    disable_automatic_gc: bool
    disable_manual_gc: bool

    def __init__(self, process_group: dist.ProcessGroup | None = None):
        # [FUNCTION DOC] HELPER: collective call (all ranks of the group must call it). Reads ranks
        # and world size from the given group and the three DCP environment switches (set to 1 to
        # enable). The coordinator (rank 0) chooses the rendezvous address and port, which are
        # broadcast to all ranks and saved. Raises AssertionError if the prefix store mode lacks
        # MASTER_ADDR or MASTER_PORT.
        self.local_rank = dist.get_node_local_rank(fallback_rank=0)
        self.global_rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.use_prefix_store = os.environ.get("DCP_USE_PREFIX_STORE", "0") == "1"
        self.disable_automatic_gc = (
            os.environ.get("DCP_DISABLE_AUTOMATIC_GC", "0") == "1"
        )
        self.disable_manual_gc = os.environ.get("DCP_DISABLE_MANUAL_GC", "0") == "1"

        # Let coordinator rank find a port on the localhost.
        # Broadcast the (master_addr, port) to all ranks; each rank in the
        # checkpoint daemon process will use TCPStore (master_addr, port)
        # for collective communication.
        dist_wrapper: _DistWrapper = _DistWrapper(
            group=process_group,
            use_dist=True,
            coordinator_rank=0,
        )

        def get_master_addr_and_port() -> tuple[str, int]:
            # [FUNCTION DOC] HELPER: runs only on the coordinator. In prefix store mode returns the
            # MASTER_ADDR and MASTER_PORT environment values (error if missing). Otherwise returns
            # MASTER_ADDR or the fully qualified hostname, and a newly found free port.
            if self.use_prefix_store:
                master_addr = os.environ.get("MASTER_ADDR")
                master_port = os.environ.get("MASTER_PORT")
                if master_addr is None:
                    raise AssertionError("DCP needs MASTER_ADDR to use prefix store")
                if master_port is None:
                    raise AssertionError("DCP needs MASTER_PORT to use prefix store")
                master_port = int(master_port)
            else:
                master_addr = os.environ.get("MASTER_ADDR")
                if master_addr is None:
                    master_addr = _get_fq_hostname()
                master_port = get_free_port()

            return master_addr, master_port

        self.tcp_store_master_addr, self.tcp_store_master_port = dist_wrapper.broadcast(
            step="get_master_addr_and_port",
            map_fun=get_master_addr_and_port,
        )


class _AsyncCheckpointProcess:
    # [CLASS DOC] HELPER: parent-side handle for the daemon checkpoint subprocess. Owns the spawn
    # context, the pipe end used to talk to the child, and the process. Sends save requests and
    # waits for metadata or exceptions; terminates the child on garbage collection. The child runs
    # the static method _checkpointing_subprocess (init phase, then a serving loop).
    def __init__(
        self,
        pg_init_info: _ProcessGroupInitInfo,
    ):
        # [FUNCTION DOC] HELPER: spawns the daemon subprocess with the init info and a pipe end,
        # closes the parent's copy of the child end, then blocks up to 1800 seconds for the init
        # complete message. Raises RuntimeError on timeout or child death, AssertionError if the
        # response is not the init complete message, or re-raises an exception sent by the child.
        self.ctx = mp.get_context("spawn")
        self._process_pipe, child_end = self.ctx.Pipe()

        self._save_process = self.ctx.Process(
            target=self._checkpointing_subprocess,
            args=(
                pg_init_info,
                child_end,
            ),
            daemon=True,
        )

        self._save_process.start()

        # Close the parent's copy of child end after we pass it into the child,
        # so the recv()s on it will fail-fast if the child process dies.
        child_end.close()

        # Wait for the checkpoint background process to initialize.
        # Using default GLOO init timeout.
        response = self._wait_for_response(timeout=1800)
        if not response == _CheckpointSaveProcessControlOpts.INIT_COMPLETE:
            raise AssertionError(f"Expected INIT_COMPLETE response, got {response}")

    def __del__(self) -> None:
        # [FUNCTION DOC] HELPER: if the child is alive, sends a terminate message and waits up to
        # 5 seconds; if still alive, logs a warning and terminates it forcibly.
        if self._save_process.is_alive():
            try:
                logger.info("Terminating the checkpoint background process.")
                self._send(_CheckpointSaveProcessControlOpts.TERMINATE)
                self._save_process.join(timeout=5)
            finally:
                if self._save_process.is_alive():
                    logger.warning(
                        "Checkpoint background process is still alive after termination request. Sending SIGTERM."
                    )
                    self._save_process.terminate()

    def _send(self, data: Any) -> None:
        # [FUNCTION DOC] HELPER: sends a picklable object to the child over the pipe.
        self._process_pipe.send(data)

    def _wait_for_response(self, timeout: float | None = None) -> Any:
        # [FUNCTION DOC] HELPER: receives the next object from the child. Raises RuntimeError if the
        # child is dead (before or during receipt, including closed pipe) or if a timeout is given
        # and nothing arrives in time; with no timeout it blocks. If the received object is an
        # exception it is raised in the parent; otherwise it is returned.
        if not self._save_process.is_alive():
            logger.info("Checkpoint background process is dead calling join()...")
            self._save_process.join()
            raise RuntimeError(
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if timeout is not None and not self._process_pipe.poll(timeout=timeout):
            raise RuntimeError(
                f"Timed out after {timeout}s while waiting for response from checkpointer process pid: {self._save_process.pid}"
            )

        try:
            response = self._process_pipe.recv()
        except EOFError:
            raise RuntimeError(  # noqa: B904
                f"Checkpoint background process is dead. Exit code: {self._save_process.exitcode}"
            )

        if isinstance(response, BaseException):
            raise response

        return response

    def save(
        self,
        staged_state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        # [FUNCTION DOC] HELPER: wraps the arguments into a request with a fresh identifier, sends
        # it to the child and blocks (no timeout) until the reply. Returns the checkpoint metadata.
        # Raises AssertionError if the reply is not metadata; child exceptions are re-raised.
        # Create a unique identifier to locate requests/responses
        # from the checkpoint daemon process.
        checkpoint_request_id = _CheckpointRequestIdentifier(checkpoint_id)
        async_cp_request = _AsyncCheckpointRequest(
            staged_state_dict=staged_state_dict,
            checkpoint_request_id=checkpoint_request_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        self._send(async_cp_request)
        result = self._wait_for_response()
        if not isinstance(result, Metadata):
            raise AssertionError(f"Expected Metadata response, got {type(result)}")
        return result

    @staticmethod
    def _execute_save(
        state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_request_id: _CheckpointRequestIdentifier,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        # [FUNCTION DOC] HELPER: runs in the child. Performs the synchronous save with the request's
        # checkpoint id, writer, planner and flags, and returns the metadata.
        from torch.distributed.checkpoint.state_dict_saver import save

        metadata = save(
            state_dict,
            checkpoint_id=checkpoint_request_id.checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        return metadata

    @staticmethod
    def _checkpointing_subprocess(
        pg_init_info: _ProcessGroupInitInfo,
        parent_conn,
    ) -> None:
        # [FUNCTION DOC] HELPER: child process entry point. Phase 1: sets up logging and rank/rendezvous
        # environment variables, initializes a gloo-only process group (optionally on a prefixed TCP
        # store), barriers, tells the parent init is complete, and optionally disables automatic
        # garbage collection. Any init failure is sent to the parent as an exception and the function
        # returns. Phase 2: loops receiving messages; a terminate message ends the loop; a request
        # triggers a save whose metadata (or exception, without exiting) is sent back; with automatic
        # gc disabled and manual gc allowed, collects after each save and freezes gc after the first.
        # A non-request, non-terminate message raises AssertionError (ends the loop). Always destroys
        # the process group and closes the pipe on exit.
        # Phase 1: Process Group Initialization
        # Only needs to execute once during the lifetime of the checkpoint background process.
        try:
            _init_logger(pg_init_info.global_rank)

            # Setup environment variables for process group initialization.
            os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
            os.environ["MASTER_ADDR"] = pg_init_info.tcp_store_master_addr
            os.environ["MASTER_PORT"] = str(pg_init_info.tcp_store_master_port)
            os.environ["LOCAL_RANK"] = str(pg_init_info.local_rank)
            os.environ["RANK"] = str(pg_init_info.global_rank)
            os.environ["WORLD_SIZE"] = str(pg_init_info.world_size)

            logger.info(
                "Initializing dist.ProcessGroup in checkpoint background process on port %s",
                pg_init_info.tcp_store_master_port,
            )
            # NOTE: GLOO backend is enforced here.
            if pg_init_info.use_prefix_store:
                logger.info(
                    "Initializing dist.ProcessGroup in checkpoint background process with prefix store"
                )
                store = PrefixStore(
                    "AsyncCheckpointProcess/",
                    TCPStore(
                        pg_init_info.tcp_store_master_addr,
                        pg_init_info.tcp_store_master_port,
                    ),
                )
                dist.init_process_group(
                    backend=dist.Backend.GLOO,
                    store=store,
                    world_size=pg_init_info.world_size,
                    rank=pg_init_info.global_rank,
                )
            else:
                dist.init_process_group(backend=dist.Backend.GLOO)
            dist.barrier()

            logger.info("Checkpoint background process is running...")
            parent_conn.send(_CheckpointSaveProcessControlOpts.INIT_COMPLETE)

            if pg_init_info.disable_automatic_gc:
                # Disable automatic garbage collection
                # GC can optionally be called manually after each checkpoint
                gc.disable()
                logger.info("Disabled automatic garbage collection")
        except BaseException as e:
            logger.error(
                f"Checkpoint background process failed during initialization: {e}"  # noqa: G004
            )
            parent_conn.send(e)
            return

        # Phase 2: Serving Loop
        try:
            first_request = True
            while True:
                logger.info("Waiting for checkpoint save request...")
                obj = parent_conn.recv()
                if (
                    isinstance(obj, _CheckpointSaveProcessControlOpts)
                    and obj == _CheckpointSaveProcessControlOpts.TERMINATE
                ):
                    logger.info("Terminating the checkpoint background process.")
                    return
                if not isinstance(obj, _AsyncCheckpointRequest):
                    raise AssertionError(
                        f"Expected _AsyncCheckpointRequest, got {type(obj)}"
                    )
                logger.info(
                    f"Received async checkpoint request with id={obj.checkpoint_request_id.checkpoint_id}"  # noqa: G004
                )

                try:
                    response = _AsyncCheckpointProcess._execute_save(
                        obj.staged_state_dict,
                        checkpoint_request_id=obj.checkpoint_request_id,
                        storage_writer=obj.storage_writer,
                        planner=obj.planner,
                        no_dist=obj.no_dist,
                        use_collectives=obj.use_collectives,
                    )
                    parent_conn.send(response)
                    logger.info(
                        f"Completed checkpoint save request for checkpoint_id={obj.checkpoint_request_id}"  # noqa: G004
                    )

                    # in theory this manual gc should not be needed as we shouldn't be leaking anything from checkpointing process
                    if (
                        pg_init_info.disable_automatic_gc
                        and not pg_init_info.disable_manual_gc
                    ):
                        del obj

                        collected_objects = gc.collect()

                        logger.info(
                            f"Manual garbage collection completed - collected {collected_objects} objects."  # noqa: G004
                        )
                        if first_request:
                            # Freeze GC to not check GC for large checkpoint save plans
                            # After freezing, subsequent gc.collect() calls will only scan
                            # NEW objects created after this point, not the frozen save plan
                            logger.info(
                                "First checkpoint request completed - freezing gc"
                            )
                            gc.freeze()
                    first_request = False
                except BaseException as e:
                    logger.error(
                        f"Checkpoint save failed for checkpoint_id={obj.checkpoint_request_id.checkpoint_id}: {e}"  # noqa: G004
                    )
                    parent_conn.send(e)
                    # Continue serving loop - don't exit process
        finally:
            logger.info("Checkpoint background process is shutting down...")
            dist.destroy_process_group()
            parent_conn.close()


_CHECKPOINT_PROCESS: _AsyncCheckpointProcess | None = None


class _ProcessBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    # [CLASS DOC] HELPER: executor that runs saves through the shared daemon process (module-level
    # singleton, created lazily on first use and reused by later saves). A one-thread pool in the
    # trainer waits for the child so the caller gets a future.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: creates a one-thread pool used to wait on the child process.
        self._executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _execute_save_impl(
        *,
        pg_init_info: _ProcessGroupInitInfo | None,
        staging_future_or_state_dict: Future[STATE_DICT_TYPE] | STATE_DICT_TYPE,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Metadata:
        # [FUNCTION DOC] HELPER: runs in the pool thread. Creates the singleton daemon process if
        # absent (needs pg_init_info, else AssertionError; creation is logged under the writer's
        # checkpoint id when present). Resolves a staging future to a state dict, sends it to the
        # daemon and returns the metadata. Sets the module global _CHECKPOINT_PROCESS.
        # KNOWN ISSUE: the process_group argument is only used for logging; it is not forwarded to the save.
        global _CHECKPOINT_PROCESS
        if _CHECKPOINT_PROCESS is None:
            if pg_init_info is None:
                raise AssertionError(
                    "pg_init_info must not be None when _CHECKPOINT_PROCESS is None"
                )
            ckpt_kwargs = {}
            if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
                ckpt_kwargs["checkpoint_id"] = ckpt_id
                ckpt_kwargs["process_group"] = process_group

            @_dcp_method_logger(**ckpt_kwargs)
            def create_checkpoint_daemon_process() -> None:
                # [FUNCTION DOC] HELPER: constructs the daemon process handle and stores it in the module global.
                global _CHECKPOINT_PROCESS
                # pyrefly: ignore [bad-argument-type]
                _CHECKPOINT_PROCESS = _AsyncCheckpointProcess(pg_init_info=pg_init_info)

            create_checkpoint_daemon_process()

        if _CHECKPOINT_PROCESS is None:
            raise AssertionError(
                "_CHECKPOINT_PROCESS must not be None after initialization"
            )
        staged_state_dict = (
            staging_future_or_state_dict.result()
            if isinstance(staging_future_or_state_dict, Future)
            else staging_future_or_state_dict
        )
        return _CHECKPOINT_PROCESS.save(
            staged_state_dict=staged_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
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
        # [FUNCTION DOC] HELPER: if the daemon does not exist yet, collectively builds the process
        # group init info on the calling thread (all ranks must call). Submits the save implementation
        # to the pool and returns a future resolving to the metadata; the pool shuts down (no wait)
        # when that future completes, so an executor instance serves one save. The first call is
        # slow because it spawns the daemon.
        """
        NOTE:

        - Checkpoint process is implemented as a daemon process.
        The AsyncCheckpointProcess' lifetime is tied to the lifetime of the
        main process (e.g. trainer process).

        - The first call to execute_save_in_process() will initialize the checkpoint
        daemon process. Subsequent async checkpoint requests will not need process
        initialization. Therefore, the first async checkpoint request will take longer to complete.

        - Process initialization can have significant overhead, dominated by latency for all ranks to spawn
        a background process + process group initialization in the background process.
        """

        global _CHECKPOINT_PROCESS
        pg_init_info: _ProcessGroupInitInfo | None = None
        if _CHECKPOINT_PROCESS is None:
            # Find a port on coordinator rank and broadcast
            # to all ranks.
            pg_init_info = _ProcessGroupInitInfo(process_group)

        f: Future = self._executor.submit(
            self._execute_save_impl,
            pg_init_info=pg_init_info,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
