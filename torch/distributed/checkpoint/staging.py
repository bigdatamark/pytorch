# [MODULE DOC] Staging utilities for async distributed checkpointing: the AsyncStager protocol,
# StagingOptions config, DefaultStager (optional background thread, pinned/shared memory,
# non-blocking copies), BlockingAsyncStager (synchronous CPU copy), and the experimental private
# _ReplicationStager (exchanges state dicts with a partner rank). The string literal after
# __all__ is not a real module docstring (just an unused expression). No import-time side effects.
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast
from typing_extensions import deprecated, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict
from torch.distributed.checkpoint._pg_transport import PGTransport
from torch.distributed.checkpoint._state_dict_stager import StateDictStager
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


__all__ = ["AsyncStager", "BlockingAsyncStager", "DefaultStager", "StagingOptions"]

"""
Experimental staging module for PyTorch Distributed Checkpointing.
This module provides advanced staging capabilities for checkpoints including:
- Asynchronous staging using ThreadPoolExecutor
- Pinned memory allocation for faster CPU-GPU transfers
- Shared memory support for multi-process scenarios
- Non-blocking CUDA operations with stream synchronization
- Caching of frequently used storages for efficient memory management
- Automatic resource cleanup and memory management
Classes:
    AsyncStager: Protocol defining the staging interface
    StagingOptions: Configuration dataclass for staging behavior
    DefaultStager: Default implementation with comprehensive staging features
    BlockingAsyncStager: Implementation of AsyncStager which stages the state_dict
    on CPU RAM and blocks until the copy is complete. Please use DefaultStager instead.
"""


@runtime_checkable
class AsyncStager(Protocol):
    # [CLASS DOC] Runtime-checkable protocol for stagers used by async save: stage produces a
    # training-safe copy of a state dict (possibly as a Future), synchronize_staging waits for
    # staging to finish (deprecated), close releases resources. The class attribute
    # _synchronize_after_execute (default True) is exposed through should_synchronize_after_execute
    # and tells async save whether to synchronize right after starting the save.
    """
    This protocol is meant to provide customization and extensibility for dcp.async_save, allowing users
    to customize how data is staged previous to executing the usual dcp.save path in parallel.
    The expected order of operations (concretely defined in `torch.distributed.state_dict_saver.async_save`)
    is the following:

    1. AsyncStager.stage_data(state_dict):
        This call gives the AsyncStager the opportunity to 'stage'
        the state_dict. The expectation and purpose of staging in this context is to create a "training-safe"
        representation of the state dict, meaning that any updates to module data after staging is complete
        should not be reflected in the state dict returned from this method. For example, in the default
        case a copy of the entire state dict is created on CPU RAM and returned here, allowing users
        to continue training without risking changes to data which is being serialized.

    2. dcp.save is called on the state_dict returned from stage in parallel. This call is responsible
        for serializing the state_dict and writing it to storage.

    3. If AsyncStager.should_synchronize_after_execute is True, this method will be called immediately after
        the serialization thread starts and before returning from dcp.async_save. If this is set to False,
        the assumption is the user has defined a custom synchronization point for the purpose of further
        optimizing save latency in the training loop (for example, by overlapping staging with the
        forward/backward pass), and it is the respondsibility of the user to call `AsyncStager.synchronize_staging`
        at the appropriate time.

    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        # [FUNCTION DOC] Property returning the _synchronize_after_execute flag.
        """
        Whether to synchronize after executing the stage.
        """
        return self._synchronize_after_execute

    def stage(
        self, state_dict: STATE_DICT_TYPE
    ) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        # [FUNCTION DOC] Protocol default: raises NotImplementedError naming the class; implementations
        # return a staged copy of state_dict, or a Future of it.
        """
        Returns a "staged" copy of `state_dict`. The expectation of the staged copy is that it is
        inoculated from any updates incurred after the stage call is complete.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stage method"
        )

    @deprecated(
        "`synchronize_staging` is deprecated and will be removed in future versions."
        "Please use staging_future from AsyncSaveResponse instead.",
        category=FutureWarning,
    )
    def synchronize_staging(self) -> None:
        # [FUNCTION DOC] Protocol default: does nothing (deprecated, emits FutureWarning when called);
        # implementations block until async staging finishes.
        """
        In the case `stage` is async in some way, this method should be called to ensure staging
        is complete and it is safe to begin modifying the original `state_dict`
        """

    def close(self) -> None:
        # [FUNCTION DOC] Protocol default: does nothing; implementations release stager resources.
        """
        Clean up all resources used by the stager.
        """


@dataclass
class StagingOptions:
    # [CLASS DOC] Dataclass of DefaultStager settings, all booleans defaulting to True:
    # use_pinned_memory, use_shared_memory, use_async_staging (background thread) and
    # use_non_blocking_copy (requires an available accelerator).
    """
    Configuration options for checkpoint staging behavior.

    Attributes:
        use_pinned_memory (bool): Enable pinned memory allocation for faster
            CPU-GPU transfers. Requires CUDA to be available. Default: True
        use_shared_memory (bool): Enable shared memory for multi-process
            scenarios. Useful when multiple processes need access to the
            same staged data. Default: True
        use_async_staging (bool): Enable asynchronous staging using a
            background thread pool. Allows overlapping computation with
            staging operations. Requires CUDA. Default: True
        use_non_blocking_copy (bool): Use non-blocking device memory
            copies with stream synchronization. Improves performance by
            allowing CPU work to continue during GPU transfers. Default: True

    Note:
        CUDA-dependent features will raise exception if CUDA is not available.
    """

    use_pinned_memory: bool = True
    use_shared_memory: bool = True
    use_async_staging: bool = True
    use_non_blocking_copy: bool = True


class DefaultStager(AsyncStager):
    # [CLASS DOC] Full-featured stager. Holds a config, a StateDictStager doing the actual copies,
    # an optional single-worker thread pool (async mode), an optional accelerator stream used for
    # non-blocking copies in async mode, and the latest staging Future. Not thread-safe.
    """
    DefaultStager provides a full-featured staging implementation that combines
    multiple optimization techniques for efficient checkpoint preparation.

    The staging process works as follows:
    1. State dictionary is submitted for staging (sync or async)
    2. Tensors are copied from GPU to optimized CPU storage
    3. CUDA operations are synchronized if non-blocking copies are used
    4. Staged state dictionary is returned or made available via Future

    Usage Patterns:
        # Synchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=False))
        staged_dict = stager.stage(state_dict)
        stager.close()

        # Asynchronous staging
        stager = DefaultStager(StagingOptions(use_async_staging=True))
        future = stager.stage(state_dict)
        # ... do other work ...
        staged_dict = future.result()
        stager.close()

    Performance Considerations:
        - Async staging provides best performance when model computation
          can overlap with staging operations
        - Pinned memory improves CPU-GPU transfer speeds but uses more memory
        - Shared memory allows efficient IPC to checkpoint process
        - Non-blocking copies reduce GPU idle time during memory transfers

    Thread Safety:
        DefaultStager is not thread-safe. Each thread should use its own
        instance, or external synchronization should be provided.
    """

    def __init__(
        self,
        config: StagingOptions = StagingOptions(),
    ):
        # [FUNCTION DOC] Create the underlying StateDictStager (pinned/shared memory per config). For
        # async staging, create a one-thread executor and, if an accelerator is available, a dedicated
        # stream (created on the calling thread). Raises AssertionError if non-blocking copy is
        # requested but no accelerator is available. Note the default config is a shared instance.
        self._config = config
        self._state_dict_stager = StateDictStager(
            pin_memory=config.use_pinned_memory, share_memory=config.use_shared_memory
        )
        self._staging_executor = None
        self._staging_stream = None
        if self._config.use_async_staging:
            self._staging_executor = ThreadPoolExecutor(max_workers=1)
            if torch.accelerator.is_available():
                # Note: stream needs to be initialized on the main thread after default cuda
                # stream is setup/used to avoid the risk of accidentally reusing the main
                # compute stream or in other cases kernels actually launching from the
                # main thread.
                self._staging_stream = torch.Stream()

        if self._config.use_non_blocking_copy:
            if not torch.accelerator.is_available():
                raise AssertionError(
                    "Non-blocking copy requires that the current accelerator is available."
                )

        self._staging_future: Future[STATE_DICT_TYPE] | None = None

    def stage(
        self,
        state_dict: STATE_DICT_TYPE,
        **kwargs: Any,
    ) -> STATE_DICT_TYPE | Future[STATE_DICT_TYPE]:
        # [FUNCTION DOC] Stage state_dict. In async mode submit the staging job to the executor, remember
        # the Future and return it; otherwise stage synchronously and return the staged dict. Extra
        # keyword arguments are forwarded to the staging step. Raises AssertionError if async mode has
        # no executor.
        """
        This function is responsible for staging staging the state_dict.
        See class docstring for more details on staging.
        If use_async_staging is True, it will return a Future object that will be
        fulfilled when staging is complete.
        If use_async_staging is False, it will return the fully staged state_dict.

        Args:
            state_dict (STATE_DICT_TYPE): The state_dict to be staged.
        """
        if self._config.use_async_staging:
            if self._staging_executor is None:
                raise AssertionError(
                    "staging_executor should not be None for async staging"
                )
            self._staging_future = self._staging_executor.submit(
                self._stage,
                state_dict,
                **kwargs,
            )
            return self._staging_future
        else:
            return self._stage(state_dict, **kwargs)

    def _stage(self, state_dict: STATE_DICT_TYPE, **kwargs: Any) -> STATE_DICT_TYPE:
        # [FUNCTION DOC] HELPER: Perform the copy via the StateDictStager. With non-blocking copy, runs
        # inside the staging stream context (if any) and then synchronizes that stream (or the
        # accelerator) to wait for queued copies; raises AssertionError if async mode lacks a stream.
        # Otherwise does a blocking stage. Returns the staged dict.
        # KNOWN ISSUE: kwargs are accepted but not used.
        if self._config.use_non_blocking_copy:
            if not (self._staging_stream or not self._config.use_async_staging):
                raise AssertionError(
                    "Non-blocking copy in a background thread for async staging needs staging_stream to be initialized."
                )
            with (
                self._staging_stream
                if self._staging_stream is not None
                else nullcontext()
            ):
                state_dict = self._state_dict_stager.stage(
                    state_dict, non_blocking=self._config.use_non_blocking_copy
                )
            # waits for the enqued copy operations to finish.
            self._staging_stream.synchronize() if self._staging_stream else torch.accelerator.synchronize()
        else:
            state_dict = self._state_dict_stager.stage(state_dict, non_blocking=False)

        return state_dict

    def close(self) -> None:
        # [FUNCTION DOC] Shut down the executor (waiting for pending work) if any, and close the
        # underlying StateDictStager, releasing cached storages. Stager must not be reused afterwards.
        """
        Clean up all resources used by the DefaultStager. Shuts down the ThreadPoolExecutor
        used for async staging operations and cleans up the underlying StateDictStager's
        cached storages. Should be called when the stager is no longer needed to prevent
        resource leaks, especially in long-running applications. After calling close(),
        the stager should not be used for further staging operations.

        Example Usage:
            stager = DefaultStager(StagingOptions(use_async_staging=True))
            future = stager.stage(state_dict)
            result = future.result()
            stager.close()  # Clean up all resources
        """
        if self._staging_executor:
            self._staging_executor.shutdown(wait=True)
        self._state_dict_stager.close()

    def synchronize_staging(self) -> None:
        # [FUNCTION DOC] Block until the most recent staging Future (if any) completes; re-raises its
        # exception. No-op if nothing was staged asynchronously.
        """
        When use_async_staging is True, this method will wait until staging is complete.
        If use_async_staging is False, this method is a no-op.
        """
        if self._staging_future is not None:
            self._staging_future.result()


class BlockingAsyncStager(AsyncStager):
    # [CLASS DOC] Synchronous stager that copies the state dict to CPU memory and returns it. State:
    # cache_staged_state_dict (reuse one pinned CPU copy across calls), type_check, and
    # state_dict_cache (the reusable copy, created lazily). Its _synchronize_after_execute is False.
    """
    An implementation of AsyncStager which stages the state_dict on CPU RAM and blocks until the copy is complete.
    This implementation also provides an option to optimize stage latency using pinned memory.

    N.B. synchronize_staging is a no-op in this case.


    """

    # default to True since the common case is to stage synchronously
    _synchronize_after_execute: bool = False

    def __init__(
        self,
        cache_staged_state_dict: bool = False,
        type_check: bool = False,
    ):
        # [FUNCTION DOC] Store the caching and type-check options and start with no cached state dict.
        """
        Initializes the BlockingAsyncStager.

        Args:
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increased memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            type_check: Whether to perform a type check during cpu_offload. Defaults to False.

        """
        self.cache_staged_state_dict = cache_staged_state_dict
        self.type_check = type_check
        self.state_dict_cache: STATE_DICT_TYPE | None = None

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        # [FUNCTION DOC] Return a CPU copy of state_dict. Without caching, allocate a fresh CPU state
        # dict and copy into it. With caching, lazily create a pinned CPU cache on first call and copy
        # into it every time (the same cached object is returned and overwritten on later calls).
        """
        Returns a copy of `state_dict` on the CPU.
        """

        if not self.cache_staged_state_dict:
            staged_state_dict = _create_cpu_state_dict(state_dict)
            _copy_state_dict(state_dict, staged_state_dict, type_check=self.type_check)
            return staged_state_dict

        if self.state_dict_cache is None:
            self.state_dict_cache = _create_cpu_state_dict(state_dict, pin_memory=True)
        return _copy_state_dict(state_dict, self.state_dict_cache)

    def synchronize_staging(self) -> None:
        # [FUNCTION DOC] No-op because staging is blocking.
        """
        No-op function, since staging is blocking.
        """

    def close(self) -> None:
        # [FUNCTION DOC] No-op; the cached state dict is not released.
        pass


class _ReplicationStager(AsyncStager):
    # [CLASS DOC] HELPER: Experimental stager that replicates a state dict to a partner rank (rank
    # plus half the world size, modulo world size) over a PGTransport and persists the received
    # replica to disk. State: process group, timeout, device, transport and storage directory
    # (a fresh temp dir if none given). Assumes an even world size. Its _synchronize_after_execute is False.
    """
    An AsyncStager implementation that replicates state_dict across training ranks
    using PGTransport.

    Args:
        pg: ProcessGroup for distributed communication
        timeout: Timeout for communication operations
        device: Device to use for tensor operations
        storage_dir: Directory to store persisted state_dicts

    Warning: This is experimental and subject to change.
    """

    _synchronize_after_execute: bool = False

    def __init__(
        self,
        pg: ProcessGroup,
        timeout: timedelta = timedelta(minutes=30),
        device: torch.device = torch.device("cpu"),
        storage_dir: str | None = None,
    ):
        # [FUNCTION DOC] Store pg, timeout and device, create a PGTransport, and set the storage
        # directory (create a temp directory when none is given); makes sure the directory exists.
        self._pg = pg
        self._timeout = timeout
        self._device = device
        self._transport = PGTransport(pg, timeout, device, None)

        # Set up storage directory for persisting exchanged state_dicts
        if storage_dir is None:
            self._storage_dir = tempfile.mkdtemp(prefix="replication_stager_")
        else:
            self._storage_dir = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)

    def stage(
        self, state_dict: STATE_DICT_TYPE
    ) -> Future[STATE_DICT_TYPE] | STATE_DICT_TYPE:
        # [FUNCTION DOC] Exchange state_dict with the partner rank: the lower-numbered rank sends then
        # receives, the other receives then sends (avoids deadlock). Persists the received replica to
        # disk and returns it. If distributed is not initialized, returns state_dict unchanged.
        # Raises RuntimeError if persisting fails. Blocking despite the Future return type.
        # KNOWN ISSUE: if world size is 1 the partner is the rank itself, so the same rank would
        # both send and receive; the even-world-size assumption is not checked.
        """
        Stage the state_dict by replicating it across ranks. Returns a state_dict representing
        the received replica.

        Perform the actual replication logic. Creates bidirectional pairs where each rank exchanges
        state_dict with its partner at (rank + world_size//2) % world_size.
        Uses simple rank-based ordering to prevent deadlocks.

        Assumes world_size is always even.
        """
        if not dist.is_initialized():
            return state_dict

        world_size = dist.get_world_size()

        current_rank = dist.get_rank()

        # Calculate partner rank using half-world offset
        # creates bidirectional pairs for replication.
        offset = world_size // 2
        partner_rank = (current_rank + offset) % world_size

        # Use simple rank-based ordering to prevent deadlocks.
        # Lower-numbered rank sends first, higher-numbered rank receives first.
        if current_rank < partner_rank:
            # Send first, then receive
            self._transport.send_checkpoint([partner_rank], state_dict)
            received_state_dict = self._transport.recv_checkpoint(partner_rank)
        else:
            # Receive first, then send
            received_state_dict = self._transport.recv_checkpoint(partner_rank)
            self._transport.send_checkpoint([partner_rank], state_dict)

        # Persist the received state_dict for future discoverability
        received_state_dict = cast(STATE_DICT_TYPE, received_state_dict)
        self._persist_state_dict(received_state_dict, current_rank, partner_rank)

        return received_state_dict

    def _persist_state_dict(
        self, state_dict: STATE_DICT_TYPE, current_rank: int, partner_rank: int
    ) -> None:
        # [FUNCTION DOC] HELPER: Save state_dict with torch.save to the persisted path via a temporary
        # file (flush, fsync, then atomic rename), creating the directory if needed and overwriting any
        # earlier replica for the same rank pair. On any failure removes the temp file (ignoring
        # cleanup errors) and raises RuntimeError with context.
        """
        Persist the received state_dict to disk for future discoverability.
        Only keeps one replica per rank, overwriting any previous replica.
        Uses atomic write pattern (temp file + rename).

        Args:
            state_dict: The state_dict received from partner rank
            current_rank: Current rank that received the state_dict
            partner_rank: Rank that sent the state_dict
        """
        final_path = self._get_persisted_path(current_rank, partner_rank)
        temp_path = final_path + ".tmp"

        try:
            # Ensure parent directory exists and is writable
            os.makedirs(os.path.dirname(final_path), exist_ok=True)

            # Write to temporary file with explicit flushing
            with open(temp_path, "wb") as f:
                torch.save(state_dict, f)
                # Flush application buffers to OS buffers
                f.flush()
                # Force OS buffers to disk for durability
                os.fsync(f.fileno())

            # Atomic rename to final location
            os.rename(temp_path, final_path)
        except Exception as e:
            # Clean up temp file if it exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass  # Ignore cleanup errors
            # Re-raise the original exception with more context
            raise RuntimeError(
                f"Failed to persist state_dict from rank {partner_rank} to rank {current_rank}: {e}"
            ) from e

    def _get_persisted_path(self, current_rank: int, partner_rank: int) -> str:
        # [FUNCTION DOC] HELPER: Return the file path in the storage directory named from the current
        # and partner ranks (pure; does not touch disk).
        """
        Get the file path where a state_dict would be persisted.

        Args:
            current_rank: Current rank

        Returns:
            File path for the persisted state_dict
        """
        filename = f"rank_{current_rank}_replica_partner_{partner_rank}.pt"
        return os.path.join(self._storage_dir, filename)

    def synchronize_staging(self) -> None:
        # [FUNCTION DOC] No-op because staging is blocking.
        """
        No-op function, since staging is blocking.
        """

    def close(self) -> None:
        # [FUNCTION DOC] No-op; persisted replica files are intentionally kept.
        """
        Clean up resources. Persisted files are intentionally left for future discovery.
        """
