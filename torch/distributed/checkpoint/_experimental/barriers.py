# [MODULE DOC]
# Barrier abstractions for synchronizing ranks around checkpoint operations: abstract Barrier,
# DistBarrier (process group barrier), TCPStoreBarrier (key-value store based), a registry keyed
# by barrier type string, a register decorator, a config dataclass and a factory from config.
# Import-time side effects: creates the registry and registers the two barrier classes in it.
"""
Barrier implementations for synchronizing distributed checkpoint operations.

This module provides abstract and concrete barrier implementations that ensure
all ranks in a distributed training environment complete their checkpoint operations
before proceeding, which is essential for data consistency.
"""

import abc
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Optional

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util


logger = logging.getLogger()


# Registry of barrier types
BARRIER_REGISTRY: dict[str, type] = {}


def register_barrier(barrier_class: type) -> type:
    # [FUNCTION DOC] Class decorator: if the class has a barrier_type attribute, records it in the
    # global registry under that name (overwriting any existing entry). Returns the class unchanged.
    """Register a barrier class in the global registry."""
    if hasattr(barrier_class, "barrier_type"):
        BARRIER_REGISTRY[barrier_class.barrier_type] = barrier_class
    return barrier_class


@dataclass
class BarrierConfig:
    # [CLASS DOC] Dataclass holding a barrier type name (None means no barrier) and a mapping of
    # constructor keyword arguments used by the factory to build that barrier.
    """
    Configuration for barrier construction.

    This class provides a flexible way to configure different barrier implementations
    with their specific constructor arguments. The barrier type will be looked up
    from a registry and instantiated with rank_info and barrier_args.

    Attributes:
        barrier_type: A string identifying the barrier type (e.g., "tcp_store").
                     If None, no barrier will be used.
        barrier_args: Dictionary of arguments to pass to the barrier constructor.
                     rank_info will be automatically injected as the first argument.

    Examples:
        # No barrier
        BarrierConfig()

        # TCPStore barrier
        BarrierConfig(
            barrier_type="tcp_store",
            barrier_args={
                'timeout_barrier_init_secs': 30,
                'barrier_prefix_list': ['checkpoint'],
                'use_checkpoint_barrier_tcpstore_libuv': False,
                'tcpstore_port': 12345,
                'master_address': 'localhost'
            }
        )
    """

    barrier_type: str | None = None
    barrier_args: dict[str, Any] = field(default_factory=dict)


def create_barrier_from_config(
    barrier_config: BarrierConfig,
) -> Optional["Barrier"]:
    # [FUNCTION DOC] Returns None if no barrier type is configured; otherwise looks up the registered
    # class and constructs it with the config's argument mapping as keyword arguments.
    # Raises ValueError for an unregistered type name.
    # KNOWN ISSUE: the BarrierConfig docstring says rank_info is injected automatically, but nothing here injects it.
    """
    Create a barrier instance from BarrierConfig.

    Args:
        barrier_config: Configuration for barrier construction.

    Returns:
        Barrier instance or None if no barrier type is configured.

    Raises:
        ValueError: If the barrier_type is not found in the registry.
    """
    if barrier_config.barrier_type is None:
        return None

    if barrier_config.barrier_type not in BARRIER_REGISTRY:
        raise ValueError(
            f"Unknown barrier type: {barrier_config.barrier_type}. "
            f"Available types: {list(BARRIER_REGISTRY.keys())}"
        )

    barrier_class = BARRIER_REGISTRY[barrier_config.barrier_type]
    return barrier_class(**barrier_config.barrier_args)


class Barrier(abc.ABC):
    # [CLASS DOC] Abstract barrier interface: construction takes implementation-specific keyword
    # arguments, and execute_barrier blocks until every rank has arrived.
    """
    Abstract base class for synchronization barriers.

    A barrier ensures that all ranks in a distributed environment reach a certain
    point in execution before any rank proceeds further, which is essential for
    coordinating operations like checkpointing across multiple processes.
    """

    @abc.abstractmethod
    def __init__(self, **kwargs: dict[str, Any]):
        # [FUNCTION DOC] Abstract constructor; the base body does nothing.
        """
        Initialize a barrier.

        Args:
            **kwargs: Keyword arguments for specific barrier implementations.
                     Common arguments may include rank information, barrier prefixes,
                     timeout settings, and other barrier-specific configuration.
        """
        # No implementation needed in the abstract base class

    @abc.abstractmethod
    def execute_barrier(self) -> None:
        # [FUNCTION DOC] Abstract: block until all participating ranks reach this point.
        """
        Execute a synchronization barrier.

        This method uses the barrier_prefix provided during initialization to
        coordinate synchronization across processes.
        """


@register_barrier
class DistBarrier(Barrier):
    # [CLASS DOC] Registered as "dist_barrier". Barrier that calls the default process group's
    # barrier; needs an initialized distributed environment. Stateless.
    """
    A barrier implementation using PyTorch's distributed barrier for synchronization.

    This barrier uses the built-in torch.distributed.barrier() function to coordinate
    synchronization across multiple processes. It's simpler than TCPStoreBarrier but
    requires an initialized process group.
    """

    barrier_type = "dist_barrier"

    def __init__(
        self,
    ) -> None:
        # [FUNCTION DOC] Raises AssertionError if the distributed package is not initialized.
        """
        Initialize a DistBarrier.

        This barrier requires an initialized PyTorch distributed process group.
        No additional arguments are needed as it uses the current process group.

        Raises:
            AssertionError: If the distributed process group is not initialized.
        """
        if not dist.is_initialized():
            raise AssertionError("DistBarrier requires an initialized process group.")

    def execute_barrier(self) -> None:
        # [FUNCTION DOC] Blocks on the default process group barrier (no explicit timeout, no prefix used).
        # KNOWN ISSUE: the docstring mentions a prefix from initialization, but this class has none.
        """
        Execute a synchronization barrier using the prefix provided during initialization.
        """
        # Note: dist.barrier() doesn't support explicit timeouts
        # The timeout is handled by the underlying implementation
        dist.barrier()


@register_barrier
class TCPStoreBarrier(Barrier):
    # [CLASS DOC] Registered as "tcp_store". Barrier built on a TCP key-value store shared by all
    # ranks (rank 0 hosts it). Keeps a per-prefix sequence counter so repeated barriers use distinct
    # key prefixes, plus rank, world size, prefix and timeout settings.
    """
    A barrier implementation using PyTorch's TCPStore for synchronization.

    This barrier uses a TCP-based distributed key-value store to coordinate
    synchronization across multiple processes. It uses a single TCP store
    for all barrier operations, with different prefixes to distinguish between
    different barrier types.
    """

    barrier_type = "tcp_store"

    def __init__(
        self,
        global_rank: int,
        global_world_size: int,
        barrier_prefix: str,
        timeout_barrier_init_secs: int,
        use_checkpoint_barrier_tcpstore_libuv: bool,
        tcpstore_port: int,
        master_address: str,
        timeout_secs: int,
    ):
        # [FUNCTION DOC] Logs the settings, stores rank, world size, prefix and wait timeout, and
        # creates the TCP store client (rank 0 acts as the server) with the init timeout. May block
        # or raise if the store cannot be reached.
        # KNOWN ISSUE: use_checkpoint_barrier_tcpstore_libuv is accepted and logged but never used.
        """
        Initialize a TCPStoreBarrier.

        Args:
            global_rank: The rank of the current process in the distributed environment.
            global_world_size: The total number of processes in the distributed environment.
            barrier_prefix: A string prefix to identify this specific barrier.
            timeout_barrier_init_secs: Timeout in seconds for initializing the TCPStore.
            use_checkpoint_barrier_tcpstore_libuv: Whether to use libuv for the TCPStore.
            tcpstore_port: Port number for the TCPStore.
            master_address: Address of the master node for the TCPStore.
            timeout_secs: Maximum time in seconds to wait for all ranks to reach the barrier.
        """
        logger.info(
            "Initializing TCPStore master_address=%s tcpstore_port=%s rank=%s "
            "world_size=%s barrier_prefix=%s timeout_barrier_init_secs=%s "
            "use_checkpoint_barrier_tcpstore_libuv=%s timeout_secs=%s",
            master_address,
            tcpstore_port,
            global_rank,
            global_world_size,
            barrier_prefix,
            timeout_barrier_init_secs,
            use_checkpoint_barrier_tcpstore_libuv,
            timeout_secs,
        )

        # Counter collection to track barrier seq on a per barrier prefix basis.
        self._tcp_store_barrier_seq: Counter = Counter()
        self._barrier_prefix = barrier_prefix

        # Store rank and world size for barrier operations
        self._global_rank = global_rank
        self._global_world_size = global_world_size
        self._timeout_secs = timeout_secs

        # Create a single TCP store for all barrier operations
        self._tcp_store = dist.TCPStore(
            master_address,
            int(tcpstore_port),
            world_size=self._global_world_size,
            timeout=timedelta(seconds=timeout_barrier_init_secs),
            is_master=(self._global_rank == 0),
        )

    def execute_barrier(self) -> None:
        # [FUNCTION DOC] Records this rank's current sequence number in the store, waits through the
        # store barrier helper until all ranks have joined that sequence (key prefix is the barrier
        # prefix plus sequence), then increments the sequence. Blocks; store errors or timeouts propagate.
        # KNOWN ISSUE: the configured timeout_secs is only logged and not passed to the wait.
        """
        Execute a synchronization barrier using the prefix provided during initialization.

        The implementation uses a sequence number that is incremented every time
        a barrier is reached. The sequence number is per barrier prefix to allow
        different barriers to operate concurrently.
        """
        barrier_prefix = self._barrier_prefix

        logger.info(
            "Executing barrier barrier_prefix=%s timeout_secs=%s",
            barrier_prefix,
            self._timeout_secs,
        )

        def _rank_key(rank: int) -> str:
            # [FUNCTION DOC] HELPER: returns the store key name for a rank.
            return f"rank{rank}"

        # Track which barrier sequence this rank is joining.
        self._tcp_store.set(
            _rank_key(self._global_rank),
            str(self._tcp_store_barrier_seq[barrier_prefix]),
        )

        # Execute barrier for that sequence number (for the specific prefix).
        store_util.barrier(
            store=self._tcp_store,
            world_size=self._global_world_size,
            key_prefix=(
                barrier_prefix + str(self._tcp_store_barrier_seq[barrier_prefix])
            ),
        )
        self._tcp_store_barrier_seq[barrier_prefix] += 1
