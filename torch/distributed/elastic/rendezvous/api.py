# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Core rendezvous API: the error hierarchy, RendezvousStoreInfo and RendezvousInfo
# result types, the abstract RendezvousHandler interface, RendezvousParameters configuration,
# and RendezvousHandlerRegistry. Import-time side effect: creates the global registry instance
# rendezvous_handler_registry (initially empty).

import socket
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from torch.distributed import Store
from torch.distributed.elastic.utils.distributed import get_free_port


__all__ = [
    "RendezvousClosedError",
    "RendezvousConnectionError",
    "RendezvousError",
    "RendezvousGracefulExitError",
    "RendezvousHandler",
    "RendezvousHandlerCreator",
    "RendezvousHandlerRegistry",
    "RendezvousInfo",
    "RendezvousParameters",
    "RendezvousStateError",
    "RendezvousStoreInfo",
    "RendezvousTimeoutError",
    "rendezvous_handler_registry",
]


class RendezvousError(Exception):
    # [CLASS DOC] Base class of all rendezvous errors.
    """Represents the base type for rendezvous errors."""


class RendezvousClosedError(RendezvousError):
    # [CLASS DOC] Raised when operating on a rendezvous that has been closed.
    """Raised when a rendezvous is closed."""


class RendezvousTimeoutError(RendezvousError):
    # [CLASS DOC] Raised when a rendezvous does not complete within its timeout.
    """Raised when a rendezvous did not complete on time."""


class RendezvousConnectionError(RendezvousError):
    # [CLASS DOC] Raised when the connection to the rendezvous backend fails.
    """Raised when the connection to a rendezvous backend has failed."""


class RendezvousStateError(RendezvousError):
    # [CLASS DOC] Raised when the stored rendezvous state is corrupt.
    """Raised when the state of a rendezvous is corrupt."""


class RendezvousGracefulExitError(RendezvousError):
    # [CLASS DOC] Raised when a node was not admitted and exits cleanly; used to unwind the stack, not a failure.
    """Raised when node wasn't included in rendezvous and gracefully exits.

    Exception is a mechanism to exit the stack, however does not mean a failure.
    """


@dataclass
class RendezvousStoreInfo:
    # [CLASS DOC] Dataclass with the master address and port that trainers use to bootstrap
    # distributed communication. Class constants name the store keys (MASTER_ADDR, MASTER_PORT)
    # under which rank 0 publishes them; build() creates an instance from a store.
    """Store address and port that can be used to bootstrap trainer distributed comms"""

    MASTER_ADDR_KEY: ClassVar[str] = "MASTER_ADDR"
    MASTER_PORT_KEY: ClassVar[str] = "MASTER_PORT"
    master_addr: str
    master_port: int

    @staticmethod
    def build(
        rank: int,
        store: Store,
        local_addr: str | None,
        server_port: int | None = None,
    ) -> "RendezvousStoreInfo":
        # [FUNCTION DOC] Factory. On rank 0 chooses the address (local_addr or the fully qualified
        # hostname) and port (server_port or a newly found free port) and writes both to the
        # store as encoded strings. Then every rank reads both keys from the store (blocking
        # until rank 0 has written them) and returns the resulting info. Args: rank, store,
        # local_addr, server_port. Side effect: store writes on rank 0.
        """Factory method, finds unused new port on rank0 host and addr/port info with all ranks.

        If master_addr/master_port is known (useful when sharing existing tcp store server) use the constructor.

        Args:
            rank: rank of the current node
            store: store to use for rendezvous
            local_addr: address of the current node, if not provided will be resolved from hostname
            server_port: port of the TCPStore server, when the TCPStore is shared.
        """
        # TODO swap to collectives comms API
        if rank == 0:
            addr = local_addr or socket.getfqdn()
            # When TCPStore is not shared, we fallback to get_free_port.
            port = server_port or get_free_port()
            store.set(
                RendezvousStoreInfo.MASTER_ADDR_KEY,
                addr.encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )
            store.set(
                RendezvousStoreInfo.MASTER_PORT_KEY,
                str(port).encode(encoding="UTF-8"),  # type: ignore[arg-type]
            )

        addr = store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode(encoding="UTF-8")
        port = int(
            store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode(encoding="UTF-8")
        )
        return RendezvousStoreInfo(master_addr=addr, master_port=port)


class RendezvousInfo:
    # [CLASS DOC] Immutable-style holder (read-only properties) of a completed rendezvous result: the
    # control-plane store, this node's rank, the world size, and bootstrap store info.
    """Holds the information about the rendezvous."""

    def __init__(
        self,
        store: Store,
        rank: int,
        world_size: int,
        bootstrap_store_info: RendezvousStoreInfo,
    ):
        # [FUNCTION DOC] Stores the store, rank, world size and bootstrap info; no validation.
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._bootstrap_store_info = bootstrap_store_info

    @property
    def store(self) -> Store:
        # [FUNCTION DOC] Returns the store used by the elastic control plane.
        """Store used by torchelastic control plane"""
        return self._store

    @property
    def rank(self) -> int:
        # [FUNCTION DOC] Returns this node's rank within the group.
        """Rank within a group"""
        return self._rank

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of nodes in the group.
        """Global group size"""
        return self._world_size

    @property
    def bootstrap_store_info(self) -> RendezvousStoreInfo | None:
        # [FUNCTION DOC] Returns the info trainers use to bootstrap distributed communication.
        """Store information that can used by trainer code to bootstrap distributed comms."""
        return self._bootstrap_store_info


class RendezvousHandler(ABC):
    # [CLASS DOC] Abstract interface every rendezvous implementation provides: report backend
    # name and run id, perform the blocking rendezvous barrier, query or set the closed state,
    # count waiting late-arriving nodes, and release resources on shutdown.
    """Main rendezvous interface.

    Note:
        Distributed Torch users normally **do not** need to implement their own
        ``RendezvousHandler``. An implementation based on C10d Store is already
        provided, and is recommended for most users.
    """

    @abstractmethod
    def get_backend(self) -> str:
        # [FUNCTION DOC] Abstract: returns the name of the rendezvous backend.
        """Return the name of the rendezvous backend."""

    @property
    def use_agent_store(self) -> bool:
        # [FUNCTION DOC] Returns whether the store from the rendezvous can be shared with user
        # applications for their lifetime. Default is false; implementations may override.
        """Indicates that store reference returned by :py:meth:`next_rendezvous` can be shared with user
        applications and will be available during application lifecycle.

        Rendezvous handler impl will share store details as instance of :py:class:`RendezvousStoreInfo`.
        Applications as a convention use `MASTER_ADDR`/`MASTER_PORT` env variables to lookup the store.
        """
        return False

    @abstractmethod
    def next_rendezvous(self) -> RendezvousInfo:
        # [FUNCTION DOC] Abstract: blocks until the rendezvous completes with this process in the
        # worker group, then returns a RendezvousInfo. Raises closed, connection, state or
        # timeout rendezvous errors.
        """Main entry-point into the rendezvous barrier.

        Blocks until the rendezvous is complete and the current process is
        included in the formed worker group, or a timeout occurs, or the
        rendezvous was marked closed.

        Returns:
            Instance of :py:class:`RendezvousInfo`.

        Raises:
            RendezvousClosedError:
                The rendezvous is closed.
            RendezvousConnectionError:
                The connection to the rendezvous backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
            RendezvousTimeoutError:
                The rendezvous did not complete on time.
        """

    @abstractmethod
    def is_closed(self) -> bool:
        # [FUNCTION DOC] Abstract: returns whether the rendezvous is closed (eventually consistent;
        # not for synchronization).
        """Check whether the rendezvous has been closed.

        A closed rendezvous means all future attempts to re-rendezvous within
        same job will fail.

        ``is_closed()`` and :py:meth:`set_closed` have semantics of eventual
        propagation and should not be used for synchronization. The intention is
        that if at least one node decides the job is finished, it will close the
        rendezvous, and other nodes will soon observe this and stop running as
        well.
        """

    @abstractmethod
    def set_closed(self):
        # [FUNCTION DOC] Abstract: marks the rendezvous as closed so future rendezvous attempts fail.
        """Mark the rendezvous as closed."""

    @abstractmethod
    def num_nodes_waiting(self) -> int:
        # [FUNCTION DOC] Abstract: returns how many late nodes are waiting outside the current group.
        """Return the number of nodes who arrived late at the rendezvous
        barrier, hence were not included in the current worker group.

        Callers should periodically call this method to check whether new
        nodes are waiting to join the job and if so admit them by calling
        :py:meth:`next_rendezvous()` (re-rendezvous).
        """

    @abstractmethod
    def get_run_id(self) -> str:
        # [FUNCTION DOC] Abstract: returns the user-defined id identifying this distributed job.
        """Return the run id of the rendezvous.

        The run id is a user-defined id that uniquely identifies an instance of
        a distributed application. It typically maps to a job id and is used to
        allow nodes to join the correct distributed application.
        """

    @abstractmethod
    def shutdown(self) -> bool:
        # [FUNCTION DOC] Abstract: releases all resources opened for the rendezvous; returns whether it succeeded.
        """Close all resources that were open for the rendezvous.

        Example::

            rdzv_handler = ...
            try:
                store, rank, world_size = rdzv_handler.next_rendezvous()
            finally:
                rdzv_handler.shutdown()
        """


class RendezvousParameters:
    # [CLASS DOC] Validated bundle of settings used to construct a handler: backend name, endpoint,
    # run id, node count bounds, optional local address, and a free-form config mapping of extra
    # backend options with typed accessors. Invariant: backend non-empty, 1 <= min_nodes <= max_nodes.
    """Hold the parameters to construct a :py:class:`RendezvousHandler`.

    Args:
        backend:
            The name of the backend to use to handle the rendezvous.
        endpoint:
            The endpoint of the rendezvous, usually in form <hostname>[:<port>].
        run_id:
            The id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        local_addr:
            The address of the local node.
        **kwargs:
            Additional parameters for the specified backend.
    """

    def __init__(
        self,
        backend: str,
        endpoint: str,
        run_id: str,
        min_nodes: int,
        max_nodes: int,
        local_addr: str | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC] Validates and stores the settings; extra keyword arguments become the
        # config mapping. Raises a value error if backend is empty, min_nodes is below one, or
        # max_nodes is below min_nodes.
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        if min_nodes < 1:
            raise ValueError(
                f"The minimum number of rendezvous nodes ({min_nodes}) must be greater than zero."
            )
        if max_nodes < min_nodes:
            raise ValueError(
                f"The maximum number of rendezvous nodes ({max_nodes}) must be greater than or "
                f"equal to the minimum number of rendezvous nodes ({min_nodes})."
            )

        self.backend = backend
        self.endpoint = endpoint
        self.run_id = run_id
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.config = kwargs
        self.local_addr = local_addr

    def get(self, key: str, default: Any = None) -> Any:
        # [FUNCTION DOC] Returns the config value for key, or default if absent.
        """Return the value for ``key`` if ``key`` exists, else ``default``."""
        return self.config.get(key, default)

    def get_as_bool(self, key: str, default: bool | None = None) -> bool | None:
        # [FUNCTION DOC] Returns the config value as a boolean. None and booleans pass through;
        # integers 1 or 0 and case-insensitive strings (1/true/t/yes/y or 0/false/f/no/n) are
        # converted. Raises a value error for any other value.
        """Return the value for ``key`` as a ``bool``."""
        value = self.get(key, default)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value == 1:
                return True
            if value == 0:
                return False
        elif isinstance(value, str):
            if value.lower() in ["1", "true", "t", "yes", "y"]:
                return True
            if value.lower() in ["0", "false", "f", "no", "n"]:
                return False
        raise ValueError(
            f"The rendezvous configuration option '{key}' does not represent a valid boolean value."
        )

    def get_as_int(self, key: str, default: int | None = None) -> int | None:
        # [FUNCTION DOC] Returns the config value converted to an integer (None passes through).
        # Raises a value error if the conversion fails.
        """Return the value for ``key`` as an ``int``."""
        value = self.get(key, default)
        if value is None:
            return value
        try:
            return int(value)
        except ValueError as e:
            raise ValueError(
                f"The rendezvous configuration option '{key}' does not represent a valid integer "
                "value."
            ) from e


RendezvousHandlerCreator = Callable[[RendezvousParameters], RendezvousHandler]


class RendezvousHandlerRegistry:
    # [CLASS DOC] Mapping from backend name to a creator callable that builds a handler from
    # RendezvousParameters. Supports registering (no conflicting duplicates) and creating handlers.
    """Represent a registry of :py:class:`RendezvousHandler` backends."""

    _registry: dict[str, RendezvousHandlerCreator]

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates an empty registry.
        self._registry = {}

    def register(self, backend: str, creator: RendezvousHandlerCreator) -> None:
        # [FUNCTION DOC] Adds creator under backend. Re-registering the identical creator is a
        # no-op; raises a value error for an empty backend name or when a different creator is
        # already registered. Mutates the registry.
        """Register a new rendezvous backend.

        Args:
            backend:
                The name of the backend.
            creator:
                The callback to invoke to construct the
                :py:class:`RendezvousHandler`.
        """
        if not backend:
            raise ValueError("The rendezvous backend name must be a non-empty string.")

        current_creator: RendezvousHandlerCreator | None
        try:
            current_creator = self._registry[backend]
        except KeyError:
            current_creator = None

        if current_creator is not None and current_creator != creator:
            raise ValueError(
                f"The rendezvous backend '{backend}' cannot be registered with '{creator}' as it "
                f"is already registered with '{current_creator}'."
            )

        self._registry[backend] = creator

    def create_handler(self, params: RendezvousParameters) -> RendezvousHandler:
        # [FUNCTION DOC] Looks up the creator for params.backend and calls it with params.
        # Raises a value error if the backend is unregistered, and a runtime error if the created
        # handler reports a different backend name than requested. Returns the handler.
        """Create a new :py:class:`RendezvousHandler`."""
        try:
            creator = self._registry[params.backend]
        except KeyError as e:
            raise ValueError(
                f"The rendezvous backend '{params.backend}' is not registered. Did you forget "
                f"to call `{self.register.__name__}`?"
            ) from e

        handler = creator(params)

        # Do some sanity check.
        if handler.get_backend() != params.backend:
            raise RuntimeError(
                f"The rendezvous backend '{handler.get_backend()}' does not match the requested "
                f"backend '{params.backend}'."
            )

        return handler


# The default global registry instance used by launcher scripts to instantiate
# rendezvous handlers.
rendezvous_handler_registry = RendezvousHandlerRegistry()
