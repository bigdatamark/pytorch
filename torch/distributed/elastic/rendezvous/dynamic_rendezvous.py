# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Dynamic rendezvous: a backend-agnostic rendezvous algorithm in which nodes coordinate through a shared, pickled
# state held in a pluggable RendezvousBackend (c10d, etcd-v2, ...) using fencing-token compare-and-set. Contains the backend
# interface, timeout and settings holders, the shared state and state-holder classes, a state machine of actions with operation
# executors, keep-alive handling, and the public DynamicRendezvousHandler plus the create_handler factory.
# Public exports: RendezvousBackend, RendezvousTimeout, RendezvousSettings, DynamicRendezvousHandler, create_handler.
# No import-time side effects besides creating a logger.
import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import torch.distributed as dist
from torch.distributed import Store
from torch.distributed.elastic.events import construct_and_record_rdzv_event, NodeState

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)
from .utils import _delay, _PeriodicTimer


__all__ = [
    "RendezvousBackend",
    "RendezvousTimeout",
    "RendezvousSettings",
    "DynamicRendezvousHandler",
    "create_handler",
]

logger = logging.getLogger(__name__)


def get_method_name(depth=2):
    # [FUNCTION DOC] HELPER: Returns the name of the function found at the given depth of the current call stack (default 2, the caller of
    # the caller), or a placeholder string if the stack is shallower. Used to label recorded events.
    if len(inspect.stack()) > depth:
        return inspect.stack()[depth].function
    return "no_method_name"


Token = Any
"""Represent an opaque fencing token used by the rendezvous backend."""


class RendezvousBackend(ABC):
    # [CLASS DOC] Abstract interface for a backend that stores the serialized rendezvous state and a fencing token. Implementations
    # provide a name, get_state and set_state, where set_state is a conditional (compare-and-set) write keyed on the token.
    """Represent a backend that holds the rendezvous state."""

    @property
    @abstractmethod
    def name(self) -> str:
        # [FUNCTION DOC] Abstract property: the name of the backend.
        """Get the name of the backend."""

    @abstractmethod
    def get_state(self) -> tuple[bytes, Token] | None:
        # [FUNCTION DOC] Abstract: returns the encoded state and its fencing token, or None if no state exists. Implementations raise
        # RendezvousConnectionError on connection failure and RendezvousStateError if the state is corrupt.
        """Get the rendezvous state.

        Returns:
            A tuple of the encoded rendezvous state and its fencing token or
            ``None`` if no state is found in the backend.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """

    @abstractmethod
    def set_state(
        self, state: bytes, token: Token | None = None
    ) -> tuple[bytes, Token, bool] | None:
        # [FUNCTION DOC] Abstract conditional write. A matching token updates the state; a mismatching token leaves it and returns the existing state;
        # a missing token sets the state only if none exists. Returns (state, token, whether the write succeeded).
        # Raises RendezvousConnectionError or RendezvousStateError as for get_state.
        """Set the rendezvous state.

        The new rendezvous state is set conditionally:

          - If the specified ``token`` matches the fencing token stored in the
            backend, the state will be updated. The new state will be returned
            to the caller along with its fencing token.
          - If the specified ``token`` does not match the fencing token stored
            in the backend, the state won't be updated; instead the existing
            state along with its fencing token will be returned to the caller.
          - If the specified ``token`` is ``None``, the new state will be set
            only if there is no existing state in the backend. Either the new
            state or the existing state along with its fencing token will be
            returned to the caller.

        Args:
            state:
                The encoded rendezvous state.
            token:
                An optional fencing token that was retrieved by a previous call
                to :py:meth:`get_state` or ``set_state()``.

        Returns:
            A tuple of the serialized rendezvous state, its fencing token, and
            a boolean value indicating whether our set attempt succeeded.

        Raises:
            RendezvousConnectionError:
                The connection to the backend has failed.
            RendezvousStateError:
                The rendezvous state is corrupt.
        """


class RendezvousTimeout:
    # [CLASS DOC] Holds four positive timeouts of a rendezvous: join (default 600 s), last call (30 s), close (30 s) and heartbeat (5 s).
    # Values left as None get the defaults; read via properties. Invariant: all timeouts are strictly positive.
    """Hold the timeout configuration of a rendezvous.

    Args:
        join:
            The time within which the rendezvous is expected to complete.
        last_call:
            An additional wait amount before completing the rendezvous once the
            rendezvous has the minimum number of required participants.
        close:
            The time within which the rendezvous is expected to close after a
            call to :py:meth:`RendezvousHandler.set_closed` or
            :py:meth:`RendezvousHandler.shutdown`.
        heartbeat:
            The time within which a keep-alive heartbeat is expected to
            complete.
    """

    _ZERO = timedelta(0)

    _DEFAULT_TIMEOUTS = {
        "join": timedelta(seconds=600),
        "last_call": timedelta(seconds=30),
        "close": timedelta(seconds=30),
        "heartbeat": timedelta(seconds=5),
    }

    _join: timedelta
    _last_call: timedelta
    _close: timedelta
    _heartbeat: timedelta

    def __init__(
        self,
        join: timedelta | None = None,
        last_call: timedelta | None = None,
        close: timedelta | None = None,
        heartbeat: timedelta | None = None,
    ) -> None:
        # [FUNCTION DOC] Applies the four timeouts, using defaults for None values; raises ValueError if any is not positive.
        self._set_timeouts(
            join=join, last_call=last_call, close=close, heartbeat=heartbeat
        )

    @property
    def join(self) -> timedelta:
        # [FUNCTION DOC] Property: the join timeout.
        """Get the join timeout."""
        return self._join

    @property
    def last_call(self) -> timedelta:
        # [FUNCTION DOC] Property: the last call timeout.
        """Get the last call timeout."""
        return self._last_call

    @property
    def close(self) -> timedelta:
        # [FUNCTION DOC] Property: the close timeout.
        """Get the close timeout."""
        return self._close

    @property
    def heartbeat(self) -> timedelta:
        # [FUNCTION DOC] Property: the keep-alive heartbeat timeout.
        """Get the keep-alive heartbeat timeout."""
        return self._heartbeat

    def _set_timeouts(self, **timeouts: timedelta | None):
        # [FUNCTION DOC] HELPER: For each named timeout, substitutes the default when None, validates it is positive (ValueError otherwise) and stores it on the instance under a private attribute of that name.
        for name, timeout in timeouts.items():
            if timeout is None:
                timeout = self._DEFAULT_TIMEOUTS[name]
            if timeout <= self._ZERO:
                raise ValueError(f"The {name} timeout ({timeout}) must be positive.")
            setattr(self, "_" + name, timeout)


@dataclass(repr=False, eq=False, frozen=True)
class RendezvousSettings:
    # [CLASS DOC] Immutable (frozen) dataclass of rendezvous settings: run id, min and max nodes, a RendezvousTimeout, the keep-alive interval and the maximum number of missed heartbeats before a node is considered dead.
    """Hold the settings of the rendezvous.

    Attributes:
        run_id:
            The run id of the rendezvous.
        min_nodes:
            The minimum number of nodes to admit to the rendezvous.
        max_nodes:
            The maximum number of nodes to admit to the rendezvous.
        timeout:
            The timeout configuration of the rendezvous.
        keep_alive_interval:
            The amount of time a node waits before sending a heartbeat to keep
            it alive in the rendezvous.
        keep_alive_max_attempt:
            The maximum number of failed heartbeat attempts after which a node
            is considered dead.
    """

    run_id: str
    min_nodes: int
    max_nodes: int
    timeout: RendezvousTimeout
    keep_alive_interval: timedelta
    keep_alive_max_attempt: int


@dataclass(eq=True, order=True, frozen=True)
class _NodeDesc:
    # [CLASS DOC] HELPER: Immutable, orderable node descriptor made of address, process id and a process-wide local id. Its text form is the three parts joined by underscores. Uniquely identifies a node in the rendezvous.
    """Describe a node in the rendezvous.

    Attributes:
        addr:
            The FQDN of the node or user specified local node address.
        pid:
            The id of the process in which the rendezvous handler runs.
        local_id:
            A process-wide unique id.
    """

    addr: str
    pid: int
    local_id: int

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the descriptor as address, pid and local id joined by underscores.
        return f"{self.addr}_{self.pid}_{self.local_id}"


class _NodeDescGenerator:
    # [CLASS DOC] HELPER: Creates node descriptors with a thread-safe counter that gives each descriptor in the process a unique local id.
    """Generate node descriptors.

    A node descriptor is a combination of an FQDN, a process id, and an auto-
    incremented integer that uniquely identifies a node in the rendezvous.
    """

    _lock: threading.Lock
    _local_id: int

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the lock and starts the local id counter at zero.
        self._lock = threading.Lock()

        # An integer that is incremented with each call to generate().
        self._local_id = 0

    def generate(self, local_addr: str | None = None) -> _NodeDesc:
        # [FUNCTION DOC] HELPER: Atomically takes the next local id and returns a node descriptor using the given local address (or the machine FQDN if none), the current pid and that id. Thread safe.
        # This method can be called by multiple threads concurrently; therefore,
        # we must increment the integer atomically.
        with self._lock:
            local_id = self._local_id

            self._local_id += 1

        return _NodeDesc(local_addr or socket.getfqdn(), os.getpid(), local_id)


class _RendezvousState:
    # [CLASS DOC] Mutable shared rendezvous state that is pickled into the backend: current round number, completion flag, optional deadline,
    # closed flag, a mapping of participant node to rank, wait list and redundancy list of nodes, and a mapping of node to last heartbeat time.
    """Hold the state of a rendezvous.

    Attributes:
        round:
            The current round of the rendezvous.
        complete:
            A boolean value indicating whether the current round of the
            rendezvous is complete.
        deadline:
            The time at which the current round of the rendezvous will be
            considered complete if it is still waiting for nodes to join.
        closed:
            A boolean value indicating whether the rendezvous is closed.
        participants:
            A dictionary of the participants and their corresponding ranks.
        wait_list:
            A set of nodes that are waiting to participate in the next round of
            the rendezvous.
        redundancy_list:
            A set of nodes that are redundant in the current round and can join
            the next rendezvous without triggering re-rendezvous.
        last_heartbeats:
            A dictionary containing each node's last heartbeat time.
    """

    round: int
    complete: bool
    deadline: datetime | None
    closed: bool
    participants: dict[_NodeDesc, int]
    wait_list: set[_NodeDesc]
    redundancy_list: set[_NodeDesc]
    last_heartbeats: dict[_NodeDesc, datetime]

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes an empty state: round zero, not complete, no deadline, not closed, empty collections.
        self.round = 0
        self.complete = False
        self.deadline = None
        self.closed = False
        self.participants = {}
        self.wait_list = set()
        self.redundancy_list = set()
        self.last_heartbeats = {}


def _remove_participant_epilogue(
    state: _RendezvousState, settings: RendezvousSettings
) -> None:
    # [FUNCTION DOC] HELPER: Shared cleanup after participants are removed. If the round was complete and nobody is left, marks it incomplete
    # and advances the round. If it was not complete and participants fell below min nodes, clears the deadline. Mutates the state and logs.
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            msg = "No participants left in the rendezvous, marking rendezvous as incomplete"
            logger.debug(msg)
            state.complete = False

            state.round += 1
    else:
        if len(state.participants) < settings.min_nodes:
            msg = (
                f"Number of participants {len(state.participants)} less than "
                f"min_nodes {settings.min_nodes}, clearing deadline in state"
            )
            logger.debug(msg)
            state.deadline = None


class _RendezvousStateHolder(ABC):
    # [CLASS DOC] HELPER: Abstract holder of the rendezvous state kept in sync with other nodes: exposes the local state, sync and mark_dirty.
    """Hold the shared rendezvous state synced with other nodes."""

    @property
    @abstractmethod
    def state(self) -> _RendezvousState:
        # [FUNCTION DOC] Abstract property: the local rendezvous state.
        """Get the local state."""

    @abstractmethod
    def sync(self) -> bool | None:
        # [FUNCTION DOC] Abstract: reads or writes the latest state; returns whether a dirty local state was successfully synced (None when nothing was written).
        """Read or writes the latest state.

        Returns:
            A boolean value indicating whether the local state, in case marked
            as dirty, was successfully synced with other nodes.
        """

    @abstractmethod
    def mark_dirty(self) -> None:
        # [FUNCTION DOC] Abstract: flags the local state as modified so the next sync writes it.
        """Mark the local state as dirty."""


class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    # [CLASS DOC] HELPER: State holder that syncs through a RendezvousBackend. Keeps the local state, last fencing token, a dirty flag, the last sync time
    # (for caching reads for cache_duration seconds, default 1) and the list of dead nodes found by the latest sanitize. Pickles the state to write and unpickles on read.
    """Hold the rendezvous state synced with other nodes via a backend.

    Args:
        backend:
            The rendezvous backend to use.
        settings:
            The rendezvous settings.
        cache_duration:
            The amount of time, in seconds, to cache the last rendezvous state
            before requesting it from the backend again.
    """

    _backend: RendezvousBackend
    _state: _RendezvousState
    _settings: RendezvousSettings
    _cache_duration: int
    _token: Token
    _dirty: bool
    _last_sync_time: float
    _dead_nodes: list[_NodeDesc]

    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        cache_duration: int = 1,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the backend, settings and cache duration and starts with an empty state, no token, not dirty and no sync time.
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING):
        # [FUNCTION DOC] HELPER: Records a rendezvous event with the given message and node state, naming it after the class and the calling method. Side effect: emits an event.
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
        )

    @property
    def state(self) -> _RendezvousState:
        # [FUNCTION DOC] Property: returns the local state object.
        """See base class."""
        return self._state

    def sync(self) -> bool | None:
        """See base class."""
        # [FUNCTION DOC] Syncs with the backend. If dirty, pickles the local state and attempts a conditional write with the stored token; otherwise reads the state,
        # unless a read happened within the cache duration, in which case returns None right away. Replaces the local state with the unpickled result (or an empty state if
        # the backend has none), updates the token and sync time, clears the dirty flag and then sanitizes (removes nodes with expired heartbeats).
        # Returns true/false for a write attempt's success, or None if no write was attempted. Logs removed dead nodes after a successful write.
        # Raises RendezvousStateError if the state cannot be unpickled; backend errors propagate.
        state_bits: bytes | None = None

        token = None

        has_set: bool | None

        if self._dirty:
            has_set = False

            state_bits = pickle.dumps(self._state)

            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:
            has_set = None

            if self._cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time >= max(
                    time.monotonic() - self._cache_duration, 0
                ):
                    return None

            get_response = self._backend.get_state()
            if get_response is not None:
                state_bits, token = get_response

        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            self._state = _RendezvousState()

        if has_set and self._dead_nodes and logger.isEnabledFor(logging.DEBUG):
            node_list = ", ".join(f"'{dead_node}'" for dead_node in self._dead_nodes)

            msg = (
                f"As part of the sync operation the node(s) {node_list} have been removed from the "
                f"rendezvous '{self._settings.run_id}' since they had no heartbeat."
            )
            self._record(message=msg)
            logger.debug(msg)

        self._token = token

        self._dirty = False

        self._last_sync_time = time.monotonic()

        self._sanitize()

        return has_set

    def _sanitize(self) -> None:
        # [FUNCTION DOC] HELPER: Removes nodes whose last heartbeat is older than keep-alive interval times max attempts from the heartbeat mapping, participants,
        # wait list and redundancy list, remembering them as dead nodes. If any participant was removed, runs the shared epilogue to advance the round or clear the deadline.
        state = self._state

        expire_time = datetime.now(timezone.utc) - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]

        participant_removed = False

        for dead_node in self._dead_nodes:
            msg = f"Detected dead node '{dead_node}', removing it from the rendezvous"
            logger.debug(msg)
            del state.last_heartbeats[dead_node]

            try:
                del state.participants[dead_node]

                participant_removed = True
            except KeyError:
                pass

            try:
                state.wait_list.remove(dead_node)
            except KeyError:
                pass

            try:
                state.redundancy_list.remove(dead_node)
            except KeyError:
                pass

        if participant_removed:
            # Common epilogue shared with the _remove_from_participants()
            # function of _DistributedRendezvousOpExecutor.
            _remove_participant_epilogue(state, self._settings)

    def mark_dirty(self) -> None:
        # [FUNCTION DOC] Marks the local state dirty so the next sync tries a conditional write that fails if another node wrote first.
        """See base class.

        If the local rendezvous state is dirty, the next sync call will try to
        write the changes back to the backend. However this attempt might fail
        if another node, which had the same state, also made changes and wrote
        them before us.
        """
        self._dirty = True


class _Action(Enum):
    # [CLASS DOC] HELPER: Enumeration of the state transition actions a rendezvous operation can request: keep alive, add/remove for participants, wait list and redundancy list, mark complete, mark closed, sync, error closed, error timeout and finish.
    """Specifies the possible actions based on the state of the rendezvous."""

    KEEP_ALIVE = 1
    ADD_TO_PARTICIPANTS = 2
    ADD_TO_WAIT_LIST = 3
    ADD_TO_REDUNDANCY_LIST = 4
    REMOVE_FROM_PARTICIPANTS = 5
    REMOVE_FROM_WAIT_LIST = 6
    REMOVE_FROM_REDUNDANCY_LIST = 7
    MARK_RENDEZVOUS_COMPLETE = 8
    MARK_RENDEZVOUS_CLOSED = 9
    SYNC = 10
    ERROR_CLOSED = 11
    ERROR_TIMEOUT = 12
    FINISH = 13


class _RendezvousContext:
    # [CLASS DOC] HELPER: Plain holder passed to state handlers: the node descriptor of this handler, the current rendezvous state and the settings.
    """Holds the context of the rendezvous.

    Attributes:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state:
            The current state of the rendezvous.
        settings:
            The rendezvous settings.
    """

    node: _NodeDesc
    state: _RendezvousState
    settings: RendezvousSettings

    def __init__(
        self, node: _NodeDesc, state: _RendezvousState, settings: RendezvousSettings
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the node, state and settings.
        self.node = node
        self.state = state
        self.settings = settings


class _RendezvousOpExecutor(ABC):
    # [CLASS DOC] HELPER: Abstract executor that runs a rendezvous operation as a state machine driven by a handler callable.
    """Execute rendezvous operations."""

    @abstractmethod
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        # [FUNCTION DOC] Abstract: runs an operation by repeatedly asking the state handler for the next action given the context and the remaining time until the deadline,
        # applying it until finished. An optional callable can produce a new deadline if the node may join the next rendezvous.
        """Execute a rendezvous operation.

        An operation is run inside a state machine and is expected to transition
        the rendezvous from one state to another.

        Args:
            state_handler:
                A callable that is expected to return the next state transition
                action based on the current state of the rendezvous.
            deadline:
                The time, in seconds, at which the operation will be considered
                timed-out.
            update_deadline:
                Function to generate a new operation deadline if the current
                node may participate in the next rendezvous.
        """


class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    # [CLASS DOC] HELPER: Operation executor that drives the rendezvous state machine against the shared state. Holds the node descriptor, a state holder (synced with other nodes) and the settings.
    # The run loop syncs the state, asks a handler for the next action, applies the matching state change locally, and marks the state dirty so the next sync pushes it.
    """Execute rendezvous operations using a shared state.

    Args:
        node:
            The node descriptor associated with the current rendezvous handler
            instance.
        state_holder:
            The ``RendezvousStateHolder`` to use to sync the rendezvous state
            with other nodes.
        settings:
            The rendezvous settings.
    """

    _node: _NodeDesc
    _state: _RendezvousState
    _state_holder: _RendezvousStateHolder
    _settings: RendezvousSettings

    def __init__(
        self,
        node: _NodeDesc,
        state_holder: _RendezvousStateHolder,
        settings: RendezvousSettings,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the node, state holder and settings.
        self._node = node
        self._state_holder = state_holder
        self._settings = settings

    def _record(self, message: str, node_state: NodeState = NodeState.RUNNING) -> None:
        # [FUNCTION DOC] HELPER: Records a rendezvous event for this node (host, pid, local id) naming the calling method. Side effect: emits an event.
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._node.addr,
            pid=self._node.pid,
            local_id=self._node.local_id,
        )

    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Callable[[timedelta], float] | None = None,
    ) -> None:
        # [FUNCTION DOC] Loops until the handler returns the finish action: syncs the state (recording whether a local write succeeded or the state was stale), builds a context and asks the handler for an action.
        # Closed and timeout actions raise RendezvousClosedError and RendezvousTimeoutError. The sync action sleeps one second. Every other action calls the matching local mutation
        # (for removal from the redundancy list also refreshing the deadline via update_deadline when given) and marks the state dirty. Returns nothing.
        """See base class."""
        action = None
        while action != _Action.FINISH:
            # Reads or writes the latest rendezvous state shared by all nodes in
            # the rendezvous. Note that our local changes might get overridden
            # by another node if that node synced its changes before us.
            has_set = self._state_holder.sync()
            if has_set is not None:
                if has_set:
                    msg = (
                        f"The node '{self._node}' has successfully synced its local changes with "
                        f"other nodes in the rendezvous '{self._settings.run_id}'."
                    )
                else:
                    msg = (
                        f"The node '{self._node}' has a stale state and failed to sync its local "
                        f"changes with other nodes in the rendezvous '{self._settings.run_id}'."
                    )

                self._record(message=msg)
                logger.debug(msg)

            self._state = self._state_holder.state

            ctx = _RendezvousContext(self._node, self._state, self._settings)

            # Determine the next action to take based on the current state of
            # the rendezvous.
            action = state_handler(ctx, deadline)

            if action == _Action.FINISH:
                continue

            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError

            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError

            if action == _Action.SYNC:
                # Delay the execution by one second to avoid overloading the
                # backend if we are asked to poll for state changes.
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.ADD_TO_REDUNDANCY_LIST:
                    self._add_to_redundancy_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.REMOVE_FROM_REDUNDANCY_LIST:
                    self._remove_from_redundancy_list()
                    # update deadline since the node may participate in rendezvous process
                    if update_deadline:
                        deadline = update_deadline(self._settings.timeout.join)
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()

                # Attempt to sync our changes back to other nodes.
                self._state_holder.mark_dirty()

    def _keep_alive(self) -> None:
        # [FUNCTION DOC] HELPER: Records and logs the event, then sets this node's last heartbeat time to now (UTC) in the local state. Pending until synced.
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.now(timezone.utc)

    def _add_to_participants(self) -> None:
        # [FUNCTION DOC] HELPER: Adds this node to the participants (rank placeholder zero) and removes it from the wait list if present, refreshes its heartbeat, sets the last-call
        # deadline when participant count equals min nodes, and completes the rendezvous when it equals max nodes.
        msg = (
            f"The node '{self._node}' added itself to the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        state = self._state

        try:
            state.wait_list.remove(self._node)
        except KeyError:
            pass

        # The ranks of the participants will be set once the rendezvous is
        # complete.
        state.participants[self._node] = 0

        self._keep_alive()

        if len(state.participants) == self._settings.min_nodes:
            state.deadline = (
                datetime.now(timezone.utc) + self._settings.timeout.last_call
            )

        if len(state.participants) == self._settings.max_nodes:
            self._mark_rendezvous_complete()

    def _add_to_wait_list(self) -> None:
        # [FUNCTION DOC] HELPER: Moves this node into the wait list (leaving the redundancy list if there) for the next round and refreshes its heartbeat.
        msg = (
            f"The node '{self._node}' added itself to the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        if self._node in self._state.redundancy_list:
            self._state.redundancy_list.remove(self._node)
        self._state.wait_list.add(self._node)

        self._keep_alive()

    def _add_to_redundancy_list(self) -> None:
        # [FUNCTION DOC] HELPER: Adds this node to the redundancy list for the next round and refreshes its heartbeat.
        msg = (
            f"The node '{self._node}' added itself to the redundancy list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.redundancy_list.add(self._node)

        self._keep_alive()

    def _remove_from_participants(self) -> None:
        # [FUNCTION DOC] HELPER: Removes this node from participants and heartbeats (raises KeyError if absent), then runs the shared epilogue that may advance the round or clear the deadline.
        msg = (
            f"The node '{self._node}' removed itself from the participants of round "
            f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        state = self._state

        del state.participants[self._node]

        del state.last_heartbeats[self._node]

        # Common epilogue shared with the sanitizer() function of
        # _BackendRendezvousStateHolder.
        _remove_participant_epilogue(state, self._settings)

    def _remove_from_wait_list(self) -> None:
        # [FUNCTION DOC] HELPER: Removes this node from the wait list and heartbeats; raises KeyError if absent.
        msg = (
            f"The node '{self._node}' removed itself from the wait list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.wait_list.remove(self._node)

        del self._state.last_heartbeats[self._node]

    def _remove_from_redundancy_list(self) -> None:
        # [FUNCTION DOC] HELPER: Removes this node from the redundancy list and heartbeats; raises KeyError if absent.
        msg = (
            f"The node '{self._node}' removed itself from the redundant list of round "
            f"{self._state.round + 1} of the rendezvous '{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.redundancy_list.remove(self._node)

        del self._state.last_heartbeats[self._node]

    def _mark_rendezvous_complete(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the current round complete, clears the deadline and assigns ranks to participants in sorted node order. Records a succeeded event.
        msg = (
            f"The node '{self._node}' marked round {self._state.round} of the rendezvous "
            f"'{self._settings.run_id}' as complete. Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.debug(msg)

        state = self._state

        state.complete = True
        state.deadline = None

        # Assign the ranks.
        for rank, node in enumerate(sorted(state.participants)):
            state.participants[node] = rank

    def _mark_rendezvous_closed(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the whole rendezvous closed in the local state and records a succeeded event.
        msg = (
            f"The node '{self._node}' marked the rendezvous '{self._settings.run_id}' as closed. "
            "Pending sync."
        )
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.debug(msg)

        self._state.closed = True


def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if this node has a recorded heartbeat at least one keep-alive interval old; false if the node has no heartbeat entry.
    """Determine whether a keep-alive heartbeat should be sent."""
    try:
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        return False

    return (
        last_heartbeat <= datetime.now(timezone.utc) - ctx.settings.keep_alive_interval
    )


class _RendezvousExitOp:
    # [CLASS DOC] HELPER: State handler for leaving the rendezvous (callable object).
    """Represent a rendezvous exit operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # [FUNCTION DOC] HELPER: Handler: if this node is a participant, returns the timeout error action past the deadline, else the remove-from-participants action; if not a participant returns finish.
        if ctx.node in ctx.state.participants:
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.REMOVE_FROM_PARTICIPANTS
        return _Action.FINISH


class _RendezvousJoinOp:
    # [CLASS DOC] HELPER: State handler (callable object) implementing the join logic of the rendezvous.
    """Represent a rendezvous join operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # [FUNCTION DOC] HELPER: Handler deciding the next join action. Closed: raises RendezvousGracefulExitError for redundant nodes, else closed error action. Redundancy-list nodes keep alive/sync while the
        # rendezvous is full, otherwise leave the list. Finish if the round is complete and this node participates. Past the deadline: within a 5 second rollback window remove self from participants or the wait list, else timeout error.
        # Complete rounds: add to wait list if there is capacity, or to the redundancy list if full. Participants of an incomplete round mark it complete once the count is within min/max and the last-call deadline has passed.
        # Non-participants add themselves to the participants. Otherwise keep alive when due, else sync (wait).
        state = ctx.state

        # A closed rendezvous means that it no longer accepts new nodes.
        if state.closed:
            if ctx.node in state.redundancy_list:
                msg = f"The rendezvous '{ctx.settings.run_id}' is closed, terminating pending rendezvous."
                raise RendezvousGracefulExitError(msg)
            return _Action.ERROR_CLOSED

        if ctx.node in state.redundancy_list:
            msg = f"The node {ctx.node} is in redundancy list"
            logger.debug(msg)
            # don't apply the timeout logic here, since we want to allow the node to rejoin
            if len(state.participants) == ctx.settings.max_nodes:
                if _should_keep_alive(ctx):
                    return _Action.KEEP_ALIVE
                else:
                    return _Action.SYNC
            else:
                # transition to waiting state that will respect timeouts.
                msg = f"The node {ctx.node} is removed from redundancy list"
                logger.debug(msg)
                return _Action.REMOVE_FROM_REDUNDANCY_LIST

        is_participant = ctx.node in state.participants

        # If we are part of the rendezvous and it is already complete there is
        # no further action to take.
        if state.complete and is_participant:
            return _Action.FINISH

        now = time.monotonic()
        if now > deadline:
            rollback_period = 5  # 5 seconds

            # If we still have time to rollback (a short period on top of the
            # operation deadline), try to remove ourself from the rendezvous.
            # It is okay if we can't though as our keep-alive will eventually
            # expire.
            if now <= deadline + rollback_period:
                # If we are part of the rendezvous, it means we couldn't find
                # enough participants to complete it on time.
                if is_participant:
                    return _Action.REMOVE_FROM_PARTICIPANTS
                # If we are in the wait list, it means we couldn't wait till the
                # next round of the rendezvous.
                if ctx.node in state.wait_list:
                    return _Action.REMOVE_FROM_WAIT_LIST
            return _Action.ERROR_TIMEOUT

        if state.complete:
            # If we are here, it means we are not part of the rendezvous. In
            # case the rendezvous has capacity for additional participants add
            # ourself to the wait list for the next round.
            if len(state.participants) < ctx.settings.max_nodes:
                if ctx.node not in state.wait_list:
                    return _Action.ADD_TO_WAIT_LIST
            elif len(state.participants) >= ctx.settings.max_nodes:
                if (
                    ctx.node not in state.redundancy_list
                    and ctx.node not in state.wait_list
                ):
                    return _Action.ADD_TO_REDUNDANCY_LIST
        elif is_participant:
            # If the rendezvous has enough number of participants including us,
            # check whether we have passed the rendezvous deadline. If yes,
            # complete it.
            if (
                len(state.participants) >= ctx.settings.min_nodes
                and len(state.participants) <= ctx.settings.max_nodes
                and state.deadline is not None
            ):
                if state.deadline < datetime.now(timezone.utc):
                    msg = (
                        f"The node '{ctx.node}' marking the rendezvous complete, "
                        f"quorum established within deadline"
                    )
                    logger.debug(msg)
                    return _Action.MARK_RENDEZVOUS_COMPLETE
                else:
                    msg = f"The node '{ctx.node}' can't complete rendezvous: deadline reached"
                    logger.debug(msg)
            else:
                msg = f"The node '{ctx.node}' can't complete rendezvous: not enough participants"
                logger.debug(msg)
        else:
            # The rendezvous is not complete yet and we are not part of it. Try
            # to join.
            return _Action.ADD_TO_PARTICIPANTS

        if _should_keep_alive(ctx):
            return _Action.KEEP_ALIVE

        # At this point either the rendezvous is not complete, but we are part
        # of it, which means we have to wait for other participants to join; or
        # the rendezvous is complete, but we are not part of it, which means we
        # have to wait for the next round.
        return _Action.SYNC


class _RendezvousCloseOp:
    # [CLASS DOC] HELPER: State handler (callable object) for closing the rendezvous.
    """Represent a rendezvous close operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # [FUNCTION DOC] HELPER: Handler: finish if already closed, timeout error action past the deadline, else mark-closed action.
        if ctx.state.closed:
            return _Action.FINISH
        if time.monotonic() > deadline:
            return _Action.ERROR_TIMEOUT
        return _Action.MARK_RENDEZVOUS_CLOSED


class _RendezvousKeepAliveOp:
    # [CLASS DOC] HELPER: State handler (callable object) for sending a keep-alive heartbeat.
    """Represent a rendezvous keep-alive update operation."""

    def __call__(self, ctx: _RendezvousContext, deadline: float) -> _Action:
        # [FUNCTION DOC] HELPER: Handler: if a heartbeat is due, returns timeout error past the deadline else the keep-alive action; if not due returns finish.
        if _should_keep_alive(ctx):
            if time.monotonic() > deadline:
                return _Action.ERROR_TIMEOUT
            return _Action.KEEP_ALIVE
        return _Action.FINISH


class DynamicRendezvousHandler(RendezvousHandler):
    # [CLASS DOC] Public rendezvous handler implementing the dynamic rendezvous among nodes. Holds a node descriptor (from a class-level generator), settings, backend name, the c10d store,
    # the state holder, the operation executor, a lock guarding heartbeat/state operations, an optional periodic keep-alive timer, and cached shared TCPStore server and bootstrap store info.
    # Create via from_backend or create_handler. next_rendezvous performs exit then join operations, starts heartbeats and returns the rank, world size and store.
    """Represent a handler that sets up a rendezvous among a set of nodes."""

    # Static
    _node_desc_generator = _NodeDescGenerator()

    _this_node: _NodeDesc
    _settings: RendezvousSettings
    _backend_name: str
    _store: Store
    _state_holder: _RendezvousStateHolder
    _op_executor: _RendezvousOpExecutor
    _heartbeat_lock: threading.Lock
    _keep_alive_timer: _PeriodicTimer | None

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        store: Store,
        backend: RendezvousBackend,
        min_nodes: int,
        max_nodes: int,
        local_addr: str | None = None,
        timeout: RendezvousTimeout | None = None,
        keep_alive_interval: int = 5,
        keep_alive_max_attempt: int = 3,
    ):
        # [FUNCTION DOC] Class method factory: builds a unique node descriptor, settings (default timeouts if none; keep-alive interval seconds converted to a duration; default interval 5 and max attempts 3),
        # a backend-based state holder, and returns a handler using the backend name. Validation errors come from the constructor.
        """Create a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
            keep_alive_interval:
                The amount of time a node waits before sending a heartbeat to keep
                it alive in the rendezvous.
            keep_alive_max_attempt:
                The maximum number of failed heartbeat attempts after which a node
                is considered dead.
        """
        # We associate each handler instance with a unique node descriptor.
        node = cls._node_desc_generator.generate(local_addr)

        settings = RendezvousSettings(
            run_id,
            min_nodes,
            max_nodes,
            timeout or RendezvousTimeout(),
            keep_alive_interval=timedelta(seconds=keep_alive_interval),
            keep_alive_max_attempt=keep_alive_max_attempt,
        )

        state_holder = _BackendRendezvousStateHolder(backend, settings)

        return cls(node, settings, backend.name, store, state_holder)

    def __init__(
        self,
        node: _NodeDesc,
        settings: RendezvousSettings,
        backend_name: str,
        store: Store,
        state_holder: _RendezvousStateHolder,
    ) -> None:
        # [FUNCTION DOC] Validates settings (non-empty run id, min nodes at least one, max nodes not below min nodes; raises ValueError otherwise) and stores the node, settings, backend name, store and state holder,
        # creates the distributed op executor, the heartbeat lock, no timer, and unset shared TCPStore server and bootstrap info.
        if not settings.run_id:
            raise ValueError("The run id must be a non-empty string.")

        if settings.min_nodes < 1:
            raise ValueError(
                f"The minimum number of nodes ({settings.min_nodes}) must be greater than zero."
            )

        if settings.max_nodes < settings.min_nodes:
            raise ValueError(
                f"The maximum number of nodes ({settings.max_nodes}) must be greater than or equal "
                f"to the minimum number of nodes ({settings.min_nodes})."
            )

        self._this_node = node

        self._settings = settings

        self._backend_name = backend_name

        self._store = store

        self._state_holder = state_holder

        self._op_executor = _DistributedRendezvousOpExecutor(
            self._this_node, self._state_holder, self._settings
        )

        self._heartbeat_lock = threading.Lock()

        self._keep_alive_timer = None

        # Cached shared store server reference
        self._shared_tcp_store_server: dist.Store | None = None

        self._bootstrap_store_info: RendezvousStoreInfo | None = None

    def _record(
        self,
        message: str,
        node_state: NodeState = NodeState.RUNNING,
        rank: int | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Records a rendezvous event for this node, with optional rank, naming the calling method. Side effect: emits an event.
        construct_and_record_rdzv_event(
            name=f"{self.__class__.__name__}.{get_method_name()}",
            run_id=self._settings.run_id,
            message=message,
            node_state=node_state,
            hostname=self._this_node.addr,
            pid=self._this_node.pid,
            local_id=self._this_node.local_id,
            rank=rank,
        )

    def _create_tcp_store_server(self, master_addr, master_port) -> dist.TCPStore:
        # [FUNCTION DOC] HELPER: Creates and returns a multi-tenant TCPStore server on the given address and port (acting as master). Opens a socket.
        return dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            is_master=True,
            multi_tenant=True,
        )

    @property
    def settings(self) -> RendezvousSettings:
        # [FUNCTION DOC] Property: the rendezvous settings.
        """Get the settings of the rendezvous."""
        return self._settings

    def get_backend(self) -> str:
        # [FUNCTION DOC] Returns the backend name given at construction.
        """See base class."""
        return self._backend_name

    @property
    def use_agent_store(self) -> bool:
        # [FUNCTION DOC] Property: true unless the environment variable TORCH_DISABLE_SHARE_RDZV_TCP_STORE is set to 1, meaning the agent shares a TCPStore server.
        """See base class."""
        return os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") != "1"

    def next_rendezvous(self) -> RendezvousInfo:
        # [FUNCTION DOC] Joins the next round: stops heartbeats, adds a small random start delay on round zero, runs the exit operation then the join operation with a join-timeout deadline,
        # starts heartbeats and gets the rank, world size and prefixed store. Any failure is recorded and re-raised (timeout, closed, backend errors). Then builds RendezvousInfo:
        # with the sharing opt-out env var set, bootstrap info is built from the store directly; otherwise rank zero creates a TCPStore server on an ephemeral port (only when none exists yet) and the bootstrap info
        # is built once and cached (rebuilt when this node newly becomes rank zero). Raises AssertionError on inconsistent cached state.
        """See base class."""
        msg = (
            f"The node '{self._this_node}' attempts to join the next round of the rendezvous "
            f"'{self._settings.run_id}'."
        )
        self._record(message=msg)
        logger.info(msg)

        try:
            self._stop_heartbeats()

            # Delay the execution for a small random amount of time if this is our
            # first run. This will slightly skew the rendezvous attempts across the
            # nodes and reduce the load on the backend.
            if self._state_holder.state.round == 0:
                _delay(seconds=(0, 0.3))

            exit_op = _RendezvousExitOp()
            join_op = _RendezvousJoinOp()

            deadline = self._get_deadline(self._settings.timeout.join)
            self._op_executor.run(exit_op, deadline)
            self._op_executor.run(join_op, deadline, self._get_deadline)

            self._start_heartbeats()

            rank, world_size = self._get_world()
            store = self._get_store()

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

        msg = (
            f"The node '{self._this_node}' has joined round {self._state_holder.state.round} of "
            f"the rendezvous '{self._settings.run_id}' as rank {rank} in a world of size "
            f"{world_size}."
        )
        self._record(message=msg, rank=rank)
        logger.info(msg)

        # opt-out option of TCPStore sharing
        if os.getenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "0") == "1":
            bootstrap_store_info = RendezvousStoreInfo.build(
                rank, store, local_addr=self._this_node.addr
            )
            return RendezvousInfo(
                store,
                rank,
                world_size,
                bootstrap_store_info,
            )

        # This will only be hit when TCPStore sharing is enabled.
        # Rebuild bootstrap store info when this is the first rendezvous or
        # when the node became rank 0 but hasn't created a TCP store server
        # yet (rank changed between rounds due to elastic membership changes).
        if self._bootstrap_store_info is None or (
            rank == 0 and self._shared_tcp_store_server is None
        ):
            server_port = 0
            if rank == 0:
                self._shared_tcp_store_server = self._create_tcp_store_server(
                    self._this_node.addr, server_port
                )
                server_port = self._shared_tcp_store_server.port
            self._bootstrap_store_info = RendezvousStoreInfo.build(
                rank,
                store,
                local_addr=self._this_node.addr,
                server_port=server_port,  # For non-0 rank, this is a no-op
            )

        if self._bootstrap_store_info is None:
            raise AssertionError
        if rank == 0:
            if self._shared_tcp_store_server is None:
                raise AssertionError

        return RendezvousInfo(
            store,
            rank,
            world_size,
            self._bootstrap_store_info,  # type: ignore[assignment]
        )

    def is_closed(self) -> bool:
        # [FUNCTION DOC] Under the heartbeat lock, syncs the state and returns whether the rendezvous is closed. Failures are recorded and re-raised.
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return self._state_holder.state.closed

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def set_closed(self) -> None:
        # [FUNCTION DOC] Under the heartbeat lock, closes the rendezvous via the close operation. Failures are recorded and re-raised.
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._close()
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def num_nodes_waiting(self) -> int:
        # [FUNCTION DOC] Under the heartbeat lock, syncs the state and returns the number of nodes in the wait list. Failures are recorded and re-raised.
        """See base class."""
        try:
            with self._heartbeat_lock:
                self._state_holder.sync()

                return len(self._state_holder.state.wait_list)

        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def get_run_id(self) -> str:
        # [FUNCTION DOC] Returns the run id from the settings.
        """See base class."""
        return self._settings.run_id

    def shutdown(self) -> bool:
        # [FUNCTION DOC] Stops heartbeats and tries to close the rendezvous. Returns true on success, false (with a warning and failed event) on a RendezvousError; other exceptions are recorded and re-raised.
        """See base class."""
        self._stop_heartbeats()

        try:
            self._close()

            return True
        except RendezvousError as ex:
            msg = (
                f"The node '{self._this_node}' has failed to shutdown the rendezvous "
                f"'{self._settings.run_id}' due to an error of type {type(ex).__name__}."
            )
            self._record(message=msg, node_state=NodeState.FAILED)
            logger.warning(msg)

            return False
        except Exception as e:
            self._record(
                message=f"{type(e).__name__}: {str(e)}",
                node_state=NodeState.FAILED,
            )
            raise

    def _close(self) -> None:
        # [FUNCTION DOC] HELPER: Runs the close operation with a close-timeout deadline, then records and logs the closure. Errors such as timeout propagate.
        op = _RendezvousCloseOp()

        deadline = self._get_deadline(self._settings.timeout.close)

        self._op_executor.run(op, deadline)

        msg = f"The node '{self._this_node}' has closed the rendezvous '{self._settings.run_id}'."
        self._record(message=msg, node_state=NodeState.SUCCEEDED)
        logger.info(msg)

    @staticmethod
    def _keep_alive_weak(weak_self) -> None:
        # [FUNCTION DOC] HELPER: Static timer callback; dereferences the weak reference to a handler and, if it is still alive, performs its keep-alive. Avoids keeping the handler alive from the timer.
        self = weak_self()
        if self is not None:
            self._keep_alive()

    def _keep_alive(self) -> None:
        # [FUNCTION DOC] HELPER: Under the heartbeat lock, runs the keep-alive operation with a heartbeat-timeout deadline and records success; a RendezvousError is recorded and logged as a warning but not raised.
        with self._heartbeat_lock:
            op = _RendezvousKeepAliveOp()

            deadline = self._get_deadline(self._settings.timeout.heartbeat)

            try:
                self._op_executor.run(op, deadline)

                msg = (
                    f"The node '{self._this_node}' has sent a keep-alive heartbeat to the rendezvous "
                    f"'{self._settings.run_id}'."
                )
                self._record(message=msg)
                logger.debug(msg)
            except RendezvousError as ex:
                msg = (
                    f"The node '{self._this_node}' has failed to send a keep-alive heartbeat to the "
                    f"rendezvous '{self._settings.run_id}' due to an error of type {type(ex).__name__}."
                )
                self._record(message=msg, node_state=NodeState.FAILED)
                logger.warning(msg)

    def _start_heartbeats(self) -> None:
        # [FUNCTION DOC] HELPER: Creates and starts a named periodic timer that calls the keep-alive callback with a weak reference to this handler at the keep-alive interval.
        self._keep_alive_timer = _PeriodicTimer(
            self._settings.keep_alive_interval, self._keep_alive_weak, weakref.ref(self)
        )

        self._keep_alive_timer.set_name(
            f"RendezvousKeepAliveTimer_{self._this_node.local_id}"
        )

        self._keep_alive_timer.start()

    def _stop_heartbeats(self) -> None:
        # [FUNCTION DOC] HELPER: Cancels the keep-alive timer if one was created; does nothing otherwise.
        if self._keep_alive_timer is None:
            return

        self._keep_alive_timer.cancel()

    def _get_world(self) -> tuple[int, int]:
        # [FUNCTION DOC] HELPER: Returns this node's rank and the participant count from the current state. Raises KeyError if the node is not a participant.
        state = self._state_holder.state

        return state.participants[self._this_node], len(state.participants)

    def _wrap_store(self, store: Store) -> Store:
        # [FUNCTION DOC] HELPER: Wraps a store in a prefix store keyed by run id and current round so rounds do not share keys.
        key_prefix = (
            f"torch.rendezvous.{self._settings.run_id}.{self._state_holder.state.round}"
        )

        return dist.PrefixStore(key_prefix, store)

    def _get_store(self) -> Store:
        # [FUNCTION DOC] HELPER: Returns the user store wrapped with the round-specific prefix.
        return self._wrap_store(self._store)

    def _get_deadline(self, timeout: timedelta) -> float:
        # [FUNCTION DOC] HELPER: Returns a monotonic-clock deadline that is the given timeout from now, in seconds.
        return time.monotonic() + timeout.total_seconds()


def _get_timeout(params: RendezvousParameters, key: str) -> timedelta | None:
    # [FUNCTION DOC] HELPER: Reads the integer config entry named by the key plus a timeout suffix and returns it as a duration in seconds, or None if absent.
    timeout = params.get_as_int(key + "_timeout")
    if timeout is None:
        return None
    return timedelta(seconds=timeout)


def create_handler(
    store: Store, backend: RendezvousBackend, params: RendezvousParameters
) -> DynamicRendezvousHandler:
    # [FUNCTION DOC] Builds a DynamicRendezvousHandler from parameters: reads optional join, last call, close and heartbeat timeouts, keep-alive interval (default 5) and max attempts (default 3),
    # then calls from_backend with run id, min/max nodes and local address. On any error records a failed event and re-raises; TypeError if the keep-alive values are explicitly None.
    # KNOWN ISSUE: the docstring names the heartbeat parameter "heartbeat", but the code reads the key "heartbeat_timeout" (every timeout key gets a "_timeout" suffix).
    """Create a new :py:class:`DynamicRendezvousHandler` from the specified parameters.

    Args:
        store:
            The C10d store to return as part of the rendezvous.
        backend:
            The backend to use to hold the rendezvous state.

    +-------------------+------------------------------------------------------+
    | Parameter         | Description                                          |
    +===================+======================================================+
    | join_timeout      | The total time, in seconds, within which the         |
    |                   | rendezvous is expected to complete. Defaults to 600  |
    |                   | seconds.                                             |
    +-------------------+------------------------------------------------------+
    | last_call_timeout | An additional wait amount, in seconds, before        |
    |                   | completing the rendezvous once the minimum number of |
    |                   | nodes has been reached. Defaults to 30 seconds.      |
    +-------------------+------------------------------------------------------+
    | close_timeout     | The time, in seconds, within which the rendezvous is |
    |                   | expected to close after a call to                    |
    |                   | :py:meth:`RendezvousHandler.set_closed` or           |
    |                   | :py:meth:`RendezvousHandler.shutdown`. Defaults to   |
    |                   | 30 seconds.                                          |
    +-------------------+------------------------------------------------------+
    | heartbeat         | The time, in seconds, within which a keep-alive      |
    |                   | heartbeat is expected to complete                    |
    +-------------------+------------------------------------------------------+
    """
    try:
        timeout = RendezvousTimeout(
            _get_timeout(params, "join"),
            _get_timeout(params, "last_call"),
            _get_timeout(params, "close"),
            _get_timeout(params, "heartbeat"),
        )
        keep_alive_interval = params.get_as_int("keep_alive_interval", 5)
        if keep_alive_interval is None:
            raise TypeError(
                "You passed 'keep_alive_interval=None' as a rendezvous configuration option"
            )
        keep_alive_max_attempt = params.get_as_int("keep_alive_max_attempt", 3)
        if keep_alive_max_attempt is None:
            raise TypeError(
                "You passed 'keep_alive_max_attempt=None' as a rendezvous configuration option"
            )

        return DynamicRendezvousHandler.from_backend(
            params.run_id,
            store,
            backend,
            params.min_nodes,
            params.max_nodes,
            params.local_addr,
            timeout,
            keep_alive_interval=keep_alive_interval,
            keep_alive_max_attempt=keep_alive_max_attempt,
        )
    except Exception as e:
        construct_and_record_rdzv_event(
            message=f"{type(e).__name__}: {str(e)}",
            run_id=params.run_id,
            node_state=NodeState.FAILED,
        )
        raise
