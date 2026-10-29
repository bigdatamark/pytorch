#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Event data model for torchelastic: EventSource and NodeState enums plus the Event
# and RdzvEvent dataclasses, each serializable to and from a JSON string. No import-time side effects.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Union


__all__ = ["EventSource", "Event", "NodeState", "RdzvEvent"]

EventMetadataValue = str | int | float | bool | None


class EventSource(str, Enum):
    # [CLASS DOC] String enum of event producers: AGENT and WORKER.
    """Known identifiers of the event producers."""

    AGENT = "AGENT"
    WORKER = "WORKER"


@dataclass
class Event:
    # [CLASS DOC] Dataclass for a generic job event: name, source (EventSource), timestamp in
    # milliseconds (default 0) and a metadata mapping of primitive values. JSON round-trippable
    # via serialize and deserialize; string conversion yields the JSON.
    """
    The class represents the generic event that occurs during the torchelastic job execution.

    The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occurred.
        metadata: additional data that is associated with the event.
    """

    name: str
    source: EventSource
    timestamp: int = 0
    metadata: dict[str, EventMetadataValue] = field(default_factory=dict)

    def __str__(self):
        # [FUNCTION DOC] Returns the JSON serialization of the event.
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "Event"]) -> "Event":
        # [FUNCTION DOC] Returns an Event: an Event input is returned unchanged; a JSON string is
        # parsed and its source name converted to EventSource before constructing the Event.
        # KNOWN ISSUE: input of any other type leaves the parsed dict undefined and fails with an
        # unbound-variable error rather than a clear type error.
        if isinstance(data, Event):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["source"] = EventSource[data_dict["source"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        return Event(**data_dict)

    def serialize(self) -> str:
        # [FUNCTION DOC] Returns the event converted to a plain mapping and dumped as a JSON string.
        return json.dumps(asdict(self))


class NodeState(str, Enum):
    # [CLASS DOC] String enum of node states in rendezvous: INIT, RUNNING, SUCCEEDED, FAILED.
    """The states that a node can be in rendezvous."""

    INIT = "INIT"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class RdzvEvent:
    # [CLASS DOC] Dataclass describing a rendezvous event: name, run id, message, hostname, pid,
    # node state, and optional master endpoint, rank, local id and error trace. JSON round-trippable.
    """
    Dataclass to represent any rendezvous event.

    Args:
        name: Event name. (E.g. Current action being performed)
        run_id: The run id of the rendezvous
        message: The message describing the event
        hostname: Hostname of the node
        pid: The process id of the node
        node_state: The state of the node (INIT, RUNNING, SUCCEEDED, FAILED)
        master_endpoint: The master endpoint for the rendezvous store, if known
        rank: The rank of the node, if known
        local_id: The local_id of the node, if defined in dynamic_rendezvous.py
        error_trace: Error stack trace, if this is an error event.
    """

    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str = ""
    rank: int | None = None
    local_id: int | None = None
    error_trace: str = ""

    def __str__(self):
        # [FUNCTION DOC] Returns the JSON serialization of the event.
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "RdzvEvent"]) -> "RdzvEvent":
        # [FUNCTION DOC] Returns an RdzvEvent: an RdzvEvent input is returned unchanged; a JSON
        # string is parsed and its node state name converted to NodeState before construction.
        # KNOWN ISSUE: input of any other type fails with an unbound-variable error.
        if isinstance(data, RdzvEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["node_state"] = NodeState[data_dict["node_state"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        return RdzvEvent(**data_dict)

    def serialize(self) -> str:
        # [FUNCTION DOC] Returns the event converted to a plain mapping and dumped as a JSON string.
        return json.dumps(asdict(self))
