#!/usr/bin/env python3
# mypy: allow-untyped-defs

# [MODULE DOC] Static rendezvous: a trivial rendezvous handler where ranks and world size are given up front and the
# only coordination is a TCPStore hosted by rank 0. Exports StaticTCPRendezvous and create_rdzv_handler.
# Default store timeout is 600 seconds. No import-time side effects besides a logger.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from typing import cast

from torch.distributed import PrefixStore, Store, TCPStore
from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
)
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint


__all__ = ["StaticTCPRendezvous", "create_rdzv_handler"]

logger = logging.getLogger(__name__)

_default_timeout_seconds = 600


class StaticTCPRendezvous(RendezvousHandler):
    """
    # [CLASS DOC] Rendezvous handler for a fixed, pre-assigned topology. Stores master address/port, rank, world size, run id and
    # a timeout (as a timedelta); the TCPStore is created lazily on the first next_rendezvous call and cached. The agent with rank 0 acts
    # as the TCPStore server. Closing/waiting concepts do not apply: is_closed is always false, set_closed is a no-op,
    # num_nodes_waiting is always 0, shutdown always returns true, and use_agent_store is true.
    Static rendezvous that is a wrapper around the TCPStore.

    Creates TCPStore based on the input parameters with the
    listener on the agent with group_rank=0
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        run_id: str,
        timeout: int,
    ):
        # [FUNCTION DOC] Stores the connection parameters and converts timeout seconds to a timedelta; the store starts unset. No I/O.
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.run_id = run_id
        self.timeout = datetime.timedelta(seconds=timeout)
        self._store: Store | None = None

    def get_backend(self) -> str:
        # [FUNCTION DOC] Returns the backend name "static".
        return "static"

    @property
    def use_agent_store(self) -> bool:
        # [FUNCTION DOC] Property: always true, meaning the agent-created store is used by the agent.
        return True

    def next_rendezvous(self) -> RendezvousInfo:
        # [FUNCTION DOC] Creates (once) a multi-tenant TCPStore at master address/port with world_size clients, hosting the server if rank is 0,
        # wraps it in a PrefixStore keyed by run_id and returns RendezvousInfo with the store, rank, world size and bootstrap store info
        # (master addr/port). Side effect: opens/binds a socket and logs. May raise store connection/timeout errors.
        logger.info("Creating TCPStore as the c10d::Store implementation")
        is_master = self.rank == 0
        if not self._store:
            self._store = TCPStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)
        # TCPStore server instance is used by trainer code
        bootstrap_store_info = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(
            store,
            self.rank,
            self.world_size,
            bootstrap_store_info,
        )

    def is_closed(self):
        # [FUNCTION DOC] Always returns false; a static rendezvous never closes.
        return False

    def set_closed(self):
        # [FUNCTION DOC] No-op.
        pass

    def num_nodes_waiting(self):
        # [FUNCTION DOC] Always returns 0.
        return 0

    def get_run_id(self) -> str:
        # [FUNCTION DOC] Returns the run id.
        return self.run_id

    def shutdown(self) -> bool:
        # [FUNCTION DOC] Does nothing and returns true.
        return True


def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    # [FUNCTION DOC] Factory: validates params and builds a StaticTCPRendezvous. Requires "rank" in the config and a non-empty endpoint
    # with an explicit port; world size is max_nodes; timeout comes from the optional config key "timeout" (int seconds) else 600.
    # Raises ValueError if rank, endpoint or port is missing.
    if "rank" not in params.config:
        raise ValueError(
            "rank is absent in RendezvousParameters."
            "Try add --node-rank to the cmd request"
        )
    endpoint = params.endpoint.strip()
    if not endpoint:
        raise ValueError(
            "endpoint is absent in RendezvousParameters"
            "Try add --master-port and --master-addr to the cmd request"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, -1)
    if master_port == -1:
        raise ValueError(
            f"Port is absent in endpoint: {endpoint}. Try launching with --master-port"
        )
    world_size = params.max_nodes
    rank = cast(int, params.config.get("rank"))
    run_id = params.run_id
    if "timeout" in params.config:
        timeout = int(params.config["timeout"])
    else:
        timeout = _default_timeout_seconds

    return StaticTCPRendezvous(
        master_addr, master_port, rank, world_size, run_id, timeout
    )
