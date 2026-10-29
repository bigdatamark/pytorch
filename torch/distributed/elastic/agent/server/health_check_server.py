#!/usr/bin/env python3
# [MODULE DOC] No-op health check server for the elastic agent. Defines the HealthCheckServer
# interface (a stub that only logs) and a factory function; downstream users may substitute a
# real tcp/http implementation. Import creates a module logger only.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

from torch.distributed.elastic.utils.logging import get_logger


log = get_logger(__name__)

__all__ = ["HealthCheckServer", "create_healthcheck_server"]


class HealthCheckServer:
    # [CLASS DOC] Interface/no-op base for an agent health check server. Holds the alive callback
    # (returns the agent's last progress time), the port and the timeout. Subclasses are expected
    # to serve alive/dead status over tcp/http; this base class only logs in start and stop.
    """
    Interface for health check monitoring server, which can be extended
    by starting tcp/http server on the specified port.

    Args:

        alive_callback: Callable[[], int], callback to last progress time of agent

        port: int, port number to start tcp/http server

        timeout: int, timeout seconds to decide agent is alive/dead
    """

    _alive_callback: Callable[[], int]
    _port: int
    _timeout: int

    def __init__(
        self, alive_callback: Callable[[], int], port: int, timeout: int
    ) -> None:
        # [FUNCTION DOC] Stores the progress callback, port and timeout seconds on the instance.
        # Starts nothing and has no other side effects.
        self._alive_callback = alive_callback
        self._port = port
        self._timeout = timeout

    def start(self) -> None:
        # [FUNCTION DOC] Does not start any server; only logs a warning that none was started.
        """
        Unsupported functionality for PyTorch, doesn't start any health check server
        """
        log.warning("No health check server started")

    def stop(self) -> None:
        # [FUNCTION DOC] No-op stop; only logs an info message.
        """
        Function to stop health check server
        """
        log.info("Stopping noop health check server.")

    @property
    def alive_callback(self) -> Callable[[], int]:
        # [FUNCTION DOC] Property returning the stored callback that reports the agent's last progress time.
        return self._alive_callback


def create_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> HealthCheckServer:
    # [FUNCTION DOC] Factory: builds and returns a HealthCheckServer from the callback, port and
    # timeout. The server is not started.
    """
    creates health check server object
    """
    return HealthCheckServer(alive_callback, port, timeout)
