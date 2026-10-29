#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Helpers for creating the TCPStore used by elastic rendezvous and finding free local ports: create_c10d_store, get_free_port and get_socket_with_port (public via __all__). Creates a module logger at import; defines constants for error message matching.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import socket
from contextlib import closing

import torch.distributed as dist
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.utils.store import barrier


__all__ = ["create_c10d_store", "get_free_port", "get_socket_with_port"]

logger = get_logger(__name__)

_ADDRESS_IN_USE = "address already in use"
_SOCKET_TIMEOUT = "Socket Timeout"

_TCP_STORE_INIT = "_tcp_store/num_members"


def create_c10d_store(
    is_server: bool,
    server_addr: str,
    server_port: int = -1,
    world_size: int = 1,
    timeout: float = (60 * 10),  # 10 min
    wait_for_workers: bool = True,
    retries=3,
    use_libuv: bool | None = None,
):
    # [FUNCTION DOC] Create a TCPStore (c10d) as server or client, optionally retrying on port collisions, and return it.
    # Args: is_server, server_addr, server_port (-1 means pick a free port; required when world_size > 1), world_size, timeout seconds, wait_for_workers (if true, also verifies all members joined via a barrier), retries, use_libuv (deprecated, ignored with a warning; libuv is controlled by the USE_LIBUV environment variable, default on).
    # Raises ValueError for the invalid port/world size combination, TimeoutError if workers did not all join, and DistNetworkError on bind failure after retries or other network errors.
    # # KNOWN ISSUE: when server_port is -1 the attempt counter starts at retries, so the less-than check against retries is never true and an address-in-use error is raised on the first attempt without retrying.
    if use_libuv is not None:
        logger.warning(
            "argument use_libuv is deprecated and ignored. Set USE_LIBUV environment "
            'variable to "0" to disable libuv, or "1" to enable it. If the env var '
            "is not set, libuv will be used by default."
        )

    # check os.environ for use_libuv
    use_libuv = os.environ.get("USE_LIBUV", "1") == "1"  # libuv is the default option

    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    if server_port != -1:
        logger.info("sever_port: %s, specified, ignoring retries", server_port)

    # only retry when server_port is NOT static
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        logger.info(
            "Creating c10d store on %s:%s\n"
            "  world_size  : %s\n"
            "  is_server   : %s\n"
            "  timeout(sec): %s\n"
            "  use_libuv   : %s\n",
            server_addr,
            port,
            world_size,
            is_server,
            timeout,
            use_libuv,
        )

        try:
            store = dist.TCPStore(
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
                wait_for_workers=wait_for_workers,
                use_libuv=use_libuv,
            )
            # skips full rank check when we don't have to wait for all workers
            if wait_for_workers:
                _check_full_rank(store, world_size, timeout=timeout)
            logger.info("Successfully created c10d store")
            return store
        except dist.DistNetworkError as e:
            if _ADDRESS_IN_USE in str(e).lower():  # this will only happen on the server
                if attempt < retries:
                    logger.warning(
                        "port: %s already in use, attempt: [%s/%s]",
                        port,
                        attempt,
                        retries,
                    )
                    attempt += 1
                else:
                    raise dist.DistNetworkError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                raise


def _check_full_rank(store, world_size, timeout):
    # [FUNCTION DOC] HELPER: Verify that all world_size members have joined by running a store barrier with the given timeout. Converts the barrier's socket timeout RuntimeError into a TimeoutError; other errors propagate.
    try:
        barrier(store, world_size, key_prefix=_TCP_STORE_INIT, barrier_timeout=timeout)
    except RuntimeError as e:
        if str(e) == _SOCKET_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for all {world_size} members to join"
            ) from e
        else:
            raise


def get_free_port():
    # [FUNCTION DOC] Return an unused TCP port number on localhost by binding a temporary socket and closing it. The port is not reserved and may be taken afterwards (race). Raises RuntimeError if no socket can be made.
    """
    Returns an unused port on localhost.

    This function finds an unused port on localhost by opening to socket to bind
    to a port and then closing it.

    Returns:
        int: an unused port on localhost

    Example:
        >>> # xdoctest: +SKIP("Nondeterministic")
        >>> get_free_port()
        63976

    .. note::
        The port returned by :func:`get_free_port` is not reserved and may be
        taken by another process after this function returns.
    """
    sock = get_socket_with_port()
    with closing(sock):
        return sock.getsockname()[1]


def get_socket_with_port() -> socket.socket:
    # [FUNCTION DOC] Return a listening localhost TCP socket bound to an OS-chosen free port, trying each address family from name resolution and logging failed attempts. The caller must close the socket. Raises RuntimeError if none can be created.
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            logger.warning("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")
