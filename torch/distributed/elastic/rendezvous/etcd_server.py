#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC] Utilities to launch and stop a local standalone etcd server subprocess (EtcdServer), mainly for single-node
# launches and tests, plus helpers find_free_port and stop_etcd. The etcd python package is imported optionally.
# No import-time side effects besides creating a logger.
import atexit
import logging
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
from typing import TextIO


try:
    import etcd  # type: ignore[import]
except ModuleNotFoundError:
    pass


logger = logging.getLogger(__name__)


def find_free_port():
    # [FUNCTION DOC] Reserves a free local port by binding and listening on a socket, trying each address family returned for localhost.
    # Returns the open socket (the caller reads its port and must close it before reuse). Prints a message and closes the socket on
    # a failed attempt. Raises RuntimeError if no socket could be created.
    # KNOWN ISSUE: if the very first socket creation fails, the cleanup refers to a socket variable that was never assigned.
    """
    Find a free port and binds a temporary socket to it so that the port can be "reserved" until used.

    .. note:: the returned socket must be closed before using the port,
              otherwise an ``address already in use`` error will happen.
              The socket should be held and closed as close to the
              consumer of the port as possible since otherwise, there
              is a greater chance of race-condition where a different
              process may see the port as being free and take it.

    Returns: a socket bound to the reserved free port

    Usage::

    sock = find_free_port()
    port = sock.getsockname()[1]
    sock.close()
    use_port(port)
    """
    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )

    for addr in addrs:
        family, type, proto, _, _ = addr
        try:
            s = socket.socket(family, type, proto)
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()  # type: ignore[possibly-undefined]
            print(f"Socket creation attempt failed: {e}")
    raise RuntimeError("Failed to create a socket")


def stop_etcd(subprocess, data_dir: str | None = None):
    # [FUNCTION DOC] Terminates and waits for the given process if it is still running, then deletes the data directory
    # (ignoring errors) if one is given. Logs both actions. The first parameter shadows the subprocess module name.
    if subprocess and subprocess.poll() is None:
        logger.info("stopping etcd server")
        subprocess.terminate()
        subprocess.wait()

    if data_dir:
        logger.info("deleting etcd data dir: %s", data_dir)
        shutil.rmtree(data_dir, ignore_errors=True)


class EtcdServer:
    # [CLASS DOC] Manages a local standalone etcd server process on random free client and peer ports at localhost.
    # State: port, host, etcd binary path (env var TORCHELASTIC_ETCD_BINARY_PATH, else a bin/etcd next to this file,
    # else etcd from PATH), a base data directory (given or a fresh temp dir), and the subprocess handle.
    # Key methods: start (with retries), stop, get_client, get_port, get_host, get_endpoint.
    # KNOWN ISSUE: the docstring documents an etcd_binary_path argument, but the constructor only takes data_dir.
    """
    .. note:: tested on etcd server v3.4.3.

    Starts and stops a local standalone etcd server on a random free
    port. Useful for single node, multi-worker launches or testing,
    where a sidecar etcd server is more convenient than having to
    separately setup an etcd server.

    This class registers a termination handler to shutdown the etcd
    subprocess on exit. This termination handler is NOT a substitute for
    calling the ``stop()`` method.

    The following fallback mechanism is used to find the etcd binary:

    1. Uses env var TORCHELASTIC_ETCD_BINARY_PATH
    2. Uses ``<this file root>/bin/etcd`` if one exists
    3. Uses ``etcd`` from ``PATH``

    Usage
    ::

     server = EtcdServer("/usr/bin/etcd", 2379, "/tmp/default.etcd")
     server.start()
     client = server.get_client()
     # use client
     server.stop()

    Args:
        etcd_binary_path: path of etcd server binary (see above for fallback path)
    """

    def __init__(self, data_dir: str | None = None):
        # [FUNCTION DOC] Resolves the etcd binary path and the base data directory (created via a temp dir if none given); no process is started.
        self._port = -1
        self._host = "localhost"

        root = os.path.dirname(__file__)
        default_etcd_bin = os.path.join(root, "bin/etcd")
        self._etcd_binary_path = os.environ.get(
            "TORCHELASTIC_ETCD_BINARY_PATH", default_etcd_bin
        )
        if not os.path.isfile(self._etcd_binary_path):
            self._etcd_binary_path = "etcd"

        self._base_data_dir = (
            data_dir if data_dir else tempfile.mkdtemp(prefix="torchelastic_etcd_data")
        )
        self._etcd_cmd = None
        self._etcd_proc: subprocess.Popen | None = None

    def _get_etcd_server_process(self) -> subprocess.Popen:
        # [FUNCTION DOC] HELPER: Returns the running etcd subprocess handle; raises RuntimeError if the server has not been started.
        if not self._etcd_proc:
            raise RuntimeError(
                "No etcd server process started. Call etcd_server.start() first"
            )
        else:
            return self._etcd_proc

    def get_port(self) -> int:
        # [FUNCTION DOC] Returns the port the server listens on (-1 before start).
        """Return the port the server is running on."""
        return self._port

    def get_host(self) -> str:
        # [FUNCTION DOC] Returns the host name the server listens on.
        """Return the host the server is running on."""
        return self._host

    def get_endpoint(self) -> str:
        # [FUNCTION DOC] Returns the endpoint string host:port.
        """Return the etcd server endpoint (host:port)."""
        return f"{self._host}:{self._port}"

    def start(
        self,
        timeout: int = 60,
        num_retries: int = 3,
        stderr: int | TextIO | None = None,
    ) -> None:
        """
        # [FUNCTION DOC] Starts the server and waits until it is ready. Tries up to num_retries times, each with its own numbered
        # subdirectory of the base data dir; on failure logs, stops the process and retries. After the final failure removes the base data
        # dir and re-raises the last error (e.g. TimeoutError or RuntimeError). Returns nothing.
        # KNOWN ISSUE: the atexit registration after the loop is unreachable, since the loop only exits by returning or raising, so no exit handler is installed.
        Start the server, and waits for it to be ready. When this function returns the server is ready to take requests.

        Args:
            timeout: time (in seconds) to wait for the server to be ready
                before giving up.
            num_retries: number of retries to start the server. Each retry
                will wait for max ``timeout`` before considering it as failed.
            stderr: the standard error file handle. Valid values are
                `subprocess.PIPE`, `subprocess.DEVNULL`, an existing file
                descriptor (a positive integer), an existing file object, and
                `None`.

        Raises:
            TimeoutError: if the server is not ready within the specified timeout
        """
        curr_retries = 0
        while True:
            try:
                data_dir = os.path.join(self._base_data_dir, str(curr_retries))
                os.makedirs(data_dir, exist_ok=True)
                return self._start(data_dir, timeout, stderr)
            except Exception as e:
                curr_retries += 1
                stop_etcd(self._etcd_proc)
                logger.warning(
                    "Failed to start etcd server, got error: %s, retrying", e
                )
                if curr_retries >= num_retries:
                    shutil.rmtree(self._base_data_dir, ignore_errors=True)
                    raise
        atexit.register(stop_etcd, self._etcd_proc, self._base_data_dir)

    def _start(
        self, data_dir: str, timeout: int = 60, stderr: int | TextIO | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Reserves two free ports (client and peer), builds the etcd command line with v2 API enabled and local URLs,
        # releases the ports, launches the subprocess (stderr as given) and waits for readiness for up to timeout seconds.
        sock = find_free_port()
        sock_peer = find_free_port()
        self._port = sock.getsockname()[1]
        peer_port = sock_peer.getsockname()[1]

        etcd_cmd = shlex.split(
            " ".join(
                [
                    self._etcd_binary_path,
                    "--enable-v2",
                    "--data-dir",
                    data_dir,
                    "--listen-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--advertise-client-urls",
                    f"http://{self._host}:{self._port}",
                    "--listen-peer-urls",
                    f"http://{self._host}:{peer_port}",
                ]
            )
        )

        logger.info("Starting etcd server: [%s]", etcd_cmd)

        sock.close()
        sock_peer.close()
        self._etcd_proc = subprocess.Popen(etcd_cmd, close_fds=True, stderr=stderr)
        self._wait_for_ready(timeout)

    def get_client(self):
        # [FUNCTION DOC] Returns a new etcd client for this server (v2 API prefix, 10 second read timeout).
        """Return an etcd client object that can be used to make requests to this server."""
        return etcd.Client(
            host=self._host, port=self._port, version_prefix="/v2", read_timeout=10
        )

    def _wait_for_ready(self, timeout: int = 60) -> None:
        # [FUNCTION DOC] HELPER: Polls once per second until the etcd client can report the server version, or timeout seconds pass.
        # Raises RuntimeError if the server process exits meanwhile, and TimeoutError on timeout.
        client = etcd.Client(
            host=f"{self._host}", port=self._port, version_prefix="/v2", read_timeout=5
        )
        max_time = time.time() + timeout

        while time.time() < max_time:
            if self._get_etcd_server_process().poll() is not None:
                # etcd server process finished
                exitcode = self._get_etcd_server_process().returncode
                raise RuntimeError(
                    f"Etcd server process exited with the code: {exitcode}"
                )
            try:
                logger.info("etcd server ready. version: %s", client.version)
                return
            except Exception:
                time.sleep(1)
        raise TimeoutError("Timed out waiting for etcd server to be ready!")

    def stop(self) -> None:
        """Stop the server and cleans up auto generated resources (e.g. data dir)."""
        # [FUNCTION DOC] Stops the server process and removes the base data directory (via stop_etcd).
        logger.info("EtcdServer stop method called")
        stop_etcd(self._etcd_proc, self._base_data_dir)
