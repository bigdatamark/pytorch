#!/usr/bin/env python3
# [MODULE DOC] Tests for create_c10d_store and get_socket_with_port in
# torch.distributed.elastic.utils.distributed: single and multi process TCPStore creation,
# port conflicts, timeouts and the USE_LIBUV switch. Import-time side effect: on Windows or
# macOS it prints a message to stderr and exits the interpreter with status 0, so no tests are
# collected there.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import socket
import sys
import unittest
from contextlib import closing

from torch.distributed import DistNetworkError, DistStoreError
from torch.distributed.elastic.utils.distributed import (
    create_c10d_store,
    get_socket_with_port,
)
from torch.testing._internal.common_utils import (
    IS_MACOS,
    IS_WINDOWS,
    MI200_ARCH,
    run_tests,
    skipIfRocmArch,
    TEST_WITH_TSAN,
    TestCase,
)


def _create_c10d_store_mp(is_server, server_addr, port, world_size, wait_for_workers):
    # [FUNCTION DOC] HELPER: Child-process target that creates a store with a 2 second timeout via
    # create_c10d_store using the given role, address, port, world size and wait_for_workers flag.
    # Raises AssertionError if no store is returned, then sets a key containing its own pid to a
    # fixed value so the parent can verify the worker connected.
    store = create_c10d_store(
        is_server,
        server_addr,
        port,
        world_size,
        wait_for_workers=wait_for_workers,
        timeout=2,
    )
    if store is None:
        raise AssertionError

    store.set(f"test_key/{os.getpid()}", b"test_value")


if IS_WINDOWS or IS_MACOS:
    print("tests incompatible with tsan or asan", file=sys.stderr)
    sys.exit(0)


class DistributedUtilTest(TestCase):
    # [CLASS DOC] Tests of create_c10d_store behavior for server and worker roles. Tests use the
    # local hostname as the address and port 0 where a free port is needed.
    def test_create_store_single_server(self):
        # [FUNCTION DOC] Creates a server store with default world size and checks a store object
        # is returned.
        store = create_c10d_store(is_server=True, server_addr=socket.gethostname())
        self.assertIsNotNone(store)

    def test_create_store_no_port_multi(self):
        # [FUNCTION DOC] Checks that creating a server for a world size of 2 without a port
        # raises ValueError.
        with self.assertRaises(ValueError):
            create_c10d_store(
                is_server=True, server_addr=socket.gethostname(), world_size=2
            )

    @unittest.skipIf(TEST_WITH_TSAN, "test incompatible with tsan")
    def test_create_store_multi(self):
        # [FUNCTION DOC] Creates a server store on a free port for world size 3 without waiting
        # for workers, starts two worker processes that connect and set pid-keyed values, then
        # checks the server can read both values and both workers exit with code 0. Skipped under
        # thread sanitizer.
        world_size = 3
        wait_for_workers = False
        localhost = socket.gethostname()

        # start the server on the main process using an available port
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )

        # worker processes will use the port that was assigned to the server
        server_port = store.port

        worker0 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )
        worker1 = mp.Process(
            target=_create_c10d_store_mp,
            args=(False, localhost, server_port, world_size, wait_for_workers),
        )

        worker0.start()
        worker1.start()

        worker0.join()
        worker1.join()

        # check test_key/pid == "test_value"
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker0.pid}").decode("UTF-8")
        )
        self.assertEqual(
            "test_value", store.get(f"test_key/{worker1.pid}").decode("UTF-8")
        )

        self.assertEqual(0, worker0.exitcode)
        self.assertEqual(0, worker1.exitcode)

    def test_create_store_timeout_on_server(self):
        # [FUNCTION DOC] Checks a server expecting two participants on a free port with a 1 second
        # timeout raises DistStoreError when no worker connects.
        with self.assertRaises(DistStoreError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=True,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_timeout_on_worker(self):
        # [FUNCTION DOC] Checks a worker connecting to port 0 with no server raises
        # DistNetworkError within the 1 second timeout.
        with self.assertRaises(DistNetworkError):
            # use any available port (port 0) since timeout is expected
            create_c10d_store(
                is_server=False,
                server_addr=socket.gethostname(),
                server_port=0,
                world_size=2,
                timeout=1,
            )

    def test_create_store_with_libuv_support(self):
        # [FUNCTION DOC] Sets the USE_LIBUV environment variable to 0 and checks the created store
        # reports libuv backend disabled, then removes the variable and checks the default store
        # has libuv enabled.
        # KNOWN ISSUE: if the first assertion fails the environment variable is left set for later
        # tests, since no cleanup guard is used.
        world_size = 1
        wait_for_workers = False
        localhost = socket.gethostname()

        os.environ["USE_LIBUV"] = "0"
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertFalse(store.libuvBackend)
        del os.environ["USE_LIBUV"]
        if "USE_LIBUV" in os.environ:
            raise AssertionError("Expected USE_LIBUV to be removed from os.environ")

        # libuv backend is enabled by default
        store = create_c10d_store(
            is_server=True,
            server_addr=localhost,
            server_port=0,
            timeout=2,
            world_size=world_size,
            wait_for_workers=wait_for_workers,
        )
        self.assertTrue(store.libuvBackend)

    def test_port_already_in_use_on_server(self):
        # [FUNCTION DOC] Creates a server store on a free port, then tries to create a second
        # server on the same port and checks DistNetworkError is raised.
        # try to create the TCPStore server twice on the same port
        # the second should fail due to a port conflict
        # first store binds onto a free port
        # try creating the second store on the port that the first store binded to
        server_addr = socket.gethostname()
        pick_free_port = 0
        store1 = create_c10d_store(
            is_server=True,
            server_addr=server_addr,
            server_port=pick_free_port,
            timeout=1,
        )
        with self.assertRaises(DistNetworkError):
            create_c10d_store(
                is_server=True, server_addr=server_addr, server_port=store1.port
            )

    @skipIfRocmArch(MI200_ARCH)
    def test_port_already_in_use_on_worker(self):
        # [FUNCTION DOC] Binds a socket with get_socket_with_port, then checks a worker store
        # pointed at that port just times out with DistNetworkError since no server exists. Skipped
        # on ROCm MI200.
        sock = get_socket_with_port()
        with closing(sock):
            port = sock.getsockname()[1]
            # on the worker port conflict shouldn't matter, it should just timeout
            # since we never created a server
            with self.assertRaises(DistNetworkError):
                create_c10d_store(
                    is_server=False,
                    server_addr=socket.gethostname(),
                    server_port=port,
                    timeout=1,
                )


if __name__ == "__main__":
    run_tests()
