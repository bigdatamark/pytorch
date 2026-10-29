# [MODULE DOC] Tests for the static TCP rendezvous handler (create_rdzv_handler in
# static_tcp_rendezvous): rejection of bad endpoints, backend name and repeated rendezvous.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from contextlib import closing

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.static_tcp_rendezvous import (
    create_rdzv_handler,
)
from torch.distributed.elastic.utils import get_socket_with_port


class StaticTCPRendezvousTest(unittest.TestCase):
    # [CLASS DOC] Builds RendezvousParameters for the "static" backend in each test and checks
    # the handler created from them.
    def test_missing_port(self):
        # [FUNCTION DOC] Endpoint "localhost" has no port, so create_rdzv_handler must
        # raise ValueError.
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_empty_endpoint(self):
        # [FUNCTION DOC] An empty endpoint must make create_rdzv_handler raise ValueError.
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr(self):
        # [FUNCTION DOC] A full IPv6 address with a port as the endpoint must make
        # create_rdzv_handler raise ValueError (IPv6 is rejected by the static handler).
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr_localhost(self):
        # [FUNCTION DOC] The IPv6 loopback endpoint with a port must also make
        # create_rdzv_handler raise ValueError.
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="[::1]:90",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
        )
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_get_backend(self):
        # [FUNCTION DOC] With a valid endpoint, timeout 60 and rank 0, the handler's get_backend
        # must return "static".
        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint="localhost:123",
            run_id="test",
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            rank=0,
        )

        static_rdzv = create_rdzv_handler(rdzv_params)
        self.assertEqual("static", static_rdzv.get_backend())

    def test_static_rdzv_multiple_calls(self):
        # [FUNCTION DOC] Reserves a free local port by opening and closing a socket, creates a
        # static handler for localhost on that port with rank 0, and calls next_rendezvous
        # twice; each result must have a store, rank 0 and world size 1.
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        master_addr = "localhost"

        rdzv_params = RendezvousParameters(
            backend="static",
            endpoint=f"{master_addr}:{master_port}",
            run_id="test_id",
            min_nodes=1,
            max_nodes=1,
            rank=0,
        )
        rdzv_handler = create_rdzv_handler(rdzv_params)

        # Call rendezvous two times
        rdzv_info = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(rdzv_info.store)
        self.assertEqual(0, rdzv_info.rank)
        self.assertEqual(1, rdzv_info.world_size)

        rdzv_info = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(rdzv_info.store)
        self.assertEqual(0, rdzv_info.rank)
        self.assertEqual(1, rdzv_info.world_size)
