# [MODULE DOC] Tests for the EtcdServer helper that launches a local etcd process: start and
# stop behavior, host, port and endpoint accessors, and a full single-node rendezvous through
# the server. Import-time side effect: if the CIRCLECI environment variable is set, prints a
# notice to stderr and exits the process with status 0, skipping all tests.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import unittest

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class EtcdServerTest(unittest.TestCase):
    # [CLASS DOC] Tests that start their own EtcdServer instances (no shared fixture) and always
    # stop them in a finally block.
    def test_etcd_server_start_stop(self):
        # [FUNCTION DOC] Starts a server and checks the port is positive, the host is localhost,
        # the endpoint is host and port joined by a colon, and the client exposes a version.
        # Stops the server afterward even on failure.
        server = EtcdServer()
        server.start()

        try:
            port = server.get_port()
            host = server.get_host()

            self.assertGreater(port, 0)
            self.assertEqual("localhost", host)
            self.assertEqual(f"{host}:{port}", server.get_endpoint())
            self.assertIsNotNone(server.get_client().version)
        finally:
            server.stop()

    def test_etcd_server_with_rendezvous(self):
        # [FUNCTION DOC] Starts a server and builds an etcd rendezvous handler with min and max
        # nodes of 1 for it (local address 127.0.0.1), calls next_rendezvous, and expects a
        # store, rank 0 and world size 1. Stops the server afterward.
        server = EtcdServer()
        server.start()

        try:
            endpoint = server.get_endpoint()
            rdzv_params = RendezvousParameters(
                backend="etcd",
                endpoint=endpoint,
                run_id="test_run_1",
                min_nodes=1,
                max_nodes=1,
                timeout=60,
                last_call_timeout=30,
                local_addr="127.0.0.1",
            )
            rdzv_handler = create_rdzv_handler(rdzv_params)
            rdzv_info = rdzv_handler.next_rendezvous()
            self.assertIsNotNone(rdzv_info.store)
            self.assertEqual(0, rdzv_info.rank)
            self.assertEqual(1, rdzv_info.world_size)
        finally:
            server.stop()
