# [MODULE DOC] Tests for the legacy etcd rendezvous handler created by create_rdzv_handler,
# using a standalone EtcdServer. Import-time side effect: if the CIRCLECI environment variable
# is set, prints a notice to stderr and exits the process with status 0, which skips every test
# in this file.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import unittest
import uuid

from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer


if os.getenv("CIRCLECI"):
    print("T85992919 temporarily disabling in circle ci", file=sys.stderr)
    sys.exit(0)


class EtcdRendezvousTest(unittest.TestCase):
    # [CLASS DOC] Starts one single-process etcd server for the class and checks that the etcd
    # rendezvous handler can be created with minimal and extended parameters and reports its run
    # id and backend name.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Starts the standalone etcd server shared by all tests.
        # start a standalone, single process etcd server to use for all tests
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Stops the standalone etcd server.
        # stop the standalone etcd server
        cls._etcd_server.stop()

    def test_etcd_rdzv_basic_params(self):
        # [FUNCTION DOC] Creates a handler from the minimum parameters (etcd backend, server
        # endpoint, random run id, one min and max node) and expects a non-None handler.
        """
        Check that we can create the handler with a minimum set of
        params
        """
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=f"{uuid.uuid4()}",
            min_nodes=1,
            max_nodes=1,
        )
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)

    def test_etcd_rdzv_additional_params(self):
        # [FUNCTION DOC] Creates a handler with extra timeout, last_call_timeout and protocol
        # parameters and checks it is non-None and that get_run_id returns the run id passed in.
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )

        etcd_rdzv = create_rdzv_handler(rdzv_params)

        self.assertIsNotNone(etcd_rdzv)
        self.assertEqual(run_id, etcd_rdzv.get_run_id())

    def test_get_backend(self):
        # [FUNCTION DOC] Creates a handler with extended parameters and checks get_backend
        # returns the string "etcd".
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(
            backend="etcd",
            endpoint=f"{self._etcd_server.get_endpoint()}",
            run_id=run_id,
            min_nodes=1,
            max_nodes=1,
            timeout=60,
            last_call_timeout=30,
            protocol="http",
        )

        etcd_rdzv = create_rdzv_handler(rdzv_params)

        self.assertEqual("etcd", etcd_rdzv.get_backend())
