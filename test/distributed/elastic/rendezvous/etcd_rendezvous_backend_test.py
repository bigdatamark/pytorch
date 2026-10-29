# [MODULE DOC] Tests for the etcd-v2 rendezvous backend and its create_backend factory.
# EtcdRendezvousBackendTest reuses the shared backend test mixin against a real local EtcdServer
# subprocess; CreateBackendTest checks parameter parsing (protocol, read timeout), error
# mapping, the stored state/store key layout and store blocking behavior. Importing requires the
# python etcd client; no import-time side effects beyond imports.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import threading
import time
from base64 import b64encode
from typing import cast, ClassVar
from unittest import TestCase

from etcd import EtcdKeyNotFound  # type: ignore[import]
from rendezvous_backend_test import RendezvousBackendTestMixin

from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.api import RendezvousStoreInfo
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import (
    create_backend,
    EtcdRendezvousBackend,
)
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore


class EtcdRendezvousBackendTest(TestCase, RendezvousBackendTestMixin):
    # [CLASS DOC] Runs the generic RendezvousBackendTestMixin tests against an
    # EtcdRendezvousBackend backed by one class-wide EtcdServer. Each test gets a fresh client,
    # the dummy key prefix is deleted first, and corruption is simulated by writing a value that
    # is not valid base64 under the run id key.
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Starts one EtcdServer for the whole class (stderr discarded)
        # and stores it on the class.
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Stops the class-wide EtcdServer started in setUpClass.
        cls._server.stop()

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Gets a client from the server, removes any leftover keys under
        # the dummy prefix (ignoring a missing-key error) and builds the backend under test with
        # run id dummy_run_id and that prefix.
        super().setUp()
        self._client = self._server.get_client()

        # Make sure we have a clean slate.
        try:
            self._client.delete("/dummy_prefix", recursive=True, dir=True)
        except EtcdKeyNotFound:
            pass

        self._backend = EtcdRendezvousBackend(
            self._client, "dummy_run_id", "/dummy_prefix"
        )

    def _corrupt_state(self) -> None:
        # [FUNCTION DOC] HELPER: Implements the mixin hook by writing a non-base64 string to the
        # backend's key, so that decoding the state fails.
        self._client.write("/dummy_prefix/dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    # [CLASS DOC] Tests for create_backend with an etcd server: the returned backend and store,
    # default and explicit protocol and read timeout, invalid values and an unreachable server.
    # Parameters are rebuilt in setUp for each test.
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Starts one EtcdServer shared by every test in this class.
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Stops the class-wide EtcdServer.
        cls._server.stop()

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Builds RendezvousParameters pointing at the server endpoint
        # with a mixed-case "hTTp" protocol and read_timeout "10", and records 10 as the
        # expected read timeout that the created store's client should report.
        super().setUp()
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint=self._server.get_endpoint(),
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            protocol="hTTp",
            read_timeout="10",
        )

        self._expected_read_timeout = 10

    def test_create_backend_returns_backend(self) -> None:
        # [FUNCTION DOC] Verifies create_backend returns an "etcd-v2" backend and an EtcdStore
        # whose client has the expected read timeout. Setting backend state must store the
        # base64 encoded bytes under the run id rendezvous key with a ttl of at most 7200
        # seconds, and setting a store key must store base64 encoded key and value under the
        # store prefix. Other tests reuse this body after altering parameters.
        backend, store = create_backend(self._params)

        self.assertEqual(backend.name, "etcd-v2")

        self.assertIsInstance(store, EtcdStore)

        etcd_store = cast(EtcdStore, store)

        self.assertEqual(etcd_store.client.read_timeout, self._expected_read_timeout)  # type: ignore[attr-defined]

        client = self._server.get_client()

        backend.set_state(b"dummy_state")

        result = client.get("/torch/elastic/rendezvous/" + self._params.run_id)

        self.assertEqual(result.value, b64encode(b"dummy_state").decode())
        self.assertLessEqual(result.ttl, 7200)

        store.set("dummy_key", "dummy_value")

        result = client.get("/torch/elastic/store/" + b64encode(b"dummy_key").decode())

        self.assertEqual(result.value, b64encode(b"dummy_value").decode())

    def test_create_backend_returns_backend_if_protocol_is_not_specified(self) -> None:
        # [FUNCTION DOC] Removes the protocol option and re-runs the returned-backend checks, so
        # the default protocol must work.
        del self._params.config["protocol"]

        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(
        self,
    ) -> None:
        # [FUNCTION DOC] Removes the read_timeout option, sets the expected timeout to the
        # default of 60 seconds and re-runs the returned-backend checks.
        del self._params.config["read_timeout"]

        self._expected_read_timeout = 60

        self.test_create_backend_returns_backend()

    def test_create_backend_raises_error_if_etcd_is_unreachable(self) -> None:
        # [FUNCTION DOC] Points the endpoint at an unreachable host and expects create_backend
        # to raise RendezvousConnectionError with the standard etcd connection failure message.
        self._params.endpoint = "dummy:1234"

        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to etcd has failed. See inner exception for details.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_protocol_is_invalid(self) -> None:
        # [FUNCTION DOC] Sets an unsupported protocol and expects create_backend to raise
        # ValueError saying the protocol must be HTTP or HTTPS.
        self._params.config["protocol"] = "dummy"

        with self.assertRaisesRegex(
            ValueError, r"^The protocol must be HTTP or HTTPS.$"
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        # [FUNCTION DOC] For read_timeout values "0" and "-10", expects create_backend to raise
        # ValueError saying the read timeout must be a positive integer (one subtest per value).
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                self._params.config["read_timeout"] = read_timeout

                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    create_backend(self._params)

    def test_get_waits_for_store_prefix_key(self) -> None:
        # [FUNCTION DOC] Checks that a store get on the master address key blocks until another
        # thread sets it. One thread reads the key and records value and elapsed time, another
        # sleeps 2 seconds then sets the value; the test expects the value "foo" and a wait of
        # at least 2 seconds. Backend state is set first so the store prefix exists.
        def store_get(store, result_dict):
            # [FUNCTION DOC] HELPER: Thread target; does a blocking get of the master address
            # key, stores the decoded string and the elapsed time (seconds) into the shared
            # result mapping.
            start_time = time.perf_counter()
            result_dict["get_result"] = store.get(
                RendezvousStoreInfo.MASTER_ADDR_KEY
            ).decode(encoding="UTF-8")
            end_time = time.perf_counter()
            result_dict["time"] = end_time - start_time

        def store_set(store):
            # [FUNCTION DOC] HELPER: Thread target; sleeps 2 seconds then sets the master
            # address key to the bytes foo in the store.
            time.sleep(2)
            store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, b"foo")

        backend, store = create_backend(self._params)
        backend.set_state(b"dummy_state")
        result_dict = {}

        get_thread = threading.Thread(target=store_get, args=(store, result_dict))
        set_thread = threading.Thread(target=store_set, args=(store,))

        get_thread.start()
        set_thread.start()

        get_thread.join()
        set_thread.join()

        if result_dict["get_result"] != "foo":
            raise AssertionError(
                f"Expected get_result == 'foo', got {result_dict['get_result']}"
            )
        if not (result_dict["time"] >= 2):
            raise AssertionError(f"Expected time >= 2, got {result_dict['time']}")
