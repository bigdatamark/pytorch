# Owner(s): ["oncall: r2p"]
# [MODULE DOC] Tests for the c10d rendezvous backend: the shared backend test mixin run
# against a TCPStore and a FileStore, and create_backend behavior for TCP and file stores
# (store type, host/port or file path defaults, is_host handling, read timeout defaults and
# validation, connection and file errors). Uses real local stores, free ports and temp files
# and imports rendezvous_backend_test from the same directory (needs that directory on the
# import path). Plain unittest with no run_tests entry point.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from base64 import b64encode
from collections.abc import Callable
from datetime import timedelta
from typing import cast, ClassVar
from unittest import mock, TestCase

from rendezvous_backend_test import RendezvousBackendTestMixin

from torch.distributed import FileStore, TCPStore
from torch.distributed.elastic.rendezvous import (
    RendezvousConnectionError,
    RendezvousError,
    RendezvousParameters,
)
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
    C10dRendezvousBackend,
    create_backend,
)
from torch.distributed.elastic.utils.distributed import get_free_port


class TCPStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    # [CLASS DOC] Runs the generic backend test mixin against C10dRendezvousBackend backed by
    # one class-wide master TCPStore on an OS-chosen port; each test builds a fresh backend
    # for run id dummy_run_id after deleting that key.
    _store: ClassVar[TCPStore]

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] Starts a master TCPStore on localhost with an OS-assigned port and
        # keeps it as the shared class store.
        cls._store = TCPStore("localhost", 0, is_master=True)  # type: ignore[call-arg]

    def setUp(self) -> None:
        # [FUNCTION DOC] Deletes the rendezvous state key for dummy_run_id from the shared
        # store and creates a new C10dRendezvousBackend over it.
        super().setUp()
        # Make sure we have a clean slate.
        self._store.delete_key("torch.rendezvous.dummy_run_id")

        self._backend = C10dRendezvousBackend(self._store, "dummy_run_id")

    def _corrupt_state(self) -> None:
        # [FUNCTION DOC] HELPER: Hook used by the mixin; overwrites the stored state with a
        # value that is not valid base64.
        self._store.set("torch.rendezvous.dummy_run_id", "non_base64")


class FileStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    # [CLASS DOC] Runs the generic backend test mixin against C10dRendezvousBackend backed by
    # a FileStore on a new temp file per test (FileStore has no delete_key, so it is
    # recreated for a clean slate).
    _store: ClassVar[FileStore]

    def setUp(self) -> None:
        # [FUNCTION DOC] Creates a temp file, a FileStore on it and a backend for
        # dummy_run_id; remembers the path for cleanup.
        super().setUp()
        _, path = tempfile.mkstemp()
        self._path = path

        # Currently, filestore doesn't implement a delete_key method, so a new
        # filestore has to be initialized for every test in order to have a
        # clean slate.
        self._store = FileStore(path)
        self._backend = C10dRendezvousBackend(self._store, "dummy_run_id")

    def tearDown(self) -> None:
        # [FUNCTION DOC] Removes the temp file backing the store.
        os.remove(self._path)

    def _corrupt_state(self) -> None:
        # [FUNCTION DOC] HELPER: Hook used by the mixin; overwrites the stored state with a
        # value that is not valid base64.
        self._store.set("torch.rendezvous.dummy_run_id", "non_base64")


class CreateBackendTest(TestCase):
    # [CLASS DOC] Tests create_backend. State: default TCP parameters on a free port
    # (_params), file-store parameters on a temp file (_params_filestore), and the expected
    # store type, endpoint and read timeout that the shared assertion helper checks.
    def setUp(self) -> None:
        # [FUNCTION DOC] Picks a free port, builds TCP parameters (is_host true, tcp store
        # type in odd case, read timeout 10) and file-store parameters on a new temp file,
        # and sets the expected store type (TCPStore), host, port and 10 second timeout.
        super().setUp()
        # For testing, the default parameters used are for tcp. If a test
        # uses parameters for file store, we set the self._params to
        # self._params_filestore.

        port = get_free_port()
        self._params = RendezvousParameters(
            backend="dummy_backend",
            endpoint=f"localhost:{port}",
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            is_host="true",
            store_type="tCp",
            read_timeout="10",
        )

        _, tmp_path = tempfile.mkstemp()

        # Parameters for filestore testing.
        self._params_filestore = RendezvousParameters(
            backend="dummy_backend",
            endpoint=tmp_path,
            run_id="dummy_run_id",
            min_nodes=1,
            max_nodes=1,
            store_type="fIlE",
        )
        self._expected_endpoint_file = tmp_path
        self._expected_temp_dir = tempfile.gettempdir()

        self._expected_endpoint_host = "localhost"
        self._expected_endpoint_port = port
        self._expected_store_type = TCPStore
        self._expected_read_timeout = timedelta(seconds=10)

    def tearDown(self) -> None:
        # [FUNCTION DOC] Removes the temp file created for the file-store parameters.
        os.remove(self._expected_endpoint_file)

    def _run_test_with_store(self, store_type: str, test_to_run: Callable):
        # [FUNCTION DOC] HELPER: For store_type "file", switches the parameters and expectations
        # to the file store (FileStore type, 300 second default timeout); then always calls
        # test_to_run. Any other store type leaves the TCP defaults.
        """
        Use this function to specify the store type to use in a test. If
        not used, the test will default to TCPStore.
        """
        if store_type == "file":
            self._params = self._params_filestore
            self._expected_store_type = FileStore
            self._expected_read_timeout = timedelta(seconds=300)

        test_to_run()

    def _assert_create_backend_returns_backend(self) -> None:
        # [FUNCTION DOC] HELPER: Calls create_backend with the current parameters and asserts
        # the backend name is c10d, the store has the expected type and read timeout, the TCP
        # host and port or the file path (or a path under the temp dir when no endpoint) match,
        # and that setting backend state stores its base64 encoding under the run id key.
        backend, store = create_backend(self._params)

        self.assertEqual(backend.name, "c10d")

        self.assertIsInstance(store, self._expected_store_type)

        typecast_store = cast(self._expected_store_type, store)
        self.assertEqual(typecast_store.timeout, self._expected_read_timeout)  # type: ignore[attr-defined]
        if self._expected_store_type == TCPStore:
            self.assertEqual(typecast_store.host, self._expected_endpoint_host)  # type: ignore[attr-defined]
            self.assertEqual(typecast_store.port, self._expected_endpoint_port)  # type: ignore[attr-defined]
        if self._expected_store_type == FileStore:
            if self._params.endpoint:
                self.assertEqual(typecast_store.path, self._expected_endpoint_file)  # type: ignore[attr-defined]
            else:
                self.assertTrue(typecast_store.path.startswith(self._expected_temp_dir))  # type: ignore[attr-defined]

        backend.set_state(b"dummy_state")

        state = store.get("torch.rendezvous." + self._params.run_id)

        self.assertEqual(state, b64encode(b"dummy_state"))

    def test_create_backend_returns_backend(self) -> None:
        # [FUNCTION DOC] Runs the backend-creation assertions for both the tcp and file store
        # types as subtests.
        for store_type in ["tcp", "file"]:
            with self.subTest(store_type=store_type):
                self._run_test_with_store(
                    store_type, self._assert_create_backend_returns_backend
                )

    def test_create_backend_returns_backend_if_is_host_is_false(self) -> None:
        # [FUNCTION DOC] Starts a master TCPStore on the expected port, sets is_host to false
        # and checks create_backend connects to it as a client.
        store = TCPStore(  # type: ignore[call-arg] # noqa: F841
            self._expected_endpoint_host,
            self._expected_endpoint_port,
            is_master=True,
        )

        self._params.config["is_host"] = "false"

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified(self) -> None:
        # [FUNCTION DOC] With the is_host option removed, create_backend must still succeed
        # (it hosts the store itself when none exists).
        del self._params.config["is_host"]

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified_and_store_already_exists(
        self,
    ) -> None:
        # [FUNCTION DOC] With a master TCPStore already running on the port and no is_host
        # option, create_backend must still succeed (it falls back to connecting as a client).
        TCPStore(  # type: ignore[call-arg]
            self._expected_endpoint_host, self._expected_endpoint_port, is_master=True
        )

        del self._params.config["is_host"]

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_port_is_not_specified(
        self,
    ) -> None:
        # [FUNCTION DOC] With the module default port patched to the free port and an
        # endpoint containing only the host, create_backend must use the default port.
        # patch default port and pass endpoint with no port specified
        with mock.patch(
            "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.DEFAULT_PORT",
            self._expected_endpoint_port,
        ):
            self._params.endpoint = self._expected_endpoint_host

            self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_file_is_not_specified(
        self,
    ) -> None:
        # [FUNCTION DOC] With an empty file-store endpoint, create_backend must create the
        # store file in the temp directory.
        self._params_filestore.endpoint = ""

        self._run_test_with_store("file", self._assert_create_backend_returns_backend)

    def test_create_backend_returns_backend_if_store_type_is_not_specified(
        self,
    ) -> None:
        # [FUNCTION DOC] Without the store_type option create_backend defaults to a TCPStore;
        # the expected timeout becomes the 60 second default if no read_timeout is given.
        del self._params.config["store_type"]

        self._expected_store_type = TCPStore
        if not self._params.get("read_timeout"):
            self._expected_read_timeout = timedelta(seconds=60)

        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(
        self,
    ) -> None:
        # [FUNCTION DOC] Without the read_timeout option the store timeout defaults to
        # 60 seconds.
        del self._params.config["read_timeout"]

        self._expected_read_timeout = timedelta(seconds=60)

        self._assert_create_backend_returns_backend()

    def test_create_backend_raises_error_if_store_is_unreachable(self) -> None:
        # [FUNCTION DOC] As a client (is_host false) with a 2 second timeout and no server
        # listening, create_backend must raise RendezvousConnectionError.
        self._params.config["is_host"] = "false"
        self._params.config["read_timeout"] = "2"

        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to the C10d store has failed. See inner exception for details.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_endpoint_is_invalid(self) -> None:
        # [FUNCTION DOC] For both host and client modes, an endpoint that is not a valid
        # address must make create_backend raise RendezvousConnectionError.
        for is_host in [True, False]:
            with self.subTest(is_host=is_host):
                self._params.config["is_host"] = str(is_host)

                self._params.endpoint = "dummy_endpoint"

                with self.assertRaisesRegex(
                    RendezvousConnectionError,
                    r"^The connection to the C10d store has failed. See inner exception for "
                    r"details.$",
                ):
                    create_backend(self._params)

    def test_create_backend_raises_error_if_store_type_is_invalid(self) -> None:
        # [FUNCTION DOC] An unknown store_type makes create_backend raise ValueError stating
        # only file and tcp are supported.
        self._params.config["store_type"] = "dummy_store_type"

        with self.assertRaisesRegex(
            ValueError,
            r"^Invalid store type given. Currently only supports file and tcp.$",
        ):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        # [FUNCTION DOC] Read timeouts 0 and -10 make create_backend raise ValueError saying
        # the timeout must be a positive integer.
        for read_timeout in ["0", "-10"]:
            with self.subTest(read_timeout=read_timeout):
                self._params.config["read_timeout"] = read_timeout

                with self.assertRaisesRegex(
                    ValueError, r"^The read timeout must be a positive integer.$"
                ):
                    create_backend(self._params)

    @mock.patch("tempfile.mkstemp")
    def test_create_backend_raises_error_if_tempfile_creation_fails(
        self, tempfile_mock
    ) -> None:
        # [FUNCTION DOC] With temp file creation patched to raise OSError and an empty file
        # endpoint, create_backend must raise RendezvousError about the failed file creation.
        tempfile_mock.side_effect = OSError("test error")
        # Set the endpoint to empty so it defaults to creating a temp file
        self._params_filestore.endpoint = ""
        with self.assertRaisesRegex(
            RendezvousError,
            r"The file creation for C10d store has failed. See inner exception for details.",
        ):
            create_backend(self._params_filestore)

    @mock.patch(
        "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.FileStore"
    )
    def test_create_backend_raises_error_if_file_path_is_invalid(
        self, filestore_mock
    ) -> None:
        # [FUNCTION DOC] With FileStore construction patched to raise RuntimeError for a bad
        # path, create_backend must raise RendezvousConnectionError.
        filestore_mock.side_effect = RuntimeError("test error")
        self._params_filestore.endpoint = "bad file path"
        with self.assertRaisesRegex(
            RendezvousConnectionError,
            r"^The connection to the C10d store has failed. See inner exception for "
            r"details.$",
        ):
            create_backend(self._params_filestore)
