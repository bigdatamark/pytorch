#!/usr/bin/env python3
# [MODULE DOC] Tests for the torch.distributed.elastic.control_plane worker HTTP server. A
# helper starts the server on a Unix domain socket in a temporary directory and the tests send
# HTTP requests (via urllib3) to check the root page, handler listing, ping, the NCCL trace dump
# handlers with query parameter validation, traceback dump, wait counter metrics, and C++
# handler registry bindings. Runs via run_tests when executed directly.
# Owner(s): ["oncall: distributed"]

import json
import os
import pickle
import socket
import tempfile
import unittest
from contextlib import contextmanager

from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,
    worker_main,
)
from torch.monitor import _WaitCounter
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    MI200_ARCH,
    requires_cuda,
    run_tests,
    skipIfRocmArch,
    TestCase,
)


class UnixHTTPConnection(HTTPConnection):
    # [CLASS DOC] HELPER: urllib3 HTTP connection that connects over an AF_UNIX stream socket
    # to a path instead of TCP. Stores the socket path; the host name passed to the base class
    # is a dummy.
    def __init__(self, socket_path: str) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base connection for host localhost and stores
        # the Unix socket path.
        super().__init__("localhost")

        self.socket_path = socket_path

    def connect(self) -> None:
        # [FUNCTION DOC] HELPER: Overrides connect to create a Unix stream socket and connect it
        # to the stored path; assigns it to the connection's sock attribute.
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixHTTPConnectionPool(HTTPConnectionPool):
    # [CLASS DOC] HELPER: urllib3 connection pool whose connections are UnixHTTPConnection
    # objects for one socket path.
    def __init__(self, socket_path: str) -> None:
        # [FUNCTION DOC] HELPER: Initializes the base pool for host localhost and stores the
        # socket path.
        super().__init__("localhost")

        self.socket_path = socket_path

    def _new_conn(self):
        # [FUNCTION DOC] HELPER: Creates a new UnixHTTPConnection for the stored socket path.
        return UnixHTTPConnection(self.socket_path)


@contextmanager
def local_worker_server() -> None:
    # [FUNCTION DOC] HELPER: Context manager that creates a temporary directory, sets the
    # TORCH_WORKER_SERVER_SOCKET environment variable to a socket file inside it (the variable
    # is left set afterward), starts the worker server with worker_main, and yields a Unix
    # socket connection pool for it. The server is stopped and the directory removed on exit.
    # KNOWN ISSUE: the return annotation says None although it is a generator context manager.
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "socket.sock")
        os.environ[TORCH_WORKER_SERVER_SOCKET] = socket_path

        with worker_main():
            pool = UnixHTTPConnectionPool(socket_path)
            yield pool


class WorkerServerTest(TestCase):
    # [CLASS DOC] Tests for the worker server and its registered request handlers. Most tests
    # start a local server through the local_worker_server helper; a few call the C++ handler
    # registry bindings directly.
    def test_worker_server(self) -> None:
        # [FUNCTION DOC] Checks the root page returns 200 with the expected HTML title and a
        # link to the handler list. Checks POST to the ping handler returns 200 with body pong,
        # GET of the handler list returns 200 with JSON containing ping, and an unknown handler
        # returns 404 with a "Handler nonexistent not found" message.
        with local_worker_server() as pool:
            resp = pool.request("GET", "/")
            self.assertEqual(resp.status, 200)
            self.assertEqual(
                resp.data,
                b"<h1>torch.distributed.WorkerServer</h1>\n"
                b'<a href="'
                b"/handler/"
                b'">Handler names</a>\n',
            )

            resp = pool.request("POST", "/handler/ping")
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.data, b"pong")

            resp = pool.request("GET", "/handler/")
            self.assertEqual(resp.status, 200)
            self.assertIn("ping", json.loads(resp.data))

            resp = pool.request("POST", "/handler/nonexistent")
            self.assertEqual(resp.status, 404)
            self.assertIn(b"Handler nonexistent not found:", resp.data)

    @requires_cuda
    def test_dump_nccl_trace_pickle(self) -> None:
        # [FUNCTION DOC] Requires CUDA. POST to the NCCL trace pickle handler must return 200
        # and a body that unpickles to a dict with a version key.
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_nccl_trace_pickle")
            self.assertEqual(resp.status, 200)
            out = pickle.loads(resp.data)
            self.assertIsInstance(out, dict)
            self.assertIn("version", out)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_params(self) -> None:
        # [FUNCTION DOC] Requires CUDA. Checks query parameter validation for the NCCL trace
        # pickle handler: a camel-case key, an unknown key, a non-boolean value and a
        # capitalized boolean value each return 400, while lowercase includecollectives=true and
        # a combination of multiple valid parameters return 200.
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?unknownkey=true"
            )
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_pickle?includecollectives=true&includestacktraces=false&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_json(self) -> None:
        # [FUNCTION DOC] Requires CUDA. Same parameter validation checks as the pickle test but
        # for the NCCL trace JSON handler (bad key case, unknown key, bad values give 400; valid
        # single and combined parameters give 200). The test name says pickle although it
        # targets JSON.
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request("POST", "/handler/dump_nccl_trace_json?unknownkey=true")
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_json?includecollectives=true&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    @skipIfRocmArch(MI200_ARCH)
    def test_tcp(self) -> None:
        # [FUNCTION DOC] Skipped on ROCm MI200. Starts a C++ _WorkerServer on TCP port 1234,
        # issues a real HTTP GET for the handler list using the requests library and expects
        # status 200, then shuts the server down. Uses a fixed port, so it can conflict with
        # other users of it.
        import requests

        from torch._C._distributed_c10d import _WorkerServer

        server = _WorkerServer("", 1234)
        out = requests.get("http://localhost:1234/handler/")
        self.assertEqual(out.status_code, 200)

        server.shutdown()

    def test_dump_traceback(self) -> None:
        # [FUNCTION DOC] POST to the traceback dump handler must return 200 with a body
        # containing the name of this test function, showing that the server dumps the python
        # stacks of the process.
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_traceback")
            self.assertEqual(resp.status, 200)
            self.assertIn(b"in test_dump_traceback\n", resp.data)

    def test_run_handler(self) -> None:
        # [FUNCTION DOC] Fetches the ping handler from the C++ registry and calls it directly
        # with fake request (body "dummy", no params) and response objects defined locally.
        # Expects status 200, content "pong" and content type text/plain.
        from torch._C._distributed_c10d import _get_handler, _Request, _Response

        handler = _get_handler("ping")

        class Request(_Request):
            # [CLASS DOC] HELPER: Minimal subclass of the C++ _Request used to call a handler
            # directly; returns a fixed body and empty parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the C++ base class.
                _Request.__init__(self)

            def body(self) -> bytes:
                # [FUNCTION DOC] HELPER: Returns the fixed request body b"dummy".
                return b"dummy"

            def params(self) -> dict[str, str]:
                # [FUNCTION DOC] HELPER: Returns an empty parameter mapping.
                return {}

        class Response(_Response):
            # [CLASS DOC] HELPER: Minimal subclass of the C++ _Response that records the status,
            # content and content type set by a handler in plain attributes.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the C++ base class.
                _Response.__init__(self)

            def set_content(self, content: str, content_type: str) -> None:
                # [FUNCTION DOC] HELPER: Stores the content string and content type on
                # the object.
                self.content = content
                self.content_type = content_type

            def set_status(self, status: int) -> None:
                # [FUNCTION DOC] HELPER: Stores the HTTP status code on the object.
                self.status = status

        req = Request()
        resp = Response()

        handler(req, resp)

        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.content, "pong")
        self.assertEqual(resp.content_type, "text/plain")

    def test_get_handler_nonexistant(self) -> None:
        # [FUNCTION DOC] Looking up a handler name that is not registered must raise ValueError
        # with a "Failed to find handler nonexistent" message.
        from torch._C._distributed_c10d import _get_handler

        with self.assertRaisesRegex(ValueError, "Failed to find handler nonexistent"):
            _get_handler("nonexistent")

    def test_get_handler_names(self) -> None:
        # [FUNCTION DOC] The list of registered handler names from the C++ registry must
        # contain ping.
        from torch._C._distributed_c10d import _get_handler_names

        names = _get_handler_names()
        self.assertIn("ping", names)

    @unittest.skipIf(IS_FBCODE, "disabled in FBCODE")
    def test_wait_counter_values(self) -> None:
        # [FUNCTION DOC] Skipped in FBCODE. Creates a _WaitCounter, uses its guard three times,
        # then POSTs to the wait counter values handler. Expects 200 and a JSON mapping that
        # includes the counter name with active_count, total_calls, total_time_us and
        # max_time_us; total_calls must be 3, active_count 0, and the times non-negative.
        # Mutates global wait counter metrics.
        """
        Test that WaitCounter values are properly tracked and returned by the handler.

        Note: This test may trigger an ASAN heap-use-after-free error during process
        shutdown due to static destruction order issues with boost regex in the logging
        framework. The test assertions pass successfully before this shutdown error occurs.
        """
        with local_worker_server() as pool:
            # Create and use a WaitCounter with a specific name
            counter_name = "test_counter"
            counter = _WaitCounter(counter_name)

            # Use the counter multiple times to generate metrics
            # Note: Using minimal/no sleep to avoid timing issues
            for i in range(3):
                with counter.guard():
                    pass  # Minimal work

            # Query the wait counter values
            resp = pool.request("POST", "/handler/wait_counter_values")
            self.assertEqual(resp.status, 200)

            # Parse the JSON response
            data = json.loads(resp.data)
            # Should be a dictionary
            self.assertIsInstance(data, dict)

            # Verify our test counter appears in the response
            self.assertIn(
                counter_name,
                data,
                lambda msg: f"{msg}\nCounter '{counter_name}' not found in response. Available counters: {list(data.keys())}",
            )

            # Verify the counter has expected metrics
            counter_data = data[counter_name]
            self.assertIn("active_count", counter_data)
            self.assertIn("total_calls", counter_data)
            self.assertIn("total_time_us", counter_data)
            self.assertIn("max_time_us", counter_data)

            # Verify the counter was called 3 times
            self.assertEqual(
                counter_data["total_calls"],
                3,
                lambda msg: f"{msg}\nExpected 3 calls, got {counter_data['total_calls']}",
            )

            # Verify active_count is 0 (no active waiters)
            self.assertEqual(
                counter_data["active_count"],
                0,
                lambda msg: f"{msg}\nExpected 0 active, got {counter_data['active_count']}",
            )

            # total_time_us and max_time_us may be 0 or very small for fast operations
            # Just verify they exist and are non-negative
            self.assertGreaterEqual(counter_data["total_time_us"], 0)
            self.assertGreaterEqual(counter_data["max_time_us"], 0)


if __name__ == "__main__":
    run_tests()
