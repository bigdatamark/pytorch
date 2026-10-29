# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for torch.distributed.debug: the debug HTTP server (endpoints, start
# methods), the PeriodicDumper, fetching from unavailable workers, fetch summaries, fetch
# timeouts, partial-data dumps and the TorchComms health check handler. Import time: builds a
# module-level requests session with retries and sets a flag recording whether aiohttp is
# importable.

import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import torch.distributed as dist
import torch.distributed.debug as debug_module
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_thread_pool,
    format_fetch_summary,
    NavLink,
    PeriodicDumper,
    Response,
    Route,
)
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    import aiohttp  # noqa: F401  # type: ignore[import]

    from torch.distributed.debug._frontend import fetch_aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def _reset_debug_server_state() -> None:
    # [FUNCTION DOC] HELPER: Sets the debug module's worker server and debug server process
    # globals back to None so tests start clean.
    debug_module._WORKER_SERVER = None
    debug_module._DEBUG_SERVER_PROC = None


class TestDebug(TestCase):
    # [CLASS DOC] Integration tests that start a real debug server on fixed local ports using a
    # single-rank TCPStore, and fetch its pages over HTTP. setUp and tearDown reset the
    # module-level server globals; tearDown also stops any server left running.
    def setUp(self) -> None:
        # [FUNCTION DOC] Calls the base setUp, then clears the debug server globals.
        super().setUp()
        _reset_debug_server_state()

    def tearDown(self) -> None:
        # [FUNCTION DOC] Calls the base tearDown; if a debug server process or worker server is
        # still registered, stops it (ignoring errors), then clears the globals.
        super().tearDown()
        if (
            debug_module._DEBUG_SERVER_PROC is not None
            or debug_module._WORKER_SERVER is not None
        ):
            try:
                stop_debug_server()
            except Exception:
                pass
        _reset_debug_server_state()

    def test_all(self) -> None:
        # [FUNCTION DOC] Creates a single-rank TCPStore, sets rank and world-size environment
        # variables (side effect on os.environ), starts the debug server on port 25998 and
        # checks pages in subtests: index, profile, stacks, wait counters, flight recorder
        # traces (nccl variants only with CUDA), a 404 for unknown paths, the tcpstore page
        # (long values truncated at about 95 characters with an ellipsis) and py-spy dumps if
        # py-spy is installed. Finally stops the server.
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25998

        def fetch(path: str) -> str:
            # [FUNCTION DOC] HELPER: Issues an HTTP GET to the local debug server for a path,
            # raises on HTTP error status and returns the response body.
            resp = session.get(f"http://localhost:{port}{path}")
            resp.raise_for_status()
            return resp.text

        start_debug_server(port=port)

        with self.subTest("index"):
            self.assertIn("torch profiler", fetch("/"))

        with self.subTest("profile"):
            self.assertIn("View 0", fetch("/profile?duration=0.01"))

        with self.subTest("stacks"):
            self.assertIn("test_all", fetch("/stacks"))

        with self.subTest("wait_counters"):
            self.assertIn("Rank 0", fetch("/wait_counters"))

        with self.subTest("fr_trace"):
            self.assertIn("Memberships", fetch("/fr_trace"))
            self.assertIn("pg_status", fetch("/fr_trace_json"))

            if torch.cuda.is_available():
                self.assertIn("Memberships", fetch("/fr_trace_nccl"))
                self.assertIn("pg_status", fetch("/fr_trace_nccl_json"))

        with self.subTest("error codes"):
            resp = session.get(f"http://localhost:{port}/blah")
            self.assertEqual(resp.status_code, 404)
            self.assertIn("Handler not found: /blah", resp.text)

        with self.subTest("tcpstore"):
            store.set("test", "value")
            store.set("test2", "a" * 1000)
            out = fetch("/tcpstore")
            self.assertIn("test: b'value'", out)
            self.assertIn("test2: b'" + "a" * 95 + "...", out)

        with self.subTest("pyspy"):
            if shutil.which("py-spy"):
                self.assertIn("test_all", fetch("/pyspy_dump"))
                self.assertIn("_frontend", fetch("/pyspy_dump?subprocesses=1"))
                self.assertIn("libc.so", fetch("/pyspy_dump?native=1"))

        stop_debug_server()

    def test_start_method_spawn(self) -> None:
        # [FUNCTION DOC] Starts the debug server on port 25997 with the spawn start method after
        # setting up a single-rank TCPStore and environment variables, waits one second, asserts
        # that both the debug server process and the worker server are registered, then stops
        # the server.
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25997

        start_debug_server(port=port, start_method="spawn")
        time.sleep(1)

        self.assertIsNotNone(debug_module._DEBUG_SERVER_PROC)
        self.assertIsNotNone(debug_module._WORKER_SERVER)

        stop_debug_server()

    def test_start_method_forkserver(self) -> None:
        # [FUNCTION DOC] Same as the spawn start method test but uses the forkserver start
        # method on port 25996: asserts the server process and worker server exist, then stops
        # the server.
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25996

        start_debug_server(port=port, start_method="forkserver")
        time.sleep(1)

        self.assertIsNotNone(debug_module._DEBUG_SERVER_PROC)
        self.assertIsNotNone(debug_module._WORKER_SERVER)

        stop_debug_server()


class _StubHandler(DebugHandler):
    # [CLASS DOC] HELPER: Minimal DebugHandler for testing PeriodicDumper without network
    # access. It has no routes or nav links, returns a fixed content string (possibly None) from
    # dump, and uses a fixed name as dump filename prefix.
    """Minimal handler for testing PeriodicDumper without network access."""

    def __init__(self, name: str, content: str | None) -> None:
        # [FUNCTION DOC] HELPER: Stores the dump file name and the fixed content to return from
        # dump.
        self._name = name
        self._content = content

    def routes(self) -> list[Route]:
        # [FUNCTION DOC] HELPER: Returns an empty list of routes.
        return []

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] HELPER: Returns an empty list of navigation links.
        return []

    def dump(self) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the stored content, which may be None.
        return self._content

    def dump_filename(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the stored name used as the dump file name.
        return self._name


class _ErrorHandler(DebugHandler):
    # [CLASS DOC] HELPER: DebugHandler stub with no routes or nav links whose dump always raises
    # an error, used to check that PeriodicDumper survives failing handlers. Its dump filename
    # is error_handler.
    """Handler whose dump() always raises."""

    def routes(self) -> list[Route]:
        # [FUNCTION DOC] HELPER: Returns an empty list of routes.
        return []

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] HELPER: Returns an empty list of navigation links.
        return []

    def dump(self) -> str | None:
        # [FUNCTION DOC] HELPER: Always raises a RuntimeError with message boom.
        raise RuntimeError("boom")

    def dump_filename(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the fixed name error_handler.
        return "error_handler"


class TestPeriodicDumper(TestCase):
    # [CLASS DOC] Tests for PeriodicDumper: periodic writing of handler dumps into a directory,
    # skipping of None dumps, handler selection, exception tolerance and idempotent stop.
    def test_writes_dump_files(self) -> None:
        # [FUNCTION DOC] Runs a dumper at a 0.1 second interval for a short time with a stub
        # handler in a temporary directory; asserts files were written, all names start with the
        # handler name prefix, and the file content equals the dump text.
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("mystub", "hello world")
            dumper = PeriodicDumper([h], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.35)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("mystub_") for f in files))
            with open(os.path.join(tmp, files[0])) as f:
                self.assertEqual(f.read(), "hello world")

    def test_skips_none_dump(self) -> None:
        # [FUNCTION DOC] Runs a dumper whose handler returns None from dump and asserts that no
        # files are created in the directory.
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("nodump", None)
            dumper = PeriodicDumper([h], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            self.assertEqual(os.listdir(tmp), [])

    def test_enabled_dumps_filter(self) -> None:
        # [FUNCTION DOC] Filters two handlers by an enabled name set before constructing the
        # dumper (the filtering is done in the test, not in the dumper) and asserts only files
        # of the included handler are written.
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _StubHandler("included", "yes")
            h2 = _StubHandler("excluded", "no")
            enabled = {"included"}
            filtered = [h for h in [h1, h2] if h.dump_filename() in enabled]
            dumper = PeriodicDumper(filtered, tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("included_") for f in files))

    def test_enabled_dumps_none_runs_all(self) -> None:
        # [FUNCTION DOC] Runs a dumper over two handlers and asserts that files are written for
        # both of them.
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _StubHandler("alpha", "a")
            h2 = _StubHandler("beta", "b")
            dumper = PeriodicDumper([h1, h2], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            alpha_files = [f for f in files if f.startswith("alpha_")]
            beta_files = [f for f in files if f.startswith("beta_")]
            self.assertGreater(len(alpha_files), 0)
            self.assertGreater(len(beta_files), 0)

    def test_survives_handler_exception(self) -> None:
        # [FUNCTION DOC] Runs a dumper over a handler that raises and a healthy one; asserts the
        # healthy handler still gets dump files and the failing one writes none.
        with tempfile.TemporaryDirectory() as tmp:
            err = _ErrorHandler()
            ok = _StubHandler("ok", "survived")
            dumper = PeriodicDumper([err, ok], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            ok_files = [f for f in files if f.startswith("ok_")]
            self.assertGreater(len(ok_files), 0)
            err_files = [f for f in files if f.startswith("error_handler_")]
            self.assertEqual(len(err_files), 0)

    def test_stop_is_idempotent(self) -> None:
        # [FUNCTION DOC] Starts a dumper with a long interval and calls stop twice; passes if no
        # error is raised.
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("idem", "data")
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            dumper.stop()
            dumper.stop()


class TestFetchUnavailableWorkers(TestCase):
    # [CLASS DOC] Tests for the fetch helpers (thread pool based and aiohttp based) when workers
    # are unreachable or slow: connection refused yields status 503, timeouts yield 408 (aiohttp
    # may give 503), and mixed good and bad workers return one response per URL.
    @staticmethod
    def _get_refused_port() -> int:
    # [FUNCTION DOC] HELPER: Static method that binds a socket to an ephemeral local port,
    # closes it and returns the port number, so connections to it are refused.
        """Return a port that will refuse connections."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def test_fetch_thread_pool_connection_refused(self) -> None:
        # [FUNCTION DOC] Fetches one URL on a closed port with the thread pool fetcher and
        # asserts one response with status 503 whose text contains ConnectionError.
        port = self._get_refused_port()
        resps = fetch_thread_pool(
            [f"http://localhost:{port}/handler/ping"], timeout=1.0
        )
        self.assertEqual(len(resps), 1)
        self.assertEqual(resps[0].status_code, 503)
        self.assertIn("ConnectionError:", resps[0].text)

    def test_fetch_thread_pool_timeout(self) -> None:
        # [FUNCTION DOC] Fetches from a listening socket that never answers with a 0.5 second
        # timeout and asserts one response with status 408 whose text contains Timeout. Closes
        # the server socket afterwards.
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            resps = fetch_thread_pool(
                [f"http://localhost:{port}/handler/ping"], timeout=0.5
            )
            self.assertEqual(len(resps), 1)
            self.assertEqual(resps[0].status_code, 408)
            self.assertIn("Timeout:", resps[0].text)
        finally:
            server.close()

    @unittest.skipUnless(HAS_AIOHTTP, "aiohttp not installed")
    def test_fetch_aiohttp_connection_refused(self) -> None:
        # [FUNCTION DOC] Skipped without aiohttp. Fetches from a closed port with the aiohttp
        # fetcher and asserts one response with status 503 whose text contains Error.
        port = self._get_refused_port()
        resps = fetch_aiohttp([f"http://localhost:{port}/handler/ping"], timeout=1.0)
        self.assertEqual(len(resps), 1)
        self.assertEqual(resps[0].status_code, 503)
        self.assertIn("Error:", resps[0].text)

    @unittest.skipUnless(HAS_AIOHTTP, "aiohttp not installed")
    def test_fetch_aiohttp_timeout(self) -> None:
        # [FUNCTION DOC] Skipped without aiohttp. Fetches from a listening socket that never
        # answers, with a 0.5 second timeout, and asserts a single response whose status is 408
        # or 503 (aiohttp may wrap timeouts in a client error).
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            resps = fetch_aiohttp(
                [f"http://localhost:{port}/handler/ping"], timeout=0.5
            )
            self.assertEqual(len(resps), 1)
            # aiohttp may report timeout as 408 or wrap it in a ClientError (503)
            self.assertIn(resps[0].status_code, (408, 503))
        finally:
            server.close()

    def test_mixed_available_and_unavailable(self) -> None:
        # [FUNCTION DOC] Starts a local HTTP server on a thread that answers POST with 200 and
        # body pong, then fetches from it and from a closed port with the thread pool fetcher;
        # asserts two responses, the first being 200 pong and the second not 200. Shuts the
        # server down afterwards.
        class _OKHandler(BaseHTTPRequestHandler):
            # [CLASS DOC] HELPER: Request handler answering any POST with status 200 and body
            # pong, with logging disabled.
            def do_POST(self):
                # [FUNCTION DOC] HELPER: Sends a 200 response with body pong.
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"pong")

            def log_message(self, *args):
                # [FUNCTION DOC] HELPER: Suppresses request logging.
                pass

        http_server = HTTPServer(("localhost", 0), _OKHandler)
        good_port = http_server.server_address[1]
        t = threading.Thread(target=http_server.serve_forever, daemon=True)
        t.start()

        bad_port = self._get_refused_port()
        try:
            urls = [
                f"http://localhost:{good_port}/handler/ping",
                f"http://localhost:{bad_port}/handler/ping",
            ]
            resps = fetch_thread_pool(urls, timeout=1.0)
            self.assertEqual(len(resps), 2)
            self.assertEqual(resps[0].status_code, 200)
            self.assertEqual(resps[0].text, "pong")
            self.assertNotEqual(resps[1].status_code, 200)
        finally:
            http_server.shutdown()


class TestFormatFetchSummary(TestCase):
    # [CLASS DOC] Tests for format_fetch_summary, which returns None when all workers responded
    # successfully and otherwise a summary text for failures.
    def test_all_success_returns_none(self) -> None:
        # [FUNCTION DOC] Asserts that the summary is None when all responses have status 200.
        addrs = ["http://host0:1", "http://host1:1"]  # @lint-ignore
        resps = [Response(200, "ok"), Response(200, "ok")]
        self.assertIsNone(format_fetch_summary(addrs, resps))

    def test_partial_failure(self) -> None:
        # [FUNCTION DOC] With one 503 among three responses, asserts the summary is produced and
        # mentions PARTIAL DATA, the 2/3 count of successes and the failing Rank 1.
        addrs = ["http://h0:1", "http://h1:1", "http://h2:1"]  # @lint-ignore
        resps = [
            Response(200, "ok"),
            Response(503, "Worker unavailable"),
            Response(200, "ok"),
        ]
        summary = format_fetch_summary(addrs, resps)
        self.assertIsNotNone(summary)
        self.assertIn("PARTIAL DATA", summary)
        self.assertIn("2/3", summary)
        self.assertIn("Rank 1", summary)

    def test_all_failed(self) -> None:
        # [FUNCTION DOC] With all responses failing (503 and 408), asserts a summary is produced
        # that mentions 0/2.
        addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        resps = [Response(503, "unavailable"), Response(408, "timeout")]
        summary = format_fetch_summary(addrs, resps)
        self.assertIsNotNone(summary)
        self.assertIn("0/2", summary)

    def test_timeout_message_included(self) -> None:
        # [FUNCTION DOC] Asserts that the failure message of a 408 response (timed out text)
        # appears in the summary.
        addrs = ["http://h0:1"]  # @lint-ignore
        resps = [Response(408, "Request timed out after 10.0s")]
        summary = format_fetch_summary(addrs, resps)
        self.assertIn("timed out", summary)


class TestFetchTimeout(TestCase):
    # [CLASS DOC] Tests for the fetch timeout configuration of debug handlers and fetch_all.
    def test_handler_default_fetch_timeout(self) -> None:
        # [FUNCTION DOC] Asserts that a handler's fetch_timeout attribute defaults to the module
        # constant _DEFAULT_FETCH_TIMEOUT.
        from torch.distributed.debug._frontend import _DEFAULT_FETCH_TIMEOUT

        handler = _StubHandler("test", "data")
        self.assertEqual(handler.fetch_timeout, _DEFAULT_FETCH_TIMEOUT)

    def test_handler_fetch_timeout_override(self) -> None:
        # [FUNCTION DOC] Asserts that fetch_timeout can be overridden by assigning on a handler
        # instance.
        handler = _StubHandler("test", "data")
        handler.fetch_timeout = 5.0
        self.assertEqual(handler.fetch_timeout, 5.0)

    def test_fetch_all_requires_timeout(self) -> None:
        # [FUNCTION DOC] Asserts that calling fetch_all without its keyword-only timeout
        # argument raises TypeError.
        from torch.distributed.debug._frontend import fetch_all

        with self.assertRaises(TypeError):
            # fetch_all requires timeout as a keyword-only argument
            fetch_all("test_endpoint")  # type: ignore[call-arg]


class TestHandlerPartialDumps(TestCase):
    # [CLASS DOC] Tests that handlers (StacksHandler dump and PeriodicDumper) report partial
    # data when some workers fail; fetch_all is mocked with patch.
    import torch.distributed.debug._debug_handlers

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    def test_stacks_handler_partial_dump(self, mock_fetch_all) -> None:
        # [FUNCTION DOC] With fetch_all mocked to return one success and one 503, asserts the
        # StacksHandler dump contains PARTIAL DATA, the 1/2 count, the successful stack text and
        # an Error: 503 entry.
        from torch.distributed.debug._debug_handlers import StacksHandler

        mock_fetch_all.return_value = (
            [
                "http://h0:1/handler/dump_traceback?",  # @lint-ignore
                "http://h1:1/handler/dump_traceback?",  # @lint-ignore
            ],
            [Response(200, "stack0"), Response(503, "Worker unavailable")],
        )
        handler = StacksHandler()
        content = handler.dump()
        self.assertIn("PARTIAL DATA", content)
        self.assertIn("1/2", content)
        self.assertIn("stack0", content)
        self.assertIn("Error: 503", content)

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    def test_stacks_handler_all_success(self, mock_fetch_all) -> None:
        # [FUNCTION DOC] With fetch_all mocked to return two successes, asserts the
        # StacksHandler dump has no PARTIAL marker and contains both stack texts.
        from torch.distributed.debug._debug_handlers import StacksHandler

        mock_fetch_all.return_value = (
            [
                "http://h0:1/handler/dump_traceback?",  # @lint-ignore
                "http://h1:1/handler/dump_traceback?",  # @lint-ignore
            ],
            [Response(200, "stack0"), Response(200, "stack1")],
        )
        handler = StacksHandler()
        content = handler.dump()
        self.assertNotIn("PARTIAL", content)
        self.assertIn("stack0", content)
        self.assertIn("stack1", content)

    def test_periodic_dumper_writes_partial_data(self) -> None:
        # [FUNCTION DOC] Has PeriodicDumper write a stub handler's partial-data text to a
        # temporary directory and asserts the written file still contains the PARTIAL DATA
        # header and worker count line.
        with tempfile.TemporaryDirectory() as tmp:
            partial_content = (
                "PARTIAL DATA: 1/2 workers responded\n"
                "  Rank 1: Worker unavailable\n"
                "\n"
                "=== Rank 0 ===\ndata0\n"
                "=== Rank 1 ===\nError: 503"
            )
            h = _StubHandler("partial_stacks", partial_content)
            dumper = PeriodicDumper([h], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            with open(os.path.join(tmp, files[0])) as f:
                content = f.read()
            self.assertIn("PARTIAL DATA", content)
            self.assertIn("1/2 workers responded", content)


class TestTorchCommsHealthCheckHandler(TestCase):
    # [CLASS DOC] Tests for TorchCommsHealthCheckHandler: its route, nav link, template, dump
    # file name, the any-unhealthy check over responses and the dump behavior with mocked
    # fetch_all and format_fetch_summary. setUp creates the handler as self.handler.
    def setUp(self) -> None:
        # [FUNCTION DOC] Calls the base setUp and creates a TorchCommsHealthCheckHandler stored
        # on self.handler.
        super().setUp()
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        self.handler = TorchCommsHealthCheckHandler()

    def test_routes(self) -> None:
        # [FUNCTION DOC] Asserts the handler exposes exactly one route with path
        # torchcomms_health_check.
        routes = self.handler.routes()
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].path, "/torchcomms_health_check")

    def test_nav_links(self) -> None:
        # [FUNCTION DOC] Asserts exactly one nav link with the health check path and the label
        # TorchComms Health.
        links = self.handler.nav_links()
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].path, "/torchcomms_health_check")
        self.assertEqual(links[0].label, "TorchComms Health")

    def test_templates(self) -> None:
        # [FUNCTION DOC] Asserts the templates mapping includes torchcomms_health_check.html and
        # that its content contains Health Status.
        templates = self.handler.templates()
        self.assertIn("torchcomms_health_check.html", templates)
        self.assertIn("Health Status", templates["torchcomms_health_check.html"])

    def test_dump_filename(self) -> None:
        # [FUNCTION DOC] Asserts dump_filename returns torchcomms_health_check.
        self.assertEqual(self.handler.dump_filename(), "torchcomms_health_check")

    # -- _any_unhealthy tests --

    def test_any_unhealthy_all_healthy(self) -> None:
        # [FUNCTION DOC] Asserts _any_unhealthy is false when all responses report healthy true.
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        resps = [
            Response(200, '{"healthy": true}'),
            Response(200, '{"healthy": true}'),
        ]
        self.assertFalse(TorchCommsHealthCheckHandler._any_unhealthy(resps))

    def test_any_unhealthy_one_unhealthy(self) -> None:
        # [FUNCTION DOC] Asserts _any_unhealthy is true when one 200 response reports healthy
        # false.
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        resps = [
            Response(200, '{"healthy": true}'),
            Response(200, '{"healthy": false}'),
        ]
        self.assertTrue(TorchCommsHealthCheckHandler._any_unhealthy(resps))

    def test_any_unhealthy_non_200_ignored(self) -> None:
        # [FUNCTION DOC] Asserts that a non-200 response (unreachable rank) is not counted as
        # unhealthy.
        """Non-200 responses (unreachable ranks) must not trigger a dump.

        Only ranks that are reachable and positively report unhealthy should
        cause an FR dump.  Treating connection failures as unhealthy would
        cause expensive dumps whenever a single rank is temporarily unreachable.
        """
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        resps = [Response(503, "unavailable")]
        self.assertFalse(TorchCommsHealthCheckHandler._any_unhealthy(resps))

    def test_any_unhealthy_bad_json_ignored(self) -> None:
        # [FUNCTION DOC] Asserts that a 200 response with a body that is not valid JSON is
        # ignored (not unhealthy).
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        resps = [Response(200, "not json")]
        self.assertFalse(TorchCommsHealthCheckHandler._any_unhealthy(resps))

    def test_any_unhealthy_missing_key_defaults_healthy(self) -> None:
        # [FUNCTION DOC] Asserts that a response JSON lacking the healthy key counts as healthy.
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        resps = [Response(200, '{"uptime": 42}')]
        self.assertFalse(TorchCommsHealthCheckHandler._any_unhealthy(resps))

    def test_any_unhealthy_empty_list(self) -> None:
        # [FUNCTION DOC] Asserts _any_unhealthy returns false for an empty response list.
        from torch.distributed.debug._debug_handlers import TorchCommsHealthCheckHandler

        self.assertFalse(TorchCommsHealthCheckHandler._any_unhealthy([]))

    # -- dump() tests --

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    @patch("torch.distributed.debug._debug_handlers.format_fetch_summary")
    def test_dump_all_healthy(self, mock_summary, mock_fetch_all) -> None:
        # [FUNCTION DOC] With mocked fetch returning two healthy responses and no summary,
        # asserts dump returns text with Rank 0 and Rank 1 sections, no Unhealthy marker, and
        # that fetch_all was called once.
        addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        resps = [
            Response(200, '{"healthy": true}'),
            Response(200, '{"healthy": true}'),
        ]
        mock_fetch_all.return_value = (addrs, resps)
        mock_summary.return_value = None

        result = self.handler.dump()
        self.assertIsNotNone(result)
        self.assertIn("Rank 0", result)
        self.assertIn("Rank 1", result)
        self.assertNotIn("Unhealthy", result)
        mock_fetch_all.assert_called_once()

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    @patch("torch.distributed.debug._debug_handlers.format_fetch_summary")
    def test_dump_partial_failure(self, mock_summary, mock_fetch_all) -> None:
        # [FUNCTION DOC] With one healthy response and one 503 and a mocked PARTIAL DATA
        # summary, asserts the dump includes the summary and Error: 503 but no Unhealthy marker
        # (no FR dump is triggered).
        addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        resps = [
            Response(200, '{"healthy": true}'),
            Response(503, "Worker unavailable"),
        ]
        mock_fetch_all.return_value = (addrs, resps)
        mock_summary.return_value = "PARTIAL DATA: 1/2 workers responded"

        result = self.handler.dump()
        self.assertIn("PARTIAL DATA", result)
        self.assertIn("Error: 503", result)
        # no unhealthy rank (503 is not parsed as json), so no FR dump
        self.assertNotIn("Unhealthy", result)

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    @patch("torch.distributed.debug._debug_handlers.format_fetch_summary")
    def test_dump_unhealthy_triggers_fr_dump(
        self, mock_summary, mock_fetch_all
    ) -> None:
        # [FUNCTION DOC] With one rank reporting unhealthy, asserts dump performs a second fetch
        # (the second call targets torchcomms_fr_dump_file) and the output contains the message
        # Unhealthy rank detected, triggering FR dump plus both ranks' trace texts.
        health_addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        health_resps = [
            Response(200, '{"healthy": true}'),
            Response(200, '{"healthy": false}'),
        ]
        dump_addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        dump_resps = [
            Response(200, "fr_trace_rank0"),
            Response(200, "fr_trace_rank1"),
        ]
        mock_fetch_all.side_effect = [
            (health_addrs, health_resps),
            (dump_addrs, dump_resps),
        ]
        mock_summary.return_value = None

        result = self.handler.dump()
        self.assertIn("Unhealthy rank detected, triggering FR dump", result)
        self.assertIn("fr_trace_rank0", result)
        self.assertIn("fr_trace_rank1", result)
        self.assertEqual(mock_fetch_all.call_count, 2)
        # second call should be for torchcomms_fr_dump_file
        self.assertEqual(
            mock_fetch_all.call_args_list[1][0][0], "torchcomms_fr_dump_file"
        )

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    @patch("torch.distributed.debug._debug_handlers.format_fetch_summary")
    def test_dump_fr_dump_partial_failure(self, mock_summary, mock_fetch_all) -> None:
        # [FUNCTION DOC] With an unhealthy rank whose follow-up FR dump fetch fails with 503,
        # asserts the dump output contains the Unhealthy rank detected message and an Error: 503
        # entry.
        health_addrs = ["http://h0:1"]  # @lint-ignore
        health_resps = [Response(200, '{"healthy": false}')]
        dump_addrs = ["http://h0:1"]  # @lint-ignore
        dump_resps = [Response(503, "dump failed")]
        mock_fetch_all.side_effect = [
            (health_addrs, health_resps),
            (dump_addrs, dump_resps),
        ]
        mock_summary.return_value = None

        result = self.handler.dump()
        self.assertIn("Unhealthy rank detected", result)
        self.assertIn("Error: 503", result)


if __name__ == "__main__":
    run_tests()
