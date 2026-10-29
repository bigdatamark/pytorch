# [MODULE DOC] Frontend HTTP server for the distributed debug server. Defines response and route
# types, the DebugHandler base class, helpers to fan out POST requests to all rank worker servers
# (aiohttp, falling back to a requests thread pool), HTML/Jinja templates, a periodic dumper, the
# FrontendServer and the process entry point main. Requires jinja2 at import time.
import asyncio
import json
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from jinja2 import DictLoader, Environment

from torch.distributed.debug._store import get_world_size, tcpstore_client


logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_FETCH_TIMEOUT: float = 60.0


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Response:
    # [CLASS DOC] Minimal HTTP response record: status code and body text. Synthetic codes are used
    # for failures (408 timeout, 503 connection error, 502 other error).
    status_code: int
    text: str

    def raise_for_status(self):
        # [FUNCTION DOC] Raises RuntimeError with status and body if status code is not 200.
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        # [FUNCTION DOC] Parses the body text as JSON and returns the result; raises on invalid JSON.
        return json.loads(self.text)


@dataclass(slots=True)
class NavLink:
    # [CLASS DOC] A navigation bar entry: URL path and display label.
    path: str
    label: str


@dataclass(slots=True)
class Route:
    # [CLASS DOC] Maps a URL path to a handler callable that takes the request handler and returns
    # the response body bytes.
    path: str
    handler: Callable[["HTTPRequestHandler"], bytes]


class DebugHandler(ABC):
    # [CLASS DOC] Abstract plugin for the frontend. Subclasses implement routes (list of Route) and
    # nav_links (list of NavLink); they may override templates (name to template text mapping,
    # default empty), dump (text for periodic dumps, default None meaning nothing) and
    # dump_filename (default: lowercase class name). fetch_timeout is set by main.
    fetch_timeout: float = _DEFAULT_FETCH_TIMEOUT

    @abstractmethod
    def routes(self) -> list[Route]: ...

    @abstractmethod
    def nav_links(self) -> list[NavLink]: ...

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns extra Jinja templates by name; default is none.
        return {}

    def dump(self) -> str | None:
        # [FUNCTION DOC] Returns text for periodic dumps, or None to skip; default None.
        return None

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the dump file name prefix, default the lowercased class name.
        return type(self).__name__.lower()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def fetch_thread_pool(urls: list[str], timeout: float) -> list[Response]:
    # [FUNCTION DOC] Sends a POST to every URL using up to 20 threads and the requests library
    # (imported lazily; ImportError propagates if missing). Never raises per-URL: timeouts become
    # status 408, connection errors 503, other errors 502. Returns responses in URL order.
    # late import for optional dependency
    import requests

    max_workers = 20

    def get(url: str) -> Response:
        # [FUNCTION DOC] HELPER: POSTs one URL with the timeout and converts errors to Response.
        try:
            resp = requests.post(url, timeout=timeout)
            return Response(resp.status_code, resp.text)
        except requests.exceptions.Timeout as e:
            return Response(408, f"Timeout: {e}")
        except requests.exceptions.ConnectionError as e:
            return Response(503, f"ConnectionError: {e}")
        except Exception as e:
            return Response(502, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resps = list(executor.map(get, urls))

    return resps


def fetch_aiohttp(urls: list[str], timeout: float) -> list[Response]:
    # [FUNCTION DOC] Sends a POST to every URL concurrently with aiohttp under one total timeout,
    # running its own event loop (so it must not be called from a running loop). aiohttp is
    # imported lazily; ImportError propagates if missing. Failures become status 408, 503 or 502
    # responses. Returns responses in URL order.
    # late import for optional dependency
    # pyrefly: ignore [missing-import]
    import aiohttp

    async def fetch(session: aiohttp.ClientSession, url: str) -> Response:
        # [FUNCTION DOC] HELPER: POSTs one URL on the session and converts errors to Response.
        try:
            async with session.post(url) as resp:
                text = await resp.text()
                return Response(resp.status, text)
        except asyncio.TimeoutError as e:
            return Response(408, f"TimeoutError: {e}")
        except aiohttp.ClientError as e:
            return Response(503, f"{type(e).__name__}: {e}")
        except Exception as e:
            return Response(502, f"{type(e).__name__}: {e}")

    async def gather(urls: list[str]) -> list[Response]:
        # [FUNCTION DOC] HELPER: Opens a client session with the total timeout and gathers all fetches.
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            return list(await asyncio.gather(*[fetch(session, url) for url in urls]))

    return asyncio.run(gather(urls))


def fetch_all(
    endpoint: str, args: str = "", *, timeout: float
) -> tuple[list[str], list[Response]]:
    # [FUNCTION DOC] Looks up each rank's worker server address in the TCP store (one key per rank
    # up to world size), builds the handler URL with endpoint and query args, and POSTs to all of
    # them with aiohttp, falling back to the thread pool if aiohttp is missing. Returns the URLs
    # and the responses, both in rank order.
    store = tcpstore_client()
    keys = [f"rank{r}" for r in range(get_world_size())]
    addrs = store.multi_get(keys)
    addrs = [f"{addr.decode()}/handler/{endpoint}?{args}" for addr in addrs]

    try:
        resps = fetch_aiohttp(addrs, timeout=timeout)
    except ImportError:
        resps = fetch_thread_pool(addrs, timeout=timeout)

    return addrs, resps


def format_json(blob: str):
    # [FUNCTION DOC] Re-serializes a JSON string with two-space indentation; raises on invalid JSON.
    parsed = json.loads(blob)
    return json.dumps(parsed, indent=2)


def format_fetch_summary(addrs: list[str], resps: list[Response]) -> str | None:
    # [FUNCTION DOC] Returns a multi-line partial-data summary listing each rank with a non-200
    # response and its text, or None when every response succeeded.
    """Return a summary string if any workers failed, or None if all succeeded."""
    failed = [(i, r) for i, r in enumerate(resps) if r.status_code != 200]
    if not failed:
        return None
    total = len(addrs)
    ok = total - len(failed)
    lines = [f"PARTIAL DATA: {ok}/{total} workers responded"]
    for rank, resp in failed:
        lines.append(f"  Rank {rank}: {resp.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template constants
# ---------------------------------------------------------------------------


BASE_TEMPLATE = """
<!doctype html>
<head>
    <title>{% block title %}{% endblock %} - PyTorch Distributed</title>
    <link rel="shortcut icon" type="image/x-icon" href="https://pytorch.org/favicon.ico?">

    <style>
        body {
            margin: 0;
            font-family:
                -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                "Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji",
                "Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.5;
            color: #212529;
            text-align: left;
            background-color: #fff;
        }
        h1, h2, h2, h4, h5, h6, .h1, .h2, .h2, .h4, .h5, .h6 {
            margin-bottom: .5rem;
            font-weight: 500;
            line-height: 1.2;
        }
        nav {
            background-color: rgba(0, 0, 0, 0.17);
            padding: 10px;
            display: flex;
            align-items: center;
            padding: 16px;
            justify-content: flex-start;
        }
        nav h1 {
            display: inline-block;
            margin: 0;
        }
        nav a {
           margin: 0 8px;
        }
        section {
            max-width: 1280px;
            padding: 16px;
            margin: 0 auto;
        }
        pre {
            white-space: pre-wrap;
            max-width: 100%;
        }
    </style>
</head>

<nav>
    <h1>Torch Distributed Debug Server</h1>

    {{ nav_links | safe }}
</nav>

<section class="content">
  {% block header %}{% endblock %}
  {% block content %}{% endblock %}
</section>
    """

RAW_RESP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{title}}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """

JSON_RESP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ format_json(resp.text) }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """


# ---------------------------------------------------------------------------
# PeriodicDumper
# ---------------------------------------------------------------------------


class PeriodicDumper:
    # [CLASS DOC] Background thread that periodically calls dump on each handler and writes
    # non-None results to timestamped text files in an output directory. State: handlers, output
    # directory, interval, a stop event and the thread. Errors are logged, never raised.
    def __init__(
        self,
        handlers: list[DebugHandler],
        output_dir: str,
        interval_seconds: float = 60.0,
    ) -> None:
        # [FUNCTION DOC] Stores the configuration and creates the stop event; does not start a thread.
        self._handlers = handlers
        self._output_dir = output_dir
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # [FUNCTION DOC] Creates the output directory if needed and starts the daemon dump thread.
        os.makedirs(self._output_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="distributed.debug.PeriodicDumper",
        )
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC] Signals the thread to stop and waits for it to finish, if it was started.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread loop. Each round, for every handler it gets the dump text
        # (logging and skipping on error or None) and writes it to a file named by the handler dump
        # filename and a timestamp; write errors are logged. Then waits the interval or until stopped.
        while not self._stop_event.is_set():
            for handler in self._handlers:
                try:
                    content = handler.dump()
                except Exception:
                    logger.exception("Failed to dump %s", handler.dump_filename())
                    continue
                if content is None:
                    continue
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{handler.dump_filename()}_{timestamp}.txt"
                path = os.path.join(self._output_dir, filename)
                try:
                    with open(path, "w") as f:
                        f.write(content)
                except Exception:
                    logger.exception("Failed to write dump to %s", path)
            self._stop_event.wait(self._interval_seconds)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _IPv6HTTPServer(ThreadingHTTPServer):
    # [CLASS DOC] HELPER: Threaded HTTP server bound over IPv6 with a request queue of 1024.
    address_family: socket.AddressFamily = socket.AF_INET6  # pyre-ignore
    request_queue_size: int = 1024


class HTTPRequestHandler(BaseHTTPRequestHandler):
    # [CLASS DOC] Request handler that forwards GET requests to its frontend server (the class
    # attribute frontend, set on a dynamically created subclass) and offers query parsing helpers.
    frontend: "FrontendServer"

    def log_message(self, format, *args):
        # [FUNCTION DOC] Routes the standard access log line to the module logger at info level.
        logger.info(
            "%s %s",
            self.client_address[0],
            format % args,
        )

    def do_GET(self):
        # [FUNCTION DOC] Delegates the request to the frontend server request dispatcher.
        self.frontend._handle_request(self)

    def get_path(self) -> str:
        # [FUNCTION DOC] Returns the URL path without the query string.
        return urlparse(self.path).path

    def get_query(self) -> dict[str, list[str]]:
        # [FUNCTION DOC] Returns the parsed query string as a mapping from name to list of values.
        return parse_qs(self.get_raw_query())

    def get_raw_query(self) -> str:
        # [FUNCTION DOC] Returns the raw query string of the request URL.
        return urlparse(self.path).query

    def get_query_arg(
        self, name: str, default: object = None, type: type = str
    ) -> object:
        # [FUNCTION DOC] Returns the first value of the named query parameter converted by the
        # given type callable (default str), or the default if absent. Conversion errors propagate.
        query = self.get_query()
        if name not in query:
            return default
        return type(query[name][0])


class FrontendServer:
    # [CLASS DOC] HTTP frontend serving the debug pages. Built from handlers: collects their nav
    # links, templates (merged with the shared base/raw/json templates into a Jinja environment)
    # and routes, then starts an IPv6 threaded HTTP server on a daemon thread. Exposes join and
    # render_template; unknown paths return 404 and handler failures 500.
    def __init__(
        self,
        port: int,
        handlers: list[DebugHandler] | None = None,
    ):
        # [FUNCTION DOC] Builds the template environment and route table from the handlers (default
        # handlers if None), creates the server on the given port and starts the serving thread.
        # Side effects: binds a socket and starts a thread. Later routes override earlier same paths.
        if handlers is None:
            from torch.distributed.debug._debug_handlers import default_handlers

            handlers = default_handlers()

        # Build nav HTML from handlers
        nav_html = "\n".join(
            f'    <a href="{link.path}">{link.label}</a> <!--@lint-ignore-->'
            for handler in handlers
            for link in handler.nav_links()
        )

        # Merge all handler templates + shared templates
        all_templates: dict[str, str] = {
            "base.html": BASE_TEMPLATE,
            "raw_resp.html": RAW_RESP_TEMPLATE,
            "json_resp.html": JSON_RESP_TEMPLATE,
        }
        for handler in handlers:
            all_templates.update(handler.templates())

        loader = DictLoader(all_templates)
        self._jinja_env = Environment(loader=loader, enable_async=True)
        self._jinja_env.globals.update(
            zip=zip,
            format_json=format_json,
            enumerate=enumerate,
            nav_links=nav_html,
        )

        # Build route table from handlers
        self._routes: dict[str, Callable[[HTTPRequestHandler], bytes]] = {}
        for handler in handlers:
            for route in handler.routes():
                self._routes[route.path] = route.handler

        self._handlers = handlers

        # Create HTTP server
        RequestHandlerClass = type(
            "HTTPRequestHandler",
            (HTTPRequestHandler,),
            {"frontend": self},
        )

        server_address = ("", port)
        self._server = _IPv6HTTPServer(server_address, RequestHandlerClass)

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
            name="distributed.debug.FrontendServer",
        )
        self._thread.start()

    def _serve(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body that serves requests forever, logging any exception.
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("got exception in frontend server")

    def join(self) -> None:
        # [FUNCTION DOC] Blocks until the serving thread ends.
        self._thread.join()

    def _handle_request(self, req: HTTPRequestHandler) -> None:
        # [FUNCTION DOC] HELPER: Dispatches a request by path. Unknown path gets a 404 error; a
        # handler exception (including SystemExit) is logged and answered with 500; otherwise sends
        # status 200 with an HTML content type and the handler bytes as body.
        path = req.get_path()
        if path not in self._routes:
            req.send_error(404, f"Handler not found: {path}")
            return

        handler = self._routes[path]
        try:
            resp = handler(req)
        # Catch SystemExit to not crash when FlightRecorder errors.
        except (Exception, SystemExit) as e:
            logger.exception(
                "Exception in frontend server when handling %s",
                path,
            )
            req.send_error(500, f"Exception: {repr(e)}")
            return

        req.send_response(200)
        req.send_header("Content-type", "text/html")
        req.end_headers()
        req.wfile.write(resp)

    def render_template(self, template: str, **kwargs: object) -> bytes:
        # [FUNCTION DOC] Renders the named Jinja template with the keyword arguments and returns
        # the result encoded as bytes. Raises if the template is unknown.
        return self._jinja_env.get_template(template).render(**kwargs).encode()


def main(
    port: int,
    dump_dir: str | None,
    dump_interval: float,
    handlers: list[DebugHandler],
    enabled_dumps: set[str],
    fetch_timeout: float = 60.0,
) -> None:
    # [FUNCTION DOC] Frontend process entry point. Sets the fetch timeout on all handlers, sets the
    # module logger to info, starts the FrontendServer on the port and, if dump_dir is given, a
    # PeriodicDumper for handlers whose dump filename is in enabled_dumps. Blocks until the server
    # thread ends, then stops the dumper.
    for handler in handlers:
        handler.fetch_timeout = fetch_timeout

    logger.setLevel(logging.INFO)

    server = FrontendServer(port=port, handlers=handlers)
    logger.info("Frontend server started on port %d", server._server.server_port)

    dumper: PeriodicDumper | None = None
    if dump_dir is not None:
        dumper = PeriodicDumper(
            [
                handler
                for handler in handlers
                if handler.dump_filename() in enabled_dumps
            ],
            dump_dir,
            dump_interval,
        )
        dumper.start()
        logger.info(
            "Periodic dumper started, writing to %s every %.0fs",
            dump_dir,
            dump_interval,
        )

    try:
        server.join()
    finally:
        if dumper is not None:
            dumper.stop()
